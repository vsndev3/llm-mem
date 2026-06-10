use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use std::time::Instant;

use crate::{
    config::MemoryConfig,
    error::{MemoryError, Result},
    layer::pending_wal::{PendingRelationEntry, PendingWal},
    llm::{LLMClient, LlmPriority, PriorityLLMClient},
    memory::{
        cache_service::CacheService,
        extractor::{FactExtractor, create_fact_extractor},
        importance::{ImportanceEvaluator, create_importance_evaluator},
        metrics::{IngestionPhase, MetricsSink, NoopMetrics},
        search_service::SearchService,
        updater::{MemoryAction, MemoryUpdater, create_memory_updater},
    },
    types::{
        reverse_relation, ContentMeta, Filters, LayerInfo, Memory, MemoryEvent, MemoryMetadata,
        MemoryResult, Message, Relation, RelationMeta,
    },
    vector_store::VectorStore,
};

pub struct IngestOptions {
    pub content: String,
    pub content_encoding: Option<String>,
    pub format_hint: Option<String>,
    pub file_name: Option<String>,
    pub auto_link: Option<bool>,
    pub generate_abstractions: Option<bool>,
    pub max_chunk_size: Option<usize>,
    pub user_metadata: Option<MemoryMetadata>,
    /// Optional explicit source override. When set, takes precedence over
    /// the auto-derived "<filename> — <title>" string and is propagated to
    /// every L0 chunk's `content_meta.source`.
    pub source: Option<String>,
    /// Whether to generate AI-powered image descriptions for ingested images.
    pub describe_images: Option<bool>,
}

/// Options for storing memory
#[derive(Debug, Clone)]
pub struct StoreOptions {
    pub deduplicate: Option<bool>,
    pub enhance: Option<bool>,
    pub llm_priority: LlmPriority,
    /// Whether to auto-link to semantically similar existing memories.
    /// None = use config default (auto_link_threshold > 0.0)
    pub auto_link: Option<bool>,
    /// Caller-supplied event time (when the event actually happened).
    /// Only meaningful for L0 raw content; the ingestion service sets it on
    /// the resulting Memory. None means fall back to created_at at query time.
    pub event_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Free-form source description (e.g., file name, URL, book title).
    /// When set, it populates the resulting Memory's `content_meta.source`
    /// so callers can later answer "where did this fact come from?".
    pub source: Option<String>,
    /// Raw image data (base64-encoded) to set on `content_meta.image_data`.
    /// Only meaningful for image L0 chunks. Callers pass the original base64
    /// content so the raw user input is preserved for display/retrieval.
    pub image_data: Option<String>,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self {
            deduplicate: None,
            enhance: None,
            llm_priority: LlmPriority::Background,
            auto_link: None,
            event_at: None,
            source: None,
            image_data: None,
        }
    }
}

/// Result of LLM relation validation.
struct RelationValidation {
    valid: bool,
    confidence: f32,
    suggested_relation: Option<String>,
}

/// Owns memory ingestion: store, add_memory,
/// content hashing, enhancement, deduplication, and classification.
///
/// Extracted from MemoryManager to reduce its god-object responsibilities.
pub struct IngestionService {
    vector_store: Box<dyn VectorStore + Send + Sync>,
    llm: Arc<PriorityLLMClient>,
    config: Arc<MemoryConfig>,
    cache: Arc<CacheService>,
    search: Arc<SearchService>,
    fact_extractor: Box<dyn FactExtractor + 'static>,
    memory_updater: Box<dyn MemoryUpdater + 'static>,
    importance_evaluator: Box<dyn ImportanceEvaluator + 'static>,
    pending_wal: std::sync::Mutex<Option<Arc<PendingWal>>>,
    metrics: Arc<dyn MetricsSink>,
}

impl IngestionService {
    pub fn new(
        vector_store: Box<dyn VectorStore + Send + Sync>,
        llm: Arc<PriorityLLMClient>,
        downstream_llm: Box<dyn LLMClient + Send + Sync>,
        config: Arc<MemoryConfig>,
        cache: Arc<CacheService>,
        search: Arc<SearchService>,
        metrics: Option<Arc<dyn MetricsSink>>,
    ) -> Self {
        let fact_extractor = create_fact_extractor(dyn_clone::clone_box(downstream_llm.as_ref()));
        let memory_updater = create_memory_updater(
            dyn_clone::clone_box(downstream_llm.as_ref()),
            dyn_clone::clone_box(vector_store.as_ref()),
            config.similarity_threshold,
            config.merge_threshold,
        );
        let importance_evaluator = create_importance_evaluator(
            dyn_clone::clone_box(downstream_llm.as_ref()),
            config.auto_metadata_analysis && config.llm_importance_scoring,
            Some(0.5),
        );

        Self {
            vector_store,
            llm,
            config,
            cache,
            search,
            fact_extractor,
            memory_updater,
            importance_evaluator,
            pending_wal: std::sync::Mutex::new(None),
            metrics: metrics.unwrap_or_else(|| Arc::new(NoopMetrics)),
        }
    }

    /// Attach the pending WAL for relation persistence across restarts.
    pub fn set_pending_wal(&self, wal: Arc<PendingWal>) {
        if let Ok(mut guard) = self.pending_wal.lock() {
            *guard = Some(wal);
        }
    }

    #[allow(dead_code)]
    pub fn metrics(&self) -> &Arc<dyn MetricsSink> {
        &self.metrics
    }

    pub fn llm_client(&self) -> &dyn LLMClient {
        self.llm.inner()
    }

    /// Generate a hash for memory content
    pub fn generate_hash(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Extract metadata enrichment for a text chunk
    pub async fn extract_metadata_enrichment(
        &self,
        text: &str,
    ) -> Result<crate::memory::extractor::ChunkMetadata> {
        let results = self
            .fact_extractor
            .extract_metadata_enrichment(&[text.to_string()])
            .await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| MemoryError::LLM("No metadata enrichment returned".to_string()))
    }

    /// Extract metadata enrichment for multiple text chunks in batch
    pub async fn extract_metadata_enrichment_batch(
        &self,
        texts: &[String],
    ) -> Result<Vec<crate::memory::extractor::ChunkMetadata>> {
        self.fact_extractor.extract_metadata_enrichment(texts).await
    }

    /// Import a fully-formed Memory directly into the vector store.
    pub async fn import_memory(&self, memory: &Memory) -> Result<()> {
        self.vector_store.insert(memory).await?;
        let level = memory.metadata.layer.level;
        self.search.insert_layer(level).await;
        Ok(())
    }

    /// Check if memory with the same content already exists.
    async fn check_duplicate(
        &self,
        content: &str,
        filters: &Filters,
        llm_priority: LlmPriority,
    ) -> Result<Option<Memory>> {
        let hash = Self::generate_hash(content);
        let start = Instant::now();
        let query_embedding = self.cache.cached_embed(content, llm_priority).await?;
        self.metrics
            .record_ingestion_timing(IngestionPhase::DedupEmbed, start.elapsed());

        let start = Instant::now();
        let candidates = self
            .vector_store
            .search_with_threshold(&query_embedding, filters, 5, Some(0.5))
            .await?;
        self.metrics
            .record_ingestion_timing(IngestionPhase::DedupSearch, start.elapsed());

        let mut best_near_dup: Option<(String, f32)> = None;

        for scored in candidates {
            let memory = scored.memory;
            if memory.metadata.hash == hash {
                if memory.content.as_ref().is_none_or(|c| c.trim().is_empty()) {
                    tracing::warn!(
                        "Found duplicate memory {} with empty content, skipping",
                        memory.id
                    );
                    continue;
                }
                tracing::debug!("Found duplicate memory with ID: {}", memory.id);
                return Ok(Some(memory));
            }
            if self.config.near_duplicate_threshold > 0.0
                && scored.score >= self.config.near_duplicate_threshold
            {
                let current = (memory.id, scored.score);
                best_near_dup = Some(match &best_near_dup {
                    Some(prev) if prev.1 >= current.1 => prev.clone(),
                    _ => current,
                });
            }
        }

        if let Some((near_id, score)) = best_near_dup {
            tracing::warn!(
                "Near-duplicate detected: new content is {:.2}% similar to existing memory {} (threshold: {:.2})",
                score * 100.0,
                near_id,
                self.config.near_duplicate_threshold
            );
        }

        Ok(None)
    }

    /// Enhance memory content with LLM-generated metadata
    async fn enhance_memory(
        &self,
        memory: &mut Memory,
        llm_priority: LlmPriority,
    ) -> Result<()> {
        let content = match &memory.content {
            Some(c) => c,
            None => return Ok(()),
        };

        let mut prompt =
            crate::memory::prompts::UNIFIED_MEMORY_ENHANCEMENT_PROMPT.replace("{{text}}", content);

        // Retrieval-augmented: include top-3 similar existing memories as context
        if self.config.auto_link_threshold > 0.0 {
            let filters = Filters::for_user_scope(
                memory.metadata.user_id.clone(),
                memory.metadata.agent_id.clone(),
                memory.metadata.run_id.clone(),
                memory.metadata.actor_id.clone(),
            );
            let embedding = self.cache.cached_embed(content, llm_priority).await?;
            if let Ok(candidates) = self
                .vector_store
                .search_with_threshold(&embedding, &filters, 3, Some(0.5))
                .await
            {
                let related: Vec<String> = candidates
                    .into_iter()
                    .filter(|s| s.memory.id != memory.id)
                    .map(|s| {
                        format!(
                            "- {}: {}",
                            s.memory.id,
                            s.memory.content.as_deref().unwrap_or("")
                        )
                    })
                    .collect();
                if !related.is_empty() {
                    prompt = format!(
                        "Related existing knowledge (use for cross-referencing):\n{}\n\n{}",
                        related.join("\n"),
                        prompt
                    );
                }
            }
        }

        let start = Instant::now();
        let res = {
            let _guard = self.llm.acquire(llm_priority).await;
            self.llm.inner().enhance_memory_unified(&prompt).await
        };
        self.metrics
            .record_ingestion_timing(IngestionPhase::MemoryEnhance, start.elapsed());
        match res {
            Ok(enhancement) => {
                if !enhancement.keywords.is_empty()
                    && !memory.metadata.custom.contains_key("keywords")
                {
                    memory.metadata.custom.insert(
                        "keywords".to_string(),
                        serde_json::Value::Array(
                            enhancement
                                .keywords
                                .into_iter()
                                .map(serde_json::Value::String)
                                .collect(),
                        ),
                    );
                }
                if !enhancement.summary.is_empty()
                    && content.len() > self.config.auto_summary_threshold
                    && !memory.metadata.custom.contains_key("summary")
                {
                    memory.metadata.custom.insert(
                        "summary".to_string(),
                        serde_json::Value::String(enhancement.summary),
                    );
                }
                if !enhancement.entities.is_empty() {
                    if memory.metadata.entities.is_empty() {
                        memory.metadata.entities = enhancement.entities;
                    } else {
                        for entity in enhancement.entities {
                            if !memory.metadata.entities.contains(&entity) {
                                memory.metadata.entities.push(entity);
                            }
                        }
                    }
                }
                if !enhancement.topics.is_empty() {
                    if memory.metadata.topics.is_empty() {
                        memory.metadata.topics = enhancement.topics;
                    } else {
                        for topic in enhancement.topics {
                            if !memory.metadata.topics.contains(&topic) {
                                memory.metadata.topics.push(topic);
                            }
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!(
                    "Unified memory enhancement failed, skipping enhancement: {}",
                    e
                );
            }
        };

        let start = Instant::now();
        if let Ok(importance) = self.importance_evaluator.evaluate_importance(memory).await {
            memory.metadata.importance_score = memory.metadata.importance_score.max(importance);
        }
        self.metrics
            .record_ingestion_timing(IngestionPhase::ImportanceScore, start.elapsed());

        Ok(())
    }

    /// Create a new memory from content and metadata with options
    pub async fn create_memory_with_options(
        &self,
        content: String,
        metadata: MemoryMetadata,
        options: &StoreOptions,
    ) -> Result<Memory> {
        if content.trim().is_empty() {
            return Err(MemoryError::Validation(
                "Content cannot be empty when creating memory".to_string(),
            ));
        }

        let start = Instant::now();
        let embedding = {
            let _guard = self.llm.acquire(options.llm_priority).await;
            self.llm.inner().embed(&content).await?
        };
        self.metrics
            .record_ingestion_timing(IngestionPhase::ContentEmbed, start.elapsed());
        let hash = Self::generate_hash(&content);

        let mut memory =
            Memory::with_content(content, embedding, MemoryMetadata { hash, ..metadata });

        if let Some(source) = &options.source {
            memory.content_meta = memory.content_meta.clone().with_source(source.clone());
        }

        if let Some(image_data) = &options.image_data {
            memory.content_meta = memory
                .content_meta
                .clone()
                .with_image_data(image_data.clone());
        }

        let enhance = options.enhance.unwrap_or(self.config.auto_metadata_analysis);
        if enhance {
            self.enhance_memory(&mut memory, options.llm_priority)
                .await?;
        }

        Ok(memory)
    }

    /// Create a new memory from content and metadata
    pub async fn create_memory(&self, content: String, metadata: MemoryMetadata) -> Result<Memory> {
        self.create_memory_with_options(content, metadata, &StoreOptions::default())
            .await
    }

    /// Store a memory in the vector store
    pub async fn store(&self, content: String, metadata: MemoryMetadata) -> Result<String> {
        self.store_with_options(content, metadata, StoreOptions::default())
            .await
    }

    /// Store a memory with Interactive LLM priority (for user-facing store operations).
    /// This ensures the store doesn't get starved by background abstraction pipeline work.
    pub async fn store_interactive(
        &self,
        content: String,
        metadata: MemoryMetadata,
    ) -> Result<String> {
        let options = StoreOptions {
            llm_priority: LlmPriority::Interactive,
            ..StoreOptions::default()
        };
        self.store_with_options(content, metadata, options).await
    }

    /// Store a memory with fine-grained control options
    pub async fn store_with_options(
        &self,
        content: String,
        metadata: MemoryMetadata,
        options: StoreOptions,
    ) -> Result<String> {
        if content.trim().is_empty() {
            return Err(MemoryError::Validation(
                "Content cannot be empty".to_string(),
            ));
        }
        if content.len() > self.config.max_content_length {
            return Err(MemoryError::Validation(format!(
                "Content length ({} bytes) exceeds maximum allowed ({} bytes)",
                content.len(),
                self.config.max_content_length,
            )));
        }

        let start = Instant::now();
        let current_count = self.vector_store.count().await?;
        self.metrics
            .record_ingestion_timing(IngestionPhase::MemoryCountCheck, start.elapsed());
        if current_count >= self.config.max_memories {
            return Err(MemoryError::Validation(format!(
                "Memory store is full ({}/{} memories). Delete old memories or increase max_memories in config.",
                current_count, self.config.max_memories,
            )));
        }

        let deduplicate = options.deduplicate.unwrap_or(self.config.skip_duplicates);
        let user_filters = Filters::for_user_scope(
            metadata.user_id.clone(),
            metadata.agent_id.clone(),
            metadata.run_id.clone(),
            metadata.actor_id.clone(),
        );
        if deduplicate
            && let Some(existing) = self
                .check_duplicate(&content, &user_filters, options.llm_priority)
                .await?
        {
            if existing
                .content
                .as_ref()
                .is_none_or(|c| c.trim().is_empty())
            {
                tracing::warn!(
                    "Existing memory {} has empty content, creating new memory instead",
                    existing.id
                );
            } else {
                tracing::info!(
                    "Duplicate memory found, returning existing ID: {}",
                    existing.id
                );
                return Ok(existing.id);
            }
        }

        let mut memory = self
            .create_memory_with_options(content, metadata, &options)
            .await?;
        // Set caller-supplied event time on the resulting memory. None leaves the
        // column NULL and readers fall back to created_at (backfill semantics).
        if let Some(event_at) = options.event_at {
            memory.event_at = Some(event_at);
        }
        let memory_id = memory.id.clone();

        for relation in &mut memory.metadata.relations {
            if relation.source == "SELF" {
                relation.source = memory_id.clone();
            }
        }

        // Multi-Vector Embedding
        let ctx_tags = &memory.metadata.context;
        let rel_texts: Vec<String> = memory
            .metadata
            .relations
            .iter()
            .map(|r| format!("{} {}", r.relation, r.target))
            .collect();

        let total_aux = ctx_tags.len() + rel_texts.len();
        if total_aux > 0 {
            let mut all_texts: Vec<String> = Vec::with_capacity(total_aux);
            all_texts.extend(ctx_tags.iter().cloned());
            all_texts.extend(rel_texts.iter().cloned());

            let start = Instant::now();
            let all_embeddings = {
                let _guard = self.llm.acquire(options.llm_priority).await;
                self.llm.inner().embed_batch(&all_texts).await?
            };
            self.metrics
                .record_ingestion_timing(IngestionPhase::AuxEmbed, start.elapsed());

            if all_embeddings.len() == total_aux {
                if !ctx_tags.is_empty() {
                    memory.context_embeddings = Some(all_embeddings[..ctx_tags.len()].to_vec());
                }
                if !rel_texts.is_empty() {
                    memory.relation_embeddings = Some(all_embeddings[ctx_tags.len()..].to_vec());
                }
            } else {
                tracing::warn!(
                    "embed_batch returned {} embeddings, expected {}; skipping auxiliary embeddings",
                    all_embeddings.len(),
                    total_aux
                );
            }
        }

        // Auto-link to semantically similar existing memories
        let do_auto_link = options
            .auto_link
            .unwrap_or(self.config.auto_link_threshold > 0.0);
        if do_auto_link {
            let start = Instant::now();
            let weights = (
                self.config.auto_link_primary_pct,
                self.config.auto_link_context_pct,
                self.config.auto_link_relation_pct,
            );
            let linked = self
                .auto_link_memory(
                    &mut memory,
                    self.config.auto_link_threshold,
                    self.config.auto_link_max_relations,
                    weights,
                )
                .await
                .unwrap_or(0);
            self.metrics
                .record_ingestion_timing(IngestionPhase::AutoLinkSearch, start.elapsed());
            if linked > 0 {
                tracing::info!(
                    "Auto-linked memory {} to {} similar memories",
                    memory_id,
                    linked
                );
            }
        }

        // Wire caller-supplied explicit relations into the graph
        let _ = self
            .wire_explicit_relations(&mut memory, "default")
            .await;

        let start = Instant::now();
        self.vector_store.insert(&memory).await?;
        self.metrics
            .record_ingestion_timing(IngestionPhase::VsInsert, start.elapsed());

        // Resolve any pending relations that target this new memory
        let _ = self
            .resolve_pending_relations_for(&memory, "default")
            .await;

        let start = Instant::now();
        self.search.insert_layer(memory.metadata.layer.level).await;
        self.metrics
            .record_ingestion_timing(IngestionPhase::LayerManifestUpdate, start.elapsed());

        // Chunk long L0 memories for better retrieval coverage.
        // The embedding model truncates at ~256 tokens, so long sessions
        // lose most of their content. Chunking gives each segment its own vector.
        self.store_content_chunks(&memory, options.llm_priority)
            .await?;

        tracing::info!(
            "Stored new memory with ID: {} (content length: {}, contexts: {}, relations: {})",
            memory_id,
            memory.content.as_ref().map_or(0, |c| c.len()),
            memory.metadata.context.len(),
            memory.metadata.relations.len(),
        );

        if self.config.contradiction_detection {
            self.check_contradictions(&memory, &user_filters, options.llm_priority)
                .await;
        }

        Ok(memory_id)
    }
}

/// Quality check result for pre-store validation.
#[derive(Debug, Clone, Default)]
pub struct StoreQualityWarnings {
    pub near_duplicates: Vec<(String, f32)>,
    pub contradictions: Vec<String>,
}

impl IngestionService {
    /// Check content quality against existing store before ingesting.
    /// Returns near-duplicate IDs with similarity scores and any contradiction explanations.
    /// The caller can present these to the user/LLM so they can decide to rewrite.
    pub async fn check_store_quality(
        &self,
        content: &str,
        metadata: &MemoryMetadata,
    ) -> crate::error::Result<StoreQualityWarnings> {
        let mut warnings = StoreQualityWarnings::default();
        let filters = Filters::for_user_scope(
            metadata.user_id.clone(),
            metadata.agent_id.clone(),
            metadata.run_id.clone(),
            metadata.actor_id.clone(),
        );

        if self.config.near_duplicate_threshold > 0.0 && !content.trim().is_empty() {
            let embedding = self
                .cache
                .cached_embed(content, LlmPriority::Interactive)
                .await?;
            if let Ok(candidates) = self
                .vector_store
                .search_with_threshold(
                    &embedding,
                    &filters,
                    5,
                    Some(self.config.near_duplicate_threshold),
                )
                .await
            {
                for scored in candidates {
                    if scored.memory.metadata.hash != Self::generate_hash(content) {
                        warnings
                            .near_duplicates
                            .push((scored.memory.id, scored.score));
                    }
                }
            }
        }

        if self.config.contradiction_detection && !content.trim().is_empty() {
            let embedding = self
                .cache
                .cached_embed(content, LlmPriority::Interactive)
                .await?;
            if let Ok(candidates) = self
                .vector_store
                .search_with_threshold(&embedding, &filters, 3, Some(0.6))
                .await
            {
                for scored in candidates {
                    let existing_content = match &scored.memory.content {
                        Some(c) if !c.trim().is_empty() => c,
                        _ => continue,
                    };
                    let prompt = format!(
                        "Compare these two statements and determine if they contradict.\n\n\
                         Statement A (new): {}\n\n\
                         Statement B (existing, ID: {}): {}\n\n\
                         Respond with JSON: {{\"contradiction\": true|false, \"explanation\": \"brief reason\"}}",
                        content, scored.memory.id, existing_content
                    );
                    let _guard = self.llm.acquire(LlmPriority::Interactive).await;
                    if let Ok(response) = self.llm.inner().complete(&prompt).await
                        && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response)
                        && parsed
                            .get("contradiction")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                    {
                        let explanation = parsed
                            .get("explanation")
                            .and_then(|v| v.as_str())
                            .unwrap_or("no explanation");
                        warnings.contradictions.push(format!(
                            "vs {} (score {:.2}): {}",
                            scored.memory.id, scored.score, explanation
                        ));
                    }
                }
            }
        }

        Ok(warnings)
    }

    /// Search for existing memories that may contradict the new memory.
    /// Uses LLM to compare the new fact against top-3 similar existing facts.
    async fn check_contradictions(
        &self,
        memory: &Memory,
        filters: &Filters,
        llm_priority: LlmPriority,
    ) {
        let content = match &memory.content {
            Some(c) if !c.trim().is_empty() => c,
            _ => return,
        };
        let embedding = {
            let _guard = self.llm.acquire(llm_priority).await;
            match self.llm.inner().embed(content).await {
                Ok(e) => e,
                Err(_) => return,
            }
        };
        let candidates = match self
            .vector_store
            .search_with_threshold(&embedding, filters, 3, Some(0.6))
            .await
        {
            Ok(c) => c,
            Err(_) => return,
        };
        for scored in candidates {
            let existing = &scored.memory;
            let existing_content = match &existing.content {
                Some(c) if !c.trim().is_empty() => c,
                _ => continue,
            };
            let prompt = format!(
                "Compare these two statements and determine if they contradict each other.\n\n\
                 Statement A (new): {}\n\n\
                 Statement B (existing, ID: {}): {}\n\n\
                 Respond with JSON: {{\"contradiction\": true|false, \"explanation\": \"brief reason\"}}",
                content, existing.id, existing_content
            );
            let result = {
                let _guard = self.llm.acquire(llm_priority).await;
                self.llm.inner().complete(&prompt).await
            };
            if let Ok(response) = result
                && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response)
                && parsed
                    .get("contradiction")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            {
                let explanation = parsed
                    .get("explanation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("no explanation provided");
                tracing::warn!(
                    "Potential contradiction: new memory {} vs existing {}: {}",
                    memory.id,
                    existing.id,
                    explanation
                );
            }
        }
    }

    /// Search for semantically similar existing memories and create
    /// auto-link relations from `memory` → each match above threshold.
    ///
    /// Mutates `memory` in-place, adding relations to both
    /// `memory.relations` (structured) and `memory.metadata.relations` (flat).
    async fn auto_link_memory(
        &self,
        memory: &mut Memory,
        threshold: f32,
        max_links: usize,
        weights: (u8, u8, u8),
    ) -> Result<usize> {
        if threshold <= 0.0 || max_links == 0 {
            return Ok(0);
        }

        let (primary_pct, ctx_pct, _rel_pct) = weights;

        let filters = Filters::for_user_scope(
            memory.metadata.user_id.clone(),
            memory.metadata.agent_id.clone(),
            memory.metadata.run_id.clone(),
            memory.metadata.actor_id.clone(),
        );

        // Wider search to capture candidates for all link types
        let search_limit = (max_links * 4).min(60);
        let scored = self
            .vector_store
            .search(&memory.embedding, &filters, search_limit)
            .await?;

        if scored.is_empty() {
            return Ok(0);
        }

        // Compute multi-vector scores for each candidate
        #[derive(Debug, Clone)]
        struct CandidateScore {
            memory: Memory,
            primary_score: f32,
            context_score: f32,
            relation_score: f32,
        }

        let mut candidates: Vec<CandidateScore> = Vec::new();
        for s in scored {
            if s.memory.id == memory.id {
                continue;
            }

            let ctx_score = SearchService::cross_max_cosine_similarity(
                &memory.context_embeddings,
                &s.memory.context_embeddings,
            );
            let rel_score = SearchService::cross_max_cosine_similarity(
                &memory.relation_embeddings,
                &s.memory.relation_embeddings,
            );

            candidates.push(CandidateScore {
                memory: s.memory,
                primary_score: s.score,
                context_score: ctx_score,
                relation_score: rel_score,
            });
        }

        if candidates.is_empty() {
            return Ok(0);
        }

        // Proportionally allocate slots across link types
        let primary_slots = (max_links * primary_pct as usize) / 100;
        let ctx_slots = (max_links * ctx_pct as usize) / 100;
        let rel_slots = max_links.saturating_sub(primary_slots + ctx_slots);

        let mut used_targets: HashSet<String> = HashSet::new();
        let mut linked = 0usize;

        // Primary links: best primary_score above threshold
        candidates.sort_by(|a, b| {
            b.primary_score
                .partial_cmp(&a.primary_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut remaining: Vec<CandidateScore> = Vec::new();
        for c in candidates {
            if linked >= primary_slots {
                remaining.push(c);
                continue;
            }
            if c.primary_score < threshold {
                remaining.push(c);
                continue;
            }
            if !used_targets.insert(c.memory.id.clone()) {
                remaining.push(c);
                continue;
            }
            let target_id = match uuid::Uuid::parse_str(&c.memory.id) {
                Ok(id) => id,
                Err(_) => {
                    remaining.push(c);
                    continue;
                }
            };
            let meta = RelationMeta::new("auto_link").with_confidence(c.primary_score);
            memory.append_relation("references", target_id, Some(c.primary_score), meta);
            memory.metadata.relations.push(Relation {
                source: memory.id.clone(),
                relation: "references".into(),
                target: c.memory.id.clone(),
                strength: Some(c.primary_score),
            });
            linked += 1;
            used_targets.insert(c.memory.id.clone());
        }

        // Context links: best context_score above threshold (from remaining)
        remaining.sort_by(|a, b| {
            b.context_score
                .partial_cmp(&a.context_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let ctx_linked_start = linked;
        let mut after_ctx: Vec<CandidateScore> = Vec::new();
        for c in remaining {
            if linked - ctx_linked_start >= ctx_slots {
                after_ctx.push(c);
                continue;
            }
            if c.context_score < threshold {
                after_ctx.push(c);
                continue;
            }
            if used_targets.contains(&c.memory.id) {
                after_ctx.push(c);
                continue;
            }
            let target_id = match uuid::Uuid::parse_str(&c.memory.id) {
                Ok(id) => id,
                Err(_) => {
                    after_ctx.push(c);
                    continue;
                }
            };
            let meta = RelationMeta::new("auto_link:context").with_confidence(c.context_score);
            memory.append_relation("context_link", target_id, Some(c.context_score), meta);
            memory.metadata.relations.push(Relation {
                source: memory.id.clone(),
                relation: "context_link".into(),
                target: c.memory.id.clone(),
                strength: Some(c.context_score),
            });
            linked += 1;
            used_targets.insert(c.memory.id.clone());
        }

        // Relation links: best relation_score above threshold (from remaining)
        after_ctx.sort_by(|a, b| {
            b.relation_score
                .partial_cmp(&a.relation_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let rel_linked_start = linked;
        for c in after_ctx {
            if linked - rel_linked_start >= rel_slots {
                break;
            }
            if c.relation_score < threshold {
                continue;
            }
            if used_targets.contains(&c.memory.id) {
                continue;
            }
            let target_id = match uuid::Uuid::parse_str(&c.memory.id) {
                Ok(id) => id,
                Err(_) => continue,
            };
            let meta =
                RelationMeta::new("auto_link:relation").with_confidence(c.relation_score);
            memory.append_relation("relation_link", target_id, Some(c.relation_score), meta);
            memory.metadata.relations.push(Relation {
                source: memory.id.clone(),
                relation: "relation_link".into(),
                target: c.memory.id.clone(),
                strength: Some(c.relation_score),
            });
            linked += 1;
            used_targets.insert(c.memory.id.clone());
        }

        Ok(linked)
    }

    /// Wire caller-supplied explicit relations (part_of, used_by, etc.) into the
    /// Memory.relations HashMap. If LLM validation is enabled, each relation is
    /// verified before wiring. Targets that don't exist yet are queued as pending.
    ///
    /// Returns the number of relations wired (both forward and reverse).
    async fn wire_explicit_relations(
        &self,
        memory: &mut Memory,
        bank_name: &str,
    ) -> Result<usize> {
        let relations: Vec<Relation> = memory.metadata.relations.clone();
        if relations.is_empty() {
            return Ok(0);
        }

        let validate = self.config.llm_relation_validation;
        let mut wired = 0usize;

        for rel in &relations {
            let target_id = match uuid::Uuid::parse_str(&rel.target) {
                Ok(id) => id,
                Err(_) => continue,
            };

            let target_memory = match self.vector_store.get(&rel.target).await? {
                Some(m) => m,
                None => {
                    if let Ok(wal_guard) = self.pending_wal.lock()
                        && let Some(ref wal) = *wal_guard {
                        let source_id = match uuid::Uuid::parse_str(&memory.id) {
                            Ok(id) => id,
                            Err(_) => continue,
                        };
                        let entry = PendingRelationEntry {
                            source_id,
                            target_id,
                            relation: rel.relation.clone(),
                            strength: rel.strength,
                            bank_name: bank_name.to_string(),
                            created_at: chrono::Utc::now(),
                        };
                        let _ = wal.insert_pending_relation(&entry);
                    }
                    tracing::debug!(
                        "Relation target {} not found: queued as pending ({} -> {})",
                        rel.target,
                        memory.id,
                        rel.relation
                    );
                    continue;
                }
            };

            if validate {
                let source_content = memory.content.as_deref().unwrap_or("");
                let target_content = target_memory.content.as_deref().unwrap_or("");

                match self
                    .validate_relation(source_content, target_content, &rel.relation)
                    .await
                {
                    Ok(v) if v.valid && v.confidence >= 0.6 => {
                        let verified_rel = v.suggested_relation.as_deref().unwrap_or(&rel.relation);
                        self.wire_relation(memory, &target_memory, verified_rel, rel.strength)
                            .await?;
                        wired += 1;
                    }
                    Ok(v) => {
                        tracing::info!(
                            "Relation {} -> {} [{}] rejected by LLM (confidence={:.2})",
                            memory.id,
                            rel.target,
                            rel.relation,
                            v.confidence
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            "LLM validation failed for {} -> {}: {}. Wiring anyway.",
                            memory.id,
                            rel.target,
                            e
                        );
                        self.wire_relation(memory, &target_memory, &rel.relation, rel.strength)
                            .await?;
                        wired += 1;
                    }
                }
            } else {
                self.wire_relation(memory, &target_memory, &rel.relation, rel.strength)
                    .await?;
                wired += 1;
            }
        }

        Ok(wired)
    }

    /// Wire a bidirectional relation between two memories.
    async fn wire_relation(
        &self,
        source: &mut Memory,
        target: &Memory,
        relation: &str,
        strength: Option<f32>,
    ) -> Result<()> {
        let target_id = match uuid::Uuid::parse_str(&target.id) {
            Ok(id) => id,
            Err(_) => return Ok(()),
        };
        let source_id = match uuid::Uuid::parse_str(&source.id) {
            Ok(id) => id,
            Err(_) => return Ok(()),
        };

        let meta = RelationMeta::new("user").with_confidence(strength.unwrap_or(0.8));
        source.append_relation(relation, target_id, strength, meta.clone());

        if let Some(reverse) = reverse_relation(relation) {
            let mut target_mut = target.clone();
            target_mut.append_relation(reverse, source_id, strength, meta);
            let _ = self.vector_store.update(&target_mut).await;
        }

        Ok(())
    }

    /// Use LLM to validate whether a claimed relation between two contents is valid.
    async fn validate_relation(
        &self,
        source_content: &str,
        target_content: &str,
        relation: &str,
    ) -> Result<RelationValidation> {
        let max_len = 1500usize;
        let src_snippet: String = source_content.chars().take(max_len).collect();
        let tgt_snippet: String = target_content.chars().take(max_len).collect();

        let prompt = format!(
            "Validate whether this relationship claim between two contents is correct.\n\
             \n\
             SOURCE:\n{src}\n\
             \n\
             TARGET:\n{tgt}\n\
             \n\
             CLAIMED RELATION: {rel}\n\
             \n\
             Respond with ONLY JSON: {{\"valid\": true|false, \"confidence\": 0.0-1.0, \"suggested_relation\": \"better relation or null\"}}",
            src = src_snippet,
            tgt = tgt_snippet,
            rel = relation
        );

        let response = {
            let _guard = self.llm.acquire(LlmPriority::Background).await;
            self.llm.inner().complete(&prompt).await?
        };

        let json_str = crate::llm::client::extract_json_from_text_tagged(&response, &["think".to_string()])
            .unwrap_or(response);
        let repaired = jsonrepair::repair_json(&json_str, &jsonrepair::Options::default())
            .unwrap_or(json_str);

        let val: serde_json::Value =
            serde_json::from_str(&repaired).map_err(|e| {
                MemoryError::LLM(format!("Failed to parse relation validation JSON: {}", e))
            })?;

        Ok(RelationValidation {
            valid: val.get("valid").and_then(|v| v.as_bool()).unwrap_or(false),
            confidence: val
                .get("confidence")
                .and_then(|v| v.as_f64())
                .map(|c| c as f32)
                .unwrap_or(0.0),
            suggested_relation: val
                .get("suggested_relation")
                .and_then(|v| v.as_str())
                .filter(|s| *s != "null" && !s.is_empty())
                .map(|s| s.to_string()),
        })
    }

    /// Resolve all pending relations that target a newly stored memory.
    pub async fn resolve_pending_relations_for(
        &self,
        memory: &Memory,
        _bank_name: &str,
    ) -> Result<usize> {
        let wal = match self.pending_wal.lock() {
            Ok(guard) => match guard.as_ref() {
                Some(w) => w.clone(),
                None => return Ok(0),
            },
            Err(_) => return Ok(0),
        };

        let target_id = match uuid::Uuid::parse_str(&memory.id) {
            Ok(id) => id,
            Err(_) => return Ok(0),
        };

        let pending = wal.load_pending_for_target(&target_id)?;
        if pending.is_empty() {
            return Ok(0);
        }

        let mut resolved = 0usize;
        let validate = self.config.llm_relation_validation;
        let target_content = memory.content.as_deref().unwrap_or("");

        for entry in &pending {
            let source_memory = match self.vector_store.get(&entry.source_id.to_string()).await? {
                Some(m) => m,
                None => continue,
            };

            let source_content = source_memory.content.as_deref().unwrap_or("");

            let relation = if validate {
                match self
                    .validate_relation(source_content, target_content, &entry.relation)
                    .await
                {
                    Ok(v) if v.valid && v.confidence >= 0.6 => v.suggested_relation
                        .as_deref()
                        .unwrap_or(&entry.relation)
                        .to_string(),
                    Ok(_) => {
                        tracing::info!(
                            "Pending relation {} -> {} [{}] rejected by LLM",
                            entry.source_id,
                            entry.target_id,
                            entry.relation
                        );
                        let _ = wal.remove_pending_relation(
                            &entry.source_id,
                            &entry.target_id,
                            &entry.relation,
                            &entry.bank_name,
                        );
                        continue;
                    }
                    Err(_) => entry.relation.clone(),
                }
            } else {
                entry.relation.clone()
            };

            let meta = RelationMeta::new("pending_resolution")
                .with_confidence(entry.strength.unwrap_or(0.8));

            let mut source_mut = source_memory;
            source_mut.append_relation(&relation, target_id, entry.strength, meta.clone());
            let _ = self.vector_store.update(&source_mut).await;

            if let Some(reverse) = reverse_relation(&relation) {
                let mut target_mut = memory.clone();
                let source_id = entry.source_id;
                target_mut.append_relation(reverse, source_id, entry.strength, meta);
                let _ = self.vector_store.update(&target_mut).await;
            }

            let _ = wal.remove_pending_relation(
                &entry.source_id,
                &entry.target_id,
                &entry.relation,
                &entry.bank_name,
            );
            resolved += 1;
        }

        if resolved > 0 {
            tracing::info!(
                "Resolved {} pending relation(s) for newly stored memory {}",
                resolved,
                memory.id
            );
        }

        Ok(resolved)
    }

    /// For long L0 memories, split the content into overlapping chunks,
    /// embed each chunk, and store as child records with `parent_id`.
    /// The chunk records use the parent memory's ID so search can resolve them.
    async fn store_content_chunks(&self, parent: &Memory, llm_priority: LlmPriority) -> Result<()> {
        let threshold = self.config.chunk_threshold_chars;
        if threshold == 0 {
            return Ok(());
        }
        if parent.metadata.layer.level != 0 || parent.metadata.parent_id.is_some() {
            return Ok(());
        }
        let content = match &parent.content {
            Some(c) if c.len() > threshold => c.as_str(),
            _ => return Ok(()),
        };

        let chunks = crate::memory::utils::chunk_text_overlapping(
            content,
            self.config.chunk_size_chars,
            self.config.chunk_overlap_chars,
        );
        if chunks.is_empty() || chunks.len() == 1 {
            return Ok(());
        }

        let parent_id = uuid::Uuid::parse_str(&parent.id).ok();

        let start = Instant::now();
        let embeddings: Vec<Vec<f32>> = {
            let _guard = self.llm.acquire(llm_priority).await;
            self.llm.inner().embed_batch(&chunks).await?
        };
        self.metrics
            .record_ingestion_timing(IngestionPhase::ContentChunkEmbed, start.elapsed());

        if embeddings.len() != chunks.len() {
            tracing::warn!(
                "Chunk embedding mismatch: got {} embeddings for {} chunks — skipping chunk index",
                embeddings.len(),
                chunks.len()
            );
            return Ok(());
        }

        for (i, chunk_text) in chunks.into_iter().enumerate() {
            let chunk_id = uuid::Uuid::new_v4();

            // Propagate parent keywords to chunks so keyword search boost applies
            let mut chunk_custom = HashMap::new();
            if let Some(keywords) = parent.metadata.custom.get("keywords") {
                chunk_custom.insert("keywords".to_string(), keywords.clone());
            }

            let mut chunk_memory = Memory::with_content(
                chunk_text,
                embeddings[i].clone(),
                MemoryMetadata {
                    layer: LayerInfo::raw_content(),
                    user_id: parent.metadata.user_id.clone(),
                    agent_id: parent.metadata.agent_id.clone(),
                    run_id: parent.metadata.run_id.clone(),
                    parent_id,
                    custom: chunk_custom,
                    ..MemoryMetadata::new()
                },
            );
            // Propagate the parent's full content_meta (source, provided_by,
            // content_type, quality_flags, custom) so chunks carry the same
            // provenance as the parent. This is especially important for
            // content_meta.source — the L0 invariant says chunks are
            // immutable excerpts of the original; provenance must survive.
            chunk_memory.content_meta = parent.content_meta.clone();
            chunk_memory.id = chunk_id.to_string();
            chunk_memory.created_at = parent.created_at;
            chunk_memory.updated_at = chrono::Utc::now();
            chunk_memory.event_at = parent.event_at;

            let _start = Instant::now();
            self.vector_store.insert(&chunk_memory).await?;
            self.metrics
                .record_ingestion_timing(IngestionPhase::VsInsert, _start.elapsed());

            let _start = Instant::now();
            self.search.insert_layer(0).await;
            self.metrics
                .record_ingestion_timing(IngestionPhase::LayerManifestUpdate, _start.elapsed());
        }

        tracing::debug!(
            "Stored {} content chunks for parent {} ({} chars total)",
            embeddings.len(),
            parent.id,
            parent.content.as_ref().map_or(0, |c| c.len())
        );

        Ok(())
    }

    /// Add memory from conversation messages with full fact extraction and update pipeline
    pub async fn add_memory(
        &self,
        messages: &[Message],
        metadata: MemoryMetadata,
    ) -> Result<Vec<MemoryResult>> {
        self.add_memory_with_event_at(messages, metadata, None)
            .await
    }

    /// Same as `add_memory` but with an explicit `event_at` to apply to every stored memory.
    pub async fn add_memory_with_event_at(
        &self,
        messages: &[Message],
        metadata: MemoryMetadata,
        event_at: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<Vec<MemoryResult>> {
        if messages.is_empty() {
            return Ok(vec![]);
        }

        let extracted_facts = self.fact_extractor.extract_facts(messages).await?;
        let mut final_extracted_facts = extracted_facts;

        if final_extracted_facts.is_empty() {
            let user_messages: Vec<_> = messages
                .iter()
                .filter(|msg| msg.role == "user")
                .cloned()
                .collect();

            if !user_messages.is_empty()
                && let Ok(user_facts) = self.fact_extractor.extract_user_facts(&user_messages).await
                && !user_facts.is_empty()
            {
                final_extracted_facts = user_facts;
            }

            if final_extracted_facts.is_empty() {
                let mut single_message_facts = Vec::new();
                for message in messages {
                    if let Ok(mut facts) = self
                        .fact_extractor
                        .extract_facts_from_text(&message.content)
                        .await
                    {
                        for fact in &mut facts {
                            fact.source_role = message.role.clone();
                        }
                        single_message_facts.extend(facts);
                    }
                }
                if !single_message_facts.is_empty() {
                    final_extracted_facts = single_message_facts;
                }
            }

            if final_extracted_facts.is_empty() {
                let user_content = messages
                    .iter()
                    .filter(|msg| msg.role == "user")
                    .map(|msg| format!("User: {}", msg.content))
                    .collect::<Vec<_>>()
                    .join("\n");

                if !user_content.trim().is_empty() {
                    let memory_id = self
                        .store_with_options(
                            user_content.clone(),
                            metadata,
                            StoreOptions {
                                event_at,
                                ..StoreOptions::default()
                            },
                        )
                        .await?;
                    return Ok(vec![MemoryResult {
                        id: memory_id,
                        memory: user_content,
                        event: MemoryEvent::Add,
                        actor_id: messages.last().and_then(|msg| msg.name.clone()),
                        role: messages.last().map(|msg| msg.role.clone()),
                        previous_memory: None,
                    }]);
                }
                return Ok(vec![]);
            }
        }

        let original_content: String = messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let res = {
            let _guard = self.llm.acquire(LlmPriority::Background).await;
            self.llm.inner().extract_keywords(&original_content).await
        };
        let extracted_keywords = match res {
            Ok(keywords) => keywords,
            Err(e) => {
                tracing::debug!("Failed to extract keywords: {}", e);
                Vec::new()
            }
        };

        let mut all_actions = Vec::new();

        for fact in &final_extracted_facts {
            let filters = Filters::for_user_scope(
                metadata.user_id.clone(),
                metadata.agent_id.clone(),
                metadata.run_id.clone(),
                metadata.actor_id.clone(),
            );

            let query_embedding = {
                let _guard = self.llm.acquire(LlmPriority::Background).await;
                self.llm.inner().embed(&fact.content).await?
            };
            let existing_memories = self
                .vector_store
                .search_with_threshold(
                    &query_embedding,
                    &filters,
                    5,
                    self.config.search_similarity_threshold,
                )
                .await?;

            let update_result = self
                .memory_updater
                .update_memories(std::slice::from_ref(fact), &existing_memories, &metadata)
                .await?;

            for action in &update_result.actions_performed {
                match action {
                    MemoryAction::Create { content, metadata } => {
                        let mut metadata_with_keywords = (**metadata).clone();
                        if !extracted_keywords.is_empty() {
                            let keywords_json: Vec<serde_json::Value> = extracted_keywords
                                .iter()
                                .map(|k| serde_json::Value::String(k.clone()))
                                .collect();
                            metadata_with_keywords.custom.insert(
                                "keywords".to_string(),
                                serde_json::Value::Array(keywords_json),
                            );
                        }
                        let memory_id = self
                            .store_with_options(
                                content.clone(),
                                metadata_with_keywords,
                                StoreOptions {
                                    event_at,
                                    ..StoreOptions::default()
                                },
                            )
                            .await?;
                        all_actions.push(MemoryResult {
                            id: memory_id,
                            memory: content.clone(),
                            event: MemoryEvent::Add,
                            actor_id: messages.last().and_then(|msg| msg.name.clone()),
                            role: messages.last().map(|msg| msg.role.clone()),
                            previous_memory: None,
                        });
                    }
                    MemoryAction::Update { id, content } => {
                        let _ = self.update(id, Some(content.clone()), None).await;
                        all_actions.push(MemoryResult {
                            id: id.clone(),
                            memory: content.clone(),
                            event: MemoryEvent::Update,
                            actor_id: messages.last().and_then(|msg| msg.name.clone()),
                            role: messages.last().map(|msg| msg.role.clone()),
                            previous_memory: None,
                        });
                    }
                    MemoryAction::Merge {
                        target_id,
                        source_ids,
                        merged_content,
                    } => {
                        let _ = self
                            .update(target_id, Some(merged_content.clone()), None)
                            .await;
                        for source_id in source_ids {
                            let _ = self.delete(source_id).await;
                        }
                        all_actions.push(MemoryResult {
                            id: target_id.clone(),
                            memory: merged_content.clone(),
                            event: MemoryEvent::Update,
                            actor_id: messages.last().and_then(|msg| msg.name.clone()),
                            role: messages.last().map(|msg| msg.role.clone()),
                            previous_memory: None,
                        });
                    }
                    MemoryAction::Delete { id } => {
                        let _ = self.delete(id).await;
                        all_actions.push(MemoryResult {
                            id: id.clone(),
                            memory: String::new(),
                            event: MemoryEvent::Delete,
                            actor_id: messages.last().and_then(|msg| msg.name.clone()),
                            role: messages.last().map(|msg| msg.role.clone()),
                            previous_memory: None,
                        });
                    }
                }
            }
        }

        // Compact WAL to persist all facts from this batch
        let _ = self.vector_store.compact().await;

        Ok(all_actions)
    }

    /// Ingest a document by extracting facts and storing them
    /// Create procedural memory using specialized prompt system
    /// Update an existing memory
    pub async fn update(
        &self,
        id: &str,
        content: Option<String>,
        relations: Option<Vec<Relation>>,
    ) -> Result<()> {
        let mut memory = self
            .vector_store
            .get(id)
            .await?
            .ok_or_else(|| MemoryError::NotFound { id: id.to_string() })?;

        if let Some(c) = content {
            memory.content = Some(c.clone());
            memory.content_meta.checksum = Some(ContentMeta::compute_checksum(&c));
            memory.embedding = {
                let _guard = self.llm.acquire(LlmPriority::Background).await;
                self.llm.inner().embed(&c).await?
            };
            memory.metadata.hash = Self::generate_hash(&c);
            if self.config.auto_metadata_analysis {
                self.enhance_memory(&mut memory, LlmPriority::Background)
                    .await?;
            }
        }

        if let Some(new_relations) = relations {
            for new_rel in new_relations {
                if !memory
                    .metadata
                    .relations
                    .iter()
                    .any(|r| r.relation == new_rel.relation && r.target == new_rel.target)
                {
                    memory.metadata.relations.push(new_rel);
                }
            }
        }

        memory.updated_at = chrono::Utc::now();
        self.vector_store.update(&memory).await?;
        self.search.insert_layer(memory.metadata.layer.level).await;

        Ok(())
    }

    /// Update a complete memory object directly
    pub async fn update_memory(&self, memory: &Memory) -> Result<()> {
        self.vector_store.update(memory).await?;
        self.search.insert_layer(memory.metadata.layer.level).await;
        Ok(())
    }

    /// Store a pre-constructed memory directly (bypassing normal pipelines)
    pub async fn store_memory(&self, memory: Memory) -> Result<String> {
        self.vector_store.insert(&memory).await?;
        self.search.insert_layer(memory.metadata.layer.level).await;
        Ok(memory.id)
    }

    /// Delete a memory by ID
    pub async fn delete(&self, id: &str) -> Result<()> {
        self.vector_store.delete(id).await?;
        Ok(())
    }

    /// Retrieve a memory by ID
    pub async fn get(&self, id: &str) -> Result<Option<Memory>> {
        self.vector_store.get(id).await
    }

    /// List memories with optional filters
    pub async fn list(&self, filters: &Filters, limit: Option<usize>) -> Result<Vec<Memory>> {
        self.vector_store.list(filters, limit).await
    }

    /// Ingest raw content through the format-aware decomposition pipeline.
    ///
    /// Detects format, parses into a DocumentNode tree, chunks into L0 memories,
    /// adds structural relations, and optionally auto-links to existing memories.
    pub async fn ingest(
        &self,
        opts: IngestOptions,
    ) -> Result<crate::ingest::feedback::IngestResult> {
        use crate::ingest::feedback::IngestResult;
        use crate::ingest::format_detect::{InputFormat, detect_format};

        let IngestOptions {
            content,
            content_encoding,
            format_hint,
            file_name,
            auto_link,
            generate_abstractions,
            max_chunk_size,
            user_metadata,
            source: explicit_source,
            describe_images,
        } = opts;

        let is_base64 = content_encoding.as_deref() == Some("base64");

        let byte_size = if is_base64 {
            base64_decode_size(&content).unwrap_or(content.len()) as u64
        } else {
            content.len() as u64
        };

        let session_id = uuid::Uuid::new_v4().to_string();
        let max_chunk = max_chunk_size.unwrap_or(2000);

        let mut fmt = if is_base64 {
            detect_format("", file_name.as_deref(), format_hint.as_deref())
        } else {
            detect_format(&content, file_name.as_deref(), format_hint.as_deref())
        };

        if fmt == InputFormat::Unknown && self.config.llm_format_detection && !is_base64 {
            let advisor = crate::llm::LLMStrategyAdvisor::new(self.llm.inner());
            if let Some(detected) = advisor.detect_format(&content).await {
                tracing::info!(
                    format = detected.name(),
                    "LLM strategy advisor detected format"
                );
                fmt = detected;
            }
        }

        let is_binary = matches!(
            fmt,
            InputFormat::Pdf
                | InputFormat::Word
                | InputFormat::Excel
                | InputFormat::ImagePng
                | InputFormat::ImageJpeg
                | InputFormat::ImageGif
                | InputFormat::ImageWebp
        );

        let is_image = fmt.is_image();

        let mut image_data: Option<Vec<u8>> = None;

        let (doc, doc_meta) = if is_binary {
            let data = match decode_base64_or_raw(&content, is_base64) {
                Ok(d) => d,
                Err(e) => {
                    return Ok(IngestResult::new(session_id, fmt.name(), fmt.mime(), byte_size)
                        .with_issue(crate::ingest::feedback::IngestIssue::blocking(
                            format!("Base64 decode failure: {}", e),
                            "Ensure the content is valid base64 with content_encoding set to 'base64'.",
                        )));
                }
            };
            if is_image {
                image_data = Some(data.clone());
            }
            match crate::ingest::parsers::parse_binary(&data, fmt) {
                Ok(result) => result,
                Err(e) => {
                    return Ok(
                        IngestResult::new(session_id, fmt.name(), fmt.mime(), byte_size)
                            .with_issue(crate::ingest::feedback::IngestIssue::blocking(
                                format!("Binary parse failure: {}", e),
                                "Try a different format or file.",
                            )),
                    );
                }
            }
        } else {
            let content_str = if is_base64 {
                match decode_base64_to_string(&content) {
                    Ok(s) => s,
                    Err(e) => {
                        return Ok(IngestResult::new(session_id, fmt.name(), fmt.mime(), byte_size)
                            .with_issue(crate::ingest::feedback::IngestIssue::blocking(
                                format!("Base64 decode failure: {}", e),
                                "Ensure the content is valid base64 with content_encoding set to 'base64'.",
                            )));
                    }
                }
            } else {
                content.clone()
            };

            match crate::ingest::parsers::parse(&content_str, fmt, file_name.as_deref()) {
                Ok(result) => result,
                Err(parse_err) => {
                    if self.config.llm_fallback_parsing {
                        let advisor = crate::llm::LLMStrategyAdvisor::new(self.llm.inner());
                        match advisor.fallback_parse(&content_str, fmt.name()).await {
                            Ok(fallback_result) => {
                                tracing::info!(
                                    "LLM fallback parser succeeded for format {}",
                                    fmt.name()
                                );
                                fallback_result
                            }
                            Err(llm_err) => {
                                tracing::warn!(
                                    error = %llm_err,
                                    "LLM fallback parser also failed"
                                );
                                return Ok(IngestResult::new(session_id, fmt.name(), fmt.mime(), byte_size)
                                    .with_issue(crate::ingest::feedback::IngestIssue::blocking(
                                        format!("Parse failure: {}. LLM fallback also failed: {}", parse_err, llm_err),
                                        "Try a different format_hint or pre-process the content into a supported format.",
                                    )));
                            }
                        }
                    } else {
                        return Ok(IngestResult::new(session_id, fmt.name(), fmt.mime(), byte_size)
                            .with_issue(crate::ingest::feedback::IngestIssue::blocking(
                                format!("Parse failure: {}", parse_err),
                                "Try a different format_hint or pre-process the content into a supported format.",
                            )));
                    }
                }
            }
        };

        // For images, preserve the raw base64 content as content_meta.image_data
        // so the original user input is retrievable (L0 stores only parser metadata).
        let raw_image_base64: Option<String> = if is_image {
            if is_base64 {
                Some(content.clone())
            } else {
                use base64::{Engine, engine::general_purpose::STANDARD};
                image_data.as_ref().map(|data| STANDARD.encode(data))
            }
        } else {
            None
        };

        let mut result = IngestResult::new(session_id, fmt.name(), fmt.mime(), byte_size);
        for warning in &doc_meta.warnings {
            result.warnings.push(warning.clone());
        }

        let chunking = crate::ingest::chunker::chunk_document(&doc, max_chunk);

        let user_id = user_metadata.as_ref().and_then(|m| m.user_id.clone());
        let agent_id = user_metadata.as_ref().and_then(|m| m.agent_id.clone());

        // Derive a free-form source description for L0 provenance.
        // Priority: caller-supplied explicit_source, else
        // "<filename> — <title>" if both present, else whichever exists.
        // None if none are available.
        let source_str: Option<String> = if let Some(explicit) = explicit_source {
            Some(explicit)
        } else {
            let doc_title = extract_document_title(&doc);
            match (file_name.as_deref(), doc_title.as_deref()) {
                (Some(f), Some(t)) => Some(format!("{} \u{2014} {}", f, t)),
                (Some(f), None) => Some(f.to_string()),
                (None, Some(t)) => Some(t.to_string()),
                (None, None) => None,
            }
        };

        let mut chunk_ids: Vec<String> = Vec::new();
        let mut chunk_memory_ids: Vec<String> = Vec::new();

        for chunk in &chunking.chunks {
            let chunk_id = uuid::Uuid::new_v4().to_string();
            let mut meta = if let Some(ref base) = user_metadata {
                let mut m = base.clone();
                m.layer = LayerInfo::raw_content();
                m
            } else {
                let mut m = MemoryMetadata::new().with_layer(LayerInfo::raw_content());
                if let Some(ref uid) = user_id {
                    m = m.with_user_id(uid.clone());
                }
                if let Some(ref aid) = agent_id {
                    m = m.with_agent_id(aid.clone());
                }
                m
            };

            meta.custom
                .insert("chunk_order".into(), serde_json::json!(chunk.order));
            meta.custom
                .insert("node_type".into(), serde_json::json!(&chunk.node_type));
            meta.custom.insert(
                "ingest_session".into(),
                serde_json::json!(&result.session_id),
            );

            let do_auto_link = auto_link.unwrap_or(true);
            let options = StoreOptions {
                llm_priority: LlmPriority::Interactive,
                auto_link: Some(do_auto_link),
                source: source_str.clone(),
                image_data: raw_image_base64.clone(),
                ..StoreOptions::default()
            };

            let memory_id = self
                .store_with_options(chunk.content.clone(), meta.clone(), options)
                .await?;

            chunk_ids.push(chunk_id.clone());
            chunk_memory_ids.push(memory_id.clone());

            result.l0_chunks.push(crate::ingest::feedback::ChunkInfo {
                id: chunk_id,
                memory_id: Some(memory_id.clone()),
                node_type: chunk.node_type.clone(),
                content_preview: chunk.content.chars().take(80).collect(),
                char_count: chunk.content.len(),
                order: chunk.order,
            });
        }

        for rel in &chunking.relations {
            if rel.source_idx < chunk_ids.len() && rel.target_idx < chunk_ids.len() {
                let source_mem_id = &chunk_memory_ids[rel.source_idx];
                let target_mem_id = &chunk_memory_ids[rel.target_idx];

                if let Some(mut source) = self.vector_store.get(source_mem_id).await? {
                    source.metadata.relations.push(Relation {
                        source: source_mem_id.clone(),
                        relation: rel.relation.clone(),
                        target: target_mem_id.clone(),
                        strength: rel.strength,
                    });
                    self.vector_store.update(&source).await?;
                }

                result
                    .relations
                    .push(crate::ingest::feedback::RelationInfo {
                        source_chunk_id: chunk_ids[rel.source_idx].clone(),
                        target_chunk_id: chunk_ids[rel.target_idx].clone(),
                        relation: rel.relation.clone(),
                        strength: rel.strength,
                    });
            }
        }

        let _generate = generate_abstractions.unwrap_or(true);

        for (i, chunk) in chunking.chunks.iter().enumerate() {
            if chunk.node_type == "table"
                && let Some(schema_desc) = infer_table_schema(&chunk.content)
            {
                let mem_id = &chunk_memory_ids[i];
                let l1_content = format!(
                    "[L1 Table Schema] {}\n\nSource table L0 chunk: {}",
                    schema_desc, mem_id
                );
                let l1_meta = MemoryMetadata::new().with_layer(LayerInfo::structural());
                let result_id = self
                    .store_with_options(
                        l1_content,
                        l1_meta,
                        StoreOptions {
                            llm_priority: LlmPriority::Interactive,
                            auto_link: Some(false),
                            ..StoreOptions::default()
                        },
                    )
                    .await?;

                if let Some(mut source) = self.vector_store.get(mem_id).await? {
                    source.metadata.relations.push(Relation {
                        source: result_id.clone(),
                        relation: "l1_of".into(),
                        target: mem_id.clone(),
                        strength: Some(0.9),
                    });
                    self.vector_store.update(&source).await?;
                }

                result
                    .l1_abstractions
                    .push(crate::ingest::feedback::AbstractionInfo {
                        id: Some(uuid::Uuid::new_v4().to_string()),
                        memory_id: Some(result_id),
                        abstraction_type: "table_schema".into(),
                        source_chunk_ids: vec![chunk_ids[i].clone()],
                        layer: 1,
                        content_preview: schema_desc.chars().take(80).collect(),
                    });
            }
        }

        let do_describe = describe_images.unwrap_or(true) && generate_abstractions.unwrap_or(true);

        if let Some(ref img_bytes) = image_data
            && do_describe
        {
            const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;

            if img_bytes.len() > MAX_IMAGE_BYTES {
                result.warnings.push(format!(
                    "Image too large for vision description ({} MB, limit {} MB). Skipping image description.",
                    img_bytes.len() / (1024 * 1024),
                    MAX_IMAGE_BYTES / (1024 * 1024)
                ));
                result.vision_status = Some(crate::ingest::feedback::VisionStatus {
                    images_ingested: 1,
                    descriptions_generated: 0,
                    outcome: crate::ingest::feedback::VisionOutcome::Unavailable,
                    detail: Some(format!(
                        "Image size {} bytes exceeds limit {}",
                        img_bytes.len(),
                        MAX_IMAGE_BYTES
                    )),
                });
            } else {
                let vision_outcome =
                    match self.llm.inner().describe_image(img_bytes, fmt.mime()).await {
                        Ok(description) => {
                            let l1_content = format!(
                                "[L1 Image Description] {}\n\nSource image L0 chunk: session {}",
                                description, result.session_id
                            );
                            let l1_meta = MemoryMetadata::new().with_layer(LayerInfo::structural());
                            let result_id = self
                                .store_with_options(
                                    l1_content,
                                    l1_meta,
                                    StoreOptions {
                                        llm_priority: LlmPriority::Interactive,
                                        auto_link: Some(false),
                                        ..StoreOptions::default()
                                    },
                                )
                                .await?;

                            for mem_id in &chunk_memory_ids {
                                if let Some(mut source) = self.vector_store.get(mem_id).await? {
                                    source.metadata.relations.push(Relation {
                                        source: result_id.clone(),
                                        relation: "l1_of".into(),
                                        target: mem_id.clone(),
                                        strength: Some(0.9),
                                    });
                                    self.vector_store.update(&source).await?;
                                }
                            }

                            result
                                .l1_abstractions
                                .push(crate::ingest::feedback::AbstractionInfo {
                                    id: Some(uuid::Uuid::new_v4().to_string()),
                                    memory_id: Some(result_id),
                                    abstraction_type: "image_description".into(),
                                    source_chunk_ids: chunk_ids.clone(),
                                    layer: 1,
                                    content_preview: description.chars().take(80).collect(),
                                });

                            crate::ingest::feedback::VisionOutcome::Succeeded
                        }
                        Err(e) => {
                            let msg = e.to_string();
                            result
                                .warnings
                                .push(format!("Image description failed: {}", msg));
                            let lower = msg.to_lowercase();
                            if lower.contains("vision description is disabled")
                                || lower.contains("vision_enabled")
                                || lower.contains("not configured")
                                || lower.contains("not available")
                                || lower.contains("no mmproj")
                                || lower.contains("requires mmproj")
                            {
                                crate::ingest::feedback::VisionOutcome::NotConfigured
                            } else {
                                crate::ingest::feedback::VisionOutcome::Failed
                            }
                        }
                    };
                result.vision_status = Some(crate::ingest::feedback::VisionStatus {
                    images_ingested: 1,
                    descriptions_generated: if vision_outcome
                        == crate::ingest::feedback::VisionOutcome::Succeeded
                    {
                        1
                    } else {
                        0
                    },
                    outcome: vision_outcome,
                    detail: None,
                });
            } // end of else (image within size limit)
        }

        if !result
            .issues
            .iter()
            .any(|i| i.severity == crate::ingest::feedback::IssueSeverity::Blocking)
            && result.l0_chunks.is_empty()
        {
            result
                .warnings
                .push("Content parsed successfully but produced no chunks".into());
        }

        result.format = fmt.name().to_string();
        result.detected_mime = fmt.mime().to_string();

        Ok(result)
    }
}

fn infer_table_schema(content: &str) -> Option<String> {
    let mut lines = content.lines();

    let header_line = lines.next()?;
    let _separator = lines.next();

    let headers: Vec<String> = header_line
        .trim()
        .trim_start_matches('|')
        .trim_end_matches('|')
        .split('|')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let mut rows: Vec<Vec<String>> = Vec::new();
    for line in lines {
        let cells: Vec<String> = line
            .trim()
            .trim_start_matches('|')
            .trim_end_matches('|')
            .split('|')
            .map(|s| s.trim().to_string())
            .collect();
        if !cells.iter().all(|s| s.is_empty()) {
            rows.push(cells);
        }
    }

    if headers.is_empty() {
        return None;
    }

    let col_types: Vec<String> = (0..headers
        .len()
        .min(rows.first().map(|r| r.len()).unwrap_or(0)))
        .map(|col| infer_column_type(col, &rows))
        .collect();

    let mut parts = Vec::new();
    for (i, header) in headers.iter().enumerate() {
        let col_type = col_types.get(i).map(|s| s.as_str()).unwrap_or("string");
        let non_null = rows
            .iter()
            .filter(|r| r.get(i).map(|c| !c.is_empty()).unwrap_or(false))
            .count();
        parts.push(format!("{} ({}, {} non-null)", header, col_type, non_null));
    }

    Some(format!(
        "Table with {} columns and {} rows. Columns: {}.{}",
        headers.len(),
        rows.len(),
        parts.join("; "),
        if rows.is_empty() {
            " Table is empty."
        } else {
            ""
        }
    ))
}

fn infer_column_type(col: usize, rows: &[Vec<String>]) -> String {
    let values: Vec<String> = rows
        .iter()
        .filter_map(|r| r.get(col).cloned())
        .filter(|v| !v.is_empty())
        .collect();

    if values.is_empty() {
        return "string".into();
    }

    let all_ints = values.iter().all(|v| v.parse::<i64>().is_ok());
    if all_ints {
        return "integer".into();
    }

    let all_floats = values
        .iter()
        .all(|v| v.parse::<f64>().is_ok() || v.replace(',', "").parse::<f64>().is_ok());
    if all_floats {
        return "float".into();
    }

    let all_bools = values.iter().all(|v| {
        matches!(
            v.to_lowercase().as_str(),
            "true" | "false" | "yes" | "no" | "1" | "0"
        )
    });
    if all_bools {
        return "boolean".into();
    }

    "string".into()
}

fn base64_decode_size(content: &str) -> Option<usize> {
    let trimmed = content.trim();
    let padding = trimmed.chars().rev().take_while(|&c| c == '=').count();
    Some((trimmed.len() / 4) * 3 - padding)
}

fn decode_base64_or_raw(content: &str, is_base64: bool) -> std::result::Result<Vec<u8>, String> {
    if is_base64 {
        base64_decode(content)
    } else {
        Ok(content.as_bytes().to_vec())
    }
}

fn decode_base64_to_string(content: &str) -> std::result::Result<String, String> {
    let bytes = base64_decode(content)?;
    String::from_utf8(bytes).map_err(|e| format!("UTF-8 decode error: {}", e))
}

fn base64_decode(content: &str) -> std::result::Result<Vec<u8>, String> {
    use base64::{Engine, engine::general_purpose::STANDARD};
    let trimmed = content.trim().replace(|c: char| c.is_whitespace(), "");
    STANDARD
        .decode(trimmed.as_bytes())
        .map_err(|e| format!("Base64 decode error: {}", e))
}

/// Extract the first non-empty `Section` title from a parsed document tree.
/// Used to build a free-form `content_meta.source` for provenance in the
/// ingest pipeline. Returns `None` if the document has no top-level section
/// or only empty-title sections.
fn extract_document_title(doc: &crate::ingest::document_tree::DocumentNode) -> Option<String> {
    if let crate::ingest::document_tree::DocumentNode::Document { children, .. } = doc {
        for child in children {
            if let crate::ingest::document_tree::DocumentNode::Section { title, .. } = child {
                let trimmed = title.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod table_schema_tests {
    use super::*;

    #[test]
    fn test_infer_table_schema_basic() {
        let content = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |\n";
        let schema = infer_table_schema(content).unwrap();
        assert!(schema.contains("Name"));
        assert!(schema.contains("Age"));
        assert!(schema.contains("integer"));
        assert!(schema.contains("2 rows"));
    }

    #[test]
    fn test_infer_table_schema_empty() {
        let content = "| Name | Age |\n|------|-----|\n";
        let schema = infer_table_schema(content).unwrap();
        assert!(schema.contains("empty"));
    }

    #[test]
    fn test_infer_table_schema_float_bool() {
        let content = "| Price | Active |\n|-------|--------|\n| 12.5 | true |\n| 3.0 | false |\n";
        let schema = infer_table_schema(content).unwrap();
        assert!(schema.contains("float"));
        assert!(schema.contains("boolean"));
    }
}

#[cfg(test)]
mod extract_document_title_tests {
    use super::*;
    use crate::ingest::document_tree::{DocumentMeta, DocumentNode};

    fn doc_with_section(title: &str) -> DocumentNode {
        DocumentNode::Document {
            children: vec![DocumentNode::Section {
                title: title.to_string(),
                level: 1,
                children: vec![],
                id: None,
            }],
            meta: DocumentMeta::new("test", "text/plain", 0),
        }
    }

    #[test]
    fn extracts_first_section_title() {
        let doc = doc_with_section("My Book");
        assert_eq!(extract_document_title(&doc), Some("My Book".to_string()));
    }

    #[test]
    fn returns_none_when_no_sections() {
        let doc = DocumentNode::Document {
            children: vec![DocumentNode::Paragraph {
                text: "Hi".into(),
                id: None,
            }],
            meta: DocumentMeta::new("test", "text/plain", 0),
        };
        assert_eq!(extract_document_title(&doc), None);
    }

    #[test]
    fn skips_empty_title_sections() {
        let doc = DocumentNode::Document {
            children: vec![
                DocumentNode::Section {
                    title: "".into(),
                    level: 1,
                    children: vec![],
                    id: None,
                },
                DocumentNode::Section {
                    title: "Real Title".into(),
                    level: 1,
                    children: vec![],
                    id: None,
                },
            ],
            meta: DocumentMeta::new("test", "text/plain", 0),
        };
        assert_eq!(extract_document_title(&doc), Some("Real Title".to_string()));
    }

    #[test]
    fn trims_whitespace_only_titles() {
        let doc = doc_with_section("   ");
        assert_eq!(extract_document_title(&doc), None);
    }

    #[test]
    fn returns_none_for_non_document_node() {
        let node = DocumentNode::Paragraph {
            text: "Hi".into(),
            id: None,
        };
        assert_eq!(extract_document_title(&node), None);
    }
}

#[cfg(test)]
mod relation_wiring_tests {
    use super::*;
    use crate::llm::{
        ClientStatus, ConversationAnalysis, DeduplicationResult, DetailedFactExtraction,
        EntityExtraction, ImportanceScore, KeywordExtraction, LanguageDetection,
        MemoryClassification, MemoryEnhancement, StructuredFactExtraction, SummaryResult,
    };
    use async_trait::async_trait;
    use std::sync::Mutex;

    fn make_embedding(val: f32) -> Vec<f32> {
        vec![val; 384]
    }

    fn make_l0(content: &str, id: &str) -> Memory {
        let mut mem = Memory {
            id: id.to_string(),
            content: Some(content.to_string()),
            content_meta: ContentMeta::default(),
            embedding: make_embedding(1.0),
            metadata: MemoryMetadata::new().with_layer(LayerInfo::raw_content()),
            ..Default::default()
        };
        if let Ok(_uuid) = uuid::Uuid::parse_str(id) {
            mem.metadata.hash = format!("{:x}", sha2::Sha256::digest(content));
            mem.content_meta.checksum = Some(mem.metadata.hash.clone());
        }
        mem
    }

    fn make_rel(source: &str, relation: &str, target: &str, strength: Option<f32>) -> Relation {
        Relation {
            source: source.to_string(),
            relation: relation.to_string(),
            target: target.to_string(),
            strength,
        }
    }

    type Store = Arc<Mutex<HashMap<String, Memory>>>;

    #[derive(Clone)]
    struct TestVectorStore {
        store: Store,
    }

    impl TestVectorStore {
        fn shared(store: Store) -> Self {
            Self { store }
        }
    }

    #[async_trait]
    impl VectorStore for TestVectorStore {
        async fn insert(&self, memory: &Memory) -> Result<()> {
            self.store.lock().unwrap().insert(memory.id.clone(), memory.clone());
            Ok(())
        }
        async fn search(&self, _: &[f32], _: &Filters, _: usize) -> Result<Vec<crate::types::ScoredMemory>> {
            Ok(vec![])
        }
        async fn search_with_threshold(&self, _: &[f32], _: &Filters, _: usize, _: Option<f32>) -> Result<Vec<crate::types::ScoredMemory>> {
            Ok(vec![])
        }
        async fn update(&self, memory: &Memory) -> Result<()> {
            self.store.lock().unwrap().insert(memory.id.clone(), memory.clone());
            Ok(())
        }
        async fn delete(&self, id: &str) -> Result<()> {
            self.store.lock().unwrap().remove(id);
            Ok(())
        }
        async fn get(&self, id: &str) -> Result<Option<Memory>> {
            Ok(self.store.lock().unwrap().get(id).cloned())
        }
        async fn list(&self, _: &Filters, _: Option<usize>) -> Result<Vec<Memory>> {
            Ok(self.store.lock().unwrap().values().cloned().collect())
        }
        async fn count(&self) -> Result<usize> {
            Ok(self.store.lock().unwrap().len())
        }
        async fn health_check(&self) -> Result<bool> {
            Ok(true)
        }
    }

    #[derive(Clone)]
    struct ControllableLLM {
        response: Arc<Mutex<String>>,
        should_error: Arc<Mutex<bool>>,
    }

    impl ControllableLLM {
        fn new(response: &str) -> Self {
            Self {
                response: Arc::new(Mutex::new(response.to_string())),
                should_error: Arc::new(Mutex::new(false)),
            }
        }
        fn set_error(&self, err: bool) {
            *self.should_error.lock().unwrap() = err;
        }
    }

    #[async_trait]
    impl LLMClient for ControllableLLM {
        async fn complete(&self, _prompt: &str) -> Result<String> {
            if *self.should_error.lock().unwrap() {
                return Err(MemoryError::LLM("test error".into()));
            }
            Ok(self.response.lock().unwrap().clone())
        }
        async fn complete_with_grammar(&self, _: &str, _: &str) -> Result<String> {
            Ok("{}".into())
        }
        async fn embed(&self, text: &str) -> Result<Vec<f32>> {
            Ok(make_embedding(text.len() as f32))
        }
        async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| make_embedding(t.len() as f32)).collect())
        }
        async fn extract_keywords(&self, _: &str) -> Result<Vec<String>> { Ok(vec![]) }
        async fn summarize(&self, _: &str, _: Option<usize>) -> Result<String> { Ok("".into()) }
        async fn health_check(&self) -> Result<bool> { Ok(true) }
        async fn extract_structured_facts(&self, _: &str) -> Result<StructuredFactExtraction> {
            Ok(StructuredFactExtraction { facts: vec![] })
        }
        async fn extract_detailed_facts(&self, _: &str) -> Result<DetailedFactExtraction> {
            Ok(DetailedFactExtraction { facts: vec![] })
        }
        async fn extract_keywords_structured(&self, _: &str) -> Result<KeywordExtraction> {
            Ok(KeywordExtraction { keywords: vec![] })
        }
        async fn classify_memory(&self, _: &str) -> Result<MemoryClassification> {
            Ok(MemoryClassification { memory_type: "Factual".into(), confidence: 0.9, reasoning: "test".into() })
        }
        async fn score_importance(&self, _: &str) -> Result<ImportanceScore> {
            Ok(ImportanceScore { score: 0.5, reasoning: "test".into() })
        }
        async fn check_duplicates(&self, _: &str) -> Result<DeduplicationResult> {
            Ok(DeduplicationResult { is_duplicate: false, similarity_score: 0.0, original_memory_id: None })
        }
        async fn generate_summary(&self, _: &str) -> Result<SummaryResult> {
            Ok(SummaryResult { summary: "".into(), key_points: vec![] })
        }
        async fn detect_language(&self, _: &str) -> Result<LanguageDetection> {
            Ok(LanguageDetection { language: "English".into(), confidence: 0.95 })
        }
        async fn extract_entities(&self, _: &str) -> Result<EntityExtraction> {
            Ok(EntityExtraction { entities: vec![] })
        }
        async fn analyze_conversation(&self, _: &str) -> Result<ConversationAnalysis> {
            Ok(ConversationAnalysis { topics: vec![], sentiment: "neutral".into(), user_intent: "informational".into(), key_information: vec![] })
        }
        async fn extract_metadata_enrichment(&self, _: &str) -> Result<crate::llm::MetadataEnrichment> {
            Ok(crate::llm::MetadataEnrichment { summary: "".into(), keywords: vec![] })
        }
        async fn extract_metadata_enrichment_batch(&self, t: &[String]) -> Result<Vec<Result<crate::llm::MetadataEnrichment>>> {
            Ok(t.iter().map(|_| Ok(crate::llm::MetadataEnrichment { summary: "".into(), keywords: vec![] })).collect())
        }
        async fn complete_batch(&self, prompts: &[String]) -> Result<Vec<Result<String>>> {
            Ok(prompts.iter().map(|p| Ok(p.to_string())).collect())
        }
        fn get_status(&self) -> ClientStatus {
            ClientStatus { backend: "test".into(), state: "ready".into(), llm_model: "test".into(), embedding_model: "test".into(), llm_available: true, embedding_available: true, last_llm_success: None, last_embedding_success: None, last_error: None, total_llm_calls: 0, total_embedding_calls: 0, total_prompt_tokens: 0, total_completion_tokens: 0, details: HashMap::new() }
        }
        fn batch_config(&self) -> (usize, u32) { (10, 4096) }
        async fn enhance_memory_unified(&self, _: &str) -> Result<MemoryEnhancement> {
            Ok(MemoryEnhancement { memory_type: "Semantic".into(), summary: String::new(), keywords: vec![], entities: vec![], topics: vec![] })
        }
        async fn describe_image(&self, _: &[u8], _: &str) -> Result<String> {
            Err(MemoryError::LLM("test: vision not available".into()))
        }
    }

    fn make_config() -> MemoryConfig {
        MemoryConfig {
            enable_abstraction: false,
            auto_metadata_analysis: false,
            llm_importance_scoring: false,
            skip_duplicates: false,
            llm_relation_validation: false,
            ..Default::default()
        }
    }

    fn make_service(
        store: Store,
        llm: &ControllableLLM,
        config: &MemoryConfig,
    ) -> IngestionService {
        let vs: Box<dyn VectorStore + Send + Sync> = Box::new(TestVectorStore::shared(Arc::clone(&store)));
        let vs2: Box<dyn VectorStore + Send + Sync> = Box::new(TestVectorStore::shared(Arc::clone(&store)));

        let priority_llm = Arc::new(PriorityLLMClient::new(
            Box::new(llm.clone()),
            2,
            2,
        ));
        let config_arc = Arc::new(config.clone());
        let cache = Arc::new(CacheService::new(priority_llm.clone(), None));
        let search = Arc::new(SearchService::new(vs2, priority_llm.clone(), config_arc.clone(), cache.clone()));

        IngestionService::new(
            vs,
            priority_llm,
            Box::new(llm.clone()),
            config_arc,
            cache,
            search,
            None,
        )
    }

    // === wire_relation tests ===

    #[tokio::test]
    async fn wire_relation_adds_bidirectional() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new("{}");
        let svc = make_service(Arc::clone(&store), &llm, &make_config());

        let target_id = uuid::Uuid::new_v4();
        let source_id = uuid::Uuid::new_v4();
        let mut source = make_l0("source content", &source_id.to_string());
        let target = make_l0("target content", &target_id.to_string());

        store.lock().unwrap().insert(target.id.clone(), target.clone());

        svc.wire_relation(&mut source, &target, "part_of", Some(0.9))
            .await
            .unwrap();

        assert!(source.has_relation_to("part_of", &target_id));
        let updated_target = store.lock().unwrap().get(&target.id).cloned().unwrap();
        assert!(updated_target.has_relation_to("has_part", &source_id));
    }

    #[tokio::test]
    async fn wire_relation_skips_invalid_target_uuid() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new("{}");
        let svc = make_service(Arc::clone(&store), &llm, &make_config());

        let source_id = uuid::Uuid::new_v4();
        let mut source = make_l0("source", &source_id.to_string());
        let target = make_l0("target", "not-a-uuid");

        let result = svc.wire_relation(&mut source, &target, "part_of", None).await;
        assert!(result.is_ok());
        assert!(source.relations.is_empty());
    }

    #[tokio::test]
    async fn wire_relation_no_reverse_for_unknown_relation() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new("{}");
        let svc = make_service(Arc::clone(&store), &llm, &make_config());

        let target_id = uuid::Uuid::new_v4().to_string();
        let source_id = uuid::Uuid::new_v4().to_string();
        let mut source = make_l0("source", &source_id);
        let target = make_l0("target", &target_id);

        store.lock().unwrap().insert(target.id.clone(), target.clone());

        svc.wire_relation(&mut source, &target, "custom_relation", Some(0.7))
            .await
            .unwrap();

        let target_uuid = uuid::Uuid::parse_str(&target_id).unwrap();
        assert!(source.has_relation_to("custom_relation", &target_uuid));
        assert!(!source.has_relation_to("has_part", &target_uuid));
    }

    // === wire_explicit_relations tests ===

    #[tokio::test]
    async fn wire_explicit_relations_empty_returns_zero() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new("{}");
        let svc = make_service(Arc::clone(&store), &llm, &make_config());

        let mut mem = make_l0("hello", &uuid::Uuid::new_v4().to_string());
        let count = svc.wire_explicit_relations(&mut mem, "default").await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn wire_explicit_relations_resolves_existing_target() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new("{}");
        let svc = make_service(Arc::clone(&store), &llm, &make_config());

        let source_id = uuid::Uuid::new_v4();
        let target_id = uuid::Uuid::new_v4();

        let mut source = make_l0("I am part of project X", &source_id.to_string());
        let target = make_l0("Project X documentation", &target_id.to_string());

        store.lock().unwrap().insert(target.id.clone(), target.clone());

        source.metadata.relations.push(make_rel(
            &source_id.to_string(),
            "part_of",
            &target_id.to_string(),
            Some(0.9),
        ));

        let count = svc.wire_explicit_relations(&mut source, "default").await.unwrap();
        assert_eq!(count, 1);
        assert!(source.has_relation_to("part_of", &target_id));

        let updated = store.lock().unwrap().get(&target.id).cloned().unwrap();
        assert!(updated.has_relation_to("has_part", &source_id));
    }

    #[tokio::test]
    async fn wire_explicit_relations_target_not_found_queues_pending() {
        let tmp = tempfile::tempdir().unwrap();
        let wal_path = tmp.path().join("test_wal.db");
        let wal = Arc::new(PendingWal::open(&wal_path).unwrap());

        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new("{}");
        let svc = make_service(Arc::clone(&store), &llm, &make_config());
        svc.set_pending_wal(Arc::clone(&wal));

        let source_id = uuid::Uuid::new_v4();
        let target_id = uuid::Uuid::new_v4();

        let mut source = make_l0("I need project Z", &source_id.to_string());
        source.metadata.relations.push(make_rel(
            &source_id.to_string(),
            "part_of",
            &target_id.to_string(),
            Some(0.8),
        ));

        let count = svc.wire_explicit_relations(&mut source, "default").await.unwrap();
        assert_eq!(count, 0);

        let pending = wal.load_pending_for_target(&target_id).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].source_id, source_id);
        assert_eq!(pending[0].relation, "part_of");
    }

    #[tokio::test]
    async fn wire_explicit_relations_mixed_found_and_queued() {
        let tmp = tempfile::tempdir().unwrap();
        let wal_path = tmp.path().join("test_wal.db");
        let wal = Arc::new(PendingWal::open(&wal_path).unwrap());

        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new("{}");
        let svc = make_service(Arc::clone(&store), &llm, &make_config());
        svc.set_pending_wal(Arc::clone(&wal));

        let source_id = uuid::Uuid::new_v4();
        let found_id = uuid::Uuid::new_v4();
        let missing_id = uuid::Uuid::new_v4();

        let found_target = make_l0("found target", &found_id.to_string());
        store.lock().unwrap().insert(found_target.id.clone(), found_target.clone());

        let mut source = make_l0("multi relation source", &source_id.to_string());
        source.metadata.relations.push(make_rel(&source_id.to_string(), "part_of", &found_id.to_string(), Some(0.9)));
        source.metadata.relations.push(make_rel(&source_id.to_string(), "references", &missing_id.to_string(), Some(0.5)));

        let count = svc.wire_explicit_relations(&mut source, "default").await.unwrap();
        assert_eq!(count, 1);
        assert!(source.has_relation_to("part_of", &found_id));

        let pending = wal.load_pending_for_target(&missing_id).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].relation, "references");
    }

    #[tokio::test]
    async fn wire_explicit_relations_invalid_target_uuid_skipped() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new("{}");
        let svc = make_service(Arc::clone(&store), &llm, &make_config());

        let source_id = uuid::Uuid::new_v4();
        let mut source = make_l0("source", &source_id.to_string());
        source.metadata.relations.push(make_rel(
            &source_id.to_string(),
            "part_of",
            "not-a-uuid",
            None,
        ));

        let count = svc.wire_explicit_relations(&mut source, "default").await.unwrap();
        assert_eq!(count, 0);
    }

    // === resolve_pending_relations_for tests ===

    #[tokio::test]
    async fn resolve_pending_relations_for_no_pending_returns_zero() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new("{}");
        let svc = make_service(Arc::clone(&store), &llm, &make_config());

        let mem = make_l0("new memory", &uuid::Uuid::new_v4().to_string());
        let count = svc.resolve_pending_relations_for(&mem, "default").await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn resolve_pending_relations_for_resolves_queued() {
        let tmp = tempfile::tempdir().unwrap();
        let wal_path = tmp.path().join("test_wal.db");
        let wal = Arc::new(PendingWal::open(&wal_path).unwrap());

        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new("{}");
        let svc = make_service(Arc::clone(&store), &llm, &make_config());
        svc.set_pending_wal(Arc::clone(&wal));

        let source_id = uuid::Uuid::new_v4();
        let target_id = uuid::Uuid::new_v4();

        let source = make_l0("source content", &source_id.to_string());
        store.lock().unwrap().insert(source.id.clone(), source.clone());

        let entry = PendingRelationEntry {
            source_id,
            target_id,
            relation: "part_of".into(),
            strength: Some(0.9),
            bank_name: "default".into(),
            created_at: chrono::Utc::now(),
        };
        wal.insert_pending_relation(&entry).unwrap();

        let target = make_l0("target content", &target_id.to_string());
        let count = svc.resolve_pending_relations_for(&target, "default").await.unwrap();
        assert_eq!(count, 1);

        let updated_source = store.lock().unwrap().get(&source_id.to_string()).cloned().unwrap();
        assert!(updated_source.has_relation_to("part_of", &target_id));

        let updated_target = store.lock().unwrap().get(&target_id.to_string()).cloned().unwrap();
        assert!(updated_target.has_relation_to("has_part", &source_id));

        let remaining = wal.load_pending_for_target(&target_id).unwrap();
        assert!(remaining.is_empty());
    }

    #[tokio::test]
    async fn resolve_pending_relations_for_no_wal_returns_zero() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new("{}");
        let svc = make_service(Arc::clone(&store), &llm, &make_config());

        let mem = make_l0("new memory", &uuid::Uuid::new_v4().to_string());
        let count = svc.resolve_pending_relations_for(&mem, "default").await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn resolve_pending_relations_source_gone_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let wal_path = tmp.path().join("test_wal.db");
        let wal = Arc::new(PendingWal::open(&wal_path).unwrap());

        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new("{}");
        let svc = make_service(Arc::clone(&store), &llm, &make_config());
        svc.set_pending_wal(Arc::clone(&wal));

        let source_id = uuid::Uuid::new_v4();
        let target_id = uuid::Uuid::new_v4();

        let entry = PendingRelationEntry {
            source_id,
            target_id,
            relation: "part_of".into(),
            strength: None,
            bank_name: "default".into(),
            created_at: chrono::Utc::now(),
        };
        wal.insert_pending_relation(&entry).unwrap();

        let target = make_l0("target content", &target_id.to_string());
        let count = svc.resolve_pending_relations_for(&target, "default").await.unwrap();
        assert_eq!(count, 0);
    }

    // === validate_relation tests ===

    #[tokio::test]
    async fn validate_relation_accepts_valid_json() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new(r#"{"valid": true, "confidence": 0.85, "suggested_relation": null}"#);
        let svc = make_service(Arc::clone(&store), &llm, &make_config());

        let result = svc
            .validate_relation("I am part of Project X", "Project X is a large system", "part_of")
            .await
            .unwrap();

        assert!(result.valid);
        assert!((result.confidence - 0.85).abs() < 0.01);
        assert!(result.suggested_relation.is_none());
    }

    #[tokio::test]
    async fn validate_relation_rejects_invalid_json() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new(r#"{"valid": false, "confidence": 0.2, "suggested_relation": null}"#);
        let svc = make_service(Arc::clone(&store), &llm, &make_config());

        let result = svc
            .validate_relation("I like pizza", "The moon orbits Earth", "part_of")
            .await
            .unwrap();

        assert!(!result.valid);
    }

    #[tokio::test]
    async fn validate_relation_parses_suggested_relation() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new(r#"{"valid": true, "confidence": 0.7, "suggested_relation": "references"}"#);
        let svc = make_service(Arc::clone(&store), &llm, &make_config());

        let result = svc
            .validate_relation("see also chapter 3", "chapter 3 content", "part_of")
            .await
            .unwrap();

        assert!(result.valid);
        assert_eq!(result.suggested_relation.as_deref(), Some("references"));
    }

    #[tokio::test]
    async fn validate_relation_handles_malformed_json() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new("not used");
        llm.set_error(true);
        let svc = make_service(Arc::clone(&store), &llm, &make_config());

        let result = svc
            .validate_relation("source", "target", "part_of")
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn validate_relation_handles_missing_fields() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new(r#"{}"#);
        let svc = make_service(Arc::clone(&store), &llm, &make_config());

        let result = svc
            .validate_relation("source", "target", "part_of")
            .await
            .unwrap();

        assert!(!result.valid);
        assert_eq!(result.confidence, 0.0);
    }

    // === wire_explicit_relations with LLM validation enabled ===

    #[tokio::test]
    async fn wire_explicit_relations_llm_validation_accepts_when_valid() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new(r#"{"valid": true, "confidence": 0.9, "suggested_relation": null}"#);
        let mut config = make_config();
        config.llm_relation_validation = true;
        let svc = make_service(Arc::clone(&store), &llm, &config);

        let source_id = uuid::Uuid::new_v4();
        let target_id = uuid::Uuid::new_v4();
        let mut source = make_l0("I belong to team Alpha", &source_id.to_string());
        let target = make_l0("Team Alpha is responsible for backend", &target_id.to_string());

        store.lock().unwrap().insert(target.id.clone(), target.clone());
        source.metadata.relations.push(make_rel(&source_id.to_string(), "part_of", &target_id.to_string(), Some(0.8)));

        let count = svc.wire_explicit_relations(&mut source, "default").await.unwrap();
        assert_eq!(count, 1);
        assert!(source.has_relation_to("part_of", &target_id));
    }

    #[tokio::test]
    async fn wire_explicit_relations_llm_validation_rejects_low_confidence() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new(r#"{"valid": true, "confidence": 0.3, "suggested_relation": null}"#);
        let mut config = make_config();
        config.llm_relation_validation = true;
        let svc = make_service(Arc::clone(&store), &llm, &config);

        let source_id = uuid::Uuid::new_v4();
        let target_id = uuid::Uuid::new_v4();
        let mut source = make_l0("unrelated source", &source_id.to_string());
        let target = make_l0("unrelated target", &target_id.to_string());

        store.lock().unwrap().insert(target.id.clone(), target.clone());
        source.metadata.relations.push(make_rel(&source_id.to_string(), "part_of", &target_id.to_string(), None));

        let count = svc.wire_explicit_relations(&mut source, "default").await.unwrap();
        assert_eq!(count, 0);
        assert!(!source.has_relation_to("part_of", &target_id));
    }

    #[tokio::test]
    async fn wire_explicit_relations_llm_suggests_different_relation() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new(r#"{"valid": true, "confidence": 0.8, "suggested_relation": "references"}"#);
        let mut config = make_config();
        config.llm_relation_validation = true;
        let svc = make_service(Arc::clone(&store), &llm, &config);

        let source_id = uuid::Uuid::new_v4();
        let target_id = uuid::Uuid::new_v4();
        let mut source = make_l0("see team Alpha docs", &source_id.to_string());
        let target = make_l0("Team Alpha documentation", &target_id.to_string());

        store.lock().unwrap().insert(target.id.clone(), target.clone());
        source.metadata.relations.push(make_rel(&source_id.to_string(), "part_of", &target_id.to_string(), None));

        let count = svc.wire_explicit_relations(&mut source, "default").await.unwrap();
        assert_eq!(count, 1);
        assert!(source.has_relation_to("references", &target_id));
        assert!(!source.has_relation_to("part_of", &target_id));
    }

    #[tokio::test]
    async fn wire_explicit_relations_llm_error_wires_anyway() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new("not used");
        llm.set_error(true);
        let mut config = make_config();
        config.llm_relation_validation = true;
        let svc = make_service(Arc::clone(&store), &llm, &config);

        let source_id = uuid::Uuid::new_v4();
        let target_id = uuid::Uuid::new_v4();
        let mut source = make_l0("source", &source_id.to_string());
        let target = make_l0("target", &target_id.to_string());

        store.lock().unwrap().insert(target.id.clone(), target.clone());
        source.metadata.relations.push(make_rel(&source_id.to_string(), "part_of", &target_id.to_string(), Some(0.5)));

        let count = svc.wire_explicit_relations(&mut source, "default").await.unwrap();
        assert_eq!(count, 1);
        assert!(source.has_relation_to("part_of", &target_id));
    }

    // === set_pending_wal tests ===

    #[tokio::test]
    async fn set_pending_wal_attaches_wal() {
        let tmp = tempfile::tempdir().unwrap();
        let wal_path = tmp.path().join("test_wal.db");
        let wal = Arc::new(PendingWal::open(&wal_path).unwrap());

        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new("{}");
        let svc = make_service(Arc::clone(&store), &llm, &make_config());

        svc.set_pending_wal(Arc::clone(&wal));

        let mem = make_l0("empty", &uuid::Uuid::new_v4().to_string());
        let count = svc.resolve_pending_relations_for(&mem, "default").await.unwrap();
        assert_eq!(count, 0);
    }

    // === store_with_options integration with relations ===

    #[tokio::test]
    async fn store_with_explicit_relations_wires_them() {
        let tmp = tempfile::tempdir().unwrap();
        let wal_path = tmp.path().join("test_wal.db");
        let wal = Arc::new(PendingWal::open(&wal_path).unwrap());

        let store = Arc::new(Mutex::new(HashMap::new()));
        let llm = ControllableLLM::new(r#"{"valid": true, "confidence": 0.9, "suggested_relation": null}"#);
        let mut config = make_config();
        config.llm_relation_validation = true;
        let svc = make_service(Arc::clone(&store), &llm, &config);
        svc.set_pending_wal(Arc::clone(&wal));

        let target_id = uuid::Uuid::new_v4();
        let target = make_l0("Project Y backend system", &target_id.to_string());
        store.lock().unwrap().insert(target.id.clone(), target.clone());

        let source_id_str = uuid::Uuid::new_v4().to_string();
        let _source_id = uuid::Uuid::parse_str(&source_id_str).unwrap();

        let mut metadata = MemoryMetadata::new().with_layer(LayerInfo::raw_content());
        metadata.relations.push(make_rel(&source_id_str, "part_of", &target_id.to_string(), Some(0.9)));

        let result_id = svc.store_with_options(
            "I work on Project Y".to_string(),
            metadata,
            StoreOptions { auto_link: Some(false), ..Default::default() },
        ).await.unwrap();

        let stored = store.lock().unwrap().get(&result_id).cloned().unwrap();
        assert!(stored.has_relation_to("part_of", &target_id));
    }
}
