use std::collections::HashMap;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use std::time::Instant;

use crate::{
    config::MemoryConfig,
    error::{MemoryError, Result},
    llm::{LLMClient, LlmPriority, PriorityLLMClient},
    memory::{
        cache_service::CacheService,
        deduplication::{create_duplicate_detector, DuplicateDetector},
        extractor::{create_fact_extractor, FactExtractor},
        importance::{create_importance_evaluator, ImportanceEvaluator},
        metrics::{IngestionPhase, MetricsSink, NoopMetrics},
        search_service::SearchService,
        updater::{create_memory_updater, MemoryAction, MemoryUpdater},
    },
    types::{
        ContentMeta, Filters, LayerInfo, Memory, MemoryEvent, MemoryMetadata, MemoryResult,
        Message, Relation, RelationMeta,
    },
    vector_store::VectorStore,
};

/// Options for storing memory
#[derive(Debug, Clone)]
pub struct StoreOptions {
    pub deduplicate: Option<bool>,
    pub enhance: Option<bool>,
    pub merge: Option<bool>,
    pub llm_priority: LlmPriority,
    /// Whether to auto-link to semantically similar existing memories.
    /// None = use config default (auto_link_threshold > 0.0)
    pub auto_link: Option<bool>,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self {
            deduplicate: None,
            enhance: None,
            merge: None,
            llm_priority: LlmPriority::Background,
            auto_link: None,
        }
    }
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
    duplicate_detector: Box<dyn DuplicateDetector + 'static>,
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
            config.auto_enhance,
            Some(0.5),
        );
        let duplicate_detector = create_duplicate_detector(
            dyn_clone::clone_box(vector_store.as_ref()),
            dyn_clone::clone_box(downstream_llm.as_ref()),
            config.auto_enhance,
            config.similarity_threshold,
            config.merge_threshold,
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
            duplicate_detector,
            metrics: metrics.unwrap_or_else(|| Arc::new(NoopMetrics)),
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
    pub async fn extract_metadata_enrichment(&self, text: &str) -> Result<crate::memory::extractor::ChunkMetadata> {
        let results = self
            .fact_extractor
            .extract_metadata_enrichment(&[text.to_string()])
            .await?;
        results.into_iter().next().ok_or_else(|| {
            MemoryError::LLM("No metadata enrichment returned".to_string())
        })
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
    async fn check_duplicate(&self, content: &str, filters: &Filters, llm_priority: LlmPriority) -> Result<Option<Memory>> {
        let hash = Self::generate_hash(content);
        let start = Instant::now();
        let query_embedding = self.cache.cached_embed(content, llm_priority).await?;
        self.metrics.record_ingestion_timing(IngestionPhase::DedupEmbed, start.elapsed());

        let start = Instant::now();
        let candidates = self
            .vector_store
            .search_with_threshold(&query_embedding, filters, 5, Some(0.5))
            .await?;
        self.metrics.record_ingestion_timing(IngestionPhase::DedupSearch, start.elapsed());

        for scored in candidates {
            let memory = scored.memory;
            if memory.metadata.hash == hash {
                if memory.content.as_ref().is_none_or(|c| c.trim().is_empty()) {
                    tracing::warn!("Found duplicate memory {} with empty content, skipping", memory.id);
                    continue;
                }
                tracing::debug!("Found duplicate memory with ID: {}", memory.id);
                return Ok(Some(memory));
            }
        }

        Ok(None)
    }

    /// Enhance memory content with LLM-generated metadata
    async fn enhance_memory(&self, memory: &mut Memory, merge: bool, llm_priority: LlmPriority) -> Result<()> {
        let content = match &memory.content {
            Some(c) => c,
            None => return Ok(()),
        };

        let prompt = crate::memory::prompts::UNIFIED_MEMORY_ENHANCEMENT_PROMPT
            .replace("{{text}}", content);

        let start = Instant::now();
        let res = {
            let _guard = self.llm.acquire(llm_priority).await;
            self.llm.inner().enhance_memory_unified(&prompt).await
        };
        self.metrics.record_ingestion_timing(IngestionPhase::MemoryEnhance, start.elapsed());
        match res {
            Ok(enhancement) => {
                if !enhancement.keywords.is_empty() && !memory.metadata.custom.contains_key("keywords") {
                    memory.metadata.custom.insert(
                        "keywords".to_string(),
                        serde_json::Value::Array(enhancement.keywords.into_iter().map(serde_json::Value::String).collect()),
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
                tracing::debug!("Unified memory enhancement failed, skipping enhancement: {}", e);
            }
        };

        let start = Instant::now();
        if let Ok(importance) = self.importance_evaluator.evaluate_importance(memory).await {
            memory.metadata.importance_score = memory.metadata.importance_score.max(importance);
        }
        self.metrics.record_ingestion_timing(IngestionPhase::ImportanceScore, start.elapsed());

        if merge
            && let Ok(duplicates) = self.duplicate_detector.detect_duplicates(memory).await
            && !duplicates.is_empty()
        {
            let mut all_memories = vec![memory.clone()];
            all_memories.extend(duplicates);
            if let Ok(merged_memory) = self.duplicate_detector.merge_memories(&all_memories).await {
                *memory = merged_memory;
                for duplicate in &all_memories[1..] {
                    let _ = self.vector_store.delete(&duplicate.id).await;
                }
            }
        }

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
        self.metrics.record_ingestion_timing(IngestionPhase::ContentEmbed, start.elapsed());
        let hash = Self::generate_hash(&content);

        let mut memory = Memory::with_content(
            content,
            embedding,
            MemoryMetadata {
                hash,
                ..metadata
            },
        );

        let enhance = options.enhance.unwrap_or(self.config.auto_enhance);
        if enhance {
            let merge = options.merge.unwrap_or(true);
            self.enhance_memory(&mut memory, merge, options.llm_priority).await?;
        }

        Ok(memory)
    }

    /// Create a new memory from content and metadata
    pub async fn create_memory(&self, content: String, metadata: MemoryMetadata) -> Result<Memory> {
        self.create_memory_with_options(content, metadata, &StoreOptions::default()).await
    }

    /// Store a memory in the vector store
    pub async fn store(&self, content: String, metadata: MemoryMetadata) -> Result<String> {
        self.store_with_options(content, metadata, StoreOptions::default()).await
    }

    /// Store a memory with Interactive LLM priority (for user-facing store operations).
    /// This ensures the store doesn't get starved by background abstraction pipeline work.
    pub async fn store_interactive(&self, content: String, metadata: MemoryMetadata) -> Result<String> {
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
            return Err(MemoryError::Validation("Content cannot be empty".to_string()));
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
        self.metrics.record_ingestion_timing(IngestionPhase::MemoryCountCheck, start.elapsed());
        if current_count >= self.config.max_memories {
            return Err(MemoryError::Validation(format!(
                "Memory store is full ({}/{} memories). Delete old memories or increase max_memories in config.",
                current_count, self.config.max_memories,
            )));
        }

        let deduplicate = options.deduplicate.unwrap_or(self.config.deduplicate);
        if deduplicate {
            let filters = Filters::for_user_scope(
                metadata.user_id.clone(),
                metadata.agent_id.clone(),
                metadata.run_id.clone(),
                metadata.actor_id.clone(),
            );
            if let Some(existing) = self.check_duplicate(&content, &filters, options.llm_priority).await? {
                // Dedup embed + search already recorded individually in check_duplicate
                if existing.content.as_ref().is_none_or(|c| c.trim().is_empty()) {
                    tracing::warn!("Existing memory {} has empty content, creating new memory instead", existing.id);
                } else {
                    tracing::info!("Duplicate memory found, returning existing ID: {}", existing.id);
                    return Ok(existing.id);
                }
            }
        }

        let mut memory = self
            .create_memory_with_options(content, metadata, &options)
            .await?;
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
            self.metrics.record_ingestion_timing(IngestionPhase::AuxEmbed, start.elapsed());

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
        let do_auto_link = options.auto_link.unwrap_or(self.config.auto_link_threshold > 0.0);
        if do_auto_link {
            let start = Instant::now();
            let linked = self
                .auto_link_memory(
                    &mut memory,
                    self.config.auto_link_threshold,
                    self.config.auto_link_max_relations,
                )
                .await
                .unwrap_or(0);
            self.metrics.record_ingestion_timing(IngestionPhase::AutoLinkSearch, start.elapsed());
            if linked > 0 {
                tracing::info!("Auto-linked memory {} to {} similar memories", memory_id, linked);
            }
        }

        let start = Instant::now();
        self.vector_store.insert(&memory).await?;
        self.metrics.record_ingestion_timing(IngestionPhase::VsInsert, start.elapsed());

        let start = Instant::now();
        self.search.insert_layer(memory.metadata.layer.level).await;
        self.metrics.record_ingestion_timing(IngestionPhase::LayerManifestUpdate, start.elapsed());

        // Chunk long L0 memories for better retrieval coverage.
        // The embedding model truncates at ~256 tokens, so long sessions
        // lose most of their content. Chunking gives each segment its own vector.
        self.store_content_chunks(&memory, options.llm_priority).await?;

        tracing::info!(
            "Stored new memory with ID: {} (content length: {}, contexts: {}, relations: {})",
            memory_id,
            memory.content.as_ref().map_or(0, |c| c.len()),
            memory.metadata.context.len(),
            memory.metadata.relations.len(),
        );
        Ok(memory_id)
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
    ) -> Result<usize> {
        if threshold <= 0.0 || max_links == 0 {
            return Ok(0);
        }

        let filters = Filters::for_user_scope(
            memory.metadata.user_id.clone(),
            memory.metadata.agent_id.clone(),
            memory.metadata.run_id.clone(),
            memory.metadata.actor_id.clone(),
        );

        let scored = self
            .vector_store
            .search(&memory.embedding, &filters, max_links + 1)
            .await?;

        let mut linked = 0;
        for s in scored {
            if s.memory.id == memory.id {
                continue;
            }
            if s.score < threshold {
                break;
            }
            let target_id = match uuid::Uuid::parse_str(&s.memory.id) {
                Ok(id) => id,
                Err(_) => continue,
            };
            let meta = RelationMeta::new("auto_link").with_confidence(s.score);
            memory.add_relation("references", vec![target_id], Some(s.score), meta);
            memory.metadata.relations.push(Relation {
                source: memory.id.clone(),
                relation: "references".into(),
                target: s.memory.id.clone(),
                strength: Some(s.score),
            });
            linked += 1;
        }
        Ok(linked)
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

        let parent_id =
            uuid::Uuid::parse_str(&parent.id).ok();

        let start = Instant::now();
        let embeddings: Vec<Vec<f32>> = {
            let _guard = self.llm.acquire(llm_priority).await;
            self.llm.inner().embed_batch(&chunks).await?
        };
        self.metrics.record_ingestion_timing(IngestionPhase::ContentChunkEmbed, start.elapsed());

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
            chunk_memory.id = chunk_id.to_string();
            chunk_memory.created_at = parent.created_at;
            chunk_memory.updated_at = chrono::Utc::now();

            let _start = Instant::now();
            self.vector_store.insert(&chunk_memory).await?;
            self.metrics.record_ingestion_timing(IngestionPhase::VsInsert, _start.elapsed());

            let _start = Instant::now();
            self.search.insert_layer(0).await;
            self.metrics.record_ingestion_timing(IngestionPhase::LayerManifestUpdate, _start.elapsed());
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
        if messages.is_empty() {
            return Ok(vec![]);
        }

        let extracted_facts = self.fact_extractor.extract_facts(messages).await?;
        let mut final_extracted_facts = extracted_facts;

        if final_extracted_facts.is_empty() {
            let user_messages: Vec<_> = messages.iter().filter(|msg| msg.role == "user").cloned().collect();

            if !user_messages.is_empty()
                && let Ok(user_facts) = self.fact_extractor.extract_user_facts(&user_messages).await
                && !user_facts.is_empty()
            {
                final_extracted_facts = user_facts;
            }

            if final_extracted_facts.is_empty() {
                let mut single_message_facts = Vec::new();
                for message in messages {
                    if let Ok(mut facts) = self.fact_extractor.extract_facts_from_text(&message.content).await {
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
                    let memory_id = self.store(user_content.clone(), metadata).await?;
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
                .search_with_threshold(&query_embedding, &filters, 5, self.config.search_similarity_threshold)
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
                        let memory_id = self.store(content.clone(), metadata_with_keywords).await?;
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
                    MemoryAction::Merge { target_id, source_ids, merged_content } => {
                        let _ = self.update(target_id, Some(merged_content.clone()), None).await;
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
            .vector_store.get(id)
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
            if self.config.auto_enhance {
                self.enhance_memory(&mut memory, true, LlmPriority::Background).await?;
            }
        }

        if let Some(new_relations) = relations {
            for new_rel in new_relations {
                if !memory.metadata.relations.iter().any(|r| r.relation == new_rel.relation && r.target == new_rel.target) {
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
        content: String,
        content_encoding: Option<String>,
        format_hint: Option<String>,
        file_name: Option<String>,
        auto_link: Option<bool>,
        generate_abstractions: Option<bool>,
        max_chunk_size: Option<usize>,
        user_metadata: Option<MemoryMetadata>,
    ) -> Result<crate::ingest::feedback::IngestResult> {
        use crate::ingest::feedback::IngestResult;
        use crate::ingest::format_detect::{InputFormat, detect_format};

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
            InputFormat::Pdf | InputFormat::Word | InputFormat::Excel
                | InputFormat::ImagePng | InputFormat::ImageJpeg
                | InputFormat::ImageGif | InputFormat::ImageWebp
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
                    return Ok(IngestResult::new(session_id, fmt.name(), fmt.mime(), byte_size)
                        .with_issue(crate::ingest::feedback::IngestIssue::blocking(
                            format!("Binary parse failure: {}", e),
                            "Try a different format or file.",
                        )));
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

            match crate::ingest::parsers::parse(
                &content_str, fmt, file_name.as_deref(),
            ) {
                Ok(result) => result,
                Err(parse_err) => {
                    if self.config.llm_fallback_parsing {
                        let advisor = crate::llm::LLMStrategyAdvisor::new(self.llm.inner());
                        match advisor.fallback_parse(&content_str, fmt.name()).await {
                            Ok(fallback_result) => {
                                tracing::info!("LLM fallback parser succeeded for format {}", fmt.name());
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

        let mut result = IngestResult::new(session_id, fmt.name(), fmt.mime(), byte_size);
        for warning in &doc_meta.warnings {
            result.warnings.push(warning.clone());
        }

        let chunking = crate::ingest::chunker::chunk_document(&doc, max_chunk);

        let user_id = user_metadata
            .as_ref()
            .and_then(|m| m.user_id.clone());
        let agent_id = user_metadata
            .as_ref()
            .and_then(|m| m.agent_id.clone());

        let mut chunk_ids: Vec<String> = Vec::new();
        let mut chunk_memory_ids: Vec<String> = Vec::new();

        for chunk in &chunking.chunks {
            let chunk_id = uuid::Uuid::new_v4().to_string();
            let mut meta = if let Some(ref base) = user_metadata {
                let mut m = base.clone();
                m.layer = LayerInfo::raw_content();
                m
            } else {
                let mut m = MemoryMetadata::new()
                    .with_layer(LayerInfo::raw_content());
                if let Some(ref uid) = user_id {
                    m = m.with_user_id(uid.clone());
                }
                if let Some(ref aid) = agent_id {
                    m = m.with_agent_id(aid.clone());
                }
                m
            };

            meta.custom.insert("chunk_order".into(), serde_json::json!(chunk.order));
            meta.custom.insert("node_type".into(), serde_json::json!(&chunk.node_type));
            meta.custom.insert("ingest_session".into(), serde_json::json!(&result.session_id));

            let do_auto_link = auto_link.unwrap_or(true);
            let options = StoreOptions {
                llm_priority: LlmPriority::Interactive,
                auto_link: Some(do_auto_link),
                ..StoreOptions::default()
            };

            let memory_id = self.store_with_options(chunk.content.clone(), meta.clone(), options).await?;

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

                result.relations.push(crate::ingest::feedback::RelationInfo {
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
                && let Some(schema_desc) = infer_table_schema(&chunk.content) {
                    let mem_id = &chunk_memory_ids[i];
                    let l1_content = format!(
                        "[L1 Table Schema] {}\n\nSource table L0 chunk: {}",
                        schema_desc, mem_id
                    );
                    let l1_meta = MemoryMetadata::new()
                        .with_layer(LayerInfo::structural());
                    let result_id = self.store_with_options(
                        l1_content,
                        l1_meta,
                        StoreOptions {
                            llm_priority: LlmPriority::Interactive,
                            auto_link: Some(false),
                            ..StoreOptions::default()
                        },
                    ).await?;

                    if let Some(mut source) = self.vector_store.get(mem_id).await? {
                        source.metadata.relations.push(Relation {
                            source: result_id.clone(),
                            relation: "l1_of".into(),
                            target: mem_id.clone(),
                            strength: Some(0.9),
                        });
                        self.vector_store.update(&source).await?;
                    }

                    result.l1_abstractions.push(crate::ingest::feedback::AbstractionInfo {
                        id: Some(uuid::Uuid::new_v4().to_string()),
                        memory_id: Some(result_id),
                        abstraction_type: "table_schema".into(),
                        source_chunk_ids: vec![chunk_ids[i].clone()],
                        layer: 1,
                        content_preview: schema_desc.chars().take(80).collect(),
                    });
                }
        }

        if let Some(ref img_bytes) = image_data {
            let vision_outcome = match self.llm.inner().describe_image(img_bytes, fmt.mime()).await {
                Ok(description) => {
                    let l1_content = format!(
                        "[L1 Image Description] {}\n\nSource image L0 chunk: session {}",
                        description,
                        result.session_id
                    );
                    let l1_meta = MemoryMetadata::new()
                        .with_layer(LayerInfo::structural());
                    let result_id = self.store_with_options(
                        l1_content,
                        l1_meta,
                        StoreOptions {
                            llm_priority: LlmPriority::Interactive,
                            auto_link: Some(false),
                            ..StoreOptions::default()
                        },
                    ).await?;

                    if let Some(ref first_id) = chunk_memory_ids.first()
                        && let Some(mut source) = self.vector_store.get(first_id).await? {
                            source.metadata.relations.push(Relation {
                                source: result_id.clone(),
                                relation: "l1_of".into(),
                                target: first_id.to_string(),
                                strength: Some(0.9),
                            });
                            self.vector_store.update(&source).await?;
                        }

                    result.l1_abstractions.push(crate::ingest::feedback::AbstractionInfo {
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
                    result.warnings.push(format!("Image description failed: {}", msg));
                    let lower = msg.to_lowercase();
                    if lower.contains("not available")
                        || lower.contains("not configured")
                        || lower.contains("not supported")
                        || lower.contains("vision disabled")
                        || lower.contains("no mmproj")
                        || lower.contains("requires mmproj")
                        || lower.contains("not found")
                    {
                        crate::ingest::feedback::VisionOutcome::NotConfigured
                    } else {
                        crate::ingest::feedback::VisionOutcome::Failed
                    }
                }
            };
            result.vision_status = Some(crate::ingest::feedback::VisionStatus {
                images_ingested: 1,
                descriptions_generated: if vision_outcome == crate::ingest::feedback::VisionOutcome::Succeeded { 1 } else { 0 },
                outcome: vision_outcome,
                detail: None,
            });
        }

        if !result.issues.iter().any(|i| i.severity == crate::ingest::feedback::IssueSeverity::Blocking)
            && result.l0_chunks.is_empty()
        {
            result.warnings.push("Content parsed successfully but produced no chunks".into());
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

    let col_types: Vec<String> = (0..headers.len().min(rows.first().map(|r| r.len()).unwrap_or(0)))
        .map(|col| infer_column_type(col, &rows))
        .collect();

    let mut parts = Vec::new();
    for (i, header) in headers.iter().enumerate() {
        let col_type = col_types.get(i).map(|s| s.as_str()).unwrap_or("string");
        let non_null = rows.iter().filter(|r| r.get(i).map(|c| !c.is_empty()).unwrap_or(false)).count();
        parts.push(format!("{} ({}, {} non-null)", header, col_type, non_null));
    }

    Some(format!(
        "Table with {} columns and {} rows. Columns: {}.{}",
        headers.len(),
        rows.len(),
        parts.join("; "),
        if rows.is_empty() { " Table is empty." } else { "" }
    ))
}

fn infer_column_type(col: usize, rows: &[Vec<String>]) -> String {
    let values: Vec<String> = rows.iter()
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

    let all_floats = values.iter().all(|v| {
        v.parse::<f64>().is_ok()
            || v.replace(',', "").parse::<f64>().is_ok()
    });
    if all_floats {
        return "float".into();
    }

    let all_bools = values.iter().all(|v| {
        matches!(v.to_lowercase().as_str(), "true" | "false" | "yes" | "no" | "1" | "0")
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
    STANDARD.decode(trimmed.as_bytes())
        .map_err(|e| format!("Base64 decode error: {}", e))
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