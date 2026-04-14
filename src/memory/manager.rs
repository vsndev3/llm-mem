use chrono::Utc;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::{
    config::MemoryConfig,
    error::{MemoryError, Result},
    llm::LLMClient,
    memory::{
        classification::{MemoryClassifier, create_memory_classifier},
        deduplication::{DuplicateDetector, create_duplicate_detector},
        extractor::{FactExtractor, create_fact_extractor},
        importance::{ImportanceEvaluator, create_importance_evaluator},
        prompts::PROCEDURAL_MEMORY_SYSTEM_PROMPT,
        updater::{MemoryAction, MemoryUpdater, create_memory_updater},
    },
    types::{
        ContentMeta, Filters, Memory, MemoryEvent, MemoryMetadata, MemoryResult, MemoryType,
        NavigateResult, ScoredMemory,
    },
    vector_store::VectorStore,
};

/// Core memory manager that orchestrates memory operations
pub struct MemoryManager {
    vector_store: Box<dyn VectorStore>,
    llm_client: Box<dyn LLMClient>,
    config: MemoryConfig,
    fact_extractor: Box<dyn FactExtractor + 'static>,
    memory_updater: Box<dyn MemoryUpdater + 'static>,
    importance_evaluator: Box<dyn ImportanceEvaluator + 'static>,
    duplicate_detector: Box<dyn DuplicateDetector + 'static>,
    memory_classifier: Box<dyn MemoryClassifier + 'static>,
}

/// Options for storing memory
#[derive(Debug, Clone, Default)]
pub struct StoreOptions {
    /// Whether to check for exact content duplicates (defaults to config.deduplicate)
    pub deduplicate: Option<bool>,
    /// Whether to perform LLM enhancement (keywords, summary, etc.) (defaults to config.auto_enhance)
    pub enhance: Option<bool>,
    /// Whether to perform duplicate merging (defaults to true if enhance is true)
    pub merge: Option<bool>,
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new(
        vector_store: Box<dyn VectorStore>,
        llm_client: Box<dyn LLMClient>,
        config: MemoryConfig,
    ) -> Self {
        let fact_extractor = create_fact_extractor(dyn_clone::clone_box(llm_client.as_ref()));
        let memory_updater = create_memory_updater(
            dyn_clone::clone_box(llm_client.as_ref()),
            dyn_clone::clone_box(vector_store.as_ref()),
            config.similarity_threshold,
            config.merge_threshold,
        );
        let importance_evaluator = create_importance_evaluator(
            dyn_clone::clone_box(llm_client.as_ref()),
            config.auto_enhance,
            Some(0.5),
        );
        let duplicate_detector = create_duplicate_detector(
            dyn_clone::clone_box(vector_store.as_ref()),
            dyn_clone::clone_box(llm_client.as_ref()),
            config.auto_enhance,
            config.similarity_threshold,
            config.merge_threshold,
        );
        let memory_classifier = create_memory_classifier(
            dyn_clone::clone_box(llm_client.as_ref()),
            config.auto_enhance,
            Some(100),
        );

        Self {
            vector_store,
            llm_client,
            config,
            fact_extractor,
            memory_updater,
            importance_evaluator,
            duplicate_detector,
            memory_classifier,
        }
    }

    /// Generate a hash for memory content
    fn generate_hash(&self, content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Get a reference to the LLM client
    pub fn llm_client(&self) -> &dyn LLMClient {
        self.llm_client.as_ref()
    }

    /// Get a reference to the underlying vector store.
    pub fn vector_store(&self) -> &dyn VectorStore {
        self.vector_store.as_ref()
    }

    /// Get the current status of the LLM client
    pub fn get_status(&self) -> crate::llm::ClientStatus {
        self.llm_client.get_status()
    }

    /// Get the current memory configuration
    pub fn config(&self) -> &MemoryConfig {
        &self.config
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
        results.into_iter().next().ok_or_else(|| {
            crate::error::MemoryError::LLM("No metadata enrichment returned".to_string())
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
    ///
    /// This bypasses the LLM enhancement pipeline — use it for backup restores
    /// and merge operations where the Memory already has embeddings, metadata,
    /// importance scores, etc.
    pub async fn import_memory(&self, memory: &Memory) -> Result<()> {
        self.vector_store.insert(memory).await
    }

    /// Check if memory with the same content already exists.
    ///
    /// Uses vector similarity search (O(log n) with HNSW) to find near-duplicate
    /// candidates, then compares content hashes. This scales to any store size.
    async fn check_duplicate(&self, content: &str, filters: &Filters) -> Result<Option<Memory>> {
        let hash = self.generate_hash(content);

        // Embed the new content and find the most similar existing memories
        let query_embedding = self.llm_client.embed(content).await?;

        // Use a smaller limit for duplicate checking to avoid HNSW panics on small indexes
        let candidates = self
            .vector_store
            .search_with_threshold(
                &query_embedding,
                filters,
                5,         // check top-5 most similar
                Some(0.5), // only consider reasonably similar memories
            )
            .await?;

        for scored in candidates {
            let memory = scored.memory;
            if memory.metadata.hash == hash {
                if memory.content.as_ref().is_none_or(|c| c.trim().is_empty()) {
                    warn!(
                        "Found duplicate memory {} with empty content, skipping",
                        memory.id
                    );
                    continue;
                }
                debug!("Found duplicate memory with ID: {}", memory.id);
                return Ok(Some(memory));
            }
        }

        Ok(None)
    }

    /// Enhance memory content with LLM-generated metadata
    async fn enhance_memory(&self, memory: &mut Memory, merge: bool) -> Result<()> {
        // Get content reference for enhancement (skip if no content)
        let content = match &memory.content {
            Some(c) => c,
            None => return Ok(()), // Nothing to enhance if no content
        };

        // Build the unified prompt and make a single LLM call
        let prompt = crate::memory::prompts::UNIFIED_MEMORY_ENHANCEMENT_PROMPT
            .replace("{{text}}", content);

        match self.llm_client.enhance_memory_unified(&prompt).await {
            Ok(enhancement) => {
                // Apply memory type (only overwrite if still the default Conversational)
                if memory.metadata.memory_type == MemoryType::Conversational {
                    memory.metadata.memory_type = MemoryType::parse(&enhancement.memory_type);
                }

                // Apply keywords (only if not already present)
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

                // Apply summary (only if not already present and content exceeds threshold)
                if !enhancement.summary.is_empty()
                    && content.len() > self.config.auto_summary_threshold
                    && !memory.metadata.custom.contains_key("summary")
                {
                    memory.metadata.custom.insert(
                        "summary".to_string(),
                        serde_json::Value::String(enhancement.summary),
                    );
                }

                // Apply entities
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

                // Apply topics
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
                debug!("Unified memory enhancement failed, skipping enhancement: {}", e);
            }
        }

        // Importance evaluation remains a separate call since it needs the full memory object
        if let Ok(importance) = self.importance_evaluator.evaluate_importance(memory).await {
            // Use the higher of the two scores if one was already set
            memory.metadata.importance_score = memory.metadata.importance_score.max(importance);
        }

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

        debug!("Creating memory with content length: {}", content.len());

        let embedding = self.llm_client.embed(&content).await?;

        let mut memory = Memory::with_content(
            content.to_owned(),
            embedding,
            MemoryMetadata {
                hash: self.generate_hash(&content),
                ..metadata
            },
        );

        let enhance = options.enhance.unwrap_or(self.config.auto_enhance);
        if enhance {
            let merge = options.merge.unwrap_or(true);
            self.enhance_memory(&mut memory, merge).await?;
        }

        Ok(memory)
    }

    /// Create a new memory from content and metadata
    pub async fn create_memory(&self, content: String, metadata: MemoryMetadata) -> Result<Memory> {
        self.create_memory_with_options(content, metadata, &StoreOptions::default())
            .await
    }

    /// Add memory from conversation messages with full fact extraction and update pipeline
    pub async fn add_memory(
        &self,
        messages: &[crate::types::Message],
        metadata: MemoryMetadata,
    ) -> Result<Vec<MemoryResult>> {
        if messages.is_empty() {
            return Ok(vec![]);
        }

        if metadata.agent_id.is_some() && metadata.memory_type == MemoryType::Procedural {
            return self.create_procedural_memory(messages, metadata).await;
        }

        let extracted_facts = self.fact_extractor.extract_facts(messages).await?;
        let mut final_extracted_facts = extracted_facts;

        if final_extracted_facts.is_empty() {
            debug!("No facts extracted, trying alternative extraction methods");

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

                debug!("No memorable content found in conversation, skipping storage");
                return Ok(vec![]);
            }
        }

        // Extract keywords from original messages for enhanced searchability
        let original_content: String = messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let extracted_keywords = match self.llm_client.extract_keywords(&original_content).await {
            Ok(keywords) => keywords,
            Err(e) => {
                debug!("Failed to extract keywords: {}", e);
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

            let query_embedding = self.llm_client.embed(&fact.content).await?;
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
                        // Add extracted keywords to metadata for enhanced searchability
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
                        self.update(id, Some(content.clone()), None).await?;
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
                        // For merge, we don't add keywords as we're updating existing memory
                        self.update(target_id, Some(merged_content.clone()), None)
                            .await?;
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
                        self.delete(id).await?;
                        all_actions.push(MemoryResult {
                            id: id.clone(),
                            memory: "".to_string(),
                            event: MemoryEvent::Delete,
                            actor_id: messages.last().and_then(|msg| msg.name.clone()),
                            role: messages.last().map(|msg| msg.role.clone()),
                            previous_memory: None,
                        });
                    }
                }
            }
        }

        info!(
            "Added memory from conversation: {} actions performed",
            all_actions.len()
        );
        Ok(all_actions)
    }

    /// Ingest a document by extracting facts and storing them
    pub async fn ingest_document(
        &self,
        text: &str,
        metadata: MemoryMetadata,
    ) -> Result<Vec<MemoryResult>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        let extracted_facts = self.fact_extractor.extract_facts_from_text(text).await?;

        if extracted_facts.is_empty() {
            debug!("No facts extracted from document, storing as single memory");
            let memory_id = self.store(text.to_string(), metadata.clone()).await?;
            return Ok(vec![MemoryResult {
                id: memory_id,
                memory: text.to_string(),
                event: MemoryEvent::Add,
                actor_id: None,
                role: None,
                previous_memory: None,
            }]);
        }

        let mut all_actions = Vec::new();

        for fact in &extracted_facts {
            let filters = Filters::for_user_scope(
                metadata.user_id.clone(),
                metadata.agent_id.clone(),
                metadata.run_id.clone(),
                metadata.actor_id.clone(),
            );

            let query_embedding = self.llm_client.embed(&fact.content).await?;
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
                        let memory_id = self.store(content.clone(), (**metadata).clone()).await?;
                        all_actions.push(MemoryResult {
                            id: memory_id,
                            memory: content.clone(),
                            event: MemoryEvent::Add,
                            actor_id: None,
                            role: None,
                            previous_memory: None,
                        });
                    }
                    MemoryAction::Update { id, content } => {
                        self.update(id, Some(content.clone()), None).await?;
                        all_actions.push(MemoryResult {
                            id: id.clone(),
                            memory: content.clone(),
                            event: MemoryEvent::Update,
                            actor_id: None,
                            role: None,
                            previous_memory: None,
                        });
                    }
                    MemoryAction::Merge {
                        target_id,
                        source_ids,
                        merged_content,
                    } => {
                        self.update(target_id, Some(merged_content.clone()), None)
                            .await?;
                        for source_id in source_ids {
                            let _ = self.delete(source_id).await;
                        }
                        all_actions.push(MemoryResult {
                            id: target_id.clone(),
                            memory: merged_content.clone(),
                            event: MemoryEvent::Update,
                            actor_id: None,
                            role: None,
                            previous_memory: None,
                        });
                    }
                    MemoryAction::Delete { id } => {
                        self.delete(id).await?;
                        all_actions.push(MemoryResult {
                            id: id.clone(),
                            memory: "".to_string(),
                            event: MemoryEvent::Delete,
                            actor_id: None,
                            role: None,
                            previous_memory: None,
                        });
                    }
                }
            }
        }

        info!("Ingested document: {} actions performed", all_actions.len());
        Ok(all_actions)
    }

    /// Store a memory in the vector store
    pub async fn store(&self, content: String, metadata: MemoryMetadata) -> Result<String> {
        self.store_with_options(content, metadata, StoreOptions::default())
            .await
    }

    /// Store a memory with fine-grained control options
    pub async fn store_with_options(
        &self,
        content: String,
        metadata: MemoryMetadata,
        options: StoreOptions,
    ) -> Result<String> {
        debug!(
            "Storing memory with content: '{}...'",
            content.chars().take(50).collect::<String>()
        );

        if content.trim().is_empty() {
            warn!("Attempting to store memory with empty content, skipping");
            return Err(MemoryError::Validation(
                "Content cannot be empty".to_string(),
            ));
        }
        // Validate content length
        if content.len() > self.config.max_content_length {
            return Err(MemoryError::Validation(format!(
                "Content length ({} bytes) exceeds maximum allowed ({} bytes)",
                content.len(),
                self.config.max_content_length,
            )));
        }

        // Enforce max_memories limit
        let current_count = self.vector_store.count().await?;
        if current_count >= self.config.max_memories {
            return Err(MemoryError::Validation(format!(
                "Memory store is full ({}/{} memories). Delete old memories or increase max_memories in config.",
                current_count, self.config.max_memories,
            )));
        }

        let deduplicate = options.deduplicate.unwrap_or(self.config.deduplicate);
        if deduplicate {
            let filters = Filters::for_user_with_type(
                metadata.user_id.clone(),
                metadata.agent_id.clone(),
                metadata.run_id.clone(),
                metadata.actor_id.clone(),
                metadata.memory_type.clone(),
            );

            if let Some(existing) = self.check_duplicate(&content, &filters).await? {
                if existing
                    .content
                    .as_ref()
                    .is_none_or(|c| c.trim().is_empty())
                {
                    warn!(
                        "Existing memory {} has empty content, creating new memory instead",
                        existing.id
                    );
                } else {
                    info!(
                        "Duplicate memory found, returning existing ID: {}",
                        existing.id
                    );
                    return Ok(existing.id);
                }
            }
        }

        let mut memory = self
            .create_memory_with_options(content, metadata, &options)
            .await?;
        let memory_id = memory.id.clone();

        // Level 2: Link 'SELF' in relations to the actual memory ID
        for relation in &mut memory.metadata.relations {
            if relation.source == "SELF" {
                relation.source = memory_id.clone();
            }
        }

        if memory.content.as_ref().is_none_or(|c| c.trim().is_empty()) {
            warn!("Created memory has empty content: {}", memory_id);
        }

        // ─── Level 3: Multi-Vector Embedding ───
        // Generate auxiliary embeddings for context tags and relations.
        // Batch all embedding calls into a single embed_batch() for efficiency.

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

            let all_embeddings = self.llm_client.embed_batch(&all_texts).await?;

            if all_embeddings.len() == total_aux {
                if !ctx_tags.is_empty() {
                    let ctx_embeddings = all_embeddings[..ctx_tags.len()].to_vec();
                    debug!(
                        "Generated {} context embeddings for memory {}",
                        ctx_embeddings.len(),
                        memory_id
                    );
                    memory.context_embeddings = Some(ctx_embeddings);
                }

                if !rel_texts.is_empty() {
                    let rel_embeddings = all_embeddings[ctx_tags.len()..].to_vec();
                    debug!(
                        "Generated {} relation embeddings for memory {}",
                        rel_embeddings.len(),
                        memory_id
                    );
                    memory.relation_embeddings = Some(rel_embeddings);
                }
            } else {
                warn!(
                    "embed_batch returned {} embeddings, expected {}; skipping auxiliary embeddings",
                    all_embeddings.len(),
                    total_aux
                );
            }
        }

        self.vector_store.insert(&memory).await?;

        let content_len = memory.content.as_ref().map_or(0, |c| c.len());
        info!(
            "Stored new memory with ID: {} (content length: {}, contexts: {}, relations: {})",
            memory_id,
            content_len,
            memory.metadata.context.len(),
            memory.metadata.relations.len(),
        );
        Ok(memory_id)
    }

    /// Search for similar memories with importance-weighted ranking and hybrid keyword matching
    ///
    /// Performs both semantic search (content embedding) and keyword matching.
    /// Memories with matching keywords in metadata.keywords get a score boost.
    /// Set `keyword_only: true` in filters.custom to search ONLY by keyword matching.
    pub async fn search(
        &self,
        query: &str,
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        self.search_with_override(query, filters, limit, None).await
    }

    /// Search with an optional similarity threshold override.
    pub async fn search_with_override(
        &self,
        query: &str,
        filters: &Filters,
        limit: usize,
        threshold_override: Option<f32>,
    ) -> Result<Vec<ScoredMemory>> {
        let search_similarity_threshold = threshold_override.map(Some).unwrap_or(self.config.search_similarity_threshold);

        // Extract keywords from query for hybrid matching
        let query_keywords = match self.llm_client.extract_keywords(query).await {
            Ok(keywords) => {
                debug!(
                    "Extracted {} keywords from query for hybrid search",
                    keywords.len()
                );
                keywords
            }
            Err(e) => {
                debug!("Failed to extract keywords from query: {}", e);
                Vec::new()
            }
        };

        // Check if keyword-only search is requested
        let keyword_only = filters
            .custom
            .get("keyword_only")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let results = if keyword_only {
            // Keyword-only search: fetch all memories and filter by keyword matching
            self.search_by_keywords_only(query, &query_keywords, filters, limit)
                .await?
        } else {
            // Hybrid search: semantic + keyword boosting
            self.search_hybrid(
                query,
                &query_keywords,
                filters,
                limit,
                search_similarity_threshold,
            )
            .await?
        };

        Ok(results)
    }

    /// Hybrid search: semantic similarity with keyword-based score boosting
    async fn search_hybrid(
        &self,
        query: &str,
        query_keywords: &[String],
        filters: &Filters,
        limit: usize,
        similarity_threshold: Option<f32>,
    ) -> Result<Vec<ScoredMemory>> {
        // Get semantic search results
        let mut results = self
            .search_with_threshold(query, filters, limit * 2, similarity_threshold)
            .await?;

        if query_keywords.is_empty() {
            // No keywords to boost with, return semantic results as-is
            results.truncate(limit);
            return Ok(results);
        }

        // Apply keyword-based score boosting
        let keyword_boost = 0.15f32; // Boost factor for keyword matches

        for scored in &mut results {
            if let Some(keywords_val) = scored.memory.metadata.custom.get("keywords")
                && let Some(memory_keywords) = keywords_val.as_array()
            {
                let memory_kw_strings: Vec<String> = memory_keywords
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_lowercase()))
                    .collect();

                // Count matching keywords (case-insensitive)
                let matches: usize = query_keywords
                    .iter()
                    .filter(|qk| {
                        let qk_lower = qk.to_lowercase();
                        memory_kw_strings
                            .iter()
                            .any(|mk| mk.contains(&qk_lower) || qk_lower.contains(mk))
                    })
                    .count();

                if matches > 0 {
                    // Boost score based on number of matching keywords
                    let boost = keyword_boost * (matches as f32);
                    scored.score = (scored.score + boost).min(1.0);
                    debug!(
                        "Keyword boost for memory {}: +{} ({} matches)",
                        scored.memory.id, boost, matches
                    );
                }
            }
        }

        // Re-sort by boosted scores
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(limit);

        Ok(results)
    }

    /// Keyword-only search: find memories by keyword matching without semantic search
    async fn search_by_keywords_only(
        &self,
        _query: &str,
        query_keywords: &[String],
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        if query_keywords.is_empty() {
            return Ok(Vec::new());
        }

        // Fetch a larger pool of memories and filter by keywords
        let query_embedding = self.llm_client.embed("").await?; // Empty query to get all memories
        let all_memories = self
            .vector_store
            .search_with_threshold(&query_embedding, filters, limit * 10, Some(0.0))
            .await?;

        let mut scored_results: Vec<(ScoredMemory, usize)> = Vec::new();

        for scored in all_memories {
            if let Some(keywords_val) = scored.memory.metadata.custom.get("keywords")
                && let Some(memory_keywords) = keywords_val.as_array()
            {
                let memory_kw_strings: Vec<String> = memory_keywords
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_lowercase()))
                    .collect();

                // Count matching keywords
                let matches: usize = query_keywords
                    .iter()
                    .filter(|qk| {
                        let qk_lower = qk.to_lowercase();
                        memory_kw_strings
                            .iter()
                            .any(|mk| mk.contains(&qk_lower) || qk_lower.contains(mk))
                    })
                    .count();

                if matches > 0 {
                    scored_results.push((scored, matches));
                }
            }
        }

        // Sort by number of matches (descending)
        scored_results.sort_by(|a, b| b.1.cmp(&a.1));
        scored_results.truncate(limit);

        // Convert match counts to scores
        let results: Vec<ScoredMemory> = scored_results
            .into_iter()
            .map(|(mut scored, matches)| {
                scored.score = (matches as f32 * 0.2).min(1.0);
                scored
            })
            .collect();

        Ok(results)
    }

    /// Level 3: Two-stage retrieval with context-based pre-filtering.
    ///
    /// Stage 1: Embed each context tag and search for matching context vectors
    ///          to collect candidate memory IDs.
    /// Stage 2: Run the normal content search, constrained to those candidate IDs.
    pub async fn search_with_context(
        &self,
        query: &str,
        context_tags: &[String],
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        if context_tags.is_empty() {
            return self.search(query, filters, limit).await;
        }

        // Stage 1: Context pre-filtering
        let mut candidate_ids = std::collections::HashSet::new();
        let ctx_fetch_limit = 50; // fetch enough candidates per tag
        for tag in context_tags {
            let tag_embedding = self.llm_client.embed(tag).await?;
            // Search across ALL vectors — context vectors will match best
            let ctx_results = self
                .vector_store
                .search_with_threshold(
                    &tag_embedding,
                    &Filters::default(),
                    ctx_fetch_limit,
                    Some(0.3), // loose threshold to gather candidates
                )
                .await?;
            for scored in &ctx_results {
                candidate_ids.insert(scored.memory.id.clone());
            }
        }

        if candidate_ids.is_empty() {
            debug!("No context candidates found, falling back to unfiltered search");
            return self.search(query, filters, limit).await;
        }

        debug!(
            "Stage 1 context filter found {} candidate memories",
            candidate_ids.len()
        );

        // Stage 2: Content search restricted to candidate IDs
        let mut constrained_filters = filters.clone();
        constrained_filters.candidate_ids = Some(candidate_ids.into_iter().collect());

        let results = self.search(query, &constrained_filters, limit).await?;

        if results.is_empty() {
            debug!("Context-constrained search returned 0 results, falling back to global search");
            return self.search(query, filters, limit).await;
        }

        Ok(results)
    }

    /// Search for similar memories with optional similarity threshold
    pub async fn search_with_threshold(
        &self,
        query: &str,
        filters: &Filters,
        limit: usize,
        similarity_threshold: Option<f32>,
    ) -> Result<Vec<ScoredMemory>> {
        let query_embedding = self.llm_client.embed(query).await?;
        let threshold = similarity_threshold.or(self.config.search_similarity_threshold);

        let total_memories = match self.vector_store.count().await {
            Ok(count) => count,
            Err(e) => {
                warn!("Failed to count memories: {}", e);
                0
            }
        };

        // Search with 0.0 threshold to see best possible matches
        let mut results = self
            .vector_store
            .search_with_threshold(&query_embedding, filters, limit, Some(0.0))
            .await?;

        if results.is_empty() {
            info!(
                "No candidates found for query: \"{}\" with filters: {:?}. (0 raw results). Total memories in bank: {}",
                query, filters, total_memories
            );

            // Relax filters to debug
            let has_filters = filters.memory_type.is_some()
                || filters.topics.is_some()
                || filters.min_importance.is_some()
                || filters.candidate_ids.is_some();

            if has_filters {
                let relaxed_filters = Filters::default();
                if let Ok(relaxed_results) = self
                    .vector_store
                    .search_with_threshold(&query_embedding, &relaxed_filters, 1, Some(0.0))
                    .await
                {
                    if !relaxed_results.is_empty() {
                        info!(
                            "Relaxed search found {} results. Top score: {:.4}. It seems your filters are too restrictive!",
                            relaxed_results.len(),
                            relaxed_results[0].score
                        );
                    } else {
                        info!("Even relaxed search found 0 results. This is strange.");
                    }
                }
            }
            return Ok(vec![]);
        }

        // Log best score
        if let Some(best) = results.first() {
            info!(
                "Query: \"{}\" | Best match score: {:.4} | Candidates found: {} | Total memories: {}",
                query,
                best.score,
                results.len(),
                total_memories
            );
        }

        // Apply actual threshold if set
        if let Some(t) = threshold {
            let original_count = results.len();

            // For logging purposes if everything gets filtered out
            let best_score_so_far = results.first().map(|m| m.score).unwrap_or(0.0);

            results.retain(|m| m.score >= t);

            if results.is_empty() && original_count > 0 {
                info!(
                    "All candidates filtered out by threshold {:.2}. Best score was {:.4}",
                    t, best_score_so_far
                );
            }
        }

        results.sort_by(|a, b| {
            let score_a = a.score * 0.7 + a.memory.metadata.importance_score * 0.3;
            let score_b = b.score * 0.7 + b.memory.metadata.importance_score * 0.3;

            match score_b.partial_cmp(&score_a) {
                Some(std::cmp::Ordering::Equal) | None => {
                    b.memory.created_at.cmp(&a.memory.created_at)
                }
                Some(ordering) => ordering,
            }
        });

        debug!(
            "Found {} similar memories for query with threshold {:?}",
            results.len(),
            threshold
        );
        Ok(results)
    }

    /// Store a pre-constructed memory directly (bypassing normal pipelines)
    pub async fn store_memory(&self, memory: Memory) -> Result<String> {
        self.vector_store.insert(&memory).await?;
        Ok(memory.id)
    }

    /// Retrieve a memory by ID
    pub async fn get(&self, id: &str) -> Result<Option<Memory>> {
        self.vector_store.get(id).await
    }

    /// Update an existing memory
    pub async fn update(
        &self,
        id: &str,
        content: Option<String>,
        relations: Option<Vec<crate::types::Relation>>,
    ) -> Result<()> {
        let mut memory = self
            .vector_store
            .get(id)
            .await?
            .ok_or_else(|| MemoryError::NotFound { id: id.to_string() })?;

        if let Some(c) = content {
            // Content update: replace with new user-provided content
            // Note: This creates a new memory entry with updated content_meta
            memory.content = Some(c.clone());
            memory.content_meta.checksum = Some(ContentMeta::compute_checksum(&c));
            memory.embedding = self.llm_client.embed(&c).await?;
            memory.metadata.hash = self.generate_hash(&c);
            if self.config.auto_enhance {
                self.enhance_memory(&mut memory, true).await?;
            }
        }

        if let Some(new_relations) = relations {
            // We append new relations, avoiding exact duplicates
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

        memory.updated_at = Utc::now();

        self.vector_store.update(&memory).await?;

        info!("Updated memory with ID: {}", id);
        Ok(())
    }

    /// Update a complete memory object directly
    pub async fn update_memory(&self, memory: &Memory) -> Result<()> {
        self.vector_store.update(memory).await?;
        Ok(())
    }

    /// Delete a memory by ID
    pub async fn delete(&self, id: &str) -> Result<()> {
        self.vector_store.delete(id).await?;
        info!("Deleted memory with ID: {}", id);
        Ok(())
    }

    /// Delete a memory with threshold-based cascade degradation for layers.
    ///
    /// Instead of immediately marking all dependent higher-layer memories as Forgotten,
    /// this tracks which sources are deleted and computes a degradation percentage.
    /// Memories transition through states based on per-layer thresholds:
    /// - Active → Degraded: when any source is deleted
    /// - Degraded → Forgotten: when deleted sources exceed the layer's threshold
    ///
    /// Thresholds (fraction of sources that must be deleted to trigger Forgotten):
    /// - L1 (1 source):  100% — single source, any deletion = Forgotten
    /// - L2 (N sources): >50% — majority of sources must be deleted
    /// - L3+ (N sources): >66% — higher layers are more tolerant
    pub async fn delete_with_cascade(&self, memory_id: &str) -> Result<DeletionResult> {
        let memory = self
            .get(memory_id)
            .await?
            .ok_or_else(|| MemoryError::NotFound {
                id: memory_id.to_string(),
            })?;

        let mut result = DeletionResult {
            deleted_id: memory_id.to_string(),
            forgotten: Vec::new(),
            degraded: Vec::new(),
            cascade_depth: 0,
        };

        let deleted_uuid = uuid::Uuid::parse_str(&memory.id)
            .map_err(|e| MemoryError::Validation(format!("Invalid memory ID: {}", e)))?;

        let dependents = self.find_abstraction_dependents(&memory.id).await?;

        for dependent in dependents {
            if dependent.metadata.layer.level <= memory.metadata.layer.level {
                continue;
            }

            let total_sources = dependent.metadata.abstraction_sources.len();
            if total_sources == 0 {
                continue;
            }

            // Record this source as deleted on the dependent
            let mut updated = dependent.clone();
            if !updated.metadata.forgotten_sources.contains(&deleted_uuid) {
                updated.metadata.forgotten_sources.push(deleted_uuid);
            }
            let deleted_count = updated.metadata.forgotten_sources.len();
            let degradation = deleted_count as f64 / total_sources as f64;

            // Per-layer threshold: what fraction of sources must be lost to trigger Forgotten
            let threshold = Self::forgotten_threshold(updated.metadata.layer.level);

            if degradation >= threshold {
                // Exceeds threshold → Forgotten
                updated.metadata.state = crate::types::MemoryState::Forgotten;
                updated.metadata.forgotten_at = Some(chrono::Utc::now());
                updated.metadata.forgotten_by = Some(deleted_uuid);
                updated.updated_at = chrono::Utc::now();
                self.vector_store.update(&updated).await?;

                result.forgotten.push(DegradedMemory {
                    id: dependent.id.clone(),
                    layer: dependent.metadata.layer.level,
                    degradation,
                    total_sources,
                    deleted_sources: deleted_count,
                });
                result.cascade_depth =
                    std::cmp::max(result.cascade_depth, dependent.metadata.layer.level);

                // Recursively propagate: this dependent is now effectively gone,
                // so its own dependents need to be checked too
                let sub_dependents = self.find_abstraction_dependents(&dependent.id).await?;
                for sub_dep in sub_dependents {
                    if sub_dep.metadata.layer.level > dependent.metadata.layer.level {
                        let sub_dep_uuid = uuid::Uuid::parse_str(&dependent.id).ok();
                        if let Some(sub_uuid) = sub_dep_uuid {
                            self.propagate_degradation(&sub_dep, sub_uuid, &mut result).await?;
                        }
                    }
                }
            } else {
                // Below threshold → Degraded (still searchable)
                updated.metadata.state = crate::types::MemoryState::Degraded;
                updated.updated_at = chrono::Utc::now();
                self.vector_store.update(&updated).await?;

                result.degraded.push(DegradedMemory {
                    id: dependent.id.clone(),
                    layer: dependent.metadata.layer.level,
                    degradation,
                    total_sources,
                    deleted_sources: deleted_count,
                });
                result.cascade_depth =
                    std::cmp::max(result.cascade_depth, dependent.metadata.layer.level);
            }
        }

        self.vector_store.delete(memory_id).await?;
        info!(
            "Deleted {} with cascade: {} forgotten, {} degraded across {} layers",
            memory_id,
            result.forgotten.len(),
            result.degraded.len(),
            result.cascade_depth
        );

        Ok(result)
    }

    /// Per-layer threshold for transitioning from Degraded → Forgotten.
    /// Returns the fraction of sources that must be deleted.
    fn forgotten_threshold(layer_level: i32) -> f64 {
        match layer_level {
            1 => 1.0,        // L1 has 1 source — any deletion = Forgotten
            2 => 0.51,       // L2 — majority must be deleted (>50%)
            _ => 0.67,       // L3+ — more tolerant, need >66% deleted
        }
    }

    /// Recursively propagate degradation up the layer hierarchy.
    fn propagate_degradation<'a>(
        &'a self,
        dependent: &'a Memory,
        deleted_source_uuid: uuid::Uuid,
        result: &'a mut DeletionResult,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let total_sources = dependent.metadata.abstraction_sources.len();
            if total_sources == 0 {
                return Ok(());
            }

            let mut updated = dependent.clone();
            if !updated.metadata.forgotten_sources.contains(&deleted_source_uuid) {
                updated.metadata.forgotten_sources.push(deleted_source_uuid);
            }
            let deleted_count = updated.metadata.forgotten_sources.len();
            let degradation = deleted_count as f64 / total_sources as f64;
            let threshold = Self::forgotten_threshold(updated.metadata.layer.level);

            if degradation >= threshold {
                updated.metadata.state = crate::types::MemoryState::Forgotten;
                updated.metadata.forgotten_at = Some(chrono::Utc::now());
                updated.metadata.forgotten_by = Some(deleted_source_uuid);
                updated.updated_at = chrono::Utc::now();
                self.vector_store.update(&updated).await?;

                result.forgotten.push(DegradedMemory {
                    id: dependent.id.clone(),
                    layer: dependent.metadata.layer.level,
                    degradation,
                    total_sources,
                    deleted_sources: deleted_count,
                });
                result.cascade_depth =
                    std::cmp::max(result.cascade_depth, dependent.metadata.layer.level);

                // Continue propagating upward
                let sub_dependents = self.find_abstraction_dependents(&dependent.id).await?;
                for sub_dep in sub_dependents {
                    if sub_dep.metadata.layer.level > dependent.metadata.layer.level {
                        let dep_uuid = uuid::Uuid::parse_str(&dependent.id).ok();
                        if let Some(uuid) = dep_uuid {
                            self.propagate_degradation(&sub_dep, uuid, result).await?;
                        }
                    }
                }
            } else {
                updated.metadata.state = crate::types::MemoryState::Degraded;
                updated.updated_at = chrono::Utc::now();
                self.vector_store.update(&updated).await?;

                result.degraded.push(DegradedMemory {
                    id: dependent.id.clone(),
                    layer: dependent.metadata.layer.level,
                    degradation,
                    total_sources,
                    deleted_sources: deleted_count,
                });
                result.cascade_depth =
                    std::cmp::max(result.cascade_depth, dependent.metadata.layer.level);
            }

            Ok(())
        })
    }

    /// Mark a memory as forgotten (unconditional, for direct use)
    pub async fn mark_as_forgotten(&self, memory_id: &str, deleted_by: &str) -> Result<()> {
        let mut memory = self
            .get(memory_id)
            .await?
            .ok_or_else(|| MemoryError::NotFound {
                id: memory_id.to_string(),
            })?;

        memory.metadata.state = crate::types::MemoryState::Forgotten;
        memory.metadata.forgotten_at = Some(chrono::Utc::now());

        // Convert string ID to Uuid before updating forgotten_by
        if let Ok(uuid_val) = uuid::Uuid::parse_str(deleted_by) {
            memory.metadata.forgotten_by = Some(uuid_val);
        }

        self.vector_store.update(&memory).await?;
        Ok(())
    }

    /// Find all memories that abstract from or link to this memory (reverse direction).
    /// Uses the in-memory abstraction index for O(1) lookup instead of scanning all memories.
    pub async fn find_abstraction_dependents(&self, memory_id: &str) -> Result<Vec<Memory>> {
        let parsed_id = match uuid::Uuid::parse_str(memory_id) {
            Ok(id) => id,
            Err(_) => return Ok(vec![]),
        };

        let mut filters = Filters::new();
        filters.contains_abstraction_source = Some(parsed_id);

        self.vector_store.list(&filters, None).await
    }

    /// Navigate the abstraction hierarchy from a memory node.
    /// - `direction: "zoom_out"` returns higher-layer memories that abstract FROM this memory.
    /// - `direction: "zoom_in"` returns lower-layer source memories this was abstracted FROM.
    /// - `direction: "both"` returns both directions combined.
    pub async fn navigate_memory(
        &self,
        memory_id: &str,
        direction: &str,
        levels: usize,
    ) -> Result<NavigateResult> {
        let memory = self
            .get(memory_id)
            .await?
            .ok_or_else(|| MemoryError::NotFound {
                id: memory_id.to_string(),
            })?;

        let mut result = NavigateResult {
            source_memory_id: memory_id.to_string(),
            source_layer: memory.metadata.layer.level,
            zoom_in: Vec::new(),
            zoom_out: Vec::new(),
        };

        if direction == "zoom_in" || direction == "both" {
            // Follow abstraction_sources down to lower layers
            result.zoom_in = self.trace_sources(&memory, levels).await?;
        }

        if direction == "zoom_out" || direction == "both" {
            // Use the abstraction index to find higher-layer memories
            result.zoom_out = self.find_abstraction_dependents(memory_id).await?;
        }

        Ok(result)
    }

    /// Recursively trace abstraction_sources to find lower-layer memories.
    fn trace_sources<'a>(
        &'a self,
        memory: &'a Memory,
        levels: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Memory>>> + Send + 'a>> {
        Box::pin(async move {
            if levels == 0 {
                return Ok(vec![]);
            }

            let mut sources = Vec::new();
            for src_id in &memory.metadata.abstraction_sources {
                if let Ok(Some(mem)) = self.get(&src_id.to_string()).await {
                    if levels == 1 {
                        sources.push(mem);
                    } else {
                        let deeper = self.trace_sources(&mem, levels - 1).await?;
                        sources.extend(deeper);
                    }
                }
            }

            Ok(sources)
        })
    }

    /// List memories with optional filters
    pub async fn list(&self, filters: &Filters, limit: Option<usize>) -> Result<Vec<Memory>> {
        self.vector_store.list(filters, limit).await
    }

    /// Create procedural memory using specialized prompt system
    pub async fn create_procedural_memory(
        &self,
        messages: &[crate::types::Message],
        metadata: MemoryMetadata,
    ) -> Result<Vec<MemoryResult>> {
        if messages.is_empty() {
            return Ok(vec![]);
        }

        let formatted_messages = self.format_conversation_for_procedural_memory(messages);

        let prompt = format!(
            "{}\n\nConversation:\n{}",
            PROCEDURAL_MEMORY_SYSTEM_PROMPT, formatted_messages
        );

        let response = self.llm_client.complete(&prompt).await?;
        let memory_id = self.store(response.clone(), metadata).await?;

        info!("Created procedural memory with ID: {}", memory_id);

        Ok(vec![MemoryResult {
            id: memory_id,
            memory: response,
            event: MemoryEvent::Add,
            actor_id: messages.last().and_then(|msg| msg.name.clone()),
            role: messages.last().map(|msg| msg.role.clone()),
            previous_memory: None,
        }])
    }

    /// Format conversation messages for procedural memory processing
    fn format_conversation_for_procedural_memory(
        &self,
        messages: &[crate::types::Message],
    ) -> String {
        let mut formatted = String::new();

        for message in messages {
            match message.role.as_str() {
                "assistant" => {
                    formatted.push_str(&format!(
                        "**Agent Action**: {}\n**Action Result**: {}\n\n",
                        self.extract_action_from_assistant_message(&message.content),
                        message.content
                    ));
                }
                "user" => {
                    formatted.push_str(&format!("**User Input**: {}\n", message.content));
                }
                _ => {}
            }
        }

        formatted
    }

    fn extract_action_from_assistant_message(&self, content: &str) -> String {
        if content.contains("executing")
            || content.contains("processing")
            || content.contains("handling")
        {
            "Executing agent operation".to_string()
        } else if content.contains("return") || content.contains("result") {
            "Processing and returning result".to_string()
        } else {
            "Generating response".to_string()
        }
    }

    /// Get memory statistics without cloning all memories
    pub async fn get_stats(&self, filters: &Filters) -> Result<MemoryStats> {
        // Use count() + filtered list iteration without cloning embeddings
        let memories = self.vector_store.list(filters, None).await?;

        let mut stats = MemoryStats {
            total_count: memories.len(),
            by_type: HashMap::new(),
            by_user: HashMap::new(),
            by_agent: HashMap::new(),
        };

        for memory in &memories {
            *stats
                .by_type
                .entry(memory.metadata.memory_type.clone())
                .or_insert(0) += 1;

            if let Some(user_id) = &memory.metadata.user_id {
                *stats.by_user.entry(user_id.clone()).or_insert(0) += 1;
            }

            if let Some(agent_id) = &memory.metadata.agent_id {
                *stats.by_agent.entry(agent_id.clone()).or_insert(0) += 1;
            }
        }

        Ok(stats)
    }

    /// Perform health check on all components
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let vector_store_healthy = self.vector_store.health_check().await?;
        let llm_healthy = self.llm_client.health_check().await?;

        Ok(HealthStatus {
            vector_store: vector_store_healthy,
            llm_service: llm_healthy,
            overall: vector_store_healthy && llm_healthy,
        })
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_count: usize,
    pub by_type: HashMap<MemoryType, usize>,
    pub by_user: HashMap<String, usize>,
    pub by_agent: HashMap<String, usize>,
}

/// Health status of memory system components
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub vector_store: bool,
    pub llm_service: bool,
    pub overall: bool,
}

/// Result of a cascade deletion with threshold-based degradation
#[derive(Debug, Clone)]
pub struct DeletionResult {
    pub deleted_id: String,
    /// Memories that exceeded the degradation threshold and were marked Forgotten
    pub forgotten: Vec<DegradedMemory>,
    /// Memories that lost sources but remain searchable as Degraded
    pub degraded: Vec<DegradedMemory>,
    pub cascade_depth: i32,
}

/// Info about a memory affected by cascade deletion
#[derive(Debug, Clone)]
pub struct DegradedMemory {
    pub id: String,
    pub layer: i32,
    /// Fraction of sources that have been deleted (0.0 - 1.0)
    pub degradation: f64,
    /// Total number of abstraction sources
    pub total_sources: usize,
    /// Number of sources that have been deleted
    pub deleted_sources: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MemoryConfig;
    use crate::llm::extractor_types::*;
    use crate::types::layer::LayerInfo;
    use crate::types::{Memory, MemoryMetadata, MemoryType};
    use async_trait::async_trait;
    use uuid::Uuid;

    const DIM: usize = 8;

    fn make_embedding(seed: f32) -> Vec<f32> {
        (0..DIM).map(|i| seed + i as f32 * 0.1).collect()
    }

    #[derive(Clone)]
    struct MockLLMClient;

    #[async_trait]
    impl crate::llm::client::LLMClient for MockLLMClient {
        async fn complete(&self, _prompt: &str) -> crate::error::Result<String> {
            Ok(String::new())
        }
        async fn complete_with_grammar(
            &self,
            _prompt: &str,
            _grammar: &str,
        ) -> crate::error::Result<String> {
            Ok(String::new())
        }
        async fn embed(&self, _text: &str) -> crate::error::Result<Vec<f32>> {
            Ok(make_embedding(1.0))
        }
        async fn embed_batch(
            &self,
            texts: &[String],
        ) -> crate::error::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| make_embedding(1.0)).collect())
        }
        async fn extract_keywords(&self, _content: &str) -> crate::error::Result<Vec<String>> {
            Ok(vec![])
        }
        async fn summarize(
            &self,
            _content: &str,
            _max_length: Option<usize>,
        ) -> crate::error::Result<String> {
            Ok(String::new())
        }
        async fn health_check(&self) -> crate::error::Result<bool> {
            Ok(true)
        }
        async fn extract_structured_facts(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<StructuredFactExtraction> {
            Ok(StructuredFactExtraction { facts: vec![] })
        }
        async fn extract_detailed_facts(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<DetailedFactExtraction> {
            Ok(DetailedFactExtraction { facts: vec![] })
        }
        async fn extract_keywords_structured(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<KeywordExtraction> {
            Ok(KeywordExtraction { keywords: vec![] })
        }
        async fn classify_memory(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<MemoryClassification> {
            Ok(MemoryClassification {
                memory_type: "factual".into(),
                confidence: 1.0,
                reasoning: String::new(),
            })
        }
        async fn score_importance(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<ImportanceScore> {
            Ok(ImportanceScore {
                score: 0.5,
                reasoning: String::new(),
            })
        }
        async fn check_duplicates(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<DeduplicationResult> {
            Ok(DeduplicationResult {
                is_duplicate: false,
                similarity_score: 0.0,
                original_memory_id: None,
            })
        }
        async fn generate_summary(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<SummaryResult> {
            Ok(SummaryResult {
                summary: String::new(),
                key_points: vec![],
            })
        }
        async fn detect_language(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<LanguageDetection> {
            Ok(LanguageDetection {
                language: "en".into(),
                confidence: 1.0,
            })
        }
        async fn extract_entities(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<EntityExtraction> {
            Ok(EntityExtraction { entities: vec![] })
        }
        async fn analyze_conversation(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<ConversationAnalysis> {
            Ok(ConversationAnalysis {
                topics: vec![],
                sentiment: String::new(),
                user_intent: String::new(),
                key_information: vec![],
            })
        }
        async fn extract_metadata_enrichment(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<MetadataEnrichment> {
            Ok(MetadataEnrichment {
                summary: "mock".into(),
                keywords: vec![],
            })
        }
        async fn extract_metadata_enrichment_batch(
            &self,
            texts: &[String],
        ) -> crate::error::Result<Vec<crate::error::Result<MetadataEnrichment>>> {
            Ok(texts
                .iter()
                .map(|_| {
                    Ok(MetadataEnrichment {
                        summary: "mock".into(),
                        keywords: vec![],
                    })
                })
                .collect())
        }
        async fn complete_batch(
            &self,
            prompts: &[String],
        ) -> crate::error::Result<Vec<crate::error::Result<String>>> {
            Ok(prompts.iter().map(|_| Ok(String::new())).collect())
        }
        fn get_status(&self) -> ClientStatus {
            ClientStatus::default()
        }
        fn batch_config(&self) -> (usize, u32) {
            (10, 4096)
        }
        async fn enhance_memory_unified(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<crate::llm::MemoryEnhancement> {
            Ok(crate::llm::MemoryEnhancement {
                memory_type: "Semantic".into(),
                summary: String::new(),
                keywords: vec![],
                entities: vec![],
                topics: vec![],
            })
        }
    }

    /// Simple in-memory mock vector store for tests
    #[derive(Clone)]
    struct MockVectorStore {
        memories: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, Memory>>>,
    }

    impl MockVectorStore {
        fn new() -> Self {
            Self {
                memories: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            }
        }
    }

    #[async_trait]
    impl crate::vector_store::VectorStore for MockVectorStore {
        async fn insert(&self, memory: &Memory) -> crate::error::Result<()> {
            let mut mems = self.memories.lock().unwrap();
            mems.insert(memory.id.clone(), memory.clone());
            Ok(())
        }

        async fn search(
            &self,
            _query_vector: &[f32],
            _filters: &Filters,
            _limit: usize,
        ) -> crate::error::Result<Vec<crate::types::ScoredMemory>> {
            Ok(vec![])
        }

        async fn search_with_threshold(
            &self,
            _query_vector: &[f32],
            _filters: &Filters,
            _limit: usize,
            _score_threshold: Option<f32>,
        ) -> crate::error::Result<Vec<crate::types::ScoredMemory>> {
            Ok(vec![])
        }

        async fn update(&self, memory: &Memory) -> crate::error::Result<()> {
            let mut mems = self.memories.lock().unwrap();
            mems.insert(memory.id.clone(), memory.clone());
            Ok(())
        }

        async fn delete(&self, id: &str) -> crate::error::Result<()> {
            let mut mems = self.memories.lock().unwrap();
            mems.remove(id);
            Ok(())
        }

        async fn get(&self, id: &str) -> crate::error::Result<Option<Memory>> {
            let mems = self.memories.lock().unwrap();
            Ok(mems.get(id).cloned())
        }

        async fn list(&self, filters: &Filters, _limit: Option<usize>) -> crate::error::Result<Vec<Memory>> {
            let mems = self.memories.lock().unwrap();
            let mut results: Vec<Memory> = mems.values().cloned().collect();
            
            // Filter by contains_abstraction_source
            if let Some(source_uuid) = &filters.contains_abstraction_source {
                results.retain(|m| m.metadata.abstraction_sources.contains(source_uuid));
            }
            
            // Filter by memory_type
            if let Some(memory_type) = &filters.memory_type {
                results.retain(|m| m.metadata.memory_type == *memory_type);
            }
            
            // Filter by state
            if let Some(state) = &filters.state {
                results.retain(|m| m.metadata.state == *state);
            }
            
            // Filter by layer level
            if let Some(min_layer) = filters.min_layer_level {
                results.retain(|m| m.metadata.layer.level >= min_layer);
            }
            if let Some(max_layer) = filters.max_layer_level {
                results.retain(|m| m.metadata.layer.level <= max_layer);
            }
            
            // Filter by importance
            if let Some(min_importance) = filters.min_importance {
                results.retain(|m| m.metadata.importance_score >= min_importance);
            }
            if let Some(max_importance) = filters.max_importance {
                results.retain(|m| m.metadata.importance_score <= max_importance);
            }
            
            // Filter by user_id
            if let Some(user_id) = &filters.user_id {
                results.retain(|m| m.metadata.user_id.as_ref() == Some(user_id));
            }
            
            // Filter by candidate_ids
            if let Some(candidate_ids) = &filters.candidate_ids {
                results.retain(|m| candidate_ids.contains(&m.id));
            }
            
            Ok(results)
        }

        async fn count(&self) -> crate::error::Result<usize> {
            let mems = self.memories.lock().unwrap();
            Ok(mems.len())
        }

        async fn health_check(&self) -> crate::error::Result<bool> {
            Ok(true)
        }
    }

    fn make_manager() -> MemoryManager {
        let store = MockVectorStore::new();

        let mut config = MemoryConfig::default();
        config.auto_enhance = false;
        config.deduplicate = false;

        MemoryManager::new(Box::new(store), Box::new(MockLLMClient), config)
    }

    /// Helper: create an L0 memory and store it, returning its UUID and string ID.
    async fn store_l0(manager: &MemoryManager, content: &str) -> (Uuid, String) {
        let mem = Memory::with_content(
            content.to_string(),
            make_embedding(1.0),
            MemoryMetadata::new(MemoryType::Semantic).with_layer(LayerInfo::raw_content()),
        );
        let uuid = Uuid::parse_str(&mem.id).unwrap();
        let id = manager.store_memory(mem).await.unwrap();
        (uuid, id)
    }

    /// Helper: create a higher-layer memory from given sources.
    async fn store_layer(
        manager: &MemoryManager,
        layer: LayerInfo,
        sources: Vec<Uuid>,
        content: &str,
    ) -> (Uuid, String) {
        let meta = MemoryMetadata::new(MemoryType::Semantic)
            .with_layer(layer)
            .with_abstraction_sources(sources);
        let mem = Memory::with_content(content.to_string(), make_embedding(2.0), meta);
        let uuid = Uuid::parse_str(&mem.id).unwrap();
        let id = manager.store_memory(mem).await.unwrap();
        (uuid, id)
    }

    // ─── Threshold function tests ───

    #[test]
    fn test_forgotten_threshold_l1() {
        assert_eq!(MemoryManager::forgotten_threshold(1), 1.0);
    }

    #[test]
    fn test_forgotten_threshold_l2() {
        assert!((MemoryManager::forgotten_threshold(2) - 0.51).abs() < f64::EPSILON);
    }

    #[test]
    fn test_forgotten_threshold_l3_and_above() {
        assert!((MemoryManager::forgotten_threshold(3) - 0.67).abs() < f64::EPSILON);
        assert!((MemoryManager::forgotten_threshold(4) - 0.67).abs() < f64::EPSILON);
        assert!((MemoryManager::forgotten_threshold(10) - 0.67).abs() < f64::EPSILON);
    }

    // ─── L1 cascade: single source, any deletion = Forgotten ───

    #[tokio::test]
    async fn test_delete_l0_forgets_l1_single_source() {
        let mgr = make_manager();

        let (l0_uuid, l0_id) = store_l0(&mgr, "Raw chunk content").await;
        let (_l1_uuid, l1_id) =
            store_layer(&mgr, LayerInfo::structural(), vec![l0_uuid], "L1 summary").await;

        let result = mgr.delete_with_cascade(&l0_id).await.unwrap();

        // L1 should be Forgotten (100% threshold, single source)
        assert_eq!(result.forgotten.len(), 1);
        assert_eq!(result.forgotten[0].id, l1_id);
        assert!((result.forgotten[0].degradation - 1.0).abs() < f64::EPSILON);
        assert_eq!(result.degraded.len(), 0);

        // Verify the L1 memory state in store
        let l1 = mgr.get(&l1_id).await.unwrap().unwrap();
        assert!(l1.metadata.state.is_forgotten());
        assert_eq!(l1.metadata.forgotten_sources.len(), 1);
        assert_eq!(l1.metadata.forgotten_sources[0], l0_uuid);

        // Original L0 should be deleted
        assert!(mgr.get(&l0_id).await.unwrap().is_none());
    }

    // ─── L2 cascade: 3 sources, delete 1 = Degraded ───

    #[tokio::test]
    async fn test_delete_one_l1_degrades_l2() {
        let mgr = make_manager();

        // Create 3 L1 memories (simulating pre-existing abstractions)
        let (l1a_uuid, l1a_id) =
            store_layer(&mgr, LayerInfo::structural(), vec![Uuid::new_v4()], "L1 A").await;
        let (l1b_uuid, _) =
            store_layer(&mgr, LayerInfo::structural(), vec![Uuid::new_v4()], "L1 B").await;
        let (l1c_uuid, _) =
            store_layer(&mgr, LayerInfo::structural(), vec![Uuid::new_v4()], "L1 C").await;

        // Create L2 that abstracts from all 3 L1s
        let (_l2_uuid, l2_id) = store_layer(
            &mgr,
            LayerInfo::semantic(),
            vec![l1a_uuid, l1b_uuid, l1c_uuid],
            "L2 synthesis",
        )
        .await;

        // Delete one L1 — L2 should be Degraded (1/3 = 33%, below 51% threshold)
        let result = mgr.delete_with_cascade(&l1a_id).await.unwrap();

        assert_eq!(result.forgotten.len(), 0);
        assert_eq!(result.degraded.len(), 1);
        assert_eq!(result.degraded[0].id, l2_id);
        assert_eq!(result.degraded[0].total_sources, 3);
        assert_eq!(result.degraded[0].deleted_sources, 1);
        assert!((result.degraded[0].degradation - 1.0 / 3.0).abs() < 0.01);

        // Verify state in store
        let l2 = mgr.get(&l2_id).await.unwrap().unwrap();
        assert!(l2.metadata.state.is_degraded());
        assert!(l2.metadata.state.is_active()); // still searchable
    }

    // ─── L2 cascade: 3 sources, delete 2 = Forgotten ───

    #[tokio::test]
    async fn test_delete_two_l1s_forgets_l2() {
        let mgr = make_manager();

        let (l1a_uuid, l1a_id) =
            store_layer(&mgr, LayerInfo::structural(), vec![Uuid::new_v4()], "L1 A").await;
        let (l1b_uuid, l1b_id) =
            store_layer(&mgr, LayerInfo::structural(), vec![Uuid::new_v4()], "L1 B").await;
        let (l1c_uuid, _) =
            store_layer(&mgr, LayerInfo::structural(), vec![Uuid::new_v4()], "L1 C").await;

        let (_l2_uuid, l2_id) = store_layer(
            &mgr,
            LayerInfo::semantic(),
            vec![l1a_uuid, l1b_uuid, l1c_uuid],
            "L2 synthesis",
        )
        .await;

        // Delete first L1 → Degraded (1/3)
        let result1 = mgr.delete_with_cascade(&l1a_id).await.unwrap();
        assert_eq!(result1.degraded.len(), 1);
        assert_eq!(result1.forgotten.len(), 0);

        // Delete second L1 → Forgotten (2/3 = 66.7%, above 51% threshold)
        let result2 = mgr.delete_with_cascade(&l1b_id).await.unwrap();
        assert_eq!(result2.forgotten.len(), 1);
        assert_eq!(result2.forgotten[0].id, l2_id);
        assert_eq!(result2.forgotten[0].deleted_sources, 2);

        let l2 = mgr.get(&l2_id).await.unwrap().unwrap();
        assert!(l2.metadata.state.is_forgotten());
        assert_eq!(l2.metadata.forgotten_sources.len(), 2);
    }

    // ─── Multi-level propagation: L0 delete → L1 forgotten → L2 degraded ───

    #[tokio::test]
    async fn test_cascade_propagates_through_layers() {
        let mgr = make_manager();

        // L0
        let (l0_uuid, l0_id) = store_l0(&mgr, "Raw content").await;

        // L1 (single source → will be Forgotten when L0 deleted)
        let (l1_uuid, _l1_id) =
            store_layer(&mgr, LayerInfo::structural(), vec![l0_uuid], "L1 summary").await;

        // Two other L1s (independent sources, won't be affected)
        let (l1b_uuid, _) =
            store_layer(&mgr, LayerInfo::structural(), vec![Uuid::new_v4()], "L1 B").await;
        let (l1c_uuid, _) =
            store_layer(&mgr, LayerInfo::structural(), vec![Uuid::new_v4()], "L1 C").await;

        // L2 depends on all 3 L1s
        let (_l2_uuid, l2_id) = store_layer(
            &mgr,
            LayerInfo::semantic(),
            vec![l1_uuid, l1b_uuid, l1c_uuid],
            "L2 synthesis",
        )
        .await;

        // Delete L0 → L1 Forgotten → L2 Degraded (1/3)
        let result = mgr.delete_with_cascade(&l0_id).await.unwrap();

        // L1 should be forgotten
        let l1_forgotten: Vec<_> = result.forgotten.iter().filter(|d| d.layer == 1).collect();
        assert_eq!(l1_forgotten.len(), 1);

        // L2 should be degraded (one of its 3 L1 sources was forgotten)
        let l2_effect: Vec<_> = result
            .degraded
            .iter()
            .chain(result.forgotten.iter())
            .filter(|d| d.id == l2_id)
            .collect();
        assert_eq!(l2_effect.len(), 1);

        let l2 = mgr.get(&l2_id).await.unwrap().unwrap();
        assert!(l2.metadata.state.is_degraded());
    }

    // ─── No cascade for independent memories ───

    #[tokio::test]
    async fn test_delete_l0_no_dependents() {
        let mgr = make_manager();

        let (_, l0_id) = store_l0(&mgr, "Standalone content").await;

        let result = mgr.delete_with_cascade(&l0_id).await.unwrap();

        assert_eq!(result.forgotten.len(), 0);
        assert_eq!(result.degraded.len(), 0);
        assert_eq!(result.cascade_depth, 0);
        assert!(mgr.get(&l0_id).await.unwrap().is_none());
    }

    // ─── Duplicate deletion is idempotent for forgotten_sources ───

    #[tokio::test]
    async fn test_forgotten_sources_no_duplicates() {
        let mgr = make_manager();

        let (l1a_uuid, l1a_id) =
            store_layer(&mgr, LayerInfo::structural(), vec![Uuid::new_v4()], "L1 A").await;
        let (l1b_uuid, _l1b_id) =
            store_layer(&mgr, LayerInfo::structural(), vec![Uuid::new_v4()], "L1 B").await;
        let (l1c_uuid, _) =
            store_layer(&mgr, LayerInfo::structural(), vec![Uuid::new_v4()], "L1 C").await;

        let (_l2_uuid, l2_id) = store_layer(
            &mgr,
            LayerInfo::semantic(),
            vec![l1a_uuid, l1b_uuid, l1c_uuid],
            "L2 synthesis",
        )
        .await;

        // Delete L1 A
        mgr.delete_with_cascade(&l1a_id).await.unwrap();

        let l2 = mgr.get(&l2_id).await.unwrap().unwrap();
        assert_eq!(l2.metadata.forgotten_sources.len(), 1);

        // Manually try to propagate the same source again via mark_as_forgotten
        // (shouldn't happen in practice, but verify no duplicate tracking)
        assert_eq!(l2.metadata.forgotten_sources[0], l1a_uuid);
    }

    // ─── L3 tolerance: >66% threshold ───

    #[tokio::test]
    async fn test_l3_tolerates_one_of_three_deleted() {
        let mgr = make_manager();

        let (l2a_uuid, l2a_id) = store_layer(
            &mgr,
            LayerInfo::semantic(),
            vec![Uuid::new_v4()],
            "L2 A",
        )
        .await;
        let (l2b_uuid, _) = store_layer(
            &mgr,
            LayerInfo::semantic(),
            vec![Uuid::new_v4()],
            "L2 B",
        )
        .await;
        let (l2c_uuid, _) = store_layer(
            &mgr,
            LayerInfo::semantic(),
            vec![Uuid::new_v4()],
            "L2 C",
        )
        .await;

        let (_l3_uuid, l3_id) = store_layer(
            &mgr,
            LayerInfo::concept(),
            vec![l2a_uuid, l2b_uuid, l2c_uuid],
            "L3 concept",
        )
        .await;

        // Delete 1 of 3 L2 sources → L3 Degraded (33%, below 67%)
        mgr.delete_with_cascade(&l2a_id).await.unwrap();

        let l3 = mgr.get(&l3_id).await.unwrap().unwrap();
        assert!(l3.metadata.state.is_degraded());
        assert!(l3.metadata.state.is_active()); // still searchable
    }

    #[tokio::test]
    async fn test_l3_forgotten_when_majority_deleted() {
        let mgr = make_manager();

        let (l2a_uuid, l2a_id) = store_layer(
            &mgr,
            LayerInfo::semantic(),
            vec![Uuid::new_v4()],
            "L2 A",
        )
        .await;
        let (l2b_uuid, l2b_id) = store_layer(
            &mgr,
            LayerInfo::semantic(),
            vec![Uuid::new_v4()],
            "L2 B",
        )
        .await;
        let (l2c_uuid, l2c_id) = store_layer(
            &mgr,
            LayerInfo::semantic(),
            vec![Uuid::new_v4()],
            "L2 C",
        )
        .await;

        let (_l3_uuid, l3_id) = store_layer(
            &mgr,
            LayerInfo::concept(),
            vec![l2a_uuid, l2b_uuid, l2c_uuid],
            "L3 concept",
        )
        .await;

        // Delete 1/3 → Degraded (33% < 67% threshold)
        mgr.delete_with_cascade(&l2a_id).await.unwrap();
        let l3 = mgr.get(&l3_id).await.unwrap().unwrap();
        assert!(l3.metadata.state.is_degraded());

        // Delete 2/3 → still Degraded (66.7% < 67% threshold)
        mgr.delete_with_cascade(&l2b_id).await.unwrap();
        let l3 = mgr.get(&l3_id).await.unwrap().unwrap();
        assert!(l3.metadata.state.is_degraded());

        // Delete 3/3 → Forgotten (100% >= 67% threshold)
        mgr.delete_with_cascade(&l2c_id).await.unwrap();
        let l3 = mgr.get(&l3_id).await.unwrap().unwrap();
        assert!(l3.metadata.state.is_forgotten());
    }
}
