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
        ScoredMemory,
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
        self.fact_extractor.extract_metadata_enrichment(text).await
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
                if memory.content.as_ref().map_or(true, |c| c.trim().is_empty()) {
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

        // Skip keyword extraction if already present
        let needs_keywords = !memory.metadata.custom.contains_key("keywords");

        // Skip summary if already present or if below threshold
        let needs_summary = (content.len() > self.config.auto_summary_threshold)
            && !memory.metadata.custom.contains_key("summary");

        let (keywords_res, summary_res, memory_type_res, entities_res, topics_res) = tokio::join!(
            async {
                if needs_keywords {
                    self.llm_client.extract_keywords(content).await.map(Some)
                } else {
                    Ok(None)
                }
            },
            async {
                if needs_summary {
                    self.llm_client
                        .summarize(content, Some(200))
                        .await
                        .map(Some)
                } else {
                    Ok(None)
                }
            },
            self.memory_classifier.classify_memory(content),
            self.memory_classifier.extract_entities(content),
            self.memory_classifier.extract_topics(content),
        );

        if let Ok(Some(keywords)) = keywords_res {
            memory.metadata.custom.insert(
                "keywords".to_string(),
                serde_json::Value::Array(
                    keywords
                        .into_iter()
                        .map(serde_json::Value::String)
                        .collect(),
                ),
            );
        }

        if let Ok(Some(summary)) = summary_res {
            memory
                .metadata
                .custom
                .insert("summary".to_string(), serde_json::Value::String(summary));
        }

        if let Ok(memory_type) = memory_type_res {
            // Only overwrite if it's the default conversational type
            if memory.metadata.memory_type == MemoryType::Conversational {
                memory.metadata.memory_type = memory_type;
            }
        }

        if let Ok(entities) = entities_res {
            // Append instead of overwrite if some already exist
            if memory.metadata.entities.is_empty() {
                memory.metadata.entities = entities;
            } else {
                for entity in entities {
                    if !memory.metadata.entities.contains(&entity) {
                        memory.metadata.entities.push(entity);
                    }
                }
            }
        }

        if let Ok(topics) = topics_res {
            // Append instead of overwrite
            if memory.metadata.topics.is_empty() {
                memory.metadata.topics = topics;
            } else {
                for topic in topics {
                    if !memory.metadata.topics.contains(&topic) {
                        memory.metadata.topics.push(topic);
                    }
                }
            }
        }

        if let Ok(importance) = self.importance_evaluator.evaluate_importance(memory).await {
            // Use the higher of the two scores if one was already set
            memory.metadata.importance_score = memory.metadata.importance_score.max(importance);
        }

        if merge {
            if let Ok(duplicates) = self.duplicate_detector.detect_duplicates(memory).await {
                if !duplicates.is_empty() {
                    let mut all_memories = vec![memory.clone()];
                    all_memories.extend(duplicates);

                    if let Ok(merged_memory) =
                        self.duplicate_detector.merge_memories(&all_memories).await
                    {
                        *memory = merged_memory;

                        for duplicate in &all_memories[1..] {
                            let _ = self.vector_store.delete(&duplicate.id).await;
                        }
                    }
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

            if !user_messages.is_empty() {
                if let Ok(user_facts) = self.fact_extractor.extract_user_facts(&user_messages).await
                {
                    if !user_facts.is_empty() {
                        final_extracted_facts = user_facts;
                    }
                }
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
            let filters = Filters {
                user_id: metadata.user_id.clone(),
                agent_id: metadata.agent_id.clone(),
                run_id: metadata.run_id.clone(),
                memory_type: None,
                actor_id: metadata.actor_id.clone(),
                min_importance: None,
                max_importance: None,
                created_after: None,
                created_before: None,
                updated_after: None,
                updated_before: None,
                entities: None,
                topics: None,
                relations: None,
                candidate_ids: None,
                custom: HashMap::new(),
            };

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
                .update_memories(&[fact.clone()], &existing_memories, &metadata)
                .await?;

            for action in &update_result.actions_performed {
                match action {
                    MemoryAction::Create { content, metadata } => {
                        // Add extracted keywords to metadata for enhanced searchability
                        let mut metadata_with_keywords = metadata.clone();
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
            let filters = Filters {
                user_id: metadata.user_id.clone(),
                agent_id: metadata.agent_id.clone(),
                run_id: metadata.run_id.clone(),
                memory_type: None,
                actor_id: metadata.actor_id.clone(),
                min_importance: None,
                max_importance: None,
                created_after: None,
                created_before: None,
                updated_after: None,
                updated_before: None,
                entities: None,
                topics: None,
                relations: None,
                candidate_ids: None,
                custom: HashMap::new(),
            };

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
                .update_memories(&[fact.clone()], &existing_memories, &metadata)
                .await?;

            for action in &update_result.actions_performed {
                match action {
                    MemoryAction::Create { content, metadata } => {
                        let memory_id = self.store(content.clone(), metadata.clone()).await?;
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
            let filters = Filters {
                user_id: metadata.user_id.clone(),
                agent_id: metadata.agent_id.clone(),
                run_id: metadata.run_id.clone(),
                memory_type: Some(metadata.memory_type.clone()),
                actor_id: metadata.actor_id.clone(),
                min_importance: None,
                max_importance: None,
                created_after: None,
                created_before: None,
                updated_after: None,
                updated_before: None,
                entities: None,
                topics: None,
                relations: None,
                candidate_ids: None,
                custom: metadata.custom.clone(),
            };

            if let Some(existing) = self.check_duplicate(&content, &filters).await? {
                if existing.content.as_ref().map_or(true, |c| c.trim().is_empty()) {
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

        if memory.content.as_ref().map_or(true, |c| c.trim().is_empty()) {
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
        let search_similarity_threshold = self.config.search_similarity_threshold;

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
            if let Some(keywords_val) = scored.memory.metadata.custom.get("keywords") {
                if let Some(memory_keywords) = keywords_val.as_array() {
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
            if let Some(keywords_val) = scored.memory.metadata.custom.get("keywords") {
                if let Some(memory_keywords) = keywords_val.as_array() {
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

    /// Delete a memory by ID
    pub async fn delete(&self, id: &str) -> Result<()> {
        self.vector_store.delete(id).await?;
        info!("Deleted memory with ID: {}", id);
        Ok(())
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

        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

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
