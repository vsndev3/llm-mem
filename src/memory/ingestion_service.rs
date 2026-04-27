use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::{
    config::MemoryConfig,
    error::{MemoryError, Result},
    llm::LLMClient,
    memory::{
        cache_service::CacheService,
        classification::{create_memory_classifier, MemoryClassifier},
        deduplication::{create_duplicate_detector, DuplicateDetector},
        extractor::{create_fact_extractor, FactExtractor},
        importance::{create_importance_evaluator, ImportanceEvaluator},
        prompts::PROCEDURAL_MEMORY_SYSTEM_PROMPT,
        search_service::SearchService,
        updater::{create_memory_updater, MemoryAction, MemoryUpdater},
    },
    types::{
        ContentMeta, Filters, Memory, MemoryEvent, MemoryMetadata, MemoryResult, MemoryType, Message,
        Relation,
    },
    vector_store::VectorStore,
};

/// Options for storing memory
#[derive(Debug, Clone, Default)]
pub struct StoreOptions {
    pub deduplicate: Option<bool>,
    pub enhance: Option<bool>,
    pub merge: Option<bool>,
}

/// Owns memory ingestion: store, add_memory, ingest_document, create_procedural_memory,
/// content hashing, enhancement, deduplication, and classification.
///
/// Extracted from MemoryManager to reduce its god-object responsibilities.
pub struct IngestionService {
    vector_store: Box<dyn VectorStore + Send + Sync>,
    llm_client: Box<dyn LLMClient + Send + Sync>,
    config: Arc<MemoryConfig>,
    cache: Arc<CacheService>,
    search: Arc<SearchService>,
    fact_extractor: Box<dyn FactExtractor + 'static>,
    memory_updater: Box<dyn MemoryUpdater + 'static>,
    importance_evaluator: Box<dyn ImportanceEvaluator + 'static>,
    duplicate_detector: Box<dyn DuplicateDetector + 'static>,
    #[allow(dead_code)]
    memory_classifier: Box<dyn MemoryClassifier + 'static>,
}

impl IngestionService {
    pub fn new(
        vector_store: Box<dyn VectorStore + Send + Sync>,
        llm_client: Box<dyn LLMClient + Send + Sync>,
        config: Arc<MemoryConfig>,
        cache: Arc<CacheService>,
        search: Arc<SearchService>,
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
            cache,
            search,
            fact_extractor,
            memory_updater,
            importance_evaluator,
            duplicate_detector,
            memory_classifier,
        }
    }

    pub fn llm_client(&self) -> &dyn LLMClient {
        self.llm_client.as_ref()
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
    async fn check_duplicate(&self, content: &str, filters: &Filters) -> Result<Option<Memory>> {
        let hash = Self::generate_hash(content);
        let query_embedding = self.cache.cached_embed(content).await?;

        let candidates = self
            .vector_store
            .search_with_threshold(&query_embedding, filters, 5, Some(0.5))
            .await?;

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
    async fn enhance_memory(&self, memory: &mut Memory, merge: bool) -> Result<()> {
        let content = match &memory.content {
            Some(c) => c,
            None => return Ok(()),
        };

        let prompt = crate::memory::prompts::UNIFIED_MEMORY_ENHANCEMENT_PROMPT
            .replace("{{text}}", content);

        match self.llm_client.enhance_memory_unified(&prompt).await {
            Ok(enhancement) => {
                if memory.metadata.memory_type == MemoryType::Conversational {
                    memory.metadata.memory_type = MemoryType::parse(&enhancement.memory_type);
                }
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
        }

        if let Ok(importance) = self.importance_evaluator.evaluate_importance(memory).await {
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

        let embedding = self.llm_client.embed(&content).await?;
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
            self.enhance_memory(&mut memory, merge).await?;
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

            let all_embeddings = self.llm_client.embed_batch(&all_texts).await?;

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

        self.vector_store.insert(&memory).await?;
        self.search.insert_layer(memory.metadata.layer.level).await;

        tracing::info!(
            "Stored new memory with ID: {} (content length: {}, contexts: {}, relations: {})",
            memory_id,
            memory.content.as_ref().map_or(0, |c| c.len()),
            memory.metadata.context.len(),
            memory.metadata.relations.len(),
        );
        Ok(memory_id)
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

        if metadata.agent_id.is_some() && metadata.memory_type == MemoryType::Procedural {
            return self.create_procedural_memory(messages, metadata).await;
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

        let extracted_keywords = match self.llm_client.extract_keywords(&original_content).await {
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

            let query_embedding = self.llm_client.embed(&fact.content).await?;
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

        Ok(all_actions)
    }

    /// Ingest a document by extracting facts and storing them
    pub async fn ingest_document(&self, text: &str, metadata: MemoryMetadata) -> Result<Vec<MemoryResult>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        let extracted_facts = self.fact_extractor.extract_facts_from_text(text).await?;

        if extracted_facts.is_empty() {
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
                .search_with_threshold(&query_embedding, &filters, 5, self.config.search_similarity_threshold)
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
                        let _ = self.update(id, Some(content.clone()), None).await;
                        all_actions.push(MemoryResult {
                            id: id.clone(),
                            memory: content.clone(),
                            event: MemoryEvent::Update,
                            actor_id: None,
                            role: None,
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
                            actor_id: None,
                            role: None,
                            previous_memory: None,
                        });
                    }
                    MemoryAction::Delete { id } => {
                        let _ = self.delete(id).await;
                        all_actions.push(MemoryResult {
                            id: id.clone(),
                            memory: String::new(),
                            event: MemoryEvent::Delete,
                            actor_id: None,
                            role: None,
                            previous_memory: None,
                        });
                    }
                }
            }
        }

        Ok(all_actions)
    }

    /// Create procedural memory using specialized prompt system
    async fn create_procedural_memory(
        &self,
        messages: &[Message],
        metadata: MemoryMetadata,
    ) -> Result<Vec<MemoryResult>> {
        if messages.is_empty() {
            return Ok(vec![]);
        }

        let formatted_messages = self.format_conversation_for_procedural_memory(messages);
        let prompt = format!("{}\n\nConversation:\n{}", PROCEDURAL_MEMORY_SYSTEM_PROMPT, formatted_messages);

        let response = self.llm_client.complete(&prompt).await?;
        let memory_id = self.store(response.clone(), metadata).await?;

        Ok(vec![MemoryResult {
            id: memory_id,
            memory: response,
            event: MemoryEvent::Add,
            actor_id: messages.last().and_then(|msg| msg.name.clone()),
            role: messages.last().map(|msg| msg.role.clone()),
            previous_memory: None,
        }])
    }

    fn format_conversation_for_procedural_memory(&self, messages: &[Message]) -> String {
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
        if content.contains("executing") || content.contains("processing") || content.contains("handling") {
            "Executing agent operation".to_string()
        } else if content.contains("return") || content.contains("result") {
            "Processing and returning result".to_string()
        } else {
            "Generating response".to_string()
        }
    }

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
            memory.embedding = self.llm_client.embed(&c).await?;
            memory.metadata.hash = Self::generate_hash(&c);
            if self.config.auto_enhance {
                self.enhance_memory(&mut memory, true).await?;
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
}