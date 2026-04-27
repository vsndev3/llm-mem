use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    config::MemoryConfig,
    error::Result,
    llm::LLMClient,
    memory::{
        abstraction_service::AbstractionService,
        cache_service::CacheService,
        ingestion_service::IngestionService,
        metrics::MetricsSink,
        search_service::SearchService,
    },
    types::{Filters, Memory, MemoryMetadata, MemoryResult, MemoryType, NavigateResult, ScoredMemory},
    vector_store::VectorStore,
};

pub use crate::memory::abstraction_service::{DeletionResult, DegradedMemory};
pub use crate::memory::ingestion_service::StoreOptions;

/// Core memory manager that orchestrates memory operations by delegating
/// to domain-specific services (SearchService, IngestionService,
/// AbstractionService, CacheService).
///
/// This is a thin facade — the god object decomposition from PLAN.md §2.1.
pub struct MemoryManager {
    vector_store: Box<dyn VectorStore>,
    llm_client: Box<dyn LLMClient>,
    config: Arc<MemoryConfig>,
    cache: Arc<CacheService>,
    search: Arc<SearchService>,
    ingestion: Arc<IngestionService>,
    abstraction: Arc<AbstractionService>,
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new(
        vector_store: Box<dyn VectorStore>,
        llm_client: Box<dyn LLMClient>,
        config: MemoryConfig,
    ) -> Self {
        let config = Arc::new(config);

        let cache = Arc::new(CacheService::new(dyn_clone::clone_box(llm_client.as_ref())));
        let search = Arc::new(SearchService::new(
            dyn_clone::clone_box(vector_store.as_ref()),
            dyn_clone::clone_box(llm_client.as_ref()),
            Arc::clone(&config),
            Arc::clone(&cache),
        ));
        let ingestion = Arc::new(IngestionService::new(
            dyn_clone::clone_box(vector_store.as_ref()),
            dyn_clone::clone_box(llm_client.as_ref()),
            Arc::clone(&config),
            Arc::clone(&cache),
            Arc::clone(&search),
        ));
        let abstraction = Arc::new(AbstractionService::new(
            dyn_clone::clone_box(vector_store.as_ref()),
            Arc::clone(&search),
        ));

        Self {
            vector_store,
            llm_client,
            config,
            cache,
            search,
            ingestion,
            abstraction,
        }
    }

    /// Set a custom metrics sink for observability.
    pub fn set_metrics_sink(&mut self, sink: Arc<dyn MetricsSink>) {
        Arc::get_mut(&mut self.cache).map(|c| c.set_metrics_sink(sink));
    }

    fn generate_hash(content: &str) -> String {
        IngestionService::generate_hash(content)
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

    // ─── Metadata enrichment ───

    /// Extract metadata enrichment for a text chunk
    pub async fn extract_metadata_enrichment(
        &self,
        text: &str,
    ) -> Result<crate::memory::extractor::ChunkMetadata> {
        self.ingestion.extract_metadata_enrichment(text).await
    }

    /// Extract metadata enrichment for multiple text chunks in batch
    pub async fn extract_metadata_enrichment_batch(
        &self,
        texts: &[String],
    ) -> Result<Vec<crate::memory::extractor::ChunkMetadata>> {
        self.ingestion.extract_metadata_enrichment_batch(texts).await
    }

    // ─── Import ───

    /// Import a fully-formed Memory directly into the vector store.
    pub async fn import_memory(&self, memory: &Memory) -> Result<()> {
        self.ingestion.import_memory(memory).await
    }

    // ─── Create ───

    /// Create a new memory from content and metadata
    pub async fn create_memory(&self, content: String, metadata: MemoryMetadata) -> Result<Memory> {
        self.ingestion.create_memory(content, metadata).await
    }

    /// Create a new memory from content and metadata with options
    pub async fn create_memory_with_options(
        &self,
        content: String,
        metadata: MemoryMetadata,
        options: &StoreOptions,
    ) -> Result<Memory> {
        self.ingestion.create_memory_with_options(content, metadata, options).await
    }

    // ─── Store ───

    /// Store a memory in the vector store
    pub async fn store(&self, content: String, metadata: MemoryMetadata) -> Result<String> {
        self.ingestion.store(content, metadata).await
    }

    /// Store a memory with fine-grained control options
    pub async fn store_with_options(
        &self,
        content: String,
        metadata: MemoryMetadata,
        options: StoreOptions,
    ) -> Result<String> {
        self.ingestion.store_with_options(content, metadata, options).await
    }

    /// Store a pre-constructed memory directly (bypassing normal pipelines)
    pub async fn store_memory(&self, memory: Memory) -> Result<String> {
        self.ingestion.store_memory(memory).await
    }

    // ─── Add / Ingest ───

    /// Add memory from conversation messages with full fact extraction
    pub async fn add_memory(
        &self,
        messages: &[crate::types::Message],
        metadata: MemoryMetadata,
    ) -> Result<Vec<MemoryResult>> {
        self.ingestion.add_memory(messages, metadata).await
    }

    /// Ingest a document by extracting facts and storing them
    pub async fn ingest_document(
        &self,
        text: &str,
        metadata: MemoryMetadata,
    ) -> Result<Vec<MemoryResult>> {
        self.ingestion.ingest_document(text, metadata).await
    }

    // ─── CRUD ───

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
        self.ingestion.update(id, content, relations).await
    }

    /// Update a complete memory object directly
    pub async fn update_memory(&self, memory: &Memory) -> Result<()> {
        self.ingestion.update_memory(memory).await
    }

    /// Delete a memory by ID
    pub async fn delete(&self, id: &str) -> Result<()> {
        self.ingestion.delete(id).await
    }

    /// List memories with optional filters
    pub async fn list(&self, filters: &Filters, limit: Option<usize>) -> Result<Vec<Memory>> {
        self.ingestion.list(filters, limit).await
    }

    // ─── Search ───

    /// Search for similar memories with importance-weighted ranking
    pub async fn search(
        &self,
        query: &str,
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        self.search.search(query, filters, limit).await
    }

    /// Search with an optional similarity threshold override.
    pub async fn search_with_override(
        &self,
        query: &str,
        filters: &Filters,
        limit: usize,
        threshold_override: Option<f32>,
    ) -> Result<Vec<ScoredMemory>> {
        self.search.search_with_override(query, filters, limit, threshold_override).await
    }

    /// Search for similar memories with optional similarity threshold
    pub async fn search_with_threshold(
        &self,
        query: &str,
        filters: &Filters,
        limit: usize,
        similarity_threshold: Option<f32>,
    ) -> Result<Vec<ScoredMemory>> {
        self.search.search_with_threshold(query, filters, limit, similarity_threshold).await
    }

    /// Two-stage retrieval with context-based pre-filtering.
    pub async fn search_with_context(
        &self,
        query: &str,
        context_tags: &[String],
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        self.search.search_with_context(query, context_tags, filters, limit).await
    }

    /// Hierarchical pyramid search across all abstraction layers.
    pub async fn search_pyramid(
        &self,
        query: &str,
        filters: &Filters,
        limit: usize,
        config: &crate::search::PyramidConfig,
    ) -> Result<Vec<crate::search::PyramidResult>> {
        self.search.search_pyramid(query, filters, limit, config).await
    }

    // ─── Layers ───

    /// Discover which layers have active memories.
    pub async fn discover_active_layers(&self) -> Vec<i32> {
        self.search.discover_active_layers().await
    }

    /// Force-refresh the layer manifest from the vector store.
    pub async fn refresh_layer_manifest(&self) -> Result<()> {
        self.search.refresh_layer_manifest().await
    }

    // ─── Cascade deletion ───

    /// Delete a memory with threshold-based cascade degradation for layers.
    pub async fn delete_with_cascade(&self, memory_id: &str) -> Result<DeletionResult> {
        self.abstraction.delete_with_cascade(memory_id).await
    }

    /// Mark a memory as forgotten (unconditional, for direct use)
    pub async fn mark_as_forgotten(&self, memory_id: &str, deleted_by: &str) -> Result<()> {
        self.abstraction.mark_as_forgotten(memory_id, deleted_by).await
    }

    /// Find all memories that abstract from or link to this memory.
    pub async fn find_abstraction_dependents(&self, memory_id: &str) -> Result<Vec<Memory>> {
        self.abstraction.find_abstraction_dependents(memory_id).await
    }

    /// Navigate the abstraction hierarchy from a memory node.
    pub async fn navigate_memory(
        &self,
        memory_id: &str,
        direction: &str,
        levels: usize,
    ) -> Result<NavigateResult> {
        self.abstraction.navigate_memory(memory_id, direction, levels).await
    }

    // ─── Procedural memory ───

    /// Create procedural memory using specialized prompt system
    pub async fn create_procedural_memory(
        &self,
        messages: &[crate::types::Message],
        metadata: MemoryMetadata,
    ) -> Result<Vec<MemoryResult>> {
        self.ingestion.add_memory(messages, metadata).await
    }

    // ─── Stats & Health ───

    /// Get memory statistics without cloning all memories
    pub async fn get_stats(&self, filters: &Filters) -> Result<crate::memory::manager::MemoryStats> {
        let memories = self.vector_store.list(filters, None).await?;

        let mut stats = crate::memory::manager::MemoryStats {
            total_count: memories.len(),
            by_type: HashMap::new(),
            by_user: HashMap::new(),
            by_agent: HashMap::new(),
        };

        for memory in &memories {
            *stats.by_type.entry(memory.metadata.memory_type.clone()).or_insert(0) += 1;
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
    pub async fn health_check(&self) -> Result<crate::memory::manager::HealthStatus> {
        let vector_store_healthy = self.vector_store.health_check().await?;
        let llm_healthy = self.llm_client.health_check().await?;

        Ok(crate::memory::manager::HealthStatus {
            vector_store: vector_store_healthy,
            llm_service: llm_healthy,
            overall: vector_store_healthy && llm_healthy,
        })
    }

    /// Classify query intent for dynamic pyramid allocation (delegates to CacheService)
    pub async fn classify_query_intent(&self, query: &str) -> crate::search::PyramidAllocationMode {
        self.cache.classify_query_intent(query, self.config.use_llm_query_classification).await
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
        async fn embed_batch(&self, texts: &[String]) -> crate::error::Result<Vec<Vec<f32>>> {
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
        async fn score_importance(&self, _prompt: &str) -> crate::error::Result<ImportanceScore> {
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
        async fn generate_summary(&self, _prompt: &str) -> crate::error::Result<SummaryResult> {
            Ok(SummaryResult {
                summary: String::new(),
                key_points: vec![],
            })
        }
        async fn detect_language(&self, _prompt: &str) -> crate::error::Result<LanguageDetection> {
            Ok(LanguageDetection {
                language: "en".into(),
                confidence: 1.0,
            })
        }
        async fn extract_entities(&self, _prompt: &str) -> crate::error::Result<EntityExtraction> {
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
                memories: std::sync::Arc::new(std::sync::Mutex::new(
                    std::collections::HashMap::new(),
                )),
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

        async fn list(
            &self,
            filters: &Filters,
            _limit: Option<usize>,
        ) -> crate::error::Result<Vec<Memory>> {
            let mems = self.memories.lock().unwrap();
            let mut results: Vec<Memory> = mems.values().cloned().collect();

            if let Some(source_uuid) = &filters.contains_abstraction_source {
                results.retain(|m| m.metadata.abstraction_sources.contains(source_uuid));
            }
            if let Some(memory_type) = &filters.memory_type {
                results.retain(|m| m.metadata.memory_type == *memory_type);
            }
            if let Some(state) = &filters.state {
                results.retain(|m| m.metadata.state == *state);
            }
            if let Some(min_layer) = filters.min_layer_level {
                results.retain(|m| m.metadata.layer.level >= min_layer);
            }
            if let Some(max_layer) = filters.max_layer_level {
                results.retain(|m| m.metadata.layer.level <= max_layer);
            }
            if let Some(min_importance) = filters.min_importance {
                results.retain(|m| m.metadata.importance_score >= min_importance);
            }
            if let Some(max_importance) = filters.max_importance {
                results.retain(|m| m.metadata.importance_score <= max_importance);
            }
            if let Some(user_id) = &filters.user_id {
                results.retain(|m| m.metadata.user_id.as_ref() == Some(user_id));
            }
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

        let config = MemoryConfig {
            auto_enhance: false,
            deduplicate: false,
            ..Default::default()
        };

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

    #[test]
    fn test_forgotten_threshold_l1() {
        assert!((crate::memory::abstraction_service::AbstractionService::forgotten_threshold_static(1) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_forgotten_threshold_l2() {
        assert!((crate::memory::abstraction_service::AbstractionService::forgotten_threshold_static(2) - 0.51).abs() < f64::EPSILON);
    }

    #[test]
    fn test_forgotten_threshold_l3_and_above() {
        assert!((crate::memory::abstraction_service::AbstractionService::forgotten_threshold_static(3) - 0.67).abs() < f64::EPSILON);
        assert!((crate::memory::abstraction_service::AbstractionService::forgotten_threshold_static(4) - 0.67).abs() < f64::EPSILON);
        assert!((crate::memory::abstraction_service::AbstractionService::forgotten_threshold_static(10) - 0.67).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_delete_l0_forgets_l1_single_source() {
        let mgr = make_manager();

        let (l0_uuid, l0_id) = store_l0(&mgr, "Raw chunk content").await;
        let (_l1_uuid, l1_id) =
            store_layer(&mgr, LayerInfo::structural(), vec![l0_uuid], "L1 summary").await;

        let result = mgr.delete_with_cascade(&l0_id).await.unwrap();

        assert_eq!(result.forgotten.len(), 1);
        assert_eq!(result.forgotten[0].id, l1_id);
        assert!((result.forgotten[0].degradation - 1.0).abs() < f64::EPSILON);
        assert_eq!(result.degraded.len(), 0);

        let l1 = mgr.get(&l1_id).await.unwrap().unwrap();
        assert!(l1.metadata.state.is_forgotten());
        assert_eq!(l1.metadata.forgotten_sources.len(), 1);
        assert_eq!(l1.metadata.forgotten_sources[0], l0_uuid);

        assert!(mgr.get(&l0_id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_delete_one_l1_degrades_l2() {
        let mgr = make_manager();

        let (l1a_uuid, l1a_id) =
            store_layer(&mgr, LayerInfo::structural(), vec![Uuid::new_v4()], "L1 A").await;
        let (l1b_uuid, _) =
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

        let result = mgr.delete_with_cascade(&l1a_id).await.unwrap();

        assert_eq!(result.forgotten.len(), 0);
        assert_eq!(result.degraded.len(), 1);
        assert_eq!(result.degraded[0].id, l2_id);
        assert_eq!(result.degraded[0].total_sources, 3);
        assert_eq!(result.degraded[0].deleted_sources, 1);
        assert!((result.degraded[0].degradation - 1.0 / 3.0).abs() < 0.01);

        let l2 = mgr.get(&l2_id).await.unwrap().unwrap();
        assert!(l2.metadata.state.is_degraded());
        assert!(l2.metadata.state.is_active());
    }

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

        let result1 = mgr.delete_with_cascade(&l1a_id).await.unwrap();
        assert_eq!(result1.degraded.len(), 1);
        assert_eq!(result1.forgotten.len(), 0);

        let result2 = mgr.delete_with_cascade(&l1b_id).await.unwrap();
        assert_eq!(result2.forgotten.len(), 1);
        assert_eq!(result2.forgotten[0].id, l2_id);
        assert_eq!(result2.forgotten[0].deleted_sources, 2);

        let l2 = mgr.get(&l2_id).await.unwrap().unwrap();
        assert!(l2.metadata.state.is_forgotten());
        assert_eq!(l2.metadata.forgotten_sources.len(), 2);
    }

    #[tokio::test]
    async fn test_cascade_propagates_through_layers() {
        let mgr = make_manager();

        let (l0_uuid, l0_id) = store_l0(&mgr, "Raw content").await;

        let (l1_uuid, _l1_id) =
            store_layer(&mgr, LayerInfo::structural(), vec![l0_uuid], "L1 summary").await;

        let (l1b_uuid, _) =
            store_layer(&mgr, LayerInfo::structural(), vec![Uuid::new_v4()], "L1 B").await;
        let (l1c_uuid, _) =
            store_layer(&mgr, LayerInfo::structural(), vec![Uuid::new_v4()], "L1 C").await;

        let (_l2_uuid, l2_id) = store_layer(
            &mgr,
            LayerInfo::semantic(),
            vec![l1_uuid, l1b_uuid, l1c_uuid],
            "L2 synthesis",
        )
        .await;

        let result = mgr.delete_with_cascade(&l0_id).await.unwrap();

        let l1_forgotten: Vec<_> = result.forgotten.iter().filter(|d| d.layer == 1).collect();
        assert_eq!(l1_forgotten.len(), 1);

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

        mgr.delete_with_cascade(&l1a_id).await.unwrap();

        let l2 = mgr.get(&l2_id).await.unwrap().unwrap();
        assert_eq!(l2.metadata.forgotten_sources.len(), 1);
        assert_eq!(l2.metadata.forgotten_sources[0], l1a_uuid);
    }

    #[tokio::test]
    async fn test_l3_tolerates_one_of_three_deleted() {
        let mgr = make_manager();

        let (l2a_uuid, l2a_id) =
            store_layer(&mgr, LayerInfo::semantic(), vec![Uuid::new_v4()], "L2 A").await;
        let (l2b_uuid, _) =
            store_layer(&mgr, LayerInfo::semantic(), vec![Uuid::new_v4()], "L2 B").await;
        let (l2c_uuid, _) =
            store_layer(&mgr, LayerInfo::semantic(), vec![Uuid::new_v4()], "L2 C").await;

        let (_l3_uuid, l3_id) = store_layer(
            &mgr,
            LayerInfo::concept(),
            vec![l2a_uuid, l2b_uuid, l2c_uuid],
            "L3 concept",
        )
        .await;

        mgr.delete_with_cascade(&l2a_id).await.unwrap();

        let l3 = mgr.get(&l3_id).await.unwrap().unwrap();
        assert!(l3.metadata.state.is_degraded());
        assert!(l3.metadata.state.is_active());
    }

    #[tokio::test]
    async fn test_l3_forgotten_when_majority_deleted() {
        let mgr = make_manager();

        let (l2a_uuid, l2a_id) =
            store_layer(&mgr, LayerInfo::semantic(), vec![Uuid::new_v4()], "L2 A").await;
        let (l2b_uuid, l2b_id) =
            store_layer(&mgr, LayerInfo::semantic(), vec![Uuid::new_v4()], "L2 B").await;
        let (l2c_uuid, l2c_id) =
            store_layer(&mgr, LayerInfo::semantic(), vec![Uuid::new_v4()], "L2 C").await;

        let (_l3_uuid, l3_id) = store_layer(
            &mgr,
            LayerInfo::concept(),
            vec![l2a_uuid, l2b_uuid, l2c_uuid],
            "L3 concept",
        )
        .await;

        mgr.delete_with_cascade(&l2a_id).await.unwrap();
        let l3 = mgr.get(&l3_id).await.unwrap().unwrap();
        assert!(l3.metadata.state.is_degraded());

        mgr.delete_with_cascade(&l2b_id).await.unwrap();
        let l3 = mgr.get(&l3_id).await.unwrap().unwrap();
        assert!(l3.metadata.state.is_degraded());

        mgr.delete_with_cascade(&l2c_id).await.unwrap();
        let l3 = mgr.get(&l3_id).await.unwrap().unwrap();
        assert!(l3.metadata.state.is_forgotten());
    }

    #[tokio::test]
    async fn test_classify_query_conceptual() {
        let mgr = make_manager();
        let mode = mgr
            .classify_query_intent("Why does this theory work?")
            .await;
        assert_eq!(mode, crate::search::PyramidAllocationMode::TopHeavy);

        let mode = mgr
            .classify_query_intent("Explain the concept of eigenvalues")
            .await;
        assert_eq!(mode, crate::search::PyramidAllocationMode::TopHeavy);

        let mode = mgr
            .classify_query_intent("What is the difference between these two principles")
            .await;
        assert_eq!(mode, crate::search::PyramidAllocationMode::TopHeavy);
    }

    #[tokio::test]
    async fn test_classify_query_factual() {
        let mgr = make_manager();
        let mode = mgr
            .classify_query_intent("What is the date of the event?")
            .await;
        assert_eq!(mode, crate::search::PyramidAllocationMode::BottomHeavy);

        let mode = mgr
            .classify_query_intent("Who created this and where is it located?")
            .await;
        assert_eq!(mode, crate::search::PyramidAllocationMode::BottomHeavy);
    }

    #[tokio::test]
    async fn test_classify_query_balanced() {
        let mgr = make_manager();
        let mode = mgr.classify_query_intent("Tell me about the project").await;
        assert_eq!(mode, crate::search::PyramidAllocationMode::Balanced);

        let mode = mgr.classify_query_intent("Review notes").await;
        assert_eq!(mode, crate::search::PyramidAllocationMode::Balanced);
    }

    #[tokio::test]
    async fn test_deep_cascade_10_layers() {
        let mgr = make_manager();

        let (root_uuid, root_id) = store_l0(&mgr, "Root content").await;

        let mut prev_uuid = root_uuid;
        let mut layer_ids = Vec::new();

        for level in 1..=9 {
            let layer = if level == 1 {
                LayerInfo::structural()
            } else if level == 2 {
                LayerInfo::semantic()
            } else if level == 3 {
                LayerInfo::concept()
            } else {
                LayerInfo::custom(level as i32, format!("L{}", level))
            };
            let (uuid, id) = store_layer(&mgr, layer, vec![prev_uuid], "abstraction").await;
            layer_ids.push((uuid, id));
            prev_uuid = uuid;
        }

        let result = mgr.delete_with_cascade(&root_id).await.unwrap();

        assert_eq!(result.forgotten.len(), 9);
        assert_eq!(result.degraded.len(), 0);
        assert_eq!(result.cascade_depth, 9);

        for (_uuid, id) in &layer_ids {
            let mem = mgr.get(id).await.unwrap().unwrap();
            assert!(mem.metadata.state.is_forgotten(), "Layer {} should be Forgotten", mem.metadata.layer.level);
        }

        assert!(mgr.get(&root_id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_wide_cascade_many_sources() {
        let mgr = make_manager();

        let mut l1_uuids = Vec::new();
        let mut l1_ids = Vec::new();
        for _ in 0..10 {
            let (uuid, id) =
                store_layer(&mgr, LayerInfo::structural(), vec![Uuid::new_v4()], "L1").await;
            l1_uuids.push(uuid);
            l1_ids.push(id);
        }

        let (_l2_uuid, l2_id) = store_layer(
            &mgr,
            LayerInfo::semantic(),
            l1_uuids.clone(),
            "L2 mega-synthesis",
        )
        .await;

        mgr.delete_with_cascade(&l1_ids[0]).await.unwrap();
        mgr.delete_with_cascade(&l1_ids[1]).await.unwrap();
        mgr.delete_with_cascade(&l1_ids[2]).await.unwrap();

        let l2 = mgr.get(&l2_id).await.unwrap().unwrap();
        assert!(l2.metadata.state.is_degraded());
        assert_eq!(l2.metadata.forgotten_sources.len(), 3);

        mgr.delete_with_cascade(&l1_ids[3]).await.unwrap();
        mgr.delete_with_cascade(&l1_ids[4]).await.unwrap();
        mgr.delete_with_cascade(&l1_ids[5]).await.unwrap();

        let l2 = mgr.get(&l2_id).await.unwrap().unwrap();
        assert!(l2.metadata.state.is_forgotten());
        assert_eq!(l2.metadata.forgotten_sources.len(), 6);
    }

    #[tokio::test]
    async fn test_discover_active_layers_empty() {
        let mgr = make_manager();
        let layers = mgr.discover_active_layers().await;
        assert!(layers.contains(&0));
    }

    #[tokio::test]
    async fn test_discover_active_layers_multi_layer() {
        let mgr = make_manager();

        let (_uuid, id0) = store_l0(&mgr, "Raw content").await;
        let (_uuid, id1) = store_layer(
            &mgr,
            LayerInfo::structural(),
            vec![Uuid::parse_str(&id0).unwrap()],
            "L1",
        )
        .await;
        let (_uuid, id2) = store_layer(
            &mgr,
            LayerInfo::semantic(),
            vec![Uuid::parse_str(&id1).unwrap()],
            "L2",
        )
        .await;
        let _ = store_layer(
            &mgr,
            LayerInfo::concept(),
            vec![Uuid::parse_str(&id2).unwrap()],
            "L3",
        )
        .await;

        let layers = mgr.discover_active_layers().await;
        assert!(layers.contains(&0));
        assert!(layers.contains(&1));
        assert!(layers.contains(&2));
        assert!(layers.contains(&3));
    }

    #[tokio::test]
    async fn test_search_pyramid_empty_bank() {
        let mgr = make_manager();
        let results = mgr
            .search_pyramid(
                "test query",
                &Filters::default(),
                10,
                &crate::search::PyramidConfig::default(),
            )
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_pyramid_returns_layer_metadata() {
        let mgr = make_manager();

        let _ = store_l0(&mgr, "The Laplace transform converts time-domain signals").await;
        let _ = store_layer(
            &mgr,
            LayerInfo::structural(),
            vec![Uuid::new_v4()],
            "Summary of Laplace transform chapter",
        )
        .await;

        let results = mgr
            .search_pyramid(
                "Laplace transform",
                &Filters::default(),
                10,
                &crate::search::PyramidConfig::default(),
            )
            .await
            .unwrap();
        for r in &results {
            assert!(r.layer >= 0);
            assert!(!r.layer_name.is_empty());
            assert!(r.search_phase == "pyramid" || r.search_phase == "graph_discovered");
        }
    }

    #[derive(Clone)]
    struct ScoringMockStore {
        memories: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, Memory>>>,
    }

    impl ScoringMockStore {
        fn new() -> Self {
            Self {
                memories: std::sync::Arc::new(std::sync::Mutex::new(
                    std::collections::HashMap::new(),
                )),
            }
        }
    }

    #[async_trait]
    impl crate::vector_store::VectorStore for ScoringMockStore {
        async fn insert(&self, memory: &Memory) -> crate::error::Result<()> {
            let mut mems = self.memories.lock().unwrap();
            mems.insert(memory.id.clone(), memory.clone());
            Ok(())
        }

        async fn search(
            &self,
            _query_vector: &[f32],
            filters: &Filters,
            limit: usize,
        ) -> crate::error::Result<Vec<crate::types::ScoredMemory>> {
            let mems = self.memories.lock().unwrap();
            let mut results: Vec<_> = mems
                .values()
                .filter(|m| {
                    if let Some(min_layer) = filters.min_layer_level {
                        if m.metadata.layer.level < min_layer { return false; }
                    }
                    if let Some(max_layer) = filters.max_layer_level {
                        if m.metadata.layer.level > max_layer { return false; }
                    }
                    true
                })
                .take(limit)
                .map(|m| crate::types::ScoredMemory {
                    memory: m.clone(),
                    score: 0.8,
                })
                .collect();
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            Ok(results)
        }

        async fn search_with_threshold(
            &self,
            _query_vector: &[f32],
            filters: &Filters,
            limit: usize,
            _score_threshold: Option<f32>,
        ) -> crate::error::Result<Vec<crate::types::ScoredMemory>> {
            let mems = self.memories.lock().unwrap();
            let mut results: Vec<_> = mems
                .values()
                .filter(|m| {
                    if let Some(min_layer) = filters.min_layer_level {
                        if m.metadata.layer.level < min_layer { return false; }
                    }
                    if let Some(max_layer) = filters.max_layer_level {
                        if m.metadata.layer.level > max_layer { return false; }
                    }
                    true
                })
                .take(limit)
                .map(|m| crate::types::ScoredMemory {
                    memory: m.clone(),
                    score: 0.8,
                })
                .collect();
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            Ok(results)
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

        async fn list(
            &self,
            _filters: &Filters,
            _limit: Option<usize>,
        ) -> crate::error::Result<Vec<Memory>> {
            let mems = self.memories.lock().unwrap();
            Ok(mems.values().cloned().collect())
        }

        async fn count(&self) -> crate::error::Result<usize> {
            let mems = self.memories.lock().unwrap();
            Ok(mems.len())
        }

        async fn health_check(&self) -> crate::error::Result<bool> {
            Ok(true)
        }
    }

    fn make_scoring_manager() -> MemoryManager {
        let store = ScoringMockStore::new();
        let config = MemoryConfig {
            auto_enhance: false,
            deduplicate: false,
            ..Default::default()
        };
        MemoryManager::new(Box::new(store), Box::new(MockLLMClient), config)
    }

    #[tokio::test]
    async fn test_search_pyramid_multi_layer_integration() {
        let mgr = make_scoring_manager();

        let _l0a = store_l0(&mgr, "The Laplace transform converts time-domain signals").await;
        let _l0b = store_l0(&mgr, "Fourier series represent periodic functions as sums of sines").await;
        let _l0c = store_l0(&mgr, "Eigenvalues determine system stability").await;

        let (l0a_uuid, _) = store_l0(&mgr, "Transfer functions relate input and output in control systems").await;
        let _l1 = store_layer(
            &mgr,
            LayerInfo::structural(),
            vec![l0a_uuid],
            "Signal processing transforms and their applications in engineering",
        )
        .await;

        let _l2 = store_layer(
            &mgr,
            LayerInfo::semantic(),
            vec![l0a_uuid],
            "Mathematical transforms provide powerful tools for analyzing dynamic systems",
        )
        .await;

        let results = mgr
            .search_pyramid(
                "signal processing",
                &Filters::default(),
                10,
                &crate::search::PyramidConfig {
                    mode: crate::search::PyramidAllocationMode::Balanced,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let mut layers_seen = std::collections::HashSet::new();
        for r in &results {
            layers_seen.insert(r.layer);
            assert!(!r.layer_name.is_empty());
            assert!(r.memory.score > 0.0);
            assert_eq!(r.search_phase, "pyramid");
        }

        assert!(
            layers_seen.len() >= 2,
            "Pyramid search should return results from multiple layers, got {:?}",
            layers_seen
        );

        for r in &results {
            assert!(r.layer >= 0, "Layer should be non-negative, got {}", r.layer);
            assert!(!r.layer_name.is_empty(), "Layer name should not be empty");
        }
    }

    #[tokio::test]
    async fn test_search_pyramid_none_mode_returns_flat() {
        let mgr = make_scoring_manager();

        let _l0 = store_l0(&mgr, "Quantum entanglement violates local realism").await;
        let _l1 = store_layer(
            &mgr,
            LayerInfo::structural(),
            vec![],
            "Quantum mechanics challenges classical intuition",
        )
        .await;

        let results = mgr
            .search_pyramid(
                "quantum",
                &Filters::default(),
                10,
                &crate::search::PyramidConfig {
                    mode: crate::search::PyramidAllocationMode::None,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        for r in &results {
            assert_eq!(r.search_phase, "flat");
        }
    }
}