//! Comprehensive integration tests for LanceDB backend
//!
//! These tests cover:
//! - Full lifecycle: insert, search, update, delete
//! - Persistence and restoration
//! - Complex filtering scenarios
//! - Layer hierarchy and cascade deletion
//! - Backup and restore workflows
//! - Performance characteristics

use async_trait::async_trait;
use llm_mem::{
    config::MemoryConfig,
    error::Result,
    llm::{
        ClientStatus, ConversationAnalysis, DeduplicationResult, DetailedFactExtraction,
        EntityExtraction, ImportanceScore, KeywordExtraction, LLMClient, LanguageDetection,
        MemoryClassification, MemoryEnhancement, StructuredFactExtraction, SummaryResult,
    },
    types::{Filters, Memory, MemoryMetadata, MemoryState, MemoryType, LayerInfo},
    VectorStore,
};
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;
use uuid::Uuid;

// ─── Mock LLM Client ──────────────────────────────────────────────────────

#[derive(Clone)]
struct MockLLMClient {
    dimension: usize,
}

impl MockLLMClient {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn make_embedding(&self, text: &str) -> Vec<f32> {
        let mut emb = vec![0.0f32; self.dimension];
        for (i, ch) in text.chars().enumerate() {
            emb[i % self.dimension] += (ch as u32 as f32) / 1000.0;
        }
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut emb {
                *v /= norm;
            }
        }
        emb
    }
}

#[async_trait]
impl LLMClient for MockLLMClient {
    async fn complete(&self, prompt: &str) -> Result<String> {
        Ok(format!("Mock: {}", &prompt[..prompt.len().min(50)]))
    }

    async fn complete_with_grammar(&self, _prompt: &str, _grammar: &str) -> Result<String> {
        Ok("{\"summary\": \"mock\", \"keywords\": []}".to_string())
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.make_embedding(text))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.make_embedding(t)).collect())
    }

    async fn extract_keywords(&self, content: &str) -> Result<Vec<String>> {
        Ok(content.split_whitespace().take(5).map(|s| s.to_lowercase()).collect())
    }

    async fn summarize(&self, content: &str, max_length: Option<usize>) -> Result<String> {
        Ok(content.chars().take(max_length.unwrap_or(100)).collect())
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }

    async fn extract_structured_facts(&self, _prompt: &str) -> Result<StructuredFactExtraction> {
        Ok(StructuredFactExtraction { facts: vec![] })
    }

    async fn extract_detailed_facts(&self, _prompt: &str) -> Result<DetailedFactExtraction> {
        Ok(DetailedFactExtraction { facts: vec![] })
    }

    async fn extract_keywords_structured(&self, content: &str) -> Result<KeywordExtraction> {
        Ok(KeywordExtraction {
            keywords: content.split_whitespace().take(5).map(|s| s.to_lowercase()).collect(),
        })
    }

    async fn classify_memory(&self, _prompt: &str) -> Result<MemoryClassification> {
        Ok(MemoryClassification {
            memory_type: "Conversational".to_string(),
            confidence: 0.9,
            reasoning: "Mock classification".to_string(),
        })
    }

    async fn score_importance(&self, _prompt: &str) -> Result<ImportanceScore> {
        Ok(ImportanceScore {
            score: 0.5,
            reasoning: "Mock importance".to_string(),
        })
    }

    async fn check_duplicates(&self, _prompt: &str) -> Result<DeduplicationResult> {
        Ok(DeduplicationResult {
            is_duplicate: false,
            similarity_score: 0.0,
            original_memory_id: None,
        })
    }

    async fn generate_summary(&self, _prompt: &str) -> Result<SummaryResult> {
        Ok(SummaryResult {
            summary: "Mock summary".to_string(),
            key_points: vec![],
        })
    }

    async fn detect_language(&self, _prompt: &str) -> Result<LanguageDetection> {
        Ok(LanguageDetection {
            language: "en".to_string(),
            confidence: 0.95,
        })
    }

    async fn extract_entities(&self, _prompt: &str) -> Result<EntityExtraction> {
        Ok(EntityExtraction { entities: vec![] })
    }

    async fn analyze_conversation(&self, _prompt: &str) -> Result<ConversationAnalysis> {
        Ok(ConversationAnalysis {
            topics: vec![],
            sentiment: "neutral".to_string(),
            user_intent: "none".to_string(),
            key_information: vec![],
        })
    }

    async fn extract_metadata_enrichment(&self, _prompt: &str) -> Result<llm_mem::llm::MetadataEnrichment> {
        Ok(llm_mem::llm::MetadataEnrichment {
            summary: "".to_string(),
            keywords: vec![],
        })
    }

    async fn extract_metadata_enrichment_batch(
        &self,
        _texts: &[String],
    ) -> Result<Vec<Result<llm_mem::llm::MetadataEnrichment>>> {
        Ok(vec![])
    }

    async fn complete_batch(&self, _prompts: &[String]) -> Result<Vec<Result<String>>> {
        Ok(vec![])
    }

    fn get_status(&self) -> ClientStatus {
        use std::collections::HashMap;
        ClientStatus {
            backend: "mock".to_string(),
            state: "ready".to_string(),
            llm_model: "mock".to_string(),
            embedding_model: "mock".to_string(),
            llm_available: true,
            embedding_available: true,
            last_llm_success: None,
            last_embedding_success: None,
            last_error: None,
            total_llm_calls: 0,
            total_embedding_calls: 0,
            total_prompt_tokens: 0,
            total_completion_tokens: 0,
            details: HashMap::new(),
        }
    }

    fn batch_config(&self) -> (usize, u32) {
        (10, 3000)
    }

    async fn enhance_memory_unified(&self, _prompt: &str) -> Result<MemoryEnhancement> {
        Ok(MemoryEnhancement {
            memory_type: "Semantic".into(),
            summary: String::new(),
            keywords: vec![],
            entities: vec![],
            topics: vec![],
        })
    }
}

// ─── Test Helpers ─────────────────────────────────────────────────────────

fn create_memory_with_layer(
    id: &str,
    content: &str,
    layer_level: i32,
    state: MemoryState,
) -> Memory {
    let embedding = vec![0.1; 384];
    let mut memory = Memory {
        id: id.to_string(),
        content: Some(content.to_string()),
        content_meta: Default::default(),
        derived_data: HashMap::new(),
        relations: HashMap::new(),
        embedding: embedding.clone(),
        metadata: MemoryMetadata::new(MemoryType::Conversational),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        context_embeddings: None,
        relation_embeddings: None,
    };
    memory.metadata.layer = LayerInfo {
        level: layer_level,
        name: None,
        schema_version: None,
    };
    memory.metadata.state = state;
    memory
}

fn create_memory_with_importance(id: &str, content: &str, importance: f32) -> Memory {
    let mut memory = create_memory_with_layer(id, content, 0, MemoryState::Active);
    memory.metadata.importance_score = importance;
    memory
}

fn create_memory_with_user(id: &str, content: &str, user_id: &str) -> Memory {
    let mut memory = create_memory_with_layer(id, content, 0, MemoryState::Active);
    memory.metadata.user_id = Some(user_id.to_string());
    memory
}

fn create_memory_with_abstraction_sources(
    id: &str,
    content: &str,
    sources: Vec<Uuid>,
) -> Memory {
    let mut memory = create_memory_with_layer(id, content, 1, MemoryState::Active);
    memory.metadata.abstraction_sources = sources;
    memory
}

// ─── LanceDB-specific Tests ───────────────────────────────────────────────

#[tokio::test]
async fn test_lancedb_persistence_and_restore() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().to_path_buf();
    
    // Phase 1: Create store and insert data
    {
        let config = llm_mem::lance_store::LanceDBConfig {
            table_name: "test_persistence".to_string(),
            database_path: db_path.clone(),
            embedding_dimension: 384,
        };
        let store = llm_mem::lance_store::LanceDBStore::new(config).await.unwrap();
        
        for i in 1..=10 {
            let memory = create_memory_with_importance(
                &format!("mem-{}", i),
                &format!("Content {}", i),
                0.5 + (i as f32 * 0.05),
            );
            store.insert(&memory).await.unwrap();
        }
        
        assert_eq!(store.count().await.unwrap(), 10);
    }
    
    // Phase 2: Reopen store from same path
    {
        let config = llm_mem::lance_store::LanceDBConfig {
            table_name: "test_persistence".to_string(),
            database_path: db_path.clone(),
            embedding_dimension: 384,
        };
        let store = llm_mem::lance_store::LanceDBStore::new(config).await.unwrap();
        
        assert_eq!(store.count().await.unwrap(), 10);
        
        let all_memories = store.list(&Filters::default(), None).await.unwrap();
        assert_eq!(all_memories.len(), 10);
        
        // Verify specific memory
        let retrieved = store.get("mem-5").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "mem-5");
    }
}

#[tokio::test]
async fn test_lancedb_complex_filtering() {
    let temp_dir = TempDir::new().unwrap();
    let config = llm_mem::lance_store::LanceDBConfig {
        table_name: "test_filtering".to_string(),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: 384,
    };
    let store = llm_mem::lance_store::LanceDBStore::new(config).await.unwrap();
    
    // Insert varied data
    let mut mem1 = create_memory_with_importance("mem-1", "Important tech news", 0.9);
    mem1.metadata.memory_type = MemoryType::Factual;
    mem1.metadata.user_id = Some("user-a".to_string());
    store.insert(&mem1).await.unwrap();
    
    let mut mem2 = create_memory_with_importance("mem-2", "Casual conversation", 0.3);
    mem2.metadata.memory_type = MemoryType::Conversational;
    mem2.metadata.user_id = Some("user-a".to_string());
    store.insert(&mem2).await.unwrap();
    
    let mut mem3 = create_memory_with_importance("mem-3", "Important science", 0.85);
    mem3.metadata.memory_type = MemoryType::Factual;
    mem3.metadata.user_id = Some("user-b".to_string());
    store.insert(&mem3).await.unwrap();
    
    // Test 1: Filter by importance and type
    let filters = Filters {
        min_importance: Some(0.7),
        memory_type: Some(MemoryType::Factual),
        ..Default::default()
    };
    
    let results = store.list(&filters, None).await.unwrap();
    assert_eq!(results.len(), 2);
    
    // Test 2: Filter by user
    let filters = Filters {
        user_id: Some("user-a".to_string()),
        ..Default::default()
    };
    
    let results = store.list(&filters, None).await.unwrap();
    assert_eq!(results.len(), 2);
    
    // Test 3: Combined filters
    let filters = Filters {
        min_importance: Some(0.5),
        user_id: Some("user-a".to_string()),
        ..Default::default()
    };
    
    let results = store.list(&filters, None).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "mem-1");
}

#[tokio::test]
async fn test_lancedb_layer_hierarchy() {
    let temp_dir = TempDir::new().unwrap();
    let config = llm_mem::lance_store::LanceDBConfig {
        table_name: "test_layers".to_string(),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: 384,
    };
    let store = llm_mem::lance_store::LanceDBStore::new(config).await.unwrap();
    
    let source_uuid = Uuid::new_v4();
    
    // L0: Raw content
    let mem_l0 = create_memory_with_layer("l0-1", "Raw conversation text", 0, MemoryState::Active);
    store.insert(&mem_l0).await.unwrap();
    
    // L1: Summary derived from L0
    let mut mem_l1 = create_memory_with_abstraction_sources("l1-1", "Summary of conversation", vec![source_uuid]);
    mem_l1.metadata.layer = LayerInfo {
        level: 1,
        name: None,
        schema_version: None,
    };
    store.insert(&mem_l1).await.unwrap();
    
    // L2: Concept derived from L1
    let mut mem_l2 = create_memory_with_layer("l2-1", "Key concept extracted", 2, MemoryState::Active);
    mem_l2.metadata.abstraction_sources = vec![source_uuid];
    store.insert(&mem_l2).await.unwrap();
    
    // Filter by layer level
    let filters = Filters {
        min_layer_level: Some(1),
        ..Default::default()
    };
    
    let results = store.list(&filters, None).await.unwrap();
    assert_eq!(results.len(), 2);
    
    // Filter by exact layer
    let filters = Filters {
        min_layer_level: Some(1),
        max_layer_level: Some(1),
        ..Default::default()
    };
    
    let results = store.list(&filters, None).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].metadata.layer.level, 1);
}

#[tokio::test]
async fn test_lancedb_cascade_deletion_scenario() {
    let temp_dir = TempDir::new().unwrap();
    let config = llm_mem::lance_store::LanceDBConfig {
        table_name: "test_cascade".to_string(),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: 384,
    };
    let store = llm_mem::lance_store::LanceDBStore::new(config).await.unwrap();
    
    let source_uuid = Uuid::new_v4();
    
    // Create L0 source
    let mem_l0 = create_memory_with_layer("l0-source", "Original content", 0, MemoryState::Active);
    store.insert(&mem_l0).await.unwrap();
    
    // Create L1 abstraction
    let mut mem_l1 = create_memory_with_abstraction_sources("l1-derived", "Derived summary", vec![source_uuid]);
    mem_l1.metadata.layer = LayerInfo {
        level: 1,
        name: None,
        schema_version: None,
    };
    store.insert(&mem_l1).await.unwrap();
    
    // Create L2 higher abstraction
    let mut mem_l2 = create_memory_with_layer("l2-concept", "Higher concept", 2, MemoryState::Active);
    mem_l2.metadata.abstraction_sources = vec![source_uuid];
    store.insert(&mem_l2).await.unwrap();
    
    // Verify all exist
    assert_eq!(store.count().await.unwrap(), 3);
    
    // Delete L0 source
    store.delete("l0-source").await.unwrap();
    assert_eq!(store.count().await.unwrap(), 2);
    
    // L1 and L2 should still exist but with broken references
    let l1 = store.get("l1-derived").await.unwrap();
    assert!(l1.is_some());
    
    let l2 = store.get("l2-concept").await.unwrap();
    assert!(l2.is_some());
}

#[tokio::test]
async fn test_lancedb_search_with_filters() {
    let temp_dir = TempDir::new().unwrap();
    let config = llm_mem::lance_store::LanceDBConfig {
        table_name: "test_search".to_string(),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: 384,
    };
    let store = llm_mem::lance_store::LanceDBStore::new(config).await.unwrap();
    
    // Insert memories with different embeddings
    let mut mem1 = create_memory_with_importance("mem-1", "Machine learning algorithms", 0.8);
    mem1.embedding = vec![0.9; 384];
    store.insert(&mem1).await.unwrap();
    
    let mut mem2 = create_memory_with_importance("mem-2", "Cooking recipes", 0.4);
    mem2.embedding = vec![0.1; 384];
    store.insert(&mem2).await.unwrap();
    
    let mut mem3 = create_memory_with_importance("mem-3", "Deep learning techniques", 0.75);
    mem3.embedding = vec![0.85; 384];
    store.insert(&mem3).await.unwrap();
    
    // Search with high importance filter
    let query_vector = vec![0.9; 384];
    let filters = Filters {
        min_importance: Some(0.7),
        ..Default::default()
    };
    
    let results = store.search(&query_vector, &filters, 10).await.unwrap();
    assert_eq!(results.len(), 2);
    
    // First result should be mem-1 (closest to query)
    assert_eq!(results[0].memory.id, "mem-1");
}

#[tokio::test]
async fn test_lancedb_update_preserves_data() {
    let temp_dir = TempDir::new().unwrap();
    let config = llm_mem::lance_store::LanceDBConfig {
        table_name: "test_update".to_string(),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: 384,
    };
    let store = llm_mem::lance_store::LanceDBStore::new(config).await.unwrap();
    
    let mut memory = create_memory_with_importance("mem-1", "Original content", 0.5);
    memory.metadata.importance_score = 0.5;
    store.insert(&memory).await.unwrap();
    
    // Update the memory
    memory.content = Some("Updated content".to_string());
    memory.metadata.importance_score = 0.9;
    memory.updated_at = chrono::Utc::now();
    store.update(&memory).await.unwrap();
    
    // Verify update
    let retrieved = store.get("mem-1").await.unwrap().unwrap();
    assert_eq!(retrieved.content, Some("Updated content".to_string()));
    assert_eq!(retrieved.metadata.importance_score, 0.9);
}

#[tokio::test]
async fn test_lancedb_batch_operations() {
    let temp_dir = TempDir::new().unwrap();
    let config = llm_mem::lance_store::LanceDBConfig {
        table_name: "test_batch".to_string(),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: 384,
    };
    let store = llm_mem::lance_store::LanceDBStore::new(config).await.unwrap();
    
    // Insert 100 memories
    for i in 0..100 {
        let memory = create_memory_with_importance(
            &format!("batch-{}", i),
            &format!("Batch content {}", i),
            (i % 10) as f32 / 10.0,
        );
        store.insert(&memory).await.unwrap();
    }
    
    assert_eq!(store.count().await.unwrap(), 100);
    
    // List with pagination
    let results = store.list(&Filters::default(), Some(20)).await.unwrap();
    assert_eq!(results.len(), 20);
    
    // Filter by importance
    let filters = Filters {
        min_importance: Some(0.8),
        ..Default::default()
    };
    let results = store.list(&filters, None).await.unwrap();
    assert_eq!(results.len(), 20); // 0.8, 0.9 appear 20 times each in 100 items
}

#[tokio::test]
async fn test_lancedb_state_filtering() {
    let temp_dir = TempDir::new().unwrap();
    let config = llm_mem::lance_store::LanceDBConfig {
        table_name: "test_state".to_string(),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: 384,
    };
    let store = llm_mem::lance_store::LanceDBStore::new(config).await.unwrap();
    
    let mem_active = create_memory_with_layer("active", "Active memory", 0, MemoryState::Active);
    store.insert(&mem_active).await.unwrap();
    
    let mem_forgotten = create_memory_with_layer("forgotten", "Forgotten memory", 0, MemoryState::Forgotten);
    store.insert(&mem_forgotten).await.unwrap();
    
    let mem_processing = create_memory_with_layer("processing", "Processing memory", 0, MemoryState::Processing);
    store.insert(&mem_processing).await.unwrap();
    
    // Filter by active state
    let mut filters = Filters::default();
    filters.state = Some(MemoryState::Active);
    
    let results = store.list(&filters, None).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "active");
    
    // Filter by forgotten state
    let mut filters = Filters::default();
    filters.state = Some(MemoryState::Forgotten);
    
    let results = store.list(&filters, None).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "forgotten");
}

#[tokio::test]
async fn test_lancedb_date_filtering() {
    use chrono::{Duration, Utc};
    
    let temp_dir = TempDir::new().unwrap();
    let config = llm_mem::lance_store::LanceDBConfig {
        table_name: "test_dates".to_string(),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: 384,
    };
    let store = llm_mem::lance_store::LanceDBStore::new(config).await.unwrap();
    
    let now = Utc::now();
    let yesterday = now - Duration::days(1);
    let week_ago = now - Duration::days(7);
    
    let mut mem1 = create_memory_with_layer("old", "Old content", 0, MemoryState::Active);
    mem1.created_at = week_ago;
    mem1.updated_at = week_ago;
    store.insert(&mem1).await.unwrap();
    
    let mut mem2 = create_memory_with_layer("recent", "Recent content", 0, MemoryState::Active);
    mem2.created_at = now - Duration::hours(2);
    mem2.updated_at = now;
    store.insert(&mem2).await.unwrap();
    
    // Filter created after yesterday (should get only recent)
    let mut filters = Filters::default();
    filters.created_after = Some(yesterday);
    
    let results = store.list(&filters, None).await.unwrap();
    assert_eq!(results.len(), 1, "Expected 1 memory created after yesterday");
    assert_eq!(results[0].id, "recent");
    
    // Filter updated after yesterday (should get only recent)
    let mut filters = Filters::default();
    filters.updated_after = Some(yesterday);
    
    let results = store.list(&filters, None).await.unwrap();
    assert_eq!(results.len(), 1, "Expected 1 memory updated after yesterday");
    assert_eq!(results[0].id, "recent");
}

#[tokio::test]
async fn test_lancedb_memory_manager_integration() {
    use llm_mem::memory::MemoryManager;
    use llm_mem::types::Message;
    
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().to_path_buf();
    
    // Create MemoryManager with LanceDB
    let llm_client: Box<dyn llm_mem::llm::LLMClient> = Box::new(MockLLMClient::new(384));
    
    let config = llm_mem::lance_store::LanceDBConfig {
        table_name: "test_manager".to_string(),
        database_path: db_path.clone(),
        embedding_dimension: 384,
    };
    let vector_store: Box<dyn llm_mem::vector_store::VectorStore> = 
        Box::new(llm_mem::lance_store::LanceDBStore::new(config).await.unwrap());
    
    let memory_config = MemoryConfig {
        max_memories: 10000,
        similarity_threshold: 0.65,
        max_search_results: 50,
        auto_enhance: false,
        deduplicate: false,
        merge_threshold: 0.75,
        auto_summary_threshold: 32768,
        max_content_length: 32768,
        document_chunk_size: 2000,
        memory_ttl_hours: None,
        search_similarity_threshold: Some(0.35),
    };
    
    let manager = MemoryManager::new(vector_store, llm_client, memory_config);
    
    // Add a memory through the manager
    let messages = vec![Message {
        role: "user".to_string(),
        content: "Test content for integration".to_string(),
        name: None,
    }];
    
    let metadata = llm_mem::types::MemoryMetadata::new(llm_mem::types::MemoryType::Conversational);
    
    let result = manager.add_memory(&messages, metadata).await;
    
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(!results.is_empty());
    let memory_id = results[0].id.clone();
    
    // Verify it was stored
    let retrieved = manager.get(&memory_id).await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id, memory_id);
    
    // Search for it
    let search_results = manager
        .search("test content", &Filters::default(), 10)
        .await
        .unwrap();
    
    assert!(!search_results.is_empty());
    assert_eq!(search_results[0].memory.id, memory_id);
}

#[tokio::test]
async fn test_lancedb_empty_and_edge_cases() {
    let temp_dir = TempDir::new().unwrap();
    let config = llm_mem::lance_store::LanceDBConfig {
        table_name: "test_edge".to_string(),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: 384,
    };
    let store = llm_mem::lance_store::LanceDBStore::new(config).await.unwrap();
    
    // Empty store operations
    assert_eq!(store.count().await.unwrap(), 0);
    
    let empty_results = store.list(&Filters::default(), None).await.unwrap();
    assert_eq!(empty_results.len(), 0);
    
    let empty_search = store.search(&vec![0.1; 384], &Filters::default(), 10).await.unwrap();
    assert_eq!(empty_search.len(), 0);
    
    // Get non-existent memory
    let result = store.get("non-existent").await.unwrap();
    assert!(result.is_none());
    
    // Delete non-existent memory
    store.delete("non-existent").await.unwrap(); // Should not error
    
    // Empty content
    let memory = create_memory_with_layer("empty-content", "", 0, MemoryState::Active);
    store.insert(&memory).await.unwrap();
    
    let retrieved = store.get("empty-content").await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().content, Some("".to_string()));
}

#[tokio::test]
async fn test_lancedb_metadata_serialization() {
    let temp_dir = TempDir::new().unwrap();
    let config = llm_mem::lance_store::LanceDBConfig {
        table_name: "test_metadata".to_string(),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: 384,
    };
    let store = llm_mem::lance_store::LanceDBStore::new(config).await.unwrap();
    
    let mut memory = create_memory_with_importance("meta-test", "Content with metadata", 0.7);
    
    // Set complex metadata
    memory.metadata.user_id = Some("test-user".to_string());
    memory.metadata.agent_id = Some("test-agent".to_string());
    memory.metadata.entities = vec!["Entity1".to_string(), "Entity2".to_string()];
    memory.metadata.topics = vec!["Topic1".to_string(), "Topic2".to_string()];
    memory.metadata.context = vec!["Context1".to_string()];
    
    store.insert(&memory).await.unwrap();
    
    // Retrieve and verify
    let retrieved = store.get("meta-test").await.unwrap().unwrap();
    
    assert_eq!(retrieved.metadata.user_id, Some("test-user".to_string()));
    assert_eq!(retrieved.metadata.agent_id, Some("test-agent".to_string()));
    assert_eq!(retrieved.metadata.entities.len(), 2);
    assert_eq!(retrieved.metadata.topics.len(), 2);
}

#[tokio::test]
async fn test_lancedb_concurrent_operations() {
    let temp_dir = TempDir::new().unwrap();
    let config = llm_mem::lance_store::LanceDBConfig {
        table_name: "test_concurrent".to_string(),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: 384,
    };
    let store = Arc::new(llm_mem::lance_store::LanceDBStore::new(config).await.unwrap());
    
    // Spawn multiple concurrent insertions
    let mut handles = vec![];
    for i in 0..20 {
        let store_clone = store.clone();
        let handle = tokio::spawn(async move {
            let memory = create_memory_with_importance(&format!("concurrent-{}", i), &format!("Content {}", i), 0.5);
            store_clone.insert(&memory).await.unwrap();
        });
        handles.push(handle);
    }
    
    // Wait for all to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify all inserted
    assert_eq!(store.count().await.unwrap(), 20);
}

#[tokio::test]
async fn test_lancedb_search_accuracy() {
    let temp_dir = TempDir::new().unwrap();
    let config = llm_mem::lance_store::LanceDBConfig {
        table_name: "test_accuracy".to_string(),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: 384,
    };
    let store = llm_mem::lance_store::LanceDBStore::new(config).await.unwrap();

    // Create memories with known embeddings
    let mut mem1 = create_memory_with_importance("similar-1", "Machine learning", 0.5);
    mem1.embedding = vec![1.0; 384];
    store.insert(&mem1).await.unwrap();

    let mut mem2 = create_memory_with_importance("similar-2", "Deep learning", 0.5);
    mem2.embedding = vec![0.95; 384];
    store.insert(&mem2).await.unwrap();

    let mut mem3 = create_memory_with_importance("dissimilar", "Cooking recipes", 0.5);
    mem3.embedding = vec![0.1; 384];
    store.insert(&mem3).await.unwrap();

    // Query with embedding close to mem1 and mem2
    let query_vector = vec![0.98; 384];

    let results = store.search(&query_vector, &Filters::default(), 10).await.unwrap();

    // Should return memories in order of similarity
    assert_eq!(results.len(), 3);

    // Scores must be in descending order (highest similarity first)
    assert!(results[0].score >= results[1].score, "First result should have highest score");
    assert!(results[1].score >= results[2].score, "Results should be ordered by score");

    // Most similar should be one of the "similar" memories
    assert!(results[0].memory.id.starts_with("similar"));

    // Verify scores are NOT all the same — the fix for #1 should produce differentiated scores
    assert!(results[0].score > results[2].score,
        "Similar memories must score higher than dissimilar ones: {} vs {}",
        results[0].score, results[2].score);

    // Scores should be in (0, 1] range (1/(1+d) where d >= 0)
    for r in &results {
        assert!(r.score > 0.0 && r.score <= 1.0, "Score {} out of valid range (0, 1]", r.score);
    }
}
