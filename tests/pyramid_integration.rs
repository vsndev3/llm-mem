//! Pyramid search integration tests with real LanceDB embeddings
//!
//! Validates the interaction between real embedding vectors, layer-scoped filters,
//! cross-layer deduplication, and slot allocation across all modes.

use async_trait::async_trait;
use llm_mem::{
    VectorStore,
    config::MemoryConfig,
    error::Result,
    llm::{
        ClientStatus, ConversationAnalysis, DeduplicationResult, DetailedFactExtraction,
        EntityExtraction, ImportanceScore, KeywordExtraction, LLMClient, LanguageDetection,
        MemoryClassification, MemoryEnhancement, StructuredFactExtraction, SummaryResult,
    },
    memory::MemoryManager,
    search::{PyramidAllocationMode, PyramidConfig},
    types::{Filters, LayerInfo, Memory, MemoryMetadata, MemoryState, MemoryType},
};
use std::collections::HashMap;
use tempfile::TempDir;
use uuid::Uuid;

// ─── Deterministic Embedding LLM Client ─────────────────────────────────────
/// Produces content-aware embeddings (same as lancedb_integration.rs) so that
/// similar content produces similar vectors and dissimilar content produces
/// dissimilar vectors. This exercises the real cosine similarity path.

#[derive(Clone)]
struct DetEmbedClient {
    dimension: usize,
}

impl DetEmbedClient {
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
impl LLMClient for DetEmbedClient {
    async fn complete(&self, prompt: &str) -> Result<String> {
        Ok("BottomHeavy".to_string())
    }
    async fn complete_with_grammar(&self, _p: &str, _g: &str) -> Result<String> {
        Ok("{}".to_string())
    }
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.make_embedding(text))
    }
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.make_embedding(t)).collect())
    }
    async fn extract_keywords(&self, content: &str) -> Result<Vec<String>> {
        Ok(content.split_whitespace().take(3).map(|s| s.to_lowercase()).collect())
    }
    async fn summarize(&self, content: &str, _ml: Option<usize>) -> Result<String> {
        Ok(content[..content.len().min(50)].to_string())
    }
    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
    async fn extract_structured_facts(&self, _p: &str) -> Result<StructuredFactExtraction> {
        Ok(StructuredFactExtraction { facts: vec![] })
    }
    async fn extract_detailed_facts(&self, _p: &str) -> Result<DetailedFactExtraction> {
        Ok(DetailedFactExtraction { facts: vec![] })
    }
    async fn extract_keywords_structured(&self, _p: &str) -> Result<KeywordExtraction> {
        Ok(KeywordExtraction { keywords: vec![] })
    }
    async fn classify_memory(&self, _p: &str) -> Result<MemoryClassification> {
        Ok(MemoryClassification {
            memory_type: "Factual".into(),
            confidence: 1.0,
            reasoning: "".into(),
        })
    }
    async fn score_importance(&self, _p: &str) -> Result<ImportanceScore> {
        Ok(ImportanceScore { score: 0.5, reasoning: "".into() })
    }
    async fn check_duplicates(&self, _p: &str) -> Result<DeduplicationResult> {
        Ok(DeduplicationResult { is_duplicate: false, similarity_score: 0.0, original_memory_id: None })
    }
    async fn generate_summary(&self, _p: &str) -> Result<SummaryResult> {
        Ok(SummaryResult { summary: "".into(), key_points: vec![] })
    }
    async fn detect_language(&self, _p: &str) -> Result<LanguageDetection> {
        Ok(LanguageDetection { language: "en".into(), confidence: 1.0 })
    }
    async fn extract_entities(&self, _p: &str) -> Result<EntityExtraction> {
        Ok(EntityExtraction { entities: vec![] })
    }
    async fn analyze_conversation(&self, _p: &str) -> Result<ConversationAnalysis> {
        Ok(ConversationAnalysis { topics: vec![], sentiment: "".into(), user_intent: "".into(), key_information: vec![] })
    }
    async fn extract_metadata_enrichment(&self, _p: &str) -> Result<llm_mem::llm::MetadataEnrichment> {
        Ok(llm_mem::llm::MetadataEnrichment { summary: "".into(), keywords: vec![] })
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
        ClientStatus {
            backend: "det-embed".into(),
            state: "ready".into(),
            llm_model: "mock".into(),
            embedding_model: "det-embed".into(),
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
    async fn enhance_memory_unified(&self, _p: &str) -> Result<MemoryEnhancement> {
        Ok(MemoryEnhancement {
            memory_type: "Semantic".into(),
            summary: String::new(),
            keywords: vec![],
            entities: vec![],
            topics: vec![],
        })
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn make_memory(
    id: &str,
    content: &str,
    layer: i32,
    sources: Vec<Uuid>,
    client: &DetEmbedClient,
) -> Memory {
    let meta = MemoryMetadata::new(MemoryType::Factual)
        .with_layer(LayerInfo::custom(
            layer,
            match layer {
                0 => "raw_content".into(),
                1 => "structural".into(),
                2 => "semantic".into(),
                3 => "concept".into(),
                _ => format!("layer_{}", layer),
            },
        ))
        .with_abstraction_sources(sources);

    Memory {
        id: id.to_string(),
        content: Some(content.to_string()),
        content_meta: Default::default(),
        derived_data: HashMap::new(),
        relations: HashMap::new(),
        embedding: client.make_embedding(content),
        metadata: meta,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        context_embeddings: None,
        relation_embeddings: None,
    }
}

async fn make_manager(temp_dir: &TempDir) -> (MemoryManager, DetEmbedClient) {
    let dim = 384;
    let client = DetEmbedClient::new(dim);

    let config = llm_mem::lance_store::LanceDBConfig {
        table_name: "pyramid_test".into(),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: dim,
    };
    let store: Box<dyn VectorStore> = Box::new(
        llm_mem::lance_store::LanceDBStore::new(config).await.unwrap(),
    );

    let mem_cfg = MemoryConfig {
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
        search_similarity_threshold: Some(0.0),
        use_llm_query_classification: false,
    };

    let mgr = MemoryManager::new(store, Box::new(client.clone()), mem_cfg);
    (mgr, client)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_pyramid_multi_layer_results() {
    let temp_dir = TempDir::new().unwrap();
    let (mgr, client) = make_manager(&temp_dir).await;

    // Store memories across L0-L2 with semantically related content
    let _l0_ids: Vec<Uuid> = vec![];
    for i in 0..5 {
        let id = format!("l0-{}", i);
        let content = format!("The user enjoys eating {} on weekends", match i {
            0 => "pizza",
            1 => "pasta",
            2 => "Italian food",
            3 => "sushi",
            4 => "Chinese cuisine",
            _ => unreachable!(),
        });
        let mem = make_memory(&id, &content, 0, vec![], &client);
        mgr.store_memory(mem).await.unwrap();
    }

    // L1: structural summaries
    for i in 0..3 {
        let id = format!("l1-{}", i);
        let content = format!("User has preferences for {} cuisine", match i {
            0 => "Italian",
            1 => "Asian",
            2 => "diverse",
            _ => unreachable!(),
        });
        let mem = make_memory(&id, &content, 1, vec![], &client);
        mgr.store_memory(mem).await.unwrap();
    }

    // L2: semantic abstractions
    for i in 0..2 {
        let id = format!("l2-{}", i);
        let content = match i {
            0 => "User dietary preferences favor international cuisine",
            _ => "User food preferences indicate diverse culinary interests",
        };
        let mem = make_memory(&id, &content, 2, vec![], &client);
        mgr.store_memory(mem).await.unwrap();
    }

    mgr.refresh_layer_manifest().await.unwrap();

    // Query
    let results = mgr
        .search_pyramid(
            "What does the user like to eat?",
            &Filters::default(),
            10,
            &PyramidConfig::default(),
        )
        .await
        .unwrap();

    // Should return results from multiple layers
    let layers: std::collections::HashSet<_> = results.iter().map(|r| r.layer).collect();
    assert!(
        layers.len() >= 2,
        "Expected results from at least 2 layers, got: {:?}",
        layers
    );

    // Results should be sorted by score descending
    for i in 1..results.len() {
        assert!(
            results[i - 1].memory.score >= results[i].memory.score,
            "Results should be sorted by score descending"
        );
    }

    assert!(!results.is_empty(), "Expected non-empty results");
}

#[tokio::test]
async fn test_pyramid_bottom_heavy_favors_l0() {
    let temp_dir = TempDir::new().unwrap();
    let (mgr, client) = make_manager(&temp_dir).await;

    // 5 memories per layer L0-L2
    for layer in [0, 1, 2] {
        for i in 0..5 {
            let id = format!("l{}-{}", layer, i);
            let content = format!("Topic {} detail {}", layer, i);
            let mem = make_memory(&id, &content, layer, vec![], &client);
            mgr.store_memory(mem).await.unwrap();
        }
    }

    mgr.refresh_layer_manifest().await.unwrap();

    let config = PyramidConfig {
        mode: PyramidAllocationMode::BottomHeavy,
        ..PyramidConfig::default()
    };

    let results = mgr
        .search_pyramid("topic detail", &Filters::default(), 10, &config)
        .await
        .unwrap();

    let layer_counts: HashMap<i32, usize> = results
        .iter()
        .fold(HashMap::new(), |mut acc, r| {
            *acc.entry(r.layer).or_insert(0) += 1;
            acc
        });

    let l0 = *layer_counts.get(&0).unwrap_or(&0);
    let l2 = *layer_counts.get(&2).unwrap_or(&0);

    assert!(
        l0 >= l2,
        "BottomHeavy: L0 ({}) should have >= L2 ({}) results",
        l0,
        l2
    );
}

#[tokio::test]
async fn test_pyramid_top_heavy_favors_higher_layers() {
    let temp_dir = TempDir::new().unwrap();
    let (mgr, client) = make_manager(&temp_dir).await;

    for layer in [0, 1, 2] {
        for i in 0..5 {
            let id = format!("l{}-{}", layer, i);
            let content = format!("Topic {} detail {}", layer, i);
            let mem = make_memory(&id, &content, layer, vec![], &client);
            mgr.store_memory(mem).await.unwrap();
        }
    }

    mgr.refresh_layer_manifest().await.unwrap();

    let config = PyramidConfig {
        mode: PyramidAllocationMode::TopHeavy,
        ..PyramidConfig::default()
    };

    let results = mgr
        .search_pyramid("topic detail", &Filters::default(), 10, &config)
        .await
        .unwrap();

    let layer_counts: HashMap<i32, usize> = results
        .iter()
        .fold(HashMap::new(), |mut acc, r| {
            *acc.entry(r.layer).or_insert(0) += 1;
            acc
        });

    let l0 = *layer_counts.get(&0).unwrap_or(&0);
    let l2 = *layer_counts.get(&2).unwrap_or(&0);

    assert!(
        l2 >= l0,
        "TopHeavy: L2 ({}) should have >= L0 ({}) results",
        l2,
        l0
    );
}

#[tokio::test]
async fn test_pyramid_balanced_distribution() {
    let temp_dir = TempDir::new().unwrap();
    let (mgr, client) = make_manager(&temp_dir).await;

    for layer in [0, 1, 2] {
        for i in 0..5 {
            let id = format!("l{}-{}", layer, i);
            let content = format!("Topic {} detail {}", layer, i);
            let mem = make_memory(&id, &content, layer, vec![], &client);
            mgr.store_memory(mem).await.unwrap();
        }
    }

    mgr.refresh_layer_manifest().await.unwrap();

    let config = PyramidConfig {
        mode: PyramidAllocationMode::Balanced,
        ..PyramidConfig::default()
    };

    let results = mgr
        .search_pyramid("topic detail", &Filters::default(), 9, &config)
        .await
        .unwrap();

    let layer_counts: HashMap<i32, usize> = results
        .iter()
        .fold(HashMap::new(), |mut acc, r| {
            *acc.entry(r.layer).or_insert(0) += 1;
            acc
        });

    // Each layer should have ~3 results (9 / 3 layers)
    for layer in [0, 1, 2] {
        let count = *layer_counts.get(&layer).unwrap_or(&0);
        assert!(
            count >= 2 && count <= 4,
            "Balanced: Layer {} should have 2-4 results, got {}",
            layer,
            count
        );
    }
}

#[tokio::test]
async fn test_pyramid_cross_layer_deduplication() {
    let temp_dir = TempDir::new().unwrap();
    let (mgr, client) = make_manager(&temp_dir).await;

    // Create L0 memory
    let l0_uuid = Uuid::new_v4();
    let mem_l0 = make_memory(
        &l0_uuid.to_string(),
        "The user enjoys Italian cuisine",
        0,
        vec![],
        &client,
    );
    mgr.store_memory(mem_l0).await.unwrap();

    // Create L1 memory that abstracts from L0
    let mem_l1 = make_memory(
        &Uuid::new_v4().to_string(),
        "User has Italian food preferences",
        1,
        vec![l0_uuid],
        &client,
    );
    mgr.store_memory(mem_l1).await.unwrap();

    mgr.refresh_layer_manifest().await.unwrap();

    let results = mgr
        .search_pyramid(
            "What does the user like to eat?",
            &Filters::default(),
            10,
            &PyramidConfig::default(),
        )
        .await
        .unwrap();

    // Both L0 and L1 may appear, but the L0 source should have a slight edge
    let l0_results: Vec<_> = results.iter().filter(|r| r.layer == 0).collect();
    let l1_results: Vec<_> = results.iter().filter(|r| r.layer == 1).collect();

    if !l0_results.is_empty() && !l1_results.is_empty() {
        // The source should rank higher or equal after dedup boost
        let best_l0 = l0_results.iter().map(|r| r.memory.score).fold(f32::MIN, f32::max);
        let best_l1 = l1_results.iter().map(|r| r.memory.score).fold(f32::MIN, f32::max);
        // After dedup boost (1.05x), source should be >= abstraction when scores were equal
        assert!(
            best_l0 >= best_l1 * 0.99,
            "Source L0 ({}) should rank close to or above abstraction L1 ({})",
            best_l0,
            best_l1
        );
    }

    // No duplicate IDs
    let ids: std::collections::HashSet<_> =
        results.iter().map(|r| r.memory.memory.id.clone()).collect();
    assert_eq!(
        ids.len(),
        results.len(),
        "No duplicate memory IDs in results"
    );
}

#[tokio::test]
async fn test_pyramid_none_mode_flat_search() {
    let temp_dir = TempDir::new().unwrap();
    let (mgr, client) = make_manager(&temp_dir).await;

    for layer in [0, 1, 2] {
        for i in 0..3 {
            let id = format!("l{}-{}", layer, i);
            let content = format!("Flat search topic {}", i);
            let mem = make_memory(&id, &content, layer, vec![], &client);
            mgr.store_memory(mem).await.unwrap();
        }
    }

    mgr.refresh_layer_manifest().await.unwrap();

    let config = PyramidConfig {
        mode: PyramidAllocationMode::None,
        ..PyramidConfig::default()
    };

    let results = mgr
        .search_pyramid("flat search topic", &Filters::default(), 5, &config)
        .await
        .unwrap();

    // All results should come from flat search phase
    assert!(results.iter().all(|r| r.search_phase == "flat"));
    assert!(results.len() <= 5);

    // Results should be sorted by score descending
    for i in 1..results.len() {
        assert!(
            results[i - 1].memory.score >= results[i].memory.score,
            "Flat search results should be sorted descending"
        );
    }
}

#[tokio::test]
async fn test_pyramid_raw_scores_preserved_across_layers() {
    let temp_dir = TempDir::new().unwrap();
    let (mgr, client) = make_manager(&temp_dir).await;

    // Memories with very different content to produce different raw scores
    let mem0 = make_memory("l0-a", "Machine learning algorithms are powerful", 0, vec![], &client);
    mgr.store_memory(mem0).await.unwrap();

    let mem1 = make_memory("l1-a", "AI methods include ML", 1, vec![], &client);
    mgr.store_memory(mem1).await.unwrap();

    let mem2 = make_memory("l2-a", "Technology concepts", 2, vec![], &client);
    mgr.store_memory(mem2).await.unwrap();

    mgr.refresh_layer_manifest().await.unwrap();

    let results = mgr
        .search_pyramid(
            "machine learning algorithms",
            &Filters::default(),
            10,
            &PyramidConfig::default(),
        )
        .await
        .unwrap();

    // Scores should be in valid cosine similarity range and NOT all identical
    // (proving raw scores are used, not re-normalized per layer)
    assert!(!results.is_empty());

    for r in &results {
        // Raw cosine scores from LanceDB are in [0, 1] after 1/(1+d) conversion
        assert!(
            r.memory.score > 0.0 && r.memory.score <= 1.0,
            "Score {} out of range",
            r.memory.score
        );
    }

    // The most relevant memory (L0 with exact content match) should rank highest
    if results.len() >= 2 {
        assert!(
            results[0].memory.score > results[results.len() - 1].memory.score,
            "Best result should score higher than worst"
        );
    }
}

#[tokio::test]
async fn test_pyramid_zero_relaxation_identical_thresholds() {
    let temp_dir = TempDir::new().unwrap();
    let (mgr, client) = make_manager(&temp_dir).await;

    for layer in [0, 1, 2, 3] {
        for i in 0..3 {
            let id = format!("l{}-{}", layer, i);
            let content = format!("Threshold test content layer {} item {}", layer, i);
            let mem = make_memory(&id, &content, layer, vec![], &client);
            mgr.store_memory(mem).await.unwrap();
        }
    }

    mgr.refresh_layer_manifest().await.unwrap();

    // relaxation = 0.0 means identical threshold across all layers
    let config = PyramidConfig {
        mode: PyramidAllocationMode::Balanced,
        layer_threshold_relaxation: 0.0,
        ..PyramidConfig::default()
    };

    assert!(config.validate().is_ok());

    let results = mgr
        .search_pyramid(
            "threshold test content",
            &Filters::default(),
            12,
            &config,
        )
        .await
        .unwrap();

    // Should return results from multiple layers
    let layers: std::collections::HashSet<_> = results.iter().map(|r| r.layer).collect();
    assert!(layers.len() >= 2, "Expected results from multiple layers with zero relaxation");
}

#[tokio::test]
async fn test_pyramid_empty_store() {
    let temp_dir = TempDir::new().unwrap();
    let (mgr, _) = make_manager(&temp_dir).await;

    let results = mgr
        .search_pyramid(
            "anything",
            &Filters::default(),
            10,
            &PyramidConfig::default(),
        )
        .await
        .unwrap();

    assert!(results.is_empty());
}

#[tokio::test]
async fn test_pyramid_single_layer_only() {
    let temp_dir = TempDir::new().unwrap();
    let (mgr, client) = make_manager(&temp_dir).await;

    for i in 0..5 {
        let id = format!("l0-{}", i);
        let content = format!("Single layer memory {}", i);
        let mem = make_memory(&id, &content, 0, vec![], &client);
        mgr.store_memory(mem).await.unwrap();
    }

    let results = mgr
        .search_pyramid(
            "single layer memory",
            &Filters::default(),
            10,
            &PyramidConfig::default(),
        )
        .await
        .unwrap();

    assert!(!results.is_empty());
    assert!(results.iter().all(|r| r.layer == 0));
}

#[tokio::test]
async fn test_pyramid_custom_layer_weights() {
    let temp_dir = TempDir::new().unwrap();
    let (mgr, client) = make_manager(&temp_dir).await;

    for layer in [0, 1, 2] {
        for i in 0..5 {
            let id = format!("l{}-{}", layer, i);
            let content = format!("Weighted search topic {}", i);
            let mem = make_memory(&id, &content, layer, vec![], &client);
            mgr.store_memory(mem).await.unwrap();
        }
    }

    mgr.refresh_layer_manifest().await.unwrap();

    // Heavily weight L2
    let mut weights = HashMap::new();
    weights.insert(0, 0.5);
    weights.insert(1, 0.5);
    weights.insert(2, 4.0);

    let config = PyramidConfig {
        mode: PyramidAllocationMode::Balanced,
        layer_weights: weights,
        ..PyramidConfig::default()
    };

    let results = mgr
        .search_pyramid("weighted search topic", &Filters::default(), 10, &config)
        .await
        .unwrap();

    let layer_counts: HashMap<i32, usize> = results
        .iter()
        .fold(HashMap::new(), |mut acc, r| {
            *acc.entry(r.layer).or_insert(0) += 1;
            acc
        });

    let l2 = *layer_counts.get(&2).unwrap_or(&0);
    let l0 = *layer_counts.get(&0).unwrap_or(&0);

    assert!(
        l2 >= l0,
        "Custom weights: L2 ({}) should have >= L0 ({}) when L2 is heavily weighted",
        l2,
        l0
    );
}

#[tokio::test]
async fn test_pyramid_graph_refinement_phase() {
    let temp_dir = TempDir::new().unwrap();
    let (mgr, client) = make_manager(&temp_dir).await;

    // Create memories with relations
    let mem1 = make_memory("rel-1", "Graph search finds related memories", 0, vec![], &client);
    mgr.store_memory(mem1).await.unwrap();

    let mem2 = make_memory("rel-2", "Related memories are discovered via graph", 0, vec![], &client);
    mgr.store_memory(mem2).await.unwrap();

    let results = mgr
        .search_pyramid(
            "graph search related memories",
            &Filters::default(),
            10,
            &PyramidConfig::default(),
        )
        .await
        .unwrap();

    // Results should include entries from pyramid phase
    assert!(!results.is_empty());
    let phases: std::collections::HashSet<_> = results.iter().map(|r| r.search_phase.clone()).collect();
    // At least the "pyramid" phase should be present
    assert!(
        phases.contains(&"pyramid".to_string()),
        "Expected pyramid phase in results, got: {:?}",
        phases
    );
}
