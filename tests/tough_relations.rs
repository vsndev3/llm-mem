#![cfg(feature = "local")]
/// Tough integration tests: real LanceDB store, real fastembed embeddings.
///
/// Run with: cargo tough-tests

use llm_mem::{
    LanceDBConfig, LanceDBStore,
    config::MemoryConfig,
    error::Result,
    llm::{
        ClientStatus, ConversationAnalysis, DeduplicationResult, DetailedFactExtraction,
        EntityExtraction, ImportanceScore, KeywordExtraction, LLMClient, LanguageDetection,
        LlmPriority, MemoryClassification, MemoryEnhancement, StructuredFactExtraction,
        SummaryResult,
    },
    memory::{MemoryManager, StoreOptions},
    types::MemoryMetadata,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ─── Real Embedding + Mock LLM Client ──────────────────────────────────────
//
// Uses real fastembed (all-MiniLM-L6-v2, 384-dim) for semantic vector search.
// LLM completion methods are mocked — they are not exercised by relation tests.

#[derive(Clone)]
struct FastEmbedClient {
    embedding: Arc<Mutex<fastembed::TextEmbedding>>,
}

impl FastEmbedClient {
    fn new() -> Self {
        let embed_options = fastembed::InitOptions::new(fastembed::EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(false);
        let embedding = fastembed::TextEmbedding::try_new(embed_options)
            .expect("Failed to initialize fastembed (all-MiniLM-L6-v2)");
        Self { embedding: Arc::new(Mutex::new(embedding)) }
    }
}

#[async_trait]
impl LLMClient for FastEmbedClient {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let emb = Arc::clone(&self.embedding);
        let t = text.to_string();
        let results = tokio::task::spawn_blocking(move || {
            emb.lock().unwrap().embed(vec![t], None)
                .map_err(|e| llm_mem::error::MemoryError::Embedding(format!("Fastembed error: {}", e)))
        }).await.unwrap()?;
        Ok(results.into_iter().next().unwrap())
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let emb = Arc::clone(&self.embedding);
        let t: Vec<String> = texts.to_vec();
        tokio::task::spawn_blocking(move || {
            emb.lock().unwrap().embed(t, None)
                .map_err(|e| llm_mem::error::MemoryError::Embedding(format!("Fastembed batch error: {}", e)))
        }).await.unwrap()
    }

    // ── Unused mock methods ──────────────────────────────────────────────

    async fn complete(&self, _prompt: &str) -> Result<String> { Ok(String::new()) }
    async fn complete_with_grammar(&self, _prompt: &str, _grammar: &str) -> Result<String> {
        Ok(r#"{"summary":"mock"}"#.into())
    }
    async fn extract_keywords(&self, _content: &str) -> Result<Vec<String>> { Ok(vec![]) }
    async fn summarize(&self, _content: &str, _max_length: Option<usize>) -> Result<String> {
        Ok(String::new())
    }
    async fn health_check(&self) -> Result<bool> { Ok(true) }
    async fn extract_structured_facts(&self, _prompt: &str) -> Result<StructuredFactExtraction> {
        Ok(StructuredFactExtraction { facts: vec![] })
    }
    async fn extract_detailed_facts(&self, _prompt: &str) -> Result<DetailedFactExtraction> {
        Ok(DetailedFactExtraction { facts: vec![] })
    }
    async fn extract_keywords_structured(&self, _prompt: &str) -> Result<KeywordExtraction> {
        Ok(KeywordExtraction { keywords: vec![] })
    }
    async fn classify_memory(&self, _prompt: &str) -> Result<MemoryClassification> {
        Ok(MemoryClassification { memory_type: "Factual".into(), confidence: 0.0, reasoning: String::new() })
    }
    async fn score_importance(&self, _prompt: &str) -> Result<ImportanceScore> {
        Ok(ImportanceScore { score: 0.0, reasoning: String::new() })
    }
    async fn check_duplicates(&self, _prompt: &str) -> Result<DeduplicationResult> {
        Ok(DeduplicationResult { is_duplicate: false, similarity_score: 0.0, original_memory_id: None })
    }
    async fn generate_summary(&self, _prompt: &str) -> Result<SummaryResult> {
        Ok(SummaryResult { summary: String::new(), key_points: vec![] })
    }
    async fn detect_language(&self, _prompt: &str) -> Result<LanguageDetection> {
        Ok(LanguageDetection { language: "English".into(), confidence: 0.0 })
    }
    async fn extract_entities(&self, _prompt: &str) -> Result<EntityExtraction> {
        Ok(EntityExtraction { entities: vec![] })
    }
    async fn analyze_conversation(&self, _prompt: &str) -> Result<ConversationAnalysis> {
        Ok(ConversationAnalysis {
            topics: vec![], sentiment: "neutral".into(),
            user_intent: "informational".into(), key_information: vec![],
        })
    }
    async fn extract_metadata_enrichment(&self, _prompt: &str) -> Result<llm_mem::llm::MetadataEnrichment> {
        Ok(llm_mem::llm::MetadataEnrichment { summary: String::new(), keywords: vec![] })
    }
    async fn extract_metadata_enrichment_batch(&self, _texts: &[String]) -> Result<Vec<Result<llm_mem::llm::MetadataEnrichment>>> {
        Ok(vec![])
    }
    async fn complete_batch(&self, _prompts: &[String]) -> Result<Vec<Result<String>>> {
        Ok(vec![])
    }
    fn get_status(&self) -> ClientStatus {
        ClientStatus {
            backend: "real-embed+mock-llm".into(), state: "ready".into(),
            llm_model: "mock".into(), embedding_model: "all-MiniLM-L6-v2".into(),
            llm_available: true, embedding_available: true, last_llm_success: None,
            last_embedding_success: None, last_error: None,
            total_llm_calls: 0, total_embedding_calls: 0, total_prompt_tokens: 0,
            total_completion_tokens: 0, details: HashMap::new(),
        }
    }
    fn batch_config(&self) -> (usize, u32) { (10, 4096) }
    async fn enhance_memory_unified(&self, _prompt: &str) -> Result<MemoryEnhancement> {
        Ok(MemoryEnhancement {
            memory_type: "Semantic".into(), summary: String::new(),
            keywords: vec![], entities: vec![], topics: vec![],
        })
    }
    async fn describe_image(&self, _image_bytes: &[u8], _mime_type: &str) -> Result<String> {
        Err(llm_mem::error::MemoryError::LLM("FastEmbedClient: vision not available".into()))
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

const DIM: usize = 384;

async fn make_client() -> FastEmbedClient {
    FastEmbedClient::new()
}

async fn make_store() -> LanceDBStore {
    let tmp = tempfile::tempdir().unwrap();
    LanceDBStore::new(LanceDBConfig {
        table_name: "tough-rel".into(),
        database_path: tmp.path().to_path_buf(),
        embedding_dimension: DIM,
    })
    .await
    .unwrap()
}

async fn make_manager_with_auto_link() -> MemoryManager {
    MemoryManager::new(
        Box::new(make_store().await),
        Box::new(make_client().await),
        MemoryConfig {
            auto_link_threshold: 0.35,
            auto_link_max_relations: 10,
            auto_enhance: false,
            deduplicate: false,
            ..MemoryConfig::default()
        },
        None,
    )
}

async fn make_manager() -> MemoryManager {
    MemoryManager::new(
        Box::new(make_store().await),
        Box::new(make_client().await),
        MemoryConfig {
            auto_enhance: false,
            deduplicate: false,
            auto_link_threshold: 0.75,
            auto_link_max_relations: 10,
            ..MemoryConfig::default()
        },
        None,
    )
}

// ─── Auto-link tests ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_tough_auto_link_creates_relations() {
    let mgr = make_manager_with_auto_link().await;
    let user_meta = MemoryMetadata::new().with_user_id("u1".into());

    mgr.store(
        "Rust is a systems programming language with zero-cost abstractions".to_string(),
        user_meta.clone(),
    )
    .await
    .unwrap();

    let options = StoreOptions {
        llm_priority: LlmPriority::Interactive,
        auto_link: Some(true),
        ..StoreOptions::default()
    };
    let id_new = mgr
        .store_with_options(
            "Rust provides memory safety without garbage collection".to_string(),
            user_meta,
            options,
        )
        .await
        .unwrap();

    let mem_new = mgr.get(&id_new).await.unwrap().unwrap();
    let ref_count = mem_new.metadata.relations.iter()
        .filter(|r| r.relation == "references").count();
    assert!(ref_count >= 1, "Auto-link should create references, got {}", ref_count);
}

#[tokio::test]
async fn test_tough_auto_link_disabled() {
    let mgr = make_manager_with_auto_link().await;
    let user_meta = MemoryMetadata::new().with_user_id("u1".into());

    mgr.store(
        "Rust is a systems programming language with zero-cost abstractions".to_string(),
        user_meta.clone(),
    )
    .await
    .unwrap();

    let options = StoreOptions {
        llm_priority: LlmPriority::Interactive,
        auto_link: Some(false),
        ..StoreOptions::default()
    };
    let id_new = mgr
        .store_with_options(
            "Rust provides memory safety without garbage collection".to_string(),
            user_meta,
            options,
        )
        .await
        .unwrap();

    let mem_new = mgr.get(&id_new).await.unwrap().unwrap();
    let auto_rels: Vec<_> = mem_new.metadata.relations.iter()
        .filter(|r| r.relation == "references").collect();
    assert!(auto_rels.is_empty(), "Auto-link disabled but found: {:?}", auto_rels);
}

#[tokio::test]
async fn test_tough_auto_link_user_scoping() {
    let mgr = make_manager_with_auto_link().await;

    mgr.store(
        "Rust is a systems programming language with zero-cost abstractions".to_string(),
        MemoryMetadata::new().with_user_id("rust_user".into()),
    )
    .await
    .unwrap();

    let options = StoreOptions {
        llm_priority: LlmPriority::Interactive,
        auto_link: Some(true),
        ..StoreOptions::default()
    };
    let id_new = mgr
        .store_with_options(
            "The weather in Paris is sunny today with mild temperatures".to_string(),
            MemoryMetadata::new().with_user_id("weather_user".into()),
            options,
        )
        .await
        .unwrap();

    let mem_new = mgr.get(&id_new).await.unwrap().unwrap();
    let auto_rels: Vec<_> = mem_new.metadata.relations.iter()
        .filter(|r| r.relation == "references").collect();
    assert!(auto_rels.is_empty(), "Different users should not auto-link: {:?}", auto_rels);
}

#[tokio::test]
async fn test_tough_auto_link_multiple_similar() {
    let mgr = make_manager_with_auto_link().await;
    let user_meta = MemoryMetadata::new().with_user_id("u1".into());

    mgr.store("Python is a popular dynamically typed language".to_string(), user_meta.clone()).await.unwrap();
    mgr.store("JavaScript dominates web development".to_string(), user_meta.clone()).await.unwrap();
    mgr.store("Go is a statically typed compiled language".to_string(), user_meta).await.unwrap();

    let options = StoreOptions {
        llm_priority: LlmPriority::Interactive,
        auto_link: Some(true),
        ..StoreOptions::default()
    };
    let id_new = mgr
        .store_with_options(
            "Programming languages like Python, JavaScript, and Go each have unique strengths".to_string(),
            MemoryMetadata::new().with_user_id("u1".into()),
            options,
        )
        .await
        .unwrap();

    let mem_new = mgr.get(&id_new).await.unwrap().unwrap();
    let ref_count = mem_new.metadata.relations.iter()
        .filter(|r| r.relation == "references").count();
    assert!(ref_count >= 1, "Multiple similar memories should produce links, got {}", ref_count);
}

// ─── E2E cycle tests ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_tough_e2e_auto_link_force_link_remove_cycle() {
    let mgr = make_manager_with_auto_link().await;
    let user_meta = MemoryMetadata::new().with_user_id("u1".into());

    let id_base = mgr
        .store(
            "Rust has a powerful macro system for metaprogramming".to_string(),
            user_meta.clone(),
        )
        .await
        .unwrap();

    let options = StoreOptions {
        llm_priority: LlmPriority::Interactive,
        auto_link: Some(true),
        ..StoreOptions::default()
    };
    let id_new = mgr
        .store_with_options(
            "Procedural macros in Rust enable custom derive implementations".to_string(),
            user_meta,
            options,
        )
        .await
        .unwrap();

    let mem_new = mgr.get(&id_new).await.unwrap().unwrap();
    assert!(
        mem_new.metadata.relations.iter().any(|r| r.relation == "references"),
        "Step 2: auto-link should create references relation"
    );

    let mut source = mgr.get(&id_new).await.unwrap().unwrap();
    source.metadata.relations.push(llm_mem::types::Relation {
        source: id_new.clone(),
        relation: "depends_on".into(),
        target: id_base.clone(),
        strength: Some(0.95),
    });
    mgr.update_memory(&source).await.unwrap();

    let reloaded = mgr.get(&id_new).await.unwrap().unwrap();
    assert!(reloaded.metadata.relations.iter().any(|r| r.relation == "references"));
    assert!(reloaded.metadata.relations.iter().any(|r| r.relation == "depends_on"));

    let mut cleaned = reloaded;
    cleaned.metadata.relations.retain(|r| !(r.relation == "references" && r.target == id_base));
    mgr.update_memory(&cleaned).await.unwrap();

    let final_mem = mgr.get(&id_new).await.unwrap().unwrap();
    assert!(
        !final_mem.metadata.relations.iter().any(|r| r.relation == "references" && r.target == id_base),
        "Step 4: auto-link references should be removed"
    );
    assert!(
        final_mem.metadata.relations.iter().any(|r| r.relation == "depends_on"),
        "Step 4: manual depends_on should survive"
    );
}

#[tokio::test]
async fn test_tough_force_link_and_remove_relation() {
    let mgr = make_manager().await;
    let meta = MemoryMetadata::new().with_user_id("u1".into());

    let id_a = mgr.store("Memory A: about databases".to_string(), meta.clone()).await.unwrap();
    let id_b = mgr.store("Memory B: about SQL queries".to_string(), meta).await.unwrap();

    let mut source = mgr.get(&id_a).await.unwrap().unwrap();
    source.metadata.relations.push(llm_mem::types::Relation {
        source: id_a.clone(), relation: "depends_on".into(),
        target: id_b.clone(), strength: Some(0.9),
    });
    mgr.update_memory(&source).await.unwrap();

    let reloaded = mgr.get(&id_a).await.unwrap().unwrap();
    assert!(reloaded.metadata.relations.iter().any(|r| r.relation == "depends_on" && r.target == id_b));

    let mut cleaned = reloaded;
    cleaned.metadata.relations.retain(|r| !(r.relation == "depends_on" && r.target == id_b));
    mgr.update_memory(&cleaned).await.unwrap();

    let final_mem = mgr.get(&id_a).await.unwrap().unwrap();
    assert!(!final_mem.metadata.relations.iter().any(|r| r.relation == "depends_on" && r.target == id_b));
}
