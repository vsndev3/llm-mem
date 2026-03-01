//! Retrieval accuracy evaluation for llm-mem.
//!
//! Measures how accurately the memory system retrieves relevant memories
//! for natural language queries, using real semantic embeddings.
//!
//! # Running
//!
//! All tests are `#[ignore]` — they won't run during normal `cargo test`.
//! Run them explicitly:
//!
//! ```bash
//! # Run all evaluation tests (requires ~90 MB embedding model download on first run)
//! cargo test --test evaluation -- --ignored --nocapture
//!
//! # Run a specific evaluation
//! cargo test --test evaluation evaluation_retrieval_accuracy -- --ignored --nocapture
//!
//! # Run only mock-LLM tests (fast, no GGUF model needed)
//! cargo test --test evaluation evaluation_retrieval -- --ignored --nocapture
//! cargo test --test evaluation evaluation_full_pipeline -- --ignored --nocapture
//! cargo test --test evaluation evaluation_type_filtered -- --ignored --nocapture
//! cargo test --test evaluation evaluation_similarity -- --ignored --nocapture
//!
//! # Run L2/L3 evaluation tests (real embeddings, mock LLM — fast)
//! cargo test --test evaluation evaluation_relation_filtered -- --ignored --nocapture
//! cargo test --test evaluation evaluation_context_retrieval -- --ignored --nocapture
//! cargo test --test evaluation evaluation_multivector_lifecycle -- --ignored --nocapture
//!
//! # Run only real-LLM tests (slow, requires GGUF model ~1.1 GB)
//! cargo test --test evaluation evaluation_real_llm -- --ignored --nocapture
//! ```
//!
//! # What it tests
//!
//! 1. **Pure retrieval accuracy** — Stores 15 memories with real embeddings,
//!    runs 15 queries, measures Recall@K, MRR, Precision@K.
//! 2. **Full pipeline accuracy** — Tests the complete MemoryManager pipeline
//!    (embedding + dedup + storage + search) with real embeddings.
//! 3. **Type-filtered retrieval** — Tests accuracy when filtering by memory type.
//! 4. **Similarity discrimination** — Tests if the system can distinguish
//!    semantically related but different memories.
//!
//! ## Level 2/3 tests (real embeddings, mock LLM)
//!
//! 4b. **Relation-filtered retrieval** — Stores memories with relation graph
//!     metadata, tests search narrowed by relation filters (Level 2).
//! 4c. **Context two-stage retrieval** — Stores memories with context tags,
//!     tests search_with_context two-stage pipeline (Level 3).
//! 4d. **Multi-vector lifecycle** — Tests store/get/update/delete with combined
//!     context + relation multi-vector embeddings (Level 3).
//!
//! ## Real LLM tests (require GGUF model)
//!
//! 5. **Real LLM pipeline accuracy** — Full end-to-end with real llama.cpp
//!    inference for fact extraction, classification, importance scoring.
//! 6. **Real LLM fact extraction** — Tests quality of LLM-extracted facts
//!    and keywords against expected terms.
//! 7. **Real LLM deduplication** — Tests LLM duplicate detection on known
//!    duplicate/distinct memory pairs.
//! 8. **Real LLM discrimination** — Hardest test: real LLM processing +
//!    semantically confusable queries.
//! 9. **Real LLM relations** — Full pipeline with relation metadata: storage,
//!    preservation, and relation-filtered search (Level 2).
//! 10. **Real LLM context retrieval** — Full pipeline with context tags:
//!     context embedding generation and two-stage retrieval (Level 3).
//! 11. **Real LLM combined L2+L3** — Production scenario with both relations
//!     and context tags, testing combined filtering and lifecycle ops.

#![cfg(feature = "local")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use llm_mem::{
    VectorStore,
    config::MemoryConfig,
    error::{MemoryError, Result},
    llm::{
        ClientStatus, ConversationAnalysis, DeduplicationResult, DetailedFactExtraction,
        EntityExtraction, ImportanceScore, KeywordExtraction, LLMClient, LanguageDetection,
        MemoryClassification, StructuredFactExtraction, SummaryResult,
    },
    memory::MemoryManager,
    types::{Filters, Memory, MemoryMetadata, MemoryType},
    vector_store::{VectorLiteConfig, VectorLiteStore},
};
use vectorlite::{IndexType, SimilarityMetric};

// ═══════════════════════════════════════════════════════════════════════════
//  Evaluation LLM Client — real embeddings, mock completions
// ═══════════════════════════════════════════════════════════════════════════

/// An LLM client that uses real fastembed embeddings (all-MiniLM-L6-v2, 384 dim)
/// but returns deterministic mock responses for all completion / extraction methods.
///
/// This allows testing embedding-based retrieval accuracy without requiring a
/// full LLM model (no GGUF download needed).
#[derive(Clone)]
struct EvalLLMClient {
    embedding: Arc<Mutex<TextEmbedding>>,
}

impl EvalLLMClient {
    fn new() -> Self {
        println!("  Initializing embedding model (all-MiniLM-L6-v2)...");

        let models_dir =
            std::env::var("LLM_MEM_MODELS_DIR").unwrap_or_else(|_| "llm-mem-models".to_string());
        println!("  Using embedding cache dir: {}", models_dir);

        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_show_download_progress(true)
                .with_cache_dir(std::path::PathBuf::from(models_dir)),
        )
        .expect("Failed to initialize embedding model — is ONNX Runtime available?");
        println!("  Embedding model ready (384 dimensions)");
        Self {
            embedding: Arc::new(Mutex::new(model)),
        }
    }
}

#[async_trait]
impl LLMClient for EvalLLMClient {
    async fn complete(&self, prompt: &str) -> Result<String> {
        Ok(format!("Mock: {}", &prompt[..prompt.len().min(50)]))
    }

    async fn complete_with_grammar(&self, _prompt: &str, _grammar: &str) -> Result<String> {
        Ok(r#"{"summary": "mock summary", "keywords": ["mock", "test"]}"#.to_string())
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let emb = Arc::clone(&self.embedding);
        let text = text.to_string();
        tokio::task::spawn_blocking(move || {
            let model = emb
                .lock()
                .map_err(|e| MemoryError::Embedding(format!("Embedding lock error: {}", e)))?;
            let mut results = model
                .embed(vec![text], None)
                .map_err(|e| MemoryError::Embedding(format!("Embedding failed: {}", e)))?;
            results
                .pop()
                .ok_or_else(|| MemoryError::Embedding("No embedding returned".into()))
        })
        .await
        .map_err(|e| MemoryError::Embedding(format!("Task join error: {}", e)))?
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let emb = Arc::clone(&self.embedding);
        let texts = texts.to_vec();
        tokio::task::spawn_blocking(move || {
            let model = emb
                .lock()
                .map_err(|e| MemoryError::Embedding(format!("Embedding lock error: {}", e)))?;
            model
                .embed(texts, None)
                .map_err(|e| MemoryError::Embedding(format!("Batch embedding failed: {}", e)))
        })
        .await
        .map_err(|e| MemoryError::Embedding(format!("Task join error: {}", e)))?
    }

    async fn extract_keywords(&self, content: &str) -> Result<Vec<String>> {
        Ok(content
            .split_whitespace()
            .take(5)
            .map(|s| s.to_lowercase())
            .collect())
    }

    async fn summarize(&self, content: &str, max_length: Option<usize>) -> Result<String> {
        let limit = max_length.unwrap_or(200);
        Ok(content.chars().take(limit).collect())
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }

    async fn extract_structured_facts(&self, _prompt: &str) -> Result<StructuredFactExtraction> {
        Ok(StructuredFactExtraction {
            facts: vec!["eval fact".into()],
        })
    }

    async fn extract_detailed_facts(&self, _prompt: &str) -> Result<DetailedFactExtraction> {
        Ok(DetailedFactExtraction { facts: vec![] })
    }

    async fn extract_keywords_structured(&self, _prompt: &str) -> Result<KeywordExtraction> {
        Ok(KeywordExtraction {
            keywords: vec!["eval".into()],
        })
    }

    async fn classify_memory(&self, _prompt: &str) -> Result<MemoryClassification> {
        Ok(MemoryClassification {
            memory_type: "factual".into(),
            confidence: 0.9,
            reasoning: "evaluation default".into(),
        })
    }

    async fn score_importance(&self, _prompt: &str) -> Result<ImportanceScore> {
        Ok(ImportanceScore {
            score: 0.5,
            reasoning: "evaluation default".into(),
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
            summary: "eval summary".into(),
            key_points: vec![],
        })
    }

    async fn detect_language(&self, _prompt: &str) -> Result<LanguageDetection> {
        Ok(LanguageDetection {
            language: "english".into(),
            confidence: 1.0,
        })
    }

    async fn extract_entities(&self, _prompt: &str) -> Result<EntityExtraction> {
        Ok(EntityExtraction { entities: vec![] })
    }

    async fn analyze_conversation(&self, _prompt: &str) -> Result<ConversationAnalysis> {
        Ok(ConversationAnalysis {
            topics: vec!["general".into()],
            sentiment: "neutral".into(),
            user_intent: "informational".into(),
            key_information: vec![],
        })
    }

    async fn extract_metadata_enrichment(&self, _prompt: &str) -> Result<llm_mem::llm::MetadataEnrichment> {
        Ok(llm_mem::llm::MetadataEnrichment {
            summary: "eval summary".into(),
            keywords: vec!["eval".into()],
        })
    }

    fn get_status(&self) -> ClientStatus {
        ClientStatus {
            backend: "evaluation".into(),
            state: "ready".into(),
            llm_model: "mock".into(),
            embedding_model: "all-MiniLM-L6-v2".into(),
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
}

// ═══════════════════════════════════════════════════════════════════════════
//  Dataset
// ═══════════════════════════════════════════════════════════════════════════

struct TestMemory {
    id: &'static str,
    content: &'static str,
    memory_type: MemoryType,
    topics: &'static [&'static str],
}

struct TestQuery {
    query: &'static str,
    /// IDs of memories that are considered relevant answers.
    expected_ids: &'static [&'static str],
}

fn evaluation_memories() -> Vec<TestMemory> {
    vec![
        TestMemory {
            id: "rust_programming",
            content: "The user has been programming in Rust for about 3 years. \
                      They particularly enjoy the ownership model and borrow checker, \
                      which makes concurrent programming safer. Before Rust they worked \
                      extensively with C++ and Python.",
            memory_type: MemoryType::Factual,
            topics: &["programming", "rust", "languages"],
        },
        TestMemory {
            id: "home_sf",
            content: "The user lives in San Francisco, California. They moved there \
                      in 2022 for a tech job. They enjoy the mild weather but find the \
                      cost of living, especially rent, quite expensive.",
            memory_type: MemoryType::Personal,
            topics: &["location", "san francisco", "housing"],
        },
        TestMemory {
            id: "pet_dog",
            content: "The user has a golden retriever named Max who is 4 years old. \
                      They adopted Max from a local animal shelter. Every morning they \
                      take Max to Golden Gate Park for a 30-minute walk.",
            memory_type: MemoryType::Personal,
            topics: &["pets", "dog", "golden retriever"],
        },
        TestMemory {
            id: "ml_project",
            content: "The user is building a recommendation system using PyTorch and Python. \
                      The project uses collaborative filtering and matrix factorization for \
                      an e-commerce platform. They are training on a dataset of 10 million \
                      user interactions.",
            memory_type: MemoryType::Factual,
            topics: &["machine learning", "pytorch", "python"],
        },
        TestMemory {
            id: "editor_setup",
            content: "The user prefers VS Code as their primary code editor with dark mode \
                      enabled. They use the vim keybindings extension and their favorite \
                      color theme is One Dark Pro. They also have GitHub Copilot installed.",
            memory_type: MemoryType::Personal,
            topics: &["editor", "vscode", "tools"],
        },
        TestMemory {
            id: "quarterly_meeting",
            content: "The user has a quarterly business review meeting scheduled for next \
                      Monday at 2pm in conference room B. They need to prepare a presentation \
                      covering Q4 revenue growth, customer acquisition metrics, and the product \
                      roadmap for the next quarter.",
            memory_type: MemoryType::Episodic,
            topics: &["meeting", "work", "schedule"],
        },
        TestMemory {
            id: "sushi_restaurant",
            content: "The user's favorite restaurant is Sushi Zen on Market Street in San \
                      Francisco. They visit every Friday evening for the chef's omakase \
                      special, which costs about $85 per person. They especially love the \
                      salmon nigiri and uni.",
            memory_type: MemoryType::Personal,
            topics: &["food", "restaurant", "sushi"],
        },
        TestMemory {
            id: "docker_debugging",
            content: "The user was debugging a Docker container that kept crashing with an \
                      OOM (out of memory) error. The solution was to increase the container \
                      memory limit from 512MB to 4GB in the docker-compose.yml file and add \
                      a restart policy with max retries.",
            memory_type: MemoryType::Procedural,
            topics: &["docker", "debugging", "devops"],
        },
        TestMemory {
            id: "japanese_study",
            content: "The user is studying Japanese language using the Genki textbook series. \
                      They have been learning for 6 months and can read hiragana and katakana \
                      fluently. Their goal is to pass the JLPT N5 exam by December.",
            memory_type: MemoryType::Personal,
            topics: &["language learning", "japanese", "education"],
        },
        TestMemory {
            id: "running_routine",
            content: "The user runs 5 kilometers every morning before work starting at 6am. \
                      They have maintained this habit for 2 years and recently completed their \
                      first half marathon with a finishing time of 1 hour and 45 minutes.",
            memory_type: MemoryType::Personal,
            topics: &["exercise", "running", "fitness"],
        },
        TestMemory {
            id: "book_scifi",
            content: "The user recently finished reading 'Project Hail Mary' by Andy Weir \
                      and rated it 5 stars. They love science fiction and also highly recommend \
                      'The Three-Body Problem' by Liu Cixin and 'Dune' by Frank Herbert.",
            memory_type: MemoryType::Conversational,
            topics: &["books", "science fiction", "reading"],
        },
        TestMemory {
            id: "work_startup",
            content: "The user works at a startup called DataFlow as a senior software \
                      engineer. DataFlow is in the data infrastructure and ETL pipeline \
                      space, has about 50 employees, and recently raised a Series B funding \
                      round of 25 million dollars.",
            memory_type: MemoryType::Factual,
            topics: &["work", "career", "startup"],
        },
        TestMemory {
            id: "smart_home",
            content: "The user has configured Home Assistant on a Raspberry Pi 4 for home \
                      automation. They control Philips Hue smart lights, a Nest thermostat, \
                      and have automated morning routines that turn on lights at sunrise and \
                      adjust the temperature to 72°F.",
            memory_type: MemoryType::Procedural,
            topics: &["smart home", "automation", "iot"],
        },
        TestMemory {
            id: "birthday_march",
            content: "The user's birthday is on March 15th. They typically celebrate with \
                      a small dinner party at home with 5 to 6 close friends. Last year \
                      they had a Japanese-themed dinner party.",
            memory_type: MemoryType::Personal,
            topics: &["birthday", "personal", "celebration"],
        },
        TestMemory {
            id: "tokyo_trip",
            content: "The user is planning a two-week trip to Tokyo, Japan in April to see \
                      the cherry blossoms during hanami season. They have booked a hotel in \
                      Shinjuku and plan to visit Akihabara, Shibuya, and take a day trip to \
                      Mount Fuji.",
            memory_type: MemoryType::Episodic,
            topics: &["travel", "japan", "tokyo"],
        },
    ]
}

fn evaluation_queries() -> Vec<TestQuery> {
    vec![
        TestQuery {
            query: "What programming languages does the user know?",
            expected_ids: &["rust_programming", "ml_project"],
        },
        TestQuery {
            query: "Where does the user live?",
            expected_ids: &["home_sf"],
        },
        TestQuery {
            query: "Tell me about the user's pets",
            expected_ids: &["pet_dog"],
        },
        TestQuery {
            query: "What code editor does the user prefer?",
            expected_ids: &["editor_setup"],
        },
        TestQuery {
            query: "What is the user's exercise or fitness routine?",
            expected_ids: &["running_routine"],
        },
        TestQuery {
            query: "What books has the user read or recommended?",
            expected_ids: &["book_scifi"],
        },
        TestQuery {
            query: "Where does the user work and what is their role?",
            expected_ids: &["work_startup"],
        },
        TestQuery {
            query: "Does the user have any upcoming travel plans?",
            expected_ids: &["tokyo_trip"],
        },
        TestQuery {
            query: "How did the user fix the Docker container issue?",
            expected_ids: &["docker_debugging"],
        },
        TestQuery {
            query: "Is the user studying any foreign languages?",
            expected_ids: &["japanese_study"],
        },
        TestQuery {
            query: "What smart home or IoT devices does the user have?",
            expected_ids: &["smart_home"],
        },
        TestQuery {
            query: "When is the user's birthday?",
            expected_ids: &["birthday_march"],
        },
        TestQuery {
            query: "What machine learning or AI projects is the user working on?",
            expected_ids: &["ml_project"],
        },
        TestQuery {
            query: "What restaurants does the user enjoy eating at?",
            expected_ids: &["sushi_restaurant"],
        },
        TestQuery {
            query: "What meetings or appointments does the user have coming up?",
            expected_ids: &["quarterly_meeting"],
        },
    ]
}

/// Harder queries that require discriminating between semantically related memories.
fn discrimination_queries() -> Vec<TestQuery> {
    vec![
        // "Japan" appears in both tokyo_trip and japanese_study
        TestQuery {
            query: "What are the user's travel plans to Japan?",
            expected_ids: &["tokyo_trip"],
        },
        TestQuery {
            query: "What Japanese language skills does the user have?",
            expected_ids: &["japanese_study"],
        },
        // Morning activities: running and dog walking
        TestQuery {
            query: "What does the user do for physical exercise?",
            expected_ids: &["running_routine"],
        },
        // Programming-related: Rust vs ML project
        TestQuery {
            query: "What Rust programming experience does the user have?",
            expected_ids: &["rust_programming"],
        },
        TestQuery {
            query: "What Python machine learning work is the user doing?",
            expected_ids: &["ml_project"],
        },
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
//  Metrics
// ═══════════════════════════════════════════════════════════════════════════

struct QueryResult {
    query: String,
    expected_ids: Vec<String>,
    retrieved_ids: Vec<String>,
    retrieved_scores: Vec<f32>,
    /// Rank of each expected ID that was found (1-indexed). Empty if not found.
    found_ranks: Vec<usize>,
}

#[derive(Debug)]
struct EvalMetrics {
    total_queries: usize,
    recall_at_1: f64,
    recall_at_3: f64,
    recall_at_5: f64,
    /// Mean Reciprocal Rank (average of 1/rank_of_first_relevant)
    mrr: f64,
    /// Average Precision@5
    precision_at_5: f64,
}

fn compute_metrics(results: &[QueryResult]) -> EvalMetrics {
    let n = results.len() as f64;

    // Recall@K: fraction of queries where ≥1 relevant result is in top K
    let recall_at = |k: usize| -> f64 {
        results
            .iter()
            .filter(|r| r.found_ranks.iter().any(|&rank| rank <= k))
            .count() as f64
            / n
    };

    // MRR: mean of 1/rank_of_first_relevant
    let mrr = results
        .iter()
        .map(|r| {
            r.found_ranks
                .iter()
                .min()
                .map(|&rank| 1.0 / rank as f64)
                .unwrap_or(0.0)
        })
        .sum::<f64>()
        / n;

    // Precision@5: average fraction of relevant results in top 5
    let precision_at_5 = results
        .iter()
        .map(|r| {
            let relevant_in_top5 = r.found_ranks.iter().filter(|&&rank| rank <= 5).count() as f64;
            let expected = r.expected_ids.len() as f64;
            relevant_in_top5 / expected.max(1.0)
        })
        .sum::<f64>()
        / n;

    EvalMetrics {
        total_queries: results.len(),
        recall_at_1: recall_at(1),
        recall_at_3: recall_at(3),
        recall_at_5: recall_at(5),
        mrr,
        precision_at_5,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn make_eval_store() -> VectorLiteStore {
    VectorLiteStore::with_config(VectorLiteConfig {
        collection_name: "eval".to_string(),
        index_type: IndexType::HNSW,
        metric: SimilarityMetric::Cosine,
        persistence_path: None,
    })
    .expect("Failed to create vector store")
}

fn make_memory(
    id: &str,
    content: &str,
    embedding: Vec<f32>,
    mem_type: MemoryType,
    topics: &[&str],
) -> Memory {
    let mut metadata = MemoryMetadata::new(mem_type);
    metadata.topics = topics.iter().map(|s| s.to_string()).collect();
    metadata.hash = Memory::compute_hash(content);
    let mut memory = Memory::with_content(content.to_string(), embedding, metadata);
    memory.id = id.to_string(); // override the UUID with our test ID
    memory
}

/// Run queries against a store and collect results.
async fn run_queries(
    model: &TextEmbedding,
    store: &VectorLiteStore,
    queries: &[TestQuery],
    k: usize,
    filters: &Filters,
) -> Vec<QueryResult> {
    let mut results = Vec::new();

    for q in queries {
        let query_embedding = model
            .embed(vec![q.query.to_string()], None)
            .expect("Query embedding failed")
            .remove(0);

        let search_results = store
            .search(&query_embedding, filters, k)
            .await
            .expect("Search failed");

        let retrieved_ids: Vec<String> = search_results
            .iter()
            .map(|sr| sr.memory.id.clone())
            .collect();
        let retrieved_scores: Vec<f32> = search_results.iter().map(|sr| sr.score).collect();

        let found_ranks: Vec<usize> = q
            .expected_ids
            .iter()
            .filter_map(|eid| {
                retrieved_ids
                    .iter()
                    .position(|rid| rid == eid)
                    .map(|pos| pos + 1) // 1-indexed
            })
            .collect();

        results.push(QueryResult {
            query: q.query.to_string(),
            expected_ids: q.expected_ids.iter().map(|s| s.to_string()).collect(),
            retrieved_ids,
            retrieved_scores,
            found_ranks,
        });
    }

    results
}

fn print_report(title: &str, num_memories: usize, results: &[QueryResult], metrics: &EvalMetrics) {
    let sep = "═".repeat(72);
    let thin = "─".repeat(72);

    println!();
    println!("{}", sep);
    println!("  {}", title);
    println!("{}", sep);
    println!("  Embedding model : all-MiniLM-L6-v2 (384 dimensions)");
    println!("  Memories stored : {}", num_memories);
    println!("  Queries tested  : {}", metrics.total_queries);
    println!("  Search depth    : K=5");
    println!("{}", thin);
    println!("  QUERY DETAILS");
    println!("{}", thin);

    for (i, r) in results.iter().enumerate() {
        let found_any = !r.found_ranks.is_empty();
        let mark = if found_any { "✓" } else { "✗" };
        let best_rank = r.found_ranks.iter().min().copied().unwrap_or(0);
        let query_short: String = r.query.chars().take(55).collect();

        if found_any {
            let best_score = if best_rank > 0 && best_rank <= r.retrieved_scores.len() {
                r.retrieved_scores[best_rank - 1]
            } else {
                0.0
            };
            println!(
                "  {} {:2}. \"{}\" → rank {} [{:.3}]",
                mark,
                i + 1,
                query_short,
                best_rank,
                best_score
            );
        } else {
            println!(
                "  {} {:2}. \"{}\" → NOT FOUND in top 5",
                mark,
                i + 1,
                query_short
            );
            println!("        Expected: {}", r.expected_ids.join(", "));
            if !r.retrieved_ids.is_empty() {
                println!(
                    "        Got:      {} [{:.3}], {} [{:.3}], ...",
                    r.retrieved_ids[0],
                    r.retrieved_scores[0],
                    r.retrieved_ids.get(1).map(|s| s.as_str()).unwrap_or("-"),
                    r.retrieved_scores.get(1).unwrap_or(&0.0),
                );
            }
        }
    }

    println!("{}", thin);
    println!("  METRICS");
    println!("{}", thin);
    println!(
        "  Recall@1      : {:5.1}%  ({}/{})",
        metrics.recall_at_1 * 100.0,
        (metrics.recall_at_1 * metrics.total_queries as f64).round() as usize,
        metrics.total_queries
    );
    println!(
        "  Recall@3      : {:5.1}%  ({}/{})",
        metrics.recall_at_3 * 100.0,
        (metrics.recall_at_3 * metrics.total_queries as f64).round() as usize,
        metrics.total_queries
    );
    println!(
        "  Recall@5      : {:5.1}%  ({}/{})",
        metrics.recall_at_5 * 100.0,
        (metrics.recall_at_5 * metrics.total_queries as f64).round() as usize,
        metrics.total_queries
    );
    println!("  MRR           : {:5.3}", metrics.mrr);
    println!("  Precision@5   : {:5.1}%", metrics.precision_at_5 * 100.0);
    println!("{}", sep);
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 1: Pure embedding retrieval accuracy
// ═══════════════════════════════════════════════════════════════════════════

/// Measures retrieval accuracy using real semantic embeddings.
///
/// Stores 15 memories directly in VectorLiteStore with fastembed embeddings,
/// then runs 15 natural language queries and checks if the expected memories
/// appear in the top-K results.
///
/// Asserts minimum quality thresholds:
/// - Recall@1 ≥ 60%, Recall@3 ≥ 80%, Recall@5 ≥ 85%, MRR ≥ 0.65
#[tokio::test]
#[ignore]
async fn evaluation_retrieval_accuracy() {
    println!("\n  Setting up evaluation...");
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    )
    .expect("Failed to initialize embedding model");

    let store = make_eval_store();
    let memories = evaluation_memories();
    let k = 5;
    let default_filters = Filters::default();

    // Store all memories with real embeddings
    println!(
        "  Storing {} memories with real embeddings...",
        memories.len()
    );
    for mem in &memories {
        let embedding = model
            .embed(vec![mem.content.to_string()], None)
            .expect("Embedding failed")
            .remove(0);
        let memory = make_memory(
            mem.id,
            mem.content,
            embedding,
            mem.memory_type.clone(),
            mem.topics,
        );
        store.insert(&memory).await.expect("Insert failed");
    }

    // Run standard queries
    let queries = evaluation_queries();
    let results = run_queries(&model, &store, &queries, k, &default_filters).await;
    let metrics = compute_metrics(&results);

    print_report(
        "llm-mem Retrieval Accuracy Evaluation",
        memories.len(),
        &results,
        &metrics,
    );

    // Assertions — conservative thresholds to avoid flaky tests
    assert!(
        metrics.recall_at_1 >= 0.60,
        "Recall@1 ({:.1}%) below minimum threshold (60%)",
        metrics.recall_at_1 * 100.0
    );
    assert!(
        metrics.recall_at_3 >= 0.80,
        "Recall@3 ({:.1}%) below minimum threshold (80%)",
        metrics.recall_at_3 * 100.0
    );
    assert!(
        metrics.recall_at_5 >= 0.85,
        "Recall@5 ({:.1}%) below minimum threshold (85%)",
        metrics.recall_at_5 * 100.0
    );
    assert!(
        metrics.mrr >= 0.65,
        "MRR ({:.3}) below minimum threshold (0.65)",
        metrics.mrr
    );
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 2: Full pipeline accuracy (MemoryManager store → search)
// ═══════════════════════════════════════════════════════════════════════════

/// Tests retrieval accuracy through the complete MemoryManager pipeline.
///
/// This exercises the full path: content → embedding → dedup check →
/// storage → query embedding → search → ranking.
///
/// Uses real embeddings but mock LLM completions.
#[tokio::test]
#[ignore]
async fn evaluation_full_pipeline_accuracy() {
    println!("\n  Setting up full pipeline evaluation...");
    let client = EvalLLMClient::new();
    let store = make_eval_store();
    let config = MemoryConfig {
        deduplicate: false, // Disable dedup for evaluation (we control the data)
        search_similarity_threshold: None, // No threshold — return top K
        ..MemoryConfig::default()
    };

    let manager = MemoryManager::new(Box::new(store), Box::new(client.clone()), config);

    let memories = evaluation_memories();
    println!(
        "  Storing {} memories via MemoryManager pipeline...",
        memories.len()
    );

    // Store memories through the full pipeline
    let mut stored_ids: HashMap<String, String> = HashMap::new(); // test_id → actual_id
    for mem in &memories {
        let metadata = MemoryMetadata::new(mem.memory_type.clone());
        match manager.store(mem.content.to_string(), metadata).await {
            Ok(id) => {
                stored_ids.insert(mem.id.to_string(), id);
            }
            Err(e) => {
                println!("  WARNING: Failed to store '{}': {}", mem.id, e);
            }
        }
    }
    println!("  Successfully stored {} memories", stored_ids.len());

    // Run queries through MemoryManager::search
    let queries = evaluation_queries();
    let default_filters = Filters::default();
    let k = 5;
    let mut results = Vec::new();

    for q in &queries {
        let search_results = manager
            .search(q.query, &default_filters, k)
            .await
            .unwrap_or_default();

        let retrieved_ids: Vec<String> = search_results
            .iter()
            .map(|sr| sr.memory.id.clone())
            .collect();
        let retrieved_scores: Vec<f32> = search_results.iter().map(|sr| sr.score).collect();

        // Map expected test IDs → actual stored IDs for comparison
        let expected_actual_ids: Vec<String> = q
            .expected_ids
            .iter()
            .filter_map(|eid| stored_ids.get(*eid).cloned())
            .collect();

        let found_ranks: Vec<usize> = expected_actual_ids
            .iter()
            .filter_map(|eid| {
                retrieved_ids
                    .iter()
                    .position(|rid| rid == eid)
                    .map(|pos| pos + 1)
            })
            .collect();

        results.push(QueryResult {
            query: q.query.to_string(),
            expected_ids: q.expected_ids.iter().map(|s| s.to_string()).collect(),
            retrieved_ids,
            retrieved_scores,
            found_ranks,
        });
    }

    let metrics = compute_metrics(&results);
    print_report(
        "llm-mem Full Pipeline Accuracy Evaluation",
        stored_ids.len(),
        &results,
        &metrics,
    );

    // Slightly lower thresholds for full pipeline (LLM processing may alter content)
    assert!(
        metrics.recall_at_1 >= 0.50,
        "Pipeline Recall@1 ({:.1}%) below minimum threshold (50%)",
        metrics.recall_at_1 * 100.0
    );
    assert!(
        metrics.recall_at_3 >= 0.70,
        "Pipeline Recall@3 ({:.1}%) below minimum threshold (70%)",
        metrics.recall_at_3 * 100.0
    );
    assert!(
        metrics.recall_at_5 >= 0.80,
        "Pipeline Recall@5 ({:.1}%) below minimum threshold (80%)",
        metrics.recall_at_5 * 100.0
    );
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 3: Type-filtered retrieval
// ═══════════════════════════════════════════════════════════════════════════

/// Tests retrieval accuracy when filtering by memory type.
///
/// Verifies that type filters correctly narrow results while still returning
/// relevant memories of the specified type.
#[tokio::test]
#[ignore]
async fn evaluation_type_filtered_retrieval() {
    println!("\n  Setting up type-filtered evaluation...");
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    )
    .expect("Failed to initialize embedding model");

    let store = make_eval_store();
    let memories = evaluation_memories();

    // Store all memories
    for mem in &memories {
        let embedding = model
            .embed(vec![mem.content.to_string()], None)
            .unwrap()
            .remove(0);
        let memory = make_memory(
            mem.id,
            mem.content,
            embedding,
            mem.memory_type.clone(),
            mem.topics,
        );
        store.insert(&memory).await.unwrap();
    }

    let k = 5;

    // Filtered queries: each targets a specific MemoryType
    let type_queries: Vec<(TestQuery, MemoryType)> = vec![
        (
            TestQuery {
                query: "What personal information do I know about the user?",
                expected_ids: &[
                    "home_sf",
                    "pet_dog",
                    "editor_setup",
                    "sushi_restaurant",
                    "japanese_study",
                ],
            },
            MemoryType::Personal,
        ),
        (
            TestQuery {
                query: "What factual information about the user's work and skills?",
                expected_ids: &["rust_programming", "ml_project", "work_startup"],
            },
            MemoryType::Factual,
        ),
        (
            TestQuery {
                query: "What procedural steps or solutions were discussed?",
                expected_ids: &["docker_debugging", "smart_home"],
            },
            MemoryType::Procedural,
        ),
        (
            TestQuery {
                query: "What upcoming events does the user have?",
                expected_ids: &["quarterly_meeting", "tokyo_trip"],
            },
            MemoryType::Episodic,
        ),
    ];

    let sep = "═".repeat(72);
    let thin = "─".repeat(72);
    println!();
    println!("{}", sep);
    println!("  Type-Filtered Retrieval Evaluation");
    println!("{}", sep);

    let mut total_found = 0;
    let mut total_expected = 0;

    for (query, mem_type) in &type_queries {
        let filters = Filters {
            memory_type: Some(mem_type.clone()),
            ..Default::default()
        };

        let query_embedding = model
            .embed(vec![query.query.to_string()], None)
            .unwrap()
            .remove(0);

        let search_results = store.search(&query_embedding, &filters, k).await.unwrap();

        let retrieved_ids: Vec<&str> = search_results
            .iter()
            .map(|sr| sr.memory.id.as_str())
            .collect();

        // All results should be of the correct type
        let all_correct_type = search_results
            .iter()
            .all(|sr| sr.memory.metadata.memory_type == *mem_type);

        let found: Vec<&&str> = query
            .expected_ids
            .iter()
            .filter(|eid| retrieved_ids.contains(eid))
            .collect();

        total_found += found.len();
        total_expected += query.expected_ids.len();

        let mark = if found.len() == query.expected_ids.len() {
            "✓"
        } else {
            "~"
        };
        let type_mark = if all_correct_type { "✓" } else { "✗" };

        println!("{}", thin);
        println!(
            "  {} Filter: {:?} | Type correct: {} | Found: {}/{}",
            mark,
            mem_type,
            type_mark,
            found.len(),
            query.expected_ids.len()
        );
        println!("    Query: \"{}\"", query.query);
        println!("    Retrieved: {:?}", retrieved_ids);
    }

    let recall = if total_expected > 0 {
        total_found as f64 / total_expected as f64
    } else {
        0.0
    };
    println!("{}", thin);
    println!(
        "  Overall filtered recall: {:.1}% ({}/{})",
        recall * 100.0,
        total_found,
        total_expected
    );
    println!("{}", sep);

    assert!(
        recall >= 0.50,
        "Type-filtered recall ({:.1}%) below minimum threshold (50%)",
        recall * 100.0
    );
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 4: Similarity discrimination
// ═══════════════════════════════════════════════════════════════════════════

/// Tests whether the system can distinguish between semantically related
/// but different memories.
///
/// For example: "travel to Japan" vs "studying Japanese language" — both
/// mention Japan, but the queries target different memories.
#[tokio::test]
#[ignore]
async fn evaluation_similarity_discrimination() {
    println!("\n  Setting up discrimination evaluation...");
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    )
    .expect("Failed to initialize embedding model");

    let store = make_eval_store();
    let memories = evaluation_memories();
    let default_filters = Filters::default();

    for mem in &memories {
        let embedding = model
            .embed(vec![mem.content.to_string()], None)
            .unwrap()
            .remove(0);
        let memory = make_memory(
            mem.id,
            mem.content,
            embedding,
            mem.memory_type.clone(),
            mem.topics,
        );
        store.insert(&memory).await.unwrap();
    }

    let queries = discrimination_queries();
    let k = 5;
    let results = run_queries(&model, &store, &queries, k, &default_filters).await;
    let metrics = compute_metrics(&results);

    print_report(
        "llm-mem Similarity Discrimination Evaluation",
        memories.len(),
        &results,
        &metrics,
    );

    // Discrimination is harder — use lower thresholds
    assert!(
        metrics.recall_at_1 >= 0.50,
        "Discrimination Recall@1 ({:.1}%) below threshold (50%)",
        metrics.recall_at_1 * 100.0
    );
    assert!(
        metrics.recall_at_3 >= 0.70,
        "Discrimination Recall@3 ({:.1}%) below threshold (70%)",
        metrics.recall_at_3 * 100.0
    );
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 4b: Relation-filtered retrieval (Level 2)
// ═══════════════════════════════════════════════════════════════════════════

/// Tests retrieval accuracy when storing memories with relations and searching
/// with relation filters.
///
/// Uses real embeddings (fastembed) + mock LLM completions.
/// Verifies that:
///   - Relation metadata is preserved through storage pipeline
///   - Relation filters correctly narrow search results
///   - Unrelated memories are excluded by relation filters
#[tokio::test]
#[ignore]
async fn evaluation_relation_filtered_retrieval() {
    use llm_mem::types::{Relation, RelationFilter};

    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Relation-Filtered Retrieval Evaluation (Level 2)");
    println!("══════════════════════════════════════════════════════════════");

    let client = EvalLLMClient::new();
    let store = make_eval_store();
    let config = MemoryConfig {
        deduplicate: false,
        search_similarity_threshold: None,
        ..MemoryConfig::default()
    };

    let manager = MemoryManager::new(Box::new(store), Box::new(client.clone()), config);

    // Memories with explicit relations
    struct RelMemory {
        id: &'static str,
        content: &'static str,
        relations: Vec<Relation>,
    }

    let rel_memories = vec![
        RelMemory {
            id: "alice_likes_pizza",
            content: "Alice loves eating pepperoni pizza from Tony's Pizzeria every Friday night",
            relations: vec![
                Relation {
                    source: "Alice".into(),
                    relation: "LIKES".into(),
                    target: "Pizza".into(),
                    strength: None,
                },
                Relation {
                    source: "Alice".into(),
                    relation: "VISITS".into(),
                    target: "Tony's Pizzeria".into(),
                    strength: None,
                },
            ],
        },
        RelMemory {
            id: "bob_likes_sushi",
            content: "Bob prefers sushi and visits Sushi Zen restaurant on weekends",
            relations: vec![
                Relation {
                    source: "Bob".into(),
                    relation: "LIKES".into(),
                    target: "Sushi".into(),
                    strength: None,
                },
                Relation {
                    source: "Bob".into(),
                    relation: "VISITS".into(),
                    target: "Sushi Zen".into(),
                    strength: None,
                },
            ],
        },
        RelMemory {
            id: "alice_works_dataflow",
            content: "Alice works at DataFlow as a senior engineer building data pipelines",
            relations: vec![
                Relation {
                    source: "Alice".into(),
                    relation: "WORKS_AT".into(),
                    target: "DataFlow".into(),
                    strength: None,
                },
                Relation {
                    source: "Alice".into(),
                    relation: "HAS_ROLE".into(),
                    target: "Senior Engineer".into(),
                    strength: None,
                },
            ],
        },
        RelMemory {
            id: "bob_works_acme",
            content: "Bob is a product manager at Acme Corp working on the mobile app",
            relations: vec![
                Relation {
                    source: "Bob".into(),
                    relation: "WORKS_AT".into(),
                    target: "Acme Corp".into(),
                    strength: None,
                },
                Relation {
                    source: "Bob".into(),
                    relation: "HAS_ROLE".into(),
                    target: "Product Manager".into(),
                    strength: None,
                },
            ],
        },
        RelMemory {
            id: "alice_knows_rust",
            content: "Alice has been programming in Rust for three years and loves the borrow checker",
            relations: vec![Relation {
                source: "Alice".into(),
                relation: "KNOWS".into(),
                target: "Rust".into(),
                strength: None,
            }],
        },
        RelMemory {
            id: "bob_knows_python",
            content: "Bob is an expert Python developer specializing in machine learning with PyTorch",
            relations: vec![
                Relation {
                    source: "Bob".into(),
                    relation: "KNOWS".into(),
                    target: "Python".into(),
                    strength: None,
                },
                Relation {
                    source: "Bob".into(),
                    relation: "USES".into(),
                    target: "PyTorch".into(),
                    strength: None,
                },
            ],
        },
        RelMemory {
            id: "alice_lives_sf",
            content: "Alice lives in San Francisco near Golden Gate Park in a two-bedroom apartment",
            relations: vec![Relation {
                source: "Alice".into(),
                relation: "LIVES_IN".into(),
                target: "San Francisco".into(),
                strength: None,
            }],
        },
        RelMemory {
            id: "bob_lives_nyc",
            content: "Bob lives in New York City in a Brooklyn apartment near Prospect Park",
            relations: vec![Relation {
                source: "Bob".into(),
                relation: "LIVES_IN".into(),
                target: "New York City".into(),
                strength: None,
            }],
        },
    ];

    println!(
        "  Storing {} memories with relations...",
        rel_memories.len()
    );

    let mut stored_ids: HashMap<String, String> = HashMap::new();

    for (i, mem) in rel_memories.iter().enumerate() {
        let mut metadata = MemoryMetadata::new(MemoryType::Factual);
        metadata.relations = mem.relations.clone();
        print!("  [{}/{}] '{}'...", i + 1, rel_memories.len(), mem.id);
        match manager.store(mem.content.to_string(), metadata).await {
            Ok(id) => {
                println!(" ✓ ({} relations)", mem.relations.len());
                stored_ids.insert(mem.id.to_string(), id);
            }
            Err(e) => {
                println!(" ✗ {}", e);
            }
        }
    }

    println!("  Stored: {}/{}\n", stored_ids.len(), rel_memories.len());

    let sep = "═".repeat(72);
    let thin = "─".repeat(72);

    // Test relation filters
    struct RelFilterTest {
        label: &'static str,
        query: &'static str,
        filter: RelationFilter,
        expected_ids: &'static [&'static str],
    }

    let filter_tests = vec![
        RelFilterTest {
            label: "LIKES Pizza",
            query: "What food does someone enjoy?",
            filter: RelationFilter {
                relation: "LIKES".into(),
                target: "Pizza".into(),
            },
            expected_ids: &["alice_likes_pizza"],
        },
        RelFilterTest {
            label: "WORKS_AT DataFlow",
            query: "Where does someone work?",
            filter: RelationFilter {
                relation: "WORKS_AT".into(),
                target: "DataFlow".into(),
            },
            expected_ids: &["alice_works_dataflow"],
        },
        RelFilterTest {
            label: "KNOWS Rust",
            query: "What programming skills does someone have?",
            filter: RelationFilter {
                relation: "KNOWS".into(),
                target: "Rust".into(),
            },
            expected_ids: &["alice_knows_rust"],
        },
        RelFilterTest {
            label: "LIVES_IN New York City",
            query: "Where does someone live?",
            filter: RelationFilter {
                relation: "LIVES_IN".into(),
                target: "New York City".into(),
            },
            expected_ids: &["bob_lives_nyc"],
        },
        RelFilterTest {
            label: "LIKES Sushi",
            query: "What food preferences are there?",
            filter: RelationFilter {
                relation: "LIKES".into(),
                target: "Sushi".into(),
            },
            expected_ids: &["bob_likes_sushi"],
        },
    ];

    println!("{}", sep);
    println!("  Relation-Filtered Search Results");
    println!("{}", sep);

    let mut passed = 0;
    let total = filter_tests.len();

    for test in &filter_tests {
        let filters = Filters {
            relations: Some(vec![test.filter.clone()]),
            ..Default::default()
        };

        let results = manager
            .search(test.query, &filters, 5)
            .await
            .unwrap_or_default();
        let retrieved_ids: Vec<String> = results.iter().map(|r| r.memory.id.clone()).collect();

        let expected_actual_ids: Vec<String> = test
            .expected_ids
            .iter()
            .filter_map(|eid| stored_ids.get(*eid).cloned())
            .collect();

        let found = expected_actual_ids
            .iter()
            .filter(|eid| retrieved_ids.contains(eid))
            .count();

        let all_have_relation = results.iter().all(|r| {
            r.memory.metadata.relations.iter().any(|rel| {
                rel.relation.eq_ignore_ascii_case(&test.filter.relation)
                    && rel.target.eq_ignore_ascii_case(&test.filter.target)
            })
        });

        let mark = if found == expected_actual_ids.len() && all_have_relation {
            "✓"
        } else {
            "~"
        };
        if found == expected_actual_ids.len() && all_have_relation {
            passed += 1;
        }

        println!("{}", thin);
        println!(
            "  {} Filter: {} | Found: {}/{} | All match filter: {} | Total results: {}",
            mark,
            test.label,
            found,
            expected_actual_ids.len(),
            all_have_relation,
            results.len()
        );
    }

    println!("{}", thin);
    println!(
        "  Relation filter accuracy: {}/{} ({:.0}%)",
        passed,
        total,
        (passed as f64 / total as f64) * 100.0
    );
    println!("{}", sep);

    assert!(
        passed >= 3,
        "Relation filter accuracy ({}/{}) below minimum threshold (3/5)",
        passed,
        total
    );
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 4c: Context two-stage retrieval (Level 3)
// ═══════════════════════════════════════════════════════════════════════════

/// Tests the two-stage context retrieval pipeline with real embeddings.
///
/// Uses real embeddings (fastembed) + mock LLM completions.
/// Stores memories with context tags, then runs search_with_context to verify:
///   - Context embeddings are generated and stored
///   - Two-stage retrieval (context pre-filter → content search) works
///   - Context-scoped search can narrow results compared to unscoped search
#[tokio::test]
#[ignore]
async fn evaluation_context_retrieval() {
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Context Two-Stage Retrieval Evaluation (Level 3)");
    println!("══════════════════════════════════════════════════════════════");

    let client = EvalLLMClient::new();
    let store = make_eval_store();
    let config = MemoryConfig {
        deduplicate: false,
        search_similarity_threshold: None,
        ..MemoryConfig::default()
    };

    let manager = MemoryManager::new(Box::new(store), Box::new(client.clone()), config);

    // Memories with explicit context tags — deliberately overlapping content
    // across different contexts to test whether context narrows results
    struct CtxMemory {
        id: &'static str,
        content: &'static str,
        context: Vec<String>,
    }

    let ctx_memories = vec![
        CtxMemory {
            id: "work_meeting_q4",
            content: "Discussed the Q4 budget allocation and revenue targets with the finance team",
            context: vec!["work".into(), "meeting".into(), "finance".into()],
        },
        CtxMemory {
            id: "work_code_review",
            content: "Reviewed the authentication module code and suggested switching to JWT tokens",
            context: vec!["work".into(), "engineering".into(), "code-review".into()],
        },
        CtxMemory {
            id: "personal_cooking",
            content: "Learned a new pasta carbonara recipe using guanciale instead of bacon",
            context: vec!["personal".into(), "cooking".into(), "italian".into()],
        },
        CtxMemory {
            id: "personal_fitness",
            content: "Completed a 10km run in 48 minutes, new personal best this month",
            context: vec!["personal".into(), "fitness".into(), "running".into()],
        },
        CtxMemory {
            id: "project_alpha_design",
            content: "Designed the microservice architecture for Project Alpha with event sourcing",
            context: vec!["work".into(), "project-alpha".into(), "architecture".into()],
        },
        CtxMemory {
            id: "project_alpha_deploy",
            content: "Deployed Project Alpha v2.1 to staging environment with canary rollout",
            context: vec!["work".into(), "project-alpha".into(), "deployment".into()],
        },
        CtxMemory {
            id: "learning_rust",
            content: "Studying Rust ownership and lifetimes, completed the Rustlings exercises",
            context: vec!["learning".into(), "programming".into(), "rust".into()],
        },
        CtxMemory {
            id: "learning_japanese",
            content: "Practiced Japanese kanji writing, learned 20 new characters from JLPT N4 list",
            context: vec!["learning".into(), "japanese".into(), "language".into()],
        },
        CtxMemory {
            id: "personal_cooking_baking",
            content: "Baked sourdough bread using a 72-hour cold fermentation technique",
            context: vec!["personal".into(), "cooking".into(), "baking".into()],
        },
        CtxMemory {
            id: "work_meeting_standup",
            content: "Daily standup: blocked on API integration, waiting for partner team's response",
            context: vec!["work".into(), "meeting".into(), "standup".into()],
        },
    ];

    println!(
        "  Storing {} memories with context tags...",
        ctx_memories.len()
    );

    let mut stored_ids: HashMap<String, String> = HashMap::new();

    for (i, mem) in ctx_memories.iter().enumerate() {
        let mut metadata = MemoryMetadata::new(MemoryType::Factual);
        metadata.context = mem.context.clone();
        print!("  [{:2}/{}] '{}'...", i + 1, ctx_memories.len(), mem.id);
        match manager.store(mem.content.to_string(), metadata).await {
            Ok(id) => {
                println!(" ✓ (context: {:?})", mem.context);
                stored_ids.insert(mem.id.to_string(), id);
            }
            Err(e) => {
                println!(" ✗ {}", e);
            }
        }
    }

    println!("  Stored: {}/{}\n", stored_ids.len(), ctx_memories.len());

    let sep = "═".repeat(72);
    let thin = "─".repeat(72);

    // Context-scoped queries
    struct CtxQuery {
        label: &'static str,
        query: &'static str,
        context_tags: Vec<String>,
        expected_ids: &'static [&'static str],
        /// IDs that should NOT appear (or rank lower) when context-filtered
        excluded_ids: &'static [&'static str],
    }

    let ctx_queries = vec![
        CtxQuery {
            label: "Work meetings only",
            query: "What was discussed recently?",
            context_tags: vec!["meeting".into()],
            expected_ids: &["work_meeting_q4", "work_meeting_standup"],
            excluded_ids: &["personal_cooking", "learning_rust"],
        },
        CtxQuery {
            label: "Project Alpha scope",
            query: "What happened with the project?",
            context_tags: vec!["project-alpha".into()],
            expected_ids: &["project_alpha_design", "project_alpha_deploy"],
            excluded_ids: &["work_meeting_q4", "personal_fitness"],
        },
        CtxQuery {
            label: "Personal cooking",
            query: "What recipes or food items were involved?",
            context_tags: vec!["cooking".into()],
            expected_ids: &["personal_cooking", "personal_cooking_baking"],
            excluded_ids: &["work_meeting_q4", "learning_rust"],
        },
        CtxQuery {
            label: "Learning activities",
            query: "What have I been studying or learning?",
            context_tags: vec!["learning".into()],
            expected_ids: &["learning_rust", "learning_japanese"],
            excluded_ids: &["work_meeting_q4", "personal_cooking"],
        },
        CtxQuery {
            label: "Fitness context",
            query: "What exercise or running activities?",
            context_tags: vec!["fitness".into()],
            expected_ids: &["personal_fitness"],
            excluded_ids: &["work_meeting_q4", "personal_cooking"],
        },
    ];

    println!("{}", sep);
    println!("  Context-Scoped Search Results");
    println!("{}", sep);

    let mut total_found = 0;
    let mut total_expected = 0;
    let mut total_excluded_absent = 0;
    let mut total_excluded_checks = 0;
    let default_filters = Filters::default();

    for cq in &ctx_queries {
        // Context-scoped search (two-stage)
        let ctx_results = manager
            .search_with_context(cq.query, &cq.context_tags, &default_filters, 5)
            .await
            .unwrap_or_default();

        let ctx_ids: Vec<String> = ctx_results.iter().map(|r| r.memory.id.clone()).collect();

        // Also run plain search for comparison
        let plain_results = manager
            .search(cq.query, &default_filters, 5)
            .await
            .unwrap_or_default();
        let plain_ids: Vec<String> = plain_results.iter().map(|r| r.memory.id.clone()).collect();

        let expected_actual: Vec<String> = cq
            .expected_ids
            .iter()
            .filter_map(|eid| stored_ids.get(*eid).cloned())
            .collect();

        let found = expected_actual
            .iter()
            .filter(|eid| ctx_ids.contains(eid))
            .count();

        total_found += found;
        total_expected += expected_actual.len();

        // Check exclusions
        let excluded_actual: Vec<String> = cq
            .excluded_ids
            .iter()
            .filter_map(|eid| stored_ids.get(*eid).cloned())
            .collect();

        let excluded_absent = excluded_actual
            .iter()
            .filter(|eid| !ctx_ids.contains(eid))
            .count();

        total_excluded_absent += excluded_absent;
        total_excluded_checks += excluded_actual.len();

        let mark = if found == expected_actual.len() {
            "✓"
        } else {
            "~"
        };

        println!("{}", thin);
        println!("  {} {} (context: {:?})", mark, cq.label, cq.context_tags);
        println!("    Query: \"{}\"", cq.query);
        println!(
            "    Context search: found {}/{} expected | excluded {}/{} unwanted | {} total results",
            found,
            expected_actual.len(),
            excluded_absent,
            excluded_actual.len(),
            ctx_ids.len()
        );
        if !ctx_results.is_empty() {
            println!(
                "    Top scores: {:.3}, {:.3}, {:.3}",
                ctx_results.get(0).unwrap().score,
                ctx_results.get(1).unwrap().score,
                ctx_results.get(2).unwrap().score,
            );
        }
        println!(
            "    Plain search:   {:?}",
            plain_ids.iter().take(3).collect::<Vec<_>>()
        );
    }

    let recall = if total_expected > 0 {
        total_found as f64 / total_expected as f64
    } else {
        0.0
    };
    let exclusion_rate = if total_excluded_checks > 0 {
        total_excluded_absent as f64 / total_excluded_checks as f64
    } else {
        1.0
    };

    println!("{}", thin);
    println!("  SUMMARY");
    println!("{}", thin);
    println!(
        "  Context recall: {:.1}% ({}/{})",
        recall * 100.0,
        total_found,
        total_expected
    );
    println!(
        "  Exclusion rate: {:.1}% ({}/{} unwanted excluded)",
        exclusion_rate * 100.0,
        total_excluded_absent,
        total_excluded_checks
    );
    println!("{}", sep);

    // Two-stage retrieval should complete without errors and find reasonable results
    // Thresholds are conservative since context filtering depends on embedding similarity
    assert!(
        stored_ids.len() >= 8,
        "Too many store failures ({}/{})",
        ctx_memories.len() - stored_ids.len(),
        ctx_memories.len()
    );
    assert!(
        recall >= 0.40,
        "Context recall ({:.1}%) below minimum threshold (40%)",
        recall * 100.0
    );
}

// ═══════════════════════════════════════════════════════════════════════════
//  Real LLM Tests — require a downloaded GGUF model
// ═══════════════════════════════════════════════════════════════════════════
//
// These tests use the actual local LLM (llama.cpp + fastembed) instead of
// mock completions. This means:
//
//   - A GGUF model must be present (auto-downloaded on first run, ~1.1 GB)
//   - Tests are MUCH slower (LLM inference for each memory)
//   - Results depend on the specific model's quality
//
// Run:
//   cargo test --test evaluation evaluation_real_llm -- --ignored --nocapture

/// Helper: create a real LLM client from default config.
///
/// Uses auto-download so the GGUF model is fetched if missing.
/// Returns the client + the config (needed for MemoryConfig).
async fn create_real_client() -> (Box<dyn llm_mem::llm::LLMClient>, llm_mem::Config) {
    let mut config = llm_mem::Config::default();
    config.apply_env_overrides(); // Allow env vars to override defaults (e.g. models_dir)

    println!("  Backend: {:?}", config.effective_backend());
    println!("  Model dir: {}", config.local.models_dir);
    println!("  LLM model: {}", config.local.llm_model_file);
    println!("  Auto-download: {}", config.local.auto_download);

    let client = llm_mem::llm::create_llm_client(&config)
        .await
        .expect("Failed to create real LLM client — is the GGUF model available?");

    // Quick health check
    let healthy = client.health_check().await.unwrap_or(false);
    assert!(healthy, "Real LLM client failed health check");

    let status = client.get_status();
    println!("  LLM model loaded: {}", status.llm_model);
    println!("  Embedding model: {}", status.embedding_model);
    println!("  LLM available: {}", status.llm_available);
    println!("  Embedding available: {}", status.embedding_available);

    (client, config)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 11: Real LLM — Combined relations + context (Level 2+3)
// ═══════════════════════════════════════════════════════════════════════════

/// End-to-end evaluation of the full Level 2+3 feature set working together
/// with the real local LLM.
///
/// Tests the production scenario: memories stored with both context tags AND
/// relations, searched using both context-scoped retrieval and relation filters
/// simultaneously.
///
/// **Requirements:** GGUF model available (~1.1 GB)
#[tokio::test]
#[ignore]
async fn evaluation_real_llm_combined_l2_l3() {
    use llm_mem::types::Relation;

    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Real LLM Combined L2+L3 Evaluation");
    println!("══════════════════════════════════════════════════════════════");
    println!("  Initializing real LLM client...");

    let (client, config) = create_real_client().await;

    let store = make_eval_store();
    let memory_config = MemoryConfig {
        deduplicate: false,
        search_similarity_threshold: None,
        ..config.memory.clone()
    };

    let manager = MemoryManager::new(Box::new(store), client, memory_config);

    // Rich memories with both context and relations
    struct FullMemory {
        id: &'static str,
        content: &'static str,
        context: Vec<String>,
        relations: Vec<Relation>,
    }

    let memories = vec![
        FullMemory {
            id: "alice_project_x",
            content: "Alice is leading the backend development for Project X, building a real-time analytics engine in Rust",
            context: vec!["work".into(), "project-x".into(), "backend".into()],
            relations: vec![
                Relation {
                    source: "Alice".into(),
                    relation: "LEADS".into(),
                    target: "Project X".into(),
                    strength: None,
                },
                Relation {
                    source: "Alice".into(),
                    relation: "USES".into(),
                    target: "Rust".into(),
                    strength: None,
                },
            ],
        },
        FullMemory {
            id: "bob_project_x",
            content: "Bob is handling the frontend for Project X, building interactive dashboards with React and D3",
            context: vec!["work".into(), "project-x".into(), "frontend".into()],
            relations: vec![
                Relation {
                    source: "Bob".into(),
                    relation: "WORKS_ON".into(),
                    target: "Project X".into(),
                    strength: None,
                },
                Relation {
                    source: "Bob".into(),
                    relation: "USES".into(),
                    target: "React".into(),
                    strength: None,
                },
            ],
        },
        FullMemory {
            id: "alice_hobby_painting",
            content: "Alice enjoys watercolor painting on weekends, recently completed a landscape series",
            context: vec!["personal".into(), "hobby".into(), "art".into()],
            relations: vec![Relation {
                source: "Alice".into(),
                relation: "ENJOYS".into(),
                target: "Watercolor Painting".into(),
                strength: None,
            }],
        },
        FullMemory {
            id: "bob_hobby_chess",
            content: "Bob plays competitive chess online, currently rated 1800 on Chess.com in rapid format",
            context: vec!["personal".into(), "hobby".into(), "gaming".into()],
            relations: vec![Relation {
                source: "Bob".into(),
                relation: "PLAYS".into(),
                target: "Chess".into(),
                strength: None,
            }],
        },
        FullMemory {
            id: "team_meeting_retro",
            content: "Sprint retrospective: discussed improving code review turnaround time, agreed on 24-hour SLA",
            context: vec!["work".into(), "meeting".into(), "team".into()],
            relations: vec![Relation {
                source: "Team".into(),
                relation: "DECIDED".into(),
                target: "24-hour Review SLA".into(),
                strength: None,
            }],
        },
        FullMemory {
            id: "alice_learns_ml",
            content: "Alice started a machine learning course on Coursera, currently studying neural networks and backpropagation",
            context: vec!["learning".into(), "ml".into(), "coursera".into()],
            relations: vec![
                Relation {
                    source: "Alice".into(),
                    relation: "STUDIES".into(),
                    target: "Machine Learning".into(),
                    strength: None,
                },
                Relation {
                    source: "Alice".into(),
                    relation: "USES".into(),
                    target: "Coursera".into(),
                    strength: None,
                },
            ],
        },
    ];

    println!(
        "\n  Storing {} memories (context + relations) through real LLM...",
        memories.len()
    );
    println!();

    let mut stored_ids: HashMap<String, String> = HashMap::new();

    for (i, mem) in memories.iter().enumerate() {
        let mut metadata = MemoryMetadata::new(MemoryType::Factual);
        metadata.context = mem.context.clone();
        metadata.relations = mem.relations.clone();
        print!(
            "  [{}/{}] '{}' (ctx: {}, rel: {})...",
            i + 1,
            memories.len(),
            mem.id,
            mem.context.len(),
            mem.relations.len()
        );
        match manager.store(mem.content.to_string(), metadata).await {
            Ok(id) => {
                println!(" ✓");
                stored_ids.insert(mem.id.to_string(), id);
            }
            Err(e) => {
                println!(" ✗ {}", e);
            }
        }
    }

    println!("\n  Stored: {}/{}", stored_ids.len(), memories.len());

    assert!(
        stored_ids.len() >= 4,
        "Too many store failures ({}/{})",
        memories.len() - stored_ids.len(),
        memories.len()
    );

    let sep = "═".repeat(72);
    let thin = "─".repeat(72);
    let default_filters = Filters::default();

    println!("\n{}", sep);
    println!("  Combined L2+L3 Search Tests");
    println!("{}", sep);

    // Test 1: Context-scoped search for work memories
    println!("{}", thin);
    println!("  Test: Context='work' + query about projects");
    let work_results = manager
        .search_with_context(
            "What are people working on?",
            &["work".into()],
            &default_filters,
            5,
        )
        .await
        .unwrap_or_default();
    println!("    Results: {}", work_results.len());
    for r in &work_results {
        println!(
            "    - {} (score: {:.3}, ctx: {:?})",
            &r.memory.id[..r.memory.id.len().min(20)],
            r.score,
            r.memory.metadata.context
        );
    }

    // Test 2: Context='personal' — should get hobbies, not work
    println!("{}", thin);
    println!("  Test: Context='personal' + query about activities");
    let personal_results = manager
        .search_with_context(
            "What activities and hobbies?",
            &["personal".into()],
            &default_filters,
            5,
        )
        .await
        .unwrap_or_default();
    println!("    Results: {}", personal_results.len());
    for r in &personal_results {
        println!(
            "    - {} (score: {:.3}, ctx: {:?})",
            &r.memory.id[..r.memory.id.len().min(20)],
            r.score,
            r.memory.metadata.context
        );
    }

    // Test 3: Context + relation filter combined
    println!("{}", thin);
    println!("  Test: Context='work' + Relation USES Rust");
    let combined_filters = Filters {
        relations: Some(vec![llm_mem::types::RelationFilter {
            relation: "USES".into(),
            target: "Rust".into(),
        }]),
        ..Default::default()
    };
    let combined_results = manager
        .search_with_context(
            "Who uses Rust at work?",
            &["work".into()],
            &combined_filters,
            5,
        )
        .await
        .unwrap_or_default();
    println!("    Results: {}", combined_results.len());
    for r in &combined_results {
        let rels: Vec<String> = r
            .memory
            .metadata
            .relations
            .iter()
            .map(|rel| format!("{}→{}", rel.relation, rel.target))
            .collect();
        println!(
            "    - {} (score: {:.3}, rels: {:?})",
            &r.memory.id[..r.memory.id.len().min(20)],
            r.score,
            rels
        );
    }

    let alice_project_x_id = stored_ids.get("alice_project_x");
    let found_alice = alice_project_x_id
        .map(|id| combined_results.iter().any(|r| r.memory.id == *id))
        .unwrap_or(false);
    println!(
        "    Alice's Rust project found: {}",
        if found_alice { "✓" } else { "~" }
    );

    // Test 4: Delete and verify cleanup
    println!("{}", thin);
    println!("  Test: Delete multi-vector memory");
    if let Some(id) = stored_ids.get("alice_project_x") {
        manager.delete(id).await.unwrap();
        let gone = manager.get(id).await.unwrap().is_none();
        println!(
            "    Delete alice_project_x: {}",
            if gone {
                "✓ cleaned up"
            } else {
                "✗ still exists!"
            }
        );
        assert!(gone, "Multi-vector memory should be fully deleted");
    }

    // Test 5: Update relations
    println!("{}", thin);
    println!("  Test: Update relations on existing memory");
    if let Some(id) = stored_ids.get("bob_project_x") {
        let new_rels = vec![
            Relation {
                source: "Bob".into(),
                relation: "WORKS_ON".into(),
                target: "Project X".into(),
                strength: None,
            },
            Relation {
                source: "Bob".into(),
                relation: "USES".into(),
                target: "React".into(),
                strength: None,
            },
            Relation {
                source: "Bob".into(),
                relation: "USES".into(),
                target: "TypeScript".into(),
                strength: None,
            },
        ];
        manager.update(id, None, Some(new_rels)).await.unwrap();
        let updated = manager.get(id).await.unwrap().unwrap();
        println!(
            "    Bob's relations updated: {} → {} relations ✓",
            2,
            updated.metadata.relations.len()
        );
        assert_eq!(updated.metadata.relations.len(), 3);
    }

    println!("{}", thin);
    println!("  All combined L2+L3 tests completed!");
    println!("{}", sep);
}
