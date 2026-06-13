//! Tough vision test: real LLM vision model decodes a minimal image.
//!
//! Creates a 1x1 red pixel PNG, ingests it through the full pipeline,
//! and verifies the LLM generates a non-trivial image description.
//! Skips gracefully if vision is not configured.
//!
//! Run: `cargo tough-tests`

#![cfg(feature = "tough-tests")]

use std::sync::Arc;

use llm_mem::{
    Config, LanceDBConfig, LanceDBStore,
    config::MemoryConfig,
    memory::MemoryManager,
    operations::{IngestRequest, MemoryOperations},
};

use tempfile::TempDir;

const DIM: usize = 384;

fn make_config() -> MemoryConfig {
    MemoryConfig {
        max_memories: 1000,
        similarity_threshold: 0.5,
        max_search_results: 50,
        memory_ttl_hours: None,
        auto_summary_threshold: 32768,
        enable_abstraction: false,
        auto_metadata_analysis: false,
        llm_importance_scoring: false,
        skip_duplicates: false,
        merge_threshold: 0.75,
        search_similarity_threshold: None,
        max_content_length: 32768,
        document_chunk_size: 4000,
        use_llm_query_classification: false,
        llm_format_detection: false,
        llm_fallback_parsing: false,
        chunk_threshold_chars: 2500,
        chunk_size_chars: 1000,
        chunk_overlap_chars: 100,
        max_cascade_fanout: 5000,
        raw_content_scan_limit: 5000,
        max_list_limit: 10000,
        max_total_candidates: 10000,
        auto_link_threshold: 0.75,
        auto_link_max_relations: 10,
        session_token_budget: 0,
        dry_run: false,
        near_duplicate_threshold: 0.0,
        contradiction_detection: false,
        access_decay_hours: 0,
        llm_relation_validation: false,
        auto_link_primary_pct: 60,
        auto_link_context_pct: 25,
        auto_link_relation_pct: 15,
        use_multi_vector_reranking: false,
        rrf_k: 60.0,
    }
}

const PNG_1X1_RED_B64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

#[tokio::test]
async fn test_tough_vision_image_description() {
    println!();
    println!("═════════════════════════════════════════════════════════════════");
    println!("  TOUGH VISION — Real LLM Vision Model Image Description");
    println!("═════════════════════════════════════════════════════════════════");
    println!();

    let mut config = Config::default();
    config.apply_env_overrides();

    if !config.llm.vision_enabled {
        println!("  ⚠ SKIP: vision_enabled is false in config.");
        println!("    Set vision_enabled = true and configure mmproj_file");
        println!("    (or use an API provider with a vision-capable model).");
        return;
    }

    println!("  Vision enabled: true");

    let client: Box<dyn llm_mem::llm::LLMClient> =
        match llm_mem::llm::create_llm_client(&config).await {
            Ok(c) => {
                let s = c.get_status();
                println!("  LLM backend: {} (model: {})", s.backend, s.llm_model);
                println!("  Embedding model: {}", s.embedding_model);
                c
            }
            Err(e) => {
                println!("  ✗ Cannot create LLM client: {}", e);
                return;
            }
        };

    let tmp = TempDir::new().expect("tempdir");
    let store = Box::new(
        LanceDBStore::new(LanceDBConfig {
            table_name: "tough-vision".into(),
            database_path: tmp.path().to_path_buf(),
            embedding_dimension: DIM,
        })
        .await
        .expect("lance db"),
    );
    let mem_config = make_config();
    let manager = Arc::new(MemoryManager::new(
        store,
        client,
        mem_config,
        None,
        llm_mem::memory::metrics::LlmBackendType::Local,
    ));

    let ops = MemoryOperations::new(manager, Some("tough-vision-user".into()), None, 100);

    println!();
    println!("  Ingesting 1x1 red pixel PNG (69 bytes) — calling vision model...");
    println!();

    let resp = match ops
        .ingest(
            IngestRequest {
                content: PNG_1X1_RED_B64.to_string(),
                content_encoding: Some("base64".to_string()),
                format_hint: Some("png".to_string()),
                file_name: Some("test.png".to_string()),
                bank: None,
                auto_link: Some(false),
                generate_abstractions: Some(true),
                max_chunk_size: None,
                metadata: None,
                source: None,
                describe_images: Some(true),
            },
            None,
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            println!("  ✗ Ingest failed: {}", e);
            panic!("tough vision test: ingest error: {}", e);
        }
    };

    assert!(resp.success, "Ingest should succeed: {:?}", resp.error);

    let data = resp.data.expect("response data");
    let status = data["status"].as_str().unwrap_or("unknown");
    println!("  Ingest status: {}", status);
    assert_eq!(status, "success", "Ingest status should be success");

    let l0 = data["l0_chunks"].as_array().expect("l0_chunks");
    println!("  L0 chunks: {}", l0.len());
    assert!(!l0.is_empty(), "Should create L0 chunks for image metadata");

    let l1 = data["l1_abstractions"].as_array().expect("l1_abstractions");
    println!("  L1 abstractions: {}", l1.len());

    let vision = data["vision_status"].as_object().expect("vision_status");
    let outcome = vision["outcome"].as_str().unwrap_or("missing");
    let generated = vision["descriptions_generated"].as_i64().unwrap_or(0);
    println!("  Vision outcome: {}", outcome);
    println!("  Descriptions generated: {}", generated);
    println!();

    match outcome {
        "succeeded" => {
            assert!(!l1.is_empty(), "Should have L1 image description");
            let description = l1[0]["content_preview"].as_str().unwrap_or("");
            println!("  Image description preview: \"{}\"", description);
            println!();

            assert!(
                description.len() > 5,
                "Description should be meaningful (got '{}', {} chars)",
                description,
                description.len()
            );

            let l1_mem_id = l1[0]["memory_id"].as_str().unwrap();
            println!("  L1 memory ID: {}", l1_mem_id);

            let abstraction_type = l1[0]["abstraction_type"].as_str().unwrap();
            assert_eq!(abstraction_type, "image_description");
            assert_eq!(l1[0]["layer"].as_i64().unwrap(), 1);

            let sep = "═".repeat(64);
            println!("{}", sep);
            println!("  TOUGH VISION — REAL LLM VISION — PASSED");
            println!("{}", sep);
            println!("  Model:      {}", resp.message);
            println!("  Preview:    \"{}\"", description);
            println!("{}", sep);
        }
        "not_configured" => {
            println!("  ⚠ SKIP: Vision is not configured on this backend.");
            println!("    For local: set vision_enabled=true and configure mmproj_file.");
            println!("    For API: use a vision-capable model (e.g., gpt-4o).");
            let detail = vision.get("detail").and_then(|d| d.as_str()).unwrap_or("");
            if !detail.is_empty() {
                println!("    Detail: {}", detail);
            }
            return;
        }
        "unavailable" => {
            let detail = vision.get("detail").and_then(|d| d.as_str()).unwrap_or("");
            println!("  ⚠ SKIP: Vision unavailable. {}", detail);
            return;
        }
        other => {
            let warnings = data["warnings"]
                .as_array()
                .map(|w| {
                    w.iter()
                        .map(|v| v.as_str().unwrap_or(""))
                        .collect::<Vec<_>>()
                        .join("; ")
                })
                .unwrap_or_default();
            println!("  ✗ Vision outcome '{}' is unexpected.", other);
            if !warnings.is_empty() {
                println!("    Warnings: {}", warnings);
            }
            panic!("tough vision test: unexpected vision outcome '{}'", other);
        }
    }
}
