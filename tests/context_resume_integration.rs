//! Integration tests for the progressive context resume feature.

use chrono::{Duration, Utc};
use llm_mem::{
    LanceDBConfig, LanceDBStore,
    config::MemoryConfig,
    memory::MemoryManager,
    operations::{ContextResumeService, GetContextResumeRequest, MemoryOperations, ResumeFilters},
    types::{MemoryMetadata, MemoryState},
};
use std::sync::Arc;
use tempfile::TempDir;

#[path = "common/mod.rs"]
mod common_cr;
use common_cr::{DIM, make_mock_client};

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
        auto_link_threshold: 0.0,
        auto_link_max_relations: 10,
        session_token_budget: 0,
        dry_run: false,
        near_duplicate_threshold: 0.92,
        contradiction_detection: false,
        access_decay_hours: 168,
        llm_relation_validation: false,
        auto_link_primary_pct: 60,
        auto_link_context_pct: 25,
        auto_link_relation_pct: 15,
        use_multi_vector_reranking: false,
    }
}

async fn make_manager() -> (Arc<MemoryManager>, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let config = LanceDBConfig {
        table_name: format!("test_resume_{}", uuid::Uuid::new_v4().simple()),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: DIM,
    };
    let store = LanceDBStore::new(config).await.unwrap();
    let manager = MemoryManager::new(
        Box::new(store),
        Box::new(make_mock_client()),
        make_config(),
        None,
        llm_mem::memory::metrics::LlmBackendType::Local,
    );
    (Arc::new(manager), temp_dir)
}

fn make_metadata(layer: i32) -> MemoryMetadata {
    let mut m = MemoryMetadata::new();
    m.state = MemoryState::Active;
    m.layer.level = layer;
    m
}

async fn store_mem(manager: &MemoryManager, id: &str, when: chrono::DateTime<Utc>, layer: i32) {
    let mut meta = make_metadata(layer);
    meta.layer.level = layer;
    let content = format!("content for {id} at layer {layer}");
    manager
        .store_with_options(
            content,
            meta,
            llm_mem::memory::StoreOptions {
                event_at: Some(when),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn test_basic_progressive_resume() {
    let (manager, _tmp) = make_manager().await;
    let now = Utc::now();

    for i in 0..30 {
        let when = now - Duration::days(i);
        store_mem(&manager, &format!("l0-{i}"), when, 0).await;
    }

    let svc = ContextResumeService::new(manager);
    let resp = svc
        .get_context_resume(None, 30 * 86400, 2.0, 5, 20, ResumeFilters::default())
        .await
        .unwrap();

    assert_eq!(resp.segments.len(), 5);
    assert!(resp.total_memories > 0);

    // Segments ordered oldest → newest.
    for window in resp.segments.windows(2) {
        let prev_end: chrono::DateTime<Utc> = window[0].end.parse().unwrap();
        let next_start: chrono::DateTime<Utc> = window[1].start.parse().unwrap();
        assert!(prev_end <= next_start, "Segments not chronological");
    }

    // Oldest segment should target layer 3, newest should target layer 0.
    assert_eq!(resp.segments[0].layer, 3);
    assert_eq!(resp.segments[4].layer, 0);
}

#[tokio::test]
async fn test_segment_layer_mapping() {
    let (manager, _tmp) = make_manager().await;
    let now = Utc::now();

    for layer in 0..=3 {
        for i in 0..5 {
            let when = now - Duration::days(i);
            store_mem(&manager, &format!("l{layer}-{i}"), when, layer).await;
        }
    }

    let svc = ContextResumeService::new(manager);
    let resp = svc
        .get_context_resume(None, 7 * 86400, 2.0, 4, 20, ResumeFilters::default())
        .await
        .unwrap();

    let layers: Vec<i32> = resp.segments.iter().map(|s| s.layer).collect();
    assert_eq!(layers, vec![3, 2, 1, 0]);
}

#[tokio::test]
async fn test_layer_fallback_to_l0() {
    let (manager, _tmp) = make_manager().await;
    let now = Utc::now();

    // Only L0 memories — no L1/L2/L3.
    for i in 0..30 {
        let when = now - Duration::days(i);
        store_mem(&manager, &format!("l0-{i}"), when, 0).await;
    }

    let svc = ContextResumeService::new(manager);
    let resp = svc
        .get_context_resume(None, 30 * 86400, 2.0, 5, 20, ResumeFilters::default())
        .await
        .unwrap();

    // Should still find memories via fallback.
    assert!(resp.total_memories > 0);
}

#[tokio::test]
async fn test_max_per_segment_enforced() {
    let (manager, _tmp) = make_manager().await;
    let now = Utc::now();

    for i in 0..100 {
        let when = now - Duration::minutes(i);
        store_mem(&manager, &format!("m-{i}"), when, 0).await;
    }

    let svc = ContextResumeService::new(manager);
    let resp = svc
        .get_context_resume(None, 86400, 2.0, 2, 5, ResumeFilters::default())
        .await
        .unwrap();

    for seg in &resp.segments {
        assert!(seg.count <= 5);
    }
}

#[tokio::test]
async fn test_empty_store() {
    let (manager, _tmp) = make_manager().await;

    let svc = ContextResumeService::new(manager);
    let resp = svc
        .get_context_resume(None, 30 * 86400, 2.0, 5, 20, ResumeFilters::default())
        .await
        .unwrap();

    assert_eq!(resp.total_memories, 0);
    assert_eq!(resp.segments.len(), 5);
    for seg in &resp.segments {
        assert_eq!(seg.count, 0);
        assert!(seg.memories.is_empty());
    }
}

#[tokio::test]
async fn test_custom_decay_factor_uniform() {
    let (manager, _tmp) = make_manager().await;
    let now = Utc::now();

    for i in 0..30 {
        let when = now - Duration::days(i);
        store_mem(&manager, &format!("m-{i}"), when, 0).await;
    }

    // decay_factor=1.0 → all segments should have equal duration.
    let svc = ContextResumeService::new(manager);
    let resp = svc
        .get_context_resume(None, 30 * 86400, 1.0, 3, 20, ResumeFilters::default())
        .await
        .unwrap();

    let durations: Vec<i64> = resp.segments.iter().map(|s| s.duration_secs).collect();
    let d0 = durations[0] as f64;
    let d1 = durations[1] as f64;
    let d2 = durations[2] as f64;

    assert!((d0 - d1).abs() / d0 < 0.05);
    assert!((d1 - d2).abs() / d1 < 0.05);
}

#[tokio::test]
async fn test_via_memory_operations() {
    let (manager, _tmp) = make_manager().await;
    let now = Utc::now();

    for i in 0..15 {
        let when = now - Duration::days(i);
        store_mem(&manager, &format!("m-{i}"), when, 0).await;
    }

    let ops = MemoryOperations::new(manager, None, None, 100);
    let req = GetContextResumeRequest {
        lookback: Some("15d".to_string()),
        segments: Some(3),
        ..Default::default()
    };

    let resp = ops.context_resume(req).await.unwrap();
    assert!(resp.success);
    let data = resp.data.unwrap();
    let seg_count = data["segment_count"].as_u64().unwrap();
    assert_eq!(seg_count, 3);
}
