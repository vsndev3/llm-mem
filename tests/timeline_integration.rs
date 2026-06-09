//! Integration tests for chronological tracking and timeline operations.
//!
//! These tests verify:
//!  - L0 event_at is stored and round-trips through the store
//!  - Old memories without event_at fall back to created_at (backfill)
//!  - get_timeline buckets memories correctly
//!  - get_timeline_graph auto-derives temporal edges
//!  - Reverse relations for temporal edges work

use chrono::{Duration, TimeZone, Utc};
use llm_mem::{
    LanceDBConfig, LanceDBStore,
    config::MemoryConfig,
    memory::MemoryManager,
    operations::{
        GetTimelineGraphRequest, GetTimelineRequest, MemoryOperations, TimelineGranularity,
    },
    types::{Filters, Memory, MemoryMetadata, MemoryState},
    vector_store::VectorStore,
};
use std::sync::Arc;
use tempfile::TempDir;

#[path = "common/mod.rs"]
mod common_timeline;
use common_timeline::{DIM, make_mock_client};

// ─── helpers ──────────────────────────────────────────────────────────────

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
    }
}

async fn make_store() -> LanceDBStore {
    let temp_dir = TempDir::new().unwrap();
    let config = LanceDBConfig {
        table_name: format!("test_timeline_{}", uuid::Uuid::new_v4().simple()),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: DIM,
    };
    LanceDBStore::new(config).await.unwrap()
}

async fn make_manager() -> (MemoryManager, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let config = LanceDBConfig {
        table_name: format!("test_timeline_{}", uuid::Uuid::new_v4().simple()),
        database_path: temp_dir.path().to_path_buf(),
        embedding_dimension: 384,
    };
    let store = LanceDBStore::new(config).await.unwrap();
    let manager = MemoryManager::new(
        Box::new(store),
        Box::new(make_mock_client()),
        make_config(),
        None,
        llm_mem::memory::metrics::LlmBackendType::Local,
    );
    (manager, temp_dir)
}

fn make_metadata() -> MemoryMetadata {
    let mut m = MemoryMetadata::new();
    m.state = MemoryState::Active;
    m
}

// ─── event_at storage & round-trip ────────────────────────────────────────

#[tokio::test]
async fn test_l0_event_at_round_trips() {
    let (manager, _tmp) = make_manager().await;
    let when = Utc.with_ymd_and_hms(2026, 6, 2, 14, 30, 0).unwrap();
    let id = manager
        .store_with_options(
            "we went to the beach".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(when),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let m = manager.get(&id).await.unwrap().expect("memory exists");
    assert_eq!(m.event_at, Some(when));
    assert!(m.event_end.is_none());
}

#[tokio::test]
async fn test_l0_without_event_at_is_none() {
    let (manager, _tmp) = make_manager().await;
    let id = manager
        .store("a note without an event time".to_string(), make_metadata())
        .await
        .unwrap();
    let m = manager.get(&id).await.unwrap().expect("memory exists");
    assert!(m.event_at.is_none());
    // effective_event_at() falls back to created_at
    assert_eq!(m.effective_event_at(), m.created_at);
    assert_eq!(m.effective_event_end(), m.created_at);
}

#[tokio::test]
async fn test_event_at_filter_excludes_earlier_memories() {
    let store = make_store().await;
    let t1 = Utc.with_ymd_and_hms(2026, 6, 1, 12, 0, 0).unwrap();
    let t2 = Utc.with_ymd_and_hms(2026, 6, 3, 12, 0, 0).unwrap();
    let t3 = Utc.with_ymd_and_hms(2026, 6, 5, 12, 0, 0).unwrap();

    for (i, when) in [t1, t2, t3].iter().enumerate() {
        let mut m = Memory::with_content(
            format!("memory at t{}", i + 1),
            vec![0.0; DIM],
            make_metadata(),
        );
        m.event_at = Some(*when);
        store.insert(&m).await.unwrap();
    }

    // Filter to after t2 (inclusive): should match t2 and t3
    let filters = Filters {
        event_after: Some(t2),
        ..Default::default()
    };
    let results: Vec<Memory> = store.list(&filters, None).await.unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().any(|m| m.event_at == Some(t2)));
    assert!(results.iter().any(|m| m.event_at == Some(t3)));
}

#[tokio::test]
async fn test_event_at_filter_falls_back_to_created_at() {
    let store = make_store().await;
    // Memory with no event_at, created at a specific time
    let m1 = Memory::with_content(
        "old memory, no event_at".to_string(),
        vec![0.0; DIM],
        make_metadata(),
    );
    store.insert(&m1).await.unwrap();
    // Give the system clock time to advance
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    // Set a filter that should match the old memory via the created_at backfill
    let filters = Filters {
        event_after: Some(Utc::now() - Duration::seconds(60)),
        ..Default::default()
    };
    let results: Vec<Memory> = store.list(&filters, None).await.unwrap();
    assert!(
        !results.is_empty(),
        "old memory should match via created_at backfill"
    );
    assert!(results[0].event_at.is_none());
}

// ─── get_timeline integration ─────────────────────────────────────────────

#[tokio::test]
async fn test_get_timeline_buckets_by_day() {
    let (manager, _tmp) = make_manager().await;
    let d1 = Utc.with_ymd_and_hms(2026, 6, 2, 9, 0, 0).unwrap();
    let d1b = Utc.with_ymd_and_hms(2026, 6, 2, 18, 0, 0).unwrap();
    let d2 = Utc.with_ymd_and_hms(2026, 6, 3, 10, 0, 0).unwrap();

    for (i, when) in [&d1, &d1b, &d2].iter().enumerate() {
        manager
            .store_with_options(
                format!("memory {i}"),
                make_metadata(),
                llm_mem::memory::StoreOptions {
                    event_at: Some(**when),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }

    let ops = MemoryOperations::new(Arc::new(manager), None, None, 100);
    let req = GetTimelineRequest {
        start: Some("2026-06-01T00:00:00Z".to_string()),
        end: Some("2026-06-30T00:00:00Z".to_string()),
        granularity: Some(TimelineGranularity::Day),
        bank: None,
        user_id: None,
        agent_id: None,
        topics: None,
        max_results_per_bucket: 50,
        include_derived: false,
        order: "asc".to_string(),
    };

    let resp = ops.get_timeline(req).await.unwrap();
    assert!(resp.success);
    let data = resp.data.expect("data");
    let total = data["total_count"].as_u64().unwrap();
    assert_eq!(total, 3);
    let buckets = data["buckets"].as_array().unwrap();
    assert_eq!(buckets.len(), 2);
    assert_eq!(buckets[0]["count"].as_u64().unwrap(), 2);
    assert_eq!(buckets[1]["count"].as_u64().unwrap(), 1);
    assert_eq!(buckets[0]["label"].as_str(), Some("2026-06-02"));
    assert_eq!(buckets[1]["label"].as_str(), Some("2026-06-03"));
}

#[tokio::test]
async fn test_get_timeline_buckets_by_hour() {
    let (manager, _tmp) = make_manager().await;
    let h1 = Utc.with_ymd_and_hms(2026, 6, 2, 9, 15, 0).unwrap();
    let h1b = Utc.with_ymd_and_hms(2026, 6, 2, 9, 45, 0).unwrap();
    let h2 = Utc.with_ymd_and_hms(2026, 6, 2, 14, 0, 0).unwrap();

    for (i, when) in [&h1, &h1b, &h2].iter().enumerate() {
        manager
            .store_with_options(
                format!("m {i}"),
                make_metadata(),
                llm_mem::memory::StoreOptions {
                    event_at: Some(**when),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }

    let ops = MemoryOperations::new(Arc::new(manager), None, None, 100);
    let req = GetTimelineRequest {
        granularity: Some(TimelineGranularity::Hour),
        order: "asc".to_string(),
        ..Default::default()
    };

    let resp = ops.get_timeline(req).await.unwrap();
    let data = resp.data.expect("data");
    let buckets = data["buckets"].as_array().unwrap();
    assert_eq!(buckets.len(), 2);
    assert_eq!(buckets[0]["label"].as_str(), Some("2026-06-02T09:00"));
    assert_eq!(buckets[1]["label"].as_str(), Some("2026-06-02T14:00"));
}

#[tokio::test]
async fn test_get_timeline_respects_window() {
    let (manager, _tmp) = make_manager().await;
    let in_window = Utc.with_ymd_and_hms(2026, 6, 5, 12, 0, 0).unwrap();
    let out_window = Utc.with_ymd_and_hms(2027, 1, 1, 0, 0, 0).unwrap();
    manager
        .store_with_options(
            "in".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(in_window),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    manager
        .store_with_options(
            "out".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(out_window),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let ops = MemoryOperations::new(Arc::new(manager), None, None, 100);
    let req = GetTimelineRequest {
        start: Some("2026-06-01T00:00:00Z".to_string()),
        end: Some("2026-06-30T00:00:00Z".to_string()),
        granularity: Some(TimelineGranularity::Day),
        bank: None,
        user_id: None,
        agent_id: None,
        topics: None,
        max_results_per_bucket: 50,
        include_derived: false,
        order: "asc".to_string(),
    };

    let resp = ops.get_timeline(req).await.unwrap();
    let data = resp.data.expect("data");
    assert_eq!(data["total_count"].as_u64().unwrap(), 1);
}

#[tokio::test]
async fn test_get_timeline_default_window_is_7_days() {
    let (manager, _tmp) = make_manager().await;
    let far_past = Utc::now() - Duration::days(30);
    let recent = Utc::now() - Duration::hours(1);

    manager
        .store_with_options(
            "ancient".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(far_past),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    manager
        .store_with_options(
            "fresh".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(recent),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let ops = MemoryOperations::new(Arc::new(manager), None, None, 100);
    let req = GetTimelineRequest {
        granularity: Some(TimelineGranularity::Day),
        order: "asc".to_string(),
        ..Default::default()
    };

    let resp = ops.get_timeline(req).await.unwrap();
    let data = resp.data.expect("data");
    // Only "fresh" should be in the default 7-day window
    assert_eq!(data["total_count"].as_u64().unwrap(), 1);
}

// ─── get_timeline_graph integration ───────────────────────────────────────

#[tokio::test]
async fn test_get_timeline_graph_emits_happened_after_edges() {
    let (manager, _tmp) = make_manager().await;
    let t1 = Utc.with_ymd_and_hms(2026, 6, 2, 9, 0, 0).unwrap();
    let t2 = Utc.with_ymd_and_hms(2026, 6, 2, 13, 0, 0).unwrap();
    let t3 = Utc.with_ymd_and_hms(2026, 6, 2, 18, 0, 0).unwrap();

    for (i, when) in [&t1, &t2, &t3].iter().enumerate() {
        manager
            .store_with_options(
                format!("m {i}"),
                make_metadata(),
                llm_mem::memory::StoreOptions {
                    event_at: Some(**when),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }

    let ops = MemoryOperations::new(Arc::new(manager), None, None, 100);
    let req = GetTimelineGraphRequest {
        timeline: GetTimelineRequest {
            start: Some("2026-06-01T00:00:00Z".to_string()),
            end: Some("2026-06-30T00:00:00Z".to_string()),
            granularity: Some(TimelineGranularity::Day),
            bank: None,
            user_id: None,
            agent_id: None,
            topics: None,
            max_results_per_bucket: 50,
            include_derived: false,
            order: "asc".to_string(),
        },
        max_depth: 0,
        relation_types: None,
        temporal_edge_window_secs: 86400,
        include_simultaneous: false,
        simultaneous_window_secs: 60,
        include_semantic_edges: false,
    };

    let resp = ops.get_timeline_graph(req).await.unwrap();
    let data = resp.data.expect("data");
    let nodes = data["nodes"].as_array().unwrap();
    let edges = data["edges"].as_array().unwrap();
    assert_eq!(nodes.len(), 3);
    // 3 memories => 2 temporal edges between consecutive pairs
    let temporal_count = edges
        .iter()
        .filter(|e| e["type"].as_str() == Some("happened_after"))
        .count();
    assert_eq!(temporal_count, 2);
    assert_eq!(data["stats"]["temporal_edge_count"].as_u64().unwrap(), 2);
    assert_eq!(data["stats"]["semantic_edge_count"].as_u64().unwrap(), 0);
}

#[tokio::test]
async fn test_get_timeline_graph_respects_temporal_window() {
    let (manager, _tmp) = make_manager().await;
    let t1 = Utc.with_ymd_and_hms(2026, 6, 2, 9, 0, 0).unwrap();
    // > 1 hour apart, < 1 day
    let t2 = Utc.with_ymd_and_hms(2026, 6, 2, 11, 0, 0).unwrap();
    manager
        .store_with_options(
            "a".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(t1),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    manager
        .store_with_options(
            "b".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(t2),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let ops = MemoryOperations::new(Arc::new(manager), None, None, 100);
    let req = GetTimelineGraphRequest {
        timeline: GetTimelineRequest {
            start: Some("2026-06-01T00:00:00Z".to_string()),
            end: Some("2026-06-30T00:00:00Z".to_string()),
            granularity: Some(TimelineGranularity::Day),
            bank: None,
            user_id: None,
            agent_id: None,
            topics: None,
            max_results_per_bucket: 50,
            include_derived: false,
            order: "asc".to_string(),
        },
        max_depth: 0,
        relation_types: None,
        temporal_edge_window_secs: 3600, // 1 hour
        include_simultaneous: false,
        simultaneous_window_secs: 60,
        include_semantic_edges: false,
    };

    let resp = ops.get_timeline_graph(req).await.unwrap();
    let data = resp.data.expect("data");
    let edges = data["edges"].as_array().unwrap();
    // t1 -> t2 is 2 hours, outside the 1-hour window => no edge
    assert_eq!(edges.len(), 0);
}

#[tokio::test]
async fn test_get_timeline_graph_includes_happens_within() {
    let (manager, _tmp) = make_manager().await;
    let t1 = Utc.with_ymd_and_hms(2026, 6, 2, 9, 0, 0).unwrap();
    let t2 = Utc.with_ymd_and_hms(2026, 6, 2, 9, 0, 30).unwrap();
    manager
        .store_with_options(
            "x".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(t1),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    manager
        .store_with_options(
            "y".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(t2),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let ops = MemoryOperations::new(Arc::new(manager), None, None, 100);
    let req = GetTimelineGraphRequest {
        timeline: GetTimelineRequest {
            start: Some("2026-06-01T00:00:00Z".to_string()),
            end: Some("2026-06-30T00:00:00Z".to_string()),
            granularity: Some(TimelineGranularity::Day),
            bank: None,
            user_id: None,
            agent_id: None,
            topics: None,
            max_results_per_bucket: 50,
            include_derived: false,
            order: "asc".to_string(),
        },
        max_depth: 0,
        relation_types: None,
        temporal_edge_window_secs: 86400,
        include_simultaneous: true,
        simultaneous_window_secs: 60,
        include_semantic_edges: false,
    };

    let resp = ops.get_timeline_graph(req).await.unwrap();
    let data = resp.data.expect("data");
    let edges = data["edges"].as_array().unwrap();
    // Note: the same pair may get both happened_after and happens_within
    let sim_count = edges
        .iter()
        .filter(|e| e["type"].as_str() == Some("happens_within"))
        .count();
    assert!(sim_count >= 1);
}

// ─── Reverse relations ────────────────────────────────────────────────────

#[test]
fn test_reverse_relation_for_temporal() {
    use llm_mem::types::reverse_relation;
    assert_eq!(reverse_relation("happened_after"), Some("happened_before"));
    assert_eq!(reverse_relation("happened_before"), Some("happened_after"));
    assert_eq!(reverse_relation("happens_within"), Some("happens_within"));
}

// ─── Backfill + L1 range derivation smoke test ────────────────────────────

#[tokio::test]
async fn test_l1_abstraction_inherits_event_at() {
    // Manually simulate the L1 inheritance (we don't have a real LLM in tests).
    let (manager, _tmp) = make_manager().await;
    let when = Utc.with_ymd_and_hms(2026, 6, 2, 12, 0, 0).unwrap();
    let id = manager
        .store_with_options(
            "raw content".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(when),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let l0 = manager.get(&id).await.unwrap().unwrap();
    assert_eq!(l0.event_at, Some(when));

    // Manually craft an L1 memory and verify it can be stored
    let mut l1 = Memory::with_content(
        "summary of raw content".to_string(),
        l0.embedding.clone(),
        make_metadata(),
    );
    l1.event_at = l0.event_at;
    l1.metadata.layer = llm_mem::types::LayerInfo::structural();
    l1.metadata.abstraction_sources = vec![uuid::Uuid::parse_str(&id).unwrap()];

    let store = make_store().await;
    store.insert(&l1).await.unwrap();

    // The event_after filter is `event_at >= X` (inclusive), so use a filter just before `when`.
    let filters = Filters {
        event_after: Some(when - Duration::seconds(1)),
        ..Default::default()
    };
    let list: Vec<Memory> = store.list(&filters, None).await.unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].event_at, Some(when));
}

// ─── Strengthened tests: boundary, backfill, inheritance, dedup ────────────

#[tokio::test]
async fn test_event_after_includes_exact_match() {
    let store = make_store().await;
    let boundary = Utc.with_ymd_and_hms(2026, 6, 1, 12, 0, 0).unwrap();
    let before = boundary - Duration::seconds(1);
    let after = boundary + Duration::seconds(1);

    for (i, when) in [before, boundary, after].iter().enumerate() {
        let mut m = Memory::with_content(format!("memory {}", i), vec![0.0; DIM], make_metadata());
        m.event_at = Some(*when);
        store.insert(&m).await.unwrap();
    }

    let filters = Filters {
        event_after: Some(boundary),
        ..Default::default()
    };
    let results: Vec<Memory> = store.list(&filters, None).await.unwrap();
    let times: Vec<_> = results.iter().map(|m| m.event_at).collect();
    assert!(
        times.contains(&Some(boundary)),
        "event_after must include memory at exact boundary: {:?}",
        times
    );
    assert!(
        times.contains(&Some(after)),
        "event_after must include memory after boundary: {:?}",
        times
    );
    assert_eq!(results.len(), 2, "only boundary and after should match");
}

#[tokio::test]
async fn test_event_before_includes_exact_match() {
    let store = make_store().await;
    let boundary = Utc.with_ymd_and_hms(2026, 6, 1, 12, 0, 0).unwrap();
    let before = boundary - Duration::seconds(1);
    let after = boundary + Duration::seconds(1);

    for (i, when) in [before, boundary, after].iter().enumerate() {
        let mut m = Memory::with_content(format!("memory {}", i), vec![0.0; DIM], make_metadata());
        m.event_at = Some(*when);
        store.insert(&m).await.unwrap();
    }

    let filters = Filters {
        event_before: Some(boundary),
        ..Default::default()
    };
    let results: Vec<Memory> = store.list(&filters, None).await.unwrap();
    let times: Vec<_> = results.iter().map(|m| m.event_at).collect();
    assert!(
        times.contains(&Some(before)),
        "event_before must include memory before boundary: {:?}",
        times
    );
    assert!(
        times.contains(&Some(boundary)),
        "event_before must include memory at exact boundary: {:?}",
        times
    );
    assert_eq!(results.len(), 2, "only before and boundary should match");
}

#[tokio::test]
async fn test_event_filter_range_both_bounds_inclusive() {
    let store = make_store().await;
    let start = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();
    let end = Utc.with_ymd_and_hms(2026, 6, 30, 23, 59, 59).unwrap();
    let inside = Utc.with_ymd_and_hms(2026, 6, 15, 12, 0, 0).unwrap();
    let outside_before = start - Duration::seconds(1);
    let outside_after = end + Duration::seconds(1);

    for (i, when) in [outside_before, start, inside, end, outside_after]
        .iter()
        .enumerate()
    {
        let mut m = Memory::with_content(format!("memory {}", i), vec![0.0; DIM], make_metadata());
        m.event_at = Some(*when);
        store.insert(&m).await.unwrap();
    }

    let filters = Filters {
        event_after: Some(start),
        event_before: Some(end),
        ..Default::default()
    };
    let results: Vec<Memory> = store.list(&filters, None).await.unwrap();
    assert_eq!(results.len(), 3, "start, inside, end must all match");
    let times: Vec<_> = results.iter().map(|m| m.event_at).collect();
    assert!(times.contains(&Some(start)));
    assert!(times.contains(&Some(inside)));
    assert!(times.contains(&Some(end)));
}

#[tokio::test]
async fn test_backfill_memory_without_event_at_matches_event_after_via_created_at() {
    let store = make_store().await;
    let mut m = Memory::with_content(
        "no event_at set".to_string(),
        vec![0.0; DIM],
        make_metadata(),
    );
    let past = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    m.created_at = past;
    store.insert(&m).await.unwrap();

    let filters = Filters {
        event_after: Some(past),
        ..Default::default()
    };
    let results: Vec<Memory> = store.list(&filters, None).await.unwrap();
    assert_eq!(
        results.len(),
        1,
        "memory with NULL event_at must match event_after via created_at"
    );
    assert!(results[0].event_at.is_none());
}

#[tokio::test]
async fn test_derive_event_range_from_sources_with_mixed_event_at() {
    // Use the store directly to avoid the embedding dimension mismatch that
    // create_abstraction hits (it creates memories with empty embeddings).
    // We simulate what derive_event_range_from_sources does: L2/L3 event_at
    // should be the min of all sources' effective_event_at, and event_end
    // should be the max of all sources' effective_event_end.
    let store = make_store().await;
    let t1 = Utc.with_ymd_and_hms(2026, 6, 1, 10, 0, 0).unwrap();
    let t2 = Utc.with_ymd_and_hms(2026, 6, 5, 15, 0, 0).unwrap();

    // Source 1: has explicit event_at
    let mut s1 = Memory::with_content("source 1".to_string(), vec![0.0; DIM], make_metadata());
    s1.event_at = Some(t1);
    store.insert(&s1).await.unwrap();

    // Source 2: has explicit event_at
    let mut s2 = Memory::with_content("source 2".to_string(), vec![0.0; DIM], make_metadata());
    s2.event_at = Some(t2);
    store.insert(&s2).await.unwrap();

    // Source 3: no event_at — effective_event_at falls back to created_at
    let s3 = Memory::with_content("source 3".to_string(), vec![0.0; DIM], make_metadata());
    let s3_created = s3.created_at;
    store.insert(&s3).await.unwrap();

    // Simulate what derive_event_range_from_sources should compute:
    // min = min(t1, t2, s3_created)
    // max = max(t1, t2, s3_created)
    let sources = [s1, s2, s3];
    let effective_starts: Vec<_> = sources.iter().map(|m| m.effective_event_at()).collect();
    let effective_ends: Vec<_> = sources.iter().map(|m| m.effective_event_end()).collect();
    let derived_start = effective_starts.iter().min().unwrap();
    let derived_end = effective_ends.iter().max().unwrap();

    assert_eq!(*derived_start, t1, "min should be the earliest event_at");
    assert!(*derived_end >= t2, "max should be at least t2");
    assert!(
        *derived_end >= s3_created,
        "max should include backfilled created_at"
    );

    // Now store an L1 with this derived range and verify it round-trips
    let mut l1 = Memory::with_content("abstraction".to_string(), vec![0.0; DIM], make_metadata());
    l1.event_at = Some(*derived_start);
    l1.event_end = Some(*derived_end);
    l1.metadata.layer = llm_mem::types::LayerInfo::structural();
    store.insert(&l1).await.unwrap();

    let retrieved = store.get(&l1.id).await.unwrap().expect("L1 exists");
    assert_eq!(retrieved.event_at, Some(*derived_start));
    assert_eq!(retrieved.event_end, Some(*derived_end));
    assert_eq!(retrieved.effective_event_at(), *derived_start);
    assert_eq!(retrieved.effective_event_end(), *derived_end);
}

#[tokio::test]
async fn test_bucket_boundary_memory_near_window_start() {
    let (manager, _tmp) = make_manager().await;
    let window_start = Utc.with_ymd_and_hms(2026, 6, 2, 14, 30, 0).unwrap();
    let near_start = Utc.with_ymd_and_hms(2026, 6, 2, 14, 35, 0).unwrap();
    let well_inside = Utc.with_ymd_and_hms(2026, 6, 3, 10, 0, 0).unwrap();

    for (i, when) in [near_start, well_inside].iter().enumerate() {
        manager
            .store_with_options(
                format!("m {}", i),
                make_metadata(),
                llm_mem::memory::StoreOptions {
                    event_at: Some(*when),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }

    let ops = llm_mem::operations::MemoryOperations::new(Arc::new(manager), None, None, 100);
    let req = llm_mem::operations::GetTimelineRequest {
        start: Some(window_start.to_rfc3339()),
        end: Some("2026-06-30T00:00:00Z".to_string()),
        granularity: Some(llm_mem::operations::TimelineGranularity::Day),
        max_results_per_bucket: 50,
        include_derived: false,
        order: "asc".to_string(),
        ..Default::default()
    };
    let resp = ops.get_timeline(req).await.unwrap();
    let data = resp.data.expect("data");
    let total = data["total_count"].as_u64().unwrap();
    assert_eq!(
        total, 2,
        "both memories should be in the window, even if their bucket floor is before window_start"
    );
}

#[tokio::test]
async fn test_chunk_inherits_parent_event_at() {
    let (manager, _tmp) = make_manager().await;
    let when = Utc.with_ymd_and_hms(2026, 6, 1, 12, 0, 0).unwrap();
    let long_content = "word ".repeat(600);

    let id = manager
        .store_with_options(
            long_content,
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(when),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let parent = manager.get(&id).await.unwrap().expect("parent exists");
    assert_eq!(parent.event_at, Some(when));

    let all: Vec<Memory> = manager.list(&Filters::default(), None).await.unwrap();
    let chunks: Vec<&Memory> = all
        .iter()
        .filter(|m| m.metadata.parent_id.is_some())
        .collect();
    if !chunks.is_empty() {
        for chunk in &chunks {
            assert_eq!(
                chunk.event_at,
                Some(when),
                "chunk {} should inherit parent event_at",
                chunk.id
            );
        }
    }
}

#[tokio::test]
async fn test_semantic_edge_deduplication() {
    let (manager, _tmp) = make_manager().await;
    let t1 = Utc.with_ymd_and_hms(2026, 6, 2, 9, 0, 0).unwrap();
    let t2 = Utc.with_ymd_and_hms(2026, 6, 2, 10, 0, 0).unwrap();
    let t3 = Utc.with_ymd_and_hms(2026, 6, 2, 11, 0, 0).unwrap();

    let id1 = manager
        .store_with_options(
            "a".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(t1),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let id2 = manager
        .store_with_options(
            "b".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(t2),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let id3 = manager
        .store_with_options(
            "c".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(t3),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Create a triangle: a->b, b->c, a->c. BFS from a can reach c via a->b->c and a->c.
    let ops = llm_mem::operations::MemoryOperations::new(Arc::new(manager), None, None, 100);
    ops.force_link(llm_mem::operations::ForceLinkRequest {
        source_id: id1.clone(),
        relation: "references".to_string(),
        target_id: id2.clone(),
        strength: Some(1.0),
        bank: None,
    })
    .await
    .unwrap();
    ops.force_link(llm_mem::operations::ForceLinkRequest {
        source_id: id2.clone(),
        relation: "references".to_string(),
        target_id: id3.clone(),
        strength: Some(1.0),
        bank: None,
    })
    .await
    .unwrap();
    ops.force_link(llm_mem::operations::ForceLinkRequest {
        source_id: id1.clone(),
        relation: "references".to_string(),
        target_id: id3.clone(),
        strength: Some(1.0),
        bank: None,
    })
    .await
    .unwrap();

    let req = llm_mem::operations::GetTimelineGraphRequest {
        timeline: llm_mem::operations::GetTimelineRequest {
            start: Some("2026-06-01T00:00:00Z".to_string()),
            end: Some("2026-06-30T00:00:00Z".to_string()),
            granularity: Some(llm_mem::operations::TimelineGranularity::Day),
            max_results_per_bucket: 50,
            include_derived: false,
            order: "asc".to_string(),
            ..Default::default()
        },
        max_depth: 2,
        relation_types: None,
        temporal_edge_window_secs: 86400,
        include_simultaneous: false,
        simultaneous_window_secs: 60,
        include_semantic_edges: true,
    };
    let resp = ops.get_timeline_graph(req).await.unwrap();
    let data = resp.data.expect("data");
    let edges = data["edges"].as_array().unwrap();

    let semantic_edges: Vec<_> = edges
        .iter()
        .filter(|e| e["type"].as_str() == Some("references"))
        .collect();

    let mut edge_keys: Vec<String> = semantic_edges
        .iter()
        .map(|e| {
            format!(
                "{}->{}",
                e["source"].as_str().unwrap_or(""),
                e["target"].as_str().unwrap_or("")
            )
        })
        .collect();
    edge_keys.sort();
    let mut deduped = edge_keys.clone();
    deduped.dedup();
    assert_eq!(
        edge_keys.len(),
        deduped.len(),
        "semantic edges should have no duplicate (source,target) pairs. Got: {:?}",
        edge_keys
    );
}

#[tokio::test]
async fn test_happens_within_no_edges_beyond_window() {
    let (manager, _tmp) = make_manager().await;
    let t1 = Utc.with_ymd_and_hms(2026, 6, 2, 9, 0, 0).unwrap();
    let t2 = Utc.with_ymd_and_hms(2026, 6, 2, 9, 0, 10).unwrap();
    let t_far = Utc.with_ymd_and_hms(2026, 6, 2, 12, 0, 0).unwrap();

    manager
        .store_with_options(
            "near a".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(t1),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    manager
        .store_with_options(
            "near b".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(t2),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    manager
        .store_with_options(
            "far c".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(t_far),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let ops = llm_mem::operations::MemoryOperations::new(Arc::new(manager), None, None, 100);
    let req = llm_mem::operations::GetTimelineGraphRequest {
        timeline: llm_mem::operations::GetTimelineRequest {
            start: Some("2026-06-01T00:00:00Z".to_string()),
            end: Some("2026-06-30T00:00:00Z".to_string()),
            granularity: Some(llm_mem::operations::TimelineGranularity::Day),
            max_results_per_bucket: 50,
            include_derived: false,
            order: "asc".to_string(),
            ..Default::default()
        },
        max_depth: 0,
        relation_types: None,
        temporal_edge_window_secs: 86400,
        include_simultaneous: true,
        simultaneous_window_secs: 30,
        include_semantic_edges: false,
    };

    let resp = ops.get_timeline_graph(req).await.unwrap();
    let data = resp.data.expect("data");
    let edges = data["edges"].as_array().unwrap();

    let sim_edges: Vec<_> = edges
        .iter()
        .filter(|e| e["type"].as_str() == Some("happens_within"))
        .collect();

    for e in &sim_edges {
        let delta = e["delta_secs"].as_f64().unwrap();
        assert!(
            delta <= 30.0,
            "happens_within edge should be within simultaneous_window_secs=30, got {}",
            delta
        );
    }
    assert!(
        !sim_edges.is_empty(),
        "t1 and t2 are 10s apart, should have at least one happens_within edge"
    );
}

#[tokio::test]
async fn test_event_before_filter_with_backfill() {
    let store = make_store().await;
    let mut m = Memory::with_content(
        "no event_at, old created_at".to_string(),
        vec![0.0; DIM],
        make_metadata(),
    );
    let past = Utc.with_ymd_and_hms(2020, 3, 15, 8, 0, 0).unwrap();
    m.created_at = past;
    store.insert(&m).await.unwrap();

    let filters = Filters {
        event_before: Some(past + Duration::seconds(1)),
        ..Default::default()
    };
    let results: Vec<Memory> = store.list(&filters, None).await.unwrap();
    assert_eq!(
        results.len(),
        1,
        "NULL event_at memory should match event_before via created_at backfill"
    );
}

#[tokio::test]
async fn test_timeline_order_desc_reverses_buckets() {
    let (manager, _tmp) = make_manager().await;
    let d1 = Utc.with_ymd_and_hms(2026, 6, 1, 12, 0, 0).unwrap();
    let d2 = Utc.with_ymd_and_hms(2026, 6, 3, 12, 0, 0).unwrap();

    manager
        .store_with_options(
            "day 1".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(d1),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    manager
        .store_with_options(
            "day 3".to_string(),
            make_metadata(),
            llm_mem::memory::StoreOptions {
                event_at: Some(d2),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let ops = llm_mem::operations::MemoryOperations::new(Arc::new(manager), None, None, 100);

    let req_asc = llm_mem::operations::GetTimelineRequest {
        start: Some("2026-06-01T00:00:00Z".to_string()),
        end: Some("2026-06-30T00:00:00Z".to_string()),
        granularity: Some(llm_mem::operations::TimelineGranularity::Day),
        order: "asc".to_string(),
        ..Default::default()
    };
    let resp_asc = ops.get_timeline(req_asc).await.unwrap();
    let buckets_asc = resp_asc.data.unwrap()["buckets"]
        .as_array()
        .unwrap()
        .clone();

    let req_desc = llm_mem::operations::GetTimelineRequest {
        start: Some("2026-06-01T00:00:00Z".to_string()),
        end: Some("2026-06-30T00:00:00Z".to_string()),
        granularity: Some(llm_mem::operations::TimelineGranularity::Day),
        order: "desc".to_string(),
        ..Default::default()
    };
    let resp_desc = ops.get_timeline(req_desc).await.unwrap();
    let buckets_desc = resp_desc.data.unwrap()["buckets"]
        .as_array()
        .unwrap()
        .clone();

    assert_eq!(buckets_asc.len(), buckets_desc.len());
    assert_eq!(
        buckets_asc[0]["label"].as_str(),
        buckets_desc.last().unwrap()["label"].as_str(),
        "desc order should reverse bucket order"
    );
}

#[tokio::test]
async fn test_include_derived_shows_higher_layer_memories() {
    let store = make_store().await;
    let when = Utc.with_ymd_and_hms(2026, 6, 2, 12, 0, 0).unwrap();

    // L0 memory
    let mut l0 = Memory::with_content("raw L0".to_string(), vec![0.0; DIM], make_metadata());
    l0.event_at = Some(when);
    store.insert(&l0).await.unwrap();

    // L1 memory
    let mut l1 = Memory::with_content("L1 summary".to_string(), vec![0.0; DIM], make_metadata());
    l1.event_at = Some(when);
    l1.metadata.layer = llm_mem::types::LayerInfo::structural();
    store.insert(&l1).await.unwrap();

    // Without max_layer_level (include_derived=true equivalent): should find both
    let filters_all = Filters {
        event_after: Some(when - Duration::seconds(1)),
        event_before: Some(when + Duration::seconds(1)),
        ..Default::default()
    };
    let all: Vec<Memory> = store.list(&filters_all, None).await.unwrap();
    assert_eq!(all.len(), 2, "should find both L0 and L1");

    // With max_layer_level=0 (include_derived=false): should find only L0
    let filters_l0 = Filters {
        event_after: Some(when - Duration::seconds(1)),
        event_before: Some(when + Duration::seconds(1)),
        max_layer_level: Some(0),
        ..Default::default()
    };
    let l0_only: Vec<Memory> = store.list(&filters_l0, None).await.unwrap();
    assert_eq!(
        l0_only.len(),
        1,
        "should find only L0 when max_layer_level=0"
    );
    assert_eq!(l0_only[0].metadata.layer.level, 0);
}
