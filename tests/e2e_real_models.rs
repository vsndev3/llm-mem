//! End-to-end test with real LLM + embedding model
//!
//! Guarded behind the `e2e` feature flag so CI doesn't require model downloads.
//!
//! Run with:
//! ```text
//! cargo test --test e2e_real_models --features e2e
//! ```
//!
//! Requires either:
//! - A local LLM model in `llm-mem-data/models/` (default: `Qwen3.5-2B-UD-Q6_K_XL.gguf`)
//! - Or OpenAI API key via `LLM_MEM_LLM_API_KEY` / `OPENAI_API_KEY` env var
//!
//! NOTE: These tests are currently skeletons. Real LLM client wiring is pending.
//!       See PLAN.md §4.2 and `llm_mem::llm::create_llm_client` for the intended
//!       construction path.

#[cfg(feature = "e2e")]
mod e2e_tests {
    use llm_mem::{
        Config,
        memory::MemoryManager,
        VectorStore,
        lance_store::{LanceDBStore, LanceDBConfig},
        operations::{MemoryOperations, requests::{StoreRequest, QueryRequest, RelationInput, GraphTraversalInput}},
    };
    use tempfile::TempDir;
    use std::sync::Arc;

    /// Verify that a real config loads and validates correctly.
    ///
    /// This is a smoke test that the configuration pipeline works end-to-end
    /// without needing to construct the full memory system.
    #[tokio::test]
    async fn test_config_loads_and_validates() {
        let config = Config::load("config.toml").or_else(|_| {
            let mut cfg = Config::default();
            cfg.apply_env_overrides();
            cfg.validate().map(|_| cfg)
        });

        match config {
            Ok(cfg) => {
                println!("Config loaded successfully:");
                println!("  LLM provider: {:?}", cfg.llm.provider);
                println!("  Embedding provider: {:?}", cfg.embedding.provider);
                println!("  Backend: {:?}", cfg.effective_backend());
            }
            Err(e) => {
                eprintln!("Skipping e2e test: unable to load config: {}", e);
                eprintln!("Set LLM_MEM_LLM_API_KEY or place a config.toml to run this test.");
            }
        }
    }

    /// Full lifecycle: store → search → pyramid search → verify LLM paths.
    #[tokio::test]
    async fn test_full_lifecycle_with_real_models() {
        let config = Config::load("config.toml").or_else(|_| {
            let mut cfg = Config::default();
            cfg.apply_env_overrides();
            cfg.validate().map(|_| cfg)
        });

        let config = match config {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Skipping e2e test: unable to load config: {}", e);
                return;
            }
        };

        // Construct real LLM client and run the full lifecycle.
        let client = match llm_mem::llm::create_llm_client(&config).await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Skipping test: cannot create LLM client: {}", e);
                return;
            }
        };

        if let Err(e) = client.health_check().await {
            eprintln!("Skipping test: client health check failed: {}", e);
            return;
        }

        let temp_dir = TempDir::new().unwrap();
        let store_cfg = LanceDBConfig {
            table_name: "e2e_lifecycle_test".into(),
            database_path: temp_dir.path().to_path_buf(),
            embedding_dimension: config.vector_store.embedding_dimension(),
        };

        let store: Box<dyn VectorStore> = Box::new(
            LanceDBStore::new(store_cfg)
                .await
                .unwrap(),
        );

        let manager = Arc::new(MemoryManager::new(
            store,
            dyn_clone::clone_box(client.as_ref()),
            config.memory.clone(),
        ));

        let ops = MemoryOperations::new(
            manager.clone(),
            Some("test_user".to_string()),
            Some("test_agent".to_string()),
            10,
        );

        // 1. Store
        let store_res = ops.store_memory(StoreRequest {
            content: "Einstein published his theory of general relativity in 1915.".to_string(),
            memory_type: "factual".to_string(),
            user_id: None,
            agent_id: None,
            topics: Some(vec!["physics".to_string(), "relativity".to_string()]),
            context: None,
            relations: None,
            metadata: None,
            bank: None,
        }).await.unwrap();

        assert!(store_res.success);
        let mem_id = store_res.data.as_ref().unwrap()["memory_id"].as_str().unwrap().to_string();

        // 2. Get and verify
        let get_res = ops.get_memory(llm_mem::operations::requests::GetRequest {
            memory_id: mem_id.clone(),
            bank: None,
        }).await.unwrap();
        assert!(get_res.success);
        let retrieved_mem = get_res.data.as_ref().unwrap()["memory"].clone();
        assert_eq!(retrieved_mem["content"].as_str(), Some("Einstein published his theory of general relativity in 1915."));

        // 3. Update
        let update_res = ops.update_memory(llm_mem::operations::requests::UpdateRequest {
            memory_id: mem_id.clone(),
            content: Some("Einstein published his theory of general relativity in 1915, explaining gravity as spacetime curvature.".to_string()),
            relations: None,
            bank: None,
        }).await.unwrap();
        assert!(update_res.success);

        // 4. Get and verify update
        let get_res2 = ops.get_memory(llm_mem::operations::requests::GetRequest {
            memory_id: mem_id.clone(),
            bank: None,
        }).await.unwrap();
        assert!(get_res2.success);
        assert_eq!(
            get_res2.data.as_ref().unwrap()["memory"]["content"].as_str(),
            Some("Einstein published his theory of general relativity in 1915, explaining gravity as spacetime curvature.")
        );

        // 5. Query
        let query_res = ops.query_memory(QueryRequest {
            query: "When did Einstein publish general relativity?".to_string(),
            bank: None,
            user_id: Some("test_user".to_string()),
            agent_id: Some("test_agent".to_string()),
            memory_type: None,
            limit: 5,
            k: None,
            min_salience: None,
            topics: None,
            context: None,
            graph_traversal: None,
            keyword_only: false,
            keyword_split_ratio: 0.0,
            created_before: None,
            created_after: None,
            pyramid_config: None,
            similarity_threshold: Some(0.1),
        }).await.unwrap();
        assert!(query_res.success);
        let memories = query_res.data.as_ref().unwrap()["memories"].as_array().unwrap();
        assert!(memories.iter().any(|m| m["id"].as_str() == Some(&mem_id)));

        println!("test_full_lifecycle_with_real_models: passed successfully!");
    }

    /// Verify that graph refinement discovers related memories with real embeddings.
    #[tokio::test]
    async fn test_graph_refinement_with_real_embeddings() {
        let config = Config::load("config.toml")
            .ok()
            .or_else(|| {
                let mut cfg = Config::default();
                cfg.apply_env_overrides();
                cfg.validate().ok().map(|_| cfg)
            });

        let config = match config {
            Some(c) => c,
            None => {
                eprintln!("Skipping e2e graph refinement test: no config available");
                return;
            }
        };

        let client = match llm_mem::llm::create_llm_client(&config).await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Skipping test: cannot create LLM client: {}", e);
                return;
            }
        };

        if let Err(e) = client.health_check().await {
            eprintln!("Skipping test: client health check failed: {}", e);
            return;
        }

        let temp_dir = TempDir::new().unwrap();
        let store_cfg = LanceDBConfig {
            table_name: "e2e_graph_refinement_test".into(),
            database_path: temp_dir.path().to_path_buf(),
            embedding_dimension: config.vector_store.embedding_dimension(),
        };

        let store: Box<dyn VectorStore> = Box::new(
            LanceDBStore::new(store_cfg)
                .await
                .unwrap(),
        );

        let manager = Arc::new(MemoryManager::new(
            store,
            dyn_clone::clone_box(client.as_ref()),
            config.memory.clone(),
        ));

        let ops = MemoryOperations::new(
            manager.clone(),
            Some("test_user".to_string()),
            Some("test_agent".to_string()),
            10,
        );

        // Store target memory B (semantically orthogonal to the query, so only discoverable via graph traversal)
        let result_b = ops.store_memory(StoreRequest {
            content: "The Mediterranean diet emphasizes olive oil, fish, and fresh vegetables for cardiovascular health.".to_string(),
            memory_type: "factual".to_string(),
            user_id: None,
            agent_id: None,
            topics: Some(vec!["quantum".to_string()]),
            context: None,
            relations: None,
            metadata: None,
            bank: None,
        }).await.unwrap();

        let mem_b_id = result_b.data.as_ref().unwrap()["memory_id"].as_str().unwrap().to_string();

        // Store entry memory A (which matches raw query keywords and links to B)
        let result_a = ops.store_memory(StoreRequest {
            content: "Einstein famously called quantum entanglement spooky action at a distance.".to_string(),
            memory_type: "factual".to_string(),
            user_id: None,
            agent_id: None,
            topics: Some(vec!["quantum".to_string()]),
            context: None,
            relations: Some(vec![RelationInput {
                relation: "explains".to_string(),
                target: mem_b_id.clone(),
            }]),
            metadata: None,
            bank: None,
        }).await.unwrap();

        let mem_a_id = result_a.data.as_ref().unwrap()["memory_id"].as_str().unwrap().to_string();

        // Query: "quantum entanglement spooky action"
        // Under pyramid search, A will be retrieved as a candidate.
        // Since B is linked to A via a relation, the lightweight graph refinement phase in `search_pyramid`
        // should traverse A's relations and pull B in as a `graph_discovered` candidate.
        let query = QueryRequest {
            query: "quantum entanglement spooky action".to_string(),
            bank: None,
            user_id: Some("test_user".to_string()),
            agent_id: Some("test_agent".to_string()),
            memory_type: None,
            limit: 10,
            k: None,
            min_salience: None,
            topics: None,
            context: None,
            graph_traversal: None, // Use default pyramid search, which has graph refinement enabled
            keyword_only: false,
            keyword_split_ratio: 0.0,
            created_before: None,
            created_after: None,
            pyramid_config: Some(llm_mem::search::PyramidConfig {
                mode: llm_mem::search::PyramidAllocationMode::Balanced,
                per_layer_multiplier: 2.0,
                layer_threshold_relaxation: 0.1,
                layer_weights: std::collections::HashMap::new(),
            }),
            similarity_threshold: Some(0.4),
        };

        let response = ops.query_memory(query).await.unwrap();
        assert!(response.success);

        let memories = response.data.as_ref().unwrap()["memories"].as_array().unwrap();

        // Assert entry memory A is found
        let found_a = memories.iter().any(|m| m["id"].as_str() == Some(&mem_a_id));
        assert!(found_a, "Entry memory A must be retrieved");

        // Assert target memory B is found (discovered via graph refinement)
        let pos_b = memories.iter().position(|m| m["id"].as_str() == Some(&mem_b_id));
        assert!(pos_b.is_some(), "Target memory B should be found via graph refinement");

        let mem_b_result = &memories[pos_b.unwrap()];
        let search_phase = mem_b_result["search_phase"].as_str().unwrap();
        println!("Memory B was retrieved via search phase: {}", search_phase);
        assert!(search_phase == "graph_discovered", "Memory B should be found via graph_discovered phase, got: {}", search_phase);
    }

    /// Compare direct embedding search against graph traversal search using real models.
    #[tokio::test]
    async fn test_graph_traversal_versus_direct_search() {
        let config = Config::load("config.toml")
            .ok()
            .or_else(|| {
                let mut cfg = Config::default();
                cfg.apply_env_overrides();
                cfg.validate().ok().map(|_| cfg)
            });

        let config = match config {
            Some(c) => c,
            None => {
                eprintln!("Skipping e2e graph traversal comparison test: no config available");
                return;
            }
        };

        let client = match llm_mem::llm::create_llm_client(&config).await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Skipping test: cannot create LLM client: {}", e);
                return;
            }
        };

        if let Err(e) = client.health_check().await {
            eprintln!("Skipping test: client health check failed: {}", e);
            return;
        }

        let temp_dir = TempDir::new().unwrap();
        let store_cfg = LanceDBConfig {
            table_name: "e2e_graph_test".into(),
            database_path: temp_dir.path().to_path_buf(),
            embedding_dimension: config.vector_store.embedding_dimension(),
        };

        let store: Box<dyn VectorStore> = Box::new(
            LanceDBStore::new(store_cfg)
                .await
                .unwrap(),
        );

        let manager = Arc::new(MemoryManager::new(
            store,
            dyn_clone::clone_box(client.as_ref()),
            config.memory.clone(),
        ));

        let ops = MemoryOperations::new(
            manager.clone(),
            Some("test_user".to_string()),
            Some("test_agent".to_string()),
            10,
        );

        // Store Memory B (Target memory: about a person, semantically distant from "Project Antigravity")
        let result_b = ops.store_memory(StoreRequest {
            content: "Dr. Katherine Johnson was a mathematician who calculated trajectories for NASA missions.".to_string(),
            memory_type: "factual".to_string(),
            user_id: None,
            agent_id: None,
            topics: None,
            context: None,
            relations: None,
            metadata: None,
            bank: None,
        }).await.unwrap();

        let mem_b_id = result_b.data.as_ref().unwrap()["memory_id"].as_str().unwrap().to_string();

        // Store Memory A (Source memory: matches the query, and points to Memory B via relation)
        let result_a = ops.store_memory(StoreRequest {
            content: "The principal architect of Project Antigravity is a key figure in the project.".to_string(),
            memory_type: "factual".to_string(),
            user_id: None,
            agent_id: None,
            topics: None,
            context: None,
            relations: Some(vec![RelationInput {
                relation: "references".to_string(),
                target: mem_b_id.clone(),
            }]),
            metadata: None,
            bank: None,
        }).await.unwrap();

        let mem_a_id = result_a.data.as_ref().unwrap()["memory_id"].as_str().unwrap().to_string();

        // Perform standard query (without graph traversal)
        let query_no_gt = QueryRequest {
            query: "Who is the principal architect of Project Antigravity?".to_string(),
            bank: None,
            user_id: Some("test_user".to_string()),
            agent_id: Some("test_agent".to_string()),
            memory_type: None,
            limit: 10,
            k: None,
            min_salience: None,
            topics: None,
            context: None,
            graph_traversal: Some(GraphTraversalInput {
                enabled: Some(false),
                ..Default::default()
            }),
            keyword_only: false,
            keyword_split_ratio: 0.2,
            created_before: None,
            created_after: None,
            pyramid_config: None,
            similarity_threshold: Some(0.1),
        };
        let response_no_gt = ops.query_memory(query_no_gt).await.unwrap();
        assert!(response_no_gt.success, "Query without graph traversal failed");

        let memories_no_gt = response_no_gt.data.as_ref().unwrap()["memories"].as_array().unwrap();
        let found_b_no_gt = memories_no_gt.iter().any(|m| m["id"].as_str() == Some(&mem_b_id));

        if found_b_no_gt {
            // If Memory B is found in standard search, assert it is ranked below Memory A
            if let Some(pos_a) = memories_no_gt.iter().position(|m| m["id"].as_str() == Some(&mem_a_id)) {
                let pos_b = memories_no_gt.iter().position(|m| m["id"].as_str() == Some(&mem_b_id)).unwrap();
                assert!(pos_a < pos_b, "Without graph traversal, Memory A must rank higher than Memory B");
            }
        }

        // Perform query with graph traversal enabled
        let query_gt = QueryRequest {
            query: "Who is the principal architect of Project Antigravity?".to_string(),
            bank: None,
            user_id: Some("test_user".to_string()),
            agent_id: Some("test_agent".to_string()),
            memory_type: None,
            limit: 10,
            k: None,
            min_salience: None,
            topics: None,
            context: None,
            graph_traversal: Some(GraphTraversalInput {
                enabled: Some(true),
                max_depth: Some(2),
                direction: Some("both".to_string()),
                relation_types: None,
                entry_point_limit: Some(5),
                include_paths: Some(true),
            }),
            keyword_only: false,
            keyword_split_ratio: 0.2,
            created_before: None,
            created_after: None,
            pyramid_config: None,
            similarity_threshold: Some(0.1),
        };
        let response_gt = ops.query_memory(query_gt).await.unwrap();
        assert!(response_gt.success, "Query with graph traversal failed");

        let memories_gt = response_gt.data.as_ref().unwrap()["memories"].as_array().unwrap();
        let pos_b_gt = memories_gt.iter().position(|m| m["id"].as_str() == Some(&mem_b_id));
        assert!(pos_b_gt.is_some(), "Memory B must be found when graph traversal is enabled");

        let mem_b_result = &memories_gt[pos_b_gt.unwrap()];
        assert!(mem_b_result["graph_info"].is_object(), "Memory B should contain graph_info metadata");
        assert_eq!(mem_b_result["graph_info"]["entry_distance"].as_i64(), Some(1), "Memory B entry distance should be 1");
        println!("E2E Graph Traversal Integration test passed successfully!");
    }
}

// When the `e2e` feature is not enabled, provide a compile-time check that the
// test file is valid but produces no actual tests.
#[cfg(not(feature = "e2e"))]
#[test]
fn test_e2e_skipped_without_feature() {
    // This test passes to confirm the file compiles without the `e2e` feature.
    // The actual e2e tests are behind `#[cfg(feature = "e2e")]`.
    println!("Skipping e2e tests — compile with --features e2e to run them");
}
