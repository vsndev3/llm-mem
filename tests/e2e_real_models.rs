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
    use llm_mem::Config;

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
    ///
    /// TODO: Wire up `llm_mem::llm::create_llm_client(&config).await` once it is
    /// re-exported at the crate root, then construct a `MemoryBankManager` and
    /// exercise the full pipeline.
    #[tokio::test]
    async fn test_full_lifecycle_with_real_models() {
        let config = Config::load("config.toml").or_else(|_| {
            let mut cfg = Config::default();
            cfg.apply_env_overrides();
            cfg.validate().map(|_| cfg)
        });

        let _config = match config {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Skipping e2e test: unable to load config: {}", e);
                return;
            }
        };

        // Pending: construct real LLM client via `llm_mem::llm::create_llm_client`
        // and run the full lifecycle. See PLAN.md §4.2.
        println!("test_full_lifecycle_with_real_models: skeleton — pending real LLM client wiring");
    }

    /// Verify that graph refinement discovers related memories with real embeddings.
    ///
    /// TODO: Same client wiring requirement as above.
    #[tokio::test]
    async fn test_graph_refinement_with_real_embeddings() {
        let config = Config::load("config.toml")
            .ok()
            .or_else(|| {
                let mut cfg = Config::default();
                cfg.apply_env_overrides();
                cfg.validate().ok().map(|_| cfg)
            });

        let _config = match config {
            Some(c) => c,
            None => {
                eprintln!("Skipping e2e graph refinement test: no config available");
                return;
            }
        };

        // Pending: construct real LLM client and exercise graph refinement.
        println!("test_graph_refinement_with_real_embeddings: skeleton — pending real LLM client wiring");
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
