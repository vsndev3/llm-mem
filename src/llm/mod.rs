pub mod circuit_breaker;
pub mod client;
pub mod cost_tracker;
pub mod extractor_types;
#[cfg(feature = "local-embed")]
pub mod fastembed_helpers;
#[cfg(all(feature = "local-llm", feature = "local-embed"))]
pub mod lazy_client;
#[cfg(feature = "local-llm")]
pub mod llama_cleanup;
#[cfg(all(feature = "local-llm", feature = "local-embed"))]
pub mod local_client;
pub mod metrics_wrapper;
#[cfg(all(feature = "local-llm", feature = "local-embed"))]
pub mod model_downloader;
pub mod priority;
pub mod strategy_advisor;

pub use circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerLLMClient, CircuitBreakerStats,
    CircuitState, backoff_duration,
};
pub use client::*;
pub use extractor_types::*;
pub use priority::{LlmPriority, PriorityLLMClient};
pub use strategy_advisor::LLMStrategyAdvisor;

#[cfg(feature = "local-llm")]
pub use llama_cleanup::cleanup_llama_backend;
