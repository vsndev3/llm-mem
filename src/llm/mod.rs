pub mod circuit_breaker;
pub mod client;
pub mod extractor_types;
#[cfg(feature = "local-embed")]
pub mod fastembed_helpers;
#[cfg(feature = "local-llm")]
pub mod llama_cleanup;
pub mod metrics_wrapper;
#[cfg(all(feature = "local-llm", feature = "local-embed"))]
pub mod lazy_client;
#[cfg(all(feature = "local-llm", feature = "local-embed"))]
pub mod local_client;
#[cfg(all(feature = "local-llm", feature = "local-embed"))]
pub mod model_downloader;
pub mod priority;
pub mod strategy_advisor;

pub use circuit_breaker::{backoff_duration, CircuitBreaker, CircuitBreakerConfig, CircuitBreakerLLMClient, CircuitBreakerStats, CircuitState};
pub use client::*;
pub use extractor_types::*;
pub use priority::{LlmPriority, PriorityLLMClient};
pub use strategy_advisor::LLMStrategyAdvisor;

#[cfg(feature = "local-llm")]
pub use llama_cleanup::cleanup_llama_backend;
