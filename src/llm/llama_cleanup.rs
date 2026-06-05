//! llama-cpp-only utilities that don't depend on fastembed.
//!
//! Kept in its own module so callers using only `local-llm` (e.g. for the
//! `LocalLLMAPIEmbed` backend with API embeddings) can still hook into
//! llama-cpp lifecycle events without pulling in `local-embed`.

use tracing::debug;

/// Cleanup function to be called on shutdown.
///
/// This allows graceful termination of llama-cpp resources.
/// Note: After calling this, the LLM client should not be used again.
pub fn cleanup_llama_backend() {
    // The LlamaBackend is stored in a static OnceLock and will be dropped
    // when the process exits. This function is a hook for any future cleanup
    // logic that may be needed.
    //
    // Note: llama-cpp-2 doesn't provide an explicit shutdown method,
    // but the backend will be properly cleaned up when the Arc<LlamaBackend>
    // is dropped on process exit.
    debug!("Cleaning up llama backend resources");
}
