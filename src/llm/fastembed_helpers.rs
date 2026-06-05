//! Helpers for the fastembed-based local embedding backend.
//!
//! This module is gated on the `local-embed` feature and is independent of
//! `local-llm`. Anything that only needs local embeddings (e.g. an API-LLM
//! client with local embeddings) depends on this module rather than the
//! full local LLM stack.

use tracing::warn;

/// Map a user-facing embedding model name to a fastembed enum variant.
pub fn parse_fastembed_model(name: &str) -> fastembed::EmbeddingModel {
    match name.to_lowercase().replace(['_', ' '], "-").as_str() {
        "all-minilm-l6-v2" | "allminilml6v2" => fastembed::EmbeddingModel::AllMiniLML6V2,
        "all-minilm-l12-v2" | "allminilml12v2" => fastembed::EmbeddingModel::AllMiniLML12V2,
        "bge-small-en-v1.5" | "bgesmallenv15" => fastembed::EmbeddingModel::BGESmallENV15,
        "bge-base-en-v1.5" | "bgebaseenv15" => fastembed::EmbeddingModel::BGEBaseENV15,
        other => {
            warn!(
                "Unknown embedding model '{}', falling back to all-MiniLM-L6-v2",
                other
            );
            fastembed::EmbeddingModel::AllMiniLML6V2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fastembed_model() {
        assert!(matches!(
            parse_fastembed_model("all-MiniLM-L6-v2"),
            fastembed::EmbeddingModel::AllMiniLML6V2
        ));
        assert!(matches!(
            parse_fastembed_model("bge-small-en-v1.5"),
            fastembed::EmbeddingModel::BGESmallENV15
        ));
        // Unknown falls back
        assert!(matches!(
            parse_fastembed_model("nonexistent-model"),
            fastembed::EmbeddingModel::AllMiniLML6V2
        ));
    }
}
