//! Graph traversal search, pyramid assembly, and relation-aware ranking
//!
//! This module provides:
//! - Graph-based search that follows relations between memories
//! - Hierarchical pyramid search that allocates results across abstraction layers
//! - Lightweight graph refinement for post-search discovery

mod graph_engine;
mod pyramid_assembler;

pub use graph_engine::{
    GraphSearchEngine, GraphSearchResult, GraphTraversalError, RelationHop, TraversalConfig,
    TraversalDirection, TraversalStrategy,
};
pub use pyramid_assembler::{
    PyramidAllocationMode, PyramidAssembler, PyramidConfig, PyramidResult,
};

use serde::{Deserialize, Serialize};

/// Result content granularity for query results.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResultGranularity {
    /// Chunk hits resolve to the complete parent memory (default).
    #[default]
    Full,
    /// Chunk hits resolve to a compact excerpt around the matched regions,
    /// under the `excerpt_max_chars` total budget.
    Excerpt,
}

/// Configuration for excerpt-granularity query results.
///
/// Normally, when a chunk (secondary index record) matches a query, the full
/// parent L0 memory is returned. For long sources (e.g. full chat sessions,
/// 10KB+), this inflates the caller's context with mostly-irrelevant text.
/// Excerpt mode assembles a compact view around the matched regions instead.
#[derive(Debug, Clone, Copy)]
pub struct ExcerptConfig {
    /// Characters of surrounding context to include on each side of a matched
    /// chunk region.
    pub window_chars: usize,
    /// Maximum excerpt length per returned memory.
    pub max_per_memory_chars: usize,
    /// Maximum total excerpt content across the whole result set. The final
    /// enforcement happens in the query operation; the per-layer search only
    /// applies the per-memory cap.
    pub max_total_chars: usize,
}

impl Default for ExcerptConfig {
    fn default() -> Self {
        Self {
            window_chars: 750,
            max_per_memory_chars: 3000,
            max_total_chars: 12000,
        }
    }
}
