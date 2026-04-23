//! Graph traversal search, pyramid assembly, and relation-aware ranking
//!
//! This module provides:
//! - Graph-based search that follows relations between memories
//! - Hierarchical pyramid search that allocates results across abstraction layers
//! - Lightweight graph refinement for post-search discovery

mod graph_engine;
mod pyramid_assembler;

pub use graph_engine::{
    GraphSearchEngine, GraphSearchResult, RelationHop, TraversalConfig, TraversalDirection,
    TraversalStrategy,
};
pub use pyramid_assembler::{
    PyramidAllocationMode, PyramidAssembler, PyramidConfig, PyramidResult,
};
