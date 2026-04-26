//! Observability hooks for memory operations.
//!
//! The `MetricsSink` trait provides a no-op default implementation so the
//! codebase compiles without a metrics backend. Implementations can be
//! swapped in at runtime to capture latency, cache hits, layer distributions,
//! and graph refinement yield.

use std::time::Duration;

/// Phase labels used by pyramid search instrumentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryPhase {
    /// Discovery of active layers
    LayerDiscovery,
    /// Per-layer vector search (parallel)
    LayerSearch,
    /// Pyramid assembly (slot allocation, dedup)
    Assembly,
    /// Graph refinement from top results
    GraphRefinement,
    /// LLM-based query intent classification
    IntentClassification,
    /// Query embedding (with cache)
    QueryEmbedding,
    /// End-to-end pyramid search
    Total,
}

/// Named cache used for hit/miss tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheName {
    /// LRU cache for query embeddings
    QueryEmbedding,
    /// LRU cache for LLM query intent classification
    QueryIntent,
    /// In-memory layer manifest cache
    LayerManifest,
}

/// Trait for capturing operational metrics.
///
/// The default implementation is a no-op. Enable metrics by passing an
/// `Arc<dyn MetricsSink>` to `MemoryManager` via `set_metrics_sink()`.
pub trait MetricsSink: Send + Sync {
    /// Record the duration of a query phase.
    fn record_query_latency(&self, _phase: QueryPhase, _duration: Duration) {}

    /// Record a cache hit.
    fn record_cache_hit(&self, _cache: CacheName) {}

    /// Record a cache miss.
    fn record_cache_miss(&self, _cache: CacheName) {}

    /// Record the distribution of results across layers.
    /// `counts` is a vector of `(layer_level, count)` pairs.
    fn record_layer_distribution(&self, _counts: &[(i32, usize)]) {}

    /// Record graph refinement yield.
    /// `discovered` is the number of new memories found via graph traversal.
    /// `base` is the number of memories from base pyramid search.
    fn record_graph_refinement_yield(&self, _discovered: usize, _base: usize) {}

    /// Record the resolved allocation mode for dynamic queries.
    fn record_allocation_mode(&self, _mode: &str) {}

    /// Record total query result count.
    fn record_result_count(&self, _count: usize) {}
}

/// No-op metrics sink that does nothing.
pub struct NoopMetrics;

impl MetricsSink for NoopMetrics {}

// Blanket implementation for all types that don't implement MetricsSink
// ensures the trait is always available even without a backend.
impl Default for NoopMetrics {
    fn default() -> Self {
        Self
    }
}
