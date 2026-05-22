//! Observability hooks for memory operations.
//!
//! The `MetricsSink` trait provides a no-op default implementation so the
//! codebase compiles without a metrics backend. Implementations can be
//! swapped in at runtime to capture latency, cache hits, layer distributions,
//! and graph refinement yield.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Duration;

/// Phase labels used by pyramid search instrumentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
/// `Arc<dyn MetricsSink>` to `MemoryManager::new()` (wired through
/// the constructor chain from `MemoryBankManager`).
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

/// Snapshot of accumulated metrics for CLI display.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSnapshot {
    pub query_latency: HashMap<String, LatencyStats>,
    pub cache_hits: HashMap<String, u64>,
    pub cache_misses: HashMap<String, u64>,
    pub layer_distribution: HashMap<i32, u64>,
    pub graph_refinement_discovered: u64,
    pub graph_refinement_base: u64,
    pub allocation_modes: HashMap<String, u64>,
    pub total_result_count: u64,
    pub total_queries: u64,
}

/// Running statistics for a single latency measurement.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LatencyStats {
    pub count: u64,
    pub sum_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

impl LatencyStats {
    fn new() -> Self {
        Self {
            count: 0,
            sum_ms: 0.0,
            min_ms: f64::MAX,
            max_ms: f64::MIN,
        }
    }

    fn record(&mut self, ms: f64) {
        self.count += 1;
        self.sum_ms += ms;
        if ms < self.min_ms {
            self.min_ms = ms;
        }
        if ms > self.max_ms {
            self.max_ms = ms;
        }
    }

    pub fn avg_ms(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_ms / self.count as f64
    }
}

/// Thread-safe metrics collector that accumulates statistics across queries.
pub struct SharedMetrics {
    query_latency: RwLock<HashMap<QueryPhase, LatencyStats>>,
    cache_hits_query_embedding: AtomicU64,
    cache_misses_query_embedding: AtomicU64,
    cache_hits_query_intent: AtomicU64,
    cache_misses_query_intent: AtomicU64,
    cache_hits_layer_manifest: AtomicU64,
    cache_misses_layer_manifest: AtomicU64,
    layer_distribution: RwLock<HashMap<i32, AtomicU64>>,
    graph_refinement_discovered: AtomicU64,
    graph_refinement_base: AtomicU64,
    allocation_modes: RwLock<HashMap<String, AtomicU64>>,
    total_result_count: AtomicU64,
    total_queries: AtomicU64,
}

impl SharedMetrics {
    pub fn new() -> Self {
        Self {
            query_latency: RwLock::new(HashMap::new()),
            cache_hits_query_embedding: AtomicU64::new(0),
            cache_misses_query_embedding: AtomicU64::new(0),
            cache_hits_query_intent: AtomicU64::new(0),
            cache_misses_query_intent: AtomicU64::new(0),
            cache_hits_layer_manifest: AtomicU64::new(0),
            cache_misses_layer_manifest: AtomicU64::new(0),
            layer_distribution: RwLock::new(HashMap::new()),
            graph_refinement_discovered: AtomicU64::new(0),
            graph_refinement_base: AtomicU64::new(0),
            allocation_modes: RwLock::new(HashMap::new()),
            total_result_count: AtomicU64::new(0),
            total_queries: AtomicU64::new(0),
        }
    }

    /// Return a point-in-time snapshot of all accumulated metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let query_latency = self
            .query_latency
            .read()
            .unwrap()
            .iter()
            .map(|(phase, stats)| (format!("{:?}", phase), stats.clone()))
            .collect();

        let cache_hits = HashMap::from([
            ("query_embedding".to_string(), self.cache_hits_query_embedding.load(Ordering::Relaxed)),
            ("query_intent".to_string(), self.cache_hits_query_intent.load(Ordering::Relaxed)),
            ("layer_manifest".to_string(), self.cache_hits_layer_manifest.load(Ordering::Relaxed)),
        ]);

        let cache_misses = HashMap::from([
            ("query_embedding".to_string(), self.cache_misses_query_embedding.load(Ordering::Relaxed)),
            ("query_intent".to_string(), self.cache_misses_query_intent.load(Ordering::Relaxed)),
            ("layer_manifest".to_string(), self.cache_misses_layer_manifest.load(Ordering::Relaxed)),
        ]);

        let layer_distribution = self
            .layer_distribution
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (*k, v.load(Ordering::Relaxed)))
            .collect();

        let allocation_modes = self
            .allocation_modes
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
            .collect();

        let total_queries = self.total_queries.load(Ordering::Relaxed);

        MetricsSnapshot {
            query_latency,
            cache_hits,
            cache_misses,
            layer_distribution,
            graph_refinement_discovered: self
                .graph_refinement_discovered
                .load(Ordering::Relaxed),
            graph_refinement_base: self.graph_refinement_base.load(Ordering::Relaxed),
            allocation_modes,
            total_result_count: self.total_result_count.load(Ordering::Relaxed),
            total_queries,
        }
    }

    /// Reset all accumulated metrics to zero.
    pub fn reset(&self) {
        *self.query_latency.write().unwrap() = HashMap::new();
        self.cache_hits_query_embedding.store(0, Ordering::Relaxed);
        self.cache_misses_query_embedding.store(0, Ordering::Relaxed);
        self.cache_hits_query_intent.store(0, Ordering::Relaxed);
        self.cache_misses_query_intent.store(0, Ordering::Relaxed);
        self.cache_hits_layer_manifest.store(0, Ordering::Relaxed);
        self.cache_misses_layer_manifest.store(0, Ordering::Relaxed);
        *self.layer_distribution.write().unwrap() = HashMap::new();
        self.graph_refinement_discovered.store(0, Ordering::Relaxed);
        self.graph_refinement_base.store(0, Ordering::Relaxed);
        *self.allocation_modes.write().unwrap() = HashMap::new();
        self.total_result_count.store(0, Ordering::Relaxed);
        self.total_queries.store(0, Ordering::Relaxed);
    }
}

impl MetricsSink for SharedMetrics {
    fn record_query_latency(&self, phase: QueryPhase, duration: Duration) {
        let ms = duration.as_secs_f64() * 1000.0;
        let mut map = self.query_latency.write().unwrap();
        let stats = map.entry(phase).or_insert_with(LatencyStats::new);
        stats.record(ms);

        // Count Total phase as a query completion
        if phase == QueryPhase::Total {
            self.total_queries.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_cache_hit(&self, cache: CacheName) {
        match cache {
            CacheName::QueryEmbedding => {
                self.cache_hits_query_embedding.fetch_add(1, Ordering::Relaxed);
            }
            CacheName::QueryIntent => {
                self.cache_hits_query_intent.fetch_add(1, Ordering::Relaxed);
            }
            CacheName::LayerManifest => {
                self.cache_hits_layer_manifest.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn record_cache_miss(&self, cache: CacheName) {
        match cache {
            CacheName::QueryEmbedding => {
                self.cache_misses_query_embedding.fetch_add(1, Ordering::Relaxed);
            }
            CacheName::QueryIntent => {
                self.cache_misses_query_intent.fetch_add(1, Ordering::Relaxed);
            }
            CacheName::LayerManifest => {
                self.cache_misses_layer_manifest.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn record_layer_distribution(&self, counts: &[(i32, usize)]) {
        let mut map = self.layer_distribution.write().unwrap();
        for &(layer, count) in counts {
            let counter = map.entry(layer).or_insert_with(|| AtomicU64::new(0));
            counter.fetch_add(count as u64, Ordering::Relaxed);
        }
    }

    fn record_graph_refinement_yield(&self, discovered: usize, base: usize) {
        self.graph_refinement_discovered
            .fetch_add(discovered as u64, Ordering::Relaxed);
        self.graph_refinement_base.fetch_add(base as u64, Ordering::Relaxed);
    }

    fn record_allocation_mode(&self, mode: &str) {
        let mut map = self.allocation_modes.write().unwrap();
        let counter = map.entry(mode.to_string()).or_insert_with(|| AtomicU64::new(0));
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn record_result_count(&self, count: usize) {
        self.total_result_count
            .fetch_add(count as u64, Ordering::Relaxed);
    }
}

impl Default for SharedMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_latency_stats_recording() {
        let mut stats = LatencyStats::new();
        stats.record(10.0);
        stats.record(20.0);
        stats.record(30.0);

        assert_eq!(stats.count, 3);
        assert!((stats.avg_ms() - 20.0).abs() < 0.01);
        assert!((stats.min_ms - 10.0).abs() < 0.01);
        assert!((stats.max_ms - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_latency_stats_empty() {
        let stats = LatencyStats::new();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.avg_ms(), 0.0);
    }

    #[test]
    fn test_shared_metrics_cache_hits_and_misses() {
        let m = SharedMetrics::new();

        m.record_cache_hit(CacheName::QueryEmbedding);
        m.record_cache_hit(CacheName::QueryEmbedding);
        m.record_cache_miss(CacheName::QueryEmbedding);

        m.record_cache_hit(CacheName::QueryIntent);
        m.record_cache_miss(CacheName::QueryIntent);
        m.record_cache_miss(CacheName::QueryIntent);

        m.record_cache_hit(CacheName::LayerManifest);

        let snap = m.snapshot();
        assert_eq!(snap.cache_hits.get("query_embedding"), Some(&2));
        assert_eq!(snap.cache_misses.get("query_embedding"), Some(&1));
        assert_eq!(snap.cache_hits.get("query_intent"), Some(&1));
        assert_eq!(snap.cache_misses.get("query_intent"), Some(&2));
        assert_eq!(snap.cache_hits.get("layer_manifest"), Some(&1));
        assert_eq!(snap.cache_misses.get("layer_manifest"), Some(&0));
    }

    #[test]
    fn test_shared_metrics_query_latency() {
        let m = SharedMetrics::new();

        m.record_query_latency(QueryPhase::LayerDiscovery, Duration::from_millis(5));
        m.record_query_latency(QueryPhase::LayerDiscovery, Duration::from_millis(15));
        m.record_query_latency(QueryPhase::Total, Duration::from_millis(100));
        m.record_query_latency(QueryPhase::Total, Duration::from_millis(200));

        let snap = m.snapshot();

        let layer_disc = &snap.query_latency["LayerDiscovery"];
        assert_eq!(layer_disc.count, 2);
        assert!((layer_disc.avg_ms() - 10.0).abs() < 0.01);

        let total = &snap.query_latency["Total"];
        assert_eq!(total.count, 2);
        assert_eq!(snap.total_queries, 2);
    }

    #[test]
    fn test_shared_metrics_layer_distribution() {
        let m = SharedMetrics::new();

        m.record_layer_distribution(&[(0, 5), (1, 3), (0, 2)]);
        m.record_layer_distribution(&[(1, 1), (2, 4)]);

        let snap = m.snapshot();
        assert_eq!(snap.layer_distribution.get(&0), Some(&7));
        assert_eq!(snap.layer_distribution.get(&1), Some(&4));
        assert_eq!(snap.layer_distribution.get(&2), Some(&4));
    }

    #[test]
    fn test_shared_metrics_allocation_modes() {
        let m = SharedMetrics::new();

        m.record_allocation_mode("dynamic");
        m.record_allocation_mode("dynamic");
        m.record_allocation_mode("fixed");

        let snap = m.snapshot();
        assert_eq!(snap.allocation_modes.get("dynamic"), Some(&2));
        assert_eq!(snap.allocation_modes.get("fixed"), Some(&1));
    }

    #[test]
    fn test_shared_metrics_graph_refinement() {
        let m = SharedMetrics::new();

        m.record_graph_refinement_yield(5, 20);
        m.record_graph_refinement_yield(3, 15);

        let snap = m.snapshot();
        assert_eq!(snap.graph_refinement_discovered, 8);
        assert_eq!(snap.graph_refinement_base, 35);
    }

    #[test]
    fn test_shared_metrics_result_count() {
        let m = SharedMetrics::new();

        m.record_result_count(10);
        m.record_result_count(5);

        let snap = m.snapshot();
        assert_eq!(snap.total_result_count, 15);
    }

    #[test]
    fn test_shared_metrics_reset() {
        let m = SharedMetrics::new();

        m.record_cache_hit(CacheName::QueryEmbedding);
        m.record_cache_miss(CacheName::QueryIntent);
        m.record_query_latency(QueryPhase::Total, Duration::from_millis(50));
        m.record_result_count(10);
        m.record_allocation_mode("balanced");
        m.record_layer_distribution(&[(0, 1)]);
        m.record_graph_refinement_yield(1, 2);

        let snap = m.snapshot();
        assert_eq!(snap.total_queries, 1);
        assert_eq!(snap.total_result_count, 10);

        m.reset();

        let snap = m.snapshot();
        assert_eq!(snap.total_queries, 0);
        assert_eq!(snap.total_result_count, 0);
        assert!(snap.cache_hits.values().all(|&v| v == 0));
        assert!(snap.cache_misses.values().all(|&v| v == 0));
        assert!(snap.query_latency.is_empty());
        assert!(snap.layer_distribution.is_empty());
        assert!(snap.allocation_modes.is_empty());
        assert_eq!(snap.graph_refinement_discovered, 0);
        assert_eq!(snap.graph_refinement_base, 0);
    }

    #[test]
    fn test_shared_metrics_concurrent_recording() {
        use std::sync::Arc;
        let m = Arc::new(SharedMetrics::new());

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let m = Arc::clone(&m);
                std::thread::spawn(move || {
                    m.record_cache_hit(CacheName::QueryEmbedding);
                    m.record_cache_miss(CacheName::QueryIntent);
                    m.record_query_latency(QueryPhase::Total, Duration::from_millis(i));
                    m.record_result_count(i as usize);
                    m.record_allocation_mode("balanced");
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let snap = m.snapshot();
        assert_eq!(snap.total_queries, 10);
        assert_eq!(snap.cache_hits.get("query_embedding"), Some(&10));
        assert_eq!(snap.cache_misses.get("query_intent"), Some(&10));
        assert_eq!(snap.allocation_modes.get("balanced"), Some(&10));
    }

    #[test]
    fn test_noop_metrics_sink() {
        let noop = NoopMetrics;
        noop.record_query_latency(QueryPhase::Total, Duration::from_secs(1));
        noop.record_cache_hit(CacheName::QueryEmbedding);
        noop.record_cache_miss(CacheName::QueryIntent);
        noop.record_layer_distribution(&[(0, 1)]);
        noop.record_graph_refinement_yield(1, 1);
        noop.record_allocation_mode("test");
        noop.record_result_count(42);
    }
}
