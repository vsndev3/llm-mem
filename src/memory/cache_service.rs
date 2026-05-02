use std::collections::VecDeque;
use std::sync::Arc;

use dashmap::DashMap;

use crate::error::Result;
use crate::llm::{LLMClient, LlmPriority, PriorityLLMClient};
use crate::memory::metrics::{CacheName, MetricsSink, NoopMetrics};
use crate::search::PyramidAllocationMode;

/// Concurrent FIFO cache backed by DashMap for O(1) lookups with an
/// ordering queue for capacity-bounded eviction.
///
/// Reads are lock-free concurrent. Writes lock only the order queue.
struct ConcurrentLru<V: Clone> {
    map: DashMap<String, V>,
    order: tokio::sync::Mutex<VecDeque<String>>,
    capacity: usize,
}

impl<V: Clone> ConcurrentLru<V> {
    fn new(capacity: usize) -> Self {
        Self {
            map: DashMap::with_capacity(capacity),
            order: tokio::sync::Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    fn get(&self, key: &str) -> Option<V> {
        self.map.get(key).map(|r| r.clone())
    }

    async fn insert(&self, key: String, value: V) {
        let mut order = self.order.lock().await;
        if let Some(pos) = order.iter().position(|k| k == &key) {
            order.remove(pos);
        }
        if order.len() >= self.capacity {
            if let Some(evicted) = order.pop_front() {
                self.map.remove(&evicted);
            }
        }
        order.push_back(key.clone());
        self.map.insert(key, value);
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.map.len()
    }
}

/// LRU cache for query embeddings and query intent classification results.
///
/// Extracted from MemoryManager to reduce its responsibilities.
/// Backed by DashMap for concurrent O(1) lookups.
pub struct CacheService {
    llm: Arc<PriorityLLMClient>,
    query_intent_cache: ConcurrentLru<PyramidAllocationMode>,
    query_embedding_cache: ConcurrentLru<Vec<f32>>,
    metrics: Arc<dyn MetricsSink>,
}

impl CacheService {
    pub fn new(llm: Arc<PriorityLLMClient>) -> Self {
        Self {
            llm,
            query_intent_cache: ConcurrentLru::new(64),
            query_embedding_cache: ConcurrentLru::new(128),
            metrics: Arc::new(NoopMetrics),
        }
    }

    pub fn set_metrics_sink(&mut self, sink: Arc<dyn MetricsSink>) {
        self.metrics = sink;
    }

    pub fn metrics(&self) -> &Arc<dyn MetricsSink> {
        &self.metrics
    }

    /// Embed a query string with LRU caching to avoid redundant LLM calls.
    pub async fn cached_embed(&self, text: &str, priority: LlmPriority) -> Result<Vec<f32>> {
        if let Some(embedding) = self.query_embedding_cache.get(text) {
            tracing::debug!("Query embedding cache hit for: {}", text);
            self.metrics.record_cache_hit(CacheName::QueryEmbedding);
            return Ok(embedding);
        }
        self.metrics.record_cache_miss(CacheName::QueryEmbedding);

        let _guard = self.llm.acquire(priority).await;
        let embedding = self.llm.inner().embed(text).await?;
        self.query_embedding_cache.insert(text.to_string(), embedding.clone()).await;

        Ok(embedding)
    }

    /// Classify query intent for dynamic pyramid allocation, using LRU cache.
    /// Always uses Interactive priority (called during user queries).
    pub async fn classify_query_intent(
        &self,
        query: &str,
        use_llm: bool,
    ) -> PyramidAllocationMode {
        if let Some(mode) = self.query_intent_cache.get(query) {
            self.metrics.record_cache_hit(CacheName::QueryIntent);
            return mode;
        }
        self.metrics.record_cache_miss(CacheName::QueryIntent);

        let mode = if use_llm {
            match Self::classify_query_with_llm(query, self.llm.inner()).await {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!(error = %e, "LLM query classification failed, falling back to keyword heuristic");
                    Self::keyword_classify(query)
                }
            }
        } else {
            Self::keyword_classify(query)
        };

        self.query_intent_cache.insert(query.to_string(), mode).await;

        mode
    }

    /// LLM-based query intent classification
    async fn classify_query_with_llm(query: &str, llm_client: &dyn LLMClient) -> Result<PyramidAllocationMode> {
        let prompt = format!(
            r#"Classify the intent of this query into one of three categories. Respond with ONLY the category name (no explanation):

Categories:
- TopHeavy: Conceptual, explanatory, or analytical queries (why, how, explain, compare, understand)
- BottomHeavy: Factual, specific, or lookup queries (what is, when, where, who, which, list, count)
- Balanced: General exploration or mixed-intent queries

Query: "{query}"

Category:"#
        );

        let response = llm_client.complete(&prompt).await?;
        let response = response.trim().to_lowercase();

        let mode = if response.contains("top") || response.contains("conceptual") || response.contains("explanatory") || response.contains("analytical") {
            PyramidAllocationMode::TopHeavy
        } else if response.contains("bottom") || response.contains("factual") || response.contains("lookup") || response.contains("specific") {
            PyramidAllocationMode::BottomHeavy
        } else {
            PyramidAllocationMode::Balanced
        };

        Ok(mode)
    }

    /// Fast keyword-based classification fallback
    fn keyword_classify(query: &str) -> PyramidAllocationMode {
        let lower = query.to_lowercase();

        let conceptual_words = [
            "why", "how", "explain", "concept", "theory", "principle",
            "understand", "meaning", "purpose", "relationship", "compare",
            "difference", "similar",
        ];
        let factual_words = [
            "what is", "when", "where", "who", "which", "date", "time",
            "place", "name", "value", "number", "count", "list", "example", "fact",
        ];

        let conceptual_score = conceptual_words.iter().filter(|w| lower.contains(**w)).count();
        let factual_score = factual_words.iter().filter(|w| lower.contains(**w)).count();

        if conceptual_score > factual_score {
            PyramidAllocationMode::TopHeavy
        } else if factual_score > conceptual_score {
            PyramidAllocationMode::BottomHeavy
        } else {
            PyramidAllocationMode::Balanced
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_miss() {
        let cache = ConcurrentLru::<String>::new(4);
        assert_eq!(cache.get("missing"), None);
    }

    #[tokio::test]
    async fn test_insert_and_get() {
        let cache = ConcurrentLru::new(4);
        cache.insert("a".to_string(), "alpha".to_string()).await;
        assert_eq!(cache.get("a"), Some("alpha".to_string()));
    }

    #[tokio::test]
    async fn test_multiple_keys() {
        let cache = ConcurrentLru::new(4);
        cache.insert("a".to_string(), 1).await;
        cache.insert("b".to_string(), 2).await;
        cache.insert("c".to_string(), 3).await;
        assert_eq!(cache.get("a"), Some(1));
        assert_eq!(cache.get("b"), Some(2));
        assert_eq!(cache.get("c"), Some(3));
    }

    #[tokio::test]
    async fn test_duplicate_key_refreshed() {
        let cache = ConcurrentLru::new(3);
        cache.insert("a".to_string(), 1).await;
        cache.insert("b".to_string(), 2).await;
        cache.insert("a".to_string(), 10).await; // re-insert a
        assert_eq!(cache.get("a"), Some(10));
        // a should be at back of queue, b at front
        cache.insert("c".to_string(), 3).await;
        cache.insert("d".to_string(), 4).await;
        // capacity 3 → b should have been evicted (FIFO, b was oldest)
        assert_eq!(cache.get("b"), None);
        assert_eq!(cache.get("a"), Some(10));
        assert_eq!(cache.get("c"), Some(3));
        assert_eq!(cache.get("d"), Some(4));
    }

    #[tokio::test]
    async fn test_capacity_eviction_fifo() {
        let cache = ConcurrentLru::new(2);
        cache.insert("first".to_string(), 1).await;
        cache.insert("second".to_string(), 2).await;
        cache.insert("third".to_string(), 3).await;
        assert_eq!(cache.get("first"), None);
        assert_eq!(cache.get("second"), Some(2));
        assert_eq!(cache.get("third"), Some(3));
    }

    #[tokio::test]
    async fn test_len() {
        let cache = ConcurrentLru::new(10);
        cache.insert("x".to_string(), 1).await;
        cache.insert("y".to_string(), 2).await;
        assert_eq!(cache.len(), 2);
    }

    #[tokio::test]
    async fn test_concurrent_reads() {
        let cache = Arc::new(ConcurrentLru::new(10));
        cache.insert("shared".to_string(), "value".to_string()).await;

        let mut handles = Vec::new();
        for _ in 0..32 {
            let c = Arc::clone(&cache);
            handles.push(tokio::spawn(async move {
                assert_eq!(c.get("shared"), Some("value".to_string()));
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_concurrent_insert_and_read() {
        let cache = Arc::new(ConcurrentLru::new(64));
        let cache_w = Arc::clone(&cache);

        let writer = tokio::spawn(async move {
            for i in 0..100 {
                cache_w.insert(format!("k{}", i), i).await;
            }
        });

        let mut readers = Vec::new();
        for _ in 0..8 {
            let c = Arc::clone(&cache);
            readers.push(tokio::spawn(async move {
                for _ in 0..50 {
                    let _ = c.get("k0");
                    tokio::task::yield_now().await;
                }
            }));
        }

        writer.await.unwrap();
        for r in readers {
            r.await.unwrap();
        }
    }
}