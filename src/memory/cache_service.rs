use std::collections::VecDeque;
use std::sync::Arc;

use crate::error::Result;
use crate::llm::LLMClient;
use crate::memory::metrics::{CacheName, MetricsSink, NoopMetrics};
use crate::search::PyramidAllocationMode;

/// LRU cache for query embeddings and query intent classification results.
///
/// Extracted from MemoryManager to reduce its responsibilities.
/// Uses `VecDeque` for O(1) front-pop eviction and linear scan for lookups.
/// Thread-safe via internal locking.
pub struct CacheService {
    llm_client: Box<dyn LLMClient + Send + Sync>,
    /// LRU cache for LLM-based query intent classification (capacity 64).
    query_intent_cache: tokio::sync::RwLock<VecDeque<(String, PyramidAllocationMode)>>,
    /// LRU cache for query embeddings — avoids redundant LLM calls for repeated queries (capacity 128).
    query_embedding_cache: tokio::sync::RwLock<VecDeque<(String, Vec<f32>)>>,
    /// Optional metrics sink for observability (no-op by default).
    metrics: Arc<dyn MetricsSink>,
}

impl CacheService {
    pub fn new(llm_client: Box<dyn LLMClient + Send + Sync>) -> Self {
        Self {
            llm_client,
            query_intent_cache: tokio::sync::RwLock::new(VecDeque::with_capacity(64)),
            query_embedding_cache: tokio::sync::RwLock::new(VecDeque::with_capacity(128)),
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
    pub async fn cached_embed(&self, text: &str) -> Result<Vec<f32>> {
        {
            let cache = self.query_embedding_cache.read().await;
            if let Some(embedding) = cache.iter().find(|(q, _)| q == text).map(|(_, e)| e.clone()) {
                tracing::debug!("Query embedding cache hit for: {}", text);
                self.metrics.record_cache_hit(CacheName::QueryEmbedding);
                return Ok(embedding);
            }
        }
        self.metrics.record_cache_miss(CacheName::QueryEmbedding);

        let embedding = self.llm_client.embed(text).await?;
        {
            let mut cache = self.query_embedding_cache.write().await;
            if let Some(pos) = cache.iter().position(|(q, _)| q == text) {
                cache.remove(pos);
            }
            if cache.len() >= 128 {
                cache.pop_front();
            }
            cache.push_back((text.to_string(), embedding.clone()));
        }

        Ok(embedding)
    }

    /// Classify query intent for dynamic pyramid allocation, using LRU cache.
    pub async fn classify_query_intent(
        &self,
        query: &str,
        use_llm: bool,
    ) -> PyramidAllocationMode {
        {
            let cache = self.query_intent_cache.read().await;
            if let Some(&mode) = cache.iter().find(|(q, _)| q == query).map(|(_, m)| m) {
                self.metrics.record_cache_hit(CacheName::QueryIntent);
                return mode;
            }
        }
        self.metrics.record_cache_miss(CacheName::QueryIntent);

        let mode = if use_llm {
            match Self::classify_query_with_llm(query, &*self.llm_client).await {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!(error = %e, "LLM query classification failed, falling back to keyword heuristic");
                    Self::keyword_classify(query)
                }
            }
        } else {
            Self::keyword_classify(query)
        };

        {
            let mut cache = self.query_intent_cache.write().await;
            if let Some(pos) = cache.iter().position(|(q, _)| q == query) {
                cache.remove(pos);
            }
            if cache.len() >= 64 {
                cache.pop_front();
            }
            cache.push_back((query.to_string(), mode));
        }

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