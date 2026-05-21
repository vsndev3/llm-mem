use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use crate::{
    config::MemoryConfig,
    error::{MemoryError, Result},
    llm::{LlmPriority, PriorityLLMClient},
    memory::cache_service::CacheService,
    memory::metrics::QueryPhase,
    search::{GraphSearchEngine, PyramidAllocationMode, PyramidAssembler, PyramidConfig, PyramidResult, TraversalConfig},
    types::{Filters, Memory, ScoredMemory},
    vector_store::VectorStore,
};

/// Owns pyramid search, hybrid search, context search, keyword search,
/// and per-layer vector search orchestration.
///
/// Extracted from MemoryManager to reduce its god-object responsibilities.
pub struct SearchService {
    vector_store: Box<dyn VectorStore + Send + Sync>,
    llm: Arc<PriorityLLMClient>,
    config: Arc<MemoryConfig>,
    cache: Arc<CacheService>,
    layer_manifest: tokio::sync::RwLock<HashSet<i32>>,
}

impl SearchService {
    pub fn new(
        vector_store: Box<dyn VectorStore + Send + Sync>,
        llm: Arc<PriorityLLMClient>,
        config: Arc<MemoryConfig>,
        cache: Arc<CacheService>,
    ) -> Self {
        let mut manifest = HashSet::new();
        manifest.insert(0);
        Self {
            vector_store,
            llm,
            config,
            cache,
            layer_manifest: tokio::sync::RwLock::new(manifest),
        }
    }

    pub fn set_layer_manifest(&self, manifest: HashSet<i32>) {
        let mut m = self.layer_manifest.blocking_write();
        *m = manifest;
    }

    pub fn layer_manifest(&self) -> &tokio::sync::RwLock<HashSet<i32>> {
        &self.layer_manifest
    }

    /// Insert a layer level into the manifest.
    pub async fn insert_layer(&self, level: i32) {
        self.layer_manifest.write().await.insert(level);
    }

    /// Discover which layers have active memories.
    /// Uses a cached manifest updated on every write for O(1) lookup.
    pub async fn discover_active_layers(&self) -> Vec<i32> {
        let manifest = self.layer_manifest.read().await;
        let mut result: Vec<i32> = manifest.iter().copied().collect();
        result.sort();
        result
    }

    /// Force-refresh the layer manifest from the vector store.
    /// Use after bulk operations that bypass the normal write paths.
    pub async fn refresh_layer_manifest(&self) -> Result<()> {
        let layer_counts = self.vector_store.count_by_layer().await?;
        let mut layers: HashSet<i32> = layer_counts
            .into_keys()
            .filter(|&l| l >= 0)
            .collect();
        layers.insert(0);
        *self.layer_manifest.write().await = layers;
        Ok(())
    }

    /// Search for similar memories with importance-weighted ranking and hybrid keyword matching
    pub async fn search(
        &self,
        query: &str,
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        self.search_with_override(query, filters, limit, None).await
    }

    /// Search with an optional similarity threshold override.
    pub async fn search_with_override(
        &self,
        query: &str,
        filters: &Filters,
        limit: usize,
        threshold_override: Option<f32>,
    ) -> Result<Vec<ScoredMemory>> {
        let search_similarity_threshold = threshold_override
            .map(Some)
            .unwrap_or(self.config.search_similarity_threshold);

        let query_keywords = {
            let _guard = self.llm.acquire(LlmPriority::Interactive).await;
            self.llm.inner().extract_keywords(query).await
        };
        let query_keywords = match query_keywords {
            Ok(keywords) => keywords,
            Err(e) => {
                tracing::debug!("Failed to extract keywords from query: {}", e);
                Vec::new()
            }
        };

        let keyword_only = filters
            .custom
            .get("keyword_only")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let results = if keyword_only {
            self.search_by_keywords_inner(query, &query_keywords, filters, limit).await?
        } else {
            self.search_hybrid(query, &query_keywords, filters, limit, search_similarity_threshold).await?
        };

        Ok(results)
    }

    /// Hybrid search: semantic similarity with keyword-based score boosting
    async fn search_hybrid(
        &self,
        query: &str,
        query_keywords: &[String],
        filters: &Filters,
        limit: usize,
        similarity_threshold: Option<f32>,
    ) -> Result<Vec<ScoredMemory>> {
        let mut results = self
            .search_with_threshold(query, filters, limit * 2, similarity_threshold)
            .await?;

        if query_keywords.is_empty() {
            results.truncate(limit);
            return Ok(results);
        }

        let keyword_boost = 0.25f32;

        for scored in &mut results {
            if let Some(keywords_val) = scored.memory.metadata.custom.get("keywords")
                && let Some(memory_keywords) = keywords_val.as_array()
            {
                let memory_kw_strings: Vec<String> = memory_keywords
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_lowercase()))
                    .collect();

                let matches: usize = query_keywords
                    .iter()
                    .filter(|qk| {
                        let qk_lower = qk.to_lowercase();
                        memory_kw_strings.iter().any(|mk| mk.contains(&qk_lower) || qk_lower.contains(mk))
                    })
                    .count();

                if matches > 0 {
                    let boost = keyword_boost * (matches as f32);
                    scored.score = (scored.score + boost).min(1.0);
                }
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(limit);
        Ok(results)
    }

    /// Keyword-only search: find memories by keyword matching without semantic search
    async fn search_by_keywords_inner(
        &self,
        query: &str,
        query_keywords: &[String],
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        if query_keywords.is_empty() {
            return Ok(Vec::new());
        }

        let scan_limit = self.config.raw_content_scan_limit.max(limit * 2);
        let mut candidates = self
            .vector_store
            .search_with_threshold(
                &self.cache.cached_embed(query, LlmPriority::Interactive).await?,
                filters,
                scan_limit,
                Some(0.1),
            )
            .await
            .unwrap_or_else(|_| Vec::new());

        let mut scored_results: Vec<(ScoredMemory, usize)> = Vec::new();

        for mut scored in candidates {
            if let Some(keywords_val) = scored.memory.metadata.custom.get("keywords")
                && let Some(memory_keywords) = keywords_val.as_array()
            {
                let memory_kw_strings: Vec<String> = memory_keywords
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_lowercase()))
                    .collect();

                let matches: usize = query_keywords
                    .iter()
                    .filter(|qk| {
                        let qk_lower = qk.to_lowercase();
                        memory_kw_strings.iter().any(|mk| mk.contains(&qk_lower) || qk_lower.contains(mk))
                    })
                    .count();

                if matches > 0 {
                    let boost = matches as f32 * 0.2;
                    scored.score = (scored.score + boost).min(1.0);
                    scored_results.push((scored, matches));
                }
            }
        }

        scored_results.sort_by(|a, b| b.1.cmp(&a.1).then(
            b.0.score.partial_cmp(&a.0.score).unwrap_or(std::cmp::Ordering::Equal),
        ));
        scored_results.truncate(limit);

        let results: Vec<ScoredMemory> = scored_results
            .into_iter()
            .map(|(sm, _)| sm)
            .collect();

        Ok(results)
    }

    /// Public keyword-only search: extracts keywords from query (via LLM), then
    /// matches against stored memory keywords.  No embeddings used.
    pub async fn search_by_keywords(
        &self,
        query: &str,
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        let query_keywords = {
            let _guard = self.llm.acquire(LlmPriority::Interactive).await;
            self.llm.inner().extract_keywords(query).await
        };
        let query_keywords = match query_keywords {
            Ok(kw) => kw,
            Err(e) => {
                tracing::debug!("Failed to extract keywords for keyword search: {}", e);
                Vec::new()
            }
        };
        self.search_by_keywords_inner(query, &query_keywords, filters, limit).await
    }

    /// Raw content scan: uses vector pre-filter to narrow candidates, then
    /// tokenises the query and matches tokens directly against stored memory
    /// content text. No LLM keywords — pure text match.
    pub async fn search_by_raw_content(
        &self,
        query: &str,
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        let tokens = Self::simple_query_keywords(query);
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        let scan_limit = self.config.raw_content_scan_limit.max(limit);
        let mut candidates = self
            .vector_store
            .search_with_threshold(
                &self.cache.cached_embed(query, LlmPriority::Interactive).await?,
                filters,
                scan_limit,
                Some(0.1),
            )
            .await
            .unwrap_or_else(|_| Vec::new());

        let mut scored: Vec<(ScoredMemory, usize)> = Vec::new();

        for sm in &mut candidates {
            let content_lower = sm.memory
                .content
                .as_deref()
                .unwrap_or("")
                .to_lowercase();

            let matches: usize = tokens
                .iter()
                .filter(|t| content_lower.contains(t.as_str()))
                .count();

            if matches > 0 {
                let boost = matches as f32 * 0.25;
                sm.score = (sm.score + boost).min(1.0);
                scored.push((sm.clone(), matches));
            }
        }

        scored.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then(b.0.score.partial_cmp(&a.0.score).unwrap_or(std::cmp::Ordering::Equal))
        });
        scored.truncate(limit);

        Ok(scored.into_iter().map(|(sm, _)| sm).collect())
    }

    /// Search for similar memories with optional similarity threshold
    pub async fn search_with_threshold(
        &self,
        query: &str,
        filters: &Filters,
        limit: usize,
        similarity_threshold: Option<f32>,
    ) -> Result<Vec<ScoredMemory>> {
        let query_embedding = self.cache.cached_embed(query, LlmPriority::Interactive).await?;
        let threshold = similarity_threshold.or(self.config.search_similarity_threshold);

        let total_memories = match self.vector_store.count().await {
            Ok(count) => count,
            Err(e) => {
                tracing::warn!("Failed to count memories: {}", e);
                0
            }
        };

        let mut results = self
            .vector_store
            .search_with_threshold(&query_embedding, filters, limit, Some(0.0))
            .await?;

        if results.is_empty() {
            tracing::info!(
                "No candidates found for query: \"{}\" with filters: {:?}. (0 raw results). Total memories in bank: {}",
                query, filters, total_memories
            );

            let has_filters = filters.topics.is_some()
                || filters.min_importance.is_some()
                || filters.candidate_ids.is_some();

            if has_filters {
                let relaxed_filters = Filters::default();
                if let Ok(relaxed_results) = self
                    .vector_store
                    .search_with_threshold(&query_embedding, &relaxed_filters, 1, Some(0.0))
                    .await
                {
                    if !relaxed_results.is_empty() {
                        tracing::info!(
                            "Relaxed search found {} results. Top score: {:.4}. It seems your filters are too restrictive!",
                            relaxed_results.len(),
                            relaxed_results[0].score
                        );
                    } else {
                        tracing::info!("Even relaxed search found 0 results. This is strange.");
                    }
                }
            }
            return Ok(vec![]);
        }

        if let Some(best) = results.first() {
            tracing::info!(
                "Query: \"{}\" | Best match score: {:.4} | Candidates found: {} | Total memories: {}",
                query, best.score, results.len(), total_memories
            );
        }

        if let Some(t) = threshold {
            let _original_count = results.len();
            let best_score_so_far = results.first().map(|m| m.score).unwrap_or(0.0);
            results.retain(|m| m.score >= t);

            if results.is_empty() {
                tracing::info!(
                    "All candidates filtered out by threshold {:.2}. Best score was {:.4}",
                    t, best_score_so_far
                );
            }
        }

        results.sort_by(|a, b| {
            let score_a = a.score * 0.7 + a.memory.metadata.importance_score * 0.3;
            let score_b = b.score * 0.7 + b.memory.metadata.importance_score * 0.3;
            match score_b.partial_cmp(&score_a) {
                Some(std::cmp::Ordering::Equal) | None => {
                    b.memory.created_at.cmp(&a.memory.created_at)
                }
                Some(ordering) => ordering,
            }
        });

        // Resolve chunk records to their parent L0 memories.
        // Chunks have `parent_id` set and serve as secondary index entries.
        // When a chunk matches the query, we return the full parent content
        // (which contains the complete session, not just the matching fragment).
        let chunk_count = results.iter().filter(|r| r.memory.metadata.parent_id.is_some()).count();
        if chunk_count > 0 {
            results = self.resolve_chunks(&results, limit).await?;
        }

        Ok(results)
    }

    /// Two-stage retrieval with context-based pre-filtering.
    pub async fn search_with_context(
        &self,
        query: &str,
        context_tags: &[String],
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        if context_tags.is_empty() {
            return self.search(query, filters, limit).await;
        }

        let mut candidate_ids = HashSet::new();
        let ctx_fetch_limit = 50;
        for tag in context_tags {
            let _guard = self.llm.acquire(LlmPriority::Interactive).await;
            let tag_embedding = self.llm.inner().embed(tag).await?;
            drop(_guard);
            let ctx_results = self
                .vector_store
                .search_with_threshold(&tag_embedding, &Filters::default(), ctx_fetch_limit, Some(0.3))
                .await?;
            for scored in &ctx_results {
                candidate_ids.insert(scored.memory.id.clone());
            }
        }

        if candidate_ids.is_empty() {
            tracing::debug!("No context candidates found, falling back to unfiltered search");
            return self.search(query, filters, limit).await;
        }

        let mut constrained_filters = filters.clone();
        constrained_filters.candidate_ids = Some(candidate_ids.into_iter().collect());

        let results = self.search(query, &constrained_filters, limit).await?;

        if results.is_empty() {
            tracing::debug!("Context-constrained search returned 0 results, falling back to global search");
            return self.search(query, filters, limit).await;
        }

        Ok(results)
    }

    /// Hierarchical pyramid search across all abstraction layers.
    pub async fn search_pyramid(
        &self,
        query: &str,
        filters: &Filters,
        limit: usize,
        config: &PyramidConfig,
        threshold_override: Option<f32>,
    ) -> Result<Vec<PyramidResult>> {
        config.validate().map_err(|e| MemoryError::Validation(e.to_string()))?;

        let total_start = Instant::now();
        let base_threshold = threshold_override.or(self.config.search_similarity_threshold);
        let metrics = self.cache.metrics();

        // Phase 0: Discover active layers
        let discover_start = Instant::now();
        let active_layers = self.discover_active_layers().await;
        metrics.record_query_latency(QueryPhase::LayerDiscovery, discover_start.elapsed());
        if active_layers.is_empty() {
            return Ok(Vec::new());
        }

        tracing::debug!(
            "Pyramid search across {} active layers: {:?}",
            active_layers.len(),
            active_layers
        );

        // Phase 1: Parallel layer searches
        let search_start = Instant::now();
        let mut layer_results: HashMap<i32, Vec<ScoredMemory>> = HashMap::new();
        let per_layer_limit = ((limit as f32 * config.per_layer_multiplier) as usize).max(5);

        // Extract simple query keywords (no LLM call) for boosting
        let query_keywords = Self::simple_query_keywords(query);

        let futures: Vec<_> = active_layers
            .iter()
            .map(|&layer| {
                let query = query.to_string();
                let filters = filters.clone();
                let layer_threshold = base_threshold.map(|t| {
                    let relaxation = 1.0 + layer as f32 * config.layer_threshold_relaxation;
                    t / relaxation
                });
                let qk = query_keywords.clone();
                async move {
                    let mut layer_filters = filters.clone();
                    layer_filters.min_layer_level = Some(layer);
                    layer_filters.max_layer_level = Some(layer);
                    let results = self
                        .search_with_threshold(&query, &layer_filters, per_layer_limit, layer_threshold)
                        .await;
                    // Apply keyword boost to the results
                    let results = results.map(|mut r| {
                        Self::boost_with_keywords(&mut r, &qk);
                        r
                    });
                    (layer, results)
                }
            })
            .collect();

        let results_all: Vec<_> = futures::future::join_all(futures).await;
        metrics.record_query_latency(QueryPhase::LayerSearch, search_start.elapsed());

        for (layer, result) in results_all {
            match result {
                Ok(results) => {
                    let count = results.len();
                    if count > 0 {
                        layer_results.insert(layer, results);
                        tracing::debug!("Layer {}: {} results", layer, count);
                    }
                }
                Err(e) => {
                    tracing::warn!("Layer {} search failed: {}", layer, e);
                }
            }
        }

        if layer_results.is_empty() {
            return Ok(Vec::new());
        }

        // Resolve Dynamic mode via LLM query intent classification
        let use_llm = self.config.use_llm_query_classification;
        let resolved_mode = if config.mode == PyramidAllocationMode::Dynamic {
            let classify_start = Instant::now();
            let mode = self.cache.classify_query_intent(query, use_llm).await;
            metrics.record_query_latency(QueryPhase::IntentClassification, classify_start.elapsed());
            metrics.record_allocation_mode(&format!("{:?}", mode));
            mode
        } else {
            config.mode
        };

        // Handle None mode
        if resolved_mode == PyramidAllocationMode::None {
            let all_results: Vec<ScoredMemory> = layer_results.into_values().flatten().collect();
            let mut assembled: Vec<PyramidResult> = all_results
                .into_iter()
                .take(limit)
                .map(|sm| PyramidResult {
                    layer: sm.memory.metadata.layer.level,
                    layer_name: sm.memory.metadata.layer.name_or_default(),
                    memory: sm,
                    search_phase: "flat".to_string(),
                    graph_path: None,
                    source: "intuitive".to_string(),
                })
                .collect();
            assembled.sort_by(|a, b| {
                b.memory.score.partial_cmp(&a.memory.score).unwrap_or(std::cmp::Ordering::Equal)
            });
            assembled.truncate(limit);
            metrics.record_query_latency(QueryPhase::Total, total_start.elapsed());
            metrics.record_result_count(assembled.len());
            return Ok(assembled);
        }

        let layer_weights = config.layer_weights.clone();

        // Phase 2: Pyramid assembly
        let assembly_start = Instant::now();
        let mut assembled = PyramidAssembler::assemble(layer_results, limit, resolved_mode, layer_weights);
        metrics.record_query_latency(QueryPhase::Assembly, assembly_start.elapsed());

        let layer_counts: Vec<(i32, usize)> = assembled
            .iter()
            .fold(HashMap::new(), |mut acc, r| {
                *acc.entry(r.layer).or_insert(0) += 1;
                acc
            })
            .into_iter()
            .collect();
        metrics.record_layer_distribution(&layer_counts);

        // Phase 3: Lightweight graph refinement
        let graph_start = Instant::now();
        let base_count = assembled.len();
        if !assembled.is_empty() {
            let entry_memories: Vec<(Memory, f32)> = assembled
                .iter()
                .take(5)
                .map(|r| (r.memory.memory.clone(), r.memory.score))
                .collect();

            if !entry_memories.is_empty() {
                let engine = GraphSearchEngine::new(TraversalConfig::default())
                    .unwrap_or_else(|_| GraphSearchEngine::new(TraversalConfig::new()).unwrap());

                let store = dyn_clone::clone_box(&*self.vector_store);
                let refine_results = engine
                    .lightweight_refine(&entry_memories, |id: String| {
                        let store = dyn_clone::clone_box(&*store);
                        async move { store.get(&id).await.unwrap_or(None) }
                    })
                    .await;

                let mut discovered = 0;
                for gr in refine_results {
                    let already_present = assembled.iter().any(|r| r.memory.memory.id == gr.memory.id);
                    if already_present { continue; }
                    discovered += 1;
                    let layer = gr.memory.metadata.layer.level;
                    let layer_name = gr.memory.metadata.layer.name_or_default();
                    assembled.push(PyramidResult {
                        memory: ScoredMemory { memory: gr.memory, score: gr.final_score },
                        layer,
                        layer_name,
                        search_phase: "graph_discovered".to_string(),
                        graph_path: Some(gr.path_from_entry),
                        source: "intuitive".to_string(),
                    });
                }
                metrics.record_graph_refinement_yield(discovered, base_count);

                for r in &mut assembled {
                    if r.layer_name.is_empty() {
                        r.layer = r.memory.memory.metadata.layer.level;
                        r.layer_name = r.memory.memory.metadata.layer.name_or_default();
                    }
                }
            }
        }
        metrics.record_query_latency(QueryPhase::GraphRefinement, graph_start.elapsed());

        assembled.sort_by(|a, b| {
            b.memory.score.partial_cmp(&a.memory.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        assembled.truncate(limit);

        metrics.record_query_latency(QueryPhase::Total, total_start.elapsed());
        metrics.record_result_count(assembled.len());

        tracing::info!("Pyramid search returned {} results (mode: {:?})", assembled.len(), resolved_mode);

        Ok(assembled)
    }

    pub async fn get_memory(&self, id: &str) -> Result<Option<Memory>> {
        self.vector_store.get(id).await
    }

    /// Replace chunk results with their parent L0 memories, keeping the best
    /// chunk score per parent. Deduplicates: if parent was already in results,
    /// keeps the original result.
    async fn resolve_chunks(
        &self,
        results: &[ScoredMemory],
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        use std::collections::HashMap;

        let mut regular: Vec<ScoredMemory> = Vec::new();
        let mut chunk_by_parent: HashMap<String, f32> = HashMap::new();

        for r in results {
            match r.memory.metadata.parent_id {
                Some(ref pid) => {
                    let score = r.score * 0.7 + r.memory.metadata.importance_score * 0.3;
                    chunk_by_parent
                        .entry(pid.to_string())
                        .and_modify(|best| {
                            if score > *best {
                                *best = score;
                            }
                        })
                        .or_insert(score);
                }
                None => {
                    regular.push(r.clone());
                }
            }
        }

        if chunk_by_parent.is_empty() {
            return Ok(results.to_vec());
        }

        // Fetch parent memories not already in regular results
        use std::collections::HashSet;
        let existing_ids: HashSet<String> = regular.iter().map(|r| r.memory.id.clone()).collect();

        for (pid_str, score) in chunk_by_parent {
            if existing_ids.contains(&pid_str) {
                continue;
            }
            if let Some(parent) = self.vector_store.get(&pid_str).await? {
                regular.push(ScoredMemory {
                    score,
                    memory: parent,
                });
            }
        }

        if regular.len() > results.len() {
            regular.sort_by(|a, b| {
                let sa = a.score * 0.7 + a.memory.metadata.importance_score * 0.3;
                let sb = b.score * 0.7 + b.memory.metadata.importance_score * 0.3;
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        regular.truncate(limit);
        Ok(regular)
    }

    pub async fn list(&self, filters: &Filters, limit: Option<usize>) -> Result<Vec<Memory>> {
        self.vector_store.list(filters, limit).await
    }

    /// Extract simple keywords from query text (no LLM call).
    /// Splits on non-alphanumeric, lowercases, filters short/stop words.
    fn simple_query_keywords(query: &str) -> Vec<String> {
        let stop_words: &[&str] = &[
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "i", "me", "my", "we", "our",
            "you", "your", "he", "she", "it", "they", "them", "this", "that", "these",
            "those", "what", "which", "who", "whom", "how", "when", "where", "why",
            "if", "then", "than", "in", "on", "at", "to", "for", "of", "with",
            "from", "by", "about", "as", "into", "through", "during", "before",
            "after", "above", "below", "between", "under", "and", "but", "or", "nor",
            "not", "so", "yet", "both", "either", "neither", "each", "every", "all",
            "any", "few", "more", "most", "other", "some", "such", "no", "only",
        ];
        query
            .split(|c: char| !c.is_alphanumeric())
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() >= 3 && !stop_words.contains(&w.as_str()))
            .collect()
    }

    /// Apply keyword boost to scored memory results.
    /// Each match between query keywords and stored memory keywords
    /// adds `keyword_boost` to the score (capped at 1.0).
    fn boost_with_keywords(results: &mut [ScoredMemory], query_keywords: &[String]) {
        if query_keywords.is_empty() {
            return;
        }
        let keyword_boost = 0.25f32;
        for scored in results.iter_mut() {
            if let Some(keywords_val) = scored.memory.metadata.custom.get("keywords")
                && let Some(memory_keywords) = keywords_val.as_array()
            {
                let memory_kw_strings: Vec<String> = memory_keywords
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_lowercase()))
                    .collect();

                let matches: usize = query_keywords
                    .iter()
                    .filter(|qk| {
                        memory_kw_strings.iter().any(|mk| {
                            mk.contains(qk.as_str()) || qk.contains(mk.as_str())
                        })
                    })
                    .count();

                if matches > 0 {
                    let boost = keyword_boost * (matches as f32);
                    scored.score = (scored.score + boost).min(1.0);
                }
            }
        }
    }
}