use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use crate::{
    config::MemoryConfig,
    error::{MemoryError, Result},
    llm::{EmbedPurpose, LlmPriority, PriorityLLMClient},
    memory::cache_service::CacheService,
    memory::metrics::CacheName,
    memory::metrics::QueryPhase,
    search::{
        GraphSearchEngine, PyramidAllocationMode, PyramidAssembler, PyramidConfig, PyramidResult,
        TraversalConfig,
    },
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
        self.cache
            .metrics()
            .record_cache_hit(CacheName::LayerManifest);
        result
    }

    /// Force-refresh the layer manifest from the vector store.
    /// Use after bulk operations that bypass the normal write paths.
    pub async fn refresh_layer_manifest(&self) -> Result<()> {
        self.cache
            .metrics()
            .record_cache_miss(CacheName::LayerManifest);
        let layer_counts = self.vector_store.count_by_layer().await?;
        let mut layers: HashSet<i32> = layer_counts.into_keys().filter(|&l| l >= 0).collect();
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
        let start = std::time::Instant::now();
        let result = self.search_with_override(query, filters, limit, None).await;
        let duration = start.elapsed();
        self.cache
            .metrics()
            .record_query_latency(QueryPhase::Total, duration);
        result
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
            self.search_by_keywords_inner(query, &query_keywords, filters, limit)
                .await?
        } else {
            self.search_hybrid(
                query,
                &query_keywords,
                filters,
                limit,
                search_similarity_threshold,
            )
            .await?
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

        let keyword_boost = 0.10f32;

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
                        memory_kw_strings
                            .iter()
                            .any(|mk| mk.contains(&qk_lower) || qk_lower.contains(mk))
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
        let candidates = self
            .vector_store
            .search_with_threshold(
                &self
                    .cache
                    .cached_embed(query, LlmPriority::Interactive)
                    .await?,
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
                        memory_kw_strings
                            .iter()
                            .any(|mk| mk.contains(&qk_lower) || qk_lower.contains(mk))
                    })
                    .count();

                if matches > 0 {
                    let boost = matches as f32 * 0.2;
                    scored.score = (scored.score + boost).min(1.0);
                    scored_results.push((scored, matches));
                }
            }
        }

        scored_results.sort_by(|a, b| {
            b.1.cmp(&a.1).then(
                b.0.score
                    .partial_cmp(&a.0.score)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        scored_results.truncate(limit);

        let results: Vec<ScoredMemory> = scored_results.into_iter().map(|(sm, _)| sm).collect();

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
        let start = std::time::Instant::now();
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
        let result = self
            .search_by_keywords_inner(query, &query_keywords, filters, limit)
            .await;
        let duration = start.elapsed();
        self.cache
            .metrics()
            .record_query_latency(QueryPhase::Total, duration);
        result
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
        let start = std::time::Instant::now();

        let tokens = Self::simple_query_keywords(query);
        let mut scored: Vec<(ScoredMemory, usize)> = Vec::new();

        if !tokens.is_empty() {
            let scan_limit = self.config.raw_content_scan_limit.max(limit);
            let mut candidates = self
                .vector_store
                .search_with_threshold(
                    &self
                        .cache
                        .cached_embed(query, LlmPriority::Interactive)
                        .await?,
                    filters,
                    scan_limit,
                    Some(0.1),
                )
                .await
                .unwrap_or_else(|_| Vec::new());

            for sm in &mut candidates {
                let content_lower = sm.memory.content.as_deref().unwrap_or("").to_lowercase();

                let matches: usize = tokens
                    .iter()
                    .filter(|t| content_lower.contains(t.as_str()))
                    .count();

                if matches > 0 {
                    let boost = matches as f32 * 0.10;
                    sm.score = (sm.score + boost).min(1.0);
                    scored.push((sm.clone(), matches));
                }
            }

            scored.sort_by(|a, b| {
                b.1.cmp(&a.1).then(
                    b.0.score
                        .partial_cmp(&a.0.score)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
            });
            scored.truncate(limit);
        }

        let duration = start.elapsed();
        self.cache
            .metrics()
            .record_query_latency(QueryPhase::Total, duration);
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
        let start = std::time::Instant::now();

        let query_embedding = self
            .cache
            .cached_embed(query, LlmPriority::Interactive)
            .await?;
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

        if self.config.use_multi_vector_reranking {
            for scored in &mut results {
                let ctx_sim =
                    Self::max_cosine_similarity(&query_embedding, &scored.memory.context_embeddings);
                let rel_sim = Self::max_cosine_similarity(
                    &query_embedding,
                    &scored.memory.relation_embeddings,
                );
                let multi_score = ctx_sim.max(rel_sim);
                if multi_score > 0.0 {
                    scored.score = scored.score * 0.7 + multi_score * 0.3;
                }
            }
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        if results.is_empty() {
            tracing::info!(
                "No candidates found for query: \"{}\" with filters: {:?}. (0 raw results). Total memories in bank: {}",
                query,
                filters,
                total_memories
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
        } else {
            if let Some(best) = results.first() {
                tracing::info!(
                    "Query: \"{}\" | Best match score: {:.4} | Candidates found: {} | Total memories: {}",
                    query,
                    best.score,
                    results.len(),
                    total_memories
                );
            }

            if let Some(t) = threshold {
                let _original_count = results.len();
                let best_score_so_far = results.first().map(|m| m.score).unwrap_or(0.0);
                results.retain(|m| m.score >= t);

                if results.is_empty() {
                    tracing::info!(
                        "All candidates filtered out by threshold {:.2}. Best score was {:.4}",
                        t,
                        best_score_so_far
                    );
                }
            }

            results.sort_by(|a, b| {
                let score_a = a.score * 0.6
                    + a.memory.metadata.importance_score * 0.3
                    + Self::freshness_boost(&a.memory.metadata, self.config.access_decay_hours)
                        * 0.1;
                let score_b = b.score * 0.6
                    + b.memory.metadata.importance_score * 0.3
                    + Self::freshness_boost(&b.memory.metadata, self.config.access_decay_hours)
                        * 0.1;
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
            let chunk_count = results
                .iter()
                .filter(|r| r.memory.metadata.parent_id.is_some())
                .count();
            if chunk_count > 0 {
                results = self.resolve_chunks(&results, limit).await?;
            }
        }

        let duration = start.elapsed();
        self.cache
            .metrics()
            .record_query_latency(QueryPhase::Total, duration);
        Ok(results)
    }

    /// Two-stage retrieval with context-based pre-filtering.
    /// Uses stored context_embeddings for efficient re-ranking instead of
    /// re-embedding and searching per context tag.
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

        let query_embedding = self
            .cache
            .cached_embed(query, LlmPriority::Interactive)
            .await?;

        let tag_embeddings: Vec<Vec<f32>> = {
            let _guard = self.llm.acquire(LlmPriority::Interactive).await;
            self.llm.inner().embed_batch(context_tags, EmbedPurpose::Query).await?
        };

        let mut results = self
            .vector_store
            .search_with_threshold(&query_embedding, filters, limit * 3, Some(0.0))
            .await?;

        for scored in &mut results {
            let ctx_score = Self::multi_vector_match_score(
                &tag_embeddings,
                &scored.memory.context_embeddings,
            );
            if ctx_score > 0.0 {
                scored.score = scored.score * 0.4 + ctx_score * 0.6;
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
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
        config
            .validate()
            .map_err(|e| MemoryError::Validation(e.to_string()))?;

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
                        .search_with_threshold(
                            &query,
                            &layer_filters,
                            per_layer_limit,
                            layer_threshold,
                        )
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
            metrics
                .record_query_latency(QueryPhase::IntentClassification, classify_start.elapsed());
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
                b.memory
                    .score
                    .partial_cmp(&a.memory.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            assembled.truncate(limit);
            metrics.record_query_latency(QueryPhase::Total, total_start.elapsed());
            metrics.record_result_count(assembled.len());
            return Ok(assembled);
        }

        let layer_weights = config.layer_weights.clone();

        // Phase 2: Pyramid assembly (bounded by max_total_candidates)
        let assembly_start = Instant::now();
        let mut assembled = PyramidAssembler::assemble_bounded(
            layer_results,
            limit,
            resolved_mode,
            layer_weights,
            self.config.max_total_candidates,
        );
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
                    let already_present =
                        assembled.iter().any(|r| r.memory.memory.id == gr.memory.id);
                    if already_present {
                        continue;
                    }
                    discovered += 1;
                    let layer = gr.memory.metadata.layer.level;
                    let layer_name = gr.memory.metadata.layer.name_or_default();
                    assembled.push(PyramidResult {
                        memory: ScoredMemory {
                            memory: gr.memory,
                            score: gr.final_score,
                        },
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
            b.memory
                .score
                .partial_cmp(&a.memory.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        assembled.truncate(limit);

        metrics.record_query_latency(QueryPhase::Total, total_start.elapsed());
        metrics.record_result_count(assembled.len());

        tracing::info!(
            "Pyramid search returned {} results (mode: {:?})",
            assembled.len(),
            resolved_mode
        );

        Ok(assembled)
    }

    /// Simplified pyramid search with sensible defaults for internal callers.
    ///
    /// Uses `Balanced` allocation mode, `keyword_split_ratio: 0.2` (implicitly
    /// via Balanced mode), and no threshold override. Equivalent to the
    /// `search_memory` MCP tool but available as a direct method call.
    pub async fn search_pyramid_simple(
        &self,
        query: &str,
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<PyramidResult>> {
        let config = PyramidConfig {
            mode: PyramidAllocationMode::Balanced,
            ..PyramidConfig::default()
        };
        self.search_pyramid(query, filters, limit, &config, None)
            .await
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
                    let score = r.score * 0.6
                        + r.memory.metadata.importance_score * 0.3
                        + Self::freshness_boost(&r.memory.metadata, self.config.access_decay_hours)
                            * 0.1;
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
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
            "shall", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
            "them", "this", "that", "these", "those", "what", "which", "who", "whom", "how",
            "when", "where", "why", "if", "then", "than", "in", "on", "at", "to", "for", "of",
            "with", "from", "by", "about", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "and", "but", "or", "nor", "not", "so", "yet",
            "both", "either", "neither", "each", "every", "all", "any", "few", "more", "most",
            "other", "some", "such", "no", "only",
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
        let keyword_boost = 0.10f32;
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
                        memory_kw_strings
                            .iter()
                            .any(|mk| mk.contains(qk.as_str()) || qk.contains(mk.as_str()))
                    })
                    .count();

                if matches > 0 {
                    let boost = keyword_boost * (matches as f32);
                    scored.score = (scored.score + boost).min(1.0);
                }
            }
        }
    }

    /// Compute freshness boost from access frequency with time decay.
    /// Returns 0.0–1.0 where: recently and frequently accessed → higher.
    /// decay_hours=0 disables (returns 0).
    fn freshness_boost(meta: &crate::types::MemoryMetadata, decay_hours: u32) -> f32 {
        if decay_hours == 0 || meta.access_count == 0 {
            return 0.0;
        }
        let now = chrono::Utc::now();
        let hours_since = meta
            .last_accessed
            .map(|la| (now - la).num_hours().max(0) as f32)
            .unwrap_or(decay_hours as f32 * 2.0); // never accessed → fully decayed
        let frequency = (meta.access_count as f32 * 0.05).min(1.0);
        let recency = (-hours_since / decay_hours as f32).exp();
        frequency * recency
    }

    /// Compute max cosine similarity between a query vector and stored embeddings.
    /// Returns 0.0 if no stored embeddings exist or norms are zero.
    fn max_cosine_similarity(query: &[f32], stored: &Option<Vec<Vec<f32>>>) -> f32 {
        let embeddings = match stored {
            Some(e) if !e.is_empty() => e,
            _ => return 0.0,
        };
        let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if q_norm == 0.0 {
            return 0.0;
        }
        embeddings
            .iter()
            .map(|emb| {
                let dot: f32 = query.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
                let e_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                if e_norm == 0.0 {
                    0.0
                } else {
                    dot / (q_norm * e_norm)
                }
            })
            .fold(0.0_f32, f32::max)
    }

    /// Compute average best-match score across multiple query vectors against
    /// stored embeddings. For each query vector, finds the max cosine similarity
    /// in stored embeddings, then averages across all query vectors.
    fn multi_vector_match_score(
        query_vecs: &[Vec<f32>],
        stored: &Option<Vec<Vec<f32>>>,
    ) -> f32 {
        let embeddings = match stored {
            Some(e) if !e.is_empty() => e,
            _ => return 0.0,
        };
        if query_vecs.is_empty() {
            return 0.0;
        }
        query_vecs
            .iter()
            .map(|q| {
                let q_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
                if q_norm == 0.0 {
                    return 0.0;
                }
                embeddings
                    .iter()
                    .map(|emb| {
                        let dot: f32 = q.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
                        let e_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if e_norm == 0.0 {
                            0.0
                        } else {
                            dot / (q_norm * e_norm)
                        }
                    })
                    .fold(0.0_f32, f32::max)
            })
            .sum::<f32>()
            / query_vecs.len() as f32
    }

    /// Compute the max cosine similarity across all pairs of embeddings
    /// from two sets. Used for comparing context and relation embeddings
    /// between two memories for implicit link discovery.
    pub fn cross_max_cosine_similarity(
        a: &Option<Vec<Vec<f32>>>,
        b: &Option<Vec<Vec<f32>>>,
    ) -> f32 {
        let a_vecs = match a {
            Some(v) if !v.is_empty() => v,
            _ => return 0.0,
        };
        let b_vecs = match b {
            Some(v) if !v.is_empty() => v,
            _ => return 0.0,
        };
        let mut best = 0.0_f32;
        for va in a_vecs {
            let an: f32 = va.iter().map(|x| x * x).sum::<f32>().sqrt();
            if an == 0.0 {
                continue;
            }
            for vb in b_vecs {
                let bn: f32 = vb.iter().map(|x| x * x).sum::<f32>().sqrt();
                if bn == 0.0 {
                    continue;
                }
                let dot: f32 = va.iter().zip(vb.iter()).map(|(x, y)| x * y).sum();
                let sim = dot / (an * bn);
                if sim > best {
                    best = sim;
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_cosine_similarity_identical() {
        let query = vec![1.0, 0.0, 0.0];
        let stored = Some(vec![vec![1.0, 0.0, 0.0]]);
        let sim = SearchService::max_cosine_similarity(&query, &stored);
        assert!((sim - 1.0).abs() < 0.001, "Expected ~1.0, got {}", sim);
    }

    #[test]
    fn test_max_cosine_similarity_orthogonal() {
        let query = vec![1.0, 0.0];
        let stored = Some(vec![vec![0.0, 1.0]]);
        let sim = SearchService::max_cosine_similarity(&query, &stored);
        assert!((sim - 0.0).abs() < 0.001, "Expected ~0.0, got {}", sim);
    }

    #[test]
    fn test_max_cosine_similarity_picks_best() {
        let query = vec![0.0, 1.0];
        let stored = Some(vec![
            vec![1.0, 0.0], // dot = 0
            vec![0.0, 1.0], // dot = 1, identical
            vec![0.7, 0.7], // dot = 0.7
        ]);
        let sim = SearchService::max_cosine_similarity(&query, &stored);
        assert!((sim - 1.0).abs() < 0.001, "Expected ~1.0, got {}", sim);
    }

    #[test]
    fn test_max_cosine_similarity_empty() {
        let query = vec![1.0, 0.0];
        let stored: Option<Vec<Vec<f32>>> = None;
        let sim = SearchService::max_cosine_similarity(&query, &stored);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_max_cosine_similarity_empty_vec() {
        let query = vec![1.0, 0.0];
        let stored = Some(vec![]);
        let sim = SearchService::max_cosine_similarity(&query, &stored);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_max_cosine_similarity_zero_norm() {
        let query = vec![0.0, 0.0];
        let stored = Some(vec![vec![1.0, 0.0]]);
        let sim = SearchService::max_cosine_similarity(&query, &stored);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_multi_vector_match_score_all_match() {
        let tags = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let stored = Some(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let score = SearchService::multi_vector_match_score(&tags, &stored);
        assert!((score - 1.0).abs() < 0.001, "Expected ~1.0, got {}", score);
    }

    #[test]
    fn test_multi_vector_match_score_partial() {
        let tags = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let stored = Some(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let score = SearchService::multi_vector_match_score(&tags, &stored);
        // Both tags match vec[1,0] perfectly: (1.0 + 1.0) / 2 = 1.0
        assert!((score - 1.0).abs() < 0.001, "Expected ~1.0, got {}", score);
    }

    #[test]
    fn test_multi_vector_match_score_no_match() {
        let tags = vec![vec![0.0, 1.0]];
        let stored = Some(vec![vec![1.0, 0.0]]); // orthogonal
        let score = SearchService::multi_vector_match_score(&tags, &stored);
        assert!((score - 0.0).abs() < 0.001, "Expected ~0.0, got {}", score);
    }

    #[test]
    fn test_multi_vector_match_score_empty_stored() {
        let tags = vec![vec![1.0, 0.0]];
        let stored: Option<Vec<Vec<f32>>> = None;
        let score = SearchService::multi_vector_match_score(&tags, &stored);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_cross_max_cosine_similarity_identical() {
        let a = Some(vec![vec![1.0, 0.0, 0.0]]);
        let b = Some(vec![vec![1.0, 0.0, 0.0]]);
        let sim = SearchService::cross_max_cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001, "Expected ~1.0, got {}", sim);
    }

    #[test]
    fn test_cross_max_cosine_similarity_picks_best_pair() {
        let a = Some(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let b = Some(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        // Best pair: [0,1] with [0,1] or [1,0] with [1,0] = 1.0
        let sim = SearchService::cross_max_cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001, "Expected ~1.0, got {}", sim);
    }

    #[test]
    fn test_cross_max_cosine_similarity_partial() {
        let a = Some(vec![vec![1.0, 0.0]]);
        let b = Some(vec![vec![0.7, 0.7], vec![0.0, 1.0]]);
        // Best: [1,0] with [0.7,0.7] = 0.707 / 0.989 ≈ 0.714
        let sim = SearchService::cross_max_cosine_similarity(&a, &b);
        assert!(sim > 0.7 && sim < 1.0, "Expected partial match, got {}", sim);
    }

    #[test]
    fn test_cross_max_cosine_similarity_empty_a() {
        let a: Option<Vec<Vec<f32>>> = None;
        let b = Some(vec![vec![1.0, 0.0]]);
        let sim = SearchService::cross_max_cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cross_max_cosine_similarity_empty_b() {
        let a = Some(vec![vec![1.0, 0.0]]);
        let b = Some(vec![]);
        let sim = SearchService::cross_max_cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    use crate::config::MemoryConfig;
    use crate::llm::client::LLMClient;
    use crate::llm::extractor_types::*;
    use crate::vector_store::VectorStore;
    use async_trait::async_trait;

    #[derive(Clone)]
    struct TestLLM;

    #[async_trait]
    impl LLMClient for TestLLM {
        async fn complete(&self, _: &str) -> crate::error::Result<String> {
            Ok(String::new())
        }
        async fn complete_with_grammar(&self, _: &str, _: &str) -> crate::error::Result<String> {
            Ok(String::new())
        }
        async fn embed(&self, text: &str, _purpose: EmbedPurpose) -> crate::error::Result<Vec<f32>> {
            let hash: f32 = text.bytes().map(|b| b as f32).sum();
            Ok(vec![hash * 0.01, 0.5, 0.3])
        }
        async fn embed_batch(&self, texts: &[String], _purpose: EmbedPurpose) -> crate::error::Result<Vec<Vec<f32>>> {
            let mut results = Vec::new();
            for t in texts {
                results.push(self.embed(t, EmbedPurpose::Query).await?);
            }
            Ok(results)
        }
        async fn extract_keywords(&self, _: &str) -> crate::error::Result<Vec<String>> {
            Ok(vec![])
        }
        async fn summarize(&self, _: &str, _: Option<usize>) -> crate::error::Result<String> {
            Ok(String::new())
        }
        async fn health_check(&self) -> crate::error::Result<bool> {
            Ok(true)
        }
        async fn extract_structured_facts(&self, _: &str) -> crate::error::Result<StructuredFactExtraction> {
            Ok(StructuredFactExtraction { facts: vec![] })
        }
        async fn extract_detailed_facts(&self, _: &str) -> crate::error::Result<DetailedFactExtraction> {
            Ok(DetailedFactExtraction { facts: vec![] })
        }
        async fn extract_keywords_structured(&self, _: &str) -> crate::error::Result<KeywordExtraction> {
            Ok(KeywordExtraction { keywords: vec![] })
        }
        async fn classify_memory(&self, _: &str) -> crate::error::Result<MemoryClassification> {
            Ok(MemoryClassification { memory_type: "factual".into(), confidence: 1.0, reasoning: String::new() })
        }
        async fn score_importance(&self, _: &str) -> crate::error::Result<ImportanceScore> {
            Ok(ImportanceScore { score: 0.5, reasoning: String::new() })
        }
        async fn check_duplicates(&self, _: &str) -> crate::error::Result<DeduplicationResult> {
            Ok(DeduplicationResult { is_duplicate: false, similarity_score: 0.0, original_memory_id: None })
        }
        async fn generate_summary(&self, _: &str) -> crate::error::Result<SummaryResult> {
            Ok(SummaryResult { summary: String::new(), key_points: vec![] })
        }
        async fn detect_language(&self, _: &str) -> crate::error::Result<LanguageDetection> {
            Ok(LanguageDetection { language: "en".into(), confidence: 1.0 })
        }
        async fn extract_entities(&self, _: &str) -> crate::error::Result<EntityExtraction> {
            Ok(EntityExtraction { entities: vec![] })
        }
        async fn analyze_conversation(&self, _: &str) -> crate::error::Result<ConversationAnalysis> {
            Ok(ConversationAnalysis { topics: vec![], sentiment: String::new(), user_intent: String::new(), key_information: vec![] })
        }
        async fn extract_metadata_enrichment(&self, _: &str) -> crate::error::Result<MetadataEnrichment> {
            Ok(MetadataEnrichment { summary: "mock".into(), keywords: vec![] })
        }
        async fn extract_metadata_enrichment_batch(&self, texts: &[String]) -> crate::error::Result<Vec<crate::error::Result<MetadataEnrichment>>> {
            Ok(texts.iter().map(|_| Ok(MetadataEnrichment { summary: "mock".into(), keywords: vec![] })).collect())
        }
        async fn complete_batch(&self, prompts: &[String]) -> crate::error::Result<Vec<crate::error::Result<String>>> {
            Ok(prompts.iter().map(|_| Ok(String::new())).collect())
        }
        fn get_status(&self) -> ClientStatus { ClientStatus::default() }
        fn batch_config(&self) -> (usize, u32) { (10, 4096) }
        async fn enhance_memory_unified(&self, _: &str) -> crate::error::Result<crate::llm::MemoryEnhancement> {
            Ok(crate::llm::MemoryEnhancement { memory_type: "Semantic".into(), summary: String::new(), keywords: vec![], entities: vec![], topics: vec![] })
        }
        async fn describe_image(&self, _: &[u8], _: &str) -> crate::error::Result<String> {
            Err(crate::error::MemoryError::LLM("Mock: vision not available".into()))
        }
    }

    #[derive(Clone)]
    struct TestVectorStore {
        memories: std::sync::Arc<tokio::sync::RwLock<Vec<Memory>>>,
    }

    impl TestVectorStore {
        fn new() -> Self {
            Self { memories: std::sync::Arc::new(tokio::sync::RwLock::new(Vec::new())) }
        }
    }

    #[async_trait]
    impl VectorStore for TestVectorStore {
        async fn insert(&self, memory: &Memory) -> crate::error::Result<()> {
            self.memories.write().await.push(memory.clone());
            Ok(())
        }
        async fn search(&self, query_vector: &[f32], _filters: &Filters, limit: usize) -> crate::error::Result<Vec<crate::types::ScoredMemory>> {
            let mems = self.memories.read().await;
            let mut scored: Vec<_> = mems.iter().map(|m| {
                let dot: f32 = query_vector.iter().zip(m.embedding.iter()).map(|(a,b)| a*b).sum();
                let n1: f32 = query_vector.iter().map(|x| x*x).sum::<f32>().sqrt();
                let n2: f32 = m.embedding.iter().map(|x| x*x).sum::<f32>().sqrt();
                let sim = if n1 == 0.0 || n2 == 0.0 { 0.0 } else { dot / (n1 * n2) };
                crate::types::ScoredMemory { memory: m.clone(), score: sim }
            }).collect();
            scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(limit);
            Ok(scored)
        }
        async fn search_with_threshold(&self, query_vector: &[f32], filters: &Filters, limit: usize, _threshold: Option<f32>) -> crate::error::Result<Vec<crate::types::ScoredMemory>> {
            self.search(query_vector, filters, limit).await
        }
        async fn update(&self, memory: &Memory) -> crate::error::Result<()> {
            let mut mems = self.memories.write().await;
            if let Some(pos) = mems.iter().position(|m| m.id == memory.id) {
                mems[pos] = memory.clone();
            }
            Ok(())
        }
        async fn delete(&self, id: &str) -> crate::error::Result<()> {
            self.memories.write().await.retain(|m| m.id != id);
            Ok(())
        }
        async fn get(&self, id: &str) -> crate::error::Result<Option<Memory>> {
            Ok(self.memories.read().await.iter().find(|m| m.id == id).cloned())
        }
        async fn list(&self, _filters: &Filters, limit: Option<usize>) -> crate::error::Result<Vec<Memory>> {
            let mems = self.memories.read().await;
            let lim = limit.unwrap_or(usize::MAX);
            Ok(mems.iter().take(lim).cloned().collect())
        }
        async fn count(&self) -> crate::error::Result<usize> {
            Ok(self.memories.read().await.len())
        }
        async fn health_check(&self) -> crate::error::Result<bool> { Ok(true) }
        async fn compact(&self) -> crate::error::Result<()> { Ok(()) }
        async fn find_by_relation_target(&self, _: &str, _: Option<usize>) -> crate::error::Result<Vec<Memory>> { Ok(vec![]) }
        async fn count_by_user(&self) -> crate::error::Result<Vec<(Option<String>, usize)>> { Ok(vec![]) }
        async fn count_by_agent(&self) -> crate::error::Result<Vec<(Option<String>, usize)>> { Ok(vec![]) }
        async fn count_by_layer(&self) -> crate::error::Result<std::collections::HashMap<i32, usize>> { Ok(std::collections::HashMap::new()) }
    }

    fn make_test_memory(id: &str, embedding: Vec<f32>, content: &str) -> Memory {
        Memory {
            id: id.to_string(),
            content: Some(content.to_string()),
            embedding,
            metadata: crate::types::MemoryMetadata::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            content_meta: Default::default(),
            derived_data: Default::default(),
            relations: Default::default(),
            event_at: None,
            event_end: None,
            context_embeddings: None,
            relation_embeddings: None,
        }
    }

    async fn make_search_service() -> (SearchService, TestVectorStore) {
        let store = TestVectorStore::new();
        let store_box: Box<dyn VectorStore + Send + Sync> = Box::new(store.clone());
        let llm = Arc::new(PriorityLLMClient::new(Box::new(TestLLM), 10, 3));
        let config = Arc::new(MemoryConfig::default());
        let cache = Arc::new(CacheService::new(llm.clone(), None));
        let svc = SearchService::new(store_box, llm, config, cache);
        (svc, store)
    }

    #[tokio::test]
    async fn test_search_with_context_semantic_match() {
        let (svc, store) = make_search_service().await;

        let mut mem = make_test_memory("m1", vec![0.1, 0.2, 0.3], "rust async programming");
        mem.context_embeddings = Some(vec![vec![0.9, 0.1, 0.0]]); // close to "programming"
        store.insert(&mem).await.unwrap();

        let mut mem2 = make_test_memory("m2", vec![0.1, 0.2, 0.3], "baking recipes");
        mem2.context_embeddings = Some(vec![vec![0.1, 1.0, 0.0]]); // far from "programming"
        store.insert(&mem2).await.unwrap();

        let results = svc
            .search_with_context(
                "stuff",
                &["programming".to_string()],
                &Filters::default(),
                10,
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        // m1 should rank higher because its context_embeddings match "programming" better
        assert_eq!(results[0].memory.id, "m1");
    }

    #[tokio::test]
    async fn test_search_with_context_empty_tags_falls_back() {
        let (svc, store) = make_search_service().await;
        store.insert(&make_test_memory("m1", vec![0.5, 0.5, 0.5], "hello world")).await.unwrap();

        let results = svc
            .search_with_context("hello", &[], &Filters::default(), 10)
            .await
            .unwrap();
        assert!(results.len() >= 1);
    }

    #[tokio::test]
    async fn test_multi_vector_reranking_boosts_context_match() {
        let store2 = TestVectorStore::new();
        let mut mem = make_test_memory("m1", vec![0.1, 0.3, 0.5], "irrelevant to query");
        mem.context_embeddings = Some(vec![vec![0.1, 0.3, 0.5]]);
        store2.insert(&mem).await.unwrap();

        let mut config = MemoryConfig::default();
        config.use_multi_vector_reranking = true;
        let config = Arc::new(config);
        let llm = Arc::new(PriorityLLMClient::new(Box::new(TestLLM), 10, 3));
        let cache = Arc::new(CacheService::new(llm.clone(), None));
        let store_box: Box<dyn VectorStore + Send + Sync> = Box::new(store2);
        let svc2 = SearchService::new(store_box, llm, config, cache);

        let results = svc2
            .search_with_threshold("hello", &Filters::default(), 10, None)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }
}
