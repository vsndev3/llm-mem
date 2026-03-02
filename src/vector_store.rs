use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::{Map, Value, json};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
    },
};
use tracing::{debug, info, warn};
use vectorlite::{
    Collection, FlatIndex, HNSWIndex, IndexType, SimilarityMetric, Vector, VectorIndex,
    VectorIndexWrapper,
};

use crate::{
    config::{VectorLiteSettings, VectorStoreConfig},
    error::{MemoryError, Result},
    types::{Filters, LayerInfo, Memory, MemoryMetadata, MemoryState, MemoryType, ScoredMemory},
};

/// Trait for vector store operations
#[async_trait]
pub trait VectorStore: Send + Sync + dyn_clone::DynClone {
    async fn insert(&self, memory: &Memory) -> Result<()>;
    async fn search(
        &self,
        query_vector: &[f32],
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>>;
    async fn search_with_threshold(
        &self,
        query_vector: &[f32],
        filters: &Filters,
        limit: usize,
        score_threshold: Option<f32>,
    ) -> Result<Vec<ScoredMemory>>;
    async fn update(&self, memory: &Memory) -> Result<()>;
    async fn delete(&self, id: &str) -> Result<()>;
    async fn get(&self, id: &str) -> Result<Option<Memory>>;
    async fn list(&self, filters: &Filters, limit: Option<usize>) -> Result<Vec<Memory>>;
    /// Return the number of memories (not vectors) in the store.
    async fn count(&self) -> Result<usize>;
    async fn health_check(&self) -> Result<bool>;
}

dyn_clone::clone_trait_object!(VectorStore);

// --- VectorLite implementation ---

#[derive(Debug, Clone)]
pub struct VectorLiteConfig {
    pub collection_name: String,
    pub index_type: IndexType,
    pub metric: SimilarityMetric,
    pub persistence_path: Option<PathBuf>,
}

impl Default for VectorLiteConfig {
    fn default() -> Self {
        Self {
            collection_name: "llm-memories".to_string(),
            index_type: IndexType::HNSW,
            metric: SimilarityMetric::Cosine,
            persistence_path: None,
        }
    }
}

impl VectorLiteConfig {
    pub fn from_store_config(store_cfg: &VectorStoreConfig) -> Self {
        Self {
            collection_name: store_cfg.collection_name.clone(),
            index_type: parse_index_type(&store_cfg.vectorlite),
            metric: parse_metric(&store_cfg.vectorlite),
            persistence_path: store_cfg
                .vectorlite
                .persistence_path
                .as_ref()
                .map(PathBuf::from),
        }
    }
}

#[derive(Clone)]
pub struct VectorLiteStore {
    index: Arc<RwLock<Option<VectorIndexWrapper>>>,
    config: VectorLiteConfig,
    next_id: Arc<AtomicU64>,
    memory_index: Arc<RwLock<HashMap<String, Memory>>>,
    // Mapping from Memory ID (String) to list of Vector IDs (u64)
    id_map: Arc<RwLock<HashMap<String, Vec<u64>>>>,
    reverse_id_map: Arc<RwLock<HashMap<u64, String>>>,
    abstraction_index: Arc<RwLock<HashMap<uuid::Uuid, std::collections::HashSet<String>>>>,
}

impl VectorLiteStore {
    pub fn with_config(config: VectorLiteConfig) -> Result<Self> {
        let mut store = Self {
            index: Arc::new(RwLock::new(None)),
            config,
            next_id: Arc::new(AtomicU64::new(0)),
            memory_index: Arc::new(RwLock::new(HashMap::new())),
            id_map: Arc::new(RwLock::new(HashMap::new())),
            abstraction_index: Arc::new(RwLock::new(HashMap::new())),
            reverse_id_map: Arc::new(RwLock::new(HashMap::new())),
        };

        store.try_load_persisted()?;
        info!("Initialized Level 3 VectorLite vector store");
        Ok(store)
    }

    fn try_load_persisted(&mut self) -> Result<()> {
        let Some(path) = &self.config.persistence_path else {
            info!("No persistence path configured, skipping load");
            return Ok(());
        };

        if !path.exists() {
            info!("Persistence file does not exist: {}", path.display());
            return Ok(());
        }

        info!("Loading persisted collection from: {}", path.display());
        let collection = Collection::load_from_file(path)
            .map_err(|e| MemoryError::VectorLite(format!("Failed to load collection: {e}")))?;

        let next_id = collection.next_id();
        info!("Collection loaded: next_id={}", next_id);

        let index_snapshot = collection
            .index_read()
            .map_err(MemoryError::VectorLite)?
            .clone();
        info!("Index snapshot cloned, len={}", index_snapshot.len());

        *self
            .index
            .write()
            .map_err(|e| MemoryError::VectorLite(e.to_string()))? = Some(index_snapshot);
        self.next_id.store(next_id, Ordering::Relaxed);

        let mut loaded_memories = 0;
        let mut loaded_vectors = 0;
        for vector_id in 0..next_id {
            let vector = match collection.get_vector(vector_id) {
                Ok(Some(item)) => item,
                _ => continue,
            };

            let (memory, external_id) = vector_to_memory(vector_id, vector);

            {
                let mut ab_index = self
                    .abstraction_index
                    .write()
                    .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
                for source in &memory.metadata.abstraction_sources {
                    ab_index.entry(*source).or_default().insert(memory.id.clone());
                }
            }

            self.memory_index
                .write()
                .map_err(|e| MemoryError::VectorLite(e.to_string()))?
                .insert(external_id.clone(), memory);
            self.id_map
                .write()
                .map_err(|e| MemoryError::VectorLite(e.to_string()))?
                .entry(external_id.clone())
                .or_insert_with(Vec::new)
                .push(vector_id);
            self.reverse_id_map
                .write()
                .map_err(|e| MemoryError::VectorLite(e.to_string()))?
                .insert(vector_id, external_id);

            loaded_vectors += 1;
            loaded_memories = self
                .memory_index
                .read()
                .map_err(|e| MemoryError::VectorLite(e.to_string()))?
                .len();
        }

        info!(
            "Loaded {} vectors into {} unique memories",
            loaded_vectors, loaded_memories
        );

        Ok(())
    }

    fn ensure_index(&self, dimension: usize) -> Result<()> {
        let mut guard = self
            .index
            .write()
            .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
        if guard.is_none() {
            let idx = match self.config.index_type {
                IndexType::Flat => VectorIndexWrapper::Flat(FlatIndex::new(dimension, Vec::new())),
                IndexType::HNSW => VectorIndexWrapper::HNSW(Box::new(HNSWIndex::new(dimension))),
            };
            *guard = Some(idx);
        }
        Ok(())
    }

    fn persist_if_enabled(&self) -> Result<()> {
        let Some(path) = &self.config.persistence_path else {
            return Ok(());
        };

        let index = self
            .index
            .read()
            .map_err(|e| MemoryError::VectorLite(e.to_string()))?
            .clone();
        let Some(index) = index else {
            return Ok(());
        };

        let collection = Collection::new(self.config.collection_name.clone(), index);
        collection
            .save_to_file(path)
            .map_err(|e| MemoryError::VectorLite(format!("Failed to persist collection: {e}")))
    }

    fn matches_filters(memory: &Memory, filters: &Filters) -> bool {
        // Level 3: candidate_ids filter (from two-stage context retrieval)
        if let Some(ref candidate_ids) = filters.candidate_ids
            && !candidate_ids.contains(&memory.id)
        {
            return false;
        }

        if let Some(ref contains_source) = filters.contains_abstraction_source {
            if !memory.metadata.abstraction_sources.contains(contains_source) {
                return false;
            }
        }

        if let Some(ref user_id) = filters.user_id
            && memory.metadata.user_id.as_ref() != Some(user_id)
        {
            return false;
        }

        if let Some(ref agent_id) = filters.agent_id
            && memory.metadata.agent_id.as_ref() != Some(agent_id)
        {
            return false;
        }

        if let Some(ref run_id) = filters.run_id
            && memory.metadata.run_id.as_ref() != Some(run_id)
        {
            return false;
        }

        if let Some(ref actor_id) = filters.actor_id
            && memory.metadata.actor_id.as_ref() != Some(actor_id)
        {
            return false;
        }

        if let Some(ref memory_type) = filters.memory_type
            && &memory.metadata.memory_type != memory_type
        {
            return false;
        }

        // Layer filtering
        if let Some(ref layer_level) = filters.custom.get("layer.level").and_then(|v| v.as_i64()) {
            if memory.metadata.layer.level != *layer_level as i32 {
                return false;
            }
        }

        // State filtering (default to Active if not specified)
        if let Some(state_value) = filters.custom.get("state").and_then(|v| v.as_str()) {
            let matches_state = match state_value {
                "active" => memory.metadata.state.is_active(),
                "forgotten" => memory.metadata.state.is_forgotten(),
                "processing" => memory.metadata.state.is_processing(),
                "invalid" => memory.metadata.state.is_invalid(),
                _ => true, // Unknown state string - don't filter
            };
            if !matches_state {
                return false;
            }
        } else {
            // Default: only show active memories
            if !memory.metadata.state.is_active() {
                return false;
            }
        }

        if let Some(min_importance) = filters.min_importance
            && memory.metadata.importance_score < min_importance
        {
            return false;
        }

        if let Some(max_importance) = filters.max_importance
            && memory.metadata.importance_score > max_importance
        {
            return false;
        }

        if let Some(created_after) = filters.created_after
            && memory.created_at < created_after
        {
            return false;
        }

        if let Some(created_before) = filters.created_before
            && memory.created_at > created_before
        {
            return false;
        }

        if let Some(updated_after) = filters.updated_after
            && memory.updated_at < updated_after
        {
            return false;
        }

        if let Some(updated_before) = filters.updated_before
            && memory.updated_at > updated_before
        {
            return false;
        }

        if let Some(ref entities) = filters.entities
            && !entities
                .iter()
                .any(|entity| memory.metadata.entities.contains(entity))
        {
            return false;
        }

        if let Some(ref topics) = filters.topics
            && !topics
                .iter()
                .any(|topic| memory.metadata.topics.contains(topic))
        {
            return false;
        }

        if let Some(ref relations) = filters.relations {
            // Check if ANY of the requested relation filters match ANY of the memory's relations
            // This is effectively an OR filter for the list of desired relations
            // But within a single RelationFilter, both predicate and target must match (AND)
            // We use case-insensitive matching to handle slight variations (e.g. "Pizza" vs "pizza")
            let has_match = relations.iter().any(|filter_rel| {
                memory.metadata.relations.iter().any(|mem_rel| {
                    (filter_rel.relation.is_empty()
                        || mem_rel.relation.eq_ignore_ascii_case(&filter_rel.relation))
                        && (filter_rel.target.is_empty()
                            || mem_rel.target.eq_ignore_ascii_case(&filter_rel.target))
                })
            });
            if !has_match {
                return false;
            }
        }

        for (key, value) in &filters.custom {
            if memory.metadata.custom.get(key) != Some(value) {
                return false;
            }
        }

        true
    }

    /// Brute-force cosine similarity search over all memories in the index.
    /// Used as a fallback when the HNSW index is too small (< 2 items) or
    /// when the HNSW search panics due to the hnsw 0.11.0 bug.
    fn brute_force_search(
        &self,
        query_vector: &[f32],
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        let memory_index = self
            .memory_index
            .read()
            .map_err(|e| MemoryError::VectorLite(e.to_string()))?;

        info!(
            "Brute force search: {} memories, query_dim={}",
            memory_index.len(),
            query_vector.len()
        );

        let mut scored: Vec<ScoredMemory> = memory_index
            .values()
            .filter(|m| {
                let matches = Self::matches_filters(m, filters);
                if !matches {
                    debug!("Memory {} filtered out", m.id);
                }
                matches
            })
            .map(|m| {
                let score = cosine_similarity(query_vector, &m.embedding);
                debug!("Memory {} score: {}", m.id, score);
                ScoredMemory {
                    memory: m.clone(),
                    score,
                }
            })
            .collect();

        info!("Brute force: {} memories passed filters", scored.len());

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(limit);

        info!(count = scored.len(), "Brute-force search completed");
        Ok(scored)
    }
}

#[async_trait]
impl VectorStore for VectorLiteStore {
    async fn insert(&self, memory: &Memory) -> Result<()> {
        self.ensure_index(memory.embedding.len())?;

        // Track successfully inserted vectors for rollback
        let mut inserted_vector_ids: Vec<u64> = Vec::new();

        // Use a result-capturing block to handle errors and trigger rollback
        let insert_result = (|| {
            // --- 1. Insert the main content vector ---
            let content_vector_id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let mut content_meta = memory_metadata_to_json(memory);
            content_meta["vector_type"] = json!("content");
            let content_vector = Vector {
                id: content_vector_id,
                values: memory.embedding.iter().map(|v| *v as f64).collect(),
                text: memory.content.clone().unwrap_or_default(),
                metadata: Some(content_meta),
            };

            {
                let mut guard = self
                    .index
                    .write()
                    .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
                let index = guard
                    .as_mut()
                    .ok_or_else(|| MemoryError::VectorLite("Index not initialized".to_string()))?;
                index.add(content_vector).map_err(MemoryError::VectorLite)?;
            }
            inserted_vector_ids.push(content_vector_id);
            self.reverse_id_map
                .write()
                .map_err(|e| MemoryError::VectorLite(e.to_string()))?
                .insert(content_vector_id, memory.id.clone());

            // --- 2. Insert context vectors (if present) ---
            if let Some(ref ctx_embeddings) = memory.context_embeddings {
                for (idx, ctx_emb) in ctx_embeddings.iter().enumerate() {
                    let ctx_tag = memory
                        .metadata
                        .context
                        .get(idx)
                        .cloned()
                        .unwrap_or_default();
                    let ctx_vector_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                    let ctx_meta = json!({
                        "id": memory.id,
                        "vector_type": "context",
                        "target_memory_id": memory.id,
                        "context_tag": ctx_tag,
                    });
                    let ctx_vector = Vector {
                        id: ctx_vector_id,
                        values: ctx_emb.iter().map(|v| *v as f64).collect(),
                        text: ctx_tag,
                        metadata: Some(ctx_meta),
                    };
                    {
                        let mut guard = self
                            .index
                            .write()
                            .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
                        let index = guard.as_mut().ok_or_else(|| {
                            MemoryError::VectorLite("Index not initialized".to_string())
                        })?;
                        index.add(ctx_vector).map_err(MemoryError::VectorLite)?;
                    }
                    inserted_vector_ids.push(ctx_vector_id);
                    self.reverse_id_map
                        .write()
                        .map_err(|e| MemoryError::VectorLite(e.to_string()))?
                        .insert(ctx_vector_id, memory.id.clone());
                }
            }

            // --- 3. Insert relation vectors (if present) ---
            if let Some(ref rel_embeddings) = memory.relation_embeddings {
                for (idx, rel_emb) in rel_embeddings.iter().enumerate() {
                    let rel_text = memory
                        .metadata
                        .relations
                        .get(idx)
                        .map(|r| format!("{} {}", r.relation, r.target))
                        .unwrap_or_default();
                    let rel_vector_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                    let rel_meta = json!({
                        "id": memory.id,
                        "vector_type": "relation",
                        "target_memory_id": memory.id,
                        "relation_text": rel_text,
                    });
                    let rel_vector = Vector {
                        id: rel_vector_id,
                        values: rel_emb.iter().map(|v| *v as f64).collect(),
                        text: rel_text,
                        metadata: Some(rel_meta),
                    };
                    {
                        let mut guard = self
                            .index
                            .write()
                            .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
                        let index = guard.as_mut().ok_or_else(|| {
                            MemoryError::VectorLite("Index not initialized".to_string())
                        })?;
                        index.add(rel_vector).map_err(MemoryError::VectorLite)?;
                    }
                    inserted_vector_ids.push(rel_vector_id);
                    self.reverse_id_map
                        .write()
                        .map_err(|e| MemoryError::VectorLite(e.to_string()))?
                        .insert(rel_vector_id, memory.id.clone());
                }
            }

            // --- 4. Update memory index and id_map ---
            self.memory_index
                .write()
                .map_err(|e| MemoryError::VectorLite(e.to_string()))?
                .insert(memory.id.clone(), memory.clone());
            self.id_map
                .write()
                .map_err(|e| MemoryError::VectorLite(e.to_string()))?
                .insert(memory.id.clone(), inserted_vector_ids.clone());

            let mut ab_index = self
                .abstraction_index
                .write()
                .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
            for source in &memory.metadata.abstraction_sources {
                ab_index.entry(*source).or_default().insert(memory.id.clone());
            }

            Result::Ok(())
        })();

        if let Err(err) = insert_result {
            // Rollback: Attempt to remove any vectors that were successfully inserted
            if !inserted_vector_ids.is_empty() {
                // Ignore rollback errors as we are already in an error state
                let _ = (|| -> Result<()> {
                    let mut guard = self
                        .index
                        .write()
                        .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
                    let index = guard.as_mut().ok_or_else(|| {
                        MemoryError::VectorLite("Index not initialized".to_string())
                    })?;

                    for &vid in &inserted_vector_ids {
                        let _ = index.delete(vid); // Best effort delete
                    }

                    let mut rev_map = self
                        .reverse_id_map
                        .write()
                        .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
                    for vid in &inserted_vector_ids {
                        rev_map.remove(vid);
                    }

                    // Note: We don't need to clean memory_index or id_map because they are updated last
                    // inside the block, so if we failed before that, they are untouched.
                    Ok(())
                })();
            }
            return Err(err);
        }

        debug!(
            memory_id = %memory.id,
            content_vectors = 1,
            context_vectors = memory.context_embeddings.as_ref().map(|v| v.len()).unwrap_or(0),
            relation_vectors = memory.relation_embeddings.as_ref().map(|v| v.len()).unwrap_or(0),
            "Multi-vector insert completed"
        );

        self.persist_if_enabled()?;
        Ok(())
    }

    async fn search(
        &self,
        query_vector: &[f32],
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        let guard = self
            .index
            .read()
            .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
        let Some(index) = guard.as_ref() else {
            debug!("Search skipped: Vector index is not initialized (empty database)");
            return Ok(Vec::new());
        };

        // Determine safe search limit to prevent HNSW panic.
        //
        // The hnsw 0.11.0 crate has a bug in copy_from_slice where it panics
        // when requested k > items in the index.
        //
        // Vectorlite's HNSWIndex::search() internally passes k*2 to hnsw's nearest(),
        // so the value WE pass must satisfy: our_k * 2 <= hnsw_internal_count.
        //
        // We use index.len() (vectorlite's alive vector count) which is always
        // <= hnsw's internal count. Dividing by 2 accounts for vectorlite's 2x multiplier.
        let index_len = index.len();
        let memory_index_len = self
            .memory_index
            .read()
            .map_err(|e| MemoryError::VectorLite(e.to_string()))?
            .len();
        info!(
            "Search starting: index_len={}, memory_index_len={}, query_dim={}, limit={}",
            index_len,
            memory_index_len,
            query_vector.len(),
            limit
        );

        // For tiny indexes (< 2 items), HNSW search is unsafe because vectorlite
        // doubles k internally. Fall back to brute-force cosine similarity scan.
        if index_len < 2 {
            info!("Index too small ({}), using brute force", index_len);
            drop(guard);
            return self.brute_force_search(query_vector, filters, limit);
        }

        let query: Vec<f64> = query_vector.iter().map(|v| *v as f64).collect();
        // Over-fetch to account for multiple vectors per memory, capped at 500.
        // CRITICAL: The hnsw crate panics if the requested k is greater than the
        // number of elements in the index. Vectorlite internally passes k*2 to hnsw.
        // We must ensure fetch_limit * 2 <= index_len.
        let safe_max = index_len / 2;
        let desired_fetch = limit.saturating_mul(4).max(1).min(500);

        info!(
            "Search params: safe_max={}, desired_fetch={}",
            safe_max, desired_fetch
        );

        // If HNSW can't safely fetch enough results (safe_max < desired_fetch),
        // fall back to brute-force. This covers small-to-medium indexes and avoids
        // the panic while still being fast since the dataset is small enough that
        // brute-force cosine scan is negligible.
        if safe_max < desired_fetch {
            info!("Safe max too small, using brute force");
            drop(guard);
            return self.brute_force_search(query_vector, filters, limit);
        }

        // CRITICAL FIX: Even if safe_max >= desired_fetch, we must ensure we don't
        // pass a fetch_limit that causes hnsw to panic.
        let fetch_limit = desired_fetch.min(safe_max);

        // Wrap search in catch_unwind as defense-in-depth against hnsw 0.11.0 panic
        let metric = self.config.metric;
        let search_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            index.search(&query, fetch_limit, metric)
        }));

        let results = match search_result {
            Ok(r) => {
                info!("HNSW search returned {} results", r.len());
                r
            }
            Err(_) => {
                tracing::warn!(
                    "HNSW search panicked (fetch_limit={}, index_len={}), falling back to brute-force",
                    fetch_limit,
                    index_len
                );
                // We must drop the guard before calling brute_force_search to avoid deadlock
                drop(guard);
                return self.brute_force_search(query_vector, filters, limit);
            }
        };
        drop(guard);

        let memory_index = self
            .memory_index
            .read()
            .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
        let reverse_id_map = self
            .reverse_id_map
            .read()
            .map_err(|e| MemoryError::VectorLite(e.to_string()))?;

        info!(
            "After search: memory_index has {} entries, reverse_id_map has {} entries",
            memory_index.len(),
            reverse_id_map.len()
        );

        // Deduplicate: multiple vectors (content/context/relation) may point
        // to the same memory. Keep the highest score per memory ID.
        let mut best_scores: HashMap<String, f32> = HashMap::new();
        for result in &results {
            if let Some(external_id) = reverse_id_map.get(&result.id) {
                let score = result.score as f32;
                let entry = best_scores.entry(external_id.clone()).or_insert(0.0);
                if score > *entry {
                    *entry = score;
                }
            } else {
                warn!("No external_id mapping for vector_id {}", result.id);
            }
        }

        info!("After dedup: {} unique memories", best_scores.len());

        let mut output: Vec<ScoredMemory> = best_scores
            .into_iter()
            .filter_map(|(mem_id, score)| {
                let memory = memory_index.get(&mem_id)?;
                if Self::matches_filters(memory, filters) {
                    Some(ScoredMemory {
                        memory: memory.clone(),
                        score,
                    })
                } else {
                    debug!("Memory {} filtered out by filters", mem_id);
                    None
                }
            })
            .collect();

        info!("After filtering: {} memories", output.len());

        // Sort by score descending
        output.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        output.truncate(limit);

        info!(
            count = output.len(),
            "VectorLite multi-vector search completed"
        );
        Ok(output)
    }

    async fn search_with_threshold(
        &self,
        query_vector: &[f32],
        filters: &Filters,
        limit: usize,
        score_threshold: Option<f32>,
    ) -> Result<Vec<ScoredMemory>> {
        let results = self.search(query_vector, filters, limit).await?;
        if let Some(threshold) = score_threshold {
            Ok(results
                .into_iter()
                .filter(|item| item.score >= threshold)
                .collect())
        } else {
            Ok(results)
        }
    }

    async fn update(&self, memory: &Memory) -> Result<()> {
        self.delete(&memory.id).await?;
        self.insert(memory).await
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let vector_ids = self
            .id_map
            .read()
            .map_err(|e| MemoryError::VectorLite(e.to_string()))?
            .get(id)
            .cloned()
            .ok_or_else(|| MemoryError::NotFound { id: id.to_string() })?;

        // Update maps FIRST to prevent race condition with search.
        // If we delete from index first, we create a window where Map Count > Index Count,
        // which causes search to panic when requesting more items than exist in Index.
        // By reducing Map Count first, we ensure `available_count` in search is always valid (<= Index Count).

        let old_memory = self.memory_index
            .write()
            .map_err(|e| MemoryError::VectorLite(e.to_string()))?
            .remove(id);

        if let Some(old_mem) = old_memory {
            let mut ab_index = self
                .abstraction_index
                .write()
                .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
            for source in &old_mem.metadata.abstraction_sources {
                if let Some(set) = ab_index.get_mut(source) {
                    set.remove(id);
                    if set.is_empty() {
                        ab_index.remove(source);
                    }
                }
            }
        }

        self.id_map
            .write()
            .map_err(|e| MemoryError::VectorLite(e.to_string()))?
            .remove(id);
        {
            let mut rev_map = self
                .reverse_id_map
                .write()
                .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
            for vid in &vector_ids {
                rev_map.remove(vid);
            }
        }

        // Delete all vectors from the index AFTER map updates
        {
            let mut guard = self
                .index
                .write()
                .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
            let index = guard
                .as_mut()
                .ok_or_else(|| MemoryError::VectorLite("Index not initialized".to_string()))?;
            for &vid in &vector_ids {
                index.delete(vid).map_err(MemoryError::VectorLite)?;
            }
        }

        self.persist_if_enabled()?;
        Ok(())
    }

    async fn get(&self, id: &str) -> Result<Option<Memory>> {
        let index = self
            .memory_index
            .read()
            .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
        Ok(index.get(id).cloned())
    }

    async fn list(&self, filters: &Filters, limit: Option<usize>) -> Result<Vec<Memory>> {
        let index = self
            .memory_index
            .read()
            .map_err(|e| MemoryError::VectorLite(e.to_string()))?;

        if let Some(source_id) = &filters.contains_abstraction_source {
            let ab_index = self
                .abstraction_index
                .read()
                .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
            
            let empty_set = std::collections::HashSet::new();
            let matching_ids = ab_index.get(source_id).unwrap_or(&empty_set);
            
            let mut memories: Vec<Memory> = matching_ids
                .iter()
                .filter_map(|id| index.get(id))
                .filter(|memory| Self::matches_filters(memory, filters))
                .cloned()
                .collect();
                
            memories.sort_by(|a, b| b.created_at.cmp(&a.created_at));
            if let Some(limit_val) = limit {
                memories.truncate(limit_val);
            }
            return Ok(memories);
        }

        let mut memories: Vec<Memory> = index
            .values()
            .filter(|memory| Self::matches_filters(memory, filters))
            .cloned()
            .collect();
        memories.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        if let Some(limit) = limit {
            memories.truncate(limit);
        }

        Ok(memories)
    }

    async fn count(&self) -> Result<usize> {
        let index = self
            .memory_index
            .read()
            .map_err(|e| MemoryError::VectorLite(e.to_string()))?;
        Ok(index.len())
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
}

// --- Helpers ---

fn parse_index_type(settings: &VectorLiteSettings) -> IndexType {
    match settings.index_type.to_lowercase().as_str() {
        "flat" => IndexType::Flat,
        _ => IndexType::HNSW,
    }
}

fn parse_metric(settings: &VectorLiteSettings) -> SimilarityMetric {
    match settings.metric.to_lowercase().as_str() {
        "euclidean" => SimilarityMetric::Euclidean,
        "manhattan" => SimilarityMetric::Manhattan,
        "dotproduct" => SimilarityMetric::DotProduct,
        _ => SimilarityMetric::Cosine,
    }
}

/// Compute cosine similarity between two vectors.
/// Returns a value in [-1, 1] where 1 means identical direction.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn memory_metadata_to_json(memory: &Memory) -> Value {
    let mut custom = Map::new();
    for (key, value) in &memory.metadata.custom {
        custom.insert(key.clone(), value.clone());
    }

    json!({
        "id": memory.id,
        "user_id": memory.metadata.user_id,
        "agent_id": memory.metadata.agent_id,
        "run_id": memory.metadata.run_id,
        "actor_id": memory.metadata.actor_id,
        "role": memory.metadata.role,
        "memory_type": format!("{:?}", memory.metadata.memory_type),
        "hash": memory.metadata.hash,
        "importance_score": memory.metadata.importance_score,
        "entities": memory.metadata.entities,
        "relations": memory.metadata.relations,
        "context": memory.metadata.context,
        "topics": memory.metadata.topics,
        "custom": custom,
        "created_at": memory.created_at.to_rfc3339(),
        "updated_at": memory.updated_at.to_rfc3339(),
    })
}

fn vector_to_memory(_vector_id: u64, vector: Vector) -> (Memory, String) {
    let metadata = vector.metadata.unwrap_or_else(|| json!({}));
    let id = metadata
        .get("id")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| _vector_id.to_string());

    let created_at = metadata
        .get("created_at")
        .and_then(|value| value.as_str())
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        .map(|value| value.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);
    let updated_at = metadata
        .get("updated_at")
        .and_then(|value| value.as_str())
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        .map(|value| value.with_timezone(&Utc))
        .unwrap_or(created_at);

    // Build memory using the new schema
    let mut memory = Memory::with_content(
        vector.text.clone(),
        vector.values.iter().map(|value| *value as f32).collect(),
        MemoryMetadata {
            user_id: metadata
                .get("user_id")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            agent_id: metadata
                .get("agent_id")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            run_id: metadata
                .get("run_id")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            actor_id: metadata
                .get("actor_id")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            role: metadata
                .get("role")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            memory_type: metadata
                .get("memory_type")
                .and_then(|v| v.as_str())
                .map(MemoryType::parse)
                .unwrap_or(MemoryType::Conversational),
            hash: metadata
                .get("hash")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            importance_score: metadata
                .get("importance_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5) as f32,
            entities: metadata
                .get("entities")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            relations: metadata
                .get("relations")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default(),
            context: metadata
                .get("context")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            topics: metadata
                .get("topics")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            custom: metadata
                .get("custom")
                .and_then(|v| v.as_object())
                .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default(),
            // Layer metadata fields
            layer: LayerInfo::default(),
            abstraction_sources: Vec::new(),
            abstraction_confidence: None,
            state: MemoryState::Active,
            forgotten_at: None,
            forgotten_by: None,
        },
    );
    
    // Override timestamps from metadata
    memory.created_at = created_at;
    memory.updated_at = updated_at;

    (memory, id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Filters, MemoryMetadata, MemoryType};

    const DIM: usize = 8;

    fn make_embedding(seed: f32) -> Vec<f32> {
        (0..DIM).map(|i| seed + i as f32 * 0.1).collect()
    }

    fn make_memory(content: &str, user_id: &str, memory_type: MemoryType) -> Memory {
        let meta = MemoryMetadata::new(memory_type).with_user_id(user_id.to_string());
        Memory::with_content(content.to_string(), make_embedding(1.0), meta)
    }

    fn make_store() -> VectorLiteStore {
        VectorLiteStore::with_config(VectorLiteConfig {
            collection_name: "test".to_string(),
            index_type: IndexType::Flat,
            metric: SimilarityMetric::Cosine,
            persistence_path: None,
        })
        .unwrap()
    }

    #[tokio::test]
    async fn test_insert_and_get() {
        let store = make_store();
        let mem = make_memory("hello world", "u1", MemoryType::Factual);
        let id = mem.id.clone();

        store.insert(&mem).await.unwrap();
        let retrieved = store.get(&id).await.unwrap();

        assert!(retrieved.is_some());
        let r = retrieved.unwrap();
        assert_eq!(r.content, Some("hello world".to_string()));
        assert_eq!(r.metadata.user_id.as_deref(), Some("u1"));
        assert_eq!(r.metadata.memory_type, MemoryType::Factual);
    }

    #[tokio::test]
    async fn test_get_nonexistent() {
        let store = make_store();
        let result = store.get("nonexistent-id").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_insert_multiple_and_list() {
        let store = make_store();
        let m1 = make_memory("mem1", "u1", MemoryType::Factual);
        let m2 = make_memory("mem2", "u1", MemoryType::Procedural);
        let m3 = make_memory("mem3", "u2", MemoryType::Factual);

        store.insert(&m1).await.unwrap();
        store.insert(&m2).await.unwrap();
        store.insert(&m3).await.unwrap();

        let all = store.list(&Filters::new(), None).await.unwrap();
        assert_eq!(all.len(), 3);

        // Filter by user
        let u1 = store.list(&Filters::for_user("u1"), None).await.unwrap();
        assert_eq!(u1.len(), 2);

        let u2 = store.list(&Filters::for_user("u2"), None).await.unwrap();
        assert_eq!(u2.len(), 1);
        assert_eq!(u2[0].content, Some("mem3".to_string()));
    }

    #[tokio::test]
    async fn test_list_with_memory_type_filter() {
        let store = make_store();
        store
            .insert(&make_memory("a", "u1", MemoryType::Factual))
            .await
            .unwrap();
        store
            .insert(&make_memory("b", "u1", MemoryType::Procedural))
            .await
            .unwrap();
        store
            .insert(&make_memory("c", "u1", MemoryType::Factual))
            .await
            .unwrap();

        let factual = store
            .list(
                &Filters::for_user("u1").with_memory_type(MemoryType::Factual),
                None,
            )
            .await
            .unwrap();
        assert_eq!(factual.len(), 2);
    }

    #[tokio::test]
    async fn test_list_with_limit() {
        let store = make_store();
        for i in 0..10 {
            store
                .insert(&make_memory(
                    &format!("mem{}", i),
                    "u1",
                    MemoryType::Factual,
                ))
                .await
                .unwrap();
        }

        let limited = store.list(&Filters::new(), Some(3)).await.unwrap();
        assert_eq!(limited.len(), 3);
    }

    #[tokio::test]
    async fn test_delete() {
        let store = make_store();
        let mem = make_memory("to delete", "u1", MemoryType::Factual);
        let id = mem.id.clone();

        store.insert(&mem).await.unwrap();
        assert!(store.get(&id).await.unwrap().is_some());

        store.delete(&id).await.unwrap();
        assert!(store.get(&id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_delete_nonexistent_returns_error() {
        let store = make_store();
        let result = store.delete("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_update() {
        let store = make_store();
        let mut mem = make_memory("original", "u1", MemoryType::Factual);
        let id = mem.id.clone();

        store.insert(&mem).await.unwrap();

        // Update by creating new memory with same ID
        mem.content = Some("updated".to_string());
        mem.embedding = make_embedding(2.0);
        mem.updated_at = chrono::Utc::now();
        store.update(&mem).await.unwrap();

        let retrieved = store.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.content, Some("updated".to_string()));
    }

    #[tokio::test]
    async fn test_search() {
        let store = make_store();
        store
            .insert(&make_memory("rust programming", "u1", MemoryType::Factual))
            .await
            .unwrap();
        store
            .insert(&make_memory("python scripting", "u1", MemoryType::Factual))
            .await
            .unwrap();

        let query_vec = make_embedding(1.0);
        let results = store.search(&query_vec, &Filters::new(), 10).await.unwrap();

        assert!(!results.is_empty());
        // All results should have score > 0
        for r in &results {
            assert!(r.score > 0.0, "Score should be positive, got {}", r.score);
        }
    }

    #[tokio::test]
    async fn test_search_with_filters() {
        let store = make_store();
        store
            .insert(&make_memory("rust for u1", "u1", MemoryType::Factual))
            .await
            .unwrap();
        store
            .insert(&make_memory("rust for u2", "u2", MemoryType::Factual))
            .await
            .unwrap();

        let query_vec = make_embedding(1.0);
        let u1_results = store
            .search(&query_vec, &Filters::for_user("u1"), 10)
            .await
            .unwrap();
        assert_eq!(u1_results.len(), 1);
        assert_eq!(u1_results[0].memory.metadata.user_id.as_deref(), Some("u1"));
    }

    #[tokio::test]
    async fn test_search_with_threshold() {
        let store = make_store();
        store
            .insert(&make_memory("content", "u1", MemoryType::Factual))
            .await
            .unwrap();

        let query_vec = make_embedding(1.0);

        // Very high threshold should filter most/all
        let results = store
            .search_with_threshold(&query_vec, &Filters::new(), 10, Some(0.9999))
            .await
            .unwrap();
        // Results depend on cosine similarity, but threshold filtering works
        for r in &results {
            assert!(r.score >= 0.9999);
        }
    }

    #[tokio::test]
    async fn test_search_empty_store() {
        let store = make_store();
        let query_vec = make_embedding(1.0);
        let results = store.search(&query_vec, &Filters::new(), 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_health_check() {
        let store = make_store();
        assert!(store.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_persistence_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_collection.bin");

        let config = VectorLiteConfig {
            collection_name: "persist-test".to_string(),
            index_type: IndexType::Flat,
            metric: SimilarityMetric::Cosine,
            persistence_path: Some(path.clone()),
        };

        // Insert and persist
        {
            let store = VectorLiteStore::with_config(config.clone()).unwrap();
            let mem = make_memory("persistent content", "u1", MemoryType::Factual);
            store.insert(&mem).await.unwrap();
        }

        // Load from persisted file
        {
            let store = VectorLiteStore::with_config(config).unwrap();
            let all = store.list(&Filters::new(), None).await.unwrap();
            assert_eq!(all.len(), 1);
            assert_eq!(all[0].content, Some("persistent content".to_string()));
        }
    }

    // --- matches_filters comprehensive tests ---

    #[test]
    fn test_matches_filters_empty_filters() {
        let mem = make_memory("test", "u1", MemoryType::Factual);
        assert!(VectorLiteStore::matches_filters(&mem, &Filters::new()));
    }

    #[test]
    fn test_matches_filters_user_id_match() {
        let mem = make_memory("test", "u1", MemoryType::Factual);
        assert!(VectorLiteStore::matches_filters(
            &mem,
            &Filters::for_user("u1")
        ));
    }

    #[test]
    fn test_matches_filters_user_id_mismatch() {
        let mem = make_memory("test", "u1", MemoryType::Factual);
        assert!(!VectorLiteStore::matches_filters(
            &mem,
            &Filters::for_user("u2")
        ));
    }

    #[test]
    fn test_matches_filters_memory_type() {
        let mem = make_memory("test", "u1", MemoryType::Procedural);
        assert!(VectorLiteStore::matches_filters(
            &mem,
            &Filters::new().with_memory_type(MemoryType::Procedural),
        ));
        assert!(!VectorLiteStore::matches_filters(
            &mem,
            &Filters::new().with_memory_type(MemoryType::Factual),
        ));
    }

    #[test]
    fn test_matches_filters_importance() {
        let meta = MemoryMetadata::new(MemoryType::Factual)
            .with_user_id("u1".to_string())
            .with_importance_score(0.7);
        let mem = Memory::with_content("test".into(), make_embedding(1.0), meta);

        let mut f = Filters::new();
        f.min_importance = Some(0.5);
        assert!(VectorLiteStore::matches_filters(&mem, &f));

        f.min_importance = Some(0.8);
        assert!(!VectorLiteStore::matches_filters(&mem, &f));

        let mut f2 = Filters::new();
        f2.max_importance = Some(0.9);
        assert!(VectorLiteStore::matches_filters(&mem, &f2));

        f2.max_importance = Some(0.5);
        assert!(!VectorLiteStore::matches_filters(&mem, &f2));
    }

    #[test]
    fn test_matches_filters_entities() {
        let meta = MemoryMetadata::new(MemoryType::Factual)
            .with_user_id("u1".to_string())
            .with_entities(vec!["Alice".into(), "Bob".into()]);
        let mem = Memory::with_content("test".into(), make_embedding(1.0), meta);

        let mut f = Filters::new();
        f.entities = Some(vec!["Alice".into()]);
        assert!(VectorLiteStore::matches_filters(&mem, &f));

        f.entities = Some(vec!["Charlie".into()]);
        assert!(!VectorLiteStore::matches_filters(&mem, &f));

        // At least one match
        f.entities = Some(vec!["Charlie".into(), "Bob".into()]);
        assert!(VectorLiteStore::matches_filters(&mem, &f));
    }

    #[test]
    fn test_matches_filters_topics() {
        let meta = MemoryMetadata::new(MemoryType::Factual)
            .with_user_id("u1".to_string())
            .with_topics(vec!["Rust".into(), "AI".into()]);
        let mem = Memory::with_content("test".into(), make_embedding(1.0), meta);

        let mut f = Filters::new();
        f.topics = Some(vec!["Rust".into()]);
        assert!(VectorLiteStore::matches_filters(&mem, &f));

        f.topics = Some(vec!["Python".into()]);
        assert!(!VectorLiteStore::matches_filters(&mem, &f));
    }

    #[test]
    fn test_matches_filters_date_range() {
        let mem = make_memory("test", "u1", MemoryType::Factual);

        let mut f = Filters::new();
        f.created_after = Some(chrono::Utc::now() - chrono::Duration::hours(1));
        assert!(VectorLiteStore::matches_filters(&mem, &f));

        f.created_after = Some(chrono::Utc::now() + chrono::Duration::hours(1));
        assert!(!VectorLiteStore::matches_filters(&mem, &f));
    }

    #[test]
    fn test_matches_filters_custom() {
        let mut meta = MemoryMetadata::new(MemoryType::Factual).with_user_id("u1".to_string());
        meta.custom.insert("key".to_string(), json!("value"));
        let mem = Memory::with_content("test".into(), make_embedding(1.0), meta);

        let mut f = Filters::new();
        f.custom.insert("key".to_string(), json!("value"));
        assert!(VectorLiteStore::matches_filters(&mem, &f));

        let mut f2 = Filters::new();
        f2.custom.insert("key".to_string(), json!("wrong"));
        assert!(!VectorLiteStore::matches_filters(&mem, &f2));
    }

    // --- Config helpers ---

    #[test]
    fn test_parse_index_type_hnsw() {
        let s = VectorLiteSettings {
            index_type: "hnsw".into(),
            metric: "cosine".into(),
            persistence_path: None,
        };
        assert!(matches!(parse_index_type(&s), IndexType::HNSW));
    }

    #[test]
    fn test_parse_index_type_flat() {
        let s = VectorLiteSettings {
            index_type: "flat".into(),
            metric: "cosine".into(),
            persistence_path: None,
        };
        assert!(matches!(parse_index_type(&s), IndexType::Flat));
    }

    #[test]
    fn test_parse_index_type_unknown_defaults_hnsw() {
        let s = VectorLiteSettings {
            index_type: "unknown".into(),
            metric: "cosine".into(),
            persistence_path: None,
        };
        assert!(matches!(parse_index_type(&s), IndexType::HNSW));
    }

    #[test]
    fn test_parse_metric_cosine() {
        let s = VectorLiteSettings {
            index_type: "hnsw".into(),
            metric: "cosine".into(),
            persistence_path: None,
        };
        assert!(matches!(parse_metric(&s), SimilarityMetric::Cosine));
    }

    #[test]
    fn test_parse_metric_euclidean() {
        let s = VectorLiteSettings {
            index_type: "hnsw".into(),
            metric: "euclidean".into(),
            persistence_path: None,
        };
        assert!(matches!(parse_metric(&s), SimilarityMetric::Euclidean));
    }

    #[test]
    fn test_parse_metric_manhattan() {
        let s = VectorLiteSettings {
            index_type: "hnsw".into(),
            metric: "manhattan".into(),
            persistence_path: None,
        };
        assert!(matches!(parse_metric(&s), SimilarityMetric::Manhattan));
    }

    #[test]
    fn test_parse_metric_dotproduct() {
        let s = VectorLiteSettings {
            index_type: "hnsw".into(),
            metric: "dotproduct".into(),
            persistence_path: None,
        };
        assert!(matches!(parse_metric(&s), SimilarityMetric::DotProduct));
    }

    #[test]
    fn test_parse_metric_unknown_defaults_cosine() {
        let s = VectorLiteSettings {
            index_type: "hnsw".into(),
            metric: "unknown".into(),
            persistence_path: None,
        };
        assert!(matches!(parse_metric(&s), SimilarityMetric::Cosine));
    }

    #[test]
    fn test_vectorlite_config_default() {
        let cfg = VectorLiteConfig::default();
        assert_eq!(cfg.collection_name, "llm-memories");
        assert!(matches!(cfg.index_type, IndexType::HNSW));
        assert!(matches!(cfg.metric, SimilarityMetric::Cosine));
        assert!(cfg.persistence_path.is_none());
    }

    #[test]
    fn test_vectorlite_config_from_store_config() {
        let store_cfg = crate::config::VectorStoreConfig {
            store_type: crate::config::VectorStoreType::Vectorlite,
            collection_name: "my-col".into(),
            vectorlite: crate::config::VectorLiteSettings {
                index_type: "flat".into(),
                metric: "euclidean".into(),
                persistence_path: Some("/data/vectors".into()),
            },
            banks_dir: "test-banks".into(),
        };
        let cfg = VectorLiteConfig::from_store_config(&store_cfg);
        assert_eq!(cfg.collection_name, "my-col");
        assert!(matches!(cfg.index_type, IndexType::Flat));
        assert!(matches!(cfg.metric, SimilarityMetric::Euclidean));
        assert_eq!(cfg.persistence_path, Some(PathBuf::from("/data/vectors")));
    }

    // --- Level 2: Relation filter tests ---

    #[tokio::test]
    async fn test_matches_filters_relations() {
        let mut meta = MemoryMetadata::new(MemoryType::Factual).with_user_id("u1".to_string());
        meta.relations = vec![
            crate::types::Relation {
                source: "SELF".into(),
                relation: "LIKES".into(),
                target: "Pizza".into(),
                strength: None,
            },
            crate::types::Relation {
                source: "SELF".into(),
                relation: "LIVES_IN".into(),
                target: "Rome".into(),
                strength: None,
            },
        ];
        let mem = Memory::with_content("test".into(), make_embedding(1.0), meta);

        // Match on relation predicate
        let mut f = Filters::new();
        f.relations = Some(vec![crate::types::RelationFilter {
            relation: "LIKES".into(),
            target: "Pizza".into(),
        }]);
        assert!(VectorLiteStore::matches_filters(&mem, &f));

        // Case-insensitive match
        f.relations = Some(vec![crate::types::RelationFilter {
            relation: "likes".into(),
            target: "pizza".into(),
        }]);
        assert!(VectorLiteStore::matches_filters(&mem, &f));

        // Empty relation means "any predicate" — match on target only
        f.relations = Some(vec![crate::types::RelationFilter {
            relation: "".into(),
            target: "Rome".into(),
        }]);
        assert!(VectorLiteStore::matches_filters(&mem, &f));

        // No match
        f.relations = Some(vec![crate::types::RelationFilter {
            relation: "HATES".into(),
            target: "Pizza".into(),
        }]);
        assert!(!VectorLiteStore::matches_filters(&mem, &f));
    }

    #[tokio::test]
    async fn test_store_memory_with_relations() {
        let store = make_store();
        let mut meta = MemoryMetadata::new(MemoryType::Factual).with_user_id("u1".to_string());
        meta.relations = vec![crate::types::Relation {
            source: "SELF".into(),
            relation: "KNOWS".into(),
            target: "Alice".into(),
            strength: None,
        }];
        let mem = Memory::with_content("Bob knows Alice".into(), make_embedding(1.0), meta);
        let id = mem.id.clone();

        store.insert(&mem).await.unwrap();
        let retrieved = store.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.metadata.relations.len(), 1);
        assert_eq!(retrieved.metadata.relations[0].target, "Alice");
    }

    // --- Level 3: Multi-vector, context, candidate_ids tests ---

    #[tokio::test]
    async fn test_multi_vector_insert_with_context_embeddings() {
        let store = make_store();
        let mut meta = MemoryMetadata::new(MemoryType::Factual).with_user_id("u1".to_string());
        meta.context = vec!["recipe".into(), "italian".into()];

        let mut mem = Memory::with_content("How to make pasta".into(), make_embedding(1.0), meta);
        mem.context_embeddings = Some(vec![
            make_embedding(2.0), // embedding for "recipe"
            make_embedding(3.0), // embedding for "italian"
        ]);

        let id = mem.id.clone();
        store.insert(&mem).await.unwrap();

        // Basic retrieval should still work
        let retrieved = store.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.content, Some("How to make pasta".to_string()));
        assert_eq!(retrieved.metadata.context, vec!["recipe", "italian"]);

        // id_map should have 3 vector IDs: 1 content + 2 context
        let id_map = store.id_map.read().unwrap();
        let vector_ids = id_map.get(&id).unwrap();
        assert_eq!(vector_ids.len(), 3);
    }

    #[tokio::test]
    async fn test_multi_vector_insert_with_relation_embeddings() {
        let store = make_store();
        let mut meta = MemoryMetadata::new(MemoryType::Factual).with_user_id("u1".to_string());
        meta.relations = vec![crate::types::Relation {
            source: "SELF".into(),
            relation: "LIKES".into(),
            target: "Pizza".into(),
            strength: None,
        }];

        let mut mem = Memory::with_content("User likes pizza".into(), make_embedding(1.0), meta);
        mem.relation_embeddings = Some(vec![make_embedding(4.0)]);

        let id = mem.id.clone();
        store.insert(&mem).await.unwrap();

        // id_map should have 2 vector IDs: 1 content + 1 relation
        let id_map = store.id_map.read().unwrap();
        let vector_ids = id_map.get(&id).unwrap();
        assert_eq!(vector_ids.len(), 2);
    }

    #[tokio::test]
    async fn test_multi_vector_insert_content_context_relations() {
        let store = make_store();
        let mut meta = MemoryMetadata::new(MemoryType::Factual).with_user_id("u1".to_string());
        meta.context = vec!["cooking".into()];
        meta.relations = vec![
            crate::types::Relation {
                source: "SELF".into(),
                relation: "USES".into(),
                target: "Tomato".into(),
                strength: None,
            },
            crate::types::Relation {
                source: "SELF".into(),
                relation: "USES".into(),
                target: "Basil".into(),
                strength: None,
            },
        ];

        let mut mem = Memory::with_content(
            "Pasta recipe uses tomato and basil".into(),
            make_embedding(1.0),
            meta,
        );
        mem.context_embeddings = Some(vec![make_embedding(2.0)]);
        mem.relation_embeddings = Some(vec![make_embedding(3.0), make_embedding(4.0)]);

        let id = mem.id.clone();
        store.insert(&mem).await.unwrap();

        // 1 content + 1 context + 2 relation = 4
        let id_map = store.id_map.read().unwrap();
        let vector_ids = id_map.get(&id).unwrap();
        assert_eq!(vector_ids.len(), 4);
    }

    #[tokio::test]
    async fn test_multi_vector_delete_cleans_all_vectors() {
        let store = make_store();
        let mut meta = MemoryMetadata::new(MemoryType::Factual).with_user_id("u1".to_string());
        meta.context = vec!["ctx1".into(), "ctx2".into()];

        let mut mem = Memory::with_content("content".into(), make_embedding(1.0), meta);
        mem.context_embeddings = Some(vec![make_embedding(2.0), make_embedding(3.0)]);
        let id = mem.id.clone();

        store.insert(&mem).await.unwrap();

        // Verify 3 vector IDs exist
        assert_eq!(store.id_map.read().unwrap().get(&id).unwrap().len(), 3);
        assert_eq!(store.reverse_id_map.read().unwrap().len(), 3);

        // Delete
        store.delete(&id).await.unwrap();

        // All traces should be gone
        assert!(store.id_map.read().unwrap().get(&id).is_none());
        assert!(store.memory_index.read().unwrap().get(&id).is_none());
        assert_eq!(store.reverse_id_map.read().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_multi_vector_update_replaces_all() {
        let store = make_store();
        let mut meta = MemoryMetadata::new(MemoryType::Factual).with_user_id("u1".to_string());
        meta.context = vec!["old-ctx".into()];

        let mut mem = Memory::with_content("old content".into(), make_embedding(1.0), meta);
        mem.context_embeddings = Some(vec![make_embedding(2.0)]);
        let id = mem.id.clone();

        store.insert(&mem).await.unwrap();
        assert_eq!(store.id_map.read().unwrap().get(&id).unwrap().len(), 2);

        // Update with new content and 3 context vectors
        let mut new_meta = MemoryMetadata::new(MemoryType::Factual).with_user_id("u1".to_string());
        new_meta.context = vec!["new1".into(), "new2".into(), "new3".into()];

        let mut updated = Memory::with_content("new content".into(), make_embedding(5.0), new_meta);
        updated.id = id.clone(); // Keep same ID
        updated.context_embeddings = Some(vec![
            make_embedding(6.0),
            make_embedding(7.0),
            make_embedding(8.0),
        ]);

        store.update(&updated).await.unwrap();

        // Should now have 4 vectors (1 content + 3 context)
        assert_eq!(store.id_map.read().unwrap().get(&id).unwrap().len(), 4);
        let retrieved = store.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.content, Some("new content".to_string()));
    }

    #[test]
    fn test_matches_filters_candidate_ids() {
        let mem = make_memory("test", "u1", MemoryType::Factual);

        // No candidate_ids filter — should pass
        let f = Filters::new();
        assert!(VectorLiteStore::matches_filters(&mem, &f));

        // Memory ID IS in candidate_ids — should pass
        let mut f_in = Filters::new();
        f_in.candidate_ids = Some(vec![mem.id.clone(), "other-id".into()]);
        assert!(VectorLiteStore::matches_filters(&mem, &f_in));

        // Memory ID NOT in candidate_ids — should fail
        let mut f_out = Filters::new();
        f_out.candidate_ids = Some(vec!["different-id".into()]);
        assert!(!VectorLiteStore::matches_filters(&mem, &f_out));

        // Empty candidate_ids — should fail (no candidates)
        let mut f_empty = Filters::new();
        f_empty.candidate_ids = Some(vec![]);
        assert!(!VectorLiteStore::matches_filters(&mem, &f_empty));
    }

    #[tokio::test]
    async fn test_search_with_candidate_ids_filter() {
        let store = make_store();

        let m1 = make_memory("alpha content", "u1", MemoryType::Factual);
        let m2 = make_memory("beta content", "u1", MemoryType::Factual);
        let m3 = make_memory("gamma content", "u1", MemoryType::Factual);

        let id1 = m1.id.clone();
        let id2 = m2.id.clone();

        store.insert(&m1).await.unwrap();
        store.insert(&m2).await.unwrap();
        store.insert(&m3).await.unwrap();

        // Search with candidate_ids restricting to m1 and m2 only
        let mut filters = Filters::new();
        filters.candidate_ids = Some(vec![id1.clone(), id2.clone()]);

        let results = store
            .search(&make_embedding(1.0), &filters, 10)
            .await
            .unwrap();

        // All results should be from the candidate set
        for r in &results {
            assert!(
                r.memory.id == id1 || r.memory.id == id2,
                "Result {} should be in candidate set",
                r.memory.id
            );
        }
    }

    #[tokio::test]
    async fn test_context_vector_search_finds_memory() {
        let store = make_store();
        let mut meta = MemoryMetadata::new(MemoryType::Factual).with_user_id("u1".to_string());
        meta.context = vec!["cooking".into()];

        // Use a distinct embedding for the "cooking" context
        let cooking_emb = make_embedding(5.0);
        let mut mem = Memory::with_content("How to make pasta".into(), make_embedding(1.0), meta);
        mem.context_embeddings = Some(vec![cooking_emb.clone()]);

        let id = mem.id.clone();
        store.insert(&mem).await.unwrap();

        // Search with the cooking context embedding — should find the memory
        let results = store
            .search(&cooking_emb, &Filters::new(), 10)
            .await
            .unwrap();

        assert!(!results.is_empty());
        // The result should map back to our memory
        assert!(results.iter().any(|r| r.memory.id == id));
    }
}
