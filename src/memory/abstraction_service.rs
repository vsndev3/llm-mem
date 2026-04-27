use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use uuid::Uuid;

use crate::{
    error::{MemoryError, Result},
    memory::search_service::SearchService,
    types::{Filters, Memory, MemoryState, NavigateResult},
    vector_store::VectorStore,
};

/// Result of a cascade delete operation.
#[derive(Debug, Clone)]
pub struct DeletionResult {
    pub deleted_id: String,
    pub forgotten: Vec<DegradedMemory>,
    pub degraded: Vec<DegradedMemory>,
    pub cascade_depth: i32,
}

/// A memory that was degraded or forgotten during cascade.
#[derive(Debug, Clone)]
pub struct DegradedMemory {
    pub id: String,
    pub layer: i32,
    pub degradation: f64,
    pub total_sources: usize,
    pub deleted_sources: usize,
}

/// Owns abstraction hierarchy operations: cascade deletion, layer navigation,
/// abstraction dependency tracking, and layer manifest management.
///
/// Extracted from MemoryManager to reduce its god-object responsibilities.
pub struct AbstractionService {
    vector_store: Box<dyn VectorStore + Send + Sync>,
    #[allow(dead_code)]
    search: Arc<SearchService>,
}

impl AbstractionService {
    /// Static version of forgotten_threshold for use in tests
    pub fn forgotten_threshold_static(layer_level: i32) -> f64 {
        Self::forgotten_threshold(layer_level)
    }

    pub fn new(
        vector_store: Box<dyn VectorStore + Send + Sync>,
        search: Arc<SearchService>,
    ) -> Self {
        Self { vector_store, search }
    }

    /// Find all memories that abstract from or link to this memory (reverse direction).
    pub async fn find_abstraction_dependents(&self, memory_id: &str) -> Result<Vec<Memory>> {
        let parsed_id = match Uuid::parse_str(memory_id) {
            Ok(id) => id,
            Err(_) => return Ok(vec![]),
        };

        let mut filters = Filters::new();
        filters.contains_abstraction_source = Some(parsed_id);
        self.vector_store.list(&filters, None).await
    }

    /// Navigate the abstraction hierarchy from a memory node.
    pub async fn navigate_memory(
        &self,
        memory_id: &str,
        direction: &str,
        levels: usize,
    ) -> Result<NavigateResult> {
        let memory = self
            .get(memory_id)
            .await?
            .ok_or_else(|| MemoryError::NotFound { id: memory_id.to_string() })?;

        let mut result = NavigateResult {
            source_memory_id: memory_id.to_string(),
            source_layer: memory.metadata.layer.level,
            zoom_in: Vec::new(),
            zoom_out: Vec::new(),
        };

        if direction == "zoom_in" || direction == "both" {
            result.zoom_in = self.trace_sources(&memory, levels).await?;
        }

        if direction == "zoom_out" || direction == "both" {
            result.zoom_out = self.find_abstraction_dependents(memory_id).await?;
        }

        Ok(result)
    }

    /// Recursively trace abstraction_sources to find lower-layer memories.
    fn trace_sources<'a>(
        &'a self,
        memory: &'a Memory,
        levels: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Memory>>> + Send + 'a>> {
        Box::pin(async move {
            if levels == 0 {
                return Ok(vec![]);
            }

            let mut sources = Vec::new();
            for src_id in &memory.metadata.abstraction_sources {
                if let Ok(Some(mem)) = self.get(&src_id.to_string()).await {
                    if levels == 1 {
                        sources.push(mem);
                    } else {
                        let deeper = self.trace_sources(&mem, levels - 1).await?;
                        sources.extend(deeper);
                    }
                }
            }

            Ok(sources)
        })
    }

    /// Mark a memory as forgotten (unconditional, for direct use)
    pub async fn mark_as_forgotten(&self, memory_id: &str, deleted_by: &str) -> Result<()> {
        let mut memory = self
            .get(memory_id)
            .await?
            .ok_or_else(|| MemoryError::NotFound { id: memory_id.to_string() })?;

        memory.metadata.state = MemoryState::Forgotten;
        memory.metadata.forgotten_at = Some(chrono::Utc::now());

        if let Ok(uuid_val) = Uuid::parse_str(deleted_by) {
            memory.metadata.forgotten_by = Some(uuid_val);
        }

        self.vector_store.update(&memory).await?;
        Ok(())
    }

    /// Delete a memory with threshold-based cascade degradation for layers.
    pub async fn delete_with_cascade(&self, memory_id: &str) -> Result<DeletionResult> {
        let memory = self
            .get(memory_id)
            .await?
            .ok_or_else(|| MemoryError::NotFound { id: memory_id.to_string() })?;

        let mut result = DeletionResult {
            deleted_id: memory_id.to_string(),
            forgotten: Vec::new(),
            degraded: Vec::new(),
            cascade_depth: 0,
        };

        let deleted_uuid = Uuid::parse_str(&memory.id)
            .map_err(|e| MemoryError::Validation(format!("Invalid memory ID: {}", e)))?;

        // BFS queue: (dependent_memory_id, deleted_source_uuid)
        let mut queue = VecDeque::new();
        let mut visited: HashSet<(String, Uuid)> = HashSet::new();

        let dependents = self.find_abstraction_dependents(&memory.id).await?;
        for dep in dependents {
            if dep.metadata.layer.level > memory.metadata.layer.level {
                let entry = (dep.id, deleted_uuid);
                if visited.insert(entry.clone()) {
                    queue.push_back(entry);
                }
            }
        }

        while let Some((dep_id, source_uuid)) = queue.pop_front() {
            let dep = match self.get(&dep_id).await? {
                Some(m) => m,
                None => continue,
            };

            let total_sources = dep.metadata.abstraction_sources.len();
            if total_sources == 0 {
                continue;
            }

            let (degraded_memory, became_forgotten) =
                self.apply_degradation(&dep, source_uuid, &mut result).await?;

            if became_forgotten {
                let dep_uuid = Uuid::parse_str(&dep.id).ok();
                if let Some(uuid) = dep_uuid {
                    let sub_dependents = self.find_abstraction_dependents(&dep.id).await?;
                    for sub_dep in sub_dependents {
                        if sub_dep.metadata.layer.level > dep.metadata.layer.level {
                            let entry = (sub_dep.id, uuid);
                            if visited.insert(entry.clone()) {
                                queue.push_back(entry);
                            }
                        }
                    }
                }
            }

            self.vector_store.update(&degraded_memory).await?;
        }

        self.vector_store.delete(memory_id).await?;
        tracing::info!(
            "Deleted {} with cascade: {} forgotten, {} degraded across {} layers",
            memory_id,
            result.forgotten.len(),
            result.degraded.len(),
            result.cascade_depth
        );

        Ok(result)
    }

    async fn apply_degradation(
        &self,
        dependent: &Memory,
        deleted_source_uuid: Uuid,
        result: &mut DeletionResult,
    ) -> Result<(Memory, bool)> {
        let total_sources = dependent.metadata.abstraction_sources.len();
        if total_sources == 0 {
            return Ok((dependent.clone(), false));
        }

        let mut updated = dependent.clone();
        if !updated.metadata.forgotten_sources.contains(&deleted_source_uuid) {
            updated.metadata.forgotten_sources.push(deleted_source_uuid);
        }
        let deleted_count = updated.metadata.forgotten_sources.len();
        let degradation = deleted_count as f64 / total_sources as f64;

        let threshold = Self::forgotten_threshold(updated.metadata.layer.level);
        let was_forgotten = dependent.metadata.state.is_forgotten();

        if degradation >= threshold {
            updated.metadata.state = MemoryState::Forgotten;
            if updated.metadata.forgotten_at.is_none() {
                updated.metadata.forgotten_at = Some(chrono::Utc::now());
            }
            updated.metadata.forgotten_by = Some(deleted_source_uuid);
            updated.updated_at = chrono::Utc::now();

            result.forgotten.push(DegradedMemory {
                id: dependent.id.clone(),
                layer: dependent.metadata.layer.level,
                degradation,
                total_sources,
                deleted_sources: deleted_count,
            });
            result.cascade_depth = std::cmp::max(result.cascade_depth, dependent.metadata.layer.level);

            Ok((updated, !was_forgotten))
        } else {
            updated.metadata.state = MemoryState::Degraded;
            updated.updated_at = chrono::Utc::now();

            result.degraded.push(DegradedMemory {
                id: dependent.id.clone(),
                layer: dependent.metadata.layer.level,
                degradation,
                total_sources,
                deleted_sources: deleted_count,
            });
            result.cascade_depth = std::cmp::max(result.cascade_depth, dependent.metadata.layer.level);

            Ok((updated, false))
        }
    }

    fn forgotten_threshold(layer_level: i32) -> f64 {
        match layer_level {
            1 => 1.0,
            2 => 0.51,
            _ => 0.67,
        }
    }

    async fn get(&self, id: &str) -> Result<Option<Memory>> {
        self.vector_store.get(id).await
    }
}