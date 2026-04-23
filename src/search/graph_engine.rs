//! Graph traversal engine for following memory relations

use crate::types::{Memory, Relation};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

/// Type alias for the relation index to reduce type complexity
type RelationIndex = HashMap<String, Vec<(String, String, Option<f32>)>>;

/// Errors during graph traversal
#[derive(Debug, Error)]
pub enum GraphTraversalError {
    #[error("Max depth exceeded: {0}")]
    MaxDepthExceeded(usize),

    #[error("Memory not found: {0}")]
    MemoryNotFound(String),

    #[error("Invalid traversal config: {0}")]
    InvalidConfig(String),
}

/// Traversal strategy for graph exploration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TraversalStrategy {
    /// Breadth-first: explore all neighbors at each depth before going deeper
    /// Better for "related content" discovery and more predictable performance
    #[default]
    BFS,
}

/// Direction of relation traversal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TraversalDirection {
    /// Follow relations FROM current node (outgoing edges)
    Outgoing,

    /// Follow relations TO current node (incoming edges)
    Incoming,

    /// Bidirectional traversal (both incoming and outgoing)
    #[default]
    Both,
}

/// Configuration for graph traversal
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// Maximum traversal depth (default: 2, max: 5)
    pub max_depth: usize,

    /// Traversal strategy (default: BFS)
    pub strategy: TraversalStrategy,

    /// Direction of traversal (default: Both)
    pub direction: TraversalDirection,

    /// Optional filter by relation types (e.g., ["derived_from", "mentions"])
    pub relation_types: Option<Vec<String>>,

    /// Minimum relation strength for filtering (future: weighted relations)
    pub min_relation_strength: Option<f32>,

    /// Maximum number of entry points from semantic search (default: 5)
    pub entry_point_limit: usize,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            max_depth: 2,
            strategy: TraversalStrategy::BFS,
            direction: TraversalDirection::Both,
            relation_types: None,
            min_relation_strength: None,
            entry_point_limit: 5,
        }
    }
}

impl TraversalConfig {
    /// Create default traversal config
    pub fn new() -> Self {
        Self::default()
    }

    /// Create config with custom max_depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth.min(5); // Cap at 5 for safety
        self
    }

    /// Create config with custom direction
    pub fn with_direction(mut self, direction: TraversalDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Create config with relation type filter
    pub fn with_relation_types(mut self, types: Vec<String>) -> Self {
        self.relation_types = Some(types);
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), GraphTraversalError> {
        if self.max_depth == 0 {
            return Err(GraphTraversalError::InvalidConfig(
                "max_depth must be at least 1".to_string(),
            ));
        }
        if self.max_depth > 5 {
            return Err(GraphTraversalError::InvalidConfig(
                "max_depth cannot exceed 5 for performance reasons".to_string(),
            ));
        }
        Ok(())
    }
}

/// Graph search result with ranking information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSearchResult {
    /// The memory found
    pub memory: Memory,

    /// Distance from entry point (number of hops)
    pub entry_distance: usize,

    /// Path from entry point to this memory
    pub path_from_entry: Vec<RelationHop>,

    /// Relation boost applied to score
    pub relation_boost: f32,

    /// Final combined score
    pub final_score: f32,

    /// Original semantic similarity score
    pub semantic_score: f32,
}

/// A single hop in the traversal path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationHop {
    /// Source memory ID
    pub from: String,

    /// Relation type
    pub relation: String,

    /// Target memory ID or entity
    pub to: String,

    /// Strength of this relation (if available)
    pub strength: Option<f32>,
}

/// Graph search engine that traverses memory relations
pub struct GraphSearchEngine {
    _config: TraversalConfig,
}

impl GraphSearchEngine {
    /// Create a new graph search engine
    pub fn new(config: TraversalConfig) -> Result<Self, GraphTraversalError> {
        config.validate()?;
        Ok(Self { _config: config })
    }

    /// Traverse graph from entry points using BFS
    ///
    /// # Arguments
    /// * `entry_memories` - Starting memories from semantic search
    /// * `all_memories` - All memories in the bank for relation lookup
    /// * `config` - Traversal configuration
    ///
    /// # Returns
    /// Ranked memories discovered through graph traversal
    pub async fn traverse(
        &self,
        entry_memories: Vec<(Memory, f32)>, // (memory, semantic_score)
        all_memories: &[Memory],
        config: &TraversalConfig,
    ) -> Result<Vec<GraphSearchResult>, GraphTraversalError> {
        if entry_memories.is_empty() {
            return Ok(Vec::new());
        }

        // Build a lookup map for fast memory access by ID
        let memory_map: HashMap<String, &Memory> =
            all_memories.iter().map(|m| (m.id.clone(), m)).collect();

        // Build an index of incoming relations for reverse lookup
        let incoming_index = self.build_incoming_index(all_memories);

        let mut results = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Initialize queue with entry points
        for (memory, semantic_score) in entry_memories {
            visited.insert(memory.id.clone());
            queue.push_back(TraversalState {
                memory_id: memory.id.clone(),
                depth: 0,
                path: Vec::new(),
                semantic_score,
                relation_boost: 0.0,
            });
        }

        // BFS traversal
        while let Some(state) = queue.pop_front() {
            // Add current memory to results
            if let Some(&memory) = memory_map.get(&state.memory_id) {
                let result = self.calculate_rank_score(
                    memory.clone(),
                    state.semantic_score,
                    state.relation_boost,
                    state.depth,
                    state.path.clone(),
                );
                results.push(result);
            }

            // Don't traverse deeper than max_depth
            if state.depth >= config.max_depth {
                continue;
            }

            // Get neighbors based on direction
            let neighbors = self.get_neighbors(
                &state.memory_id,
                config.direction,
                &memory_map,
                &incoming_index,
                config,
            );

            // Add unvisited neighbors to queue
            for (neighbor_id, relation, strength) in neighbors {
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id.clone());

                // Calculate relation boost
                let boost = self.calculate_relation_boost(&relation, strength, state.depth + 1);

                // Build path
                let mut new_path = state.path.clone();
                new_path.push(RelationHop {
                    from: state.memory_id.clone(),
                    relation: relation.clone(),
                    to: neighbor_id.clone(),
                    strength,
                });

                queue.push_back(TraversalState {
                    memory_id: neighbor_id,
                    depth: state.depth + 1,
                    path: new_path,
                    // Neighbor memories don't have semantic scores from entry query
                    // Use a base score for discovery
                    semantic_score: state.semantic_score * 0.8,
                    relation_boost: state.relation_boost + boost,
                });
            }
        }

        // Sort by final score
        results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Build an index of incoming relations for reverse lookup
    fn build_incoming_index(
        &self,
        memories: &[Memory],
    ) -> HashMap<String, Vec<(String, String, Option<f32>)>> {
        let mut index: HashMap<String, Vec<(String, String, Option<f32>)>> = HashMap::new();

        for memory in memories {
            for relation in &memory.metadata.relations {
                // If relation.target is a memory ID (starts with "mem-" or is a UUID format)
                // For now, we'll just index all relations
                // In production, you'd want to validate that target is actually a memory ID
                index.entry(relation.target.clone()).or_default().push((
                    memory.id.clone(),
                    relation.relation.clone(),
                    relation.strength,
                ));
            }
        }

        index
    }

    /// Get neighboring memory IDs based on traversal direction
    fn get_neighbors(
        &self,
        memory_id: &str,
        direction: TraversalDirection,
        memory_map: &HashMap<String, &Memory>,
        incoming_index: &RelationIndex,
        config: &TraversalConfig,
    ) -> Vec<(String, String, Option<f32>)> {
        let mut neighbors = Vec::new();

        match direction {
            TraversalDirection::Outgoing => {
                // Follow relations FROM this memory
                if let Some(memory) = memory_map.get(memory_id) {
                    for relation in &memory.metadata.relations {
                        if self.should_follow_relation(relation, config) {
                            // Only include if target looks like a memory ID
                            if self.is_memory_id(&relation.target) {
                                neighbors.push((
                                    relation.target.clone(),
                                    relation.relation.clone(),
                                    relation.strength,
                                ));
                            }
                        }
                    }
                }
            }
            TraversalDirection::Incoming => {
                // Follow relations TO this memory (reverse lookup)
                if let Some(incoming) = incoming_index.get(memory_id) {
                    for (source_id, relation, strength) in incoming {
                        if config
                            .relation_types
                            .as_ref()
                            .map(|types| types.contains(relation))
                            .unwrap_or(true)
                        {
                            neighbors.push((source_id.clone(), relation.clone(), *strength));
                        }
                    }
                }
            }
            TraversalDirection::Both => {
                // Combine both directions
                let outgoing = self.get_neighbors(
                    memory_id,
                    TraversalDirection::Outgoing,
                    memory_map,
                    incoming_index,
                    config,
                );
                let incoming = self.get_neighbors(
                    memory_id,
                    TraversalDirection::Incoming,
                    memory_map,
                    incoming_index,
                    config,
                );
                neighbors.extend(outgoing);
                neighbors.extend(incoming);
            }
        }

        neighbors
    }

    /// Check if a relation should be followed based on config
    fn should_follow_relation(&self, relation: &Relation, config: &TraversalConfig) -> bool {
        // Filter by relation type if specified
        if let Some(ref types) = config.relation_types
            && !types.contains(&relation.relation)
        {
            return false;
        }

        // Filter by strength if specified
        if let Some(min_strength) = config.min_relation_strength
            && let Some(strength) = relation.strength
            && strength < min_strength
        {
            return false;
        }

        true
    }

    /// Check if a string looks like a memory ID
    fn is_memory_id(&self, s: &str) -> bool {
        // Simple heuristic: memory IDs are UUIDs or start with "mem-"
        s.starts_with("mem-") || s.len() == 36 && s.chars().nth(8) == Some('-')
        // UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    }

    /// Calculate boost for a relation
    fn calculate_relation_boost(
        &self,
        _relation: &str,
        strength: Option<f32>,
        depth: usize,
    ) -> f32 {
        // Base boost decreases with depth
        let base_boost = 0.15 / (depth as f32);

        // Apply strength if available
        if let Some(s) = strength {
            base_boost * s
        } else {
            base_boost
        }
    }

    /// Calculate final rank score using multi-factor scoring
    fn calculate_rank_score(
        &self,
        memory: Memory,
        semantic_score: f32,
        relation_boost: f32,
        distance: usize,
        path: Vec<RelationHop>,
    ) -> GraphSearchResult {
        // Scoring formula from plan:
        // final_score = (semantic_score * 0.5) + (relation_boost * 0.3) + ((1.0 / (graph_distance + 1)) * 0.2)

        let distance_score = 1.0 / (distance as f32 + 1.0);
        let final_score = (semantic_score * 0.5) + (relation_boost * 0.3) + (distance_score * 0.2);

        GraphSearchResult {
            memory,
            entry_distance: distance,
            path_from_entry: path,
            relation_boost,
            final_score,
            semantic_score,
        }
    }

    /// Lightweight 1-hop graph refinement from a small set of entry memories.
    ///
    /// Unlike `traverse()`, this does NOT load all memories. It only fetches
    /// the specific neighbor IDs discovered from the entry memories' relations.
    ///
    /// # Arguments
    /// * `entry_memories` - Top memories from semantic/pyramid search with scores
    /// * `get_memory` - Async closure to fetch a memory by ID (O(1) per call)
    ///
    /// # Returns
    /// Newly discovered memories not already in the entry set
    pub async fn lightweight_refine<F, Fut>(
        &self,
        entry_memories: &[(Memory, f32)],
        get_memory: F,
    ) -> Vec<GraphSearchResult>
    where
        F: Fn(String) -> Fut + Send,
        Fut: std::future::Future<Output = Option<Memory>> + Send,
    {
        if entry_memories.is_empty() {
            return Vec::new();
        }

        // Build set of already-seen IDs
        let seen: HashSet<String> = entry_memories.iter().map(|(m, _)| m.id.clone()).collect();

        // Collect all neighbor IDs from entry memories (1-hop, both directions)
        let mut neighbor_ids: Vec<(String, String, String, Option<f32>, f32)> = Vec::new();
        // (neighbor_id, from_id, relation, strength, entry_semantic_score)

        for (entry_mem, entry_score) in entry_memories {
            // Outgoing relations
            for relation in &entry_mem.metadata.relations {
                if self.is_memory_id(&relation.target) && !seen.contains(&relation.target) {
                    neighbor_ids.push((
                        relation.target.clone(),
                        entry_mem.id.clone(),
                        relation.relation.clone(),
                        relation.strength,
                        *entry_score,
                    ));
                }
            }
        }

        if neighbor_ids.is_empty() {
            return Vec::new();
        }

        // Deduplicate neighbor IDs — keep first occurrence per neighbor
        let mut unique_neighbors: Vec<(String, String, String, Option<f32>, f32)> = Vec::new();
        let mut seen_ids: HashSet<String> = HashSet::new();
        for item in neighbor_ids {
            if seen_ids.insert(item.0.clone()) {
                unique_neighbors.push(item);
            }
        }

        // Fetch neighbor memories concurrently
        let mut futures = Vec::new();
        for (nid, _, _, _, _) in &unique_neighbors {
            let get_mem = &get_memory;
            futures.push(async move { (nid.clone(), get_mem(nid.clone()).await) });
        }

        let fetched: Vec<_> = futures::future::join_all(futures).await;

        // Build results for successfully fetched neighbors
        let mut results = Vec::new();
        for (nid, maybe_memory) in fetched {
            let memory = match maybe_memory {
                Some(m) => m,
                None => continue,
            };

            // Find the entry info for this neighbor
            let (from_id, relation, strength, entry_score) = unique_neighbors
                .iter()
                .find(|(id, _, _, _, _)| id == &nid)
                .map(|(_, from, rel, strg, sc)| (from.clone(), rel.clone(), *strg, *sc))
                .unwrap_or_else(|| (String::new(), String::new(), None, 0.5));

            let boost = self.calculate_relation_boost(&relation, strength, 1);
            let path = vec![RelationHop {
                from: from_id,
                relation,
                to: nid,
                strength,
            }];

            let result = self.calculate_rank_score(memory, entry_score * 0.8, boost, 1, path);
            results.push(result);
        }

        // Sort by final score descending
        results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }
}

/// Internal state for BFS traversal
struct TraversalState {
    memory_id: String,
    depth: usize,
    path: Vec<RelationHop>,
    semantic_score: f32,
    relation_boost: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Memory, MemoryMetadata, MemoryType};

    #[test]
    fn test_traversal_config_default() {
        let config = TraversalConfig::new();
        assert_eq!(config.max_depth, 2);
        assert_eq!(config.strategy, TraversalStrategy::BFS);
        assert_eq!(config.direction, TraversalDirection::Both);
        assert!(config.relation_types.is_none());
    }

    #[test]
    fn test_traversal_config_builder() {
        let config = TraversalConfig::new()
            .with_max_depth(3)
            .with_direction(TraversalDirection::Outgoing)
            .with_relation_types(vec!["derived_from".to_string()]);

        assert_eq!(config.max_depth, 3);
        assert_eq!(config.direction, TraversalDirection::Outgoing);
        assert_eq!(
            config.relation_types,
            Some(vec!["derived_from".to_string()])
        );
    }

    #[test]
    fn test_traversal_config_validation() {
        // Valid config
        let config = TraversalConfig::new();
        assert!(config.validate().is_ok());

        // Invalid: depth too high
        let config = TraversalConfig {
            max_depth: 10,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Invalid: depth zero
        let config = TraversalConfig {
            max_depth: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_is_memory_id() {
        let engine = GraphSearchEngine::new(TraversalConfig::default()).unwrap();

        assert!(engine.is_memory_id("mem-123"));
        assert!(engine.is_memory_id("550e8400-e29b-41d4-a716-446655440000"));
        assert!(!engine.is_memory_id("Alice"));
        assert!(!engine.is_memory_id("Pizza"));
    }

    #[test]
    fn test_calculate_relation_boost() {
        let engine = GraphSearchEngine::new(TraversalConfig::default()).unwrap();

        // No strength
        let boost = engine.calculate_relation_boost("derived_from", None, 1);
        assert!((boost - 0.15).abs() < f32::EPSILON);

        // With strength
        let boost = engine.calculate_relation_boost("derived_from", Some(0.8), 1);
        assert!((boost - 0.12).abs() < 0.001);

        // Decreases with depth
        let boost_depth2 = engine.calculate_relation_boost("derived_from", None, 2);
        assert!(boost_depth2 < boost);
    }

    #[test]
    fn test_calculate_rank_score() {
        let engine = GraphSearchEngine::new(TraversalConfig::default()).unwrap();

        let memory = Memory::with_content(
            "Test content".to_string(),
            vec![0.0; 384],
            MemoryMetadata::new(MemoryType::Factual),
        );

        let result = engine.calculate_rank_score(
            memory,
            0.8, // semantic_score
            0.1, // relation_boost
            1,   // distance
            vec![],
        );

        assert_eq!(result.semantic_score, 0.8);
        assert_eq!(result.relation_boost, 0.1);
        assert_eq!(result.entry_distance, 1);
        assert!(result.final_score > 0.0);
        assert!(result.final_score <= 1.0);
    }

    #[tokio::test]
    async fn test_lightweight_refine_empty_entry() {
        let engine = GraphSearchEngine::new(TraversalConfig::default()).unwrap();
        let entry: Vec<(Memory, f32)> = vec![];

        let results = engine
            .lightweight_refine(&entry, |_id| async { None })
            .await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_lightweight_refine_no_relations() {
        let engine = GraphSearchEngine::new(TraversalConfig::default()).unwrap();
        let memory = Memory::with_content(
            "Test".to_string(),
            vec![0.0; 384],
            MemoryMetadata::new(MemoryType::Factual),
        );
        let entry = vec![(memory, 0.9)];

        let results = engine
            .lightweight_refine(&entry, |_id| async { None })
            .await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_lightweight_refine_discovers_neighbor() {
        let engine = GraphSearchEngine::new(TraversalConfig::default()).unwrap();

        let neighbor_id = "550e8400-e29b-41d4-a716-446655440000".to_string();

        let mut entry_metadata = MemoryMetadata::new(MemoryType::Factual);
        entry_metadata.relations.push(Relation {
            source: "entry".to_string(),
            relation: "references".to_string(),
            target: neighbor_id.clone(),
            strength: Some(0.8),
        });
        let entry_mem =
            Memory::with_content("Entry content".to_string(), vec![0.0; 384], entry_metadata);

        let mut neighbor_metadata = MemoryMetadata::new(MemoryType::Factual);
        neighbor_metadata.hash = "hash".to_string();
        let mut neighbor_mem = Memory::with_content(
            "Neighbor content".to_string(),
            vec![0.0; 384],
            neighbor_metadata,
        );
        neighbor_mem.id = neighbor_id.clone();
        let neighbor_mem_clone = neighbor_mem.clone();
        let neighbor_id_clone = neighbor_id.clone();
        let entry = vec![(entry_mem, 0.9)];

        let results = engine
            .lightweight_refine(&entry, |id: String| {
                let nid = neighbor_id_clone.clone();
                let nm = neighbor_mem_clone.clone();
                async move { if id == nid { Some(nm) } else { None } }
            })
            .await;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory.id, neighbor_id);
        assert_eq!(results[0].entry_distance, 1);
        assert_eq!(results[0].path_from_entry.len(), 1);
        assert_eq!(results[0].path_from_entry[0].relation, "references");
    }

    #[tokio::test]
    async fn test_lightweight_refine_skips_non_memory_ids() {
        let engine = GraphSearchEngine::new(TraversalConfig::default()).unwrap();

        let mut entry_metadata = MemoryMetadata::new(MemoryType::Factual);
        entry_metadata.relations.push(Relation {
            source: "entry".to_string(),
            relation: "mentions".to_string(),
            target: "Alice".to_string(), // Not a memory ID
            strength: None,
        });
        let entry_mem = Memory::with_content("Entry".to_string(), vec![0.0; 384], entry_metadata);

        let entry = vec![(entry_mem, 0.9)];
        let results = engine
            .lightweight_refine(&entry, |_id| async { None })
            .await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_lightweight_refine_deduplicates_neighbors() {
        let engine = GraphSearchEngine::new(TraversalConfig::default()).unwrap();

        let shared_id = "550e8400-e29b-41d4-a716-446655440000".to_string();

        let mut meta1 = MemoryMetadata::new(MemoryType::Factual);
        meta1.relations.push(Relation {
            source: "e1".to_string(),
            relation: "ref".to_string(),
            target: shared_id.clone(),
            strength: None,
        });

        let mut meta2 = MemoryMetadata::new(MemoryType::Factual);
        meta2.relations.push(Relation {
            source: "e2".to_string(),
            relation: "ref".to_string(),
            target: shared_id.clone(),
            strength: None,
        });

        let e1 = Memory::with_content("E1".to_string(), vec![0.0; 384], meta1);
        let e2 = Memory::with_content("E2".to_string(), vec![0.0; 384], meta2);

        let mut neighbor_meta = MemoryMetadata::new(MemoryType::Factual);
        neighbor_meta.hash = "hash".to_string();
        let mut neighbor = Memory::with_content("N".to_string(), vec![0.0; 384], neighbor_meta);
        neighbor.id = shared_id.clone();

        let entry = vec![(e1, 0.9), (e2, 0.8)];
        let shared_id_clone = shared_id.clone();
        let neighbor_clone = neighbor.clone();

        let results = engine
            .lightweight_refine(&entry, |id: String| {
                let sid = shared_id_clone.clone();
                let n = neighbor_clone.clone();
                async move { if id == sid { Some(n) } else { None } }
            })
            .await;

        // Should only appear once despite two entry points referencing it
        assert_eq!(results.len(), 1);
    }
}
