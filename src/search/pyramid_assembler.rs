//! Hierarchical pyramid search assembler
//!
//! Allocates result slots across abstraction layers, normalizes scores,
//! deduplicates across layers, and produces a pyramid-shaped result set.

use crate::types::ScoredMemory;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Allocation strategy for distributing result slots across layers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PyramidAllocationMode {
    /// More concrete L0 facts, fewer abstract L3+ concepts (40/25/20/15)
    #[default]
    BottomHeavy,
    /// Equal distribution across layers (25/25/25/25)
    Balanced,
    /// More abstract concepts, fewer concrete facts (15/20/25/40)
    TopHeavy,
    /// LLM classifies query intent, then picks mode dynamically
    Dynamic,
}

/// Result of pyramid assembly with per-result metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyramidResult {
    pub memory: ScoredMemory,
    pub layer: i32,
    pub layer_name: String,
    pub search_phase: String,
    pub graph_path: Option<Vec<crate::search::RelationHop>>,
}

/// Per-layer search results keyed by layer level
pub type LayerResults = HashMap<i32, Vec<ScoredMemory>>;

/// Configuration for pyramid search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyramidConfig {
    /// Allocation mode (default: BottomHeavy)
    pub mode: PyramidAllocationMode,
    /// Custom per-layer weight overrides (layer_level -> weight)
    #[serde(default)]
    pub layer_weights: HashMap<String, f32>,
}

impl Default for PyramidConfig {
    fn default() -> Self {
        Self {
            mode: PyramidAllocationMode::BottomHeavy,
            layer_weights: HashMap::new(),
        }
    }
}

/// Core pyramid assembler
pub struct PyramidAssembler;

impl PyramidAssembler {
    /// Assemble layer-parallel search results into a pyramid-shaped output.
    ///
    /// 1. Normalize scores per layer to [0, 1]
    /// 2. Allocate slots per layer according to mode
    /// 3. Pick top-N from each layer
    /// 4. Cross-layer deduplication via abstraction_sources
    /// 5. Return sorted results
    pub fn assemble(
        layer_results: LayerResults,
        total_limit: usize,
        mode: PyramidAllocationMode,
        layer_weights: HashMap<i32, f32>,
    ) -> Vec<PyramidResult> {
        if layer_results.is_empty() {
            return Vec::new();
        }

        let active_layers: Vec<i32> = layer_results.keys().copied().filter(|&l| l >= 0).collect();
        if active_layers.is_empty() {
            return Vec::new();
        }

        // Step 1: Compute per-layer slot allocation
        let mut allocations =
            Self::compute_allocations(&active_layers, total_limit, mode, &layer_weights);

        // Step 2: Redistribute slots for empty layers
        let mut did_redistribute = true;
        while did_redistribute {
            did_redistribute = false;
            // Collect orphaned slots first (breaks the double-borrow)
            let mut orphans: Vec<(i32, usize)> = Vec::new();
            for (&layer, &slots) in &allocations {
                let count = layer_results.get(&layer).map(|v| v.len()).unwrap_or(0);
                if count == 0 && slots > 0 {
                    orphans.push((layer, slots));
                }
            }
            // Apply zeroing and redistribution
            for (layer, orphaned) in orphans {
                *allocations.get_mut(&layer).unwrap() = 0;
                did_redistribute = true;
                Self::redistribute_slots(&active_layers, &mut allocations, orphaned);
            }
        }

        // Step 3: Normalize scores per layer and pick top-N
        let mut picked: Vec<PyramidResult> = Vec::new();

        for (&layer, slots) in &allocations {
            if *slots == 0 {
                continue;
            }

            let mut results = layer_results.get(&layer).cloned().unwrap_or_default();

            // Normalize scores to [0, 1] within this layer
            Self::normalize_scores(&mut results);

            // Sort by normalized score descending
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Pick top-N for this layer (clamped to available results)
            let take = (*slots).min(results.len());
            let taken = results.drain(..take).collect::<Vec<_>>();

            for scored in taken {
                let layer_name = scored.memory.metadata.layer.name_or_default();
                picked.push(PyramidResult {
                    memory: scored,
                    layer,
                    layer_name,
                    search_phase: "pyramid".to_string(),
                    graph_path: None,
                });
            }
        }

        // Step 4: Cross-layer deduplication via abstraction_sources
        picked = Self::deduplicate_across_layers(picked);

        // Step 5: Final sort by score descending
        picked.sort_by(|a, b| {
            b.memory
                .score
                .partial_cmp(&a.memory.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to total_limit
        picked.truncate(total_limit);

        picked
    }

    /// Compute slot allocation per layer based on mode.
    fn compute_allocations(
        active_layers: &[i32],
        total_limit: usize,
        mode: PyramidAllocationMode,
        custom_weights: &HashMap<i32, f32>,
    ) -> HashMap<i32, usize> {
        let weights: HashMap<i32, f32> = if !custom_weights.is_empty() {
            custom_weights.clone()
        } else {
            match mode {
                PyramidAllocationMode::BottomHeavy => {
                    // Lower layers get more weight
                    active_layers
                        .iter()
                        .map(|&l| {
                            let max_l = *active_layers.iter().max().unwrap_or(&3) as f32;
                            let w = if max_l > 0.0 {
                                1.0 + (max_l - l as f32) / max_l * 1.5
                            } else {
                                1.0
                            };
                            (l, w)
                        })
                        .collect()
                }
                PyramidAllocationMode::Balanced => {
                    active_layers.iter().map(|&l| (l, 1.0f32)).collect()
                }
                PyramidAllocationMode::TopHeavy => {
                    // Higher layers get more weight
                    active_layers
                        .iter()
                        .map(|&l| {
                            let max_l = *active_layers.iter().max().unwrap_or(&3) as f32;
                            let w = if max_l > 0.0 {
                                1.0 + l as f32 / max_l * 1.5
                            } else {
                                1.0
                            };
                            (l, w)
                        })
                        .collect()
                }
                PyramidAllocationMode::Dynamic => {
                    // Dynamic mode: caller should have resolved to a concrete mode
                    // Fall back to balanced
                    active_layers.iter().map(|&l| (l, 1.0f32)).collect()
                }
            }
        };

        // Normalize weights to sum to total_limit
        let total_weight: f32 = weights.values().sum();
        if total_weight <= 0.0 {
            let equal = total_limit / active_layers.len().max(1);
            return active_layers.iter().map(|&l| (l, equal)).collect();
        }

        let mut allocations: HashMap<i32, usize> = HashMap::new();
        let mut assigned = 0;

        // Assign floor values first
        for (&layer, &weight) in &weights {
            let slots = ((weight / total_weight) * total_limit as f32).floor() as usize;
            allocations.insert(layer, slots);
            assigned += slots;
        }

        // Distribute remainder by fractional part
        let mut remainders: Vec<(i32, f32)> = weights
            .iter()
            .map(|(&l, &w)| {
                let slots = ((w / total_weight) * total_limit as f32).floor() as usize;
                (l, (w / total_weight) * total_limit as f32 - slots as f32)
            })
            .collect();
        remainders.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let remainder = total_limit - assigned;
        for i in 0..remainder {
            if let Some(&(layer, _)) = remainders.get(i) {
                *allocations.get_mut(&layer).unwrap() += 1;
            }
        }

        allocations
    }

    /// Redistribute orphaned slots to layers with remaining capacity, weighted.
    fn redistribute_slots(
        active_layers: &[i32],
        allocations: &mut HashMap<i32, usize>,
        orphaned: usize,
    ) {
        // Give to layers that have results but haven't hit their capacity
        let mut candidates: Vec<(i32, f32)> = Vec::new();
        for &layer in active_layers {
            let current = allocations.get(&layer).copied().unwrap_or(0) as f32;
            // Weight by current allocation (proportional growth)
            let weight = current.max(1.0);
            candidates.push((layer, weight));
        }

        let total_w: f32 = candidates.iter().map(|&(_, w)| w).sum();
        if total_w <= 0.0 || candidates.is_empty() {
            return;
        }

        let mut distributed = 0;
        let mut adds: HashMap<i32, usize> = HashMap::new();

        for &(layer, w) in &candidates {
            let add = ((w / total_w) * orphaned as f32).floor() as usize;
            if add > 0 {
                *adds.entry(layer).or_insert(0) += add;
                distributed += add;
            }
        }

        for (&layer, add) in &adds {
            *allocations.get_mut(&layer).unwrap() += *add;
        }

        // Give remainder to highest-weight layer
        if distributed < orphaned
            && let Some(&best) = candidates
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            *allocations.get_mut(&best.0).unwrap() += orphaned - distributed;
        }
    }

    /// Normalize scores within a set of results to [0, 1] range.
    /// If all scores are equal, leave them as-is.
    fn normalize_scores(results: &mut [ScoredMemory]) {
        if results.is_empty() {
            return;
        }

        let max_score = results
            .iter()
            .map(|r| r.score)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_score = results
            .iter()
            .map(|r| r.score)
            .fold(f32::INFINITY, f32::min);
        let range = max_score - min_score;

        if range <= 1e-6 {
            // All scores are essentially equal, no normalization needed
            return;
        }

        for result in results {
            result.score = (result.score - min_score) / range;
        }
    }

    /// Deduplicate across layers using abstraction_sources.
    ///
    /// If a higher-layer memory and its lower-layer source both appear,
    /// keep both but ensure the concrete source has a slight edge.
    pub fn deduplicate_across_layers(mut results: Vec<PyramidResult>) -> Vec<PyramidResult> {
        if results.len() <= 1 {
            return results;
        }

        // Build a map of all abstraction source IDs that appear in results
        let mut source_ids: HashMap<String, i32> = HashMap::new();
        for result in &results {
            let sources = &result.memory.memory.metadata.abstraction_sources;
            for src in sources {
                let src_id = src.to_string();
                source_ids.insert(src_id, result.layer);
            }
        }

        // Boost concrete sources slightly so they rank higher when tied with their abstractions
        for result in &mut results {
            let id = result.memory.memory.id.clone();
            if source_ids.contains_key(&id) {
                // This memory is a source for a higher-layer result — boost it
                result.memory.score = (result.memory.score * 1.05).min(1.0);
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentMeta, LayerInfo, Memory, MemoryMetadata, MemoryState, MemoryType};
    use chrono::Utc;
    use std::collections::HashMap;
    use uuid::Uuid;

    fn make_memory(id: &str, layer: i32, score: f32) -> ScoredMemory {
        let metadata = MemoryMetadata {
            user_id: None,
            agent_id: None,
            run_id: None,
            actor_id: None,
            role: None,
            memory_type: MemoryType::Factual,
            hash: "hash".to_string(),
            importance_score: 0.5,
            entities: vec![],
            relations: vec![],
            context: vec![],
            topics: vec![],
            custom: HashMap::new(),
            layer: LayerInfo::custom(layer, format!("layer_{}", layer)),
            abstraction_sources: vec![],
            abstraction_confidence: None,
            state: MemoryState::Active,
            forgotten_at: None,
            forgotten_by: None,
            forgotten_sources: vec![],
            last_abstraction_failure: None,
            abstraction_retry_after: None,
        };

        let memory = Memory {
            id: id.to_string(),
            content: Some(format!("content {}", layer)),
            content_meta: ContentMeta::default(),
            derived_data: HashMap::new(),
            relations: HashMap::new(),
            embedding: vec![0.0; 384],
            metadata,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            context_embeddings: None,
            relation_embeddings: None,
        };

        ScoredMemory { memory, score }
    }

    #[test]
    fn test_bottom_heavy_allocation() {
        let mut results = LayerResults::new();
        for layer in [0, 1, 2, 3] {
            results.insert(
                layer,
                (0..5)
                    .map(|i| make_memory(&format!("l{}-{}", layer, i), layer, 0.5 + i as f32 * 0.1))
                    .collect(),
            );
        }

        let assembled = PyramidAssembler::assemble(
            results,
            10,
            PyramidAllocationMode::BottomHeavy,
            HashMap::new(),
        );

        assert_eq!(assembled.len(), 10);

        // L0 should have the most results
        let layer_counts: HashMap<i32, usize> =
            assembled.iter().fold(HashMap::new(), |mut acc, r| {
                *acc.entry(r.layer).or_insert(0) += 1;
                acc
            });

        let l0 = *layer_counts.get(&0).unwrap_or(&0);
        let l3 = *layer_counts.get(&3).unwrap_or(&0);
        assert!(
            l0 >= l3,
            "L0 ({}) should have >= L3 ({}) in bottom-heavy mode",
            l0,
            l3
        );
    }

    #[test]
    fn test_balanced_allocation() {
        let mut results = LayerResults::new();
        for layer in [0, 1, 2, 3] {
            results.insert(
                layer,
                (0..5)
                    .map(|i| make_memory(&format!("l{}-{}", layer, i), layer, 0.5 + i as f32 * 0.1))
                    .collect(),
            );
        }

        let assembled =
            PyramidAssembler::assemble(results, 8, PyramidAllocationMode::Balanced, HashMap::new());

        assert_eq!(assembled.len(), 8);

        let layer_counts: HashMap<i32, usize> =
            assembled.iter().fold(HashMap::new(), |mut acc, r| {
                *acc.entry(r.layer).or_insert(0) += 1;
                acc
            });

        // Each layer should have ~2 results
        for layer in [0, 1, 2, 3] {
            let count = *layer_counts.get(&layer).unwrap_or(&0);
            assert!(
                count >= 1 && count <= 3,
                "Layer {} should have 1-3 results in balanced mode, got {}",
                layer,
                count
            );
        }
    }

    #[test]
    fn test_empty_layer_redistribution() {
        let mut results = LayerResults::new();
        // L1 and L3 have results, L0 and L2 are empty
        results.insert(
            1,
            (0..5)
                .map(|i| make_memory(&format!("l1-{}", i), 1, 0.5 + i as f32 * 0.1))
                .collect(),
        );
        results.insert(
            3,
            (0..5)
                .map(|i| make_memory(&format!("l3-{}", i), 3, 0.5 + i as f32 * 0.1))
                .collect(),
        );

        let assembled =
            PyramidAssembler::assemble(results, 8, PyramidAllocationMode::Balanced, HashMap::new());

        assert_eq!(assembled.len(), 8);
        // All results should be from L1 or L3
        for r in &assembled {
            assert!(r.layer == 1 || r.layer == 3);
        }
    }

    #[test]
    fn test_score_normalization() {
        let mut results = vec![
            make_memory("a", 0, 0.2),
            make_memory("b", 0, 0.5),
            make_memory("c", 0, 0.8),
        ];

        PyramidAssembler::normalize_scores(&mut results);

        // After normalization: min→0, max→1
        assert!((results[0].score - 0.0).abs() < 1e-5);
        assert!((results[2].score - 1.0).abs() < 1e-5);
        assert!((results[1].score - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_top_heavy_allocation() {
        let mut results = LayerResults::new();
        for layer in [0, 1, 2, 3] {
            results.insert(
                layer,
                (0..5)
                    .map(|i| make_memory(&format!("l{}-{}", layer, i), layer, 0.5 + i as f32 * 0.1))
                    .collect(),
            );
        }

        let assembled = PyramidAssembler::assemble(
            results,
            10,
            PyramidAllocationMode::TopHeavy,
            HashMap::new(),
        );

        assert_eq!(assembled.len(), 10);

        let layer_counts: HashMap<i32, usize> =
            assembled.iter().fold(HashMap::new(), |mut acc, r| {
                *acc.entry(r.layer).or_insert(0) += 1;
                acc
            });

        let l0 = *layer_counts.get(&0).unwrap_or(&0);
        let l3 = *layer_counts.get(&3).unwrap_or(&0);
        assert!(
            l3 >= l0,
            "L3 ({}) should have >= L0 ({}) in top-heavy mode",
            l3,
            l0
        );
    }

    #[test]
    fn test_empty_input() {
        let results: LayerResults = HashMap::new();
        let assembled = PyramidAssembler::assemble(
            results,
            10,
            PyramidAllocationMode::default(),
            HashMap::new(),
        );
        assert!(assembled.is_empty());
    }

    #[test]
    fn test_single_layer() {
        let mut results = LayerResults::new();
        results.insert(
            0,
            (0..5)
                .map(|i| make_memory(&format!("m{}", i), 0, 0.3 + i as f32 * 0.15))
                .collect(),
        );

        let assembled = PyramidAssembler::assemble(
            results,
            3,
            PyramidAllocationMode::default(),
            HashMap::new(),
        );

        assert_eq!(assembled.len(), 3);
        assert!(assembled.iter().all(|r| r.layer == 0));
        // Should be sorted by score descending
        for i in 1..assembled.len() {
            assert!(
                assembled[i - 1].memory.score >= assembled[i].memory.score,
                "Results should be sorted descending"
            );
        }
    }

    #[test]
    fn test_deduplicate_across_layers_boosts_sources() {
        let source_id = Uuid::new_v4();
        let mut l1_memory = make_memory(&source_id.to_string(), 1, 0.6);
        l1_memory.memory.metadata.abstraction_sources = vec![];

        let mut l2_memory = make_memory("l2-abstract", 2, 0.6);
        l2_memory.memory.metadata.abstraction_sources = vec![source_id];

        let results = vec![
            PyramidResult {
                memory: l2_memory,
                layer: 2,
                layer_name: "semantic".to_string(),
                search_phase: "pyramid".to_string(),
                graph_path: None,
            },
            PyramidResult {
                memory: l1_memory,
                layer: 1,
                layer_name: "structural".to_string(),
                search_phase: "pyramid".to_string(),
                graph_path: None,
            },
        ];

        let deduped = PyramidAssembler::deduplicate_across_layers(results);
        assert_eq!(deduped.len(), 2);

        // The L1 source should have been boosted above the L2 abstraction
        let l1_score = deduped.iter().find(|r| r.layer == 1).unwrap().memory.score;
        let l2_score = deduped.iter().find(|r| r.layer == 2).unwrap().memory.score;
        assert!(
            l1_score > l2_score,
            "Source L1 ({}) should score higher than abstraction L2 ({})",
            l1_score,
            l2_score
        );
    }

    #[test]
    fn test_deduplicate_no_sources() {
        let results = vec![
            PyramidResult {
                memory: make_memory("a", 0, 0.7),
                layer: 0,
                layer_name: "raw".to_string(),
                search_phase: "pyramid".to_string(),
                graph_path: None,
            },
            PyramidResult {
                memory: make_memory("b", 1, 0.8),
                layer: 1,
                layer_name: "structural".to_string(),
                search_phase: "pyramid".to_string(),
                graph_path: None,
            },
        ];

        let deduped = PyramidAssembler::deduplicate_across_layers(results);
        assert_eq!(deduped.len(), 2);
        // Scores unchanged since no abstraction_sources
        assert!((deduped[0].memory.score - 0.7).abs() < 1e-5);
        assert!((deduped[1].memory.score - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_custom_layer_weights() {
        let mut results = LayerResults::new();
        for layer in [0, 1, 2, 3] {
            results.insert(
                layer,
                (0..5)
                    .map(|i| make_memory(&format!("l{}-{}", layer, i), layer, 0.5 + i as f32 * 0.1))
                    .collect(),
            );
        }

        // Give L0 and L1 much more weight
        let mut weights = HashMap::new();
        weights.insert(0, 3.0);
        weights.insert(1, 3.0);
        weights.insert(2, 0.5);
        weights.insert(3, 0.5);

        let assembled =
            PyramidAssembler::assemble(results, 10, PyramidAllocationMode::Balanced, weights);
        assert_eq!(assembled.len(), 10);

        let layer_counts: HashMap<i32, usize> =
            assembled.iter().fold(HashMap::new(), |mut acc, r| {
                *acc.entry(r.layer).or_insert(0) += 1;
                acc
            });

        let low = layer_counts.get(&0).unwrap_or(&0) + layer_counts.get(&1).unwrap_or(&0);
        let high = layer_counts.get(&2).unwrap_or(&0) + layer_counts.get(&3).unwrap_or(&0);
        assert!(
            low > high,
            "L0+L1 ({}) should have more results than L2+L3 ({}) with custom weights",
            low,
            high
        );
    }

    #[test]
    fn test_normalize_equal_scores_unchanged() {
        let mut results = vec![
            make_memory("a", 0, 0.5),
            make_memory("b", 0, 0.5),
            make_memory("c", 0, 0.5),
        ];

        PyramidAssembler::normalize_scores(&mut results);
        // All scores equal — should remain unchanged
        for r in &results {
            assert!((r.score - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_normalize_empty() {
        let mut results: Vec<ScoredMemory> = vec![];
        PyramidAssembler::normalize_scores(&mut results);
        assert!(results.is_empty());
    }

    #[test]
    fn test_normalize_single_result() {
        let mut results = vec![make_memory("only", 0, 0.73)];
        PyramidAssembler::normalize_scores(&mut results);
        // Single result — range is 0, so score stays unchanged
        assert!((results[0].score - 0.73).abs() < 1e-5);
    }

    #[test]
    fn test_assemble_limit_exceeds_available() {
        let mut results = LayerResults::new();
        results.insert(0, vec![make_memory("m0", 0, 0.8)]);
        results.insert(1, vec![make_memory("m1", 1, 0.9)]);

        // Ask for 10 but only 2 exist
        let assembled = PyramidAssembler::assemble(
            results,
            10,
            PyramidAllocationMode::Balanced,
            HashMap::new(),
        );
        assert_eq!(assembled.len(), 2);
    }
}
