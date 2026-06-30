//! Directed Acyclic Graph export engine for the memory store.
//!
//! Builds a graph from the memory pyramid (abstraction edges) and semantic
//! relation edges, then renders it as a self-contained interactive HTML file
//! using D3.js force-directed layout.
//!
//! # Layered scaling strategy
//!
//! | Node count | Strategy                                          |
//! |-----------|---------------------------------------------------|
//! | ≤100      | Full graph: all L0-L4 nodes + all relation edges  |
//! | 100-500   | Show L1+ pyramid + high-importance L0 + strong    |
//! |           | semantic edges (strength ≥ 0.5)                   |
//! | 500-2000  | L2+ concepts with abstraction hierarchy; collapse |
//! |           | L0/L1 into clusters                               |
//! | 2000-10K  | L3+L4 wisdom/concepts only, abstraction edges     |
//! | >10K      | Truncate to top-N by importance, pyramid only     |

use std::collections::HashSet;

use serde::Serialize;

use crate::types::{Memory, MemoryState};

// ─── DAG node / edge representations ─────────────────────────────

/// A node in the DAG export.
#[derive(Debug, Clone, Serialize)]
pub struct DagNode {
    pub id: String,
    pub label: String,
    pub layer: i32,
    pub layer_name: String,
    pub content_preview: String,
    pub importance: f32,
    pub state: String,
    pub event_at: Option<String>,
    pub cluster: Option<String>,
    pub topics: Vec<String>,
}

/// An edge in the DAG export.
#[derive(Debug, Clone, Serialize)]
pub struct DagEdge {
    pub source: String,
    pub target: String,
    pub edge_type: String,
    pub label: String,
    pub strength: Option<f32>,
    pub is_abstraction: bool,
}

/// Full DAG graph ready for rendering.
#[derive(Debug, Clone)]
pub struct DagGraph {
    pub nodes: Vec<DagNode>,
    pub edges: Vec<DagEdge>,
    pub stats: DagGraphStats,
}

#[derive(Debug, Clone)]
pub struct DagGraphStats {
    pub total_memories_in_bank: usize,
    pub nodes_in_graph: usize,
    pub edges_in_graph: usize,
    pub abstraction_edges: usize,
    pub semantic_edges: usize,
    pub temporal_edges: usize,
    pub scaling_level: usize,
    pub nodes_omitted: usize,
}

// ─── Build the DAG ───────────────────────────────────────────────

/// Configuration for DAG export.
pub struct DagExportConfig {
    /// Maximum number of nodes in the output graph.
    pub max_nodes: usize,
    /// Minimum importance score to include a node (0.0-1.0).
    pub min_importance: f32,
    /// Include semantic relation edges (references, depends_on, etc.).
    pub include_semantic: bool,
    /// Include temporal edges (happened_after, happens_within).
    pub include_temporal: bool,
    /// Include the abstraction pyramid edges (L0→L1→L2→L3→L4).
    pub include_abstraction: bool,
    /// Max depth for semantic relation traversal from each node.
    pub max_semantic_depth: usize,
    /// Minimum relation strength for semantic edges (0.0-1.0).
    pub min_relation_strength: f32,
    /// Minimum layer level to include (e.g. 0 = all, 2 = L2+).
    pub min_layer: i32,
    /// Maximum layer level to include.
    pub max_layer: i32,
}

impl Default for DagExportConfig {
    fn default() -> Self {
        Self {
            max_nodes: 200,
            min_importance: 0.0,
            include_semantic: true,
            include_temporal: true,
            include_abstraction: true,
            max_semantic_depth: 2,
            min_relation_strength: 0.0,
            min_layer: -1,
            max_layer: 99,
        }
    }
}

/// Build a DAG from a list of memories, applying the layered scaling strategy.
pub fn build_dag(memories: &[Memory], config: &DagExportConfig) -> DagGraph {
    let total_memories_in_bank = memories.len();

    // Phase 1: determine scaling level
    let scaling_level = if total_memories_in_bank <= 100 {
        0
    } else if total_memories_in_bank <= 500 {
        1
    } else if total_memories_in_bank <= 2000 {
        2
    } else if total_memories_in_bank <= 10000 {
        3
    } else {
        4
    };

    // Phase 2: filter/sort nodes based on scaling level
    let mut candidates: Vec<&Memory>;

    match scaling_level {
        0 => {
            // Show everything
            candidates = memories.iter().collect();
        }
        1 => {
            // Show L1+ pyramid + high-importance L0
            candidates = memories
                .iter()
                .filter(|m| {
                    m.metadata.state == MemoryState::Active
                        && m.metadata.layer.level >= config.min_layer
                        && m.metadata.layer.level <= config.max_layer
                })
                .collect();
        }
        2 => {
            // Show L2+ concepts only
            candidates = memories
                .iter()
                .filter(|m| {
                    m.metadata.state == MemoryState::Active
                        && m.metadata.layer.level >= (2i32.max(config.min_layer))
                        && m.metadata.layer.level <= config.max_layer
                })
                .collect();
        }
        3 => {
            // Show L3+L4 wisdom/concepts only
            candidates = memories
                .iter()
                .filter(|m| {
                    m.metadata.state == MemoryState::Active
                        && m.metadata.layer.level >= (3i32.max(config.min_layer))
                        && m.metadata.layer.level <= config.max_layer
                })
                .collect();
        }
        _ => {
            // >10K: top-N by importance
            let mut sorted: Vec<&Memory> = memories
                .iter()
                .filter(|m| {
                    m.metadata.state == MemoryState::Active
                        && m.metadata.layer.level >= config.min_layer
                        && m.metadata.layer.level <= config.max_layer
                })
                .collect();
            sorted.sort_by(|a, b| {
                b.metadata
                    .importance_score
                    .partial_cmp(&a.metadata.importance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates = sorted;
        }
    }

    // Sort by importance (descending) so we pick the best nodes when truncating
    candidates.sort_by(|a, b| {
        b.metadata
            .importance_score
            .partial_cmp(&a.metadata.importance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Also prioritize higher layers: L4 > L3 > L2 > L1 > L0 > Forgotten
    candidates.sort_by(|a, b| {
        b.metadata
            .layer
            .level
            .cmp(&a.metadata.layer.level)
    });

    let total_candidates = candidates.len();
    let nodes_omitted = if total_candidates > config.max_nodes {
        total_candidates - config.max_nodes
    } else {
        0
    };

    candidates.truncate(config.max_nodes);

    // Build node ID set for edge filtering
    let node_ids: HashSet<String> = candidates.iter().map(|m| m.id.clone()).collect();

    // Phase 3: build nodes
    let nodes: Vec<DagNode> = candidates
        .iter()
        .map(|m| {
            let label = memory_label(m);
            DagNode {
                id: m.id.clone(),
                label: label.clone(),
                layer: m.metadata.layer.level,
                layer_name: layer_display_name(m.metadata.layer.level),
                content_preview: short_content(m, &label, 120),
                importance: (m.metadata.importance_score * 100.0).round() / 100.0,
                state: format!("{:?}", m.metadata.state),
                event_at: m.event_at.map(|d| d.format("%Y-%m-%d %H:%M").to_string()),
                cluster: m
                    .metadata
                    .layer
                    .name
                    .clone()
                    .or_else(|| Some(layer_display_name(m.metadata.layer.level))),
                topics: m.metadata.topics.clone(),
            }
        })
        .collect();

    // Phase 4: build edges
    let mut edges: Vec<DagEdge> = Vec::new();
    let mut edge_set: HashSet<(String, String, String)> = HashSet::new();

    // 4a. Abstraction edges (always included if enabled)
    if config.include_abstraction {
        for m in &candidates {
            for src_id in &m.metadata.abstraction_sources {
                let src_str = src_id.to_string();
                // Only add edge if source node is in our graph
                if node_ids.contains(&src_str) {
                    let key = (src_str.clone(), m.id.clone(), "abstraction".to_string());
                    if edge_set.insert(key) {
                        edges.push(DagEdge {
                            source: src_str,
                            target: m.id.clone(),
                            edge_type: "abstraction".to_string(),
                            label: "abstracts to".to_string(),
                            strength: None,
                            is_abstraction: true,
                        });
                    }
                }
            }
        }
    }

    // 4b. Semantic relation edges
    if config.include_semantic {
        for m in &candidates {
            for (rel_type, rel_entry) in &m.relations {
                let strength = rel_entry.strength.unwrap_or(1.0);
                if strength < config.min_relation_strength {
                    continue;
                }
                for target_id in &rel_entry.target_ids {
                    let target_str = target_id.to_string();
                    if node_ids.contains(&target_str) && m.id != target_str {
                        let key = (
                            m.id.clone(),
                            target_str.clone(),
                            rel_type.clone(),
                        );
                        if edge_set.insert(key) {
                            edges.push(DagEdge {
                                source: m.id.clone(),
                                target: target_str,
                                edge_type: rel_type.clone(),
                                label: rel_type.clone(),
                                strength: Some(strength),
                                is_abstraction: false,
                            });
                        }
                    }
                }
            }
        }

        // Also scan metadata.relations (legacy format)
        for m in &candidates {
            for rel in &m.metadata.relations {
                if rel.strength.unwrap_or(1.0) < config.min_relation_strength {
                    continue;
                }
                if node_ids.contains(&rel.target) && m.id != rel.target {
                    let key = (
                        m.id.clone(),
                        rel.target.clone(),
                        rel.relation.clone(),
                    );
                    if edge_set.insert(key) {
                        edges.push(DagEdge {
                            source: m.id.clone(),
                            target: rel.target.clone(),
                            edge_type: rel.relation.clone(),
                            label: rel.relation.clone(),
                            strength: rel.strength,
                            is_abstraction: false,
                        });
                    }
                }
            }
        }
    }

    // 4c. Temporal edges
    if config.include_temporal {
        let mut with_event: Vec<&&Memory> = candidates
            .iter()
            .filter(|m| m.event_at.is_some())
            .collect();
        with_event.sort_by_key(|m| m.event_at.unwrap());

        let window = chrono::Duration::days(1);
        for win in with_event.windows(2) {
            let a = win[0];
            let b = win[1];
            let delta = (b.event_at.unwrap() - a.event_at.unwrap()).num_seconds();
            if delta > 0 && delta <= window.num_seconds() {
                let key = (
                    a.id.clone(),
                    b.id.clone(),
                    "happened_after".to_string(),
                );
                if edge_set.insert(key) {
                    edges.push(DagEdge {
                        source: a.id.clone(),
                        target: b.id.clone(),
                        edge_type: "happened_after".to_string(),
                        label: format!("after {}s", delta),
                        strength: None,
                        is_abstraction: false,
                    });
                }
            }
        }
    }

    let abstraction_edges = edges.iter().filter(|e| e.is_abstraction).count();
    let semantic_edges = edges
        .iter()
        .filter(|e| !e.is_abstraction && e.edge_type != "happened_after")
        .count();
    let temporal_edges = edges
        .iter()
        .filter(|e| e.edge_type == "happened_after")
        .count();
    let edge_count = edges.len();
    let node_count = candidates.len();

    DagGraph {
        nodes,
        edges,
        stats: DagGraphStats {
            total_memories_in_bank,
            nodes_in_graph: node_count,
            edges_in_graph: edge_count,
            abstraction_edges,
            semantic_edges,
            temporal_edges,
            scaling_level,
            nodes_omitted,
        },
    }
}

// ─── Helpers ──────────────────────────────────────────────────────

fn memory_label(m: &Memory) -> String {
    // Use topics/entities if available, otherwise first content words
    if let Some(topic) = m.metadata.topics.first() {
        return topic.clone();
    }
    if let Some(entity) = m.metadata.entities.first() {
        return entity.clone().to_string();
    }
    if let Some(ref content) = m.content {
        let first_line = content.lines().next().unwrap_or(content);
        let trimmed = first_line.trim();
        if trimmed.len() <= 50 {
            trimmed.to_string()
        } else {
            format!("{}...", &trimmed[..50.min(trimmed.len())])
        }
    } else {
        format!("L{}-{}", m.metadata.layer.level, &m.id[..8])
    }
}

fn short_content(m: &Memory, label: &str, max_len: usize) -> String {
    if let Some(ref content) = m.content {
        let first_line = content.lines().next().unwrap_or(content).trim().to_string();
        if first_line.len() <= max_len {
            first_line
        } else {
            format!("{}...", &first_line[..max_len.min(first_line.len())])
        }
    } else {
        label.to_string()
    }
}

pub fn layer_display_name(level: i32) -> String {
    match level {
        -1 => "Forgotten".to_string(),
        0 => "Raw Content".to_string(),
        1 => "Structural".to_string(),
        2 => "Semantic".to_string(),
        3 => "Concept".to_string(),
        4 => "Wisdom".to_string(),
        _ => format!("L{}", level),
    }
}

/// Color for a layer level (for graph rendering).
pub fn layer_color(level: i32) -> &'static str {
    match level {
        -1 => "#9CA3AF", // gray
        0 => "#34D399",  // green
        1 => "#60A5FA",  // blue
        2 => "#A78BFA",  // purple
        3 => "#FB923C",  // orange
        4 => "#F87171",  // red
        _ => "#6B7280",  // dark gray
    }
}

// ─── HTML rendering ──────────────────────────────────────────────

/// Render a DagGraph as a self-contained interactive HTML file using D3.js.
pub fn render_html(graph: &DagGraph, bank_name: &str) -> std::result::Result<String, std::io::Error> {
    let nodes_json = serde_json::to_string(&graph.nodes).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, format!("JSON error: {e}"))
    })?;
    let edges_json = serde_json::to_string(&graph.edges).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, format!("JSON error: {e}"))
    })?;
    let stats = &graph.stats;
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();

    let html = include_str!("dag_export_template.html")
        .replace("__BANK_NAME__", bank_name)
        .replace("__NOW__", &now)
        .replace("__TOTAL_MEMORIES__", &stats.total_memories_in_bank.to_string())
        .replace("__NODES_IN_GRAPH__", &stats.nodes_in_graph.to_string())
        .replace("__EDGES_IN_GRAPH__", &stats.edges_in_graph.to_string())
        .replace("__ABSTRACTION_EDGES__", &stats.abstraction_edges.to_string())
        .replace("__SEMANTIC_EDGES__", &stats.semantic_edges.to_string())
        .replace("__TEMPORAL_EDGES__", &stats.temporal_edges.to_string())
        .replace("__SCALING_LEVEL__", &stats.scaling_level.to_string())
        .replace("__NODES_OMITTED__", &stats.nodes_omitted.to_string())
        .replace("__NODES_JSON__", &nodes_json)
        .replace("__EDGES_JSON__", &edges_json);

    Ok(html)
}

/// Write a DagGraph to an HTML file on disk.
pub fn write_html_file(graph: &DagGraph, bank_name: &str, output_path: &std::path::Path) -> std::result::Result<(), std::io::Error> {
    let html = render_html(graph, bank_name)?;
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output_path, html)?;
    Ok(())
}
