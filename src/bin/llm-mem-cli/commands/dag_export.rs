//! CLI handler for `dag-export` — renders the memory bank as an
//! interactive DAG visualization in a self-contained HTML file.

use std::path::PathBuf;

use llm_mem::{
    System,
    dag_export::{build_dag, write_html_file, DagExportConfig},
    types::{Filters, MemoryState},
};

pub async fn handle_dag_export(
    system: &System,
    bank: &str,
    output: &PathBuf,
    max_nodes: usize,
    min_importance: f32,
    include_semantic: bool,
    include_temporal: bool,
    include_abstraction: bool,
    min_layer: i32,
    max_layer: i32,
    min_relation_strength: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let manager = system
        .bank_manager
        .resolve_bank(Some(bank))
        .await
        .map_err(|e| format!("Failed to resolve bank '{}': {}", bank, e))?;

    eprintln!("Loading memories from bank '{}'...", bank);

    let memories: Vec<llm_mem::types::Memory> = manager
        .list(
            &Filters {
                state: Some(MemoryState::Active),
                min_layer_level: Some(min_layer),
                max_layer_level: Some(max_layer),
                ..Filters::default()
            },
            Some(100_000),
        )
        .await
        .map_err(|e| format!("Failed to list memories: {}", e))?;

    let total = memories.len();
    eprintln!("Loaded {} memories.", total);

    let config = DagExportConfig {
        max_nodes,
        min_importance,
        include_semantic,
        include_temporal,
        include_abstraction,
        max_semantic_depth: 2,
        min_relation_strength,
        min_layer,
        max_layer,
    };

    eprintln!("Building DAG graph...");
    let graph = build_dag(&memories, &config);

    let stats = &graph.stats;
    eprintln!(
        "Graph built: {} nodes, {} edges ({} abstraction, {} semantic, {} temporal).",
        stats.nodes_in_graph,
        stats.edges_in_graph,
        stats.abstraction_edges,
        stats.semantic_edges,
        stats.temporal_edges,
    );
    eprintln!(
        "Scaling level: {} ({} total in bank, {} omitted).",
        stats.scaling_level, stats.total_memories_in_bank, stats.nodes_omitted,
    );

    eprintln!("Rendering HTML to {}...", output.display());
    write_html_file(&graph, bank, output)?;

    eprintln!("Done. Open {} in a browser to explore the graph.", output.display());

    Ok(())
}
