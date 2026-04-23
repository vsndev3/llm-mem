use crate::OutputFormat;
use llm_mem::System;
use llm_mem::operations::MemoryOperationPayload;
use std::fmt::Write;

/// Computed layer statistics from a set of memories.
#[derive(Debug, PartialEq)]
pub(crate) struct LayerStatsResult {
    pub layer_counts: std::collections::HashMap<i32, usize>,
    pub state_counts: std::collections::HashMap<String, usize>,
    pub type_counts: std::collections::HashMap<String, usize>,
    pub total_abstraction_sources: usize,
    pub total: usize,
}

impl LayerStatsResult {
    pub fn active(&self) -> usize {
        self.state_counts.get("Active").copied().unwrap_or(0)
    }
    pub fn forgotten(&self) -> usize {
        self.state_counts.get("Forgotten").copied().unwrap_or(0)
    }
    pub fn max_layer(&self) -> i32 {
        self.layer_counts.keys().max().copied().unwrap_or(0)
    }
    pub fn avg_sources_per_active(&self) -> f64 {
        let non_forgotten = self.total.saturating_sub(self.forgotten());
        if non_forgotten == 0 {
            0.0
        } else {
            self.total_abstraction_sources as f64 / non_forgotten as f64
        }
    }
}

/// Compute layer statistics from a slice of memory JSON values.
pub(crate) fn compute_layer_stats(memories: &[serde_json::Value]) -> LayerStatsResult {
    let mut layer_counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    let mut state_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut type_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut total_abstraction_sources = 0usize;

    for memory in memories {
        if let serde_json::Value::Object(mem_obj) = memory {
            // Fields are nested inside "metadata"
            let meta = mem_obj.get("metadata").and_then(|m| m.as_object());
            if let Some(meta_obj) = meta {
                if let Some(serde_json::Value::Number(layer_num)) = meta_obj.get("layer")
                    && let Some(layer) = layer_num.as_i64()
                {
                    *layer_counts.entry(layer as i32).or_insert(0) += 1;
                }
                if let Some(serde_json::Value::String(state)) = meta_obj.get("state") {
                    *state_counts.entry(state.clone()).or_insert(0) += 1;
                }
                if let Some(serde_json::Value::String(mt)) = meta_obj.get("memory_type") {
                    *type_counts.entry(mt.clone()).or_insert(0) += 1;
                }
                if let Some(serde_json::Value::Array(sources)) = meta_obj.get("abstraction_sources")
                {
                    total_abstraction_sources += sources.len();
                }
            }
        }
    }

    LayerStatsResult {
        layer_counts,
        state_counts,
        type_counts,
        total_abstraction_sources,
        total: memories.len(),
    }
}

/// Build a JSON object from layer stats.
fn layer_stats_to_json(bank: &str, stats: &LayerStatsResult) -> serde_json::Value {
    serde_json::json!({
        "bank": bank,
        "total_memories": stats.total,
        "active": stats.active(),
        "forgotten": stats.forgotten(),
        "max_layer": stats.max_layer(),
        "by_layer": stats.layer_counts,
        "by_state": stats.state_counts,
        "by_type": stats.type_counts,
        "total_abstraction_sources": stats.total_abstraction_sources
    })
}

/// Format layer stats output as a string for a given format.
pub fn format_layer_stats_output(
    bank: &str,
    stats: &LayerStatsResult,
    format: OutputFormat,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut buf = String::new();
    match format {
        OutputFormat::Table => {
            writeln!(buf, "Layer Statistics for bank '{}'", bank)?;
            writeln!(buf, "{}", "=".repeat(60))?;
            writeln!(buf)?;
            writeln!(buf, "Overview:")?;
            writeln!(buf, "  Total memories:     {}", stats.total)?;
            writeln!(buf, "  Active:             {}", stats.active())?;
            writeln!(buf, "  Forgotten:          {}", stats.forgotten())?;
            writeln!(buf, "  Max layer:          {}", stats.max_layer())?;
            writeln!(buf)?;
            writeln!(buf, "By Layer:")?;
            writeln!(buf, "  {:<25} {:>8} {:>10}", "Layer", "Count", "% of Total")?;
            writeln!(buf, "  {}", "-".repeat(45))?;
            let mut levels: Vec<_> = stats.layer_counts.keys().collect();
            levels.sort();
            for level in levels {
                let count = stats.layer_counts.get(level).unwrap();
                let pct = if stats.total > 0 {
                    (*count as f64 / stats.total as f64) * 100.0
                } else {
                    0.0
                };
                let layer_name = crate::commands::stats::layer_display_name(&level.to_string());
                writeln!(buf, "  {:<25} {:>8} {:>9.1}%", layer_name, count, pct)?;
            }
            writeln!(buf)?;
            writeln!(buf, "Abstraction Metrics:")?;
            writeln!(
                buf,
                "  Total source links: {}",
                stats.total_abstraction_sources
            )?;
            if stats.total > stats.forgotten() {
                writeln!(
                    buf,
                    "  Avg sources/memory: {:.2}",
                    stats.avg_sources_per_active()
                )?;
            }
        }
        _ => {
            let json = layer_stats_to_json(bank, stats);
            writeln!(buf, "{}", serde_json::to_string_pretty(&json)?)?;
        }
    }
    Ok(buf)
}

/// Handle the layer-stats command
pub async fn handle_layer_stats(
    system: &System,
    bank: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build the payload for list_memories operation (to get all memories for layer stats)
    let payload = MemoryOperationPayload {
        bank: Some(bank.to_string()),
        ..Default::default()
    };

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.list_memories(payload).await {
        Ok(response) => {
            // For layer stats, we want to compute and display layer statistics
            if let Some(data) = &response.data {
                // data is {"count": N, "memories": [...]}, extract the memories array
                let memories_value = if let serde_json::Value::Object(obj) = data {
                    obj.get("memories").cloned()
                } else if data.is_array() {
                    Some(data.clone())
                } else {
                    None
                };
                if let Some(serde_json::Value::Array(memories)) = &memories_value {
                    if memories.is_empty() {
                        println!("No memories found in bank '{}'", bank);
                        return Ok(());
                    }

                    // Calculate layer statistics
                    let stats = compute_layer_stats(memories);

                    // Print statistics based on format
                    let output = format_layer_stats_output(bank, &stats, format)?;
                    print!("{}", output);
                } else {
                    // Fall back to regular output formatting
                    crate::output::print_response(&response, format)?;
                }
            } else {
                crate::output::print_response(&response, format)?;
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_memory(layer: i32, state: &str, mem_type: &str, sources: usize) -> serde_json::Value {
        let srcs: Vec<serde_json::Value> =
            (0..sources).map(|i| json!(format!("mem-{}", i))).collect();
        json!({
            "metadata": {
                "layer": layer,
                "state": state,
                "memory_type": mem_type,
                "abstraction_sources": srcs
            }
        })
    }

    #[test]
    fn test_compute_layer_stats_empty() {
        let stats = compute_layer_stats(&[]);
        assert_eq!(stats.total, 0);
        assert!(stats.layer_counts.is_empty());
        assert_eq!(stats.active(), 0);
        assert_eq!(stats.forgotten(), 0);
        assert_eq!(stats.max_layer(), 0);
        assert_eq!(stats.avg_sources_per_active(), 0.0);
    }

    #[test]
    fn test_compute_layer_stats_basic() {
        let memories = vec![
            make_memory(0, "Active", "observation", 0),
            make_memory(0, "Active", "observation", 0),
            make_memory(1, "Active", "document", 2),
            make_memory(2, "Active", "concept", 3),
        ];
        let stats = compute_layer_stats(&memories);
        assert_eq!(stats.total, 4);
        assert_eq!(stats.active(), 4);
        assert_eq!(stats.forgotten(), 0);
        assert_eq!(stats.max_layer(), 2);
        assert_eq!(*stats.layer_counts.get(&0).unwrap(), 2);
        assert_eq!(*stats.layer_counts.get(&1).unwrap(), 1);
        assert_eq!(*stats.layer_counts.get(&2).unwrap(), 1);
        assert_eq!(stats.total_abstraction_sources, 5);
    }

    #[test]
    fn test_compute_layer_stats_with_forgotten() {
        let memories = vec![
            make_memory(0, "Active", "observation", 0),
            make_memory(-1, "Forgotten", "observation", 0),
            make_memory(1, "Active", "document", 4),
        ];
        let stats = compute_layer_stats(&memories);
        assert_eq!(stats.active(), 2);
        assert_eq!(stats.forgotten(), 1);
        assert_eq!(stats.max_layer(), 1);
        // avg_sources_per_active: 4 / (3-1) = 2.0
        assert!((stats.avg_sources_per_active() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_layer_stats_all_forgotten() {
        let memories = vec![
            make_memory(-1, "Forgotten", "obs", 0),
            make_memory(-1, "Forgotten", "obs", 0),
        ];
        let stats = compute_layer_stats(&memories);
        assert_eq!(stats.forgotten(), 2);
        assert_eq!(stats.avg_sources_per_active(), 0.0);
    }

    #[test]
    fn test_compute_layer_stats_type_counts() {
        let memories = vec![
            make_memory(0, "Active", "observation", 0),
            make_memory(0, "Active", "document", 0),
            make_memory(0, "Active", "observation", 0),
            make_memory(1, "Active", "concept", 0),
        ];
        let stats = compute_layer_stats(&memories);
        assert_eq!(*stats.type_counts.get("observation").unwrap(), 2);
        assert_eq!(*stats.type_counts.get("document").unwrap(), 1);
        assert_eq!(*stats.type_counts.get("concept").unwrap(), 1);
    }

    #[test]
    fn test_layer_stats_result_helpers() {
        let stats = LayerStatsResult {
            layer_counts: [(0, 5), (1, 3), (2, 2)].into_iter().collect(),
            state_counts: [("Active".to_string(), 8), ("Forgotten".to_string(), 2)]
                .into_iter()
                .collect(),
            type_counts: [("observation".to_string(), 10)].into_iter().collect(),
            total_abstraction_sources: 15,
            total: 10,
        };
        assert_eq!(stats.active(), 8);
        assert_eq!(stats.forgotten(), 2);
        assert_eq!(stats.max_layer(), 2);
        // avg: 15 / (10-2) = 1.875
        assert!((stats.avg_sources_per_active() - 1.875).abs() < f64::EPSILON);
    }
}
