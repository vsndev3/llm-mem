use crate::OutputFormat;
use llm_mem::System;
use llm_mem::operations::MemoryOperationPayload;
use std::fmt::Write;

/// Map a layer number string to its display name
pub(crate) fn layer_display_name(layer: &str) -> String {
    match layer {
        "0" => "L0 (Raw Content)".to_string(),
        "1" => "L1 (Structural)".to_string(),
        "2" => "L2 (Semantic)".to_string(),
        "3" => "L3 (Concept)".to_string(),
        "4" => "L4 (Wisdom)".to_string(),
        "-1" => "L-1 (Forgotten)".to_string(),
        _ => format!("L{} (Custom)", layer),
    }
}

/// Compute aggregate counts from a slice of memory JSON values.
/// Returns (type_counts, state_counts, layer_counts).
pub(crate) fn compute_memory_counts(
    memories: &[serde_json::Value],
) -> (
    std::collections::HashMap<String, usize>,
    std::collections::HashMap<String, usize>,
    std::collections::HashMap<String, usize>,
) {
    let mut type_counts = std::collections::HashMap::new();
    let mut state_counts = std::collections::HashMap::new();
    let mut layer_counts = std::collections::HashMap::new();

    for memory in memories {
        if let serde_json::Value::Object(mem_obj) = memory {
            // Fields are nested inside "metadata"
            let meta = mem_obj.get("metadata").and_then(|m| m.as_object());
            if let Some(meta_obj) = meta {
                if let Some(serde_json::Value::String(mt)) = meta_obj.get("memory_type") {
                    *type_counts.entry(mt.clone()).or_insert(0) += 1;
                }
                if let Some(serde_json::Value::String(state)) = meta_obj.get("state") {
                    *state_counts.entry(state.clone()).or_insert(0) += 1;
                }
                if let Some(serde_json::Value::Number(layer_num)) = meta_obj.get("layer")
                    && let Some(layer) = layer_num.as_i64()
                {
                    *layer_counts.entry(layer.to_string()).or_insert(0) += 1;
                }
            }
        }
    }

    (type_counts, state_counts, layer_counts)
}

/// Build a JSON object from computed stats for structured output.
fn stats_to_json(
    bank: &str,
    total_count: usize,
    type_counts: &std::collections::HashMap<String, usize>,
    state_counts: &std::collections::HashMap<String, usize>,
    layer_counts: &std::collections::HashMap<String, usize>,
) -> serde_json::Value {
    serde_json::json!({
        "bank": bank,
        "total_memories": total_count,
        "by_type": type_counts,
        "by_state": state_counts,
        "by_layer": layer_counts
    })
}

/// Format stats output as a string for a given format.
pub fn format_stats_output(
    bank: &str,
    total_count: usize,
    type_counts: &std::collections::HashMap<String, usize>,
    state_counts: &std::collections::HashMap<String, usize>,
    layer_counts: &std::collections::HashMap<String, usize>,
    format: OutputFormat,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut buf = String::new();
    match format {
        OutputFormat::Table => {
            writeln!(buf, "Statistics for bank '{}'", bank)?;
            writeln!(buf, "{}", "=".repeat(50))?;
            writeln!(buf, "Total memories: {}", total_count)?;
            writeln!(buf)?;

            if !type_counts.is_empty() {
                writeln!(buf, "By Memory Type:")?;
                for (mem_type, count) in type_counts {
                    let percentage = if total_count > 0 {
                        (*count as f64 / total_count as f64) * 100.0
                    } else {
                        0.0
                    };
                    writeln!(buf, "  {:<20} {:>5} ({:.1}%)", mem_type, count, percentage)?;
                }
                writeln!(buf)?;
            }

            if !state_counts.is_empty() {
                writeln!(buf, "By State:")?;
                for (state, count) in state_counts {
                    let percentage = if total_count > 0 {
                        (*count as f64 / total_count as f64) * 100.0
                    } else {
                        0.0
                    };
                    writeln!(buf, "  {:<20} {:>5} ({:.1}%)", state, count, percentage)?;
                }
                writeln!(buf)?;
            }

            if !layer_counts.is_empty() {
                writeln!(buf, "By Layer:")?;
                let mut layers: Vec<_> = layer_counts.keys().collect();
                layers.sort_by_key(|k| k.parse::<i32>().unwrap_or(0));
                for layer in layers {
                    let count = *layer_counts.get(layer).unwrap();
                    let percentage = if total_count > 0 {
                        (count as f64 / total_count as f64) * 100.0
                    } else {
                        0.0
                    };
                    let display = layer_display_name(layer);
                    writeln!(buf, "  {:<25} {:>8} {:>9.1}%", display, count, percentage)?;
                }
            }
        }
        _ => {
            let json = stats_to_json(bank, total_count, type_counts, state_counts, layer_counts);
            writeln!(buf, "{}", serde_json::to_string_pretty(&json)?)?;
        }
    }
    Ok(buf)
}

/// Handle the stats command
pub async fn handle_stats(
    system: &System,
    bank: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build the payload for list_memories operation (to get all memories for stats)
    let payload = MemoryOperationPayload {
        bank: Some(bank.to_string()),
        ..Default::default()
    };

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.list_memories(payload).await {
        Ok(response) => {
            // For stats, we want to compute and display statistics
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

                    // Calculate statistics
                    let total_count = memories.len();
                    let (type_counts, state_counts, layer_counts) = compute_memory_counts(memories);

                    // Print statistics based on format
                    let output = format_stats_output(
                        bank,
                        total_count,
                        &type_counts,
                        &state_counts,
                        &layer_counts,
                        format,
                    )?;
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

    // --- layer_display_name tests ---

    #[test]
    fn test_layer_display_name_known_layers() {
        assert_eq!(layer_display_name("0"), "L0 (Raw Content)");
        assert_eq!(layer_display_name("1"), "L1 (Structural)");
        assert_eq!(layer_display_name("2"), "L2 (Semantic)");
        assert_eq!(layer_display_name("3"), "L3 (Concept)");
        assert_eq!(layer_display_name("4"), "L4 (Wisdom)");
        assert_eq!(layer_display_name("-1"), "L-1 (Forgotten)");
    }

    #[test]
    fn test_layer_display_name_custom_layer() {
        assert_eq!(layer_display_name("5"), "L5 (Custom)");
        assert_eq!(layer_display_name("10"), "L10 (Custom)");
    }

    // --- compute_memory_counts tests ---

    #[test]
    fn test_compute_memory_counts_empty() {
        let (types, states, layers) = compute_memory_counts(&[]);
        assert!(types.is_empty());
        assert!(states.is_empty());
        assert!(layers.is_empty());
    }

    #[test]
    fn test_compute_memory_counts_single_memory() {
        let memories = vec![json!({
            "metadata": {
                "memory_type": "observation",
                "state": "Active",
                "layer": 0
            }
        })];
        let (types, states, layers) = compute_memory_counts(&memories);
        assert_eq!(types.get("observation"), Some(&1));
        assert_eq!(states.get("Active"), Some(&1));
        assert_eq!(layers.get("0"), Some(&1));
    }

    #[test]
    fn test_compute_memory_counts_multiple_memories() {
        let memories = vec![
            json!({"metadata": {"memory_type": "observation", "state": "Active", "layer": 0}}),
            json!({"metadata": {"memory_type": "observation", "state": "Active", "layer": 0}}),
            json!({"metadata": {"memory_type": "document", "state": "Active", "layer": 1}}),
            json!({"metadata": {"memory_type": "observation", "state": "Forgotten", "layer": -1}}),
        ];
        let (types, states, layers) = compute_memory_counts(&memories);
        assert_eq!(types.get("observation"), Some(&3));
        assert_eq!(types.get("document"), Some(&1));
        assert_eq!(states.get("Active"), Some(&3));
        assert_eq!(states.get("Forgotten"), Some(&1));
        assert_eq!(layers.get("0"), Some(&2));
        assert_eq!(layers.get("1"), Some(&1));
        assert_eq!(layers.get("-1"), Some(&1));
    }

    #[test]
    fn test_compute_memory_counts_missing_fields() {
        let memories = vec![
            json!({"metadata": {"memory_type": "observation"}}), // missing state and layer
            json!({"metadata": {"state": "Active"}}),            // missing type and layer
            json!({"metadata": {"layer": 2}}),                   // missing type and state
        ];
        let (types, states, layers) = compute_memory_counts(&memories);
        assert_eq!(types.len(), 1);
        assert_eq!(states.len(), 1);
        assert_eq!(layers.len(), 1);
    }

    #[test]
    fn test_compute_memory_counts_non_object_values() {
        let memories = vec![json!("not an object"), json!(42), json!(null)];
        let (types, states, layers) = compute_memory_counts(&memories);
        assert!(types.is_empty());
        assert!(states.is_empty());
        assert!(layers.is_empty());
    }
}
