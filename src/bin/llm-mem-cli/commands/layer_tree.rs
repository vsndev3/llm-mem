use llm_mem::operations::MemoryOperationPayload;
use llm_mem::System;
use crate::OutputFormat;

/// Get the display name for a layer level in tree view.
pub(crate) fn layer_tree_name(level: i32) -> String {
    match level {
        0 => "Raw Content".to_string(),
        1 => "Structural".to_string(),
        2 => "Semantic".to_string(),
        3 => "Concept".to_string(),
        4 => "Wisdom".to_string(),
        -1 => "Forgotten".to_string(),
        _ => format!("Custom L{}", level),
    }
}

/// Group memories by layer, optionally filtering out forgotten and applying from_layer.
/// Returns a HashMap mapping layer number to its memories.
pub(crate) fn group_memories_by_layer(
    memories: &[serde_json::Value],
    show_forgotten: bool,
    from_layer: Option<i32>,
) -> std::collections::HashMap<i32, Vec<serde_json::Value>> {
    let mut by_layer: std::collections::HashMap<i32, Vec<serde_json::Value>> = std::collections::HashMap::new();
    for memory in memories {
        if let serde_json::Value::Object(mem_obj) = memory {
            let meta = mem_obj.get("metadata").and_then(|m| m.as_object());
            if let Some(meta_obj) = meta {
                if !show_forgotten {
                    if let Some(serde_json::Value::String(state)) = meta_obj.get("state") {
                        if state == "Forgotten" {
                            continue;
                        }
                    }
                }
                if let Some(serde_json::Value::Number(layer_num)) = meta_obj.get("layer") {
                    if let Some(layer) = layer_num.as_i64() {
                        let layer_int = layer as i32;
                        if let Some(from) = from_layer {
                            if layer_int < from {
                                continue;
                            }
                        }
                        by_layer.entry(layer_int).or_default().push(memory.clone());
                    }
                }
            }
        }
    }
    by_layer
}

/// Truncate content for display (max 60 chars).
pub(crate) fn truncate_content(content: &str) -> String {
    if content.len() > 60 {
        format!("{}...", &content[..60])
    } else {
        content.to_string()
    }
}

/// Extract the content string from a memory JSON value.
pub(crate) fn extract_content(memory: &serde_json::Value) -> &str {
    if let serde_json::Value::Object(mem_obj) = memory {
        if let Some(serde_json::Value::String(content_str)) = mem_obj.get("content") {
            return content_str.as_str();
        }
    }
    "[no content]"
}

/// Extract the id string from a memory JSON value.
pub(crate) fn extract_id(memory: &serde_json::Value) -> &str {
    if let serde_json::Value::Object(mem_obj) = memory {
        if let Some(serde_json::Value::String(id_str)) = mem_obj.get("id") {
            return id_str.as_str();
        }
    }
    "[no id]"
}

/// Handle the layer-tree command
pub async fn handle_layer_tree(
    system: &System,
    bank: &str,
    from_layer: Option<&i32>,
    max_depth: usize,
    show_ids: bool,
    show_forgotten: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build the payload for list_memories operation (to get all memories for layer tree)
    let mut payload = MemoryOperationPayload::default();
    payload.bank = Some(bank.to_string());
    // No limit to get all memories for the tree

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.list_memories(payload).await {
        Ok(response) => {
            // For layer tree, we want to build and display the hierarchy
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
                    
                    // Group memories by layer
                    let from_val = from_layer.copied();
                    let by_layer = group_memories_by_layer(memories, show_forgotten, from_val);
                    
                    // Print the tree
                    println!("Layer Hierarchy for bank '{}'", bank);
                    println!("{}", "=".repeat(60));
                    println!();
                    
                    let mut levels: Vec<_> = by_layer.keys().collect();
                    levels.sort();
                    
                    for (idx, level) in levels.iter().enumerate() {
                        let is_last_level = idx == levels.len() - 1;
                        let memories_at_level = by_layer.get(level).unwrap();
                        let name = layer_tree_name(**level);
                        
                        // Print layer header
                        if is_last_level {
                            println!(
                                "└── Layer {} ({}) - {} memories",
                                level,
                                name,
                                memories_at_level.len()
                            );
                            print_layer_branch(memories_at_level.clone(), "    ", show_ids, max_depth);
                        } else {
                            println!(
                                "├── Layer {} ({}) - {} memories",
                                level,
                                name,
                                memories_at_level.len()
                            );
                            print_layer_branch(memories_at_level.clone(), "│   ", show_ids, max_depth);
                        }
                    }
                    
                    println!();
                    println!("Legend:");
                    println!("  L0: Raw user content (chunks, documents)");
                    println!("  L1: Structural abstractions (summaries, sections)");
                    println!("  L2: Semantic links (cross-references)");
                    println!("  L3: Concepts (domain theories, principles)");
                    println!("  L4: Wisdom (mental models, paradigms)");
                    println!();
                    println!("Tip: Use --from-layer N to start from layer N");
                    println!("     Use --max-depth N to limit memories shown per layer");
                    println!("     Use --show-ids to display memory UUIDs");
                    println!("     Use --show-forgotten to include forgotten memories");
                } else {
                    // Fall back to regular output formatting
                    crate::output::print_response(&response, OutputFormat::Json)?;
                }
            } else {
                crate::output::print_response(&response, OutputFormat::Json)?;
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}

/// Helper function to print a branch of the layer tree
fn print_layer_branch(
    memories: Vec<serde_json::Value>,
    prefix: &str,
    show_ids: bool,
    max_depth: usize,
) {
    let display_count = std::cmp::min(memories.len(), max_depth);
    
    for (idx, memory) in memories.iter().take(display_count).enumerate() {
        let is_last = idx == display_count - 1 || idx == memories.len() - 1;
        let branch = if is_last { "└──" } else { "├──" };
        
        let content = extract_content(memory);
        let truncated = truncate_content(content);
        
        if show_ids {
            let memory_id = extract_id(memory);
            println!("{} {} {} [{}]", prefix, branch, truncated, memory_id);
        } else {
            println!("{} {} {}", prefix, branch, truncated);
        }
    }
    
    if memories.len() > max_depth {
        let remaining = memories.len() - max_depth;
        println!("{} ... and {} more", prefix, remaining);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_memory(layer: i32, state: &str, content: &str) -> serde_json::Value {
        json!({
            "metadata": {
                "layer": layer,
                "state": state
            },
            "content": content,
            "id": format!("mem-{}-{}", layer, content.len())
        })
    }

    // --- layer_tree_name tests ---

    #[test]
    fn test_layer_tree_name_known() {
        assert_eq!(layer_tree_name(0), "Raw Content");
        assert_eq!(layer_tree_name(1), "Structural");
        assert_eq!(layer_tree_name(2), "Semantic");
        assert_eq!(layer_tree_name(3), "Concept");
        assert_eq!(layer_tree_name(4), "Wisdom");
        assert_eq!(layer_tree_name(-1), "Forgotten");
    }

    #[test]
    fn test_layer_tree_name_custom() {
        assert_eq!(layer_tree_name(5), "Custom L5");
        assert_eq!(layer_tree_name(99), "Custom L99");
    }

    // --- truncate_content tests ---

    #[test]
    fn test_truncate_content_short() {
        assert_eq!(truncate_content("hello"), "hello");
    }

    #[test]
    fn test_truncate_content_exactly_60() {
        let s = "a".repeat(60);
        assert_eq!(truncate_content(&s), s);
    }

    #[test]
    fn test_truncate_content_over_60() {
        let s = "a".repeat(80);
        let expected = format!("{}...", "a".repeat(60));
        assert_eq!(truncate_content(&s), expected);
    }

    // --- extract_content tests ---

    #[test]
    fn test_extract_content_present() {
        let mem = json!({"content": "hello world"});
        assert_eq!(extract_content(&mem), "hello world");
    }

    #[test]
    fn test_extract_content_missing() {
        let mem = json!({"other": "value"});
        assert_eq!(extract_content(&mem), "[no content]");
    }

    #[test]
    fn test_extract_content_non_object() {
        assert_eq!(extract_content(&json!(42)), "[no content]");
    }

    // --- extract_id tests ---

    #[test]
    fn test_extract_id_present() {
        let mem = json!({"id": "abc-123"});
        assert_eq!(extract_id(&mem), "abc-123");
    }

    #[test]
    fn test_extract_id_missing() {
        let mem = json!({"content": "stuff"});
        assert_eq!(extract_id(&mem), "[no id]");
    }

    // --- group_memories_by_layer tests ---

    #[test]
    fn test_group_empty() {
        let result = group_memories_by_layer(&[], false, None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_group_basic_layering() {
        let memories = vec![
            make_memory(0, "Active", "mem a"),
            make_memory(0, "Active", "mem b"),
            make_memory(1, "Active", "mem c"),
            make_memory(2, "Active", "mem d"),
        ];
        let result = group_memories_by_layer(&memories, false, None);
        assert_eq!(result.get(&0).unwrap().len(), 2);
        assert_eq!(result.get(&1).unwrap().len(), 1);
        assert_eq!(result.get(&2).unwrap().len(), 1);
    }

    #[test]
    fn test_group_filters_forgotten_by_default() {
        let memories = vec![
            make_memory(0, "Active", "mem a"),
            make_memory(-1, "Forgotten", "mem b"),
        ];
        let result = group_memories_by_layer(&memories, false, None);
        assert_eq!(result.get(&0).unwrap().len(), 1);
        assert!(result.get(&-1).is_none());
    }

    #[test]
    fn test_group_includes_forgotten_when_requested() {
        let memories = vec![
            make_memory(0, "Active", "mem a"),
            make_memory(-1, "Forgotten", "mem b"),
        ];
        let result = group_memories_by_layer(&memories, true, None);
        assert_eq!(result.get(&0).unwrap().len(), 1);
        assert_eq!(result.get(&-1).unwrap().len(), 1);
    }

    #[test]
    fn test_group_from_layer_filter() {
        let memories = vec![
            make_memory(0, "Active", "mem a"),
            make_memory(1, "Active", "mem b"),
            make_memory(2, "Active", "mem c"),
            make_memory(3, "Active", "mem d"),
        ];
        let result = group_memories_by_layer(&memories, false, Some(2));
        assert!(result.get(&0).is_none());
        assert!(result.get(&1).is_none());
        assert_eq!(result.get(&2).unwrap().len(), 1);
        assert_eq!(result.get(&3).unwrap().len(), 1);
    }

    #[test]
    fn test_group_non_object_values_ignored() {
        let memories = vec![json!("not an object"), json!(42)];
        let result = group_memories_by_layer(&memories, false, None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_group_missing_layer_field_ignored() {
        let memories = vec![json!({"metadata": {"state": "Active"}, "content": "no layer"})];
        let result = group_memories_by_layer(&memories, false, None);
        assert!(result.is_empty());
    }
}