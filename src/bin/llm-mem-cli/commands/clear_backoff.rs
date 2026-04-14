use llm_mem::operations::MemoryOperationResponse;
use llm_mem::System;
use crate::output;
use crate::OutputFormat;

/// Clear abstraction backoff timers to force retry failed abstractions.
pub async fn handle_clear_backoff(
    system: &System,
    bank: &str,
    layer: Option<i32>,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = if let Some(layer_level) = layer {
        // Clear backoff for a specific layer in a specific bank
        let bank_manager = &system.bank_manager;
        let bank_obj = bank_manager.get_or_create(bank).await?;

        // Get all memories at the specified layer and clear their backoff timers
        let mut filters = llm_mem::types::Filters::new();
        filters.min_layer_level = Some(layer_level);
        filters.max_layer_level = Some(layer_level);
        let results = bank_obj.list(&filters, None).await?;

        let mut cleared_count = 0;
        for m in results {
            if m.metadata.abstraction_retry_after.is_some() || m.metadata.last_abstraction_failure.is_some() {
                let mut memory = m;
                memory.metadata.abstraction_retry_after = None;
                memory.metadata.last_abstraction_failure = None;
                bank_obj.update_memory(&memory).await?;
                cleared_count += 1;
            }
        }

        MemoryOperationResponse::success_with_data(
            "Backoff timers cleared",
            serde_json::json!({
                "message": format!(
                    "Cleared backoff timers for {} memories at L{} in bank '{}'",
                    cleared_count, layer_level, bank
                ),
                "bank": bank,
                "layer": layer_level,
                "cleared_count": cleared_count,
            }),
        )
    } else {
        // Clear backoff across all layers (L0, L1, L2) and all banks
        let results_l0 = system.bank_manager.clear_abstraction_backoff(0).await?;
        let results_l1 = system.bank_manager.clear_abstraction_backoff(1).await?;
        let results_l2 = system.bank_manager.clear_abstraction_backoff(2).await?;

        let mut total_cleared = 0usize;
        let mut by_layer = serde_json::Map::new();

        for (bank_name, count) in &results_l0 {
            total_cleared += count;
            let entry: &mut serde_json::Map<String, serde_json::Value> = by_layer
                .entry(bank_name.clone())
                .or_insert_with(|| serde_json::json!({}))
                .as_object_mut()
                .unwrap();
            entry.insert("L0".to_string(), serde_json::json!(*count));
        }
        for (bank_name, count) in &results_l1 {
            total_cleared += count;
            let entry: &mut serde_json::Map<String, serde_json::Value> = by_layer
                .entry(bank_name.clone())
                .or_insert_with(|| serde_json::json!({}))
                .as_object_mut()
                .unwrap();
            entry.insert("L1".to_string(), serde_json::json!(*count));
        }
        for (bank_name, count) in &results_l2 {
            total_cleared += count;
            let entry: &mut serde_json::Map<String, serde_json::Value> = by_layer
                .entry(bank_name.clone())
                .or_insert_with(|| serde_json::json!({}))
                .as_object_mut()
                .unwrap();
            entry.insert("L2".to_string(), serde_json::json!(*count));
        }

        MemoryOperationResponse::success_with_data(
            "Backoff timers cleared",
            serde_json::json!({
                "message": format!(
                    "Cleared backoff timers for {} memories across all banks",
                    total_cleared
                ),
                "total_cleared": total_cleared,
                "by_layer": by_layer,
            }),
        )
    };

    output::print_response(&response, format)?;
    Ok(())
}
