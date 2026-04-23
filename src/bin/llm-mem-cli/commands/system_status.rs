use crate::OutputFormat;
use crate::output;
use llm_mem::System;
use llm_mem::mcp::{dir_size_bytes, format_bytes};
use llm_mem::operations::MemoryOperationResponse;
use llm_mem::types::Filters;

/// Determine the readiness message based on LLM status fields.
pub(crate) fn determine_readiness(
    llm_available: bool,
    embedding_available: bool,
    state: &str,
) -> (&'static str, bool) {
    let ready = llm_available && embedding_available && state == "ready";
    if ready {
        (
            "READY — System is fully operational. You can store and query memories.",
            true,
        )
    } else if state == "initializing" {
        (
            "PREPARING — Models are loading or downloading. Please wait and call system_status again in a few seconds.",
            false,
        )
    } else {
        (
            "NOT READY — System encountered an error during initialization. Check 'last_error' for details.",
            false,
        )
    }
}

/// Handle the system-status command
pub async fn handle_system_status(
    system: &System,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get LLM status from the bank manager
    let llm_status = system.bank_manager.get_llm_status();

    // Get bank information
    let banks = system.bank_manager.list_banks().await?;
    let bank_names: Vec<String> = banks.iter().map(|b| b.name.clone()).collect();
    let bank_count = banks.len();

    // Determine clear readiness state
    let (readiness, ready_to_use) = determine_readiness(
        llm_status.llm_available,
        llm_status.embedding_available,
        &llm_status.state,
    );

    // Compute real disk usage
    let models_size = dir_size_bytes(&system.models_dir);
    let banks_size = dir_size_bytes(system.bank_manager.banks_dir());

    // Gather layer statistics across all banks
    let mut total_memories: u64 = 0;
    let mut by_layer: std::collections::HashMap<usize, (u64, u64, u64, u64, u64)> =
        std::collections::HashMap::new();
    let mut state_counts: [u64; 4] = [0; 4]; // active, forgotten, processing, invalid
    let mut max_layer: usize = 0;

    for bank_info in &banks {
        if let Ok(bank) = system.bank_manager.get_or_create(&bank_info.name).await
            && let Ok(memories) = bank.list(&Filters::new(), None).await
        {
            for memory in &memories {
                let level = memory.metadata.layer.level as usize;
                let state = memory.metadata.state.as_str();
                total_memories += 1;

                let entry = by_layer.entry(level).or_insert((0, 0, 0, 0, 0));
                entry.0 += 1; // count
                match state {
                    "active" => {
                        entry.1 += 1;
                        state_counts[0] += 1;
                    }
                    "forgotten" => {
                        entry.2 += 1;
                        state_counts[1] += 1;
                    }
                    "processing" => {
                        entry.3 += 1;
                        state_counts[2] += 1;
                    }
                    "invalid" => {
                        entry.4 += 1;
                        state_counts[3] += 1;
                    }
                    _ => {}
                }
                if level > max_layer {
                    max_layer = level;
                }
            }
        }
    }

    let mut by_layer_json = serde_json::Map::new();
    for (level, (count, active, forgotten, processing, invalid)) in &by_layer {
        by_layer_json.insert(
            level.to_string(),
            serde_json::json!({
                "count": count,
                "active": active,
                "forgotten": forgotten,
                "processing": processing,
                "invalid": invalid,
            }),
        );
    }

    // Get abstraction pipeline status
    let pipeline_status = system.bank_manager.get_pipeline_status().await;

    // Build the response
    let mut guide = serde_json::Map::new();
    guide.insert("ready_to_use".to_string(), serde_json::json!(ready_to_use));
    guide.insert(
        "readiness_message".to_string(),
        serde_json::json!(readiness),
    );
    guide.insert(
        "system_status".to_string(),
        serde_json::to_value(&llm_status)?,
    );

    guide.insert(
        "disk_usage".to_string(),
        serde_json::json!({
            "models_dir": system.models_dir.display().to_string(),
            "models_size_bytes": models_size,
            "models_size_human": format_bytes(models_size),
            "banks_dir": system.bank_manager.banks_dir().display().to_string(),
            "banks_size_bytes": banks_size,
            "banks_size_human": format_bytes(banks_size),
            "total_size_bytes": models_size + banks_size,
            "total_size_human": format_bytes(models_size + banks_size),
        }),
    );

    guide.insert(
        "active_banks".to_string(),
        serde_json::json!({
            "count": bank_count,
            "names": bank_names,
        }),
    );

    guide.insert(
        "layer_statistics".to_string(),
        serde_json::json!({
            "total_memories": total_memories,
            "by_layer": by_layer_json,
            "max_layer": max_layer,
            "state_counts": {
                "active": state_counts[0],
                "forgotten": state_counts[1],
                "processing": state_counts[2],
                "invalid": state_counts[3],
            },
        }),
    );

    guide.insert(
        "abstraction_pipeline".to_string(),
        serde_json::json!({
            "enabled": pipeline_status.enabled,
            "workers_running": pipeline_status.workers_running,
            "pending_l0_count": pipeline_status.pending_l0_count,
            "pending_l1_count": pipeline_status.pending_l1_count,
            "pending_l2_count": pipeline_status.pending_l2_count,
            "config": {
                "min_memories_for_l1": pipeline_status.config.min_memories_for_l1,
                "l1_processing_delay_secs": pipeline_status.config.l1_processing_delay_secs,
                "max_concurrent_tasks": pipeline_status.config.max_concurrent_tasks,
            },
        }),
    );

    guide.insert("usage_guide".to_string(), serde_json::json!("llm-mem is a persistent semantic knowledge index using a layered memory architecture (L0-L4+)."));

    let response = MemoryOperationResponse::success_with_data(
        "System status retrieved",
        serde_json::json!(guide),
    );

    output::print_response(&response, format)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determine_readiness_fully_ready() {
        let (msg, ready) = determine_readiness(true, true, "ready");
        assert!(ready);
        assert!(msg.contains("READY"));
        assert!(msg.contains("fully operational"));
    }

    #[test]
    fn test_determine_readiness_initializing() {
        let (msg, ready) = determine_readiness(false, false, "initializing");
        assert!(!ready);
        assert!(msg.contains("PREPARING"));
    }

    #[test]
    fn test_determine_readiness_initializing_partial() {
        let (msg, ready) = determine_readiness(true, false, "initializing");
        assert!(!ready);
        assert!(msg.contains("PREPARING"));
    }

    #[test]
    fn test_determine_readiness_not_ready_error() {
        let (msg, ready) = determine_readiness(false, false, "error");
        assert!(!ready);
        assert!(msg.contains("NOT READY"));
    }

    #[test]
    fn test_determine_readiness_llm_missing() {
        let (msg, ready) = determine_readiness(false, true, "ready");
        assert!(!ready);
        assert!(msg.contains("NOT READY"));
    }

    #[test]
    fn test_determine_readiness_embedding_missing() {
        let (msg, ready) = determine_readiness(true, false, "ready");
        assert!(!ready);
        assert!(msg.contains("NOT READY"));
    }
}
