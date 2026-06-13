use crate::OutputFormat;
use llm_mem::MemoryOperations;
use llm_mem::System;
use llm_mem::operations::ListRequest;

/// Handle the list command
pub async fn handle_list(
    system: &System,
    bank: &str,
    limit: usize,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let req = ListRequest {
        bank: Some(bank.to_string()),
        limit,
        ..Default::default()
    };

    let manager = system
        .bank_manager
        .resolve_bank(Some(bank))
        .await
        .map_err(|e| format!("Failed to resolve bank: {}", e))?;
    let ops = MemoryOperations::new(manager, None, None, 1000);
    match ops.list_memories(req).await {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
