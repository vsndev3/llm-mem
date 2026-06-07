use crate::OutputFormat;
use llm_mem::MemoryOperations;
use llm_mem::System;
use llm_mem::operations::GetRequest;

/// Handle the show command
pub async fn handle_show(
    system: &System,
    bank: &str,
    memory_id: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let req = GetRequest {
        memory_id: memory_id.to_string(),
        bank: Some(bank.to_string()),
    };

    let manager = system
        .bank_manager
        .resolve_bank(Some(bank))
        .await
        .map_err(|e| format!("Failed to resolve bank: {}", e))?;
    let ops = MemoryOperations::new(manager, None, None, 1000);
    match ops.get_memory(req).await {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
