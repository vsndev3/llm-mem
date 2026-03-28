use llm_mem::operations::MemoryOperationPayload;
use llm_mem::System;
use crate::OutputFormat;

/// Handle the list command
pub async fn handle_list(
    system: &System,
    bank: &str,
    limit: usize,
    format: OutputFormat,
    memory_type: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build the payload for list_memories operation
    let mut payload = MemoryOperationPayload::default();
    payload.bank = Some(bank.to_string());
    payload.limit = Some(limit);
    
    if let Some(mt) = memory_type {
        payload.memory_type = Some(mt.to_string());
    }

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.list_memories(payload).await {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}