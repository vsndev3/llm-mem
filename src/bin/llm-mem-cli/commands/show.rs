use crate::OutputFormat;
use llm_mem::System;
use llm_mem::operations::MemoryOperationPayload;

/// Handle the show command
pub async fn handle_show(
    system: &System,
    bank: &str,
    memory_id: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build the payload for get_memory operation
    let payload = MemoryOperationPayload {
        memory_id: Some(memory_id.to_string()),
        bank: Some(bank.to_string()),
        ..Default::default()
    };

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.get_memory(payload).await {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
