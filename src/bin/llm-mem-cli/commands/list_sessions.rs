use crate::OutputFormat;
use llm_mem::System;
use llm_mem::operations::MemoryOperationPayload;

/// Handle the list-sessions command
pub async fn handle_list_sessions(
    system: &System,
    bank: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build the payload for list_document_sessions operation
    let payload = MemoryOperationPayload {
        bank: Some(bank.to_string()),
        ..Default::default()
    };

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.list_document_sessions(payload) {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
