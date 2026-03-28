use llm_mem::operations::MemoryOperationPayload;
use llm_mem::System;
use crate::OutputFormat;

/// Handle the list-sessions command
pub async fn handle_list_sessions(
    system: &System,
    bank: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build the payload for list_document_sessions operation
    let mut payload = MemoryOperationPayload::default();
    payload.bank = Some(bank.to_string());

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