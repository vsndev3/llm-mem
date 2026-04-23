use crate::OutputFormat;
use llm_mem::System;
use llm_mem::operations::MemoryOperationPayload;

/// Handle the doc-status command
pub async fn handle_doc_status(
    system: &System,
    session_id: &str,
    bank: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build the payload for status_process_document operation
    let payload = MemoryOperationPayload {
        session_id: Some(session_id.to_string()),
        bank: Some(bank.to_string()),
        ..Default::default()
    };

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.status_process_document(payload) {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
