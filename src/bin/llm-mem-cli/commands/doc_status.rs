use llm_mem::operations::MemoryOperationPayload;
use llm_mem::System;
use crate::OutputFormat;

/// Handle the doc-status command
pub async fn handle_doc_status(
    system: &System,
    session_id: &str,
    bank: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build the payload for status_process_document operation
    let mut payload = MemoryOperationPayload::default();
    payload.session_id = Some(session_id.to_string());
    payload.bank = Some(bank.to_string());

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