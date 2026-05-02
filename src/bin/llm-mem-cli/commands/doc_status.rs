use crate::OutputFormat;
use llm_mem::System;
use llm_mem::operations::StatusProcessDocumentRequest;

/// Handle the doc-status command
pub async fn handle_doc_status(
    system: &System,
    session_id: &str,
    bank: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let _bank = bank;
    let req = StatusProcessDocumentRequest {
        session_id: session_id.to_string(),
    };

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.status_process_document(req) {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
