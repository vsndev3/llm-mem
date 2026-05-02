use crate::OutputFormat;
use llm_mem::System;
use llm_mem::operations::ListDocumentSessionsRequest;

/// Handle the list-sessions command
pub async fn handle_list_sessions(
    system: &System,
    bank: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let req = ListDocumentSessionsRequest {
        bank: Some(bank.to_string()),
    };

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.list_document_sessions(req) {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
