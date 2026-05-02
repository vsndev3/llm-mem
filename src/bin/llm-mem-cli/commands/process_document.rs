use crate::OutputFormat;
use llm_mem::System;
use llm_mem::operations::ProcessDocumentRequest;

/// Handle the process-document command
pub async fn handle_process_document(
    system: &System,
    session_id: &str,
    partial_closure: bool,
    bank: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let payload = ProcessDocumentRequest {
        session_id: session_id.to_string(),
        partial_closure,
    };

    let _bank = bank;
    let operations = system.operations.lock().await;
    match operations.process_document(payload).await {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
