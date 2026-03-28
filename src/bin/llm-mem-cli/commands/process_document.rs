use llm_mem::operations::MemoryOperationPayload;
use llm_mem::System;
use crate::OutputFormat;

/// Handle the process-document command
pub async fn handle_process_document(
    system: &System,
    session_id: &str,
    partial_closure: bool,
    bank: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build the payload for process_document operation
    let mut payload = MemoryOperationPayload::default();
    payload.session_id = Some(session_id.to_string());
    payload.partial_closure = Some(partial_closure);
    payload.bank = Some(bank.to_string());

    // Execute the operation
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