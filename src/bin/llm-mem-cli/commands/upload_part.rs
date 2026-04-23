use crate::OutputFormat;
use llm_mem::System;
use llm_mem::operations::MemoryOperationPayload;
use std::path::Path;

/// Handle the upload-part command (upload a document part)
pub async fn handle_upload_part(
    system: &System,
    session_id: &str,
    part_index: usize,
    file_path: &Path,
    bank: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Check if file exists
    if !file_path.exists() {
        eprintln!("Error: File not found: {}", file_path.display());
        return Ok(());
    }

    // Build the payload for store_document_part operation
    let payload = MemoryOperationPayload {
        session_id: Some(session_id.to_string()),
        part_index: Some(part_index),
        file_path: Some(file_path.to_string_lossy().to_string()),
        bank: Some(bank.to_string()),
        ..Default::default()
    };

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.store_document_part(payload) {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
