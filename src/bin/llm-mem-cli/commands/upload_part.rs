use crate::OutputFormat;
use llm_mem::System;
use llm_mem::operations::StoreDocumentPartRequest;
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

    let _bank = bank;
    let content = std::fs::read_to_string(file_path).unwrap_or_default();
    let req = StoreDocumentPartRequest {
        session_id: session_id.to_string(),
        part_index,
        content,
    };

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.store_document_part(req) {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
