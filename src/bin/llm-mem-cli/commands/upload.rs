use crate::OutputFormat;
use llm_mem::System;
use llm_mem::operations::MemoryOperationPayload;
use std::path::Path;

#[derive(Debug)]
pub struct UploadConfig<'a> {
    pub file_path: &'a Path,
    pub bank: &'a str,
    pub process_immediately: bool,
    pub chunk_size: Option<&'a usize>,
    pub memory_type: Option<&'a str>,
    pub context: Vec<String>,
}

/// Handle the upload command (simple upload with auto-chunking and processing)
pub async fn handle_upload(
    system: &System,
    config: UploadConfig<'_>,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let UploadConfig {
        file_path,
        bank,
        process_immediately,
        chunk_size,
        memory_type,
        context,
    } = config;
    // Check if file exists
    if !file_path.exists() {
        eprintln!("Error: File not found: {}", file_path.display());
        return Ok(());
    }

    // Build the payload for upload_document operation
    let mut payload = MemoryOperationPayload {
        file_path: Some(file_path.to_string_lossy().to_string()),
        process_immediately: Some(process_immediately),
        bank: Some(bank.to_string()),
        ..Default::default()
    };
    if let Some(size) = chunk_size {
        payload.chunk_size = Some(*size);
    }
    if let Some(mt) = memory_type {
        payload.memory_type = Some(mt.to_string());
    }
    if !context.is_empty() {
        payload.context = Some(context);
    }

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.upload_document(payload).await {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
            if response.success {
                // Extract session_id to avoid returning reference to temporary data
                if let Some(data) = &response.data
                    && let Some(session_id_value) = data.get("session_id")
                    && let Some(session_id) = session_id_value.as_str()
                {
                    println!(
                        "Upload started. Use 'doc-status --session-id {}' to check processing status.",
                        session_id
                    );
                }
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
