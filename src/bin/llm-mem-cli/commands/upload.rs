use crate::OutputFormat;
use llm_mem::System;
use llm_mem::operations::UploadDocumentRequest;
use std::path::Path;

#[derive(Debug)]
pub struct UploadConfig<'a> {
    pub file_path: &'a Path,
    pub bank: &'a str,
    pub process_immediately: bool,
    pub chunk_size: Option<&'a usize>,
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
        context,
    } = config;
    // Check if file exists
    if !file_path.exists() {
        eprintln!("Error: File not found: {}", file_path.display());
        return Ok(());
    }

    // Build the request for upload_document operation
    let req = UploadDocumentRequest {
        file_path: file_path.to_string_lossy().to_string(),
        file_name: None,
        mime_type: None,
        topics: None,
        context: if context.is_empty() {
            None
        } else {
            Some(context)
        },
        user_id: None,
        agent_id: None,
        chunk_size: chunk_size.copied(),
        process_immediately,
        bank: Some(bank.to_string()),
        event_at: None,
    };

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.upload_document(req).await {
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
