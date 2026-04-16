use llm_mem::operations::MemoryOperationPayload;
use serde_json::from_str;
use llm_mem::System;
use crate::OutputFormat;

#[derive(Debug)]
pub struct BeginUploadConfig<'a> {
    pub file_name: &'a str,
    pub total_size: usize,
    pub mime_type: Option<&'a str>,
    pub bank: &'a str,
    pub memory_type: Option<&'a str>,
    pub context: Vec<String>,
    pub metadata: Option<&'a str>,
}

/// Handle the begin-upload command (start document storage session)
pub async fn handle_begin_upload(
    system: &System,
    config: BeginUploadConfig<'_>,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let BeginUploadConfig {
        file_name,
        total_size,
        mime_type,
        bank,
        memory_type,
        context,
        metadata,
    } = config;
    // Build the payload for begin_store_document operation
    let mut payload = MemoryOperationPayload {
        file_name: Some(file_name.to_string()),
        total_size: Some(total_size),
        bank: Some(bank.to_string()),
        ..Default::default()
    };
    if let Some(mt) = mime_type {
        payload.mime_type = Some(mt.to_string());
    }
    if let Some(mt) = memory_type {
        payload.memory_type = Some(mt.to_string());
    }
    if !context.is_empty() {
        payload.context = Some(context);
    }
    
    // Parse custom metadata if provided
    if let Some(metadata_str) = metadata
        && let Ok(metadata_json) = from_str::<serde_json::Value>(metadata_str)
            && let serde_json::Value::Object(map) = metadata_json {
                // Convert serde_json::Map to HashMap
                let hashmap: std::collections::HashMap<String, serde_json::Value> = map.into_iter().collect();
                payload.metadata = Some(hashmap);
            }

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.begin_store_document(payload) {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
            if response.success {
                // Extract session_id to avoid returning reference to temporary data
                if let Some(data) = &response.data
                    && let Some(session_id_value) = data.get("session_id")
                        && let Some(session_id) = session_id_value.as_str() {
                            println!("Document session started with ID: {}", session_id);
                            println!("Use 'upload-part' to upload parts and 'process-document' when complete.");
                        }
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}