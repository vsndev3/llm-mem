use llm_mem::operations::MemoryOperationPayload;
use serde_json::from_str;
use llm_mem::System;
use crate::OutputFormat;

/// Handle the begin-upload command (start document storage session)
pub async fn handle_begin_upload(
    system: &System,
    file_name: &str,
    total_size: usize,
    mime_type: Option<&str>,
    bank: &str,
    memory_type: Option<&str>,
    context: Vec<String>,
    metadata: Option<&str>,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build the payload for begin_store_document operation
    let mut payload = MemoryOperationPayload::default();
    payload.file_name = Some(file_name.to_string());
    payload.total_size = Some(total_size);
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
    if let Some(metadata_str) = metadata {
        if let Ok(metadata_json) = from_str::<serde_json::Value>(metadata_str) {
            if let serde_json::Value::Object(map) = metadata_json {
                // Convert serde_json::Map to HashMap
                let hashmap: std::collections::HashMap<String, serde_json::Value> = map.into_iter().collect();
                payload.metadata = Some(hashmap);
            }
        }
    }
    
    payload.bank = Some(bank.to_string());

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.begin_store_document(payload) {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
            if response.success {
                // Extract session_id to avoid returning reference to temporary data
                if let Some(data) = &response.data {
                    if let Some(session_id_value) = data.get("session_id") {
                        if let Some(session_id) = session_id_value.as_str() {
                            println!("Document session started with ID: {}", session_id);
                            println!("Use 'upload-part' to upload parts and 'process-document' when complete.");
                        }
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}