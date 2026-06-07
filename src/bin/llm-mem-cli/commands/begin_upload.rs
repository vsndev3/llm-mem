use crate::OutputFormat;
use llm_mem::System;
use llm_mem::operations::BeginStoreDocumentRequest;
use serde_json::from_str;

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
    let mut req = BeginStoreDocumentRequest {
        file_name: file_name.to_string(),
        file_type: None,
        total_size,
        md5sum: None,
        memory_type: "conversation".to_string(),
        topics: None,
        context: if context.is_empty() {
            None
        } else {
            Some(context)
        },
        bank: Some(bank.to_string()),
        ..Default::default()
    };
    if let Some(mt) = mime_type {
        req.file_type = Some(mt.to_string());
    }
    if let Some(mt) = memory_type {
        req.memory_type = mt.to_string();
    }

    // Parse custom metadata if provided
    if let Some(metadata_str) = metadata
        && let Ok(metadata_json) = from_str::<serde_json::Value>(metadata_str)
        && let serde_json::Value::Object(map) = metadata_json
    {
        // Convert serde_json::Map to HashMap
        let hashmap: std::collections::HashMap<String, serde_json::Value> =
            map.into_iter().collect();
        req.metadata = Some(hashmap);
    }

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.begin_store_document(req) {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
            if response.success {
                // Extract session_id to avoid returning reference to temporary data
                if let Some(data) = &response.data
                    && let Some(session_id_value) = data.get("session_id")
                    && let Some(session_id) = session_id_value.as_str()
                {
                    println!("Document session started with ID: {}", session_id);
                    println!(
                        "Use 'upload-part' to upload parts and 'process-document' when complete."
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
