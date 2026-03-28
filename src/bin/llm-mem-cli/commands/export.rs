use llm_mem::operations::MemoryOperationPayload;
use std::path::Path;
use llm_mem::System;
use crate::OutputFormat;

/// Handle the export command
pub async fn handle_export(
    system: &System,
    bank_param: &str,
    output: Option<&Path>,
    pretty: bool,
    format_param: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build the payload for export operation (we'll use list_memories with no limit)
    let mut payload = MemoryOperationPayload::default();
    payload.bank = Some(bank_param.to_string());
    // No limit to get all memories

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.list_memories(payload).await {
        Ok(response) => {
            // For export, we want to format specially
            if let Some(data) = &response.data {
                if format_param == OutputFormat::Json {
                    let json_str = if pretty {
                        serde_json::to_string_pretty(data)?
                    } else {
                        serde_json::to_string(data)?
                    };
                    
                    if let Some(output_path) = output {
                        tokio::fs::write(output_path, json_str).await?;
                        println!("Exported to {}", output_path.display());
                    } else {
                        println!("{}", json_str);
                    }
                } else {
                    // Fall back to regular output formatting for other formats
                    crate::output::print_response(&response, format_param)?;
                }
            } else {
                crate::output::print_response(&response, format_param)?;
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}