use llm_mem::operations::MemoryOperationPayload;
use llm_mem::System;
use crate::{OutputFormat, SearchMode};

/// Handle the search command
pub async fn handle_search(
    system: &System,
    bank: &str,
    query: &str,
    _mode: SearchMode,
    limit: usize,
    _case_insensitive: bool,
    _show_scores: bool,
    threshold: Option<f32>,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build the payload for query_memory operation
    let payload = MemoryOperationPayload {
        query: Some(query.to_string()),
        bank: Some(bank.to_string()),
        limit: Some(limit),
        similarity_threshold: threshold,
        ..Default::default()
    };
    
    // For text search, we might need to handle case sensitivity differently
    // but the operations layer should handle that
    
    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.query_memory(payload).await {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}