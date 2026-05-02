use crate::{OutputFormat, SearchMode};
use llm_mem::System;
use llm_mem::operations::QueryRequest;

#[derive(Debug)]
pub struct SearchConfig<'a> {
    pub bank: &'a str,
    pub query: &'a str,
    pub mode: SearchMode,
    pub limit: usize,
    pub case_insensitive: bool,
    pub show_scores: bool,
    pub threshold: Option<f32>,
}

/// Handle the search command
pub async fn handle_search(
    system: &System,
    config: SearchConfig<'_>,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let SearchConfig {
        bank,
        query,
        mode: _mode,
        limit,
        case_insensitive: _case_insensitive,
        show_scores: _show_scores,
        threshold,
    } = config;
    let req = QueryRequest {
        query: query.to_string(),
        bank: Some(bank.to_string()),
        limit,
        similarity_threshold: threshold,
        ..Default::default()
    };

    // For text search, we might need to handle case sensitivity differently
    // but the operations layer should handle that

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.query_memory(req).await {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
