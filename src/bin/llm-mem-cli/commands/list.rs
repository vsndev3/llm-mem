use crate::OutputFormat;
use llm_mem::System;
use llm_mem::operations::ListRequest;

/// Handle the list command
pub async fn handle_list(
    system: &System,
    bank: &str,
    limit: usize,
    format: OutputFormat,
    memory_type: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let req = ListRequest {
        bank: Some(bank.to_string()),
        limit,
        memory_type: memory_type.map(|s| s.to_string()),
        ..Default::default()
    };

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.list_memories(req).await {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
