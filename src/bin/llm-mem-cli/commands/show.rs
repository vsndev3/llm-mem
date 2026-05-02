use crate::OutputFormat;
use llm_mem::System;
use llm_mem::operations::GetRequest;

/// Handle the show command
pub async fn handle_show(
    system: &System,
    bank: &str,
    memory_id: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let req = GetRequest {
        memory_id: memory_id.to_string(),
        bank: Some(bank.to_string()),
    };

    // Execute the operation
    let operations = system.operations.lock().await;
    match operations.get_memory(req).await {
        Ok(response) => {
            crate::output::print_response(&response, format)?;
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
