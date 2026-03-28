use llm_mem::operations::MemoryOperationResponse;
use llm_mem::System;
use crate::OutputFormat;

/// Handle the list-banks command
pub async fn handle_list_banks(system: &System, format: OutputFormat) -> Result<(), Box<dyn std::error::Error>> {
    // We need to access the bank manager directly for this operation
    let banks = system.bank_manager.list_banks().await?;
    
    // Format as a simple response
    let mut bank_infos = Vec::new();
    for bank in &banks {
        bank_infos.push(serde_json::json!({
            "name": bank.name,
            "path": bank.path,
            "memory_count": bank.memory_count,
            "description": bank.description,
            "loaded": bank.loaded
        }));
    }
    
    let response = MemoryOperationResponse::success_with_data(
        "Memory banks listed successfully",
        serde_json::json!({
            "count": banks.len(),
            "banks": bank_infos
        })
    );
    
    crate::output::print_response(&response, format)?;
    Ok(())
}