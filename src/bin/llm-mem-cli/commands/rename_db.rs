use llm_mem::memory_bank::MemoryBankManager;

/// Handle the rename-db command
pub async fn handle_db_rename(
    bank_manager: &MemoryBankManager,
    old_name: &str,
    new_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    match bank_manager.rename_bank(old_name, new_name).await {
        Ok(()) => {
            println!("Successfully renamed bank '{}' to '{}'", old_name, new_name);
            Ok(())
        }
        Err(e) => Err(format!("Failed to rename bank: {}", e).into()),
    }
}
