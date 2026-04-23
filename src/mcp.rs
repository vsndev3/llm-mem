use anyhow::Result;
use rmcp::{
    RoleServer, ServerHandler,
    model::{
        CallToolRequestParam, CallToolResult, Content, ErrorData, ListToolsResult,
        PaginatedRequestParam, ServerCapabilities, ServerInfo, Tool,
    },
    service::RequestContext,
};
use serde_json::{Map, json};
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};

use crate::{
    config::Config,
    llm::create_llm_client,
    memory_bank::MemoryBankManager,
    operations::{
        MemoryOperations, OperationError, get_mcp_tool_definitions, get_operation_error_message,
        map_mcp_arguments_to_payload, operation_error_to_mcp_error_code,
    },
    types::Filters,
};

/// Service for handling MCP tool calls related to memory management.
///
/// Supports multiple named "memory banks" — each bank is an isolated memory
/// store with its own database file. Tools accept an optional `bank` parameter;
/// if omitted, the `"default"` bank is used.
pub struct MemoryMcpService {
    bank_manager: MemoryBankManager,
    agent_id: Option<String>,
    default_limit: usize,
    models_dir: PathBuf,
}

impl MemoryMcpService {
    /// Create a new memory MCP service with default config path
    pub async fn new() -> Result<Self> {
        Self::with_agent(None).await
    }

    /// Create a new memory MCP service with default config path and optional agent.
    ///
    /// If no config file is found, defaults to the local backend
    /// (llama.cpp + fastembed) with models in `./llm-mem-models/`.
    pub async fn with_agent(agent_id: Option<String>) -> Result<Self> {
        match Self::find_default_config_path() {
            Some(config_path) => {
                info!("Found config at: {:?}", config_path);
                Self::with_config_path_and_agent(config_path, agent_id).await
            }
            None => {
                info!("No config file found, using local backend defaults");
                let config = Config::default();
                Self::with_config_and_agent(config, agent_id).await
            }
        }
    }

    /// Create a new memory MCP service with specific config path
    pub async fn with_config_path<P: AsRef<Path> + Clone + std::fmt::Debug>(
        path: P,
    ) -> Result<Self> {
        Self::with_config_path_and_agent(path, None).await
    }

    /// Create a new memory MCP service with specific config path and agent
    pub async fn with_config_path_and_agent<P: AsRef<Path> + Clone + std::fmt::Debug>(
        path: P,
        agent_id: Option<String>,
    ) -> Result<Self> {
        let config = Config::load(path.clone())?;
        info!("Loaded configuration from: {:?}", path);
        Self::with_config_and_agent(config, agent_id).await
    }

    /// Create a new memory MCP service from a pre-built Config
    pub async fn with_config_and_agent(config: Config, agent_id: Option<String>) -> Result<Self> {
        let backend = config.effective_backend();
        let llm_client = create_llm_client(&config).await?;
        info!("Initialized LLM client (backend: {:?})", backend);

        let banks_dir = PathBuf::from(&config.vector_store.banks_dir);
        let bank_manager = MemoryBankManager::new(
            banks_dir,
            llm_client,
            config.vector_store.clone(),
            config.memory.clone(),
        )?;
        info!("Initialized memory bank manager");

        bank_manager.default_bank().await?;
        info!("Default memory bank loaded");

        // Start the abstraction pipeline for progressive layer creation
        bank_manager.start_abstraction_pipeline().await.ok();
        info!("Abstraction pipeline initialization requested");

        let service = Self {
            bank_manager,
            agent_id,
            default_limit: 100,
            models_dir: PathBuf::from(config.llm.models_dir.clone()),
        };

        // Startup recovery:
        // 1. First, try to resume any sessions that were interrupted in 'processing' state
        service.auto_resume_sessions().await;

        // 2. Then, cleanup any sessions that are truly stalled (30 mins timeout)
        let _ = service.bank_manager.cleanup_stalled_sessions(1800).await;

        Ok(service)
    }

    /// Startup recovery: resume interrupted uploads and processing
    async fn auto_resume_sessions(&self) {
        if let Ok(banks) = self.bank_manager.list_banks().await {
            for bank_info in banks {
                let bank_name = &bank_info.name;

                // Try to resolve operations for this bank to see if there are interrupted sessions
                if let Ok(ops) = self.resolve_operations_with_sessions(Some(bank_name)).await
                    && let Ok(response) =
                        ops.list_document_sessions(
                            crate::operations::MemoryOperationPayload::default(),
                        )
                    && let Some(sessions_val) =
                        response.data.and_then(|d| d.get("sessions").cloned())
                    && let Ok(sessions) = serde_json::from_value::<
                        Vec<crate::document_session::DocumentSession>,
                    >(sessions_val)
                {
                    for session in sessions {
                        match session.status {
                            crate::document_session::SessionStatus::Processing => {
                                info!(
                                    "Found stalled session {} in bank {} (status: Processing), resuming",
                                    session.session_id, bank_name
                                );

                                let payload = crate::operations::MemoryOperationPayload {
                                    session_id: Some(session.session_id.clone()),
                                    bank: Some(bank_name.clone()),
                                    partial_closure: Some(true), // Allow processing even if part count differs
                                    ..Default::default()
                                };

                                // Trigger re-processing (will auto-reset status if stale)
                                if let Err(e) = ops.process_document(payload).await {
                                    error!(
                                        "Failed to auto-resume session {}: {}",
                                        session.session_id, e
                                    );
                                }
                            }
                            crate::document_session::SessionStatus::Uploading => {
                                // Resume interrupted upload if we have file info and MD5
                                let file_path = session
                                    .metadata
                                    .custom_metadata
                                    .as_ref()
                                    .and_then(|m| m.get("file_path").and_then(|v| v.as_str()));

                                let expected_md5 = session.metadata.md5sum.as_deref();

                                if let Some(file_path) = file_path {
                                    let path = std::path::Path::new(file_path);
                                    if path.exists() {
                                        // Verify file hasn't changed
                                        let content = match std::fs::read_to_string(path) {
                                            Ok(c) => c,
                                            Err(e) => {
                                                warn!(
                                                    "Cannot read file for resume {}: {}",
                                                    file_path, e
                                                );
                                                continue;
                                            }
                                        };

                                        let actual_md5 = format!("{:x}", md5::compute(&content));
                                        if let Some(expected_md5) = expected_md5
                                            && actual_md5 != expected_md5
                                        {
                                            warn!(
                                                "File {} changed since upload started (MD5 mismatch), skipping resume",
                                                file_path
                                            );
                                            continue;
                                        }

                                        info!(
                                            "Found interrupted upload {} in bank {} (status: Uploading), resuming with file: {}",
                                            session.session_id, bank_name, file_path
                                        );

                                        let payload = crate::operations::MemoryOperationPayload {
                                            session_id: Some(session.session_id.clone()),
                                            bank: Some(bank_name.clone()),
                                            file_path: Some(file_path.to_string()),
                                            process_immediately: Some(true),
                                            ..Default::default()
                                        };

                                        // Re-trigger upload (will skip already uploaded parts)
                                        if let Err(e) = ops.upload_document(payload).await {
                                            error!(
                                                "Failed to auto-resume upload session {}: {}",
                                                session.session_id, e
                                            );
                                        }
                                    } else {
                                        warn!(
                                            "File {} for session {} no longer exists, marking as failed",
                                            file_path, session.session_id
                                        );
                                        let _ = ops.cancel_process_document(
                                            crate::operations::MemoryOperationPayload {
                                                session_id: Some(session.session_id.clone()),
                                                bank: Some(bank_name.clone()),
                                                ..Default::default()
                                            },
                                        );
                                    }
                                } else {
                                    warn!(
                                        "Upload session {} has no file_path metadata, cannot resume",
                                        session.session_id
                                    );
                                }
                            }
                            _ => {} // Ignore completed, failed, cancelled
                        }
                    }
                }
            }
        }
    }

    /// Resolve the bank-aware MemoryOperations for a tool call.
    async fn resolve_operations(
        &self,
        bank_name: Option<&str>,
    ) -> Result<MemoryOperations, ErrorData> {
        let manager = self
            .bank_manager
            .resolve_bank(bank_name)
            .await
            .map_err(|e| ErrorData {
                code: rmcp::model::ErrorCode(-32603),
                message: format!("Failed to resolve memory bank: {}", e).into(),
                data: None,
            })?;

        Ok(MemoryOperations::new(
            manager,
            None,
            self.agent_id.clone(),
            self.default_limit,
        ))
    }

    /// Resolve MemoryOperations with session manager for document operations.
    async fn resolve_operations_with_sessions(
        &self,
        bank_name: Option<&str>,
    ) -> Result<MemoryOperations, ErrorData> {
        let (manager, session_manager) = self
            .bank_manager
            .resolve_bank_with_sessions(bank_name)
            .await
            .map_err(|e| ErrorData {
                code: rmcp::model::ErrorCode(-32603),
                message: format!("Failed to resolve memory bank: {}", e).into(),
                data: None,
            })?;

        Ok(MemoryOperations::with_session_manager(
            manager,
            session_manager,
            None,
            self.agent_id.clone(),
            self.default_limit,
        ))
    }

    /// Tool implementation for storing a memory
    async fn store_memory(
        &self,
        arguments: &Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, ErrorData> {
        let payload = map_mcp_arguments_to_payload(arguments, &self.agent_id);
        let ops = self.resolve_operations(payload.bank.as_deref()).await?;
        match ops.store_memory(payload).await {
            Ok(response) => {
                // Notify pipeline for immediate cascade processing
                self.bank_manager.notify_new_memory().await;
                let json = serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to serialize response: {}", e).into(),
                    data: None,
                })?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => {
                error!("Failed to store memory: {}", e);
                Err(self.operation_error_to_mcp_error(e))
            }
        }
    }

    /// Tool implementation for adding memory from conversation
    async fn add_memory(
        &self,
        arguments: &Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, ErrorData> {
        let payload = map_mcp_arguments_to_payload(arguments, &self.agent_id);
        let ops = self.resolve_operations(payload.bank.as_deref()).await?;
        match ops.add_memory(payload).await {
            Ok(response) => {
                // Notify pipeline for immediate cascade processing
                self.bank_manager.notify_new_memory().await;
                let json = serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to serialize response: {}", e).into(),
                    data: None,
                })?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => {
                error!("Failed to add memory: {}", e);
                Err(self.operation_error_to_mcp_error(e))
            }
        }
    }

    /// Tool implementation for updating a memory
    async fn update_memory(
        &self,
        arguments: &Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, ErrorData> {
        let payload = map_mcp_arguments_to_payload(arguments, &self.agent_id);
        let ops = self.resolve_operations(payload.bank.as_deref()).await?;
        match ops.update_memory(payload).await {
            Ok(response) => {
                let json = serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to serialize response: {}", e).into(),
                    data: None,
                })?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => {
                error!("Failed to update memory: {}", e);
                Err(self.operation_error_to_mcp_error(e))
            }
        }
    }

    /// Tool implementation for querying memories
    async fn query_memory(
        &self,
        arguments: &Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, ErrorData> {
        let payload = map_mcp_arguments_to_payload(arguments, &self.agent_id);
        let ops = self.resolve_operations(payload.bank.as_deref()).await?;
        match ops.query_memory(payload).await {
            Ok(response) => {
                let json = serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to serialize response: {}", e).into(),
                    data: None,
                })?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => {
                error!("Failed to query memories: {}", e);
                Err(self.operation_error_to_mcp_error(e))
            }
        }
    }

    /// Tool implementation for listing memories
    async fn list_memories(
        &self,
        arguments: &Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, ErrorData> {
        let payload = map_mcp_arguments_to_payload(arguments, &self.agent_id);
        let ops = self.resolve_operations(payload.bank.as_deref()).await?;
        match ops.list_memories(payload).await {
            Ok(response) => {
                let json = serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to serialize response: {}", e).into(),
                    data: None,
                })?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => {
                error!("Failed to list memories: {}", e);
                Err(self.operation_error_to_mcp_error(e))
            }
        }
    }

    /// Tool implementation for getting a specific memory by ID
    async fn get_memory(
        &self,
        arguments: &Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, ErrorData> {
        let payload = map_mcp_arguments_to_payload(arguments, &self.agent_id);
        let ops = self.resolve_operations(payload.bank.as_deref()).await?;
        match ops.get_memory(payload).await {
            Ok(response) => {
                let json = serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to serialize response: {}", e).into(),
                    data: None,
                })?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => {
                error!("Failed to get memory: {}", e);
                Err(self.operation_error_to_mcp_error(e))
            }
        }
    }

    /// Tool implementation for navigating the abstraction hierarchy
    async fn navigate_memory(
        &self,
        arguments: &Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, ErrorData> {
        let payload = map_mcp_arguments_to_payload(arguments, &self.agent_id);
        let ops = self.resolve_operations(payload.bank.as_deref()).await?;
        match ops.navigate_memory(payload).await {
            Ok(response) => {
                let json = serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to serialize response: {}", e).into(),
                    data: None,
                })?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => {
                error!("Failed to navigate memory: {}", e);
                Err(self.operation_error_to_mcp_error(e))
            }
        }
    }

    /// Tool implementation for listing memory banks
    async fn list_memory_banks(&self) -> Result<CallToolResult, ErrorData> {
        match self.bank_manager.list_banks().await {
            Ok(banks) => {
                let data = json!({
                    "success": true,
                    "message": "Memory banks listed successfully",
                    "count": banks.len(),
                    "banks_dir": self.bank_manager.banks_dir().display().to_string(),
                    "banks": banks,
                });
                let json = serde_json::to_string_pretty(&data).map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to serialize response: {}", e).into(),
                    data: None,
                })?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => {
                error!("Failed to list memory banks: {}", e);
                Err(ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to list memory banks: {}", e).into(),
                    data: None,
                })
            }
        }
    }

    /// Tool implementation for creating a memory bank
    async fn create_memory_bank(
        &self,
        arguments: &Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, ErrorData> {
        let name = arguments
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ErrorData {
                code: rmcp::model::ErrorCode(-32602),
                message: "Missing required parameter 'name'".into(),
                data: None,
            })?;

        let description = arguments
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        match self.bank_manager.create_bank(name, description).await {
            Ok(bank_info) => {
                let data = json!({
                    "success": true,
                    "message": format!("Memory bank '{}' ready", bank_info.name),
                    "bank": bank_info,
                });
                let json = serde_json::to_string_pretty(&data).map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to serialize response: {}", e).into(),
                    data: None,
                })?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => {
                error!("Failed to create memory bank: {}", e);
                Err(ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to create memory bank: {}", e).into(),
                    data: None,
                })
            }
        }
    }

    /// Tool implementation for backing up a memory bank
    async fn backup_bank(
        &self,
        arguments: &Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, ErrorData> {
        let name = arguments
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("default");

        let dest_dir = if let Some(dest) = arguments.get("destination").and_then(|v| v.as_str()) {
            PathBuf::from(dest)
        } else {
            // Default to ~/llm-mem-backups/
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("llm-mem-backups")
        };

        match self.bank_manager.backup_bank(name, &dest_dir).await {
            Ok((backup_path, manifest)) => {
                let data = json!({
                    "success": true,
                    "message": format!("Bank '{}' backed up successfully (v{})", name, manifest.version),
                    "backup_path": backup_path.display().to_string(),
                    "manifest": {
                        "version": manifest.version,
                        "created_at": manifest.created_at,
                        "memory_count": manifest.memory_count,
                        "sha256": manifest.sha256,
                        "size_bytes": manifest.size_bytes,
                    }
                });
                let json = serde_json::to_string_pretty(&data).map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to serialize response: {}", e).into(),
                    data: None,
                })?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => {
                error!("Failed to backup bank '{}': {}", name, e);
                Err(ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to backup bank: {}", e).into(),
                    data: None,
                })
            }
        }
    }

    /// Tool implementation for restoring a memory bank from a backup file
    async fn restore_bank(
        &self,
        arguments: &Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, ErrorData> {
        let name = arguments
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("default");

        let source = arguments
            .get("source")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ErrorData {
                code: rmcp::model::ErrorCode(-32602),
                message: "Missing required parameter 'source' (path to the backup .db file)".into(),
                data: None,
            })?;

        let mode = arguments
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("replace");

        // For replace mode, require explicit confirmation (overwrites data)
        if mode == "replace" {
            let confirm = arguments
                .get("confirm")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !confirm {
                return Err(ErrorData {
                    code: rmcp::model::ErrorCode(-32602),
                    message: "Replace-restore will overwrite the current bank data. Set confirm: true after verifying with the user.".into(),
                    data: None,
                });
            }
        }

        let source_path = PathBuf::from(source);

        match mode {
            "merge" => {
                match self
                    .bank_manager
                    .merge_from_backup(name, &source_path)
                    .await
                {
                    Ok(result) => {
                        let data = json!({
                            "success": true,
                            "message": format!(
                                "Merged backup into bank '{}': {} imported, {} duplicates skipped",
                                name, result.imported, result.skipped_duplicates
                            ),
                            "imported": result.imported,
                            "skipped_duplicates": result.skipped_duplicates,
                            "total_after_merge": result.total_after_merge,
                            "source": source,
                        });
                        let json = serde_json::to_string_pretty(&data).map_err(|e| ErrorData {
                            code: rmcp::model::ErrorCode(-32603),
                            message: format!("Failed to serialize response: {}", e).into(),
                            data: None,
                        })?;
                        Ok(CallToolResult::success(vec![Content::text(json)]))
                    }
                    Err(e) => {
                        error!("Failed to merge into bank '{}': {}", name, e);
                        Err(ErrorData {
                            code: rmcp::model::ErrorCode(-32603),
                            message: format!("Failed to merge backup: {}", e).into(),
                            data: None,
                        })
                    }
                }
            }
            _ => {
                // "replace" mode (default)
                match self.bank_manager.restore_bank(name, &source_path).await {
                    Ok(restored_path) => {
                        let data = json!({
                            "success": true,
                            "message": format!("Bank '{}' restored from backup (replace mode)", name),
                            "restored_path": restored_path.display().to_string(),
                            "source": source,
                        });
                        let json = serde_json::to_string_pretty(&data).map_err(|e| ErrorData {
                            code: rmcp::model::ErrorCode(-32603),
                            message: format!("Failed to serialize response: {}", e).into(),
                            data: None,
                        })?;
                        Ok(CallToolResult::success(vec![Content::text(json)]))
                    }
                    Err(e) => {
                        error!("Failed to restore bank '{}': {}", name, e);
                        Err(ErrorData {
                            code: rmcp::model::ErrorCode(-32603),
                            message: format!("Failed to restore bank: {}", e).into(),
                            data: None,
                        })
                    }
                }
            }
        }
    }

    /// Tool implementation for renaming a memory bank
    async fn rename_memory_bank(
        &self,
        arguments: &Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, ErrorData> {
        let old_name = arguments
            .get("old_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ErrorData {
                code: rmcp::model::ErrorCode(-32602),
                message: "Missing required parameter 'old_name'".into(),
                data: None,
            })?;

        let new_name = arguments
            .get("new_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ErrorData {
                code: rmcp::model::ErrorCode(-32602),
                message: "Missing required parameter 'new_name'".into(),
                data: None,
            })?;

        match self.bank_manager.rename_bank(old_name, new_name).await {
            Ok(()) => {
                let data = json!({
                    "success": true,
                    "message": format!("Bank renamed from '{}' to '{}'", old_name, new_name),
                    "old_name": old_name,
                    "new_name": new_name,
                });
                let json = serde_json::to_string_pretty(&data).map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to serialize response: {}", e).into(),
                    data: None,
                })?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => {
                error!(
                    "Failed to rename bank from '{}' to '{}': {}",
                    old_name, new_name, e
                );
                Err(ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to rename bank: {}", e).into(),
                    data: None,
                })
            }
        }
    }

    /// Tool implementation for cleaning up resources
    async fn cleanup_resources(
        &self,
        arguments: &Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, ErrorData> {
        let target = arguments
            .get("target")
            .and_then(|v| v.as_str())
            .unwrap_or("models");

        // For bank deletion, require a specific confirmation phrase
        // For model cleanup, a simple boolean confirm is fine
        if target == "banks" {
            let confirm_str = arguments
                .get("confirm")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if confirm_str != "I confirm this data will be permanently lost" {
                return Err(ErrorData {
                    code: rmcp::model::ErrorCode(-32602),
                    message: "Bank deletion requires user confirmation. Ask the user to confirm, then pass confirm: \"I confirm this data will be permanently lost\"".into(),
                    data: None,
                });
            }
        } else {
            let confirm = arguments
                .get("confirm")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !confirm {
                return Err(ErrorData {
                    code: rmcp::model::ErrorCode(-32602),
                    message: "Cleanup aborted. 'confirm' parameter must be true.".into(),
                    data: None,
                });
            }
        }

        match target {
            "models" => {
                info!(
                    "Cleaning up models directory: {}",
                    self.models_dir.display()
                );
                if self.models_dir.exists() {
                    match tokio::fs::remove_dir_all(&self.models_dir).await {
                        Ok(_) => {
                            // Recreate empty directory
                            let _ = tokio::fs::create_dir_all(&self.models_dir).await;

                            let msg = "Models directory cleaned up successfully. You may need to restart the server or subsequent calls will re-download models.";
                            Ok(CallToolResult::success(vec![Content::text(
                                json!({
                                    "success": true,
                                    "message": msg
                                })
                                .to_string(),
                            )]))
                        }
                        Err(e) => {
                            error!("Failed to remove models directory: {}", e);
                            Err(ErrorData {
                                code: rmcp::model::ErrorCode(-32603),
                                message: format!("Failed to cleanup models: {}", e).into(),
                                data: None,
                            })
                        }
                    }
                } else {
                    Ok(CallToolResult::success(vec![Content::text(
                        json!({
                            "success": true,
                            "message": "Models directory does not exist, nothing to clean."
                        })
                        .to_string(),
                    )]))
                }
            }
            "banks" => {
                let name = arguments.get("name").and_then(|v| v.as_str());

                if let Some(bank_name) = name {
                    match self.bank_manager.delete_bank(bank_name).await {
                        Ok(true) => Ok(CallToolResult::success(vec![Content::text(
                            json!({
                                "success": true,
                                "message": format!("Memory bank '{}' deleted successfully.", bank_name)
                            })
                            .to_string(),
                        )])),
                        Ok(false) => {
                             // If explicit name provided but not found
                             Ok(CallToolResult::success(vec![Content::text(
                                json!({
                                    "success": false,
                                    "message": format!("Memory bank '{}' not found or already deleted.", bank_name)
                                })
                                .to_string(),
                            )]))
                        }
                        Err(e) => Err(ErrorData {
                            code: rmcp::model::ErrorCode(-32603),
                            message: format!("Failed to delete bank '{}': {}", bank_name, e).into(),
                            data: None,
                        }),
                    }
                } else {
                    // Delete ALL banks
                    match self.bank_manager.list_banks().await {
                        Ok(banks) => {
                            let mut deleted_count = 0;
                            let mut errors = Vec::new();
                            for bank in banks {
                                match self.bank_manager.delete_bank(&bank.name).await {
                                    Ok(_) => deleted_count += 1,
                                    Err(e) => errors.push(format!("{}: {}", bank.name, e)),
                                }
                            }

                            if errors.is_empty() {
                                Ok(CallToolResult::success(vec![Content::text(
                                    json!({
                                        "success": true,
                                        "message": format!("Deleted {} memory banks.", deleted_count)
                                    })
                                    .to_string(),
                                )]))
                            } else {
                                let msg = format!(
                                    "Deleted {} banks. Errors: {}",
                                    deleted_count,
                                    errors.join("; ")
                                );
                                Ok(CallToolResult::success(vec![Content::text(
                                    json!({
                                        "success": false,
                                        "message": msg
                                    })
                                    .to_string(),
                                )]))
                            }
                        }
                        Err(e) => Err(ErrorData {
                            code: rmcp::model::ErrorCode(-32603),
                            message: format!("Failed to list banks for deletion: {}", e).into(),
                            data: None,
                        }),
                    }
                }
            }
            _ => Err(ErrorData {
                code: rmcp::model::ErrorCode(-32602),
                message: format!(
                    "Unknown target '{}'. Only 'models' and 'banks' are supported.",
                    target
                )
                .into(),
                data: None,
            }),
        }
    }

    /// Find default configuration file path
    fn find_default_config_path() -> Option<PathBuf> {
        // Current directory
        if let Ok(current_dir) = std::env::current_dir() {
            let current_config = current_dir.join("config.toml");
            if current_config.exists() {
                return Some(current_config);
            }
        }

        // User home config directory
        if let Some(home_dir) = dirs::home_dir() {
            let user_config = home_dir.join(".config").join("llm-mem").join("config.toml");
            if user_config.exists() {
                return Some(user_config);
            }
        }

        // System config directory
        #[cfg(target_os = "macos")]
        let system_config = Path::new("/usr/local/etc/llm-mem/config.toml");
        #[cfg(target_os = "linux")]
        let system_config = Path::new("/etc/llm-mem/config.toml");
        #[cfg(target_os = "windows")]
        let system_config = Path::new("C:\\ProgramData\\memo\\config.toml");

        if system_config.exists() {
            return Some(system_config.to_path_buf());
        }

        None
    }

    /// Convert OperationError to MCP ErrorData
    fn operation_error_to_mcp_error(&self, error: OperationError) -> ErrorData {
        ErrorData {
            code: rmcp::model::ErrorCode(operation_error_to_mcp_error_code(&error)),
            message: get_operation_error_message(&error).into(),
            data: None,
        }
    }
}

impl ServerHandler for MemoryMcpService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: rmcp::model::ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: rmcp::model::Implementation::from_build_env(),
            instructions: Some(
                "A persistent semantic memory and knowledge index for AI agents. \
                 This system lets you store, connect, and retrieve any kind of knowledge — code architecture, \
                 project documentation, research findings, web references, conversations, specifications, or any \
                 structured/unstructured information — using natural language queries powered by vector embeddings. \
                 \n\
                 Core philosophy: Store ATOMIC KNOWLEDGE UNITS (not raw documents). Each memory is a searchable \
                 fact or insight with source metadata pointing back to the original material (file paths, URLs, \
                 page numbers, line ranges). The memory acts as a semantic index: store what you need to know, \
                 reference where to find the full detail. \
                 \n\
                 Three-level knowledge model: (1) Node — standalone facts stored as vector embeddings for \
                 semantic search. (2) Edge — relations connecting nodes into a knowledge graph. \
                 (3) Context — semantic filter tags enabling scoped retrieval across domains. \
                 \n\
                 Use banks for hard isolation between projects/domains. Use context tags for soft grouping \
                 within a bank. Always call system_status first to get detailed usage guidance and verify readiness."
                    .to_string(),
            ),
        }
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let tool_definitions = get_mcp_tool_definitions();
        let tools: Vec<Tool> = tool_definitions
            .into_iter()
            .map(|def| Tool {
                name: def.name.into(),
                title: def.title,
                description: def.description.map(|d| d.into()),
                input_schema: def
                    .input_schema
                    .as_object()
                    .cloned()
                    .unwrap_or_default()
                    .into(),
                output_schema: def
                    .output_schema
                    .and_then(|schema| schema.as_object().cloned())
                    .map(|obj| obj.into()),
                annotations: None,
                icons: None,
                meta: None,
            })
            .collect();

        Ok(ListToolsResult {
            tools,
            next_cursor: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let tool_name = &request.name;
        let empty_args = Map::new();

        match tool_name.as_ref() {
            "system_status" => {
                let status = self.bank_manager.get_llm_status();
                let banks = self.bank_manager.list_banks().await.unwrap_or_default();
                let bank_names: Vec<String> = banks.iter().map(|b| b.name.clone()).collect();
                let bank_count = banks.len();

                // Determine clear readiness state
                let ready_to_use =
                    status.llm_available && status.embedding_available && status.state == "ready";
                let readiness = if ready_to_use {
                    "READY — System is fully operational. You can store and query memories."
                } else if status.state == "initializing" {
                    "PREPARING — Models are loading or downloading. Please wait and call system_status again in a few seconds."
                } else {
                    "NOT READY — System encountered an error during initialization. Check 'last_error' for details."
                };

                // Compute disk usage
                let models_size = dir_size_bytes(&self.models_dir);
                let banks_size = dir_size_bytes(self.bank_manager.banks_dir());

                // Active document processing
                let active_sessions = self
                    .bank_manager
                    .list_all_active_sessions()
                    .await
                    .unwrap_or_default();
                let mut session_info = Vec::new();
                for s in active_sessions {
                    session_info.push(json!({
                        "session_id": s.session_id,
                        "file_name": s.metadata.file_name,
                        "status": s.status.as_str(),
                        "progress": format!("{}/{}", s.received_parts, s.expected_parts),
                        "started_at": s.created_at,
                    }));
                }

                // Gather layer statistics across all banks
                use std::collections::HashMap;

                #[derive(Default)]
                struct LayerCountStats {
                    count: u64,
                    active: u64,
                    forgotten: u64,
                    processing: u64,
                    invalid: u64,
                }

                let mut total_memories: u64 = 0;
                let mut by_layer: HashMap<usize, LayerCountStats> = HashMap::new();
                let mut state_counts = json!({
                    "active": 0u64,
                    "forgotten": 0u64,
                    "processing": 0u64,
                    "invalid": 0u64
                });
                let mut max_layer: usize = 0;

                // Collect stats from each bank
                let banks = self.bank_manager.list_banks().await.unwrap_or_default();
                for bank_info in &banks {
                    if let Ok(bank) = self.bank_manager.get_or_create(&bank_info.name).await
                        && let Ok(memories) = bank.list(&Filters::new(), None).await
                    {
                        for memory in &memories {
                            let level = memory.metadata.layer.level as usize;
                            let state = memory.metadata.state.as_str();

                            // Update total
                            total_memories += 1;

                            // Update by_layer
                            let layer_entry = by_layer.entry(level).or_default();
                            layer_entry.count += 1;
                            match state {
                                "active" => layer_entry.active += 1,
                                "forgotten" => layer_entry.forgotten += 1,
                                "processing" => layer_entry.processing += 1,
                                "invalid" => layer_entry.invalid += 1,
                                _ => {}
                            }

                            // Update state counts
                            if let Some(count) = state_counts[state].as_u64() {
                                state_counts[state] = json!(count + 1);
                            }

                            // Update max_layer
                            if level > max_layer {
                                max_layer = level;
                            }
                        }
                    }
                }

                // Build by_layer JSON
                let mut by_layer_json = serde_json::Map::new();
                for (level, stats) in by_layer {
                    by_layer_json.insert(
                        level.to_string(),
                        json!({
                            "count": stats.count,
                            "active": stats.active,
                            "forgotten": stats.forgotten,
                            "processing": stats.processing,
                            "invalid": stats.invalid
                        }),
                    );
                }

                let layer_stats = json!({
                    "total_memories": total_memories,
                    "by_layer": by_layer_json,
                    "max_layer": max_layer,
                    "state_counts": state_counts,
                });

                // Get abstraction pipeline status
                let pipeline_status = self.bank_manager.get_pipeline_status().await;

                let mut guide = json!({
                    "ready_to_use": ready_to_use,
                    "readiness_message": readiness,
                    "system_status": status,
                    "disk_usage": {
                        "models_dir": self.models_dir.display().to_string(),
                        "models_size_bytes": models_size,
                        "models_size_human": format_bytes(models_size),
                        "banks_dir": self.bank_manager.banks_dir().display().to_string(),
                        "banks_size_bytes": banks_size,
                        "banks_size_human": format_bytes(banks_size),
                        "total_size_bytes": models_size + banks_size,
                        "total_size_human": format_bytes(models_size + banks_size),
                    },
                    "active_banks": {
                        "count": bank_count,
                        "names": bank_names,
                    },
                    "layer_statistics": layer_stats,
                    "abstraction_pipeline": json!({
                        "enabled": pipeline_status.enabled,
                        "workers_running": pipeline_status.workers_running,
                        "pending_l0_count": pipeline_status.pending_l0_count,
                        "pending_l1_count": pipeline_status.pending_l1_count,
                        "pending_l2_count": pipeline_status.pending_l2_count,
                        "config": {
                            "min_memories_for_l1": pipeline_status.config.min_memories_for_l1,
                            "l1_processing_delay_secs": pipeline_status.config.l1_processing_delay_secs,
                            "max_concurrent_tasks": pipeline_status.config.max_concurrent_tasks
                        }
                    }),
                });

                if !session_info.is_empty() {
                    guide["document_processing_active"] = json!(session_info);
                }

                guide["usage_guide"] = json!({
                    "overview": "llm-mem is a persistent semantic knowledge index using a layered memory architecture (L0-L4+). \
                                 It combines high-fidelity verbatim storage with AI-powered progressive abstraction: raw content (L0) → \
                                 structural summaries (L1) → semantic links (L2) → domain concepts (L3) → mental models/wisdom (L4+). \
                                 Background workers automatically create higher abstractions, enabling bidirectional navigation \
                                 (zoom in/out) across abstraction levels. Works for any domain: codebases, documentation, research, \
                                 conversations, specifications, or any structured/unstructured information.",

                    "core_philosophy": {
                        "hybrid_storage": "Each memory is a searchable knowledge pointer. For documents, the system stores the EXACT original text in \
                                           semantic chunks (Verbatim Content Storage), ensuring high-fidelity retrieval. For manual entries, you can \
                                           store either raw content (add_content_memory) or AI-extracted atomic insights (add_intuitive_memory).",
                        "progressive_abstraction": "Background workers automatically create higher-layer abstractions: L0 chunks → L1 summaries → L2 semantic links → \
                                                    L3 concepts → L4+ wisdom. This mimics human cognitive organization: sensory input → episodic → semantic → conceptual.",
                        "bidirectional_navigation": "Navigate the abstraction hierarchy: zoom_out() from concrete to abstract (find higher-level insights), \
                                                     zoom_in() from abstract to concrete (find source evidence), search_at_layer() for targeted queries.",
                        "layered_relations": "Relations carry layer semantics: structural (chunk_of, summary_of), semantic (related_to, extends, contradicts), \
                                              conceptual (emerges_from, instance_of, broader_than). Higher layers emerge from multiple lower-layer memories.",
                        "what_to_ask": "Before storing, ask: (1) 'What would someone search for to find this?' → that's your content. \
                                         (2) 'Where does this come from?' → that's your metadata.custom source fields. \
                                         (3) 'What is this related to?' → those are your relations. \
                                         (4) 'What domain/scope does this belong to?' → those are your context tags. \
                                         (5) 'Should this be isolated from other work?' → that determines which bank to use."
                    },

                    "layered_memory_architecture": {
                        "description": "Memories exist at different abstraction levels (L0-L4+), with background workers creating progressive abstractions.",
                        "L0_raw_content": {
                            "level": 0,
                            "what": "User-provided, immutable content — verbatim document chunks, raw facts, direct observations.",
                            "how": "Use begin_store_document for files (auto-chunked), add_content_memory for raw content, add_intuitive_memory for AI-processed facts.",
                            "when": "Store any discrete knowledge: document chunks, API contracts, config values, research findings, conversation takeaways.",
                            "example": "The Laplace transform converts differential equations to algebraic equations — DDI0301H Chapter 3, Section 2.1"
                        },
                        "L1_structural": {
                            "level": 1,
                            "what": "Structural abstractions — summaries, section headers, document organization.",
                            "how": "Automatically created by background L0→L1 worker using LLM summarization.",
                            "when": "Created automatically when sufficient L0 content accumulates (configurable threshold).",
                            "example": "Chapter 3 covers mathematical transforms for signal processing — includes Laplace, Fourier, and Z-transforms"
                        },
                        "L2_semantic": {
                            "level": 2,
                            "what": "Semantic links — cross-document connections, thematic relationships.",
                            "how": "Automatically created by background L1→L2 worker identifying semantic relationships.",
                            "when": "Created when related L1 summaries share themes or concepts.",
                            "example": "Relates ODEs to Control Theory — Laplace transforms enable frequency-domain analysis of feedback systems"
                        },
                        "L3_concept": {
                            "level": 3,
                            "what": "Domain concepts — theories, principles, abstract patterns emerging from multiple sources.",
                            "how": "Automatically created by background L2→L3 worker synthesizing conceptual insights.",
                            "when": "Created when sufficient L2 clusters reveal underlying concepts.",
                            "example": "Linear Algebra is about vector spaces and linear mappings — foundational for signal processing, ML, and physics"
                        },
                        "L4_wisdom": {
                            "level": 4,
                            "what": "Mental models, paradigms, universal principles — the deepest abstraction level.",
                            "how": "Created by synthesizing L3 concepts into overarching frameworks.",
                            "when": "Emerges from cross-domain conceptual integration.",
                            "example": "Mathematical duality: time-domain ↔ frequency-domain via transforms — a recurring pattern across physics and engineering"
                        },
                        "navigation_api": {
                            "zoom_out": "Navigate from concrete to abstract: given L0 memory, find L1+ abstractions built from it.",
                            "zoom_in": "Navigate from abstract to concrete: given L3 concept, find L0 source evidence.",
                            "search_at_layer": "Search within a specific abstraction level for targeted queries."
                        }
                    },

                    "domain_patterns": {
                        "description": "How to organize memory for different types of information sources:",
                        "codebase": {
                            "what_to_store": "Module responsibilities, API contracts, architectural decisions, dependency relationships, \
                                               config/environment details, known gotchas, build/deploy procedures, key algorithms.",
                            "source_metadata": "file_path, line_range, function_name, commit_hash, repo_url",
                            "context_tags": "module name, layer (frontend/backend/infra), language, framework",
                            "example_content": "AuthService handles JWT token validation and refresh — entry point is validate_token() in src/auth/service.rs:45-80",
                            "bank_strategy": "One bank per project/repo, or 'default' for a single project. Use context tags for modules."
                        },
                        "documents": {
                            "what_to_store": "Use begin_store_document for full files. The system will automatically chunk, summarize, and extract keywords while preserving verbatim text.",
                            "source_metadata": "source_file, page_number, section_name, author, date, version",
                            "context_tags": "document type (spec, requirements, design-doc, manual), project, topic",
                            "example_content": "The API rate limit is 1000 requests/minute per API key, with a burst allowance of 50.",
                            "bank_strategy": "Same bank as the project it belongs to, or a dedicated 'docs' bank for cross-project reference material."
                        },
                        "web_references": {
                            "what_to_store": "The key insight or answer you found. Summarize what matters, don't store the full page.",
                            "source_metadata": "url, domain, date_accessed, author (if known), title",
                            "context_tags": "topic, technology, problem-domain",
                            "example_content": "Tokio select! macro requires all branches to be cancel-safe — use tokio::sync::mpsc instead of oneshot for repeated operations",
                            "bank_strategy": "Store in the project bank where you'll need it, or a 'research' bank for general reference."
                        },
                        "conversations_and_decisions": {
                            "what_to_store": "Decisions made, action items, preferences expressed, requirements clarified. Focus on outcomes, not transcripts.",
                            "source_metadata": "conversation_id, date, participants, meeting_name",
                            "context_tags": "project, topic, decision-type",
                            "example_content": "Team decided to use PostgreSQL over MongoDB for the analytics service due to complex join requirements — 2026-02-15 arch meeting",
                            "bank_strategy": "Same bank as the project. Use context tags like 'decisions', 'action-items', 'preferences'."
                        },
                        "large_content_strategy": {
                            "principle": "For large sources, use begin_store_document. It creates a hierarchy of memories: \
                                           (1) Section headers as nodes linked by 'part_of'. \
                                           (2) Content chunks as nodes linked by 'next_chunk'/'previous_chunk'. \
                                           (3) Cross-document semantic links using 'references'.",
                            "retrieval_pattern": "Search finds verbatim chunks. Use graph traversal to fetch 'next_chunk' or parent headers for more context. \
                                                  The memory system is your semantic index; the graph links provide the structure."
                        }
                    },

                    "banks_and_user_id": {
                        "banks": "Banks are completely isolated memory stores (separate database files). Use different banks for different projects or domains. \
                                  The 'default' bank is used when no bank is specified. Create a new bank with create_memory_bank when starting a new project or topic \
                                  that should have its own isolated memory space.",
                        "user_id": "Optional. Only needed if multiple users share the same bank and you want to filter memories per user. \
                                    In most cases (single user per bank, or using separate banks per user), omit user_id entirely.",
                        "when_to_create_new_bank": "Create a new bank when: (1) starting a new project, (2) you want memories completely separate from other work, \
                                                    or (3) you need different memory contexts that should never mix. Within a bank, use 'context' tags for softer grouping."
                    },

                    "memory_types": {
                        "conversational": "Dialog and interaction memories (default)",
                        "factual": "Verified facts, data points, specifications, configs",
                        "semantic": "Conceptual knowledge, definitions, explanations",
                        "episodic": "Events, incidents, experiences with temporal context",
                        "procedural": "How-to knowledge, processes, workflows, build/deploy steps",
                        "personal": "User preferences, habits, and personal info"
                    },

                    "critical_guidelines": {
                        "VERBATIM_for_documents": "For documents and code, store the EXACT original text in semantic chunks (L0). \
                                                    The system will enrich these with AI-generated keywords, summaries (L1), and concepts (L3+) automatically.",

                        "ATOMIC_for_insights": "For manual facts and decisions, keep memories atomic (5-50 words). One memory = one searchable insight.",

                        "ALWAYS_include_source": "Every memory MUST have source attribution in metadata.custom — file paths, URLs, page numbers, line ranges, \
                                                   commit hashes, dates. This is how you or another agent can fetch the full original when needed.",

                        "content_field_subject_focus": "The 'content' field should describe a clear SUBJECT with identifying info. \
                                                        Write it as a complete, searchable statement. Ask: 'If someone searched for this topic, \
                                                        would this content match their query?'",

                        "relations_action_focus": "Relations should use descriptive verb predicates: 'next_chunk', 'part_of', 'references', \
                                                   'depends_on', 'implements', 'configures', 'supersedes', 'authored_by'. \
                                                   For layered memories: 'summary_of', 'emerges_from', 'instance_of'.",

                        "context_broad_categories": "Context tags should be BROAD categories enabling future discovery. Use 3-5 relevant tags. \
                                                      Think about what domain, layer, project, or topic this belongs to.",

                        "focus_on_what_matters": "Not everything needs to be stored. Prioritize: (1) Information you'll need to recall later. \
                                                  (2) Decisions and their rationale. (3) Non-obvious facts that are hard to re-derive. \
                                                  (4) Connections between concepts. Skip trivial or easily re-derivable information.",

                        "layer_aware_storage": "L0 memories are created by you (user input). L1+ memories are created automatically by background workers. \
                                                Focus on providing high-quality L0 content; the system handles abstraction.",

                        "deletion_cascade": "Deleting L0 memories marks higher-layer abstractions (L1+) as 'forgotten' (soft delete) to preserve \
                                             referential integrity. Forgotten memories can be restored or permanently deleted later."
                    },

                    "tips": [
                        "Always call system_status first to verify the system is ready and check layer_statistics.",
                        "Use begin_store_document for files; it handles chunking and verbatim storage for you.",
                        "Use 'context' tags for soft grouping within a bank; use separate banks for hard isolation.",
                        "Semantic search works by meaning — query 'authentication flow' will find memories about 'JWT token validation' and 'login endpoint'.",
                        "The 'relations' field builds a knowledge graph — useful for connecting modules, services, people, concepts, and dependencies.",
                        "Include source attribution in 'metadata' (file paths, URLs, line numbers) so you can fetch the original material when the memory summary isn't enough.",
                        "When querying returns 0 results, check: (1) Correct bank? (2) Data actually stored? (3) Try different phrasing.",
                        "Use layer_statistics to monitor abstraction progress: L0→L1→L2→L3+ creation by background workers.",
                        "For navigation across abstraction levels, use zoom_in() to find source evidence or zoom_out() to find higher-level insights.",
                        "When you retrieve a memory and need more detail, use its source metadata to fetch the original file, URL, or document section.",
                        "Search at specific layers using search_at_layer() for targeted queries (e.g., only concepts at L3, only raw content at L0)."
                    ]
                });

                match serde_json::to_string_pretty(&guide) {
                    Ok(json) => Ok(CallToolResult::success(vec![Content::text(json)])),
                    Err(e) => Err(ErrorData {
                        code: rmcp::model::ErrorCode(-32603),
                        message: format!("Failed to serialize status: {}", e).into(),
                        data: None,
                    }),
                }
            }
            "cleanup_resources" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                self.cleanup_resources(args).await
            }
            "add_content_memory" | "store_memory" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                self.store_memory(args).await
            }
            "add_intuitive_memory" | "add_memory" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                self.add_memory(args).await
            }
            "begin_store_document" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                let payload = map_mcp_arguments_to_payload(args, &self.agent_id);
                let ops = self
                    .resolve_operations_with_sessions(payload.bank.as_deref())
                    .await?;
                match ops.begin_store_document(payload) {
                    Ok(response) => {
                        let json =
                            serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                                code: rmcp::model::ErrorCode(-32603),
                                message: format!("Failed to serialize response: {}", e).into(),
                                data: None,
                            })?;
                        Ok(CallToolResult::success(vec![Content::text(json)]))
                    }
                    Err(e) => {
                        error!("Failed to begin document store: {}", e);
                        Err(self.operation_error_to_mcp_error(e))
                    }
                }
            }
            "store_document_part" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                let payload = map_mcp_arguments_to_payload(args, &self.agent_id);
                let ops = self
                    .resolve_operations_with_sessions(payload.bank.as_deref())
                    .await?;
                match ops.store_document_part(payload) {
                    Ok(response) => {
                        self.bank_manager.notify_new_memory().await;
                        let json =
                            serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                                code: rmcp::model::ErrorCode(-32603),
                                message: format!("Failed to serialize response: {}", e).into(),
                                data: None,
                            })?;
                        Ok(CallToolResult::success(vec![Content::text(json)]))
                    }
                    Err(e) => {
                        error!("Failed to store document part: {}", e);
                        Err(self.operation_error_to_mcp_error(e))
                    }
                }
            }
            "process_document" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                let payload = map_mcp_arguments_to_payload(args, &self.agent_id);
                let ops = self
                    .resolve_operations_with_sessions(payload.bank.as_deref())
                    .await?;
                match ops.process_document(payload).await {
                    Ok(response) => {
                        self.bank_manager.notify_new_memory().await;
                        let json =
                            serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                                code: rmcp::model::ErrorCode(-32603),
                                message: format!("Failed to serialize response: {}", e).into(),
                                data: None,
                            })?;
                        Ok(CallToolResult::success(vec![Content::text(json)]))
                    }
                    Err(e) => {
                        error!("Failed to process document: {}", e);
                        Err(self.operation_error_to_mcp_error(e))
                    }
                }
            }
            "upload_document" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                let payload = map_mcp_arguments_to_payload(args, &self.agent_id);
                let ops = self
                    .resolve_operations_with_sessions(payload.bank.as_deref())
                    .await?;
                match ops.upload_document(payload).await {
                    Ok(response) => {
                        self.bank_manager.notify_new_memory().await;
                        let json =
                            serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                                code: rmcp::model::ErrorCode(-32603),
                                message: format!("Failed to serialize response: {}", e).into(),
                                data: None,
                            })?;
                        Ok(CallToolResult::success(vec![Content::text(json)]))
                    }
                    Err(e) => {
                        error!("Failed to upload document: {}", e);
                        Err(self.operation_error_to_mcp_error(e))
                    }
                }
            }
            "status_process_document" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                let payload = map_mcp_arguments_to_payload(args, &self.agent_id);
                let ops = self
                    .resolve_operations_with_sessions(payload.bank.as_deref())
                    .await?;
                match ops.status_process_document(payload) {
                    Ok(response) => {
                        let json =
                            serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                                code: rmcp::model::ErrorCode(-32603),
                                message: format!("Failed to serialize response: {}", e).into(),
                                data: None,
                            })?;
                        Ok(CallToolResult::success(vec![Content::text(json)]))
                    }
                    Err(e) => {
                        error!("Failed to get document status: {}", e);
                        Err(self.operation_error_to_mcp_error(e))
                    }
                }
            }
            "list_document_sessions" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                let payload = map_mcp_arguments_to_payload(args, &self.agent_id);
                let ops = self
                    .resolve_operations_with_sessions(payload.bank.as_deref())
                    .await?;
                match ops.list_document_sessions(payload) {
                    Ok(response) => {
                        let json =
                            serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                                code: rmcp::model::ErrorCode(-32603),
                                message: format!("Failed to serialize response: {}", e).into(),
                                data: None,
                            })?;
                        Ok(CallToolResult::success(vec![Content::text(json)]))
                    }
                    Err(e) => {
                        error!("Failed to list document sessions: {}", e);
                        Err(self.operation_error_to_mcp_error(e))
                    }
                }
            }
            "cancel_process_document" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                let payload = map_mcp_arguments_to_payload(args, &self.agent_id);
                let ops = self
                    .resolve_operations_with_sessions(payload.bank.as_deref())
                    .await?;
                match ops.cancel_process_document(payload) {
                    Ok(response) => {
                        let json =
                            serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                                code: rmcp::model::ErrorCode(-32603),
                                message: format!("Failed to serialize response: {}", e).into(),
                                data: None,
                            })?;
                        Ok(CallToolResult::success(vec![Content::text(json)]))
                    }
                    Err(e) => {
                        error!("Failed to cancel document session: {}", e);
                        Err(self.operation_error_to_mcp_error(e))
                    }
                }
            }
            "update_memory" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                self.update_memory(args).await
            }
            "query_memory" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                self.query_memory(args).await
            }
            "list_memories" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                self.list_memories(args).await
            }
            "get_memory" => {
                if let Some(arguments) = &request.arguments {
                    self.get_memory(arguments).await
                } else {
                    Err(ErrorData {
                        code: rmcp::model::ErrorCode(-32602),
                        message: "Missing arguments. 'memory_id' is required.".into(),
                        data: None,
                    })
                }
            }
            "navigate_memory" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                self.navigate_memory(args).await
            }
            "list_memory_banks" => self.list_memory_banks().await,
            "create_memory_bank" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                self.create_memory_bank(args).await
            }
            "backup_bank" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                self.backup_bank(args).await
            }
            "restore_bank" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                self.restore_bank(args).await
            }
            "rename_memory_bank" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                self.rename_memory_bank(args).await
            }
            "start_abstraction_pipeline" => match self.bank_manager.start_pipeline_manual().await {
                Ok(message) => {
                    let json = serde_json::to_string_pretty(&json!({
                        "success": true,
                        "message": message
                    }))
                    .map_err(|e| ErrorData {
                        code: rmcp::model::ErrorCode(-32603),
                        message: format!("Failed to serialize response: {}", e).into(),
                        data: None,
                    })?;
                    Ok(CallToolResult::success(vec![Content::text(json)]))
                }
                Err(e) => Err(ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to start pipeline: {}", e).into(),
                    data: None,
                }),
            },
            "stop_abstraction_pipeline" => match self.bank_manager.stop_pipeline().await {
                Ok(message) => {
                    let json = serde_json::to_string_pretty(&json!({
                        "success": true,
                        "message": message
                    }))
                    .map_err(|e| ErrorData {
                        code: rmcp::model::ErrorCode(-32603),
                        message: format!("Failed to serialize response: {}", e).into(),
                        data: None,
                    })?;
                    Ok(CallToolResult::success(vec![Content::text(json)]))
                }
                Err(e) => Err(ErrorData {
                    code: rmcp::model::ErrorCode(-32603),
                    message: format!("Failed to stop pipeline: {}", e).into(),
                    data: None,
                }),
            },
            "trigger_abstraction" => {
                let args = request.arguments.as_ref().unwrap_or(&empty_args);
                let target_layer = args
                    .get("target_layer")
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32);

                match self
                    .bank_manager
                    .trigger_abstraction_now(target_layer)
                    .await
                {
                    Ok(result) => {
                        let json = serde_json::to_string_pretty(&json!({
                            "success": true,
                            "l0_to_l1_created": result.l0_to_l1_created,
                            "l1_to_l2_created": result.l1_to_l2_created,
                            "l2_to_l3_created": result.l2_to_l3_created,
                            "errors": result.errors
                        }))
                        .map_err(|e| ErrorData {
                            code: rmcp::model::ErrorCode(-32603),
                            message: format!("Failed to serialize response: {}", e).into(),
                            data: None,
                        })?;
                        Ok(CallToolResult::success(vec![Content::text(json)]))
                    }
                    Err(e) => Err(ErrorData {
                        code: rmcp::model::ErrorCode(-32603),
                        message: format!("Failed to trigger abstraction: {}", e).into(),
                        data: None,
                    }),
                }
            }
            _ => Err(ErrorData {
                code: rmcp::model::ErrorCode(-32601),
                message: format!("Unknown tool: {}", tool_name).into(),
                data: None,
            }),
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Recursively compute the total size of a directory in bytes.
pub fn dir_size_bytes(path: &Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    walkdir(path)
}

fn walkdir(path: &Path) -> u64 {
    let mut total: u64 = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                total += walkdir(&p);
            } else if let Ok(meta) = p.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

/// Format bytes into a human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
