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
use tracing::{error, info};

use crate::{
    config::Config,
    llm::create_llm_client,
    memory_bank::MemoryBankManager,
    operations::{
        MemoryOperations, OperationError, get_mcp_tool_definitions, get_operation_error_message,
        map_mcp_arguments_to_payload, operation_error_to_mcp_error_code,
    },
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

        // Pre-load the default bank
        bank_manager.default_bank().await?;
        info!("Default memory bank loaded");

        Ok(Self {
            bank_manager,
            agent_id,
            default_limit: 100,
            models_dir: PathBuf::from(config.local.models_dir.clone()),
        })
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
                code: rmcp::model::ErrorCode(-32603).into(),
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
                code: rmcp::model::ErrorCode(-32603).into(),
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
                let json = serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode(-32603).into(),
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
                let json = serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode(-32603).into(),
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
                    code: rmcp::model::ErrorCode(-32603).into(),
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
                    code: rmcp::model::ErrorCode(-32603).into(),
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
                    code: rmcp::model::ErrorCode(-32603).into(),
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
                    code: rmcp::model::ErrorCode(-32603).into(),
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
                    code: rmcp::model::ErrorCode(-32603).into(),
                    message: format!("Failed to serialize response: {}", e).into(),
                    data: None,
                })?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => {
                error!("Failed to list memory banks: {}", e);
                Err(ErrorData {
                    code: rmcp::model::ErrorCode(-32603).into(),
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
                code: rmcp::model::ErrorCode(-32602).into(),
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
                    code: rmcp::model::ErrorCode(-32603).into(),
                    message: format!("Failed to serialize response: {}", e).into(),
                    data: None,
                })?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => {
                error!("Failed to create memory bank: {}", e);
                Err(ErrorData {
                    code: rmcp::model::ErrorCode(-32603).into(),
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
                    code: rmcp::model::ErrorCode(-32603).into(),
                    message: format!("Failed to serialize response: {}", e).into(),
                    data: None,
                })?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => {
                error!("Failed to backup bank '{}': {}", name, e);
                Err(ErrorData {
                    code: rmcp::model::ErrorCode(-32603).into(),
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
                code: rmcp::model::ErrorCode(-32602).into(),
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
                    code: rmcp::model::ErrorCode(-32602).into(),
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
                            code: rmcp::model::ErrorCode(-32603).into(),
                            message: format!("Failed to serialize response: {}", e).into(),
                            data: None,
                        })?;
                        Ok(CallToolResult::success(vec![Content::text(json)]))
                    }
                    Err(e) => {
                        error!("Failed to merge into bank '{}': {}", name, e);
                        Err(ErrorData {
                            code: rmcp::model::ErrorCode(-32603).into(),
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
                            code: rmcp::model::ErrorCode(-32603).into(),
                            message: format!("Failed to serialize response: {}", e).into(),
                            data: None,
                        })?;
                        Ok(CallToolResult::success(vec![Content::text(json)]))
                    }
                    Err(e) => {
                        error!("Failed to restore bank '{}': {}", name, e);
                        Err(ErrorData {
                            code: rmcp::model::ErrorCode(-32603).into(),
                            message: format!("Failed to restore bank: {}", e).into(),
                            data: None,
                        })
                    }
                }
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
                    code: rmcp::model::ErrorCode(-32602).into(),
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
                    code: rmcp::model::ErrorCode(-32602).into(),
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
                                code: rmcp::model::ErrorCode(-32603).into(),
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
                            code: rmcp::model::ErrorCode(-32603).into(),
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
                            code: rmcp::model::ErrorCode(-32603).into(),
                            message: format!("Failed to list banks for deletion: {}", e).into(),
                            data: None,
                        }),
                    }
                }
            }
            _ => Err(ErrorData {
                code: rmcp::model::ErrorCode(-32602).into(),
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
            code: rmcp::model::ErrorCode(operation_error_to_mcp_error_code(&error)).into(),
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

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, ErrorData>> + Send + '_ {
        async move {
            let tool_definitions = get_mcp_tool_definitions();
            let tools: Vec<Tool> = tool_definitions
                .into_iter()
                .map(|def| Tool {
                    name: def.name.into(),
                    title: def.title.map(|t| t.into()),
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
    }

    fn call_tool(
        &self,
        request: CallToolRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CallToolResult, ErrorData>> + Send + '_ {
        async move {
            let tool_name = &request.name;
            let empty_args = Map::new();

            match tool_name.as_ref() {
                "system_status" => {
                    let status = self.bank_manager.get_llm_status();
                    let banks = self.bank_manager.list_banks().await.unwrap_or_default();
                    let bank_names: Vec<String> = banks.iter().map(|b| b.name.clone()).collect();
                    let bank_count = banks.len();

                    // Determine clear readiness state
                    let ready_to_use = status.llm_available
                        && status.embedding_available
                        && status.state == "ready";
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
                    let active_sessions = self.bank_manager.list_all_active_sessions().await.unwrap_or_default();
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
                    });

                    if !session_info.is_empty() {
                        guide["document_processing_active"] = json!(session_info);
                    }

                    guide["usage_guide"] = json!({
                        "overview": "llm-mem is a persistent semantic knowledge index. It stores atomic knowledge units as vector embeddings \
                                     for retrieval by meaning, not keywords. The memory acts as an INDEX into your knowledge: store the essential \
                                     insight, and always include source metadata so you (or another agent) can fetch the full original material \
                                     when needed. This works for any domain: codebases, documentation, research, web references, conversations, \
                                     specifications, or any structured/unstructured information.",

                            "core_philosophy": {
                                "memory_as_index": "Each memory is a searchable knowledge pointer. Store the KEY INSIGHT you'd want to find later, \
                                                     plus metadata pointing to the full source (file path, URL, page number, line range, API endpoint, etc.). \
                                                     When you retrieve a memory and need more detail, use the source reference to fetch the original material.",
                                "atomic_not_bulk": "One memory = one searchable fact, decision, or insight. NEVER store raw documents, entire files, or large \
                                                     blocks of text. Instead, decompose into 5-50 atomic memories per source document. Each memory should answer \
                                                     one specific question someone might ask later.",
                                "what_to_ask": "Before storing, ask: (1) 'What would someone search for to find this?' → that's your content. \
                                                 (2) 'Where does this come from?' → that's your metadata.custom source fields. \
                                                 (3) 'What is this related to?' → those are your relations. \
                                                 (4) 'What domain/scope does this belong to?' → those are your context tags. \
                                                 (5) 'Should this be isolated from other work?' → that determines which bank to use."
                            },

                            "three_levels_of_memory": {
                                "level_1_node_knowledge": {
                                    "what": "Each memory is a standalone knowledge node — a single fact, observation, decision, or insight stored with its vector embedding.",
                                    "how": "Use add_content_memory to store raw content as-is, or add_intuitive_memory for AI-processed facts. Use query_memory with a natural language 'query' to find similar nodes by meaning (hybrid semantic + keyword search).",
                                    "when": "Store any discrete piece of knowledge: a fact, architectural decision, API contract, config value, research finding, \
                                             requirement, preference, conversation takeaway, or URL bookmark with summary."
                                },
                                "level_2_edge_knowledge": {
                                    "what": "Relations connect nodes into a knowledge graph. Each relation has a predicate (the relationship type) and a target (what it connects to).",
                                    "how": "Use the 'relations' parameter in add_content_memory, add_intuitive_memory, or update_memory. Each relation needs 'relation' (verb/predicate) and 'target' (the connected entity/concept).",
                                    "when": "Use when information has meaningful connections: 'AuthService depends_on TokenValidator', 'RFC-42 supersedes RFC-31', \
                                             'deploy.yaml configures production-cluster', 'user prefers dark-mode'."
                                },
                                "level_3_context_filtering": {
                                    "what": "Context tags act as semantic attention filters. They are embedded as separate vectors, enabling two-stage retrieval: first narrow by context, then search by content.",
                                    "how": "Use the 'context' parameter (array of strings) in add_content_memory, add_intuitive_memory, and query_memory. Context tags are domain/scope labels.",
                                    "when": "Use when memories belong to specific domains or scopes: 'backend', 'auth-module', 'sprint-12', 'api-design', \
                                             'research-papers', 'meeting-notes'. This prevents cross-domain noise in results."
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
                                    "what_to_store": "Key facts, requirements, decisions, specifications, definitions. Each section → 1-5 atomic memories.",
                                    "source_metadata": "source_file, page_number, section_name, author, date, version",
                                    "context_tags": "document type (spec, requirements, design-doc, manual), project, topic",
                                    "example_content": "API rate limit is 1000 requests/minute per API key, with burst allowance of 50 — see API Design Doc v2.3 section 4.1",
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
                                    "principle": "For large sources (entire codebases, book-length documents, multi-page research), create a HIERARCHY of memories: \
                                                   (1) A high-level overview memory describing the whole source and what it contains. \
                                                   (2) Mid-level memories for each major section/module/chapter. \
                                                   (3) Detail-level memories for specific important facts. \
                                                   Always include source references so the full original can be fetched on demand.",
                                    "retrieval_pattern": "When a query matches a high-level memory, use its source metadata and relations to drill down. \
                                                          The memory system is your MAP; the source files/URLs/documents are the TERRITORY."
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
                                "NEVER_store_raw_content": "CRITICAL: NEVER store raw documents, entire files, full web pages, or large text blocks. \
                                                             Always decompose into atomic memories. One source document → 5-50 memories, each answering one specific question.",

                                "ALWAYS_include_source": "Every memory MUST have source attribution in metadata.custom — file paths, URLs, page numbers, line ranges, \
                                                           commit hashes, dates. This is how you or another agent can fetch the full original when needed. \
                                                           The memory is the INDEX; the source is the CONTENT.",

                                "content_field_subject_focus": "The 'content' field should describe a clear SUBJECT with identifying info. \
                                                                Write it as a complete, searchable statement. Ask: 'If someone searched for this topic, \
                                                                would this content match their query?'",

                                "relations_action_focus": "Relations should use descriptive verb predicates: 'depends_on', 'implements', 'configures', \
                                                           'supersedes', 'authored_by', 'deployed_to'. Avoid vague predicates like 'related_to' or 'has'.",

                                "context_broad_categories": "Context tags should be BROAD categories enabling future discovery. Use 3-5 relevant tags. \
                                                              Think about what domain, layer, project, or topic this belongs to.",

                                "focus_on_what_matters": "Not everything needs to be stored. Prioritize: (1) Information you'll need to recall later. \
                                                          (2) Decisions and their rationale. (3) Non-obvious facts that are hard to re-derive. \
                                                          (4) Connections between concepts. Skip trivial or easily re-derivable information."
                            },

                            "tips": [
                                "Always call system_status first to verify the system is ready.",
                                "Use 'context' tags for soft grouping within a bank; use separate banks for hard isolation.",
                                "Semantic search works by meaning — query 'authentication flow' will find memories about 'JWT token validation' and 'login endpoint'.",
                                "The 'relations' field builds a knowledge graph — useful for connecting modules, services, people, concepts, and dependencies.",
                                "Include source attribution in 'metadata' (file paths, URLs, line numbers) so you can fetch the original material when the memory summary isn't enough.",
                                "When querying returns 0 results, check: (1) Correct bank? (2) Data actually stored? (3) Try different phrasing — semantic search matches meaning, not exact words.",
                                "For large sources, create overview memories first, then detail memories. Use relations to connect them into a navigable hierarchy.",
                                "When you retrieve a memory and need more detail, use its source metadata to fetch the original file, URL, or document section."
                            ]
                        });

                    match serde_json::to_string_pretty(&guide) {
                        Ok(json) => Ok(CallToolResult::success(vec![Content::text(json)])),
                        Err(e) => Err(ErrorData {
                            code: rmcp::model::ErrorCode(-32603).into(),
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
                            let json = serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                                code: rmcp::model::ErrorCode(-32603).into(),
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
                            let json = serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                                code: rmcp::model::ErrorCode(-32603).into(),
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
                            let json = serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                                code: rmcp::model::ErrorCode(-32603).into(),
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
                "status_process_document" => {
                    let args = request.arguments.as_ref().unwrap_or(&empty_args);
                    let payload = map_mcp_arguments_to_payload(args, &self.agent_id);
                    let ops = self
                        .resolve_operations_with_sessions(payload.bank.as_deref())
                        .await?;
                    match ops.status_process_document(payload) {
                        Ok(response) => {
                            let json = serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                                code: rmcp::model::ErrorCode(-32603).into(),
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
                "cancel_process_document" => {
                    let args = request.arguments.as_ref().unwrap_or(&empty_args);
                    let payload = map_mcp_arguments_to_payload(args, &self.agent_id);
                    let ops = self
                        .resolve_operations_with_sessions(payload.bank.as_deref())
                        .await?;
                    match ops.cancel_process_document(payload) {
                        Ok(response) => {
                            let json = serde_json::to_string_pretty(&response).map_err(|e| ErrorData {
                                code: rmcp::model::ErrorCode(-32603).into(),
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
                            code: rmcp::model::ErrorCode(-32602).into(),
                            message: "Missing arguments. 'memory_id' is required.".into(),
                            data: None,
                        })
                    }
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
                _ => Err(ErrorData {
                    code: rmcp::model::ErrorCode(-32601).into(),
                    message: format!("Unknown tool: {}", tool_name).into(),
                    data: None,
                }),
            }
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Recursively compute the total size of a directory in bytes.
fn dir_size_bytes(path: &Path) -> u64 {
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
fn format_bytes(bytes: u64) -> String {
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
