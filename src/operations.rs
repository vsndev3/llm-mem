use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use thiserror::Error;
use tracing::{error, info, warn};

use crate::{
    memory::{MemoryManager, utils::chunk_markdown},
    search::{GraphSearchEngine, TraversalConfig, TraversalDirection},
    types::{Filters, Memory, MemoryMetadata, MemoryType},
};

// ─── Error types ───────────────────────────────────────────────────────────

/// Common error types for memory operations
#[derive(Debug, Error)]
pub enum OperationError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Runtime error: {0}")]
    Runtime(String),

    #[error("Memory not found: {0}")]
    MemoryNotFound(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl From<crate::error::MemoryError> for OperationError {
    fn from(err: crate::error::MemoryError) -> Self {
        OperationError::Runtime(format!("Core error: {}", err))
    }
}

pub type OperationResult<T> = Result<T, OperationError>;

// ─── Request/Response types ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationInput {
    pub relation: String,
    pub target: String,
}

/// Graph traversal configuration for query requests
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphTraversalInput {
    /// Enable graph traversal (default: false)
    pub enabled: Option<bool>,

    /// Maximum traversal depth (default: 2, max: 5)
    pub max_depth: Option<usize>,

    /// Traversal direction: "outgoing", "incoming", or "both" (default: "both")
    pub direction: Option<String>,

    /// Optional relation type filter (e.g., ["derived_from", "mentions"])
    pub relation_types: Option<Vec<String>>,

    /// Maximum number of entry points from semantic search (default: 5)
    pub entry_point_limit: Option<usize>,

    /// Include traversal paths in response (default: false)
    pub include_paths: Option<bool>,
}

impl GraphTraversalInput {
    /// Convert to TraversalConfig
    pub fn to_config(&self) -> Option<TraversalConfig> {
        if self.enabled.unwrap_or(false) {
            let direction = match self.direction.as_deref() {
                Some("outgoing") => TraversalDirection::Outgoing,
                Some("incoming") => TraversalDirection::Incoming,
                _ => TraversalDirection::Both,
            };

            let mut config = TraversalConfig::new().with_direction(direction);

            if let Some(depth) = self.max_depth {
                config = config.with_max_depth(depth);
            }

            if let Some(ref types) = self.relation_types {
                config = config.with_relation_types(types.clone());
            }

            if let Some(limit) = self.entry_point_limit {
                config.entry_point_limit = limit;
            }

            Some(config)
        } else {
            None
        }
    }
}

/// Common data structure for memory operation payloads
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryOperationPayload {
    pub content: Option<String>,
    pub messages: Option<Vec<crate::types::Message>>,
    pub query: Option<String>,
    pub memory_id: Option<String>,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub memory_type: Option<String>,
    pub topics: Option<Vec<String>>,
    pub context: Option<Vec<String>>,
    pub keywords: Option<Vec<String>>,
    pub relations: Option<Vec<RelationInput>>,
    pub source_memory_id: Option<String>, // Links intuitive memory to content memory
    pub limit: Option<usize>,
    pub min_salience: Option<f64>,
    pub k: Option<usize>,
    pub keyword_only: Option<bool>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub created_after: Option<String>,
    pub created_before: Option<String>,
    pub bank: Option<String>,
    pub graph_traversal: Option<GraphTraversalInput>,

    // Document session management
    pub session_id: Option<String>,
    pub part_index: Option<usize>,
    pub file_name: Option<String>,
    pub total_size: Option<usize>,
    pub mime_type: Option<String>,

    // Auto-chunk upload
    pub file_path: Option<String>,
    pub chunk_size: Option<usize>,
    pub process_immediately: Option<bool>,
    pub partial_closure: Option<bool>,

    /// Override for similarity threshold (0.0–1.0)
    pub similarity_threshold: Option<f32>,

    /// Navigation direction: "zoom_in", "zoom_out", or "both"
    pub direction: Option<String>,
    /// Number of levels to traverse when navigating
    pub levels: Option<usize>,
}

/// Common response structure for memory operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOperationResponse {
    pub success: bool,
    pub message: String,
    pub data: Option<serde_json::Value>,
    pub error: Option<String>,
}

impl MemoryOperationResponse {
    pub fn success(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            data: None,
            error: None,
        }
    }

    pub fn success_with_data(message: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            success: true,
            message: message.into(),
            data: Some(data),
            error: None,
        }
    }

    pub fn error(error: impl Into<String>) -> Self {
        Self {
            success: false,
            message: "Operation failed".to_string(),
            data: None,
            error: Some(error.into()),
        }
    }
}

// ─── Parameter helpers ─────────────────────────────────────────────────────

#[derive(Debug)]
pub struct QueryParams {
    pub query: String,
    pub limit: usize,
    pub min_salience: Option<f64>,
    pub memory_type: Option<String>,
    pub topics: Option<Vec<String>>,
    pub context: Option<Vec<String>>,
    pub keyword_only: bool,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub created_after: Option<chrono::DateTime<chrono::Utc>>,
    pub created_before: Option<chrono::DateTime<chrono::Utc>>,
    pub graph_traversal: Option<TraversalConfig>,
    pub include_paths: bool,
    pub similarity_threshold: Option<f32>,
}

impl QueryParams {
    pub fn from_payload(
        payload: &MemoryOperationPayload,
        default_limit: usize,
    ) -> OperationResult<Self> {
        let query = payload
            .query
            .as_ref()
            .ok_or_else(|| OperationError::InvalidInput("Query is required".to_string()))?
            .clone();

        let limit = payload.limit.or(payload.k).unwrap_or(default_limit);

        let created_after = payload
            .created_after
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        let created_before = payload
            .created_before
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        let graph_traversal = payload
            .graph_traversal
            .as_ref()
            .and_then(|gt| gt.to_config());

        let include_paths = payload
            .graph_traversal
            .as_ref()
            .and_then(|gt| gt.include_paths)
            .unwrap_or(false);

        Ok(Self {
            query,
            limit,
            min_salience: payload.min_salience,
            memory_type: payload.memory_type.clone(),
            topics: payload.topics.clone(),
            context: payload.context.clone(),
            keyword_only: payload.keyword_only.unwrap_or(false),
            user_id: payload.user_id.clone(),
            agent_id: payload.agent_id.clone(),
            created_after,
            created_before,
            graph_traversal,
            include_paths,
            similarity_threshold: payload.similarity_threshold,
        })
    }
}

pub struct StoreParams {
    pub content: String,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub memory_type: String,
    pub topics: Option<Vec<String>>,
    pub context: Option<Vec<String>>,
    pub relations: Option<Vec<RelationInput>>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl StoreParams {
    pub fn from_payload(
        payload: &MemoryOperationPayload,
        default_user_id: Option<String>,
        default_agent_id: Option<String>,
    ) -> OperationResult<Self> {
        let content = payload
            .content
            .as_ref()
            .ok_or_else(|| {
                OperationError::InvalidInput(
                    "Content is required for store_memory. \
                 Please provide a single, atomic fact as a string. \
                 Example: { \"content\": \"The user's favorite color is blue.\" }"
                        .to_string(),
                )
            })?
            .clone();

        let user_id = payload.user_id.clone().or(default_user_id);

        let agent_id = payload.agent_id.clone().or(default_agent_id);

        let memory_type = payload
            .memory_type
            .clone()
            .unwrap_or_else(|| "conversational".to_string());

        Ok(Self {
            content,
            user_id,
            agent_id,
            memory_type,
            topics: payload.topics.clone(),
            context: payload.context.clone(),
            relations: payload.relations.clone(),
            metadata: payload.metadata.clone(),
        })
    }
}

pub struct AddMemoryParams {
    pub messages: Vec<crate::types::Message>,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub memory_type: String,
    pub topics: Option<Vec<String>>,
    pub context: Option<Vec<String>>,
    pub relations: Option<Vec<RelationInput>>,
    pub source_memory_id: Option<String>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl AddMemoryParams {
    pub fn from_payload(
        payload: &MemoryOperationPayload,
        default_user_id: Option<String>,
        default_agent_id: Option<String>,
    ) -> OperationResult<Self> {
        let messages = payload.messages.clone().unwrap_or_default();

        if messages.is_empty() {
            return Err(OperationError::InvalidInput(
                "Messages are required for add_memory. \
                 Please provide an array of message objects, each containing 'role' and 'content'. \
                 Example: { \"messages\": [ { \"role\": \"user\", \"content\": \"Hello\" } ] }"
                    .to_string(),
            ));
        }

        let user_id = payload.user_id.clone().or(default_user_id);

        let agent_id = payload.agent_id.clone().or(default_agent_id);

        let memory_type = payload
            .memory_type
            .clone()
            .unwrap_or_else(|| "conversational".to_string());

        Ok(Self {
            messages,
            user_id,
            agent_id,
            memory_type,
            topics: payload.topics.clone(),
            context: payload.context.clone(),
            relations: payload.relations.clone(),
            source_memory_id: payload.source_memory_id.clone(),
            metadata: payload.metadata.clone(),
        })
    }
}

pub struct IngestDocumentParams {
    pub content: String,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub memory_type: String,
    pub topics: Option<Vec<String>>,
    pub context: Option<Vec<String>>,
    pub relations: Option<Vec<RelationInput>>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl IngestDocumentParams {
    pub fn from_payload(
        payload: &MemoryOperationPayload,
        default_user_id: Option<String>,
        default_agent_id: Option<String>,
    ) -> OperationResult<Self> {
        let content = payload
            .content
            .as_ref()
            .ok_or_else(|| {
                OperationError::InvalidInput(
                    "Content is required for ingest_document. \
                 Please provide the full text of the document as a string. \
                 Example: { \"content\": \"This is the document text...\" }"
                        .to_string(),
                )
            })?
            .clone();

        let user_id = payload.user_id.clone().or(default_user_id);

        let agent_id = payload.agent_id.clone().or(default_agent_id);

        let memory_type = payload
            .memory_type
            .clone()
            .unwrap_or_else(|| "semantic".to_string());

        Ok(Self {
            content,
            user_id,
            agent_id,
            memory_type,
            topics: payload.topics.clone(),
            context: payload.context.clone(),
            relations: payload.relations.clone(),
            metadata: payload.metadata.clone(),
        })
    }
}

pub struct FilterParams {
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub memory_type: Option<String>,
    pub limit: usize,
    pub created_after: Option<chrono::DateTime<chrono::Utc>>,
    pub created_before: Option<chrono::DateTime<chrono::Utc>>,
    pub relations: Option<Vec<RelationInput>>,
}

pub struct BeginStoreDocumentParams {
    pub file_name: String,
    pub file_type: Option<String>,
    pub total_size: usize,
    pub md5sum: Option<String>,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub memory_type: String,
    pub topics: Option<Vec<String>>,
    pub context: Option<Vec<String>>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl BeginStoreDocumentParams {
    pub fn from_payload(
        payload: &MemoryOperationPayload,
        default_user_id: Option<String>,
        default_agent_id: Option<String>,
    ) -> OperationResult<Self> {
        let file_name = payload
            .file_name
            .clone()
            .or_else(|| {
                payload
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get("file_name"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .ok_or_else(|| {
                OperationError::InvalidInput(
                    "file_name is required for begin_store_document.".to_string(),
                )
            })?;

        let total_size = payload
            .total_size
            .or_else(|| {
                payload
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get("total_size"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
            })
            .ok_or_else(|| {
                OperationError::InvalidInput(
                    "total_size is required for begin_store_document.".to_string(),
                )
            })?;

        let file_type = payload.mime_type.clone().or_else(|| {
            payload
                .metadata
                .as_ref()
                .and_then(|m| m.get("file_type"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        });

        let md5sum = payload
            .metadata
            .as_ref()
            .and_then(|m| m.get("md5sum"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let user_id = payload.user_id.clone().or(default_user_id);
        let agent_id = payload.agent_id.clone().or(default_agent_id);

        let memory_type = payload
            .memory_type
            .clone()
            .unwrap_or_else(|| "semantic".to_string());

        Ok(Self {
            file_name,
            file_type,
            total_size,
            md5sum,
            user_id,
            agent_id,
            memory_type,
            topics: payload.topics.clone(),
            context: payload.context.clone(),
            metadata: payload.metadata.clone(),
        })
    }
}

pub struct StoreDocumentPartParams {
    pub session_id: String,
    pub part_index: usize,
    pub content: String,
}

impl StoreDocumentPartParams {
    pub fn from_payload(payload: &MemoryOperationPayload) -> OperationResult<Self> {
        let session_id = payload
            .session_id
            .clone()
            .or_else(|| payload.memory_id.clone())
            .ok_or_else(|| {
                OperationError::InvalidInput(
                    "session_id is required for store_document_part".to_string(),
                )
            })?;

        let part_index = payload
            .part_index
            .or_else(|| {
                payload
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get("part_index"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
            })
            .ok_or_else(|| {
                OperationError::InvalidInput(
                    "part_index is required for store_document_part".to_string(),
                )
            })?;

        let content = payload
            .content
            .as_ref()
            .ok_or_else(|| {
                OperationError::InvalidInput(
                    "content is required for store_document_part".to_string(),
                )
            })?
            .clone();

        Ok(Self {
            session_id,
            part_index,
            content,
        })
    }
}

pub struct ProcessDocumentParams {
    pub session_id: String,
    pub partial_closure: bool,
}

impl ProcessDocumentParams {
    pub fn from_payload(payload: &MemoryOperationPayload) -> OperationResult<Self> {
        let session_id = payload
            .session_id
            .clone()
            .or_else(|| payload.memory_id.clone())
            .ok_or_else(|| {
                OperationError::InvalidInput(
                    "session_id is required for process_document".to_string(),
                )
            })?;

        let partial_closure = payload.partial_closure.unwrap_or(false);

        Ok(Self {
            session_id,
            partial_closure,
        })
    }
}

pub struct UploadDocumentParams {
    pub file_path: String,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
    pub memory_type: Option<String>,
    pub topics: Option<Vec<String>>,
    pub context: Option<Vec<String>>,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub chunk_size: Option<usize>,
    pub process_immediately: bool,
}

impl UploadDocumentParams {
    pub fn from_payload(payload: &MemoryOperationPayload) -> OperationResult<Self> {
        let file_path = payload
            .file_path
            .as_ref()
            .ok_or_else(|| {
                OperationError::InvalidInput(
                    "file_path is required for upload_document".to_string(),
                )
            })?
            .clone();

        let file_name = payload.file_name.clone();
        let mime_type = payload.mime_type.clone();
        let memory_type = payload.memory_type.clone();
        let topics = payload.topics.clone();
        let context = payload.context.clone();
        let user_id = payload.user_id.clone();
        let agent_id = payload.agent_id.clone();
        let chunk_size = payload.chunk_size;
        let process_immediately = payload.process_immediately.unwrap_or(true);

        Ok(Self {
            file_path,
            file_name,
            mime_type,
            memory_type,
            topics,
            context,
            user_id,
            agent_id,
            chunk_size,
            process_immediately,
        })
    }
}

pub struct StatusProcessDocumentParams {
    pub session_id: String,
}

impl StatusProcessDocumentParams {
    pub fn from_payload(payload: &MemoryOperationPayload) -> OperationResult<Self> {
        let session_id = payload
            .session_id
            .clone()
            .or_else(|| payload.memory_id.clone())
            .ok_or_else(|| {
                OperationError::InvalidInput(
                    "session_id is required for status_process_document".to_string(),
                )
            })?;

        Ok(Self { session_id })
    }
}

pub struct CancelProcessDocumentParams {
    pub session_id: String,
}

impl CancelProcessDocumentParams {
    pub fn from_payload(payload: &MemoryOperationPayload) -> OperationResult<Self> {
        let session_id = payload
            .session_id
            .clone()
            .or_else(|| payload.memory_id.clone())
            .ok_or_else(|| {
                OperationError::InvalidInput(
                    "session_id is required for cancel_process_document".to_string(),
                )
            })?;

        Ok(Self { session_id })
    }
}

impl FilterParams {
    pub fn from_payload(
        payload: &MemoryOperationPayload,
        default_limit: usize,
    ) -> OperationResult<Self> {
        let limit = payload.limit.or(payload.k).unwrap_or(default_limit);

        let created_after = payload
            .created_after
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        let created_before = payload
            .created_before
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        let _graph_traversal = payload
            .graph_traversal
            .as_ref()
            .and_then(|gt| gt.to_config());

        let _include_paths = payload
            .graph_traversal
            .as_ref()
            .and_then(|gt| gt.include_paths)
            .unwrap_or(false);

        Ok(Self {
            user_id: payload.user_id.clone(),
            agent_id: payload.agent_id.clone(),
            memory_type: payload.memory_type.clone(),
            limit,
            created_after,
            created_before,
            relations: payload.relations.clone(),
        })
    }
}

// ─── Core operations ───────────────────────────────────────────────────────

use crate::document_session::{
    DocumentMetadata, DocumentSession, DocumentSessionManager, ProcessingResult, SessionStatus,
};

/// Core operations handler for memory tools
pub struct MemoryOperations {
    memory_manager: std::sync::Arc<MemoryManager>,
    session_manager: Option<std::sync::Arc<DocumentSessionManager>>,
    default_user_id: Option<String>,
    default_agent_id: Option<String>,
    default_limit: usize,
}

impl MemoryOperations {
    pub fn new(
        memory_manager: std::sync::Arc<MemoryManager>,
        default_user_id: Option<String>,
        default_agent_id: Option<String>,
        default_limit: usize,
    ) -> Self {
        Self {
            memory_manager,
            session_manager: None,
            default_user_id,
            default_agent_id,
            default_limit,
        }
    }

    pub fn with_session_manager(
        memory_manager: std::sync::Arc<MemoryManager>,
        session_manager: std::sync::Arc<DocumentSessionManager>,
        default_user_id: Option<String>,
        default_agent_id: Option<String>,
        default_limit: usize,
    ) -> Self {
        Self {
            memory_manager,
            session_manager: Some(session_manager),
            default_user_id,
            default_agent_id,
            default_limit,
        }
    }

    pub async fn store_memory(
        &self,
        payload: MemoryOperationPayload,
    ) -> OperationResult<MemoryOperationResponse> {
        let params = StoreParams::from_payload(
            &payload,
            self.default_user_id.clone(),
            self.default_agent_id.clone(),
        )?;

        info!("Storing memory for user: {:?}", params.user_id);

        let memory_type = MemoryType::parse_with_result(&params.memory_type)
            .map_err(|e| OperationError::InvalidInput(format!("Invalid memory_type: {}", e)))?;

        let mut metadata = MemoryMetadata::new(memory_type);
        metadata.user_id = params.user_id.clone();
        metadata.agent_id = params.agent_id.clone();

        if let Some(topics) = params.topics {
            metadata.topics = topics;
        }

        if let Some(context) = params.context {
            metadata.context = context;
        }

        if let Some(raw_relations) = params.relations {
            metadata.relations = raw_relations
                .into_iter()
                .map(|r| crate::types::Relation {
                    source: "SELF".to_string(), // Correctly set after storage
                    relation: r.relation,
                    target: r.target,
                    strength: None, // Default to None, can be set later if needed
                })
                .collect();
        }

        if let Some(custom_metadata) = params.metadata {
            metadata.custom = custom_metadata;
        }

        match self.memory_manager.store(params.content, metadata).await {
            Ok(memory_id) => {
                // Now that we have the memory ID, we need to update the source of the relations
                // However, since we store the memory in one go, the current architecture
                // relies on the store function to handle persistence.
                // For a Level 1.5 graph, "SELF" is a placeholder that implicitly means "this memory".
                // Detailed graph logic would replace "SELF" with the actual ID post-creation if we had a graph DB.

                info!("Memory stored successfully with ID: {}", memory_id);
                let data = json!({
                    "memory_id": memory_id,
                    "user_id": params.user_id,
                    "agent_id": params.agent_id
                });
                Ok(MemoryOperationResponse::success_with_data(
                    "Memory stored successfully",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to store memory: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to store memory: {}",
                    e
                )))
            }
        }
    }

    pub async fn add_memory(
        &self,
        payload: MemoryOperationPayload,
    ) -> OperationResult<MemoryOperationResponse> {
        let params = AddMemoryParams::from_payload(
            &payload,
            self.default_user_id.clone(),
            self.default_agent_id.clone(),
        )?;

        info!(
            "Adding memory from conversation for user: {:?}",
            params.user_id
        );

        let memory_type = MemoryType::parse_with_result(&params.memory_type)
            .map_err(|e| OperationError::InvalidInput(format!("Invalid memory_type: {}", e)))?;

        let mut metadata = MemoryMetadata::new(memory_type);
        metadata.user_id = params.user_id.clone();
        metadata.agent_id = params.agent_id.clone();

        if let Some(topics) = params.topics {
            metadata.topics = topics;
        }

        if let Some(context) = params.context {
            metadata.context = context;
        }

        if let Some(raw_relations) = params.relations {
            metadata.relations = raw_relations
                .into_iter()
                .map(|r| crate::types::Relation {
                    source: "SELF".to_string(),
                    relation: r.relation,
                    target: r.target,
                    strength: None,
                })
                .collect();
        }

        // Add automatic relation to source memory (for linking intuitive memories to content memories)
        if let Some(source_id) = params.source_memory_id {
            metadata.relations.push(crate::types::Relation {
                source: "SELF".to_string(),
                relation: "derived_from".to_string(),
                target: source_id,
                strength: None,
            });
        }

        if let Some(custom_metadata) = params.metadata {
            metadata.custom = custom_metadata;
        }

        match self
            .memory_manager
            .add_memory(&params.messages, metadata)
            .await
        {
            Ok(results) => {
                info!(
                    "Memory added successfully, {} actions performed",
                    results.len()
                );
                let data = json!({
                    "results": results,
                    "user_id": params.user_id,
                    "agent_id": params.agent_id
                });
                Ok(MemoryOperationResponse::success_with_data(
                    "Memory added successfully",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to add memory: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to add memory: {}",
                    e
                )))
            }
        }
    }

    pub async fn ingest_document(
        &self,
        payload: MemoryOperationPayload,
    ) -> OperationResult<MemoryOperationResponse> {
        let params = IngestDocumentParams::from_payload(
            &payload,
            self.default_user_id.clone(),
            self.default_agent_id.clone(),
        )?;

        info!("Ingesting document for user: {:?}", params.user_id);

        let memory_type = MemoryType::parse_with_result(&params.memory_type)
            .map_err(|e| OperationError::InvalidInput(format!("Invalid memory_type: {}", e)))?;

        let mut metadata = MemoryMetadata::new(memory_type);
        metadata.user_id = params.user_id.clone();
        metadata.agent_id = params.agent_id.clone();

        if let Some(topics) = params.topics {
            metadata.topics = topics;
        }

        if let Some(context) = params.context {
            metadata.context = context;
        }

        if let Some(raw_relations) = params.relations {
            metadata.relations = raw_relations
                .into_iter()
                .map(|r| crate::types::Relation {
                    source: "SELF".to_string(),
                    relation: r.relation,
                    target: r.target,
                    strength: None,
                })
                .collect();
        }

        if let Some(custom_metadata) = params.metadata {
            metadata.custom = custom_metadata;
        }

        match self
            .memory_manager
            .ingest_document(&params.content, metadata)
            .await
        {
            Ok(results) => {
                info!(
                    "Document ingested successfully, {} actions performed",
                    results.len()
                );
                let data = json!({
                    "results": results,
                    "user_id": params.user_id,
                    "agent_id": params.agent_id
                });
                Ok(MemoryOperationResponse::success_with_data(
                    "Document ingested successfully",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to add memory: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to add memory: {}",
                    e
                )))
            }
        }
    }

    pub async fn update_memory(
        &self,
        payload: MemoryOperationPayload,
    ) -> OperationResult<MemoryOperationResponse> {
        let memory_id = payload
            .memory_id
            .ok_or_else(|| OperationError::InvalidInput("Memory ID is required".to_string()))?;

        info!("Updating memory: {}", memory_id);

        let relations = payload.relations.map(|rels| {
            rels.into_iter()
                .map(|r| crate::types::Relation {
                    source: "SELF".to_string(),
                    relation: r.relation,
                    target: r.target,
                    strength: None,
                })
                .collect()
        });

        match self
            .memory_manager
            .update(&memory_id, payload.content, relations)
            .await
        {
            Ok(_) => Ok(MemoryOperationResponse::success(
                "Memory updated successfully",
            )),
            Err(e) => {
                error!("Failed to update memory: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to update memory: {}",
                    e
                )))
            }
        }
    }

    pub async fn query_memory(
        &self,
        payload: MemoryOperationPayload,
    ) -> OperationResult<MemoryOperationResponse> {
        let params = QueryParams::from_payload(&payload, self.default_limit)?;

        info!("Querying memories with query: {}", params.query);

        let memory_type = params
            .memory_type
            .map(|t| MemoryType::parse_with_result(&t))
            .transpose()
            .map_err(|e| OperationError::InvalidInput(format!("Invalid memory_type: {}", e)))?;

        let mut filters = Filters::default();

        if let Some(user_id) = params.user_id {
            filters.user_id = Some(user_id);
        }
        if let Some(agent_id) = params.agent_id {
            filters.agent_id = Some(agent_id);
        }
        if let Some(memory_type) = memory_type {
            filters.memory_type = Some(memory_type);
        }
        if let Some(topics) = params.topics {
            filters.topics = Some(topics);
        }
        if let Some(created_after) = params.created_after {
            filters.created_after = Some(created_after);
        }
        if let Some(created_before) = params.created_before {
            filters.created_before = Some(created_before);
        }

        // Pass keyword_only flag to filters for hybrid search
        if params.keyword_only {
            filters
                .custom
                .insert("keyword_only".to_string(), serde_json::Value::Bool(true));
        }

        // Check if graph traversal is enabled
        let search_result = if let Some(ref graph_config) = params.graph_traversal {
            info!("Graph traversal enabled, performing graph search");

            // Phase 1: Get entry points from semantic search
            let entry_point_limit = graph_config.entry_point_limit.min(10);
            let entry_memories = if let Some(ref context_tags) = params.context {
                self.memory_manager
                    .search_with_context(&params.query, context_tags, &filters, entry_point_limit)
                    .await?
            } else {
                self.memory_manager
                    .search(&params.query, &filters, entry_point_limit)
                    .await?
            };

            if entry_memories.is_empty() {
                info!("No entry points found for graph traversal");
                return Ok(MemoryOperationResponse::success_with_data(
                    "Graph traversal: No entry points found",
                    json!({
                        "count": 0,
                        "message": "No matching memories found to use as graph traversal entry points",
                        "memories": []
                    }),
                ));
            }

            // Phase 2: Get all memories for graph traversal (needed for relation lookup)
            let all_memories = self.memory_manager.list(&Filters::default(), None).await?;

            // Phase 3: Traverse graph
            let engine = GraphSearchEngine::new(graph_config.clone())
                .map_err(|e| OperationError::Runtime(format!("Invalid graph config: {}", e)))?;

            let entry_tuples: Vec<(Memory, f32)> = entry_memories
                .into_iter()
                .map(|sm| (sm.memory, sm.score))
                .collect();

            let graph_results = engine
                .traverse(entry_tuples, &all_memories, graph_config)
                .await
                .map_err(|e| OperationError::Runtime(format!("Graph traversal failed: {}", e)))?;

            // Convert graph results to response format
            let memories_json: Vec<Value> = graph_results
                .iter()
                .take(params.limit)
                .map(|gr| {
                    let mut memory_json = memory_to_json(&gr.memory);

                    // Add graph info if requested
                    if params.include_paths {
                        let graph_info = json!({
                            "entry_distance": gr.entry_distance,
                            "path_from_entry": gr.path_from_entry,
                            "relation_boost": gr.relation_boost,
                            "final_score": gr.final_score,
                            "semantic_score": gr.semantic_score,
                        });
                        memory_json["graph_info"] = graph_info;
                    }

                    memory_json
                })
                .collect();

            let count = memories_json.len();
            let message = format!(
                "Graph search returned {} memories (depth: {})",
                count, graph_config.max_depth
            );

            let data = json!({
                "count": count,
                "message": message,
                "graph_traversal": true,
                "memories": memories_json
            });

            return Ok(MemoryOperationResponse::success_with_data(&message, data));
        } else {
            // Standard semantic search (no graph traversal)
            if let Some(ref context_tags) = params.context {
                self.memory_manager
                    .search_with_context(&params.query, context_tags, &filters, params.limit)
                    .await
            } else {
                self.memory_manager
                    .search_with_override(&params.query, &filters, params.limit, params.similarity_threshold)
                    .await
            }
        };

        match search_result {
            Ok(memories) => {
                let count = memories.len();
                let best_score = memories.first().map(|m| m.score);

                info!("Found {} memories", count);

                let memories_json: Vec<Value> = memories
                    .into_iter()
                    .map(|scored_memory| memory_to_json(&scored_memory.memory))
                    .collect();

                // Build informative message for the user
                let message = if count == 0 {
                    let threshold_hint = if let Some(th) = params.similarity_threshold {
                        format!(" Current --threshold: {:.2}.", th)
                    } else {
                        " Try passing --threshold 0.1 to lower the similarity cutoff.".to_string()
                    };
                    format!(
                        "Query returned 0 memories. All candidates may have been filtered by the similarity threshold.{}",
                        threshold_hint
                    )
                } else {
                    match best_score {
                        Some(score) => format!(
                            "Query returned {} memories. Best match score: {:.4}",
                            count, score
                        ),
                        None => format!("Query returned {} memories", count),
                    }
                };

                let data = json!({
                    "count": count,
                    "best_score": best_score,
                    "message": message,
                    "graph_traversal": false,
                    "memories": memories_json
                });

                Ok(MemoryOperationResponse::success_with_data(&message, data))
            }
            Err(e) => {
                error!("Failed to query memories: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to query memories: {}",
                    e
                )))
            }
        }
    }

    pub async fn list_memories(
        &self,
        payload: MemoryOperationPayload,
    ) -> OperationResult<MemoryOperationResponse> {
        let params = FilterParams::from_payload(&payload, self.default_limit)?;

        info!("Listing memories with filters");

        let mut filters = Filters::default();

        if let Some(user_id) = params.user_id {
            filters.user_id = Some(user_id);
        }
        if let Some(agent_id) = params.agent_id {
            filters.agent_id = Some(agent_id);
        }
        if let Some(memory_type) = params.memory_type
            && let Ok(mt) = MemoryType::parse_with_result(&memory_type)
        {
            filters.memory_type = Some(mt);
        }
        if let Some(created_after) = params.created_after {
            filters.created_after = Some(created_after);
        }
        if let Some(created_before) = params.created_before {
            filters.created_before = Some(created_before);
        }
        if let Some(relations) = params.relations {
            filters.relations = Some(
                relations
                    .into_iter()
                    .map(|r| crate::types::RelationFilter {
                        relation: r.relation,
                        target: r.target,
                    })
                    .collect(),
            );
        }

        match self.memory_manager.list(&filters, Some(params.limit)).await {
            Ok(memories) => {
                let count = memories.len();
                info!("Listed {} memories", count);

                let memories_json: Vec<Value> = memories
                    .into_iter()
                    .map(|memory| memory_to_json(&memory))
                    .collect();

                let data = json!({
                    "count": count,
                    "memories": memories_json
                });

                Ok(MemoryOperationResponse::success_with_data(
                    "List completed successfully",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to list memories: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to list memories: {}",
                    e
                )))
            }
        }
    }

    pub async fn get_memory(
        &self,
        payload: MemoryOperationPayload,
    ) -> OperationResult<MemoryOperationResponse> {
        let memory_id = payload
            .memory_id
            .ok_or_else(|| OperationError::InvalidInput("Memory ID is required".to_string()))?;

        info!("Getting memory with ID: {}", memory_id);

        match self.memory_manager.get(&memory_id).await {
            Ok(Some(memory)) => {
                let mut memory_json = memory_to_json(&memory);

                // Enrich with reverse-direction links: which higher-layer memories
                // abstract FROM this one (zoom_out targets).
                if let Ok(dependents) =
                    self.memory_manager.find_abstraction_dependents(&memory_id).await
                    && !dependents.is_empty()
                {
                    let ids: Vec<Value> = dependents
                        .iter()
                        .map(|m| Value::String(m.id.clone()))
                        .collect();
                    if let Some(meta) = memory_json.get_mut("metadata") {
                        meta.as_object_mut()
                            .map(|obj| obj.insert("abstracted_into".into(), Value::Array(ids)));
                    }
                }

                let data = json!({
                    "memory": memory_json
                });
                Ok(MemoryOperationResponse::success_with_data(
                    "Memory retrieved successfully",
                    data,
                ))
            }
            Ok(None) => {
                error!("Memory not found: {}", memory_id);
                Err(OperationError::MemoryNotFound(memory_id))
            }
            Err(e) => {
                error!("Failed to get memory: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to get memory: {}",
                    e
                )))
            }
        }
    }

    /// Navigate the abstraction hierarchy from a memory node.
    /// Allows LLM clients to traverse both towards abstraction (zoom_out)
    /// and towards detail (zoom_in).
    pub async fn navigate_memory(
        &self,
        payload: MemoryOperationPayload,
    ) -> OperationResult<MemoryOperationResponse> {
        let memory_id = payload
            .memory_id
            .ok_or_else(|| OperationError::InvalidInput("Memory ID is required".to_string()))?;

        let direction = payload.direction.as_deref().unwrap_or("both");

        let levels = payload.levels.unwrap_or(1).min(5);

        info!(
            "Navigating memory {}: direction={}, levels={}",
            memory_id, direction, levels
        );

        match self
            .memory_manager
            .navigate_memory(&memory_id, direction, levels)
            .await
        {
            Ok(nav_result) => {
                let zoom_in_json: Vec<Value> = nav_result
                    .zoom_in
                    .iter()
                    .map(memory_to_json)
                    .collect();
                let zoom_out_json: Vec<Value> = nav_result
                    .zoom_out
                    .iter()
                    .map(memory_to_json)
                    .collect();

                let data = json!({
                    "source_memory_id": nav_result.source_memory_id,
                    "source_layer": nav_result.source_layer,
                    "zoom_in": zoom_in_json,
                    "zoom_out": zoom_out_json,
                });
                Ok(MemoryOperationResponse::success_with_data(
                    "Navigation completed",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to navigate memory: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to navigate memory: {}",
                    e
                )))
            }
        }
    }

    // ─── Document Session Operations ─────────────────────────────────────────

    pub fn begin_store_document(
        &self,
        payload: MemoryOperationPayload,
    ) -> OperationResult<MemoryOperationResponse> {
        let session_manager = self.session_manager.as_ref().ok_or_else(|| {
            OperationError::Runtime("Document session manager not configured".to_string())
        })?;

        let params = BeginStoreDocumentParams::from_payload(
            &payload,
            self.default_user_id.clone(),
            self.default_agent_id.clone(),
        )?;

        info!(
            "Beginning document storage session for file: {}",
            params.file_name
        );

        let metadata = DocumentMetadata {
            file_name: params.file_name,
            file_type: params.file_type,
            total_size: params.total_size,
            md5sum: params.md5sum,
            user_id: params.user_id,
            agent_id: params.agent_id,
            memory_type: params.memory_type,
            topics: params.topics,
            context: params.context,
            custom_metadata: params
                .metadata
                .map(|m| serde_json::Value::Object(m.into_iter().collect())),
        };

        match session_manager.begin_session(metadata) {
            Ok(response) => {
                info!("Created document session: {}", response.session_id);
                let data =
                    serde_json::to_value(&response).map_err(OperationError::Serialization)?;
                Ok(MemoryOperationResponse::success_with_data(
                    "Document session created",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to create document session: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to create document session: {}",
                    e
                )))
            }
        }
    }

    pub fn store_document_part(
        &self,
        payload: MemoryOperationPayload,
    ) -> OperationResult<MemoryOperationResponse> {
        let session_manager = self.session_manager.as_ref().ok_or_else(|| {
            OperationError::Runtime("Document session manager not configured".to_string())
        })?;

        let params = StoreDocumentPartParams::from_payload(&payload)?;

        info!(
            "Storing document part {} for session {}",
            params.part_index, params.session_id
        );

        match session_manager.store_part(&params.session_id, params.part_index, &params.content) {
            Ok(()) => {
                // Get session info for progress reporting
                let session = session_manager.get_session(&params.session_id);
                let (received, expected) = session
                    .map(|s| (s.received_parts, s.expected_parts))
                    .unwrap_or((params.part_index + 1, 0));

                let remaining = expected.saturating_sub(received);
                let progress_msg = if expected > 0 {
                    format!(
                        "Part {} stored for session {} ({}/{}, {} remaining)",
                        params.part_index, params.session_id, received, expected, remaining
                    )
                } else {
                    format!(
                        "Part {} stored for session {}",
                        params.part_index, params.session_id
                    )
                };

                // Include progress data in response
                let data = json!({
                    "session_id": params.session_id,
                    "part_index": params.part_index,
                    "received_parts": received,
                    "expected_parts": expected,
                    "remaining_parts": remaining
                });

                Ok(MemoryOperationResponse::success_with_data(
                    progress_msg,
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to store document part: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to store document part: {}",
                    e
                )))
            }
        }
    }

    pub async fn upload_document(
        &self,
        payload: MemoryOperationPayload,
    ) -> OperationResult<MemoryOperationResponse> {
        let session_manager = self.session_manager.as_ref().ok_or_else(|| {
            OperationError::Runtime("Document session manager not configured".to_string())
        })?;

        let params = UploadDocumentParams::from_payload(&payload)?;

        info!(
            "Auto-chunk upload: file={}, process_immediately={}",
            params.file_path, params.process_immediately
        );

        let file_path = std::path::Path::new(&params.file_path);

        if !file_path.exists() {
            return Err(OperationError::InvalidInput(format!(
                "File not found: {}",
                params.file_path
            )));
        }

        // Read file content (for large files, could be streamed from disk in background)
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| OperationError::Runtime(format!("Failed to read file: {}", e)))?;

        let file_name = params.file_name.unwrap_or_else(|| {
            file_path
                .file_name()
                .unwrap_or(std::ffi::OsStr::new("unknown"))
                .to_string_lossy()
                .to_string()
        });

        let total_size = content.len();
        let chunk_size = params
            .chunk_size
            .unwrap_or(self.memory_manager.config().document_chunk_size);

        // Calculate expected chunks (char-based) BEFORE creating session
        let total_chars = content.chars().count();
        let expected_chunks = total_chars.div_ceil(chunk_size).max(1);

        // Create session
        use crate::document_session::{DocumentMetadata, SessionStatus};

        // Store file_path in custom_metadata for resume support
        let custom_metadata = Some(json!({
            "file_path": params.file_path
        }));

        let metadata = DocumentMetadata {
            file_name: file_name.clone(),
            file_type: Some(params.mime_type.unwrap_or_else(|| "text/plain".to_string())),
            total_size,
            md5sum: Some(format!("{:x}", md5::compute(&content))),
            user_id: params.user_id,
            agent_id: params.agent_id,
            memory_type: params.memory_type.unwrap_or_else(|| "semantic".to_string()),
            topics: params.topics,
            context: params.context,
            custom_metadata,
        };

        let session_response = session_manager
            .begin_session(metadata)
            .map_err(|e| OperationError::Runtime(format!("Failed to create session: {}", e)))?;

        let session_id = session_response.session_id;

        // Update session with correct char-based chunk count
        session_manager
            .update_expected_parts(&session_id, expected_chunks)
            .map_err(|e| {
                OperationError::Runtime(format!("Failed to update expected parts: {}", e))
            })?;

        info!(
            "Created session {} for file {} ({} bytes, chunk size: {} bytes, {} chunks)",
            session_id, file_name, total_size, chunk_size, expected_chunks
        );

        // Clone variables for background task and response
        let session_id_clone = session_id.clone();
        let file_name_clone = file_name.clone();
        let content_clone = content.clone();
        let session_manager_clone = session_manager.clone();
        let memory_manager_clone = self.memory_manager.clone();
        let process_immediately = params.process_immediately;

        // Spawn background task for chunk upload + processing (non-blocking)
        tokio::spawn(async move {
            info!(
                "Background task: uploading {} in {} chunks",
                file_name_clone, expected_chunks
            );

            // Check for existing parts (resume support)
            let existing_parts = session_manager_clone
                .get_parts(&session_id_clone)
                .unwrap_or_default();
            let already_uploaded = existing_parts.len();
            
            if already_uploaded > 0 {
                info!(
                    "Resuming upload: {} chunks already exist, will skip them",
                    already_uploaded
                );
            }

            // Stream chunks one-by-one
            let chars: Vec<char> = content_clone.chars().collect();
            let total_chars = chars.len();
            let mut actual_parts = 0;
            let mut offset = 0;

            let _ = session_manager_clone.update_status(
                &session_id_clone,
                SessionStatus::Uploading,
                None,
            );

            while offset < total_chars {
                let end = std::cmp::min(offset + chunk_size, total_chars);
                let chunk: String = chars[offset..end].iter().collect();

                // Skip already uploaded parts (resume support)
                if actual_parts < already_uploaded {
                    offset = end;
                    actual_parts += 1;
                    continue;
                }

                if let Err(e) =
                    session_manager_clone.store_part(&session_id_clone, actual_parts, &chunk)
                {
                    error!("Failed to store chunk {}: {}", actual_parts, e);
                    let _ = session_manager_clone.update_status(
                        &session_id_clone,
                        SessionStatus::Failed,
                        Some(&format!("Chunk upload failed: {}", e)),
                    );
                    return;
                }

                actual_parts += 1;
                offset = end;

                // Log progress: every chunk for small uploads (< 20), every 10 chunks for medium, every 100 for large
                let log_interval = if expected_chunks <= 20 { 1 } else if expected_chunks <= 100 { 10 } else { 100 };
                if actual_parts % log_interval == 0 || actual_parts == expected_chunks {
                    info!(
                        "Uploaded {}/{} chunks ({:.0}%)",
                        actual_parts,
                        expected_chunks,
                        (actual_parts as f64 / expected_chunks as f64) * 100.0
                    );
                }
            }

            info!(
                "Stored all {} chunks for session {}",
                actual_parts, session_id_clone
            );

            // Update session with actual chunk count
            let _ = session_manager_clone.update_expected_parts(&session_id_clone, actual_parts);

            // Optionally start processing
            if process_immediately {
                // Small delay to ensure all parts are committed to DB
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                match session_manager_clone.get_session(&session_id_clone) {
                    Ok(session) => {
                        let _ = session_manager_clone.update_status(
                            &session_id_clone,
                            SessionStatus::Processing,
                            None,
                        );

                        let parts = match session_manager_clone.get_parts(&session_id_clone) {
                            Ok(p) => p,
                            Err(e) => {
                                error!("Failed to get parts: {}", e);
                                let _ = session_manager_clone.update_status(
                                    &session_id_clone,
                                    SessionStatus::Failed,
                                    Some(&format!("Failed to get parts: {}", e)),
                                );
                                return;
                            }
                        };

                        let full_content: String =
                            parts.into_iter().map(|(_, content)| content).collect();

                        if let Err(e) = MemoryOperations::process_document_task(
                            session_id_clone.clone(),
                            full_content,
                            session,
                            memory_manager_clone.clone(),
                            session_manager_clone.clone(),
                        )
                        .await
                        {
                            error!(
                                "Document processing failed for session {}: {}",
                                session_id_clone, e
                            );
                            let _ = session_manager_clone.update_status(
                                &session_id_clone,
                                SessionStatus::Failed,
                                Some(&e.to_string()),
                            );
                        }
                    }
                    Err(e) => {
                        error!("Failed to get session for processing: {}", e);
                    }
                }
            }
        });

        // Return immediately (background task handles upload + processing)
        Ok(MemoryOperationResponse::success_with_data(
            format!(
                "File upload started: {} (session: {})",
                file_name, session_id
            ),
            json!({
                "session_id": session_id,
                "file_name": file_name,
                "total_size": total_size,
                "chunk_size": chunk_size,
                "estimated_chunks": expected_chunks,
                "process_immediately": process_immediately,
                "status": "uploading"
            }),
        ))
    }

    pub async fn process_document(
        &self,
        payload: MemoryOperationPayload,
    ) -> OperationResult<MemoryOperationResponse> {
        let session_manager = self.session_manager.clone().ok_or_else(|| {
            OperationError::Runtime("Document session manager not configured".to_string())
        })?;

        let params = ProcessDocumentParams::from_payload(&payload)?;

        info!(
            "Processing document for session {} (partial_closure={})",
            params.session_id, params.partial_closure
        );

        let session = session_manager.get_session(&params.session_id)?;

        // State check: prevent double processing
        // Exception: allow processing if session was left in "Processing" state from a crash
        if session.status == SessionStatus::Processing {
            info!(
                "Session {} was left in Processing state (possible crash), resetting and resuming",
                params.session_id
            );
            // Reset status to allow resumption
            session_manager.update_status(
                &params.session_id,
                SessionStatus::Uploading,
                Some("Resuming after crash"),
            )?;
        }

        let parts = session_manager.get_parts(&params.session_id)?;

        // Handle partial closure - allow finalizing with fewer parts than expected
        if parts.len() != session.expected_parts {
            if params.partial_closure {
                info!(
                    "Partial closure requested for session {}: processing {}/{} parts",
                    params.session_id,
                    parts.len(),
                    session.expected_parts
                );
            } else {
                return Err(OperationError::InvalidInput(format!(
                    "Cannot finalize: expected {} parts but received {}. \
                     Before calling finalize, send each chunk as a separate 'store_document_part' request with: \
                     - session_id: '{}' \
                     - part_index: 0, 1, 2, ... (sequential) \
                     - content: the text chunk. \
                     Once all {} parts are stored, call finalize to begin processing. \
                     Or set partial_closure=true to finalize with the current parts.",
                    session.expected_parts,
                    parts.len(),
                    params.session_id,
                    session.expected_parts
                )));
            }
        }

        session_manager.update_status(&params.session_id, SessionStatus::Processing, None)?;

        let full_content: String = parts.into_iter().map(|(_, content)| content).collect();

        // Spawn background task
        let session_id = params.session_id.clone();
        let memory_manager = self.memory_manager.clone();
        let session_manager_clone = session_manager.clone();

        tokio::spawn(async move {
            if let Err(e) = Self::process_document_task(
                session_id.clone(),
                full_content,
                session,
                memory_manager,
                session_manager_clone.clone(),
            )
            .await
            {
                error!(
                    "Document processing background task failed for session {}: {}",
                    session_id, e
                );
                let _ = session_manager_clone.update_status(
                    &session_id,
                    SessionStatus::Failed,
                    Some(&e.to_string()),
                );
            }
        });

        Ok(MemoryOperationResponse::success(
            "Document processing started in background",
        ))
    }

    async fn process_document_task(
        session_id: String,
        full_content: String,
        session: DocumentSession,
        memory_manager: std::sync::Arc<MemoryManager>,
        session_manager: std::sync::Arc<DocumentSessionManager>,
    ) -> crate::error::Result<()> {
        let memory_type = MemoryType::parse_with_result(&session.metadata.memory_type)
            .unwrap_or(MemoryType::Semantic);

        let mut metadata = MemoryMetadata::new(memory_type);
        metadata.user_id = session.metadata.user_id.clone();
        metadata.agent_id = session.metadata.agent_id.clone();

        if let Some(topics) = session.metadata.topics {
            metadata.topics = topics;
        }

        if let Some(context) = session.metadata.context {
            metadata.context = context;
        }

        if let Some(custom) = session.metadata.custom_metadata
            && let serde_json::Value::Object(map) = custom
        {
            for (k, v) in map {
                metadata.custom.insert(k, v);
            }
        }

        metadata.custom.insert(
            "file_path".to_string(),
            serde_json::Value::String(session.metadata.file_name.clone()),
        );

        // Chunking
        let chunk_size = memory_manager.config().document_chunk_size;
        let chunks = chunk_markdown(&full_content, chunk_size);
        let total_chunks = chunks.len();

        info!(
            "Document split into {} chunks for session {}",
            total_chunks, session_id
        );

        // Check for existing processing result to resume from
        let (start_chunk, initial_memories_created) = if let Some(existing_result) =
            &session.processing_result
        {
            info!(
                "Resuming session {} from chunk {} (previously processed {} chunks, created {} memories)",
                session_id,
                existing_result.chunks_processed,
                existing_result.chunks_processed,
                existing_result.memories_created
            );
            (
                existing_result.chunks_processed,
                existing_result.memories_created,
            )
        } else {
            info!(
                "Starting fresh processing for session {} ({} chunks)",
                session_id, total_chunks
            );
            (0, 0)
        };

        let mut created_ids = Vec::new();
        let mut previous_id: Option<String> = None;
        let mut header_stack: Vec<(usize, String, String)> = Vec::new(); // level, title, memory_id

        // Initial progress update (preserve existing progress if resuming)
        let initial_progress = ProcessingResult {
            total_chunks,
            chunks_processed: start_chunk,
            memories_created: initial_memories_created,
            summary: Some(format!("Starting processing of {} chunks...", total_chunks)),
            chunks_enriched: 0,
            chunks_enriching_end: 0,
        };
        let _ = session_manager.store_processing_result(&session_id, &initial_progress);

        info!(
            "Starting to process {} chunks for session {} (headers tracking: enabled)",
            total_chunks, session_id
        );

        // Track timing for ETA calculations
        let processing_start = std::time::Instant::now();

        // Interleaved enrich-then-store: process chunks in batches to minimize
        // waste on interruption. Each batch: enrich via LLM → store each chunk.
        // On resume, at most one batch worth of enrichment work is lost.
        let (batch_size, _) = memory_manager.llm_client().batch_config();
        let batch_size = batch_size.max(1);
        let remaining_chunks: Vec<(usize, &String)> = chunks
            .iter()
            .enumerate()
            .skip(start_chunk)
            .collect();

        for batch_slice in remaining_chunks.chunks(batch_size) {
            let batch_start_idx = batch_slice[0].0;
            let batch_end_idx = batch_start_idx + batch_slice.len();
            let batch_texts: Vec<String> = batch_slice.iter().map(|(_, t)| (*t).clone()).collect();

            // Phase 1: Enrich this batch via LLM
            let in_flight_progress = ProcessingResult {
                total_chunks,
                chunks_processed: batch_start_idx,
                memories_created: initial_memories_created + created_ids.len(),
                summary: Some(format!(
                    "Enriching metadata: batch {}-{} of {} chunks",
                    batch_start_idx + 1,
                    batch_end_idx,
                    total_chunks
                )),
                chunks_enriched: batch_start_idx,
                chunks_enriching_end: batch_end_idx,
            };
            let _ = session_manager
                .store_processing_result(&session_id, &in_flight_progress);

            let batch_enrichments: Vec<crate::memory::extractor::ChunkMetadata> = match memory_manager
                .extract_metadata_enrichment_batch(&batch_texts)
                .await
            {
                Ok(results) => results,
                Err(e) => {
                    warn!(
                        "Batch enrichment failed: {}. Using un-enriched text as fallback.",
                        e
                    );
                    batch_texts
                        .iter()
                        .map(|text| crate::memory::extractor::ChunkMetadata {
                            summary: text.trim().to_string(),
                            keywords: vec![],
                        })
                        .collect()
                }
            };

            // Phase 2: Store each chunk in this batch with its enrichment
            for (batch_offset, &(i, chunk_text)) in batch_slice.iter().enumerate() {
                let mut chunk_metadata = metadata.clone();
                chunk_metadata
                    .custom
                    .insert("chunk_index".to_string(), json!(i));
                chunk_metadata
                    .custom
                    .insert("total_chunks".to_string(), json!(total_chunks));

                // Track headers in this chunk
                let chunk_headers = crate::memory::utils::extract_headers(chunk_text);
                for (level, title) in &chunk_headers {
                    let level = *level;
                    let title = title.clone();
                    // Pop headers with level >= current
                    while header_stack.last().is_some_and(|(l, _, _)| *l >= level) {
                        header_stack.pop();
                    }

                    // Create an explicit node for the header
                    let mut header_meta = metadata.clone();
                    header_meta
                        .custom
                        .insert("is_header".to_string(), json!(true));
                    header_meta
                        .custom
                        .insert("header_level".to_string(), json!(level));

                    // Link header to its parent in the stack
                    if let Some((_, _, parent_id)) = header_stack.last() {
                        header_meta.relations.push(crate::types::Relation {
                            source: "SELF".to_string(),
                            relation: "part_of".to_string(),
                            target: parent_id.clone(),
                            strength: Some(1.0),
                        });
                    }

                    match memory_manager
                        .store_with_options(
                            format!("Header: {}", title),
                            header_meta,
                            crate::memory::manager::StoreOptions {
                                deduplicate: Some(false),
                                merge: Some(false),
                                ..Default::default()
                            },
                        )
                        .await
                    {
                        Ok(h_id) => {
                            header_stack.push((level, title, h_id));
                        }
                        Err(e) => {
                            error!("Failed to store header node {}: {}", title, e);
                        }
                    }
                }

                // Add hierarchy metadata
                if let Some((_, _, current_header_id)) = header_stack.last() {
                    // Link chunk to the current active header node
                    chunk_metadata.relations.push(crate::types::Relation {
                        source: "SELF".to_string(),
                        relation: "part_of".to_string(),
                        target: current_header_id.clone(),
                        strength: Some(1.0),
                    });
                }

                // Apply enrichment from the batch
                if let Some(enrichment) = batch_enrichments.get(batch_offset) {
                    let mut keywords = enrichment.keywords.clone();
                    // Also add headers found in this chunk to keywords
                    for (_, title) in &chunk_headers {
                        if !keywords.contains(title) {
                            keywords.push(title.clone());
                        }
                    }

                    chunk_metadata
                        .custom
                        .insert("summary".to_string(), json!(enrichment.summary));
                    chunk_metadata
                        .custom
                        .insert("keywords".to_string(), json!(keywords));
                }

                // Store verbatim
                let memory_id = memory_manager
                    .store_with_options(
                        chunk_text.clone(),
                        chunk_metadata,
                        crate::memory::manager::StoreOptions {
                            deduplicate: Some(false),
                            merge: Some(false),
                            ..Default::default()
                        },
                    )
                    .await?;
                created_ids.push(memory_id.clone());

                // Linking
                if let Some(prev) = previous_id {
                    // Link prev -> next
                    let _ = memory_manager
                        .update(
                            &prev,
                            None,
                            Some(vec![crate::types::Relation {
                                source: prev.clone(),
                                relation: "next_chunk".to_string(),
                                target: memory_id.clone(),
                                strength: Some(1.0),
                            }]),
                        )
                        .await;

                    // Link next -> prev
                    let _ = memory_manager
                        .update(
                            &memory_id,
                            None,
                            Some(vec![crate::types::Relation {
                                source: memory_id.clone(),
                                relation: "previous_chunk".to_string(),
                                target: prev,
                                strength: Some(1.0),
                            }]),
                        )
                        .await;
                }

                previous_id = Some(memory_id);

                // Update status after every chunk — chunks_processed advances per stored chunk
                let progress = ProcessingResult {
                    total_chunks,
                    chunks_processed: i + 1,
                    memories_created: initial_memories_created + created_ids.len(),
                    summary: Some(format!("Processing chunk {}/{}", i + 1, total_chunks)),
                    chunks_enriched: i + 1,
                    chunks_enriching_end: batch_end_idx,
                };
                let _ = session_manager.store_processing_result(&session_id, &progress);

                // Log progress every 50 chunks (or every 10 for small documents) with timing and ETA
                let progress_interval = if total_chunks <= 50 { 10 } else { 50 };
                if (i + 1) % progress_interval == 0 {
                    let elapsed = processing_start.elapsed();
                    let elapsed_secs = elapsed.as_secs_f64();
                    let chunks_per_sec = (i + 1) as f64 / elapsed_secs;
                    let remaining = total_chunks - (i + 1);
                    let eta_secs = remaining as f64 / chunks_per_sec;

                    // Format ETA nicely
                    let eta_formatted = if eta_secs < 60.0 {
                        format!("{:.0}s", eta_secs)
                    } else if eta_secs < 3600.0 {
                        format!("{:.1}m", eta_secs / 60.0)
                    } else {
                        format!("{:.1}h", eta_secs / 3600.0)
                    };

                    info!(
                        "Processing chunk {}/{} ({}%) - {} memories created, {} remaining | Elapsed: {:.1}s, ETA: {} ({:.1} chunks/sec)",
                        i + 1,
                        total_chunks,
                        ((i + 1) as f64 / total_chunks as f64 * 100.0).round(),
                        initial_memories_created + created_ids.len(),
                        remaining,
                        elapsed_secs,
                        eta_formatted,
                        chunks_per_sec
                    );
                }
            }
        }

        let processing_result = ProcessingResult {
            total_chunks,
            chunks_processed: total_chunks,
            memories_created: initial_memories_created + created_ids.len(),
            summary: Some(format!(
                "Document ingestion completed. Split into {} chunks.",
                total_chunks
            )),
            chunks_enriched: total_chunks,
            chunks_enriching_end: total_chunks,
        };

        session_manager.store_processing_result(&session_id, &processing_result)?;

        let total_elapsed = processing_start.elapsed();
        info!(
            "Processing completed for session {}: {} chunks processed, {} memories created in {:.1}s",
            session_id,
            total_chunks,
            created_ids.len(),
            total_elapsed.as_secs_f64()
        );

        // Log overall document queue progress
        if let Ok(all_sessions) = session_manager.list_all_sessions() {
            let total_docs = all_sessions.len();
            let completed = all_sessions.iter().filter(|s| matches!(s.status, SessionStatus::Completed)).count();
            let processing = all_sessions.iter().filter(|s| matches!(s.status, SessionStatus::Processing)).count();
            let failed = all_sessions.iter().filter(|s| matches!(s.status, SessionStatus::Failed)).count();
            let pending = all_sessions.iter().filter(|s| matches!(s.status, SessionStatus::Uploading)).count();
            info!(
                "Document queue progress: {}/{} completed, {} processing, {} pending, {} failed",
                completed, total_docs, processing, pending, failed
            );
        }

        // Step 2: Cross-document linking (Best Effort)
        info!("Starting cross-document linking for session {}", session_id);
        let _ = session_manager.update_status(
            &session_id,
            SessionStatus::Processing,
            Some("Linking related documents..."),
        );

        if let Err(e) = Self::process_cross_links(created_ids, memory_manager).await {
            warn!(
                "Cross-document linking failed for session {}: {}",
                session_id, e
            );
        }

        session_manager.update_status(&session_id, SessionStatus::Completed, None)?;

        Ok(())
    }

    async fn process_cross_links(
        new_ids: Vec<String>,
        memory_manager: std::sync::Arc<MemoryManager>,
    ) -> crate::error::Result<()> {
        info!(
            "Starting cross-document linking for {} new memories",
            new_ids.len()
        );

        for id in new_ids {
            let memory = match memory_manager.get(&id).await? {
                Some(m) => m,
                None => continue,
            };

            // Get keywords for this chunk
            let keywords = memory
                .metadata
                .custom
                .get("keywords")
                .and_then(|v| v.as_array());

            let keywords_vec: Vec<String> = if let Some(k) = keywords {
                k.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            } else {
                continue;
            };

            if keywords_vec.is_empty() {
                continue;
            }

            // For each keyword, try to find a relevant node in ANOTHER document
            for keyword in keywords_vec.iter().take(3) {
                // Limit to top 3 keywords to avoid explosion
                let mut filters = Filters::new();
                // Filter out current document if we have the file_path
                if let Some(path) = memory
                    .metadata
                    .custom
                    .get("file_path")
                    .and_then(|v| v.as_str())
                {
                    filters
                        .custom
                        .insert("exclude_file_path".to_string(), json!(path));
                }

                // Search for the keyword
                let results = memory_manager.search(keyword, &filters, 3).await?;

                for scored in results {
                    // Only link if it's a different memory and likely a different document
                    if scored.memory.id == id {
                        continue;
                    }

                    // Check if the target is a header or has high similarity
                    let is_header = scored
                        .memory
                        .metadata
                        .custom
                        .get("is_header")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);

                    if is_header || scored.score > 0.85 {
                        info!(
                            "Creating cross-link: {} --(references)--> {} (keyword: {})",
                            id, scored.memory.id, keyword
                        );

                        let _ = memory_manager
                            .update(
                                &id,
                                None,
                                Some(vec![crate::types::Relation {
                                    source: id.clone(),
                                    relation: "references".to_string(),
                                    target: scored.memory.id.clone(),
                                    strength: Some(scored.score),
                                }]),
                            )
                            .await;
                    }
                }
            }
        }

        Ok(())
    }

    pub fn status_process_document(
        &self,
        payload: MemoryOperationPayload,
    ) -> OperationResult<MemoryOperationResponse> {
        let session_manager = self.session_manager.as_ref().ok_or_else(|| {
            OperationError::Runtime("Document session manager not configured".to_string())
        })?;

        let params = StatusProcessDocumentParams::from_payload(&payload)?;

        info!("Getting status for session: {}", params.session_id);

        match session_manager.get_status(&params.session_id) {
            Ok(status) => {
                let data = serde_json::to_value(&status).map_err(OperationError::Serialization)?;
                Ok(MemoryOperationResponse::success_with_data(
                    "Session status retrieved",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to get session status: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to get session status: {}",
                    e
                )))
            }
        }
    }

    pub fn list_document_sessions(
        &self,
        _payload: MemoryOperationPayload,
    ) -> OperationResult<MemoryOperationResponse> {
        let session_manager = self.session_manager.as_ref().ok_or_else(|| {
            OperationError::Runtime("Document session manager not configured".to_string())
        })?;

        match session_manager.list_all_sessions() {
            Ok(sessions) => Ok(MemoryOperationResponse::success_with_data(
                "Retrieved document sessions",
                json!({
                    "sessions": sessions
                }),
            )),
            Err(e) => {
                error!("Failed to list sessions: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to list sessions: {}",
                    e
                )))
            }
        }
    }

    pub fn cancel_process_document(
        &self,
        payload: MemoryOperationPayload,
    ) -> OperationResult<MemoryOperationResponse> {
        let session_manager = self.session_manager.as_ref().ok_or_else(|| {
            OperationError::Runtime("Document session manager not configured".to_string())
        })?;

        let params = CancelProcessDocumentParams::from_payload(&payload)?;

        info!("Cancelling session: {}", params.session_id);

        match session_manager.cancel_session(&params.session_id) {
            Ok(()) => Ok(MemoryOperationResponse::success(format!(
                "Session {} cancelled",
                params.session_id
            ))),
            Err(e) => {
                error!("Failed to cancel session: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to cancel session: {}",
                    e
                )))
            }
        }
    }
}

// ─── MCP tool definitions ──────────────────────────────────────────────────

/// MCP tool definition
pub struct McpToolDefinition {
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub input_schema: Value,
    pub output_schema: Option<Value>,
}

/// Get all MCP tool definitions
pub fn get_mcp_tool_definitions() -> Vec<McpToolDefinition> {
    vec![
        McpToolDefinition {
            name: "system_status".into(),
            title: Some("System Status".into()),
            description: Some(
                "IMPORTANT: Call this tool first before any other tool. \
                 Returns the current status of the memory system including: \
                 backend type (local or remote), model availability and reachability, \
                 token usage statistics, model download status, and configuration details. \
                 It is preferable to use 'default' as the bank name for other tools, unless situations warrant otherwise.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "backend": {"type": "string"},
                    "state": {"type": "string"},
                    "llm_model": {"type": "string"},
                    "embedding_model": {"type": "string"},
                    "llm_available": {"type": "boolean"},
                    "embedding_available": {"type": "boolean"},
                    "total_llm_calls": {"type": "integer"},
                    "total_embedding_calls": {"type": "integer"},
                    "total_prompt_tokens": {"type": "integer"},
                    "total_completion_tokens": {"type": "integer"},
                    "details": {"type": "object"}
                }
            })),
        },
        McpToolDefinition {
            name: "add_content_memory".into(),
            title: Some("Add Content Memory (Raw/Unprocessed)".into()),
            description: Some("Add raw content to memory WITHOUT any AI transformation. The content is stored and embedded EXACTLY AS-IS, preserving all original phrases, keywords, and structure. Use this when: (1) you need EXACT PHRASE searchability - e.g., finding 'vegan chili' or '#PlankChallenge' later, (2) storing conversation logs, documents, or code snippets where original text matters, (3) you want predictable semantic search based on the actual content, not AI-extracted interpretations. For AI-processed structured facts and insights instead, use add_intuitive_memory.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The specific fact or piece of information to store. Should be concise and atomic."
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata key-value pairs (e.g., source file, page number, timestamp, author). Strongly recommended for tracing the origin of information."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional. Only needed if multiple users share the same bank. Omit for single-user setups."
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Agent ID associated with the memory"
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["conversational", "procedural", "factual", "semantic", "episodic", "personal"],
                        "description": "Type of memory",
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of topics associated with the memory"
                    },
                    "context": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of context tags associated with the memory"
                    },
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "relation": {"type": "string"},
                                "target": {"type": "string"}
                            },
                            "required": ["relation", "target"]
                        },
                        "description": "Optional list of relations to other entities or memories"
                    },
                    "bank": {
                        "type": "string",
                        "description": "Optional memory bank name. Defaults to 'default' if not specified."
                    }
                },
                "required": ["content"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string"},
                            "user_id": {"type": "string"},
                            "agent_id": {"type": "string"}
                        }
                    },
                    "error": {"type": "string"}
                }
            })),
        },
        McpToolDefinition {
            name: "add_intuitive_memory".into(),
            title: Some("Add Intuitive Memory (AI-Processed/Structured)".into()),
            description: Some("Add memories with AI-powered extraction and structuring. The LLM analyzes your content, extracts key facts, organizes them into atomic insights, and generates searchable keywords. Use this when: (1) you want STRUCTURED, REASONING-READY memories - the AI extracts key facts and relationships, (2) you need CONDENSED insights from long conversations or documents, (3) you want AUTOMATIC KEYWORD EXTRACTION for hybrid search (searches will match both semantic meaning AND extracted keywords). IMPORTANT: Original text is TRANSFORMED by AI (e.g., 'I shared my vegan chili recipe' becomes '{\"topic\": \"Recipe sharing\", \"dish\": \"vegan chili\"}'). For preserving exact original phrases instead, use add_content_memory.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "description": "The role of the speaker (e.g., 'user', 'assistant', 'system')"},
                                "content": {"type": "string", "description": "The content of the message"},
                                "name": {"type": "string", "description": "Optional name of the speaker"}
                            },
                            "required": ["role", "content"]
                        },
                        "description": "The list of messages to extract facts from."
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata key-value pairs to attach to the extracted memories."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user ID."
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Optional agent ID."
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["conversational", "procedural", "factual", "semantic", "episodic", "personal"],
                        "description": "Type of memory to assign to the extracted facts. Defaults to 'conversational'.",
                    },
                    "bank": {
                        "type": "string",
                        "description": "Optional memory bank name. Defaults to 'default' if not specified."
                    },
                    "source_memory_id": {
                        "type": "string",
                        "description": "Optional memory ID to link this intuitive memory to. Automatically creates a 'derived_from' relation, enabling navigation from structured insights back to source content. Use this when creating an intuitive memory based on a content memory created with add_content_memory."
                    }
                },
                "required": ["messages"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "memory": {"type": "string"},
                                        "event": {"type": "string"},
                                        "actor_id": {"type": "string"},
                                        "role": {"type": "string"},
                                        "previous_memory": {"type": "string"}
                                    }
                                }
                            },
                            "user_id": {"type": "string"},
                            "agent_id": {"type": "string"}
                        }
                    },
                    "error": {"type": "string"}
                }
            })),
        },
        McpToolDefinition {
            name: "begin_store_document".into(),
            title: Some("Begin Document Ingestion Session".into()),
            description: Some("Start a new session for multi-part document ingestion. Returns session_id and chunk requirements. Use this for large files to avoid payload limits.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_name": { "type": "string", "description": "Name of the file" },
                    "total_size": { "type": "integer", "description": "Total size in bytes" },
                    "mime_type": { "type": "string", "description": "Optional MIME type" },
                    "user_id": { "type": "string" },
                    "agent_id": { "type": "string" },
                    "memory_type": {
                        "type": "string",
                        "enum": ["conversational", "procedural", "factual", "semantic", "episodic", "personal"],
                        "description": "Type of memory to assign to the extracted facts. Defaults to 'semantic'.",
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of topics associated with the memory"
                    },
                    "context": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of context tags associated with the memory"
                    },
                    "bank": { "type": "string", "description": "Optional memory bank name. Defaults to 'default' if not specified." }
                },
                "required": ["file_name", "total_size"]
            }),
            output_schema: None,
        },
        McpToolDefinition {
            name: "upload_document".into(),
            title: Some("Upload Document (Auto-Chunk)".into()),
            description: Some("Upload a file with automatic server-side chunking. The file is read, split into chunks, and stored in a session. Optionally starts processing immediately. Simpler than manual chunking with begin_store_document + store_document_part.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "Absolute path to the file to upload" },
                    "file_name": { "type": "string", "description": "Optional name for the file (defaults to basename of file_path)" },
                    "mime_type": { "type": "string", "description": "Optional MIME type (defaults to text/plain)" },
                    "chunk_size": { "type": "integer", "description": "Optional chunk size in characters (defaults to document_chunk_size from config)" },
                    "process_immediately": { "type": "boolean", "description": "If true, starts processing after upload (default: true). If false, call process_document later." },
                    "memory_type": {
                        "type": "string",
                        "enum": ["conversational", "procedural", "factual", "semantic", "episodic", "personal"],
                        "description": "Type of memory. Defaults to 'semantic'.",
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of topics"
                    },
                    "context": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of context tags"
                    },
                    "user_id": { "type": "string" },
                    "agent_id": { "type": "string" },
                    "bank": { "type": "string", "description": "Optional memory bank name." }
                },
                "required": ["file_path"]
            }),
            output_schema: None,
        },
        McpToolDefinition {
            name: "store_document_part".into(),
            title: Some("Store Document Part".into()),
            description: Some("Upload a single part of a document session.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string", "description": "Session ID from begin_store_document" },
                    "part_index": { "type": "integer", "description": "0-based index of the part" },
                    "content": { "type": "string", "description": "Text content of this part" },
                    "bank": { "type": "string", "description": "Optional memory bank name." }
                },
                "required": ["session_id", "part_index", "content"]
            }),
            output_schema: None,
        },
        McpToolDefinition {
            name: "process_document".into(),
            title: Some("Process Document Session".into()),
            description: Some("Finalize a document session and start background processing (chunking, metadata enrichment, and graph indexing).".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string", "description": "Session ID to process" },
                    "bank": { "type": "string", "description": "Optional memory bank name." }
                },
                "required": ["session_id"]
            }),
            output_schema: None,
        },
        McpToolDefinition {
            name: "status_process_document".into(),
            title: Some("Get Document Processing Status".into()),
            description: Some("Check the status of a document processing session.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string", "description": "Session ID to check" },
                    "bank": { "type": "string", "description": "Optional memory bank name." }
                },
                "required": ["session_id"]
            }),
            output_schema: None,
        },
        McpToolDefinition {
            name: "list_document_sessions".into(),
            title: Some("List Document Sessions".into()),
            description: Some("Lists all document ingestion sessions and their current status (uploading, processing, completed, failed, cancelled). Use this to check progress, find failed sessions that need retrying, or see history of ingested documents.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "bank": {
                        "type": "string",
                        "description": "Optional memory bank name. Defaults to 'default' if not specified."
                    }
                },
                "required": []
            }),
            output_schema: None,
        },
        McpToolDefinition {
            name: "cancel_process_document".into(),
            title: Some("Cancel Document Session".into()),
            description: Some("Cancel an active document session and cleanup parts.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string", "description": "Session ID to cancel" },
                    "bank": { "type": "string", "description": "Optional memory bank name." }
                },
                "required": ["session_id"]
            }),
            output_schema: None,
        },
        McpToolDefinition {
            name: "update_memory".into(),
            title: Some("Update Memory".into()),
            description: Some("Update an existing memory (content and/or relations) by ID. Use this to refine knowledge or add new graph connections found later.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the memory to update"
                    },
                    "content": {
                        "type": "string",
                        "description": "New content for the memory (optional)"
                    },
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "relation": { "type": "string", "description": "The type of relationship" },
                                "target": { "type": "string", "description": "The target entity" }
                            },
                            "required": ["relation", "target"]
                        },
                        "description": "New relations to append to existing ones (optional)"
                    },
                    "bank": {
                        "type": "string",
                        "description": "Memory bank name (default: 'default')"
                    }
                },
                "required": ["memory_id"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"}
                },
                "required": ["success", "message"]
            })),
        },
        McpToolDefinition {
            name: "query_memory".into(),
            title: Some("Query Memory (Hybrid Search + Graph Traversal)".into()),
            description: Some(
                "Search memories using hybrid semantic + keyword search with optional graph traversal. \
                \n\n\
                **Standard Search (Default)**: Performs semantic similarity search and boosts scores for memories with matching keywords in metadata.keywords. \
                Use 'keyword_only': true to search ONLY by keyword matching (faster, no embedding required). \
                \n\n\
                **Graph Traversal (Optional)**: Enable graph_traversal to follow memory relations (derived_from, mentions, knows, etc.) \
                and discover related content through multi-hop reasoning. Use this for: \
                - Finding all insights derived from a conversation (provenance search) \
                - Discovering related memories via any relation type \
                - Multi-hop reasoning (e.g., 'find facts that mention X, then facts related to those') \
                - Navigating from insights back to source content \
                \n\n\
                Use the 'bank' parameter to search in a specific memory bank. Ensure system_status is called at least once.".into()
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query string for semantic search"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["conversational", "procedural", "factual", "semantic", "episodic", "personal"],
                        "description": "Type of memory to filter by"
                    },
                    "min_salience": {
                        "type": "number",
                        "description": "Minimum salience/importance score threshold (0-1)"
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topics to filter memories by"
                    },
                    "context": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Context tags for semantic scoping. The query will be matched against context embeddings to pre-filter results."
                    },
                    "keyword_only": {
                        "type": "boolean",
                        "description": "If true, search ONLY by keyword matching without semantic similarity. Useful for exact phrase matching when you know keywords were extracted. Default: false (hybrid search).",
                        "default": false
                    },
                    "user_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "created_after": {
                        "type": "string",
                        "description": "Find memories created after this ISO 8601 datetime"
                    },
                    "created_before": {
                        "type": "string",
                        "description": "Find memories created before this ISO 8601 datetime"
                    },
                    "bank": {
                        "type": "string",
                        "description": "Memory bank name to search in (default: 'default')"
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Override the similarity threshold (0.0-1.0). Lower values return more results. Default uses config value (~0.2).",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "graph_traversal": {
                        "type": "object",
                        "description": "Optional: Enable graph traversal to follow memory relations (derived_from, mentions, knows, etc.) and discover related content through multi-hop reasoning. \
                        Use cases: (1) Provenance search - find all insights derived from a conversation, \
                        (2) Context expansion - find memories related to a concept via any relation, \
                        (3) Multi-hop reasoning - find facts that mention X, then facts related to those, \
                        (4) Source navigation - find the raw content an insight came from. \
                        Default: disabled (standard semantic search only).",
                        "properties": {
                            "enabled": {
                                "type": "boolean",
                                "description": "Enable graph traversal (default: false)",
                                "default": false
                            },
                            "max_depth": {
                                "type": "integer",
                                "description": "Maximum number of hops to traverse from entry points (default: 2, max: 5). Higher values discover more distant relations but increase query time. Recommended: 2-3 for most use cases.",
                                "default": 2,
                                "minimum": 1,
                                "maximum": 5
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["outgoing", "incoming", "both"],
                                "description": "Traversal direction: 'outgoing' follows relations FROM the entry memory (e.g., find all memories this memory references), 'incoming' follows relations TO the entry memory (e.g., find all memories that reference this one), 'both' for bidirectional traversal (default, most comprehensive)",
                                "default": "both"
                            },
                            "relation_types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional filter to only follow specific relation types (e.g., [\"derived_from\", \"mentions\", \"knows\"]). If omitted, all relation types are followed. Use this to constrain traversal to specific relationship patterns."
                            },
                            "entry_point_limit": {
                                "type": "integer",
                                "description": "Maximum number of top-scoring memories from semantic search to use as graph traversal entry points (default: 5, max: 10). Higher values provide broader coverage but may increase query time.",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 10
                            },
                            "include_paths": {
                                "type": "boolean",
                                "description": "Include detailed traversal paths and graph scoring information in response (default: false). When true, each result includes 'graph_info' with entry_distance (hops from entry), path_from_entry (relation chain), relation_boost, and final_score breakdown.",
                                "default": false
                            }
                        }
                    }
                },
                "required": ["query"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "count": {"type": "number"},
                    "memories": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["success", "count", "memories"]
            })),
        },
        McpToolDefinition {
            name: "list_memories".into(),
            title: Some("List Memories".into()),
            description: Some("Retrieve memories with optional filtering. Use the 'bank' parameter to list from a specific memory bank.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of memories to return",
                        "default": 100,
                        "maximum": 1000
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["conversational", "procedural", "factual", "semantic", "episodic", "personal"]
                    },
                    "user_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "created_after": {"type": "string"},
                    "created_before": {"type": "string"},
                    "bank": {
                        "type": "string",
                        "description": "Memory bank name to list from (default: 'default')"
                    }
                }
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "count": {"type": "number"},
                    "memories": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["success", "count", "memories"]
            })),
        },
        McpToolDefinition {
            name: "get_memory".into(),
            title: Some("Get Memory by ID".into()),
            description: Some("Retrieve a specific memory by its exact ID. Use the 'bank' parameter to look in a specific memory bank.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Exact ID of the memory to retrieve"
                    },
                    "bank": {
                        "type": "string",
                        "description": "Memory bank name to look in (default: 'default')"
                    }
                },
                "required": ["memory_id"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "memory": {"type": "object"}
                },
                "required": ["success", "memory"]
            })),
        },
        McpToolDefinition {
            name: "navigate_memory".into(),
            title: Some("Navigate Memory Abstraction Hierarchy".into()),
            description: Some(
                "Traverse the layered abstraction hierarchy from a memory node in either direction. \
                 'zoom_out' returns higher-layer (more abstract) memories that were derived FROM this memory. \
                 'zoom_in' returns lower-layer (more detailed) source memories that this memory was abstracted FROM. \
                 'both' returns both directions. Use this to explore the knowledge graph built by the abstraction pipeline. \
                 The get_memory tool also includes an 'abstracted_into' field in metadata showing which higher-layer memories reference this one.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to navigate from"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["zoom_in", "zoom_out", "both"],
                        "description": "Direction to navigate: 'zoom_out' towards abstraction, 'zoom_in' towards detail, 'both' for both directions (default: 'both')"
                    },
                    "levels": {
                        "type": "integer",
                        "description": "Number of levels to traverse for zoom_in (default: 1, max: 5)",
                        "minimum": 1,
                        "maximum": 5
                    },
                    "bank": {
                        "type": "string",
                        "description": "Memory bank name (default: 'default')"
                    }
                },
                "required": ["memory_id"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "source_memory_id": {"type": "string"},
                    "source_layer": {"type": "integer"},
                    "zoom_in": {
                        "type": "array",
                        "description": "Lower-layer (more detailed) memories this was abstracted FROM",
                        "items": {"type": "object"}
                    },
                    "zoom_out": {
                        "type": "array",
                        "description": "Higher-layer (more abstract) memories that abstract FROM this one",
                        "items": {"type": "object"}
                    }
                },
                "required": ["success", "source_memory_id", "source_layer"]
            })),
        },
        McpToolDefinition {
            name: "list_memory_banks".into(),
            title: Some("List Memory Banks".into()),
            description: Some(
                "List all available memory banks. Each bank is an isolated memory store \
                 with its own database file. Returns bank names, paths, memory counts, \
                 and descriptions. Use different banks to organize memories by project, \
                 topic, or domain.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "banks_dir": {"type": "string"},
                    "banks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "path": {"type": "string"},
                                "memory_count": {"type": "integer"},
                                "description": {"type": "string"},
                                "loaded": {"type": "boolean"}
                            }
                        }
                    }
                }
            })),
        },
        McpToolDefinition {
            name: "create_memory_bank".into(),
            title: Some("Create Memory Bank".into()),
            description: Some(
                "Create a new named memory bank for organizing memories by context. \
                 Bank names may contain only alphanumeric characters, hyphens, and \
                 underscores (max 64 chars). If the bank already exists, returns its \
                 info without modification.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the memory bank (e.g., 'my-project', 'research_notes'). Only alphanumeric, hyphens, and underscores allowed."
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional human-readable description of the bank's purpose"
                    }
                },
                "required": ["name"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "bank": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "path": {"type": "string"},
                            "memory_count": {"type": "integer"},
                            "description": {"type": "string"},
                            "loaded": {"type": "boolean"}
                        }
                    }
                }
            })),
        },
        McpToolDefinition {
            name: "backup_bank".into(),
            title: Some("Backup Memory Bank".into()),
            description: Some(
                "Create a versioned backup of a memory bank. Each backup produces a \
                 timestamped .db file and a .manifest.json sidecar containing the version \
                 number, memory count, and SHA-256 checksum for integrity verification. \
                 Multiple backups of the same bank are kept side-by-side with incrementing \
                 version numbers.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the bank to back up (default: 'default')",
                        "default": "default"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination directory for the backup file. Defaults to ~/llm-mem-backups/ if omitted."
                    }
                }
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "backup_path": {"type": "string"},
                    "manifest": {
                        "type": "object",
                        "properties": {
                            "version": {"type": "integer"},
                            "created_at": {"type": "string"},
                            "memory_count": {"type": "integer"},
                            "sha256": {"type": "string"},
                            "size_bytes": {"type": "integer"}
                        }
                    }
                }
            })),
        },
        McpToolDefinition {
            name: "restore_bank".into(),
            title: Some("Restore Memory Bank".into()),
            description: Some(
                "Restore a memory bank from a backup .db file. Supports two modes: \
                 'replace' (default) overwrites the bank entirely — requires confirm: true. \
                 'merge' additively imports memories from the backup, skipping duplicates \
                 (matched by content hash) — no confirmation needed. \
                 If a .manifest.json sidecar exists, the SHA-256 checksum is verified \
                 before restoring.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the bank to restore into (default: 'default')",
                        "default": "default"
                    },
                    "source": {
                        "type": "string",
                        "description": "Absolute path to the backup .db file to restore from"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["replace", "merge"],
                        "description": "'replace' overwrites the bank (requires confirm). 'merge' additively imports non-duplicate memories.",
                        "default": "replace"
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Required for 'replace' mode. Ask the user for confirmation first."
                    }
                },
                "required": ["source"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "restored_path": {"type": "string"},
                    "imported": {"type": "integer"},
                    "skipped_duplicates": {"type": "integer"},
                    "total_after_merge": {"type": "integer"},
                    "source": {"type": "string"}
                }
            })),
        },
        McpToolDefinition {
            name: "rename_memory_bank".into(),
            title: Some("Rename Memory Bank".into()),
            description: Some(
                "Rename a memory bank, including its database file and session database. \
                 This operation is atomic — both the main database (.db) and session \
                 database (.sessions.db) are renamed together, ensuring consistency. \
                 If the rename fails at any point, the operation is rolled back. \
                 Bank names may contain only alphanumeric characters, hyphens, and \
                 underscores (max 64 chars).".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "old_name": {
                        "type": "string",
                        "description": "Current name of the bank to rename (default: 'default')"
                    },
                    "new_name": {
                        "type": "string",
                        "description": "New name for the bank. Must be unique and follow naming rules (alphanumeric, hyphens, underscores, max 64 chars)."
                    }
                },
                "required": ["old_name", "new_name"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "old_name": {"type": "string"},
                    "new_name": {"type": "string"}
                }
            })),
        },
        McpToolDefinition {
            name: "cleanup_resources".into(),
            title: Some("Cleanup Resources".into()),
            description: Some(
                "Cleanup system resources. Supports selective deletion of memory banks or full models cleanup. \
                 For bank deletion: you MUST ask the user for explicit confirmation before calling this tool. \
                 Pass their confirmation as a specific phrase in the 'confirm' field. \
                 For model cleanup: set confirm to true.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "enum": ["models", "banks"],
                        "description": "Resource type to cleanup. 'models' deletes LLM files. 'banks' deletes memory stores.",
                        "default": "models"
                    },
                    "name": {
                        "type": "string",
                        "description": "Specific bank name to delete. If omitted when target='banks', ALL banks will be deleted!"
                    },
                    "confirm": {
                        "description": "For target='models': set to true. For target='banks': MUST be the exact string 'I confirm this data will be permanently lost' — ask the user before sending this."
                    }
                },
                "required": ["confirm"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"}
                }
            })),
        },
        McpToolDefinition {
            name: "start_abstraction_pipeline".into(),
            title: Some("Start Abstraction Pipeline".into()),
            description: Some(
                "Start the background abstraction pipeline workers (L0→L1→L2→L3+). \
                 The pipeline creates progressive abstractions: L0 raw content → L1 summaries → L2 semantic links → L3 concepts. \
                 Use this when auto_enhance is disabled or you want to manually control abstraction processing. \
                 Once started, workers run continuously in the background.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"}
                }
            })),
        },
        McpToolDefinition {
            name: "stop_abstraction_pipeline".into(),
            title: Some("Stop Abstraction Pipeline".into()),
            description: Some(
                "Stop the background abstraction pipeline workers. Workers will finish current tasks and shut down gracefully. \
                 Use this to pause abstraction processing or conserve resources.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"}
                }
            })),
        },
        McpToolDefinition {
            name: "trigger_abstraction".into(),
            title: Some("Trigger Abstraction Now".into()),
            description: Some(
                "Trigger immediate one-shot abstraction processing. Unlike start_abstraction_pipeline, this runs once and doesn't start background workers. \
                 Use target_layer: 1 for L0→L1 (summaries), 2 for L1→L2 (semantic links), 3 for L2→L3 (concepts), or 0/all for all layers. \
                 Requires the pipeline to be running (call start_abstraction_pipeline first if needed).".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "target_layer": {
                        "type": "integer",
                        "description": "Target layer: 1=L0→L1, 2=L1→L2, 3=L2→L3, 0=all. Default: 1",
                        "default": 1
                    }
                },
                "required": []
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "l0_to_l1_created": {"type": "integer"},
                    "l1_to_l2_created": {"type": "integer"},
                    "l2_to_l3_created": {"type": "integer"},
                    "errors": {"type": "array", "items": {"type": "string"}}
                }
            })),
        },
    ]
}

/// Map MCP arguments to MemoryOperationPayload
pub fn map_mcp_arguments_to_payload(
    arguments: &serde_json::Map<String, Value>,
    default_agent_id: &Option<String>,
) -> MemoryOperationPayload {
    let mut payload = MemoryOperationPayload::default();

    if let Some(content) = arguments.get("content").and_then(|v| v.as_str()) {
        payload.content = Some(content.to_string());
    }
    if let Some(query) = arguments.get("query").and_then(|v| v.as_str()) {
        payload.query = Some(query.to_string());
    }
    if let Some(memory_id) = arguments.get("memory_id").and_then(|v| v.as_str()) {
        payload.memory_id = Some(memory_id.to_string());
    }
    if let Some(user_id) = arguments.get("user_id").and_then(|v| v.as_str()) {
        payload.user_id = Some(user_id.to_string());
    }
    if let Some(agent_id) = arguments.get("agent_id").and_then(|v| v.as_str()) {
        payload.agent_id = Some(agent_id.to_string());
    } else {
        payload.agent_id = default_agent_id.clone();
    }
    if let Some(memory_type) = arguments.get("memory_type").and_then(|v| v.as_str()) {
        payload.memory_type = Some(memory_type.to_string());
    }
    if let Some(topics) = arguments.get("topics").and_then(|v| v.as_array()) {
        payload.topics = Some(
            topics
                .iter()
                .filter_map(|v| v.as_str())
                .map(String::from)
                .collect(),
        );
    }
    if let Some(context) = arguments.get("context").and_then(|v| v.as_array()) {
        payload.context = Some(
            context
                .iter()
                .filter_map(|v| v.as_str())
                .map(String::from)
                .collect(),
        );
    }
    if let Some(keywords) = arguments.get("keywords").and_then(|v| v.as_array()) {
        payload.keywords = Some(
            keywords
                .iter()
                .filter_map(|v| v.as_str())
                .map(String::from)
                .collect(),
        );
    }
    if let Some(relations) = arguments.get("relations").and_then(|v| v.as_array()) {
        payload.relations = Some(
            relations
                .iter()
                .filter_map(|v| v.as_object())
                .filter_map(|obj| {
                    let relation = obj.get("relation")?.as_str()?.to_string();
                    let target = obj.get("target")?.as_str()?.to_string();
                    Some(RelationInput { relation, target })
                })
                .collect(),
        );
    }
    if let Some(limit) = arguments.get("limit").and_then(|v| v.as_u64()) {
        payload.limit = Some(limit as usize);
    }
    if let Some(k) = arguments.get("k").and_then(|v| v.as_u64()) {
        payload.k = Some(k as usize);
    }
    if let Some(min_salience) = arguments.get("min_salience").and_then(|v| v.as_f64()) {
        payload.min_salience = Some(min_salience);
    }
    if let Some(created_after) = arguments.get("created_after").and_then(|v| v.as_str()) {
        payload.created_after = Some(created_after.to_string());
    }
    if let Some(created_before) = arguments.get("created_before").and_then(|v| v.as_str()) {
        payload.created_before = Some(created_before.to_string());
    }
    if let Some(bank) = arguments.get("bank").and_then(|v| v.as_str()) {
        payload.bank = Some(bank.to_string());
    }
    if let Some(file_path) = arguments.get("file_path").and_then(|v| v.as_str()) {
        payload.file_path = Some(file_path.to_string());
    }
    if let Some(file_name) = arguments.get("file_name").and_then(|v| v.as_str()) {
        payload.file_name = Some(file_name.to_string());
    }
    if let Some(chunk_size) = arguments.get("chunk_size").and_then(|v| v.as_u64()) {
        payload.chunk_size = Some(chunk_size as usize);
    }
    if let Some(process_immediately) = arguments
        .get("process_immediately")
        .and_then(|v| v.as_bool())
    {
        payload.process_immediately = Some(process_immediately);
    }
    if let Some(partial_closure) = arguments.get("partial_closure").and_then(|v| v.as_bool()) {
        payload.partial_closure = Some(partial_closure);
    }
    if let Some(threshold) = arguments.get("similarity_threshold").and_then(|v| v.as_f64()) {
        payload.similarity_threshold = Some(threshold as f32);
    }
    if let Some(direction) = arguments.get("direction").and_then(|v| v.as_str()) {
        payload.direction = Some(direction.to_string());
    }
    if let Some(levels) = arguments.get("levels").and_then(|v| v.as_u64()) {
        payload.levels = Some(levels as usize);
    }
    if let Some(graph_traversal) = arguments.get("graph_traversal").and_then(|v| v.as_object()) {
        // Parse graph_traversal object
        let gt_input = GraphTraversalInput {
            enabled: graph_traversal.get("enabled").and_then(|v| v.as_bool()),
            max_depth: graph_traversal
                .get("max_depth")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            direction: graph_traversal
                .get("direction")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            relation_types: graph_traversal
                .get("relation_types")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .map(|s| s.to_string())
                        .collect()
                }),
            entry_point_limit: graph_traversal
                .get("entry_point_limit")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            include_paths: graph_traversal
                .get("include_paths")
                .and_then(|v| v.as_bool()),
        };
        payload.graph_traversal = Some(gt_input);
    }
    if let Some(messages) = arguments.get("messages").and_then(|v| v.as_array()) {
        let mut parsed_messages = Vec::new();
        for msg in messages {
            if let Some(obj) = msg.as_object()
                && let (Some(role), Some(content)) = (
                    obj.get("role").and_then(|v| v.as_str()),
                    obj.get("content").and_then(|v| v.as_str()),
                )
            {
                parsed_messages.push(crate::types::Message {
                    role: role.to_string(),
                    content: content.to_string(),
                    name: obj
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                });
            }
        }
        if !parsed_messages.is_empty() {
            payload.messages = Some(parsed_messages);
        }
    }

    // Metadata can be passed as an object to populate the 'custom' metadata field
    if let Some(metadata) = arguments.get("metadata").and_then(|v| v.as_object()) {
        let mut custom_metadata = HashMap::new();
        for (k, v) in metadata {
            custom_metadata.insert(k.clone(), v.clone());
        }
        payload.metadata = Some(custom_metadata);
    }

    // Document session management
    if let Some(session_id) = arguments.get("session_id").and_then(|v| v.as_str()) {
        payload.session_id = Some(session_id.to_string());
    }
    if let Some(part_index) = arguments.get("part_index").and_then(|v| v.as_u64()) {
        payload.part_index = Some(part_index as usize);
    }
    if let Some(file_name) = arguments.get("file_name").and_then(|v| v.as_str()) {
        payload.file_name = Some(file_name.to_string());
    }
    if let Some(total_size) = arguments.get("total_size").and_then(|v| v.as_u64()) {
        payload.total_size = Some(total_size as usize);
    }
    if let Some(mime_type) = arguments.get("mime_type").and_then(|v| v.as_str()) {
        payload.mime_type = Some(mime_type.to_string());
    }

    payload
}

/// Convert OperationError to MCP error code
pub fn operation_error_to_mcp_error_code(error: &OperationError) -> i32 {
    match error {
        OperationError::InvalidInput(_) => -32602,
        OperationError::Runtime(_) => -32603,
        OperationError::MemoryNotFound(_) => -32601,
        OperationError::Serialization(_) => -32603,
    }
}

/// Get error message from OperationError
pub fn get_operation_error_message(error: &OperationError) -> String {
    match error {
        OperationError::InvalidInput(msg) => msg.clone(),
        OperationError::Runtime(msg) => msg.clone(),
        OperationError::MemoryNotFound(msg) => msg.clone(),
        OperationError::Serialization(e) => format!("Serialization error: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- MemoryOperationPayload tests ---

    #[test]
    fn test_payload_default() {
        let payload = MemoryOperationPayload::default();
        assert!(payload.content.is_none());
        assert!(payload.query.is_none());
        assert!(payload.memory_id.is_none());
        assert!(payload.user_id.is_none());
        assert!(payload.limit.is_none());
    }

    #[test]
    fn test_payload_serialization() {
        let payload = MemoryOperationPayload {
            content: Some("test content".into()),
            user_id: Some("u1".into()),
            memory_type: Some("factual".into()),
            ..Default::default()
        };
        let json = serde_json::to_string(&payload).unwrap();
        let restored: MemoryOperationPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.content.as_deref(), Some("test content"));
        assert_eq!(restored.user_id.as_deref(), Some("u1"));
    }

    // --- MemoryOperationResponse tests ---

    #[test]
    fn test_response_success() {
        let r = MemoryOperationResponse::success("ok");
        assert!(r.success);
        assert_eq!(r.message, "ok");
        assert!(r.data.is_none());
        assert!(r.error.is_none());
    }

    #[test]
    fn test_response_success_with_data() {
        let data = json!({"id": "abc"});
        let r = MemoryOperationResponse::success_with_data("stored", data.clone());
        assert!(r.success);
        assert_eq!(r.message, "stored");
        assert_eq!(r.data, Some(data));
    }

    #[test]
    fn test_response_error() {
        let r = MemoryOperationResponse::error("something went wrong");
        assert!(!r.success);
        assert_eq!(r.error.as_deref(), Some("something went wrong"));
        assert_eq!(r.message, "Operation failed");
    }

    #[test]
    fn test_response_serialization() {
        let r = MemoryOperationResponse::success_with_data("ok", json!({"count": 5}));
        let json_str = serde_json::to_string(&r).unwrap();
        let restored: MemoryOperationResponse = serde_json::from_str(&json_str).unwrap();
        assert!(restored.success);
        assert_eq!(restored.data.unwrap()["count"], 5);
    }

    // --- QueryParams tests ---

    #[test]
    fn test_query_params_valid() {
        let payload = MemoryOperationPayload {
            query: Some("search term".into()),
            limit: Some(20),
            min_salience: Some(0.5),
            user_id: Some("u1".into()),
            ..Default::default()
        };
        let params = QueryParams::from_payload(&payload, 10).unwrap();
        assert_eq!(params.query, "search term");
        assert_eq!(params.limit, 20);
        assert_eq!(params.min_salience, Some(0.5));
        assert_eq!(params.user_id.as_deref(), Some("u1"));
    }

    #[test]
    fn test_query_params_missing_query() {
        let payload = MemoryOperationPayload::default();
        let result = QueryParams::from_payload(&payload, 10);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OperationError::InvalidInput(_)
        ));
    }

    #[test]
    fn test_query_params_uses_k_fallback() {
        let payload = MemoryOperationPayload {
            query: Some("test".into()),
            k: Some(5),
            ..Default::default()
        };
        let params = QueryParams::from_payload(&payload, 10).unwrap();
        assert_eq!(params.limit, 5);
    }

    #[test]
    fn test_query_params_uses_default_limit() {
        let payload = MemoryOperationPayload {
            query: Some("test".into()),
            ..Default::default()
        };
        let params = QueryParams::from_payload(&payload, 42).unwrap();
        assert_eq!(params.limit, 42);
    }

    #[test]
    fn test_query_params_date_parsing() {
        let payload = MemoryOperationPayload {
            query: Some("test".into()),
            created_after: Some("2024-01-01T00:00:00Z".into()),
            created_before: Some("2024-12-31T23:59:59Z".into()),
            ..Default::default()
        };
        let params = QueryParams::from_payload(&payload, 10).unwrap();
        assert!(params.created_after.is_some());
        assert!(params.created_before.is_some());
    }

    #[test]
    fn test_query_params_invalid_date_ignored() {
        let payload = MemoryOperationPayload {
            query: Some("test".into()),
            created_after: Some("not-a-date".into()),
            ..Default::default()
        };
        let params = QueryParams::from_payload(&payload, 10).unwrap();
        assert!(params.created_after.is_none()); // Invalid date silently ignored
    }

    // --- StoreParams tests ---

    #[test]
    fn test_store_params_valid() {
        let payload = MemoryOperationPayload {
            content: Some("memory content".into()),
            user_id: Some("user1".into()),
            topics: Some(vec!["rust".into()]),
            ..Default::default()
        };
        let params = StoreParams::from_payload(&payload, None, None).unwrap();
        assert_eq!(params.content, "memory content");
        assert_eq!(params.user_id.as_deref(), Some("user1"));
        assert_eq!(params.memory_type, "conversational"); // default
        assert_eq!(params.topics, Some(vec!["rust".into()]));
    }

    #[test]
    fn test_store_params_missing_content() {
        let payload = MemoryOperationPayload {
            user_id: Some("u1".into()),
            ..Default::default()
        };
        let result = StoreParams::from_payload(&payload, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_store_params_uses_default_user_id() {
        let payload = MemoryOperationPayload {
            content: Some("test".into()),
            ..Default::default()
        };
        let params =
            StoreParams::from_payload(&payload, Some("default_user".into()), None).unwrap();
        assert_eq!(params.user_id.as_deref(), Some("default_user"));
    }

    #[test]
    fn test_store_params_no_user_id_is_ok() {
        let payload = MemoryOperationPayload {
            content: Some("test".into()),
            ..Default::default()
        };
        let result = StoreParams::from_payload(&payload, None, None);
        assert!(result.is_ok());
        assert!(result.unwrap().user_id.is_none());
    }

    // --- AddMemoryParams tests ---

    #[test]
    fn test_add_memory_params_valid() {
        let payload = MemoryOperationPayload {
            messages: Some(vec![crate::types::Message {
                role: "user".into(),
                content: "hello".into(),
                name: None,
            }]),
            user_id: Some("user1".into()),
            ..Default::default()
        };
        let params = AddMemoryParams::from_payload(&payload, None, None).unwrap();
        assert_eq!(params.messages.len(), 1);
        assert_eq!(params.messages[0].content, "hello");
        assert_eq!(params.user_id.as_deref(), Some("user1"));
        assert_eq!(params.memory_type, "conversational"); // default
    }

    #[test]
    fn test_add_memory_params_missing_messages() {
        let payload = MemoryOperationPayload {
            user_id: Some("u1".into()),
            ..Default::default()
        };
        let result = AddMemoryParams::from_payload(&payload, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_memory_params_empty_messages() {
        let payload = MemoryOperationPayload {
            messages: Some(vec![]),
            ..Default::default()
        };
        let result = AddMemoryParams::from_payload(&payload, None, None);
        assert!(result.is_err());
    }

    // --- IngestDocumentParams tests ---

    #[test]
    fn test_ingest_document_params_valid() {
        let payload = MemoryOperationPayload {
            content: Some("document content".into()),
            user_id: Some("user1".into()),
            ..Default::default()
        };
        let params = IngestDocumentParams::from_payload(&payload, None, None).unwrap();
        assert_eq!(params.content, "document content");
        assert_eq!(params.user_id.as_deref(), Some("user1"));
        assert_eq!(params.memory_type, "semantic"); // default
    }

    #[test]
    fn test_ingest_document_params_missing_content() {
        let payload = MemoryOperationPayload {
            user_id: Some("u1".into()),
            ..Default::default()
        };
        let result = IngestDocumentParams::from_payload(&payload, None, None);
        assert!(result.is_err());
    }

    // --- FilterParams tests ---

    #[test]
    fn test_filter_params() {
        let payload = MemoryOperationPayload {
            user_id: Some("u1".into()),
            memory_type: Some("factual".into()),
            limit: Some(25),
            ..Default::default()
        };
        let params = FilterParams::from_payload(&payload, 100).unwrap();
        assert_eq!(params.user_id.as_deref(), Some("u1"));
        assert_eq!(params.memory_type.as_deref(), Some("factual"));
        assert_eq!(params.limit, 25);
    }

    #[test]
    fn test_filter_params_default_limit() {
        let payload = MemoryOperationPayload::default();
        let params = FilterParams::from_payload(&payload, 50).unwrap();
        assert_eq!(params.limit, 50);
    }

    // --- MCP tool definitions ---

    #[test]
    fn test_get_mcp_tool_definitions_count() {
        let tools = get_mcp_tool_definitions();
        // 18 + 3 pipeline control + upload_document + navigate_memory + rename_memory_bank = 24
        assert_eq!(tools.len(), 24);
    }

    #[test]
    fn test_mcp_tool_names() {
        let tools = get_mcp_tool_definitions();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"system_status"));
        assert!(names.contains(&"add_content_memory"));
        assert!(names.contains(&"add_intuitive_memory"));
        assert!(names.contains(&"begin_store_document"));
        assert!(names.contains(&"store_document_part"));
        assert!(names.contains(&"process_document"));
        assert!(names.contains(&"status_process_document"));
        assert!(names.contains(&"list_document_sessions"));
        assert!(names.contains(&"cancel_process_document"));
        assert!(names.contains(&"query_memory"));
        assert!(names.contains(&"list_memories"));
        assert!(names.contains(&"get_memory"));
        assert!(names.contains(&"navigate_memory"));
        assert!(names.contains(&"list_memory_banks"));
        assert!(names.contains(&"create_memory_bank"));
        assert!(names.contains(&"backup_bank"));
        assert!(names.contains(&"restore_bank"));
        assert!(names.contains(&"rename_memory_bank"));
        assert!(names.contains(&"cleanup_resources"));
        // New pipeline control tools
        assert!(names.contains(&"start_abstraction_pipeline"));
        assert!(names.contains(&"stop_abstraction_pipeline"));
        assert!(names.contains(&"trigger_abstraction"));
    }

    #[test]
    fn test_rename_memory_bank_tool_definition() {
        let tools = get_mcp_tool_definitions();
        let rename_tool = tools
            .iter()
            .find(|t| t.name == "rename_memory_bank")
            .unwrap();

        // Check title and description exist
        assert!(rename_tool.title.is_some());
        assert!(rename_tool.description.is_some());
        assert!(rename_tool.description.as_ref().unwrap().contains("atomic"));
        assert!(rename_tool.description.as_ref().unwrap().contains("session database"));

        // Check input schema has required fields
        let required = rename_tool.input_schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "old_name"));
        assert!(required.iter().any(|v| v == "new_name"));

        // Check properties exist
        let props = rename_tool.input_schema["properties"].as_object().unwrap();
        assert!(props.contains_key("old_name"));
        assert!(props.contains_key("new_name"));

        // Check output schema
        assert!(rename_tool.output_schema.is_some());
        let output_props = rename_tool.output_schema.as_ref().unwrap()["properties"].as_object().unwrap();
        assert!(output_props.contains_key("success"));
        assert!(output_props.contains_key("message"));
        assert!(output_props.contains_key("old_name"));
        assert!(output_props.contains_key("new_name"));
    }

    #[test]
    fn test_system_status_is_first_tool() {
        let tools = get_mcp_tool_definitions();
        assert_eq!(tools[0].name, "system_status");
    }

    #[test]
    fn test_mcp_tools_have_descriptions() {
        for tool in get_mcp_tool_definitions() {
            assert!(
                tool.description.is_some(),
                "Tool {} missing description",
                tool.name
            );
            assert!(!tool.description.as_ref().unwrap().is_empty());
        }
    }

    #[test]
    fn test_mcp_tools_store_requires_content() {
        let tools = get_mcp_tool_definitions();
        let store = tools
            .iter()
            .find(|t| t.name == "add_content_memory")
            .unwrap();
        let required = store.input_schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "content"));
    }

    #[test]
    fn test_mcp_tools_query_requires_query() {
        let tools = get_mcp_tool_definitions();
        let query = tools.iter().find(|t| t.name == "query_memory").unwrap();
        let required = query.input_schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "query"));
    }

    #[test]
    fn test_mcp_tools_get_requires_memory_id() {
        let tools = get_mcp_tool_definitions();
        let get = tools.iter().find(|t| t.name == "get_memory").unwrap();
        let required = get.input_schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "memory_id"));
    }

    // --- map_mcp_arguments_to_payload ---

    #[test]
    fn test_map_arguments_basic() {
        let mut args = serde_json::Map::new();
        args.insert("content".into(), json!("memory text"));
        args.insert("user_id".into(), json!("u1"));
        args.insert("memory_type".into(), json!("factual"));

        let payload = map_mcp_arguments_to_payload(&args, &None);
        assert_eq!(payload.content.as_deref(), Some("memory text"));
        assert_eq!(payload.user_id.as_deref(), Some("u1"));
        assert_eq!(payload.memory_type.as_deref(), Some("factual"));
    }

    #[test]
    fn test_map_arguments_with_default_agent() {
        let args = serde_json::Map::new();
        let payload = map_mcp_arguments_to_payload(&args, &Some("agent1".into()));
        assert_eq!(payload.agent_id.as_deref(), Some("agent1"));
        assert!(payload.user_id.is_none()); // user_id is not auto-generated
    }

    #[test]
    fn test_map_arguments_explicit_overrides_default() {
        let mut args = serde_json::Map::new();
        args.insert("user_id".into(), json!("explicit_user"));
        args.insert("agent_id".into(), json!("explicit_agent"));

        let payload = map_mcp_arguments_to_payload(&args, &Some("default_agent".into()));
        assert_eq!(payload.user_id.as_deref(), Some("explicit_user"));
        assert_eq!(payload.agent_id.as_deref(), Some("explicit_agent"));
    }

    #[test]
    fn test_map_arguments_topics_and_keywords() {
        let mut args = serde_json::Map::new();
        args.insert("topics".into(), json!(["rust", "ai"]));
        args.insert("keywords".into(), json!(["memory", "vector"]));

        let payload = map_mcp_arguments_to_payload(&args, &None);
        assert_eq!(
            payload.topics.as_ref().unwrap(),
            &vec!["rust".to_string(), "ai".to_string()]
        );
        assert_eq!(
            payload.keywords.as_ref().unwrap(),
            &vec!["memory".to_string(), "vector".to_string()]
        );
    }

    #[test]
    fn test_map_arguments_numeric_fields() {
        let mut args = serde_json::Map::new();
        args.insert("query".into(), json!("test"));
        args.insert("limit".into(), json!(25));
        args.insert("k".into(), json!(10));
        args.insert("min_salience".into(), json!(0.5));

        let payload = map_mcp_arguments_to_payload(&args, &None);
        assert_eq!(payload.limit, Some(25));
        assert_eq!(payload.k, Some(10));
        assert_eq!(payload.min_salience, Some(0.5));
    }

    #[test]
    fn test_map_arguments_date_fields() {
        let mut args = serde_json::Map::new();
        args.insert("created_after".into(), json!("2024-01-01T00:00:00Z"));
        args.insert("created_before".into(), json!("2024-12-31T23:59:59Z"));

        let payload = map_mcp_arguments_to_payload(&args, &None);
        assert_eq!(
            payload.created_after.as_deref(),
            Some("2024-01-01T00:00:00Z")
        );
        assert_eq!(
            payload.created_before.as_deref(),
            Some("2024-12-31T23:59:59Z")
        );
    }

    // --- error codes ---

    #[test]
    fn test_operation_error_codes() {
        assert_eq!(
            operation_error_to_mcp_error_code(&OperationError::InvalidInput("".into())),
            -32602
        );
        assert_eq!(
            operation_error_to_mcp_error_code(&OperationError::Runtime("".into())),
            -32603
        );
        assert_eq!(
            operation_error_to_mcp_error_code(&OperationError::MemoryNotFound("".into())),
            -32601
        );
    }

    #[test]
    fn test_operation_error_display() {
        let e = OperationError::InvalidInput("bad input".into());
        assert_eq!(e.to_string(), "Invalid input: bad input");

        let e = OperationError::Runtime("crash".into());
        assert_eq!(e.to_string(), "Runtime error: crash");

        let e = OperationError::MemoryNotFound("id-42".into());
        assert_eq!(e.to_string(), "Memory not found: id-42");
    }

    #[test]
    fn test_get_operation_error_message() {
        let e = OperationError::InvalidInput("x".into());
        assert_eq!(get_operation_error_message(&e), "x");

        let e = OperationError::Runtime("y".into());
        assert_eq!(get_operation_error_message(&e), "y");
    }

    #[test]
    fn test_operation_error_from_memory_error() {
        let mem_err = crate::error::MemoryError::config("bad config");
        let op_err: OperationError = mem_err.into();
        match op_err {
            OperationError::Runtime(msg) => assert!(msg.contains("bad config")),
            _ => panic!("expected Runtime"),
        }
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────────

/// Convert a Memory object to JSON
fn memory_to_json(memory: &Memory) -> Value {
    let mut metadata_obj = json!({});

    if let Some(user_id) = &memory.metadata.user_id {
        metadata_obj["user_id"] = Value::String(user_id.clone());
    }
    if let Some(agent_id) = &memory.metadata.agent_id {
        metadata_obj["agent_id"] = Value::String(agent_id.clone());
    }
    if let Some(run_id) = &memory.metadata.run_id {
        metadata_obj["run_id"] = Value::String(run_id.clone());
    }
    if let Some(actor_id) = &memory.metadata.actor_id {
        metadata_obj["actor_id"] = Value::String(actor_id.clone());
    }
    if let Some(role) = &memory.metadata.role {
        metadata_obj["role"] = Value::String(role.clone());
    }

    metadata_obj["memory_type"] = Value::String(format!("{:?}", memory.metadata.memory_type));
    metadata_obj["hash"] = Value::String(memory.metadata.hash.clone());
    metadata_obj["importance_score"] = Value::Number(
        serde_json::Number::from_f64(memory.metadata.importance_score as f64).unwrap(),
    );

    if !memory.metadata.entities.is_empty() {
        metadata_obj["entities"] = Value::Array(
            memory
                .metadata
                .entities
                .iter()
                .map(|e| Value::String(e.clone()))
                .collect(),
        );
    }
    if !memory.metadata.topics.is_empty() {
        metadata_obj["topics"] = Value::Array(
            memory
                .metadata
                .topics
                .iter()
                .map(|t| Value::String(t.clone()))
                .collect(),
        );
    }

    if !memory.metadata.context.is_empty() {
        metadata_obj["context"] = Value::Array(
            memory
                .metadata
                .context
                .iter()
                .map(|c| Value::String(c.clone()))
                .collect(),
        );
    }

    if !memory.metadata.relations.is_empty() {
        metadata_obj["relations"] =
            serde_json::to_value(&memory.metadata.relations).unwrap_or(json!([]));
    }

    if !memory.metadata.custom.is_empty() {
        metadata_obj["custom"] = Value::Object(
            memory
                .metadata
                .custom
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        );
    }

    metadata_obj["state"] = Value::String(format!("{:?}", memory.metadata.state));
    metadata_obj["layer"] = Value::Number(serde_json::Number::from(memory.metadata.layer.level));
    if let Some(layer_name) = &memory.metadata.layer.name {
        metadata_obj["layer_name"] = Value::String(layer_name.clone());
    }

    if !memory.metadata.abstraction_sources.is_empty() {
        metadata_obj["abstraction_sources"] = Value::Array(
            memory.metadata.abstraction_sources.iter().map(|s| Value::String(s.to_string())).collect(),
        );
    }

    json!({
        "id": memory.id,
        "content": memory.content,
        "created_at": memory.created_at.to_rfc3339(),
        "updated_at": memory.updated_at.to_rfc3339(),
        "metadata": metadata_obj
    })
}

#[cfg(test)]
mod tests_metadata {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_map_mcp_arguments_with_metadata() {
        let args = json!({
            "content": "Test memory content",
            "metadata": {
                "source": "technical_manual.pdf",
                "page": 42,
                "section": "Safety"
            },
            "user_id": "test_user"
        });

        let args_map = args.as_object().unwrap();
        let payload = map_mcp_arguments_to_payload(args_map, &None);

        assert_eq!(payload.content, Some("Test memory content".to_string()));
        assert_eq!(payload.user_id, Some("test_user".to_string()));

        // Verify metadata was mapped correctly
        let metadata = payload.metadata.expect("Metadata should be present");
        assert_eq!(metadata.get("source"), Some(&json!("technical_manual.pdf")));
        assert_eq!(metadata.get("page"), Some(&json!(42)));
        assert_eq!(metadata.get("section"), Some(&json!("Safety")));
    }
}

#[cfg(test)]
mod tests_graph {
    use super::*;
    use crate::types::{RelationEntry, RelationMeta};
    use serde_json::json;
    use uuid::Uuid;

    #[test]
    fn test_store_memory_with_relations() {
        let args = json!({
            "content": "Graph test content",
            "relations": [
                { "relation": "LIKES", "target": "Pizza" },
                { "relation": "LIVES_IN", "target": "New York" }
            ],
            "user_id": "test_graph_user"
        });

        let args_map = args.as_object().unwrap();
        let payload = map_mcp_arguments_to_payload(args_map, &None);

        assert!(payload.relations.is_some());
        let rels = payload.relations.unwrap();
        assert_eq!(rels.len(), 2);

        let r1 = rels.iter().find(|r| r.target == "Pizza").unwrap();
        assert_eq!(r1.relation, "LIKES");

        let r2 = rels.iter().find(|r| r.target == "New York").unwrap();
        assert_eq!(r2.relation, "LIVES_IN");
    }

    #[test]
    fn test_memory_serialization_with_relations() {
        use crate::types::{Memory, MemoryMetadata, MemoryType, Relation};

        // Create a memory with relations
        let mut metadata = MemoryMetadata::new(MemoryType::Conversational);
        metadata.relations = vec![Relation {
            source: "SELF".to_string(),
            relation: "KNOWS".to_string(),
            target: "Alice".to_string(),
            strength: None,
        }];

        let mut memory = Memory::with_content("Bob knows Alice".to_string(), vec![], metadata);
        memory.relations.insert(
            "knows".to_string(),
            RelationEntry::new(vec![Uuid::new_v4()], None, RelationMeta::new("test")),
        );

        // Serialize
        let json = memory_to_json(&memory);

        // Verify relations are present
        let relations = json["metadata"]["relations"]
            .as_array()
            .expect("Relations should be an array");
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0]["relation"], "KNOWS");
        assert_eq!(relations[0]["target"], "Alice");
    }

    #[test]
    fn test_update_payload_mapping() {
        let args = json!({
            "memory_id": "mem-update-123",
            "relations": [
                { "relation": "DISLIKES", "target": "Broccoli" }
            ],
            "content": "Updated content"
        });

        let args_map = args.as_object().unwrap();
        let payload = map_mcp_arguments_to_payload(args_map, &None);

        assert_eq!(payload.memory_id.unwrap(), "mem-update-123");
        assert!(payload.relations.is_some());
        let rels = payload.relations.unwrap();
        assert_eq!(rels[0].relation, "DISLIKES");
        assert_eq!(rels[0].target, "Broccoli");
    }
}

// --- Level 3: Context parameter tests ---

#[cfg(test)]
mod tests_context {
    use super::*;
    use crate::types::{Memory, MemoryMetadata, MemoryType};
    use serde_json::json;

    #[test]
    fn test_payload_context_mapping() {
        let args = json!({
            "content": "How to boil pasta",
            "context": ["recipe", "italian", "cooking"],
            "user_id": "test_user"
        });

        let args_map = args.as_object().unwrap();
        let payload = map_mcp_arguments_to_payload(args_map, &None);

        assert!(payload.context.is_some());
        let ctx = payload.context.unwrap();
        assert_eq!(ctx.len(), 3);
        assert_eq!(ctx[0], "recipe");
        assert_eq!(ctx[1], "italian");
        assert_eq!(ctx[2], "cooking");
    }

    #[test]
    fn test_payload_context_absent_is_none() {
        let args = json!({
            "content": "No context here",
            "user_id": "test_user"
        });

        let args_map = args.as_object().unwrap();
        let payload = map_mcp_arguments_to_payload(args_map, &None);

        assert!(payload.context.is_none());
    }

    #[test]
    fn test_store_params_extracts_context() {
        let payload = MemoryOperationPayload {
            content: Some("test content".into()),
            user_id: Some("u1".into()),
            context: Some(vec!["project-alpha".into()]),
            ..Default::default()
        };

        let params = StoreParams::from_payload(&payload, None, None).unwrap();
        assert!(params.context.is_some());
        assert_eq!(params.context.unwrap(), vec!["project-alpha"]);
    }

    #[test]
    fn test_query_params_extracts_context() {
        let payload = MemoryOperationPayload {
            query: Some("find memories".into()),
            context: Some(vec!["recipes".into()]),
            ..Default::default()
        };

        let params = QueryParams::from_payload(&payload, 10).unwrap();
        assert!(params.context.is_some());
        assert_eq!(params.context.unwrap(), vec!["recipes"]);
    }

    #[test]
    fn test_memory_to_json_includes_context() {
        let mut meta = MemoryMetadata::new(MemoryType::Factual);
        meta.context = vec!["recipe".into(), "italian".into()];

        let memory = Memory::with_content("Test context".to_string(), vec![], meta);

        let json = memory_to_json(&memory);
        let ctx = json["metadata"]["context"]
            .as_array()
            .expect("context should be an array");
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx[0], "recipe");
        assert_eq!(ctx[1], "italian");
    }

    #[test]
    fn test_memory_to_json_omits_empty_context() {
        let meta = MemoryMetadata::new(MemoryType::Factual);

        let memory = Memory::with_content("No context".to_string(), vec![], meta);

        let json = memory_to_json(&memory);
        // Empty context should not appear in JSON output (key absent → Null in serde_json)
        assert!(json["metadata"]["context"].is_null() || json["metadata"].get("context").is_none());
    }

    #[test]
    fn test_store_payload_with_context_and_relations() {
        let args = json!({
            "content": "Complex memory",
            "context": ["work", "meeting"],
            "relations": [
                { "relation": "DISCUSSED", "target": "Budget" }
            ],
            "user_id": "test_user"
        });

        let args_map = args.as_object().unwrap();
        let payload = map_mcp_arguments_to_payload(args_map, &None);

        assert!(payload.context.is_some());
        assert!(payload.relations.is_some());
        assert_eq!(payload.context.as_ref().unwrap().len(), 2);
        assert_eq!(payload.relations.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_relation_input_serialization() {
        let input = RelationInput {
            relation: "KNOWS".into(),
            target: "Alice".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let restored: RelationInput = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.relation, "KNOWS");
        assert_eq!(restored.target, "Alice");
    }
}
