use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::search::PyramidConfig;
use crate::types::Message;

// ─── Error types ───────────────────────────────────────────────────────────

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

impl From<crate::search::GraphTraversalError> for OperationError {
    fn from(err: crate::search::GraphTraversalError) -> Self {
        OperationError::Runtime(format!("Graph traversal error: {}", err))
    }
}

pub type OperationResult<T> = Result<T, OperationError>;

pub fn operation_error_to_mcp_error_code(error: &OperationError) -> i32 {
    match error {
        OperationError::InvalidInput(_) => -32602,
        OperationError::Runtime(_) => -32603,
        OperationError::MemoryNotFound(_) => -32601,
        OperationError::Serialization(_) => -32603,
    }
}

pub fn get_operation_error_message(error: &OperationError) -> String {
    match error {
        OperationError::InvalidInput(msg) => msg.clone(),
        OperationError::Runtime(msg) => msg.clone(),
        OperationError::MemoryNotFound(msg) => msg.clone(),
        OperationError::Serialization(e) => format!("Serialization error: {}", e),
    }
}

// ─── Request types ─────────────────────────────────────────────────────────

/// Input for a relation from SELF to a target memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationInput {
    pub relation: String,
    pub target: String,
}

/// Graph traversal configuration for query requests
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphTraversalInput {
    pub enabled: Option<bool>,
    pub max_depth: Option<usize>,
    pub direction: Option<String>,
    pub relation_types: Option<Vec<String>>,
    pub entry_point_limit: Option<usize>,
    pub include_paths: Option<bool>,
}

impl GraphTraversalInput {
    pub fn to_config(&self) -> Option<crate::search::TraversalConfig> {
        if self.enabled.unwrap_or(false) {
            let direction = match self.direction.as_deref() {
                Some("outgoing") => crate::search::TraversalDirection::Outgoing,
                Some("incoming") => crate::search::TraversalDirection::Incoming,
                _ => crate::search::TraversalDirection::Both,
            };

            let mut config = crate::search::TraversalConfig::new().with_direction(direction);

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

// ─── Store / Ingest requests ────────────────────────────────────────────────

/// Request for storing a single atomic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreRequest {
    pub content: String,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    #[serde(default = "default_memory_type_store")]
    pub memory_type: String,
    pub topics: Option<Vec<String>>,
    pub context: Option<Vec<String>>,
    pub relations: Option<Vec<RelationInput>>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub bank: Option<String>,
}

fn default_memory_type_store() -> String {
    "conversational".to_string()
}

fn default_memory_type_semantic() -> String {
    "semantic".to_string()
}

/// Request for adding memory from conversation messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddMemoryRequest {
    pub messages: Vec<Message>,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    #[serde(default = "default_memory_type_store")]
    pub memory_type: String,
    pub topics: Option<Vec<String>>,
    pub context: Option<Vec<String>>,
    pub relations: Option<Vec<RelationInput>>,
    pub source_memory_id: Option<String>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub bank: Option<String>,
}

// ─── Read requests ──────────────────────────────────────────────────────────

/// Request for querying memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Legacy alias for `limit`
    #[serde(default)]
    pub k: Option<usize>,
    pub min_salience: Option<f64>,
    pub memory_type: Option<String>,
    pub topics: Option<Vec<String>>,
    pub context: Option<Vec<String>>,
    #[serde(default)]
    pub keyword_only: bool,
    #[serde(default = "default_keyword_split_ratio")]
    pub keyword_split_ratio: f32,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub created_after: Option<String>,
    pub created_before: Option<String>,
    pub graph_traversal: Option<GraphTraversalInput>,
    pub pyramid_config: Option<PyramidConfig>,
    pub similarity_threshold: Option<f32>,
    pub bank: Option<String>,
}

fn default_limit() -> usize {
    10
}

fn default_keyword_split_ratio() -> f32 {
    0.2
}

impl Default for QueryRequest {
    fn default() -> Self {
        Self {
            query: String::new(),
            limit: default_limit(),
            k: None,
            min_salience: None,
            memory_type: None,
            topics: None,
            context: None,
            keyword_only: false,
            keyword_split_ratio: default_keyword_split_ratio(),
            user_id: None,
            agent_id: None,
            created_after: None,
            created_before: None,
            graph_traversal: None,
            pyramid_config: None,
            similarity_threshold: None,
            bank: None,
        }
    }
}

/// Request for listing memories with filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListRequest {
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub memory_type: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub k: Option<usize>,
    pub created_after: Option<String>,
    pub created_before: Option<String>,
    pub relations: Option<Vec<RelationInput>>,
    pub bank: Option<String>,
}

impl Default for ListRequest {
    fn default() -> Self {
        Self {
            user_id: None,
            agent_id: None,
            memory_type: None,
            limit: default_limit(),
            k: None,
            created_after: None,
            created_before: None,
            relations: None,
            bank: None,
        }
    }
}

/// Request for getting a memory by ID
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetRequest {
    pub memory_id: String,
    pub bank: Option<String>,
}

/// Request for updating a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRequest {
    pub memory_id: String,
    pub content: Option<String>,
    pub relations: Option<Vec<RelationInput>>,
    pub bank: Option<String>,
}

/// Request for navigating the abstraction hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigateRequest {
    pub memory_id: String,
    #[serde(default = "default_direction")]
    pub direction: String,
    #[serde(default = "default_levels")]
    pub levels: usize,
    pub bank: Option<String>,
}

fn default_direction() -> String {
    "both".to_string()
}

fn default_levels() -> usize {
    1
}

// ─── Document session requests ──────────────────────────────────────────────

/// Request for beginning a document storage session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeginStoreDocumentRequest {
    pub file_name: String,
    pub file_type: Option<String>,
    pub total_size: usize,
    pub md5sum: Option<String>,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    #[serde(default = "default_memory_type_semantic")]
    pub memory_type: String,
    pub topics: Option<Vec<String>>,
    pub context: Option<Vec<String>>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub bank: Option<String>,
}

impl Default for BeginStoreDocumentRequest {
    fn default() -> Self {
        Self {
            file_name: String::new(),
            file_type: None,
            total_size: 0,
            md5sum: None,
            user_id: None,
            agent_id: None,
            memory_type: default_memory_type_semantic(),
            topics: None,
            context: None,
            metadata: None,
            bank: None,
        }
    }
}

/// Request for storing a document part
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreDocumentPartRequest {
    pub session_id: String,
    pub part_index: usize,
    pub content: String,
}

/// Request for processing a document session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessDocumentRequest {
    pub session_id: String,
    #[serde(default)]
    pub partial_closure: bool,
}

/// Request for uploading a document file
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UploadDocumentRequest {
    pub file_path: String,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
    pub memory_type: Option<String>,
    pub topics: Option<Vec<String>>,
    pub context: Option<Vec<String>>,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub chunk_size: Option<usize>,
    #[serde(default = "default_true")]
    pub process_immediately: bool,
    pub bank: Option<String>,
}

fn default_true() -> bool {
    true
}

/// Request for checking document processing status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusProcessDocumentRequest {
    pub session_id: String,
}

/// Request for listing document sessions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ListDocumentSessionsRequest {
    pub bank: Option<String>,
}

/// Request for cancelling document processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelProcessDocumentRequest {
    pub session_id: String,
}

// ─── Response type ──────────────────────────────────────────────────────────

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

#[cfg(test)]
mod tests {
    use super::*;

    // ── StoreRequest ──

    #[test]
    fn test_store_request_serialization() {
        let req = StoreRequest {
            content: "test content".into(),
            user_id: Some("u1".into()),
            agent_id: None,
            memory_type: "factual".into(),
            topics: Some(vec!["rust".into()]),
            context: None,
            relations: None,
            metadata: None,
            bank: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let restored: StoreRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.content, "test content");
        assert_eq!(restored.user_id.as_deref(), Some("u1"));
        assert_eq!(restored.memory_type, "factual");
    }

    #[test]
    fn test_store_request_default_memory_type() {
        let json = r#"{"content": "hello"}"#;
        let req: StoreRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.memory_type, "conversational");
    }

    #[test]
    fn test_store_request_missing_content_fails() {
        let json = r#"{"user_id": "u1"}"#;
        let result: Result<StoreRequest, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    // ── AddMemoryRequest ──

    #[test]
    fn test_add_memory_request_serialization() {
        let json = r#"{
            "messages": [{"role": "user", "content": "hello"}],
            "user_id": "u1",
            "memory_type": "procedural"
        }"#;
        let req: AddMemoryRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.memory_type, "procedural");
    }

    #[test]
    fn test_add_memory_request_missing_messages_fails() {
        let json = r#"{"user_id": "u1"}"#;
        let result: Result<AddMemoryRequest, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    // ── QueryRequest ──

    #[test]
    fn test_query_request_serialization() {
        let json = r#"{"query": "search term", "limit": 20, "min_salience": 0.5}"#;
        let req: QueryRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.query, "search term");
        assert_eq!(req.limit, 20);
        assert_eq!(req.min_salience, Some(0.5));
    }

    #[test]
    fn test_query_request_default_limit() {
        let json = r#"{"query": "test"}"#;
        let req: QueryRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.limit, 10);
    }

    #[test]
    fn test_query_request_missing_query_fails() {
        let json = r#"{"limit": 5}"#;
        let result: Result<QueryRequest, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_query_request_with_graph_traversal() {
        let json = r#"{
            "query": "test",
            "graph_traversal": {
                "enabled": true,
                "max_depth": 3,
                "direction": "outgoing"
            }
        }"#;
        let req: QueryRequest = serde_json::from_str(json).unwrap();
        let gt = req.graph_traversal.as_ref().unwrap();
        assert!(gt.enabled.unwrap());
        assert_eq!(gt.max_depth, Some(3));
        assert_eq!(gt.direction.as_deref(), Some("outgoing"));
    }

    // ── ListRequest ──

    #[test]
    fn test_list_request_serialization() {
        let json = r#"{"user_id": "u1", "memory_type": "factual", "limit": 50}"#;
        let req: ListRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.user_id.as_deref(), Some("u1"));
        assert_eq!(req.memory_type.as_deref(), Some("factual"));
        assert_eq!(req.limit, 50);
    }

    #[test]
    fn test_list_request_empty() {
        let json = r#"{}"#;
        let req: ListRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.limit, 10);
    }

    // ── GetRequest ──

    #[test]
    fn test_get_request_serialization() {
        let json = r#"{"memory_id": "abc-123"}"#;
        let req: GetRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.memory_id, "abc-123");
    }

    #[test]
    fn test_get_request_missing_id_fails() {
        let json = r#"{}"#;
        let result: Result<GetRequest, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    // ── UpdateRequest ──

    #[test]
    fn test_update_request_serialization() {
        let json = r#"{"memory_id": "abc", "content": "updated"}"#;
        let req: UpdateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.memory_id, "abc");
        assert_eq!(req.content.as_deref(), Some("updated"));
    }

    #[test]
    fn test_update_request_content_optional() {
        let json = r#"{"memory_id": "abc"}"#;
        let req: UpdateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.content, None);
    }

    // ── NavigateRequest ──

    #[test]
    fn test_navigate_request_defaults() {
        let json = r#"{"memory_id": "abc"}"#;
        let req: NavigateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.direction, "both");
        assert_eq!(req.levels, 1);
    }

    #[test]
    fn test_navigate_request_custom() {
        let json = r#"{"memory_id": "abc", "direction": "zoom_in", "levels": 3}"#;
        let req: NavigateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.direction, "zoom_in");
        assert_eq!(req.levels, 3);
    }

    // ── Document session requests ──

    #[test]
    fn test_begin_store_document_request() {
        let json = r#"{"file_name": "doc.txt", "total_size": 1024}"#;
        let req: BeginStoreDocumentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.file_name, "doc.txt");
        assert_eq!(req.total_size, 1024);
        assert_eq!(req.memory_type, "semantic");
    }

    #[test]
    fn test_store_document_part_request() {
        let json = r#"{"session_id": "s1", "part_index": 0, "content": "chunk"}"#;
        let req: StoreDocumentPartRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.session_id, "s1");
        assert_eq!(req.part_index, 0);
    }

    #[test]
    fn test_process_document_request() {
        let json = r#"{"session_id": "s1", "partial_closure": true}"#;
        let req: ProcessDocumentRequest = serde_json::from_str(json).unwrap();
        assert!(req.partial_closure);
    }

    #[test]
    fn test_upload_document_request() {
        let json = r#"{"file_path": "/tmp/doc.txt"}"#;
        let req: UploadDocumentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.file_path, "/tmp/doc.txt");
        assert!(req.process_immediately);
    }

    #[test]
    fn test_status_process_document_request() {
        let json = r#"{"session_id": "s1"}"#;
        let req: StatusProcessDocumentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.session_id, "s1");
    }

    #[test]
    fn test_cancel_process_document_request() {
        let json = r#"{"session_id": "s1"}"#;
        let req: CancelProcessDocumentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.session_id, "s1");
    }

    #[test]
    fn test_list_document_sessions_request_default() {
        let json = r#"{}"#;
        let req: ListDocumentSessionsRequest = serde_json::from_str(json).unwrap();
        assert!(req.bank.is_none());
    }

    // ── MemoryOperationResponse ──

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
        let data = serde_json::json!({"id": "abc"});
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
    fn test_response_serialization_roundtrip() {
        let r = MemoryOperationResponse::success_with_data("ok", serde_json::json!({"count": 5}));
        let json_str = serde_json::to_string(&r).unwrap();
        let restored: MemoryOperationResponse = serde_json::from_str(&json_str).unwrap();
        assert!(restored.success);
        assert_eq!(restored.data.unwrap()["count"], 5);
    }

    // ── RelationInput ──

    #[test]
    fn test_relation_input_serialization() {
        let json = r#"{"relation": "derived_from", "target": "mem-123"}"#;
        let rel: RelationInput = serde_json::from_str(json).unwrap();
        assert_eq!(rel.relation, "derived_from");
        assert_eq!(rel.target, "mem-123");
    }

    // ── GraphTraversalInput ──

    #[test]
    fn test_graph_traversal_to_config_enabled() {
        let gt = GraphTraversalInput {
            enabled: Some(true),
            max_depth: Some(3),
            direction: Some("outgoing".into()),
            relation_types: Some(vec!["derived_from".into()]),
            entry_point_limit: Some(5),
            include_paths: Some(true),
        };
        let config = gt.to_config().unwrap();
        assert_eq!(config.max_depth, 3);
        assert_eq!(config.entry_point_limit, 5);
    }

    #[test]
    fn test_graph_traversal_to_config_disabled() {
        let gt = GraphTraversalInput {
            enabled: Some(false),
            ..Default::default()
        };
        assert!(gt.to_config().is_none());
    }

    #[test]
    fn test_graph_traversal_to_config_default_direction() {
        let gt = GraphTraversalInput {
            enabled: Some(true),
            ..Default::default()
        };
        let config = gt.to_config().unwrap();
        assert_eq!(config.direction, crate::search::TraversalDirection::Both);
    }
}
