use std::collections::HashMap;

use crate::search::TraversalConfig;

use super::requests::{
    AddMemoryRequest, BeginStoreDocumentRequest, CancelProcessDocumentRequest,
    IngestDocumentRequest, ListRequest, MemoryOperationPayload, OperationError, OperationResult,
    ProcessDocumentRequest, QueryRequest, RelationInput, StatusProcessDocumentRequest,
    StoreDocumentPartRequest, StoreRequest, UploadDocumentRequest,
};

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
    pub pyramid_config: crate::search::PyramidConfig,
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
            pyramid_config: payload.pyramid_config.clone().unwrap_or_default(),
        })
    }
}

impl From<QueryRequest> for QueryParams {
    fn from(req: QueryRequest) -> Self {
        let limit = req.k.unwrap_or(req.limit);
        let created_after = req
            .created_after
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        let created_before = req
            .created_before
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        let graph_traversal = req.graph_traversal.as_ref().and_then(|gt| gt.to_config());

        let include_paths = req
            .graph_traversal
            .as_ref()
            .and_then(|gt| gt.include_paths)
            .unwrap_or(false);

        Self {
            query: req.query,
            limit,
            min_salience: req.min_salience,
            memory_type: req.memory_type,
            topics: req.topics,
            context: req.context,
            keyword_only: req.keyword_only,
            user_id: req.user_id,
            agent_id: req.agent_id,
            created_after,
            created_before,
            graph_traversal,
            include_paths,
            similarity_threshold: req.similarity_threshold,
            pyramid_config: req.pyramid_config.unwrap_or_default(),
        }
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

impl From<StoreRequest> for StoreParams {
    fn from(req: StoreRequest) -> Self {
        Self {
            content: req.content,
            user_id: req.user_id,
            agent_id: req.agent_id,
            memory_type: req.memory_type,
            topics: req.topics,
            context: req.context,
            relations: req.relations,
            metadata: req.metadata,
        }
    }
}

impl From<IngestDocumentRequest> for StoreParams {
    fn from(req: IngestDocumentRequest) -> Self {
        Self {
            content: req.content,
            user_id: req.user_id,
            agent_id: req.agent_id,
            memory_type: req.memory_type,
            topics: req.topics,
            context: req.context,
            relations: req.relations,
            metadata: req.metadata,
        }
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

impl From<AddMemoryRequest> for AddMemoryParams {
    fn from(req: AddMemoryRequest) -> Self {
        Self {
            messages: req.messages,
            user_id: req.user_id,
            agent_id: req.agent_id,
            memory_type: req.memory_type,
            topics: req.topics,
            context: req.context,
            relations: req.relations,
            source_memory_id: req.source_memory_id,
            metadata: req.metadata,
        }
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

impl From<IngestDocumentRequest> for IngestDocumentParams {
    fn from(req: IngestDocumentRequest) -> Self {
        Self {
            content: req.content,
            user_id: req.user_id,
            agent_id: req.agent_id,
            memory_type: req.memory_type,
            topics: req.topics,
            context: req.context,
            relations: req.relations,
            metadata: req.metadata,
        }
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

impl From<BeginStoreDocumentRequest> for BeginStoreDocumentParams {
    fn from(req: BeginStoreDocumentRequest) -> Self {
        Self {
            file_name: req.file_name,
            file_type: req.file_type,
            total_size: req.total_size,
            md5sum: req.md5sum,
            user_id: req.user_id,
            agent_id: req.agent_id,
            memory_type: req.memory_type,
            topics: req.topics,
            context: req.context,
            metadata: req.metadata,
        }
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

impl From<StoreDocumentPartRequest> for StoreDocumentPartParams {
    fn from(req: StoreDocumentPartRequest) -> Self {
        Self {
            session_id: req.session_id,
            part_index: req.part_index,
            content: req.content,
        }
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

impl From<ProcessDocumentRequest> for ProcessDocumentParams {
    fn from(req: ProcessDocumentRequest) -> Self {
        Self {
            session_id: req.session_id,
            partial_closure: req.partial_closure,
        }
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

impl From<UploadDocumentRequest> for UploadDocumentParams {
    fn from(req: UploadDocumentRequest) -> Self {
        Self {
            file_path: req.file_path,
            file_name: req.file_name,
            mime_type: req.mime_type,
            memory_type: req.memory_type,
            topics: req.topics,
            context: req.context,
            user_id: req.user_id,
            agent_id: req.agent_id,
            chunk_size: req.chunk_size,
            process_immediately: req.process_immediately,
        }
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

impl From<StatusProcessDocumentRequest> for StatusProcessDocumentParams {
    fn from(req: StatusProcessDocumentRequest) -> Self {
        Self {
            session_id: req.session_id,
        }
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

impl From<CancelProcessDocumentRequest> for CancelProcessDocumentParams {
    fn from(req: CancelProcessDocumentRequest) -> Self {
        Self {
            session_id: req.session_id,
        }
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

impl From<ListRequest> for FilterParams {
    fn from(req: ListRequest) -> Self {
        let limit = req.k.unwrap_or(req.limit);
        let created_after = req
            .created_after
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        let created_before = req
            .created_before
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        Self {
            user_id: req.user_id,
            agent_id: req.agent_id,
            memory_type: req.memory_type,
            limit,
            created_after,
            created_before,
            relations: req.relations,
        }
    }
}
