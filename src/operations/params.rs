use std::collections::HashMap;

use crate::search::TraversalConfig;

use super::requests::{
    AddMemoryRequest, BeginStoreDocumentRequest, CancelProcessDocumentRequest,
    ListRequest, ProcessDocumentRequest, QueryRequest, RelationInput, StatusProcessDocumentRequest,
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
    pub keyword_split_ratio: f32,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub created_after: Option<chrono::DateTime<chrono::Utc>>,
    pub created_before: Option<chrono::DateTime<chrono::Utc>>,
    pub graph_traversal: Option<TraversalConfig>,
    pub include_paths: bool,
    pub similarity_threshold: Option<f32>,
    pub pyramid_config: crate::search::PyramidConfig,
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
            keyword_split_ratio: req.keyword_split_ratio,
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

pub struct FilterParams {
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub memory_type: Option<String>,
    pub limit: usize,
    pub created_after: Option<chrono::DateTime<chrono::Utc>>,
    pub created_before: Option<chrono::DateTime<chrono::Utc>>,
    pub relations: Option<Vec<RelationInput>>,
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

impl From<CancelProcessDocumentRequest> for CancelProcessDocumentParams {
    fn from(req: CancelProcessDocumentRequest) -> Self {
        Self {
            session_id: req.session_id,
        }
    }
}
