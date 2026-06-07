use std::collections::HashMap;

use crate::search::TraversalConfig;

use super::requests::{
    AddMemoryRequest, BeginStoreDocumentRequest, CancelProcessDocumentRequest,
    CreateAbstractionRequest, ForceLinkRequest,
    ListRequest, ProcessDocumentRequest, QueryRequest, RelationInput, RemoveRelationRequest,
    StatusProcessDocumentRequest,
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
    pub event_after: Option<chrono::DateTime<chrono::Utc>>,
    pub event_before: Option<chrono::DateTime<chrono::Utc>>,
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

        let event_after = req
            .event_after
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));
        let event_before = req
            .event_before
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
            event_after,
            event_before,
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
    pub auto_link: Option<bool>,
    pub event_at: Option<chrono::DateTime<chrono::Utc>>,
    pub source: Option<String>,
    pub force: bool,
}

impl From<StoreRequest> for StoreParams {
    fn from(req: StoreRequest) -> Self {
        let event_at = req
            .event_at
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));
        Self {
            content: req.content,
            user_id: req.user_id,
            agent_id: req.agent_id,
            memory_type: req.memory_type,
            topics: req.topics,
            context: req.context,
            relations: req.relations,
            metadata: req.metadata,
            auto_link: req.auto_link,
            event_at,
            source: req.source,
            force: req.force,
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
    pub event_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl From<AddMemoryRequest> for AddMemoryParams {
    fn from(req: AddMemoryRequest) -> Self {
        let event_at = req
            .event_at
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));
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
            event_at,
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
    pub event_after: Option<chrono::DateTime<chrono::Utc>>,
    pub event_before: Option<chrono::DateTime<chrono::Utc>>,
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

        let event_after = req
            .event_after
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));
        let event_before = req
            .event_before
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
            event_after,
            event_before,
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
    pub event_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl From<BeginStoreDocumentRequest> for BeginStoreDocumentParams {
    fn from(req: BeginStoreDocumentRequest) -> Self {
        let event_at = req
            .event_at
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));
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
            event_at,
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
    pub event_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl From<UploadDocumentRequest> for UploadDocumentParams {
    fn from(req: UploadDocumentRequest) -> Self {
        let event_at = req
            .event_at
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));
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
            event_at,
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

// ─── User control params ────────────────────────────────────────────────────

pub struct CreateAbstractionParams {
    pub content: String,
    pub source_ids: Vec<String>,
    pub target_layer: i32,
    pub relation_type: Option<String>,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
}

impl From<CreateAbstractionRequest> for CreateAbstractionParams {
    fn from(req: CreateAbstractionRequest) -> Self {
        Self {
            content: req.content,
            source_ids: req.source_ids,
            target_layer: req.target_layer,
            relation_type: req.relation_type,
            user_id: req.user_id,
            agent_id: req.agent_id,
        }
    }
}

pub struct ForceLinkParams {
    pub source_id: String,
    pub relation: String,
    pub target_id: String,
    pub strength: Option<f32>,
}

impl From<ForceLinkRequest> for ForceLinkParams {
    fn from(req: ForceLinkRequest) -> Self {
        Self {
            source_id: req.source_id,
            relation: req.relation,
            target_id: req.target_id,
            strength: req.strength,
        }
    }
}

pub struct RemoveRelationParams {
    pub memory_id: String,
    pub relation_type: String,
    pub target_id: String,
}

impl From<RemoveRelationRequest> for RemoveRelationParams {
    fn from(req: RemoveRelationRequest) -> Self {
        Self {
            memory_id: req.memory_id,
            relation_type: req.relation_type,
            target_id: req.target_id,
        }
    }
}
