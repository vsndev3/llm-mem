use std::collections::HashMap;

use crate::{
    operations::OperationError,
    types::{MemoryMetadata, MemoryType, Relation},
};

pub(crate) fn build_metadata(
    memory_type_str: &str,
    user_id: Option<String>,
    agent_id: Option<String>,
    topics: Option<Vec<String>>,
    context: Option<Vec<String>>,
    raw_relations: Option<Vec<crate::operations::RelationInput>>,
    custom_metadata: Option<HashMap<String, serde_json::Value>>,
) -> Result<MemoryMetadata, OperationError> {
    let memory_type = MemoryType::parse_with_result(memory_type_str)
        .map_err(|e| OperationError::InvalidInput(format!("Invalid memory_type: {}", e)))?;

    let mut metadata = MemoryMetadata::new(memory_type);
    metadata.user_id = user_id;
    metadata.agent_id = agent_id;

    if let Some(topics) = topics {
        metadata.topics = topics;
    }

    if let Some(context) = context {
        metadata.context = context;
    }

    if let Some(raw_relations) = raw_relations {
        metadata.relations = raw_relations
            .into_iter()
            .map(|r| Relation {
                source: "SELF".to_string(),
                relation: r.relation,
                target: r.target,
                strength: None,
            })
            .collect();
    }

    if let Some(custom_metadata) = custom_metadata {
        metadata.custom = custom_metadata;
    }

    Ok(metadata)
}
