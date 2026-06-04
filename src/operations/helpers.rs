use std::collections::HashMap;

use crate::{
    operations::OperationError,
    types::{MemoryMetadata, Relation},
};

pub(crate) fn build_metadata(
    _memory_type_str: &str,
    user_id: Option<String>,
    agent_id: Option<String>,
    topics: Option<Vec<String>>,
    context: Option<Vec<String>>,
    raw_relations: Option<Vec<crate::operations::RelationInput>>,
    custom_metadata: Option<HashMap<String, serde_json::Value>>,
) -> Result<MemoryMetadata, OperationError> {
    let mut metadata = MemoryMetadata::new();
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

/// Parse an optional ISO 8601 string into a `DateTime<Utc>`.
/// Returns `Ok(None)` if the input is `None`, and `Err` if the string is non-empty but invalid.
#[allow(dead_code)]
pub(crate) fn parse_optional_iso8601(
    label: &str,
    value: Option<&str>,
) -> Result<Option<chrono::DateTime<chrono::Utc>>, OperationError> {
    match value {
        None => Ok(None),
        Some(s) if s.trim().is_empty() => Ok(None),
        Some(s) => chrono::DateTime::parse_from_rfc3339(s)
            .map(|dt| Some(dt.with_timezone(&chrono::Utc)))
            .map_err(|e| {
                OperationError::InvalidInput(format!(
                    "{label} must be a valid ISO 8601 datetime (got '{s}': {e})"
                ))
            }),
    }
}
