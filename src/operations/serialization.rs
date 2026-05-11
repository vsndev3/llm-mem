use serde_json::{Value, json};

use crate::types::Memory;

pub(crate) fn memory_to_json(memory: &Memory) -> Value {
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
            memory
                .metadata
                .abstraction_sources
                .iter()
                .map(|s| Value::String(s.to_string()))
                .collect(),
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
