use arrow_array::{Array, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use futures::StreamExt;
use lancedb::connect;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::Table;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::info;
use uuid::Uuid;

use crate::config::LanceDBSettings;
use crate::error::{MemoryError, Result};
use crate::types::{DerivedEntry, Filters, Memory, MemoryMetadata, RelationEntry, ScoredMemory};

fn build_filter_expression(filters: &Filters) -> Result<Option<String>> {
    let mut expressions: Vec<String> = Vec::new();

    if let Some(min_importance) = filters.min_importance {
        expressions.push(format!("importance_score >= {}", min_importance));
    }

    if let Some(max_importance) = filters.max_importance {
        expressions.push(format!("importance_score <= {}", max_importance));
    }

    if let Some(created_after) = &filters.created_after {
        expressions.push(format!("created_at > '{}'", created_after.to_rfc3339()));
    }

    if let Some(created_before) = &filters.created_before {
        expressions.push(format!("created_at < '{}'", created_before.to_rfc3339()));
    }

    if let Some(updated_after) = &filters.updated_after {
        expressions.push(format!("updated_at > '{}'", updated_after.to_rfc3339()));
    }

    if let Some(updated_before) = &filters.updated_before {
        expressions.push(format!("updated_at < '{}'", updated_before.to_rfc3339()));
    }

    // Event time filter: rows with event_at set filter on event_at;
    // rows with event_at NULL fall back to created_at (backfill semantics).
    // Inclusive bounds: event_after is >=, event_before is <=
    if let Some(event_after) = &filters.event_after {
        let ts = event_after.to_rfc3339();
        expressions.push(format!(
            "((event_at IS NOT NULL AND event_at >= '{ts}') OR (event_at IS NULL AND created_at >= '{ts}'))"
        ));
    }
    if let Some(event_before) = &filters.event_before {
        let ts = event_before.to_rfc3339();
        expressions.push(format!(
            "((event_at IS NOT NULL AND event_at <= '{ts}') OR (event_at IS NULL AND created_at <= '{ts}'))"
        ));
    }

    if let Some(ref user_id) = filters.user_id {
        expressions.push(format!("user_id = '{}'", user_id.replace('\'', "''")));
    }

    if let Some(ref agent_id) = filters.agent_id {
        expressions.push(format!("agent_id = '{}'", agent_id.replace('\'', "''")));
    }

    if let Some(ref entities) = filters.entities {
        for entity in entities {
            expressions.push(format!(
                "metadata_json LIKE '%{}%'",
                entity.replace('\'', "''")
            ));
        }
    }

    if let Some(ref topics) = filters.topics {
        for topic in topics {
            expressions.push(format!(
                "metadata_json LIKE '%{}%'",
                topic.replace('\'', "''")
            ));
        }
    }

    if let Some(ref candidate_ids) = filters.candidate_ids {
        let id_list = candidate_ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "''")))
            .collect::<Vec<_>>()
            .join(",");
        if !id_list.is_empty() {
            expressions.push(format!("id IN ({})", id_list));
        }
    }

    if let Some(ref state) = filters.state {
        let state_str = serde_json::to_string(state)
            .unwrap_or_else(|_| format!("{:?}", state))
            .trim_matches('"')
            .to_string()
            .to_lowercase();
        expressions.push(format!("state = '{}'", state_str));
    }

    if let Some(min_layer) = filters.min_layer_level {
        expressions.push(format!("layer_level >= {}", min_layer));
    }

    if let Some(max_layer) = filters.max_layer_level {
        expressions.push(format!("layer_level <= {}", max_layer));
    }

    if let Some(ref relations) = filters.relations {
        for relation in relations {
            let relation_str = relation.relation.replace('\'', "''");
            let target_str = relation.target.replace('\'', "''");
            if !relation_str.is_empty() && !target_str.is_empty() {
                expressions.push(format!(
                    "(metadata_json LIKE '%\"relation\":\"{}\"%' AND (metadata_json LIKE '%\"target\":\"{}\"%' OR relations_json LIKE '%\"{}\"%'))",
                    relation_str, target_str, target_str
                ));
            } else if !relation_str.is_empty() {
                expressions.push(format!(
                    "metadata_json LIKE '%\"relation\":\"{}\"%'",
                    relation_str
                ));
            } else if !target_str.is_empty() {
                expressions.push(format!(
                    "(metadata_json LIKE '%\"target\":\"{}\"%' OR relations_json LIKE '%\"{}\"%')",
                    target_str, target_str
                ));
            }
        }
    }

    if let Some(ref source) = filters.contains_abstraction_source {
        // Use a broad LIKE match on the UUID within metadata_json.
        // UUIDs are unique enough that false positives are negligible.
        // This is format-agnostic (works regardless of JSON whitespace/ordering).
        let source_str = source.to_string().replace('\'', "''");
        expressions.push(format!("metadata_json LIKE '%\"{}\"%'", source_str));
    }

    Ok(if expressions.is_empty() {
        None
    } else {
        Some(expressions.join(" AND "))
    })
}

/// Shared LanceDB table schema definition.
/// Used by both table creation and record insertion to prevent drift.
fn table_schema(embedding_dimension: i32) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                embedding_dimension,
            ),
            false,
        ),
        Field::new("content", DataType::Utf8, true),
        Field::new("metadata_json", DataType::Utf8, true),
        Field::new("content_meta_json", DataType::Utf8, true),
        Field::new("derived_data_json", DataType::Utf8, true),
        Field::new("relations_json", DataType::Utf8, true),
        Field::new("context_embeddings_json", DataType::Utf8, true),
        Field::new("relation_embeddings_json", DataType::Utf8, true),
        Field::new("created_at", DataType::Utf8, false),
        Field::new("updated_at", DataType::Utf8, false),
        // Event time (caller-supplied for L0, derived range for L1+). Nullable for backfill compat.
        Field::new("event_at", DataType::Utf8, true),
        Field::new("event_end", DataType::Utf8, true),
        // Dedicated columns for efficient filtering
        Field::new("importance_score", DataType::Float32, true),
        Field::new("state", DataType::Utf8, true),
        Field::new("layer_level", DataType::Int32, true),
        Field::new("user_id", DataType::Utf8, true),
        Field::new("agent_id", DataType::Utf8, true),
    ]))
}

#[derive(Clone)]
pub struct LanceDBConfig {
    pub table_name: String,
    pub database_path: PathBuf,
    pub embedding_dimension: usize,
}

impl LanceDBConfig {
    pub fn from_settings(settings: &LanceDBSettings) -> Self {
        Self {
            table_name: settings.table_name.clone(),
            database_path: PathBuf::from(&settings.database_path),
            embedding_dimension: settings.embedding_dimension,
        }
    }
}

impl Default for LanceDBConfig {
    fn default() -> Self {
        Self {
            table_name: "memories".to_string(),
            database_path: PathBuf::from("./lancedb"),
            embedding_dimension: 384,
        }
    }
}

pub struct LanceDBStore {
    table: Arc<Table>,
    config: LanceDBConfig,
    write_count: Arc<AtomicU64>,
    write_lock: Arc<tokio::sync::Mutex<()>>,
    user_counts: Arc<DashMap<Option<String>, AtomicU64>>,
    agent_counts: Arc<DashMap<Option<String>, AtomicU64>>,
    layer_counts: Arc<DashMap<i32, AtomicU64>>,
    max_list_limit: usize,
    relation_index: Arc<tokio::sync::RwLock<RelationIndex>>,
}

type RelationIndex = Option<HashMap<String, Vec<String>>>;

impl Clone for LanceDBStore {
    fn clone(&self) -> Self {
        Self {
            table: Arc::clone(&self.table),
            config: self.config.clone(),
            write_count: Arc::clone(&self.write_count),
            write_lock: Arc::clone(&self.write_lock),
            user_counts: Arc::clone(&self.user_counts),
            agent_counts: Arc::clone(&self.agent_counts),
            layer_counts: Arc::clone(&self.layer_counts),
            max_list_limit: self.max_list_limit,
            relation_index: Arc::clone(&self.relation_index),
        }
    }
}

impl LanceDBStore {
    pub async fn new(config: LanceDBConfig) -> Result<Self> {
        let db = connect(config.database_path.to_string_lossy().as_ref())
            .execute()
            .await
            .map_err(|e| MemoryError::VectorStore(format!("LanceDB connection failed: {e}")))?;

        let table = match db.open_table(&config.table_name).execute().await {
            Ok(t) => t,
            Err(_) => {
                info!("Creating new LanceDB table: {}", config.table_name);
                let dimension = config.embedding_dimension as i32;
                let schema = table_schema(dimension);

                let empty_batch = RecordBatch::new_empty(schema);
                let batches: Vec<RecordBatch> = vec![empty_batch];

                db.create_table(&config.table_name, batches)
                    .execute()
                    .await
                    .map_err(|e| {
                        MemoryError::VectorStore(format!("LanceDB table creation failed: {e}"))
                    })?
            }
        };

        Ok(Self {
            table: Arc::new(table),
            config,
            write_count: Arc::new(AtomicU64::new(0)),
            write_lock: Arc::new(tokio::sync::Mutex::new(())),
            user_counts: Arc::new(DashMap::new()),
            agent_counts: Arc::new(DashMap::new()),
            layer_counts: Arc::new(DashMap::new()),
            max_list_limit: 10000,
            relation_index: Arc::new(tokio::sync::RwLock::new(None)),
        })
    }

    fn escape_filter_value(value: &str) -> String {
        value.replace('\'', "''")
    }

    fn batch_row_to_memory(batch: &RecordBatch, row: usize) -> Result<Memory> {
        let id_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let vector_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        let content_array = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let metadata_array = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let content_meta_array = batch
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let derived_data_array = batch
            .column(5)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let relations_array = batch
            .column(6)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let context_embeddings_array = batch
            .column(7)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let relation_embeddings_array = batch
            .column(8)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let created_at_array = batch
            .column(9)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let updated_at_array = batch
            .column(10)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let event_at_array = batch
            .column(11)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let event_end_array = batch
            .column(12)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let id = id_array.value(row).to_string();

        let list_slice = vector_array.value(row);
        let float_array = list_slice.as_any().downcast_ref::<Float32Array>().unwrap();
        let mut embedding = Vec::with_capacity(float_array.len());
        for j in 0..float_array.len() {
            embedding.push(float_array.value(j));
        }

        let content = if content_array.is_null(row) {
            None
        } else {
            Some(content_array.value(row).to_string())
        };

        let metadata = serde_json::from_str(if metadata_array.is_null(row) {
            "{}"
        } else {
            metadata_array.value(row)
        })
        .unwrap_or_else(|_| MemoryMetadata::new());

        let content_meta: crate::types::ContentMeta =
            serde_json::from_str(if content_meta_array.is_null(row) {
                "{}"
            } else {
                content_meta_array.value(row)
            })
            .unwrap_or_default();

        let derived_data: std::collections::HashMap<String, DerivedEntry> =
            serde_json::from_str(if derived_data_array.is_null(row) {
                "{}"
            } else {
                derived_data_array.value(row)
            })
            .unwrap_or_default();

        let relations: std::collections::HashMap<String, RelationEntry> =
            serde_json::from_str(if relations_array.is_null(row) {
                "{}"
            } else {
                relations_array.value(row)
            })
            .unwrap_or_default();

        let created_at = DateTime::parse_from_rfc3339(created_at_array.value(row))
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or(Utc::now());

        let updated_at = DateTime::parse_from_rfc3339(updated_at_array.value(row))
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or(Utc::now());

        let event_at = if event_at_array.is_null(row) {
            None
        } else {
            DateTime::parse_from_rfc3339(event_at_array.value(row))
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
        };
        let event_end = if event_end_array.is_null(row) {
            None
        } else {
            DateTime::parse_from_rfc3339(event_end_array.value(row))
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
        };

        let context_embeddings_json = if context_embeddings_array.is_null(row) {
            "[]"
        } else {
            context_embeddings_array.value(row)
        };
        let context_embeddings: Option<Vec<Vec<f32>>> =
            if context_embeddings_json == "[]" || context_embeddings_json.is_empty() {
                None
            } else {
                serde_json::from_str(context_embeddings_json).ok()
            };

        let relation_embeddings_json = if relation_embeddings_array.is_null(row) {
            "[]"
        } else {
            relation_embeddings_array.value(row)
        };
        let relation_embeddings: Option<Vec<Vec<f32>>> =
            if relation_embeddings_json == "[]" || relation_embeddings_json.is_empty() {
                None
            } else {
                serde_json::from_str(relation_embeddings_json).ok()
            };

        Ok(Memory {
            id,
            content,
            content_meta,
            derived_data,
            relations,
            embedding,
            metadata,
            created_at,
            updated_at,
            event_at,
            event_end,
            context_embeddings,
            relation_embeddings,
        })
    }
}

#[async_trait]
impl crate::vector_store::VectorStore for LanceDBStore {
    async fn insert(&self, memory: &Memory) -> Result<()> {
        // Increment counters
        let user_id = memory.metadata.user_id.clone();
        let agent_id = memory.metadata.agent_id.clone();
        let layer_level = memory.metadata.layer.level;

        self.user_counts
            .entry(user_id.clone())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
        self.agent_counts
            .entry(agent_id.clone())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
        self.layer_counts
            .entry(layer_level)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);

        let metadata_json = serde_json::to_string(&memory.metadata)
            .map_err(|e| MemoryError::VectorStore(format!("Metadata serialization failed: {e}")))?;
        let content_meta_json = serde_json::to_string(&memory.content_meta).map_err(|e| {
            MemoryError::VectorStore(format!("ContentMeta serialization failed: {e}"))
        })?;
        let derived_data_json = serde_json::to_string(&memory.derived_data).map_err(|e| {
            MemoryError::VectorStore(format!("DerivedData serialization failed: {e}"))
        })?;
        let relations_json = serde_json::to_string(&memory.relations).map_err(|e| {
            MemoryError::VectorStore(format!("Relations serialization failed: {e}"))
        })?;
        let context_embeddings_json =
            serde_json::to_string(&memory.context_embeddings).map_err(|e| {
                MemoryError::VectorStore(format!("Context embeddings serialization failed: {e}"))
            })?;
        let relation_embeddings_json =
            serde_json::to_string(&memory.relation_embeddings).map_err(|e| {
                MemoryError::VectorStore(format!("Relation embeddings serialization failed: {e}"))
            })?;

        let dimension = self.config.embedding_dimension as i32;

        use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};

        let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), dimension);
        for val in &memory.embedding {
            builder.values().append_value(*val);
        }
        builder.append(true);
        let list_array = builder.finish();

        // Extract values for dedicated filter columns (use serde for consistency with filtering)
        let importance_score = memory.metadata.importance_score;
        let state_str = serde_json::to_string(&memory.metadata.state)
            .unwrap_or_else(|_| format!("{:?}", memory.metadata.state))
            .trim_matches('"')
            .to_string()
            .to_lowercase();
        let layer_level = memory.metadata.layer.level;
        let user_id = memory.metadata.user_id.clone().unwrap_or_default();
        let agent_id = memory.metadata.agent_id.clone().unwrap_or_default();

        let schema = table_schema(dimension);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![memory.id.clone()])),
                Arc::new(list_array),
                Arc::new(StringArray::from_iter(vec![memory.content.as_deref()])),
                Arc::new(StringArray::from(vec![metadata_json])),
                Arc::new(StringArray::from(vec![content_meta_json])),
                Arc::new(StringArray::from(vec![derived_data_json])),
                Arc::new(StringArray::from(vec![relations_json])),
                Arc::new(StringArray::from(vec![context_embeddings_json])),
                Arc::new(StringArray::from(vec![relation_embeddings_json])),
                Arc::new(StringArray::from(vec![memory.created_at.to_rfc3339()])),
                Arc::new(StringArray::from(vec![memory.updated_at.to_rfc3339()])),
                Arc::new(StringArray::from_iter(vec![
                    memory.event_at.map(|d| d.to_rfc3339()),
                ])),
                Arc::new(StringArray::from_iter(vec![
                    memory.event_end.map(|d| d.to_rfc3339()),
                ])),
                Arc::new(Float32Array::from(vec![importance_score])),
                Arc::new(StringArray::from(vec![state_str])),
                Arc::new(Int32Array::from(vec![layer_level])),
                Arc::new(StringArray::from(vec![user_id])),
                Arc::new(StringArray::from(vec![agent_id])),
            ],
        )
        .map_err(|e| MemoryError::VectorStore(format!("RecordBatch creation failed: {e}")))?;

        let batches: Vec<RecordBatch> = vec![batch];

        let _lock = self.write_lock.lock().await;
        self.table
            .add(batches)
            .execute()
            .await
            .map_err(|e| MemoryError::VectorStore(format!("LanceDB insert failed: {e}")))?;

        self.invalidate_relation_index().await;

        let count = self.write_count.fetch_add(1, Ordering::Relaxed) + 1;
        if count.is_multiple_of(5) {
            let _ = self.compact_lancedb().await;
        }

        Ok(())
    }

    async fn search(
        &self,
        query_vector: &[f32],
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        let mut query = self
            .table
            .query()
            .nearest_to(query_vector)
            .unwrap()
            .distance_type(lancedb::DistanceType::Cosine)
            .limit(limit);

        if let Some(filter_expr) = build_filter_expression(filters)? {
            query = query.only_if(&filter_expr);
        }

        let results = query
            .execute()
            .await
            .map_err(|e| MemoryError::VectorStore(format!("LanceDB search failed: {e}")))?;

        let mut scored_memories = Vec::new();
        let mut stream = results;

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result
                .map_err(|e| MemoryError::VectorStore(format!("Failed to get batch: {e}")))?;

            // Extract distance column if present (LanceDB adds _distance for vector search)
            let distances: Option<Float32Array> = batch
                .column_by_name("_distance")
                .and_then(|col| col.as_any().downcast_ref::<Float32Array>())
                .cloned();

            for i in 0..batch.num_rows() {
                let memory = Self::batch_row_to_memory(&batch, i)?;

                // Convert cosine distance to similarity score.
                // Cosine distance range [0, 2] → score range (0.33, 1.0]
                // distance 0 (identical) → 1.0, distance 2 (opposite) → ~0.33
                let score = match &distances {
                    Some(dist) => {
                        let d = dist.value(i);
                        1.0 / (1.0 + d)
                    }
                    None => 0.5,
                };

                scored_memories.push(ScoredMemory { memory, score, semantic_score: Some(score) });
            }
        }

        Ok(scored_memories)
    }

    async fn search_with_threshold(
        &self,
        query_vector: &[f32],
        filters: &Filters,
        limit: usize,
        score_threshold: Option<f32>,
    ) -> Result<Vec<ScoredMemory>> {
        let results = self.search(query_vector, filters, limit).await?;
        if let Some(threshold) = score_threshold {
            Ok(results
                .into_iter()
                .filter(|item| item.score >= threshold)
                .collect())
        } else {
            Ok(results)
        }
    }

    async fn update(&self, memory: &Memory) -> Result<()> {
        self.delete(&memory.id).await?;
        self.insert(memory).await
    }

    async fn delete(&self, id: &str) -> Result<()> {
        // Get memory first to decrement counters
        if let Some(memory) = self.get(id).await? {
            let user_id = memory.metadata.user_id.clone();
            let agent_id = memory.metadata.agent_id.clone();
            let layer_level = memory.metadata.layer.level;

            if let Some(entry) = self.user_counts.get_mut(&user_id) {
                entry.fetch_sub(1, Ordering::Relaxed);
            }
            if let Some(entry) = self.agent_counts.get_mut(&agent_id) {
                entry.fetch_sub(1, Ordering::Relaxed);
            }
            if let Some(entry) = self.layer_counts.get_mut(&layer_level) {
                entry.fetch_sub(1, Ordering::Relaxed);
            }
        }

        let escaped_id = Self::escape_filter_value(id);
        let _lock = self.write_lock.lock().await;
        self.table
            .delete(&format!("id = '{escaped_id}'"))
            .await
            .map_err(|e| MemoryError::VectorStore(format!("LanceDB delete failed: {e}")))?;
        self.invalidate_relation_index().await;
        Ok(())
    }

    async fn get(&self, id: &str) -> Result<Option<Memory>> {
        let escaped_id = Self::escape_filter_value(id);
        let query = self.table.query().only_if(format!("id = '{escaped_id}'"));
        let mut results = query
            .execute()
            .await
            .map_err(|e| MemoryError::VectorStore(format!("LanceDB get failed: {e}")))?;

        if let Some(batch_result) = results.next().await {
            let batch = batch_result
                .map_err(|e| MemoryError::VectorStore(format!("Failed to get batch: {e}")))?;
            if batch.num_rows() == 0 {
                return Ok(None);
            }

            let memory = Self::batch_row_to_memory(&batch, 0)?;
            return Ok(Some(memory));
        }

        Ok(None)
    }

    async fn list(&self, filters: &Filters, limit: Option<usize>) -> Result<Vec<Memory>> {
        let effective_limit = limit
            .unwrap_or(self.max_list_limit)
            .min(self.max_list_limit);
        let mut query = self.table.query().limit(effective_limit);

        if let Some(filter_expr) = build_filter_expression(filters)? {
            query = query.only_if(&filter_expr);
        }

        let results = query
            .execute()
            .await
            .map_err(|e| MemoryError::VectorStore(format!("LanceDB list failed: {e}")))?;

        let mut memories = Vec::new();
        let mut stream = results;

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result
                .map_err(|e| MemoryError::VectorStore(format!("Failed to get batch: {e}")))?;
            if batch.num_rows() == 0 {
                continue;
            }

            for i in 0..batch.num_rows() {
                let memory = Self::batch_row_to_memory(&batch, i)?;
                memories.push(memory);
            }
        }

        Ok(memories)
    }

    async fn count(&self) -> Result<usize> {
        let count = self
            .table
            .count_rows(None)
            .await
            .map_err(|e| MemoryError::VectorStore(format!("LanceDB count failed: {e}")))?;
        Ok(count)
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }

    /// Compact the LanceDB table to ensure durability across process restarts.
    /// See: lance_store bug where writes only go to WAL and need compaction.
    async fn compact(&self) -> Result<()> {
        self.compact_lancedb().await
    }

    async fn find_by_relation_target(
        &self,
        target: &str,
        limit: Option<usize>,
    ) -> Result<Vec<Memory>> {
        let index = self.get_or_build_relation_index().await?;
        if let Some(source_ids) = index.get(target) {
            let escaped: Vec<String> = source_ids
                .iter()
                .take(limit.unwrap_or(source_ids.len()))
                .map(|id| format!("'{}'", id.replace('\'', "''")))
                .collect();
            if escaped.is_empty() {
                return Ok(Vec::new());
            }
            let id_list = escaped.join(",");
            let filter = format!("id IN ({})", id_list);
            let mut query = self.table.query().only_if(&filter);
            if let Some(lim) = limit {
                query = query.limit(lim);
            }
            let results = query.execute().await.map_err(|e| {
                MemoryError::VectorStore(format!("LanceDB find_by_relation_target failed: {e}"))
            })?;
            let mut memories = Vec::new();
            let mut stream = results;
            while let Some(batch_result) = stream.next().await {
                let batch = batch_result
                    .map_err(|e| MemoryError::VectorStore(format!("Failed to get batch: {e}")))?;
                if batch.num_rows() == 0 {
                    continue;
                }
                for i in 0..batch.num_rows() {
                    let memory = Self::batch_row_to_memory(&batch, i)?;
                    memories.push(memory);
                }
            }
            Ok(memories)
        } else {
            Ok(Vec::new())
        }
    }

    async fn count_by_user(&self) -> Result<Vec<(Option<String>, usize)>> {
        let mut result = Vec::new();
        for entry in self.user_counts.iter() {
            let count = entry.value().load(Ordering::Relaxed) as usize;
            if count > 0 {
                result.push((entry.key().clone(), count));
            }
        }
        Ok(result)
    }

    async fn count_by_agent(&self) -> Result<Vec<(Option<String>, usize)>> {
        let mut result = Vec::new();
        for entry in self.agent_counts.iter() {
            let count = entry.value().load(Ordering::Relaxed) as usize;
            if count > 0 {
                result.push((entry.key().clone(), count));
            }
        }
        Ok(result)
    }

    async fn count_by_layer(&self) -> Result<HashMap<i32, usize>> {
        let mut result = HashMap::new();
        for entry in self.layer_counts.iter() {
            let count = entry.value().load(Ordering::Relaxed) as usize;
            if count > 0 {
                result.insert(*entry.key(), count);
            }
        }
        Ok(result)
    }
}

impl LanceDBStore {
    async fn build_relation_index(&self) -> Result<HashMap<String, Vec<String>>> {
        use crate::vector_store::VectorStore;
        let all_memories = self.list(&Filters::default(), None).await?;
        let mut index: HashMap<String, Vec<String>> = HashMap::new();
        for memory in all_memories {
            for (_relation_type, entry) in memory.relations.iter() {
                for target_id in &entry.target_ids {
                    let target_str: String = target_id.to_string();
                    if Uuid::parse_str(&target_str).is_ok() {
                        index.entry(target_str).or_default().push(memory.id.clone());
                    }
                }
            }
            for rel in &memory.metadata.relations {
                if Uuid::parse_str(&rel.target).is_ok() {
                    index
                        .entry(rel.target.clone())
                        .or_default()
                        .push(memory.id.clone());
                }
            }
        }
        Ok(index)
    }

    async fn get_or_build_relation_index(&self) -> Result<HashMap<String, Vec<String>>> {
        let read_guard = self.relation_index.read().await;
        if let Some(ref index) = *read_guard {
            let cloned = index.clone();
            drop(read_guard);
            return Ok(cloned);
        }
        drop(read_guard);

        let mut write_guard = self.relation_index.write().await;
        if let Some(ref index) = *write_guard {
            let cloned = index.clone();
            return Ok(cloned);
        }
        let index = self.build_relation_index().await?;
        *write_guard = Some(index.clone());
        Ok(index)
    }

    async fn invalidate_relation_index(&self) {
        let mut write_guard = self.relation_index.write().await;
        *write_guard = None;
    }

    async fn compact_lancedb(&self) -> Result<()> {
        use lancedb::table::{CompactionOptions, OptimizeAction};
        let stats = self
            .table
            .optimize(OptimizeAction::Compact {
                options: CompactionOptions::default(),
                remap_options: None,
            })
            .await
            .map_err(|e| MemoryError::VectorStore(format!("LanceDB optimize failed: {e}")))?;
        tracing::debug!(
            "LanceDB compact complete: fragments_removed={:?}, fragments_added={:?}",
            stats.compaction.as_ref().map(|c| c.fragments_removed),
            stats.compaction.as_ref().map(|c| c.fragments_added),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ContentMeta;
    use crate::vector_store::VectorStore;
    use tempfile::TempDir;

    fn create_test_memory(id: &str, content: &str) -> Memory {
        Memory {
            id: id.to_string(),
            content: Some(content.to_string()),
            content_meta: ContentMeta::default(),
            derived_data: std::collections::HashMap::new(),
            relations: std::collections::HashMap::new(),
            embedding: vec![0.1; 384],
            metadata: MemoryMetadata::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            event_at: None,
            event_end: None,
            context_embeddings: None,
            relation_embeddings: None,
        }
    }

    async fn create_test_store() -> (LanceDBStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = LanceDBConfig {
            table_name: "test_memories".to_string(),
            database_path: temp_dir.path().to_path_buf(),
            embedding_dimension: 384,
        };
        let store = LanceDBStore::new(config).await.unwrap();
        (store, temp_dir)
    }

    #[tokio::test]
    async fn test_insert_and_get() {
        let (store, _temp_dir) = create_test_store().await;
        let memory = create_test_memory("test-1", "Hello, world!");

        store.insert(&memory).await.unwrap();

        let retrieved = store.get("test-1").await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, "test-1");
        assert_eq!(retrieved.content, Some("Hello, world!".to_string()));
    }

    #[tokio::test]
    async fn test_insert_and_list() {
        let (store, _temp_dir) = create_test_store().await;

        store
            .insert(&create_test_memory("test-1", "First memory"))
            .await
            .unwrap();
        store
            .insert(&create_test_memory("test-2", "Second memory"))
            .await
            .unwrap();
        store
            .insert(&create_test_memory("test-3", "Third memory"))
            .await
            .unwrap();

        let memories = store.list(&Filters::default(), None).await.unwrap();
        assert_eq!(memories.len(), 3);
    }

    #[tokio::test]
    async fn test_count() {
        let (store, _temp_dir) = create_test_store().await;

        assert_eq!(store.count().await.unwrap(), 0);

        store
            .insert(&create_test_memory("test-1", "First"))
            .await
            .unwrap();
        store
            .insert(&create_test_memory("test-2", "Second"))
            .await
            .unwrap();

        assert_eq!(store.count().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_delete() {
        let (store, _temp_dir) = create_test_store().await;

        store
            .insert(&create_test_memory("test-1", "To delete"))
            .await
            .unwrap();
        assert_eq!(store.count().await.unwrap(), 1);

        store.delete("test-1").await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);

        let retrieved = store.get("test-1").await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_update() {
        let (store, _temp_dir) = create_test_store().await;

        let mut memory = create_test_memory("test-1", "Original");
        store.insert(&memory).await.unwrap();

        memory.content = Some("Updated content".to_string());
        memory.updated_at = Utc::now();
        store.update(&memory).await.unwrap();

        let retrieved = store.get("test-1").await.unwrap().unwrap();
        assert_eq!(retrieved.content, Some("Updated content".to_string()));
    }

    #[tokio::test]
    async fn test_filter_by_importance() {
        let (store, _temp_dir) = create_test_store().await;

        let mut memory1 = create_test_memory("test-1", "Low importance");
        memory1.metadata.importance_score = 0.3;

        let mut memory2 = create_test_memory("test-2", "High importance");
        memory2.metadata.importance_score = 0.9;

        store.insert(&memory1).await.unwrap();
        store.insert(&memory2).await.unwrap();

        let filters = Filters {
            min_importance: Some(0.5),
            ..Default::default()
        };

        let results = store.list(&filters, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].metadata.importance_score, 0.9);
    }

    #[tokio::test]
    async fn test_filter_by_user_id() {
        let (store, _temp_dir) = create_test_store().await;

        let mut memory1 = create_test_memory("test-1", "User A");
        memory1.metadata.user_id = Some("user-a".to_string());

        let mut memory2 = create_test_memory("test-2", "User B");
        memory2.metadata.user_id = Some("user-b".to_string());

        store.insert(&memory1).await.unwrap();
        store.insert(&memory2).await.unwrap();

        let filters = Filters {
            user_id: Some("user-a".to_string()),
            ..Default::default()
        };

        let results = store.list(&filters, None).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_candidate_ids_filter() {
        let (store, _temp_dir) = create_test_store().await;

        store
            .insert(&create_test_memory("mem-1", "First"))
            .await
            .unwrap();
        store
            .insert(&create_test_memory("mem-2", "Second"))
            .await
            .unwrap();
        store
            .insert(&create_test_memory("mem-3", "Third"))
            .await
            .unwrap();

        let filters = Filters {
            candidate_ids: Some(vec!["mem-1".to_string(), "mem-3".to_string()]),
            ..Default::default()
        };

        let results = store.list(&filters, None).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_multi_vector_storage() {
        let (store, _temp_dir) = create_test_store().await;

        let mut memory = create_test_memory("test-1", "With multi-vectors");
        memory.context_embeddings = Some(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        memory.relation_embeddings = Some(vec![vec![0.5, 0.6]]);

        store.insert(&memory).await.unwrap();

        let retrieved = store.get("test-1").await.unwrap().unwrap();
        assert!(retrieved.context_embeddings.is_some());
        assert!(retrieved.relation_embeddings.is_some());
    }

    #[tokio::test]
    async fn test_limit_and_offset() {
        let (store, _temp_dir) = create_test_store().await;

        for i in 1..=5 {
            store
                .insert(&create_test_memory(
                    &format!("test-{}", i),
                    &format!("Memory {}", i),
                ))
                .await
                .unwrap();
        }

        let results = store.list(&Filters::default(), Some(3)).await.unwrap();
        assert_eq!(results.len(), 3);
    }
}
