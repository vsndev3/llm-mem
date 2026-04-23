use arrow_array::{Array, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use lancedb::connect;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::Table;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use crate::config::LanceDBSettings;
use crate::error::{MemoryError, Result};
use crate::types::{
    DerivedEntry, Filters, Memory, MemoryMetadata, MemoryType, RelationEntry, ScoredMemory,
};

fn build_filter_expression(filters: &Filters) -> Result<Option<String>> {
    let mut expressions: Vec<String> = Vec::new();

    if let Some(memory_type) = &filters.memory_type {
        let type_str = serde_json::to_string(memory_type)
            .unwrap_or_else(|_| format!("{:?}", memory_type))
            .trim_matches('"')
            .to_string();
        expressions.push(format!("memory_type = '{}'", type_str));
    }

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

    if let Some(ref user_id) = filters.user_id {
        expressions.push(format!(
            "metadata_json LIKE '%\"user_id\":\"{}\"%'",
            user_id.replace('\'', "''")
        ));
    }

    if let Some(ref agent_id) = filters.agent_id {
        expressions.push(format!(
            "metadata_json LIKE '%\"agent_id\":\"{}\"%'",
            agent_id.replace('\'', "''")
        ));
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
            expressions.push(format!(
                "metadata_json LIKE '%\"relation\":\"{}\"%'",
                relation_str
            ));
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
        // Dedicated columns for efficient filtering
        Field::new("importance_score", DataType::Float32, true),
        Field::new("memory_type", DataType::Utf8, true),
        Field::new("state", DataType::Utf8, true),
        Field::new("layer_level", DataType::Int32, true),
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

#[derive(Clone)]
pub struct LanceDBStore {
    table: Arc<Table>,
    config: LanceDBConfig,
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
        .unwrap_or_else(|_| MemoryMetadata::new(MemoryType::Conversational));

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
            context_embeddings,
            relation_embeddings,
        })
    }
}

#[async_trait]
impl crate::vector_store::VectorStore for LanceDBStore {
    async fn insert(&self, memory: &Memory) -> Result<()> {
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
        let memory_type_str = serde_json::to_string(&memory.metadata.memory_type)
            .unwrap_or_else(|_| format!("{:?}", memory.metadata.memory_type))
            .trim_matches('"')
            .to_string();
        let state_str = serde_json::to_string(&memory.metadata.state)
            .unwrap_or_else(|_| format!("{:?}", memory.metadata.state))
            .trim_matches('"')
            .to_string()
            .to_lowercase();
        let layer_level = memory.metadata.layer.level;

        let schema = table_schema(dimension);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![memory.id.clone()])),
                Arc::new(list_array),
                Arc::new(StringArray::from(vec![
                    memory.content.clone().unwrap_or_default(),
                ])),
                Arc::new(StringArray::from(vec![metadata_json])),
                Arc::new(StringArray::from(vec![content_meta_json])),
                Arc::new(StringArray::from(vec![derived_data_json])),
                Arc::new(StringArray::from(vec![relations_json])),
                Arc::new(StringArray::from(vec![context_embeddings_json])),
                Arc::new(StringArray::from(vec![relation_embeddings_json])),
                Arc::new(StringArray::from(vec![memory.created_at.to_rfc3339()])),
                Arc::new(StringArray::from(vec![memory.updated_at.to_rfc3339()])),
                Arc::new(Float32Array::from(vec![importance_score])),
                Arc::new(StringArray::from(vec![memory_type_str])),
                Arc::new(StringArray::from(vec![state_str])),
                Arc::new(Int32Array::from(vec![layer_level])),
            ],
        )
        .map_err(|e| MemoryError::VectorStore(format!("RecordBatch creation failed: {e}")))?;

        let batches: Vec<RecordBatch> = vec![batch];
        self.table
            .add(batches)
            .execute()
            .await
            .map_err(|e| MemoryError::VectorStore(format!("LanceDB insert failed: {e}")))?;

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

                // Convert distance to similarity score: 1 / (1 + distance)
                // LanceDB returns L2/cosine distances where lower = more similar.
                // This formula maps distance 0 → score 1.0, large distance → score ~0.0
                let score = match &distances {
                    Some(dist) => {
                        let d = dist.value(i);
                        1.0 / (1.0 + d)
                    }
                    None => 0.5,
                };

                scored_memories.push(ScoredMemory { memory, score });
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
        let escaped_id = Self::escape_filter_value(id);
        self.table
            .delete(&format!("id = '{escaped_id}'"))
            .await
            .map_err(|e| MemoryError::VectorStore(format!("LanceDB delete failed: {e}")))?;
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
        let mut query = self.table.query();
        if let Some(lim) = limit {
            query = query.limit(lim);
        }

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
            metadata: MemoryMetadata::new(MemoryType::Conversational),
            created_at: Utc::now(),
            updated_at: Utc::now(),
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
    async fn test_filter_by_memory_type() {
        let (store, _temp_dir) = create_test_store().await;

        let mut memory1 = create_test_memory("test-1", "Conversational");
        memory1.metadata.memory_type = MemoryType::Conversational;

        let mut memory2 = create_test_memory("test-2", "Factual");
        memory2.metadata.memory_type = MemoryType::Factual;

        store.insert(&memory1).await.unwrap();
        store.insert(&memory2).await.unwrap();

        let filters = Filters {
            memory_type: Some(MemoryType::Conversational),
            ..Default::default()
        };

        let results = store.list(&filters, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].metadata.memory_type, MemoryType::Conversational);
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
