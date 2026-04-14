use async_trait::async_trait;

use crate::{
    error::Result,
    types::{Filters, Memory, ScoredMemory},
};

/// Trait for vector store operations
#[async_trait]
pub trait VectorStore: Send + Sync + dyn_clone::DynClone {
    async fn insert(&self, memory: &Memory) -> Result<()>;
    async fn search(
        &self,
        query_vector: &[f32],
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>>;
    async fn search_with_threshold(
        &self,
        query_vector: &[f32],
        filters: &Filters,
        limit: usize,
        score_threshold: Option<f32>,
    ) -> Result<Vec<ScoredMemory>>;
    async fn update(&self, memory: &Memory) -> Result<()>;
    async fn delete(&self, id: &str) -> Result<()>;
    async fn get(&self, id: &str) -> Result<Option<Memory>>;
    async fn list(&self, filters: &Filters, limit: Option<usize>) -> Result<Vec<Memory>>;
    /// Return the number of memories (not vectors) in the store.
    async fn count(&self) -> Result<usize>;
    async fn health_check(&self) -> Result<bool>;
}

// --- VectorLite implementation (legacy, feature-gated) ---
#[cfg(feature = "vector-lite")]
mod vectorlite_impl {
    use super::*;
    use std::{
        collections::HashMap,
        path::PathBuf,
        sync::{
            Arc, RwLock,
            atomic::{AtomicU64, Ordering},
        },
    };
    use tracing::info;
    use vectorlite::{
        Collection, IndexType, SimilarityMetric,
        VectorIndexWrapper,
    };

    use crate::{
        config::{VectorLiteSettings, VectorStoreConfig},
        error::MemoryError,
    };

    #[derive(Debug, Clone)]
    pub struct VectorLiteConfig {
        pub collection_name: String,
        pub index_type: IndexType,
        pub metric: SimilarityMetric,
        pub persistence_path: Option<PathBuf>,
    }

    impl Default for VectorLiteConfig {
        fn default() -> Self {
            Self {
                collection_name: "llm-memories".to_string(),
                index_type: IndexType::HNSW,
                metric: SimilarityMetric::Cosine,
                persistence_path: None,
            }
        }
    }

    impl VectorLiteConfig {
        pub fn from_store_config(store_cfg: &VectorStoreConfig) -> Self {
            Self {
                collection_name: store_cfg.collection_name.clone(),
                index_type: parse_index_type(&store_cfg.vectorlite),
                metric: parse_metric(&store_cfg.vectorlite),
                persistence_path: store_cfg
                    .vectorlite
                    .persistence_path
                    .as_ref()
                    .map(PathBuf::from),
            }
        }
    }

    #[derive(Clone)]
    pub struct VectorLiteStore {
        index: Arc<RwLock<Option<VectorIndexWrapper>>>,
        config: VectorLiteConfig,
        next_id: Arc<AtomicU64>,
        memory_index: Arc<RwLock<HashMap<String, Memory>>>,
        _id_map: Arc<RwLock<HashMap<String, Vec<u64>>>>,
        _reverse_id_map: Arc<RwLock<HashMap<u64, String>>>,
        _abstraction_index: Arc<RwLock<HashMap<uuid::Uuid, std::collections::HashSet<String>>>>,
    }

    impl VectorLiteStore {
        pub fn with_config(config: VectorLiteConfig) -> crate::error::Result<Self> {
            let mut store = Self {
                index: Arc::new(RwLock::new(None)),
                config,
                next_id: Arc::new(AtomicU64::new(0)),
                memory_index: Arc::new(RwLock::new(HashMap::new())),
                _id_map: Arc::new(RwLock::new(HashMap::new())),
                _abstraction_index: Arc::new(RwLock::new(HashMap::new())),
                _reverse_id_map: Arc::new(RwLock::new(HashMap::new())),
            };

            store.try_load_persisted()?;
            info!("Initialized VectorLite vector store");
            Ok(store)
        }

        fn try_load_persisted(&mut self) -> crate::error::Result<()> {
            let Some(path) = &self.config.persistence_path else {
                info!("No persistence path configured, skipping load");
                return Ok(());
            };

            if !path.exists() {
                info!("Persistence file does not exist: {}", path.display());
                return Ok(());
            }

            info!("Loading persisted collection from: {}", path.display());
            let collection = Collection::load_from_file(path)
                .map_err(|e| MemoryError::VectorStore(format!("Failed to load collection: {e}")))?;

            let next_id = collection.next_id();
            info!("Collection loaded: next_id={}", next_id);

            let index_snapshot = collection
                .index_read()
                .map_err(MemoryError::VectorStore)?
                .clone();

            *self
                .index
                .write()
                .map_err(|e| MemoryError::VectorStore(e.to_string()))? = Some(index_snapshot);
            self.next_id.store(next_id, Ordering::Relaxed);

            info!("Loaded {} vectors", next_id);
            Ok(())
        }
    }

    fn parse_index_type(settings: &VectorLiteSettings) -> IndexType {
        match settings.index_type.to_lowercase().as_str() {
            "flat" => IndexType::Flat,
            _ => IndexType::HNSW,
        }
    }

    fn parse_metric(settings: &VectorLiteSettings) -> SimilarityMetric {
        match settings.metric.to_lowercase().as_str() {
            "euclidean" => SimilarityMetric::Euclidean,
            "manhattan" => SimilarityMetric::Manhattan,
            "dotproduct" => SimilarityMetric::DotProduct,
            _ => SimilarityMetric::Cosine,
        }
    }

    #[async_trait]
    impl VectorStore for VectorLiteStore {
        async fn insert(&self, memory: &Memory) -> Result<()> {
            // VectorLiteStore has its own insert method, delegate to it
            // For now, use a simple implementation that adds to in-memory index
            let mut memory_index = self.memory_index.write().map_err(|e| {
                MemoryError::VectorStore(format!("Failed to acquire write lock: {e}"))
            })?;
            memory_index.insert(memory.id.clone(), memory.clone());
            Ok(())
        }

        async fn search(
            &self,
            query_vector: &[f32],
            filters: &Filters,
            limit: usize,
        ) -> Result<Vec<ScoredMemory>> {
            // Simple in-memory search
            let memory_index = self.memory_index.read().map_err(|e| {
                MemoryError::VectorStore(format!("Failed to acquire read lock: {e}"))
            })?;
            
            let mut results: Vec<ScoredMemory> = memory_index
                .values()
                .filter(|m| matches_filters(m, filters))
                .map(|m| {
                    let score = cosine_similarity(query_vector, &m.embedding);
                    ScoredMemory {
                        memory: m.clone(),
                        score,
                    }
                })
                .collect();
            
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(limit);
            Ok(results)
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
                Ok(results.into_iter().filter(|r| r.score >= threshold).collect())
            } else {
                Ok(results)
            }
        }

        async fn update(&self, memory: &Memory) -> Result<()> {
            let mut memory_index = self.memory_index.write().map_err(|e| {
                MemoryError::VectorStore(format!("Failed to acquire write lock: {e}"))
            })?;
            memory_index.insert(memory.id.clone(), memory.clone());
            Ok(())
        }

        async fn delete(&self, id: &str) -> Result<()> {
            let mut memory_index = self.memory_index.write().map_err(|e| {
                MemoryError::VectorStore(format!("Failed to acquire write lock: {e}"))
            })?;
            memory_index.remove(id);
            Ok(())
        }

        async fn get(&self, id: &str) -> Result<Option<Memory>> {
            let memory_index = self.memory_index.read().map_err(|e| {
                MemoryError::VectorStore(format!("Failed to acquire read lock: {e}"))
            })?;
            Ok(memory_index.get(id).cloned())
        }

        async fn list(&self, filters: &Filters, limit: Option<usize>) -> Result<Vec<Memory>> {
            let memory_index = self.memory_index.read().map_err(|e| {
                MemoryError::VectorStore(format!("Failed to acquire read lock: {e}"))
            })?;
            
            let mut results: Vec<Memory> = memory_index
                .values()
                .filter(|m| matches_filters(m, filters))
                .cloned()
                .collect();
            
            if let Some(lim) = limit {
                results.truncate(lim);
            }
            Ok(results)
        }

        async fn count(&self) -> Result<usize> {
            let memory_index = self.memory_index.read().map_err(|e| {
                MemoryError::VectorStore(format!("Failed to acquire read lock: {e}"))
            })?;
            Ok(memory_index.len())
        }

        async fn health_check(&self) -> Result<bool> {
            Ok(true)
        }
    }

    fn matches_filters(memory: &Memory, filters: &Filters) -> bool {
        if let Some(ref mt) = filters.memory_type
            && &memory.metadata.memory_type != mt
        {
            return false;
        }
        if let Some(min_imp) = filters.min_importance
            && memory.metadata.importance_score < min_imp
        {
            return false;
        }
        if let Some(max_imp) = filters.max_importance
            && memory.metadata.importance_score > max_imp
        {
            return false;
        }
        // Add more filter checks as needed
        true
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum();
        let norm_b: f32 = b.iter().map(|x| x * x).sum();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a.sqrt() * norm_b.sqrt())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::types::{Memory, MemoryMetadata, MemoryType};

        fn make_test_memory(id: &str, embedding: Vec<f32>) -> Memory {
            Memory {
                id: id.to_string(),
                content: Some(format!("Content for {}", id)),
                content_meta: Default::default(),
                derived_data: Default::default(),
                relations: Default::default(),
                embedding,
                metadata: MemoryMetadata::new(MemoryType::Factual),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                context_embeddings: None,
                relation_embeddings: None,
            }
        }

        #[test]
        fn test_cosine_similarity_identical_vectors() {
            let a = vec![1.0, 2.0, 3.0];
            let b = vec![1.0, 2.0, 3.0];
            let sim = cosine_similarity(&a, &b);
            assert!((sim - 1.0).abs() < 1e-6);
        }

        #[test]
        fn test_cosine_similarity_orthogonal_vectors() {
            let a = vec![1.0, 0.0, 0.0];
            let b = vec![0.0, 1.0, 0.0];
            let sim = cosine_similarity(&a, &b);
            assert!((sim - 0.0).abs() < 1e-6);
        }

        #[test]
        fn test_cosine_similarity_opposite_vectors() {
            let a = vec![1.0, 0.0, 0.0];
            let b = vec![-1.0, 0.0, 0.0];
            let sim = cosine_similarity(&a, &b);
            assert!((sim - (-1.0)).abs() < 1e-6);
        }

        #[test]
        fn test_matches_filters_memory_type() {
            let mem = make_test_memory("test", vec![1.0, 0.0]);
            let filters = Filters {
                memory_type: Some(MemoryType::Factual),
                ..Default::default()
            };
            assert!(matches_filters(&mem, &filters));

            let filters_wrong = Filters {
                memory_type: Some(MemoryType::Conversational),
                ..Default::default()
            };
            assert!(!matches_filters(&mem, &filters_wrong));
        }

        #[test]
        fn test_matches_filters_min_importance() {
            let mut mem = make_test_memory("test", vec![1.0, 0.0]);
            mem.metadata.importance_score = 0.7;

            let filters = Filters {
                min_importance: Some(0.5),
                ..Default::default()
            };
            assert!(matches_filters(&mem, &filters));

            let filters_higher = Filters {
                min_importance: Some(0.8),
                ..Default::default()
            };
            assert!(!matches_filters(&mem, &filters_higher));
        }

        #[test]
        fn test_matches_filters_max_importance() {
            let mut mem = make_test_memory("test", vec![1.0, 0.0]);
            mem.metadata.importance_score = 0.5;

            let filters = Filters {
                max_importance: Some(0.8),
                ..Default::default()
            };
            assert!(matches_filters(&mem, &filters));

            let filters_lower = Filters {
                max_importance: Some(0.3),
                ..Default::default()
            };
            assert!(!matches_filters(&mem, &filters_lower));
        }

        #[tokio::test]
        async fn test_vector_lite_store_insert_and_get() {
            let config = VectorLiteConfig::default();
            let store = VectorLiteStore::with_config(config).unwrap();

            let mem = make_test_memory("mem1", vec![1.0, 0.0, 0.0]);
            store.insert(&mem).await.unwrap();

            let retrieved = store.get("mem1").await.unwrap();
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap().id, "mem1");
        }

        #[tokio::test]
        async fn test_vector_lite_store_update() {
            let config = VectorLiteConfig::default();
            let store = VectorLiteStore::with_config(config).unwrap();

            let mut mem = make_test_memory("mem2", vec![1.0, 0.0, 0.0]);
            mem.content = Some("Original".to_string());
            store.insert(&mem).await.unwrap();

            mem.content = Some("Updated".to_string());
            store.update(&mem).await.unwrap();

            let retrieved = store.get("mem2").await.unwrap();
            assert_eq!(retrieved.unwrap().content, Some("Updated".to_string()));
        }

        #[tokio::test]
        async fn test_vector_lite_store_delete() {
            let config = VectorLiteConfig::default();
            let store = VectorLiteStore::with_config(config).unwrap();

            let mem = make_test_memory("mem3", vec![1.0, 0.0, 0.0]);
            store.insert(&mem).await.unwrap();
            assert!(store.get("mem3").await.unwrap().is_some());

            store.delete("mem3").await.unwrap();
            assert!(store.get("mem3").await.unwrap().is_none());
        }

        #[tokio::test]
        async fn test_vector_lite_store_search() {
            let config = VectorLiteConfig::default();
            let store = VectorLiteStore::with_config(config).unwrap();

            let mem1 = make_test_memory("mem4", vec![1.0, 0.0, 0.0]);
            let mem2 = make_test_memory("mem5", vec![0.0, 1.0, 0.0]);
            let mem3 = make_test_memory("mem6", vec![0.5, 0.5, 0.0]);

            store.insert(&mem1).await.unwrap();
            store.insert(&mem2).await.unwrap();
            store.insert(&mem3).await.unwrap();

            let query = vec![1.0, 0.0, 0.0];
            let results = store.search(&query, &Filters::default(), 3).await.unwrap();

            assert_eq!(results.len(), 3);
            assert_eq!(results[0].memory.id, "mem4");
            assert!(results[0].score > results[1].score);
        }

        #[tokio::test]
        async fn test_vector_lite_store_search_with_filters() {
            let config = VectorLiteConfig::default();
            let store = VectorLiteStore::with_config(config).unwrap();

            let mut mem1 = make_test_memory("mem7", vec![1.0, 0.0, 0.0]);
            mem1.metadata.memory_type = MemoryType::Factual;
            mem1.metadata.importance_score = 0.8;

            let mut mem2 = make_test_memory("mem8", vec![0.0, 1.0, 0.0]);
            mem2.metadata.memory_type = MemoryType::Conversational;
            mem2.metadata.importance_score = 0.3;

            store.insert(&mem1).await.unwrap();
            store.insert(&mem2).await.unwrap();

            let query = vec![1.0, 0.0, 0.0];
            let filters = Filters {
                memory_type: Some(MemoryType::Factual),
                min_importance: Some(0.5),
                ..Default::default()
            };
            let results = store.search(&query, &filters, 10).await.unwrap();

            assert_eq!(results.len(), 1);
            assert_eq!(results[0].memory.id, "mem7");
        }

        #[tokio::test]
        async fn test_vector_lite_store_list_with_limit() {
            let config = VectorLiteConfig::default();
            let store = VectorLiteStore::with_config(config).unwrap();

            for i in 0..5 {
                let mem = make_test_memory(&format!("mem{}", i), vec![1.0, 0.0, 0.0]);
                store.insert(&mem).await.unwrap();
            }

            let all = store.list(&Filters::default(), None).await.unwrap();
            assert_eq!(all.len(), 5);

            let limited = store.list(&Filters::default(), Some(2)).await.unwrap();
            assert_eq!(limited.len(), 2);
        }

        #[tokio::test]
        async fn test_vector_lite_store_count() {
            let config = VectorLiteConfig::default();
            let store = VectorLiteStore::with_config(config).unwrap();

            assert_eq!(store.count().await.unwrap(), 0);

            let mem = make_test_memory("mem9", vec![1.0, 0.0, 0.0]);
            store.insert(&mem).await.unwrap();
            assert_eq!(store.count().await.unwrap(), 1);

            let mem2 = make_test_memory("mem10", vec![0.0, 1.0, 0.0]);
            store.insert(&mem2).await.unwrap();
            assert_eq!(store.count().await.unwrap(), 2);
        }

        #[tokio::test]
        async fn test_vector_lite_store_health_check() {
            let config = VectorLiteConfig::default();
            let store = VectorLiteStore::with_config(config).unwrap();

            assert!(store.health_check().await.unwrap());
        }
    }
}

#[cfg(feature = "vector-lite")]
pub use vectorlite_impl::{VectorLiteConfig, VectorLiteStore};

dyn_clone::clone_trait_object!(VectorStore);
