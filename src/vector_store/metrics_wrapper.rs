use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use std::collections::HashMap;

use crate::{
    error::Result,
    memory::metrics::{MetricsSink, StoragePhase},
    types::{Filters, Memory, ScoredMemory},
};

use super::{PruneStats, VectorStore};

/// A decorator that wraps a `VectorStore` and records timing metrics
/// for each operation via a `MetricsSink`.
#[derive(Clone)]
pub struct MetricsVectorStore {
    inner: Box<dyn VectorStore>,
    metrics: Arc<dyn MetricsSink>,
}

impl MetricsVectorStore {
    pub fn new(inner: Box<dyn VectorStore>, metrics: Arc<dyn MetricsSink>) -> Self {
        Self { inner, metrics }
    }

    pub fn inner(&self) -> &dyn VectorStore {
        &*self.inner
    }
}

#[async_trait]
impl VectorStore for MetricsVectorStore {
    async fn insert(&self, memory: &Memory) -> Result<()> {
        let start = Instant::now();
        let result = self.inner.insert(memory).await;
        self.metrics
            .record_storage_timing(StoragePhase::VsInsert, start.elapsed());
        result
    }

    async fn search(
        &self,
        query_vector: &[f32],
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        let start = Instant::now();
        let result = self.inner.search(query_vector, filters, limit).await;
        self.metrics
            .record_storage_timing(StoragePhase::VsSearch, start.elapsed());
        result
    }

    async fn search_with_threshold(
        &self,
        query_vector: &[f32],
        filters: &Filters,
        limit: usize,
        score_threshold: Option<f32>,
    ) -> Result<Vec<ScoredMemory>> {
        let start = Instant::now();
        let result = self
            .inner
            .search_with_threshold(query_vector, filters, limit, score_threshold)
            .await;
        self.metrics
            .record_storage_timing(StoragePhase::VsSearchWithThreshold, start.elapsed());
        result
    }

    async fn update(&self, memory: &Memory) -> Result<()> {
        let start = Instant::now();
        let result = self.inner.update(memory).await;
        self.metrics
            .record_storage_timing(StoragePhase::VsUpdate, start.elapsed());
        result
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let start = Instant::now();
        let result = self.inner.delete(id).await;
        self.metrics
            .record_storage_timing(StoragePhase::VsDelete, start.elapsed());
        result
    }

    async fn get(&self, id: &str) -> Result<Option<Memory>> {
        let start = Instant::now();
        let result = self.inner.get(id).await;
        self.metrics
            .record_storage_timing(StoragePhase::VsGet, start.elapsed());
        result
    }

    async fn list(&self, filters: &Filters, limit: Option<usize>) -> Result<Vec<Memory>> {
        let start = Instant::now();
        let result = self.inner.list(filters, limit).await;
        self.metrics
            .record_storage_timing(StoragePhase::VsList, start.elapsed());
        result
    }

    async fn count(&self) -> Result<usize> {
        let start = Instant::now();
        let result = self.inner.count().await;
        self.metrics
            .record_storage_timing(StoragePhase::VsCount, start.elapsed());
        result
    }

    async fn health_check(&self) -> Result<bool> {
        self.inner.health_check().await
    }

    async fn compact(&self) -> Result<()> {
        let start = Instant::now();
        let result = self.inner.compact().await;
        self.metrics
            .record_storage_timing(StoragePhase::VsCompact, start.elapsed());
        result
    }

    async fn prune(
        &self,
        older_than_days: Option<i64>,
        delete_unverified: bool,
    ) -> Result<PruneStats> {
        let start = Instant::now();
        let result = self
            .inner
            .prune(older_than_days, delete_unverified)
            .await;
        self.metrics
            .record_storage_timing(StoragePhase::VsPrune, start.elapsed());
        result
    }

    async fn find_by_relation_target(
        &self,
        target: &str,
        limit: Option<usize>,
    ) -> Result<Vec<Memory>> {
        let start = Instant::now();
        let result = self.inner.find_by_relation_target(target, limit).await;
        self.metrics
            .record_storage_timing(StoragePhase::VsFindByRelation, start.elapsed());
        result
    }

    async fn count_by_user(&self) -> Result<Vec<(Option<String>, usize)>> {
        self.inner.count_by_user().await
    }

    async fn count_by_agent(&self) -> Result<Vec<(Option<String>, usize)>> {
        self.inner.count_by_agent().await
    }

    async fn count_by_layer(&self) -> Result<HashMap<i32, usize>> {
        self.inner.count_by_layer().await
    }
}
