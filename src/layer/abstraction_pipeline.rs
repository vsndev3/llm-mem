use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, Notify, RwLock};
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::prompts::{build_l1_prompt, build_l2_prompt, build_l3_prompt};
use crate::{
    error::{MemoryError, Result},
    memory::MemoryManager,
    types::{Filters, LayerInfo, Memory, MemoryMetadata, MemoryType, RelationMeta},
};

/// Configuration for abstraction pipeline
#[derive(Debug, Clone)]
pub struct AbstractionConfig {
    pub enabled: bool,
    pub min_memories_for_l1: usize,
    pub l1_processing_delay: Duration,
    pub max_concurrent_tasks: usize,
}

impl Default for AbstractionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_memories_for_l1: 5,
            l1_processing_delay: Duration::from_secs(30),
            max_concurrent_tasks: 3,
        }
    }
}

/// Pending abstraction task
#[derive(Debug, Clone)]
pub struct PendingAbstraction {
    pub memory_id: Uuid,
    pub current_level: i32,
    pub target_level: i32,
    pub retry_count: u32,
    pub queued_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct L1Extraction {
    pub summary: String,
    pub structure_type: String,
    pub key_entities: Vec<String>,
    pub suggested_title: String,
    pub confidence: f32,
}

/// Manages background tasks that create higher-layer abstractions.
///
/// The pipeline monitors all loaded memory banks and creates progressive
/// abstractions: L0→L1 (structural summaries), L1→L2 (semantic synthesis),
/// L2→L3 (conceptual insights). A single unified worker cascades all layers
/// in one pass, and a `Notify` channel allows immediate wake-up when new
/// memories are stored.
pub struct AbstractionPipeline {
    /// Single-bank fallback — used when `banks` is empty (backwards compat)
    pub memory_manager: Arc<MemoryManager>,
    /// Shared bank registry for multi-bank processing
    banks: Arc<RwLock<HashMap<String, Arc<MemoryManager>>>>,
    pub config: AbstractionConfig,
    pub pending_queue: Arc<DashMap<Uuid, PendingAbstraction>>,
    shutdown_tx: broadcast::Sender<()>,
    /// Notify channel — wakes the unified worker immediately when a new memory is stored
    wake_notify: Arc<Notify>,
}

impl AbstractionPipeline {
    pub fn new(memory_manager: Arc<MemoryManager>, config: AbstractionConfig) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        Self {
            memory_manager,
            banks: Arc::new(RwLock::new(HashMap::new())),
            config,
            pending_queue: Arc::new(DashMap::new()),
            shutdown_tx,
            wake_notify: Arc::new(Notify::new()),
        }
    }

    /// Create a pipeline with shared bank registry for multi-bank processing
    pub fn with_banks(
        memory_manager: Arc<MemoryManager>,
        banks: Arc<RwLock<HashMap<String, Arc<MemoryManager>>>>,
        config: AbstractionConfig,
    ) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        Self {
            memory_manager,
            banks,
            config,
            pending_queue: Arc::new(DashMap::new()),
            shutdown_tx,
            wake_notify: Arc::new(Notify::new()),
        }
    }

    /// Notify the pipeline that new memory has been stored — wakes the worker immediately
    pub fn notify_new_memory(&self) {
        self.wake_notify.notify_one();
    }

    /// Expose shutdown sender so it can be triggered externally
    pub fn get_shutdown_sender(&self) -> broadcast::Sender<()> {
        self.shutdown_tx.clone()
    }

    /// Get all pending abstraction tasks for visualization
    pub fn get_pending_abstractions(&self) -> Vec<PendingAbstraction> {
        self.pending_queue
            .iter()
            .map(|item| item.value().clone())
            .collect()
    }

    /// Start a single unified pipeline worker that cascades L0→L1→L2→L3
    /// for all banks. Wakes on timer interval OR immediately via notify.
    pub fn start_unified_worker(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            self.unified_worker().await;
        })
    }

    /// Legacy: start only L0→L1 worker (delegates to unified worker)
    pub fn start_l0_to_l1_worker(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        self.start_unified_worker()
    }

    /// Legacy: no-op — unified worker handles L1→L2
    pub fn start_l1_to_l2_worker(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async {})
    }

    /// Legacy: no-op — unified worker handles L2→L3
    pub fn start_l2_to_l3_worker(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async {})
    }

    /// Unified worker loop: cascades L0→L1→L2→L3 across all banks.
    /// Wakes on either the polling interval or an immediate notify signal.
    async fn unified_worker(&self) {
        let mut interval = tokio::time::interval(self.config.l1_processing_delay);
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut consecutive_idle: u32 = 0;

        // Allow the first tick to pass immediately (tokio interval fires once immediately)
        interval.tick().await;

        loop {
            tokio::select! {
                biased; // prioritize shutdown
                _ = shutdown_rx.recv() => {
                    info!("Unified pipeline worker shutting down");
                    break;
                }
                _ = self.wake_notify.notified() => {
                    if !self.config.enabled { continue; }
                    debug!("Pipeline woke up: new memory notification");
                    // Small delay to batch rapid-fire stores
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    let result = self.run_full_pipeline_pass_internal().await;
                    if result.any_work_done() {
                        consecutive_idle = 0;
                    }
                }
                _ = interval.tick() => {
                    if !self.config.enabled { continue; }
                    let result = self.run_full_pipeline_pass_internal().await;
                    if result.any_work_done() {
                        consecutive_idle = 0;
                        info!("Pipeline cycle complete: {:?}", result);
                    } else {
                        consecutive_idle += 1;
                        if consecutive_idle <= 1 {
                            debug!("Pipeline idle (cycle {})", consecutive_idle);
                        }
                    }
                }
            }
        }
    }

    /// Get all MemoryManagers to process (multi-bank or single-bank fallback)
    async fn get_bank_managers(&self) -> Vec<(String, Arc<MemoryManager>)> {
        let banks = self.banks.read().await;
        if banks.is_empty() {
            // Fallback to single memory_manager
            vec![("default".to_string(), self.memory_manager.clone())]
        } else {
            banks.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        }
    }

    /// Run a full cascade pass across all banks: L0→L1, then L1→L2, then L2→L3
    async fn run_full_pipeline_pass_internal(&self) -> PipelinePassResult {
        let mut result = PipelinePassResult::default();
        let bank_managers = self.get_bank_managers().await;

        for (bank_name, manager) in &bank_managers {
            let bank_result = self.run_bank_pipeline_pass(bank_name, manager).await;
            result.l0_to_l1_created += bank_result.l0_to_l1_created;
            result.l1_to_l2_created += bank_result.l1_to_l2_created;
            result.l2_to_l3_created += bank_result.l2_to_l3_created;
            result.errors.extend(bank_result.errors);
        }

        result
    }

    /// Run cascade for a single bank's MemoryManager
    async fn run_bank_pipeline_pass(&self, bank_name: &str, manager: &Arc<MemoryManager>) -> PipelinePassResult {
        let mut result = PipelinePassResult::default();

        // Phase 1: L0 → L1
        let l0_count = Self::count_at_layer(manager, 0).await.unwrap_or(0);
        if l0_count >= self.config.min_memories_for_l1 {
            let pending = Self::find_pending_abstractions(manager, 0).await.unwrap_or_default();
            if !pending.is_empty() {
                info!("[{}] L0→L1: {} pending out of {} L0 memories", bank_name, pending.len(), l0_count);
                for memory_id in pending {
                    // Register only the item currently being processed for viz
                    self.pending_queue.insert(memory_id, PendingAbstraction {
                        memory_id,
                        current_level: 0,
                        target_level: 1,
                        retry_count: 0,
                        queued_at: Utc::now(),
                    });
                    match self.create_l1_abstraction_for(manager, memory_id).await {
                        Ok(l1_id) => {
                            result.l0_to_l1_created += 1;
                            info!("[{}] L0→L1 created: {} → {}", bank_name, memory_id, l1_id);
                        }
                        Err(e) => {
                            result.errors.push(format!("[{}] L0→L1 failed for {}: {}", bank_name, memory_id, e));
                            warn!("[{}] L0→L1 failed for {}: {}", bank_name, memory_id, e);
                        }
                    }
                    self.pending_queue.remove(&memory_id);
                }
            }
        }

        // Phase 2: L1 → L2 (cascade — runs immediately after L1s are created)
        let l1_count = Self::count_at_layer(manager, 1).await.unwrap_or(0);
        if l1_count >= 3 {
            loop {
                let group = Self::find_unabstracted_group_for(manager, 1, 3).await.unwrap_or_default();
                if group.len() < 3 { break; }
                info!("[{}] L1→L2: processing group of {} L1 memories", bank_name, group.len());
                // Register group in pending queue for viz
                for &id in &group {
                    self.pending_queue.insert(id, PendingAbstraction {
                        memory_id: id,
                        current_level: 1,
                        target_level: 2,
                        retry_count: 0,
                        queued_at: Utc::now(),
                    });
                }
                match self.create_l2_abstraction_for(manager, group.clone()).await {
                    Ok(l2_id) => {
                        result.l1_to_l2_created += 1;
                        info!("[{}] L1→L2 created: {}", bank_name, l2_id);
                    }
                    Err(e) => {
                        result.errors.push(format!("[{}] L1→L2 failed: {}", bank_name, e));
                        warn!("[{}] L1→L2 failed: {}", bank_name, e);
                        for id in &group { self.pending_queue.remove(id); }
                        break;
                    }
                }
                for id in &group { self.pending_queue.remove(id); }
            }
        }

        // Phase 3: L2 → L3 (cascade — runs immediately after L2s are created)
        let l2_count = Self::count_at_layer(manager, 2).await.unwrap_or(0);
        if l2_count >= 3 {
            loop {
                let group = Self::find_unabstracted_group_for(manager, 2, 3).await.unwrap_or_default();
                if group.len() < 3 { break; }
                info!("[{}] L2→L3: processing group of {} L2 memories", bank_name, group.len());
                // Register group in pending queue for viz
                for &id in &group {
                    self.pending_queue.insert(id, PendingAbstraction {
                        memory_id: id,
                        current_level: 2,
                        target_level: 3,
                        retry_count: 0,
                        queued_at: Utc::now(),
                    });
                }
                match self.create_l3_abstraction_for(manager, group.clone()).await {
                    Ok(l3_id) => {
                        result.l2_to_l3_created += 1;
                        info!("[{}] L2→L3 created: {}", bank_name, l3_id);
                    }
                    Err(e) => {
                        result.errors.push(format!("[{}] L2→L3 failed: {}", bank_name, e));
                        warn!("[{}] L2→L3 failed: {}", bank_name, e);
                        for id in &group { self.pending_queue.remove(id); }
                        break;
                    }
                }
                for id in &group { self.pending_queue.remove(id); }
            }
        }

        result
    }

    /// Backward compat: count memories at a layer for the default bank
    pub async fn count_memories_at_layer(&self, level: i32) -> Result<usize> {
        Self::count_at_layer(&self.memory_manager, level).await
    }

    /// Backward compat: find pending L0 abstractions for the default bank
    pub async fn find_pending_l0_abstractions(&self) -> Result<Vec<Uuid>> {
        Self::find_pending_abstractions(&self.memory_manager, 0).await
    }

    /// Backward compat: create L1 from default bank's memory manager
    pub async fn create_l1_abstraction(&self, memory_id: Uuid) -> Result<String> {
        self.create_l1_abstraction_for(&self.memory_manager, memory_id).await
    }

    // ── Static helpers: work with any MemoryManager ────────────────────

    /// Count memories at a given layer level for a specific manager
    async fn count_at_layer(manager: &MemoryManager, level: i32) -> Result<usize> {
        let mut filters = Filters::new();
        filters
            .custom
            .insert("layer.level".to_string(), serde_json::json!(level));
        let results = manager.list(&filters, None).await?;
        Ok(results.len())
    }

    /// Find L0 memories that have no corresponding L1 abstraction
    async fn find_pending_abstractions(manager: &MemoryManager, level: i32) -> Result<Vec<Uuid>> {
        let mut filters = Filters::new();
        filters
            .custom
            .insert("layer.level".to_string(), serde_json::json!(level));
        let results = manager.list(&filters, None).await?;

        let mut f_upper = Filters::new();
        f_upper
            .custom
            .insert("layer.level".to_string(), serde_json::json!(level + 1));
        let upper_memories = manager.list(&f_upper, None).await?;

        let mut abstracted_sources = std::collections::HashSet::new();
        for m in upper_memories {
            for src in &m.metadata.abstraction_sources {
                abstracted_sources.insert(*src);
            }
        }

        let mut pending = Vec::new();
        for m in results {
            if let Ok(id) = Uuid::parse_str(&m.id) {
                if !abstracted_sources.contains(&id) {
                    pending.push(id);
                }
            }
        }
        Ok(pending)
    }

    /// Find a group of unabstracted memories at a given layer level
    async fn find_unabstracted_group_for(
        manager: &MemoryManager,
        layer: i32,
        size: usize,
    ) -> Result<Vec<Uuid>> {
        let mut filters = Filters::new();
        filters
            .custom
            .insert("layer.level".to_string(), serde_json::json!(layer));
        let results = manager.list(&filters, None).await?;

        let mut upper_filters = Filters::new();
        upper_filters
            .custom
            .insert("layer.level".to_string(), serde_json::json!(layer + 1));
        let upper_memories = manager.list(&upper_filters, None).await?;

        let mut abstracted_sources = std::collections::HashSet::new();
        for m in upper_memories {
            for src in &m.metadata.abstraction_sources {
                abstracted_sources.insert(*src);
            }
        }

        let mut pending = Vec::new();
        for m in results {
            if let Ok(id) = Uuid::parse_str(&m.id) {
                if !abstracted_sources.contains(&id) {
                    pending.push(id);
                    if pending.len() == size {
                        break;
                    }
                }
            }
        }
        Ok(pending)
    }

    // ── Instance methods with explicit manager parameter ─────────────

    /// Create L1 abstraction for a specific manager (multi-bank variant)
    async fn create_l1_abstraction_for(
        &self,
        manager: &MemoryManager,
        memory_id: Uuid,
    ) -> Result<String> {
        let l0_memory = manager
            .get(&memory_id.to_string())
            .await?
            .ok_or_else(|| MemoryError::NotFound {
                id: memory_id.to_string(),
            })?;

        let prompt = build_l1_prompt(&l0_memory);
        let llm_response = manager.llm_client().complete(&prompt).await?;

        let json_start = llm_response.find('{').unwrap_or(0);
        let json_end = llm_response
            .rfind('}')
            .unwrap_or(llm_response.len().saturating_sub(1))
            + 1;
        let json_str = if json_start < json_end {
            &llm_response[json_start..json_end]
        } else {
            "{}"
        };

        let extraction: L1Extraction =
            serde_json::from_str(json_str).unwrap_or_else(|_| L1Extraction {
                summary: "Summary generation failed to parse.".to_string(),
                structure_type: "chunk".to_string(),
                key_entities: vec![],
                suggested_title: "Untitled".to_string(),
                confidence: 0.0,
            });

        let mut l1_memory = Memory::with_content(
            extraction.summary,
            l0_memory.embedding.clone(),
            MemoryMetadata::new(MemoryType::Semantic)
                .with_layer(LayerInfo::structural())
                .with_abstraction_sources(vec![memory_id]),
        );
        l1_memory.metadata.abstraction_confidence = Some(extraction.confidence);

        l1_memory.add_relation(
            "summary_of",
            vec![memory_id],
            Some(0.9),
            RelationMeta::new("llm:structural-abstraction").with_confidence(0.85),
        );

        let l1_id = manager.store_memory(l1_memory).await?;
        Ok(l1_id)
    }

    /// Create L2 abstraction for a specific manager (multi-bank variant)
    async fn create_l2_abstraction_for(
        &self,
        manager: &MemoryManager,
        memory_ids: Vec<Uuid>,
    ) -> Result<String> {
        let mut memories = Vec::new();
        for id in &memory_ids {
            if let Some(m) = manager.get(&id.to_string()).await? {
                memories.push(m);
            }
        }

        if memories.len() < 2 {
            return Err(MemoryError::Validation(format!(
                "Need at least 2 source memories for L2 abstraction, found {}",
                memories.len()
            )));
        }

        let memory_refs: Vec<&Memory> = memories.iter().collect();
        let prompt = build_l2_prompt(&memory_refs);
        let llm_response = manager.llm_client().complete(&prompt).await?;

        let json_start = llm_response.find('{').unwrap_or(0);
        let json_end = llm_response
            .rfind('}')
            .unwrap_or(llm_response.len().saturating_sub(1))
            + 1;
        let json_str = if json_start < json_end {
            &llm_response[json_start..json_end]
        } else {
            "{}"
        };

        let extraction: L2Extraction =
            serde_json::from_str(json_str).unwrap_or_else(|_| L2Extraction {
                synthesis: "L2 Synthesis failed.".to_string(),
                theme: "Unknown Theme".to_string(),
                shared_entities: vec![],
                confidence: 0.0,
            });

        let mut avg_embedding = vec![0.0f32; memories[0].embedding.len()];
        for m in &memories {
            for (i, v) in m.embedding.iter().enumerate() {
                if i < avg_embedding.len() {
                    avg_embedding[i] += v;
                }
            }
        }
        let count_f = memories.len() as f32;
        for v in &mut avg_embedding {
            *v /= count_f;
        }

        let mut meta = MemoryMetadata::new(MemoryType::Semantic)
            .with_layer(LayerInfo::semantic())
            .with_abstraction_sources(memory_ids.clone());
        meta.abstraction_confidence = Some(extraction.confidence);
        meta.topics.push(extraction.theme);

        let mut l2_memory = Memory::with_content(extraction.synthesis, avg_embedding, meta);

        l2_memory.add_relation(
            "synthesizes",
            memory_ids.clone(),
            Some(0.9),
            RelationMeta::new("llm:semantic-abstraction").with_confidence(0.85),
        );

        let l2_id = manager.store_memory(l2_memory).await?;
        Ok(l2_id)
    }

    /// Create L3 abstraction for a specific manager (multi-bank variant)
    async fn create_l3_abstraction_for(
        &self,
        manager: &MemoryManager,
        memory_ids: Vec<Uuid>,
    ) -> Result<String> {
        let mut memories = Vec::new();
        for id in &memory_ids {
            if let Some(m) = manager.get(&id.to_string()).await? {
                memories.push(m);
            }
        }

        if memories.len() < 2 {
            return Err(MemoryError::Validation(format!(
                "Need at least 2 source memories for L3 abstraction, found {}",
                memories.len()
            )));
        }

        let memory_refs: Vec<&Memory> = memories.iter().collect();
        let prompt = build_l3_prompt(&memory_refs);
        let llm_response = manager.llm_client().complete(&prompt).await?;

        let json_start = llm_response.find('{').unwrap_or(0);
        let json_end = llm_response
            .rfind('}')
            .unwrap_or(llm_response.len().saturating_sub(1))
            + 1;
        let json_str = if json_start < json_end {
            &llm_response[json_start..json_end]
        } else {
            "{}"
        };

        let extraction: L3Extraction =
            serde_json::from_str(json_str).unwrap_or_else(|_| L3Extraction {
                insight: "L3 Insight failed.".to_string(),
                concept: "Unknown Concept".to_string(),
                implications: vec![],
                confidence: 0.0,
            });

        let mut avg_embedding = vec![0.0f32; memories[0].embedding.len()];
        for m in &memories {
            for (i, v) in m.embedding.iter().enumerate() {
                if i < avg_embedding.len() {
                    avg_embedding[i] += v;
                }
            }
        }
        let count_f = memories.len() as f32;
        for v in &mut avg_embedding {
            *v /= count_f;
        }

        let mut meta = MemoryMetadata::new(MemoryType::Semantic)
            .with_layer(LayerInfo::concept())
            .with_abstraction_sources(memory_ids.clone());
        meta.abstraction_confidence = Some(extraction.confidence);
        meta.topics.push(extraction.concept);

        let mut l3_memory = Memory::with_content(extraction.insight, avg_embedding, meta);

        l3_memory.add_relation(
            "abstracts_to_concept",
            memory_ids.clone(),
            Some(0.9),
            RelationMeta::new("llm:conceptual-abstraction").with_confidence(0.85),
        );

        let l3_id = manager.store_memory(l3_memory).await?;
        Ok(l3_id)
    }

    // ── Public API: full pipeline pass + backward compat ─────────────

    /// Run a full cascade pass across all banks (public API for external trigger)
    pub async fn run_full_pipeline_pass(&self) -> Result<PipelinePassResult> {
        Ok(self.run_full_pipeline_pass_internal().await)
    }

    /// Backward compat: process all L1→L2 for the default bank
    pub async fn process_l1_to_l2(&self) -> Result<usize> {
        let mut created = 0;
        loop {
            let group = Self::find_unabstracted_group_for(&self.memory_manager, 1, 3).await?;
            if group.len() < 3 { break; }
            self.create_l2_abstraction_for(&self.memory_manager, group).await?;
            created += 1;
        }
        Ok(created)
    }

    /// Backward compat: process all L2→L3 for the default bank
    pub async fn process_l2_to_l3(&self) -> Result<usize> {
        let mut created = 0;
        loop {
            let group = Self::find_unabstracted_group_for(&self.memory_manager, 2, 3).await?;
            if group.len() < 3 { break; }
            self.create_l3_abstraction_for(&self.memory_manager, group).await?;
            created += 1;
        }
        Ok(created)
    }

    /// Backward compat: create L2 from default bank's memory manager
    pub async fn create_l2_abstraction(&self, memory_ids: Vec<Uuid>) -> Result<String> {
        self.create_l2_abstraction_for(&self.memory_manager, memory_ids).await
    }

    /// Backward compat: create L3 from default bank's memory manager
    pub async fn create_l3_abstraction(&self, memory_ids: Vec<Uuid>) -> Result<String> {
        self.create_l3_abstraction_for(&self.memory_manager, memory_ids).await
    }
}

/// Result of a full pipeline pass across all banks
#[derive(Debug, Clone, Default)]
pub struct PipelinePassResult {
    pub l0_to_l1_created: usize,
    pub l1_to_l2_created: usize,
    pub l2_to_l3_created: usize,
    pub errors: Vec<String>,
}

impl PipelinePassResult {
    pub fn any_work_done(&self) -> bool {
        self.l0_to_l1_created > 0 || self.l1_to_l2_created > 0 || self.l2_to_l3_created > 0
    }

    pub fn total_created(&self) -> usize {
        self.l0_to_l1_created + self.l1_to_l2_created + self.l2_to_l3_created
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct L2Extraction {
    pub synthesis: String,
    pub theme: String,
    pub shared_entities: Vec<String>,
    pub confidence: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct L3Extraction {
    pub insight: String,
    pub concept: String,
    pub implications: Vec<String>,
    pub confidence: f32,
}
