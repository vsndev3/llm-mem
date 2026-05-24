use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Notify, RwLock, broadcast};
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::pending_wal::{PendingWal, PendingWalEntry};
use super::prompts::{
    build_l1_prompt, build_l1_retry_prompt, build_l2_prompt, build_l2_retry_prompt,
    build_l3_prompt, build_l3_retry_prompt,
};
use crate::{
    error::{MemoryError, Result},
    llm::{LlmPriority, client::extract_json_from_text_tagged},
    memory::MemoryManager,
    types::{Filters, LayerInfo, Memory, MemoryMetadata, RelationMeta},
};

/// Safely truncate a string to at most `max_chars` characters, respecting UTF-8 boundaries.
fn safe_truncate(s: &str, max_chars: usize) -> &str {
    if s.len() <= max_chars {
        return s;
    }
    s.char_indices()
        .take(max_chars)
        .last()
        .map(|(idx, c)| &s[..idx + c.len_utf8()])
        .unwrap_or("")
}

/// Safely get a prefix of at most `max_chars` characters.
fn safe_prefix(s: &str, max_chars: usize) -> &str {
    safe_truncate(s, max_chars)
}

/// Safely get a suffix of at most `max_chars` characters.
fn safe_suffix(s: &str, max_chars: usize) -> &str {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        return s;
    }
    let skip = char_count - max_chars;
    let start = s.char_indices()
        .nth(skip)
        .map(|(idx, _)| idx)
        .unwrap_or(s.len());
    &s[start..]
}
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
    /// Optional WAL for persisting in-flight abstraction state to SQLite.
    /// When set, pending items survive process crashes and are re-queued on startup.
    wal: Option<Arc<PendingWal>>,
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
            wal: None,
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
            wal: None,
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

    /// Attach a WAL for persisting pending abstraction state.
    /// Call before starting the worker. Any previously persisted entries
    /// are loaded into the in-memory `pending_queue`.
    pub fn attach_wal(&mut self, wal: PendingWal) -> Result<()> {
        // Restore any previously persisted entries
        let entries = wal.load_all()?;
        let restored = entries.len();
        for entry in &entries {
            self.pending_queue.insert(
                entry.memory_id,
                PendingAbstraction {
                    memory_id: entry.memory_id,
                    current_level: entry.current_level,
                    target_level: entry.target_level,
                    retry_count: entry.retry_count,
                    queued_at: entry.queued_at,
                },
            );
        }
        self.wal = Some(Arc::new(wal));
        if restored > 0 {
            info!(
                "WAL attached: restored {} pending abstraction(s) from previous session",
                restored
            );
        } else {
            info!("WAL attached: no pending abstractions to restore");
        }
        Ok(())
    }

    /// Persist a pending item to the WAL (if configured).
    fn wal_insert(&self, memory_id: Uuid, current_level: i32, target_level: i32, retry_count: u32, bank_name: &str) {
        if let Some(ref wal) = self.wal {
            let entry = PendingWalEntry {
                memory_id,
                current_level,
                target_level,
                retry_count,
                queued_at: Utc::now(),
                bank_name: bank_name.to_string(),
            };
            if let Err(e) = wal.insert(&entry) {
                warn!("WAL insert failed for {}: {}", memory_id, e);
            }
        }
    }

    /// Remove a completed item from the WAL (if configured).
    fn wal_remove(&self, memory_id: &Uuid, bank_name: &str) {
        if let Some(ref wal) = self.wal
            && let Err(e) = wal.remove(memory_id, bank_name) {
                warn!("WAL remove failed for {}: {}", memory_id, e);
            }
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
                    consecutive_idle = 0; // reset on new activity
                    let result = self.run_full_pipeline_pass_internal().await;
                    if result.any_work_done() {
                        info!("Pipeline processed new memories: {:?}", result);
                    }
                }
                _ = interval.tick() => {
                    if !self.config.enabled { continue; }
                    let result = self.run_full_pipeline_pass_internal().await;
                    if result.any_work_done() {
                        consecutive_idle = 0;
                        info!("Pipeline cycle produced work: {:?}", result);
                    } else {
                        consecutive_idle += 1;
                        if consecutive_idle == 1 {
                            debug!("Pipeline idle — no unabstracted memories or all in backoff");
                        } else if consecutive_idle.is_multiple_of(4) {
                            info!("Pipeline idle for {} consecutive cycles", consecutive_idle);
                        }
                        if consecutive_idle >= Self::MAX_IDLE_CYCLES {
                            info!(
                                "Pipeline auto-stopping after {} idle cycles (no pending work)",
                                consecutive_idle
                            );
                            break;
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
    async fn run_bank_pipeline_pass(
        &self,
        bank_name: &str,
        manager: &Arc<MemoryManager>,
    ) -> PipelinePassResult {
        let mut result = PipelinePassResult::default();

        // Phase 1: L0 → L1
        let l0_count = Self::count_at_layer(manager, 0).await.unwrap_or(0);
        if l0_count >= self.config.min_memories_for_l1 {
            let pending = Self::find_pending_abstractions(manager, 0)
                .await
                .unwrap_or_default();
            if !pending.is_empty() {
                info!(
                    "[{}] L0→L1: {} pending out of {} L0 memories",
                    bank_name,
                    pending.len(),
                    l0_count
                );
                for memory_id in pending {
                    // Register only the item currently being processed for viz
                    self.pending_queue.insert(
                        memory_id,
                        PendingAbstraction {
                            memory_id,
                            current_level: 0,
                            target_level: 1,
                            retry_count: 0,
                            queued_at: Utc::now(),
                        },
                    );
                    self.wal_insert(memory_id, 0, 1, 0, bank_name);
                    match self.create_l1_abstraction_for(manager, memory_id).await {
                        Ok(l1_id) => {
                            result.l0_to_l1_created += 1;
                            info!("[{}] L0→L1 created: {} → {}", bank_name, memory_id, l1_id);
                            // Clear failure tracking on success
                            let _ = Self::clear_abstraction_failure(manager, memory_id).await;
                        }
                        Err(e) => {
                            result.errors.push(format!(
                                "[{}] L0→L1 failed for {}: {}",
                                bank_name, memory_id, e
                            ));
                            warn!("[{}] L0→L1 failed for {}: {}", bank_name, memory_id, e);
                            // Record failure with exponential backoff
                            let _ = Self::record_abstraction_failure(
                                manager,
                                memory_id,
                                &e.to_string(),
                            )
                            .await;
                        }
                    }
                    self.pending_queue.remove(&memory_id);
                    self.wal_remove(&memory_id, bank_name);
                }
            } else {
                // Log why zero eligible despite enough total L0s
                if let Ok((total, abstracted, backoff, eligible)) =
                    Self::layer_pending_breakdown(manager, 0).await
                {
                    info!(
                        "[{}] L0→L1 stalled: {} total L0s ({} already abstracted, {} in backoff, {} eligible)",
                        bank_name, total, abstracted, backoff, eligible
                    );
                }
            }
        }

        // Phase 2: L1 → L2 (cascade — runs immediately after L1s are created)
        let l1_count = Self::count_at_layer(manager, 1).await.unwrap_or(0);
        if l1_count >= 1 {
            loop {
                let group = Self::find_unabstracted_group_for(manager, 1, 2)
                    .await
                    .unwrap_or_default();
                if group.len() < Self::MIN_SOURCES_FOR_L2 {
                    if let Ok((total, abstracted, backoff, eligible)) =
                        Self::layer_pending_breakdown(manager, 1).await
                    {
                        info!(
                            "[{}] L1→L2 stalled: {} total L1s ({} already abstracted, {} in backoff, {} eligible, need at least {})",
                            bank_name, total, abstracted, backoff, eligible, Self::MIN_SOURCES_FOR_L2
                        );
                    }
                    break;
                }
                info!(
                    "[{}] L1→L2: processing group of {} L1 memories",
                    bank_name,
                    group.len()
                );
                // Register group in pending queue for viz
                for &id in &group {
                    self.pending_queue.insert(
                        id,
                        PendingAbstraction {
                            memory_id: id,
                            current_level: 1,
                            target_level: 2,
                            retry_count: 0,
                            queued_at: Utc::now(),
                        },
                    );
                    self.wal_insert(id, 1, 2, 0, bank_name);
                }
                match self.create_l2_abstraction_for(manager, group.clone()).await {
                    Ok(l2_id) => {
                        result.l1_to_l2_created += 1;
                        info!("[{}] L1→L2 created: {}", bank_name, l2_id);
                        for &id in &group {
                            let _ = Self::clear_abstraction_failure(manager, id).await;
                            self.pending_queue.remove(&id);
                            self.wal_remove(&id, bank_name);
                        }
                    }
                    Err(e) => {
                        result
                            .errors
                            .push(format!("[{}] L1→L2 failed: {}", bank_name, e));
                        warn!("[{}] L1→L2 failed: {}", bank_name, e);
                        for &id in &group {
                            let _ =
                                Self::record_abstraction_failure(manager, id, &e.to_string()).await;
                            self.pending_queue.remove(&id);
                            self.wal_remove(&id, bank_name);
                        }
                        // Continue to try remaining groups — don't break the phase
                    }
                }
            }
        }

        // Phase 3: L2 → L3 (cascade — runs immediately after L2s are created)
        let l2_count = Self::count_at_layer(manager, 2).await.unwrap_or(0);
        if l2_count >= 1 {
            loop {
                let group = Self::find_unabstracted_group_for(manager, 2, 2)
                    .await
                    .unwrap_or_default();
                if group.len() < Self::MIN_SOURCES_FOR_L3 {
                    if let Ok((total, abstracted, backoff, eligible)) =
                        Self::layer_pending_breakdown(manager, 2).await
                    {
                        info!(
                            "[{}] L2→L3 stalled: {} total L2s ({} already abstracted, {} in backoff, {} eligible, need at least {})",
                            bank_name, total, abstracted, backoff, eligible, Self::MIN_SOURCES_FOR_L3
                        );
                    }
                    break;
                }
                info!(
                    "[{}] L2→L3: processing group of {} L2 memories",
                    bank_name,
                    group.len()
                );
                // Register group in pending queue for viz
                for &id in &group {
                    self.pending_queue.insert(
                        id,
                        PendingAbstraction {
                            memory_id: id,
                            current_level: 2,
                            target_level: 3,
                            retry_count: 0,
                            queued_at: Utc::now(),
                        },
                    );
                    self.wal_insert(id, 2, 3, 0, bank_name);
                }
                match self.create_l3_abstraction_for(manager, group.clone()).await {
                    Ok(l3_id) => {
                        result.l2_to_l3_created += 1;
                        info!("[{}] L2→L3 created: {}", bank_name, l3_id);
                        for &id in &group {
                            let _ = Self::clear_abstraction_failure(manager, id).await;
                            self.pending_queue.remove(&id);
                            self.wal_remove(&id, bank_name);
                        }
                    }
                    Err(e) => {
                        result
                            .errors
                            .push(format!("[{}] L2→L3 failed: {}", bank_name, e));
                        warn!("[{}] L2→L3 failed: {}", bank_name, e);
                        for &id in &group {
                            let _ =
                                Self::record_abstraction_failure(manager, id, &e.to_string()).await;
                            self.pending_queue.remove(&id);
                            self.wal_remove(&id, bank_name);
                        }
                        // Continue to try remaining groups — don't break the phase
                    }
                }
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
        self.create_l1_abstraction_for(&self.memory_manager, memory_id)
            .await
    }

    // ── Static helpers: work with any MemoryManager ────────────────────

    /// Filter out chunk records (secondary index entries with parent_id set).
    /// These should participate in retrieval but not in the abstraction pipeline.
    fn exclude_chunks(memories: Vec<Memory>) -> Vec<Memory> {
        memories
            .into_iter()
            .filter(|m| m.metadata.parent_id.is_none())
            .collect()
    }

    /// Count memories at a given layer level for a specific manager
    async fn count_at_layer(manager: &MemoryManager, level: i32) -> Result<usize> {
        let mut filters = Filters::new();
        filters.min_layer_level = Some(level);
        filters.max_layer_level = Some(level);
        let results = manager.list(&filters, None).await?;
        // Exclude chunk records (used only as secondary index entries)
        let count = results.iter().filter(|m| m.metadata.parent_id.is_none()).count();
        Ok(count)
    }

    /// Find L0 memories that have no corresponding L1 abstraction and are not in backoff
    async fn find_pending_abstractions(manager: &MemoryManager, level: i32) -> Result<Vec<Uuid>> {
        let mut filters = Filters::new();
        filters.min_layer_level = Some(level);
        filters.max_layer_level = Some(level);
        let results: Vec<_> = manager
            .list(&filters, None)
            .await?
            .into_iter()
            .filter(|m| m.metadata.parent_id.is_none())
            .collect();

        let mut f_upper = Filters::new();
        f_upper.min_layer_level = Some(level + 1);
        f_upper.max_layer_level = Some(level + 1);
        let upper_memories = manager.list(&f_upper, None).await?;

        let mut abstracted_sources = std::collections::HashSet::new();
        for m in upper_memories {
            for src in &m.metadata.abstraction_sources {
                abstracted_sources.insert(*src);
            }
        }

        let mut pending = Vec::new();
        for m in results {
            if let Ok(id) = Uuid::parse_str(&m.id)
                && !abstracted_sources.contains(&id)
                && !Self::is_in_abstraction_backoff(&m.metadata)
            {
                pending.push(id);
            }
        }
        Ok(pending)
    }

    /// Diagnostic breakdown of layer memories: (total, already_abstracted, in_backoff, eligible)
    async fn layer_pending_breakdown(
        manager: &MemoryManager,
        level: i32,
    ) -> Result<(usize, usize, usize, usize)> {
        let mut filters = Filters::new();
        filters.min_layer_level = Some(level);
        filters.max_layer_level = Some(level);
        let results: Vec<_> = manager
            .list(&filters, None)
            .await?
            .into_iter()
            .filter(|m| m.metadata.parent_id.is_none())
            .collect();

        let mut f_upper = Filters::new();
        f_upper.min_layer_level = Some(level + 1);
        f_upper.max_layer_level = Some(level + 1);
        let upper_memories = manager.list(&f_upper, None).await?;

        let mut abstracted_sources = std::collections::HashSet::new();
        for m in upper_memories {
            for src in &m.metadata.abstraction_sources {
                abstracted_sources.insert(*src);
            }
        }

        let total = results.len();
        let mut already_abstracted = 0;
        let mut in_backoff = 0;
        let mut eligible = 0;

        for m in &results {
            if let Ok(id) = Uuid::parse_str(&m.id) {
                if abstracted_sources.contains(&id) {
                    already_abstracted += 1;
                } else if Self::is_in_abstraction_backoff(&m.metadata) {
                    in_backoff += 1;
                } else {
                    eligible += 1;
                }
            }
        }

        Ok((total, already_abstracted, in_backoff, eligible))
    }

    /// Find a group of unabstracted memories at a given layer level that are not in backoff
    async fn find_unabstracted_group_for(
        manager: &MemoryManager,
        layer: i32,
        size: usize,
    ) -> Result<Vec<Uuid>> {
        let mut filters = Filters::new();
        filters.min_layer_level = Some(layer);
        filters.max_layer_level = Some(layer);
        let results = Self::exclude_chunks(manager.list(&filters, None).await?);

        let mut upper_filters = Filters::new();
        upper_filters.min_layer_level = Some(layer + 1);
        upper_filters.max_layer_level = Some(layer + 1);
        let upper_memories = manager.list(&upper_filters, None).await?;

        let mut abstracted_sources = std::collections::HashSet::new();
        for m in &upper_memories {
            for src in &m.metadata.abstraction_sources {
                abstracted_sources.insert(*src);
            }
        }

        debug!(
            "find_unabstracted_group_for(layer={}): {} total, {} at upper L{}, {} source IDs, need group size {}",
            layer,
            results.len(),
            upper_memories.len(),
            layer + 1,
            abstracted_sources.len(),
            size
        );

        let mut pending = Vec::new();
        for m in results {
            if let Ok(id) = Uuid::parse_str(&m.id)
                && !abstracted_sources.contains(&id)
                && !Self::is_in_abstraction_backoff(&m.metadata)
            {
                pending.push(id);
                if pending.len() == size {
                    break;
                }
            }
        }
        Ok(pending)
    }

    /// Maximum number of abstraction failures before backoff is cleared
    /// and the memory is returned to the eligible pool for immediate retry.
    const MAX_ABSTRACTION_FAILURES: u32 = 5;
    const MAX_IDLE_CYCLES: u32 = 12;
    const MIN_SOURCES_FOR_L2: usize = 2;
    const MIN_SOURCES_FOR_L3: usize = 2;

    /// After MAX_ABSTRACTION_FAILURES (5): apply 1hr cooldown and reset counter
    async fn record_abstraction_failure(
        manager: &MemoryManager,
        memory_id: Uuid,
        error_msg: &str,
    ) -> Result<()> {
        if let Some(mut memory) = manager.get(&memory_id.to_string()).await? {
            let now = Utc::now();

            memory.metadata.abstraction_failure_count += 1;
            let failure_count = memory.metadata.abstraction_failure_count;

            if failure_count >= Self::MAX_ABSTRACTION_FAILURES {
                memory.metadata.last_abstraction_failure = Some(now);
                memory.metadata.abstraction_retry_after =
                    Some(now + chrono::Duration::seconds(3600));
                memory.metadata.abstraction_failure_count = 0;
                info!(
                    "[abstraction] {} failures exceeded for {} — 1hr cooldown before retry ({})",
                    failure_count, memory_id, error_msg
                );
            } else {
                let previous_retry_after = memory.metadata.abstraction_retry_after;
                let previous_failure_time = memory.metadata.last_abstraction_failure;

                memory.metadata.last_abstraction_failure = Some(now);

                let backoff_secs = if previous_retry_after.is_some() {
                    if let Some(last_failure) = previous_failure_time {
                        let elapsed = (now - last_failure).num_seconds().max(1) as u64;
                        (elapsed * 2).min(3600)
                    } else {
                        120
                    }
                } else {
                    60
                };

                memory.metadata.abstraction_retry_after =
                    Some(now + chrono::Duration::seconds(backoff_secs as i64));

                debug!(
                    "[abstraction] Failure {}/{} for {}: backoff {}s ({})",
                    failure_count,
                    Self::MAX_ABSTRACTION_FAILURES,
                    memory_id,
                    backoff_secs,
                    error_msg
                );
            }

            manager.update_memory(&memory).await?;
            Ok(())
        } else {
            Err(MemoryError::NotFound {
                id: memory_id.to_string(),
            })
        }
    }

    /// Clear abstraction failure tracking from a memory's metadata (called on successful abstraction)
    async fn clear_abstraction_failure(manager: &MemoryManager, memory_id: Uuid) -> Result<()> {
        if let Some(mut memory) = manager.get(&memory_id.to_string()).await? {
            memory.metadata.last_abstraction_failure = None;
            memory.metadata.abstraction_retry_after = None;
            memory.metadata.abstraction_failure_count = 0;
            manager.update_memory(&memory).await?;
        }
        Ok(())
    }

    /// Check if a memory is currently in abstraction backoff (should not be retried yet)
    fn is_in_abstraction_backoff(metadata: &crate::types::MemoryMetadata) -> bool {
        if let Some(retry_after) = metadata.abstraction_retry_after {
            Utc::now() < retry_after
        } else {
            false
        }
    }

    /// Count memories at a given layer that are NOT yet abstracted (for pipeline status)
    /// This is the true "pending" count — memories that have no upper-layer abstraction
    /// referencing them and are not currently in backoff.
    pub async fn count_unabstracted_at_layer(manager: &MemoryManager, level: i32) -> Result<usize> {
        let mut filters = Filters::new();
        filters.min_layer_level = Some(level);
        filters.max_layer_level = Some(level);
        let results = Self::exclude_chunks(manager.list(&filters, None).await?);

        let mut f_upper = Filters::new();
        f_upper.min_layer_level = Some(level + 1);
        f_upper.max_layer_level = Some(level + 1);
        let upper_memories = manager.list(&f_upper, None).await?;

        let mut abstracted_sources = std::collections::HashSet::new();
        for m in upper_memories {
            for src in &m.metadata.abstraction_sources {
                abstracted_sources.insert(*src);
            }
        }

        let mut unabstracted_count = 0;
        for m in results {
            if let Ok(id) = Uuid::parse_str(&m.id)
                && !abstracted_sources.contains(&id)
                && !Self::is_in_abstraction_backoff(&m.metadata)
            {
                unabstracted_count += 1;
            }
        }
        Ok(unabstracted_count)
    }

    /// Clear abstraction backoff timers for all memories at a given layer.
    /// This allows the pipeline to retry failed abstractions immediately.
    /// Returns the number of memories that had their backoff cleared.
    pub async fn clear_backoff_timers(manager: &MemoryManager, layer: i32) -> Result<usize> {
        let mut filters = Filters::new();
        filters.min_layer_level = Some(layer);
        filters.max_layer_level = Some(layer);
        let results = manager.list(&filters, None).await?;

        let mut cleared_count = 0;
        for m in results {
            // Only clear if the memory has backoff timers set
            if m.metadata.abstraction_retry_after.is_some()
                || m.metadata.last_abstraction_failure.is_some()
            {
                let mut memory = m;
                memory.metadata.abstraction_retry_after = None;
                memory.metadata.last_abstraction_failure = None;
                memory.metadata.abstraction_failure_count = 0;
                manager.update_memory(&memory).await?;
                cleared_count += 1;
            }
        }
        Ok(cleared_count)
    }

    // ── GBNF grammars for JSON-constrained abstraction generation ────
    // Kept for future use; see https://github.com/ggml-org/llama.cpp/issues/21730

    #[allow(dead_code)]
    const L1_GRAMMAR: &str = r##"root ::= object
object ::= "{" ws "\"summary\"" ws ":" ws string ws "," ws "\"structure_type\"" ws ":" ws struct-type ws "," ws "\"key_entities\"" ws ":" ws key-array ws "," ws "\"suggested_title\"" ws ":" ws string ws "," ws "\"confidence\"" ws ":" ws number ws "}"
struct-type ::= "\"chunk\"" | "\"section\"" | "\"chapter\"" | "\"document\"" | "\"conversational_thread\""
key-array ::= "[" ws (string (ws "," ws string)*)? ws "]"
string ::= "\"" [^"\\]* "\""
number ::= ("0" | "1") ("." [0-9]+)?
ws ::= [ \t\n]*"##;

    #[allow(dead_code)]
    const L2_GRAMMAR: &str = r##"root ::= object
object ::= "{" ws "\"synthesis\"" ws ":" ws string ws "," ws "\"theme\"" ws ":" ws string ws "," ws "\"shared_entities\"" ws ":" ws key-array ws "," ws "\"confidence\"" ws ":" ws number ws "}"
key-array ::= "[" ws (string (ws "," ws string)*)? ws "]"
string ::= "\"" [^"\\]* "\""
number ::= ("0" | "1") ("." [0-9]+)?
ws ::= [ \t\n]*"##;

    #[allow(dead_code)]
    const L3_GRAMMAR: &str = r##"root ::= object
object ::= "{" ws "\"insight\"" ws ":" ws string ws "," ws "\"concept\"" ws ":" ws string ws "," ws "\"implications\"" ws ":" ws key-array ws "," ws "\"confidence\"" ws ":" ws number ws "}"
key-array ::= "[" ws (string (ws "," ws string)*)? ws "]"
string ::= "\"" [^"\\]* "\""
number ::= ("0" | "1") ("." [0-9]+)?
ws ::= [ \t\n]*"##;

    // ── Instance methods with explicit manager parameter ─────────────

    /// Create L1 abstraction for a specific manager (multi-bank variant)
    async fn create_l1_abstraction_for(
        &self,
        manager: &MemoryManager,
        memory_id: Uuid,
    ) -> Result<String> {
        let l0_memory =
            manager
                .get(&memory_id.to_string())
                .await?
                .ok_or_else(|| MemoryError::NotFound {
                    id: memory_id.to_string(),
                })?;

        let section_headers: Vec<String> = {
            let mut headers = Vec::new();
            for rel in &l0_memory.metadata.relations {
                if rel.relation == "part_of"
                    && let Ok(Some(parent)) = manager.get(&rel.target).await
                    && parent.metadata.custom.get("is_header").and_then(|v| v.as_bool()).unwrap_or(false)
                    && let Some(header_level) = parent.metadata.custom.get("header_level").and_then(|v| v.as_u64())
                    && let Some(title) = parent.content
                {
                    headers.push((header_level, title));
                }
            }
            headers.sort_by_key(|(lvl, _)| *lvl);
            headers.into_iter().map(|(_, title)| title).collect()
        };

        let context = super::prompts::L1Context {
            file_name: l0_memory.metadata.custom.get("file_path").and_then(|v| v.as_str()),
            chunk_index: l0_memory.metadata.custom.get("chunk_index").and_then(|v| v.as_u64()).map(|n| n as usize),
            total_chunks: l0_memory.metadata.custom.get("total_chunks").and_then(|v| v.as_u64()).map(|n| n as usize),
            section_headers: &section_headers,
        };

        let prompt = build_l1_prompt(&l0_memory, &context);
        debug!(
            "L1 LLM request for {} ({} bytes): \"{}\"...\"{}\"",
            memory_id,
            prompt.len(),
            safe_prefix(&prompt, 200),
            safe_suffix(&prompt, 200)
        );
        let mut llm_response = {
            let _guard = manager.priority_client().acquire(LlmPriority::Background).await;
            manager.priority_client().inner().complete(&prompt).await?
        };
        debug!(
            "L1 LLM response for {} ({} bytes): \"{}\"...\"{}\"",
            memory_id,
            llm_response.len(),
            safe_prefix(&llm_response, 200),
            safe_suffix(&llm_response, 200)
        );

        const MAX_JSON_PARSE_RETRIES: u32 = 2;
        let mut retry_count = 0;
        let mut extraction = try_parse_l1(&llm_response);

        while extraction.is_none() && retry_count < MAX_JSON_PARSE_RETRIES {
            let parse_error = diagnose_l1_parse_error(&llm_response);
            debug!(
                "L1 JSON parse failed (attempt {}/{}): {}. Raw LLM response ({} bytes): {}",
                retry_count + 1,
                MAX_JSON_PARSE_RETRIES + 1,
                parse_error,
                llm_response.len(),
                safe_prefix(&llm_response, 500)
            );
            let retry_prompt = build_l1_retry_prompt(&l0_memory, &context, &llm_response, &parse_error);
            debug!(
                "L1 retry LLM request for {} ({} bytes): \"{}\"...\"{}\"",
                memory_id,
                retry_prompt.len(),
                safe_prefix(&retry_prompt, 200),
                safe_suffix(&retry_prompt, 200)
            );
            llm_response = {
                let _guard = manager.priority_client().acquire(LlmPriority::Background).await;
                manager.priority_client().inner().complete(&retry_prompt).await?
            };
            debug!(
                "L1 retry LLM response for {} ({} bytes): \"{}\"...\"{}\"",
                memory_id,
                llm_response.len(),
                safe_prefix(&llm_response, 200),
                safe_suffix(&llm_response, 200)
            );
            extraction = try_parse_l1(&llm_response);
            retry_count += 1;
        }

        let extraction = extraction.ok_or_else(|| {
            let diag = diagnose_l1_parse_error(&llm_response);
            let fail_path = Self::write_failure_log(memory_id, "l1", &llm_response, &diag);
            let fail_display = fail_path.unwrap_or_else(|| "N/A".to_string());
            info!(
                "L1 JSON parse failed after {} retries ({} bytes, {}). Failure log: {}",
                MAX_JSON_PARSE_RETRIES + 1,
                llm_response.len(),
                diag,
                fail_display
            );
            MemoryError::LLM(format!(
                "L1 JSON parse failed after {} attempts ({} bytes): {}. See {}",
                MAX_JSON_PARSE_RETRIES + 1,
                llm_response.len(),
                diag,
                fail_display
            ))
        })?;

        let mut l1_memory = Memory::with_content(
            extraction.summary,
            l0_memory.embedding.clone(),
            MemoryMetadata::new()
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
}

/// Attempt to parse JSON from an LLM response into an L1Extraction.
/// Uses serde_json::Value as intermediate to be more lenient with edge cases.
fn try_parse_l1(llm_response: &str) -> Option<L1Extraction> {
    let json_str = extract_json_from_text_tagged(llm_response, &["think".to_string()])?;
    let repaired = jsonrepair::repair_json(&json_str, &jsonrepair::Options::default())
        .unwrap_or(json_str);

    let value: serde_json::Value = serde_json::from_str(&repaired).ok()?;
    let obj = value.as_object()?;
    let summary = obj.get("summary").and_then(|v| v.as_str())?.to_string();
    let structure_type = obj.get("structure_type").and_then(|v| v.as_str())?.to_string();
    let key_entities: Vec<String> = obj
        .get("key_entities")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let suggested_title = obj
        .get("suggested_title")
        .and_then(|v| v.as_str())?
        .to_string();
    let confidence = obj
        .get("confidence")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .or_else(|| obj.get("confidence").and_then(|v| v.as_i64()).map(|i| i as f32))?;

    Some(L1Extraction {
        summary,
        structure_type,
        key_entities,
        suggested_title,
        confidence,
    })
}

/// Attempt to repair and parse JSON into a target type T.
fn try_parse_json_with_repair<T: serde::de::DeserializeOwned>(llm_response: &str) -> Option<T> {
    let json_str = extract_json_from_text_tagged(llm_response, &["think".to_string()])?;
    let repaired = jsonrepair::repair_json(&json_str, &jsonrepair::Options::default())
        .unwrap_or(json_str);
    serde_json::from_str(&repaired).ok()
}

/// Diagnose why L1 JSON parsing failed, returning a human-readable error message.
fn diagnose_l1_parse_error(llm_response: &str) -> String {
    match extract_json_from_text_tagged(llm_response, &["think".to_string()]) {
        Some(json_str) => {
            let repaired = jsonrepair::repair_json(&json_str, &jsonrepair::Options::default())
                .unwrap_or(json_str);
            match serde_json::from_str::<serde_json::Value>(&repaired) {
                Err(e) => format!(
                    "JSON syntax error at line {} column {}: {} (extracted: {})",
                    e.line(),
                    e.column(),
                    e,
                    safe_prefix(&repaired, 200)
                ),
                Ok(value) => {
                    let missing: Vec<String> = [
                        ("summary", "string", value.get("summary").and_then(|v| v.as_str()).is_none()),
                        ("structure_type", "string", value.get("structure_type").and_then(|v| v.as_str()).is_none()),
                        ("key_entities", "array", value.get("key_entities").and_then(|v| v.as_array()).is_none()),
                        ("suggested_title", "string", value.get("suggested_title").and_then(|v| v.as_str()).is_none()),
                        ("confidence", "number", value.get("confidence").and_then(|v| v.as_f64()).or_else(|| value.get("confidence").and_then(|v| v.as_i64()).map(|i| i as f64)).is_none()),
                    ]
                    .iter()
                    .filter(|(_, _, missing)| *missing)
                    .map(|(name, expected, _)| format!("{} (expected {})", name, expected))
                    .collect::<Vec<_>>();
                    if missing.is_empty() {
                        "JSON extracted but failed to match expected L1 schema (all fields present, wrong types?)".to_string()
                    } else {
                        format!("JSON parsed but missing or wrong-type fields: {}", missing.join(", "))
                    }
                }
            }
        }
        None => "No valid JSON block found in response (missing or unclosed braces)".to_string(),
    }
}

/// Generic JSON diagnostic for L2/L3 failures.
fn diagnose_json_error(llm_response: &str) -> String {
    match extract_json_from_text_tagged(llm_response, &["think".to_string()]) {
        Some(json_str) => {
            let len = json_str.len();
            match serde_json::from_str::<serde_json::Value>(&json_str) {
                Err(e) => format!("JSON syntax error at line {}: {} ({} bytes)", e.line(), e, len),
                Ok(_) => format!("JSON valid but failed to match expected schema ({} bytes)", len),
            }
        }
        None => "No valid JSON block found in response".to_string(),
    }
}

impl AbstractionPipeline {
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

        if memories.is_empty() {
            return Err(MemoryError::Validation(format!(
                "Need at least 1 source memory for L2 abstraction, found {}",
                memories.len()
            )));
        }

        let memory_refs: Vec<&Memory> = memories.iter().collect();
        let prompt = build_l2_prompt(&memory_refs);
        debug!(
            "L2 LLM request ({} bytes): \"{}\"...\"{}\"",
            prompt.len(),
            safe_prefix(&prompt, 200),
            safe_suffix(&prompt, 200)
        );
        let llm_response = {
            let _guard = manager.priority_client().acquire(LlmPriority::Background).await;
            manager.priority_client().inner().complete(&prompt).await?
        };
        debug!(
            "L2 LLM response ({} bytes): \"{}\"...\"{}\"",
            llm_response.len(),
            safe_prefix(&llm_response, 200),
            safe_suffix(&llm_response, 200)
        );

        const L2_MAX_JSON_PARSE_RETRIES: u32 = 2;
        let mut l2_retry_count = 0;
        let mut llm_response = llm_response;
        let mut extraction: Option<L2Extraction> = try_parse_json_with_repair(&llm_response);

        while extraction.is_none() && l2_retry_count < L2_MAX_JSON_PARSE_RETRIES {
            debug!(
                "L2 JSON parse failed (attempt {}/{}). Raw LLM response ({} bytes): {}",
                l2_retry_count + 1,
                L2_MAX_JSON_PARSE_RETRIES + 1,
                llm_response.len(),
                safe_prefix(&llm_response, 500)
            );
            let memory_refs_clone: Vec<&Memory> = memories.iter().collect();
            let retry_prompt =
                build_l2_retry_prompt(&memory_refs_clone, &llm_response);
            llm_response = {
                let _guard = manager.priority_client().acquire(LlmPriority::Background).await;
                manager.priority_client().inner().complete(&retry_prompt).await?
            };
            extraction = try_parse_json_with_repair(&llm_response);
            l2_retry_count += 1;
        }

        let extraction: L2Extraction = extraction.unwrap_or_else(|| {
            let diag = diagnose_json_error(&llm_response);
            let log_id = memory_ids.first().copied().unwrap_or(Uuid::nil());
            let fail_path = Self::write_failure_log(log_id, "l2", &llm_response, &diag);
            info!(
                "L2 JSON parse failed after {} retries. Failure: {}",
                L2_MAX_JSON_PARSE_RETRIES + 1,
                fail_path.unwrap_or_else(|| "N/A".to_string())
            );
            L2Extraction {
                synthesis: "L2 Synthesis failed.".to_string(),
                theme: "Unknown Theme".to_string(),
                shared_entities: vec![],
                confidence: 0.0,
            }
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

        let mut meta = MemoryMetadata::new()
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

        if memories.is_empty() {
            return Err(MemoryError::Validation(format!(
                "Need at least 1 source memory for L3 abstraction, found {}",
                memories.len()
            )));
        }

        let memory_refs: Vec<&Memory> = memories.iter().collect();
        let prompt = build_l3_prompt(&memory_refs);
        debug!(
            "L3 LLM request ({} bytes): \"{}\"...\"{}\"",
            prompt.len(),
            safe_prefix(&prompt, 200),
            safe_suffix(&prompt, 200)
        );
        let llm_response = {
            let _guard = manager.priority_client().acquire(LlmPriority::Background).await;
            manager.priority_client().inner().complete(&prompt).await?
        };
        debug!(
            "L3 LLM response ({} bytes): \"{}\"...\"{}\"",
            llm_response.len(),
            safe_prefix(&llm_response, 200),
            safe_suffix(&llm_response, 200)
        );

        const L3_MAX_JSON_PARSE_RETRIES: u32 = 2;
        let mut l3_retry_count = 0;
        let mut llm_response = llm_response;
        let mut extraction: Option<L3Extraction> = try_parse_json_with_repair(&llm_response);

        while extraction.is_none() && l3_retry_count < L3_MAX_JSON_PARSE_RETRIES {
            debug!(
                "L3 JSON parse failed (attempt {}/{}). Raw LLM response ({} bytes): {}",
                l3_retry_count + 1,
                L3_MAX_JSON_PARSE_RETRIES + 1,
                llm_response.len(),
                safe_prefix(&llm_response, 500)
            );
            let memory_refs_clone: Vec<&Memory> = memories.iter().collect();
            let retry_prompt =
                build_l3_retry_prompt(&memory_refs_clone, &llm_response);
            llm_response = {
                let _guard = manager.priority_client().acquire(LlmPriority::Background).await;
                manager.priority_client().inner().complete(&retry_prompt).await?
            };
            extraction = try_parse_json_with_repair(&llm_response);
            l3_retry_count += 1;
        }

        let extraction: L3Extraction = extraction.unwrap_or_else(|| {
            let diag = diagnose_json_error(&llm_response);
            let log_id = memory_ids.first().copied().unwrap_or(Uuid::nil());
            let fail_path = Self::write_failure_log(log_id, "l3", &llm_response, &diag);
            info!(
                "L3 JSON parse failed after {} retries. Failure: {}",
                L3_MAX_JSON_PARSE_RETRIES + 1,
                fail_path.unwrap_or_else(|| "N/A".to_string())
            );
            L3Extraction {
                insight: "L3 Insight failed.".to_string(),
                concept: "Unknown Concept".to_string(),
                implications: vec![],
                confidence: 0.0,
            }
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

        let mut meta = MemoryMetadata::new()
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
            let group = Self::find_unabstracted_group_for(&self.memory_manager, 1, 2).await?;
            if group.len() < 2 {
                break;
            }
            self.create_l2_abstraction_for(&self.memory_manager, group)
                .await?;
            created += 1;
        }
        Ok(created)
    }

    /// Backward compat: process all L2→L3 for the default bank
    pub async fn process_l2_to_l3(&self) -> Result<usize> {
        let mut created = 0;
        loop {
            let group = Self::find_unabstracted_group_for(&self.memory_manager, 2, 2).await?;
            if group.len() < 2 {
                break;
            }
            self.create_l3_abstraction_for(&self.memory_manager, group)
                .await?;
            created += 1;
        }
        Ok(created)
    }

    /// Backward compat: create L2 from default bank's memory manager
    pub async fn create_l2_abstraction(&self, memory_ids: Vec<Uuid>) -> Result<String> {
        self.create_l2_abstraction_for(&self.memory_manager, memory_ids)
            .await
    }

    /// Backward compat: create L3 from default bank's memory manager
    pub async fn create_l3_abstraction(&self, memory_ids: Vec<Uuid>) -> Result<String> {
        self.create_l3_abstraction_for(&self.memory_manager, memory_ids)
            .await
    }

    /// Write a failed abstraction's full LLM response and diagnostic to a log file.
    fn write_failure_log(memory_id: Uuid, layer: &str, response: &str, diagnostic: &str) -> Option<String> {
        let dir = std::path::PathBuf::from("llm-mem-data/failures");
        let _ = std::fs::create_dir_all(&dir);
        let ts = Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("{}_{}.txt", layer, ts);
        let path = dir.join(&filename);
        let content = format!(
            "Memory: {}\nLayer: {}\nTime: {}\nDiagnostic: {}\n\nLLM Response ({} bytes):\n{}",
            memory_id,
            layer,
            Utc::now().to_rfc3339(),
            diagnostic,
            response.len(),
            response
        );
        std::fs::write(&path, &content).ok()?;
        Some(path.display().to_string())
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

#[cfg(test)]
mod parse_tests {
    use super::*;

    #[test]
    fn test_try_parse_l1_with_latex_in_key_entities() {
        let response = r#"{
  "summary": "This section introduces the concept of higher-dimensional Euclidean spaces, denoted as $\\mathbb{R}^n$. It establishes the necessary mathematical framework to proceed with the analysis, preparing the reader for concepts involving infinite dimensions.",
  "structure_type": "section",
  "key_entities": ["Higher-Dimensional Euclidean Spaces", "$\\mathbb{R}^n$"],
  "suggested_title": "Introduction to Higher-Dimensional Euclidean Spaces",
  "confidence": 0.98
}"#;
        let result = try_parse_l1(response);
        assert!(result.is_some(), "try_parse_l1 returned None for valid JSON. Diagnostic: {}", diagnose_l1_parse_error(response));
    }

    #[test]
    fn test_try_parse_l1_simple() {
        let response = r#"{
  "summary": "hello world",
  "structure_type": "section",
  "key_entities": ["foo", "bar"],
  "suggested_title": "Test",
  "confidence": 0.95
}"#;
        let result = try_parse_l1(response);
        assert!(result.is_some(), "try_parse_l1 returned None for simple JSON. Diagnostic: {}", diagnose_l1_parse_error(response));
    }
}
