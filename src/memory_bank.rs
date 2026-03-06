//! Memory bank management — namespaced, isolated memory stores.
//!
//! A "memory bank" is a named, independent memory store with its own database file.
//! This allows organizing memories by context (e.g., per-project, per-domain,
//! per-conversation topic).
//!
//! # Architecture
//!
//! - Each bank has its own `VectorLiteStore` and `MemoryManager`
//! - All banks share a single `LLMClient` (expensive resource)
//! - Banks are stored as individual `.db` files in a `banks_dir`
//! - A `"default"` bank is always available (backward compatible)
//! - Banks are lazily loaded on first access

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use tracing::{error, info, warn};

use crate::{
    config::{MemoryConfig, VectorStoreConfig},
    document_session::DocumentSessionManager,
    error::{MemoryError, Result},
    layer::abstraction_pipeline::{AbstractionConfig, AbstractionPipeline},
    llm::LLMClient,
    memory::MemoryManager,
    types::Filters,
    vector_store::{VectorLiteConfig, VectorLiteStore, VectorStore},
};

/// Default bank name used when no bank is specified.
pub const DEFAULT_BANK_NAME: &str = "default";

/// Status of the abstraction pipeline workers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStatus {
    /// Whether the pipeline is enabled
    pub enabled: bool,
    /// Whether workers are running
    pub workers_running: bool,
    /// Number of L0 memories waiting for L1 abstraction
    pub pending_l0_count: usize,
    /// Number of L1 memories waiting for L2 abstraction
    pub pending_l1_count: usize,
    /// Number of L2 memories waiting for L3 abstraction
    pub pending_l2_count: usize,
    /// Configuration settings
    pub config: PipelineConfigStatus,
}

/// Configuration status for the abstraction pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfigStatus {
    /// Minimum L0 memories before creating L1
    pub min_memories_for_l1: usize,
    /// Delay between L0→L1 processing cycles
    pub l1_processing_delay_secs: u64,
    /// Maximum concurrent abstraction tasks
    pub max_concurrent_tasks: usize,
}

/// Result of triggering abstraction processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractionTriggerResult {
    pub l0_to_l1_created: usize,
    pub l1_to_l2_created: usize,
    pub l2_to_l3_created: usize,
    pub errors: Vec<String>,
}

/// Metadata written alongside each backup file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupManifest {
    /// Monotonically increasing version counter for this bank's backups.
    pub version: u64,
    /// ISO 8601 timestamp of when the backup was created.
    pub created_at: String,
    /// Name of the bank that was backed up.
    pub bank_name: String,
    /// Number of memories in the bank at backup time.
    pub memory_count: usize,
    /// SHA-256 hex digest of the `.db` file.
    pub sha256: String,
    /// Original database file size in bytes.
    pub size_bytes: u64,
}

/// Result of a merge-restore operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    /// How many memories were imported from the backup.
    pub imported: usize,
    /// How many were skipped because they already existed (same content hash).
    pub skipped_duplicates: usize,
    /// Total memories in the bank after the merge.
    pub total_after_merge: usize,
}

/// Information about a memory bank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBankInfo {
    /// Bank name (used as identifier in API calls).
    pub name: String,
    /// Absolute path to the bank's database file.
    pub path: String,
    /// Number of memories currently stored.
    pub memory_count: usize,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Whether this bank is currently loaded in memory.
    pub loaded: bool,
}

/// Manages multiple named memory banks.
///
/// Each bank is an isolated memory store with its own persistence file.
/// Banks share a single LLM client for efficiency.
pub struct MemoryBankManager {
    /// Loaded banks: name → MemoryManager
    banks: RwLock<HashMap<String, Arc<MemoryManager>>>,
    /// Document session managers: name → DocumentSessionManager
    session_managers: RwLock<HashMap<String, Arc<DocumentSessionManager>>>,
    /// Shared LLM client (cloned per bank)
    llm_client: Box<dyn LLMClient>,
    /// Memory processing config (shared across banks)
    memory_config: MemoryConfig,
    /// Vector store config template
    store_config: VectorStoreConfig,
    /// Directory where bank `.db` files are stored
    banks_dir: PathBuf,
    /// Bank descriptions (persisted in a metadata file)
    descriptions: RwLock<HashMap<String, String>>,
    /// Abstraction pipeline for progressive layer creation (shared across banks)
    abstraction_pipeline: Arc<Mutex<Option<Arc<AbstractionPipeline>>>>,
}

impl MemoryBankManager {
    /// Create a new bank manager.
    ///
    /// - `banks_dir` — directory for bank `.db` files (created if missing)
    /// - `llm_client` — shared LLM client (cloned per bank)
    /// - `store_config` — vector store config template
    /// - `memory_config` — memory processing config
    pub fn new(
        banks_dir: PathBuf,
        llm_client: Box<dyn LLMClient>,
        store_config: VectorStoreConfig,
        memory_config: MemoryConfig,
    ) -> Result<Self> {
        // Create banks directory
        std::fs::create_dir_all(&banks_dir).map_err(|e| {
            MemoryError::config(format!(
                "Failed to create banks directory '{}': {}",
                banks_dir.display(),
                e
            ))
        })?;

        // Load descriptions from metadata file
        let initial_descriptions = {
            let meta_path = banks_dir.join("banks.json");
            if meta_path.exists() {
                std::fs::read_to_string(&meta_path)
                    .ok()
                    .and_then(|data| serde_json::from_str::<HashMap<String, String>>(&data).ok())
                    .unwrap_or_default()
            } else {
                HashMap::new()
            }
        };

        let manager = Self {
            banks: RwLock::new(HashMap::new()),
            session_managers: RwLock::new(HashMap::new()),
            llm_client,
            memory_config,
            store_config,
            banks_dir: banks_dir.clone(),
            descriptions: RwLock::new(initial_descriptions),
            abstraction_pipeline: Arc::new(Mutex::new(None)),
        };

        info!(
            "Memory bank manager initialized (dir: {})",
            banks_dir.display()
        );
        Ok(manager)
    }

    /// Initialize and start the abstraction pipeline workers.
    ///
    /// This should be called after the MemoryBankManager is fully initialized
    /// and at least one bank has been loaded.
    pub async fn start_abstraction_pipeline(&self) -> Result<()> {
        // Check if already started
        {
            let pipeline_guard = self.abstraction_pipeline.lock().await;
            if pipeline_guard.is_some() {
                info!("Abstraction pipeline already started");
                return Ok(());
            }
        }

        // Get the default bank's memory manager to create the pipeline
        let default_bank = self.default_bank().await?;

        // Create abstraction pipeline config
        let config = AbstractionConfig {
            enabled: self.memory_config.auto_enhance,
            min_memories_for_l1: 5,
            l1_processing_delay: std::time::Duration::from_secs(30),
            max_concurrent_tasks: 3,
        };

        info!(
            "Starting abstraction pipeline (enabled={}, min_l0_for_l1={}, delay={}s)",
            config.enabled, config.min_memories_for_l1, config.l1_processing_delay.as_secs()
        );

        // Create the pipeline
        let pipeline = Arc::new(AbstractionPipeline::new(default_bank, config.clone()));

        // Start background workers
        if config.enabled {
            let _l0_to_l1_handle = pipeline.clone().start_l0_to_l1_worker();
            let _l1_to_l2_handle = pipeline.clone().start_l1_to_l2_worker();
            let _l2_to_l3_handle = pipeline.clone().start_l2_to_l3_worker();
            info!("Abstraction pipeline workers started (L0→L1, L1→L2, L2→L3)");
        } else {
            info!("Abstraction pipeline created but disabled (auto_enhance=false)");
        }

        // Store the pipeline
        {
            let mut pipeline_guard = self.abstraction_pipeline.lock().await;
            *pipeline_guard = Some(pipeline);
        }

        Ok(())
    }

    /// Get the current pipeline status
    pub async fn get_pipeline_status(&self) -> PipelineStatus {
        let config = AbstractionConfig {
            enabled: self.memory_config.auto_enhance,
            min_memories_for_l1: 5,
            l1_processing_delay: std::time::Duration::from_secs(30),
            max_concurrent_tasks: 3,
        };

        // Check if pipeline is running
        let workers_running = {
            let pipeline_guard = self.abstraction_pipeline.lock().await;
            pipeline_guard.is_some()
        };

        // Count pending memories at each layer (from all banks)
        let mut pending_l0_count = 0;
        let mut pending_l1_count = 0;
        let mut pending_l2_count = 0;

        if let Ok(banks) = self.list_banks().await {
            for bank_info in &banks {
                if let Ok(bank) = self.get_or_create(&bank_info.name).await {
                    // Count L0 memories
                    if let Ok(l0_memories) = bank.list(&{
                        let mut f = Filters::new();
                        f.custom.insert("layer.level".to_string(), serde_json::json!(0));
                        f
                    }, None).await {
                        pending_l0_count += l0_memories.len();
                    }
                    // Count L1 memories
                    if let Ok(l1_memories) = bank.list(&{
                        let mut f = Filters::new();
                        f.custom.insert("layer.level".to_string(), serde_json::json!(1));
                        f
                    }, None).await {
                        pending_l1_count += l1_memories.len();
                    }
                    // Count L2 memories
                    if let Ok(l2_memories) = bank.list(&{
                        let mut f = Filters::new();
                        f.custom.insert("layer.level".to_string(), serde_json::json!(2));
                        f
                    }, None).await {
                        pending_l2_count += l2_memories.len();
                    }
                }
            }
        }

        PipelineStatus {
            enabled: config.enabled,
            workers_running,
            pending_l0_count,
            pending_l1_count,
            pending_l2_count,
            config: PipelineConfigStatus {
                min_memories_for_l1: config.min_memories_for_l1,
                l1_processing_delay_secs: config.l1_processing_delay.as_secs(),
                max_concurrent_tasks: config.max_concurrent_tasks,
            },
        }
    }

    /// Start the abstraction pipeline manually (even if auto_enhance is false)
    pub async fn start_pipeline_manual(&self) -> Result<String> {
        // Check if already running
        {
            let pipeline_guard = self.abstraction_pipeline.lock().await;
            if pipeline_guard.is_some() {
                // If it's already there, just ensure workers are "started" (they already are if Some)
                return Ok("Abstraction pipeline is already running and active.".to_string());
            }
        }

        // Get the default bank's memory manager
        let default_bank = self.default_bank().await?;

        // Create config (always enabled for manual start)
        let config = AbstractionConfig {
            enabled: true,
            min_memories_for_l1: 5,
            l1_processing_delay: std::time::Duration::from_secs(30),
            max_concurrent_tasks: 3,
        };

        let pipeline = Arc::new(AbstractionPipeline::new(default_bank, config.clone()));

        // Start workers
        let _l0_to_l1_handle = pipeline.clone().start_l0_to_l1_worker();
        let _l1_to_l2_handle = pipeline.clone().start_l1_to_l2_worker();
        let _l2_to_l3_handle = pipeline.clone().start_l2_to_l3_worker();

        // Store the pipeline
        {
            let mut pipeline_guard = self.abstraction_pipeline.lock().await;
            *pipeline_guard = Some(pipeline);
        }

        Ok("Abstraction pipeline started successfully. Workers: L0→L1, L1→L2, L2→L3".to_string())
    }

    /// Stop the abstraction pipeline
    pub async fn stop_pipeline(&self) -> Result<String> {
        let pipeline_guard = self.abstraction_pipeline.lock().await;
        if let Some(pipeline) = pipeline_guard.as_ref() {
            // Send shutdown signal
            let _ = pipeline.get_shutdown_sender().send(());
            drop(pipeline_guard);
            
            // Clear the pipeline
            {
                let mut pipeline_guard = self.abstraction_pipeline.lock().await;
                *pipeline_guard = None;
            }
            
            Ok("Abstraction pipeline stopped. Workers will finish current tasks and shut down.".to_string())
        } else {
            Ok("Abstraction pipeline is not running".to_string())
        }
    }

    /// Trigger immediate abstraction processing (one-shot, doesn't start workers)
    pub async fn trigger_abstraction_now(&self, target_layer: Option<i32>) -> Result<AbstractionTriggerResult> {
        // Get existing pipeline or create a temporary one for this one-shot trigger
        let (pipeline, is_temp) = {
            let pipeline_guard = self.abstraction_pipeline.lock().await;
            if let Some(p) = pipeline_guard.as_ref() {
                (Arc::clone(p), false)
            } else {
                // Create a temporary pipeline using the default bank
                let default_bank = self.default_bank().await?;
                let config = AbstractionConfig {
                    enabled: true, // Force enabled for one-shot
                    ..Default::default()
                };
                (Arc::new(AbstractionPipeline::new(default_bank, config)), true)
            }
        };
        
        let mut result = AbstractionTriggerResult {
            l0_to_l1_created: 0,
            l1_to_l2_created: 0,
            l2_to_l3_created: 0,
            errors: vec![],
        };

        let target = target_layer.unwrap_or(1);
        
        // Process specific layer or all
        if target == 1 || target == 0 {
            // L0 → L1
            let pending = pipeline.find_pending_l0_abstractions().await.unwrap_or_default();
            for memory_id in pending {
                match pipeline.create_l1_abstraction(memory_id).await {
                    Ok(_) => result.l0_to_l1_created += 1,
                    Err(e) => result.errors.push(format!("L0→L1 failed for {}: {}", memory_id, e)),
                }
            }
        }

        if target == 2 || target == 0 {
            // L1 → L2 (simplified - would need implementation in pipeline)
            result.l1_to_l2_created = 0; // TODO: Implement in pipeline
        }

        if target == 3 || target == 0 {
            // L2 → L3 (simplified - would need implementation in pipeline)
            result.l2_to_l3_created = 0; // TODO: Implement in pipeline
        }

        if is_temp {
            info!("One-shot abstraction complete (temporary pipeline)");
        } else {
            info!("One-shot abstraction complete (reusing active pipeline)");
        }

        Ok(result)
    }

    /// Get or create a memory bank by name.
    ///
    /// If the bank is already loaded, returns it immediately.
    /// If not loaded but the `.db` file exists on disk, loads it.
    /// If neither, creates a new empty bank.
    pub async fn get_or_create(&self, name: &str) -> Result<Arc<MemoryManager>> {
        let sanitized = Self::sanitize_name(name)?;

        // Fast path: already loaded
        {
            let banks = self.banks.read().await;
            if let Some(manager) = banks.get(&sanitized) {
                return Ok(Arc::clone(manager));
            }
        }

        // Slow path: create/load + insert
        let manager = self.create_bank_manager(&sanitized)?;
        let manager = Arc::new(manager);

        let mut banks = self.banks.write().await;
        // Double-check after acquiring write lock
        if let Some(existing) = banks.get(&sanitized) {
            return Ok(Arc::clone(existing));
        }
        banks.insert(sanitized.clone(), Arc::clone(&manager));
        info!("Memory bank '{}' loaded", sanitized);
        Ok(manager)
    }

    /// Get the default bank.
    pub async fn default_bank(&self) -> Result<Arc<MemoryManager>> {
        self.get_or_create(DEFAULT_BANK_NAME).await
    }

    /// Resolve the bank from an optional name (defaults to "default").
    pub async fn resolve_bank(&self, bank_name: Option<&str>) -> Result<Arc<MemoryManager>> {
        match bank_name {
            Some(name) if !name.is_empty() => self.get_or_create(name).await,
            _ => self.default_bank().await,
        }
    }

    /// Resolve both the MemoryManager and DocumentSessionManager for a bank.
    ///
    /// This is used for document ingestion operations that need both
    /// the memory store and the session state manager.
    pub async fn resolve_bank_with_sessions(
        &self,
        bank_name: Option<&str>,
    ) -> Result<(Arc<MemoryManager>, Arc<DocumentSessionManager>)> {
        let name = match bank_name {
            Some(n) if !n.is_empty() => Self::sanitize_name(n)?,
            _ => DEFAULT_BANK_NAME.to_string(),
        };

        let manager = self.get_or_create(&name).await?;
        let session_manager = self.get_or_create_session_manager(&name).await?;

        Ok((manager, session_manager))
    }

    /// Get or create a DocumentSessionManager for a bank.
    async fn get_or_create_session_manager(&self, name: &str) -> Result<Arc<DocumentSessionManager>> {
        // Fast path: already loaded
        {
            let managers = self.session_managers.read().await;
            if let Some(manager) = managers.get(name) {
                return Ok(Arc::clone(manager));
            }
        }

        // Slow path: create + insert
        let session_db_path = self.session_manager_path(name);
        let session_manager = DocumentSessionManager::new(session_db_path, None)?;
        let session_manager = Arc::new(session_manager);

        let mut managers = self.session_managers.write().await;
        // Double-check after acquiring write lock
        if let Some(existing) = managers.get(name) {
            return Ok(Arc::clone(existing));
        }
        managers.insert(name.to_string(), Arc::clone(&session_manager));
        info!("Document session manager initialized for bank '{}'", name);
        Ok(session_manager)
    }

    /// Compute the session database path for a bank.
    fn session_manager_path(&self, name: &str) -> PathBuf {
        self.banks_dir.join(format!("{}.sessions.db", name))
    }

    /// Create a new bank explicitly, with an optional description.
    ///
    /// Returns info about the created bank. If the bank already exists,
    /// returns its info without modification.
    pub async fn create_bank(
        &self,
        name: &str,
        description: Option<String>,
    ) -> Result<MemoryBankInfo> {
        let sanitized = Self::sanitize_name(name)?;
        let manager = self.get_or_create(&sanitized).await?;

        // Store description
        if let Some(desc) = &description {
            let mut descs = self.descriptions.write().await;
            descs.insert(sanitized.clone(), desc.clone());
            drop(descs);
            self.persist_descriptions().await;
        }

        let db_path = self.bank_path(&sanitized);
        let count = manager
            .list(&Filters::default(), None)
            .await
            .map(|v| v.len())
            .unwrap_or(0);

        Ok(MemoryBankInfo {
            name: sanitized,
            path: db_path.display().to_string(),
            memory_count: count,
            description,
            loaded: true,
        })
    }

    /// List all known memory banks.
    ///
    /// Discovers banks from both loaded instances and `.db` files on disk.
    pub async fn list_banks(&self) -> Result<Vec<MemoryBankInfo>> {
        let mut bank_names: Vec<String> = Vec::new();

        // Collect from loaded banks
        {
            let banks = self.banks.read().await;
            for name in banks.keys() {
                if !bank_names.contains(name) {
                    bank_names.push(name.clone());
                }
            }
        }

        // Discover from disk
        if let Ok(entries) = std::fs::read_dir(&self.banks_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "db") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        let name = stem.to_string();
                        if !bank_names.contains(&name) {
                            bank_names.push(name);
                        }
                    }
                }
            }
        }

        // Discover from descriptions metadata (banks may exist in banks.json
        // even if their .db file hasn't been created yet)
        {
            let descriptions = self.descriptions.read().await;
            for name in descriptions.keys() {
                if !bank_names.contains(name) {
                    bank_names.push(name.clone());
                }
            }
        }

        // Ensure "default" is always present
        if !bank_names.contains(&DEFAULT_BANK_NAME.to_string()) {
            bank_names.push(DEFAULT_BANK_NAME.to_string());
        }

        bank_names.sort();

        let loaded_banks = self.banks.read().await;
        let descriptions = self.descriptions.read().await;

        let mut infos = Vec::new();
        for name in &bank_names {
            let db_path = self.bank_path(name);
            let is_loaded = loaded_banks.contains_key(name);

            let memory_count = if let Some(manager) = loaded_banks.get(name) {
                manager
                    .list(&Filters::default(), None)
                    .await
                    .map(|v| v.len())
                    .unwrap_or(0)
            } else {
                0 // Not loaded — we don't load just to count
            };

            infos.push(MemoryBankInfo {
                name: name.clone(),
                path: db_path.display().to_string(),
                memory_count,
                description: descriptions.get(name).cloned(),
                loaded: is_loaded,
            });
        }

        Ok(infos)
    }

    /// Delete a memory bank.
    ///
    /// Removes it from loaded memory, deletes the database file, and updates metadata.
    /// Returns true if the bank existed (loaded, described, or file on disk) and was deleted.
    ///
    /// Note: Deleting the 'default' bank is allowed; it will be recreated empty on next use.
    pub async fn delete_bank(&self, name: &str) -> Result<bool> {
        // Sanitize name first (though we might want to be lenient if trying to cleanup weird files)
        // But let's stick to safe names for now.
        let sanitized = Self::sanitize_name(name)?;

        // 1. Remove from loaded banks (stops new operations)
        {
            let mut banks = self.banks.write().await;
            banks.remove(&sanitized);
        }

        // 2. Remove description
        let mut had_desc = false;
        {
            let mut descs = self.descriptions.write().await;
            if descs.remove(&sanitized).is_some() {
                had_desc = true;
            }
        }
        if had_desc {
            self.persist_descriptions().await;
        }

        // 3. Delete physical file
        let db_path = self.bank_path(&sanitized);
        let file_existed = if db_path.exists() {
            tokio::fs::remove_file(&db_path).await.map_err(|e| {
                MemoryError::VectorLite(format!(
                    "Failed to delete bank file '{}': {}",
                    db_path.display(),
                    e
                ))
            })?;
            true
        } else {
            false
        };

        if had_desc || file_existed {
            info!("Deleted memory bank: {}", sanitized);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Back up a bank's database file to a destination directory.
    ///
    /// Creates a versioned backup: `<bank>_v<N>_<timestamp>.db` plus a
    /// `.manifest.json` sidecar containing the version, timestamp, memory
    /// count, and SHA-256 checksum of the database file.
    ///
    /// Returns `(backup_db_path, manifest)`.
    pub async fn backup_bank(
        &self,
        name: &str,
        dest_dir: &Path,
    ) -> Result<(PathBuf, BackupManifest)> {
        let sanitized = Self::sanitize_name(name)?;
        let src = self.bank_path(&sanitized);

        if !src.exists() {
            return Err(MemoryError::config(format!(
                "Bank '{}' has no database file ({})",
                sanitized,
                src.display()
            )));
        }

        // Ensure destination directory exists
        tokio::fs::create_dir_all(dest_dir).await.map_err(|e| {
            MemoryError::config(format!(
                "Failed to create backup directory '{}': {}",
                dest_dir.display(),
                e
            ))
        })?;

        // Determine next version number by scanning existing manifests
        let version = Self::next_backup_version(dest_dir, &sanitized).await;

        let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%S").to_string();
        let stem = format!("{}_v{}_{}", sanitized, version, timestamp);
        let dest_file = dest_dir.join(format!("{}.db", stem));

        // Copy the database file
        tokio::fs::copy(&src, &dest_file).await.map_err(|e| {
            MemoryError::config(format!(
                "Failed to copy bank file to '{}': {}",
                dest_file.display(),
                e
            ))
        })?;

        // Compute checksum of the copied file
        let sha256 = Self::file_sha256(&dest_file).await?;
        let file_size = tokio::fs::metadata(&dest_file)
            .await
            .map(|m| m.len())
            .unwrap_or(0);

        // Count memories (if the bank is loaded)
        let memory_count = {
            let banks = self.banks.read().await;
            if let Some(mgr) = banks.get(&sanitized) {
                mgr.list(&Filters::default(), None)
                    .await
                    .map(|v| v.len())
                    .unwrap_or(0)
            } else {
                0
            }
        };

        let manifest = BackupManifest {
            version,
            created_at: chrono::Utc::now().to_rfc3339(),
            bank_name: sanitized.clone(),
            memory_count,
            sha256,
            size_bytes: file_size,
        };

        // Write the manifest sidecar
        let manifest_path = dest_dir.join(format!("{}.manifest.json", stem));
        let manifest_json = serde_json::to_string_pretty(&manifest).map_err(|e| {
            MemoryError::config(format!("Failed to serialize backup manifest: {}", e))
        })?;
        tokio::fs::write(&manifest_path, manifest_json)
            .await
            .map_err(|e| {
                MemoryError::config(format!(
                    "Failed to write manifest to '{}': {}",
                    manifest_path.display(),
                    e
                ))
            })?;

        info!(
            "Backed up bank '{}' v{} to {} (sha256: {})",
            sanitized,
            version,
            dest_file.display(),
            &manifest.sha256[..16],
        );
        Ok((dest_file, manifest))
    }

    /// Restore a bank from a backup `.db` file (replace mode).
    ///
    /// Unloads the bank if currently loaded, verifies the backup's integrity
    /// if a `.manifest.json` sidecar exists, then copies the backup file into the
    /// banks directory. The bank will be lazily re-loaded on next access.
    ///
    /// Returns the path of the restored database file.
    pub async fn restore_bank(&self, name: &str, source_file: &Path) -> Result<PathBuf> {
        let sanitized = Self::sanitize_name(name)?;
        Self::validate_source_file(source_file)?;

        // If a manifest sidecar exists, verify checksum
        Self::verify_backup_integrity(source_file).await?;

        // Unload the bank if currently loaded so the old db handle is dropped
        {
            let mut banks = self.banks.write().await;
            if banks.remove(&sanitized).is_some() {
                info!("Unloaded bank '{}' before restore", sanitized);
            }
        }

        let dest = self.bank_path(&sanitized);
        tokio::fs::copy(source_file, &dest).await.map_err(|e| {
            MemoryError::config(format!(
                "Failed to copy backup file to '{}': {}",
                dest.display(),
                e
            ))
        })?;

        info!(
            "Restored bank '{}' from {} (replace mode)",
            sanitized,
            source_file.display()
        );
        Ok(dest)
    }

    /// Merge-restore: import memories from a backup file into the current bank.
    ///
    /// Opens the backup `.db` as a read-only store, lists all memories, and
    /// inserts those whose content hash is not already present in the target bank.
    /// This is additive — existing data is never deleted or overwritten.
    ///
    /// Returns a `MergeResult` with import/skip counts.
    pub async fn merge_from_backup(&self, name: &str, source_file: &Path) -> Result<MergeResult> {
        let sanitized = Self::sanitize_name(name)?;
        Self::validate_source_file(source_file)?;

        // Verify integrity if manifest exists
        Self::verify_backup_integrity(source_file).await?;

        // Open the backup db as a temporary read-only VectorLiteStore
        let backup_store = VectorLiteStore::with_config(VectorLiteConfig {
            collection_name: format!("backup-import-{}", sanitized),
            persistence_path: Some(source_file.to_path_buf()),
            ..VectorLiteConfig::from_store_config(&self.store_config)
        })?;

        let backup_memories = backup_store.list(&Filters::default(), None).await?;

        // Get or create the target bank
        let target = self.get_or_create(&sanitized).await?;

        // Collect existing content hashes for fast dedup
        let existing_memories = target.list(&Filters::default(), None).await?;
        let existing_hashes: std::collections::HashSet<String> = existing_memories
            .iter()
            .map(|m| m.metadata.hash.clone())
            .collect();

        let mut imported = 0usize;
        let mut skipped = 0usize;

        for memory in &backup_memories {
            if existing_hashes.contains(&memory.metadata.hash) {
                skipped += 1;
                continue;
            }
            // Import the memory directly into the target bank's store
            target.import_memory(memory).await?;
            imported += 1;
        }

        let total = target.list(&Filters::default(), None).await?.len();

        info!(
            "Merge into bank '{}': imported {}, skipped {} duplicates, total now {}",
            sanitized, imported, skipped, total
        );

        Ok(MergeResult {
            imported,
            skipped_duplicates: skipped,
            total_after_merge: total,
        })
    }

    /// List available backup manifests for a bank in a directory.
    pub async fn list_backups(dest_dir: &Path, bank_name: &str) -> Result<Vec<BackupManifest>> {
        let sanitized = Self::sanitize_name(bank_name)?;
        let prefix = format!("{}_v", sanitized);
        let mut manifests = Vec::new();

        if let Ok(mut entries) = tokio::fs::read_dir(dest_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with(&prefix) && name_str.ends_with(".manifest.json") {
                    if let Ok(data) = tokio::fs::read_to_string(entry.path()).await {
                        if let Ok(manifest) = serde_json::from_str::<BackupManifest>(&data) {
                            manifests.push(manifest);
                        }
                    }
                }
            }
        }

        manifests.sort_by_key(|m| m.version);
        Ok(manifests)
    }

    /// Get the LLM client status (shared across banks).
    pub fn get_llm_status(&self) -> crate::llm::ClientStatus {
        self.llm_client.get_status()
    }

    /// Get the banks directory path.
    pub fn banks_dir(&self) -> &Path {
        &self.banks_dir
    }

    /// List all active document sessions across all banks
    pub async fn list_all_active_sessions(&self) -> Result<Vec<crate::document_session::DocumentSession>> {
        let managers = self.session_managers.read().await;
        let mut all_sessions = Vec::new();
        for manager in managers.values() {
            if let Ok(sessions) = manager.list_active_sessions() {
                all_sessions.extend(sessions);
            }
        }
        Ok(all_sessions)
    }

    /// Mark all document sessions that have been 'processing' too long as failed.
    pub async fn cleanup_stalled_sessions(&self, timeout_seconds: u64) -> Result<usize> {
        let managers = self.session_managers.read().await;
        let mut total_failed = 0;
        for manager in managers.values() {
            if let Ok(count) = manager.fail_stalled_sessions(timeout_seconds) {
                total_failed += count;
            }
        }
        Ok(total_failed)
    }

    // ── Internal ────────────────────────────────────────────────────────

    /// Compute the database file path for a bank.
    fn bank_path(&self, name: &str) -> PathBuf {
        self.banks_dir.join(format!("{}.db", name))
    }

    /// Validate that a source file exists and is a regular file.
    fn validate_source_file(source_file: &Path) -> Result<()> {
        if !source_file.exists() {
            return Err(MemoryError::config(format!(
                "Source backup file does not exist: {}",
                source_file.display()
            )));
        }
        if !source_file.is_file() {
            return Err(MemoryError::config(format!(
                "Source path is not a file: {}",
                source_file.display()
            )));
        }
        Ok(())
    }

    /// Compute SHA-256 hex digest of a file.
    async fn file_sha256(path: &Path) -> Result<String> {
        let data = tokio::fs::read(path).await.map_err(|e| {
            MemoryError::config(format!(
                "Failed to read file for checksum '{}': {}",
                path.display(),
                e
            ))
        })?;
        let mut hasher = Sha256::new();
        hasher.update(&data);
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// If a `.manifest.json` sidecar exists for `source_file`, verify the
    /// SHA-256 checksum matches. If no manifest exists, skip verification.
    async fn verify_backup_integrity(source_file: &Path) -> Result<()> {
        // Derive the manifest path from the db path:
        // e.g. bank_v1_20260215T120000.db → bank_v1_20260215T120000.manifest.json
        let manifest_path = source_file.with_extension("manifest.json");
        if !manifest_path.exists() {
            // Legacy backup without manifest — skip verification
            return Ok(());
        }

        let manifest_data = tokio::fs::read_to_string(&manifest_path)
            .await
            .map_err(|e| {
                MemoryError::config(format!(
                    "Failed to read manifest '{}': {}",
                    manifest_path.display(),
                    e
                ))
            })?;
        let manifest: BackupManifest = serde_json::from_str(&manifest_data)
            .map_err(|e| MemoryError::config(format!("Invalid backup manifest: {}", e)))?;

        let actual_sha256 = Self::file_sha256(source_file).await?;
        if actual_sha256 != manifest.sha256 {
            return Err(MemoryError::config(format!(
                "Backup integrity check failed! Expected SHA-256 {} but got {}. \
                 The backup file may be corrupted.",
                manifest.sha256, actual_sha256
            )));
        }

        info!(
            "Backup integrity verified (v{}, sha256: {})",
            manifest.version,
            &manifest.sha256[..16]
        );
        Ok(())
    }

    /// Determine the next version number for backups of a given bank.
    async fn next_backup_version(dest_dir: &Path, bank_name: &str) -> u64 {
        let manifests = Self::list_backups(dest_dir, bank_name)
            .await
            .unwrap_or_default();
        manifests.iter().map(|m| m.version).max().unwrap_or(0) + 1
    }

    /// Create a new MemoryManager for a bank.
    fn create_bank_manager(&self, name: &str) -> Result<MemoryManager> {
        let db_path = self.bank_path(name);

        let vl_config = VectorLiteConfig {
            collection_name: format!("bank-{}", name),
            persistence_path: Some(db_path.clone()),
            ..VectorLiteConfig::from_store_config(&self.store_config)
        };

        let store_result = VectorLiteStore::with_config(vl_config.clone());
        
        let store = match store_result {
            Ok(s) => Box::new(s),
            Err(e) => {
                // Check if it's a corruption error (like the UTF-8 error)
                let err_msg = e.to_string();
                if err_msg.contains("UTF-8") || err_msg.contains("load collection") {
                    warn!("Memory bank '{}' appears to be corrupted: {}. Moving to .corrupted and starting fresh.", name, err_msg);
                    
                    let corrupted_path = db_path.with_extension("db.corrupted");
                    if let Err(move_err) = std::fs::rename(&db_path, &corrupted_path) {
                        error!("Failed to move corrupted bank file: {}", move_err);
                        return Err(e); // If we can't move it, we still fail
                    }
                    
                    // Try again with a fresh store
                    Box::new(VectorLiteStore::with_config(vl_config).map_err(|retry_err| {
                        error!("Failed to create fresh bank after corruption: {}", retry_err);
                        retry_err
                    })?)
                } else {
                    return Err(e);
                }
            }
        };

        let client = dyn_clone::clone_box(self.llm_client.as_ref());

        Ok(MemoryManager::new(
            store,
            client,
            self.memory_config.clone(),
        ))
    }

    /// Validate and sanitize a bank name.
    ///
    /// Rules: alphanumeric + hyphens + underscores, 1–64 chars, no path separators.
    fn sanitize_name(name: &str) -> Result<String> {
        let trimmed = name.trim();

        if trimmed.is_empty() {
            return Err(MemoryError::config("Bank name cannot be empty"));
        }

        if trimmed.len() > 64 {
            return Err(MemoryError::config(
                "Bank name too long (max 64 characters)",
            ));
        }

        // Only allow safe filesystem characters
        let valid = trimmed
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_');

        if !valid {
            return Err(MemoryError::config(format!(
                "Bank name '{}' contains invalid characters. \
                 Only alphanumeric, hyphens, and underscores are allowed.",
                trimmed
            )));
        }

        Ok(trimmed.to_string())
    }

    /// Persist bank descriptions to `banks.json`.
    async fn persist_descriptions(&self) {
        let descs = self.descriptions.read().await;
        let meta_path = self.banks_dir.join("banks.json");

        match serde_json::to_string_pretty(&*descs) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&meta_path, json) {
                    warn!("Failed to persist bank descriptions: {}", e);
                }
            }
            Err(e) => {
                warn!("Failed to serialize bank descriptions: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_name_valid() {
        assert_eq!(
            MemoryBankManager::sanitize_name("my-project").unwrap(),
            "my-project"
        );
        assert_eq!(
            MemoryBankManager::sanitize_name("project_123").unwrap(),
            "project_123"
        );
        assert_eq!(
            MemoryBankManager::sanitize_name("default").unwrap(),
            "default"
        );
        assert_eq!(
            MemoryBankManager::sanitize_name("  trimmed  ").unwrap(),
            "trimmed"
        );
    }

    #[test]
    fn test_sanitize_name_invalid_chars() {
        assert!(MemoryBankManager::sanitize_name("path/traversal").is_err());
        assert!(MemoryBankManager::sanitize_name("../escape").is_err());
        assert!(MemoryBankManager::sanitize_name("has spaces").is_err());
        assert!(MemoryBankManager::sanitize_name("special!chars").is_err());
        assert!(MemoryBankManager::sanitize_name("dot.name").is_err());
    }

    #[test]
    fn test_sanitize_name_empty() {
        assert!(MemoryBankManager::sanitize_name("").is_err());
        assert!(MemoryBankManager::sanitize_name("   ").is_err());
    }

    #[test]
    fn test_sanitize_name_too_long() {
        let long = "a".repeat(65);
        assert!(MemoryBankManager::sanitize_name(&long).is_err());

        let ok = "a".repeat(64);
        assert!(MemoryBankManager::sanitize_name(&ok).is_ok());
    }

    #[test]
    fn test_bank_path() {
        let banks_dir = PathBuf::from("/tmp/test-banks");
        // We can't create a full MemoryBankManager without an LLM client,
        // so just test the path logic directly.
        let path = banks_dir.join(format!("{}.db", "my-project"));
        assert_eq!(path, PathBuf::from("/tmp/test-banks/my-project.db"));
    }
}
