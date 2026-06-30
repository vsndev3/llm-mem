//! Cross-process advisory file lock for serializing database writes.
//!
//! Unlike the previous process-lifetime exclusive lock, this module provides
//! **operation-scoped write locking**: the lock is acquired only for the
//! duration of a database mutation (insert, update, delete, compact, prune)
//! and released immediately after.
//!
//! ## Design
//!
//! - **Writes**: must acquire the exclusive lock before touching LanceDB or
//!   SQLite. Only one process writes at a time.
//! - **Reads**: no lock needed. LanceDB (Lance format) and WAL-mode SQLite
//!   both support concurrent readers natively.
//! - **In-process serialization**: a `std::sync::Mutex` prevents multiple
//!   threads within the same process from contending on the same file lock
//!   simultaneously (the flock itself does not block same-fd re-acquisition).
//!
//! ## Usage
//!
//! ```rust,ignore
//! let lock_mgr = InstanceLockManager::open(&banks_dir)?;
//!
//! // Async write:
//! let _guard = lock_mgr.acquire_write().await?;
//! store.insert(&memory).await?;
//! // guard drops -> lock released
//!
//! // Sync write (for SQLite-backed managers):
//! let _guard = lock_mgr.acquire_write_sync()?;
//! conn.execute("INSERT ...", params![])?;
//! ```

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tracing::{debug, error, info, warn};

use crate::error::{MemoryError, Result};

pub struct InstanceLockManager {
    file: Arc<File>,
    local: Arc<Mutex<()>>,
    #[allow(dead_code)]
    lock_path: PathBuf,
}

pub struct WriteGuard {
    file: Arc<File>,
}

impl Drop for WriteGuard {
    fn drop(&mut self) {
        if let Err(e) = fs2::FileExt::unlock(self.file.as_ref()) {
            warn!("Failed to release instance write lock: {}", e);
        }
        debug!("Instance write lock released");
    }
}

impl InstanceLockManager {
    /// Open (or create) the lock file in the banks directory.
    ///
    /// The file descriptor is held for the process lifetime so the lock
    /// remains valid across acquire/release cycles. The lock itself is
    /// NOT acquired here — use `acquire_write` or `acquire_write_sync`.
    pub fn open(banks_dir: &Path) -> Result<Self> {
        if !banks_dir.exists() {
            std::fs::create_dir_all(banks_dir).map_err(|e| {
                MemoryError::config(format!(
                    "Failed to create banks directory '{}': {}",
                    banks_dir.display(),
                    e
                ))
            })?;
        }

        let lock_path = banks_dir.join(".llm-mem.lock");
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|e| {
                MemoryError::config(format!(
                    "Failed to open lock file '{}': {}",
                    lock_path.display(),
                    e
                ))
            })?;

        info!(
            "Instance lock manager opened (lock file: {})",
            lock_path.display()
        );

        Ok(Self {
            file: Arc::new(file),
            local: Arc::new(Mutex::new(())),
            lock_path,
        })
    }

    /// Acquire the exclusive cross-process write lock (async).
    ///
    /// Blocks until the lock is available. The lock is released when the
    /// returned `WriteGuard` is dropped.
    ///
    /// Uses `spawn_blocking` internally so the async executor is not blocked
    /// while waiting for another process to release the lock.
    pub async fn acquire_write(&self) -> Result<WriteGuard> {
        let file = self.file.clone();
        let local = self.local.clone();

        tokio::task::spawn_blocking(move || {
            let _local = local.lock().map_err(|e| {
                MemoryError::config(format!("Failed to acquire local instance lock: {}", e))
            })?;
            fs2::FileExt::lock_exclusive(file.as_ref()).map_err(|e| {
                MemoryError::config(format!(
                    "Failed to acquire cross-process instance lock: {}",
                    e
                ))
            })?;
            debug!("Instance write lock acquired (async)");
            Ok(WriteGuard {
                file: file.clone(),
            })
        })
        .await
        .map_err(|e| {
            MemoryError::config(format!("Instance lock task panicked: {}", e))
        })?
    }

    /// Acquire the exclusive cross-process write lock (synchronous).
    ///
    /// For use in synchronous code paths (e.g. SQLite-backed managers).
    /// Blocks the calling thread until the lock is available.
    pub fn acquire_write_sync(&self) -> Result<WriteGuard> {
        let _local = self.local.lock().map_err(|e| {
            MemoryError::config(format!("Failed to acquire local instance lock: {}", e))
        })?;
        fs2::FileExt::lock_exclusive(self.file.as_ref()).map_err(|e| {
            MemoryError::config(format!(
                "Failed to acquire cross-process instance lock: {}",
                e
            ))
        })?;
        debug!("Instance write lock acquired (sync)");
        Ok(WriteGuard {
            file: self.file.clone(),
        })
    }
}

// ── Process-lifetime guard (backward-compat / convenience) ──────────

pub struct InstanceGuard {
    _file: File,
    lock_path: PathBuf,
    mode: &'static str,
}

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        if let Err(e) = fs2::FileExt::unlock(&self._file) {
            error!("Failed to release instance lock: {}", e);
        }
        let _ = std::fs::remove_file(&self.lock_path);
        info!("{} instance lock released", self.mode);
    }
}

/// Acquire an exclusive cross-process lock on the given banks directory
/// and hold it for the process lifetime.
///
/// This is the legacy process-lifetime API using `try_lock_exclusive` for
/// fast-fail detection of another instance. For operation-scoped locking,
/// use `InstanceLockManager::open` + `acquire_write` instead.
#[allow(dead_code)]
pub fn acquire(banks_dir: &Path, mode: &'static str) -> InstanceGuard {
    if !banks_dir.exists() {
        std::fs::create_dir_all(banks_dir).unwrap_or_else(|e| {
            error!(
                "Failed to create banks directory {}: {}",
                banks_dir.display(),
                e
            );
            std::process::exit(1);
        });
    }

    let lock_path = banks_dir.join(".llm-mem.lock");
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .unwrap_or_else(|e| {
            error!(
                "Failed to open lock file {}: {}",
                lock_path.display(),
                e
            );
            std::process::exit(1);
        });

    match fs2::FileExt::try_lock_exclusive(&file) {
        Ok(()) => {
            info!(
                "{} instance lock acquired at {}",
                mode,
                lock_path.display()
            );
            InstanceGuard {
                _file: file,
                lock_path,
                mode,
            }
        }
        Err(_) => {
            error!(
                "Another llm-mem instance ({mode}) is already running with the same database at {}. \
                 Only one process can access the database at a time to prevent data corruption. \
                 Stop the other instance first.",
                banks_dir.display()
            );
            std::process::exit(1);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_banks_dir() -> (PathBuf, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let banks = dir.path().join("banks");
        (banks, dir)
    }

    // ── Basic acquire / release ─────────────────────────────────

    #[test]
    fn test_open_and_acquire_sync() {
        let (banks_dir, _temp) = temp_banks_dir();
        let mgr = InstanceLockManager::open(&banks_dir).expect("open failed");
        assert!(banks_dir.join(".llm-mem.lock").exists());

        let guard = mgr.acquire_write_sync().expect("acquire failed");
        drop(guard);

        let guard2 = mgr.acquire_write_sync().expect("re-acquire failed");
        drop(guard2);
    }

    #[tokio::test]
    async fn test_open_and_acquire_async() {
        let (banks_dir, _temp) = temp_banks_dir();
        let mgr = InstanceLockManager::open(&banks_dir).expect("open failed");

        let guard = mgr.acquire_write().await.expect("acquire failed");
        drop(guard);

        let guard2 = mgr.acquire_write().await.expect("re-acquire failed");
        drop(guard2);
    }

    #[test]
    fn test_guard_releases_on_drop() {
        let (banks_dir, _temp) = temp_banks_dir();
        let mgr = InstanceLockManager::open(&banks_dir).expect("open failed");

        {
            let _guard = mgr.acquire_write_sync().expect("acquire failed");
        }

        let guard2 = mgr.acquire_write_sync().expect("re-acquire after drop failed");
        drop(guard2);
    }

    #[tokio::test]
    async fn test_async_guard_releases_on_drop() {
        let (banks_dir, _temp) = temp_banks_dir();
        let mgr = InstanceLockManager::open(&banks_dir).expect("open failed");

        {
            let _guard = mgr.acquire_write().await.expect("acquire failed");
        }

        let guard2 = mgr.acquire_write().await.expect("re-acquire after drop failed");
        drop(guard2);
    }

    #[test]
    fn test_non_existent_dir_created() {
        let temp = tempfile::TempDir::new().expect("temp");
        let banks_dir = temp.path().join("new_banks");
        assert!(!banks_dir.exists());

        let _mgr = InstanceLockManager::open(&banks_dir).expect("open");
        assert!(banks_dir.exists());
        assert!(banks_dir.join(".llm-mem.lock").exists());
    }

    #[test]
    fn test_write_guard_does_not_remove_lock_file() {
        let (banks_dir, _temp) = temp_banks_dir();
        let mgr = InstanceLockManager::open(&banks_dir).expect("open");
        let lock_path = banks_dir.join(".llm-mem.lock");

        {
            let _guard = mgr.acquire_write_sync().expect("acquire");
            assert!(lock_path.exists());
        }
        assert!(lock_path.exists(), "lock file should persist after guard drop");
    }

    #[test]
    fn test_different_dirs_independent() {
        let dir1 = tempfile::TempDir::new().expect("dir1");
        let dir2 = tempfile::TempDir::new().expect("dir2");

        let mgr1 = InstanceLockManager::open(dir1.path()).expect("open 1");
        let mgr2 = InstanceLockManager::open(dir2.path()).expect("open 2");

        let _guard1 = mgr1.acquire_write_sync().expect("acquire 1");
        let _guard2 = mgr2.acquire_write_sync().expect("acquire 2");
    }

    // ── Legacy acquire (try_lock_exclusive semantics) ─────────────

    #[test]
    fn test_legacy_acquire_lifetime_lock() {
        let (banks_dir, _temp) = temp_banks_dir();
        let guard = acquire(&banks_dir, "TEST");
        assert!(banks_dir.join(".llm-mem.lock").exists());
        drop(guard);
        assert!(!banks_dir.join(".llm-mem.lock").exists());
    }

    /// Multiple rapid acquire/release cycles work correctly.
    #[test]
    fn test_rapid_acquire_release() {
        let (banks_dir, _temp) = temp_banks_dir();
        let mgr = InstanceLockManager::open(&banks_dir).expect("open");

        for _ in 0..100 {
            let _guard = mgr.acquire_write_sync().expect("acquire");
        }
    }

    #[tokio::test]
    async fn test_rapid_acquire_release_async() {
        let (banks_dir, _temp) = temp_banks_dir();
        let mgr = Arc::new(InstanceLockManager::open(&banks_dir).expect("open"));

        for _ in 0..50 {
            let _guard = mgr.acquire_write().await.expect("acquire");
        }
    }
}
