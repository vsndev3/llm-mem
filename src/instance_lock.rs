//! Cross-process advisory file lock to prevent concurrent instances from
//! accessing the same database and corrupting it.
//!
//! Both MCP server and CLI acquire this lock at startup, scoped to the
//! banks directory. Only one writer process can hold the lock at a time.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use tracing::{error, info};

/// Acquired lock that must be held for the process lifetime.
/// Drops the lock and removes the lock file when falling out of scope.
pub struct InstanceGuard {
    _file: File,
    lock_path: PathBuf,
    mode: &'static str,
}

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        if let Err(e) = fs2::FileExt::unlock(&self._file) {
            // Best-effort cleanup; the OS will release on process exit anyway
            error!("Failed to release instance lock: {}", e);
        }
        let _ = std::fs::remove_file(&self.lock_path);
        info!("{} instance lock released", self.mode);
    }
}

/// Acquire an exclusive cross-process lock on the given banks directory.
///
/// Creates a `.llm-mem.lock` file and acquires an advisory lock (`flock` on
/// Unix, `LockFileEx` on Windows). If another process already holds the lock,
/// prints a clear error message and exits the process.
///
/// `mode` is a human-readable label for log messages (e.g. `"MCP"` or `"CLI"`).
pub fn acquire(banks_dir: &Path, mode: &'static str) -> InstanceGuard {
    if !banks_dir.exists() {
        std::fs::create_dir_all(banks_dir).unwrap_or_else(|e| {
            error!("Failed to create banks directory {}: {}", banks_dir.display(), e);
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
            info!("{} instance lock acquired at {}", mode, lock_path.display());
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
