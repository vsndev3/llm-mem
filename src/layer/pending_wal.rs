//! Write-Ahead Log (WAL) for in-flight abstraction state.
//!
//! Persists the `pending_queue` from `AbstractionPipeline` to a SQLite table
//! so that in-progress abstractions survive process crashes. On startup, the
//! pipeline reads this table and re-queues any pending items.
//!
//! Schema mirrors `PendingAbstraction`: memory_id, current_level, target_level,
//! retry_count, queued_at, plus a bank_name for multi-bank deployments.

use chrono::{DateTime, Utc};
use rusqlite::{Connection, params};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::error::{MemoryError, Result};

/// A persisted pending abstraction task — mirrors `PendingAbstraction` with an
/// additional `bank_name` field for multi-bank deployments.
#[derive(Debug, Clone)]
pub struct PendingWalEntry {
    pub memory_id: Uuid,
    pub current_level: i32,
    pub target_level: i32,
    pub retry_count: u32,
    pub queued_at: DateTime<Utc>,
    pub bank_name: String,
}

/// A persisted pending relation — a caller-supplied relation whose target
/// doesn't exist yet. Stored in the WAL until the target memory arrives.
#[derive(Debug, Clone)]
pub struct PendingRelationEntry {
    pub source_id: Uuid,
    pub target_id: Uuid,
    pub relation: String,
    pub strength: Option<f32>,
    pub bank_name: String,
    pub created_at: DateTime<Utc>,
}

/// SQLite-backed persistence for pending abstraction tasks and pending relations.
pub struct PendingWal {
    conn: Arc<Mutex<Connection>>,
}

impl PendingWal {
    /// Open (or create) the WAL database at the given path.
    pub fn open(db_path: &PathBuf) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                MemoryError::config(format!(
                    "Failed to create WAL directory '{}': {}",
                    parent.display(),
                    e
                ))
            })?;
        }

        let conn = Connection::open(db_path).map_err(|e| {
            MemoryError::config(format!(
                "Failed to open abstraction WAL database at '{}': {}",
                db_path.display(),
                e
            ))
        })?;

        let wal = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        wal.initialize_table()?;
        Ok(wal)
    }

    /// Create both tables if they don't exist.
    fn initialize_table(&self) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire WAL lock: {}", e)))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS pending_abstractions (
                memory_id   TEXT NOT NULL,
                bank_name   TEXT NOT NULL DEFAULT 'default',
                current_level INTEGER NOT NULL,
                target_level  INTEGER NOT NULL,
                retry_count   INTEGER NOT NULL DEFAULT 0,
                queued_at     TEXT NOT NULL,
                PRIMARY KEY (memory_id, bank_name)
            );

            CREATE TABLE IF NOT EXISTS pending_relations (
                source_id  TEXT NOT NULL,
                target_id  TEXT NOT NULL,
                relation   TEXT NOT NULL,
                strength   REAL,
                bank_name  TEXT NOT NULL DEFAULT 'default',
                created_at TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, relation, bank_name)
            );",
        )
        .map_err(|e| {
            MemoryError::config(format!(
                "Failed to create WAL tables: {}",
                e
            ))
        })?;

        debug!("Abstraction WAL tables initialized");
        Ok(())
    }

    /// Insert a pending abstraction entry (idempotent — uses INSERT OR REPLACE).
    pub fn insert(&self, entry: &PendingWalEntry) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire WAL lock: {}", e)))?;

        conn.execute(
            "INSERT OR REPLACE INTO pending_abstractions
                (memory_id, bank_name, current_level, target_level, retry_count, queued_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                entry.memory_id.to_string(),
                entry.bank_name,
                entry.current_level,
                entry.target_level,
                entry.retry_count,
                entry.queued_at.to_rfc3339(),
            ],
        )
        .map_err(|e| {
            MemoryError::config(format!(
                "Failed to insert pending abstraction {}: {}",
                entry.memory_id, e
            ))
        })?;

        Ok(())
    }

    /// Remove a pending abstraction entry after successful completion.
    pub fn remove(&self, memory_id: &Uuid, bank_name: &str) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire WAL lock: {}", e)))?;

        conn.execute(
            "DELETE FROM pending_abstractions WHERE memory_id = ?1 AND bank_name = ?2",
            params![memory_id.to_string(), bank_name],
        )
        .map_err(|e| {
            MemoryError::config(format!(
                "Failed to remove pending abstraction {}: {}",
                memory_id, e
            ))
        })?;

        Ok(())
    }

    /// Load all pending abstraction entries (called on startup to re-queue).
    pub fn load_all(&self) -> Result<Vec<PendingWalEntry>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire WAL lock: {}", e)))?;

        let mut stmt = conn
            .prepare(
                "SELECT memory_id, bank_name, current_level, target_level, retry_count, queued_at
                 FROM pending_abstractions",
            )
            .map_err(|e| MemoryError::config(format!("Failed to prepare WAL query: {}", e)))?;

        let entries = stmt
            .query_map([], |row| {
                let memory_id_str: String = row.get(0)?;
                let bank_name: String = row.get(1)?;
                let current_level: i32 = row.get(2)?;
                let target_level: i32 = row.get(3)?;
                let retry_count: u32 = row.get(4)?;
                let queued_at_str: String = row.get(5)?;

                Ok((
                    memory_id_str,
                    bank_name,
                    current_level,
                    target_level,
                    retry_count,
                    queued_at_str,
                ))
            })
            .map_err(|e| {
                MemoryError::config(format!("Failed to query pending_abstractions: {}", e))
            })?
            .filter_map(|row_result| match row_result {
                Ok((mid, bank, cl, tl, rc, qa)) => {
                    let uuid = Uuid::parse_str(&mid).ok()?;
                    let queued_at = DateTime::parse_from_rfc3339(&qa)
                        .map(|dt| dt.with_timezone(&Utc))
                        .ok()?;
                    Some(PendingWalEntry {
                        memory_id: uuid,
                        current_level: cl,
                        target_level: tl,
                        retry_count: rc,
                        queued_at,
                        bank_name: bank,
                    })
                }
                Err(e) => {
                    warn!("Skipping malformed WAL entry: {}", e);
                    None
                }
            })
            .collect();

        Ok(entries)
    }

    /// Remove all entries for a given bank (used during bank deletion).
    pub fn clear_bank(&self, bank_name: &str) -> Result<usize> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire WAL lock: {}", e)))?;

        let removed = conn
            .execute(
                "DELETE FROM pending_abstractions WHERE bank_name = ?1",
                params![bank_name],
            )
            .map_err(|e| {
                MemoryError::config(format!(
                    "Failed to clear WAL for bank '{}': {}",
                    bank_name, e
                ))
            })?;

        let _ = conn.execute(
            "DELETE FROM pending_relations WHERE bank_name = ?1",
            params![bank_name],
        );

        Ok(removed)
    }

    // ── Pending Relation Methods ──────────────────────────────────────────

    /// Insert a pending relation (idempotent — INSERT OR REPLACE).
    pub fn insert_pending_relation(&self, entry: &PendingRelationEntry) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire WAL lock: {}", e)))?;

        conn.execute(
            "INSERT OR REPLACE INTO pending_relations
                (source_id, target_id, relation, strength, bank_name, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                entry.source_id.to_string(),
                entry.target_id.to_string(),
                entry.relation,
                entry.strength,
                entry.bank_name,
                entry.created_at.to_rfc3339(),
            ],
        )
        .map_err(|e| {
            MemoryError::config(format!(
                "Failed to insert pending relation {}→{}: {}",
                entry.source_id, entry.target_id, e
            ))
        })?;

        Ok(())
    }

    /// Remove a resolved pending relation.
    pub fn remove_pending_relation(
        &self,
        source_id: &Uuid,
        target_id: &Uuid,
        relation: &str,
        bank_name: &str,
    ) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire WAL lock: {}", e)))?;

        conn.execute(
            "DELETE FROM pending_relations
             WHERE source_id = ?1 AND target_id = ?2 AND relation = ?3 AND bank_name = ?4",
            params![source_id.to_string(), target_id.to_string(), relation, bank_name],
        )
        .map_err(|e| {
            MemoryError::config(format!(
                "Failed to remove pending relation {}→{}/{}: {}",
                source_id, target_id, relation, e
            ))
        })?;

        Ok(())
    }

    /// Load all pending relations targeting a specific memory (for resolution
    /// when the target memory arrives).
    pub fn load_pending_for_target(&self, target_id: &Uuid) -> Result<Vec<PendingRelationEntry>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire WAL lock: {}", e)))?;

        let mut stmt = conn
            .prepare(
                "SELECT source_id, target_id, relation, strength, bank_name, created_at
                 FROM pending_relations WHERE target_id = ?1",
            )
            .map_err(|e| MemoryError::config(format!("Failed to prepare query: {}", e)))?;

        let entries = stmt
            .query_map(params![target_id.to_string()], |row| {
                let sid: String = row.get(0)?;
                let tid: String = row.get(1)?;
                let rel: String = row.get(2)?;
                let strength: Option<f32> = row.get(3)?;
                let bank: String = row.get(4)?;
                let created: String = row.get(5)?;
                Ok((sid, tid, rel, strength, bank, created))
            })
            .map_err(|e| MemoryError::config(format!("Failed to query pending_relations: {}", e)))?
            .filter_map(|row_result| match row_result {
                Ok((sid, tid, rel, strength, bank, created)) => {
                    let source_id = Uuid::parse_str(&sid).ok()?;
                    let target_id = Uuid::parse_str(&tid).ok()?;
                    let created_at = DateTime::parse_from_rfc3339(&created)
                        .map(|dt| dt.with_timezone(&Utc))
                        .ok()?;
                    Some(PendingRelationEntry {
                        source_id,
                        target_id,
                        relation: rel,
                        strength,
                        bank_name: bank,
                        created_at,
                    })
                }
                Err(e) => {
                    warn!("Skipping malformed pending_relation entry: {}", e);
                    None
                }
            })
            .collect();

        Ok(entries)
    }

    /// Load all pending relations (for startup recovery).
    pub fn load_all_pending_relations(&self) -> Result<Vec<PendingRelationEntry>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire WAL lock: {}", e)))?;

        let mut stmt = conn
            .prepare(
                "SELECT source_id, target_id, relation, strength, bank_name, created_at
                 FROM pending_relations",
            )
            .map_err(|e| MemoryError::config(format!("Failed to prepare query: {}", e)))?;

        let entries = stmt
            .query_map([], |row| {
                let sid: String = row.get(0)?;
                let tid: String = row.get(1)?;
                let rel: String = row.get(2)?;
                let strength: Option<f32> = row.get(3)?;
                let bank: String = row.get(4)?;
                let created: String = row.get(5)?;
                Ok((sid, tid, rel, strength, bank, created))
            })
            .map_err(|e| {
                MemoryError::config(format!("Failed to query pending_relations: {}", e))
            })?
            .filter_map(|row_result| match row_result {
                Ok((sid, tid, rel, strength, bank, created)) => {
                    let source_id = Uuid::parse_str(&sid).ok()?;
                    let target_id = Uuid::parse_str(&tid).ok()?;
                    let created_at = DateTime::parse_from_rfc3339(&created)
                        .map(|dt| dt.with_timezone(&Utc))
                        .ok()?;
                    Some(PendingRelationEntry {
                        source_id,
                        target_id,
                        relation: rel,
                        strength,
                        bank_name: bank,
                        created_at,
                    })
                }
                Err(e) => {
                    warn!("Skipping malformed pending_relation entry: {}", e);
                    None
                }
            })
            .collect();

        Ok(entries)
    }

    /// Clear all pending relations for a bank (used during bank deletion).
    pub fn clear_pending_relations_for_bank(&self, bank_name: &str) -> Result<usize> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire WAL lock: {}", e)))?;

        let removed = conn
            .execute(
                "DELETE FROM pending_relations WHERE bank_name = ?1",
                params![bank_name],
            )
            .map_err(|e| {
                MemoryError::config(format!(
                    "Failed to clear pending_relations for bank '{}': {}",
                    bank_name, e
                ))
            })?;

        Ok(removed)
    }

    /// Count pending entries (for diagnostics).
    pub fn count(&self) -> Result<usize> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire WAL lock: {}", e)))?;

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM pending_abstractions", [], |row| {
                row.get(0)
            })
            .map_err(|e| {
                MemoryError::config(format!("Failed to count pending_abstractions: {}", e))
            })?;

        Ok(count as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_wal(dir: &TempDir) -> PendingWal {
        let path = dir.path().join("test_wal.db");
        PendingWal::open(&path).unwrap()
    }

    #[test]
    fn test_insert_and_load() {
        let dir = TempDir::new().unwrap();
        let wal = make_wal(&dir);

        let entry = PendingWalEntry {
            memory_id: Uuid::new_v4(),
            current_level: 0,
            target_level: 1,
            retry_count: 0,
            queued_at: Utc::now(),
            bank_name: "default".to_string(),
        };

        wal.insert(&entry).unwrap();

        let loaded = wal.load_all().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].memory_id, entry.memory_id);
        assert_eq!(loaded[0].current_level, 0);
        assert_eq!(loaded[0].target_level, 1);
        assert_eq!(loaded[0].bank_name, "default");
    }

    #[test]
    fn test_remove() {
        let dir = TempDir::new().unwrap();
        let wal = make_wal(&dir);

        let entry = PendingWalEntry {
            memory_id: Uuid::new_v4(),
            current_level: 0,
            target_level: 1,
            retry_count: 0,
            queued_at: Utc::now(),
            bank_name: "default".to_string(),
        };

        wal.insert(&entry).unwrap();
        assert_eq!(wal.count().unwrap(), 1);

        wal.remove(&entry.memory_id, "default").unwrap();
        assert_eq!(wal.count().unwrap(), 0);
    }

    #[test]
    fn test_idempotent_insert() {
        let dir = TempDir::new().unwrap();
        let wal = make_wal(&dir);

        let id = Uuid::new_v4();
        let entry = PendingWalEntry {
            memory_id: id,
            current_level: 0,
            target_level: 1,
            retry_count: 0,
            queued_at: Utc::now(),
            bank_name: "default".to_string(),
        };

        wal.insert(&entry).unwrap();
        wal.insert(&entry).unwrap();
        assert_eq!(wal.count().unwrap(), 1);
    }

    #[test]
    fn test_multi_bank() {
        let dir = TempDir::new().unwrap();
        let wal = make_wal(&dir);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        wal.insert(&PendingWalEntry {
            memory_id: id1,
            current_level: 0,
            target_level: 1,
            retry_count: 0,
            queued_at: Utc::now(),
            bank_name: "bank_a".to_string(),
        })
        .unwrap();

        // Same memory_id but different bank — should coexist
        wal.insert(&PendingWalEntry {
            memory_id: id1,
            current_level: 0,
            target_level: 1,
            retry_count: 0,
            queued_at: Utc::now(),
            bank_name: "bank_b".to_string(),
        })
        .unwrap();

        wal.insert(&PendingWalEntry {
            memory_id: id2,
            current_level: 1,
            target_level: 2,
            retry_count: 1,
            queued_at: Utc::now(),
            bank_name: "bank_a".to_string(),
        })
        .unwrap();

        assert_eq!(wal.count().unwrap(), 3);

        // Clear one bank
        let removed = wal.clear_bank("bank_a").unwrap();
        assert_eq!(removed, 2);
        assert_eq!(wal.count().unwrap(), 1);

        // The remaining entry should be bank_b
        let loaded = wal.load_all().unwrap();
        assert_eq!(loaded[0].bank_name, "bank_b");
    }

    #[test]
    fn test_persistence_across_reopen() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("persist_wal.db");

        let id = Uuid::new_v4();
        {
            let wal = PendingWal::open(&path).unwrap();
            wal.insert(&PendingWalEntry {
                memory_id: id,
                current_level: 0,
                target_level: 1,
                retry_count: 2,
                queued_at: Utc::now(),
                bank_name: "default".to_string(),
            })
            .unwrap();
        }

        // Reopen
        let wal2 = PendingWal::open(&path).unwrap();
        let loaded = wal2.load_all().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].memory_id, id);
        assert_eq!(loaded[0].retry_count, 2);
    }

    #[test]
    fn test_empty_load() {
        let dir = TempDir::new().unwrap();
        let wal = make_wal(&dir);
        let loaded = wal.load_all().unwrap();
        assert!(loaded.is_empty());
    }
}
