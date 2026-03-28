//! Document session management for stateful, resumable document ingestion.
//!
//! This module provides a session-based API for uploading large documents
//! in chunks, preventing payload limits and enabling resumable processing.

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::{debug, info, trace};
use uuid::Uuid;

use crate::error::{MemoryError, Result};

/// Default chunk size in bytes (8KB)
pub const DEFAULT_CHUNK_SIZE_BYTES: usize = 8192;

/// Status of a document processing session
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SessionStatus {
    Uploading,
    Processing,
    Completed,
    Failed,
    Cancelled,
}

impl SessionStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            SessionStatus::Uploading => "uploading",
            SessionStatus::Processing => "processing",
            SessionStatus::Completed => "completed",
            SessionStatus::Failed => "failed",
            SessionStatus::Cancelled => "cancelled",
        }
    }

    pub fn parse_str(s: &str) -> Self {
        match s {
            "uploading" => SessionStatus::Uploading,
            "processing" => SessionStatus::Processing,
            "completed" => SessionStatus::Completed,
            "failed" => SessionStatus::Failed,
            "cancelled" => SessionStatus::Cancelled,
            _ => SessionStatus::Uploading,
        }
    }
}

/// Metadata for a document being uploaded
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Original file name or path
    pub file_name: String,
    /// MIME type or file extension
    pub file_type: Option<String>,
    /// Total size in bytes
    pub total_size: usize,
    /// Optional MD5 checksum
    pub md5sum: Option<String>,
    /// User ID for ownership
    pub user_id: Option<String>,
    /// Agent ID for ownership
    pub agent_id: Option<String>,
    /// Memory type for extracted memories
    pub memory_type: String,
    /// Topics to associate with extracted memories
    pub topics: Option<Vec<String>>,
    /// Context tags for semantic scoping
    pub context: Option<Vec<String>>,
    /// Custom metadata
    pub custom_metadata: Option<serde_json::Value>,
}

/// A document upload session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSession {
    pub session_id: String,
    pub status: SessionStatus,
    pub expected_parts: usize,
    pub received_parts: usize,
    pub chunk_size_bytes: usize,
    pub metadata: DocumentMetadata,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub error_message: Option<String>,
    pub processing_result: Option<ProcessingResult>,
}

/// Result of document processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub total_chunks: usize,
    pub chunks_processed: usize,
    pub memories_created: usize,
    pub summary: Option<String>,
    /// Number of chunks that have had metadata enrichment completed (LLM extraction phase)
    #[serde(default)]
    pub chunks_enriched: usize,
    /// Upper bound of the current in-flight enrichment batch (chunks_enriched..chunks_enriching_end are active)
    #[serde(default)]
    pub chunks_enriching_end: usize,
}

/// Chunk progress status for visualization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChunkStatus {
    Queued,
    Uploading,
    Enriching,
    Embedding,
    Completed,
    Failed,
}

impl ChunkStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChunkStatus::Queued => "queued",
            ChunkStatus::Uploading => "uploading",
            ChunkStatus::Enriching => "enriching",
            ChunkStatus::Embedding => "embedding",
            ChunkStatus::Completed => "completed",
            ChunkStatus::Failed => "failed",
        }
    }
}

/// Progress information for a single chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkProgress {
    pub index: usize,
    pub status: ChunkStatus,
    pub progress: f32,
    pub timestamp: DateTime<Utc>,
    pub error: Option<String>,
}

/// Response for begin_store_document operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeginStoreDocumentResponse {
    pub session_id: String,
    pub chunk_size_bytes: usize,
    pub expected_parts: usize,
    pub estimated_time_seconds: f64,
}

/// Response for status_process_document operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusProcessDocumentResponse {
    pub session_id: String,
    pub status: SessionStatus,
    pub chunks_received: usize,
    pub total_chunks: usize,
    pub chunks_processed: Option<usize>,
    pub result: Option<ProcessingResult>,
    pub error: Option<String>,
}

/// Manager for document sessions using SQLite persistence
pub struct DocumentSessionManager {
    conn: Arc<Mutex<Connection>>,
    chunk_size_bytes: usize,
}

impl DocumentSessionManager {
    /// Create a new session manager with a SQLite database at the given path
    pub fn new(db_path: PathBuf, chunk_size_bytes: Option<usize>) -> Result<Self> {
        let conn = Connection::open(&db_path).map_err(|e| {
            MemoryError::config(format!(
                "Failed to open document session database at '{}': {}",
                db_path.display(),
                e
            ))
        })?;

        let manager = Self {
            conn: Arc::new(Mutex::new(conn)),
            chunk_size_bytes: chunk_size_bytes.unwrap_or(DEFAULT_CHUNK_SIZE_BYTES),
        };

        manager.initialize_tables()?;
        info!(
            "Document session manager initialized (db: {}, chunk_size: {} bytes)",
            db_path.display(),
            manager.chunk_size_bytes
        );
        Ok(manager)
    }

    /// Initialize database tables
    fn initialize_tables(&self) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire database lock: {}", e)))?;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS document_sessions (
                session_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'uploading',
                expected_parts INTEGER NOT NULL,
                received_parts INTEGER NOT NULL DEFAULT 0,
                chunk_size_bytes INTEGER NOT NULL,
                file_name TEXT NOT NULL,
                file_type TEXT,
                total_size INTEGER NOT NULL,
                md5sum TEXT,
                user_id TEXT,
                agent_id TEXT,
                memory_type TEXT NOT NULL,
                topics_json TEXT,
                context_json TEXT,
                custom_metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error_message TEXT,
                processing_result_json TEXT
            );

            CREATE TABLE IF NOT EXISTS document_parts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                part_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES document_sessions(session_id) ON DELETE CASCADE,
                UNIQUE(session_id, part_index)
            );

            CREATE INDEX IF NOT EXISTS idx_document_parts_session ON document_parts(session_id);
            "#,
        )
        .map_err(|e| {
            MemoryError::config(format!(
                "Failed to initialize document session tables: {}",
                e
            ))
        })?;

        debug!("Document session tables initialized");
        Ok(())
    }

    /// Begin a new document upload session
    pub fn begin_session(&self, metadata: DocumentMetadata) -> Result<BeginStoreDocumentResponse> {
        let session_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let expected_parts = metadata.total_size.div_ceil(self.chunk_size_bytes);
        let expected_parts = expected_parts.max(1);

        let estimated_time_seconds =
            Self::estimate_processing_time(metadata.total_size, expected_parts);

        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire database lock: {}", e)))?;

        conn.execute(
            r#"
            INSERT INTO document_sessions (
                session_id, status, expected_parts, received_parts, chunk_size_bytes,
                file_name, file_type, total_size, md5sum, user_id, agent_id,
                memory_type, topics_json, context_json, custom_metadata_json,
                created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)
            "#,
            params![
                session_id,
                SessionStatus::Uploading.as_str(),
                expected_parts as i32,
                0i32,
                self.chunk_size_bytes as i32,
                metadata.file_name,
                metadata.file_type,
                metadata.total_size as i32,
                metadata.md5sum,
                metadata.user_id,
                metadata.agent_id,
                metadata.memory_type,
                metadata
                    .topics
                    .as_ref()
                    .map(|t| serde_json::to_string(t).unwrap_or_default()),
                metadata
                    .context
                    .as_ref()
                    .map(|c| serde_json::to_string(c).unwrap_or_default()),
                metadata.custom_metadata.as_ref().map(|m| m.to_string()),
                now.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )
        .map_err(|e| MemoryError::config(format!("Failed to create document session: {}", e)))?;

        info!(
            "Created document session {} for '{}' ({} bytes, {} parts expected)",
            session_id, metadata.file_name, metadata.total_size, expected_parts
        );

        Ok(BeginStoreDocumentResponse {
            session_id,
            chunk_size_bytes: self.chunk_size_bytes,
            expected_parts,
            estimated_time_seconds,
        })
    }

    /// Store a document part
    pub fn store_part(&self, session_id: &str, part_index: usize, content: &str) -> Result<()> {
        let now = Utc::now();
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire database lock: {}", e)))?;

        let session_exists: bool = conn
            .query_row(
                "SELECT 1 FROM document_sessions WHERE session_id = ?1",
                params![session_id],
                |_| Ok(true),
            )
            .unwrap_or(false);

        if !session_exists {
            return Err(MemoryError::NotFound {
                id: session_id.to_string(),
            });
        }

        conn.execute(
            r#"
            INSERT OR REPLACE INTO document_parts (session_id, part_index, content, created_at)
            VALUES (?1, ?2, ?3, ?4)
            "#,
            params![session_id, part_index as i32, content, now.to_rfc3339()],
        )
        .map_err(|e| MemoryError::config(format!("Failed to store document part: {}", e)))?;

        conn.execute(
            r#"
            UPDATE document_sessions
            SET received_parts = (SELECT COUNT(*) FROM document_parts WHERE session_id = ?1),
                updated_at = ?2
            WHERE session_id = ?1
            "#,
            params![session_id, now.to_rfc3339()],
        )
        .map_err(|e| {
            MemoryError::config(format!("Failed to update session received_parts: {}", e))
        })?;

        debug!(
            "Stored part {} for session {} (content length: {})",
            part_index,
            session_id,
            content.len()
        );
        Ok(())
    }

    /// Update session status
    pub fn update_status(
        &self,
        session_id: &str,
        status: SessionStatus,
        error: Option<&str>,
    ) -> Result<()> {
        let now = Utc::now();
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire database lock: {}", e)))?;

        conn.execute(
            r#"
            UPDATE document_sessions
            SET status = ?1, error_message = ?2, updated_at = ?3
            WHERE session_id = ?4
            "#,
            params![status.as_str(), error, now.to_rfc3339(), session_id],
        )
        .map_err(|e| MemoryError::config(format!("Failed to update session status: {}", e)))?;

        debug!("Updated session {} status to {:?}", session_id, status);
        Ok(())
    }

    /// Update expected parts count (used when actual chunks differ from estimate)
    pub fn update_expected_parts(&self, session_id: &str, expected_parts: usize) -> Result<()> {
        let now = Utc::now();
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire database lock: {}", e)))?;

        conn.execute(
            r#"
            UPDATE document_sessions
            SET expected_parts = ?1, updated_at = ?2
            WHERE session_id = ?3
            "#,
            params![expected_parts as i32, now.to_rfc3339(), session_id],
        )
        .map_err(|e| {
            MemoryError::config(format!("Failed to update session expected_parts: {}", e))
        })?;

        debug!(
            "Updated session {} expected_parts to {}",
            session_id, expected_parts
        );
        Ok(())
    }

    /// Store processing result without changing status
    pub fn store_processing_result(
        &self,
        session_id: &str,
        result: &ProcessingResult,
    ) -> Result<()> {
        let now = Utc::now();
        let result_json = serde_json::to_string(result).unwrap_or_default();
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire database lock: {}", e)))?;

        conn.execute(
            r#"
            UPDATE document_sessions
            SET processing_result_json = ?1,
                updated_at = ?2
            WHERE session_id = ?3
            "#,
            params![result_json, now.to_rfc3339(), session_id],
        )
        .map_err(|e| MemoryError::config(format!("Failed to store processing result: {}", e)))?;

        debug!(
            "Stored processing result for session {}: {} memories created",
            session_id, result.memories_created
        );
        Ok(())
    }

    /// Get session status
    pub fn get_status(&self, session_id: &str) -> Result<StatusProcessDocumentResponse> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire database lock: {}", e)))?;

        let result = conn.query_row(
            r#"
            SELECT session_id, status, expected_parts, received_parts,
                   error_message, processing_result_json
            FROM document_sessions
            WHERE session_id = ?1
            "#,
            params![session_id],
            |row| {
                let session_id: String = row.get(0)?;
                let status_str: String = row.get(1)?;
                let expected_parts: i32 = row.get(2)?;
                let received_parts: i32 = row.get(3)?;
                let error_message: Option<String> = row.get(4)?;
                let processing_result_json: Option<String> = row.get(5)?;

                let processing_result = processing_result_json
                    .and_then(|s| serde_json::from_str::<ProcessingResult>(&s).ok());

                Ok(StatusProcessDocumentResponse {
                    session_id,
                    status: SessionStatus::parse_str(&status_str),
                    chunks_received: received_parts as usize,
                    total_chunks: expected_parts as usize,
                    chunks_processed: processing_result.as_ref().map(|r| r.chunks_processed),
                    result: processing_result,
                    error: error_message,
                })
            },
        );

        match result {
            Ok(status) => Ok(status),
            Err(rusqlite::Error::QueryReturnedNoRows) => Err(MemoryError::NotFound {
                id: session_id.to_string(),
            }),
            Err(e) => Err(MemoryError::config(format!(
                "Failed to get session status: {}",
                e
            ))),
        }
    }

    /// Get session metadata
    pub fn get_session(&self, session_id: &str) -> Result<DocumentSession> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire database lock: {}", e)))?;

        let result = conn.query_row(
            r#"
            SELECT session_id, status, expected_parts, received_parts, chunk_size_bytes,
                   file_name, file_type, total_size, md5sum, user_id, agent_id,
                   memory_type, topics_json, context_json, custom_metadata_json,
                   created_at, updated_at, error_message, processing_result_json
            FROM document_sessions
            WHERE session_id = ?1
            "#,
            params![session_id],
            |row| {
                let session_id: String = row.get(0)?;
                let status_str: String = row.get(1)?;
                let expected_parts: i32 = row.get(2)?;
                let received_parts: i32 = row.get(3)?;
                let chunk_size_bytes: i32 = row.get(4)?;
                let file_name: String = row.get(5)?;
                let file_type: Option<String> = row.get(6)?;
                let total_size: i32 = row.get(7)?;
                let md5sum: Option<String> = row.get(8)?;
                let user_id: Option<String> = row.get(9)?;
                let agent_id: Option<String> = row.get(10)?;
                let memory_type: String = row.get(11)?;
                let topics_json: Option<String> = row.get(12)?;
                let context_json: Option<String> = row.get(13)?;
                let custom_metadata_json: Option<String> = row.get(14)?;
                let created_at_str: String = row.get(15)?;
                let updated_at_str: String = row.get(16)?;
                let error_message: Option<String> = row.get(17)?;
                let processing_result_json: Option<String> = row.get(18)?;

                let topics = topics_json.and_then(|s| serde_json::from_str(&s).ok());
                let context = context_json.and_then(|s| serde_json::from_str(&s).ok());
                let custom_metadata =
                    custom_metadata_json.and_then(|s| serde_json::from_str(&s).ok());
                let processing_result =
                    processing_result_json.and_then(|s| serde_json::from_str(&s).ok());

                let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());
                let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                Ok(DocumentSession {
                    session_id,
                    status: SessionStatus::parse_str(&status_str),
                    expected_parts: expected_parts as usize,
                    received_parts: received_parts as usize,
                    chunk_size_bytes: chunk_size_bytes as usize,
                    metadata: DocumentMetadata {
                        file_name,
                        file_type,
                        total_size: total_size as usize,
                        md5sum,
                        user_id,
                        agent_id,
                        memory_type,
                        topics,
                        context,
                        custom_metadata,
                    },
                    created_at,
                    updated_at,
                    error_message,
                    processing_result,
                })
            },
        );

        match result {
            Ok(session) => Ok(session),
            Err(rusqlite::Error::QueryReturnedNoRows) => Err(MemoryError::NotFound {
                id: session_id.to_string(),
            }),
            Err(e) => Err(MemoryError::config(format!("Failed to get session: {}", e))),
        }
    }

    /// List all document sessions (completed, failed, etc.)
    pub fn list_all_sessions(&self) -> Result<Vec<DocumentSession>> {
        let session_ids = {
            let conn = self.conn.lock().map_err(|e| {
                MemoryError::config(format!("Failed to acquire database lock: {}", e))
            })?;

            let mut stmt = conn
                .prepare("SELECT session_id FROM document_sessions ORDER BY created_at DESC")
                .map_err(|e| {
                    MemoryError::config(format!("Failed to prepare sessions query: {}", e))
                })?;

            stmt.query_map([], |row| row.get::<_, String>(0))
                .map_err(|e| MemoryError::config(format!("Failed to query sessions: {}", e)))?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| MemoryError::config(format!("Failed to collect sessions: {}", e)))?
        };

        let mut sessions = Vec::new();
        for id in session_ids {
            if let Ok(session) = self.get_session(&id) {
                sessions.push(session);
            }
        }

        Ok(sessions)
    }

    /// Mark 'processing' sessions that haven't been updated for a long time as 'failed' (timeout).
    /// Returns the number of sessions failed.
    pub fn fail_stalled_sessions(&self, timeout_seconds: u64) -> Result<usize> {
        let now = Utc::now();
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire database lock: {}", e)))?;

        let mut stmt = conn
            .prepare(
                "SELECT session_id, updated_at FROM document_sessions WHERE status = 'processing'",
            )
            .map_err(|e| {
                MemoryError::config(format!("Failed to prepare stalled sessions query: {}", e))
            })?;

        let processing_sessions = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| {
                MemoryError::config(format!("Failed to query processing sessions: {}", e))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| {
                MemoryError::config(format!("Failed to collect processing sessions: {}", e))
            })?;

        let mut failed_count = 0;
        for (id, updated_at_str) in processing_sessions {
            if let Ok(updated_at) = DateTime::parse_from_rfc3339(&updated_at_str) {
                let updated_at = updated_at.with_timezone(&Utc);
                let age = now.signed_duration_since(updated_at);

                if age.num_seconds() > timeout_seconds as i64 {
                    conn.execute(
                        "UPDATE document_sessions SET status = 'failed', error_message = ?1, updated_at = ?2 WHERE session_id = ?3",
                        params![
                            format!("Processing timed out after {} seconds", timeout_seconds),
                            now.to_rfc3339(),
                            id
                        ],
                    )
                    .map_err(|e| {
                        MemoryError::config(format!("Failed to mark session as failed: {}", e))
                    })?;
                    failed_count += 1;
                    info!("Marked stalled session {} as failed (timeout)", id);
                }
            }
        }

        Ok(failed_count)
    }

    /// Get all parts for a session, ordered by part_index
    pub fn get_parts(&self, session_id: &str) -> Result<Vec<(usize, String)>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire database lock: {}", e)))?;

        let mut stmt = conn
            .prepare(
                "SELECT part_index, content FROM document_parts WHERE session_id = ?1 ORDER BY part_index",
            )
            .map_err(|e| {
                MemoryError::config(format!("Failed to prepare parts query: {}", e))
            })?;

        let parts = stmt
            .query_map(params![session_id], |row| {
                let part_index: i32 = row.get(0)?;
                let content: String = row.get(1)?;
                Ok((part_index as usize, content))
            })
            .map_err(|e| MemoryError::config(format!("Failed to query parts: {}", e)))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| MemoryError::config(format!("Failed to collect parts: {}", e)))?;

        trace!("Retrieved {} parts for session {}", parts.len(), session_id);
        Ok(parts)
    }

    /// Cancel a session and clean up its parts
    pub fn cancel_session(&self, session_id: &str) -> Result<()> {
        let now = Utc::now();
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire database lock: {}", e)))?;

        conn.execute(
            "DELETE FROM document_parts WHERE session_id = ?1",
            params![session_id],
        )
        .map_err(|e| MemoryError::config(format!("Failed to delete parts: {}", e)))?;

        conn.execute(
            "UPDATE document_sessions SET status = 'cancelled', updated_at = ?1 WHERE session_id = ?2",
            params![now.to_rfc3339(), session_id],
        )
        .map_err(|e| {
            MemoryError::config(format!("Failed to cancel session: {}", e))
        })?;

        info!("Cancelled session {}", session_id);
        Ok(())
    }

    /// Delete a session and all its parts
    pub fn delete_session(&self, session_id: &str) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire database lock: {}", e)))?;

        conn.execute(
            "DELETE FROM document_parts WHERE session_id = ?1",
            params![session_id],
        )
        .map_err(|e| MemoryError::config(format!("Failed to delete parts: {}", e)))?;

        conn.execute(
            "DELETE FROM document_sessions WHERE session_id = ?1",
            params![session_id],
        )
        .map_err(|e| MemoryError::config(format!("Failed to delete session: {}", e)))?;

        info!("Deleted session {}", session_id);
        Ok(())
    }

    /// Get detailed chunk progress for a session
    pub fn get_session_chunk_progress(&self, session_id: &str) -> Result<Vec<ChunkProgress>> {
        let session = self.get_session(session_id)?;
        let parts = self.get_parts(session_id)?;

        let mut chunk_progress = Vec::new();

        for (index, _) in &parts {
            let progress = if *index < session.received_parts {
                // Chunk received and ready for processing
                if session.status == SessionStatus::Processing {
                    let pr = session.processing_result.as_ref();
                    let is_processed = pr.map(|r| r.chunks_processed > *index).unwrap_or(false);
                    let is_enriched = pr.map(|r| r.chunks_enriched > *index).unwrap_or(false);
                    let is_in_flight = pr.map(|r| *index >= r.chunks_enriched && *index < r.chunks_enriching_end).unwrap_or(false);

                    if is_processed {
                        ChunkProgress {
                            index: *index,
                            status: ChunkStatus::Completed,
                            progress: 1.0,
                            timestamp: session.updated_at,
                            error: None,
                        }
                    } else if is_enriched {
                        ChunkProgress {
                            index: *index,
                            status: ChunkStatus::Embedding,
                            progress: 0.5,
                            timestamp: session.updated_at,
                            error: None,
                        }
                    } else if is_in_flight {
                        ChunkProgress {
                            index: *index,
                            status: ChunkStatus::Enriching,
                            progress: 0.25,
                            timestamp: session.updated_at,
                            error: None,
                        }
                    } else {
                        ChunkProgress {
                            index: *index,
                            status: ChunkStatus::Queued,
                            progress: 0.0,
                            timestamp: session.updated_at,
                            error: None,
                        }
                    }
                } else {
                    ChunkProgress {
                        index: *index,
                        status: ChunkStatus::Completed,
                        progress: 1.0,
                        timestamp: session.updated_at,
                        error: None,
                    }
                }
            } else if *index == session.received_parts {
                // Currently being uploaded
                ChunkProgress {
                    index: *index,
                    status: ChunkStatus::Uploading,
                    progress: 0.5,
                    timestamp: session.updated_at,
                    error: None,
                }
            } else {
                // Not yet received
                ChunkProgress {
                    index: *index,
                    status: ChunkStatus::Queued,
                    progress: 0.0,
                    timestamp: session.created_at,
                    error: None,
                }
            };

            chunk_progress.push(progress);
        }

        // Add queued chunks that haven't been uploaded yet
        let received_count = session.received_parts;
        for index in received_count..session.expected_parts {
            chunk_progress.push(ChunkProgress {
                index,
                status: ChunkStatus::Queued,
                progress: 0.0,
                timestamp: session.created_at,
                error: None,
            });
        }

        // Add error status if session failed
        if session.status == SessionStatus::Failed {
            if let Some(error) = &session.error_message {
                if let Some(last_chunk) = chunk_progress.last_mut() {
                    last_chunk.status = ChunkStatus::Failed;
                    last_chunk.error = Some(error.clone());
                }
            }
        }

        Ok(chunk_progress)
    }

    /// List all active sessions (uploading or processing)
    pub fn list_active_sessions(&self) -> Result<Vec<DocumentSession>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::config(format!("Failed to acquire database lock: {}", e)))?;

        let mut stmt = conn
            .prepare(
                r#"
                SELECT session_id, status, expected_parts, received_parts, chunk_size_bytes,
                       file_name, file_type, total_size, md5sum, user_id, agent_id,
                       memory_type, topics_json, context_json, custom_metadata_json,
                       created_at, updated_at, error_message, processing_result_json
                FROM document_sessions
                WHERE status IN ('uploading', 'processing')
                ORDER BY created_at DESC
                "#,
            )
            .map_err(|e| MemoryError::config(format!("Failed to prepare sessions query: {}", e)))?;

        let sessions = stmt
            .query_map([], |row| {
                let session_id: String = row.get(0)?;
                let status_str: String = row.get(1)?;
                let expected_parts: i32 = row.get(2)?;
                let received_parts: i32 = row.get(3)?;
                let chunk_size_bytes: i32 = row.get(4)?;
                let file_name: String = row.get(5)?;
                let file_type: Option<String> = row.get(6)?;
                let total_size: i32 = row.get(7)?;
                let md5sum: Option<String> = row.get(8)?;
                let user_id: Option<String> = row.get(9)?;
                let agent_id: Option<String> = row.get(10)?;
                let memory_type: String = row.get(11)?;
                let topics_json: Option<String> = row.get(12)?;
                let context_json: Option<String> = row.get(13)?;
                let custom_metadata_json: Option<String> = row.get(14)?;
                let created_at_str: String = row.get(15)?;
                let updated_at_str: String = row.get(16)?;
                let error_message: Option<String> = row.get(17)?;
                let processing_result_json: Option<String> = row.get(18)?;

                let topics = topics_json.and_then(|s| serde_json::from_str(&s).ok());
                let context = context_json.and_then(|s| serde_json::from_str(&s).ok());
                let custom_metadata =
                    custom_metadata_json.and_then(|s| serde_json::from_str(&s).ok());
                let processing_result =
                    processing_result_json.and_then(|s| serde_json::from_str(&s).ok());

                let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());
                let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                Ok(DocumentSession {
                    session_id,
                    status: SessionStatus::parse_str(&status_str),
                    expected_parts: expected_parts as usize,
                    received_parts: received_parts as usize,
                    chunk_size_bytes: chunk_size_bytes as usize,
                    metadata: DocumentMetadata {
                        file_name,
                        file_type,
                        total_size: total_size as usize,
                        md5sum,
                        user_id,
                        agent_id,
                        memory_type,
                        topics,
                        context,
                        custom_metadata,
                    },
                    created_at,
                    updated_at,
                    error_message,
                    processing_result,
                })
            })
            .map_err(|e| MemoryError::config(format!("Failed to query sessions: {}", e)))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| MemoryError::config(format!("Failed to collect sessions: {}", e)))?;

        Ok(sessions)
    }

    /// Estimate processing time based on document size
    fn estimate_processing_time(total_size: usize, expected_parts: usize) -> f64 {
        let base_time = 1.0;
        let time_per_kb = 0.1;
        let time_per_chunk_for_llm = 0.5;

        let size_kb = total_size as f64 / 1024.0;
        let llm_time = expected_parts as f64 * time_per_chunk_for_llm;

        base_time + (size_kb * time_per_kb) + llm_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    struct TestContext {
        _dir: TempDir,
        manager: DocumentSessionManager,
    }

    fn make_manager() -> TestContext {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("test.sessions.db");
        let manager = DocumentSessionManager::new(db_path, Some(1024)).unwrap();
        TestContext { _dir: dir, manager }
    }

    fn make_metadata() -> DocumentMetadata {
        DocumentMetadata {
            file_name: "test.md".to_string(),
            file_type: Some("text/markdown".to_string()),
            total_size: 5000,
            md5sum: None,
            user_id: Some("user1".to_string()),
            agent_id: None,
            memory_type: "semantic".to_string(),
            topics: Some(vec!["test".to_string()]),
            context: None,
            custom_metadata: None,
        }
    }

    #[test]
    fn test_begin_session() {
        let ctx = make_manager();
        let metadata = make_metadata();

        let response = ctx.manager.begin_session(metadata).unwrap();
        assert!(!response.session_id.is_empty());
        assert_eq!(response.chunk_size_bytes, 1024);
        assert_eq!(response.expected_parts, 5);
        assert!(response.estimated_time_seconds > 0.0);
    }

    #[test]
    fn test_store_and_get_parts() {
        let ctx = make_manager();
        let metadata = make_metadata();

        let response = ctx.manager.begin_session(metadata).unwrap();
        let session_id = &response.session_id;

        ctx.manager
            .store_part(session_id, 0, "part 0 content")
            .unwrap();
        ctx.manager
            .store_part(session_id, 1, "part 1 content")
            .unwrap();

        let parts = ctx.manager.get_parts(session_id).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], (0, "part 0 content".to_string()));
        assert_eq!(parts[1], (1, "part 1 content".to_string()));
    }

    #[test]
    fn test_get_status() {
        let ctx = make_manager();
        let metadata = make_metadata();

        let response = ctx.manager.begin_session(metadata).unwrap();
        let session_id = &response.session_id;

        let status = ctx.manager.get_status(session_id).unwrap();
        assert_eq!(status.status, SessionStatus::Uploading);
        assert_eq!(status.chunks_received, 0);
        assert_eq!(status.total_chunks, 5);

        ctx.manager.store_part(session_id, 0, "content").unwrap();

        let status = ctx.manager.get_status(session_id).unwrap();
        assert_eq!(status.chunks_received, 1);
    }

    #[test]
    fn test_update_status() {
        let ctx = make_manager();
        let metadata = make_metadata();

        let response = ctx.manager.begin_session(metadata).unwrap();
        let session_id = &response.session_id;

        ctx.manager
            .update_status(session_id, SessionStatus::Processing, None)
            .unwrap();

        let status = ctx.manager.get_status(session_id).unwrap();
        assert_eq!(status.status, SessionStatus::Processing);

        ctx.manager
            .update_status(session_id, SessionStatus::Failed, Some("Test error"))
            .unwrap();

        let status = ctx.manager.get_status(session_id).unwrap();
        assert_eq!(status.status, SessionStatus::Failed);
        assert_eq!(status.error, Some("Test error".to_string()));
    }

    #[test]
    fn test_cancel_session() {
        let ctx = make_manager();
        let metadata = make_metadata();

        let response = ctx.manager.begin_session(metadata).unwrap();
        let session_id = &response.session_id;

        ctx.manager.store_part(session_id, 0, "content").unwrap();

        ctx.manager.cancel_session(session_id).unwrap();

        let status = ctx.manager.get_status(session_id).unwrap();
        assert_eq!(status.status, SessionStatus::Cancelled);

        let parts = ctx.manager.get_parts(session_id).unwrap();
        assert!(parts.is_empty());
    }

    #[test]
    fn test_delete_session() {
        let ctx = make_manager();
        let metadata = make_metadata();

        let response = ctx.manager.begin_session(metadata).unwrap();
        let session_id = &response.session_id;

        ctx.manager.store_part(session_id, 0, "content").unwrap();

        ctx.manager.delete_session(session_id).unwrap();

        let result = ctx.manager.get_status(session_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_session_not_found() {
        let ctx = make_manager();
        let result = ctx.manager.get_status("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_store_processing_result() {
        let ctx = make_manager();
        let metadata = make_metadata();

        let response = ctx.manager.begin_session(metadata).unwrap();
        let session_id = &response.session_id;

        let result = ProcessingResult {
            total_chunks: 5,
            chunks_processed: 5,
            memories_created: 10,
            summary: Some("Test summary".to_string()),
            chunks_enriched: 5,
            chunks_enriching_end: 5,
        };

        ctx.manager
            .store_processing_result(session_id, &result)
            .unwrap();
        ctx.manager
            .update_status(session_id, SessionStatus::Completed, None)
            .unwrap();

        let status = ctx.manager.get_status(session_id).unwrap();
        assert_eq!(status.status, SessionStatus::Completed);
        assert!(status.result.is_some());
        let result = status.result.unwrap();
        assert_eq!(result.memories_created, 10);
    }

    #[test]
    fn test_list_active_sessions() {
        let ctx = make_manager();

        let metadata1 = make_metadata();
        let response1 = ctx.manager.begin_session(metadata1).unwrap();

        let mut metadata2 = make_metadata();
        metadata2.file_name = "test2.md".to_string();
        let response2 = ctx.manager.begin_session(metadata2).unwrap();

        ctx.manager
            .update_status(&response1.session_id, SessionStatus::Completed, None)
            .unwrap();

        let active = ctx.manager.list_active_sessions().unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].session_id, response2.session_id);
    }

    #[test]
    fn test_list_all_sessions() {
        let ctx = make_manager();

        let metadata1 = make_metadata();
        let response1 = ctx.manager.begin_session(metadata1).unwrap();

        let mut metadata2 = make_metadata();
        metadata2.file_name = "test2.md".to_string();
        let _response2 = ctx.manager.begin_session(metadata2).unwrap();

        ctx.manager
            .update_status(&response1.session_id, SessionStatus::Completed, None)
            .unwrap();

        let all = ctx.manager.list_all_sessions().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_fail_stalled_sessions() {
        let ctx = make_manager();
        let metadata = make_metadata();
        let response = ctx.manager.begin_session(metadata).unwrap();
        let session_id = &response.session_id;

        // Set to processing
        ctx.manager
            .update_status(session_id, SessionStatus::Processing, None)
            .unwrap();

        // Should not fail with 0 timeout if it was just updated
        let failed = ctx.manager.fail_stalled_sessions(100).unwrap();
        assert_eq!(failed, 0);

        // Manually backdate updated_at in the DB to simulate a stall
        {
            let conn = ctx.manager.conn.lock().unwrap();
            let stalled_time = Utc::now() - chrono::Duration::seconds(200);
            conn.execute(
                "UPDATE document_sessions SET updated_at = ?1 WHERE session_id = ?2",
                params![stalled_time.to_rfc3339(), session_id],
            )
            .unwrap();
        }

        // Now it should fail
        let failed = ctx.manager.fail_stalled_sessions(100).unwrap();
        assert_eq!(failed, 1);

        let status = ctx.manager.get_status(session_id).unwrap();
        assert_eq!(status.status, SessionStatus::Failed);
        assert!(status.error.unwrap().contains("timed out"));
    }

    #[test]
    fn test_session_status_serialization() {
        assert_eq!(
            SessionStatus::parse_str("uploading"),
            SessionStatus::Uploading
        );
        assert_eq!(
            SessionStatus::parse_str("processing"),
            SessionStatus::Processing
        );
        assert_eq!(
            SessionStatus::parse_str("completed"),
            SessionStatus::Completed
        );
        assert_eq!(SessionStatus::parse_str("failed"), SessionStatus::Failed);
        assert_eq!(
            SessionStatus::parse_str("cancelled"),
            SessionStatus::Cancelled
        );
        assert_eq!(
            SessionStatus::parse_str("unknown"),
            SessionStatus::Uploading
        );
    }

    #[test]
    fn test_estimate_processing_time() {
        let time = DocumentSessionManager::estimate_processing_time(1024, 2);
        assert!(time > 0.0);
        assert!(time < 100.0);
    }
}
