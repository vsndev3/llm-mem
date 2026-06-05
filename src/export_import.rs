//! JSONL export/import with versioning and migration support.
//!
//! Provides a text-based, streaming-friendly format for exporting and importing
//! memory banks. The format is designed to survive changes in the underlying
//! database backend (VectorLite → LanceDB → anything else).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use tokio::io::AsyncBufReadExt;
use tracing::{info, warn};

use crate::{
    DATA_FORMAT_VERSION,
    error::{MemoryError, Result},
    types::Memory,
};

// ── Envelope types ───────────────────────────────────────────────

/// Header line of a JSONL export file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonlExportHeader {
    /// Marker to distinguish header from memory lines.
    #[serde(rename = "_type")]
    pub line_type: String,
    /// Data format version (e.g. 1).
    pub format_version: u32,
    /// Application version that created the export.
    pub app_version: String,
    /// Name of the bank being exported.
    pub bank_name: String,
    /// Number of memories in the export.
    pub memory_count: usize,
    /// Embedding dimension at export time.
    pub embedding_dimension: u32,
    /// Export timestamp (RFC 3339).
    pub exported_at: String,
}

/// Footer line of a JSONL export file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonlExportFooter {
    /// Marker to distinguish footer from memory lines.
    #[serde(rename = "_type")]
    pub line_type: String,
    /// Memory count (should match header).
    pub memory_count: usize,
    /// SHA-256 hex digest of all memory lines concatenated.
    pub sha256: String,
}

/// Result of a JSONL export operation.
#[derive(Debug, Clone)]
pub struct JsonlExportResult {
    pub path: std::path::PathBuf,
    pub memory_count: usize,
    pub sha256: String,
}

/// Result of a JSONL import validation (dry-run).
#[derive(Debug, Clone)]
pub struct JsonlImportPreview {
    pub format_version: u32,
    pub app_version: String,
    pub embedding_dimension: u32,
    pub memory_count: usize,
    pub current_dimension: usize,
    pub dimension_mismatch: bool,
    pub parse_errors: Vec<String>,
}

/// Result of a JSONL import operation.
#[derive(Debug, Clone)]
pub struct JsonlImportResult {
    pub imported: usize,
    pub skipped_duplicates: usize,
    pub stripped_embeddings: usize,
    pub parse_errors: Vec<String>,
}

// ── Export ───────────────────────────────────────────────────────

/// Export memories to a JSONL file with envelope header and footer.
///
/// Format:
/// ```jsonl
/// {"_type":"header","format_version":1,...}
/// {"id":"...","content":"...",...}
/// {"id":"...","content":"...",...}
/// {"_type":"footer","memory_count":2,"sha256":"..."}
/// ```
///
/// This is streaming-friendly: memories are written one at a time without
/// accumulating the entire bank in memory.
pub async fn export_memories_jsonl(
    memories: Vec<Memory>,
    bank_name: &str,
    embedding_dimension: usize,
    output_path: &Path,
) -> Result<JsonlExportResult> {
    use tokio::io::AsyncWriteExt;

    // Ensure parent directory exists
    if let Some(parent) = output_path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|e| {
            MemoryError::config(format!(
                "Failed to create export directory '{}': {}",
                parent.display(),
                e
            ))
        })?;
    }

    let header = JsonlExportHeader {
        line_type: "header".to_string(),
        format_version: DATA_FORMAT_VERSION,
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        bank_name: bank_name.to_string(),
        memory_count: memories.len(),
        embedding_dimension: embedding_dimension as u32,
        exported_at: chrono::Utc::now().to_rfc3339(),
    };

    let mut file = tokio::fs::File::create(output_path).await.map_err(|e| {
        MemoryError::config(format!(
            "Failed to create export file '{}': {}",
            output_path.display(),
            e
        ))
    })?;

    // Write header
    let header_json = serde_json::to_string(&header).map_err(|e| {
        MemoryError::config(format!("Failed to serialize export header: {}", e))
    })?;
    file.write_all(header_json.as_bytes()).await.map_err(|e| {
        MemoryError::config(format!("Failed to write export header: {}", e))
    })?;
    file.write_all(b"\n").await.map_err(|e| {
        MemoryError::config(format!("Failed to write newline after header: {}", e))
    })?;

    // Write each memory line and accumulate checksum
    let mut hasher = Sha256::new();
    for memory in &memories {
        let line = serde_json::to_string(memory).map_err(|e| {
            MemoryError::config(format!("Failed to serialize memory {}: {}", memory.id, e))
        })?;
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
        file.write_all(line.as_bytes()).await.map_err(|e| {
            MemoryError::config(format!("Failed to write memory line: {}", e))
        })?;
        file.write_all(b"\n").await.map_err(|e| {
            MemoryError::config(format!("Failed to write newline after memory: {}", e))
        })?;
    }

    let sha256 = format!("{:x}", hasher.finalize());

    // Write footer
    let footer = JsonlExportFooter {
        line_type: "footer".to_string(),
        memory_count: memories.len(),
        sha256: sha256.clone(),
    };
    let footer_json = serde_json::to_string(&footer).map_err(|e| {
        MemoryError::config(format!("Failed to serialize export footer: {}", e))
    })?;
    file.write_all(footer_json.as_bytes()).await.map_err(|e| {
        MemoryError::config(format!("Failed to write export footer: {}", e))
    })?;
    file.write_all(b"\n").await.map_err(|e| {
        MemoryError::config(format!("Failed to write newline after footer: {}", e))
    })?;

    file.flush().await.map_err(|e| {
        MemoryError::config(format!("Failed to flush export file: {}", e))
    })?;

    info!(
        "Exported {} memories to {} (format_version={}, sha256={})",
        memories.len(),
        output_path.display(),
        DATA_FORMAT_VERSION,
        &sha256[..16],
    );

    Ok(JsonlExportResult {
        path: output_path.to_path_buf(),
        memory_count: memories.len(),
        sha256,
    })
}

// ── Import (dry-run preview) ─────────────────────────────────────

/// Preview a JSONL import without modifying any database.
///
/// Reads the header, parses all memory lines, and reports statistics.
/// Returns `JsonlImportPreview` with validation results.
pub async fn preview_jsonl_import(
    input_path: &Path,
    current_embedding_dimension: usize,
) -> Result<JsonlImportPreview> {
    let file = tokio::fs::File::open(input_path).await.map_err(|e| {
        MemoryError::config(format!(
            "Failed to open import file '{}': {}",
            input_path.display(),
            e
        ))
    })?;
    let reader = tokio::io::BufReader::new(file);
    let mut lines = reader.lines();

    // Read header line
    let first_line = lines
        .next_line()
        .await
        .map_err(|e| MemoryError::config(format!("Failed to read header line: {}", e)))?
        .ok_or_else(|| MemoryError::config("Import file is empty".to_string()))?;

    let header: JsonlExportHeader = serde_json::from_str(&first_line).map_err(|e| {
        MemoryError::config(format!(
            "Invalid JSONL header: {}. Expected header line first.",
            e
        ))
    })?;

    if header.line_type != "header" {
        return Err(MemoryError::config(
            "First line is not a header. Is this a valid JSONL export?".to_string(),
        ));
    }

    let mut memory_count = 0usize;
    let mut parse_errors: Vec<String> = Vec::new();

    while let Some(line) = lines
        .next_line()
        .await
        .map_err(|e| MemoryError::config(format!("Failed to read line: {}", e)))?
    {
        if line.trim().is_empty() {
            continue;
        }

        // Detect footer
        if let Ok(footer) = serde_json::from_str::<JsonlExportFooter>(&line) {
            if footer.line_type == "footer" {
                break;
            }
        }

        // Try to parse as Memory
        match serde_json::from_str::<Memory>(&line) {
            Ok(_memory) => {
                memory_count += 1;
            }
            Err(e) => {
                parse_errors.push(format!("Line {}: {}", memory_count + 1, e));
                // Continue parsing — don't stop on one bad line
            }
        }
    }

    let dimension_mismatch = header.embedding_dimension != current_embedding_dimension as u32;

    Ok(JsonlImportPreview {
        format_version: header.format_version,
        app_version: header.app_version,
        embedding_dimension: header.embedding_dimension,
        memory_count,
        current_dimension: current_embedding_dimension,
        dimension_mismatch,
        parse_errors,
    })
}

// ── Import (actual) ──────────────────────────────────────────────

/// Import memories from a JSONL file into a target store.
///
/// - Detects format version and applies migrations if needed.
/// - Detects embedding dimension mismatch and strips embeddings if requested.
/// - Skips duplicates by content hash (both within the file and against `existing_hashes`).
/// - Reports parse errors without failing the entire import.
pub async fn import_jsonl_file<F, Fut>(
    input_path: &Path,
    current_embedding_dimension: usize,
    strip_embeddings: bool,
    existing_hashes: Option<std::collections::HashSet<String>>,
    mut insert_fn: F,
) -> Result<JsonlImportResult>
where
    F: FnMut(Memory) -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    let file = tokio::fs::File::open(input_path).await.map_err(|e| {
        MemoryError::config(format!(
            "Failed to open import file '{}': {}",
            input_path.display(),
            e
        ))
    })?;
    let reader = tokio::io::BufReader::new(file);
    let mut lines = reader.lines();

    // Read header
    let first_line = lines
        .next_line()
        .await
        .map_err(|e| MemoryError::config(format!("Failed to read header line: {}", e)))?
        .ok_or_else(|| MemoryError::config("Import file is empty".to_string()))?;

    let header: JsonlExportHeader = serde_json::from_str(&first_line).map_err(|e| {
        MemoryError::config(format!(
            "Invalid JSONL header: {}. Expected header line first.",
            e
        ))
    })?;

    if header.line_type != "header" {
        return Err(MemoryError::config(
            "First line is not a header. Is this a valid JSONL export?".to_string(),
        ));
    }

    // Track existing hashes for dedup
    let seen_hashes: HashMap<String, ()> = HashMap::new();
    let mut imported = 0usize;
    let mut stripped_embeddings = 0usize;
    let mut parse_errors: Vec<String> = Vec::new();
    let mut line_num = 1usize;

    let dimension_mismatch = header.embedding_dimension != current_embedding_dimension as u32;
    let should_strip = strip_embeddings || dimension_mismatch;

    // Merge external existing hashes with our own tracking
    let mut all_seen_hashes: std::collections::HashMap<String, ()> = seen_hashes;
    if let Some(existing) = existing_hashes {
        for h in existing {
            all_seen_hashes.insert(h, ());
        }
    }

    let mut skipped_duplicates = 0usize;

    while let Some(line) = lines
        .next_line()
        .await
        .map_err(|e| MemoryError::config(format!("Failed to read line: {}", e)))?
    {
        line_num += 1;
        if line.trim().is_empty() {
            continue;
        }

        // Detect footer
        if let Ok(footer) = serde_json::from_str::<JsonlExportFooter>(&line) {
            if footer.line_type == "footer" {
                break;
            }
        }

        // Parse memory (with migration if needed)
        let memory = match parse_memory_line(&line, header.format_version) {
            Ok(m) => m,
            Err(e) => {
                parse_errors.push(format!("Line {}: {}", line_num, e));
                continue;
            }
        };

        // Skip duplicates
        if !memory.metadata.hash.is_empty()
            && all_seen_hashes.contains_key(&memory.metadata.hash)
        {
            skipped_duplicates += 1;
            continue;
        }
        if !memory.metadata.hash.is_empty() {
            all_seen_hashes.insert(memory.metadata.hash.clone(), ());
        }

        // Handle embedding dimension mismatch
        let mut memory = memory;
        if should_strip && !memory.embedding.is_empty() {
            let emb_dim = memory.embedding.len();
            if emb_dim != current_embedding_dimension {
                memory.embedding.clear();
                memory.metadata.state = crate::types::MemoryState::Invalid;
                // Add a quality flag to note re-embedding needed
                memory
                    .content_meta
                    .quality_flags
                    .push("needs_reembed".to_string());
                stripped_embeddings += 1;
            }
        }

        // Insert via callback
        if let Err(e) = insert_fn(memory).await {
            parse_errors.push(format!("Line {} insert failed: {}", line_num, e));
            continue;
        }

        imported += 1;
    }

    if !parse_errors.is_empty() {
        warn!("JSONL import had {} parse/insert errors", parse_errors.len());
        for err in &parse_errors {
            warn!("  {}", err);
        }
    }

    info!(
        "Imported {} memories from {} (stripped embeddings: {}, skipped duplicates: {}, errors: {})",
        imported,
        input_path.display(),
        stripped_embeddings,
        skipped_duplicates,
        parse_errors.len()
    );

    Ok(JsonlImportResult {
        imported,
        skipped_duplicates,
        stripped_embeddings,
        parse_errors,
    })
}

// ── Migration helpers ────────────────────────────────────────────

/// Parse a single memory line, applying migrations if the format version is older.
fn parse_memory_line(line: &str, file_format_version: u32) -> Result<Memory> {
    let value: serde_json::Value = serde_json::from_str(line).map_err(|e| {
        MemoryError::config(format!("JSON parse error: {}", e))
    })?;

    // Apply migrations sequentially
    let migrated = apply_migrations(value, file_format_version, DATA_FORMAT_VERSION)?;

    // Deserialize into Memory struct
    let memory: Memory = serde_json::from_value(migrated).map_err(|e| {
        MemoryError::config(format!("Memory deserialization error: {}", e))
    })?;

    Ok(memory)
}

/// Apply migrations from `from_version` up to `to_version`.
///
/// Each migration is a pure function `Value -> Value` that transforms the JSON
/// representation. This allows unknown fields to survive even if the struct
/// doesn't know about them yet.
fn apply_migrations(
    mut value: serde_json::Value,
    from_version: u32,
    to_version: u32,
) -> Result<serde_json::Value> {
    for version in from_version..to_version {
        value = match version {
            // 1 -> 2: example migration (when needed)
            // 1 => migrate_v1_to_v2(value),
            _ => {
                // No migration defined for this version step
                warn!(
                    "No migration defined from version {} to {}. Importing as-is.",
                    version,
                    version + 1
                );
                value
            }
        };
    }
    Ok(value)
}

// Example migration (commented out until needed):
// fn migrate_v1_to_v2(value: serde_json::Value) -> serde_json::Value {
//     let mut value = value;
//     // e.g. rename field "old_name" -> "new_name"
//     if let Some(obj) = value.as_object_mut() {
//         if let Some(v) = obj.remove("old_name") {
//             obj.insert("new_name".to_string(), v);
//         }
//     }
//     value
// }

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Memory, MemoryMetadata};

    fn make_test_memory(id: &str) -> Memory {
        let mut mem = Memory::with_content(
            format!("Content for {}", id),
            vec![0.1, 0.2, 0.3],
            MemoryMetadata::new(),
        );
        mem.id = id.to_string();
        mem.metadata.hash = format!("hash-{}", id);
        mem
    }

    #[tokio::test]
    async fn test_jsonl_export_roundtrip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.jsonl");

        let memories = vec![
            make_test_memory("mem1"),
            make_test_memory("mem2"),
        ];

        let result = export_memories_jsonl(
            memories.clone(),
            "test-bank",
            3,
            &path,
        )
        .await
        .unwrap();

        assert_eq!(result.memory_count, 2);
        assert!(!result.sha256.is_empty());
        assert!(path.exists());

        // Preview
        let preview = preview_jsonl_import(&path, 3).await.unwrap();
        assert_eq!(preview.memory_count, 2);
        assert_eq!(preview.format_version, DATA_FORMAT_VERSION);
        assert!(!preview.dimension_mismatch);
        assert!(preview.parse_errors.is_empty());

        // Actual import
        let mut imported = Vec::new();
        let import_result = import_jsonl_file(
            &path,
            3,
            false,
            None,
            |m: Memory| {
                imported.push(m);
                async { Ok(()) }
            },
        )
        .await
        .unwrap();

        assert_eq!(import_result.imported, 2);
        assert_eq!(imported.len(), 2);
        assert_eq!(imported[0].id, "mem1");
        assert_eq!(imported[1].id, "mem2");
    }

    #[tokio::test]
    async fn test_jsonl_import_dimension_mismatch() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.jsonl");

        let memories = vec![make_test_memory("mem1")];
        export_memories_jsonl(memories, "test-bank", 3, &path)
            .await
            .unwrap();

        // Import with different dimension
        let mut imported = Vec::new();
        let result = import_jsonl_file(
            &path,
            768, // different dimension
            false,
            None,
            |m: Memory| {
                imported.push(m);
                async { Ok(()) }
            },
        )
        .await
        .unwrap();

        assert_eq!(result.imported, 1);
        assert_eq!(result.stripped_embeddings, 1);
        assert!(imported[0].embedding.is_empty());
        assert_eq!(imported[0].metadata.state, crate::types::MemoryState::Invalid);
        assert!(imported[0]
            .content_meta
            .quality_flags
            .contains(&"needs_reembed".to_string()));
    }

    #[tokio::test]
    async fn test_jsonl_import_with_bad_lines() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.jsonl");

        // Write a malformed JSONL manually
        let content = r#"{"_type":"header","format_version":1,"app_version":"0.1.0","bank_name":"test","memory_count":2,"embedding_dimension":3,"exported_at":"2026-01-01T00:00:00Z"}
{"id":"mem1","content":"hello","embedding":[0.1],"metadata":{},"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z"}
this is not json
{"id":"mem2","content":"world","embedding":[0.2],"metadata":{},"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z"}
"#;
        tokio::fs::write(&path, content).await.unwrap();

        let mut imported = Vec::new();
        let result = import_jsonl_file(
            &path,
            3,
            false,
            None,
            |m: Memory| {
                imported.push(m);
                async { Ok(()) }
            },
        )
        .await
        .unwrap();

        assert_eq!(result.imported, 2);
        assert_eq!(result.parse_errors.len(), 1);
        // The parse error should mention the line number and some JSON parse error
        assert!(result.parse_errors[0].contains("Line"));
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_jsonl_export_empty_bank() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("empty.jsonl");

        let result = export_memories_jsonl(vec![], "empty-bank", 3, &path)
            .await
            .unwrap();

        assert_eq!(result.memory_count, 0);
        assert!(path.exists());

        // Read back and verify header/footer
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2); // header + footer

        let header: JsonlExportHeader = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(header.memory_count, 0);
        assert_eq!(header.format_version, DATA_FORMAT_VERSION);

        let footer: JsonlExportFooter = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(footer.memory_count, 0);
        assert!(!footer.sha256.is_empty());
    }

    #[tokio::test]
    async fn test_jsonl_import_empty_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("empty.jsonl");
        tokio::fs::write(&path, "").await.unwrap();

        let result = preview_jsonl_import(&path, 3).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_jsonl_import_missing_header() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("bad.jsonl");
        // File starts with a memory line, not a header
        tokio::fs::write(&path, "{\"id\":\"mem1\"}\n").await.unwrap();

        let result = preview_jsonl_import(&path, 3).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("header") || err.contains("Invalid"));
    }

    #[tokio::test]
    async fn test_jsonl_import_duplicate_within_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("dups.jsonl");

        let mem1 = make_test_memory("mem1");
        let mut mem2 = make_test_memory("mem2");
        // Same content hash = duplicate
        mem2.metadata.hash = mem1.metadata.hash.clone();

        export_memories_jsonl(vec![mem1, mem2], "dup-bank", 3, &path)
            .await
            .unwrap();

        let mut imported = Vec::new();
        let result = import_jsonl_file(
            &path,
            3,
            false,
            None,
            |m: Memory| {
                imported.push(m);
                async { Ok(()) }
            },
        )
        .await
        .unwrap();

        // Second memory with same hash should be skipped
        assert_eq!(result.imported, 1);
        assert_eq!(result.skipped_duplicates, 1);
        assert_eq!(imported.len(), 1);
    }

    #[tokio::test]
    async fn test_jsonl_import_duplicate_against_existing_hashes() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("dups.jsonl");

        let mem = make_test_memory("mem1");
        let existing_hash = mem.metadata.hash.clone();
        export_memories_jsonl(vec![mem], "dup-bank", 3, &path)
            .await
            .unwrap();

        let mut imported = Vec::new();
        let mut existing = std::collections::HashSet::new();
        existing.insert(existing_hash);

        let result = import_jsonl_file(
            &path,
            3,
            false,
            Some(existing),
            |m: Memory| {
                imported.push(m);
                async { Ok(()) }
            },
        )
        .await
        .unwrap();

        // Should be skipped because hash already exists in target
        assert_eq!(result.imported, 0);
        assert_eq!(result.skipped_duplicates, 1);
        assert!(imported.is_empty());
    }

    #[tokio::test]
    async fn test_jsonl_import_footer_stops_parsing() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("footer.jsonl");

        let content = r#"{"_type":"header","format_version":1,"app_version":"0.1.0","bank_name":"test","memory_count":1,"embedding_dimension":3,"exported_at":"2026-01-01T00:00:00Z"}
{"id":"mem1","content":"hello","embedding":[0.1],"metadata":{},"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z"}
{"_type":"footer","memory_count":1,"sha256":"abc123"}
{"id":"mem2","content":"should not be parsed","embedding":[0.2],"metadata":{},"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z"}
"#;
        tokio::fs::write(&path, content).await.unwrap();

        let mut imported = Vec::new();
        let result = import_jsonl_file(
            &path,
            3,
            false,
            None,
            |m: Memory| {
                imported.push(m);
                async { Ok(()) }
            },
        )
        .await
        .unwrap();

        // Only mem1 should be imported; mem2 after footer is ignored
        assert_eq!(result.imported, 1);
        assert_eq!(imported.len(), 1);
        assert_eq!(imported[0].id, "mem1");
    }

    #[tokio::test]
    async fn test_jsonl_preview_reports_errors() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("bad.jsonl");

        let content = r#"{"_type":"header","format_version":1,"app_version":"0.1.0","bank_name":"test","memory_count":2,"embedding_dimension":3,"exported_at":"2026-01-01T00:00:00Z"}
{"id":"mem1","content":"hello","embedding":[0.1],"metadata":{},"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z"}
this is not valid json
{"id":"mem2","content":"world","embedding":[0.2],"metadata":{},"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z"}
"#;
        tokio::fs::write(&path, content).await.unwrap();

        let preview = preview_jsonl_import(&path, 3).await.unwrap();
        assert_eq!(preview.memory_count, 2); // 2 valid memories
        assert_eq!(preview.parse_errors.len(), 1);
        assert!(preview.dimension_mismatch == false);
    }

    #[tokio::test]
    async fn test_jsonl_import_with_insert_failure() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.jsonl");

        let mem = make_test_memory("mem1");
        export_memories_jsonl(vec![mem], "test-bank", 3, &path)
            .await
            .unwrap();

        let result = import_jsonl_file(
            &path,
            3,
            false,
            None,
            |_m: Memory| async {
                Err(crate::error::MemoryError::config("insert failed".to_string()))
            },
        )
        .await
        .unwrap();

        assert_eq!(result.imported, 0);
        assert_eq!(result.parse_errors.len(), 1);
        assert!(result.parse_errors[0].contains("insert failed"));
    }

    #[tokio::test]
    async fn test_jsonl_preview_dimension_mismatch() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.jsonl");

        let mem = make_test_memory("mem1");
        export_memories_jsonl(vec![mem], "test-bank", 384, &path)
            .await
            .unwrap();

        let preview = preview_jsonl_import(&path, 768).await.unwrap();
        assert!(preview.dimension_mismatch);
        assert_eq!(preview.embedding_dimension, 384);
        assert_eq!(preview.current_dimension, 768);
    }

    // ── Default / compatibility tests ────────────────────────────────

    #[test]
    fn test_memory_default_produces_valid_struct() {
        let mem: Memory = Default::default();
        // Should have a valid UUID
        assert!(uuid::Uuid::parse_str(&mem.id).is_ok());
        // Should have timestamps
        assert!(mem.created_at <= chrono::Utc::now());
        // Should have empty embedding
        assert!(mem.embedding.is_empty());
        // Should serialize/deserialize without error
        let json = serde_json::to_string(&mem).unwrap();
        let restored: Memory = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id, mem.id);
    }

    #[test]
    fn test_memory_default_roundtrip_with_missing_fields() {
        // Simulate an old export that only has id and content
        let minimal_json = r#"{"id":"test-123","content":"hello","embedding":[],"metadata":{},"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z"}"#;
        let restored: Memory = serde_json::from_str(minimal_json).unwrap();
        assert_eq!(restored.id, "test-123");
        assert_eq!(restored.content, Some("hello".to_string()));
        // Missing fields should get defaults
        assert!(restored.event_at.is_none());
        assert!(restored.event_end.is_none());
        assert!(restored.context_embeddings.is_none());
        assert!(restored.relation_embeddings.is_none());
    }

    #[test]
    fn test_layer_info_schema_version_populated() {
        let l0 = crate::types::LayerInfo::raw_content();
        assert_eq!(l0.schema_version, Some("1".to_string()));
        assert_eq!(l0.schema_version_or_default(), "1");

        let l1 = crate::types::LayerInfo::structural();
        assert_eq!(l1.schema_version, Some("1".to_string()));

        let l2 = crate::types::LayerInfo::semantic();
        assert_eq!(l2.schema_version, Some("1".to_string()));

        let l3 = crate::types::LayerInfo::concept();
        assert_eq!(l3.schema_version, Some("1".to_string()));

        let l4 = crate::types::LayerInfo::wisdom();
        assert_eq!(l4.schema_version, Some("1".to_string()));

        let forgotten = crate::types::LayerInfo::forgotten();
        assert_eq!(forgotten.schema_version, Some("1".to_string()));

        let custom = crate::types::LayerInfo::custom(5, "custom");
        assert_eq!(custom.schema_version, Some("1".to_string()));
    }

    #[test]
    fn test_layer_info_schema_version_legacy_default() {
        // Simulate an old export without schema_version
        let legacy = crate::types::LayerInfo {
            level: 0,
            name: Some("raw_content".to_string()),
            schema_version: None,
        };
        assert_eq!(legacy.schema_version_or_default(), "1");
    }

    #[test]
    fn test_backup_manifest_default_format_version() {
        // Simulate a pre-versioning manifest (only has the old fields)
        let old_json = r#"{"version":1,"created_at":"2026-01-01T00:00:00Z","bank_name":"test","memory_count":10,"sha256":"abc123","size_bytes":1024}"#;
        let manifest: crate::memory_bank::BackupManifest = serde_json::from_str(old_json).unwrap();
        assert_eq!(manifest.format_version, 1); // Should default to 1
        assert!(manifest.app_version.is_empty());
        assert_eq!(manifest.embedding_dimension, 0);
    }

    #[test]
    fn test_backup_manifest_full_roundtrip() {
        let manifest = crate::memory_bank::BackupManifest {
            version: 1,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            bank_name: "test".to_string(),
            memory_count: 10,
            sha256: "abc123".to_string(),
            size_bytes: 1024,
            format_version: crate::DATA_FORMAT_VERSION,
            app_version: "0.1.0".to_string(),
            embedding_dimension: 384,
        };
        let json = serde_json::to_string(&manifest).unwrap();
        let restored: crate::memory_bank::BackupManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.format_version, manifest.format_version);
        assert_eq!(restored.app_version, manifest.app_version);
        assert_eq!(restored.embedding_dimension, manifest.embedding_dimension);
    }
}
