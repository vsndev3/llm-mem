//! Content schema with provenance tracking
//!
//! This module defines the content architecture that separates:
//! - User-provided content (immutable)
//! - Derived data (AI-generated, with provenance)
//! - Relations (links to other memories, with provenance)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// External content reference (file, URL, book, database, etc.)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContentPointer {
    /// URI: file://, http://, book://, database://, etc.
    pub uri: String,
    /// Type: "file", "url", "book", "database", etc.
    pub pointer_type: String,
    /// Optional location within source: "bytes:1024-2048", "chapter:3", "line:100-200"
    pub location: Option<String>,
}

impl ContentPointer {
    pub fn new(uri: impl Into<String>, pointer_type: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            pointer_type: pointer_type.into(),
            location: None,
        }
    }

    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Create a file pointer
    pub fn file(path: impl Into<String>) -> Self {
        Self {
            uri: format!("file://{}", path.into()),
            pointer_type: "file".to_string(),
            location: None,
        }
    }

    /// Create a URL pointer
    pub fn url(url: impl Into<String>) -> Self {
        Self {
            uri: url.into(),
            pointer_type: "url".to_string(),
            location: None,
        }
    }
}

/// Metadata for content (provenance + input characteristics)
///
/// This is set when content is first stored and should generally not change.
/// Exceptions: quality_flags can be updated as new validations occur.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ContentMeta {
    // ── Provenance ──
    /// Who provided this content? "user", "user:alice", "agent:bot", "import:file"
    pub provided_by: Option<String>,
    /// When was content provided?
    pub provided_at: Option<DateTime<Utc>>,
    /// Original source description (if imported)
    pub source: Option<String>,

    // ── Content characteristics ──
    /// Content type classification (free-form): "factual", "opinion", "conversation", "chunk", etc.
    pub content_type: Option<String>,
    /// Quality/validation flags (free-form): "factual_error", "outdated", "needs_review"
    pub quality_flags: Vec<String>,
    /// Checksum/hash of content for deduplication (sha256)
    pub checksum: Option<String>,

    // ── Chunking info (if this is a chunk of larger content) ──
    /// Reference to parent content if this is a chunk
    pub parent_pointer: Option<ContentPointer>,
    /// Chunk sequence number (0-indexed)
    pub chunk_index: Option<usize>,
    /// Total number of chunks in the series
    pub chunk_total: Option<usize>,

    // ── Extensibility ──
    /// Additional user-provided metadata
    pub custom: HashMap<String, serde_json::Value>,
}

impl ContentMeta {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_provided_by(mut self, provided_by: impl Into<String>) -> Self {
        self.provided_by = Some(provided_by.into());
        self
    }

    pub fn with_content_type(mut self, content_type: impl Into<String>) -> Self {
        self.content_type = Some(content_type.into());
        self
    }

    pub fn with_quality_flag(mut self, flag: impl Into<String>) -> Self {
        self.quality_flags.push(flag.into());
        self
    }

    pub fn with_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.checksum = Some(checksum.into());
        self
    }

    pub fn with_chunk_info(
        mut self,
        parent: Option<ContentPointer>,
        index: usize,
        total: usize,
    ) -> Self {
        self.parent_pointer = parent;
        self.chunk_index = Some(index);
        self.chunk_total = Some(total);
        self
    }

    /// Compute SHA256 checksum of content
    pub fn compute_checksum(content: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Metadata for derived data (provenance chain)
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DerivedMeta {
    // ── Provenance chain ──
    /// Which memory IDs this was derived from
    pub source_memory_ids: Vec<Uuid>,
    /// When was this derived?
    pub derived_at: DateTime<Utc>,
    /// What process created this? "llm:summarize", "embedding:v1", "validator:math"
    pub derived_by: String,
    /// Model/version used (if AI-generated): "qwen2.5-1.5b", "gpt-4o-mini"
    pub model: Option<String>,

    // ── Quality ──
    /// Confidence in this derived data (0.0-1.0)
    pub confidence: Option<f32>,
    /// Warnings about this derivation
    pub warnings: Vec<String>,

    // ── Versioning for backtracking ──
    /// Version number (incremented on regeneration)
    pub version: usize,
    /// If this entry was replaced, the ID of the replacement
    pub superseded_by: Option<Uuid>,

    // ── Extensibility ──
    pub custom: HashMap<String, serde_json::Value>,
}

impl DerivedMeta {
    pub fn new(derived_by: impl Into<String>) -> Self {
        Self {
            derived_by: derived_by.into(),
            derived_at: Utc::now(),
            version: 1,
            ..Default::default()
        }
    }

    pub fn with_source_memory_ids(mut self, ids: Vec<Uuid>) -> Self {
        self.source_memory_ids = ids;
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warnings.push(warning.into());
        self
    }
}

/// A derived data entry with value + provenance metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DerivedEntry {
    /// The actual derived value (summary, keywords, embedding, fact_check, etc.)
    pub value: serde_json::Value,
    /// Provenance metadata
    pub meta: DerivedMeta,
}

impl DerivedEntry {
    pub fn new(value: serde_json::Value, meta: DerivedMeta) -> Self {
        Self { value, meta }
    }

    /// Create a summary entry
    pub fn summary(summary: impl Into<String>, meta: DerivedMeta) -> Self {
        Self {
            value: serde_json::Value::String(summary.into()),
            meta,
        }
    }

    /// Create a keywords entry
    pub fn keywords(keywords: Vec<String>, meta: DerivedMeta) -> Self {
        Self {
            value: serde_json::to_value(&keywords).unwrap(),
            meta,
        }
    }

    /// Create an embedding entry
    pub fn embedding(vec: Vec<f32>, meta: DerivedMeta) -> Self {
        Self {
            value: serde_json::to_value(&vec).unwrap(),
            meta,
        }
    }
}

/// Metadata for relations (provenance + quality)
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct RelationMeta {
    /// When was this relation created?
    pub created_at: DateTime<Utc>,
    /// Who/what created this? "user", "llm:auto-link", "system:dedup"
    pub created_by: String,
    /// Model used (if AI-generated)
    pub model: Option<String>,
    /// Confidence in this relation (0.0-1.0)
    pub confidence: Option<f32>,
    /// Additional metadata
    pub custom: HashMap<String, serde_json::Value>,
}

impl RelationMeta {
    pub fn new(created_by: impl Into<String>) -> Self {
        Self {
            created_at: Utc::now(),
            created_by: created_by.into(),
            ..Default::default()
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }
}

/// A relation entry linking to other memories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RelationEntry {
    /// Target memory IDs
    pub target_ids: Vec<Uuid>,
    /// Relation strength (0.0-1.0)
    pub strength: Option<f32>,
    /// Relation metadata (provenance, confidence)
    pub meta: RelationMeta,
}

impl RelationEntry {
    pub fn new(target_ids: Vec<Uuid>, strength: Option<f32>, meta: RelationMeta) -> Self {
        Self {
            target_ids,
            strength,
            meta,
        }
    }

    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = Some(strength.clamp(0.0, 1.0));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_pointer_new() {
        let ptr = ContentPointer::new("file:///test.txt", "file");
        assert_eq!(ptr.uri, "file:///test.txt");
        assert_eq!(ptr.pointer_type, "file");
        assert!(ptr.location.is_none());
    }

    #[test]
    fn test_content_pointer_with_location() {
        let ptr = ContentPointer::file("/books/hobbit.txt")
            .with_location("chapter:3");
        assert_eq!(ptr.uri, "file:///books/hobbit.txt");
        assert_eq!(ptr.pointer_type, "file");
        assert_eq!(ptr.location, Some("chapter:3".to_string()));
    }

    #[test]
    fn test_content_pointer_url() {
        let ptr = ContentPointer::url("https://example.com/page");
        assert_eq!(ptr.uri, "https://example.com/page");
        assert_eq!(ptr.pointer_type, "url");
    }

    #[test]
    fn test_content_meta_default() {
        let meta = ContentMeta::new();
        assert!(meta.provided_by.is_none());
        assert!(meta.content_type.is_none());
        assert!(meta.quality_flags.is_empty());
        assert!(meta.checksum.is_none());
        assert!(meta.chunk_index.is_none());
    }

    #[test]
    fn test_content_meta_builder() {
        let meta = ContentMeta::new()
            .with_provided_by("user:alice")
            .with_content_type("factual")
            .with_quality_flag("needs_review")
            .with_checksum("abc123");

        assert_eq!(meta.provided_by, Some("user:alice".to_string()));
        assert_eq!(meta.content_type, Some("factual".to_string()));
        assert_eq!(meta.quality_flags, vec!["needs_review"]);
        assert_eq!(meta.checksum, Some("abc123".to_string()));
    }

    #[test]
    fn test_content_meta_chunk_info() {
        let parent = ContentPointer::file("/large.txt");
        let meta = ContentMeta::new()
            .with_chunk_info(Some(parent.clone()), 2, 10);

        assert_eq!(meta.chunk_index, Some(2));
        assert_eq!(meta.chunk_total, Some(10));
        assert_eq!(meta.parent_pointer, Some(parent));
    }

    #[test]
    fn test_content_meta_compute_checksum() {
        let checksum1 = ContentMeta::compute_checksum("hello");
        let checksum2 = ContentMeta::compute_checksum("hello");
        let checksum3 = ContentMeta::compute_checksum("world");

        assert_eq!(checksum1, checksum2);
        assert_ne!(checksum1, checksum3);
        assert_eq!(checksum1.len(), 64); // SHA256 hex length
    }

    #[test]
    fn test_derived_meta_new() {
        let meta = DerivedMeta::new("llm:summarize");
        assert_eq!(meta.derived_by, "llm:summarize");
        assert_eq!(meta.version, 1);
        assert!(meta.source_memory_ids.is_empty());
        assert!(meta.model.is_none());
    }

    #[test]
    fn test_derived_meta_builder() {
        let id = Uuid::new_v4();
        let meta = DerivedMeta::new("llm:extract")
            .with_source_memory_ids(vec![id])
            .with_model("qwen2.5-1.5b")
            .with_confidence(0.92)
            .with_warning("Low confidence on entity extraction");

        assert_eq!(meta.source_memory_ids, vec![id]);
        assert_eq!(meta.model, Some("qwen2.5-1.5b".to_string()));
        assert_eq!(meta.confidence, Some(0.92));
        assert_eq!(meta.warnings.len(), 1);
    }

    #[test]
    fn test_derived_entry_summary() {
        let meta = DerivedMeta::new("llm:summarize");
        let entry = DerivedEntry::summary("This is a summary", meta.clone());

        assert_eq!(entry.value.as_str(), Some("This is a summary"));
        assert_eq!(entry.meta.derived_by, "llm:summarize");
    }

    #[test]
    fn test_derived_entry_keywords() {
        let meta = DerivedMeta::new("llm:keywords");
        let entry = DerivedEntry::keywords(vec!["rust".into(), "memory".into()], meta);

        let keywords = entry.value.as_array().unwrap();
        assert_eq!(keywords.len(), 2);
        assert_eq!(keywords[0].as_str(), Some("rust"));
    }

    #[test]
    fn test_derived_entry_embedding() {
        let meta = DerivedMeta::new("embedding:v1");
        let entry = DerivedEntry::embedding(vec![0.1, 0.2, 0.3], meta);

        let emb = entry.value.as_array().unwrap();
        assert_eq!(emb.len(), 3);
        // Use approximate comparison for floating point
        assert!((emb[0].as_f64().unwrap() - 0.1).abs() < 0.0001);
    }

    #[test]
    fn test_relation_meta_new() {
        let meta = RelationMeta::new("llm:auto-link");
        assert_eq!(meta.created_by, "llm:auto-link");
        assert!(meta.model.is_none());
        assert!(meta.confidence.is_none());
    }

    #[test]
    fn test_relation_entry_new() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let meta = RelationMeta::new("user");
        let entry = RelationEntry::new(vec![id1, id2], Some(0.85), meta);

        assert_eq!(entry.target_ids.len(), 2);
        assert_eq!(entry.strength, Some(0.85));
        assert_eq!(entry.meta.created_by, "user");
    }

    #[test]
    fn test_derived_meta_versioning() {
        let meta1 = DerivedMeta::new("llm:v1");
        let mut meta2 = DerivedMeta::new("llm:v2");
        meta2.version = 2;
        meta2.superseded_by = Some(Uuid::new_v4());

        assert_eq!(meta1.version, 1);
        assert_eq!(meta2.version, 2);
        assert!(meta1.superseded_by.is_none());
        assert!(meta2.superseded_by.is_some());
    }
}
