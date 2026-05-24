use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResult {
    pub status: IngestStatus,
    pub session_id: String,
    pub l0_chunks: Vec<ChunkInfo>,
    pub relations: Vec<RelationInfo>,
    pub l1_abstractions: Vec<AbstractionInfo>,
    pub issues: Vec<IngestIssue>,
    pub format_hints_available: Vec<String>,
    pub warnings: Vec<String>,
    pub format: String,
    pub detected_mime: String,
    pub byte_size: u64,
    /// Vision/image description result, set when image content is ingested.
    /// None if no image was in the content.
    pub vision_status: Option<VisionStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionStatus {
    pub images_ingested: usize,
    pub descriptions_generated: usize,
    pub outcome: VisionOutcome,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum VisionOutcome {
    /// Vision description succeeded for all images
    Succeeded,
    /// Vision is not configured (vision_enabled = false, or no API key / mmproj)
    NotConfigured,
    /// Vision description was attempted but failed (LLM error, timeout, etc.)
    Failed,
    /// Vision description was not available for this image (unsupported format, too large, etc.)
    Unavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum IngestStatus {
    Success,
    Partial,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    pub id: String,
    pub memory_id: Option<String>,
    pub node_type: String,
    pub content_preview: String,
    pub char_count: usize,
    pub order: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationInfo {
    pub source_chunk_id: String,
    pub target_chunk_id: String,
    pub relation: String,
    pub strength: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractionInfo {
    pub id: Option<String>,
    pub memory_id: Option<String>,
    pub abstraction_type: String,
    pub source_chunk_ids: Vec<String>,
    pub layer: i32,
    pub content_preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestIssue {
    pub severity: IssueSeverity,
    pub message: String,
    pub suggestion: Option<String>,
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum IssueSeverity {
    Warning,
    Error,
    Blocking,
}

impl IngestResult {
    pub fn new(session_id: String, format: &str, mime: &str, byte_size: u64) -> Self {
        Self {
            status: IngestStatus::Success,
            session_id,
            l0_chunks: Vec::new(),
            relations: Vec::new(),
            l1_abstractions: Vec::new(),
            issues: Vec::new(),
            format_hints_available: Vec::new(),
            warnings: Vec::new(),
            format: format.to_string(),
            detected_mime: mime.to_string(),
            byte_size,
            vision_status: None,
        }
    }

    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warnings.push(warning.into());
        self
    }

    pub fn with_issue(mut self, issue: IngestIssue) -> Self {
        if issue.severity == IssueSeverity::Blocking {
            self.status = IngestStatus::Failed;
        } else if self.status == IngestStatus::Success {
            self.status = IngestStatus::Partial;
        }
        self.issues.push(issue);
        self
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.format_hints_available.push(hint.into());
        self
    }
}

impl IngestIssue {
    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: IssueSeverity::Warning,
            message: message.into(),
            suggestion: None,
            context: None,
        }
    }

    pub fn error(message: impl Into<String>, suggestion: impl Into<String>) -> Self {
        Self {
            severity: IssueSeverity::Error,
            message: message.into(),
            suggestion: Some(suggestion.into()),
            context: None,
        }
    }

    pub fn blocking(message: impl Into<String>, suggestion: impl Into<String>) -> Self {
        Self {
            severity: IssueSeverity::Blocking,
            message: message.into(),
            suggestion: Some(suggestion.into()),
            context: None,
        }
    }
}
