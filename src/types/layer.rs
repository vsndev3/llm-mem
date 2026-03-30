//! Layered memory architecture types
//!
//! This module defines the types for scalable layered memory organization:
//! - `LayerInfo`: Identifies a memory's position in the abstraction hierarchy
//! - `MemoryState`: Lifecycle state of a memory (Active, Forgotten, etc.)

use serde::{Deserialize, Serialize};

/// Identifies a memory's position in the abstraction hierarchy
///
/// Layers are scalable - there's no fixed maximum level. Higher levels
/// represent more abstract, synthesized knowledge.
///
/// # Layer Levels
/// - `0`: Raw content (user-provided, immutable)
/// - `1`: Structural (summaries, document structure)
/// - `2`: Semantic (cross-document links)
/// - `3`: Concept (domain concepts, theories)
/// - `4+`: Wisdom (mental models, paradigms)
/// - `-1`: Forgotten (soft-deleted, preserved for referential integrity)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayerInfo {
    /// Layer level: 0 = raw content, higher = more abstract
    /// Negative values reserved for special states (e.g., -1 = forgotten)
    pub level: i32,

    /// Optional layer name for human readability
    /// Examples: "raw_content", "structural", "semantic", "concept", "wisdom"
    pub name: Option<String>,

    /// Schema version for this layer (for migration purposes)
    pub schema_version: Option<String>,
}

impl LayerInfo {
    /// Create layer info for raw content (L0)
    pub fn raw_content() -> Self {
        Self {
            level: 0,
            name: Some("raw_content".to_string()),
            schema_version: None,
        }
    }

    /// Create layer info for structural abstractions (L1)
    pub fn structural() -> Self {
        Self {
            level: 1,
            name: Some("structural".to_string()),
            schema_version: None,
        }
    }

    /// Create layer info for semantic links (L2)
    pub fn semantic() -> Self {
        Self {
            level: 2,
            name: Some("semantic".to_string()),
            schema_version: None,
        }
    }

    /// Create layer info for concepts (L3)
    pub fn concept() -> Self {
        Self {
            level: 3,
            name: Some("concept".to_string()),
            schema_version: None,
        }
    }

    /// Create layer info for wisdom/paradigms (L4)
    pub fn wisdom() -> Self {
        Self {
            level: 4,
            name: Some("wisdom".to_string()),
            schema_version: None,
        }
    }

    /// Create layer info for forgotten memories
    ///
    /// Forgotten memories are soft-deleted: they preserve referential
    /// integrity for higher-layer abstractions that reference them.
    pub fn forgotten() -> Self {
        Self {
            level: -1,
            name: Some("forgotten".to_string()),
            schema_version: None,
        }
    }

    /// Create custom layer info
    pub fn custom(level: i32, name: impl Into<String>) -> Self {
        Self {
            level,
            name: Some(name.into()),
            schema_version: None,
        }
    }

    /// Check if this layer is in forgotten state
    pub fn is_forgotten(&self) -> bool {
        self.level < 0
    }

    /// Check if this is raw content layer (L0)
    pub fn is_raw_content(&self) -> bool {
        self.level == 0
    }

    /// Get the layer name or a default string representation
    pub fn name_or_default(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| format!("layer_{}", self.level))
    }
}

impl Default for LayerInfo {
    fn default() -> Self {
        Self::raw_content()
    }
}

/// Lifecycle state of a memory
///
/// Memories transition through states during their lifecycle:
/// - New memories start as `Active`
/// - During abstraction processing, they become `Processing`
/// - If validation fails, they become `Invalid`
/// - When deleted but referenced by higher layers, they become `Forgotten`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum MemoryState {
    /// Memory is active and searchable
    #[default]
    Active,

    /// Memory has been deleted but higher-layer abstractions still reference it
    ///
    /// Acts like "0" in mathematics - preserves structural integrity of the
    /// abstraction hierarchy. Forgotten memories:
    /// - Are excluded from normal searches
    /// - Can be queried explicitly with state filters
    /// - May be restored if all dependents are updated
    /// - Still occupy storage until no longer referenced
    Forgotten,

    /// Memory is being processed (intermediate state during abstraction)
    ///
    /// This state prevents race conditions during background processing.
    /// If a memory remains in this state too long, it should be reviewed.
    Processing,

    /// Memory has lost some (but not all) of its abstraction sources.
    ///
    /// Degraded memories remain searchable but carry reduced confidence.
    /// They track which sources were deleted via `forgotten_sources`.
    /// Once enough sources are lost (per-layer threshold), the memory
    /// transitions to `Forgotten`.
    Degraded,

    /// Memory failed validation or abstraction (requires review)
    ///
    /// Invalid memories are not searchable and should be either:
    /// - Corrected and reprocessed
    /// - Deleted (if no higher layers depend on them)
    /// - Marked as Forgotten (if higher layers depend on them)
    Invalid,
}

impl MemoryState {
    /// Check if this memory is active (includes Degraded — still searchable)
    pub fn is_active(&self) -> bool {
        matches!(self, MemoryState::Active | MemoryState::Degraded)
    }

    /// Check if this memory is degraded (lost some abstraction sources)
    pub fn is_degraded(&self) -> bool {
        matches!(self, MemoryState::Degraded)
    }

    /// Check if this memory is forgotten
    pub fn is_forgotten(&self) -> bool {
        matches!(self, MemoryState::Forgotten)
    }

    /// Check if this memory is being processed
    pub fn is_processing(&self) -> bool {
        matches!(self, MemoryState::Processing)
    }

    /// Check if this memory is invalid
    pub fn is_invalid(&self) -> bool {
        matches!(self, MemoryState::Invalid)
    }

    /// Get the string representation of the memory state
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryState::Active => "active",
            MemoryState::Degraded => "degraded",
            MemoryState::Forgotten => "forgotten",
            MemoryState::Processing => "processing",
            MemoryState::Invalid => "invalid",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_info_raw_content() {
        let layer = LayerInfo::raw_content();
        assert_eq!(layer.level, 0);
        assert_eq!(layer.name, Some("raw_content".to_string()));
        assert!(!layer.is_forgotten());
        assert!(layer.is_raw_content());
    }

    #[test]
    fn test_layer_info_forgotten() {
        let layer = LayerInfo::forgotten();
        assert_eq!(layer.level, -1);
        assert!(layer.is_forgotten());
        assert!(!layer.is_raw_content());
    }

    #[test]
    fn test_layer_info_custom() {
        let layer = LayerInfo::custom(5, "custom_layer");
        assert_eq!(layer.level, 5);
        assert_eq!(layer.name, Some("custom_layer".to_string()));
        assert_eq!(layer.name_or_default(), "custom_layer");
    }

    #[test]
    fn test_layer_info_name_or_default() {
        let named = LayerInfo::concept();
        assert_eq!(named.name_or_default(), "concept");

        let unnamed = LayerInfo {
            level: 99,
            name: None,
            schema_version: None,
        };
        assert_eq!(unnamed.name_or_default(), "layer_99");
    }

    #[test]
    fn test_memory_state_predicates() {
        assert!(MemoryState::Active.is_active());
        assert!(!MemoryState::Active.is_forgotten());
        assert!(!MemoryState::Active.is_processing());
        assert!(!MemoryState::Active.is_invalid());
        assert!(!MemoryState::Active.is_degraded());

        assert!(!MemoryState::Forgotten.is_active());
        assert!(MemoryState::Forgotten.is_forgotten());

        assert!(MemoryState::Processing.is_processing());
        assert!(MemoryState::Invalid.is_invalid());

        // Degraded is treated as active (still searchable)
        assert!(MemoryState::Degraded.is_active());
        assert!(MemoryState::Degraded.is_degraded());
        assert!(!MemoryState::Degraded.is_forgotten());
        assert!(!MemoryState::Degraded.is_processing());
        assert!(!MemoryState::Degraded.is_invalid());
    }

    #[test]
    fn test_memory_state_degraded_as_str() {
        assert_eq!(MemoryState::Degraded.as_str(), "degraded");
        assert_eq!(MemoryState::Active.as_str(), "active");
        assert_eq!(MemoryState::Forgotten.as_str(), "forgotten");
    }

    #[test]
    fn test_layer_info_default() {
        let default = LayerInfo::default();
        assert_eq!(default.level, 0);
        assert!(default.is_raw_content());
    }

    #[test]
    fn test_memory_state_default() {
        let default = MemoryState::default();
        assert!(default.is_active());
    }
}
