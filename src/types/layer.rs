//! Layered memory architecture types
//!
//! This module defines the types for scalable layered memory organization:
//! - `LayerInfo`: Identifies a memory's position in the abstraction hierarchy
//! - `MemoryState`: Lifecycle state of a memory (Active, Forgotten, etc.)
//! - `LayerRelationType`: Typed relations between memories at different layers

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
        self.name.clone().unwrap_or_else(|| format!("layer_{}", self.level))
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MemoryState {
    /// Memory is active and searchable
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

    /// Memory failed validation or abstraction (requires review)
    ///
    /// Invalid memories are not searchable and should be either:
    /// - Corrected and reprocessed
    /// - Deleted (if no higher layers depend on them)
    /// - Marked as Forgotten (if higher layers depend on them)
    Invalid,
}

impl MemoryState {
    /// Check if this memory is active
    pub fn is_active(&self) -> bool {
        matches!(self, MemoryState::Active)
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
}

impl Default for MemoryState {
    fn default() -> Self {
        MemoryState::Active
    }
}

/// Type of relation between memories, with layer semantics
///
/// Different relation types are typical at different abstraction layers:
/// - Structural relations (L0↔L1): chunk_of, summary_of, next_section
/// - Semantic relations (L1↔L2): related_to, extends, prerequisite
/// - Conceptual relations (L2↔L3+): emerges_from, instance_of, broader_than
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum LayerRelationType {
    // === Structural Relations (L0 ↔ L1) ===
    /// This memory is a chunk of a larger document
    /// Source: L0 (chunk), Target: L1 (document/section)
    ChunkOf,

    /// This memory summarizes source memories
    /// Source: L1 (summary), Target: L0 (content)
    SummaryOf,

    /// This memory is the next section/chapter
    /// Source: L0/L1, Target: L0/L1 (sequential)
    NextSection,

    /// This memory is the parent section
    /// Source: L0/L1 (child), Target: L1 (parent)
    ParentSection,

    // === Semantic Relations (L1 ↔ L2) ===
    /// This memory semantically relates to target (cross-reference)
    /// Source: L1/L2, Target: L1/L2
    RelatedTo,

    /// This memory extends or builds upon target
    /// Source: L1/L2, Target: L1/L2
    Extends,

    /// This memory contradicts or challenges target
    /// Source: L1/L2, Target: L1/L2
    Contradicts,

    /// This memory is a prerequisite for understanding target
    /// Source: L1/L2 (prerequisite), Target: L1/L2 (dependent)
    Prerequisite,

    /// This memory applies the concept in target
    /// Source: L1 (application), Target: L2/L3 (concept)
    AppliesConcept,

    // === Conceptual Relations (L2 ↔ L3+) ===
    /// This memory emerges from multiple lower-layer memories
    /// Source: L3 (concept), Target: L2 (semantic links)
    EmergesFrom,

    /// This memory is an instance/example of a concept
    /// Source: L1/L2 (instance), Target: L3 (concept)
    InstanceOf,

    /// This memory is a broader category containing target
    /// Source: L3 (broader), Target: L3 (narrower)
    BroaderThan,

    /// This memory is a narrower specialization of target
    /// Source: L3 (narrower), Target: L3 (broader)
    NarrowerThan,

    // === Custom/Extensible ===
    /// User-defined relation type
    Custom(String),
}

impl LayerRelationType {
    /// Get the typical source layer for this relation type
    ///
    /// This is a heuristic - actual layers may vary in practice.
    pub fn source_layer(&self) -> i32 {
        match self {
            LayerRelationType::ChunkOf => 0,
            LayerRelationType::SummaryOf => 1,
            LayerRelationType::NextSection => 0,
            LayerRelationType::ParentSection => 0,

            LayerRelationType::RelatedTo => 1,
            LayerRelationType::Extends => 1,
            LayerRelationType::Contradicts => 1,
            LayerRelationType::Prerequisite => 1,
            LayerRelationType::AppliesConcept => 1,

            LayerRelationType::EmergesFrom => 3,
            LayerRelationType::InstanceOf => 1,
            LayerRelationType::BroaderThan => 3,
            LayerRelationType::NarrowerThan => 3,

            LayerRelationType::Custom(_) => 0,
        }
    }

    /// Get the typical target layer for this relation type
    ///
    /// This is a heuristic - actual layers may vary in practice.
    pub fn target_layer(&self) -> i32 {
        match self {
            LayerRelationType::ChunkOf => 1,
            LayerRelationType::SummaryOf => 0,
            LayerRelationType::NextSection => 0,
            LayerRelationType::ParentSection => 1,

            LayerRelationType::RelatedTo => 1,
            LayerRelationType::Extends => 1,
            LayerRelationType::Contradicts => 1,
            LayerRelationType::Prerequisite => 1,
            LayerRelationType::AppliesConcept => 2,

            LayerRelationType::EmergesFrom => 2,
            LayerRelationType::InstanceOf => 3,
            LayerRelationType::BroaderThan => 3,
            LayerRelationType::NarrowerThan => 3,

            LayerRelationType::Custom(_) => 0,
        }
    }

    /// Check if this relation type is structural (L0↔L1)
    pub fn is_structural(&self) -> bool {
        matches!(
            self,
            LayerRelationType::ChunkOf
                | LayerRelationType::SummaryOf
                | LayerRelationType::NextSection
                | LayerRelationType::ParentSection
        )
    }

    /// Check if this relation type is semantic (L1↔L2)
    pub fn is_semantic(&self) -> bool {
        matches!(
            self,
            LayerRelationType::RelatedTo
                | LayerRelationType::Extends
                | LayerRelationType::Contradicts
                | LayerRelationType::Prerequisite
                | LayerRelationType::AppliesConcept
        )
    }

    /// Check if this relation type is conceptual (L2↔L3+)
    pub fn is_conceptual(&self) -> bool {
        matches!(
            self,
            LayerRelationType::EmergesFrom
                | LayerRelationType::InstanceOf
                | LayerRelationType::BroaderThan
                | LayerRelationType::NarrowerThan
        )
    }
}

impl std::fmt::Display for LayerRelationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayerRelationType::ChunkOf => write!(f, "chunk_of"),
            LayerRelationType::SummaryOf => write!(f, "summary_of"),
            LayerRelationType::NextSection => write!(f, "next_section"),
            LayerRelationType::ParentSection => write!(f, "parent_section"),
            LayerRelationType::RelatedTo => write!(f, "related_to"),
            LayerRelationType::Extends => write!(f, "extends"),
            LayerRelationType::Contradicts => write!(f, "contradicts"),
            LayerRelationType::Prerequisite => write!(f, "prerequisite"),
            LayerRelationType::AppliesConcept => write!(f, "applies_concept"),
            LayerRelationType::EmergesFrom => write!(f, "emerges_from"),
            LayerRelationType::InstanceOf => write!(f, "instance_of"),
            LayerRelationType::BroaderThan => write!(f, "broader_than"),
            LayerRelationType::NarrowerThan => write!(f, "narrower_than"),
            LayerRelationType::Custom(s) => write!(f, "{}", s),
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

        assert!(!MemoryState::Forgotten.is_active());
        assert!(MemoryState::Forgotten.is_forgotten());

        assert!(MemoryState::Processing.is_processing());
        assert!(MemoryState::Invalid.is_invalid());
    }

    #[test]
    fn test_layer_relation_type_source_target_layers() {
        let chunk = LayerRelationType::ChunkOf;
        assert_eq!(chunk.source_layer(), 0);
        assert_eq!(chunk.target_layer(), 1);

        let summary = LayerRelationType::SummaryOf;
        assert_eq!(summary.source_layer(), 1);
        assert_eq!(summary.target_layer(), 0);

        let emerges = LayerRelationType::EmergesFrom;
        assert_eq!(emerges.source_layer(), 3);
        assert_eq!(emerges.target_layer(), 2);
    }

    #[test]
    fn test_layer_relation_type_classification() {
        assert!(LayerRelationType::ChunkOf.is_structural());
        assert!(!LayerRelationType::ChunkOf.is_semantic());
        assert!(!LayerRelationType::ChunkOf.is_conceptual());

        assert!(!LayerRelationType::RelatedTo.is_structural());
        assert!(LayerRelationType::RelatedTo.is_semantic());
        assert!(!LayerRelationType::RelatedTo.is_conceptual());

        assert!(!LayerRelationType::EmergesFrom.is_structural());
        assert!(!LayerRelationType::EmergesFrom.is_semantic());
        assert!(LayerRelationType::EmergesFrom.is_conceptual());
    }

    #[test]
    fn test_layer_relation_type_display() {
        assert_eq!(format!("{}", LayerRelationType::ChunkOf), "chunk_of");
        assert_eq!(format!("{}", LayerRelationType::RelatedTo), "related_to");
        assert_eq!(format!("{}", LayerRelationType::Custom("my_rel".into())), "my_rel");
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
