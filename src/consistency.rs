//! Database consistency checking and repair.
//!
//! Provides tools to verify structural integrity of a memory bank and
//! optionally repair detected issues. Designed for recovery scenarios
//! where a bank may have been partially written or merged from multiple
//! sources.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::{
    error::Result,
    types::{Filters, Memory, MemoryState},
    vector_store::VectorStore,
};

// ── Issue taxonomy ─────────────────────────────────────────────────

/// Severity of a detected issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Data loss or broken invariant.
    Error,
    /// Potential data quality issue.
    Warning,
    /// Informational observation.
    Info,
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IssueSeverity::Error => write!(f, "ERROR"),
            IssueSeverity::Warning => write!(f, "WARNING"),
            IssueSeverity::Info => write!(f, "INFO"),
        }
    }
}

/// Category of consistency issue (used for selective fixing).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IssueKind {
    /// L1+ memory whose abstraction_sources reference non-existent memories.
    OrphanedAbstraction,
    /// Memory state doesn't match reality (e.g. Active but sources deleted).
    StaleState,
    /// Memory has no embedding vector.
    MissingEmbedding,
    /// Content hash doesn't match stored hash.
    HashMismatch,
    /// Forgotten memory with no dependents (can be purged).
    UnreferencedForgotten,
    /// Duplicate memories (same content hash).
    DuplicateContent,
    /// L0 with abstraction_sources (invalid for raw content).
    InvalidLayerStructure,
}

impl IssueKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            IssueKind::OrphanedAbstraction => "orphaned-abstractions",
            IssueKind::StaleState => "stale-states",
            IssueKind::MissingEmbedding => "missing-embeddings",
            IssueKind::HashMismatch => "hash-mismatches",
            IssueKind::UnreferencedForgotten => "unreferenced-forgotten",
            IssueKind::DuplicateContent => "duplicate-content",
            IssueKind::InvalidLayerStructure => "invalid-layer-structure",
        }
    }

    /// Parse from CLI string.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "orphaned-abstractions" => Some(IssueKind::OrphanedAbstraction),
            "stale-states" => Some(IssueKind::StaleState),
            "missing-embeddings" => Some(IssueKind::MissingEmbedding),
            "hash-mismatches" => Some(IssueKind::HashMismatch),
            "unreferenced-forgotten" => Some(IssueKind::UnreferencedForgotten),
            "duplicate-content" => Some(IssueKind::DuplicateContent),
            "invalid-layer-structure" => Some(IssueKind::InvalidLayerStructure),
            _ => None,
        }
    }

    /// All possible issue kinds.
    pub fn all() -> Vec<Self> {
        vec![
            IssueKind::OrphanedAbstraction,
            IssueKind::StaleState,
            IssueKind::MissingEmbedding,
            IssueKind::HashMismatch,
            IssueKind::UnreferencedForgotten,
            IssueKind::DuplicateContent,
            IssueKind::InvalidLayerStructure,
        ]
    }
}

impl std::fmt::Display for IssueKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A single detected consistency issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyIssue {
    pub kind: IssueKind,
    pub severity: IssueSeverity,
    pub memory_id: String,
    pub message: String,
}

/// Result of a consistency check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyReport {
    pub total_memories: usize,
    pub issues: Vec<ConsistencyIssue>,
    pub errors: usize,
    pub warnings: usize,
    pub infos: usize,
}

impl ConsistencyReport {
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }
}

/// Result of a fix operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixReport {
    pub fixed: usize,
    pub skipped: usize,
    pub deleted: usize,
    pub details: Vec<String>,
}

// ── Checker ────────────────────────────────────────────────────────

/// Run all consistency checks on a vector store and return a report.
pub async fn check_consistency(store: &dyn VectorStore) -> Result<ConsistencyReport> {
    let all_memories = store.list(&Filters::default(), None).await?;
    let total = all_memories.len();

    // Build lookup maps
    let id_set: HashSet<&str> = all_memories.iter().map(|m| m.id.as_str()).collect();
    let id_to_memory: HashMap<&str, &Memory> =
        all_memories.iter().map(|m| (m.id.as_str(), m)).collect();

    // Build reverse dependency map: source_uuid → set of dependent memory IDs
    let mut dependents: HashMap<Uuid, HashSet<&str>> = HashMap::new();
    for m in &all_memories {
        for src in &m.metadata.abstraction_sources {
            dependents.entry(*src).or_default().insert(&m.id);
        }
    }

    let mut issues = Vec::new();

    for memory in &all_memories {
        let level = memory.metadata.layer.level;

        // 1. Orphaned abstractions — L1+ referencing missing sources
        if level > 0 {
            let mut missing_sources = Vec::new();
            for src in &memory.metadata.abstraction_sources {
                let src_str = src.to_string();
                if !id_set.contains(src_str.as_str()) {
                    missing_sources.push(src_str);
                }
            }
            if !missing_sources.is_empty() {
                let all_missing =
                    missing_sources.len() == memory.metadata.abstraction_sources.len();
                let severity = if all_missing {
                    IssueSeverity::Error
                } else {
                    IssueSeverity::Warning
                };
                issues.push(ConsistencyIssue {
                    kind: IssueKind::OrphanedAbstraction,
                    severity,
                    memory_id: memory.id.clone(),
                    message: format!(
                        "L{} memory references {} missing source(s): [{}]{}",
                        level,
                        missing_sources.len(),
                        missing_sources
                            .iter()
                            .take(3)
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", "),
                        if missing_sources.len() > 3 { "..." } else { "" }
                    ),
                });
            }
        }

        // 2. Invalid layer structure — L0 with abstraction_sources
        if level == 0 && !memory.metadata.abstraction_sources.is_empty() {
            issues.push(ConsistencyIssue {
                kind: IssueKind::InvalidLayerStructure,
                severity: IssueSeverity::Error,
                memory_id: memory.id.clone(),
                message: format!(
                    "L0 memory has {} abstraction_sources (should be empty)",
                    memory.metadata.abstraction_sources.len()
                ),
            });
        }

        // 3. Missing embedding
        if memory.embedding.is_empty() {
            issues.push(ConsistencyIssue {
                kind: IssueKind::MissingEmbedding,
                severity: IssueSeverity::Error,
                memory_id: memory.id.clone(),
                message: "Memory has no embedding vector".to_string(),
            });
        }

        // 4. Hash mismatch
        if let Some(content) = &memory.content {
            let expected = compute_sha256(content);
            if !memory.metadata.hash.is_empty() && memory.metadata.hash != expected {
                issues.push(ConsistencyIssue {
                    kind: IssueKind::HashMismatch,
                    severity: IssueSeverity::Warning,
                    memory_id: memory.id.clone(),
                    message: format!(
                        "Content hash mismatch: stored={} computed={}",
                        &memory.metadata.hash[..16.min(memory.metadata.hash.len())],
                        &expected[..16]
                    ),
                });
            }
        }

        // 5. Stale state — Active but has deleted sources
        if level > 0 && memory.metadata.state == MemoryState::Active {
            let missing_count = memory
                .metadata
                .abstraction_sources
                .iter()
                .filter(|src| !id_set.contains(src.to_string().as_str()))
                .count();
            if missing_count > 0 && !memory.metadata.abstraction_sources.is_empty() {
                let total_sources = memory.metadata.abstraction_sources.len();
                if missing_count == total_sources {
                    issues.push(ConsistencyIssue {
                        kind: IssueKind::StaleState,
                        severity: IssueSeverity::Error,
                        memory_id: memory.id.clone(),
                        message: format!(
                            "L{} memory is Active but all {} sources are missing (should be Forgotten)",
                            level, total_sources
                        ),
                    });
                } else {
                    issues.push(ConsistencyIssue {
                        kind: IssueKind::StaleState,
                        severity: IssueSeverity::Warning,
                        memory_id: memory.id.clone(),
                        message: format!(
                            "L{} memory is Active but {}/{} sources are missing (should be Degraded)",
                            level, missing_count, total_sources
                        ),
                    });
                }
            }
        }

        // 6. Unreferenced forgotten — Forgotten with no dependents
        if memory.metadata.state == MemoryState::Forgotten {
            let mem_uuid = Uuid::parse_str(&memory.id).ok();
            let has_dependents = mem_uuid
                .map(|u| {
                    dependents
                        .get(&u)
                        .map(|deps| {
                            deps.iter().any(|dep_id| {
                                id_to_memory
                                    .get(dep_id)
                                    .map(|d| d.metadata.state != MemoryState::Forgotten)
                                    .unwrap_or(false)
                            })
                        })
                        .unwrap_or(false)
                })
                .unwrap_or(false);

            if !has_dependents {
                issues.push(ConsistencyIssue {
                    kind: IssueKind::UnreferencedForgotten,
                    severity: IssueSeverity::Info,
                    memory_id: memory.id.clone(),
                    message: format!(
                        "Forgotten L{} memory has no active dependents (can be purged)",
                        level
                    ),
                });
            }
        }
    }

    // 7. Duplicate content hashes
    let mut hash_owners: HashMap<&str, Vec<&str>> = HashMap::new();
    for m in &all_memories {
        if !m.metadata.hash.is_empty() {
            hash_owners
                .entry(m.metadata.hash.as_str())
                .or_default()
                .push(&m.id);
        }
    }
    for (hash, ids) in &hash_owners {
        if ids.len() > 1 {
            for id in ids {
                issues.push(ConsistencyIssue {
                    kind: IssueKind::DuplicateContent,
                    severity: IssueSeverity::Warning,
                    memory_id: id.to_string(),
                    message: format!(
                        "Shares content hash {} with {} other memor{}",
                        &hash[..16.min(hash.len())],
                        ids.len() - 1,
                        if ids.len() - 1 == 1 { "y" } else { "ies" }
                    ),
                });
            }
        }
    }

    let errors = issues
        .iter()
        .filter(|i| i.severity == IssueSeverity::Error)
        .count();
    let warnings = issues
        .iter()
        .filter(|i| i.severity == IssueSeverity::Warning)
        .count();
    let infos = issues
        .iter()
        .filter(|i| i.severity == IssueSeverity::Info)
        .count();

    Ok(ConsistencyReport {
        total_memories: total,
        issues,
        errors,
        warnings,
        infos,
    })
}

// ── Fixer ──────────────────────────────────────────────────────────

/// Apply fixes for issues of the specified kinds.
///
/// The `purge` flag controls whether unreferenced Forgotten memories are
/// hard-deleted. When false, they are left as-is.
///
/// When `fix_kinds` is None, all fixable issue kinds are applied.
pub async fn fix_issues(
    store: &dyn VectorStore,
    fix_kinds: Option<&[IssueKind]>,
    purge: bool,
) -> Result<FixReport> {
    let report = check_consistency(store).await?;

    let all_kinds = IssueKind::all();
    let kinds: HashSet<&IssueKind> = fix_kinds
        .map(|k| k.iter().collect())
        .unwrap_or_else(|| all_kinds.iter().collect());

    let all_memories = store.list(&Filters::default(), None).await?;
    let id_set: HashSet<String> = all_memories.iter().map(|m| m.id.clone()).collect();

    let mut fixed = 0usize;
    let mut skipped = 0usize;
    let mut deleted = 0usize;
    let mut details = Vec::new();

    for issue in &report.issues {
        if !kinds.contains(&issue.kind) {
            skipped += 1;
            continue;
        }

        match &issue.kind {
            IssueKind::OrphanedAbstraction => {
                // Level-aware handling:
                // - L1 with ALL sources missing → delete (no grounded info)
                // - L2+ with some sources missing → remove dead refs, mark Degraded
                // - L2+ with ALL sources missing → delete
                if let Some(memory) = store.get(&issue.memory_id).await? {
                    let level = memory.metadata.layer.level;
                    let total_sources = memory.metadata.abstraction_sources.len();
                    let missing_count = memory
                        .metadata
                        .abstraction_sources
                        .iter()
                        .filter(|s| !id_set.contains(&s.to_string()))
                        .count();
                    let all_missing = missing_count == total_sources && total_sources > 0;

                    if level == 1 && all_missing {
                        // L1 without any L0 → no grounded info, delete
                        store.delete(&memory.id).await?;
                        deleted += 1;
                        details.push(format!(
                            "Deleted L1 memory {} (all {} L0 sources missing)",
                            &memory.id[..8],
                            total_sources
                        ));
                    } else if all_missing {
                        // L2+ with all sources missing → delete
                        store.delete(&memory.id).await?;
                        deleted += 1;
                        details.push(format!(
                            "Deleted L{} memory {} (all {} sources missing)",
                            level,
                            &memory.id[..8],
                            total_sources
                        ));
                    } else {
                        // Partial loss → remove dead refs, mark Degraded
                        let mut updated = memory.clone();
                        let live_sources: Vec<Uuid> = updated
                            .metadata
                            .abstraction_sources
                            .iter()
                            .filter(|s| id_set.contains(&s.to_string()))
                            .copied()
                            .collect();
                        let removed = total_sources - live_sources.len();
                        updated.metadata.abstraction_sources = live_sources;
                        updated.metadata.state = MemoryState::Degraded;
                        updated.metadata.forgotten_sources = memory
                            .metadata
                            .abstraction_sources
                            .iter()
                            .filter(|s| !id_set.contains(&s.to_string()))
                            .copied()
                            .collect();
                        store.update(&updated).await?;
                        fixed += 1;
                        details.push(format!(
                            "Degraded L{} memory {} (removed {} dead refs, kept {})",
                            level,
                            &memory.id[..8],
                            removed,
                            updated.metadata.abstraction_sources.len()
                        ));
                    }
                }
            }
            IssueKind::StaleState => {
                if let Some(memory) = store.get(&issue.memory_id).await? {
                    let total_sources = memory.metadata.abstraction_sources.len();
                    let missing_count = memory
                        .metadata
                        .abstraction_sources
                        .iter()
                        .filter(|s| !id_set.contains(&s.to_string()))
                        .count();
                    let mut updated = memory.clone();

                    if missing_count == total_sources && total_sources > 0 {
                        updated.metadata.state = MemoryState::Forgotten;
                        updated.metadata.forgotten_at = Some(chrono::Utc::now());
                    } else if missing_count > 0 {
                        updated.metadata.state = MemoryState::Degraded;
                        updated.metadata.forgotten_sources = memory
                            .metadata
                            .abstraction_sources
                            .iter()
                            .filter(|s| !id_set.contains(&s.to_string()))
                            .copied()
                            .collect();
                    }

                    store.update(&updated).await?;
                    fixed += 1;
                    details.push(format!(
                        "Updated state of L{} memory {} → {:?}",
                        memory.metadata.layer.level,
                        &memory.id[..8],
                        updated.metadata.state
                    ));
                }
            }
            IssueKind::HashMismatch => {
                if let Some(memory) = store.get(&issue.memory_id).await? {
                    if let Some(content) = &memory.content {
                        let mut updated = memory.clone();
                        updated.metadata.hash = compute_sha256(content);
                        store.update(&updated).await?;
                        fixed += 1;
                        details.push(format!(
                            "Recomputed hash for memory {}",
                            &memory.id[..8]
                        ));
                    }
                }
            }
            IssueKind::MissingEmbedding => {
                // Can't fix without LLM; mark as Invalid
                if let Some(memory) = store.get(&issue.memory_id).await? {
                    let mut updated = memory.clone();
                    updated.metadata.state = MemoryState::Invalid;
                    store.update(&updated).await?;
                    fixed += 1;
                    details.push(format!(
                        "Marked memory {} as Invalid (missing embedding)",
                        &memory.id[..8]
                    ));
                }
            }
            IssueKind::UnreferencedForgotten => {
                if purge {
                    store.delete(&issue.memory_id).await?;
                    deleted += 1;
                    details.push(format!(
                        "Purged unreferenced forgotten memory {}",
                        &issue.memory_id[..8.min(issue.memory_id.len())]
                    ));
                } else {
                    skipped += 1;
                }
            }
            IssueKind::DuplicateContent => {
                // Handled in batch below to pick the "keeper" correctly
                skipped += 1;
            }
            IssueKind::InvalidLayerStructure => {
                if let Some(memory) = store.get(&issue.memory_id).await? {
                    let mut updated = memory.clone();
                    updated.metadata.abstraction_sources.clear();
                    store.update(&updated).await?;
                    fixed += 1;
                    details.push(format!(
                        "Cleared abstraction_sources from L0 memory {}",
                        &memory.id[..8]
                    ));
                }
            }
        }
    }

    // Batch-handle duplicate content: group by hash, keep newest, delete rest
    if kinds.contains(&IssueKind::DuplicateContent) {
        let mut hash_groups: HashMap<String, Vec<&Memory>> = HashMap::new();
        for m in &all_memories {
            if !m.metadata.hash.is_empty() {
                hash_groups
                    .entry(m.metadata.hash.clone())
                    .or_default()
                    .push(m);
            }
        }
        for (hash, mut group) in hash_groups {
            if group.len() <= 1 {
                continue;
            }
            // Keep the newest (by updated_at), delete the rest
            group.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
            let keeper = group[0];
            for dup in &group[1..] {
                store.delete(&dup.id).await?;
                deleted += 1;
                details.push(format!(
                    "Deleted duplicate memory {} (kept {} for hash {})",
                    &dup.id[..8],
                    &keeper.id[..8],
                    &hash[..16.min(hash.len())]
                ));
            }
        }
    }

    Ok(FixReport {
        fixed,
        skipped,
        deleted,
        details,
    })
}

// ── Helpers ────────────────────────────────────────────────────────

fn compute_sha256(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}
