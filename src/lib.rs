//! # llm-mem: Scalable Layered Memory for AI Agents
//!
//! A standalone MCP memory server with an embedded vector store and **scalable layered memory architecture**.
//! Built in Rust as a single self-contained crate — no external databases required.
//!
//! ## Layered Memory Architecture
//!
//! llm-mem implements a cognitive-inspired memory system where memories exist at different levels of abstraction:
//!
//! | Layer | Name | Description | Example |
//! |-------|------|-------------|---------|
//! | L4 | Wisdom | Mental models, paradigms | "Linear Algebra is about transformations" |
//! | L3 | Concept | Domain theories, principles | "Eigenvalues and Eigenvectors" |
//! | L2 | Semantic | Cross-document links | "Relates ODEs to Control Theory" |
//! | L1 | Structural | Summaries, document structure | "Chapter 3: Laplace Transforms Summary" |
//! | L0 | Raw Content | User-provided, immutable content | "The Laplace transform is defined as..." |
//! | L-1 | Forgotten | Soft-deleted (preserved for referential integrity) | [Referenced by higher layers] |
//!
//! ### Key Features
//!
//! - **Progressive Abstraction**: Background workers create higher-layer abstractions (L0→L1→L2→L3→L4)
//! - **Bidirectional Navigation**: `zoom_in()` to see sources, `zoom_out()` to see abstractions
//! - **Layer Filtering**: Search at specific abstraction levels
//! - **Graceful Degradation**: Deletion marks higher layers as "forgotten" instead of deleting
//! - **Provenance Tracking**: Each abstraction tracks source memories
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use llm_mem::MemoryMcpService;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Start MCP server with default config (local inference)
//!     let service = MemoryMcpService::new().await?;
//!     // Service runs as MCP stdio server
//!     Ok(())
//! }
//! ```
//!
//! ## Components
//!
//! - **MCP Server** (`MemoryMcpService`): Exposes memory tools via Model Context Protocol
//! - **Memory Manager** (`MemoryManager`): Orchestrates AI pipeline (fact extraction, importance scoring, deduplication)
//! - **Vector Store** (`VectorLiteStore`): Embedded vector database with HNSW/Flat indexes
//! - **Layer Navigator** (`layer::navigation::LayerNavigator`): Navigate across abstraction layers
//! - **Abstraction Pipeline** (`layer::abstraction_pipeline::AbstractionPipeline`): Background workers for progressive abstraction
//!
//! ## CLI Tools
//!
//! - `llm-mem-mcp`: MCP server binary
//! - `llm-mem-inspect`: CLI inspection tool with layer visualization
//!   - `layer-stats`: Show layer statistics
//!   - `layer-tree`: Show layer hierarchy as ASCII tree
//!
//! ## Documentation
//!
//! - [README](../README.md) - User guide and quick start
//! - [MIGRATION_V4.md](../MIGRATION_V4.md) - Migration guide from v3 to v4

pub mod config;
pub mod consistency;
pub mod document_session;
pub mod error;
pub mod layer;
pub mod llm;
pub mod mcp;
pub mod memory;
pub mod memory_bank;
pub mod operations;
pub mod search;
pub mod types;
pub mod vector_store;

// Re-export key types for convenient access
pub use config::Config;
pub use document_session::{
    BeginStoreDocumentResponse, DocumentMetadata, DocumentSession, DocumentSessionManager,
    ProcessingResult, SessionStatus, StatusProcessDocumentResponse,
};
pub use error::MemoryError;
pub use layer::navigation::LayerNavigator;
pub use mcp::MemoryMcpService;
pub use memory::MemoryManager;
pub use memory_bank::{BackupManifest, DuplicateStrategy, MemoryBankInfo, MemoryBankManager, MergeResult, MultiMergeResult};
pub use operations::{MemoryOperationPayload, MemoryOperationResponse, MemoryOperations};
pub use search::{
    GraphSearchEngine, GraphSearchResult, TraversalConfig, TraversalDirection, TraversalStrategy,
};
pub use types::{
    ContentMeta, DerivedEntry, DerivedMeta, LayerInfo, Memory, MemoryMetadata, MemoryState,
    MemoryType, RelationEntry, RelationMeta, ScoredMemory,
};
pub use vector_store::VectorStore;

// CLI System struct for shared use
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

/// CLI system context shared across commands
#[allow(dead_code)]
pub struct System {
    pub bank_manager: Arc<crate::memory_bank::MemoryBankManager>,
    pub memory_manager: Arc<crate::memory::MemoryManager>,
    pub session_manager: Arc<crate::document_session::DocumentSessionManager>,
    pub operations: Arc<TokioMutex<crate::operations::MemoryOperations>>,
    pub models_dir: std::path::PathBuf,
}

