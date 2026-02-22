pub mod config;
pub mod error;
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
pub use error::MemoryError;
pub use mcp::MemoryMcpService;
pub use memory::MemoryManager;
pub use memory_bank::{BackupManifest, MemoryBankInfo, MemoryBankManager, MergeResult};
pub use operations::{MemoryOperationPayload, MemoryOperationResponse, MemoryOperations};
pub use search::{
    GraphRankScore, GraphSearchEngine, GraphSearchResult, TraversalConfig, TraversalDirection,
    TraversalStrategy,
};
pub use types::{Memory, MemoryMetadata, MemoryType, ScoredMemory};
pub use vector_store::VectorStore;
