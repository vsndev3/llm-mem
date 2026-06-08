use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{info, warn};

/// Vector store backend type
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum VectorStoreType {
    Vectorlite,
    #[default]
    LanceDB,
}

/// Provider type for LLM and embedding services
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// Local inference (llama.cpp for LLM, fastembed for embedding)
    #[default]
    Local,
    /// OpenAI-compatible API
    #[serde(alias = "openai")]
    Api,
}

/// LLM backend type (derived from llm.provider + embedding.provider)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LLMBackend {
    /// Both LLM and embeddings use local inference
    Local,
    /// Both LLM and embeddings use API
    #[serde(alias = "openai")]
    API,
    /// LLM via API, embeddings via local fastembed
    APILLMLocalEmbed,
    /// LLM via local llama.cpp, embeddings via API
    LocalLLMAPIEmbed,
}

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// LLM (language model) configuration for queries, extraction, summarization
    #[serde(default)]
    pub llm: LlmConfig,
    /// Embedding model configuration for vector similarity search
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub vector_store: VectorStoreConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
}

/// LLM configuration — all settings for the language model (local + API)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// Provider: "local" (embedded llama.cpp) or "api" (OpenAI-compatible)
    pub provider: ProviderType,

    // --- Local provider settings (embedded llama.cpp) ---
    /// GGUF model filename (auto-downloaded if auto_download = true)
    pub model_file: String,
    /// Directory for model files
    pub models_dir: String,
    /// Number of GPU layers to offload (0 = CPU only)
    pub gpu_layers: u32,
    /// Context window size in tokens
    pub context_size: u32,
    /// CPU threads for inference (0 = auto-detect)
    pub cpu_threads: i32,
    /// Auto-download known models from HuggingFace
    pub auto_download: bool,
    /// Cache downloaded models
    pub cache_model: bool,
    /// Custom directory for model caching (default: ~/.cache/llm-mem/models)
    pub cache_dir: Option<String>,
    /// Timeout in seconds for LLM completion calls (applies to both local and API)
    pub llm_timeout_secs: u64,
    /// Use grammar-constrained sampling for structured output (local only)
    pub use_grammar: bool,
    /// Proxy URL for model downloads (overrides HTTPS_PROXY env var)
    pub proxy_url: Option<String>,

    /// Path to mmproj (multimodal projector) GGUF file for vision-enabled models
    /// Required for local vision/image description via llama.cpp.
    /// Set to None to disable local vision (API vision may still be used if configured).
    pub mmproj_file: Option<String>,
    /// Enable vision/image description features
    pub vision_enabled: bool,
    /// Prompt template for image description. The image is attached alongside
    /// this text. Default: ask for a detailed, searchable description optimized
    /// for memory retrieval.
    pub vision_prompt_template: String,

    // --- API provider settings ---
    /// API endpoint URL (e.g. "https://api.openai.com/v1")
    pub api_url: String,
    /// API key (or set LLM_MEM_LLM_API_KEY / OPENAI_API_KEY env var)
    pub api_key: String,
    /// Model identifier (e.g. "gpt-4o-mini", "meta-llama/llama-3.1-8b-instruct")
    pub model: String,

    // --- Generation settings (both providers) ---
    /// Sampling temperature (0.0 to 2.0)
    pub temperature: f32,
    /// Maximum tokens to generate per completion
    pub max_tokens: u32,

    // --- Advanced settings ---
    /// Request format: "auto" (default), "rig", or "raw" (API only)
    pub request_format: RequestFormat,
    /// API dialect for raw HTTP requests
    pub api_dialect: ApiDialect,
    /// Custom dialect configuration (only used if api_dialect = "custom")
    pub custom_dialect: Option<CustomDialectConfig>,
    /// Use structured output mode (JSON schema validation) for API
    pub use_structured_output: bool,
    /// Max retry attempts for structured output validation
    pub structured_output_retries: u32,
    /// Max concurrent requests (0 = unlimited, 1 = sequential)
    pub max_concurrent_requests: usize,
    /// XML tags to strip from LLM output (e.g. ["think", "reason"])
    pub strip_tags: Vec<String>,
    /// Max items per batch request
    pub batch_size: usize,
    /// Max tokens per batch request
    pub batch_max_tokens: u32,
    /// Base timeout in seconds for batch calls
    pub batch_timeout_secs: u64,
    /// Timeout multiplier for batch requests
    pub batch_timeout_multiplier: f64,
}

/// Request format mode for API-based LLM clients
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum RequestFormat {
    /// Automatically detect and use the appropriate format (default)
    /// Tries rig-core first, falls back to raw HTTP on 422 errors
    #[default]
    Auto,
    /// Use rig-core's completion API (may format messages as complex arrays)
    Rig,
    /// Use raw HTTP requests with plain string content (bypasses rig-core)
    Raw,
}

/// API dialect for raw HTTP requests
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ApiDialect {
    /// OpenAI-compatible chat completions (default)
    #[default]
    #[serde(alias = "openai-chat")]
    OpenAIChat,
    /// OpenAI-compatible standard completions
    #[serde(alias = "openai-completion")]
    OpenAICompletion,
    /// Anthropic-style completions
    Anthropic,
    /// Ollama native chat API (/api/chat)
    #[serde(alias = "ollama-chat")]
    OllamaChat,
    /// Ollama native completion API (/api/generate)
    #[serde(alias = "ollama-completion")]
    OllamaCompletion,
    /// Fully custom dialect defined in configuration
    Custom,
}

/// Configuration for a fully custom API dialect
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CustomDialectConfig {
    /// Path to append to base_url (e.g., "/v1/generate")
    pub endpoint_path: String,
    /// JSON template for the request body.
    /// Placeholders: {{prompt}}, {{model}}, {{temperature}}, {{max_tokens}}
    pub request_body_template: String,
    /// JSON pointer to the extracted text in the response (e.g., "/results/0/text")
    pub response_content_pointer: String,
}

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// Provider: "local" (fastembed) or "api" (OpenAI-compatible)
    pub provider: ProviderType,
    /// Model name (local: fastembed name like "all-MiniLM-L6-v2"; api: e.g. "text-embedding-3-small")
    pub model: String,
    /// API endpoint URL (for api provider)
    pub api_url: String,
    /// API key (for api provider, or set LLM_MEM_EMBEDDING_API_KEY env var)
    pub api_key: String,
    /// Request format: "auto" (default), "rig", or "raw" (API only)
    pub request_format: RequestFormat,
    /// API dialect for raw HTTP requests
    pub api_dialect: ApiDialect,
    /// Custom dialect configuration (only used if api_dialect = "custom")
    pub custom_dialect: Option<CustomDialectConfig>,
    /// Batch size for embedding requests
    pub batch_size: usize,
    /// Timeout in seconds for embedding requests
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    #[serde(default)]
    pub store_type: VectorStoreType,
    #[serde(default = "default_collection_name")]
    pub collection_name: String,
    #[serde(default)]
    pub vectorlite: VectorLiteSettings,
    #[serde(default)]
    pub lancedb: LanceDBSettings,
    /// Directory for memory bank database files (default: ./llm-mem-data/banks)
    #[serde(default = "default_banks_dir")]
    pub banks_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorLiteSettings {
    #[serde(default = "default_index_type")]
    pub index_type: String,
    #[serde(default = "default_metric")]
    pub metric: String,
    pub persistence_path: Option<String>,
    #[serde(default)]
    pub embedding_dimension: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanceDBSettings {
    #[serde(default = "default_lancedb_table_name")]
    pub table_name: String,
    #[serde(default = "default_lancedb_path")]
    pub database_path: String,
    #[serde(default = "default_embedding_dimension")]
    pub embedding_dimension: usize,
}

/// HTTP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

/// Memory manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    pub max_memories: usize,
    pub similarity_threshold: f32,
    pub max_search_results: usize,
    pub memory_ttl_hours: Option<u64>,
    pub auto_summary_threshold: usize,
    pub auto_enhance: bool,
    pub deduplicate: bool,
    pub merge_threshold: f32,
    pub search_similarity_threshold: Option<f32>,
    /// Maximum content length in bytes before rejection (default: 32768)
    pub max_content_length: usize,
    /// Chunk size (in characters) used for document ingestion (default: 2000).
    /// Use 0 to disable document chunking (stored as single memory).
    pub document_chunk_size: usize,
    /// Use LLM-based query intent classification instead of keyword heuristic
    pub use_llm_query_classification: bool,
    /// Use LLM to detect format when content-based detection returns Unknown.
    /// The first 4KB of content is sent to the LLM for format identification.
    pub llm_format_detection: bool,
    /// Use LLM as fallback parser when structured parsers fail.
    /// LLM extracts summary, entities, and content type from raw text.
    pub llm_fallback_parsing: bool,
    /// When storing L0 memories, chunk long content into overlapping segments for
    /// embedding. Triggers when content exceeds this many characters. Set to 0 to disable.
    /// Default matches chunk_size_chars (1000) so any content exceeding the safe
    /// embedding window is automatically split.
    pub chunk_threshold_chars: usize,
    /// Target size of each sub-chunk in characters (default: 1000, ~250 tokens at
    /// ~4 chars/token for English prose). Must fit within the embedding model's token limit.
    pub chunk_size_chars: usize,
    /// Overlap between consecutive chunks in characters (default: 100)
    pub chunk_overlap_chars: usize,
    /// Maximum number of abstraction dependents to load at once during cascade
    /// deletion. Prevents OOM when a single memory is the source for thousands
    /// of abstractions (default: 5000).
    pub max_cascade_fanout: usize,
    /// Maximum number of raw content candidates to scan during keyword/text
    /// search. When a vector pre-filter returns no candidates, this caps the
    /// fallback list() scan (default: 5000).
    pub raw_content_scan_limit: usize,
    /// Hard ceiling for list() queries with no limit specified. Prevents
    /// accidental full-table scans (default: 10000).
    pub max_list_limit: usize,
    /// Maximum total candidates held in memory during pyramid assembly.
    /// When pyramid search collects results across multiple layers, this caps
    /// the total in-memory candidates before final ranking and truncation.
    /// Prevents OOM on broad queries against very large knowledge bases.
    /// Set to 0 for unlimited (default: 10000).
    pub max_total_candidates: usize,
    /// Similarity threshold for automatic cross-linking of new memories to
    /// semantically similar existing memories. 0.0 disables auto-linking.
    /// (default: 0.75)
    pub auto_link_threshold: f32,
    /// Maximum number of auto-link relations to create per stored memory
    /// (default: 10)
    pub auto_link_max_relations: usize,
    /// Per-session LLM token budget. When exceeded, expensive operations
    /// (enhance, classify, abstract) are skipped. 0 = unlimited.
    /// (default: 0)
    pub session_token_budget: u64,
    /// Dry-run mode: log what would be done without executing LLM calls.
    /// Useful for cost estimation before enabling auto-enhance.
    /// (default: false)
    pub dry_run: bool,
    /// Cosine similarity threshold for near-duplicate detection.
    /// When a new memory's embedding similarity to an existing memory exceeds
    /// this, a warning is logged. 0.0 disables. (default: 0.92)
    pub near_duplicate_threshold: f32,
    /// Whether to check for factual contradictions when storing new memories.
    /// The top-3 semantically similar memories are given to the LLM to check
    /// for contradictions. (default: false)
    pub contradiction_detection: bool,
    /// Time decay half-life in hours for access-frequency boosting.
    /// Recent accesses boost score; older accesses decay. 0 disables.
    /// (default: 168 = 1 week)
    pub access_decay_hours: u32,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    pub enabled: bool,
    pub log_directory: String,
    pub level: String,
    pub max_size_mb: u64,
    pub max_files: usize,
}

// --- Default impls ---

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: ProviderType::Local,
            // Local provider
            model_file: "gemma-4-E2B-it-Q8_0.gguf".to_string(),
            models_dir: "llm-mem-data/models".to_string(),
            gpu_layers: 0,
            context_size: 16644,
            cpu_threads: 0,
            auto_download: true,
            cache_model: true,
            cache_dir: None,
            llm_timeout_secs: 120,
            use_grammar: false,
            proxy_url: None,
            // Vision
            mmproj_file: None,
            vision_enabled: false,
            vision_prompt_template: "Describe this image in detail, focusing on what would make it searchable in a personal memory system. What objects, people, text, colors, scene elements, actions, and contextual cues are visible? What domain or topic does this image relate to? Be specific and concrete.".to_string(),
            // API provider
            api_url: "https://api.openai.com/v1".to_string(),
            api_key: String::new(),
            model: "gpt-4o-mini".to_string(),
            // Generation
            temperature: 0.7,
            max_tokens: 4096,
            // Advanced
            request_format: RequestFormat::Auto,
            api_dialect: ApiDialect::OpenAIChat,
            custom_dialect: None,
            use_structured_output: true,
            structured_output_retries: 2,
            max_concurrent_requests: 1,
            strip_tags: vec!["think".to_string()],
            batch_size: 10,
            batch_max_tokens: 3000,
            batch_timeout_secs: 120,
            batch_timeout_multiplier: 1.0,
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: ProviderType::Local,
            model: "all-MiniLM-L6-v2".to_string(),
            api_url: "https://api.openai.com/v1".to_string(),
            api_key: String::new(),
            request_format: RequestFormat::default(),
            api_dialect: ApiDialect::default(),
            custom_dialect: None,
            batch_size: 64,
            timeout_secs: 30,
        }
    }
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            store_type: VectorStoreType::default(),
            collection_name: default_collection_name(),
            vectorlite: VectorLiteSettings::default(),
            lancedb: LanceDBSettings::default(),
            banks_dir: default_banks_dir(),
        }
    }
}

impl VectorStoreConfig {
    pub fn embedding_dimension(&self) -> usize {
        match self.store_type {
            VectorStoreType::LanceDB => self.lancedb.embedding_dimension,
            #[cfg(feature = "vector-lite")]
            VectorStoreType::Vectorlite => self.vectorlite.embedding_dimension.unwrap_or(384),
            #[cfg(not(feature = "vector-lite"))]
            VectorStoreType::Vectorlite => self.lancedb.embedding_dimension,
        }
    }
}

impl Default for VectorLiteSettings {
    fn default() -> Self {
        Self {
            index_type: default_index_type(),
            metric: default_metric(),
            persistence_path: None,
            embedding_dimension: None,
        }
    }
}

impl Default for LanceDBSettings {
    fn default() -> Self {
        Self {
            table_name: default_lancedb_table_name(),
            database_path: default_lancedb_path(),
            embedding_dimension: default_embedding_dimension(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memories: 10000,
            similarity_threshold: 0.65,
            max_search_results: 50,
            memory_ttl_hours: None,
            auto_summary_threshold: 32768,
            auto_enhance: true,
            deduplicate: true,
            merge_threshold: 0.75,
            search_similarity_threshold: Some(0.2),
            max_content_length: 32768,
            document_chunk_size: 2000,
            use_llm_query_classification: false,
            llm_format_detection: false,
            llm_fallback_parsing: false,
            chunk_threshold_chars: 1000,
            chunk_size_chars: 1000,
            chunk_overlap_chars: 100,
            max_cascade_fanout: 5000,
            raw_content_scan_limit: 5000,
            max_list_limit: 10000,
            max_total_candidates: 10000,
            auto_link_threshold: 0.75,
            auto_link_max_relations: 10,
            session_token_budget: 0,
            dry_run: false,
            near_duplicate_threshold: 0.92,
            contradiction_detection: false,
            access_decay_hours: 168,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            log_directory: "llm-mem-data/logs".to_string(),
            level: "info".to_string(),
            max_size_mb: 1,
            max_files: 5,
        }
    }
}

fn default_collection_name() -> String {
    "llm-memories".to_string()
}

fn default_banks_dir() -> String {
    "llm-mem-data/banks".to_string()
}

fn default_index_type() -> String {
    "hnsw".to_string()
}

fn default_metric() -> String {
    "cosine".to_string()
}

fn default_lancedb_table_name() -> String {
    "memories".to_string()
}

fn default_lancedb_path() -> String {
    "./lancedb".to_string()
}

fn default_embedding_dimension() -> usize {
    384
}

impl Config {
    /// Generate a comprehensive, commented-out TOML configuration template
    pub fn template() -> String {
        r#"# LLM Memory Manager Configuration
#
# Two things to configure:
#   [llm]       - Language model for queries, extraction, summarization
#   [embedding] - Embedding model for vector similarity search
#
# Both default to local inference (no API keys needed).

# ── LLM ──────────────────────────────────────────────────────────────────────
# The language model used for memory extraction, summarization, and queries.
#
# provider = "local"   → embedded llama.cpp (default, works offline)
# provider = "api"     → OpenAI-compatible API (OpenAI, OpenRouter, llama-server, etc.)

[llm]
provider = "local"

# -- Local provider (embedded llama.cpp) --
# model_file: GGUF model filename (auto-downloaded if auto_download = true)
        # model_file = "gemma-4-E2B-it-Q8_0.gguf"
# models_dir = "llm-mem-data/models"    # directory for model files
# gpu_layers = 0                         # GPU layers to offload (0 = CPU only)
# context_size = 16644                   # context window in tokens
# cpu_threads = 0                        # 0 = auto-detect
# auto_download = true                   # auto-download models from HuggingFace

# -- API provider (uncomment and set provider = "api" to use) --
# api_url = "https://api.openai.com/v1"  # API endpoint URL
# api_key = ""                            # or set LLM_MEM_LLM_API_KEY env var
# model = "gpt-4o-mini"                  # model identifier
#
# Common api_url values:
#   OpenAI:       https://api.openai.com/v1
#   OpenRouter:   https://openrouter.ai/api/v1
#   llama-server: http://localhost:8080/v1
#   Ollama:       http://localhost:11434/v1

# -- Generation settings (both providers) --
# temperature = 0.7
# max_tokens = 4096

# -- Advanced settings --
# request_format = "auto"                # "auto", "rig", or "raw" (API only)
# api_dialect = "openai-chat"            # API protocol: openai-chat, openai-completion,
#                                        #   anthropic, ollama-chat, ollama-completion, custom
#
# -- Custom dialect (only used when api_dialect = "custom") --
# custom_dialect = { endpoint_path = "/v1/chat/completions",
#                    request_body_template = '''
#                    { "model": "{{model}}", "messages": [{"role": "user", "content": "{{prompt}}"}],
#                      "temperature": {{temperature}}, "max_tokens": {{max_tokens}} }''',
#                    response_content_pointer = "/choices/0/message/content" }
#
# use_structured_output = true           # JSON schema validation (API only)
# structured_output_retries = 2          # retries for structured output
# max_concurrent_requests = 1            # concurrent API requests (0 = unlimited)
# strip_tags = ["think"]                 # XML tags to strip from LLM output
# batch_size = 10                        # items per batch request
# batch_max_tokens = 3000                # tokens per batch request
# batch_timeout_secs = 120               # base timeout for batch calls
# batch_timeout_multiplier = 1.0         # timeout multiplier
# use_grammar = false                    # grammar-constrained sampling (local only)
# cache_model = true                     # cache downloaded models (local only)
# cache_dir = ""                         # custom cache directory for model files (default: ~/.cache/llm-mem/models)
# proxy_url = ""                         # proxy URL for model downloads (overrides HTTPS_PROXY env var)
# llm_timeout_secs = 120                 # completion timeout
#
# -- Vision / image description (local only) --
# vision_enabled = false                 # Enable AI-powered image description via LLM.
#                                        # When true, ingested images (PNG/JPEG/GIF/WebP) get an
#                                        # AI-generated description stored alongside the original.
#                                        # Requires a vision-capable model and its mmproj file.
# mmproj_file = ""                       # Path to multimodal projection GGUF file for vision.
#                                        # Leave empty to auto-use mmproj-F16.gguf (Gemma 4 E2B).
#                                        # Auto-downloaded if missing when auto_download = true.
#                                        # The default mmproj is from:
#                                        #   huggingface.co/unsloth/gemma-4-E2B-it-GGUF

# ── Embedding ─────────────────────────────────────────────────────────────────
# The embedding model used for vector similarity search across memories.
#
# provider = "local"   → fastembed (default, works offline)
# provider = "api"     → OpenAI-compatible embedding API

[embedding]
provider = "local"

# model = "all-MiniLM-L6-v2"            # fastembed model name (local)
#
# -- API provider (uncomment and set provider = "api" to use) --
# api_url = "https://api.openai.com/v1"
# api_key = ""                            # or set LLM_MEM_EMBEDDING_API_KEY env var
# model = "text-embedding-3-small"
# batch_size = 64
# timeout_secs = 30
#
# -- Advanced settings (API only) --
# request_format = "auto"                # "auto", "rig", or "raw"
# api_dialect = "openai-chat"            # API protocol: openai-chat, openai-completion,
#                                        #   anthropic, ollama-chat, ollama-completion, custom
#
# -- Custom dialect (only used when api_dialect = "custom") --
# custom_dialect = { endpoint_path = "/v1/embeddings",
#                    request_body_template = '''
#                    { "model": "{{model}}", "input": {{prompt}} }''',
#                    response_content_pointer = "/data" }

# ── Vector Store ──────────────────────────────────────────────────────────────
# [vector_store]
# banks_dir = "llm-mem-data/banks"       # directory for memory bank databases
# collection_name = "llm-memories"
# store_type = "lancedb"                 # or "vectorlite" (legacy)
#
# [vector_store.lancedb]
# table_name = "memories"
# database_path = "./lancedb"
# embedding_dimension = 384              # must match your embedding model
#
# [vector_store.vectorlite]              # legacy VectorLite config
# index_type = "hnsw"
# metric = "cosine"

# ── Memory ────────────────────────────────────────────────────────────────────
# [memory]
# max_memories = 10000
# similarity_threshold = 0.65            # deduplication threshold (0.0 - 1.0)
# search_similarity_threshold = 0.2      # search relevance threshold (0.0 - 1.0)
# max_search_results = 50
# auto_enhance = true                    # auto-enrich memories with metadata
# deduplicate = true
# merge_threshold = 0.75                 # similarity to merge memories (0.0 - 1.0)
# auto_summary_threshold = 32768
# max_content_length = 32768
# document_chunk_size = 2000
#
# -- LLM-assisted ingestion (off by default to avoid unexpected API costs) --
# llm_format_detection = false           # When content format can't be detected by
#                                        # extension or MIME sniffing, ask the LLM to
#                                        # identify it. Sends first 4KB of content.
#                                        # Useful for files with missing/wrong extensions.
# llm_fallback_parsing = false           # When a structured parser fails (e.g., malformed
#                                        # JSON, damaged file), ask the LLM to extract
#                                        # summary, entities, and content type from the raw
#                                        # text. The original content is preserved alongside
#                                        # the LLM's analysis. Requires an API or local LLM
#                                        # to be configured in [llm].

# ── Server ────────────────────────────────────────────────────────────────────
# [server]
# host = "0.0.0.0"
# port = 8000

# ── Logging ───────────────────────────────────────────────────────────────────
# [logging]
# enabled = false
# log_directory = "llm-mem-data/logs"
# level = "info"
# max_size_mb = 1
# max_files = 5
"#
        .to_string()
    }

    /// Load configuration from a TOML file, then override with environment variables
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut config: Config = toml::from_str(&content)?;
        config.apply_env_overrides();
        config.validate()?;
        Ok(config)
    }

    /// Override config values from environment variables.
    ///
    /// Supported env vars:
    /// - `LLM_MEM_LLM_API_KEY` → `llm.api_key`
    /// - `LLM_MEM_LLM_API_BASE_URL` → `llm.api_url`
    /// - `LLM_MEM_LLM_MODEL` → `llm.model`
    /// - `LLM_MEM_EMBEDDING_API_KEY` → `embedding.api_key`
    /// - `LLM_MEM_EMBEDDING_API_BASE_URL` → `embedding.api_url`
    /// - `LLM_MEM_EMBEDDING_MODEL` → `embedding.model`
    /// - `LLM_MEM_MODELS_DIR` → `llm.models_dir`
    /// - `LLM_MEM_GPU_LAYERS` → `llm.gpu_layers`
    /// - `LLM_MEM_CONTEXT_SIZE` → `llm.context_size`
    /// - `LLM_MEM_TEMPERATURE` → `llm.temperature`
    /// - `LLM_MEM_MAX_TOKENS` → `llm.max_tokens`
    /// - `LLM_MEM_CPU_THREADS` → `llm.cpu_threads`
    /// - `LLM_MEM_MAX_CONCURRENT_REQUESTS` → `llm.max_concurrent_requests`
    /// - `OPENAI_API_KEY` → fallback for both `llm.api_key` and `embedding.api_key`
    pub fn apply_env_overrides(&mut self) {
        // LLM local overrides
        if let Ok(val) = std::env::var("LLM_MEM_MODELS_DIR") {
            self.llm.models_dir = val;
        }
        if let Ok(val) = std::env::var("LLM_MEM_GPU_LAYERS") {
            if let Ok(layers) = val.parse::<u32>() {
                info!(
                    "Environment override: LLM_MEM_GPU_LAYERS={} -> gpu_layers={}",
                    val, layers
                );
                self.llm.gpu_layers = layers;
            } else {
                warn!("Invalid LLM_MEM_GPU_LAYERS value: {} (expected u32)", val);
            }
        }
        if let Ok(val) = std::env::var("LLM_MEM_CONTEXT_SIZE")
            && let Ok(size) = val.parse::<u32>()
        {
            self.llm.context_size = size;
        }
        if let Ok(val) = std::env::var("LLM_MEM_TEMPERATURE")
            && let Ok(temp) = val.parse::<f32>()
        {
            self.llm.temperature = temp;
        }
        if let Ok(val) = std::env::var("LLM_MEM_MAX_TOKENS")
            && let Ok(tokens) = val.parse::<u32>()
        {
            self.llm.max_tokens = tokens;
        }
        if let Ok(val) = std::env::var("LLM_MEM_CPU_THREADS")
            && let Ok(threads) = val.parse::<i32>()
        {
            self.llm.cpu_threads = threads;
        }
        if let Ok(val) = std::env::var("LLM_MEM_MAX_CONCURRENT_REQUESTS")
            && let Ok(count) = val.parse::<usize>()
        {
            self.llm.max_concurrent_requests = count;
        }

        // LLM API overrides
        if let Ok(val) = std::env::var("LLM_MEM_LLM_API_KEY") {
            self.llm.api_key = val;
        }
        if let Ok(val) = std::env::var("LLM_MEM_LLM_API_BASE_URL") {
            self.llm.api_url = val;
        }
        if let Ok(val) = std::env::var("LLM_MEM_LLM_MODEL") {
            self.llm.model = val;
        }

        // Embedding overrides
        if let Ok(val) = std::env::var("LLM_MEM_EMBEDDING_API_KEY") {
            self.embedding.api_key = val;
        }
        if let Ok(val) = std::env::var("LLM_MEM_EMBEDDING_API_BASE_URL") {
            self.embedding.api_url = val;
        }
        if let Ok(val) = std::env::var("LLM_MEM_EMBEDDING_MODEL") {
            self.embedding.model = val;
        }

        // Fallback: OPENAI_API_KEY fills any still-empty keys
        if let Ok(val) = std::env::var("OPENAI_API_KEY") {
            if self.llm.api_key.is_empty() {
                self.llm.api_key = val.clone();
            }
            if self.embedding.api_key.is_empty() {
                self.embedding.api_key = val;
            }
        }
    }

    /// Determine the effective backend based on provider configuration.
    pub fn effective_backend(&self) -> LLMBackend {
        match (&self.llm.provider, &self.embedding.provider) {
            (ProviderType::Local, ProviderType::Local) => LLMBackend::Local,
            (ProviderType::Api, ProviderType::Api) => LLMBackend::API,
            (ProviderType::Api, ProviderType::Local) => LLMBackend::APILLMLocalEmbed,
            (ProviderType::Local, ProviderType::Api) => LLMBackend::LocalLLMAPIEmbed,
        }
    }

    /// Validate that required configuration values are present and in valid ranges
    pub fn validate(&self) -> Result<()> {
        // Build-time feature check: a config that asks for `provider = "local"`
        // must be running in a build that has the corresponding feature enabled.
        // Catches misconfigurations early instead of failing deep inside the
        // LLM-client factory.
        if self.llm.provider == ProviderType::Local {
            #[cfg(not(feature = "local-llm"))]
            {
                bail!(
                    "Config sets llm.provider = \"local\" but this build does not have the 'local-llm' feature enabled.\n\
                     Rebuild with: cargo build --features local-llm\n\
                     Or use the 'local' aggregate: cargo build --features local\n\
                     Or change llm.provider to \"api\" in config.toml."
                );
            }
        }
        if self.embedding.provider == ProviderType::Local {
            #[cfg(not(feature = "local-embed"))]
            {
                bail!(
                    "Config sets embedding.provider = \"local\" but this build does not have the 'local-embed' feature enabled.\n\
                     Rebuild with: cargo build --features local-embed\n\
                     Or use the 'local' aggregate: cargo build --features local\n\
                     Or change embedding.provider to \"api\" in config.toml."
                );
            }
        }

        match self.effective_backend() {
            LLMBackend::API => {
                if self.llm.api_key.is_empty() {
                    bail!(
                        "LLM API key is not configured.\n\
                         Set it in config.toml under [llm].api_key, \
                         or via env var LLM_MEM_LLM_API_KEY or OPENAI_API_KEY.\n\
                         API URL: {}",
                        self.llm.api_url
                    );
                }
                if self.embedding.api_key.is_empty() {
                    bail!(
                        "Embedding API key is not configured.\n\
                         Set it in config.toml under [embedding].api_key, \
                         or via env var LLM_MEM_EMBEDDING_API_KEY or OPENAI_API_KEY.\n\
                         API URL: {}",
                        self.embedding.api_url
                    );
                }
            }
            LLMBackend::APILLMLocalEmbed => {
                if self.llm.api_key.is_empty() {
                    bail!(
                        "LLM API key is not configured (llm.provider = \"api\").\n\
                         Set it in config.toml under [llm].api_key, \
                         or via env var LLM_MEM_LLM_API_KEY or OPENAI_API_KEY.\n\
                         API URL: {}",
                        self.llm.api_url
                    );
                }
                if self.embedding.model.is_empty() {
                    bail!(
                        "Local embedding model is not configured.\n\
                         Set model in [embedding] section (e.g., \"all-MiniLM-L6-v2\")"
                    );
                }
            }
            LLMBackend::LocalLLMAPIEmbed => {
                if self.embedding.api_key.is_empty() {
                    bail!(
                        "Embedding API key is not configured (embedding.provider = \"api\").\n\
                         Set it in config.toml under [embedding].api_key, \
                         or via env var LLM_MEM_EMBEDDING_API_KEY or OPENAI_API_KEY.\n\
                         API URL: {}",
                        self.embedding.api_url
                    );
                }
                if self.llm.model_file.is_empty() {
                    bail!(
                        "Local LLM model file is not configured.\n\
                         Set model_file in [llm] section"
                    );
                }
            }
            LLMBackend::Local => {
                // No API keys needed for fully local backend
            }
        }

        // Numeric bounds checks
        if self.memory.similarity_threshold < 0.0 || self.memory.similarity_threshold > 1.0 {
            bail!(
                "memory.similarity_threshold must be between 0.0 and 1.0 (got {})",
                self.memory.similarity_threshold
            );
        }
        if self.memory.merge_threshold < 0.0 || self.memory.merge_threshold > 1.0 {
            bail!(
                "memory.merge_threshold must be between 0.0 and 1.0 (got {})",
                self.memory.merge_threshold
            );
        }
        if let Some(thresh) = self.memory.search_similarity_threshold
            && !(0.0..=1.0).contains(&thresh)
        {
            bail!(
                "memory.search_similarity_threshold must be between 0.0 and 1.0 (got {})",
                thresh
            );
        }
        if self.llm.temperature < 0.0 || self.llm.temperature > 2.0 {
            bail!(
                "llm.temperature must be between 0.0 and 2.0 (got {})",
                self.llm.temperature
            );
        }
        if self.llm.context_size == 0 {
            bail!("llm.context_size must be greater than 0");
        }
        if self.llm.max_tokens == 0 {
            bail!("llm.max_tokens must be greater than 0");
        }
        if self.llm.batch_max_tokens == 0 {
            bail!("llm.batch_max_tokens must be greater than 0");
        }
        if self.llm.batch_max_tokens > self.llm.max_tokens {
            bail!(
                "llm.batch_max_tokens ({}) cannot be greater than llm.max_tokens ({})",
                self.llm.batch_max_tokens,
                self.llm.max_tokens
            );
        }
        if self.embedding.batch_size == 0 {
            bail!("embedding.batch_size must be greater than 0");
        }
        if self.memory.max_memories == 0 {
            bail!("memory.max_memories must be greater than 0");
        }
        if self.memory.max_content_length == 0 {
            bail!("memory.max_content_length must be greater than 0");
        }

        // Context Size Safety Guard
        let uses_local_llm = matches!(
            self.effective_backend(),
            LLMBackend::Local | LLMBackend::LocalLLMAPIEmbed
        );
        if uses_local_llm {
            let estimated_chunk_tokens = self.memory.document_chunk_size / 2;
            let required_min_context = estimated_chunk_tokens as u32 + self.llm.max_tokens + 512;

            if self.llm.context_size < required_min_context {
                bail!(
                    "Configuration Error: llm.context_size ({}) is too small for the configured \
                     memory.document_chunk_size ({}) and llm.max_tokens ({}).\n\n\
                     Required minimum context: ~{} tokens.\n\
                     Please increase 'llm.context_size' in your config.toml.",
                    self.llm.context_size,
                    self.memory.document_chunk_size,
                    self.llm.max_tokens,
                    required_min_context
                );
            }
        }

        if self.memory.chunk_threshold_chars > 0
            && self.memory.chunk_size_chars > 0
            && self.memory.chunk_threshold_chars > self.memory.chunk_size_chars
        {
            tracing::warn!(
                "memory.chunk_threshold_chars ({}) is greater than memory.chunk_size_chars ({}). \
                 This allows L0 memories up to {} chars to go unsplit even though the embedding \
                 model's safe window is only {} chars. Consider setting chunk_threshold_chars \
                 <= chunk_size_chars to guarantee full retrieval coverage.",
                self.memory.chunk_threshold_chars,
                self.memory.chunk_size_chars,
                self.memory.chunk_threshold_chars,
                self.memory.chunk_size_chars
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex;
    use tempfile::NamedTempFile;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn create_temp_config(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file
    }

    fn valid_toml() -> &'static str {
        r#"
[llm]
provider = "api"
api_url = "https://api.openai.com/v1"
api_key = "sk-test-key"
model = "gpt-4o-mini"
temperature = 0.5
max_tokens = 1024
batch_max_tokens = 1024

[embedding]
provider = "api"
api_url = "https://api.openai.com/v1"
api_key = "sk-embed-key"
model = "text-embedding-3-small"
batch_size = 32
timeout_secs = 60

[vector_store]
store_type = "lancedb"
collection_name = "test-memories"

[vector_store.lancedb]
table_name = "test_memories"
database_path = "./test_lancedb"
embedding_dimension = 384

[vector_store.vectorlite]
index_type = "hnsw"
metric = "cosine"

[memory]
max_memories = 5000
similarity_threshold = 0.7
max_search_results = 25
auto_enhance = false
deduplicate = false
merge_threshold = 0.8

[server]
host = "127.0.0.1"
port = 9000

[logging]
enabled = true
log_directory = "/tmp/logs"
level = "debug"
"#
    }

    fn clear_env_vars() {
        for var in &[
            "LLM_MEM_LLM_API_KEY",
            "LLM_MEM_LLM_API_BASE_URL",
            "LLM_MEM_LLM_MODEL",
            "LLM_MEM_EMBEDDING_API_KEY",
            "LLM_MEM_EMBEDDING_API_BASE_URL",
            "LLM_MEM_EMBEDDING_MODEL",
            "OPENAI_API_KEY",
        ] {
            unsafe {
                std::env::remove_var(var);
            }
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = Config::default();

        assert_eq!(config.llm.provider, ProviderType::Local);
        assert_eq!(config.llm.api_url, "https://api.openai.com/v1");
        assert!(config.llm.api_key.is_empty());
        assert_eq!(config.llm.model, "gpt-4o-mini");
        assert_eq!(config.llm.model_file, "gemma-4-E2B-it-Q8_0.gguf");
        assert_eq!(config.llm.temperature, 0.7);
        assert_eq!(config.llm.max_tokens, 4096);

        assert_eq!(config.embedding.provider, ProviderType::Local);
        assert_eq!(config.embedding.model, "all-MiniLM-L6-v2");
        assert_eq!(config.embedding.batch_size, 64);
        assert_eq!(config.embedding.timeout_secs, 30);

        assert_eq!(config.vector_store.collection_name, "llm-memories");
        assert_eq!(config.memory.max_memories, 10000);
        assert_eq!(config.server.host, "0.0.0.0");
        assert!(!config.logging.enabled);
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.llm.model, config.llm.model);
        assert_eq!(parsed.embedding.model, config.embedding.model);
        assert_eq!(parsed.memory.max_memories, config.memory.max_memories);
    }

    #[test]
    fn test_vectorstore_type_default() {
        let vst: VectorStoreType = VectorStoreType::default();
        assert!(matches!(vst, VectorStoreType::LanceDB));
    }

    #[test]
    fn test_memory_config_defaults() {
        let mc = MemoryConfig::default();
        assert!(mc.memory_ttl_hours.is_none());
        assert_eq!(mc.auto_summary_threshold, 32768);
        assert_eq!(mc.search_similarity_threshold, Some(0.2));
    }

    #[test]
    fn test_config_file_not_found() {
        let result = Config::load("/nonexistent/path/config.toml");
        assert!(result.is_err());
    }

    #[test]
    fn test_vectorlite_config_from_store_config() {
        let store_cfg = VectorStoreConfig {
            store_type: VectorStoreType::Vectorlite,
            collection_name: "my-collection".to_string(),
            vectorlite: VectorLiteSettings {
                index_type: "flat".to_string(),
                metric: "euclidean".to_string(),
                persistence_path: Some("/tmp/data".to_string()),
                embedding_dimension: None,
            },
            lancedb: LanceDBSettings::default(),
            banks_dir: "test-banks".to_string(),
        };

        #[cfg(feature = "vector-lite")]
        let vl_cfg = crate::vector_store::VectorLiteConfig::from_store_config(&store_cfg);
        #[cfg(feature = "vector-lite")]
        {
            assert_eq!(vl_cfg.collection_name, "my-collection");
            assert_eq!(
                vl_cfg.persistence_path,
                Some(std::path::PathBuf::from("/tmp/data"))
            );
        }

        #[cfg(not(feature = "vector-lite"))]
        let _ = store_cfg; // Suppress unused variable warning
    }

    #[test]
    fn test_logging_config_defaults() {
        let config = LoggingConfig::default();
        assert_eq!(config.max_size_mb, 1);
        assert_eq!(config.max_files, 5);
        assert!(!config.enabled);
    }

    #[test]
    fn test_llm_config_defaults() {
        let lc = LlmConfig::default();
        assert_eq!(lc.models_dir, "llm-mem-data/models");
        assert_eq!(lc.model_file, "gemma-4-E2B-it-Q8_0.gguf");
        assert_eq!(lc.gpu_layers, 0);
        assert_eq!(lc.context_size, 16644);
        assert!((lc.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(lc.max_tokens, 4096);
        assert!(lc.proxy_url.is_none());
        assert!(lc.auto_download);
        assert!(!lc.use_grammar);
        assert_eq!(lc.strip_tags.len(), 1);
        assert_eq!(lc.strip_tags[0], "think");
    }

    #[test]
    fn test_llm_config_serialization_roundtrip() {
        let lc = LlmConfig::default();
        let toml_str = toml::to_string(&lc).unwrap();
        let restored: LlmConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(restored.models_dir, lc.models_dir);
        assert_eq!(restored.model_file, lc.model_file);
        assert_eq!(restored.gpu_layers, lc.gpu_layers);
        assert_eq!(restored.context_size, lc.context_size);
    }

    #[test]
    fn test_effective_backend_both_local() {
        let config = Config::default();
        assert_eq!(config.effective_backend(), LLMBackend::Local);
    }

    #[test]
    fn test_effective_backend_both_api() {
        let config = Config {
            llm: LlmConfig {
                provider: ProviderType::Api,
                ..Default::default()
            },
            embedding: EmbeddingConfig {
                provider: ProviderType::Api,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(config.effective_backend(), LLMBackend::API);
    }

    #[test]
    fn test_effective_backend_api_llm_local_embed() {
        let config = Config {
            llm: LlmConfig {
                provider: ProviderType::Api,
                ..Default::default()
            },
            embedding: EmbeddingConfig {
                provider: ProviderType::Local,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(config.effective_backend(), LLMBackend::APILLMLocalEmbed);
    }

    #[test]
    fn test_effective_backend_local_llm_api_embed() {
        let config = Config {
            llm: LlmConfig {
                provider: ProviderType::Local,
                ..Default::default()
            },
            embedding: EmbeddingConfig {
                provider: ProviderType::Api,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(config.effective_backend(), LLMBackend::LocalLLMAPIEmbed);
    }

    #[test]
    fn test_llm_backend_enum_serde() {
        let local: LLMBackend = serde_json::from_str(r#""local""#).unwrap();
        assert_eq!(local, LLMBackend::Local);
        let api: LLMBackend = serde_json::from_str(r#""openai""#).unwrap();
        assert_eq!(api, LLMBackend::API);
        let api2: LLMBackend = serde_json::from_str(r#""api""#).unwrap();
        assert_eq!(api2, LLMBackend::API);
    }

    #[test]
    fn test_local_backend_validation_passes_without_api_keys() {
        // Config::default() is the local backend, which requires both
        // `local-llm` and `local-embed` features. Under no-local builds
        // validation correctly rejects this; this test only runs on builds
        // that support the local backend.
        #[cfg(all(feature = "local-llm", feature = "local-embed"))]
        {
            let config = Config::default();
            assert!(config.validate().is_ok());
        }
    }

    #[test]
    fn test_api_dialect_defaults_and_serde() {
        let config = LlmConfig::default();
        assert_eq!(config.api_dialect, ApiDialect::OpenAIChat);

        let json = r#"{"api_dialect": "anthropic"}"#;
        let decoded: LlmConfig = serde_json::from_str(json).unwrap();
        assert_eq!(decoded.api_dialect, ApiDialect::Anthropic);

        let json_custom = r#"{
            "api_dialect": "custom",
            "custom_dialect": {
                "endpoint_path": "/gen",
                "request_body_template": "{\"q\": \"{{prompt}}\"}",
                "response_content_pointer": "/ans"
            }
        }"#;
        let decoded_custom: LlmConfig = serde_json::from_str(json_custom).unwrap();
        assert_eq!(decoded_custom.api_dialect, ApiDialect::Custom);
        let custom = decoded_custom.custom_dialect.unwrap();
        assert_eq!(custom.endpoint_path, "/gen");
    }

    #[test]
    fn test_config_load_and_env_overrides() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env_vars();

        // 1. Load valid full config
        {
            let file = create_temp_config(valid_toml());
            let config = Config::load(file.path()).unwrap();

            assert_eq!(config.llm.api_key, "sk-test-key");
            assert_eq!(config.llm.api_url, "https://api.openai.com/v1");
            assert_eq!(config.llm.model, "gpt-4o-mini");
            assert_eq!(config.llm.temperature, 0.5);
            assert_eq!(config.llm.max_tokens, 1024);

            assert_eq!(config.embedding.api_key, "sk-embed-key");
            assert_eq!(config.embedding.model, "text-embedding-3-small");
            assert_eq!(config.embedding.batch_size, 32);
            assert_eq!(config.embedding.timeout_secs, 60);
        }

        // 2. API provider with missing LLM API key fails
        {
            let toml = "[llm]\nprovider = \"api\"\napi_key = \"\"\n[embedding]\nprovider = \"api\"\napi_key = \"sk-embed\"\n";
            let file = create_temp_config(toml);
            assert!(Config::load(file.path()).is_err());
        }

        // 3. Local providers don't need API keys
        #[cfg(all(feature = "local-llm", feature = "local-embed"))]
        {
            let toml = "[llm]\nprovider = \"local\"\n[embedding]\nprovider = \"local\"\n";
            let file = create_temp_config(toml);
            let config = Config::load(file.path()).unwrap();
            assert_eq!(config.effective_backend(), LLMBackend::Local);
        }

        // 4. OPENAI_API_KEY fallback
        {
            let toml = "[llm]\nprovider = \"api\"\napi_key = \"\"\n[embedding]\nprovider = \"api\"\napi_key = \"\"\n";
            let file = create_temp_config(toml);
            unsafe {
                std::env::set_var("OPENAI_API_KEY", "sk-from-env");
            }
            let config = Config::load(file.path()).unwrap();
            assert_eq!(config.llm.api_key, "sk-from-env");
            assert_eq!(config.embedding.api_key, "sk-from-env");
            clear_env_vars();
        }

        // 5. Specific env vars override file values
        {
            let toml = "[llm]\nprovider = \"api\"\napi_key = \"from-file\"\napi_url = \"https://old.api.com/v1\"\nmodel = \"old-model\"\n[embedding]\nprovider = \"api\"\napi_key = \"from-file\"\napi_url = \"https://old-embed.api.com/v1\"\nmodel = \"old-embed-model\"\n";
            let file = create_temp_config(toml);
            unsafe {
                std::env::set_var("LLM_MEM_LLM_API_KEY", "sk-env-llm");
                std::env::set_var("LLM_MEM_LLM_API_BASE_URL", "https://new.api.com/v1");
                std::env::set_var("LLM_MEM_LLM_MODEL", "gpt-5");
                std::env::set_var("LLM_MEM_EMBEDDING_API_KEY", "sk-env-embed");
                std::env::set_var(
                    "LLM_MEM_EMBEDDING_API_BASE_URL",
                    "https://new-embed.api.com/v1",
                );
                std::env::set_var("LLM_MEM_EMBEDDING_MODEL", "new-embed-model");
            }
            let config = Config::load(file.path()).unwrap();
            assert_eq!(config.llm.api_key, "sk-env-llm");
            assert_eq!(config.llm.api_url, "https://new.api.com/v1");
            assert_eq!(config.llm.model, "gpt-5");
            assert_eq!(config.embedding.api_key, "sk-env-embed");
            assert_eq!(config.embedding.api_url, "https://new-embed.api.com/v1");
            assert_eq!(config.embedding.model, "new-embed-model");
            clear_env_vars();
        }

        // 6. Specific key takes precedence over OPENAI_API_KEY
        {
            let toml = "[llm]\nprovider = \"api\"\napi_key = \"\"\n[embedding]\nprovider = \"api\"\napi_key = \"\"\n";
            let file = create_temp_config(toml);
            unsafe {
                std::env::set_var("LLM_MEM_LLM_API_KEY", "sk-specific-llm");
                std::env::set_var("OPENAI_API_KEY", "sk-fallback");
            }
            let config = Config::load(file.path()).unwrap();
            assert_eq!(config.llm.api_key, "sk-specific-llm");
            assert_eq!(config.embedding.api_key, "sk-fallback");
            clear_env_vars();
        }
    }

    #[test]
    fn test_local_provider_requires_local_features() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env_vars();

        // Mixed config: llm=local, embedding=api
        // Config::load() calls validate() internally, so a feature mismatch
        // surfaces as a load error before the factory is ever called.
        let toml =
            "[llm]\nprovider = \"local\"\n[embedding]\nprovider = \"api\"\napi_key = \"sk\"\n";
        let file = create_temp_config(toml);
        let result = Config::load(file.path());

        #[cfg(feature = "local-llm")]
        assert!(
            result.is_ok(),
            "local-llm build must accept llm.provider = \"local\": {:?}",
            result.err()
        );

        #[cfg(not(feature = "local-llm"))]
        {
            let err = result.expect_err(
                "build without local-llm must reject llm.provider = \"local\" at config-load time",
            );
            let msg = err.to_string();
            assert!(
                msg.contains("local-llm"),
                "error should mention the feature: {}",
                msg
            );
            assert!(
                msg.contains("llm.provider"),
                "error should mention the offending field: {}",
                msg
            );
        }

        // Same on the embedding side
        let toml =
            "[llm]\nprovider = \"api\"\napi_key = \"sk\"\n[embedding]\nprovider = \"local\"\n";
        let file = create_temp_config(toml);
        let result = Config::load(file.path());

        #[cfg(feature = "local-embed")]
        assert!(
            result.is_ok(),
            "local-embed build must accept embedding.provider = \"local\": {:?}",
            result.err()
        );

        #[cfg(not(feature = "local-embed"))]
        {
            let err = result.expect_err(
                "build without local-embed must reject embedding.provider = \"local\" at config-load time",
            );
            let msg = err.to_string();
            assert!(
                msg.contains("local-embed"),
                "error should mention the feature: {}",
                msg
            );
            assert!(
                msg.contains("embedding.provider"),
                "error should mention the offending field: {}",
                msg
            );
        }
    }
}
