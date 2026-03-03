use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{info, warn};

/// Vector store backend type
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum VectorStoreType {
    #[default]
    Vectorlite,
}

/// LLM backend type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LLMBackend {
    Local,
    OpenAI,
}

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// Backend selection. If not specified, auto-detected:
    /// - OpenAI if API keys are present in [llm] and [embedding]
    /// - Local otherwise (uses llama.cpp + fastembed)
    #[serde(default)]
    pub backend: Option<LLMBackend>,
    /// Local inference configuration (used when backend = "local")
    #[serde(default)]
    pub local: LocalConfig,
    /// API-based LLM configuration (used when backend = "openai")
    #[serde(default)]
    pub api_llm: ApiLlmConfig,
    #[serde(default)]
    pub vector_store: VectorStoreConfig,
    #[serde(default)]
    pub llm: LLMConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
}

/// Local inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LocalConfig {
    /// Directory for model files (default: ./llm-mem-models)
    pub models_dir: String,
    /// LLM GGUF model filename
    pub llm_model_file: String,
    /// Embedding model name for fastembed (e.g. "all-MiniLM-L6-v2")
    pub embedding_model: String,
    /// Number of GPU layers to offload (0 = CPU only)
    pub gpu_layers: u32,
    /// Context window size in tokens
    pub context_size: u32,
    /// Sampling temperature
    pub temperature: f32,
    /// Maximum tokens to generate per completion
    pub max_tokens: u32,
    /// Proxy URL for model downloads (overrides HTTPS_PROXY env var)
    /// Format: http://host:port or http://user:pass@host:port
    pub proxy_url: Option<String>,
    /// Whether to auto-download known models when missing (default: true)
    pub auto_download: bool,
    /// Whether to cache downloaded models (default: true)
    pub cache_model: bool,
    /// Custom directory for model caching (default: ~/.cache/llm-mem/models)
    pub cache_dir: Option<String>,
    /// Timeout in seconds for LLM completion calls (default: 120)
    pub llm_timeout_secs: u64,
    /// Maximum number of concurrent LLM requests (default: 1)
    pub max_concurrent_requests: usize,
    /// Number of CPU threads to use for inference (0 = auto-detect, default: 0)
    pub cpu_threads: i32,
    /// Use grammar-constrained sampling for structured output with local LLM (default: false)
    pub use_grammar: bool,
    /// XML tags to strip from LLM output (e.g., ["think", "reason", "thought"])
    pub strip_llm_tags: Vec<String>,
}

/// API-based LLM configuration (OpenAI, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiLlmConfig {
    /// Use API provider's structured output mode when available (default: true)
    /// For OpenAI: uses response_format with JSON Schema validation
    /// For other APIs: enables enhanced JSON extraction with retry logic
    pub use_structured_output: bool,
    /// Maximum retry attempts for structured output validation (default: 2)
    pub structured_output_retries: u32,
    /// XML tags to strip from LLM output (e.g., ["think", "reason", "thought"])
    pub strip_llm_tags: Vec<String>,
}

impl Default for ApiLlmConfig {
    fn default() -> Self {
        Self {
            use_structured_output: true,
            structured_output_retries: 2,
            strip_llm_tags: vec!["think".to_string()], // Default to stripping think tags
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    #[serde(default)]
    pub store_type: VectorStoreType,
    #[serde(default = "default_collection_name")]
    pub collection_name: String,
    #[serde(default)]
    pub vectorlite: VectorLiteSettings,
    /// Directory for memory bank database files (default: ./llm-mem-banks)
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
}

/// LLM configuration for rig framework
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LLMConfig {
    pub api_base_url: String,
    pub api_key: String,
    pub model_efficient: String,
    pub temperature: f32,
    pub max_tokens: u32,
}

/// HTTP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

/// Embedding service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    pub api_base_url: String,
    pub model_name: String,
    pub api_key: String,
    pub batch_size: usize,
    pub timeout_secs: u64,
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
    /// Chunk size (in characters) used for document ingestion (default: 4000)
    pub document_chunk_size: usize,
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

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            models_dir: "llm-mem-data/models".to_string(),
            llm_model_file: "Qwen3.5-2B-UD-Q6_K_XL.gguf".to_string(),
            embedding_model: "all-MiniLM-L6-v2".to_string(),
            gpu_layers: 0,
            context_size: 16644,
            temperature: 0.7,
            max_tokens: 1024,
            proxy_url: None,
            auto_download: true,
            cache_model: true,
            cache_dir: None,
            llm_timeout_secs: 120,
            max_concurrent_requests: 1,
            cpu_threads: 0, // 0 = auto-detect (uses all available cores)
            use_grammar: false, // Disabled - llama.cpp grammar can crash with certain models
            strip_llm_tags: vec!["think".to_string()], // Default to stripping think tags
        }
    }
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            store_type: VectorStoreType::default(),
            collection_name: default_collection_name(),
            vectorlite: VectorLiteSettings::default(),
            banks_dir: default_banks_dir(),
        }
    }
}

impl Default for VectorLiteSettings {
    fn default() -> Self {
        Self {
            index_type: default_index_type(),
            metric: default_metric(),
            persistence_path: None,
        }
    }
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            api_base_url: "https://api.openai.com/v1".to_string(),
            api_key: String::new(),
            model_efficient: "gpt-4o-mini".to_string(),
            temperature: 0.7,
            max_tokens: 2048,
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

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            api_base_url: "https://api.openai.com/v1".to_string(),
            model_name: "text-embedding-3-small".to_string(),
            api_key: String::new(),
            batch_size: 64,
            timeout_secs: 30,
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
            search_similarity_threshold: Some(0.35),
            max_content_length: 32768,
            document_chunk_size: 2000,
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

impl Config {
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
    /// - `LLM_MEM_LLM_API_BASE_URL` → `llm.api_base_url`
    /// - `LLM_MEM_LLM_MODEL` → `llm.model_efficient`
    /// - `LLM_MEM_EMBEDDING_API_KEY` → `embedding.api_key`
    /// - `LLM_MEM_EMBEDDING_API_BASE_URL` → `embedding.api_base_url`
    /// - `LLM_MEM_EMBEDDING_MODEL` → `embedding.model_name`
    /// - `LLM_MEM_MODELS_DIR` → `local.models_dir`
    /// - `LLM_MEM_GPU_LAYERS` → `local.gpu_layers`
    /// - `LLM_MEM_CONTEXT_SIZE` → `local.context_size`
    /// - `LLM_MEM_TEMPERATURE` → `local.temperature`
    /// - `LLM_MEM_MAX_TOKENS` → `local.max_tokens`
    /// - `LLM_MEM_CPU_THREADS` → `local.cpu_threads`
    /// - `LLM_MEM_MAX_CONCURRENT_REQUESTS` → `local.max_concurrent_requests`
    /// - `OPENAI_API_KEY` → fallback for both `llm.api_key` and `embedding.api_key`
    pub fn apply_env_overrides(&mut self) {
        // Local overrides
        if let Ok(val) = std::env::var("LLM_MEM_MODELS_DIR") {
            self.local.models_dir = val;
        }
        if let Ok(val) = std::env::var("LLM_MEM_GPU_LAYERS") {
            if let Ok(layers) = val.parse::<u32>() {
                info!("Environment override: LLM_MEM_GPU_LAYERS={} -> gpu_layers={}", val, layers);
                self.local.gpu_layers = layers;
            } else {
                warn!("Invalid LLM_MEM_GPU_LAYERS value: {} (expected u32)", val);
            }
        }
        if let Ok(val) = std::env::var("LLM_MEM_CONTEXT_SIZE") {
            if let Ok(size) = val.parse::<u32>() {
                self.local.context_size = size;
            }
        }
        if let Ok(val) = std::env::var("LLM_MEM_TEMPERATURE") {
            if let Ok(temp) = val.parse::<f32>() {
                self.local.temperature = temp;
            }
        }
        if let Ok(val) = std::env::var("LLM_MEM_MAX_TOKENS") {
            if let Ok(tokens) = val.parse::<u32>() {
                self.local.max_tokens = tokens;
            }
        }
        if let Ok(val) = std::env::var("LLM_MEM_CPU_THREADS") {
            if let Ok(threads) = val.parse::<i32>() {
                self.local.cpu_threads = threads;
            }
        }
        if let Ok(val) = std::env::var("LLM_MEM_MAX_CONCURRENT_REQUESTS") {
            if let Ok(count) = val.parse::<usize>() {
                self.local.max_concurrent_requests = count;
            }
        }

        // LLM overrides
        if let Ok(val) = std::env::var("LLM_MEM_LLM_API_KEY") {
            self.llm.api_key = val;
        }
        if let Ok(val) = std::env::var("LLM_MEM_LLM_API_BASE_URL") {
            self.llm.api_base_url = val;
        }
        if let Ok(val) = std::env::var("LLM_MEM_LLM_MODEL") {
            self.llm.model_efficient = val;
        }

        // Embedding overrides
        if let Ok(val) = std::env::var("LLM_MEM_EMBEDDING_API_KEY") {
            self.embedding.api_key = val;
        }
        if let Ok(val) = std::env::var("LLM_MEM_EMBEDDING_API_BASE_URL") {
            self.embedding.api_base_url = val;
        }
        if let Ok(val) = std::env::var("LLM_MEM_EMBEDDING_MODEL") {
            self.embedding.model_name = val;
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

    /// Determine the effective backend based on configuration.
    ///
    /// Priority:
    /// 1. Explicit `backend` field in config → use that
    /// 2. API keys present in [llm] + [embedding] → OpenAI
    /// 3. Otherwise → Local
    pub fn effective_backend(&self) -> LLMBackend {
        if let Some(ref backend) = self.backend {
            return backend.clone();
        }
        // Auto-detect: use OpenAI if both API keys are set
        if !self.llm.api_key.is_empty() && !self.embedding.api_key.is_empty() {
            LLMBackend::OpenAI
        } else {
            LLMBackend::Local
        }
    }

    /// Validate that required configuration values are present and in valid ranges
    fn validate(&self) -> Result<()> {
        match self.effective_backend() {
            LLMBackend::OpenAI => {
                if self.llm.api_key.is_empty() {
                    bail!(
                        "LLM API key is not configured.\n\
                         Set it in config.toml under [llm].api_key, \
                         or via env var LLM_MEM_LLM_API_KEY or OPENAI_API_KEY.\n\
                         API base URL: {}",
                        self.llm.api_base_url
                    );
                }
                if self.embedding.api_key.is_empty() {
                    bail!(
                        "Embedding API key is not configured.\n\
                         Set it in config.toml under [embedding].api_key, \
                         or via env var LLM_MEM_EMBEDDING_API_KEY or OPENAI_API_KEY.\n\
                         API base URL: {}",
                        self.embedding.api_base_url
                    );
                }
            }
            LLMBackend::Local => {
                // No API keys needed for local backend
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
        if let Some(thresh) = self.memory.search_similarity_threshold {
            if thresh < 0.0 || thresh > 1.0 {
                bail!(
                    "memory.search_similarity_threshold must be between 0.0 and 1.0 (got {})",
                    thresh
                );
            }
        }
        if self.local.temperature < 0.0 || self.local.temperature > 2.0 {
            bail!(
                "local.temperature must be between 0.0 and 2.0 (got {})",
                self.local.temperature
            );
        }
        if self.llm.temperature < 0.0 || self.llm.temperature > 2.0 {
            bail!(
                "llm.temperature must be between 0.0 and 2.0 (got {})",
                self.llm.temperature
            );
        }
        if self.local.context_size == 0 {
            bail!("local.context_size must be greater than 0");
        }
        if self.local.max_tokens == 0 {
            bail!("local.max_tokens must be greater than 0");
        }
        if self.llm.max_tokens == 0 {
            bail!("llm.max_tokens must be greater than 0");
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

        // Context Size Safety Guard:
        // Ensure context_size is large enough for (chunk_size / 2) + max_tokens + overhead
        // We use 2 chars per token as a conservative estimate (most tokens are 4 chars).
        if self.effective_backend() == LLMBackend::Local {
            let estimated_chunk_tokens = self.memory.document_chunk_size / 2;
            let required_min_context = estimated_chunk_tokens as u32 + self.local.max_tokens + 512; // 512 for instructions/prompt
            
            if self.local.context_size < required_min_context {
                bail!(
                    "Configuration Error: local.context_size ({}) is too small for the configured \
                     memory.document_chunk_size ({}) and local.max_tokens ({}).\n\n\
                     Required minimum context: ~{} tokens.\n\
                     Please increase 'local.context_size' in your config.toml.",
                    self.local.context_size,
                    self.memory.document_chunk_size,
                    self.local.max_tokens,
                    required_min_context
                );
            }
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

    /// All Config::load tests must be serialized because they interact with
    /// process-global environment variables.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn create_temp_config(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file
    }

    fn valid_toml() -> &'static str {
        r#"
[llm]
api_base_url = "https://api.openai.com/v1"
api_key = "sk-test-key"
model_efficient = "gpt-4o-mini"
temperature = 0.5
max_tokens = 1024

[embedding]
api_base_url = "https://api.openai.com/v1"
api_key = "sk-embed-key"
model_name = "text-embedding-3-small"
batch_size = 32
timeout_secs = 60

[vector_store]
store_type = "vectorlite"
collection_name = "test-memories"

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

    // ── Tests that DON'T call Config::load (no env var interaction) ──

    #[test]
    fn test_config_defaults() {
        let config = Config::default();

        assert_eq!(config.llm.api_base_url, "https://api.openai.com/v1");
        assert!(config.llm.api_key.is_empty());
        assert_eq!(config.llm.model_efficient, "gpt-4o-mini");
        assert_eq!(config.llm.temperature, 0.7);
        assert_eq!(config.llm.max_tokens, 2048);

        assert_eq!(config.embedding.model_name, "text-embedding-3-small");
        assert_eq!(config.embedding.batch_size, 64);
        assert_eq!(config.embedding.timeout_secs, 30);

        assert_eq!(config.vector_store.collection_name, "llm-memories");
        assert_eq!(config.vector_store.vectorlite.index_type, "hnsw");
        assert_eq!(config.vector_store.vectorlite.metric, "cosine");

        assert_eq!(config.memory.max_memories, 10000);
        assert_eq!(config.memory.similarity_threshold, 0.65);
        assert!(config.memory.auto_enhance);
        assert!(config.memory.deduplicate);
        assert_eq!(config.memory.merge_threshold, 0.75);

        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8000);

        assert!(!config.logging.enabled);
        assert_eq!(config.logging.level, "info");
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.llm.model_efficient, config.llm.model_efficient);
        assert_eq!(parsed.embedding.model_name, config.embedding.model_name);
        assert_eq!(parsed.memory.max_memories, config.memory.max_memories);
        assert_eq!(parsed.server.port, config.server.port);
    }

    #[test]
    fn test_vectorstore_type_default() {
        let vst: VectorStoreType = VectorStoreType::default();
        matches!(vst, VectorStoreType::Vectorlite);
    }

    #[test]
    fn test_memory_config_defaults() {
        let mc = MemoryConfig::default();
        assert!(mc.memory_ttl_hours.is_none());
        assert_eq!(mc.auto_summary_threshold, 32768);
        assert_eq!(mc.search_similarity_threshold, Some(0.35));
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
            },
            banks_dir: "test-banks".to_string(),
        };

        let vl_cfg = crate::vector_store::VectorLiteConfig::from_store_config(&store_cfg);
        assert_eq!(vl_cfg.collection_name, "my-collection");
        assert_eq!(
            vl_cfg.persistence_path,
            Some(std::path::PathBuf::from("/tmp/data"))
        );
    }

    // ── Local backend and effective_backend tests ──

    #[test]
    fn test_logging_config_defaults() {
        let config = LoggingConfig::default();
        assert_eq!(config.max_size_mb, 1);
        assert_eq!(config.max_files, 5);
        assert!(!config.enabled);
    }

    #[test]
    fn test_local_config_defaults() {
        let lc = LocalConfig::default();
        // Since models_dir is relative to the executable path which varies in tests
        assert_eq!(lc.models_dir, "llm-mem-data/models");
        assert_eq!(lc.llm_model_file, "Qwen3.5-2B-UD-Q6_K_XL.gguf");
        assert_eq!(lc.embedding_model, "all-MiniLM-L6-v2");
        assert_eq!(lc.gpu_layers, 0);
        assert_eq!(lc.context_size, 16644);
        assert!((lc.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(lc.max_tokens, 1024);
        assert!(lc.proxy_url.is_none());
        assert!(lc.auto_download);
        assert!(!lc.use_grammar); // Disabled by default
        assert_eq!(lc.strip_llm_tags.len(), 1);
        assert_eq!(lc.strip_llm_tags[0], "think");
    }

    #[test]
    fn test_local_config_serialization_roundtrip() {
        let lc = LocalConfig::default();
        let toml_str = toml::to_string(&lc).unwrap();
        let restored: LocalConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(restored.models_dir, lc.models_dir);
        assert_eq!(restored.llm_model_file, lc.llm_model_file);
        assert_eq!(restored.embedding_model, lc.embedding_model);
        assert_eq!(restored.gpu_layers, lc.gpu_layers);
        assert_eq!(restored.context_size, lc.context_size);
    }

    #[test]
    fn test_effective_backend_explicit_local() {
        let mut config = Config::default();
        config.backend = Some(LLMBackend::Local);
        assert_eq!(config.effective_backend(), LLMBackend::Local);
    }

    #[test]
    fn test_effective_backend_explicit_openai() {
        let mut config = Config::default();
        config.backend = Some(LLMBackend::OpenAI);
        assert_eq!(config.effective_backend(), LLMBackend::OpenAI);
    }

    #[test]
    fn test_effective_backend_auto_detect_openai_when_keys_present() {
        let mut config = Config::default();
        config.backend = None;
        config.llm.api_key = "sk-test".to_string();
        config.embedding.api_key = "sk-embed".to_string();
        assert_eq!(config.effective_backend(), LLMBackend::OpenAI);
    }

    #[test]
    fn test_effective_backend_auto_detect_local_when_no_keys() {
        let mut config = Config::default();
        config.backend = None;
        config.llm.api_key = String::new();
        config.embedding.api_key = String::new();
        assert_eq!(config.effective_backend(), LLMBackend::Local);
    }

    #[test]
    fn test_effective_backend_auto_detect_local_when_partial_keys() {
        // Only LLM key set, no embedding key => local
        let mut config = Config::default();
        config.backend = None;
        config.llm.api_key = "sk-test".to_string();
        config.embedding.api_key = String::new();
        assert_eq!(config.effective_backend(), LLMBackend::Local);

        // Only embedding key set, no LLM key => local
        config.llm.api_key = String::new();
        config.embedding.api_key = "sk-embed".to_string();
        assert_eq!(config.effective_backend(), LLMBackend::Local);
    }

    #[test]
    fn test_llm_backend_enum_serde() {
        let local: LLMBackend = serde_json::from_str(r#""local""#).unwrap();
        assert_eq!(local, LLMBackend::Local);
        let openai: LLMBackend = serde_json::from_str(r#""openai""#).unwrap();
        assert_eq!(openai, LLMBackend::OpenAI);

        let local_str = serde_json::to_string(&LLMBackend::Local).unwrap();
        assert_eq!(local_str, r#""local""#);
        let openai_str = serde_json::to_string(&LLMBackend::OpenAI).unwrap();
        assert_eq!(openai_str, r#""openai""#);
    }

    #[test]
    fn test_local_backend_validation_passes_without_api_keys() {
        let mut config = Config::default();
        config.backend = Some(LLMBackend::Local);
        config.llm.api_key = String::new();
        config.embedding.api_key = String::new();
        // Local backend should validate successfully without API keys
        assert!(config.validate().is_ok());
    }

    // ── Tests that call Config::load (env var sensitive, serialized) ──

    #[test]
    fn test_config_load_and_env_overrides() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env_vars();

        // 1. Load valid full config
        {
            let file = create_temp_config(valid_toml());
            let config = Config::load(file.path()).unwrap();

            assert_eq!(config.llm.api_key, "sk-test-key");
            assert_eq!(config.llm.api_base_url, "https://api.openai.com/v1");
            assert_eq!(config.llm.model_efficient, "gpt-4o-mini");
            assert_eq!(config.llm.temperature, 0.5);
            assert_eq!(config.llm.max_tokens, 1024);

            assert_eq!(config.embedding.api_key, "sk-embed-key");
            assert_eq!(config.embedding.model_name, "text-embedding-3-small");
            assert_eq!(config.embedding.batch_size, 32);
            assert_eq!(config.embedding.timeout_secs, 60);

            assert_eq!(config.vector_store.collection_name, "test-memories");
            assert_eq!(config.memory.max_memories, 5000);
            assert!(!config.memory.auto_enhance);

            assert_eq!(config.server.host, "127.0.0.1");
            assert_eq!(config.server.port, 9000);

            assert!(config.logging.enabled);
            assert_eq!(config.logging.level, "debug");
        }

        // 2. Explicit OpenAI backend with missing LLM API key fails validation
        {
            let toml = "backend = \"openai\"\n[llm]\napi_key = \"\"\n[embedding]\napi_key = \"sk-embed\"\n";
            let file = create_temp_config(toml);
            let result = Config::load(file.path());
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("LLM API key"));
        }

        // 3. Explicit OpenAI backend with missing embedding API key fails validation
        {
            let toml =
                "backend = \"openai\"\n[llm]\napi_key = \"sk-llm\"\n[embedding]\napi_key = \"\"\n";
            let file = create_temp_config(toml);
            let result = Config::load(file.path());
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("Embedding API key")
            );
        }

        // 3b. Missing API keys without explicit backend auto-detects as local (no error)
        {
            let toml = "[llm]\napi_key = \"\"\n[embedding]\napi_key = \"\"\n";
            let file = create_temp_config(toml);
            let config = Config::load(file.path()).unwrap();
            assert_eq!(config.effective_backend(), LLMBackend::Local);
        }

        // 4. Invalid TOML
        {
            let file = create_temp_config("this is not valid TOML {{{");
            assert!(Config::load(file.path()).is_err());
        }

        // 5. Partial TOML uses defaults
        {
            let toml = "[llm]\napi_key = \"sk-test\"\n[embedding]\napi_key = \"sk-test\"\n";
            let file = create_temp_config(toml);
            let config = Config::load(file.path()).unwrap();
            assert_eq!(config.llm.model_efficient, "gpt-4o-mini");
            assert_eq!(config.llm.temperature, 0.7);
            assert_eq!(config.memory.max_memories, 10000);
            assert_eq!(config.vector_store.collection_name, "llm-memories");
        }

        // 6. OPENAI_API_KEY fallback fills both keys
        {
            let toml = "[llm]\napi_key = \"\"\n[embedding]\napi_key = \"\"\n";
            let file = create_temp_config(toml);
            unsafe {
                std::env::set_var("OPENAI_API_KEY", "sk-from-env");
            }
            let config = Config::load(file.path()).unwrap();
            assert_eq!(config.llm.api_key, "sk-from-env");
            assert_eq!(config.embedding.api_key, "sk-from-env");
            clear_env_vars();
        }

        // 7. Specific env vars override file values
        {
            let toml = "[llm]\napi_key = \"from-file\"\napi_base_url = \"https://old.api.com/v1\"\nmodel_efficient = \"old-model\"\n[embedding]\napi_key = \"from-file\"\napi_base_url = \"https://old-embed.api.com/v1\"\nmodel_name = \"old-embed-model\"\n";
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
            assert_eq!(config.llm.api_base_url, "https://new.api.com/v1");
            assert_eq!(config.llm.model_efficient, "gpt-5");
            assert_eq!(config.embedding.api_key, "sk-env-embed");
            assert_eq!(
                config.embedding.api_base_url,
                "https://new-embed.api.com/v1"
            );
            assert_eq!(config.embedding.model_name, "new-embed-model");
            clear_env_vars();
        }

        // 8. Specific key takes precedence over OPENAI_API_KEY
        {
            let toml = "[llm]\napi_key = \"\"\n[embedding]\napi_key = \"\"\n";
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
}
