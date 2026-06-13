use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::Semaphore;

use async_trait::async_trait;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
#[allow(deprecated)]
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText, mtmd_default_marker,
};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use tracing::{debug, error, info};

use crate::config::LlmConfig;
use crate::error::{MemoryError, Result};
use crate::llm::extractor_types::*;

use super::EmbedPurpose;
use super::client::{LLMClient, UsageCounters};

/// Local LLM client using llama.cpp for completions and fastembed for embeddings.
///
/// This enables fully self-contained inference with no external API calls.
/// Models are loaded from disk (GGUF for LLM, ONNX for embeddings).
pub struct LocalLLMClient {
    model: Arc<LlamaModel>,
    // We keep a handle to the backend to prevent Drop, though it's static now
    backend: Arc<LlamaBackend>,
    embedding: Arc<Mutex<fastembed::TextEmbedding>>,
    config: LlmConfig,
    embedding_model_name: String,
    model_path: PathBuf,
    counters: UsageCounters,
    concurrency_limiter: Arc<Semaphore>,
    query_prefix: String,
    document_prefix: String,
}

// Global LlamaBackend instance to prevent multi-initialization errors in tests
static LLAMA_BACKEND: OnceLock<std::result::Result<Arc<LlamaBackend>, String>> = OnceLock::new();

impl Clone for LocalLLMClient {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            backend: Arc::clone(&self.backend),
            embedding: Arc::clone(&self.embedding),
            config: self.config.clone(),
            embedding_model_name: self.embedding_model_name.clone(),
            model_path: self.model_path.clone(),
            counters: self.counters.clone(),
            concurrency_limiter: Arc::clone(&self.concurrency_limiter),
            query_prefix: self.query_prefix.clone(),
            document_prefix: self.document_prefix.clone(),
        }
    }
}

/// Parameters for synchronous text generation
pub struct GenerateParams<'a> {
    pub model: &'a LlamaModel,
    pub backend: &'a LlamaBackend,
    pub prompt: &'a str,
    pub max_tokens: u32,
    pub temperature: f32,
    pub max_context_size: u32,
    pub cpu_threads: i32,
    pub grammar: Option<&'a str>,
}

/// Parameters for JSON extraction
pub struct ExtractJsonParams<'a> {
    pub model: &'a LlamaModel,
    pub backend: &'a LlamaBackend,
    pub prompt: &'a str,
    pub max_tokens: u32,
    pub temperature: f32,
    pub context_size: u32,
    pub cpu_threads: i32,
    pub strip_tags: &'a [String],
}

impl LocalLLMClient {
    /// Create a new local LLM client.
    ///
    /// - Ensures the GGUF model file exists (auto-downloads known models)
    /// - Initializes the llama.cpp backend and loads the GGUF model
    /// - Initializes fastembed for local embeddings (auto-downloads on first run)
    /// - Creates the models directory if it doesn't exist
    pub async fn new(
        config: &LlmConfig,
        embedding_model: &str,
        query_prefix: &str,
        document_prefix: &str,
    ) -> Result<Self> {
        let models_dir = PathBuf::from(&config.models_dir);

        // Create models directory if it doesn't exist
        std::fs::create_dir_all(&models_dir).map_err(|e| {
            MemoryError::config(format!(
                "Failed to create models directory '{}': {}",
                models_dir.display(),
                e
            ))
        })?;

        // Ensure model file exists — auto-download if it's a known model
        let model_path = if config.auto_download {
            let result = super::model_downloader::ensure_model(
                &models_dir,
                &config.model_file,
                config.proxy_url.as_deref(),
                config.cache_model,
                config.cache_dir.as_deref(),
            )
            .await?;

            if result.freshly_downloaded {
                info!(
                    "Model downloaded: {} ({})",
                    config.model_file,
                    super::model_downloader::format_size(result.size_bytes)
                );
            }

            result.path
        } else {
            let path = models_dir.join(&config.model_file);
            if !path.exists() {
                return Err(MemoryError::config(format!(
                    "LLM model file not found: {path}\n\n\
                     Auto-download is disabled (auto_download = false).\n\
                     Download the model manually or set auto_download = true in config.toml.\n\n\
                     Recommended (Qwen2.5 1.5B, ~1.1 GB):\n\
                       curl -L -o {path} \\\n\
                         https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                    path = path.display(),
                )));
            }
            path
        };

        info!("Initializing llama.cpp backend...");
        info!("GPU layers configured: {}", config.gpu_layers);

        // Ensure backend is initialized only once (process-wide)
        let backend_result = LLAMA_BACKEND.get_or_init(|| {
            // Forward llama.cpp internal logs to our tracing system
            llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());

            LlamaBackend::init()
                .map(Arc::new)
                .map_err(|e| format!("Failed to initialize llama.cpp backend: {}", e))
        });

        let backend = match backend_result {
            Ok(b) => b.clone(),
            Err(e) => return Err(MemoryError::LLM(e.clone())),
        };

        info!("Loading LLM model: {}", model_path.display());
        info!("Model params: gpu_layers={}", config.gpu_layers);
        let model_params = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers);
        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
            .map_err(|e| MemoryError::LLM(format!("Failed to load model: {}", e)))?;
        info!("LLM model loaded successfully");

        info!(
            "Initializing embedding model: {} (will download on first run)",
            embedding_model
        );

        // Use centralized cache directory for embedding model (consistent with LLM model caching)
        let embed_cache_dir = if config.cache_model {
            if let Some(custom) = &config.cache_dir {
                let p = PathBuf::from(custom);
                if p.is_relative() {
                    std::env::current_dir().unwrap_or_default().join(&p)
                } else {
                    p
                }
            } else if let Some(home) = dirs::home_dir() {
                home.join(".cache").join("llm-mem").join("models")
            } else {
                models_dir.clone()
            }
        } else {
            models_dir.clone()
        };

        let embed_model = super::fastembed_helpers::parse_fastembed_model(embedding_model);
        let embed_options = fastembed::InitOptions::new(embed_model)
            .with_cache_dir(embed_cache_dir.clone())
            .with_show_download_progress(true);
        let embedding = fastembed::TextEmbedding::try_new(embed_options).map_err(|e| {
            MemoryError::Embedding(format!("Failed to initialize embedding model: {}", e))
        })?;
        info!(
            "Embedding model initialized (cache: {})",
            embed_cache_dir.display()
        );

        let semaphore_permits = if config.max_concurrent_requests == 0 {
            1
        } else {
            config.max_concurrent_requests
        };

        info!(
            "Local LLM client ready (gpu_layers={}, ctx={}, concurrency={})",
            config.gpu_layers, config.context_size, semaphore_permits
        );

        Ok(Self {
            model: Arc::new(model),
            backend,
            embedding: Arc::new(Mutex::new(embedding)),
            config: config.clone(),
            embedding_model_name: embedding_model.to_string(),
            model_path,
            counters: UsageCounters::default(),
            concurrency_limiter: Arc::new(Semaphore::new(semaphore_permits)),
            query_prefix: query_prefix.to_string(),
            document_prefix: document_prefix.to_string(),
        })
    }

    // ── Synchronous inference helpers (run inside spawn_blocking) ───────

    /// Generate a text completion synchronously using llama.cpp.
    fn generate_sync(
        model: &LlamaModel,
        backend: &LlamaBackend,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        max_context_size: u32,
        cpu_threads: i32,
    ) -> Result<String> {
        Self::generate_sync_with_grammar(GenerateParams {
            model,
            backend,
            prompt,
            max_tokens,
            temperature,
            max_context_size,
            cpu_threads,
            grammar: None,
        })
    }

    /// Generate a text completion with optional grammar-constrained sampling.
    fn generate_sync_with_grammar(params: GenerateParams) -> Result<String> {
        let GenerateParams {
            model,
            backend,
            prompt,
            max_tokens,
            temperature,
            max_context_size,
            cpu_threads,
            grammar,
        } = params;

        let formatted = format_chatml_prompt(prompt);

        // 1. Tokenize first to determine required context size
        let tokens = model
            .str_to_token(&formatted, AddBos::Always)
            .map_err(|e| MemoryError::LLM(format!("Tokenization failed: {}", e)))?;

        if tokens.is_empty() {
            return Ok(String::new());
        }

        let prompt_tokens = tokens.len() as u32;
        let needed_tokens = prompt_tokens + max_tokens + 16; // +16 safety margin

        // 2. Dynamic context sizing with bins properties
        // Bins: 4096, 8192, 16384, 32768, etc.
        let mut context_size = 4096;
        while context_size < needed_tokens {
            context_size *= 2;
            if context_size > max_context_size {
                // If the next bin exceeds our hard limit, break.
                // We will check specifically below.
                break;
            }
        }

        // If the request requires more than the configured max context, reject it.
        // But if the binning logic jumped over the max, clamp it if it fits, or reject.
        if needed_tokens > max_context_size {
            return Err(MemoryError::LLM(format!(
                "Input too long: {} tokens required (prompt: {} + gen: {}), but max context is {}.\n\n\
                 To fix this, either:\n\
                 1. Increase 'local.context_size' in your config.toml\n\
                 2. Decrease 'memory.document_chunk_size' for future document ingestions.",
                needed_tokens, prompt_tokens, max_tokens, max_context_size
            )));
        }

        // Clamp to max_context_size if the bin overshoot it but we are still within limits
        if context_size > max_context_size {
            context_size = max_context_size;
        }

        // SAFETY: llama.cpp will abort (crash the process) if we try to decode a batch
        // larger than n_batch. We set n_batch = context_size to avoid this, but we
        // also check here to return a proper Result instead of crashing.
        if prompt_tokens > context_size {
            return Err(MemoryError::LLM(format!(
                "Prompt tokens ({}) exceed context window ({})",
                prompt_tokens, context_size
            )));
        }

        // 3. Create context with calculated size
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(context_size))
            .with_n_batch(context_size);

        // Set thread count (0 = auto-detect, uses all available cores)
        if cpu_threads > 0 {
            ctx_params = ctx_params.with_n_threads(cpu_threads);
        }

        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| MemoryError::LLM(format!("Context creation failed: {}", e)))?;

        // Fill batch with prompt tokens
        let mut batch = LlamaBatch::new(context_size as usize, 1);
        let last_idx = tokens.len() - 1;
        for (i, &token) in tokens.iter().enumerate() {
            batch
                .add(token, i as i32, &[0], i == last_idx)
                .map_err(|e| MemoryError::LLM(format!("Batch add failed: {}", e)))?;
        }

        // Decode prompt (prefill)
        ctx.decode(&mut batch)
            .map_err(|e| MemoryError::LLM(format!("Prompt decode failed: {}", e)))?;

        // Set up sampler with optional grammar
        let mut sampler = if let Some(grammar_str) = grammar {
            LlamaSampler::chain_simple([
                LlamaSampler::grammar(model, grammar_str, "root")
                    .map_err(|e| MemoryError::LLM(format!("Grammar sampler failed: {}", e)))?,
                LlamaSampler::temp(temperature),
                LlamaSampler::dist(42),
            ])
        } else {
            LlamaSampler::chain_simple([LlamaSampler::temp(temperature), LlamaSampler::dist(42)])
        };

        // Auto-regressive generation loop
        let mut output_tokens: Vec<LlamaToken> = Vec::new();
        let mut n_cur = tokens.len();

        for _ in 0..max_tokens {
            let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(new_token);

            // Stop on end-of-generation
            if params.model.is_eog_token(new_token) {
                break;
            }

            output_tokens.push(new_token);

            // Prepare next token for decoding
            batch.clear();
            batch
                .add(new_token, n_cur as i32, &[0], true)
                .map_err(|e| MemoryError::LLM(format!("Batch add failed: {}", e)))?;
            n_cur += 1;

            ctx.decode(&mut batch)
                .map_err(|e| MemoryError::LLM(format!("Decode failed: {}", e)))?;
        }

        // Detokenize
        let output: String = output_tokens
            .iter()
            .filter_map(|&t| {
                model
                    .token_to_piece_bytes(t, 32, true, None)
                    .ok()
                    .and_then(|bytes| String::from_utf8(bytes).ok())
            })
            .collect();

        Ok(output.trim().to_string())
    }

    /// Generate a completion and try to parse JSON from the output.
    fn extract_json_sync<T: serde::de::DeserializeOwned>(
        params: ExtractJsonParams,
    ) -> Result<(T, String)> {
        let ExtractJsonParams {
            model,
            backend,
            prompt,
            max_tokens,
            temperature,
            context_size,
            cpu_threads,
            strip_tags,
        } = params;

        let json_prompt = format!(
            "{}\n\nIMPORTANT: Respond ONLY with a valid JSON object. \
             No markdown code fences, no explanation, no extra text. Just raw JSON.",
            prompt
        );

        let response = Self::generate_sync(
            model,
            backend,
            &json_prompt,
            max_tokens,
            temperature,
            context_size,
            cpu_threads,
        )?;

        // Try to extract and parse JSON
        let json_str = extract_json_from_text(&response, strip_tags).ok_or_else(|| {
            MemoryError::Parse(format!(
                "No valid JSON found in model output: {}",
                &response[..response.len().min(200)]
            ))
        })?;

        let parsed: T = serde_json::from_str(&json_str).map_err(|e| {
            MemoryError::Parse(format!("JSON parse failed: {} in: {}", e, json_str))
        })?;

        Ok((parsed, response))
    }

    /// Run a structured extraction with a fallback function.
    ///
    /// Attempts JSON extraction from the model; on failure, calls the
    /// fallback closure with the raw text output.
    async fn run_extraction<T, F>(&self, prompt: &str, max_tokens: u32, fallback: F) -> Result<T>
    where
        T: serde::de::DeserializeOwned + Send + 'static,
        F: FnOnce(&str) -> T + Send + 'static,
    {
        let model = Arc::clone(&self.model);
        let backend = Arc::clone(&self.backend);
        let temperature = self.config.temperature;
        let context_size = self.config.context_size;
        let cpu_threads = self.config.cpu_threads;
        let prompt_owned = prompt.to_string();
        let timeout_secs = self.config.llm_timeout_secs;
        let strip_tags = self.config.strip_tags.clone();

        // Acquire a permit to limit concurrent LLM executions.
        // This prevents overloading the system with too many parallel llama.cpp instances.
        let _permit = self
            .concurrency_limiter
            .acquire()
            .await
            .map_err(|e| MemoryError::LLM(format!("Semaphore error: {}", e)))?;

        tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::task::spawn_blocking(move || {
                match Self::extract_json_sync::<T>(ExtractJsonParams {
                    model: &model,
                    backend: &backend,
                    prompt: &prompt_owned,
                    max_tokens,
                    temperature,
                    context_size,
                    cpu_threads,
                    strip_tags: &strip_tags,
                }) {
                    Ok((result, _)) => Ok(result),
                    Err(e) => {
                        debug!("JSON extraction failed ({}), using text fallback", e);
                        let response = Self::generate_sync(
                            &model,
                            &backend,
                            &prompt_owned,
                            max_tokens,
                            temperature,
                            context_size,
                            cpu_threads,
                        )?;
                        Ok(fallback(&response))
                    }
                }
            }),
        )
        .await
        .map_err(|_| MemoryError::LLM(format!("LLM extraction timed out after {}s", timeout_secs)))?
        .map_err(|e| MemoryError::LLM(format!("Task join error: {}", e)))?
    }

    /// Run a structured extraction with grammar-constrained sampling.
    ///
    /// Uses grammar-constrained sampling when enabled in config, falls back to
    /// regular extraction otherwise.
    async fn run_extraction_with_grammar<T, F>(
        &self,
        prompt: &str,
        max_tokens: u32,
        fallback: F,
        grammar: &str,
    ) -> Result<T>
    where
        T: serde::de::DeserializeOwned + Send + 'static,
        F: FnOnce(&str) -> T + Send + 'static,
    {
        // Check if grammar is enabled in config
        if !self.config.use_grammar {
            return self.run_extraction(prompt, max_tokens, fallback).await;
        }

        let model = Arc::clone(&self.model);
        let backend = Arc::clone(&self.backend);
        let temperature = self.config.temperature;
        let context_size = self.config.context_size;
        let cpu_threads = self.config.cpu_threads;
        let prompt_owned = prompt.to_string();
        let grammar_owned = grammar.to_string();
        let timeout_secs = self.config.llm_timeout_secs;

        let _permit = self
            .concurrency_limiter
            .acquire()
            .await
            .map_err(|e| MemoryError::LLM(format!("Semaphore error: {}", e)))?;

        tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::task::spawn_blocking(move || {
                // Try with grammar-constrained sampling first
                match Self::generate_sync_with_grammar(GenerateParams {
                    model: &model,
                    backend: &backend,
                    prompt: &prompt_owned,
                    max_tokens,
                    temperature,
                    max_context_size: context_size,
                    cpu_threads,
                    grammar: Some(&grammar_owned),
                }) {
                    Ok(response) => {
                        // Try to parse the grammar-constrained output
                        match serde_json::from_str::<T>(&response) {
                            Ok(result) => Ok(result),
                            Err(_) => Ok(fallback(&response)),
                        }
                    }
                    Err(e) => {
                        debug!(
                            "Grammar-constrained generation failed ({}), using fallback",
                            e
                        );
                        // Fall back to regular generation
                        let response = Self::generate_sync(
                            &model,
                            &backend,
                            &prompt_owned,
                            max_tokens,
                            temperature,
                            context_size,
                            cpu_threads,
                        )?;
                        Ok(fallback(&response))
                    }
                }
            }),
        )
        .await
        .map_err(|_| MemoryError::LLM(format!("LLM extraction timed out after {}s", timeout_secs)))?
        .map_err(|e| MemoryError::LLM(format!("Task join error: {}", e)))?
    }

    /// Calculate the timeout for a batch request.
    ///
    /// Formula: batch_timeout_secs * batch_timeout_multiplier * sqrt(batch_size)
    /// This scales the timeout based on the number of items in the batch.
    fn calculate_batch_timeout(&self, batch_size: usize) -> u64 {
        let sqrt_size = (batch_size as f64).sqrt();
        let timeout = self.config.batch_timeout_secs as f64
            * self.config.batch_timeout_multiplier
            * sqrt_size;
        timeout.ceil() as u64
    }

    /// Generate text completion with a custom timeout.
    ///
    /// This is an internal method used for batch operations that need longer timeouts.
    async fn complete_with_timeout(&self, prompt: &str, timeout_secs: u64) -> Result<String> {
        let start_time = std::time::Instant::now();
        let prompt_len = prompt.len();

        self.counters.llm_calls.fetch_add(1, Ordering::Relaxed);
        self.counters
            .prompt_tokens
            .fetch_add((prompt.len() / 4) as u64, Ordering::Relaxed);

        let model = Arc::clone(&self.model);
        let backend = Arc::clone(&self.backend);
        let max_tokens = self.config.max_tokens;
        let temperature = self.config.temperature;
        let context_size = self.config.context_size;
        let cpu_threads = self.config.cpu_threads;
        let prompt = prompt.to_string();

        // Acquire a permit to limit concurrent LLM executions.
        let _permit = self
            .concurrency_limiter
            .acquire()
            .await
            .map_err(|e| MemoryError::LLM(format!("Semaphore error: {}", e)))?;

        info!(
            "LLM request (local): model={}, prompt_chars={}, prompt_tokens_est={}, context_size={}, cpu_threads={}",
            self.config
                .model_file
                .rsplit('/')
                .next()
                .unwrap_or(&self.config.model_file),
            prompt_len,
            prompt_len / 4,
            context_size,
            cpu_threads
        );

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::task::spawn_blocking(move || {
                Self::generate_sync(
                    &model,
                    &backend,
                    &prompt,
                    max_tokens,
                    temperature,
                    context_size,
                    cpu_threads,
                )
            }),
        )
        .await
        .map_err(|_| MemoryError::LLM(format!("LLM completion timed out after {}s", timeout_secs)))?
        .map_err(|e| MemoryError::LLM(format!("Task join error: {}", e)))?;

        match &result {
            Ok(response) => {
                let elapsed = start_time.elapsed();
                info!(
                    "LLM response (local): response_chars={}, response_tokens_est={}, time_ms={}",
                    response.len(),
                    response.len() / 4,
                    elapsed.as_millis()
                );

                self.counters
                    .completion_tokens
                    .fetch_add((response.len() / 4) as u64, Ordering::Relaxed);
                if let Ok(mut ts) = self.counters.last_llm_success.lock() {
                    *ts = Some(chrono::Utc::now());
                }
            }
            Err(e) => {
                if let Ok(mut last) = self.counters.last_error.lock() {
                    *last = Some(e.to_string());
                }
            }
        }
        result
    }
}

// ── LLMClient trait implementation ─────────────────────────────────────────

#[async_trait]
impl LLMClient for LocalLLMClient {
    async fn complete(&self, prompt: &str) -> Result<String> {
        self.counters.llm_calls.fetch_add(1, Ordering::Relaxed);
        self.counters
            .prompt_tokens
            .fetch_add((prompt.len() / 4) as u64, Ordering::Relaxed);

        let model = Arc::clone(&self.model);
        let backend = Arc::clone(&self.backend);
        let max_tokens = self.config.max_tokens;
        let temperature = self.config.temperature;
        let context_size = self.config.context_size;
        let cpu_threads = self.config.cpu_threads;
        let prompt = prompt.to_string();

        let timeout_secs = self.config.llm_timeout_secs;

        // Acquire a permit to limit concurrent LLM executions.
        let _permit = self
            .concurrency_limiter
            .acquire()
            .await
            .map_err(|e| MemoryError::LLM(format!("Semaphore error: {}", e)))?;

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::task::spawn_blocking(move || {
                Self::generate_sync(
                    &model,
                    &backend,
                    &prompt,
                    max_tokens,
                    temperature,
                    context_size,
                    cpu_threads,
                )
            }),
        )
        .await
        .map_err(|_| MemoryError::LLM(format!("LLM completion timed out after {}s", timeout_secs)))?
        .map_err(|e| MemoryError::LLM(format!("Task join error: {}", e)))?;

        match &result {
            Ok(response) => {
                self.counters
                    .completion_tokens
                    .fetch_add((response.len() / 4) as u64, Ordering::Relaxed);
                if let Ok(mut ts) = self.counters.last_llm_success.lock() {
                    *ts = Some(chrono::Utc::now());
                }
            }
            Err(e) => {
                if let Ok(mut last) = self.counters.last_error.lock() {
                    *last = Some(e.to_string());
                }
            }
        }
        result
    }

    async fn complete_with_grammar(&self, prompt: &str, grammar: &str) -> Result<String> {
        self.counters.llm_calls.fetch_add(1, Ordering::Relaxed);
        self.counters
            .prompt_tokens
            .fetch_add((prompt.len() / 4) as u64, Ordering::Relaxed);

        let model = Arc::clone(&self.model);
        let backend = Arc::clone(&self.backend);
        let max_tokens = self.config.max_tokens;
        let temperature = self.config.temperature;
        let context_size = self.config.context_size;
        let cpu_threads = self.config.cpu_threads;
        let prompt = prompt.to_string();
        let grammar = grammar.to_string();
        let timeout_secs = self.config.llm_timeout_secs;

        let _permit = self
            .concurrency_limiter
            .acquire()
            .await
            .map_err(|e| MemoryError::LLM(format!("Semaphore error: {}", e)))?;

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::task::spawn_blocking(move || {
                Self::generate_sync_with_grammar(GenerateParams {
                    model: &model,
                    backend: &backend,
                    prompt: &prompt,
                    max_tokens,
                    temperature,
                    max_context_size: context_size,
                    cpu_threads,
                    grammar: Some(&grammar),
                })
            }),
        )
        .await
        .map_err(|_| MemoryError::LLM(format!("LLM completion timed out after {}s", timeout_secs)))?
        .map_err(|e| MemoryError::LLM(format!("Task join error: {}", e)))?;

        match &result {
            Ok(response) => {
                self.counters
                    .completion_tokens
                    .fetch_add((response.len() / 4) as u64, Ordering::Relaxed);
                if let Ok(mut ts) = self.counters.last_llm_success.lock() {
                    *ts = Some(chrono::Utc::now());
                }
            }
            Err(e) => {
                if let Ok(mut last) = self.counters.last_error.lock() {
                    *last = Some(e.to_string());
                }
            }
        }
        result
    }

    async fn embed(&self, text: &str, purpose: EmbedPurpose) -> Result<Vec<f32>> {
        self.counters
            .embedding_calls
            .fetch_add(1, Ordering::Relaxed);

        let embedding = Arc::clone(&self.embedding);
        let text = match purpose {
            EmbedPurpose::Query if !self.query_prefix.is_empty() => {
                format!("{}{}", self.query_prefix, text)
            }
            EmbedPurpose::Document if !self.document_prefix.is_empty() => {
                format!("{}{}", self.document_prefix, text)
            }
            _ => text.to_string(),
        };
        let timeout_secs = self.config.llm_timeout_secs;

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::task::spawn_blocking(move || {
                let emb = embedding
                    .lock()
                    .map_err(|e| MemoryError::Embedding(format!("Lock poisoned: {}", e)))?;
                let results = emb
                    .embed(vec![text], None)
                    .map_err(|e| MemoryError::Embedding(format!("Embedding failed: {}", e)))?;
                results
                    .into_iter()
                    .next()
                    .ok_or_else(|| MemoryError::Embedding("No embedding generated".to_string()))
            }),
        )
        .await
        .map_err(|_| {
            MemoryError::Embedding(format!("Embedding timed out after {}s", timeout_secs))
        })?
        .map_err(|e| MemoryError::Embedding(format!("Task join error: {}", e)))?;

        match &result {
            Ok(_) => {
                if let Ok(mut ts) = self.counters.last_embedding_success.lock() {
                    *ts = Some(chrono::Utc::now());
                }
            }
            Err(e) => {
                if let Ok(mut last) = self.counters.last_error.lock() {
                    *last = Some(e.to_string());
                }
            }
        }
        result
    }

    async fn embed_batch(&self, texts: &[String], purpose: EmbedPurpose) -> Result<Vec<Vec<f32>>> {
        let embedding = Arc::clone(&self.embedding);
        let texts: Vec<String> = texts
            .iter()
            .map(|t| match purpose {
                EmbedPurpose::Query if !self.query_prefix.is_empty() => {
                    format!("{}{}", self.query_prefix, t)
                }
                EmbedPurpose::Document if !self.document_prefix.is_empty() => {
                    format!("{}{}", self.document_prefix, t)
                }
                _ => t.clone(),
            })
            .collect();
        let timeout_secs = self.config.llm_timeout_secs;

        tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::task::spawn_blocking(move || {
                let emb = embedding
                    .lock()
                    .map_err(|e| MemoryError::Embedding(format!("Lock poisoned: {}", e)))?;
                emb.embed(texts, None)
                    .map_err(|e| MemoryError::Embedding(format!("Batch embedding failed: {}", e)))
            }),
        )
        .await
        .map_err(|_| {
            MemoryError::Embedding(format!("Batch embedding timed out after {}s", timeout_secs))
        })?
        .map_err(|e| MemoryError::Embedding(format!("Task join error: {}", e)))?
    }

    async fn extract_keywords(&self, content: &str) -> Result<Vec<String>> {
        let prompt = format!(
            "Extract 3-10 search-relevant keywords or short noun phrases from the following text. \
             Return ONLY comma-separated keywords. No sentences, no explanations, no markdown. \
             Capitalize proper nouns. Output nothing besides the comma-separated list.\n\n\
             TEXT:\n{}",
            content
        );
        let wrapped = format_chatml_prompt(&prompt);
        let response = self.complete(&wrapped).await?;
        // Strip XML tags (e.g., <think>...</think>) before parsing keywords
        let cleaned = strip_llm_tags(&response, &self.config.strip_tags);

        let keywords: Vec<String> = cleaned
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.len() < 100)
            .collect();

        info!("Extracted {} keywords: {:?}", keywords.len(), keywords);
        Ok(keywords)
    }

    async fn summarize(&self, content: &str, _max_length: Option<usize>) -> Result<String> {
        let response = self.complete(content).await?;
        // Strip XML tags before returning summary
        let cleaned = strip_llm_tags(&response, &self.config.strip_tags);

        // Log for debugging
        if response.contains("<think>") || response.contains("</think>") {
            info!(
                "Summary extraction: stripped think tags from LLM response. Original length: {}, Cleaned length: {}",
                response.len(),
                cleaned.len()
            );
        }

        Ok(cleaned.trim().to_string())
    }

    async fn health_check(&self) -> Result<bool> {
        match self.embed("health check", crate::llm::EmbedPurpose::Query).await {
            Ok(_) => {
                info!("Local LLM health check passed");
                Ok(true)
            }
            Err(e) => {
                error!("Local LLM health check failed: {}", e);
                Ok(false)
            }
        }
    }

    async fn extract_structured_facts(&self, prompt: &str) -> Result<StructuredFactExtraction> {
        self.run_extraction(prompt, self.config.max_tokens, |response| {
            let facts: Vec<String> = response
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.trim_start_matches("- ").trim().to_string())
                .collect();
            StructuredFactExtraction {
                facts: if facts.is_empty() {
                    vec![response.to_string()]
                } else {
                    facts
                },
            }
        })
        .await
    }

    async fn extract_detailed_facts(&self, prompt: &str) -> Result<DetailedFactExtraction> {
        self.run_extraction(prompt, self.config.max_tokens, |response| {
            let facts: Vec<StructuredFact> = response
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| StructuredFact {
                    content: l.trim_start_matches("- ").trim().to_string(),
                    importance: 0.5,
                    category: "general".to_string(),
                    entities: vec![],
                    source_role: "unknown".to_string(),
                })
                .collect();
            DetailedFactExtraction {
                facts: if facts.is_empty() {
                    vec![StructuredFact {
                        content: response.to_string(),
                        importance: 0.5,
                        category: "general".to_string(),
                        entities: vec![],
                        source_role: "unknown".to_string(),
                    }]
                } else {
                    facts
                },
            }
        })
        .await
    }

    async fn extract_keywords_structured(&self, prompt: &str) -> Result<KeywordExtraction> {
        let strip_tags = self.config.strip_tags.clone();
        self.run_extraction(prompt, 500, move |response| {
            // Strip XML tags (e.g., <think>...</think>) before parsing keywords
            let cleaned = strip_llm_tags(response, &strip_tags);
            let keywords: Vec<String> = cleaned
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            KeywordExtraction { keywords }
        })
        .await
    }

    async fn classify_memory(&self, prompt: &str) -> Result<MemoryClassification> {
        let strip_tags = self.config.strip_tags.clone();
        self.run_extraction(prompt, 500, move |response| {
            // Strip XML tags before parsing classification
            let cleaned = strip_llm_tags(response, &strip_tags);
            let lower = cleaned.to_lowercase();
            let memory_type = if lower.contains("conversational") {
                "Conversational"
            } else if lower.contains("procedural") {
                "Procedural"
            } else if lower.contains("factual") {
                "Factual"
            } else if lower.contains("semantic") {
                "Semantic"
            } else if lower.contains("episodic") {
                "Episodic"
            } else if lower.contains("personal") {
                "Personal"
            } else {
                "Conversational"
            };
            MemoryClassification {
                memory_type: memory_type.to_string(),
                confidence: 0.6,
                reasoning: format!("Local model classification: {}", response),
            }
        })
        .await
    }

    async fn score_importance(&self, prompt: &str) -> Result<ImportanceScore> {
        self.run_extraction(prompt, 500, |response| {
            // Try to find a number in the response
            let score = response
                .split_whitespace()
                .find_map(|w| {
                    w.trim_matches(|c: char| !c.is_ascii_digit() && c != '.')
                        .parse::<f32>()
                        .ok()
                })
                .map(|s| if s > 1.0 { s / 10.0 } else { s }) // normalize 0-10 to 0-1
                .unwrap_or(0.5);
            ImportanceScore {
                score: score.clamp(0.0, 1.0),
                reasoning: format!("Local model scoring: {}", response),
            }
        })
        .await
    }

    async fn check_duplicates(&self, prompt: &str) -> Result<DeduplicationResult> {
        self.run_extraction(prompt, 500, |response| {
            let lower = response.to_lowercase();
            // Conservative check for positive affirmation
            let is_duplicate = (lower.contains("yes") && !lower.contains("no"))
                || lower.contains("is a duplicate")
                || lower.contains("are duplicates");

            DeduplicationResult {
                is_duplicate,
                similarity_score: if is_duplicate { 0.9 } else { 0.1 },
                original_memory_id: None,
            }
        })
        .await
    }

    async fn generate_summary(&self, prompt: &str) -> Result<SummaryResult> {
        self.run_extraction(prompt, 1000, |response| SummaryResult {
            summary: response.to_string(),
            key_points: response
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.trim().to_string())
                .collect(),
        })
        .await
    }

    async fn detect_language(&self, prompt: &str) -> Result<LanguageDetection> {
        self.run_extraction(prompt, 200, |response| {
            let lower = response.to_lowercase();
            let language = if lower.contains("english") {
                "en"
            } else if lower.contains("spanish") {
                "es"
            } else if lower.contains("french") {
                "fr"
            } else if lower.contains("german") {
                "de"
            } else if lower.contains("chinese") {
                "zh"
            } else if lower.contains("japanese") {
                "ja"
            } else {
                "en"
            };
            LanguageDetection {
                language: language.to_string(),
                confidence: 0.7,
            }
        })
        .await
    }

    async fn extract_entities(&self, prompt: &str) -> Result<EntityExtraction> {
        self.run_extraction(prompt, 1000, |_response| EntityExtraction {
            entities: vec![],
        })
        .await
    }

    async fn analyze_conversation(&self, prompt: &str) -> Result<ConversationAnalysis> {
        self.run_extraction(prompt, 1500, |response| {
            let lines: Vec<String> = response
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.trim().to_string())
                .collect();
            ConversationAnalysis {
                topics: lines.clone(),
                sentiment: "neutral".to_string(),
                user_intent: "information".to_string(),
                key_information: lines,
            }
        })
        .await
    }

    async fn extract_metadata_enrichment(&self, prompt: &str) -> Result<MetadataEnrichment> {
        let strip_tags = self.config.strip_tags.clone();
        self.run_extraction_with_grammar(
            prompt,
            1000,
            move |response| {
                // Fallback: try to parse JSON from response or use split-based extraction
                if let Some(json_str) = extract_json_from_text(response, &strip_tags)
                    && let Ok(enrichment) = serde_json::from_str::<MetadataEnrichment>(&json_str)
                {
                    return enrichment;
                }
                // Last resort fallback - strip tags before parsing
                let cleaned = strip_llm_tags(response, &strip_tags);
                MetadataEnrichment {
                    summary: cleaned.clone(),
                    keywords: cleaned
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect(),
                }
            },
            metadata_enrichment_grammar(),
        )
        .await
    }

    async fn extract_metadata_enrichment_batch(
        &self,
        texts: &[String],
    ) -> Result<Vec<Result<MetadataEnrichment>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        debug!(
            "Local batch extracting metadata enrichment for {} texts in a single call",
            texts.len()
        );

        // Generate simple deterministic IDs that the LLM can reliably echo back
        let texts_with_ids: Vec<MetadataEnrichmentWithId> = texts
            .iter()
            .enumerate()
            .map(|(idx, text)| MetadataEnrichmentWithId {
                id: format!("chunk_{}", idx),
                text: text.clone(),
            })
            .collect();

        let texts_json =
            serde_json::to_string(&texts_with_ids).unwrap_or_else(|_| "[]".to_string());
        let prompt = crate::memory::prompts::METADATA_ENRICHMENT_BATCH_PROMPT
            .replace("{{texts}}", &texts_json);
        let wrapped_prompt = format_chatml_prompt(&prompt);

        // Calculate batch-aware timeout
        let batch_timeout = self.calculate_batch_timeout(texts.len());
        debug!(
            "Using batch timeout: {}s (batch_size={}, timeout={}s, multiplier={})",
            batch_timeout,
            texts.len(),
            self.config.batch_timeout_secs,
            self.config.batch_timeout_multiplier
        );

        let response = match self
            .complete_with_timeout(&wrapped_prompt, batch_timeout)
            .await
        {
            Ok(res) => res,
            Err(e) => {
                let mut errors = Vec::new();
                for _ in 0..texts.len() {
                    errors.push(Err(crate::error::MemoryError::LLM(format!(
                        "Batch call failed: {}",
                        e
                    ))));
                }
                return Ok(errors);
            }
        };

        let parsed: Vec<MetadataEnrichmentResponseWithId> =
            match super::client::extract_json_from_text(&response) {
                Some(json_str) => match serde_json::from_str(&json_str) {
                    Ok(arr) => arr,
                    Err(e) => {
                        let mut errors = Vec::new();
                        for _ in 0..texts.len() {
                            errors.push(Err(crate::error::MemoryError::LLM(format!(
                                "Failed to parse JSON: {}",
                                e
                            ))));
                        }
                        return Ok(errors);
                    }
                },
                None => {
                    let mut errors = Vec::new();
                    for _ in 0..texts.len() {
                        errors.push(Err(crate::error::MemoryError::LLM(
                            "No JSON array found".to_string(),
                        )));
                    }
                    return Ok(errors);
                }
            };

        let mut id_to_response: std::collections::HashMap<String, MetadataEnrichment> =
            std::collections::HashMap::new();
        for resp in parsed {
            id_to_response.insert(
                resp.id.clone(),
                MetadataEnrichment {
                    summary: resp.summary,
                    keywords: resp.keywords,
                },
            );
        }

        let mut results = Vec::new();
        for text_with_id in texts_with_ids.iter() {
            match id_to_response.get(&text_with_id.id) {
                Some(enrichment) => {
                    results.push(Ok(enrichment.clone()));
                }
                None => {
                    results.push(Err(crate::error::MemoryError::LLM(format!(
                        "No response found for text with ID '{}'",
                        &text_with_id.id
                    ))));
                }
            }
        }

        Ok(results)
    }

    async fn complete_batch(&self, prompts: &[String]) -> Result<Vec<Result<String>>> {
        if prompts.is_empty() {
            return Ok(vec![]);
        }

        debug!(
            "Local batch completing {} prompts in a single call",
            prompts.len()
        );

        let prompts_json = serde_json::to_string(prompts).unwrap_or_else(|_| "[]".to_string());
        let master_prompt =
            crate::memory::prompts::COMPLETE_BATCH_PROMPT.replace("{{prompts}}", &prompts_json);
        let wrapped_prompt = format_chatml_prompt(&master_prompt);

        // Calculate batch-aware timeout
        let batch_timeout = self.calculate_batch_timeout(prompts.len());
        debug!(
            "Using batch timeout: {}s (batch_size={}, timeout={}s, multiplier={})",
            batch_timeout,
            prompts.len(),
            self.config.batch_timeout_secs,
            self.config.batch_timeout_multiplier
        );

        let response = match self
            .complete_with_timeout(&wrapped_prompt, batch_timeout)
            .await
        {
            Ok(res) => res,
            Err(e) => {
                let mut errors = Vec::new();
                for _ in 0..prompts.len() {
                    errors.push(Err(crate::error::MemoryError::LLM(format!(
                        "Batch call failed: {}",
                        e
                    ))));
                }
                return Ok(errors);
            }
        };

        let parsed: Vec<String> = match super::client::extract_json_from_text(&response) {
            Some(json_str) => match serde_json::from_str(&json_str) {
                Ok(arr) => arr,
                Err(e) => {
                    let mut errors = Vec::new();
                    for _ in 0..prompts.len() {
                        errors.push(Err(crate::error::MemoryError::LLM(format!(
                            "Failed to parse batch JSON: {}",
                            e
                        ))));
                    }
                    return Ok(errors);
                }
            },
            None => {
                let mut errors = Vec::new();
                for _ in 0..prompts.len() {
                    errors.push(Err(crate::error::MemoryError::LLM(
                        "No JSON array found in batch response".to_string(),
                    )));
                }
                return Ok(errors);
            }
        };

        if parsed.len() != prompts.len() {
            let mut errors = Vec::new();
            for _ in 0..prompts.len() {
                errors.push(Err(crate::error::MemoryError::LLM(format!(
                    "Batch length mismatch: expected {}, got {}",
                    prompts.len(),
                    parsed.len()
                ))));
            }
            return Ok(errors);
        }

        Ok(parsed.into_iter().map(Ok).collect())
    }

    fn get_status(&self) -> ClientStatus {
        let last_llm = self
            .counters
            .last_llm_success
            .lock()
            .ok()
            .and_then(|ts| ts.map(|t| t.to_rfc3339()));
        let last_emb = self
            .counters
            .last_embedding_success
            .lock()
            .ok()
            .and_then(|ts| ts.map(|t| t.to_rfc3339()));
        let last_err = self.counters.last_error.lock().ok().and_then(|e| e.clone());

        // Model file size
        let model_size = std::fs::metadata(&self.model_path)
            .map(|m| m.len())
            .unwrap_or(0);

        let mut details = HashMap::new();
        details.insert(
            "gpu_layers".into(),
            serde_json::json!(self.config.gpu_layers),
        );
        details.insert(
            "context_size".into(),
            serde_json::json!(self.config.context_size),
        );
        details.insert(
            "models_dir".into(),
            serde_json::json!(self.config.models_dir),
        );
        details.insert(
            "llm_model_path".into(),
            serde_json::json!(self.model_path.display().to_string()),
        );
        details.insert("llm_model_size_bytes".into(), serde_json::json!(model_size));
        details.insert(
            "llm_model_size_mb".into(),
            serde_json::json!(format!("{:.1}", model_size as f64 / 1_048_576.0)),
        );
        details.insert("embedding_model_loaded".into(), serde_json::json!(true));

        ClientStatus {
            backend: "local".to_string(),
            state: "ready".to_string(),
            llm_model: self.config.model_file.clone(),
            embedding_model: self.embedding_model_name.clone(),
            llm_available: true,
            embedding_available: true,
            last_llm_success: last_llm,
            last_embedding_success: last_emb,
            last_error: last_err,
            total_llm_calls: self.counters.llm_calls.load(Ordering::Relaxed),
            total_embedding_calls: self.counters.embedding_calls.load(Ordering::Relaxed),
            total_prompt_tokens: self.counters.prompt_tokens.load(Ordering::Relaxed),
            total_completion_tokens: self.counters.completion_tokens.load(Ordering::Relaxed),
            details,
        }
    }

    fn batch_config(&self) -> (usize, u32) {
        // Local models usually handle one at a time, but we can parallelize prompts
        // Use a reasonable default for local if not specified
        (self.config.batch_size, self.config.batch_max_tokens)
    }

    async fn enhance_memory_unified(&self, prompt: &str) -> Result<MemoryEnhancement> {
        self.run_extraction(prompt, 1000, |response| {
            // Try to extract JSON from the response first
            if let Some(json_str) = extract_json_from_text(response, &[])
                && let Ok(enrichment) = serde_json::from_str::<MemoryEnhancement>(&json_str)
            {
                return enrichment;
            }
            // Fallback: return defaults with the raw text as summary
            MemoryEnhancement {
                memory_type: "Semantic".to_string(),
                summary: response.trim().to_string(),
                keywords: vec![],
                entities: vec![],
                topics: vec![],
            }
        })
        .await
    }

    async fn describe_image(&self, image_bytes: &[u8], _mime_type: &str) -> Result<String> {
        let _ = check_vision_available(&self.config).map_err(MemoryError::LLM)?;
        let mmproj_path = resolve_mmproj_path(&self.config).await?;

        let model = Arc::clone(&self.model);
        let backend = Arc::clone(&self.backend);
        let image_bytes = image_bytes.to_vec();
        let max_tokens = self.config.max_tokens;
        let temperature = self.config.temperature;
        let context_size = self.config.context_size;
        let cpu_threads = self.config.cpu_threads;
        let timeout_secs = self.config.llm_timeout_secs;

        let _permit = self
            .concurrency_limiter
            .acquire()
            .await
            .map_err(|e| MemoryError::LLM(format!("Semaphore error: {}", e)))?;

        let mmproj_path_str = mmproj_path.to_string_lossy().to_string();
        let prompt_template = self.config.vision_prompt_template.clone();

        tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::task::spawn_blocking(move || {
                generate_vision_sync(&VisionParams {
                    model: &model,
                    backend: &backend,
                    mmproj_path: &mmproj_path_str,
                    image_bytes: &image_bytes,
                    prompt_template: &prompt_template,
                    max_tokens,
                    temperature,
                    context_size,
                    cpu_threads,
                })
            }),
        )
        .await
        .map_err(|_| {
            MemoryError::LLM(format!(
                "Vision completion timed out after {}s",
                timeout_secs
            ))
        })?
        .map_err(|e| MemoryError::LLM(format!("Task join error: {}", e)))?
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Generate a text description of an image using llama.cpp's multimodal (MTMD) API.
struct VisionParams<'a> {
    model: &'a LlamaModel,
    backend: &'a LlamaBackend,
    mmproj_path: &'a str,
    image_bytes: &'a [u8],
    prompt_template: &'a str,
    max_tokens: u32,
    temperature: f32,
    context_size: u32,
    cpu_threads: i32,
}

fn generate_vision_sync(params: &VisionParams) -> Result<String> {
    let marker = mtmd_default_marker();
    let mtmd_params = MtmdContextParams {
        use_gpu: false,
        print_timings: false,
        n_threads: params.cpu_threads,
        media_marker: std::ffi::CString::new(marker)
            .map_err(|e| MemoryError::LLM(format!("mtmd init: {}", e)))?,
    };

    let mtmd_ctx = MtmdContext::init_from_file(params.mmproj_path, params.model, &mtmd_params)
        .map_err(|e| MemoryError::LLM(format!("Failed to init multimodal context: {}", e)))?;

    if !mtmd_ctx.support_vision() {
        return Err(MemoryError::LLM("Model does not support vision".into()));
    }

    let bitmap = MtmdBitmap::from_buffer(&mtmd_ctx, params.image_bytes)
        .map_err(|e| MemoryError::LLM(format!("Failed to load image bitmap: {}", e)))?;

    let text = MtmdInputText {
        text: params.prompt_template.to_string(),
        add_special: true,
        parse_special: true,
    };

    let chunks = mtmd_ctx
        .tokenize(text, &[&bitmap])
        .map_err(|e| MemoryError::LLM(format!("Failed to tokenize multimodal input: {}", e)))?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(
            NonZeroU32::new(params.context_size).unwrap_or(NonZeroU32::new(2048).unwrap()),
        ))
        .with_n_batch(512);

    let mut ctx = params
        .model
        .new_context(params.backend, ctx_params)
        .map_err(|e| MemoryError::LLM(format!("Failed to create context: {}", e)))?;

    let n_batch: i32 = 512;

    let n_past = chunks
        .eval_chunks(&mtmd_ctx, &ctx, 0, 0, n_batch, true)
        .map_err(|e| MemoryError::LLM(format!("Failed to evaluate multimodal chunks: {}", e)))?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(params.temperature),
        LlamaSampler::dist(42),
    ]);

    let mut response = String::new();
    let mut output_tokens: Vec<LlamaToken> = Vec::new();
    let max = params.max_tokens.min(1024) as usize;
    let mut n_cur = n_past + chunks.total_tokens() as i32;

    for _ in 0..max {
        let new_token = sampler.sample(&ctx, n_cur - 1);
        sampler.accept(new_token);

        if params.model.is_eog_token(new_token) {
            break;
        }

        output_tokens.push(new_token);

        let mut batch = LlamaBatch::new(1, 1);
        batch
            .add(new_token, n_cur, &[0], true)
            .map_err(|e| MemoryError::LLM(format!("Batch add error: {}", e)))?;

        ctx.decode(&mut batch)
            .map_err(|e| MemoryError::LLM(format!("Decode error: {}", e)))?;

        n_cur += 1;
    }

    #[allow(deprecated)]
    if !output_tokens.is_empty() {
        response = params
            .model
            .tokens_to_str(&output_tokens, Special::Plaintext)
            .map_err(|e| MemoryError::LLM(format!("Token decode error: {}", e)))?;
    }

    Ok(response.trim().to_string())
}

/// Resolve the mmproj path for vision/image description.
///
/// Resolution order:
/// 1. If `mmproj_file` is explicitly set, use that path.
/// 2. Otherwise, look for the default `mmproj-F16.gguf` in models_dir.
///
/// In both cases, if the file doesn't exist and `auto_download = true`,
/// attempt to download it via the known models registry.
pub async fn resolve_mmproj_path(config: &LlmConfig) -> Result<PathBuf> {
    let models_dir = PathBuf::from(&config.models_dir);

    let path = match &config.mmproj_file {
        Some(p) => PathBuf::from(p),
        None => {
            let filename = super::model_downloader::DEFAULT_MMPROJ_FILENAME;
            models_dir.join(filename)
        }
    };

    if !path.exists() && config.auto_download {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(super::model_downloader::DEFAULT_MMPROJ_FILENAME);
        let result = super::model_downloader::ensure_model(
            &models_dir,
            filename,
            config.proxy_url.as_deref(),
            config.cache_model,
            config.cache_dir.as_deref(),
        )
        .await?;
        if result.freshly_downloaded {
            info!(
                "Vision projection model downloaded: {} ({})",
                filename,
                super::model_downloader::format_size(result.size_bytes)
            );
        }
        return Ok(result.path);
    }

    if path.exists() {
        return Ok(path);
    }

    let help = if config.auto_download {
        "Set mmproj_file in [llm] config to the correct path."
    } else {
        "Enable auto_download = true or download manually."
    };

    Err(MemoryError::LLM(format!(
        "Vision projection model not found: {}\n{}",
        path.display(),
        help
    )))
}

/// Check whether vision description can be attempted.
/// Returns Ok(()) if vision is configured and available, or an error describing why not.
pub fn check_vision_available(config: &LlmConfig) -> std::result::Result<(), String> {
    if !config.vision_enabled {
        return Err("Vision description is disabled. Set vision_enabled = true to enable.".into());
    }
    Ok(())
}

/// Generate GBNF grammar for MetadataEnrichment JSON schema
fn metadata_enrichment_grammar() -> &'static str {
    r#"
root ::= object
object ::= "{" ws "\"summary\":" ws string ws "," ws "\"keywords\":" ws array ws "}"
array ::= "[" ws (string (ws "," ws string)*)? ws "]"
string ::= "\"" char* "\""
char ::= [^"\\\x00-\x1F] | "\\" ["\\/bfnrt] | "\\u" hex hex hex hex
hex ::= [0-9a-fA-F]
ws ::= [ \t\n]*
"#
}

/// Format a prompt using ChatML template.
///
/// Most instruction-tuned GGUF models (Qwen, SmolLM, etc.) use this format.
fn format_chatml_prompt(prompt: &str) -> String {
    format!(
        "<|im_start|>system\n\
         You are a precise AI assistant. Follow instructions exactly. \
         When asked for JSON, respond with only valid JSON, no markdown.\n\
         <|im_end|>\n\
         <|im_start|>user\n\
         {}\n\
         <|im_end|>\n\
         <|im_start|>assistant\n",
        prompt
    )
}

/// Strip XML-style tags (e.g., <think>...</think>, <reason>...</reason>) from LLM output
/// Supports multiple tag types and handles missing closing tags gracefully
fn strip_xml_tags(text: &str, tags: &[String]) -> String {
    let mut result = text.to_string();

    for tag in tags {
        // Strip <tag>...</tag> blocks (with or without closing tag)
        loop {
            let open_tag = format!("<{}", tag);
            let close_tag = format!("</{}>", tag);

            if let Some(start) = result.find(&open_tag) {
                // Find the end of the opening tag (>)
                if let Some(tag_end) = result[start..].find('>') {
                    let content_start = start + tag_end + 1;
                    // Try to find closing tag first
                    if let Some(close_pos) = result[content_start..].find(&close_tag) {
                        let before = &result[..start];
                        let after = &result[content_start + close_pos + close_tag.len()..];
                        result = format!("{}{}", before, after);
                        continue;
                    } else {
                        // No closing tag found - strip from opening tag to end of text
                        // This handles malformed LLM output gracefully
                        result = result[..start].to_string();
                        continue;
                    }
                }
            }
            break;
        }
    }

    result.trim().to_string()
}

/// Strip configured XML tags from LLM output
fn strip_llm_tags(text: &str, tags: &[String]) -> String {
    strip_xml_tags(text, tags)
}

/// Extract a JSON object or array from text that may contain surrounding prose.
pub(crate) fn extract_json_from_text(text: &str, strip_tags: &[String]) -> Option<String> {
    // First strip configured XML tags
    let text = strip_llm_tags(text, strip_tags);
    let text = text.trim();

    // Strip markdown code fences if present
    let text = if let Some(stripped) = text.strip_prefix("```json") {
        let end = stripped.rfind("```").unwrap_or(stripped.len());
        if end > 0 { &stripped[..end] } else { stripped }
    } else if let Some(stripped) = text.strip_prefix("```") {
        let end = stripped.rfind("```").unwrap_or(stripped.len());
        if end > 0 { &stripped[..end] } else { stripped }
    } else {
        text
    };
    let text = text.trim();

    let start = text.find('{').or_else(|| text.find('['))?;
    let open_byte = text.as_bytes()[start];
    let close_byte = if open_byte == b'{' { b'}' } else { b']' };

    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, byte) in text[start..].bytes().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }
        match byte {
            b'\\' if in_string => escape_next = true,
            b'"' => in_string = !in_string,
            b if b == open_byte && !in_string => depth += 1,
            b if b == close_byte && !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..start + i + 1].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_simple_object() {
        let text = r#"Here is the result: {"facts": ["a", "b"]} done"#;
        let json = extract_json_from_text(text, &[]).unwrap();
        assert_eq!(json, r#"{"facts": ["a", "b"]}"#);
    }

    #[test]
    fn test_extract_json_with_code_fence() {
        let text = "```json\n{\"key\": \"value\"}\n```";
        let json = extract_json_from_text(text, &[]).unwrap();
        assert_eq!(json, r#"{"key": "value"}"#);
    }

    #[test]
    fn test_extract_json_nested() {
        let text = r#"{"outer": {"inner": [1, 2, 3]}}"#;
        let json = extract_json_from_text(text, &[]).unwrap();
        assert_eq!(json, text);
    }

    #[test]
    fn test_extract_json_with_escaped_quotes() {
        let text = r#"{"text": "he said \"hello\""}"#;
        let json = extract_json_from_text(text, &[]).unwrap();
        assert_eq!(json, text);
    }

    #[test]
    fn test_extract_json_array() {
        let text = r#"Result: ["one", "two", "three"]"#;
        let json = extract_json_from_text(text, &[]).unwrap();
        assert_eq!(json, r#"["one", "two", "three"]"#);
    }

    #[test]
    fn test_extract_json_none_for_no_json() {
        assert!(extract_json_from_text("no json here", &[]).is_none());
    }

    #[test]
    fn test_strip_think_tags() {
        let text = "<think>\nThinking...\n</think>\n{\"result\": \"success\"}";
        let json = extract_json_from_text(text, &["think".to_string()]).unwrap();
        assert_eq!(json, r#"{"result": "success"}"#);
    }

    #[test]
    fn test_strip_think_tags_missing_closing() {
        // Test missing closing tag - should strip from opening tag to end
        let text = "Some text <think>\nThinking without closing tag";
        let result = strip_llm_tags(text, &["think".to_string()]);
        assert_eq!(result, "Some text");
    }

    #[test]
    fn test_strip_think_tags_multiple() {
        // Test multiple think tags
        let text = "<think>first</think> middle <think>second</think> end";
        let result = strip_llm_tags(text, &["think".to_string()]);
        assert_eq!(result, "middle  end");
    }

    #[test]
    fn test_strip_think_tags_nested_content() {
        // Test think tags with JSON-like content inside
        let text = "<think>{\"temp\": \"thinking\"}</think>{\"result\": \"success\"}";
        let json = extract_json_from_text(text, &["think".to_string()]).unwrap();
        assert_eq!(json, r#"{"result": "success"}"#);
    }

    #[test]
    fn test_strip_multiple_tag_types() {
        // Test stripping multiple tag types
        let text = "<think>thinking</think><reason>reasoning</reason>final";
        let result = strip_llm_tags(text, &["think".to_string(), "reason".to_string()]);
        assert_eq!(result, "final");
    }

    #[test]
    fn test_format_chatml() {
        let formatted = format_chatml_prompt("Hello");
        assert!(formatted.contains("<|im_start|>user"));
        assert!(formatted.contains("Hello"));
        assert!(formatted.contains("<|im_start|>assistant"));
    }

    // ── Vision mmproj resolution tests ────────────────────────────────

    fn make_test_llm_config(
        mmproj_file: Option<&str>,
        vision_enabled: bool,
        auto_download: bool,
    ) -> crate::config::LlmConfig {
        crate::config::LlmConfig {
            mmproj_file: mmproj_file.map(std::string::ToString::to_string),
            vision_enabled,
            auto_download,
            models_dir: "/nonexistent/path/for/testing/models".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_check_vision_available_enabled() {
        let config = make_test_llm_config(None, true, false);
        assert!(check_vision_available(&config).is_ok());
    }

    #[test]
    fn test_check_vision_available_disabled() {
        let config = make_test_llm_config(None, false, false);
        let err = check_vision_available(&config).unwrap_err();
        assert!(err.contains("vision_enabled"));
    }

    #[test]
    fn test_check_vision_available_with_mmproj() {
        let config = make_test_llm_config(Some("/tmp/mmproj.gguf"), true, false);
        assert!(check_vision_available(&config).is_ok());
    }

    #[tokio::test]
    async fn test_resolve_mmproj_missing_no_auto_download() {
        let config =
            make_test_llm_config(Some("/nonexistent/absolutely/missing.gguf"), true, false);
        let err = resolve_mmproj_path(&config).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not found") || msg.contains("Enable auto_download"));
    }

    #[tokio::test]
    async fn test_resolve_mmproj_default_missing_no_auto_download() {
        let config = make_test_llm_config(None, true, false);
        let err = resolve_mmproj_path(&config).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not found") || msg.contains("Enable auto_download"));
    }

    #[tokio::test]
    async fn test_resolve_mmproj_uses_default_filename() {
        let config = make_test_llm_config(None, true, false);
        let err = resolve_mmproj_path(&config).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("mmproj-F16.gguf"));
    }

    #[tokio::test]
    async fn test_resolve_mmproj_uses_explicit_path() {
        let config = make_test_llm_config(Some("/custom/vision.gguf"), true, false);
        let err = resolve_mmproj_path(&config).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("vision.gguf"));
    }
}

/// Metadata enrichment with ID field for batch processing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct MetadataEnrichmentWithId {
    pub id: String,
    pub text: String,
}

/// Metadata enrichment response with ID for matching
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct MetadataEnrichmentResponseWithId {
    pub id: String,
    pub summary: String,
    pub keywords: Vec<String>,
}
