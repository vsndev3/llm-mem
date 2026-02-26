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
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use tracing::{debug, error, info, warn};

use crate::config::LocalConfig;
use crate::error::{MemoryError, Result};
use crate::llm::extractor_types::*;

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
    config: LocalConfig,
    model_path: PathBuf,
    counters: UsageCounters,
    concurrency_limiter: Arc<Semaphore>,
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
            model_path: self.model_path.clone(),
            counters: self.counters.clone(),
            concurrency_limiter: Arc::clone(&self.concurrency_limiter),
        }
    }
}

impl LocalLLMClient {
    /// Create a new local LLM client.
    ///
    /// - Ensures the GGUF model file exists (auto-downloads known models)
    /// - Initializes the llama.cpp backend and loads the GGUF model
    /// - Initializes fastembed for local embeddings (auto-downloads on first run)
    /// - Creates the models directory if it doesn't exist
    pub async fn new(config: &LocalConfig) -> Result<Self> {
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
                &config.llm_model_file,
                config.proxy_url.as_deref(),
            )
            .await?;

            if result.freshly_downloaded {
                info!(
                    "Model downloaded: {} ({})",
                    config.llm_model_file,
                    super::model_downloader::format_size(result.size_bytes)
                );
            }

            result.path
        } else {
            let path = models_dir.join(&config.llm_model_file);
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
        let model_params = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers);
        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
            .map_err(|e| MemoryError::LLM(format!("Failed to load model: {}", e)))?;
        info!("LLM model loaded successfully");

        info!(
            "Initializing embedding model: {} (will download on first run)",
            config.embedding_model
        );
        let embed_model = parse_fastembed_model(&config.embedding_model);
        let embed_options = fastembed::InitOptions::new(embed_model)
            .with_cache_dir(models_dir)
            .with_show_download_progress(true);
        let embedding = fastembed::TextEmbedding::try_new(embed_options).map_err(|e| {
            MemoryError::Embedding(format!("Failed to initialize embedding model: {}", e))
        })?;
        info!("Embedding model initialized");

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
            backend: backend,
            embedding: Arc::new(Mutex::new(embedding)),
            config: config.clone(),
            model_path: model_path,
            counters: UsageCounters::default(),
            concurrency_limiter: Arc::new(Semaphore::new(semaphore_permits)),
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
                "Input too long: {} tokens required (prompt: {} + gen: {}), but max context is {}. Please shorten your input.",
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

        // Set up sampler
        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::temp(temperature), LlamaSampler::dist(42)]);

        // Auto-regressive generation loop
        let mut output_tokens: Vec<LlamaToken> = Vec::new();
        let mut n_cur = tokens.len();

        for _ in 0..max_tokens {
            let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(new_token);

            // Stop on end-of-generation
            if model.is_eog_token(new_token) {
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
        model: &LlamaModel,
        backend: &LlamaBackend,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        context_size: u32,
        cpu_threads: i32,
    ) -> Result<(T, String)> {
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
        let json_str = extract_json_from_text(&response).ok_or_else(|| {
            MemoryError::Parse(format!(
                "No valid JSON found in model output: {}",
                &response[..response.len().min(200)]
            ))
        })?;

        let parsed: T = serde_json::from_str(json_str).map_err(|e| {
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
                match Self::extract_json_sync::<T>(
                    &model,
                    &backend,
                    &prompt_owned,
                    max_tokens,
                    temperature,
                    context_size,
                    cpu_threads,
                ) {
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

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.counters
            .embedding_calls
            .fetch_add(1, Ordering::Relaxed);

        let embedding = Arc::clone(&self.embedding);
        let text = text.to_string();
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

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let embedding = Arc::clone(&self.embedding);
        let texts: Vec<String> = texts.to_vec();
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
        let response = self.complete(content).await?;
        Ok(response
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect())
    }

    async fn summarize(&self, content: &str, _max_length: Option<usize>) -> Result<String> {
        let response = self.complete(content).await?;
        Ok(response.trim().to_string())
    }

    async fn health_check(&self) -> Result<bool> {
        match self.embed("health check").await {
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
        self.run_extraction(prompt, 500, |response| {
            let keywords: Vec<String> = response
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            KeywordExtraction { keywords }
        })
        .await
    }

    async fn classify_memory(&self, prompt: &str) -> Result<MemoryClassification> {
        self.run_extraction(prompt, 500, |response| {
            let lower = response.to_lowercase();
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
        self.run_extraction(prompt, 1000, |response| {
            MetadataEnrichment {
                summary: response.to_string(),
                keywords: response
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            }
        })
        .await
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
            llm_model: self.config.llm_model_file.clone(),
            embedding_model: self.config.embedding_model.clone(),
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
}

// ── Helpers ────────────────────────────────────────────────────────────────

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

/// Extract a JSON object or array from text that may contain surrounding prose.
fn extract_json_from_text(text: &str) -> Option<&str> {
    let text = text.trim();

    // Strip markdown code fences if present
    let text = if text.starts_with("```json") {
        let end = text.rfind("```").unwrap_or(text.len());
        if end > 7 { &text[7..end] } else { &text[7..] }
    } else if text.starts_with("```") {
        let end = text.rfind("```").unwrap_or(text.len());
        if end > 3 { &text[3..end] } else { &text[3..] }
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
                    return Some(&text[start..start + i + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Map a user-facing embedding model name to a fastembed enum variant.
fn parse_fastembed_model(name: &str) -> fastembed::EmbeddingModel {
    match name.to_lowercase().replace(['_', ' '], "-").as_str() {
        "all-minilm-l6-v2" | "allminilml6v2" => fastembed::EmbeddingModel::AllMiniLML6V2,
        "all-minilm-l12-v2" | "allminilml12v2" => fastembed::EmbeddingModel::AllMiniLML12V2,
        "bge-small-en-v1.5" | "bgesmallenv15" => fastembed::EmbeddingModel::BGESmallENV15,
        "bge-base-en-v1.5" | "bgebaseenv15" => fastembed::EmbeddingModel::BGEBaseENV15,
        other => {
            warn!(
                "Unknown embedding model '{}', falling back to all-MiniLM-L6-v2",
                other
            );
            fastembed::EmbeddingModel::AllMiniLML6V2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_simple_object() {
        let text = r#"Here is the result: {"facts": ["a", "b"]} done"#;
        let json = extract_json_from_text(text).unwrap();
        assert_eq!(json, r#"{"facts": ["a", "b"]}"#);
    }

    #[test]
    fn test_extract_json_with_code_fence() {
        let text = "```json\n{\"key\": \"value\"}\n```";
        let json = extract_json_from_text(text).unwrap();
        assert_eq!(json, r#"{"key": "value"}"#);
    }

    #[test]
    fn test_extract_json_nested() {
        let text = r#"{"outer": {"inner": [1, 2, 3]}}"#;
        let json = extract_json_from_text(text).unwrap();
        assert_eq!(json, text);
    }

    #[test]
    fn test_extract_json_with_escaped_quotes() {
        let text = r#"{"text": "he said \"hello\""}"#;
        let json = extract_json_from_text(text).unwrap();
        assert_eq!(json, text);
    }

    #[test]
    fn test_extract_json_array() {
        let text = r#"Result: ["one", "two", "three"]"#;
        let json = extract_json_from_text(text).unwrap();
        assert_eq!(json, r#"["one", "two", "three"]"#);
    }

    #[test]
    fn test_extract_json_none_for_no_json() {
        assert!(extract_json_from_text("no json here").is_none());
    }

    #[test]
    fn test_format_chatml() {
        let formatted = format_chatml_prompt("Hello");
        assert!(formatted.contains("<|im_start|>user"));
        assert!(formatted.contains("Hello"));
        assert!(formatted.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_parse_fastembed_model() {
        assert!(matches!(
            parse_fastembed_model("all-MiniLM-L6-v2"),
            fastembed::EmbeddingModel::AllMiniLML6V2
        ));
        assert!(matches!(
            parse_fastembed_model("bge-small-en-v1.5"),
            fastembed::EmbeddingModel::BGESmallENV15
        ));
        // Unknown falls back
        assert!(matches!(
            parse_fastembed_model("nonexistent-model"),
            fastembed::EmbeddingModel::AllMiniLML6V2
        ));
    }
}
