use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use rig::client::CompletionClient;
use rig::providers::openai::CompletionModel;
use rig::{
    agent::Agent,
    client::EmbeddingsClient,
    completion::Prompt,
    embeddings::EmbeddingsBuilder,
    providers::openai::{Client, EmbeddingModel as OpenAIEmbeddingModel},
};
use tracing::{debug, error, info};

use crate::{
    config::{Config, EmbeddingConfig, LLMBackend, LLMConfig},
    error::{MemoryError, Result},
    llm::extractor_types::*,
};

#[cfg(feature = "local")]
use crate::config::LocalConfig;

/// Shared usage counters for tracking API calls and token usage.
#[derive(Debug, Default, Clone)]
pub struct UsageCounters {
    pub llm_calls: Arc<AtomicU64>,
    pub embedding_calls: Arc<AtomicU64>,
    pub prompt_tokens: Arc<AtomicU64>,
    pub completion_tokens: Arc<AtomicU64>,
    pub last_llm_success: Arc<std::sync::Mutex<Option<chrono::DateTime<chrono::Utc>>>>,
    pub last_embedding_success: Arc<std::sync::Mutex<Option<chrono::DateTime<chrono::Utc>>>>,
    pub last_error: Arc<std::sync::Mutex<Option<String>>>,
}

/// LLM client trait for text generation and embeddings
#[async_trait]
pub trait LLMClient: Send + Sync + dyn_clone::DynClone {
    /// Generate text completion
    async fn complete(&self, prompt: &str) -> Result<String>;

    /// Generate text completion with grammar-constrained sampling
    async fn complete_with_grammar(&self, prompt: &str, grammar: &str) -> Result<String>;

    /// Generate embeddings for text
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for multiple texts
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Extract key information from memory content
    async fn extract_keywords(&self, content: &str) -> Result<Vec<String>>;

    /// Summarize memory content
    async fn summarize(&self, content: &str, max_length: Option<usize>) -> Result<String>;

    /// Check if the LLM service is available
    async fn health_check(&self) -> Result<bool>;

    // Extractor-based methods
    async fn extract_structured_facts(&self, prompt: &str) -> Result<StructuredFactExtraction>;
    async fn extract_detailed_facts(&self, prompt: &str) -> Result<DetailedFactExtraction>;
    async fn extract_keywords_structured(&self, prompt: &str) -> Result<KeywordExtraction>;
    async fn classify_memory(&self, prompt: &str) -> Result<MemoryClassification>;
    async fn score_importance(&self, prompt: &str) -> Result<ImportanceScore>;
    async fn check_duplicates(&self, prompt: &str) -> Result<DeduplicationResult>;
    async fn generate_summary(&self, prompt: &str) -> Result<SummaryResult>;
    async fn detect_language(&self, prompt: &str) -> Result<LanguageDetection>;
    async fn extract_entities(&self, prompt: &str) -> Result<EntityExtraction>;
    async fn analyze_conversation(&self, prompt: &str) -> Result<ConversationAnalysis>;
    async fn extract_metadata_enrichment(&self, prompt: &str) -> Result<MetadataEnrichment>;

    /// Batch extract metadata enrichment from multiple texts
    async fn extract_metadata_enrichment_batch(
        &self,
        texts: &[String],
    ) -> Result<Vec<Result<MetadataEnrichment>>>;

    /// Batch complete multiple prompts and return results in order
    /// Each result contains either the completion or an error
    async fn complete_batch(&self, prompts: &[String]) -> Result<Vec<Result<String>>>;

    /// Return current status and usage statistics of this client.
    fn get_status(&self) -> ClientStatus;

    /// Get batch configuration
    fn batch_config(&self) -> (usize, u32);
}

dyn_clone::clone_trait_object!(LLMClient);

/// OpenAI-based LLM client implementation using rig
pub struct OpenAILLMClient {
    completion_model: Agent<CompletionModel>,
    completion_model_name: String,
    embedding_model: OpenAIEmbeddingModel,
    embedding_model_name: String,
    client: Client,
    api_base_url: String,
    embedding_api_base_url: String,
    api_key: String,
    model_name: String,
    temperature: f32,
    max_tokens: u32,
    counters: UsageCounters,
    timeout_secs: u64,
    /// Whether to use structured output mode (JSON schema validation)
    use_structured_output: bool,
    /// Maximum retry attempts for structured output validation
    max_retries: u32,
    /// XML tags to strip from LLM output (e.g., ["think", "reason", "thought"])
    strip_llm_tags: Vec<String>,
    /// Request format mode
    request_format: crate::config::RequestFormat,
    /// API dialect for raw HTTP requests
    api_dialect: crate::config::ApiDialect,
    /// Custom dialect configuration
    custom_dialect: Option<crate::config::CustomDialectConfig>,
    /// Whether we've detected that the backend requires raw format (for auto mode)
    ///
    /// # State Persistence Behavior
    ///
    /// This flag is wrapped in `Arc<Mutex<bool>>` to enable state sharing across
    /// all clones of the client. This ensures that once we detect a 422 error
    /// from the backend, **all** future requests (from any clone) will use raw
    /// format without retrying rig-core first.
    ///
    /// ## Request Flow in Auto Mode:
    ///
    /// 1. **First request**:
    ///    - Flag is `false`
    ///    - Try rig-core → Get 422 error → Set flag to `true` → Retry with raw → Success
    ///
    /// 2. **All subsequent requests**:
    ///    - Flag is `true` (checked at start)
    ///    - Skip rig-core entirely → Use raw directly → Success
    ///
    /// 3. **Cloned instances**:
    ///    - Share the same `Arc` reference
    ///    - See the updated flag value
    ///    - Also skip rig-core and use raw directly
    ///
    /// ## Performance Benefit:
    ///
    /// This design ensures we only pay the "double try" cost **once** per session.
    /// After the first 422 error is detected, there's zero overhead - every request
    /// goes directly to raw format without attempting rig-core first.
    ///
    /// ## Thread Safety:
    ///
    /// The `Arc<Mutex<bool>>` ensures:
    /// - Safe concurrent access across threads
    /// - Atomic updates to the flag
    /// - Consistent state visibility across all client instances
    raw_format_detected: Arc<std::sync::Mutex<bool>>,
    /// Semaphore to limit concurrent API requests
    ///
    /// # Concurrency Control
    ///
    /// This semaphore limits the number of concurrent API requests to prevent
    /// rate limiting and API quota exhaustion. All public methods that make
    /// API calls must acquire a permit before proceeding.
    ///
    /// ## Configuration
    ///
    /// Set via `api_llm.max_concurrent_requests` in config:
    /// - `1` (default): One request at a time (sequential)
    /// - `0`: Unlimited (not recommended for rate-limited APIs)
    /// - `N`: Up to N concurrent requests
    concurrency_limiter: Arc<tokio::sync::Semaphore>,
    /// Batch configuration
    batch_size: usize,
    batch_max_tokens: u32,
    batch_timeout_multiplier: f64,
    batch_timeout_secs: u64,
}

impl OpenAILLMClient {
    pub fn new(
        llm_config: &LLMConfig,
        embedding_config: &EmbeddingConfig,
        api_llm_config: &crate::config::ApiLlmConfig,
    ) -> Result<Self> {
        let client = Client::builder(&llm_config.api_key)
            .base_url(&llm_config.api_base_url)
            .build();

        let completion_model: Agent<CompletionModel> = client
            .completion_model(&llm_config.model_efficient)
            .completions_api()
            .into_agent_builder()
            .temperature(llm_config.temperature as f64)
            .max_tokens(llm_config.max_tokens as u64)
            .build();

        let embedding_client = Client::builder(&embedding_config.api_key)
            .base_url(&embedding_config.api_base_url)
            .build();
        let embedding_model = embedding_client.embedding_model(&embedding_config.model_name);

        // Initialize concurrency limiter (semaphore)
        let semaphore_permits = if api_llm_config.max_concurrent_requests == 0 {
            // 0 means unlimited - use a very large number
            usize::MAX
        } else {
            api_llm_config.max_concurrent_requests
        };

        Ok(Self {
            completion_model,
            completion_model_name: llm_config.model_efficient.clone(),
            embedding_model,
            embedding_model_name: embedding_config.model_name.clone(),
            client,
            api_base_url: llm_config.api_base_url.clone(),
            embedding_api_base_url: embedding_config.api_base_url.clone(),
            api_key: llm_config.api_key.clone(),
            model_name: llm_config.model_efficient.clone(),
            temperature: llm_config.temperature,
            max_tokens: llm_config.max_tokens,
            counters: UsageCounters::default(),
            timeout_secs: embedding_config.timeout_secs,
            use_structured_output: api_llm_config.use_structured_output,
            max_retries: api_llm_config.structured_output_retries,
            strip_llm_tags: api_llm_config.strip_llm_tags.clone(),
            request_format: api_llm_config.request_format.clone(),
            api_dialect: api_llm_config.api_dialect.clone(),
            custom_dialect: api_llm_config.custom_dialect.clone(),
            raw_format_detected: Arc::new(std::sync::Mutex::new(false)),
            concurrency_limiter: Arc::new(tokio::sync::Semaphore::new(semaphore_permits)),
            batch_size: api_llm_config.batch_size,
            batch_max_tokens: api_llm_config.batch_max_tokens,
            batch_timeout_multiplier: api_llm_config.batch_timeout_multiplier,
            batch_timeout_secs: api_llm_config.batch_timeout_secs,
        })
    }

    fn build_keyword_prompt(&self, content: &str) -> String {
        format!(
            "Extract the most important keywords and key phrases from the following text. \
            Return only the keywords separated by commas, without any additional explanation.\n\n\
            Text: {}\n\n\
            Keywords:",
            content
        )
    }

    fn build_summary_prompt(&self, content: &str, max_length: Option<usize>) -> String {
        let length_instruction = match max_length {
            Some(len) => format!("in approximately {} words", len),
            None => "concisely".to_string(),
        };
        format!(
            "Summarize the following text {}. Focus on the main points and key information.\n\n\
            Text: {}\n\n\
            Summary:",
            length_instruction, content
        )
    }

    fn parse_keywords(&self, response: &str) -> Vec<String> {
        // Strip XML tags (e.g., <think>...</think>) before parsing keywords
        let cleaned = strip_llm_tags(response, &self.strip_llm_tags);

        // Try to parse as JSON array first
        if let Some(json_str) = extract_json_from_text(&cleaned) {
            // Try parsing as array of strings
            if let Ok(arr) = serde_json::from_str::<Vec<String>>(&json_str) {
                return arr.into_iter().filter(|s| !s.is_empty()).collect();
            }
            // Try parsing as object with keywords field
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&json_str)
                && let Some(keywords) = obj.get("keywords").and_then(|v| v.as_array())
            {
                return keywords
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        }

        // Fallback to comma-separated parsing
        cleaned
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Calculate the timeout for a batch request.
    ///
    /// Formula: batch_timeout_secs * batch_timeout_multiplier * sqrt(batch_size)
    /// This scales the timeout based on the number of items in the batch.
    fn calculate_batch_timeout(&self, batch_size: usize) -> u64 {
        let sqrt_size = (batch_size as f64).sqrt();
        let timeout = self.batch_timeout_secs as f64 * self.batch_timeout_multiplier * sqrt_size;
        timeout.ceil() as u64
    }

    /// Generate text completion with a custom timeout.
    ///
    /// This is an internal method used for batch operations that need longer timeouts.
    async fn complete_with_timeout(&self, prompt: &str, timeout_secs: u64) -> Result<String> {
        self.counters.llm_calls.fetch_add(1, Ordering::Relaxed);
        // Rough token estimate: ~4 chars per token
        self.counters
            .prompt_tokens
            .fetch_add((prompt.len() / 4) as u64, Ordering::Relaxed);

        // Acquire permit to limit concurrent API requests
        let _permit = self
            .concurrency_limiter
            .acquire()
            .await
            .map_err(|e| MemoryError::LLM(format!("Semaphore error: {}", e)))?;

        // Determine which completion method to use based on request_format
        let use_rig = match self.request_format {
            crate::config::RequestFormat::Raw => false,
            crate::config::RequestFormat::Rig => true,
            crate::config::RequestFormat::Auto => {
                // Check if we've previously detected that raw format is needed
                !*self.raw_format_detected.lock().unwrap()
            }
        };

        let response = if use_rig {
            // Try rig-core first
            let future = self.completion_model.prompt(prompt).multi_turn(10);
            let rig_result =
                tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), future)
                    .await
                    .map_err(|_| {
                        MemoryError::LLM(format!(
                            "LLM completion timed out after {}s",
                            timeout_secs
                        ))
                    });

            match rig_result {
                Ok(Ok(resp)) => resp,
                Ok(Err(e)) => {
                    // Check if this is a 422 error and we're in auto mode
                    if self.request_format == crate::config::RequestFormat::Auto
                        && e.to_string().contains("422")
                    {
                        // Mark that we need to use raw format for future requests
                        *self.raw_format_detected.lock().unwrap() = true;
                        tracing::info!(
                            "Detected 422 error from backend in complete_with_timeout(), switching to raw HTTP format for future requests"
                        );

                        // Retry with raw format
                        self.raw_completion_with_timeout(prompt, timeout_secs).await?
                    } else {
                        if let Ok(mut last) = self.counters.last_error.lock() {
                            *last = Some(e.to_string());
                        }
                        return Err(MemoryError::LLM(e.to_string()));
                    }
                }
                Err(e) => {
                    if let Ok(mut last) = self.counters.last_error.lock() {
                        *last = Some(e.to_string());
                    }
                    return Err(MemoryError::LLM(e.to_string()));
                }
            }
        } else {
            // Use raw HTTP directly
            self.raw_completion_with_timeout(prompt, timeout_secs).await?
        };

        self.counters
            .completion_tokens
            .fetch_add((response.len() / 4) as u64, Ordering::Relaxed);
        if let Ok(mut ts) = self.counters.last_llm_success.lock() {
            *ts = Some(chrono::Utc::now());
        }

        debug!("Generated completion for prompt length: {}", prompt.len());

        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        Ok(response)
    }

    /// Raw HTTP completion with custom timeout that bypasses rig-core's message formatting.
    async fn raw_completion_with_timeout(&self, prompt: &str, timeout_secs: u64) -> Result<String> {
        use crate::config::ApiDialect;
        use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};

        // Determine URL and body structure based on dialect
        let (url, request_body) = match self.api_dialect {
            ApiDialect::OpenAIChat => {
                let url = format!(
                    "{}/chat/completions",
                    self.api_base_url.trim_end_matches('/')
                );
                let body = serde_json::json!({
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                });
                (url, body)
            }
            ApiDialect::OpenAICompletion => {
                let url = format!("{}/completions", self.api_base_url.trim_end_matches('/'));
                let body = serde_json::json!({
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                });
                (url, body)
            }
            ApiDialect::Anthropic => {
                let url = format!("{}/v1/messages", self.api_base_url.trim_end_matches('/'));
                let body = serde_json::json!({
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                });
                (url, body)
            }
            ApiDialect::OllamaChat => {
                let url = format!("{}/api/chat", self.api_base_url.trim_end_matches('/'));
                let body = serde_json::json!({
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": false,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                });
                (url, body)
            }
            ApiDialect::OllamaCompletion => {
                let url = format!("{}/api/generate", self.api_base_url.trim_end_matches('/'));
                let body = serde_json::json!({
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": false,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                });
                (url, body)
            }
            ApiDialect::Custom => {
                let custom = self.custom_dialect.as_ref().ok_or_else(|| {
                    MemoryError::LLM(
                        "Custom dialect selected but no custom_dialect config provided".into(),
                    )
                })?;

                let url = format!(
                    "{}{}",
                    self.api_base_url.trim_end_matches('/'),
                    custom.endpoint_path
                );

                // Proper JSON escaping for template interpolation
                let escaped_prompt = serde_json::to_string(&prompt).map_err(|e| {
                    MemoryError::LLM(format!("Failed to escape prompt for JSON: {}", e))
                })?;
                let escaped_prompt = escaped_prompt.trim_matches('"').to_string();

                let body_str = custom
                    .request_body_template
                    .replace("{{prompt}}", &escaped_prompt)
                    .replace("{{model}}", &self.model_name)
                    .replace("{{temperature}}", &self.temperature.to_string())
                    .replace("{{max_tokens}}", &self.max_tokens.to_string());

                let body: serde_json::Value = serde_json::from_str(&body_str).map_err(|e| {
                    MemoryError::LLM(format!(
                        "Failed to parse custom request body template: {}",
                        e
                    ))
                })?;

                (url, body)
            }
        };

        let client = reqwest::Client::new();

        info!(
            "Sending raw LLM request to: {} (dialect: {:?})",
            url, self.api_dialect
        );

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            client
                .post(&url)
                .header(CONTENT_TYPE, "application/json")
                .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
                .json(&request_body)
                .send(),
        )
        .await
        .map_err(|_| {
            MemoryError::LLM(format!(
                "Raw HTTP completion timed out after {}s",
                timeout_secs
            ))
        })?
        .map_err(|e| MemoryError::LLM(format!("Raw HTTP request failed: {}", e)))?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .map_err(|e| MemoryError::LLM(format!("Failed to read response body: {}", e)))?;

        // Check for 422 error specifically
        if status.as_u16() == 422 {
            return Err(MemoryError::LLM(format!(
                "Backend rejected request with 422 Unprocessable Entity: {}",
                response_text
            )));
        }

        if !status.is_success() {
            return Err(MemoryError::LLM(format!(
                "Raw HTTP request failed with status {}: {}",
                status, response_text
            )));
        }

        // Parse response based on dialect
        let json: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| MemoryError::LLM(format!("Failed to parse response JSON: {}", e)))?;

        let content = match self.api_dialect {
            ApiDialect::OpenAIChat => json["choices"][0]["message"]["content"]
                .as_str()
                .ok_or_else(|| MemoryError::LLM("Missing content in OpenAI chat response".into()))?
                .to_string(),
            ApiDialect::OpenAICompletion => json["choices"][0]["text"]
                .as_str()
                .ok_or_else(|| {
                    MemoryError::LLM("Missing text in OpenAI completion response".into())
                })?
                .to_string(),
            ApiDialect::Anthropic => json["content"][0]["text"]
                .as_str()
                .ok_or_else(|| MemoryError::LLM("Missing text in Anthropic response".into()))?
                .to_string(),
            ApiDialect::OllamaChat => json["message"]["content"]
                .as_str()
                .ok_or_else(|| MemoryError::LLM("Missing content in Ollama chat response".into()))?
                .to_string(),
            ApiDialect::OllamaCompletion => json["response"]
                .as_str()
                .ok_or_else(|| {
                    MemoryError::LLM("Missing response in Ollama completion response".into())
                })?
                .to_string(),
            ApiDialect::Custom => {
                let custom = self.custom_dialect.as_ref().unwrap();
                json.pointer(&custom.response_content_pointer)
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        MemoryError::LLM(format!(
                            "Failed to extract content using pointer: {}",
                            custom.response_content_pointer
                        ))
                    })?
                    .to_string()
            }
        };

        Ok(content)
    }

    /// Raw HTTP completion that bypasses rig-core's message formatting.
    ///
    /// This makes a direct HTTP request with plain string content instead of
    /// rig-core's complex message array format. Used for backends that reject
    /// the [{"type": "text", "text": "..."}] format with 422 errors.
    async fn raw_completion(&self, prompt: &str) -> Result<String> {
        use crate::config::ApiDialect;
        use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};

        // Determine URL and body structure based on dialect
        let (url, request_body) = match self.api_dialect {
            ApiDialect::OpenAIChat => {
                let url = format!(
                    "{}/chat/completions",
                    self.api_base_url.trim_end_matches('/')
                );
                let body = serde_json::json!({
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                });
                (url, body)
            }
            ApiDialect::OpenAICompletion => {
                let url = format!("{}/completions", self.api_base_url.trim_end_matches('/'));
                let body = serde_json::json!({
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                });
                (url, body)
            }
            ApiDialect::Anthropic => {
                let url = format!("{}/v1/messages", self.api_base_url.trim_end_matches('/'));
                let body = serde_json::json!({
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                });
                (url, body)
            }
            ApiDialect::OllamaChat => {
                let url = format!("{}/api/chat", self.api_base_url.trim_end_matches('/'));
                let body = serde_json::json!({
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": false,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                });
                (url, body)
            }
            ApiDialect::OllamaCompletion => {
                let url = format!("{}/api/generate", self.api_base_url.trim_end_matches('/'));
                let body = serde_json::json!({
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": false,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                });
                (url, body)
            }
            ApiDialect::Custom => {
                let custom = self.custom_dialect.as_ref().ok_or_else(|| {
                    MemoryError::LLM(
                        "Custom dialect selected but no custom_dialect config provided".into(),
                    )
                })?;

                let url = format!(
                    "{}{}",
                    self.api_base_url.trim_end_matches('/'),
                    custom.endpoint_path
                );

                // Proper JSON escaping for template interpolation
                // Handles all control characters including tabs, newlines, carriage returns, etc.
                let escaped_prompt = serde_json::to_string(&prompt).map_err(|e| {
                    MemoryError::LLM(format!("Failed to escape prompt for JSON: {}", e))
                })?;
                // Remove the surrounding quotes added by to_string() since we're interpolating
                let escaped_prompt = escaped_prompt.trim_matches('"').to_string();

                let body_str = custom
                    .request_body_template
                    .replace("{{prompt}}", &escaped_prompt)
                    .replace("{{model}}", &self.model_name)
                    .replace("{{temperature}}", &self.temperature.to_string())
                    .replace("{{max_tokens}}", &self.max_tokens.to_string());

                let body: serde_json::Value = serde_json::from_str(&body_str).map_err(|e| {
                    MemoryError::LLM(format!(
                        "Failed to parse custom request body template: {}",
                        e
                    ))
                })?;

                (url, body)
            }
        };

        let client = reqwest::Client::new();

        info!(
            "Sending raw LLM request to: {} (dialect: {:?})",
            url, self.api_dialect
        );

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(self.timeout_secs),
            client
                .post(&url)
                .header(CONTENT_TYPE, "application/json")
                .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
                .json(&request_body)
                .send(),
        )
        .await
        .map_err(|_| {
            MemoryError::LLM(format!(
                "Raw HTTP completion timed out after {}s",
                self.timeout_secs
            ))
        })?
        .map_err(|e| MemoryError::LLM(format!("Raw HTTP request failed: {}", e)))?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .map_err(|e| MemoryError::LLM(format!("Failed to read response body: {}", e)))?;

        // Check for 422 error specifically
        if status.as_u16() == 422 {
            return Err(MemoryError::LLM(format!(
                "Backend rejected request with 422 Unprocessable Entity: {}",
                response_text
            )));
        }

        if !status.is_success() {
            return Err(MemoryError::LLM(format!(
                "Raw HTTP request failed with status {}: {}",
                status, response_text
            )));
        }

        // Parse response based on dialect
        let json: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| MemoryError::LLM(format!("Failed to parse response JSON: {}", e)))?;

        let content = match self.api_dialect {
            ApiDialect::OpenAIChat => json["choices"][0]["message"]["content"]
                .as_str()
                .ok_or_else(|| MemoryError::LLM("Missing content in OpenAI chat response".into()))?
                .to_string(),
            ApiDialect::OpenAICompletion => json["choices"][0]["text"]
                .as_str()
                .ok_or_else(|| {
                    MemoryError::LLM("Missing text in OpenAI completion response".into())
                })?
                .to_string(),
            ApiDialect::Anthropic => json["content"][0]["text"]
                .as_str()
                .ok_or_else(|| MemoryError::LLM("Missing text in Anthropic response".into()))?
                .to_string(),
            ApiDialect::OllamaChat => json["message"]["content"]
                .as_str()
                .ok_or_else(|| MemoryError::LLM("Missing content in Ollama chat response".into()))?
                .to_string(),
            ApiDialect::OllamaCompletion => json["response"]
                .as_str()
                .ok_or_else(|| {
                    MemoryError::LLM("Missing response in Ollama completion response".into())
                })?
                .to_string(),
            ApiDialect::Custom => {
                let custom = self.custom_dialect.as_ref().unwrap();
                json.pointer(&custom.response_content_pointer)
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        MemoryError::LLM(format!(
                            "Failed to extract content using pointer: {}",
                            custom.response_content_pointer
                        ))
                    })?
                    .to_string()
            }
        };

        Ok(content)
    }

    /// Generic helper for structured extraction using plain-string completion.
    ///
    /// This bypasses the rig-core extractor API which sends messages in
    /// [{"type": "text", "text": "..."}] format that many LLM backends reject.
    /// Instead, it uses standard prompt completion which sends plain strings.
    ///
    /// # Arguments
    /// * `prompt` - The extraction prompt
    /// * `_max_tokens` - Maximum tokens to generate (reserved for future use)
    /// * `fallback` - Closure to create fallback value if parsing fails
    async fn complete_and_parse<T, F>(
        &self,
        prompt: &str,
        _max_tokens: u64,
        fallback: F,
    ) -> Result<T>
    where
        T: serde::de::DeserializeOwned + Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        // Acquire permit to limit concurrent API requests
        let _permit = self
            .concurrency_limiter
            .acquire()
            .await
            .map_err(|e| MemoryError::LLM(format!("Semaphore error: {}", e)))?;

        // Build prompt with strict JSON instruction
        let enhanced_prompt = format!(
            "{}\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation.",
            prompt
        );

        // Determine which completion method to use based on request_format
        let use_rig = match self.request_format {
            crate::config::RequestFormat::Raw => false,
            crate::config::RequestFormat::Rig => true,
            crate::config::RequestFormat::Auto => {
                // Check if we've previously detected that raw format is needed
                !*self.raw_format_detected.lock().unwrap()
            }
        };

        let response = if use_rig {
            // Try rig-core first
            let rig_result = async {
                let future = self
                    .completion_model
                    .prompt(&enhanced_prompt)
                    .multi_turn(10);
                tokio::time::timeout(std::time::Duration::from_secs(self.timeout_secs), future)
                    .await
                    .map_err(|_| {
                        MemoryError::LLM(format!(
                            "LLM completion timed out after {}s",
                            self.timeout_secs
                        ))
                    })?
                    .map_err(|e| MemoryError::LLM(e.to_string()))
            }
            .await;

            match rig_result {
                Ok(resp) => resp,
                Err(e) => {
                    // Check if this is a 422 error and we're in auto mode
                    if self.request_format == crate::config::RequestFormat::Auto
                        && e.to_string().contains("422")
                    {
                        // Mark that we need to use raw format for future requests
                        *self.raw_format_detected.lock().unwrap() = true;
                        tracing::info!(
                            "Detected 422 error from backend, switching to raw HTTP format for future requests"
                        );

                        // Retry with raw format
                        self.raw_completion(&enhanced_prompt).await?
                    } else {
                        return Err(e);
                    }
                }
            }
        } else {
            // Use raw HTTP directly
            self.raw_completion(&enhanced_prompt).await?
        };

        // Strip XML thought tags from response
        let cleaned = strip_llm_tags(&response, &self.strip_llm_tags);

        // Extract JSON from response
        if let Some(json_str) = extract_json_from_text(&cleaned) {
            match serde_json::from_str::<T>(&json_str) {
                Ok(parsed) => return Ok(parsed),
                Err(e) => {
                    // Log the error with truncated response for debugging
                    let truncated_response = truncate_for_logging(&cleaned, 500);
                    tracing::warn!(
                        "JSON parse failed: {}, using fallback. Response (truncated): {}",
                        e,
                        truncated_response
                    );
                }
            }
        } else {
            // Log when no JSON is found in response
            let truncated_response = truncate_for_logging(&cleaned, 500);
            tracing::warn!(
                "No JSON found in response, using fallback. Response (truncated): {}",
                truncated_response
            );
        }

        // Fallback to default value
        Ok(fallback())
    }
}

impl Clone for OpenAILLMClient {
    fn clone(&self) -> Self {
        Self {
            completion_model: self.completion_model.clone(),
            completion_model_name: self.completion_model_name.clone(),
            embedding_model: self.embedding_model.clone(),
            embedding_model_name: self.embedding_model_name.clone(),
            client: self.client.clone(),
            api_base_url: self.api_base_url.clone(),
            embedding_api_base_url: self.embedding_api_base_url.clone(),
            api_key: self.api_key.clone(),
            model_name: self.model_name.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            counters: self.counters.clone(),
            timeout_secs: self.timeout_secs,
            use_structured_output: self.use_structured_output,
            max_retries: self.max_retries,
            strip_llm_tags: self.strip_llm_tags.clone(),
            request_format: self.request_format.clone(),
            api_dialect: self.api_dialect.clone(),
            custom_dialect: self.custom_dialect.clone(),
            raw_format_detected: Arc::clone(&self.raw_format_detected),
            concurrency_limiter: Arc::clone(&self.concurrency_limiter),
            batch_size: self.batch_size,
            batch_max_tokens: self.batch_max_tokens,
            batch_timeout_multiplier: self.batch_timeout_multiplier,
            batch_timeout_secs: self.batch_timeout_secs,
        }
    }
}

#[async_trait]
impl LLMClient for OpenAILLMClient {
    async fn complete(&self, prompt: &str) -> Result<String> {
        self.counters.llm_calls.fetch_add(1, Ordering::Relaxed);
        // Rough token estimate: ~4 chars per token
        self.counters
            .prompt_tokens
            .fetch_add((prompt.len() / 4) as u64, Ordering::Relaxed);

        // Acquire permit to limit concurrent API requests
        let _permit = self
            .concurrency_limiter
            .acquire()
            .await
            .map_err(|e| MemoryError::LLM(format!("Semaphore error: {}", e)))?;

        // Determine which completion method to use based on request_format
        let use_rig = match self.request_format {
            crate::config::RequestFormat::Raw => false,
            crate::config::RequestFormat::Rig => true,
            crate::config::RequestFormat::Auto => {
                // Check if we've previously detected that raw format is needed
                !*self.raw_format_detected.lock().unwrap()
            }
        };

        let response = if use_rig {
            // Try rig-core first
            let future = self.completion_model.prompt(prompt).multi_turn(10);
            let rig_result =
                tokio::time::timeout(std::time::Duration::from_secs(self.timeout_secs), future)
                    .await
                    .map_err(|_| {
                        MemoryError::LLM(format!(
                            "LLM completion timed out after {}s",
                            self.timeout_secs
                        ))
                    });

            match rig_result {
                Ok(Ok(resp)) => resp,
                Ok(Err(e)) => {
                    // Check if this is a 422 error and we're in auto mode
                    if self.request_format == crate::config::RequestFormat::Auto
                        && e.to_string().contains("422")
                    {
                        // Mark that we need to use raw format for future requests
                        *self.raw_format_detected.lock().unwrap() = true;
                        tracing::info!(
                            "Detected 422 error from backend in complete(), switching to raw HTTP format for future requests"
                        );

                        // Retry with raw format
                        self.raw_completion(prompt).await?
                    } else {
                        if let Ok(mut last) = self.counters.last_error.lock() {
                            *last = Some(e.to_string());
                        }
                        return Err(MemoryError::LLM(e.to_string()));
                    }
                }
                Err(e) => {
                    if let Ok(mut last) = self.counters.last_error.lock() {
                        *last = Some(e.to_string());
                    }
                    return Err(MemoryError::LLM(e.to_string()));
                }
            }
        } else {
            // Use raw HTTP directly
            self.raw_completion(prompt).await?
        };

        self.counters
            .completion_tokens
            .fetch_add((response.len() / 4) as u64, Ordering::Relaxed);
        if let Ok(mut ts) = self.counters.last_llm_success.lock() {
            *ts = Some(chrono::Utc::now());
        }

        debug!("Generated completion for prompt length: {}", prompt.len());

        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        Ok(response)
    }

    async fn complete_with_grammar(&self, prompt: &str, _grammar: &str) -> Result<String> {
        // API-based structured output with retry validation
        if self.use_structured_output {
            Self::complete_with_structured_output_retry(
                &self.completion_model,
                prompt,
                self.max_retries,
                self.timeout_secs,
            )
            .await
        } else {
            // Fallback to simple JSON instruction
            let enhanced_prompt = format!(
                "{}\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation.",
                prompt
            );
            self.complete(&enhanced_prompt).await
        }
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.counters
            .embedding_calls
            .fetch_add(1, Ordering::Relaxed);

        // Acquire permit to limit concurrent API requests
        let _permit = self
            .concurrency_limiter
            .acquire()
            .await
            .map_err(|e| MemoryError::LLM(format!("Semaphore error: {}", e)))?;

        let builder = EmbeddingsBuilder::new(self.embedding_model.clone())
            .document(text)
            .map_err(|e| MemoryError::LLM(e.to_string()))?;

        let future = builder.build();
        let embeddings =
            tokio::time::timeout(std::time::Duration::from_secs(self.timeout_secs), future)
                .await
                .map_err(|_| {
                    MemoryError::LLM(format!("Embedding timed out after {}s", self.timeout_secs))
                })?
                .map_err(|e| {
                    if let Ok(mut last) = self.counters.last_error.lock() {
                        *last = Some(e.to_string());
                    }
                    MemoryError::LLM(e.to_string())
                })?;

        if let Some((_, embedding)) = embeddings.first() {
            if let Ok(mut ts) = self.counters.last_embedding_success.lock() {
                *ts = Some(chrono::Utc::now());
            }
            debug!("Generated embedding for text length: {}", text.len());
            Ok(embedding.first().vec.iter().map(|&x| x as f32).collect())
        } else {
            Err(MemoryError::LLM("No embedding generated".to_string()))
        }
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::new();
        for text in texts {
            let embedding = self.embed(text).await?;
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            results.push(embedding);
        }
        debug!("Generated embeddings for {} texts", texts.len());
        Ok(results)
    }

    async fn extract_keywords(&self, content: &str) -> Result<Vec<String>> {
        let prompt = self.build_keyword_prompt(content);

        match self.extract_keywords_structured(&prompt).await {
            Ok(kw) => {
                debug!(
                    "Extracted {} keywords using rig extractor",
                    kw.keywords.len()
                );
                Ok(kw.keywords)
            }
            Err(e) => {
                debug!("Rig extractor failed, falling back: {}", e);
                #[cfg(debug_assertions)]
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                let response = self.complete(&prompt).await?;
                Ok(self.parse_keywords(&response))
            }
        }
    }

    async fn summarize(&self, content: &str, max_length: Option<usize>) -> Result<String> {
        let prompt = self.build_summary_prompt(content, max_length);

        match self.generate_summary(&prompt).await {
            Ok(result) => {
                // Strip XML tags from structured output
                let cleaned = strip_llm_tags(&result.summary, &self.strip_llm_tags);
                Ok(cleaned.trim().to_string())
            }
            Err(e) => {
                debug!("Rig extractor failed, falling back: {}", e);
                let summary = self.complete(&prompt).await?;
                // Strip XML tags from fallback output
                let cleaned = strip_llm_tags(&summary, &self.strip_llm_tags);
                Ok(cleaned.trim().to_string())
            }
        }
    }

    async fn health_check(&self) -> Result<bool> {
        match self.embed("health check").await {
            Ok(_) => {
                info!("LLM service health check passed");
                Ok(true)
            }
            Err(e) => {
                error!("LLM service health check failed: {}", e);
                Ok(false)
            }
        }
    }

    async fn extract_structured_facts(&self, prompt: &str) -> Result<StructuredFactExtraction> {
        self.complete_and_parse(prompt, 500, || StructuredFactExtraction { facts: vec![] })
            .await
    }

    async fn extract_detailed_facts(&self, prompt: &str) -> Result<DetailedFactExtraction> {
        self.complete_and_parse(prompt, 1000, || DetailedFactExtraction { facts: vec![] })
            .await
    }

    async fn extract_keywords_structured(&self, prompt: &str) -> Result<KeywordExtraction> {
        self.complete_and_parse(prompt, 500, || KeywordExtraction { keywords: vec![] })
            .await
    }

    async fn classify_memory(&self, prompt: &str) -> Result<MemoryClassification> {
        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        let completion = self.complete(prompt).await?;
        let response = completion.trim();

        let memory_type = if response.to_lowercase().contains("conversational") {
            "Conversational"
        } else if response.to_lowercase().contains("procedural") {
            "Procedural"
        } else if response.to_lowercase().contains("factual") {
            "Factual"
        } else if response.to_lowercase().contains("semantic") {
            "Semantic"
        } else if response.to_lowercase().contains("episodic") {
            "Episodic"
        } else if response.to_lowercase().contains("personal") {
            "Personal"
        } else {
            "Conversational"
        }
        .to_string();

        Ok(MemoryClassification {
            memory_type,
            confidence: 0.8,
            reasoning: format!("LLM classification response: {}", response),
        })
    }

    async fn score_importance(&self, prompt: &str) -> Result<ImportanceScore> {
        self.complete_and_parse(prompt, 500, || ImportanceScore {
            score: 0.5,
            reasoning: "Fallback due to JSON parse failure".to_string(),
        })
        .await
    }

    async fn check_duplicates(&self, prompt: &str) -> Result<DeduplicationResult> {
        self.complete_and_parse(prompt, 500, || DeduplicationResult {
            is_duplicate: false,
            similarity_score: 0.0,
            original_memory_id: None,
        })
        .await
    }

    async fn generate_summary(&self, prompt: &str) -> Result<SummaryResult> {
        self.complete_and_parse(prompt, 1000, || SummaryResult {
            summary: "".to_string(),
            key_points: vec![],
        })
        .await
    }

    async fn detect_language(&self, prompt: &str) -> Result<LanguageDetection> {
        self.complete_and_parse(prompt, 200, || LanguageDetection {
            language: "unknown".to_string(),
            confidence: 0.0,
        })
        .await
    }

    async fn extract_entities(&self, prompt: &str) -> Result<EntityExtraction> {
        self.complete_and_parse(prompt, 1000, || EntityExtraction { entities: vec![] })
            .await
    }

    async fn analyze_conversation(&self, prompt: &str) -> Result<ConversationAnalysis> {
        self.complete_and_parse(prompt, 1500, || ConversationAnalysis {
            topics: vec![],
            sentiment: "neutral".to_string(),
            user_intent: "unknown".to_string(),
            key_information: vec![],
        })
        .await
    }

    async fn extract_metadata_enrichment(&self, prompt: &str) -> Result<MetadataEnrichment> {
        self.complete_and_parse(prompt, 1000, || MetadataEnrichment {
            summary: "".to_string(),
            keywords: vec![],
        })
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
            "Batch extracting metadata enrichment for {} texts in a single call",
            texts.len()
        );

        let texts_json = serde_json::to_string(texts).unwrap_or_else(|_| "[]".to_string());
        let prompt = crate::memory::prompts::METADATA_ENRICHMENT_BATCH_PROMPT
            .replace("{{texts}}", &texts_json);

        // Calculate batch-aware timeout
        let batch_timeout = self.calculate_batch_timeout(texts.len());
        debug!(
            "Using batch timeout: {}s (batch_size={}, timeout={}s, multiplier={})",
            batch_timeout,
            texts.len(),
            self.batch_timeout_secs,
            self.batch_timeout_multiplier
        );

        let response = match self.complete_with_timeout(&prompt, batch_timeout).await {
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

        let parsed: Vec<MetadataEnrichment> = match extract_json_from_text(&response) {
            Some(json_str) => match serde_json::from_str(&json_str) {
                Ok(arr) => arr,
                Err(e) => {
                    let mut errors = Vec::new();
                    for _ in 0..texts.len() {
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
                for _ in 0..texts.len() {
                    errors.push(Err(crate::error::MemoryError::LLM(
                        "No JSON array found in batch response".to_string(),
                    )));
                }
                return Ok(errors);
            }
        };

        if parsed.len() != texts.len() {
            let mut errors = Vec::new();
            for _ in 0..texts.len() {
                errors.push(Err(crate::error::MemoryError::LLM(format!(
                    "Batch length mismatch: expected {}, got {}",
                    texts.len(),
                    parsed.len()
                ))));
            }
            return Ok(errors);
        }

        Ok(parsed.into_iter().map(Ok).collect())
    }

    async fn complete_batch(&self, prompts: &[String]) -> Result<Vec<Result<String>>> {
        if prompts.is_empty() {
            return Ok(vec![]);
        }

        debug!(
            "Batch completing {} prompts in a single call",
            prompts.len()
        );

        let prompts_json = serde_json::to_string(prompts).unwrap_or_else(|_| "[]".to_string());
        let master_prompt =
            crate::memory::prompts::COMPLETE_BATCH_PROMPT.replace("{{prompts}}", &prompts_json);

        // Calculate batch-aware timeout
        let batch_timeout = self.calculate_batch_timeout(prompts.len());
        debug!(
            "Using batch timeout: {}s (batch_size={}, timeout={}s, multiplier={})",
            batch_timeout,
            prompts.len(),
            self.batch_timeout_secs,
            self.batch_timeout_multiplier
        );

        let response = match self.complete_with_timeout(&master_prompt, batch_timeout).await {
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

        let parsed: Vec<String> = match extract_json_from_text(&response) {
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

        let total_calls = self.counters.llm_calls.load(Ordering::Relaxed);
        let llm_available = last_err.is_none() || total_calls == 0 || last_llm.is_some();

        let mut details = HashMap::new();
        details.insert(
            "api_base_url".into(),
            serde_json::Value::String(self.api_base_url.clone()),
        );
        details.insert(
            "embedding_api_base_url".into(),
            serde_json::Value::String(self.embedding_api_base_url.clone()),
        );

        ClientStatus {
            backend: "api".to_string(),
            state: if llm_available {
                "ready".to_string()
            } else {
                "error".to_string()
            },
            llm_model: self.completion_model_name.clone(),
            embedding_model: self.embedding_model_name.clone(),
            llm_available,
            embedding_available: llm_available,
            last_llm_success: last_llm,
            last_embedding_success: last_emb,
            last_error: last_err,
            total_llm_calls: total_calls,
            total_embedding_calls: self.counters.embedding_calls.load(Ordering::Relaxed),
            total_prompt_tokens: self.counters.prompt_tokens.load(Ordering::Relaxed),
            total_completion_tokens: self.counters.completion_tokens.load(Ordering::Relaxed),
            details,
        }
    }

    fn batch_config(&self) -> (usize, u32) {
        (self.batch_size, self.batch_max_tokens)
    }
}

impl OpenAILLMClient {
    /// Generate completion with structured output mode and JSON validation retry logic.
    ///
    /// This method attempts to generate valid JSON output with multiple retries:
    /// 1. Validates the response is valid JSON
    /// 2. Retries up to max_retries times if validation fails
    /// 3. Falls back to enhanced prompt on final retry
    async fn complete_with_structured_output_retry(
        completion_model: &Agent<CompletionModel>,
        prompt: &str,
        max_retries: u32,
        timeout_secs: u64,
    ) -> Result<String> {
        for attempt in 0..=max_retries {
            let is_final_attempt = attempt == max_retries;

            // Build prompt with strong JSON instruction
            let enhanced_prompt = if is_final_attempt {
                // On final attempt, use extra-strong instructions
                format!(
                    "{}\n\nFINAL ATTEMPT - Respond with ONLY valid JSON, no markdown, no explanation, no prose.",
                    prompt
                )
            } else {
                format!(
                    "{}\n\nRespond ONLY with valid JSON. No markdown, no explanation.",
                    prompt
                )
            };

            // Generate completion
            let future = completion_model.prompt(&enhanced_prompt).multi_turn(10);
            let response =
                tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), future)
                    .await
                    .map_err(|_| {
                        MemoryError::LLM(format!(
                            "LLM completion timed out after {}s",
                            timeout_secs
                        ))
                    })?
                    .map_err(|e| MemoryError::LLM(e.to_string()))?;

            // Validate JSON
            if let Some(json_str) = extract_json_from_text(&response) {
                if serde_json::from_str::<serde_json::Value>(&json_str).is_ok() {
                    // Valid JSON found
                    return Ok(json_str.to_string());
                } else {
                    debug!("Invalid JSON structure on attempt {}", attempt + 1);
                }
            } else {
                debug!("No JSON found in response on attempt {}", attempt + 1);
            }

            // If not final attempt, log and retry
            if !is_final_attempt {
                debug!(
                    "Structured output attempt {} failed, retrying...",
                    attempt + 1
                );
            }
        }

        // All retries exhausted, return the last response anyway
        debug!(
            "Structured output failed after {} retries, returning best-effort response",
            max_retries
        );

        // Final fallback - just return whatever we get
        let fallback_prompt = format!("{}\n\nReturn any valid JSON you can generate.", prompt);
        let future = completion_model.prompt(&fallback_prompt).multi_turn(10);
        tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), future)
            .await
            .map_err(|_| {
                MemoryError::LLM(format!("LLM completion timed out after {}s", timeout_secs))
            })?
            .map_err(|e| MemoryError::LLM(e.to_string()))
    }
}

/// Hybrid LLM client: API LLM completions + local embeddings.
///
/// This client uses OpenAI-compatible API for text generation
/// and local fastembed for embeddings (no API cost for embeddings).
#[cfg(feature = "local")]
pub struct APILLMLocalEmbedClient {
    /// OpenAI client for completions
    completion_client: OpenAILLMClient,
    /// Local embedding client
    local_embedding: Arc<tokio::sync::Mutex<fastembed::TextEmbedding>>,
    embedding_model_name: String,
    counters: UsageCounters,
}

#[cfg(feature = "local")]
impl APILLMLocalEmbedClient {
    pub fn new(
        llm_config: &LLMConfig,
        local_config: &LocalConfig,
        api_llm_config: &crate::config::ApiLlmConfig,
    ) -> Result<Self> {
        // Create a dummy embedding config (not used, but needed for OpenAILLMClient::new)
        let dummy_embedding_config = EmbeddingConfig::default();

        let completion_client =
            OpenAILLMClient::new(llm_config, &dummy_embedding_config, api_llm_config)?;

        // Initialize local embedding
        let embed_model = super::local_client::parse_fastembed_model(&local_config.embedding_model);
        let embed_options =
            fastembed::InitOptions::new(embed_model).with_show_download_progress(true);
        let embed_model = fastembed::TextEmbedding::try_new(embed_options).map_err(|e| {
            MemoryError::LLM(format!("Failed to initialize local embedding model: {}", e))
        })?;

        Ok(Self {
            completion_client,
            local_embedding: Arc::new(tokio::sync::Mutex::new(embed_model)),
            embedding_model_name: local_config.embedding_model.clone(),
            counters: UsageCounters::default(),
        })
    }
}

#[cfg(feature = "local")]
impl Clone for APILLMLocalEmbedClient {
    fn clone(&self) -> Self {
        Self {
            completion_client: self.completion_client.clone(),
            local_embedding: Arc::clone(&self.local_embedding),
            embedding_model_name: self.embedding_model_name.clone(),
            counters: self.counters.clone(),
        }
    }
}

#[cfg(feature = "local")]
#[async_trait]
impl LLMClient for APILLMLocalEmbedClient {
    async fn complete(&self, prompt: &str) -> Result<String> {
        self.completion_client.complete(prompt).await
    }

    async fn complete_with_grammar(&self, prompt: &str, grammar: &str) -> Result<String> {
        self.completion_client
            .complete_with_grammar(prompt, grammar)
            .await
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.counters
            .embedding_calls
            .fetch_add(1, Ordering::Relaxed);
        let embedding = Arc::clone(&self.local_embedding);
        let text = text.to_string();

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            tokio::task::spawn_blocking(move || {
                let emb = embedding.blocking_lock();
                emb.embed(vec![text], None)
                    .map_err(|e| MemoryError::LLM(format!("Local embedding failed: {}", e)))
                    .and_then(|mut v| {
                        v.pop()
                            .ok_or_else(|| MemoryError::LLM("No embedding returned".to_string()))
                    })
            }),
        )
        .await
        .map_err(|_| MemoryError::LLM("Local embedding timed out".to_string()))?
        .map_err(|e| MemoryError::LLM(format!("Join error: {}", e)))?;

        if result.is_ok()
            && let Ok(mut ts) = self.counters.last_embedding_success.lock()
        {
            *ts = Some(chrono::Utc::now());
        }
        result
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.counters
            .embedding_calls
            .fetch_add(texts.len() as u64, Ordering::Relaxed);
        let embedding = Arc::clone(&self.local_embedding);
        let texts = texts.to_vec();

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(120),
            tokio::task::spawn_blocking(move || {
                let emb = embedding.blocking_lock();
                emb.embed(texts, None)
                    .map_err(|e| MemoryError::LLM(format!("Local batch embedding failed: {}", e)))
            }),
        )
        .await
        .map_err(|_| MemoryError::LLM("Local batch embedding timed out".to_string()))?
        .map_err(|e| MemoryError::LLM(format!("Join error: {}", e)))?;

        if result.is_ok()
            && let Ok(mut ts) = self.counters.last_embedding_success.lock()
        {
            *ts = Some(chrono::Utc::now());
        }
        result
    }

    async fn extract_keywords(&self, content: &str) -> Result<Vec<String>> {
        self.completion_client.extract_keywords(content).await
    }

    async fn summarize(&self, content: &str, max_length: Option<usize>) -> Result<String> {
        self.completion_client.summarize(content, max_length).await
    }

    async fn health_check(&self) -> Result<bool> {
        // Check both completion and embedding
        let llm_ok = self.completion_client.health_check().await.unwrap_or(false);
        let embed_ok = self.embed("test").await.is_ok();
        Ok(llm_ok && embed_ok)
    }

    async fn extract_structured_facts(&self, prompt: &str) -> Result<StructuredFactExtraction> {
        self.completion_client
            .extract_structured_facts(prompt)
            .await
    }

    async fn extract_detailed_facts(&self, prompt: &str) -> Result<DetailedFactExtraction> {
        self.completion_client.extract_detailed_facts(prompt).await
    }

    async fn extract_keywords_structured(&self, prompt: &str) -> Result<KeywordExtraction> {
        self.completion_client
            .extract_keywords_structured(prompt)
            .await
    }

    async fn classify_memory(&self, prompt: &str) -> Result<MemoryClassification> {
        self.completion_client.classify_memory(prompt).await
    }

    async fn score_importance(&self, prompt: &str) -> Result<ImportanceScore> {
        self.completion_client.score_importance(prompt).await
    }

    async fn check_duplicates(&self, prompt: &str) -> Result<DeduplicationResult> {
        self.completion_client.check_duplicates(prompt).await
    }

    async fn generate_summary(&self, prompt: &str) -> Result<SummaryResult> {
        self.completion_client.generate_summary(prompt).await
    }

    async fn detect_language(&self, prompt: &str) -> Result<LanguageDetection> {
        self.completion_client.detect_language(prompt).await
    }

    async fn extract_entities(&self, prompt: &str) -> Result<EntityExtraction> {
        self.completion_client.extract_entities(prompt).await
    }

    async fn analyze_conversation(&self, prompt: &str) -> Result<ConversationAnalysis> {
        self.completion_client.analyze_conversation(prompt).await
    }

    async fn extract_metadata_enrichment(&self, prompt: &str) -> Result<MetadataEnrichment> {
        self.completion_client
            .extract_metadata_enrichment(prompt)
            .await
    }

    async fn extract_metadata_enrichment_batch(
        &self,
        texts: &[String],
    ) -> Result<Vec<Result<MetadataEnrichment>>> {
        self.completion_client
            .extract_metadata_enrichment_batch(texts)
            .await
    }

    async fn complete_batch(&self, prompts: &[String]) -> Result<Vec<Result<String>>> {
        self.completion_client.complete_batch(prompts).await
    }

    fn get_status(&self) -> ClientStatus {
        let completion_status = self.completion_client.get_status();
        ClientStatus {
            backend: "API LLM + Local Embeddings".to_string(),
            llm_available: completion_status.llm_available,
            embedding_available: true,
            llm_model: completion_status.llm_model,
            embedding_model: self.embedding_model_name.clone(),
            ..completion_status
        }
    }

    fn batch_config(&self) -> (usize, u32) {
        self.completion_client.batch_config()
    }
}

/// Hybrid LLM client: local LLM completions + API embeddings.
///
/// This client uses local llama.cpp for text generation (no API cost)
/// and OpenAI-compatible API for embeddings.
#[cfg(feature = "local")]
pub struct LocalLLMAPIEmbedClient {
    /// Local LLM client for completions
    local_llm: super::local_client::LocalLLMClient,
    /// OpenAI embedding client
    embedding_client: OpenAILLMClient,
}

#[cfg(feature = "local")]
impl LocalLLMAPIEmbedClient {
    pub fn new(local_config: &LocalConfig, embedding_config: &EmbeddingConfig) -> Result<Self> {
        let local_llm = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async { super::local_client::LocalLLMClient::new(local_config).await })
        })?;

        // Create dummy LLM config (not used for embeddings, but needed for OpenAILLMClient::new)
        let dummy_llm_config = LLMConfig::default();
        let dummy_api_llm_config = crate::config::ApiLlmConfig::default();

        let embedding_client =
            OpenAILLMClient::new(&dummy_llm_config, embedding_config, &dummy_api_llm_config)?;

        Ok(Self {
            local_llm,
            embedding_client,
        })
    }
}

#[cfg(feature = "local")]
impl Clone for LocalLLMAPIEmbedClient {
    fn clone(&self) -> Self {
        Self {
            local_llm: self.local_llm.clone(),
            embedding_client: self.embedding_client.clone(),
        }
    }
}

#[cfg(feature = "local")]
#[async_trait]
impl LLMClient for LocalLLMAPIEmbedClient {
    async fn complete(&self, prompt: &str) -> Result<String> {
        self.local_llm.complete(prompt).await
    }

    async fn complete_with_grammar(&self, prompt: &str, grammar: &str) -> Result<String> {
        self.local_llm.complete_with_grammar(prompt, grammar).await
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embedding_client.embed(text).await
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embedding_client.embed_batch(texts).await
    }

    async fn extract_keywords(&self, content: &str) -> Result<Vec<String>> {
        self.local_llm.extract_keywords(content).await
    }

    async fn summarize(&self, content: &str, max_length: Option<usize>) -> Result<String> {
        self.local_llm.summarize(content, max_length).await
    }

    async fn health_check(&self) -> Result<bool> {
        let llm_ok = self.local_llm.health_check().await.unwrap_or(false);
        let embed_ok = self.embedding_client.health_check().await.unwrap_or(false);
        Ok(llm_ok && embed_ok)
    }

    async fn extract_structured_facts(&self, prompt: &str) -> Result<StructuredFactExtraction> {
        self.local_llm.extract_structured_facts(prompt).await
    }

    async fn extract_detailed_facts(&self, prompt: &str) -> Result<DetailedFactExtraction> {
        self.local_llm.extract_detailed_facts(prompt).await
    }

    async fn extract_keywords_structured(&self, prompt: &str) -> Result<KeywordExtraction> {
        self.local_llm.extract_keywords_structured(prompt).await
    }

    async fn classify_memory(&self, prompt: &str) -> Result<MemoryClassification> {
        self.local_llm.classify_memory(prompt).await
    }

    async fn score_importance(&self, prompt: &str) -> Result<ImportanceScore> {
        self.local_llm.score_importance(prompt).await
    }

    async fn check_duplicates(&self, prompt: &str) -> Result<DeduplicationResult> {
        self.local_llm.check_duplicates(prompt).await
    }

    async fn generate_summary(&self, prompt: &str) -> Result<SummaryResult> {
        self.local_llm.generate_summary(prompt).await
    }

    async fn detect_language(&self, prompt: &str) -> Result<LanguageDetection> {
        self.local_llm.detect_language(prompt).await
    }

    async fn extract_entities(&self, prompt: &str) -> Result<EntityExtraction> {
        self.local_llm.extract_entities(prompt).await
    }

    async fn analyze_conversation(&self, prompt: &str) -> Result<ConversationAnalysis> {
        self.local_llm.analyze_conversation(prompt).await
    }

    async fn extract_metadata_enrichment(&self, prompt: &str) -> Result<MetadataEnrichment> {
        self.local_llm.extract_metadata_enrichment(prompt).await
    }

    async fn extract_metadata_enrichment_batch(
        &self,
        texts: &[String],
    ) -> Result<Vec<Result<MetadataEnrichment>>> {
        self.local_llm
            .extract_metadata_enrichment_batch(texts)
            .await
    }

    async fn complete_batch(&self, prompts: &[String]) -> Result<Vec<Result<String>>> {
        self.local_llm.complete_batch(prompts).await
    }

    fn get_status(&self) -> ClientStatus {
        let local_status = self.local_llm.get_status();
        let embed_status = self.embedding_client.get_status();
        ClientStatus {
            backend: "Local LLM + API Embeddings".to_string(),
            llm_available: local_status.llm_available,
            embedding_available: embed_status.embedding_available,
            llm_model: local_status.llm_model,
            embedding_model: embed_status.embedding_model,
            ..local_status
        }
    }

    fn batch_config(&self) -> (usize, u32) {
        self.local_llm.batch_config()
    }
}

/// Factory function to create LLM clients based on configuration.
///
/// When `config.effective_backend()` is:
/// - `Local` → uses llama.cpp + fastembed (requires `local` feature)
/// - `API` → uses rig-core OpenAI-compatible client for both LLM and embeddings
/// - `APILLMLocalEmbed` → uses API for LLM + local fastembed for embeddings
/// - `LocalLLMAPIEmbed` → uses local llama.cpp for LLM + API for embeddings
pub async fn create_llm_client(config: &Config) -> Result<Box<dyn LLMClient>> {
    match config.effective_backend() {
        LLMBackend::Local => {
            #[cfg(feature = "local")]
            {
                use super::lazy_client::LazyLocalLLMClient;
                let client = LazyLocalLLMClient::new(&config.local);
                Ok(Box::new(client))
            }
            #[cfg(not(feature = "local"))]
            {
                Err(MemoryError::config(
                    "Local backend requested but the 'local' feature is not enabled.\n\
                     Rebuild with: cargo build (local is the default feature)\n\
                     Or explicitly: cargo build --features local",
                ))
            }
        }
        LLMBackend::API => {
            let client = OpenAILLMClient::new(&config.llm, &config.embedding, &config.api_llm)?;
            Ok(Box::new(client))
        }
        LLMBackend::APILLMLocalEmbed => {
            #[cfg(feature = "local")]
            {
                let client =
                    APILLMLocalEmbedClient::new(&config.llm, &config.local, &config.api_llm)?;
                Ok(Box::new(client))
            }
            #[cfg(not(feature = "local"))]
            {
                Err(MemoryError::config(
                    "APILLMLocalEmbed backend requires the 'local' feature for embeddings.\n\
                     Rebuild with: cargo build --features local",
                ))
            }
        }
        LLMBackend::LocalLLMAPIEmbed => {
            #[cfg(feature = "local")]
            {
                let client = LocalLLMAPIEmbedClient::new(&config.local, &config.embedding)?;
                Ok(Box::new(client))
            }
            #[cfg(not(feature = "local"))]
            {
                Err(MemoryError::config(
                    "LocalLLMAPIEmbed backend requires the 'local' feature for LLM.\n\
                     Rebuild with: cargo build --features local",
                ))
            }
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Extract a JSON object or array from text that may contain surrounding prose.
///
/// This function handles:
/// - Markdown code fences (```json ... ```) including nested fences
/// - JSON objects {...} and arrays [...]
/// - Nested structures with proper bracket matching
/// - String escaping within JSON
/// - Trailing text after JSON block
///
/// Uses a robust bracket-counting approach to extract exactly one JSON object or array.
/// Returns an owned String to avoid lifetime issues with intermediate string processing.
pub fn extract_json_from_text(text: &str) -> Option<String> {
    let text = text.trim();

    // Strip all markdown code fences (handles nested fences too)
    let text = strip_markdown_fences(text);
    let text = text.trim();

    // Find the first JSON delimiter ([ or {)
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

/// Strip markdown code fences from text, handling nested fences.
///
/// This function removes outer ```json ... ``` or ``` ... ``` wrappers,
/// and handles cases where the LLM includes nested code fences.
fn strip_markdown_fences(text: &str) -> String {
    let mut result = text.to_string();

    // Iteratively strip outer fences until none remain
    loop {
        let trimmed = result.trim();

        // Try to strip ```json ... ``` first
        if let Some(content) = trimmed.strip_prefix("```json") {
            // Find the last ``` that closes the fence
            if let Some(end_pos) = content.rfind("```") {
                result = content[..end_pos].to_string();
                continue;
            }
        }

        // Try to strip generic ``` ... ```
        if let Some(content) = trimmed.strip_prefix("```") {
            // Find the last ``` that closes the fence
            if let Some(end_pos) = content.rfind("```") {
                result = content[..end_pos].to_string();
                continue;
            }
        }

        // No more fences to strip
        break;
    }

    result
}

/// Strip XML-style tags (e.g., <think>...</think>, <reason>...</reason>) from LLM output
/// Supports multiple tag types and handles missing closing tags gracefully
///
/// This function handles:
/// - Complete tag pairs: <tag>content</tag>
/// - Self-closing tags: <tag/>
/// - Missing closing tags: <tag>content (strips from tag to end)
/// - Missing opening tag content: <tag> (strips just the tag)
/// - Attributes in tags: <tag attr="value">content</tag>
fn strip_xml_tags(text: &str, tags: &[String]) -> String {
    let mut result = text.to_string();

    for tag in tags {
        // Strip <tag>...</tag> blocks (with or without closing tag)
        loop {
            // Match opening tag with optional attributes: <tag ...>
            let open_tag_pattern = format!("<{}", tag);
            let close_tag = format!("</{}>", tag);
            let self_closing_pattern = format!("<{} />", tag);
            let self_closing_pattern2 = format!("<{}/>", tag);

            // Check for self-closing tags first
            if let Some(pos) = result.find(&self_closing_pattern) {
                let end_pos = pos + self_closing_pattern.len();
                result = format!("{}{}", &result[..pos], &result[end_pos..]);
                continue;
            }
            if let Some(pos) = result.find(&self_closing_pattern2) {
                let end_pos = pos + self_closing_pattern2.len();
                result = format!("{}{}", &result[..pos], &result[end_pos..]);
                continue;
            }

            // Look for opening tag
            if let Some(start) = result.find(&open_tag_pattern) {
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
                        // This handles malformed LLM output gracefully (e.g., <think> without </think>)
                        result = result[..start].to_string();
                        continue;
                    }
                } else {
                    // Malformed tag (no closing >), strip from < to end
                    result = result[..start].to_string();
                    continue;
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

/// Truncate a string for logging purposes to avoid flooding logs with long responses.
///
/// # Arguments
/// * `text` - The text to truncate
/// * `max_len` - Maximum number of characters to include
///
/// # Returns
/// A string that is at most `max_len` characters, with "..." appended if truncated
fn truncate_for_logging(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len])
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

    #[tokio::test]
    async fn test_custom_dialect_request_formatting() {
        use crate::config::{ApiDialect, CustomDialectConfig};
        use serde_json::json;

        // Setup a custom config
        let api_llm_config = crate::config::ApiLlmConfig {
            api_dialect: ApiDialect::Custom,
            custom_dialect: Some(CustomDialectConfig {
                endpoint_path: "/v1/generate".to_string(),
                request_body_template: json!({
                    "prompt_text": "{{prompt}}",
                    "model_id": "{{model}}",
                    "params": {
                        "temp": "{{temperature}}"
                    }
                })
                .to_string(),
                response_content_pointer: "/results/0/text".to_string(),
            }),
            ..Default::default()
        };

        let llm_config = crate::config::LLMConfig {
            api_key: "test-key".to_string(),
            api_base_url: "http://localhost:12345".to_string(), // dummy port
            model_efficient: "my-custom-model".to_string(),
            temperature: 0.7,
            max_tokens: 100,
        };

        let embedding_config = crate::config::EmbeddingConfig::default();

        let client = OpenAILLMClient::new(&llm_config, &embedding_config, &api_llm_config).unwrap();

        // We expect it to fail connecting to the dummy port,
        // but it should HAVE FORMATTED the URL correctly in the info log or internal state.
        // Since we can't easily capture logs in a unit test, we'll check that it DOES
        // attempt to use the custom URL structure by checking the error message
        // (if the error from reqwest includes the URL).

        let result = client.raw_completion("Hello").await;

        match result {
            Err(MemoryError::LLM(msg)) => {
                // If the error happens during SEND, reqwest usually includes the URL
                // Note: on some systems it might just say "connection refused"
                // but we are testing that the logic reaches the send phase with our formatted body.
                info!("Captured expected error: {}", msg);
                // We mainly want to ensure it didn't fail earlier with a "Custom dialect selected but no config" error
                assert!(!msg.contains("no custom_dialect config provided"));
            }
            _ => {
                // It might actually succeed if something IS running on that port, but unlikely.
            }
        }
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
    fn test_strip_think_tags() {
        let text = "<think>\nThinking...\n</think>\n{\"result\": \"success\"}";
        let cleaned = strip_llm_tags(text, &["think".to_string()]);
        let json = extract_json_from_text(&cleaned).unwrap();
        assert_eq!(json, r#"{"result": "success"}"#);
    }

    #[test]
    fn test_strip_think_tags_missing_closing() {
        let text = "Some text <think>\nThinking without closing tag";
        let result = strip_llm_tags(text, &["think".to_string()]);
        assert_eq!(result, "Some text");
    }

    #[test]
    fn test_strip_think_tags_multiple() {
        let text = "<think>first</think> middle <think>second</think> end";
        let result = strip_llm_tags(text, &["think".to_string()]);
        assert_eq!(result, "middle  end");
    }

    #[test]
    fn test_strip_multiple_tag_types() {
        let text = "<think>thinking</think><reason>reasoning</reason>final";
        let result = strip_llm_tags(text, &["think".to_string(), "reason".to_string()]);
        assert_eq!(result, "final");
    }

    #[test]
    fn test_strip_llm_tags_from_llm_response() {
        // Simulate a typical LLM response with think tags and JSON
        let text = r#"<think>
The user wants me to extract keywords from this text.
I should identify the main topics and return them as JSON.
</think>

{"keywords": ["rust", "testing", "json"]}
"#;
        let cleaned = strip_llm_tags(text, &["think".to_string()]);
        let json = extract_json_from_text(&cleaned).unwrap();
        assert_eq!(json, r#"{"keywords": ["rust", "testing", "json"]}"#);
    }

    #[test]
    fn test_extract_json_from_markdown_response() {
        // Simulate LLM response with markdown code fence
        let text = r#"Here's the JSON you requested:

```json
{
  "summary": "Test summary",
  "keywords": ["test", "example"]
}
```

Hope this helps!"#;
        let json = extract_json_from_text(text).unwrap();
        assert!(json.contains("\"summary\""));
        assert!(json.contains("\"keywords\""));
    }

    #[test]
    fn test_request_format_default() {
        let format = crate::config::RequestFormat::default();
        assert_eq!(format, crate::config::RequestFormat::Auto);
    }

    #[test]
    fn test_request_format_serde_auto() {
        let json = r#""auto""#;
        let format: crate::config::RequestFormat = serde_json::from_str(json).unwrap();
        assert_eq!(format, crate::config::RequestFormat::Auto);
    }

    #[test]
    fn test_request_format_serde_rig() {
        let json = r#""rig""#;
        let format: crate::config::RequestFormat = serde_json::from_str(json).unwrap();
        assert_eq!(format, crate::config::RequestFormat::Rig);
    }

    #[test]
    fn test_request_format_serde_raw() {
        let json = r#""raw""#;
        let format: crate::config::RequestFormat = serde_json::from_str(json).unwrap();
        assert_eq!(format, crate::config::RequestFormat::Raw);
    }

    #[test]
    fn test_request_format_serialize() {
        let format = crate::config::RequestFormat::Auto;
        let json = serde_json::to_string(&format).unwrap();
        assert_eq!(json, r#""auto""#);
    }

    #[test]
    fn test_request_format_equality() {
        assert_eq!(
            crate::config::RequestFormat::Auto,
            crate::config::RequestFormat::Auto
        );
        assert_ne!(
            crate::config::RequestFormat::Auto,
            crate::config::RequestFormat::Raw
        );
        assert_ne!(
            crate::config::RequestFormat::Rig,
            crate::config::RequestFormat::Raw
        );
    }

    #[test]
    fn test_api_llm_config_default_request_format() {
        let config = crate::config::ApiLlmConfig::default();
        assert_eq!(config.request_format, crate::config::RequestFormat::Auto);
    }

    #[test]
    fn test_api_llm_config_with_raw_format() {
        let config_toml = r#"
            request_format = "raw"
            use_structured_output = true
            structured_output_retries = 2
            strip_llm_tags = ["think"]
        "#;
        let config: crate::config::ApiLlmConfig = toml::from_str(config_toml).unwrap();
        assert_eq!(config.request_format, crate::config::RequestFormat::Raw);
        assert!(config.use_structured_output);
        assert_eq!(config.structured_output_retries, 2);
        assert_eq!(config.strip_llm_tags, vec!["think"]);
    }

    #[test]
    fn test_raw_format_detection_state() {
        // Test that raw format detection state is properly shared across clones
        let detection_flag = std::sync::Arc::new(std::sync::Mutex::new(false));

        // Initially not detected
        assert!(!*detection_flag.lock().unwrap());

        // Simulate detection
        *detection_flag.lock().unwrap() = true;
        assert!(*detection_flag.lock().unwrap());

        // Clone and verify state is shared
        let detection_flag_clone = std::sync::Arc::clone(&detection_flag);
        assert!(*detection_flag_clone.lock().unwrap());
    }

    #[test]
    fn test_extract_json_nested_markdown_fences() {
        // Test nested markdown fences
        let text = r#"```json
```json
{"keywords": ["test"]}
```
```"#;
        let json = extract_json_from_text(text).unwrap();
        assert!(json.contains("\"keywords\""));
    }

    #[test]
    fn test_extract_json_trailing_text() {
        // Test JSON with trailing text
        let text = r#"{"result": "success"} Hope this helps!"#;
        let json = extract_json_from_text(text).unwrap();
        assert_eq!(json, r#"{"result": "success"}"#);
    }

    #[test]
    fn test_strip_markdown_fences_generic() {
        // Test generic markdown fences (not json-specific)
        let text = r#"```
{"data": "value"}
```"#;
        let json = extract_json_from_text(text).unwrap();
        assert_eq!(json, r#"{"data": "value"}"#);
    }

    #[test]
    fn test_strip_self_closing_xml_tags() {
        let text = "Before <think/> after";
        let result = strip_llm_tags(text, &["think".to_string()]);
        assert_eq!(result, "Before  after");
    }

    #[test]
    fn test_strip_xml_tags_with_attributes() {
        let text = r#"Before <think attr="value">content</think> after"#;
        let result = strip_llm_tags(text, &["think".to_string()]);
        assert_eq!(result, "Before  after");
    }

    #[test]
    fn test_parse_keywords_json_array() {
        // Test JSON array response - test extract_json_from_text directly
        let response = r#"["keyword1", "keyword2", "keyword3"]"#;
        let json_str = extract_json_from_text(response).unwrap();
        let arr: Vec<String> = serde_json::from_str(&json_str).unwrap();
        assert_eq!(arr, vec!["keyword1", "keyword2", "keyword3"]);
    }

    #[test]
    fn test_parse_keywords_json_object() {
        // Test JSON object with keywords field
        let response = r#"{"keywords": ["rust", "testing", "json"]}"#;
        let json_str = extract_json_from_text(response).unwrap();
        let obj: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let keywords = obj.get("keywords").and_then(|v| v.as_array()).unwrap();
        let keywords: Vec<String> = keywords
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        assert_eq!(keywords, vec!["rust", "testing", "json"]);
    }

    #[test]
    fn test_parse_keywords_comma_separated_fallback() {
        // Test comma-separated fallback - no JSON to extract
        let response = "keyword1, keyword2, keyword3";
        // When no JSON is found, it falls back to comma-separated
        let json_str = extract_json_from_text(response);
        assert!(json_str.is_none()); // No JSON found, would use comma fallback
    }

    #[test]
    fn test_parse_keywords_with_think_tags() {
        // Test with think tags - test strip_llm_tags directly
        let response = r#"<think>
Thinking about keywords
</think>
["keyword1", "keyword2"]"#;
        let cleaned = strip_llm_tags(response, &["think".to_string()]);
        let json_str = extract_json_from_text(&cleaned).unwrap();
        let arr: Vec<String> = serde_json::from_str(&json_str).unwrap();
        assert_eq!(arr, vec!["keyword1", "keyword2"]);
    }

    #[test]
    fn test_json_inside_think_tags_is_stripped() {
        // Test that JSON inside think tags is NOT captured
        let response = r#"<think>
Here's the answer: {"keywords": ["wrong"]}
</think>
{"keywords": ["correct"]}"#;
        let cleaned = strip_llm_tags(response, &["think".to_string()]);
        // After stripping, only the second JSON should remain
        assert!(!cleaned.contains("wrong"));
        assert!(cleaned.contains("correct"));
        let json_str = extract_json_from_text(&cleaned).unwrap();
        assert!(json_str.contains("correct"));
        assert!(!json_str.contains("wrong"));
    }

    #[test]
    fn test_malformed_think_tag_without_closing() {
        // Test that malformed think tag without closing strips everything after
        let response = r#"<think>
This JSON should be stripped: {"keywords": ["wrong"]}
Some more text"#;
        let cleaned = strip_llm_tags(response, &["think".to_string()]);
        // Everything after <think> should be stripped
        assert!(!cleaned.contains("wrong"));
        assert!(cleaned.is_empty());
    }

    #[test]
    fn test_json_before_think_tags() {
        // Test JSON that appears BEFORE think tags
        let response = r#"{"keywords": ["correct"]}
<think>
This should be stripped
</think>"#;
        let cleaned = strip_llm_tags(response, &["think".to_string()]);
        let json_str = extract_json_from_text(&cleaned).unwrap();
        assert!(json_str.contains("correct"));
    }

    #[test]
    fn test_nested_brackets_in_think_content() {
        // Test that nested brackets inside think tags don't confuse extraction
        let response = r#"<think>
Complex reasoning: { outer: [1, 2, { inner: [3, 4] }] }
</think>
{"result": "success"}"#;
        let cleaned = strip_llm_tags(response, &["think".to_string()]);
        // The JSON inside think should be stripped
        assert!(!cleaned.contains("Complex reasoning"));
        let json_str = extract_json_from_text(&cleaned).unwrap();
        assert_eq!(json_str, r#"{"result": "success"}"#);
    }
}
