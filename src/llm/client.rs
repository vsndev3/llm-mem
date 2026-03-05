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

    /// Return current status and usage statistics of this client.
    fn get_status(&self) -> ClientStatus;
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
            raw_format_detected: Arc::new(std::sync::Mutex::new(false)),
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
        cleaned
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Raw HTTP completion that bypasses rig-core's message formatting.
    ///
    /// This makes a direct HTTP request with plain string content instead of
    /// rig-core's complex message array format. Used for backends that reject
    /// the [{"type": "text", "text": "..."}] format with 422 errors.
    async fn raw_completion(&self, prompt: &str) -> Result<String> {
        use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};

        // Build request body with plain string content
        let request_body = serde_json::json!({
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

        let client = reqwest::Client::new();
        let url = format!("{}/chat/completions", self.api_base_url.trim_end_matches('/'));

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(self.timeout_secs),
            client
                .post(&url)
                .header(CONTENT_TYPE, "application/json")
                .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
                .json(&request_body)
                .send()
        )
        .await
        .map_err(|_| MemoryError::LLM(format!("Raw HTTP completion timed out after {}s", self.timeout_secs)))?
        .map_err(|e| MemoryError::LLM(format!("Raw HTTP request failed: {}", e)))?;

        let status = response.status();
        let response_text = response.text().await
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

        // Parse response
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&response_text)
            && let Some(content) = json["choices"][0]["message"]["content"].as_str()
        {
            return Ok(content.to_string());
        }

        Err(MemoryError::LLM(format!("Invalid response format: {}", response_text)))
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
    async fn complete_and_parse<T, F>(&self, prompt: &str, _max_tokens: u64, fallback: F) -> Result<T>
    where
        T: serde::de::DeserializeOwned + Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
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
                let future = self.completion_model.prompt(&enhanced_prompt).multi_turn(10);
                tokio::time::timeout(std::time::Duration::from_secs(self.timeout_secs), future)
                    .await
                    .map_err(|_| {
                        MemoryError::LLM(format!(
                            "LLM completion timed out after {}s",
                            self.timeout_secs
                        ))
                    })?
                    .map_err(|e| MemoryError::LLM(e.to_string()))
            }.await;

            match rig_result {
                Ok(resp) => resp,
                Err(e) => {
                    // Check if this is a 422 error and we're in auto mode
                    if self.request_format == crate::config::RequestFormat::Auto &&
                       e.to_string().contains("422") {
                        // Mark that we need to use raw format for future requests
                        *self.raw_format_detected.lock().unwrap() = true;
                        tracing::info!("Detected 422 error from backend, switching to raw HTTP format for future requests");

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
            match serde_json::from_str::<T>(json_str) {
                Ok(parsed) => return Ok(parsed),
                Err(e) => {
                    debug!("JSON parse failed: {}, using fallback", e);
                }
            }
        } else {
            debug!("No JSON found in response, using fallback");
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
            raw_format_detected: Arc::clone(&self.raw_format_detected),
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
            let rig_result = tokio::time::timeout(std::time::Duration::from_secs(self.timeout_secs), future)
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
                    if self.request_format == crate::config::RequestFormat::Auto &&
                       e.to_string().contains("422") {
                        // Mark that we need to use raw format for future requests
                        *self.raw_format_detected.lock().unwrap() = true;
                        tracing::info!("Detected 422 error from backend in complete(), switching to raw HTTP format for future requests");

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
        self.complete_and_parse(
            prompt,
            500,
            || StructuredFactExtraction { facts: vec![] },
        )
        .await
    }

    async fn extract_detailed_facts(&self, prompt: &str) -> Result<DetailedFactExtraction> {
        self.complete_and_parse(
            prompt,
            1000,
            || DetailedFactExtraction { facts: vec![] },
        )
        .await
    }

    async fn extract_keywords_structured(&self, prompt: &str) -> Result<KeywordExtraction> {
        self.complete_and_parse(
            prompt,
            500,
            || KeywordExtraction { keywords: vec![] },
        )
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
        self.complete_and_parse(
            prompt,
            500,
            || ImportanceScore { score: 0.5, reasoning: "Fallback due to JSON parse failure".to_string() },
        )
        .await
    }

    async fn check_duplicates(&self, prompt: &str) -> Result<DeduplicationResult> {
        self.complete_and_parse(
            prompt,
            500,
            || DeduplicationResult { is_duplicate: false, similarity_score: 0.0, original_memory_id: None },
        )
        .await
    }

    async fn generate_summary(&self, prompt: &str) -> Result<SummaryResult> {
        self.complete_and_parse(
            prompt,
            1000,
            || SummaryResult { summary: "".to_string(), key_points: vec![] },
        )
        .await
    }

    async fn detect_language(&self, prompt: &str) -> Result<LanguageDetection> {
        self.complete_and_parse(
            prompt,
            200,
            || LanguageDetection { language: "unknown".to_string(), confidence: 0.0 },
        )
        .await
    }

    async fn extract_entities(&self, prompt: &str) -> Result<EntityExtraction> {
        self.complete_and_parse(
            prompt,
            1000,
            || EntityExtraction { entities: vec![] },
        )
        .await
    }

    async fn analyze_conversation(&self, prompt: &str) -> Result<ConversationAnalysis> {
        self.complete_and_parse(
            prompt,
            1500,
            || ConversationAnalysis {
                topics: vec![],
                sentiment: "neutral".to_string(),
                user_intent: "unknown".to_string(),
                key_information: vec![],
            },
        )
        .await
    }

    async fn extract_metadata_enrichment(&self, prompt: &str) -> Result<MetadataEnrichment> {
        self.complete_and_parse(
            prompt,
            1000,
            || MetadataEnrichment { summary: "".to_string(), keywords: vec![] },
        )
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
                if serde_json::from_str::<serde_json::Value>(json_str).is_ok() {
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
        let fallback_prompt = format!(
            "{}\n\nReturn any valid JSON you can generate.",
            prompt
        );
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

        let completion_client = OpenAILLMClient::new(llm_config, &dummy_embedding_config, api_llm_config)?;

        // Initialize local embedding
        let embed_model = super::local_client::parse_fastembed_model(&local_config.embedding_model);
        let embed_options = fastembed::InitOptions::new(embed_model)
            .with_show_download_progress(true);
        let embed_model = fastembed::TextEmbedding::try_new(embed_options)
            .map_err(|e| MemoryError::LLM(format!("Failed to initialize local embedding model: {}", e)))?;

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
        self.completion_client.complete_with_grammar(prompt, grammar).await
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.counters.embedding_calls.fetch_add(1, Ordering::Relaxed);
        let embedding = Arc::clone(&self.local_embedding);
        let text = text.to_string();

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            tokio::task::spawn_blocking(move || {
                let emb = embedding
                    .blocking_lock();
                emb.embed(vec![text], None)
                    .map_err(|e| MemoryError::LLM(format!("Local embedding failed: {}", e)))
                    .and_then(|mut v| v.pop().ok_or_else(|| MemoryError::LLM("No embedding returned".to_string())))
            }),
        )
        .await
        .map_err(|_| MemoryError::LLM("Local embedding timed out".to_string()))?
        .map_err(|e| MemoryError::LLM(format!("Join error: {}", e)))?;

        if result.is_ok() {
            if let Ok(mut ts) = self.counters.last_embedding_success.lock() {
                *ts = Some(chrono::Utc::now());
            }
        }
        result
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.counters.embedding_calls.fetch_add(texts.len() as u64, Ordering::Relaxed);
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

        if result.is_ok() {
            if let Ok(mut ts) = self.counters.last_embedding_success.lock() {
                *ts = Some(chrono::Utc::now());
            }
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
        self.completion_client.extract_structured_facts(prompt).await
    }

    async fn extract_detailed_facts(&self, prompt: &str) -> Result<DetailedFactExtraction> {
        self.completion_client.extract_detailed_facts(prompt).await
    }

    async fn extract_keywords_structured(&self, prompt: &str) -> Result<KeywordExtraction> {
        self.completion_client.extract_keywords_structured(prompt).await
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
        self.completion_client.extract_metadata_enrichment(prompt).await
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
    pub fn new(
        local_config: &LocalConfig,
        embedding_config: &EmbeddingConfig,
    ) -> Result<Self> {
        let local_llm = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                super::local_client::LocalLLMClient::new(local_config).await
            })
        })?;

        // Create dummy LLM config (not used for embeddings, but needed for OpenAILLMClient::new)
        let dummy_llm_config = LLMConfig::default();
        let dummy_api_llm_config = crate::config::ApiLlmConfig::default();

        let embedding_client = OpenAILLMClient::new(&dummy_llm_config, embedding_config, &dummy_api_llm_config)?;

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
                let client = APILLMLocalEmbedClient::new(&config.llm, &config.local, &config.api_llm)?;
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
/// - Markdown code fences (```json ... ```)
/// - JSON objects {...} and arrays [...]
/// - Nested structures with proper bracket matching
/// - String escaping within JSON
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
        assert_eq!(crate::config::RequestFormat::Auto, crate::config::RequestFormat::Auto);
        assert_ne!(crate::config::RequestFormat::Auto, crate::config::RequestFormat::Raw);
        assert_ne!(crate::config::RequestFormat::Rig, crate::config::RequestFormat::Raw);
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
}
