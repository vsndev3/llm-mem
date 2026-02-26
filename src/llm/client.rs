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
    counters: UsageCounters,
    timeout_secs: u64,
}

impl OpenAILLMClient {
    pub fn new(llm_config: &LLMConfig, embedding_config: &EmbeddingConfig) -> Result<Self> {
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
            counters: UsageCounters::default(),
            timeout_secs: embedding_config.timeout_secs,
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
        response
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
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
            counters: self.counters.clone(),
            timeout_secs: self.timeout_secs,
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

        let future = self.completion_model.prompt(prompt).multi_turn(10);

        let response =
            tokio::time::timeout(std::time::Duration::from_secs(self.timeout_secs), future)
                .await
                .map_err(|_| {
                    MemoryError::LLM(format!(
                        "LLM completion timed out after {}s",
                        self.timeout_secs
                    ))
                })?
                .map_err(|e| {
                    if let Ok(mut last) = self.counters.last_error.lock() {
                        *last = Some(e.to_string());
                    }
                    MemoryError::LLM(e.to_string())
                })?;

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
            Ok(result) => Ok(result.summary.trim().to_string()),
            Err(e) => {
                debug!("Rig extractor failed, falling back: {}", e);
                let summary = self.complete(&prompt).await?;
                Ok(summary.trim().to_string())
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
        let extractor = self
            .client
            .extractor_completions_api::<StructuredFactExtraction>(&self.completion_model_name)
            .preamble(prompt)
            .build();
        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        extractor
            .extract("")
            .await
            .map_err(|e| MemoryError::LLM(e.to_string()))
    }

    async fn extract_detailed_facts(&self, prompt: &str) -> Result<DetailedFactExtraction> {
        let extractor = self
            .client
            .extractor_completions_api::<DetailedFactExtraction>(&self.completion_model_name)
            .preamble(prompt)
            .build();
        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        extractor
            .extract("")
            .await
            .map_err(|e| MemoryError::LLM(e.to_string()))
    }

    async fn extract_keywords_structured(&self, prompt: &str) -> Result<KeywordExtraction> {
        let extractor = self
            .client
            .extractor_completions_api::<KeywordExtraction>(&self.completion_model_name)
            .preamble(prompt)
            .max_tokens(500)
            .build();
        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        extractor
            .extract("")
            .await
            .map_err(|e| MemoryError::LLM(e.to_string()))
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
        let extractor = self
            .client
            .extractor_completions_api::<ImportanceScore>(&self.completion_model_name)
            .preamble(prompt)
            .max_tokens(500)
            .build();
        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        extractor
            .extract("")
            .await
            .map_err(|e| MemoryError::LLM(e.to_string()))
    }

    async fn check_duplicates(&self, prompt: &str) -> Result<DeduplicationResult> {
        let extractor = self
            .client
            .extractor_completions_api::<DeduplicationResult>(&self.completion_model_name)
            .preamble(prompt)
            .max_tokens(500)
            .build();
        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        extractor
            .extract("")
            .await
            .map_err(|e| MemoryError::LLM(e.to_string()))
    }

    async fn generate_summary(&self, prompt: &str) -> Result<SummaryResult> {
        let extractor = self
            .client
            .extractor_completions_api::<SummaryResult>(&self.completion_model_name)
            .preamble(prompt)
            .max_tokens(1000)
            .build();
        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        extractor
            .extract("")
            .await
            .map_err(|e| MemoryError::LLM(e.to_string()))
    }

    async fn detect_language(&self, prompt: &str) -> Result<LanguageDetection> {
        let extractor = self
            .client
            .extractor_completions_api::<LanguageDetection>(&self.completion_model_name)
            .preamble(prompt)
            .max_tokens(200)
            .build();
        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        extractor
            .extract("")
            .await
            .map_err(|e| MemoryError::LLM(e.to_string()))
    }

    async fn extract_entities(&self, prompt: &str) -> Result<EntityExtraction> {
        let extractor = self
            .client
            .extractor_completions_api::<EntityExtraction>(&self.completion_model_name)
            .preamble(prompt)
            .max_tokens(1000)
            .build();
        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        extractor
            .extract("")
            .await
            .map_err(|e| MemoryError::LLM(e.to_string()))
    }

    async fn analyze_conversation(&self, prompt: &str) -> Result<ConversationAnalysis> {
        let extractor = self
            .client
            .extractor_completions_api::<ConversationAnalysis>(&self.completion_model_name)
            .preamble(prompt)
            .max_tokens(1500)
            .build();
        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        extractor
            .extract("")
            .await
            .map_err(|e| MemoryError::LLM(e.to_string()))
    }

    async fn extract_metadata_enrichment(&self, prompt: &str) -> Result<MetadataEnrichment> {
        let extractor = self
            .client
            .extractor_completions_api::<MetadataEnrichment>(&self.completion_model_name)
            .preamble(prompt)
            .max_tokens(1000)
            .build();
        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        extractor
            .extract("")
            .await
            .map_err(|e| MemoryError::LLM(e.to_string()))
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
            backend: "openai".to_string(),
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

/// Factory function to create LLM clients based on configuration.
///
/// When `config.effective_backend()` is:
/// - `Local` → uses llama.cpp + fastembed (requires `local` feature)
/// - `OpenAI` → uses rig-core OpenAI-compatible client
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
        LLMBackend::OpenAI => {
            let client = OpenAILLMClient::new(&config.llm, &config.embedding)?;
            Ok(Box::new(client))
        }
    }
}
