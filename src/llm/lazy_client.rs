use async_trait::async_trait;
use tokio::sync::watch;
use tracing::{error, info};

use super::client::LLMClient;
use super::extractor_types::*;
use super::local_client::LocalLLMClient;
use crate::config::LlmConfig;
use crate::error::{MemoryError, Result};

#[derive(Clone)]
enum ClientState {
    Initializing,
    Ready(Box<LocalLLMClient>),
    Failed(String),
}

#[derive(Clone)]
pub struct LazyLocalLLMClient {
    state_rx: watch::Receiver<ClientState>,
    // Config kept for status reporting
    config: LlmConfig,
    embedding_model_name: String,
}

impl LazyLocalLLMClient {
    pub fn new(config: &LlmConfig, embedding_model: &str) -> Self {
        let (state_tx, state_rx) = watch::channel(ClientState::Initializing);

        let client = Self {
            state_rx,
            config: config.clone(),
            embedding_model_name: embedding_model.to_string(),
        };

        let config_clone = config.clone();
        let embedding_model_clone = embedding_model.to_string();
        tokio::spawn(async move {
            info!("Starting background initialization of LocalLLMClient...");
            match LocalLLMClient::new(&config_clone, &embedding_model_clone).await {
                Ok(local_client) => {
                    info!("LocalLLMClient initialized successfully in background.");
                    let _ = state_tx.send(ClientState::Ready(Box::new(local_client)));
                }
                Err(e) => {
                    error!("Failed to initialize LocalLLMClient: {}", e);
                    let _ = state_tx.send(ClientState::Failed(e.to_string()));
                }
            }
        });

        client
    }

    /// Wait for initialization to complete and return the client or error.
    async fn get_client(&self) -> Result<LocalLLMClient> {
        let mut rx = self.state_rx.clone();
        loop {
            // Check current state
            {
                let state = rx.borrow();
                match &*state {
                    ClientState::Ready(client) => return Ok((**client).clone()),
                    ClientState::Failed(err) => {
                        return Err(MemoryError::LLM(format!(
                            "LLM initialization failed: {}",
                            err
                        )));
                    }
                    ClientState::Initializing => {
                        // continue to wait
                    }
                }
            }

            // Wait for change
            if rx.changed().await.is_err() {
                return Err(MemoryError::LLM(
                    "LLM initialization task panicked or was cancelled".to_string(),
                ));
            }
        }
    }
}

#[async_trait]
impl LLMClient for LazyLocalLLMClient {
    async fn complete(&self, prompt: &str) -> Result<String> {
        self.get_client().await?.complete(prompt).await
    }

    async fn complete_with_grammar(&self, prompt: &str, grammar: &str) -> Result<String> {
        self.get_client()
            .await?
            .complete_with_grammar(prompt, grammar)
            .await
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.get_client().await?.embed(text).await
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.get_client().await?.embed_batch(texts).await
    }

    async fn extract_keywords(&self, content: &str) -> Result<Vec<String>> {
        self.get_client().await?.extract_keywords(content).await
    }

    async fn summarize(&self, content: &str, max_length: Option<usize>) -> Result<String> {
        self.get_client()
            .await?
            .summarize(content, max_length)
            .await
    }

    async fn health_check(&self) -> Result<bool> {
        let client = {
            let state = self.state_rx.borrow();
            match &*state {
                ClientState::Ready(client) => Some((**client).clone()),
                ClientState::Initializing => None,
                ClientState::Failed(_) => return Ok(false),
            }
        };

        if let Some(client) = client {
            client.health_check().await
        } else {
            Ok(true) // Report healthy while loading
        }
    }

    async fn extract_structured_facts(&self, prompt: &str) -> Result<StructuredFactExtraction> {
        self.get_client()
            .await?
            .extract_structured_facts(prompt)
            .await
    }

    async fn extract_detailed_facts(&self, prompt: &str) -> Result<DetailedFactExtraction> {
        self.get_client()
            .await?
            .extract_detailed_facts(prompt)
            .await
    }

    async fn extract_keywords_structured(&self, prompt: &str) -> Result<KeywordExtraction> {
        self.get_client()
            .await?
            .extract_keywords_structured(prompt)
            .await
    }

    async fn classify_memory(&self, prompt: &str) -> Result<MemoryClassification> {
        self.get_client().await?.classify_memory(prompt).await
    }

    async fn score_importance(&self, prompt: &str) -> Result<ImportanceScore> {
        self.get_client().await?.score_importance(prompt).await
    }

    async fn check_duplicates(&self, prompt: &str) -> Result<DeduplicationResult> {
        self.get_client().await?.check_duplicates(prompt).await
    }

    async fn generate_summary(&self, prompt: &str) -> Result<SummaryResult> {
        self.get_client().await?.generate_summary(prompt).await
    }

    async fn detect_language(&self, prompt: &str) -> Result<LanguageDetection> {
        self.get_client().await?.detect_language(prompt).await
    }

    async fn extract_entities(&self, prompt: &str) -> Result<EntityExtraction> {
        self.get_client().await?.extract_entities(prompt).await
    }

    async fn analyze_conversation(&self, prompt: &str) -> Result<ConversationAnalysis> {
        self.get_client().await?.analyze_conversation(prompt).await
    }

    async fn extract_metadata_enrichment(&self, prompt: &str) -> Result<MetadataEnrichment> {
        self.get_client()
            .await?
            .extract_metadata_enrichment(prompt)
            .await
    }

    async fn extract_metadata_enrichment_batch(
        &self,
        texts: &[String],
    ) -> Result<Vec<Result<MetadataEnrichment>>> {
        self.get_client()
            .await?
            .extract_metadata_enrichment_batch(texts)
            .await
    }

    async fn complete_batch(&self, prompts: &[String]) -> Result<Vec<Result<String>>> {
        self.get_client().await?.complete_batch(prompts).await
    }

    fn get_status(&self) -> ClientStatus {
        let state = self.state_rx.borrow();
        match &*state {
            ClientState::Ready(client) => (**client).get_status(),
            ClientState::Initializing => ClientStatus {
                backend: "Local (Initializing)".to_string(),
                llm_model: self.config.model_file.clone(),
                ..Default::default()
            },
            ClientState::Failed(e) => ClientStatus {
                backend: format!("Local (Failed: {})", e),
                llm_model: self.config.model_file.clone(),
                ..Default::default()
            },
        }
    }

    fn batch_config(&self) -> (usize, u32) {
        let state = self.state_rx.borrow();
        match &*state {
            ClientState::Ready(client) => (**client).batch_config(),
            _ => (self.config.batch_size, self.config.batch_max_tokens),
        }
    }
}
