use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;

use crate::error::Result;
use crate::llm::client::LLMClient;
use crate::llm::extractor_types::*;
use crate::memory::metrics::{LlmBackendType, LlmOperationType, MetricsSink};

/// Decorator that wraps any `LLMClient` and records metrics for every call.
#[derive(Clone)]
pub struct MetricsLLMClient {
    inner: Box<dyn LLMClient>,
    metrics: Arc<dyn MetricsSink>,
    backend_type: LlmBackendType,
}

impl MetricsLLMClient {
    pub fn new(
        inner: Box<dyn LLMClient>,
        metrics: Arc<dyn MetricsSink>,
        backend_type: LlmBackendType,
    ) -> Self {
        Self { inner, metrics, backend_type }
    }
}

#[async_trait]
impl LLMClient for MetricsLLMClient {
    async fn complete(&self, prompt: &str) -> Result<String> {
        let start = Instant::now();
        let result = self.inner.complete(prompt).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (prompt.len() / 4) as u64;
        let completion_tokens = if success {
            (result.as_ref().unwrap().len() / 4) as u64
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::Completion,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn complete_with_grammar(&self, prompt: &str, grammar: &str) -> Result<String> {
        let start = Instant::now();
        let result = self.inner.complete_with_grammar(prompt, grammar).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (prompt.len() / 4) as u64;
        let completion_tokens = if success {
            (result.as_ref().unwrap().len() / 4) as u64
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::CompletionWithGrammar,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let start = Instant::now();
        let result = self.inner.embed(text).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        self.metrics.record_embedding_request(self.backend_type, duration, success, 1);
        result
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let start = Instant::now();
        let result = self.inner.embed_batch(texts).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        self.metrics.record_embedding_request(self.backend_type, duration, success, texts.len());
        result
    }

    async fn extract_keywords(&self, content: &str) -> Result<Vec<String>> {
        let start = Instant::now();
        let result = self.inner.extract_keywords(content).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (content.len() / 4) as u64;
        let completion_tokens = if success {
            result
                .as_ref()
                .map(|v| v.iter().map(|s| s.len()).sum::<usize>())
                .unwrap_or(0) as u64
                / 4
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::KeywordExtraction,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn summarize(&self, content: &str, max_length: Option<usize>) -> Result<String> {
        let start = Instant::now();
        let result = self.inner.summarize(content, max_length).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (content.len() / 4) as u64;
        let completion_tokens = if success {
            (result.as_ref().unwrap().len() / 4) as u64
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::SummaryGeneration,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn health_check(&self) -> Result<bool> {
        self.inner.health_check().await
    }

    async fn extract_structured_facts(&self, prompt: &str) -> Result<StructuredFactExtraction> {
        let start = Instant::now();
        let result = self.inner.extract_structured_facts(prompt).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (prompt.len() / 4) as u64;
        let completion_tokens = if success {
            let v = result.as_ref().unwrap();
            v.facts.iter().map(|s| s.len()).sum::<usize>() as u64 / 4
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::StructuredFactExtraction,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn extract_detailed_facts(&self, prompt: &str) -> Result<DetailedFactExtraction> {
        let start = Instant::now();
        let result = self.inner.extract_detailed_facts(prompt).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (prompt.len() / 4) as u64;
        let completion_tokens = if success {
            let v = result.as_ref().unwrap();
            v.facts
                .iter()
                .map(|f| {
                    f.content.len()
                        + f.category.len()
                        + f.entities.iter().map(|e| e.len()).sum::<usize>()
                })
                .sum::<usize>() as u64
                / 4
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::DetailedFactExtraction,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn extract_keywords_structured(&self, prompt: &str) -> Result<KeywordExtraction> {
        let start = Instant::now();
        let result = self.inner.extract_keywords_structured(prompt).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (prompt.len() / 4) as u64;
        let completion_tokens = if success {
            result
                .as_ref()
                .map(|v| v.keywords.iter().map(|s| s.len()).sum::<usize>())
                .unwrap_or(0) as u64
                / 4
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::KeywordExtraction,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn classify_memory(&self, prompt: &str) -> Result<MemoryClassification> {
        let start = Instant::now();
        let result = self.inner.classify_memory(prompt).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (prompt.len() / 4) as u64;
        let completion_tokens = if success {
            let v = result.as_ref().unwrap();
            (v.memory_type.len() + v.reasoning.len()) as u64 / 4
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::MemoryClassification,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn score_importance(&self, prompt: &str) -> Result<ImportanceScore> {
        let start = Instant::now();
        let result = self.inner.score_importance(prompt).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (prompt.len() / 4) as u64;
        let completion_tokens = if success {
            let v = result.as_ref().unwrap();
            v.reasoning.len() as u64 / 4
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::ImportanceScoring,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn check_duplicates(&self, prompt: &str) -> Result<DeduplicationResult> {
        let start = Instant::now();
        let result = self.inner.check_duplicates(prompt).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (prompt.len() / 4) as u64;
        let completion_tokens = if success {
            let v = result.as_ref().unwrap();
            v.original_memory_id.iter().map(|s| s.len()).sum::<usize>() as u64 / 4
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::DuplicateChecking,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn generate_summary(&self, prompt: &str) -> Result<SummaryResult> {
        let start = Instant::now();
        let result = self.inner.generate_summary(prompt).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (prompt.len() / 4) as u64;
        let completion_tokens = if success {
            let v = result.as_ref().unwrap();
            (v.summary.len()
                + v.key_points.iter().map(|s| s.len()).sum::<usize>()) as u64
                / 4
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::SummaryGeneration,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn detect_language(&self, prompt: &str) -> Result<LanguageDetection> {
        let start = Instant::now();
        let result = self.inner.detect_language(prompt).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (prompt.len() / 4) as u64;
        let completion_tokens = if success {
            let v = result.as_ref().unwrap();
            v.language.len() as u64 / 4
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::LanguageDetection,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn extract_entities(&self, prompt: &str) -> Result<EntityExtraction> {
        let start = Instant::now();
        let result = self.inner.extract_entities(prompt).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (prompt.len() / 4) as u64;
        let completion_tokens = if success {
            let v = result.as_ref().unwrap();
            v.entities
                .iter()
                .map(|e| e.text.len() + e.label.len())
                .sum::<usize>() as u64
                / 4
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::EntityExtraction,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn analyze_conversation(&self, prompt: &str) -> Result<ConversationAnalysis> {
        let start = Instant::now();
        let result = self.inner.analyze_conversation(prompt).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (prompt.len() / 4) as u64;
        let completion_tokens = if success {
            let v = result.as_ref().unwrap();
            (v.topics.iter().map(|s| s.len()).sum::<usize>()
                + v.sentiment.len()
                + v.user_intent.len()
                + v.key_information.iter().map(|s| s.len()).sum::<usize>()) as u64
                / 4
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::ConversationAnalysis,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn extract_metadata_enrichment(&self, prompt: &str) -> Result<MetadataEnrichment> {
        let start = Instant::now();
        let result = self.inner.extract_metadata_enrichment(prompt).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (prompt.len() / 4) as u64;
        let completion_tokens = if success {
            let v = result.as_ref().unwrap();
            (v.summary.len()
                + v.keywords.iter().map(|s| s.len()).sum::<usize>()) as u64
                / 4
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::MetadataEnrichment,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn extract_metadata_enrichment_batch(
        &self,
        texts: &[String],
    ) -> Result<Vec<Result<MetadataEnrichment>>> {
        let start = Instant::now();
        let results = self.inner.extract_metadata_enrichment_batch(texts).await;
        let total_duration = start.elapsed();

        // Record one metric per text in the batch
        let total_prompt_chars: usize = texts.iter().map(|t| t.len()).sum();
        let prompt_tokens = (total_prompt_chars / 4) as u64;

        let mut total_completion_chars = 0;
        let mut success_count = 0;
        if let Ok(ref results) = results {
            for r in results {
                match r {
                    Ok(v) => {
                        success_count += 1;
                        total_completion_chars += v.summary.len()
                            + v.keywords.iter().map(|s| s.len()).sum::<usize>();
                    }
                    Err(_) => {}
                }
            }
        }
        let completion_tokens = (total_completion_chars / 4) as u64;
        let overall_success = success_count == texts.len();

        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::MetadataEnrichmentBatch,
            total_duration,
            overall_success,
            prompt_tokens,
            completion_tokens,
        );

        results
    }

    async fn complete_batch(&self, prompts: &[String]) -> Result<Vec<Result<String>>> {
        let start = Instant::now();
        let results = self.inner.complete_batch(prompts).await;
        let total_duration = start.elapsed();

        // Record one metric per prompt in the batch
        let total_prompt_chars: usize = prompts.iter().map(|p| p.len()).sum();
        let prompt_tokens = (total_prompt_chars / 4) as u64;

        let mut total_completion_chars = 0;
        let mut success_count = 0;
        if let Ok(ref results) = results {
            for r in results {
                match r {
                    Ok(v) => {
                        success_count += 1;
                        total_completion_chars += v.len();
                    }
                    Err(_) => {}
                }
            }
        }
        let completion_tokens = (total_completion_chars / 4) as u64;
        let overall_success = success_count == prompts.len();

        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::Completion,
            total_duration,
            overall_success,
            prompt_tokens,
            completion_tokens,
        );

        results
    }

    fn get_status(&self) -> ClientStatus {
        self.inner.get_status()
    }

    fn batch_config(&self) -> (usize, u32) {
        self.inner.batch_config()
    }

    async fn enhance_memory_unified(&self, prompt: &str) -> Result<MemoryEnhancement> {
        let start = Instant::now();
        let result = self.inner.enhance_memory_unified(prompt).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        let prompt_tokens = (prompt.len() / 4) as u64;
        let completion_tokens = if success {
            let v = result.as_ref().unwrap();
            (v.memory_type.len()
                + v.summary.len()
                + v.keywords.iter().map(|s| s.len()).sum::<usize>()
                + v.entities.iter().map(|s| s.len()).sum::<usize>()
                + v.topics.iter().map(|s| s.len()).sum::<usize>()) as u64
                / 4
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::MemoryEnhancement,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }

    async fn describe_image(&self, image_bytes: &[u8], mime_type: &str) -> Result<String> {
        let start = Instant::now();
        let result = self.inner.describe_image(image_bytes, mime_type).await;
        let duration = start.elapsed();
        let success = result.is_ok();
        // For image description, prompt tokens is estimated from mime_type string
        // (the actual "prompt" is the image itself, so we use a minimal estimate)
        let prompt_tokens = 0;
        let completion_tokens = if success {
            (result.as_ref().unwrap().len() / 4) as u64
        } else {
            0
        };
        self.metrics.record_llm_request(
            self.backend_type,
            LlmOperationType::ImageDescription,
            duration,
            success,
            prompt_tokens,
            completion_tokens,
        );
        result
    }
}
