//! Shared test utilities for integration tests.

use async_trait::async_trait;
use llm_mem::error::Result;
use llm_mem::llm::{
    ClientStatus, ConversationAnalysis, DeduplicationResult, DetailedFactExtraction,
    EntityExtraction, ImportanceScore, KeywordExtraction, LLMClient, LanguageDetection,
    MemoryClassification, MemoryEnhancement, StructuredFactExtraction, SummaryResult,
};
use std::collections::HashMap;

pub const DIM: usize = 384;

#[derive(Clone)]
pub struct MockLLMClient {
    pub dimension: usize,
}

impl MockLLMClient {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Generate a simple deterministic embedding from text.
    pub fn make_embedding(&self, text: &str) -> Vec<f32> {
        let mut emb = vec![0.0f32; self.dimension];
        for (i, ch) in text.chars().enumerate() {
            emb[i % self.dimension] += (ch as u32 as f32) / 1000.0;
        }
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut emb {
                *v /= norm;
            }
        }
        emb
    }
}

#[async_trait]
impl LLMClient for MockLLMClient {
    async fn complete(&self, prompt: &str) -> Result<String> {
        Ok(format!(
            "Mock completion for: {}",
            &prompt[..prompt.len().min(50)]
        ))
    }

    async fn complete_with_grammar(&self, _prompt: &str, _grammar: &str) -> Result<String> {
        Ok("{\"summary\": \"mock summary\", \"keywords\": [\"mock\", \"test\"]}".to_string())
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.make_embedding(text))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.make_embedding(t)).collect())
    }

    async fn extract_keywords(&self, content: &str) -> Result<Vec<String>> {
        Ok(content
            .split_whitespace()
            .take(5)
            .map(|s| s.to_lowercase())
            .collect())
    }

    async fn summarize(&self, content: &str, max_length: Option<usize>) -> Result<String> {
        let limit = max_length.unwrap_or(100);
        Ok(content.chars().take(limit).collect())
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }

    async fn extract_structured_facts(&self, _prompt: &str) -> Result<StructuredFactExtraction> {
        Ok(StructuredFactExtraction {
            facts: vec!["mock fact".into()],
        })
    }

    async fn extract_detailed_facts(&self, _prompt: &str) -> Result<DetailedFactExtraction> {
        Ok(DetailedFactExtraction { facts: vec![] })
    }

    async fn extract_keywords_structured(&self, _prompt: &str) -> Result<KeywordExtraction> {
        Ok(KeywordExtraction {
            keywords: vec!["mock".into()],
        })
    }

    async fn classify_memory(&self, _prompt: &str) -> Result<MemoryClassification> {
        Ok(MemoryClassification {
            memory_type: "Factual".into(),
            confidence: 0.9,
            reasoning: "mock".into(),
        })
    }

    async fn score_importance(&self, _prompt: &str) -> Result<ImportanceScore> {
        Ok(ImportanceScore {
            score: 0.7,
            reasoning: "mock importance".into(),
        })
    }

    async fn check_duplicates(&self, _prompt: &str) -> Result<DeduplicationResult> {
        Ok(DeduplicationResult {
            is_duplicate: false,
            similarity_score: 0.0,
            original_memory_id: None,
        })
    }

    async fn generate_summary(&self, _prompt: &str) -> Result<SummaryResult> {
        Ok(SummaryResult {
            summary: "mock summary".into(),
            key_points: vec!["point1".into()],
        })
    }

    async fn detect_language(&self, _prompt: &str) -> Result<LanguageDetection> {
        Ok(LanguageDetection {
            language: "English".into(),
            confidence: 0.95,
        })
    }

    async fn extract_entities(&self, _prompt: &str) -> Result<EntityExtraction> {
        Ok(EntityExtraction { entities: vec![] })
    }

    async fn analyze_conversation(&self, _prompt: &str) -> Result<ConversationAnalysis> {
        Ok(ConversationAnalysis {
            topics: vec!["mock_topic".into()],
            sentiment: "neutral".into(),
            user_intent: "informational".into(),
            key_information: vec![],
        })
    }

    async fn extract_metadata_enrichment(
        &self,
        _prompt: &str,
    ) -> Result<llm_mem::llm::MetadataEnrichment> {
        Ok(llm_mem::llm::MetadataEnrichment {
            summary: "mock summary".into(),
            keywords: vec!["mock".into(), "test".into()],
        })
    }

    async fn extract_metadata_enrichment_batch(
        &self,
        texts: &[String],
    ) -> Result<Vec<Result<llm_mem::llm::MetadataEnrichment>>> {
        let mut results = Vec::new();
        for _ in texts {
            results.push(Ok(llm_mem::llm::MetadataEnrichment {
                summary: "mock summary".into(),
                keywords: vec!["mock".into(), "test".into()],
            }));
        }
        Ok(results)
    }

    async fn complete_batch(&self, prompts: &[String]) -> Result<Vec<Result<String>>> {
        let mut results = Vec::new();
        for p in prompts {
            results.push(self.complete(p).await);
        }
        Ok(results)
    }

    fn get_status(&self) -> ClientStatus {
        ClientStatus {
            backend: "mock".to_string(),
            state: "ready".to_string(),
            llm_model: "mock-model".to_string(),
            embedding_model: format!("mock-embed-dim{}", self.dimension),
            llm_available: true,
            embedding_available: true,
            last_llm_success: None,
            last_embedding_success: None,
            last_error: None,
            total_llm_calls: 0,
            total_embedding_calls: 0,
            total_prompt_tokens: 0,
            total_completion_tokens: 0,
            details: HashMap::new(),
        }
    }

    fn batch_config(&self) -> (usize, u32) {
        (10, 4096)
    }

    async fn enhance_memory_unified(&self, _prompt: &str) -> Result<MemoryEnhancement> {
        Ok(MemoryEnhancement {
            memory_type: "Semantic".into(),
            summary: String::new(),
            keywords: vec![],
            entities: vec![],
            topics: vec![],
        })
    }

    async fn describe_image(&self, _image_bytes: &[u8], _mime_type: &str) -> Result<String> {
        Err(llm_mem::error::MemoryError::LLM(
            "Mock: vision not available".into(),
        ))
    }
}

pub fn make_mock_client() -> MockLLMClient {
    MockLLMClient::new(DIM)
}
