use crate::{
    error::Result,
    llm::LLMClient,
    types::{Memory, MemoryType},
};
use async_trait::async_trait;
use tracing::debug;

/// Trait for evaluating memory importance
#[async_trait]
pub trait ImportanceEvaluator: Send + Sync {
    /// Evaluate the importance of a memory
    async fn evaluate_importance(&self, memory: &Memory) -> Result<f32>;

    /// Evaluate importance for multiple memories
    async fn evaluate_batch(&self, memories: &[Memory]) -> Result<Vec<f32>>;
}

/// LLM-based importance evaluator
pub struct LLMImportanceEvaluator {
    llm_client: Box<dyn LLMClient>,
}

impl LLMImportanceEvaluator {
    pub fn new(llm_client: Box<dyn LLMClient>) -> Self {
        Self { llm_client }
    }

    fn create_importance_prompt(&self, memory: &Memory) -> String {
        let memory_type_context = match memory.metadata.memory_type {
            MemoryType::Personal => "personal information, preferences, or characteristics",
            MemoryType::Factual => "factual information, data, or objective statements",
            MemoryType::Procedural => "instructions, procedures, or how-to information",
            MemoryType::Conversational => "conversational context or dialogue",
            MemoryType::Semantic => "concepts, meanings, or general knowledge",
            MemoryType::Episodic => "specific events, experiences, or temporal information",
            MemoryType::Relation => "relationship edges between entities in the knowledge graph",
        };

        format!(
            r#"Evaluate the importance of this memory on a scale from 0.0 to 1.0, where:
- 0.0-0.2: Trivial information (small talk, temporary states)
- 0.2-0.4: Low importance (minor preferences, casual mentions)
- 0.4-0.6: Medium importance (useful context, moderate preferences)
- 0.6-0.8: High importance (key facts, strong preferences, important context)
- 0.8-1.0: Critical importance (core identity, critical facts, essential information)

Memory Type: {} ({})
Content: "{}"
Created: {}

Respond with only a number between 0.0 and 1.0:"#,
            format!("{:?}", memory.metadata.memory_type),
            memory_type_context,
            memory.content.as_deref().unwrap_or("[no content]"),
            memory.created_at.format("%Y-%m-%d %H:%M:%S")
        )
    }
}

#[async_trait]
impl ImportanceEvaluator for LLMImportanceEvaluator {
    async fn evaluate_importance(&self, memory: &Memory) -> Result<f32> {
        let prompt = self.create_importance_prompt(memory);

        match self.llm_client.score_importance(&prompt).await {
            Ok(importance_score) => Ok(importance_score.score.clamp(0.0, 1.0)),
            Err(e) => {
                debug!(
                    "Rig extractor failed, falling back to traditional method: {}",
                    e
                );

                #[cfg(debug_assertions)]
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;

                let response = self.llm_client.complete(&prompt).await?;
                let importance = response
                    .trim()
                    .parse::<f32>()
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);

                Ok(importance)
            }
        }
    }

    async fn evaluate_batch(&self, memories: &[Memory]) -> Result<Vec<f32>> {
        let mut results = Vec::with_capacity(memories.len());
        for memory in memories {
            let importance = self.evaluate_importance(memory).await?;
            results.push(importance);
        }
        Ok(results)
    }
}

/// Rule-based importance evaluator for faster evaluation
pub struct RuleBasedImportanceEvaluator;

impl RuleBasedImportanceEvaluator {
    pub fn new() -> Self {
        Self
    }

    fn evaluate_by_content_length(&self, content: &str) -> f32 {
        let length = content.len();
        match length {
            0..=20 => 0.1,
            21..=50 => 0.2,
            51..=100 => 0.3,
            101..=200 => 0.4,
            201..=500 => 0.5,
            501..=1000 => 0.6,
            _ => 0.7,
        }
    }

    fn evaluate_by_memory_type(&self, memory_type: &MemoryType) -> f32 {
        match memory_type {
            MemoryType::Personal => 0.8,
            MemoryType::Factual => 0.7,
            MemoryType::Procedural => 0.6,
            MemoryType::Semantic => 0.5,
            MemoryType::Episodic => 0.4,
            MemoryType::Conversational => 0.3,
            MemoryType::Relation => 0.4,
        }
    }

    fn evaluate_by_keywords(&self, content: &str) -> f32 {
        let important_keywords = [
            "important",
            "critical",
            "remember",
            "never",
            "always",
            "prefer",
            "like",
            "dislike",
            "hate",
            "love",
            "name",
            "birthday",
            "address",
            "phone",
            "email",
            "password",
            "secret",
            "private",
            "confidential",
        ];

        let content_lower = content.to_lowercase();
        let keyword_count = important_keywords
            .iter()
            .filter(|&&keyword| content_lower.contains(keyword))
            .count();

        (keyword_count as f32 * 0.1).min(0.5)
    }
}

#[async_trait]
impl ImportanceEvaluator for RuleBasedImportanceEvaluator {
    async fn evaluate_importance(&self, memory: &Memory) -> Result<f32> {
        let content_str = memory.content.as_deref().unwrap_or("");
        let content_score = self.evaluate_by_content_length(content_str);
        let type_score = self.evaluate_by_memory_type(&memory.metadata.memory_type);
        let keyword_score = self.evaluate_by_keywords(content_str);

        let importance =
            (content_score * 0.3 + type_score * 0.5 + keyword_score * 0.2).clamp(0.0, 1.0);

        Ok(importance)
    }

    async fn evaluate_batch(&self, memories: &[Memory]) -> Result<Vec<f32>> {
        let mut results = Vec::with_capacity(memories.len());
        for memory in memories {
            let importance = self.evaluate_importance(memory).await?;
            results.push(importance);
        }
        Ok(results)
    }
}

/// Hybrid evaluator that combines LLM and rule-based approaches
pub struct HybridImportanceEvaluator {
    llm_evaluator: LLMImportanceEvaluator,
    rule_evaluator: RuleBasedImportanceEvaluator,
    llm_threshold: f32,
}

impl HybridImportanceEvaluator {
    pub fn new(llm_client: Box<dyn LLMClient>, llm_threshold: f32) -> Self {
        Self {
            llm_evaluator: LLMImportanceEvaluator::new(llm_client),
            rule_evaluator: RuleBasedImportanceEvaluator::new(),
            llm_threshold,
        }
    }
}

#[async_trait]
impl ImportanceEvaluator for HybridImportanceEvaluator {
    async fn evaluate_importance(&self, memory: &Memory) -> Result<f32> {
        let rule_score = self.rule_evaluator.evaluate_importance(memory).await?;

        if rule_score >= self.llm_threshold {
            let llm_score = self.llm_evaluator.evaluate_importance(memory).await?;
            Ok((llm_score * 0.7 + rule_score * 0.3).clamp(0.0, 1.0))
        } else {
            Ok(rule_score)
        }
    }

    async fn evaluate_batch(&self, memories: &[Memory]) -> Result<Vec<f32>> {
        let mut results = Vec::with_capacity(memories.len());
        for memory in memories {
            let importance = self.evaluate_importance(memory).await?;
            results.push(importance);
        }
        Ok(results)
    }
}

/// Factory function to create importance evaluators
pub fn create_importance_evaluator(
    llm_client: Box<dyn LLMClient>,
    use_llm: bool,
    hybrid_threshold: Option<f32>,
) -> Box<dyn ImportanceEvaluator> {
    match (use_llm, hybrid_threshold) {
        (true, Some(threshold)) => Box::new(HybridImportanceEvaluator::new(llm_client, threshold)),
        (true, None) => Box::new(LLMImportanceEvaluator::new(llm_client)),
        (false, _) => Box::new(RuleBasedImportanceEvaluator::new()),
    }
}
