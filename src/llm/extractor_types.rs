use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Status information returned by an LLM client backend.
///
/// Captures runtime details like backend type, model info, availability,
/// and usage statistics. Used by the `system_status` MCP tool.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClientStatus {
    /// Backend type: "local" or "openai"
    pub backend: String,
    /// Current operational state: "ready", "initializing", "error", "degraded"
    pub state: String,

    // ── Model information ──
    /// LLM model name or path
    pub llm_model: String,
    /// Embedding model name
    pub embedding_model: String,

    // ── Availability ──
    /// Whether the LLM service is currently reachable / loaded
    pub llm_available: bool,
    /// Whether the embedding service is available
    pub embedding_available: bool,
    /// ISO 8601 timestamp of last successful LLM call (None if never called)
    pub last_llm_success: Option<String>,
    /// ISO 8601 timestamp of last successful embedding call
    pub last_embedding_success: Option<String>,
    /// Last error message (if any)
    pub last_error: Option<String>,

    // ── Usage statistics (since process start) ──
    pub total_llm_calls: u64,
    pub total_embedding_calls: u64,
    /// Approximate prompt tokens processed (estimated for local, reported for API)
    pub total_prompt_tokens: u64,
    /// Approximate completion tokens generated
    pub total_completion_tokens: u64,

    // ── Backend-specific details ──
    /// Extra key-value details depending on backend.
    ///
    /// For local: gpu_layers, context_size, models_dir, llm_model_path,
    ///            llm_model_size_bytes, embedding_model_loaded
    /// For OpenAI: api_base_url, embedding_api_base_url
    pub details: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StructuredFactExtraction {
    pub facts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DetailedFactExtraction {
    pub facts: Vec<StructuredFact>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StructuredFact {
    pub content: String,
    pub importance: f32,
    pub category: String,
    pub entities: Vec<String>,
    pub source_role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct KeywordExtraction {
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MemoryClassification {
    pub memory_type: String,
    pub confidence: f32,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ImportanceScore {
    pub score: f32,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DeduplicationResult {
    pub is_duplicate: bool,
    pub similarity_score: f32,
    pub original_memory_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SummaryResult {
    pub summary: String,
    pub key_points: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MetadataEnrichment {
    pub summary: String,
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LanguageDetection {
    pub language: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EntityExtraction {
    pub entities: Vec<Entity>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Entity {
    pub text: String,
    pub label: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConversationAnalysis {
    pub topics: Vec<String>,
    pub sentiment: String,
    pub user_intent: String,
    pub key_information: Vec<String>,
}
