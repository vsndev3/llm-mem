use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde::de::{self, Visitor};
use std::collections::HashMap;
use std::fmt;

/// Custom deserializer that accepts either a single string or an array of strings
/// Returns a Vec<String> in both cases
fn string_or_vec_string<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringOrVecVisitor;

    impl<'de> Visitor<'de> for StringOrVecVisitor {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string or an array of strings")
        }

        fn visit_str<E>(self, value: &str) -> Result<Vec<String>, E>
        where
            E: de::Error,
        {
            // Single string - wrap in vec
            Ok(vec![value.to_string()])
        }

        fn visit_string<E>(self, value: String) -> Result<Vec<String>, E>
        where
            E: de::Error,
        {
            // Single string - wrap in vec
            Ok(vec![value])
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Vec<String>, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            // Array of strings
            let mut vec = Vec::new();
            while let Some(element) = seq.next_element::<String>()? {
                vec.push(element);
            }
            Ok(vec)
        }
    }

    deserializer.deserialize_any(StringOrVecVisitor)
}

/// Custom deserializer for KeywordExtraction that accepts:
/// 1. A raw array: ["keyword1", "keyword2"]
/// 2. An object with keywords field: {"keywords": ["keyword1", "keyword2"]}
/// 3. A single string: "keyword1, keyword2"
fn keyword_extraction_deserializer<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct KeywordExtractionVisitor;

    impl<'de> Visitor<'de> for KeywordExtractionVisitor {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string, array of strings, or object with keywords field")
        }

        fn visit_str<E>(self, value: &str) -> Result<Vec<String>, E>
        where
            E: de::Error,
        {
            // Single string - split by comma or wrap in vec
            let trimmed = value.trim();
            if trimmed.contains(',') {
                Ok(trimmed.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
            } else if !trimmed.is_empty() {
                Ok(vec![trimmed.to_string()])
            } else {
                Ok(vec![])
            }
        }

        fn visit_string<E>(self, value: String) -> Result<Vec<String>, E>
        where
            E: de::Error,
        {
            self.visit_str(&value)
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Vec<String>, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            // Array of strings
            let mut vec = Vec::new();
            while let Some(element) = seq.next_element::<String>()? {
                vec.push(element);
            }
            Ok(vec)
        }

        fn visit_map<M>(self, mut map: M) -> Result<Vec<String>, M::Error>
        where
            M: de::MapAccess<'de>,
        {
            // Object - look for keywords field
            let mut keywords: Option<Vec<String>> = None;
            while let Some(key) = map.next_key::<String>()? {
                if key == "keywords" || key == "tags" || key == "terms" || key == "items" {
                    keywords = Some(map.next_value()?);
                } else {
                    // Skip unknown fields
                    let _: de::IgnoredAny = map.next_value()?;
                }
            }
            Ok(keywords.unwrap_or_default())
        }
    }

    deserializer.deserialize_any(KeywordExtractionVisitor)
}

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
    #[serde(deserialize_with = "string_or_vec_string")]
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

/// Keyword extraction result - transparent wrapper around Vec<String>
/// Accepts raw arrays, objects with keywords field, or comma-separated strings
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct KeywordExtraction {
    #[serde(deserialize_with = "keyword_extraction_deserializer")]
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
    #[serde(alias = "importance", alias = "priority", alias = "relevance")]
    pub score: f32,
    #[serde(default = "default_reasoning", alias = "reason", alias = "explanation")]
    pub reasoning: String,
}

fn default_reasoning() -> String {
    "Importance score computed by LLM".to_string()
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
    #[serde(deserialize_with = "string_or_vec_string")]
    pub key_points: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MetadataEnrichment {
    pub summary: String,
    #[serde(deserialize_with = "string_or_vec_string")]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_extraction_from_raw_array() {
        let json = r#"["keyword1", "keyword2", "keyword3"]"#;
        let result: KeywordExtraction = serde_json::from_str(json).unwrap();
        assert_eq!(result.keywords, vec!["keyword1", "keyword2", "keyword3"]);
    }

    #[test]
    fn test_keyword_extraction_from_object() {
        let json = r#"{"keywords": ["rust", "testing"]}"#;
        let result: KeywordExtraction = serde_json::from_str(json).unwrap();
        assert_eq!(result.keywords, vec!["rust", "testing"]);
    }

    #[test]
    fn test_keyword_extraction_from_comma_separated() {
        let json = r#""keyword1, keyword2, keyword3""#;
        let result: KeywordExtraction = serde_json::from_str(json).unwrap();
        assert_eq!(result.keywords, vec!["keyword1", "keyword2", "keyword3"]);
    }

    #[test]
    fn test_keyword_extraction_from_single_string() {
        let json = r#""single_keyword""#;
        let result: KeywordExtraction = serde_json::from_str(json).unwrap();
        assert_eq!(result.keywords, vec!["single_keyword"]);
    }

    #[test]
    fn test_keyword_extraction_with_tags_alias() {
        let json = r#"{"tags": ["tag1", "tag2"]}"#;
        let result: KeywordExtraction = serde_json::from_str(json).unwrap();
        assert_eq!(result.keywords, vec!["tag1", "tag2"]);
    }

    #[test]
    fn test_keyword_extraction_with_terms_alias() {
        let json = r#"{"terms": ["term1", "term2"]}"#;
        let result: KeywordExtraction = serde_json::from_str(json).unwrap();
        assert_eq!(result.keywords, vec!["term1", "term2"]);
    }

    #[test]
    fn test_keyword_extraction_empty_array() {
        let json = r#"[]"#;
        let result: KeywordExtraction = serde_json::from_str(json).unwrap();
        assert!(result.keywords.is_empty());
    }

    #[test]
    fn test_importance_score_with_score_field() {
        let json = r#"{"score": 0.8, "reasoning": "Very important"}"#;
        let result: ImportanceScore = serde_json::from_str(json).unwrap();
        assert_eq!(result.score, 0.8);
        assert_eq!(result.reasoning, "Very important");
    }

    #[test]
    fn test_importance_score_with_importance_alias() {
        let json = r#"{"importance": 0.7, "reasoning": "Important"}"#;
        let result: ImportanceScore = serde_json::from_str(json).unwrap();
        assert_eq!(result.score, 0.7);
        assert_eq!(result.reasoning, "Important");
    }

    #[test]
    fn test_importance_score_with_priority_alias() {
        let json = r#"{"priority": 0.9, "reasoning": "Critical"}"#;
        let result: ImportanceScore = serde_json::from_str(json).unwrap();
        assert_eq!(result.score, 0.9);
        assert_eq!(result.reasoning, "Critical");
    }

    #[test]
    fn test_importance_score_with_relevance_alias() {
        let json = r#"{"relevance": 0.6, "reasoning": "Moderately relevant"}"#;
        let result: ImportanceScore = serde_json::from_str(json).unwrap();
        assert_eq!(result.score, 0.6);
        assert_eq!(result.reasoning, "Moderately relevant");
    }

    #[test]
    fn test_importance_score_with_reason_alias() {
        let json = r#"{"score": 0.5, "reason": "Some reason"}"#;
        let result: ImportanceScore = serde_json::from_str(json).unwrap();
        assert_eq!(result.score, 0.5);
        assert_eq!(result.reasoning, "Some reason");
    }

    #[test]
    fn test_importance_score_with_explanation_alias() {
        let json = r#"{"score": 0.4, "explanation": "Detailed explanation"}"#;
        let result: ImportanceScore = serde_json::from_str(json).unwrap();
        assert_eq!(result.score, 0.4);
        assert_eq!(result.reasoning, "Detailed explanation");
    }

    #[test]
    fn test_importance_score_missing_reasoning_uses_default() {
        let json = r#"{"score": 0.5}"#;
        let result: ImportanceScore = serde_json::from_str(json).unwrap();
        assert_eq!(result.score, 0.5);
        assert_eq!(result.reasoning, "Importance score computed by LLM");
    }

    #[test]
    fn test_keyword_extraction_serialization() {
        let extraction = KeywordExtraction {
            keywords: vec!["rust".to_string(), "testing".to_string()],
        };
        let json = serde_json::to_string(&extraction).unwrap();
        // Should serialize as a plain array due to #[serde(transparent)]
        assert_eq!(json, r#"["rust","testing"]"#);
    }

    #[test]
    fn test_importance_score_serialization() {
        let score = ImportanceScore {
            score: 0.75,
            reasoning: "Test reasoning".to_string(),
        };
        let json = serde_json::to_string(&score).unwrap();
        assert!(json.contains("\"score\":0.75"));
        assert!(json.contains("\"reasoning\":\"Test reasoning\""));
    }
}
