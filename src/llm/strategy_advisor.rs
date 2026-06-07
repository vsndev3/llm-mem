use super::LLMClient;
use crate::{
    error::Result,
    ingest::{
        document_tree::{DocumentMeta, DocumentNode},
        format_detect::InputFormat,
    },
};

pub struct LLMStrategyAdvisor<'a> {
    client: &'a dyn LLMClient,
    max_content_bytes: usize,
}

#[derive(Debug, serde::Deserialize)]
struct LLMFallbackResponse {
    summary: String,
    #[serde(default)]
    entities: Vec<String>,
    #[serde(default)]
    title: String,
    #[serde(default)]
    content_type: String,
}

impl<'a> LLMStrategyAdvisor<'a> {
    pub fn new(client: &'a dyn LLMClient) -> Self {
        Self {
            client,
            max_content_bytes: 4096,
        }
    }

    pub async fn detect_format(&self, content: &str) -> Option<InputFormat> {
        let preview = &content[..content.len().min(self.max_content_bytes)];

        let prompt = format!(
            "Analyze this content and identify its format. Reply with ONLY the format \
             name from this list, nothing else: markdown, json, yaml, toml, rust, \
             python, javascript, typescript, go, cpp, csv, plaintext, html, xml, sql, \
             shell, dockerfile, unknown.\n\nContent:\n```\n{}\n```",
            preview
        );

        let response = match self.client.complete(&prompt).await {
            Ok(r) => r.trim().to_lowercase(),
            Err(_) => return None,
        };

        match response.as_str() {
            "markdown" => Some(InputFormat::Markdown),
            "json" => Some(InputFormat::Json),
            "yaml" => Some(InputFormat::Yaml),
            "toml" => Some(InputFormat::Toml),
            "rust" => Some(InputFormat::Rust),
            "python" => Some(InputFormat::Python),
            "javascript" => Some(InputFormat::JavaScript),
            "typescript" => Some(InputFormat::TypeScript),
            "go" => Some(InputFormat::Go),
            "cpp" | "c++" => Some(InputFormat::Cpp),
            "csv" => Some(InputFormat::Csv),
            "plaintext" | "text" => Some(InputFormat::PlainText),
            _ => None,
        }
    }

    pub async fn fallback_parse(
        &self,
        content: &str,
        original_format: &str,
    ) -> Result<(DocumentNode, DocumentMeta)> {
        let total_bytes = content.len() as u64;
        let preview = &content[..content.len().min(self.max_content_bytes * 4)];

        let prompt = format!(
            r#"Analyze the following content (reported as: {}). Respond with ONLY valid JSON, no markdown fences:

{{
    "summary": "A clear 2-3 sentence summary of the content",
    "entities": ["entity1", "entity2", "entity3"],
    "title": "inferred title or empty string",
    "content_type": "one of: documentation, code, data, prose, config, log, other"
}}

Content:
```
{}
```"#,
            original_format, preview
        );

        let response = self.client.complete(&prompt).await?;

        let analysis = match serde_json::from_str::<LLMFallbackResponse>(&response) {
            Ok(v) => v,
            Err(_) => LLMFallbackResponse {
                summary: response.trim().to_string(),
                entities: vec![],
                title: String::new(),
                content_type: "other".to_string(),
            },
        };

        let mut meta = DocumentMeta::new(original_format, "application/octet-stream", total_bytes);
        meta.parser_confidence = 0.5;
        meta.custom
            .insert("llm_summary".into(), analysis.summary.clone());
        meta.custom
            .insert("llm_content_type".into(), analysis.content_type.clone());
        meta.custom.insert(
            "llm_advisory".into(),
            "Content parsed via LLM fallback. May be incomplete.".into(),
        );
        if !analysis.title.is_empty() {
            meta.custom
                .insert("llm_title".into(), analysis.title.clone());
        } else {
            meta.custom.insert("llm_title".into(), String::new());
        }
        meta.warnings
            .push("Parsed via LLM fallback. Structured parsers failed.".into());

        let raw_node = DocumentNode::Raw {
            content: content.to_string(),
            mime_type: "application/octet-stream".to_string(),
            id: None,
        };

        let mut children = vec![raw_node];

        if !analysis.summary.is_empty() {
            children.push(DocumentNode::Paragraph {
                text: analysis.summary,
                id: Some("llm-summary".into()),
            });
        }

        if !analysis.entities.is_empty() {
            children.push(DocumentNode::List {
                items: analysis.entities,
                ordered: false,
                id: Some("llm-entities".into()),
            });
        }

        let doc = if !analysis.title.is_empty() {
            DocumentNode::Document {
                children: vec![DocumentNode::Section {
                    title: analysis.title,
                    level: 1,
                    children,
                    id: Some("llm-fallback-root".into()),
                }],
                meta: meta.clone(),
            }
        } else {
            DocumentNode::Document {
                children,
                meta: meta.clone(),
            }
        };

        Ok((doc, meta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::extractor_types::*;
    use async_trait::async_trait;

    #[derive(Clone)]
    struct ErrorMock;

    #[async_trait]
    impl LLMClient for ErrorMock {
        async fn complete(&self, _prompt: &str) -> Result<String> {
            Err(crate::error::MemoryError::LLM("fail".into()))
        }
        async fn complete_with_grammar(&self, _p: &str, _g: &str) -> Result<String> {
            Ok("".into())
        }
        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![])
        }
        async fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![])
        }
        async fn extract_keywords(&self, _c: &str) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn summarize(&self, _c: &str, _l: Option<usize>) -> Result<String> {
            Ok("".into())
        }
        async fn health_check(&self) -> Result<bool> {
            Ok(true)
        }
        async fn extract_structured_facts(&self, _p: &str) -> Result<StructuredFactExtraction> {
            Ok(StructuredFactExtraction { facts: vec![] })
        }
        async fn extract_detailed_facts(&self, _p: &str) -> Result<DetailedFactExtraction> {
            Ok(DetailedFactExtraction { facts: vec![] })
        }
        async fn extract_keywords_structured(&self, _p: &str) -> Result<KeywordExtraction> {
            Ok(KeywordExtraction { keywords: vec![] })
        }
        async fn classify_memory(&self, _p: &str) -> Result<MemoryClassification> {
            Ok(MemoryClassification {
                memory_type: "Factual".into(),
                confidence: 0.0,
                reasoning: String::new(),
            })
        }
        async fn score_importance(&self, _p: &str) -> Result<ImportanceScore> {
            Ok(ImportanceScore {
                score: 0.0,
                reasoning: String::new(),
            })
        }
        async fn check_duplicates(&self, _p: &str) -> Result<DeduplicationResult> {
            Ok(DeduplicationResult {
                is_duplicate: false,
                similarity_score: 0.0,
                original_memory_id: None,
            })
        }
        async fn generate_summary(&self, _p: &str) -> Result<SummaryResult> {
            Ok(SummaryResult {
                summary: String::new(),
                key_points: vec![],
            })
        }
        async fn detect_language(&self, _p: &str) -> Result<LanguageDetection> {
            Ok(LanguageDetection {
                language: "en".into(),
                confidence: 1.0,
            })
        }
        async fn extract_entities(&self, _p: &str) -> Result<EntityExtraction> {
            Ok(EntityExtraction { entities: vec![] })
        }
        async fn analyze_conversation(&self, _p: &str) -> Result<ConversationAnalysis> {
            Ok(ConversationAnalysis {
                topics: vec![],
                sentiment: String::new(),
                user_intent: String::new(),
                key_information: vec![],
            })
        }
        async fn extract_metadata_enrichment(&self, _p: &str) -> Result<MetadataEnrichment> {
            Ok(MetadataEnrichment {
                summary: "mock".into(),
                keywords: vec![],
            })
        }
        async fn extract_metadata_enrichment_batch(
            &self,
            _t: &[String],
        ) -> Result<Vec<Result<MetadataEnrichment>>> {
            Ok(vec![])
        }
        async fn complete_batch(&self, _p: &[String]) -> Result<Vec<Result<String>>> {
            Ok(vec![])
        }
        fn get_status(&self) -> ClientStatus {
            ClientStatus::default()
        }
        fn batch_config(&self) -> (usize, u32) {
            (10, 4096)
        }
        async fn enhance_memory_unified(&self, _p: &str) -> Result<MemoryEnhancement> {
            Ok(MemoryEnhancement {
                memory_type: "Semantic".into(),
                summary: String::new(),
                keywords: vec![],
                entities: vec![],
                topics: vec![],
            })
        }
        async fn describe_image(&self, _b: &[u8], _m: &str) -> Result<String> {
            Err(crate::error::MemoryError::LLM("nope".into()))
        }
    }

    #[tokio::test]
    async fn test_detect_format_llm_error_returns_none() {
        let advisor = LLMStrategyAdvisor::new(&ErrorMock);
        assert!(advisor.detect_format("content").await.is_none());
        assert!(advisor.fallback_parse("content", "unknown").await.is_err());
    }

    #[derive(Clone)]
    struct FormatMock(String);

    #[async_trait]
    impl LLMClient for FormatMock {
        async fn complete(&self, prompt: &str) -> Result<String> {
            if prompt.contains("identify its format") {
                Ok(self.0.clone())
            } else {
                Ok(r#"{"summary":"Test summary","entities":["alpha","beta"],"title":"Test Doc","content_type":"documentation"}"#.to_string())
            }
        }
        async fn complete_with_grammar(&self, _p: &str, _g: &str) -> Result<String> {
            Ok("{}".into())
        }
        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![0.0; 384])
        }
        async fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![vec![0.0; 384]])
        }
        async fn extract_keywords(&self, _c: &str) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn summarize(&self, _c: &str, _l: Option<usize>) -> Result<String> {
            Ok(String::new())
        }
        async fn health_check(&self) -> Result<bool> {
            Ok(true)
        }
        async fn extract_structured_facts(&self, _p: &str) -> Result<StructuredFactExtraction> {
            Ok(StructuredFactExtraction { facts: vec![] })
        }
        async fn extract_detailed_facts(&self, _p: &str) -> Result<DetailedFactExtraction> {
            Ok(DetailedFactExtraction { facts: vec![] })
        }
        async fn extract_keywords_structured(&self, _p: &str) -> Result<KeywordExtraction> {
            Ok(KeywordExtraction { keywords: vec![] })
        }
        async fn classify_memory(&self, _p: &str) -> Result<MemoryClassification> {
            Ok(MemoryClassification {
                memory_type: "Factual".into(),
                confidence: 0.0,
                reasoning: String::new(),
            })
        }
        async fn score_importance(&self, _p: &str) -> Result<ImportanceScore> {
            Ok(ImportanceScore {
                score: 0.0,
                reasoning: String::new(),
            })
        }
        async fn check_duplicates(&self, _p: &str) -> Result<DeduplicationResult> {
            Ok(DeduplicationResult {
                is_duplicate: false,
                similarity_score: 0.0,
                original_memory_id: None,
            })
        }
        async fn generate_summary(&self, _p: &str) -> Result<SummaryResult> {
            Ok(SummaryResult {
                summary: String::new(),
                key_points: vec![],
            })
        }
        async fn detect_language(&self, _p: &str) -> Result<LanguageDetection> {
            Ok(LanguageDetection {
                language: "en".into(),
                confidence: 1.0,
            })
        }
        async fn extract_entities(&self, _p: &str) -> Result<EntityExtraction> {
            Ok(EntityExtraction { entities: vec![] })
        }
        async fn analyze_conversation(&self, _p: &str) -> Result<ConversationAnalysis> {
            Ok(ConversationAnalysis {
                topics: vec![],
                sentiment: String::new(),
                user_intent: String::new(),
                key_information: vec![],
            })
        }
        async fn extract_metadata_enrichment(&self, _p: &str) -> Result<MetadataEnrichment> {
            Ok(MetadataEnrichment {
                summary: "mock".into(),
                keywords: vec![],
            })
        }
        async fn extract_metadata_enrichment_batch(
            &self,
            _t: &[String],
        ) -> Result<Vec<Result<MetadataEnrichment>>> {
            Ok(vec![])
        }
        async fn complete_batch(&self, _p: &[String]) -> Result<Vec<Result<String>>> {
            Ok(vec![])
        }
        fn get_status(&self) -> ClientStatus {
            ClientStatus::default()
        }
        fn batch_config(&self) -> (usize, u32) {
            (10, 4096)
        }
        async fn enhance_memory_unified(&self, _p: &str) -> Result<MemoryEnhancement> {
            Ok(MemoryEnhancement {
                memory_type: "Semantic".into(),
                summary: String::new(),
                keywords: vec![],
                entities: vec![],
                topics: vec![],
            })
        }
        async fn describe_image(&self, _b: &[u8], _m: &str) -> Result<String> {
            Ok("test image".to_string())
        }
    }

    #[tokio::test]
    async fn test_detect_format_python() {
        let mock = FormatMock("python".into());
        let advisor = LLMStrategyAdvisor::new(&mock);
        assert_eq!(
            advisor.detect_format("def foo():\n    pass").await,
            Some(InputFormat::Python)
        );
    }

    #[tokio::test]
    async fn test_detect_format_json() {
        let mock = FormatMock("json".into());
        let advisor = LLMStrategyAdvisor::new(&mock);
        assert_eq!(advisor.detect_format("{}").await, Some(InputFormat::Json));
    }

    #[tokio::test]
    async fn test_detect_format_unknown() {
        let mock = FormatMock("unknown".into());
        let advisor = LLMStrategyAdvisor::new(&mock);
        assert_eq!(advisor.detect_format("some data").await, None);
    }

    #[tokio::test]
    async fn test_detect_format_unsupported() {
        let mock = FormatMock("html".into());
        let advisor = LLMStrategyAdvisor::new(&mock);
        assert_eq!(advisor.detect_format("data").await, None);
    }

    #[tokio::test]
    async fn test_fallback_parse_success() {
        let mock = FormatMock("python".into());
        let advisor = LLMStrategyAdvisor::new(&mock);
        let (doc, meta) = advisor
            .fallback_parse("test content here", "unknown")
            .await
            .unwrap();

        assert_eq!(meta.format, "unknown");
        assert_eq!(meta.parser_confidence, 0.5);
        assert_eq!(meta.custom.get("llm_summary").unwrap(), "Test summary");
        assert_eq!(
            meta.custom.get("llm_content_type").unwrap(),
            "documentation"
        );
        assert_eq!(meta.custom.get("llm_title").unwrap(), "Test Doc");
        assert!(meta.custom.contains_key("llm_advisory"));
        assert!(!meta.warnings.is_empty());

        match &doc {
            DocumentNode::Document {
                children,
                meta: doc_meta,
            } => {
                assert_eq!(doc_meta.format, "unknown");
                // First child is Section (has title), second is Raw
                match &children[0] {
                    DocumentNode::Section {
                        title,
                        children: section_children,
                        ..
                    } => {
                        assert_eq!(title, "Test Doc");
                        assert!(
                            section_children
                                .iter()
                                .any(|c| matches!(c, DocumentNode::Raw { .. }))
                        );
                        assert!(
                            section_children
                                .iter()
                                .any(|c| matches!(c, DocumentNode::Paragraph { .. }))
                        );
                    }
                    _ => panic!("Expected Section"),
                }
            }
            _ => panic!("Expected Document"),
        }
    }

    #[tokio::test]
    async fn test_fallback_parse_no_title() {
        #[derive(Clone)]
        struct NoTitleMock;
        #[async_trait]
        impl LLMClient for NoTitleMock {
            async fn complete(&self, _p: &str) -> Result<String> {
                Ok(r#"{"summary":"Summary here","entities":["a"],"title":"","content_type":"data"}"#.to_string())
            }
            async fn complete_with_grammar(&self, _p: &str, _g: &str) -> Result<String> {
                Ok("".into())
            }
            async fn embed(&self, _t: &str) -> Result<Vec<f32>> {
                Ok(vec![0.0])
            }
            async fn embed_batch(&self, _t: &[String]) -> Result<Vec<Vec<f32>>> {
                Ok(vec![])
            }
            async fn extract_keywords(&self, _c: &str) -> Result<Vec<String>> {
                Ok(vec![])
            }
            async fn summarize(&self, _c: &str, _l: Option<usize>) -> Result<String> {
                Ok(String::new())
            }
            async fn health_check(&self) -> Result<bool> {
                Ok(true)
            }
            async fn extract_structured_facts(&self, _p: &str) -> Result<StructuredFactExtraction> {
                Ok(StructuredFactExtraction { facts: vec![] })
            }
            async fn extract_detailed_facts(&self, _p: &str) -> Result<DetailedFactExtraction> {
                Ok(DetailedFactExtraction { facts: vec![] })
            }
            async fn extract_keywords_structured(&self, _p: &str) -> Result<KeywordExtraction> {
                Ok(KeywordExtraction { keywords: vec![] })
            }
            async fn classify_memory(&self, _p: &str) -> Result<MemoryClassification> {
                Ok(MemoryClassification {
                    memory_type: "Factual".into(),
                    confidence: 0.0,
                    reasoning: String::new(),
                })
            }
            async fn score_importance(&self, _p: &str) -> Result<ImportanceScore> {
                Ok(ImportanceScore {
                    score: 0.0,
                    reasoning: String::new(),
                })
            }
            async fn check_duplicates(&self, _p: &str) -> Result<DeduplicationResult> {
                Ok(DeduplicationResult {
                    is_duplicate: false,
                    similarity_score: 0.0,
                    original_memory_id: None,
                })
            }
            async fn generate_summary(&self, _p: &str) -> Result<SummaryResult> {
                Ok(SummaryResult {
                    summary: String::new(),
                    key_points: vec![],
                })
            }
            async fn detect_language(&self, _p: &str) -> Result<LanguageDetection> {
                Ok(LanguageDetection {
                    language: "en".into(),
                    confidence: 1.0,
                })
            }
            async fn extract_entities(&self, _p: &str) -> Result<EntityExtraction> {
                Ok(EntityExtraction { entities: vec![] })
            }
            async fn analyze_conversation(&self, _p: &str) -> Result<ConversationAnalysis> {
                Ok(ConversationAnalysis {
                    topics: vec![],
                    sentiment: String::new(),
                    user_intent: String::new(),
                    key_information: vec![],
                })
            }
            async fn extract_metadata_enrichment(&self, _p: &str) -> Result<MetadataEnrichment> {
                Ok(MetadataEnrichment {
                    summary: "mock".into(),
                    keywords: vec![],
                })
            }
            async fn extract_metadata_enrichment_batch(
                &self,
                _t: &[String],
            ) -> Result<Vec<Result<MetadataEnrichment>>> {
                Ok(vec![])
            }
            async fn complete_batch(&self, _p: &[String]) -> Result<Vec<Result<String>>> {
                Ok(vec![])
            }
            fn get_status(&self) -> ClientStatus {
                ClientStatus::default()
            }
            fn batch_config(&self) -> (usize, u32) {
                (10, 4096)
            }
            async fn enhance_memory_unified(&self, _p: &str) -> Result<MemoryEnhancement> {
                Ok(MemoryEnhancement {
                    memory_type: "Semantic".into(),
                    summary: String::new(),
                    keywords: vec![],
                    entities: vec![],
                    topics: vec![],
                })
            }
            async fn describe_image(&self, _b: &[u8], _m: &str) -> Result<String> {
                Ok("".into())
            }
        }

        let advisor = LLMStrategyAdvisor::new(&NoTitleMock);
        let (doc, _meta) = advisor.fallback_parse("data", "unknown").await.unwrap();
        match &doc {
            DocumentNode::Document { children, .. } => {
                assert!(
                    children
                        .iter()
                        .any(|c| matches!(c, DocumentNode::Raw { .. }))
                );
                assert!(
                    children
                        .iter()
                        .any(|c| matches!(c, DocumentNode::Paragraph { .. }))
                );
            }
            _ => panic!("Expected Document"),
        }
    }

    #[tokio::test]
    async fn test_fallback_parse_malformed_json_handled() {
        #[derive(Clone)]
        struct MalformedMock;
        #[async_trait]
        impl LLMClient for MalformedMock {
            async fn complete(&self, _p: &str) -> Result<String> {
                Ok("Not valid JSON at all".to_string())
            }
            async fn complete_with_grammar(&self, _p: &str, _g: &str) -> Result<String> {
                Ok("".into())
            }
            async fn embed(&self, _t: &str) -> Result<Vec<f32>> {
                Ok(vec![0.0])
            }
            async fn embed_batch(&self, _t: &[String]) -> Result<Vec<Vec<f32>>> {
                Ok(vec![])
            }
            async fn extract_keywords(&self, _c: &str) -> Result<Vec<String>> {
                Ok(vec![])
            }
            async fn summarize(&self, _c: &str, _l: Option<usize>) -> Result<String> {
                Ok(String::new())
            }
            async fn health_check(&self) -> Result<bool> {
                Ok(true)
            }
            async fn extract_structured_facts(&self, _p: &str) -> Result<StructuredFactExtraction> {
                Ok(StructuredFactExtraction { facts: vec![] })
            }
            async fn extract_detailed_facts(&self, _p: &str) -> Result<DetailedFactExtraction> {
                Ok(DetailedFactExtraction { facts: vec![] })
            }
            async fn extract_keywords_structured(&self, _p: &str) -> Result<KeywordExtraction> {
                Ok(KeywordExtraction { keywords: vec![] })
            }
            async fn classify_memory(&self, _p: &str) -> Result<MemoryClassification> {
                Ok(MemoryClassification {
                    memory_type: "Factual".into(),
                    confidence: 0.0,
                    reasoning: String::new(),
                })
            }
            async fn score_importance(&self, _p: &str) -> Result<ImportanceScore> {
                Ok(ImportanceScore {
                    score: 0.0,
                    reasoning: String::new(),
                })
            }
            async fn check_duplicates(&self, _p: &str) -> Result<DeduplicationResult> {
                Ok(DeduplicationResult {
                    is_duplicate: false,
                    similarity_score: 0.0,
                    original_memory_id: None,
                })
            }
            async fn generate_summary(&self, _p: &str) -> Result<SummaryResult> {
                Ok(SummaryResult {
                    summary: String::new(),
                    key_points: vec![],
                })
            }
            async fn detect_language(&self, _p: &str) -> Result<LanguageDetection> {
                Ok(LanguageDetection {
                    language: "en".into(),
                    confidence: 1.0,
                })
            }
            async fn extract_entities(&self, _p: &str) -> Result<EntityExtraction> {
                Ok(EntityExtraction { entities: vec![] })
            }
            async fn analyze_conversation(&self, _p: &str) -> Result<ConversationAnalysis> {
                Ok(ConversationAnalysis {
                    topics: vec![],
                    sentiment: String::new(),
                    user_intent: String::new(),
                    key_information: vec![],
                })
            }
            async fn extract_metadata_enrichment(&self, _p: &str) -> Result<MetadataEnrichment> {
                Ok(MetadataEnrichment {
                    summary: "mock".into(),
                    keywords: vec![],
                })
            }
            async fn extract_metadata_enrichment_batch(
                &self,
                _t: &[String],
            ) -> Result<Vec<Result<MetadataEnrichment>>> {
                Ok(vec![])
            }
            async fn complete_batch(&self, _p: &[String]) -> Result<Vec<Result<String>>> {
                Ok(vec![])
            }
            fn get_status(&self) -> ClientStatus {
                ClientStatus::default()
            }
            fn batch_config(&self) -> (usize, u32) {
                (10, 4096)
            }
            async fn enhance_memory_unified(&self, _p: &str) -> Result<MemoryEnhancement> {
                Ok(MemoryEnhancement {
                    memory_type: "Semantic".into(),
                    summary: String::new(),
                    keywords: vec![],
                    entities: vec![],
                    topics: vec![],
                })
            }
            async fn describe_image(&self, _b: &[u8], _m: &str) -> Result<String> {
                Ok("".into())
            }
        }

        let advisor = LLMStrategyAdvisor::new(&MalformedMock);
        let (doc, meta) = advisor.fallback_parse("raw text", "unknown").await.unwrap();
        assert_eq!(
            meta.custom.get("llm_summary").unwrap(),
            "Not valid JSON at all"
        );
        assert_eq!(meta.custom.get("llm_content_type").unwrap(), "other");
        assert!(meta.custom.get("llm_title").unwrap().is_empty());
        match &doc {
            DocumentNode::Document { children, .. } => {
                assert!(
                    children
                        .iter()
                        .any(|c| matches!(c, DocumentNode::Raw { .. }))
                );
                assert!(
                    children
                        .iter()
                        .any(|c| matches!(c, DocumentNode::Paragraph { .. }))
                );
            }
            _ => panic!("Expected Document"),
        }
    }
}
