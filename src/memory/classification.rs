use crate::{MemoryError, error::Result, llm::LLMClient, types::MemoryType};
use async_trait::async_trait;
use tracing::debug;

/// Strip XML-style tags (e.g., <think>...</think>) from text
fn strip_llm_tags(text: &str, tags: &[String]) -> String {
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

/// Trait for classifying memory types
#[async_trait]
pub trait MemoryClassifier: Send + Sync {
    /// Classify the type of a memory based on its content
    async fn classify_memory(&self, content: &str) -> Result<MemoryType>;

    /// Classify multiple memories in batch
    async fn classify_batch(&self, contents: &[String]) -> Result<Vec<MemoryType>>;

    /// Extract entities from memory content
    async fn extract_entities(&self, content: &str) -> Result<Vec<String>>;

    /// Extract topics from memory content
    async fn extract_topics(&self, content: &str) -> Result<Vec<String>>;
}

/// LLM-based memory classifier
pub struct LLMMemoryClassifier {
    llm_client: Box<dyn LLMClient>,
}

impl LLMMemoryClassifier {
    pub fn new(llm_client: Box<dyn LLMClient>) -> Self {
        Self { llm_client }
    }

    fn create_classification_prompt(&self, content: &str) -> String {
        format!(
            r#"Classify the following memory content into one of these categories:

1. Conversational - Dialogue, conversations, or interactive exchanges
2. Procedural - Instructions, how-to information, or step-by-step processes
3. Factual - Objective facts, data, or verifiable information
4. Semantic - Concepts, meanings, definitions, or general knowledge
5. Episodic - Specific events, experiences, or temporal information
6. Personal - Personal preferences, characteristics, or individual-specific information

Content: "{}"

Respond with only the category name (e.g., "Conversational", "Procedural", etc.):"#,
            content
        )
    }

    fn create_entity_extraction_prompt(&self, content: &str) -> String {
        format!(
            r#"Extract named entities from the following text. Focus on:
- People (names, roles, titles)
- Organizations (companies, institutions)
- Locations (cities, countries, places)
- Products (software, tools, brands)
- Concepts (technical terms, important keywords)

Text: "{}"

Return the entities as a comma-separated list. If no entities found, return "None"."#,
            content
        )
    }

    fn create_topic_extraction_prompt(&self, content: &str) -> String {
        format!(
            r#"Extract the main topics or themes from the following text. Focus on:
- Subject areas (technology, business, health, etc.)
- Activities (programming, cooking, traveling, etc.)
- Domains (AI, finance, education, etc.)
- Key themes or concepts

Text: "{}"

Return the topics as a comma-separated list. If no clear topics, return "None"."#,
            content
        )
    }

    fn parse_list_response(&self, response: &str) -> Vec<String> {
        // Strip XML tags (e.g., <think>...</think>) before parsing
        let cleaned = strip_llm_tags(response, &["think".to_string(), "reason".to_string()]);

        if cleaned.trim().to_lowercase() == "none" {
            return Vec::new();
        }

        cleaned
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }
}

#[async_trait]
impl MemoryClassifier for LLMMemoryClassifier {
    async fn classify_memory(&self, content: &str) -> Result<MemoryType> {
        let prompt = self.create_classification_prompt(content);

        match self.llm_client.classify_memory(&prompt).await {
            Ok(classification) => {
                let memory_type = MemoryType::parse(&classification.memory_type);
                Ok(memory_type)
            }
            Err(e) => {
                debug!(
                    "Rig extractor failed, falling back to traditional method: {}",
                    e
                );

                #[cfg(debug_assertions)]
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;

                let response = self.llm_client.complete(&prompt).await?;
                Ok(MemoryType::parse(&response))
            }
        }
    }

    async fn classify_batch(&self, contents: &[String]) -> Result<Vec<MemoryType>> {
        let mut results = Vec::with_capacity(contents.len());
        for content in contents {
            let memory_type = self.classify_memory(content).await?;
            results.push(memory_type);
        }
        Ok(results)
    }

    async fn extract_entities(&self, content: &str) -> Result<Vec<String>> {
        let prompt = self.create_entity_extraction_prompt(content);

        match self.llm_client.extract_entities(&prompt).await {
            Ok(entity_extraction) => {
                let entities: Vec<String> = entity_extraction
                    .entities
                    .into_iter()
                    .map(|entity| entity.text)
                    .collect();
                Ok(entities)
            }
            Err(e) => {
                debug!(
                    "Rig extractor failed, falling back to traditional method: {}",
                    e
                );
                #[cfg(debug_assertions)]
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;

                let response = self.llm_client.complete(&prompt).await?;
                Ok(self.parse_list_response(&response))
            }
        }
    }

    async fn extract_topics(&self, content: &str) -> Result<Vec<String>> {
        let prompt = self.create_topic_extraction_prompt(content);

        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        let response = self.llm_client.complete(&prompt).await?;
        Ok(self.parse_list_response(&response))
    }
}

/// Rule-based memory classifier for faster processing
pub struct RuleBasedMemoryClassifier;

impl Default for RuleBasedMemoryClassifier {
    fn default() -> Self {
        Self
    }
}

impl RuleBasedMemoryClassifier {
    pub fn new() -> Self {
        Self
    }

    fn classify_by_keywords(&self, content: &str) -> Option<MemoryType> {
        let content_lower = content.to_lowercase();

        let personal_keywords = [
            "i like",
            "i prefer",
            "my name",
            "i am",
            "i work",
            "i live",
            "my favorite",
            "i hate",
            "i love",
            "my birthday",
            "my phone",
            "my email",
            "my address",
            "i want",
            "i need",
            "i think",
        ];

        let procedural_keywords = [
            "how to",
            "step",
            "first",
            "then",
            "next",
            "finally",
            "instructions",
            "procedure",
            "process",
            "method",
            "way to",
            "tutorial",
            "guide",
            "recipe",
            "algorithm",
        ];

        let factual_keywords = [
            "fact",
            "data",
            "statistics",
            "number",
            "date",
            "time",
            "location",
            "address",
            "phone",
            "email",
            "website",
            "price",
            "cost",
            "amount",
        ];

        let episodic_keywords = [
            "yesterday",
            "today",
            "tomorrow",
            "last week",
            "next month",
            "happened",
            "occurred",
            "event",
            "meeting",
            "appointment",
            "remember when",
            "that time",
            "experience",
            "story",
        ];

        let semantic_keywords = [
            "definition",
            "meaning",
            "concept",
            "theory",
            "principle",
            "knowledge",
            "understanding",
            "explanation",
            "describes",
            "refers to",
            "means",
            "is defined as",
        ];

        if personal_keywords
            .iter()
            .any(|&keyword| content_lower.contains(keyword))
        {
            return Some(MemoryType::Personal);
        }

        if procedural_keywords
            .iter()
            .any(|&keyword| content_lower.contains(keyword))
        {
            return Some(MemoryType::Procedural);
        }

        if episodic_keywords
            .iter()
            .any(|&keyword| content_lower.contains(keyword))
        {
            return Some(MemoryType::Episodic);
        }

        if factual_keywords
            .iter()
            .any(|&keyword| content_lower.contains(keyword))
        {
            return Some(MemoryType::Factual);
        }

        if semantic_keywords
            .iter()
            .any(|&keyword| content_lower.contains(keyword))
        {
            return Some(MemoryType::Semantic);
        }

        None
    }

    fn extract_simple_entities(&self, content: &str) -> Vec<String> {
        let mut entities = Vec::new();

        let words: Vec<&str> = content.split_whitespace().collect();

        for word in words {
            if word.len() > 2 && word.chars().next().unwrap().is_uppercase() {
                let clean_word = word.trim_matches(|c: char| !c.is_alphabetic());
                if !clean_word.is_empty() && clean_word.len() > 2 {
                    entities.push(clean_word.to_string());
                }
            }
        }

        entities.sort();
        entities.dedup();
        entities
    }

    fn extract_simple_topics(&self, content: &str) -> Vec<String> {
        let mut topics = Vec::new();
        let content_lower = content.to_lowercase();

        let tech_keywords = [
            "programming",
            "software",
            "computer",
            "ai",
            "machine learning",
            "database",
        ];
        if tech_keywords
            .iter()
            .any(|&keyword| content_lower.contains(keyword))
        {
            topics.push("Technology".to_string());
        }

        let business_keywords = [
            "business", "company", "meeting", "project", "work", "office",
        ];
        if business_keywords
            .iter()
            .any(|&keyword| content_lower.contains(keyword))
        {
            topics.push("Business".to_string());
        }

        let personal_keywords = ["family", "friend", "hobby", "interest", "personal"];
        if personal_keywords
            .iter()
            .any(|&keyword| content_lower.contains(keyword))
        {
            topics.push("Personal".to_string());
        }

        let health_keywords = ["health", "medical", "doctor", "medicine", "exercise"];
        if health_keywords
            .iter()
            .any(|&keyword| content_lower.contains(keyword))
        {
            topics.push("Health".to_string());
        }

        topics
    }
}

#[async_trait]
impl MemoryClassifier for RuleBasedMemoryClassifier {
    async fn classify_memory(&self, content: &str) -> Result<MemoryType> {
        self.classify_by_keywords(content)
            .ok_or(MemoryError::NotFound { id: "".to_owned() })
    }

    async fn classify_batch(&self, contents: &[String]) -> Result<Vec<MemoryType>> {
        let mut results = Vec::with_capacity(contents.len());
        for content in contents {
            let memory_type = self
                .classify_by_keywords(content)
                .ok_or(MemoryError::NotFound { id: "".to_owned() })?;
            results.push(memory_type);
        }
        Ok(results)
    }

    async fn extract_entities(&self, content: &str) -> Result<Vec<String>> {
        Ok(self.extract_simple_entities(content))
    }

    async fn extract_topics(&self, content: &str) -> Result<Vec<String>> {
        Ok(self.extract_simple_topics(content))
    }
}

/// Hybrid classifier that combines LLM and rule-based approaches
pub struct HybridMemoryClassifier {
    llm_classifier: LLMMemoryClassifier,
    rule_classifier: RuleBasedMemoryClassifier,
    use_llm_threshold: usize,
}

impl HybridMemoryClassifier {
    pub fn new(llm_client: Box<dyn LLMClient>, use_llm_threshold: usize) -> Self {
        Self {
            llm_classifier: LLMMemoryClassifier::new(llm_client),
            rule_classifier: RuleBasedMemoryClassifier::new(),
            use_llm_threshold,
        }
    }
}

#[async_trait]
impl MemoryClassifier for HybridMemoryClassifier {
    async fn classify_memory(&self, content: &str) -> Result<MemoryType> {
        if content.len() > self.use_llm_threshold {
            self.llm_classifier.classify_memory(content).await
        } else {
            self.rule_classifier.classify_memory(content).await
        }
    }

    async fn classify_batch(&self, contents: &[String]) -> Result<Vec<MemoryType>> {
        let mut results = Vec::with_capacity(contents.len());
        for content in contents {
            let memory_type = self.classify_memory(content).await?;
            results.push(memory_type);
        }
        Ok(results)
    }

    async fn extract_entities(&self, content: &str) -> Result<Vec<String>> {
        if content.len() > self.use_llm_threshold {
            self.llm_classifier.extract_entities(content).await
        } else {
            self.rule_classifier.extract_entities(content).await
        }
    }

    async fn extract_topics(&self, content: &str) -> Result<Vec<String>> {
        if content.len() > self.use_llm_threshold {
            self.llm_classifier.extract_topics(content).await
        } else {
            self.rule_classifier.extract_topics(content).await
        }
    }
}

/// Factory function to create memory classifiers
pub fn create_memory_classifier(
    llm_client: Box<dyn LLMClient>,
    use_llm: bool,
    hybrid_threshold: Option<usize>,
) -> Box<dyn MemoryClassifier> {
    match (use_llm, hybrid_threshold) {
        (true, Some(threshold)) => Box::new(HybridMemoryClassifier::new(llm_client, threshold)),
        (true, None) => Box::new(LLMMemoryClassifier::new(llm_client)),
        (false, _) => Box::new(RuleBasedMemoryClassifier::new()),
    }
}
