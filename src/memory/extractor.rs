use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::{
    error::Result,
    llm::{DetailedFactExtraction, LLMClient, StructuredFactExtraction},
    memory::utils::{
        LanguageInfo, detect_language, filter_messages_by_role, filter_messages_by_roles,
        parse_messages, remove_code_blocks,
    },
    types::Message,
};

/// Extracted fact from conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    pub content: String,
    pub importance: f32,
    pub category: FactCategory,
    pub entities: Vec<String>,
    pub language: Option<LanguageInfo>,
    pub source_role: String,
}

/// Categories of facts that can be extracted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactCategory {
    Personal,
    Preference,
    Factual,
    Procedural,
    Contextual,
}

/// Metadata enriched from a document chunk
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ChunkMetadata {
    pub summary: String,
    pub keywords: Vec<String>,
}

/// Extraction strategy based on conversation analysis
#[derive(Debug, Clone)]
pub enum ExtractionStrategy {
    DualChannel,
    UserOnly,
    AssistantOnly,
    ProceduralMemory,
}

/// Trait for fact extraction from conversations
#[async_trait]
pub trait FactExtractor: Send + Sync {
    /// Extract facts from a conversation with enhanced dual prompt system
    async fn extract_facts(&self, messages: &[Message]) -> Result<Vec<ExtractedFact>>;

    /// Extract user-only facts
    async fn extract_user_facts(&self, messages: &[Message]) -> Result<Vec<ExtractedFact>>;

    /// Extract assistant-only facts
    async fn extract_assistant_facts(&self, messages: &[Message]) -> Result<Vec<ExtractedFact>>;

    /// Extract facts from a single text with language detection
    async fn extract_facts_from_text(&self, text: &str) -> Result<Vec<ExtractedFact>>;

    /// Extract facts from filtered messages (only specific roles)
    async fn extract_facts_filtered(
        &self,
        messages: &[Message],
        allowed_roles: &[&str],
    ) -> Result<Vec<ExtractedFact>>;

    /// Extract only meaningful assistant facts that contain user-relevant information
    async fn extract_meaningful_assistant_facts(
        &self,
        messages: &[Message],
    ) -> Result<Vec<ExtractedFact>>;

    /// Extract metadata (summary and keywords) from a text chunk
    async fn extract_metadata_enrichment(&self, text: &str) -> Result<ChunkMetadata>;
}

/// LLM-based fact extractor implementation
pub struct LLMFactExtractor {
    llm_client: Box<dyn LLMClient>,
}

impl LLMFactExtractor {
    pub fn new(llm_client: Box<dyn LLMClient>) -> Self {
        Self { llm_client }
    }

    fn build_user_memory_prompt(&self, messages: &[Message]) -> String {
        let current_date = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let conversation = parse_messages(messages);

        format!(
            r#"You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences.
Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts.

# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE USER'S MESSAGES. DO NOT INCLUDE INFORMATION FROM ASSISTANT OR SYSTEM MESSAGES.

Types of Information to Remember:
1. Personal Preferences
2. Important Personal Details
3. Plans and Intentions
4. Activity and Service Preferences
5. Health and Wellness Preferences
6. Professional Details
7. Miscellaneous Information

Return the facts in the following JSON format:
{{
  "facts": ["fact 1", "fact 2", "fact 3"]
}}

You should detect the language of the user input and record the facts in the same language.

Remember:
- Today's date is {current_date}.
- If you do not find anything relevant, return {{"facts": []}}.
- Make sure to return valid JSON only, no additional text.

Conversation:
{}

JSON Response:"#,
            conversation
        )
    }

    fn build_user_focused_assistant_prompt(&self, messages: &[Message]) -> String {
        let current_date = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let conversation = parse_messages(messages);

        format!(
            r#"You are a Strict Personal Information Filter, specialized in extracting ONLY direct facts about the USER from assistant responses.

# EXTRACT ONLY (must meet ALL criteria):
- Direct user preferences explicitly stated by the user
- User's background, interests, or situation explicitly mentioned
- User's specific needs or requests clearly stated

# DO NOT EXTRACT:
- Technical explanations
- Suggestions or recommendations
- Educational content
- Information about the assistant's capabilities

Return only direct user facts in the following JSON format:
{{
  "facts": ["fact 1", "fact 2", "fact 3"]
}}

If no direct user facts exist, return {{"facts": []}}.

Remember:
- Today's date is {current_date}.
- Make sure to return valid JSON only, no additional text.

Conversation:
{}

JSON Response:"#,
            conversation
        )
    }

    fn build_assistant_memory_prompt(&self, messages: &[Message]) -> String {
        let current_date = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let conversation = parse_messages(messages);

        format!(
            r#"You are an Assistant Information Organizer, specialized in accurately storing facts about the AI assistant from conversations.

# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE ASSISTANT'S MESSAGES.

Return the facts in the following JSON format:
{{
  "facts": ["fact 1", "fact 2", "fact 3"]
}}

Remember:
- Today's date is {current_date}.
- If you do not find anything relevant, return {{"facts": []}}.
- Make sure to return valid JSON only, no additional text.

Conversation:
{}

JSON Response:"#,
            conversation
        )
    }

    fn build_conversation_extraction_prompt(&self, messages: &[Message]) -> String {
        let conversation = messages
            .iter()
            .map(|msg| format!("{}: {}", msg.role, msg.content))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"Extract important facts from the following conversation. Focus on:
1. Personal information
2. Factual statements
3. Procedures and how-to information
4. Important context

Return the facts as a JSON array:
[
  {{
    "content": "Natural language description of the fact",
    "importance": 0.8,
    "category": "Personal|Preference|Factual|Procedural|Contextual",
    "entities": ["entity1", "entity2"]
  }}
]

Conversation:
{}

Facts (JSON only):"#,
            conversation
        )
    }

    fn build_text_extraction_prompt(&self, text: &str) -> String {
        format!(
            r#"Extract important facts from the following text. Focus on:
1. Key information and claims
2. Important details and specifics
3. Relationships and connections
4. Actionable information

Return the facts as a JSON array:
[
  {{
    "content": "Natural language description of the fact",
    "importance": 0.8,
    "category": "Personal|Preference|Factual|Procedural|Contextual",
    "entities": ["entity1", "entity2"]
  }}
]

Text:
{}

Facts (JSON only):"#,
            text
        )
    }

    fn build_metadata_enrichment_prompt(&self, text: &str) -> String {
        crate::memory::prompts::METADATA_ENRICHMENT_PROMPT.replace("{{text}}", text)
    }

    fn parse_structured_facts(&self, structured: StructuredFactExtraction) -> Vec<ExtractedFact> {
        let mut facts = Vec::new();
        for fact_str in structured.facts {
            let language = detect_language(&fact_str);
            facts.push(ExtractedFact {
                content: fact_str,
                importance: 0.7,
                category: FactCategory::Personal,
                entities: vec![],
                language: Some(language),
                source_role: "unknown".to_string(),
            });
        }
        facts
    }

    fn parse_detailed_facts(&self, detailed: DetailedFactExtraction) -> Vec<ExtractedFact> {
        let mut facts = Vec::new();
        for structured_fact in detailed.facts {
            let category = match structured_fact.category.as_str() {
                "Personal" => FactCategory::Personal,
                "Preference" => FactCategory::Preference,
                "Factual" => FactCategory::Factual,
                "Procedural" => FactCategory::Procedural,
                "Contextual" => FactCategory::Contextual,
                _ => FactCategory::Factual,
            };

            let language = detect_language(&structured_fact.content);
            facts.push(ExtractedFact {
                content: structured_fact.content,
                importance: structured_fact.importance,
                category,
                entities: structured_fact.entities,
                language: Some(language),
                source_role: structured_fact.source_role,
            });
        }
        facts
    }

    fn parse_facts_response_fallback(&self, response: &str) -> Result<Vec<ExtractedFact>> {
        let cleaned_response = remove_code_blocks(response);

        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&cleaned_response) {
            if let Some(facts_array) = json_value.get("facts").and_then(|v| v.as_array()) {
                let mut facts = Vec::new();
                for fact_value in facts_array {
                    if let Some(fact_str) = fact_value.as_str() {
                        facts.push(ExtractedFact {
                            content: fact_str.to_string(),
                            importance: 0.7,
                            category: FactCategory::Personal,
                            entities: vec![],
                            language: Some(detect_language(fact_str)),
                            source_role: "unknown".to_string(),
                        });
                    }
                }
                return Ok(facts);
            }
        }

        Ok(vec![ExtractedFact {
            content: response.trim().to_string(),
            importance: 0.5,
            category: FactCategory::Factual,
            entities: vec![],
            language: None,
            source_role: "unknown".to_string(),
        }])
    }

    fn analyze_conversation_context(&self, messages: &[Message]) -> ExtractionStrategy {
        let mut has_user = false;
        let mut has_assistant = false;

        for msg in messages {
            match msg.role.as_str() {
                "user" => has_user = true,
                "assistant" => has_assistant = true,
                _ => {}
            }
        }

        let is_procedural = self.detect_procedural_pattern(messages);

        if is_procedural {
            ExtractionStrategy::ProceduralMemory
        } else if has_user && has_assistant {
            ExtractionStrategy::DualChannel
        } else if has_user {
            ExtractionStrategy::UserOnly
        } else if has_assistant {
            ExtractionStrategy::AssistantOnly
        } else {
            ExtractionStrategy::UserOnly
        }
    }

    fn detect_procedural_pattern(&self, messages: &[Message]) -> bool {
        let procedural_keywords = [
            "executing",
            "processing",
            "steps",
            "actions",
            "final result",
            "output",
            "continue",
        ];

        let mut has_procedural_keywords = false;
        let mut has_alternating_pattern = false;

        for message in messages {
            if message.role == "user" {
                continue;
            }
            let content_lower = message.content.to_lowercase();
            for keyword in &procedural_keywords {
                if content_lower.contains(keyword) {
                    has_procedural_keywords = true;
                    break;
                }
            }
            if has_procedural_keywords {
                break;
            }
        }

        if messages.len() >= 4 {
            let mut user_assistant_alternation = 0;
            for i in 1..messages.len() {
                if messages[i - 1].role != messages[i].role {
                    user_assistant_alternation += 1;
                }
            }
            has_alternating_pattern = user_assistant_alternation >= messages.len() / 2;
        }

        has_procedural_keywords && has_alternating_pattern
    }

    async fn extract_procedural_facts(&self, messages: &[Message]) -> Result<Vec<ExtractedFact>> {
        let mut procedural_facts = Vec::new();

        for message in messages {
            if message.role == "assistant" {
                let action_description = self.extract_action_from_message(&message.content);

                if !action_description.is_empty() {
                    procedural_facts.push(ExtractedFact {
                        content: format!("Executed: {}", action_description),
                        importance: 0.8,
                        category: FactCategory::Procedural,
                        entities: self.extract_entities_from_content(&message.content),
                        language: Some(detect_language(&message.content)),
                        source_role: "assistant".to_string(),
                    });
                }
            } else if message.role == "user" {
                procedural_facts.push(ExtractedFact {
                    content: format!("User request: {}", message.content),
                    importance: 0.6,
                    category: FactCategory::Contextual,
                    entities: self.extract_entities_from_content(&message.content),
                    language: Some(detect_language(&message.content)),
                    source_role: "user".to_string(),
                });
            }
        }

        Ok(procedural_facts)
    }

    fn extract_action_from_message(&self, content: &str) -> String {
        let chars: Vec<char> = content.chars().collect();
        let limit = chars.len().min(100);
        chars.into_iter().take(limit).collect::<String>()
    }

    fn extract_entities_from_content(&self, content: &str) -> Vec<String> {
        let mut entities = Vec::new();

        let patterns = [
            r"[A-Z][a-z]+ [A-Z][a-z]+",
            r"\b(?:http|https)://\S+",
            r"\b[A-Z]{2,}\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
        ];

        for pattern in &patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                for match_result in regex.find_iter(content) {
                    entities.push(match_result.as_str().to_string());
                }
            }
        }

        entities
    }

    async fn intelligent_fact_filtering(
        &self,
        facts: Vec<ExtractedFact>,
    ) -> Result<Vec<ExtractedFact>> {
        if facts.is_empty() {
            return Ok(facts);
        }

        let mut filtered_facts: Vec<ExtractedFact> = Vec::new();
        let mut seen_contents = std::collections::HashSet::new();

        for fact in &facts {
            let content_normalized = fact.content.to_lowercase().trim().to_string();

            if seen_contents.contains(&content_normalized) {
                debug!("Skipping duplicate fact: {}", content_normalized);
                continue;
            }

            let mut is_semantically_duplicate = false;
            for existing_fact in &filtered_facts {
                if self.are_facts_semantically_similar(&fact.content, &existing_fact.content) {
                    is_semantically_duplicate = true;
                    break;
                }
            }

            if is_semantically_duplicate {
                continue;
            }

            if fact.importance >= 0.5 {
                seen_contents.insert(content_normalized);
                filtered_facts.push(fact.clone());
            }
        }

        filtered_facts.sort_by(|a, b| {
            let category_order = |cat: &FactCategory| match cat {
                FactCategory::Personal => 4,
                FactCategory::Preference => 3,
                FactCategory::Factual => 2,
                FactCategory::Procedural => 1,
                FactCategory::Contextual => 0,
            };

            let category_cmp = category_order(&a.category).cmp(&category_order(&b.category));
            if category_cmp != std::cmp::Ordering::Equal {
                return category_cmp.reverse();
            }

            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        info!(
            "Filtered {} facts down to {} high-quality facts",
            facts.len(),
            filtered_facts.len()
        );
        Ok(filtered_facts)
    }

    fn are_facts_semantically_similar(&self, fact1: &str, fact2: &str) -> bool {
        let fact1_lower = fact1.to_lowercase();
        let fact2_lower = fact2.to_lowercase();

        if fact1_lower.trim() == fact2_lower.trim() {
            return true;
        }

        let words1: std::collections::HashSet<&str> = fact1_lower.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = fact2_lower.split_whitespace().collect();

        let intersection: std::collections::HashSet<_> = words1.intersection(&words2).collect();
        let union_size = words1.len().max(words2.len());
        let jaccard_similarity = intersection.len() as f64 / union_size as f64;

        jaccard_similarity > 0.7
    }

    fn add_source_role_to_facts(
        &self,
        mut facts: Vec<ExtractedFact>,
        source_role: &str,
    ) -> Vec<ExtractedFact> {
        for fact in &mut facts {
            fact.source_role = source_role.to_string();
        }
        facts
    }
}

#[async_trait]
impl FactExtractor for LLMFactExtractor {
    async fn extract_facts(&self, messages: &[Message]) -> Result<Vec<ExtractedFact>> {
        if messages.is_empty() {
            return Ok(vec![]);
        }

        let extraction_strategy = self.analyze_conversation_context(messages);

        let all_facts = match extraction_strategy {
            ExtractionStrategy::DualChannel => {
                let user_facts = self.extract_user_facts(messages).await?;

                let all_facts = if let Ok(assistant_facts) =
                    self.extract_meaningful_assistant_facts(messages).await
                {
                    [user_facts, assistant_facts].concat()
                } else {
                    user_facts
                };

                info!(
                    "Extracted {} facts using dual-channel strategy from {} messages",
                    all_facts.len(),
                    messages.len()
                );
                all_facts
            }
            ExtractionStrategy::UserOnly => {
                let user_facts = self.extract_user_facts(messages).await?;
                info!(
                    "Extracted {} facts using user-only strategy from {} messages",
                    user_facts.len(),
                    messages.len()
                );
                user_facts
            }
            ExtractionStrategy::AssistantOnly => {
                let assistant_facts = self.extract_assistant_facts(messages).await?;
                info!(
                    "Extracted {} facts using assistant-only strategy from {} messages",
                    assistant_facts.len(),
                    messages.len()
                );
                assistant_facts
            }
            ExtractionStrategy::ProceduralMemory => {
                let all_facts = self.extract_procedural_facts(messages).await?;
                info!(
                    "Extracted {} procedural facts from {} messages",
                    all_facts.len(),
                    messages.len()
                );
                all_facts
            }
        };

        let filtered_facts = self.intelligent_fact_filtering(all_facts).await?;
        debug!("Final extracted facts: {:?}", filtered_facts);
        Ok(filtered_facts)
    }

    async fn extract_user_facts(&self, messages: &[Message]) -> Result<Vec<ExtractedFact>> {
        if messages.is_empty() {
            return Ok(vec![]);
        }

        let user_messages = filter_messages_by_role(messages, "user");
        if user_messages.is_empty() {
            return Ok(vec![]);
        }

        let prompt = self.build_user_memory_prompt(&user_messages);

        match self.llm_client.extract_structured_facts(&prompt).await {
            Ok(structured_facts) => {
                let facts = self.parse_structured_facts(structured_facts);
                let facts_with_role = self.add_source_role_to_facts(facts, "user");
                info!(
                    "Extracted {} user facts using rig extractor",
                    facts_with_role.len()
                );
                Ok(facts_with_role)
            }
            Err(e) => {
                debug!("Rig extractor failed, falling back: {}", e);

                #[cfg(debug_assertions)]
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;

                let response = self.llm_client.complete(&prompt).await?;
                let facts = self.parse_facts_response_fallback(&response)?;
                let facts_with_role = self.add_source_role_to_facts(facts, "user");
                Ok(facts_with_role)
            }
        }
    }

    async fn extract_assistant_facts(&self, messages: &[Message]) -> Result<Vec<ExtractedFact>> {
        if messages.is_empty() {
            return Ok(vec![]);
        }

        let assistant_messages = filter_messages_by_role(messages, "assistant");
        if assistant_messages.is_empty() {
            return Ok(vec![]);
        }

        let prompt = self.build_assistant_memory_prompt(&assistant_messages);

        match self.llm_client.extract_structured_facts(&prompt).await {
            Ok(structured_facts) => {
                let facts = self.parse_structured_facts(structured_facts);
                let facts_with_role = self.add_source_role_to_facts(facts, "assistant");
                info!(
                    "Extracted {} assistant facts using rig extractor",
                    facts_with_role.len()
                );
                Ok(facts_with_role)
            }
            Err(e) => {
                debug!("Rig extractor failed, falling back: {}", e);

                #[cfg(debug_assertions)]
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;

                let response = self.llm_client.complete(&prompt).await?;
                let facts = self.parse_facts_response_fallback(&response)?;
                let facts_with_role = self.add_source_role_to_facts(facts, "assistant");
                Ok(facts_with_role)
            }
        }
    }

    async fn extract_facts_from_text(&self, text: &str) -> Result<Vec<ExtractedFact>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        let prompt = self.build_text_extraction_prompt(text);

        match self.llm_client.extract_detailed_facts(&prompt).await {
            Ok(detailed_facts) => {
                let facts = self.parse_detailed_facts(detailed_facts);
                let facts_with_language: Vec<_> = facts
                    .into_iter()
                    .map(|mut fact| {
                        fact.language = Some(detect_language(text));
                        fact
                    })
                    .collect();
                Ok(facts_with_language)
            }
            Err(e) => {
                debug!("Rig extractor failed, falling back: {}", e);

                #[cfg(debug_assertions)]
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;

                let response = self.llm_client.complete(&prompt).await?;
                let facts = self.parse_facts_response_fallback(&response)?;
                let facts_with_language: Vec<_> = facts
                    .into_iter()
                    .map(|mut fact| {
                        fact.language = Some(detect_language(text));
                        fact
                    })
                    .collect();
                Ok(facts_with_language)
            }
        }
    }

    async fn extract_facts_filtered(
        &self,
        messages: &[Message],
        allowed_roles: &[&str],
    ) -> Result<Vec<ExtractedFact>> {
        if messages.is_empty() {
            return Ok(vec![]);
        }

        let filtered_messages = filter_messages_by_roles(messages, allowed_roles);
        if filtered_messages.is_empty() {
            return Ok(vec![]);
        }

        let prompt = self.build_conversation_extraction_prompt(&filtered_messages);

        match self.llm_client.extract_detailed_facts(&prompt).await {
            Ok(detailed_facts) => {
                let facts = self.parse_detailed_facts(detailed_facts);
                let facts_with_role =
                    self.add_source_role_to_facts(facts, &allowed_roles.join(","));
                Ok(facts_with_role)
            }
            Err(e) => {
                debug!("Rig extractor failed, falling back: {}", e);

                #[cfg(debug_assertions)]
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;

                let response = self.llm_client.complete(&prompt).await?;
                let facts = self.parse_facts_response_fallback(&response)?;
                let facts_with_role =
                    self.add_source_role_to_facts(facts, &allowed_roles.join(","));
                Ok(facts_with_role)
            }
        }
    }

    async fn extract_meaningful_assistant_facts(
        &self,
        messages: &[Message],
    ) -> Result<Vec<ExtractedFact>> {
        if messages.is_empty() {
            return Ok(vec![]);
        }

        let assistant_messages = filter_messages_by_role(messages, "assistant");
        if assistant_messages.is_empty() {
            return Ok(vec![]);
        }

        let prompt = self.build_user_focused_assistant_prompt(&assistant_messages);

        match self.llm_client.extract_structured_facts(&prompt).await {
            Ok(structured_facts) => {
                let facts = self.parse_structured_facts(structured_facts);
                let facts_with_role = self.add_source_role_to_facts(facts, "assistant");
                Ok(facts_with_role)
            }
            Err(e) => {
                debug!("Rig extractor failed, falling back: {}", e);

                #[cfg(debug_assertions)]
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;

                let response = self.llm_client.complete(&prompt).await?;
                let facts = self.parse_facts_response_fallback(&response)?;
                let facts_with_role = self.add_source_role_to_facts(facts, "assistant");
                Ok(facts_with_role)
            }
        }
    }

    async fn extract_metadata_enrichment(&self, text: &str) -> Result<ChunkMetadata> {
        let prompt = self.build_metadata_enrichment_prompt(text);

        match self.llm_client.extract_metadata_enrichment(&prompt).await {
            Ok(metadata) => Ok(ChunkMetadata {
                summary: metadata.summary,
                keywords: metadata.keywords,
            }),
            Err(e) => {
                debug!("Metadata enrichment extraction failed, falling back: {}", e);

                #[cfg(debug_assertions)]
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;

                let response = self.llm_client.complete(&prompt).await?;

                // Fallback: try to parse the response as JSON manually if the structured extractor failed
                if let Some(json_str) = extract_json_from_text(&response) {
                    if let Ok(metadata) = serde_json::from_str::<ChunkMetadata>(json_str) {
                        return Ok(metadata);
                    }
                }

                Ok(ChunkMetadata {
                    summary: response.trim().to_string(),
                    keywords: vec![],
                })
            }
        }
    }
}

/// Extract a JSON object or array from text that may contain surrounding prose.
fn extract_json_from_text(text: &str) -> Option<&str> {
    let text = text.trim();

    // Strip markdown code fences if present
    let text = if text.starts_with("```json") {
        let end = text.rfind("```").unwrap_or(text.len());
        if end > 7 {
            &text[7..end]
        } else {
            &text[7..]
        }
    } else if text.starts_with("```") {
        let end = text.rfind("```").unwrap_or(text.len());
        if end > 3 {
            &text[3..end]
        } else {
            &text[3..]
        }
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

/// Factory function to create fact extractors
pub fn create_fact_extractor(llm_client: Box<dyn LLMClient>) -> Box<dyn FactExtractor + 'static> {
    Box::new(LLMFactExtractor::new(llm_client))
}
