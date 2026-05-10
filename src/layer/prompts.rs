use crate::types::Memory;

/// Generates the prompt for creating an L1 structural abstraction
pub fn build_l1_prompt(memory: &Memory) -> String {
    let content = memory
        .content
        .as_deref()
        .unwrap_or("[No content available]");

    format!(
        r#"You are creating a structural abstraction of the following content.

SOURCE MEMORY (L0):
{}

TASK: Generate a concise summary that:
1. Captures the main topic in 1-2 sentences
2. Identifies the document structure (if applicable): chapter, section, subsection
3. Notes any key entities mentioned

OUTPUT FORMAT: Return exactly a valid JSON object matching this schema:
{{
  "summary": "2-3 sentence summary",
  "structure_type": "chunk|section|chapter|document",
  "key_entities": ["entity1", "entity2"],
  "suggested_title": "Brief descriptive title",
  "confidence": 0.95
}}
"#,
        content
    )
}

/// Generates a retry prompt when L1 JSON parsing failed, including the original
/// task, the malformed response, and a description of the parse error.
pub fn build_l1_retry_prompt(
    memory: &Memory,
    previous_response: &str,
    parse_error: &str,
) -> String {
    let content = memory
        .content
        .as_deref()
        .unwrap_or("[No content available]");

    format!(
        r#"Your previous response could not be parsed as valid JSON. Please fix the output and try again.

PARSE ERROR: {}

YOUR PREVIOUS RESPONSE (MALFORMED):
{}

ORIGINAL TASK:
You are creating a structural abstraction of the following content.

SOURCE MEMORY (L0):
{}

TASK: Generate a concise summary that:
1. Captures the main topic in 1-2 sentences
2. Identifies the document structure (if applicable): chapter, section, subsection
3. Notes any key entities mentioned

OUTPUT FORMAT: Return ONLY a valid JSON object with no surrounding text, matching this EXACT schema:
{{
  "summary": "2-3 sentence summary",
  "structure_type": "chunk|section|chapter|document",
  "key_entities": ["entity1", "entity2"],
  "suggested_title": "Brief descriptive title",
  "confidence": 0.95
}}

IMPORTANT: Return ONLY the JSON object. Do NOT wrap it in markdown code fences. Do NOT include any text before or after the JSON. Ensure all strings are properly closed and the JSON is complete.
"#,
        parse_error,
        previous_response,
        content
    )
}

/// Generates the prompt for creating an L2 semantic abstraction from multiple L1 memories
pub fn build_l2_prompt(memories: &[&Memory]) -> String {
    let mut combined_content = String::new();
    for (i, m) in memories.iter().enumerate() {
        let content = m.content.as_deref().unwrap_or("[Empty]");
        combined_content.push_str(&format!("MEMORY {}:\n{}\n\n", i + 1, content));
    }

    format!(
        r#"You are synthesizing several L1 summaries to create an L2 semantic abstraction. Look for connections and themes across these memories.

SOURCE L1 MEMORIES:
{}
TASK: Generate a meaningful semantic synthesis that:
1. Identifies the overarching theme or conclusion across these memories.
2. Extracts facts or assertions that span multiple memories.
3. Groups related entities together.

OUTPUT FORMAT: Return ONLY a valid JSON object (no markdown fences, no surrounding text) matching this EXACT schema:
{{
  "synthesis": "A coherent synthesis paragraph",
  "theme": "The main theme connecting them",
  "shared_entities": ["entity1", "entity2"],
  "confidence": 0.85
}}
IMPORTANT: Return ONLY the JSON. No markdown fences. No surrounding text. Ensure all commas and braces are correct.
"#,
        combined_content
    )
}

/// Generates a retry prompt for L2 JSON parsing failures.
pub fn build_l2_retry_prompt(
    memories: &[&Memory],
    previous_response: &str,
) -> String {
    let mut combined_content = String::new();
    for (i, m) in memories.iter().enumerate() {
        let content = m.content.as_deref().unwrap_or("[Empty]");
        combined_content.push_str(&format!("MEMORY {}:\n{}\n\n", i + 1, content));
    }

    format!(
        r#"Your previous response could not be parsed as valid JSON. Please fix the output and try again.

YOUR PREVIOUS RESPONSE (MALFORMED):
{}

ORIGINAL TASK:
You are synthesizing several L1 summaries to create an L2 semantic abstraction.

SOURCE L1 MEMORIES:
{}

OUTPUT FORMAT: Return ONLY a valid JSON object (no markdown fences, no surrounding text) matching this EXACT schema:
{{
  "synthesis": "A coherent synthesis paragraph",
  "theme": "The main theme connecting them",
  "shared_entities": ["entity1", "entity2"],
  "confidence": 0.85
}}
IMPORTANT: Return ONLY the JSON. No markdown fences. No surrounding text. Ensure all commas and braces are correct. Close all strings properly.
"#,
        previous_response,
        combined_content
    )
}

/// Generates the prompt for creating an L3 conceptual abstraction from multiple L2 memories
pub fn build_l3_prompt(memories: &[&Memory]) -> String {
    let mut combined_content = String::new();
    for (i, m) in memories.iter().enumerate() {
        let content = m.content.as_deref().unwrap_or("[Empty]");
        combined_content.push_str(&format!("MEMORY {}:\n{}\n\n", i + 1, content));
    }

    format!(
        r#"You are analyzing high-level L2 thematic topics to extract core philosophical themes, user mental models, or universal concepts (L3 abstraction).

SOURCE L2 THEMES:
{}
TASK: Generate a profound conceptual insight that captures:
1. The global conceptual or philosophical takeaway.
2. An abstraction that explains the deeper "why" or "how" behind these themes.
3. Long-term actionable insights or universal facts.

OUTPUT FORMAT: Return ONLY a valid JSON object (no markdown fences, no surrounding text) matching this EXACT schema:
{{
  "insight": "A profound insight paragraph",
  "concept": "The universal concept or mental model name",
  "implications": ["implication 1", "implication 2"],
  "confidence": 0.80
}}
IMPORTANT: Return ONLY the JSON. No markdown fences. No surrounding text. Ensure all commas and braces are correct.
"#,
        combined_content
    )
}

/// Generates a retry prompt for L3 JSON parsing failures.
pub fn build_l3_retry_prompt(
    memories: &[&Memory],
    previous_response: &str,
) -> String {
    let mut combined_content = String::new();
    for (i, m) in memories.iter().enumerate() {
        let content = m.content.as_deref().unwrap_or("[Empty]");
        combined_content.push_str(&format!("MEMORY {}:\n{}\n\n", i + 1, content));
    }

    format!(
        r#"Your previous response could not be parsed as valid JSON. Please fix the output and try again.

YOUR PREVIOUS RESPONSE (MALFORMED):
{}

ORIGINAL TASK:
You are analyzing high-level L2 thematic topics to extract core philosophical themes, user mental models, or universal concepts (L3 abstraction).

SOURCE L2 THEMES:
{}

OUTPUT FORMAT: Return ONLY a valid JSON object (no markdown fences, no surrounding text) matching this EXACT schema:
{{
  "insight": "A profound insight paragraph",
  "concept": "The universal concept or mental model name",
  "implications": ["implication 1", "implication 2"],
  "confidence": 0.80
}}
IMPORTANT: Return ONLY the JSON. No markdown fences. No surrounding text. Ensure all commas and braces are correct. Close all strings properly.
"#,
        previous_response,
        combined_content
    )
}
