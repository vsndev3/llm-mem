use crate::types::Memory;

/// Safely truncate a string to at most `max_chars` characters respecting UTF-8 boundaries.
fn safe_truncate_with_note(content: &str, max_chars: usize) -> String {
    if content.len() <= max_chars {
        return content.to_string();
    }
    let byte_at = content
        .char_indices()
        .take(max_chars)
        .last()
        .map(|(idx, c)| idx + c.len_utf8())
        .unwrap_or(0);
    format!(
        "{}...\n[content truncated, {} total chars]",
        &content[..byte_at],
        content.len()
    )
}

/// Context information to enrich the L1 abstraction prompt
pub struct L1Context<'a> {
    pub file_name: Option<&'a str>,
    pub chunk_index: Option<usize>,
    pub total_chunks: Option<usize>,
    pub section_headers: &'a [String],
}

/// Generates the prompt for creating an L1 structural abstraction
pub fn build_l1_prompt(memory: &Memory, context: &L1Context) -> String {
    let content = memory
        .content
        .as_deref()
        .unwrap_or("[No content available]");

    let trimmed = safe_truncate_with_note(content, 3000);

    let mut context_line = String::new();
    if let Some(name) = context.file_name {
        context_line.push_str(&format!("Document: {}\n", name));
    }
    if let (Some(idx), Some(total)) = (context.chunk_index, context.total_chunks) {
        context_line.push_str(&format!("Chunk: {} of {}\n", idx + 1, total));
    }
    if !context.section_headers.is_empty() {
        context_line.push_str("Section path: ");
        context_line.push_str(&context.section_headers.join(" > "));
        context_line.push('\n');
    }

    let section_guide = if !context.section_headers.is_empty() {
        format!(
            "This chunk falls under the section: \"{}\". Use this for contextualizing the summary.\n",
            context.section_headers.join(" > ")
        )
    } else {
        String::new()
    };

    format!(
        r#"You are analyzing a chunk from a document and creating a concise, informative structural summary.

DOCUMENT CONTEXT:
{context_line}
SOURCE CONTENT:
{trimmed}

TASK: Write a focused summary (2-4 sentences) that:
1. Captures the SPECIFIC information in this chunk — not generic statements about what a document is
2. Identifies the key concepts, definitions, or facts presented
3. Notes how this chunk fits into the document structure (e.g., "Part of the explanation of [topic]")

{section_guide}
GUIDELINES FOR A GOOD SUMMARY:
- Be specific and concrete — use actual terms from the content
- Avoid meta-commentary like "This document provides" or "This text describes"
- If the chunk defines a term, include the definition
- If the chunk presents facts, capture the key facts
- Keep it under 200 words

OUTPUT FORMAT: Return exactly a valid JSON object matching this schema:
{{
  "summary": "Specific, concrete summary of this chunk's content",
  "structure_type": "chunk|section|chapter|document",
  "key_entities": ["entity1", "entity2"],
  "suggested_title": "Brief descriptive title matching the content",
  "confidence": 0.95
}}

IMPORTANT: Return ONLY the JSON object. No markdown fences. No surrounding text. Ensure all strings are properly closed.
"#,
    )
}

/// Generates a retry prompt when L1 JSON parsing failed
pub fn build_l1_retry_prompt(
    memory: &Memory,
    _context: &L1Context,
    previous_response: &str,
    parse_error: &str,
) -> String {
    let content = memory
        .content
        .as_deref()
        .unwrap_or("[No content available]");

    let trimmed = safe_truncate_with_note(content, 3000);

    format!(
        r#"Your previous response could not be parsed as valid JSON. Please fix the output and try again.

PARSE ERROR: {}

YOUR PREVIOUS RESPONSE (MALFORMED):
{}

ORIGINAL TASK:
You are analyzing a document chunk and creating a concise, informative structural summary.

SOURCE CONTENT:
{}

TASK: Write a focused summary (2-4 sentences) that:
1. Captures the SPECIFIC information in this chunk — not generic statements
2. Identifies the key concepts, definitions, or facts presented

GUIDELINES:
- Be specific and concrete — use actual terms from the content
- Avoid meta-commentary like "This document provides"
- Keep it under 200 words

OUTPUT FORMAT: Return ONLY a valid JSON object with no surrounding text:
{{
  "summary": "Specific, concrete summary of this chunk's content",
  "structure_type": "chunk|section|chapter|document",
  "key_entities": ["entity1", "entity2"],
  "suggested_title": "Brief descriptive title matching the content",
  "confidence": 0.95
}}

IMPORTANT: Return ONLY the JSON object. Do NOT wrap it in markdown code fences. Do NOT include any text before or after the JSON.
"#,
        parse_error, previous_response, trimmed
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
pub fn build_l2_retry_prompt(memories: &[&Memory], previous_response: &str) -> String {
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
        previous_response, combined_content
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
pub fn build_l3_retry_prompt(memories: &[&Memory], previous_response: &str) -> String {
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
        previous_response, combined_content
    )
}
