use crate::types::Memory;

/// Generates the prompt for creating an L1 structural abstraction
pub fn build_l1_prompt(memory: &Memory) -> String {
    let content = memory.content.as_deref().unwrap_or("[No content available]");
    
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
"#, content)
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

OUTPUT FORMAT: Return exactly a valid JSON object matching this schema:
{{
  "synthesis": "A coherent synthesis paragraph",
  "theme": "The main theme connecting them",
  "shared_entities": ["entity1", "entity2"],
  "confidence": 0.85
}}
"#, combined_content)
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

OUTPUT FORMAT: Return exactly a valid JSON object matching this schema:
{{
  "insight": "A profound insight paragraph",
  "concept": "The universal concept or mental model name",
  "implications": ["implication 1", "implication 2"],
  "confidence": 0.80
}}
"#, combined_content)
}
