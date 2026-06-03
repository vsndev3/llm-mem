use rmcp::model::Prompt;
use serde_json::Map;

use super::{PromptResult, arg, assistant_text, ok_result, user_text};

pub fn prompt_def() -> Prompt {
    super::make_prompt(
        "explore_knowledge",
        "Explore what the memory system knows about a topic. Searches notes and summary \
         cards, follows links between them, and builds a picture of connected knowledge. \
         Use this to answer questions or find gaps in stored knowledge.",
        vec![
            arg("query", "The topic or question to explore", true),
            arg("bank", "Memory bank name (default: 'default')", false),
            arg("depth", "How deep to follow links: 1 (quick) to 5 (deep), default 3", false),
        ],
    )
}

pub fn get(arguments: Option<&Map<String, serde_json::Value>>) -> PromptResult {
    let query = arguments
        .and_then(|a| a.get("query"))
        .and_then(|v| v.as_str())
        .unwrap_or("general knowledge");

    let bank = arguments
        .and_then(|a| a.get("bank"))
        .and_then(|v| v.as_str())
        .unwrap_or("default");

    let depth = arguments
        .and_then(|a| a.get("depth"))
        .and_then(|v| v.as_i64())
        .unwrap_or(3);

    let description = format!("Explore knowledge about: {} (depth: {})", query, depth);

    let messages = vec![
        user_text(format!(
            "Explore what is stored about: **\"{query}\"**\n\
             Bank: `{bank}` | Depth: {depth}\n\n\
             **Step 1 — Search across all layers**\n\
             Call `search_memory`:\n\
             ```json\n\
             {{\"query\": \"{query}\", \"k\": 15, \"bank\": \"{bank}\"}}\n\
             ```\n\
             Look at every result. Note whether each is a raw note or a summary card.\n\n\
             **Step 2 — Follow the links**\n\
             Pick the top 3 results. For each, call `navigate_memory`:\n\
             ```json\n\
             {{\"memory_id\": \"<id>\", \"direction\": \"both\", \"levels\": {depth}, \"bank\": \"{bank}\"}}\n\
             ```\n\
             This shows what each note is connected to — related notes, summaries, and concepts.\n\n\
             **Step 3 — Report what you found**\n\
             Write a short answer to: \"{query}\"\n\
             Include:\n\
             - Direct facts that answer the question\n\
             - Related concepts you discovered through links\n\
             - What is MISSING — what should be stored to answer better?"
        )),
        assistant_text(format!(
            "Exploring: \"{query}\"\n\n\
             Step 1 — searching:\n\
             ```json\n\
             {{\"query\": \"{query}\", \"k\": 15, \"bank\": \"{bank}\"}}\n\
             ```"
        )),
        user_text(
            "Finish all 3 steps. If search returns nothing, try a broader query. \
             Focus on understanding the connections, not just finding answers.",
        ),
    ];

    ok_result(messages, description)
}
