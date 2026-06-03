use rmcp::{ErrorData, model::{GetPromptResult, Prompt}};
use serde_json::Map;

use super::{PromptResult, arg, assistant_text, make_prompt, ok_result, user_text};

pub fn prompt_def() -> Prompt {
    super::make_prompt(
        "build_memory_graph",
        "Build a mind map of connected facts, then create summary cards and higher-level \
         concepts yourself. First store and link individual notes, then group related notes \
         into summaries, and group summaries into big-picture concepts. \
         Use this to grow a rich, multi-layer knowledge web.",
        vec![
            arg("topic", "Central topic for the mind map (e.g., 'machine learning', 'project architecture')", false),
            arg("bank", "Memory bank name (default: 'default')", false),
        ],
    )
}

pub fn get(arguments: Option<&Map<String, serde_json::Value>>) -> PromptResult {
    let topic = arguments
        .and_then(|a| a.get("topic"))
        .and_then(|v| v.as_str())
        .unwrap_or("general");

    let bank = arguments
        .and_then(|a| a.get("bank"))
        .and_then(|v| v.as_str())
        .unwrap_or("default");

    let messages = vec![
        user_text(format!(
            "Build a layered mind map about: **{topic}**\n\
             Bank: `{bank}`\n\n\
             The mind map has 3 levels:\n\
             - **Notes** (raw facts) — the details\n\
             - **Summary cards** — groups of related notes boiled into one insight\n\
             - **Concepts** — the big picture that ties summaries together\n\n\
             Example — topic \"Rust\":\n\
             - Note: \"Tokio tasks are spawned with tokio::spawn\"\n\
             - Note: \"async blocks return a future that must be .awaited\"\n\
             - Note: \"Futures do nothing unless polled by an executor\"\n\
               → Summary card: \"Rust async runtime needs an executor (like Tokio) to drive futures\"\n\
             - Note: \"Cargo workspaces allow multiple crates in one repo\"\n\
             - Note: \"workspace members share a Cargo.lock\"\n\
               → Summary card: \"Cargo workspaces manage multi-crate projects\"\n\
               → Concept: \"Rust project structure spans async runtime choices and package organization\"\n\n\
             ---\n\n\
             **PART 1: Store and link notes**\n\n\
             Repeat this 3-step loop for every fact:\n\n\
             **Step 1 — SEARCH** (avoid duplicates)\n\
             Call `search_memory` before storing:\n\
             ```json\n\
             {{\"query\": \"<describe the fact>\", \"bank\": \"{bank}\"}}\n\
             ```\n\
             If it already exists, skip to Step 3.\n\n\
             **Step 2 — STORE** (one fact per note)\n\
             Call `add_content_memory`:\n\
             ```json\n\
             {{\n\
               \"content\": \"<one single fact>\",\n\
               \"context\": [\"{topic}\", \"<tag1>\"],\n\
               \"bank\": \"{bank}\"\n\
             }}\n\
             ```\n\
             Save the memory_id.\n\n\
             **Step 3 — LINK** (connect to the mind map)\n\
             Call `force_link`:\n\
             ```json\n\
             {{\n\
               \"source_id\": \"<new_note_id>\",\n\
               \"relation\": \"<type>\",\n\
               \"target_id\": \"<related_note_id>\",\n\
               \"bank\": \"{bank}\"\n\
             }}\n\
             ```\n\
             Relation types: `depends_on`, `references`, `part_of`, `extends`, `contradicts`\n\n\
             ---\n\n\
             **PART 2: Build summary cards and concepts**\n\n\
             When you have 5+ notes stored, do this:\n\n\
             **Step 4 — Group and summarize**\n\
             Look at your notes. Find groups of 3-5 notes that share a theme.\n\
             For each group, write a one-sentence summary and call `create_abstraction`:\n\
             ```json\n\
             {{\n\
               \"content\": \"<your one-sentence summary of this group>\",\n\
               \"source_ids\": [\"<id1>\", \"<id2>\", \"<id3>\"],\n\
               \"target_layer\": 1,\n\
               \"bank\": \"{bank}\"\n\
             }}\n\
             ```\n\
             Save the new summary card id. Link summaries to each other with `force_link` \
             if they are related.\n\n\
             **Step 5 — Build the big picture**\n\
             When you have 3+ summary cards, group them into a concept:\n\
             ```json\n\
             {{\n\
               \"content\": \"<one-sentence big-picture insight tying the summaries together>\",\n\
               \"source_ids\": [\"<summary_id1>\", \"<summary_id2>\"],\n\
               \"target_layer\": 2,\n\
               \"bank\": \"{bank}\"\n\
             }}\n\
             ```\n\
             This is the top of your mind map. One concept that ties everything together."
        )),
        assistant_text(format!(
            "Building layered mind map for: {topic}\n\n\
             Part 1 — storing and linking notes.\n\
             Step 1 — searching for existing notes:\n\
             ```json\n\
             {{\"query\": \"{topic}\", \"bank\": \"{bank}\"}}\n\
             ```"
        )),
        user_text(
            "Do Part 1 first (store and link notes). When you have 5+ notes, move to Part 2 \
             and create summary cards and concepts yourself. When done, report: \
             notes stored, links created, summary cards created, concepts created, \
             and describe the mind map you built.",
        ),
    ];

    ok_result(
        messages,
        format!("Build a layered mind map of notes, summaries, and concepts about: {}.", topic),
    )
}
