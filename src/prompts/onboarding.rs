use rmcp::model::Prompt;
use serde_json::Map;

use super::{PromptResult, assistant_text, make_prompt, ok_result, user_text};

pub fn prompt_def() -> Prompt {
    super::make_prompt(
        "quick_start",
        "Quick walkthrough: check system, store a note, find it by searching, view its \
         summary card. Call this when you want to learn how to use the memory system.",
        vec![],
    )
}

pub fn get(_arguments: Option<&Map<String, serde_json::Value>>) -> PromptResult {
    let messages = vec![
        user_text(
            "Learn llm-mem by doing. It is a memory system that stores facts as notes, \
             auto-creates summary cards, and links related notes into a mind map.\n\n\
             Do these 4 steps in order:\n\n\
             1. CHECK SYSTEM — Call `system_status`. If ready_to_use is false, stop and tell the user.\n\n\
             2. STORE A NOTE — Call `add_content_memory`:\n\
             ```json\n\
             {\"content\": \"The project uses Rust with tokio async runtime\", \n\
              \"context\": [\"rust\", \"architecture\"]}\n\
             ```\n\
             Save the memory_id from the response.\n\n\
             3. FIND YOUR NOTE — Call `search_memory`:\n\
             ```json\n\
             {\"query\": \"what language does the project use\"}\n\
             ```\n\
             Confirm your note appears.\n\n\
             4. VIEW SUMMARY — Call `navigate_memory` with your memory_id:\n\
             ```json\n\
             {\"memory_id\": \"<your_id>\", \"direction\": \"zoom_out\"}\n\
             ```\n\
             This shows any summary cards created from your notes.\n\n\
             Done! After step 4, tell the user what you learned.",
        ),
        assistant_text(
            "Starting the walkthrough. Step 1: checking system status.\n\n\
             ```json\n\
             {}\n\
             ```\n\n\
             Calling `system_status`...",
        ),
        user_text(
            "Do all 4 steps now. After finishing, give a short summary of what each step did.",
        ),
    ];

    ok_result(
        messages,
        "Quick walkthrough: check system, store a note, find it, view its summary.",
    )
}
