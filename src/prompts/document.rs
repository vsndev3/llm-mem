use rmcp::{ErrorData, model::{GetPromptResult, Prompt}};
use serde_json::Map;

use super::{PromptResult, arg, assistant_text, make_prompt, ok_result, user_text};

pub fn auto_process_prompt_def() -> Prompt {
    super::make_prompt(
        "auto_process_document",
        "Ingest a file end-to-end with no manual steps. Uploads, waits for chunking, \
         triggers summary creation, and reports results. Use this when you just want \
         the document loaded quickly.",
        vec![
            arg("file_path", "Absolute path to the file to process", true),
            arg("bank", "Memory bank name (default: 'default')", false),
            arg("chunk_size", "Chunk size in characters (default: server config value)", false),
        ],
    )
}

pub fn guided_ingest_prompt_def() -> Prompt {
    super::make_prompt(
        "guided_document_ingest",
        "Upload a file and create note chunks, then STOP for review. You decide which \
         facts to extract, which notes to link, and whether to create summary cards. \
         Use this when quality matters more than speed.",
        vec![
            arg("file_path", "Absolute path to the file to ingest", true),
            arg("bank", "Memory bank name (default: 'default')", false),
        ],
    )
}

pub fn get_auto_process(arguments: Option<&Map<String, serde_json::Value>>) -> PromptResult {
    let file_path = arguments
        .and_then(|a| a.get("file_path"))
        .and_then(|v| v.as_str())
        .unwrap_or("<FILE_PATH>");

    let bank = arguments
        .and_then(|a| a.get("bank"))
        .and_then(|v| v.as_str())
        .unwrap_or("default");

    let chunk_size_param = arguments
        .and_then(|a| a.get("chunk_size"))
        .and_then(|v| v.as_i64())
        .map(|cs| format!(",\n  \"chunk_size\": {}", cs))
        .unwrap_or_default();

    let messages = vec![
        user_text(format!(
            "Process this file fully automatically: `{file_path}`\n\
             Bank: `{bank}`\n\n\
             Do these 4 steps in order. Do NOT stop to ask questions.\n\n\
             **Step 1 — Upload**\n\
             Call `upload_document`:\n\
             ```json\n\
             {{\n\
               \"file_path\": \"{file_path}\",\n\
               \"process_immediately\": true,\n\
               \"memory_type\": \"semantic\",\n\
               \"bank\": \"{bank}\"{chunk_size_param}\n\
             }}\n\
             ```\n\
             Save the session_id from the response.\n\n\
             **Step 2 — Wait for chunks**\n\
             Call `document_status` every few seconds until status is \"completed\":\n\
             ```json\n\
             {{\"session_id\": \"<session_id>\", \"bank\": \"{bank}\"}}\n\
             ```\n\n\
             **Step 3 — Create summaries**\n\
             Call `trigger_abstraction`:\n\
             ```json\n\
             {{\"target_layer\": 1}}\n\
             ```\n\n\
             **Step 4 — Report**\n\
             Call `list_memories` to see what was created:\n\
             ```json\n\
             {{\"limit\": 50, \"bank\": \"{bank}\"}}\n\
             ```\n\
             Then tell the user: how many note chunks, how many summary cards, any problems."
        )),
        assistant_text(format!(
            "Processing `{file_path}` automatically.\n\n\
             Step 1 — uploading:\n\
             ```json\n\
             {{\n\
               \"file_path\": \"{file_path}\",\n\
               \"process_immediately\": true,\n\
               \"memory_type\": \"semantic\",\n\
               \"bank\": \"{bank}\"{chunk_size_param}\n\
             }}\n\
             ```"
        )),
        user_text(
            "Do all 4 steps. If any step fails, retry once. Then give a summary.",
        ),
    ];

    ok_result(
        messages,
        format!("Auto-process {}: upload, chunk, summarize, report.", file_path),
    )
}

pub fn get_guided_ingest(arguments: Option<&Map<String, serde_json::Value>>) -> PromptResult {
    let file_path = arguments
        .and_then(|a| a.get("file_path"))
        .and_then(|v| v.as_str())
        .unwrap_or("<FILE_PATH>");

    let bank = arguments
        .and_then(|a| a.get("bank"))
        .and_then(|v| v.as_str())
        .unwrap_or("default");

    let messages = vec![
        user_text(format!(
            "Guided ingestion of: `{file_path}`\n\
             Bank: `{bank}`\n\n\
             **Phase A — Upload and chunk**\n\
             Call `upload_document`:\n\
             ```json\n\
             {{\n\
               \"file_path\": \"{file_path}\",\n\
               \"process_immediately\": true,\n\
               \"memory_type\": \"semantic\",\n\
               \"bank\": \"{bank}\"\n\
             }}\n\
             ```\n\
             Wait with `document_status` until completed.\n\
             Then call `list_memories` to see all note chunks:\n\
             ```json\n\
             {{\"limit\": 100, \"bank\": \"{bank}\"}}\n\
             ```\n\n\
             **STOP HERE. Review every chunk, then do these 3 things:**\n\n\
             1. **Extract facts** — For chunks with definitions or key facts, call \
             `add_content_memory` for each single fact.\n\n\
             2. **Link related chunks** — For chunks that refer to each other, call \
             `force_link` with relation `references` or `part_of`.\n\n\
             3. **Create summary cards** — For 3+ chunks sharing a theme, write a \
             one-sentence summary and call `create_abstraction`:\n\
             ```json\n\
             {{\n\
               \"content\": \"<your one-sentence summary>\",\n\
               \"source_ids\": [\"<chunk_id1>\", \"<chunk_id2>\", \"<chunk_id3>\"],\n\
               \"target_layer\": 1,\n\
               \"bank\": \"{bank}\"\n\
             }}\n\
             ```\n\n\
             After finishing, report what you extracted, linked, and summarized."
        )),
        assistant_text(format!(
            "Starting guided ingestion of `{file_path}`.\n\n\
             Phase A — uploading:\n\
             ```json\n\
             {{\n\
               \"file_path\": \"{file_path}\",\n\
               \"process_immediately\": true,\n\
               \"memory_type\": \"semantic\",\n\
               \"bank\": \"{bank}\"\n\
             }}\n\
             ```"
        )),
        user_text(
            "After upload, STOP and review each chunk before extracting, linking, or \
             summarizing. Take your time. Explain what you decided and why.",
        ),
    ];

    ok_result(
        messages,
        format!("Guided ingest of {}: upload, review, extract, link, summarize.", file_path),
    )
}
