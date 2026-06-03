pub mod document;
pub mod exploration;
pub mod memory_building;
pub mod onboarding;

use rmcp::{
    ErrorData,
    model::{GetPromptResult, Prompt, PromptArgument, PromptMessage, PromptMessageRole},
};
use serde_json::Map;

pub(crate) type PromptResult = Result<GetPromptResult, ErrorData>;

fn user_text(text: impl Into<String>) -> PromptMessage {
    PromptMessage::new_text(PromptMessageRole::User, text)
}

fn assistant_text(text: impl Into<String>) -> PromptMessage {
    PromptMessage::new_text(PromptMessageRole::Assistant, text)
}

fn arg(name: &str, description: &str, required: bool) -> PromptArgument {
    PromptArgument::new(name)
        .with_description(description)
        .with_required(required)
}

pub(crate) fn make_prompt(
    name: &str,
    description: &str,
    arguments: Vec<PromptArgument>,
) -> Prompt {
    Prompt::new(name, Some(description), if arguments.is_empty() { None } else { Some(arguments) })
}

fn ok_result(messages: Vec<PromptMessage>, description: impl Into<String>) -> PromptResult {
    Ok(GetPromptResult::new(messages).with_description(description))
}

pub fn list_all_prompts() -> Vec<Prompt> {
    let mut prompts: Vec<Prompt> = vec![
        onboarding::prompt_def(),
        memory_building::prompt_def(),
        document::auto_process_prompt_def(),
        document::guided_ingest_prompt_def(),
        exploration::prompt_def(),
    ];
    prompts.sort_by(|a, b| a.name.cmp(&b.name));
    prompts
}

pub fn get_prompt(
    name: &str,
    arguments: Option<&Map<String, serde_json::Value>>,
) -> PromptResult {
    match name {
        "quick_start" => onboarding::get(arguments),
        "build_memory_graph" => memory_building::get(arguments),
        "auto_process_document" => document::get_auto_process(arguments),
        "guided_document_ingest" => document::get_guided_ingest(arguments),
        "explore_knowledge" => exploration::get(arguments),
        _ => Err(ErrorData::invalid_params(
            format!("prompt '{}' not found", name),
            Some(serde_json::json!({
                "available_prompts": list_all_prompts().iter().map(|p| &p.name).collect::<Vec<_>>()
            })),
        )),
    }
}
