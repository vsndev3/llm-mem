use llm_mem::prompts;

#[test]
fn test_list_all_prompts_returns_five() {
    let prompts = prompts::list_all_prompts();
    assert_eq!(prompts.len(), 5, "Expected 5 prompts");

    let names: Vec<&str> = prompts.iter().map(|p| p.name.as_str()).collect();
    assert!(names.contains(&"auto_process_document"));
    assert!(names.contains(&"build_memory_graph"));
    assert!(names.contains(&"explore_knowledge"));
    assert!(names.contains(&"guided_document_ingest"));
    assert!(names.contains(&"quick_start"));
}

#[test]
fn test_prompts_are_sorted_by_name() {
    let prompts = prompts::list_all_prompts();
    let names: Vec<&str> = prompts.iter().map(|p| p.name.as_str()).collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted, "Prompts should be sorted alphabetically");
}

#[test]
fn test_each_prompt_has_name_and_description() {
    for p in prompts::list_all_prompts() {
        assert!(!p.name.is_empty(), "Prompt name should not be empty");
        assert!(
            p.description.is_some(),
            "Prompt '{}' should have a description",
            p.name
        );
        assert!(
            !p.description.as_ref().unwrap().is_empty(),
            "Prompt '{}' description should not be empty",
            p.name
        );
    }
}

#[test]
fn test_get_prompt_unknown_returns_error() {
    let result = prompts::get_prompt("nonexistent", None);
    assert!(result.is_err(), "Unknown prompt should return error");
}

#[test]
fn test_quick_start_no_args() {
    let result = prompts::get_prompt("quick_start", None).expect("quick_start should succeed");
    assert!(!result.messages.is_empty(), "Should have messages");
    assert!(result.description.is_some());

    let has_user_msg = result
        .messages
        .iter()
        .any(|m| matches!(m.role, rmcp::model::PromptMessageRole::User));
    let has_assistant_msg = result
        .messages
        .iter()
        .any(|m| matches!(m.role, rmcp::model::PromptMessageRole::Assistant));
    assert!(has_user_msg, "Should have user messages");
    assert!(has_assistant_msg, "Should have assistant messages");
}

#[test]
fn test_build_memory_graph_no_args() {
    let result =
        prompts::get_prompt("build_memory_graph", None).expect("build_memory_graph should succeed");
    assert!(
        result.messages.len() >= 3,
        "Should have at least 3 messages (user/assistant/user)"
    );
    assert!(result.description.is_some());
}

#[test]
fn test_build_memory_graph_with_topic_and_bank() {
    use serde_json::{Map, json};

    let mut args = Map::new();
    args.insert("topic".into(), json!("machine learning"));
    args.insert("bank".into(), json!("research"));

    let result = prompts::get_prompt("build_memory_graph", Some(&args)).expect("should succeed");

    let content = serde_json::to_string(&result.messages).unwrap();
    assert!(
        content.contains("machine learning"),
        "Should contain the topic"
    );
    assert!(content.contains("research"), "Should contain the bank name");
}

#[test]
fn test_auto_process_document_with_file_path() {
    use serde_json::{Map, json};

    let mut args = Map::new();
    args.insert("file_path".into(), json!("/tmp/test.pdf"));
    args.insert("bank".into(), json!("docs"));

    let result = prompts::get_prompt("auto_process_document", Some(&args)).expect("should succeed");
    assert!(result.messages.len() >= 3);

    let content = serde_json::to_string(&result.messages).unwrap();
    assert!(
        content.contains("/tmp/test.pdf"),
        "Should contain the file path"
    );
    assert!(content.contains("docs"), "Should contain the bank name");
}

#[test]
fn test_auto_process_document_with_chunk_size() {
    use serde_json::{Map, json};

    let mut args = Map::new();
    args.insert("file_path".into(), json!("/tmp/test.md"));
    args.insert("chunk_size".into(), json!(3000));

    let result = prompts::get_prompt("auto_process_document", Some(&args)).expect("should succeed");
    let content = serde_json::to_string(&result.messages).unwrap();
    assert!(content.contains("3000"), "Should contain the chunk size");
}

#[test]
fn test_guided_document_ingest() {
    use serde_json::{Map, json};

    let mut args = Map::new();
    args.insert("file_path".into(), json!("/tmp/paper.pdf"));

    let result =
        prompts::get_prompt("guided_document_ingest", Some(&args)).expect("should succeed");
    assert!(result.messages.len() >= 3);

    let content = serde_json::to_string(&result.messages).unwrap();
    assert!(content.contains("/tmp/paper.pdf"));
    assert!(
        content.contains("STOP"),
        "Should instruct to stop and review"
    );
    assert!(
        content.contains("Extract facts"),
        "Should mention fact extraction"
    );
    assert!(content.contains("Link related"), "Should mention linking");
    assert!(
        content.contains("summary cards"),
        "Should mention summaries"
    );
}

#[test]
fn test_explore_knowledge_with_query() {
    use serde_json::{Map, json};

    let mut args = Map::new();
    args.insert("query".into(), json!("neural networks"));
    args.insert("depth".into(), json!(5));

    let result = prompts::get_prompt("explore_knowledge", Some(&args)).expect("should succeed");
    assert!(result.messages.len() >= 3);

    let content = serde_json::to_string(&result.messages).unwrap();
    assert!(
        content.contains("neural networks"),
        "Should contain the query"
    );
    assert!(content.contains("5"), "Should contain the depth");
    assert!(
        result
            .description
            .as_ref()
            .unwrap()
            .contains("neural networks")
    );
}

#[test]
fn test_explore_knowledge_default_values() {
    let result =
        prompts::get_prompt("explore_knowledge", None).expect("should succeed with no args");
    let content = serde_json::to_string(&result.messages).unwrap();
    assert!(
        content.contains("general knowledge"),
        "Should use default query"
    );
    assert!(content.contains("default"), "Should use default bank");
    assert!(content.contains("3"), "Should use default depth");
}

#[test]
fn test_prompt_definitions_have_correct_arguments() {
    let prompts = prompts::list_all_prompts();

    let auto = prompts
        .iter()
        .find(|p| p.name == "auto_process_document")
        .unwrap();
    let args = auto.arguments.as_ref().expect("should have arguments");
    let arg_names: Vec<&str> = args.iter().map(|a| a.name.as_str()).collect();
    assert!(arg_names.contains(&"file_path"));
    assert!(arg_names.contains(&"bank"));
    assert!(arg_names.contains(&"chunk_size"));

    let explore = prompts
        .iter()
        .find(|p| p.name == "explore_knowledge")
        .unwrap();
    let args = explore.arguments.as_ref().expect("should have arguments");
    let arg_names: Vec<&str> = args.iter().map(|a| a.name.as_str()).collect();
    assert!(arg_names.contains(&"query"));
    assert!(arg_names.contains(&"bank"));
    assert!(arg_names.contains(&"depth"));

    let quick = prompts.iter().find(|p| p.name == "quick_start").unwrap();
    assert!(
        quick.arguments.is_none(),
        "quick_start should have no arguments"
    );
}

#[test]
fn test_required_arguments_marked_correctly() {
    let prompts = prompts::list_all_prompts();

    let auto = prompts
        .iter()
        .find(|p| p.name == "auto_process_document")
        .unwrap();
    let args = auto.arguments.as_ref().unwrap();
    let file_path_arg = args.iter().find(|a| a.name == "file_path").unwrap();
    assert_eq!(
        file_path_arg.required,
        Some(true),
        "file_path should be required"
    );

    let bank_arg = args.iter().find(|a| a.name == "bank").unwrap();
    assert_eq!(bank_arg.required, Some(false), "bank should be optional");

    let explore = prompts
        .iter()
        .find(|p| p.name == "explore_knowledge")
        .unwrap();
    let args = explore.arguments.as_ref().unwrap();
    let query_arg = args.iter().find(|a| a.name == "query").unwrap();
    assert_eq!(query_arg.required, Some(true), "query should be required");
}

#[test]
fn test_all_prompt_messages_have_valid_roles() {
    for prompt_name in &[
        "quick_start",
        "build_memory_graph",
        "auto_process_document",
        "guided_document_ingest",
        "explore_knowledge",
    ] {
        let result = prompts::get_prompt(prompt_name, None)
            .unwrap_or_else(|_| panic!("{} should succeed", prompt_name));

        for msg in &result.messages {
            match msg.role {
                rmcp::model::PromptMessageRole::User
                | rmcp::model::PromptMessageRole::Assistant => {}
            }
        }
    }
}
