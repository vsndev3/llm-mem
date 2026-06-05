#[cfg(test)]
mod responses {
    use crate::operations::*;
    use crate::operations::{
        get_mcp_tool_definitions, get_operation_error_message, operation_error_to_mcp_error_code,
    };
    use serde_json::json;

    // --- MemoryOperationResponse tests ---

    #[test]
    fn test_response_success() {
        let r = MemoryOperationResponse::success("ok");
        assert!(r.success);
        assert_eq!(r.message, "ok");
        assert!(r.data.is_none());
        assert!(r.error.is_none());
    }

    #[test]
    fn test_response_success_with_data() {
        let data = json!({"id": "abc"});
        let r = MemoryOperationResponse::success_with_data("stored", data.clone());
        assert!(r.success);
        assert_eq!(r.message, "stored");
        assert_eq!(r.data, Some(data));
    }

    #[test]
    fn test_response_error() {
        let r = MemoryOperationResponse::error("something went wrong");
        assert!(!r.success);
        assert_eq!(r.error.as_deref(), Some("something went wrong"));
        assert_eq!(r.message, "Operation failed");
    }

    #[test]
    fn test_response_serialization() {
        let r = MemoryOperationResponse::success_with_data("ok", json!({"count": 5}));
        let json_str = serde_json::to_string(&r).unwrap();
        let restored: MemoryOperationResponse = serde_json::from_str(&json_str).unwrap();
        assert!(restored.success);
        assert_eq!(restored.data.unwrap()["count"], 5);
    }

    // --- MCP tool definitions ---

    #[test]
    fn test_get_mcp_tool_definitions_count() {
        let tools = get_mcp_tool_definitions();
        assert_eq!(tools.len(), 32);
    }

    #[test]
    fn test_mcp_tool_names() {
        let tools = get_mcp_tool_definitions();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"system_status"));
        assert!(names.contains(&"health_check"));
        assert!(names.contains(&"add_content_memory"));
        assert!(names.contains(&"add_intuitive_memory"));
        assert!(names.contains(&"upload_document"));
        assert!(names.contains(&"document_status"));
        assert!(names.contains(&"cancel_document"));
        assert!(names.contains(&"query_memory"));
        assert!(names.contains(&"list_memories"));
        assert!(names.contains(&"get_memory"));
        assert!(names.contains(&"navigate_memory"));
        assert!(names.contains(&"get_timeline"));
        assert!(names.contains(&"get_timeline_graph"));
        assert!(names.contains(&"list_memory_banks"));
        assert!(names.contains(&"create_memory_bank"));
        assert!(names.contains(&"backup_bank"));
        assert!(names.contains(&"restore_bank"));
        assert!(names.contains(&"rename_memory_bank"));
        assert!(names.contains(&"cleanup_resources"));
        assert!(names.contains(&"start_abstraction_pipeline"));
        assert!(names.contains(&"stop_abstraction_pipeline"));
        assert!(names.contains(&"trigger_abstraction"));
    }

    #[test]
    fn test_rename_memory_bank_tool_definition() {
        let tools = get_mcp_tool_definitions();
        let rename_tool = tools
            .iter()
            .find(|t| t.name == "rename_memory_bank")
            .unwrap();

        assert!(rename_tool.title.is_some());
        assert!(rename_tool.description.is_some());
        assert!(rename_tool.description.as_ref().unwrap().contains("atomic"));
        assert!(
            rename_tool
                .description
                .as_ref()
                .unwrap()
                .contains("session database")
        );

        let required = rename_tool.input_schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "old_name"));
        assert!(required.iter().any(|v| v == "new_name"));

        let props = rename_tool.input_schema["properties"].as_object().unwrap();
        assert!(props.contains_key("old_name"));
        assert!(props.contains_key("new_name"));

        assert!(rename_tool.output_schema.is_some());
        let output_props = rename_tool.output_schema.as_ref().unwrap()["properties"]
            .as_object()
            .unwrap();
        assert!(output_props.contains_key("success"));
        assert!(output_props.contains_key("message"));
        assert!(output_props.contains_key("old_name"));
        assert!(output_props.contains_key("new_name"));
    }

    #[test]
    fn test_system_status_is_first_tool() {
        let tools = get_mcp_tool_definitions();
        assert_eq!(tools[0].name, "system_status");
    }

    #[test]
    fn test_mcp_tools_have_descriptions() {
        for tool in get_mcp_tool_definitions() {
            assert!(
                tool.description.is_some(),
                "Tool {} missing description",
                tool.name
            );
            assert!(!tool.description.as_ref().unwrap().is_empty());
        }
    }

    #[test]
    fn test_mcp_tools_store_requires_content() {
        let tools = get_mcp_tool_definitions();
        let store = tools
            .iter()
            .find(|t| t.name == "add_content_memory")
            .unwrap();
        let required = store.input_schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "content"));
    }

    #[test]
    fn test_mcp_tools_query_requires_query() {
        let tools = get_mcp_tool_definitions();
        let query = tools.iter().find(|t| t.name == "query_memory").unwrap();
        let required = query.input_schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "query"));
    }

    #[test]
    fn test_mcp_tools_get_requires_memory_id() {
        let tools = get_mcp_tool_definitions();
        let get = tools.iter().find(|t| t.name == "get_memory").unwrap();
        let required = get.input_schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "memory_id"));
    }

    #[test]
    fn test_health_check_tool_definition() {
        let tools = get_mcp_tool_definitions();
        let hc = tools
            .iter()
            .find(|t| t.name == "health_check")
            .expect("health_check tool must be registered");

        assert!(hc.title.is_some());
        assert!(hc.description.is_some());
        assert!(hc.description.as_ref().unwrap().contains("embed"));
        assert!(hc.description.as_ref().unwrap().contains("complete"));

        let props = hc.input_schema["properties"].as_object().unwrap();
        assert!(props.contains_key("live"));
        assert!(props.contains_key("embed_only"));
        assert!(props.contains_key("llm_only"));
        assert!(props.contains_key("embed_timeout_secs"));
        assert!(props.contains_key("llm_timeout_secs"));
        // No required params — all live-check options are optional.
        let required = hc.input_schema["required"].as_array().unwrap();
        assert!(required.is_empty(), "health_check must have no required args");

        let output_props = hc.output_schema.as_ref().unwrap()["properties"]
            .as_object()
            .unwrap();
        assert!(output_props.contains_key("healthy"));
        assert!(output_props.contains_key("backend"));
        assert!(output_props.contains_key("checks"));
    }

    // --- error codes ---

    #[test]
    fn test_operation_error_codes() {
        assert_eq!(
            operation_error_to_mcp_error_code(&OperationError::InvalidInput("".into())),
            -32602
        );
        assert_eq!(
            operation_error_to_mcp_error_code(&OperationError::Runtime("".into())),
            -32603
        );
        assert_eq!(
            operation_error_to_mcp_error_code(&OperationError::MemoryNotFound("".into())),
            -32601
        );
    }

    #[test]
    fn test_operation_error_display() {
        let e = OperationError::InvalidInput("bad input".into());
        assert_eq!(e.to_string(), "Invalid input: bad input");

        let e = OperationError::Runtime("crash".into());
        assert_eq!(e.to_string(), "Runtime error: crash");

        let e = OperationError::MemoryNotFound("id-42".into());
        assert_eq!(e.to_string(), "Memory not found: id-42");
    }

    #[test]
    fn test_get_operation_error_message() {
        let e = OperationError::InvalidInput("x".into());
        assert_eq!(get_operation_error_message(&e), "x");

        let e = OperationError::Runtime("y".into());
        assert_eq!(get_operation_error_message(&e), "y");
    }

    #[test]
    fn test_operation_error_from_memory_error() {
        let mem_err = crate::error::MemoryError::config("bad config");
        let op_err: OperationError = mem_err.into();
        match op_err {
            OperationError::Runtime(msg) => assert!(msg.contains("bad config")),
            _ => panic!("expected Runtime"),
        }
    }
}

#[cfg(test)]
mod tests_graph {
    use crate::operations::serialization::memory_to_json;
    use crate::types::{RelationEntry, RelationMeta};
    use uuid::Uuid;

    #[test]
    fn test_memory_serialization_with_relations() {
        use crate::types::{Memory, MemoryMetadata, Relation};

        let mut metadata = MemoryMetadata::new();
        metadata.relations = vec![Relation {
            source: "SELF".to_string(),
            relation: "KNOWS".to_string(),
            target: "Alice".to_string(),
            strength: None,
        }];

        let mut memory = Memory::with_content("Bob knows Alice".to_string(), vec![], metadata);
        memory.relations.insert(
            "knows".to_string(),
            RelationEntry::new(vec![Uuid::new_v4()], None, RelationMeta::new("test")),
        );

        let json = memory_to_json(&memory);

        let relations = json["metadata"]["relations"]
            .as_array()
            .expect("Relations should be an array");
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0]["relation"], "KNOWS");
        assert_eq!(relations[0]["target"], "Alice");
    }
}

#[cfg(test)]
mod tests_context {
    use crate::operations::serialization::memory_to_json;
    use crate::operations::RelationInput;
    use crate::types::{Memory, MemoryMetadata};

    #[test]
    fn test_memory_to_json_includes_context() {
        let mut meta = MemoryMetadata::new();
        meta.context = vec!["recipe".into(), "italian".into()];

        let memory = Memory::with_content("Test context".to_string(), vec![], meta);

        let json = memory_to_json(&memory);
        let ctx = json["metadata"]["context"]
            .as_array()
            .expect("context should be an array");
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx[0], "recipe");
        assert_eq!(ctx[1], "italian");
    }

    #[test]
    fn test_memory_to_json_omits_empty_context() {
        let meta = MemoryMetadata::new();

        let memory = Memory::with_content("No context".to_string(), vec![], meta);

        let json = memory_to_json(&memory);
        assert!(json["metadata"]["context"].is_null() || json["metadata"].get("context").is_none());
    }

    #[test]
    fn test_relation_input_serialization() {
        let input = RelationInput {
            relation: "KNOWS".into(),
            target: "Alice".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let restored: RelationInput = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.relation, "KNOWS");
        assert_eq!(restored.target, "Alice");
    }
}
