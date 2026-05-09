#[cfg(test)]
mod tests {
    use crate::operations::*;
    use crate::operations::{
        get_mcp_tool_definitions, get_operation_error_message, operation_error_to_mcp_error_code,
    };
    use serde_json::json;

    // --- MemoryOperationPayload tests ---

    #[test]
    fn test_payload_default() {
        let payload = MemoryOperationPayload::default();
        assert!(payload.content.is_none());
        assert!(payload.query.is_none());
        assert!(payload.memory_id.is_none());
        assert!(payload.user_id.is_none());
        assert!(payload.limit.is_none());
    }

    #[test]
    fn test_payload_serialization() {
        let payload = MemoryOperationPayload {
            content: Some("test content".into()),
            user_id: Some("u1".into()),
            memory_type: Some("factual".into()),
            ..Default::default()
        };
        let json = serde_json::to_string(&payload).unwrap();
        let restored: MemoryOperationPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.content.as_deref(), Some("test content"));
        assert_eq!(restored.user_id.as_deref(), Some("u1"));
    }

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

    // --- QueryParams tests ---

    #[test]
    fn test_query_params_valid() {
        let payload = MemoryOperationPayload {
            query: Some("search term".into()),
            limit: Some(20),
            min_salience: Some(0.5),
            user_id: Some("u1".into()),
            ..Default::default()
        };
        let params = QueryParams::from_payload(&payload, 10).unwrap();
        assert_eq!(params.query, "search term");
        assert_eq!(params.limit, 20);
        assert_eq!(params.min_salience, Some(0.5));
        assert_eq!(params.user_id.as_deref(), Some("u1"));
    }

    #[test]
    fn test_query_params_missing_query() {
        let payload = MemoryOperationPayload::default();
        let result = QueryParams::from_payload(&payload, 10);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OperationError::InvalidInput(_)
        ));
    }

    #[test]
    fn test_query_params_uses_k_fallback() {
        let payload = MemoryOperationPayload {
            query: Some("test".into()),
            k: Some(5),
            ..Default::default()
        };
        let params = QueryParams::from_payload(&payload, 10).unwrap();
        assert_eq!(params.limit, 5);
    }

    #[test]
    fn test_query_params_uses_default_limit() {
        let payload = MemoryOperationPayload {
            query: Some("test".into()),
            ..Default::default()
        };
        let params = QueryParams::from_payload(&payload, 42).unwrap();
        assert_eq!(params.limit, 42);
    }

    #[test]
    fn test_query_params_date_parsing() {
        let payload = MemoryOperationPayload {
            query: Some("test".into()),
            created_after: Some("2024-01-01T00:00:00Z".into()),
            created_before: Some("2024-12-31T23:59:59Z".into()),
            ..Default::default()
        };
        let params = QueryParams::from_payload(&payload, 10).unwrap();
        assert!(params.created_after.is_some());
        assert!(params.created_before.is_some());
    }

    #[test]
    fn test_query_params_invalid_date_ignored() {
        let payload = MemoryOperationPayload {
            query: Some("test".into()),
            created_after: Some("not-a-date".into()),
            ..Default::default()
        };
        let params = QueryParams::from_payload(&payload, 10).unwrap();
        assert!(params.created_after.is_none());
    }

    // --- StoreParams tests ---

    #[test]
    fn test_store_params_valid() {
        let payload = MemoryOperationPayload {
            content: Some("memory content".into()),
            user_id: Some("user1".into()),
            topics: Some(vec!["rust".into()]),
            ..Default::default()
        };
        let params = StoreParams::from_payload(&payload, None, None).unwrap();
        assert_eq!(params.content, "memory content");
        assert_eq!(params.user_id.as_deref(), Some("user1"));
        assert_eq!(params.memory_type, "conversational");
        assert_eq!(params.topics, Some(vec!["rust".into()]));
    }

    #[test]
    fn test_store_params_missing_content() {
        let payload = MemoryOperationPayload {
            user_id: Some("u1".into()),
            ..Default::default()
        };
        let result = StoreParams::from_payload(&payload, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_store_params_uses_default_user_id() {
        let payload = MemoryOperationPayload {
            content: Some("test".into()),
            ..Default::default()
        };
        let params =
            StoreParams::from_payload(&payload, Some("default_user".into()), None).unwrap();
        assert_eq!(params.user_id.as_deref(), Some("default_user"));
    }

    #[test]
    fn test_store_params_no_user_id_is_ok() {
        let payload = MemoryOperationPayload {
            content: Some("test".into()),
            ..Default::default()
        };
        let result = StoreParams::from_payload(&payload, None, None);
        assert!(result.is_ok());
        assert!(result.unwrap().user_id.is_none());
    }

    // --- AddMemoryParams tests ---

    #[test]
    fn test_add_memory_params_valid() {
        let payload = MemoryOperationPayload {
            messages: Some(vec![crate::types::Message {
                role: "user".into(),
                content: "hello".into(),
                name: None,
            }]),
            user_id: Some("user1".into()),
            ..Default::default()
        };
        let params = AddMemoryParams::from_payload(&payload, None, None).unwrap();
        assert_eq!(params.messages.len(), 1);
        assert_eq!(params.messages[0].content, "hello");
        assert_eq!(params.user_id.as_deref(), Some("user1"));
        assert_eq!(params.memory_type, "conversational");
    }

    #[test]
    fn test_add_memory_params_missing_messages() {
        let payload = MemoryOperationPayload {
            user_id: Some("u1".into()),
            ..Default::default()
        };
        let result = AddMemoryParams::from_payload(&payload, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_memory_params_empty_messages() {
        let payload = MemoryOperationPayload {
            messages: Some(vec![]),
            ..Default::default()
        };
        let result = AddMemoryParams::from_payload(&payload, None, None);
        assert!(result.is_err());
    }

    // --- IngestDocumentParams tests ---

    #[test]
    fn test_ingest_document_params_valid() {
        let payload = MemoryOperationPayload {
            content: Some("document content".into()),
            user_id: Some("user1".into()),
            ..Default::default()
        };
        let params = IngestDocumentParams::from_payload(&payload, None, None).unwrap();
        assert_eq!(params.content, "document content");
        assert_eq!(params.user_id.as_deref(), Some("user1"));
        assert_eq!(params.memory_type, "semantic");
    }

    #[test]
    fn test_ingest_document_params_missing_content() {
        let payload = MemoryOperationPayload {
            user_id: Some("u1".into()),
            ..Default::default()
        };
        let result = IngestDocumentParams::from_payload(&payload, None, None);
        assert!(result.is_err());
    }

    // --- FilterParams tests ---

    #[test]
    fn test_filter_params() {
        let payload = MemoryOperationPayload {
            user_id: Some("u1".into()),
            memory_type: Some("factual".into()),
            limit: Some(25),
            ..Default::default()
        };
        let params = FilterParams::from_payload(&payload, 100).unwrap();
        assert_eq!(params.user_id.as_deref(), Some("u1"));
        assert_eq!(params.memory_type.as_deref(), Some("factual"));
        assert_eq!(params.limit, 25);
    }

    #[test]
    fn test_filter_params_default_limit() {
        let payload = MemoryOperationPayload::default();
        let params = FilterParams::from_payload(&payload, 50).unwrap();
        assert_eq!(params.limit, 50);
    }

    // --- MCP tool definitions ---

    #[test]
    fn test_get_mcp_tool_definitions_count() {
        let tools = get_mcp_tool_definitions();
        assert_eq!(tools.len(), 24);
    }

    #[test]
    fn test_mcp_tool_names() {
        let tools = get_mcp_tool_definitions();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"system_status"));
        assert!(names.contains(&"add_content_memory"));
        assert!(names.contains(&"add_intuitive_memory"));
        assert!(names.contains(&"begin_store_document"));
        assert!(names.contains(&"store_document_part"));
        assert!(names.contains(&"process_document"));
        assert!(names.contains(&"status_process_document"));
        assert!(names.contains(&"list_document_sessions"));
        assert!(names.contains(&"cancel_process_document"));
        assert!(names.contains(&"query_memory"));
        assert!(names.contains(&"list_memories"));
        assert!(names.contains(&"get_memory"));
        assert!(names.contains(&"navigate_memory"));
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
        use crate::types::{Memory, MemoryMetadata, MemoryType, Relation};

        let mut metadata = MemoryMetadata::new(MemoryType::Conversational);
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
    use crate::operations::{MemoryOperationPayload, QueryParams, RelationInput, StoreParams};
    use crate::types::{Memory, MemoryMetadata, MemoryType};

    #[test]
    fn test_store_params_extracts_context() {
        let payload = MemoryOperationPayload {
            content: Some("test content".into()),
            user_id: Some("u1".into()),
            context: Some(vec!["project-alpha".into()]),
            ..Default::default()
        };

        let params = StoreParams::from_payload(&payload, None, None).unwrap();
        assert!(params.context.is_some());
        assert_eq!(params.context.unwrap(), vec!["project-alpha"]);
    }

    #[test]
    fn test_query_params_extracts_context() {
        let payload = MemoryOperationPayload {
            query: Some("find memories".into()),
            context: Some(vec!["recipes".into()]),
            ..Default::default()
        };

        let params = QueryParams::from_payload(&payload, 10).unwrap();
        assert!(params.context.is_some());
        assert_eq!(params.context.unwrap(), vec!["recipes"]);
    }

    #[test]
    fn test_memory_to_json_includes_context() {
        let mut meta = MemoryMetadata::new(MemoryType::Factual);
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
        let meta = MemoryMetadata::new(MemoryType::Factual);

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
