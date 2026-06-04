use serde_json::{json, Value};

pub struct McpToolDefinition {
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub input_schema: Value,
    pub output_schema: Option<Value>,
}

pub fn get_mcp_tool_definitions() -> Vec<McpToolDefinition> {
    vec![
        McpToolDefinition {
            name: "system_status".into(),
            title: Some("System Status".into()),
            description: Some(
                "IMPORTANT: Call this tool first before any other tool. \
                 Returns the current status of the memory system including: \
                 backend type (local or remote), model availability and reachability, \
                 token usage statistics, model download status, and configuration details. \
                 It is preferable to use 'default' as the bank name for other tools, unless situations warrant otherwise.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "backend": {"type": "string"},
                    "state": {"type": "string"},
                    "llm_model": {"type": "string"},
                    "embedding_model": {"type": "string"},
                    "llm_available": {"type": "boolean"},
                    "embedding_available": {"type": "boolean"},
                    "total_llm_calls": {"type": "integer"},
                    "total_embedding_calls": {"type": "integer"},
                    "total_prompt_tokens": {"type": "integer"},
                    "total_completion_tokens": {"type": "integer"},
                    "details": {"type": "object"}
                }
            })),
        },
        McpToolDefinition {
            name: "add_content_memory".into(),
            title: Some("Add Content Memory (Raw/Unprocessed)".into()),
            description: Some("Add raw content to memory WITHOUT any AI transformation. The content is stored and embedded EXACTLY AS-IS, preserving all original phrases, keywords, and structure. Use this when: (1) you need EXACT PHRASE searchability - e.g., finding 'vegan chili' or '#PlankChallenge' later, (2) storing conversation logs, documents, or code snippets where original text matters, (3) you want predictable semantic search based on the actual content, not AI-extracted interpretations. For AI-processed structured facts and insights instead, use add_intuitive_memory.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The specific fact or piece of information to store. Should be concise and atomic."
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata key-value pairs (e.g., source file, page number, timestamp, author). Strongly recommended for tracing the origin of information."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional. Only needed if multiple users share the same bank. Omit for single-user setups."
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Agent ID associated with the memory"
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["conversational", "procedural", "factual", "semantic", "episodic", "personal"],
                        "description": "Type of memory",
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of topics associated with the memory"
                    },
                    "context": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of context tags associated with the memory"
                    },
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "relation": {"type": "string"},
                                "target": {"type": "string"}
                            },
                            "required": ["relation", "target"]
                        },
                        "description": "Optional list of relations to other entities or memories"
                    },
                    "bank": {
                        "type": "string",
                        "description": "Optional memory bank name. Defaults to 'default' if not specified."
                    },
                    "auto_link": {
                        "type": "boolean",
                        "description": "Whether to automatically create 'references' relations to semantically similar existing memories. Defaults to server config (threshold 0.75). Set false to disable for this call."
                    },
                    "event_at": {
                        "type": "string",
                        "description": "Optional ISO 8601 datetime describing when the event actually happened (i.e. the date the content refers to, not when it was stored). Used by get_timeline / get_timeline_graph to form a chronological graph. Only meaningful for L0 raw content; higher layers derive it automatically. If omitted, falls back to created_at at query time."
                    }
                },
                "required": ["content"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string"},
                            "user_id": {"type": "string"},
                            "agent_id": {"type": "string"}
                        }
                    },
                    "error": {"type": "string"}
                }
            })),
        },
        McpToolDefinition {
            name: "store_memories".into(),
            title: Some("Store Multiple Content Memories (Batch)".into()),
            description: Some(
                "Store multiple content memories in a single call. \
                 Each item is stored independently as raw content. \
                 Use this for bulk ingestion — much faster than calling \
                 add_content_memory multiple times.\n\n\
                 Rules:\n\
                 - items array cannot be empty. Each item must have non-empty content.\n\
                 - Invalid items cause the entire batch to be rejected before any storage.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "The fact or information to store"},
                                "memory_type": {"type": "string", "description": "Type of memory (default: 'conversational')"},
                                "topics": {"type": "array", "items": {"type": "string"}},
                                "context": {"type": "array", "items": {"type": "string"}},
                                "relations": {"type": "array", "items": {"type": "object", "properties": {"relation": {"type": "string"}, "target": {"type": "string"}}, "required": ["relation", "target"]}},
                                "metadata": {"type": "object"},
                                "event_at": {"type": "string", "description": "Optional ISO 8601 datetime describing when this item's event actually happened. Used by get_timeline."}
                            },
                            "required": ["content"]
                        }
                    },
                    "bank": {"type": "string", "description": "Memory bank name (default: 'default')"}
                },
                "required": ["items"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "results": {"type": "array", "items": {"type": "object"}},
                            "total": {"type": "integer"}
                        }
                    },
                    "error": {"type": "string"}
                }
            })),
        },
        McpToolDefinition {
            name: "add_intuitive_memory".into(),
            title: Some("Add Intuitive Memory (AI-Processed/Structured)".into()),
            description: Some("Add memories with AI-powered extraction and structuring. The LLM analyzes your content, extracts key facts, organizes them into atomic insights, and generates searchable keywords. Use this when: (1) you want STRUCTURED, REASONING-READY memories - the AI extracts key facts and relationships, (2) you need CONDENSED insights from long conversations or documents, (3) you want AUTOMATIC KEYWORD EXTRACTION for hybrid search (searches will match both semantic meaning AND extracted keywords). IMPORTANT: Original text is TRANSFORMED by AI (e.g., 'I shared my vegan chili recipe' becomes '{\"topic\": \"Recipe sharing\", \"dish\": \"vegan chili\"}'). For preserving exact original phrases instead, use add_content_memory.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "description": "The role of the speaker (e.g., 'user', 'assistant', 'system')"},
                                "content": {"type": "string", "description": "The content of the message"},
                                "name": {"type": "string", "description": "Optional name of the speaker"}
                            },
                            "required": ["role", "content"]
                        },
                        "description": "The list of messages to extract facts from."
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata key-value pairs to attach to the extracted memories."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user ID."
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Optional agent ID."
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["conversational", "procedural", "factual", "semantic", "episodic", "personal"],
                        "description": "Type of memory to assign to the extracted facts. Defaults to 'conversational'.",
                    },
                    "bank": {
                        "type": "string",
                        "description": "Optional memory bank name. Defaults to 'default' if not specified."
                    },
                    "source_memory_id": {
                        "type": "string",
                        "description": "Optional memory ID to link this intuitive memory to. Automatically creates a 'derived_from' relation, enabling navigation from structured insights back to source content. Use this when creating an intuitive memory based on a content memory created with add_content_memory."
                    },
                    "event_at": {
                        "type": "string",
                        "description": "Optional ISO 8601 datetime. If provided, applied to all extracted memories that don't carry their own event_at. Used by get_timeline / get_timeline_graph to form a chronological graph."
                    }
                },
                "required": ["messages"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "memory": {"type": "string"},
                                        "event": {"type": "string"},
                                        "actor_id": {"type": "string"},
                                        "role": {"type": "string"},
                                        "previous_memory": {"type": "string"}
                                    }
                                }
                            },
                            "user_id": {"type": "string"},
                            "agent_id": {"type": "string"}
                        }
                    },
                    "error": {"type": "string"}
                }
            })),
        },
        McpToolDefinition {
            name: "upload_document".into(),
            title: Some("Upload Document (Auto-Chunk)".into()),
            description: Some("Upload a file with automatic server-side chunking and processing. The file is read, split into chunks, and ingested into memory. Supports all file types and handles chunking internally.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "Absolute path to the file to upload" },
                    "file_name": { "type": "string", "description": "Optional name for the file (defaults to basename of file_path)" },
                    "mime_type": { "type": "string", "description": "Optional MIME type (defaults to text/plain)" },
                    "chunk_size": { "type": "integer", "description": "Optional chunk size in characters (defaults to document_chunk_size from config)" },
                     "process_immediately": { "type": "boolean", "description": "If true, starts processing after upload (default: true)" },
                    "memory_type": {
                        "type": "string",
                        "enum": ["conversational", "procedural", "factual", "semantic", "episodic", "personal"],
                        "description": "Type of memory. Defaults to 'semantic'.",
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of topics"
                    },
                    "context": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of context tags"
                    },
                    "user_id": { "type": "string" },
                    "agent_id": { "type": "string" },
                    "bank": { "type": "string", "description": "Optional memory bank name." },
                    "event_at": { "type": "string", "description": "ISO 8601 datetime for when the document's events occurred. Defaults to the upload time if not provided." }
                },
                "required": ["file_path"]
            }),
            output_schema: None,
        },
        McpToolDefinition {
            name: "document_status".into(),
            title: Some("Get Document Session Status".into()),
            description: Some("Check the status of a specific document session (by session_id) or list all document sessions. Leave session_id empty to list all sessions.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string", "description": "Optional. If provided, returns status for that session. If omitted, lists all sessions." },
                    "bank": { "type": "string", "description": "Optional memory bank name. Defaults to 'default' if not specified." }
                },
                "required": []
            }),
            output_schema: None,
        },
        McpToolDefinition {
            name: "cancel_document".into(),
            title: Some("Cancel Document Session".into()),
            description: Some("Cancel an active document session and cleanup parts.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string", "description": "Session ID to cancel" },
                    "bank": { "type": "string", "description": "Optional memory bank name." }
                },
                "required": ["session_id"]
            }),
            output_schema: None,
        },
        McpToolDefinition {
            name: "update_memory".into(),
            title: Some("Update Memory".into()),
            description: Some("Update an existing memory (content and/or relations) by ID. Use this to refine knowledge or add new graph connections found later.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the memory to update"
                    },
                    "content": {
                        "type": "string",
                        "description": "New content for the memory (optional)"
                    },
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "relation": { "type": "string", "description": "The type of relationship" },
                                "target": { "type": "string", "description": "The target entity" }
                            },
                            "required": ["relation", "target"]
                        },
                        "description": "New relations to append to existing ones (optional)"
                    },
                    "bank": {
                        "type": "string",
                        "description": "Memory bank name (default: 'default')"
                    }
                },
                "required": ["memory_id"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"}
                },
                "required": ["success", "message"]
            })),
        },
        McpToolDefinition {
            name: "query_memory".into(),
            title: Some("Query Memory (Hybrid Search + Graph Traversal)".into()),
            description: Some(
                "Search memories using hybrid semantic + keyword search with optional graph traversal. \
                \n\n\
                **Standard Search (Default)**: Performs semantic similarity search and boosts scores for memories with matching keywords in metadata.keywords. \
                Use 'keyword_only': true to search ONLY by keyword matching (faster, no embedding required). \
                \n\n\
                **Graph Traversal (Optional)**: Enable graph_traversal to follow memory relations (derived_from, mentions, knows, etc.) \
                and discover related content through multi-hop reasoning. Use this for: \
                - Finding all insights derived from a conversation (provenance search) \
                - Discovering related memories via any relation type \
                - Multi-hop reasoning (e.g., 'find facts that mention X, then facts related to those') \
                - Navigating from insights back to source content \
                \n\n\
                Use the 'bank' parameter to search in a specific memory bank. Ensure system_status is called at least once.".into()
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query string for semantic search"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["conversational", "procedural", "factual", "semantic", "episodic", "personal"],
                        "description": "Type of memory to filter by"
                    },
                    "min_salience": {
                        "type": "number",
                        "description": "Minimum salience/importance score threshold (0-1)"
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topics to filter memories by"
                    },
                    "context": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Context tags for semantic scoping. The query will be matched against context embeddings to pre-filter results."
                    },
                    "keyword_only": {
                        "type": "boolean",
                        "description": "If true, search ONLY by keyword matching without semantic similarity. Useful for exact phrase matching when you know keywords were extracted. Default: false (hybrid search).",
                        "default": false
                    },
                    "keyword_split_ratio": {
                        "type": "number",
                        "description": "Ratio of results to fill from raw keyword matching vs. semantic/intuitive search. 0.0 = all semantic (default), 1.0 = all keyword, 0.2 = 20% keyword + 80% semantic. Each result is tagged with 'source': 'intuitive' or 'raw'.",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.2
                    },
                    "user_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "created_after": {
                        "type": "string",
                        "description": "Find memories created after this ISO 8601 datetime"
                    },
                    "created_before": {
                        "type": "string",
                        "description": "Find memories created before this ISO 8601 datetime"
                    },
                    "event_after": {
                        "type": "string",
                        "description": "Find memories whose event_at (or, if absent, created_at) is after this ISO 8601 datetime. Use with get_timeline-style time-window queries."
                    },
                    "event_before": {
                        "type": "string",
                        "description": "Find memories whose event_at (or, if absent, created_at) is before this ISO 8601 datetime."
                    },
                    "bank": {
                        "type": "string",
                        "description": "Memory bank name to search in (default: 'default')"
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Override the similarity threshold (0.0-1.0). Lower values return more results. Default uses config value (~0.2).",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "pyramid_config": {
                        "type": "object",
                        "description": "Optional: Configure hierarchical pyramid search across abstraction layers (L0 raw content → L4 strategic insights). \
                        Pyramid search distributes result slots across layers, so you get both concrete facts and abstract insights in a single query. \
                        When omitted, defaults to bottom-heavy allocation favoring concrete L0 facts.",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["bottom_heavy", "balanced", "top_heavy", "dynamic", "none"],
                                "description": "Allocation strategy across layers. \
                                bottom_heavy: More L0 facts, fewer abstract concepts (default). \
                                balanced: Equal distribution across layers. \
                                top_heavy: More abstract concepts, fewer concrete facts. \
                                dynamic: LLM classifies query intent automatically (requires use_llm_query_classification config flag). \
                                none: Skip pyramid assembly, return flat results sorted by raw score.",
                                "default": "bottom_heavy"
                            },
                            "layer_weights": {
                                "type": "object",
                                "description": "Custom per-layer weight overrides for fine-grained control. Keys are layer levels (0-4), values are positive weights. \
                                Higher weight = more result slots allocated to that layer.",
                                "additionalProperties": {"type": "number"}
                            },
                            "per_layer_multiplier": {
                                "type": "number",
                                "description": "Multiplier for per-layer search limit (default: 2.0). Actual per-layer limit = (total_limit * multiplier).max(5). \
                                Higher values search more memories per layer before assembly, improving result quality at the cost of speed.",
                                "default": 2.0
                            }
                        }
                    },
                    "graph_traversal": {
                        "type": "object",
                        "description": "Optional: Enable graph traversal to follow memory relations (derived_from, mentions, knows, etc.) and discover related content through multi-hop reasoning. \
                        Use cases: (1) Provenance search - find all insights derived from a conversation, \
                        (2) Context expansion - find memories related to a concept via any relation, \
                        (3) Multi-hop reasoning - find facts that mention X, then facts related to those, \
                        (4) Source navigation - find the raw content an insight came from. \
                        Default: disabled (standard semantic search only).",
                        "properties": {
                            "enabled": {
                                "type": "boolean",
                                "description": "Enable graph traversal (default: false)",
                                "default": false
                            },
                            "max_depth": {
                                "type": "integer",
                                "description": "Maximum number of hops to traverse from entry points (default: 2, max: 5). Higher values discover more distant relations but increase query time. Recommended: 2-3 for most use cases.",
                                "default": 2,
                                "minimum": 1,
                                "maximum": 5
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["outgoing", "incoming", "both"],
                                "description": "Traversal direction: 'outgoing' follows relations FROM the entry memory (e.g., find all memories this memory references), 'incoming' follows relations TO the entry memory (e.g., find all memories that reference this one), 'both' for bidirectional traversal (default, most comprehensive)",
                                "default": "both"
                            },
                            "relation_types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional filter to only follow specific relation types (e.g., [\"derived_from\", \"mentions\", \"knows\"]). If omitted, all relation types are followed. Use this to constrain traversal to specific relationship patterns."
                            },
                            "entry_point_limit": {
                                "type": "integer",
                                "description": "Maximum number of top-scoring memories from semantic search to use as graph traversal entry points (default: 5, max: 10). Higher values provide broader coverage but may increase query time.",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 10
                            },
                            "include_paths": {
                                "type": "boolean",
                                "description": "Include detailed traversal paths and graph scoring information in response (default: false). When true, each result includes 'graph_info' with entry_distance (hops from entry), path_from_entry (relation chain), relation_boost, and final_score breakdown.",
                                "default": false
                            }
                        }
                    }
                },
                "required": ["query"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "count": {"type": "number"},
                    "memories": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["success", "count", "memories"]
            })),
        },
        McpToolDefinition {
            name: "search_memory".into(),
            title: Some("Search Memory (Simple)".into()),
            description: Some(
                "Search memories across all abstraction layers with sensible defaults. \
                 Use this for everyday retrieval — no configuration needed. \
                 For advanced queries with graph traversal, custom pyramid allocation, \
                 or keyword/semantic split control, use query_memory instead.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for — a natural language question, topic, or phrase"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                        "default": 10
                    },
                    "bank": {
                        "type": "string",
                        "description": "Memory bank name to search in (default: 'default')"
                    }
                },
                "required": ["query"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "count": {"type": "number"},
                    "memories": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["success", "count", "memories"]
            })),
        },
        McpToolDefinition {
            name: "list_memories".into(),
            title: Some("List Memories".into()),
            description: Some("Retrieve memories with optional filtering. Use the 'bank' parameter to list from a specific memory bank.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of memories to return",
                        "default": 100,
                        "maximum": 1000
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["conversational", "procedural", "factual", "semantic", "episodic", "personal"]
                    },
                    "user_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "created_after": {"type": "string"},
                    "created_before": {"type": "string"},
                    "event_after": {
                        "type": "string",
                        "description": "ISO 8601 — only return memories whose event_at (or, if absent, created_at) is after this."
                    },
                    "event_before": {
                        "type": "string",
                        "description": "ISO 8601 — only return memories whose event_at (or, if absent, created_at) is before this."
                    },
                    "bank": {
                        "type": "string",
                        "description": "Memory bank name to list from (default: 'default')"
                    }
                }
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "count": {"type": "number"},
                    "memories": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["success", "count", "memories"]
            })),
        },
        McpToolDefinition {
            name: "get_memory".into(),
            title: Some("Get Memory by ID".into()),
            description: Some("Retrieve a specific memory by its exact ID. Use the 'bank' parameter to look in a specific memory bank.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Exact ID of the memory to retrieve"
                    },
                    "bank": {
                        "type": "string",
                        "description": "Memory bank name to look in (default: 'default')"
                    }
                },
                "required": ["memory_id"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "memory": {"type": "object"}
                },
                "required": ["success", "memory"]
            })),
        },
        McpToolDefinition {
            name: "navigate_memory".into(),
            title: Some("Navigate Memory Abstraction Hierarchy".into()),
            description: Some(
                "Traverse the layered abstraction hierarchy from a memory node in either direction. \
                 'zoom_out' returns higher-layer (more abstract) memories that were derived FROM this memory. \
                 'zoom_in' returns lower-layer (more detailed) source memories that this memory was abstracted FROM. \
                 'both' returns both directions. Use this to explore the knowledge graph built by the abstraction pipeline. \
                 The get_memory tool also includes an 'abstracted_into' field in metadata showing which higher-layer memories reference this one.\n\n\
                 Rules:\n\
                 - memory_id must be a valid ID of an existing memory.\n\
                 - levels is clamped to max 5.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to navigate from"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["zoom_in", "zoom_out", "both"],
                        "description": "Direction to navigate: 'zoom_out' towards abstraction, 'zoom_in' towards detail, 'both' for both directions (default: 'both')"
                    },
                    "levels": {
                        "type": "integer",
                        "description": "Number of levels to traverse for zoom_in (default: 1, max: 5)",
                        "minimum": 1,
                        "maximum": 5
                    },
                    "bank": {
                        "type": "string",
                        "description": "Memory bank name (default: 'default')"
                    }
                },
                "required": ["memory_id"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "source_memory_id": {"type": "string"},
                    "source_layer": {"type": "integer"},
                    "zoom_in": {
                        "type": "array",
                        "description": "Lower-layer (more detailed) memories this was abstracted FROM",
                        "items": {"type": "object"}
                    },
                    "zoom_out": {
                        "type": "array",
                        "description": "Higher-layer (more abstract) memories that abstract FROM this one",
                        "items": {"type": "object"}
                    }
                },
                "required": ["success", "source_memory_id", "source_layer"]
            })),
        },
        McpToolDefinition {
            name: "get_timeline".into(),
            title: Some("Get Timeline of Memories".into()),
            description: Some(
                "Return a chronological list of memories grouped by time bucket. \
                 Use this to answer 'what happened in the last 2 days', 'show me \
                 events from this week', or to browse a memory bank on a timeline. \
                 Memories are bucketed by their `event_at` (the date the content \
                 refers to), not by `created_at` (when stored). If `event_at` is \
                 missing, it falls back to `created_at`. \n\n\
                 Use `granularity` to control bucket size (hour, day, week, month, none). \
                 Use `start` / `end` to bound the time window (defaults: end=now, \
                 start=end-7d). Set `include_derived=true` to also return L1+ \
                 abstractions (L0 raw content only by default). For a full graph \
                 (nodes + edges) see get_timeline_graph.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "ISO 8601 datetime — start of time window. Default: end - 7 days."
                    },
                    "end": {
                        "type": "string",
                        "description": "ISO 8601 datetime — end of time window. Default: now."
                    },
                    "granularity": {
                        "type": "string",
                        "enum": ["hour", "day", "week", "month", "none"],
                        "description": "Bucket size. 'none' = single bucket covering the whole window. Default: 'day'."
                    },
                    "bank": {"type": "string", "description": "Memory bank (default: 'default')"},
                    "user_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter to memories tagged with any of these topics."
                    },
                    "max_results_per_bucket": {
                        "type": "integer",
                        "description": "Cap on memories returned per bucket. Default: 50."
                    },
                    "include_derived": {
                        "type": "boolean",
                        "description": "Include L1+ derived memories (default: false, L0 only)."
                    },
                    "order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort order within each bucket. Default: 'asc' (chronological)."
                    }
                }
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "granularity": {"type": "string"},
                    "total_count": {"type": "integer"},
                    "bucket_count": {"type": "integer"},
                    "buckets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "string"},
                                "end": {"type": "string"},
                                "label": {"type": "string", "description": "Human-readable bucket label, e.g. '2026-06-02'."},
                                "count": {"type": "integer"},
                                "memories": {"type": "array", "items": {"type": "object"}}
                            }
                        }
                    }
                }
            })),
        },
        McpToolDefinition {
            name: "get_timeline_graph".into(),
            title: Some("Get Chronological Graph".into()),
            description: Some(
                "Return a chronological graph of memories: nodes (memories sorted by \
                 event_at) plus edges. Edges include: (1) auto-derived temporal edges \
                 (`happened_after`, `happens_within`) computed from event_at proximity, \
                 and (2) optional semantic edges (derived_from, mentions, etc.) traversed \
                 from the existing relation graph. Use this to render a timeline as a \
                 network diagram, run graph algorithms over the chronological graph, or \
                 explore multi-hop relations within a time window. \n\n\
                 Parameters mirror get_timeline (start, end, granularity, bank, filters) \
                 plus graph-specific options: max_depth, relation_types, \
                 temporal_edge_window_secs, include_simultaneous, include_semantic_edges.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "ISO 8601 — start of time window."},
                    "end": {"type": "string", "description": "ISO 8601 — end of time window."},
                    "granularity": {"type": "string", "enum": ["hour", "day", "week", "month", "none"]},
                    "bank": {"type": "string"},
                    "user_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}},
                    "max_results_per_bucket": {"type": "integer"},
                    "include_derived": {"type": "boolean"},
                    "order": {"type": "string", "enum": ["asc", "desc"]},
                    "max_depth": {
                        "type": "integer",
                        "description": "Semantic-relation hops from each timeline node (default 1, max 3)."
                    },
                    "relation_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional whitelist of semantic relation types to follow."
                    },
                    "temporal_edge_window_secs": {
                        "type": "integer",
                        "description": "Window (seconds) for auto `happened_after` edges (default 86400 = 1 day)."
                    },
                    "include_simultaneous": {
                        "type": "boolean",
                        "description": "Also auto-derive `happens_within` edges for near-simultaneous events."
                    },
                    "simultaneous_window_secs": {
                        "type": "integer",
                        "description": "Window (seconds) for `happens_within` (default 60)."
                    },
                    "include_semantic_edges": {
                        "type": "boolean",
                        "description": "Include semantic-relation edges (default true)."
                    }
                }
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "granularity": {"type": "string"},
                    "stats": {
                        "type": "object",
                        "properties": {
                            "node_count": {"type": "integer"},
                            "edge_count": {"type": "integer"},
                            "temporal_edge_count": {"type": "integer"},
                            "semantic_edge_count": {"type": "integer"}
                        }
                    },
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "event_at": {"type": "string"},
                                "event_end": {"type": "string"},
                                "layer": {"type": "integer"},
                                "bucket": {"type": "string"},
                                "memory": {"type": "object"}
                            }
                        }
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "type": {"type": "string"},
                                "delta_secs": {"type": "integer", "description": "For temporal edges: time between events in seconds."},
                                "depth": {"type": "integer", "description": "For semantic edges: hop count from the timeline node."}
                            }
                        }
                    }
                }
            })),
        },
        McpToolDefinition {
            name: "list_memory_banks".into(),
            title: Some("List Memory Banks".into()),
            description: Some(
                "List all available memory banks. Each bank is an isolated memory store \
                 with its own database file. Returns bank names, paths, memory counts, \
                 and descriptions. Use different banks to organize memories by project, \
                 topic, or domain.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "banks_dir": {"type": "string"},
                    "banks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "path": {"type": "string"},
                                "memory_count": {"type": "integer"},
                                "description": {"type": "string"},
                                "loaded": {"type": "boolean"}
                            }
                        }
                    }
                }
            })),
        },
        McpToolDefinition {
            name: "create_memory_bank".into(),
            title: Some("Create Memory Bank".into()),
            description: Some(
                "Create a new named memory bank for organizing memories by context. \
                 Bank names may contain only alphanumeric characters, hyphens, and \
                 underscores (max 64 chars). If the bank already exists, returns its \
                 info without modification.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the memory bank (e.g., 'my-project', 'research_notes'). Only alphanumeric, hyphens, and underscores allowed."
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional human-readable description of the bank's purpose"
                    }
                },
                "required": ["name"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "bank": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "path": {"type": "string"},
                            "memory_count": {"type": "integer"},
                            "description": {"type": "string"},
                            "loaded": {"type": "boolean"}
                        }
                    }
                }
            })),
        },
        McpToolDefinition {
            name: "backup_bank".into(),
            title: Some("Backup Memory Bank".into()),
            description: Some(
                "Create a versioned backup of a memory bank. Each backup produces a \
                 timestamped .db file and a .manifest.json sidecar containing the version \
                 number, memory count, and SHA-256 checksum for integrity verification. \
                 Multiple backups of the same bank are kept side-by-side with incrementing \
                 version numbers.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the bank to back up (default: 'default')",
                        "default": "default"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination directory for the backup file. Defaults to ~/llm-mem-backups/ if omitted."
                    }
                }
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "backup_path": {"type": "string"},
                    "manifest": {
                        "type": "object",
                        "properties": {
                            "version": {"type": "integer"},
                            "created_at": {"type": "string"},
                            "memory_count": {"type": "integer"},
                            "sha256": {"type": "string"},
                            "size_bytes": {"type": "integer"}
                        }
                    }
                }
            })),
        },
        McpToolDefinition {
            name: "restore_bank".into(),
            title: Some("Restore Memory Bank".into()),
            description: Some(
                "Restore a memory bank from a backup .db file. Supports two modes: \
                 'replace' (default) overwrites the bank entirely — requires confirm: true. \
                 'merge' additively imports memories from the backup, skipping duplicates \
                 (matched by content hash) — no confirmation needed. \
                 If a .manifest.json sidecar exists, the SHA-256 checksum is verified \
                 before restoring.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the bank to restore into (default: 'default')",
                        "default": "default"
                    },
                    "source": {
                        "type": "string",
                        "description": "Absolute path to the backup .db file to restore from"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["replace", "merge"],
                        "description": "'replace' overwrites the bank (requires confirm). 'merge' additively imports non-duplicate memories.",
                        "default": "replace"
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Required for 'replace' mode. Ask the user for confirmation first."
                    }
                },
                "required": ["source"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "restored_path": {"type": "string"},
                    "imported": {"type": "integer"},
                    "skipped_duplicates": {"type": "integer"},
                    "total_after_merge": {"type": "integer"},
                    "source": {"type": "string"}
                }
            })),
        },
        McpToolDefinition {
            name: "rename_memory_bank".into(),
            title: Some("Rename Memory Bank".into()),
            description: Some(
                "Rename a memory bank, including its database file and session database. \
                 This operation is atomic — both the main database (.db) and session \
                 database (.sessions.db) are renamed together, ensuring consistency. \
                 If the rename fails at any point, the operation is rolled back. \
                 Bank names may contain only alphanumeric characters, hyphens, and \
                 underscores (max 64 chars).".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "old_name": {
                        "type": "string",
                        "description": "Current name of the bank to rename (default: 'default')"
                    },
                    "new_name": {
                        "type": "string",
                        "description": "New name for the bank. Must be unique and follow naming rules (alphanumeric, hyphens, underscores, max 64 chars)."
                    }
                },
                "required": ["old_name", "new_name"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "old_name": {"type": "string"},
                    "new_name": {"type": "string"}
                }
            })),
        },
        McpToolDefinition {
            name: "cleanup_resources".into(),
            title: Some("Cleanup Resources".into()),
            description: Some(
                "Cleanup system resources. Supports selective deletion of memory banks or full models cleanup. \
                 For bank deletion: you MUST ask the user for explicit confirmation before calling this tool. \
                 Pass their confirmation as a specific phrase in the 'confirm' field. \
                 For model cleanup: set confirm to true.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "enum": ["models", "banks"],
                        "description": "Resource type to cleanup. 'models' deletes LLM files. 'banks' deletes memory stores.",
                        "default": "models"
                    },
                    "name": {
                        "type": "string",
                        "description": "Specific bank name to delete. If omitted when target='banks', ALL banks will be deleted!"
                    },
                    "confirm": {
                        "description": "For target='models': set to true. For target='banks': MUST be the exact string 'I confirm this data will be permanently lost' — ask the user before sending this."
                    }
                },
                "required": ["confirm"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"}
                }
            })),
        },
        McpToolDefinition {
            name: "help".into(),
            title: Some("Help & Usage Guide".into()),
            description: Some(
                "Get the full usage guide for llm-mem: layered memory architecture, domain patterns, memory types, and best practices. Call this once to understand the system, not on every operation.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            output_schema: None,
        },
        McpToolDefinition {
            name: "start_abstraction_pipeline".into(),
            title: Some("Start Abstraction Pipeline".into()),
            description: Some(
                "Start the background abstraction pipeline workers (L0→L1→L2→L3+). \
                 The pipeline creates progressive abstractions: L0 raw content → L1 summaries → L2 semantic links → L3 concepts. \
                 Use this when auto_enhance is disabled or you want to manually control abstraction processing. \
                 Once started, workers run continuously in the background.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"}
                }
            })),
        },
        McpToolDefinition {
            name: "stop_abstraction_pipeline".into(),
            title: Some("Stop Abstraction Pipeline".into()),
            description: Some(
                "Stop the background abstraction pipeline workers. Workers will finish current tasks and shut down gracefully. \
                 Use this to pause abstraction processing or conserve resources.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"}
                }
            })),
        },
        McpToolDefinition {
            name: "trigger_abstraction".into(),
            title: Some("Trigger Abstraction Now".into()),
            description: Some(
                "Trigger immediate one-shot abstraction processing. Unlike start_abstraction_pipeline, this runs once and doesn't start background workers. \
                 Use target_layer: 1 for L0→L1 (summaries), 2 for L1→L2 (semantic links), 3 for L2→L3 (concepts), or 0/all for all layers. \
                 Requires the pipeline to be running (call start_abstraction_pipeline first if needed).".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "target_layer": {
                        "type": "integer",
                        "description": "Target layer: 1=L0→L1, 2=L1→L2, 3=L2→L3, 0=all. Default: 1",
                        "default": 1
                    }
                },
                "required": []
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "l0_to_l1_created": {"type": "integer"},
                    "l1_to_l2_created": {"type": "integer"},
                    "l2_to_l3_created": {"type": "integer"},
                    "errors": {"type": "array", "items": {"type": "string"}}
                }
            })),
        },
        McpToolDefinition {
            name: "check_consistency".into(),
            title: Some("Check Memory Bank Consistency".into()),
            description: Some(
                "Run a consistency check on a memory bank to detect orphaned abstractions, \
                 broken relations, and other integrity issues. Returns a report with all \
                 detected issues categorized by severity.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "bank": {
                        "type": "string",
                        "description": "Memory bank name to check (default: 'default')"
                    }
                },
                "required": []
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "total_memories": {"type": "integer"},
                    "issues": {
                        "type": "array",
                        "items": {"type": "object"}
                    },
                    "errors": {"type": "integer"},
                    "warnings": {"type": "integer"},
                    "infos": {"type": "integer"}
                }
            })),
        },
        McpToolDefinition {
            name: "create_abstraction".into(),
            title: Some("Create Manual Abstraction".into()),
            description: Some(
                "Create a manual abstraction (L1/L2/L3) from specific source memory IDs. \
                 Unlike the automatic pipeline, this lets you specify exact sources and content. \
                 Use target_layer: 1 for L0→L1 (summaries), 2 for L1→L2 (semantic links), 3+ for higher layers.\n\n\
                 Rules:\n\
                 - content cannot be empty.\n\
                 - source_ids must be valid UUIDs of existing Active/Degraded memories. No duplicates.\n\
                 - target_layer must be >= 1 and strictly higher than all source layers.\n\
                 - The reverse relation (e.g., 'summarized_by') is created on each source automatically.\n\
                 - Default relation types by layer: 'summary_of' (L1), 'synthesizes' (L2), 'abstracts_to_concept' (L3+).".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The abstraction content (summary, synthesis, or concept)"
                    },
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of source memory IDs this abstraction derives from"
                    },
                    "target_layer": {
                        "type": "integer",
                        "description": "Target abstraction layer: 1=structural summary, 2=semantic, 3=concept, etc.",
                        "default": 1
                    },
                    "relation_type": {
                        "type": "string",
                        "description": "Relation type linking this abstraction to sources. Defaults based on layer: 'summary_of' (L1), 'synthesizes' (L2), 'abstracts_to_concept' (L3+)"
                    },
                    "user_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "bank": {
                        "type": "string",
                        "description": "Optional memory bank name"
                    }
                },
                "required": ["content", "source_ids", "target_layer"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string"},
                            "target_layer": {"type": "integer"},
                            "relation_type": {"type": "string"},
                            "source_count": {"type": "integer"},
                            "reverse_relation": {"type": "string"},
                            "reverse_created": {"type": "boolean"}
                        }
                    }
                }
            })),
        },
        McpToolDefinition {
            name: "force_link".into(),
            title: Some("Force Link Two Memories".into()),
            description: Some(
                "Create a direct relation between two existing memories. \
                 The reverse relation is created automatically (e.g., 'references' creates 'referenced_by' on the target). \
                 Use this to manually connect memories the auto-linker missed, \
                 or to create custom relation types (contradicts, supports, depends_on, etc.).\n\n\
                 Rules:\n\
                 - source_id and target_id must be different valid UUIDs of existing Active/Degraded memories.\n\
                 - relation cannot be empty. Known types: references, contradicts, supports, depends_on, part_of, extends, similar_to, summary_of, synthesizes.\n\
                 - Hierarchical relations (summary_of, part_of, synthesizes) require the source to be at a higher layer than the target.\n\
                 - Duplicate links are rejected — check before linking.\n\
                 - strength is clamped to 0.0-1.0 (default 1.0).".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "ID of the source memory (the 'from' side)"
                    },
                    "relation": {
                        "type": "string",
                        "description": "Relation type: references, contradicts, supports, depends_on, part_of, etc."
                    },
                    "target_id": {
                        "type": "string",
                        "description": "ID of the target memory (the 'to' side)"
                    },
                    "strength": {
                        "type": "number",
                        "description": "Optional relation strength (0.0-1.0). Default: 1.0"
                    },
                    "bank": {
                        "type": "string",
                        "description": "Optional memory bank name"
                    }
                },
                "required": ["source_id", "relation", "target_id"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            "relation": {"type": "string"},
                            "target_id": {"type": "string"},
                            "reverse_relation": {"type": "string"},
                            "reverse_created": {"type": "boolean"}
                        }
                    }
                }
            })),
        },
        McpToolDefinition {
            name: "remove_relation".into(),
            title: Some("Remove Relation".into()),
            description: Some(
                "Remove a specific relation from a memory. \
                 Use this to clean up false positives from auto-linking or manual links. \
                 Specify the relation type and target ID to remove.\n\n\
                 Rules:\n\
                 - The reverse relation on the target is removed automatically.\n\
                 - relation_type and target_id cannot be empty. target_id must be a valid UUID.\n\
                 - Returns an error if the relation does not exist on the memory.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to remove the relation from"
                    },
                    "relation_type": {
                        "type": "string",
                        "description": "The relation type to remove (e.g., 'references', 'contradicts')"
                    },
                    "target_id": {
                        "type": "string",
                        "description": "The target memory ID in the relation"
                    },
                    "bank": {
                        "type": "string",
                        "description": "Optional memory bank name"
                    }
                },
                "required": ["memory_id", "relation_type", "target_id"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string"},
                            "removed_relation": {"type": "string"},
                            "removed_target": {"type": "string"},
                            "reverse_relation": {"type": "string"},
                            "reverse_cleaned": {"type": "boolean"}
                        }
                    }
                }
            })),
        },
        McpToolDefinition {
            name: "ingest".into(),
            title: Some("Ingest Content (Universal Decomposition)".into()),
            description: Some(
                "Ingest raw content in any format. Automatically detects format, decomposes into \
                 semantic chunks (L0), creates structural relations between chunks, and optionally \
                 auto-links to existing memories. Supports markdown, JSON, YAML, TOML, plain text, \
                 CSV, code (Rust/Python/JS/TS/Go/C++/Java + any brace-based language), PDF, DOCX, \
                 and images (PNG/JPEG/GIF/WebP). For binary formats, base64-encode the content and \
                 set content_encoding to 'base64'.\n\n\
                 Each L0 chunk preserves the exact raw content — nothing is added or summarized. \
                 Returns structured feedback with chunk IDs, relations, and any warnings about \
                 ambiguous parsing decisions.".into(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Raw content to ingest. For binary formats, base64-encoded."
                    },
                    "content_encoding": {
                        "type": "string",
                        "enum": ["base64"],
                        "description": "Content encoding (only 'base64' supported). Required for binary formats like PDF, DOCX, images."
                    },
                    "format_hint": {
                        "type": "string",
                        "description": "Optional format hint: markdown, json, yaml, toml, text"
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Optional file name for extension-based detection and provenance"
                    },
                    "bank": {
                        "type": "string",
                        "description": "Optional memory bank name"
                    },
                    "auto_link": {
                        "type": "boolean",
                        "description": "Auto-link chunks to existing memories (default: true)"
                    },
                    "generate_abstractions": {
                        "type": "boolean",
                        "description": "Generate L1+ interpretations (default: true)"
                    },
                    "max_chunk_size": {
                        "type": "integer",
                        "description": "Max characters per L0 chunk (default: 2000)"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "User-provided metadata (tags, source URL, etc.) attached to all chunks"
                    }
                },
                "required": ["content"]
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["success", "partial", "failed"]},
                            "session_id": {"type": "string"},
                            "format": {"type": "string"},
                            "detected_mime": {"type": "string"},
                            "byte_size": {"type": "integer"},
                            "l0_chunks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "memory_id": {"type": "string"},
                                        "node_type": {"type": "string"},
                                        "content_preview": {"type": "string"},
                                        "char_count": {"type": "integer"},
                                        "order": {"type": "integer"}
                                    }
                                }
                            },
                            "relations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_chunk_id": {"type": "string"},
                                        "target_chunk_id": {"type": "string"},
                                        "relation": {"type": "string"},
                                        "strength": {"type": "number"}
                                    }
                                }
                            },
                            "issues": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "severity": {"type": "string"},
                                        "message": {"type": "string"},
                                        "suggestion": {"type": "string"}
                                    }
                                }
                            },
                            "warnings": {"type": "array", "items": {"type": "string"}},
                            "format_hints_available": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            })),
        },
    ]
}
