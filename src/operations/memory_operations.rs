use serde_json::{Value, json};
use tracing::{error, info};

use crate::{
    memory::{MemoryManager},
    search::{GraphSearchEngine, TraversalConfig, TraversalDirection, GraphSearchResult, RelationHop},
    types::{Filters, Memory, ScoredMemory},
};

use super::params::*;
use super::requests::{
    AddMemoryRequest, BeginStoreDocumentRequest, CancelProcessDocumentRequest,
    GetRequest, ListDocumentSessionsRequest,
    ListRequest, MemoryOperationResponse, NavigateRequest,
    OperationError, OperationResult, ProcessDocumentRequest, QueryRequest,
    StoreDocumentPartRequest, StoreRequest, StatusProcessDocumentRequest, UpdateRequest,
    UploadDocumentRequest,
};
use super::serialization::memory_to_json;

use crate::document_session::{
    DocumentMetadata, DocumentSessionManager, SessionStatus,
};

/// Core operations handler for memory tools
pub struct MemoryOperations {
    memory_manager: std::sync::Arc<MemoryManager>,
    session_manager: Option<std::sync::Arc<DocumentSessionManager>>,
    default_user_id: Option<String>,
    default_agent_id: Option<String>,
    #[allow(dead_code)]
    default_limit: usize,
}

impl MemoryOperations {
    pub fn new(
        memory_manager: std::sync::Arc<MemoryManager>,
        default_user_id: Option<String>,
        default_agent_id: Option<String>,
        default_limit: usize,
    ) -> Self {
        Self {
            memory_manager,
            session_manager: None,
            default_user_id,
            default_agent_id,
            default_limit,
        }
    }

    pub fn with_session_manager(
        memory_manager: std::sync::Arc<MemoryManager>,
        session_manager: std::sync::Arc<DocumentSessionManager>,
        default_user_id: Option<String>,
        default_agent_id: Option<String>,
        default_limit: usize,
    ) -> Self {
        Self {
            memory_manager,
            session_manager: Some(session_manager),
            default_user_id,
            default_agent_id,
            default_limit,
        }
    }

    pub async fn store_memory(
        &self,
        req: StoreRequest,
    ) -> OperationResult<MemoryOperationResponse> {
        let mut params: StoreParams = req.into();
        params.user_id = params.user_id.or(self.default_user_id.clone());
        params.agent_id = params.agent_id.or(self.default_agent_id.clone());

        info!("Storing memory for user: {:?}", params.user_id);

        let metadata = super::helpers::build_metadata(
            &params.memory_type,
            params.user_id.clone(),
            params.agent_id.clone(),
            params.topics,
            params.context,
            params.relations,
            params.metadata,
        )?;


        match self.memory_manager.store_interactive(params.content, metadata).await {
            Ok(memory_id) => {
                // SELF in relations is resolved by store_with_options()
                info!("Memory stored successfully with ID: {}", memory_id);
                let data = json!({
                    "memory_id": memory_id,
                    "user_id": params.user_id,
                    "agent_id": params.agent_id
                });
                Ok(MemoryOperationResponse::success_with_data(
                    "Memory stored successfully",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to store memory: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to store memory: {}",
                    e
                )))
            }
        }
    }

    pub async fn add_memory(
        &self,
        req: AddMemoryRequest,
    ) -> OperationResult<MemoryOperationResponse> {
        let mut params: AddMemoryParams = req.into();
        params.user_id = params.user_id.or(self.default_user_id.clone());
        params.agent_id = params.agent_id.or(self.default_agent_id.clone());

        info!(
            "Adding memory from conversation for user: {:?}",
            params.user_id
        );

        let mut metadata = super::helpers::build_metadata(
            &params.memory_type,
            params.user_id.clone(),
            params.agent_id.clone(),
            params.topics,
            params.context,
            params.relations,
            params.metadata,
        )?;

        // Add automatic relation to source memory (for linking intuitive memories to content memories)
        if let Some(source_id) = params.source_memory_id {
            metadata.relations.push(crate::types::Relation {
                source: "SELF".to_string(),
                relation: "derived_from".to_string(),
                target: source_id,
                strength: None,
            });
        }

        match self
            .memory_manager
            .add_memory(&params.messages, metadata)
            .await
        {
            Ok(results) => {
                info!(
                    "Memory added successfully, {} actions performed",
                    results.len()
                );
                let data = json!({
                    "results": results,
                    "user_id": params.user_id,
                    "agent_id": params.agent_id
                });
                Ok(MemoryOperationResponse::success_with_data(
                    "Memory added successfully",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to add memory: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to add memory: {}",
                    e
                )))
            }
        }
    }

    pub async fn update_memory(
        &self,
        req: UpdateRequest,
    ) -> OperationResult<MemoryOperationResponse> {
        let memory_id = req.memory_id.clone();

        info!("Updating memory: {}", memory_id);

        let relations = req.relations.map(|rels| {
            rels.into_iter()
                .map(|r| crate::types::Relation {
                    source: "SELF".to_string(),
                    relation: r.relation,
                    target: r.target,
                    strength: None,
                })
                .collect()
        });

        match self
            .memory_manager
            .update(&memory_id, req.content, relations)
            .await
        {
            Ok(_) => Ok(MemoryOperationResponse::success(
                "Memory updated successfully",
            )),
            Err(e) => {
                error!("Failed to update memory: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to update memory: {}",
                    e
                )))
            }
        }
    }

    pub async fn query_memory(
        &self,
        req: QueryRequest,
    ) -> OperationResult<MemoryOperationResponse> {
        let params: QueryParams = req.into();

        info!("Querying memories with query: {}", params.query);

        let mut filters = Filters::default();

        if let Some(ref user_id) = params.user_id {
            filters.user_id = Some(user_id.clone());
        }
        if let Some(ref agent_id) = params.agent_id {
            filters.agent_id = Some(agent_id.clone());
        }
        if let Some(ref topics) = params.topics {
            filters.topics = Some(topics.clone());
        }
        if let Some(ref created_after) = params.created_after {
            filters.created_after = Some(*created_after);
        }
        if let Some(ref created_before) = params.created_before {
            filters.created_before = Some(*created_before);
        }

        // Pass keyword_only flag to filters for hybrid search
        if params.keyword_only {
            filters
                .custom
                .insert("keyword_only".to_string(), serde_json::Value::Bool(true));
        }

        // Check if deep graph traversal is enabled (legacy path)
        if let Some(ref graph_config) = params.graph_traversal {
            return self
                .handle_graph_traversal(&params, &filters, graph_config)
                .await;
        }

        // Default: Pyramid search with graph refinement.
        // When keyword_split_ratio > 0, also run keyword search and merge.
        let split_ratio = params.keyword_split_ratio.clamp(0.0, 1.0);
        let pyramid_results: Vec<crate::search::PyramidResult>;
        let keyword_results: Option<Vec<ScoredMemory>>;

        if split_ratio > 0.0 {
            let semantic_count =
                ((params.limit as f32 * (1.0 - split_ratio)).ceil() as usize).max(1);
            let keyword_count = params.limit.saturating_sub(semantic_count);

            let pyramid_fut = self.memory_manager.search_pyramid(
                &params.query,
                &filters,
                params.limit,
                &params.pyramid_config,
                params.similarity_threshold,
            );
            let keyword_fut = if keyword_count > 0 {
                Some(self.memory_manager.search_by_raw_content(
                    &params.query,
                    &filters,
                    keyword_count,
                ))
            } else {
                None
            };

            let (pyramid_res, kw_res) = tokio::join!(pyramid_fut, async {
                if let Some(fut) = keyword_fut {
                    fut.await.ok()
                } else {
                    None
                }
            });

            pyramid_results = pyramid_res
                .map_err(|e| OperationError::Runtime(format!("Pyramid search failed: {}", e)))?;
            keyword_results = kw_res;
        } else {
            pyramid_results = self
                .memory_manager
                .search_pyramid(
                    &params.query,
                    &filters,
                    params.limit,
                    &params.pyramid_config,
                    params.similarity_threshold,
                )
                .await
                .map_err(|e| OperationError::Runtime(format!("Pyramid search failed: {}", e)))?;
            keyword_results = None;
        }

        // Merge keyword results into pyramid results if split is active
        let mut all_results: Vec<crate::search::PyramidResult> = pyramid_results;

        if let Some(kw_results) = keyword_results {
            let semantic_ids: std::collections::HashSet<String> = all_results
                .iter()
                .map(|r| r.memory.memory.id.clone())
                .collect();

            let keyword_limit = params
                .limit
                .saturating_sub(all_results.len());
            let mut kw_added = 0usize;

            for kw in kw_results {
                if kw_added >= keyword_limit {
                    break;
                }
                if !semantic_ids.contains(&kw.memory.id) {
                    let layer = kw.memory.metadata.layer.level;
                    let layer_name = kw.memory.metadata.layer.name_or_default();
                    all_results.push(crate::search::PyramidResult {
                        memory: kw,
                        layer,
                        layer_name,
                        search_phase: "keyword_merged".to_string(),
                        graph_path: None,
                        source: "raw".to_string(),
                    });
                    kw_added += 1;
                }
            }

            all_results.sort_by(|a, b| {
                b.memory
                    .score
                    .partial_cmp(&a.memory.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            all_results.truncate(params.limit);
        }

        let count = all_results.len();
        let best_score = all_results.first().map(|r| r.memory.score);

        let memories_json: Vec<Value> = all_results
            .into_iter()
            .map(|r| {
                let mut memory_json = memory_to_json(&r.memory.memory);
                memory_json["score"] = json!(r.memory.score);
                memory_json["layer"] = json!(r.layer);
                memory_json["layer_name"] = json!(r.layer_name);
                memory_json["search_phase"] = json!(r.search_phase);
                memory_json["source"] = json!(r.source);
                if let Some(ref path) = r.graph_path {
                    memory_json["graph_path"] = serde_json::to_value(path).unwrap_or(json!(null));
                }
                let neighbors: Vec<Value> = r.memory.memory.metadata.relations.iter()
                    .map(|rel| json!({
                        "relation": rel.relation,
                        "target_id": rel.target,
                        "strength": rel.strength,
                    }))
                    .collect();
                if !neighbors.is_empty() {
                    memory_json["neighbors"] = json!(neighbors);
                }
                memory_json
            })
            .collect();

        let message = if count == 0 {
            let threshold_hint = if let Some(th) = params.similarity_threshold {
                format!(" Current --threshold: {:.2}.", th)
            } else {
                " Try passing --threshold 0.1 to lower the similarity cutoff.".to_string()
            };
            format!(
                "Query returned 0 memories. All candidates may have been filtered by the similarity threshold.{}",
                threshold_hint
            )
        } else {
            match best_score {
                Some(score) => format!(
                    "Pyramid search returned {} memories. Best match score: {:.4}",
                    count, score
                ),
                None => format!("Pyramid search returned {} memories", count),
            }
        };

        let data = json!({
            "count": count,
            "best_score": best_score,
            "message": message,
            "search_mode": "pyramid",
            "graph_traversal": false,
            "memories": memories_json
        });

        Ok(MemoryOperationResponse::success_with_data(&message, data))
    }

    /// Handle deep graph traversal using lightweight 1-hop refinement.
    /// BFS graph traversal from entry memories supporting direction and multi-hop depth.
    async fn handle_graph_traversal(
        &self,
        params: &QueryParams,
        filters: &Filters,
        graph_config: &TraversalConfig,
    ) -> OperationResult<MemoryOperationResponse> {
        use std::collections::{HashSet, VecDeque};

        info!("Graph traversal enabled, performing graph traversal (direction: {:?}, max_depth: {})",
            graph_config.direction, graph_config.max_depth);

        let entry_point_limit = graph_config.entry_point_limit.min(10);
        let entry_memories = if let Some(ref context_tags) = params.context {
            self.memory_manager
                .search_with_context(&params.query, context_tags, filters, entry_point_limit)
                .await?
        } else {
            self.memory_manager
                .search_with_override(&params.query, filters, entry_point_limit, params.similarity_threshold)
                .await?
        };

        if entry_memories.is_empty() {
            info!("No entry points found for graph traversal");
            return Ok(MemoryOperationResponse::success_with_data(
                "Graph traversal: No entry points found",
                json!({
                    "count": 0,
                    "message": "No matching memories found to use as graph traversal entry points",
                    "memories": []
                }),
            ));
        }

        let engine = GraphSearchEngine::new(graph_config.clone())
            .map_err(|e| OperationError::Runtime(format!("Invalid graph config: {}", e)))?;

        let mgr = &self.memory_manager;
        let use_outgoing = graph_config.direction == TraversalDirection::Outgoing
            || graph_config.direction == TraversalDirection::Both;
        let use_incoming = graph_config.direction == TraversalDirection::Incoming
            || graph_config.direction == TraversalDirection::Both;

        // BFS state
        let mut visited: HashSet<String> = HashSet::new();
        let mut results: Vec<GraphSearchResult> = Vec::new();
        let mut queue: VecDeque<(Memory, f32, usize, Vec<RelationHop>)> = VecDeque::new();

        for sm in &entry_memories {
            if visited.insert(sm.memory.id.clone()) {
                queue.push_back((sm.memory.clone(), sm.score, 0, vec![]));
            }
        }

        while let Some((memory, score, depth, path)) = queue.pop_front() {
            let boost = if depth == 0 { 0.0 } else {
                engine.calculate_rank_score(memory.clone(), score, 0.0, depth, vec![]).relation_boost
            };
            results.push(GraphSearchResult {
                memory: memory.clone(),
                entry_distance: depth,
                path_from_entry: path.clone(),
                relation_boost: boost,
                final_score: (score * 0.5) + (boost * 0.3) + ((1.0 / (depth as f32 + 1.0)) * 0.2),
                semantic_score: score,
            });

            if depth >= graph_config.max_depth {
                continue;
            }

            // Outgoing traversal
            if use_outgoing {
                for relation in &memory.metadata.relations {
                    if visited.contains(&relation.target) {
                        continue;
                    }
                    if let Ok(Some(target_mem)) = mgr.get(&relation.target).await {
                        if visited.insert(target_mem.id.clone()) {
                            let mut new_path = path.clone();
                            new_path.push(RelationHop {
                                from: memory.id.clone(),
                                relation: relation.relation.clone(),
                                to: target_mem.id.clone(),
                                strength: relation.strength,
                            });
                            queue.push_back((target_mem, score * 0.8, depth + 1, new_path));
                        }
                    }
                }
            }

            // Incoming traversal
            if use_incoming {
                if let Ok(incoming_mems) = mgr.find_incoming_relations(&memory.id, Some(20)).await {
                    for source_mem in incoming_mems {
                        if visited.contains(&source_mem.id) {
                            continue;
                        }
                        if !visited.insert(source_mem.id.clone()) {
                            continue;
                        }
                        let found_rel = source_mem.metadata.relations.iter()
                            .find(|r| r.target == memory.id);
                        let rel_type = found_rel.map(|r| r.relation.clone()).unwrap_or_default();
                        let rel_strength = found_rel.and_then(|r| r.strength);
                        let mut new_path = path.clone();
                        new_path.push(RelationHop {
                            from: source_mem.id.clone(),
                            relation: rel_type,
                            to: memory.id.clone(),
                            strength: rel_strength,
                        });
                        queue.push_back((source_mem, score * 0.8, depth + 1, new_path));
                    }
                }
            }
        }

        results.sort_by(|a, b| {
            b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        let memories_json: Vec<Value> = results
            .iter()
            .take(params.limit)
            .map(|gr| {
                let mut memory_json = memory_to_json(&gr.memory);
                memory_json["search_phase"] = json!(if gr.entry_distance == 0 {
                    "graph_entry"
                } else {
                    "graph_discovered"
                });
                if params.include_paths {
                    memory_json["graph_info"] = json!({
                        "entry_distance": gr.entry_distance,
                        "path_from_entry": gr.path_from_entry,
                        "relation_boost": gr.relation_boost,
                        "final_score": gr.final_score,
                        "semantic_score": gr.semantic_score,
                    });
                }
                memory_json
            })
            .collect();

        let entry_count = results.iter().filter(|r| r.entry_distance == 0).count();
        let discovered_count = results.len() - entry_count;
        let message = format!(
            "Graph search returned {} memories ({} entry, {} discovered, depth: {})",
            results.len(), entry_count, discovered_count, graph_config.max_depth
        );

        let data = json!({
            "count": memories_json.len(),
            "message": message,
            "graph_traversal": true,
            "memories": memories_json
        });

        Ok(MemoryOperationResponse::success_with_data(&message, data))
    }

    pub async fn list_memories(
        &self,
        req: ListRequest,
    ) -> OperationResult<MemoryOperationResponse> {
        let params: FilterParams = req.into();

        info!("Listing memories with filters");

        let mut filters = Filters::default();

        if let Some(user_id) = params.user_id {
            filters.user_id = Some(user_id);
        }
        if let Some(agent_id) = params.agent_id {
            filters.agent_id = Some(agent_id);
        }
        if let Some(created_after) = params.created_after {
            filters.created_after = Some(created_after);
        }
        if let Some(created_before) = params.created_before {
            filters.created_before = Some(created_before);
        }
        if let Some(relations) = params.relations {
            filters.relations = Some(
                relations
                    .into_iter()
                    .map(|r| crate::types::RelationFilter {
                        relation: r.relation,
                        target: r.target,
                    })
                    .collect(),
            );
        }

        let limit_arg = if params.limit == 0 { None } else { Some(params.limit) };
        match self.memory_manager.list(&filters, limit_arg).await {
            Ok(memories) => {
                let count = memories.len();
                info!("Listed {} memories", count);

                let memories_json: Vec<Value> = memories
                    .into_iter()
                    .map(|memory| memory_to_json(&memory))
                    .collect();

                let data = json!({
                    "count": count,
                    "memories": memories_json
                });

                Ok(MemoryOperationResponse::success_with_data(
                    "List completed successfully",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to list memories: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to list memories: {}",
                    e
                )))
            }
        }
    }

    pub async fn get_memory(
        &self,
        req: GetRequest,
    ) -> OperationResult<MemoryOperationResponse> {
        let memory_id = req.memory_id.clone();

        info!("Getting memory with ID: {}", memory_id);

        match self.memory_manager.get(&memory_id).await {
            Ok(Some(memory)) => {
                let mut memory_json = memory_to_json(&memory);

                // Enrich with reverse-direction links: which higher-layer memories
                // abstract FROM this one (zoom_out targets).
                if let Ok(dependents) = self
                    .memory_manager
                    .find_abstraction_dependents(&memory_id)
                    .await
                    && !dependents.is_empty()
                {
                    let ids: Vec<Value> = dependents
                        .iter()
                        .map(|m| Value::String(m.id.clone()))
                        .collect();
                    if let Some(meta) = memory_json.get_mut("metadata") {
                        meta.as_object_mut()
                            .map(|obj| obj.insert("abstracted_into".into(), Value::Array(ids)));
                    }
                }

                let data = json!({
                    "memory": memory_json
                });
                Ok(MemoryOperationResponse::success_with_data(
                    "Memory retrieved successfully",
                    data,
                ))
            }
            Ok(None) => {
                error!("Memory not found: {}", memory_id);
                Err(OperationError::MemoryNotFound(memory_id))
            }
            Err(e) => {
                error!("Failed to get memory: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to get memory: {}",
                    e
                )))
            }
        }
    }

    /// Navigate the abstraction hierarchy from a memory node.
    /// Allows LLM clients to traverse both towards abstraction (zoom_out)
    /// and towards detail (zoom_in).
    pub async fn navigate_memory(
        &self,
        req: NavigateRequest,
    ) -> OperationResult<MemoryOperationResponse> {
        let memory_id = req.memory_id.clone();
        let direction = req.direction.as_str();
        let levels = req.levels.min(5);

        info!(
            "Navigating memory {}: direction={}, levels={}",
            memory_id, direction, levels
        );

        match self
            .memory_manager
            .navigate_memory(&memory_id, direction, levels)
            .await
        {
            Ok(nav_result) => {
                let zoom_in_json: Vec<Value> =
                    nav_result.zoom_in.iter().map(memory_to_json).collect();
                let zoom_out_json: Vec<Value> =
                    nav_result.zoom_out.iter().map(memory_to_json).collect();

                let data = json!({
                    "source_memory_id": nav_result.source_memory_id,
                    "source_layer": nav_result.source_layer,
                    "zoom_in": zoom_in_json,
                    "zoom_out": zoom_out_json,
                });
                Ok(MemoryOperationResponse::success_with_data(
                    "Navigation completed",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to navigate memory: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to navigate memory: {}",
                    e
                )))
            }
        }
    }

    // ─── Document Session Operations ─────────────────────────────────────────

    pub fn begin_store_document(
        &self,
        req: BeginStoreDocumentRequest,
    ) -> OperationResult<MemoryOperationResponse> {
        let session_manager = self.get_session_manager()?;

        let mut params: BeginStoreDocumentParams = req.into();
        params.user_id = params.user_id.or(self.default_user_id.clone());
        params.agent_id = params.agent_id.or(self.default_agent_id.clone());

        info!(
            "Beginning document storage session for file: {}",
            params.file_name
        );

        let metadata = DocumentMetadata {
            file_name: params.file_name,
            file_type: params.file_type,
            total_size: params.total_size,
            md5sum: params.md5sum,
            user_id: params.user_id,
            agent_id: params.agent_id,
            memory_type: params.memory_type,
            topics: params.topics,
            context: params.context,
            custom_metadata: params
                .metadata
                .map(|m| serde_json::Value::Object(m.into_iter().collect())),
        };

        match session_manager.begin_session(metadata) {
            Ok(response) => {
                info!("Created document session: {}", response.session_id);
                let data =
                    serde_json::to_value(&response).map_err(OperationError::Serialization)?;
                Ok(MemoryOperationResponse::success_with_data(
                    "Document session created",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to create document session: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to create document session: {}",
                    e
                )))
            }
        }
    }

    pub fn store_document_part(
        &self,
        req: StoreDocumentPartRequest,
    ) -> OperationResult<MemoryOperationResponse> {
        let session_manager = self.get_session_manager()?;

        let params: StoreDocumentPartParams = req.into();

        info!(
            "Storing document part {} for session {}",
            params.part_index, params.session_id
        );

        match session_manager.store_part(&params.session_id, params.part_index, &params.content) {
            Ok(()) => {
                // Get session info for progress reporting
                let session = session_manager.get_session(&params.session_id);
                let (received, expected) = session
                    .map(|s| (s.received_parts, s.expected_parts))
                    .unwrap_or((params.part_index + 1, 0));

                let remaining = expected.saturating_sub(received);
                let progress_msg = if expected > 0 {
                    format!(
                        "Part {} stored for session {} ({}/{}, {} remaining)",
                        params.part_index, params.session_id, received, expected, remaining
                    )
                } else {
                    format!(
                        "Part {} stored for session {}",
                        params.part_index, params.session_id
                    )
                };

                // Include progress data in response
                let data = json!({
                    "session_id": params.session_id,
                    "part_index": params.part_index,
                    "received_parts": received,
                    "expected_parts": expected,
                    "remaining_parts": remaining
                });

                Ok(MemoryOperationResponse::success_with_data(
                    progress_msg,
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to store document part: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to store document part: {}",
                    e
                )))
            }
        }
    }

    pub async fn upload_document(
        &self,
        req: UploadDocumentRequest,
    ) -> OperationResult<MemoryOperationResponse> {
        let session_manager = self.get_session_manager()?;

        let mut params: UploadDocumentParams = req.into();
        params.user_id = params.user_id.or(self.default_user_id.clone());
        params.agent_id = params.agent_id.or(self.default_agent_id.clone());

        info!(
            "Auto-chunk upload: file={}, process_immediately={}",
            params.file_path, params.process_immediately
        );

        let file_path = std::path::Path::new(&params.file_path);

        if !file_path.exists() {
            return Err(OperationError::InvalidInput(format!(
                "File not found: {}",
                params.file_path
            )));
        }

        // Read file content (for large files, could be streamed from disk in background)
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| OperationError::Runtime(format!("Failed to read file: {}", e)))?;

        let file_name = params.file_name.unwrap_or_else(|| {
            file_path
                .file_name()
                .unwrap_or(std::ffi::OsStr::new("unknown"))
                .to_string_lossy()
                .to_string()
        });

        let total_size = content.len();
        let chunk_size = params
            .chunk_size
            .unwrap_or(self.memory_manager.config().document_chunk_size);

        // Calculate expected chunks (char-based) BEFORE creating session
        let total_chars = content.chars().count();
        let expected_chunks = total_chars.div_ceil(chunk_size).max(1);

        // Create session
        use crate::document_session::DocumentMetadata;

        // Store file_path in custom_metadata for resume support
        let custom_metadata = Some(json!({
            "file_path": params.file_path
        }));

        let metadata = DocumentMetadata {
            file_name: file_name.clone(),
            file_type: Some(params.mime_type.unwrap_or_else(|| "text/plain".to_string())),
            total_size,
            md5sum: Some(format!("{:x}", md5::compute(&content))),
            user_id: params.user_id,
            agent_id: params.agent_id,
            memory_type: params.memory_type.unwrap_or_else(|| "semantic".to_string()),
            topics: params.topics,
            context: params.context,
            custom_metadata,
        };

        let session_response = session_manager
            .begin_session(metadata)
            .map_err(|e| OperationError::Runtime(format!("Failed to create session: {}", e)))?;

        let session_id = session_response.session_id;

        // Update session with correct char-based chunk count
        session_manager
            .update_expected_parts(&session_id, expected_chunks)
            .map_err(|e| {
                OperationError::Runtime(format!("Failed to update expected parts: {}", e))
            })?;

        info!(
            "Created session {} for file {} ({} bytes, chunk size: {} bytes, {} chunks)",
            session_id, file_name, total_size, chunk_size, expected_chunks
        );

        let memory_manager = self.memory_manager.clone();

        tokio::spawn(super::document_pipeline::upload_chunked_task(
            session_id.clone(),
            content,
            chunk_size,
            params.process_immediately,
            memory_manager,
            session_manager.clone(),
        ));

        // Return immediately (background task handles upload + processing)
        Ok(MemoryOperationResponse::success_with_data(
            format!(
                "File upload started: {} (session: {})",
                file_name, session_id
            ),
            json!({
                "session_id": session_id,
                "file_name": file_name,
                "total_size": total_size,
                "chunk_size": chunk_size,
                "estimated_chunks": expected_chunks,
                "process_immediately": params.process_immediately,
                "status": "uploading"
            }),
        ))
    }

    pub async fn process_document(
        &self,
        req: ProcessDocumentRequest,
    ) -> OperationResult<MemoryOperationResponse> {
        let session_manager = self.session_manager.clone().ok_or_else(|| {
            OperationError::Runtime("Document session manager not configured".to_string())
        })?;

        let params: ProcessDocumentParams = req.into();

        info!(
            "Processing document for session {} (partial_closure={})",
            params.session_id, params.partial_closure
        );

        let session = session_manager.get_session(&params.session_id)?;

        // State check: prevent double processing
        // Exception: allow processing if session was left in "Processing" state from a crash
        if session.status == SessionStatus::Processing {
            info!(
                "Session {} was left in Processing state (possible crash), resetting and resuming",
                params.session_id
            );
            // Reset status to allow resumption
            session_manager.update_status(
                &params.session_id,
                SessionStatus::Uploading,
                Some("Resuming after crash"),
            )?;
        }

        let parts = session_manager.get_parts(&params.session_id)?;

        // Handle partial closure - allow finalizing with fewer parts than expected
        if parts.len() != session.expected_parts {
            if params.partial_closure {
                info!(
                    "Partial closure requested for session {}: processing {}/{} parts",
                    params.session_id,
                    parts.len(),
                    session.expected_parts
                );
            } else {
                return Err(OperationError::InvalidInput(format!(
                    "Cannot finalize: expected {} parts but received {}. \
                     Before calling finalize, send each chunk as a separate 'store_document_part' request with: \
                     - session_id: '{}' \
                     - part_index: 0, 1, 2, ... (sequential) \
                     - content: the text chunk. \
                     Once all {} parts are stored, call finalize to begin processing. \
                     Or set partial_closure=true to finalize with the current parts.",
                    session.expected_parts,
                    parts.len(),
                    params.session_id,
                    session.expected_parts
                )));
            }
        }

        session_manager.update_status(&params.session_id, SessionStatus::Processing, None)?;

        let full_content: String = parts.into_iter().map(|(_, content)| content).collect();

        // Spawn background task
        let session_id = params.session_id.clone();
        let memory_manager = self.memory_manager.clone();
        let session_manager_clone = session_manager.clone();

        tokio::spawn(async move {
            if let Err(e) = super::document_pipeline::process_document_task(
                session_id.clone(),
                full_content,
                session,
                memory_manager,
                session_manager_clone.clone(),
            )
            .await
            {
                error!(
                    "Document processing background task failed for session {}: {}",
                    session_id, e
                );
                let _ = session_manager_clone.update_status(
                    &session_id,
                    SessionStatus::Failed,
                    Some(&e.to_string()),
                );
            }
        });

        Ok(MemoryOperationResponse::success(
            "Document processing started in background",
        ))
    }

    pub fn status_process_document(
        &self,
        req: StatusProcessDocumentRequest,
    ) -> OperationResult<MemoryOperationResponse> {
        let session_manager = self.get_session_manager()?;

        let params: StatusProcessDocumentParams = req.into();

        info!("Getting status for session: {}", params.session_id);

        match session_manager.get_status(&params.session_id) {
            Ok(status) => {
                let data = serde_json::to_value(&status).map_err(OperationError::Serialization)?;
                Ok(MemoryOperationResponse::success_with_data(
                    "Session status retrieved",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to get session status: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to get session status: {}",
                    e
                )))
            }
        }
    }

    pub fn list_document_sessions(
        &self,
        _req: ListDocumentSessionsRequest,
    ) -> OperationResult<MemoryOperationResponse> {
        let session_manager = self.get_session_manager()?;

        match session_manager.list_all_sessions() {
            Ok(sessions) => Ok(MemoryOperationResponse::success_with_data(
                "Retrieved document sessions",
                json!({
                    "sessions": sessions
                }),
            )),
            Err(e) => {
                error!("Failed to list sessions: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to list sessions: {}",
                    e
                )))
            }
        }
    }

    pub fn cancel_process_document(
        &self,
        req: CancelProcessDocumentRequest,
    ) -> OperationResult<MemoryOperationResponse> {
        let session_manager = self.get_session_manager()?;

        let params: CancelProcessDocumentParams = req.into();

        info!("Cancelling session: {}", params.session_id);

        match session_manager.cancel_session(&params.session_id) {
            Ok(()) => Ok(MemoryOperationResponse::success(format!(
                "Session {} cancelled",
                params.session_id
            ))),
            Err(e) => {
                error!("Failed to cancel session: {}", e);
                Err(OperationError::Runtime(format!(
                    "Failed to cancel session: {}",
                    e
                )))
            }
        }
    }

    fn get_session_manager(
        &self,
    ) -> OperationResult<&std::sync::Arc<crate::document_session::DocumentSessionManager>> {
        self.session_manager
            .as_ref()
            .ok_or_else(|| OperationError::Runtime("Document session manager not configured".to_string()))
    }
}
