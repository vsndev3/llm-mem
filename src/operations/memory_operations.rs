use serde_json::{Value, json};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::{
    error::MemoryError,
    llm::LlmPriority,
    memory::{MemoryManager, StoreOptions},
    search::{GraphSearchEngine, TraversalConfig, TraversalDirection, GraphSearchResult, RelationHop},
    types::{Filters, LayerInfo, Memory, MemoryMetadata, RelationMeta, ScoredMemory, reverse_relation},
    layer::abstraction_pipeline::derive_event_range_from_sources,
};

use super::params::*;
use super::requests::{
    AddMemoryRequest, BeginStoreDocumentRequest, CancelProcessDocumentRequest,
    CreateAbstractionRequest, ForceLinkRequest,
    GetContextResumeRequest, GetRequest, GetTimelineGraphRequest, GetTimelineRequest, IngestRequest, ListDocumentSessionsRequest,
    ListRequest, MemoryOperationResponse, NavigateRequest, ProcessDocumentRequest, QueryRequest,
    RemoveRelationRequest, SearchMemoryRequest, StoreDocumentPartRequest,
    StoreMemoriesRequest, StoreRequest,
    StatusProcessDocumentRequest, UpdateRequest,
    UploadDocumentRequest,
};
use super::serialization::memory_to_json;
use super::context_resume::{ContextResumeResponse, ContextResumeService, ResumeFilters};
use super::timeline::{TimelineGraphResponse, TimelineResponse, TimelineService};

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
    ) -> crate::error::Result<MemoryOperationResponse> {
        let mut params: StoreParams = req.into();
        params.user_id = params.user_id.or(self.default_user_id.clone());
        params.agent_id = params.agent_id.or(self.default_agent_id.clone());

        if params.content.trim().is_empty() {
            return Err(MemoryError::InvalidInput(
                "Content cannot be empty".into(),
            ));
        }

        info!("Storing memory for user: {:?}", params.user_id);

        let has_context = params.context.is_some();
        let has_relations = params.relations.is_some();

        let metadata = super::helpers::build_metadata(
            &params.memory_type,
            params.user_id.clone(),
            params.agent_id.clone(),
            params.topics,
            params.context,
            params.relations,
            params.metadata,
        )?;


        let mut relation_warnings: Vec<String> = Vec::new();
        for rel in &metadata.relations {
            if uuid::Uuid::parse_str(&rel.target).is_ok() {
                match self.memory_manager.get(&rel.target).await {
                    Ok(Some(_)) => {}
                    _ => {
                        relation_warnings.push(format!(
                            "Target memory '{}' not found (relation: '{}')",
                            rel.target, rel.relation
                        ));
                    }
                }
            }
        }

        let auto_link = params.auto_link;
        let event_at = params.event_at;
        let source = params.source;
        let store_options = StoreOptions {
            llm_priority: LlmPriority::Interactive,
            auto_link,
            event_at,
            source,
            ..StoreOptions::default()
        };

        let quality_warnings = self.memory_manager
            .check_store_quality(&params.content, &metadata)
            .await
            .unwrap_or_default();

        // Block store if quality issues found and not forced
        if !params.force {
            let mut issues: Vec<String> = Vec::new();
            if !quality_warnings.near_duplicates.is_empty() {
                let dup_list: Vec<String> = quality_warnings.near_duplicates.iter()
                    .map(|(id, score)| format!("{} ({:.0}% similar)", id, score * 100.0))
                    .collect();
                issues.push(format!(
                    "Near-duplicate content detected. Existing memories: {}. Either update the existing memories instead, or set force:true to store anyway.",
                    dup_list.join(", ")
                ));
            }
            if !quality_warnings.contradictions.is_empty() {
                issues.push(format!(
                    "Factual contradictions detected: {}. Review and resolve these conflicts, or set force:true to store anyway.",
                    quality_warnings.contradictions.join("; ")
                ));
            }
            if !issues.is_empty() {
                return Err(MemoryError::InvalidInput(issues.join(" ")));
            }
        }

        match self.memory_manager.store_with_options(params.content, metadata, store_options).await {
            Ok(memory_id) => {
                info!("Memory stored successfully with ID: {}", memory_id);

                let mut warnings = relation_warnings;

                if let Ok(Some(stored)) = self.memory_manager.get(&memory_id).await {
                    if has_context && stored.context_embeddings.is_none() {
                        warnings.push("Context embeddings were not created for provided context".to_string());
                    }
                    if has_relations && stored.relation_embeddings.is_none() {
                        warnings.push("Relation embeddings were not created for provided relations".to_string());
                    }
                }

                let mut data = json!({
                    "memory_id": memory_id,
                    "user_id": params.user_id,
                    "agent_id": params.agent_id
                });
                if !warnings.is_empty() {
                    data["warnings"] = json!(warnings);
                }
                if !quality_warnings.near_duplicates.is_empty() {
                    data["near_duplicates"] = json!(quality_warnings.near_duplicates.iter().map(|(id, score)| {
                        json!({"memory_id": id, "similarity": format!("{:.2}", score * 100.0)})
                    }).collect::<Vec<_>>());
                    data["hint"] = json!("Your content is very similar to existing memories. Consider updating the existing ones instead of storing a near-duplicate.");
                }
                if !quality_warnings.contradictions.is_empty() {
                    data["contradictions"] = json!(quality_warnings.contradictions);
                    data["hint"] = json!("Your content may contradict existing memories. Review and resolve before proceeding.");
                }
                Ok(MemoryOperationResponse::success_with_data(
                    "Memory stored successfully",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to store memory: {}", e);
                Err(MemoryError::Internal(format!(
                    "Failed to store memory: {}",
                    e
                )))
            }
        }
    }

    /// Simplified search with sensible defaults (Balanced pyramid mode, keyword_split_ratio: 0.2).
    pub async fn search_memory(
        &self,
        req: SearchMemoryRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
        let query_req: QueryRequest = req.into();
        self.query_memory(query_req).await
    }

    /// Store multiple content memories in a single call.
    pub async fn store_memories(
        &self,
        req: StoreMemoriesRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
        if req.items.is_empty() {
            return Err(MemoryError::InvalidInput(
                "Items array cannot be empty".into(),
            ));
        }

        for (i, item) in req.items.iter().enumerate() {
            if item.content.trim().is_empty() {
                return Err(MemoryError::InvalidInput(format!(
                    "Item {} has empty content",
                    i
                )));
            }
        }

        let mut results = Vec::new();
        let mut failed_count: usize = 0;
        let mut succeeded_count: usize = 0;
        for item in &req.items {
            let store_req = StoreRequest {
                content: item.content.clone(),
                user_id: self.default_user_id.clone(),
                agent_id: self.default_agent_id.clone(),
                memory_type: item.memory_type.clone().unwrap_or_else(|| "conversational".to_string()),
                topics: item.topics.clone(),
                context: item.context.clone(),
                relations: item.relations.clone(),
                metadata: item.metadata.clone(),
                bank: req.bank.clone(),
                auto_link: None,
                event_at: item.event_at.clone(),
                source: item.source.clone(),
                force: req.force,
            };
            match self.store_memory(store_req).await {
                Ok(response) => {
                    succeeded_count += 1;
                    if let Some(data) = response.data {
                        results.push(json!({ "status": "ok", "data": data }));
                    } else {
                        results.push(json!({ "status": "ok", "message": response.message }));
                    }
                }
                Err(e) => {
                    failed_count += 1;
                    results.push(json!({ "status": "error", "error": format!("{}", e) }));
                }
            }
        }
        let data = json!({
            "results": results,
            "total": req.items.len(),
            "succeeded_count": succeeded_count,
            "failed_count": failed_count
        });
        if succeeded_count >= 5 {
            let mm = self.memory_manager.clone();
            tokio::spawn(async move {
                match crate::consistency::check_consistency(mm.vector_store()).await {
                    Ok(report) => {
                        info!(
                            "Post-bulk consistency check: {} memories, {} errors, {} warnings, {} infos",
                            report.total_memories, report.errors, report.warnings, report.infos
                        );
                        if report.errors > 0 || report.warnings > 0 {
                            warn!(
                                "Consistency issues detected after bulk store: {} errors, {} warnings",
                                report.errors, report.warnings
                            );
                        }
                    }
                    Err(e) => {
                        error!("Post-bulk consistency check failed: {}", e);
                    }
                }
            });
        }
        if failed_count > 0 {
            Ok(MemoryOperationResponse {
                success: false,
                message: "Partial batch failure".to_string(),
                data: Some(data),
                error: Some(format!("{} of {} items failed", failed_count, req.items.len())),
            })
        } else {
            Ok(MemoryOperationResponse::success_with_data("Batch store completed", data))
        }
    }

    pub async fn add_memory(
        &self,
        req: AddMemoryRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
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
        if let Some(ref source_id) = params.source_memory_id {
            if source_id.trim().is_empty() {
                return Err(MemoryError::InvalidInput(
                    "source_memory_id cannot be empty".into(),
                ));
            }
            match self.memory_manager.get(source_id).await {
                Ok(Some(_)) => {}
                _ => {
                    return Err(MemoryError::NotFound { id: format!(
                        "Source memory '{}' not found",
                        source_id
                    ) } );
                }
            }
            metadata.relations.push(crate::types::Relation {
                source: "SELF".to_string(),
                relation: "derived_from".to_string(),
                target: source_id.clone(),
                strength: None,
            });
        }

        match self
            .memory_manager
            .add_memory_with_event_at(&params.messages, metadata, params.event_at)
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
                Err(MemoryError::Internal(format!(
                    "Failed to add memory: {}",
                    e
                )))
            }
        }
    }

    pub async fn update_memory(
        &self,
        req: UpdateRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
        let memory_id = req.memory_id.clone();

        if memory_id.trim().is_empty() {
            return Err(MemoryError::InvalidInput(
                "Memory ID cannot be empty".into(),
            ));
        }

        if let Some(ref content) = req.content
            && content.trim().is_empty()
        {
            return Err(MemoryError::InvalidInput(
                "Content cannot be empty — use the content field to update, or omit it to only update relations".into(),
            ));
        }

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
                Err(MemoryError::Internal(format!(
                    "Failed to update memory: {}",
                    e
                )))
            }
        }
    }

    pub async fn query_memory(
        &self,
        req: QueryRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
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
        if let Some(ref event_after) = params.event_after {
            filters.event_after = Some(*event_after);
        }
        if let Some(ref event_before) = params.event_before {
            filters.event_before = Some(*event_before);
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
                .map_err(|e| MemoryError::Internal(format!("Pyramid search failed: {}", e)))?;
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
                .map_err(|e| MemoryError::Internal(format!("Pyramid search failed: {}", e)))?;
            keyword_results = None;
        }

        // Merge keyword results using Reciprocal Rank Fusion (RRF).
        let all_results: Vec<crate::search::PyramidResult> = if let Some(kw_results) = keyword_results {
            if !kw_results.is_empty() {
                let k: f32 = 60.0;
                let mut merged: std::collections::HashMap<String, (usize, f32, f32, ScoredMemory)> = std::collections::HashMap::new();

                for (i, r) in pyramid_results.iter().enumerate() {
                    merged.entry(r.memory.memory.id.clone()).or_insert_with(|| {
                        (i, 1.0 / (k + i as f32 + 1.0), r.memory.score, r.memory.clone())
                    });
                }
                for (j, kw) in kw_results.iter().enumerate() {
                    let rrf = 1.0 / (k + j as f32 + 1.0);
                    merged.entry(kw.memory.id.clone())
                        .and_modify(|(_, score, _, _)| *score += rrf)
                        .or_insert_with(|| (j, rrf, 0.0, ScoredMemory { memory: kw.memory.clone(), score: rrf }));
                }

                let mut ranked: Vec<_> = merged.into_values().collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let mut results: Vec<crate::search::PyramidResult> = ranked.into_iter().map(|(_, rrf, vec_score, scored)| {
                    let layer = scored.memory.metadata.layer.level;
                    let layer_name = scored.memory.metadata.layer.name_or_default();
                    let score = if vec_score > 0.0 { rrf * 0.7 + vec_score * 0.3 } else { rrf };
                    crate::search::PyramidResult {
                        memory: ScoredMemory { memory: scored.memory, score },
                        layer,
                        layer_name,
                        search_phase: "rrf_merged".to_string(),
                        graph_path: None,
                        source: "hybrid".to_string(),
                    }
                }).collect();
                results.truncate(params.limit);
                results
            } else {
                pyramid_results
            }
        } else {
            pyramid_results
        };
        let count = all_results.len();
        let best_score = all_results.first().map(|r| r.memory.score);

        // Track access for frequency-boosted ranking (top 20 results)
        for r in all_results.iter().take(20) {
            self.memory_manager.increment_access(&r.memory.memory.id);
        }

        let memories_json: Vec<Value> = all_results
            .clone()
            .into_iter()
            .map(|r| {
                let mut memory_json = memory_to_json(&r.memory.memory);
                memory_json["score"] = json!(r.memory.score);
                memory_json["layer"] = json!(r.layer);
                memory_json["layer_name"] = json!(r.layer_name);
                memory_json["search_phase"] = json!(r.search_phase);
                memory_json["source"] = json!(r.source);
                memory_json["access_count"] = json!(r.memory.memory.metadata.access_count);
                if let Some(la) = r.memory.memory.metadata.last_accessed {
                    memory_json["last_accessed"] = json!(la.to_rfc3339());
                }
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

        // Build a context hint summarizing what was found
        let layer_summary: std::collections::HashMap<String, usize> =
            all_results.iter().fold(Default::default(), |mut acc, r| {
                *acc.entry(r.layer_name.clone()).or_default() += 1;
                acc
            });
        let topics: Vec<&str> = all_results.iter()
            .flat_map(|r| r.memory.memory.metadata.topics.iter().map(|s| s.as_str()))
            .filter(|t| !t.is_empty())
            .take(5)
            .collect();

        let data = json!({
            "count": count,
            "best_score": best_score,
            "message": message,
            "search_mode": if split_ratio > 0.0 { "rrf_hybrid" } else { "pyramid" },
            "graph_traversal": false,
            "context_hint": {
                "layer_distribution": layer_summary,
                "top_topics": topics,
                "query": params.query,
            },
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
    ) -> crate::error::Result<MemoryOperationResponse> {
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
            .map_err(|e| MemoryError::Internal(format!("Invalid graph config: {}", e)))?;

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
                    if let Ok(Some(target_mem)) = mgr.get(&relation.target).await
                        && visited.insert(target_mem.id.clone()) {
                            let mut new_path = path.clone();
                            new_path.push(RelationHop {
                                from: memory.id.clone(),
                                relation: relation.relation.clone(),
                                to: target_mem.id.clone(),
                                strength: relation.strength,
                            });
                            queue.push_back((target_mem, score * graph_config.score_decay, depth + 1, new_path));
                        }
                }
            }

            // Incoming traversal
            if use_incoming
                && let Ok(incoming_mems) = mgr.find_incoming_relations(&memory.id, Some(20)).await {
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
                        queue.push_back((source_mem, score * graph_config.score_decay, depth + 1, new_path));
                    }
                }
        }

        results.retain(|r| r.final_score >= graph_config.min_discovery_score);

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
    ) -> crate::error::Result<MemoryOperationResponse> {
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
        if let Some(event_after) = params.event_after {
            filters.event_after = Some(event_after);
        }
        if let Some(event_before) = params.event_before {
            filters.event_before = Some(event_before);
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
                Err(MemoryError::Internal(format!(
                    "Failed to list memories: {}",
                    e
                )))
            }
        }
    }

    pub async fn get_memory(
        &self,
        req: GetRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
        let memory_id = req.memory_id.clone();

        info!("Getting memory with ID: {}", memory_id);

        match self.memory_manager.get(&memory_id).await {
            Ok(Some(memory)) => {
                // Track access for frequency-boosted ranking
                self.memory_manager.increment_access(&memory_id);

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
                Err(MemoryError::NotFound { id: memory_id })
            }
            Err(e) => {
                error!("Failed to get memory: {}", e);
                Err(MemoryError::Internal(format!(
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
    ) -> crate::error::Result<MemoryOperationResponse> {
        let memory_id = req.memory_id.clone();
        let direction = req.direction.as_str();
        let levels = req.levels.min(5);

        if memory_id.trim().is_empty() {
            return Err(MemoryError::InvalidInput(
                "Memory ID cannot be empty".into(),
            ));
        }

        match self.memory_manager.get(&memory_id).await {
            Ok(Some(_)) => {}
            Ok(None) => {
                return Err(MemoryError::NotFound { id: memory_id });
            }
            Err(e) => {
                return Err(MemoryError::Internal(format!("{}", e)));
            }
        }

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
                Err(MemoryError::Internal(format!(
                    "Failed to navigate memory: {}",
                    e
                )))
            }
        }
    }

    // ─── Timeline / chronological graph ────────────────────────────────────

    /// Return a bucketed chronological list of memories (see `get_timeline` MCP tool).
    pub async fn get_timeline(
        &self,
        req: GetTimelineRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
        let svc = TimelineService::new(self.memory_manager.clone());
        let response: TimelineResponse = svc.get_timeline(req).await?;
        let data = serde_json::to_value(&response).map_err(MemoryError::Serialization)?;
        Ok(MemoryOperationResponse::success_with_data(
            "Timeline retrieved successfully",
            data,
        ))
    }

    /// Return nodes + edges forming a chronological graph (see `get_timeline_graph` MCP tool).
    pub async fn get_timeline_graph(
        &self,
        req: GetTimelineGraphRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
        let svc = TimelineService::new(self.memory_manager.clone());
        let response: TimelineGraphResponse = svc.get_timeline_graph(req).await?;
        let data = serde_json::to_value(&response).map_err(MemoryError::Serialization)?;
        Ok(MemoryOperationResponse::success_with_data(
            "Timeline graph retrieved successfully",
            data,
        ))
    }

    /// Progressive context resume — exponential decay curve over memory layers.
    pub async fn context_resume(
        &self,
        req: GetContextResumeRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
        let lookback_secs = super::context_resume::parse_lookback(
            req.lookback.as_deref().unwrap_or("30d"),
        )
        .map_err(MemoryError::InvalidInput)?;

        let svc = ContextResumeService::new(self.memory_manager.clone());
        let response: ContextResumeResponse = svc
            .get_context_resume(
                req.end.as_deref(),
                lookback_secs,
                req.decay_factor.unwrap_or(2.0),
                req.segments.unwrap_or(5),
                req.max_per_segment,
                ResumeFilters {
                    user_id: req.user_id,
                    agent_id: req.agent_id,
                    topics: req.topics,
                },
            )
            .await?;

        let data = serde_json::to_value(&response).map_err(MemoryError::Serialization)?;
        Ok(MemoryOperationResponse::success_with_data(
            "Context resume retrieved successfully",
            data,
        ))
    }

    // ─── User control methods ────────────────────────────────────────────────

    pub async fn create_abstraction(
        &self,
        req: CreateAbstractionRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
        let params: CreateAbstractionParams = req.into();
        let user_id = params.user_id.or(self.default_user_id.clone());
        let agent_id = params.agent_id.or(self.default_agent_id.clone());

        if params.content.trim().is_empty() {
            return Err(MemoryError::InvalidInput(
                "Content cannot be empty".into(),
            ));
        }

        if params.source_ids.is_empty() {
            return Err(MemoryError::InvalidInput(
                "At least one source memory ID is required".into(),
            ));
        }

        if params.target_layer < 1 {
            return Err(MemoryError::InvalidInput(format!(
                "Target layer must be >= 1, got {}",
                params.target_layer
            )));
        }

        let mut seen_ids = std::collections::HashSet::new();
        for src_id in &params.source_ids {
            if !seen_ids.insert(src_id) {
                return Err(MemoryError::InvalidInput(format!(
                    "Duplicate source ID: '{}'",
                    src_id
                )));
            }
        }

        let mut source_uuids = Vec::with_capacity(params.source_ids.len());
        for src_id in &params.source_ids {
            let uuid = Uuid::parse_str(src_id).map_err(|_| {
                MemoryError::InvalidInput(format!(
                    "Source ID '{}' is not a valid UUID",
                    src_id
                ))
            })?;
            source_uuids.push(uuid);
        }

        let mut source_memories = Vec::with_capacity(params.source_ids.len());
        let mut max_source_layer = 0;
        for src_id in params.source_ids.iter() {
            match self.memory_manager.get(src_id).await {
                Ok(Some(m)) => {
                    if !m.metadata.state.is_active() {
                        return Err(MemoryError::InvalidInput(format!(
                            "Source memory '{}' is in '{}' state and cannot be used for abstraction",
                            src_id, m.metadata.state.as_str()
                        )));
                    }
                    max_source_layer = max_source_layer.max(m.metadata.layer.level);
                    source_memories.push(m);
                }
                _ => {
                    return Err(MemoryError::NotFound { id: format!(
                        "Source memory '{}' not found",
                        src_id
                    ) } );
                }
            }
        }

        if params.target_layer <= max_source_layer {
            return Err(MemoryError::InvalidInput(format!(
                "Target layer ({}) must be higher than all source layers (max source layer: {})",
                params.target_layer, max_source_layer
            )));
        }

        let layer_info = LayerInfo::custom(params.target_layer, format!("manual_layer_{}", params.target_layer));
        let relation_type = params
            .relation_type
            .as_deref()
            .unwrap_or(match params.target_layer {
                1 => "summary_of",
                2 => "synthesizes",
                _ => "abstracts_to_concept",
            });

        let mut metadata = MemoryMetadata::new()
            .with_layer(layer_info)
            .with_abstraction_sources(source_uuids.clone());
        metadata.user_id = user_id;
        metadata.agent_id = agent_id;
        metadata.abstraction_confidence = Some(1.0);

        let mut memory = crate::types::Memory::with_content(
            params.content,
            Vec::new(),
            metadata,
        );

        derive_event_range_from_sources(&source_memories, &mut memory);

        let meta = RelationMeta::new("manual_abstraction").with_confidence(1.0);
        memory.add_relation(relation_type.to_string(), source_uuids.clone(), Some(1.0), meta);

        for src_id in &params.source_ids {
            memory.metadata.relations.push(crate::types::Relation {
                source: memory.id.clone(),
                relation: relation_type.to_string(),
                target: src_id.clone(),
                strength: Some(1.0),
            });
        }

        let reverse_type = reverse_relation(relation_type);
        if let Some(rev) = reverse_type {
            let reverse_meta = RelationMeta::new("manual_abstraction:auto_reverse").with_confidence(1.0);
            let abstraction_uuid = Uuid::parse_str(&memory.id).ok();
            for mut src in source_memories {
                if let Some(abs_uuid) = abstraction_uuid {
                    src.append_relation(rev.to_string(), abs_uuid, Some(1.0), reverse_meta.clone());
                    src.metadata.relations.push(crate::types::Relation {
                        source: src.id.clone(),
                        relation: rev.to_string(),
                        target: memory.id.clone(),
                        strength: Some(1.0),
                    });
                    if let Err(e) = self.memory_manager.update_memory(&src).await {
                        warn!("Failed to add reverse relation to source '{}': {}", src.id, e);
                    }
                }
            }
        }

        match self.memory_manager.store_memory(memory).await {
            Ok(memory_id) => {
                info!("Manual abstraction created: {}", memory_id);
                let mut data = json!({
                    "memory_id": memory_id,
                    "target_layer": params.target_layer,
                    "relation_type": relation_type,
                    "source_count": params.source_ids.len(),
                });
                if let Some(rev) = reverse_type {
                    data["reverse_relation"] = json!(rev);
                    data["reverse_created"] = json!(true);
                }
                Ok(MemoryOperationResponse::success_with_data(
                    "Abstraction created successfully",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to create abstraction: {}", e);
                Err(MemoryError::Internal(format!(
                    "Failed to create abstraction: {}", e,
                )))
            }
        }
    }

    pub async fn force_link(
        &self,
        req: ForceLinkRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
        let params: ForceLinkParams = req.into();

        let relation_type = params.relation.trim();
        if relation_type.is_empty() {
            return Err(MemoryError::InvalidInput(
                "Relation type cannot be empty".into(),
            ));
        }

        if params.source_id == params.target_id {
            return Err(MemoryError::InvalidInput(
                "Cannot link a memory to itself".into(),
            ));
        }

        let source_uuid = Uuid::parse_str(&params.source_id).map_err(|_| {
            MemoryError::InvalidInput(format!(
                "Source ID '{}' is not a valid UUID",
                params.source_id
            ))
        })?;

        let target_uuid = Uuid::parse_str(&params.target_id).map_err(|_| {
            MemoryError::InvalidInput(format!(
                "Target ID '{}' is not a valid UUID",
                params.target_id
            ))
        })?;

        let strength = params
            .strength
            .map(|s| s.clamp(0.0, 1.0))
            .unwrap_or(1.0);

        let mut source = match self.memory_manager.get(&params.source_id).await {
            Ok(Some(m)) => m,
            _ => {
                return Err(MemoryError::NotFound { id: format!(
                    "Source memory '{}' not found",
                    params.source_id
                ) } );
            }
        };

        let mut target = match self.memory_manager.get(&params.target_id).await {
            Ok(Some(m)) => m,
            _ => {
                return Err(MemoryError::NotFound { id: format!(
                    "Target memory '{}' not found",
                    params.target_id
                ) } );
            }
        };

        if !source.metadata.state.is_active() {
            return Err(MemoryError::InvalidInput(format!(
                "Source memory is in '{}' state and cannot be linked (must be Active or Degraded)",
                source.metadata.state.as_str()
            )));
        }

        if !target.metadata.state.is_active() {
            return Err(MemoryError::InvalidInput(format!(
                "Target memory is in '{}' state and cannot be linked (must be Active or Degraded)",
                target.metadata.state.as_str()
            )));
        }

        if source.has_relation_to(relation_type, &target_uuid) {
            return Err(MemoryError::InvalidInput(format!(
                "Relation '{}' from '{}' to '{}' already exists",
                relation_type, params.source_id, params.target_id
            )));
        }

        let source_layer = source.metadata.layer.level;
        let target_layer = target.metadata.layer.level;

        if matches!(relation_type, "summary_of" | "part_of" | "synthesizes")
            && source_layer >= target_layer
        {
            return Err(MemoryError::InvalidInput(format!(
                "Hierarchical relation '{}' requires source (L{}) to be at a higher layer than target (L{}). \
                 Use a non-hierarchical relation type like 'references' or 'similar_to' instead.",
                relation_type, source_layer, target_layer
            )));
        }

        if matches!(relation_type, "references" | "similar_to" | "extends" | "depends_on")
            && source_layer > 0
            && target_layer > 0
            && target_layer > source_layer
        {
            warn!(
                "force_link: cross-layer relation '{}' from L{} → L{} — lower depending on higher may break abstraction semantics",
                relation_type, source_layer, target_layer
            );
        }

        let meta =
            RelationMeta::new("manual_link").with_confidence(strength);
        source.append_relation(relation_type.to_string(), target_uuid, Some(strength), meta);

        if source
            .metadata
            .relations
            .iter()
            .any(|r| r.relation == relation_type && r.target == params.target_id)
        {
            return Err(MemoryError::InvalidInput(format!(
                "Relation '{}' from '{}' to '{}' already exists in metadata",
                relation_type, params.source_id, params.target_id
            )));
        }

        source.metadata.relations.push(crate::types::Relation {
            source: params.source_id.clone(),
            relation: relation_type.to_string(),
            target: params.target_id.clone(),
            strength: Some(strength),
        });

        match self.memory_manager.update_memory(&source).await {
            Ok(_) => {
                info!(
                    "Force-linked {} --[{}]--> {}",
                    params.source_id, relation_type, params.target_id
                );
            }
            Err(e) => {
                error!("Failed to force-link: {}", e);
                return Err(MemoryError::Internal(format!(
                    "Failed to force-link: {}",
                    e
                )));
            }
        }

        let reverse_name = reverse_relation(relation_type);
        if let Some(reverse_type) = reverse_name {
            let reverse_meta =
                RelationMeta::new("manual_link:auto_reverse").with_confidence(strength);
            target.append_relation(reverse_type.to_string(), source_uuid, Some(strength), reverse_meta);

            if !target
                .metadata
                .relations
                .iter()
                .any(|r| r.relation == reverse_type && r.target == params.source_id)
            {
                target.metadata.relations.push(crate::types::Relation {
                    source: params.target_id.clone(),
                    relation: reverse_type.to_string(),
                    target: params.source_id.clone(),
                    strength: Some(strength),
                });
            }

            match self.memory_manager.update_memory(&target).await {
                Ok(_) => {
                    info!(
                        "Auto-created reverse link {} --[{}]--> {}",
                        params.target_id, reverse_type, params.source_id
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to create reverse relation: {} (forward link already saved)",
                        e
                    );
                }
            }
        }

        let mut data = json!({
            "source_id": params.source_id,
            "relation": relation_type,
            "target_id": params.target_id,
        });
        if let Some(rt) = reverse_name {
            data["reverse_relation"] = json!(rt);
            data["reverse_created"] = json!(true);
        } else {
            data["reverse_relation"] = json!(null);
            data["reverse_created"] = json!(false);
        }

        Ok(MemoryOperationResponse::success_with_data(
            "Relation created successfully",
            data,
        ))
    }

    pub async fn remove_relation(
        &self,
        req: RemoveRelationRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
        let params: RemoveRelationParams = req.into();

        let relation_type = params.relation_type.trim();
        if relation_type.is_empty() {
            return Err(MemoryError::InvalidInput(
                "Relation type cannot be empty".into(),
            ));
        }

        if params.target_id.trim().is_empty() {
            return Err(MemoryError::InvalidInput(
                "Target ID cannot be empty".into(),
            ));
        }

        let target_uuid = Uuid::parse_str(&params.target_id).map_err(|_| {
            MemoryError::InvalidInput(format!(
                "Target ID '{}' is not a valid UUID",
                params.target_id
            ))
        })?;

        let mut memory = match self.memory_manager.get(&params.memory_id).await {
            Ok(Some(m)) => m,
            Ok(None) => {
                return Err(MemoryError::NotFound { id: params.memory_id });
            }
            Err(e) => {
                return Err(MemoryError::Internal(format!("{}", e)));
            }
        };

        let found_in_map = memory
            .relations
            .get(relation_type)
            .map(|e| e.target_ids.contains(&target_uuid))
            .unwrap_or(false);

        let found_in_vec = memory
            .metadata
            .relations
            .iter()
            .any(|r| r.relation == relation_type && r.target == params.target_id);

        if !found_in_map && !found_in_vec {
            return Err(MemoryError::InvalidInput(format!(
                "No relation '{}' to '{}' found on memory '{}'",
                relation_type, params.target_id, params.memory_id
            )));
        }

        if let Some(entry) = memory.relations.get_mut(relation_type) {
            entry.target_ids.retain(|id| id != &target_uuid);
            if entry.target_ids.is_empty() {
                memory.relations.remove(relation_type);
            }
        }

        memory.metadata.relations.retain(|r| {
            !(r.relation == relation_type && r.target == params.target_id)
        });

        match self.memory_manager.update_memory(&memory).await {
            Ok(_) => {
                info!(
                    "Removed relation '{}' → '{}' from {}",
                    relation_type, params.target_id, params.memory_id
                );
            }
            Err(e) => {
                error!("Failed to remove relation: {}", e);
                return Err(MemoryError::Internal(format!(
                    "Failed to remove relation: {}",
                    e
                )));
            }
        }

        let reverse_name = reverse_relation(relation_type);
        if let Some(reverse_type) = reverse_name
            && let Ok(Some(mut target)) = self.memory_manager.get(&params.target_id).await
        {
            let source_uuid = Uuid::parse_str(&params.memory_id).ok();
            if let Some(src_uuid) = source_uuid {
                if let Some(entry) = target.relations.get_mut(reverse_type) {
                    entry.target_ids.retain(|id| id != &src_uuid);
                    if entry.target_ids.is_empty() {
                        target.relations.remove(reverse_type);
                    }
                }
                target.metadata.relations.retain(|r| {
                    !(r.relation == reverse_type && r.target == params.memory_id)
                });
                match self.memory_manager.update_memory(&target).await {
                    Ok(_) => {
                        info!(
                            "Auto-removed reverse relation '{}' → '{}' from {}",
                            reverse_type, params.memory_id, params.target_id
                        );
                    }
                    Err(e) => {
                        warn!(
                            "Failed to remove reverse relation: {} (forward removal already saved)",
                            e
                        );
                    }
                }
            }
        }

        let mut data = json!({
            "memory_id": params.memory_id,
            "removed_relation": relation_type,
            "removed_target": params.target_id,
        });
        if let Some(rt) = reverse_name {
            data["reverse_relation"] = json!(rt);
            data["reverse_cleaned"] = json!(true);
        } else {
            data["reverse_relation"] = json!(null);
            data["reverse_cleaned"] = json!(false);
        }

        Ok(MemoryOperationResponse::success_with_data(
            "Relation removed successfully",
            data,
        ))
    }

    /// Ingest raw content through the format-aware decomposition pipeline
    pub async fn ingest(
        &self,
        req: IngestRequest,
        agent_id: Option<String>,
    ) -> crate::error::Result<MemoryOperationResponse> {
        let user_id = req
            .metadata
            .as_ref()
            .and_then(|m| m.get("user_id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| self.default_user_id.clone());

        let base_meta = MemoryMetadata::new()
            .with_user_id(user_id.unwrap_or_default())
            .with_agent_id(agent_id.unwrap_or_default());

        match self.memory_manager.ingest(
            crate::memory::ingestion_service::IngestOptions {
                content: req.content,
                content_encoding: req.content_encoding,
                format_hint: req.format_hint,
                file_name: req.file_name,
                auto_link: req.auto_link,
                generate_abstractions: req.generate_abstractions,
                max_chunk_size: req.max_chunk_size,
                user_metadata: Some(base_meta),
                source: req.source,
                describe_images: req.describe_images,
            },
        ).await {
            Ok(result) => {
                let data = serde_json::to_value(&result).unwrap_or(json!({"status": "serialized"}));
                Ok(MemoryOperationResponse::success_with_data(
                    format!(
                        "Ingested {} chunk(s) from {} as {}",
                        result.l0_chunks.len(),
                        result.format,
                        result.detected_mime
                    ),
                    data,
                ))
            }
            Err(e) => {
                error!("Ingest failed: {}", e);
                Err(MemoryError::Internal(format!("Ingest failed: {}", e)))
            }
        }
    }

    // ─── Document Session Operations ─────────────────────────────────────────

    pub fn begin_store_document(
        &self,
        req: BeginStoreDocumentRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
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
            event_at: params.event_at,
        };

        match session_manager.begin_session(metadata) {
            Ok(response) => {
                info!("Created document session: {}", response.session_id);
                let data =
                    serde_json::to_value(&response).map_err(MemoryError::Serialization)?;
                Ok(MemoryOperationResponse::success_with_data(
                    "Document session created",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to create document session: {}", e);
                Err(MemoryError::Internal(format!(
                    "Failed to create document session: {}",
                    e
                )))
            }
        }
    }

    pub fn store_document_part(
        &self,
        req: StoreDocumentPartRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
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
                Err(MemoryError::Internal(format!(
                    "Failed to store document part: {}",
                    e
                )))
            }
        }
    }

    pub async fn upload_document(
        &self,
        req: UploadDocumentRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
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
            return Err(MemoryError::InvalidInput(format!(
                "File not found: {}",
                params.file_path
            )));
        }

        let file_name = params.file_name.unwrap_or_else(|| {
            file_path
                .file_name()
                .unwrap_or(std::ffi::OsStr::new("unknown"))
                .to_string_lossy()
                .to_string()
        });

        // Detect format and parse file content
        let file_ext = file_path
            .extension()
            .unwrap_or_default()
            .to_string_lossy()
            .to_lowercase();

        let is_binary = matches!(
            file_ext.as_str(),
            "docx" | "doc" | "pdf" | "xlsx" | "xls"
        );

        let content = if is_binary {
            let data = std::fs::read(file_path)
                .map_err(|e| MemoryError::Internal(format!("Failed to read file: {}", e)))?;

            let fmt = crate::ingest::format_detect::format_from_extension(&file_name)
                .unwrap_or(crate::ingest::InputFormat::Unknown);

            let (doc, _) = crate::ingest::parsers::parse_binary(&data, fmt)
                .map_err(|e| MemoryError::Internal(format!("Failed to parse document: {}", e)))?;

            let mut text = String::new();
            doc.flatten_to_text(&mut text);
            text
        } else {
            std::fs::read_to_string(file_path)
                .map_err(|e| MemoryError::Internal(format!("Failed to read file: {}", e)))?
        };

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

        let detected_mime = if is_binary {
            let fmt = crate::ingest::format_detect::format_from_extension(&file_name)
                .unwrap_or(crate::ingest::InputFormat::Unknown);
            fmt.mime().to_string()
        } else {
            params.mime_type.clone().unwrap_or_else(|| "text/plain".to_string())
        };

        let metadata = DocumentMetadata {
            file_name: file_name.clone(),
            file_type: Some(detected_mime),
            total_size,
            md5sum: Some(format!("{:x}", md5::compute(content.as_bytes()))),
            user_id: params.user_id,
            agent_id: params.agent_id,
            memory_type: params.memory_type.unwrap_or_else(|| "semantic".to_string()),
            topics: params.topics,
            context: params.context,
            custom_metadata,
            event_at: params.event_at,
        };

        let session_response = session_manager
            .begin_session(metadata)
            .map_err(|e| MemoryError::Internal(format!("Failed to create session: {}", e)))?;

        let session_id = session_response.session_id;

        // Update session with correct char-based chunk count
        session_manager
            .update_expected_parts(&session_id, expected_chunks)
            .map_err(|e| {
                MemoryError::Internal(format!("Failed to update expected parts: {}", e))
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
                "status": "uploading",
                "poll_after_seconds": 3
            }),
        ))
    }

    pub async fn process_document(
        &self,
        req: ProcessDocumentRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
        let session_manager = self.session_manager.clone().ok_or_else(|| {
            MemoryError::Internal("Document session manager not configured".to_string())
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
                return Err(MemoryError::InvalidInput(format!(
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
    ) -> crate::error::Result<MemoryOperationResponse> {
        let session_manager = self.get_session_manager()?;

        let params: StatusProcessDocumentParams = req.into();

        info!("Getting status for session: {}", params.session_id);

        match session_manager.get_status(&params.session_id) {
            Ok(status) => {
                let data = serde_json::to_value(&status).map_err(MemoryError::Serialization)?;
                Ok(MemoryOperationResponse::success_with_data(
                    "Session status retrieved",
                    data,
                ))
            }
            Err(e) => {
                error!("Failed to get session status: {}", e);
                Err(MemoryError::Internal(format!(
                    "Failed to get session status: {}",
                    e
                )))
            }
        }
    }

    pub fn list_document_sessions(
        &self,
        _req: ListDocumentSessionsRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
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
                Err(MemoryError::Internal(format!(
                    "Failed to list sessions: {}",
                    e
                )))
            }
        }
    }

    pub fn cancel_process_document(
        &self,
        req: CancelProcessDocumentRequest,
    ) -> crate::error::Result<MemoryOperationResponse> {
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
                Err(MemoryError::Internal(format!(
                    "Failed to cancel session: {}",
                    e
                )))
            }
        }
    }

    fn get_session_manager(
        &self,
    ) -> crate::error::Result<&std::sync::Arc<crate::document_session::DocumentSessionManager>> {
        self.session_manager
            .as_ref()
            .ok_or_else(|| MemoryError::Internal("Document session manager not configured".to_string()))
    }
}
