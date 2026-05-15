use serde_json::json;
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::{
    document_session::{
        DocumentSession, DocumentSessionManager, ProcessingResult, SessionStatus,
    },
    memory::{
        MemoryManager,
        utils::{chunk_markdown, extract_headers},
    },
    types::{Filters, MemoryMetadata, Relation},
};

pub(crate) async fn process_document_task(
    session_id: String,
    full_content: String,
    session: DocumentSession,
    memory_manager: Arc<MemoryManager>,
    session_manager: Arc<DocumentSessionManager>,
) -> crate::error::Result<()> {
    let mut metadata = MemoryMetadata::new();
    metadata.user_id = session.metadata.user_id.clone();
    metadata.agent_id = session.metadata.agent_id.clone();

    if let Some(topics) = session.metadata.topics {
        metadata.topics = topics;
    }

    if let Some(context) = session.metadata.context {
        metadata.context = context;
    }

    if let Some(custom) = session.metadata.custom_metadata
        && let serde_json::Value::Object(map) = custom
    {
        for (k, v) in map {
            metadata.custom.insert(k, v);
        }
    }

    metadata.custom.insert(
        "file_path".to_string(),
        serde_json::Value::String(session.metadata.file_name.clone()),
    );

    let chunk_size = memory_manager.config().document_chunk_size;
    let chunks = chunk_markdown(&full_content, chunk_size);
    let total_chunks = chunks.len();

    info!(
        "Document split into {} chunks for session {}",
        total_chunks, session_id
    );

    let (start_chunk, initial_memories_created) =
        if let Some(existing_result) = &session.processing_result {
            info!(
                "Resuming session {} from chunk {} (previously processed {} chunks, created {} memories)",
                session_id,
                existing_result.chunks_processed,
                existing_result.chunks_processed,
                existing_result.memories_created
            );
            (
                existing_result.chunks_processed,
                existing_result.memories_created,
            )
        } else {
            info!(
                "Starting fresh processing for session {} ({} chunks)",
                session_id, total_chunks
            );
            (0, 0)
        };

    let mut created_ids = Vec::new();
    let mut previous_id: Option<String> = None;
    let mut header_stack: Vec<(usize, String, String)> = Vec::new();

    let initial_progress = ProcessingResult {
        total_chunks,
        chunks_processed: start_chunk,
        memories_created: initial_memories_created,
        summary: Some(format!("Starting processing of {} chunks...", total_chunks)),
        chunks_enriched: 0,
        chunks_enriching_end: 0,
    };
    let _ = session_manager.store_processing_result(&session_id, &initial_progress);

    info!(
        "Starting to process {} chunks for session {} (headers tracking: enabled)",
        total_chunks, session_id
    );

    let processing_start = std::time::Instant::now();

    let (batch_size, _) = memory_manager.llm_client().batch_config();
    let batch_size = batch_size.max(1);
    let remaining_chunks: Vec<(usize, &String)> =
        chunks.iter().enumerate().skip(start_chunk).collect();

    for batch_slice in remaining_chunks.chunks(batch_size) {
        let batch_start_idx = batch_slice[0].0;
        let batch_end_idx = batch_start_idx + batch_slice.len();
        let batch_texts: Vec<String> = batch_slice.iter().map(|(_, t)| (*t).clone()).collect();

        let in_flight_progress = ProcessingResult {
            total_chunks,
            chunks_processed: batch_start_idx,
            memories_created: initial_memories_created + created_ids.len(),
            summary: Some(format!(
                "Enriching metadata: batch {}-{} of {} chunks",
                batch_start_idx + 1,
                batch_end_idx,
                total_chunks
            )),
            chunks_enriched: batch_start_idx,
            chunks_enriching_end: batch_end_idx,
        };
        let _ = session_manager.store_processing_result(&session_id, &in_flight_progress);

        let batch_enrichments: Vec<crate::memory::extractor::ChunkMetadata> =
            match memory_manager
                .extract_metadata_enrichment_batch(&batch_texts)
                .await
            {
                Ok(results) => results,
                Err(e) => {
                    warn!(
                        "Batch enrichment failed: {}. Using un-enriched text as fallback.",
                        e
                    );
                    batch_texts
                        .iter()
                        .map(|text| crate::memory::extractor::ChunkMetadata {
                            summary: text.trim().to_string(),
                            keywords: vec![],
                        })
                        .collect()
                }
            };

        for (batch_offset, &(i, chunk_text)) in batch_slice.iter().enumerate() {
            let mut chunk_metadata = metadata.clone();
            chunk_metadata
                .custom
                .insert("chunk_index".to_string(), json!(i));
            chunk_metadata
                .custom
                .insert("total_chunks".to_string(), json!(total_chunks));

            let chunk_headers = extract_headers(chunk_text);
            for (level, title) in &chunk_headers {
                let level = *level;
                let title = title.clone();
                while header_stack.last().is_some_and(|(l, _, _)| *l >= level) {
                    header_stack.pop();
                }

                let mut header_meta = metadata.clone();
                header_meta
                    .custom
                    .insert("is_header".to_string(), json!(true));
                header_meta
                    .custom
                    .insert("header_level".to_string(), json!(level));

                if let Some((_, _, parent_id)) = header_stack.last() {
                    header_meta.relations.push(Relation {
                        source: "SELF".to_string(),
                        relation: "part_of".to_string(),
                        target: parent_id.clone(),
                        strength: Some(1.0),
                    });
                }

                match memory_manager
                    .store_with_options(
                        format!("Header: {}", title),
                        header_meta,
                        crate::memory::manager::StoreOptions {
                            deduplicate: Some(true),
                            merge: Some(false),
                            ..Default::default()
                        },
                    )
                    .await
                {
                    Ok(h_id) => {
                        header_stack.push((level, title, h_id));
                    }
                    Err(e) => {
                        error!("Failed to store header node {}: {}", title, e);
                    }
                }
            }

            if let Some((_, _, current_header_id)) = header_stack.last() {
                chunk_metadata.relations.push(Relation {
                    source: "SELF".to_string(),
                    relation: "part_of".to_string(),
                    target: current_header_id.clone(),
                    strength: Some(1.0),
                });
            }

            if let Some(enrichment) = batch_enrichments.get(batch_offset) {
                let mut keywords = enrichment.keywords.clone();
                for (_, title) in &chunk_headers {
                    if !keywords.contains(title) {
                        keywords.push(title.clone());
                    }
                }

                chunk_metadata
                    .custom
                    .insert("summary".to_string(), json!(enrichment.summary));
                chunk_metadata
                    .custom
                    .insert("keywords".to_string(), json!(keywords));
            }

            let memory_id = memory_manager
                .store_with_options(
                    chunk_text.clone(),
                    chunk_metadata,
                    crate::memory::manager::StoreOptions {
                        deduplicate: Some(true),
                        merge: Some(false),
                        ..Default::default()
                    },
                )
                .await?;
            created_ids.push(memory_id.clone());

            if let Some(prev) = previous_id {
                let _ = memory_manager
                    .update(
                        &prev,
                        None,
                        Some(vec![Relation {
                            source: prev.clone(),
                            relation: "next_chunk".to_string(),
                            target: memory_id.clone(),
                            strength: Some(1.0),
                        }]),
                    )
                    .await;

                let _ = memory_manager
                    .update(
                        &memory_id,
                        None,
                        Some(vec![Relation {
                            source: memory_id.clone(),
                            relation: "previous_chunk".to_string(),
                            target: prev,
                            strength: Some(1.0),
                        }]),
                    )
                    .await;
            }

            previous_id = Some(memory_id);

            let progress = ProcessingResult {
                total_chunks,
                chunks_processed: i + 1,
                memories_created: initial_memories_created + created_ids.len(),
                summary: Some(format!("Processing chunk {}/{}", i + 1, total_chunks)),
                chunks_enriched: i + 1,
                chunks_enriching_end: batch_end_idx,
            };
            let _ = session_manager.store_processing_result(&session_id, &progress);

            let progress_interval = if total_chunks <= 50 { 10 } else { 50 };
            if (i + 1) % progress_interval == 0 {
                let elapsed = processing_start.elapsed();
                let elapsed_secs = elapsed.as_secs_f64();
                let chunks_per_sec = (i + 1) as f64 / elapsed_secs;
                let remaining = total_chunks - (i + 1);
                let eta_secs = remaining as f64 / chunks_per_sec;

                let eta_formatted = if eta_secs < 60.0 {
                    format!("{:.0}s", eta_secs)
                } else if eta_secs < 3600.0 {
                    format!("{:.1}m", eta_secs / 60.0)
                } else {
                    format!("{:.1}h", eta_secs / 3600.0)
                };

                info!(
                    "Processing chunk {}/{} ({}%) - {} memories created, {} remaining | Elapsed: {:.1}s, ETA: {} ({:.1} chunks/sec)",
                    i + 1,
                    total_chunks,
                    ((i + 1) as f64 / total_chunks as f64 * 100.0).round(),
                    initial_memories_created + created_ids.len(),
                    remaining,
                    elapsed_secs,
                    eta_formatted,
                    chunks_per_sec
                );
            }
        }
    }

    let processing_result = ProcessingResult {
        total_chunks,
        chunks_processed: total_chunks,
        memories_created: initial_memories_created + created_ids.len(),
        summary: Some(format!(
            "Document ingestion completed. Split into {} chunks.",
            total_chunks
        )),
        chunks_enriched: total_chunks,
        chunks_enriching_end: total_chunks,
    };

    session_manager.store_processing_result(&session_id, &processing_result)?;

    let total_elapsed = processing_start.elapsed();
    info!(
        "Processing completed for session {}: {} chunks processed, {} memories created in {:.1}s",
        session_id,
        total_chunks,
        created_ids.len(),
        total_elapsed.as_secs_f64()
    );

    if let Ok(all_sessions) = session_manager.list_all_sessions() {
        let total_docs = all_sessions.len();
        let completed = all_sessions
            .iter()
            .filter(|s| matches!(s.status, SessionStatus::Completed))
            .count();
        let processing = all_sessions
            .iter()
            .filter(|s| matches!(s.status, SessionStatus::Processing))
            .count();
        let failed = all_sessions
            .iter()
            .filter(|s| matches!(s.status, SessionStatus::Failed))
            .count();
        let pending = all_sessions
            .iter()
            .filter(|s| matches!(s.status, SessionStatus::Uploading))
            .count();
        info!(
            "Document queue progress: {}/{} completed, {} processing, {} pending, {} failed",
            completed, total_docs, processing, pending, failed
        );
    }

    info!("Starting cross-document linking for session {}", session_id);
    let _ = session_manager.update_status(
        &session_id,
        SessionStatus::Processing,
        Some("Linking related documents..."),
    );

    if let Err(e) = process_cross_links(created_ids, memory_manager).await {
        warn!(
            "Cross-document linking failed for session {}: {}",
            session_id, e
        );
    }

    session_manager.update_status(&session_id, SessionStatus::Completed, None)?;

    Ok(())
}

pub(crate) async fn process_cross_links(
    new_ids: Vec<String>,
    memory_manager: Arc<MemoryManager>,
) -> crate::error::Result<()> {
    info!(
        "Starting cross-document linking for {} new memories",
        new_ids.len()
    );

    for id in new_ids {
        let memory = match memory_manager.get(&id).await? {
            Some(m) => m,
            None => continue,
        };

        let keywords = memory
            .metadata
            .custom
            .get("keywords")
            .and_then(|v| v.as_array());

        let keywords_vec: Vec<String> = if let Some(k) = keywords {
            k.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        } else {
            continue;
        };

        if keywords_vec.is_empty() {
            continue;
        }

        for keyword in keywords_vec.iter().take(3) {
            let mut filters = Filters::new();
            if let Some(path) = memory
                .metadata
                .custom
                .get("file_path")
                .and_then(|v| v.as_str())
            {
                filters
                    .custom
                    .insert("exclude_file_path".to_string(), json!(path));
            }

            let results = memory_manager.search(keyword, &filters, 3).await?;

            for scored in results {
                if scored.memory.id == id {
                    continue;
                }

                let is_header = scored
                    .memory
                    .metadata
                    .custom
                    .get("is_header")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                if is_header || scored.score > 0.85 {
                    info!(
                        "Creating cross-link: {} --(references)--> {} (keyword: {})",
                        id, scored.memory.id, keyword
                    );

                    let _ = memory_manager
                        .update(
                            &id,
                            None,
                            Some(vec![Relation {
                                source: id.clone(),
                                relation: "references".to_string(),
                                target: scored.memory.id.clone(),
                                strength: Some(scored.score),
                            }]),
                        )
                        .await;
                }
            }
        }
    }

    Ok(())
}

pub(crate) async fn upload_chunked_task(
    session_id: String,
    content: String,
    chunk_size: usize,
    process_immediately: bool,
    memory_manager: std::sync::Arc<crate::memory::MemoryManager>,
    session_manager: std::sync::Arc<crate::document_session::DocumentSessionManager>,
) {
    use crate::document_session::SessionStatus;

    let total_chars = content.chars().count();
    let expected_chunks = total_chars.div_ceil(chunk_size).max(1);

    info!(
        "Background task: uploading {} in {} chunks",
        session_id, expected_chunks
    );

    let existing_parts = session_manager
        .get_parts(&session_id)
        .unwrap_or_default();
    let already_uploaded = existing_parts.len();

    if already_uploaded > 0 {
        info!(
            "Resuming upload: {} chunks already exist, will skip them",
            already_uploaded
        );
    }

    let chars: Vec<char> = content.chars().collect();
    let mut actual_parts = 0;
    let mut offset = 0;

    let _ = session_manager.update_status(
        &session_id,
        SessionStatus::Uploading,
        None,
    );

    while offset < total_chars {
        let end = std::cmp::min(offset + chunk_size, total_chars);
        let chunk: String = chars[offset..end].iter().collect();

        if actual_parts < already_uploaded {
            offset = end;
            actual_parts += 1;
            continue;
        }

        if let Err(e) =
            session_manager.store_part(&session_id, actual_parts, &chunk)
        {
            error!("Failed to store chunk {}: {}", actual_parts, e);
            let _ = session_manager.update_status(
                &session_id,
                SessionStatus::Failed,
                Some(&format!("Chunk upload failed: {}", e)),
            );
            return;
        }

        actual_parts += 1;
        offset = end;

        let log_interval = if expected_chunks <= 20 {
            1
        } else if expected_chunks <= 100 {
            10
        } else {
            100
        };
        if actual_parts % log_interval == 0 || actual_parts == expected_chunks {
            info!(
                "Uploaded {}/{} chunks ({:.0}%)",
                actual_parts,
                expected_chunks,
                (actual_parts as f64 / expected_chunks as f64) * 100.0
            );
        }
    }

    info!(
        "Stored all {} chunks for session {}",
        actual_parts, session_id
    );

    let _ = session_manager.update_expected_parts(&session_id, actual_parts);

    if process_immediately {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        match session_manager.get_session(&session_id) {
            Ok(session) => {
                let _ = session_manager.update_status(
                    &session_id,
                    SessionStatus::Processing,
                    None,
                );

                let parts = match session_manager.get_parts(&session_id) {
                    Ok(p) => p,
                    Err(e) => {
                        error!("Failed to get parts: {}", e);
                        let _ = session_manager.update_status(
                            &session_id,
                            SessionStatus::Failed,
                            Some(&format!("Failed to get parts: {}", e)),
                        );
                        return;
                    }
                };

                let full_content: String =
                    parts.into_iter().map(|(_, content)| content).collect();

                if let Err(e) = process_document_task(
                    session_id.clone(),
                    full_content,
                    session,
                    memory_manager.clone(),
                    session_manager.clone(),
                )
                .await
                {
                    error!(
                        "Document processing failed for session {}: {}",
                        session_id, e
                    );
                    let _ = session_manager.update_status(
                        &session_id,
                        SessionStatus::Failed,
                        Some(&e.to_string()),
                    );
                }
            }
            Err(e) => {
                error!("Failed to get session for processing: {}", e);
            }
        }
    }
}
