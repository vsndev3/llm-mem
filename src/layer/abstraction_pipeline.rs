use std::sync::Arc;
use std::time::Duration;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use dashmap::DashMap;
use uuid::Uuid;
use tracing::{info, warn};

use crate::{
    error::{MemoryError, Result},
    memory::MemoryManager,
    types::{Filters, LayerInfo, Memory, MemoryMetadata, MemoryType, RelationMeta},
};
use super::prompts::{build_l1_prompt, build_l2_prompt, build_l3_prompt};

/// Configuration for abstraction pipeline
#[derive(Debug, Clone)]
pub struct AbstractionConfig {
    pub enabled: bool,
    pub min_memories_for_l1: usize,
    pub l1_processing_delay: Duration,
    pub max_concurrent_tasks: usize,
}

impl Default for AbstractionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_memories_for_l1: 5,
            l1_processing_delay: Duration::from_secs(30),
            max_concurrent_tasks: 3,
        }
    }
}

/// Pending abstraction task
#[derive(Debug, Clone)]
pub struct PendingAbstraction {
    pub memory_id: Uuid,
    pub current_level: i32,
    pub target_level: i32,
    pub retry_count: u32,
    pub queued_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct L1Extraction {
    pub summary: String,
    pub structure_type: String,
    pub key_entities: Vec<String>,
    pub suggested_title: String,
    pub confidence: f32,
}

/// Manages background tasks that create higher-layer abstractions
pub struct AbstractionPipeline {
    pub memory_manager: Arc<MemoryManager>,
    pub config: AbstractionConfig,
    pub pending_queue: Arc<DashMap<Uuid, PendingAbstraction>>,
    shutdown_tx: broadcast::Sender<()>,
}

impl AbstractionPipeline {
    pub fn new(memory_manager: Arc<MemoryManager>, config: AbstractionConfig) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        Self {
            memory_manager,
            config,
            pending_queue: Arc::new(DashMap::new()),
            shutdown_tx,
        }
    }

    /// Expose shutdown sender so it can be triggered externally
    pub fn get_shutdown_sender(&self) -> broadcast::Sender<()> {
        self.shutdown_tx.clone()
    }

    /// Start a background worker for L0 -> L1 abstractions
    pub fn start_l0_to_l1_worker(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            self.l0_to_l1_worker().await;
        })
    }

    async fn l0_to_l1_worker(&self) {
        let mut interval = tokio::time::interval(self.config.l1_processing_delay);
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if !self.config.enabled {
                        continue;
                    }
                    
                    let l0_count = self.count_memories_at_layer(0).await.unwrap_or(0);
                    if l0_count < self.config.min_memories_for_l1 {
                        continue;
                    }

                    let pending_ids = self.find_pending_l0_abstractions().await.unwrap_or_default();
                    
                    for memory_id in pending_ids {
                        if let Err(e) = self.create_l1_abstraction(memory_id).await {
                            warn!("L1 abstraction failed for {}: {}", memory_id, e);
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("L0->L1 worker shutting down");
                    break;
                }
            }
        }
    }

    pub async fn count_memories_at_layer(&self, level: i32) -> Result<usize> {
        let mut filters = Filters::new();
        filters.custom.insert("layer.level".to_string(), serde_json::json!(level));
        let results = self.memory_manager.list(&filters, None).await?;
        Ok(results.len())
    }

    pub async fn find_pending_l0_abstractions(&self) -> Result<Vec<Uuid>> {
        let mut filters = Filters::new();
        filters.custom.insert("layer.level".to_string(), serde_json::json!(0));
        let results = self.memory_manager.list(&filters, None).await?;
        
        let mut f1 = Filters::new();
        f1.custom.insert("layer.level".to_string(), serde_json::json!(1));
        let l1_memories = self.memory_manager.list(&f1, None).await?;
        
        let mut abstracted_sources = std::collections::HashSet::new();
        for m in l1_memories {
            for src in &m.metadata.abstraction_sources {
                abstracted_sources.insert(*src);
            }
        }

        let mut pending = Vec::new();
        for m in results {
            if let Ok(id) = Uuid::parse_str(&m.id)
                && !abstracted_sources.contains(&id)
            {
                pending.push(id);
            }
        }

        Ok(pending)
    }

    /// Create L1 structural abstraction from L0 memory
    pub async fn create_l1_abstraction(&self, memory_id: Uuid) -> Result<String> {
        let l0_memory = self.memory_manager.get(&memory_id.to_string()).await?
            .ok_or_else(|| MemoryError::NotFound { id: memory_id.to_string() })?;
        
        let prompt = build_l1_prompt(&l0_memory);
        let llm_response = self.memory_manager.llm_client().complete(&prompt).await?;
        
        let json_start = llm_response.find('{').unwrap_or(0);
        let json_end = llm_response.rfind('}').unwrap_or(llm_response.len().saturating_sub(1)) + 1;
        let json_str = if json_start < json_end { &llm_response[json_start..json_end] } else { "{}" };
        
        let extraction: L1Extraction = serde_json::from_str(json_str)
            .unwrap_or_else(|_| L1Extraction {
                summary: "Summary generation failed to parse.".to_string(),
                structure_type: "chunk".to_string(),
                key_entities: vec![],
                suggested_title: "Untitled".to_string(),
                confidence: 0.0,
            });
            
        let mut l1_memory = Memory::with_content(
            extraction.summary,
            l0_memory.embedding.clone(),
            MemoryMetadata::new(MemoryType::Semantic)
                .with_layer(LayerInfo::structural())
                .with_abstraction_sources(vec![memory_id]),
        );
        l1_memory.metadata.abstraction_confidence = Some(extraction.confidence);
        
        l1_memory.add_relation(
            "summary_of",
            vec![memory_id],
            Some(0.9),
            RelationMeta::new("llm:structural-abstraction").with_confidence(0.85),
        );
        
        let l1_id = self.memory_manager.store_memory(l1_memory).await?;
        info!("Created L1 abstraction {} from L0 {}", l1_id, memory_id);
        
        Ok(l1_id)
    }

    pub fn start_l1_to_l2_worker(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            self.l1_to_l2_worker().await;
        })
    }

    async fn l1_to_l2_worker(&self) {
        let mut interval = tokio::time::interval(self.config.l1_processing_delay * 2);
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if !self.config.enabled { continue; }
                    let _ = self.process_l1_to_l2().await;
                }
                _ = shutdown_rx.recv() => {
                    info!("L1->L2 worker shutting down");
                    break;
                }
            }
        }
    }

    async fn process_l1_to_l2(&self) -> Result<()> {
        let l1_count = self.count_memories_at_layer(1).await.unwrap_or(0);
        if l1_count < 3 { return Ok(()); }

        let group = self.find_unabstracted_group(1, 3).await?;
        if group.len() < 3 { return Ok(()); }

        self.create_l2_abstraction(group).await?;
        Ok(())
    }

    pub fn start_l2_to_l3_worker(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            self.l2_to_l3_worker().await;
        })
    }

    async fn l2_to_l3_worker(&self) {
        let mut interval = tokio::time::interval(self.config.l1_processing_delay * 4);
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if !self.config.enabled { continue; }
                    let _ = self.process_l2_to_l3().await;
                }
                _ = shutdown_rx.recv() => {
                    info!("L2->L3 worker shutting down");
                    break;
                }
            }
        }
    }

    async fn process_l2_to_l3(&self) -> Result<()> {
        let l2_count = self.count_memories_at_layer(2).await.unwrap_or(0);
        if l2_count < 3 { return Ok(()); }

        let group = self.find_unabstracted_group(2, 3).await?;
        if group.len() < 3 { return Ok(()); }

        self.create_l3_abstraction(group).await?;
        Ok(())
    }

    async fn find_unabstracted_group(&self, layer: i32, size: usize) -> Result<Vec<Uuid>> {
        let mut filters = Filters::new();
        filters.custom.insert("layer.level".to_string(), serde_json::json!(layer));
        let results = self.memory_manager.list(&filters, None).await?;
        
        let mut upper_filters = Filters::new();
        upper_filters.custom.insert("layer.level".to_string(), serde_json::json!(layer + 1));
        let upper_memories = self.memory_manager.list(&upper_filters, None).await?;
        
        let mut abstracted_sources = std::collections::HashSet::new();
        for m in upper_memories {
            for src in &m.metadata.abstraction_sources {
                abstracted_sources.insert(*src);
            }
        }

        let mut pending = Vec::new();
        for m in results {
            if let Ok(id) = Uuid::parse_str(&m.id)
                && !abstracted_sources.contains(&id)
            {
                pending.push(id);
                if pending.len() == size {
                    break;
                }
            }
        }

        Ok(pending)
    }

    pub async fn create_l2_abstraction(&self, memory_ids: Vec<Uuid>) -> Result<String> {
        let mut memories = Vec::new();
        for id in &memory_ids {
            if let Some(m) = self.memory_manager.get(&id.to_string()).await? {
                memories.push(m);
            }
        }
        
        let memory_refs: Vec<&Memory> = memories.iter().collect();
        let prompt = build_l2_prompt(&memory_refs);
        let llm_response = self.memory_manager.llm_client().complete(&prompt).await?;
        
        let json_start = llm_response.find('{').unwrap_or(0);
        let json_end = llm_response.rfind('}').unwrap_or(llm_response.len().saturating_sub(1)) + 1;
        let json_str = if json_start < json_end { &llm_response[json_start..json_end] } else { "{}" };
        
        let extraction: L2Extraction = serde_json::from_str(json_str)
            .unwrap_or_else(|_| L2Extraction {
                synthesis: "L2 Synthesis failed.".to_string(),
                theme: "Unknown Theme".to_string(),
                shared_entities: vec![],
                confidence: 0.0,
            });
            
        // Calculate average embedding
        let mut avg_embedding = vec![0.0f32; memories[0].embedding.len()];
        for m in &memories {
            for (i, v) in m.embedding.iter().enumerate() {
                if i < avg_embedding.len() { avg_embedding[i] += v; }
            }
        }
        let count_f = memories.len() as f32;
        for v in &mut avg_embedding { *v /= count_f; }

        let mut meta = MemoryMetadata::new(MemoryType::Semantic)
            .with_layer(LayerInfo::semantic())
            .with_abstraction_sources(memory_ids.clone());
        meta.abstraction_confidence = Some(extraction.confidence);
        meta.topics.push(extraction.theme);

        let mut l2_memory = Memory::with_content(
            extraction.synthesis,
            avg_embedding,
            meta,
        );
        
        l2_memory.add_relation(
            "synthesizes",
            memory_ids.clone(),
            Some(0.9),
            RelationMeta::new("llm:semantic-abstraction").with_confidence(0.85),
        );
        
        let l2_id = self.memory_manager.store_memory(l2_memory).await?;
        info!("Created L2 abstraction {} from {} L1 memories", l2_id, memory_ids.len());
        Ok(l2_id)
    }

    pub async fn create_l3_abstraction(&self, memory_ids: Vec<Uuid>) -> Result<String> {
        let mut memories = Vec::new();
        for id in &memory_ids {
            if let Some(m) = self.memory_manager.get(&id.to_string()).await? {
                memories.push(m);
            }
        }
        
        let memory_refs: Vec<&Memory> = memories.iter().collect();
        let prompt = build_l3_prompt(&memory_refs);
        let llm_response = self.memory_manager.llm_client().complete(&prompt).await?;
        
        let json_start = llm_response.find('{').unwrap_or(0);
        let json_end = llm_response.rfind('}').unwrap_or(llm_response.len().saturating_sub(1)) + 1;
        let json_str = if json_start < json_end { &llm_response[json_start..json_end] } else { "{}" };
        
        let extraction: L3Extraction = serde_json::from_str(json_str)
            .unwrap_or_else(|_| L3Extraction {
                insight: "L3 Insight failed.".to_string(),
                concept: "Unknown Concept".to_string(),
                implications: vec![],
                confidence: 0.0,
            });
            
        let mut avg_embedding = vec![0.0f32; memories[0].embedding.len()];
        for m in &memories {
            for (i, v) in m.embedding.iter().enumerate() {
                if i < avg_embedding.len() { avg_embedding[i] += v; }
            }
        }
        let count_f = memories.len() as f32;
        for v in &mut avg_embedding { *v /= count_f; }

        let mut meta = MemoryMetadata::new(MemoryType::Semantic)
            .with_layer(LayerInfo::concept())
            .with_abstraction_sources(memory_ids.clone());
        meta.abstraction_confidence = Some(extraction.confidence);
        meta.topics.push(extraction.concept);

        let mut l3_memory = Memory::with_content(
            extraction.insight,
            avg_embedding,
            meta,
        );
        
        l3_memory.add_relation(
            "abstracts_to_concept",
            memory_ids.clone(),
            Some(0.9),
            RelationMeta::new("llm:conceptual-abstraction").with_confidence(0.85),
        );
        
        let l3_id = self.memory_manager.store_memory(l3_memory).await?;
        info!("Created L3 abstraction {} from {} L2 memories", l3_id, memory_ids.len());
        Ok(l3_id)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct L2Extraction {
    pub synthesis: String,
    pub theme: String,
    pub shared_entities: Vec<String>,
    pub confidence: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct L3Extraction {
    pub insight: String,
    pub concept: String,
    pub implications: Vec<String>,
    pub confidence: f32,
}
