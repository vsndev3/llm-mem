use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    error::{MemoryError, Result},
    memory::MemoryManager,
    types::{Filters, Memory, MemoryState, ScoredMemory},
};

/// Navigation operations for layered memory traversal
pub struct LayerNavigator {
    memory_manager: Arc<MemoryManager>,
}

impl LayerNavigator {
    pub fn new(memory_manager: Arc<MemoryManager>) -> Self {
        Self { memory_manager }
    }

    /// Zoom out: navigate from concrete to abstract
    /// Returns memories at higher abstraction levels
    pub async fn zoom_out(&self, memory_id: &str, levels: usize) -> Result<Vec<Memory>> {
        let start_memory = self.memory_manager.get(memory_id).await?
            .ok_or_else(|| MemoryError::NotFound { id: memory_id.to_string() })?;
        
        let target_level = start_memory.metadata.layer.level + levels as i32;
        let mut results = Vec::new();
        
        // Find memories that abstract from this one
        let abstracted = self.find_abstractions_of(memory_id, target_level).await?;
        results.extend(abstracted);
        
        // Also find semantic links at current level if levels > 0 based on doc (mock or real logic?)
        // For now, this is returning the straightforward abstracted targets
        if levels > 0 {
            let linked = self.find_semantic_links(memory_id).await?;
            results.extend(linked);
        }
        
        Ok(results)
    }
    
    /// Zoom in: navigate from abstract to concrete
    /// Returns source memories that this abstraction was built from
    pub async fn zoom_in(&self, memory_id: &str, levels: usize) -> Result<Vec<Memory>> {
        let start_memory = self.memory_manager.get(memory_id).await?
            .ok_or_else(|| MemoryError::NotFound { id: memory_id.to_string() })?;
        
        let _target_level = start_memory.metadata.layer.level - levels as i32;
        let mut results = Vec::new();
        
        // Follow abstraction sources down to target level
        let sources = self.trace_abstraction_sources(memory_id, levels).await?;
        results.extend(sources);
        
        Ok(results)
    }

    /// Helper to find abstractions of a specific memory at a target layer
    async fn find_abstractions_of(&self, memory_id: &str, target_level: i32) -> Result<Vec<Memory>> {
        // Query memory manager for memories with `layer.level` == target_level
        // AND `abstraction_sources` includes `memory_id`
        
        let mut filters = Filters::new();
        filters.custom.insert("layer.level".to_string(), serde_json::json!(target_level));
        
        let candidates = self.memory_manager.list(&filters, None).await?;
        
        let parsed_id = match uuid::Uuid::parse_str(memory_id) {
            Ok(id) => id,
            Err(_) => return Ok(vec![]),
        };
        
        let mut matches = Vec::new();
        for memory in candidates {
            if memory.metadata.abstraction_sources.contains(&parsed_id) {
                matches.push(memory);
            }
        }
        
        Ok(matches)
    }

    /// Helper to find semantic links (relations)
    async fn find_semantic_links(&self, memory_id: &str) -> Result<Vec<Memory>> {
        let start_memory = self.memory_manager.get(memory_id).await?
            .ok_or_else(|| MemoryError::NotFound { id: memory_id.to_string() })?;
            
        let mut linked = Vec::new();
        for (_, rel) in &start_memory.relations {
            for target_id in &rel.target_ids {
                if let Ok(Some(mem)) = self.memory_manager.get(&target_id.to_string()).await {
                    linked.push(mem);
                }
            }
        }
        
        Ok(linked)
    }
    
    /// Helper to find abstraction sources tracing down
    fn trace_abstraction_sources<'a>(&'a self, memory_id: &'a str, levels: usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Memory>>> + Send + 'a>> {
        Box::pin(async move {
            if levels == 0 {
                return Ok(vec![]);
            }
            
            let start_memory = self.memory_manager.get(memory_id).await?
                .ok_or_else(|| MemoryError::NotFound { id: memory_id.to_string() })?;
                
            let mut sources = Vec::new();
            for src_id in &start_memory.metadata.abstraction_sources {
                if let Ok(Some(mem)) = self.memory_manager.get(&src_id.to_string()).await {
                    if levels == 1 {
                        sources.push(mem);
                    } else {
                        let deeper_sources = self.trace_abstraction_sources(&src_id.to_string(), levels - 1).await?;
                        sources.extend(deeper_sources);
                    }
                }
            }
            
            Ok(sources)
        })
    }

    /// Search at a specific abstraction layer
    pub async fn search_at_layer(
        &self,
        query: &str,
        layer_level: i32,
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        let mut layer_filters = filters.clone();
        layer_filters.custom.insert(
            "layer.level".to_string(),
            serde_json::json!(layer_level),
        );
        
        self.memory_manager.search(query, &layer_filters, limit).await
    }
    
    /// Get all memories at a specific layer
    pub async fn get_layer(&self, layer_level: i32, limit: Option<usize>) -> Result<Vec<Memory>> {
        let mut filters = Filters::new();
        filters.custom.insert(
            "layer.level".to_string(),
            serde_json::json!(layer_level),
        );
        filters.custom.insert(
            "state".to_string(),
            serde_json::json!("active"),
        );
        
        self.memory_manager.list(&filters, limit).await
    }
    
    /// Get layer statistics
    pub async fn get_layer_stats(&self) -> Result<LayerStats> {
        let all_memories = self.memory_manager.list(&Filters::new(), None).await?;
        
        let mut stats = LayerStats::default();
        
        for memory in &all_memories {
            let level = memory.metadata.layer.level;
            
            // Increment the count for this layer
            let layer_count = stats.by_layer.entry(level).or_insert_with(LayerCount::default);
            layer_count.increment();
            
            if memory.metadata.state == MemoryState::Forgotten {
                layer_count.forgotten += 1;
                stats.forgotten_count += 1;
            } else if memory.metadata.state == MemoryState::Active {
                layer_count.active += 1;
            }
        }
        
        stats.total_memories = all_memories.len();
        stats.max_layer = stats.by_layer.keys().max().copied().unwrap_or(0);
        
        Ok(stats)
    }
}

/// Statistics about memory layers
#[derive(Debug, Clone, Default)]
pub struct LayerStats {
    pub total_memories: usize,
    pub forgotten_count: usize,
    pub by_layer: HashMap<i32, LayerCount>,
    pub max_layer: i32,
}

#[derive(Debug, Clone, Default)]
pub struct LayerCount {
    pub count: usize,
    pub active: usize,
    pub forgotten: usize,
}

impl LayerCount {
    pub fn increment(&mut self) {
        self.count += 1;
    }
}
