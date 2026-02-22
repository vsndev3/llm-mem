use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::{
    error::{MemoryError, Result},
    llm::LLMClient,
    memory::extractor::{ExtractedFact, FactCategory},
    memory::utils::remove_code_blocks,
    types::{Memory, MemoryMetadata, MemoryType, ScoredMemory},
    vector_store::VectorStore,
};

/// Actions that can be performed on memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryAction {
    Create {
        content: String,
        metadata: MemoryMetadata,
    },
    Update {
        id: String,
        content: String,
    },
    Delete {
        id: String,
    },
    Merge {
        target_id: String,
        source_ids: Vec<String>,
        merged_content: String,
    },
}

/// Result of memory update operations
#[derive(Debug, Clone)]
pub struct UpdateResult {
    pub actions_performed: Vec<MemoryAction>,
    pub memories_created: Vec<String>,
    pub memories_updated: Vec<String>,
    pub memories_deleted: Vec<String>,
}

/// Trait for updating memories based on extracted facts
#[async_trait]
pub trait MemoryUpdater: Send + Sync {
    /// Update memories based on extracted facts and existing memories
    async fn update_memories(
        &self,
        facts: &[ExtractedFact],
        existing_memories: &[ScoredMemory],
        metadata: &MemoryMetadata,
    ) -> Result<UpdateResult>;

    /// Determine if two memories should be merged
    async fn should_merge(&self, memory1: &Memory, memory2: &Memory) -> Result<bool>;

    /// Merge multiple memories into one
    async fn merge_memories(&self, memories: &[Memory]) -> Result<String>;
}

/// LLM-based memory updater implementation
pub struct LLMMemoryUpdater {
    llm_client: Box<dyn LLMClient>,
    #[allow(dead_code)]
    vector_store: Box<dyn VectorStore>,
    #[allow(dead_code)]
    similarity_threshold: f32,
    merge_threshold: f32,
}

impl LLMMemoryUpdater {
    pub fn new(
        llm_client: Box<dyn LLMClient>,
        vector_store: Box<dyn VectorStore>,
        similarity_threshold: f32,
        merge_threshold: f32,
    ) -> Self {
        Self {
            llm_client,
            vector_store,
            similarity_threshold,
            merge_threshold,
        }
    }

    fn build_update_prompt(
        &self,
        facts: &[ExtractedFact],
        existing_memories: &[ScoredMemory],
    ) -> String {
        let facts_text = facts
            .iter()
            .enumerate()
            .map(|(i, fact)| {
                format!(
                    "{}. {} (importance: {:.2})",
                    i, fact.content, fact.importance
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let memories_text = existing_memories
            .iter()
            .enumerate()
            .map(|(i, scored_memory)| {
                format!(
                    "{}. {} (score: {:.2})",
                    i, scored_memory.memory.content, scored_memory.score
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"Given the following extracted facts and existing memories, determine what actions to take.

EXTRACTED FACTS:
{}

EXISTING MEMORIES:
{}

For each fact, decide one of the following actions:
3. IGNORE - Ignore the fact if it's redundant
2. MERGE - Merge with existing memories if related
1. UPDATE - Update an existing memory if it adds new information
0. CREATE - Create a new memory if completely novel

IMPORTANT: Use ONLY the memory indexes (numbers) from the EXISTING MEMORIES list.

Return your decisions as a JSON array:
[
  {{
    "action": "CREATE|UPDATE|MERGE|IGNORE",
    "fact_index": 0,
    "memory_ids": ["0", "1"],
    "content": "new or updated content",
    "reasoning": "explanation"
  }}
]

Decisions (JSON only):"#,
            facts_text, memories_text
        )
    }

    fn build_merge_prompt(&self, memories: &[Memory]) -> String {
        let memories_text = memories
            .iter()
            .enumerate()
            .map(|(i, memory)| format!("{}. {}", i, memory.content))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"Merge the following related memories into a single, comprehensive memory.
Preserve all important information while removing redundancy.

MEMORIES TO MERGE:
{}

Return only the merged content without any additional explanation:"#,
            memories_text
        )
    }

    fn parse_update_decisions(&self, response: &str) -> Result<Vec<UpdateDecision>> {
        let cleaned_response = remove_code_blocks(response);

        let json_start = cleaned_response.find('[').unwrap_or(0);
        let json_end = cleaned_response
            .rfind(']')
            .map(|i| i + 1)
            .unwrap_or(cleaned_response.len());
        let json_str = &cleaned_response[json_start..json_end];

        match serde_json::from_str::<Vec<serde_json::Value>>(json_str) {
            Ok(decisions_json) => {
                let mut decisions = Vec::new();
                for decision_json in decisions_json {
                    if let Ok(decision) = self.parse_single_decision(&decision_json) {
                        decisions.push(decision);
                    }
                }
                Ok(decisions)
            }
            Err(e) => {
                warn!("Failed to parse update decisions: {}", e);

                if let Ok(extracted_json) = self.extract_json_from_response(&cleaned_response) {
                    if let Ok(decisions_json) =
                        serde_json::from_str::<Vec<serde_json::Value>>(&extracted_json)
                    {
                        let mut decisions = Vec::new();
                        for decision_json in decisions_json {
                            if let Ok(decision) = self.parse_single_decision(&decision_json) {
                                decisions.push(decision);
                            }
                        }
                        return Ok(decisions);
                    }
                }

                Ok(vec![])
            }
        }
    }

    fn extract_json_from_response(&self, response: &str) -> Result<String> {
        let text = response.trim();

        if let Some(pattern) = regex::Regex::new(r"```(?:json)?\s*(.*?)\s*```")
            .unwrap()
            .find(text)
        {
            let json_str = &text[pattern.start() + 3 + 3..pattern.end() - 3];
            Ok(json_str.trim().to_string())
        } else {
            Ok(text.to_string())
        }
    }

    fn parse_single_decision(&self, value: &serde_json::Value) -> Result<UpdateDecision> {
        let action = value["action"]
            .as_str()
            .ok_or_else(|| MemoryError::Parse("Missing action field".to_string()))?;

        let fact_index = value["fact_index"]
            .as_u64()
            .ok_or_else(|| MemoryError::Parse("Missing fact_index field".to_string()))?
            as usize;

        let memory_ids = value["memory_ids"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();

        let content = value["content"].as_str().map(|s| s.to_string());

        let reasoning = value["reasoning"]
            .as_str()
            .map(|s| s.to_string())
            .unwrap_or_default();

        Ok(UpdateDecision {
            action: action.to_string(),
            fact_index,
            memory_ids,
            content,
            reasoning,
        })
    }
}

/// Internal structure for update decisions
#[derive(Debug, Clone)]
struct UpdateDecision {
    action: String,
    fact_index: usize,
    memory_ids: Vec<String>,
    content: Option<String>,
    reasoning: String,
}

/// UUID mapping structure to handle LLM index-to-real-ID mapping
#[derive(Debug, Clone)]
struct UuidMapping {
    temp_to_real: HashMap<String, String>,
    #[allow(dead_code)]
    real_to_temp: HashMap<String, String>,
}

impl UuidMapping {
    fn new() -> Self {
        Self {
            temp_to_real: HashMap::new(),
            real_to_temp: HashMap::new(),
        }
    }

    fn create_from_existing_memories(&mut self, existing_memories: &[ScoredMemory]) {
        for (idx, scored_memory) in existing_memories.iter().enumerate() {
            let temp_uuid = idx.to_string();
            let real_uuid = scored_memory.memory.id.clone();

            self.temp_to_real
                .insert(temp_uuid.clone(), real_uuid.clone());
            self.real_to_temp.insert(real_uuid, temp_uuid);
        }
    }

    fn resolve_memory_ids(&self, llm_ids: &[String]) -> Vec<String> {
        llm_ids
            .iter()
            .filter_map(|llm_id| self.temp_to_real.get(llm_id).cloned())
            .collect()
    }
}

#[async_trait]
impl MemoryUpdater for LLMMemoryUpdater {
    async fn update_memories(
        &self,
        facts: &[ExtractedFact],
        existing_memories: &[ScoredMemory],
        metadata: &MemoryMetadata,
    ) -> Result<UpdateResult> {
        if facts.is_empty() {
            return Ok(UpdateResult {
                actions_performed: vec![],
                memories_created: vec![],
                memories_updated: vec![],
                memories_deleted: vec![],
            });
        }

        let mut uuid_mapping = UuidMapping::new();
        uuid_mapping.create_from_existing_memories(existing_memories);

        let prompt = self.build_update_prompt(facts, existing_memories);

        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        let response = self.llm_client.complete(&prompt).await?;
        let decisions = self.parse_update_decisions(&response)?;

        let mut result = UpdateResult {
            actions_performed: vec![],
            memories_created: vec![],
            memories_updated: vec![],
            memories_deleted: vec![],
        };

        for decision in decisions {
            if decision.fact_index >= facts.len() {
                warn!("Invalid fact index in decision: {}", decision.fact_index);
                continue;
            }

            let fact = &facts[decision.fact_index];

            match decision.action.as_str() {
                "CREATE" => {
                    let memory_type = match fact.category {
                        FactCategory::Personal => MemoryType::Factual,
                        FactCategory::Preference => MemoryType::Conversational,
                        FactCategory::Factual => MemoryType::Factual,
                        FactCategory::Procedural => MemoryType::Procedural,
                        FactCategory::Contextual => MemoryType::Conversational,
                    };

                    let action = MemoryAction::Create {
                        content: decision.content.unwrap_or_else(|| fact.content.clone()),
                        metadata: MemoryMetadata {
                            memory_type,
                            ..metadata.clone()
                        },
                    };

                    result.actions_performed.push(action);
                    debug!("Decided to CREATE memory for fact: {}", fact.content);
                }
                "UPDATE" => {
                    let resolved_ids = uuid_mapping.resolve_memory_ids(&decision.memory_ids);

                    if let Some(memory_id) = resolved_ids.first() {
                        if self.vector_store.get(memory_id).await.is_ok() {
                            let action = MemoryAction::Update {
                                id: memory_id.clone(),
                                content: decision.content.unwrap_or_else(|| fact.content.clone()),
                            };
                            result.actions_performed.push(action);
                            result.memories_updated.push(memory_id.clone());
                        } else {
                            let create_action = MemoryAction::Create {
                                content: decision.content.unwrap_or_else(|| fact.content.clone()),
                                metadata: MemoryMetadata {
                                    memory_type: match fact.category {
                                        FactCategory::Personal => MemoryType::Personal,
                                        FactCategory::Preference => MemoryType::Personal,
                                        FactCategory::Factual => MemoryType::Factual,
                                        FactCategory::Procedural => MemoryType::Procedural,
                                        FactCategory::Contextual => MemoryType::Conversational,
                                    },
                                    ..metadata.clone()
                                },
                            };
                            result.actions_performed.push(create_action);
                        }
                    } else {
                        let create_action = MemoryAction::Create {
                            content: decision.content.unwrap_or_else(|| fact.content.clone()),
                            metadata: MemoryMetadata {
                                memory_type: match fact.category {
                                    FactCategory::Personal => MemoryType::Personal,
                                    FactCategory::Preference => MemoryType::Personal,
                                    FactCategory::Factual => MemoryType::Factual,
                                    FactCategory::Procedural => MemoryType::Procedural,
                                    FactCategory::Contextual => MemoryType::Conversational,
                                },
                                ..metadata.clone()
                            },
                        };
                        result.actions_performed.push(create_action);
                    }
                }
                "MERGE" => {
                    let resolved_ids = uuid_mapping.resolve_memory_ids(&decision.memory_ids);

                    let mut valid_ids = Vec::new();
                    for memory_id in &resolved_ids {
                        if self.vector_store.get(memory_id).await.is_ok() {
                            valid_ids.push(memory_id.clone());
                        }
                    }

                    if valid_ids.len() >= 2 {
                        let target_id = valid_ids[0].clone();
                        let source_ids = valid_ids[1..].to_vec();

                        let action = MemoryAction::Merge {
                            target_id: target_id.clone(),
                            source_ids: source_ids.clone(),
                            merged_content: decision
                                .content
                                .unwrap_or_else(|| fact.content.clone()),
                        };

                        result.actions_performed.push(action);
                        result.memories_updated.push(target_id);
                        result.memories_deleted.extend(source_ids);
                    } else if valid_ids.len() == 1 {
                        let update_action = MemoryAction::Update {
                            id: valid_ids[0].clone(),
                            content: decision.content.unwrap_or_else(|| fact.content.clone()),
                        };
                        result.actions_performed.push(update_action);
                        result.memories_updated.push(valid_ids[0].clone());
                    } else {
                        let create_action = MemoryAction::Create {
                            content: decision.content.unwrap_or_else(|| fact.content.clone()),
                            metadata: MemoryMetadata {
                                memory_type: match fact.category {
                                    FactCategory::Personal => MemoryType::Personal,
                                    FactCategory::Preference => MemoryType::Personal,
                                    FactCategory::Factual => MemoryType::Factual,
                                    FactCategory::Procedural => MemoryType::Procedural,
                                    FactCategory::Contextual => MemoryType::Conversational,
                                },
                                ..metadata.clone()
                            },
                        };
                        result.actions_performed.push(create_action);
                    }
                }
                "DELETE" => {
                    let resolved_ids = uuid_mapping.resolve_memory_ids(&decision.memory_ids);

                    for memory_id in resolved_ids {
                        if self.vector_store.get(&memory_id).await.is_ok() {
                            let action = MemoryAction::Delete {
                                id: memory_id.clone(),
                            };
                            result.actions_performed.push(action);
                            result.memories_deleted.push(memory_id);
                        }
                    }
                }
                "IGNORE" => {
                    debug!(
                        "Decided to IGNORE fact: {} (reason: {})",
                        fact.content, decision.reasoning
                    );
                }
                _ => {
                    warn!("Unknown action in decision: {}", decision.action);
                }
            }
        }

        info!(
            "Memory update completed: {} actions performed",
            result.actions_performed.len()
        );
        Ok(result)
    }

    async fn should_merge(&self, memory1: &Memory, memory2: &Memory) -> Result<bool> {
        let embedding1 = &memory1.embedding;
        let embedding2 = &memory2.embedding;

        let dot_product: f32 = embedding1
            .iter()
            .zip(embedding2.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(false);
        }

        let similarity = dot_product / (norm1 * norm2);
        Ok(similarity >= self.merge_threshold)
    }

    async fn merge_memories(&self, memories: &[Memory]) -> Result<String> {
        if memories.is_empty() {
            return Err(MemoryError::validation("No memories to merge"));
        }

        if memories.len() == 1 {
            return Ok(memories[0].content.clone());
        }

        let prompt = self.build_merge_prompt(memories);

        #[cfg(debug_assertions)]
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        let merged_content = self.llm_client.complete(&prompt).await?;

        Ok(merged_content.trim().to_string())
    }
}

/// Factory function to create memory updaters
pub fn create_memory_updater(
    llm_client: Box<dyn LLMClient>,
    vector_store: Box<dyn VectorStore>,
    similarity_threshold: f32,
    merge_threshold: f32,
) -> Box<dyn MemoryUpdater + 'static> {
    Box::new(LLMMemoryUpdater::new(
        llm_client,
        vector_store,
        similarity_threshold,
        merge_threshold,
    ))
}
