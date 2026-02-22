use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Core memory structure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Memory {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub metadata: MemoryMetadata,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,

    /// Transient embedding vectors for context tags (not stored in JSON, used for indexing)
    #[serde(skip, default)]
    pub context_embeddings: Option<Vec<Vec<f32>>>,

    /// Transient embedding vectors for relations (not stored in JSON, used for indexing)
    #[serde(skip, default)]
    pub relation_embeddings: Option<Vec<Vec<f32>>>,
}

/// Relationship between this memory and another entity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Relation {
    pub source: String,        // The subject (usually this memory's ID or an entity)
    pub relation: String,      // The predicate (e.g. "mentions", "causes", "next_to")
    pub target: String,        // The object (another memory ID or entity name)
    pub strength: Option<f32>, // Optional strength/weight of this relation (0.0-1.0), used for graph traversal ranking
}

/// Memory metadata for filtering and organization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryMetadata {
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
    pub actor_id: Option<String>,
    pub role: Option<String>,
    pub memory_type: MemoryType,
    pub hash: String,
    pub importance_score: f32,
    pub entities: Vec<String>,
    pub relations: Vec<Relation>,
    pub context: Vec<String>,
    pub topics: Vec<String>,
    pub custom: HashMap<String, serde_json::Value>,
}

/// Types of memory supported by the system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MemoryType {
    Conversational,
    Procedural,
    Factual,
    Semantic,
    Episodic,
    Personal,
    #[serde(rename = "relation_edge")]
    Relation,
}

impl MemoryType {
    pub fn parse(memory_type_str: &str) -> Self {
        match memory_type_str.to_lowercase().as_str() {
            "conversational" => MemoryType::Conversational,
            "procedural" => MemoryType::Procedural,
            "factual" => MemoryType::Factual,
            "semantic" => MemoryType::Semantic,
            "episodic" => MemoryType::Episodic,
            "personal" => MemoryType::Personal,
            "relation_edge" | "relation" => MemoryType::Relation,
            _ => MemoryType::Conversational,
        }
    }

    pub fn parse_with_result(memory_type_str: &str) -> Result<Self, String> {
        match memory_type_str.to_lowercase().as_str() {
            "conversational" => Ok(MemoryType::Conversational),
            "procedural" => Ok(MemoryType::Procedural),
            "factual" => Ok(MemoryType::Factual),
            "semantic" => Ok(MemoryType::Semantic),
            "episodic" => Ok(MemoryType::Episodic),
            "personal" => Ok(MemoryType::Personal),
            "relation_edge" | "relation" => Ok(MemoryType::Relation),
            _ => Err(format!("Invalid memory type: {}", memory_type_str)),
        }
    }
}

/// Memory search result with similarity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredMemory {
    pub memory: Memory,
    pub score: f32,
}

/// Memory operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryResult {
    pub id: String,
    pub memory: String,
    pub event: MemoryEvent,
    pub actor_id: Option<String>,
    pub role: Option<String>,
    pub previous_memory: Option<String>,
}

/// Types of memory operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryEvent {
    Add,
    Update,
    Delete,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationFilter {
    pub relation: String,
    pub target: String,
}

/// Filters for memory search and retrieval
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Filters {
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
    pub actor_id: Option<String>,
    pub memory_type: Option<MemoryType>,
    pub min_importance: Option<f32>,
    pub max_importance: Option<f32>,
    pub created_after: Option<DateTime<Utc>>,
    pub created_before: Option<DateTime<Utc>>,
    pub updated_after: Option<DateTime<Utc>>,
    pub updated_before: Option<DateTime<Utc>>,
    pub entities: Option<Vec<String>>,
    pub topics: Option<Vec<String>>,
    pub relations: Option<Vec<RelationFilter>>,
    /// Level 3: Restrict results to memories whose IDs are in this set.
    /// Used by two-stage context retrieval to narrow down candidates.
    #[serde(skip, default)]
    pub candidate_ids: Option<Vec<String>>,
    pub custom: HashMap<String, serde_json::Value>,
}

/// Message structure for LLM interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
}

/// Memory action determined by LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAction {
    pub id: Option<String>,
    pub text: String,
    pub event: MemoryEvent,
    pub old_memory: Option<String>,
}

impl Memory {
    pub fn new(content: String, embedding: Vec<f32>, metadata: MemoryMetadata) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            content,
            embedding,
            metadata,
            created_at: now,
            updated_at: now,
            context_embeddings: None,
            relation_embeddings: None,
        }
    }

    pub fn update_content(&mut self, content: String, embedding: Vec<f32>) {
        self.content = content;
        self.embedding = embedding;
        self.updated_at = Utc::now();
        self.metadata.hash = Self::compute_hash(&self.content);
    }

    pub fn compute_hash(content: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

impl MemoryMetadata {
    pub fn new(memory_type: MemoryType) -> Self {
        Self {
            user_id: None,
            agent_id: None,
            run_id: None,
            actor_id: None,
            role: None,
            memory_type,
            hash: String::new(),
            importance_score: 0.5,
            entities: Vec::new(),
            relations: Vec::new(),
            context: Vec::new(),
            topics: Vec::new(),
            custom: HashMap::new(),
        }
    }

    pub fn with_user_id(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    pub fn with_agent_id(mut self, agent_id: String) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    pub fn with_run_id(mut self, run_id: String) -> Self {
        self.run_id = Some(run_id);
        self
    }

    pub fn with_actor_id(mut self, actor_id: String) -> Self {
        self.actor_id = Some(actor_id);
        self
    }

    pub fn with_role(mut self, role: String) -> Self {
        self.role = Some(role);
        self
    }

    pub fn with_importance_score(mut self, score: f32) -> Self {
        self.importance_score = score.clamp(0.0, 1.0);
        self
    }

    pub fn with_entities(mut self, entities: Vec<String>) -> Self {
        self.entities = entities;
        self
    }

    pub fn with_topics(mut self, topics: Vec<String>) -> Self {
        self.topics = topics;
        self
    }

    pub fn add_entity(&mut self, entity: String) {
        if !self.entities.contains(&entity) {
            self.entities.push(entity);
        }
    }

    pub fn add_topic(&mut self, topic: String) {
        if !self.topics.contains(&topic) {
            self.topics.push(topic);
        }
    }
}

impl Filters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn for_user(user_id: &str) -> Self {
        Self {
            user_id: Some(user_id.to_string()),
            ..Default::default()
        }
    }

    pub fn for_agent(agent_id: &str) -> Self {
        Self {
            agent_id: Some(agent_id.to_string()),
            ..Default::default()
        }
    }

    pub fn for_run(run_id: &str) -> Self {
        Self {
            run_id: Some(run_id.to_string()),
            ..Default::default()
        }
    }

    pub fn with_memory_type(mut self, memory_type: MemoryType) -> Self {
        self.memory_type = Some(memory_type);
        self
    }
}

impl Message {
    pub fn user<S: Into<String>>(content: S) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            name: None,
        }
    }

    pub fn assistant<S: Into<String>>(content: S) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            name: None,
        }
    }

    pub fn system<S: Into<String>>(content: S) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            name: None,
        }
    }

    pub fn with_name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Memory tests ---

    #[test]
    fn test_memory_new() {
        let meta = MemoryMetadata::new(MemoryType::Factual);
        let mem = Memory::new("hello world".to_string(), vec![1.0, 2.0, 3.0], meta);

        assert_eq!(mem.content, "hello world");
        assert_eq!(mem.embedding, vec![1.0, 2.0, 3.0]);
        assert_eq!(mem.metadata.memory_type, MemoryType::Factual);
        assert!(!mem.id.is_empty());
        assert!(Uuid::parse_str(&mem.id).is_ok());
        assert_eq!(mem.created_at, mem.updated_at);
    }

    #[test]
    fn test_memory_update_content() {
        let meta = MemoryMetadata::new(MemoryType::Conversational);
        let mut mem = Memory::new("old content".to_string(), vec![1.0], meta);
        let old_updated = mem.updated_at;

        std::thread::sleep(std::time::Duration::from_millis(10));
        mem.update_content("new content".to_string(), vec![2.0, 3.0]);

        assert_eq!(mem.content, "new content");
        assert_eq!(mem.embedding, vec![2.0, 3.0]);
        assert!(mem.updated_at >= old_updated);
        assert_eq!(mem.metadata.hash, Memory::compute_hash("new content"));
    }

    #[test]
    fn test_memory_compute_hash_deterministic() {
        let h1 = Memory::compute_hash("test content");
        let h2 = Memory::compute_hash("test content");
        assert_eq!(h1, h2);
        assert!(!h1.is_empty());
    }

    #[test]
    fn test_memory_compute_hash_different_content() {
        let h1 = Memory::compute_hash("content A");
        let h2 = Memory::compute_hash("content B");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_memory_unique_ids() {
        let meta = MemoryMetadata::new(MemoryType::Factual);
        let m1 = Memory::new("a".into(), vec![], meta.clone());
        let m2 = Memory::new("b".into(), vec![], meta);
        assert_ne!(m1.id, m2.id);
    }

    #[test]
    fn test_memory_serialization_roundtrip() {
        let meta = MemoryMetadata::new(MemoryType::Semantic)
            .with_user_id("u1".to_string())
            .with_importance_score(0.9);
        let mem = Memory::new("test".into(), vec![0.5; 4], meta);

        let json = serde_json::to_string(&mem).unwrap();
        let restored: Memory = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id, mem.id);
        assert_eq!(restored.content, mem.content);
        assert_eq!(restored.embedding, mem.embedding);
        assert_eq!(restored.metadata.user_id, mem.metadata.user_id);
    }

    // --- MemoryType tests ---

    #[test]
    fn test_memory_type_parse_all_variants() {
        assert_eq!(
            MemoryType::parse("conversational"),
            MemoryType::Conversational
        );
        assert_eq!(MemoryType::parse("procedural"), MemoryType::Procedural);
        assert_eq!(MemoryType::parse("factual"), MemoryType::Factual);
        assert_eq!(MemoryType::parse("semantic"), MemoryType::Semantic);
        assert_eq!(MemoryType::parse("episodic"), MemoryType::Episodic);
        assert_eq!(MemoryType::parse("personal"), MemoryType::Personal);
    }

    #[test]
    fn test_memory_type_parse_case_insensitive() {
        assert_eq!(
            MemoryType::parse("CONVERSATIONAL"),
            MemoryType::Conversational
        );
        assert_eq!(MemoryType::parse("Factual"), MemoryType::Factual);
        assert_eq!(MemoryType::parse("PERSONAL"), MemoryType::Personal);
    }

    #[test]
    fn test_memory_type_parse_unknown_defaults_to_conversational() {
        assert_eq!(MemoryType::parse("unknown"), MemoryType::Conversational);
        assert_eq!(MemoryType::parse(""), MemoryType::Conversational);
        assert_eq!(MemoryType::parse("foobar"), MemoryType::Conversational);
    }

    #[test]
    fn test_memory_type_parse_with_result_valid() {
        assert!(MemoryType::parse_with_result("factual").is_ok());
        assert_eq!(
            MemoryType::parse_with_result("procedural").unwrap(),
            MemoryType::Procedural
        );
    }

    #[test]
    fn test_memory_type_parse_with_result_invalid() {
        let result = MemoryType::parse_with_result("invalid");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid memory type"));
    }

    #[test]
    fn test_memory_type_serialization() {
        let mt = MemoryType::Episodic;
        let json = serde_json::to_string(&mt).unwrap();
        let restored: MemoryType = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, mt);
    }

    // --- MemoryMetadata tests ---

    #[test]
    fn test_metadata_new_defaults() {
        let meta = MemoryMetadata::new(MemoryType::Factual);
        assert!(meta.user_id.is_none());
        assert!(meta.agent_id.is_none());
        assert!(meta.run_id.is_none());
        assert!(meta.actor_id.is_none());
        assert!(meta.role.is_none());
        assert_eq!(meta.memory_type, MemoryType::Factual);
        assert!(meta.hash.is_empty());
        assert_eq!(meta.importance_score, 0.5);
        assert!(meta.entities.is_empty());
        assert!(meta.topics.is_empty());
        assert!(meta.custom.is_empty());
    }

    #[test]
    fn test_metadata_builder_chain() {
        let meta = MemoryMetadata::new(MemoryType::Personal)
            .with_user_id("user1".to_string())
            .with_agent_id("agent1".to_string())
            .with_run_id("run1".to_string())
            .with_actor_id("actor1".to_string())
            .with_role("user".to_string())
            .with_importance_score(0.85)
            .with_entities(vec!["Alice".into(), "Bob".into()])
            .with_topics(vec!["AI".into()]);

        assert_eq!(meta.user_id.as_deref(), Some("user1"));
        assert_eq!(meta.agent_id.as_deref(), Some("agent1"));
        assert_eq!(meta.run_id.as_deref(), Some("run1"));
        assert_eq!(meta.actor_id.as_deref(), Some("actor1"));
        assert_eq!(meta.role.as_deref(), Some("user"));
        assert_eq!(meta.importance_score, 0.85);
        assert_eq!(meta.entities, vec!["Alice", "Bob"]);
        assert_eq!(meta.topics, vec!["AI"]);
    }

    #[test]
    fn test_metadata_importance_clamped() {
        let meta1 = MemoryMetadata::new(MemoryType::Factual).with_importance_score(2.0);
        assert_eq!(meta1.importance_score, 1.0);

        let meta2 = MemoryMetadata::new(MemoryType::Factual).with_importance_score(-1.0);
        assert_eq!(meta2.importance_score, 0.0);

        let meta3 = MemoryMetadata::new(MemoryType::Factual).with_importance_score(0.5);
        assert_eq!(meta3.importance_score, 0.5);
    }

    #[test]
    fn test_metadata_add_entity_dedup() {
        let mut meta = MemoryMetadata::new(MemoryType::Factual);
        meta.add_entity("Alice".to_string());
        meta.add_entity("Bob".to_string());
        meta.add_entity("Alice".to_string()); // duplicate
        assert_eq!(meta.entities, vec!["Alice", "Bob"]);
    }

    #[test]
    fn test_metadata_add_topic_dedup() {
        let mut meta = MemoryMetadata::new(MemoryType::Factual);
        meta.add_topic("Rust".to_string());
        meta.add_topic("AI".to_string());
        meta.add_topic("Rust".to_string()); // duplicate
        assert_eq!(meta.topics, vec!["Rust", "AI"]);
    }

    // --- Filters tests ---

    #[test]
    fn test_filters_new_is_empty() {
        let f = Filters::new();
        assert!(f.user_id.is_none());
        assert!(f.agent_id.is_none());
        assert!(f.run_id.is_none());
        assert!(f.memory_type.is_none());
        assert!(f.custom.is_empty());
    }

    #[test]
    fn test_filters_for_user() {
        let f = Filters::for_user("user123");
        assert_eq!(f.user_id.as_deref(), Some("user123"));
        assert!(f.agent_id.is_none());
    }

    #[test]
    fn test_filters_for_agent() {
        let f = Filters::for_agent("agent456");
        assert_eq!(f.agent_id.as_deref(), Some("agent456"));
        assert!(f.user_id.is_none());
    }

    #[test]
    fn test_filters_for_run() {
        let f = Filters::for_run("run789");
        assert_eq!(f.run_id.as_deref(), Some("run789"));
    }

    #[test]
    fn test_filters_with_memory_type() {
        let f = Filters::for_user("u1").with_memory_type(MemoryType::Procedural);
        assert_eq!(f.user_id.as_deref(), Some("u1"));
        assert_eq!(f.memory_type, Some(MemoryType::Procedural));
    }

    #[test]
    fn test_filters_serialization() {
        let f = Filters::for_user("u1").with_memory_type(MemoryType::Factual);
        let json = serde_json::to_string(&f).unwrap();
        let restored: Filters = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.user_id, f.user_id);
        assert_eq!(restored.memory_type, f.memory_type);
    }

    // --- Message tests ---

    #[test]
    fn test_message_user() {
        let msg = Message::user("hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "hello");
        assert!(msg.name.is_none());
    }

    #[test]
    fn test_message_assistant() {
        let msg = Message::assistant("response");
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content, "response");
    }

    #[test]
    fn test_message_system() {
        let msg = Message::system("you are helpful");
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content, "you are helpful");
    }

    #[test]
    fn test_message_with_name() {
        let msg = Message::user("hi").with_name("Alice");
        assert_eq!(msg.name.as_deref(), Some("Alice"));
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::user("test").with_name("Bob");
        let json = serde_json::to_string(&msg).unwrap();
        let restored: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.role, "user");
        assert_eq!(restored.content, "test");
        assert_eq!(restored.name.as_deref(), Some("Bob"));
    }

    // --- ScoredMemory tests ---

    #[test]
    fn test_scored_memory() {
        let meta = MemoryMetadata::new(MemoryType::Factual);
        let mem = Memory::new("fact".into(), vec![1.0], meta);
        let scored = ScoredMemory {
            memory: mem.clone(),
            score: 0.95,
        };
        assert_eq!(scored.score, 0.95);
        assert_eq!(scored.memory.content, "fact");
    }

    // --- MemoryEvent tests ---

    #[test]
    fn test_memory_event_equality() {
        assert_eq!(MemoryEvent::Add, MemoryEvent::Add);
        assert_ne!(MemoryEvent::Add, MemoryEvent::Update);
        assert_ne!(MemoryEvent::Delete, MemoryEvent::None);
    }

    // --- MemoryResult tests ---

    #[test]
    fn test_memory_result_construction() {
        let result = MemoryResult {
            id: "id1".into(),
            memory: "content".into(),
            event: MemoryEvent::Add,
            actor_id: Some("actor".into()),
            role: Some("user".into()),
            previous_memory: None,
        };
        assert_eq!(result.id, "id1");
        assert_eq!(result.event, MemoryEvent::Add);
    }

    // --- MemoryAction tests ---

    #[test]
    fn test_memory_action_serialization() {
        let action = MemoryAction {
            id: Some("act-1".into()),
            text: "do something".into(),
            event: MemoryEvent::Update,
            old_memory: Some("old".into()),
        };
        let json = serde_json::to_string(&action).unwrap();
        let restored: MemoryAction = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id, Some("act-1".into()));
        assert_eq!(restored.text, "do something");
    }

    // --- Level 2: Relation & RelationFilter tests ---

    #[test]
    fn test_relation_construction() {
        let rel = Relation {
            source: "mem-1".into(),
            relation: "LIKES".into(),
            target: "Pizza".into(),
            strength: None,
        };
        assert_eq!(rel.source, "mem-1");
        assert_eq!(rel.relation, "LIKES");
        assert_eq!(rel.target, "Pizza");
    }

    #[test]
    fn test_relation_serialization_roundtrip() {
        let rel = Relation {
            source: "mem-1".into(),
            relation: "CAUSES".into(),
            target: "mem-2".into(),
            strength: None,
        };
        let json = serde_json::to_string(&rel).unwrap();
        let restored: Relation = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, rel);
    }

    #[test]
    fn test_relation_equality() {
        let r1 = Relation {
            source: "a".into(),
            relation: "KNOWS".into(),
            target: "b".into(),
            strength: None,
        };
        let r2 = Relation {
            source: "a".into(),
            relation: "KNOWS".into(),
            target: "b".into(),
            strength: None,
        };
        let r3 = Relation {
            source: "a".into(),
            relation: "HATES".into(),
            target: "b".into(),
            strength: None,
        };
        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
    }

    #[test]
    fn test_relation_filter_construction() {
        let rf = RelationFilter {
            relation: "LIKES".into(),
            target: "Pizza".into(),
        };
        assert_eq!(rf.relation, "LIKES");
        assert_eq!(rf.target, "Pizza");
    }

    #[test]
    fn test_metadata_relations_default_empty() {
        let meta = MemoryMetadata::new(MemoryType::Factual);
        assert!(meta.relations.is_empty());
    }

    #[test]
    fn test_metadata_with_relations() {
        let mut meta = MemoryMetadata::new(MemoryType::Personal);
        meta.relations.push(Relation {
            source: "SELF".into(),
            relation: "LIKES".into(),
            target: "Rust".into(),
            strength: None,
        });
        meta.relations.push(Relation {
            source: "SELF".into(),
            relation: "USES".into(),
            target: "Linux".into(),
            strength: None,
        });
        assert_eq!(meta.relations.len(), 2);
        assert_eq!(meta.relations[0].target, "Rust");
        assert_eq!(meta.relations[1].relation, "USES");
    }

    #[test]
    fn test_memory_type_relation_parse() {
        assert_eq!(MemoryType::parse("relation"), MemoryType::Relation);
        assert_eq!(MemoryType::parse("relation_edge"), MemoryType::Relation);
        assert_eq!(MemoryType::parse("Relation"), MemoryType::Relation);
    }

    #[test]
    fn test_memory_type_relation_parse_with_result() {
        assert!(MemoryType::parse_with_result("relation").is_ok());
        assert_eq!(
            MemoryType::parse_with_result("relation_edge").unwrap(),
            MemoryType::Relation
        );
    }

    #[test]
    fn test_memory_type_relation_serialization() {
        let mt = MemoryType::Relation;
        let json = serde_json::to_string(&mt).unwrap();
        assert_eq!(json, "\"relation_edge\""); // serde rename
        let restored: MemoryType = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, MemoryType::Relation);
    }

    #[test]
    fn test_filters_with_relations() {
        let mut f = Filters::new();
        f.relations = Some(vec![RelationFilter {
            relation: "LIKES".into(),
            target: "Pizza".into(),
        }]);
        assert!(f.relations.is_some());
        assert_eq!(f.relations.as_ref().unwrap().len(), 1);
    }

    // --- Level 3: Context, Embeddings, candidate_ids tests ---

    #[test]
    fn test_metadata_context_default_empty() {
        let meta = MemoryMetadata::new(MemoryType::Factual);
        assert!(meta.context.is_empty());
    }

    #[test]
    fn test_metadata_with_context() {
        let mut meta = MemoryMetadata::new(MemoryType::Factual);
        meta.context = vec!["recipe".into(), "italian".into()];
        assert_eq!(meta.context.len(), 2);
        assert_eq!(meta.context[0], "recipe");
    }

    #[test]
    fn test_metadata_context_serialization() {
        let mut meta = MemoryMetadata::new(MemoryType::Factual);
        meta.context = vec!["project-alpha".into()];
        let json = serde_json::to_string(&meta).unwrap();
        let restored: MemoryMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.context, vec!["project-alpha"]);
    }

    #[test]
    fn test_memory_transient_embeddings_skip_serialization() {
        let mut meta = MemoryMetadata::new(MemoryType::Factual);
        meta.context = vec!["test".into()];
        let mut mem = Memory::new("content".into(), vec![1.0, 2.0], meta);
        mem.context_embeddings = Some(vec![vec![0.1, 0.2]]);
        mem.relation_embeddings = Some(vec![vec![0.3, 0.4]]);

        let json = serde_json::to_string(&mem).unwrap();
        let restored: Memory = serde_json::from_str(&json).unwrap();

        // Transient fields should NOT survive serialization
        assert!(restored.context_embeddings.is_none());
        assert!(restored.relation_embeddings.is_none());
        // But regular fields survive
        assert_eq!(restored.content, "content");
        assert_eq!(restored.metadata.context, vec!["test"]);
    }

    #[test]
    fn test_memory_transient_embeddings_default_none() {
        let meta = MemoryMetadata::new(MemoryType::Factual);
        let mem = Memory::new("test".into(), vec![1.0], meta);
        assert!(mem.context_embeddings.is_none());
        assert!(mem.relation_embeddings.is_none());
    }

    #[test]
    fn test_filters_candidate_ids_default_none() {
        let f = Filters::new();
        assert!(f.candidate_ids.is_none());
    }

    #[test]
    fn test_filters_candidate_ids_skip_serialization() {
        let mut f = Filters::new();
        f.candidate_ids = Some(vec!["id-1".into(), "id-2".into()]);

        let json = serde_json::to_string(&f).unwrap();
        let restored: Filters = serde_json::from_str(&json).unwrap();
        // candidate_ids should NOT survive serialization (serde skip)
        assert!(restored.candidate_ids.is_none());
    }
}
