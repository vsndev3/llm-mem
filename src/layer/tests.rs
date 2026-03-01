use super::abstraction_pipeline::*;
use super::prompts::*;
use crate::types::{Memory, ContentMeta, MemoryMetadata, MemoryType};
use uuid::Uuid;

#[test]
fn test_build_l1_prompt() {
    let content = "The mitochondria is the powerhouse of the cell.".to_string();
    let memory = Memory::with_content(
        content.clone(),
        vec![0.1, 0.2],
        MemoryMetadata::new(MemoryType::Semantic),
    );
    
    let prompt = build_l1_prompt(&memory);
    assert!(prompt.contains(&content));
    assert!(prompt.contains("SOURCE MEMORY (L0)"));
    assert!(prompt.contains("TASK: Generate a concise summary"));
    assert!(prompt.contains("OUTPUT FORMAT"));
}
