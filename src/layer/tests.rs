use super::prompts::*;
use crate::types::{Memory, MemoryMetadata};

#[test]
fn test_build_l1_prompt() {
    let content = "The mitochondria is the powerhouse of the cell.".to_string();
    let memory = Memory::with_content(content.clone(), vec![0.1, 0.2], MemoryMetadata::new());

    let context = L1Context {
        file_name: Some("biology.md"),
        chunk_index: Some(2),
        total_chunks: Some(10),
        section_headers: &["Cell Biology".to_string(), "Organelles".to_string()],
    };
    let prompt = build_l1_prompt(&memory, &context);
    assert!(prompt.contains(&content));
    assert!(prompt.contains("biology.md"));
    assert!(prompt.contains("3 of 10"));
    assert!(prompt.contains("Cell Biology"));
    assert!(prompt.contains("OUTPUT FORMAT"));
}
