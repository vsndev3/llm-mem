use crate::ingest::document_tree::DocumentNode;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub content: String,
    pub node_type: String,
    pub parent_id: Option<String>,
    pub section_path: Vec<String>,
    pub order: usize,
}

#[derive(Debug, Clone)]
pub struct StructuralRelation {
    pub source_idx: usize,
    pub target_idx: usize,
    pub relation: String,
    pub strength: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct ChunkingResult {
    pub chunks: Vec<Chunk>,
    pub relations: Vec<StructuralRelation>,
}

pub fn chunk_document(doc: &DocumentNode, max_chunk_size: usize) -> ChunkingResult {
    let mut chunks = Vec::new();
    let mut relations = Vec::new();
    let mut order = 0usize;

    let section_path = Vec::new();

    match doc {
        DocumentNode::Document { children, .. } => {
            let prev_count = chunks.len();
            walk_children(
                children,
                &section_path,
                max_chunk_size,
                &mut chunks,
                &mut relations,
                &mut order,
            );

            for child in children {
                let child_chunks = find_chunks_for_node(child, &chunks, prev_count);
                if child_chunks.len() > 1 {
                    for i in 1..child_chunks.len() {
                        relations.push(StructuralRelation {
                            source_idx: child_chunks[i - 1],
                            target_idx: child_chunks[i],
                            relation: "follows".into(),
                            strength: Some(1.0),
                        });
                    }
                }
            }
        }
        _ => {
            chunk_node(doc, &section_path, max_chunk_size, &mut chunks, &mut order);
        }
    }

    ChunkingResult { chunks, relations }
}

fn walk_children(
    children: &[DocumentNode],
    section_path: &[String],
    max_chunk_size: usize,
    chunks: &mut Vec<Chunk>,
    relations: &mut Vec<StructuralRelation>,
    order: &mut usize,
) {
    let mut prev_sibling: Option<usize> = None;

    for child in children {
        let start_idx = chunks.len();
        chunk_node(child, section_path, max_chunk_size, chunks, order);

        if let Some(prev) = prev_sibling {
            for end_idx in start_idx..chunks.len() {
                relations.push(StructuralRelation {
                    source_idx: prev,
                    target_idx: end_idx,
                    relation: "follows".into(),
                    strength: Some(0.5),
                });
            }
        }

        if !child.children().is_empty() {
            let mut child_path = section_path.to_vec();
            if let DocumentNode::Section {
                title, level: _, ..
            } = child
            {
                child_path.push(title.clone());
            }

            let parent_end_idx = chunks.len() - 1;
            let before_sub = chunks.len();
            walk_children(
                child.children(),
                &child_path,
                max_chunk_size,
                chunks,
                relations,
                order,
            );

            for sub_idx in before_sub..chunks.len() {
                for parent_idx in start_idx..=parent_end_idx {
                    relations.push(StructuralRelation {
                        source_idx: sub_idx,
                        target_idx: parent_idx,
                        relation: "section_of".into(),
                        strength: Some(1.0),
                    });
                }
            }
        }

        for idx in start_idx..chunks.len() {
            prev_sibling = Some(idx);
        }
    }
}

fn chunk_node(
    node: &DocumentNode,
    section_path: &[String],
    max_chunk_size: usize,
    chunks: &mut Vec<Chunk>,
    order: &mut usize,
) {
    match node {
        DocumentNode::Section {
            title, children, ..
        } => {
            let content = format_heading_content(title, children);
            if content.len() <= max_chunk_size || children.is_empty() {
                chunks.push(Chunk {
                    content,
                    node_type: "section".into(),
                    parent_id: None,
                    section_path: section_path.to_vec(),
                    order: *order,
                });
                *order += 1;
            }
        }
        DocumentNode::Paragraph { text, .. } => {
            if text.len() <= max_chunk_size {
                chunks.push(Chunk {
                    content: text.clone(),
                    node_type: "paragraph".into(),
                    parent_id: None,
                    section_path: section_path.to_vec(),
                    order: *order,
                });
                *order += 1;
            } else {
                let sub_chunks = split_text(text, max_chunk_size);
                for sub in &sub_chunks {
                    chunks.push(Chunk {
                        content: sub.clone(),
                        node_type: "paragraph".into(),
                        parent_id: None,
                        section_path: section_path.to_vec(),
                        order: *order,
                    });
                    *order += 1;
                }
            }
        }
        DocumentNode::CodeBlock { language, code, .. } => {
            let content = if code.len() > max_chunk_size {
                format!("```{}\n{}...\n```", language, &code[..max_chunk_size - 20])
            } else {
                format!("```{}\n{}\n```", language, code)
            };
            chunks.push(Chunk {
                content,
                node_type: "code_block".into(),
                parent_id: None,
                section_path: section_path.to_vec(),
                order: *order,
            });
            *order += 1;
        }
        DocumentNode::Table {
            headers,
            rows,
            caption: _,
            ..
        } => {
            let content = format_table(headers, rows);
            if content.len() <= max_chunk_size {
                chunks.push(Chunk {
                    content,
                    node_type: "table".into(),
                    parent_id: None,
                    section_path: section_path.to_vec(),
                    order: *order,
                });
                *order += 1;
            } else {
                let mid = rows.len() / 2;
                let (first, second) = rows.split_at(mid);
                let content1 = format_table(headers, first);
                let content2 = format_table(headers, second);
                chunks.push(Chunk {
                    content: content1,
                    node_type: "table".into(),
                    parent_id: None,
                    section_path: section_path.to_vec(),
                    order: *order,
                });
                *order += 1;
                chunks.push(Chunk {
                    content: content2,
                    node_type: "table".into(),
                    parent_id: None,
                    section_path: section_path.to_vec(),
                    order: *order,
                });
                *order += 1;
            }
        }
        DocumentNode::List { items, .. } => {
            let content = items.join("\n");
            chunks.push(Chunk {
                content,
                node_type: "list".into(),
                parent_id: None,
                section_path: section_path.to_vec(),
                order: *order,
            });
            *order += 1;
        }
        DocumentNode::KeyValue { key, value, .. } => {
            let content = format!("{}: {}", key, value_node_to_string(value));
            chunks.push(Chunk {
                content,
                node_type: "key_value".into(),
                parent_id: None,
                section_path: section_path.to_vec(),
                order: *order,
            });
            *order += 1;
        }
        DocumentNode::Image { alt_text, .. } => {
            chunks.push(Chunk {
                content: format!("[Image: {}]", alt_text),
                node_type: "image".into(),
                parent_id: None,
                section_path: section_path.to_vec(),
                order: *order,
            });
            *order += 1;
        }
        DocumentNode::Raw { content, .. } => {
            chunks.push(Chunk {
                content: content.clone(),
                node_type: "raw".into(),
                parent_id: None,
                section_path: section_path.to_vec(),
                order: *order,
            });
            *order += 1;
        }
        DocumentNode::Document { .. } => {}
    }
}

fn format_heading_content(title: &str, children: &[DocumentNode]) -> String {
    let mut parts = vec![format!("# {}", title)];
    for child in children {
        match child {
            DocumentNode::Paragraph { text, .. } => parts.push(text.clone()),
            DocumentNode::List { items, .. } => parts.push(items.join("\n")),
            DocumentNode::CodeBlock { language, code, .. } => {
                parts.push(format!("```{}\n{}\n```", language, code));
            }
            DocumentNode::Table { headers, rows, .. } => {
                parts.push(format_table(headers, rows));
            }
            _ => {}
        }
    }
    parts.join("\n\n")
}

fn format_table(headers: &[String], rows: &[Vec<String>]) -> String {
    let mut lines = Vec::new();
    lines.push(format!("| {} |", headers.join(" | ")));
    lines.push(format!(
        "|{}|",
        headers
            .iter()
            .map(|_| "---".to_string())
            .collect::<Vec<_>>()
            .join(" | ")
    ));
    for row in rows {
        lines.push(format!("| {} |", row.join(" | ")));
    }
    lines.join("\n")
}

fn value_node_to_string(value: &crate::ingest::document_tree::ValueNode) -> String {
    match value {
        crate::ingest::document_tree::ValueNode::Scalar(s) => s.clone(),
        crate::ingest::document_tree::ValueNode::Null => "null".into(),
        crate::ingest::document_tree::ValueNode::List(items) => {
            format!(
                "[{}]",
                items
                    .iter()
                    .map(value_node_to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        crate::ingest::document_tree::ValueNode::Object(pairs) => {
            let parts: Vec<String> = pairs
                .iter()
                .map(|(k, v)| format!("{}: {}", k, value_node_to_string(v)))
                .collect();
            format!("{{ {} }}", parts.join(", "))
        }
    }
}

fn split_text(text: &str, max_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < text.len() {
        let mut end = (start + max_size).min(text.len());
        if end < text.len() {
            while end > start && !text.is_char_boundary(end) {
                end -= 1;
            }
            if let Some(pos) = text[..end].rfind(". ") {
                if pos > start + max_size / 2 {
                    end = pos + 2;
                }
            } else if let Some(pos) = text[..end].rfind(' ')
                && pos > start + max_size / 2
            {
                end = pos + 1;
            }
        }
        chunks.push(text[start..end].trim().to_string());
        start = end;
    }

    chunks
}

fn find_chunks_for_node(node: &DocumentNode, chunks: &[Chunk], start_offset: usize) -> Vec<usize> {
    let mut indices = Vec::new();
    for (i, chunk) in chunks.iter().enumerate().skip(start_offset) {
        match node {
            DocumentNode::Section { .. } if chunk.node_type == "section" => indices.push(i),
            DocumentNode::Paragraph { .. } if chunk.node_type == "paragraph" => indices.push(i),
            DocumentNode::CodeBlock { .. } if chunk.node_type == "code_block" => indices.push(i),
            DocumentNode::Table { .. } if chunk.node_type == "table" => indices.push(i),
            DocumentNode::List { .. } if chunk.node_type == "list" => indices.push(i),
            DocumentNode::KeyValue { .. } if chunk.node_type == "key_value" => indices.push(i),
            _ => {}
        }
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::document_tree::DocumentNode;

    #[test]
    fn test_chunk_simple_paragraphs() {
        let doc = DocumentNode::Document {
            children: vec![
                DocumentNode::Paragraph {
                    text: "First.".into(),
                    id: None,
                },
                DocumentNode::Paragraph {
                    text: "Second.".into(),
                    id: None,
                },
            ],
            meta: crate::ingest::document_tree::DocumentMeta::new("test", "text/plain", 0),
        };
        let result = chunk_document(&doc, 2000);
        assert_eq!(result.chunks.len(), 2);
        assert_eq!(result.chunks[0].node_type, "paragraph");
        assert!(
            result.relations.iter().any(|r| r.relation == "follows"),
            "Expected follows relation"
        );
    }

    #[test]
    fn test_chunk_section_hierarchy() {
        let doc = DocumentNode::Document {
            children: vec![DocumentNode::Section {
                title: "Section 1".into(),
                level: 1,
                children: vec![DocumentNode::Paragraph {
                    text: "Body text.".into(),
                    id: None,
                }],
                id: None,
            }],
            meta: crate::ingest::document_tree::DocumentMeta::new("test", "text/plain", 0),
        };
        let result = chunk_document(&doc, 2000);
        assert_eq!(result.chunks.len(), 2);
        assert!(result.relations.iter().any(|r| r.relation == "section_of"));
    }

    #[test]
    fn test_chunk_code_block() {
        let doc = DocumentNode::Document {
            children: vec![DocumentNode::CodeBlock {
                language: "rust".into(),
                code: "fn main() {}".into(),
                id: None,
            }],
            meta: crate::ingest::document_tree::DocumentMeta::new("test", "text/plain", 0),
        };
        let result = chunk_document(&doc, 2000);
        assert_eq!(result.chunks.len(), 1);
        assert!(result.chunks[0].content.contains("fn main()"));
    }

    #[test]
    fn test_chunk_key_value() {
        let doc = DocumentNode::Document {
            children: vec![DocumentNode::KeyValue {
                key: "name".into(),
                value: crate::ingest::document_tree::ValueNode::Scalar("Alice".into()),
                id: None,
            }],
            meta: crate::ingest::document_tree::DocumentMeta::new("test", "text/plain", 0),
        };
        let result = chunk_document(&doc, 2000);
        assert_eq!(result.chunks.len(), 1);
        assert!(result.chunks[0].content.contains("Alice"));
    }

    #[test]
    fn test_split_text_long() {
        let long = "A".repeat(3000);
        let result = split_text(&long, 1000);
        assert!(result.len() >= 3);
        for c in &result {
            assert!(c.len() <= 1000);
        }
    }
}
