use crate::ingest::document_tree::{DocumentMeta, DocumentNode};

pub fn parse_markdown(content: &str, byte_size: u64) -> Result<(DocumentNode, DocumentMeta), String> {
    let parser = pulldown_cmark::Parser::new_ext(content, pulldown_cmark::Options::all());
    let mut nodes = Vec::new();

    let mut current_section: Option<(String, u8, Vec<DocumentNode>)> = None;
    let mut current_text = String::new();
    let mut current_code_block: Option<(String, String)> = None;
    let mut current_list: Option<(Vec<String>, bool)> = None;
    let mut in_heading = false;
    let mut heading_text = String::new();
    let mut heading_level: u8 = 0;

    let mut in_table_head = false;
    let mut table_headers: Vec<String> = Vec::new();
    let mut table_rows: Vec<Vec<String>> = Vec::new();
    let mut current_row_cells: Vec<String> = Vec::new();
    let mut has_active_table = false;

    for event in parser {
        match event {
            pulldown_cmark::Event::Start(tag) => match tag {
                pulldown_cmark::Tag::Heading { level, .. } => {
                    commit_paragraph(&mut current_text, &mut nodes, &mut current_section);
                    commit_list(&mut current_list, &mut nodes, &mut current_section);
                    commit_table(&mut has_active_table, &mut table_headers, &mut table_rows, &mut nodes, &mut current_section);
                    in_heading = true;
                    heading_text.clear();
                    heading_level = level as u8;
                }
                pulldown_cmark::Tag::Paragraph => {
                    current_text.clear();
                }
                pulldown_cmark::Tag::CodeBlock(kind) => {
                    commit_paragraph(&mut current_text, &mut nodes, &mut current_section);
                    commit_list(&mut current_list, &mut nodes, &mut current_section);
                    commit_table(&mut has_active_table, &mut table_headers, &mut table_rows, &mut nodes, &mut current_section);
                    current_code_block = Some((kind_to_language(&kind), String::new()));
                }
                pulldown_cmark::Tag::List(ordered) => {
                    commit_paragraph(&mut current_text, &mut nodes, &mut current_section);
                    current_list = Some((Vec::new(), ordered.is_some()));
                }
                pulldown_cmark::Tag::Table(_align) => {
                    commit_paragraph(&mut current_text, &mut nodes, &mut current_section);
                    commit_list(&mut current_list, &mut nodes, &mut current_section);
                    has_active_table = true;
                    table_headers.clear();
                    table_rows.clear();
                }
                pulldown_cmark::Tag::TableHead => {
                    in_table_head = true;
                }
                pulldown_cmark::Tag::TableRow => {
                    current_row_cells.clear();
                }
                _ => {}
            },

            pulldown_cmark::Event::End(tag) => match tag {
                pulldown_cmark::TagEnd::Heading(_level) => {
                    in_heading = false;
                    if let Some((title, level, children)) = current_section.take()
                        && (!children.is_empty() || !title.is_empty()) {
                            nodes.push(DocumentNode::Section { title, level, children, id: None });
                        }
                    let title = heading_text.trim().to_string();
                    current_section = Some((title, heading_level, Vec::new()));
                    heading_text.clear();
                }
                pulldown_cmark::TagEnd::Paragraph => {
                    commit_paragraph(&mut current_text, &mut nodes, &mut current_section);
                }
                pulldown_cmark::TagEnd::CodeBlock => {
                    if let Some((lang, code)) = current_code_block.take() {
                        push_leaf(&mut nodes, &mut current_section, DocumentNode::CodeBlock {
                            language: lang,
                            code: code.trim().to_string(),
                            id: None,
                        });
                    }
                }
                pulldown_cmark::TagEnd::List(_) => {
                    commit_list(&mut current_list, &mut nodes, &mut current_section);
                }
                pulldown_cmark::TagEnd::Table => {
                    commit_table(&mut has_active_table, &mut table_headers, &mut table_rows, &mut nodes, &mut current_section);
                }
                pulldown_cmark::TagEnd::TableHead => {
                    if !current_row_cells.is_empty() {
                        table_headers = std::mem::take(&mut current_row_cells);
                    }
                    in_table_head = false;
                }
                pulldown_cmark::TagEnd::TableRow => {
                    if !current_row_cells.is_empty() {
                        let cells = std::mem::take(&mut current_row_cells);
                        if in_table_head {
                            table_headers = cells;
                        } else {
                            table_rows.push(cells);
                        }
                    }
                }
                _ => {}
            },

            pulldown_cmark::Event::Text(text) | pulldown_cmark::Event::Code(text) => {
                if let Some((_, code)) = &mut current_code_block {
                    code.push_str(&text);
                } else if in_heading {
                    heading_text.push_str(&text);
                } else if let Some((items, _)) = &mut current_list {
                    let trimmed = text.trim().to_string();
                    if !trimmed.is_empty() {
                        items.push(trimmed);
                    }
                } else if has_active_table {
                    let trimmed = text.trim().to_string();
                    if !trimmed.is_empty() {
                        current_row_cells.push(trimmed);
                    }
                } else {
                    current_text.push_str(&text);
                }
            }

            pulldown_cmark::Event::SoftBreak | pulldown_cmark::Event::HardBreak => {
                if !has_active_table && !in_heading {
                    current_text.push(' ');
                }
            }
            _ => {}
        }
    }

    commit_paragraph(&mut current_text, &mut nodes, &mut current_section);
    commit_list(&mut current_list, &mut nodes, &mut current_section);
    commit_table(&mut has_active_table, &mut table_headers, &mut table_rows, &mut nodes, &mut current_section);

    if let Some((title, level, children)) = current_section {
        nodes.push(DocumentNode::Section { title, level, children, id: None });
    }

    let meta = DocumentMeta::new("markdown", "text/markdown", byte_size);

    let doc = DocumentNode::Document { children: nodes, meta: meta.clone() };
    Ok((doc, meta))
}

fn push_leaf(nodes: &mut Vec<DocumentNode>, section: &mut Option<(String, u8, Vec<DocumentNode>)>, node: DocumentNode) {
    if let Some((_, _, children)) = section {
        children.push(node);
    } else {
        nodes.push(node);
    }
}

fn commit_paragraph(text: &mut String, nodes: &mut Vec<DocumentNode>, section: &mut Option<(String, u8, Vec<DocumentNode>)>) {
    let trimmed = text.trim().to_string();
    if !trimmed.is_empty() {
        push_leaf(nodes, section, DocumentNode::Paragraph { text: trimmed, id: None });
    }
    text.clear();
}

fn commit_list(list: &mut Option<(Vec<String>, bool)>, nodes: &mut Vec<DocumentNode>, section: &mut Option<(String, u8, Vec<DocumentNode>)>) {
    if let Some((items, ordered)) = list.take()
        && !items.is_empty() {
            push_leaf(nodes, section, DocumentNode::List { items, ordered, id: None });
        }
}

fn commit_table(active: &mut bool, headers: &mut Vec<String>, rows: &mut Vec<Vec<String>>, nodes: &mut Vec<DocumentNode>, section: &mut Option<(String, u8, Vec<DocumentNode>)>) {
    if *active {
        let h = std::mem::take(headers);
        let r = std::mem::take(rows);
        if !h.is_empty() || !r.is_empty() {
            push_leaf(nodes, section, DocumentNode::Table { headers: h, rows: r, caption: None, id: None });
        }
        *active = false;
    }
}

fn kind_to_language(kind: &pulldown_cmark::CodeBlockKind) -> String {
    match kind {
        pulldown_cmark::CodeBlockKind::Indented => "text".to_string(),
        pulldown_cmark::CodeBlockKind::Fenced(lang) => {
            let s = lang.as_ref();
            if s.is_empty() { "text".to_string() } else { s.to_string() }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markdown_headings_and_paragraphs() {
        let md = "# Title\n\nIntro paragraph.\n\n## Section 1\n\nBody text here.\n";
        let (doc, _meta) = parse_markdown(md, md.len() as u64).unwrap();
        if let DocumentNode::Document { children, .. } = doc {
            assert!(children.len() >= 2, "Expected at least 2 top-level items, got {}", children.len());
            let sections: Vec<_> = children.iter().filter(|n| matches!(n, DocumentNode::Section { .. })).collect();
            assert_eq!(sections.len(), 2, "Expected 2 sections");
        }
    }

    #[test]
    fn test_markdown_code_block() {
        let md = "```rust\nfn main() {}\n```\n";
        let (doc, _meta) = parse_markdown(md, md.len() as u64).unwrap();
        if let DocumentNode::Document { children, .. } = doc {
            assert_eq!(children.len(), 1);
            match &children[0] {
                DocumentNode::CodeBlock { language, code, .. } => {
                    assert_eq!(language, "rust");
                    assert_eq!(code, "fn main() {}");
                }
                other => panic!("Expected CodeBlock, got {:?}", other.node_type()),
            }
        }
    }

    #[test]
    fn test_markdown_table() {
        let md = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |\n";
        let (doc, _meta) = parse_markdown(md, md.len() as u64).unwrap();
        if let DocumentNode::Document { children, .. } = doc {
            let tables: Vec<_> = children.iter().filter(|n| matches!(n, DocumentNode::Table { .. })).collect();
            assert!(!tables.is_empty(), "Expected Table, got: {:?}", children.iter().map(|c| c.node_type()).collect::<Vec<_>>());
            if let DocumentNode::Table { headers, rows, .. } = &tables[0] {
                assert!(!headers.is_empty(), "Table should have headers");
                assert!(!rows.is_empty(), "Table should have rows");
            }
        }
    }

    #[test]
    fn test_markdown_list() {
        let md = "- item1\n- item2\n- item3\n";
        let (doc, _meta) = parse_markdown(md, md.len() as u64).unwrap();
        if let DocumentNode::Document { children, .. } = doc {
            let lists: Vec<_> = children.iter().filter(|n| matches!(n, DocumentNode::List { .. })).collect();
            assert!(!lists.is_empty(), "Expected List, got: {:?}", children.iter().map(|c| c.node_type()).collect::<Vec<_>>());
            if let DocumentNode::List { items, ordered, .. } = &lists[0] {
                assert!(!items.is_empty());
                assert!(!ordered);
            }
        }
    }

    #[test]
    fn test_markdown_empty() {
        let (doc, _meta) = parse_markdown("", 0).unwrap();
        if let DocumentNode::Document { children, .. } = doc {
            assert!(children.is_empty());
        }
    }
}
