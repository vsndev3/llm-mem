use crate::ingest::document_tree::{DocumentMeta, DocumentNode};

pub fn parse_text(content: &str, byte_size: u64) -> Result<(DocumentNode, DocumentMeta), String> {
    let children = if content.trim().is_empty() {
        vec![]
    } else {
        content
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .map(|p| DocumentNode::Paragraph {
                text: p.trim().to_string(),
                id: None,
            })
            .collect()
    };

    let meta = DocumentMeta::new("text", "text/plain", byte_size);

    let doc = DocumentNode::Document {
        children,
        meta: meta.clone(),
    };

    Ok((doc, meta))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_paragraphs() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let (doc, _meta) = parse_text(text, text.len() as u64).unwrap();

        if let DocumentNode::Document { children, .. } = doc {
            assert_eq!(children.len(), 3);
            if let DocumentNode::Paragraph { text: t, .. } = &children[0] {
                assert_eq!(t, "First paragraph.");
            }
        }
    }

    #[test]
    fn test_text_single_line() {
        let text = "Just one line.";
        let (doc, _meta) = parse_text(text, text.len() as u64).unwrap();

        if let DocumentNode::Document { children, .. } = doc {
            assert_eq!(children.len(), 1);
        }
    }

    #[test]
    fn test_text_empty() {
        let (doc, _meta) = parse_text("", 0).unwrap();
        if let DocumentNode::Document { children, .. } = doc {
            assert_eq!(children.len(), 0);
        }
    }

    #[test]
    fn test_text_whitespace_only() {
        let (doc, _meta) = parse_text("   \n\n  \n", 0).unwrap();
        if let DocumentNode::Document { children, .. } = doc {
            assert_eq!(children.len(), 0);
        }
    }
}
