use crate::ingest::document_tree::{DocumentMeta, DocumentNode, ValueNode};

pub fn parse_toml(content: &str, byte_size: u64) -> Result<(DocumentNode, DocumentMeta), String> {
    let value: toml::Value =
        toml::from_str(content).map_err(|e| format!("TOML parse error: {}", e))?;

    let children = toml_table_to_nodes(&value);
    let meta = DocumentMeta::new("toml", "text/toml", byte_size);

    let doc = DocumentNode::Document {
        children,
        meta: meta.clone(),
    };

    Ok((doc, meta))
}

fn toml_table_to_nodes(value: &toml::Value) -> Vec<DocumentNode> {
    match value {
        toml::Value::Table(table) => {
            let mut children = Vec::new();
            let mut sections = Vec::new();
            let mut simple_pairs = Vec::new();

            for (key, val) in table {
                match val {
                    toml::Value::Table(_inner) => {
                        let sub_children = toml_table_to_nodes(val);
                        sections.push(DocumentNode::Section {
                            title: key.clone(),
                            level: 2,
                            children: sub_children,
                            id: None,
                        });
                    }
                    toml::Value::Array(_arr) => {
                        simple_pairs.push(DocumentNode::KeyValue {
                            key: key.clone(),
                            value: ValueNode::from_toml(val),
                            id: None,
                        });
                    }
                    _ => {
                        simple_pairs.push(DocumentNode::KeyValue {
                            key: key.clone(),
                            value: ValueNode::from_toml(val),
                            id: None,
                        });
                    }
                }
            }

            children.extend(simple_pairs);
            children.extend(sections);
            children
        }
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toml_simple() {
        let toml_str = "name = \"Alice\"\nage = 30\n";
        let (doc, _meta) = parse_toml(toml_str, toml_str.len() as u64).unwrap();

        if let DocumentNode::Document { children, .. } = doc {
            assert_eq!(children.len(), 2);
        }
    }

    #[test]
    fn test_toml_section() {
        let toml_str = "[server]\nhost = \"localhost\"\nport = 8080\n";
        let (doc, _meta) = parse_toml(toml_str, toml_str.len() as u64).unwrap();

        if let DocumentNode::Document { children, .. } = doc {
            assert_eq!(children.len(), 1);
            if let DocumentNode::Section { title, level, children, .. } = &children[0] {
                assert_eq!(title, "server");
                assert_eq!(*level, 2);
                assert_eq!(children.len(), 2);
            }
        }
    }

    #[test]
    fn test_toml_empty() {
        let (doc, _meta) = parse_toml("", 0).unwrap();
        if let DocumentNode::Document { children, .. } = doc {
            assert!(children.is_empty());
        }
    }
}
