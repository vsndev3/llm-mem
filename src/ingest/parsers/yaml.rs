use crate::ingest::document_tree::{DocumentMeta, DocumentNode, ValueNode};

pub fn parse_yaml(content: &str, byte_size: u64) -> Result<(DocumentNode, DocumentMeta), String> {
    let docs: Vec<serde_yaml::Value> = if content.trim().starts_with("---") {
        let mut docs = Vec::new();
        for doc_str in content.split("\n---") {
            let trimmed = doc_str.trim();
            if trimmed.is_empty() || trimmed == "---" {
                continue;
            }
            let value: serde_yaml::Value = serde_yaml::from_str(trimmed)
                .map_err(|e| format!("YAML parse error in multi-doc: {}", e))?;
            docs.push(value);
        }
        if docs.is_empty() {
            return Err("YAML document starts with --- but contains no content".into());
        }
        docs
    } else {
        let value: serde_yaml::Value =
            serde_yaml::from_str(content).map_err(|e| format!("YAML parse error: {}", e))?;
        vec![value]
    };

    let mut children = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        let prefix = if docs.len() > 1 {
            format!("doc[{}]", i)
        } else {
            "root".to_string()
        };
        children.extend(yaml_value_to_nodes(doc, &prefix, 0));
    }

    let meta = DocumentMeta::new("yaml", "text/yaml", byte_size);

    let doc = DocumentNode::Document {
        children,
        meta: meta.clone(),
    };

    Ok((doc, meta))
}

fn yaml_value_to_nodes(value: &serde_yaml::Value, prefix: &str, depth: usize) -> Vec<DocumentNode> {
    if depth > 20 {
        return vec![DocumentNode::Raw {
            content: format!("{:?}", value),
            mime_type: "text/yaml".into(),
            id: None,
        }];
    }

    match value {
        serde_yaml::Value::Mapping(map) => {
            let mut children = Vec::new();
            for (k, v) in map {
                let key_str = match k {
                    serde_yaml::Value::String(s) => s.clone(),
                    other => format!("{:?}", other),
                };
                children.extend(yaml_value_to_nodes(v, &key_str, depth + 1));
            }
            children
        }
        serde_yaml::Value::Sequence(seq) => {
            let mut children = Vec::new();
            for (i, v) in seq.iter().enumerate() {
                let key = format!("{}[{}]", prefix, i);
                children.extend(yaml_value_to_nodes(v, &key, depth + 1));
            }
            children
        }
        _ => {
            vec![DocumentNode::KeyValue {
                key: prefix.to_string(),
                value: ValueNode::from_yaml(value),
                id: None,
            }]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yaml_simple() {
        let yaml = "name: Alice\nage: 30\n";
        let (doc, _meta) = parse_yaml(yaml, yaml.len() as u64).unwrap();

        if let DocumentNode::Document { children, .. } = doc {
            assert_eq!(children.len(), 2);
        }
    }

    #[test]
    fn test_yaml_nested() {
        let yaml = "server:\n  host: localhost\n  port: 8080\n";
        let (doc, _meta) = parse_yaml(yaml, yaml.len() as u64).unwrap();

        if let DocumentNode::Document { children, .. } = doc {
            assert_eq!(children.len(), 2);
        }
    }

    #[test]
    fn test_yaml_list() {
        let yaml = "items:\n  - one\n  - two\n  - three\n";
        let (doc, _meta) = parse_yaml(yaml, yaml.len() as u64).unwrap();

        if let DocumentNode::Document { children, .. } = doc {
            assert_eq!(children.len(), 3);
        }
    }
}
