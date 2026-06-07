use crate::ingest::document_tree::{DocumentMeta, DocumentNode, ValueNode};

pub fn parse_json(content: &str, byte_size: u64) -> Result<(DocumentNode, DocumentMeta), String> {
    let value: serde_json::Value =
        serde_json::from_str(content).map_err(|e| format!("JSON parse error: {}", e))?;

    let children = json_value_to_nodes(&value, "root");
    let meta = DocumentMeta::new("json", "application/json", byte_size);

    let doc = DocumentNode::Document {
        children,
        meta: meta.clone(),
    };

    Ok((doc, meta))
}

fn json_value_to_nodes(value: &serde_json::Value, key: &str) -> Vec<DocumentNode> {
    match value {
        serde_json::Value::Object(obj) => {
            let mut children = Vec::new();
            for (k, v) in obj {
                children.extend(json_value_to_nodes(v, k));
            }
            children
        }
        serde_json::Value::Array(arr) => {
            let mut children = Vec::new();
            for (i, v) in arr.iter().enumerate() {
                let idx_key = format!("{}[{}]", key, i);
                children.extend(json_value_to_nodes(v, &idx_key));
            }
            children
        }
        _ => {
            vec![DocumentNode::KeyValue {
                key: key.to_string(),
                value: ValueNode::from_json(value),
                id: None,
            }]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_simple_object() {
        let json = r#"{"name": "Alice", "age": 30}"#;
        let (doc, _meta) = parse_json(json, json.len() as u64).unwrap();

        if let DocumentNode::Document { children, .. } = doc {
            assert_eq!(children.len(), 2);
            let keys: Vec<_> = children
                .iter()
                .filter_map(|n| {
                    if let DocumentNode::KeyValue { key, .. } = n {
                        Some(key.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            assert!(keys.contains(&"name"));
            assert!(keys.contains(&"age"));
        }
    }

    #[test]
    fn test_json_array() {
        let json = r#"["a", "b", "c"]"#;
        let (doc, _meta) = parse_json(json, json.len() as u64).unwrap();

        if let DocumentNode::Document { children, .. } = doc {
            assert_eq!(children.len(), 3);
        }
    }

    #[test]
    fn test_json_nested() {
        let json = r#"{"user": {"name": "Alice", "address": {"city": "NYC"}}}"#;
        let (doc, _meta) = parse_json(json, json.len() as u64).unwrap();

        if let DocumentNode::Document { children, .. } = doc {
            assert_eq!(children.len(), 2);
        }
    }
}
