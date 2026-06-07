use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum DocumentNode {
    Document {
        children: Vec<DocumentNode>,
        meta: DocumentMeta,
    },

    Section {
        title: String,
        level: u8,
        children: Vec<DocumentNode>,
        id: Option<String>,
    },

    Paragraph {
        text: String,
        id: Option<String>,
    },

    Table {
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
        caption: Option<String>,
        id: Option<String>,
    },

    CodeBlock {
        language: String,
        code: String,
        id: Option<String>,
    },

    Image {
        alt_text: String,
        mime_type: String,
        id: Option<String>,
    },

    List {
        items: Vec<String>,
        ordered: bool,
        id: Option<String>,
    },

    KeyValue {
        key: String,
        value: ValueNode,
        id: Option<String>,
    },

    Raw {
        content: String,
        mime_type: String,
        id: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValueNode {
    Scalar(String),
    List(Vec<ValueNode>),
    Object(Vec<(String, ValueNode)>),
    Null,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DocumentMeta {
    pub format: String,
    pub detected_mime: String,
    pub byte_size: u64,
    pub page_count: Option<u32>,
    pub sheet_names: Vec<String>,
    pub parser_confidence: f32,
    pub warnings: Vec<String>,
    pub custom: HashMap<String, String>,
}

impl DocumentMeta {
    pub fn new(format: &str, mime: &str, byte_size: u64) -> Self {
        Self {
            format: format.to_string(),
            detected_mime: mime.to_string(),
            byte_size,
            page_count: None,
            sheet_names: vec![],
            parser_confidence: 1.0,
            warnings: vec![],
            custom: HashMap::new(),
        }
    }
}

impl DocumentNode {
    pub fn flatten_to_text(&self, out: &mut String) {
        match self {
            DocumentNode::Document { children, .. } => {
                for child in children {
                    child.flatten_to_text(out);
                }
            }
            DocumentNode::Section {
                title, children, ..
            } => {
                out.push_str(title);
                out.push('\n');
                for child in children {
                    child.flatten_to_text(out);
                }
            }
            DocumentNode::Paragraph { text, .. } => {
                out.push_str(text);
                out.push('\n');
            }
            DocumentNode::Table { headers, rows, .. } => {
                out.push_str(&headers.join("\t"));
                out.push('\n');
                for row in rows {
                    out.push_str(&row.join("\t"));
                    out.push('\n');
                }
            }
            DocumentNode::CodeBlock { language, code, .. } => {
                out.push_str("```");
                out.push_str(language);
                out.push('\n');
                out.push_str(code);
                out.push_str("```\n");
            }
            DocumentNode::Image { alt_text, .. } => {
                out.push_str(&format!("[Image: {}]", alt_text));
                out.push('\n');
            }
            DocumentNode::List { items, ordered, .. } => {
                for item in items {
                    if *ordered {
                        out.push('1');
                        out.push('.');
                    } else {
                        out.push('-');
                    }
                    out.push(' ');
                    out.push_str(item);
                    out.push('\n');
                }
            }
            DocumentNode::KeyValue { key, value, .. } => {
                out.push_str(key);
                out.push_str(": ");
                flatten_value_to_text(value, out);
                out.push('\n');
            }
            DocumentNode::Raw { content, .. } => {
                out.push_str(content);
                out.push('\n');
            }
        }
    }

    pub fn node_type(&self) -> &'static str {
        match self {
            DocumentNode::Document { .. } => "document",
            DocumentNode::Section { .. } => "section",
            DocumentNode::Paragraph { .. } => "paragraph",
            DocumentNode::Table { .. } => "table",
            DocumentNode::CodeBlock { .. } => "code_block",
            DocumentNode::Image { .. } => "image",
            DocumentNode::List { .. } => "list",
            DocumentNode::KeyValue { .. } => "key_value",
            DocumentNode::Raw { .. } => "raw",
        }
    }

    pub fn id(&self) -> Option<&str> {
        match self {
            DocumentNode::Document { .. } => None,
            DocumentNode::Section { id, .. }
            | DocumentNode::Paragraph { id, .. }
            | DocumentNode::Table { id, .. }
            | DocumentNode::CodeBlock { id, .. }
            | DocumentNode::Image { id, .. }
            | DocumentNode::List { id, .. }
            | DocumentNode::KeyValue { id, .. }
            | DocumentNode::Raw { id, .. } => id.as_deref(),
        }
    }

    pub fn children(&self) -> &[DocumentNode] {
        match self {
            DocumentNode::Document { children, .. } => children.as_slice(),
            DocumentNode::Section { children, .. } => children.as_slice(),
            _ => &[],
        }
    }
}

impl ValueNode {
    pub fn from_json(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => ValueNode::Null,
            serde_json::Value::Bool(b) => ValueNode::Scalar(b.to_string()),
            serde_json::Value::Number(n) => ValueNode::Scalar(n.to_string()),
            serde_json::Value::String(s) => ValueNode::Scalar(s.clone()),
            serde_json::Value::Array(arr) => {
                ValueNode::List(arr.iter().map(ValueNode::from_json).collect())
            }
            serde_json::Value::Object(obj) => {
                let mut pairs = Vec::new();
                for (k, v) in obj {
                    pairs.push((k.clone(), ValueNode::from_json(v)));
                }
                ValueNode::Object(pairs)
            }
        }
    }

    pub fn from_yaml(value: &serde_yaml::Value) -> Self {
        match value {
            serde_yaml::Value::Null => ValueNode::Null,
            serde_yaml::Value::Bool(b) => ValueNode::Scalar(b.to_string()),
            serde_yaml::Value::Number(n) => {
                ValueNode::Scalar(format!("{}", n.as_f64().unwrap_or(0.0)))
            }
            serde_yaml::Value::String(s) => ValueNode::Scalar(s.clone()),
            serde_yaml::Value::Sequence(seq) => {
                ValueNode::List(seq.iter().map(ValueNode::from_yaml).collect())
            }
            serde_yaml::Value::Mapping(map) => {
                let mut pairs = Vec::new();
                for (k, v) in map {
                    let key_str = match k {
                        serde_yaml::Value::String(s) => s.clone(),
                        other => format!("{:?}", other),
                    };
                    pairs.push((key_str, ValueNode::from_yaml(v)));
                }
                ValueNode::Object(pairs)
            }
            _ => ValueNode::Scalar(format!("{:?}", value)),
        }
    }

    pub fn from_toml(value: &toml::Value) -> Self {
        match value {
            toml::Value::String(s) => ValueNode::Scalar(s.clone()),
            toml::Value::Integer(i) => ValueNode::Scalar(i.to_string()),
            toml::Value::Float(f) => ValueNode::Scalar(f.to_string()),
            toml::Value::Boolean(b) => ValueNode::Scalar(b.to_string()),
            toml::Value::Datetime(d) => ValueNode::Scalar(d.to_string()),
            toml::Value::Array(arr) => {
                ValueNode::List(arr.iter().map(ValueNode::from_toml).collect())
            }
            toml::Value::Table(table) => {
                let mut pairs = Vec::new();
                for (k, v) in table {
                    pairs.push((k.clone(), ValueNode::from_toml(v)));
                }
                ValueNode::Object(pairs)
            }
        }
    }
}

fn flatten_value_to_text(value: &ValueNode, out: &mut String) {
    match value {
        ValueNode::Scalar(s) => out.push_str(s),
        ValueNode::List(items) => {
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    out.push(' ');
                }
                flatten_value_to_text(item, out);
            }
        }
        ValueNode::Object(pairs) => {
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    out.push(' ');
                }
                out.push_str(k);
                out.push('=');
                flatten_value_to_text(v, out);
            }
        }
        ValueNode::Null => out.push_str("null"),
    }
}
