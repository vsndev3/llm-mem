use crate::ingest::document_tree::{DocumentMeta, DocumentNode};

pub fn parse_code(
    content: &str,
    byte_size: u64,
    language: &str,
) -> Result<(DocumentNode, DocumentMeta), String> {
    let content = content.trim();
    if content.is_empty() {
        return Err(format!("Empty {} source", language));
    }

    let style = detect_style(content);
    let children = match style {
        CodeStyle::BraceBased => parse_brace_based(content, language),
        CodeStyle::IndentBased => parse_indent_based(content, language),
    };

    let children = if children.is_empty() {
        vec![DocumentNode::CodeBlock {
            language: language.into(),
            code: content.to_string(),
            id: None,
        }]
    } else {
        children
    };

    let meta = DocumentMeta::new(language, mime_for_language(language), byte_size);
    Ok((
        DocumentNode::Document {
            children,
            meta: meta.clone(),
        },
        meta,
    ))
}

#[derive(Debug, PartialEq)]
enum CodeStyle {
    BraceBased,
    IndentBased,
}

fn detect_style(content: &str) -> CodeStyle {
    let brace_lines: usize = content
        .lines()
        .filter(|l| {
            let trimmed = l.trim();
            !trimmed.starts_with("//") && !trimmed.starts_with('#') && trimmed.contains('{')
        })
        .count();

    let indent_lines: usize = content
        .lines()
        .filter(|l| l.starts_with([' ', '\t']) && !l.trim().is_empty())
        .count();

    let total = content.lines().count().max(1);

    if brace_lines > 0 && brace_lines * 5 > total {
        CodeStyle::BraceBased
    } else if indent_lines * 3 > total {
        CodeStyle::IndentBased
    } else {
        CodeStyle::BraceBased
    }
}

fn parse_brace_based(content: &str, language: &str) -> Vec<DocumentNode> {
    let mut nodes = Vec::new();
    let mut pos = 0usize;
    let chars: Vec<char> = content.chars().collect();

    while pos < chars.len() {
        if let Some(body) = find_next_top_level_block(&chars, pos) {
            if body.start > pos {
                let between = chars[pos..body.start]
                    .iter()
                    .collect::<String>()
                    .trim()
                    .to_string();
                if !between.is_empty() && !is_comment_or_blank(&between) {
                    nodes.push(DocumentNode::CodeBlock {
                        language: language.into(),
                        code: between,
                        id: None,
                    });
                }
            }

            let code: String = chars[body.start..body.end].iter().collect();
            let trimmed = code.trim().to_string();
            if !trimmed.is_empty() {
                nodes.push(DocumentNode::CodeBlock {
                    language: language.into(),
                    code: trimmed,
                    id: None,
                });
            }

            pos = body.end;
        } else {
            let remaining: String = chars[pos..].iter().collect();
            let trimmed = remaining.trim().to_string();
            if !trimmed.is_empty() && !is_comment_or_blank(&trimmed) {
                nodes.push(DocumentNode::CodeBlock {
                    language: language.into(),
                    code: trimmed,
                    id: None,
                });
            }
            break;
        }
    }

    nodes
}

struct Block {
    start: usize,
    end: usize,
}

fn find_next_top_level_block(chars: &[char], from: usize) -> Option<Block> {
    let mut in_string = false;
    let mut string_char = ' ';
    let mut in_single_comment = false;
    let mut in_multi_comment = false;
    let mut paren_depth = 0i32;

    let mut block_start = None;

    for i in from..chars.len() {
        let ch = chars[i];

        if in_single_comment {
            if ch == '\n' {
                in_single_comment = false;
            }
            continue;
        }

        if in_multi_comment {
            if ch == '*' && i + 1 < chars.len() && chars[i + 1] == '/' {
                in_multi_comment = false;
            }
            continue;
        }

        if in_string {
            if ch == string_char && !is_escaped(chars, i) {
                in_string = false;
            }
            continue;
        }

        if ch == '/' && i + 1 < chars.len() {
            if chars[i + 1] == '/' {
                in_single_comment = true;
                continue;
            }
            if chars[i + 1] == '*' {
                in_multi_comment = true;
                continue;
            }
        }

        match ch {
            '"' | '\'' | '`' => {
                in_string = true;
                string_char = ch;
            }
            '(' | '[' => {
                paren_depth += 1;
            }
            ')' | ']' => {
                if paren_depth > 0 {
                    paren_depth -= 1;
                }
            }
            '{' if paren_depth == 0 => {
                let start = find_block_start(chars, i);
                block_start = Some(start);
            }
            _ => {}
        }

        if let Some(start) = block_start {
            if let Some(body) = find_braced_body_from_chars(chars, i) {
                return Some(Block {
                    start,
                    end: body.end,
                });
            } else {
                break;
            }
        }
    }

    None
}

fn find_block_start(chars: &[char], brace_pos: usize) -> usize {
    let mut pos = brace_pos;
    while pos > 0 {
        pos -= 1;
        if chars[pos] == '\n' {
            return pos + 1;
        }
    }
    0
}

fn find_braced_body_from_chars(chars: &[char], open_brace_pos: usize) -> Option<Block> {
    let mut brace_depth = 0i32;
    let mut in_string = false;
    let mut string_char = ' ';

    for i in open_brace_pos..chars.len() {
        let ch = chars[i];

        if in_string {
            if ch == string_char && !is_escaped(chars, i) {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' | '\'' | '`' => {
                in_string = true;
                string_char = ch;
            }
            '{' => brace_depth += 1,
            '}' => {
                brace_depth -= 1;
            }
            _ => {}
        }

        if brace_depth == 0 {
            return Some(Block {
                start: open_brace_pos,
                end: i + 1,
            });
        }
    }

    if brace_depth == 0 {
        Some(Block {
            start: open_brace_pos,
            end: chars.len(),
        })
    } else {
        None
    }
}

fn is_escaped(chars: &[char], pos: usize) -> bool {
    if pos == 0 {
        return false;
    }
    let mut backslash_count = 0usize;
    let mut p = pos;
    while p > 0 && chars[p - 1] == '\\' {
        backslash_count += 1;
        p -= 1;
    }
    backslash_count % 2 == 1
}

fn parse_indent_based(content: &str, language: &str) -> Vec<DocumentNode> {
    let lines: Vec<&str> = content.lines().collect();
    let mut nodes = Vec::new();
    let mut i = 0usize;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
            i += 1;
            continue;
        }

        let indent = count_leading_spaces(line);

        if indent == 0 || i == 0 {
            let mut block_lines = vec![line.to_string()];
            i += 1;

            while i < lines.len() {
                let next = lines[i];
                if next.trim().is_empty() {
                    block_lines.push(String::new());
                    i += 1;
                    continue;
                }
                let next_indent = count_leading_spaces(next);
                if next_indent == 0 {
                    break;
                }
                block_lines.push(next.to_string());
                i += 1;
            }

            let code = block_lines.join("\n").trim().to_string();
            if !code.is_empty() {
                nodes.push(DocumentNode::CodeBlock {
                    language: language.into(),
                    code,
                    id: None,
                });
            }
        } else {
            i += 1;
        }
    }

    nodes
}

fn count_leading_spaces(line: &str) -> usize {
    line.chars().take_while(|c| c.is_whitespace()).count()
}

fn is_comment_or_blank(s: &str) -> bool {
    let trimmed = s.trim();
    trimmed.is_empty()
        || trimmed.starts_with("//")
        || trimmed.starts_with('#')
        || trimmed == "/*"
        || trimmed.starts_with("/*")
        || trimmed.ends_with("*/")
}

fn mime_for_language(lang: &str) -> &str {
    match lang {
        "rust" => "text/x-rust",
        "python" => "text/x-python",
        "javascript" | "js" => "text/javascript",
        "typescript" | "ts" => "text/typescript",
        "go" => "text/x-go",
        "cpp" => "text/x-c++",
        _ => "text/plain",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::document_tree::DocumentNode;

    fn count_code_blocks(doc: &DocumentNode) -> usize {
        if let DocumentNode::Document { children, .. } = doc {
            children
                .iter()
                .filter(|c| matches!(c, DocumentNode::CodeBlock { .. }))
                .count()
        } else {
            0
        }
    }

    #[test]
    fn test_detect_style_brace() {
        assert_eq!(
            detect_style("fn main() {\n    println!(\"hi\");\n}\n"),
            CodeStyle::BraceBased
        );
        assert_eq!(
            detect_style("function foo() {\n    return 1;\n}\n"),
            CodeStyle::BraceBased
        );
    }

    #[test]
    fn test_detect_style_indent() {
        assert_eq!(
            detect_style("def foo():\n    return 1\n\ndef bar():\n    pass\n"),
            CodeStyle::IndentBased
        );
    }

    #[test]
    fn test_rust_fn() {
        let src = "fn hello() {\n    println!(\"world\");\n}\n";
        let (doc, _meta) = parse_code(src, src.len() as u64, "rust").unwrap();
        assert!(count_code_blocks(&doc) >= 1);
    }

    #[test]
    fn test_rust_multiple() {
        let src = "fn a() {}\nstruct Foo { x: i32 }\nfn b() {}\n";
        let (doc, _meta) = parse_code(src, src.len() as u64, "rust").unwrap();
        assert!(
            count_code_blocks(&doc) >= 3,
            "Got {} blocks",
            count_code_blocks(&doc)
        );
    }

    #[test]
    fn test_python_def() {
        let src = "def hello():\n    return \"world\"\n";
        let (doc, _meta) = parse_code(src, src.len() as u64, "python").unwrap();
        assert!(count_code_blocks(&doc) >= 1);
    }

    #[test]
    fn test_python_class() {
        let src = "class Foo:\n    def bar(self):\n        pass\n";
        let (doc, _meta) = parse_code(src, src.len() as u64, "python").unwrap();
        assert!(count_code_blocks(&doc) >= 1);
    }

    #[test]
    fn test_js_function() {
        let src = "function hello() {\n    return 'world';\n}\n";
        let (doc, _meta) = parse_code(src, src.len() as u64, "javascript").unwrap();
        assert!(count_code_blocks(&doc) >= 1);
    }

    #[test]
    fn test_js_class() {
        let src = "class Foo {\n    bar() {}\n}\n";
        let (doc, _meta) = parse_code(src, src.len() as u64, "javascript").unwrap();
        assert!(count_code_blocks(&doc) >= 1);
    }

    #[test]
    fn test_go_multiple() {
        let src = "package main\n\nfunc hello() {\n    fmt.Println(\"hi\")\n}\n\nfunc main() {\n    hello()\n}\n";
        let (doc, _meta) = parse_code(src, src.len() as u64, "go").unwrap();
        assert!(
            count_code_blocks(&doc) >= 2,
            "Got {} blocks",
            count_code_blocks(&doc)
        );
    }

    #[test]
    fn test_cpp_class() {
        let src = "class Foo {\npublic:\n    Foo() {}\n    int bar() { return 0; }\n};\n";
        let (doc, _meta) = parse_code(src, src.len() as u64, "cpp").unwrap();
        assert!(count_code_blocks(&doc) >= 1);
    }

    #[test]
    fn test_unknown_language_brace() {
        let src = "class Foo {\n    void bar() {}\n}\n";
        let (doc, _meta) = parse_code(src, src.len() as u64, "java").unwrap();
        assert!(count_code_blocks(&doc) >= 1);
    }

    #[test]
    fn test_empty() {
        assert!(parse_code("", 0, "rust").is_err());
    }

    #[test]
    fn test_strings_not_confused() {
        let src = r#"fn main() {
    let s = "this has { braces } inside";
    let t = 'single { here } too';
    let u = `backtick { here }`;
    println!("{}", s);
}
"#;
        let (doc, _meta) = parse_code(src, src.len() as u64, "rust").unwrap();
        assert_eq!(
            count_code_blocks(&doc),
            1,
            "Braces in strings should not split blocks"
        );
    }

    #[test]
    fn test_comments_not_confused() {
        let src = "fn main() {\n    // this { is a comment\n    /* this { is\n       a { multi-line } comment */\n    println!(\"hi\");\n}\n";
        let (doc, _meta) = parse_code(src, src.len() as u64, "rust").unwrap();
        assert_eq!(
            count_code_blocks(&doc),
            1,
            "Braces in comments should not split blocks"
        );
    }
}
