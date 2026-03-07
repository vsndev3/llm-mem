use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;
use text_splitter::MarkdownSplitter;
use tracing::debug;

// Pre-compiled regexes — compiled once at first use
static RE_CODE_BLOCK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$").unwrap());
// Matches <think>...</think> or <think>... (with or without closing tag)
// The non-greedy .*? ensures we stop at the first closing tag if present
// If no closing tag, (?s) allows . to match newlines until end of input
static RE_THINK_TAG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<think>.*?(?:</think>|$)").unwrap());
static RE_JSON_BLOCK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"```(?:json)?\s*(.*?)\s*```").unwrap());

/// Language information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageInfo {
    pub language_code: String,
    pub language_name: String,
    pub confidence: f32,
}

/// Split markdown text into chunks using semantic boundaries
pub fn chunk_markdown(text: &str, max_chunk_size: usize) -> Vec<String> {
    let splitter = MarkdownSplitter::new(max_chunk_size);
    splitter.chunks(text).map(|s| s.to_string()).collect()
}

/// Extract markdown headers and common TRM heading patterns from a text string
pub fn extract_headers(text: &str) -> Vec<(usize, String)> {
    let mut headers = Vec::new();
    
    // Regex for Chapter/Section patterns
    // Matches: "Chapter 1 Introduction", "1.1 About", "Appendix A", "Section 2.1"
    static RE_TRM_HEADER: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"^(?i)(?:Chapter|Section|Appendix)\s+([A-Z0-9.]+)|^\s*(\d+\.\d+(?:\.\d+)*)\s+").unwrap()
    });

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // 1. Standard Markdown headers
        if trimmed.starts_with('#') {
            let level = trimmed.chars().take_while(|&c| c == '#').count();
            if level > 0 && level <= 6 {
                let title = trimmed.trim_start_matches('#').trim().to_string();
                if !title.is_empty() {
                    headers.push((level, title));
                    continue;
                }
            }
        }

        // 2. TRM Heading patterns (Chapter 1, 1.1, etc.)
        if let Some(_caps) = RE_TRM_HEADER.captures(trimmed) {
            let title = trimmed.to_string();
            // Assign level based on pattern
            let level = if trimmed.to_lowercase().starts_with("chapter")
                || trimmed.to_lowercase().starts_with("appendix")
            {
                1
            } else {
                // Count dots for level (e.g. 1.1 -> level 2, 1.1.1 -> level 3)
                let dots = trimmed.chars().filter(|&c| c == '.').count();
                (dots + 1).min(6)
            };

            // Avoid adding pure ToC lines with dots (e.g. "1.1 About .... 1-2")
            if !trimmed.contains("....") {
                headers.push((level, title));
            }
        }
    }
    headers
}

/// Extract and remove code blocks from text
pub fn remove_code_blocks(content: &str) -> String {
    if let Some(match_result) = RE_CODE_BLOCK.find(content.trim()) {
        let inner_content = &content[match_result.start() + 3..match_result.end() - 3];
        let cleaned = inner_content.trim();

        RE_THINK_TAG
            .replace_all(cleaned, "")
            .replace("\n\n\n", "\n\n")
            .trim()
            .to_string()
    } else {
        RE_THINK_TAG
            .replace_all(content, "")
            .replace("\n\n\n", "\n\n")
            .trim()
            .to_string()
    }
}

/// Extract JSON content from text, removing enclosing triple backticks
pub fn extract_json(text: &str) -> String {
    let text = text.trim();

    if let Some(pattern) = RE_JSON_BLOCK.find(text) {
        let json_str = &text[pattern.start() + 3 + 3..pattern.end() - 3];
        json_str.trim().to_string()
    } else {
        text.to_string()
    }
}

/// Detect language of the input text
pub fn detect_language(text: &str) -> LanguageInfo {
    let clean_text = text.trim().to_lowercase();

    // Chinese detection
    if clean_text
        .chars()
        .any(|c| (c as u32) > 0x4E00 && (c as u32) < 0x9FFF)
    {
        return LanguageInfo {
            language_code: "zh".to_string(),
            language_name: "Chinese".to_string(),
            confidence: 0.9,
        };
    }

    // Japanese detection (Hiragana, Katakana)
    if clean_text.chars().any(|c| {
        (c as u32 >= 0x3040 && c as u32 <= 0x30FF) || ((c as u32) >= 0x4E00 && (c as u32) < 0x9FFF)
    }) {
        return LanguageInfo {
            language_code: "ja".to_string(),
            language_name: "Japanese".to_string(),
            confidence: 0.8,
        };
    }

    // Korean detection
    if clean_text
        .chars()
        .any(|c| c as u32 >= 0xAC00 && c as u32 <= 0xD7AF)
    {
        return LanguageInfo {
            language_code: "ko".to_string(),
            language_name: "Korean".to_string(),
            confidence: 0.8,
        };
    }

    // Russian/Cyrillic detection
    if clean_text
        .chars()
        .any(|c| c as u32 >= 0x0400 && c as u32 <= 0x04FF)
    {
        return LanguageInfo {
            language_code: "ru".to_string(),
            language_name: "Russian".to_string(),
            confidence: 0.9,
        };
    }

    // Arabic detection
    if clean_text
        .chars()
        .any(|c| c as u32 >= 0x0600 && c as u32 <= 0x06FF)
    {
        return LanguageInfo {
            language_code: "ar".to_string(),
            language_name: "Arabic".to_string(),
            confidence: 0.9,
        };
    }

    // Default to English
    LanguageInfo {
        language_code: "en".to_string(),
        language_name: "English".to_string(),
        confidence: 0.7,
    }
}

/// Parse messages from conversation
pub fn parse_messages(messages: &[crate::types::Message]) -> String {
    let mut response = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => response.push_str(&format!("system: {}\n", msg.content)),
            "user" => response.push_str(&format!("user: {}\n", msg.content)),
            "assistant" => response.push_str(&format!("assistant: {}\n", msg.content)),
            _ => debug!("Unknown message role: {}", msg.role),
        }
    }

    response
}

/// Sanitize text for special characters
pub fn sanitize_for_cypher(text: &str) -> String {
    let char_map = HashMap::from([
        ("...", "_ellipsis_"),
        ("'", "_apostrophe_"),
        ("\"", "_quote_"),
        ("\\", "_backslash_"),
        ("/", "_slash_"),
        ("|", "_pipe_"),
        ("&", "_ampersand_"),
        ("=", "_equals_"),
        ("+", "_plus_"),
        ("*", "_asterisk_"),
        ("%", "_percent_"),
        ("#", "_hash_"),
        ("@", "_at_"),
        ("!", "_bang_"),
        ("?", "_question_"),
        ("(", "_lparen_"),
        (")", "_rparen_"),
        ("[", "_lbracket_"),
        ("]", "_rbracket_"),
        ("{", "_lbrace_"),
        ("}", "_rbrace_"),
        ("<", "_langle_"),
        (">", "_rangle_"),
    ]);

    let mut sanitized = text.to_string();

    for (old, new) in &char_map {
        sanitized = sanitized.replace(old, new);
    }

    while sanitized.contains("__") {
        sanitized = sanitized.replace("__", "_");
    }

    sanitized
        .trim_start_matches('_')
        .trim_end_matches('_')
        .to_string()
}

/// Filter message history by role
pub fn filter_messages_by_role(
    messages: &[crate::types::Message],
    role: &str,
) -> Vec<crate::types::Message> {
    messages
        .iter()
        .filter(|msg| msg.role == role)
        .cloned()
        .collect()
}

/// Filter messages by multiple roles
pub fn filter_messages_by_roles(
    messages: &[crate::types::Message],
    roles: &[&str],
) -> Vec<crate::types::Message> {
    messages
        .iter()
        .filter(|msg| roles.contains(&msg.role.as_str()))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    // --- remove_code_blocks tests ---

    #[test]
    fn test_remove_code_blocks_plain_text() {
        let input = "Hello, this is plain text.";
        let result = remove_code_blocks(input);
        assert_eq!(result, "Hello, this is plain text.");
    }

    #[test]
    fn test_remove_code_blocks_with_code_block() {
        let input = "```json\n{\"key\": \"value\"}\n```";
        let result = remove_code_blocks(input);
        assert!(result.contains("\"key\""));
    }

    #[test]
    fn test_remove_code_blocks_with_think_tags() {
        let input = "before <think>thinking stuff</think> after";
        let result = remove_code_blocks(input);
        assert!(!result.contains("<think>"));
        assert!(!result.contains("thinking stuff"));
        assert!(result.contains("before"));
        assert!(result.contains("after"));
    }

    #[test]
    fn test_remove_code_blocks_with_think_tags_missing_closing() {
        // Test missing closing tag
        let input = "before <think>thinking stuff without closing";
        let result = remove_code_blocks(input);
        assert!(!result.contains("<think>"));
        assert!(!result.contains("thinking stuff without closing"));
        assert!(result.contains("before"));
    }

    #[test]
    fn test_remove_code_blocks_with_think_tags_multiline() {
        // Test multiline think tag content
        let input = "before <think>\nLine 1\nLine 2\nLine 3\n</think> after";
        let result = remove_code_blocks(input);
        assert!(!result.contains("<think>"));
        assert!(!result.contains("</think>"));
        assert!(!result.contains("Line 1"));
        assert!(result.contains("before"));
        assert!(result.contains("after"));
    }

    #[test]
    fn test_remove_code_blocks_empty() {
        assert_eq!(remove_code_blocks(""), "");
    }

    // --- extract_json tests ---

    #[test]
    fn test_extract_json_from_code_block() {
        // Without language label
        let input = "```\n{\"key\": \"value\"}\n```";
        let result = extract_json(input);
        assert!(
            result.contains("key"),
            "Should contain key, got: {}",
            result
        );
    }

    #[test]
    fn test_extract_json_plain() {
        let input = r#"{"name": "test"}"#;
        let result = extract_json(input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_extract_json_whitespace() {
        let result = extract_json("   {\"a\": 1}   ");
        assert_eq!(result, "{\"a\": 1}");
    }

    // --- detect_language tests ---

    #[test]
    fn test_detect_language_english() {
        let info = detect_language("Hello world, this is a test");
        assert_eq!(info.language_code, "en");
        assert_eq!(info.language_name, "English");
    }

    #[test]
    fn test_detect_language_chinese() {
        let info = detect_language("你好世界");
        assert_eq!(info.language_code, "zh");
        assert_eq!(info.language_name, "Chinese");
        assert!(info.confidence >= 0.9);
    }

    #[test]
    fn test_detect_language_japanese() {
        let info = detect_language("こんにちは世界");
        // Japanese has hiragana, detected as ja
        assert!(info.language_code == "ja" || info.language_code == "zh");
    }

    #[test]
    fn test_detect_language_korean() {
        let info = detect_language("안녕하세요");
        assert_eq!(info.language_code, "ko");
        assert_eq!(info.language_name, "Korean");
    }

    #[test]
    fn test_detect_language_russian() {
        let info = detect_language("Привет мир");
        assert_eq!(info.language_code, "ru");
        assert_eq!(info.language_name, "Russian");
    }

    #[test]
    fn test_detect_language_arabic() {
        let info = detect_language("مرحبا بالعالم");
        assert_eq!(info.language_code, "ar");
        assert_eq!(info.language_name, "Arabic");
    }

    #[test]
    fn test_detect_language_empty_defaults_english() {
        let info = detect_language("");
        assert_eq!(info.language_code, "en");
    }

    // --- parse_messages tests ---

    #[test]
    fn test_parse_messages_all_roles() {
        let messages = vec![
            Message::system("be helpful"),
            Message::user("hi there"),
            Message::assistant("hello!"),
        ];
        let result = parse_messages(&messages);
        assert!(result.contains("system: be helpful"));
        assert!(result.contains("user: hi there"));
        assert!(result.contains("assistant: hello!"));
    }

    #[test]
    fn test_parse_messages_empty() {
        let result = parse_messages(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_messages_unknown_role() {
        let msg = Message {
            role: "custom_role".into(),
            content: "test".into(),
            name: None,
        };
        let result = parse_messages(&[msg]);
        // Unknown role is just skipped (debug logged)
        assert!(!result.contains("custom_role"));
    }

    // --- sanitize_for_cypher tests ---

    #[test]
    fn test_sanitize_simple() {
        let result = sanitize_for_cypher("hello world");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_sanitize_special_chars() {
        let result = sanitize_for_cypher("hello's world");
        assert!(result.contains("_apostrophe_"));
        assert!(!result.contains("'"));
    }

    #[test]
    fn test_sanitize_multiple_specials() {
        let result = sanitize_for_cypher("a&b=c");
        assert!(result.contains("_ampersand_"));
        assert!(result.contains("_equals_"));
    }

    #[test]
    fn test_sanitize_ellipsis() {
        let result = sanitize_for_cypher("wait...");
        assert!(
            result.contains("ellipsis"),
            "Should contain ellipsis replacement, got: {}",
            result
        );
    }

    #[test]
    fn test_sanitize_empty() {
        let result = sanitize_for_cypher("");
        assert!(result.is_empty());
    }

    // --- filter_messages_by_role tests ---

    #[test]
    fn test_filter_by_role() {
        let messages = vec![
            Message::user("q1"),
            Message::assistant("a1"),
            Message::user("q2"),
            Message::system("sys"),
        ];

        let user_msgs = filter_messages_by_role(&messages, "user");
        assert_eq!(user_msgs.len(), 2);
        assert_eq!(user_msgs[0].content, "q1");
        assert_eq!(user_msgs[1].content, "q2");

        let sys_msgs = filter_messages_by_role(&messages, "system");
        assert_eq!(sys_msgs.len(), 1);
    }

    #[test]
    fn test_filter_by_role_no_match() {
        let messages = vec![Message::user("hi")];
        let result = filter_messages_by_role(&messages, "assistant");
        assert!(result.is_empty());
    }

    // --- filter_messages_by_roles tests ---

    #[test]
    fn test_filter_by_roles() {
        let messages = vec![
            Message::user("q1"),
            Message::assistant("a1"),
            Message::system("sys"),
        ];

        let result = filter_messages_by_roles(&messages, &["user", "assistant"]);
        assert_eq!(result.len(), 2);

        let result2 = filter_messages_by_roles(&messages, &["system"]);
        assert_eq!(result2.len(), 1);
    }

    #[test]
    fn test_filter_by_roles_empty_roles() {
        let messages = vec![Message::user("hi")];
        let result = filter_messages_by_roles(&messages, &[]);
        assert!(result.is_empty());
    }

    // --- LanguageInfo tests ---

    #[test]
    fn test_language_info_serialization() {
        let info = LanguageInfo {
            language_code: "en".into(),
            language_name: "English".into(),
            confidence: 0.95,
        };
        let json = serde_json::to_string(&info).unwrap();
        let restored: LanguageInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.language_code, "en");
        assert_eq!(restored.confidence, 0.95);
    }
}
