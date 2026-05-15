/// Strip XML-style tags (e.g., <think>...</think>) from text
pub fn strip_llm_tags(text: &str, tags: &[String]) -> String {
    let mut result = text.to_string();

    for tag in tags {
        loop {
            let open_tag = format!("<{}", tag);
            let close_tag = format!("</{}>", tag);

            if let Some(start) = result.find(&open_tag)
                && let Some(tag_end) = result[start..].find('>')
            {
                let content_start = start + tag_end + 1;
                if let Some(close_pos) = result[content_start..].find(&close_tag) {
                    let before = &result[..start];
                    let after = &result[content_start + close_pos + close_tag.len()..];
                    result = format!("{}{}", before, after);
                    continue;
                } else {
                    result = result[..start].to_string();
                    continue;
                }
            }
            break;
        }
    }

    result.trim().to_string()
}
