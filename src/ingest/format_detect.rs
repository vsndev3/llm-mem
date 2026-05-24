#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    Markdown,
    Json,
    Yaml,
    Toml,
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Cpp,
    Pdf,
    Excel,
    Word,
    Csv,
    ImagePng,
    ImageJpeg,
    ImageGif,
    ImageWebp,
    PlainText,
    Unknown,
}

impl InputFormat {
    pub fn name(&self) -> &'static str {
        match self {
            InputFormat::Markdown => "markdown",
            InputFormat::Json => "json",
            InputFormat::Yaml => "yaml",
            InputFormat::Toml => "toml",
            InputFormat::Rust => "rust",
            InputFormat::Python => "python",
            InputFormat::JavaScript => "javascript",
            InputFormat::TypeScript => "typescript",
            InputFormat::Go => "go",
            InputFormat::Cpp => "cpp",
            InputFormat::Pdf => "pdf",
            InputFormat::Excel => "excel",
            InputFormat::Word => "word",
            InputFormat::Csv => "csv",
            InputFormat::ImagePng => "png",
            InputFormat::ImageJpeg => "jpeg",
            InputFormat::ImageGif => "gif",
            InputFormat::ImageWebp => "webp",
            InputFormat::PlainText => "text",
            InputFormat::Unknown => "unknown",
        }
    }

    pub fn mime(&self) -> &'static str {
        match self {
            InputFormat::Markdown => "text/markdown",
            InputFormat::Json => "application/json",
            InputFormat::Yaml => "text/yaml",
            InputFormat::Toml => "text/toml",
            InputFormat::Rust => "text/x-rust",
            InputFormat::Python => "text/x-python",
            InputFormat::JavaScript => "text/javascript",
            InputFormat::TypeScript => "text/typescript",
            InputFormat::Go => "text/x-go",
            InputFormat::Cpp => "text/x-c++",
            InputFormat::Pdf => "application/pdf",
            InputFormat::Excel => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            InputFormat::Word => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            InputFormat::Csv => "text/csv",
            InputFormat::ImagePng => "image/png",
            InputFormat::ImageJpeg => "image/jpeg",
            InputFormat::ImageGif => "image/gif",
            InputFormat::ImageWebp => "image/webp",
            InputFormat::PlainText => "text/plain",
            InputFormat::Unknown => "application/octet-stream",
        }
    }

    pub fn category(&self) -> &'static str {
        match self {
            InputFormat::Markdown
            | InputFormat::Json
            | InputFormat::Yaml
            | InputFormat::Toml
            | InputFormat::Rust
            | InputFormat::Python
            | InputFormat::JavaScript
            | InputFormat::TypeScript
            | InputFormat::Go
            | InputFormat::Cpp
            | InputFormat::PlainText
            | InputFormat::Csv => "text",

            InputFormat::Pdf | InputFormat::Excel | InputFormat::Word => "document",

            InputFormat::ImagePng
            | InputFormat::ImageJpeg
            | InputFormat::ImageGif
            | InputFormat::ImageWebp => "image",

            InputFormat::Unknown => "binary",
        }
    }

    pub fn is_image(&self) -> bool {
        matches!(
            self,
            InputFormat::ImagePng
                | InputFormat::ImageJpeg
                | InputFormat::ImageGif
                | InputFormat::ImageWebp
        )
    }
}

pub fn detect_format(content: &str, file_name: Option<&str>, format_hint: Option<&str>) -> InputFormat {
    if let Some(hint) = format_hint {
        return format_from_hint(hint);
    }

    if let Some(name) = file_name
        && let Some(fmt) = format_from_extension(name) {
            return fmt;
        }

    let bytes = content.as_bytes();
    let mime = tree_magic_mini::from_u8(bytes);

    mime_to_format(mime)
}

fn format_from_hint(hint: &str) -> InputFormat {
    match hint.to_lowercase().as_str() {
        "markdown" | "md" => InputFormat::Markdown,
        "json" => InputFormat::Json,
        "yaml" | "yml" => InputFormat::Yaml,
        "toml" => InputFormat::Toml,
        "rust" | "rs" => InputFormat::Rust,
        "python" | "py" => InputFormat::Python,
        "javascript" | "js" => InputFormat::JavaScript,
        "typescript" | "ts" => InputFormat::TypeScript,
        "go" => InputFormat::Go,
        "cpp" | "c++" | "cxx" | "hpp" => InputFormat::Cpp,
        "pdf" => InputFormat::Pdf,
        "excel" | "xlsx" | "xls" => InputFormat::Excel,
        "word" | "docx" | "doc" => InputFormat::Word,
        "csv" => InputFormat::Csv,
        "png" | "image/png" => InputFormat::ImagePng,
        "jpeg" | "jpg" | "image/jpeg" => InputFormat::ImageJpeg,
        "gif" | "image/gif" => InputFormat::ImageGif,
        "webp" | "image/webp" => InputFormat::ImageWebp,
        "text" | "plaintext" | "txt" => InputFormat::PlainText,
        _ => InputFormat::Unknown,
    }
}

fn format_from_extension(file_name: &str) -> Option<InputFormat> {
    let ext = file_name.rsplit('.').next()?.to_lowercase();
    match ext.as_str() {
        "md" | "markdown" => Some(InputFormat::Markdown),
        "json" => Some(InputFormat::Json),
        "yaml" | "yml" => Some(InputFormat::Yaml),
        "toml" => Some(InputFormat::Toml),
        "rs" => Some(InputFormat::Rust),
        "py" => Some(InputFormat::Python),
        "js" => Some(InputFormat::JavaScript),
        "ts" | "tsx" => Some(InputFormat::TypeScript),
        "go" => Some(InputFormat::Go),
        "cpp" | "cc" | "cxx" | "hpp" | "h" | "c" => Some(InputFormat::Cpp),
        "pdf" => Some(InputFormat::Pdf),
        "xlsx" | "xls" => Some(InputFormat::Excel),
        "docx" | "doc" => Some(InputFormat::Word),
        "csv" => Some(InputFormat::Csv),
        "png" => Some(InputFormat::ImagePng),
        "jpg" | "jpeg" => Some(InputFormat::ImageJpeg),
        "gif" => Some(InputFormat::ImageGif),
        "webp" => Some(InputFormat::ImageWebp),
        "txt" => Some(InputFormat::PlainText),
        _ => None,
    }
}

fn mime_to_format(mime: &str) -> InputFormat {
    match mime {
        "text/markdown" => InputFormat::Markdown,
        "application/json" => InputFormat::Json,
        "text/yaml" | "application/x-yaml" => InputFormat::Yaml,
        "application/pdf" => InputFormat::Pdf,
        "text/csv" => InputFormat::Csv,
        "image/png" => InputFormat::ImagePng,
        "image/jpeg" => InputFormat::ImageJpeg,
        "image/gif" => InputFormat::ImageGif,
        "image/webp" => InputFormat::ImageWebp,
        "text/plain" => InputFormat::PlainText,
        m if m.starts_with("text/") => {
            match m {
                "text/html" | "text/css" => InputFormat::Unknown,
                _ => InputFormat::PlainText,
            }
        }
        _ => InputFormat::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_markdown_by_extension() {
        assert_eq!(
            detect_format("# Hello", Some("readme.md"), None),
            InputFormat::Markdown
        );
    }

    #[test]
    fn test_detect_json_by_extension() {
        assert_eq!(
            detect_format("{}", Some("config.json"), None),
            InputFormat::Json
        );
    }

    #[test]
    fn test_format_hint_overrides_extension() {
        assert_eq!(
            detect_format("key: value", Some("data.json"), Some("yaml")),
            InputFormat::Yaml
        );
    }

    #[test]
    fn test_detect_text_by_mime() {
        assert_eq!(
            detect_format("Hello world", None, None),
            InputFormat::PlainText
        );
    }

    #[test]
    fn test_detect_pdf_by_mime() {
        let pdf = "%PDF-1.4";
        assert_eq!(
            detect_format(pdf, None, None),
            InputFormat::Pdf
        );
    }
}
