pub mod code;
pub mod csv;
pub mod docx;
pub mod excel;
pub mod image;
pub mod json;
pub mod markdown;
pub mod pdf;
pub mod text;
pub mod toml;
pub mod yaml;

use crate::ingest::document_tree::DocumentNode;
use crate::ingest::format_detect::InputFormat;

use super::DocumentMeta;

pub fn parse(
    content: &str,
    format: InputFormat,
    file_name: Option<&str>,
) -> Result<(DocumentNode, DocumentMeta), String> {
    let byte_size = content.len() as u64;

    match format {
        InputFormat::Markdown => markdown::parse_markdown(content, byte_size),
        InputFormat::Json => json::parse_json(content, byte_size),
        InputFormat::Yaml => yaml::parse_yaml(content, byte_size),
        InputFormat::Toml => toml::parse_toml(content, byte_size),
        InputFormat::PlainText => text::parse_text(content, byte_size),
        InputFormat::Csv => csv::parse_csv(content, byte_size),
        InputFormat::Excel => excel::parse_excel(content, byte_size, file_name),
        InputFormat::Rust => code::parse_code(content, byte_size, "rust"),
        InputFormat::Python => code::parse_code(content, byte_size, "python"),
        InputFormat::JavaScript => code::parse_code(content, byte_size, "javascript"),
        InputFormat::TypeScript => code::parse_code(content, byte_size, "typescript"),
        _ => {
            let _meta = DocumentMeta::new(format.name(), format.mime(), byte_size);
            let file_context = file_name.unwrap_or("unknown");
            Err(format!(
                "Format '{}' is not yet supported for parsing. File: {}. Content preview: {}",
                format.name(),
                file_context,
                &content[..content.len().min(200)]
            ))
        }
    }
}

pub fn parse_binary(
    data: &[u8],
    format: InputFormat,
) -> Result<(DocumentNode, DocumentMeta), String> {
    let byte_size = data.len() as u64;

    match format {
        InputFormat::Pdf => pdf::parse_pdf_bytes(data, byte_size),
        InputFormat::Word => docx::parse_docx_bytes(data, byte_size),
        InputFormat::Excel => excel::parse_excel_bytes(data, byte_size),
        InputFormat::ImagePng | InputFormat::ImageJpeg | InputFormat::ImageGif | InputFormat::ImageWebp => {
            image::parse_image_bytes(data, byte_size)
        }
        _ => Err(format!(
            "Binary parsing not supported for format: {}", format.name()
        )),
    }
}
