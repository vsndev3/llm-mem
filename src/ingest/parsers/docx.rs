use std::io::Cursor;

use crate::ingest::document_tree::{DocumentMeta, DocumentNode};

pub fn parse_docx_bytes(data: &[u8], byte_size: u64) -> Result<(DocumentNode, DocumentMeta), String> {
    let cursor = Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| format!("Failed to open DOCX (ZIP): {}", e))?;

    let doc_xml = {
        let mut file = archive.by_name("word/document.xml")
            .map_err(|_| "DOCX missing word/document.xml".to_string())?;
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut file, &mut buf)
            .map_err(|e| format!("Failed to read document.xml: {}", e))?;
        buf
    };

    let mut reader = quick_xml::Reader::from_str(&doc_xml);
    reader.config_mut().trim_text(true);

    let mut paragraphs = Vec::new();
    let mut current_text = String::new();
    let mut in_paragraph = false;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(quick_xml::events::Event::Start(ref e)) => {
                if e.name().as_ref() == b"w:p" {
                    in_paragraph = true;
                    current_text.clear();
                }
            }
            Ok(quick_xml::events::Event::Text(ref e)) => {
                if in_paragraph
                    && let Ok(t) = e.unescape() {
                        current_text.push_str(&t);
                    }
            }
            Ok(quick_xml::events::Event::End(ref e)) => {
                if e.name().as_ref() == b"w:p" {
                    in_paragraph = false;
                    let trimmed = current_text.trim().to_string();
                    if !trimmed.is_empty() {
                        paragraphs.push(DocumentNode::Paragraph {
                            text: trimmed,
                            id: None,
                        });
                    }
                    current_text.clear();
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(e) => {
                return Err(format!("XML parse error in DOCX: {}", e));
            }
            _ => {}
        }
        buf.clear();
    }

    if paragraphs.is_empty() {
        return Err("DOCX contains no extractable text".into());
    }

    let meta = DocumentMeta::new(
        "word",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        byte_size,
    );

    Ok((DocumentNode::Document {
        children: paragraphs,
        meta: meta.clone(),
    }, meta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn minimal_docx() -> Vec<u8> {
        let document_xml = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:r>
        <w:t>Hello World</w:t>
      </w:r>
    </w:p>
    <w:p>
      <w:r>
        <w:t>This is a test document.</w:t>
      </w:r>
    </w:p>
  </w:body>
</w:document>"#;

        let mut buf = Vec::new();
        {
            let mut zip_writer = zip::ZipWriter::new(Cursor::new(&mut buf));
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);

            zip_writer.start_file("[Content_Types].xml", options).unwrap();
            zip_writer.write_all(b"<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"xml\" ContentType=\"application/xml\"/></Types>").unwrap();

            zip_writer.start_file("word/document.xml", options).unwrap();
            zip_writer.write_all(document_xml.as_bytes()).unwrap();

            zip_writer.finish().unwrap();
        }

        buf
    }

    #[test]
    fn test_docx_parses_text() {
        let data = minimal_docx();
        let result = parse_docx_bytes(&data, data.len() as u64).unwrap();
        if let DocumentNode::Document { children, .. } = result.0 {
            assert_eq!(children.len(), 2);
            assert!(children.iter().any(|c| {
                matches!(c, DocumentNode::Paragraph { text, .. } if text.contains("Hello World"))
            }));
        } else {
            panic!("Expected Document");
        }
    }

    #[test]
    fn test_docx_empty_rejected() {
        let mut buf = Vec::new();
        {
            let mut zip_writer = zip::ZipWriter::new(Cursor::new(&mut buf));
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip_writer.start_file("word/document.xml", options).unwrap();
            zip_writer.write_all(b"<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body></w:body></w:document>").unwrap();
            zip_writer.finish().unwrap();
        }

        let result = parse_docx_bytes(&buf, buf.len() as u64);
        assert!(result.is_err());
    }
}
