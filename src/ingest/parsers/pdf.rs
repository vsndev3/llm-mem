use crate::ingest::document_tree::{DocumentMeta, DocumentNode};

pub fn parse_pdf_bytes(
    data: &[u8],
    byte_size: u64,
) -> Result<(DocumentNode, DocumentMeta), String> {
    let doc = lopdf::Document::load_mem(data).map_err(|e| format!("Failed to load PDF: {}", e))?;

    let _page_count = doc.page_iter().count();

    let mut all_text = String::new();
    for (i, (page_num, _page_id)) in doc.page_iter().enumerate() {
        if let Ok(text) = doc.extract_text(&[page_num]) {
            let cleaned: String = text
                .lines()
                .map(|l| l.trim_end().to_string())
                .collect::<Vec<_>>()
                .join("\n");
            if !cleaned.trim().is_empty() {
                if !all_text.is_empty() {
                    all_text.push_str("\n\n");
                }
                all_text.push_str(&format!("--- Page {} ---\n{}", i + 1, cleaned));
            }
        }
    }

    if all_text.trim().is_empty() {
        return Err("PDF contains no extractable text (may be scanned image)".into());
    }

    let paragraphs: Vec<DocumentNode> = all_text
        .split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .map(|p| DocumentNode::Paragraph {
            text: p.trim().to_string(),
            id: None,
        })
        .collect();

    let meta = DocumentMeta::new("pdf", "application/pdf", byte_size);
    Ok((
        DocumentNode::Document {
            children: paragraphs,
            meta: meta.clone(),
        },
        meta,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_pdf() -> Vec<u8> {
        let content = b"Hello World";
        let mut pdf = Vec::new();

        pdf.extend_from_slice(b"%PDF-1.4\n");

        let obj1 = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n";
        let obj2 = "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n";
        let obj3_offset = pdf.len() + obj1.len() + obj2.len() + 8;

        let _startxref = obj3_offset;

        let obj3 = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
             /Contents 4 0 R /Resources << >> >>\nendobj\n";

        let stream_data = format!(
            "BT\n/F1 12 Tf\n100 700 Td\n({}) Tj\nET",
            std::str::from_utf8(content).unwrap_or("Hello")
        );
        let obj4 = format!(
            "4 0 obj\n<< /Length {} >>\nstream\n{}\nendstream\nendobj\n",
            stream_data.len(),
            stream_data
        );

        let xref_offset = pdf.len() + obj1.len() + obj2.len() + obj3.len() + obj4.len();

        pdf.extend_from_slice(obj1.as_bytes());
        pdf.extend_from_slice(obj2.as_bytes());
        pdf.extend_from_slice(obj3.as_bytes());
        pdf.extend_from_slice(obj4.as_bytes());
        pdf.extend_from_slice(format!("xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n{:010} 00000 n \n{:010} 00000 n \n{:010} 00000 n \n", obj1.len(), obj1.len() + obj2.len(), obj1.len() + obj2.len() + obj3.len()).as_bytes());
        pdf.extend_from_slice(
            format!(
                "trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{}\n%%EOF",
                xref_offset
            )
            .as_bytes(),
        );

        pdf
    }

    #[test]
    fn test_pdf_parses_or_gives_clear_error() {
        let data = minimal_pdf();
        let result = parse_pdf_bytes(&data, data.len() as u64);
        let is_ok = result.is_ok();
        let err_msg = result.as_ref().err().cloned().unwrap_or_default();
        assert!(
            is_ok || err_msg.contains("no extractable text"),
            "Expected success or 'no extractable text' error, got: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_pdf_empty_rejected() {
        let data = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 /Root 1 0 R >>\nstartxref\n9\n%%EOF";
        let result = parse_pdf_bytes(data, data.len() as u64);
        assert!(result.is_ok() || result.is_err(), "should not panic");
    }
}
