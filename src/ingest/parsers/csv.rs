use crate::ingest::document_tree::{DocumentMeta, DocumentNode};

pub fn parse_csv(content: &str, byte_size: u64) -> Result<(DocumentNode, DocumentMeta), String> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(content.as_bytes());

    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| format!("CSV header error: {}", e))?
        .iter()
        .map(|h| h.to_string())
        .collect();

    let mut rows = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| format!("CSV record error: {}", e))?;
        let row: Vec<String> = (0..headers.len())
            .map(|i| record.get(i).unwrap_or("").to_string())
            .collect();
        if !row.is_empty() {
            rows.push(row);
        }
    }

    if headers.is_empty() && rows.is_empty() {
        return Err("CSV is empty".into());
    }

    let meta = DocumentMeta::new("csv", "text/csv", byte_size);

    let table = DocumentNode::Table {
        headers,
        rows,
        caption: None,
        id: None,
    };

    Ok((table, meta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::document_tree::DocumentNode;

    #[test]
    fn test_csv_basic() {
        let csv = "Name,Age\nAlice,30\nBob,25\n";
        let (doc, meta) = parse_csv(csv, csv.len() as u64).unwrap();
        assert_eq!(meta.format, "csv");
        if let DocumentNode::Table { headers, rows, .. } = doc {
            assert_eq!(headers, vec!["Name", "Age"]);
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0], vec!["Alice", "30"]);
        } else {
            panic!("Expected Table");
        }
    }

    #[test]
    fn test_csv_no_rows() {
        let csv = "Name,Age\n";
        let (doc, _meta) = parse_csv(csv, csv.len() as u64).unwrap();
        if let DocumentNode::Table { rows, .. } = doc {
            assert!(rows.is_empty());
        } else {
            panic!("Expected Table");
        }
    }

    #[test]
    fn test_csv_empty() {
        let csv = "\n";
        assert!(parse_csv(csv, csv.len() as u64).is_err());
    }

    #[test]
    fn test_csv_extra_columns() {
        let csv = "Name\nAlice,30,extra\n";
        let (doc, _meta) = parse_csv(csv, csv.len() as u64).unwrap();
        if let DocumentNode::Table { rows, .. } = doc {
            assert_eq!(rows[0], vec!["Alice"]);
        } else {
            panic!("Expected Table");
        }
    }
}
