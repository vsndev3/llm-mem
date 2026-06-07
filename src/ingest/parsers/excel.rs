use std::io::Cursor;

use calamine::{Reader, open_workbook_auto_from_rs};

use crate::ingest::document_tree::{DocumentMeta, DocumentNode};

pub fn parse_excel(
    _content: &str,
    _byte_size: u64,
    _file_name: Option<&str>,
) -> Result<(DocumentNode, DocumentMeta), String> {
    Err(
        "Excel (.xlsx/.xls) is a binary format. Binary file ingestion is planned for Phase 3. \
         For now, export your data to CSV and use the 'csv' format instead."
            .into(),
    )
}

pub fn parse_excel_bytes(
    data: &[u8],
    byte_size: u64,
) -> Result<(DocumentNode, DocumentMeta), String> {
    let cursor = Cursor::new(data);
    let mut workbook = open_workbook_auto_from_rs(cursor)
        .map_err(|e| format!("Failed to open Excel workbook: {}", e))?;

    let sheet_names = workbook.sheet_names().to_vec();

    let mut tables = Vec::new();

    for sheet_name in &sheet_names {
        let range = workbook
            .worksheet_range(sheet_name)
            .map_err(|e| format!("Failed to read sheet '{}': {}", sheet_name, e))?;

        let rows: Vec<Vec<String>> = range
            .rows()
            .map(|row| row.iter().map(|cell| cell.to_string()).collect())
            .collect();

        if rows.is_empty() {
            continue;
        }

        let (headers, data_rows) = if rows.len() >= 2 {
            (rows[0].clone(), rows[1..].to_vec())
        } else {
            let auto_headers: Vec<String> = (0..rows[0].len())
                .map(|i| format!("Column_{}", i + 1))
                .collect();
            (auto_headers, rows.clone())
        };

        tables.push(DocumentNode::Section {
            title: sheet_name.clone(),
            level: 1,
            children: vec![DocumentNode::Table {
                headers,
                rows: data_rows,
                caption: None,
                id: None,
            }],
            id: None,
        });
    }

    if tables.is_empty() {
        return Err("Excel workbook contains no data".into());
    }

    let meta = DocumentMeta::new(
        "excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        byte_size,
    );

    Ok((
        DocumentNode::Document {
            children: tables,
            meta: meta.clone(),
        },
        meta,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_excel_binary_not_yet_supported() {
        let result = parse_excel("data", 4, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("binary format"));
    }
}
