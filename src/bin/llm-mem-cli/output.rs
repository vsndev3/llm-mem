use crate::OutputFormat;
use std::fmt::Write;

/// Format a MemoryOperationResponse into a string in the specified format
pub fn format_response(
    response: &llm_mem::operations::MemoryOperationResponse,
    format: OutputFormat,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut buf = String::new();
    match format {
        OutputFormat::Table => format_table(response, &mut buf)?,
        OutputFormat::Detail => format_detail(response, &mut buf)?,
        OutputFormat::Json => format_json(response, true, &mut buf)?,
        OutputFormat::Jsonl => format_jsonl(response, true, &mut buf)?,
        OutputFormat::Csv => format_csv(response, &mut buf)?,
    }
    Ok(buf)
}

/// Print a MemoryOperationResponse in the specified format
pub fn print_response(
    response: &llm_mem::operations::MemoryOperationResponse,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let output = format_response(response, format)?;
    print!("{}", output);
    Ok(())
}

/// Display output with pagination in interactive mode.
/// Shows one page at a time; press Enter to continue or 'q' to quit.
pub fn paginate_output(text: &str) {
    if text.is_empty() {
        return;
    }
    let term_height = crossterm::terminal::size()
        .map(|(_, h)| h as usize)
        .unwrap_or(24);
    let page_size = term_height.saturating_sub(2);

    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= page_size {
        print!("{}", text);
        if !text.ends_with('\n') {
            println!();
        }
        return;
    }

    let total = lines.len();
    let mut idx = 0;
    while idx < total {
        let end = (idx + page_size).min(total);
        for line in &lines[idx..end] {
            println!("{}", line);
        }
        idx = end;

        if idx < total {
            eprint!(
                "-- More ({}/{} lines, Enter=next page, q=quit) --",
                idx, total
            );
            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_err() {
                break;
            }
            // Clear the prompt line
            eprint!("\r\x1b[2K");
            if input.trim().eq_ignore_ascii_case("q") {
                break;
            }
        }
    }
}

/// Format response as a table into the given buffer
fn format_table(
    response: &llm_mem::operations::MemoryOperationResponse,
    buf: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    if response.success {
        writeln!(buf, "{}", response.message)?;

        if let Some(data) = &response.data {
            // Handle different data types for table display
            if let serde_json::Value::Array(array) = data {
                if array.is_empty() {
                    writeln!(buf, "(No data)")?;
                } else {
                    // Try to display as a simple table if it's an array of objects
                    if array.iter().all(|v| v.is_object()) {
                        format_array_of_objects_as_table(array, buf)?;
                    } else {
                        // Fall back to JSON display for complex data
                        writeln!(buf, "{}", serde_json::to_string_pretty(data)?)?;
                    }
                }
            } else if let serde_json::Value::Object(obj) = data {
                // Simple key-value display for objects
                for (key, value) in obj {
                    writeln!(buf, "{}: {}", key, value)?;
                }
            } else {
                // Primitive values or other types
                writeln!(buf, "{}", data)?;
            }
        } else if let Some(error) = &response.error {
            writeln!(buf, "Error: {}", error)?;
        }
    } else {
        writeln!(buf, "Error: {}", response.message)?;
        if let Some(error) = &response.error {
            writeln!(buf, "Details: {}", error)?;
        }
    }

    Ok(())
}

/// Format response in detail format into the given buffer
fn format_detail(
    response: &llm_mem::operations::MemoryOperationResponse,
    buf: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    if response.success {
        writeln!(buf, "{}", response.message)?;
        writeln!(buf)?;

        if let Some(data) = &response.data {
            writeln!(buf, "{}", serde_json::to_string_pretty(data)?)?;
        } else if let Some(error) = &response.error {
            writeln!(buf, "Error: {}", error)?;
        }
    } else {
        writeln!(buf, "Error: {}", response.message)?;
        if let Some(error) = &response.error {
            writeln!(buf, "Details: {}", error)?;
        }
    }

    Ok(())
}

/// Format response as JSON into the given buffer
fn format_json(
    response: &llm_mem::operations::MemoryOperationResponse,
    pretty: bool,
    buf: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    if pretty {
        writeln!(buf, "{}", serde_json::to_string_pretty(&response)?)?;
    } else {
        writeln!(buf, "{}", serde_json::to_string(&response)?)?;
    }
    Ok(())
}

/// Format response as JSONL into the given buffer
fn format_jsonl(
    response: &llm_mem::operations::MemoryOperationResponse,
    pretty: bool,
    buf: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    // For JSONL, we typically want to output each item in an array separately
    if let Some(data) = &response.data {
        if let serde_json::Value::Array(array) = data {
            for item in array {
                if pretty {
                    writeln!(buf, "{}", serde_json::to_string_pretty(item)?)?;
                } else {
                    writeln!(buf, "{}", serde_json::to_string(item)?)?;
                }
            }
        } else {
            // If not an array, just output the whole response as one JSON line
            let json_output = if pretty {
                serde_json::to_string_pretty(data)?
            } else {
                serde_json::to_string(data)?
            };
            writeln!(buf, "{}", json_output)?;
        }
    } else {
        // No data, output the response itself
        let json_output = if pretty {
            serde_json::to_string_pretty(response)?
        } else {
            serde_json::to_string(response)?
        };
        writeln!(buf, "{}", json_output)?;
    }
    Ok(())
}

/// Format response as CSV into the given buffer
fn format_csv(
    response: &llm_mem::operations::MemoryOperationResponse,
    buf: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    if response.success {
        writeln!(buf, "{}", response.message)?;

        if let Some(data) = &response.data {
            if let serde_json::Value::Array(array) = data {
                if !array.is_empty() && array.iter().all(|v| v.is_object()) {
                    format_array_of_objects_as_csv(array, buf)?;
                } else {
                    // Fall back to JSON display
                    writeln!(buf, "{}", serde_json::to_string_pretty(data)?)?;
                }
            } else {
                // For non-array data, just show the message
                writeln!(buf, "(Data not suitable for CSV display)")?;
            }
        }
    } else {
        writeln!(buf, "Error: {}", response.message)?;
        if let Some(error) = &response.error {
            writeln!(buf, "Details: {}", error)?;
        }
    }

    Ok(())
}

/// Helper function to collect all keys from an array of JSON objects, sorted.
fn collect_sorted_keys(array: &[serde_json::Value]) -> Vec<&str> {
    let mut all_keys = std::collections::HashSet::new();
    for obj in array {
        if let serde_json::Value::Object(map) = obj {
            for key in map.keys() {
                all_keys.insert(key.as_str());
            }
        }
    }
    let mut keys: Vec<_> = all_keys.into_iter().collect();
    keys.sort();
    keys
}

/// Helper function to format a serde_json::Value as a display string for tables.
fn format_value_for_display(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
        _ => "complex".to_string(),
    }
}

/// Helper function to format a serde_json::Value as a CSV cell string.
fn format_value_for_csv(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => {
            let escaped = s.replace('"', "\"\"");
            format!("\"{}\"", escaped)
        }
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "".to_string(),
        _ => "\"complex\"".to_string(),
    }
}

/// Helper function to format an array of objects as a table into the given buffer
fn format_array_of_objects_as_table(
    array: &[serde_json::Value],
    buf: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    if array.is_empty() {
        return Ok(());
    }

    let keys = collect_sorted_keys(array);

    // Print header
    for (i, key) in keys.iter().enumerate() {
        if i > 0 {
            write!(buf, " | ")?;
        }
        write!(buf, "{:<20}", key)?;
    }
    writeln!(buf)?;

    // Print separator
    for (i, _key) in keys.iter().enumerate() {
        if i > 0 {
            write!(buf, "-+-")?;
        }
        write!(buf, "{:-<20}", "")?;
    }
    writeln!(buf)?;

    // Print rows
    for obj in array {
        if let serde_json::Value::Object(map) = obj {
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    write!(buf, " | ")?;
                }
                if let Some(value) = map.get(*key) {
                    write!(buf, "{:<20}", format_value_for_display(value))?;
                } else {
                    write!(buf, "{:<20}", "")?;
                }
            }
            writeln!(buf)?;
        }
    }

    Ok(())
}

/// Helper function to format an array of objects as CSV into the given buffer
fn format_array_of_objects_as_csv(
    array: &[serde_json::Value],
    buf: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    if array.is_empty() {
        return Ok(());
    }

    let keys = collect_sorted_keys(array);

    // Print header
    for (i, key) in keys.iter().enumerate() {
        if i > 0 {
            write!(buf, ",")?;
        }
        write!(buf, "\"{}\"", key)?;
    }
    writeln!(buf)?;

    // Print rows
    for obj in array {
        if let serde_json::Value::Object(map) = obj {
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    write!(buf, ",")?;
                }
                if let Some(value) = map.get(*key) {
                    write!(buf, "{}", format_value_for_csv(value))?;
                } else {
                    write!(buf, "")?;
                }
            }
            writeln!(buf)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_mem::operations::MemoryOperationResponse;
    use serde_json::json;

    fn make_success_response(msg: &str, data: serde_json::Value) -> MemoryOperationResponse {
        MemoryOperationResponse::success_with_data(msg, data)
    }

    fn make_error_response(err: &str) -> MemoryOperationResponse {
        MemoryOperationResponse::error(err)
    }

    fn make_success_no_data(msg: &str) -> MemoryOperationResponse {
        MemoryOperationResponse::success(msg)
    }

    // --- collect_sorted_keys tests ---

    #[test]
    fn test_collect_sorted_keys_empty_array() {
        let array: Vec<serde_json::Value> = vec![];
        let keys = collect_sorted_keys(&array);
        assert!(keys.is_empty());
    }

    #[test]
    fn test_collect_sorted_keys_single_object() {
        let array = vec![json!({"name": "Alice", "age": 30})];
        let keys = collect_sorted_keys(&array);
        assert_eq!(keys, vec!["age", "name"]);
    }

    #[test]
    fn test_collect_sorted_keys_multiple_objects_different_keys() {
        let array = vec![
            json!({"name": "Alice", "age": 30}),
            json!({"name": "Bob", "email": "bob@example.com"}),
        ];
        let keys = collect_sorted_keys(&array);
        assert_eq!(keys, vec!["age", "email", "name"]);
    }

    #[test]
    fn test_collect_sorted_keys_non_object_items_skipped() {
        let array = vec![json!({"key": "value"}), json!("not an object"), json!(42)];
        let keys = collect_sorted_keys(&array);
        assert_eq!(keys, vec!["key"]);
    }

    // --- format_value_for_display tests ---

    #[test]
    fn test_format_value_string() {
        assert_eq!(format_value_for_display(&json!("hello")), "hello");
    }

    #[test]
    fn test_format_value_number() {
        assert_eq!(format_value_for_display(&json!(42)), "42");
        assert_eq!(format_value_for_display(&json!(3.14)), "3.14");
    }

    #[test]
    fn test_format_value_bool() {
        assert_eq!(format_value_for_display(&json!(true)), "true");
        assert_eq!(format_value_for_display(&json!(false)), "false");
    }

    #[test]
    fn test_format_value_null() {
        assert_eq!(format_value_for_display(&json!(null)), "null");
    }

    #[test]
    fn test_format_value_complex() {
        assert_eq!(format_value_for_display(&json!([1, 2, 3])), "complex");
        assert_eq!(
            format_value_for_display(&json!({"nested": true})),
            "complex"
        );
    }

    // --- format_value_for_csv tests ---

    #[test]
    fn test_csv_format_string() {
        assert_eq!(format_value_for_csv(&json!("hello")), "\"hello\"");
    }

    #[test]
    fn test_csv_format_string_with_quotes() {
        assert_eq!(
            format_value_for_csv(&json!("he said \"hi\"")),
            "\"he said \"\"hi\"\"\""
        );
    }

    #[test]
    fn test_csv_format_number() {
        assert_eq!(format_value_for_csv(&json!(42)), "42");
    }

    #[test]
    fn test_csv_format_bool() {
        assert_eq!(format_value_for_csv(&json!(true)), "true");
    }

    #[test]
    fn test_csv_format_null() {
        assert_eq!(format_value_for_csv(&json!(null)), "");
    }

    #[test]
    fn test_csv_format_complex() {
        assert_eq!(format_value_for_csv(&json!([1, 2])), "\"complex\"");
    }

    // --- print_response tests (output to stdout, verify no panic) ---

    #[test]
    fn test_print_response_table_success_with_array_data() {
        let response = make_success_response(
            "OK",
            json!([
                {"id": "1", "name": "test"},
                {"id": "2", "name": "test2"}
            ]),
        );
        let result = print_response(&response, OutputFormat::Table);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_response_table_success_with_object_data() {
        let response = make_success_response("OK", json!({"key": "value", "count": 42}));
        let result = print_response(&response, OutputFormat::Table);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_response_table_success_with_primitive_data() {
        let response = make_success_response("OK", json!("just a string"));
        let result = print_response(&response, OutputFormat::Table);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_response_table_success_empty_array() {
        let response = make_success_response("OK", json!([]));
        let result = print_response(&response, OutputFormat::Table);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_response_table_error_response() {
        let response = make_error_response("something went wrong");
        let result = print_response(&response, OutputFormat::Table);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_response_table_success_no_data() {
        let response = make_success_no_data("No data here");
        let result = print_response(&response, OutputFormat::Table);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_response_detail_success() {
        let response =
            make_success_response("Details", json!({"id": "abc", "content": "memory content"}));
        let result = print_response(&response, OutputFormat::Detail);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_response_detail_error() {
        let response = make_error_response("not found");
        let result = print_response(&response, OutputFormat::Detail);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_response_json_success() {
        let response = make_success_response("OK", json!({"data": true}));
        let result = print_response(&response, OutputFormat::Json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_response_jsonl_with_array() {
        let response = make_success_response(
            "OK",
            json!([
                {"id": 1}, {"id": 2}, {"id": 3}
            ]),
        );
        let result = print_response(&response, OutputFormat::Jsonl);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_response_jsonl_with_non_array() {
        let response = make_success_response("OK", json!({"single": true}));
        let result = print_response(&response, OutputFormat::Jsonl);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_response_jsonl_no_data() {
        let response = make_success_no_data("empty");
        let result = print_response(&response, OutputFormat::Jsonl);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_response_csv_success_array() {
        let response = make_success_response(
            "OK",
            json!([
                {"name": "alice", "score": 95},
                {"name": "bob", "score": 87}
            ]),
        );
        let result = print_response(&response, OutputFormat::Csv);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_response_csv_non_array_data() {
        let response = make_success_response("OK", json!({"key": "value"}));
        let result = print_response(&response, OutputFormat::Csv);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_response_csv_error() {
        let response = make_error_response("csv error");
        let result = print_response(&response, OutputFormat::Csv);
        assert!(result.is_ok());
    }

    // --- print_array_of_objects_as_table tests ---

    #[test]
    fn test_table_empty_array() {
        let mut buf = String::new();
        let result = format_array_of_objects_as_table(&[], &mut buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_table_single_row() {
        let array = vec![json!({"id": "1", "value": "hello"})];
        let mut buf = String::new();
        let result = format_array_of_objects_as_table(&array, &mut buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_table_multiple_rows() {
        let array = vec![
            json!({"id": "1", "name": "Alice"}),
            json!({"id": "2", "name": "Bob"}),
            json!({"id": "3", "name": "Charlie"}),
        ];
        let mut buf = String::new();
        let result = format_array_of_objects_as_table(&array, &mut buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_table_heterogeneous_keys() {
        let array = vec![
            json!({"id": "1", "name": "Alice"}),
            json!({"id": "2", "email": "bob@test.com"}),
        ];
        let mut buf = String::new();
        let result = format_array_of_objects_as_table(&array, &mut buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_table_with_null_and_bool_values() {
        let array = vec![
            json!({"active": true, "name": null}),
            json!({"active": false, "name": "test"}),
        ];
        let mut buf = String::new();
        let result = format_array_of_objects_as_table(&array, &mut buf);
        assert!(result.is_ok());
    }

    // --- print_array_of_objects_as_csv tests ---

    #[test]
    fn test_csv_empty_array() {
        let mut buf = String::new();
        let result = format_array_of_objects_as_csv(&[], &mut buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_csv_single_row() {
        let array = vec![json!({"id": "1", "value": "hello"})];
        let mut buf = String::new();
        let result = format_array_of_objects_as_csv(&array, &mut buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_csv_with_special_chars() {
        let array = vec![json!({"name": "has \"quotes\"", "desc": "comma, here"})];
        let mut buf = String::new();
        let result = format_array_of_objects_as_csv(&array, &mut buf);
        assert!(result.is_ok());
    }

    // --- print_json tests ---

    #[test]
    fn test_print_json_compact() {
        let response = make_success_response("OK", json!({"key": "val"}));
        let mut buf = String::new();
        let result = format_json(&response, false, &mut buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_json_pretty() {
        let response = make_success_response("OK", json!({"key": "val"}));
        let mut buf = String::new();
        let result = format_json(&response, true, &mut buf);
        assert!(result.is_ok());
    }
}
