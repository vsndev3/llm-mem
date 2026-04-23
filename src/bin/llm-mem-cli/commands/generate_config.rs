use crate::OutputFormat;
use anyhow::Result;
use std::path::Path;

/// Handle the generate-config command
pub async fn handle_generate_config(
    output_path: &Path,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Generate the full config with all defaults using the template method
    let config_toml = llm_mem::config::Config::template();

    // Write to file
    std::fs::write(output_path, &config_toml)?;

    // Print confirmation
    let response = match format {
        OutputFormat::Table => {
            format!(
                "Config written to: {}\n\nTo use, copy to config.toml or specify with --config",
                output_path.display()
            )
        }
        OutputFormat::Json => serde_json::to_string_pretty(&serde_json::json!({
            "status": "success",
            "message": "Config written to file",
            "path": output_path.to_string_lossy().to_string(),
            "note": "Copy to config.toml or specify with --config"
        }))
        .unwrap_or_else(|_| {
            serde_json::json!({
                "status": "success",
                "path": output_path.to_string_lossy().to_string()
            })
            .to_string()
        }),
        OutputFormat::Jsonl => serde_json::to_string_pretty(&serde_json::json!({
            "status": "success",
            "path": output_path.to_string_lossy().to_string()
        }))
        .unwrap_or_else(|_| {
            serde_json::json!({
                "status": "success",
                "path": output_path.to_string_lossy().to_string()
            })
            .to_string()
        }),
        OutputFormat::Detail | OutputFormat::Csv => {
            format!("Config file: {}\nStatus: success", output_path.display())
        }
    };

    println!("{}", response);
    Ok(())
}
