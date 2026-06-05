use crate::OutputFormat;
use llm_mem::{
    HealthCheckOptions, HealthReport, config::Config, diagnostics::run_health_check,
    format_report_table,
};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveScope {
    Both,
    EmbedOnly,
    LlmOnly,
}

#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    pub live: bool,
    pub scope: LiveScope,
    pub embed_timeout_secs: u64,
    pub llm_timeout_secs: u64,
    pub format: OutputFormat,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            live: false,
            scope: LiveScope::Both,
            embed_timeout_secs: 15,
            llm_timeout_secs: 30,
            format: OutputFormat::Table,
        }
    }
}

/// Run the health-check command.
///
/// Loads the config (caller is expected to have already done so via
/// `load_configuration`) and runs the diagnostic suite. Static checks are
/// always run; live checks (a real LLM ping and a real embed ping) only run
/// when `cfg.live` is true.
pub async fn handle_health_check(
    config: &Config,
    cfg: HealthCheckConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let opts = HealthCheckOptions {
        live: cfg.live,
        embed_timeout: Duration::from_secs(cfg.embed_timeout_secs),
        llm_timeout: Duration::from_secs(cfg.llm_timeout_secs),
        embed_only: matches!(cfg.scope, LiveScope::EmbedOnly),
        llm_only: matches!(cfg.scope, LiveScope::LlmOnly),
    };

    let report = run_health_check(config, &opts).await?;
    print_report(&report, cfg.format);
    if report.healthy {
        Ok(())
    } else {
        // Exit code 2 so scripts can distinguish "unhealthy" from
        // "could not even run the check" (which exits 1).
        std::process::exit(2);
    }
}

fn print_report(report: &HealthReport, format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(report).unwrap());
        }
        OutputFormat::Jsonl => {
            // One JSON object per line: the report itself, then each check.
            println!("{}", serde_json::to_string(report).unwrap());
            for c in &report.checks {
                println!("{}", serde_json::to_string(c).unwrap());
            }
        }
        OutputFormat::Csv => {
            println!("name,category,status,duration_ms,detail");
            for c in &report.checks {
                let detail = c.detail.replace('"', "\"\"");
                println!(
                    "{},{},{:?},{},\"{}\"",
                    c.name, c.category, c.status, c.duration_ms, detail
                );
            }
        }
        _ => {
            println!("{}", format_report_table(report));
        }
    }
}

