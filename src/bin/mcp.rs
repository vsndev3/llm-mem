use anyhow::{Context, anyhow};
use clap::Parser;
use llm_mem::{MemoryMcpService, config::Config};
use rmcp::{ServiceExt, transport::stdio};
use rolling_file::{BasicRollingFileAppender, RollingConditionBasic};
use std::path::PathBuf;
use tracing::{error, info};

#[derive(Parser)]
#[command(name = "llm-mem-mcp")]
#[command(about = "MCP server for LLM memory management")]
#[command(version)]
struct Cli {
    /// Path to the configuration file. If not provided, searches:
    /// ./config.toml, ~/.config/llm-mem/config.toml, /etc/llm-mem/config.toml
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Agent identifier for memory operations
    #[arg(long)]
    agent: Option<String>,

    /// Proxy URL for model downloads (overrides HTTPS_PROXY env var)
    /// Format: http://host:port or http://user:pass@host:port
    #[arg(long)]
    proxy: Option<String>,

    /// Directory for memory bank database files (overrides config banks_dir)
    #[arg(long)]
    banks_dir: Option<PathBuf>,

    /// Disable grammar-constrained sampling for local LLM structured output
    /// (grammar is enabled by default via config)
    #[arg(long)]
    no_grammar: bool,

    /// Disable structured output mode for API-based LLM (OpenAI, etc.)
    /// (structured output is enabled by default via config)
    #[arg(long)]
    no_structured_output: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Load configuration first to determine log directory
    let mut config = if let Some(config_path) = &cli.config {
        Config::load(config_path)
            .with_context(|| format!("Failed to load config from {:?}", config_path))?
    } else {
        Config::default()
    };

    // Apply CLI overrides to config
    if let Some(proxy) = &cli.proxy {
        config.local.proxy_url = Some(proxy.clone());
    }
    if let Some(banks_dir) = &cli.banks_dir {
        config.vector_store.banks_dir = banks_dir.display().to_string();
    }
    if cli.no_grammar {
        config.local.use_grammar = false;
    }
    if cli.no_structured_output {
        config.api_llm.use_structured_output = false;
    }

    // Ensure log directory exists
    let log_dir = if config.logging.log_directory.is_empty() {
        PathBuf::from("llm-mem-data/logs")
    } else {
        PathBuf::from(&config.logging.log_directory)
    };

    if !log_dir.exists() {
        std::fs::create_dir_all(&log_dir)
            .with_context(|| format!("Failed to create log directory: {:?}", log_dir))?;
    }

    // Setup logging to file with size-based rotation
    let log_file_path = log_dir.join("llm-mem-mcp.log");
    let max_size = config.logging.max_size_mb * 1024 * 1024;

    let file_appender = BasicRollingFileAppender::new(
        &log_file_path,
        RollingConditionBasic::new().max_size(max_size),
        config.logging.max_files,
    )
    .with_context(|| format!("Failed to initialize logging to {:?}", log_file_path))?;

    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    let filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy();

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(non_blocking)
        .with_ansi(false)
        .init();

    info!("Starting LLM Memory MCP Server");
    info!("Configuration loaded from: {:?}", cli.config);
    info!("Logging to directory: {:?}", log_dir);

    // Initialize service with the loaded config
    let service = MemoryMcpService::with_config_and_agent(config, cli.agent.clone())
        .await
        .map_err(|e| anyhow!("Failed to initialize memory service: {}", e))?;

    let running_service = service
        .serve(stdio())
        .await
        .map_err(|e| anyhow!("Failed to start MCP server: {}", e))?;

    info!("MCP server initialized successfully");

    match running_service.waiting().await {
        Ok(reason) => info!("Server shutdown: {:?}", reason),
        Err(e) => error!("Server error: {:?}", e),
    }

    Ok(())
}
