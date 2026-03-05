use anyhow::{Context, anyhow};
use clap::Parser;
use llm_mem::{MemoryMcpService, config::Config};
use rmcp::{ServiceExt, transport::stdio};
use rolling_file::{BasicRollingFileAppender, RollingConditionBasic};
use std::path::PathBuf;
use tracing::{error, info};
use tokio::signal;

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

    /// Request format mode for API-based LLM: "auto" (default), "rig", or "raw"
    /// - auto: tries rig-core first, falls back to raw HTTP on 422 errors
    /// - rig: always uses rig-core completion API (may cause 422 errors with strict backends)
    /// - raw: always uses raw HTTP requests with plain strings (bypasses rig-core)
    #[arg(long)]
    request_format: Option<String>,

    /// Disable model caching in ~/.cache/llm-mem/models/
    /// (caching is enabled by default)
    #[arg(long)]
    no_cache_model: bool,

    /// Custom directory for model caching (overrides default ~/.cache/llm-mem/models)
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Load configuration first to determine log directory
    let mut config = if let Some(config_path) = &cli.config {
        Config::load(config_path)
            .with_context(|| format!("Failed to load config from {:?}", config_path))?
    } else {
        // No config file - use defaults but still apply environment variable overrides
        let mut config = Config::default();
        config.apply_env_overrides();
        config
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
    if let Some(request_format_str) = &cli.request_format {
        match request_format_str.to_lowercase().as_str() {
            "auto" => config.api_llm.request_format = llm_mem::config::RequestFormat::Auto,
            "rig" => config.api_llm.request_format = llm_mem::config::RequestFormat::Rig,
            "raw" => config.api_llm.request_format = llm_mem::config::RequestFormat::Raw,
            _ => {
                eprintln!("Invalid --request-format value: {}. Valid options: auto, rig, raw", request_format_str);
                std::process::exit(1);
            }
        }
    }
    if cli.no_cache_model {
        config.local.cache_model = false;
    }
    if let Some(cache_dir) = &cli.cache_dir {
        config.local.cache_dir = Some(cache_dir.display().to_string());
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

    // Create a task to wait for shutdown signals
    let shutdown_task = tokio::spawn(async {
        // Wait for SIGINT or SIGTERM
        match signal::ctrl_c().await {
            Ok(()) => {
                info!("Received SIGINT (Ctrl+C), initiating graceful shutdown...");
            }
            Err(e) => error!("Failed to listen for Ctrl+C: {}", e),
        }
    });

    // Wait for either the server to stop or a shutdown signal
    tokio::select! {
        result = running_service.waiting() => {
            match result {
                Ok(reason) => info!("Server shutdown: {:?}", reason),
                Err(e) => error!("Server error: {:?}", e),
            }
        }
        _ = shutdown_task => {
            info!("Shutdown signal received");
        }
    }

    // Cleanup llama-cpp backend resources
    #[cfg(feature = "local")]
    llm_mem::llm::local_client::cleanup_llama_backend();
    info!("Graceful shutdown complete");

    Ok(())
}
