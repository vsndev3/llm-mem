use anyhow::{Context, anyhow};
use clap::Parser;
use llm_mem::{MemoryMcpService, config::Config};
use rmcp::{ServiceExt, transport::stdio};
use std::path::PathBuf;
use tokio::signal;
use tracing::{error, info};
use tracing_subscriber::fmt::{format::Writer, time::FormatTime};

#[derive(Parser)]
#[command(name = "llm-mem-mcp")]
#[command(about = "MCP server for LLM memory management")]
#[command(version = env!("BUILD_VERSION"))]
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

    /// Generate a commented-out configuration template and exit
    #[arg(long)]
    generate_config: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if cli.generate_config {
        println!("{}", Config::template());
        return Ok(());
    }

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
        config.llm.proxy_url = Some(proxy.clone());
    }
    if let Some(banks_dir) = &cli.banks_dir {
        config.vector_store.banks_dir = banks_dir.display().to_string();
    }
    if cli.no_grammar {
        config.llm.use_grammar = false;
    }
    if cli.no_structured_output {
        config.llm.use_structured_output = false;
    }
    if let Some(request_format_str) = &cli.request_format {
        match request_format_str.to_lowercase().as_str() {
            "auto" => config.llm.request_format = llm_mem::config::RequestFormat::Auto,
            "rig" => config.llm.request_format = llm_mem::config::RequestFormat::Rig,
            "raw" => config.llm.request_format = llm_mem::config::RequestFormat::Raw,
            _ => {
                eprintln!(
                    "Invalid --request-format value: {}. Valid options: auto, rig, raw",
                    request_format_str
                );
                std::process::exit(1);
            }
        }
    }
    if cli.no_cache_model {
        config.llm.cache_model = false;
    }
    if let Some(cache_dir) = &cli.cache_dir {
        config.llm.cache_dir = Some(cache_dir.display().to_string());
    }

    // Setup compact stderr logging
    let filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy();

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_ansi(false)
        .with_timer(CompactLocalTime)
        .compact()
        .init();

    info!("Starting LLM Memory MCP Server");
    info!("Configuration loaded from: {:?}", cli.config);

    // Acquire cross-process lock BEFORE initializing the database.
    // This prevents concurrent instances from corrupting LanceDB/SQLite.
    let banks_dir = std::path::PathBuf::from(&config.vector_store.banks_dir);
    let _instance_guard = llm_mem::instance_lock::acquire(&banks_dir, "MCP");

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
    #[cfg(feature = "local-llm")]
    llm_mem::llm::cleanup_llama_backend();
    info!("Graceful shutdown complete");

    Ok(())
}

/// Compact local-time formatter for tracing — produces `YYYY-MM-DD HH:MM:SS` timestamps.
struct CompactLocalTime;

impl FormatTime for CompactLocalTime {
    fn format_time(&self, w: &mut Writer<'_>) -> std::fmt::Result {
        write!(w, "{}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"))
    }
}
