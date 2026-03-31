#!/usr/bin/env rust
//! # llm-mem CLI Tool
//!
//! A standalone interactive CLI for llm-mem memory management with document upload,
//! status checking, and memory inspection capabilities.
//!
//! Features:
//! - Interactive REPL mode with command history and tab completion
//! - Single command mode for scripting
//! - Batch mode for executing multiple commands
//! - Document upload with auto-chunking and processing
//! - Document status checking
//! - Memory listing, searching, and inspection
//! - Layer statistics and visualization
//! - System status monitoring

use clap::{Parser, Subcommand};
use llm_mem::{
    config::Config,
    document_session::{DocumentSessionManager, SessionStatus},
    llm::create_llm_client,
    memory_bank::MemoryBankManager,
    memory::MemoryManager,
    operations::{MemoryOperationPayload, MemoryOperations},
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn, error};
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

mod commands;
mod log_capture;
mod repl;
mod output;

// Re-export System for use in modules
pub use llm_mem::System;

#[derive(Parser)]
#[command(name = "llm-mem")]
#[command(about = "Interactive CLI for llm-mem memory management")]
#[command(version = env!("BUILD_VERSION"))]
struct Cli {
    /// Path to the configuration file. If not provided, searches:
    /// ./config.toml, ~/.config/llm-mem/config.toml, /etc/llm-mem/config.toml
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Directory for memory bank database files (overrides config banks_dir)
    #[arg(long)]
    banks_dir: Option<PathBuf>,

    /// Run in REPL mode (default when no subcommand provided)
    #[arg(long)]
    repl: bool,

    /// Execute single command and exit
    #[arg(long)]
    single: bool,

    /// Execute commands from file (batch mode)
    #[arg(long)]
    batch: Option<PathBuf>,

    /// Subcommands for single command mode
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Upload a document with automatic chunking and processing
    Upload {
        /// Path to file to upload
        #[arg(long)]
        file_path: PathBuf,

        /// Bank name (default: "default")
        #[arg(long, default_value = "default")]
        bank: String,

        /// Start processing immediately after upload (default: true)
        #[arg(long, default_value_t = true)]
        process_immediately: bool,

        /// Chunk size in bytes (optional)
        #[arg(long)]
        chunk_size: Option<usize>,

        /// Memory type (optional)
        #[arg(long)]
        memory_type: Option<String>,

        /// Context tags (optional, can be specified multiple times)
        #[arg(long)]
        context: Vec<String>,

        /// Output format (table, json, jsonl, csv)
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Begin a document storage session (multi-step upload)
    BeginUpload {
        /// File name
        #[arg(long)]
        file_name: String,

        /// Total file size in bytes
        #[arg(long)]
        total_size: usize,

        /// MIME type (optional)
        #[arg(long)]
        mime_type: Option<String>,

        /// Bank name (default: "default")
        #[arg(long, default_value = "default")]
        bank: String,

        /// Memory type (optional)
        #[arg(long)]
        memory_type: Option<String>,

        /// Context tags (optional, can be specified multiple times)
        #[arg(long)]
        context: Vec<String>,

        /// Custom metadata as JSON string (optional)
        #[arg(long)]
        metadata: Option<String>,

        /// Output format (table, json, jsonl, csv)
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Upload a part of a document (multi-step upload)
    UploadPart {
        /// Session ID from begin-upload
        #[arg(long)]
        session_id: String,

        /// Part index (0-based)
        #[arg(long)]
        part_index: usize,

        /// Path to file part
        #[arg(long)]
        file_path: PathBuf,

        /// Bank name (default: "default")
        #[arg(long, default_value = "default")]
        bank: String,

        /// Output format (table, json, jsonl, csv)
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Process an uploaded document
    ProcessDocument {
        /// Session ID from begin-upload
        #[arg(long)]
        session_id: String,

        /// Allow processing even if part count differs (default: false)
        #[arg(long)]
        partial_closure: bool,

        /// Bank name (default: "default")
        #[arg(long, default_value = "default")]
        bank: String,

        /// Output format (table, json, jsonl, csv)
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Check status of a document processing session
    DocStatus {
        /// Session ID
        #[arg(long)]
        session_id: String,

        /// Bank name (default: "default")
        #[arg(long, default_value = "default")]
        bank: String,

        /// Output format (table, json, jsonl, csv)
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// List all document sessions
    ListSessions {
        /// Bank name (default: "default")
        #[arg(long, default_value = "default")]
        bank: String,

        /// Output format (table, json, jsonl, csv)
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// List memories in a bank
    List {
        /// Bank name (default: "default")
        #[arg(long, default_value = "default")]
        bank: String,

        /// Limit number of results
        #[arg(long, default_value_t = 50)]
        limit: usize,

        /// Output format
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Filter by memory type
        #[arg(long)]
        memory_type: Option<String>,
    },

    /// Show detailed information about a specific memory
    Show {
        /// Bank name
        #[arg(long, default_value = "default")]
        bank: String,

        /// Memory ID
        #[arg(long)]
        memory_id: String,

        /// Output format
        #[arg(long, value_enum, default_value_t = OutputFormat::Detail)]
        format: OutputFormat,
    },

    /// Search memories using text or semantic search
    Search {
        /// Bank name
        #[arg(long, default_value = "default")]
        bank: String,

        /// Search query
        #[arg(long)]
        query: String,

        /// Search mode
        #[arg(long, value_enum, default_value_t = SearchMode::Text)]
        mode: SearchMode,

        /// Limit number of results
        #[arg(long, default_value_t = 10)]
        limit: usize,

        /// Case insensitive search (for text mode)
        #[arg(long)]
        case_insensitive: bool,

        /// Show similarity scores (for text mode)
        #[arg(long)]
        show_scores: bool,

        /// Similarity threshold override (0.0-1.0). Lower values return more results.
        #[arg(long)]
        threshold: Option<f32>,

        /// Output format
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Export bank data to JSON
    Export {
        /// Bank name
        #[arg(long, default_value = "default")]
        bank: String,

        /// Output file (stdout if not specified)
        #[arg(long)]
        output: Option<PathBuf>,

        /// Pretty print JSON
        #[arg(long)]
        pretty: bool,
    },

    /// Show statistics about a bank
    Stats {
        /// Bank name
        #[arg(long, default_value = "default")]
        bank: String,

        /// Output format (table, json, jsonl, csv)
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Show layer statistics
    LayerStats {
        /// Bank name
        #[arg(long, default_value = "default")]
        bank: String,

        /// Output format
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Show layer hierarchy as ASCII tree
    LayerTree {
        /// Bank name
        #[arg(long, default_value = "default")]
        bank: String,

        /// Start from specific layer level
        #[arg(long)]
        from_layer: Option<i32>,

        /// Maximum depth to display
        #[arg(long, default_value_t = 5)]
        max_depth: usize,

        /// Show memory IDs
        #[arg(long)]
        show_ids: bool,

        /// Show forgotten memories
        #[arg(long)]
        show_forgotten: bool,
    },

    /// Generate a configuration file with default values
    GenerateConfig {
        /// Output file path
        #[arg(long)]
        output: std::path::PathBuf,

        /// Output format (table, json, jsonl, csv)
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

 /// Visualize document processing and memory abstractions in real-time
    Viz,

    /// List all memory banks
    ListBanks {
        /// Output format (table, detail, json, jsonl, csv)
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Check system status and readiness
    SystemStatus {
        /// Output format (table, json, jsonl, csv)
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Database management: export, merge, check, fix
    Db {
        #[command(subcommand)]
        command: DbCommand,
    },
}

#[derive(Subcommand)]
enum DbCommand {
    /// Export a bank to a portable .db file
    Export {
        /// Bank name to export
        #[arg(long, default_value = "default")]
        bank: String,

        /// Output path (file or directory)
        #[arg(long)]
        output: std::path::PathBuf,

        /// Include session data (.sessions.db)
        #[arg(long)]
        include_sessions: bool,
    },

    /// Merge one or more source databases into a target bank
    Merge {
        /// Source bank names or .db file paths
        #[arg(long, required = true, num_args = 1..)]
        sources: Vec<String>,

        /// Target bank name (created if it doesn't exist)
        #[arg(long)]
        into: String,

        /// Duplicate handling: keep-newest, keep-first, keep-all
        #[arg(long, default_value = "keep-newest")]
        on_duplicate: String,

        /// Show what would be merged without changing anything
        #[arg(long)]
        dry_run: bool,
    },

    /// Check database consistency and integrity
    Check {
        /// Bank name to check
        #[arg(long)]
        bank: Option<String>,

        /// Check an external .db file
        #[arg(long)]
        file: Option<std::path::PathBuf>,

        /// Check all banks
        #[arg(long)]
        all: bool,

        /// Show detailed issue information
        #[arg(long)]
        verbose: bool,
    },

    /// Fix consistency issues in a bank
    Fix {
        /// Bank name to fix
        #[arg(long, default_value = "default")]
        bank: String,

        /// Fix specific issue types (can be repeated). All types if omitted.
        /// Valid: orphaned-abstractions, stale-states, missing-embeddings,
        ///        hash-mismatches, unreferenced-forgotten, duplicate-content,
        ///        invalid-layer-structure
        #[arg(long)]
        fix: Vec<String>,

        /// Show what would be fixed without changing anything
        #[arg(long)]
        dry_run: bool,

        /// Skip automatic backup before fixing
        #[arg(long)]
        no_backup: bool,

        /// Hard-delete unreferenced Forgotten memories
        #[arg(long)]
        purge: bool,
    },
}

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq)]
pub(crate) enum OutputFormat {
    Table,
    Detail,
    Json,
    Jsonl,
    Csv,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub(crate) enum SearchMode {
    /// Text-based search (grep-like) - fast, no LLM required
    Text,
    /// Semantic search using embeddings - requires LLM
    Semantic,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing with buffer capture for viz log panel.
    // The stderr layer uses RUST_LOG / default=INFO. The buffer layer captures
    // everything down to TRACE so the viz can filter client-side.
    let log_buffer = log_capture::global_log_buffer();
    let buffer_layer = log_capture::BufferLayer::new(log_buffer);

    let stderr_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(tracing::Level::WARN.into())
        .from_env_lossy();
    let buffer_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(tracing::Level::TRACE.into())
        // Squelch noisy third-party crates
        .parse_lossy("trace,rustyline=warn,reqwest=info,hyper=info,h2=info,rustls=info");

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(log_capture::TuiAwareStderr).with_filter(stderr_filter))
        .with(buffer_layer.with_filter(buffer_filter))
        .init();

    let cli = Cli::parse();

    // Handle generate config if needed (could be added later)
    // For now, focus on core functionality

    // Load configuration
    let mut config = load_configuration(&cli)?;

    // Apply CLI overrides
    apply_cli_overrides(&mut config, &cli)?;

    // Initialize system components
    let system = initialize_system(&config).await?;

    // Resume any interrupted document processing sessions
    auto_resume_sessions(&system).await;

    // Determine execution mode
    if cli.single || cli.command.is_some() {
        // Single command mode
        execute_single_command(&system, &cli).await?
    } else if let Some(batch_file) = &cli.batch {
        // Batch mode
        execute_batch_mode(&system, batch_file).await?
    } else {
        // REPL mode (default)
        run_repl(&system).await?
    }

    Ok(())
}

/// Load configuration using auto-discovery or explicit path
fn load_configuration(cli: &Cli) -> Result<Config, Box<dyn std::error::Error>> {
    if let Some(config_path) = &cli.config {
        // Explicit config path provided
        eprintln!("Config: {}", config_path.display());
        Config::load(config_path)
            .map_err(|e| format!("Failed to load config from {:?}: {}", config_path, e).into())
    } else {
        // Auto-discover configuration
        let search_paths = [
            std::env::current_dir().ok().map(|p| p.join("config.toml")),
            dirs::config_dir().map(|p| p.join("llm-mem/config.toml")),
            Some(PathBuf::from("/etc/llm-mem/config.toml")),
        ];
        let found = search_paths.iter().flatten().find(|p| p.exists());
        match found {
            Some(config_path) => {
                eprintln!("Config: {}", config_path.display());
                info!("Found config at: {:?}", config_path);
                Config::load(&config_path)
                    .map_err(|e| format!("Failed to load config from {:?}: {}", config_path, e).into())
            }
            None => {
                eprintln!("Config: using defaults (no config file found)");
                info!("No config file found, using defaults");
                let mut config = Config::default();
                config.apply_env_overrides();
                Ok(config)
            }
        }
    }
}

/// Apply CLI overrides to configuration
fn apply_cli_overrides(config: &mut Config, cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(banks_dir) = &cli.banks_dir {
        config.vector_store.banks_dir = banks_dir.display().to_string();
    }
    Ok(())
}

/// Initialize system components (LLM client, bank manager, operations)
async fn initialize_system(config: &Config) -> Result<System, Box<dyn std::error::Error>> {
    info!("Initializing LLM client");
    let llm_client = create_llm_client(config).await?;

    info!("Initializing memory bank manager");
    let banks_dir = PathBuf::from(&config.vector_store.banks_dir);
    let bank_manager = MemoryBankManager::new(
        banks_dir,
        llm_client,
        config.vector_store.clone(),
        config.memory.clone(),
    )?;

    // Get the default bank's managers using the public method
    let (memory_manager, session_manager) = bank_manager
        .resolve_bank_with_sessions(Some("default"))
        .await?;

    // Create operations instance
    let operations = MemoryOperations::with_session_manager(
        memory_manager.clone(),
        session_manager.clone(),
        None, // default_user_id
        None, // default_agent_id
        100,  // default_limit
    );

    let bank_manager = Arc::new(bank_manager);

    // Start the abstraction pipeline for progressive layer creation
    bank_manager.start_abstraction_pipeline().await.ok();

    Ok(System {
        bank_manager,
        memory_manager: memory_manager,
        session_manager: session_manager,
        operations: Arc::new(Mutex::new(operations)),
        models_dir: PathBuf::from(&config.llm.models_dir),
    })
}

/// Startup recovery: resume interrupted document sessions
async fn auto_resume_sessions(system: &System) {
    let banks = match system.bank_manager.list_banks().await {
        Ok(b) => b,
        Err(e) => {
            warn!("Could not list banks for session recovery: {}", e);
            return;
        }
    };

    for bank_info in banks {
        let bank_name = &bank_info.name;

        let (memory_manager, session_manager) =
            match system.bank_manager.resolve_bank_with_sessions(Some(bank_name)).await {
                Ok(pair) => pair,
                Err(e) => {
                    warn!("Could not resolve bank {} for recovery: {}", bank_name, e);
                    continue;
                }
            };

        let sessions = match session_manager.list_all_sessions() {
            Ok(s) => s,
            Err(e) => {
                warn!("Could not list sessions in bank {}: {}", bank_name, e);
                continue;
            }
        };

        let ops = MemoryOperations::with_session_manager(
            memory_manager,
            session_manager,
            None,
            None,
            100,
        );

        for session in sessions {
            match session.status {
                SessionStatus::Processing => {
                    eprintln!(
                        "Resuming interrupted session {} in bank {} ({} chunks)...",
                        session.session_id, bank_name, session.expected_parts
                    );
                    info!(
                        "Found stalled session {} in bank {} (status: Processing), resuming",
                        session.session_id, bank_name
                    );

                    let payload = MemoryOperationPayload {
                        session_id: Some(session.session_id.clone()),
                        bank: Some(bank_name.clone()),
                        partial_closure: Some(true),
                        ..Default::default()
                    };

                    if let Err(e) = ops.process_document(payload).await {
                        error!(
                            "Failed to auto-resume session {}: {}",
                            session.session_id, e
                        );
                    }
                }
                SessionStatus::Uploading => {
                    // Resume interrupted upload if we have file info and MD5
                    let file_path = session
                        .metadata
                        .custom_metadata
                        .as_ref()
                        .and_then(|m| m.get("file_path").and_then(|v| v.as_str()))
                        .map(|s| s.to_string());

                    let expected_md5 = session.metadata.md5sum.as_deref().map(|s| s.to_string());

                    if let Some(file_path) = file_path {
                        let path = std::path::Path::new(&file_path);
                        if path.exists() {
                            let content = match std::fs::read_to_string(path) {
                                Ok(c) => c,
                                Err(e) => {
                                    warn!("Cannot read file for resume {}: {}", file_path, e);
                                    continue;
                                }
                            };

                            let actual_md5 = format!("{:x}", md5::compute(&content));
                            if let Some(expected_md5) = &expected_md5 {
                                if actual_md5 != *expected_md5 {
                                    warn!(
                                        "File {} changed since upload started (MD5 mismatch), skipping",
                                        file_path
                                    );
                                    continue;
                                }
                            }

                            eprintln!(
                                "Resuming interrupted upload {} in bank {} from file: {}",
                                session.session_id, bank_name, file_path
                            );
                            info!(
                                "Found interrupted upload {} in bank {}, resuming with file: {}",
                                session.session_id, bank_name, file_path
                            );

                            let payload = MemoryOperationPayload {
                                session_id: Some(session.session_id.clone()),
                                bank: Some(bank_name.clone()),
                                file_path: Some(file_path),
                                process_immediately: Some(true),
                                ..Default::default()
                            };

                            if let Err(e) = ops.upload_document(payload).await {
                                error!(
                                    "Failed to auto-resume upload session {}: {}",
                                    session.session_id, e
                                );
                            }
                        } else {
                            warn!(
                                "File {} for session {} no longer exists, cannot resume",
                                file_path, session.session_id
                            );
                        }
                    } else {
                        warn!(
                            "Upload session {} has no file_path metadata, cannot resume",
                            session.session_id
                        );
                    }
                }
                _ => {} // Completed, failed, cancelled — nothing to do
            }
        }
    }
}

/// Execute a single command and exit
async fn execute_single_command(system: &System, cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(command) = &cli.command {
        match command {
            Commands::Upload {
                file_path,
                bank,
                process_immediately,
                chunk_size,
                memory_type,
                context,
                format,
            } => {
                commands::upload::handle_upload(
                    system,
                    file_path,
                    bank,
                    *process_immediately,
                    chunk_size.as_ref(),
                    memory_type.as_deref(),
                    context.clone(),
                    *format,
                )
                .await?
            }
            Commands::BeginUpload {
                file_name,
                total_size,
                mime_type,
                bank,
                memory_type,
                context,
                metadata,
                format,
            } => {
                commands::begin_upload::handle_begin_upload(
                    system,
                    file_name,
                    *total_size,
                    mime_type.as_deref(),
                    bank,
                    memory_type.as_deref(),
                    context.clone(),
                    metadata.as_deref(),
                    *format,
                )
                .await?
            }
            Commands::UploadPart {
                session_id,
                part_index,
                file_path,
                bank,
                format,
            } => {
                commands::upload_part::handle_upload_part(
                    system,
                    session_id,
                    *part_index,
                    file_path,
                    bank,
                    *format,
                )
                .await?
            }
            Commands::ProcessDocument {
                session_id,
                partial_closure,
                bank,
                format,
            } => {
                commands::process_document::handle_process_document(
                    system,
                    session_id,
                    *partial_closure,
                    bank,
                    *format,
                )
                .await?
            }
            Commands::DocStatus {
                session_id,
                bank,
                format,
            } => {
                commands::doc_status::handle_doc_status(
                    system,
                    session_id,
                    bank,
                    *format,
                )
                .await?
            }
            Commands::ListSessions {
                bank,
                format,
            } => {
                commands::list_sessions::handle_list_sessions(
                    system,
                    bank,
                    *format,
                )
                .await?
            }
            Commands::List {
                bank,
                limit,
                format,
                memory_type,
            } => {
                commands::list::handle_list(
                    system,
                    bank,
                    *limit,
                    *format,
                    memory_type.as_deref(),
                )
                .await?
            }
            Commands::Show {
                bank,
                memory_id,
                format,
            } => {
                commands::show::handle_show(
                    system,
                    bank,
                    memory_id,
                    *format,
                )
                .await?
            }
            Commands::Search {
                bank,
                query,
                mode,
                limit,
                case_insensitive,
                show_scores,
                threshold,
                format,
            } => {
                commands::search::handle_search(
                    system,
                    bank,
                    query,
                    *mode,
                    *limit,
                    *case_insensitive,
                    *show_scores,
                    *threshold,
                    *format,
                )
                .await?
            }
            Commands::Export {
                bank,
                output,
                pretty,
            } => {
                commands::export::handle_export(
                    system,
                    bank,
                    output.as_deref(),
                    *pretty,
                    OutputFormat::Table, // Default format for export command
                )
                .await?
            }
            Commands::Stats { bank, format } => {
                commands::stats::handle_stats(system, &bank, *format).await?
            }
            Commands::LayerStats { bank, format } => {
                commands::layer_stats::handle_layer_stats(system, bank, *format).await?
            }
            Commands::LayerTree {
                bank,
                from_layer,
                max_depth,
                show_ids,
                show_forgotten,
            } => {
                commands::layer_tree::handle_layer_tree(
                    system,
                    bank,
                    from_layer.as_ref(),
                    *max_depth,
                    *show_ids,
                    *show_forgotten,
                )
                .await?
            }
            Commands::GenerateConfig { output, format } => {
                commands::generate_config::handle_generate_config(&output, *format).await?
            }
            Commands::Viz => {
                commands::viz::handle_viz(system, None).await?
            }
            Commands::ListBanks { format } => {
                commands::list_banks::handle_list_banks(system, *format).await?
            }
            Commands::SystemStatus { format } => {
                commands::system_status::handle_system_status(system, *format).await?
            }
            Commands::Db { command } => {
                match command {
                    DbCommand::Export { bank, output, include_sessions } => {
                        commands::db::handle_db_export(system, bank, output, *include_sessions).await?
                    }
                    DbCommand::Merge { sources, into, on_duplicate, dry_run } => {
                        commands::db::handle_db_merge(system, sources, into, on_duplicate, *dry_run).await?
                    }
                    DbCommand::Check { bank, file, all, verbose } => {
                        commands::db::handle_db_check(
                            system,
                            bank.as_deref(),
                            file.as_deref(),
                            *all,
                            *verbose,
                        ).await?
                    }
                    DbCommand::Fix { bank, fix, dry_run, no_backup, purge } => {
                        commands::db::handle_db_fix(system, bank, fix, *dry_run, *no_backup, *purge).await?
                    }
                }
            }
        }
    }
    Ok(())
}

/// Execute commands from a batch file
async fn execute_batch_mode(_system: &System, batch_file: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let content = tokio::fs::read_to_string(batch_file).await?;
    let lines: Vec<&str> = content.lines().collect();

    for (line_num, line) in lines.iter().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue; // Skip empty lines and comments
        }

        // Simple command parsing - in a real implementation, we'd use a proper parser
        println!("Executing line {}: {}", line_num + 1, line);
        // For now, we'll just note that batch mode needs more sophisticated command parsing
        // This would integrate with the REPL command execution logic
    }

    Ok(())
}

/// Run the REPL (Read-Eval-Print Loop)
async fn run_repl(system: &System) -> Result<(), Box<dyn std::error::Error>> {
    println!("llm-mem interactive CLI");
    println!("Type 'help' for available commands, 'exit' to quit");
    repl::repl_loop(system).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    // --- CLI argument parsing tests ---

    #[test]
    fn test_cli_no_args_defaults_to_repl() {
        let cli = Cli::try_parse_from(["llm-mem"]).unwrap();
        assert!(!cli.repl);
        assert!(!cli.single);
        assert!(cli.command.is_none());
        assert!(cli.config.is_none());
        assert!(cli.banks_dir.is_none());
        assert!(cli.batch.is_none());
    }

    #[test]
    fn test_cli_repl_flag() {
        let cli = Cli::try_parse_from(["llm-mem", "--repl"]).unwrap();
        assert!(cli.repl);
    }

    #[test]
    fn test_cli_single_flag() {
        let cli = Cli::try_parse_from(["llm-mem", "--single"]).unwrap();
        assert!(cli.single);
    }

    #[test]
    fn test_cli_config_path() {
        let cli = Cli::try_parse_from(["llm-mem", "--config", "/path/to/config.toml"]).unwrap();
        assert_eq!(cli.config.unwrap(), PathBuf::from("/path/to/config.toml"));
    }

    #[test]
    fn test_cli_config_short_flag() {
        let cli = Cli::try_parse_from(["llm-mem", "-c", "/tmp/config.toml"]).unwrap();
        assert_eq!(cli.config.unwrap(), PathBuf::from("/tmp/config.toml"));
    }

    #[test]
    fn test_cli_banks_dir() {
        let cli = Cli::try_parse_from(["llm-mem", "--banks-dir", "/data/banks"]).unwrap();
        assert_eq!(cli.banks_dir.unwrap(), PathBuf::from("/data/banks"));
    }

    #[test]
    fn test_cli_batch_mode() {
        let cli = Cli::try_parse_from(["llm-mem", "--batch", "commands.txt"]).unwrap();
        assert_eq!(cli.batch.unwrap(), PathBuf::from("commands.txt"));
    }

    // --- Subcommand parsing tests ---

    #[test]
    fn test_cli_upload_command() {
        let cli = Cli::try_parse_from([
            "llm-mem", "upload", "--file-path", "/tmp/doc.txt",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::Upload { file_path, bank, process_immediately, format, .. } => {
                assert_eq!(file_path, PathBuf::from("/tmp/doc.txt"));
                assert_eq!(bank, "default");
                assert!(process_immediately);
                assert_eq!(format, OutputFormat::Table);
            }
            _ => panic!("Expected Upload command"),
        }
    }

    #[test]
    fn test_cli_upload_with_options() {
        let cli = Cli::try_parse_from([
            "llm-mem", "upload",
            "--file-path", "/tmp/doc.txt",
            "--bank", "mybank",
            "--chunk-size", "4096",
            "--memory-type", "document",
            "--context", "project-x",
            "--context", "docs",
            "--format", "json",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::Upload { file_path, bank, chunk_size, memory_type, context, format, .. } => {
                assert_eq!(file_path, PathBuf::from("/tmp/doc.txt"));
                assert_eq!(bank, "mybank");
                assert_eq!(chunk_size, Some(4096));
                assert_eq!(memory_type, Some("document".to_string()));
                assert_eq!(context, vec!["project-x".to_string(), "docs".to_string()]);
                assert_eq!(format, OutputFormat::Json);
            }
            _ => panic!("Expected Upload command"),
        }
    }

    #[test]
    fn test_cli_begin_upload_command() {
        let cli = Cli::try_parse_from([
            "llm-mem", "begin-upload",
            "--file-name", "large.pdf",
            "--total-size", "1048576",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::BeginUpload { file_name, total_size, bank, .. } => {
                assert_eq!(file_name, "large.pdf");
                assert_eq!(total_size, 1048576);
                assert_eq!(bank, "default");
            }
            _ => panic!("Expected BeginUpload command"),
        }
    }

    #[test]
    fn test_cli_upload_part_command() {
        let cli = Cli::try_parse_from([
            "llm-mem", "upload-part",
            "--session-id", "sess-123",
            "--part-index", "0",
            "--file-path", "/tmp/part0.bin",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::UploadPart { session_id, part_index, file_path, bank, .. } => {
                assert_eq!(session_id, "sess-123");
                assert_eq!(part_index, 0);
                assert_eq!(file_path, PathBuf::from("/tmp/part0.bin"));
                assert_eq!(bank, "default");
            }
            _ => panic!("Expected UploadPart command"),
        }
    }

    #[test]
    fn test_cli_process_document_command() {
        let cli = Cli::try_parse_from([
            "llm-mem", "process-document",
            "--session-id", "sess-abc",
            "--partial-closure",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::ProcessDocument { session_id, partial_closure, bank, .. } => {
                assert_eq!(session_id, "sess-abc");
                assert!(partial_closure);
                assert_eq!(bank, "default");
            }
            _ => panic!("Expected ProcessDocument command"),
        }
    }

    #[test]
    fn test_cli_doc_status_command() {
        let cli = Cli::try_parse_from([
            "llm-mem", "doc-status",
            "--session-id", "sess-xyz",
            "--bank", "test-bank",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::DocStatus { session_id, bank, .. } => {
                assert_eq!(session_id, "sess-xyz");
                assert_eq!(bank, "test-bank");
            }
            _ => panic!("Expected DocStatus command"),
        }
    }

    #[test]
    fn test_cli_list_sessions_command() {
        let cli = Cli::try_parse_from([
            "llm-mem", "list-sessions", "--format", "jsonl",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::ListSessions { bank, format } => {
                assert_eq!(bank, "default");
                assert_eq!(format, OutputFormat::Jsonl);
            }
            _ => panic!("Expected ListSessions command"),
        }
    }

    #[test]
    fn test_cli_list_command() {
        let cli = Cli::try_parse_from([
            "llm-mem", "list",
            "--bank", "mybank",
            "--limit", "25",
            "--memory-type", "observation",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::List { bank, limit, memory_type, format } => {
                assert_eq!(bank, "mybank");
                assert_eq!(limit, 25);
                assert_eq!(memory_type, Some("observation".to_string()));
                assert_eq!(format, OutputFormat::Table);
            }
            _ => panic!("Expected List command"),
        }
    }

    #[test]
    fn test_cli_list_defaults() {
        let cli = Cli::try_parse_from(["llm-mem", "list"]).unwrap();
        match cli.command.unwrap() {
            Commands::List { bank, limit, memory_type, format } => {
                assert_eq!(bank, "default");
                assert_eq!(limit, 50);
                assert!(memory_type.is_none());
                assert_eq!(format, OutputFormat::Table);
            }
            _ => panic!("Expected List command"),
        }
    }

    #[test]
    fn test_cli_show_command() {
        let cli = Cli::try_parse_from([
            "llm-mem", "show",
            "--memory-id", "mem-123",
            "--format", "json",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::Show { memory_id, format, bank } => {
                assert_eq!(memory_id, "mem-123");
                assert_eq!(format, OutputFormat::Json);
                assert_eq!(bank, "default");
            }
            _ => panic!("Expected Show command"),
        }
    }

    #[test]
    fn test_cli_search_command() {
        let cli = Cli::try_parse_from([
            "llm-mem", "search",
            "--query", "rust programming",
            "--mode", "semantic",
            "--limit", "5",
            "--case-insensitive",
            "--show-scores",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::Search { query, mode, limit, case_insensitive, show_scores, .. } => {
                assert_eq!(query, "rust programming");
                assert!(matches!(mode, SearchMode::Semantic));
                assert_eq!(limit, 5);
                assert!(case_insensitive);
                assert!(show_scores);
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_cli_search_defaults() {
        let cli = Cli::try_parse_from([
            "llm-mem", "search", "--query", "test",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::Search { mode, limit, case_insensitive, show_scores, .. } => {
                assert!(matches!(mode, SearchMode::Text));
                assert_eq!(limit, 10);
                assert!(!case_insensitive);
                assert!(!show_scores);
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_cli_export_command() {
        let cli = Cli::try_parse_from([
            "llm-mem", "export",
            "--bank", "prod",
            "--output", "/tmp/export.json",
            "--pretty",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::Export { bank, output, pretty } => {
                assert_eq!(bank, "prod");
                assert_eq!(output, Some(PathBuf::from("/tmp/export.json")));
                assert!(pretty);
            }
            _ => panic!("Expected Export command"),
        }
    }

    #[test]
    fn test_cli_stats_command() {
        let cli = Cli::try_parse_from([
            "llm-mem", "stats", "--bank", "analytics",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::Stats { bank, format, .. } => {
                assert_eq!(bank, "analytics");
                assert_eq!(format, OutputFormat::Table);
            }
            _ => panic!("Expected Stats command"),
        }
    }

    #[test]
    fn test_cli_layer_stats_command() {
        let cli = Cli::try_parse_from([
            "llm-mem", "layer-stats", "--format", "json",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::LayerStats { bank, format } => {
                assert_eq!(bank, "default");
                assert_eq!(format, OutputFormat::Json);
            }
            _ => panic!("Expected LayerStats command"),
        }
    }

    #[test]
    fn test_cli_layer_tree_command() {
        let cli = Cli::try_parse_from([
            "llm-mem", "layer-tree",
            "--from-layer", "1",
            "--max-depth", "3",
            "--show-ids",
            "--show-forgotten",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::LayerTree { bank, from_layer, max_depth, show_ids, show_forgotten } => {
                assert_eq!(bank, "default");
                assert_eq!(from_layer, Some(1));
                assert_eq!(max_depth, 3);
                assert!(show_ids);
                assert!(show_forgotten);
            }
            _ => panic!("Expected LayerTree command"),
        }
    }

    #[test]
    fn test_cli_layer_tree_defaults() {
        let cli = Cli::try_parse_from(["llm-mem", "layer-tree"]).unwrap();
        match cli.command.unwrap() {
            Commands::LayerTree { from_layer, max_depth, show_ids, show_forgotten, .. } => {
                assert!(from_layer.is_none());
                assert_eq!(max_depth, 5);
                assert!(!show_ids);
                assert!(!show_forgotten);
            }
            _ => panic!("Expected LayerTree command"),
        }
    }

    #[test]
    fn test_cli_list_banks_command() {
        let cli = Cli::try_parse_from(["llm-mem", "list-banks"]).unwrap();
        assert!(matches!(cli.command.unwrap(), Commands::ListBanks { .. }));
    }

    #[test]
    fn test_cli_system_status_command() {
        let cli = Cli::try_parse_from(["llm-mem", "system-status"]).unwrap();
        assert!(matches!(cli.command.unwrap(), Commands::SystemStatus { .. }));
    }

    // --- OutputFormat enum tests ---

    #[test]
    fn test_output_format_all_variants() {
        let formats = [
            ("table", OutputFormat::Table),
            ("detail", OutputFormat::Detail),
            ("json", OutputFormat::Json),
            ("jsonl", OutputFormat::Jsonl),
            ("csv", OutputFormat::Csv),
        ];
        for (name, expected) in &formats {
            let cli = Cli::try_parse_from(["llm-mem", "list", "--format", name]).unwrap();
            match cli.command.unwrap() {
                Commands::List { format, .. } => assert_eq!(format, *expected),
                _ => panic!("Expected List command"),
            }
        }
    }

    #[test]
    fn test_output_format_invalid() {
        let result = Cli::try_parse_from(["llm-mem", "list", "--format", "xml"]);
        assert!(result.is_err());
    }

    // --- SearchMode enum tests ---

    #[test]
    fn test_search_mode_text() {
        let cli = Cli::try_parse_from([
            "llm-mem", "search", "--query", "x", "--mode", "text",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::Search { mode, .. } => assert!(matches!(mode, SearchMode::Text)),
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_search_mode_semantic() {
        let cli = Cli::try_parse_from([
            "llm-mem", "search", "--query", "x", "--mode", "semantic",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::Search { mode, .. } => assert!(matches!(mode, SearchMode::Semantic)),
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_search_mode_invalid() {
        let result = Cli::try_parse_from([
            "llm-mem", "search", "--query", "x", "--mode", "hybrid",
        ]);
        assert!(result.is_err());
    }

    // --- apply_cli_overrides tests ---

    #[test]
    fn test_apply_cli_overrides_banks_dir() {
        let cli = Cli::try_parse_from(["llm-mem", "--banks-dir", "/custom/banks"]).unwrap();
        let mut config = Config::default();
        apply_cli_overrides(&mut config, &cli).unwrap();
        assert_eq!(config.vector_store.banks_dir, "/custom/banks");
    }

    #[test]
    fn test_apply_cli_overrides_no_banks_dir() {
        let cli = Cli::try_parse_from(["llm-mem"]).unwrap();
        let mut config = Config::default();
        let original_dir = config.vector_store.banks_dir.clone();
        apply_cli_overrides(&mut config, &cli).unwrap();
        assert_eq!(config.vector_store.banks_dir, original_dir);
    }

    // --- load_configuration tests ---

    #[test]
    fn test_load_configuration_no_config_uses_defaults() {
        let cli = Cli::try_parse_from(["llm-mem"]).unwrap();
        let result = load_configuration(&cli);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_configuration_nonexistent_path_errors() {
        let cli = Cli::try_parse_from(["llm-mem", "--config", "/nonexistent/path/config.toml"]).unwrap();
        let result = load_configuration(&cli);
        assert!(result.is_err());
    }
}