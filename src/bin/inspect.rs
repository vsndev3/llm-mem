use clap::{Parser, Subcommand};
use llm_mem::types::{Filters, MemoryState, MemoryType};
use llm_mem::vector_store::VectorStore;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(feature = "vector-lite")]
use llm_mem::vector_store::{VectorLiteConfig, VectorLiteStore};

#[cfg(not(feature = "vector-lite"))]
use llm_mem::lance_store::{LanceDBConfig, LanceDBStore};

#[derive(Parser)]
#[command(name = "llm-mem-inspect")]
#[command(about = "CLI tool for inspecting llm-mem memory banks")]
#[command(version = env!("BUILD_VERSION"))]
struct Cli {
    /// Banks directory path
    #[arg(short, long, default_value = "llm-mem-data/banks")]
    banks_dir: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List all memory banks
    ListBanks,

    /// List memories in a specific bank
    List {
        /// Bank name
        #[arg(short, long, default_value = "default")]
        bank: String,

        /// Limit number of results
        #[arg(short, long, default_value = "50")]
        limit: usize,

        /// Output format
        #[arg(short, long, value_enum, default_value = "table")]
        format: OutputFormat,

        /// Filter by memory type
        #[arg(long)]
        memory_type: Option<String>,
    },

    /// Show detailed information about a specific memory
    Show {
        /// Bank name
        #[arg(short, long, default_value = "default")]
        bank: String,

        /// Memory ID
        memory_id: String,

        /// Output format
        #[arg(short, long, value_enum, default_value = "detail")]
        format: OutputFormat,
    },

    /// Export bank data to JSON
    Export {
        /// Bank name
        #[arg(short, long, default_value = "default")]
        bank: String,

        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Pretty print JSON
        #[arg(long)]
        pretty: bool,
    },

    /// Show statistics about a bank
    Stats {
        /// Bank name
        #[arg(short, long, default_value = "default")]
        bank: String,
    },

    /// Search memories using text or semantic search
    Search {
        /// Bank name
        #[arg(short, long, default_value = "default")]
        bank: String,

        /// Search query
        query: String,

        /// Search mode
        #[arg(short, long, value_enum, default_value = "text")]
        mode: SearchMode,

        /// Limit number of results
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Case insensitive search (for text mode)
        #[arg(short, long)]
        case_insensitive: bool,

        /// Show similarity scores (for text mode)
        #[arg(long)]
        show_scores: bool,
    },

    /// Show layer statistics
    LayerStats {
        /// Bank name
        #[arg(short, long, default_value = "default")]
        bank: String,

        /// Output format
        #[arg(short, long, value_enum, default_value = "table")]
        format: OutputFormat,
    },

    /// Show layer hierarchy as ASCII tree
    LayerTree {
        /// Bank name
        #[arg(short, long, default_value = "default")]
        bank: String,

        /// Start from specific layer level
        #[arg(short, long)]
        from_layer: Option<i32>,

        /// Maximum depth to display
        #[arg(short, long, default_value = "5")]
        max_depth: usize,

        /// Show memory IDs
        #[arg(long)]
        show_ids: bool,

        /// Show forgotten memories
        #[arg(long)]
        show_forgotten: bool,
    },
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum OutputFormat {
    Table,
    Detail,
    Json,
    Jsonl,
    Csv,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum SearchMode {
    /// Text-based search (grep-like) - fast, no LLM required
    Text,
    /// Semantic search using embeddings - requires LLM (not implemented in CLI)
    Semantic,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::ListBanks => list_banks(&cli.banks_dir).await,
        Commands::List {
            bank,
            limit,
            format,
            memory_type,
        } => list_memories(&cli.banks_dir, &bank, limit, format, memory_type).await,
        Commands::Show {
            bank,
            memory_id,
            format,
        } => show_memory(&cli.banks_dir, &bank, &memory_id, format).await,
        Commands::Export {
            bank,
            output,
            pretty,
        } => export_bank(&cli.banks_dir, &bank, output, pretty).await,
        Commands::Stats { bank } => show_stats(&cli.banks_dir, &bank, OutputFormat::Table).await,
        Commands::Search {
            bank,
            query,
            mode,
            limit,
            case_insensitive,
            show_scores,
        } => {
            search_memories(
                &cli.banks_dir,
                &bank,
                &query,
                mode,
                limit,
                case_insensitive,
                show_scores,
            )
            .await
        }
        Commands::LayerStats { bank, format } => {
            show_layer_stats(&cli.banks_dir, &bank, format).await
        }
        Commands::LayerTree {
            bank,
            from_layer,
            max_depth,
            show_ids,
            show_forgotten,
        } => {
            show_layer_tree(
                &cli.banks_dir,
                &bank,
                from_layer,
                max_depth,
                show_ids,
                show_forgotten,
            )
            .await
        }
    }
}

async fn list_banks(banks_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // List .db files in the directory
    let mut entries = tokio::fs::read_dir(banks_dir).await?;
    let mut banks = Vec::new();

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().map(|e| e == "db").unwrap_or(false) {
            let name = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let metadata = tokio::fs::metadata(&path).await?;
            let size = metadata.len();
            banks.push((name, path, size));
        }
    }

    println!("Memory Banks in {}:", banks_dir.display());
    println!("{:<30} {:<50} {:<15}", "Name", "Path", "Size");
    println!("{}", "-".repeat(95));

    for (name, path, size) in banks {
        println!(
            "{:<30} {:<50} {:<15}",
            name,
            path.display().to_string(),
            format_bytes(size)
        );
    }

    Ok(())
}

async fn list_memories(
    banks_dir: &Path,
    bank_name: &str,
    limit: usize,
    format: OutputFormat,
    memory_type: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = banks_dir.join(format!("{}.db", bank_name));

    if !db_path.exists() {
        eprintln!("Bank '{}' not found at: {}", bank_name, db_path.display());
        return Ok(());
    }

    #[cfg(feature = "vector-lite")]
    let store = {
        let config = VectorLiteConfig {
            collection_name: format!("bank-{}", bank_name),
            persistence_path: Some(db_path.clone()),
            ..VectorLiteConfig::default()
        };
        VectorLiteStore::with_config(config)?
    };

    #[cfg(not(feature = "vector-lite"))]
    let store = {
        let config = LanceDBConfig {
            table_name: format!("bank-{}", bank_name),
            database_path: db_path.clone(),
            embedding_dimension: 384,
        };
        LanceDBStore::new(config).await?
    };

    let mut filters = Filters::default();
    if let Some(mt) = memory_type {
        filters.memory_type = Some(MemoryType::parse(&mt));
    }

    let memories = store.list(&filters, Some(limit)).await?;

    match format {
        OutputFormat::Table => {
            print_table_header();
            for (idx, memory) in memories.iter().enumerate() {
                print_memory_row(idx + 1, memory);
            }
            println!(
                "\nTotal: {} memories (showing up to {})",
                memories.len(),
                limit
            );
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&memories)?;
            println!("{}", json);
        }
        OutputFormat::Jsonl => {
            for memory in memories {
                println!("{}", serde_json::to_string(&memory)?);
            }
        }
        OutputFormat::Csv => {
            println!(
                "id,content,memory_type,created_at,updated_at,entities,relations,context,importance_score"
            );
            for memory in memories {
                let content_str = memory.content.as_deref().unwrap_or("");
                let memory_type_str = format!("{:?}", memory.metadata.memory_type);
                println!(
                    "{},{},{},{},{},{},{},{},{}",
                    memory.id,
                    escape_csv(content_str),
                    memory_type_str,
                    memory.created_at,
                    memory.updated_at,
                    escape_csv(&memory.metadata.entities.join(";")),
                    escape_csv(
                        &memory
                            .metadata
                            .relations
                            .iter()
                            .map(|r| format!("{}:{}", r.relation, r.target))
                            .collect::<Vec<_>>()
                            .join(";")
                    ),
                    escape_csv(&memory.metadata.context.join(";")),
                    memory.metadata.importance_score
                );
            }
        }
        OutputFormat::Detail => {
            for (idx, memory) in memories.iter().enumerate() {
                println!("\n{}", "=".repeat(80));
                println!("Memory #{}", idx + 1);
                print_memory_detail(memory);
            }
        }
    }

    Ok(())
}

async fn show_memory(
    banks_dir: &Path,
    bank_name: &str,
    memory_id: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = banks_dir.join(format!("{}.db", bank_name));

    if !db_path.exists() {
        eprintln!("Bank '{}' not found at: {}", bank_name, db_path.display());
        return Ok(());
    }

    #[cfg(feature = "vector-lite")]
    let store = {
        let config = VectorLiteConfig {
            collection_name: format!("bank-{}", bank_name),
            persistence_path: Some(db_path.clone()),
            ..VectorLiteConfig::default()
        };
        VectorLiteStore::with_config(config)?
    };

    #[cfg(not(feature = "vector-lite"))]
    let store = {
        let config = LanceDBConfig {
            table_name: format!("bank-{}", bank_name),
            database_path: db_path.clone(),
            embedding_dimension: 384,
        };
        LanceDBStore::new(config).await?
    };

    match store.get(memory_id).await? {
        Some(memory) => match format {
            OutputFormat::Json | OutputFormat::Jsonl => {
                println!("{}", serde_json::to_string_pretty(&memory)?);
            }
            _ => {
                print_memory_detail(&memory);
            }
        },
        None => {
            eprintln!("Memory '{}' not found in bank '{}'", memory_id, bank_name);
        }
    }

    Ok(())
}

async fn export_bank(
    banks_dir: &Path,
    bank_name: &str,
    output: Option<PathBuf>,
    pretty: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = banks_dir.join(format!("{}.db", bank_name));

    if !db_path.exists() {
        eprintln!("Bank '{}' not found at: {}", bank_name, db_path.display());
        return Ok(());
    }

    #[cfg(feature = "vector-lite")]
    let store = {
        let config = VectorLiteConfig {
            collection_name: format!("bank-{}", bank_name),
            persistence_path: Some(db_path.clone()),
            ..VectorLiteConfig::default()
        };
        VectorLiteStore::with_config(config)?
    };

    #[cfg(not(feature = "vector-lite"))]
    let store = {
        let config = LanceDBConfig {
            table_name: format!("bank-{}", bank_name),
            database_path: db_path.clone(),
            embedding_dimension: 384,
        };
        LanceDBStore::new(config).await?
    };
    let memories = store.list(&Filters::default(), None).await?;

    let export_data = json!({
        "bank_name": bank_name,
        "export_timestamp": chrono::Utc::now().to_rfc3339(),
        "memory_count": memories.len(),
        "memories": memories
    });

    let json_str = if pretty {
        serde_json::to_string_pretty(&export_data)?
    } else {
        serde_json::to_string(&export_data)?
    };

    match output {
        Some(path) => {
            tokio::fs::write(&path, json_str).await?;
            println!("Exported {} memories to {}", memories.len(), path.display());
        }
        None => {
            println!("{}", json_str);
        }
    }

    Ok(())
}

async fn show_stats(banks_dir: &Path, bank_name: &str, format: OutputFormat) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = banks_dir.join(format!("{}.db", bank_name));

    if !db_path.exists() {
        eprintln!("Bank '{}' not found at: {}", bank_name, db_path.display());
        return Ok(());
    }

    let metadata = tokio::fs::metadata(&db_path).await?;
    let file_size = metadata.len();

   #[cfg(feature = "vector-lite")]
    let store = {
        let config = VectorLiteConfig {
            collection_name: format!("bank-{}", bank_name),
            persistence_path: Some(db_path.clone()),
            ..VectorLiteConfig::default()
        };
        VectorLiteStore::with_config(config)?
    };

    #[cfg(not(feature = "vector-lite"))]
    let store = {
        let config = LanceDBConfig {
            table_name: format!("bank-{}", bank_name),
            database_path: db_path.clone(),
            embedding_dimension: 384,
        };
        LanceDBStore::new(config).await?
    };
    let memories = store.list(&Filters::default(), None).await?;

    let mut layer_counts: HashMap<i32, usize> = HashMap::new();
    let mut state_counts: HashMap<String, usize> = HashMap::new();
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    let mut total_abstraction_sources = 0usize;

    for memory in &memories {
        let level = memory.metadata.layer.level;
        *layer_counts.entry(level).or_insert(0) += 1;

        let state_str = format!("{:?}", memory.metadata.state);
        *state_counts.entry(state_str).or_insert(0) += 1;

        let type_str = format!("{:?}", memory.metadata.memory_type);
        *type_counts.entry(type_str).or_insert(0) += 1;

        total_abstraction_sources += memory.metadata.abstraction_sources.len();
    }

    let max_layer = layer_counts.keys().max().copied().unwrap_or(0);
    let forgotten_count = state_counts.get("Forgotten").copied().unwrap_or(0);
    let active_count = state_counts.get("Active").copied().unwrap_or(0);

    match format {
        OutputFormat::Table => {
            println!("Layer Statistics for bank '{}'", bank_name);
            println!("{}", "=".repeat(60));
            println!();
            println!("Overview:");
            println!("  Total memories:     {}", memories.len());
            println!("  Active:             {}", active_count);
            println!("  Forgotten:          {}", forgotten_count);
            println!("  Max layer:          {}", max_layer);
            println!();
            println!("By Layer:");
            println!("  {:<25} {:>8} {:>10}", "Layer", "Count", "% of Total");
            println!("  {}", "-".repeat(45));
            let mut levels: Vec<_> = layer_counts.keys().collect();
            levels.sort();
            for level in levels {
                let count = layer_counts.get(level).unwrap();
                let pct = if !memories.is_empty() {
                    (*count as f64 / memories.len() as f64) * 100.0
                } else {
                    0.0
                };
                let layer_name = match level {
                    0 => "L0 (Raw Content)",
                    1 => "L1 (Structural)",
                    2 => "L2 (Semantic)",
                    3 => "L3 (Concept)",
                    4 => "L4 (Wisdom)",
                    -1 => "L-1 (Forgotten)",
                    _ => &format!("L{} (Custom)", level),
                };
                println!("  {:<25} {:>8} {:>9.1}%", layer_name, count, pct);
            }
            println!();
            println!("Abstraction Metrics:");
            println!("  Total source links: {}", total_abstraction_sources);
            if memories.len() > forgotten_count {
                let non_forgotten = memories.len() - forgotten_count;
                println!(
                    "  Avg sources/memory: {:.2}",
                    total_abstraction_sources as f64 / non_forgotten as f64
                );
            }
        }
        OutputFormat::Json => {
            let json = json!({
                "bank": bank_name,
                "total_memories": memories.len(),
                "active": active_count,
                "forgotten": forgotten_count,
                "max_layer": max_layer,
                "by_layer": layer_counts,
                "by_state": state_counts,
                "by_type": type_counts,
                "total_abstraction_sources": total_abstraction_sources
            });
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        _ => {
            eprintln!("Layer stats only supports 'table' and 'json' formats");
        }
    }

    Ok(())
}

async fn show_layer_tree(
    banks_dir: &Path,
    bank_name: &str,
    from_layer: Option<i32>,
    max_depth: usize,
    show_ids: bool,
    show_forgotten: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = banks_dir.join(format!("{}.db", bank_name));

    if !db_path.exists() {
        eprintln!("Bank '{}' not found at: {}", bank_name, db_path.display());
        return Ok(());
    }

    #[cfg(feature = "vector-lite")]
    let store = {
        let config = VectorLiteConfig {
            collection_name: format!("bank-{}", bank_name),
            persistence_path: Some(db_path.clone()),
            ..VectorLiteConfig::default()
        };
        VectorLiteStore::with_config(config)?
    };

    #[cfg(not(feature = "vector-lite"))]
    let store = {
        let config = LanceDBConfig {
            table_name: format!("bank-{}", bank_name),
            database_path: db_path.clone(),
            embedding_dimension: 384,
        };
        LanceDBStore::new(config).await?
    };
    let memories = store.list(&Filters::default(), None).await?;

    // Group memories by layer
    let mut by_layer: HashMap<i32, Vec<_>> = HashMap::new();
    for memory in &memories {
        if !show_forgotten && memory.metadata.state == MemoryState::Forgotten {
            continue;
        }
        if let Some(from) = from_layer
            && memory.metadata.layer.level < from
        {
            continue;
        }
        by_layer
            .entry(memory.metadata.layer.level)
            .or_default()
            .push(memory);
    }

    println!("Layer Hierarchy for bank '{}'", bank_name);
    println!("{}", "=".repeat(60));
    println!();

    let mut levels: Vec<_> = by_layer.keys().collect();
    levels.sort();

    for (idx, level) in levels.iter().enumerate() {
        let is_last_level = idx == levels.len() - 1;
        let memories_at_level = by_layer.get(level).unwrap();

        let layer_name = match level {
            0 => "Raw Content",
            1 => "Structural",
            2 => "Semantic",
            3 => "Concept",
            4 => "Wisdom",
            -1 => "Forgotten",
            _ => &format!("Custom L{}", level),
        };

        // Print layer header
        if is_last_level {
            println!(
                "└── Layer {} ({}) - {} memories",
                level,
                layer_name,
                memories_at_level.len()
            );
            print_layer_branch(memories_at_level.clone(), "    ", show_ids, max_depth);
        } else {
            println!(
                "├── Layer {} ({}) - {} memories",
                level,
                layer_name,
                memories_at_level.len()
            );
            print_layer_branch(memories_at_level.clone(), "│   ", show_ids, max_depth);
        }
    }

    println!();
    println!("Legend:");
    println!("  L0: Raw user content (chunks, documents)");
    println!("  L1: Structural abstractions (summaries, sections)");
    println!("  L2: Semantic links (cross-references)");
    println!("  L3: Concepts (domain theories, principles)");
    println!("  L4: Wisdom (mental models, paradigms)");
    println!();
    println!("Tip: Use --from-layer N to start from layer N");
    println!("     Use --max-depth N to limit memories shown per layer");
    println!("     Use --show-ids to display memory UUIDs");
    println!("     Use --show-forgotten to include forgotten memories");

    Ok(())
}

fn print_layer_branch(
    memories: Vec<&llm_mem::types::Memory>,
    prefix: &str,
    show_ids: bool,
    max_depth: usize,
) {
    let display_count = std::cmp::min(memories.len(), max_depth);

    for (idx, memory) in memories.iter().take(display_count).enumerate() {
        let is_last = idx == display_count - 1 || idx == memories.len() - 1;
        let branch = if is_last { "└──" } else { "├──" };

        let content = memory.content.as_deref().unwrap_or("[no content]");
        let truncated = if content.chars().count() > 60 {
            format!("{}...", content.chars().take(60).collect::<String>())
        } else {
            content.to_string()
        };

        if show_ids {
            println!("{} {} {} [{}]", prefix, branch, truncated, memory.id);
        } else {
            println!("{} {} {}", prefix, branch, truncated);
        }
    }

    if memories.len() > max_depth {
        let remaining = memories.len() - max_depth;
        println!("{} ... and {} more", prefix, remaining);
    }
}

// Display and formatting functions (backend-agnostic — they operate on Memory types, not VectorStore)

fn format_bytes(bytes: u64) -> String {
    use std::fmt::Write;
    let mut result = String::new();
    write!(&mut result, "{} bytes", bytes).unwrap();
    result
}

fn print_table_header() {
    println!("{:<4} {:<40} {:<15} {:<10}", "#", "Content", "Type", "Importance");
}

fn print_memory_row(idx: usize, memory: &llm_mem::types::Memory) {
    let content = memory.content.as_deref().unwrap_or("[no content]");
    let truncated = if content.chars().count() > 38 {
        format!("{}...", content.chars().take(38).collect::<String>())
    } else {
        content.to_string()
    };
    println!(
        "{:<4} {:<40} {:<15} {:<10}",
        idx,
        truncated,
        format!("{:?}", memory.metadata.memory_type),
        memory.metadata.importance_score
    );
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn print_memory_detail(memory: &llm_mem::types::Memory) {
    println!("Memory ID: {}", memory.id);
    println!("Content: {:?}", memory.content);
    println!("Metadata: {:?}", memory.metadata);
    println!("Created: {}", memory.created_at);
    println!("Updated: {}", memory.updated_at);
}

async fn search_memories(
    _banks_dir: &std::path::Path,
    _bank_name: &str,
    _query: &str,
    _mode: SearchMode,
    _limit: usize,
    _case_insensitive: bool,
    _show_scores: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Search functionality requires LLM initialization");
    Ok(())
}

async fn show_layer_stats(
    banks_dir: &std::path::Path,
    bank_name: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = banks_dir.join(format!("{}.db", bank_name));

    #[cfg(feature = "vector-lite")]
    {
        use llm_mem::vector_store::{VectorLiteConfig, VectorLiteStore};
        let config = VectorLiteConfig {
            collection_name: format!("bank-{}", bank_name),
            persistence_path: Some(db_path),
            ..VectorLiteConfig::default()
        };
        let store = VectorLiteStore::with_config(config)?;

        let all_memories = store.list(&Filters::default(), None).await?;
        let mut layer_counts: HashMap<i32, usize> = HashMap::new();
        for m in &all_memories {
            *layer_counts.entry(m.metadata.layer.level).or_default() += 1;
        }

        match format {
            OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(&layer_counts)?);
            }
            _ => {
                println!("Layer statistics for bank '{}':", bank_name);
                for level in 0..=4 {
                    let count = layer_counts.get(&level).unwrap_or(&0);
                    println!("  L{}: {} memories", level, count);
                }
                println!("  Total: {} memories", all_memories.len());
            }
        }
        return Ok(());
    }

    #[cfg(not(feature = "vector-lite"))]
    {
        use llm_mem::lance_store::{LanceDBConfig, LanceDBStore};
        let config = LanceDBConfig {
            table_name: format!("bank-{}", bank_name),
            database_path: db_path.clone(),
            embedding_dimension: 384,
        };

        if !db_path.exists() {
            eprintln!("Bank '{}' not found at {}", bank_name, db_path.display());
            return Ok(());
        }

        let store = LanceDBStore::new(config).await?;
        let all_memories = store.list(&Filters::default(), None).await?;
        let mut layer_counts: HashMap<i32, usize> = HashMap::new();
        for m in &all_memories {
            *layer_counts.entry(m.metadata.layer.level).or_default() += 1;
        }

        match format {
            OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(&layer_counts)?);
            }
            _ => {
                println!("Layer statistics for bank '{}':", bank_name);
                for level in 0..=4 {
                    let count = layer_counts.get(&level).unwrap_or(&0);
                    println!("  L{}: {} memories", level, count);
                }
                println!("  Total: {} memories", all_memories.len());
            }
        }
        return Ok(());
    }
}
