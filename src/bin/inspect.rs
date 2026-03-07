use clap::{Parser, Subcommand};
use llm_mem::{
    types::{Filters, MemoryState, MemoryType},
    vector_store::{VectorLiteConfig, VectorLiteStore, VectorStore},
};
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "llm-mem-inspect")]
#[command(about = "CLI tool for inspecting llm-mem memory banks")]
#[command(version)]
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
        Commands::Stats { bank } => show_stats(&cli.banks_dir, &bank).await,
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
        Commands::LayerStats { bank, format } => show_layer_stats(&cli.banks_dir, &bank, format).await,
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

async fn list_banks(banks_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
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
    banks_dir: &PathBuf,
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

    let config = VectorLiteConfig {
        collection_name: format!("bank-{}", bank_name),
        persistence_path: Some(db_path),
        ..VectorLiteConfig::default()
    };

    let store = VectorLiteStore::with_config(config)?;

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
                println!(
                    "{},{},{},{},{},{},{},{},{}",
                    memory.id,
                    escape_csv(content_str),
                    format!("{:?}", memory.metadata.memory_type),
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
    banks_dir: &PathBuf,
    bank_name: &str,
    memory_id: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = banks_dir.join(format!("{}.db", bank_name));

    if !db_path.exists() {
        eprintln!("Bank '{}' not found at: {}", bank_name, db_path.display());
        return Ok(());
    }

    let config = VectorLiteConfig {
        collection_name: format!("bank-{}", bank_name),
        persistence_path: Some(db_path),
        ..VectorLiteConfig::default()
    };

    let store = VectorLiteStore::with_config(config)?;

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
    banks_dir: &PathBuf,
    bank_name: &str,
    output: Option<PathBuf>,
    pretty: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = banks_dir.join(format!("{}.db", bank_name));

    if !db_path.exists() {
        eprintln!("Bank '{}' not found at: {}", bank_name, db_path.display());
        return Ok(());
    }

    let config = VectorLiteConfig {
        collection_name: format!("bank-{}", bank_name),
        persistence_path: Some(db_path),
        ..VectorLiteConfig::default()
    };

    let store = VectorLiteStore::with_config(config)?;
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

async fn show_stats(
    banks_dir: &PathBuf,
    bank_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = banks_dir.join(format!("{}.db", bank_name));

    if !db_path.exists() {
        eprintln!("Bank '{}' not found at: {}", bank_name, db_path.display());
        return Ok(());
    }

    let metadata = tokio::fs::metadata(&db_path).await?;
    let file_size = metadata.len();

    let config = VectorLiteConfig {
        collection_name: format!("bank-{}", bank_name),
        persistence_path: Some(db_path),
        ..VectorLiteConfig::default()
    };

    let store = VectorLiteStore::with_config(config)?;
    let memories = store.list(&Filters::default(), None).await?;

    let total_count = memories.len();
    let mut type_counts = std::collections::HashMap::new();
    let mut total_entities = 0usize;
    let mut total_relations = 0usize;
    let mut total_contexts = 0usize;

    for memory in &memories {
        *type_counts
            .entry(format!("{:?}", memory.metadata.memory_type))
            .or_insert(0usize) += 1;
        total_entities += memory.metadata.entities.len();
        total_relations += memory.metadata.relations.len();
        total_contexts += memory.metadata.context.len();
    }

    println!("Statistics for bank '{}'", bank_name);
    println!("{}", "=".repeat(50));
    println!("File size: {}", format_bytes(file_size));
    println!("Total memories: {}", total_count);
    println!();
    println!("By Memory Type:");
    for (mem_type, count) in type_counts {
        println!(
            "  {:<20} {:>5} ({:.1}%)",
            mem_type,
            count,
            if total_count > 0 {
                (count as f64 / total_count as f64) * 100.0
            } else {
                0.0
            }
        );
    }
    println!();
    println!("Totals:");
    println!("  Entities:   {}", total_entities);
    println!("  Relations:  {}", total_relations);
    println!("  Contexts:   {}", total_contexts);
    println!();
    println!("Averages per memory:");
    if total_count > 0 {
        println!(
            "  Entities:   {:.2}",
            total_entities as f64 / total_count as f64
        );
        println!(
            "  Relations:  {:.2}",
            total_relations as f64 / total_count as f64
        );
        println!(
            "  Contexts:   {:.2}",
            total_contexts as f64 / total_count as f64
        );
    }

    Ok(())
}

async fn search_memories(
    banks_dir: &PathBuf,
    bank_name: &str,
    query: &str,
    mode: SearchMode,
    limit: usize,
    case_insensitive: bool,
    show_scores: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = banks_dir.join(format!("{}.db", bank_name));

    if !db_path.exists() {
        eprintln!("Bank '{}' not found at: {}", bank_name, db_path.display());
        return Ok(());
    }

    match mode {
        SearchMode::Text => {
            text_search(&db_path, query, limit, case_insensitive, show_scores).await
        }
        SearchMode::Semantic => {
            eprintln!("Semantic search requires LLM initialization.");
            eprintln!("Use 'text' mode instead for CLI inspection:");
            eprintln!(
                "  llm-mem-inspect search -b {} --mode text '{}'",
                bank_name, query
            );
            Ok(())
        }
    }
}

async fn text_search(
    db_path: &Path,
    query: &str,
    limit: usize,
    case_insensitive: bool,
    show_scores: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = VectorLiteConfig {
        collection_name: "temp".to_string(),
        persistence_path: Some(db_path.to_path_buf()),
        ..VectorLiteConfig::default()
    };

    let store = VectorLiteStore::with_config(config)?;
    let memories = store.list(&Filters::default(), None).await?;

    let search_term = if case_insensitive {
        query.to_lowercase()
    } else {
        query.to_string()
    };

    let mut results: Vec<(f32, &llm_mem::types::Memory)> = Vec::new();

    for memory in &memories {
        let content_str = memory.content.as_deref().unwrap_or("");
        let score = calculate_text_match_score(content_str, &search_term, case_insensitive);

        // Also check entities, relations, and context
        let entity_score: f32 = memory
            .metadata
            .entities
            .iter()
            .map(|e| calculate_text_match_score(e, &search_term, case_insensitive) * 0.8)
            .fold(0.0, f32::max);

        let relation_score: f32 = memory
            .metadata
            .relations
            .iter()
            .map(|r| {
                let rel_text = format!("{} {}", r.relation, r.target);
                calculate_text_match_score(&rel_text, &search_term, case_insensitive) * 0.7
            })
            .fold(0.0, f32::max);

        let context_score: f32 = memory
            .metadata
            .context
            .iter()
            .map(|c| calculate_text_match_score(c, &search_term, case_insensitive) * 0.6)
            .fold(0.0, f32::max);

        let total_score = score
            .max(entity_score)
            .max(relation_score)
            .max(context_score);

        if total_score > 0.0 {
            results.push((total_score, memory));
        }
    }

    // Sort by score (descending)
    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);

    if results.is_empty() {
        println!("No memories found matching '{}'", query);
        println!();
        println!("Tips:");
        println!("  - Try a broader search term");
        println!("  - Use --case-insensitive for case-insensitive search");
        println!("  - List all memories: llm-mem-inspect list");
    } else {
        println!("Found {} memories matching '{}'", results.len(), query);
        println!();

        if show_scores {
            println!("{:>4} {:<8} {:<36} {:<50}", "#", "Score", "ID", "Content");
            println!("{}", "-".repeat(110));
            for (idx, (score, memory)) in results.iter().enumerate() {
                let content_str = memory.content.as_deref().unwrap_or("");
                let content = if content_str.len() > 47 {
                    format!("{}...", &content_str[..47])
                } else {
                    content_str.to_string()
                };
                println!(
                    "{:>4} {:<8.3} {:<36} {}",
                    idx + 1,
                    score,
                    memory.id,
                    content
                );
            }
        } else {
            println!("{:>4} {:<36} {:<50}", "#", "ID", "Content");
            println!("{}", "-".repeat(95));
            for (idx, (_, memory)) in results.iter().enumerate() {
                let content_str = memory.content.as_deref().unwrap_or("");
                let content = if content_str.len() > 47 {
                    format!("{}...", &content_str[..47])
                } else {
                    content_str.to_string()
                };
                println!("{:>4} {:<36} {}", idx + 1, memory.id, content);
            }
        }

        println!();
        println!("Use 'show <memory-id>' to see full details of any result");
    }

    Ok(())
}

fn calculate_text_match_score(text: &str, query: &str, case_insensitive: bool) -> f32 {
    let text_to_search = if case_insensitive {
        text.to_lowercase()
    } else {
        text.to_string()
    };

    let query_lower = query.to_lowercase();
    let text_lower = text_to_search.to_lowercase();

    // Exact match
    if text_lower == query_lower {
        return 1.0;
    }

    // Contains full query
    if text_lower.contains(&query_lower) {
        // Score based on position (earlier is better) and length ratio
        let position_score = 0.7;
        let length_ratio = query.len() as f32 / text.len() as f32;
        return position_score + (length_ratio * 0.2);
    }

    // Word-by-word matching for multi-word queries
    let query_words: Vec<&str> = query_lower.split_whitespace().collect();
    if query_words.len() > 1 {
        let mut word_matches = 0;
        for word in &query_words {
            if text_lower.contains(word) {
                word_matches += 1;
            }
        }
        if word_matches > 0 {
            return (word_matches as f32 / query_words.len() as f32) * 0.5;
        }
    }

    0.0
}

fn print_table_header() {
    println!(
        "{:>4} {:<36} {:<50} {:<15}",
        "#", "ID", "Content (truncated)", "Type"
    );
    println!("{}", "-".repeat(110));
}

fn print_memory_row(idx: usize, memory: &llm_mem::types::Memory) {
    let content_str = memory.content.as_deref().unwrap_or("");
    let content = if content_str.len() > 47 {
        format!("{}...", &content_str[..47])
    } else {
        content_str.to_string()
    };

    println!(
        "{:>4} {:<36} {:<50} {:<15}",
        idx,
        memory.id,
        content,
        format!("{:?}", memory.metadata.memory_type)
    );
}

fn print_memory_detail(memory: &llm_mem::types::Memory) {
    println!("Memory ID: {}", memory.id);
    println!("{}", "=".repeat(60));
    println!();
    println!("Content:");
    println!("{}", memory.content.as_deref().unwrap_or("[no content]"));
    println!();
    println!("Type: {:?}", memory.metadata.memory_type);
    println!("Importance: {:.2}", memory.metadata.importance_score);
    println!();

    if !memory.metadata.entities.is_empty() {
        println!("Entities ({}):", memory.metadata.entities.len());
        for entity in &memory.metadata.entities {
            println!("  • {}", entity);
        }
        println!();
    }

    if !memory.metadata.relations.is_empty() {
        println!("Relations ({}):", memory.metadata.relations.len());
        for relation in &memory.metadata.relations {
            println!("  • {} → {}", relation.relation, relation.target);
        }
        println!();
    }

    if !memory.metadata.context.is_empty() {
        println!("Context ({}):", memory.metadata.context.len());
        for ctx in &memory.metadata.context {
            println!("  • {}", ctx);
        }
        println!();
    }

    println!("Created: {}", memory.created_at);
    println!("Updated: {}", memory.updated_at);

    if !memory.metadata.custom.is_empty() {
        println!();
        println!("Custom Metadata:");
        for (key, value) in memory.metadata.custom.iter() {
            println!("  {}: {}", key, value);
        }
    }
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

async fn show_layer_stats(
    banks_dir: &Path,
    bank_name: &str,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = banks_dir.join(format!("{}.db", bank_name));

    if !db_path.exists() {
        eprintln!("Bank '{}' not found at: {}", bank_name, db_path.display());
        return Ok(());
    }

    let config = VectorLiteConfig {
        collection_name: format!("bank-{}", bank_name),
        persistence_path: Some(db_path),
        ..VectorLiteConfig::default()
    };

    let store = VectorLiteStore::with_config(config)?;
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

    let config = VectorLiteConfig {
        collection_name: format!("bank-{}", bank_name),
        persistence_path: Some(db_path),
        ..VectorLiteConfig::default()
    };

    let store = VectorLiteStore::with_config(config)?;
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
            println!("└── Layer {} ({}) - {} memories", level, layer_name, memories_at_level.len());
            print_layer_branch(memories_at_level.clone(), "    ", show_ids, max_depth);
        } else {
            println!("├── Layer {} ({}) - {} memories", level, layer_name, memories_at_level.len());
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
        let truncated = if content.len() > 60 {
            format!("{}...", &content[..60])
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
