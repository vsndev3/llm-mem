//! Migration tool for layered memory architecture
//!
//! This tool migrates existing memories to the new layered architecture:
//! - Tags all existing memories as L0 (raw content)
//! - Initializes layer metadata fields with defaults
//! - Creates a migration report

#[cfg(feature = "vector-lite")]
use llm_mem::{
    VectorStore,
    config::Config,
    types::{Filters, LayerInfo, MemoryState},
    vector_store::{VectorLiteConfig, VectorLiteStore},
};
#[cfg(feature = "vector-lite")]
use vectorlite::{IndexType, SimilarityMetric};

#[cfg(feature = "vector-lite")]
#[derive(Parser, Debug)]
#[command(author, version = env!("BUILD_VERSION"), about, long_about = None)]
struct Args {
    /// Path to config file
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Path to vector store persistence file
    #[arg(short, long)]
    store_path: Option<String>,

    /// Dry run - don't modify anything
    #[arg(long, default_value = "false")]
    dry_run: bool,

    /// Verbose output
    #[arg(short, long, default_value = "false")]
    verbose: bool,
}

#[cfg(feature = "vector-lite")]
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Layered Memory Migration Tool");
    info!("==============================");

    // Load configuration
    let config = if std::path::Path::new(&args.config).exists() {
        info!("Loading config from: {}", args.config);
        Config::load(&args.config)?
    } else {
        info!("Config file not found, using defaults");
        Config::default()
    };

    // Determine persistence path
    let persistence_path = args
        .store_path
        .or_else(|| config.vector_store.vectorlite.persistence_path.clone());

    if persistence_path.is_none() {
        error!("No persistence path configured. Migration requires a persistent store.");
        error!("   Use --store-path to specify the path, or configure it in config.toml");
        std::process::exit(1);
    }

    let persistence_path = PathBuf::from(persistence_path.unwrap());
    info!("Vector store path: {}", persistence_path.display());

    if !persistence_path.exists() {
        info!("Persistence file does not exist. No migration needed.");
        return Ok(());
    }

    // Initialize vector store
    info!("Initializing vector store...");
    let store = VectorLiteStore::with_config(VectorLiteConfig {
        collection_name: "llm-memories".to_string(),
        index_type: IndexType::Flat, // Use Flat for migration
        metric: SimilarityMetric::Cosine,
        persistence_path: Some(persistence_path.clone()),
    })?;

    // Get all memories
    info!("Loading all memories...");
    let all_memories = store.list(&Filters::new(), None).await?;
    let total_count = all_memories.len();
    info!("Found {} memories", total_count);

    if total_count == 0 {
        info!("No memories to migrate");
        return Ok(());
    }

    // Analyze current state
    let mut already_migrated = 0;
    let mut needs_migration = 0;
    let mut forgotten_count = 0;

    for memory in &all_memories {
        if memory.metadata.layer.level == 0
            && memory.metadata.layer.name == Some("raw_content".to_string())
        {
            already_migrated += 1;
        } else if memory.metadata.layer.level < 0 {
            forgotten_count += 1;
        } else {
            needs_migration += 1;
        }
    }

    info!("📈 Migration analysis:");
    info!("   - Already migrated: {}", already_migrated);
    info!("   - Needs migration: {}", needs_migration);
    info!("   - Forgotten state: {}", forgotten_count);

    if args.dry_run {
        info!("🔍 Dry run mode - no changes will be made");
        return Ok(());
    }

    // Perform migration
    info!("🚀 Starting migration...");
    let mut migrated_count = 0;
    let mut error_count = 0;

    for mut memory in all_memories {
        // Skip already migrated memories
        if memory.metadata.layer.level == 0
            && memory.metadata.layer.name == Some("raw_content".to_string())
        {
            continue;
        }

        // Skip forgotten memories (they should keep their state)
        if memory.metadata.state == MemoryState::Forgotten {
            continue;
        }

        // Apply L0 layer tag
        memory.metadata.layer = LayerInfo::raw_content();

        // Ensure state is active
        if memory.metadata.state == MemoryState::Active {
            // Already active, no change needed
        } else {
            memory.metadata.state = MemoryState::Active;
        }

        // Update memory in store
        if let Err(e) = store.update(&memory).await {
            error!("Failed to update memory {}: {}", memory.id, e);
            error_count += 1;
            continue;
        }

        migrated_count += 1;

        if args.verbose {
            info!("Migrated: {}", memory.id);
        }
    }

    // Report results
    info!("Migration complete!");
    info!("Results:");
    info!("   - Migrated: {}", migrated_count);
    info!("   - Errors: {}", error_count);
    info!("   - Skipped (already migrated): {}", already_migrated);
    info!("   - Skipped (forgotten): {}", forgotten_count);

    if error_count > 0 {
        warn!(
            "{} memories failed to migrate. Check logs for details.",
            error_count
        );
    }

    Ok(())
}

#[cfg(not(feature = "vector-lite"))]
fn main() {
    eprintln!("migrate_layers requires the vector-lite feature");
    eprintln!("Run with: cargo run --bin migrate_layers --features vector-lite");
    std::process::exit(1);
}
