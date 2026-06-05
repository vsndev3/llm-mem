use llm_mem::{
    System,
    consistency::{IssueKind, IssueSeverity},
    memory_bank::DuplicateStrategy,
};

// ── Export ──────────────────────────────────────────────────────────

pub async fn handle_db_export(
    system: &System,
    bank: &str,
    output: &std::path::Path,
    include_sessions: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (dest, manifest) = system
        .bank_manager
        .export_bank(bank, output, include_sessions)
        .await?;

    println!("Exported bank '{}' to {}", bank, dest.display());
    println!(
        "  Memories: {}, Size: {} bytes, SHA-256: {}",
        manifest.memory_count,
        manifest.size_bytes,
        &manifest.sha256[..16.min(manifest.sha256.len())]
    );
    if include_sessions {
        let session_path = dest.with_extension("sessions.db");
        if session_path.exists() {
            println!("  Sessions: {}", session_path.display());
        } else {
            println!("  Sessions: not available");
        }
    }

    Ok(())
}

// ── Merge ──────────────────────────────────────────────────────────

pub async fn handle_db_merge(
    system: &System,
    sources: &[String],
    into: &str,
    on_duplicate: &str,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let strategy = DuplicateStrategy::parse(on_duplicate).ok_or_else(|| {
        format!(
            "Invalid duplicate strategy '{}'. Use: keep-newest, keep-first, keep-all",
            on_duplicate
        )
    })?;

    let result = system
        .bank_manager
        .merge_sources(sources, into, strategy, dry_run)
        .await?;

    if dry_run {
        println!("Dry-run merge into '{}':", into);
    } else {
        println!("Merged into '{}':", into);
    }
    println!(
        "  Imported: {}, Skipped duplicates: {}, Total: {}",
        result.imported, result.skipped_duplicates, result.total_after_merge
    );
    if !result.sources.is_empty() {
        println!("  Per-source breakdown:");
        for (src, count) in &result.sources {
            println!("    {}: {} imported", src, count);
        }
    }

    Ok(())
}

// ── Check ──────────────────────────────────────────────────────────

pub async fn handle_db_check(
    system: &System,
    bank: Option<&str>,
    file: Option<&std::path::Path>,
    all: bool,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if all {
        let banks = system.bank_manager.list_banks().await?;
        let mut any_issues = false;
        for info in &banks {
            println!("Checking bank '{}'...", info.name);
            match system.bank_manager.check_bank(&info.name).await {
                Ok(report) => {
                    print_check_report(&info.name, &report, verbose);
                    if !report.is_clean() {
                        any_issues = true;
                    }
                }
                Err(e) => {
                    eprintln!("  Error loading bank '{}': {}", info.name, e);
                    any_issues = true;
                }
            }
        }
        if !any_issues {
            println!("All banks are clean.");
        }
    } else if let Some(path) = file {
        println!("Checking file {}...", path.display());
        let report = system.bank_manager.check_file(path).await?;
        print_check_report(&path.display().to_string(), &report, verbose);
    } else {
        let name = bank.unwrap_or("default");
        println!("Checking bank '{}'...", name);
        let report = system.bank_manager.check_bank(name).await?;
        print_check_report(name, &report, verbose);
    }

    Ok(())
}

fn print_check_report(
    _name: &str,
    report: &llm_mem::consistency::ConsistencyReport,
    verbose: bool,
) {
    println!(
        "  {} memories, {} errors, {} warnings, {} info",
        report.total_memories, report.errors, report.warnings, report.infos
    );

    if report.is_clean() {
        println!("  Status: CLEAN");
        return;
    }

    // Group by kind for summary
    let mut by_kind: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for issue in &report.issues {
        *by_kind.entry(issue.kind.as_str()).or_default() += 1;
    }
    println!("  Issues by type:");
    for (kind, count) in &by_kind {
        println!("    {}: {}", kind, count);
    }

    if verbose {
        println!("  Details:");
        for issue in &report.issues {
            let sev = match issue.severity {
                IssueSeverity::Error => "ERR ",
                IssueSeverity::Warning => "WARN",
                IssueSeverity::Info => "INFO",
            };
            println!(
                "    [{}] {} {} — {}",
                sev,
                &issue.memory_id[..8.min(issue.memory_id.len())],
                issue.kind,
                issue.message
            );
        }
    }
}

// ── Export JSONL ───────────────────────────────────────────────────

pub async fn handle_db_export_jsonl(
    system: &System,
    bank: &str,
    output: &std::path::Path,
    include_sessions: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let result = system
        .bank_manager
        .export_bank_jsonl(bank, output)
        .await?;

    println!("Exported bank '{}' to JSONL {}", bank, result.path.display());
    println!(
        "  Memories: {}, Format version: {}, SHA-256: {}",
        result.memory_count,
        llm_mem::DATA_FORMAT_VERSION,
        &result.sha256[..16.min(result.sha256.len())]
    );

    // Copy session DB if requested and available
    if include_sessions {
        let session_src = system
            .bank_manager
            .banks_dir()
            .join(format!("{}.sessions.db", bank));
        if session_src.exists() {
            let session_dest = output.with_extension("sessions.db");
            tokio::fs::copy(&session_src, &session_dest).await.map_err(|e| {
                format!(
                    "Failed to copy session DB from {} to {}: {}",
                    session_src.display(),
                    session_dest.display(),
                    e
                )
            })?;
            println!("  Sessions: {}", session_dest.display());
        } else {
            println!("  Sessions: not available (no .sessions.db found)");
        }
    }

    Ok(())
}

// ── Import JSONL ───────────────────────────────────────────────────

pub async fn handle_db_import(
    system: &System,
    bank: &str,
    input: &std::path::Path,
    strip_embeddings: bool,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if dry_run {
        let preview = system
            .bank_manager
            .preview_jsonl_import(input)
            .await?;

        println!("Preview import of '{}' into bank '{}'", input.display(), bank);
        println!("  Format version: {} (app: {})", preview.format_version, preview.app_version);
        println!("  Memories in file: {}", preview.memory_count);
        println!(
            "  Embedding dimension: {} (current: {})",
            preview.embedding_dimension,
            preview.current_dimension
        );
        if preview.dimension_mismatch {
            println!("  ⚠️  Dimension mismatch — embeddings will be stripped on import");
        }
        if !preview.parse_errors.is_empty() {
            println!("  ⚠️  Parse errors: {}", preview.parse_errors.len());
            for err in &preview.parse_errors {
                println!("    {}", err);
            }
        }
        println!(
            "  Estimated to import: {} memories",
            preview.memory_count
        );
        return Ok(());
    }

    let result = system
        .bank_manager
        .import_bank_jsonl(bank, input, strip_embeddings)
        .await?;

    println!("Imported '{}' into bank '{}'", input.display(), bank);
    println!(
        "  Imported: {}, Skipped duplicates: {}, Stripped embeddings: {}",
        result.imported,
        result.skipped_duplicates,
        result.stripped_embeddings
    );
    if !result.parse_errors.is_empty() {
        println!("  ⚠️  Warnings: {}", result.parse_errors.len());
        for err in &result.parse_errors {
            println!("    {}", err);
        }
    }
    if result.stripped_embeddings > 0 {
        println!("  ⚠️  {} memories need re-embedding. Run the abstraction pipeline or `db fix` to regenerate.", result.stripped_embeddings);
    }

    Ok(())
}

// ── Fix ────────────────────────────────────────────────────────────

pub async fn handle_db_fix(
    system: &System,
    bank: &str,
    fix_kinds: &[String],
    dry_run: bool,
    no_backup: bool,
    purge: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Parse requested fix kinds (empty = all)
    let kinds: Option<Vec<IssueKind>> = if fix_kinds.is_empty() {
        None
    } else {
        let mut parsed = Vec::new();
        for s in fix_kinds {
            match IssueKind::parse(s) {
                Some(k) => parsed.push(k),
                None => {
                    return Err(format!(
                        "Unknown fix kind '{}'. Valid: {}",
                        s,
                        IssueKind::all()
                            .iter()
                            .map(|k| k.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                    .into());
                }
            }
        }
        Some(parsed)
    };

    if dry_run {
        // In dry-run mode, just run check and show what would be fixed
        let report = system.bank_manager.check_bank(bank).await?;
        let all_kinds = IssueKind::all();
        let effective_kinds: std::collections::HashSet<&IssueKind> = kinds
            .as_ref()
            .map(|k| k.iter().collect())
            .unwrap_or_else(|| all_kinds.iter().collect());

        let fixable: Vec<_> = report
            .issues
            .iter()
            .filter(|i| effective_kinds.contains(&i.kind))
            .collect();

        println!("Dry-run fix for bank '{}':", bank);
        println!("  {} issue(s) would be addressed:", fixable.len());
        for issue in &fixable {
            println!(
                "    [{}] {} — {}",
                issue.severity, issue.kind, issue.message
            );
        }
        return Ok(());
    }

    // Auto-backup before fixing (unless --no-backup)
    if !no_backup {
        let backup_dir = system.bank_manager.banks_dir().join("backups");
        match system.bank_manager.backup_bank(bank, &backup_dir).await {
            Ok((path, _)) => println!("Auto-backup: {}", path.display()),
            Err(e) => {
                eprintln!("Warning: auto-backup failed: {}. Proceeding anyway.", e);
            }
        }
    }

    let report = system
        .bank_manager
        .fix_bank(bank, kinds.as_deref(), purge)
        .await?;

    println!("Fix results for bank '{}':", bank);
    println!(
        "  Fixed: {}, Deleted: {}, Skipped: {}",
        report.fixed, report.deleted, report.skipped
    );
    if !report.details.is_empty() {
        println!("  Details:");
        for detail in &report.details {
            println!("    {}", detail);
        }
    }

    Ok(())
}
