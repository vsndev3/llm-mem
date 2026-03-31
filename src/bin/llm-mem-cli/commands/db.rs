use llm_mem::{
    consistency::{IssueSeverity, IssueKind},
    memory_bank::DuplicateStrategy,
    System,
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
