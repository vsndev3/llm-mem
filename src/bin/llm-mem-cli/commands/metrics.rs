use crate::OutputFormat;
use crate::System;

pub async fn handle_metrics(
    system: &System,
    reset: bool,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = system.metrics.snapshot();

    if format == OutputFormat::Json {
        let json = serde_json::to_string_pretty(&snapshot).unwrap_or_else(|_| "{}".to_string());
        println!("{}", json);
    } else {
        print_table(&snapshot);
    }

    if reset {
        system.metrics.reset();
        eprintln!("Metrics cleared");
    }

    Ok(())
}

fn fmt_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.1}s", ms / 1000.0)
    } else {
        format!("{:.0}ms", ms)
    }
}

fn print_latency_table(
    title: &str,
    entries: &std::collections::HashMap<String, llm_mem::memory::metrics::LatencyStats>,
) {
    let mut out = String::new();
    let hdr = |s: &str| format!("\x1b[1m{}\x1b[0m", s);

    out.push_str(&format!("{}:\n", hdr(title)));
    if entries.is_empty() {
        out.push_str("  (no data yet)\n");
    } else {
        out.push_str("  Phase                  Count    Avg        Min        Max\n");
        out.push_str("  ───────────────────    ─────    ────────   ────────   ────────\n");
        let mut sorted: Vec<_> = entries.iter().collect();
        sorted.sort_by_key(|(k, _)| (*k).clone());
        for (phase, stats) in sorted {
            let min_str = if stats.min_ms == f64::MAX {
                "-".to_string()
            } else {
                fmt_ms(stats.min_ms)
            };
            let max_str = if stats.max_ms == f64::MIN {
                "-".to_string()
            } else {
                fmt_ms(stats.max_ms)
            };
            out.push_str(&format!(
                "  {:<22} {:<7} {:<10} {:<10} {:<10}\n",
                phase,
                stats.count,
                fmt_ms(stats.avg_ms()),
                min_str,
                max_str,
            ));
        }
    }
    out.push('\n');
    print!("{}", out);
}

fn print_table(snapshot: &llm_mem::memory::metrics::MetricsSnapshot) {
    // ── Query latency ──
    print_latency_table("Query Latency", &snapshot.query_latency);

    // ── Ingestion timing ──
    print_latency_table("Ingestion Timing", &snapshot.ingestion_timing);

    // ── Storage Operations ──
    print_latency_table("Storage Operations", &snapshot.storage_timing);

    // ── Cache hits/misses ──
    let hdr = |s: &str| format!("\x1b[1m{}\x1b[0m", s);
    let mut out = String::new();
    out.push_str(&format!("{}:\n", hdr("Cache Performance")));
    if snapshot.cache_hits.is_empty() && snapshot.cache_misses.is_empty() {
        out.push_str("  (no data yet)\n");
    } else {
        out.push_str("  Cache                Hits      Misses    Hit Rate\n");
        out.push_str("  ───────────────────  ───────   ───────   ────────\n");
        for cache_name in ["query_embedding", "query_intent", "layer_manifest"] {
            let hits = *snapshot.cache_hits.get(cache_name).unwrap_or(&0);
            let misses = *snapshot.cache_misses.get(cache_name).unwrap_or(&0);
            let total = hits + misses;
            let rate = if total == 0 {
                "-".to_string()
            } else {
                format!("{:.1}%", (hits as f64 / total as f64) * 100.0)
            };
            out.push_str(&format!(
                "  {:<21} {:<9} {:<9} {:<10}\n",
                cache_name, hits, misses, rate
            ));
        }
    }
    out.push('\n');
    print!("{}", out);

    // ── LLM Operations ──
    out = String::new();
    out.push_str(&format!("{}:\n", hdr("LLM Operations")));
    if snapshot.llm_operations.is_empty() {
        out.push_str("  (no data yet)\n");
    } else {
        for (backend, ops) in &snapshot.llm_operations {
            out.push_str(&format!("  [{}]\n", backend));
            out.push_str("    Operation              Count  Succ  Fail  Avg        Total prompt  Total completion\n");
            out.push_str("    ───────────────────    ───── ───── ───── ──────── ──────────── ────────────────\n");
            let mut sorted: Vec<_> = ops.iter().collect();
            sorted.sort_by_key(|(k, _)| (*k).clone());
            for (op, stats) in sorted {
                out.push_str(&format!(
                    "    {:<22} {:<5} {:<5} {:<5} {:<10} {:<14} {:<18}\n",
                    op,
                    stats.count,
                    stats.success_count,
                    stats.failure_count,
                    fmt_ms(stats.latency.avg_ms()),
                    stats.total_prompt_tokens,
                    stats.total_completion_tokens,
                ));
            }
        }
    }
    out.push('\n');
    print!("{}", out);

    // ── Embedding Operations ──
    out = String::new();
    out.push_str(&format!("{}:\n", hdr("Embedding Operations")));
    if snapshot.embedding_operations.is_empty() {
        out.push_str("  (no data yet)\n");
    } else {
        for (backend, stats) in &snapshot.embedding_operations {
            out.push_str(&format!(
                "  [{}] Count: {}  Success: {}  Failed: {}  Avg: {}  Total texts: {}\n",
                backend,
                stats.count,
                stats.success_count,
                stats.failure_count,
                fmt_ms(stats.latency.avg_ms()),
                stats.total_texts,
            ));
        }
    }
    out.push('\n');
    print!("{}", out);

    // ── Layer distribution ──
    out = String::new();
    out.push_str(&format!("{}:\n", hdr("Layer Distribution")));
    if snapshot.layer_distribution.is_empty() {
        out.push_str("  (no data yet)\n");
    } else {
        out.push_str("  Layer    Count\n");
        out.push_str("  ─────    ─────\n");
        let mut layers: Vec<_> = snapshot.layer_distribution.keys().cloned().collect();
        layers.sort();
        for layer in layers {
            let count = snapshot.layer_distribution[&layer];
            out.push_str(&format!("  {:<8} {:<5}\n", layer, count));
        }
    }
    out.push('\n');
    print!("{}", out);

    // ── Graph refinement ──
    out = String::new();
    out.push_str(&format!("{}:\n", hdr("Graph Refinement")));
    out.push_str(&format!(
        "  Discovered: {}\n  Base:       {}\n",
        snapshot.graph_refinement_discovered, snapshot.graph_refinement_base
    ));
    out.push('\n');
    print!("{}", out);

    // ── Allocation modes ──
    out = String::new();
    out.push_str(&format!("{}:\n", hdr("Allocation Modes")));
    if snapshot.allocation_modes.is_empty() {
        out.push_str("  (no data yet)\n");
    } else {
        out.push_str("  Mode         Count\n");
        out.push_str("  ───────────  ─────\n");
        let mut modes: Vec<_> = snapshot.allocation_modes.iter().collect();
        modes.sort_by_key(|(k, _)| (*k).clone());
        for (mode, count) in &modes {
            out.push_str(&format!("  {:<12} {:<5}\n", mode, count));
        }
    }
    out.push('\n');
    print!("{}", out);

    // ── Summary ──
    out = String::new();
    out.push_str(&format!("{}:\n", hdr("Summary")));
    out.push_str(&format!(
        "  Total queries:     {}\n",
        snapshot.total_queries
    ));
    out.push_str(&format!(
        "  Total results:     {}\n",
        snapshot.total_result_count
    ));
    if snapshot.total_queries > 0 {
        out.push_str(&format!(
            "  Avg results/query: {:.1}\n",
            snapshot.total_result_count as f64 / snapshot.total_queries as f64
        ));
    }
    out.push_str(&format!(
        "  Total LLM tokens:  {} prompt + {} completion = {} total\n",
        snapshot.total_llm_prompt_tokens,
        snapshot.total_llm_completion_tokens,
        snapshot.total_llm_prompt_tokens + snapshot.total_llm_completion_tokens,
    ));
    print!("{}", out);
}
