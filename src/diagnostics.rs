//! End-to-end configuration health checks.
//!
//! Verifies a [`Config`] is not just syntactically valid but actually
//! operational: storage is reachable, the LLM/embed client can be constructed,
//! and (when opted in) the live backend responds to a tiny ping.
//!
//! Designed to be surfaced through every entry point:
//! - Library: [`run_health_check`]
//! - CLI: `llm-mem health-check`
//! - MCP: `health_check` tool
//!
//! All three call into the same function so the report shape is consistent
//! regardless of how the check was triggered.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::time::timeout;

use crate::config::{ApiDialect, Config, LLMBackend, RequestFormat};
use crate::error::{MemoryError, Result};
use crate::llm::{EmbedPurpose, create_llm_client};

/// Status of an individual check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    /// Check passed.
    Pass,
    /// Check failed (see `detail` for the error).
    Fail,
    /// Check was deliberately not run (e.g. feature disabled, --skip-live).
    Skip,
}

impl CheckStatus {
    /// `true` only for [`CheckStatus::Pass`].
    pub fn is_pass(self) -> bool {
        matches!(self, CheckStatus::Pass)
    }
}

/// A single named check in a [`HealthReport`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Stable name, e.g. `"config.load"`, `"embedding.live"`.
    pub name: String,
    /// Human-readable category, e.g. `"config"`, `"storage"`, `"embedding"`.
    pub category: String,
    pub status: CheckStatus,
    /// Short human-readable description of what was checked.
    pub detail: String,
    /// Wall-clock duration of the check.
    pub duration_ms: u64,
}

impl CheckResult {
    fn pass(
        category: impl Into<String>,
        name: impl Into<String>,
        detail: impl Into<String>,
        duration: Duration,
    ) -> Self {
        Self {
            name: name.into(),
            category: category.into(),
            status: CheckStatus::Pass,
            detail: detail.into(),
            duration_ms: duration.as_millis() as u64,
        }
    }

    fn fail(
        category: impl Into<String>,
        name: impl Into<String>,
        detail: impl Into<String>,
        duration: Duration,
    ) -> Self {
        Self {
            name: name.into(),
            category: category.into(),
            status: CheckStatus::Fail,
            detail: detail.into(),
            duration_ms: duration.as_millis() as u64,
        }
    }

    fn skip(
        category: impl Into<String>,
        name: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            category: category.into(),
            status: CheckStatus::Skip,
            detail: detail.into(),
            duration_ms: 0,
        }
    }
}

/// Summary of all checks performed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Resolved backend (Local / API / APILLMLocalEmbed / LocalLLMAPIEmbed).
    pub backend: String,
    /// `true` iff every non-skipped check passed.
    pub healthy: bool,
    /// LLM model identifier from config.
    pub llm_model: String,
    /// Embedding model identifier from config.
    pub embedding_model: String,
    /// `true` if live checks were actually run.
    pub live_run: bool,
    /// Per-check results in execution order.
    pub checks: Vec<CheckResult>,
}

impl HealthReport {
    /// Count of checks in each status.
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut p = 0;
        let mut f = 0;
        let mut s = 0;
        for c in &self.checks {
            match c.status {
                CheckStatus::Pass => p += 1,
                CheckStatus::Fail => f += 1,
                CheckStatus::Skip => s += 1,
            }
        }
        (p, f, s)
    }

    /// First failing check, if any.
    pub fn first_failure(&self) -> Option<&CheckResult> {
        self.checks.iter().find(|c| c.status == CheckStatus::Fail)
    }
}

/// Options controlling which checks run.
#[derive(Debug, Clone)]
pub struct HealthCheckOptions {
    /// If `true`, perform a tiny `embed("ping")` and `complete("...pong")`.
    /// Default: `false` to avoid surprising API token consumption.
    pub live: bool,
    /// Timeout for the embedding live check.
    pub embed_timeout: Duration,
    /// Timeout for the LLM live check.
    pub llm_timeout: Duration,
    /// `true` => only run the embedding live check (if `live` is `true`).
    pub embed_only: bool,
    /// `true` => only run the LLM live check (if `live` is `true`).
    pub llm_only: bool,
}

impl Default for HealthCheckOptions {
    fn default() -> Self {
        Self {
            live: false,
            embed_timeout: Duration::from_secs(15),
            llm_timeout: Duration::from_secs(30),
            embed_only: false,
            llm_only: false,
        }
    }
}

impl HealthCheckOptions {
    fn should_run_llm(&self) -> bool {
        self.live && !self.embed_only
    }

    fn should_run_embed(&self) -> bool {
        self.live && !self.llm_only
    }
}

/// Run the full health-check suite against a config.
///
/// Always non-mutating on the config: it does not touch the banks directory
/// for writes beyond a probe file, and live checks use throwaway prompts
/// (`"ping"`, `"...pong"`).
///
/// Returns a [`HealthReport`] even if individual checks fail — callers
/// inspect `report.healthy` rather than treating `Err` as a hard failure.
pub async fn run_health_check(config: &Config, opts: &HealthCheckOptions) -> Result<HealthReport> {
    let backend = config.effective_backend();
    let mut checks: Vec<CheckResult> = Vec::new();

    // 1. Config-level (already validated by Config::load, but re-check here
    //    for callers that hand us a Config that came from somewhere else).
    checks.push(check_config(config));

    // 2. Build features vs provider.
    checks.push(check_build_features(config));

    // 3. Credentials present.
    checks.push(check_api_credentials(config));

    // 4. Storage writable.
    checks.push(check_storage(config).await);

    // 5. LLM client construction (catches misconfigured base URL, missing
    //    model, etc. without making a network call).
    let llm_client = match create_llm_client(config).await {
        Ok(c) => {
            checks.push(CheckResult::pass(
                "client",
                "llm.construct",
                format!("LLM client built for backend {:?}", backend),
                Duration::ZERO,
            ));
            Some(c)
        }
        Err(e) => {
            checks.push(CheckResult::fail(
                "client",
                "llm.construct",
                format!("Failed to build LLM client: {e}"),
                Duration::ZERO,
            ));
            None
        }
    };
    let llm_client_ref = llm_client.as_deref();

    // 6. Live embed ping.
    if opts.should_run_embed() {
        if let Some(client) = llm_client_ref {
            checks.push(check_embed_live(client, opts.embed_timeout).await);
        }
    } else if opts.live {
        checks.push(CheckResult::skip(
            "embedding",
            "embedding.live",
            "Skipped (--llm-only)",
        ));
    } else {
        checks.push(CheckResult::skip(
            "embedding",
            "embedding.live",
            "Skipped (pass --live to execute a ping embedding)",
        ));
    }

    // 7. Live LLM ping.
    if opts.should_run_llm() {
        if let Some(client) = llm_client_ref {
            checks.push(check_llm_live(client, opts.llm_timeout).await);
        }
    } else if opts.live {
        checks.push(CheckResult::skip(
            "llm",
            "llm.live",
            "Skipped (--embed-only)",
        ));
    } else {
        checks.push(CheckResult::skip(
            "llm",
            "llm.live",
            "Skipped (pass --live to execute a ping completion)",
        ));
    }

    let healthy = checks.iter().all(|c| c.status != CheckStatus::Fail);

    Ok(HealthReport {
        backend: format!("{backend:?}"),
        healthy,
        llm_model: config.effective_model().to_string(),
        embedding_model: config.embedding.model.clone(),
        live_run: opts.live,
        checks,
    })
}

fn check_config(config: &Config) -> CheckResult {
    let start = Instant::now();
    // Config::validate is the source of truth; it already runs at load time.
    // Re-running here is cheap and means a Config built in-memory is also covered.
    match config.validate() {
        Ok(()) => CheckResult::pass(
            "config",
            "config.validate",
            "Configuration is structurally valid",
            start.elapsed(),
        ),
        Err(e) => CheckResult::fail("config", "config.validate", format!("{e}"), start.elapsed()),
    }
}

fn check_build_features(config: &Config) -> CheckResult {
    let start = Instant::now();
    let backend = config.effective_backend();
    let mut issues: Vec<&'static str> = Vec::new();
    // Always populate (even if empty) so the let-mut lint is satisfied and
    // the cfg branches below are exercised on every build.
    let _ = &mut issues;

    if backend == LLMBackend::Local || backend == LLMBackend::LocalLLMAPIEmbed {
        #[cfg(not(feature = "local-llm"))]
        issues.push("build is missing 'local-llm' feature");
    }
    if backend == LLMBackend::Local || backend == LLMBackend::APILLMLocalEmbed {
        #[cfg(not(feature = "local-embed"))]
        issues.push("build is missing 'local-embed' feature");
    }

    if issues.is_empty() {
        CheckResult::pass(
            "build",
            "build.features",
            "Build features match provider choice",
            start.elapsed(),
        )
    } else {
        CheckResult::fail(
            "build",
            "build.features",
            issues.join("; "),
            start.elapsed(),
        )
    }
}

fn check_api_credentials(config: &Config) -> CheckResult {
    let start = Instant::now();
    let backend = config.effective_backend();
    let mut missing: Vec<&'static str> = Vec::new();

    let llm_is_api = matches!(
        backend,
        LLMBackend::API | LLMBackend::APILLMLocalEmbed | LLMBackend::LocalLLMAPIEmbed
    );
    let embed_is_api = matches!(backend, LLMBackend::API | LLMBackend::LocalLLMAPIEmbed);

    if llm_is_api && config.llm.api_key.trim().is_empty() {
        missing.push("llm.api_key (or LLM_MEM_LLM_API_KEY env var)");
    }
    if embed_is_api && config.embedding.api_key.trim().is_empty() {
        missing.push("embedding.api_key (or LLM_MEM_EMBEDDING_API_KEY env var)");
    }

    if missing.is_empty() {
        CheckResult::pass(
            "credentials",
            "credentials.api_keys",
            "Required API keys are set",
            start.elapsed(),
        )
    } else {
        CheckResult::fail(
            "credentials",
            "credentials.api_keys",
            format!("Missing: {}", missing.join(", ")),
            start.elapsed(),
        )
    }
}

async fn check_storage(config: &Config) -> CheckResult {
    let start = Instant::now();
    let path = std::path::PathBuf::from(&config.vector_store.banks_dir);

    // Try to ensure the directory exists.
    if let Err(e) = tokio::fs::create_dir_all(&path).await {
        return CheckResult::fail(
            "storage",
            "storage.banks_dir",
            format!("Cannot create banks dir {}: {e}", path.display()),
            start.elapsed(),
        );
    }

    // Probe write access with a temp file.
    let probe = path.join(".llm-mem-health-check-probe");
    match tokio::fs::write(&probe, b"ok").await {
        Ok(()) => {
            let _ = tokio::fs::remove_file(&probe).await;
            CheckResult::pass(
                "storage",
                "storage.banks_dir",
                format!("Banks dir is writable: {}", path.display()),
                start.elapsed(),
            )
        }
        Err(e) => CheckResult::fail(
            "storage",
            "storage.banks_dir",
            format!("Banks dir {} is not writable: {e}", path.display()),
            start.elapsed(),
        ),
    }
}

async fn check_embed_live(client: &dyn crate::llm::LLMClient, t: Duration) -> CheckResult {
    let start = Instant::now();
    // Two unrelated English phrases. A healthy embedding model will produce
    // vectors whose cosine similarity is strictly between 0 and 1; this
    // catches degenerate models (returning the same vector for everything)
    // and broken models (returning zeros or random noise).
    let probes = ["hello world".to_string(), "goodbye sky".to_string()];

    let result = timeout(t, client.embed_batch(&probes, EmbedPurpose::Query)).await;
    let vecs = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            return CheckResult::fail(
                "embedding",
                "embedding.live",
                format!("Embedding backend error: {e}"),
                start.elapsed(),
            );
        }
        Err(_) => {
            return CheckResult::fail(
                "embedding",
                "embedding.live",
                format!("Embedding backend timed out after {}s", t.as_secs()),
                start.elapsed(),
            );
        }
    };

    if vecs.len() != 2 {
        return CheckResult::fail(
            "embedding",
            "embedding.live",
            format!("Expected 2 embeddings, got {}", vecs.len()),
            start.elapsed(),
        );
    }

    if let Err(e) = validate_embedding(&vecs[0], "probe[0]=\"hello world\"") {
        return CheckResult::fail("embedding", "embedding.live", e, start.elapsed());
    }
    if let Err(e) = validate_embedding(&vecs[1], "probe[1]=\"goodbye sky\"") {
        return CheckResult::fail("embedding", "embedding.live", e, start.elapsed());
    }
    if vecs[0].len() != vecs[1].len() {
        return CheckResult::fail(
            "embedding",
            "embedding.live",
            format!(
                "Dimension mismatch: probe[0]={}, probe[1]={}",
                vecs[0].len(),
                vecs[1].len()
            ),
            start.elapsed(),
        );
    }

    let sim = match cosine_similarity(&vecs[0], &vecs[1]) {
        Some(s) => s,
        None => {
            return CheckResult::fail(
                "embedding",
                "embedding.live",
                "Could not compute cosine similarity (zero-norm vector)",
                start.elapsed(),
            );
        }
    };

    if !sim.is_finite() {
        return CheckResult::fail(
            "embedding",
            "embedding.live",
            format!("Cosine similarity is not finite: {sim}"),
            start.elapsed(),
        );
    }
    if sim >= 1.0 {
        return CheckResult::fail(
            "embedding",
            "embedding.live",
            format!(
                "Cosine similarity = {sim:.4} (== 1.0 means the model returns the same vector for every input)"
            ),
            start.elapsed(),
        );
    }
    if sim <= 0.0 {
        return CheckResult::fail(
            "embedding",
            "embedding.live",
            format!(
                "Cosine similarity = {sim:.4} (<= 0.0 means the model output is not semantically meaningful)"
            ),
            start.elapsed(),
        );
    }

    CheckResult::pass(
        "embedding",
        "embedding.live",
        format!(
            "dim={} cos_sim(\"hello world\", \"goodbye sky\") = {:.4}",
            vecs[0].len(),
            sim
        ),
        start.elapsed(),
    )
}

/// Cosine similarity between two same-length f32 vectors.
///
/// Returns `None` if either vector is zero-norm, in which case similarity is
/// undefined.
fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f64> {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        return None;
    }
    Some(dot / denom)
}

/// Reject obvious-bad embedding vectors:
/// - empty
/// - contains non-finite values (NaN, +/-Inf)
/// - all zeros (norm below epsilon)
fn validate_embedding(v: &[f32], label: &str) -> std::result::Result<(), String> {
    if v.is_empty() {
        return Err(format!("{label}: empty vector"));
    }
    let mut sumsq = 0.0_f64;
    for (i, x) in v.iter().enumerate() {
        if !x.is_finite() {
            return Err(format!("{label}: non-finite value at index {i} ({x})"));
        }
        sumsq += (*x as f64) * (*x as f64);
    }
    if sumsq.sqrt() < 1e-6 {
        return Err(format!("{label}: vector is all zeros (norm below 1e-6)"));
    }
    Ok(())
}

async fn check_llm_live(client: &dyn crate::llm::LLMClient, t: Duration) -> CheckResult {
    let start = Instant::now();
    let probe = "Reply with the single word: pong";
    match timeout(t, client.complete(probe)).await {
        Ok(Ok(s)) if s.trim().is_empty() => CheckResult::fail(
            "llm",
            "llm.live",
            "LLM backend returned an empty response",
            start.elapsed(),
        ),
        Ok(Ok(s)) => CheckResult::pass(
            "llm",
            "llm.live",
            format!("LLM backend responded ({} chars)", s.len()),
            start.elapsed(),
        ),
        Ok(Err(e)) => CheckResult::fail(
            "llm",
            "llm.live",
            format!("LLM backend error: {e}"),
            start.elapsed(),
        ),
        Err(_) => CheckResult::fail(
            "llm",
            "llm.live",
            format!("LLM backend timed out after {}s", t.as_secs()),
            start.elapsed(),
        ),
    }
}

/// Format a [`HealthReport`] as a human-readable table suitable for a TTY.
///
/// Used by the CLI; the MCP tool returns the raw [`HealthReport`] as JSON.
pub fn format_report_table(report: &HealthReport) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s, "Health check (backend: {})", report.backend);
    let _ = writeln!(s, "  llm_model:       {}", report.llm_model);
    let _ = writeln!(s, "  embedding_model: {}", report.embedding_model);
    let _ = writeln!(s, "  live_run:        {}", report.live_run);
    let (p, f, sk) = report.counts();
    let _ = writeln!(s, "  results:         {p} pass, {f} fail, {sk} skip");
    let _ = writeln!(s);

    // Column widths.
    let name_w = report
        .checks
        .iter()
        .map(|c| c.name.len())
        .max()
        .unwrap_or(12)
        .max(12);
    let status_w = 6;
    let dur_w = 7;
    let _ = writeln!(
        s,
        "  {:<name_w$}  {:<status_w$}  {:<dur_w$}  detail",
        "check", "status", "ms"
    );
    let _ = writeln!(
        s,
        "  {:-<name_w$}  {:-<status_w$}  {:-<dur_w$}  ------",
        "", "", ""
    );
    for c in &report.checks {
        let status = match c.status {
            CheckStatus::Pass => "PASS",
            CheckStatus::Fail => "FAIL",
            CheckStatus::Skip => "SKIP",
        };
        let _ = writeln!(
            s,
            "  {:<name_w$}  {:<status_w$}  {:<dur_w$}  {}",
            c.name, status, c.duration_ms, c.detail
        );
    }

    let _ = writeln!(s);
    if report.healthy {
        let _ = writeln!(s, "OK — system is healthy");
    } else if let Some(f) = report.first_failure() {
        let _ = writeln!(s, "UNHEALTHY — first failure: {}: {}", f.name, f.detail);
    } else {
        let _ = writeln!(s, "UNHEALTHY");
    }
    s
}

// Reference unused types to silence dead-code warnings in case future code
// wants to branch on these (also documents the exhaustive coverage).
#[allow(dead_code)]
fn _exhaustive_request_formats(r: RequestFormat) -> &'static str {
    match r {
        RequestFormat::Auto => "auto",
        RequestFormat::Rig => "rig",
        RequestFormat::Raw => "raw",
    }
}

#[allow(dead_code)]
fn _exhaustive_dialects(d: ApiDialect) -> &'static str {
    match d {
        ApiDialect::OpenAIChat => "openai_chat",
        ApiDialect::OpenAICompletion => "openai_completion",
        ApiDialect::Anthropic => "anthropic",
        ApiDialect::OllamaChat => "ollama_chat",
        ApiDialect::OllamaCompletion => "ollama_completion",
        ApiDialect::Custom => "custom",
    }
}

#[allow(dead_code)]
const _MEM_ERROR_USED: fn(MemoryError) = |_| {};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn write_config_to_tempfile(content: &str) -> (tempfile::NamedTempFile, std::path::PathBuf) {
        use std::io::Write as _;
        let mut f = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("create temp config file");
        f.write_all(content.as_bytes()).expect("write config");
        f.flush().unwrap();
        let dir = tempfile::tempdir().expect("create temp banks dir");
        let path = dir.keep();
        (f, path)
    }

    const API_OK: &str = r#"
[llm]
provider = "api"
api_key = "sk-test"
api_url = "https://api.example.com/v1"
model = "gpt-test"
[embedding]
provider = "api"
api_key = "sk-test"
api_url = "https://api.example.com/v1"
model = "text-embedding-test"
"#;

    const API_MISSING_LLM_KEY: &str = r#"
[llm]
provider = "api"
api_key = ""
api_url = "https://api.example.com/v1"
model = "gpt-test"
[embedding]
provider = "api"
api_key = "sk-test"
api_url = "https://api.example.com/v1"
model = "text-embedding-test"
"#;

    #[test]
    fn static_checks_pass_for_well_formed_api_config() {
        let (file, banks_dir) = write_config_to_tempfile(API_OK);
        let mut config = Config::load(file.path()).expect("valid api config should load");
        config.vector_store.banks_dir = banks_dir.display().to_string();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let report = run_health_check(&config, &HealthCheckOptions::default())
                .await
                .unwrap();
            assert!(report.healthy, "report not healthy: {report:?}");
            let (p, f, sk) = report.counts();
            assert_eq!(f, 0, "no failures expected: {report:?}");
            assert!(p >= 4, "expected >=4 passing static checks, got {p}");
            assert!(sk >= 2, "expected >=2 skipped live checks, got {sk}");
            assert!(!report.live_run);
            for c in &report.checks {
                if c.name == "embedding.live" || c.name == "llm.live" {
                    assert_eq!(c.status, CheckStatus::Skip);
                }
            }
        });
    }

    #[test]
    fn missing_api_key_is_reported_as_failure() {
        // Config::load() calls validate() which will already reject this
        // (api_key must be non-empty for an api provider), so the load itself
        // fails. Verify that's the user-visible behavior: a missing key is
        // caught at config-load time, before health-check ever runs.
        let (file, _banks_dir) = write_config_to_tempfile(API_MISSING_LLM_KEY);
        let res = Config::load(file.path());
        assert!(
            res.is_err(),
            "Config::load must reject an api config with empty api_key"
        );
        let err = res.err().unwrap().to_string();
        assert!(
            err.contains("api_key") || err.contains("API key"),
            "error should mention api_key: {err}"
        );
    }

    #[test]
    fn report_table_includes_status_and_detail() {
        let (file, banks_dir) = write_config_to_tempfile(API_OK);
        let mut config = Config::load(file.path()).expect("valid api config should load");
        config.vector_store.banks_dir = banks_dir.display().to_string();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let report = run_health_check(&config, &HealthCheckOptions::default())
                .await
                .unwrap();
            let s = format_report_table(&report);
            assert!(s.contains("Health check"));
            assert!(s.contains("backend:"));
            assert!(s.contains("embedding.live"));
            assert!(s.contains("llm.live"));
            assert!(s.contains("OK") || s.contains("UNHEALTHY"));
        });
    }

    #[test]
    fn health_check_options_should_run_flags() {
        let mut opts = HealthCheckOptions::default();
        assert!(!opts.should_run_llm());
        assert!(!opts.should_run_embed());

        opts.live = true;
        assert!(opts.should_run_llm());
        assert!(opts.should_run_embed());

        opts.embed_only = true;
        assert!(!opts.should_run_llm());
        assert!(opts.should_run_embed());

        opts.embed_only = false;
        opts.llm_only = true;
        assert!(opts.should_run_llm());
        assert!(!opts.should_run_embed());
    }

    // --- cosine_similarity ---

    #[test]
    fn cosine_similarity_identical_unit_vectors_is_one() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let s = cosine_similarity(&a, &b).unwrap();
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_orthogonal_is_zero() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let s = cosine_similarity(&a, &b).unwrap();
        assert!(s.abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_opposite_is_negative_one() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let s = cosine_similarity(&a, &b).unwrap();
        assert!((s + 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_zero_vector_is_none() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!(cosine_similarity(&a, &b).is_none());
        assert!(cosine_similarity(&b, &a).is_none());
    }

    #[test]
    fn cosine_similarity_scale_invariant() {
        let a = vec![0.3, 0.4]; // norm 0.5
        let b = vec![6.0, 8.0]; // norm 10.0, same direction
        let s = cosine_similarity(&a, &b).unwrap();
        assert!((s - 1.0).abs() < 1e-9, "expected 1.0, got {s}");
    }

    #[test]
    fn cosine_similarity_realistic_embeddings() {
        // Simulate two moderately-similar embeddings in 384-dim space.
        let mut a: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = a
            .iter()
            .enumerate()
            .map(|(i, x)| x + ((i as f32) * 0.02).cos() * 0.1)
            .collect();
        // Normalize both to unit length for a clean check.
        let na = (a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>()).sqrt();
        let nb = (b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>()).sqrt();
        for x in &mut a {
            *x = (*x as f64 / na) as f32;
        }
        let b: Vec<f32> = b.iter().map(|x| (*x as f64 / nb) as f32).collect();
        let s = cosine_similarity(&a, &b).unwrap();
        assert!(s > 0.0 && s < 1.0, "expected 0 < sim < 1, got {s}");
    }

    // --- validate_embedding ---

    #[test]
    fn validate_embedding_rejects_empty() {
        let v: Vec<f32> = vec![];
        let err = validate_embedding(&v, "test").unwrap_err();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn validate_embedding_rejects_nan() {
        let v = vec![0.1, 0.2, f32::NAN, 0.4];
        let err = validate_embedding(&v, "test").unwrap_err();
        assert!(err.contains("non-finite"), "got: {err}");
        assert!(err.contains("index 2"), "got: {err}");
    }

    #[test]
    fn validate_embedding_rejects_infinity() {
        let v = vec![0.1, 0.2, f32::INFINITY, 0.4];
        let err = validate_embedding(&v, "test").unwrap_err();
        assert!(err.contains("non-finite"), "got: {err}");
    }

    #[test]
    fn validate_embedding_rejects_all_zeros() {
        let v = vec![0.0_f32; 384];
        let err = validate_embedding(&v, "test").unwrap_err();
        assert!(err.contains("all zeros"), "got: {err}");
    }

    #[test]
    fn validate_embedding_accepts_realistic_vector() {
        let v: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
        assert!(validate_embedding(&v, "test").is_ok());
    }

    #[test]
    fn validate_embedding_accepts_tiny_but_nonzero() {
        // Norm well above 1e-6 threshold.
        let v = vec![1e-4_f32; 10000];
        assert!(validate_embedding(&v, "test").is_ok());
    }
}
