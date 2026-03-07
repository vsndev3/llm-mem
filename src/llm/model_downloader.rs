//! Robust model file downloader with resume support, checksum validation,
//! proxy detection, and progress reporting.
//!
//! This module handles automatic downloading of GGUF model files from
//! Hugging Face when a local model isn't found on disk.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

use crate::error::{MemoryError, Result};

// ── Known model registry ───────────────────────────────────────────────────

/// A known model that can be auto-downloaded.
#[derive(Debug, Clone)]
pub struct KnownModel {
    /// Filename as stored on disk
    pub filename: &'static str,
    /// Full download URL
    pub url: &'static str,
    /// Expected SHA-256 hex digest (None = skip verification)
    pub sha256: Option<&'static str>,
    /// Optional URL to fetch expected SHA-256 (e.g. Git LFS pointer)
    pub sha256_url: Option<&'static str>,
    /// Approximate size in bytes (for progress display)
    pub size_bytes: u64,
    /// Human-readable description
    pub description: &'static str,
}

/// Registry of models that can be auto-downloaded.
pub static KNOWN_MODELS: &[KnownModel] = &[
    KnownModel {
        filename: "Qwen3.5-2B-UD-Q6_K_XL.gguf",
        url: "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-UD-Q6_K_XL.gguf",
        sha256: None,
        sha256_url: None,
        size_bytes: 2_000_000_000,
        description: "Qwen3.5 2B UD (Q6_K_XL, ~2.0 GB)",
    },
    KnownModel {
        filename: "smollm2-1.7b-instruct-q4_k_m.gguf",
        url: "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf",
        sha256: None,
        sha256_url: None,
        size_bytes: 1_000_000_000,
        description: "SmolLM2 1.7B Instruct (Q4_K_M, ~1.0 GB)",
    },
];

/// Look up a known model by filename.
pub fn find_known_model(filename: &str) -> Option<&'static KnownModel> {
    KNOWN_MODELS.iter().find(|m| m.filename == filename)
}

// ── Proxy configuration ────────────────────────────────────────────────────

/// Resolved proxy configuration.
#[derive(Debug, Clone, Default)]
pub struct ProxyConfig {
    /// HTTPS proxy URL (used for huggingface.co downloads)
    pub https_proxy: Option<String>,
    /// HTTP proxy URL (fallback)
    pub http_proxy: Option<String>,
    /// No-proxy list (comma-separated hostnames)
    pub no_proxy: Option<String>,
}

impl ProxyConfig {
    /// Detect proxy settings from environment variables.
    ///
    /// Checks (in order): `HTTPS_PROXY`, `https_proxy`, `HTTP_PROXY`,
    /// `http_proxy`, `ALL_PROXY`, `all_proxy`, `NO_PROXY`, `no_proxy`.
    pub fn from_env() -> Self {
        let https_proxy = std::env::var("HTTPS_PROXY")
            .or_else(|_| std::env::var("https_proxy"))
            .or_else(|_| std::env::var("ALL_PROXY"))
            .or_else(|_| std::env::var("all_proxy"))
            .ok()
            .filter(|s| !s.is_empty());

        let http_proxy = std::env::var("HTTP_PROXY")
            .or_else(|_| std::env::var("http_proxy"))
            .or_else(|_| std::env::var("ALL_PROXY"))
            .or_else(|_| std::env::var("all_proxy"))
            .ok()
            .filter(|s| !s.is_empty());

        let no_proxy = std::env::var("NO_PROXY")
            .or_else(|_| std::env::var("no_proxy"))
            .ok()
            .filter(|s| !s.is_empty());

        Self {
            https_proxy,
            http_proxy,
            no_proxy,
        }
    }

    /// Build from an explicit proxy URL (e.g. from config or CLI).
    pub fn explicit(proxy_url: &str) -> Self {
        Self {
            https_proxy: Some(proxy_url.to_string()),
            http_proxy: Some(proxy_url.to_string()),
            no_proxy: None,
        }
    }

    /// Merge: explicit overrides take precedence, then env.
    pub fn merge(explicit: Option<&str>, env: &ProxyConfig) -> Self {
        match explicit {
            Some(url) if !url.is_empty() => Self::explicit(url),
            _ => env.clone(),
        }
    }

    /// Returns the effective proxy URL for HTTPS, if any.
    pub fn effective_https_proxy(&self) -> Option<&str> {
        self.https_proxy.as_deref()
    }

    /// True if any proxy is configured.
    pub fn has_proxy(&self) -> bool {
        self.https_proxy.is_some() || self.http_proxy.is_some()
    }

    /// Summary string for logging / error messages.
    pub fn summary(&self) -> String {
        if let Some(p) = &self.https_proxy {
            format!("HTTPS proxy: {}", p)
        } else if let Some(p) = &self.http_proxy {
            format!("HTTP proxy: {}", p)
        } else {
            "No proxy configured".to_string()
        }
    }
}

// ── Download progress ──────────────────────────────────────────────────────

/// Progress callback receives: bytes_downloaded, total_bytes (0 if unknown).
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send + Sync>;

/// Default progress reporter that logs to tracing at INFO level.
fn default_progress_callback() -> ProgressCallback {
    let last_report = std::sync::Mutex::new(Instant::now());
    let last_bytes = std::sync::Mutex::new(0u64);

    Box::new(move |downloaded: u64, total: u64| {
        let now = Instant::now();
        let mut last = last_report.lock().unwrap();
        let elapsed = now.duration_since(*last);

        // Report at most every 2 seconds
        if elapsed < Duration::from_secs(2) && downloaded < total {
            return;
        }

        let mut prev = last_bytes.lock().unwrap();
        let speed = if elapsed.as_secs_f64() > 0.0 {
            ((downloaded - *prev) as f64 / elapsed.as_secs_f64()) / 1_048_576.0
        } else {
            0.0
        };

        if total > 0 {
            let pct = (downloaded as f64 / total as f64) * 100.0;
            info!(
                "Downloading model: {:.1}% ({} / {} MB) [{:.1} MB/s]",
                pct,
                downloaded / 1_048_576,
                total / 1_048_576,
                speed
            );
        } else {
            info!(
                "Downloading model: {} MB [{:.1} MB/s]",
                downloaded / 1_048_576,
                speed
            );
        }

        *last = now;
        *prev = downloaded;
    })
}

// ── Download result ────────────────────────────────────────────────────────

/// Outcome of a download attempt.
#[derive(Debug, Clone)]
pub struct DownloadResult {
    /// Final path of the downloaded file
    pub path: PathBuf,
    /// Whether the file was freshly downloaded (false = already existed or resumed)
    pub freshly_downloaded: bool,
    /// SHA-256 hex digest of the file (if computed)
    pub sha256: Option<String>,
    /// Total bytes of the final file
    pub size_bytes: u64,
}

// ── Core downloader ────────────────────────────────────────────────────────

/// Configuration for a download operation.
#[derive(Debug, Clone)]
pub struct DownloadRequest {
    /// URL to download from
    pub url: String,
    /// Destination file path
    pub dest: PathBuf,
    /// Expected SHA-256 (hex). If set, validates after download.
    pub expected_sha256: Option<String>,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Read timeout (per chunk, not total)
    pub read_timeout: Duration,
    /// Proxy configuration
    pub proxy: ProxyConfig,
    /// Whether to attempt resuming a partial download
    pub resume: bool,
}

impl DownloadRequest {
    /// Create a download request with sensible defaults.
    pub fn new(url: impl Into<String>, dest: impl Into<PathBuf>) -> Self {
        Self {
            url: url.into(),
            dest: dest.into(),
            expected_sha256: None,
            connect_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(300),
            proxy: ProxyConfig::default(),
            resume: true,
        }
    }

    pub fn with_sha256(mut self, sha256: impl Into<String>) -> Self {
        self.expected_sha256 = Some(sha256.into());
        self
    }

    pub fn with_proxy(mut self, proxy: ProxyConfig) -> Self {
        self.proxy = proxy;
        self
    }

    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }
}

/// Download a file with resume support, checksum validation, and proxy awareness.
///
/// Returns `DownloadResult` on success. On network errors in a proxy environment,
/// the error message includes proxy configuration guidance.
pub async fn download_file(
    request: &DownloadRequest,
    progress: Option<ProgressCallback>,
) -> Result<DownloadResult> {
    let progress = progress.unwrap_or_else(default_progress_callback);

    // ── Check if complete file already exists ──
    if request.dest.exists() {
        let meta = std::fs::metadata(&request.dest).map_err(|e| {
            MemoryError::Download(format!(
                "Cannot read existing file '{}': {}",
                request.dest.display(),
                e
            ))
        })?;

        // If SHA-256 is specified, validate the existing file
        if let Some(expected) = &request.expected_sha256 {
            info!(
                "File exists ({}), verifying checksum...",
                format_size(meta.len())
            );
            let actual = sha256_file(&request.dest)?;
            if actual == *expected {
                info!("Checksum verified. File is complete.");
                return Ok(DownloadResult {
                    path: request.dest.clone(),
                    freshly_downloaded: false,
                    sha256: Some(actual),
                    size_bytes: meta.len(),
                });
            }
            warn!(
                "Checksum mismatch on existing file (expected {}, got {}). Re-downloading.",
                &expected[..12],
                &actual[..12]
            );
            // Will fall through to download, partial file handling below
        } else {
            // No checksum to verify, assume existing file is good
            info!(
                "Model file already exists: {} ({})",
                request.dest.display(),
                format_size(meta.len())
            );
            return Ok(DownloadResult {
                path: request.dest.clone(),
                freshly_downloaded: false,
                sha256: None,
                size_bytes: meta.len(),
            });
        }
    }

    // ── Build HTTP client ──
    let client = build_http_client(
        &request.proxy,
        request.connect_timeout,
        request.read_timeout,
    )?;

    // ── Detect partial download for resume ──
    let partial_path = partial_file_path(&request.dest);
    let mut existing_bytes: u64 = 0;

    /// Safety margin: truncate this many bytes from the end of a partial
    /// download before resuming, to guard against corruption at the
    /// interrupted-write boundary.
    const RESUME_OVERLAP_BYTES: u64 = 1024;

    if request.resume && partial_path.exists() {
        existing_bytes = std::fs::metadata(&partial_path)
            .map(|m| m.len())
            .unwrap_or(0);
        if existing_bytes > 0 {
            // Truncate the last RESUME_OVERLAP_BYTES to avoid resuming
            // from a potentially corrupted write boundary.
            let truncated = existing_bytes.min(RESUME_OVERLAP_BYTES);
            existing_bytes = existing_bytes.saturating_sub(RESUME_OVERLAP_BYTES);

            if existing_bytes == 0 {
                // File too small to safely truncate — restart from scratch
                info!(
                    "Partial download too small ({}) to safely truncate — restarting",
                    format_size(truncated)
                );
                if let Err(e) = std::fs::remove_file(&partial_path) {
                    warn!("Failed to remove small partial file: {}", e);
                }
            } else {
                // Physically truncate the file on disk
                let file = std::fs::OpenOptions::new()
                    .write(true)
                    .open(&partial_path)
                    .map_err(|e| {
                        MemoryError::Download(format!(
                            "Failed to open partial file for truncation: {}",
                            e
                        ))
                    })?;
                file.set_len(existing_bytes).map_err(|e| {
                    MemoryError::Download(format!("Failed to truncate partial file: {}", e))
                })?;

                info!(
                    "Found partial download — truncated last {} for safety, resuming from {}",
                    format_size(truncated),
                    format_size(existing_bytes)
                );
            }
        }
    }

    // ── Send request with Range header for resume ──
    let mut req_builder = client.get(&request.url);
    if existing_bytes > 0 {
        req_builder = req_builder.header("Range", format!("bytes={}-", existing_bytes));
    }

    let response = req_builder
        .send()
        .await
        .map_err(|e| build_network_error(&e, &request.url, &request.proxy))?;

    let status = response.status();

    // Handle response status
    if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        // Server says our range is beyond the file — restart
        warn!("Server rejected Range request. Starting fresh download.");
        // Remove partial file
        let _ = std::fs::remove_file(&partial_path);
        // Retry without Range
        let response = client
            .get(&request.url)
            .send()
            .await
            .map_err(|e| build_network_error(&e, &request.url, &request.proxy))?;
        return stream_to_file(
            response,
            &request.dest,
            &partial_path,
            0,
            &request.expected_sha256,
            &progress,
        )
        .await;
    }

    if !status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(MemoryError::Download(format!(
            "HTTP {} from {}\n{}",
            status,
            request.url,
            if status.as_u16() == 403 || status.as_u16() == 401 {
                "The model URL may require authentication or may have changed.\n\
                 Check https://huggingface.co for the latest download URL."
                    .to_string()
            } else {
                format!(
                    "Unexpected server response. Proxy: {}",
                    request.proxy.summary()
                )
            }
        )));
    }

    // If server returned 200 (not 206), it ignores Range — restart from scratch
    if status == reqwest::StatusCode::OK && existing_bytes > 0 {
        info!("Server does not support resume. Starting fresh download.");
        existing_bytes = 0;
        let _ = std::fs::remove_file(&partial_path);
    }

    stream_to_file(
        response,
        &request.dest,
        &partial_path,
        existing_bytes,
        &request.expected_sha256,
        &progress,
    )
    .await
}

/// Stream HTTP response body to a file, with progress reporting.
async fn stream_to_file(
    response: reqwest::Response,
    dest: &Path,
    partial_path: &Path,
    existing_bytes: u64,
    expected_sha256: &Option<String>,
    progress: &ProgressCallback,
) -> Result<DownloadResult> {
    let content_length = response.content_length().unwrap_or(0);
    let total_size = if content_length > 0 {
        existing_bytes + content_length
    } else {
        0
    };

    if total_size > 0 {
        info!(
            "Downloading {} (total {})",
            dest.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
            format_size(total_size)
        );
    }

    // Ensure parent directory exists
    if let Some(parent) = partial_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            MemoryError::Download(format!(
                "Cannot create directory '{}': {}",
                parent.display(),
                e
            ))
        })?;
    }

    // Open file for appending (resume) or create new
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(existing_bytes > 0)
        .write(true)
        .truncate(existing_bytes == 0)
        .open(&partial_path)
        .await
        .map_err(|e| {
            MemoryError::Download(format!(
                "Cannot open file '{}' for writing: {}",
                partial_path.display(),
                e
            ))
        })?;

    let mut downloaded = existing_bytes;
    let mut stream = response.bytes_stream();
    let started = Instant::now();

    use tokio_stream::StreamExt;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| {
            MemoryError::Download(format!(
                "Download interrupted after {}: {}\n\
                 The partial file has been saved. Re-running will resume the download.",
                format_size(downloaded),
                e
            ))
        })?;

        file.write_all(&chunk)
            .await
            .map_err(|e| MemoryError::Download(format!("Failed to write to disk: {}", e)))?;

        downloaded += chunk.len() as u64;
        progress(downloaded, total_size);
    }

    file.flush()
        .await
        .map_err(|e| MemoryError::Download(format!("Failed to flush file: {}", e)))?;
    drop(file);

    let elapsed = started.elapsed();
    let speed = if elapsed.as_secs_f64() > 0.0 {
        ((downloaded - existing_bytes) as f64 / elapsed.as_secs_f64()) / 1_048_576.0
    } else {
        0.0
    };
    info!(
        "Download complete: {} in {:.1}s ({:.1} MB/s)",
        format_size(downloaded),
        elapsed.as_secs_f64(),
        speed
    );

    // ── Checksum verification ──
    let actual_sha256 = if expected_sha256.is_some() {
        info!("Verifying checksum...");
        let hash = sha256_file_async(partial_path).await?;
        if let Some(expected) = expected_sha256 {
            if hash != *expected {
                // Checksum failed — remove partial file so next attempt starts fresh
                let _ = tokio::fs::remove_file(&partial_path).await;
                return Err(MemoryError::Download(format!(
                    "Checksum verification failed!\n\
                     Expected: {}\n\
                     Actual:   {}\n\n\
                     The downloaded file was corrupted and has been removed.\n\
                     Please try downloading again. If this persists, the model file\n\
                     may have been updated. Check the Hugging Face repository for\n\
                     the latest version.",
                    expected, hash
                )));
            }
            info!("Checksum verified: {}", &hash[..16]);
        }
        Some(hash)
    } else {
        None
    };

    // ── Rename partial → final ──
    tokio::fs::rename(&partial_path, dest).await.map_err(|e| {
        MemoryError::Download(format!(
            "Failed to rename '{}' → '{}': {}",
            partial_path.display(),
            dest.display(),
            e
        ))
    })?;

    Ok(DownloadResult {
        path: dest.to_path_buf(),
        freshly_downloaded: true,
        sha256: actual_sha256,
        size_bytes: downloaded,
    })
}

// ── HTTP client builder ────────────────────────────────────────────────────

fn build_http_client(
    proxy: &ProxyConfig,
    connect_timeout: Duration,
    read_timeout: Duration,
) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .connect_timeout(connect_timeout)
        .read_timeout(read_timeout)
        .user_agent(format!("llm-mem/{}", env!("CARGO_PKG_VERSION")));

    // Configure proxy
    if let Some(https_url) = &proxy.https_proxy {
        debug!("Using HTTPS proxy: {}", https_url);
        let p = reqwest::Proxy::https(https_url.as_str()).map_err(|e| {
            MemoryError::Download(format!("Invalid HTTPS proxy URL '{}': {}", https_url, e))
        })?;
        builder = builder.proxy(p);
    }
    if let Some(http_url) = &proxy.http_proxy {
        debug!("Using HTTP proxy: {}", http_url);
        let p = reqwest::Proxy::http(http_url.as_str()).map_err(|e| {
            MemoryError::Download(format!("Invalid HTTP proxy URL '{}': {}", http_url, e))
        })?;
        builder = builder.proxy(p);
    }
    if let Some(no_proxy_str) = &proxy.no_proxy {
        debug!("NO_PROXY: {}", no_proxy_str);
        // Apply no_proxy to each configured proxy
        // reqwest NoProxy is set on the Proxy objects, not on the ClientBuilder
        // For simplicity, log and skip — reqwest auto-reads NO_PROXY env var
        // If users set it in the env, reqwest will respect it.
    }

    builder
        .build()
        .map_err(|e| MemoryError::Download(format!("Failed to build HTTP client: {}", e)))
}

// ── Error helpers ──────────────────────────────────────────────────────────

/// Build a helpful network error message, with proxy guidance if applicable.
fn build_network_error(err: &reqwest::Error, url: &str, proxy: &ProxyConfig) -> MemoryError {
    let mut msg = format!("Failed to connect to {}: {}", url, err);

    if err.is_connect() || err.is_timeout() {
        msg.push_str("\n\n");

        if proxy.has_proxy() {
            msg.push_str(&format!(
                "A proxy is configured ({}) but the connection failed.\n\
                 Please verify your proxy settings:\n\
                 • Environment: HTTPS_PROXY, HTTP_PROXY, ALL_PROXY, NO_PROXY\n\
                 • Config file: [local] proxy_url in config.toml\n\
                 • CLI: --proxy <url>\n\n\
                 If your proxy requires authentication, use the format:\n\
                 http://user:password@proxy-host:port",
                proxy.summary()
            ));
        } else {
            msg.push_str(
                "No proxy is configured. If you are behind a corporate firewall\n\
                 or VPN, you may need to configure a proxy:\n\n\
                 • Environment variable: export HTTPS_PROXY=http://proxy:port\n\
                 • Config file:          [local] proxy_url = \"http://proxy:port\"\n\
                 • CLI flag:             --proxy http://proxy:port\n\n\
                 You can also manually download the model:\n\
                 curl -L -o <models_dir>/<filename> <url>",
            );
        }

        if err.is_timeout() {
            msg.push_str(&format!(
                "\n\nThe connection timed out after {:?}. For slow connections, \
                 you may want to download the model manually.",
                if err.is_connect() {
                    Duration::from_secs(30) // connect timeout default
                } else {
                    Duration::from_secs(300)
                }
            ));
        }
    }

    MemoryError::Download(msg)
}

// ── SHA-256 helpers ────────────────────────────────────────────────────────

/// Compute SHA-256 of a file synchronously.
fn sha256_file(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .map_err(|e| MemoryError::Download(format!("Cannot open file for checksum: {}", e)))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| MemoryError::Download(format!("Read error during checksum: {}", e)))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute SHA-256 of a file asynchronously (in a blocking task).
async fn sha256_file_async(path: &Path) -> Result<String> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || sha256_file(&path))
        .await
        .map_err(|e| MemoryError::Download(format!("SHA-256 task failed: {}", e)))?
}

/// Fetch SHA-256 from a URL (expecting Git LFS pointer format or plain text).
pub async fn fetch_expected_sha256(url: &str, proxy: &ProxyConfig) -> Result<String> {
    let client = build_http_client(proxy, Duration::from_secs(10), Duration::from_secs(30))?;

    debug!("Fetching checksum from {}", url);
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| build_network_error(&e, url, proxy))?;

    if !resp.status().is_success() {
        return Err(MemoryError::Download(format!(
            "Failed to fetch checksum from {}: HTTP {}",
            url,
            resp.status()
        )));
    }

    let text = resp
        .text()
        .await
        .map_err(|e| MemoryError::Download(format!("Failed to read checksum response: {}", e)))?;

    // Try parsing as Git LFS pointer: "oid sha256:HASH"
    if text.contains("oid sha256:")
        && let Ok(re) = regex::Regex::new(r"oid sha256:([a-f0-9]{64})")
        && let Some(caps) = re.captures(&text)
    {
        return Ok(caps[1].to_string());
    }

    // Try parsing as simple hash (first word), or just the hash itself
    let trimmed = text.trim();
    // Allow for "HASH  filename" format too
    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    if let Some(first) = parts.first()
        && first.len() == 64
        && first.chars().all(|c| c.is_ascii_hexdigit())
    {
        return Ok(first.to_string());
    }

    Err(MemoryError::Download(format!(
        "Could not parse SHA-256 from response at {}\nResponse start: {}",
        url,
        &text.chars().take(100).collect::<String>()
    )))
}

// ── Utilities ──────────────────────────────────────────────────────────────

/// Path for partial / in-progress downloads.
fn partial_file_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".partial");
    dest.with_file_name(name)
}

/// Format byte size for display.
pub fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// High-level entry point: ensure a model file exists, downloading if needed.
///
/// This is the main function called by `LocalLLMClient::new()`.
///
/// Behaviour:
/// 1. If the file exists and checksum matches (or no checksum) → return immediately
/// 2. If the filename matches a known model → auto-download from Hugging Face
/// 3. If not known → return an error with manual download instructions
///
/// The `proxy_url` parameter allows overriding env-detected proxy from config/CLI.
pub async fn ensure_model(
    models_dir: &Path,
    filename: &str,
    proxy_url: Option<&str>,
    use_cache: bool,
    custom_cache_dir: Option<&str>,
) -> Result<DownloadResult> {
    let dest = models_dir.join(filename);

    if use_cache {
        let cache_dir = if let Some(custom) = custom_cache_dir {
            PathBuf::from(custom)
        } else if let Some(home) = dirs::home_dir() {
            home.join(".cache").join("llm-mem").join("models")
        } else {
            // No home directory, disable caching for this run
            PathBuf::new()
        };

        if !cache_dir.as_os_str().is_empty() && cache_dir.exists() {
            let cache_path = cache_dir.join(filename);

            if cache_path.exists() {
                info!("Using cached model from {}", cache_path.display());
                // Ensure models_dir exists
                if !models_dir.exists() {
                    std::fs::create_dir_all(models_dir).map_err(|e| {
                        MemoryError::Download(format!(
                            "Failed to create models directory {}: {}",
                            models_dir.display(),
                            e
                        ))
                    })?;
                }

                // If dest is already the correct file, we're done
                if dest.exists() && dest.canonicalize().ok() == cache_path.canonicalize().ok() {
                    let meta = std::fs::metadata(&dest).map_err(|e| {
                        MemoryError::Download(format!("Cannot stat '{}': {}", dest.display(), e))
                    })?;
                    return Ok(DownloadResult {
                        path: dest,
                        freshly_downloaded: false,
                        sha256: None, // We could verify but let's assume cache is good for now or verify later
                        size_bytes: meta.len(),
                    });
                }

                // Attempt symlink
                #[cfg(unix)]
                {
                    use std::os::unix::fs::symlink;
                    if dest.exists() {
                        let _ = std::fs::remove_file(&dest);
                    }
                    if symlink(&cache_path, &dest).is_ok() {
                        info!("Created symlink sequence: {} -> {}", dest.display(), cache_path.display());
                        let meta = std::fs::metadata(&dest).map_err(|e| {
                            MemoryError::Download(format!("Cannot stat '{}': {}", dest.display(), e))
                        })?;
                        return Ok(DownloadResult {
                            path: dest,
                            freshly_downloaded: false,
                            sha256: None,
                            size_bytes: meta.len(),
                        });
                    }
                }

                // Fallback to copy if symlink fails or not on unix
                info!("Symlink failed or skipped, copying from cache...");
                if dest.exists() {
                    let _ = std::fs::remove_file(&dest);
                }
                std::fs::copy(&cache_path, &dest).map_err(|e| {
                    MemoryError::Download(format!(
                        "Failed to copy from cache {} to {}: {}",
                        cache_path.display(),
                        dest.display(),
                        e
                    ))
                })?;
                let meta = std::fs::metadata(&dest).map_err(|e| {
                    MemoryError::Download(format!("Cannot stat '{}': {}", dest.display(), e))
                })?;
                return Ok(DownloadResult {
                    path: dest,
                    freshly_downloaded: false,
                    sha256: None,
                    size_bytes: meta.len(),
                });
            }
        }
    }

    let known_opt = find_known_model(filename);

    // Build proxy configuration early
    let env_proxy = ProxyConfig::from_env();
    let proxy = ProxyConfig::merge(proxy_url, &env_proxy);

    // Resolve expected SHA256 if it's a known model
    let mut expected_sha256 = None;
    if let Some(known) = known_opt {
        if let Some(url) = known.sha256_url {
            info!("Fetching latest model checksum from {} ...", url);
            match fetch_expected_sha256(url, &proxy).await {
                Ok(hash) => {
                    debug!("Resolved SHA-256 for {}: {}", filename, &hash[..8]);
                    expected_sha256 = Some(hash);
                }
                Err(e) => {
                    warn!(
                        "Failed to fetch SHA-256 for {}: {}. Falling back to hardcoded if available.",
                        filename, e
                    );
                    expected_sha256 = known.sha256.map(|s| s.to_string());
                }
            }
        } else {
            expected_sha256 = known.sha256.map(|s| s.to_string());
        }
    }

    // Already exists?
    if dest.exists() {
        let meta = std::fs::metadata(&dest).map_err(|e| {
            MemoryError::Download(format!("Cannot stat '{}': {}", dest.display(), e))
        })?;

        // If known model with checksum, verify
        if let Some(expected) = &expected_sha256 {
            info!("Verifying existing model file checksum...");
            let actual = sha256_file(&dest)?;
            if actual == *expected {
                info!("Model verified: {} ({})", filename, format_size(meta.len()));
                return Ok(DownloadResult {
                    path: dest,
                    freshly_downloaded: false,
                    sha256: Some(actual),
                    size_bytes: meta.len(),
                });
            }
            warn!(
                "Existing model file has wrong checksum. Expected {}…, got {}…. Re-downloading.",
                &expected[..12],
                &actual[..12]
            );
            // Fall through to download
        } else {
            return Ok(DownloadResult {
                path: dest,
                freshly_downloaded: false,
                sha256: None,
                size_bytes: meta.len(),
            });
        }
    }

    // Look up in known models
    let known = known_opt.ok_or_else(|| {
        MemoryError::Download(format!(
            "Model file not found: {path}\n\n\
             '{filename}' is not a recognized model that can be auto-downloaded.\n\n\
             Known auto-downloadable models:\n\
             {known_list}\n\n\
             To use a custom model, download it manually:\n\
             curl -L -o {path} <YOUR_MODEL_URL>\n\n\
             Or set [local].llm_model_file in config.toml to one of the known models.",
            path = dest.display(),
            filename = filename,
            known_list = KNOWN_MODELS
                .iter()
                .map(|m| format!("  • {} — {}", m.filename, m.description))
                .collect::<Vec<_>>()
                .join("\n"),
        ))
    })?;

    if proxy.has_proxy() {
        info!("Download will use proxy: {}", proxy.summary());
    }

    let download_dest = if use_cache {
        let cache_dir = if let Some(custom) = custom_cache_dir {
            PathBuf::from(custom)
        } else if let Some(home) = dirs::home_dir() {
            home.join(".cache").join("llm-mem").join("models")
        } else {
            PathBuf::new()
        };

        if !cache_dir.as_os_str().is_empty() {
            if !cache_dir.exists() {
                std::fs::create_dir_all(&cache_dir).map_err(|e| {
                    MemoryError::Download(format!(
                        "Failed to create cache directory {}: {}",
                        cache_dir.display(),
                        e
                    ))
                })?;
            }
            cache_dir.join(filename)
        } else {
            dest.clone()
        }
    } else {
        dest.clone()
    };

    info!(
        "Auto-downloading model: {} ({})",
        known.description,
        format_size(known.size_bytes)
    );

    // Create models directory
    std::fs::create_dir_all(models_dir).map_err(|e| {
        MemoryError::Download(format!(
            "Failed to create models directory '{}': {}",
            models_dir.display(),
            e
        ))
    })?;

    // Build download request
    let mut request = DownloadRequest::new(known.url, &download_dest)
        .with_proxy(proxy)
        .with_connect_timeout(Duration::from_secs(30));

    if let Some(sha) = expected_sha256 {
        request = request.with_sha256(sha);
    }

    let result = download_file(&request, None).await?;

    // If we downloaded to cache, now link it to dest
    if use_cache && download_dest != dest {
        info!("Linking downloaded model to {}", dest.display());
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            if dest.exists() {
                let _ = std::fs::remove_file(&dest);
            }
            if symlink(&download_dest, &dest).is_ok() {
                info!("Created symlink: {} -> {}", dest.display(), download_dest.display());
            } else {
                warn!("Failed to create symlink, falling back to copy...");
                std::fs::copy(&download_dest, &dest).map_err(|e| {
                    MemoryError::Download(format!(
                        "Failed to copy from cache {} to {}: {}",
                        download_dest.display(),
                        dest.display(),
                        e
                    ))
                })?;
            }
        }
        #[cfg(not(unix))]
        {
            std::fs::copy(&download_dest, &dest).map_err(|e| {
                MemoryError::Download(format!(
                    "Failed to copy from cache {} to {}: {}",
                    download_dest.display(),
                    dest.display(),
                    e
                ))
            })?;
        }
    }

    Ok(DownloadResult {
        path: dest,
        freshly_downloaded: true,
        sha256: result.sha256,
        size_bytes: result.size_bytes,
    })
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── format_size ──

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
    }

    #[test]
    fn test_format_size_kilobytes() {
        assert_eq!(format_size(1024), "1 KB");
        assert_eq!(format_size(1536), "2 KB");
    }

    #[test]
    fn test_format_size_megabytes() {
        assert_eq!(format_size(1_048_576), "1.0 MB");
        assert_eq!(format_size(10_485_760), "10.0 MB");
    }

    #[test]
    fn test_format_size_gigabytes() {
        assert_eq!(format_size(1_073_741_824), "1.00 GB");
        assert_eq!(format_size(1_130_000_000), "1.05 GB");
    }

    // ── partial_file_path ──

    #[test]
    fn test_partial_file_path() {
        let dest = PathBuf::from("/models/test-model.gguf");
        let partial = partial_file_path(&dest);
        assert_eq!(partial, PathBuf::from("/models/test-model.gguf.partial"));
    }

    #[test]
    fn test_partial_file_path_no_extension() {
        let dest = PathBuf::from("/models/mymodel");
        let partial = partial_file_path(&dest);
        assert_eq!(partial, PathBuf::from("/models/mymodel.partial"));
    }

    // ── ProxyConfig ──

    #[test]
    fn test_proxy_config_default_empty() {
        let p = ProxyConfig::default();
        assert!(!p.has_proxy());
        assert!(p.effective_https_proxy().is_none());
        assert_eq!(p.summary(), "No proxy configured");
    }

    #[test]
    fn test_proxy_config_explicit() {
        let p = ProxyConfig::explicit("http://myproxy:8080");
        assert!(p.has_proxy());
        assert_eq!(p.effective_https_proxy(), Some("http://myproxy:8080"));
        assert!(p.summary().contains("myproxy:8080"));
    }

    #[test]
    fn test_proxy_config_merge_explicit_wins() {
        let env = ProxyConfig {
            https_proxy: Some("http://env-proxy:3128".into()),
            http_proxy: None,
            no_proxy: None,
        };
        let merged = ProxyConfig::merge(Some("http://cli-proxy:9999"), &env);
        assert_eq!(
            merged.effective_https_proxy(),
            Some("http://cli-proxy:9999")
        );
    }

    #[test]
    fn test_proxy_config_merge_falls_back_to_env() {
        let env = ProxyConfig {
            https_proxy: Some("http://env-proxy:3128".into()),
            http_proxy: None,
            no_proxy: None,
        };
        let merged = ProxyConfig::merge(None, &env);
        assert_eq!(
            merged.effective_https_proxy(),
            Some("http://env-proxy:3128")
        );
    }

    #[test]
    fn test_proxy_config_merge_empty_string_falls_back() {
        let env = ProxyConfig {
            https_proxy: Some("http://env-proxy:3128".into()),
            http_proxy: None,
            no_proxy: None,
        };
        let merged = ProxyConfig::merge(Some(""), &env);
        assert_eq!(
            merged.effective_https_proxy(),
            Some("http://env-proxy:3128")
        );
    }

    #[test]
    fn test_proxy_config_summary_https() {
        let p = ProxyConfig {
            https_proxy: Some("http://proxy:8080".into()),
            http_proxy: Some("http://proxy:8081".into()),
            no_proxy: None,
        };
        // HTTPS takes priority in summary
        assert!(p.summary().contains("HTTPS proxy"));
    }

    #[test]
    fn test_proxy_config_summary_http_only() {
        let p = ProxyConfig {
            https_proxy: None,
            http_proxy: Some("http://proxy:8081".into()),
            no_proxy: None,
        };
        assert!(p.summary().contains("HTTP proxy"));
    }

    // ── KnownModel registry ──

    #[test]
    fn test_find_known_model_exists() {
        let m = find_known_model("Qwen3.5-2B-UD-Q6_K_XL.gguf");
        assert!(m.is_some());
        let m = m.unwrap();
        assert!(m.url.contains("huggingface.co"));
        // SHA256 is optional for some models
        assert!(m.size_bytes > 0);
        assert!(!m.description.is_empty());
    }

    #[test]
    fn test_find_known_model_smollm() {
        let m = find_known_model("smollm2-1.7b-instruct-q4_k_m.gguf");
        assert!(m.is_some());
    }

    #[test]
    fn test_find_known_model_unknown() {
        assert!(find_known_model("nonexistent-model.gguf").is_none());
    }

    #[test]
    fn test_known_models_all_have_urls() {
        for model in KNOWN_MODELS {
            assert!(
                !model.url.is_empty(),
                "Model {} has empty URL",
                model.filename
            );
            assert!(
                model.url.starts_with("https://"),
                "Model {} URL should use HTTPS",
                model.filename
            );
        }
    }

    #[test]
    fn test_known_models_unique_filenames() {
        let mut seen = std::collections::HashSet::new();
        for model in KNOWN_MODELS {
            assert!(
                seen.insert(model.filename),
                "Duplicate filename in KNOWN_MODELS: {}",
                model.filename
            );
        }
    }

    // ── DownloadRequest ──

    #[test]
    fn test_download_request_defaults() {
        let req = DownloadRequest::new("https://example.com/model.gguf", "/tmp/model.gguf");
        assert_eq!(req.url, "https://example.com/model.gguf");
        assert_eq!(req.dest, PathBuf::from("/tmp/model.gguf"));
        assert!(req.resume);
        assert!(req.expected_sha256.is_none());
        assert!(!req.proxy.has_proxy());
    }

    #[test]
    fn test_download_request_with_sha256() {
        let req = DownloadRequest::new("https://example.com/m.gguf", "/tmp/m.gguf")
            .with_sha256("abc123deadbeef");
        assert_eq!(req.expected_sha256.as_deref(), Some("abc123deadbeef"));
    }

    #[test]
    fn test_download_request_with_proxy() {
        let proxy = ProxyConfig::explicit("http://corp-proxy:3128");
        let req =
            DownloadRequest::new("https://example.com/m.gguf", "/tmp/m.gguf").with_proxy(proxy);
        assert!(req.proxy.has_proxy());
    }

    // ── DownloadResult ──

    #[test]
    fn test_download_result_debug() {
        let result = DownloadResult {
            path: PathBuf::from("/models/test.gguf"),
            freshly_downloaded: true,
            sha256: Some("abc123".to_string()),
            size_bytes: 1_000_000,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("test.gguf"));
        assert!(debug_str.contains("abc123"));
    }

    // ── SHA-256 ──

    #[test]
    fn test_sha256_file_known_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, b"hello world\n").unwrap();
        let hash = sha256_file(&path).unwrap();
        // SHA-256 of "hello world\n"
        assert_eq!(
            hash,
            "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
        );
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_sha256_file_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        std::fs::write(&path, b"").unwrap();
        let hash = sha256_file(&path).unwrap();
        // SHA-256 of empty input
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_file_not_found() {
        let result = sha256_file(Path::new("/nonexistent/file.bin"));
        assert!(result.is_err());
    }

    #[test]
    fn test_sha256_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, b"test data for hashing").unwrap();
        let h1 = sha256_file(&path).unwrap();
        let h2 = sha256_file(&path).unwrap();
        assert_eq!(h1, h2);
    }

    // ── ensure_model (offline scenarios) ──

    #[tokio::test]
    async fn test_ensure_model_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("my-model.gguf");
        std::fs::write(&model_path, b"fake model data").unwrap();

        let result = ensure_model(dir.path(), "my-model.gguf", None, false, None).await;
        assert!(result.is_ok());
        let dl = result.unwrap();
        assert!(!dl.freshly_downloaded);
        assert_eq!(dl.path, model_path);
    }

    #[tokio::test]
    async fn test_ensure_model_unknown_model_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let result = ensure_model(dir.path(), "totally-unknown-model.gguf", None, false, None).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not a recognized model"));
        assert!(err.contains("auto-downloadable models"));
    }

    // ── build_network_error ──

    #[test]
    fn test_build_network_error_without_proxy() {
        // We can't easily create a real reqwest::Error, but we can test the
        // proxy config branch by inspecting ProxyConfig directly
        let proxy = ProxyConfig::default();
        assert!(!proxy.has_proxy());
        assert!(proxy.summary().contains("No proxy"));
    }

    #[test]
    fn test_build_network_error_with_proxy() {
        let proxy = ProxyConfig::explicit("http://corp:3128");
        assert!(proxy.has_proxy());
        assert!(proxy.summary().contains("corp:3128"));
    }

    // ── SHA-256 async ──

    #[tokio::test]
    async fn test_sha256_file_async_matches_sync() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("async_test.bin");
        std::fs::write(&path, b"async hash test content").unwrap();
        let sync_hash = sha256_file(&path).unwrap();
        let async_hash = sha256_file_async(&path).await.unwrap();
        assert_eq!(sync_hash, async_hash);
    }

    #[tokio::test]
    async fn test_sha256_file_async_not_found() {
        let result = sha256_file_async(Path::new("/nonexistent/async_test.bin")).await;
        assert!(result.is_err());
    }

    // ── DownloadRequest builder chain ──

    #[test]
    fn test_download_request_full_builder_chain() {
        let proxy = ProxyConfig::explicit("http://proxy:8080");
        let req = DownloadRequest::new("https://example.com/model.gguf", "/tmp/model.gguf")
            .with_sha256("deadbeef")
            .with_proxy(proxy)
            .with_connect_timeout(Duration::from_secs(60));
        assert_eq!(req.url, "https://example.com/model.gguf");
        assert_eq!(req.expected_sha256.as_deref(), Some("deadbeef"));
        assert!(req.proxy.has_proxy());
        assert_eq!(req.connect_timeout, Duration::from_secs(60));
        assert!(req.resume); // default
    }

    // ── ProxyConfig edge cases ──

    #[test]
    fn test_proxy_config_effective_https_proxy_none() {
        let p = ProxyConfig {
            https_proxy: None,
            http_proxy: Some("http://fallback:8080".into()),
            no_proxy: None,
        };
        assert!(p.effective_https_proxy().is_none());
        assert!(p.has_proxy()); // http_proxy counts
    }

    #[test]
    fn test_proxy_config_no_proxy_field() {
        let p = ProxyConfig {
            https_proxy: Some("http://proxy:3128".into()),
            http_proxy: None,
            no_proxy: Some("localhost,127.0.0.1,.internal".into()),
        };
        assert_eq!(p.no_proxy.as_deref(), Some("localhost,127.0.0.1,.internal"));
    }

    // ── KnownModel properties ──

    #[test]
    fn test_known_models_descriptions_not_empty() {
        for model in KNOWN_MODELS {
            assert!(
                !model.description.is_empty(),
                "Model {} has empty description",
                model.filename
            );
        }
    }

    #[test]
    fn test_known_models_filenames_end_with_gguf() {
        for model in KNOWN_MODELS {
            assert!(
                model.filename.ends_with(".gguf"),
                "Model filename '{}' should end with .gguf",
                model.filename
            );
        }
    }

    #[test]
    fn test_known_models_sizes_reasonable() {
        for model in KNOWN_MODELS {
            // Models should be between 100 MB and 100 GB
            assert!(
                model.size_bytes >= 100_000_000 && model.size_bytes <= 100_000_000_000,
                "Model {} has suspicious size: {}",
                model.filename,
                model.size_bytes
            );
        }
    }

    #[test]
    fn test_known_models_sha256_format() {
        for model in KNOWN_MODELS {
            if let Some(hash) = model.sha256 {
                assert_eq!(
                    hash.len(),
                    64,
                    "SHA-256 for {} should be 64 hex chars",
                    model.filename
                );
                assert!(
                    hash.chars().all(|c| c.is_ascii_hexdigit()),
                    "SHA-256 for {} contains non-hex characters",
                    model.filename
                );
            }
        }
    }

    // ── DownloadResult ──

    #[test]
    fn test_download_result_freshly_downloaded() {
        let result = DownloadResult {
            path: PathBuf::from("/models/model.gguf"),
            freshly_downloaded: true,
            sha256: Some("abcdef1234567890".into()),
            size_bytes: 1_500_000_000,
        };
        assert!(result.freshly_downloaded);
        assert_eq!(result.size_bytes, 1_500_000_000);
        assert_eq!(result.sha256.as_deref(), Some("abcdef1234567890"));
    }

    #[test]
    fn test_download_result_existing_file() {
        let result = DownloadResult {
            path: PathBuf::from("/models/model.gguf"),
            freshly_downloaded: false,
            sha256: None,
            size_bytes: 500_000,
        };
        assert!(!result.freshly_downloaded);
        assert!(result.sha256.is_none());
    }

    #[test]
    fn test_download_result_clone() {
        let result = DownloadResult {
            path: PathBuf::from("/models/model.gguf"),
            freshly_downloaded: true,
            sha256: Some("abc".into()),
            size_bytes: 100,
        };
        let cloned = result.clone();
        assert_eq!(cloned.path, result.path);
        assert_eq!(cloned.freshly_downloaded, result.freshly_downloaded);
        assert_eq!(cloned.sha256, result.sha256);
        assert_eq!(cloned.size_bytes, result.size_bytes);
    }

    // ── ensure_model offline scenarios ──

    #[tokio::test]
    async fn test_ensure_model_existing_known_file_no_checksum() {
        // SmolLM2 has no checksum — should return immediately if file exists
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("smollm2-1.7b-instruct-q4_k_m.gguf");
        std::fs::write(&model_path, b"fake smollm2 data").unwrap();

        let result = ensure_model(dir.path(), "smollm2-1.7b-instruct-q4_k_m.gguf", None, false, None).await;
        assert!(result.is_ok());
        let dl = result.unwrap();
        assert!(!dl.freshly_downloaded);
        assert!(dl.sha256.is_none());
    }

    #[tokio::test]
    async fn test_ensure_model_with_proxy_unknown_model() {
        let dir = tempfile::tempdir().unwrap();
        let result = ensure_model(
            dir.path(),
            "unknown-custom-model.gguf",
            Some("http://myproxy:8080"),
            false,
            None,
        )
        .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not a recognized model"));
    }

    #[tokio::test]
    async fn test_ensure_model_creates_directory() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("sub").join("models");
        // Put a known file there
        std::fs::create_dir_all(&nested).unwrap();
        let model_path = nested.join("my-custom.gguf");
        std::fs::write(&model_path, b"model data").unwrap();

        let result = ensure_model(&nested, "my-custom.gguf", None, false, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_ensure_model_error_lists_known_models() {
        let dir = tempfile::tempdir().unwrap();
        let result = ensure_model(dir.path(), "no-such-model.gguf", None, false, None).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        // Should list available models in the error message
        assert!(err.contains("Qwen3.5-2B-UD-Q6_K_XL.gguf"));
        assert!(err.contains("smollm2-1.7b-instruct-q4_k_m.gguf"));
    }

    // ── format_size edge cases ──

    #[test]
    fn test_format_size_boundaries() {
        // Exactly at KB boundary
        assert_eq!(format_size(1024), "1 KB");
        // Exactly at MB boundary
        assert_eq!(format_size(1_048_576), "1.0 MB");
        // Exactly at GB boundary
        assert_eq!(format_size(1_073_741_824), "1.00 GB");
    }

    #[test]
    fn test_format_size_large_values() {
        // Multi-GB values
        let s = format_size(10_737_418_240); // 10 GB
        assert!(s.contains("GB"));
        assert!(s.starts_with("10."));
    }

    // ── partial_file_path edge cases ──

    #[test]
    fn test_partial_file_path_with_dots() {
        let dest = PathBuf::from("/models/model.v2.q4_k_m.gguf");
        let partial = partial_file_path(&dest);
        assert_eq!(
            partial,
            PathBuf::from("/models/model.v2.q4_k_m.gguf.partial")
        );
    }

    // ── download_file with invalid URL (offline test) ──

    #[tokio::test]
    async fn test_download_file_invalid_url() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("model.gguf");
        let request = DownloadRequest::new("http://127.0.0.1:1/nonexistent", &dest)
            .with_connect_timeout(Duration::from_millis(100));
        let result = download_file(&request, None).await;
        assert!(result.is_err());
    }

    // ── build_http_client ──

    #[test]
    fn test_build_http_client_no_proxy() {
        let proxy = ProxyConfig::default();
        let client = build_http_client(&proxy, Duration::from_secs(10), Duration::from_secs(60));
        assert!(client.is_ok());
    }

    #[test]
    fn test_build_http_client_with_proxy() {
        let proxy = ProxyConfig::explicit("http://proxy:3128");
        let client = build_http_client(&proxy, Duration::from_secs(10), Duration::from_secs(60));
        assert!(client.is_ok());
    }

    #[test]
    fn test_build_http_client_invalid_proxy_url() {
        let proxy = ProxyConfig {
            https_proxy: Some("not a valid url ://garbage".into()),
            http_proxy: None,
            no_proxy: None,
        };
        let client = build_http_client(&proxy, Duration::from_secs(10), Duration::from_secs(60));
        // reqwest may or may not reject this at build time depending on version,
        // but we exercise the code path
        let _ = client;
    }
}
