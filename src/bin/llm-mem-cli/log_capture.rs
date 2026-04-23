//! Tracing layer that captures log events into a shared ring buffer.
//!
//! The viz command reads from this buffer to display live logs in the TUI panel.
//! The `savelog` command can enable file logging via the global file sink.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tracing::Level;
use tracing_subscriber::Layer;

/// When true, stderr log output is suppressed (TUI owns the screen).
static TUI_ACTIVE: AtomicBool = AtomicBool::new(false);

pub fn set_tui_active(active: bool) {
    TUI_ACTIVE.store(active, Ordering::SeqCst);
}

/// A writer that forwards to stderr unless the TUI is active.
pub struct TuiAwareStderr;

impl Write for TuiAwareStderr {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if TUI_ACTIVE.load(Ordering::SeqCst) {
            Ok(buf.len()) // silently discard
        } else {
            std::io::stderr().write(buf)
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if !TUI_ACTIVE.load(Ordering::SeqCst) {
            std::io::stderr().flush()
        } else {
            Ok(())
        }
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for TuiAwareStderr {
    type Writer = TuiAwareStderr;

    fn make_writer(&'a self) -> Self::Writer {
        TuiAwareStderr
    }
}

/// Maximum number of log entries retained in the ring buffer.
const MAX_ENTRIES: usize = 2000;

/// A captured log entry.
#[derive(Debug, Clone)]
pub struct CapturedLog {
    pub timestamp: String,
    pub level: Level,
    pub target: String,
    pub message: String,
}

/// Shared ring buffer for captured logs.
pub type LogBuffer = Arc<Mutex<VecDeque<CapturedLog>>>;

/// Global log buffer singleton.
static LOG_BUFFER: OnceLock<LogBuffer> = OnceLock::new();

/// Get (or initialize) the global log buffer.
pub fn global_log_buffer() -> LogBuffer {
    LOG_BUFFER
        .get_or_init(|| Arc::new(Mutex::new(VecDeque::with_capacity(MAX_ENTRIES))))
        .clone()
}

// ── File sink for `savelog` ────────────────────────────────────────────────

/// State for the file log sink.
struct FileSink {
    file: File,
    level: Level,
    path: PathBuf,
}

static FILE_SINK: OnceLock<Mutex<Option<FileSink>>> = OnceLock::new();

fn file_sink() -> &'static Mutex<Option<FileSink>> {
    FILE_SINK.get_or_init(|| Mutex::new(None))
}

/// Start logging to a file at the given level. Returns the resolved path.
pub fn start_file_log(path: PathBuf, level: Level) -> std::io::Result<PathBuf> {
    let file = OpenOptions::new().create(true).append(true).open(&path)?;
    let resolved = path.canonicalize().unwrap_or_else(|_| path.clone());
    if let Ok(mut sink) = file_sink().lock() {
        *sink = Some(FileSink {
            file,
            level,
            path: resolved.clone(),
        });
    }
    Ok(resolved)
}

/// Stop file logging and return the path that was being written to.
pub fn stop_file_log() -> Option<PathBuf> {
    if let Ok(mut sink) = file_sink().lock() {
        sink.take().map(|s| s.path)
    } else {
        None
    }
}

/// Check if file logging is active.
pub fn file_log_status() -> Option<(PathBuf, Level)> {
    if let Ok(sink) = file_sink().lock() {
        sink.as_ref().map(|s| (s.path.clone(), s.level))
    } else {
        None
    }
}

/// Write a log entry to the file sink if active and the level matches.
fn write_to_file_sink(entry: &CapturedLog) {
    if let Ok(mut sink) = file_sink().lock()
        && let Some(ref mut s) = *sink
        && entry.level <= s.level
    {
        let _ = writeln!(
            s.file,
            "{} {:>5} [{}] {}",
            entry.timestamp, entry.level, entry.target, entry.message,
        );
    }
}

/// A tracing layer that pushes events into the shared ring buffer.
pub struct BufferLayer {
    buffer: LogBuffer,
}

impl BufferLayer {
    pub fn new(buffer: LogBuffer) -> Self {
        Self { buffer }
    }
}

impl<S> Layer<S> for BufferLayer
where
    S: tracing::Subscriber,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        // Extract the message from the event fields
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);

        let entry = CapturedLog {
            timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
            level: *event.metadata().level(),
            target: event.metadata().target().to_string(),
            message: visitor.message,
        };

        if let Ok(mut buf) = self.buffer.lock() {
            buf.push_back(entry.clone());
            while buf.len() > MAX_ENTRIES {
                buf.pop_front();
            }
        }

        // Also write to file sink if active
        write_to_file_sink(&entry);
    }
}

/// Visitor that extracts the `message` field from a tracing event.
#[derive(Default)]
struct MessageVisitor {
    message: String,
}

impl tracing::field::Visit for MessageVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{:?}", value);
            // Strip surrounding quotes if present
            if self.message.starts_with('"') && self.message.ends_with('"') {
                self.message = self.message[1..self.message.len() - 1].to_string();
            }
        } else if self.message.is_empty() {
            // Fallback: use first field as message
            self.message = format!("{} = {:?}", field.name(), value);
        } else {
            // Append additional fields
            self.message
                .push_str(&format!(" {}={:?}", field.name(), value));
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else if self.message.is_empty() {
            self.message = format!("{} = {}", field.name(), value);
        } else {
            self.message
                .push_str(&format!(" {}={}", field.name(), value));
        }
    }
}
