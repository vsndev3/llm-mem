use crate::{System, OutputFormat};
use llm_mem::document_session::SessionStatus;
use std::borrow::Cow::{self, Borrowed, Owned};
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::{Hinter, HistoryHinter};
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Cmd, CompletionType, Config, Editor, EventHandler, KeyCode, KeyEvent, Modifiers};
use rustyline::{Context, Helper};

/// Command history file path
const HISTORY_FILE: &str = ".llm-mem_history";

/// All top-level REPL commands.
const COMMANDS: &[&str] = &[
    "upload", "begin-upload", "upload-part", "process-document", "doc-status",
    "list-sessions", "list", "show", "search", "export", "stats",
    "layer-stats", "layer-tree", "list-banks", "system-status",
    "generate-config", "viz", "savelog", "use", "help", "exit", "quit",
];

/// Returns the known flags for a given command.
fn flags_for_command(cmd: &str) -> &'static [&'static str] {
    match cmd {
        "upload" => &["--file-path", "--bank", "--chunk-size", "--memory-type", "--context", "--format", "--process-immediately"],
        "begin-upload" => &["--file-name", "--size", "--mime-type", "--bank", "--memory-type", "--context", "--metadata", "--format"],
        "upload-part" => &["--session-id", "--part-index", "--file-path", "--bank", "--format"],
        "process-document" => &["--session-id", "--bank", "--partial-closure", "--format"],
        "doc-status" => &["--session-id", "--bank", "--format"],
        "list-sessions" => &["--bank", "--format"],
        "list" => &["--bank", "--limit", "--memory-type", "--format"],
        "show" => &["--memory-id", "--bank", "--format"],
        "search" => &["--query", "--mode", "--limit", "--bank", "--case-insensitive", "--show-scores", "--format"],
        "export" => &["--bank", "--output", "--pretty", "--format"],
        "stats" => &["--bank", "--format"],
        "layer-stats" => &["--bank", "--format"],
        "layer-tree" => &["--bank", "--from-layer", "--max-depth", "--show-ids", "--show-forgotten"],
        "list-banks" => &["--format"],
        "system-status" => &["--format"],
        "generate-config" => &["--output", "--format"],
        "viz" => &["--bank"],
        "savelog" => &["--level", "--stop"],
        _ => &[],
    }
}

/// Values for flags that accept a fixed set of choices.
fn values_for_flag(flag: &str) -> &'static [&'static str] {
    match flag {
        "--format" => &["table", "detail", "json", "jsonl", "csv"],
        "--mode" => &["text", "semantic"],
        _ => &[],
    }
}

/// The REPL helper providing completion, hints, validation, and highlighting.
struct ReplHelper {
    hinter: HistoryHinter,
}

impl ReplHelper {
    fn new() -> Self {
        ReplHelper {
            hinter: HistoryHinter {},
        }
    }
}

impl Helper for ReplHelper {}

impl Completer for ReplHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let line_to_pos = &line[..pos];
        let parts: Vec<&str> = line_to_pos.split_whitespace().collect();

        // If line is empty or we're still typing the first word
        if parts.is_empty() || (parts.len() == 1 && !line_to_pos.ends_with(' ')) {
            let prefix = parts.first().copied().unwrap_or("");
            let start = pos - prefix.len();
            let matches: Vec<Pair> = COMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(prefix))
                .map(|cmd| Pair {
                    display: cmd.to_string(),
                    replacement: cmd.to_string(),
                })
                .collect();
            return Ok((start, matches));
        }

        let command = parts[0];
        let current_token = if line_to_pos.ends_with(' ') { "" } else { parts.last().copied().unwrap_or("") };
        let start = pos - current_token.len();

        // After "help", suggest command names for per-command help
        if command == "help" {
            let prefix = if parts.len() == 1 && line_to_pos.ends_with(' ') {
                ""
            } else if parts.len() == 2 && !line_to_pos.ends_with(' ') {
                current_token
            } else {
                return Ok((pos, Vec::new()));
            };
            let actual_start = pos - prefix.len();
            let matches: Vec<Pair> = COMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(prefix))
                .map(|cmd| Pair {
                    display: cmd.to_string(),
                    replacement: cmd.to_string(),
                })
                .collect();
            return Ok((actual_start, matches));
        }

        // Check if the previous token is a flag that expects a value
        let prev_token = if line_to_pos.ends_with(' ') {
            parts.last().copied().unwrap_or("")
        } else if parts.len() >= 2 {
            parts[parts.len() - 2]
        } else {
            ""
        };

        if prev_token.starts_with("--") && line_to_pos.ends_with(' ') {
            // Suggest values for the previous flag
            let values = values_for_flag(prev_token);
            if !values.is_empty() {
                let matches: Vec<Pair> = values
                    .iter()
                    .map(|v| Pair {
                        display: v.to_string(),
                        replacement: v.to_string(),
                    })
                    .collect();
                return Ok((pos, matches));
            }
        }

        if !line_to_pos.ends_with(' ') && parts.len() >= 2 {
            let before_current = parts[parts.len() - 2];
            if before_current.starts_with("--") {
                let values = values_for_flag(before_current);
                if !values.is_empty() {
                    let matches: Vec<Pair> = values
                        .iter()
                        .filter(|v| v.starts_with(current_token))
                        .map(|v| Pair {
                            display: v.to_string(),
                            replacement: v.to_string(),
                        })
                        .collect();
                    return Ok((start, matches));
                }
            }
        }

        // Complete flags for the current command
        if current_token.starts_with('-') || line_to_pos.ends_with(' ') {
            let flags = flags_for_command(command);
            let prefix = if line_to_pos.ends_with(' ') { "" } else { current_token };
            let actual_start = if line_to_pos.ends_with(' ') { pos } else { start };
            let matches: Vec<Pair> = flags
                .iter()
                .filter(|f| f.starts_with(prefix))
                .map(|f| Pair {
                    display: f.to_string(),
                    replacement: f.to_string(),
                })
                .collect();
            return Ok((actual_start, matches));
        }

        Ok((pos, Vec::new()))
    }
}

impl Hinter for ReplHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, ctx: &Context<'_>) -> Option<String> {
        self.hinter.hint(line, pos, ctx)
    }
}

impl Highlighter for ReplHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        default: bool,
    ) -> Cow<'b, str> {
        if default {
            // Bold green prompt
            Owned(format!("\x1b[1;32m{}\x1b[0m", prompt))
        } else {
            Borrowed(prompt)
        }
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        // Dim gray hint text
        Owned(format!("\x1b[2m{}\x1b[0m", hint))
    }
}

impl Validator for ReplHelper {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        let input = ctx.input();
        // Trailing backslash means "continue on next line"
        if input.ends_with('\\') {
            Ok(ValidationResult::Incomplete)
        } else {
            Ok(ValidationResult::Valid(None))
        }
    }

    fn validate_while_typing(&self) -> bool {
        false
    }
}

/// Parse REPL-style `--key value` arguments into a map.
/// Boolean flags (keys without a following value or whose next token starts with `--`)
/// get the value "true". Unknown tokens without `--` prefix are collected into "positional".
pub(crate) fn parse_repl_args<'a>(args: &[&'a str]) -> std::collections::HashMap<String, Vec<&'a str>> {
    let mut map: std::collections::HashMap<String, Vec<&'a str>> = std::collections::HashMap::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(key) = args[i].strip_prefix("--") {
            // Check if next arg is a value (not another flag)
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                map.entry(key.to_string()).or_default().push(args[i + 1]);
                i += 2;
            } else {
                // Boolean flag
                map.entry(key.to_string()).or_default().push("true");
                i += 1;
            }
        } else {
            map.entry("positional".to_string()).or_default().push(args[i]);
            i += 1;
        }
    }
    map
}

/// Get a single value from parsed args, returning a default if not present.
pub(crate) fn get_arg<'a>(parsed: &'a std::collections::HashMap<String, Vec<&'a str>>, key: &str, default: &'a str) -> &'a str {
    parsed.get(key).and_then(|v| v.first()).copied().unwrap_or(default)
}

/// Get a required single value from parsed args.
pub(crate) fn require_arg<'a>(parsed: &'a std::collections::HashMap<String, Vec<&'a str>>, key: &str) -> Result<&'a str, String> {
    parsed.get(key)
        .and_then(|v| v.first())
        .copied()
        .ok_or_else(|| format!("Error: --{} is required", key))
}

/// Parse the --format flag from a raw args slice, returning the given default if absent.
fn parse_format_from_args(args: &[&str], default: OutputFormat) -> OutputFormat {
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--format" && i + 1 < args.len() {
            return match args[i + 1] {
                "table" => OutputFormat::Table,
                "detail" => OutputFormat::Detail,
                "json" => OutputFormat::Json,
                "jsonl" => OutputFormat::Jsonl,
                "csv" => OutputFormat::Csv,
                _ => default,
            };
        }
        i += 1;
    }
    default
}

/// Join multiline input: strip trailing backslashes and collapse continuation lines.
fn join_multiline(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    for line in input.lines() {
        let trimmed = line.trim_end();
        if let Some(prefix) = trimmed.strip_suffix('\\') {
            result.push_str(prefix);
            result.push(' ');
        } else {
            result.push_str(trimmed);
        }
    }
    result
}

/// Build the REPL prompt, showing background processing status if active.
async fn build_prompt(system: &System) -> String {
    if let Ok(sessions) = system.bank_manager.list_all_active_sessions().await {
        let mut uploading = 0u32;
        let mut processing = 0u32;
        let mut total_chunks = 0usize;
        let mut processed_chunks = 0usize;

        for s in &sessions {
            match s.status {
                SessionStatus::Uploading => { uploading += 1; }
                SessionStatus::Processing => {
                    processing += 1;
                    if let Some(ref r) = s.processing_result {
                        total_chunks += r.total_chunks;
                        processed_chunks += r.chunks_processed;
                    }
                }
                _ => {}
            }
        }

        if processing > 0 && total_chunks > 0 {
            let pct = (processed_chunks as f64 / total_chunks as f64 * 100.0) as u32;
            return format!("llm-mem(processing: {}%)> ", pct);
        } else if processing > 0 {
            return format!("llm-mem(processing: {})> ", processing);
        } else if uploading > 0 {
            return format!("llm-mem(uploading: {})> ", uploading);
        }
    }
    "llm-mem> ".to_string()
}

/// Run the REPL (Read-Eval-Print Loop)
pub async fn repl_loop(system: &System) -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::builder()
        .completion_type(CompletionType::List)
        .build();

    let helper = ReplHelper::new();
    let mut rl = Editor::with_config(config)?;
    rl.set_helper(Some(helper));

    // Ctrl+Enter (Ctrl+J) inserts a newline for multiline editing
    rl.bind_sequence(
        KeyEvent(KeyCode::Char('j'), Modifiers::CTRL),
        EventHandler::Simple(Cmd::Newline),
    );

    if rl.load_history(HISTORY_FILE).is_err() {
        // No history file yet, that's okay
    }
    
    loop {
        let prompt = build_prompt(system).await;
        match rl.readline(&prompt) {
            Ok(line) => {
                let joined = join_multiline(&line);
                let trimmed = joined.trim();
                if trimmed.is_empty() {
                    continue;
                }
                
                let _ = rl.add_history_entry(trimmed);
                
                if trimmed == "exit" || trimmed == "quit" {
                    break;
                } else if trimmed == "help" {
                    print_help();
                    continue;
                } else if trimmed.starts_with("help ") {
                    let cmd = trimmed["help ".len()..].trim();
                    match command_help(cmd) {
                        Some(text) => println!("{}", text),
                        None => println!("Unknown command: '{}'. Type 'help' for available commands.", cmd),
                    }
                    continue;
                } else if trimmed.starts_with("use ") {
                    handle_use_command(system, trimmed).await?;
                    continue;
                }
                
                if let Err(e) = execute_repl_command(system, trimmed).await {
                    eprintln!("Error: {}", e);
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::WindowResized) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                eprintln!("Error: {}", err);
                break;
            }
        }
    }
    
    let _ = rl.save_history(HISTORY_FILE);
    Ok(())
}

fn print_help() {
    println!("Available commands:");
    println!("  upload --file-path <path> [options]     - Upload a document with auto-chunking");
    println!("  begin-upload --file-name <name> --size <N> [options] - Start document session");
    println!("  upload-part --session-id <id> --part-index <N> --file-path <path> - Upload document part");
    println!("  process-document --session-id <id> [options] - Process uploaded document");
    println!("  doc-status --session-id <id>            - Check document processing status");
    println!("  list-sessions                           - List all document sessions");
    println!("  list [--bank NAME] [--limit N] [options] - List memories");
    println!("  show --memory-id <ID> [options]         - Show memory details");
    println!("  search --query <text> [options]         - Search memories");
    println!("  export --bank NAME [--output FILE] [--pretty] - Export bank data");
    println!("  stats [--bank NAME]                     - Show bank statistics");
    println!("  layer-stats [--bank NAME] [--format FMT] - Show layer statistics");
    println!("  layer-tree [--bank NAME] [options]      - Show layer hierarchy as tree");
    println!("  list-banks                              - List all memory banks");
    println!("  system-status                           - Check system status");
    println!("  generate-config --output <file>         - Generate config file with defaults");
    println!("  viz [--bank NAME]                       - Live document processing dashboard");
    println!("  savelog [--level LEVEL] <file>           - Start logging to file (stop with --stop)");
    println!("  use <bank>                              - Switch active bank");
    println!("  help                                    - Show this help");
    println!("  help <command>                          - Show detailed help for a command");
    println!("  exit/quit                               - Exit the REPL");
    println!();
    println!("Options commonly available:");
    println!("  --bank NAME        - Specify memory bank (default: 'default')");
    println!("  --format FMT       - Output format: table, json, jsonl, csv (default: table)");
    println!("  --limit N          - Limit number of results");
    println!();
    println!("Editing:");
    println!("  Tab                - Autocomplete commands and flags");
    println!("  Right arrow        - Accept inline history suggestion");
    println!("  Up/Down arrows     - Navigate command history");
    println!("  \\  (backslash)     - Continue command on next line");
    println!("  Ctrl+J             - Insert newline (multiline editing)");
    println!("  Ctrl+C             - Cancel current input");
    println!("  Ctrl+D             - Exit the REPL");
    println!();
}

/// Returns detailed help text for a specific command, or None if unknown.
fn command_help(cmd: &str) -> Option<&'static str> {
    match cmd {
        "upload" => Some(
"upload - Upload a document with automatic chunking and processing

  Reads a local file, splits it into chunks, and stores each chunk as a memory
  in the specified bank. By default the document is processed immediately after
  upload (extraction, classification, layer placement).

  USAGE
    upload --file-path <PATH> [OPTIONS]

  REQUIRED
    --file-path <PATH>          Path to the file to upload

  OPTIONS
    --bank <NAME>               Memory bank to store into (default: \"default\")
    --process-immediately <BOOL>
                                Start processing right after upload (default: true)
    --chunk-size <BYTES>        Custom chunk size in bytes (optional)
    --memory-type <TYPE>        Tag memories with a type label (optional)
    --context <TAG>             Context tags; can be repeated (optional)
    --format <FMT>              Output format: table, json, jsonl, csv (default: table)

  EXAMPLES
    upload --file-path ./notes.md
    upload --file-path ./data.csv --bank research --format json
    upload --file-path ./big.txt --chunk-size 4096 --context project-x"
        ),

        "begin-upload" => Some(
"begin-upload - Begin a multi-part document upload session

  Creates a new document storage session for uploading large files in parts.
  Returns a session ID that you use with upload-part and process-document.

  USAGE
    begin-upload --file-name <NAME> --size <BYTES> [OPTIONS]

  REQUIRED
    --file-name <NAME>          Original file name
    --size <BYTES>              Total file size in bytes

  OPTIONS
    --mime-type <MIME>           MIME type of the file (optional, auto-detected)
    --bank <NAME>               Memory bank (default: \"default\")
    --memory-type <TYPE>        Tag memories with a type label (optional)
    --context <TAG>             Context tags; can be repeated (optional)
    --metadata <JSON>           Custom metadata as a JSON string (optional)
    --format <FMT>              Output format: table, json, jsonl, csv (default: table)

  EXAMPLES
    begin-upload --file-name report.pdf --size 1048576
    begin-upload --file-name data.json --size 500000 --bank analytics --mime-type application/json"
        ),

        "upload-part" => Some(
"upload-part - Upload one part of a multi-part document

  Sends a file chunk for an active upload session created with begin-upload.
  Each part is identified by a zero-based index.

  USAGE
    upload-part --session-id <ID> --part-index <N> --file-path <PATH> [OPTIONS]

  REQUIRED
    --session-id <ID>           Session ID returned by begin-upload
    --part-index <N>            Zero-based part index (0, 1, 2, ...)
    --file-path <PATH>          Path to the file part on disk

  OPTIONS
    --bank <NAME>               Memory bank (default: \"default\")
    --format <FMT>              Output format: table, json, jsonl, csv (default: table)

  EXAMPLES
    upload-part --session-id abc123 --part-index 0 --file-path ./part0.bin
    upload-part --session-id abc123 --part-index 1 --file-path ./part1.bin --bank docs"
        ),

        "process-document" => Some(
"process-document - Process an uploaded document

  Triggers extraction, classification, and layer placement for a document
  that was uploaded via begin-upload / upload-part. The document session
  must have all expected parts unless --partial-closure is set.

  USAGE
    process-document --session-id <ID> [OPTIONS]

  REQUIRED
    --session-id <ID>           Session ID from begin-upload

  OPTIONS
    --bank <NAME>               Memory bank (default: \"default\")
    --partial-closure           Allow processing even if some parts are missing
    --format <FMT>              Output format: table, json, jsonl, csv (default: table)

  EXAMPLES
    process-document --session-id abc123
    process-document --session-id abc123 --partial-closure --bank research"
        ),

        "doc-status" => Some(
"doc-status - Check the status of a document processing session

  Shows the current state of a document upload/processing session including
  how many parts have been received and whether processing is complete.

  USAGE
    doc-status --session-id <ID> [OPTIONS]

  REQUIRED
    --session-id <ID>           Session ID to check

  OPTIONS
    --bank <NAME>               Memory bank (default: \"default\")
    --format <FMT>              Output format: table, json, jsonl, csv (default: table)

  EXAMPLES
    doc-status --session-id abc123
    doc-status --session-id abc123 --bank docs --format json"
        ),

        "list-sessions" => Some(
"list-sessions - List all document upload sessions

  Shows all document storage sessions in the specified bank, including
  their status (uploading, processing, complete, failed).

  USAGE
    list-sessions [OPTIONS]

  OPTIONS
    --bank <NAME>               Memory bank (default: \"default\")
    --format <FMT>              Output format: table, json, jsonl, csv (default: table)

  EXAMPLES
    list-sessions
    list-sessions --bank research --format json"
        ),

        "list" => Some(
"list - List memories stored in a bank

  Displays memories in the specified bank with optional filtering by type.
  Results are paginated with --limit.

  USAGE
    list [OPTIONS]

  OPTIONS
    --bank <NAME>               Memory bank (default: \"default\")
    --limit <N>                 Maximum number of memories to return (default: 50)
    --memory-type <TYPE>        Filter memories by type (optional)
    --format <FMT>              Output format: table, json, jsonl, csv (default: table)

  EXAMPLES
    list
    list --limit 20 --format json
    list --bank research --memory-type observation
    list --bank notes --limit 100 --format csv"
        ),

        "show" => Some(
"show - Show detailed information about a specific memory

  Retrieves and displays the full content, metadata, layer placement,
  importance score, and all other fields for a single memory by its ID.

  USAGE
    show --memory-id <ID> [OPTIONS]

  REQUIRED
    --memory-id <ID>            The unique ID of the memory to display

  OPTIONS
    --bank <NAME>               Memory bank (default: \"default\")
    --format <FMT>              Output format: table, detail, json, jsonl, csv
                                (default: detail)

  EXAMPLES
    show --memory-id mem_abc123
    show --memory-id mem_abc123 --bank research --format json"
        ),

        "search" => Some(
"search - Search memories using text or semantic search

  Finds memories matching a query. In text mode, performs substring/keyword
  matching across memory content. In semantic mode, uses vector embeddings
  to find conceptually similar memories.

  USAGE
    search --query <TEXT> [OPTIONS]

  REQUIRED
    --query <TEXT>              The search query string

  OPTIONS
    --bank <NAME>               Memory bank (default: \"default\")
    --mode <MODE>               Search mode: text or semantic (default: text)
    --limit <N>                 Maximum number of results (default: 10)
    --case-insensitive          Ignore case in text search
    --show-scores               Display similarity/relevance scores
    --format <FMT>              Output format: table, json, jsonl, csv (default: table)

  EXAMPLES
    search --query \"database migration\"
    search --query \"API design\" --mode semantic --limit 5
    search --query config --case-insensitive --show-scores
    search --query \"error handling\" --bank project-x --format json"
        ),

        "export" => Some(
"export - Export all bank data to JSON

  Dumps every memory in the bank as a JSON document. Output goes to stdout
  by default, or to a file with --output. Use --pretty for human-readable
  indented JSON.

  USAGE
    export [OPTIONS]

  OPTIONS
    --bank <NAME>               Memory bank (default: \"default\")
    --output <FILE>             Write to file instead of stdout (optional)
    --pretty                    Pretty-print the JSON output

  EXAMPLES
    export
    export --bank research --pretty
    export --output backup.json --pretty
    export --bank notes --output notes-export.json"
        ),

        "stats" => Some(
"stats - Show statistics about a memory bank

  Displays aggregate information about a bank: total memory count,
  breakdowns by type, layer distribution, and storage details.

  USAGE
    stats [OPTIONS]

  OPTIONS
    --bank <NAME>               Memory bank (default: \"default\")

  EXAMPLES
    stats
    stats --bank research"
        ),

        "layer-stats" => Some(
"layer-stats - Show layer statistics

  Displays per-layer statistics including memory counts, average importance,
  and content size for each abstraction layer in the bank.

  USAGE
    layer-stats [OPTIONS]

  OPTIONS
    --bank <NAME>               Memory bank (default: \"default\")
    --format <FMT>              Output format: table, json, jsonl, csv (default: table)

  EXAMPLES
    layer-stats
    layer-stats --bank research --format json"
        ),

        "layer-tree" => Some(
"layer-tree - Show layer hierarchy as an ASCII tree

  Visualises the abstraction layer structure. Memories are grouped by layer
  and displayed in a tree, optionally showing IDs and forgotten memories.

  USAGE
    layer-tree [OPTIONS]

  OPTIONS
    --bank <NAME>               Memory bank (default: \"default\")
    --from-layer <N>            Start from a specific layer level (optional)
    --max-depth <N>             Maximum tree depth to display (default: 5)
    --show-ids                  Include memory IDs in the output
    --show-forgotten            Include forgotten/archived memories

  EXAMPLES
    layer-tree
    layer-tree --bank research --max-depth 3
    layer-tree --show-ids --show-forgotten
    layer-tree --from-layer 2 --max-depth 2 --bank notes"
        ),

        "list-banks" => Some(
"list-banks - List all memory banks

  Shows every memory bank that currently exists on disk, along with
  basic metadata such as memory count and last-modified time.

  USAGE
    list-banks

  This command takes no arguments.

  EXAMPLES
    list-banks"
        ),

        "system-status" => Some(
"system-status - Check system status and readiness

  Reports the health of each subsystem: bank manager, memory manager,
  session manager, operations handler, and LLM connectivity. Useful for
  verifying that llm-mem is correctly configured and ready to use.

  USAGE
    system-status

  This command takes no arguments.

  EXAMPLES
    system-status"
        ),

        "generate-config" => Some(
"generate-config - Generate a configuration file with default values

  Creates a complete TOML configuration file with all default values and
  helpful comments. Useful for creating a starting point for custom configs.

  USAGE
    generate-config --output <FILE> [OPTIONS]

  REQUIRED
    --output <FILE>          Path where config file should be written

  OPTIONS
    --format <FMT>           Output format: table, json, jsonl, csv (default: table)

  EXAMPLES
    generate-config --output config.toml
    generate-config --output config.toml --format json"
        ),

        "viz" => Some(
"viz - Live document processing dashboard

  Opens a full-screen TUI showing real-time progress of document uploads,
  chunk processing, and layer abstractions across all banks. Each chunk is
  rendered as a colored block that changes as it moves through the pipeline.

  USAGE
    viz [OPTIONS]

  OPTIONS
    --bank <NAME>               Filter to a specific bank (default: all banks)

  KEY BINDINGS
    ESC / q                     Quit the dashboard
    l                           Toggle activity log panel

  EXAMPLES
    viz
    viz --bank research"
        ),

        "savelog" => Some(
"savelog - Start or stop file logging

  Saves tracing logs to a file. Useful for capturing API request/response
  details for debugging. Logs are appended if the file already exists.

  USAGE
    savelog [--level LEVEL] <FILE>   Start logging to file
    savelog --stop                   Stop logging
    savelog                          Show current status

  OPTIONS
    --level LEVEL       Log level: trace, debug, info, warn, error (default: debug)
    --stop              Stop the active file log

  EXAMPLES
    savelog mylog.txt
    savelog --level trace api_debug.log
    savelog --stop"
        ),

        "use" => Some(
"use - Switch the active memory bank

  Changes the default bank for subsequent commands in this REPL session.
  If the bank does not exist, it will be created automatically.

  USAGE
    use <BANK>

  ARGUMENTS
    <BANK>                      Name of the bank to switch to

  EXAMPLES
    use research
    use my-project"
        ),

        "help" => Some(
"help - Show help information

  Without arguments, lists all available commands with a short description.
  With a command name, shows detailed usage, flags, and examples.

  USAGE
    help
    help <COMMAND>

  EXAMPLES
    help
    help search
    help upload"
        ),

        "exit" | "quit" => Some(
"exit / quit - Exit the REPL

  Saves command history and exits the interactive session.

  USAGE
    exit
    quit"
        ),

        _ => None,
    }
}

/// Handle the 'use' command to switch active bank in REPL context
async fn handle_use_command(system: &System, input: &str) -> Result<(), Box<dyn std::error::Error>> {
    let parts: Vec<&str> = input.split_whitespace().collect();
    if parts.len() != 2 {
        println!("Usage: use <bank-name>");
        return Ok(());
    }
    
    let bank_name = parts[1];
    
    // Check if bank exists, create if not
    match system.bank_manager.get_or_create(bank_name).await {
        Ok(_) => {
            println!("Switched to bank '{}'", bank_name);
            Ok(())
        }
        Err(e) => {
            eprintln!("Error switching to bank '{}': {}", bank_name, e);
            Ok(())
        }
    }
}

/// Parse and execute a command in REPL context
async fn execute_repl_command(system: &System, input: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Simple space-based parsing for REPL (could be enhanced)
    let parts: Vec<&str> = input.split_whitespace().collect();
    if parts.is_empty() {
        return Ok(());
    }
    
    let command = parts[0];
    let args = &parts[1..];
    
    // Convert args to a format we can parse with clap
    // For simplicity in REPL, we'll handle common commands manually
    match command {
        "upload" => handle_upload_repl(system, args).await?,
        "begin-upload" => handle_begin_upload_repl(system, args).await?,
        "upload-part" => handle_upload_part_repl(system, args).await?,
        "process-document" => handle_process_document_repl(system, args).await?,
        "doc-status" => handle_doc_status_repl(system, args).await?,
        "list-sessions" => handle_list_sessions_repl(system, args).await?,
        "list" => handle_list_repl(system, args).await?,
        "show" => handle_show_repl(system, args).await?,
        "search" => handle_search_repl(system, args).await?,
        "export" => handle_export_repl(system, args).await?,
        "stats" => handle_stats_repl(system, args).await?,
        "layer-stats" => handle_layer_stats_repl(system, args).await?,
        "layer-tree" => handle_layer_tree_repl(system, args).await?,
        "list-banks" => handle_list_banks_repl(system, args).await?,
        "system-status" => handle_system_status_repl(system, args).await?,
        "generate-config" => handle_generate_config_repl(system, args).await?,
        "viz" => handle_viz_repl(system, args).await?,
        "savelog" => handle_savelog_repl(args)?,
        _ => {
            println!("Unknown command: {}", command);
            println!("Type 'help' for available commands");
        }
    }
    
    Ok(())
}

// REPL-specific handlers that parse simple arguments
async fn handle_upload_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let format = parse_format_from_args(args, OutputFormat::Detail);
    let mut file_path = None;
    let mut bank = "default";
    let mut process_immediately = true;
    
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--file-path" => {
                if i + 1 < args.len() {
                    file_path = Some(args[i + 1]);
                    i += 2;
                } else {
                    println!("Error: --file-path requires a value");
                    return Ok(());
                }
            }
            "--bank" => {
                if i + 1 < args.len() {
                    bank = args[i + 1];
                    i += 2;
                } else {
                    println!("Error: --bank requires a value");
                    return Ok(());
                }
            }
            "--process-immediately" => {
                if i + 1 < args.len() {
                    process_immediately = args[i + 1].parse().unwrap_or(true);
                    i += 2;
                } else {
                    println!("Error: --process-immediately requires a value");
                    return Ok(());
                }
            }
            "--format" => { i += 2; }
            _ => { i += 1; }
        }
    }
    
    let file_path = file_path.ok_or_else(|| "Error: --file-path is required")?;
    let path = std::path::Path::new(file_path);
    
    crate::commands::upload::handle_upload(
        system,
        path,
        bank,
        process_immediately,
        None,
        None,
        Vec::new(),
        format,
    ).await
}

async fn handle_begin_upload_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let format = parse_format_from_args(args, OutputFormat::Detail);
    let mut file_name = None;
    let mut total_size = None;
    let mut bank = "default";
    
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--file-name" => {
                if i + 1 < args.len() {
                    file_name = Some(args[i + 1]);
                    i += 2;
                } else {
                    println!("Error: --file-name requires a value");
                    return Ok(());
                }
            }
            "--size" => {
                if i + 1 < args.len() {
                    total_size = Some(args[i + 1].parse().map_err(|_| "Invalid size value")?);
                    i += 2;
                } else {
                    println!("Error: --size requires a value");
                    return Ok(());
                }
            }
            "--bank" => {
                if i + 1 < args.len() {
                    bank = args[i + 1];
                    i += 2;
                } else {
                    println!("Error: --bank requires a value");
                    return Ok(());
                }
            }
            "--format" => { i += 2; }
            _ => { i += 1; }
        }
    }
    
    let file_name = file_name.ok_or_else(|| "Error: --file-name is required")?;
    let total_size = total_size.ok_or_else(|| "Error: --size is required")?;
    
    crate::commands::begin_upload::handle_begin_upload(
        system,
        file_name,
        total_size,
        None,
        bank,
        None,
        Vec::new(),
        None,
        format,
    ).await
}

async fn handle_upload_part_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let format = parse_format_from_args(args, OutputFormat::Detail);
    let mut session_id = None;
    let mut part_index = None;
    let mut file_path = None;
    let mut bank = "default";
    
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--session-id" => {
                if i + 1 < args.len() {
                    session_id = Some(args[i + 1]);
                    i += 2;
                } else {
                    println!("Error: --session-id requires a value");
                    return Ok(());
                }
            }
            "--part-index" => {
                if i + 1 < args.len() {
                    part_index = Some(args[i + 1].parse().map_err(|_| "Invalid part index value")?);
                    i += 2;
                } else {
                    println!("Error: --part-index requires a value");
                    return Ok(());
                }
            }
            "--file-path" => {
                if i + 1 < args.len() {
                    file_path = Some(args[i + 1]);
                    i += 2;
                } else {
                    println!("Error: --file-path requires a value");
                    return Ok(());
                }
            }
            "--bank" => {
                if i + 1 < args.len() {
                    bank = args[i + 1];
                    i += 2;
                } else {
                    println!("Error: --bank requires a value");
                    return Ok(());
                }
            }
            "--format" => { i += 2; }
            _ => { i += 1; }
        }
    }
    
    let session_id = session_id.ok_or_else(|| "Error: --session-id is required")?;
    let part_index = part_index.ok_or_else(|| "Error: --part-index is required")?;
    let file_path = file_path.ok_or_else(|| "Error: --file-path is required")?;
    let path = std::path::Path::new(file_path);
    
    // Call the actual upload-part handler
    crate::commands::upload_part::handle_upload_part(
        system,
        session_id,
        part_index,
        path,
        bank,
        format,
    ).await
}

async fn handle_process_document_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let format = parse_format_from_args(args, OutputFormat::Detail);
    let mut session_id = None;
    let mut bank = "default";
    let mut partial_closure = false;
    
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--session-id" => {
                if i + 1 < args.len() {
                    session_id = Some(args[i + 1]);
                    i += 2;
                } else {
                    println!("Error: --session-id requires a value");
                    return Ok(());
                }
            }
            "--bank" => {
                if i + 1 < args.len() {
                    bank = args[i + 1];
                    i += 2;
                } else {
                    println!("Error: --bank requires a value");
                    return Ok(());
                }
            }
            "--partial-closure" => {
                if i + 1 < args.len() {
                    partial_closure = args[i + 1].parse().unwrap_or(false);
                    i += 2;
                } else {
                    println!("Error: --partial-closure requires a value");
                    return Ok(());
                }
            }
            "--format" => { i += 2; }
            _ => { i += 1; }
        }
    }
    
    let session_id = session_id.ok_or_else(|| "Error: --session-id is required")?;
    
    crate::commands::process_document::handle_process_document(
        system,
        session_id,
        partial_closure,
        bank,
        format,
    ).await
}

async fn handle_doc_status_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let format = parse_format_from_args(args, OutputFormat::Detail);
    let mut session_id = None;
    let mut bank = "default";
    
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--session-id" => {
                if i + 1 < args.len() {
                    session_id = Some(args[i + 1]);
                    i += 2;
                } else {
                    println!("Error: --session-id requires a value");
                    return Ok(());
                }
            }
            "--bank" => {
                if i + 1 < args.len() {
                    bank = args[i + 1];
                    i += 2;
                } else {
                    println!("Error: --bank requires a value");
                    return Ok(());
                }
            }
            "--format" => { i += 2; }
            _ => { i += 1; }
        }
    }
    
    let session_id = session_id.ok_or_else(|| "Error: --session-id is required")?;
    
    crate::commands::doc_status::handle_doc_status(
        system,
        session_id,
        bank,
        format,
    ).await
}

async fn handle_list_sessions_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let format = parse_format_from_args(args, OutputFormat::Detail);
    let mut bank = "default";
    
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--bank" => {
                if i + 1 < args.len() {
                    bank = args[i + 1];
                    i += 2;
                } else {
                    println!("Error: --bank requires a value");
                    return Ok(());
                }
            }
            "--format" => { i += 2; }
            _ => { i += 1; }
        }
    }
    
    crate::commands::list_sessions::handle_list_sessions(
        system,
        bank,
        format,
    ).await
}

async fn handle_list_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let format = parse_format_from_args(args, OutputFormat::Detail);
    let mut bank = "default";
    let mut limit = 50usize;
    let mut memory_type = None;
    
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--bank" => {
                if i + 1 < args.len() {
                    bank = args[i + 1];
                    i += 2;
                } else {
                    println!("Error: --bank requires a value");
                    return Ok(());
                }
            }
            "--limit" => {
                if i + 1 < args.len() {
                    limit = args[i + 1].parse().map_err(|_| "Invalid limit value")?;
                    i += 2;
                } else {
                    println!("Error: --limit requires a value");
                    return Ok(());
                }
            }
            "--memory-type" => {
                if i + 1 < args.len() {
                    memory_type = Some(args[i + 1]);
                    i += 2;
                } else {
                    println!("Error: --memory-type requires a value");
                    return Ok(());
                }
            }
            "--format" => { i += 2; } // already parsed
            _ => { i += 1; }
        }
    }
    
    let mut payload = llm_mem::operations::MemoryOperationPayload::default();
    payload.bank = Some(bank.to_string());
    payload.limit = Some(limit);
    if let Some(mt) = memory_type {
        payload.memory_type = Some(mt.to_string());
    }
    let operations = system.operations.lock().await;
    match operations.list_memories(payload).await {
        Ok(response) => {
            let output = crate::output::format_response(&response, format)?;
            crate::output::paginate_output(&output);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
    Ok(())
}

async fn handle_show_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let format = parse_format_from_args(args, OutputFormat::Detail);
    let mut memory_id = None;
    let mut bank = "default";
    
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--memory-id" => {
                if i + 1 < args.len() {
                    memory_id = Some(args[i + 1]);
                    i += 2;
                } else {
                    println!("Error: --memory-id requires a value");
                    return Ok(());
                }
            }
            "--bank" => {
                if i + 1 < args.len() {
                    bank = args[i + 1];
                    i += 2;
                } else {
                    println!("Error: --bank requires a value");
                    return Ok(());
                }
            }
            "--format" => { i += 2; }
            _ => { i += 1; }
        }
    }
    
    let memory_id = memory_id.ok_or_else(|| "Error: --memory-id is required")?;
    
    let mut payload = llm_mem::operations::MemoryOperationPayload::default();
    payload.memory_id = Some(memory_id.to_string());
    payload.bank = Some(bank.to_string());
    let operations = system.operations.lock().await;
    match operations.get_memory(payload).await {
        Ok(response) => {
            let output = crate::output::format_response(&response, format)?;
            crate::output::paginate_output(&output);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
    Ok(())
}

async fn handle_search_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let format = parse_format_from_args(args, OutputFormat::Detail);
    let mut query = None;
    let mut bank = "default";
    let mut limit = 10usize;
    
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--query" => {
                if i + 1 < args.len() {
                    query = Some(args[i + 1]);
                    i += 2;
                } else {
                    println!("Error: --query requires a value");
                    return Ok(());
                }
            }
            "--bank" => {
                if i + 1 < args.len() {
                    bank = args[i + 1];
                    i += 2;
                } else {
                    println!("Error: --bank requires a value");
                    return Ok(());
                }
            }
            "--limit" => {
                if i + 1 < args.len() {
                    limit = args[i + 1].parse().map_err(|_| "Invalid limit value")?;
                    i += 2;
                } else {
                    println!("Error: --limit requires a value");
                    return Ok(());
                }
            }
            "--format" => { i += 2; }
            _ => { i += 1; }
        }
    }
    
    let query = query.ok_or_else(|| "Error: --query is required")?;
    
    let mut payload = llm_mem::operations::MemoryOperationPayload::default();
    payload.query = Some(query.to_string());
    payload.bank = Some(bank.to_string());
    payload.limit = Some(limit);
    let operations = system.operations.lock().await;
    match operations.query_memory(payload).await {
        Ok(response) => {
            let output = crate::output::format_response(&response, format)?;
            crate::output::paginate_output(&output);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
    Ok(())
}

async fn handle_export_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let format = parse_format_from_args(args, OutputFormat::Json);
    let mut bank = "default";
    let mut output = None;
    let mut pretty = true; // default pretty in REPL
    
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--bank" => {
                if i + 1 < args.len() {
                    bank = args[i + 1];
                    i += 2;
                } else {
                    println!("Error: --bank requires a value");
                    return Ok(());
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output = Some(args[i + 1]);
                    i += 2;
                } else {
                    println!("Error: --output requires a value");
                    return Ok(());
                }
            }
            "--pretty" => {
                pretty = true;
                i += 1;
            }
            "--format" => { i += 2; }
            _ => { i += 1; }
        }
    }
    
    crate::commands::export::handle_export(
        system,
        bank,
        output.map(|s| std::path::Path::new(s)),
        pretty,
        format,
    ).await
}

async fn handle_stats_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let format = parse_format_from_args(args, OutputFormat::Table);
    let mut bank = "default";
    
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--bank" => {
                if i + 1 < args.len() {
                    bank = args[i + 1];
                    i += 2;
                } else {
                    println!("Error: --bank requires a value");
                    return Ok(());
                }
            }
            "--format" => { i += 2; }
            _ => { i += 1; }
        }
    }
    
    // Build payload and get memories for stats computation
    let mut payload = llm_mem::operations::MemoryOperationPayload::default();
    payload.bank = Some(bank.to_string());
    let operations = system.operations.lock().await;
    match operations.list_memories(payload).await {
        Ok(response) => {
            if let Some(data) = &response.data {
                // data is {"count": N, "memories": [...]}, extract the memories array
                let memories_value = if let serde_json::Value::Object(obj) = data {
                    obj.get("memories").cloned()
                } else if data.is_array() {
                    Some(data.clone())
                } else {
                    None
                };
                if let Some(serde_json::Value::Array(memories)) = &memories_value {
                    if memories.is_empty() {
                        println!("No memories found in bank '{}'", bank);
                        return Ok(());
                    }
                    let total_count = memories.len();
                    let (type_counts, state_counts, layer_counts) =
                        crate::commands::stats::compute_memory_counts(memories);
                    let output = crate::commands::stats::format_stats_output(
                        bank, total_count, &type_counts, &state_counts, &layer_counts, format,
                    )?;
                    crate::output::paginate_output(&output);
                } else {
                    let output = crate::output::format_response(&response, format)?;
                    crate::output::paginate_output(&output);
                }
            } else {
                let output = crate::output::format_response(&response, format)?;
                crate::output::paginate_output(&output);
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
    Ok(())
}

async fn handle_layer_stats_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let format = parse_format_from_args(args, OutputFormat::Table);
    let mut bank = "default";
    
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--bank" => {
                if i + 1 < args.len() {
                    bank = args[i + 1];
                    i += 2;
                } else {
                    println!("Error: --bank requires a value");
                    return Ok(());
                }
            }
            "--format" => { i += 2; }
            _ => { i += 1; }
        }
    }
    
    // Build payload and get memories for layer stats computation
    let mut payload = llm_mem::operations::MemoryOperationPayload::default();
    payload.bank = Some(bank.to_string());
    let operations = system.operations.lock().await;
    match operations.list_memories(payload).await {
        Ok(response) => {
            if let Some(data) = &response.data {
                // data is {"count": N, "memories": [...]}, extract the memories array
                let memories_value = if let serde_json::Value::Object(obj) = data {
                    obj.get("memories").cloned()
                } else if data.is_array() {
                    Some(data.clone())
                } else {
                    None
                };
                if let Some(serde_json::Value::Array(memories)) = &memories_value {
                    if memories.is_empty() {
                        println!("No memories found in bank '{}'", bank);
                        return Ok(());
                    }
                    let stats = crate::commands::layer_stats::compute_layer_stats(memories);
                    let output = crate::commands::layer_stats::format_layer_stats_output(
                        bank, &stats, format,
                    )?;
                    crate::output::paginate_output(&output);
                } else {
                    let output = crate::output::format_response(&response, format)?;
                    crate::output::paginate_output(&output);
                }
            } else {
                let output = crate::output::format_response(&response, format)?;
                crate::output::paginate_output(&output);
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
    Ok(())
}

async fn handle_layer_tree_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    // Simple argument parsing for REPL
    let mut bank = "default";
    let mut from_layer = None;
    let mut max_depth = 5usize;
    let mut show_ids = false;
    let mut show_forgotten = false;
    
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--bank" => {
                if i + 1 < args.len() {
                    bank = args[i + 1];
                    i += 2;
                } else {
                    println!("Error: --bank requires a value");
                    return Ok(());
                }
            }
            "--from-layer" => {
                if i + 1 < args.len() {
                    from_layer = Some(args[i + 1].parse().map_err(|_| "Invalid from-layer value")?);
                    i += 2;
                } else {
                    println!("Error: --from-layer requires a value");
                    return Ok(());
                }
            }
            "--max-depth" => {
                if i + 1 < args.len() {
                    max_depth = args[i + 1].parse().map_err(|_| "Invalid max-depth value")?;
                    i += 2;
                } else {
                    println!("Error: --max-depth requires a value");
                    return Ok(());
                }
            }
            "--show-ids" => {
                show_ids = true;
                i += 1;
            }
            "--show-forgotten" => {
                show_forgotten = true;
                i += 1;
            }
            "--format" => { i += 2; }
            _ => { i += 1; }
        }
    }
    
    // Call the actual layer-tree handler
    crate::commands::layer_tree::handle_layer_tree(
        system,
        bank,
        from_layer.as_ref(),
        max_depth,
        show_ids,
        show_forgotten,
    ).await
}

async fn handle_list_banks_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let format = parse_format_from_args(args, OutputFormat::Detail);
    crate::commands::list_banks::handle_list_banks(system, format).await
}

async fn handle_system_status_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let format = parse_format_from_args(args, OutputFormat::Detail);
    crate::commands::system_status::handle_system_status(system, format).await
}

async fn handle_generate_config_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let format = parse_format_from_args(args, OutputFormat::Detail);
    let mut output = None;
    
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--output" => {
                if i + 1 < args.len() {
                    output = Some(args[i + 1].to_string());
                    i += 2;
                } else {
                    println!("Error: --output requires a value");
                    return Ok(());
                }
            }
            "--format" => { i += 2; }
            _ => { i += 1; }
        }
    }
    
    let output_path = output.ok_or("Missing required --output argument")?;
    crate::commands::generate_config::handle_generate_config(
        std::path::Path::new(&output_path),
        format,
    )
    .await
}

async fn handle_viz_repl(system: &System, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let mut bank = None;
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--bank" => {
                if i + 1 < args.len() {
                    bank = Some(args[i + 1]);
                    i += 2;
                } else {
                    println!("Error: --bank requires a value");
                    return Ok(());
                }
            }
            _ => { i += 1; }
        }
    }
    crate::commands::viz::handle_viz(system, bank).await
}

fn handle_savelog_repl(args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    use crate::log_capture;
    use tracing::Level;

    // savelog --stop
    if args.contains(&"--stop") {
        if let Some(path) = log_capture::stop_file_log() {
            println!("File logging stopped. Log written to: {}", path.display());
        } else {
            println!("File logging is not active.");
        }
        return Ok(());
    }

    // savelog (no args) — show status
    if args.is_empty() {
        if let Some((path, level)) = log_capture::file_log_status() {
            println!("File logging active: {} (level: {})", path.display(), level);
        } else {
            println!("File logging is not active.");
            println!("Usage: savelog [--level info|debug|trace] <file>");
        }
        return Ok(());
    }

    // Parse: savelog [--level LEVEL] <file>
    let mut level = Level::DEBUG;
    let mut file_path = None;
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--level" => {
                if i + 1 < args.len() {
                    level = match args[i + 1].to_lowercase().as_str() {
                        "trace" => Level::TRACE,
                        "debug" => Level::DEBUG,
                        "info" => Level::INFO,
                        "warn" => Level::WARN,
                        "error" => Level::ERROR,
                        other => {
                            println!("Unknown level '{}'. Use: trace, debug, info, warn, error", other);
                            return Ok(());
                        }
                    };
                    i += 2;
                } else {
                    println!("Error: --level requires a value");
                    return Ok(());
                }
            }
            arg if !arg.starts_with('-') => {
                file_path = Some(arg);
                i += 1;
            }
            _ => { i += 1; }
        }
    }

    let Some(path_str) = file_path else {
        println!("Error: file path required. Usage: savelog [--level LEVEL] <file>");
        return Ok(());
    };

    let path = std::path::PathBuf::from(path_str);
    match log_capture::start_file_log(path, level) {
        Ok(resolved) => println!("Logging to {} (level: {})", resolved.display(), level),
        Err(e) => println!("Failed to open log file: {}", e),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_repl_args tests ---

    #[test]
    fn test_parse_empty_args() {
        let result = parse_repl_args(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_single_key_value() {
        let result = parse_repl_args(&["--bank", "mybank"]);
        assert_eq!(result.get("bank").unwrap(), &vec!["mybank"]);
    }

    #[test]
    fn test_parse_multiple_key_values() {
        let result = parse_repl_args(&["--bank", "mybank", "--limit", "25"]);
        assert_eq!(result.get("bank").unwrap(), &vec!["mybank"]);
        assert_eq!(result.get("limit").unwrap(), &vec!["25"]);
    }

    #[test]
    fn test_parse_boolean_flag_at_end() {
        let result = parse_repl_args(&["--query", "test", "--case-insensitive"]);
        assert_eq!(result.get("query").unwrap(), &vec!["test"]);
        assert_eq!(result.get("case-insensitive").unwrap(), &vec!["true"]);
    }

    #[test]
    fn test_parse_boolean_flag_before_another_flag() {
        let result = parse_repl_args(&["--show-ids", "--bank", "x"]);
        assert_eq!(result.get("show-ids").unwrap(), &vec!["true"]);
        assert_eq!(result.get("bank").unwrap(), &vec!["x"]);
    }

    #[test]
    fn test_parse_repeated_keys() {
        let result = parse_repl_args(&["--context", "a", "--context", "b"]);
        assert_eq!(result.get("context").unwrap(), &vec!["a", "b"]);
    }

    #[test]
    fn test_parse_positional_args() {
        let result = parse_repl_args(&["some-value", "--bank", "x"]);
        assert_eq!(result.get("positional").unwrap(), &vec!["some-value"]);
        assert_eq!(result.get("bank").unwrap(), &vec!["x"]);
    }

    // --- get_arg tests ---

    #[test]
    fn test_get_arg_present() {
        let parsed = parse_repl_args(&["--bank", "mybank"]);
        assert_eq!(get_arg(&parsed, "bank", "default"), "mybank");
    }

    #[test]
    fn test_get_arg_missing_uses_default() {
        let parsed = parse_repl_args(&[]);
        assert_eq!(get_arg(&parsed, "bank", "default"), "default");
    }

    #[test]
    fn test_get_arg_returns_first_of_repeated() {
        let parsed = parse_repl_args(&["--bank", "first", "--bank", "second"]);
        assert_eq!(get_arg(&parsed, "bank", "default"), "first");
    }

    // --- require_arg tests ---

    #[test]
    fn test_require_arg_present() {
        let parsed = parse_repl_args(&["--session-id", "abc-123"]);
        assert_eq!(require_arg(&parsed, "session-id").unwrap(), "abc-123");
    }

    #[test]
    fn test_require_arg_missing_returns_error() {
        let parsed = parse_repl_args(&[]);
        let err = require_arg(&parsed, "session-id").unwrap_err();
        assert!(err.contains("--session-id is required"));
    }

    // --- Integration parsing patterns ---

    #[test]
    fn test_upload_arg_pattern() {
        let args = ["--file-path", "/tmp/doc.txt", "--bank", "mybank"];
        let parsed = parse_repl_args(&args);
        assert_eq!(get_arg(&parsed, "file-path", ""), "/tmp/doc.txt");
        assert_eq!(get_arg(&parsed, "bank", "default"), "mybank");
        assert_eq!(get_arg(&parsed, "process-immediately", "true"), "true");
    }

    #[test]
    fn test_search_arg_pattern() {
        let args = ["--query", "rust programming", "--mode", "semantic", "--limit", "5", "--case-insensitive"];
        let parsed = parse_repl_args(&args);
        assert_eq!(get_arg(&parsed, "query", ""), "rust programming");
        assert_eq!(get_arg(&parsed, "mode", "text"), "semantic");
        assert_eq!(get_arg(&parsed, "limit", "10"), "5");
        assert_eq!(get_arg(&parsed, "case-insensitive", "false"), "true");
    }

    #[test]
    fn test_layer_tree_arg_pattern() {
        let args = ["--from-layer", "2", "--max-depth", "3", "--show-ids", "--show-forgotten"];
        let parsed = parse_repl_args(&args);
        assert_eq!(get_arg(&parsed, "from-layer", ""), "2");
        assert_eq!(get_arg(&parsed, "max-depth", "5"), "3");
        assert_eq!(get_arg(&parsed, "show-ids", "false"), "true");
        assert_eq!(get_arg(&parsed, "show-forgotten", "false"), "true");
    }

    #[test]
    fn test_begin_upload_arg_pattern() {
        let args = ["--file-name", "large.pdf", "--size", "1048576", "--bank", "docs"];
        let parsed = parse_repl_args(&args);
        assert_eq!(require_arg(&parsed, "file-name").unwrap(), "large.pdf");
        assert_eq!(require_arg(&parsed, "size").unwrap(), "1048576");
        assert_eq!(get_arg(&parsed, "bank", "default"), "docs");
    }

    #[test]
    fn test_export_arg_pattern() {
        let args = ["--bank", "prod", "--output", "/tmp/export.json", "--pretty"];
        let parsed = parse_repl_args(&args);
        assert_eq!(get_arg(&parsed, "bank", "default"), "prod");
        assert_eq!(get_arg(&parsed, "output", ""), "/tmp/export.json");
        assert_eq!(get_arg(&parsed, "pretty", "false"), "true");
    }

    // --- join_multiline tests ---

    #[test]
    fn test_join_multiline_single_line() {
        assert_eq!(join_multiline("search --query hello"), "search --query hello");
    }

    #[test]
    fn test_join_multiline_continuation() {
        assert_eq!(
            join_multiline("search --query hello \\\n--limit 5"),
            "search --query hello  --limit 5"
        );
    }

    #[test]
    fn test_join_multiline_multiple_continuations() {
        assert_eq!(
            join_multiline("upload \\\n--file-path /tmp/doc.txt \\\n--bank mybank"),
            "upload  --file-path /tmp/doc.txt  --bank mybank"
        );
    }

    #[test]
    fn test_join_multiline_no_trailing_backslash() {
        assert_eq!(join_multiline("list --bank default"), "list --bank default");
    }

    #[test]
    fn test_join_multiline_empty() {
        assert_eq!(join_multiline(""), "");
    }

    // --- flags_for_command tests ---

    #[test]
    fn test_flags_for_known_commands() {
        assert!(flags_for_command("upload").contains(&"--file-path"));
        assert!(flags_for_command("search").contains(&"--query"));
        assert!(flags_for_command("search").contains(&"--mode"));
        assert!(flags_for_command("list").contains(&"--limit"));
        assert!(flags_for_command("layer-tree").contains(&"--show-ids"));
    }

    #[test]
    fn test_flags_for_unknown_command() {
        assert!(flags_for_command("unknown").is_empty());
    }

    #[test]
    fn test_flags_for_no_arg_commands() {
        // list-banks and system-status now accept --format
        assert_eq!(flags_for_command("list-banks"), &["--format"]);
        assert_eq!(flags_for_command("system-status"), &["--format"]);
    }

    // --- values_for_flag tests ---

    #[test]
    fn test_values_for_format_flag() {
        let vals = values_for_flag("--format");
        assert!(vals.contains(&"table"));
        assert!(vals.contains(&"json"));
        assert!(vals.contains(&"csv"));
    }

    #[test]
    fn test_values_for_mode_flag() {
        let vals = values_for_flag("--mode");
        assert!(vals.contains(&"text"));
        assert!(vals.contains(&"semantic"));
    }

    #[test]
    fn test_values_for_unknown_flag() {
        assert!(values_for_flag("--bank").is_empty());
    }

    // --- COMMANDS constant test ---

    #[test]
    fn test_all_commands_present() {
        assert!(COMMANDS.contains(&"upload"));
        assert!(COMMANDS.contains(&"search"));
        assert!(COMMANDS.contains(&"list"));
        assert!(COMMANDS.contains(&"help"));
        assert!(COMMANDS.contains(&"exit"));
        assert!(COMMANDS.contains(&"quit"));
        assert!(COMMANDS.contains(&"use"));
        assert!(COMMANDS.contains(&"system-status"));
        assert!(COMMANDS.contains(&"layer-tree"));
    }

    // --- command_help tests ---

    #[test]
    fn test_command_help_returns_some_for_all_commands() {
        let cmds = [
            "upload", "begin-upload", "upload-part", "process-document",
            "doc-status", "list-sessions", "list", "show", "search",
            "export", "stats", "layer-stats", "layer-tree", "list-banks",
            "system-status", "use", "help", "exit", "quit",
        ];
        for cmd in cmds {
            assert!(command_help(cmd).is_some(), "Missing help for '{}'", cmd);
        }
    }

    #[test]
    fn test_command_help_unknown_returns_none() {
        assert!(command_help("nonexistent").is_none());
        assert!(command_help("").is_none());
    }

    #[test]
    fn test_command_help_contains_usage() {
        // Every command help should include USAGE section
        let cmds_with_usage = [
            "upload", "begin-upload", "upload-part", "process-document",
            "doc-status", "list-sessions", "list", "show", "search",
            "export", "stats", "layer-stats", "layer-tree", "list-banks",
            "system-status", "use", "help",
        ];
        for cmd in cmds_with_usage {
            let text = command_help(cmd).unwrap();
            assert!(text.contains("USAGE"), "Help for '{}' missing USAGE section", cmd);
        }
    }

    #[test]
    fn test_command_help_contains_examples() {
        // Commands with examples
        let cmds = [
            "upload", "search", "list", "show", "export", "stats",
            "layer-tree", "list-banks", "system-status",
        ];
        for cmd in cmds {
            let text = command_help(cmd).unwrap();
            assert!(text.contains("EXAMPLES"), "Help for '{}' missing EXAMPLES section", cmd);
        }
    }

    #[test]
    fn test_command_help_upload_details() {
        let text = command_help("upload").unwrap();
        assert!(text.contains("--file-path"));
        assert!(text.contains("--bank"));
        assert!(text.contains("--chunk-size"));
        assert!(text.contains("--format"));
        assert!(text.contains("REQUIRED"));
    }

    #[test]
    fn test_command_help_search_details() {
        let text = command_help("search").unwrap();
        assert!(text.contains("--query"));
        assert!(text.contains("--mode"));
        assert!(text.contains("semantic"));
        assert!(text.contains("--case-insensitive"));
        assert!(text.contains("--show-scores"));
    }
}