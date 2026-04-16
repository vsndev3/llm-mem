//! Visualization command for live block progress display

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use llm_mem::{System, document_session::{ChunkProgress, ChunkStatus}};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use std::{
    collections::VecDeque,
    io,
    time::{Duration, Instant},
};
use tracing::Level;
use uuid::Uuid;

use crate::log_capture;

/// Maximum number of activity log entries retained.
const MAX_LOG_ENTRIES: usize = 200;

#[derive(Debug, Clone, Default)]
pub struct VizData {
    pub sessions: Vec<VizSession>,
    pub abstractions: Vec<VizAbstraction>,
    /// Memory counts by layer level (for full-bank grid display)
    pub memory_layers: Vec<(i32, usize)>,
    /// IDs of memories currently being processed (for blink overlay)
    pub processing_ids: std::collections::HashSet<Uuid>,
    /// Bank summaries for the side panel
    pub banks: Vec<VizBankInfo>,
    /// Pipeline config/status for the side panel
    pub pipeline_info: Option<VizPipelineInfo>,
}

#[derive(Debug, Clone)]
pub struct VizBankInfo {
    pub name: String,
    pub memory_count: usize,
    pub loaded: bool,
}

#[derive(Debug, Clone)]
pub struct VizPipelineInfo {
    pub enabled: bool,
    pub pending_count: usize,
    pub min_memories_for_l1: usize,
    pub delay_secs: u64,
}

#[derive(Debug, Clone)]
pub struct VizSession {
    pub session_id: String,
    pub file_name: String,
    pub chunks: Vec<ChunkProgress>,
    pub status: String,
    #[allow(dead_code)]
    pub progress_percent: f32,
}

#[derive(Debug, Clone)]
pub struct VizAbstraction {
    #[allow(dead_code)]
    pub memory_id: Uuid,
    pub current_level: i32,
    pub target_level: i32,
    #[allow(dead_code)]
    pub retry_count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlockStage {
    Queued,
    Uploading,
    Enriching,
    Embedding,
    AbstractingL0toL1,
    AbstractingL1toL2,
    AbstractingL2toL3,
    Completed,
    Failed,
    // Settled memory layer stages
    MemoryL0,
    MemoryL1,
    MemoryL2,
    MemoryL3,
}

impl BlockStage {
    fn color(&self) -> Color {
        match self {
            BlockStage::Queued => Color::Gray,
            BlockStage::Uploading => Color::Blue,
            BlockStage::Enriching => Color::Rgb(255, 165, 0), // Orange (active batch)
            BlockStage::Embedding => Color::Rgb(139, 90, 43), // Brown (enriched, awaiting storage)
            BlockStage::AbstractingL0toL1 => Color::Rgb(255, 165, 0), // Orange (L0 source color)
            BlockStage::AbstractingL1toL2 => Color::Red,               // Red (L1 source color)
            BlockStage::AbstractingL2toL3 => Color::Magenta,           // Magenta (L2 source color)
            BlockStage::Completed => Color::Green,
            BlockStage::Failed => Color::Red,
            BlockStage::MemoryL0 => Color::Rgb(255, 165, 0),  // Orange
            BlockStage::MemoryL1 => Color::Red,
            BlockStage::MemoryL2 => Color::Magenta,
            BlockStage::MemoryL3 => Color::Cyan,
        }
    }

    fn unicode_char(&self) -> &'static str {
        match self {
            BlockStage::Queued => "░",
            BlockStage::Uploading => "▓",
            BlockStage::Enriching => "▒",
            BlockStage::Embedding => "█",  // full block for completed enrichment
            BlockStage::AbstractingL0toL1 => "▒",
            BlockStage::AbstractingL1toL2 => "▒",
            BlockStage::AbstractingL2toL3 => "▒",
            BlockStage::Completed => "█",
            BlockStage::Failed => "✗",
            BlockStage::MemoryL0 => "█",
            BlockStage::MemoryL1 => "█",
            BlockStage::MemoryL2 => "█",
            BlockStage::MemoryL3 => "█",
        }
    }

    fn name(&self) -> &'static str {
        match self {
            BlockStage::Queued => "Queued",
            BlockStage::Uploading => "Uploading",
            BlockStage::Enriching => "Enriching",
            BlockStage::Embedding => "Embedding",
            BlockStage::AbstractingL0toL1 => "L0→L1",
            BlockStage::AbstractingL1toL2 => "L1→L2",
            BlockStage::AbstractingL2toL3 => "L2→L3",
            BlockStage::Completed => "Completed",
            BlockStage::Failed => "Failed",
            BlockStage::MemoryL0 => "L0",
            BlockStage::MemoryL1 => "L1",
            BlockStage::MemoryL2 => "L2",
            BlockStage::MemoryL3 => "L3",
        }
    }

    /// Whether this stage represents an actively processing state (should blink).
    /// Only the current in-flight batch blinks, not already-enriched/embedded chunks.
    fn is_active(&self) -> bool {
        matches!(
            self,
            BlockStage::Uploading
                | BlockStage::Enriching
                | BlockStage::AbstractingL0toL1
                | BlockStage::AbstractingL1toL2
                | BlockStage::AbstractingL2toL3
        )
    }
}

pub struct VizApp {
    data: VizData,
    should_quit: bool,
    /// Current log level filter. None = logs hidden; Some(level) = show logs at that level and above.
    log_level: Option<Level>,
    log_entries: VecDeque<LogEntry>,
    /// Index of last captured log entry consumed from the global buffer
    log_cursor: usize,
    /// Previous snapshot for change detection
    prev_chunks: std::collections::HashMap<String, Vec<ChunkStatus>>,
    prev_abstractions: usize,
    /// Tick counter for blinking animation (toggled every ~500ms)
    tick: u64,
    last_tick: Instant,
    /// Previous summary for detecting enrichment progress changes
    prev_summaries: std::collections::HashMap<String, String>,
    /// Scroll offset for the activity log (0 = bottom / most recent)
    log_scroll: usize,
}

#[derive(Debug, Clone)]
struct LogEntry {
    timestamp: String,
    message: String,
    level: Level,
}

fn level_color(level: Level) -> Color {
    match level {
        Level::ERROR => Color::Red,
        Level::WARN => Color::Yellow,
        Level::INFO => Color::White,
        Level::DEBUG => Color::Cyan,
        Level::TRACE => Color::DarkGray,
    }
}

/// Cycle log levels: off → info → warn → debug → trace → off
fn cycle_log_level(current: Option<Level>) -> Option<Level> {
    match current {
        None => Some(Level::INFO),
        Some(Level::INFO) => Some(Level::WARN),
        Some(Level::WARN) => Some(Level::DEBUG),
        Some(Level::DEBUG) => Some(Level::TRACE),
        _ => None,
    }
}

fn log_level_label(level: Option<Level>) -> &'static str {
    match level {
        None => "off",
        Some(Level::INFO) => "info",
        Some(Level::WARN) => "warn",
        Some(Level::DEBUG) => "debug",
        Some(Level::TRACE) => "trace",
        Some(Level::ERROR) => "error",
    }
}

impl VizApp {
    pub fn new() -> Self {
        let mut app = Self {
            data: VizData::default(),
            should_quit: false,
            log_level: Some(Level::INFO),
            log_entries: VecDeque::new(),
            log_cursor: 0,
            prev_chunks: std::collections::HashMap::new(),
            prev_abstractions: 0,
            tick: 0,
            last_tick: Instant::now(),
            prev_summaries: std::collections::HashMap::new(),
            log_scroll: 0,
        };
        app.push_log(Level::INFO, "Viz started. Monitoring sessions...".into());
        app
    }

    fn advance_tick(&mut self) {
        if self.last_tick.elapsed() >= Duration::from_millis(500) {
            self.tick = self.tick.wrapping_add(1);
            self.last_tick = Instant::now();
        }
    }

    fn push_log(&mut self, level: Level, message: String) {
        let now = chrono::Local::now().format("%H:%M:%S").to_string();
        self.log_entries.push_back(LogEntry {
            timestamp: now,
            message,
            level,
        });
        while self.log_entries.len() > MAX_LOG_ENTRIES {
            self.log_entries.pop_front();
        }
    }

    /// Drain new entries from the global tracing log buffer.
    fn drain_tracing_logs(&mut self) {
        let buffer = log_capture::global_log_buffer();
        if let Ok(buf) = buffer.lock() {
            // Only consume entries we haven't seen yet
            let new_count = buf.len();
            if new_count > self.log_cursor {
                for entry in buf.iter().skip(self.log_cursor) {
                    self.log_entries.push_back(LogEntry {
                        timestamp: entry.timestamp.clone(),
                        message: entry.message.clone(),
                        level: entry.level,
                    });
                }
                self.log_cursor = new_count;
            }
            while self.log_entries.len() > MAX_LOG_ENTRIES {
                self.log_entries.pop_front();
            }
        }
    }

    /// Collect data and diff against previous snapshot to produce log entries.
    pub async fn collect_data(&mut self, system: &System) {
        let mut sessions = Vec::new();

        if let Ok(banks) = system.bank_manager.list_banks().await {
            for bank_info in banks {
                let bank_name = bank_info.name;
                if let Some(session_manager) = system.bank_manager.get_session_manager(&bank_name).await
                    && let Ok(active_sessions) = session_manager.list_active_sessions() {
                        for session in active_sessions {
                            if let Ok(chunks) = session_manager.get_session_chunk_progress(&session.session_id) {
                                // Calculate progress using processing_result for accurate tracking
                                let progress_percent = if let Some(ref pr) = session.processing_result {
                                    if pr.total_chunks > 0 {
                                        // Weight: enrichment = 30%, storage = 70%
                                        let enrich_pct = pr.chunks_enriched as f32 / pr.total_chunks as f32 * 30.0;
                                        let store_pct = pr.chunks_processed as f32 / pr.total_chunks as f32 * 70.0;
                                        enrich_pct + store_pct
                                    } else {
                                        0.0
                                    }
                                } else if session.expected_parts > 0 {
                                    // Fallback: upload progress
                                    (session.received_parts as f32 / session.expected_parts as f32) * 100.0
                                } else {
                                    0.0
                                };

                                sessions.push(VizSession {
                                    session_id: session.session_id.clone(),
                                    file_name: session.metadata.file_name.clone(),
                                    chunks,
                                    status: session.status.as_str().to_string(),
                                    progress_percent,
                                });

                                // Log processing summary changes (enrichment/storage progress)
                                if let Some(ref pr) = session.processing_result
                                    && let Some(ref summary) = pr.summary {
                                        let prev_summary = self.prev_summaries.get(&session.session_id);
                                        if prev_summary.map(|s| s.as_str()) != Some(summary.as_str()) {
                                            self.push_log(
                                                Level::INFO,
                                                format!("{}: {}", session.metadata.file_name, summary),
                                            );
                                            self.prev_summaries.insert(
                                                session.session_id.clone(),
                                                summary.clone(),
                                            );
                                        }
                                    }
                            }
                        }
                    }
            }
        }

        let mut abstractions = Vec::new();
        let mut processing_ids = std::collections::HashSet::new();
        let mut pipeline_info = None;
        if let Some(pipeline) = system.bank_manager.get_abstraction_pipeline().await {
            let pending = pipeline.get_pending_abstractions();
            pipeline_info = Some(VizPipelineInfo {
                enabled: pipeline.config.enabled,
                pending_count: pending.len(),
                min_memories_for_l1: pipeline.config.min_memories_for_l1,
                delay_secs: pipeline.config.l1_processing_delay.as_secs(),
            });
            for task in pending {
                processing_ids.insert(task.memory_id);
                abstractions.push(VizAbstraction {
                    memory_id: task.memory_id,
                    current_level: task.current_level,
                    target_level: task.target_level,
                    retry_count: task.retry_count,
                });
            }
        }

        // Collect memory counts by layer from all banks
        let mut layer_counts: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
        let mut viz_banks = Vec::new();
        if let Ok(banks) = system.bank_manager.list_banks().await {
            for bank_info in &banks {
                viz_banks.push(VizBankInfo {
                    name: bank_info.name.clone(),
                    memory_count: bank_info.memory_count,
                    loaded: bank_info.loaded,
                });
                if let Ok(bank) = system.bank_manager.get_or_create(&bank_info.name).await {
                    for level in 0..=3 {
                        let mut f = llm_mem::types::Filters::new();
                        f.min_layer_level = Some(level);
                        f.max_layer_level = Some(level);
                        if let Ok(memories) = bank.list(&f, None).await {
                            *layer_counts.entry(level).or_insert(0) += memories.len();
                        }
                    }
                }
            }
        }
        let memory_layers: Vec<(i32, usize)> = layer_counts.into_iter().collect();

        // --- Diff and log changes ---
        for session in &sessions {
            let cur_statuses: Vec<ChunkStatus> = session.chunks.iter().map(|c| c.status.clone()).collect();
            if let Some(prev) = self.prev_chunks.get(&session.session_id) {
                let prev_completed = prev.iter().filter(|s| **s == ChunkStatus::Completed).count();
                let cur_completed = cur_statuses.iter().filter(|s| **s == ChunkStatus::Completed).count();
                let prev_failed = prev.iter().filter(|s| **s == ChunkStatus::Failed).count();
                let cur_failed = cur_statuses.iter().filter(|s| **s == ChunkStatus::Failed).count();
                let prev_enriched = prev.iter().filter(|s| **s == ChunkStatus::Enriching || **s == ChunkStatus::Embedding || **s == ChunkStatus::Completed).count();
                let cur_enriched = cur_statuses.iter().filter(|s| **s == ChunkStatus::Enriching || **s == ChunkStatus::Embedding || **s == ChunkStatus::Completed).count();

                let new_completed = cur_completed.saturating_sub(prev_completed);
                let new_failed = cur_failed.saturating_sub(prev_failed);
                let new_enriched = cur_enriched.saturating_sub(prev_enriched);

                if new_completed > 0 {
                    self.push_log(
                        Level::INFO,
                        format!("{}: {} chunk(s) stored ({}/{})", session.file_name, new_completed, cur_completed, session.chunks.len()),
                    );
                }
                if new_enriched > 0 && new_completed == 0 {
                    self.push_log(
                        Level::INFO,
                        format!("{}: {} chunk(s) enriched ({}/{})", session.file_name, new_enriched, cur_enriched, session.chunks.len()),
                    );
                }
                if new_failed > 0 {
                    self.push_log(
                        Level::ERROR,
                        format!("{}: {} chunk(s) failed", session.file_name, new_failed),
                    );
                }
            } else {
                // New session appeared
                self.push_log(
                    Level::INFO,
                    format!("New session: {} ({} chunks)", session.file_name, session.chunks.len()),
                );
            }
            self.prev_chunks.insert(session.session_id.clone(), cur_statuses);
        }

        let abs_count = abstractions.len();
        if abs_count != self.prev_abstractions && abs_count > 0 {
            self.push_log(
                Level::INFO,
                format!("{} pending abstraction(s)", abs_count),
            );
        }
        self.prev_abstractions = abs_count;

        self.data.sessions = sessions;
        self.data.abstractions = abstractions;
        self.data.memory_layers = memory_layers;
        self.data.processing_ids = processing_ids;
        self.data.banks = viz_banks;
        self.data.pipeline_info = pipeline_info;

        // Drain captured tracing logs (LLM requests, responses, etc.)
        self.drain_tracing_logs();
    }
}

pub async fn handle_viz(system: &System, _bank: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    // Suppress stderr logging while TUI owns the screen
    log_capture::set_tui_active(true);

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = VizApp::new();
    app.collect_data(system).await;

    let result = run_app(&mut terminal, &mut app, system).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    // Restore stderr logging
    log_capture::set_tui_active(false);

    result
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut VizApp,
    system: &System,
) -> Result<(), Box<dyn std::error::Error>> {
    let poll_interval = Duration::from_millis(500);

    loop {
        if event::poll(poll_interval)? {
            match event::read()? {
                Event::Key(key) => {
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Esc | KeyCode::Char('q') => {
                                app.should_quit = true;
                                break;
                            }
                            KeyCode::Char('l') => app.log_level = cycle_log_level(app.log_level),
                            KeyCode::Up => {
                                app.log_scroll = app.log_scroll.saturating_add(1);
                            }
                            KeyCode::Down => {
                                app.log_scroll = app.log_scroll.saturating_sub(1);
                            }
                            _ => {}
                        }
                    }
                }
                Event::Resize(..) => {}
                _ => {}
            }
        }

        app.collect_data(system).await;
        app.advance_tick();
        terminal.draw(|f| ui(f, app, f.area()))?;

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

// ── UI Layout ──────────────────────────────────────────────────────────────

fn ui(f: &mut Frame, app: &VizApp, area: Rect) {
    // Outer border wrapping the entire screen
    let outer = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " llm-mem viz ",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ));
    let inner = outer.inner(area);
    f.render_widget(outer, area);

    if app.log_level.is_some() {
        let rows: [Rect; 4] = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // status bar
                Constraint::Min(6),     // grid + sessions
                Constraint::Length(2),  // legend
                Constraint::Length(10), // log panel
            ])
            .areas(inner);

        render_status_bar(f, app, rows[0]);
        render_main_area(f, app, rows[1]);
        render_legend(f, rows[2]);
        render_log_panel(f, app, rows[3]);
    } else {
        let rows: [Rect; 3] = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // status bar
                Constraint::Min(6),   // grid + sessions
                Constraint::Length(2), // legend
            ])
            .areas(inner);

        render_status_bar(f, app, rows[0]);
        render_main_area(f, app, rows[1]);
        render_legend(f, rows[2]);
    }
}

fn render_status_bar(f: &mut Frame, app: &VizApp, area: Rect) {
    let data = &app.data;

    let enriched: usize = data.sessions.iter()
        .flat_map(|s| s.chunks.iter())
        .filter(|c| matches!(c.status, ChunkStatus::Embedding | ChunkStatus::Completed))
        .count();
    let completed: usize = data.sessions.iter()
        .flat_map(|s| s.chunks.iter())
        .filter(|c| c.status == ChunkStatus::Completed)
        .count();
    let total: usize = data.sessions.iter()
        .flat_map(|s| s.chunks.iter())
        .count();
    let failed: usize = data.sessions.iter()
        .flat_map(|s| s.chunks.iter())
        .filter(|c| c.status == ChunkStatus::Failed)
        .count();

    // Overall progress: weighted 30% enrichment + 70% storage
    let pct = if total > 0 {
        let enrich_pct = enriched as f32 / total as f32 * 30.0;
        let store_pct = completed as f32 / total as f32 * 70.0;
        (enrich_pct + store_pct) as u32
    } else {
        0
    };

    let log_label = log_level_label(app.log_level);

    // Memory layer summary
    let total_memories: usize = data.memory_layers.iter().map(|(_, c)| c).sum();
    let layer_detail: String = data.memory_layers.iter()
        .filter(|(_, c)| *c > 0)
        .map(|(l, c)| format!("L{}:{}", l, c))
        .collect::<Vec<_>>()
        .join(" ");

    let phase = if total > 0 && completed == total {
        "Done"
    } else if total > 0 && enriched < total {
        "Enriching"
    } else if total > 0 && completed < total {
        "Storing"
    } else {
        "Idle"
    };

    let chunks_detail = if total > 0 && enriched < total {
        // During enrichment phase, show enriched progress
        format!("Enriched: {}/{}  Stored: {}/{}", enriched, total, completed, total)
    } else if total > 0 && completed < total {
        // During storage phase, show stored progress
        format!("Enriched: {}/{}  Stored: {}/{}", enriched, total, completed, total)
    } else {
        format!("Chunks: {}/{}", completed, total)
    };

    let mem_info = if total_memories > 0 {
        format!("  Memories: {} ({})  Processing: {}", total_memories, layer_detail, data.processing_ids.len())
    } else {
        String::new()
    };

    let text = format!(
        " Sessions: {}  {}  Failed: {}  Abstractions: {}  Phase: {}  Progress: {}%{}    [q] quit  [l] log:{}  [↑↓] scroll ",
        data.sessions.len(),
        chunks_detail,
        failed,
        data.abstractions.len(),
        phase,
        pct,
        mem_info,
        log_label,
    );

    let bar = Paragraph::new(text)
        .style(Style::default().fg(Color::White))
        .block(Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(bar, area);
}

fn render_main_area(f: &mut Frame, app: &VizApp, area: Rect) {
    let has_memories = app.data.memory_layers.iter().any(|(_, c)| *c > 0);
    if app.data.sessions.is_empty() && app.data.abstractions.is_empty() && !has_memories {
        let msg = Paragraph::new("No active sessions. Upload a document to see live progress.")
            .style(Style::default().fg(Color::DarkGray))
            .wrap(Wrap { trim: true })
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" Chunks "));
        f.render_widget(msg, area);
        return;
    }

    let cols: [Rect; 2] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(20), Constraint::Length(36)])
        .areas(area);

    render_grid(f, app, cols[0]);
    render_session_list(f, app, cols[1]);
}

fn render_grid(f: &mut Frame, app: &VizApp, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(" Memory Bank ", Style::default().fg(Color::White)));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let max_cols = inner.width.max(1);
    let max_rows = inner.height;
    let blink_on = app.tick.is_multiple_of(2);
    let mut state = GridState::new(max_cols, max_rows);

    // Render document session chunks first
    for session in &app.data.sessions {
        for chunk in &session.chunks {
            if state.row >= state.max_rows { break; }
            state.push_with_blink(&chunk_to_stage(chunk), blink_on);
        }
        if state.row >= state.max_rows { break; }
    }

    // Render all bank memories by layer, with active processing ones blinking
    for &(level, count) in &app.data.memory_layers {
        let stage = match level {
            0 => BlockStage::MemoryL0,
            1 => BlockStage::MemoryL1,
            2 => BlockStage::MemoryL2,
            _ => BlockStage::MemoryL3,
        };
        // For L0 memories: some may be actively being processed to L1
        // Show those as blinking AbstractingL0toL1 instead
        let (active_count, settled_count) = if level == 0 {
            let active = app.data.abstractions.iter()
                .filter(|a| a.current_level == 0 && a.target_level == 1)
                .count();
            (active.min(count), count.saturating_sub(active))
        } else if level == 1 {
            let active = app.data.abstractions.iter()
                .filter(|a| a.current_level == 1 && a.target_level == 2)
                .count();
            (active.min(count), count.saturating_sub(active))
        } else if level == 2 {
            let active = app.data.abstractions.iter()
                .filter(|a| a.current_level == 2 && a.target_level == 3)
                .count();
            (active.min(count), count.saturating_sub(active))
        } else {
            (0, count)
        };

        // Render active (blinking) blocks
        let active_stage = match level {
            0 => BlockStage::AbstractingL0toL1,
            1 => BlockStage::AbstractingL1toL2,
            2 => BlockStage::AbstractingL2toL3,
            _ => stage.clone(),
        };
        for _ in 0..active_count {
            if state.row >= state.max_rows { break; }
            state.push_with_blink(&active_stage, blink_on);
        }

        // Render settled (non-blinking) blocks
        for _ in 0..settled_count {
            if state.row >= state.max_rows { break; }
            state.push_with_blink(&stage, blink_on);
        }
    }

    if !state.current_spans.is_empty() {
        state.flush_line();
    }

    let grid = Paragraph::new(state.lines);
    f.render_widget(grid, inner);
}

fn render_session_list(f: &mut Frame, app: &VizApp, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(" Sessions ", Style::default().fg(Color::White)));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line<'static>> = Vec::new();

    // Active upload sessions
    for session in &app.data.sessions {
        let enriched = session.chunks.iter().filter(|c| matches!(c.status, ChunkStatus::Embedding | ChunkStatus::Completed)).count();
        let completed = session.chunks.iter().filter(|c| c.status == ChunkStatus::Completed).count();
        let total = session.chunks.len();

        let name = if session.file_name.len() > 30 {
            format!("{}…", &session.file_name[..29])
        } else {
            session.file_name.clone()
        };

        let status_color = match session.status.as_str() {
            "processing" => Color::Yellow,
            "uploading" => Color::Blue,
            "completed" => Color::Green,
            "failed" => Color::Red,
            _ => Color::White,
        };

        lines.push(Line::from(vec![
            Span::styled("● ", Style::default().fg(status_color)),
            Span::styled(name, Style::default().fg(Color::White)),
        ]));

        let detail = if enriched < total {
            format!("enriching {}/{}", enriched, total)
        } else if completed < total {
            format!("storing {}/{}", completed, total)
        } else {
            format!("{}/{} done", completed, total)
        };

        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                format!("{} ", session.status),
                Style::default().fg(status_color),
            ),
            Span::styled(
                detail,
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        if lines.len() < inner.height as usize {
            lines.push(Line::from(""));
        }
    }

    // When no active sessions, show bank & pipeline info
    if app.data.sessions.is_empty() {
        // Banks
        for bank in &app.data.banks {
            let status = if bank.loaded { "●" } else { "○" };
            let status_color = if bank.loaded { Color::Green } else { Color::DarkGray };
            lines.push(Line::from(vec![
                Span::styled(format!("{} ", status), Style::default().fg(status_color)),
                Span::styled(bank.name.clone(), Style::default().fg(Color::White)),
            ]));
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format!("{} memories", bank.memory_count),
                    Style::default().fg(Color::DarkGray),
                ),
            ]));
        }

        // Layer breakdown
        if !app.data.memory_layers.is_empty() {
            if !lines.is_empty() && lines.len() < inner.height as usize {
                lines.push(Line::from(""));
            }
            lines.push(Line::from(vec![
                Span::styled("Layers", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            ]));
            for &(level, count) in &app.data.memory_layers {
                if count == 0 { continue; }
                let color = match level {
                    0 => Color::Rgb(255, 165, 0),
                    1 => Color::Red,
                    2 => Color::Magenta,
                    _ => Color::Cyan,
                };
                let label = match level {
                    0 => "L0 raw",
                    1 => "L1 structural",
                    2 => "L2 semantic",
                    _ => "L3 concept",
                };
                lines.push(Line::from(vec![
                    Span::styled("  █ ", Style::default().fg(color)),
                    Span::styled(
                        format!("{}: {}", label, count),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
            }
        }

        // Pipeline status
        if let Some(ref pi) = app.data.pipeline_info {
            if lines.len() < inner.height as usize {
                lines.push(Line::from(""));
            }
            lines.push(Line::from(vec![
                Span::styled("Pipeline", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            ]));
            let (status_str, status_color) = if !pi.enabled {
                ("disabled", Color::DarkGray)
            } else if pi.pending_count > 0 {
                ("active", Color::Yellow)
            } else {
                ("idle", Color::Green)
            };
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(status_str, Style::default().fg(status_color)),
            ]));
            if pi.pending_count > 0 {
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(
                        format!("{} pending", pi.pending_count),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
            }
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format!("min={} delay={}s", pi.min_memories_for_l1, pi.delay_secs),
                    Style::default().fg(Color::DarkGray),
                ),
            ]));
        }
    }

    let paragraph = Paragraph::new(lines);
    f.render_widget(paragraph, inner);
}

fn render_legend(f: &mut Frame, area: Rect) {
    let stages = [
        BlockStage::MemoryL0,
        BlockStage::MemoryL1,
        BlockStage::MemoryL2,
        BlockStage::MemoryL3,
        BlockStage::AbstractingL0toL1,
        BlockStage::AbstractingL1toL2,
        BlockStage::AbstractingL2toL3,
        BlockStage::Queued,
        BlockStage::Enriching,
        BlockStage::Completed,
        BlockStage::Failed,
    ];

    let mut spans = Vec::new();
    for stage in &stages {
        spans.push(Span::styled(
            format!(" {} ", stage.unicode_char()),
            Style::default().fg(stage.color()),
        ));
        spans.push(Span::styled(
            stage.name(),
            Style::default().fg(stage.color()),
        ));
        spans.push(Span::raw("  "));
    }

    let legend = Paragraph::new(Line::from(spans))
        .block(Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(legend, area);
}

fn render_log_panel(f: &mut Frame, app: &VizApp, area: Rect) {
    // Filter entries by current log level
    let filter_level = app.log_level.unwrap_or(Level::ERROR);
    let filtered: Vec<&LogEntry> = app.log_entries
        .iter()
        .filter(|e| e.level <= filter_level)
        .collect();

    // Compute inner area for visible line count (block borders take 2 rows)
    let visible = area.height.saturating_sub(2) as usize;
    // Clamp scroll so we don't scroll past the beginning
    let max_scroll = filtered.len().saturating_sub(visible);
    let scroll = app.log_scroll.min(max_scroll);
    let start = max_scroll.saturating_sub(scroll);
    let lines: Vec<Line> = filtered
        .iter()
        .skip(start)
        .take(visible)
        .map(|entry| {
            Line::from(vec![
                Span::styled(
                    format!("{} ", entry.timestamp),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    entry.message.clone(),
                    Style::default().fg(level_color(entry.level)),
                ),
            ])
        })
        .collect();

    let title = if scroll > 0 {
        format!(" Activity Log ({}) ↑{} ", log_level_label(app.log_level), scroll)
    } else {
        format!(" Activity Log ({}) ", log_level_label(app.log_level))
    };

    let paragraph = Paragraph::new(lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(Span::styled(title, Style::default().fg(Color::White))));
    f.render_widget(paragraph, area);
}

// ── Grid helpers ───────────────────────────────────────────────────────────

struct GridState {
    lines: Vec<Line<'static>>,
    current_spans: Vec<Span<'static>>,
    col: u16,
    row: u16,
    max_rows: u16,
    max_cols: u16,
}

impl GridState {
    fn new(max_cols: u16, max_rows: u16) -> Self {
        Self {
            lines: Vec::new(),
            current_spans: Vec::new(),
            col: 0,
            row: 0,
            max_rows,
            max_cols,
        }
    }

    fn push_with_blink(&mut self, stage: &BlockStage, blink_on: bool) {
        let style = if stage.is_active() && !blink_on {
            // Blink to target layer color on alternate ticks
            let target_color = match stage {
                BlockStage::AbstractingL0toL1 => Color::Red,     // Target: L1
                BlockStage::AbstractingL1toL2 => Color::Magenta, // Target: L2
                BlockStage::AbstractingL2toL3 => Color::Cyan,    // Target: L3
                _ => Color::DarkGray,
            };
            Style::default().fg(target_color)
        } else {
            Style::default().fg(stage.color())
        };
        self.current_spans.push(Span::styled(
            stage.unicode_char().to_string(),
            style,
        ));
        self.col += 1;
        if self.col >= self.max_cols {
            self.flush_line();
        }
    }

    fn flush_line(&mut self) {
        self.lines.push(Line::from(std::mem::take(&mut self.current_spans)));
        self.col = 0;
        self.row += 1;
    }
}

fn chunk_to_stage(chunk: &ChunkProgress) -> BlockStage {
    match chunk.status {
        ChunkStatus::Queued => BlockStage::Queued,
        ChunkStatus::Uploading => BlockStage::Uploading,
        ChunkStatus::Enriching => BlockStage::Enriching,
        ChunkStatus::Embedding => BlockStage::Embedding,
        ChunkStatus::Completed => BlockStage::Completed,
        ChunkStatus::Failed => BlockStage::Failed,
    }
}


