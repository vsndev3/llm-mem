//! Progressive context resume — exponential decay curve over memory layers.
//!
//! Returns a compact but comprehensive snapshot of recent work that an LLM can
//! use to resume context. The most recent time window returns L0 memories at
//! full precision; progressively older windows fetch higher-layer abstractions
//! (L1 summaries, L2 semantic links, L3 concepts), producing a logarithmic
//! "dumbbell" precision curve that peaks at the current time.

use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::memory::MemoryManager;
use crate::operations::serialization::memory_to_json;
use crate::types::{Filters, MemoryState};

use super::OperationError;

// ─── Response types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextResumeSegment {
    pub label: String,
    pub layer: i32,
    pub start: String,
    pub end: String,
    pub duration_secs: i64,
    pub count: usize,
    pub memories: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextResumeResponse {
    pub start: String,
    pub end: String,
    pub total_lookback_secs: i64,
    pub decay_factor: f64,
    pub segment_count: usize,
    pub segments: Vec<ContextResumeSegment>,
    pub total_memories: usize,
}

// ─── Service ───────────────────────────────────────────────────────────────

pub struct ContextResumeService {
    manager: Arc<MemoryManager>,
}

impl ContextResumeService {
    pub fn new(manager: Arc<MemoryManager>) -> Self {
        Self { manager }
    }

    /// Main entry point.
    pub async fn get_context_resume(
        &self,
        end: Option<&str>,
        lookback_secs: i64,
        decay_factor: f64,
        segments: usize,
        max_per_segment: usize,
        bank_filters: ResumeFilters,
    ) -> Result<ContextResumeResponse, OperationError> {
        let end_dt = match end {
            Some(s) => parse_iso(s, "end")?,
            None => Utc::now(),
        };
        let start_dt = end_dt - Duration::seconds(lookback_secs);
        let total_lookback_secs = (end_dt - start_dt).num_seconds();

        let boundaries = compute_segments(start_dt, end_dt, segments, decay_factor);

        let mut result_segments: Vec<ContextResumeSegment> = Vec::with_capacity(segments);

        // boundaries[0] = most recent segment, boundaries[last] = oldest.
        for (i, (seg_start, seg_end)) in boundaries.iter().enumerate() {
            let target_layer = i32::try_from(i).unwrap_or(i32::MAX).min(3);
            let label = segment_label(seg_start, seg_end, &end_dt);
            let duration_secs = (*seg_end - *seg_start).num_seconds();

            let mems = self
                .fetch_segment(*seg_start, *seg_end, target_layer, max_per_segment, &bank_filters)
                .await?;

            let count = mems.len();
            result_segments.push(ContextResumeSegment {
                label,
                layer: target_layer,
                start: seg_start.to_rfc3339(),
                end: seg_end.to_rfc3339(),
                duration_secs,
                count,
                memories: mems.iter().map(memory_to_json).collect(),
            });
        }

        // Reverse so the output reads oldest → newest (chronological).
        result_segments.reverse();

        let total_memories: usize = result_segments.iter().map(|s| s.count).sum();

        Ok(ContextResumeResponse {
            start: start_dt.to_rfc3339(),
            end: end_dt.to_rfc3339(),
            total_lookback_secs,
            decay_factor,
            segment_count: result_segments.len(),
            segments: result_segments,
            total_memories,
        })
    }

    /// Fetch memories for one segment. Tries the target layer first, then
    /// falls back to progressively lower layers until results are found.
    async fn fetch_segment(
        &self,
        seg_start: DateTime<Utc>,
        seg_end: DateTime<Utc>,
        target_layer: i32,
        max: usize,
        filters: &ResumeFilters,
    ) -> Result<Vec<crate::types::Memory>, OperationError> {
        for layer in (0..=target_layer).rev() {
            let f = Filters {
                event_after: Some(seg_start),
                event_before: Some(seg_end),
                min_layer_level: Some(layer),
                max_layer_level: Some(layer),
                user_id: filters.user_id.clone(),
                agent_id: filters.agent_id.clone(),
                topics: filters.topics.clone(),
                ..Filters::default()
            };

            let mut mems = self
                .manager
                .list(&f, Some(max + 100))
                .await
                .map_err(|e| OperationError::Runtime(format!("list failed: {e}")))?;

            mems.retain(|m| m.metadata.state == MemoryState::Active);
            mems.sort_by_key(|m| m.effective_event_at());
            mems.truncate(max);

            if !mems.is_empty() {
                return Ok(mems);
            }
        }
        Ok(Vec::new())
    }
}

// ─── Segment math ──────────────────────────────────────────────────────────

/// Compute segment boundaries using exponential decay.
/// Returns `segments` tuples of `(start, end)`, ordered most-recent first.
pub(crate) fn compute_segments(
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    segments: usize,
    decay_factor: f64,
) -> Vec<(DateTime<Utc>, DateTime<Utc>)> {
    if segments == 0 {
        return Vec::new();
    }

    let total_secs = (window_end - window_start).num_seconds() as f64;
    if total_secs <= 0.0 {
        return vec![(window_start, window_end)];
    }

    // Geometric series: weight_i = decay_factor^i
    let total_weight: f64 = (0..segments).map(|i| decay_factor.powi(i as i32)).sum();

    let mut boundaries = Vec::with_capacity(segments);
    let mut cursor = window_end;

    for i in 0..segments {
        let weight = decay_factor.powi(i as i32);
        let seg_secs = (total_secs * weight / total_weight).round() as i64;
        let seg_start = cursor - Duration::seconds(seg_secs);

        // Clamp to window start for the oldest segment.
        let clamped_start = if seg_start < window_start {
            window_start
        } else {
            seg_start
        };

        boundaries.push((clamped_start, cursor));
        cursor = clamped_start;
    }

    boundaries
}

/// Human-readable label for a segment relative to "now" (end_dt).
fn segment_label(
    seg_start: &DateTime<Utc>,
    seg_end: &DateTime<Utc>,
    _end_dt: &DateTime<Utc>,
) -> String {
    let duration = *seg_end - *seg_start;
    let secs = duration.num_seconds();

    if secs >= 86400 * 30 {
        let months = secs / (86400 * 30);
        format!("{}mo starting {}", months, seg_start.format("%Y-%m-%d"))
    } else if secs >= 86400 * 7 {
        let weeks = secs / (86400 * 7);
        format!(
            "{}w {} — {}",
            weeks,
            seg_start.format("%Y-%m-%d"),
            seg_end.format("%Y-%m-%d")
        )
    } else if secs >= 86400 {
        let days = secs / 86400;
        format!(
            "{}d {} — {}",
            days,
            seg_start.format("%m-%d"),
            seg_end.format("%m-%d")
        )
    } else if secs >= 3600 {
        let hours = secs / 3600;
        format!(
            "{}h {} — {}",
            hours,
            seg_start.format("%H:%M"),
            seg_end.format("%H:%M")
        )
    } else {
        format!("{}s", secs)
    }
}

fn parse_iso(s: &str, label: &str) -> Result<DateTime<Utc>, OperationError> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| {
            OperationError::InvalidInput(format!(
                "{label} must be valid ISO 8601 (got '{s}': {e})"
            ))
        })
}

/// Lightweight filter struct that the MCP layer passes in (user_id, agent_id, topics).
#[derive(Debug, Clone, Default)]
pub struct ResumeFilters {
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub topics: Option<Vec<String>>,
}

/// Parse a lookback string like "30d", "12h", "6w" into seconds.
pub fn parse_lookback(s: &str) -> Result<i64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("lookback string is empty".into());
    }
    let (num_str, unit) = s.split_at(s.len() - 1);
    let num: i64 = num_str
        .parse()
        .map_err(|_| format!("invalid lookback '{s}' (expected e.g. '30d', '12h', '1w')"))?;
    match unit.to_ascii_lowercase().as_str() {
        "s" => Ok(num),
        "m" => Ok(num * 60),
        "h" => Ok(num * 3600),
        "d" => Ok(num * 86400),
        "w" => Ok(num * 86400 * 7),
        other => Err(format!("unknown unit '{other}' (use s/m/h/d/w)")),
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn parse_lookback_days() {
        assert_eq!(parse_lookback("30d").unwrap(), 30 * 86400);
    }

    #[test]
    fn parse_lookback_hours() {
        assert_eq!(parse_lookback("12h").unwrap(), 12 * 3600);
    }

    #[test]
    fn parse_lookback_rejects_bad() {
        assert!(parse_lookback("5x").is_err());
    }

    #[test]
    fn compute_segments_basic() {
        let start = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2026, 2, 1, 0, 0, 0).unwrap(); // 31 days

        let segs = compute_segments(start, end, 5, 2.0);

        assert_eq!(segs.len(), 5);

        // Most recent segment should end at `end`.
        assert_eq!(segs[0].1, end);

        // Oldest segment should start at `start`.
        assert_eq!(segs[4].0, start);

        // Segments should tile the window with no gaps (within rounding).
        for i in 0..segs.len() - 1 {
            assert_eq!(segs[i].0, segs[i + 1].1);
        }
    }

    #[test]
    fn compute_segments_exponential_growth() {
        let start = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2026, 2, 1, 0, 0, 0).unwrap();

        let segs = compute_segments(start, end, 4, 2.0);

        // With decay_factor=2, each segment should be ~2x the previous.
        let d0 = (segs[0].1 - segs[0].0).num_seconds();
        let d1 = (segs[1].1 - segs[1].0).num_seconds();
        let d2 = (segs[2].1 - segs[2].0).num_seconds();
        let d3 = (segs[3].1 - segs[3].0).num_seconds();

        // d1 ≈ 2*d0, d2 ≈ 2*d1, d3 ≈ 2*d2 (within rounding).
        let ratio_1_0 = d1 as f64 / d0 as f64;
        let ratio_2_1 = d2 as f64 / d1 as f64;
        let ratio_3_2 = d3 as f64 / d2 as f64;

        assert!((ratio_1_0 - 2.0).abs() < 0.1, "ratio_1_0 = {ratio_1_0}");
        assert!((ratio_2_1 - 2.0).abs() < 0.1, "ratio_2_1 = {ratio_2_1}");
        assert!((ratio_3_2 - 2.0).abs() < 0.1, "ratio_3_2 = {ratio_3_2}");
    }

    #[test]
    fn compute_segments_single() {
        let start = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2026, 1, 2, 0, 0, 0).unwrap();

        let segs = compute_segments(start, end, 1, 2.0);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].0, start);
        assert_eq!(segs[0].1, end);
    }

    #[test]
    fn compute_segments_zero_returns_empty() {
        let start = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2026, 1, 2, 0, 0, 0).unwrap();
        let segs = compute_segments(start, end, 0, 2.0);
        assert!(segs.is_empty());
    }

    #[test]
    fn segment_label_days() {
        let start = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2026, 6, 4, 0, 0, 0).unwrap();
        let now = end;
        let label = segment_label(&start, &end, &now);
        assert!(label.contains("3d"), "label = {label}");
    }

    #[test]
    fn segment_label_weeks() {
        let start = Utc.with_ymd_and_hms(2026, 5, 18, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();
        let now = end;
        let label = segment_label(&start, &end, &now);
        assert!(label.contains("w"), "label = {label}");
    }
}
