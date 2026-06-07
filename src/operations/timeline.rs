//! Chronological / timeline views over the memory store.
//!
//! Provides two MCP-facing operations:
//! - `get_timeline` — bucketed chronological list of memories
//! - `get_timeline_graph` — nodes + edges forming a chronological graph
//!
//! Memories are bucketed by their `event_at` (the date the content refers to).
//! If `event_at` is missing, readers fall back to `created_at`. Higher-layer
//! memories (L1+) carry an `event_at` derived from their sources.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::{
    DateTime, Datelike, Duration, NaiveDate, NaiveDateTime, TimeZone, Timelike, Utc, Weekday,
};
use serde::{Deserialize, Serialize};

use crate::memory::MemoryManager;
use crate::operations::requests::{
    GetTimelineGraphRequest, GetTimelineRequest, TimelineGranularity,
};
use crate::types::{Filters, LayerInfo, Memory, MemoryState};

use super::serialization::memory_to_json;
use crate::error::MemoryError;

/// Effective (start, end) of a memory in chronological terms.
/// For L0 with `event_at`: (event_at, event_at).
/// For L1+ with a range: (event_at, event_end) — may be equal.
fn memory_window(m: &Memory) -> (DateTime<Utc>, DateTime<Utc>) {
    (m.effective_event_at(), m.effective_event_end())
}

/// Service that turns list results into chronological views.
pub struct TimelineService {
    manager: Arc<MemoryManager>,
}

impl TimelineService {
    pub fn new(manager: Arc<MemoryManager>) -> Self {
        Self { manager }
    }

    /// Resolve a `(start, end)` window from the request, defaulting to (now-7d, now).
    pub(crate) fn resolve_window(
        start: Option<&str>,
        end: Option<&str>,
    ) -> crate::error::Result<(DateTime<Utc>, DateTime<Utc>)> {
        let parse = |s: &str, label: &str| -> crate::error::Result<DateTime<Utc>> {
            DateTime::parse_from_rfc3339(s)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| {
                    MemoryError::InvalidInput(format!(
                        "{label} must be a valid ISO 8601 datetime (got '{s}': {e})"
                    ))
                })
        };
        let end_dt = match end {
            Some(s) => parse(s, "end")?,
            None => Utc::now(),
        };
        let start_dt = match start {
            Some(s) => parse(s, "start")?,
            None => end_dt - Duration::days(7),
        };
        if start_dt > end_dt {
            return Err(MemoryError::InvalidInput(format!(
                "start ({start_dt}) must be <= end ({end_dt})"
            )));
        }
        Ok((start_dt, end_dt))
    }

    /// Build a `Filters` for the timeline request — same shape used by
    /// `list_memories` and `query_memory`.
    pub(crate) fn build_filters(
        req: &GetTimelineRequest,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Filters {
        Filters {
            event_after: Some(start),
            event_before: Some(end),
            user_id: req.user_id.clone(),
            agent_id: req.agent_id.clone(),
            topics: req.topics.clone(),
            max_layer_level: if !req.include_derived { Some(0) } else { None },
            ..Filters::default()
        }
    }

    /// Top-level handler for `get_timeline`.
    pub async fn get_timeline(
        &self,
        req: GetTimelineRequest,
    ) -> crate::error::Result<TimelineResponse> {
        let (start, end) = Self::resolve_window(req.start.as_deref(), req.end.as_deref())?;
        let granularity = req.granularity.unwrap_or_default();
        let order_desc = req.order.eq_ignore_ascii_case("desc");

        let filters = Self::build_filters(&req, start, end);
        let mut memories = self
            .manager
            .list(&filters, Some(50_000))
            .await
            .map_err(|e| MemoryError::Internal(format!("list memories failed: {e}")))?;

        // Drop non-active memories (Forgotten/Processing/Invalid) — they don't belong on a timeline.
        memories.retain(|m| m.metadata.state == MemoryState::Active);

        // Sort globally by effective event_at so bucket boundaries are deterministic.
        memories.sort_by_key(|m| memory_window(m).0);
        if order_desc {
            memories.reverse();
        }

        let mut buckets = bucketize(
            &memories,
            start,
            end,
            granularity,
            req.max_results_per_bucket,
            order_desc,
        );
        buckets = finalize_bucket_order(buckets, order_desc);
        let total_count = memories.len();

        Ok(TimelineResponse {
            start: start.to_rfc3339(),
            end: end.to_rfc3339(),
            granularity,
            total_count,
            bucket_count: buckets.len(),
            buckets,
        })
    }

    /// Top-level handler for `get_timeline_graph`.
    pub async fn get_timeline_graph(
        &self,
        req: GetTimelineGraphRequest,
    ) -> crate::error::Result<TimelineGraphResponse> {
        let (start, end) =
            Self::resolve_window(req.timeline.start.as_deref(), req.timeline.end.as_deref())?;
        let granularity = req.timeline.granularity.unwrap_or_default();
        let order_desc = req.timeline.order.eq_ignore_ascii_case("desc");

        let filters = Self::build_filters(&req.timeline, start, end);
        let memories = self
            .manager
            .list(&filters, Some(50_000))
            .await
            .map_err(|e| MemoryError::Internal(format!("list memories failed: {e}")))?
            .into_iter()
            .filter(|m| m.metadata.state == MemoryState::Active)
            .collect::<Vec<_>>();

        let nodes: Vec<TimelineNode> = memories
            .iter()
            .map(|m| TimelineNode {
                id: m.id.clone(),
                event_at: m.event_at.map(|d| d.to_rfc3339()),
                event_end: m.event_end.map(|d| d.to_rfc3339()),
                layer: m.metadata.layer.level,
                bucket: Some(bucket_label(memory_window(m).0, granularity).unwrap_or_default()),
                memory: memory_to_json(m),
            })
            .collect();

        // Auto-derive temporal edges (happened_after + optional happens_within).
        let mut edges: Vec<TimelineEdge> = Vec::new();
        let mut sorted = memories.clone();
        sorted.sort_by_key(|m| memory_window(m).0);

        for win in sorted.windows(2) {
            let (a, b) = (&win[0], &win[1]);
            let (a_start, a_end) = memory_window(a);
            let (b_start, _b_end) = memory_window(b);
            if b_start <= a_end {
                continue; // No clear ordering (overlap) — skip temporal edge
            }
            let delta = (b_start - a_start).num_seconds();
            if delta <= req.temporal_edge_window_secs {
                edges.push(TimelineEdge {
                    source: a.id.clone(),
                    target: b.id.clone(),
                    edge_type: "happened_after".to_string(),
                    delta_secs: Some(delta),
                    depth: None,
                });
            }
        }
        if req.include_simultaneous {
            for i in 0..sorted.len() {
                for j in (i + 1)..sorted.len() {
                    let (a, b) = (&sorted[i], &sorted[j]);
                    let (a_start, _) = memory_window(a);
                    let (b_start, _) = memory_window(b);
                    let delta = (b_start - a_start).num_seconds().abs();
                    if delta > req.simultaneous_window_secs {
                        break; // sorted by time, no further pair can be within window
                    }
                    if a.id != b.id {
                        edges.push(TimelineEdge {
                            source: a.id.clone(),
                            target: b.id.clone(),
                            edge_type: "happens_within".to_string(),
                            delta_secs: Some(delta),
                            depth: None,
                        });
                    }
                }
            }
        }

        // Optional: traverse semantic relations up to `max_depth` from each timeline node.
        let semantic_edge_count = if req.include_semantic_edges && req.max_depth > 0 {
            let mut sem_count = 0usize;
            let depth = req.max_depth.min(3);
            // We approximate the semantic graph by scanning each node's `relations` map.
            // A more elaborate implementation would reuse GraphSearchEngine; this is a
            // pragmatic, deterministic scan for the common case.
            let by_id: std::collections::HashMap<&str, &Memory> =
                memories.iter().map(|m| (m.id.as_str(), m)).collect();
            // Track emitted edges to avoid duplicates when multiple paths reach the same target.
            let mut emitted: std::collections::HashSet<(String, String, String)> =
                std::collections::HashSet::new();
            for m in &memories {
                let mut visited = std::collections::HashSet::new();
                visited.insert(m.id.clone());
                let mut frontier: Vec<(String, usize)> = Vec::new();
                // direct relations
                for entry in m.relations.values() {
                    for tid in &entry.target_ids {
                        frontier.push((tid.to_string(), 1));
                    }
                }
                while let Some((cur_id, d)) = frontier.pop() {
                    if !visited.insert(cur_id.clone()) {
                        continue;
                    }
                    if d > depth {
                        continue;
                    }
                    if let Some(target) = by_id.get(cur_id.as_str()) {
                        for (rtype, entry) in &target.relations {
                            if let Some(whitelist) = &req.relation_types
                                && !whitelist.iter().any(|w| w == rtype)
                            {
                                continue;
                            }
                            for tid in &entry.target_ids {
                                let key = (m.id.clone(), tid.to_string(), rtype.clone());
                                if !emitted.insert(key) {
                                    continue;
                                }
                                edges.push(TimelineEdge {
                                    source: m.id.clone(),
                                    target: tid.to_string(),
                                    edge_type: rtype.clone(),
                                    delta_secs: None,
                                    depth: Some(d),
                                });
                                sem_count += 1;
                                if d < depth {
                                    frontier.push((tid.to_string(), d + 1));
                                }
                            }
                        }
                    }
                }
            }
            sem_count
        } else {
            0
        };

        let temporal_edge_count = edges
            .iter()
            .filter(|e| matches!(e.edge_type.as_str(), "happened_after" | "happens_within"))
            .count();

        // Sort edges: temporal first (by source time), then semantic.
        edges.sort_by(|a, b| {
            let a_is_temp = matches!(a.edge_type.as_str(), "happened_after" | "happens_within");
            let b_is_temp = matches!(b.edge_type.as_str(), "happened_after" | "happens_within");
            b_is_temp.cmp(&a_is_temp).then_with(|| {
                a.source
                    .cmp(&b.source)
                    .then_with(|| a.target.cmp(&b.target))
            })
        });

        Ok(TimelineGraphResponse {
            start: start.to_rfc3339(),
            end: end.to_rfc3339(),
            granularity,
            stats: TimelineGraphStats {
                node_count: nodes.len(),
                edge_count: edges.len(),
                temporal_edge_count,
                semantic_edge_count,
            },
            nodes,
            edges,
            _order_desc: order_desc,
        })
    }
}

// ─── Bucketing ────────────────────────────────────────────────────────────────

/// Floor a `DateTime<Utc>` to the start of its bucket.
pub(crate) fn floor_to_bucket(
    dt: DateTime<Utc>,
    granularity: TimelineGranularity,
) -> DateTime<Utc> {
    match granularity {
        TimelineGranularity::None => dt,
        TimelineGranularity::Hour => Utc
            .with_ymd_and_hms(dt.year(), dt.month(), dt.day(), dt.hour(), 0, 0)
            .unwrap(),
        TimelineGranularity::Day => Utc
            .with_ymd_and_hms(dt.year(), dt.month(), dt.day(), 0, 0, 0)
            .unwrap(),
        TimelineGranularity::Week => {
            // ISO weeks start on Monday; floor to the Monday 00:00 UTC of the same week.
            let weekday = dt.weekday();
            let days_from_monday = weekday.num_days_from_monday() as i64;
            let monday_date = dt.date_naive() - Duration::days(days_from_monday);
            let nd =
                NaiveDate::from_ymd_opt(monday_date.year(), monday_date.month(), monday_date.day())
                    .unwrap();
            let nt = NaiveDateTime::new(nd, chrono::NaiveTime::from_hms_opt(0, 0, 0).unwrap());
            Utc.from_utc_datetime(&nt)
        }
        TimelineGranularity::Month => Utc
            .with_ymd_and_hms(dt.year(), dt.month(), 1, 0, 0, 0)
            .unwrap(),
    }
}

/// The end of a bucket (exclusive). For simplicity we use the start of the next bucket.
pub(crate) fn next_bucket_start(
    bucket_start: DateTime<Utc>,
    granularity: TimelineGranularity,
) -> DateTime<Utc> {
    match granularity {
        TimelineGranularity::None => bucket_start + Duration::days(365 * 100), // "infinity" for a flat list
        TimelineGranularity::Hour => bucket_start + Duration::hours(1),
        TimelineGranularity::Day => bucket_start + Duration::days(1),
        TimelineGranularity::Week => bucket_start + Duration::weeks(1),
        TimelineGranularity::Month => {
            // Add one calendar month
            let nd = bucket_start.date_naive();
            let (y, m) = if nd.month() == 12 {
                (nd.year() + 1, 1)
            } else {
                (nd.year(), nd.month() + 1)
            };
            let new_date = NaiveDate::from_ymd_opt(y, m, 1).unwrap();
            let nt =
                NaiveDateTime::new(new_date, chrono::NaiveTime::from_hms_opt(0, 0, 0).unwrap());
            Utc.from_utc_datetime(&nt)
        }
    }
}

/// Human-readable label for a bucket (e.g. "2026-06-02", "2026-06-02T14:00", "2026-W22", "2026-06").
pub(crate) fn bucket_label(
    bucket_start: DateTime<Utc>,
    granularity: TimelineGranularity,
) -> Option<String> {
    match granularity {
        TimelineGranularity::None => None,
        TimelineGranularity::Hour => Some(bucket_start.format("%Y-%m-%dT%H:00").to_string()),
        TimelineGranularity::Day => Some(bucket_start.format("%Y-%m-%d").to_string()),
        TimelineGranularity::Week => Some(iso_week_label(bucket_start)),
        TimelineGranularity::Month => Some(bucket_start.format("%Y-%m").to_string()),
    }
}

fn iso_week_label(dt: DateTime<Utc>) -> String {
    let iso = dt.iso_week();
    format!("{}-W{:02}", iso.year(), iso.week())
}

pub(crate) fn bucketize(
    memories: &[Memory],
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    granularity: TimelineGranularity,
    max_per_bucket: usize,
    _order_desc: bool,
) -> Vec<TimelineBucket> {
    if memories.is_empty() {
        return Vec::new();
    }

    // Group memories by bucket start. BTreeMap keeps chronological order.
    let mut groups: BTreeMap<DateTime<Utc>, Vec<Memory>> = BTreeMap::new();
    for m in memories {
        let (mem_start, _) = memory_window(m);
        // Clamp by the memory's actual event start, not the bucket boundary,
        // so memories near bucket edges at the window start aren't dropped.
        if mem_start < start || mem_start > end {
            continue;
        }
        let bs = floor_to_bucket(mem_start, granularity);
        groups.entry(bs).or_default().push(m.clone());
    }

    groups
        .into_iter()
        .map(|(bs, mut mems)| {
            // Truncate per-bucket to max_per_bucket; preserve global order (already sorted).
            if mems.len() > max_per_bucket {
                mems.truncate(max_per_bucket);
            }
            let be = next_bucket_start(bs, granularity);
            TimelineBucket {
                start: bs.to_rfc3339(),
                end: be.to_rfc3339(),
                label: bucket_label(bs, granularity),
                count: mems.len(),
                memories: mems.iter().map(memory_to_json).collect(),
            }
        })
        .collect::<Vec<_>>()
}

/// Finalize bucket order: reverse if descending.
fn finalize_bucket_order(
    mut buckets: Vec<TimelineBucket>,
    order_desc: bool,
) -> Vec<TimelineBucket> {
    if order_desc {
        buckets.reverse();
    }
    buckets
}

// ─── Response types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineResponse {
    pub start: String,
    pub end: String,
    pub granularity: TimelineGranularity,
    pub total_count: usize,
    pub bucket_count: usize,
    pub buckets: Vec<TimelineBucket>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineBucket {
    pub start: String,
    pub end: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub count: usize,
    pub memories: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineGraphResponse {
    pub start: String,
    pub end: String,
    pub granularity: TimelineGranularity,
    pub stats: TimelineGraphStats,
    pub nodes: Vec<TimelineNode>,
    pub edges: Vec<TimelineEdge>,
    /// Internal — controls display order in the CLI; not part of the public schema.
    #[serde(skip)]
    pub _order_desc: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineGraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub temporal_edge_count: usize,
    pub semantic_edge_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineNode {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_end: Option<String>,
    pub layer: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bucket: Option<String>,
    pub memory: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEdge {
    pub source: String,
    pub target: String,
    #[serde(rename = "type")]
    pub edge_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_secs: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<usize>,
}

// Keep `LayerInfo` import "used" so we don't break the public re-exports.
const _: fn() = || {
    let _ = std::any::type_name::<LayerInfo>();
};

#[allow(dead_code)]
const _SILENCE_WEEKDAY: Weekday = Weekday::Mon;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LayerInfo, Memory, MemoryMetadata, MemoryState};
    use chrono::TimeZone;

    fn make_memory(id: &str, when: DateTime<Utc>, layer: i32) -> Memory {
        let mut m = Memory::with_content(
            format!("content for {id}"),
            vec![0.0; 8],
            MemoryMetadata::new(),
        );
        m.id = id.to_string();
        m.event_at = Some(when);
        m.metadata.layer = LayerInfo {
            level: layer,
            name: None,
            schema_version: None,
        };
        m.metadata.state = MemoryState::Active;
        m
    }

    #[test]
    fn floor_to_bucket_day() {
        let dt = Utc.with_ymd_and_hms(2026, 6, 2, 14, 35, 22).unwrap();
        let floored = floor_to_bucket(dt, TimelineGranularity::Day);
        assert_eq!(floored, Utc.with_ymd_and_hms(2026, 6, 2, 0, 0, 0).unwrap());
    }

    #[test]
    fn floor_to_bucket_hour() {
        let dt = Utc.with_ymd_and_hms(2026, 6, 2, 14, 35, 22).unwrap();
        let floored = floor_to_bucket(dt, TimelineGranularity::Hour);
        assert_eq!(floored, Utc.with_ymd_and_hms(2026, 6, 2, 14, 0, 0).unwrap());
    }

    #[test]
    fn floor_to_bucket_month() {
        let dt = Utc.with_ymd_and_hms(2026, 6, 2, 14, 35, 22).unwrap();
        let floored = floor_to_bucket(dt, TimelineGranularity::Month);
        assert_eq!(floored, Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap());
    }

    #[test]
    fn floor_to_bucket_week_monday() {
        // 2026-06-03 is a Wednesday; Monday is 2026-06-01.
        let dt = Utc.with_ymd_and_hms(2026, 6, 3, 12, 0, 0).unwrap();
        let floored = floor_to_bucket(dt, TimelineGranularity::Week);
        assert_eq!(floored, Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap());
    }

    #[test]
    fn next_bucket_start_month_year_boundary() {
        let dec = Utc.with_ymd_and_hms(2026, 12, 1, 0, 0, 0).unwrap();
        let jan = next_bucket_start(dec, TimelineGranularity::Month);
        assert_eq!(jan, Utc.with_ymd_and_hms(2027, 1, 1, 0, 0, 0).unwrap());
    }

    #[test]
    fn bucket_label_format() {
        let dt = Utc.with_ymd_and_hms(2026, 6, 2, 14, 0, 0).unwrap();
        assert_eq!(
            bucket_label(dt, TimelineGranularity::Hour).as_deref(),
            Some("2026-06-02T14:00")
        );
        assert_eq!(
            bucket_label(dt, TimelineGranularity::Day).as_deref(),
            Some("2026-06-02")
        );
        assert_eq!(
            bucket_label(dt, TimelineGranularity::Month).as_deref(),
            Some("2026-06")
        );
        // Week of 2026-06-02 (Tue) is ISO week 23.
        assert_eq!(
            bucket_label(dt, TimelineGranularity::Week).as_deref(),
            Some("2026-W23")
        );
    }

    #[test]
    fn bucketize_groups_by_day() {
        let day1_a = Utc.with_ymd_and_hms(2026, 6, 2, 9, 0, 0).unwrap();
        let day1_b = Utc.with_ymd_and_hms(2026, 6, 2, 18, 30, 0).unwrap();
        let day2 = Utc.with_ymd_and_hms(2026, 6, 3, 10, 0, 0).unwrap();
        let mems = vec![
            make_memory("a", day1_a, 0),
            make_memory("b", day1_b, 0),
            make_memory("c", day2, 0),
        ];
        let start = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2026, 6, 30, 0, 0, 0).unwrap();
        let buckets = bucketize(&mems, start, end, TimelineGranularity::Day, 50, false);
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0].count, 2);
        assert_eq!(buckets[1].count, 1);
        assert_eq!(buckets[0].label.as_deref(), Some("2026-06-02"));
        assert_eq!(buckets[1].label.as_deref(), Some("2026-06-03"));
    }

    #[test]
    fn bucketize_respects_max_per_bucket() {
        let day = Utc.with_ymd_and_hms(2026, 6, 2, 0, 0, 0).unwrap();
        let mems: Vec<Memory> = (0..5)
            .map(|i| {
                let when = day + Duration::hours(i);
                make_memory(&format!("m{i}"), when, 0)
            })
            .collect();
        let start = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2026, 6, 30, 0, 0, 0).unwrap();
        let buckets = bucketize(&mems, start, end, TimelineGranularity::Day, 3, false);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].count, 3);
    }

    #[test]
    fn bucketize_drops_memories_outside_window() {
        let inside = Utc.with_ymd_and_hms(2026, 6, 5, 12, 0, 0).unwrap();
        let outside = Utc.with_ymd_and_hms(2027, 1, 1, 12, 0, 0).unwrap();
        let mems = vec![make_memory("in", inside, 0), make_memory("out", outside, 0)];
        let start = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2026, 6, 30, 0, 0, 0).unwrap();
        let buckets = bucketize(&mems, start, end, TimelineGranularity::Day, 50, false);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].count, 1);
    }

    #[test]
    fn memory_window_prefers_event_at() {
        let dt = Utc.with_ymd_and_hms(2026, 6, 2, 0, 0, 0).unwrap();
        let mut m = make_memory("x", dt, 0);
        m.event_at = Some(dt);
        m.event_end = Some(dt + Duration::hours(2));
        let (s, e) = memory_window(&m);
        assert_eq!(s, dt);
        assert_eq!(e, dt + Duration::hours(2));
    }

    #[test]
    fn memory_window_falls_back_to_created_at() {
        let mut m = make_memory("x", Utc::now(), 0);
        m.event_at = None;
        let (s, e) = memory_window(&m);
        assert_eq!(s, m.created_at);
        assert_eq!(e, m.created_at);
    }

    #[test]
    fn reverse_relation_temporal() {
        assert_eq!(reverse_relation("happened_after"), Some("happened_before"));
        assert_eq!(reverse_relation("happened_before"), Some("happened_after"));
        assert_eq!(reverse_relation("happens_within"), Some("happens_within"));
    }

    // helper: pull reverse_relation into scope
    use crate::types::reverse_relation;
}
