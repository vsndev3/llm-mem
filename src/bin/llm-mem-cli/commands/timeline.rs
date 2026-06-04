use crate::OutputFormat;
use llm_mem::System;
use llm_mem::MemoryOperations;
use llm_mem::operations::{GetTimelineGraphRequest, GetTimelineRequest, TimelineGranularity};

use chrono::Duration;

/// Parse a relative "since" expression (e.g. "2d", "12h", "30m", "1w") into
/// `(start_iso, end_iso)` where end is now and start is end - duration.
pub(crate) fn parse_since_to_window(since: &str) -> Result<(String, String), String> {
    let s = since.trim();
    if s.is_empty() {
        return Err("since string is empty".into());
    }
    let (num_str, unit) = s.split_at(s.len() - 1);
    let num: i64 = num_str
        .parse()
        .map_err(|_| format!("invalid since value '{s}' (expected e.g. '2d', '12h', '30m', '1w')"))?;
    let unit = unit.to_ascii_lowercase();
    let duration = match unit.as_str() {
        "s" => Duration::seconds(num),
        "m" => Duration::minutes(num),
        "h" => Duration::hours(num),
        "d" => Duration::days(num),
        "w" => Duration::weeks(num),
        other => return Err(format!("unknown unit '{other}' (use s/m/h/d/w)")),
    };
    let end = chrono::Utc::now();
    let start = end - duration;
    Ok((start.to_rfc3339(), end.to_rfc3339()))
}

pub(crate) fn parse_granularity(s: &str) -> Result<TimelineGranularity, String> {
    match s.to_ascii_lowercase().as_str() {
        "hour" => Ok(TimelineGranularity::Hour),
        "day" => Ok(TimelineGranularity::Day),
        "week" => Ok(TimelineGranularity::Week),
        "month" => Ok(TimelineGranularity::Month),
        "none" => Ok(TimelineGranularity::None),
        other => Err(format!("unknown granularity '{other}' (use hour/day/week/month/none)")),
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn handle_timeline(
    system: &System,
    bank: &str,
    since: Option<&str>,
    start: Option<&str>,
    end: Option<&str>,
    granularity: Option<&str>,
    include_derived: bool,
    max_per_bucket: usize,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut resolved_start = start.map(|s| s.to_string());
    let mut resolved_end = end.map(|s| s.to_string());
    if let Some(s) = since {
        let (s, e) = parse_since_to_window(s)
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        resolved_start = Some(s);
        resolved_end = Some(e);
    }

    let req = GetTimelineRequest {
        start: resolved_start,
        end: resolved_end,
        granularity: granularity.map(parse_granularity).transpose()?,
        bank: Some(bank.to_string()),
        user_id: None,
        agent_id: None,
        topics: None,
        max_results_per_bucket: max_per_bucket,
        include_derived,
        order: "asc".to_string(),
    };

    let manager = system.bank_manager.resolve_bank(Some(bank)).await
        .map_err(|e| format!("Failed to resolve bank: {}", e))?;
    let ops = MemoryOperations::new(manager, None, None, 1000);
    match ops.get_timeline(req).await {
        Ok(response) => crate::output::print_response(&response, format)?,
        Err(e) => eprintln!("Error: {}", e),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn handle_timeline_graph(
    system: &System,
    bank: &str,
    since: Option<&str>,
    start: Option<&str>,
    end: Option<&str>,
    granularity: Option<&str>,
    include_derived: bool,
    max_per_bucket: usize,
    max_depth: usize,
    temporal_window_secs: i64,
    include_semantic_edges: bool,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut resolved_start = start.map(|s| s.to_string());
    let mut resolved_end = end.map(|s| s.to_string());
    if let Some(s) = since {
        let (s, e) = parse_since_to_window(s)
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        resolved_start = Some(s);
        resolved_end = Some(e);
    }

    let timeline = GetTimelineRequest {
        start: resolved_start,
        end: resolved_end,
        granularity: granularity.map(parse_granularity).transpose()?,
        bank: Some(bank.to_string()),
        user_id: None,
        agent_id: None,
        topics: None,
        max_results_per_bucket: max_per_bucket,
        include_derived,
        order: "asc".to_string(),
    };

    let req = GetTimelineGraphRequest {
        timeline,
        max_depth,
        relation_types: None,
        temporal_edge_window_secs: temporal_window_secs,
        include_simultaneous: false,
        simultaneous_window_secs: 60,
        include_semantic_edges,
    };

    let manager = system.bank_manager.resolve_bank(Some(bank)).await
        .map_err(|e| format!("Failed to resolve bank: {}", e))?;
    let ops = MemoryOperations::new(manager, None, None, 1000);
    match ops.get_timeline_graph(req).await {
        Ok(response) => crate::output::print_response(&response, format)?,
        Err(e) => eprintln!("Error: {}", e),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_since_days() {
        let (s, e) = parse_since_to_window("2d").unwrap();
        let start: chrono::DateTime<chrono::Utc> = s.parse().unwrap();
        let end: chrono::DateTime<chrono::Utc> = e.parse().unwrap();
        let diff = end - start;
        // Allow 1 second of slop
        assert!(diff.num_seconds() >= 2 * 86400 - 1);
        assert!(diff.num_seconds() <= 2 * 86400 + 1);
    }

    #[test]
    fn parse_since_hours() {
        let (s, e) = parse_since_to_window("12h").unwrap();
        let start: chrono::DateTime<chrono::Utc> = s.parse().unwrap();
        let end: chrono::DateTime<chrono::Utc> = e.parse().unwrap();
        let diff = end - start;
        assert!(diff.num_seconds() >= 12 * 3600 - 1);
        assert!(diff.num_seconds() <= 12 * 3600 + 1);
    }

    #[test]
    fn parse_since_rejects_bad_unit() {
        assert!(parse_since_to_window("5x").is_err());
    }

    #[test]
    fn parse_granularity_all() {
        assert!(matches!(parse_granularity("day").unwrap(), TimelineGranularity::Day));
        assert!(matches!(parse_granularity("week").unwrap(), TimelineGranularity::Week));
        assert!(matches!(parse_granularity("month").unwrap(), TimelineGranularity::Month));
        assert!(matches!(parse_granularity("hour").unwrap(), TimelineGranularity::Hour));
        assert!(matches!(parse_granularity("none").unwrap(), TimelineGranularity::None));
    }

    #[test]
    fn parse_granularity_unknown() {
        assert!(parse_granularity("year").is_err());
    }
}
