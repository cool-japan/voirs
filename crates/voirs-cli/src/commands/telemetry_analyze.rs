//! Telemetry data analysis and visualization
//!
//! Provides commands for analyzing collected telemetry data, generating reports,
//! and visualizing usage patterns.

use crate::telemetry::{EventType, TelemetryEvent, TelemetryStorage};
use anyhow::{Context, Result};
use chrono::Timelike;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Output format for analysis reports
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    /// Plain text format
    Text,
    /// JSON format
    Json,
    /// Markdown format
    Markdown,
}

/// Get the default telemetry storage path
fn get_storage_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("voirs")
        .join("telemetry")
}

/// Analyze telemetry data
pub async fn analyze_telemetry(
    output: Option<PathBuf>,
    format: Option<OutputFormat>,
    time_range: Option<TimeRange>,
    verbose: bool,
) -> Result<()> {
    // Load telemetry storage
    let storage_path = get_storage_path();
    let storage =
        TelemetryStorage::new(&storage_path).context("Failed to initialize telemetry storage")?;

    // Get all events
    let mut all_events = storage.get_all_events().await?;

    // Filter by time range if specified
    if let Some(range) = &time_range {
        all_events.retain(|e| range.contains(e.timestamp));
    }

    if verbose {
        eprintln!("Analyzing {} telemetry events...", all_events.len());
    }

    // Generate analysis
    let analysis = generate_analysis(&all_events);

    // Format output
    let output_format = format.unwrap_or(OutputFormat::Text);
    let formatted = format_analysis(&analysis, output_format)?;

    // Output results
    if let Some(path) = output {
        std::fs::write(&path, formatted)
            .with_context(|| format!("Failed to write output to {}", path.display()))?;
        println!("Analysis written to {}", path.display());
    } else {
        println!("{}", formatted);
    }

    Ok(())
}

/// Generate insights from telemetry data
pub async fn generate_insights(
    min_confidence: f32,
    max_insights: usize,
    verbose: bool,
) -> Result<()> {
    let storage_path = get_storage_path();
    let storage =
        TelemetryStorage::new(&storage_path).context("Failed to initialize telemetry storage")?;

    let all_events = storage.get_all_events().await?;

    if verbose {
        eprintln!("Generating insights from {} events...", all_events.len());
    }

    let insights = find_insights(&all_events, min_confidence);

    // Display insights
    println!("\n📊 Telemetry Insights\n");
    println!("═══════════════════════════════════════════════════════\n");

    for (i, insight) in insights.iter().take(max_insights).enumerate() {
        println!(
            "{}. {} (Confidence: {:.1}%)",
            i + 1,
            insight.message,
            insight.confidence * 100.0
        );
        if !insight.recommendation.is_empty() {
            println!("   💡 {}", insight.recommendation);
        }
        println!();
    }

    if insights.is_empty() {
        println!("No significant insights found. Continue using VoiRS to collect more data.\n");
    }

    Ok(())
}

/// Compare telemetry data between two time periods
pub async fn compare_periods(period1: TimeRange, period2: TimeRange, verbose: bool) -> Result<()> {
    let storage_path = get_storage_path();
    let storage =
        TelemetryStorage::new(&storage_path).context("Failed to initialize telemetry storage")?;

    let all_events = storage.get_all_events().await?;

    let events1: Vec<_> = all_events
        .iter()
        .filter(|e| period1.contains(e.timestamp))
        .collect();
    let events2: Vec<_> = all_events
        .iter()
        .filter(|e| period2.contains(e.timestamp))
        .collect();

    if verbose {
        eprintln!(
            "Comparing {} events (period 1) vs {} events (period 2)",
            events1.len(),
            events2.len()
        );
    }

    let events1_owned: Vec<TelemetryEvent> = events1.iter().map(|e| (*e).clone()).collect();
    let events2_owned: Vec<TelemetryEvent> = events2.iter().map(|e| (*e).clone()).collect();

    let analysis1 = generate_analysis(&events1_owned);
    let analysis2 = generate_analysis(&events2_owned);

    // Display comparison
    println!("\n📊 Telemetry Comparison\n");
    println!("═══════════════════════════════════════════════════════\n");

    println!("Period 1: {} events", events1.len());
    println!("Period 2: {} events", events2.len());
    println!(
        "Change: {:+.1}%\n",
        calculate_change(events1.len(), events2.len())
    );

    // Compare metrics
    compare_metric(
        "Total Commands",
        analysis1.command_count,
        analysis2.command_count,
    );
    compare_metric(
        "Synthesis Requests",
        analysis1.synthesis_count,
        analysis2.synthesis_count,
    );
    compare_metric("Error Rate", analysis1.error_rate, analysis2.error_rate);
    compare_metric(
        "Avg Performance (ms)",
        analysis1.avg_performance_ms,
        analysis2.avg_performance_ms,
    );

    Ok(())
}

/// Time range for filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: chrono::DateTime<chrono::Utc>,
    pub end: chrono::DateTime<chrono::Utc>,
}

impl TimeRange {
    /// Create a new time range
    pub fn new(start: chrono::DateTime<chrono::Utc>, end: chrono::DateTime<chrono::Utc>) -> Self {
        Self { start, end }
    }

    /// Check if a timestamp is within the range
    pub fn contains(&self, timestamp: chrono::DateTime<chrono::Utc>) -> bool {
        timestamp >= self.start && timestamp <= self.end
    }

    /// Create a range for the last N days
    pub fn last_days(days: u32) -> Self {
        let now = chrono::Utc::now();
        let start = now - chrono::Duration::days(days as i64);
        Self { start, end: now }
    }

    /// Create a range for the last N hours
    pub fn last_hours(hours: u32) -> Self {
        let now = chrono::Utc::now();
        let start = now - chrono::Duration::hours(hours as i64);
        Self { start, end: now }
    }
}

/// Analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryAnalysis {
    pub total_events: usize,
    pub command_count: usize,
    pub synthesis_count: usize,
    pub error_count: usize,
    pub error_rate: f32,
    pub avg_performance_ms: f32,
    pub command_distribution: HashMap<String, usize>,
    pub error_distribution: HashMap<String, usize>,
    pub peak_hours: Vec<usize>,
}

/// Generate analysis from events
fn generate_analysis(events: &[TelemetryEvent]) -> TelemetryAnalysis {
    let total_events = events.len();
    let command_count = events
        .iter()
        .filter(|e| matches!(e.event_type, EventType::CommandExecuted))
        .count();
    let synthesis_count = events
        .iter()
        .filter(|e| matches!(e.event_type, EventType::SynthesisRequest))
        .count();
    let error_count = events
        .iter()
        .filter(|e| matches!(e.event_type, EventType::Error))
        .count();

    let error_rate = if total_events > 0 {
        (error_count as f32 / total_events as f32) * 100.0
    } else {
        0.0
    };

    // Calculate average performance
    let performance_events: Vec<_> = events
        .iter()
        .filter_map(|e| {
            if matches!(e.event_type, EventType::Performance) {
                e.metadata
                    .get("duration_ms")
                    .and_then(|v| v.parse::<f32>().ok())
            } else {
                None
            }
        })
        .collect();

    let avg_performance_ms = if !performance_events.is_empty() {
        performance_events.iter().sum::<f32>() / performance_events.len() as f32
    } else {
        0.0
    };

    // Command distribution
    let mut command_distribution = HashMap::new();
    for event in events.iter() {
        if matches!(event.event_type, EventType::CommandExecuted) {
            if let Some(command) = event.metadata.get("command") {
                *command_distribution.entry(command.clone()).or_insert(0) += 1;
            }
        }
    }

    // Error distribution
    let mut error_distribution = HashMap::new();
    for event in events.iter() {
        if matches!(event.event_type, EventType::Error) {
            if let Some(severity) = event.metadata.get("severity") {
                *error_distribution.entry(severity.clone()).or_insert(0) += 1;
            }
        }
    }

    // Peak hours (0-23)
    let mut hour_counts = vec![0usize; 24];
    for event in events.iter() {
        let hour = event.timestamp.hour() as usize;
        hour_counts[hour] += 1;
    }

    let max_count = *hour_counts.iter().max().unwrap_or(&0);
    let peak_hours: Vec<usize> = hour_counts
        .iter()
        .enumerate()
        .filter(|(_, &count)| count > max_count / 2) // Hours with >50% of peak activity
        .map(|(hour, _)| hour)
        .collect();

    TelemetryAnalysis {
        total_events,
        command_count,
        synthesis_count,
        error_count,
        error_rate,
        avg_performance_ms,
        command_distribution,
        error_distribution,
        peak_hours,
    }
}

/// Format analysis results
fn format_analysis(analysis: &TelemetryAnalysis, format: OutputFormat) -> Result<String> {
    match format {
        OutputFormat::Json => Ok(serde_json::to_string_pretty(analysis)?),
        OutputFormat::Text => Ok(format_text_analysis(analysis)),
        _ => Ok(format_text_analysis(analysis)),
    }
}

/// Format as plain text
fn format_text_analysis(analysis: &TelemetryAnalysis) -> String {
    let mut output = String::new();

    output.push_str("\n📊 Telemetry Analysis Report\n");
    output.push_str("═══════════════════════════════════════════════════════\n\n");

    output.push_str(&format!("Total Events:        {}\n", analysis.total_events));
    output.push_str(&format!(
        "Commands Executed:   {}\n",
        analysis.command_count
    ));
    output.push_str(&format!(
        "Synthesis Requests:  {}\n",
        analysis.synthesis_count
    ));
    output.push_str(&format!(
        "Errors:              {} ({:.1}%)\n",
        analysis.error_count, analysis.error_rate
    ));
    output.push_str(&format!(
        "Avg Performance:     {:.1}ms\n\n",
        analysis.avg_performance_ms
    ));

    // Top commands
    if !analysis.command_distribution.is_empty() {
        output.push_str("Top Commands:\n");
        let mut commands: Vec<_> = analysis.command_distribution.iter().collect();
        commands.sort_by(|a, b| b.1.cmp(a.1));
        for (i, (cmd, count)) in commands.iter().take(5).enumerate() {
            output.push_str(&format!("  {}. {} ({})\n", i + 1, cmd, count));
        }
        output.push('\n');
    }

    // Peak hours
    if !analysis.peak_hours.is_empty() {
        output.push_str("Peak Activity Hours: ");
        output.push_str(
            &analysis
                .peak_hours
                .iter()
                .map(|h| format!("{}:00", h))
                .collect::<Vec<_>>()
                .join(", "),
        );
        output.push_str("\n\n");
    }

    output
}

/// Insight from telemetry data
#[derive(Debug, Clone)]
pub struct Insight {
    pub message: String,
    pub recommendation: String,
    pub confidence: f32,
}

/// Find insights from telemetry data
fn find_insights(events: &[TelemetryEvent], min_confidence: f32) -> Vec<Insight> {
    let mut insights = Vec::new();
    let analysis = generate_analysis(events);

    // High error rate insight
    if analysis.error_rate > 10.0 {
        insights.push(Insight {
            message: format!("High error rate detected ({:.1}%)", analysis.error_rate),
            recommendation: "Review error logs and consider filing a bug report".to_string(),
            confidence: (analysis.error_rate / 100.0).min(1.0),
        });
    }

    // Slow performance insight
    if analysis.avg_performance_ms > 5000.0 {
        insights.push(Insight {
            message: format!(
                "Slow average performance ({:.1}ms)",
                analysis.avg_performance_ms
            ),
            recommendation: "Consider optimizing models or using GPU acceleration".to_string(),
            confidence: (analysis.avg_performance_ms / 10000.0).min(1.0),
        });
    }

    // Underutilized features
    if analysis.synthesis_count > 100 && analysis.command_count < 10 {
        insights.push(Insight {
            message: "You're mainly using synthesis - explore other VoiRS features".to_string(),
            recommendation: "Try batch processing, voice cloning, or interactive mode".to_string(),
            confidence: 0.8,
        });
    }

    // Filter by confidence
    insights.retain(|i| i.confidence >= min_confidence);

    // Sort by confidence
    insights.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    insights
}

/// Calculate percentage change
fn calculate_change(old_value: usize, new_value: usize) -> f32 {
    if old_value == 0 {
        if new_value > 0 {
            100.0
        } else {
            0.0
        }
    } else {
        ((new_value as f32 - old_value as f32) / old_value as f32) * 100.0
    }
}

/// Compare a metric between two periods
fn compare_metric(name: &str, value1: impl std::fmt::Display, value2: impl std::fmt::Display) {
    println!(
        "{:20} {:>12} -> {:>12}",
        name,
        format!("{}", value1),
        format!("{}", value2)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_range_contains() {
        let start = chrono::Utc::now() - chrono::Duration::days(1);
        let end = chrono::Utc::now();
        let middle = chrono::Utc::now() - chrono::Duration::hours(12);

        let range = TimeRange::new(start, end);
        assert!(range.contains(middle));
        assert!(!range.contains(start - chrono::Duration::hours(1)));
        assert!(!range.contains(end + chrono::Duration::hours(1)));
    }

    #[test]
    fn test_time_range_last_days() {
        let now = chrono::Utc::now();
        let range = TimeRange::last_days(7);

        // Test that the range includes times from the past 7 days
        assert!(range.contains(now - chrono::Duration::hours(1)));
        assert!(range.contains(now - chrono::Duration::days(1)));
        assert!(range.contains(now - chrono::Duration::days(6)));

        // Test that it excludes times from more than 7 days ago
        assert!(!range.contains(now - chrono::Duration::days(8)));
    }

    #[test]
    fn test_generate_analysis_empty() {
        let events = vec![];
        let analysis = generate_analysis(&events);
        assert_eq!(analysis.total_events, 0);
        assert_eq!(analysis.error_rate, 0.0);
    }

    #[test]
    fn test_calculate_change() {
        assert_eq!(calculate_change(100, 150), 50.0);
        assert_eq!(calculate_change(100, 50), -50.0);
        assert_eq!(calculate_change(0, 100), 100.0);
    }

    #[test]
    fn test_find_insights_high_error_rate() {
        // Create mock events with high error rate
        let mut events = Vec::new();
        let now = chrono::Utc::now();

        for i in 0..20 {
            let mut metadata = crate::telemetry::EventMetadata::new();
            metadata.set("message", "test error");
            metadata.set("severity", "high");

            events.push(TelemetryEvent {
                id: format!("event_{}", i),
                event_type: EventType::Error,
                timestamp: now,
                metadata,
                user_id: Some("test_user".to_string()),
                session_id: "test_session".to_string(),
            });
        }

        for i in 20..100 {
            let mut metadata = crate::telemetry::EventMetadata::new();
            metadata.set("command", "test");

            events.push(TelemetryEvent {
                id: format!("event_{}", i),
                event_type: EventType::CommandExecuted,
                timestamp: now,
                metadata,
                user_id: Some("test_user".to_string()),
                session_id: "test_session".to_string(),
            });
        }

        let insights = find_insights(&events, 0.1);
        assert!(!insights.is_empty());
    }

    #[test]
    fn test_format_text_analysis() {
        let analysis = TelemetryAnalysis {
            total_events: 100,
            command_count: 80,
            synthesis_count: 50,
            error_count: 5,
            error_rate: 5.0,
            avg_performance_ms: 1500.0,
            command_distribution: HashMap::new(),
            error_distribution: HashMap::new(),
            peak_hours: vec![9, 14, 20],
        };

        let text = format_text_analysis(&analysis);
        assert!(text.contains("Total Events:"));
        assert!(text.contains("100"));
        assert!(text.contains("5.0%"));
    }
}
