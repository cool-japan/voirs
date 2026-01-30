//! Performance report generation and formatting.

use super::{Bottleneck, ComparisonResult, ProfileSession, ProfilerConfig};
use crate::VoirsError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Format for performance reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    /// Human-readable text format
    Text,
    /// JSON format
    Json,
    /// Markdown format
    Markdown,
    /// HTML format
    Html,
}

/// A comprehensive performance report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Report title
    pub title: String,

    /// Session information
    pub session: ReportSession,

    /// Stage-by-stage breakdown
    pub stage_breakdown: Vec<StageBreakdown>,

    /// Memory analysis
    pub memory_analysis: Option<MemoryAnalysis>,

    /// Detected bottlenecks
    pub bottlenecks: Vec<Bottleneck>,

    /// Comparison with baseline (if available)
    pub comparison: Option<ComparisonResult>,

    /// Overall performance summary
    pub summary: PerformanceSummary,

    /// Recommendations
    pub recommendations: Vec<String>,
}

impl PerformanceReport {
    /// Generate human-readable summary.
    pub fn summary(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!("=== {} ===\n\n", self.title));
        output.push_str(&format!("Session: {}\n", self.session.name));
        output.push_str(&format!(
            "Duration: {:.2}s\n\n",
            self.session.duration_seconds
        ));

        output.push_str("Stage Breakdown:\n");
        for stage in &self.stage_breakdown {
            output.push_str(&format!(
                "  - {}: {:.2}ms ({:.1}%)\n",
                stage.stage_name, stage.avg_duration_ms, stage.percentage_of_total
            ));
        }

        if let Some(memory) = &self.memory_analysis {
            output.push_str("\nMemory Usage:\n");
            output.push_str(&format!("  - Peak: {:.2} MB\n", memory.peak_mb));
            output.push_str(&format!("  - Average: {:.2} MB\n", memory.average_mb));
        }

        if !self.bottlenecks.is_empty() {
            output.push_str(&format!("\nBottlenecks ({}):\n", self.bottlenecks.len()));
            for bottleneck in &self.bottlenecks {
                output.push_str(&format!("  - {}\n", bottleneck.summary()));
            }
        }

        if !self.recommendations.is_empty() {
            output.push_str(&format!(
                "\nRecommendations ({}):\n",
                self.recommendations.len()
            ));
            for (i, rec) in self.recommendations.iter().enumerate() {
                output.push_str(&format!("  {}. {}\n", i + 1, rec));
            }
        }

        output
    }

    /// Get list of detected bottlenecks.
    pub fn bottlenecks(&self) -> &[Bottleneck] {
        &self.bottlenecks
    }

    /// Get performance summary.
    pub fn performance_summary(&self) -> &PerformanceSummary {
        &self.summary
    }
}

/// Session information in the report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSession {
    pub id: String,
    pub name: String,
    pub duration_seconds: f64,
    pub timestamp: String,
}

/// Stage-by-stage performance breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageBreakdown {
    pub stage_name: String,
    pub execution_count: usize,
    pub total_duration_ms: f64,
    pub avg_duration_ms: f64,
    pub min_duration_ms: f64,
    pub max_duration_ms: f64,
    pub percentage_of_total: f64,
}

/// Memory usage analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysis {
    pub peak_mb: f64,
    pub average_mb: f64,
    pub total_allocated_mb: f64,
    pub growth_percent: f64,
}

/// Overall performance summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_duration_ms: f64,
    pub throughput: f64,
    pub efficiency_score: f64,
    pub bottleneck_count: usize,
    pub regression_detected: bool,
}

/// Report generator for creating performance reports.
pub struct ReportGenerator {
    config: ProfilerConfig,
}

impl ReportGenerator {
    /// Create a new report generator.
    pub fn new(config: ProfilerConfig) -> Self {
        Self { config }
    }

    /// Generate a performance report from a session.
    pub async fn generate(
        &self,
        session: &ProfileSession,
        comparison: Option<ComparisonResult>,
    ) -> Result<PerformanceReport, VoirsError> {
        let report_session = ReportSession {
            id: session.id.clone(),
            name: session.name.clone(),
            duration_seconds: session.duration().map(|d| d.as_secs_f64()).unwrap_or(0.0),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let stage_breakdown = self.generate_stage_breakdown(session);
        let memory_analysis = self.generate_memory_analysis(session);
        let summary = self.generate_summary(session, &comparison);
        let recommendations = self.generate_recommendations(session);

        Ok(PerformanceReport {
            title: format!("Performance Report: {}", session.name),
            session: report_session,
            stage_breakdown,
            memory_analysis,
            bottlenecks: session.bottlenecks.clone(),
            comparison,
            summary,
            recommendations,
        })
    }

    /// Generate aggregate report from multiple sessions.
    pub async fn generate_aggregate(
        &self,
        sessions: &[ProfileSession],
    ) -> Result<PerformanceReport, VoirsError> {
        if sessions.is_empty() {
            return Err(VoirsError::config_error(
                "No sessions provided for aggregate report",
            ));
        }

        // Create aggregated session data
        let aggregated_session = self.aggregate_sessions(sessions)?;

        // Generate report from aggregated data
        self.generate(&aggregated_session, None).await
    }

    /// Aggregate multiple sessions into a single session with averaged metrics.
    fn aggregate_sessions(
        &self,
        sessions: &[ProfileSession],
    ) -> Result<ProfileSession, VoirsError> {
        use std::collections::HashMap;

        let session_count = sessions.len();

        // Aggregate metadata
        let aggregated_name = format!("Aggregate Report ({} sessions)", session_count);
        let aggregated_id = format!("aggregate-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"));

        // Calculate time range
        let earliest_start = sessions
            .iter()
            .map(|s| s.start_time)
            .min()
            .unwrap_or_else(std::time::Instant::now);

        let latest_end = sessions
            .iter()
            .filter_map(|s| s.end_time)
            .max()
            .unwrap_or_else(std::time::Instant::now);

        // Aggregate stage metrics across all sessions
        let mut aggregated_stage_metrics = HashMap::new();

        for session in sessions {
            for (stage_name, metrics) in &session.stage_metrics {
                let entry = aggregated_stage_metrics
                    .entry(stage_name.clone())
                    .or_insert_with(Vec::new);
                entry.push(metrics.clone());
            }
        }

        // Calculate averaged stage metrics
        use crate::profiling::{PipelineStage, StageMetrics};
        use std::time::Duration;

        let averaged_stage_metrics: HashMap<String, StageMetrics> = aggregated_stage_metrics
            .into_iter()
            .map(|(stage_name, metrics_list)| {
                let count = metrics_list.len();

                // Get the stage enum from the first metrics entry
                let stage = metrics_list
                    .first()
                    .map(|m| m.stage)
                    .unwrap_or(PipelineStage::FullPipeline);

                let avg_execution_count = metrics_list
                    .iter()
                    .map(|m| m.execution_count)
                    .sum::<usize>()
                    / count.max(1);

                let avg_total_duration = Duration::from_secs_f64(
                    metrics_list
                        .iter()
                        .map(|m| m.total_duration.as_secs_f64())
                        .sum::<f64>()
                        / count as f64,
                );

                let avg_avg_duration = Duration::from_secs_f64(
                    metrics_list
                        .iter()
                        .map(|m| m.avg_duration.as_secs_f64())
                        .sum::<f64>()
                        / count as f64,
                );

                let min_duration = metrics_list
                    .iter()
                    .map(|m| m.min_duration)
                    .min()
                    .unwrap_or(Duration::from_secs(0));

                let max_duration = metrics_list
                    .iter()
                    .map(|m| m.max_duration)
                    .max()
                    .unwrap_or(Duration::from_secs(0));

                let avg_percentage = metrics_list
                    .iter()
                    .map(|m| m.percentage_of_total)
                    .sum::<f64>()
                    / count as f64;

                let avg_input_size =
                    metrics_list.iter().map(|m| m.avg_input_size).sum::<f64>() / count as f64;

                let avg_output_size =
                    metrics_list.iter().map(|m| m.avg_output_size).sum::<f64>() / count as f64;

                let avg_throughput =
                    metrics_list.iter().map(|m| m.throughput).sum::<f64>() / count as f64;

                let averaged_metrics = StageMetrics {
                    stage,
                    execution_count: avg_execution_count,
                    total_duration: avg_total_duration,
                    avg_duration: avg_avg_duration,
                    min_duration,
                    max_duration,
                    std_deviation: Duration::from_secs(0), // Placeholder for aggregated
                    percentage_of_total: avg_percentage,
                    avg_input_size,
                    avg_output_size,
                    throughput: avg_throughput,
                };

                (stage_name, averaged_metrics)
            })
            .collect();

        // Aggregate memory snapshots
        let mut all_memory_snapshots = Vec::new();
        for session in sessions {
            all_memory_snapshots.extend(session.memory_snapshots.clone());
        }

        // Aggregate bottlenecks (deduplicate by component)
        let mut bottleneck_map: HashMap<String, super::Bottleneck> = HashMap::new();
        for session in sessions {
            for bottleneck in &session.bottlenecks {
                bottleneck_map
                    .entry(bottleneck.component.clone())
                    .or_insert_with(|| bottleneck.clone());
            }
        }
        let aggregated_bottlenecks: Vec<_> = bottleneck_map.into_values().collect();

        // Calculate total duration
        let total_duration = if latest_end > earliest_start {
            Some(latest_end.duration_since(earliest_start))
        } else {
            None
        };

        // Create aggregated session
        Ok(ProfileSession {
            id: aggregated_id,
            name: aggregated_name,
            start_time: earliest_start,
            end_time: Some(latest_end),
            duration: total_duration,
            stage_metrics: averaged_stage_metrics,
            memory_snapshots: all_memory_snapshots,
            bottlenecks: aggregated_bottlenecks,
            metadata: HashMap::new(),
        })
    }

    /// Save report to file.
    pub async fn save_report(
        &self,
        report: &PerformanceReport,
        path: &Path,
    ) -> Result<(), VoirsError> {
        let json = serde_json::to_string_pretty(report)
            .map_err(|e| VoirsError::config_error(format!("Failed to serialize report: {}", e)))?;

        tokio::fs::write(path, json)
            .await
            .map_err(|e| VoirsError::config_error(format!("Failed to write report: {}", e)))?;

        Ok(())
    }

    fn generate_stage_breakdown(&self, session: &ProfileSession) -> Vec<StageBreakdown> {
        session
            .stage_metrics
            .iter()
            .map(|(name, metrics)| StageBreakdown {
                stage_name: name.clone(),
                execution_count: metrics.execution_count,
                total_duration_ms: metrics.total_duration.as_secs_f64() * 1000.0,
                avg_duration_ms: metrics.avg_duration.as_secs_f64() * 1000.0,
                min_duration_ms: metrics.min_duration.as_secs_f64() * 1000.0,
                max_duration_ms: metrics.max_duration.as_secs_f64() * 1000.0,
                percentage_of_total: metrics.percentage_of_total,
            })
            .collect()
    }

    fn generate_memory_analysis(&self, session: &ProfileSession) -> Option<MemoryAnalysis> {
        if session.memory_snapshots.is_empty() {
            return None;
        }

        let peak_mb = session
            .memory_snapshots
            .iter()
            .map(|s| s.allocated_mb())
            .fold(0.0f64, f64::max);

        let average_mb = session
            .memory_snapshots
            .iter()
            .map(|s| s.allocated_mb())
            .sum::<f64>()
            / session.memory_snapshots.len() as f64;

        let first_mb = session.memory_snapshots.first()?.allocated_mb();
        let growth_percent = if first_mb > 0.0 {
            ((peak_mb - first_mb) / first_mb) * 100.0
        } else {
            0.0
        };

        Some(MemoryAnalysis {
            peak_mb,
            average_mb,
            total_allocated_mb: peak_mb,
            growth_percent,
        })
    }

    fn generate_summary(
        &self,
        session: &ProfileSession,
        comparison: &Option<ComparisonResult>,
    ) -> PerformanceSummary {
        let total_duration_ms = session
            .duration()
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        let efficiency_score = if !session.bottlenecks.is_empty() {
            100.0 - (session.bottlenecks.len() as f64 * 10.0).min(50.0)
        } else {
            100.0
        };

        let regression_detected = comparison.as_ref().is_some_and(|c| c.has_regression);

        // Calculate throughput as operations per second
        // Sum up execution counts across all stages and divide by total duration
        let total_operations: usize = session
            .stage_metrics
            .values()
            .map(|m| m.execution_count)
            .sum();

        let throughput = if let Some(duration) = session.duration() {
            let duration_secs = duration.as_secs_f64();
            if duration_secs > 0.0 {
                total_operations as f64 / duration_secs
            } else {
                0.0
            }
        } else {
            0.0
        };

        PerformanceSummary {
            total_duration_ms,
            throughput,
            efficiency_score,
            bottleneck_count: session.bottlenecks.len(),
            regression_detected,
        }
    }

    fn generate_recommendations(&self, session: &ProfileSession) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Add bottleneck-specific recommendations
        for bottleneck in &session.bottlenecks {
            if !bottleneck.recommendation.is_empty() {
                recommendations.push(bottleneck.recommendation.clone());
            }
        }

        // Add general recommendations
        if session.bottlenecks.len() > 2 {
            recommendations.push(
                "Multiple bottlenecks detected. Consider profiling individual stages separately."
                    .to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations.push("Performance looks good! No major issues detected.".to_string());
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_format() {
        assert_eq!(ReportFormat::Json, ReportFormat::Json);
        assert_ne!(ReportFormat::Json, ReportFormat::Text);
    }

    #[tokio::test]
    async fn test_report_generator_creation() {
        let config = ProfilerConfig::default();
        let _generator = ReportGenerator::new(config);
        // Just check it compiles and creates
    }
}
