//! Production Monitoring Utilities
//!
//! Provides helper functions and utilities for monitoring vocoder
//! performance in production environments.

use super::{AdvancedProfiler, PerformanceReport, ProfilerMetrics};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Production monitoring configuration
#[derive(Debug, Clone)]
pub struct ProductionMonitoringConfig {
    /// Enable automatic performance logging
    pub enable_logging: bool,
    /// Log file path (if logging enabled)
    pub log_path: Option<String>,
    /// Minimum operations before generating reports
    pub min_operations_for_report: u64,
    /// Alert threshold for RTF (operations/s)
    pub rtf_alert_threshold: f32,
    /// Alert threshold for latency increase (%)
    pub latency_alert_threshold: f32,
}

impl Default for ProductionMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_logging: true,
            log_path: Some("vocoder_performance.log".to_string()),
            min_operations_for_report: 10,
            rtf_alert_threshold: 0.9, // Alert if RTF > 0.9 (getting close to real-time limit)
            latency_alert_threshold: 20.0, // Alert if latency increases by 20%
        }
    }
}

/// Production performance monitor
pub struct ProductionMonitor {
    profiler: AdvancedProfiler,
    config: ProductionMonitoringConfig,
    last_report_operations: u64,
}

impl ProductionMonitor {
    /// Create a new production monitor
    pub fn new() -> Self {
        Self {
            profiler: AdvancedProfiler::new(),
            config: ProductionMonitoringConfig::default(),
            last_report_operations: 0,
        }
    }

    /// Create a new production monitor with custom configuration
    pub fn with_config(config: ProductionMonitoringConfig) -> Self {
        Self {
            profiler: AdvancedProfiler::new(),
            config,
            last_report_operations: 0,
        }
    }

    /// Get the underlying profiler
    pub fn profiler(&self) -> &AdvancedProfiler {
        &self.profiler
    }

    /// Check if alerts should be raised based on current metrics
    pub fn check_alerts(&self) -> Vec<Alert> {
        let metrics = self.profiler.get_metrics();
        let mut alerts = Vec::new();

        // Check RTF threshold
        if metrics.rtf_stats.avg_rtf > self.config.rtf_alert_threshold {
            alerts.push(Alert {
                severity: AlertSeverity::Warning,
                category: AlertCategory::Performance,
                message: format!(
                    "RTF ({:.3}x) exceeds threshold ({:.3}x)",
                    metrics.rtf_stats.avg_rtf, self.config.rtf_alert_threshold
                ),
                metric_value: Some(metrics.rtf_stats.avg_rtf),
            });
        }

        // Check for high RTF variability
        let rtf_range = metrics.rtf_stats.max_rtf - metrics.rtf_stats.min_rtf;
        if rtf_range > 0.3 {
            alerts.push(Alert {
                severity: AlertSeverity::Info,
                category: AlertCategory::Stability,
                message: format!("High RTF variability detected (range: {:.3}x)", rtf_range),
                metric_value: Some(rtf_range),
            });
        }

        // Check for low real-time percentage
        if metrics.rtf_stats.realtime_percentage < 95.0 {
            alerts.push(Alert {
                severity: AlertSeverity::Warning,
                category: AlertCategory::Performance,
                message: format!(
                    "Only {:.1}% of operations meet real-time constraints",
                    metrics.rtf_stats.realtime_percentage
                ),
                metric_value: Some(metrics.rtf_stats.realtime_percentage),
            });
        }

        alerts
    }

    /// Generate and optionally log a performance report
    pub fn generate_and_log_report(&mut self) -> crate::Result<PerformanceReport> {
        let metrics = self.profiler.get_metrics();

        // Check if enough operations have been performed
        if metrics.total_operations < self.config.min_operations_for_report {
            return Err(crate::VocoderError::Other(format!(
                "Not enough operations for report (have: {}, need: {})",
                metrics.total_operations, self.config.min_operations_for_report
            )));
        }

        let report = self.profiler.generate_report();

        // Log if enabled
        if self.config.enable_logging {
            if let Some(ref log_path) = self.config.log_path {
                self.log_report(&report, log_path)?;
            }
        }

        self.last_report_operations = metrics.total_operations;

        Ok(report)
    }

    /// Log a performance report to file
    fn log_report(&self, report: &PerformanceReport, path: &str) -> crate::Result<()> {
        let mut file = File::options()
            .create(true)
            .append(true)
            .open(path)
            .map_err(crate::VocoderError::IoError)?;

        let timestamp = chrono::Utc::now();
        writeln!(file, "\n=== Performance Report - {} ===", timestamp)
            .map_err(crate::VocoderError::IoError)?;
        writeln!(file, "{}", report).map_err(crate::VocoderError::IoError)?;

        Ok(())
    }

    /// Export metrics as JSON
    pub fn export_metrics_json(&self, path: &Path) -> crate::Result<()> {
        let metrics = self.profiler.get_metrics();
        let json = serde_json::to_string_pretty(&MetricsExport::from_metrics(&metrics))
            .map_err(|e| crate::VocoderError::Other(format!("JSON serialization failed: {}", e)))?;

        std::fs::write(path, json).map_err(crate::VocoderError::IoError)?;

        Ok(())
    }

    /// Get a summary of current performance status
    pub fn get_status_summary(&self) -> StatusSummary {
        let metrics = self.profiler.get_metrics();

        let health = if metrics.rtf_stats.avg_rtf < 0.5 {
            HealthStatus::Excellent
        } else if metrics.rtf_stats.avg_rtf < 0.8 {
            HealthStatus::Good
        } else if metrics.rtf_stats.avg_rtf < 1.0 {
            HealthStatus::Warning
        } else {
            HealthStatus::Critical
        };

        StatusSummary {
            total_operations: metrics.total_operations,
            avg_rtf: metrics.rtf_stats.avg_rtf,
            avg_latency_ms: metrics.avg_latency.as_secs_f32() * 1000.0,
            realtime_percentage: metrics.rtf_stats.realtime_percentage,
            health,
        }
    }
}

impl Default for ProductionMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning - attention needed
    Warning,
    /// Critical - immediate action required
    Critical,
}

/// Alert category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertCategory {
    /// Performance-related alert
    Performance,
    /// Stability-related alert
    Stability,
    /// Resource usage alert
    Resource,
    /// Quality-related alert
    Quality,
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert category
    pub category: AlertCategory,
    /// Alert message
    pub message: String,
    /// Associated metric value
    pub metric_value: Option<f32>,
}

impl std::fmt::Display for Alert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let severity_icon = match self.severity {
            AlertSeverity::Info => "ℹ️ ",
            AlertSeverity::Warning => "⚠️ ",
            AlertSeverity::Critical => "🔴",
        };

        write!(
            f,
            "{} [{:?}/{:?}] {}",
            severity_icon, self.severity, self.category, self.message
        )
    }
}

/// Health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// System performing excellently
    Excellent,
    /// System performing well
    Good,
    /// System needs attention
    Warning,
    /// System in critical state
    Critical,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Excellent => write!(f, "⭐ Excellent"),
            Self::Good => write!(f, "✅ Good"),
            Self::Warning => write!(f, "⚠️  Warning"),
            Self::Critical => write!(f, "🔴 Critical"),
        }
    }
}

/// Status summary
#[derive(Debug, Clone)]
pub struct StatusSummary {
    /// Total operations processed
    pub total_operations: u64,
    /// Average RTF
    pub avg_rtf: f32,
    /// Average latency in milliseconds
    pub avg_latency_ms: f32,
    /// Percentage of operations meeting real-time constraints
    pub realtime_percentage: f32,
    /// Overall health status
    pub health: HealthStatus,
}

impl std::fmt::Display for StatusSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Vocoder Performance Status")?;
        writeln!(f, "==========================")?;
        writeln!(f, "Health:       {}", self.health)?;
        writeln!(f, "Operations:   {}", self.total_operations)?;
        writeln!(f, "Avg RTF:      {:.3}x", self.avg_rtf)?;
        writeln!(f, "Avg Latency:  {:.2} ms", self.avg_latency_ms)?;
        writeln!(f, "Real-time %:  {:.1}%", self.realtime_percentage)
    }
}

/// Metrics export structure (for JSON serialization)
#[derive(Debug, Clone, serde::Serialize)]
struct MetricsExport {
    total_operations: u64,
    avg_latency_ms: f32,
    min_latency_ms: f32,
    max_latency_ms: f32,
    p50_latency_ms: f32,
    p95_latency_ms: f32,
    p99_latency_ms: f32,
    avg_rtf: f32,
    min_rtf: f32,
    max_rtf: f32,
    realtime_percentage: f32,
    total_frames: u64,
    avg_frames_per_second: f32,
    preprocessing_ms: f32,
    inference_ms: f32,
    postprocessing_ms: f32,
}

impl MetricsExport {
    fn from_metrics(metrics: &ProfilerMetrics) -> Self {
        Self {
            total_operations: metrics.total_operations,
            avg_latency_ms: metrics.avg_latency.as_secs_f32() * 1000.0,
            min_latency_ms: metrics.min_latency.unwrap_or_default().as_secs_f32() * 1000.0,
            max_latency_ms: metrics.max_latency.unwrap_or_default().as_secs_f32() * 1000.0,
            p50_latency_ms: metrics.p50_latency.as_secs_f32() * 1000.0,
            p95_latency_ms: metrics.p95_latency.as_secs_f32() * 1000.0,
            p99_latency_ms: metrics.p99_latency.as_secs_f32() * 1000.0,
            avg_rtf: metrics.rtf_stats.avg_rtf,
            min_rtf: metrics.rtf_stats.min_rtf,
            max_rtf: metrics.rtf_stats.max_rtf,
            realtime_percentage: metrics.rtf_stats.realtime_percentage,
            total_frames: metrics.throughput_stats.total_frames,
            avg_frames_per_second: metrics.throughput_stats.avg_frames_per_second,
            preprocessing_ms: metrics.latency_breakdown.preprocessing.as_secs_f32() * 1000.0,
            inference_ms: metrics.latency_breakdown.inference.as_secs_f32() * 1000.0,
            postprocessing_ms: metrics.latency_breakdown.postprocessing.as_secs_f32() * 1000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_monitor_creation() {
        let monitor = ProductionMonitor::new();
        assert_eq!(monitor.profiler().get_metrics().total_operations, 0);
    }

    #[test]
    fn test_status_summary() {
        let monitor = ProductionMonitor::new();
        let summary = monitor.get_status_summary();
        assert_eq!(summary.total_operations, 0);
        assert_eq!(summary.health, HealthStatus::Excellent);
    }

    #[test]
    fn test_health_status_display() {
        assert_eq!(format!("{}", HealthStatus::Excellent), "⭐ Excellent");
        assert_eq!(format!("{}", HealthStatus::Good), "✅ Good");
        assert_eq!(format!("{}", HealthStatus::Warning), "⚠️  Warning");
        assert_eq!(format!("{}", HealthStatus::Critical), "🔴 Critical");
    }

    #[test]
    fn test_alert_creation() {
        let alert = Alert {
            severity: AlertSeverity::Warning,
            category: AlertCategory::Performance,
            message: "Test alert".to_string(),
            metric_value: Some(0.95),
        };

        let display = format!("{}", alert);
        assert!(display.contains("Warning"));
        assert!(display.contains("Performance"));
        assert!(display.contains("Test alert"));
    }
}
