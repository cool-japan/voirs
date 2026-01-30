//! Integration between Performance Profiling and Production Monitoring
//!
//! This module provides seamless integration between the detailed performance
//! profiling system and the production monitoring infrastructure, enabling
//! unified observability across development and production environments.

use std::sync::Arc;
use std::time::Duration;

use crate::production_monitoring::ProductionMonitor;
use crate::profiling::{PerformanceProfiler, ProfilingConfig, ProfilingReport};
use crate::{AcousticError, Result};

/// Integrated profiling and monitoring system
///
/// This combines the detailed performance profiling capabilities with
/// production monitoring for comprehensive observability.
pub struct IntegratedMonitor {
    /// Performance profiler for detailed timing and memory analysis
    profiler: PerformanceProfiler,
    /// Production monitor for high-level metrics and alerting
    production_monitor: Arc<ProductionMonitor>,
    /// Configuration
    config: IntegrationConfig,
}

/// Configuration for integrated monitoring
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    /// Enable automatic metric sync
    pub auto_sync_metrics: bool,
    /// Sync interval (seconds)
    pub sync_interval_secs: u64,
    /// Sync slow operations to production alerts
    pub sync_slow_operations: bool,
    /// Slow operation threshold (milliseconds)
    pub slow_operation_threshold_ms: u64,
    /// Enable profiling report export to monitoring
    pub export_profiling_reports: bool,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            auto_sync_metrics: true,
            sync_interval_secs: 60,
            sync_slow_operations: true,
            slow_operation_threshold_ms: 100,
            export_profiling_reports: true,
        }
    }
}

impl IntegrationConfig {
    /// Production configuration with conservative settings
    pub fn production() -> Self {
        Self {
            auto_sync_metrics: true,
            sync_interval_secs: 300,
            sync_slow_operations: true,
            slow_operation_threshold_ms: 200,
            export_profiling_reports: false, // Reduce overhead
        }
    }

    /// Development configuration with detailed tracking
    pub fn development() -> Self {
        Self {
            auto_sync_metrics: true,
            sync_interval_secs: 30,
            sync_slow_operations: true,
            slow_operation_threshold_ms: 50,
            export_profiling_reports: true,
        }
    }
}

impl IntegratedMonitor {
    /// Create a new integrated monitor
    pub fn new(profiling_config: ProfilingConfig, integration_config: IntegrationConfig) -> Self {
        Self {
            profiler: PerformanceProfiler::new(profiling_config),
            production_monitor: Arc::new(ProductionMonitor::new()),
            config: integration_config,
        }
    }

    /// Create with default configurations
    pub fn default_config() -> Self {
        Self::new(
            ProfilingConfig::production(),
            IntegrationConfig::production(),
        )
    }

    /// Get the performance profiler
    pub fn profiler(&self) -> &PerformanceProfiler {
        &self.profiler
    }

    /// Get the production monitor
    pub fn production_monitor(&self) -> &Arc<ProductionMonitor> {
        &self.production_monitor
    }

    /// Begin a profiled span with automatic production monitoring
    pub fn begin_span(&self, operation: impl Into<String>) -> Result<ProfiledSpan> {
        let operation_name = operation.into();
        let span_guard = self.profiler.begin_span(operation_name.clone())?;

        Ok(ProfiledSpan {
            operation_name,
            span_guard,
            production_monitor: Arc::clone(&self.production_monitor),
            start_time: std::time::Instant::now(),
            sync_to_production: self.config.auto_sync_metrics,
        })
    }

    /// Sync profiling statistics to production monitoring
    pub fn sync_to_production(&self) -> Result<SyncReport> {
        let report = self.profiler.generate_report()?;

        let mut synced_operations = 0;
        let mut slow_operations = 0;

        // Sync each operation's statistics
        for profile in &report.operation_profiles {
            synced_operations += 1;

            // Record in production monitoring
            let avg_duration = profile.timing.avg_duration;
            self.production_monitor.record_synthesis(
                avg_duration,
                profile.failure_count == 0,
                profile.invocation_count,
            );

            // Check for slow operations
            if self.config.sync_slow_operations
                && avg_duration.as_millis() as u64 > self.config.slow_operation_threshold_ms
            {
                slow_operations += 1;

                // Update health status for slow operations
                self.production_monitor
                    .update_health(&format!("{}_latency", profile.operation), false);
            }
        }

        Ok(SyncReport {
            synced_operations,
            slow_operations_detected: slow_operations,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Generate a comprehensive unified report
    pub fn generate_unified_report(&self) -> Result<UnifiedReport> {
        let profiling_report = self.profiler.generate_report()?;
        let production_metrics = self.production_monitor.get_metrics()?;
        let health_status = self.production_monitor.get_health()?;

        Ok(UnifiedReport {
            profiling_report,
            production_metrics,
            health_status,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Export profiling report as JSON
    pub fn export_json_report(&self) -> Result<String> {
        let profiling_report = self.profiler.generate_report()?;
        profiling_report.to_json()
    }

    /// Analyze performance and generate recommendations
    pub fn analyze_performance(&self) -> Result<PerformanceAnalysis> {
        let report = self.profiler.generate_report()?;

        let slowest = report.slowest_operations(5);
        let memory_intensive = report.most_memory_intensive(5);

        let mut recommendations = Vec::new();

        // Analyze slow operations
        for profile in &slowest {
            if profile.timing.avg_duration.as_millis() > 100 {
                recommendations.push(format!(
                    "Operation '{}' is slow (avg: {:?}). Consider optimization.",
                    profile.operation, profile.timing.avg_duration
                ));
            }

            // Check for high variance (P99 >> P50)
            let p99_ms = profile.timing.p99_duration.as_millis();
            let p50_ms = profile.timing.median_duration.as_millis();
            if p99_ms > p50_ms * 3 {
                recommendations.push(format!(
                    "Operation '{}' has high latency variance (P99: {}ms, P50: {}ms). Investigate inconsistent performance.",
                    profile.operation, p99_ms, p50_ms
                ));
            }
        }

        // Analyze memory usage
        for profile in &memory_intensive {
            if profile.memory.peak_memory_bytes > 100 * 1024 * 1024 {
                // > 100MB
                recommendations.push(format!(
                    "Operation '{}' uses significant memory (peak: {} MB). Consider memory optimization.",
                    profile.operation,
                    profile.memory.peak_memory_bytes / (1024 * 1024)
                ));
            }
        }

        // Check failure rates
        for profile in &report.operation_profiles {
            if profile.invocation_count > 0 {
                let failure_rate =
                    profile.failure_count as f64 / profile.invocation_count as f64 * 100.0;
                if failure_rate > 1.0 {
                    recommendations.push(format!(
                        "Operation '{}' has high failure rate ({:.1}%). Investigate error handling.",
                        profile.operation, failure_rate
                    ));
                }
            }
        }

        Ok(PerformanceAnalysis {
            total_operations: report.operation_profiles.len(),
            slow_operations: slowest.len(),
            memory_intensive_operations: memory_intensive.len(),
            recommendations,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Clear all profiling data
    pub fn clear_profiling_data(&self) -> Result<()> {
        self.profiler.clear()
    }
}

/// Profiled span with automatic production monitoring
pub struct ProfiledSpan {
    operation_name: String,
    span_guard: crate::profiling::SpanGuard,
    production_monitor: Arc<ProductionMonitor>,
    start_time: std::time::Instant,
    sync_to_production: bool,
}

impl ProfiledSpan {
    /// Add a tag to this span
    pub fn add_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.span_guard.add_tag(key, value);
    }

    /// Get the span ID
    pub fn id(&self) -> crate::profiling::SpanId {
        self.span_guard.id()
    }

    /// Mark this operation as a synthesis request
    pub fn mark_as_synthesis(&self, phoneme_count: usize, success: bool) {
        if self.sync_to_production {
            let duration = self.start_time.elapsed();
            self.production_monitor
                .record_synthesis(duration, success, phoneme_count);
        }
    }

    /// Record an error for this operation
    pub fn record_error(&self, error_type: &str) {
        if self.sync_to_production {
            self.production_monitor.record_error(error_type);
        }
    }
}

impl Drop for ProfiledSpan {
    fn drop(&mut self) {
        // ProfiledSpan is dropped, but the inner span_guard will handle
        // the actual span completion
    }
}

/// Report of synchronization between profiling and production monitoring
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SyncReport {
    /// Number of operations synced
    pub synced_operations: usize,
    /// Number of slow operations detected
    pub slow_operations_detected: usize,
    /// Sync timestamp
    pub timestamp: std::time::SystemTime,
}

/// Unified report combining profiling and production metrics
#[derive(Debug, Clone)]
pub struct UnifiedReport {
    /// Detailed profiling report
    pub profiling_report: ProfilingReport,
    /// Production metrics snapshot
    pub production_metrics: crate::production_monitoring::MetricsSnapshot,
    /// Health status
    pub health_status: crate::production_monitoring::HealthStatus,
    /// Report timestamp
    pub timestamp: std::time::SystemTime,
}

/// Performance analysis with recommendations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformanceAnalysis {
    /// Total number of operations profiled
    pub total_operations: usize,
    /// Number of slow operations
    pub slow_operations: usize,
    /// Number of memory-intensive operations
    pub memory_intensive_operations: usize,
    /// Performance recommendations
    pub recommendations: Vec<String>,
    /// Analysis timestamp
    pub timestamp: std::time::SystemTime,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_integrated_monitor_creation() {
        let monitor = IntegratedMonitor::default_config();
        assert!(monitor.profiler().config().is_ok());
    }

    #[test]
    fn test_profiled_span() {
        let monitor = IntegratedMonitor::default_config();

        {
            let _span = monitor.begin_span("test_operation").unwrap();
            thread::sleep(Duration::from_millis(10));
        }

        let report = monitor.profiler().generate_report().unwrap();
        assert_eq!(report.total_spans_recorded, 1);
    }

    #[test]
    fn test_profiled_span_with_tags() {
        let monitor = IntegratedMonitor::default_config();

        {
            let mut span = monitor.begin_span("test_operation").unwrap();
            span.add_tag("model", "vits");
            span.add_tag("language", "en-US");
            thread::sleep(Duration::from_millis(10));
        }

        let spans = monitor.profiler().get_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert!(spans[0].tags.contains_key("model"));
        assert!(spans[0].tags.contains_key("language"));
    }

    #[test]
    fn test_sync_to_production() {
        let monitor = IntegratedMonitor::new(
            ProfilingConfig::development(),
            IntegrationConfig::development(),
        );

        // Create some profiling data
        for _ in 0..5 {
            let _span = monitor.begin_span("synthesis_op").unwrap();
            thread::sleep(Duration::from_millis(10));
        }

        let sync_report = monitor.sync_to_production().unwrap();
        assert!(sync_report.synced_operations > 0);
    }

    #[test]
    fn test_unified_report_generation() {
        let monitor = IntegratedMonitor::default_config();

        {
            let _span = monitor.begin_span("test_op").unwrap();
            thread::sleep(Duration::from_millis(10));
        }

        let report = monitor.generate_unified_report().unwrap();
        assert_eq!(report.profiling_report.total_spans_recorded, 1);
    }

    #[test]
    fn test_performance_analysis() {
        let monitor = IntegratedMonitor::new(
            ProfilingConfig::development(),
            IntegrationConfig::development(),
        );

        // Create some test data
        for _ in 0..3 {
            let _span = monitor.begin_span("fast_op").unwrap();
            thread::sleep(Duration::from_millis(5));
        }

        for _ in 0..2 {
            let _span = monitor.begin_span("slow_op").unwrap();
            thread::sleep(Duration::from_millis(150)); // Trigger slow op detection
        }

        let analysis = monitor.analyze_performance().unwrap();
        assert!(analysis.total_operations > 0);
        assert!(analysis.recommendations.len() > 0);
    }

    #[test]
    fn test_mark_as_synthesis() {
        let monitor = IntegratedMonitor::default_config();

        {
            let span = monitor.begin_span("synthesis_request").unwrap();
            span.mark_as_synthesis(50, true);
            thread::sleep(Duration::from_millis(10));
        }

        let metrics = monitor.production_monitor().get_metrics().unwrap();
        assert!(metrics.total_requests > 0);
    }

    #[test]
    fn test_record_error() {
        let monitor = IntegratedMonitor::default_config();

        {
            let span = monitor.begin_span("failing_operation").unwrap();
            span.record_error("InferenceError");
            thread::sleep(Duration::from_millis(5));
        }

        // Error should be recorded in production monitor
        let metrics = monitor.production_monitor().get_metrics().unwrap();
        assert!(!metrics.error_counts.is_empty());
    }

    #[test]
    fn test_json_export() {
        let monitor = IntegratedMonitor::default_config();

        {
            let _span = monitor.begin_span("test_op").unwrap();
            thread::sleep(Duration::from_millis(5));
        }

        let json = monitor.export_json_report().unwrap();
        assert!(json.contains("operation_profiles"));
        assert!(json.contains("timestamp"));
    }

    #[test]
    fn test_clear_profiling_data() {
        let monitor = IntegratedMonitor::default_config();

        {
            let _span = monitor.begin_span("test_op").unwrap();
            thread::sleep(Duration::from_millis(5));
        }

        assert_eq!(monitor.profiler().get_spans().unwrap().len(), 1);

        monitor.clear_profiling_data().unwrap();

        assert_eq!(monitor.profiler().get_spans().unwrap().len(), 0);
    }

    #[test]
    fn test_config_presets() {
        let prod_config = IntegrationConfig::production();
        assert_eq!(prod_config.sync_interval_secs, 300);
        assert!(!prod_config.export_profiling_reports);

        let dev_config = IntegrationConfig::development();
        assert_eq!(dev_config.sync_interval_secs, 30);
        assert!(dev_config.export_profiling_reports);
    }
}
