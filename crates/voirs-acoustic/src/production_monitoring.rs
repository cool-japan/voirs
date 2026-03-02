//! Advanced production monitoring and observability
//!
//! This module provides comprehensive monitoring, metrics collection, and observability
//! features for production deployments of the acoustic model system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

use crate::{AcousticError, Result};

/// Production monitoring system for acoustic models
pub struct ProductionMonitor {
    /// Metrics collector
    metrics: Arc<Mutex<MetricsCollector>>,
    /// Health checker
    health: Arc<Mutex<HealthChecker>>,
    /// Alert manager
    alerts: Arc<Mutex<AlertManager>>,
    /// Performance tracker
    performance: Arc<Mutex<PerformanceTracker>>,
}

impl ProductionMonitor {
    /// Create a new production monitor
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(MetricsCollector::new())),
            health: Arc::new(Mutex::new(HealthChecker::new())),
            alerts: Arc::new(Mutex::new(AlertManager::new())),
            performance: Arc::new(Mutex::new(PerformanceTracker::new())),
        }
    }

    /// Record a synthesis request
    pub fn record_synthesis(&self, duration: Duration, success: bool, phoneme_count: usize) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.record_synthesis(duration, success, phoneme_count);
        }

        if let Ok(mut performance) = self.performance.lock() {
            performance.record_latency(duration);
        }

        // Check for alerts
        if let Ok(mut alerts) = self.alerts.lock() {
            if duration.as_millis() > 1000 {
                alerts.trigger_alert(Alert {
                    severity: AlertSeverity::Warning,
                    message: format!("High synthesis latency: {}ms", duration.as_millis()),
                    timestamp: SystemTime::now(),
                });
            }
        }
    }

    /// Record an error by type
    pub fn record_error(&self, error_type: &str) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.record_error(error_type);
        }
    }

    /// Record model usage
    pub fn record_model_usage(&self, model_name: &str) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.record_model_usage(model_name);
        }
    }

    /// Sample resource usage
    pub fn sample_resources(&self, memory_mb: f64, cpu_percent: f64) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.sample_resources(memory_mb, cpu_percent);
        }
    }

    /// Update health status
    pub fn update_health(&self, component: &str, healthy: bool) {
        if let Ok(mut health) = self.health.lock() {
            health.update_component(component, healthy);
        }
    }

    /// Get current metrics snapshot
    pub fn get_metrics(&self) -> Result<MetricsSnapshot> {
        self.metrics
            .lock()
            .map_err(|_| AcousticError::ConfigError {
                message: "Failed to lock metrics".to_string(),
            })
            .map(|m| m.snapshot())
    }

    /// Get health status
    pub fn get_health(&self) -> Result<HealthStatus> {
        self.health
            .lock()
            .map_err(|_| AcousticError::ConfigError {
                message: "Failed to lock health".to_string(),
            })
            .map(|h| h.status())
    }

    /// Get active alerts
    pub fn get_alerts(&self) -> Result<Vec<Alert>> {
        self.alerts
            .lock()
            .map_err(|_| AcousticError::ConfigError {
                message: "Failed to lock alerts".to_string(),
            })
            .map(|a| a.get_active_alerts())
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> Result<PerformanceSummary> {
        self.performance
            .lock()
            .map_err(|_| AcousticError::ConfigError {
                message: "Failed to lock performance".to_string(),
            })
            .map(|p| p.summary())
    }

    /// Generate comprehensive monitoring report
    pub fn generate_report(&self) -> Result<MonitoringReport> {
        Ok(MonitoringReport {
            timestamp: SystemTime::now(),
            metrics: self.get_metrics()?,
            health: self.get_health()?,
            alerts: self.get_alerts()?,
            performance: self.get_performance_summary()?,
        })
    }

    /// Detect performance degradation
    pub fn detect_degradation(&self) -> Result<Option<PerformanceDegradation>> {
        self.performance
            .lock()
            .map_err(|_| AcousticError::ConfigError {
                message: "Failed to lock performance tracker".to_string(),
            })
            .map(|p| p.detect_degradation())
    }

    /// Detect performance anomalies
    pub fn detect_anomalies(&self) -> Result<Vec<PerformanceAnomaly>> {
        self.performance
            .lock()
            .map_err(|_| AcousticError::ConfigError {
                message: "Failed to lock performance tracker".to_string(),
            })
            .map(|p| p.detect_anomalies())
    }

    /// Get error breakdown by type
    pub fn get_error_breakdown(&self) -> Result<HashMap<String, u64>> {
        self.metrics
            .lock()
            .map_err(|_| AcousticError::ConfigError {
                message: "Failed to lock metrics".to_string(),
            })
            .map(|m| m.snapshot().error_counts)
    }

    /// Get model usage statistics
    pub fn get_model_usage(&self) -> Result<HashMap<String, u64>> {
        self.metrics
            .lock()
            .map_err(|_| AcousticError::ConfigError {
                message: "Failed to lock metrics".to_string(),
            })
            .map(|m| m.snapshot().model_usage)
    }
}

impl Default for ProductionMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics collector for tracking system usage
#[derive(Debug)]
pub struct MetricsCollector {
    /// Total synthesis requests
    total_requests: u64,
    /// Successful synthesis requests
    successful_requests: u64,
    /// Failed synthesis requests
    failed_requests: u64,
    /// Total synthesis time
    total_duration: Duration,
    /// Total phonemes processed
    total_phonemes: u64,
    /// Request timestamps for rate calculation
    request_timestamps: Vec<Instant>,
    /// Start time
    start_time: Instant,
    /// Error counts by error type
    error_counts: HashMap<String, u64>,
    /// Request counts by model type
    model_usage: HashMap<String, u64>,
    /// Throughput history (requests per minute over time)
    throughput_history: Vec<(Instant, f64)>,
    /// Peak throughput (requests per second)
    peak_throughput: f64,
    /// Resource usage samples
    resource_samples: Vec<ResourceUsage>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_duration: Duration::ZERO,
            total_phonemes: 0,
            request_timestamps: Vec::new(),
            start_time: Instant::now(),
            error_counts: HashMap::new(),
            model_usage: HashMap::new(),
            throughput_history: Vec::new(),
            peak_throughput: 0.0,
            resource_samples: Vec::new(),
        }
    }

    /// Record an error by type
    pub fn record_error(&mut self, error_type: &str) {
        *self.error_counts.entry(error_type.to_string()).or_insert(0) += 1;
    }

    /// Record model usage
    pub fn record_model_usage(&mut self, model_name: &str) {
        *self.model_usage.entry(model_name.to_string()).or_insert(0) += 1;
    }

    /// Sample resource usage
    pub fn sample_resources(&mut self, memory_mb: f64, cpu_percent: f64) {
        let sample = ResourceUsage {
            timestamp: Instant::now(),
            memory_mb,
            cpu_percent,
        };
        self.resource_samples.push(sample);

        // Keep only last 1000 samples
        if self.resource_samples.len() > 1000 {
            self.resource_samples.remove(0);
        }
    }

    /// Calculate current throughput and update peak
    fn update_throughput(&mut self) {
        if self.request_timestamps.len() < 2 {
            return;
        }

        // Calculate requests per second over last minute
        let now = Instant::now();
        let minute_ago = now - Duration::from_secs(60);
        let recent_requests = self
            .request_timestamps
            .iter()
            .filter(|&&ts| ts > minute_ago)
            .count();

        let throughput = recent_requests as f64 / 60.0;
        if throughput > self.peak_throughput {
            self.peak_throughput = throughput;
        }

        // Record throughput history (every minute)
        if self.throughput_history.is_empty()
            || now.duration_since(
                self.throughput_history
                    .last()
                    .expect("checked non-empty in condition above")
                    .0,
            ) > Duration::from_secs(60)
        {
            self.throughput_history.push((now, throughput));

            // Keep only last 24 hours
            if self.throughput_history.len() > 1440 {
                self.throughput_history.remove(0);
            }
        }
    }

    pub fn record_synthesis(&mut self, duration: Duration, success: bool, phoneme_count: usize) {
        self.total_requests += 1;
        if success {
            self.successful_requests += 1;
        } else {
            self.failed_requests += 1;
        }
        self.total_duration += duration;
        self.total_phonemes += phoneme_count as u64;
        self.request_timestamps.push(Instant::now());

        // Keep only last 1000 timestamps for rate calculation
        if self.request_timestamps.len() > 1000 {
            self.request_timestamps.remove(0);
        }

        // Update throughput tracking
        self.update_throughput();

        debug!(
            "Metrics: total={}, success={}, duration={}ms",
            self.total_requests,
            self.successful_requests,
            duration.as_millis()
        );
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let uptime = self.start_time.elapsed();
        let requests_per_second = if uptime.as_secs() > 0 {
            self.total_requests as f64 / uptime.as_secs_f64()
        } else {
            0.0
        };

        let avg_duration = if self.total_requests > 0 {
            self.total_duration / self.total_requests as u32
        } else {
            Duration::ZERO
        };

        let success_rate = if self.total_requests > 0 {
            (self.successful_requests as f64 / self.total_requests as f64) * 100.0
        } else {
            0.0
        };

        let error_rate = if self.total_requests > 0 {
            (self.failed_requests as f64 / self.total_requests as f64) * 100.0
        } else {
            0.0
        };

        // Calculate average resource usage
        let avg_memory = if !self.resource_samples.is_empty() {
            self.resource_samples
                .iter()
                .map(|s| s.memory_mb)
                .sum::<f64>()
                / self.resource_samples.len() as f64
        } else {
            0.0
        };

        let avg_cpu = if !self.resource_samples.is_empty() {
            self.resource_samples
                .iter()
                .map(|s| s.cpu_percent)
                .sum::<f64>()
                / self.resource_samples.len() as f64
        } else {
            0.0
        };

        MetricsSnapshot {
            total_requests: self.total_requests,
            successful_requests: self.successful_requests,
            failed_requests: self.failed_requests,
            success_rate,
            error_rate,
            avg_duration_ms: avg_duration.as_millis() as u64,
            total_phonemes: self.total_phonemes,
            requests_per_second,
            peak_throughput: self.peak_throughput,
            uptime_seconds: uptime.as_secs(),
            error_counts: self.error_counts.clone(),
            model_usage: self.model_usage.clone(),
            avg_memory_mb: avg_memory,
            avg_cpu_percent: avg_cpu,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of current metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub success_rate: f64,
    pub error_rate: f64,
    pub avg_duration_ms: u64,
    pub total_phonemes: u64,
    pub requests_per_second: f64,
    pub peak_throughput: f64,
    pub uptime_seconds: u64,
    pub error_counts: HashMap<String, u64>,
    pub model_usage: HashMap<String, u64>,
    pub avg_memory_mb: f64,
    pub avg_cpu_percent: f64,
}

/// Resource usage sample
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub timestamp: Instant,
    pub memory_mb: f64,
    pub cpu_percent: f64,
}

/// Health checker for monitoring component health
#[derive(Debug)]
pub struct HealthChecker {
    /// Component health states
    components: HashMap<String, ComponentHealth>,
    /// Last check time
    last_check: Instant,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
            last_check: Instant::now(),
        }
    }

    pub fn update_component(&mut self, name: &str, healthy: bool) {
        let health = self
            .components
            .entry(name.to_string())
            .or_insert_with(|| ComponentHealth {
                healthy: true,
                last_update: Instant::now(),
                failure_count: 0,
            });

        health.healthy = healthy;
        health.last_update = Instant::now();

        if !healthy {
            health.failure_count += 1;
            warn!(
                "Component {} is unhealthy (failures: {})",
                name, health.failure_count
            );
        } else if health.failure_count > 0 {
            info!(
                "Component {} recovered after {} failures",
                name, health.failure_count
            );
            health.failure_count = 0;
        }

        self.last_check = Instant::now();
    }

    pub fn status(&self) -> HealthStatus {
        let all_healthy = self.components.values().all(|h| h.healthy);
        let component_count = self.components.len();
        let healthy_count = self.components.values().filter(|h| h.healthy).count();

        HealthStatus {
            overall_healthy: all_healthy,
            component_count,
            healthy_count,
            last_check: self.last_check,
            components: self
                .components
                .iter()
                .map(|(name, health)| (name.clone(), health.healthy))
                .collect(),
        }
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct ComponentHealth {
    healthy: bool,
    last_update: Instant,
    failure_count: u32,
}

/// Health status report
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub overall_healthy: bool,
    pub component_count: usize,
    pub healthy_count: usize,
    pub last_check: Instant,
    pub components: HashMap<String, bool>,
}

/// Serializable health status (without Instant)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableHealthStatus {
    pub overall_healthy: bool,
    pub component_count: usize,
    pub healthy_count: usize,
    pub components: HashMap<String, bool>,
}

impl From<HealthStatus> for SerializableHealthStatus {
    fn from(status: HealthStatus) -> Self {
        Self {
            overall_healthy: status.overall_healthy,
            component_count: status.component_count,
            healthy_count: status.healthy_count,
            components: status.components,
        }
    }
}

/// Alert manager for handling system alerts
#[derive(Debug)]
pub struct AlertManager {
    /// Active alerts
    alerts: Vec<Alert>,
    /// Maximum alerts to keep
    max_alerts: usize,
}

impl AlertManager {
    pub fn new() -> Self {
        Self {
            alerts: Vec::new(),
            max_alerts: 100,
        }
    }

    pub fn trigger_alert(&mut self, alert: Alert) {
        warn!("{:?}: {}", alert.severity, alert.message);
        self.alerts.push(alert);

        // Keep only recent alerts
        if self.alerts.len() > self.max_alerts {
            self.alerts.remove(0);
        }
    }

    pub fn get_active_alerts(&self) -> Vec<Alert> {
        // Return alerts from last 24 hours
        let cutoff = SystemTime::now() - Duration::from_secs(86400);
        self.alerts
            .iter()
            .filter(|a| a.timestamp > cutoff)
            .cloned()
            .collect()
    }

    pub fn clear_alerts(&mut self) {
        self.alerts.clear();
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

/// System alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Performance tracker with degradation detection
#[derive(Debug)]
pub struct PerformanceTracker {
    /// Latency samples
    latencies: Vec<Duration>,
    /// Maximum samples to keep
    max_samples: usize,
    /// Baseline performance (established after initial samples)
    baseline_p50: Option<Duration>,
    baseline_p95: Option<Duration>,
    /// Performance history for trend analysis
    performance_history: Vec<(Instant, Duration)>,
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            latencies: Vec::new(),
            max_samples: 1000,
            baseline_p50: None,
            baseline_p95: None,
            performance_history: Vec::new(),
        }
    }

    pub fn record_latency(&mut self, latency: Duration) {
        self.latencies.push(latency);
        if self.latencies.len() > self.max_samples {
            self.latencies.remove(0);
        }

        // Record for history tracking
        self.performance_history.push((Instant::now(), latency));
        if self.performance_history.len() > 10000 {
            self.performance_history.remove(0);
        }

        // Establish baseline after 100 samples
        if self.baseline_p50.is_none() && self.latencies.len() >= 100 {
            self.establish_baseline();
        }
    }

    /// Establish performance baseline
    fn establish_baseline(&mut self) {
        if self.latencies.len() < 100 {
            return;
        }

        let mut sorted = self.latencies.clone();
        sorted.sort();

        self.baseline_p50 = Some(sorted[sorted.len() / 2]);
        self.baseline_p95 = Some(sorted[(sorted.len() as f64 * 0.95) as usize]);

        info!(
            "Performance baseline established: p50={:?}, p95={:?}",
            self.baseline_p50, self.baseline_p95
        );
    }

    /// Detect performance degradation compared to baseline
    pub fn detect_degradation(&self) -> Option<PerformanceDegradation> {
        if self.latencies.len() < 50 || self.baseline_p50.is_none() {
            return None;
        }

        let mut recent_latencies: Vec<_> = self.latencies.iter().rev().take(50).copied().collect();
        recent_latencies.sort();

        let recent_p50 = recent_latencies[recent_latencies.len() / 2];
        let recent_p95 = recent_latencies[(recent_latencies.len() as f64 * 0.95) as usize];

        let baseline_p50 = self.baseline_p50?;
        let baseline_p95 = self.baseline_p95?;

        // Check for significant degradation (>50% worse)
        let p50_degradation =
            (recent_p50.as_millis() as f64 / baseline_p50.as_millis() as f64) - 1.0;
        let p95_degradation =
            (recent_p95.as_millis() as f64 / baseline_p95.as_millis() as f64) - 1.0;

        if p50_degradation > 0.5 || p95_degradation > 0.5 {
            Some(PerformanceDegradation {
                baseline_p50_ms: baseline_p50.as_millis() as u64,
                current_p50_ms: recent_p50.as_millis() as u64,
                p50_degradation_percent: p50_degradation * 100.0,
                baseline_p95_ms: baseline_p95.as_millis() as u64,
                current_p95_ms: recent_p95.as_millis() as u64,
                p95_degradation_percent: p95_degradation * 100.0,
            })
        } else {
            None
        }
    }

    /// Detect anomalies in recent performance
    pub fn detect_anomalies(&self) -> Vec<PerformanceAnomaly> {
        if self.latencies.len() < 100 {
            return Vec::new();
        }

        // Calculate mean and standard deviation
        let mean_ms = self
            .latencies
            .iter()
            .map(|d| d.as_millis() as f64)
            .sum::<f64>()
            / self.latencies.len() as f64;

        let variance = self
            .latencies
            .iter()
            .map(|d| {
                let diff = d.as_millis() as f64 - mean_ms;
                diff * diff
            })
            .sum::<f64>()
            / self.latencies.len() as f64;

        let std_dev = variance.sqrt();

        // Find outliers (>3 standard deviations from mean)
        let mut anomalies = Vec::new();
        for (i, latency) in self.latencies.iter().enumerate().rev().take(50) {
            let z_score = (latency.as_millis() as f64 - mean_ms) / std_dev;
            if z_score.abs() > 3.0 {
                anomalies.push(PerformanceAnomaly {
                    latency_ms: latency.as_millis() as u64,
                    z_score,
                    deviation_from_mean_percent: ((latency.as_millis() as f64 / mean_ms - 1.0)
                        * 100.0),
                    sample_index: i,
                });
            }
        }

        anomalies
    }

    pub fn summary(&self) -> PerformanceSummary {
        if self.latencies.is_empty() {
            return PerformanceSummary::default();
        }

        let mut sorted = self.latencies.clone();
        sorted.sort();

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let median = sorted[sorted.len() / 2];
        let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];
        let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];

        let avg = Duration::from_millis(
            self.latencies.iter().map(|d| d.as_millis()).sum::<u128>() as u64
                / self.latencies.len() as u64,
        );

        PerformanceSummary {
            sample_count: self.latencies.len(),
            min_ms: min.as_millis() as u64,
            max_ms: max.as_millis() as u64,
            avg_ms: avg.as_millis() as u64,
            median_ms: median.as_millis() as u64,
            p95_ms: p95.as_millis() as u64,
            p99_ms: p99.as_millis() as u64,
        }
    }
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub sample_count: usize,
    pub min_ms: u64,
    pub max_ms: u64,
    pub avg_ms: u64,
    pub median_ms: u64,
    pub p95_ms: u64,
    pub p99_ms: u64,
}

impl Default for PerformanceSummary {
    fn default() -> Self {
        Self {
            sample_count: 0,
            min_ms: 0,
            max_ms: 0,
            avg_ms: 0,
            median_ms: 0,
            p95_ms: 0,
            p99_ms: 0,
        }
    }
}

/// Performance degradation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDegradation {
    pub baseline_p50_ms: u64,
    pub current_p50_ms: u64,
    pub p50_degradation_percent: f64,
    pub baseline_p95_ms: u64,
    pub current_p95_ms: u64,
    pub p95_degradation_percent: f64,
}

/// Performance anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnomaly {
    pub latency_ms: u64,
    pub z_score: f64,
    pub deviation_from_mean_percent: f64,
    pub sample_index: usize,
}

/// Comprehensive monitoring report
#[derive(Debug, Clone)]
pub struct MonitoringReport {
    pub timestamp: SystemTime,
    pub metrics: MetricsSnapshot,
    pub health: HealthStatus,
    pub alerts: Vec<Alert>,
    pub performance: PerformanceSummary,
}

impl MonitoringReport {
    /// Generate a human-readable summary
    pub fn summary(&self) -> String {
        let timestamp_secs = self
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        format!(
            "=== Production Monitoring Report ===\n\
             Timestamp: {}\n\
             \n\
             Metrics:\n\
             - Total Requests: {}\n\
             - Success Rate: {:.2}%\n\
             - Avg Duration: {}ms\n\
             - Requests/sec: {:.2}\n\
             - Uptime: {}s\n\
             \n\
             Health:\n\
             - Overall: {}\n\
             - Components: {}/{} healthy\n\
             \n\
             Performance:\n\
             - Min/Avg/Max: {}/{}ms/{} ms\n\
             - P95/P99: {}/{} ms\n\
             - Samples: {}\n\
             \n\
             Alerts: {} active\n\
             ",
            timestamp_secs,
            self.metrics.total_requests,
            self.metrics.success_rate,
            self.metrics.avg_duration_ms,
            self.metrics.requests_per_second,
            self.metrics.uptime_seconds,
            if self.health.overall_healthy {
                "HEALTHY"
            } else {
                "DEGRADED"
            },
            self.health.healthy_count,
            self.health.component_count,
            self.performance.min_ms,
            self.performance.avg_ms,
            self.performance.max_ms,
            self.performance.p95_ms,
            self.performance.p99_ms,
            self.performance.sample_count,
            self.alerts.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_monitor_creation() {
        let monitor = ProductionMonitor::new();
        let metrics = monitor.get_metrics().unwrap();
        assert_eq!(metrics.total_requests, 0);
    }

    #[test]
    fn test_metrics_recording() {
        let monitor = ProductionMonitor::new();
        monitor.record_synthesis(Duration::from_millis(100), true, 50);
        let metrics = monitor.get_metrics().unwrap();
        assert_eq!(metrics.total_requests, 1);
        assert_eq!(metrics.successful_requests, 1);
    }

    #[test]
    fn test_health_checker() {
        let monitor = ProductionMonitor::new();
        monitor.update_health("model", true);
        let health = monitor.get_health().unwrap();
        assert!(health.overall_healthy);
        assert_eq!(health.healthy_count, 1);
    }

    #[test]
    fn test_alert_manager() {
        let monitor = ProductionMonitor::new();
        monitor.record_synthesis(Duration::from_millis(2000), true, 50);
        let alerts = monitor.get_alerts().unwrap();
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_monitoring_report() {
        let monitor = ProductionMonitor::new();
        monitor.record_synthesis(Duration::from_millis(100), true, 50);
        let report = monitor.generate_report().unwrap();
        assert_eq!(report.metrics.total_requests, 1);
        let summary = report.summary();
        assert!(summary.contains("Total Requests: 1"));
    }

    #[test]
    fn test_error_tracking() {
        let monitor = ProductionMonitor::new();
        monitor.record_error("InferenceError");
        monitor.record_error("ModelError");
        monitor.record_error("InferenceError");

        let error_breakdown = monitor.get_error_breakdown().unwrap();
        assert_eq!(*error_breakdown.get("InferenceError").unwrap(), 2);
        assert_eq!(*error_breakdown.get("ModelError").unwrap(), 1);
    }

    #[test]
    fn test_model_usage_tracking() {
        let monitor = ProductionMonitor::new();
        monitor.record_model_usage("VITS");
        monitor.record_model_usage("FastSpeech2");
        monitor.record_model_usage("VITS");
        monitor.record_model_usage("VITS");

        let model_usage = monitor.get_model_usage().unwrap();
        assert_eq!(*model_usage.get("VITS").unwrap(), 3);
        assert_eq!(*model_usage.get("FastSpeech2").unwrap(), 1);
    }

    #[test]
    fn test_resource_usage_tracking() {
        let monitor = ProductionMonitor::new();
        monitor.sample_resources(512.0, 45.5);
        monitor.sample_resources(524.0, 48.2);
        monitor.sample_resources(518.0, 46.8);

        let metrics = monitor.get_metrics().unwrap();
        assert!(metrics.avg_memory_mb > 510.0 && metrics.avg_memory_mb < 530.0);
        assert!(metrics.avg_cpu_percent > 45.0 && metrics.avg_cpu_percent < 49.0);
    }

    #[test]
    fn test_throughput_tracking() {
        let monitor = ProductionMonitor::new();
        // Record multiple requests
        for _ in 0..10 {
            monitor.record_synthesis(Duration::from_millis(50), true, 20);
        }

        let metrics = monitor.get_metrics().unwrap();
        assert!(metrics.peak_throughput >= 0.0);
        assert!(metrics.requests_per_second >= 0.0);
    }

    #[test]
    fn test_error_rate_calculation() {
        let monitor = ProductionMonitor::new();
        monitor.record_synthesis(Duration::from_millis(100), true, 50);
        monitor.record_synthesis(Duration::from_millis(150), false, 40);
        monitor.record_synthesis(Duration::from_millis(120), true, 45);
        monitor.record_synthesis(Duration::from_millis(110), false, 42);

        let metrics = monitor.get_metrics().unwrap();
        assert_eq!(metrics.total_requests, 4);
        assert_eq!(metrics.successful_requests, 2);
        assert_eq!(metrics.failed_requests, 2);
        assert_eq!(metrics.success_rate, 50.0);
        assert_eq!(metrics.error_rate, 50.0);
    }

    #[test]
    fn test_performance_degradation_detection() {
        let monitor = ProductionMonitor::new();

        // Establish baseline with good performance
        for _ in 0..150 {
            monitor.record_synthesis(Duration::from_millis(100), true, 50);
        }

        // Add degraded performance samples
        for _ in 0..60 {
            monitor.record_synthesis(Duration::from_millis(250), true, 50);
        }

        let degradation = monitor.detect_degradation().unwrap();
        assert!(degradation.is_some());

        if let Some(deg) = degradation {
            assert!(deg.p50_degradation_percent > 50.0);
            assert!(deg.current_p50_ms > deg.baseline_p50_ms);
        }
    }

    #[test]
    fn test_performance_anomaly_detection() {
        let monitor = ProductionMonitor::new();

        // Add normal performance samples
        for _ in 0..100 {
            monitor.record_synthesis(Duration::from_millis(100), true, 50);
        }

        // Add anomalous samples
        monitor.record_synthesis(Duration::from_millis(5000), true, 50);
        monitor.record_synthesis(Duration::from_millis(4500), true, 50);

        let anomalies = monitor.detect_anomalies().unwrap();
        assert!(!anomalies.is_empty());

        for anomaly in &anomalies {
            assert!(anomaly.z_score.abs() > 3.0);
            assert!(anomaly.latency_ms > 1000);
        }
    }

    #[test]
    fn test_comprehensive_metrics_snapshot() {
        let monitor = ProductionMonitor::new();

        // Record various activities
        monitor.record_synthesis(Duration::from_millis(100), true, 50);
        monitor.record_synthesis(Duration::from_millis(150), false, 40);
        monitor.record_error("TimeoutError");
        monitor.record_model_usage("VITS");
        monitor.sample_resources(512.0, 45.5);

        let metrics = monitor.get_metrics().unwrap();

        // Verify all new fields are populated
        assert_eq!(metrics.total_requests, 2);
        assert_eq!(metrics.successful_requests, 1);
        assert_eq!(metrics.failed_requests, 1);
        assert!(metrics.error_rate > 0.0);
        assert!(metrics.peak_throughput >= 0.0);
        assert!(!metrics.error_counts.is_empty());
        assert!(!metrics.model_usage.is_empty());
        assert!(metrics.avg_memory_mb > 0.0);
        assert!(metrics.avg_cpu_percent > 0.0);
    }
}
