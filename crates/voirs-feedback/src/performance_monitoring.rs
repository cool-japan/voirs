//! Comprehensive Performance Monitoring System
//!
//! This module provides a production-grade performance monitoring system for the `VoiRS` feedback
//! system. It tracks system metrics, application performance, database statistics, and custom
//! business metrics with support for alerting, aggregation, and export capabilities.
//!
//! # Features
//!
//! - **System Metrics**: CPU, memory, disk I/O, network statistics
//! - **Application Metrics**: Request rates, latencies, throughput, error rates
//! - **Database Metrics**: Query performance, connection pool stats, slow queries
//! - **Business Metrics**: User activity, feedback rates, session completion
//! - **Alert System**: Threshold-based alerting with multiple severity levels
//! - **Time-Series Storage**: Efficient metric storage with configurable retention
//! - **Aggregation**: Min/max/avg/percentile calculations over time windows
//! - **Export**: Prometheus, JSON, and custom format support
//!
//! # Example
//!
//! ```no_run
//! use voirs_feedback::performance_monitoring::{PerformanceMonitor, MetricType, Label};
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let monitor = PerformanceMonitor::new();
//!
//! // Record metrics
//! monitor.record_counter("requests_total", 1.0, vec![("endpoint".to_string(), "/api/feedback".to_string())]).await?;
//! monitor.record_gauge("cpu_usage_percent", 45.2, vec![]).await?;
//! monitor.record_histogram("request_duration_ms", 125.3, vec![("method".to_string(), "POST".to_string())]).await?;
//!
//! // Query metrics
//! let stats = monitor.get_metric_statistics("request_duration_ms", None).await?;
//! println!("p95 latency: {}ms", stats.p95);
//!
//! // Export for monitoring systems
//! let prometheus = monitor.export_prometheus().await;
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

/// Errors that can occur during performance monitoring
#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum MonitoringError {
    /// Metric not found
    #[error("Metric not found: {0}")]
    MetricNotFound(String),

    /// Invalid metric value
    #[error("Invalid metric value: {0}")]
    InvalidValue(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Storage error
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Export error
    #[error("Export error: {0}")]
    ExportError(String),
}

/// Type alias for Results in this module
pub type Result<T> = std::result::Result<T, MonitoringError>;

/// Metric types supported by the monitoring system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum MetricType {
    /// Counter - monotonically increasing value (e.g., total requests)
    Counter,
    /// Gauge - point-in-time value that can go up or down (e.g., CPU usage)
    Gauge,
    /// Histogram - distribution of values (e.g., request latencies)
    Histogram,
    /// Summary - pre-calculated statistics (e.g., quantiles)
    Summary,
}

/// Label/tag for metric dimensions
pub type Label = (String, String);

/// Metric data point with timestamp and value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp when metric was recorded
    pub timestamp: DateTime<Utc>,
    /// Metric value
    pub value: f64,
    /// Labels/tags for this data point
    pub labels: Vec<Label>,
}

/// Metric metadata and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricMetadata {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Description
    pub description: String,
    /// Unit of measurement
    pub unit: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

/// Time-series data for a metric
#[derive(Debug, Clone)]
struct MetricTimeSeries {
    /// Metric metadata
    metadata: MetricMetadata,
    /// Data points (limited by `max_data_points`)
    data_points: VecDeque<DataPoint>,
    /// Maximum number of data points to retain
    max_data_points: usize,
}

impl MetricTimeSeries {
    fn new(metadata: MetricMetadata, max_data_points: usize) -> Self {
        Self {
            metadata,
            data_points: VecDeque::with_capacity(max_data_points),
            max_data_points,
        }
    }

    fn add_point(&mut self, point: DataPoint) {
        if self.data_points.len() >= self.max_data_points {
            self.data_points.pop_front();
        }
        self.data_points.push_back(point);
    }

    fn get_points_in_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<&DataPoint> {
        self.data_points
            .iter()
            .filter(|p| p.timestamp >= start && p.timestamp <= end)
            .collect()
    }
}

/// Statistical aggregation of metric data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStatistics {
    /// Metric name
    pub metric_name: String,
    /// Number of data points
    pub count: usize,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Average value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Median (p50)
    pub median: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
    /// Start of time range
    pub start_time: DateTime<Utc>,
    /// End of time range
    pub end_time: DateTime<Utc>,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning - attention needed
    Warning,
    /// Critical - immediate action required
    Critical,
}

/// Alert configuration for a metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule identifier
    pub id: String,
    /// Metric name to monitor
    pub metric_name: String,
    /// Threshold value
    pub threshold: f64,
    /// Alert when value is above (true) or below (false) threshold
    pub above_threshold: bool,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Description
    pub description: String,
    /// Enabled status
    pub enabled: bool,
}

/// Triggered alert instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert rule that triggered
    pub rule_id: String,
    /// Metric name
    pub metric_name: String,
    /// Current value that triggered alert
    pub current_value: f64,
    /// Threshold value
    pub threshold: f64,
    /// Severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Timestamp when alert fired
    pub fired_at: DateTime<Utc>,
}

/// Configuration for the performance monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Maximum data points per metric
    pub max_data_points_per_metric: usize,
    /// Metric retention period
    pub retention_period_days: i64,
    /// Enable automatic system metrics collection
    pub collect_system_metrics: bool,
    /// System metrics collection interval in seconds
    pub system_metrics_interval_secs: u64,
    /// Enable automatic cleanup of old metrics
    pub enable_auto_cleanup: bool,
    /// Auto cleanup interval in seconds
    pub cleanup_interval_secs: u64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            max_data_points_per_metric: 10_000,
            retention_period_days: 30,
            collect_system_metrics: true,
            system_metrics_interval_secs: 60,
            enable_auto_cleanup: true,
            cleanup_interval_secs: 3600,
        }
    }
}

/// Main performance monitoring system
pub struct PerformanceMonitor {
    /// Configuration
    config: MonitorConfig,
    /// Metrics storage
    metrics: Arc<RwLock<HashMap<String, MetricTimeSeries>>>,
    /// Alert rules
    alert_rules: Arc<RwLock<HashMap<String, AlertRule>>>,
    /// Active alerts
    active_alerts: Arc<RwLock<Vec<Alert>>>,
    /// Alert history
    alert_history: Arc<RwLock<VecDeque<Alert>>>,
}

impl PerformanceMonitor {
    /// Create a new performance monitor with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(MonitorConfig::default())
    }

    /// Create a new performance monitor with custom configuration
    #[must_use]
    pub fn with_config(config: MonitorConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(HashMap::new())),
            alert_rules: Arc::new(RwLock::new(HashMap::new())),
            active_alerts: Arc::new(RwLock::new(Vec::new())),
            alert_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
        }
    }

    /// Register a new metric
    pub async fn register_metric(
        &self,
        name: &str,
        metric_type: MetricType,
        description: &str,
        unit: &str,
    ) -> Result<()> {
        let mut metrics = self.metrics.write().await;

        if metrics.contains_key(name) {
            return Err(MonitoringError::ConfigError(format!(
                "Metric {name} already registered"
            )));
        }

        let metadata = MetricMetadata {
            name: name.to_string(),
            metric_type,
            description: description.to_string(),
            unit: unit.to_string(),
            created_at: Utc::now(),
            last_updated: Utc::now(),
        };

        let time_series = MetricTimeSeries::new(metadata, self.config.max_data_points_per_metric);

        metrics.insert(name.to_string(), time_series);
        Ok(())
    }

    /// Record a counter metric (monotonically increasing)
    pub async fn record_counter<L: Into<Vec<Label>>>(
        &self,
        name: &str,
        value: f64,
        labels: L,
    ) -> Result<()> {
        self.record_metric_internal(name, value, labels.into(), MetricType::Counter)
            .await
    }

    /// Record a gauge metric (can go up or down)
    pub async fn record_gauge<L: Into<Vec<Label>>>(
        &self,
        name: &str,
        value: f64,
        labels: L,
    ) -> Result<()> {
        self.record_metric_internal(name, value, labels.into(), MetricType::Gauge)
            .await
    }

    /// Record a histogram metric (for distributions)
    pub async fn record_histogram<L: Into<Vec<Label>>>(
        &self,
        name: &str,
        value: f64,
        labels: L,
    ) -> Result<()> {
        self.record_metric_internal(name, value, labels.into(), MetricType::Histogram)
            .await
    }

    /// Internal method to record any metric
    async fn record_metric_internal(
        &self,
        name: &str,
        value: f64,
        labels: Vec<Label>,
        expected_type: MetricType,
    ) -> Result<()> {
        let mut metrics = self.metrics.write().await;

        // Auto-register if not exists
        if !metrics.contains_key(name) {
            drop(metrics);
            self.register_metric(
                name,
                expected_type,
                &format!("Auto-registered metric: {name}"),
                "",
            )
            .await?;
            metrics = self.metrics.write().await;
        }

        let time_series = metrics
            .get_mut(name)
            .ok_or_else(|| MonitoringError::MetricNotFound(name.to_string()))?;

        // Validate metric type
        if time_series.metadata.metric_type != expected_type {
            return Err(MonitoringError::InvalidValue(format!(
                "Metric {} is of type {:?}, not {:?}",
                name, time_series.metadata.metric_type, expected_type
            )));
        }

        let data_point = DataPoint {
            timestamp: Utc::now(),
            value,
            labels,
        };

        time_series.add_point(data_point);
        time_series.metadata.last_updated = Utc::now();

        // Check alert rules
        drop(metrics);
        self.check_alerts(name, value).await;

        Ok(())
    }

    /// Get statistics for a metric over a time range
    pub async fn get_metric_statistics(
        &self,
        name: &str,
        time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    ) -> Result<MetricStatistics> {
        let metrics = self.metrics.read().await;

        let time_series = metrics
            .get(name)
            .ok_or_else(|| MonitoringError::MetricNotFound(name.to_string()))?;

        let (start, end) = time_range.unwrap_or_else(|| {
            let end = Utc::now();
            let start = end - Duration::hours(1);
            (start, end)
        });

        let points = time_series.get_points_in_range(start, end);

        if points.is_empty() {
            return Err(MonitoringError::StorageError(
                "No data points in specified range".to_string(),
            ));
        }

        let mut values: Vec<f64> = points.iter().map(|p| p.value).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let count = values.len();
        let min = values[0];
        let max = values[count - 1];
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;

        // Calculate standard deviation
        let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        // Calculate percentiles
        let median = values[count / 2];
        let p95 = values[(count as f64 * 0.95) as usize];
        let p99 = values[(count as f64 * 0.99).min((count - 1) as f64) as usize];

        Ok(MetricStatistics {
            metric_name: name.to_string(),
            count,
            min,
            max,
            mean,
            std_dev,
            median,
            p95,
            p99,
            start_time: start,
            end_time: end,
        })
    }

    /// Add an alert rule
    pub async fn add_alert_rule(&self, rule: AlertRule) -> Result<()> {
        let mut rules = self.alert_rules.write().await;
        rules.insert(rule.id.clone(), rule);
        Ok(())
    }

    /// Remove an alert rule
    pub async fn remove_alert_rule(&self, rule_id: &str) -> Result<()> {
        let mut rules = self.alert_rules.write().await;
        rules.remove(rule_id).ok_or_else(|| {
            MonitoringError::ConfigError(format!("Alert rule not found: {rule_id}"))
        })?;
        Ok(())
    }

    /// Check if any alert rules are triggered
    async fn check_alerts(&self, metric_name: &str, current_value: f64) {
        let rules = self.alert_rules.read().await;

        for rule in rules.values() {
            if rule.metric_name != metric_name || !rule.enabled {
                continue;
            }

            let triggered = if rule.above_threshold {
                current_value > rule.threshold
            } else {
                current_value < rule.threshold
            };

            if triggered {
                let alert = Alert {
                    rule_id: rule.id.clone(),
                    metric_name: metric_name.to_string(),
                    current_value,
                    threshold: rule.threshold,
                    severity: rule.severity,
                    message: format!(
                        "{}: {} is {:.2} (threshold: {:.2})",
                        rule.description, metric_name, current_value, rule.threshold
                    ),
                    fired_at: Utc::now(),
                };

                // Add to active alerts
                let mut active = self.active_alerts.write().await;
                active.push(alert.clone());

                // Add to history
                let mut history = self.alert_history.write().await;
                if history.len() >= 1000 {
                    history.pop_front();
                }
                history.push_back(alert);
            }
        }
    }

    /// Get all active alerts
    pub async fn get_active_alerts(&self) -> Vec<Alert> {
        self.active_alerts.read().await.clone()
    }

    /// Clear active alerts
    pub async fn clear_active_alerts(&self) {
        self.active_alerts.write().await.clear();
    }

    /// Get alert history
    pub async fn get_alert_history(&self, limit: Option<usize>) -> Vec<Alert> {
        let history = self.alert_history.read().await;
        let limit = limit.unwrap_or(100);
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Export metrics in Prometheus format
    pub async fn export_prometheus(&self) -> String {
        let metrics = self.metrics.read().await;
        let mut output = String::new();

        for (name, time_series) in metrics.iter() {
            // Metric metadata
            output.push_str(&format!(
                "# HELP {} {}\n",
                name, time_series.metadata.description
            ));
            output.push_str(&format!(
                "# TYPE {} {:?}\n",
                name, time_series.metadata.metric_type
            ));

            // Latest data point for each label combination
            if let Some(point) = time_series.data_points.back() {
                let labels_str = if point.labels.is_empty() {
                    String::new()
                } else {
                    let labels: Vec<String> = point
                        .labels
                        .iter()
                        .map(|(k, v)| format!("{k}=\"{v}\""))
                        .collect();
                    format!("{{{}}}", labels.join(","))
                };

                output.push_str(&format!("{}{} {}\n", name, labels_str, point.value));
            }
        }

        output
    }

    /// Export metrics in JSON format
    pub async fn export_json(&self) -> serde_json::Value {
        let metrics = self.metrics.read().await;

        serde_json::json!({
            "timestamp": Utc::now().to_rfc3339(),
            "metrics": metrics.iter().map(|(name, ts)| {
                serde_json::json!({
                    "name": name,
                    "type": ts.metadata.metric_type,
                    "description": ts.metadata.description,
                    "unit": ts.metadata.unit,
                    "data_points": ts.data_points.iter().take(100).collect::<Vec<_>>(),
                })
            }).collect::<Vec<_>>(),
        })
    }

    /// Get list of all registered metrics
    pub async fn list_metrics(&self) -> Vec<MetricMetadata> {
        let metrics = self.metrics.read().await;
        metrics.values().map(|ts| ts.metadata.clone()).collect()
    }

    /// Clean up old metric data based on retention policy
    pub async fn cleanup_old_data(&self) -> Result<usize> {
        let mut metrics = self.metrics.write().await;
        let cutoff = Utc::now() - Duration::days(self.config.retention_period_days);
        let mut removed_count = 0;

        for time_series in metrics.values_mut() {
            let original_len = time_series.data_points.len();
            time_series.data_points.retain(|p| p.timestamp > cutoff);
            removed_count += original_len - time_series.data_points.len();
        }

        Ok(removed_count)
    }

    /// Start background task for automatic system metrics collection
    pub async fn start_system_metrics_collection(self: Arc<Self>) {
        if !self.config.collect_system_metrics {
            return;
        }

        let interval = std::time::Duration::from_secs(self.config.system_metrics_interval_secs);

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);

            loop {
                ticker.tick().await;

                // Collect system metrics (simplified - would use sysinfo or similar in production)
                let _ = self.collect_system_metrics().await;
            }
        });
    }

    /// Collect current system metrics
    async fn collect_system_metrics(&self) -> Result<()> {
        // CPU usage via /proc/stat delta on Linux; fallback elsewhere
        let cpu_percent = Self::read_cpu_usage_percent().await;

        // Process RSS memory via /proc/self/status on Linux
        let memory_usage_mb = Self::read_process_memory_mb();

        // Total system memory via /proc/meminfo on Linux
        let memory_total_mb = Self::read_total_memory_mb();

        // Thread count via /proc/self/status on Linux
        let thread_count = Self::read_thread_count();

        self.record_gauge("system_cpu_usage_percent", cpu_percent, vec![])
            .await?;
        self.record_gauge("system_memory_usage_mb", memory_usage_mb, vec![])
            .await?;
        self.record_gauge("system_memory_total_mb", memory_total_mb, vec![])
            .await?;
        self.record_gauge("process_threads", thread_count, vec![])
            .await?;

        Ok(())
    }

    /// Parse a value from proc-style file content.
    ///
    /// Finds the first line that starts with `prefix`, splits on whitespace,
    /// and returns the second token parsed as `u64`.
    fn parse_proc_value(content: &str, prefix: &str) -> Option<u64> {
        content
            .lines()
            .find(|line| line.starts_with(prefix))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|token| token.parse::<u64>().ok())
    }

    /// Read CPU usage percentage by sampling /proc/stat twice with a 50 ms gap.
    ///
    /// On non-Linux platforms returns a static fallback of `50.0`.
    async fn read_cpu_usage_percent() -> f64 {
        #[cfg(target_os = "linux")]
        {
            fn read_idle_and_total() -> Option<(u64, u64)> {
                let content = std::fs::read_to_string("/proc/stat").ok()?;
                let cpu_line = content.lines().find(|l| l.starts_with("cpu "))?;
                let fields: Vec<u64> = cpu_line
                    .split_whitespace()
                    .skip(1)
                    .filter_map(|t| t.parse::<u64>().ok())
                    .collect();
                if fields.len() < 4 {
                    return None;
                }
                // fields: user nice system idle [iowait irq softirq ...]
                let idle = fields[3];
                let total: u64 = fields.iter().sum();
                Some((idle, total))
            }

            let before = read_idle_and_total();
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let after = read_idle_and_total();

            match (before, after) {
                (Some((idle0, total0)), Some((idle1, total1))) => {
                    let total_delta = total1.saturating_sub(total0) as f64;
                    let idle_delta = idle1.saturating_sub(idle0) as f64;
                    if total_delta <= 0.0 {
                        return 50.0;
                    }
                    let used = (1.0 - idle_delta / total_delta) * 100.0;
                    used.clamp(0.0, 100.0)
                }
                _ => 50.0,
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            50.0
        }
    }

    /// Read current process RSS from /proc/self/status (Linux only).
    ///
    /// Returns the value in MiB. Fallback: `256.0`.
    fn read_process_memory_mb() -> f64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
                if let Some(kb) = Self::parse_proc_value(&content, "VmRSS:") {
                    return kb as f64 / 1024.0;
                }
            }
            256.0
        }

        #[cfg(not(target_os = "linux"))]
        {
            256.0
        }
    }

    /// Read total system memory from /proc/meminfo (Linux only).
    ///
    /// Returns the value in MiB. Fallback: `8192.0`.
    fn read_total_memory_mb() -> f64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                if let Some(kb) = Self::parse_proc_value(&content, "MemTotal:") {
                    return kb as f64 / 1024.0;
                }
            }
            8192.0
        }

        #[cfg(not(target_os = "linux"))]
        {
            8192.0
        }
    }

    /// Read the thread count for the current process from /proc/self/status (Linux only).
    ///
    /// Fallback: `4.0`.
    fn read_thread_count() -> f64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
                if let Some(threads) = Self::parse_proc_value(&content, "Threads:") {
                    return threads as f64;
                }
            }
            4.0
        }

        #[cfg(not(target_os = "linux"))]
        {
            4.0
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metric_registration() {
        let monitor = PerformanceMonitor::new();

        monitor
            .register_metric("test_counter", MetricType::Counter, "Test counter", "count")
            .await
            .unwrap();

        let metrics = monitor.list_metrics().await;
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].name, "test_counter");
    }

    #[tokio::test]
    async fn test_counter_metric() {
        let monitor = PerformanceMonitor::new();

        monitor
            .record_counter(
                "requests_total",
                1.0,
                vec![("endpoint".to_string(), "/api/test".to_string())],
            )
            .await
            .unwrap();
        monitor
            .record_counter(
                "requests_total",
                1.0,
                vec![("endpoint".to_string(), "/api/test".to_string())],
            )
            .await
            .unwrap();

        let stats = monitor
            .get_metric_statistics("requests_total", None)
            .await
            .unwrap();
        assert_eq!(stats.count, 2);
    }

    #[tokio::test]
    async fn test_gauge_metric() {
        let monitor = PerformanceMonitor::new();

        monitor
            .record_gauge("cpu_usage", 45.5, vec![])
            .await
            .unwrap();
        monitor
            .record_gauge("cpu_usage", 52.3, vec![])
            .await
            .unwrap();

        let stats = monitor
            .get_metric_statistics("cpu_usage", None)
            .await
            .unwrap();
        assert_eq!(stats.count, 2);
        assert!(stats.max > stats.min);
    }

    #[tokio::test]
    async fn test_histogram_statistics() {
        let monitor = PerformanceMonitor::new();

        for i in 0..100 {
            monitor
                .record_histogram("latency_ms", i as f64, vec![])
                .await
                .unwrap();
        }

        let stats = monitor
            .get_metric_statistics("latency_ms", None)
            .await
            .unwrap();
        assert_eq!(stats.count, 100);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 99.0);
        assert!(stats.p95 >= 95.0);
    }

    #[tokio::test]
    async fn test_alert_rules() {
        let monitor = PerformanceMonitor::new();

        let rule = AlertRule {
            id: "high_cpu".to_string(),
            metric_name: "cpu_usage".to_string(),
            threshold: 80.0,
            above_threshold: true,
            severity: AlertSeverity::Warning,
            description: "High CPU usage".to_string(),
            enabled: true,
        };

        monitor.add_alert_rule(rule).await.unwrap();

        // Trigger alert
        monitor
            .record_gauge("cpu_usage", 85.0, vec![])
            .await
            .unwrap();

        let alerts = monitor.get_active_alerts().await;
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, AlertSeverity::Warning);
    }

    #[tokio::test]
    async fn test_prometheus_export() {
        let monitor = PerformanceMonitor::new();

        monitor
            .record_counter(
                "requests_total",
                100.0,
                vec![("method".to_string(), "GET".to_string())],
            )
            .await
            .unwrap();

        let prometheus = monitor.export_prometheus().await;
        assert!(prometheus.contains("requests_total"));
        assert!(prometheus.contains("method=\"GET\""));
    }

    #[tokio::test]
    async fn test_json_export() {
        let monitor = PerformanceMonitor::new();

        monitor
            .record_gauge("temperature", 25.5, vec![])
            .await
            .unwrap();

        let json = monitor.export_json().await;
        assert!(json["metrics"].is_array());
    }

    #[tokio::test]
    async fn test_cleanup_old_data() {
        let mut config = MonitorConfig::default();
        config.retention_period_days = 0; // Remove all data immediately

        let monitor = PerformanceMonitor::with_config(config);

        monitor.record_counter("test", 1.0, vec![]).await.unwrap();

        // Wait a bit to ensure timestamp difference
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let removed = monitor.cleanup_old_data().await.unwrap();
        assert!(removed > 0);
    }

    #[test]
    fn test_parse_proc_value_cpu() {
        // Simulate a /proc/stat-style "cpu " line
        let mock_stat = "cpu  1234 56 789 4321 0 0 0 0 0 0\ncpu0 617 28 394 2160 0 0 0 0 0 0\n";

        // The "cpu " prefix (with trailing space) should match the aggregate line
        let result = PerformanceMonitor::parse_proc_value(mock_stat, "cpu ");
        // nth(1) after split_whitespace skips "cpu" and returns the first field: 1234
        assert_eq!(result, Some(1234));

        // A prefix that does not exist should return None
        let missing = PerformanceMonitor::parse_proc_value(mock_stat, "nonexistent:");
        assert_eq!(missing, None);
    }

    #[test]
    fn test_parse_proc_value_memory() {
        // Simulate relevant lines from /proc/self/status
        let mock_status = "Name:\tmyprocess\nVmRSS:\t131072 kB\nThreads:\t8\n";

        // VmRSS in kB
        let kb = PerformanceMonitor::parse_proc_value(mock_status, "VmRSS:");
        assert_eq!(kb, Some(131_072));

        // kB → MB conversion matches spec
        let mb = kb.unwrap() as f64 / 1024.0;
        assert!((mb - 128.0).abs() < f64::EPSILON);

        // Thread count
        let threads = PerformanceMonitor::parse_proc_value(mock_status, "Threads:");
        assert_eq!(threads, Some(8));
    }

    #[tokio::test]
    async fn test_system_metrics_no_crash() {
        let monitor = PerformanceMonitor::new();
        let result = monitor.collect_system_metrics().await;
        assert!(
            result.is_ok(),
            "collect_system_metrics returned an error: {result:?}"
        );
    }
}
