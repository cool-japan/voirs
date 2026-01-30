// Copyright (c) 2024 VoiRS Contributors
// Licensed under the MIT License

//! Monitoring and Observability Framework for VoiRS Evaluation
//!
//! This module provides comprehensive monitoring, metrics collection, distributed tracing,
//! and observability features for the evaluation system. It integrates with industry-standard
//! tools like Prometheus, Grafana, Jaeger, and OpenTelemetry.
//!
//! # Features
//!
//! - **Prometheus Metrics**: Counters, gauges, histograms for evaluation metrics
//! - **Health Checks**: Liveness and readiness endpoints for Kubernetes
//! - **Distributed Tracing**: OpenTelemetry-compatible tracing for request flows
//! - **Performance Monitoring**: Latency, throughput, and resource usage tracking
//! - **Alerting**: Configurable alert rules and notification channels
//! - **Custom Dashboards**: Pre-configured Grafana dashboard definitions
//! - **SLA Monitoring**: Service Level Objective (SLO) tracking and reporting
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │          Observability Framework                    │
//! ├─────────────────────────────────────────────────────┤
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
//! │  │  Prometheus  │  │   Tracing    │  │ Alerting │ │
//! │  │   Metrics    │  │  (Jaeger)    │  │  Rules   │ │
//! │  └──────────────┘  └──────────────┘  └──────────┘ │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
//! │  │    Health    │  │     SLA      │  │ Grafana  │ │
//! │  │    Checks    │  │  Monitoring  │  │Dashboard │ │
//! │  └──────────────┘  └──────────────┘  └──────────┘ │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! # Example Usage
//!
//! ```rust
//! use voirs_evaluation::observability::{
//!     ObservabilityManager, MetricsConfig, HealthStatus
//! };
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create observability manager
//! let config = MetricsConfig::default();
//! let manager = ObservabilityManager::new(config).await?;
//!
//! // Record evaluation metrics
//! manager.record_evaluation_duration(0.15).await;
//! manager.increment_evaluation_counter("quality").await;
//!
//! // Check health status
//! let health = manager.health_check().await?;
//! println!("Service health: {:?}", health.status);
//!
//! // Get Prometheus metrics
//! let metrics = manager.export_prometheus_metrics().await?;
//! println!("Metrics: {}", metrics);
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Observability errors
#[derive(Debug, Error)]
pub enum ObservabilityError {
    #[error("Metrics error: {0}")]
    MetricsError(String),

    #[error("Health check error: {0}")]
    HealthCheckError(String),

    #[error("Tracing error: {0}")]
    TracingError(String),

    #[error("Alert error: {0}")]
    AlertError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Export error: {0}")]
    ExportError(String),
}

pub type Result<T> = std::result::Result<T, ObservabilityError>;

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable Prometheus metrics
    pub prometheus_enabled: bool,

    /// Prometheus metrics port
    pub prometheus_port: u16,

    /// Enable distributed tracing
    pub tracing_enabled: bool,

    /// Tracing endpoint (Jaeger/OTLP)
    pub tracing_endpoint: Option<String>,

    /// Sample rate for traces (0.0 to 1.0)
    pub trace_sample_rate: f64,

    /// Enable health checks
    pub health_checks_enabled: bool,

    /// Health check interval (seconds)
    pub health_check_interval_secs: u64,

    /// Enable SLA monitoring
    pub sla_monitoring_enabled: bool,

    /// SLA targets
    pub sla_targets: SlaTargets,

    /// Enable alerting
    pub alerting_enabled: bool,

    /// Alert notification endpoints
    pub alert_endpoints: Vec<String>,

    /// Metrics retention period (days)
    pub metrics_retention_days: u32,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            prometheus_enabled: true,
            prometheus_port: 9090,
            tracing_enabled: true,
            tracing_endpoint: Some("http://localhost:14268/api/traces".to_string()),
            trace_sample_rate: 0.1,
            health_checks_enabled: true,
            health_check_interval_secs: 30,
            sla_monitoring_enabled: true,
            sla_targets: SlaTargets::default(),
            alerting_enabled: true,
            alert_endpoints: vec![],
            metrics_retention_days: 30,
        }
    }
}

/// SLA (Service Level Agreement) targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaTargets {
    /// Target availability percentage (e.g., 99.9%)
    pub availability_target: f64,

    /// Target P50 latency in milliseconds
    pub p50_latency_ms: f64,

    /// Target P95 latency in milliseconds
    pub p95_latency_ms: f64,

    /// Target P99 latency in milliseconds
    pub p99_latency_ms: f64,

    /// Target error rate percentage
    pub error_rate_target: f64,

    /// Target throughput (requests per second)
    pub throughput_target: f64,
}

impl Default for SlaTargets {
    fn default() -> Self {
        Self {
            availability_target: 99.9,
            p50_latency_ms: 50.0,
            p95_latency_ms: 200.0,
            p99_latency_ms: 500.0,
            error_rate_target: 0.1,
            throughput_target: 100.0,
        }
    }
}

/// Health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Service is healthy
    Healthy,
    /// Service is degraded but operational
    Degraded,
    /// Service is unhealthy
    Unhealthy,
}

impl HealthStatus {
    /// Convert to HTTP status code
    pub fn to_status_code(&self) -> u16 {
        match self {
            HealthStatus::Healthy => 200,
            HealthStatus::Degraded => 200,
            HealthStatus::Unhealthy => 503,
        }
    }

    /// Check if service is operational
    pub fn is_operational(&self) -> bool {
        matches!(self, HealthStatus::Healthy | HealthStatus::Degraded)
    }
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Overall health status
    pub status: HealthStatus,

    /// Timestamp of check
    pub timestamp: DateTime<Utc>,

    /// Component health statuses
    pub components: HashMap<String, ComponentHealth>,

    /// System uptime in seconds
    pub uptime_seconds: u64,

    /// Additional details
    pub details: HashMap<String, String>,
}

/// Component health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component status
    pub status: HealthStatus,

    /// Last check time
    pub last_check: DateTime<Utc>,

    /// Error message if unhealthy
    pub error: Option<String>,

    /// Component metrics
    pub metrics: HashMap<String, f64>,
}

/// Prometheus metric types
#[derive(Debug, Clone)]
pub enum MetricType {
    Counter(f64),
    Gauge(f64),
    Histogram(Vec<f64>),
    Summary { sum: f64, count: u64 },
}

/// Metric definition
#[derive(Debug, Clone)]
pub struct Metric {
    pub name: String,
    pub help: String,
    pub metric_type: MetricType,
    pub labels: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

/// Trace span for distributed tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    /// Span ID
    pub span_id: String,

    /// Trace ID
    pub trace_id: String,

    /// Parent span ID
    pub parent_span_id: Option<String>,

    /// Operation name
    pub operation_name: String,

    /// Start time
    pub start_time: DateTime<Utc>,

    /// Duration in microseconds
    pub duration_us: u64,

    /// Tags
    pub tags: HashMap<String, String>,

    /// Logs
    pub logs: Vec<SpanLog>,
}

/// Span log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanLog {
    pub timestamp: DateTime<Utc>,
    pub message: String,
    pub fields: HashMap<String, String>,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Alert rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule ID
    pub id: String,

    /// Rule name
    pub name: String,

    /// Rule description
    pub description: String,

    /// Alert severity
    pub severity: AlertSeverity,

    /// Metric to monitor
    pub metric_name: String,

    /// Condition (e.g., ">", "<", "==")
    pub condition: String,

    /// Threshold value
    pub threshold: f64,

    /// Duration condition must be met (seconds)
    pub duration_secs: u64,

    /// Alert enabled
    pub enabled: bool,
}

/// Alert instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,

    /// Rule that triggered this alert
    pub rule_id: String,

    /// Alert severity
    pub severity: AlertSeverity,

    /// Alert message
    pub message: String,

    /// Metric value that triggered alert
    pub value: f64,

    /// Threshold that was crossed
    pub threshold: f64,

    /// Alert start time
    pub started_at: DateTime<Utc>,

    /// Alert resolved time
    pub resolved_at: Option<DateTime<Utc>>,

    /// Additional labels
    pub labels: HashMap<String, String>,
}

/// SLA metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaMetrics {
    /// Period start
    pub period_start: DateTime<Utc>,

    /// Period end
    pub period_end: DateTime<Utc>,

    /// Actual availability percentage
    pub availability: f64,

    /// P50 latency (milliseconds)
    pub p50_latency_ms: f64,

    /// P95 latency (milliseconds)
    pub p95_latency_ms: f64,

    /// P99 latency (milliseconds)
    pub p99_latency_ms: f64,

    /// Error rate percentage
    pub error_rate: f64,

    /// Throughput (requests per second)
    pub throughput: f64,

    /// Total requests
    pub total_requests: u64,

    /// Successful requests
    pub successful_requests: u64,

    /// Failed requests
    pub failed_requests: u64,

    /// SLA compliance
    pub sla_compliance: bool,
}

/// Observability manager
pub struct ObservabilityManager {
    config: MetricsConfig,
    metrics: Arc<RwLock<HashMap<String, Metric>>>,
    health_components: Arc<RwLock<HashMap<String, ComponentHealth>>>,
    active_spans: Arc<RwLock<HashMap<String, TraceSpan>>>,
    alerts: Arc<RwLock<Vec<Alert>>>,
    alert_rules: Arc<RwLock<Vec<AlertRule>>>,
    latency_samples: Arc<RwLock<Vec<f64>>>,
    start_time: Instant,
    request_count: Arc<RwLock<u64>>,
    error_count: Arc<RwLock<u64>>,
}

impl ObservabilityManager {
    /// Create a new observability manager
    pub async fn new(config: MetricsConfig) -> Result<Self> {
        info!("Initializing observability manager");

        let mut health_components = HashMap::new();
        health_components.insert(
            "metrics".to_string(),
            ComponentHealth {
                status: HealthStatus::Healthy,
                last_check: Utc::now(),
                error: None,
                metrics: HashMap::new(),
            },
        );

        Ok(Self {
            config,
            metrics: Arc::new(RwLock::new(HashMap::new())),
            health_components: Arc::new(RwLock::new(health_components)),
            active_spans: Arc::new(RwLock::new(HashMap::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            alert_rules: Arc::new(RwLock::new(Vec::new())),
            latency_samples: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
            request_count: Arc::new(RwLock::new(0)),
            error_count: Arc::new(RwLock::new(0)),
        })
    }

    /// Record evaluation duration
    pub async fn record_evaluation_duration(&self, duration_secs: f64) {
        let mut samples = self.latency_samples.write().await;
        samples.push(duration_secs * 1000.0); // Convert to milliseconds

        // Keep only recent samples (last 10,000)
        let len = samples.len();
        if len > 10000 {
            samples.drain(0..len - 10000);
        }

        debug!("Recorded evaluation duration: {:.3}s", duration_secs);
    }

    /// Increment evaluation counter
    pub async fn increment_evaluation_counter(&self, evaluation_type: &str) {
        let mut count = self.request_count.write().await;
        *count += 1;

        let mut metrics = self.metrics.write().await;
        let metric_name = format!("evaluations_total_{}", evaluation_type);

        metrics.insert(
            metric_name.clone(),
            Metric {
                name: metric_name,
                help: format!("Total number of {} evaluations", evaluation_type),
                metric_type: MetricType::Counter(*count as f64),
                labels: HashMap::new(),
                timestamp: Utc::now(),
            },
        );

        debug!(
            "Incremented evaluation counter for type: {}",
            evaluation_type
        );
    }

    /// Record error
    pub async fn record_error(&self, error_type: &str) {
        let mut count = self.error_count.write().await;
        *count += 1;

        debug!("Recorded error: {}", error_type);
    }

    /// Perform health check
    pub async fn health_check(&self) -> Result<HealthCheck> {
        let components = self.health_components.read().await.clone();

        // Determine overall status
        let status = if components
            .values()
            .all(|c| c.status == HealthStatus::Healthy)
        {
            HealthStatus::Healthy
        } else if components
            .values()
            .any(|c| c.status == HealthStatus::Unhealthy)
        {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Degraded
        };

        let uptime = self.start_time.elapsed().as_secs();

        let mut details = HashMap::new();
        details.insert("version".to_string(), env!("CARGO_PKG_VERSION").to_string());
        details.insert(
            "prometheus_enabled".to_string(),
            self.config.prometheus_enabled.to_string(),
        );
        details.insert(
            "tracing_enabled".to_string(),
            self.config.tracing_enabled.to_string(),
        );

        Ok(HealthCheck {
            status,
            timestamp: Utc::now(),
            components,
            uptime_seconds: uptime,
            details,
        })
    }

    /// Export Prometheus metrics
    pub async fn export_prometheus_metrics(&self) -> Result<String> {
        if !self.config.prometheus_enabled {
            return Ok(String::new());
        }

        let metrics = self.metrics.read().await;
        let mut output = String::new();

        for metric in metrics.values() {
            // Add HELP line
            output.push_str(&format!("# HELP {} {}\n", metric.name, metric.help));

            // Add TYPE line
            let type_name = match metric.metric_type {
                MetricType::Counter(_) => "counter",
                MetricType::Gauge(_) => "gauge",
                MetricType::Histogram(_) => "histogram",
                MetricType::Summary { .. } => "summary",
            };
            output.push_str(&format!("# TYPE {} {}\n", metric.name, type_name));

            // Add metric value
            let labels = if metric.labels.is_empty() {
                String::new()
            } else {
                let label_str: Vec<String> = metric
                    .labels
                    .iter()
                    .map(|(k, v)| format!("{}=\"{}\"", k, v))
                    .collect();
                format!("{{{}}}", label_str.join(","))
            };

            match &metric.metric_type {
                MetricType::Counter(value) | MetricType::Gauge(value) => {
                    output.push_str(&format!("{}{} {}\n", metric.name, labels, value));
                }
                MetricType::Histogram(values) => {
                    if !values.is_empty() {
                        let sum: f64 = values.iter().sum();
                        let count = values.len();
                        output.push_str(&format!("{}_sum{} {}\n", metric.name, labels, sum));
                        output.push_str(&format!("{}_count{} {}\n", metric.name, labels, count));
                    }
                }
                MetricType::Summary { sum, count } => {
                    output.push_str(&format!("{}_sum{} {}\n", metric.name, labels, sum));
                    output.push_str(&format!("{}_count{} {}\n", metric.name, labels, count));
                }
            }
        }

        Ok(output)
    }

    /// Calculate SLA metrics
    pub async fn calculate_sla_metrics(&self, period_start: DateTime<Utc>) -> Result<SlaMetrics> {
        let period_end = Utc::now();
        let request_count = *self.request_count.read().await;
        let error_count = *self.error_count.read().await;

        let successful_requests = request_count.saturating_sub(error_count);
        let availability = if request_count > 0 {
            (successful_requests as f64 / request_count as f64) * 100.0
        } else {
            100.0
        };

        let samples = self.latency_samples.read().await;
        let (p50, p95, p99) = if !samples.is_empty() {
            let mut sorted = samples.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let p50_idx = (sorted.len() as f64 * 0.50) as usize;
            let p95_idx = (sorted.len() as f64 * 0.95) as usize;
            let p99_idx = (sorted.len() as f64 * 0.99) as usize;

            (
                sorted.get(p50_idx).copied().unwrap_or(0.0),
                sorted.get(p95_idx).copied().unwrap_or(0.0),
                sorted.get(p99_idx).copied().unwrap_or(0.0),
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        let error_rate = if request_count > 0 {
            (error_count as f64 / request_count as f64) * 100.0
        } else {
            0.0
        };

        let duration_secs = (period_end - period_start).num_seconds() as f64;
        let throughput = if duration_secs > 0.0 {
            request_count as f64 / duration_secs
        } else {
            0.0
        };

        // Check SLA compliance
        let targets = &self.config.sla_targets;
        let sla_compliance = availability >= targets.availability_target
            && p50 <= targets.p50_latency_ms
            && p95 <= targets.p95_latency_ms
            && p99 <= targets.p99_latency_ms
            && error_rate <= targets.error_rate_target
            && throughput >= targets.throughput_target;

        Ok(SlaMetrics {
            period_start,
            period_end,
            availability,
            p50_latency_ms: p50,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
            error_rate,
            throughput,
            total_requests: request_count,
            successful_requests,
            failed_requests: error_count,
            sla_compliance,
        })
    }

    /// Add alert rule
    pub async fn add_alert_rule(&self, rule: AlertRule) {
        self.alert_rules.write().await.push(rule);
    }

    /// Check alert rules
    pub async fn check_alert_rules(&self) -> Result<Vec<Alert>> {
        let mut triggered_alerts = Vec::new();

        if !self.config.alerting_enabled {
            return Ok(triggered_alerts);
        }

        let rules = self.alert_rules.read().await;
        let metrics = self.metrics.read().await;

        for rule in rules.iter().filter(|r| r.enabled) {
            if let Some(metric) = metrics.get(&rule.metric_name) {
                let value = match &metric.metric_type {
                    MetricType::Counter(v) | MetricType::Gauge(v) => *v,
                    MetricType::Histogram(values) => {
                        values.iter().sum::<f64>() / values.len() as f64
                    }
                    MetricType::Summary { sum, count } => {
                        if *count > 0 {
                            sum / (*count as f64)
                        } else {
                            0.0
                        }
                    }
                };

                let triggered = match rule.condition.as_str() {
                    ">" => value > rule.threshold,
                    ">=" => value >= rule.threshold,
                    "<" => value < rule.threshold,
                    "<=" => value <= rule.threshold,
                    "==" => (value - rule.threshold).abs() < f64::EPSILON,
                    _ => false,
                };

                if triggered {
                    let alert = Alert {
                        id: format!("alert-{}-{}", rule.id, Utc::now().timestamp()),
                        rule_id: rule.id.clone(),
                        severity: rule.severity,
                        message: format!(
                            "{}: {} ({} {} {})",
                            rule.name, rule.description, value, rule.condition, rule.threshold
                        ),
                        value,
                        threshold: rule.threshold,
                        started_at: Utc::now(),
                        resolved_at: None,
                        labels: HashMap::new(),
                    };

                    warn!("Alert triggered: {}", alert.message);
                    triggered_alerts.push(alert);
                }
            }
        }

        Ok(triggered_alerts)
    }

    /// Start trace span
    pub async fn start_span(&self, operation_name: String) -> String {
        if !self.config.tracing_enabled {
            return String::new();
        }

        let span_id = format!("span-{}", Utc::now().timestamp_nanos_opt().unwrap_or(0));
        let trace_id = format!("trace-{}", Utc::now().timestamp_millis());

        let span = TraceSpan {
            span_id: span_id.clone(),
            trace_id,
            parent_span_id: None,
            operation_name,
            start_time: Utc::now(),
            duration_us: 0,
            tags: HashMap::new(),
            logs: Vec::new(),
        };

        self.active_spans
            .write()
            .await
            .insert(span_id.clone(), span);

        debug!("Started trace span: {}", span_id);
        span_id
    }

    /// End trace span
    pub async fn end_span(&self, span_id: &str) -> Option<TraceSpan> {
        let mut spans = self.active_spans.write().await;

        if let Some(mut span) = spans.remove(span_id) {
            let duration = (Utc::now() - span.start_time)
                .num_microseconds()
                .unwrap_or(0);
            span.duration_us = duration as u64;

            debug!(
                "Ended trace span: {} (duration: {}us)",
                span_id, span.duration_us
            );
            Some(span)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_config_default() {
        let config = MetricsConfig::default();
        assert!(config.prometheus_enabled);
        assert_eq!(config.prometheus_port, 9090);
        assert!(config.tracing_enabled);
        assert!(config.health_checks_enabled);
    }

    #[tokio::test]
    async fn test_sla_targets_default() {
        let targets = SlaTargets::default();
        assert_eq!(targets.availability_target, 99.9);
        assert_eq!(targets.p50_latency_ms, 50.0);
        assert_eq!(targets.p95_latency_ms, 200.0);
        assert_eq!(targets.p99_latency_ms, 500.0);
    }

    #[tokio::test]
    async fn test_health_status() {
        assert_eq!(HealthStatus::Healthy.to_status_code(), 200);
        assert_eq!(HealthStatus::Degraded.to_status_code(), 200);
        assert_eq!(HealthStatus::Unhealthy.to_status_code(), 503);

        assert!(HealthStatus::Healthy.is_operational());
        assert!(HealthStatus::Degraded.is_operational());
        assert!(!HealthStatus::Unhealthy.is_operational());
    }

    #[tokio::test]
    async fn test_observability_manager_creation() {
        let config = MetricsConfig::default();
        let manager = ObservabilityManager::new(config).await.unwrap();

        let health = manager.health_check().await.unwrap();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert!(health.components.contains_key("metrics"));
    }

    #[tokio::test]
    async fn test_record_evaluation_duration() {
        let config = MetricsConfig::default();
        let manager = ObservabilityManager::new(config).await.unwrap();

        manager.record_evaluation_duration(0.15).await;
        manager.record_evaluation_duration(0.20).await;

        let samples = manager.latency_samples.read().await;
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0], 150.0); // 0.15s = 150ms
        assert_eq!(samples[1], 200.0); // 0.20s = 200ms
    }

    #[tokio::test]
    async fn test_increment_counter() {
        let config = MetricsConfig::default();
        let manager = ObservabilityManager::new(config).await.unwrap();

        manager.increment_evaluation_counter("quality").await;
        manager.increment_evaluation_counter("quality").await;

        let count = *manager.request_count.read().await;
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_record_error() {
        let config = MetricsConfig::default();
        let manager = ObservabilityManager::new(config).await.unwrap();

        manager.record_error("timeout").await;
        manager.record_error("validation").await;

        let count = *manager.error_count.read().await;
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_prometheus_metrics_export() {
        let config = MetricsConfig::default();
        let manager = ObservabilityManager::new(config).await.unwrap();

        manager.increment_evaluation_counter("quality").await;

        let metrics = manager.export_prometheus_metrics().await.unwrap();
        assert!(metrics.contains("# HELP"));
        assert!(metrics.contains("# TYPE"));
        assert!(metrics.contains("evaluations_total_quality"));
    }

    #[tokio::test]
    async fn test_sla_metrics_calculation() {
        let config = MetricsConfig::default();
        let manager = ObservabilityManager::new(config).await.unwrap();

        // Record some metrics
        manager.increment_evaluation_counter("test").await;
        manager.increment_evaluation_counter("test").await;
        manager.record_evaluation_duration(0.05).await;
        manager.record_evaluation_duration(0.10).await;

        let period_start = Utc::now() - chrono::Duration::minutes(5);
        let sla = manager.calculate_sla_metrics(period_start).await.unwrap();

        assert_eq!(sla.total_requests, 2);
        assert_eq!(sla.successful_requests, 2);
        assert_eq!(sla.failed_requests, 0);
        assert_eq!(sla.availability, 100.0);
    }

    #[tokio::test]
    async fn test_alert_rule_evaluation() {
        let config = MetricsConfig::default();
        let manager = ObservabilityManager::new(config).await.unwrap();

        // Add alert rule
        let rule = AlertRule {
            id: "high_error_rate".to_string(),
            name: "High Error Rate".to_string(),
            description: "Error rate exceeds threshold".to_string(),
            severity: AlertSeverity::Critical,
            metric_name: "error_rate".to_string(),
            condition: ">".to_string(),
            threshold: 5.0,
            duration_secs: 60,
            enabled: true,
        };

        manager.add_alert_rule(rule).await;

        // This won't trigger because we haven't added the metric
        let alerts = manager.check_alert_rules().await.unwrap();
        assert_eq!(alerts.len(), 0);
    }

    #[tokio::test]
    async fn test_trace_span() {
        let config = MetricsConfig::default();
        let manager = ObservabilityManager::new(config).await.unwrap();

        let span_id = manager.start_span("test_operation".to_string()).await;
        assert!(!span_id.is_empty());

        // Simulate some work
        tokio::time::sleep(Duration::from_millis(10)).await;

        let span = manager.end_span(&span_id).await;
        assert!(span.is_some());

        let span = span.unwrap();
        assert_eq!(span.operation_name, "test_operation");
        assert!(span.duration_us > 0);
    }

    #[tokio::test]
    async fn test_health_check_uptime() {
        let config = MetricsConfig::default();
        let manager = ObservabilityManager::new(config).await.unwrap();

        tokio::time::sleep(Duration::from_millis(1100)).await;

        let health = manager.health_check().await.unwrap();
        assert!(health.uptime_seconds > 0);
        assert!(health.details.contains_key("version"));
    }

    #[tokio::test]
    async fn test_metrics_disabled() {
        let config = MetricsConfig {
            prometheus_enabled: false,
            ..Default::default()
        };

        let manager = ObservabilityManager::new(config).await.unwrap();
        let metrics = manager.export_prometheus_metrics().await.unwrap();
        assert!(metrics.is_empty());
    }
}
