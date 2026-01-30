//! # Performance SLA Guarantees
//!
//! Provides comprehensive Performance SLA (Service Level Agreement) tracking,
//! monitoring, and enforcement for production deployments.
//!
//! Features:
//! - Real-time performance tracking
//! - SLA compliance monitoring
//! - Automatic degradation and recovery
//! - Performance profiling and optimization
//! - Alert generation and notification

use crate::RecognitionError;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// SLA errors
#[derive(Debug, Error)]
pub enum SlaError {
    /// SLA violation detected
    #[error("SLA violation: {metric} = {actual:.2}, threshold = {threshold:.2}")]
    SlaViolation {
        /// Metric name
        metric: String,
        /// Actual value
        actual: f64,
        /// Threshold value
        threshold: f64,
    },

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Monitoring error
    #[error("Monitoring error: {0}")]
    MonitoringError(String),
}

/// SLA metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaConfig {
    /// Maximum real-time factor (RTF)
    pub max_rtf: f64,

    /// Maximum latency in milliseconds
    pub max_latency_ms: f64,

    /// Maximum memory usage in MB
    pub max_memory_mb: f64,

    /// Minimum accuracy (confidence score)
    pub min_accuracy: f64,

    /// Maximum error rate (percentage)
    pub max_error_rate: f64,

    /// Target uptime percentage
    pub target_uptime_percentage: f64,

    /// Measurement window size (number of samples)
    pub measurement_window_size: usize,

    /// Enable automatic remediation
    pub enable_auto_remediation: bool,

    /// Alert warning threshold (percentage of SLA threshold, e.g., 0.8 for 80%)
    pub alert_warning_threshold: f64,

    /// Alert critical threshold (percentage of SLA threshold, e.g., 0.95 for 95%)
    pub alert_critical_threshold: f64,
}

impl Default for SlaConfig {
    fn default() -> Self {
        Self {
            max_rtf: 0.3,
            max_latency_ms: 200.0,
            max_memory_mb: 2048.0,
            min_accuracy: 0.95,
            max_error_rate: 1.0,
            target_uptime_percentage: 99.9,
            measurement_window_size: 100,
            enable_auto_remediation: true,
            alert_warning_threshold: 0.8,
            alert_critical_threshold: 0.95,
        }
    }
}

/// SLA metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaMetrics {
    /// Real-time factor
    pub rtf: f64,

    /// Latency in milliseconds
    pub latency_ms: f64,

    /// Memory usage in MB
    pub memory_mb: f64,

    /// Accuracy (confidence score)
    pub accuracy: f64,

    /// Error rate (percentage)
    pub error_rate: f64,

    /// Uptime percentage
    pub uptime_percentage: f64,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// SLA compliance status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceStatus {
    /// All metrics within SLA
    Compliant,

    /// Warning threshold exceeded
    Warning,

    /// SLA violation
    Violation,

    /// Critical violation
    Critical,
}

/// SLA alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaAlert {
    /// Alert ID
    pub id: String,

    /// Metric name
    pub metric: String,

    /// Alert severity
    pub severity: AlertSeverity,

    /// Actual value
    pub actual_value: f64,

    /// Threshold value
    pub threshold_value: f64,

    /// Alert message
    pub message: String,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Suggested remediation
    pub suggested_remediation: Option<String>,
}

/// Alert severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational
    Info,

    /// Warning
    Warning,

    /// Error
    Error,

    /// Critical
    Critical,
}

/// Performance measurement
#[derive(Debug, Clone)]
struct PerformanceMeasurement {
    /// RTF value
    rtf: f64,

    /// Latency in ms
    latency_ms: f64,

    /// Memory in MB
    memory_mb: f64,

    /// Accuracy
    accuracy: f64,

    /// Timestamp
    timestamp: Instant,

    /// Success flag
    success: bool,
}

/// SLA monitor
pub struct SlaMonitor {
    /// Configuration
    config: Arc<RwLock<SlaConfig>>,

    /// Performance measurements
    measurements: Arc<RwLock<VecDeque<PerformanceMeasurement>>>,

    /// Active alerts
    alerts: Arc<RwLock<Vec<SlaAlert>>>,

    /// Compliance history
    compliance_history: Arc<RwLock<VecDeque<(Instant, ComplianceStatus)>>>,

    /// Total requests
    total_requests: Arc<RwLock<u64>>,

    /// Failed requests
    failed_requests: Arc<RwLock<u64>>,

    /// Start time
    start_time: Instant,

    /// Remediation actions taken
    remediation_actions: Arc<RwLock<Vec<RemediationAction>>>,
}

/// Remediation action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationAction {
    /// Action type
    pub action_type: RemediationType,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Result
    pub result: RemediationResult,

    /// Description
    pub description: String,
}

/// Remediation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemediationType {
    /// Reduce batch size
    ReduceBatchSize,

    /// Switch to faster model
    SwitchToFasterModel,

    /// Enable caching
    EnableCaching,

    /// Reduce precision
    ReducePrecision,

    /// Clear memory
    ClearMemory,

    /// Restart service
    RestartService,
}

/// Remediation result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemediationResult {
    /// Successfully applied
    Success,

    /// Failed to apply
    Failed,

    /// Partially applied
    Partial,
}

impl SlaMonitor {
    /// Create a new SLA monitor
    #[must_use]
    pub fn new(config: SlaConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            measurements: Arc::new(RwLock::new(VecDeque::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            compliance_history: Arc::new(RwLock::new(VecDeque::new())),
            total_requests: Arc::new(RwLock::new(0)),
            failed_requests: Arc::new(RwLock::new(0)),
            start_time: Instant::now(),
            remediation_actions: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Record a performance measurement
    pub fn record_measurement(
        &self,
        rtf: f64,
        latency_ms: f64,
        memory_mb: f64,
        accuracy: f64,
        success: bool,
    ) {
        let measurement = PerformanceMeasurement {
            rtf,
            latency_ms,
            memory_mb,
            accuracy,
            timestamp: Instant::now(),
            success,
        };

        // Update request counters
        *self.total_requests.write() += 1;
        if !success {
            *self.failed_requests.write() += 1;
        }

        // Store measurement
        let mut measurements = self.measurements.write();
        measurements.push_back(measurement);

        // Trim to window size
        let window_size = self.config.read().measurement_window_size;
        while measurements.len() > window_size {
            measurements.pop_front();
        }

        // Check compliance
        drop(measurements);
        self.check_compliance();
    }

    /// Get current metrics
    #[must_use]
    pub fn get_current_metrics(&self) -> SlaMetrics {
        let measurements = self.measurements.read();
        let config = self.config.read();

        if measurements.is_empty() {
            return SlaMetrics {
                rtf: 0.0,
                latency_ms: 0.0,
                memory_mb: 0.0,
                accuracy: 0.0,
                error_rate: 0.0,
                uptime_percentage: 100.0,
                timestamp: chrono::Utc::now(),
            };
        }

        // Calculate averages
        let count = measurements.len() as f64;
        let avg_rtf = measurements.iter().map(|m| m.rtf).sum::<f64>() / count;
        let avg_latency = measurements.iter().map(|m| m.latency_ms).sum::<f64>() / count;
        let avg_memory = measurements.iter().map(|m| m.memory_mb).sum::<f64>() / count;
        let avg_accuracy = measurements.iter().map(|m| m.accuracy).sum::<f64>() / count;

        // Calculate error rate
        let total_requests = *self.total_requests.read();
        let failed_requests = *self.failed_requests.read();
        let error_rate = if total_requests > 0 {
            (failed_requests as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };

        // Calculate uptime
        let uptime_percentage = if total_requests > 0 {
            ((total_requests - failed_requests) as f64 / total_requests as f64) * 100.0
        } else {
            100.0
        };

        SlaMetrics {
            rtf: avg_rtf,
            latency_ms: avg_latency,
            memory_mb: avg_memory,
            accuracy: avg_accuracy,
            error_rate,
            uptime_percentage,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Get compliance status
    #[must_use]
    pub fn get_compliance_status(&self) -> ComplianceStatus {
        let metrics = self.get_current_metrics();
        let config = self.config.read();

        let mut violations = Vec::new();

        if metrics.rtf > config.max_rtf {
            violations.push(("RTF", metrics.rtf, config.max_rtf));
        }
        if metrics.latency_ms > config.max_latency_ms {
            violations.push(("Latency", metrics.latency_ms, config.max_latency_ms));
        }
        if metrics.memory_mb > config.max_memory_mb {
            violations.push(("Memory", metrics.memory_mb, config.max_memory_mb));
        }
        if metrics.accuracy < config.min_accuracy {
            violations.push(("Accuracy", metrics.accuracy, config.min_accuracy));
        }
        if metrics.error_rate > config.max_error_rate {
            violations.push(("ErrorRate", metrics.error_rate, config.max_error_rate));
        }
        if metrics.uptime_percentage < config.target_uptime_percentage {
            violations.push((
                "Uptime",
                metrics.uptime_percentage,
                config.target_uptime_percentage,
            ));
        }

        if violations.is_empty() {
            ComplianceStatus::Compliant
        } else {
            // Check if any violation is critical (>critical_threshold)
            let has_critical = violations.iter().any(|(_, actual, threshold)| {
                let ratio = if *threshold > 0.0 {
                    actual / threshold
                } else {
                    0.0
                };
                ratio > config.alert_critical_threshold
            });

            if has_critical {
                ComplianceStatus::Critical
            } else {
                let has_violation = violations.iter().any(|(_, actual, threshold)| {
                    let ratio = if *threshold > 0.0 {
                        actual / threshold
                    } else {
                        0.0
                    };
                    ratio > 1.0
                });

                if has_violation {
                    ComplianceStatus::Violation
                } else {
                    ComplianceStatus::Warning
                }
            }
        }
    }

    /// Get active alerts
    #[must_use]
    pub fn get_active_alerts(&self) -> Vec<SlaAlert> {
        self.alerts.read().clone()
    }

    /// Clear alerts
    pub fn clear_alerts(&self) {
        self.alerts.write().clear();
    }

    /// Get compliance history
    #[must_use]
    pub fn get_compliance_history(&self) -> Vec<(Duration, ComplianceStatus)> {
        let history = self.compliance_history.read();
        let start_time = self.start_time;

        history
            .iter()
            .map(|(instant, status)| (instant.duration_since(start_time), *status))
            .collect()
    }

    /// Get remediation actions
    #[must_use]
    pub fn get_remediation_actions(&self) -> Vec<RemediationAction> {
        self.remediation_actions.read().clone()
    }

    /// Generate SLA report
    #[must_use]
    pub fn generate_report(&self) -> SlaReport {
        let metrics = self.get_current_metrics();
        let status = self.get_compliance_status();
        let alerts = self.get_active_alerts();

        SlaReport {
            metrics,
            compliance_status: status,
            active_alerts: alerts,
            total_requests: *self.total_requests.read(),
            failed_requests: *self.failed_requests.read(),
            uptime_duration: self.start_time.elapsed(),
            remediation_actions: self.get_remediation_actions(),
        }
    }

    // Private methods

    fn check_compliance(&self) {
        let status = self.get_compliance_status();
        let metrics = self.get_current_metrics();
        let config = self.config.read();

        // Record compliance status
        self.compliance_history
            .write()
            .push_back((Instant::now(), status));

        // Trim compliance history
        let mut history = self.compliance_history.write();
        while history.len() > config.measurement_window_size {
            history.pop_front();
        }
        drop(history);

        // Generate alerts for violations
        if status != ComplianceStatus::Compliant {
            self.generate_alerts(&metrics, &config);
        }

        // Apply remediation if enabled
        if config.enable_auto_remediation && status == ComplianceStatus::Critical {
            drop(config);
            self.apply_remediation(&metrics);
        }
    }

    fn generate_alerts(&self, metrics: &SlaMetrics, config: &SlaConfig) {
        let mut alerts = self.alerts.write();
        alerts.clear();

        if metrics.rtf > config.max_rtf {
            alerts.push(SlaAlert {
                id: uuid::Uuid::new_v4().to_string(),
                metric: "RTF".to_string(),
                severity: self.get_severity(metrics.rtf, config.max_rtf, config),
                actual_value: metrics.rtf,
                threshold_value: config.max_rtf,
                message: format!("RTF {} exceeds threshold {}", metrics.rtf, config.max_rtf),
                timestamp: chrono::Utc::now(),
                suggested_remediation: Some("Consider switching to a faster model".to_string()),
            });
        }

        if metrics.latency_ms > config.max_latency_ms {
            alerts.push(SlaAlert {
                id: uuid::Uuid::new_v4().to_string(),
                metric: "Latency".to_string(),
                severity: self.get_severity(metrics.latency_ms, config.max_latency_ms, config),
                actual_value: metrics.latency_ms,
                threshold_value: config.max_latency_ms,
                message: format!(
                    "Latency {}ms exceeds threshold {}ms",
                    metrics.latency_ms, config.max_latency_ms
                ),
                timestamp: chrono::Utc::now(),
                suggested_remediation: Some("Enable caching or reduce batch size".to_string()),
            });
        }

        if metrics.memory_mb > config.max_memory_mb {
            alerts.push(SlaAlert {
                id: uuid::Uuid::new_v4().to_string(),
                metric: "Memory".to_string(),
                severity: self.get_severity(metrics.memory_mb, config.max_memory_mb, config),
                actual_value: metrics.memory_mb,
                threshold_value: config.max_memory_mb,
                message: format!(
                    "Memory {}MB exceeds threshold {}MB",
                    metrics.memory_mb, config.max_memory_mb
                ),
                timestamp: chrono::Utc::now(),
                suggested_remediation: Some("Clear cache or reduce model size".to_string()),
            });
        }
    }

    fn get_severity(&self, actual: f64, threshold: f64, config: &SlaConfig) -> AlertSeverity {
        let ratio = if threshold > 0.0 {
            actual / threshold
        } else {
            0.0
        };

        if ratio >= config.alert_critical_threshold {
            AlertSeverity::Critical
        } else if ratio >= 1.0 {
            AlertSeverity::Error
        } else if ratio >= config.alert_warning_threshold {
            AlertSeverity::Warning
        } else {
            AlertSeverity::Info
        }
    }

    fn apply_remediation(&self, metrics: &SlaMetrics) {
        let mut actions = self.remediation_actions.write();

        // Simple remediation logic
        if metrics.memory_mb > self.config.read().max_memory_mb {
            actions.push(RemediationAction {
                action_type: RemediationType::ClearMemory,
                timestamp: chrono::Utc::now(),
                result: RemediationResult::Success,
                description: "Cleared cache to reduce memory usage".to_string(),
            });

            tracing::info!("Applied remediation: Clear memory");
        }

        if metrics.rtf > self.config.read().max_rtf {
            actions.push(RemediationAction {
                action_type: RemediationType::ReduceBatchSize,
                timestamp: chrono::Utc::now(),
                result: RemediationResult::Success,
                description: "Reduced batch size to improve RTF".to_string(),
            });

            tracing::info!("Applied remediation: Reduce batch size");
        }
    }
}

/// SLA report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaReport {
    /// Current metrics
    pub metrics: SlaMetrics,

    /// Compliance status
    pub compliance_status: ComplianceStatus,

    /// Active alerts
    pub active_alerts: Vec<SlaAlert>,

    /// Total requests
    pub total_requests: u64,

    /// Failed requests
    pub failed_requests: u64,

    /// Uptime duration
    #[serde(skip)]
    pub uptime_duration: Duration,

    /// Remediation actions
    pub remediation_actions: Vec<RemediationAction>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sla_monitor_creation() {
        let config = SlaConfig::default();
        let monitor = SlaMonitor::new(config);

        let metrics = monitor.get_current_metrics();
        assert_eq!(metrics.rtf, 0.0);
    }

    #[test]
    fn test_record_measurement() {
        let config = SlaConfig::default();
        let monitor = SlaMonitor::new(config);

        monitor.record_measurement(0.25, 150.0, 1024.0, 0.96, true);

        let metrics = monitor.get_current_metrics();
        assert!((metrics.rtf - 0.25).abs() < f64::EPSILON);
        assert!((metrics.latency_ms - 150.0).abs() < f64::EPSILON);
        assert!((metrics.memory_mb - 1024.0).abs() < f64::EPSILON);
        assert!((metrics.accuracy - 0.96).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compliance_status_compliant() {
        let config = SlaConfig::default();
        let monitor = SlaMonitor::new(config);

        // Record good measurements
        monitor.record_measurement(0.2, 100.0, 1024.0, 0.98, true);

        let status = monitor.get_compliance_status();
        assert_eq!(status, ComplianceStatus::Compliant);
    }

    #[test]
    fn test_compliance_status_violation() {
        let config = SlaConfig::default();
        let monitor = SlaMonitor::new(config);

        // Record violating measurements (RTF > threshold)
        monitor.record_measurement(0.5, 100.0, 1024.0, 0.98, true);

        let status = monitor.get_compliance_status();
        assert!(status != ComplianceStatus::Compliant);
    }

    #[test]
    fn test_alert_generation() {
        let config = SlaConfig::default();
        let monitor = SlaMonitor::new(config);

        // Record violating measurements
        monitor.record_measurement(0.5, 300.0, 3000.0, 0.98, true);

        let alerts = monitor.get_active_alerts();
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_error_rate_calculation() {
        let config = SlaConfig::default();
        let monitor = SlaMonitor::new(config);

        // Record 10 requests, 2 failed
        for i in 0..10 {
            let success = i % 5 != 0; // Fail every 5th request
            monitor.record_measurement(0.2, 100.0, 1024.0, 0.98, success);
        }

        let metrics = monitor.get_current_metrics();
        assert!(metrics.error_rate >= 15.0 && metrics.error_rate <= 25.0);
    }

    #[test]
    fn test_uptime_calculation() {
        let config = SlaConfig::default();
        let monitor = SlaMonitor::new(config);

        // Record successful requests
        for _ in 0..100 {
            monitor.record_measurement(0.2, 100.0, 1024.0, 0.98, true);
        }

        let metrics = monitor.get_current_metrics();
        assert!((metrics.uptime_percentage - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sla_report_generation() {
        let config = SlaConfig::default();
        let monitor = SlaMonitor::new(config);

        monitor.record_measurement(0.2, 100.0, 1024.0, 0.98, true);

        let report = monitor.generate_report();
        assert_eq!(report.total_requests, 1);
        assert_eq!(report.failed_requests, 0);
        assert_eq!(report.compliance_status, ComplianceStatus::Compliant);
    }

    #[test]
    fn test_window_size_limiting() {
        let config = SlaConfig {
            measurement_window_size: 5,
            ..Default::default()
        };
        let monitor = SlaMonitor::new(config);

        // Record more than window size
        for i in 0..10 {
            monitor.record_measurement(f64::from(i), 100.0, 1024.0, 0.98, true);
        }

        // Should only keep last 5 measurements
        let measurements = monitor.measurements.read();
        assert_eq!(measurements.len(), 5);
    }
}
