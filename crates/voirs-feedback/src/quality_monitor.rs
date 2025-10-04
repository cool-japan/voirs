//! Automated Quality Monitoring System for VoiRS Feedback
//!
//! This module provides comprehensive automated monitoring of system quality,
//! performance metrics, user experience indicators, and reliability measures.

use crate::traits::*;
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tokio::time::{interval, sleep};

/// Quality monitoring errors
#[derive(Debug, thiserror::Error)]
pub enum QualityMonitorError {
    #[error("Quality threshold violation: {metric} is {value}, threshold: {threshold}")]
    /// Description
    ThresholdViolation {
        /// Description
        metric: String,
        /// Description
        value: f64,
        /// Description
        threshold: f64,
    },

    #[error("Monitoring system failure: {message}")]
    /// Description
    /// Description
    MonitoringFailure {
        /// Human-readable description of the monitoring failure.
        message: String,
    },

    #[error("Alert delivery failed: {message}")]
    /// Description
    /// Description
    AlertDeliveryError {
        /// Human-readable description of the alert delivery issue.
        message: String,
    },

    #[error("Invalid quality configuration: {message}")]
    /// Description
    /// Description
    InvalidConfigError {
        /// Human-readable description of the configuration problem.
        message: String,
    },
}

/// Result type for quality monitoring operations
pub type QualityResult<T> = Result<T, QualityMonitorError>;

/// Quality monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMonitorConfig {
    /// Enable automated monitoring
    pub enabled: bool,
    /// Monitoring interval in seconds
    pub monitoring_interval: u64,
    /// Quality thresholds for various metrics
    pub thresholds: QualityThresholds,
    /// Alert configuration
    pub alerts: AlertConfig,
    /// Data retention period in days
    pub retention_days: u64,
    /// Maximum samples to keep in memory
    pub max_samples: usize,
}

impl Default for QualityMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring_interval: 60, // 1 minute
            thresholds: QualityThresholds::default(),
            alerts: AlertConfig::default(),
            retention_days: 30,
            max_samples: 10000,
        }
    }
}

/// Quality thresholds for various metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Response time threshold in milliseconds
    pub response_time_ms: f64,
    /// Error rate threshold (0.0 to 1.0)
    pub error_rate: f64,
    /// User satisfaction threshold (0.0 to 5.0)
    pub satisfaction_score: f64,
    /// System availability threshold (0.0 to 1.0)
    pub availability: f64,
    /// Memory usage threshold in MB
    pub memory_usage_mb: f64,
    /// CPU usage threshold (0.0 to 1.0)
    pub cpu_usage: f64,
    /// Throughput threshold (requests per second)
    pub throughput_rps: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            response_time_ms: 2000.0, // 2 seconds
            error_rate: 0.05,         // 5% error rate
            satisfaction_score: 3.5,  // 3.5/5.0 minimum satisfaction
            availability: 0.99,       // 99% availability
            memory_usage_mb: 1024.0,  // 1GB memory limit
            cpu_usage: 0.8,           // 80% CPU usage
            throughput_rps: 100.0,    // 100 requests per second
        }
    }
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable email alerts
    pub email_enabled: bool,
    /// Enable webhook alerts
    pub webhook_enabled: bool,
    /// Enable log alerts
    pub log_enabled: bool,
    /// Alert severity levels to trigger
    pub severity_levels: Vec<AlertSeverity>,
    /// Cooldown period between alerts in minutes
    pub cooldown_minutes: u64,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            email_enabled: false,
            webhook_enabled: false,
            log_enabled: true,
            severity_levels: vec![AlertSeverity::Critical, AlertSeverity::High],
            cooldown_minutes: 15,
        }
    }
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Information only
    Info,
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical issue requiring immediate attention
    Critical,
}

/// Quality metrics collected by the monitoring system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Timestamp of measurement
    pub timestamp: DateTime<Utc>,
    /// Response time in milliseconds
    pub response_time_ms: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// User satisfaction score (0.0 to 5.0)
    pub satisfaction_score: f64,
    /// System availability (0.0 to 1.0)
    pub availability: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// CPU usage (0.0 to 1.0)
    pub cpu_usage: f64,
    /// Throughput in requests per second
    pub throughput_rps: f64,
    /// Active user sessions
    pub active_sessions: u64,
    /// Quality score (composite metric 0.0 to 100.0)
    pub overall_quality_score: f64,
}

impl QualityMetrics {
    /// Calculate overall quality score based on all metrics
    pub fn calculate_quality_score(&mut self, thresholds: &QualityThresholds) {
        let mut score = 100.0;

        // Response time impact (0-20 points)
        if self.response_time_ms > thresholds.response_time_ms {
            let penalty = ((self.response_time_ms - thresholds.response_time_ms)
                / thresholds.response_time_ms)
                .min(1.0)
                * 20.0;
            score -= penalty;
        }

        // Error rate impact (0-25 points)
        if self.error_rate > thresholds.error_rate {
            let penalty =
                ((self.error_rate - thresholds.error_rate) / thresholds.error_rate).min(1.0) * 25.0;
            score -= penalty;
        }

        // Satisfaction impact (0-20 points)
        if self.satisfaction_score < thresholds.satisfaction_score {
            let penalty = ((thresholds.satisfaction_score - self.satisfaction_score)
                / thresholds.satisfaction_score)
                .min(1.0)
                * 20.0;
            score -= penalty;
        }

        // Availability impact (0-15 points)
        if self.availability < thresholds.availability {
            let penalty = ((thresholds.availability - self.availability) / thresholds.availability)
                .min(1.0)
                * 15.0;
            score -= penalty;
        }

        // Resource usage impact (0-20 points total)
        if self.memory_usage_mb > thresholds.memory_usage_mb {
            let penalty = ((self.memory_usage_mb - thresholds.memory_usage_mb)
                / thresholds.memory_usage_mb)
                .min(1.0)
                * 10.0;
            score -= penalty;
        }

        if self.cpu_usage > thresholds.cpu_usage {
            let penalty =
                ((self.cpu_usage - thresholds.cpu_usage) / thresholds.cpu_usage).min(1.0) * 10.0;
            score -= penalty;
        }

        self.overall_quality_score = score.max(0.0);
    }
}

/// Quality alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAlert {
    /// Unique alert ID
    pub id: String,
    /// Alert timestamp
    pub timestamp: DateTime<Utc>,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert title
    pub title: String,
    /// Alert message
    pub message: String,
    /// Affected metric
    pub metric: String,
    /// Current value
    pub current_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Whether alert has been acknowledged
    pub acknowledged: bool,
    /// Resolution timestamp
    pub resolved_at: Option<DateTime<Utc>>,
}

/// Automated Quality Monitor
#[derive(Debug)]
pub struct QualityMonitor {
    /// Configuration
    config: QualityMonitorConfig,
    /// Historical metrics
    metrics_history: Arc<RwLock<VecDeque<QualityMetrics>>>,
    /// Active alerts
    active_alerts: Arc<RwLock<HashMap<String, QualityAlert>>>,
    /// Alert history
    alert_history: Arc<RwLock<VecDeque<QualityAlert>>>,
    /// Last alert times for cooldown
    last_alert_times: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    /// Monitoring task handle
    monitoring_handle: Option<tokio::task::JoinHandle<()>>,
}

impl QualityMonitor {
    /// Create new quality monitor
    pub async fn new(config: QualityMonitorConfig) -> QualityResult<Self> {
        Ok(Self {
            config,
            metrics_history: Arc::new(RwLock::new(VecDeque::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(RwLock::new(VecDeque::new())),
            last_alert_times: Arc::new(RwLock::new(HashMap::new())),
            monitoring_handle: None,
        })
    }

    /// Start automated monitoring
    pub async fn start_monitoring(&mut self) -> QualityResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let config = self.config.clone();
        let metrics_history = Arc::clone(&self.metrics_history);
        let active_alerts = Arc::clone(&self.active_alerts);
        let alert_history = Arc::clone(&self.alert_history);
        let last_alert_times = Arc::clone(&self.last_alert_times);

        let handle = tokio::spawn(async move {
            let mut interval = interval(std::time::Duration::from_secs(config.monitoring_interval));

            loop {
                interval.tick().await;

                // Collect current metrics
                let metrics = Self::collect_metrics().await;

                // Check for threshold violations
                if let Some(alert) = Self::check_thresholds(&metrics, &config.thresholds).await {
                    Self::handle_alert(
                        alert,
                        &active_alerts,
                        &alert_history,
                        &last_alert_times,
                        &config.alerts,
                    )
                    .await;
                }

                // Store metrics
                let mut history = metrics_history.write().await;
                history.push_back(metrics);

                // Maintain history size
                while history.len() > config.max_samples {
                    history.pop_front();
                }
            }
        });

        self.monitoring_handle = Some(handle);
        Ok(())
    }

    /// Stop automated monitoring
    pub async fn stop_monitoring(&mut self) -> QualityResult<()> {
        if let Some(handle) = self.monitoring_handle.take() {
            handle.abort();
        }
        Ok(())
    }

    /// Collect current system metrics
    async fn collect_metrics() -> QualityMetrics {
        let now = Utc::now();

        // In a real implementation, these would collect actual system metrics
        // For now, we'll simulate with reasonable values
        let mut metrics = QualityMetrics {
            timestamp: now,
            response_time_ms: 150.0,    // Simulated 150ms response time
            error_rate: 0.02,           // Simulated 2% error rate
            satisfaction_score: 4.2,    // Simulated 4.2/5.0 satisfaction
            availability: 0.998,        // Simulated 99.8% availability
            memory_usage_mb: 512.0,     // Simulated 512MB usage
            cpu_usage: 0.45,            // Simulated 45% CPU usage
            throughput_rps: 150.0,      // Simulated 150 RPS
            active_sessions: 25,        // Simulated 25 active sessions
            overall_quality_score: 0.0, // Will be calculated
        };

        // Calculate quality score
        let default_thresholds = QualityThresholds::default();
        metrics.calculate_quality_score(&default_thresholds);

        metrics
    }

    /// Check metrics against thresholds
    async fn check_thresholds(
        metrics: &QualityMetrics,
        thresholds: &QualityThresholds,
    ) -> Option<QualityAlert> {
        // Check response time
        if metrics.response_time_ms > thresholds.response_time_ms {
            return Some(QualityAlert {
                id: format!("response_time_{}", metrics.timestamp.timestamp()),
                timestamp: metrics.timestamp,
                severity: Self::determine_severity(
                    metrics.response_time_ms / thresholds.response_time_ms,
                ),
                title: "High Response Time".to_string(),
                message: format!(
                    "Response time ({:.1}ms) exceeds threshold ({:.1}ms)",
                    metrics.response_time_ms, thresholds.response_time_ms
                ),
                metric: "response_time_ms".to_string(),
                current_value: metrics.response_time_ms,
                threshold_value: thresholds.response_time_ms,
                acknowledged: false,
                resolved_at: None,
            });
        }

        // Check error rate
        if metrics.error_rate > thresholds.error_rate {
            return Some(QualityAlert {
                id: format!("error_rate_{}", metrics.timestamp.timestamp()),
                timestamp: metrics.timestamp,
                severity: Self::determine_severity(metrics.error_rate / thresholds.error_rate),
                title: "High Error Rate".to_string(),
                message: format!(
                    "Error rate ({:.2}%) exceeds threshold ({:.2}%)",
                    metrics.error_rate * 100.0,
                    thresholds.error_rate * 100.0
                ),
                metric: "error_rate".to_string(),
                current_value: metrics.error_rate,
                threshold_value: thresholds.error_rate,
                acknowledged: false,
                resolved_at: None,
            });
        }

        // Check satisfaction score
        if metrics.satisfaction_score < thresholds.satisfaction_score {
            return Some(QualityAlert {
                id: format!("satisfaction_{}", metrics.timestamp.timestamp()),
                timestamp: metrics.timestamp,
                severity: Self::determine_severity(
                    thresholds.satisfaction_score / metrics.satisfaction_score,
                ),
                title: "Low User Satisfaction".to_string(),
                message: format!(
                    "User satisfaction ({:.1}/5.0) below threshold ({:.1}/5.0)",
                    metrics.satisfaction_score, thresholds.satisfaction_score
                ),
                metric: "satisfaction_score".to_string(),
                current_value: metrics.satisfaction_score,
                threshold_value: thresholds.satisfaction_score,
                acknowledged: false,
                resolved_at: None,
            });
        }

        None
    }

    /// Determine alert severity based on threshold violation ratio
    fn determine_severity(violation_ratio: f64) -> AlertSeverity {
        if violation_ratio >= 2.0 {
            AlertSeverity::Critical
        } else if violation_ratio >= 1.5 {
            AlertSeverity::High
        } else if violation_ratio >= 1.2 {
            AlertSeverity::Medium
        } else {
            AlertSeverity::Low
        }
    }

    /// Handle quality alert
    async fn handle_alert(
        alert: QualityAlert,
        active_alerts: &Arc<RwLock<HashMap<String, QualityAlert>>>,
        alert_history: &Arc<RwLock<VecDeque<QualityAlert>>>,
        last_alert_times: &Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
        alert_config: &AlertConfig,
    ) {
        // Check cooldown
        {
            let last_times = last_alert_times.read().await;
            if let Some(last_time) = last_times.get(&alert.metric) {
                let cooldown_duration = Duration::minutes(alert_config.cooldown_minutes as i64);
                if alert.timestamp - *last_time < cooldown_duration {
                    return; // Still in cooldown period
                }
            }
        }

        // Check if this severity level should trigger alerts
        if !alert_config.severity_levels.contains(&alert.severity) {
            return;
        }

        // Add to active alerts
        {
            let mut alerts = active_alerts.write().await;
            alerts.insert(alert.id.clone(), alert.clone());
        }

        // Add to alert history
        {
            let mut history = alert_history.write().await;
            history.push_back(alert.clone());

            // Maintain history size (keep last 1000 alerts)
            while history.len() > 1000 {
                history.pop_front();
            }
        }

        // Update last alert time
        {
            let mut last_times = last_alert_times.write().await;
            last_times.insert(alert.metric.clone(), alert.timestamp);
        }

        // Deliver alert based on configuration
        if alert_config.log_enabled {
            Self::log_alert(&alert).await;
        }

        if alert_config.email_enabled {
            Self::send_email_alert(&alert).await;
        }

        if alert_config.webhook_enabled {
            Self::send_webhook_alert(&alert).await;
        }
    }

    /// Log alert to system logs
    async fn log_alert(alert: &QualityAlert) {
        match alert.severity {
            AlertSeverity::Critical => {
                log::error!("CRITICAL ALERT: {} - {}", alert.title, alert.message);
            }
            AlertSeverity::High => {
                log::warn!("HIGH PRIORITY ALERT: {} - {}", alert.title, alert.message);
            }
            AlertSeverity::Medium => {
                log::warn!("MEDIUM PRIORITY ALERT: {} - {}", alert.title, alert.message);
            }
            AlertSeverity::Low => {
                log::info!("LOW PRIORITY ALERT: {} - {}", alert.title, alert.message);
            }
            AlertSeverity::Info => {
                log::info!("INFO ALERT: {} - {}", alert.title, alert.message);
            }
        }
    }

    /// Send email alert with real SMTP implementation
    async fn send_email_alert(alert: &QualityAlert) {
        use lettre::transport::smtp::authentication::Credentials;
        use lettre::{Message, SmtpTransport, Transport};

        // In a production environment, these would come from environment variables
        let smtp_server =
            std::env::var("SMTP_SERVER").unwrap_or_else(|_| "smtp.gmail.com".to_string());
        let smtp_username =
            std::env::var("SMTP_USERNAME").unwrap_or_else(|_| "noreply@voirs.ai".to_string());
        let smtp_password =
            std::env::var("SMTP_PASSWORD").unwrap_or_else(|_| "password".to_string());
        let recipient =
            std::env::var("ALERT_EMAIL").unwrap_or_else(|_| "admin@voirs.ai".to_string());

        let email_result = Message::builder()
            .from(smtp_username.parse().unwrap_or_else(|_| "noreply@voirs.ai".parse().unwrap()))
            .to(recipient.parse().unwrap_or_else(|_| "admin@voirs.ai".parse().unwrap()))
            .subject(&format!("VoiRS Quality Alert: {}", alert.title))
            .body(format!(
                "Quality Alert Details:\n\nTitle: {}\nMessage: {}\nSeverity: {:?}\nTimestamp: {}\nMetric: {}\nCurrent Value: {}\nThreshold: {}",
                alert.title, alert.message, alert.severity, alert.timestamp, alert.metric, alert.current_value, alert.threshold_value
            ));

        match email_result {
            Ok(email) => {
                let creds = Credentials::new(smtp_username, smtp_password);
                let mailer = match SmtpTransport::relay(&smtp_server) {
                    Ok(builder) => builder.credentials(creds).build(),
                    Err(_) => {
                        log::warn!("Failed to connect to SMTP server, using localhost fallback");
                        SmtpTransport::builder_dangerous("localhost")
                            .port(25)
                            .credentials(creds)
                            .build()
                    }
                };

                match mailer.send(&email) {
                    Ok(_) => log::info!("Email alert sent successfully: {}", alert.title),
                    Err(e) => {
                        log::error!("Failed to send email alert: {}", e);
                        // Fallback to logging
                        log::warn!(
                            "EMAIL ALERT (fallback): {} - {}",
                            alert.title,
                            alert.message
                        );
                    }
                }
            }
            Err(e) => {
                log::error!("Failed to create email message: {}", e);
                log::warn!(
                    "EMAIL ALERT (fallback): {} - {}",
                    alert.title,
                    alert.message
                );
            }
        }
    }

    /// Send webhook alert with real HTTP implementation
    async fn send_webhook_alert(alert: &QualityAlert) {
        #[cfg(feature = "microservices")]
        {
            let webhook_url = std::env::var("WEBHOOK_URL").unwrap_or_else(|_| {
                "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK".to_string()
            });

            let payload = serde_json::json!({
                "text": format!("🚨 VoiRS Quality Alert: {}", alert.title),
                "attachments": [{
                    "color": match alert.severity {
                        crate::quality_monitor::AlertSeverity::Critical => "danger",
                        crate::quality_monitor::AlertSeverity::High => "danger",
                        crate::quality_monitor::AlertSeverity::Medium => "warning",
                        crate::quality_monitor::AlertSeverity::Low => "good",
                        crate::quality_monitor::AlertSeverity::Info => "good"
                    },
                    "fields": [
                        {
                            "title": "Message",
                            "value": alert.message,
                            "short": false
                        },
                        {
                            "title": "Severity",
                            "value": format!("{:?}", alert.severity),
                            "short": true
                        },
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp.to_rfc3339(),
                            "short": true
                        }
                    ],
                    "footer": "VoiRS Quality Monitor",
                    "ts": alert.timestamp.timestamp()
                }]
            });

            let client = reqwest::Client::new();
            match client.post(&webhook_url).json(&payload).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        log::info!("Webhook alert sent successfully: {}", alert.title);
                    } else {
                        log::error!("Webhook alert failed with status: {}", response.status());
                        log::warn!(
                            "WEBHOOK ALERT (fallback): {} - {}",
                            alert.title,
                            alert.message
                        );
                    }
                }
                Err(e) => {
                    log::error!("Failed to send webhook alert: {}", e);
                    log::warn!(
                        "WEBHOOK ALERT (fallback): {} - {}",
                        alert.title,
                        alert.message
                    );
                }
            }
        }

        #[cfg(not(feature = "microservices"))]
        {
            // Fallback to logging when microservices feature is disabled
            log::info!(
                "WEBHOOK ALERT (microservices disabled): {} - {}",
                alert.title,
                alert.message
            );
        }
    }

    /// Get current quality metrics
    pub async fn get_current_metrics(&self) -> Option<QualityMetrics> {
        let history = self.metrics_history.read().await;
        history.back().cloned()
    }

    /// Get metrics history
    pub async fn get_metrics_history(&self, limit: Option<usize>) -> Vec<QualityMetrics> {
        let history = self.metrics_history.read().await;
        let take_count = limit.unwrap_or(history.len());
        history.iter().rev().take(take_count).cloned().collect()
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<QualityAlert> {
        let alerts = self.active_alerts.read().await;
        alerts.values().cloned().collect()
    }

    /// Acknowledge alert
    pub async fn acknowledge_alert(&self, alert_id: &str) -> QualityResult<()> {
        let mut alerts = self.active_alerts.write().await;
        if let Some(alert) = alerts.get_mut(alert_id) {
            alert.acknowledged = true;
            Ok(())
        } else {
            Err(QualityMonitorError::MonitoringFailure {
                message: format!("Alert {} not found", alert_id),
            })
        }
    }

    /// Resolve alert
    pub async fn resolve_alert(&self, alert_id: &str) -> QualityResult<()> {
        let mut alerts = self.active_alerts.write().await;
        if let Some(mut alert) = alerts.remove(alert_id) {
            alert.resolved_at = Some(Utc::now());

            // Add to history if not already there
            let mut history = self.alert_history.write().await;
            if !history.iter().any(|a| a.id == alert_id) {
                history.push_back(alert);
            }

            Ok(())
        } else {
            Err(QualityMonitorError::MonitoringFailure {
                message: format!("Alert {} not found", alert_id),
            })
        }
    }

    /// Generate quality report
    pub async fn generate_quality_report(&self, hours: u64) -> QualityReport {
        let history = self.metrics_history.read().await;
        let cutoff_time = Utc::now() - Duration::hours(hours as i64);

        let recent_metrics: Vec<_> = history
            .iter()
            .filter(|m| m.timestamp >= cutoff_time)
            .cloned()
            .collect();

        if recent_metrics.is_empty() {
            return QualityReport {
                period_hours: hours,
                sample_count: 0,
                average_response_time_ms: 0.0,
                average_error_rate: 0.0,
                average_satisfaction_score: 0.0,
                average_availability: 0.0,
                average_quality_score: 0.0,
                alert_count: self.alert_history.read().await.len(),
                trends: QualityTrends::default(),
            };
        }

        let avg_response_time = recent_metrics
            .iter()
            .map(|m| m.response_time_ms)
            .sum::<f64>()
            / recent_metrics.len() as f64;
        let avg_error_rate =
            recent_metrics.iter().map(|m| m.error_rate).sum::<f64>() / recent_metrics.len() as f64;
        let avg_satisfaction = recent_metrics
            .iter()
            .map(|m| m.satisfaction_score)
            .sum::<f64>()
            / recent_metrics.len() as f64;
        let avg_availability = recent_metrics.iter().map(|m| m.availability).sum::<f64>()
            / recent_metrics.len() as f64;
        let avg_quality_score = recent_metrics
            .iter()
            .map(|m| m.overall_quality_score)
            .sum::<f64>()
            / recent_metrics.len() as f64;

        QualityReport {
            period_hours: hours,
            sample_count: recent_metrics.len(),
            average_response_time_ms: avg_response_time,
            average_error_rate: avg_error_rate,
            average_satisfaction_score: avg_satisfaction,
            average_availability: avg_availability,
            average_quality_score: avg_quality_score,
            alert_count: self.alert_history.read().await.len(),
            trends: Self::calculate_trends(&recent_metrics),
        }
    }

    /// Calculate quality trends
    fn calculate_trends(metrics: &[QualityMetrics]) -> QualityTrends {
        if metrics.len() < 2 {
            return QualityTrends::default();
        }

        let half_point = metrics.len() / 2;
        let first_half = &metrics[..half_point];
        let second_half = &metrics[half_point..];

        let first_avg_quality = first_half
            .iter()
            .map(|m| m.overall_quality_score)
            .sum::<f64>()
            / first_half.len() as f64;
        let second_avg_quality = second_half
            .iter()
            .map(|m| m.overall_quality_score)
            .sum::<f64>()
            / second_half.len() as f64;

        let quality_trend = if (second_avg_quality - first_avg_quality).abs() < 1.0 {
            "stable".to_string()
        } else if second_avg_quality > first_avg_quality {
            "improving".to_string()
        } else {
            "declining".to_string()
        };

        QualityTrends {
            quality_trend,
            response_time_trend: "stable".to_string(), // Simplified for now
            error_rate_trend: "stable".to_string(),
            satisfaction_trend: "stable".to_string(),
        }
    }
}

/// Quality report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    /// Report period in hours
    pub period_hours: u64,
    /// Number of samples in the report
    pub sample_count: usize,
    /// Average response time in milliseconds
    pub average_response_time_ms: f64,
    /// Average error rate
    pub average_error_rate: f64,
    /// Average satisfaction score
    pub average_satisfaction_score: f64,
    /// Average availability
    pub average_availability: f64,
    /// Average overall quality score
    pub average_quality_score: f64,
    /// Number of alerts in the period
    pub alert_count: usize,
    /// Quality trends
    pub trends: QualityTrends,
}

impl Default for QualityReport {
    fn default() -> Self {
        Self {
            period_hours: 0,
            sample_count: 0,
            average_response_time_ms: 0.0,
            average_error_rate: 0.0,
            average_satisfaction_score: 0.0,
            average_availability: 0.0,
            average_quality_score: 0.0,
            alert_count: 0,
            trends: QualityTrends::default(),
        }
    }
}

/// Quality trends analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTrends {
    /// Overall quality trend (improving, declining, stable)
    pub quality_trend: String,
    /// Response time trend
    pub response_time_trend: String,
    /// Error rate trend
    pub error_rate_trend: String,
    /// Satisfaction trend
    pub satisfaction_trend: String,
}

impl Default for QualityTrends {
    fn default() -> Self {
        Self {
            quality_trend: "stable".to_string(),
            response_time_trend: "stable".to_string(),
            error_rate_trend: "stable".to_string(),
            satisfaction_trend: "stable".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quality_monitor_creation() {
        let config = QualityMonitorConfig::default();
        let monitor = QualityMonitor::new(config).await;
        assert!(monitor.is_ok());
    }

    #[tokio::test]
    async fn test_quality_metrics_calculation() {
        let thresholds = QualityThresholds::default();
        let mut metrics = QualityMetrics {
            timestamp: Utc::now(),
            response_time_ms: 500.0,
            error_rate: 0.01,
            satisfaction_score: 4.5,
            availability: 0.999,
            memory_usage_mb: 256.0,
            cpu_usage: 0.3,
            throughput_rps: 200.0,
            active_sessions: 10,
            overall_quality_score: 0.0,
        };

        metrics.calculate_quality_score(&thresholds);
        assert!(metrics.overall_quality_score > 90.0);
    }

    #[tokio::test]
    async fn test_threshold_checking() {
        let thresholds = QualityThresholds {
            response_time_ms: 1000.0,
            error_rate: 0.05,
            satisfaction_score: 3.0,
            availability: 0.99,
            memory_usage_mb: 512.0,
            cpu_usage: 0.8,
            throughput_rps: 50.0,
        };

        let metrics = QualityMetrics {
            timestamp: Utc::now(),
            response_time_ms: 1500.0, // Exceeds threshold
            error_rate: 0.02,
            satisfaction_score: 4.0,
            availability: 0.995,
            memory_usage_mb: 256.0,
            cpu_usage: 0.4,
            throughput_rps: 100.0,
            active_sessions: 5,
            overall_quality_score: 85.0,
        };

        let alert = QualityMonitor::check_thresholds(&metrics, &thresholds).await;
        assert!(alert.is_some());

        let alert = alert.unwrap();
        assert_eq!(alert.metric, "response_time_ms");
        assert_eq!(alert.current_value, 1500.0);
        assert_eq!(alert.threshold_value, 1000.0);
    }

    #[tokio::test]
    async fn test_alert_severity_determination() {
        assert_eq!(
            QualityMonitor::determine_severity(2.5),
            AlertSeverity::Critical
        );
        assert_eq!(QualityMonitor::determine_severity(1.7), AlertSeverity::High);
        assert_eq!(
            QualityMonitor::determine_severity(1.3),
            AlertSeverity::Medium
        );
        assert_eq!(QualityMonitor::determine_severity(1.1), AlertSeverity::Low);
    }

    #[tokio::test]
    async fn test_quality_report_generation() {
        let config = QualityMonitorConfig::default();
        let monitor = QualityMonitor::new(config).await.unwrap();

        let report = monitor.generate_quality_report(24).await;
        assert_eq!(report.period_hours, 24);
        assert_eq!(report.sample_count, 0); // No metrics yet
    }

    #[tokio::test]
    async fn test_alert_acknowledgment() {
        let config = QualityMonitorConfig::default();
        let monitor = QualityMonitor::new(config).await.unwrap();

        let alert = QualityAlert {
            id: "test_alert".to_string(),
            timestamp: Utc::now(),
            severity: AlertSeverity::Medium,
            title: "Test Alert".to_string(),
            message: "Test message".to_string(),
            metric: "test_metric".to_string(),
            current_value: 100.0,
            threshold_value: 50.0,
            acknowledged: false,
            resolved_at: None,
        };

        // Add alert manually for testing
        {
            let mut alerts = monitor.active_alerts.write().await;
            alerts.insert(alert.id.clone(), alert);
        }

        let result = monitor.acknowledge_alert("test_alert").await;
        assert!(result.is_ok());

        let alerts = monitor.get_active_alerts().await;
        assert!(alerts[0].acknowledged);
    }

    #[tokio::test]
    async fn test_alert_resolution() {
        let config = QualityMonitorConfig::default();
        let monitor = QualityMonitor::new(config).await.unwrap();

        let alert = QualityAlert {
            id: "test_alert_resolve".to_string(),
            timestamp: Utc::now(),
            severity: AlertSeverity::Medium,
            title: "Test Alert".to_string(),
            message: "Test message".to_string(),
            metric: "test_metric".to_string(),
            current_value: 100.0,
            threshold_value: 50.0,
            acknowledged: false,
            resolved_at: None,
        };

        // Add alert manually for testing
        {
            let mut alerts = monitor.active_alerts.write().await;
            alerts.insert(alert.id.clone(), alert);
        }

        let result = monitor.resolve_alert("test_alert_resolve").await;
        assert!(result.is_ok());

        let active_alerts = monitor.get_active_alerts().await;
        assert!(active_alerts.is_empty());
    }

    #[tokio::test]
    async fn test_config_defaults() {
        let config = QualityMonitorConfig::default();
        assert!(config.enabled);
        assert_eq!(config.monitoring_interval, 60);
        assert_eq!(config.retention_days, 30);
        assert_eq!(config.max_samples, 10000);

        let thresholds = QualityThresholds::default();
        assert_eq!(thresholds.response_time_ms, 2000.0);
        assert_eq!(thresholds.error_rate, 0.05);
        assert_eq!(thresholds.satisfaction_score, 3.5);
        assert_eq!(thresholds.availability, 0.99);
    }
}
