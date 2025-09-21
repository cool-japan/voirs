//! Real-time performance monitoring and alerting
//!
//! This module provides continuous monitoring of system performance with
//! configurable thresholds, alerts, and automated responses.

use super::{GpuMetrics, MemoryMetrics, PerformanceMetrics, SynthesisMetrics, SystemMetrics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, watch, RwLock};

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Monitoring interval
    pub interval: Duration,
    /// Enable monitoring
    pub enabled: bool,
    /// Alert thresholds
    pub thresholds: AlertThresholds,
    /// Alert channels configuration
    pub alerts: AlertConfig,
    /// Monitoring targets
    pub targets: Vec<MonitorTarget>,
    /// Historical data retention
    pub retention_duration: Duration,
    /// Auto-recovery settings
    pub auto_recovery: AutoRecoveryConfig,
}

/// Alert threshold configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// CPU usage percentage (0-100)
    pub cpu_usage_percent: Option<f64>,
    /// Memory usage percentage (0-100)
    pub memory_usage_percent: Option<f64>,
    /// GPU utilization percentage (0-100)
    pub gpu_utilization_percent: Option<f64>,
    /// GPU memory usage percentage (0-100)
    pub gpu_memory_percent: Option<f64>,
    /// Real-time factor minimum threshold
    pub min_real_time_factor: Option<f64>,
    /// Maximum synthesis time in milliseconds
    pub max_synthesis_time_ms: Option<f64>,
    /// Maximum queue depth
    pub max_queue_depth: Option<usize>,
    /// Minimum success rate percentage
    pub min_success_rate_percent: Option<f64>,
    /// Maximum error rate percentage
    pub max_error_rate_percent: Option<f64>,
    /// GPU temperature threshold in Celsius
    pub max_gpu_temperature: Option<f64>,
    /// Disk usage percentage
    pub max_disk_usage_percent: Option<f64>,
    /// Network bandwidth usage in bytes/sec
    pub max_network_bps: Option<u64>,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable console logging of alerts
    pub console_logging: bool,
    /// Enable file logging of alerts
    pub file_logging: Option<std::path::PathBuf>,
    /// Enable email notifications
    pub email_notifications: Option<EmailConfig>,
    /// Enable webhook notifications
    pub webhook_notifications: Option<WebhookConfig>,
    /// Alert cooldown period to prevent spam
    pub cooldown_duration: Duration,
    /// Maximum alerts per hour
    pub max_alerts_per_hour: usize,
}

/// Email notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailConfig {
    /// SMTP server address
    pub smtp_server: String,
    /// SMTP port
    pub smtp_port: u16,
    /// Username for authentication
    pub username: String,
    /// Password for authentication (should be encrypted/secured)
    pub password: String,
    /// Sender email address
    pub from_email: String,
    /// Recipient email addresses
    pub to_emails: Vec<String>,
    /// Use TLS encryption
    pub use_tls: bool,
}

/// Webhook notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// Webhook URL
    pub url: String,
    /// HTTP method (GET, POST, etc.)
    pub method: String,
    /// Custom headers
    pub headers: HashMap<String, String>,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Retry attempts
    pub retry_attempts: usize,
}

/// Monitoring target configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorTarget {
    /// Target name/identifier
    pub name: String,
    /// Target type
    pub target_type: MonitorTargetType,
    /// Enable monitoring for this target
    pub enabled: bool,
    /// Custom thresholds for this target
    pub custom_thresholds: Option<AlertThresholds>,
    /// Alert severity level
    pub severity: AlertSeverity,
}

/// Types of monitoring targets
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MonitorTargetType {
    /// System-wide CPU monitoring
    SystemCpu,
    /// System-wide memory monitoring
    SystemMemory,
    /// GPU monitoring
    Gpu,
    /// Synthesis performance monitoring
    SynthesisPerformance,
    /// Queue depth monitoring
    QueueDepth,
    /// Error rate monitoring
    ErrorRate,
    /// Disk I/O monitoring
    DiskIo,
    /// Network I/O monitoring
    NetworkIo,
    /// Custom metric monitoring
    Custom(String),
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alerts
    Info,
    /// Warning alerts
    Warning,
    /// Critical alerts (requires immediate attention)
    Critical,
    /// Emergency alerts (system may be unusable)
    Emergency,
}

/// Auto-recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoRecoveryConfig {
    /// Enable automatic recovery actions
    pub enabled: bool,
    /// Recovery actions to attempt
    pub actions: Vec<RecoveryAction>,
    /// Maximum recovery attempts per hour
    pub max_attempts_per_hour: usize,
    /// Delay between recovery attempts
    pub retry_delay: Duration,
}

/// Recovery action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    /// Reduce batch size
    ReduceBatchSize { min_size: usize },
    /// Clear caches
    ClearCaches,
    /// Restart worker threads
    RestartWorkers,
    /// Reduce parallel processing
    ReduceParallelism { min_threads: usize },
    /// Enable memory optimization
    EnableMemoryOptimization,
    /// Switch to lower quality mode
    ReduceQuality,
    /// Pause processing temporarily
    PauseProcessing { duration: Duration },
    /// Custom command execution
    CustomCommand { command: String, args: Vec<String> },
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    /// Alert ID
    pub id: String,
    /// Alert timestamp
    pub timestamp: u64,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Target that triggered the alert
    pub target: MonitorTargetType,
    /// Alert message
    pub message: String,
    /// Current metric value
    pub current_value: f64,
    /// Threshold that was exceeded
    pub threshold_value: f64,
    /// Metric name
    pub metric_name: String,
    /// Additional context
    pub context: HashMap<String, String>,
    /// Whether the alert is resolved
    pub resolved: bool,
    /// Resolution timestamp
    pub resolved_at: Option<u64>,
}

/// Alert manager for handling notifications and recovery
#[derive(Debug)]
pub struct AlertManager {
    /// Configuration
    config: AlertConfig,
    /// Active alerts
    active_alerts: Arc<RwLock<HashMap<String, PerformanceAlert>>>,
    /// Alert history
    alert_history: Arc<RwLock<Vec<PerformanceAlert>>>,
    /// Alert cooldown tracking
    alert_cooldowns: Arc<RwLock<HashMap<String, Instant>>>,
    /// Alerts sent in current hour
    hourly_alert_count: Arc<RwLock<usize>>,
    /// Last hour reset time
    last_hour_reset: Arc<RwLock<Instant>>,
}

/// Performance monitor
pub struct PerformanceMonitor {
    /// Monitor configuration
    config: MonitorConfig,
    /// Alert manager
    alert_manager: AlertManager,
    /// Current metrics
    current_metrics: Arc<RwLock<Option<PerformanceMetrics>>>,
    /// Monitoring status
    is_running: Arc<RwLock<bool>>,
    /// Metrics history for trend analysis
    metrics_history: Arc<RwLock<Vec<PerformanceMetrics>>>,
    /// Alert sender channel
    alert_sender: mpsc::UnboundedSender<PerformanceAlert>,
    /// Alert receiver channel
    alert_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<PerformanceAlert>>>>,
    /// Shutdown signal
    shutdown_sender: Option<watch::Sender<bool>>,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(config: MonitorConfig) -> Self {
        let (alert_sender, alert_receiver) = mpsc::unbounded_channel();
        let (shutdown_sender, _) = watch::channel(false);

        Self {
            alert_manager: AlertManager::new(config.alerts.clone()),
            config,
            current_metrics: Arc::new(RwLock::new(None)),
            is_running: Arc::new(RwLock::new(false)),
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            alert_sender,
            alert_receiver: Arc::new(RwLock::new(Some(alert_receiver))),
            shutdown_sender: Some(shutdown_sender),
        }
    }

    /// Start monitoring
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            return Ok(());
        }
        *is_running = true;
        drop(is_running);

        tracing::info!(
            "Starting performance monitor with interval: {:?}",
            self.config.interval
        );

        // Start alert processing task
        self.start_alert_processor().await?;

        // Start monitoring loop
        self.start_monitoring_loop().await?;

        Ok(())
    }

    /// Stop monitoring
    pub async fn stop(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut is_running = self.is_running.write().await;
        if !*is_running {
            return Ok(());
        }
        *is_running = false;

        // Send shutdown signal
        if let Some(sender) = &self.shutdown_sender {
            let _ = sender.send(true);
        }

        tracing::info!("Stopped performance monitor");
        Ok(())
    }

    /// Update current metrics
    pub async fn update_metrics(&self, metrics: PerformanceMetrics) {
        // Store current metrics
        let mut current = self.current_metrics.write().await;
        *current = Some(metrics.clone());
        drop(current);

        // Add to history
        let mut history = self.metrics_history.write().await;
        history.push(metrics.clone());

        // Maintain history size
        let max_history =
            (self.config.retention_duration.as_secs() / self.config.interval.as_secs()) as usize;
        if history.len() > max_history {
            history.remove(0);
        }
        drop(history);

        // Check for alerts
        self.check_alerts(&metrics).await;
    }

    /// Check metrics against alert thresholds
    async fn check_alerts(&self, metrics: &PerformanceMetrics) {
        for target in &self.config.targets {
            if !target.enabled {
                continue;
            }

            let thresholds = target
                .custom_thresholds
                .as_ref()
                .unwrap_or(&self.config.thresholds);

            if let Some(alert) = self.check_target_alerts(&target, metrics, thresholds).await {
                let _ = self.alert_sender.send(alert);
            }
        }
    }

    /// Check alerts for a specific target
    async fn check_target_alerts(
        &self,
        target: &MonitorTarget,
        metrics: &PerformanceMetrics,
        thresholds: &AlertThresholds,
    ) -> Option<PerformanceAlert> {
        match target.target_type {
            MonitorTargetType::SystemCpu => {
                if let Some(threshold) = thresholds.cpu_usage_percent {
                    if metrics.system.cpu_usage > threshold {
                        return Some(
                            self.create_alert(
                                target,
                                "High CPU usage detected",
                                metrics.system.cpu_usage,
                                threshold,
                                "cpu_usage_percent",
                            )
                            .await,
                        );
                    }
                }
            }
            MonitorTargetType::SystemMemory => {
                let memory_usage_percent = (metrics.system.memory_used as f64
                    / (metrics.system.memory_used + metrics.system.memory_available) as f64)
                    * 100.0;

                if let Some(threshold) = thresholds.memory_usage_percent {
                    if memory_usage_percent > threshold {
                        return Some(
                            self.create_alert(
                                target,
                                "High memory usage detected",
                                memory_usage_percent,
                                threshold,
                                "memory_usage_percent",
                            )
                            .await,
                        );
                    }
                }
            }
            MonitorTargetType::Gpu => {
                if let Some(ref gpu_metrics) = metrics.gpu {
                    // Check GPU utilization
                    if let Some(threshold) = thresholds.gpu_utilization_percent {
                        if gpu_metrics.utilization > threshold {
                            return Some(
                                self.create_alert(
                                    target,
                                    "High GPU utilization detected",
                                    gpu_metrics.utilization,
                                    threshold,
                                    "gpu_utilization_percent",
                                )
                                .await,
                            );
                        }
                    }

                    // Check GPU memory
                    let gpu_memory_percent =
                        (gpu_metrics.memory_used as f64 / gpu_metrics.memory_total as f64) * 100.0;
                    if let Some(threshold) = thresholds.gpu_memory_percent {
                        if gpu_memory_percent > threshold {
                            return Some(
                                self.create_alert(
                                    target,
                                    "High GPU memory usage detected",
                                    gpu_memory_percent,
                                    threshold,
                                    "gpu_memory_percent",
                                )
                                .await,
                            );
                        }
                    }

                    // Check GPU temperature
                    if let Some(threshold) = thresholds.max_gpu_temperature {
                        if gpu_metrics.temperature > threshold {
                            return Some(
                                self.create_alert(
                                    target,
                                    "High GPU temperature detected",
                                    gpu_metrics.temperature,
                                    threshold,
                                    "gpu_temperature",
                                )
                                .await,
                            );
                        }
                    }
                }
            }
            MonitorTargetType::SynthesisPerformance => {
                // Check real-time factor
                if let Some(threshold) = thresholds.min_real_time_factor {
                    if metrics.synthesis.real_time_factor < threshold {
                        return Some(
                            self.create_alert(
                                target,
                                "Poor synthesis performance detected",
                                metrics.synthesis.real_time_factor,
                                threshold,
                                "real_time_factor",
                            )
                            .await,
                        );
                    }
                }

                // Check synthesis time
                if let Some(threshold) = thresholds.max_synthesis_time_ms {
                    if metrics.synthesis.avg_synthesis_time_ms > threshold {
                        return Some(
                            self.create_alert(
                                target,
                                "High synthesis time detected",
                                metrics.synthesis.avg_synthesis_time_ms,
                                threshold,
                                "synthesis_time_ms",
                            )
                            .await,
                        );
                    }
                }
            }
            MonitorTargetType::QueueDepth => {
                if let Some(threshold) = thresholds.max_queue_depth {
                    if metrics.synthesis.queue_depth > threshold {
                        return Some(
                            self.create_alert(
                                target,
                                "High queue depth detected",
                                metrics.synthesis.queue_depth as f64,
                                threshold as f64,
                                "queue_depth",
                            )
                            .await,
                        );
                    }
                }
            }
            MonitorTargetType::ErrorRate => {
                let error_rate = if metrics.synthesis.total_operations > 0 {
                    (metrics.synthesis.failed_operations as f64
                        / metrics.synthesis.total_operations as f64)
                        * 100.0
                } else {
                    0.0
                };

                if let Some(threshold) = thresholds.max_error_rate_percent {
                    if error_rate > threshold {
                        return Some(
                            self.create_alert(
                                target,
                                "High error rate detected",
                                error_rate,
                                threshold,
                                "error_rate_percent",
                            )
                            .await,
                        );
                    }
                }
            }
            MonitorTargetType::DiskIo => {
                let total_disk_bps = metrics.system.disk_read_bps + metrics.system.disk_write_bps;
                if let Some(threshold) = thresholds.max_network_bps {
                    if total_disk_bps > threshold {
                        return Some(
                            self.create_alert(
                                target,
                                "High disk I/O detected",
                                total_disk_bps as f64,
                                threshold as f64,
                                "disk_io_bps",
                            )
                            .await,
                        );
                    }
                }
            }
            MonitorTargetType::NetworkIo => {
                if let Some(threshold) = thresholds.max_network_bps {
                    if metrics.system.network_bps > threshold {
                        return Some(
                            self.create_alert(
                                target,
                                "High network I/O detected",
                                metrics.system.network_bps as f64,
                                threshold as f64,
                                "network_io_bps",
                            )
                            .await,
                        );
                    }
                }
            }
            MonitorTargetType::Custom(_) => {
                // Custom monitoring logic would go here
            }
        }

        None
    }

    /// Create a performance alert
    async fn create_alert(
        &self,
        target: &MonitorTarget,
        message: &str,
        current_value: f64,
        threshold_value: f64,
        metric_name: &str,
    ) -> PerformanceAlert {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let alert_id = format!(
            "{}_{}_{}_{}",
            target.name,
            metric_name,
            timestamp,
            fastrand::u32(..)
        );

        let mut context = HashMap::new();
        context.insert("target_name".to_string(), target.name.clone());
        context.insert("metric_name".to_string(), metric_name.to_string());
        context.insert("current_value".to_string(), current_value.to_string());
        context.insert("threshold_value".to_string(), threshold_value.to_string());

        PerformanceAlert {
            id: alert_id,
            timestamp,
            severity: target.severity.clone(),
            target: target.target_type.clone(),
            message: message.to_string(),
            current_value,
            threshold_value,
            metric_name: metric_name.to_string(),
            context,
            resolved: false,
            resolved_at: None,
        }
    }

    /// Start alert processing task
    async fn start_alert_processor(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut receiver = self.alert_receiver.write().await;
        if let Some(rx) = receiver.take() {
            let alert_manager = self.alert_manager.clone();
            let auto_recovery = self.config.auto_recovery.clone();

            tokio::spawn(async move {
                let mut rx = rx;
                while let Some(alert) = rx.recv().await {
                    // Process the alert
                    alert_manager.process_alert(alert.clone()).await;

                    // Attempt auto-recovery if enabled
                    if auto_recovery.enabled {
                        Self::attempt_auto_recovery(&alert, &auto_recovery).await;
                    }
                }
            });
        }

        Ok(())
    }

    /// Start monitoring loop
    async fn start_monitoring_loop(&self) -> Result<(), Box<dyn std::error::Error>> {
        let is_running = self.is_running.clone();
        let interval = self.config.interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                let running = is_running.read().await;
                if !*running {
                    break;
                }
                drop(running);

                // Monitoring loop would collect metrics here
                // For now, this is a placeholder since metrics come from external sources
                tracing::debug!("Performance monitoring tick");
            }
        });

        Ok(())
    }

    /// Attempt automatic recovery
    async fn attempt_auto_recovery(alert: &PerformanceAlert, config: &AutoRecoveryConfig) {
        for action in &config.actions {
            match action {
                RecoveryAction::ReduceBatchSize { min_size } => {
                    tracing::info!("Auto-recovery: Reducing batch size (min: {})", min_size);
                    // Implementation would reduce batch size in the synthesis system
                }
                RecoveryAction::ClearCaches => {
                    tracing::info!("Auto-recovery: Clearing caches");
                    // Implementation would clear system caches
                }
                RecoveryAction::RestartWorkers => {
                    tracing::info!("Auto-recovery: Restarting worker threads");
                    // Implementation would restart worker thread pool
                }
                RecoveryAction::ReduceParallelism { min_threads } => {
                    tracing::info!("Auto-recovery: Reducing parallelism (min: {})", min_threads);
                    // Implementation would reduce thread count
                }
                RecoveryAction::EnableMemoryOptimization => {
                    tracing::info!("Auto-recovery: Enabling memory optimization");
                    // Implementation would enable memory optimization features
                }
                RecoveryAction::ReduceQuality => {
                    tracing::info!("Auto-recovery: Reducing synthesis quality");
                    // Implementation would reduce quality settings
                }
                RecoveryAction::PauseProcessing { duration } => {
                    tracing::info!("Auto-recovery: Pausing processing for {:?}", duration);
                    tokio::time::sleep(*duration).await;
                }
                RecoveryAction::CustomCommand { command, args } => {
                    tracing::info!(
                        "Auto-recovery: Executing custom command: {} {:?}",
                        command,
                        args
                    );
                    // Implementation would execute custom recovery command
                }
            }

            // Wait between recovery actions
            tokio::time::sleep(config.retry_delay).await;
        }
    }

    /// Get current monitoring status
    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }

    /// Get current metrics
    pub async fn get_current_metrics(&self) -> Option<PerformanceMetrics> {
        self.current_metrics.read().await.clone()
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<PerformanceAlert> {
        self.alert_manager.get_active_alerts().await
    }

    /// Get alert history
    pub async fn get_alert_history(&self, limit: Option<usize>) -> Vec<PerformanceAlert> {
        self.alert_manager.get_alert_history(limit).await
    }

    /// Update monitor configuration
    pub async fn update_config(&mut self, config: MonitorConfig) {
        self.config = config;
    }
}

impl AlertManager {
    /// Create a new alert manager
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(RwLock::new(Vec::new())),
            alert_cooldowns: Arc::new(RwLock::new(HashMap::new())),
            hourly_alert_count: Arc::new(RwLock::new(0)),
            last_hour_reset: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Process a performance alert
    async fn process_alert(&self, alert: PerformanceAlert) {
        // Check cooldown
        if self.is_in_cooldown(&alert).await {
            return;
        }

        // Check hourly limit
        if !self.can_send_alert().await {
            tracing::warn!(
                "Alert rate limit exceeded, skipping alert: {}",
                alert.message
            );
            return;
        }

        // Add to active alerts
        let mut active = self.active_alerts.write().await;
        active.insert(alert.id.clone(), alert.clone());
        drop(active);

        // Add to history
        let mut history = self.alert_history.write().await;
        history.push(alert.clone());
        drop(history);

        // Set cooldown
        self.set_cooldown(&alert).await;

        // Send notifications
        self.send_notifications(&alert).await;

        // Increment hourly count
        self.increment_hourly_count().await;
    }

    /// Check if alert is in cooldown period
    async fn is_in_cooldown(&self, alert: &PerformanceAlert) -> bool {
        let cooldowns = self.alert_cooldowns.read().await;
        if let Some(&last_sent) = cooldowns.get(&alert.metric_name) {
            last_sent.elapsed() < self.config.cooldown_duration
        } else {
            false
        }
    }

    /// Set cooldown for alert type
    async fn set_cooldown(&self, alert: &PerformanceAlert) {
        let mut cooldowns = self.alert_cooldowns.write().await;
        cooldowns.insert(alert.metric_name.clone(), Instant::now());
    }

    /// Check if we can send more alerts this hour
    async fn can_send_alert(&self) -> bool {
        // Reset hourly count if needed
        let mut last_reset = self.last_hour_reset.write().await;
        if last_reset.elapsed() >= Duration::from_secs(3600) {
            *last_reset = Instant::now();
            let mut count = self.hourly_alert_count.write().await;
            *count = 0;
        }
        drop(last_reset);

        let count = self.hourly_alert_count.read().await;
        *count < self.config.max_alerts_per_hour
    }

    /// Increment hourly alert count
    async fn increment_hourly_count(&self) {
        let mut count = self.hourly_alert_count.write().await;
        *count += 1;
    }

    /// Send alert notifications
    async fn send_notifications(&self, alert: &PerformanceAlert) {
        // Console logging
        if self.config.console_logging {
            match alert.severity {
                AlertSeverity::Info => {
                    tracing::info!("ALERT [{}]: {}", alert.severity_string(), alert.message)
                }
                AlertSeverity::Warning => {
                    tracing::warn!("ALERT [{}]: {}", alert.severity_string(), alert.message)
                }
                AlertSeverity::Critical => {
                    tracing::error!("ALERT [{}]: {}", alert.severity_string(), alert.message)
                }
                AlertSeverity::Emergency => {
                    tracing::error!("ALERT [{}]: {}", alert.severity_string(), alert.message)
                }
            }
        }

        // File logging
        if let Some(ref log_path) = self.config.file_logging {
            let log_entry = format!(
                "{} [{}] {}: {} (current: {:.2}, threshold: {:.2})\n",
                alert.timestamp,
                alert.severity_string(),
                alert.metric_name,
                alert.message,
                alert.current_value,
                alert.threshold_value
            );

            if let Err(e) = tokio::fs::write(log_path, log_entry).await {
                tracing::error!("Failed to write alert to log file: {}", e);
            }
        }

        // Email notifications
        if let Some(ref email_config) = self.config.email_notifications {
            if let Err(e) = self.send_email_alert(alert, email_config).await {
                tracing::error!("Failed to send email alert: {}", e);
            }
        }

        // Webhook notifications
        if let Some(ref webhook_config) = self.config.webhook_notifications {
            if let Err(e) = self.send_webhook_alert(alert, webhook_config).await {
                tracing::error!("Failed to send webhook alert: {}", e);
            }
        }
    }

    /// Send email alert
    async fn send_email_alert(
        &self,
        alert: &PerformanceAlert,
        config: &EmailConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Email sending implementation would go here
        // For now, just log the attempt
        tracing::info!(
            "Would send email alert to {:?}: {}",
            config.to_emails,
            alert.message
        );
        Ok(())
    }

    /// Send webhook alert
    async fn send_webhook_alert(
        &self,
        alert: &PerformanceAlert,
        config: &WebhookConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let payload = serde_json::to_value(alert)?;

        for attempt in 0..=config.retry_attempts {
            let mut request = match config.method.to_uppercase().as_str() {
                "POST" => client.post(&config.url),
                "PUT" => client.put(&config.url),
                "PATCH" => client.patch(&config.url),
                _ => client.get(&config.url),
            };

            // Add headers
            for (key, value) in &config.headers {
                request = request.header(key, value);
            }

            // Add JSON body for POST/PUT/PATCH
            if matches!(
                config.method.to_uppercase().as_str(),
                "POST" | "PUT" | "PATCH"
            ) {
                request = request.json(&payload);
            }

            let response = request
                .timeout(Duration::from_secs(config.timeout_seconds))
                .send()
                .await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    tracing::info!("Webhook alert sent successfully to {}", config.url);
                    return Ok(());
                }
                Ok(resp) => {
                    tracing::warn!(
                        "Webhook alert failed with status {}: {}",
                        resp.status(),
                        config.url
                    );
                }
                Err(e) => {
                    tracing::warn!("Webhook alert attempt {} failed: {}", attempt + 1, e);
                }
            }

            if attempt < config.retry_attempts {
                tokio::time::sleep(Duration::from_secs(2_u64.pow(attempt as u32))).await;
            }
        }

        Err("All webhook attempts failed".into())
    }

    /// Get active alerts
    async fn get_active_alerts(&self) -> Vec<PerformanceAlert> {
        let active = self.active_alerts.read().await;
        active.values().cloned().collect()
    }

    /// Get alert history
    async fn get_alert_history(&self, limit: Option<usize>) -> Vec<PerformanceAlert> {
        let history = self.alert_history.read().await;
        if let Some(limit) = limit {
            history.iter().rev().take(limit).cloned().collect()
        } else {
            history.clone()
        }
    }

    /// Resolve an alert
    pub async fn resolve_alert(&self, alert_id: &str) -> bool {
        let mut active = self.active_alerts.write().await;
        if let Some(mut alert) = active.remove(alert_id) {
            alert.resolved = true;
            alert.resolved_at = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            );

            // Update in history
            let mut history = self.alert_history.write().await;
            if let Some(hist_alert) = history.iter_mut().find(|a| a.id == alert_id) {
                hist_alert.resolved = true;
                hist_alert.resolved_at = alert.resolved_at;
            }

            true
        } else {
            false
        }
    }
}

impl Clone for AlertManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            active_alerts: self.active_alerts.clone(),
            alert_history: self.alert_history.clone(),
            alert_cooldowns: self.alert_cooldowns.clone(),
            hourly_alert_count: self.hourly_alert_count.clone(),
            last_hour_reset: self.last_hour_reset.clone(),
        }
    }
}

impl PerformanceAlert {
    /// Get severity as string
    pub fn severity_string(&self) -> &'static str {
        match self.severity {
            AlertSeverity::Info => "INFO",
            AlertSeverity::Warning => "WARNING",
            AlertSeverity::Critical => "CRITICAL",
            AlertSeverity::Emergency => "EMERGENCY",
        }
    }
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(10),
            enabled: false,
            thresholds: AlertThresholds::default(),
            alerts: AlertConfig::default(),
            targets: vec![
                MonitorTarget {
                    name: "system_cpu".to_string(),
                    target_type: MonitorTargetType::SystemCpu,
                    enabled: true,
                    custom_thresholds: None,
                    severity: AlertSeverity::Warning,
                },
                MonitorTarget {
                    name: "system_memory".to_string(),
                    target_type: MonitorTargetType::SystemMemory,
                    enabled: true,
                    custom_thresholds: None,
                    severity: AlertSeverity::Warning,
                },
                MonitorTarget {
                    name: "synthesis_performance".to_string(),
                    target_type: MonitorTargetType::SynthesisPerformance,
                    enabled: true,
                    custom_thresholds: None,
                    severity: AlertSeverity::Critical,
                },
            ],
            retention_duration: Duration::from_secs(3600), // 1 hour
            auto_recovery: AutoRecoveryConfig::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_usage_percent: Some(80.0),
            memory_usage_percent: Some(85.0),
            gpu_utilization_percent: Some(95.0),
            gpu_memory_percent: Some(90.0),
            min_real_time_factor: Some(0.8),
            max_synthesis_time_ms: Some(5000.0),
            max_queue_depth: Some(20),
            min_success_rate_percent: Some(90.0),
            max_error_rate_percent: Some(10.0),
            max_gpu_temperature: Some(85.0),
            max_disk_usage_percent: Some(90.0),
            max_network_bps: Some(1_000_000_000), // 1 GB/s
        }
    }
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            console_logging: true,
            file_logging: None,
            email_notifications: None,
            webhook_notifications: None,
            cooldown_duration: Duration::from_secs(300), // 5 minutes
            max_alerts_per_hour: 20,
        }
    }
}

impl Default for AutoRecoveryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            actions: vec![
                RecoveryAction::ReduceBatchSize { min_size: 8 },
                RecoveryAction::ClearCaches,
                RecoveryAction::ReduceParallelism { min_threads: 2 },
            ],
            max_attempts_per_hour: 5,
            retry_delay: Duration::from_secs(30),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_monitor_creation() {
        let config = MonitorConfig::default();
        let monitor = PerformanceMonitor::new(config);
        assert!(!monitor.is_running().await);
    }

    #[tokio::test]
    async fn test_alert_creation() {
        let config = MonitorConfig::default();
        let monitor = PerformanceMonitor::new(config);

        let target = MonitorTarget {
            name: "test_cpu".to_string(),
            target_type: MonitorTargetType::SystemCpu,
            enabled: true,
            custom_thresholds: None,
            severity: AlertSeverity::Warning,
        };

        let alert = monitor
            .create_alert(
                &target,
                "Test alert message",
                85.0,
                80.0,
                "cpu_usage_percent",
            )
            .await;

        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert_eq!(alert.current_value, 85.0);
        assert_eq!(alert.threshold_value, 80.0);
        assert!(!alert.resolved);
    }

    #[tokio::test]
    async fn test_alert_manager() {
        let config = AlertConfig::default();
        let manager = AlertManager::new(config);

        let mut alert = PerformanceAlert {
            id: "test_alert".to_string(),
            timestamp: 1234567890,
            severity: AlertSeverity::Warning,
            target: MonitorTargetType::SystemCpu,
            message: "Test alert".to_string(),
            current_value: 85.0,
            threshold_value: 80.0,
            metric_name: "cpu_usage".to_string(),
            context: HashMap::new(),
            resolved: false,
            resolved_at: None,
        };

        // Process alert
        manager.process_alert(alert.clone()).await;

        // Check active alerts
        let active = manager.get_active_alerts().await;
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "test_alert");

        // Resolve alert
        let resolved = manager.resolve_alert("test_alert").await;
        assert!(resolved);

        // Check active alerts again
        let active = manager.get_active_alerts().await;
        assert_eq!(active.len(), 0);
    }

    #[test]
    fn test_alert_thresholds_default() {
        let thresholds = AlertThresholds::default();
        assert_eq!(thresholds.cpu_usage_percent, Some(80.0));
        assert_eq!(thresholds.memory_usage_percent, Some(85.0));
        assert_eq!(thresholds.min_real_time_factor, Some(0.8));
    }

    #[test]
    fn test_recovery_actions() {
        let action = RecoveryAction::ReduceBatchSize { min_size: 8 };
        match action {
            RecoveryAction::ReduceBatchSize { min_size } => {
                assert_eq!(min_size, 8);
            }
            _ => panic!("Wrong action type"),
        }
    }

    #[tokio::test]
    async fn test_monitor_start_stop() {
        let config = MonitorConfig::default();
        let mut monitor = PerformanceMonitor::new(config);

        assert!(!monitor.is_running().await);

        monitor.start().await.unwrap();
        assert!(monitor.is_running().await);

        monitor.stop().await.unwrap();
        assert!(!monitor.is_running().await);
    }
}
