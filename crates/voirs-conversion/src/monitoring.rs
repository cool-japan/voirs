//! Real-time quality monitoring system for voice conversion
//!
//! This module provides real-time monitoring capabilities for tracking quality metrics,
//! detecting performance issues, and generating alerts during voice conversion operations.

use crate::quality::{ArtifactDetector, ArtifactType, DetectedArtifacts, QualityAssessment};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// Real-time quality monitoring system
#[derive(Debug)]
pub struct QualityMonitor {
    /// Configuration for monitoring behavior
    config: MonitorConfig,
    /// Quality metrics collector
    collector: Arc<Mutex<MetricsCollector>>,
    /// Alert system for quality issues
    alerter: Arc<Mutex<AlertSystem>>,
    /// Background monitoring task handle
    monitor_task: Option<JoinHandle<()>>,
    /// Channel for sending quality data
    quality_sender: Option<mpsc::UnboundedSender<QualityEvent>>,
    /// Performance tracker
    performance_tracker: Arc<Mutex<PerformanceTracker>>,
}

/// Configuration for quality monitoring
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Monitoring interval in milliseconds
    pub monitoring_interval_ms: u64,
    /// Maximum history length for trend analysis
    pub max_history_length: usize,
    /// Quality threshold for alerts (0.0 to 1.0)
    pub quality_alert_threshold: f32,
    /// Latency threshold for alerts in milliseconds
    pub latency_alert_threshold_ms: u64,
    /// Enable artifact detection alerts
    pub enable_artifact_alerts: bool,
    /// Enable performance tracking
    pub enable_performance_tracking: bool,
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
    /// Report generation interval in seconds
    pub report_interval_seconds: u64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            monitoring_interval_ms: 100,
            max_history_length: 10000,
            quality_alert_threshold: 0.7,
            latency_alert_threshold_ms: 100,
            enable_artifact_alerts: true,
            enable_performance_tracking: true,
            enable_trend_analysis: true,
            report_interval_seconds: 300, // 5 minutes
        }
    }
}

/// Quality event types for monitoring
#[derive(Debug, Clone)]
pub enum QualityEvent {
    /// Quality update with metrics
    QualityUpdate {
        /// Timestamp when the quality measurement was taken
        timestamp: Instant,
        /// Unique identifier for the conversion session
        session_id: String,
        /// Overall quality score from 0.0 to 1.0
        overall_quality: f32,
        /// Detected audio artifacts if any
        artifacts: Option<DetectedArtifacts>,
        /// Processing latency in milliseconds
        processing_latency_ms: u64,
        /// Additional quality metrics as key-value pairs
        metadata: HashMap<String, f32>,
    },
    /// Performance update
    PerformanceUpdate {
        /// Timestamp when the performance measurement was taken
        timestamp: Instant,
        /// Unique identifier for the conversion session
        session_id: String,
        /// CPU usage as a percentage (0.0 to 100.0)
        cpu_usage_percent: f32,
        /// Memory usage in megabytes
        memory_usage_mb: f64,
        /// Processing throughput in samples per second
        throughput_samples_per_sec: f64,
        /// Current processing queue length
        queue_length: usize,
    },
    /// Quality alert
    QualityAlert {
        /// The quality alert to be processed
        alert: QualityAlert,
    },
    /// System status update
    SystemStatus {
        /// Timestamp of the status update
        timestamp: Instant,
        /// Current system status
        status: SystemStatus,
    },
}

/// Quality alert structure
#[derive(Debug, Clone)]
pub struct QualityAlert {
    /// Timestamp when the alert was generated
    pub timestamp: Instant,
    /// Session identifier associated with this alert
    pub session_id: String,
    /// Type of alert being raised
    pub alert_type: AlertType,
    /// Severity level of the alert
    pub severity: AlertSeverity,
    /// Human-readable alert message
    pub message: String,
    /// Optional suggested action to resolve the issue
    pub suggested_action: Option<String>,
    /// Additional metadata about the alert as key-value pairs
    pub metadata: HashMap<String, String>,
}

/// Alert types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    /// Quality score has fallen below acceptable threshold
    QualityDegradation,
    /// Processing latency exceeds configured limits
    HighLatency,
    /// Audio artifacts detected in output
    ArtifactsDetected,
    /// General performance degradation detected
    PerformanceIssue,
    /// System resources are critically overloaded
    SystemOverload,
    /// Memory usage approaching system limits
    MemoryPressure,
    /// Conversion session has failed
    SessionFailure,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert, no action required
    Info,
    /// Warning level, attention recommended
    Warning,
    /// Critical alert requiring immediate attention
    Critical,
}

/// System status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemStatus {
    /// System operating normally with good performance
    Healthy,
    /// System performance is degraded but functional
    Degraded,
    /// System resources are overloaded, quality may suffer
    Overloaded,
    /// System is experiencing failures
    Failing,
}

/// Metrics collector
#[derive(Debug)]
pub struct MetricsCollector {
    /// Quality history
    quality_history: VecDeque<QualityDataPoint>,
    /// Per-session metrics
    session_metrics: HashMap<String, SessionMetrics>,
    /// Aggregate statistics
    aggregate_stats: AggregateStats,
    /// Trend analyzer
    trend_analyzer: TrendAnalyzer,
}

/// Quality data point
#[derive(Debug, Clone)]
pub struct QualityDataPoint {
    /// Timestamp of this quality measurement
    pub timestamp: Instant,
    /// Session identifier for this measurement
    pub session_id: String,
    /// Overall quality score from 0.0 to 1.0
    pub overall_quality: f32,
    /// Aggregate artifact severity score
    pub artifact_score: f32,
    /// Processing latency in milliseconds
    pub processing_latency_ms: u64,
    /// Severity scores for each detected artifact type
    pub artifacts_by_type: HashMap<ArtifactType, f32>,
    /// Additional quality metrics as key-value pairs
    pub metadata: HashMap<String, f32>,
}

/// Per-session metrics
#[derive(Debug, Clone)]
pub struct SessionMetrics {
    /// Unique session identifier
    pub session_id: String,
    /// When the session started
    pub start_time: Instant,
    /// Total number of audio samples processed
    pub samples_processed: u64,
    /// Average quality score for this session
    pub average_quality: f32,
    /// Recent quality scores showing trend over time
    pub quality_trend: VecDeque<f32>,
    /// Count of each artifact type detected
    pub artifact_counts: HashMap<ArtifactType, u64>,
    /// Performance metrics for this session
    pub performance_metrics: PerformanceMetrics,
    /// Alerts generated during this session
    pub alerts: Vec<QualityAlert>,
}

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Average processing latency in milliseconds
    pub average_latency_ms: f64,
    /// Peak processing latency in milliseconds
    pub peak_latency_ms: u64,
    /// Processing throughput in samples per second
    pub throughput_samples_per_sec: f64,
    /// CPU usage as a percentage (0.0 to 100.0)
    pub cpu_usage_percent: f32,
    /// Memory usage in megabytes
    pub memory_usage_mb: f64,
}

/// Aggregate statistics
#[derive(Debug, Clone, Default)]
pub struct AggregateStats {
    /// Total number of quality data points collected
    pub total_points: u64,
    /// Overall average quality across all measurements
    pub overall_avg_quality: f32,
    /// Variance in quality measurements
    pub quality_variance: f32,
    /// Average processing latency in milliseconds
    pub average_latency_ms: f64,
    /// Average throughput in samples per second
    pub throughput_samples_per_sec: f64,
}

/// Trend analyzer
#[derive(Debug, Default)]
pub struct TrendAnalyzer {
    /// Current quality trend direction
    pub quality_trend: TrendDirection,
    /// Current latency trend direction
    pub latency_trend: TrendDirection,
    /// Current throughput trend direction
    pub throughput_trend: TrendDirection,
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Trend direction not yet determined
    Unknown,
    /// Metric is improving over time
    Improving,
    /// Metric is stable with minimal variation
    Stable,
    /// Metric is degrading over time
    Degrading,
}

impl Default for TrendDirection {
    fn default() -> Self {
        TrendDirection::Unknown
    }
}

/// Alert system
#[derive(Debug)]
pub struct AlertSystem {
    config: AlertConfig,
    active_alerts: HashMap<String, Vec<QualityAlert>>,
    alert_history: VecDeque<QualityAlert>,
    handlers: Vec<Box<dyn AlertHandler>>,
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Minimum seconds between duplicate alerts
    pub alert_cooldown_seconds: u64,
    /// Maximum number of alerts to keep in history
    pub max_alert_history: usize,
    /// Enable email notifications for alerts
    pub enable_email_alerts: bool,
    /// Enable Slack notifications for alerts
    pub enable_slack_alerts: bool,
    /// Enable webhook notifications for alerts
    pub enable_webhook_alerts: bool,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            alert_cooldown_seconds: 60,
            max_alert_history: 1000,
            enable_email_alerts: false,
            enable_slack_alerts: false,
            enable_webhook_alerts: false,
        }
    }
}

/// Alert handler trait
pub trait AlertHandler: Send + Sync + std::fmt::Debug {
    /// Process and handle a quality alert
    fn handle_alert(&mut self, alert: &QualityAlert) -> Result<()>;
    /// Get the name of this alert handler
    fn name(&self) -> &str;
}

/// Logging alert handler
#[derive(Debug)]
pub struct LoggingAlertHandler;

impl AlertHandler for LoggingAlertHandler {
    fn handle_alert(&mut self, alert: &QualityAlert) -> Result<()> {
        match alert.severity {
            AlertSeverity::Info => {
                info!("[ALERT] {}: {}", alert.alert_type.as_str(), alert.message)
            }
            AlertSeverity::Warning => {
                warn!("[ALERT] {}: {}", alert.alert_type.as_str(), alert.message)
            }
            AlertSeverity::Critical => {
                error!("[ALERT] {}: {}", alert.alert_type.as_str(), alert.message)
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "logging"
    }
}

/// Performance tracker
#[derive(Debug, Default)]
pub struct PerformanceTracker {
    /// When performance tracking started
    pub start_time: Option<Instant>,
    /// Current number of active conversion sessions
    pub active_sessions: usize,
    /// Total number of sessions since start
    pub total_sessions: u64,
    /// Current system resource usage
    pub system_resources: SystemResources,
    /// Historical performance trend data
    pub performance_trends: PerformanceTrends,
}

/// Performance trends
#[derive(Debug, Default)]
pub struct PerformanceTrends {
    /// Historical CPU usage measurements
    pub cpu_trend: VecDeque<f32>,
    /// Historical memory usage measurements
    pub memory_trend: VecDeque<f64>,
    /// Historical throughput measurements
    pub throughput_trend: VecDeque<f64>,
}

/// System resource usage data
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SystemResources {
    /// CPU usage as a percentage (0.0 to 100.0)
    pub cpu_usage_percent: f32,
    /// Memory usage in megabytes
    pub memory_usage_mb: f64,
    /// GPU usage percentage if GPU is available
    pub gpu_usage_percent: Option<f32>,
    /// Disk I/O rate in megabytes per second
    pub disk_io_mb_per_sec: f64,
    /// Network I/O rate in megabytes per second
    pub network_io_mb_per_sec: f64,
}

/// Dashboard data structure for real-time monitoring visualization
#[derive(Debug)]
pub struct QualityDashboard {
    /// Dashboard data for each active session
    pub current_sessions: HashMap<String, SessionDashboard>,
    /// System-wide overview metrics
    pub system_overview: SystemOverview,
    /// Historical trend data for visualization
    pub trends: DashboardTrends,
    /// Most recent alerts for display
    pub recent_alerts: VecDeque<QualityAlert>,
}

/// Per-session dashboard data
#[derive(Debug, Clone)]
pub struct SessionDashboard {
    /// Unique session identifier
    pub session_id: String,
    /// When this session started
    pub start_time: Instant,
    /// Most recent quality score
    pub current_quality: f32,
    /// Recent quality scores for trend visualization
    pub quality_trend: Vec<f32>,
    /// Current processing latency in milliseconds
    pub current_latency_ms: u64,
    /// Current throughput in samples per second
    pub throughput_samples_per_sec: f64,
    /// Currently detected artifact types
    pub active_artifacts: Vec<ArtifactType>,
    /// Total number of alerts for this session
    pub alert_count: usize,
}

/// System overview for dashboard
#[derive(Debug, Clone, Default)]
pub struct SystemOverview {
    /// Number of currently active sessions
    pub active_sessions: usize,
    /// Total sessions processed today
    pub total_sessions_today: u64,
    /// Average quality score across all sessions
    pub average_quality: f32,
    /// Overall system load as a percentage
    pub system_load_percent: f32,
    /// Total memory usage in megabytes
    pub memory_usage_mb: f64,
    /// System uptime in hours
    pub uptime_hours: f64,
    /// Number of alerts in the last hour
    pub alerts_last_hour: usize,
}

/// Trend data for dashboard visualization
#[derive(Debug, Default)]
pub struct DashboardTrends {
    /// Quality measurements over time for trend charts
    pub quality_over_time: Vec<(Instant, f32)>,
    /// Latency measurements over time for trend charts
    pub latency_over_time: Vec<(Instant, f64)>,
    /// Throughput measurements over time for trend charts
    pub throughput_over_time: Vec<(Instant, f64)>,
    /// System resource usage over time for trend charts
    pub resource_usage_over_time: Vec<(Instant, SystemResources)>,
}

/// Default implementation
impl Default for QualityMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// Implementation of main QualityMonitor
impl QualityMonitor {
    /// Create new quality monitor with default configuration
    pub fn new() -> Self {
        Self::with_config(MonitorConfig::default())
    }

    /// Create new quality monitor with custom configuration
    pub fn with_config(config: MonitorConfig) -> Self {
        let collector = Arc::new(Mutex::new(MetricsCollector::new(config.max_history_length)));
        let alerter = Arc::new(Mutex::new(AlertSystem::new()));
        let performance_tracker = Arc::new(Mutex::new(PerformanceTracker::default()));

        Self {
            config,
            collector,
            alerter,
            monitor_task: None,
            quality_sender: None,
            performance_tracker,
        }
    }

    /// Start real-time monitoring
    pub async fn start_monitoring(&mut self) -> Result<()> {
        if self.monitor_task.is_some() {
            return Err(Error::validation("Monitoring already started".to_string()));
        }

        let (sender, mut receiver) = mpsc::unbounded_channel();
        self.quality_sender = Some(sender);

        let collector = Arc::clone(&self.collector);
        let alerter = Arc::clone(&self.alerter);
        let performance_tracker = Arc::clone(&self.performance_tracker);
        let config = self.config.clone();

        let monitor_task = tokio::spawn(async move {
            let mut report_timer =
                tokio::time::interval(Duration::from_secs(config.report_interval_seconds));

            loop {
                tokio::select! {
                    // Handle incoming quality events
                    event = receiver.recv() => {
                        match event {
                            Some(quality_event) => {
                                Self::process_quality_event(
                                    quality_event,
                                    &collector,
                                    &alerter,
                                    &performance_tracker,
                                    &config,
                                ).await;
                            }
                            None => {
                                info!("Quality monitoring channel closed");
                                break;
                            }
                        }
                    }

                    // Generate periodic reports
                    _ = report_timer.tick() => {
                        if let Err(e) = Self::generate_periodic_report(&collector, &performance_tracker).await {
                            error!("Failed to generate periodic report: {}", e);
                        }
                    }
                }
            }
        });

        self.monitor_task = Some(monitor_task);

        info!(
            "Quality monitoring started with interval {}ms",
            self.config.monitoring_interval_ms
        );
        Ok(())
    }

    /// Stop monitoring
    pub async fn stop_monitoring(&mut self) -> Result<()> {
        if let Some(task) = self.monitor_task.take() {
            task.abort();
            info!("Quality monitoring stopped");
        }

        self.quality_sender = None;
        Ok(())
    }

    /// Submit quality data for monitoring
    pub fn submit_quality_data(
        &self,
        session_id: String,
        overall_quality: f32,
        artifacts: Option<DetectedArtifacts>,
        processing_latency_ms: u64,
        metadata: HashMap<String, f32>,
    ) -> Result<()> {
        if let Some(sender) = &self.quality_sender {
            let event = QualityEvent::QualityUpdate {
                timestamp: Instant::now(),
                session_id,
                overall_quality,
                artifacts,
                processing_latency_ms,
                metadata,
            };

            sender
                .send(event)
                .map_err(|e| Error::runtime(format!("Failed to submit quality data: {e}")))?;
        }

        Ok(())
    }

    /// Submit performance data for monitoring
    pub fn submit_performance_data(
        &self,
        session_id: String,
        cpu_usage_percent: f32,
        memory_usage_mb: f64,
        throughput_samples_per_sec: f64,
        queue_length: usize,
    ) -> Result<()> {
        if let Some(sender) = &self.quality_sender {
            let event = QualityEvent::PerformanceUpdate {
                timestamp: Instant::now(),
                session_id,
                cpu_usage_percent,
                memory_usage_mb,
                throughput_samples_per_sec,
                queue_length,
            };

            sender
                .send(event)
                .map_err(|e| Error::runtime(format!("Failed to submit performance data: {e}")))?;
        }

        Ok(())
    }

    /// Get current dashboard data
    pub async fn get_dashboard(&self) -> Result<QualityDashboard> {
        let collector = self.collector.lock().unwrap();
        let alerter = self.alerter.lock().unwrap();
        let performance_tracker = self.performance_tracker.lock().unwrap();

        let mut current_sessions = HashMap::new();

        for (session_id, metrics) in &collector.session_metrics {
            current_sessions.insert(
                session_id.clone(),
                SessionDashboard {
                    session_id: session_id.clone(),
                    start_time: metrics.start_time,
                    current_quality: metrics.quality_trend.back().copied().unwrap_or(0.0),
                    quality_trend: metrics.quality_trend.iter().copied().collect(),
                    current_latency_ms: metrics.performance_metrics.peak_latency_ms,
                    throughput_samples_per_sec: metrics
                        .performance_metrics
                        .throughput_samples_per_sec,
                    active_artifacts: metrics.artifact_counts.keys().copied().collect(),
                    alert_count: metrics.alerts.len(),
                },
            );
        }

        let system_overview = SystemOverview {
            active_sessions: collector.session_metrics.len(), // Use actual session count
            total_sessions_today: performance_tracker.total_sessions,
            average_quality: collector.aggregate_stats.overall_avg_quality,
            system_load_percent: performance_tracker.system_resources.cpu_usage_percent,
            memory_usage_mb: performance_tracker.system_resources.memory_usage_mb,
            uptime_hours: performance_tracker
                .start_time
                .map(|start| start.elapsed().as_secs_f64() / 3600.0)
                .unwrap_or(0.0),
            alerts_last_hour: alerter
                .alert_history
                .iter()
                .filter(|alert| alert.timestamp.elapsed() < Duration::from_secs(3600))
                .count(),
        };

        let trends = DashboardTrends {
            quality_over_time: collector
                .quality_history
                .iter()
                .map(|point| (point.timestamp, point.overall_quality))
                .collect(),
            latency_over_time: collector
                .quality_history
                .iter()
                .map(|point| (point.timestamp, point.processing_latency_ms as f64))
                .collect(),
            throughput_over_time: {
                // Build throughput tracking from performance trends and quality history
                let mut throughput_data = Vec::new();
                let throughput_trends = &performance_tracker.performance_trends.throughput_trend;

                if !throughput_trends.is_empty() && !collector.quality_history.is_empty() {
                    let trend_count = throughput_trends.len();
                    let history_count = collector.quality_history.len();

                    // Use quality history timestamps as reference points for throughput data
                    let step_size = if history_count >= trend_count {
                        history_count / trend_count
                    } else {
                        1
                    };

                    for (i, &throughput) in throughput_trends.iter().enumerate() {
                        let history_index = (i * step_size).min(history_count.saturating_sub(1));
                        if let Some(quality_point) = collector.quality_history.get(history_index) {
                            throughput_data.push((quality_point.timestamp, throughput));
                        }
                    }
                }
                throughput_data
            },
            resource_usage_over_time: {
                // Build resource usage tracking from performance trends
                let mut resource_data = Vec::new();
                let cpu_trends = &performance_tracker.performance_trends.cpu_trend;
                let memory_trends = &performance_tracker.performance_trends.memory_trend;

                if let Some(start_time) = performance_tracker.start_time {
                    let trend_count = cpu_trends.len().min(memory_trends.len());

                    // Create evenly distributed timestamps based on system start time
                    for i in 0..trend_count {
                        let timestamp = start_time + Duration::from_secs((i as u64) * 60); // 1-minute intervals
                        let cpu_usage = cpu_trends.get(i).copied().unwrap_or(0.0);
                        let memory_usage = memory_trends.get(i).copied().unwrap_or(0.0);

                        let resources = SystemResources {
                            cpu_usage_percent: cpu_usage,
                            memory_usage_mb: memory_usage,
                            gpu_usage_percent: performance_tracker
                                .system_resources
                                .gpu_usage_percent,
                            disk_io_mb_per_sec: performance_tracker
                                .system_resources
                                .disk_io_mb_per_sec,
                            network_io_mb_per_sec: performance_tracker
                                .system_resources
                                .network_io_mb_per_sec,
                        };
                        resource_data.push((timestamp, resources));
                    }
                }
                resource_data
            },
        };

        let recent_alerts = alerter
            .alert_history
            .iter()
            .rev()
            .take(20)
            .cloned()
            .collect();

        Ok(QualityDashboard {
            current_sessions,
            system_overview,
            trends,
            recent_alerts,
        })
    }

    // Private implementation methods

    async fn process_quality_event(
        event: QualityEvent,
        collector: &Arc<Mutex<MetricsCollector>>,
        alerter: &Arc<Mutex<AlertSystem>>,
        performance_tracker: &Arc<Mutex<PerformanceTracker>>,
        config: &MonitorConfig,
    ) {
        match event {
            QualityEvent::QualityUpdate {
                timestamp,
                session_id,
                overall_quality,
                artifacts,
                processing_latency_ms,
                metadata,
            } => {
                // Update collector and check for new session
                let is_new_session = {
                    let mut collector_guard = collector.lock().unwrap();
                    let data_point = QualityDataPoint {
                        timestamp,
                        session_id: session_id.clone(),
                        overall_quality,
                        artifact_score: artifacts.as_ref().map(|a| a.overall_score).unwrap_or(0.0),
                        processing_latency_ms,
                        artifacts_by_type: artifacts
                            .as_ref()
                            .map(|a| a.artifact_types.clone())
                            .unwrap_or_default(),
                        metadata,
                    };
                    collector_guard.add_quality_data_point(data_point)
                };

                // Update active session count if this is a new session
                if is_new_session {
                    let mut tracker_guard = performance_tracker.lock().unwrap();
                    tracker_guard.active_sessions += 1;
                    tracker_guard.total_sessions += 1;
                }

                // Check for quality alerts
                if overall_quality < config.quality_alert_threshold {
                    let alert = QualityAlert {
                        timestamp,
                        session_id: session_id.clone(),
                        alert_type: AlertType::QualityDegradation,
                        severity: if overall_quality < 0.2 {
                            AlertSeverity::Critical
                        } else {
                            AlertSeverity::Warning
                        },
                        message: format!("Quality degraded to {overall_quality:.2}"),
                        suggested_action: Some(
                            "Consider adjusting conversion parameters".to_string(),
                        ),
                        metadata: HashMap::new(),
                    };

                    let mut alerter_guard = alerter.lock().unwrap();
                    alerter_guard.add_alert(alert);
                }

                // Check for latency alerts
                if processing_latency_ms > config.latency_alert_threshold_ms {
                    let alert = QualityAlert {
                        timestamp,
                        session_id: session_id.clone(),
                        alert_type: AlertType::HighLatency,
                        severity: AlertSeverity::Warning,
                        message: format!("High processing latency: {processing_latency_ms}ms"),
                        suggested_action: Some(
                            "Check system resources and consider load balancing".to_string(),
                        ),
                        metadata: HashMap::new(),
                    };

                    let mut alerter_guard = alerter.lock().unwrap();
                    alerter_guard.add_alert(alert);
                }
            }

            QualityEvent::PerformanceUpdate {
                timestamp: _,
                session_id: _,
                cpu_usage_percent,
                memory_usage_mb,
                throughput_samples_per_sec,
                queue_length,
            } => {
                let mut tracker_guard = performance_tracker.lock().unwrap();
                tracker_guard.update_performance_metrics(
                    cpu_usage_percent,
                    memory_usage_mb,
                    throughput_samples_per_sec,
                    queue_length,
                );
            }

            QualityEvent::QualityAlert { .. } => {
                // Handle external alerts
            }

            QualityEvent::SystemStatus { .. } => {
                // Handle system status updates
            }
        }
    }

    async fn generate_periodic_report(
        collector: &Arc<Mutex<MetricsCollector>>,
        performance_tracker: &Arc<Mutex<PerformanceTracker>>,
    ) -> Result<()> {
        let (stats, performance_stats) = {
            let collector_guard = collector.lock().unwrap();
            let tracker_guard = performance_tracker.lock().unwrap();
            (
                collector_guard.aggregate_stats.clone(),
                tracker_guard.system_resources.clone(),
            )
        };

        info!("=== Quality Monitoring Report ===");
        info!("Total data points: {}", stats.total_points);
        info!("Average quality: {:.3}", stats.overall_avg_quality);
        info!(
            "System CPU usage: {:.1}%",
            performance_stats.cpu_usage_percent
        );
        info!(
            "System memory usage: {:.1} MB",
            performance_stats.memory_usage_mb
        );

        Ok(())
    }
}

// Implementation of helper structs
impl MetricsCollector {
    fn new(max_history_length: usize) -> Self {
        Self {
            quality_history: VecDeque::with_capacity(max_history_length),
            session_metrics: HashMap::new(),
            aggregate_stats: AggregateStats::default(),
            trend_analyzer: TrendAnalyzer::default(),
        }
    }

    fn add_quality_data_point(&mut self, data_point: QualityDataPoint) -> bool {
        let session_id = data_point.session_id.clone();
        let timestamp = data_point.timestamp;
        let artifacts_by_type = data_point.artifacts_by_type.clone();
        let overall_quality = data_point.overall_quality;

        self.quality_history.push_back(data_point);
        if self.quality_history.len() > self.quality_history.capacity() {
            self.quality_history.pop_front();
        }

        // Update session metrics
        let is_new_session = !self.session_metrics.contains_key(&session_id);
        let session_metrics = self
            .session_metrics
            .entry(session_id.clone())
            .or_insert_with(|| SessionMetrics {
                session_id: session_id.clone(),
                start_time: timestamp,
                samples_processed: 0,
                average_quality: overall_quality,
                quality_trend: VecDeque::with_capacity(100),
                artifact_counts: HashMap::new(),
                performance_metrics: PerformanceMetrics::default(),
                alerts: Vec::new(),
            });

        session_metrics.samples_processed += 1;
        session_metrics.quality_trend.push_back(overall_quality);
        if session_metrics.quality_trend.len() > 100 {
            session_metrics.quality_trend.pop_front();
        }

        // Update artifact counts
        for artifact_type in artifacts_by_type.keys() {
            *session_metrics
                .artifact_counts
                .entry(*artifact_type)
                .or_insert(0) += 1;
        }

        // Update aggregate statistics
        self.update_aggregate_stats(overall_quality);

        // Return whether this was a new session
        is_new_session
    }

    fn update_aggregate_stats(&mut self, overall_quality: f32) {
        self.aggregate_stats.total_points += 1;

        // Update running average
        let n = self.aggregate_stats.total_points as f32;
        self.aggregate_stats.overall_avg_quality =
            (self.aggregate_stats.overall_avg_quality * (n - 1.0) + overall_quality) / n;
    }
}

impl AlertSystem {
    fn new() -> Self {
        let handlers: Vec<Box<dyn AlertHandler>> = vec![Box::new(LoggingAlertHandler)];

        Self {
            config: AlertConfig::default(),
            active_alerts: HashMap::new(),
            alert_history: VecDeque::with_capacity(1000),
            handlers,
        }
    }

    fn add_alert(&mut self, alert: QualityAlert) {
        // Check cooldown period
        let should_add = self
            .alert_history
            .iter()
            .filter(|existing| {
                existing.session_id == alert.session_id
                    && existing.alert_type.as_str() == alert.alert_type.as_str()
            })
            .all(|existing| {
                existing.timestamp.elapsed()
                    > Duration::from_secs(self.config.alert_cooldown_seconds)
            });

        if should_add {
            // Add to active alerts
            self.active_alerts
                .entry(alert.session_id.clone())
                .or_default()
                .push(alert.clone());

            // Add to history
            self.alert_history.push_back(alert.clone());
            if self.alert_history.len() > self.config.max_alert_history {
                self.alert_history.pop_front();
            }

            // Notify handlers
            for handler in &mut self.handlers {
                if let Err(e) = handler.handle_alert(&alert) {
                    error!("Alert handler '{}' failed: {}", handler.name(), e);
                }
            }
        }
    }
}

impl PerformanceTracker {
    fn update_performance_metrics(
        &mut self,
        cpu_usage_percent: f32,
        memory_usage_mb: f64,
        throughput_samples_per_sec: f64,
        _queue_length: usize,
    ) {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        self.system_resources.cpu_usage_percent = cpu_usage_percent;
        self.system_resources.memory_usage_mb = memory_usage_mb;

        // Update trends
        self.performance_trends
            .throughput_trend
            .push_back(throughput_samples_per_sec);
        if self.performance_trends.throughput_trend.len() > 100 {
            self.performance_trends.throughput_trend.pop_front();
        }
    }
}

// Utility implementations
impl AlertType {
    /// Get the string representation of this alert type
    pub fn as_str(&self) -> &'static str {
        match self {
            AlertType::QualityDegradation => "quality_degradation",
            AlertType::HighLatency => "high_latency",
            AlertType::ArtifactsDetected => "artifacts_detected",
            AlertType::PerformanceIssue => "performance_issue",
            AlertType::SystemOverload => "system_overload",
            AlertType::MemoryPressure => "memory_pressure",
            AlertType::SessionFailure => "session_failure",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_config_default() {
        let config = MonitorConfig::default();
        assert_eq!(config.monitoring_interval_ms, 100);
        assert_eq!(config.quality_alert_threshold, 0.7);
    }

    #[test]
    fn test_alert_type_as_str() {
        assert_eq!(
            AlertType::QualityDegradation.as_str(),
            "quality_degradation"
        );
        assert_eq!(AlertType::HighLatency.as_str(), "high_latency");
    }

    #[tokio::test]
    async fn test_quality_monitor_creation() {
        let monitor = QualityMonitor::new();
        assert!(monitor.monitor_task.is_none());
        assert!(monitor.quality_sender.is_none());
    }
}
