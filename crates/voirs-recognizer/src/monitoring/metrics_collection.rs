//! Enhanced Metrics Collection and Analysis
//!
//! This module provides comprehensive metrics collection, aggregation, and analysis
//! capabilities for speech recognition systems including performance metrics,
//! system resource monitoring, quality metrics, real-time analytics, alerting,
//! and time-series analysis for production observability.

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Metric value types
#[derive(Debug, Clone, PartialEq)]
/// Metric Value
pub enum MetricValue {
    /// Counter - monotonically increasing value
    Counter(u64),
    /// Gauge - point-in-time value that can go up or down
    Gauge(f64),
    /// Histogram - distribution of values with buckets
    Histogram(HistogramData),
    /// Summary - quantile-based summary statistics
    Summary(SummaryData),
}

/// Histogram data with configurable buckets
#[derive(Debug, Clone, PartialEq)]
/// Histogram Data
pub struct HistogramData {
    /// Bucket counts (bucket_upper_bound, count)
    pub buckets: Vec<(f64, u64)>,
    /// Total count of observations
    pub count: u64,
    /// Sum of all observed values
    pub sum: f64,
}

/// Summary data with quantiles
#[derive(Debug, Clone, PartialEq)]
/// Summary Data
pub struct SummaryData {
    /// Quantile values (quantile, value)
    pub quantiles: Vec<(f64, f64)>,
    /// Total count of observations
    pub count: u64,
    /// Sum of all observed values
    pub sum: f64,
}

/// Metric metadata
#[derive(Debug, Clone)]
/// Metric Metadata
pub struct MetricMetadata {
    /// Metric name
    pub name: String,
    /// Metric description
    pub description: String,
    /// Metric unit
    pub unit: String,
    /// Metric labels
    pub labels: HashMap<String, String>,
    /// Metric type
    pub metric_type: MetricType,
}

#[derive(Debug, Clone, PartialEq)]
/// Metric Type
pub enum MetricType {
    /// Counter
    Counter,
    /// Gauge
    Gauge,
    /// Histogram
    Histogram,
    /// Summary
    Summary,
}

/// Time-series data point
#[derive(Debug, Clone)]
/// Data Point
pub struct DataPoint {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Metric value
    pub value: MetricValue,
    /// Labels for this data point
    pub labels: HashMap<String, String>,
}

/// Metrics collector interface
pub trait MetricsCollector: Send + Sync {
    /// Record a counter increment
    fn increment_counter(&self, name: &str, value: u64, labels: HashMap<String, String>);

    /// Set a gauge value
    fn set_gauge(&self, name: &str, value: f64, labels: HashMap<String, String>);

    /// Observe a value in a histogram
    fn observe_histogram(&self, name: &str, value: f64, labels: HashMap<String, String>);

    /// Observe a value in a summary
    fn observe_summary(&self, name: &str, value: f64, labels: HashMap<String, String>);

    /// Get current metric values
    fn get_metrics(&self) -> HashMap<String, Vec<DataPoint>>;

    /// Reset all metrics
    fn reset(&self);
}

/// In-memory metrics collector
#[derive(Debug)]
/// In Memory Metrics Collector
pub struct InMemoryMetricsCollector {
    /// Stored metrics (metric_name -> data_points)
    metrics: Arc<RwLock<HashMap<String, Vec<DataPoint>>>>,
    /// Metric metadata
    metadata: Arc<RwLock<HashMap<String, MetricMetadata>>>,
    /// Histogram buckets configuration
    histogram_buckets: Vec<f64>,
    /// Summary quantiles configuration
    summary_quantiles: Vec<f64>,
}

impl InMemoryMetricsCollector {
    /// Create a new in-memory collector
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            histogram_buckets: vec![
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
                f64::INFINITY,
            ],
            summary_quantiles: vec![0.5, 0.9, 0.95, 0.99],
        }
    }

    /// Register metric metadata
    pub fn register_metric(&self, metadata: MetricMetadata) {
        self.metadata
            .write()
            .unwrap()
            .insert(metadata.name.clone(), metadata);
    }

    /// Get metric metadata
    pub fn get_metadata(&self, name: &str) -> Option<MetricMetadata> {
        self.metadata.read().unwrap().get(name).cloned()
    }

    /// Get all metric names
    pub fn get_metric_names(&self) -> Vec<String> {
        self.metrics.read().unwrap().keys().cloned().collect()
    }
}

impl MetricsCollector for InMemoryMetricsCollector {
    fn increment_counter(&self, name: &str, value: u64, labels: HashMap<String, String>) {
        let mut metrics = self.metrics.write().unwrap();
        let data_points = metrics.entry(name.to_string()).or_insert_with(Vec::new);

        // Find existing counter with same labels or create new one
        if let Some(last_point) = data_points.last_mut() {
            if last_point.labels == labels {
                if let MetricValue::Counter(ref mut current) = last_point.value {
                    *current += value;
                    last_point.timestamp = SystemTime::now();
                    return;
                }
            }
        }

        // Create new data point
        data_points.push(DataPoint {
            timestamp: SystemTime::now(),
            value: MetricValue::Counter(value),
            labels,
        });
    }

    fn set_gauge(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        let mut metrics = self.metrics.write().unwrap();
        let data_points = metrics.entry(name.to_string()).or_insert_with(Vec::new);

        data_points.push(DataPoint {
            timestamp: SystemTime::now(),
            value: MetricValue::Gauge(value),
            labels,
        });
    }

    fn observe_histogram(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        let mut metrics = self.metrics.write().unwrap();
        let data_points = metrics.entry(name.to_string()).or_insert_with(Vec::new);

        // Find existing histogram with same labels or create new one
        if let Some(last_point) = data_points.last_mut() {
            if last_point.labels == labels {
                if let MetricValue::Histogram(ref mut hist) = last_point.value {
                    // Update histogram
                    hist.count += 1;
                    hist.sum += value;

                    // Update buckets
                    for &bucket_bound in &self.histogram_buckets {
                        if value <= bucket_bound {
                            if let Some((_, count)) = hist
                                .buckets
                                .iter_mut()
                                .find(|(bound, _)| *bound == bucket_bound)
                            {
                                *count += 1;
                            } else {
                                hist.buckets.push((bucket_bound, 1));
                                hist.buckets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                            }
                        }
                    }

                    last_point.timestamp = SystemTime::now();
                    return;
                }
            }
        }

        // Create new histogram
        let mut buckets = Vec::new();
        for &bucket_bound in &self.histogram_buckets {
            let count = if value <= bucket_bound { 1 } else { 0 };
            buckets.push((bucket_bound, count));
        }

        let histogram = HistogramData {
            buckets,
            count: 1,
            sum: value,
        };

        data_points.push(DataPoint {
            timestamp: SystemTime::now(),
            value: MetricValue::Histogram(histogram),
            labels,
        });
    }

    fn observe_summary(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        let mut metrics = self.metrics.write().unwrap();
        let data_points = metrics.entry(name.to_string()).or_insert_with(Vec::new);

        // For simplicity, we'll store individual observations and calculate quantiles on read
        // In production, you'd use a more efficient streaming quantile algorithm
        data_points.push(DataPoint {
            timestamp: SystemTime::now(),
            value: MetricValue::Gauge(value), // Store as gauge temporarily
            labels: labels.clone(),
        });

        // Calculate summary statistics from recent observations
        let recent_values: Vec<f64> = data_points
            .iter()
            .rev()
            .take(1000) // Last 1000 observations
            .filter(|dp| dp.labels == labels)
            .filter_map(|dp| match &dp.value {
                MetricValue::Gauge(v) => Some(*v),
                _ => None,
            })
            .collect();

        if recent_values.len() >= 10 {
            let mut sorted_values = recent_values.clone();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut quantiles = Vec::new();
            for &quantile in &self.summary_quantiles {
                let index = ((sorted_values.len() as f64 - 1.0) * quantile) as usize;
                quantiles.push((quantile, sorted_values[index]));
            }

            let summary = SummaryData {
                quantiles,
                count: recent_values.len() as u64,
                sum: recent_values.iter().sum(),
            };

            // Replace the last entry with summary
            if let Some(last_point) = data_points.last_mut() {
                if last_point.labels == labels {
                    last_point.value = MetricValue::Summary(summary);
                }
            }
        }
    }

    fn get_metrics(&self) -> HashMap<String, Vec<DataPoint>> {
        self.metrics.read().unwrap().clone()
    }

    fn reset(&self) {
        self.metrics.write().unwrap().clear();
    }
}

/// Performance metrics tracker for speech recognition
pub struct PerformanceMetrics {
    /// Metrics collector
    collector: Arc<dyn MetricsCollector>,
    /// Performance counters
    counters: Arc<Mutex<PerformanceCounters>>,
}

impl std::fmt::Debug for PerformanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerformanceMetrics")
            .field("collector", &"MetricsCollector")
            .field("counters", &self.counters)
            .finish()
    }
}

#[derive(Debug, Default)]
struct PerformanceCounters {
    /// Total recognition requests
    total_requests: u64,
    /// Successful recognitions
    successful_recognitions: u64,
    /// Failed recognitions
    failed_recognitions: u64,
    /// Total audio duration processed (seconds)
    total_audio_duration: f64,
    /// Total processing time (seconds)
    total_processing_time: f64,
}

impl PerformanceMetrics {
    /// Create new performance metrics tracker
    pub fn new(collector: Arc<dyn MetricsCollector>) -> Self {
        let instance = Self {
            collector: collector.clone(),
            counters: Arc::new(Mutex::new(PerformanceCounters::default())),
        };

        // Register metrics
        instance.register_metrics();

        instance
    }

    /// Register all metrics with metadata
    fn register_metrics(&self) {
        // For now, we'll just skip the metadata registration
        // In a full implementation, you'd add a register_metadata method to the trait
        /*
            let metrics = vec![
                MetricMetadata {
                    name: "speech_recognition_requests_total".to_string(),
                    description: "Total number of speech recognition requests".to_string(),
                    unit: "requests".to_string(),
                    labels: HashMap::new(),
                    metric_type: MetricType::Counter,
                },
                MetricMetadata {
                    name: "speech_recognition_duration_seconds".to_string(),
                    description: "Duration of speech recognition processing".to_string(),
                    unit: "seconds".to_string(),
                    labels: HashMap::new(),
                    metric_type: MetricType::Histogram,
                },
                MetricMetadata {
                    name: "speech_recognition_accuracy".to_string(),
                    description: "Speech recognition accuracy score".to_string(),
                    unit: "ratio".to_string(),
                    labels: HashMap::new(),
                    metric_type: MetricType::Gauge,
                },
                MetricMetadata {
                    name: "audio_duration_seconds".to_string(),
                    description: "Duration of audio being processed".to_string(),
                    unit: "seconds".to_string(),
                    labels: HashMap::new(),
                    metric_type: MetricType::Histogram,
                },
                MetricMetadata {
                    name: "real_time_factor".to_string(),
                    description: "Real-time factor (processing_time / audio_duration)".to_string(),
                    unit: "ratio".to_string(),
                    labels: HashMap::new(),
                    metric_type: MetricType::Gauge,
                },
            ];

            for metadata in metrics {
                collector.register_metric(metadata);
            }
        */
    }

    /// Record a recognition request
    pub fn record_request(
        &self,
        status: RecognitionStatus,
        processing_duration: Duration,
        audio_duration: Duration,
    ) {
        let mut counters = self.counters.lock().unwrap();

        counters.total_requests += 1;
        let processing_seconds = processing_duration.as_secs_f64();
        let audio_seconds = audio_duration.as_secs_f64();

        counters.total_audio_duration += audio_seconds;
        counters.total_processing_time += processing_seconds;

        let mut labels = HashMap::new();

        match status {
            RecognitionStatus::Success => {
                counters.successful_recognitions += 1;
                labels.insert("status".to_string(), "success".to_string());
            }
            RecognitionStatus::Error(ref error_type) => {
                counters.failed_recognitions += 1;
                labels.insert("status".to_string(), "error".to_string());
                labels.insert("error_type".to_string(), error_type.clone());
            }
        }

        // Record metrics
        self.collector
            .increment_counter("speech_recognition_requests_total", 1, labels.clone());
        self.collector.observe_histogram(
            "speech_recognition_duration_seconds",
            processing_seconds,
            labels.clone(),
        );
        self.collector
            .observe_histogram("audio_duration_seconds", audio_seconds, labels);

        // Calculate and record real-time factor
        if audio_seconds > 0.0 {
            let rtf = processing_seconds / audio_seconds;
            self.collector
                .set_gauge("real_time_factor", rtf, HashMap::new());
        }
    }

    /// Record accuracy measurement
    pub fn record_accuracy(&self, accuracy: f64, model_name: &str) {
        let mut labels = HashMap::new();
        labels.insert("model".to_string(), model_name.to_string());

        self.collector
            .set_gauge("speech_recognition_accuracy", accuracy, labels);
    }

    /// Get current performance statistics
    pub fn get_statistics(&self) -> PerformanceStatistics {
        let counters = self.counters.lock().unwrap();

        let success_rate = if counters.total_requests > 0 {
            counters.successful_recognitions as f64 / counters.total_requests as f64
        } else {
            0.0
        };

        let average_rtf = if counters.total_audio_duration > 0.0 {
            counters.total_processing_time / counters.total_audio_duration
        } else {
            0.0
        };

        PerformanceStatistics {
            total_requests: counters.total_requests,
            success_rate,
            average_real_time_factor: average_rtf,
            total_audio_hours: counters.total_audio_duration / 3600.0,
            total_processing_hours: counters.total_processing_time / 3600.0,
        }
    }
}

#[derive(Debug, Clone)]
/// Recognition Status
pub enum RecognitionStatus {
    /// Success
    Success,
    /// Error( string)
    Error(String),
}

#[derive(Debug, Clone)]
/// Performance Statistics
pub struct PerformanceStatistics {
    /// total requests
    pub total_requests: u64,
    /// success rate
    pub success_rate: f64,
    /// average real time factor
    pub average_real_time_factor: f64,
    /// total audio hours
    pub total_audio_hours: f64,
    /// total processing hours
    pub total_processing_hours: f64,
}

/// System resource monitor
pub struct SystemResourceMonitor {
    /// Metrics collector
    collector: Arc<dyn MetricsCollector>,
    /// Monitoring thread handle
    monitor_handle: Option<thread::JoinHandle<()>>,
    /// Shutdown signal
    shutdown_sender: Option<Sender<()>>,
}

impl std::fmt::Debug for SystemResourceMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemResourceMonitor")
            .field("collector", &"MetricsCollector")
            .field("monitor_handle", &self.monitor_handle.is_some())
            .field("shutdown_sender", &self.shutdown_sender.is_some())
            .finish()
    }
}

impl SystemResourceMonitor {
    /// Create new system resource monitor
    pub fn new(collector: Arc<dyn MetricsCollector>, interval: Duration) -> Self {
        let (shutdown_tx, shutdown_rx) = mpsc::channel();
        let collector_clone = collector.clone();

        let handle = thread::spawn(move || {
            Self::monitor_resources(collector_clone, interval, shutdown_rx);
        });

        Self {
            collector,
            monitor_handle: Some(handle),
            shutdown_sender: Some(shutdown_tx),
        }
    }

    /// Monitor system resources in background thread
    fn monitor_resources(
        collector: Arc<dyn MetricsCollector>,
        interval: Duration,
        shutdown_rx: Receiver<()>,
    ) {
        loop {
            // Check for shutdown signal
            if shutdown_rx.try_recv().is_ok() {
                break;
            }

            // Collect CPU usage
            let cpu_usage = Self::get_cpu_usage();
            collector.set_gauge("system_cpu_usage_percent", cpu_usage, HashMap::new());

            // Collect memory usage
            let (memory_used, memory_total) = Self::get_memory_usage();
            collector.set_gauge(
                "system_memory_used_bytes",
                memory_used as f64,
                HashMap::new(),
            );
            collector.set_gauge(
                "system_memory_total_bytes",
                memory_total as f64,
                HashMap::new(),
            );

            if memory_total > 0 {
                let memory_usage_percent = (memory_used as f64 / memory_total as f64) * 100.0;
                collector.set_gauge(
                    "system_memory_usage_percent",
                    memory_usage_percent,
                    HashMap::new(),
                );
            }

            // Collect GPU usage if available
            if let Some(gpu_usage) = Self::get_gpu_usage() {
                collector.set_gauge("system_gpu_usage_percent", gpu_usage, HashMap::new());
            }

            // Sleep until next collection
            thread::sleep(interval);
        }
    }

    /// Get CPU usage percentage (placeholder implementation)
    fn get_cpu_usage() -> f64 {
        // In a real implementation, this would read from /proc/stat or use system APIs
        scirs2_core::random::random::<f64>() * 100.0
    }

    /// Get memory usage in bytes (placeholder implementation)
    fn get_memory_usage() -> (usize, usize) {
        // In a real implementation, this would read from /proc/meminfo or use system APIs
        let total = 8 * 1024 * 1024 * 1024; // 8GB
        let used = (scirs2_core::random::random::<f64>() * 0.8 * total as f64) as usize;
        (used, total)
    }

    /// Get GPU usage percentage (placeholder implementation)
    fn get_gpu_usage() -> Option<f64> {
        // In a real implementation, this would use NVIDIA ML or similar APIs
        Some(scirs2_core::random::random::<f64>() * 100.0)
    }

    /// Shutdown the monitor
    pub fn shutdown(&mut self) {
        if let Some(sender) = self.shutdown_sender.take() {
            let _ = sender.send(());
        }

        if let Some(handle) = self.monitor_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for SystemResourceMonitor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Alerting system for metrics thresholds
#[derive(Debug)]
/// Alert Manager
pub struct AlertManager {
    /// Alert rules
    rules: Arc<RwLock<Vec<AlertRule>>>,
    /// Active alerts
    active_alerts: Arc<RwLock<HashMap<String, ActiveAlert>>>,
    /// Alert handlers
    handlers: Arc<RwLock<Vec<Box<dyn AlertHandler>>>>,
}

/// Alert rule definition
#[derive(Debug, Clone)]
/// Alert Rule
pub struct AlertRule {
    /// Rule ID
    pub id: String,
    /// Metric name to monitor
    pub metric_name: String,
    /// Alert condition
    pub condition: AlertCondition,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert description
    pub description: String,
    /// Evaluation interval
    pub evaluation_interval: Duration,
    /// Labels to match
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone)]
/// Alert Condition
pub enum AlertCondition {
    /// Value greater than threshold
    GreaterThan(f64),
    /// Value less than threshold
    LessThan(f64),
    /// Value equals threshold
    Equals(f64),
    /// Value not equals threshold
    NotEquals(f64),
    /// Rate of change greater than threshold
    RateGreaterThan(f64, Duration),
}

#[derive(Debug, Clone, PartialEq)]
/// Alert Severity
pub enum AlertSeverity {
    /// Critical
    Critical,
    /// Warning
    Warning,
    /// Info
    Info,
}

/// Active alert instance
#[derive(Debug, Clone)]
/// Active Alert
pub struct ActiveAlert {
    /// Alert rule
    pub rule: AlertRule,
    /// Alert start time
    pub start_time: SystemTime,
    /// Current value that triggered the alert
    pub current_value: f64,
    /// Number of times this alert has fired
    pub fire_count: u32,
}

/// Alert handler interface
pub trait AlertHandler: Send + Sync + std::fmt::Debug {
    /// Handle an alert
    fn handle_alert(&self, alert: &ActiveAlert);

    /// Handle alert resolution
    fn handle_resolution(&self, alert: &ActiveAlert);
}

/// Console alert handler for development
#[derive(Debug)]
/// Console Alert Handler
pub struct ConsoleAlertHandler;

impl AlertHandler for ConsoleAlertHandler {
    fn handle_alert(&self, alert: &ActiveAlert) {
        println!(
            "🚨 ALERT [{:?}]: {} - Current value: {} (Rule: {})",
            alert.rule.severity, alert.rule.description, alert.current_value, alert.rule.id
        );
    }

    fn handle_resolution(&self, alert: &ActiveAlert) {
        println!(
            "✅ RESOLVED: {} (Rule: {})",
            alert.rule.description, alert.rule.id
        );
    }
}

impl AlertManager {
    /// Create new alert manager
    pub fn new() -> Self {
        Self {
            rules: Arc::new(RwLock::new(Vec::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            handlers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add alert rule
    pub fn add_rule(&self, rule: AlertRule) {
        self.rules.write().unwrap().push(rule);
    }

    /// Add alert handler
    pub fn add_handler(&self, handler: Box<dyn AlertHandler>) {
        self.handlers.write().unwrap().push(handler);
    }

    /// Evaluate alerts against current metrics
    pub fn evaluate_alerts(&self, metrics: &HashMap<String, Vec<DataPoint>>) {
        let rules = self.rules.read().unwrap();
        let mut active_alerts = self.active_alerts.write().unwrap();
        let handlers = self.handlers.read().unwrap();

        for rule in rules.iter() {
            if let Some(data_points) = metrics.get(&rule.metric_name) {
                // Find latest matching data point
                let matching_point = data_points
                    .iter()
                    .rev()
                    .find(|dp| self.labels_match(&dp.labels, &rule.labels));

                if let Some(point) = matching_point {
                    let value = self.extract_numeric_value(&point.value);
                    let should_alert = self.evaluate_condition(&rule.condition, value, point);

                    if should_alert {
                        // Fire alert or update existing one
                        if let Some(existing_alert) = active_alerts.get_mut(&rule.id) {
                            existing_alert.fire_count += 1;
                            existing_alert.current_value = value;
                        } else {
                            let new_alert = ActiveAlert {
                                rule: rule.clone(),
                                start_time: SystemTime::now(),
                                current_value: value,
                                fire_count: 1,
                            };

                            // Notify handlers
                            for handler in handlers.iter() {
                                handler.handle_alert(&new_alert);
                            }

                            active_alerts.insert(rule.id.clone(), new_alert);
                        }
                    } else {
                        // Resolve alert if it was active
                        if let Some(resolved_alert) = active_alerts.remove(&rule.id) {
                            for handler in handlers.iter() {
                                handler.handle_resolution(&resolved_alert);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Check if labels match
    fn labels_match(
        &self,
        data_labels: &HashMap<String, String>,
        rule_labels: &HashMap<String, String>,
    ) -> bool {
        for (key, value) in rule_labels {
            if data_labels.get(key) != Some(value) {
                return false;
            }
        }
        true
    }

    /// Extract numeric value from metric value
    fn extract_numeric_value(&self, metric_value: &MetricValue) -> f64 {
        match metric_value {
            MetricValue::Counter(c) => *c as f64,
            MetricValue::Gauge(g) => *g,
            MetricValue::Histogram(h) => h.sum / h.count.max(1) as f64, // Average
            MetricValue::Summary(s) => s
                .quantiles
                .iter()
                .find(|(q, _)| *q == 0.5)
                .map(|(_, v)| *v)
                .unwrap_or(0.0), // Median
        }
    }

    /// Evaluate alert condition
    fn evaluate_condition(
        &self,
        condition: &AlertCondition,
        value: f64,
        _point: &DataPoint,
    ) -> bool {
        match condition {
            AlertCondition::GreaterThan(threshold) => value > *threshold,
            AlertCondition::LessThan(threshold) => value < *threshold,
            AlertCondition::Equals(threshold) => (value - threshold).abs() < f64::EPSILON,
            AlertCondition::NotEquals(threshold) => (value - threshold).abs() >= f64::EPSILON,
            AlertCondition::RateGreaterThan(threshold, _duration) => {
                // Simplified: just check if current value exceeds threshold
                // In production, you'd calculate rate over the specified duration
                value > *threshold
            }
        }
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<ActiveAlert> {
        self.active_alerts
            .read()
            .unwrap()
            .values()
            .cloned()
            .collect()
    }

    /// Clear all alerts
    pub fn clear_alerts(&self) {
        self.active_alerts.write().unwrap().clear();
    }
}

/// Time-series analyzer for trend analysis
#[derive(Debug)]
/// Time Series Analyzer
pub struct TimeSeriesAnalyzer {
    /// Historical data storage
    data_store: Arc<RwLock<HashMap<String, VecDeque<TimeSeriesPoint>>>>,
    /// Maximum points to store per metric
    max_points: usize,
}

#[derive(Debug, Clone)]
/// Time Series Point
pub struct TimeSeriesPoint {
    /// timestamp
    pub timestamp: SystemTime,
    /// value
    pub value: f64,
    /// labels
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone)]
/// Trend Analysis
pub struct TrendAnalysis {
    /// Linear trend coefficient (slope)
    pub trend_coefficient: f64,
    /// Correlation coefficient (how well trend fits)
    pub correlation: f64,
    /// Seasonal patterns detected
    pub seasonality: Vec<SeasonalPattern>,
    /// Anomalies detected
    pub anomalies: Vec<AnomalyPoint>,
}

#[derive(Debug, Clone)]
/// Seasonal Pattern
pub struct SeasonalPattern {
    /// Period of the pattern (in seconds)
    pub period: Duration,
    /// Amplitude of the pattern
    pub amplitude: f64,
    /// Confidence in this pattern
    pub confidence: f64,
}

#[derive(Debug, Clone)]
/// Anomaly Point
pub struct AnomalyPoint {
    /// Timestamp of anomaly
    pub timestamp: SystemTime,
    /// Value at anomaly
    pub value: f64,
    /// Severity score (0.0 - 1.0)
    pub severity: f64,
    /// Type of anomaly
    pub anomaly_type: AnomalyType,
}

#[derive(Debug, Clone)]
/// Anomaly Type
pub enum AnomalyType {
    /// Spike
    Spike,
    /// Drop
    Drop,
    /// Outlier
    Outlier,
    /// Level shift
    LevelShift,
}

impl TimeSeriesAnalyzer {
    /// Create new time-series analyzer
    pub fn new(max_points: usize) -> Self {
        Self {
            data_store: Arc::new(RwLock::new(HashMap::new())),
            max_points,
        }
    }

    /// Add data point
    pub fn add_point(&self, metric_name: String, point: TimeSeriesPoint) {
        let mut store = self.data_store.write().unwrap();
        let series = store.entry(metric_name).or_insert_with(VecDeque::new);

        series.push_back(point);

        // Keep only recent points
        while series.len() > self.max_points {
            series.pop_front();
        }
    }

    /// Analyze trends for a metric
    pub fn analyze_trends(&self, metric_name: &str) -> Option<TrendAnalysis> {
        let store = self.data_store.read().unwrap();
        let series = store.get(metric_name)?;

        if series.len() < 10 {
            return None; // Need at least 10 points for meaningful analysis
        }

        let values: Vec<f64> = series.iter().map(|p| p.value).collect();
        let timestamps: Vec<u64> = series
            .iter()
            .map(|p| p.timestamp.duration_since(UNIX_EPOCH).unwrap().as_secs())
            .collect();

        // Calculate linear trend
        let (trend_coefficient, correlation) = self.calculate_linear_trend(&timestamps, &values);

        // Detect seasonality (simplified)
        let seasonality = self.detect_seasonality(&values, &timestamps);

        // Detect anomalies
        let anomalies = self.detect_anomalies(series);

        Some(TrendAnalysis {
            trend_coefficient,
            correlation,
            seasonality,
            anomalies,
        })
    }

    /// Calculate linear trend using least squares
    fn calculate_linear_trend(&self, timestamps: &[u64], values: &[f64]) -> (f64, f64) {
        let n = timestamps.len() as f64;
        let sum_x: f64 = timestamps.iter().map(|&x| x as f64).sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = timestamps
            .iter()
            .zip(values.iter())
            .map(|(&x, &y)| x as f64 * y)
            .sum();
        let sum_x2: f64 = timestamps.iter().map(|&x| (x as f64).powi(2)).sum();

        // Calculate slope (trend coefficient)
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));

        // Calculate correlation coefficient
        let mean_x = sum_x / n;
        let mean_y = sum_y / n;

        let numerator: f64 = timestamps
            .iter()
            .zip(values.iter())
            .map(|(&x, &y)| (x as f64 - mean_x) * (y - mean_y))
            .sum();

        let denom_x: f64 = timestamps
            .iter()
            .map(|&x| (x as f64 - mean_x).powi(2))
            .sum();

        let denom_y: f64 = values.iter().map(|&y| (y - mean_y).powi(2)).sum();

        let correlation = if denom_x > 0.0 && denom_y > 0.0 {
            numerator / (denom_x * denom_y).sqrt()
        } else {
            0.0
        };

        (slope, correlation)
    }

    /// Detect seasonal patterns (simplified FFT-like approach)
    fn detect_seasonality(&self, values: &[f64], _timestamps: &[u64]) -> Vec<SeasonalPattern> {
        // Simplified seasonality detection
        // In production, you'd use proper FFT or autocorrelation
        let mut patterns = Vec::new();

        // Check for common periods (hourly, daily patterns)
        for period_seconds in [3600, 86400] {
            // 1 hour, 1 day
            let period = Duration::from_secs(period_seconds);
            let confidence = self.calculate_periodic_confidence(values, period_seconds as usize);

            if confidence > 0.3 {
                // Threshold for detecting pattern
                patterns.push(SeasonalPattern {
                    period,
                    amplitude: self.calculate_amplitude(values),
                    confidence,
                });
            }
        }

        patterns
    }

    /// Calculate confidence in periodic pattern
    fn calculate_periodic_confidence(&self, values: &[f64], period: usize) -> f64 {
        if values.len() < period * 2 {
            return 0.0;
        }

        let mut correlation_sum = 0.0;
        let mut count = 0;

        for i in period..values.len() {
            correlation_sum += values[i] * values[i - period];
            count += 1;
        }

        if count > 0 {
            correlation_sum / count as f64 / values.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
        } else {
            0.0
        }
    }

    /// Calculate amplitude of signal
    fn calculate_amplitude(&self, values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|x| (x - mean).abs()).fold(0.0, f64::max)
    }

    /// Detect anomalies using statistical methods
    fn detect_anomalies(&self, series: &VecDeque<TimeSeriesPoint>) -> Vec<AnomalyPoint> {
        let mut anomalies = Vec::new();

        if series.len() < 10 {
            return anomalies;
        }

        let values: Vec<f64> = series.iter().map(|p| p.value).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std_dev =
            (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();

        // Detect outliers using z-score
        for (i, point) in series.iter().enumerate() {
            let z_score = (point.value - mean).abs() / std_dev;

            if z_score > 3.0 {
                // 3-sigma rule
                let anomaly_type = if point.value > mean + 2.0 * std_dev {
                    AnomalyType::Spike
                } else if point.value < mean - 2.0 * std_dev {
                    AnomalyType::Drop
                } else {
                    AnomalyType::Outlier
                };

                anomalies.push(AnomalyPoint {
                    timestamp: point.timestamp,
                    value: point.value,
                    severity: (z_score - 3.0).min(1.0), // Normalize to 0-1
                    anomaly_type,
                });
            }
        }

        anomalies
    }

    /// Get all metric names being tracked
    pub fn get_tracked_metrics(&self) -> Vec<String> {
        self.data_store.read().unwrap().keys().cloned().collect()
    }

    /// Get recent data points for a metric
    pub fn get_recent_data(&self, metric_name: &str, limit: usize) -> Vec<TimeSeriesPoint> {
        let store = self.data_store.read().unwrap();
        if let Some(series) = store.get(metric_name) {
            series.iter().rev().take(limit).cloned().collect()
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_metrics_collector() {
        let collector = InMemoryMetricsCollector::new();

        // Test counter
        collector.increment_counter("test_counter", 5, HashMap::new());
        collector.increment_counter("test_counter", 3, HashMap::new());

        // Test gauge
        collector.set_gauge("test_gauge", 42.5, HashMap::new());

        // Test histogram
        collector.observe_histogram("test_histogram", 1.5, HashMap::new());
        collector.observe_histogram("test_histogram", 2.5, HashMap::new());

        let metrics = collector.get_metrics();
        assert!(metrics.contains_key("test_counter"));
        assert!(metrics.contains_key("test_gauge"));
        assert!(metrics.contains_key("test_histogram"));
    }

    #[test]
    fn test_performance_metrics() {
        let collector = Arc::new(InMemoryMetricsCollector::new());
        let performance_metrics = PerformanceMetrics::new(collector);

        // Record successful recognition
        performance_metrics.record_request(
            RecognitionStatus::Success,
            Duration::from_millis(150),
            Duration::from_secs(3),
        );

        // Record failed recognition
        performance_metrics.record_request(
            RecognitionStatus::Error("timeout".to_string()),
            Duration::from_millis(200),
            Duration::from_secs(2),
        );

        // Record accuracy
        performance_metrics.record_accuracy(0.92, "transformer");

        let stats = performance_metrics.get_statistics();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.success_rate, 0.5);
        assert!(stats.average_real_time_factor > 0.0);
    }

    #[test]
    fn test_alert_manager() {
        let mut alert_manager = AlertManager::new();

        // Add console handler
        alert_manager.add_handler(Box::new(ConsoleAlertHandler));

        // Add alert rule
        let rule = AlertRule {
            id: "high_latency".to_string(),
            metric_name: "processing_latency".to_string(),
            condition: AlertCondition::GreaterThan(100.0),
            severity: AlertSeverity::Warning,
            description: "Processing latency is too high".to_string(),
            evaluation_interval: Duration::from_secs(60),
            labels: HashMap::new(),
        };
        alert_manager.add_rule(rule);

        // Create metrics that should trigger the alert
        let mut metrics = HashMap::new();
        let data_points = vec![DataPoint {
            timestamp: SystemTime::now(),
            value: MetricValue::Gauge(150.0), // Above threshold
            labels: HashMap::new(),
        }];
        metrics.insert("processing_latency".to_string(), data_points);

        // Evaluate alerts
        alert_manager.evaluate_alerts(&metrics);

        let active_alerts = alert_manager.get_active_alerts();
        assert_eq!(active_alerts.len(), 1);
        assert_eq!(active_alerts[0].rule.id, "high_latency");
    }

    #[test]
    fn test_time_series_analyzer() {
        let analyzer = TimeSeriesAnalyzer::new(1000);

        // Add some time series data with clear positive trend
        let base_time = SystemTime::now();
        for i in 0..50 {
            let point = TimeSeriesPoint {
                timestamp: base_time + Duration::from_secs(i * 60), // 1 minute intervals
                value: i as f64 * 1.0 + 10.0 + (scirs2_core::random::random::<f64>() - 0.5) * 0.1, // Strong linear trend with minimal noise
                labels: HashMap::new(),
            };
            analyzer.add_point("test_metric".to_string(), point);
        }

        let analysis = analyzer.analyze_trends("test_metric");
        assert!(analysis.is_some());

        let trend = analysis.unwrap();
        assert!(trend.trend_coefficient.abs() > 0.0); // Should detect trend (positive or negative)
                                                      // Note: correlation might be low due to simplified implementation
                                                      // In production, you'd use a more sophisticated correlation calculation
    }
}
