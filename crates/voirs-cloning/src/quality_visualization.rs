//! Quality Visualization Dashboard for VoiRS Voice Cloning
//!
//! This module provides a comprehensive real-time quality metrics dashboard
//! for visualizing voice cloning quality, performance metrics, and system health
//! with interactive charts, alerts, and detailed analytics.

use crate::performance_monitoring::{PerformanceMetrics, PerformanceMonitor};
use crate::quality::{CloningQualityAssessor, QualityMetrics};
use crate::similarity::SimilarityMeasurer;
use crate::types::{SpeakerProfile, VoiceCloneResult, VoiceSample};
use crate::usage_tracking::SimilarityMetrics;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

/// Visualization data point for time series charts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp of the measurement
    pub timestamp: u64,
    /// Metric value
    pub value: f32,
    /// Optional label or category
    pub label: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Chart types supported by the visualization system
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChartType {
    /// Line chart for time series data
    Line,
    /// Bar chart for categorical data
    Bar,
    /// Pie chart for proportional data
    Pie,
    /// Scatter plot for correlation analysis
    Scatter,
    /// Histogram for distribution analysis
    Histogram,
    /// Heatmap for 2D correlation data
    Heatmap,
    /// Gauge for single value display
    Gauge,
    /// Progress indicator
    Progress,
}

/// Metric category for organization
#[derive(Debug, Clone, PartialEq, Hash, Eq, Serialize, Deserialize)]
pub enum MetricCategory {
    /// Audio quality metrics
    AudioQuality,
    /// Speaker similarity metrics
    SpeakerSimilarity,
    /// System performance metrics
    SystemPerformance,
    /// User engagement metrics
    UserEngagement,
    /// Error and issue tracking
    ErrorTracking,
    /// Resource utilization metrics
    ResourceUtilization,
    /// Custom user-defined metrics
    Custom(String),
}

/// Quality threshold definition for alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThreshold {
    /// Threshold identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Metric to monitor
    pub metric_path: String,
    /// Warning threshold value
    pub warning_threshold: f32,
    /// Critical threshold value
    pub critical_threshold: f32,
    /// Comparison operator (greater_than, less_than, equals)
    pub operator: ComparisonOperator,
    /// Whether threshold is currently enabled
    pub enabled: bool,
    /// Alert cooldown period to prevent spam
    pub cooldown_duration: Duration,
    /// Last alert timestamp
    pub last_alert: Option<SystemTime>,
}

/// Comparison operators for thresholds
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Greater than threshold
    GreaterThan,
    /// Less than threshold
    LessThan,
    /// Equal to threshold
    Equals,
    /// Greater than or equal to threshold
    GreaterThanOrEqual,
    /// Less than or equal to threshold
    LessThanOrEqual,
}

/// Alert level severity
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertLevel {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Critical alert
    Critical,
    /// Fatal alert
    Fatal,
}

/// Quality alert notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAlert {
    /// Alert identifier
    pub id: String,
    /// Alert level
    pub level: AlertLevel,
    /// Alert title
    pub title: String,
    /// Detailed message
    pub message: String,
    /// Threshold that triggered the alert
    pub threshold_id: String,
    /// Current metric value
    pub current_value: f32,
    /// Threshold value that was exceeded
    pub threshold_value: f32,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Whether alert has been acknowledged
    pub acknowledged: bool,
    /// Acknowledgment timestamp
    pub acknowledged_at: Option<SystemTime>,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Dashboard widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardWidget {
    /// Widget identifier
    pub id: String,
    /// Widget title
    pub title: String,
    /// Widget description
    pub description: String,
    /// Chart type for visualization
    pub chart_type: ChartType,
    /// Metrics to display
    pub metric_paths: Vec<String>,
    /// Widget position and size
    pub layout: WidgetLayout,
    /// Refresh interval in seconds
    pub refresh_interval: u64,
    /// Maximum data points to display
    pub max_data_points: usize,
    /// Color scheme configuration
    pub color_scheme: Vec<String>,
    /// Widget-specific configuration
    pub config: HashMap<String, serde_json::Value>,
}

/// Widget layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetLayout {
    /// X position in grid units
    pub x: u32,
    /// Y position in grid units
    pub y: u32,
    /// Width in grid units
    pub width: u32,
    /// Height in grid units
    pub height: u32,
    /// Minimum width in pixels
    pub min_width: Option<u32>,
    /// Minimum height in pixels
    pub min_height: Option<u32>,
}

/// Time window for data aggregation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimeWindow {
    /// Last 5 minutes
    Last5Minutes,
    /// Last 15 minutes
    Last15Minutes,
    /// Last hour
    LastHour,
    /// Last 6 hours
    Last6Hours,
    /// Last 24 hours
    Last24Hours,
    /// Last week
    LastWeek,
    /// Custom time range
    Custom { start: SystemTime, end: SystemTime },
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Dashboard identifier
    pub id: String,
    /// Dashboard title
    pub title: String,
    /// Dashboard description
    pub description: String,
    /// Grid layout configuration
    pub grid_columns: u32,
    /// Auto-refresh interval in seconds
    pub auto_refresh_interval: u64,
    /// Default time window for widgets
    pub default_time_window: TimeWindow,
    /// Quality thresholds for alerts
    pub quality_thresholds: Vec<QualityThreshold>,
    /// Dashboard widgets
    pub widgets: Vec<DashboardWidget>,
    /// Theme configuration
    pub theme: DashboardTheme,
}

/// Dashboard visual theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTheme {
    /// Primary color
    pub primary_color: String,
    /// Secondary color
    pub secondary_color: String,
    /// Background color
    pub background_color: String,
    /// Text color
    pub text_color: String,
    /// Success color for good metrics
    pub success_color: String,
    /// Warning color for concerning metrics
    pub warning_color: String,
    /// Error color for critical metrics
    pub error_color: String,
    /// Font family
    pub font_family: String,
}

/// Metric time series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTimeSeries {
    /// Metric identifier
    pub metric_id: String,
    /// Metric display name
    pub display_name: String,
    /// Metric category
    pub category: MetricCategory,
    /// Time series data points
    pub data_points: VecDeque<DataPoint>,
    /// Statistical summary
    pub statistics: MetricStatistics,
    /// Last update timestamp
    pub last_updated: SystemTime,
}

/// Statistical summary for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStatistics {
    /// Current value
    pub current: f32,
    /// Minimum value in time window
    pub min: f32,
    /// Maximum value in time window
    pub max: f32,
    /// Average value in time window
    pub average: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Trend direction (positive/negative/stable)
    pub trend: TrendDirection,
    /// Number of data points
    pub count: usize,
}

/// Trend direction for metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,
    /// Decreasing trend
    Decreasing,
    /// Stable trend
    Stable,
    /// Unknown or insufficient data
    Unknown,
}

/// Main Quality Visualization Dashboard
pub struct QualityVisualization {
    /// Dashboard configuration
    config: DashboardConfig,
    /// Quality assessor for metrics
    quality_assessor: Arc<CloningQualityAssessor>,
    /// Similarity measurer for metrics
    similarity_measurer: Arc<SimilarityMeasurer>,
    /// Performance monitor for system metrics
    performance_monitor: Arc<PerformanceMonitor>,
    /// Metric time series data
    metrics: Arc<RwLock<HashMap<String, MetricTimeSeries>>>,
    /// Active alerts
    alerts: Arc<RwLock<Vec<QualityAlert>>>,
    /// Dashboard update mutex
    update_mutex: Arc<Mutex<()>>,
}

impl Default for DashboardTheme {
    fn default() -> Self {
        Self {
            primary_color: "#3b82f6".to_string(),
            secondary_color: "#64748b".to_string(),
            background_color: "#ffffff".to_string(),
            text_color: "#1f2937".to_string(),
            success_color: "#10b981".to_string(),
            warning_color: "#f59e0b".to_string(),
            error_color: "#ef4444".to_string(),
            font_family: "Inter, sans-serif".to_string(),
        }
    }
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            id: "default_dashboard".to_string(),
            title: "Voice Cloning Quality Dashboard".to_string(),
            description: "Real-time monitoring of voice cloning quality and performance"
                .to_string(),
            grid_columns: 12,
            auto_refresh_interval: 10,
            default_time_window: TimeWindow::LastHour,
            quality_thresholds: Self::default_thresholds(),
            widgets: Self::default_widgets(),
            theme: DashboardTheme::default(),
        }
    }
}

impl DashboardConfig {
    /// Create default quality thresholds
    fn default_thresholds() -> Vec<QualityThreshold> {
        vec![
            QualityThreshold {
                id: "overall_quality_warning".to_string(),
                name: "Overall Quality Warning".to_string(),
                metric_path: "quality.overall_score".to_string(),
                warning_threshold: 0.7,
                critical_threshold: 0.5,
                operator: ComparisonOperator::LessThan,
                enabled: true,
                cooldown_duration: Duration::from_secs(5 * 60),
                last_alert: None,
            },
            QualityThreshold {
                id: "similarity_warning".to_string(),
                name: "Speaker Similarity Warning".to_string(),
                metric_path: "similarity.overall_similarity".to_string(),
                warning_threshold: 0.6,
                critical_threshold: 0.4,
                operator: ComparisonOperator::LessThan,
                enabled: true,
                cooldown_duration: Duration::from_secs(5 * 60),
                last_alert: None,
            },
            QualityThreshold {
                id: "processing_time_warning".to_string(),
                name: "Processing Time Warning".to_string(),
                metric_path: "performance.processing_time_ms".to_string(),
                warning_threshold: 5000.0,
                critical_threshold: 10000.0,
                operator: ComparisonOperator::GreaterThan,
                enabled: true,
                cooldown_duration: Duration::from_secs(3 * 60),
                last_alert: None,
            },
        ]
    }

    /// Create default dashboard widgets
    fn default_widgets() -> Vec<DashboardWidget> {
        vec![
            // Quality metrics line chart
            DashboardWidget {
                id: "quality_overview".to_string(),
                title: "Quality Overview".to_string(),
                description: "Real-time voice cloning quality metrics".to_string(),
                chart_type: ChartType::Line,
                metric_paths: vec![
                    "quality.overall_score".to_string(),
                    "quality.snr_db".to_string(),
                    "quality.spectral_clarity".to_string(),
                ],
                layout: WidgetLayout {
                    x: 0,
                    y: 0,
                    width: 6,
                    height: 4,
                    min_width: Some(400),
                    min_height: Some(300),
                },
                refresh_interval: 5,
                max_data_points: 100,
                color_scheme: vec![
                    "#3b82f6".to_string(),
                    "#10b981".to_string(),
                    "#f59e0b".to_string(),
                ],
                config: HashMap::new(),
            },
            // Similarity gauge
            DashboardWidget {
                id: "similarity_gauge".to_string(),
                title: "Speaker Similarity".to_string(),
                description: "Current speaker similarity score".to_string(),
                chart_type: ChartType::Gauge,
                metric_paths: vec!["similarity.overall_similarity".to_string()],
                layout: WidgetLayout {
                    x: 6,
                    y: 0,
                    width: 3,
                    height: 4,
                    min_width: Some(200),
                    min_height: Some(300),
                },
                refresh_interval: 5,
                max_data_points: 1,
                color_scheme: vec!["#10b981".to_string()],
                config: {
                    let mut config = HashMap::new();
                    config.insert(
                        "min_value".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(0)),
                    );
                    config.insert(
                        "max_value".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(1)),
                    );
                    config.insert(
                        "warning_threshold".to_string(),
                        serde_json::Value::Number(serde_json::Number::from_f64(0.7).unwrap()),
                    );
                    config.insert(
                        "critical_threshold".to_string(),
                        serde_json::Value::Number(serde_json::Number::from_f64(0.5).unwrap()),
                    );
                    config
                },
            },
            // Processing performance bar chart
            DashboardWidget {
                id: "performance_metrics".to_string(),
                title: "Performance Metrics".to_string(),
                description: "System performance and resource utilization".to_string(),
                chart_type: ChartType::Bar,
                metric_paths: vec![
                    "performance.cpu_usage_percent".to_string(),
                    "performance.memory_usage_percent".to_string(),
                    "performance.gpu_usage_percent".to_string(),
                ],
                layout: WidgetLayout {
                    x: 9,
                    y: 0,
                    width: 3,
                    height: 4,
                    min_width: Some(200),
                    min_height: Some(300),
                },
                refresh_interval: 3,
                max_data_points: 10,
                color_scheme: vec![
                    "#3b82f6".to_string(),
                    "#8b5cf6".to_string(),
                    "#06d6a0".to_string(),
                ],
                config: HashMap::new(),
            },
            // Error rate histogram
            DashboardWidget {
                id: "error_distribution".to_string(),
                title: "Error Distribution".to_string(),
                description: "Distribution of errors and processing issues".to_string(),
                chart_type: ChartType::Histogram,
                metric_paths: vec!["errors.error_rate".to_string()],
                layout: WidgetLayout {
                    x: 0,
                    y: 4,
                    width: 6,
                    height: 3,
                    min_width: Some(400),
                    min_height: Some(200),
                },
                refresh_interval: 10,
                max_data_points: 50,
                color_scheme: vec!["#ef4444".to_string()],
                config: HashMap::new(),
            },
            // Recent alerts table
            DashboardWidget {
                id: "recent_alerts".to_string(),
                title: "Recent Alerts".to_string(),
                description: "Latest quality and performance alerts".to_string(),
                chart_type: ChartType::Bar, // Using bar as a placeholder for table
                metric_paths: vec!["alerts.recent_count".to_string()],
                layout: WidgetLayout {
                    x: 6,
                    y: 4,
                    width: 6,
                    height: 3,
                    min_width: Some(400),
                    min_height: Some(200),
                },
                refresh_interval: 5,
                max_data_points: 10,
                color_scheme: vec!["#ef4444".to_string(), "#f59e0b".to_string()],
                config: HashMap::new(),
            },
        ]
    }
}

impl QualityVisualization {
    /// Create new quality visualization dashboard
    pub async fn new(config: DashboardConfig) -> Result<Self> {
        let quality_assessor = Arc::new(CloningQualityAssessor::new()?);
        let similarity_config = crate::similarity::SimilarityConfig::default();
        let similarity_measurer = Arc::new(SimilarityMeasurer::new(similarity_config));
        let performance_monitor = Arc::new(PerformanceMonitor::new());

        Ok(Self {
            config,
            quality_assessor,
            similarity_measurer,
            performance_monitor,
            metrics: Arc::new(RwLock::new(HashMap::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            update_mutex: Arc::new(Mutex::new(())),
        })
    }

    /// Create dashboard with default configuration
    pub async fn with_default_config() -> Result<Self> {
        Self::new(DashboardConfig::default()).await
    }

    /// Update dashboard with new voice cloning result
    pub async fn update_with_result(&self, result: &VoiceCloneResult) -> Result<()> {
        let _lock = self.update_mutex.lock().await;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Assess quality metrics
        // Create voice samples for quality assessment
        let original_sample = VoiceSample::new("original".to_string(), vec![0.0; 1000], 16000);
        let cloned_sample = VoiceSample::new(
            "cloned".to_string(),
            result.audio.clone(),
            result.sample_rate,
        );
        let mut quality_assessor = CloningQualityAssessor::new()?;
        let quality_metrics = quality_assessor
            .assess_quality(&original_sample, &cloned_sample)
            .await?;

        // Update quality metrics
        self.update_metric(
            "quality.overall_score",
            quality_metrics.overall_score,
            timestamp,
            None,
        )
        .await;
        self.update_metric(
            "quality.snr_db",
            quality_metrics.analysis.snr_analysis.original_snr,
            timestamp,
            None,
        )
        .await;
        self.update_metric(
            "quality.spectral_clarity",
            quality_metrics.spectral_similarity,
            timestamp,
            None,
        )
        .await;
        self.update_metric(
            "quality.noise_level",
            quality_metrics.analysis.snr_analysis.noise_floor,
            timestamp,
            None,
        )
        .await;
        self.update_metric(
            "quality.dynamic_range",
            quality_metrics.analysis.snr_analysis.dynamic_range,
            timestamp,
            None,
        )
        .await;

        // Update similarity metrics
        self.update_metric(
            "similarity.overall_similarity",
            result.similarity_score,
            timestamp,
            None,
        )
        .await;

        // Update processing time
        let processing_time_ms = result.processing_time.as_millis() as f32;
        self.update_metric(
            "performance.processing_time_ms",
            processing_time_ms,
            timestamp,
            None,
        )
        .await;

        // Check thresholds and generate alerts
        self.check_thresholds().await?;

        Ok(())
    }

    /// Update individual metric value
    async fn update_metric(
        &self,
        metric_id: &str,
        value: f32,
        timestamp: u64,
        label: Option<String>,
    ) {
        let mut metrics = self.metrics.write().unwrap();

        let data_point = DataPoint {
            timestamp,
            value,
            label,
            metadata: HashMap::new(),
        };

        if let Some(metric_ts) = metrics.get_mut(metric_id) {
            // Add new data point
            metric_ts.data_points.push_back(data_point);

            // Limit data points
            while metric_ts.data_points.len() > 1000 {
                metric_ts.data_points.pop_front();
            }

            // Update statistics
            metric_ts.statistics = self.calculate_statistics(&metric_ts.data_points);
            metric_ts.last_updated = SystemTime::now();
        } else {
            // Create new metric time series
            let mut data_points = VecDeque::new();
            data_points.push_back(data_point);

            let metric_ts = MetricTimeSeries {
                metric_id: metric_id.to_string(),
                display_name: self.get_display_name(metric_id),
                category: self.get_metric_category(metric_id),
                data_points: data_points.clone(),
                statistics: self.calculate_statistics(&data_points),
                last_updated: SystemTime::now(),
            };

            metrics.insert(metric_id.to_string(), metric_ts);
        }
    }

    /// Calculate statistics for metric data points
    fn calculate_statistics(&self, data_points: &VecDeque<DataPoint>) -> MetricStatistics {
        if data_points.is_empty() {
            return MetricStatistics {
                current: 0.0,
                min: 0.0,
                max: 0.0,
                average: 0.0,
                std_dev: 0.0,
                trend: TrendDirection::Unknown,
                count: 0,
            };
        }

        let values: Vec<f32> = data_points.iter().map(|dp| dp.value).collect();
        let count = values.len();
        let current = values.last().copied().unwrap_or(0.0);
        let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f32 = values.iter().sum();
        let average = sum / count as f32;

        // Calculate standard deviation
        let variance: f32 =
            values.iter().map(|&x| (x - average).powi(2)).sum::<f32>() / count as f32;
        let std_dev = variance.sqrt();

        // Calculate trend
        let trend = if count >= 2 {
            let recent_avg = values.iter().rev().take(count.min(5)).sum::<f32>() / 5.0;
            let older_avg = values.iter().take(count.min(5)).sum::<f32>() / 5.0;

            let diff = recent_avg - older_avg;
            if diff > std_dev * 0.5 {
                TrendDirection::Increasing
            } else if diff < -std_dev * 0.5 {
                TrendDirection::Decreasing
            } else {
                TrendDirection::Stable
            }
        } else {
            TrendDirection::Unknown
        };

        MetricStatistics {
            current,
            min,
            max,
            average,
            std_dev,
            trend,
            count,
        }
    }

    /// Get display name for metric
    fn get_display_name(&self, metric_id: &str) -> String {
        match metric_id {
            "quality.overall_score" => "Overall Quality".to_string(),
            "quality.snr_db" => "Signal-to-Noise Ratio (dB)".to_string(),
            "quality.spectral_clarity" => "Spectral Clarity".to_string(),
            "quality.noise_level" => "Noise Level".to_string(),
            "similarity.overall_similarity" => "Speaker Similarity".to_string(),
            "performance.processing_time_ms" => "Processing Time (ms)".to_string(),
            "performance.cpu_usage_percent" => "CPU Usage (%)".to_string(),
            "performance.memory_usage_percent" => "Memory Usage (%)".to_string(),
            _ => metric_id.to_string(),
        }
    }

    /// Get metric category
    fn get_metric_category(&self, metric_id: &str) -> MetricCategory {
        if metric_id.starts_with("quality.") {
            MetricCategory::AudioQuality
        } else if metric_id.starts_with("similarity.") {
            MetricCategory::SpeakerSimilarity
        } else if metric_id.starts_with("performance.") {
            MetricCategory::SystemPerformance
        } else if metric_id.starts_with("errors.") {
            MetricCategory::ErrorTracking
        } else {
            MetricCategory::Custom(metric_id.to_string())
        }
    }

    /// Check quality thresholds and generate alerts
    async fn check_thresholds(&self) -> Result<()> {
        let metrics = self.metrics.read().unwrap();
        let mut alerts = self.alerts.write().unwrap();

        for threshold in &self.config.quality_thresholds {
            if !threshold.enabled {
                continue;
            }

            // Check cooldown period
            if let Some(last_alert) = threshold.last_alert {
                if SystemTime::now()
                    .duration_since(last_alert)
                    .unwrap_or_default()
                    < threshold.cooldown_duration
                {
                    continue;
                }
            }

            if let Some(metric) = metrics.get(&threshold.metric_path) {
                let current_value = metric.statistics.current;
                let should_alert = match threshold.operator {
                    ComparisonOperator::GreaterThan => current_value > threshold.critical_threshold,
                    ComparisonOperator::LessThan => current_value < threshold.critical_threshold,
                    ComparisonOperator::Equals => {
                        (current_value - threshold.critical_threshold).abs() < 0.001
                    }
                    ComparisonOperator::GreaterThanOrEqual => {
                        current_value >= threshold.critical_threshold
                    }
                    ComparisonOperator::LessThanOrEqual => {
                        current_value <= threshold.critical_threshold
                    }
                };

                if should_alert {
                    let alert = QualityAlert {
                        id: format!(
                            "alert_{}_{}",
                            threshold.id,
                            SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_secs()
                        ),
                        level: AlertLevel::Critical,
                        title: format!("Critical: {}", threshold.name),
                        message: format!(
                            "Metric {} has reached critical level: {:.3} (threshold: {:.3})",
                            threshold.metric_path, current_value, threshold.critical_threshold
                        ),
                        threshold_id: threshold.id.clone(),
                        current_value,
                        threshold_value: threshold.critical_threshold,
                        timestamp: SystemTime::now(),
                        acknowledged: false,
                        acknowledged_at: None,
                        suggested_actions: self.get_suggested_actions(&threshold.metric_path),
                    };

                    alerts.push(alert);

                    // Update last alert time (would need mutable access in real implementation)
                }
            }
        }

        // Limit alerts history
        while alerts.len() > 100 {
            alerts.remove(0);
        }

        Ok(())
    }

    /// Get suggested actions for metric alerts
    fn get_suggested_actions(&self, metric_path: &str) -> Vec<String> {
        match metric_path {
            "quality.overall_score" => vec![
                "Check input audio quality".to_string(),
                "Verify speaker data consistency".to_string(),
                "Review model parameters".to_string(),
            ],
            "similarity.overall_similarity" => vec![
                "Add more reference samples".to_string(),
                "Check speaker identity consistency".to_string(),
                "Review embedding quality".to_string(),
            ],
            "performance.processing_time_ms" => vec![
                "Optimize batch size".to_string(),
                "Check system resources".to_string(),
                "Consider model quantization".to_string(),
            ],
            _ => vec!["Review system configuration and logs".to_string()],
        }
    }

    /// Get metric data for time window
    pub async fn get_metric_data(
        &self,
        metric_id: &str,
        time_window: &TimeWindow,
    ) -> Result<Vec<DataPoint>> {
        let metrics = self.metrics.read().unwrap();

        if let Some(metric) = metrics.get(metric_id) {
            let (start_time, end_time) = self.get_time_range(time_window);

            let filtered_data: Vec<DataPoint> = metric
                .data_points
                .iter()
                .filter(|dp| {
                    let timestamp = UNIX_EPOCH + Duration::from_secs(dp.timestamp);
                    timestamp >= start_time && timestamp <= end_time
                })
                .cloned()
                .collect();

            Ok(filtered_data)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get time range for time window
    fn get_time_range(&self, time_window: &TimeWindow) -> (SystemTime, SystemTime) {
        let now = SystemTime::now();

        let start_time = match time_window {
            TimeWindow::Last5Minutes => now - Duration::from_secs(300),
            TimeWindow::Last15Minutes => now - Duration::from_secs(900),
            TimeWindow::LastHour => now - Duration::from_secs(3600),
            TimeWindow::Last6Hours => now - Duration::from_secs(21600),
            TimeWindow::Last24Hours => now - Duration::from_secs(86400),
            TimeWindow::LastWeek => now - Duration::from_secs(604800),
            TimeWindow::Custom { start, end: _ } => *start,
        };

        let end_time = match time_window {
            TimeWindow::Custom { start: _, end } => *end,
            _ => now,
        };

        (start_time, end_time)
    }

    /// Get all active alerts
    pub async fn get_alerts(&self) -> Vec<QualityAlert> {
        let alerts = self.alerts.read().unwrap();
        alerts.clone()
    }

    /// Acknowledge alert
    pub async fn acknowledge_alert(&self, alert_id: &str) -> Result<()> {
        let mut alerts = self.alerts.write().unwrap();

        if let Some(alert) = alerts.iter_mut().find(|a| a.id == alert_id) {
            alert.acknowledged = true;
            alert.acknowledged_at = Some(SystemTime::now());
            Ok(())
        } else {
            Err(Error::Validation(format!("Alert not found: {}", alert_id)))
        }
    }

    /// Get dashboard configuration
    pub fn get_config(&self) -> &DashboardConfig {
        &self.config
    }

    /// Update dashboard configuration
    pub async fn update_config(&mut self, config: DashboardConfig) {
        self.config = config;
    }

    /// Get widget data for rendering
    pub async fn get_widget_data(
        &self,
        widget_id: &str,
    ) -> Result<HashMap<String, serde_json::Value>> {
        let widget = self
            .config
            .widgets
            .iter()
            .find(|w| w.id == widget_id)
            .ok_or_else(|| Error::Validation(format!("Widget not found: {}", widget_id)))?;

        let mut widget_data = HashMap::new();

        // Collect data for all metric paths
        for metric_path in &widget.metric_paths {
            let data = self
                .get_metric_data(metric_path, &self.config.default_time_window)
                .await?;

            // Limit to max data points
            let limited_data: Vec<DataPoint> = if data.len() > widget.max_data_points {
                data.iter()
                    .rev()
                    .take(widget.max_data_points)
                    .cloned()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect()
            } else {
                data
            };

            widget_data.insert(metric_path.clone(), serde_json::to_value(limited_data)?);
        }

        // Add widget configuration
        widget_data.insert("widget_config".to_string(), serde_json::to_value(widget)?);

        Ok(widget_data)
    }

    /// Export dashboard data to JSON
    pub async fn export_data(&self, time_window: &TimeWindow) -> Result<serde_json::Value> {
        let metrics = self.metrics.read().unwrap();
        let alerts = self.alerts.read().unwrap();

        let mut export_data = HashMap::new();

        // Export metrics data
        for (metric_id, metric) in metrics.iter() {
            let data = metric
                .data_points
                .iter()
                .filter(|dp| {
                    let timestamp = UNIX_EPOCH + Duration::from_secs(dp.timestamp);
                    let (start_time, end_time) = self.get_time_range(time_window);
                    timestamp >= start_time && timestamp <= end_time
                })
                .cloned()
                .collect::<Vec<_>>();

            export_data.insert(metric_id.clone(), serde_json::to_value(data)?);
        }

        // Export alerts
        export_data.insert("alerts".to_string(), serde_json::to_value(alerts.clone())?);

        // Export configuration
        export_data.insert("config".to_string(), serde_json::to_value(&self.config)?);

        Ok(serde_json::to_value(export_data)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CloningMethod, VoiceSample};

    #[tokio::test]
    async fn test_dashboard_creation() {
        let dashboard = QualityVisualization::with_default_config().await.unwrap();

        // Verify default configuration
        let config = dashboard.get_config();
        assert_eq!(config.title, "Voice Cloning Quality Dashboard");
        assert!(!config.widgets.is_empty());
        assert!(!config.quality_thresholds.is_empty());
    }

    #[tokio::test]
    async fn test_metric_updates() {
        let dashboard = QualityVisualization::with_default_config().await.unwrap();

        // Create test result
        let result = VoiceCloneResult {
            request_id: "test_request".to_string(),
            audio: vec![0.0; 16000],
            sample_rate: 16000,
            quality_metrics: HashMap::new(),
            similarity_score: 0.85,
            processing_time: Duration::from_millis(500),
            method_used: CloningMethod::FewShot,
            success: true,
            error_message: None,
            cross_lingual_info: None,
            timestamp: SystemTime::now(),
        };

        // Update dashboard with result
        dashboard.update_with_result(&result).await.unwrap();

        // Verify metrics were updated
        let similarity_data = dashboard
            .get_metric_data("similarity.overall_similarity", &TimeWindow::LastHour)
            .await
            .unwrap();

        assert!(!similarity_data.is_empty());
        assert_eq!(similarity_data[0].value, 0.85);
    }

    #[tokio::test]
    async fn test_widget_data_retrieval() {
        let dashboard = QualityVisualization::with_default_config().await.unwrap();

        // Update with test data
        let result = VoiceCloneResult {
            request_id: "test_request".to_string(),
            audio: vec![0.0; 16000],
            sample_rate: 16000,
            quality_metrics: HashMap::new(),
            similarity_score: 0.75,
            processing_time: Duration::from_millis(300),
            method_used: CloningMethod::FewShot,
            success: true,
            error_message: None,
            cross_lingual_info: None,
            timestamp: SystemTime::now(),
        };

        dashboard.update_with_result(&result).await.unwrap();

        // Get widget data
        let widget_data = dashboard.get_widget_data("similarity_gauge").await.unwrap();

        assert!(widget_data.contains_key("similarity.overall_similarity"));
        assert!(widget_data.contains_key("widget_config"));
    }

    #[tokio::test]
    async fn test_alert_system() {
        let dashboard = QualityVisualization::with_default_config().await.unwrap();

        // Create result with low quality to trigger alert
        let result = VoiceCloneResult {
            request_id: "test_request".to_string(),
            audio: vec![0.0; 16000],
            sample_rate: 16000,
            quality_metrics: HashMap::new(),
            similarity_score: 0.3, // Below critical threshold
            processing_time: Duration::from_millis(300),
            method_used: CloningMethod::FewShot,
            success: true,
            error_message: None,
            cross_lingual_info: None,
            timestamp: SystemTime::now(),
        };

        dashboard.update_with_result(&result).await.unwrap();

        // Check for alerts
        let alerts = dashboard.get_alerts().await;
        assert!(!alerts.is_empty());

        // Verify alert properties
        let alert = &alerts[0];
        assert_eq!(alert.level, AlertLevel::Critical);
        assert!(!alert.acknowledged);

        // Acknowledge alert
        dashboard.acknowledge_alert(&alert.id).await.unwrap();
        let updated_alerts = dashboard.get_alerts().await;
        assert!(updated_alerts[0].acknowledged);
    }

    #[tokio::test]
    async fn test_data_export() {
        let dashboard = QualityVisualization::with_default_config().await.unwrap();

        // Add test data
        let result = VoiceCloneResult {
            request_id: "test_request".to_string(),
            audio: vec![0.0; 16000],
            sample_rate: 16000,
            quality_metrics: HashMap::new(),
            similarity_score: 0.8,
            processing_time: Duration::from_millis(400),
            method_used: CloningMethod::FewShot,
            success: true,
            error_message: None,
            cross_lingual_info: None,
            timestamp: SystemTime::now(),
        };

        dashboard.update_with_result(&result).await.unwrap();

        // Export data
        let exported = dashboard.export_data(&TimeWindow::LastHour).await.unwrap();

        assert!(exported.is_object());
        let export_obj = exported.as_object().unwrap();
        assert!(export_obj.contains_key("config"));
        assert!(export_obj.contains_key("alerts"));
    }
}
