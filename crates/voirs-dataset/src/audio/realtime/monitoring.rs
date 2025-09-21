//! Quality monitoring and alerting for real-time audio processing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Quality monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMonitoringConfig {
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Quality metrics to monitor
    pub monitored_metrics: Vec<QualityMetric>,
    /// Alert configuration
    pub alert_config: AlertConfig,
    /// Visualization configuration
    pub visualization_config: VisualizationConfig,
    /// Historical data configuration
    pub historical_data_config: HistoricalDataConfig,
}

/// Quality metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityMetric {
    /// Signal-to-noise ratio
    SignalToNoiseRatio,
    /// Total harmonic distortion + noise
    TotalHarmonicDistortionNoise,
    /// Dynamic range
    DynamicRange,
    /// Noise floor
    NoiseFloor,
    /// Latency
    Latency,
    /// Jitter
    Jitter,
    /// Dropout rate
    DropoutRate,
    /// CPU usage
    CPUUsage,
    /// Memory usage
    MemoryUsage,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Alert thresholds
    pub thresholds: HashMap<String, AlertThreshold>,
    /// Alert actions
    pub actions: Vec<AlertAction>,
    /// Alert aggregation
    pub aggregation: AlertAggregationConfig,
    /// Alert suppression
    pub suppression: AlertSuppressionConfig,
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThreshold {
    /// Warning threshold
    pub warning: f32,
    /// Critical threshold
    pub critical: f32,
    /// Recovery threshold
    pub recovery: f32,
    /// Hysteresis
    pub hysteresis: f32,
}

/// Alert actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertAction {
    /// Log alert
    Log,
    /// Send notification
    Notification,
    /// Trigger correction
    TriggerCorrection,
    /// Escalate alert
    Escalate,
    /// Custom action
    Custom(String),
}

/// Alert aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertAggregationConfig {
    /// Aggregation window
    pub window: Duration,
    /// Aggregation method
    pub method: AggregationMethod,
    /// Minimum alert count
    pub min_count: usize,
}

/// Aggregation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Count-based aggregation
    Count,
    /// Rate-based aggregation
    Rate,
    /// Time-based aggregation
    Time,
    /// Severity-based aggregation
    Severity,
}

/// Alert suppression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertSuppressionConfig {
    /// Suppression window
    pub window: Duration,
    /// Suppression rules
    pub rules: Vec<SuppressionRule>,
}

/// Suppression rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuppressionRule {
    /// Suppress duplicate alerts
    SuppressDuplicates,
    /// Suppress alerts below threshold
    SuppressBelowThreshold(f32),
    /// Suppress alerts during maintenance
    SuppressDuringMaintenance,
    /// Custom suppression rule
    Custom(String),
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Visualization types
    pub visualization_types: Vec<VisualizationType>,
    /// Update interval
    pub update_interval: Duration,
    /// Visualization parameters
    pub parameters: VisualizationParameters,
}

/// Visualization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationType {
    /// Real-time waveform
    RealTimeWaveform,
    /// Spectrum analyzer
    SpectrumAnalyzer,
    /// Quality metrics dashboard
    QualityMetricsDashboard,
    /// Latency monitor
    LatencyMonitor,
    /// System performance monitor
    SystemPerformanceMonitor,
}

/// Visualization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationParameters {
    /// Window size for visualization
    pub window_size: Duration,
    /// Refresh rate
    pub refresh_rate: f32,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Display resolution
    pub resolution: (u32, u32),
}

/// Color schemes for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorScheme {
    /// Default color scheme
    Default,
    /// High contrast
    HighContrast,
    /// Monochrome
    Monochrome,
    /// Custom color scheme
    Custom(Vec<String>),
}

/// Historical data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalDataConfig {
    /// Data retention period
    pub retention_period: Duration,
    /// Sampling rate for historical data
    pub sampling_rate: f32,
    /// Storage format
    pub storage_format: StorageFormat,
    /// Compression settings
    pub compression: HistoricalDataCompression,
}

/// Storage formats for historical data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageFormat {
    /// Binary format
    Binary,
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// Database format
    Database,
}

/// Historical data compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalDataCompression {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: String,
    /// Compression level
    pub level: u8,
}

/// Alert definition
#[derive(Debug, Clone, Serialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Alert timestamp
    #[serde(skip)]
    pub timestamp: Instant,
    /// Alert metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    /// Quality alert
    Quality,
    /// Performance alert
    Performance,
    /// System alert
    System,
    /// Error alert
    Error,
    /// Warning alert
    Warning,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

impl Alert {
    /// Create a new alert
    pub fn new(
        id: String,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: String,
    ) -> Self {
        Self {
            id,
            alert_type,
            severity,
            message,
            timestamp: Instant::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the alert
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl Default for QualityMonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_millis(100),
            monitored_metrics: vec![
                QualityMetric::SignalToNoiseRatio,
                QualityMetric::DynamicRange,
                QualityMetric::Latency,
            ],
            alert_config: AlertConfig::default(),
            visualization_config: VisualizationConfig::default(),
            historical_data_config: HistoricalDataConfig::default(),
        }
    }
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            thresholds: HashMap::new(),
            actions: vec![AlertAction::Log, AlertAction::Notification],
            aggregation: AlertAggregationConfig::default(),
            suppression: AlertSuppressionConfig::default(),
        }
    }
}

impl Default for AlertAggregationConfig {
    fn default() -> Self {
        Self {
            window: Duration::from_secs(60),
            method: AggregationMethod::Count,
            min_count: 3,
        }
    }
}

impl Default for AlertSuppressionConfig {
    fn default() -> Self {
        Self {
            window: Duration::from_secs(300),
            rules: vec![SuppressionRule::SuppressDuplicates],
        }
    }
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            visualization_types: vec![
                VisualizationType::RealTimeWaveform,
                VisualizationType::QualityMetricsDashboard,
            ],
            update_interval: Duration::from_millis(50),
            parameters: VisualizationParameters::default(),
        }
    }
}

impl Default for VisualizationParameters {
    fn default() -> Self {
        Self {
            window_size: Duration::from_secs(5),
            refresh_rate: 30.0,
            color_scheme: ColorScheme::Default,
            resolution: (1920, 1080),
        }
    }
}

impl Default for HistoricalDataConfig {
    fn default() -> Self {
        Self {
            retention_period: Duration::from_secs(86400), // 24 hours
            sampling_rate: 10.0,                          // 10 Hz
            storage_format: StorageFormat::Binary,
            compression: HistoricalDataCompression::default(),
        }
    }
}

impl Default for HistoricalDataCompression {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: "zstd".to_string(),
            level: 3,
        }
    }
}
