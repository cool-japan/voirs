//! Quality control and monitoring for real-time audio processing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Quality control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityControlConfig {
    /// Quality thresholds
    pub quality_thresholds: QualityThresholds,
    /// Adaptive quality control
    pub adaptive_control: AdaptiveQualityConfig,
    /// Quality reporting
    pub quality_reporting: QualityReportingConfig,
    /// Correction strategies
    pub correction_strategies: Vec<CorrectionStrategy>,
}

/// Quality thresholds for real-time monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum SNR threshold
    pub min_snr: f32,
    /// Maximum THD+N threshold
    pub max_thd_n: f32,
    /// Minimum dynamic range
    pub min_dynamic_range: f32,
    /// Maximum noise floor
    pub max_noise_floor: f32,
    /// Latency threshold
    pub max_latency: Duration,
}

/// Adaptive quality control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveQualityConfig {
    /// Enable adaptive control
    pub enabled: bool,
    /// Adaptation rate
    pub adaptation_rate: f32,
    /// Control parameters
    pub control_parameters: AdaptiveControlParameters,
    /// Feedback mechanism
    pub feedback_mechanism: FeedbackMechanism,
}

/// Adaptive control parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveControlParameters {
    /// Learning rate
    pub learning_rate: f32,
    /// Smoothing factor
    pub smoothing_factor: f32,
    /// Adaptation window
    pub adaptation_window: Duration,
    /// Stability threshold
    pub stability_threshold: f32,
}

/// Feedback mechanisms for adaptive control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackMechanism {
    /// Proportional feedback
    Proportional,
    /// Proportional-Integral feedback
    ProportionalIntegral,
    /// Proportional-Integral-Derivative feedback
    ProportionalIntegralDerivative,
    /// Adaptive feedback
    Adaptive,
}

/// Quality reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReportingConfig {
    /// Reporting interval
    pub reporting_interval: Duration,
    /// Report format
    pub report_format: ReportFormat,
    /// Include visualizations
    pub include_visualizations: bool,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f32>,
}

/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    /// JSON format
    JSON,
    /// Binary format
    Binary,
    /// Text format
    Text,
    /// Custom format
    Custom(String),
}

/// Correction strategies for quality issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrectionStrategy {
    /// Automatic gain control
    AutomaticGainControl,
    /// Noise suppression
    NoiseSuppression,
    /// Dynamic range compression
    DynamicRangeCompression,
    /// Equalization
    Equalization,
    /// Buffer size adjustment
    BufferSizeAdjustment,
}

/// Quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Overall quality score
    pub overall_score: f32,
    /// Individual metric scores
    pub metric_scores: HashMap<String, f32>,
    /// Quality confidence
    pub confidence: f32,
    /// Recommendations
    pub recommendations: Vec<String>,
}

impl QualityAssessment {
    /// Create a new quality assessment
    pub fn new() -> Self {
        Self {
            overall_score: 0.0,
            metric_scores: HashMap::new(),
            confidence: 0.0,
            recommendations: Vec::new(),
        }
    }

    /// Add a metric score
    pub fn add_metric(&mut self, name: String, score: f32) {
        self.metric_scores.insert(name, score);
    }

    /// Calculate overall score from metrics
    pub fn calculate_overall_score(&mut self) {
        if self.metric_scores.is_empty() {
            self.overall_score = 0.0;
            return;
        }

        let sum: f32 = self.metric_scores.values().sum();
        self.overall_score = sum / self.metric_scores.len() as f32;
    }

    /// Add a recommendation
    pub fn add_recommendation(&mut self, recommendation: String) {
        self.recommendations.push(recommendation);
    }
}

impl Default for QualityControlConfig {
    fn default() -> Self {
        Self {
            quality_thresholds: QualityThresholds::default(),
            adaptive_control: AdaptiveQualityConfig::default(),
            quality_reporting: QualityReportingConfig::default(),
            correction_strategies: vec![
                CorrectionStrategy::AutomaticGainControl,
                CorrectionStrategy::NoiseSuppression,
            ],
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_snr: 20.0,
            max_thd_n: 0.1,
            min_dynamic_range: 40.0,
            max_noise_floor: -60.0,
            max_latency: Duration::from_millis(20),
        }
    }
}

impl Default for AdaptiveQualityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            adaptation_rate: 0.1,
            control_parameters: AdaptiveControlParameters::default(),
            feedback_mechanism: FeedbackMechanism::ProportionalIntegral,
        }
    }
}

impl Default for AdaptiveControlParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            smoothing_factor: 0.9,
            adaptation_window: Duration::from_secs(5),
            stability_threshold: 0.05,
        }
    }
}

impl Default for QualityReportingConfig {
    fn default() -> Self {
        Self {
            reporting_interval: Duration::from_secs(1),
            report_format: ReportFormat::JSON,
            include_visualizations: true,
            alert_thresholds: HashMap::new(),
        }
    }
}

impl Default for QualityAssessment {
    fn default() -> Self {
        Self::new()
    }
}
