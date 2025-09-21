//! Latency configuration and optimization for real-time audio processing

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Latency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyConfig {
    /// Target latency
    pub target_latency: Duration,
    /// Maximum acceptable latency
    pub max_latency: Duration,
    /// Latency monitoring
    pub latency_monitoring: LatencyMonitoringConfig,
    /// Latency optimization
    pub latency_optimization: LatencyOptimizationConfig,
}

/// Latency monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMonitoringConfig {
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Latency measurement methods
    pub measurement_methods: Vec<LatencyMeasurementMethod>,
    /// Statistical analysis
    pub statistical_analysis: StatisticalAnalysisConfig,
}

/// Latency measurement methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatencyMeasurementMethod {
    /// Round-trip time measurement
    RoundTripTime,
    /// Buffer delay measurement
    BufferDelay,
    /// Processing time measurement
    ProcessingTime,
    /// Network latency measurement
    NetworkLatency,
}

/// Statistical analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysisConfig {
    /// Window size for analysis
    pub window_size: Duration,
    /// Percentiles to track
    pub percentiles: Vec<f32>,
    /// Enable outlier detection
    pub outlier_detection: bool,
    /// Outlier threshold
    pub outlier_threshold: f32,
}

/// Latency optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LatencyOptimizationConfig {
    /// Automatic optimization
    pub automatic_optimization: AutomaticOptimizationConfig,
    /// Manual optimization
    pub manual_optimization: ManualOptimizationConfig,
}

/// Automatic optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomaticOptimizationConfig {
    /// Enable automatic optimization
    pub enabled: bool,
    /// Optimization interval
    pub optimization_interval: Duration,
    /// Stability requirements
    pub stability_requirements: StabilityRequirements,
    /// Convergence criteria
    pub convergence_criteria: ConvergenceCriteria,
}

/// Stability requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityRequirements {
    /// Minimum stable period
    pub min_stable_period: Duration,
    /// Maximum latency variance
    pub max_latency_variance: f32,
    /// Stability threshold
    pub stability_threshold: f32,
}

/// Convergence criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceCriteria {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f32,
    /// Improvement threshold
    pub improvement_threshold: f32,
}

/// Manual optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualOptimizationConfig {
    /// Buffer size adjustments
    pub buffer_size_adjustments: Vec<usize>,
    /// Thread priority adjustments
    pub thread_priority_adjustments: Vec<String>,
    /// Algorithm parameter adjustments
    pub algorithm_parameter_adjustments: std::collections::HashMap<String, f32>,
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            target_latency: Duration::from_millis(10),
            max_latency: Duration::from_millis(20),
            latency_monitoring: LatencyMonitoringConfig::default(),
            latency_optimization: LatencyOptimizationConfig::default(),
        }
    }
}

impl Default for LatencyMonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_millis(10),
            measurement_methods: vec![
                LatencyMeasurementMethod::ProcessingTime,
                LatencyMeasurementMethod::BufferDelay,
            ],
            statistical_analysis: StatisticalAnalysisConfig::default(),
        }
    }
}

impl Default for StatisticalAnalysisConfig {
    fn default() -> Self {
        Self {
            window_size: Duration::from_secs(10),
            percentiles: vec![50.0, 95.0, 99.0],
            outlier_detection: true,
            outlier_threshold: 3.0,
        }
    }
}

impl Default for AutomaticOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimization_interval: Duration::from_secs(30),
            stability_requirements: StabilityRequirements::default(),
            convergence_criteria: ConvergenceCriteria::default(),
        }
    }
}

impl Default for StabilityRequirements {
    fn default() -> Self {
        Self {
            min_stable_period: Duration::from_secs(5),
            max_latency_variance: 0.1,
            stability_threshold: 0.05,
        }
    }
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 0.01,
            improvement_threshold: 0.001,
        }
    }
}

impl Default for ManualOptimizationConfig {
    fn default() -> Self {
        Self {
            buffer_size_adjustments: vec![512, 1024, 2048],
            thread_priority_adjustments: vec!["high".to_string(), "realtime".to_string()],
            algorithm_parameter_adjustments: std::collections::HashMap::new(),
        }
    }
}
