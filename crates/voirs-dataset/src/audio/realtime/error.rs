//! Error handling and recovery for real-time audio processing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Error handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHandlingConfig {
    /// Error recovery strategies
    pub recovery_strategies: Vec<ErrorRecoveryStrategy>,
    /// Error reporting
    pub error_reporting: ErrorReportingConfig,
    /// Graceful degradation
    pub graceful_degradation: GracefulDegradationConfig,
    /// Fallback processing
    pub fallback_processing: FallbackProcessingConfig,
}

/// Error recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorRecoveryStrategy {
    /// Retry processing
    Retry { max_attempts: usize },
    /// Skip problematic data
    Skip,
    /// Use fallback processing
    Fallback,
    /// Interpolate missing data
    Interpolate,
    /// Reset processing state
    Reset,
}

/// Error reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReportingConfig {
    /// Error log level
    pub log_level: LogLevel,
    /// Error aggregation
    pub error_aggregation: bool,
    /// Error rate monitoring
    pub error_rate_monitoring: bool,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f32>,
}

/// Log levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    /// Trace level
    Trace,
    /// Debug level
    Debug,
    /// Info level
    Info,
    /// Warning level
    Warning,
    /// Error level
    Error,
    /// Critical level
    Critical,
}

/// Graceful degradation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GracefulDegradationConfig {
    /// Enable graceful degradation
    pub enabled: bool,
    /// Degradation levels
    pub degradation_levels: Vec<DegradationLevel>,
    /// Recovery conditions
    pub recovery_conditions: Vec<RecoveryCondition>,
}

/// Degradation levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DegradationLevel {
    /// Reduce processing quality
    ReduceQuality,
    /// Increase buffer size
    IncreaseBufferSize,
    /// Disable non-essential processing
    DisableNonEssential,
    /// Reduce sample rate
    ReduceSampleRate,
    /// Mono processing
    MonoProcessing,
}

/// Recovery conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryCondition {
    /// System load threshold
    SystemLoadThreshold(f32),
    /// Error rate threshold
    ErrorRateThreshold(f32),
    /// Latency threshold
    LatencyThreshold(Duration),
    /// Quality threshold
    QualityThreshold(f32),
}

/// Fallback processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackProcessingConfig {
    /// Enable fallback processing
    pub enabled: bool,
    /// Fallback chain
    pub fallback_chain: Vec<FallbackProcessor>,
    /// Fallback triggers
    pub fallback_triggers: Vec<FallbackTrigger>,
}

/// Fallback processors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackProcessor {
    /// Simple pass-through
    PassThrough,
    /// Basic noise reduction
    BasicNoiseReduction,
    /// Simple normalization
    SimpleNormalization,
    /// Silence detection
    SilenceDetection,
}

/// Fallback triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackTrigger {
    /// Processing timeout
    ProcessingTimeout(Duration),
    /// Quality threshold
    QualityThreshold(f32),
    /// Error rate threshold
    ErrorRateThreshold(f32),
    /// System overload
    SystemOverload,
}

impl Default for ErrorHandlingConfig {
    fn default() -> Self {
        Self {
            recovery_strategies: vec![
                ErrorRecoveryStrategy::Retry { max_attempts: 3 },
                ErrorRecoveryStrategy::Fallback,
            ],
            error_reporting: ErrorReportingConfig::default(),
            graceful_degradation: GracefulDegradationConfig::default(),
            fallback_processing: FallbackProcessingConfig::default(),
        }
    }
}

impl Default for ErrorReportingConfig {
    fn default() -> Self {
        Self {
            log_level: LogLevel::Warning,
            error_aggregation: true,
            error_rate_monitoring: true,
            alert_thresholds: HashMap::new(),
        }
    }
}

impl Default for GracefulDegradationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            degradation_levels: vec![
                DegradationLevel::DisableNonEssential,
                DegradationLevel::IncreaseBufferSize,
                DegradationLevel::ReduceQuality,
            ],
            recovery_conditions: vec![
                RecoveryCondition::SystemLoadThreshold(0.7),
                RecoveryCondition::ErrorRateThreshold(0.1),
            ],
        }
    }
}

impl Default for FallbackProcessingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            fallback_chain: vec![
                FallbackProcessor::BasicNoiseReduction,
                FallbackProcessor::SimpleNormalization,
                FallbackProcessor::PassThrough,
            ],
            fallback_triggers: vec![
                FallbackTrigger::ProcessingTimeout(Duration::from_millis(100)),
                FallbackTrigger::QualityThreshold(0.3),
            ],
        }
    }
}
