//! # `VoiRS` Evaluation
//!
//! Comprehensive quality evaluation and assessment framework for the `VoiRS` ecosystem.
//! This crate provides tools for evaluating speech synthesis quality, pronunciation accuracy,
//! and comparative analysis between different models or systems.
//!
//! ## Features
//!
//! - **Quality Evaluation**: Objective and subjective quality metrics
//! - **Pronunciation Assessment**: Phoneme-level accuracy scoring
//! - **Comparative Analysis**: Side-by-side evaluation of different systems
//! - **Perceptual Metrics**: Human-perception-aligned quality measures
//! - **Automated Scoring**: ML-based quality prediction
//!
//! ## Quick Start
//!
//! ```rust
//! use voirs_evaluation::quality::QualityEvaluator;
//! use voirs_evaluation::traits::QualityEvaluator as QualityEvaluatorTrait;
//! use voirs_sdk::AudioBuffer;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create quality evaluator
//!     let evaluator = QualityEvaluator::new().await?;
//!     
//!     // Create test audio buffers
//!     let generated = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
//!     let reference = AudioBuffer::new(vec![0.12; 16000], 16000, 1);
//!     
//!     // Evaluate quality
//!     let quality = evaluator.evaluate_quality(&generated, Some(&reference), None).await?;
//!     println!("Quality score: {:.2}", quality.overall_score);
//!     
//! #   Ok(())
//! # }
//! ```

#![allow(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

// Re-export core VoiRS types
pub use voirs_recognizer::traits::{PhonemeAlignment, Transcript};
pub use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme, VoirsError};

// Public API modules
pub mod accuracy_benchmarks;
pub mod advanced_preprocessing;
pub mod audio;
pub mod automated_benchmarks;
pub mod benchmark_runner;
pub mod benchmarks;
/// Commercial tool comparison framework for speech evaluation systems
pub mod commercial_tool_comparison;
pub mod comparison;
pub mod compliance;
/// Cross-language evaluation accuracy validation framework
pub mod cross_language_validation;
/// Advanced data quality validation and dataset management utilities
pub mod data_quality_validation;
pub mod dataset_management;
pub mod distributed;
/// Enhanced error message generation utilities
pub mod error_enhancement;
pub mod fuzzing;
/// Ground truth dataset management for evaluation validation
pub mod ground_truth_dataset;
pub mod integration;
/// Enhanced logging and debugging utilities
pub mod logging;
/// Metric reliability and reproducibility testing framework
pub mod metric_reliability_testing;
pub mod perceptual;
pub mod performance;
/// Performance enhancement utilities for faster evaluation
pub mod performance_enhancements;
pub mod performance_monitor;
pub mod platform;
/// Plugin system for custom evaluation metrics
pub mod plugins;
/// Numerical precision utilities for high-accuracy calculations
pub mod precision;
pub mod pronunciation;
/// Protocol documentation and compliance validation utilities
pub mod protocol_documentation;
pub mod quality;
/// R statistical analysis integration (optional, requires R installation)
#[cfg(feature = "r-integration")]
pub mod r_integration;
/// R package creation foundation for VoiRS evaluation
#[cfg(feature = "r-integration")]
pub mod r_package_foundation;
pub mod regression_detector;
pub mod regression_testing;
/// REST API interface for evaluation services
pub mod rest_api;
pub mod statistical;
/// Enhanced statistical analysis utilities
pub mod statistical_enhancements;
pub mod traits;
pub mod validation;
/// WebSocket interface for real-time evaluation services
pub mod websocket;

// Python bindings (optional, enabled with "python" feature)
#[cfg(feature = "python")]
pub mod python;

// Re-export performance optimizations
pub use performance::{multi_gpu, LRUCache, PersistentCache, SlidingWindowProcessor};

// Re-export all public types from traits
pub use traits::*;

// Note: Feature module types are not glob re-exported to avoid ambiguity.
// Import from specific modules: evaluation::audio::*, evaluation::perceptual::*, etc.
// Or use the prelude: use voirs_evaluation::prelude::*;

// Re-export R integration when feature is enabled
#[cfg(feature = "r-integration")]
pub use r_integration::*;

// Re-export Python bindings when feature is enabled
#[cfg(feature = "python")]
pub use python::*;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Convenient prelude for common imports
pub mod prelude {
    //! Prelude module for convenient imports

    pub use crate::traits::{
        ComparativeEvaluator, ComparisonMetric, ComparisonResult, EvaluationResult,
        PronunciationEvaluator, PronunciationMetric, PronunciationScore,
        QualityEvaluator as QualityEvaluatorTrait, QualityMetric, QualityScore,
        SelfEvaluationResult, SelfEvaluator,
    };

    pub use crate::audio::{
        AudioFormat, AudioLoader, LoadOptions, StreamingConfig, StreamingEvaluator,
    };
    pub use crate::comparison::ComparativeEvaluatorImpl;
    pub use crate::compliance::{
        ComplianceChecker, ComplianceConfig, ComplianceResult, ComplianceStatus,
    };
    pub use crate::integration::{EcosystemConfig, EcosystemEvaluator, EcosystemResults};
    pub use crate::perceptual::{
        EnhancedMultiListenerSimulator, IntelligibilityMonitor, MultiListenerConfig,
    };
    pub use crate::performance_enhancements::{CacheStats, OptimizedQualityEvaluator};
    pub use crate::platform::{DeploymentConfig, PlatformCompatibility, PlatformInfo};
    pub use crate::plugins::{
        EvaluationContext, ExampleMetricPlugin, MetricPlugin, MetricResult, PluginConfig,
        PluginError, PluginInfo, PluginManager,
    };
    pub use crate::pronunciation::PronunciationEvaluatorImpl;
    pub use crate::quality::{
        AdvancedSpectralAnalysis, AgeGroup, ChildrenEvaluationConfig, ChildrenEvaluationResult,
        ChildrenSpeechEvaluator, CochlearImplantStrategy, CulturalRegion, ElderlyAgeGroup,
        ElderlyPathologicalConfig, ElderlyPathologicalEvaluator, ElderlyPathologicalResult,
        EmotionType, EmotionalEvaluationConfig, EmotionalSpeechEvaluationResult,
        EmotionalSpeechEvaluator, ExpressionStyle, HearingAidType, ModelArchitecture, NeuralConfig,
        NeuralEvaluator, NeuralQualityAssessment, PathologicalCondition, PersonalityTrait,
        PsychoacousticAnalysis, PsychoacousticConfig, PsychoacousticEvaluator, QualityEvaluator,
        SeverityLevel, SingingEvaluationConfig, SingingEvaluationResult, SingingEvaluator,
        SpectralAnalysisConfig, SpectralAnalyzer,
    };
    pub use crate::validation::{ValidationConfig, ValidationFramework, ValidationResult};
    pub use crate::websocket::{
        RealtimeAnalysis, SessionConfig, WebSocketConfig, WebSocketError, WebSocketMessage,
        WebSocketSessionManager,
    };

    // Re-export R integration when feature is enabled
    #[cfg(feature = "r-integration")]
    pub use crate::r_integration::{
        RAnovaResult, RArimaModel, RDataFrame, RGamModel, RKmeansResult, RLinearModel,
        RLogisticModel, RPcaResult, RRandomForestModel, RSession, RSurvivalModel, RTestResult,
        RTimeSeriesResult, RValue,
    };

    // Re-export SDK types
    pub use voirs_recognizer::traits::{PhonemeAlignment, Transcript};
    pub use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme, VoirsError};

    // Re-export async trait
    pub use async_trait::async_trait;
}

// ============================================================================
// Error Types
// ============================================================================

/// Evaluation-specific error types
#[derive(Debug, thiserror::Error)]
pub enum EvaluationError {
    /// Quality evaluation failed
    #[error("Quality evaluation failed: {message}")]
    QualityEvaluationError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Pronunciation evaluation failed
    #[error("Pronunciation evaluation failed: {message}")]
    PronunciationEvaluationError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Comparison evaluation failed
    #[error("Comparison evaluation failed: {message}")]
    ComparisonError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Metric calculation failed
    #[error("Metric calculation failed: {metric} - {message}")]
    MetricCalculationError {
        /// Metric name
        metric: String,
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Audio processing error
    #[error("Audio processing error: {message}")]
    AudioProcessingError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// General processing error
    #[error("Processing error: {message}")]
    ProcessingError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigurationError {
        /// Error message
        message: String,
    },

    /// Model error
    #[error("Model error: {message}")]
    ModelError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Invalid input
    #[error("Invalid input: {message}")]
    InvalidInput {
        /// Error message
        message: String,
    },

    /// Feature not supported
    #[error("Feature not supported: {feature}")]
    FeatureNotSupported {
        /// Feature name
        feature: String,
    },
}

impl From<EvaluationError> for VoirsError {
    fn from(err: EvaluationError) -> Self {
        match err {
            EvaluationError::QualityEvaluationError { message, source } => {
                VoirsError::ModelError {
                    model_type: voirs_sdk::error::ModelType::Vocoder, // Use closest type
                    message,
                    source,
                }
            }
            EvaluationError::PronunciationEvaluationError { message, source } => {
                VoirsError::ModelError {
                    model_type: voirs_sdk::error::ModelType::ASR,
                    message,
                    source,
                }
            }
            EvaluationError::ComparisonError { message, source: _ } => VoirsError::AudioError {
                message,
                buffer_info: None,
            },
            EvaluationError::MetricCalculationError {
                metric,
                message,
                source: _,
            } => VoirsError::AudioError {
                message: format!("Metric calculation failed: {metric} - {message}"),
                buffer_info: None,
            },
            EvaluationError::AudioProcessingError { message, source: _ } => {
                VoirsError::AudioError {
                    message,
                    buffer_info: None,
                }
            }
            EvaluationError::ConfigurationError { message } => VoirsError::ConfigError {
                field: "evaluation".to_string(),
                message,
            },
            EvaluationError::ModelError { message, source } => VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::Vocoder,
                message,
                source,
            },
            EvaluationError::InvalidInput { message } => VoirsError::ConfigError {
                field: "input".to_string(),
                message: format!("Invalid input: {message}"),
            },
            EvaluationError::FeatureNotSupported { feature } => VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::Vocoder,
                message: format!("Feature not supported: {feature}"),
                source: None,
            },
            EvaluationError::ProcessingError { message, source: _ } => VoirsError::AudioError {
                message,
                buffer_info: None,
            },
        }
    }
}

impl From<VoirsError> for EvaluationError {
    fn from(err: VoirsError) -> Self {
        match err {
            VoirsError::ModelError {
                model_type: _,
                message,
                source,
            } => EvaluationError::ModelError { message, source },
            VoirsError::AudioError {
                message,
                buffer_info: _,
            } => EvaluationError::AudioProcessingError {
                message,
                source: None,
            },
            VoirsError::ConfigError { field: _, message } => {
                EvaluationError::ConfigurationError { message }
            }
            VoirsError::G2pError { message, .. } => EvaluationError::ModelError {
                message,
                source: None,
            },
            VoirsError::NetworkError { message, .. } => EvaluationError::ModelError {
                message: format!("Network error: {message}"),
                source: None,
            },
            VoirsError::IoError {
                path, operation, ..
            } => EvaluationError::ModelError {
                message: format!("IO error: {} on {}", operation, path.display()),
                source: None,
            },
            VoirsError::DataValidationFailed { data_type, reason } => {
                EvaluationError::InvalidInput {
                    message: format!("Validation failed for {data_type}: {reason}"),
                }
            }
            VoirsError::TextPreprocessingError { message, .. } => {
                EvaluationError::AudioProcessingError {
                    message: format!("Text preprocessing error: {message}"),
                    source: None,
                }
            }
            VoirsError::NotImplemented { feature } => {
                EvaluationError::FeatureNotSupported { feature }
            }
            VoirsError::ResourceExhausted { resource, details } => EvaluationError::ModelError {
                message: format!("Resource exhausted: {resource}: {details}"),
                source: None,
            },
            VoirsError::InternalError { component, message } => EvaluationError::ModelError {
                message: format!("Internal error in {component}: {message}"),
                source: None,
            },
            _ => EvaluationError::ModelError {
                message: format!("Unknown error: {err}"),
                source: None,
            },
        }
    }
}

impl From<scirs2_fft::error::FFTError> for EvaluationError {
    fn from(err: scirs2_fft::error::FFTError) -> Self {
        EvaluationError::AudioProcessingError {
            message: format!("FFT computation error: {err}"),
            source: Some(Box::new(err)),
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Create a default quality evaluation configuration
#[must_use]
pub fn default_quality_config() -> QualityEvaluationConfig {
    QualityEvaluationConfig::default()
}

/// Create a default pronunciation evaluation configuration
#[must_use]
pub fn default_pronunciation_config() -> PronunciationEvaluationConfig {
    PronunciationEvaluationConfig::default()
}

/// Create a default comparison configuration
#[must_use]
pub fn default_comparison_config() -> ComparisonConfig {
    ComparisonConfig::default()
}

/// Validate audio compatibility for evaluation
pub fn validate_audio_compatibility(
    audio1: &AudioBuffer,
    audio2: &AudioBuffer,
) -> Result<(), EvaluationError> {
    if audio1.sample_rate() != audio2.sample_rate() {
        return Err(EvaluationError::InvalidInput {
            message: format!(
                "Sample rate mismatch: {} vs {}",
                audio1.sample_rate(),
                audio2.sample_rate()
            ),
        });
    }

    if audio1.channels() != audio2.channels() {
        return Err(EvaluationError::InvalidInput {
            message: format!(
                "Channel count mismatch: {} vs {}",
                audio1.channels(),
                audio2.channels()
            ),
        });
    }

    Ok(())
}

/// Calculate statistical correlation between two score vectors
#[must_use]
pub fn calculate_correlation(scores1: &[f32], scores2: &[f32]) -> f32 {
    if scores1.len() != scores2.len() || scores1.is_empty() {
        return 0.0;
    }

    let n = scores1.len() as f32;
    let mean1 = scores1.iter().sum::<f32>() / n;
    let mean2 = scores2.iter().sum::<f32>() / n;

    let mut numerator = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;

    for (&s1, &s2) in scores1.iter().zip(scores2.iter()) {
        let diff1 = s1 - mean1;
        let diff2 = s2 - mean2;
        numerator += diff1 * diff2;
        sum_sq1 += diff1 * diff1;
        sum_sq2 += diff2 * diff2;
    }

    let denominator = (sum_sq1 * sum_sq2).sqrt();
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Convert quality score to human-readable label
#[must_use]
pub fn quality_score_to_label(score: f32) -> &'static str {
    match score {
        s if s >= 0.9 => "Excellent",
        s if s >= 0.8 => "Good",
        s if s >= 0.7 => "Fair",
        s if s >= 0.6 => "Poor",
        _ => "Very Poor",
    }
}

/// Convert pronunciation score to human-readable label
#[must_use]
pub fn pronunciation_score_to_label(score: f32) -> &'static str {
    match score {
        s if s >= 0.95 => "Native-like",
        s if s >= 0.85 => "Very Good",
        s if s >= 0.75 => "Good",
        s if s >= 0.65 => "Acceptable",
        s if s >= 0.5 => "Needs Improvement",
        _ => "Poor",
    }
}

/// Utility function to normalize scores to 0-1 range
pub fn normalize_scores(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }

    let min_score = scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    if max_score > min_score {
        let range = max_score - min_score;
        for score in scores {
            *score = (*score - min_score) / range;
        }
    }
}

/// Calculate weighted average of scores
#[must_use]
pub fn weighted_average(scores: &[f32], weights: &[f32]) -> f32 {
    if scores.len() != weights.len() || scores.is_empty() {
        return 0.0;
    }

    let weighted_sum: f32 = scores.iter().zip(weights.iter()).map(|(s, w)| s * w).sum();
    let weight_sum: f32 = weights.iter().sum();

    if weight_sum > 0.0 {
        weighted_sum / weight_sum
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_audio_compatibility_validation() {
        let audio1 = AudioBuffer::new(vec![0.1, 0.2, 0.3], 16000, 1);
        let audio2 = AudioBuffer::new(vec![0.4, 0.5, 0.6], 16000, 1);

        // Should be compatible
        assert!(validate_audio_compatibility(&audio1, &audio2).is_ok());

        // Different sample rates
        let audio3 = AudioBuffer::new(vec![0.1, 0.2, 0.3], 22050, 1);
        assert!(validate_audio_compatibility(&audio1, &audio3).is_err());

        // Different channel counts
        let audio4 = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 16000, 2);
        assert!(validate_audio_compatibility(&audio1, &audio4).is_err());
    }

    #[test]
    fn test_correlation_calculation() {
        let scores1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scores2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let correlation = calculate_correlation(&scores1, &scores2);
        assert!((correlation - 1.0).abs() < 0.001); // Perfect correlation

        let scores3 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let correlation_neg = calculate_correlation(&scores1, &scores3);
        assert!((correlation_neg + 1.0).abs() < 0.001); // Perfect negative correlation
    }

    #[test]
    fn test_quality_score_labels() {
        assert_eq!(quality_score_to_label(0.95), "Excellent");
        assert_eq!(quality_score_to_label(0.85), "Good");
        assert_eq!(quality_score_to_label(0.75), "Fair");
        assert_eq!(quality_score_to_label(0.65), "Poor");
        assert_eq!(quality_score_to_label(0.45), "Very Poor");
    }

    #[test]
    fn test_pronunciation_score_labels() {
        assert_eq!(pronunciation_score_to_label(0.97), "Native-like");
        assert_eq!(pronunciation_score_to_label(0.87), "Very Good");
        assert_eq!(pronunciation_score_to_label(0.77), "Good");
        assert_eq!(pronunciation_score_to_label(0.67), "Acceptable");
        assert_eq!(pronunciation_score_to_label(0.57), "Needs Improvement");
        assert_eq!(pronunciation_score_to_label(0.37), "Poor");
    }

    #[test]
    fn test_score_normalization() {
        let mut scores = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        normalize_scores(&mut scores);

        assert!((scores[0] - 0.0).abs() < 0.001);
        assert!((scores[4] - 1.0).abs() < 0.001);
        assert!(scores.iter().all(|&s| (0.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_weighted_average() {
        let scores = vec![0.8, 0.6, 0.9];
        let weights = vec![0.5, 0.3, 0.2];

        let avg = weighted_average(&scores, &weights);
        let expected = (0.8 * 0.5 + 0.6 * 0.3 + 0.9 * 0.2) / (0.5 + 0.3 + 0.2);
        assert!((avg - expected).abs() < 0.001);
    }

    #[test]
    fn test_default_configs() {
        let quality_config = default_quality_config();
        assert!(quality_config.objective_metrics);

        let pronunciation_config = default_pronunciation_config();
        assert!(pronunciation_config.phoneme_level_scoring);

        let comparison_config = default_comparison_config();
        assert!(comparison_config.enable_statistical_analysis);
    }
}
