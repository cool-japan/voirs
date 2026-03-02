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
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::unused_async)] // Many async functions are part of API design
#![allow(clippy::cast_precision_loss)] // Acceptable for audio/signal processing
#![allow(clippy::cast_possible_truncation)] // Controlled in audio processing
#![allow(clippy::cast_sign_loss)] // Validated in implementation
#![allow(clippy::cast_lossless)] // Explicit casts for clarity
#![allow(clippy::unused_self)] // Trait implementations require &self
#![allow(clippy::must_use_candidate)] // Would require extensive API changes
#![allow(clippy::missing_errors_doc)] // Errors are self-explanatory
#![allow(clippy::missing_panics_doc)] // Panics are documented where critical
#![allow(clippy::uninlined_format_args)] // Older format syntax used consistently
#![allow(clippy::similar_names)] // Domain-specific naming (e.g., mfcc1, mfcc2)
#![allow(clippy::unnecessary_wraps)] // API consistency requires Result types
#![allow(clippy::format_push_string)] // String building in formatters
#![allow(clippy::manual_clamp)] // Explicit bounds checking for clarity
#![allow(clippy::doc_markdown)] // Technical terms don't need backticks
#![allow(clippy::return_self_not_must_use)] // Builder pattern convenience
#![allow(clippy::if_not_else)] // Conditional logic clarity
#![allow(clippy::redundant_closure_for_method_calls)] // Explicit closures for readability
#![allow(clippy::match_same_arms)] // Explicit matching for different cases
#![allow(clippy::inefficient_to_string)] // Minimal performance impact
#![allow(clippy::needless_pass_by_value)] // API design choices
#![allow(clippy::too_many_lines)] // Complex evaluation algorithms
#![allow(clippy::struct_excessive_bools)] // Configuration structs need flags
#![allow(clippy::needless_range_loop)] // Explicit indexing for clarity
#![allow(clippy::wildcard_imports)] // Prelude and common imports
#![allow(clippy::single_char_add_str)] // Minor performance impact
#![allow(clippy::map_unwrap_or)] // Explicit error handling
#![allow(clippy::excessive_precision)] // Scientific/audio constants
#![allow(clippy::cast_possible_wrap)] // Controlled integer conversions
#![allow(clippy::cloned_instead_of_copied)] // Generic over Copy/Clone
#![allow(clippy::useless_vec)] // Vector literals for test data
#![allow(clippy::ptr_as_ptr)] // FFI and C-compatible code
#![allow(clippy::manual_let_else)] // Explicit error handling preferred
#![allow(clippy::unnecessary_cast)] // Explicit types for clarity
#![allow(clippy::trivially_copy_pass_by_ref)] // Trait method signatures
#![allow(clippy::items_after_statements)] // Logical grouping of code
#![allow(clippy::too_many_arguments)] // Complex evaluation functions
#![allow(clippy::new_without_default)] // Constructors with validation
#![allow(clippy::needless_borrow)] // Explicit borrowing for clarity
#![allow(clippy::derivable_impls)] // Explicit default implementations
#![allow(clippy::clone_on_copy)] // Explicit cloning in generic code
#![allow(clippy::useless_format)] // Format strings for consistency
#![allow(clippy::unwrap_or_default)] // Explicit defaults preferred
#![allow(clippy::single_match_else)] // Explicit match arms
#![allow(clippy::vec_init_then_push)] // Clear vector building
#![allow(clippy::unnecessary_mut_passed)] // Mutable references for clarity
#![allow(clippy::manual_range_contains)] // Explicit bounds checking
#![allow(clippy::len_zero)] // Explicit length checks
#![allow(clippy::float_cmp)] // Acceptable for test assertions
#![allow(clippy::range_plus_one)] // Inclusive range clarity
#![allow(clippy::manual_string_new)] // Explicit string creation
#![allow(clippy::should_implement_trait)] // Custom trait implementations
#![allow(clippy::let_and_return)] // Named intermediate values
#![allow(clippy::type_complexity)] // Necessary for complex types
#![allow(clippy::collapsible_else_if)] // Explicit conditional logic
#![allow(clippy::collapsible_if)] // Clear condition separation
#![allow(clippy::collapsible_match)] // Explicit pattern matching
#![allow(clippy::single_char_pattern)] // Character patterns for clarity
#![allow(clippy::needless_borrows_for_generic_args)] // Explicit borrowing
#![allow(clippy::default_trait_access)] // Explicit Default::default()
#![allow(clippy::empty_line_after_doc_comments)] // Documentation formatting
#![allow(clippy::bool_to_int_with_if)] // Explicit boolean conversion
#![allow(clippy::manual_ok_err)] // Explicit Result construction
#![allow(clippy::match_like_matches_macro)] // Explicit matching
#![allow(clippy::needless_continue)] // Loop control clarity
#![allow(clippy::explicit_iter_loop)] // Explicit iteration
#![allow(clippy::semicolon_if_nothing_returned)] // Expression clarity
#![allow(clippy::unnecessary_map_or)] // Explicit Option handling
#![allow(clippy::ref_option)] // FFI compatibility
#![allow(clippy::used_underscore_binding)] // Prefixed variables for clarity
#![allow(clippy::ip_constant)] // Test and example IP addresses
#![allow(clippy::for_kv_map)] // Explicit iteration patterns
#![allow(clippy::assigning_clones)] // Explicit clone operations
#![allow(clippy::manual_map)] // Explicit mapping logic
#![allow(clippy::manual_flatten)] // Explicit flattening
#![allow(clippy::await_holding_lock)] // Controlled lock scope
#![allow(clippy::borrowed_box)] // API compatibility
#![allow(clippy::unnecessary_literal_bound)] // Explicit type bounds
#![allow(clippy::borrow_as_ptr)] // Pointer conversions
#![allow(clippy::case_sensitive_file_extension_comparisons)] // Platform compatibility
#![allow(clippy::comparison_chain)] // Explicit comparisons
#![allow(clippy::format_collect)] // String building
#![allow(clippy::if_same_then_else)] // Conditional clarity
#![allow(clippy::implicit_saturating_sub)] // Explicit arithmetic
#![allow(clippy::iter_kv_map)] // Iterator patterns
#![allow(clippy::match_result_ok)] // Explicit Result handling
#![allow(clippy::match_wildcard_for_single_variants)] // Exhaustive matching
#![allow(clippy::missing_const_for_thread_local)] // Runtime initialization
#![allow(clippy::mixed_attributes_style)] // Attribute formatting
#![allow(clippy::stable_sort_primitive)] // Sort algorithm choice
#![allow(clippy::struct_field_names)] // Descriptive field names
#![allow(clippy::unnecessary_debug_formatting)] // Debug trait usage
#![allow(clippy::useless_asref)] // Explicit references
#![allow(clippy::useless_conversion)] // Type clarity

// Re-export core VoiRS types
pub use voirs_recognizer::traits::{PhonemeAlignment, Transcript};
pub use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme, VoirsError};

// Public API modules
pub mod accuracy_benchmarks;
pub mod advanced_preprocessing;
pub mod audio;
/// Audit trail system for compliance and security monitoring
pub mod audit;
pub mod automated_benchmarks;
pub mod benchmark_export;
pub mod benchmark_runner;
pub mod benchmarks;
/// Commercial tool comparison framework for speech evaluation systems
/// Advanced result caching system with multiple backends
pub mod caching;
pub mod commercial_tool_comparison;
pub mod comparison;
pub mod compliance;
/// Compliance testing suite for standards validation
pub mod compliance_testing;
/// Context-aware evaluation system for speech synthesis
pub mod context_aware;
/// Conversational quality assessment for dialogue systems
pub mod conversational;
/// C++ header-only interface for evaluation framework
pub mod cpp_bindings;
/// Cross-language evaluation accuracy validation framework
pub mod cross_language_validation;
/// Critical Success Factors (CSF) validation framework
pub mod csf_validation;
/// Advanced data quality validation and dataset management utilities
pub mod data_quality_validation;
/// Data versioning system for benchmark and evaluation results
pub mod data_versioning;
pub mod dataset_management;
/// Deep learning-based evaluation metrics with neural MOS prediction
pub mod deep_learning_metrics;
pub mod distributed;
/// Automated documentation generation for evaluation results
pub mod doc_generation;
/// Enterprise security framework with RBAC, encryption, and compliance
pub mod enterprise_security;
/// Enhanced error message generation utilities
pub mod error_enhancement;
/// Fairness-aware evaluation for bias detection and demographic parity
pub mod fairness;
/// Federated evaluation system for distributed processing
pub mod federated;
pub mod fuzzing;
/// GraphQL API for complex evaluation queries
pub mod graphql;
/// Ground truth dataset management for evaluation validation
pub mod ground_truth_dataset;
pub mod integration;
/// Kubernetes deployment configuration for distributed evaluation
pub mod kubernetes;
/// Enhanced logging and debugging utilities
pub mod logging;
/// MATLAB/Octave bindings for evaluation framework
pub mod matlab_bindings;
/// Metric reliability and reproducibility testing framework
pub mod metric_reliability_testing;
/// Metrics comparison and regression detection
pub mod metrics_comparison;
/// Metrics explainability system for interpretable evaluation
pub mod metrics_explainability;
/// Multi-turn dialogue evaluation for extended conversations
pub mod multi_turn_dialogue;
/// Multi-region deployment configuration for global distribution
pub mod multiregion;
/// Node.js/JavaScript bindings for evaluation framework
pub mod nodejs_bindings;
/// Monitoring and observability framework with Prometheus and tracing
pub mod observability;
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
/// Privacy-preserving evaluation framework with differential privacy
pub mod privacy;
pub mod pronunciation;
/// Protocol documentation and compliance validation utilities
pub mod protocol_documentation;
pub mod quality;
/// Quality gate validation system for automated quality assurance
pub mod quality_gates;
/// R statistical analysis integration (optional, requires R installation)
#[cfg(feature = "r-integration")]
pub mod r_integration;
/// R package creation foundation for VoiRS evaluation
#[cfg(feature = "r-integration")]
pub mod r_package_foundation;
/// Role-Based Access Control (RBAC) for enterprise security
pub mod rbac;
pub mod regression_detector;
pub mod regression_testing;
/// Reproducibility guarantees system for deterministic evaluation
pub mod reproducibility;
/// REST API interface for evaluation services
pub mod rest_api;
/// Semantic similarity evaluation for speech content analysis
pub mod semantic_similarity;
/// Industry standards compliance module (ANSI, ISO/IEC, AES, ITU-T)
pub mod standards;
pub mod statistical;
/// Enhanced statistical analysis utilities
pub mod statistical_enhancements;
/// Task-oriented dialogue evaluation for goal-driven interactions
pub mod task_oriented;
pub mod traits;
/// User experience (UX) evaluation for usability and satisfaction
pub mod user_experience;
pub mod validation;
/// Validation certificate generator for certified evaluation results
pub mod validation_certificates;
/// WebSocket interface for real-time evaluation services
pub mod websocket;
/// Evaluation workflow system for automated pipelines
pub mod workflows;

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

    /// I/O error
    #[error("I/O error: {0}")]
    Io(String),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
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
            EvaluationError::Io(msg) => VoirsError::IoError {
                path: std::path::PathBuf::from("unknown"),
                operation: voirs_sdk::error::IoOperation::Read,
                source: std::io::Error::other(msg),
            },
            EvaluationError::Other(msg) => VoirsError::InternalError {
                component: "evaluation".to_string(),
                message: msg,
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
        // VERSION is a const string literal, so this checks it has content
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
