//! # `VoiRS` Recognition
//!
//! Voice recognition and analysis capabilities for the `VoiRS` ecosystem.
//! This crate provides automatic speech recognition (ASR), phoneme recognition,
//! and comprehensive audio analysis functionality.
//!
//! ## Features
//!
//! - **ASR Models**: Whisper, `DeepSpeech`, `Wav2Vec2` support
//! - **Phoneme Recognition**: Forced alignment and automatic recognition
//! - **Audio Analysis**: Quality metrics, prosody, speaker characteristics
//! - **Streaming Support**: Real-time processing capabilities
//! - **Multi-language**: Support for multiple languages and accents
//!
//! ## Quick Start
//!
//! ### Basic Audio Analysis
//!
//! ```rust,no_run
//! use voirs_recognizer::prelude::*;
//! use voirs_recognizer::RecognitionError;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), RecognitionError> {
//!     // Create audio buffer with some sample data
//!     let samples = vec![0.0f32; 16000]; // 1 second of silence at 16kHz
//!     let audio = AudioBuffer::mono(samples, 16000);
//!     
//!     // Create audio analyzer for comprehensive analysis
//!     let analyzer_config = AudioAnalysisConfig::default();
//!     let analyzer = AudioAnalyzerImpl::new(analyzer_config).await?;
//!     let analysis = analyzer.analyze(&audio, Some(&AudioAnalysisConfig::default())).await?;
//!     
//!     // Access quality metrics
//!     if let Some(snr) = analysis.quality_metrics.get("snr") {
//!         println!("Audio analysis complete: SNR = {:.2}", snr);
//!     }
//!     
//!     Ok(())
//! }
//! ```
//!
//! ### ASR Accuracy Validation
//!
//! ```rust,no_run
//! use voirs_recognizer::prelude::*;
//! use voirs_recognizer::{RecognitionError, asr::BenchmarkingConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), RecognitionError> {
//!     // Create a benchmarking suite with default configuration
//!     let benchmark_config = BenchmarkingConfig::default();
//!     let benchmark_suite = ASRBenchmarkingSuite::new(benchmark_config).await?;
//!     
//!     // Create an accuracy validator with standard requirements
//!     let accuracy_validator = AccuracyValidator::new_standard();
//!     
//!     // Validate accuracy against standard benchmarks
//!     let validation_report = accuracy_validator.validate_accuracy(&benchmark_suite).await?;
//!     
//!     // Generate and display validation report
//!     let summary = accuracy_validator.generate_summary_report(&validation_report);
//!     println!("Accuracy Validation Results:\n{}", summary);
//!     
//!     // Check if all requirements passed
//!     if validation_report.overall_passed {
//!         println!("✅ All accuracy requirements passed!");
//!     } else {
//!         println!("❌ Some accuracy requirements failed.");
//!         println!("Passed: {}/{}", validation_report.passed_requirements, validation_report.total_requirements);
//!     }
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Performance Tuning Guide
//!
//! ### Model Selection for Performance
//!
//! Choose the appropriate model size based on your performance requirements:
//!
//! ```rust,no_run
//! use voirs_recognizer::prelude::*;
//! use voirs_recognizer::asr::whisper::WhisperConfig;
//!
//! // For real-time applications with tight latency constraints
//! let fast_config = WhisperConfig {
//!     model_size: "tiny".to_string(),
//!     ..Default::default()
//! };
//!
//! // For balanced performance and accuracy  
//! let balanced_config = WhisperConfig {
//!     model_size: "base".to_string(),
//!     ..Default::default()
//! };
//!
//! // For highest accuracy (higher latency)
//! let accurate_config = WhisperConfig {
//!     model_size: "small".to_string(),
//!     ..Default::default()
//! };
//! ```
//!
//! ### Memory Optimization
//!
//! - **Model Quantization**: Use INT8 or FP16 quantization to reduce memory usage
//! - **Batch Processing**: Process multiple audio files together for better throughput
//! - **Memory Pools**: Enable GPU memory pooling for efficient tensor reuse
//!
//! ### Real-time Processing Optimization
//!
//! ```rust,no_run
//! use voirs_recognizer::prelude::*;
//! use voirs_recognizer::integration::config::{StreamingConfig, LatencyMode};
//!
//! // Configure for ultra-low latency
//! let streaming_config = StreamingConfig {
//!     latency_mode: LatencyMode::UltraLow,
//!     chunk_size: 1600,          // Smaller chunks for lower latency (100ms at 16kHz)
//!     overlap: 400,              // Minimal overlap (25ms at 16kHz)  
//!     buffer_duration: 3.0,      // Limited buffer for speed
//! };
//! ```
//!
//! ### Performance Monitoring
//!
//! Monitor your application's performance to ensure it meets requirements:
//!
//! ```rust,no_run
//! use voirs_recognizer::prelude::*;
//! use std::time::Duration;
//!
//! let validator = PerformanceValidator::new()
//!     .with_verbose(true);
//!
//! let requirements = PerformanceRequirements {
//!     max_rtf: 0.3,              // Real-time factor < 0.3
//!     max_memory_usage: 2_000_000_000, // < 2GB
//!     max_startup_time_ms: 5000, // < 5 seconds
//!     max_streaming_latency_ms: 200, // < 200ms
//! };
//!
//! // Validate streaming latency
//! let latency = Duration::from_millis(150);
//! let (latency_ms, passed) = validator.validate_streaming_latency(latency);
//! if !passed {
//!     println!("Streaming latency {} ms exceeds requirement {} ms",
//!              latency_ms, requirements.max_streaming_latency_ms);
//! }
//! ```
//!
//! ### Platform-Specific Optimizations
//!
//! #### GPU Acceleration
//! - Enable CUDA support for NVIDIA GPUs
//! - Use Metal acceleration on Apple Silicon
//! - Configure appropriate batch sizes for your GPU memory
//!
//! #### SIMD Optimizations
//! - `VoiRS` automatically detects and uses SIMD instructions (AVX2, NEON)
//! - Ensure your CPU supports these instruction sets for optimal performance
//! - No manual configuration required - optimizations are applied automatically
//!
//! #### Multi-threading
//! - Use `num_cpus::get()` to optimize thread pool sizes
//! - Enable parallel processing for batch operations
//! - Balance thread count with memory usage
//!

#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

// Re-export core VoiRS types
/// Item
pub use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme, VoirsError};

// Public API modules
pub mod analysis;
pub mod asr;
pub mod audio_formats;
pub mod audio_utilities;
pub mod error_enhancement;
pub mod error_recovery;
pub mod integration;
pub mod memory_optimization;
pub mod monitoring;
pub mod performance;
pub mod phoneme;
pub mod preprocessing;
pub mod training;
pub mod traits;
pub mod wake_word;

#[cfg(feature = "wasm")]
/// Pub
pub mod wasm;

// C API bindings (optional)
#[cfg(feature = "c-api")]
/// Pub
pub mod c_api;

// REST API bindings (optional)
#[cfg(feature = "rest-api")]
/// Pub
pub mod rest_api;

// Python bindings (optional)
#[cfg(feature = "python")]
pub mod python;

// Re-export Python module when the feature is enabled
#[cfg(feature = "python")]
/// Item
pub use python::*;

// Re-export key types from traits to avoid ambiguous glob re-exports
/// Item
pub use traits::{
    ASRConfig, ASRFeature, ASRMetadata, ASRModel, AudioAnalysis, AudioAnalysisConfig,
    AudioAnalyzer, AudioAnalyzerMetadata, AudioStream, PhonemeAlignment, PhonemeRecognitionConfig,
    PhonemeRecognizer, PhonemeRecognizerMetadata, RecognitionResult, Transcript, TranscriptChunk,
    TranscriptStream,
};

// Re-export specific implementations to avoid conflicts
/// Item
pub use analysis::AudioAnalyzerImpl;
/// Item
pub use asr::{ASRBackend, ASRBenchmarkingSuite, AccuracyValidator, IntelligentASRFallback};

// Advanced optimization exports
/// Item
pub use asr::advanced_optimization::{
    AdvancedOptimizationConfig, KnowledgeDistillationOptimizer, MixedPrecisionOptimizer,
    OptimizationObjective, OptimizationPlatform, ProgressivePruningOptimizer,
};
/// Item
pub use asr::optimization_integration::{
    ModelStats, OptimizationPipeline, OptimizationResults, OptimizationSummary,
};
/// Item
pub use audio_formats::{
    load_audio, load_audio_with_sample_rate, AudioFormat, AudioLoadConfig, UniversalAudioLoader,
};
/// Item
pub use audio_utilities::{
    analyze_audio_quality, extract_speech_segments, load_and_preprocess, optimize_for_recognition,
    split_audio_smart, AudioQualityReport, AudioUtilities,
};
/// Item
pub use performance::{
    PerformanceMetrics, PerformanceRequirements, PerformanceValidator, ValidationResult,
};
#[cfg(feature = "forced-align")]
/// Item
pub use phoneme::ForcedAlignModel;

#[cfg(feature = "mfa")]
/// Item
pub use phoneme::MFAModel;
/// Item
pub use preprocessing::{AudioPreprocessingConfig, AudioPreprocessor};
/// Item
pub use wake_word::{
    EnergyOptimizer, NeuralWakeWordModel, TemplateWakeWordModel, TrainingPhase, TrainingProgress,
    TrainingValidationReport, WakeWordConfig, WakeWordDetection, WakeWordDetector,
    WakeWordDetectorImpl, WakeWordModel, WakeWordStats, WakeWordTrainer, WakeWordTrainerImpl,
    WakeWordTrainingData,
};

// Convenience functions for quick start examples
/// Simple audio loading function for quick start examples
pub fn load_audio_simple(path: &str) -> Result<Vec<f32>, RecognitionError> {
    let audio_buffer = load_audio(path)?;
    Ok(audio_buffer.samples().to_vec())
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Convenient prelude for common imports
pub mod prelude {
    /// Prelude module for convenient imports
    pub use crate::traits::{
        ASRConfig, ASRMetadata, ASRModel, AudioAnalysis, AudioAnalysisConfig, AudioAnalyzer,
        AudioAnalyzerMetadata, AudioStream, PhonemeAlignment, PhonemeRecognitionConfig,
        PhonemeRecognizer, PhonemeRecognizerMetadata, RecognitionResult, Transcript,
        TranscriptChunk, TranscriptStream,
    };

    // Re-export from submodules when they're implemented
    // #[cfg(feature = "whisper")]
    // pub use crate::asr::WhisperModel;

    #[cfg(feature = "whisper-pure")]
    /// Item
    pub use crate::asr::PureRustWhisper;

    #[cfg(feature = "deepspeech")]
    /// Item
    pub use crate::asr::DeepSpeechModel;

    #[cfg(feature = "wav2vec2")]
    /// Item
    pub use crate::asr::Wav2Vec2Model;

    #[cfg(feature = "forced-align")]
    /// Item
    pub use crate::phoneme::ForcedAlignModel;

    #[cfg(feature = "mfa")]
    /// Item
    pub use crate::phoneme::MFAModel;

    /// Item
    pub use crate::analysis::AudioAnalyzerImpl;

    // Re-export ASR utilities
    /// Item
    pub use crate::asr::{
        ASRBackend, ASRBenchmarkingSuite, AccuracyValidator, IntelligentASRFallback,
    };

    // Re-export audio format utilities
    /// Item
    pub use crate::audio_formats::{
        load_audio, load_audio_with_sample_rate, AudioFormat, AudioLoadConfig, UniversalAudioLoader,
    };

    // Re-export audio utilities
    /// Item
    pub use crate::audio_utilities::{
        analyze_audio_quality, extract_speech_segments, load_and_preprocess,
        optimize_for_recognition, split_audio_smart, AudioQualityReport, AudioUtilities,
    };

    // Re-export performance utilities
    /// Item
    pub use crate::performance::{
        PerformanceMetrics, PerformanceRequirements, PerformanceValidator, ValidationResult,
    };

    // Re-export integration utilities
    /// Item
    pub use crate::integration::{
        ComponentInfo, IntegratedPerformanceMonitor, IntegrationConfig, PipelineProcessingConfig,
        UnifiedVoirsPipeline, VoirsIntegrationManager,
    };

    // Re-export wake word utilities
    /// Item
    pub use crate::wake_word::{
        EnergyOptimizer, NeuralWakeWordModel, TemplateWakeWordModel, TrainingPhase,
        TrainingProgress, TrainingValidationReport, WakeWordConfig, WakeWordDetection,
        WakeWordDetector, WakeWordDetectorImpl, WakeWordModel, WakeWordStats, WakeWordTrainer,
        WakeWordTrainerImpl, WakeWordTrainingData,
    };

    // Re-export SDK types
    /// Item
    pub use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme, VoirsError};

    // Re-export async trait
    /// Item
    pub use async_trait::async_trait;

    // Re-export error enhancement utilities
    /// Item
    pub use crate::error_enhancement::{
        enhance_recognition_error, get_quick_fixes, is_error_recoverable, ErrorEnhancer,
    };
}

// ============================================================================
// Error Types
// ============================================================================

/// Recognition-specific error types
#[derive(Debug, thiserror::Error)]
/// Recognition Error
pub enum RecognitionError {
    /// Model loading failed
    #[error("Failed to load model: {message}")]
    ModelLoadError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Model operation error
    #[error("Model error: {message}")]
    ModelError {
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

    /// Transcription error
    #[error("Transcription failed: {message}")]
    TranscriptionError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Phoneme recognition error
    #[error("Phoneme recognition failed: {message}")]
    PhonemeRecognitionError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Audio analysis error
    #[error("Audio analysis failed: {message}")]
    AudioAnalysisError {
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

    /// Feature not supported
    #[error("Feature not supported: {feature}")]
    FeatureNotSupported {
        /// Feature name
        feature: String,
    },

    /// Invalid input
    #[error("Invalid input: {message}")]
    InvalidInput {
        /// Error message
        message: String,
    },

    /// Resource error
    #[error("Resource error: {message}")]
    ResourceError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Unsupported audio format
    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    /// Invalid audio format
    #[error("Invalid audio format: {0}")]
    InvalidFormat(String),

    /// Model not found with suggestions
    #[error("Model '{model}' not found. Available models: {available:?}")]
    ModelNotFound {
        /// Model name
        model: String,
        /// Available models
        available: Vec<String>,
        /// Suggested alternatives
        suggestions: Vec<String>,
    },

    /// Language not supported with suggestions
    #[error("Language '{language}' not supported. Supported languages: {supported:?}")]
    LanguageNotSupported {
        /// Requested language
        language: String,
        /// Supported languages
        supported: Vec<String>,
        /// Suggested alternatives
        suggestions: Vec<String>,
    },

    /// GPU/Device not available with fallback info
    #[error("Device '{device}' not available: {reason}. Fallback: {fallback}")]
    DeviceNotAvailable {
        /// Requested device
        device: String,
        /// Reason for unavailability
        reason: String,
        /// Fallback device
        fallback: String,
    },

    /// Insufficient memory with recommendation
    #[error("Insufficient memory: need {required_mb}MB, have {available_mb}MB. Recommendation: {recommendation}")]
    InsufficientMemory {
        /// Required memory in MB
        required_mb: u64,
        /// Available memory in MB
        available_mb: u64,
        /// Recommendation to resolve
        recommendation: String,
    },

    /// Recognition timeout with retry suggestion
    #[error("Recognition timed out after {timeout_ms}ms. Audio duration: {audio_duration_ms}ms. Suggestion: {suggestion}")]
    RecognitionTimeout {
        /// Timeout duration in milliseconds
        timeout_ms: u64,
        /// Audio duration in milliseconds
        audio_duration_ms: u64,
        /// Suggestion for resolution
        suggestion: String,
    },

    /// Memory error
    #[error("Memory error: {message}")]
    MemoryError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Training error
    #[error("Training error: {message}")]
    TrainingError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

impl From<RecognitionError> for VoirsError {
    fn from(err: RecognitionError) -> Self {
        match err {
            RecognitionError::ModelLoadError { message, source } => VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::ASR,
                message,
                source,
            },
            RecognitionError::ModelError { message, source } => VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::ASR,
                message,
                source,
            },
            RecognitionError::AudioProcessingError { message, source: _ } => {
                VoirsError::AudioError {
                    message,
                    buffer_info: None,
                }
            }
            RecognitionError::TranscriptionError { message, source } => VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::ASR,
                message,
                source,
            },
            RecognitionError::PhonemeRecognitionError { message, source } => {
                VoirsError::ModelError {
                    model_type: voirs_sdk::error::ModelType::ASR,
                    message,
                    source,
                }
            }
            RecognitionError::AudioAnalysisError { message, source: _ } => VoirsError::AudioError {
                message,
                buffer_info: None,
            },
            RecognitionError::ConfigurationError { message } => VoirsError::ConfigError {
                field: "ASR".to_string(),
                message,
            },
            RecognitionError::FeatureNotSupported { feature } => VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::ASR,
                message: format!("Feature not supported: {feature}"),
                source: None,
            },
            RecognitionError::InvalidInput { message } => VoirsError::ConfigError {
                field: "Input".to_string(),
                message: format!("Invalid input: {message}"),
            },
            RecognitionError::ResourceError { message, source } => VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::ASR,
                message: format!("Resource error: {message}"),
                source,
            },
            RecognitionError::UnsupportedFormat(format) => VoirsError::AudioError {
                message: format!("Unsupported audio format: {format}"),
                buffer_info: None,
            },
            RecognitionError::InvalidFormat(format) => VoirsError::AudioError {
                message: format!("Invalid audio format: {format}"),
                buffer_info: None,
            },
            RecognitionError::ModelNotFound {
                model,
                available,
                suggestions,
            } => VoirsError::VoiceNotFound {
                voice: model,
                available,
                suggestions,
            },
            RecognitionError::LanguageNotSupported {
                language,
                supported,
                suggestions: _,
            } => VoirsError::LanguageNotSupported {
                language,
                supported,
            },
            RecognitionError::DeviceNotAvailable {
                device,
                reason,
                fallback,
            } => VoirsError::DeviceError {
                device,
                message: format!("Device not available: {reason}"),
                recovery_hint: Some(format!("Use fallback device: {fallback}")),
            },
            RecognitionError::InsufficientMemory {
                required_mb,
                available_mb,
                recommendation: _,
            } => VoirsError::GpuOutOfMemory {
                device: "ASR".to_string(),
                used_mb: required_mb as u32,
                available_mb: available_mb as u32,
            },
            RecognitionError::RecognitionTimeout {
                timeout_ms,
                audio_duration_ms,
                suggestion,
            } => VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::ASR,
                message: format!(
                    "Recognition timed out after {}ms. Audio duration: {}ms. Suggestion: {}",
                    timeout_ms, audio_duration_ms, suggestion
                ),
                source: None,
            },
            RecognitionError::MemoryError { message, source } => VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::ASR,
                message: format!("Memory error: {message}"),
                source,
            },
            RecognitionError::TrainingError { message, source } => VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::ASR,
                message: format!("Training error: {message}"),
                source,
            },
        }
    }
}

impl From<VoirsError> for RecognitionError {
    fn from(err: VoirsError) -> Self {
        match err {
            VoirsError::ModelError {
                model_type: _,
                message,
                source,
            } => RecognitionError::ModelError { message, source },
            VoirsError::AudioError {
                message,
                buffer_info: _,
            } => RecognitionError::AudioProcessingError {
                message,
                source: None,
            },
            VoirsError::ConfigError { field: _, message } => {
                RecognitionError::ConfigurationError { message }
            }
            VoirsError::SerializationError { format: _, message } => {
                RecognitionError::InvalidInput { message }
            }
            VoirsError::NetworkError {
                message,
                source,
                retry_count: _,
                max_retries: _,
            } => RecognitionError::ResourceError { message, source },
            VoirsError::IoError {
                path: _,
                operation: _,
                source,
            } => RecognitionError::ResourceError {
                message: format!("I/O error: {source}"),
                source: Some(Box::new(source)),
            },
            // Catch-all for other VoirsError variants
            _ => RecognitionError::ModelError {
                message: format!("VoiRS error: {err}"),
                source: Some(Box::new(err)),
            },
        }
    }
}

impl From<candle_core::Error> for RecognitionError {
    fn from(err: candle_core::Error) -> Self {
        RecognitionError::ModelError {
            message: format!("Candle error: {err}"),
            source: Some(Box::new(err)),
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Check if a model file exists and is valid
pub fn validate_model_file(path: &std::path::Path) -> RecognitionResult<()> {
    if !path.exists() {
        return Err(RecognitionError::ModelLoadError {
            message: format!("Model file not found: {}", path.display()),
            source: None,
        }
        .into());
    }

    if !path.is_file() {
        return Err(RecognitionError::ModelLoadError {
            message: format!("Path is not a file: {}", path.display()),
            source: None,
        }
        .into());
    }

    Ok(())
}

/// Create a default ASR configuration for a given language
#[must_use]
/// default asr config
pub fn default_asr_config(language: LanguageCode) -> ASRConfig {
    ASRConfig {
        language: Some(language),
        ..Default::default()
    }
}

/// Create a default phoneme recognition configuration for a given language
#[must_use]
/// default phoneme config
pub fn default_phoneme_config(language: LanguageCode) -> PhonemeRecognitionConfig {
    PhonemeRecognitionConfig {
        language,
        ..Default::default()
    }
}

/// Create a default audio analysis configuration
#[must_use]
/// default analysis config
pub fn default_analysis_config() -> AudioAnalysisConfig {
    AudioAnalysisConfig::default()
}

/// Convert confidence score to human-readable label
#[must_use]
/// confidence to label
pub fn confidence_to_label(confidence: f32) -> &'static str {
    match confidence {
        c if c >= 0.9 => "Very High",
        c if c >= 0.7 => "High",
        c if c >= 0.5 => "Medium",
        c if c >= 0.3 => "Low",
        _ => "Very Low",
    }
}

/// Utility function to merge transcripts
#[must_use]
/// merge transcripts
pub fn merge_transcripts(transcripts: &[Transcript]) -> Transcript {
    if transcripts.is_empty() {
        return Transcript {
            text: String::new(),
            language: LanguageCode::EnUs,
            confidence: 0.0,
            word_timestamps: Vec::new(),
            sentence_boundaries: Vec::new(),
            processing_duration: None,
        };
    }

    let mut merged_text = String::new();
    let mut all_word_timestamps = Vec::new();
    let mut all_sentence_boundaries = Vec::new();
    let mut total_confidence = 0.0;
    let mut total_duration = std::time::Duration::ZERO;

    for transcript in transcripts {
        if !merged_text.is_empty() {
            merged_text.push(' ');
        }
        merged_text.push_str(&transcript.text);

        all_word_timestamps.extend(transcript.word_timestamps.clone());
        all_sentence_boundaries.extend(transcript.sentence_boundaries.clone());
        total_confidence += transcript.confidence;

        if let Some(duration) = transcript.processing_duration {
            total_duration += duration;
        }
    }

    Transcript {
        text: merged_text,
        language: transcripts[0].language,
        confidence: total_confidence / transcripts.len() as f32,
        word_timestamps: all_word_timestamps,
        sentence_boundaries: all_sentence_boundaries,
        processing_duration: Some(total_duration),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_version() {
        assert!(!VERSION.is_empty(), "VERSION should not be empty");
    }

    #[test]
    fn test_confidence_to_label() {
        assert_eq!(confidence_to_label(0.95), "Very High");
        assert_eq!(confidence_to_label(0.8), "High");
        assert_eq!(confidence_to_label(0.6), "Medium");
        assert_eq!(confidence_to_label(0.4), "Low");
        assert_eq!(confidence_to_label(0.2), "Very Low");
    }

    #[test]
    fn test_default_configs() {
        let asr_config = default_asr_config(LanguageCode::EnUs);
        assert_eq!(asr_config.language, Some(LanguageCode::EnUs));
        assert!(asr_config.word_timestamps);

        let phoneme_config = default_phoneme_config(LanguageCode::EnUs);
        assert_eq!(phoneme_config.language, LanguageCode::EnUs);
        assert!(phoneme_config.word_alignment);

        let analysis_config = default_analysis_config();
        assert!(analysis_config.quality_metrics);
        assert!(analysis_config.prosody_analysis);
    }

    #[test]
    fn test_merge_transcripts() {
        let transcript1 = Transcript {
            text: "Hello".to_string(),
            language: LanguageCode::EnUs,
            confidence: 0.9,
            word_timestamps: vec![],
            sentence_boundaries: vec![],
            processing_duration: Some(std::time::Duration::from_millis(100)),
        };

        let transcript2 = Transcript {
            text: "world".to_string(),
            language: LanguageCode::EnUs,
            confidence: 0.8,
            word_timestamps: vec![],
            sentence_boundaries: vec![],
            processing_duration: Some(std::time::Duration::from_millis(150)),
        };

        let merged = merge_transcripts(&[transcript1, transcript2]);
        assert_eq!(merged.text, "Hello world");
        assert!((merged.confidence - 0.85).abs() < f32::EPSILON);
        assert_eq!(
            merged.processing_duration,
            Some(std::time::Duration::from_millis(250))
        );
    }
}
