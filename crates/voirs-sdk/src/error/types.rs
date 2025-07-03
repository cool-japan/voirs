//! Core error types for VoiRS operations.
//!
//! This module provides a comprehensive error type system for the VoiRS SDK,
//! categorized by operation type and severity level.

use std::{collections::HashMap, path::PathBuf, time::Duration};
use thiserror::Error;
use serde::{Deserialize, Serialize};

/// Main error type for VoiRS operations with enhanced categorization
#[derive(Debug, Error)]
pub enum VoirsError {
    // === Voice Management Errors ===
    /// Voice not found error with suggestions
    #[error("Voice '{voice}' not found. Available voices: {available:?}")]
    VoiceNotFound {
        voice: String,
        available: Vec<String>,
        suggestions: Vec<String>,
    },

    /// Voice unavailable due to missing dependencies
    #[error("Voice '{voice}' is unavailable: {reason}")]
    VoiceUnavailable { voice: String, reason: String },

    /// Voice configuration invalid
    #[error("Voice '{voice}' has invalid configuration: {issue}")]
    VoiceConfigurationInvalid { voice: String, issue: String },

    // === Synthesis Errors ===
    /// Synthesis operation failed with context
    #[error("Synthesis failed for text: '{text}' (length: {text_length})")]
    SynthesisFailed {
        text: String,
        text_length: usize,
        stage: SynthesisStage,
        #[source]
        cause: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Synthesis quality degraded
    #[error("Synthesis quality degraded: {reason}")]
    SynthesisQualityDegraded { reason: String, metrics: QualityMetrics },

    /// Synthesis interrupted
    #[error("Synthesis was interrupted at stage '{stage}'")]
    SynthesisInterrupted { stage: SynthesisStage, reason: String },

    // === Device and Hardware Errors ===
    /// Device or hardware error with recovery hints
    #[error("Device error ({device}): {message}")]
    DeviceError {
        device: String,
        message: String,
        recovery_hint: Option<String>,
    },

    /// Device not available error with alternatives
    #[error("Device '{device}' is not available")]
    DeviceNotAvailable {
        device: String,
        alternatives: Vec<String>,
    },

    /// Unsupported device error
    #[error("Device '{device}' is not supported")]
    UnsupportedDevice { device: String },

    /// GPU out of memory
    #[error("GPU out of memory on device '{device}': {used_mb}MB used, {available_mb}MB available")]
    GpuOutOfMemory {
        device: String,
        used_mb: u32,
        available_mb: u32,
    },

    // === File and I/O Errors ===
    /// File I/O error with operation context
    #[error("I/O error during '{operation}' at path '{path}'")]
    IoError {
        path: PathBuf,
        operation: IoOperation,
        #[source]
        source: std::io::Error,
    },

    /// File format not supported
    #[error("Unsupported file format '{format}' for path '{path}'")]
    UnsupportedFileFormat { path: PathBuf, format: String },

    /// File corrupted or invalid
    #[error("File is corrupted or invalid: '{path}'")]
    FileCorrupted { path: PathBuf, reason: String },

    // === Configuration Errors ===
    /// Configuration error with field context
    #[error("Configuration error in field '{field}': {message}")]
    ConfigError { field: String, message: String },

    /// Invalid configuration error with validation details
    #[error("Invalid configuration for field '{field}': {reason}")]
    InvalidConfiguration {
        field: String,
        value: String,
        reason: String,
        valid_values: Option<Vec<String>>,
    },

    /// Configuration migration failed
    #[error("Configuration migration failed from version '{from}' to '{to}': {reason}")]
    ConfigMigrationFailed {
        from: String,
        to: String,
        reason: String,
    },

    // === Model Errors ===
    /// Model loading or processing error
    #[error("Model error for '{model_type}': {message}")]
    ModelError {
        model_type: ModelType,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Model not found
    #[error("Model '{model_name}' not found at path '{path}'")]
    ModelNotFound { model_name: String, path: PathBuf },

    /// Model version mismatch
    #[error("Model version mismatch: expected {expected}, found {found}")]
    ModelVersionMismatch { expected: String, found: String },

    /// Model incompatible with device
    #[error("Model '{model_name}' is incompatible with device '{device}'")]
    ModelDeviceIncompatible { model_name: String, device: String },

    // === Audio Processing Errors ===
    /// Audio processing error with buffer info
    #[error("Audio processing error: {message}")]
    AudioError {
        message: String,
        buffer_info: Option<AudioBufferInfo>,
    },

    /// Audio format conversion failed
    #[error("Audio format conversion failed from '{from}' to '{to}': {reason}")]
    AudioFormatConversionFailed {
        from: String,
        to: String,
        reason: String,
    },

    /// Audio buffer overflow/underflow
    #[error("Audio buffer {error_type}: {details}")]
    AudioBufferError {
        error_type: AudioBufferErrorType,
        details: String,
    },

    // === Language Processing Errors ===
    /// G2P (Grapheme-to-Phoneme) conversion error
    #[error("G2P conversion failed for text '{text}': {message}")]
    G2pError {
        text: String,
        message: String,
        language: Option<String>,
    },

    /// Text preprocessing error
    #[error("Text preprocessing failed: {message}")]
    TextPreprocessingError { message: String, text_sample: String },

    /// Language not supported
    #[error("Language '{language}' is not supported")]
    LanguageNotSupported {
        language: String,
        supported: Vec<String>,
    },

    // === Network and Remote Errors ===
    /// Network or download error with retry info
    #[error("Network error: {message}")]
    NetworkError {
        message: String,
        retry_count: u32,
        max_retries: u32,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Download failed
    #[error("Download failed for '{url}': {reason}")]
    DownloadFailed {
        url: String,
        reason: String,
        bytes_downloaded: u64,
        total_bytes: Option<u64>,
    },

    /// Authentication failed
    #[error("Authentication failed for service '{service}': {reason}")]
    AuthenticationFailed { service: String, reason: String },

    // === Pipeline and State Errors ===
    /// Pipeline not ready error
    #[error("Pipeline is not ready for operation")]
    PipelineNotReady,

    /// Invalid state transition error
    #[error("Invalid state transition from '{from}' to '{to}': {reason}")]
    InvalidStateTransition {
        from: String,
        to: String,
        reason: String,
    },

    /// Component synchronization failed
    #[error("Component '{component}' synchronization failed: {reason}")]
    ComponentSynchronizationFailed { component: String, reason: String },

    // === Timing and Performance Errors ===
    /// Timeout error with operation context
    #[error("Operation '{operation}' timed out after {duration:?}")]
    TimeoutError {
        operation: String,
        duration: Duration,
        expected_duration: Option<Duration>,
    },

    /// Performance degradation detected
    #[error("Performance degradation detected: {metric} = {value} (threshold: {threshold})")]
    PerformanceDegradation {
        metric: String,
        value: f64,
        threshold: f64,
    },

    /// Real-time constraint violation
    #[error("Real-time constraint violated: {constraint}")]
    RealTimeConstraintViolation { constraint: String },

    // === Memory and Resource Errors ===
    /// Out of memory error
    #[error("Out of memory: {message}")]
    OutOfMemory { message: String, requested_mb: u32 },

    /// Resource exhausted
    #[error("Resource '{resource}' exhausted: {details}")]
    ResourceExhausted { resource: String, details: String },

    /// Memory leak detected
    #[error("Memory leak detected in component '{component}': {leaked_mb}MB")]
    MemoryLeak { component: String, leaked_mb: u32 },

    // === Serialization and Data Errors ===
    /// Serialization error with format context
    #[error("Serialization error ({format}): {message}")]
    SerializationError { format: String, message: String },

    /// Data validation failed
    #[error("Data validation failed for '{data_type}': {reason}")]
    DataValidationFailed { data_type: String, reason: String },

    /// Schema mismatch
    #[error("Schema mismatch: expected version {expected}, found {found}")]
    SchemaMismatch { expected: String, found: String },

    // === Generic and Internal Errors ===
    /// Generic internal error with context
    #[error("Internal error in '{component}': {message}")]
    InternalError { component: String, message: String },

    /// Feature not implemented
    #[error("Feature '{feature}' is not implemented")]
    NotImplemented { feature: String },

    /// Deprecated functionality used
    #[error("Deprecated functionality '{feature}' used. Use '{replacement}' instead")]
    DeprecatedFunctionality { feature: String, replacement: String },
}

impl Clone for VoirsError {
    fn clone(&self) -> Self {
        match self {
            Self::VoiceNotFound { voice, available, suggestions } => Self::VoiceNotFound {
                voice: voice.clone(),
                available: available.clone(),
                suggestions: suggestions.clone(),
            },
            Self::VoiceUnavailable { voice, reason } => Self::VoiceUnavailable {
                voice: voice.clone(),
                reason: reason.clone(),
            },
            Self::VoiceConfigurationInvalid { voice, issue } => Self::VoiceConfigurationInvalid {
                voice: voice.clone(),
                issue: issue.clone(),
            },
            Self::SynthesisFailed { text, text_length, stage, cause: _ } => Self::SynthesisFailed {
                text: text.clone(),
                text_length: *text_length,
                stage: *stage,
                cause: format!("Original error (cannot clone): {}", "error").into(),
            },
            Self::SynthesisQualityDegraded { reason, metrics } => Self::SynthesisQualityDegraded {
                reason: reason.clone(),
                metrics: metrics.clone(),
            },
            Self::SynthesisInterrupted { stage, reason } => Self::SynthesisInterrupted {
                stage: *stage,
                reason: reason.clone(),
            },
            Self::DeviceError { device, message, recovery_hint } => Self::DeviceError {
                device: device.clone(),
                message: message.clone(),
                recovery_hint: recovery_hint.clone(),
            },
            Self::DeviceNotAvailable { device, alternatives } => Self::DeviceNotAvailable {
                device: device.clone(),
                alternatives: alternatives.clone(),
            },
            Self::UnsupportedDevice { device } => Self::UnsupportedDevice {
                device: device.clone(),
            },
            Self::GpuOutOfMemory { device, used_mb, available_mb } => Self::GpuOutOfMemory {
                device: device.clone(),
                used_mb: *used_mb,
                available_mb: *available_mb,
            },
            Self::IoError { path, operation, source: _ } => Self::IoError {
                path: path.clone(),
                operation: *operation,
                source: std::io::Error::new(std::io::ErrorKind::Other, "Cloned error"),
            },
            Self::UnsupportedFileFormat { path, format } => Self::UnsupportedFileFormat {
                path: path.clone(),
                format: format.clone(),
            },
            Self::FileCorrupted { path, reason } => Self::FileCorrupted {
                path: path.clone(),
                reason: reason.clone(),
            },
            Self::ConfigError { field, message } => Self::ConfigError {
                field: field.clone(),
                message: message.clone(),
            },
            Self::InvalidConfiguration { field, value, reason, valid_values } => Self::InvalidConfiguration {
                field: field.clone(),
                value: value.clone(),
                reason: reason.clone(),
                valid_values: valid_values.clone(),
            },
            Self::ConfigMigrationFailed { from, to, reason } => Self::ConfigMigrationFailed {
                from: from.clone(),
                to: to.clone(),
                reason: reason.clone(),
            },
            Self::ModelError { model_type, message, source: _ } => Self::ModelError {
                model_type: *model_type,
                message: message.clone(),
                source: None,
            },
            Self::ModelNotFound { model_name, path } => Self::ModelNotFound {
                model_name: model_name.clone(),
                path: path.clone(),
            },
            Self::ModelVersionMismatch { expected, found } => Self::ModelVersionMismatch {
                expected: expected.clone(),
                found: found.clone(),
            },
            Self::ModelDeviceIncompatible { model_name, device } => Self::ModelDeviceIncompatible {
                model_name: model_name.clone(),
                device: device.clone(),
            },
            Self::AudioError { message, buffer_info } => Self::AudioError {
                message: message.clone(),
                buffer_info: buffer_info.clone(),
            },
            Self::AudioFormatConversionFailed { from, to, reason } => Self::AudioFormatConversionFailed {
                from: from.clone(),
                to: to.clone(),
                reason: reason.clone(),
            },
            Self::AudioBufferError { error_type, details } => Self::AudioBufferError {
                error_type: *error_type,
                details: details.clone(),
            },
            Self::G2pError { text, message, language } => Self::G2pError {
                text: text.clone(),
                message: message.clone(),
                language: language.clone(),
            },
            Self::TextPreprocessingError { message, text_sample } => Self::TextPreprocessingError {
                message: message.clone(),
                text_sample: text_sample.clone(),
            },
            Self::LanguageNotSupported { language, supported } => Self::LanguageNotSupported {
                language: language.clone(),
                supported: supported.clone(),
            },
            Self::NetworkError { message, retry_count, max_retries, source: _ } => Self::NetworkError {
                message: message.clone(),
                retry_count: *retry_count,
                max_retries: *max_retries,
                source: None,
            },
            Self::DownloadFailed { url, reason, bytes_downloaded, total_bytes } => Self::DownloadFailed {
                url: url.clone(),
                reason: reason.clone(),
                bytes_downloaded: *bytes_downloaded,
                total_bytes: *total_bytes,
            },
            Self::AuthenticationFailed { service, reason } => Self::AuthenticationFailed {
                service: service.clone(),
                reason: reason.clone(),
            },
            Self::PipelineNotReady => Self::PipelineNotReady,
            Self::InvalidStateTransition { from, to, reason } => Self::InvalidStateTransition {
                from: from.clone(),
                to: to.clone(),
                reason: reason.clone(),
            },
            Self::ComponentSynchronizationFailed { component, reason } => Self::ComponentSynchronizationFailed {
                component: component.clone(),
                reason: reason.clone(),
            },
            Self::TimeoutError { operation, duration, expected_duration } => Self::TimeoutError {
                operation: operation.clone(),
                duration: *duration,
                expected_duration: *expected_duration,
            },
            Self::PerformanceDegradation { metric, value, threshold } => Self::PerformanceDegradation {
                metric: metric.clone(),
                value: *value,
                threshold: *threshold,
            },
            Self::RealTimeConstraintViolation { constraint } => Self::RealTimeConstraintViolation {
                constraint: constraint.clone(),
            },
            Self::OutOfMemory { message, requested_mb } => Self::OutOfMemory {
                message: message.clone(),
                requested_mb: *requested_mb,
            },
            Self::ResourceExhausted { resource, details } => Self::ResourceExhausted {
                resource: resource.clone(),
                details: details.clone(),
            },
            Self::MemoryLeak { component, leaked_mb } => Self::MemoryLeak {
                component: component.clone(),
                leaked_mb: *leaked_mb,
            },
            Self::SerializationError { format, message } => Self::SerializationError {
                format: format.clone(),
                message: message.clone(),
            },
            Self::DataValidationFailed { data_type, reason } => Self::DataValidationFailed {
                data_type: data_type.clone(),
                reason: reason.clone(),
            },
            Self::SchemaMismatch { expected, found } => Self::SchemaMismatch {
                expected: expected.clone(),
                found: found.clone(),
            },
            Self::InternalError { component, message } => Self::InternalError {
                component: component.clone(),
                message: message.clone(),
            },
            Self::NotImplemented { feature } => Self::NotImplemented {
                feature: feature.clone(),
            },
            Self::DeprecatedFunctionality { feature, replacement } => Self::DeprecatedFunctionality {
                feature: feature.clone(),
                replacement: replacement.clone(),
            },
        }
    }
}

/// Synthesis stage identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynthesisStage {
    TextPreprocessing,
    G2pConversion,
    AcousticModeling,
    Vocoding,
    PostProcessing,
    AudioFinalization,
}

impl std::fmt::Display for SynthesisStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TextPreprocessing => write!(f, "text preprocessing"),
            Self::G2pConversion => write!(f, "G2P conversion"),
            Self::AcousticModeling => write!(f, "acoustic modeling"),
            Self::Vocoding => write!(f, "vocoding"),
            Self::PostProcessing => write!(f, "post-processing"),
            Self::AudioFinalization => write!(f, "audio finalization"),
        }
    }
}

/// Model type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    G2p,
    Acoustic,
    Vocoder,
    Preprocessor,
    Postprocessor,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::G2p => write!(f, "G2P"),
            Self::Acoustic => write!(f, "acoustic"),
            Self::Vocoder => write!(f, "vocoder"),
            Self::Preprocessor => write!(f, "preprocessor"),
            Self::Postprocessor => write!(f, "postprocessor"),
        }
    }
}

/// I/O operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IoOperation {
    Read,
    Write,
    Create,
    Delete,
    Copy,
    Move,
    Metadata,
}

impl std::fmt::Display for IoOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "read"),
            Self::Write => write!(f, "write"),
            Self::Create => write!(f, "create"),
            Self::Delete => write!(f, "delete"),
            Self::Copy => write!(f, "copy"),
            Self::Move => write!(f, "move"),
            Self::Metadata => write!(f, "metadata"),
        }
    }
}

/// Audio buffer error type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioBufferErrorType {
    Overflow,
    Underflow,
    InvalidFormat,
    SizeExceeded,
    Corruption,
}

impl std::fmt::Display for AudioBufferErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overflow => write!(f, "overflow"),
            Self::Underflow => write!(f, "underflow"),
            Self::InvalidFormat => write!(f, "invalid format"),
            Self::SizeExceeded => write!(f, "size exceeded"),
            Self::Corruption => write!(f, "corruption"),
        }
    }
}

/// Quality metrics for synthesis operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Real-time factor (processing_time / audio_duration)
    pub real_time_factor: f32,
    /// Mean opinion score (1.0-5.0)
    pub mean_opinion_score: Option<f32>,
    /// Signal-to-noise ratio in dB
    pub signal_to_noise_ratio: Option<f32>,
    /// Perceptual quality score (0.0-1.0)
    pub perceptual_quality: Option<f32>,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            real_time_factor: 1.0,
            mean_opinion_score: None,
            signal_to_noise_ratio: None,
            perceptual_quality: None,
        }
    }
}

/// Audio buffer information for error context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioBufferInfo {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
    /// Buffer size in samples
    pub buffer_size: usize,
    /// Duration in seconds
    pub duration: f32,
    /// Audio format
    pub format: String,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Information (non-error conditions)
    Info,
    /// Warning (potentially problematic)
    Warning,
    /// Error (operation failed but recoverable)
    Error,
    /// Critical (system-wide failure)
    Critical,
    /// Fatal (unrecoverable error)
    Fatal,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
            Self::Fatal => write!(f, "FATAL"),
        }
    }
}

/// Error context with additional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Component where error occurred
    pub component: String,
    /// Operation being performed
    pub operation: String,
    /// Additional context data
    pub context: HashMap<String, String>,
    /// Timestamp when error occurred
    pub timestamp: std::time::SystemTime,
    /// Stack trace if available
    pub stack_trace: Option<String>,
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self {
            severity: ErrorSeverity::Error,
            component: "unknown".to_string(),
            operation: "unknown".to_string(),
            context: HashMap::new(),
            timestamp: std::time::SystemTime::now(),
            stack_trace: None,
        }
    }
}

impl VoirsError {
    /// Get the severity level of this error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            // Fatal errors
            Self::OutOfMemory { .. } | Self::GpuOutOfMemory { .. } => ErrorSeverity::Fatal,

            // Critical errors
            Self::ModelNotFound { .. }
            | Self::DeviceNotAvailable { .. }
            | Self::UnsupportedDevice { .. } => ErrorSeverity::Critical,

            // Regular errors
            Self::SynthesisFailed { .. }
            | Self::VoiceNotFound { .. }
            | Self::IoError { .. }
            | Self::ConfigError { .. }
            | Self::ModelError { .. }
            | Self::AudioError { .. }
            | Self::G2pError { .. }
            | Self::NetworkError { .. }
            | Self::TimeoutError { .. }
            | Self::InvalidStateTransition { .. } => ErrorSeverity::Error,

            // Warnings
            Self::SynthesisQualityDegraded { .. }
            | Self::PerformanceDegradation { .. }
            | Self::VoiceUnavailable { .. }
            | Self::DeprecatedFunctionality { .. } => ErrorSeverity::Warning,

            // Info
            Self::NotImplemented { .. } => ErrorSeverity::Info,

            // Default to Error for unspecified cases
            _ => ErrorSeverity::Error,
        }
    }

    /// Get the component associated with this error
    pub fn component(&self) -> &str {
        match self {
            Self::VoiceNotFound { .. }
            | Self::VoiceUnavailable { .. }
            | Self::VoiceConfigurationInvalid { .. } => "voice",

            Self::SynthesisFailed { .. }
            | Self::SynthesisQualityDegraded { .. }
            | Self::SynthesisInterrupted { .. } => "synthesis",

            Self::DeviceError { .. }
            | Self::DeviceNotAvailable { .. }
            | Self::UnsupportedDevice { .. }
            | Self::GpuOutOfMemory { .. } => "device",

            Self::IoError { .. }
            | Self::UnsupportedFileFormat { .. }
            | Self::FileCorrupted { .. } => "io",

            Self::ConfigError { .. }
            | Self::InvalidConfiguration { .. }
            | Self::ConfigMigrationFailed { .. } => "config",

            Self::ModelError { .. }
            | Self::ModelNotFound { .. }
            | Self::ModelVersionMismatch { .. }
            | Self::ModelDeviceIncompatible { .. } => "model",

            Self::AudioError { .. }
            | Self::AudioFormatConversionFailed { .. }
            | Self::AudioBufferError { .. } => "audio",

            Self::G2pError { .. }
            | Self::TextPreprocessingError { .. }
            | Self::LanguageNotSupported { .. } => "language",

            Self::NetworkError { .. }
            | Self::DownloadFailed { .. }
            | Self::AuthenticationFailed { .. } => "network",

            Self::PipelineNotReady
            | Self::InvalidStateTransition { .. }
            | Self::ComponentSynchronizationFailed { .. } => "pipeline",

            Self::TimeoutError { .. }
            | Self::PerformanceDegradation { .. }
            | Self::RealTimeConstraintViolation { .. } => "performance",

            Self::OutOfMemory { .. }
            | Self::ResourceExhausted { .. }
            | Self::MemoryLeak { .. } => "memory",

            Self::SerializationError { .. }
            | Self::DataValidationFailed { .. }
            | Self::SchemaMismatch { .. } => "data",

            Self::InternalError { component, .. } => component,
            Self::NotImplemented { .. } | Self::DeprecatedFunctionality { .. } => "system",
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Unrecoverable errors
            Self::OutOfMemory { .. }
            | Self::GpuOutOfMemory { .. }
            | Self::UnsupportedDevice { .. }
            | Self::ModelNotFound { .. }
            | Self::FileCorrupted { .. }
            | Self::LanguageNotSupported { .. }
            | Self::PipelineNotReady => false,

            // Potentially recoverable with retry
            Self::NetworkError { .. }
            | Self::DownloadFailed { .. }
            | Self::TimeoutError { .. }
            | Self::DeviceNotAvailable { .. }
            | Self::ComponentSynchronizationFailed { .. } => true,

            // Configuration errors are recoverable with fixes
            Self::ConfigError { .. }
            | Self::InvalidConfiguration { .. }
            | Self::VoiceConfigurationInvalid { .. } => true,

            // Most other errors are potentially recoverable
            _ => true,
        }
    }

    /// Get suggested recovery actions
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            Self::VoiceNotFound { suggestions, .. } => suggestions.clone(),

            Self::DeviceNotAvailable { alternatives, .. } => {
                let mut suggestions = vec!["Check device availability".to_string()];
                for alt in alternatives {
                    suggestions.push(format!("Try using device: {}", alt));
                }
                suggestions
            }

            Self::NetworkError { retry_count, max_retries, .. } => {
                if retry_count < max_retries {
                    vec!["Retry the operation".to_string(), "Check network connection".to_string()]
                } else {
                    vec!["Check network connection".to_string(), "Try again later".to_string()]
                }
            }

            Self::TimeoutError { .. } => vec![
                "Increase timeout duration".to_string(),
                "Check system performance".to_string(),
                "Retry with smaller batch size".to_string(),
            ],

            Self::OutOfMemory { .. } => vec![
                "Reduce batch size".to_string(),
                "Close other applications".to_string(),
                "Use CPU instead of GPU".to_string(),
            ],

            Self::ConfigError { .. } => vec![
                "Check configuration file syntax".to_string(),
                "Validate configuration values".to_string(),
                "Reset to default configuration".to_string(),
            ],

            Self::ModelError { .. } => vec![
                "Re-download the model".to_string(),
                "Check model compatibility".to_string(),
                "Verify model integrity".to_string(),
            ],

            _ => vec!["Retry the operation".to_string()],
        }
    }

    /// Convert error to context with additional metadata
    pub fn with_context(self, component: impl Into<String>, operation: impl Into<String>) -> ErrorWithContext {
        let mut context = ErrorContext {
            severity: self.severity(),
            component: component.into(),
            operation: operation.into(),
            context: HashMap::new(),
            timestamp: std::time::SystemTime::now(),
            stack_trace: if cfg!(debug_assertions) {
                Some(std::backtrace::Backtrace::force_capture().to_string())
            } else {
                None
            },
        };

        // Add error-specific context
        match &self {
            Self::VoiceNotFound { voice, .. } => {
                context.context.insert("voice".to_string(), voice.clone());
            }
            Self::SynthesisFailed { text_length, .. } => {
                context.context.insert("text_length".to_string(), text_length.to_string());
            }
            Self::DeviceError { device, .. } => {
                context.context.insert("device".to_string(), device.clone());
            }
            Self::IoError { path, operation, .. } => {
                context.context.insert("path".to_string(), path.display().to_string());
                context.context.insert("io_operation".to_string(), operation.to_string());
            }
            _ => {}
        }

        ErrorWithContext { error: self, context }
    }
}

/// Error with additional context information
#[derive(Debug)]
pub struct ErrorWithContext {
    pub error: VoirsError,
    pub context: ErrorContext,
}

impl std::fmt::Display for ErrorWithContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} in {}: {}",
            self.context.severity, self.context.component, self.context.operation, self.error
        )
    }
}

impl std::error::Error for ErrorWithContext {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Result type alias for VoiRS operations
pub type Result<T> = std::result::Result<T, VoirsError>;

/// Result type with context
pub type ContextResult<T> = std::result::Result<T, ErrorWithContext>;