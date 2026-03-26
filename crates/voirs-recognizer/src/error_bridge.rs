//! # Error Bridge
//!
//! Standardized error handling patterns for cross-crate error conversion
//! and unified error reporting across the `VoiRS` ecosystem.

use crate::sdk_bridge::ErrorCode;
use crate::RecognitionError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tracing::{error, warn};

/// Error bridge for cross-crate error handling
#[derive(Debug, Error)]
pub enum ErrorBridgeError {
    /// Error conversion failed
    #[error("Error conversion failed: {0}")]
    ConversionFailed(String),

    /// Unknown error type
    #[error("Unknown error type: {0}")]
    UnknownErrorType(String),

    /// Error context missing
    #[error("Error context missing: {0}")]
    ContextMissing(String),
}

/// Trait for converting errors to standardized `VoiRS` error format
pub trait ToVoirsError {
    /// Convert to `VoiRS` error
    fn to_voirs_error(&self) -> RecognitionError;

    /// Convert to error code
    fn to_error_code(&self) -> ErrorCode;

    /// Get error context
    fn error_context(&self) -> ErrorContext {
        ErrorContext::default()
    }
}

/// Trait for errors that can be recovered from
pub trait RecoverableError {
    /// Check if error is recoverable
    fn is_recoverable(&self) -> bool;

    /// Suggest recovery action
    fn recovery_action(&self) -> Option<RecoveryAction>;

    /// Retry delay in milliseconds
    fn retry_delay_ms(&self) -> Option<u64> {
        None
    }
}

/// Error context for detailed error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Operation that caused the error
    pub operation: String,

    /// Component where error occurred
    pub component: String,

    /// Additional context data
    pub context_data: HashMap<String, String>,

    /// Timestamp when error occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Thread/task ID if available
    pub thread_id: Option<String>,

    /// Stack trace if available
    pub stack_trace: Option<String>,
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self {
            operation: "unknown".to_string(),
            component: "voirs-recognizer".to_string(),
            context_data: HashMap::new(),
            timestamp: chrono::Utc::now(),
            thread_id: None,
            stack_trace: None,
        }
    }
}

/// Recovery actions for recoverable errors
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryAction {
    /// Retry the operation
    Retry,

    /// Use fallback method
    UseFallback,

    /// Skip and continue
    SkipAndContinue,

    /// Reset and retry
    ResetAndRetry,

    /// Reduce quality/performance and retry
    ReduceQualityAndRetry,

    /// Clear cache and retry
    ClearCacheAndRetry,

    /// Manual intervention required
    ManualInterventionRequired,
}

/// Standardized error reporting
pub struct ErrorReporter {
    /// Error history
    error_history: parking_lot::RwLock<Vec<ErrorReport>>,

    /// Error count by type
    error_counts: parking_lot::RwLock<HashMap<String, usize>>,

    /// Maximum history size
    max_history_size: usize,

    /// Error callbacks
    callbacks: parking_lot::RwLock<Vec<ErrorCallback>>,
}

/// Error report for logging and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReport {
    /// Error code
    pub error_code: ErrorCode,

    /// Error message
    pub message: String,

    /// Error context
    pub context: ErrorContext,

    /// Recovery attempted
    pub recovery_attempted: Option<RecoveryAction>,

    /// Recovery successful
    pub recovery_successful: bool,

    /// Error severity
    pub severity: ErrorSeverity,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Debug level - for development
    Debug,

    /// Info level - informational
    Info,

    /// Warning level - might cause issues
    Warning,

    /// Error level - operation failed
    Error,

    /// Critical level - system unstable
    Critical,

    /// Fatal level - system must shut down
    Fatal,
}

/// Error callback function type
pub type ErrorCallback = Arc<dyn Fn(&ErrorReport) + Send + Sync>;

impl ErrorReporter {
    /// Create a new error reporter
    #[must_use]
    pub fn new() -> Self {
        Self {
            error_history: parking_lot::RwLock::new(Vec::new()),
            error_counts: parking_lot::RwLock::new(HashMap::new()),
            max_history_size: 1000,
            callbacks: parking_lot::RwLock::new(Vec::new()),
        }
    }

    /// Report an error
    pub fn report(&self, report: ErrorReport) {
        // Log based on severity
        match report.severity {
            ErrorSeverity::Debug => tracing::debug!("Error: {}", report.message),
            ErrorSeverity::Info => tracing::info!("Error: {}", report.message),
            ErrorSeverity::Warning => warn!("Error: {}", report.message),
            ErrorSeverity::Error => error!("Error: {}", report.message),
            ErrorSeverity::Critical => error!("CRITICAL Error: {}", report.message),
            ErrorSeverity::Fatal => error!("FATAL Error: {}", report.message),
        }

        // Update error counts
        {
            let mut counts = self.error_counts.write();
            let error_type = format!("{:?}", report.error_code);
            *counts.entry(error_type).or_insert(0) += 1;
        }

        // Call registered callbacks
        {
            let callbacks = self.callbacks.read();
            for callback in callbacks.iter() {
                callback(&report);
            }
        }

        // Add to history (with size limit)
        {
            let mut history = self.error_history.write();
            history.push(report);

            // Trim history if needed
            if history.len() > self.max_history_size {
                let excess = history.len() - self.max_history_size;
                history.drain(0..excess);
            }
        }
    }

    /// Register error callback
    pub fn register_callback(&self, callback: ErrorCallback) {
        self.callbacks.write().push(callback);
    }

    /// Get error history
    pub fn get_history(&self) -> Vec<ErrorReport> {
        self.error_history.read().clone()
    }

    /// Get error counts
    pub fn get_error_counts(&self) -> HashMap<String, usize> {
        self.error_counts.read().clone()
    }

    /// Clear error history
    pub fn clear_history(&self) {
        self.error_history.write().clear();
    }

    /// Get error rate (errors per minute)
    pub fn get_error_rate(&self, window_minutes: u32) -> f64 {
        let history = self.error_history.read();
        let now = chrono::Utc::now();
        let window_start = now - chrono::Duration::minutes(i64::from(window_minutes));

        let recent_errors = history
            .iter()
            .filter(|r| r.context.timestamp > window_start)
            .count();

        recent_errors as f64 / f64::from(window_minutes)
    }

    /// Get errors by severity
    pub fn get_errors_by_severity(&self, severity: ErrorSeverity) -> Vec<ErrorReport> {
        self.error_history
            .read()
            .iter()
            .filter(|r| r.severity == severity)
            .cloned()
            .collect()
    }

    /// Get most common errors
    pub fn get_most_common_errors(&self, limit: usize) -> Vec<(String, usize)> {
        let counts = self.error_counts.read();
        let mut counts_vec: Vec<_> = counts.iter().map(|(k, v)| (k.clone(), *v)).collect();
        counts_vec.sort_by_key(|b| std::cmp::Reverse(b.1));
        counts_vec.truncate(limit);
        counts_vec
    }
}

impl Default for ErrorReporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementation of `ToVoirsError` for `RecognitionError`
impl ToVoirsError for RecognitionError {
    fn to_voirs_error(&self) -> RecognitionError {
        // RecognitionError doesn't derive Clone, so we reconstruct based on type
        // In a real implementation, you would clone the error or use Arc
        match self {
            RecognitionError::ModelLoadError { message, .. } => RecognitionError::ModelLoadError {
                message: message.clone(),
                source: None,
            },
            RecognitionError::ModelError { message, .. } => RecognitionError::ModelError {
                message: message.clone(),
                source: None,
            },
            RecognitionError::AudioProcessingError { message, .. } => {
                RecognitionError::AudioProcessingError {
                    message: message.clone(),
                    source: None,
                }
            }
            RecognitionError::TranscriptionError { message, .. } => {
                RecognitionError::TranscriptionError {
                    message: message.clone(),
                    source: None,
                }
            }
            RecognitionError::PhonemeRecognitionError { message, .. } => {
                RecognitionError::PhonemeRecognitionError {
                    message: message.clone(),
                    source: None,
                }
            }
            RecognitionError::AudioAnalysisError { message, .. } => {
                RecognitionError::AudioAnalysisError {
                    message: message.clone(),
                    source: None,
                }
            }
            RecognitionError::ConfigurationError { message } => {
                RecognitionError::ConfigurationError {
                    message: message.clone(),
                }
            }
            RecognitionError::FeatureNotSupported { feature } => {
                RecognitionError::FeatureNotSupported {
                    feature: feature.clone(),
                }
            }
            RecognitionError::InvalidInput { message } => RecognitionError::InvalidInput {
                message: message.clone(),
            },
            RecognitionError::ResourceError { message, .. } => RecognitionError::ResourceError {
                message: message.clone(),
                source: None,
            },
            RecognitionError::UnsupportedFormat(msg) => {
                RecognitionError::UnsupportedFormat(msg.clone())
            }
            RecognitionError::InvalidFormat(msg) => RecognitionError::InvalidFormat(msg.clone()),
            RecognitionError::ModelNotFound {
                model,
                available,
                suggestions,
            } => RecognitionError::ModelNotFound {
                model: model.clone(),
                available: available.clone(),
                suggestions: suggestions.clone(),
            },
            RecognitionError::LanguageNotSupported {
                language,
                supported,
                suggestions,
            } => RecognitionError::LanguageNotSupported {
                language: language.clone(),
                supported: supported.clone(),
                suggestions: suggestions.clone(),
            },
            RecognitionError::DeviceNotAvailable {
                device,
                reason,
                fallback,
            } => RecognitionError::DeviceNotAvailable {
                device: device.clone(),
                reason: reason.clone(),
                fallback: fallback.clone(),
            },
            RecognitionError::InsufficientMemory {
                required_mb,
                available_mb,
                recommendation,
            } => RecognitionError::InsufficientMemory {
                required_mb: *required_mb,
                available_mb: *available_mb,
                recommendation: recommendation.clone(),
            },
            RecognitionError::RecognitionTimeout {
                timeout_ms,
                audio_duration_ms,
                suggestion,
            } => RecognitionError::RecognitionTimeout {
                timeout_ms: *timeout_ms,
                audio_duration_ms: *audio_duration_ms,
                suggestion: suggestion.clone(),
            },
            RecognitionError::MemoryError { message, .. } => RecognitionError::MemoryError {
                message: message.clone(),
                source: None,
            },
            RecognitionError::TrainingError { message, .. } => RecognitionError::TrainingError {
                message: message.clone(),
                source: None,
            },
            RecognitionError::SynchronizationError { message } => {
                RecognitionError::SynchronizationError {
                    message: message.clone(),
                }
            }
        }
    }

    fn to_error_code(&self) -> ErrorCode {
        match self {
            RecognitionError::ModelLoadError { .. } => ErrorCode::ModelLoadError,
            RecognitionError::ModelError { .. } => ErrorCode::InferenceError,
            RecognitionError::AudioProcessingError { .. } => ErrorCode::AudioProcessingError,
            RecognitionError::TranscriptionError { .. } => ErrorCode::InferenceError,
            RecognitionError::PhonemeRecognitionError { .. } => ErrorCode::InferenceError,
            RecognitionError::AudioAnalysisError { .. } => ErrorCode::AudioProcessingError,
            RecognitionError::ConfigurationError { .. } => ErrorCode::ConfigError,
            RecognitionError::FeatureNotSupported { .. } => ErrorCode::GenericError,
            RecognitionError::InvalidInput { .. } => ErrorCode::GenericError,
            RecognitionError::ResourceError { .. } => ErrorCode::ResourceExhausted,
            RecognitionError::UnsupportedFormat(_) => ErrorCode::GenericError,
            RecognitionError::InvalidFormat(_) => ErrorCode::GenericError,
            RecognitionError::ModelNotFound { .. } => ErrorCode::ModelLoadError,
            RecognitionError::LanguageNotSupported { .. } => ErrorCode::GenericError,
            RecognitionError::DeviceNotAvailable { .. } => ErrorCode::GpuError,
            RecognitionError::InsufficientMemory { .. } => ErrorCode::MemoryError,
            RecognitionError::RecognitionTimeout { .. } => ErrorCode::TimeoutError,
            RecognitionError::MemoryError { .. } => ErrorCode::MemoryError,
            RecognitionError::TrainingError { .. } => ErrorCode::InferenceError,
            RecognitionError::SynchronizationError { .. } => ErrorCode::GenericError,
        }
    }

    fn error_context(&self) -> ErrorContext {
        ErrorContext {
            operation: "recognition".to_string(),
            component: "voirs-recognizer".to_string(),
            context_data: {
                let mut data = HashMap::new();
                data.insert("error_variant".to_string(), self.to_string());
                data
            },
            timestamp: chrono::Utc::now(),
            thread_id: Some(format!("{:?}", std::thread::current().id())),
            stack_trace: None,
        }
    }
}

/// Implementation of `RecoverableError` for `RecognitionError`
impl RecoverableError for RecognitionError {
    fn is_recoverable(&self) -> bool {
        match self {
            RecognitionError::ModelLoadError { .. } => false, // Need manual intervention
            RecognitionError::ModelError { .. } => true,      // Can retry
            RecognitionError::AudioProcessingError { .. } => true, // Can retry or skip
            RecognitionError::TranscriptionError { .. } => true, // Can retry
            RecognitionError::PhonemeRecognitionError { .. } => true, // Can retry
            RecognitionError::AudioAnalysisError { .. } => true, // Can retry
            RecognitionError::ConfigurationError { .. } => false, // Need config fix
            RecognitionError::FeatureNotSupported { .. } => false, // Feature not available
            RecognitionError::InvalidInput { .. } => false,   // User error
            RecognitionError::ResourceError { .. } => true,   // Can retry
            RecognitionError::UnsupportedFormat(_) => false,  // Format not supported
            RecognitionError::InvalidFormat(_) => false,      // Invalid format
            RecognitionError::ModelNotFound { .. } => false,  // Model doesn't exist
            RecognitionError::LanguageNotSupported { .. } => false, // Language not supported
            RecognitionError::DeviceNotAvailable { .. } => true, // Can use fallback
            RecognitionError::InsufficientMemory { .. } => true, // Can reduce batch size
            RecognitionError::RecognitionTimeout { .. } => true, // Can retry with longer timeout
            RecognitionError::MemoryError { .. } => true,     // Can reduce batch size
            RecognitionError::TrainingError { .. } => true,   // Can retry
            RecognitionError::SynchronizationError { .. } => false, // Indicates critical internal error
        }
    }

    fn recovery_action(&self) -> Option<RecoveryAction> {
        match self {
            RecognitionError::ModelError { .. } => Some(RecoveryAction::Retry),
            RecognitionError::AudioProcessingError { .. } => Some(RecoveryAction::UseFallback),
            RecognitionError::TranscriptionError { .. } => Some(RecoveryAction::Retry),
            RecognitionError::PhonemeRecognitionError { .. } => Some(RecoveryAction::Retry),
            RecognitionError::AudioAnalysisError { .. } => Some(RecoveryAction::UseFallback),
            RecognitionError::ResourceError { .. } => Some(RecoveryAction::Retry),
            RecognitionError::DeviceNotAvailable { .. } => Some(RecoveryAction::UseFallback),
            RecognitionError::InsufficientMemory { .. } => {
                Some(RecoveryAction::ReduceQualityAndRetry)
            }
            RecognitionError::RecognitionTimeout { .. } => {
                Some(RecoveryAction::ReduceQualityAndRetry)
            }
            RecognitionError::MemoryError { .. } => Some(RecoveryAction::ReduceQualityAndRetry),
            RecognitionError::TrainingError { .. } => Some(RecoveryAction::Retry),
            _ => None,
        }
    }

    fn retry_delay_ms(&self) -> Option<u64> {
        match self {
            RecognitionError::ModelError { .. } => Some(1000), // 1 second
            RecognitionError::AudioProcessingError { .. } => Some(500), // 500 ms
            RecognitionError::TranscriptionError { .. } => Some(1000), // 1 second
            RecognitionError::PhonemeRecognitionError { .. } => Some(1000), // 1 second
            RecognitionError::AudioAnalysisError { .. } => Some(500), // 500 ms
            RecognitionError::ResourceError { .. } => Some(2000), // 2 seconds
            RecognitionError::DeviceNotAvailable { .. } => Some(500), // 500 ms
            RecognitionError::InsufficientMemory { .. } => Some(1000), // 1 second
            RecognitionError::RecognitionTimeout { .. } => Some(5000), // 5 seconds
            RecognitionError::MemoryError { .. } => Some(1000), // 1 second
            RecognitionError::TrainingError { .. } => Some(2000), // 2 seconds
            _ => None,
        }
    }
}

/// Global error reporter instance
static ERROR_REPORTER: once_cell::sync::Lazy<ErrorReporter> =
    once_cell::sync::Lazy::new(ErrorReporter::new);

/// Get global error reporter
#[must_use]
pub fn global_error_reporter() -> &'static ErrorReporter {
    &ERROR_REPORTER
}

/// Helper function to report an error
pub fn report_error(
    error: &RecognitionError,
    severity: ErrorSeverity,
    recovery_attempted: Option<RecoveryAction>,
    recovery_successful: bool,
) {
    let report = ErrorReport {
        error_code: error.to_error_code(),
        message: error.to_string(),
        context: error.error_context(),
        recovery_attempted,
        recovery_successful,
        severity,
    };

    global_error_reporter().report(report);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_creation() {
        let context = ErrorContext::default();
        assert_eq!(context.component, "voirs-recognizer");
        assert!(!context.operation.is_empty());
    }

    #[test]
    fn test_error_reporter_creation() {
        let reporter = ErrorReporter::new();
        assert_eq!(reporter.get_history().len(), 0);
        assert_eq!(reporter.get_error_counts().len(), 0);
    }

    #[test]
    fn test_error_reporting() {
        let reporter = ErrorReporter::new();

        let report = ErrorReport {
            error_code: ErrorCode::InferenceError,
            message: "Test error".to_string(),
            context: ErrorContext::default(),
            recovery_attempted: None,
            recovery_successful: false,
            severity: ErrorSeverity::Error,
        };

        reporter.report(report);

        let history = reporter.get_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].message, "Test error");
    }

    #[test]
    fn test_error_counts() {
        let reporter = ErrorReporter::new();

        for _ in 0..3 {
            let report = ErrorReport {
                error_code: ErrorCode::InferenceError,
                message: "Test error".to_string(),
                context: ErrorContext::default(),
                recovery_attempted: None,
                recovery_successful: false,
                severity: ErrorSeverity::Error,
            };
            reporter.report(report);
        }

        let counts = reporter.get_error_counts();
        assert_eq!(counts.get("InferenceError"), Some(&3));
    }

    #[test]
    fn test_error_rate_calculation() {
        let reporter = ErrorReporter::new();

        for _ in 0..5 {
            let report = ErrorReport {
                error_code: ErrorCode::InferenceError,
                message: "Test error".to_string(),
                context: ErrorContext::default(),
                recovery_attempted: None,
                recovery_successful: false,
                severity: ErrorSeverity::Error,
            };
            reporter.report(report);
        }

        let rate = reporter.get_error_rate(1);
        assert!(rate >= 5.0);
    }

    #[test]
    fn test_errors_by_severity() {
        let reporter = ErrorReporter::new();

        let report1 = ErrorReport {
            error_code: ErrorCode::InferenceError,
            message: "Error 1".to_string(),
            context: ErrorContext::default(),
            recovery_attempted: None,
            recovery_successful: false,
            severity: ErrorSeverity::Error,
        };

        let report2 = ErrorReport {
            error_code: ErrorCode::ConfigError,
            message: "Error 2".to_string(),
            context: ErrorContext::default(),
            recovery_attempted: None,
            recovery_successful: false,
            severity: ErrorSeverity::Critical,
        };

        reporter.report(report1);
        reporter.report(report2);

        let errors = reporter.get_errors_by_severity(ErrorSeverity::Error);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].message, "Error 1");
    }

    #[test]
    fn test_recognition_error_conversion() {
        let error = RecognitionError::ModelError {
            message: "Test".to_string(),
            source: None,
        };
        assert_eq!(error.to_error_code(), ErrorCode::InferenceError);
        assert!(error.is_recoverable());
        assert_eq!(error.recovery_action(), Some(RecoveryAction::Retry));
    }

    #[test]
    fn test_most_common_errors() {
        let reporter = ErrorReporter::new();

        // Report different types of errors with different frequencies
        for _ in 0..5 {
            reporter.report(ErrorReport {
                error_code: ErrorCode::InferenceError,
                message: "Inference error".to_string(),
                context: ErrorContext::default(),
                recovery_attempted: None,
                recovery_successful: false,
                severity: ErrorSeverity::Error,
            });
        }

        for _ in 0..3 {
            reporter.report(ErrorReport {
                error_code: ErrorCode::ConfigError,
                message: "Config error".to_string(),
                context: ErrorContext::default(),
                recovery_attempted: None,
                recovery_successful: false,
                severity: ErrorSeverity::Error,
            });
        }

        let most_common = reporter.get_most_common_errors(5);
        assert_eq!(most_common.len(), 2);
        assert_eq!(most_common[0].1, 5); // InferenceError is most common
        assert_eq!(most_common[1].1, 3); // ConfigError is second
    }

    #[test]
    fn test_error_callback() {
        let reporter = ErrorReporter::new();
        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let called_clone = called.clone();

        let callback: ErrorCallback = Arc::new(move |_report| {
            called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
        });

        reporter.register_callback(callback);

        reporter.report(ErrorReport {
            error_code: ErrorCode::InferenceError,
            message: "Test".to_string(),
            context: ErrorContext::default(),
            recovery_attempted: None,
            recovery_successful: false,
            severity: ErrorSeverity::Error,
        });

        assert!(called.load(std::sync::atomic::Ordering::SeqCst));
    }
}
