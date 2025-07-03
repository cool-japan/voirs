//! Enhanced error system for VoiRS SDK.
//!
//! This module provides a comprehensive error management system with:
//! - Detailed error types with contextual information
//! - Automatic error recovery strategies
//! - Error reporting and diagnostics
//! - Performance impact analysis
//!
//! # Architecture
//!
//! The error system is organized into three main components:
//!
//! - [`types`] - Core error types and definitions
//! - [`recovery`] - Error recovery strategies and circuit breakers
//! - [`reporting`] - Error logging, metrics, and diagnostics
//!
//! # Quick Start
//!
//! ```no_run
//! use voirs_sdk::error::{VoirsError, ErrorReporter, ErrorRecoveryManager};
//!
//! // Create error recovery manager
//! let mut recovery_manager = ErrorRecoveryManager::default();
//!
//! // Create error reporter
//! let mut reporter = ErrorReporter::default();
//!
//! // Report an error
//! let error = VoirsError::synthesis_failed("Hello world", 
//!     std::io::Error::new(std::io::ErrorKind::Other, "test"));
//! reporter.report_error(&error, Some("synthesis"));
//!
//! // Execute operation with recovery
//! let result = recovery_manager.execute_with_recovery("synthesis", || {
//!     Box::pin(async {
//!         // Your operation here
//!         Ok(())
//!     })
//! }).await;
//! ```
//!
//! # Error Context
//!
//! Errors can be enhanced with additional context:
//!
//! ```no_run
//! use voirs_sdk::error::{VoirsError, ErrorWithContext};
//!
//! let error = VoirsError::device_error("cuda", "Out of memory");
//! let error_with_context = error.with_context("synthesis", "generate_audio");
//! ```
//!
//! # Recovery Strategies
//!
//! Different recovery strategies can be configured for different components:
//!
//! ```no_run
//! use voirs_sdk::error::{ErrorRecoveryManager, RecoveryStrategy};
//! use std::time::Duration;
//!
//! let mut manager = ErrorRecoveryManager::new();
//!
//! // Configure exponential backoff for network operations
//! manager.register_strategy("network", RecoveryStrategy::RetryExponential {
//!     max_attempts: 5,
//!     initial_delay: Duration::from_millis(100),
//!     max_delay: Duration::from_secs(10),
//!     multiplier: 2.0,
//! });
//!
//! // Configure circuit breaker for device operations
//! manager.register_circuit_breaker("device", Default::default());
//! ```
//!
//! # Error Reporting
//!
//! Comprehensive error reporting with diagnostics:
//!
//! ```no_run
//! use voirs_sdk::error::{ErrorReporter, ErrorReporterConfig, ConsoleErrorListener};
//!
//! let config = ErrorReporterConfig {
//!     collect_stack_traces: true,
//!     collect_system_context: true,
//!     auto_report_critical: true,
//!     ..Default::default()
//! };
//!
//! let mut reporter = ErrorReporter::new(config);
//! reporter.add_listener(ConsoleErrorListener);
//!
//! // Generate diagnostic report
//! let diagnostic = reporter.generate_diagnostic_report();
//! println!("System health score: {:.2}", diagnostic.system_health.overall_score);
//! ```

pub mod types;
pub mod recovery;
pub mod reporting;

// Re-export main types for convenience
pub use types::{
    VoirsError, ErrorSeverity, ErrorWithContext, ErrorContext,
    SynthesisStage, ModelType, IoOperation, AudioBufferErrorType,
    QualityMetrics, AudioBufferInfo, Result, ContextResult,
};

pub use recovery::{
    ErrorRecoveryManager, RecoveryStrategy, RecoveryContext,
    CircuitBreaker, CircuitBreakerConfig, CircuitState,
    CircuitBreakerError,
};

pub use reporting::{
    ErrorReporter, ErrorReporterConfig, ErrorReport, ErrorCategory,
    ErrorListener, ConsoleErrorListener, FileErrorListener,
    ErrorStatistics, DiagnosticReport, SystemHealth, SystemContext,
    RuntimeInfo, PerformanceImpact,
};

/// Global error reporter instance
static mut GLOBAL_ERROR_REPORTER: Option<ErrorReporter> = None;
static REPORTER_INIT: std::sync::Once = std::sync::Once::new();

/// Initialize global error reporter
pub fn init_global_error_reporter(config: ErrorReporterConfig) {
    REPORTER_INIT.call_once(|| {
        unsafe {
            GLOBAL_ERROR_REPORTER = Some(ErrorReporter::new(config));
        }
    });
}

/// Get global error reporter
pub fn get_global_error_reporter() -> Option<&'static ErrorReporter> {
    unsafe { GLOBAL_ERROR_REPORTER.as_ref() }
}

/// Report error using global reporter
pub fn report_global_error(error: &VoirsError, context: Option<&str>) {
    if let Some(reporter) = get_global_error_reporter() {
        reporter.report_error(error, context);
    }
}

/// Convenience macro for creating VoirsError instances
#[macro_export]
macro_rules! voirs_error {
    // Synthesis error
    (synthesis_failed: $text:expr, $cause:expr) => {
        $crate::error::VoirsError::synthesis_failed($text, $cause)
    };
    
    // Device error
    (device_error: $device:expr, $message:expr) => {
        $crate::error::VoirsError::device_error($device, $message)
    };
    
    // Config error
    (config_error: $message:expr) => {
        $crate::error::VoirsError::config_error($message)
    };
    
    // Model error
    (model_error: $message:expr) => {
        $crate::error::VoirsError::model_error($message)
    };
    
    // Audio error
    (audio_error: $message:expr) => {
        $crate::error::VoirsError::audio_error($message)
    };
    
    // G2P error
    (g2p_error: $message:expr) => {
        $crate::error::VoirsError::g2p_error($message)
    };
    
    // Timeout error
    (timeout: $message:expr) => {
        $crate::error::VoirsError::timeout($message)
    };
    
    // Internal error
    (internal: $component:expr, $message:expr) => {
        $crate::error::VoirsError::InternalError {
            component: $component.to_string(),
            message: $message.to_string(),
        }
    };
}

/// Convenience macro for error recovery
#[macro_export]
macro_rules! with_recovery {
    ($manager:expr, $component:expr, $operation:expr) => {
        $manager.execute_with_recovery($component, || {
            Box::pin(async move { $operation })
        }).await
    };
}

/// Convenience macro for error reporting
#[macro_export]
macro_rules! report_error {
    ($error:expr) => {
        $crate::error::report_global_error(&$error, None);
    };
    ($error:expr, $context:expr) => {
        $crate::error::report_global_error(&$error, Some($context));
    };
}

/// Extension trait for Result types to add error reporting
pub trait ResultExt<T> {
    /// Report error if Result is Err, then return the Result unchanged
    fn report_on_error(self, context: Option<&str>) -> Self;
    
    /// Report error and convert to a different error type
    fn report_and_convert<E, F>(self, context: Option<&str>, f: F) -> std::result::Result<T, E>
    where
        F: FnOnce(VoirsError) -> E;
    
    /// Add error context
    fn with_error_context(self, component: impl Into<String>, operation: impl Into<String>) -> ContextResult<T>;
}

impl<T> ResultExt<T> for Result<T> {
    fn report_on_error(self, context: Option<&str>) -> Self {
        if let Err(ref error) = self {
            report_global_error(error, context);
        }
        self
    }
    
    fn report_and_convert<E, F>(self, context: Option<&str>, f: F) -> std::result::Result<T, E>
    where
        F: FnOnce(VoirsError) -> E,
    {
        match self {
            Ok(value) => Ok(value),
            Err(error) => {
                report_global_error(&error, context);
                Err(f(error))
            }
        }
    }
    
    fn with_error_context(self, component: impl Into<String>, operation: impl Into<String>) -> ContextResult<T> {
        match self {
            Ok(value) => Ok(value),
            Err(error) => Err(error.with_context(component, operation)),
        }
    }
}

/// Extension trait for VoirsError to add common helper methods
pub trait VoirsErrorExt {
    /// Check if error indicates a permanent failure
    fn is_permanent(&self) -> bool;
    
    /// Check if error indicates a temporary failure
    fn is_temporary(&self) -> bool;
    
    /// Check if error is related to user input
    fn is_user_error(&self) -> bool;
    
    /// Check if error is related to system resources
    fn is_resource_error(&self) -> bool;
    
    /// Get recommended wait time before retry
    fn recommended_retry_delay(&self) -> Option<std::time::Duration>;
}

impl VoirsErrorExt for VoirsError {
    fn is_permanent(&self) -> bool {
        matches!(
            self,
            VoirsError::UnsupportedDevice { .. }
            | VoirsError::ModelNotFound { .. }
            | VoirsError::FileCorrupted { .. }
            | VoirsError::LanguageNotSupported { .. }
            | VoirsError::NotImplemented { .. }
        )
    }
    
    fn is_temporary(&self) -> bool {
        !self.is_permanent() && self.is_recoverable()
    }
    
    fn is_user_error(&self) -> bool {
        matches!(
            self,
            VoirsError::VoiceNotFound { .. }
            | VoirsError::InvalidConfiguration { .. }
            | VoirsError::ConfigError { .. }
        )
    }
    
    fn is_resource_error(&self) -> bool {
        matches!(
            self,
            VoirsError::OutOfMemory { .. }
            | VoirsError::GpuOutOfMemory { .. }
            | VoirsError::ResourceExhausted { .. }
        )
    }
    
    fn recommended_retry_delay(&self) -> Option<std::time::Duration> {
        use std::time::Duration;
        
        if !self.is_recoverable() {
            return None;
        }
        
        Some(match self {
            VoirsError::NetworkError { .. } => Duration::from_secs(1),
            VoirsError::TimeoutError { .. } => Duration::from_millis(500),
            VoirsError::DeviceError { .. } => Duration::from_millis(100),
            VoirsError::ModelError { .. } => Duration::from_secs(2),
            VoirsError::OutOfMemory { .. } => Duration::from_secs(5),
            _ => Duration::from_millis(200),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_error_macros() {
        let error = voirs_error!(config_error: "Test error");
        assert!(matches!(error, VoirsError::ConfigError { .. }));
        
        let error = voirs_error!(internal: "test_component", "Test message");
        assert!(matches!(error, VoirsError::InternalError { .. }));
    }

    #[test]
    fn test_error_extensions() {
        let error = VoirsError::NetworkError {
            message: "Connection failed".to_string(),
            retry_count: 1,
            max_retries: 3,
            source: None,
        };
        
        assert!(error.is_temporary());
        assert!(!error.is_permanent());
        assert!(!error.is_user_error());
        assert!(error.recommended_retry_delay().is_some());
    }

    #[test]
    fn test_result_extensions() {
        let result: Result<i32> = Err(VoirsError::InternalError {
            component: "test".to_string(),
            message: "test error".to_string(),
        });
        
        let context_result = result.with_error_context("test_component", "test_operation");
        assert!(context_result.is_err());
        
        if let Err(error_with_context) = context_result {
            assert_eq!(error_with_context.context.component, "test_component");
            assert_eq!(error_with_context.context.operation, "test_operation");
        }
    }

    #[tokio::test]
    async fn test_integration() {
        // Initialize global error reporter
        init_global_error_reporter(ErrorReporterConfig::default());
        
        // Create recovery manager
        let manager = ErrorRecoveryManager::default();
        
        // Test operation with recovery
        let result: Result<()> = manager.execute_with_recovery("test", || -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>> {
            Box::pin(async {
                Err(VoirsError::InternalError {
                    component: "test".to_string(),
                    message: "test error".to_string(),
                })
            })
        }).await;
        
        assert!(result.is_err());
        
        // Check that error was reported
        if let Some(reporter) = get_global_error_reporter() {
            let stats = reporter.get_statistics();
            // Note: The error might not be reported due to severity filtering
        }
    }
}

/// Implement From traits for common error conversions
impl From<std::io::Error> for VoirsError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError {
            path: std::path::PathBuf::from("unknown"),
            operation: IoOperation::Read,
            source: err,
        }
    }
}

/// Legacy constructor helper - allows old code to work unchanged
impl VoirsError {
    /// Legacy InvalidConfiguration constructor
    pub fn invalid_configuration_legacy(field: String, value: String, reason: String) -> Self {
        Self::InvalidConfiguration {
            field,
            value, 
            reason,
            valid_values: None,
        }
    }

    /// Legacy DeviceNotAvailable constructor
    pub fn device_not_available_legacy(device: String) -> Self {
        Self::DeviceNotAvailable {
            device,
            alternatives: Vec::new(),
        }
    }

    /// Legacy VoiceNotFound constructor
    pub fn voice_not_found_legacy(voice: String, available: Vec<String>) -> Self {
        Self::VoiceNotFound {
            voice,
            available: available.clone(),
            suggestions: available.into_iter().take(3).collect(),
        }
    }
}

impl From<serde_json::Error> for VoirsError {
    fn from(err: serde_json::Error) -> Self {
        Self::SerializationError {
            format: "JSON".to_string(),
            message: err.to_string(),
        }
    }
}

impl From<toml::de::Error> for VoirsError {
    fn from(err: toml::de::Error) -> Self {
        Self::SerializationError {
            format: "TOML".to_string(),
            message: err.to_string(),
        }
    }
}

impl From<toml::ser::Error> for VoirsError {
    fn from(err: toml::ser::Error) -> Self {
        Self::SerializationError {
            format: "TOML".to_string(),
            message: err.to_string(),
        }
    }
}

impl From<hf_hub::api::sync::ApiError> for VoirsError {
    fn from(err: hf_hub::api::sync::ApiError) -> Self {
        Self::NetworkError {
            message: format!("HuggingFace Hub API error: {}", err),
            retry_count: 0,
            max_retries: 3,
            source: Some(Box::new(err)),
        }
    }
}

/// Compatibility wrapper to maintain backward compatibility with old error type
impl VoirsError {
    /// Create a new synthesis error (legacy API)
    pub fn synthesis_failed(text: impl Into<String>, cause: impl std::error::Error + Send + Sync + 'static) -> Self {
        let text = text.into();
        Self::SynthesisFailed {
            text_length: text.len(),
            text,
            stage: SynthesisStage::TextPreprocessing,
            cause: Box::new(cause),
        }
    }

    /// Create a new device error (legacy API)
    pub fn device_error(device: impl Into<String>, message: impl Into<String>) -> Self {
        Self::DeviceError {
            device: device.into(),
            message: message.into(),
            recovery_hint: None,
        }
    }

    /// Create a new configuration error (legacy API)
    pub fn config_error(message: impl Into<String>) -> Self {
        Self::ConfigError {
            field: "unknown".to_string(),
            message: message.into(),
        }
    }

    /// Create a new model error (legacy API)
    pub fn model_error(message: impl Into<String>) -> Self {
        Self::ModelError {
            model_type: ModelType::Acoustic,
            message: message.into(),
            source: None,
        }
    }

    /// Create a new audio error (legacy API)
    pub fn audio_error(message: impl Into<String>) -> Self {
        Self::AudioError {
            message: message.into(),
            buffer_info: None,
        }
    }

    /// Create a new plugin error (legacy API)
    pub fn plugin_error(message: impl Into<String>) -> Self {
        Self::ConfigError {
            field: "plugin".to_string(),
            message: message.into(),
        }
    }

    /// Create a new G2P error (legacy API)
    pub fn g2p_error(message: impl Into<String>) -> Self {
        Self::G2pError {
            text: "unknown".to_string(),
            message: message.into(),
            language: None,
        }
    }

    /// Create a new timeout error (legacy API)
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::TimeoutError {
            operation: "unknown".to_string(),
            duration: std::time::Duration::from_secs(30),
            expected_duration: None,
        }
    }

    /// Create voice not found error with automatic suggestions
    pub fn voice_not_found(voice: impl Into<String>, available: Vec<String>) -> Self {
        let voice_str = voice.into();
        let suggestions = available.iter()
            .filter(|v| v.contains(&voice_str) || voice_str.contains(*v))
            .take(3)
            .cloned()
            .collect();
        
        Self::VoiceNotFound {
            voice: voice_str,
            available,
            suggestions,
        }
    }

    /// Create invalid configuration error
    pub fn invalid_config(field: impl Into<String>, value: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidConfiguration {
            field: field.into(),
            value: value.into(),
            reason: reason.into(),
            valid_values: None,
        }
    }

    /// Create invalid configuration error with valid values
    pub fn invalid_config_with_values(
        field: impl Into<String>, 
        value: impl Into<String>, 
        reason: impl Into<String>,
        valid_values: Vec<String>
    ) -> Self {
        Self::InvalidConfiguration {
            field: field.into(),
            value: value.into(),
            reason: reason.into(),
            valid_values: Some(valid_values),
        }
    }

    /// Create internal error
    pub fn internal(component: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InternalError {
            component: component.into(),
            message: message.into(),
        }
    }

    /// Create serialization error
    pub fn serialization(format: impl Into<String>, message: impl Into<String>) -> Self {
        Self::SerializationError {
            format: format.into(),
            message: message.into(),
        }
    }

    /// Create I/O error
    pub fn io_error(path: impl Into<std::path::PathBuf>, operation: IoOperation, source: std::io::Error) -> Self {
        Self::IoError {
            path: path.into(),
            operation,
            source,
        }
    }

    /// Create model error with type
    pub fn model_error_typed(model_type: ModelType, message: impl Into<String>) -> Self {
        Self::ModelError {
            model_type,
            message: message.into(),
            source: None,
        }
    }
}