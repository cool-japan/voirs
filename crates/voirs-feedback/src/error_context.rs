//! Enhanced Error Context System
//!
//! This module provides comprehensive error context tracking and debugging information
//! to improve troubleshooting and error resolution in production environments.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Error severity levels for better prioritization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Low priority errors that don't affect functionality
    Low,
    /// Medium priority errors that may impact performance
    Medium,
    /// High priority errors that affect core functionality
    High,
    /// Critical errors that require immediate attention
    Critical,
}

/// Error category for better classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Network-related errors
    Network,
    /// Database and persistence errors
    Database,
    /// Audio processing errors
    AudioProcessing,
    /// Authentication and authorization errors
    Authentication,
    /// Configuration and setup errors
    Configuration,
    /// External service integration errors
    ExternalService,
    /// Resource allocation and memory errors
    Resource,
    /// User input validation errors
    Validation,
    /// Unknown or uncategorized errors
    Unknown,
}

/// Comprehensive error context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Unique error identifier for tracking
    pub error_id: String,
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Error category
    pub category: ErrorCategory,
    /// Primary error message
    pub message: String,
    /// Detailed error description
    pub details: Option<String>,
    /// Error source/origin component
    pub source: String,
    /// Function or method where error occurred
    pub function: Option<String>,
    /// User ID associated with the error (if applicable)
    pub user_id: Option<String>,
    /// Session ID associated with the error (if applicable)
    pub session_id: Option<String>,
    /// Additional metadata for debugging
    pub metadata: HashMap<String, String>,
    /// Stack trace or call chain
    pub stack_trace: Option<String>,
    /// Suggested resolution steps
    pub resolution_steps: Vec<String>,
    /// Related error IDs for correlation
    pub related_errors: Vec<String>,
    /// Timestamp when error occurred
    pub timestamp: u64,
    /// Time spent processing before error
    pub processing_time: Option<Duration>,
    /// Error retry count (if applicable)
    pub retry_count: u32,
    /// Whether error is recoverable
    pub recoverable: bool,
}

impl ErrorContext {
    /// Create a new error context with minimal information
    pub fn new<S: Into<String>>(
        severity: ErrorSeverity,
        category: ErrorCategory,
        message: S,
        source: S,
    ) -> Self {
        Self {
            error_id: uuid::Uuid::new_v4().to_string(),
            severity,
            category,
            message: message.into(),
            details: None,
            source: source.into(),
            function: None,
            user_id: None,
            session_id: None,
            metadata: HashMap::new(),
            stack_trace: None,
            resolution_steps: Vec::new(),
            related_errors: Vec::new(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            processing_time: None,
            retry_count: 0,
            recoverable: false,
        }
    }

    /// Add detailed description
    pub fn with_details<S: Into<String>>(mut self, details: S) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Add function name where error occurred
    pub fn with_function<S: Into<String>>(mut self, function: S) -> Self {
        self.function = Some(function.into());
        self
    }

    /// Add user context
    pub fn with_user<S: Into<String>>(mut self, user_id: S) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Add session context
    pub fn with_session<S: Into<String>>(mut self, session_id: S) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Add metadata key-value pair
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add multiple metadata entries
    pub fn with_metadata_map(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata.extend(metadata);
        self
    }

    /// Add stack trace information
    pub fn with_stack_trace<S: Into<String>>(mut self, stack_trace: S) -> Self {
        self.stack_trace = Some(stack_trace.into());
        self
    }

    /// Add resolution steps
    pub fn with_resolution_steps(mut self, steps: Vec<String>) -> Self {
        self.resolution_steps = steps;
        self
    }

    /// Add related error ID for correlation
    pub fn with_related_error<S: Into<String>>(mut self, error_id: S) -> Self {
        self.related_errors.push(error_id.into());
        self
    }

    /// Set processing time before error occurred
    pub fn with_processing_time(mut self, duration: Duration) -> Self {
        self.processing_time = Some(duration);
        self
    }

    /// Set retry count
    pub fn with_retry_count(mut self, count: u32) -> Self {
        self.retry_count = count;
        self
    }

    /// Mark error as recoverable
    pub fn as_recoverable(mut self) -> Self {
        self.recoverable = true;
        self
    }

    /// Convert to JSON string for logging
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Create a summary string for quick logging
    pub fn summary(&self) -> String {
        format!(
            "[{}] {} - {} in {} (ID: {})",
            match self.severity {
                ErrorSeverity::Low => "LOW",
                ErrorSeverity::Medium => "MED",
                ErrorSeverity::High => "HIGH",
                ErrorSeverity::Critical => "CRIT",
            },
            match self.category {
                ErrorCategory::Network => "NET",
                ErrorCategory::Database => "DB",
                ErrorCategory::AudioProcessing => "AUDIO",
                ErrorCategory::Authentication => "AUTH",
                ErrorCategory::Configuration => "CONFIG",
                ErrorCategory::ExternalService => "EXT",
                ErrorCategory::Resource => "RES",
                ErrorCategory::Validation => "VAL",
                ErrorCategory::Unknown => "UNK",
            },
            self.message,
            self.source,
            &self.error_id[..8]
        )
    }

    /// Check if error is user-facing
    pub fn is_user_facing(&self) -> bool {
        matches!(
            self.category,
            ErrorCategory::Validation | ErrorCategory::Authentication
        ) || self.user_id.is_some()
    }

    /// Check if error requires immediate attention
    pub fn requires_immediate_attention(&self) -> bool {
        matches!(self.severity, ErrorSeverity::Critical)
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())?;
        if let Some(details) = &self.details {
            write!(f, "\nDetails: {}", details)?;
        }
        if !self.resolution_steps.is_empty() {
            write!(f, "\nResolution steps:")?;
            for (i, step) in self.resolution_steps.iter().enumerate() {
                write!(f, "\n  {}. {}", i + 1, step)?;
            }
        }
        Ok(())
    }
}

/// Error context builder for fluent API
pub struct ErrorContextBuilder {
    context: ErrorContext,
}

impl ErrorContextBuilder {
    /// Start building a new error context
    pub fn new<S: Into<String>>(
        severity: ErrorSeverity,
        category: ErrorCategory,
        message: S,
        source: S,
    ) -> Self {
        Self {
            context: ErrorContext::new(severity, category, message, source),
        }
    }

    /// Add details
    pub fn details<S: Into<String>>(mut self, details: S) -> Self {
        self.context = self.context.with_details(details);
        self
    }

    /// Add function name
    pub fn function<S: Into<String>>(mut self, function: S) -> Self {
        self.context = self.context.with_function(function);
        self
    }

    /// Add user context
    pub fn user<S: Into<String>>(mut self, user_id: S) -> Self {
        self.context = self.context.with_user(user_id);
        self
    }

    /// Add session context
    pub fn session<S: Into<String>>(mut self, session_id: S) -> Self {
        self.context = self.context.with_session(session_id);
        self
    }

    /// Add metadata
    pub fn metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.context = self.context.with_metadata(key, value);
        self
    }

    /// Mark as recoverable
    pub fn recoverable(mut self) -> Self {
        self.context = self.context.as_recoverable();
        self
    }

    /// Build the final error context
    pub fn build(self) -> ErrorContext {
        self.context
    }
}

/// Macro for creating error contexts with source location
#[macro_export]
macro_rules! error_context {
    ($severity:expr, $category:expr, $message:expr) => {
        $crate::error_context::ErrorContext::new(
            $severity,
            $category,
            $message,
            format!("{}:{}", file!(), line!()),
        )
        .with_function(format!("{}::{}", module_path!(), function_name!()))
    };
}

/// Common error context patterns for frequent use cases
impl ErrorContext {
    /// Create database connection error
    pub fn database_connection_error<S: Into<String>>(message: S, source: S) -> Self {
        Self::new(
            ErrorSeverity::Critical,
            ErrorCategory::Database,
            message,
            source,
        )
        .with_resolution_steps(vec![
            "Check database connection string".to_string(),
            "Verify database server is running".to_string(),
            "Check network connectivity".to_string(),
            "Validate database credentials".to_string(),
        ])
    }

    /// Create audio processing error
    pub fn audio_processing_error<S: Into<String>>(message: S, source: S) -> Self {
        Self::new(
            ErrorSeverity::High,
            ErrorCategory::AudioProcessing,
            message,
            source,
        )
        .with_resolution_steps(vec![
            "Check audio format compatibility".to_string(),
            "Verify audio buffer size".to_string(),
            "Validate sample rate settings".to_string(),
        ])
    }

    /// Create validation error
    pub fn validation_error<S: Into<String>>(message: S, source: S) -> Self {
        Self::new(
            ErrorSeverity::Medium,
            ErrorCategory::Validation,
            message,
            source,
        )
        .with_resolution_steps(vec![
            "Check input data format".to_string(),
            "Validate required fields".to_string(),
            "Review validation rules".to_string(),
        ])
        .as_recoverable()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_creation() {
        let context = ErrorContext::new(
            ErrorSeverity::High,
            ErrorCategory::Database,
            "Connection failed",
            "postgres_manager",
        );

        assert_eq!(context.severity, ErrorSeverity::High);
        assert_eq!(context.category, ErrorCategory::Database);
        assert_eq!(context.message, "Connection failed");
        assert_eq!(context.source, "postgres_manager");
        assert!(!context.error_id.is_empty());
    }

    #[test]
    fn test_error_context_builder() {
        let context = ErrorContextBuilder::new(
            ErrorSeverity::Medium,
            ErrorCategory::AudioProcessing,
            "Processing failed",
            "audio_processor",
        )
        .details("Invalid sample rate")
        .function("process_audio")
        .user("user123")
        .session("session456")
        .metadata("sample_rate", "44100")
        .recoverable()
        .build();

        assert_eq!(context.severity, ErrorSeverity::Medium);
        assert_eq!(context.details, Some("Invalid sample rate".to_string()));
        assert_eq!(context.function, Some("process_audio".to_string()));
        assert_eq!(context.user_id, Some("user123".to_string()));
        assert_eq!(context.session_id, Some("session456".to_string()));
        assert_eq!(
            context.metadata.get("sample_rate"),
            Some(&"44100".to_string())
        );
        assert!(context.recoverable);
    }

    #[test]
    fn test_error_context_summary() {
        let context = ErrorContext::new(
            ErrorSeverity::Critical,
            ErrorCategory::Network,
            "Connection timeout",
            "network_manager",
        );

        let summary = context.summary();
        assert!(summary.contains("CRIT"));
        assert!(summary.contains("NET"));
        assert!(summary.contains("Connection timeout"));
        assert!(summary.contains("network_manager"));
    }

    #[test]
    fn test_error_context_display() {
        let context = ErrorContext::new(
            ErrorSeverity::High,
            ErrorCategory::Database,
            "Query failed",
            "db_manager",
        )
        .with_details("Syntax error in SQL")
        .with_resolution_steps(vec![
            "Check SQL syntax".to_string(),
            "Validate table names".to_string(),
        ]);

        let display_string = format!("{}", context);
        assert!(display_string.contains("Query failed"));
        assert!(display_string.contains("Syntax error in SQL"));
        assert!(display_string.contains("Check SQL syntax"));
    }

    #[test]
    fn test_predefined_error_contexts() {
        let db_error =
            ErrorContext::database_connection_error("Connection refused", "postgres_manager");
        assert_eq!(db_error.severity, ErrorSeverity::Critical);
        assert_eq!(db_error.category, ErrorCategory::Database);
        assert!(!db_error.resolution_steps.is_empty());

        let audio_error = ErrorContext::audio_processing_error("Invalid format", "audio_processor");
        assert_eq!(audio_error.severity, ErrorSeverity::High);
        assert_eq!(audio_error.category, ErrorCategory::AudioProcessing);

        let validation_error = ErrorContext::validation_error("Missing field", "validator");
        assert_eq!(validation_error.severity, ErrorSeverity::Medium);
        assert_eq!(validation_error.category, ErrorCategory::Validation);
        assert!(validation_error.recoverable);
    }

    #[test]
    fn test_error_context_json_serialization() {
        let context = ErrorContext::new(
            ErrorSeverity::Medium,
            ErrorCategory::Configuration,
            "Invalid config",
            "config_manager",
        );

        let json = context.to_json().unwrap();
        assert!(json.contains("Invalid config"));
        assert!(json.contains("config_manager"));

        // Test deserialization
        let deserialized: ErrorContext = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.message, context.message);
        assert_eq!(deserialized.source, context.source);
    }

    #[test]
    fn test_user_facing_detection() {
        let user_error =
            ErrorContext::validation_error("Invalid input", "validator").with_user("user123");
        assert!(user_error.is_user_facing());

        let system_error =
            ErrorContext::database_connection_error("Connection failed", "db_manager");
        assert!(!system_error.is_user_facing());
    }

    #[test]
    fn test_immediate_attention_requirement() {
        let critical_error = ErrorContext::new(
            ErrorSeverity::Critical,
            ErrorCategory::Database,
            "Database down",
            "db_manager",
        );
        assert!(critical_error.requires_immediate_attention());

        let medium_error = ErrorContext::new(
            ErrorSeverity::Medium,
            ErrorCategory::AudioProcessing,
            "Processing slow",
            "audio_processor",
        );
        assert!(!medium_error.requires_immediate_attention());
    }
}
