//! Enhanced error handling with diagnostic context and recovery suggestions
//!
//! This module provides advanced error handling capabilities including:
//! - Detailed error context with operation tracking
//! - Recovery suggestions for common error scenarios
//! - Error categorization by severity
//! - Error chain tracking for root cause analysis
//! - Performance impact analysis

use std::collections::HashMap;
use std::fmt;

/// Error severity levels for categorizing acoustic processing errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorSeverity {
    /// Critical error - system cannot continue
    Critical,
    /// Major error - significant functionality impaired
    Major,
    /// Minor error - degraded performance but functional
    Minor,
    /// Warning - potential issue that may escalate
    Warning,
    /// Info - non-error diagnostic information
    Info,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
            ErrorSeverity::Major => write!(f, "MAJOR"),
            ErrorSeverity::Minor => write!(f, "MINOR"),
            ErrorSeverity::Warning => write!(f, "WARNING"),
            ErrorSeverity::Info => write!(f, "INFO"),
        }
    }
}

/// Error category for grouping related errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Model loading and initialization errors
    ModelLoading,
    /// Inference and forward pass errors
    Inference,
    /// Input validation and processing errors
    InputValidation,
    /// Configuration and parameter errors
    Configuration,
    /// Resource allocation and management errors
    ResourceManagement,
    /// Performance and optimization errors
    Performance,
    /// Backend-specific errors (Candle, ONNX)
    Backend,
    /// File I/O and storage errors
    FileSystem,
    /// Network and remote resource errors
    Network,
    /// Memory allocation and management errors
    Memory,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::ModelLoading => write!(f, "Model Loading"),
            ErrorCategory::Inference => write!(f, "Inference"),
            ErrorCategory::InputValidation => write!(f, "Input Validation"),
            ErrorCategory::Configuration => write!(f, "Configuration"),
            ErrorCategory::ResourceManagement => write!(f, "Resource Management"),
            ErrorCategory::Performance => write!(f, "Performance"),
            ErrorCategory::Backend => write!(f, "Backend"),
            ErrorCategory::FileSystem => write!(f, "File System"),
            ErrorCategory::Network => write!(f, "Network"),
            ErrorCategory::Memory => write!(f, "Memory"),
        }
    }
}

/// Recovery suggestion for resolving errors
#[derive(Debug, Clone)]
pub struct RecoverySuggestion {
    /// Description of the recovery action
    pub action: String,
    /// Expected impact of the recovery action
    pub impact: String,
    /// Priority level (higher = more recommended)
    pub priority: u8,
    /// Whether this action is safe to automate
    pub automatable: bool,
}

impl RecoverySuggestion {
    /// Create a new recovery suggestion
    pub fn new(action: impl Into<String>, impact: impl Into<String>) -> Self {
        Self {
            action: action.into(),
            impact: impact.into(),
            priority: 5,
            automatable: false,
        }
    }

    /// Set priority level (0-10)
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.min(10);
        self
    }

    /// Mark as safe for automation
    pub fn automatable(mut self) -> Self {
        self.automatable = true;
        self
    }
}

/// Enhanced error context with diagnostic information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Error category
    pub category: ErrorCategory,
    /// Operation that failed
    pub operation: String,
    /// Additional context information
    pub context: HashMap<String, String>,
    /// Recovery suggestions
    pub suggestions: Vec<RecoverySuggestion>,
    /// Related errors (error chain)
    pub related_errors: Vec<String>,
    /// Performance impact estimate (0.0-1.0)
    pub performance_impact: Option<f32>,
}

impl ErrorContext {
    /// Create new error context
    pub fn new(
        severity: ErrorSeverity,
        category: ErrorCategory,
        operation: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            category,
            operation: operation.into(),
            context: HashMap::new(),
            suggestions: Vec::new(),
            related_errors: Vec::new(),
            performance_impact: None,
        }
    }

    /// Add context information
    pub fn add_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Add recovery suggestion
    pub fn add_suggestion(mut self, suggestion: RecoverySuggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    /// Add related error
    pub fn add_related_error(mut self, error: impl Into<String>) -> Self {
        self.related_errors.push(error.into());
        self
    }

    /// Set performance impact
    pub fn with_performance_impact(mut self, impact: f32) -> Self {
        self.performance_impact = Some(impact.clamp(0.0, 1.0));
        self
    }

    /// Get formatted error report
    pub fn format_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!(
            "[{}] {}: {}\n",
            self.severity, self.category, self.operation
        ));

        if !self.context.is_empty() {
            report.push_str("\nContext:\n");
            for (key, value) in &self.context {
                report.push_str(&format!("  {}: {}\n", key, value));
            }
        }

        if let Some(impact) = self.performance_impact {
            report.push_str(&format!("\nPerformance Impact: {:.1}%\n", impact * 100.0));
        }

        if !self.suggestions.is_empty() {
            report.push_str("\nRecovery Suggestions:\n");
            let mut sorted_suggestions = self.suggestions.clone();
            sorted_suggestions.sort_by(|a, b| b.priority.cmp(&a.priority));

            for (i, suggestion) in sorted_suggestions.iter().enumerate() {
                let auto_marker = if suggestion.automatable {
                    " [AUTO]"
                } else {
                    ""
                };
                report.push_str(&format!(
                    "  {}. {}{}\n     Impact: {}\n",
                    i + 1,
                    suggestion.action,
                    auto_marker,
                    suggestion.impact
                ));
            }
        }

        if !self.related_errors.is_empty() {
            report.push_str("\nRelated Errors:\n");
            for error in &self.related_errors {
                report.push_str(&format!("  - {}\n", error));
            }
        }

        report
    }
}

/// Error context builder for common error scenarios
pub struct ErrorContextBuilder;

impl ErrorContextBuilder {
    /// Build context for model loading errors
    pub fn model_loading_error(model_path: &str, reason: &str) -> ErrorContext {
        ErrorContext::new(
            ErrorSeverity::Critical,
            ErrorCategory::ModelLoading,
            "Model loading failed",
        )
        .add_context("model_path", model_path)
        .add_context("failure_reason", reason)
        .add_suggestion(
            RecoverySuggestion::new(
                "Verify model file exists and is accessible",
                "Ensures the model file is present and readable",
            )
            .with_priority(9)
            .automatable(),
        )
        .add_suggestion(
            RecoverySuggestion::new(
                "Check model file format (should be .safetensors or .onnx)",
                "Validates compatibility with supported formats",
            )
            .with_priority(8),
        )
        .add_suggestion(
            RecoverySuggestion::new(
                "Verify sufficient memory available for model loading",
                "Prevents out-of-memory errors during loading",
            )
            .with_priority(7)
            .automatable(),
        )
        .with_performance_impact(1.0)
    }

    /// Build context for inference errors
    pub fn inference_error(
        operation: &str,
        input_shape: &str,
        error_details: &str,
    ) -> ErrorContext {
        ErrorContext::new(
            ErrorSeverity::Major,
            ErrorCategory::Inference,
            format!("Inference failed: {}", operation),
        )
        .add_context("input_shape", input_shape)
        .add_context("error_details", error_details)
        .add_suggestion(
            RecoverySuggestion::new(
                "Verify input tensor dimensions match model expectations",
                "Ensures proper tensor shape for model processing",
            )
            .with_priority(9),
        )
        .add_suggestion(
            RecoverySuggestion::new(
                "Check for NaN or Inf values in input data",
                "Prevents numerical instability in calculations",
            )
            .with_priority(8)
            .automatable(),
        )
        .add_suggestion(
            RecoverySuggestion::new(
                "Try reducing batch size if memory errors occur",
                "Reduces memory pressure during inference",
            )
            .with_priority(6),
        )
        .with_performance_impact(0.8)
    }

    /// Build context for input validation errors
    pub fn input_validation_error(
        field_name: &str,
        actual_value: &str,
        expected_value: &str,
    ) -> ErrorContext {
        ErrorContext::new(
            ErrorSeverity::Major,
            ErrorCategory::InputValidation,
            "Input validation failed",
        )
        .add_context("field", field_name)
        .add_context("actual_value", actual_value)
        .add_context("expected_value", expected_value)
        .add_suggestion(
            RecoverySuggestion::new(
                format!(
                    "Adjust {} to be within valid range: {}",
                    field_name, expected_value
                ),
                "Ensures parameter complies with model constraints",
            )
            .with_priority(9),
        )
        .add_suggestion(
            RecoverySuggestion::new(
                "Use configuration presets for known-good settings",
                "Provides tested parameter combinations",
            )
            .with_priority(7),
        )
        .with_performance_impact(0.3)
    }

    /// Build context for configuration errors
    pub fn configuration_error(config_name: &str, issue: &str) -> ErrorContext {
        ErrorContext::new(
            ErrorSeverity::Major,
            ErrorCategory::Configuration,
            "Configuration error",
        )
        .add_context("configuration", config_name)
        .add_context("issue", issue)
        .add_suggestion(
            RecoverySuggestion::new(
                "Review configuration documentation for valid parameter ranges",
                "Ensures all parameters are correctly specified",
            )
            .with_priority(8),
        )
        .add_suggestion(
            RecoverySuggestion::new(
                "Reset to default configuration if custom settings fail",
                "Provides a known-working baseline configuration",
            )
            .with_priority(7)
            .automatable(),
        )
        .with_performance_impact(0.5)
    }

    /// Build context for resource management errors
    pub fn resource_error(resource_type: &str, requested: &str, available: &str) -> ErrorContext {
        ErrorContext::new(
            ErrorSeverity::Critical,
            ErrorCategory::ResourceManagement,
            format!("{} resource exhausted", resource_type),
        )
        .add_context("resource_type", resource_type)
        .add_context("requested", requested)
        .add_context("available", available)
        .add_suggestion(
            RecoverySuggestion::new(
                "Reduce batch size to lower memory requirements",
                "Decreases peak memory usage during processing",
            )
            .with_priority(9),
        )
        .add_suggestion(
            RecoverySuggestion::new(
                "Enable memory pooling and caching to reduce allocations",
                "Reuses allocated memory buffers efficiently",
            )
            .with_priority(8)
            .automatable(),
        )
        .add_suggestion(
            RecoverySuggestion::new(
                "Use streaming synthesis for long sequences",
                "Processes data in chunks to reduce memory footprint",
            )
            .with_priority(7),
        )
        .with_performance_impact(1.0)
    }

    /// Build context for performance degradation
    pub fn performance_degradation(
        operation: &str,
        expected_ms: f32,
        actual_ms: f32,
    ) -> ErrorContext {
        let slowdown_factor = actual_ms / expected_ms;
        ErrorContext::new(
            if slowdown_factor > 3.0 {
                ErrorSeverity::Major
            } else {
                ErrorSeverity::Minor
            },
            ErrorCategory::Performance,
            "Performance degradation detected",
        )
        .add_context("operation", operation)
        .add_context("expected_duration_ms", format!("{:.2}", expected_ms))
        .add_context("actual_duration_ms", format!("{:.2}", actual_ms))
        .add_context("slowdown_factor", format!("{:.2}x", slowdown_factor))
        .add_suggestion(
            RecoverySuggestion::new(
                "Enable GPU acceleration if available (CUDA, Metal, CoreML)",
                "Can provide up to 10x speedup for synthesis operations",
            )
            .with_priority(9),
        )
        .add_suggestion(
            RecoverySuggestion::new(
                "Enable result caching for repeated synthesis requests",
                "Eliminates recomputation for identical inputs",
            )
            .with_priority(8)
            .automatable(),
        )
        .add_suggestion(
            RecoverySuggestion::new(
                "Use lower quality settings for faster synthesis",
                "Trades output quality for reduced processing time",
            )
            .with_priority(6),
        )
        .add_suggestion(
            RecoverySuggestion::new(
                "Profile operation to identify specific bottlenecks",
                "Enables targeted optimization of slow components",
            )
            .with_priority(5),
        )
        .with_performance_impact((slowdown_factor - 1.0) / slowdown_factor)
    }

    /// Build context for memory allocation errors
    pub fn memory_allocation_error(requested_bytes: usize, operation: &str) -> ErrorContext {
        ErrorContext::new(
            ErrorSeverity::Critical,
            ErrorCategory::Memory,
            "Memory allocation failed",
        )
        .add_context("requested_bytes", format!("{}", requested_bytes))
        .add_context(
            "requested_mb",
            format!("{:.2}", requested_bytes as f64 / 1024.0 / 1024.0),
        )
        .add_context("operation", operation)
        .add_suggestion(
            RecoverySuggestion::new(
                "Close other applications to free system memory",
                "Increases available memory for synthesis operations",
            )
            .with_priority(8),
        )
        .add_suggestion(
            RecoverySuggestion::new(
                "Use memory pooling to reduce allocation overhead",
                "Reuses pre-allocated buffers for synthesis",
            )
            .with_priority(9)
            .automatable(),
        )
        .add_suggestion(
            RecoverySuggestion::new(
                "Process in smaller batches or chunks",
                "Reduces peak memory requirements",
            )
            .with_priority(7),
        )
        .with_performance_impact(1.0)
    }

    /// Build context for backend errors
    pub fn backend_error(backend_name: &str, operation: &str, error_message: &str) -> ErrorContext {
        ErrorContext::new(
            ErrorSeverity::Major,
            ErrorCategory::Backend,
            format!("{} backend error", backend_name),
        )
        .add_context("backend", backend_name)
        .add_context("operation", operation)
        .add_context("error_message", error_message)
        .add_suggestion(
            RecoverySuggestion::new(
                format!("Try alternative backend if {} fails", backend_name),
                "Provides fallback execution path",
            )
            .with_priority(8),
        )
        .add_suggestion(
            RecoverySuggestion::new(
                "Update backend to latest version",
                "May include bug fixes and performance improvements",
            )
            .with_priority(6),
        )
        .with_performance_impact(0.9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_builder() {
        let ctx = ErrorContext::new(
            ErrorSeverity::Major,
            ErrorCategory::Inference,
            "Test operation",
        )
        .add_context("key1", "value1")
        .add_context("key2", "value2")
        .add_suggestion(RecoverySuggestion::new("Action 1", "Impact 1").with_priority(8))
        .with_performance_impact(0.5);

        assert_eq!(ctx.severity, ErrorSeverity::Major);
        assert_eq!(ctx.category, ErrorCategory::Inference);
        assert_eq!(ctx.context.len(), 2);
        assert_eq!(ctx.suggestions.len(), 1);
        assert_eq!(ctx.performance_impact, Some(0.5));
    }

    #[test]
    fn test_model_loading_error_context() {
        let ctx = ErrorContextBuilder::model_loading_error(
            "/path/to/model.safetensors",
            "File not found",
        );

        assert_eq!(ctx.severity, ErrorSeverity::Critical);
        assert_eq!(ctx.category, ErrorCategory::ModelLoading);
        assert!(!ctx.suggestions.is_empty());
        assert_eq!(ctx.performance_impact, Some(1.0));
    }

    #[test]
    fn test_inference_error_context() {
        let ctx =
            ErrorContextBuilder::inference_error("forward_pass", "[1, 80, 256]", "Shape mismatch");

        assert_eq!(ctx.severity, ErrorSeverity::Major);
        assert_eq!(ctx.category, ErrorCategory::Inference);
        assert!(ctx.suggestions.len() >= 2);
    }

    #[test]
    fn test_performance_degradation_context() {
        let ctx = ErrorContextBuilder::performance_degradation("synthesis", 100.0, 500.0);

        assert!(matches!(
            ctx.severity,
            ErrorSeverity::Major | ErrorSeverity::Minor
        ));
        assert_eq!(ctx.category, ErrorCategory::Performance);
        assert!(!ctx.suggestions.is_empty());
    }

    #[test]
    fn test_error_report_formatting() {
        let ctx = ErrorContext::new(
            ErrorSeverity::Major,
            ErrorCategory::Inference,
            "Test operation",
        )
        .add_context("input_shape", "[1, 80, 256]")
        .add_suggestion(
            RecoverySuggestion::new("Try reducing batch size", "Reduces memory usage")
                .with_priority(8),
        );

        let report = ctx.format_report();
        assert!(report.contains("MAJOR"));
        assert!(report.contains("Inference"));
        assert!(report.contains("input_shape"));
        assert!(report.contains("Recovery Suggestions"));
    }

    #[test]
    fn test_recovery_suggestion_priority() {
        let mut ctx = ErrorContext::new(ErrorSeverity::Major, ErrorCategory::Inference, "Test");

        ctx = ctx.add_suggestion(RecoverySuggestion::new("Low priority", "").with_priority(3));
        ctx = ctx.add_suggestion(RecoverySuggestion::new("High priority", "").with_priority(9));
        ctx = ctx.add_suggestion(RecoverySuggestion::new("Medium priority", "").with_priority(6));

        let report = ctx.format_report();
        // High priority should appear first
        let high_pos = report.find("High priority").unwrap();
        let medium_pos = report.find("Medium priority").unwrap();
        let low_pos = report.find("Low priority").unwrap();

        assert!(high_pos < medium_pos);
        assert!(medium_pos < low_pos);
    }

    #[test]
    fn test_automatable_suggestions() {
        let suggestion =
            RecoverySuggestion::new("Enable caching", "Improves performance").automatable();

        assert!(suggestion.automatable);

        let ctx = ErrorContext::new(
            ErrorSeverity::Minor,
            ErrorCategory::Performance,
            "Slow synthesis",
        )
        .add_suggestion(suggestion);

        let report = ctx.format_report();
        assert!(report.contains("[AUTO]"));
    }

    #[test]
    fn test_error_severity_display() {
        assert_eq!(format!("{}", ErrorSeverity::Critical), "CRITICAL");
        assert_eq!(format!("{}", ErrorSeverity::Major), "MAJOR");
        assert_eq!(format!("{}", ErrorSeverity::Minor), "MINOR");
        assert_eq!(format!("{}", ErrorSeverity::Warning), "WARNING");
        assert_eq!(format!("{}", ErrorSeverity::Info), "INFO");
    }

    #[test]
    fn test_error_category_display() {
        assert_eq!(format!("{}", ErrorCategory::ModelLoading), "Model Loading");
        assert_eq!(format!("{}", ErrorCategory::Inference), "Inference");
        assert_eq!(format!("{}", ErrorCategory::Performance), "Performance");
    }

    #[test]
    fn test_resource_error_context() {
        let ctx = ErrorContextBuilder::resource_error("Memory", "2048 MB", "1024 MB");

        assert_eq!(ctx.severity, ErrorSeverity::Critical);
        assert_eq!(ctx.category, ErrorCategory::ResourceManagement);
        assert!(!ctx.suggestions.is_empty());
    }

    #[test]
    fn test_input_validation_error_context() {
        let ctx = ErrorContextBuilder::input_validation_error("speed", "5.0", "0.1-3.0");

        assert_eq!(ctx.severity, ErrorSeverity::Major);
        assert_eq!(ctx.category, ErrorCategory::InputValidation);
        assert!(ctx.context.contains_key("field"));
        assert!(ctx.context.contains_key("actual_value"));
        assert!(ctx.context.contains_key("expected_value"));
    }
}
