//! Enhanced error handling for dataset operations
//!
//! This module provides comprehensive error types with detailed context
//! for debugging and error reporting.

use thiserror::Error;

/// Comprehensive dataset error types with context
#[derive(Error, Debug)]
pub enum DetailedDatasetError {
    #[error("IO error at {location}: {source}")]
    IoError {
        location: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Dataset loading failed for {dataset} at {path}: {reason}")]
    LoadError {
        dataset: String,
        path: String,
        reason: String,
    },

    #[error("Invalid format for {file}: expected {expected}, found {found}")]
    FormatError {
        file: String,
        expected: String,
        found: String,
    },

    #[error("Configuration error in {component}: {details}")]
    ConfigError { component: String, details: String },

    #[error("Audio processing error for {sample_id}: {operation} failed - {details}")]
    AudioError {
        sample_id: String,
        operation: String,
        details: String,
    },

    #[error("Network error during {operation}: {details}")]
    NetworkError { operation: String, details: String },

    #[error("Validation error for {sample_id}: {check} failed - {details}")]
    ValidationError {
        sample_id: String,
        check: String,
        details: String,
    },

    #[error("Preprocessing error for {sample_id} in step {step}: {details}")]
    PreprocessingError {
        sample_id: String,
        step: String,
        details: String,
    },

    #[error("Index {index} out of bounds for dataset with {size} samples")]
    IndexError { index: usize, size: usize },

    #[error("Memory allocation error: {details}")]
    MemoryError { details: String },

    #[error("Timeout error during {operation}: exceeded {timeout_seconds}s")]
    TimeoutError {
        operation: String,
        timeout_seconds: u64,
    },

    #[error("Permission error: {operation} denied for {resource}")]
    PermissionError { operation: String, resource: String },

    #[error("Dependency error: {dependency} not available - {details}")]
    DependencyError { dependency: String, details: String },
}

/// Error context for adding additional information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Error location (file, function, line)
    pub location: String,
    /// Sample ID if applicable
    pub sample_id: Option<String>,
    /// Operation being performed
    pub operation: Option<String>,
    /// Additional context data
    pub context_data: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    /// Create new error context
    pub fn new(location: String) -> Self {
        Self {
            location,
            sample_id: None,
            operation: None,
            context_data: std::collections::HashMap::new(),
        }
    }

    /// Add sample ID to context
    pub fn with_sample_id(mut self, sample_id: String) -> Self {
        self.sample_id = Some(sample_id);
        self
    }

    /// Add operation to context
    pub fn with_operation(mut self, operation: String) -> Self {
        self.operation = Some(operation);
        self
    }

    /// Add context data
    pub fn with_data(mut self, key: String, value: String) -> Self {
        self.context_data.insert(key, value);
        self
    }
}

/// Error chain for tracking error propagation
#[derive(Debug)]
pub struct ErrorChain {
    /// Primary error
    pub primary: DetailedDatasetError,
    /// Chain of contributing errors
    pub chain: Vec<DetailedDatasetError>,
    /// Error context
    pub context: ErrorContext,
}

impl ErrorChain {
    /// Create new error chain
    pub fn new(error: DetailedDatasetError, context: ErrorContext) -> Self {
        Self {
            primary: error,
            chain: Vec::new(),
            context,
        }
    }

    /// Add error to chain
    pub fn add_error(mut self, error: DetailedDatasetError) -> Self {
        self.chain.push(error);
        self
    }

    /// Get full error message with chain
    pub fn full_message(&self) -> String {
        let mut message = format!("Primary error: {}", self.primary);

        if !self.chain.is_empty() {
            message.push_str("\\nError chain:");
            for (i, error) in self.chain.iter().enumerate() {
                message.push_str(&format!("\\n  {}: {error}", i + 1));
            }
        }

        message.push_str(&format!("\\nContext: {}", self.context.location));
        if let Some(sample_id) = &self.context.sample_id {
            message.push_str(&format!(" (sample: {sample_id})"));
        }
        if let Some(operation) = &self.context.operation {
            message.push_str(&format!(" (operation: {operation})"));
        }

        message
    }
}

/// Error reporter for logging and metrics
pub struct ErrorReporter {
    /// Error counts by type
    error_counts: std::collections::HashMap<String, usize>,
    /// Recent errors
    recent_errors: Vec<ErrorChain>,
    /// Maximum number of recent errors to keep
    max_recent: usize,
}

impl ErrorReporter {
    /// Create new error reporter
    pub fn new(max_recent: usize) -> Self {
        Self {
            error_counts: std::collections::HashMap::new(),
            recent_errors: Vec::new(),
            max_recent,
        }
    }

    /// Report an error
    pub fn report_error(&mut self, error_chain: ErrorChain) {
        // Count error type
        let error_type = self.get_error_type(&error_chain.primary);
        *self.error_counts.entry(error_type).or_insert(0) += 1;

        // Add to recent errors
        self.recent_errors.push(error_chain);
        if self.recent_errors.len() > self.max_recent {
            self.recent_errors.remove(0);
        }
    }

    /// Get error statistics
    pub fn get_error_stats(&self) -> std::collections::HashMap<String, usize> {
        self.error_counts.clone()
    }

    /// Get recent errors
    pub fn get_recent_errors(&self) -> &[ErrorChain] {
        &self.recent_errors
    }

    /// Clear error history
    pub fn clear(&mut self) {
        self.error_counts.clear();
        self.recent_errors.clear();
    }

    /// Get error type string
    fn get_error_type(&self, error: &DetailedDatasetError) -> String {
        match error {
            DetailedDatasetError::IoError { .. } => "IO".to_string(),
            DetailedDatasetError::LoadError { .. } => "Load".to_string(),
            DetailedDatasetError::FormatError { .. } => "Format".to_string(),
            DetailedDatasetError::ConfigError { .. } => "Config".to_string(),
            DetailedDatasetError::AudioError { .. } => "Audio".to_string(),
            DetailedDatasetError::NetworkError { .. } => "Network".to_string(),
            DetailedDatasetError::ValidationError { .. } => "Validation".to_string(),
            DetailedDatasetError::PreprocessingError { .. } => "Preprocessing".to_string(),
            DetailedDatasetError::IndexError { .. } => "Index".to_string(),
            DetailedDatasetError::MemoryError { .. } => "Memory".to_string(),
            DetailedDatasetError::TimeoutError { .. } => "Timeout".to_string(),
            DetailedDatasetError::PermissionError { .. } => "Permission".to_string(),
            DetailedDatasetError::DependencyError { .. } => "Dependency".to_string(),
        }
    }
}

/// Convenience macros for error creation
#[macro_export]
macro_rules! dataset_error {
    (io, $location:expr, $source:expr) => {
        DetailedDatasetError::IoError {
            location: $location.to_string(),
            source: $source,
        }
    };

    (load, $dataset:expr, $path:expr, $reason:expr) => {
        DetailedDatasetError::LoadError {
            dataset: $dataset.to_string(),
            path: $path.to_string(),
            reason: $reason.to_string(),
        }
    };

    (audio, $sample_id:expr, $operation:expr, $details:expr) => {
        DetailedDatasetError::AudioError {
            sample_id: $sample_id.to_string(),
            operation: $operation.to_string(),
            details: $details.to_string(),
        }
    };
}
