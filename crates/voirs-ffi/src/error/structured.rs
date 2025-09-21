use std::backtrace::Backtrace;
use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;

/// Hierarchical error code system for comprehensive error classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum VoirsErrorCategory {
    /// General success
    Success = 0,

    /// Input validation errors (1000-1999)
    Input = 1000,

    /// Configuration errors (2000-2999)
    Configuration = 2000,

    /// Resource errors (3000-3999)
    Resource = 3000,

    /// Processing errors (4000-4999)
    Processing = 4000,

    /// System errors (5000-5999)
    System = 5000,

    /// Network errors (6000-6999)
    Network = 6000,

    /// Security errors (7000-7999)
    Security = 7000,

    /// Internal errors (8000-8999)
    Internal = 8000,

    /// Unknown errors (9000-9999)
    Unknown = 9000,
}

/// Detailed error subcodes for specific error types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum VoirsErrorSubcode {
    /// General success
    Success = 0,

    /// Input validation subcodes (1000-1099)
    InvalidParameter = 1001,
    NullPointer = 1002,
    InvalidFormat = 1003,
    InvalidRange = 1004,

    /// Configuration subcodes (2000-2099)
    ConfigurationMissing = 2001,
    ConfigurationInvalid = 2002,
    ConfigurationConflict = 2003,

    /// Resource subcodes (3000-3099)
    OutOfMemory = 3001,
    FileNotFound = 3002,
    PermissionDenied = 3003,
    ResourceExhausted = 3004,
    VoiceNotFound = 3005,

    /// Processing subcodes (4000-4099)
    SynthesisFailed = 4001,
    ParseError = 4002,
    ConversionError = 4003,
    ProcessingTimeout = 4004,
    OperationCancelled = 4005,

    /// System subcodes (5000-5099)
    IoError = 5001,
    SystemCallFailed = 5002,
    ThreadingError = 5003,

    /// Network subcodes (6000-6099)
    NetworkTimeout = 6001,
    ConnectionFailed = 6002,

    /// Security subcodes (7000-7099)
    AuthenticationFailed = 7001,
    AuthorizationFailed = 7002,

    /// Internal subcodes (8000-8099)
    InternalError = 8001,
    AssertionFailed = 8002,
    StateCorruption = 8003,

    /// Unknown subcodes (9000-9099)
    UnknownError = 9001,
}

/// Comprehensive error context information
#[derive(Debug, Clone)]
pub struct VoirsErrorContext {
    /// Error message
    pub message: String,

    /// Function or module where error occurred
    pub location: String,

    /// Additional context information
    pub context: HashMap<String, String>,

    /// Stack trace (when available)
    pub backtrace: Option<String>,

    /// Error timestamp
    pub timestamp: std::time::SystemTime,

    /// Thread ID where error occurred
    pub thread_id: String,

    /// Error severity level
    pub severity: VoirsErrorSeverity,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum VoirsErrorSeverity {
    /// Informational message
    Info = 0,

    /// Warning - operation completed with issues
    Warning = 1,

    /// Error - operation failed but system is stable
    Error = 2,

    /// Critical - system instability or data loss
    Critical = 3,

    /// Fatal - immediate program termination required
    Fatal = 4,
}

/// Structured error with hierarchy and context
#[derive(Debug, Clone)]
pub struct VoirsStructuredError {
    /// Error category
    pub category: VoirsErrorCategory,

    /// Error subcode
    pub subcode: VoirsErrorSubcode,

    /// Error context
    pub context: VoirsErrorContext,

    /// Chain of related errors
    pub error_chain: Vec<VoirsStructuredError>,

    /// Unique error ID for tracking
    pub error_id: String,
}

impl VoirsStructuredError {
    /// Create a new structured error
    pub fn new(
        category: VoirsErrorCategory,
        subcode: VoirsErrorSubcode,
        message: String,
        location: String,
    ) -> Self {
        Self {
            category,
            subcode,
            context: VoirsErrorContext {
                message,
                location,
                context: HashMap::new(),
                backtrace: capture_backtrace(),
                timestamp: std::time::SystemTime::now(),
                thread_id: format!("{:?}", std::thread::current().id()),
                severity: VoirsErrorSeverity::Error,
            },
            error_chain: Vec::new(),
            error_id: generate_error_id(),
        }
    }

    /// Add context information to the error
    pub fn with_context(mut self, key: String, value: String) -> Self {
        self.context.context.insert(key, value);
        self
    }

    /// Set error severity
    pub fn with_severity(mut self, severity: VoirsErrorSeverity) -> Self {
        self.context.severity = severity;
        self
    }

    /// Add an error to the chain
    pub fn chain_error(mut self, error: VoirsStructuredError) -> Self {
        self.error_chain.push(error);
        self
    }

    /// Get full error message including context
    pub fn full_message(&self) -> String {
        let mut message = format!(
            "[{}] {} ({})",
            self.error_id, self.context.message, self.context.location
        );

        if !self.context.context.is_empty() {
            message.push_str(&format!(" - Context: {:?}", self.context.context));
        }

        if !self.error_chain.is_empty() {
            message.push_str(" - Caused by:");
            for error in &self.error_chain {
                message.push_str(&format!(" {}", error.context.message));
            }
        }

        message
    }

    /// Convert to FFI-safe error code
    pub fn to_error_code(&self) -> crate::VoirsErrorCode {
        match self.subcode {
            VoirsErrorSubcode::Success => crate::VoirsErrorCode::Success,
            VoirsErrorSubcode::InvalidParameter
            | VoirsErrorSubcode::NullPointer
            | VoirsErrorSubcode::InvalidFormat
            | VoirsErrorSubcode::InvalidRange => crate::VoirsErrorCode::InvalidParameter,
            VoirsErrorSubcode::ConfigurationMissing
            | VoirsErrorSubcode::ConfigurationInvalid
            | VoirsErrorSubcode::ConfigurationConflict => {
                crate::VoirsErrorCode::InitializationFailed
            }
            VoirsErrorSubcode::OutOfMemory => crate::VoirsErrorCode::OutOfMemory,
            VoirsErrorSubcode::VoiceNotFound => crate::VoirsErrorCode::VoiceNotFound,
            VoirsErrorSubcode::SynthesisFailed => crate::VoirsErrorCode::SynthesisFailed,
            VoirsErrorSubcode::IoError => crate::VoirsErrorCode::IoError,
            VoirsErrorSubcode::OperationCancelled => crate::VoirsErrorCode::OperationCancelled,
            _ => crate::VoirsErrorCode::InternalError,
        }
    }
}

impl fmt::Display for VoirsStructuredError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.full_message())
    }
}

impl std::error::Error for VoirsStructuredError {}

/// Global error aggregator for collecting and analyzing errors
static ERROR_AGGREGATOR: std::sync::LazyLock<Mutex<ErrorAggregator>> =
    std::sync::LazyLock::new(|| Mutex::new(ErrorAggregator::new()));

/// Error aggregation system
pub struct ErrorAggregator {
    /// Collected errors
    errors: Vec<VoirsStructuredError>,

    /// Error statistics
    stats: HashMap<VoirsErrorCategory, u64>,

    /// Maximum number of errors to store
    max_errors: usize,
}

impl ErrorAggregator {
    /// Create a new error aggregator
    fn new() -> Self {
        Self {
            errors: Vec::new(),
            stats: HashMap::new(),
            max_errors: 1000,
        }
    }

    /// Add an error to the aggregator
    pub fn add_error(&mut self, error: VoirsStructuredError) {
        // Update statistics
        *self.stats.entry(error.category).or_insert(0) += 1;

        // Add to error collection
        self.errors.push(error);

        // Maintain maximum size
        if self.errors.len() > self.max_errors {
            self.errors.remove(0);
        }
    }

    /// Get error statistics
    pub fn get_stats(&self) -> HashMap<VoirsErrorCategory, u64> {
        self.stats.clone()
    }

    /// Get recent errors
    pub fn get_recent_errors(&self, count: usize) -> Vec<VoirsStructuredError> {
        self.errors.iter().rev().take(count).cloned().collect()
    }

    /// Clear all errors
    pub fn clear(&mut self) {
        self.errors.clear();
        self.stats.clear();
    }
}

/// Capture stack trace if available
fn capture_backtrace() -> Option<String> {
    if std::env::var("RUST_BACKTRACE").is_ok() {
        Some(format!("{}", Backtrace::capture()))
    } else {
        None
    }
}

/// Generate unique error ID
fn generate_error_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static ERROR_COUNTER: AtomicU64 = AtomicU64::new(0);

    let id = ERROR_COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("ERR-{:08X}", id)
}

/// Add error to global aggregator
pub fn add_error_to_aggregator(error: VoirsStructuredError) {
    if let Ok(mut aggregator) = ERROR_AGGREGATOR.lock() {
        aggregator.add_error(error);
    }
}

/// Get error statistics from global aggregator
pub fn get_error_stats() -> HashMap<VoirsErrorCategory, u64> {
    ERROR_AGGREGATOR
        .lock()
        .map(|a| a.get_stats())
        .unwrap_or_default()
}

/// Get recent errors from global aggregator
pub fn get_recent_errors(count: usize) -> Vec<VoirsStructuredError> {
    ERROR_AGGREGATOR
        .lock()
        .map(|a| a.get_recent_errors(count))
        .unwrap_or_default()
}

/// Clear all errors from global aggregator
pub fn clear_error_aggregator() {
    if let Ok(mut aggregator) = ERROR_AGGREGATOR.lock() {
        aggregator.clear();
    }
}

/// C API functions for structured error handling
#[no_mangle]
pub unsafe extern "C" fn voirs_get_error_stats(
    stats: *mut VoirsErrorCategory,
    counts: *mut u64,
    max_entries: usize,
) -> usize {
    if stats.is_null() || counts.is_null() {
        return 0;
    }

    let error_stats = get_error_stats();
    let mut written = 0;

    for (category, count) in error_stats.iter().take(max_entries) {
        unsafe {
            *stats.add(written) = *category;
            *counts.add(written) = *count;
        }
        written += 1;
    }

    written
}

#[no_mangle]
pub unsafe extern "C" fn voirs_get_recent_errors(
    error_ids: *mut *const u8,
    max_entries: usize,
) -> usize {
    if error_ids.is_null() {
        return 0;
    }

    let errors = get_recent_errors(max_entries);
    let mut written = 0;

    for error in errors.iter().take(max_entries) {
        let c_string = std::ffi::CString::new(error.error_id.clone()).unwrap_or_default();
        unsafe {
            *error_ids.add(written) = c_string.into_raw() as *const u8;
        }
        written += 1;
    }

    written
}

#[no_mangle]
pub extern "C" fn voirs_clear_error_aggregator() {
    clear_error_aggregator();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structured_error_creation() {
        let error = VoirsStructuredError::new(
            VoirsErrorCategory::Input,
            VoirsErrorSubcode::InvalidParameter,
            "Test error message".to_string(),
            "test_function".to_string(),
        );

        assert_eq!(error.category, VoirsErrorCategory::Input);
        assert_eq!(error.subcode, VoirsErrorSubcode::InvalidParameter);
        assert_eq!(error.context.message, "Test error message");
        assert!(!error.error_id.is_empty());
    }

    #[test]
    fn test_error_context() {
        let error = VoirsStructuredError::new(
            VoirsErrorCategory::Processing,
            VoirsErrorSubcode::SynthesisFailed,
            "Synthesis failed".to_string(),
            "synthesis_module".to_string(),
        )
        .with_context("voice_id".to_string(), "en_US_female".to_string())
        .with_severity(VoirsErrorSeverity::Critical);

        assert_eq!(error.context.severity, VoirsErrorSeverity::Critical);
        assert_eq!(
            error.context.context.get("voice_id"),
            Some(&"en_US_female".to_string())
        );
    }

    #[test]
    fn test_error_chain() {
        let root_error = VoirsStructuredError::new(
            VoirsErrorCategory::System,
            VoirsErrorSubcode::IoError,
            "File read failed".to_string(),
            "file_io".to_string(),
        );

        let chained_error = VoirsStructuredError::new(
            VoirsErrorCategory::Processing,
            VoirsErrorSubcode::SynthesisFailed,
            "Synthesis failed due to file error".to_string(),
            "synthesis".to_string(),
        )
        .chain_error(root_error);

        assert_eq!(chained_error.error_chain.len(), 1);
        assert_eq!(
            chained_error.error_chain[0].subcode,
            VoirsErrorSubcode::IoError
        );
    }

    #[test]
    fn test_error_aggregator() {
        clear_error_aggregator();

        let error1 = VoirsStructuredError::new(
            VoirsErrorCategory::Input,
            VoirsErrorSubcode::InvalidParameter,
            "Error 1".to_string(),
            "test".to_string(),
        );

        let error2 = VoirsStructuredError::new(
            VoirsErrorCategory::Processing,
            VoirsErrorSubcode::SynthesisFailed,
            "Error 2".to_string(),
            "test".to_string(),
        );

        add_error_to_aggregator(error1);
        add_error_to_aggregator(error2);

        let stats = get_error_stats();
        assert_eq!(stats.get(&VoirsErrorCategory::Input), Some(&1));
        assert_eq!(stats.get(&VoirsErrorCategory::Processing), Some(&1));

        let recent = get_recent_errors(2);
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_error_to_ffi_code() {
        let error = VoirsStructuredError::new(
            VoirsErrorCategory::Input,
            VoirsErrorSubcode::InvalidParameter,
            "Test".to_string(),
            "test".to_string(),
        );

        assert_eq!(
            error.to_error_code(),
            crate::VoirsErrorCode::InvalidParameter
        );
    }
}
