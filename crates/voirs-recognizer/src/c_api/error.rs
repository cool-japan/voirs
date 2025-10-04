//! Error handling utilities for the C API.

use super::types::VoirsError;
use std::ffi::{c_char, CString};
use std::ptr;
use std::sync::Mutex;

/// Global error handler for the C API
pub struct VoirsErrorHandler {
    last_error_message: Mutex<Option<CString>>,
}

impl VoirsErrorHandler {
    /// new
    pub fn new() -> Self {
        Self {
            last_error_message: Mutex::new(None),
        }
    }

    /// Set the last error message
    pub fn set_error(&self, message: &str) {
        if let Ok(mut last_error) = self.last_error_message.lock() {
            *last_error = CString::new(message).ok();
        }
    }

    /// Get the last error message
    pub fn get_error(&self) -> *const c_char {
        if let Ok(last_error) = self.last_error_message.lock() {
            match last_error.as_ref() {
                Some(c_string) => c_string.as_ptr(),
                None => ptr::null(),
            }
        } else {
            ptr::null()
        }
    }

    /// Clear the last error message
    pub fn clear_error(&self) {
        if let Ok(mut last_error) = self.last_error_message.lock() {
            *last_error = None;
        }
    }
}

impl Default for VoirsErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Global error handler instance
static ERROR_HANDLER: std::sync::LazyLock<VoirsErrorHandler> =
    std::sync::LazyLock::new(|| VoirsErrorHandler::new());

/// Get the last error message from the global error handler
///
/// Returns a pointer to a null-terminated C string containing the last error message,
/// or null if no error has occurred. The returned pointer is valid until the next
/// error occurs or until voirs_clear_error() is called.
#[no_mangle]
pub extern "C" fn voirs_get_last_error() -> *const c_char {
    ERROR_HANDLER.get_error()
}

/// Clear the last error message
#[no_mangle]
pub extern "C" fn voirs_clear_error() {
    ERROR_HANDLER.clear_error();
}

/// Convert a VoirsError to a human-readable string
///
/// # Arguments
/// * `error` - The error code to convert
///
/// # Returns
/// A pointer to a null-terminated C string describing the error.
/// The returned pointer is valid for the lifetime of the program.
#[no_mangle]
/// Item
pub extern "C" fn voirs_error_to_string(error: VoirsError) -> *const c_char {
    let message = match error {
        VoirsError::Success => "Success",
        VoirsError::InvalidArgument => "Invalid argument provided",
        VoirsError::NullPointer => "Null pointer provided where non-null expected",
        VoirsError::InitializationFailed => "Failed to initialize recognizer",
        VoirsError::ModelLoadFailed => "Failed to load recognition model",
        VoirsError::RecognitionFailed => "Speech recognition failed",
        VoirsError::UnsupportedFormat => "Audio format not supported",
        VoirsError::OutOfMemory => "Out of memory",
        VoirsError::InternalError => "Internal error occurred",
        VoirsError::StreamingNotStarted => "Streaming mode not started",
        VoirsError::InvalidConfiguration => "Invalid configuration provided",
    };

    // Store static strings for each error type
    // This ensures the pointers remain valid
    match error {
        VoirsError::Success => "Success\0".as_ptr() as *const c_char,
        VoirsError::InvalidArgument => "Invalid argument provided\0".as_ptr() as *const c_char,
        VoirsError::NullPointer => {
            "Null pointer provided where non-null expected\0".as_ptr() as *const c_char
        }
        VoirsError::InitializationFailed => {
            "Failed to initialize recognizer\0".as_ptr() as *const c_char
        }
        VoirsError::ModelLoadFailed => {
            "Failed to load recognition model\0".as_ptr() as *const c_char
        }
        VoirsError::RecognitionFailed => "Speech recognition failed\0".as_ptr() as *const c_char,
        VoirsError::UnsupportedFormat => "Audio format not supported\0".as_ptr() as *const c_char,
        VoirsError::OutOfMemory => "Out of memory\0".as_ptr() as *const c_char,
        VoirsError::InternalError => "Internal error occurred\0".as_ptr() as *const c_char,
        VoirsError::StreamingNotStarted => "Streaming mode not started\0".as_ptr() as *const c_char,
        VoirsError::InvalidConfiguration => {
            "Invalid configuration provided\0".as_ptr() as *const c_char
        }
    }
}

/// Check if an error code represents success
///
/// # Arguments
/// * `error` - The error code to check
///
/// # Returns
/// true if the error represents success, false otherwise
#[no_mangle]
/// Item
pub extern "C" fn voirs_is_success(error: VoirsError) -> bool {
    matches!(error, VoirsError::Success)
}

/// Check if an error code represents a failure
///
/// # Arguments
/// * `error` - The error code to check
///
/// # Returns
/// true if the error represents a failure, false otherwise
#[no_mangle]
/// Item
pub extern "C" fn voirs_is_error(error: VoirsError) -> bool {
    !matches!(error, VoirsError::Success)
}

/// Internal function to handle errors and set error messages
pub fn handle_error(error: VoirsError, message: &str) -> VoirsError {
    if !matches!(error, VoirsError::Success) {
        ERROR_HANDLER.set_error(message);
    }
    error
}

/// Macro for handling errors with automatic message generation
macro_rules! handle_error_with_message {
    ($error:expr, $message:expr) => {
        crate::c_api::error::handle_error($error, $message)
    };
    ($error:expr) => {
        crate::c_api::error::handle_error($error, &format!("Error: {:?}", $error))
    };
}

pub(crate) use handle_error_with_message;

/// Convert Rust errors to VoirsError codes
pub trait ToVoirsError {
    /// To voirs error
    fn to_voirs_error(&self) -> VoirsError;
}

impl ToVoirsError for crate::RecognitionError {
    fn to_voirs_error(&self) -> VoirsError {
        match self {
            crate::RecognitionError::ModelLoadError { .. } => VoirsError::ModelLoadFailed,
            crate::RecognitionError::AudioProcessingError { .. } => VoirsError::UnsupportedFormat,
            crate::RecognitionError::ConfigurationError { .. } => VoirsError::InvalidConfiguration,
            crate::RecognitionError::TranscriptionError { .. } => VoirsError::RecognitionFailed,
            _ => VoirsError::InternalError,
        }
    }
}

impl ToVoirsError for std::io::Error {
    fn to_voirs_error(&self) -> VoirsError {
        match self.kind() {
            std::io::ErrorKind::NotFound => VoirsError::InvalidArgument,
            std::io::ErrorKind::PermissionDenied => VoirsError::InvalidArgument,
            std::io::ErrorKind::OutOfMemory => VoirsError::OutOfMemory,
            _ => VoirsError::InternalError,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_handler() {
        let handler = VoirsErrorHandler::new();

        // Initially no error
        assert!(handler.get_error().is_null());

        // Set an error
        handler.set_error("Test error message");
        let error_ptr = handler.get_error();
        assert!(!error_ptr.is_null());

        // Verify the error message
        let error_msg = unsafe { std::ffi::CStr::from_ptr(error_ptr).to_str().unwrap() };
        assert_eq!(error_msg, "Test error message");

        // Clear the error
        handler.clear_error();
        assert!(handler.get_error().is_null());
    }

    #[test]
    fn test_error_to_string() {
        let success_str = voirs_error_to_string(VoirsError::Success);
        assert!(!success_str.is_null());

        let error_str = voirs_error_to_string(VoirsError::InvalidArgument);
        assert!(!error_str.is_null());

        let success_msg = unsafe { std::ffi::CStr::from_ptr(success_str).to_str().unwrap() };
        assert_eq!(success_msg, "Success");
    }

    #[test]
    fn test_success_failure_checks() {
        assert!(voirs_is_success(VoirsError::Success));
        assert!(!voirs_is_success(VoirsError::InvalidArgument));

        assert!(!voirs_is_error(VoirsError::Success));
        assert!(voirs_is_error(VoirsError::InvalidArgument));
    }

    #[test]
    fn test_global_error_handler() {
        // Clear any existing error
        voirs_clear_error();
        assert!(voirs_get_last_error().is_null());

        // Set an error
        ERROR_HANDLER.set_error("Global test error");
        let error_ptr = voirs_get_last_error();
        assert!(!error_ptr.is_null());

        // Clear the error
        voirs_clear_error();
        assert!(voirs_get_last_error().is_null());
    }
}
