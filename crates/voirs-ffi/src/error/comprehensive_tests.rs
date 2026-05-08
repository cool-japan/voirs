//! Comprehensive error handling tests for VoiRS FFI

#[cfg(test)]
mod tests {
    use crate::*;
    use std::ptr;

    #[test]
    fn test_error_message_retrieval() {
        let error_codes = [
            VoirsErrorCode::Success,
            VoirsErrorCode::InvalidParameter,
            VoirsErrorCode::OutOfMemory,
            VoirsErrorCode::VoiceNotFound,
            VoirsErrorCode::SynthesisFailed,
        ];

        for &error_code in &error_codes {
            let message_ptr = voirs_error_message(error_code);
            assert!(!message_ptr.is_null());

            let message = unsafe { std::ffi::CStr::from_ptr(message_ptr) };
            let message_str = message.to_str().unwrap();
            assert!(!message_str.is_empty());
            assert!(message_str.len() > 3); // Should be descriptive
        }
    }

    #[test]
    fn test_error_state_management() {
        // Test error state functions
        voirs_clear_error();

        let has_error = voirs_has_error();
        assert_eq!(has_error, 0); // Should have no error initially

        // Get last error (should be null or empty when no error)
        let last_error = voirs_get_last_error();
        // This could be null or point to empty string, both are valid
    }

    #[test]
    fn test_voice_operations_with_invalid_id() {
        // Test voice setting with invalid pipeline ID
        let voice_name = std::ffi::CString::new("test_voice").unwrap();
        let result = voirs_set_voice(999999, voice_name.as_ptr());

        // Should return an error code (non-zero)
        assert_ne!(result, 0);
    }

    #[test]
    fn test_structured_error_functionality() {
        use crate::error::structured::*;

        // Test basic structured error creation
        let error = VoirsStructuredError::new(
            VoirsErrorCategory::Input,
            VoirsErrorSubcode::InvalidParameter,
            "Test error".to_string(),
            "test_function".to_string(),
        );

        assert_eq!(error.category, VoirsErrorCategory::Input);
        assert_eq!(error.subcode, VoirsErrorSubcode::InvalidParameter);
        assert!(!error.error_id.is_empty());

        // Test conversion to FFI error code
        let ffi_code = error.to_error_code();
        assert_eq!(ffi_code, VoirsErrorCode::InvalidParameter);

        // Test error with context
        let error_with_context = error
            .with_context("param_name".to_string(), "invalid_value".to_string())
            .with_severity(VoirsErrorSeverity::Critical);

        assert_eq!(
            error_with_context.context.severity,
            VoirsErrorSeverity::Critical
        );
        assert!(!error_with_context.context.context.is_empty());
    }

    #[test]
    fn test_error_aggregation() {
        use crate::error::structured::*;
        use once_cell::sync::Lazy;
        use std::sync::Mutex;

        // Use a test-specific lock to ensure sequential execution
        static TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Clear any existing errors
        clear_error_aggregator();

        // Small delay to ensure clear completes
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Add some test errors
        let error1 = VoirsStructuredError::new(
            VoirsErrorCategory::Input,
            VoirsErrorSubcode::InvalidParameter,
            "Test error 1".to_string(),
            "test".to_string(),
        );

        let error2 = VoirsStructuredError::new(
            VoirsErrorCategory::Processing,
            VoirsErrorSubcode::SynthesisFailed,
            "Test error 2".to_string(),
            "test".to_string(),
        );

        add_error_to_aggregator(error1);
        add_error_to_aggregator(error2);

        // Small delay to ensure errors are added
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Check statistics (allow for concurrent test interference by using >= instead of ==)
        let stats = get_error_stats();
        assert!(
            stats.get(&VoirsErrorCategory::Input).copied().unwrap_or(0) >= 1,
            "Expected at least 1 Input error, got {}",
            stats.get(&VoirsErrorCategory::Input).copied().unwrap_or(0)
        );
        assert!(
            stats
                .get(&VoirsErrorCategory::Processing)
                .copied()
                .unwrap_or(0)
                >= 1,
            "Expected at least 1 Processing error, got {}",
            stats
                .get(&VoirsErrorCategory::Processing)
                .copied()
                .unwrap_or(0)
        );

        // Check recent errors (allow for more due to possible concurrent tests)
        let recent = get_recent_errors(10);
        assert!(
            recent.len() >= 2,
            "Expected at least 2 recent errors, got {}",
            recent.len()
        );

        // Clean up
        clear_error_aggregator();
        std::thread::sleep(std::time::Duration::from_millis(10));

        let stats_after_clear = get_error_stats();
        assert!(stats_after_clear.is_empty());
    }

    #[test]
    fn test_c_api_error_functions() {
        use crate::error::structured::*;

        // Test C API error statistics function
        let mut categories = [VoirsErrorCategory::Success; 10];
        let mut counts = [0u64; 10];

        // Clear first
        clear_error_aggregator();

        // Add a test error
        let error = VoirsStructuredError::new(
            VoirsErrorCategory::Input,
            VoirsErrorSubcode::InvalidParameter,
            "Test".to_string(),
            "test".to_string(),
        );
        add_error_to_aggregator(error);

        // Get stats via C API
        let count =
            unsafe { voirs_get_error_stats(categories.as_mut_ptr(), counts.as_mut_ptr(), 10) };

        assert!(count > 0);

        // Test recent errors C API
        let mut error_ids = [ptr::null::<u8>(); 5];
        let recent_count = unsafe { voirs_get_recent_errors(error_ids.as_mut_ptr(), 5) };

        assert!(recent_count > 0);

        // Clean up
        voirs_clear_error_aggregator();
    }

    #[test]
    fn test_threading_safety() {
        // Test thread count configuration
        let initial_count = voirs_get_global_thread_count();
        assert!(initial_count > 0);

        let result = voirs_set_global_thread_count(4);
        assert_eq!(result, VoirsErrorCode::Success);

        let new_count = voirs_get_global_thread_count();
        assert_eq!(new_count, 4);

        // Reset to initial
        voirs_set_global_thread_count(initial_count);
    }

    #[test]
    fn test_platform_info() {
        use crate::platform::*;

        let info = PlatformInfo::current();
        assert!(!info.os.is_empty());
        assert!(!info.arch.is_empty());
        assert!(info.cpu_cores > 0);
        assert!(info.total_memory > 0);

        // Test C API functions
        let optimal_threads = voirs_get_optimal_threads();
        assert!(optimal_threads > 0);
        assert!(optimal_threads <= 8);

        let optimal_buffer_size = voirs_get_optimal_buffer_size();
        assert!(optimal_buffer_size > 0);
    }
}
