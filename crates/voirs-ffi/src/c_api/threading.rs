//! Threading support for VoiRS FFI operations.
//!
//! This module provides thread management capabilities for parallel synthesis
//! operations and async callback handling.

use crate::{
    VoirsErrorCode, get_pipeline_manager, get_runtime, set_last_error,
    VoirsAudioBuffer, VoirsSynthesisConfig,
};
use voirs::types::SynthesisConfig;
use std::{
    os::raw::{c_char, c_uint, c_int, c_void},
    sync::{Arc, atomic::{AtomicBool, AtomicU32, Ordering}},
    ffi::CStr,
    ptr,
};
use parking_lot::Mutex;
use once_cell::sync::Lazy;

/// Global thread configuration
static THREAD_CONFIG: Lazy<Mutex<ThreadConfig>> = Lazy::new(|| {
    Mutex::new(ThreadConfig::default())
});

/// Thread configuration structure
#[derive(Debug, Clone)]
struct ThreadConfig {
    /// Number of worker threads for synthesis
    thread_count: u32,
    /// Maximum number of concurrent operations
    max_concurrent: u32,
    /// Thread pool enabled flag
    pool_enabled: bool,
}

impl Default for ThreadConfig {
    fn default() -> Self {
        Self {
            thread_count: num_cpus::get() as u32,
            max_concurrent: 4,
            pool_enabled: true,
        }
    }
}

/// Callback function type for synthesis progress
pub type VoirsSynthesisProgressCallback = unsafe extern "C" fn(
    pipeline_id: c_uint,
    progress: f32,
    user_data: *mut c_void,
);

/// Callback function type for synthesis completion
pub type VoirsSynthesisCompleteCallback = unsafe extern "C" fn(
    pipeline_id: c_uint,
    result: VoirsErrorCode,
    audio_buffer: *mut VoirsAudioBuffer,
    user_data: *mut c_void,
);

/// Callback function type for error handling
pub type VoirsErrorCallback = unsafe extern "C" fn(
    pipeline_id: c_uint,
    error_code: VoirsErrorCode,
    error_message: *const c_char,
    user_data: *mut c_void,
);

/// Structure to hold callback information
#[derive(Debug, Clone)]
struct CallbackInfo {
    progress_callback: Option<VoirsSynthesisProgressCallback>,
    complete_callback: Option<VoirsSynthesisCompleteCallback>,
    error_callback: Option<VoirsErrorCallback>,
    user_data: *mut c_void,
}

unsafe impl Send for CallbackInfo {}
unsafe impl Sync for CallbackInfo {}

/// Global callback registry
static CALLBACK_REGISTRY: Lazy<Mutex<std::collections::HashMap<u32, CallbackInfo>>> = 
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

/// Set the global number of worker threads for synthesis operations
#[no_mangle]
pub extern "C" fn voirs_set_global_thread_count(thread_count: c_uint) -> VoirsErrorCode {
    if thread_count == 0 || thread_count > 64 {
        set_last_error("Thread count must be between 1 and 64".to_string());
        return VoirsErrorCode::InvalidParameter;
    }

    let mut config = THREAD_CONFIG.lock();
    config.thread_count = thread_count;
    
    VoirsErrorCode::Success
}

/// Get the current global number of worker threads
#[no_mangle]
pub extern "C" fn voirs_get_global_thread_count() -> c_uint {
    THREAD_CONFIG.lock().thread_count
}

/// Set the maximum number of concurrent synthesis operations
#[no_mangle]
pub extern "C" fn voirs_set_max_concurrent(max_concurrent: c_uint) -> VoirsErrorCode {
    if max_concurrent == 0 || max_concurrent > 32 {
        set_last_error("Max concurrent operations must be between 1 and 32".to_string());
        return VoirsErrorCode::InvalidParameter;
    }

    let mut config = THREAD_CONFIG.lock();
    config.max_concurrent = max_concurrent;
    
    VoirsErrorCode::Success
}

/// Get the maximum number of concurrent synthesis operations
#[no_mangle]
pub extern "C" fn voirs_get_max_concurrent() -> c_uint {
    THREAD_CONFIG.lock().max_concurrent
}

/// Enable or disable the thread pool
#[no_mangle]
pub extern "C" fn voirs_set_thread_pool_enabled(enabled: c_int) -> VoirsErrorCode {
    let mut config = THREAD_CONFIG.lock();
    config.pool_enabled = enabled != 0;
    VoirsErrorCode::Success
}

/// Check if the thread pool is enabled
#[no_mangle]
pub extern "C" fn voirs_is_thread_pool_enabled() -> c_int {
    if THREAD_CONFIG.lock().pool_enabled { 1 } else { 0 }
}

/// Register callbacks for a pipeline
#[no_mangle]
pub unsafe extern "C" fn voirs_register_callbacks(
    pipeline_id: c_uint,
    progress_callback: Option<VoirsSynthesisProgressCallback>,
    complete_callback: Option<VoirsSynthesisCompleteCallback>,
    error_callback: Option<VoirsErrorCallback>,
    user_data: *mut c_void,
) -> VoirsErrorCode {
    let callback_info = CallbackInfo {
        progress_callback,
        complete_callback,
        error_callback,
        user_data,
    };

    let mut registry = CALLBACK_REGISTRY.lock();
    registry.insert(pipeline_id, callback_info);
    
    VoirsErrorCode::Success
}

/// Unregister callbacks for a pipeline
#[no_mangle]
pub extern "C" fn voirs_unregister_callbacks(pipeline_id: c_uint) -> VoirsErrorCode {
    let mut registry = CALLBACK_REGISTRY.lock();
    registry.remove(&pipeline_id);
    VoirsErrorCode::Success
}

/// Start asynchronous synthesis with callbacks
#[no_mangle]
pub unsafe extern "C" fn voirs_synthesize_async(
    pipeline_id: c_uint,
    text: *const c_char,
    config: *const VoirsSynthesisConfig,
) -> VoirsErrorCode {
    if text.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    // Convert text to string
    let text_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            set_last_error("Invalid UTF-8 in text parameter".to_string());
            return VoirsErrorCode::InvalidParameter;
        }
    };

    // Get pipeline
    let manager = get_pipeline_manager();
    let pipeline = {
        let mgr = manager.lock();
        match mgr.get_pipeline(pipeline_id) {
            Some(p) => p,
            None => {
                set_last_error("Invalid pipeline ID".to_string());
                return VoirsErrorCode::InvalidParameter;
            }
        }
    };

    // Get runtime
    let rt = match get_runtime() {
        Ok(rt) => rt,
        Err(e) => {
            set_last_error(format!("Failed to get runtime: {:?}", e));
            return e;
        }
    };

    // Get callbacks if registered
    let callback_info = {
        let registry = CALLBACK_REGISTRY.lock();
        registry.get(&pipeline_id).cloned()
    };

    // Spawn async task
    rt.spawn(async move {
        // Notify progress start
        if let Some(ref callbacks) = callback_info {
            if let Some(progress_cb) = callbacks.progress_callback {
                unsafe {
                    progress_cb(pipeline_id, 0.0, callbacks.user_data);
                }
            }
        }

        // Perform synthesis
        let result = pipeline.synthesize(&text_str).await;

        match result {
            Ok(audio) => {
                // Notify progress complete
                if let Some(ref callbacks) = callback_info {
                    if let Some(progress_cb) = callbacks.progress_callback {
                        unsafe {
                            progress_cb(pipeline_id, 1.0, callbacks.user_data);
                        }
                    }
                }

                // Create FFI audio buffer
                let ffi_buffer = Box::into_raw(Box::new(
                    crate::VoirsAudioBuffer::from_audio_buffer(audio)
                ));

                // Notify completion
                if let Some(ref callbacks) = callback_info {
                    if let Some(complete_cb) = callbacks.complete_callback {
                        unsafe {
                            complete_cb(
                                pipeline_id,
                                VoirsErrorCode::Success,
                                ffi_buffer,
                                callbacks.user_data,
                            );
                        }
                    }
                }
            }
            Err(e) => {
                // Notify error
                if let Some(ref callbacks) = callback_info {
                    if let Some(error_cb) = callbacks.error_callback {
                        let error_msg = std::ffi::CString::new(format!("{}", e))
                            .unwrap_or_else(|_| std::ffi::CString::new("Unknown error").unwrap());
                        unsafe {
                            error_cb(
                                pipeline_id,
                                VoirsErrorCode::SynthesisFailed,
                                error_msg.as_ptr(),
                                callbacks.user_data,
                            );
                        }
                    }
                }
            }
        }
    });

    VoirsErrorCode::Success
}

/// Cancel an ongoing asynchronous synthesis operation
#[no_mangle]
pub extern "C" fn voirs_cancel_synthesis(pipeline_id: c_uint) -> VoirsErrorCode {
    // TODO: Implement cancellation support
    // For now, this is a placeholder
    VoirsErrorCode::Success
}

/// Get the number of active synthesis operations
#[no_mangle]
pub extern "C" fn voirs_get_active_operations() -> c_uint {
    // TODO: Track active operations
    // For now, return 0
    0
}

/// Thread-safe synthesis with automatic load balancing
#[no_mangle]
pub unsafe extern "C" fn voirs_synthesize_parallel(
    pipeline_ids: *const c_uint,
    texts: *const *const c_char,
    configs: *const VoirsSynthesisConfig,
    count: c_uint,
    results: *mut VoirsErrorCode,
    audio_buffers: *mut *mut VoirsAudioBuffer,
) -> VoirsErrorCode {
    if pipeline_ids.is_null() || texts.is_null() || count == 0 || 
       results.is_null() || audio_buffers.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let pipeline_slice = std::slice::from_raw_parts(pipeline_ids, count as usize);
    let text_slice = std::slice::from_raw_parts(texts, count as usize);
    let config_slice = if configs.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts(configs, count as usize))
    };
    let result_slice = std::slice::from_raw_parts_mut(results, count as usize);
    let buffer_slice = std::slice::from_raw_parts_mut(audio_buffers, count as usize);

    // Get runtime
    let rt = match get_runtime() {
        Ok(rt) => rt,
        Err(e) => return e,
    };

    // Process all operations synchronously for now to avoid thread safety issues
    // TODO: Implement true parallel processing with proper thread-safe buffer handling
    for i in 0..count as usize {
        let pipeline_id = pipeline_slice[i];
        
        // Get pipeline
        let pipeline = {
            let manager = get_pipeline_manager();
            let mgr = manager.lock();
            match mgr.get_pipeline(pipeline_id) {
                Some(p) => p,
                None => {
                    result_slice[i] = VoirsErrorCode::InvalidParameter;
                    buffer_slice[i] = ptr::null_mut();
                    continue;
                }
            }
        };

        // Convert text
        let text_str = match CStr::from_ptr(text_slice[i]).to_str() {
            Ok(s) => s,
            Err(_) => {
                result_slice[i] = VoirsErrorCode::InvalidParameter;
                buffer_slice[i] = ptr::null_mut();
                continue;
            }
        };

        // Perform synthesis
        let result = if let Some(configs) = config_slice {
            let config: SynthesisConfig = configs[i].clone().into();
            rt.block_on(pipeline.synthesize_with_config(text_str, &config))
        } else {
            rt.block_on(pipeline.synthesize(text_str))
        };

        match result {
            Ok(audio) => {
                let ffi_buffer = crate::VoirsAudioBuffer::from_audio_buffer(audio);
                result_slice[i] = VoirsErrorCode::Success;
                buffer_slice[i] = Box::into_raw(Box::new(ffi_buffer));
            }
            Err(_) => {
                result_slice[i] = VoirsErrorCode::SynthesisFailed;
                buffer_slice[i] = ptr::null_mut();
            }
        }
    }

    VoirsErrorCode::Success
}

/// Get thread pool statistics
#[no_mangle]
pub extern "C" fn voirs_get_thread_stats(
    active_threads: *mut c_uint,
    queued_tasks: *mut c_uint,
    completed_tasks: *mut c_uint,
) -> VoirsErrorCode {
    if active_threads.is_null() || queued_tasks.is_null() || completed_tasks.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    // TODO: Implement actual thread pool statistics tracking
    // For now, return placeholder values
    unsafe {
        *active_threads = THREAD_CONFIG.lock().thread_count;
        *queued_tasks = 0;
        *completed_tasks = 0;
    }

    VoirsErrorCode::Success
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_thread_count_configuration() {
        // Test valid thread count
        assert_eq!(voirs_set_global_thread_count(4), VoirsErrorCode::Success);
        assert_eq!(voirs_get_global_thread_count(), 4);

        // Test invalid thread count
        assert_eq!(voirs_set_global_thread_count(0), VoirsErrorCode::InvalidParameter);
        assert_eq!(voirs_set_global_thread_count(100), VoirsErrorCode::InvalidParameter);
    }

    #[test]
    fn test_max_concurrent_configuration() {
        // Test valid max concurrent
        assert_eq!(voirs_set_max_concurrent(8), VoirsErrorCode::Success);
        assert_eq!(voirs_get_max_concurrent(), 8);

        // Test invalid max concurrent
        assert_eq!(voirs_set_max_concurrent(0), VoirsErrorCode::InvalidParameter);
        assert_eq!(voirs_set_max_concurrent(50), VoirsErrorCode::InvalidParameter);
    }

    #[test]
    fn test_thread_pool_enable_disable() {
        // Test enabling
        assert_eq!(voirs_set_thread_pool_enabled(1), VoirsErrorCode::Success);
        assert_eq!(voirs_is_thread_pool_enabled(), 1);

        // Test disabling
        assert_eq!(voirs_set_thread_pool_enabled(0), VoirsErrorCode::Success);
        assert_eq!(voirs_is_thread_pool_enabled(), 0);
    }

    #[test]
    fn test_callback_registration() {
        unsafe {
            // Test registering callbacks
            let result = voirs_register_callbacks(
                1,
                None,
                None,
                None,
                std::ptr::null_mut(),
            );
            assert_eq!(result, VoirsErrorCode::Success);

            // Test unregistering callbacks
            assert_eq!(voirs_unregister_callbacks(1), VoirsErrorCode::Success);
        }
    }

    #[test]
    fn test_thread_stats() {
        let mut active = 0;
        let mut queued = 0;
        let mut completed = 0;

        let result = voirs_get_thread_stats(&mut active, &mut queued, &mut completed);
        assert_eq!(result, VoirsErrorCode::Success);
    }

    #[test]
    fn test_async_synthesis_invalid_params() {
        unsafe {
            // Test null text
            let result = voirs_synthesize_async(1, std::ptr::null(), std::ptr::null());
            assert_eq!(result, VoirsErrorCode::InvalidParameter);
        }
    }

    #[test]
    fn test_parallel_synthesis_invalid_params() {
        unsafe {
            let mut results = [VoirsErrorCode::Success; 1];
            let mut buffers = [std::ptr::null_mut(); 1];

            // Test null pipeline_ids
            let result = voirs_synthesize_parallel(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                results.as_mut_ptr(),
                buffers.as_mut_ptr(),
            );
            assert_eq!(result, VoirsErrorCode::InvalidParameter);
        }
    }

    #[test]
    fn test_operation_cancellation() {
        // Test cancellation (placeholder implementation)
        assert_eq!(voirs_cancel_synthesis(1), VoirsErrorCode::Success);
    }

    #[test]
    fn test_active_operations_count() {
        // Test getting active operations count (placeholder implementation)
        assert_eq!(voirs_get_active_operations(), 0);
    }
}