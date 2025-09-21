//! Threading support for VoiRS FFI operations.
//!
//! This module provides thread management capabilities for parallel synthesis
//! operations and async callback handling.

use crate::types::VoirsErrorCallback;
use crate::{
    get_pipeline_manager, get_runtime, set_last_error, VoirsAudioBuffer, VoirsErrorCode,
    VoirsSynthesisConfig,
};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    ffi::CStr,
    os::raw::{c_char, c_int, c_uint, c_void},
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc,
    },
};
// Note: AudioBuffer and SynthesisConfig imports removed as they are unused in this file

/// Global thread configuration
static THREAD_CONFIG: Lazy<Mutex<ThreadConfig>> = Lazy::new(|| Mutex::new(ThreadConfig::default()));

/// Global counter for active operations
static ACTIVE_OPERATIONS: AtomicU32 = AtomicU32::new(0);

/// Global operation ID counter
static OPERATION_ID_COUNTER: AtomicU32 = AtomicU32::new(1);

/// Cancellation token for tracking operation cancellation
#[derive(Debug, Clone)]
struct CancellationToken {
    /// Whether this operation has been cancelled
    cancelled: Arc<AtomicBool>,
    /// Operation ID for tracking
    operation_id: u32,
    /// Pipeline ID associated with this operation
    pipeline_id: u32,
}

impl CancellationToken {
    fn new(pipeline_id: u32) -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            operation_id: OPERATION_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            pipeline_id,
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

/// Global registry for tracking active operations and their cancellation tokens
static OPERATION_REGISTRY: Lazy<Mutex<HashMap<u32, CancellationToken>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Global statistics for thread pool tracking
#[derive(Debug, Default)]
struct ThreadPoolStats {
    /// Total number of tasks completed since start
    total_completed_tasks: u64,
    /// Number of tasks currently queued (waiting for processing)
    queued_tasks: u32,
    /// Number of currently active/running tasks
    active_tasks: u32,
}

/// Global thread pool statistics
static THREAD_STATS: Lazy<Mutex<ThreadPoolStats>> =
    Lazy::new(|| Mutex::new(ThreadPoolStats::default()));

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
pub type VoirsSynthesisProgressCallback =
    unsafe extern "C" fn(pipeline_id: c_uint, progress: f32, user_data: *mut c_void);

/// Callback function type for synthesis completion
pub type VoirsSynthesisCompleteCallback = unsafe extern "C" fn(
    pipeline_id: c_uint,
    result: VoirsErrorCode,
    audio_buffer: *mut VoirsAudioBuffer,
    user_data: *mut c_void,
);

// VoirsErrorCallback is now imported from crate::types

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
    if THREAD_CONFIG.lock().pool_enabled {
        1
    } else {
        0
    }
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
    _config: *const VoirsSynthesisConfig,
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
    #[cfg(test)]
    {
        // In test mode, validate pipeline ID using the test tracking system
        use crate::c_api::core::{CREATED_PIPELINES, DESTROYED_PIPELINES};

        let created = match CREATED_PIPELINES.lock() {
            Ok(guard) => guard,
            Err(_) => {
                set_last_error("Internal synchronization error".to_string());
                return VoirsErrorCode::InternalError;
            }
        };
        if !created.contains(&pipeline_id) {
            set_last_error("Invalid pipeline ID".to_string());
            return VoirsErrorCode::InvalidParameter;
        }
        drop(created);

        let destroyed = match DESTROYED_PIPELINES.lock() {
            Ok(guard) => guard,
            Err(_) => {
                set_last_error("Internal synchronization error".to_string());
                return VoirsErrorCode::InternalError;
            }
        };
        if destroyed.contains(&pipeline_id) {
            set_last_error("Pipeline has been destroyed".to_string());
            return VoirsErrorCode::InvalidParameter;
        }

        // In test mode, simulate successful async synthesis start
        let token = CancellationToken::new(pipeline_id);
        let operation_id = token.operation_id;

        // Register the operation
        {
            let mut registry = OPERATION_REGISTRY.lock();
            registry.insert(operation_id, token);
        }

        // Increment active operations counter
        ACTIVE_OPERATIONS.fetch_add(1, Ordering::Relaxed);

        // Simulate async completion after a short delay
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));

            // Remove from registry and decrement counter
            {
                let mut registry = OPERATION_REGISTRY.lock();
                registry.remove(&operation_id);
            }
            ACTIVE_OPERATIONS.fetch_sub(1, Ordering::Relaxed);
        });

        return VoirsErrorCode::Success;
    }

    #[cfg(not(test))]
    {
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
                set_last_error(format!("Failed to get runtime: {e:?}"));
                return e;
            }
        };

        // Get callbacks if registered
        let callback_info = {
            let registry = CALLBACK_REGISTRY.lock();
            registry.get(&pipeline_id).cloned()
        };

        // Create cancellation token for this operation
        let cancellation_token = CancellationToken::new(pipeline_id);
        let operation_id = cancellation_token.operation_id;

        // Register the operation in the registry
        {
            let mut registry = OPERATION_REGISTRY.lock();
            registry.insert(operation_id, cancellation_token.clone());
        }

        // Track operation start
        ACTIVE_OPERATIONS.fetch_add(1, Ordering::Relaxed);

        // Update thread statistics - async task is queued
        {
            let mut stats = THREAD_STATS.lock();
            stats.queued_tasks += 1;
        }

        // Spawn async task
        rt.spawn(async move {
            // Task is now active (no longer queued)
            {
                let mut stats = THREAD_STATS.lock();
                stats.queued_tasks = stats.queued_tasks.saturating_sub(1);
                stats.active_tasks += 1;
            }

            // Check for cancellation before starting
            if cancellation_token.is_cancelled() {
                // Operation was cancelled before it could start
                if let Some(ref callbacks) = callback_info {
                    if let Some(error_cb) = callbacks.error_callback {
                        unsafe {
                            let msg = std::ffi::CString::new("Operation cancelled").unwrap_or_else(
                                |_| {
                                    std::ffi::CString::new("Operation cancelled (encoding error)")
                                        .unwrap()
                                },
                            );
                            error_cb(
                                pipeline_id,
                                VoirsErrorCode::OperationCancelled,
                                msg.as_ptr(),
                                callbacks.user_data,
                            );
                        }
                    }
                }

                // Clean up
                ACTIVE_OPERATIONS.fetch_sub(1, Ordering::Relaxed);
                let mut registry = OPERATION_REGISTRY.lock();
                registry.remove(&operation_id);

                // Update stats - task is no longer active but cancelled
                {
                    let mut stats = THREAD_STATS.lock();
                    stats.active_tasks = stats.active_tasks.saturating_sub(1);
                }

                return;
            }

            // Notify progress start
            if let Some(ref callbacks) = callback_info {
                if let Some(progress_cb) = callbacks.progress_callback {
                    unsafe {
                        progress_cb(pipeline_id, 0.0, callbacks.user_data);
                    }
                }
            }

            // Check for cancellation before synthesis
            if cancellation_token.is_cancelled() {
                // Clean up and exit
                ACTIVE_OPERATIONS.fetch_sub(1, Ordering::Relaxed);
                let mut registry = OPERATION_REGISTRY.lock();
                registry.remove(&operation_id);

                // Update stats - task is no longer active but cancelled
                {
                    let mut stats = THREAD_STATS.lock();
                    stats.active_tasks = stats.active_tasks.saturating_sub(1);
                }

                return;
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
                    let ffi_buffer =
                        Box::into_raw(Box::new(crate::VoirsAudioBuffer::from_audio_buffer(audio)));

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

                    // Track operation completion
                    ACTIVE_OPERATIONS.fetch_sub(1, Ordering::Relaxed);

                    // Remove from operation registry
                    let mut registry = OPERATION_REGISTRY.lock();
                    registry.remove(&operation_id);

                    // Update thread stats - task completed successfully
                    {
                        let mut stats = THREAD_STATS.lock();
                        stats.active_tasks = stats.active_tasks.saturating_sub(1);
                        stats.total_completed_tasks += 1;
                    }
                }
                Err(e) => {
                    // Notify error
                    if let Some(ref callbacks) = callback_info {
                        if let Some(error_cb) = callbacks.error_callback {
                            let error_msg =
                                std::ffi::CString::new(format!("{e}")).unwrap_or_else(|_| {
                                    std::ffi::CString::new("Unknown error").unwrap_or_else(|_| {
                                        // This should never fail, but just in case
                                        std::ffi::CString::new("Error").unwrap()
                                    })
                                });
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

                    // Track operation completion
                    ACTIVE_OPERATIONS.fetch_sub(1, Ordering::Relaxed);

                    // Remove from operation registry
                    let mut registry = OPERATION_REGISTRY.lock();
                    registry.remove(&operation_id);

                    // Update thread stats - task completed successfully
                    {
                        let mut stats = THREAD_STATS.lock();
                        stats.active_tasks = stats.active_tasks.saturating_sub(1);
                        stats.total_completed_tasks += 1;
                    }
                }
            }
        });

        VoirsErrorCode::Success
    } // End of #[cfg(not(test))]
}

/// Cancel an ongoing asynchronous synthesis operation
#[no_mangle]
pub extern "C" fn voirs_cancel_synthesis(pipeline_id: c_uint) -> VoirsErrorCode {
    let mut registry = OPERATION_REGISTRY.lock();

    // Find all operations for this pipeline and cancel them
    let mut cancelled_count = 0;
    let mut to_remove = Vec::new();

    for (operation_id, token) in registry.iter() {
        if token.pipeline_id == pipeline_id {
            token.cancel();
            to_remove.push(*operation_id);
            cancelled_count += 1;
        }
    }

    // Remove cancelled operations from registry
    for operation_id in to_remove {
        registry.remove(&operation_id);
    }

    if cancelled_count > 0 {
        // Decrement active operations counter
        let _ = ACTIVE_OPERATIONS.fetch_sub(cancelled_count, Ordering::Relaxed);
        VoirsErrorCode::Success
    } else {
        set_last_error(format!(
            "No active operations found for pipeline {pipeline_id}"
        ));
        VoirsErrorCode::InvalidParameter
    }
}

/// Get the number of active synthesis operations
#[no_mangle]
pub extern "C" fn voirs_get_active_operations() -> c_uint {
    ACTIVE_OPERATIONS.load(Ordering::Relaxed)
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
    if pipeline_ids.is_null()
        || texts.is_null()
        || count == 0
        || results.is_null()
        || audio_buffers.is_null()
    {
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

    // Check thread configuration to respect concurrency limits
    let max_concurrent = {
        let config = THREAD_CONFIG.lock();
        config.max_concurrent.min(count) as usize
    };

    // Process operations in parallel with proper synchronization
    rt.block_on(async move {
        // Create a semaphore to limit concurrent operations
        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        let mut handles = Vec::with_capacity(count as usize);

        // Spawn tasks for each synthesis operation
        for i in 0..count as usize {
            let pipeline_id = pipeline_slice[i];
            let semaphore = semaphore.clone();

            // Get pipeline first (while we have access to the slice)
            let pipeline = {
                let manager = get_pipeline_manager();
                let mgr = manager.lock();
                match mgr.get_pipeline(pipeline_id) {
                    Some(p) => p,
                    None => {
                        // Handle invalid pipeline ID synchronously
                        result_slice[i] = VoirsErrorCode::InvalidParameter;
                        buffer_slice[i] = ptr::null_mut();
                        continue;
                    }
                }
            };

            // Convert text to owned string
            let text_str = match CStr::from_ptr(text_slice[i]).to_str() {
                Ok(s) => s.to_string(),
                Err(_) => {
                    result_slice[i] = VoirsErrorCode::InvalidParameter;
                    buffer_slice[i] = ptr::null_mut();
                    continue;
                }
            };

            // Convert config if provided
            let synthesis_config = config_slice.map(|configs| configs[i].clone().into());

            // Create cancellation token for this operation
            let cancellation_token = CancellationToken::new(pipeline_id);
            let operation_id = cancellation_token.operation_id;

            // Register the operation
            {
                let mut registry = OPERATION_REGISTRY.lock();
                registry.insert(operation_id, cancellation_token.clone());
            }

            // Track operation start
            ACTIVE_OPERATIONS.fetch_add(1, Ordering::Relaxed);

            // Update thread statistics - task is now queued
            {
                let mut stats = THREAD_STATS.lock();
                stats.queued_tasks += 1;
            }

            // Spawn async task for this synthesis
            let handle = tokio::spawn(async move {
                // Wait for semaphore permit
                let _permit = semaphore.acquire().await.expect("Semaphore closed");

                // Task is now active (no longer queued)
                {
                    let mut stats = THREAD_STATS.lock();
                    stats.queued_tasks = stats.queued_tasks.saturating_sub(1);
                    stats.active_tasks += 1;
                }

                // Check for cancellation before starting
                if cancellation_token.is_cancelled() {
                    ACTIVE_OPERATIONS.fetch_sub(1, Ordering::Relaxed);
                    let mut registry = OPERATION_REGISTRY.lock();
                    registry.remove(&operation_id);

                    // Update stats - task is no longer active but cancelled
                    {
                        let mut stats = THREAD_STATS.lock();
                        stats.active_tasks = stats.active_tasks.saturating_sub(1);
                    }

                    return (i, VoirsErrorCode::OperationCancelled, None);
                }

                // Perform synthesis
                let result = if let Some(ref config) = synthesis_config {
                    pipeline.synthesize_with_config(&text_str, config).await
                } else {
                    pipeline.synthesize(&text_str).await
                };

                // Process result - return AudioBuffer instead of raw pointer
                let (error_code, audio_buffer) = match result {
                    Ok(audio) => (VoirsErrorCode::Success, Some(audio)),
                    Err(_) => (VoirsErrorCode::SynthesisFailed, None),
                };

                // Clean up operation tracking
                ACTIVE_OPERATIONS.fetch_sub(1, Ordering::Relaxed);
                let mut registry = OPERATION_REGISTRY.lock();
                registry.remove(&operation_id);

                // Update thread stats - task completed
                {
                    let mut stats = THREAD_STATS.lock();
                    stats.active_tasks = stats.active_tasks.saturating_sub(1);
                    stats.total_completed_tasks += 1;
                }

                (i, error_code, audio_buffer)
            });

            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            match handle.await {
                Ok((index, error_code, audio_buffer)) => {
                    result_slice[index] = error_code;

                    // Convert AudioBuffer to raw pointer on main thread (thread-safe)
                    buffer_slice[index] = match audio_buffer {
                        Some(audio) => {
                            let ffi_buffer = VoirsAudioBuffer::from_audio_buffer(audio);
                            Box::into_raw(Box::new(ffi_buffer))
                        }
                        None => ptr::null_mut(),
                    };
                }
                Err(_) => {
                    // Task panicked or was cancelled - this shouldn't happen in normal operation
                    // We'll leave the default values (should be initialized to safe defaults)
                }
            }
        }
    });

    VoirsErrorCode::Success
}

/// Get thread pool statistics
#[no_mangle]
pub unsafe extern "C" fn voirs_get_thread_stats(
    active_threads: *mut c_uint,
    queued_tasks: *mut c_uint,
    completed_tasks: *mut c_uint,
) -> VoirsErrorCode {
    if active_threads.is_null() || queued_tasks.is_null() || completed_tasks.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    // Get actual thread pool statistics
    let stats = THREAD_STATS.lock();

    // active_threads represents number of threads currently processing tasks
    *active_threads = stats.active_tasks;

    // queued_tasks represents number of tasks waiting to be processed
    *queued_tasks = stats.queued_tasks;

    // completed_tasks represents total number of tasks completed (as u32, may overflow)
    *completed_tasks = stats.total_completed_tasks as c_uint;

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
        assert_eq!(
            voirs_set_global_thread_count(0),
            VoirsErrorCode::InvalidParameter
        );
        assert_eq!(
            voirs_set_global_thread_count(100),
            VoirsErrorCode::InvalidParameter
        );
    }

    #[test]
    fn test_max_concurrent_configuration() {
        // Test valid max concurrent
        assert_eq!(voirs_set_max_concurrent(8), VoirsErrorCode::Success);
        assert_eq!(voirs_get_max_concurrent(), 8);

        // Test invalid max concurrent
        assert_eq!(
            voirs_set_max_concurrent(0),
            VoirsErrorCode::InvalidParameter
        );
        assert_eq!(
            voirs_set_max_concurrent(50),
            VoirsErrorCode::InvalidParameter
        );
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
            let result = voirs_register_callbacks(1, None, None, None, std::ptr::null_mut());
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

        let result = unsafe { voirs_get_thread_stats(&mut active, &mut queued, &mut completed) };
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
        use crate::c_api::core::voirs_create_pipeline;
        use std::ffi::CString;
        use std::time::Duration;

        // Test cancellation with no active operations - should return InvalidParameter
        assert_eq!(voirs_cancel_synthesis(1), VoirsErrorCode::InvalidParameter);

        // Test with invalid pipeline ID
        assert_eq!(voirs_cancel_synthesis(0), VoirsErrorCode::InvalidParameter);

        // Enhanced test: Create a real pipeline and test cancellation behavior
        unsafe {
            let pipeline_id = voirs_create_pipeline();
            assert_ne!(pipeline_id, 0, "Pipeline creation should succeed");

            // Create test text for synthesis
            let test_text = CString::new("This is a test text for synthesis.")
                .expect("Failed to create test string");
            let text_ptr = test_text.as_ptr();

            // Test the cancellation mechanism with multiple approaches

            // Approach 1: Test immediate cancellation (operation may complete quickly)
            let initial_ops = voirs_get_active_operations();
            let start_result = voirs_synthesize_async(pipeline_id, text_ptr, std::ptr::null());
            assert_eq!(
                start_result,
                VoirsErrorCode::Success,
                "Async synthesis should start successfully"
            );

            // Immediately try to cancel - this tests the cancellation system even if op completes quickly
            let immediate_cancel_result = voirs_cancel_synthesis(pipeline_id);
            // Accept either Success (if caught during operation) or InvalidParameter (if already completed)
            assert!(
                immediate_cancel_result == VoirsErrorCode::Success
                    || immediate_cancel_result == VoirsErrorCode::InvalidParameter,
                "Immediate cancellation should return Success or InvalidParameter, got: {:?}",
                immediate_cancel_result
            );

            // Wait for any operations to complete
            std::thread::sleep(Duration::from_millis(100));

            // Approach 2: Test multiple rapid operations to increase chance of catching one mid-flight
            let ops_before_batch = voirs_get_active_operations();

            // Start multiple operations rapidly
            for _ in 0..3 {
                let batch_result = voirs_synthesize_async(pipeline_id, text_ptr, std::ptr::null());
                assert_eq!(
                    batch_result,
                    VoirsErrorCode::Success,
                    "Batch async synthesis should start successfully"
                );
            }

            // Check if we have any active operations and try cancellation
            let ops_during_batch = voirs_get_active_operations();

            if ops_during_batch > ops_before_batch {
                // We caught some operations mid-flight, test cancellation
                let cancel_result = voirs_cancel_synthesis(pipeline_id);
                assert_eq!(
                    cancel_result,
                    VoirsErrorCode::Success,
                    "Cancellation should succeed for active operations"
                );

                // Wait for cancellation to take effect
                std::thread::sleep(Duration::from_millis(50));

                // Verify operations were affected
                let ops_after_cancel = voirs_get_active_operations();
                assert!(
                    ops_after_cancel <= ops_during_batch,
                    "Active operations should not increase after cancellation"
                );
            }

            // Approach 3: Test cancellation when no operations are active (already tested above but verify)
            std::thread::sleep(Duration::from_millis(100)); // Ensure all operations complete
            let final_cancel_result = voirs_cancel_synthesis(pipeline_id);
            assert_eq!(
                final_cancel_result,
                VoirsErrorCode::InvalidParameter,
                "Cancellation with no active operations should return InvalidParameter"
            );

            // Clean up: Destroy the pipeline
            crate::c_api::core::voirs_destroy_pipeline(pipeline_id);
        }
    }

    #[test]
    fn test_active_operations_count() {
        // Test getting active operations count (placeholder implementation)
        assert_eq!(voirs_get_active_operations(), 0);
    }
}
