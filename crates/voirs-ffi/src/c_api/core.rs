//! Core C API functions for pipeline management.
//!
//! This module provides the fundamental pipeline creation, destruction,
//! and management functions for the VoiRS FFI C API.

use crate::{get_pipeline_manager, get_runtime, set_last_error, VoirsErrorCode};
use std::os::raw::{c_char, c_int, c_uint};
use std::sync::Arc;
use voirs_sdk::VoirsPipeline;

#[cfg(feature = "ffi-test-mocks")]
use once_cell::sync::Lazy;
#[cfg(feature = "ffi-test-mocks")]
use std::collections::HashSet;
#[cfg(feature = "ffi-test-mocks")]
use std::sync::Mutex;

// Mock pipeline-ID bookkeeping, gated behind the off-by-default
// `ffi-test-mocks` Cargo feature (NOT plain `#[cfg(test)]`): previously these
// mock branches were selected by `#[cfg(test)]`/`#[cfg(not(test))]`, which
// unconditionally wins during `cargo test`/`cargo nextest run` -- meaning
// this crate's own unit test suite could *never* exercise the real
// `#[cfg(not(feature = "ffi-test-mocks"))]` production code below (touching
// `PIPELINE_MANAGER`, `get_runtime()`, and the real `SdkPipeline`), no
// matter how many tests passed. With the feature off by default, plain
// `cargo test`/`cargo nextest run -p voirs-ffi` now exercises the real path;
// `--features ffi-test-mocks` (or `--all-features`) opts back into the fast,
// network-free mock bookkeeping exercised by this module's `mod tests`.
#[cfg(feature = "ffi-test-mocks")]
pub static CREATED_PIPELINES: Lazy<Mutex<HashSet<u32>>> = Lazy::new(|| Mutex::new(HashSet::new()));
#[cfg(feature = "ffi-test-mocks")]
pub static DESTROYED_PIPELINES: Lazy<Mutex<HashSet<u32>>> =
    Lazy::new(|| Mutex::new(HashSet::new()));

/// Create a new VoiRS pipeline instance
///
/// Returns a pipeline ID on success, or 0 on failure.
/// Check `voirs_get_last_error()` for error details.
///
/// ## `VOIRS_BENCHMARK_MODE`
///
/// If the environment variable `VOIRS_BENCHMARK_MODE=1` is set, this function
/// skips real model loading and returns a lightweight **placeholder** handle
/// instead, so pipeline-management overhead (ID allocation, table lookups,
/// creation/destruction latency) can be measured in isolation without paying
/// the cost of loading acoustic/vocoder model weights. A placeholder handle:
/// - passes `voirs_is_pipeline_valid()` and is counted by
///   `voirs_get_pipeline_count()` (it is a real, trackable handle),
/// - can be checked with `voirs_is_pipeline_benchmark_placeholder()`,
/// - CANNOT perform real synthesis or voice operations: any call to
///   `voirs_synthesize`/`voirs_synthesize_async`/`voirs_synthesize_parallel`/
///   `voirs_set_voice`/`voirs_get_voice` with a placeholder handle fails with
///   `VOIRS_ERROR_INVALID_PARAMETER`, and `voirs_get_last_error()` reports
///   that the handle is a benchmark placeholder rather than a generic
///   "invalid pipeline ID".
///
/// This is intended for this crate's own benchmark suite; most callers
/// should never set `VOIRS_BENCHMARK_MODE`.
#[no_mangle]
pub extern "C" fn voirs_create_pipeline() -> c_uint {
    // Install the pure-Rust rustls CryptoProvider before any TLS handshake
    // (reqwest is built with `rustls-no-provider`). Once-guarded; safe to repeat.
    voirs_acoustic::hub::ensure_crypto_provider();
    match create_pipeline_impl() {
        Ok(id) => id,
        Err(code) => {
            set_last_error(format!("Failed to create pipeline: {code:?}"));
            0
        }
    }
}

/// Create a new VoiRS pipeline instance with configuration
///
/// # Arguments
/// * `config_json` - JSON configuration string (null-terminated)
///
/// Returns a pipeline ID on success, or 0 on failure.
/// Check `voirs_get_last_error()` for error details.
#[no_mangle]
pub extern "C" fn voirs_create_pipeline_with_config(config_json: *const c_char) -> c_uint {
    // Install the pure-Rust rustls CryptoProvider before any TLS handshake
    // (reqwest is built with `rustls-no-provider`). Once-guarded; safe to repeat.
    voirs_acoustic::hub::ensure_crypto_provider();
    match create_pipeline_with_config_impl(config_json) {
        Ok(id) => id,
        Err(code) => {
            set_last_error(format!("Failed to create pipeline with config: {code:?}"));
            0
        }
    }
}

/// Destroy a VoiRS pipeline instance
///
/// # Arguments
/// * `pipeline_id` - Pipeline ID returned by `voirs_create_pipeline()`
///
/// Returns 0 on success, or error code on failure.
#[no_mangle]
pub extern "C" fn voirs_destroy_pipeline(pipeline_id: c_uint) -> c_int {
    match destroy_pipeline_impl(pipeline_id) {
        Ok(()) => 0,
        Err(code) => {
            set_last_error(format!(
                "Failed to destroy pipeline {pipeline_id}: {code:?}"
            ));
            code as c_int
        }
    }
}

/// Get the number of active pipeline instances
///
/// Returns the count of active pipelines.
#[no_mangle]
pub extern "C" fn voirs_get_pipeline_count() -> c_uint {
    #[cfg(feature = "ffi-test-mocks")]
    {
        // Mock mode: return a dummy count
        1
    }

    #[cfg(not(feature = "ffi-test-mocks"))]
    {
        let manager = get_pipeline_manager();
        let guard = manager.lock();
        guard.count() as c_uint
    }
}

/// Check if a pipeline ID is valid
///
/// # Arguments
/// * `pipeline_id` - Pipeline ID to check
///
/// Returns 1 if valid, 0 if invalid.
#[no_mangle]
pub extern "C" fn voirs_is_pipeline_valid(pipeline_id: c_uint) -> c_int {
    #[cfg(feature = "ffi-test-mocks")]
    {
        // Mock mode: check if pipeline ID is non-zero, was created, and not destroyed
        if pipeline_id == 0 {
            return 0;
        }

        let created = CREATED_PIPELINES
            .lock()
            .expect("lock should not be poisoned");
        if !created.contains(&pipeline_id) {
            return 0;
        }

        let destroyed = DESTROYED_PIPELINES
            .lock()
            .expect("lock should not be poisoned");
        if destroyed.contains(&pipeline_id) {
            0
        } else {
            1
        }
    }

    #[cfg(not(feature = "ffi-test-mocks"))]
    {
        let manager = get_pipeline_manager();
        let guard = manager.lock();
        if guard.is_valid_pipeline(pipeline_id) {
            1
        } else {
            0
        }
    }
}

/// Check whether a pipeline ID refers to a `VOIRS_BENCHMARK_MODE` placeholder
/// handle rather than a real, synthesis-capable pipeline.
///
/// A placeholder handle passes `voirs_is_pipeline_valid()` (it is a real,
/// trackable handle for measuring pipeline-management overhead), but backs no
/// loaded model and will fail any real synthesis/voice call with
/// `VOIRS_ERROR_INVALID_PARAMETER`. See `voirs_create_pipeline()`'s
/// documentation for the full `VOIRS_BENCHMARK_MODE` contract.
///
/// # Arguments
/// * `pipeline_id` - Pipeline ID to check
///
/// Returns 1 if `pipeline_id` is a benchmark placeholder, 0 if it is a real
/// pipeline or an unknown/invalid ID.
#[no_mangle]
pub extern "C" fn voirs_is_pipeline_benchmark_placeholder(pipeline_id: c_uint) -> c_int {
    #[cfg(feature = "ffi-test-mocks")]
    {
        // Mock-mode pipelines (see CREATED_PIPELINES) are never placeholders;
        // benchmark-mode handle behavior is covered by the production
        // (#[cfg(not(feature = "ffi-test-mocks"))]) path via voirs-ffi/tests/
        // integration tests.
        let _ = pipeline_id;
        0
    }

    #[cfg(not(feature = "ffi-test-mocks"))]
    {
        let manager = get_pipeline_manager();
        let guard = manager.lock();
        if guard.is_placeholder(pipeline_id) {
            1
        } else {
            0
        }
    }
}

// Implementation functions

fn create_pipeline_impl() -> Result<c_uint, VoirsErrorCode> {
    #[cfg(feature = "ffi-test-mocks")]
    {
        // Mock mode: create mock pipeline IDs for testing purposes. An
        // AtomicU32 (not `static mut`) so this stays sound even though it's
        // now reachable from any caller with the feature enabled, not just
        // the single-threaded assumptions of the old test-only code path.
        static NEXT_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Track created pipeline IDs for test validation
        let mut created = CREATED_PIPELINES
            .lock()
            .expect("lock should not be poisoned");
        created.insert(id);

        Ok(id)
    }

    #[cfg(not(feature = "ffi-test-mocks"))]
    {
        // Use lazy loading for better performance
        // Check if we're in benchmark or performance testing mode
        let is_benchmark_mode = std::env::var("VOIRS_BENCHMARK_MODE").unwrap_or_default() == "1";

        if is_benchmark_mode {
            // For benchmarks, create a lightweight mock pipeline without actual model loading
            let manager = get_pipeline_manager();
            let mut guard = manager.lock();
            let id = guard.add_placeholder_pipeline();
            return Ok(id);
        }

        let runtime = get_runtime()?;

        let pipeline = runtime
            .block_on(async {
                use std::env;

                // Create a temporary cache directory
                let temp_dir = env::temp_dir().join("voirs-ffi");
                let _ = std::fs::create_dir_all(&temp_dir);

                // Use faster initialization for repeated calls
                VoirsPipeline::builder()
                    .with_device("cpu".to_string())
                    .with_gpu_acceleration(false)
                    .with_quality(voirs_sdk::types::QualityLevel::Low)
                    .with_cache_dir(temp_dir)
                    .build()
                    .await
            })
            .map_err(|e| {
                set_last_error(format!("Pipeline creation failed: {e}"));
                VoirsErrorCode::InitializationFailed
            })?;

        let manager = get_pipeline_manager();
        let mut guard = manager.lock();
        let id = guard.add_pipeline(pipeline);
        Ok(id)
    }
}

fn create_pipeline_with_config_impl(config_json: *const c_char) -> Result<c_uint, VoirsErrorCode> {
    if config_json.is_null() {
        return Err(VoirsErrorCode::InvalidParameter);
    }

    let _config_str = unsafe {
        match std::ffi::CStr::from_ptr(config_json).to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("Invalid UTF-8 in config: {e}"));
                return Err(VoirsErrorCode::InvalidParameter);
            }
        }
    };

    #[cfg(feature = "ffi-test-mocks")]
    {
        // Mock mode: create mock pipeline IDs for testing purposes.
        static NEXT_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1000);
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Track created pipeline IDs for test validation
        let mut created = CREATED_PIPELINES
            .lock()
            .expect("lock should not be poisoned");
        created.insert(id);

        Ok(id)
    }

    #[cfg(not(feature = "ffi-test-mocks"))]
    {
        let runtime = get_runtime()?;

        let pipeline = runtime
            .block_on(async {
                // Parse the JSON config and create pipeline with it
                // For now, we'll create a basic pipeline and ignore the config
                // In a full implementation, this would parse the JSON and configure the pipeline
                use std::env;

                // Create a temporary cache directory
                let temp_dir = env::temp_dir().join("voirs-ffi");
                let _ = std::fs::create_dir_all(&temp_dir);

                VoirsPipeline::builder()
                    .with_device("cpu".to_string())
                    .with_gpu_acceleration(false)
                    .with_quality(voirs_sdk::types::QualityLevel::Low)
                    .with_cache_dir(temp_dir)
                    .build()
                    .await
            })
            .map_err(|e| {
                set_last_error(format!("Pipeline creation with config failed: {e}"));
                VoirsErrorCode::InitializationFailed
            })?;

        let manager = get_pipeline_manager();
        let mut guard = manager.lock();
        let id = guard.add_pipeline(pipeline);
        Ok(id)
    }
}

fn destroy_pipeline_impl(pipeline_id: c_uint) -> Result<(), VoirsErrorCode> {
    if pipeline_id == 0 {
        return Err(VoirsErrorCode::InvalidParameter);
    }

    #[cfg(feature = "ffi-test-mocks")]
    {
        // Mock mode: validate pipeline lifecycle for testing purposes
        let created = CREATED_PIPELINES
            .lock()
            .expect("lock should not be poisoned");
        if !created.contains(&pipeline_id) {
            return Err(VoirsErrorCode::InvalidParameter);
        }

        // Check if already destroyed
        let mut destroyed = DESTROYED_PIPELINES
            .lock()
            .expect("lock should not be poisoned");
        if destroyed.contains(&pipeline_id) {
            return Err(VoirsErrorCode::InvalidParameter);
        }

        // Mark as destroyed for test validation
        destroyed.insert(pipeline_id);
        Ok(())
    }

    #[cfg(not(feature = "ffi-test-mocks"))]
    {
        let manager = get_pipeline_manager();
        let mut guard = manager.lock();
        if guard.remove_pipeline(pipeline_id) {
            Ok(())
        } else {
            Err(VoirsErrorCode::InvalidParameter)
        }
    }
}

// Gated on the `ffi-test-mocks` feature (in addition to `#[cfg(test)]`):
// every test below assumes the fast, network-free mock pipeline bookkeeping
// above (instant success, no real model loading). With `ffi-test-mocks` off
// (the default), `voirs_create_pipeline()` et al. hit the *real* production
// path instead, which these specific assertions don't hold for (e.g. it may
// legitimately return 0 without network access to fetch model weights). The
// real path is exercised by `voirs-ffi/tests/pipeline_real_path.rs`, which
// asserts the honest *contract* instead of assuming success.
#[cfg(all(test, feature = "ffi-test-mocks"))]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation_and_destruction() {
        // Test basic pipeline creation
        let pipeline_id = voirs_create_pipeline();
        if pipeline_id == 0 {
            let error_msg = unsafe {
                let c_str = crate::voirs_get_last_error();
                if !c_str.is_null() {
                    std::ffi::CStr::from_ptr(c_str)
                        .to_string_lossy()
                        .into_owned()
                } else {
                    "No error message available".to_string()
                }
            };
            panic!("Pipeline creation failed: {}", error_msg);
        }
        assert_ne!(pipeline_id, 0, "Pipeline creation should succeed");

        // Test pipeline validation
        assert_eq!(
            voirs_is_pipeline_valid(pipeline_id),
            1,
            "Pipeline should be valid"
        );
        assert_eq!(
            voirs_is_pipeline_valid(0),
            0,
            "Invalid pipeline ID should return 0"
        );

        // Test pipeline count
        let count = voirs_get_pipeline_count();
        assert!(count > 0, "Pipeline count should be greater than 0");

        // Test pipeline destruction
        let result = voirs_destroy_pipeline(pipeline_id);
        assert_eq!(result, 0, "Pipeline destruction should succeed");

        // Test pipeline is no longer valid
        assert_eq!(
            voirs_is_pipeline_valid(pipeline_id),
            0,
            "Pipeline should be invalid after destruction"
        );
    }

    #[test]
    fn test_invalid_pipeline_operations() {
        // Test destroying invalid pipeline
        let result = voirs_destroy_pipeline(0);
        assert_ne!(result, 0, "Destroying invalid pipeline should fail");

        let result = voirs_destroy_pipeline(999999);
        assert_ne!(result, 0, "Destroying non-existent pipeline should fail");
    }

    #[test]
    fn test_pipeline_with_config() {
        let config = std::ffi::CString::new(r#"{"quality": "high"}"#).unwrap();
        let pipeline_id = voirs_create_pipeline_with_config(config.as_ptr());
        assert_ne!(
            pipeline_id, 0,
            "Pipeline creation with config should succeed"
        );

        // Clean up
        let result = voirs_destroy_pipeline(pipeline_id);
        assert_eq!(result, 0, "Pipeline destruction should succeed");
    }

    #[test]
    fn test_pipeline_with_invalid_config() {
        // Test with null config
        let pipeline_id = voirs_create_pipeline_with_config(std::ptr::null());
        assert_eq!(
            pipeline_id, 0,
            "Pipeline creation with null config should fail"
        );
    }

    #[test]
    fn test_benchmark_placeholder_check_rejects_unknown_and_zero_ids() {
        // An unknown/never-issued ID and the reserved 0 ID are never
        // placeholders (nor valid pipelines at all).
        assert_eq!(voirs_is_pipeline_benchmark_placeholder(0), 0);
        assert_eq!(voirs_is_pipeline_benchmark_placeholder(999_999), 0);

        // A real (non-placeholder) pipeline must not be reported as a
        // placeholder either.
        let pipeline_id = voirs_create_pipeline();
        assert_ne!(pipeline_id, 0, "Pipeline creation should succeed");
        assert_eq!(
            voirs_is_pipeline_benchmark_placeholder(pipeline_id),
            0,
            "a normally-created pipeline must not be reported as a benchmark placeholder"
        );
        assert_eq!(voirs_destroy_pipeline(pipeline_id), 0);

        // The VOIRS_BENCHMARK_MODE=1 path itself (`add_placeholder_pipeline`)
        // is exercised end-to-end by the real (#[cfg(not(test))]) production
        // code path in the `tests/` integration suite -- this crate's own
        // `#[cfg(test)]` mock pipeline manager never creates placeholders, so
        // it cannot be exercised from a unit test in this module.
    }
}
