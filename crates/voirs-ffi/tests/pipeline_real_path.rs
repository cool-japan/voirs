//! Integration tests that exercise the REAL (non-mock) pipeline C API.
//!
//! `c_api::core`/`c_api::voice`/`c_api::threading`'s pipeline-management
//! functions previously carried `#[cfg(test)]` mock branches that
//! unconditionally won during `cargo test`/`cargo nextest run`, so this
//! crate's own unit test suite could pass 100% while never once touching the
//! real `PIPELINE_MANAGER`/`get_runtime()`/`SdkPipeline` code a real C/Python/
//! Node caller actually hits. The mock branches now live behind the
//! off-by-default `ffi-test-mocks` Cargo feature (see `Cargo.toml`), so this
//! file -- which is *not* gated behind that feature -- exercises the real
//! production path whenever it runs without `ffi-test-mocks` enabled (i.e.
//! plain `cargo test`/`cargo nextest run -p voirs-ffi`, without
//! `--all-features`).
//!
//! This whole file is disabled when `ffi-test-mocks` IS enabled (e.g. via
//! `--all-features`): with that feature on, `voirs_create_pipeline()` et al.
//! switch to the mock bookkeeping (see `c_api::core`'s own `mod tests`),
//! under which the benchmark-placeholder-specific assertions below (which
//! depend on the real `PipelineManager::add_placeholder_pipeline`) would not
//! hold -- the mock path has no concept of a "placeholder" pipeline.
#![cfg(not(feature = "ffi-test-mocks"))]

use std::ffi::CString;
use voirs::c_api::core::*;
use voirs::c_api::voice::*;
use voirs::{voirs_clear_error, voirs_get_last_error, voirs_has_error};

/// Read and clear the current thread's last-error message, if any.
fn take_last_error() -> Option<String> {
    unsafe {
        let ptr = voirs_get_last_error();
        if ptr.is_null() {
            return None;
        }
        let message = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
        voirs::voirs_free_string(ptr);
        Some(message)
    }
}

/// `voirs_create_pipeline()`'s real (non-benchmark-mode) contract, without
/// assuming network/model-weight availability in the test environment:
/// either it honestly succeeds (and then behaves like a real pipeline
/// through the rest of its lifecycle), or it honestly fails with a
/// diagnostic message -- never a handle that looks valid but silently does
/// nothing.
#[test]
fn test_real_pipeline_creation_contract() {
    unsafe {
        voirs_clear_error();
        let pipeline_id = voirs_create_pipeline();

        if pipeline_id == 0 {
            // Honest failure: a diagnostic must be available explaining why
            // (e.g. no network access to fetch model weights in this
            // environment), never a silent/unexplained 0.
            assert_ne!(voirs_has_error(), 0, "a failed creation must set an error");
            let message = take_last_error();
            assert!(
                message.is_some() && !message.as_deref().unwrap_or_default().is_empty(),
                "voirs_get_last_error() must be non-empty after a failed creation"
            );
            assert_eq!(voirs_is_pipeline_valid(0), 0);
            return;
        }

        // Honest success: the handle must behave like a real, trackable
        // pipeline through its whole lifecycle.
        assert_eq!(
            voirs_is_pipeline_valid(pipeline_id),
            1,
            "a successfully created pipeline must be valid"
        );
        assert_eq!(
            voirs_is_pipeline_benchmark_placeholder(pipeline_id),
            0,
            "a normally-created (non-VOIRS_BENCHMARK_MODE) pipeline must not \
             be reported as a benchmark placeholder"
        );
        assert!(voirs_get_pipeline_count() >= 1);

        assert_eq!(
            voirs_destroy_pipeline(pipeline_id),
            0,
            "destroying a just-created pipeline must succeed"
        );
        assert_eq!(
            voirs_is_pipeline_valid(pipeline_id),
            0,
            "a destroyed pipeline must become invalid"
        );
        // Destroying it again must honestly fail, not silently "succeed".
        assert_ne!(voirs_destroy_pipeline(pipeline_id), 0);
    }
}

/// A never-issued pipeline ID must be honestly rejected by `voirs_set_voice`/
/// `voirs_get_voice` through the REAL (not mock) `get_pipeline_manager()`
/// lookup, with a generic (non-placeholder) diagnostic message.
#[test]
fn test_real_voice_operations_reject_unknown_pipeline() {
    let never_issued_id = 0xFFFF_FF00u32;

    unsafe {
        voirs_clear_error();
        let voice_id = CString::new("default").expect("no interior NUL");
        let result = voirs_set_voice(never_issued_id, voice_id.as_ptr());
        assert_ne!(result, 0, "setting voice on an unknown pipeline must fail");
        let message = take_last_error().unwrap_or_default();
        assert!(
            message.contains(&never_issued_id.to_string()),
            "error message should reference the offending pipeline ID: {message}"
        );
        assert!(
            !message.contains("VOIRS_BENCHMARK_MODE"),
            "a never-issued ID is not a benchmark placeholder, so the message \
             must be the generic invalid-ID message, not the placeholder-specific \
             one: {message}"
        );

        let voice_ptr = voirs_get_voice(never_issued_id);
        assert!(voice_ptr.is_null());
    }
}

/// End-to-end test of `VOIRS_BENCHMARK_MODE=1` through the REAL production
/// `create_pipeline_impl` -> `PipelineManager::add_placeholder_pipeline`
/// path. Deterministic and network-free by design (that is the entire point
/// of benchmark mode), unlike `test_real_pipeline_creation_contract` above.
///
/// This is the regression test for the "fabricated handle" finding: a
/// placeholder handle must pass `voirs_is_pipeline_valid()` (by design, for
/// measuring pipeline-management overhead) AND be honestly distinguishable
/// via `voirs_is_pipeline_benchmark_placeholder()` AND honestly refuse real
/// voice/synthesis operations with a specific, discoverable diagnostic
/// message -- not a generic "invalid pipeline ID" indistinguishable from a
/// typo'd handle.
#[test]
fn test_benchmark_mode_placeholder_is_honestly_distinct() {
    std::env::set_var("VOIRS_BENCHMARK_MODE", "1");

    unsafe {
        let pipeline_id = voirs_create_pipeline();
        assert_ne!(
            pipeline_id, 0,
            "VOIRS_BENCHMARK_MODE creation must deterministically succeed \
             (no model loading involved)"
        );

        // Passes validity checks (by design -- it's a real, trackable handle)...
        assert_eq!(voirs_is_pipeline_valid(pipeline_id), 1);
        // ...but is honestly distinguishable as a placeholder...
        assert_eq!(
            voirs_is_pipeline_benchmark_placeholder(pipeline_id),
            1,
            "a VOIRS_BENCHMARK_MODE handle must be reported as a placeholder"
        );
        assert!(voirs_get_pipeline_count() >= 1);

        // ...and honestly refuses real voice operations with a SPECIFIC
        // diagnostic, not a fabricated success and not an indistinguishable
        // generic error.
        voirs_clear_error();
        let voice_id = CString::new("default").expect("no interior NUL");
        let result = voirs_set_voice(pipeline_id, voice_id.as_ptr());
        assert_ne!(
            result, 0,
            "a benchmark placeholder must not silently succeed at voice operations"
        );
        let message = take_last_error().unwrap_or_default();
        assert!(
            message.contains("VOIRS_BENCHMARK_MODE") || message.contains("placeholder"),
            "error message must specifically explain this is a benchmark \
             placeholder, not a generic invalid-ID message: {message}"
        );

        assert_eq!(voirs_destroy_pipeline(pipeline_id), 0);
        assert_eq!(voirs_is_pipeline_valid(pipeline_id), 0);
    }

    std::env::remove_var("VOIRS_BENCHMARK_MODE");
}

/// `voirs_get_pipeline_count()` must reflect the REAL `PipelineManager`'s
/// bookkeeping across multiple creations/destructions, not a hardcoded
/// value (the mock branch, by contrast, always returns `1`).
#[test]
fn test_real_pipeline_count_tracks_creation_and_destruction() {
    std::env::set_var("VOIRS_BENCHMARK_MODE", "1");

    unsafe {
        let before = voirs_get_pipeline_count();

        let id1 = voirs_create_pipeline();
        assert_ne!(id1, 0);
        let after_one = voirs_get_pipeline_count();
        assert_eq!(after_one, before + 1);

        let id2 = voirs_create_pipeline();
        assert_ne!(id2, 0);
        assert_ne!(id1, id2, "each pipeline must get a distinct ID");
        let after_two = voirs_get_pipeline_count();
        assert_eq!(after_two, before + 2);

        assert_eq!(voirs_destroy_pipeline(id1), 0);
        assert_eq!(voirs_get_pipeline_count(), before + 1);

        assert_eq!(voirs_destroy_pipeline(id2), 0);
        assert_eq!(voirs_get_pipeline_count(), before);
    }

    std::env::remove_var("VOIRS_BENCHMARK_MODE");
}
