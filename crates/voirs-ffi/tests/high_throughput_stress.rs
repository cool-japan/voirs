//! High-throughput stress tests for VoiRS FFI
//!
//! These tests validate the system's ability to handle high-volume
//! synthesis requests and burst loads typical of production environments.

use std::ffi::CString;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use voirs::c_api::config::*;
use voirs::c_api::core::*;
use voirs::c_api::synthesis::*;
use voirs::{VoirsErrorCode, VoirsSynthesisResult};

/// Serializes the tests in this file that mutate the process-wide
/// `VOIRS_BENCHMARK_MODE` environment variable.
///
/// `cargo nextest run` (this workspace's preferred/documented runner) gives
/// every `#[test]` its own process, making this a no-op there; plain `cargo
/// test` runs every `#[test]` in this file as threads within one process by
/// default, where an unguarded concurrent `set_var`/`remove_var` from two of
/// these tests would race (see `voirs-ffi/tests/pipeline_real_path.rs`'s
/// identical guard for the full rationale).
static TEST_SERIALIZATION: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serialize_test() -> std::sync::MutexGuard<'static, ()> {
    TEST_SERIALIZATION
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[test]
fn test_burst_synthesis_load() {
    let _guard = serialize_test();
    // This stress-tests pipeline-management throughput under concurrent
    // load (creation/validation/destruction plumbing), not real model
    // loading -- the comment below explicitly notes "we'll just test
    // pipeline creation/validation rather than full synthesis". Use
    // VOIRS_BENCHMARK_MODE so 50 concurrent pipeline creations don't each
    // require network access to fetch model weights.
    std::env::set_var("VOIRS_BENCHMARK_MODE", "1");

    // Test burst of 50 requests using C FFI directly
    let burst_size = 50;
    let max_duration = Duration::from_secs(30);

    let start_time = Instant::now();
    let success_count = Arc::new(AtomicU64::new(0));
    let error_count = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();

    for i in 0..burst_size {
        let success_count = Arc::clone(&success_count);
        let error_count = Arc::clone(&error_count);

        let handle = std::thread::spawn(move || {
            unsafe {
                // Create pipeline
                let pipeline = voirs_create_pipeline();
                if pipeline == 0 {
                    error_count.fetch_add(1, Ordering::Relaxed);
                    return;
                }

                // Create config
                let config = voirs_config_create_synthesis_default();
                // Note: voirs_config_create_synthesis_default() returns a struct, not a pointer

                // For the stress test, we'll just test pipeline creation/validation
                // rather than full synthesis which requires more complex setup

                // Test pipeline validation
                if voirs_is_pipeline_valid(pipeline) == 1 {
                    success_count.fetch_add(1, Ordering::Relaxed);
                } else {
                    error_count.fetch_add(1, Ordering::Relaxed);
                }

                // Cleanup (no need to destroy config since it's a struct, not a pointer)
                voirs_destroy_pipeline(pipeline);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    let duration = start_time.elapsed();
    let successful = success_count.load(Ordering::Relaxed);
    let errors = error_count.load(Ordering::Relaxed);

    println!(
        "Burst load test: {} successful, {} errors in {:?}",
        successful, errors, duration
    );

    // Requirements: At least 70% success rate (relaxed for C FFI), completed within time limit
    assert!(
        duration <= max_duration,
        "Burst test took too long: {:?}",
        duration
    );
    assert!(
        successful >= (burst_size * 70 / 100),
        "Success rate too low: {}/{}",
        successful,
        burst_size
    );

    // Throughput should be reasonable
    let throughput = successful as f64 / duration.as_secs_f64();
    assert!(
        throughput >= 1.0,
        "Throughput too low: {:.2} ops/sec",
        throughput
    );

    std::env::remove_var("VOIRS_BENCHMARK_MODE");
}

#[test]
fn test_basic_stress_infrastructure() {
    let _guard = serialize_test();
    // Validates stress-testing plumbing (pipeline creation/destruction
    // under concurrent load), not real model loading -- see
    // test_burst_synthesis_load's identical rationale for VOIRS_BENCHMARK_MODE.
    std::env::set_var("VOIRS_BENCHMARK_MODE", "1");

    // Basic test to validate stress testing infrastructure is working
    let test_count = 10;
    let success_count = Arc::new(AtomicU64::new(0));
    let error_count = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();

    for i in 0..test_count {
        let success_count = Arc::clone(&success_count);
        let error_count = Arc::clone(&error_count);

        let handle = std::thread::spawn(move || {
            unsafe {
                // Test basic FFI pipeline creation and destruction
                let pipeline = voirs_create_pipeline();
                if pipeline != 0 {
                    let _config = voirs_config_create_synthesis_default();
                    // Config is a struct, so it's always valid
                    success_count.fetch_add(1, Ordering::Relaxed);
                    voirs_destroy_pipeline(pipeline);
                } else {
                    error_count.fetch_add(1, Ordering::Relaxed);
                }
            }

            // Small delay to simulate work
            std::thread::sleep(Duration::from_millis(10));
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    let successful = success_count.load(Ordering::Relaxed);
    let errors = error_count.load(Ordering::Relaxed);

    println!(
        "Stress infrastructure test: {} successful, {} errors",
        successful, errors
    );

    // All basic operations should succeed
    assert_eq!(
        successful, test_count,
        "Basic stress infrastructure should work"
    );
    assert_eq!(errors, 0, "No errors expected in basic infrastructure test");

    std::env::remove_var("VOIRS_BENCHMARK_MODE");
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_stress_test_runner() {
        let _guard = serialize_test();
        // Integration test that runs multiple stress scenarios. Validates
        // FFI plumbing, not real model loading -- see
        // test_burst_synthesis_load's identical rationale for
        // VOIRS_BENCHMARK_MODE.
        std::env::set_var("VOIRS_BENCHMARK_MODE", "1");

        unsafe {
            // Quick validation that the FFI infrastructure works
            let pipeline = voirs_create_pipeline();
            if pipeline != 0 {
                let _config = voirs_config_create_synthesis_default();
                // Config is a struct, so no need to destroy it
                voirs_destroy_pipeline(pipeline);
                println!("All stress test infrastructure validated successfully");
            } else {
                std::env::remove_var("VOIRS_BENCHMARK_MODE");
                panic!("Failed to create pipeline for stress test validation");
            }
        }

        std::env::remove_var("VOIRS_BENCHMARK_MODE");
    }
}
