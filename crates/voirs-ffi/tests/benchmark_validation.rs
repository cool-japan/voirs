//! Benchmark validation tests for VoiRS FFI
//!
//! These tests validate that performance benchmarks meet expected thresholds
//! and detect performance regressions.

use std::time::{Duration, Instant};
use voirs_ffi::c_api::config::*;
use voirs_ffi::c_api::core::*;

#[test]
fn test_pipeline_creation_performance() {
    // Enable benchmark mode for faster pipeline creation testing
    std::env::set_var("VOIRS_BENCHMARK_MODE", "1");

    // Test that pipeline creation meets performance threshold
    let iterations = 10;
    let start_time = Instant::now();

    for _ in 0..iterations {
        unsafe {
            let pipeline_id = voirs_create_pipeline();
            assert!(pipeline_id > 0, "Pipeline creation should succeed");

            let result = voirs_destroy_pipeline(pipeline_id);
            assert_eq!(result, 0, "Pipeline destruction should succeed");
        }
    }

    let duration = start_time.elapsed();
    let avg_duration = duration / iterations;

    println!(
        "Pipeline creation/destruction average time: {:?}",
        avg_duration
    );

    // Threshold: Pipeline creation should take less than 100ms on average
    // In benchmark mode, this should be much faster
    let threshold = Duration::from_millis(
        if std::env::var("VOIRS_BENCHMARK_MODE").unwrap_or_default() == "1" {
            10 // 10ms threshold for benchmark mode
        } else {
            100 // 100ms threshold for normal mode
        },
    );

    assert!(
        avg_duration < threshold,
        "Pipeline creation too slow: {:?} > {:?}",
        avg_duration,
        threshold
    );

    // Clean up environment variable
    std::env::remove_var("VOIRS_BENCHMARK_MODE");
}

#[test]
fn test_config_creation_performance() {
    // Test that config creation meets performance threshold
    let iterations = 1000;
    let start_time = Instant::now();

    for _ in 0..iterations {
        unsafe {
            let config = voirs_config_create_synthesis_default();
            // Basic validation that config was created properly
            assert!(config.speed > 0.0, "Config should have valid speed");
        }
    }

    let duration = start_time.elapsed();
    let avg_duration = duration / iterations;

    println!("Config creation average time: {:?}", avg_duration);

    // Threshold: Config creation should take less than 1ms on average
    assert!(
        avg_duration < Duration::from_millis(1),
        "Config creation too slow: {:?} > 1ms",
        avg_duration
    );
}

#[test]
fn test_pipeline_validation_performance() {
    // Test that pipeline validation meets performance threshold
    unsafe {
        let pipeline_id = voirs_create_pipeline();
        assert!(pipeline_id > 0, "Pipeline creation should succeed");

        let iterations = 10000;
        let start_time = Instant::now();

        for _ in 0..iterations {
            let is_valid = voirs_is_pipeline_valid(pipeline_id);
            assert_eq!(is_valid, 1, "Pipeline should be valid");
        }

        let duration = start_time.elapsed();
        let avg_duration = duration / iterations;

        println!("Pipeline validation average time: {:?}", avg_duration);

        // Threshold: Pipeline validation should take less than 1µs on average
        assert!(
            avg_duration < Duration::from_micros(1),
            "Pipeline validation too slow: {:?} > 1µs",
            avg_duration
        );

        // Cleanup
        let result = voirs_destroy_pipeline(pipeline_id);
        assert_eq!(result, 0, "Pipeline destruction should succeed");
    }
}

#[test]
fn test_error_handling_performance() {
    // Test that error handling doesn't significantly impact performance
    let iterations = 1000;

    // Test valid operations
    let start_time = Instant::now();
    for _ in 0..iterations {
        unsafe {
            let count = voirs_get_pipeline_count();
            let _ = count; // Use the value to prevent optimization
        }
    }
    let valid_duration = start_time.elapsed();

    // Test invalid operations (should trigger error handling)
    let start_time = Instant::now();
    for _ in 0..iterations {
        unsafe {
            let is_valid = voirs_is_pipeline_valid(99999); // Invalid ID
            assert_eq!(is_valid, 0, "Invalid pipeline should return 0");
        }
    }
    let invalid_duration = start_time.elapsed();

    let valid_avg = valid_duration / iterations;
    let invalid_avg = invalid_duration / iterations;

    println!("Valid operation average time: {:?}", valid_avg);
    println!("Invalid operation average time: {:?}", invalid_avg);

    // Error handling should not be more than 10x slower
    let max_allowed_ratio = 10;
    let actual_ratio = invalid_duration.as_nanos() / valid_duration.as_nanos().max(1);

    assert!(
        actual_ratio < max_allowed_ratio,
        "Error handling too slow: {}x slower than valid operations",
        actual_ratio
    );
}

#[test]
fn test_concurrent_access_performance() {
    // Test that concurrent access doesn't degrade performance excessively
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::thread;

    let operations_per_thread = 100;
    let thread_count = 4;
    let total_operations = operations_per_thread * thread_count;

    let success_count = Arc::new(AtomicU64::new(0));
    let start_time = Instant::now();

    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let success_count = Arc::clone(&success_count);
            thread::spawn(move || {
                for _ in 0..operations_per_thread {
                    unsafe {
                        let pipeline_id = voirs_create_pipeline();
                        if pipeline_id > 0 {
                            let config = voirs_config_create_synthesis_default();
                            if config.speed > 0.0 {
                                success_count.fetch_add(1, Ordering::Relaxed);
                            }
                            voirs_destroy_pipeline(pipeline_id);
                        }
                    }
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    let duration = start_time.elapsed();
    let successful_ops = success_count.load(Ordering::Relaxed);
    let throughput = successful_ops as f64 / duration.as_secs_f64();

    println!(
        "Concurrent access: {} operations in {:?}",
        successful_ops, duration
    );
    println!("Throughput: {:.2} operations/second", throughput);

    // Validation
    assert_eq!(
        successful_ops, total_operations,
        "All operations should succeed"
    );

    // Threshold: Should handle at least 10 operations per second under concurrent load
    assert!(
        throughput >= 10.0,
        "Concurrent throughput too low: {:.2} ops/sec < 10 ops/sec",
        throughput
    );
}

#[test]
fn test_memory_allocation_performance() {
    // Test that memory allocation patterns don't cause performance degradation
    let iterations = 100;
    let allocation_size = 1024 * 1024; // 1MB

    let mut total_alloc_time = Duration::new(0, 0);
    let mut total_dealloc_time = Duration::new(0, 0);

    for _ in 0..iterations {
        // Allocation timing
        let start = Instant::now();
        let buffer = vec![0u8; allocation_size];
        let alloc_time = start.elapsed();
        total_alloc_time += alloc_time;

        // Use the buffer to prevent optimization
        assert_eq!(buffer.len(), allocation_size);

        // Deallocation timing
        let start = Instant::now();
        drop(buffer);
        let dealloc_time = start.elapsed();
        total_dealloc_time += dealloc_time;
    }

    let avg_alloc_time = total_alloc_time / iterations;
    let avg_dealloc_time = total_dealloc_time / iterations;

    println!("Average allocation time (1MB): {:?}", avg_alloc_time);
    println!("Average deallocation time (1MB): {:?}", avg_dealloc_time);

    // Thresholds: 1MB allocation/deallocation should take less than 10ms
    assert!(
        avg_alloc_time < Duration::from_millis(10),
        "Memory allocation too slow: {:?} > 10ms",
        avg_alloc_time
    );
    assert!(
        avg_dealloc_time < Duration::from_millis(10),
        "Memory deallocation too slow: {:?} > 10ms",
        avg_dealloc_time
    );
}

#[cfg(test)]
mod performance_validation {
    use super::*;

    #[test]
    fn test_benchmark_infrastructure_validation() {
        // Quick validation that benchmark infrastructure is working
        let start = Instant::now();

        // Perform a simple operation
        unsafe {
            let pipeline_id = voirs_create_pipeline();
            assert!(pipeline_id > 0);
            voirs_destroy_pipeline(pipeline_id);
        }

        let duration = start.elapsed();

        // Should complete reasonably quickly
        assert!(
            duration < Duration::from_secs(1),
            "Basic operation took too long: {:?}",
            duration
        );

        println!(
            "Benchmark infrastructure validation completed in {:?}",
            duration
        );
    }
}
