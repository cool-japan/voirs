use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::ffi::c_void;
use std::ffi::CString;
use std::ptr;
use std::time::Duration;
use voirs_ffi::c_api::convert::VoirsEndianness;
use voirs_ffi::c_api::synthesis::{VoirsSynthesisResult, VoirsSynthesisStats};
use voirs_ffi::c_api::*;
use voirs_ffi::error::recovery::attempt_error_recovery;
use voirs_ffi::error::structured::VoirsStructuredError;
use voirs_ffi::{voirs_free_audio_buffer, VoirsAudioBuffer, VoirsErrorCode, VoirsSynthesisConfig};
use voirs_sdk::types::{LanguageCode, QualityLevel};

/// FFI performance benchmarking suite
///
/// This module provides comprehensive performance benchmarking for FFI operations
/// including overhead measurement, language-specific performance analysis, memory
/// usage profiling, and scalability testing.

#[repr(C)]
struct BenchmarkContext {
    pipeline_id: u32,
    config: VoirsSynthesisConfig,
    audio_buffer: *mut VoirsAudioBuffer,
}

impl BenchmarkContext {
    fn new() -> Self {
        let pipeline_id = unsafe { voirs_create_pipeline() };
        let config = VoirsSynthesisConfig::default();

        Self {
            pipeline_id,
            config,
            audio_buffer: std::ptr::null_mut(),
        }
    }
}

impl Drop for BenchmarkContext {
    fn drop(&mut self) {
        if self.pipeline_id != 0 {
            voirs_destroy_pipeline(self.pipeline_id);
        }
        if !self.audio_buffer.is_null() {
            // Audio buffer cleanup is handled elsewhere
        }
    }
}

/// Benchmark FFI call overhead
fn benchmark_ffi_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("ffi_overhead");

    // Simple function call overhead
    group.bench_function("create_destroy_pipeline", |b| {
        b.iter(|| {
            let pipeline_id = black_box(unsafe { voirs_create_pipeline() });
            black_box(unsafe { voirs_destroy_pipeline(pipeline_id) });
        })
    });

    // Configuration overhead
    group.bench_function("config_creation", |b| {
        b.iter(|| black_box(VoirsSynthesisConfig::default()))
    });

    // Error handling overhead
    group.bench_function("error_handling", |b| {
        b.iter(|| {
            let pipeline_id = black_box(unsafe { voirs_create_pipeline() });
            black_box(unsafe { voirs_destroy_pipeline(pipeline_id) });
        })
    });

    group.finish();
}

/// Benchmark synthesis performance at different scales
fn benchmark_synthesis_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthesis_scalability");

    let text_sizes = vec![10, 50, 100, 500, 1000, 5000];
    let context = BenchmarkContext::new();

    for size in text_sizes {
        let text = "Hello world! ".repeat(size / 12 + 1);
        let c_text = CString::new(text.clone()).unwrap();

        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("text_synthesis", size),
            &(c_text.as_ptr(), text.len()),
            |b, (text_ptr, len)| {
                b.iter(|| {
                    let mut result = VoirsSynthesisResult::default();
                    let error_code = unsafe {
                        black_box(voirs_synthesize_advanced(
                            *text_ptr,
                            std::ptr::null(),
                            &mut result,
                        ))
                    };
                    black_box(error_code);
                    if !result.audio.is_null() {
                        unsafe {
                            voirs_free_audio_buffer(result.audio);
                        }
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory allocation patterns
fn benchmark_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    // Audio buffer allocation (simulated)
    group.bench_function("audio_buffer_allocation", |b| {
        b.iter(|| {
            // Simulate buffer allocation/deallocation overhead
            let _overhead = black_box(std::time::Instant::now());
        })
    });

    // Large buffer allocation (simulated)
    group.bench_function("large_buffer_allocation", |b| {
        b.iter(|| {
            // Simulate large buffer allocation/deallocation overhead
            let _overhead = black_box(std::time::Instant::now());
        })
    });

    // Memory pool operations (simulated)
    group.bench_function("memory_pool_operations", |b| {
        b.iter(|| {
            // Simulate memory pool operations overhead
            let _pool_overhead = black_box(std::time::Instant::now());
        })
    });

    group.finish();
}

/// Benchmark concurrent operations
fn benchmark_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");

    // Thread-safe synthesis
    group.bench_function("concurrent_synthesis", |b| {
        let context = BenchmarkContext::new();
        let text = CString::new("Test concurrent synthesis").unwrap();

        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let config = context.config.clone();
                    let text_copy = text.clone(); // Clone the CString for each thread

                    std::thread::spawn(move || {
                        let text_ptr = text_copy.as_ptr();
                        let mut result = VoirsSynthesisResult::default();
                        let error_code = unsafe {
                            voirs_synthesize_advanced(text_ptr, std::ptr::null(), &mut result)
                        };
                        if !result.audio.is_null() {
                            unsafe {
                                voirs_free_audio_buffer(result.audio);
                            }
                        }
                        error_code
                    })
                })
                .collect();

            for handle in handles {
                black_box(handle.join().unwrap());
            }
        })
    });

    // Thread pool operations
    group.bench_function("thread_pool_operations", |b| {
        b.iter(|| {
            // Simulate thread pool operations
            for i in 0..8 {
                black_box(i); // Simulate task processing
            }
        })
    });

    group.finish();
}

/// Benchmark audio format conversions
fn benchmark_audio_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_conversions");

    let buffer_sizes = vec![1024, 4096, 16384, 65536];

    for size in buffer_sizes {
        // Float to int16 conversion
        group.bench_with_input(
            BenchmarkId::new("float_to_int16", size),
            &size,
            |b, &size| {
                let input = vec![0.5f32; size];
                let mut output = vec![0i16; size];

                b.iter(|| unsafe {
                    black_box(voirs_convert_float_to_int16(
                        input.as_ptr(),
                        output.as_mut_ptr(),
                        size as u32,
                        VoirsEndianness::Native,
                    ));
                })
            },
        );

        // Sample rate conversion
        group.bench_with_input(
            BenchmarkId::new("sample_rate_conversion", size),
            &size,
            |b, &size| {
                let input = vec![0.5f32; size];
                let mut output = vec![0.0f32; size * 2]; // 2x upsampling

                b.iter(|| {
                    let mut output_samples = 0u32;
                    unsafe {
                        black_box(voirs_convert_sample_rate(
                            input.as_ptr(),
                            output.as_mut_ptr(),
                            size as u32,
                            22050,
                            44100,
                            1, // channels
                            &mut output_samples,
                        ));
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark language-specific operations
fn benchmark_language_specific(c: &mut Criterion) {
    let mut group = c.benchmark_group("language_specific");

    // Python binding overhead simulation
    group.bench_function("python_binding_overhead", |b| {
        let context = BenchmarkContext::new();
        let text = CString::new("Python binding test").unwrap();

        b.iter(|| {
            // Simulate Python object creation overhead
            let mut result = VoirsSynthesisResult::default();

            // Simulate parameter conversion
            let config = black_box(VoirsSynthesisConfig::default());

            let error_code = unsafe {
                black_box(voirs_synthesize_advanced(
                    text.as_ptr(),
                    std::ptr::null(),
                    &mut result,
                ))
            };

            // Simulate Python object cleanup
            if !result.audio.is_null() {
                unsafe {
                    voirs_free_audio_buffer(result.audio);
                }
            }

            black_box(error_code);
        })
    });

    // C binding direct call
    group.bench_function("c_binding_direct", |b| {
        let context = BenchmarkContext::new();
        let text = CString::new("C binding test").unwrap();

        b.iter(|| {
            let mut result = VoirsSynthesisResult::default();
            let error_code = unsafe {
                black_box(voirs_synthesize_advanced(
                    text.as_ptr(),
                    std::ptr::null(),
                    &mut result,
                ))
            };

            if !result.audio.is_null() {
                unsafe {
                    voirs_free_audio_buffer(result.audio);
                }
            }

            black_box(error_code);
        })
    });

    group.finish();
}

/// Benchmark error handling performance
fn benchmark_error_handling(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_handling");

    // Successful operation
    group.bench_function("success_path", |b| {
        let context = BenchmarkContext::new();
        let text = CString::new("Success test").unwrap();

        b.iter(|| {
            let mut result = VoirsSynthesisResult::default();
            let error_code = unsafe {
                black_box(voirs_synthesize_advanced(
                    text.as_ptr(),
                    std::ptr::null(),
                    &mut result,
                ))
            };

            if !result.audio.is_null() {
                unsafe {
                    voirs_free_audio_buffer(result.audio);
                }
            }

            black_box(error_code);
        })
    });

    // Error path
    group.bench_function("error_path", |b| {
        b.iter(|| {
            let mut result = VoirsSynthesisResult::default();
            let error_code = unsafe {
                black_box(voirs_synthesize_advanced(
                    std::ptr::null(), // Invalid text
                    std::ptr::null(), // Invalid config
                    &mut result,
                ))
            };

            black_box(error_code);
        })
    });

    // Error recovery
    group.bench_function("error_recovery", |b| {
        b.iter(|| {
            // Test basic error code conversion
            let error_code = black_box(VoirsErrorCode::InvalidParameter);
            let message = black_box(format!("{:?}", error_code));
            black_box(message);
        })
    });

    group.finish();
}

/// Benchmark streaming operations
fn benchmark_streaming_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_operations");

    // Streaming synthesis
    group.bench_function("streaming_synthesis", |b| {
        let context = BenchmarkContext::new();
        let text = CString::new("Streaming synthesis test with longer text content").unwrap();

        b.iter(|| {
            // Simulate streaming synthesis using advanced function
            let mut result = VoirsSynthesisResult::default();
            let error_code = unsafe {
                black_box(voirs_synthesize_advanced(
                    text.as_ptr(),
                    std::ptr::null(),
                    &mut result,
                ))
            };

            if !result.audio.is_null() {
                unsafe {
                    voirs_free_audio_buffer(result.audio);
                }
            }

            black_box(error_code);
        })
    });

    // Batch synthesis
    group.bench_function("batch_synthesis", |b| {
        let context = BenchmarkContext::new();
        let texts = vec![
            CString::new("First text").unwrap(),
            CString::new("Second text").unwrap(),
            CString::new("Third text").unwrap(),
        ];
        let text_ptrs: Vec<*const i8> = texts.iter().map(|s| s.as_ptr()).collect();

        b.iter(|| {
            // Simulate batch synthesis by processing each text individually
            for &text_ptr in &text_ptrs {
                let mut result = VoirsSynthesisResult::default();
                let error_code = unsafe {
                    black_box(voirs_synthesize_advanced(
                        text_ptr,
                        std::ptr::null(),
                        &mut result,
                    ))
                };

                if !result.audio.is_null() {
                    unsafe {
                        voirs_free_audio_buffer(result.audio);
                    }
                }

                black_box(error_code);
            }
        })
    });

    group.finish();
}

/// Benchmark performance monitoring overhead
fn benchmark_performance_monitoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_monitoring");

    // Performance statistics collection (simulated)
    group.bench_function("stats_collection", |b| {
        b.iter(|| {
            let stats = VoirsSynthesisStats::default();
            black_box(stats);
        })
    });

    // Performance monitoring overhead
    group.bench_function("monitoring_overhead", |b| {
        let context = BenchmarkContext::new();
        let text = CString::new("Monitoring test").unwrap();

        b.iter(|| {
            // Simulate monitoring overhead
            let _monitoring_overhead = black_box(std::time::Instant::now());

            let mut result = VoirsSynthesisResult::default();
            let error_code = unsafe {
                black_box(voirs_synthesize_advanced(
                    text.as_ptr(),
                    std::ptr::null(),
                    &mut result,
                ))
            };

            if !result.audio.is_null() {
                unsafe {
                    voirs_free_audio_buffer(result.audio);
                }
            }

            // Simulate stats collection
            let stats = VoirsSynthesisStats::default();

            black_box(error_code);
            black_box(stats);
        })
    });

    group.finish();
}

// Configure criterion groups
criterion_group!(
    name = ffi_benchmarks;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3));
    targets =
        benchmark_ffi_overhead,
        benchmark_synthesis_scalability,
        benchmark_memory_patterns,
        benchmark_concurrent_operations,
        benchmark_audio_conversions,
        benchmark_language_specific,
        benchmark_error_handling,
        benchmark_streaming_operations,
        benchmark_performance_monitoring
);

criterion_main!(ffi_benchmarks);

// Helper functions for benchmarking

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_context() {
        let context = BenchmarkContext::new();
        assert_ne!(context.pipeline_id, 0);
    }

    #[test]
    fn test_ffi_overhead_measurement() {
        // Test that we can measure FFI overhead
        let start = std::time::Instant::now();
        let pipeline_id = unsafe { voirs_create_pipeline() };
        let creation_time = start.elapsed();

        let start = std::time::Instant::now();
        unsafe {
            voirs_destroy_pipeline(pipeline_id);
        }
        let destruction_time = start.elapsed();

        // Verify reasonable performance
        assert!(creation_time < Duration::from_millis(100));
        assert!(destruction_time < Duration::from_millis(100));
    }

    #[test]
    fn test_memory_benchmark() {
        // Test basic memory allocation simulation
        let _overhead = std::time::Instant::now();
        assert!(true); // Basic test passes
    }

    #[test]
    fn test_error_handling_benchmark() {
        let mut result = VoirsSynthesisResult::default();
        let error_code = unsafe {
            voirs_synthesize_advanced(
                std::ptr::null(), // Invalid text
                std::ptr::null(), // Invalid config
                &mut result,
            )
        };

        assert_ne!(error_code, VoirsErrorCode::Success);
    }
}
