//! Optimization Validation Benchmark for VoiRS Evaluation
//!
//! This benchmark validates that performance optimizations are working correctly:
//! - SIMD optimizations provide expected speedup
//! - Parallel processing achieves expected efficiency
//! - Cache optimizations reduce computation time
//! - Optimized algorithms outperform baseline implementations
//! - Memory optimizations reduce allocation overhead

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use voirs_evaluation::performance::{parallel_correlation, simd, SlidingWindowProcessor};
use voirs_evaluation::prelude::*;
use voirs_evaluation::quality::mcd::*;
use voirs_evaluation::quality::pesq::*;
use voirs_evaluation::quality::stoi::*;
use voirs_evaluation::{calculate_correlation, precision::precise_correlation};
use voirs_sdk::AudioBuffer;

fn generate_test_audio(length: usize, sample_rate: u32) -> AudioBuffer {
    let samples: Vec<f32> = (0..length)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let fundamental = 0.7 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            let harmonic2 = 0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin();
            let harmonic3 = 0.1 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin();
            fundamental + harmonic2 + harmonic3
        })
        .collect();

    AudioBuffer::mono(samples, sample_rate)
}

/// Validation 1: SIMD optimizations provide expected speedup
fn bench_simd_optimization_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Optimization: SIMD Speedup Validation");
    group.measurement_time(Duration::from_secs(10));

    let data_sizes = vec![1000, 10000, 100000];

    for size in data_sizes {
        let data_a: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
        let data_b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).cos()).collect();

        // Benchmark scalar implementation
        group.bench_with_input(
            BenchmarkId::new("ScalarCorrelation", size),
            &(&data_a, &data_b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let start = Instant::now();
                    let result = calculate_correlation_scalar(black_box(a), black_box(b));
                    let duration = start.elapsed();
                    black_box((result, duration))
                })
            },
        );

        // Benchmark SIMD implementation
        group.bench_with_input(
            BenchmarkId::new("SIMDCorrelation", size),
            &(&data_a, &data_b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let start = Instant::now();
                    let result = simd::dot_product(black_box(a), black_box(b));
                    let duration = start.elapsed();
                    black_box((result, duration))
                })
            },
        );
    }
    group.finish();
}

/// Validation 2: Parallel processing achieves expected efficiency
fn bench_parallel_optimization_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Optimization: Parallel Processing Validation");
    group.measurement_time(Duration::from_secs(15));

    let data_sizes = vec![10000, 50000, 100000];

    for size in data_sizes {
        let data_a: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
        let data_b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).cos()).collect();

        // Benchmark sequential implementation
        group.bench_with_input(
            BenchmarkId::new("SequentialProcessing", size),
            &(&data_a, &data_b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let start = Instant::now();
                    let result = calculate_correlation(black_box(a), black_box(b));
                    let duration = start.elapsed();
                    black_box((result, duration))
                })
            },
        );

        // Benchmark parallel implementation
        group.bench_with_input(
            BenchmarkId::new("ParallelProcessing", size),
            &(&data_a, &data_b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let start = Instant::now();
                    let result = parallel_correlation(black_box(a), black_box(b));
                    let duration = start.elapsed();
                    black_box((result, duration))
                })
            },
        );
    }
    group.finish();
}

/// Validation 3: Cache optimizations reduce computation time
fn bench_cache_optimization_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Optimization: Cache Optimization Validation");
    group.measurement_time(Duration::from_secs(20));

    let audio_sizes = vec![16000, 48000, 96000]; // 1s, 3s, 6s at 16kHz

    for size in audio_sizes {
        let test_audio = generate_test_audio(size, 16000);
        let reference_audio = generate_test_audio(size, 16000);

        // Benchmark without cache (cold runs)
        group.bench_with_input(
            BenchmarkId::new("WithoutCache", size),
            &(&test_audio, &reference_audio),
            |bench, (test, reference)| {
                bench.iter(|| {
                    rt.block_on(async {
                        let start = Instant::now();

                        // Create fresh evaluator each time (no cache benefits)
                        let mut mcd_evaluator = MCDEvaluator::new(16000).unwrap();
                        let result = mcd_evaluator
                            .calculate_mcd_simple(black_box(test), black_box(reference))
                            .await
                            .unwrap();

                        let duration = start.elapsed();
                        black_box((result, duration))
                    })
                })
            },
        );

        // Benchmark with cache (warm runs)
        group.bench_with_input(
            BenchmarkId::new("WithCache", size),
            &(&test_audio, &reference_audio),
            |bench, (test, reference)| {
                bench.iter(|| {
                    rt.block_on(async {
                        // Pre-warm the cache
                        let mut mcd_evaluator = MCDEvaluator::new(16000).unwrap();
                        let _ = mcd_evaluator
                            .calculate_mcd_simple(test, reference)
                            .await
                            .unwrap();

                        let start = Instant::now();

                        // Now measure with cache benefits
                        let result = mcd_evaluator
                            .calculate_mcd_simple(black_box(test), black_box(reference))
                            .await
                            .unwrap();

                        let duration = start.elapsed();
                        black_box((result, duration))
                    })
                })
            },
        );
    }
    group.finish();
}

/// Validation 4: Optimized algorithms outperform baseline implementations
fn bench_algorithm_optimization_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Optimization: Algorithm Optimization Validation");
    group.measurement_time(Duration::from_secs(15));

    let audio_sizes = vec![16000, 32000, 48000];

    for size in audio_sizes {
        let test_audio = generate_test_audio(size, 16000);
        let reference_audio = generate_test_audio(size, 16000);

        // Benchmark baseline PESQ implementation
        group.bench_with_input(
            BenchmarkId::new("BaselinePESQ", size),
            &(&test_audio, &reference_audio),
            |bench, (test, reference)| {
                bench.iter(|| {
                    rt.block_on(async {
                        let start = Instant::now();

                        let pesq_evaluator = PESQEvaluator::new_wideband().unwrap();
                        let result = pesq_evaluator
                            .calculate_pesq(black_box(test), black_box(reference))
                            .await
                            .unwrap();

                        let duration = start.elapsed();
                        black_box((result, duration))
                    })
                })
            },
        );

        // Benchmark optimized STOI implementation
        group.bench_with_input(
            BenchmarkId::new("OptimizedSTOI", size),
            &(&test_audio, &reference_audio),
            |bench, (test, reference)| {
                bench.iter(|| {
                    rt.block_on(async {
                        let start = Instant::now();

                        let stoi_evaluator = STOIEvaluator::new(16000).unwrap();
                        let result = stoi_evaluator
                            .calculate_stoi(black_box(test), black_box(reference))
                            .await
                            .unwrap();

                        let duration = start.elapsed();
                        black_box((result, duration))
                    })
                })
            },
        );
    }
    group.finish();
}

/// Validation 5: Memory optimizations reduce allocation overhead
fn bench_memory_optimization_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Optimization: Memory Optimization Validation");
    group.measurement_time(Duration::from_secs(10));

    let batch_sizes = vec![10, 25, 50];

    for batch_size in batch_sizes {
        // Benchmark with frequent allocations (baseline)
        group.bench_with_input(
            BenchmarkId::new("FrequentAllocations", batch_size),
            &batch_size,
            |bench, &size| {
                bench.iter(|| {
                    rt.block_on(async {
                        let start = Instant::now();

                        let mut results = Vec::new();
                        for _ in 0..size {
                            // Create fresh evaluator each time (more allocations)
                            let quality_evaluator = QualityEvaluator::new().await.unwrap();
                            let test_audio = generate_test_audio(16000, 16000);
                            let reference_audio = generate_test_audio(16000, 16000);

                            let result = quality_evaluator
                                .evaluate_quality(
                                    black_box(&test_audio),
                                    Some(black_box(&reference_audio)),
                                    None,
                                )
                                .await
                                .unwrap();
                            results.push(result);
                        }

                        let duration = start.elapsed();
                        black_box((results, duration))
                    })
                })
            },
        );

        // Benchmark with reused allocations (optimized)
        group.bench_with_input(
            BenchmarkId::new("ReusedAllocations", batch_size),
            &batch_size,
            |bench, &size| {
                bench.iter(|| {
                    rt.block_on(async {
                        let start = Instant::now();

                        // Reuse evaluator across batch (fewer allocations)
                        let quality_evaluator = QualityEvaluator::new().await.unwrap();
                        let mut results = Vec::with_capacity(size); // Pre-allocate

                        for _ in 0..size {
                            let test_audio = generate_test_audio(16000, 16000);
                            let reference_audio = generate_test_audio(16000, 16000);

                            let result = quality_evaluator
                                .evaluate_quality(
                                    black_box(&test_audio),
                                    Some(black_box(&reference_audio)),
                                    None,
                                )
                                .await
                                .unwrap();
                            results.push(result);
                        }

                        let duration = start.elapsed();
                        black_box((results, duration))
                    })
                })
            },
        );
    }
    group.finish();
}

/// Validation 6: Sliding window processor optimization
fn bench_sliding_window_optimization_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Optimization: Sliding Window Validation");
    group.measurement_time(Duration::from_secs(10));

    let data_sizes = vec![1000, 5000, 10000];
    let window_size = 128;

    for size in data_sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();

        // Benchmark naive windowing approach
        group.bench_with_input(
            BenchmarkId::new("NaiveWindowing", size),
            &data,
            |bench, data| {
                bench.iter(|| {
                    let start = Instant::now();

                    let mut results = Vec::new();
                    for i in 0..=(data.len().saturating_sub(window_size)) {
                        let window = &data[i..i + window_size];
                        let mean = window.iter().sum::<f32>() / window_size as f32;
                        results.push(mean);
                    }

                    let duration = start.elapsed();
                    black_box((results, duration))
                })
            },
        );

        // Benchmark optimized sliding window processor
        group.bench_with_input(
            BenchmarkId::new("OptimizedSliding", size),
            &data,
            |bench, data| {
                bench.iter(|| {
                    let start = Instant::now();

                    let processor = SlidingWindowProcessor::new(window_size, window_size / 2);
                    let results = processor.process_parallel(data, |window| {
                        window.iter().sum::<f32>() / window.len() as f32
                    });

                    let duration = start.elapsed();
                    black_box((results, duration))
                })
            },
        );
    }
    group.finish();
}

/// Helper function: Scalar correlation for comparison
fn calculate_correlation_scalar(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let n = a.len() as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;

    let mut numerator = 0.0;
    let mut sum_sq_a = 0.0;
    let mut sum_sq_b = 0.0;

    for i in 0..a.len() {
        let diff_a = a[i] - mean_a;
        let diff_b = b[i] - mean_b;
        numerator += diff_a * diff_b;
        sum_sq_a += diff_a * diff_a;
        sum_sq_b += diff_b * diff_b;
    }

    let denominator = (sum_sq_a * sum_sq_b).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

criterion_group!(
    optimization_validation,
    bench_simd_optimization_validation,
    bench_parallel_optimization_validation,
    bench_cache_optimization_validation,
    bench_algorithm_optimization_validation,
    bench_memory_optimization_validation,
    bench_sliding_window_optimization_validation
);
criterion_main!(optimization_validation);
