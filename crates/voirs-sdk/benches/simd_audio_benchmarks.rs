//! SIMD Audio Processing Performance Benchmarks
//!
//! This benchmark suite measures the performance impact of SIMD optimizations
//! for audio processing operations in VoiRS SDK.
//!
//! ## Running Benchmarks
//!
//! ```bash
//! # Run all SIMD benchmarks
//! cargo bench --bench simd_audio_benchmarks
//!
//! # Run specific benchmark group
//! cargo bench --bench simd_audio_benchmarks -- apply_gain
//! cargo bench --bench simd_audio_benchmarks -- normalize
//! cargo bench --bench simd_audio_benchmarks -- mix
//! cargo bench --bench simd_audio_benchmarks -- rms
//! cargo bench --bench simd_audio_benchmarks -- peak
//! ```
//!
//! ## Expected Results
//!
//! On AVX2-enabled CPUs, expect 2-4x speedup for buffers > 64 samples.
//! On ARM64 NEON CPUs, expect 2-3x speedup for buffers > 64 samples.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use voirs_sdk::AudioBuffer;

/// Benchmark buffer sizes representing different use cases
const BUFFER_SIZES: &[usize] = &[
    32,    // Small: Below SIMD threshold (scalar path)
    64,    // Threshold: Boundary case
    128,   // Small SIMD: First SIMD benefit
    1024,  // Medium: Typical synthesis chunk
    4410,  // Large: 100ms at 44.1kHz
    44100, // Very large: 1 second at 44.1kHz
];

/// Create test audio buffer with sine wave content
fn create_test_buffer(size: usize, sample_rate: u32) -> AudioBuffer {
    let frequency = 440.0; // A4 note
    let amplitude = 0.5;
    let samples: Vec<f32> = (0..size)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin()
        })
        .collect();

    AudioBuffer::new(samples, sample_rate, 1)
}

/// Benchmark apply_gain operation
///
/// Tests the performance of gain application (dB to linear conversion + scalar multiplication).
/// SIMD optimization should provide 3-4x speedup for buffers > 64 samples on AVX2.
fn bench_apply_gain(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_gain");

    for &size in BUFFER_SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || create_test_buffer(size, 44100),
                |mut buffer| {
                    buffer.apply_gain(black_box(6.0)).unwrap();
                    buffer
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark normalize operation
///
/// Tests the performance of audio normalization (peak detection + gain application).
/// SIMD optimization includes:
/// - SIMD absolute value + max reduction for peak finding
/// - SIMD scalar multiplication for gain application
///
/// Expected speedup: 2-3x for buffers > 64 samples
fn bench_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize");

    for &size in BUFFER_SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || create_test_buffer(size, 44100),
                |mut buffer| {
                    buffer.normalize(black_box(0.8)).unwrap();
                    buffer
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark mix operation
///
/// Tests the performance of audio mixing (fused multiply-add operation).
/// SIMD FMA should provide 3-5x speedup for buffers > 64 samples.
fn bench_mix(c: &mut Criterion) {
    let mut group = c.benchmark_group("mix");

    for &size in BUFFER_SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let other = create_test_buffer(size, 44100);

            b.iter_batched(
                || create_test_buffer(size, 44100),
                |mut buffer| {
                    buffer.mix(black_box(&other), black_box(0.5)).unwrap();
                    buffer
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark RMS calculation
///
/// Tests the performance of RMS (Root Mean Square) calculation.
/// SIMD optimization uses sum-of-squares reduction.
/// Expected speedup: 2-4x for buffers > 64 samples
fn bench_rms(c: &mut Criterion) {
    let mut group = c.benchmark_group("rms");

    for &size in BUFFER_SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let buffer = create_test_buffer(size, 44100);

            b.iter(|| {
                black_box(buffer.rms());
            });
        });
    }

    group.finish();
}

/// Benchmark peak detection
///
/// Tests the performance of peak amplitude detection.
/// SIMD optimization uses absolute value + max reduction.
/// Expected speedup: 2-3x for buffers > 64 samples
fn bench_peak(c: &mut Criterion) {
    let mut group = c.benchmark_group("peak");

    for &size in BUFFER_SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let buffer = create_test_buffer(size, 44100);

            b.iter(|| {
                black_box(buffer.peak());
            });
        });
    }

    group.finish();
}

/// Benchmark threshold comparison
///
/// Compares performance at SIMD threshold boundary (64 samples).
/// This benchmark helps validate the threshold choice.
fn bench_threshold_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold_comparison");

    // Test around the threshold
    let threshold_sizes = [32, 48, 64, 80, 96, 128];

    for &size in &threshold_sizes {
        group.bench_with_input(BenchmarkId::new("apply_gain", size), &size, |b, &size| {
            b.iter_batched(
                || create_test_buffer(size, 44100),
                |mut buffer| {
                    buffer.apply_gain(6.0).unwrap();
                    buffer
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark complete audio processing pipeline
///
/// Tests the combined performance of multiple operations in sequence,
/// simulating a real-world audio processing workflow.
fn bench_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_pipeline");

    for &size in &[1024, 4410, 44100] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let other = create_test_buffer(size, 44100);

            b.iter_batched(
                || create_test_buffer(size, 44100),
                |mut buffer| {
                    // Typical audio processing pipeline
                    buffer.apply_gain(black_box(3.0)).unwrap();
                    buffer.mix(black_box(&other), black_box(0.3)).unwrap();
                    buffer.normalize(black_box(0.9)).unwrap();

                    // Calculate metrics
                    let _rms = buffer.rms();
                    let _peak = buffer.peak();

                    buffer
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_apply_gain,
    bench_normalize,
    bench_mix,
    bench_rms,
    bench_peak,
    bench_threshold_comparison,
    bench_pipeline
);

criterion_main!(benches);
