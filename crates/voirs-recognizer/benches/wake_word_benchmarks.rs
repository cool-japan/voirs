//! Benchmarks for wake word detection signal processing
//!
//! This benchmark suite measures the performance of:
//! - Audio energy computation for VAD
//! - Zero crossing rate computation
//! - Sliding window processing
//! - MFCC feature extraction

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Benchmark audio energy computation (RMS)
fn benchmark_energy_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy_computation");

    // Test different audio lengths
    for duration_secs in [1, 5, 10, 30] {
        let size = duration_secs * 16000;
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(duration_secs),
            &size,
            |b, &size| {
                let samples = vec![0.1f32; size];

                b.iter(|| {
                    // Compute RMS energy (critical for wake word detection)
                    let energy: f32 =
                        samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32;
                    black_box(energy.sqrt())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark sliding window processing
fn benchmark_sliding_window_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("sliding_window_processing");

    // Test different window sizes
    for window_size in [256, 512, 1024, 2048] {
        group.throughput(Throughput::Elements(window_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(window_size),
            &window_size,
            |b, &window_size| {
                let samples = vec![0.1f32; window_size * 10];
                let hop_size = window_size / 2;

                b.iter(|| {
                    let mut energies = Vec::new();
                    for i in (0..samples.len() - window_size).step_by(hop_size) {
                        let window = &samples[i..i + window_size];
                        let energy: f32 =
                            window.iter().map(|&s| s * s).sum::<f32>() / window.len() as f32;
                        energies.push(energy.sqrt());
                    }
                    black_box(energies)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark MFCC feature extraction
fn benchmark_feature_extraction_mfcc(c: &mut Criterion) {
    let mut group = c.benchmark_group("mfcc_feature_extraction");

    // Test different frame sizes
    for frame_size in [256, 512, 1024] {
        group.throughput(Throughput::Elements(frame_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(frame_size),
            &frame_size,
            |b, &frame_size| {
                let samples = vec![0.1f32; frame_size];

                b.iter(|| {
                    // Simple energy-based feature (placeholder for MFCC)
                    let energy: f32 = samples.iter().map(|&s| s * s).sum::<f32>();
                    let zcr = samples.windows(2).filter(|w| (w[0] * w[1]) < 0.0).count() as f32
                        / samples.len() as f32;

                    black_box((energy, zcr))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark zero crossing rate computation
fn benchmark_zero_crossing_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_crossing_rate");

    // Test different audio lengths
    for duration_secs in [1, 5, 10] {
        let size = duration_secs * 16000;
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(duration_secs),
            &size,
            |b, &size| {
                let samples = vec![0.1f32; size];

                b.iter(|| {
                    let zcr = samples.windows(2).filter(|w| (w[0] * w[1]) < 0.0).count() as f32
                        / (samples.len() - 1) as f32;

                    black_box(zcr)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_energy_computation,
    benchmark_sliding_window_processing,
    benchmark_feature_extraction_mfcc,
    benchmark_zero_crossing_rate
);
criterion_main!(benches);
