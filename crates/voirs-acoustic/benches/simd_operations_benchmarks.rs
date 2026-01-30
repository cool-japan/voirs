//! Comprehensive benchmarks for SIMD-optimized operations
//!
//! This benchmark suite compares scalar and SIMD implementations across various
//! input sizes to demonstrate performance improvements.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use voirs_acoustic::mel::ops::MelOps;
use voirs_acoustic::prosody::simd_ops::*;
use voirs_acoustic::MelSpectrogram;

/// Generate test mel spectrogram
fn generate_test_mel(n_mels: usize, n_frames: usize) -> MelSpectrogram {
    let data: Vec<Vec<f32>> = (0..n_mels)
        .map(|i| {
            (0..n_frames)
                .map(|j| ((i * n_frames + j) as f32 * 0.01).sin() * 50.0 + 100.0)
                .collect()
        })
        .collect();

    MelSpectrogram::new(data, 22050, 256)
}

/// Generate test F0 contour
fn generate_test_f0(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            if i % 5 == 0 {
                0.0 // Unvoiced
            } else {
                100.0 + (i as f32 * 0.1).sin() * 20.0 // Voiced with variation
            }
        })
        .collect()
}

/// Benchmark mel unit norm normalization (scalar vs SIMD)
fn bench_mel_unit_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("mel_unit_norm");

    for size in [64, 128, 256, 512].iter() {
        let n_frames = *size;
        let n_mels = 80;
        let total_elements = n_mels * n_frames;

        group.throughput(Throughput::Elements(total_elements as u64));

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, &_size| {
            b.iter(|| {
                let mut mel = generate_test_mel(n_mels, n_frames);
                MelOps::normalize_unit_norm(&mut mel).unwrap();
                black_box(mel);
            });
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &_size| {
            b.iter(|| {
                let mut mel = generate_test_mel(n_mels, n_frames);
                MelOps::normalize_unit_norm_simd(&mut mel).unwrap();
                black_box(mel);
            });
        });
    }

    group.finish();
}

/// Benchmark mel smoothing (scalar vs SIMD)
fn bench_mel_smoothing(c: &mut Criterion) {
    let mut group = c.benchmark_group("mel_smoothing");

    for size in [64, 128, 256, 512].iter() {
        let n_frames = *size;
        let n_mels = 80;
        let kernel_size = 5;
        let total_elements = n_mels * n_frames;

        group.throughput(Throughput::Elements(total_elements as u64));

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, &_size| {
            b.iter(|| {
                let mut mel = generate_test_mel(n_mels, n_frames);
                MelOps::smooth(&mut mel, kernel_size).unwrap();
                black_box(mel);
            });
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &_size| {
            b.iter(|| {
                let mut mel = generate_test_mel(n_mels, n_frames);
                MelOps::smooth_simd(&mut mel, kernel_size).unwrap();
                black_box(mel);
            });
        });
    }

    group.finish();
}

/// Benchmark F0 contour smoothing
fn bench_f0_smoothing(c: &mut Criterion) {
    let mut group = c.benchmark_group("f0_smoothing");

    for size in [100, 500, 1000, 2000].iter() {
        let length = *size;

        group.throughput(Throughput::Elements(length as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &_size| {
            b.iter(|| {
                let mut f0 = generate_test_f0(length);
                smooth_f0_contour_simd(&mut f0, 0.7);
                black_box(f0);
            });
        });
    }

    group.finish();
}

/// Benchmark pitch shifting
fn bench_pitch_shifting(c: &mut Criterion) {
    let mut group = c.benchmark_group("pitch_shifting");

    for size in [100, 500, 1000, 2000].iter() {
        let length = *size;

        group.throughput(Throughput::Elements(length as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &_size| {
            b.iter(|| {
                let mut f0 = generate_test_f0(length);
                shift_pitch_simd(&mut f0, 12.0); // +1 octave
                black_box(f0);
            });
        });
    }

    group.finish();
}

/// Benchmark energy envelope smoothing
fn bench_energy_smoothing(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy_smoothing");

    for size in [100, 500, 1000, 2000].iter() {
        let length = *size;
        let window_size = 5;

        group.throughput(Throughput::Elements(length as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &_size| {
            b.iter(|| {
                let mut energy: Vec<f32> =
                    (0..length).map(|i| (i as f32 * 0.01).sin().abs()).collect();
                smooth_energy_envelope_simd(&mut energy, window_size);
                black_box(energy);
            });
        });
    }

    group.finish();
}

/// Benchmark duration scaling
fn bench_duration_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("duration_scaling");

    for size in [100, 500, 1000, 2000].iter() {
        let length = *size;

        group.throughput(Throughput::Elements(length as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &_size| {
            b.iter(|| {
                let signal: Vec<f32> = (0..length).map(|i| (i as f32 * 0.01).sin()).collect();
                let scaled = scale_duration_simd(&signal, 1.5);
                black_box(scaled);
            });
        });
    }

    group.finish();
}

/// Benchmark RMS energy computation
fn bench_rms_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("rms_energy");

    for size in [100, 500, 1000, 2000, 5000].iter() {
        let length = *size;

        group.throughput(Throughput::Elements(length as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &_size| {
            b.iter(|| {
                let signal: Vec<f32> = (0..length).map(|i| (i as f32 * 0.01).sin()).collect();
                let rms = compute_rms_energy_simd(&signal);
                black_box(rms);
            });
        });
    }

    group.finish();
}

/// Benchmark F0 linear interpolation
fn bench_f0_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("f0_interpolation");

    for num_keyframes in [5, 10, 20, 50].iter() {
        let n = *num_keyframes;
        let output_length = n * 20; // 20 samples between keyframes

        group.throughput(Throughput::Elements(output_length as u64));

        group.bench_with_input(
            BenchmarkId::new("simd", num_keyframes),
            num_keyframes,
            |b, &_n| {
                b.iter(|| {
                    let keyframes: Vec<f32> = (0..n).map(|i| 100.0 + i as f32 * 5.0).collect();
                    let keyframe_times: Vec<usize> = (0..n).map(|i| i * 20).collect();
                    let interpolated =
                        interpolate_f0_linear_simd(&keyframes, &keyframe_times, output_length);
                    black_box(interpolated);
                });
            },
        );
    }

    group.finish();
}

/// End-to-end benchmark: Complete prosody processing pipeline
fn bench_prosody_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("prosody_pipeline");

    for size in [200, 500, 1000].iter() {
        let length = *size;

        group.throughput(Throughput::Elements(length as u64));

        group.bench_with_input(BenchmarkId::new("complete", size), size, |b, &_size| {
            b.iter(|| {
                // 1. Generate F0 contour
                let mut f0 = generate_test_f0(length);

                // 2. Smooth F0
                smooth_f0_contour_simd(&mut f0, 0.7);

                // 3. Apply pitch shift
                shift_pitch_simd(&mut f0, 2.0);

                // 4. Generate energy envelope
                let mut energy: Vec<f32> =
                    (0..length).map(|i| (i as f32 * 0.01).sin().abs()).collect();

                // 5. Smooth energy
                smooth_energy_envelope_simd(&mut energy, 5);

                // 6. Compute RMS
                let rms = compute_rms_energy_simd(&energy);

                black_box((f0, energy, rms));
            });
        });
    }

    group.finish();
}

/// End-to-end benchmark: Complete mel processing pipeline
fn bench_mel_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("mel_pipeline");

    for size in [128, 256, 512].iter() {
        let n_frames = *size;
        let n_mels = 80;

        group.throughput(Throughput::Elements((n_mels * n_frames) as u64));

        group.bench_with_input(
            BenchmarkId::new("complete_simd", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    // 1. Generate mel spectrogram
                    let mut mel = generate_test_mel(n_mels, n_frames);

                    // 2. Smooth
                    MelOps::smooth_simd(&mut mel, 5).unwrap();

                    // 3. Normalize
                    MelOps::normalize_unit_norm_simd(&mut mel).unwrap();

                    black_box(mel);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_mel_unit_norm,
    bench_mel_smoothing,
    bench_f0_smoothing,
    bench_pitch_shifting,
    bench_energy_smoothing,
    bench_duration_scaling,
    bench_rms_energy,
    bench_f0_interpolation,
    bench_prosody_pipeline,
    bench_mel_pipeline,
);

criterion_main!(benches);
