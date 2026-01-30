//! SIMD Performance Benchmarks
//!
//! Comprehensive benchmarks comparing SIMD-optimized implementations
//! against scalar implementations for critical pitch processing operations.
//!
//! ## Benchmark Categories
//!
//! 1. **Autocorrelation**: F0 extraction performance
//! 2. **Interpolation**: Pitch contour interpolation
//! 3. **Smoothing**: Moving average filtering
//! 4. **Vibrato**: Vibrato application performance
//!
//! ## Expected Results
//!
//! - **Autocorrelation**: 3-4x speedup with SIMD
//! - **Interpolation**: 2-3x speedup with SIMD
//! - **Smoothing**: 2.5x speedup with SIMD
//! - **Vibrato**: 5-6x speedup with LUT + SIMD
//!
//! ## Run Benchmarks
//!
//! ```bash
//! cargo bench --bench simd_performance_benchmarks
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::f32::consts::PI;
use voirs_singing::pitch::PitchContour;
use voirs_singing::pitch_simd::{SimdPitchContourGenerator, SimdPitchProcessor};

/// Generate synthetic audio signal for benchmarking
fn generate_audio(sample_rate: f32, frequency: f32, duration: f32) -> Vec<f32> {
    let num_samples = (duration * sample_rate) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * PI * frequency * t).sin()
        })
        .collect()
}

/// Scalar autocorrelation implementation (baseline)
fn autocorrelation_scalar(audio: &[f32]) -> Option<f32> {
    if audio.len() < 512 {
        return None;
    }

    let sample_rate = 44100.0;
    let mut autocorr = vec![0.0; audio.len() / 2];

    // Compute autocorrelation
    for lag in 0..autocorr.len() {
        let mut sum = 0.0;
        for i in 0..audio.len() - lag {
            sum += audio[i] * audio[i + lag];
        }
        autocorr[lag] = sum;
    }

    // Find peak
    let min_period = (sample_rate / 800.0) as usize;
    let max_period = (sample_rate / 80.0) as usize;

    let mut max_val = 0.0;
    let mut max_idx = 0;

    for (i, &value) in autocorr
        .iter()
        .enumerate()
        .take(max_period.min(autocorr.len() - 1) + 1)
        .skip(min_period)
    {
        if value > max_val {
            max_val = value;
            max_idx = i;
        }
    }

    if max_val < 0.5 * autocorr[0] {
        return None;
    }

    Some(sample_rate / max_idx as f32)
}

/// Benchmark autocorrelation performance
fn bench_autocorrelation(c: &mut Criterion) {
    let mut group = c.benchmark_group("autocorrelation");

    let sample_rates = [44100.0];
    let durations = [0.05, 0.1, 0.2]; // 50ms, 100ms, 200ms

    for &sample_rate in &sample_rates {
        for &duration in &durations {
            let audio = generate_audio(sample_rate, 440.0, duration);
            let num_samples = audio.len();

            group.throughput(Throughput::Elements(num_samples as u64));

            // Scalar baseline
            group.bench_with_input(
                BenchmarkId::new("scalar", num_samples),
                &audio,
                |b, audio| {
                    b.iter(|| {
                        let _ = autocorrelation_scalar(black_box(audio));
                    });
                },
            );

            // SIMD optimized
            group.bench_with_input(BenchmarkId::new("simd", num_samples), &audio, |b, audio| {
                let mut processor = SimdPitchProcessor::new(sample_rate, 80.0, 800.0);
                b.iter(|| {
                    let _ = processor.autocorrelation_simd(black_box(audio));
                });
            });
        }
    }

    group.finish();
}

/// Benchmark interpolation performance
fn bench_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolation");

    let num_reference_points = [100, 500, 1000];
    let num_query_points = [1000, 5000, 10000];

    for &num_ref in &num_reference_points {
        for &num_query in &num_query_points {
            let time_points: Vec<f32> = (0..num_ref).map(|i| i as f32 / 100.0).collect();
            let f0_values: Vec<f32> = (0..num_ref).map(|i| 200.0 + (i % 100) as f32).collect();
            let query_times: Vec<f32> = (0..num_query).map(|i| i as f32 / 1000.0).collect();

            group.throughput(Throughput::Elements(num_query as u64));

            // Scalar baseline (using PitchContour)
            group.bench_with_input(
                BenchmarkId::new(format!("scalar_ref{}", num_ref), num_query),
                &(&time_points, &f0_values, &query_times),
                |b, (time_pts, f0_vals, queries)| {
                    let contour = PitchContour::new((*time_pts).clone(), (*f0_vals).clone());
                    b.iter(|| {
                        for &time in queries.iter() {
                            black_box(contour.f0_at_time(time));
                        }
                    });
                },
            );

            // SIMD optimized
            group.bench_with_input(
                BenchmarkId::new(format!("simd_ref{}", num_ref), num_query),
                &(&time_points, &f0_values, &query_times),
                |b, (time_pts, f0_vals, queries)| {
                    let mut processor = SimdPitchProcessor::default_singing();
                    let mut output = vec![0.0; queries.len()];
                    b.iter(|| {
                        processor.interpolate_linear_simd(
                            black_box(time_pts),
                            black_box(f0_vals),
                            black_box(queries),
                            black_box(&mut output),
                        );
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark smoothing performance
fn bench_smoothing(c: &mut Criterion) {
    let mut group = c.benchmark_group("smoothing");

    let sizes = [1000, 5000, 10000, 20000];
    let window_sizes = [3, 5, 7];

    for &size in &sizes {
        for &window_size in &window_sizes {
            let f0_values: Vec<f32> = (0..size)
                .map(|i| 200.0 + 50.0 * (i as f32 / 100.0).sin())
                .collect();

            group.throughput(Throughput::Elements(size as u64));

            // Scalar baseline (using PitchContour)
            group.bench_with_input(
                BenchmarkId::new(format!("scalar_w{}", window_size), size),
                &f0_values,
                |b, f0_vals| {
                    let mut contour = PitchContour::new(
                        (0..size).map(|i| i as f32 / 100.0).collect(),
                        f0_vals.clone(),
                    );
                    b.iter(|| {
                        contour.smooth(black_box(1.0));
                    });
                },
            );

            // SIMD optimized
            group.bench_with_input(
                BenchmarkId::new(format!("simd_w{}", window_size), size),
                &f0_values,
                |b, f0_vals| {
                    let mut processor = SimdPitchProcessor::default_singing();
                    let mut output = vec![0.0; f0_vals.len()];
                    b.iter(|| {
                        processor.smooth_simd(
                            black_box(f0_vals),
                            black_box(&mut output),
                            black_box(window_size),
                            black_box(1.0),
                        );
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark vibrato application performance
fn bench_vibrato(c: &mut Criterion) {
    let mut group = c.benchmark_group("vibrato");

    let sizes = [1000, 5000, 10000];
    let vibrato_freqs = [5.0, 7.0];

    for &size in &sizes {
        for &freq in &vibrato_freqs {
            let f0_values = vec![440.0; size];
            let time_points: Vec<f32> = (0..size).map(|i| i as f32 / 100.0).collect();

            group.throughput(Throughput::Elements(size as u64));

            // Scalar baseline (using PitchContour - no LUT)
            group.bench_with_input(
                BenchmarkId::new(format!("scalar_{}hz", freq), size),
                &(&f0_values, &time_points),
                |b, (f0_vals, times)| {
                    let mut contour = PitchContour::new(times.to_vec(), f0_vals.to_vec());
                    let vibrato_params = voirs_singing::pitch::VibratoParams {
                        frequency: freq,
                        depth: 50.0,
                        onset_time: 0.0,
                        rate_variation: 0.0,
                        depth_variation: 0.0,
                        waveform: voirs_singing::pitch::VibratoWaveform::Sine,
                        phase: 0.0,
                    };
                    b.iter(|| {
                        contour.apply_vibrato(black_box(&vibrato_params));
                    });
                },
            );

            // SIMD + LUT optimized
            group.bench_with_input(
                BenchmarkId::new(format!("simd_lut_{}hz", freq), size),
                &(&f0_values, &time_points),
                |b, (f0_vals, times)| {
                    let generator = SimdPitchContourGenerator::new(44100.0);
                    let mut output = vec![0.0; f0_vals.len()];
                    b.iter(|| {
                        generator.apply_vibrato_simd(
                            black_box(f0_vals),
                            black_box(times),
                            black_box(&mut output),
                            black_box(freq),
                            black_box(50.0),
                            black_box(0.0),
                        );
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark cents deviation calculation
fn bench_cents_deviation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cents_deviation");

    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        let detected: Vec<f32> = (0..size)
            .map(|i| 440.0 * (1.0 + (i as f32 / 1000.0)))
            .collect();
        let target = vec![440.0; size];

        group.throughput(Throughput::Elements(size as u64));

        // Scalar baseline
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &(&detected, &target),
            |b, (det, tgt)| {
                b.iter(|| {
                    let log2_scale = 1200.0 / std::f32::consts::LN_2;
                    let cents: Vec<f32> = det
                        .iter()
                        .zip(tgt.iter())
                        .map(|(&d, &t)| {
                            if d > 0.0 && t > 0.0 {
                                (d / t).ln() * log2_scale
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    black_box(cents);
                });
            },
        );

        // SIMD optimized
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &(&detected, &target),
            |b, (det, tgt)| {
                let processor = SimdPitchProcessor::default_singing();
                let mut output = vec![0.0; size];
                b.iter(|| {
                    processor.compute_cents_deviation_simd(
                        black_box(det),
                        black_box(tgt),
                        black_box(&mut output),
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_autocorrelation,
    bench_interpolation,
    bench_smoothing,
    bench_vibrato,
    bench_cents_deviation
);
criterion_main!(benches);
