//! SciRS2 Performance Benchmarks
//!
//! Comprehensive benchmarks comparing SIMD-optimized operations against
//! scalar implementations to quantify performance improvements.
//!
//! Run with: `cargo bench --bench scirs2_benchmarks`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use voirs_acoustic::scirs2_ops::{
    NormalizationMethod, SciRS2MelOps, SciRS2NumericOps, SciRS2ParallelOps,
};
use voirs_acoustic::MelSpectrogram;

fn create_test_mel(n_mels: usize, n_frames: usize, seed: u64) -> MelSpectrogram {
    fastrand::seed(seed);

    let mut data = Vec::with_capacity(n_mels);
    for _ in 0..n_mels {
        let mut channel = Vec::with_capacity(n_frames);
        for _ in 0..n_frames {
            channel.push(fastrand::f32() * 100.0 - 50.0);
        }
        data.push(channel);
    }

    MelSpectrogram {
        data,
        sample_rate: 16000,
        hop_length: 256,
        n_mels,
        n_frames,
    }
}

fn scalar_normalize_min_max(mel: &mut MelSpectrogram) {
    let all_values: Vec<f32> = mel.data.iter().flatten().copied().collect();
    let min_val = all_values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = all_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;

    for channel in &mut mel.data {
        for val in channel.iter_mut() {
            *val = (*val - min_val) / range;
        }
    }
}

fn scalar_normalize_z_score(mel: &mut MelSpectrogram) {
    let all_values: Vec<f32> = mel.data.iter().flatten().copied().collect();
    let mean = all_values.iter().sum::<f32>() / all_values.len() as f32;

    let variance =
        all_values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / all_values.len() as f32;
    let std = variance.sqrt();

    for channel in &mut mel.data {
        for val in channel.iter_mut() {
            *val = (*val - mean) / std;
        }
    }
}

fn bench_simd_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_normalization");

    for size in [100, 500, 1000, 2000, 5000].iter() {
        let n_mels = 80;
        let n_frames = *size;
        let total_elements = n_mels * n_frames;

        group.throughput(Throughput::Elements(total_elements as u64));

        // SIMD min-max
        group.bench_with_input(BenchmarkId::new("simd_min_max", size), size, |b, _| {
            let mut mel = create_test_mel(n_mels, n_frames, 42);
            b.iter(|| {
                SciRS2MelOps::normalize_min_max_simd(black_box(&mut mel)).unwrap();
            });
        });

        // Scalar min-max
        group.bench_with_input(BenchmarkId::new("scalar_min_max", size), size, |b, _| {
            let mut mel = create_test_mel(n_mels, n_frames, 42);
            b.iter(|| {
                scalar_normalize_min_max(black_box(&mut mel));
            });
        });

        // SIMD z-score
        group.bench_with_input(BenchmarkId::new("simd_z_score", size), size, |b, _| {
            let mut mel = create_test_mel(n_mels, n_frames, 42);
            b.iter(|| {
                SciRS2MelOps::normalize_z_score_simd(black_box(&mut mel)).unwrap();
            });
        });

        // Scalar z-score
        group.bench_with_input(BenchmarkId::new("scalar_z_score", size), size, |b, _| {
            let mut mel = create_test_mel(n_mels, n_frames, 42);
            b.iter(|| {
                scalar_normalize_z_score(black_box(&mut mel));
            });
        });
    }

    group.finish();
}

fn bench_parallel_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_processing");

    for batch_size in [1, 2, 4, 8, 16, 32].iter() {
        let n_mels = 80;
        let n_frames = 500;

        group.throughput(Throughput::Elements(
            (*batch_size * n_mels * n_frames) as u64,
        ));

        // Parallel batch normalization
        group.bench_with_input(
            BenchmarkId::new("parallel_batch", batch_size),
            batch_size,
            |b, &size| {
                let mut mels: Vec<MelSpectrogram> = (0..size)
                    .map(|i| create_test_mel(n_mels, n_frames, i as u64))
                    .collect();
                b.iter(|| {
                    SciRS2MelOps::batch_normalize_parallel(
                        black_box(&mut mels),
                        NormalizationMethod::MinMax,
                    )
                    .unwrap();
                });
            },
        );

        // Sequential batch normalization
        group.bench_with_input(
            BenchmarkId::new("sequential_batch", batch_size),
            batch_size,
            |b, &size| {
                let mut mels: Vec<MelSpectrogram> = (0..size)
                    .map(|i| create_test_mel(n_mels, n_frames, i as u64))
                    .collect();
                b.iter(|| {
                    for mel in black_box(&mut mels) {
                        scalar_normalize_min_max(mel);
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_complex_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_operations");

    for size in [256, 512, 1024, 2048, 4096].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let real: Vec<f32> = (0..*size).map(|i| (i as f32 * 0.01).sin()).collect();
        let imag: Vec<f32> = (0..*size).map(|i| (i as f32 * 0.01).cos()).collect();

        group.bench_with_input(BenchmarkId::new("complex_transform", size), size, |b, _| {
            b.iter(|| SciRS2NumericOps::complex_mel_transform(black_box(&real), black_box(&imag)));
        });

        let complex = SciRS2NumericOps::complex_mel_transform(&real, &imag);

        group.bench_with_input(
            BenchmarkId::new("magnitude_spectrum", size),
            size,
            |b, _| {
                b.iter(|| SciRS2NumericOps::compute_magnitude_spectrum(black_box(&complex)));
            },
        );

        group.bench_with_input(BenchmarkId::new("phase_spectrum", size), size, |b, _| {
            b.iter(|| SciRS2NumericOps::compute_phase_spectrum(black_box(&complex)));
        });
    }

    group.finish();
}

fn bench_ndarray_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_conversion");

    for size in [(80, 100), (80, 500), (80, 2000), (128, 2000)].iter() {
        let (n_mels, n_frames) = *size;
        let total = n_mels * n_frames;

        group.throughput(Throughput::Elements(total as u64));

        let mel = create_test_mel(n_mels, n_frames, 42);

        group.bench_with_input(
            BenchmarkId::new("to_ndarray", format!("{}x{}", n_mels, n_frames)),
            &mel,
            |b, mel| {
                b.iter(|| SciRS2MelOps::to_ndarray(black_box(mel)).unwrap());
            },
        );

        let arr = SciRS2MelOps::to_ndarray(&mel).unwrap();

        group.bench_with_input(
            BenchmarkId::new("from_ndarray", format!("{}x{}", n_mels, n_frames)),
            &arr,
            |b, arr| {
                b.iter(|| SciRS2MelOps::from_ndarray(black_box(arr), 16000));
            },
        );
    }

    group.finish();
}

fn bench_parallel_synthesis(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_synthesis");

    let synthesize = Arc::new(|text: &str| {
        // Simulate light processing work
        let iterations = text.len() * 1000;
        let mut sum = 0.0_f32;
        for i in 0..iterations {
            sum += (i as f32).sin();
        }
        vec![sum, text.len() as f32]
    });

    for batch_size in [1, 2, 4, 8, 16].iter() {
        let texts: Vec<String> = (0..*batch_size)
            .map(|i| format!("text_sample_{}", i))
            .collect();

        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("parallel", batch_size),
            &texts,
            |b, texts| {
                b.iter(|| {
                    SciRS2ParallelOps::parallel_synthesis(black_box(texts), synthesize.clone())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sequential", batch_size),
            &texts,
            |b, texts| {
                b.iter(|| {
                    texts
                        .iter()
                        .map(|text| synthesize(text.as_str()))
                        .collect::<Vec<_>>()
                });
            },
        );
    }

    group.finish();
}

fn bench_numerical_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical_stability");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let numerator: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let denominator: Vec<f32> = (0..*size)
            .map(|i| if i % 10 == 0 { 0.0 } else { i as f32 })
            .collect();

        group.bench_with_input(BenchmarkId::new("safe_divide", size), size, |b, _| {
            b.iter(|| {
                SciRS2NumericOps::safe_divide(black_box(&numerator), black_box(&denominator), 1e-10)
            });
        });

        let mel_values: Vec<f32> = (0..*size).map(|i| (i as f32 + 1.0) * 0.1).collect();

        group.bench_with_input(BenchmarkId::new("log_mel", size), size, |b, _| {
            b.iter(|| SciRS2NumericOps::compute_log_mel(black_box(&mel_values), 1e-10));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_simd_normalization,
    bench_parallel_processing,
    bench_complex_operations,
    bench_ndarray_conversion,
    bench_parallel_synthesis,
    bench_numerical_stability,
);

criterion_main!(benches);
