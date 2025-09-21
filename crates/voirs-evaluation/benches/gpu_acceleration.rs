use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use tokio::runtime::Runtime;
use voirs_evaluation::performance::*;
use voirs_evaluation::prelude::*;
use voirs_sdk::AudioBuffer;

fn generate_test_data(size: usize) -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
    let y: Vec<f32> = (0..size)
        .map(|i| ((i + 100) as f32 * 0.001).cos())
        .collect();
    (x, y)
}

fn generate_test_audio(length: usize, sample_rate: u32) -> AudioBuffer {
    let samples: Vec<f32> = (0..length)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();
    AudioBuffer::mono(samples, sample_rate)
}

fn bench_correlation_implementations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Correlation Implementations");
    group.measurement_time(Duration::from_secs(10));

    let data_sizes = vec![1000, 10000, 50000, 100000];

    for size in data_sizes {
        let (data_x, data_y) = generate_test_data(size);

        // Standard correlation
        group.bench_with_input(
            BenchmarkId::new("Standard_Correlation", size),
            &(data_x.clone(), data_y.clone()),
            |bench, (x, y)| {
                bench.iter(|| voirs_evaluation::calculate_correlation(black_box(x), black_box(y)))
            },
        );

        // Parallel correlation
        group.bench_with_input(
            BenchmarkId::new("Parallel_Correlation", size),
            &(data_x.clone(), data_y.clone()),
            |bench, (x, y)| bench.iter(|| parallel_correlation(black_box(x), black_box(y))),
        );
    }
    group.finish();
}

fn bench_fft_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Performance");
    group.measurement_time(Duration::from_secs(10));

    let fft_sizes = vec![512, 1024, 2048, 4096];

    for size in fft_sizes {
        let signal: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
        let batch_signals = vec![signal.clone(); 10];

        // Parallel FFT batch processing
        group.bench_with_input(
            BenchmarkId::new("Parallel_FFT_Batch", size),
            &batch_signals,
            |bench, signals| bench.iter(|| parallel_fft_batch(black_box(signals))),
        );
    }
    group.finish();
}

fn bench_parallel_batch_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Parallel Batch Processing");
    group.measurement_time(Duration::from_secs(15));

    let batch_sizes = vec![5, 10, 20];
    let audio_length = 16000; // 1 second

    let evaluator = rt.block_on(async { QualityEvaluator::new().await.unwrap() });

    for batch_size in batch_sizes {
        let mut test_audios = Vec::new();
        let mut reference_audios = Vec::new();

        for _ in 0..batch_size {
            test_audios.push(generate_test_audio(audio_length, 16000));
            reference_audios.push(generate_test_audio(audio_length, 16000));
        }

        // Sequential processing
        group.bench_with_input(
            BenchmarkId::new("Sequential", batch_size),
            &(test_audios.clone(), reference_audios.clone()),
            |bench, (tests, references)| {
                bench.iter(|| {
                    rt.block_on(async {
                        let mut results = Vec::new();
                        for (test, reference) in tests.iter().zip(references.iter()) {
                            let result = evaluator
                                .evaluate_quality(black_box(test), Some(black_box(reference)), None)
                                .await;
                            results.push(result);
                        }
                        black_box(results)
                    })
                })
            },
        );

        // Chunked processing
        group.bench_with_input(
            BenchmarkId::new("Chunked", batch_size),
            &(test_audios.clone(), reference_audios.clone()),
            |bench, (tests, references)| {
                bench.iter(|| {
                    rt.block_on(async {
                        let chunk_size = 5;
                        let mut all_results = Vec::new();

                        for (test_chunk, ref_chunk) in
                            tests.chunks(chunk_size).zip(references.chunks(chunk_size))
                        {
                            let mut chunk_results = Vec::new();
                            for (test, reference) in test_chunk.iter().zip(ref_chunk.iter()) {
                                let result = evaluator
                                    .evaluate_quality(
                                        black_box(test),
                                        Some(black_box(reference)),
                                        None,
                                    )
                                    .await;
                                chunk_results.push(result);
                            }
                            all_results.extend(chunk_results);
                        }
                        black_box(all_results)
                    })
                })
            },
        );
    }
    group.finish();
}

fn bench_chunked_audio_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Chunked Audio Processing");
    group.measurement_time(Duration::from_secs(10));

    let audio_lengths = vec![16000, 48000, 80000]; // 1s, 3s, 5s
    let chunk_sizes = vec![1000, 2000, 4000];

    for audio_length in audio_lengths {
        let audio_data: Vec<f32> = (0..audio_length)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        for chunk_size in &chunk_sizes {
            group.bench_with_input(
                BenchmarkId::new(
                    "ChunkedProcessing",
                    format!("{audio_length}samples_{chunk_size}chunk"),
                ),
                &(audio_data.clone(), *chunk_size),
                |bench, (data, chunk_size)| {
                    bench.iter(|| {
                        process_audio_chunks(black_box(data), *chunk_size, |chunk| {
                            // Simulate some audio processing
                            chunk.iter().map(|x| x.abs()).sum::<f32>() / chunk.len() as f32
                        })
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vector Operations");
    group.measurement_time(Duration::from_secs(8));

    let data_sizes = vec![1000, 10000, 100000];

    for size in data_sizes {
        let data_a: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
        let data_b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).cos()).collect();

        // Scalar dot product
        group.bench_with_input(
            BenchmarkId::new("Scalar_DotProduct", size),
            &(data_a.clone(), data_b.clone()),
            |bench, (a, b)| {
                bench.iter(|| {
                    let result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                    black_box(result)
                })
            },
        );

        // Vector addition
        group.bench_with_input(
            BenchmarkId::new("Scalar_VectorAdd", size),
            &(data_a.clone(), data_b.clone()),
            |bench, (a, b)| {
                bench.iter(|| {
                    let result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
                    black_box(result)
                })
            },
        );

        // Chunked vector operations
        group.bench_with_input(
            BenchmarkId::new("Chunked_VectorAdd", size),
            &(data_a.clone(), data_b.clone()),
            |bench, (a, b)| {
                bench.iter(|| {
                    let paired_data: Vec<_> = a.iter().zip(b.iter()).collect();
                    let result = process_audio_chunks(&paired_data, 1000, |chunk| {
                        chunk.iter().map(|(x, y)| *x + *y).collect::<Vec<f32>>()
                    });
                    let flattened: Vec<f32> = result.into_iter().flatten().collect();
                    black_box(flattened)
                })
            },
        );
    }
    group.finish();
}

fn bench_spectral_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spectral Features");
    group.measurement_time(Duration::from_secs(10));

    let spectrum_sizes = vec![256, 512, 1024, 2048];

    for size in spectrum_sizes {
        let spectrum: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).exp()).collect();

        // CPU spectral features calculation
        group.bench_with_input(
            BenchmarkId::new("CPU_SpectralFeatures", size),
            &spectrum,
            |bench, spectrum| {
                bench.iter(|| {
                    // Calculate spectral centroid and spread
                    let total_energy: f32 = spectrum.iter().sum();
                    let centroid = spectrum
                        .iter()
                        .enumerate()
                        .map(|(i, &mag)| i as f32 * mag)
                        .sum::<f32>()
                        / total_energy;

                    let spread = spectrum
                        .iter()
                        .enumerate()
                        .map(|(i, &mag)| (i as f32 - centroid).powi(2) * mag)
                        .sum::<f32>()
                        / total_energy;

                    black_box((centroid, spread))
                })
            },
        );
    }
    group.finish();
}

fn bench_memory_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Allocation Patterns");
    group.measurement_time(Duration::from_secs(8));

    let sizes = vec![1000, 5000, 10000, 20000];

    for size in sizes {
        // Pre-allocated vectors
        group.bench_with_input(
            BenchmarkId::new("PreAllocated", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    let mut result = Vec::with_capacity(size);
                    for i in 0..size {
                        result.push((i as f32).sin());
                    }
                    black_box(result)
                })
            },
        );

        // Growing vectors
        group.bench_with_input(BenchmarkId::new("Growing", size), &size, |bench, &size| {
            bench.iter(|| {
                let mut result = Vec::new();
                for i in 0..size {
                    result.push((i as f32).sin());
                }
                black_box(result)
            })
        });

        // Iterator collect
        group.bench_with_input(
            BenchmarkId::new("IteratorCollect", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    let result: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    gpu_benches,
    bench_correlation_implementations,
    bench_fft_performance,
    bench_parallel_batch_processing,
    bench_chunked_audio_processing,
    bench_vector_operations,
    bench_spectral_features,
    bench_memory_allocation_patterns
);
criterion_main!(gpu_benches);
