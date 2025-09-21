use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use tokio::runtime::Runtime;
use voirs_evaluation::prelude::*;
use voirs_sdk::AudioBuffer;

fn generate_test_audio(length: usize, sample_rate: u32) -> AudioBuffer {
    let samples: Vec<f32> = (0..length).map(|i| ((i as f32 * 0.001).sin())).collect();
    AudioBuffer::mono(samples, sample_rate)
}

fn bench_memory_usage_small_batches(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Memory Usage - Small Batches");
    group.measurement_time(Duration::from_secs(10));

    let batch_sizes = vec![1, 5, 10, 20];
    let audio_length = 16000; // 1 second at 16kHz

    let evaluator = rt.block_on(async { QualityEvaluator::new().await.unwrap() });

    for batch_size in batch_sizes {
        let mut test_audios = Vec::new();
        let mut reference_audios = Vec::new();

        for _ in 0..batch_size {
            test_audios.push(generate_test_audio(audio_length, 16000));
            reference_audios.push(generate_test_audio(audio_length, 16000));
        }

        group.bench_with_input(
            BenchmarkId::new("SmallBatch", batch_size),
            &(test_audios, reference_audios),
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
                        // Force memory allocation to be measured
                        black_box(results)
                    })
                })
            },
        );
    }
    group.finish();
}

fn bench_memory_usage_large_audio(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Memory Usage - Large Audio");
    group.measurement_time(Duration::from_secs(15));

    let audio_lengths = vec![
        16000,  // 1 second
        48000,  // 3 seconds
        80000,  // 5 seconds
        160000, // 10 seconds
    ];

    let evaluator = rt.block_on(async { QualityEvaluator::new().await.unwrap() });

    for audio_length in audio_lengths {
        let test_audio = generate_test_audio(audio_length, 16000);
        let reference_audio = generate_test_audio(audio_length, 16000);

        group.bench_with_input(
            BenchmarkId::new("LargeAudio", format!("{audio_length}samples")),
            &(test_audio, reference_audio),
            |bench, (test, reference)| {
                bench.iter(|| {
                    rt.block_on(async {
                        let result = evaluator
                            .evaluate_quality(black_box(test), Some(black_box(reference)), None)
                            .await;
                        black_box(result)
                    })
                })
            },
        );
    }
    group.finish();
}

fn bench_streaming_vs_batch_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Streaming vs Batch Processing");
    group.measurement_time(Duration::from_secs(10));

    let total_samples = 20;
    let audio_length = 16000; // 1 second

    let mut test_audios = Vec::new();
    let mut reference_audios = Vec::new();

    for _ in 0..total_samples {
        test_audios.push(generate_test_audio(audio_length, 16000));
        reference_audios.push(generate_test_audio(audio_length, 16000));
    }

    let evaluator = rt.block_on(async { QualityEvaluator::new().await.unwrap() });

    // Sequential processing - one by one
    group.bench_function("OneByOne", |bench| {
        bench.iter(|| {
            rt.block_on(async {
                let mut results = Vec::new();
                for (test, reference) in test_audios.iter().zip(reference_audios.iter()) {
                    let result = evaluator
                        .evaluate_quality(black_box(test), Some(black_box(reference)), None)
                        .await;
                    results.push(result);
                }
                black_box(results)
            })
        })
    });

    // Chunked processing - in groups of 5
    group.bench_function("Chunked", |bench| {
        bench.iter(|| {
            rt.block_on(async {
                let mut all_results = Vec::new();
                for (test_chunk, ref_chunk) in test_audios.chunks(5).zip(reference_audios.chunks(5))
                {
                    let mut chunk_results = Vec::new();
                    for (test, reference) in test_chunk.iter().zip(ref_chunk.iter()) {
                        let result = evaluator
                            .evaluate_quality(black_box(test), Some(black_box(reference)), None)
                            .await;
                        chunk_results.push(result);
                    }
                    all_results.extend(chunk_results);
                }
                black_box(all_results)
            })
        })
    });

    group.finish();
}

fn bench_parallel_vs_sequential_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel vs Sequential Memory");
    group.measurement_time(Duration::from_secs(10));

    let data_sizes = vec![1000, 10000, 50000];

    for data_size in data_sizes {
        let data_a: Vec<f32> = (0..data_size).map(|i| (i as f32 * 0.001).sin()).collect();
        let data_b: Vec<f32> = (0..data_size)
            .map(|i| ((i + 100) as f32 * 0.001).sin())
            .collect();

        // Sequential processing
        group.bench_with_input(
            BenchmarkId::new("Sequential", data_size),
            &(data_a.clone(), data_b.clone()),
            |bench, (x, y)| {
                bench.iter(|| {
                    // Simulate memory-intensive computation
                    let mut result = Vec::with_capacity(x.len());
                    for (xi, yi) in x.iter().zip(y.iter()) {
                        result.push(xi * yi + xi.powi(2) + yi.powi(2));
                    }
                    black_box(result)
                })
            },
        );

        // Parallel processing using chunked approach
        group.bench_with_input(
            BenchmarkId::new("Chunked", data_size),
            &(data_a.clone(), data_b.clone()),
            |bench, (x, y)| {
                bench.iter(|| {
                    let chunk_size = 1000;
                    let result = voirs_evaluation::performance::process_audio_chunks(
                        &x.iter().zip(y.iter()).collect::<Vec<_>>(),
                        chunk_size,
                        |chunk| {
                            chunk
                                .iter()
                                .map(|(xi, yi)| *xi * *yi + xi.powi(2) + yi.powi(2))
                                .collect::<Vec<f32>>()
                        },
                    );
                    black_box(result)
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

fn bench_correlation_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("Correlation Memory Patterns");
    group.measurement_time(Duration::from_secs(8));

    let data_sizes = vec![1000, 10000, 100000];

    for data_size in data_sizes {
        let data_a: Vec<f32> = (0..data_size).map(|i| (i as f32 * 0.001).sin()).collect();
        let data_b: Vec<f32> = (0..data_size).map(|i| (i as f32 * 0.001).cos()).collect();

        // Sequential correlation
        group.bench_with_input(
            BenchmarkId::new("Sequential", data_size),
            &(data_a.clone(), data_b.clone()),
            |bench, (x, y)| {
                bench.iter(|| voirs_evaluation::calculate_correlation(black_box(x), black_box(y)))
            },
        );

        // Parallel correlation
        group.bench_with_input(
            BenchmarkId::new("Parallel", data_size),
            &(data_a.clone(), data_b.clone()),
            |bench, (x, y)| {
                bench.iter(|| {
                    voirs_evaluation::performance::parallel_correlation(black_box(x), black_box(y))
                })
            },
        );

        // Chunked correlation
        group.bench_with_input(
            BenchmarkId::new("Chunked", data_size),
            &(data_a.clone(), data_b.clone()),
            |bench, (x, y)| {
                bench.iter(|| {
                    let chunk_size = 1000;
                    let results = voirs_evaluation::performance::process_audio_chunks(
                        &x.iter().zip(y.iter()).collect::<Vec<_>>(),
                        chunk_size,
                        |chunk| {
                            let (chunk_x, chunk_y): (Vec<_>, Vec<_>) =
                                chunk.iter().copied().unzip();
                            voirs_evaluation::calculate_correlation(&chunk_x, &chunk_y)
                        },
                    );
                    // Average the chunk correlations (simplified approach)
                    let avg = results.iter().sum::<f32>() / results.len() as f32;
                    black_box(avg)
                })
            },
        );
    }
    group.finish();
}

fn bench_audio_buffer_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Audio Buffer Operations");
    group.measurement_time(Duration::from_secs(8));

    let audio_lengths = vec![1000, 10000, 100000];

    for length in audio_lengths {
        let samples: Vec<f32> = (0..length).map(|i| (i as f32 * 0.001).sin()).collect();

        // Creating AudioBuffer
        group.bench_with_input(
            BenchmarkId::new("CreateBuffer", length),
            &samples,
            |bench, samples| {
                bench.iter(|| {
                    let buffer = AudioBuffer::mono(black_box(samples.clone()), 16000);
                    black_box(buffer)
                })
            },
        );

        let audio_buffer = AudioBuffer::mono(samples.clone(), 16000);

        // Accessing samples
        group.bench_with_input(
            BenchmarkId::new("AccessSamples", length),
            &audio_buffer,
            |bench, buffer| {
                bench.iter(|| {
                    let samples = buffer.samples();
                    let sum: f32 = black_box(samples).iter().sum();
                    black_box(sum)
                })
            },
        );

        // Cloning buffer
        group.bench_with_input(
            BenchmarkId::new("CloneBuffer", length),
            &audio_buffer,
            |bench, buffer| {
                bench.iter(|| {
                    let cloned = black_box(buffer).clone();
                    black_box(cloned)
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    memory_benches,
    bench_memory_usage_small_batches,
    bench_memory_usage_large_audio,
    bench_streaming_vs_batch_processing,
    bench_parallel_vs_sequential_memory,
    bench_memory_allocation_patterns,
    bench_correlation_memory_patterns,
    bench_audio_buffer_operations
);
criterion_main!(memory_benches);
