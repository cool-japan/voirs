use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use tokio::runtime::Runtime;
use voirs_evaluation::prelude::*;
use voirs_evaluation::statistical::*;
use voirs_sdk::AudioBuffer;

fn generate_test_audio(length: usize, sample_rate: u32) -> AudioBuffer {
    let samples: Vec<f32> = (0..length)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                + 0.2 * rand::random::<f32>()
                - 0.1
        })
        .collect();

    AudioBuffer::mono(samples, sample_rate)
}

fn bench_statistical_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("Statistical Tests");
    group.measurement_time(Duration::from_secs(5));

    let sample_sizes = vec![100, 500, 1000, 5000];

    for sample_size in sample_sizes {
        let data_a: Vec<f32> = (0..sample_size).map(|_| rand::random::<f32>()).collect();
        let data_b: Vec<f32> = (0..sample_size).map(|_| rand::random::<f32>()).collect();

        let analyzer = StatisticalAnalyzer::new();

        group.bench_with_input(
            BenchmarkId::new("PairedTTest", sample_size),
            &(data_a.clone(), data_b.clone()),
            |bench, (data1, data2)| {
                bench.iter(|| analyzer.paired_t_test(black_box(data1), black_box(data2), None))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("MannWhitneyU", sample_size),
            &(data_a.clone(), data_b.clone()),
            |bench, (data1, data2)| {
                bench
                    .iter(|| analyzer.mann_whitney_u_test(black_box(data1), black_box(data2), None))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Correlation", sample_size),
            &(data_a.clone(), data_b.clone()),
            |bench, (data1, data2)| {
                bench.iter(|| analyzer.correlation_test(black_box(data1), black_box(data2)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("BootstrapCI", sample_size),
            &data_a,
            |bench, data| {
                bench.iter(|| {
                    fn mean(x: &[f32]) -> f32 {
                        x.iter().sum::<f32>() / x.len() as f32
                    }
                    analyzer.bootstrap_confidence_interval(black_box(data), mean)
                })
            },
        );
    }
    group.finish();
}

fn bench_quality_evaluator(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Quality Evaluator");
    group.measurement_time(Duration::from_secs(15));

    let audio_lengths = vec![16000, 32000, 48000]; // 1s, 2s, 3s at 16kHz

    let evaluator = rt.block_on(async { QualityEvaluator::new().await.unwrap() });

    for length in audio_lengths {
        let test_audio = generate_test_audio(length, 16000);
        let reference_audio = generate_test_audio(length, 16000);

        group.bench_with_input(
            BenchmarkId::new("QualityEvaluation", format!("{length}samples")),
            &(test_audio.clone(), reference_audio.clone()),
            |bench, (test, reference)| {
                bench.iter(|| {
                    rt.block_on(async {
                        evaluator
                            .evaluate_quality(black_box(test), Some(black_box(reference)), None)
                            .await
                    })
                })
            },
        );

        // Test without reference (self-evaluation)
        group.bench_with_input(
            BenchmarkId::new("SelfEvaluation", format!("{length}samples")),
            &test_audio.clone(),
            |bench, test| {
                bench.iter(|| {
                    rt.block_on(async {
                        evaluator
                            .evaluate_quality(black_box(test), None, None)
                            .await
                    })
                })
            },
        );
    }
    group.finish();
}

fn bench_batch_evaluation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Batch Evaluation");
    group.measurement_time(Duration::from_secs(20));

    let batch_sizes = vec![5, 10, 20];
    let evaluator = rt.block_on(async { QualityEvaluator::new().await.unwrap() });

    for batch_size in batch_sizes {
        let mut test_audios = Vec::new();
        let mut reference_audios = Vec::new();

        for _ in 0..batch_size {
            test_audios.push(generate_test_audio(16000, 16000)); // 1 second
            reference_audios.push(generate_test_audio(16000, 16000));
        }

        group.bench_with_input(
            BenchmarkId::new("BatchEvaluation", batch_size),
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
                        black_box(results)
                    })
                })
            },
        );
    }
    group.finish();
}

fn bench_performance_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Performance Optimizations");
    group.measurement_time(Duration::from_secs(5));

    let data_sizes = vec![1000, 10000, 100000];

    for data_size in data_sizes {
        let data_a: Vec<f32> = (0..data_size).map(|_| rand::random::<f32>()).collect();
        let data_b: Vec<f32> = (0..data_size).map(|_| rand::random::<f32>()).collect();

        // Sequential correlation
        group.bench_with_input(
            BenchmarkId::new("SequentialCorrelation", data_size),
            &(data_a.clone(), data_b.clone()),
            |bench, (x, y)| {
                bench.iter(|| voirs_evaluation::calculate_correlation(black_box(x), black_box(y)))
            },
        );

        // Parallel correlation
        group.bench_with_input(
            BenchmarkId::new("ParallelCorrelation", data_size),
            &(data_a.clone(), data_b.clone()),
            |bench, (x, y)| {
                bench.iter(|| {
                    voirs_evaluation::performance::parallel_correlation(black_box(x), black_box(y))
                })
            },
        );

        // Chunked processing
        group.bench_with_input(
            BenchmarkId::new("ChunkedProcessing", data_size),
            &data_a,
            |bench, data| {
                bench.iter(|| {
                    voirs_evaluation::performance::process_audio_chunks(
                        black_box(data),
                        1000,
                        |chunk| chunk.iter().sum::<f32>(),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_correlation_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Correlation Calculations");
    group.measurement_time(Duration::from_secs(8));

    let data_sizes = vec![1000, 10000, 100000];

    for data_size in data_sizes {
        let data_a: Vec<f32> = (0..data_size).map(|i| (i as f32 * 0.001).sin()).collect();
        let data_b: Vec<f32> = (0..data_size).map(|i| (i as f32 * 0.001).cos()).collect();

        // Standard correlation
        group.bench_with_input(
            BenchmarkId::new("StandardCorrelation", data_size),
            &(data_a.clone(), data_b.clone()),
            |bench, (x, y)| {
                bench.iter(|| voirs_evaluation::calculate_correlation(black_box(x), black_box(y)))
            },
        );

        // Parallel correlation
        group.bench_with_input(
            BenchmarkId::new("ParallelCorrelation", data_size),
            &(data_a.clone(), data_b.clone()),
            |bench, (x, y)| {
                bench.iter(|| {
                    voirs_evaluation::performance::parallel_correlation(black_box(x), black_box(y))
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
    benches,
    bench_statistical_tests,
    bench_quality_evaluator,
    bench_batch_evaluation,
    bench_performance_optimizations,
    bench_correlation_calculations,
    bench_audio_buffer_operations
);
criterion_main!(benches);
