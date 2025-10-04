//! Performance Standards Benchmark for VoiRS Evaluation v0.1.0
//!
//! This benchmark validates the performance requirements for version 0.1.0:
//! - Real-time factor < 0.1 for all metrics
//! - Memory usage < 1GB for batch processing
//! - Parallel efficiency > 80% on multi-core
//! - Statistical test reliability and precision

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use voirs_evaluation::prelude::*;
use voirs_evaluation::quality::{mcd::*, pesq::*, stoi::*};
use voirs_evaluation::statistical::*;
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

/// Performance Standard 1: Real-time factor < 0.1 for all metrics
fn bench_real_time_factor_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Performance: Real-time Factor < 0.1");
    group.measurement_time(Duration::from_secs(20));

    let audio_durations = vec![1.0, 2.0, 5.0]; // seconds

    for duration in audio_durations {
        let samples = (duration * 16000.0) as usize;
        let test_audio = generate_test_audio(samples, 16000);
        let reference_audio = generate_test_audio(samples, 16000);

        group.bench_with_input(
            BenchmarkId::new("QualityEvaluation", format!("{}s", duration)),
            &(test_audio, reference_audio, duration),
            |bench, (test, reference, duration)| {
                bench.iter(|| {
                    rt.block_on(async {
                        let start_time = Instant::now();

                        let quality_evaluator = QualityEvaluator::new().await.unwrap();
                        let result = quality_evaluator
                            .evaluate_quality(black_box(test), Some(black_box(reference)), None)
                            .await
                            .unwrap();

                        let processing_time = start_time.elapsed().as_secs_f32();
                        let real_time_factor = processing_time / duration;

                        // Log but don't assert during benchmark
                        if real_time_factor >= 0.1 {
                            eprintln!("Warning: Real-time factor {} exceeds 0.1 threshold for {} second audio",
                                real_time_factor, duration);
                        }

                        black_box((result, processing_time, real_time_factor))
                    })
                })
            },
        );
    }
    group.finish();
}

/// Performance Standard 2: Memory usage < 1GB for batch processing
fn bench_memory_usage_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Performance: Memory Usage < 1GB");
    group.measurement_time(Duration::from_secs(30));

    let batch_sizes = vec![20, 50, 100]; // More conservative batch sizes

    for batch_size in batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("BatchProcessing", batch_size),
            &batch_size,
            |bench, &size| {
                bench.iter(|| {
                    rt.block_on(async {
                        let quality_evaluator = QualityEvaluator::new().await.unwrap();

                        // Generate batch of audio files (3 seconds each)
                        let mut test_audios = Vec::new();
                        let mut reference_audios = Vec::new();

                        for _ in 0..size {
                            test_audios.push(generate_test_audio(48000, 16000)); // 3 seconds
                            reference_audios.push(generate_test_audio(48000, 16000));
                        }

                        // Process batch
                        let mut results = Vec::new();
                        for (test, reference) in test_audios.iter().zip(reference_audios.iter()) {
                            let result = quality_evaluator
                                .evaluate_quality(black_box(test), Some(black_box(reference)), None)
                                .await
                                .unwrap();
                            results.push(result);
                        }

                        // Estimate memory usage (simplified)
                        let audio_memory = (test_audios.len() + reference_audios.len()) * 48000 * 4; // 4 bytes per f32
                        let estimated_memory_gb = audio_memory as f32 / (1024.0 * 1024.0 * 1024.0);

                        // Log but don't assert during benchmark
                        if estimated_memory_gb >= 1.0 {
                            eprintln!("Warning: Memory usage {}GB exceeds 1GB threshold for batch size {}",
                                estimated_memory_gb, size);
                        }

                        black_box((results, estimated_memory_gb))
                    })
                })
            },
        );
    }
    group.finish();
}

/// Performance Standard 3: Parallel efficiency > 80% on multi-core
fn bench_parallel_efficiency_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Performance: Parallel Efficiency > 80%");
    group.measurement_time(Duration::from_secs(15));

    let data_sizes = vec![10000, 50000, 100000];

    for data_size in data_sizes {
        group.bench_with_input(
            BenchmarkId::new("ParallelCorrelation", data_size),
            &data_size,
            |bench, &size| {
                bench.iter(|| {
                    let data_a: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
                    let data_b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).cos()).collect();

                    // Measure sequential performance
                    let start_sequential = Instant::now();
                    let sequential_result =
                        voirs_evaluation::calculate_correlation(&data_a, &data_b);
                    let sequential_time = start_sequential.elapsed().as_secs_f32();

                    // Measure parallel performance
                    let start_parallel = Instant::now();
                    let parallel_result =
                        voirs_evaluation::performance::parallel_correlation(&data_a, &data_b);
                    let parallel_time = start_parallel.elapsed().as_secs_f32();

                    // Calculate speedup and efficiency
                    let speedup = sequential_time / parallel_time.max(0.001); // Avoid division by zero
                    let num_cores = num_cpus::get() as f32;
                    let efficiency = speedup / num_cores;

                    // Log but don't assert during benchmark
                    if efficiency < 0.8 {
                        eprintln!(
                            "Warning: Parallel efficiency {}% below 80% threshold",
                            efficiency * 100.0
                        );
                    }

                    black_box((sequential_result, parallel_result, speedup, efficiency))
                })
            },
        );
    }
    group.finish();
}

/// Precision Standard: MCD calculation precision < 0.01 dB variance
fn bench_mcd_precision_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Performance: MCD Precision < 0.01 dB");
    group.measurement_time(Duration::from_secs(10));

    let repetitions = vec![20, 50, 100];

    for reps in repetitions {
        group.bench_with_input(
            BenchmarkId::new("MCDRepeatability", reps),
            &reps,
            |bench, &repetitions| {
                bench.iter(|| {
                    rt.block_on(async {
                        let mut mcd_evaluator = MCDEvaluator::new(16000).unwrap();
                        let test_audio = generate_test_audio(16000, 16000); // 1 second
                        let reference_audio = generate_test_audio(16000, 16000);

                        let mut mcd_scores = Vec::new();

                        // Repeat MCD calculation multiple times on same audio
                        for _ in 0..repetitions {
                            let result = mcd_evaluator
                                .calculate_mcd_simple(&test_audio, &reference_audio)
                                .await
                                .unwrap();
                            mcd_scores.push(result);
                        }

                        // Calculate variance in MCD scores
                        let mean = mcd_scores.iter().sum::<f32>() / mcd_scores.len() as f32;
                        let variance = mcd_scores
                            .iter()
                            .map(|score| (score - mean).powi(2))
                            .sum::<f32>()
                            / mcd_scores.len() as f32;

                        let std_deviation = variance.sqrt();

                        // Log but don't assert during benchmark
                        if std_deviation >= 0.01 {
                            eprintln!(
                                "Warning: MCD precision {} dB exceeds 0.01 dB threshold",
                                std_deviation
                            );
                        }

                        black_box((mcd_scores, variance, std_deviation))
                    })
                })
            },
        );
    }
    group.finish();
}

/// Statistical Test Reliability: Type I error rate validation
fn bench_statistical_reliability_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Performance: Statistical Test Reliability");
    group.measurement_time(Duration::from_secs(10));

    let test_counts = vec![100, 500, 1000];

    for test_count in test_counts {
        group.bench_with_input(
            BenchmarkId::new("TypeIErrorRate", test_count),
            &test_count,
            |bench, &count| {
                bench.iter(|| {
                    let analyzer = StatisticalAnalyzer::new();
                    let mut false_positives = 0;
                    let significance_level = 0.05;

                    // Perform paired t-tests on similar distributions (null hypothesis true)
                    for _ in 0..count {
                        let sample_size = 30;
                        let data_a: Vec<f32> = (0..sample_size)
                            .map(|_| scirs2_core::random::random::<f32>())
                            .collect();
                        let data_b: Vec<f32> = (0..sample_size)
                            .map(|_| scirs2_core::random::random::<f32>())
                            .collect();

                        if let Ok(result) = analyzer.paired_t_test(&data_a, &data_b, None) {
                            if result.p_value < significance_level {
                                false_positives += 1;
                            }
                        }
                    }

                    let type_i_error_rate = false_positives as f32 / count as f32;

                    // Log but don't assert during benchmark (allow some variation)
                    if type_i_error_rate > 0.08 {
                        eprintln!(
                            "Warning: Type I error rate {} exceeds expected range around 0.05",
                            type_i_error_rate
                        );
                    }

                    black_box((false_positives, type_i_error_rate))
                })
            },
        );
    }
    group.finish();
}

/// Overall Integration Performance Test
fn bench_integration_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Performance: Integration Test");
    group.measurement_time(Duration::from_secs(25));

    let scenarios = vec![(10, 1.0), (20, 2.0), (50, 5.0)]; // (batch_size, audio_duration)

    for (batch_size, duration) in scenarios {
        let samples = (duration * 16000.0) as usize;

        group.bench_with_input(
            BenchmarkId::new("FullWorkflow", format!("{}x{}s", batch_size, duration)),
            &(batch_size, samples),
            |bench, &(size, samples)| {
                bench.iter(|| {
                    rt.block_on(async {
                        let start_time = Instant::now();

                        let quality_evaluator = QualityEvaluator::new().await.unwrap();

                        // Generate test batch
                        let mut results = Vec::new();
                        for _ in 0..size {
                            let test_audio = generate_test_audio(samples, 16000);
                            let reference_audio = generate_test_audio(samples, 16000);

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

                        let total_time = start_time.elapsed().as_secs_f32();
                        let total_audio_duration = size as f32 * (samples as f32 / 16000.0);
                        let real_time_factor = total_time / total_audio_duration;

                        // Log performance metrics
                        eprintln!(
                            "Processed {}x{:.1}s audio in {:.2}s (RTF: {:.3})",
                            size,
                            samples as f32 / 16000.0,
                            total_time,
                            real_time_factor
                        );

                        black_box((results, total_time, real_time_factor))
                    })
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    performance_standards,
    bench_real_time_factor_validation,
    bench_memory_usage_validation,
    bench_parallel_efficiency_validation,
    bench_mcd_precision_validation,
    bench_statistical_reliability_validation,
    bench_integration_performance
);
criterion_main!(performance_standards);
