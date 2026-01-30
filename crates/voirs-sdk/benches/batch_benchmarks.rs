//! Batch processing performance benchmarks.
//!
//! This benchmark suite measures the performance characteristics of the batch
//! processing system, including:
//! - Throughput scaling with batch size
//! - Concurrency impact on performance
//! - Voice switching overhead
//! - Custom parameter (speed/pitch) overhead
//! - Scheduling strategy comparison
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench batch_benchmarks
//! ```
//!
//! ## Expected Results
//!
//! - Small batches (1-10): ~10-50ms per request
//! - Medium batches (10-100): ~5-20ms per request (parallel benefit)
//! - Large batches (100-1000): ~2-10ms per request (maximum parallelization)
//! - Voice switching: +5-15ms overhead per switch
//! - Custom parameters: +1-3ms overhead per request

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use tokio::runtime::Runtime;
use voirs_sdk::batch::{BatchConfig, BatchProcessor, BatchRequest, SchedulingStrategy};
use voirs_sdk::VoirsPipelineBuilder;

/// Create a test pipeline for benchmarking
fn create_test_pipeline() -> Arc<voirs_sdk::VoirsPipeline> {
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();
        Arc::new(pipeline)
    })
}

/// Benchmark batch processing throughput with different batch sizes
fn bench_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_throughput");

    // Test batch sizes: 1, 5, 10, 25, 50, 100
    let batch_sizes = [1, 5, 10, 25, 50, 100];

    for &size in &batch_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let rt = Runtime::new().unwrap();
            let pipeline = create_test_pipeline();
            let config = BatchConfig::default();
            let processor = BatchProcessor::new(pipeline, config);

            b.to_async(&rt).iter(|| async {
                let requests: Vec<_> = (0..size)
                    .map(|i| BatchRequest::new(format!("Test sentence {}", i), None))
                    .collect();

                black_box(processor.process(requests).await.unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark impact of different concurrency levels
fn bench_concurrency_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_concurrency");

    // Test concurrency levels: 1, 2, 4, 8
    let concurrency_levels = [1, 2, 4, 8];
    let batch_size = 20; // Fixed batch size for comparison

    for &concurrency in &concurrency_levels {
        group.bench_with_input(
            BenchmarkId::from_parameter(concurrency),
            &concurrency,
            |b, &concurrency| {
                let rt = Runtime::new().unwrap();
                let pipeline = create_test_pipeline();
                let config = BatchConfig {
                    max_concurrency: concurrency,
                    ..Default::default()
                };
                let processor = BatchProcessor::new(pipeline, config);

                b.to_async(&rt).iter(|| async {
                    let requests: Vec<_> = (0..batch_size)
                        .map(|i| BatchRequest::new(format!("Test sentence {}", i), None))
                        .collect();

                    black_box(processor.process(requests).await.unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark voice switching overhead in batch processing
fn bench_voice_switching(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_voice_switching");

    let rt = Runtime::new().unwrap();
    let pipeline = create_test_pipeline();
    let config = BatchConfig::default();
    let processor = BatchProcessor::new(pipeline, config);

    // Baseline: no voice switching
    group.bench_function("no_voice_switch", |b| {
        b.to_async(&rt).iter(|| async {
            let requests: Vec<_> = (0..10)
                .map(|i| BatchRequest::new(format!("Test sentence {}", i), None))
                .collect();

            black_box(processor.process(requests).await.unwrap());
        });
    });

    // With voice switching on every request
    group.bench_function("all_voice_switch", |b| {
        b.to_async(&rt).iter(|| async {
            let requests: Vec<_> = (0..10)
                .map(|i| {
                    let voice = format!("voice-{}", i % 3);
                    BatchRequest::new(format!("Test sentence {}", i), Some(&voice))
                })
                .collect();

            black_box(processor.process(requests).await.unwrap());
        });
    });

    // Mixed: some voice switching
    group.bench_function("mixed_voice_switch", |b| {
        b.to_async(&rt).iter(|| async {
            let requests: Vec<_> = (0..10)
                .map(|i| {
                    if i % 3 == 0 {
                        let voice = format!("voice-{}", i / 3);
                        BatchRequest::new(format!("Test sentence {}", i), Some(&voice))
                    } else {
                        BatchRequest::new(format!("Test sentence {}", i), None)
                    }
                })
                .collect();

            black_box(processor.process(requests).await.unwrap());
        });
    });

    group.finish();
}

/// Benchmark custom parameter (speed/pitch) overhead
fn bench_custom_parameters(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_custom_params");

    let rt = Runtime::new().unwrap();
    let pipeline = create_test_pipeline();
    let config = BatchConfig::default();
    let processor = BatchProcessor::new(pipeline, config);

    // Baseline: default parameters
    group.bench_function("default_params", |b| {
        b.to_async(&rt).iter(|| async {
            let requests: Vec<_> = (0..10)
                .map(|i| BatchRequest::new(format!("Test sentence {}", i), None))
                .collect();

            black_box(processor.process(requests).await.unwrap());
        });
    });

    // Speed only
    group.bench_function("custom_speed", |b| {
        b.to_async(&rt).iter(|| async {
            let requests: Vec<_> = (0..10)
                .map(|i| BatchRequest::new(format!("Test sentence {}", i), None).with_speed(1.2))
                .collect();

            black_box(processor.process(requests).await.unwrap());
        });
    });

    // Pitch only
    group.bench_function("custom_pitch", |b| {
        b.to_async(&rt).iter(|| async {
            let requests: Vec<_> = (0..10)
                .map(|i| BatchRequest::new(format!("Test sentence {}", i), None).with_pitch(2.0))
                .collect();

            black_box(processor.process(requests).await.unwrap());
        });
    });

    // Both speed and pitch
    group.bench_function("custom_speed_and_pitch", |b| {
        b.to_async(&rt).iter(|| async {
            let requests: Vec<_> = (0..10)
                .map(|i| {
                    BatchRequest::new(format!("Test sentence {}", i), None)
                        .with_speed(1.2)
                        .with_pitch(2.0)
                })
                .collect();

            black_box(processor.process(requests).await.unwrap());
        });
    });

    // Variable parameters (different for each request)
    group.bench_function("variable_params", |b| {
        b.to_async(&rt).iter(|| async {
            let requests: Vec<_> = (0..10)
                .map(|i| {
                    let speed = 0.8 + (i as f32 * 0.1);
                    let pitch = -3.0 + (i as f32 * 0.6);
                    BatchRequest::new(format!("Test sentence {}", i), None)
                        .with_speed(speed)
                        .with_pitch(pitch)
                })
                .collect();

            black_box(processor.process(requests).await.unwrap());
        });
    });

    group.finish();
}

/// Benchmark different scheduling strategies
fn bench_scheduling_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_scheduling");

    let strategies = [
        ("fifo", SchedulingStrategy::FIFO),
        ("priority", SchedulingStrategy::PriorityBased),
        ("load_balanced", SchedulingStrategy::LoadBalanced),
        ("shortest_first", SchedulingStrategy::ShortestFirst),
        ("adaptive", SchedulingStrategy::Adaptive),
    ];

    for (name, strategy) in &strategies {
        group.bench_function(*name, |b| {
            let rt = Runtime::new().unwrap();
            let pipeline = create_test_pipeline();
            let config = BatchConfig {
                scheduling_strategy: *strategy,
                ..Default::default()
            };
            let processor = BatchProcessor::new(pipeline, config);

            b.to_async(&rt).iter(|| async {
                // Create requests with varying priorities and text lengths
                let requests: Vec<_> = (0..20)
                    .map(|i| {
                        let text = if i % 3 == 0 {
                            "Short text"
                        } else if i % 3 == 1 {
                            "Medium length text for synthesis testing"
                        } else {
                            "This is a longer text that will take more time to process through the synthesis pipeline"
                        };
                        BatchRequest::new(text, None).with_priority(i % 5)
                    })
                    .collect();

                black_box(processor.process(requests).await.unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark retry mechanism overhead
fn bench_retry_mechanism(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_retry");

    let rt = Runtime::new().unwrap();
    let pipeline = create_test_pipeline();

    // With retries enabled
    group.bench_function("retry_enabled", |b| {
        let config = BatchConfig {
            retry_failed: true,
            max_retries: 3,
            ..Default::default()
        };
        let processor = BatchProcessor::new(pipeline.clone(), config);

        b.to_async(&rt).iter(|| async {
            let requests: Vec<_> = (0..10)
                .map(|i| BatchRequest::new(format!("Test sentence {}", i), None))
                .collect();

            black_box(processor.process(requests).await.unwrap());
        });
    });

    // With retries disabled
    group.bench_function("retry_disabled", |b| {
        let config = BatchConfig {
            retry_failed: false,
            ..Default::default()
        };
        let processor = BatchProcessor::new(pipeline.clone(), config);

        b.to_async(&rt).iter(|| async {
            let requests: Vec<_> = (0..10)
                .map(|i| BatchRequest::new(format!("Test sentence {}", i), None))
                .collect();

            black_box(processor.process(requests).await.unwrap());
        });
    });

    group.finish();
}

/// Benchmark memory efficiency with large batches
fn bench_large_batch_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_large_memory");
    group.sample_size(10); // Reduce sample size for large batches

    let batch_sizes = [100, 200, 500];

    for &size in &batch_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let rt = Runtime::new().unwrap();
            let pipeline = create_test_pipeline();
            let config = BatchConfig {
                max_batch_size: size,
                memory_limit_mb: 2048, // 2GB limit
                ..Default::default()
            };
            let processor = BatchProcessor::new(pipeline, config);

            b.to_async(&rt).iter(|| async {
                let requests: Vec<_> = (0..size)
                    .map(|i| {
                        BatchRequest::new(
                            format!("This is test sentence number {} for batch processing", i),
                            None,
                        )
                    })
                    .collect();

                black_box(processor.process(requests).await.unwrap());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_batch_sizes,
    bench_concurrency_levels,
    bench_voice_switching,
    bench_custom_parameters,
    bench_scheduling_strategies,
    bench_retry_mechanism,
    bench_large_batch_memory,
);

criterion_main!(benches);
