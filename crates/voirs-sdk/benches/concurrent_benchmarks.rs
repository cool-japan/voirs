//! Concurrent Operations Benchmarks
//!
//! This benchmark suite measures performance of concurrent synthesis operations,
//! thread safety, and scalability under load.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::Arc;
use std::time::Duration;
use voirs_sdk::prelude::*;

/// Helper to create a shared pipeline for concurrent benchmarks
async fn create_shared_pipeline() -> Arc<VoirsPipeline> {
    let pipeline = VoirsPipelineBuilder::new()
        .with_quality(QualityLevel::Medium)
        .with_test_mode(true)
        .build()
        .await
        .unwrap();
    Arc::new(pipeline)
}

/// Benchmark concurrent synthesis operations
fn bench_concurrent_synthesis(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_synthesis");
    group.measurement_time(Duration::from_secs(20));

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_shared_pipeline());

    let test_texts = vec![
        "First concurrent synthesis task.",
        "Second concurrent synthesis task.",
        "Third concurrent synthesis task.",
        "Fourth concurrent synthesis task.",
    ];

    let concurrency_levels = vec![1, 2, 4, 8, 16];

    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::new("tasks", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.to_async(&runtime).iter(|| async {
                    let mut handles = vec![];

                    for i in 0..concurrency {
                        let pipeline = Arc::clone(&pipeline);
                        let text = test_texts[i % test_texts.len()];

                        let handle = tokio::spawn(async move {
                            let audio = pipeline.synthesize(text).await.unwrap();
                            black_box(audio);
                        });

                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.await.unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent voice switching
fn bench_concurrent_voice_switching(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_voice_switching");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_shared_pipeline());

    let voices = vec!["voice-1", "voice-2", "voice-3", "voice-4"];

    group.bench_function("parallel_switches", |b| {
        b.to_async(&runtime).iter(|| async {
            let mut handles = vec![];

            for voice in &voices {
                let pipeline = Arc::clone(&pipeline);
                let voice = voice.to_string();

                let handle = tokio::spawn(async move {
                    pipeline.set_voice(&voice).await.unwrap();
                });

                handles.push(handle);
            }

            for handle in handles {
                handle.await.unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark mixed concurrent operations
fn bench_mixed_concurrent_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_concurrent_ops");
    group.measurement_time(Duration::from_secs(15));

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_shared_pipeline());

    group.bench_function("synthesis_and_queries", |b| {
        b.to_async(&runtime).iter(|| async {
            let mut handles = vec![];

            // Synthesis tasks
            for i in 0..4 {
                let pipeline = Arc::clone(&pipeline);
                let text = format!("Synthesis task {}", i);

                let handle = tokio::spawn(async move {
                    let audio = pipeline.synthesize(&text).await.unwrap();
                    black_box(audio);
                });

                handles.push(handle);
            }

            // Query tasks
            for _ in 0..4 {
                let pipeline = Arc::clone(&pipeline);

                let handle = tokio::spawn(async move {
                    let voices = pipeline.list_voices().await.unwrap();
                    black_box(voices);
                });

                handles.push(handle);
            }

            // State queries
            for _ in 0..4 {
                let pipeline_clone = Arc::clone(&pipeline);

                let handle = tokio::spawn(async move {
                    let state = pipeline_clone.get_state().await;
                    black_box(state);
                });

                handles.push(handle);
            }

            for handle in handles {
                handle.await.unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark task queue saturation
fn bench_task_queue_saturation(c: &mut Criterion) {
    let mut group = c.benchmark_group("task_queue_saturation");
    group.measurement_time(Duration::from_secs(20));

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_shared_pipeline());

    let queue_sizes = vec![10, 50, 100, 200];

    for queue_size in queue_sizes {
        group.bench_with_input(
            BenchmarkId::new("queue_size", queue_size),
            &queue_size,
            |b, &queue_size| {
                b.to_async(&runtime).iter(|| async {
                    let mut handles = vec![];

                    for i in 0..queue_size {
                        let pipeline = Arc::clone(&pipeline);
                        let text = format!("Task {}", i);

                        let handle = tokio::spawn(async move {
                            let audio = pipeline.synthesize(&text).await.unwrap();
                            black_box(audio);
                        });

                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.await.unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark reader-writer patterns
fn bench_read_write_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_write_patterns");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_shared_pipeline());

    group.bench_function("heavy_read", |b| {
        b.to_async(&runtime).iter(|| async {
            let mut handles = vec![];

            // Many readers
            for _ in 0..20 {
                let pipeline_clone = Arc::clone(&pipeline);

                let handle = tokio::spawn(async move {
                    let voices = pipeline_clone.list_voices().await.unwrap();
                    black_box(voices);
                });

                handles.push(handle);
            }

            // Few writers
            for i in 0..2 {
                let pipeline_clone = Arc::clone(&pipeline);
                let text = format!("Write task {}", i);

                let handle = tokio::spawn(async move {
                    let audio = pipeline_clone.synthesize(&text).await.unwrap();
                    black_box(audio);
                });

                handles.push(handle);
            }

            for handle in handles {
                handle.await.unwrap();
            }
        });
    });

    group.bench_function("balanced", |b| {
        b.to_async(&runtime).iter(|| async {
            let mut handles = vec![];

            // Balanced readers and writers
            for i in 0..10 {
                let pipeline_read = Arc::clone(&pipeline);

                let handle = tokio::spawn(async move {
                    let voices = pipeline_read.list_voices().await.unwrap();
                    black_box(voices);
                });

                handles.push(handle);

                let pipeline_write = Arc::clone(&pipeline);
                let text = format!("Task {}", i);

                let handle = tokio::spawn(async move {
                    let audio = pipeline_write.synthesize(&text).await.unwrap();
                    black_box(audio);
                });

                handles.push(handle);
            }

            for handle in handles {
                handle.await.unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark contention under load
fn bench_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("contention");
    group.measurement_time(Duration::from_secs(15));

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_shared_pipeline());

    group.bench_function("high_contention", |b| {
        b.to_async(&runtime).iter(|| async {
            let mut handles = vec![];

            // All tasks trying to synthesize the same text
            for _ in 0..16 {
                let pipeline = Arc::clone(&pipeline);

                let handle = tokio::spawn(async move {
                    let audio = pipeline.synthesize("Contention test").await.unwrap();
                    black_box(audio);
                });

                handles.push(handle);
            }

            for handle in handles {
                handle.await.unwrap();
            }
        });
    });

    group.finish();
}

criterion_group!(
    concurrent_benches,
    bench_concurrent_synthesis,
    bench_concurrent_voice_switching,
    bench_mixed_concurrent_ops,
    bench_task_queue_saturation,
    bench_read_write_patterns,
    bench_contention
);

criterion_main!(concurrent_benches);
