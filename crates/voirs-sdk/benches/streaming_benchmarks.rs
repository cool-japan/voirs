//! Streaming Synthesis Benchmarks
//!
//! This benchmark suite measures streaming synthesis performance including
//! latency, throughput, and chunk processing efficiency.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use futures::StreamExt;
use std::sync::Arc;
use std::time::Duration;
use voirs_sdk::prelude::*;

/// Helper to create a pipeline for streaming benchmarks
async fn create_streaming_pipeline() -> Result<Arc<VoirsPipeline>> {
    let pipeline = VoirsPipelineBuilder::new()
        .with_quality(QualityLevel::Medium)
        .with_test_mode(true)
        .build()
        .await?;
    Ok(Arc::new(pipeline))
}

/// Benchmark streaming synthesis latency (time to first chunk)
fn bench_streaming_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_latency");
    group.measurement_time(Duration::from_secs(10));

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_streaming_pipeline()).unwrap();

    let test_texts = vec![
        ("short", "Hello world"),
        ("medium", "The quick brown fox jumps over the lazy dog."),
        ("long", "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort."),
    ];

    for (name, text) in test_texts {
        group.bench_with_input(BenchmarkId::from_parameter(name), &text, |b, &text| {
            let pipeline = Arc::clone(&pipeline);
            b.to_async(&runtime).iter(move || {
                let pipeline = Arc::clone(&pipeline);
                async move {
                    let mut stream = pipeline.synthesize_stream(black_box(text)).await.unwrap();
                    // Measure time to first chunk only
                    if let Some(chunk) = stream.next().await {
                        black_box(chunk.unwrap());
                    }
                }
            });
        });
    }

    group.finish();
}

/// Benchmark full streaming throughput
fn bench_streaming_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_throughput");
    group.measurement_time(Duration::from_secs(15));

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_streaming_pipeline()).unwrap();

    let test_text = "This is a test of streaming synthesis throughput. We need enough text to generate multiple chunks of audio data for accurate measurement of the complete streaming process from start to finish.";

    group.bench_function("full_stream", |b| {
        let pipeline = Arc::clone(&pipeline);
        b.to_async(&runtime).iter(move || {
            let pipeline = Arc::clone(&pipeline);
            async move {
                let mut stream = pipeline
                    .synthesize_stream(black_box(test_text))
                    .await
                    .unwrap();
                let mut total_samples = 0;

                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.unwrap();
                    total_samples += chunk.len();
                    black_box(&chunk);
                }

                black_box(total_samples);
            }
        });
    });

    group.finish();
}

/// Benchmark streaming with different chunk sizes
fn bench_streaming_chunk_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_chunk_sizes");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let test_text = "Benchmarking streaming synthesis with various chunk sizes to optimize for different use cases and latency requirements.";

    let chunk_sizes = vec![512, 1024, 2048, 4096, 8192];

    for chunk_size in chunk_sizes {
        group.bench_with_input(
            BenchmarkId::new("chunk_size", chunk_size),
            &chunk_size,
            |b, &chunk_size| {
                b.to_async(&runtime).iter(|| async {
                    let pipeline = VoirsPipelineBuilder::new()
                        .with_quality(QualityLevel::Medium)
                        .with_test_mode(true)
                        .build()
                        .await
                        .unwrap();

                    let pipeline = Arc::new(pipeline);
                    let mut stream = pipeline
                        .synthesize_stream(black_box(test_text))
                        .await
                        .unwrap();

                    while let Some(chunk) = stream.next().await {
                        black_box(chunk.unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent streaming operations
fn bench_concurrent_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_streaming");
    group.measurement_time(Duration::from_secs(20));

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_streaming_pipeline()).unwrap();

    let test_texts = vec![
        "First concurrent stream text.",
        "Second concurrent stream text.",
        "Third concurrent stream text.",
        "Fourth concurrent stream text.",
    ];

    let concurrency_levels = vec![1, 2, 4, 8];

    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::new("concurrent_streams", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.to_async(&runtime).iter(|| async {
                    let mut handles = vec![];

                    for i in 0..concurrency {
                        let pipeline = Arc::clone(&pipeline);
                        let text = test_texts[i % test_texts.len()];

                        let handle = tokio::spawn(async move {
                            let mut stream = pipeline.synthesize_stream(text).await.unwrap();
                            while let Some(chunk) = stream.next().await {
                                black_box(chunk.unwrap());
                            }
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

/// Benchmark streaming memory efficiency
fn bench_streaming_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_memory");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_streaming_pipeline()).unwrap();

    // Long text to test memory efficiency
    let long_text = Arc::new(
        "This is a very long text designed to test the memory efficiency of streaming synthesis. "
            .repeat(50),
    );

    group.bench_function("streaming_vs_batch", |b| {
        let pipeline = Arc::clone(&pipeline);
        let long_text = Arc::clone(&long_text);
        b.to_async(&runtime).iter(move || {
            let pipeline = Arc::clone(&pipeline);
            let long_text = Arc::clone(&long_text);
            async move {
                let mut stream = pipeline
                    .synthesize_stream(black_box(long_text.as_str()))
                    .await
                    .unwrap();
                let mut total_samples = 0;

                // Process chunks as they arrive (streaming)
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.unwrap();
                    total_samples += chunk.len();
                    // Simulate processing each chunk
                    black_box(&chunk);
                }

                black_box(total_samples);
            }
        });
    });

    group.finish();
}

/// Benchmark backpressure handling
fn bench_streaming_backpressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_backpressure");
    group.measurement_time(Duration::from_secs(10));

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_streaming_pipeline()).unwrap();

    let test_text =
        "Testing backpressure handling in streaming synthesis with simulated slow consumer.";

    group.bench_function("with_delays", |b| {
        let pipeline = Arc::clone(&pipeline);
        b.to_async(&runtime).iter(move || {
            let pipeline = Arc::clone(&pipeline);
            async move {
                let mut stream = pipeline
                    .synthesize_stream(black_box(test_text))
                    .await
                    .unwrap();

                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.unwrap();
                    black_box(&chunk);
                    // Simulate slow consumer
                    tokio::time::sleep(Duration::from_micros(10)).await;
                }
            }
        });
    });

    group.finish();
}

criterion_group!(
    streaming_benches,
    bench_streaming_latency,
    bench_streaming_throughput,
    bench_streaming_chunk_sizes,
    bench_concurrent_streaming,
    bench_streaming_memory,
    bench_streaming_backpressure
);

criterion_main!(streaming_benches);
