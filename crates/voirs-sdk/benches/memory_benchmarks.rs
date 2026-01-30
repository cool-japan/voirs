//! Memory Management and Caching Benchmarks
//!
//! This benchmark suite measures memory allocation, cache performance,
//! and resource management efficiency in the VoiRS SDK.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::Arc;
use std::time::Duration;
use voirs_sdk::prelude::*;

/// Helper to create a pipeline for memory benchmarks
async fn create_memory_test_pipeline() -> Result<Arc<VoirsPipeline>> {
    let pipeline = VoirsPipelineBuilder::new()
        .with_quality(QualityLevel::Medium)
        .with_test_mode(true)
        .build()
        .await?;
    Ok(Arc::new(pipeline))
}

/// Benchmark cache hit vs miss performance
fn bench_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_performance");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_memory_test_pipeline()).unwrap();

    let test_text = "This text will be cached for performance testing.";

    group.bench_function("cache_miss", |b| {
        b.to_async(&runtime).iter(|| async {
            // Clear cache before each run to ensure miss
            let unique_text = format!("{} {}", test_text, fastrand::u64(..));
            let audio = pipeline.synthesize(black_box(&unique_text)).await.unwrap();
            black_box(audio);
        });
    });

    // Populate cache
    runtime.block_on(pipeline.synthesize(test_text)).unwrap();

    group.bench_function("cache_hit", |b| {
        b.to_async(&runtime).iter(|| async {
            let audio = pipeline.synthesize(black_box(test_text)).await.unwrap();
            black_box(audio);
        });
    });

    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_memory_test_pipeline()).unwrap();

    // Test different sizes for allocation patterns
    let text_sizes = vec![
        ("small", "Hello"),
        ("medium", "The quick brown fox jumps over the lazy dog."),
        ("large", "In the beginning God created the heavens and the earth. Now the earth was formless and empty, darkness was over the surface of the deep, and the Spirit of God was hovering over the waters."),
    ];

    for (name, text) in text_sizes {
        group.bench_with_input(BenchmarkId::new("text_size", name), &text, |b, &text| {
            b.to_async(&runtime).iter(|| async {
                let audio = pipeline.synthesize(black_box(text)).await.unwrap();
                black_box(audio);
            });
        });
    }

    group.finish();
}

/// Benchmark audio buffer cloning and resampling
fn bench_buffer_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_operations");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_memory_test_pipeline()).unwrap();

    // Generate test audio
    let audio = runtime
        .block_on(pipeline.synthesize("Test audio for buffer operations benchmark"))
        .unwrap();

    group.bench_function("clone", |b| {
        b.iter(|| {
            let cloned = black_box(audio.clone());
            black_box(cloned);
        });
    });

    group.bench_function("to_vec", |b| {
        b.iter(|| {
            let samples = black_box(audio.samples().to_vec());
            black_box(samples);
        });
    });

    group.bench_function("resample_down", |b| {
        b.iter(|| {
            let resampled = black_box(audio.resample(16000).unwrap());
            black_box(resampled);
        });
    });

    group.bench_function("resample_up", |b| {
        b.iter(|| {
            let resampled = black_box(audio.resample(48000).unwrap());
            black_box(resampled);
        });
    });

    group.finish();
}

/// Benchmark memory pool efficiency
fn bench_memory_pools(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pools");
    group.measurement_time(Duration::from_secs(10));

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_memory_test_pipeline()).unwrap();

    group.bench_function("sequential_allocations", |b| {
        b.to_async(&runtime).iter(|| async {
            for i in 0..10 {
                let text = format!("Sequential allocation test {}", i);
                let audio = pipeline.synthesize(&text).await.unwrap();
                black_box(audio);
            }
        });
    });

    group.bench_function("concurrent_allocations", |b| {
        b.to_async(&runtime).iter(|| async {
            let mut handles = vec![];

            for i in 0..10 {
                let pipeline = Arc::clone(&pipeline);
                let text = format!("Concurrent allocation test {}", i);

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
    });

    group.finish();
}

/// Benchmark cache eviction performance
fn bench_cache_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_eviction");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_memory_test_pipeline()).unwrap();

    group.bench_function("cache_overflow", |b| {
        b.to_async(&runtime).iter(|| async {
            // Generate many unique texts to trigger eviction
            for i in 0..100 {
                let text = format!("Cache eviction test text number {}", i);
                let audio = pipeline.synthesize(&text).await.unwrap();
                black_box(audio);
            }
        });
    });

    group.finish();
}

/// Benchmark memory pressure scenarios
fn bench_memory_pressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pressure");
    group.measurement_time(Duration::from_secs(15));

    let runtime = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("large_batch", |b| {
        b.to_async(&runtime).iter(|| async {
            let pipeline = VoirsPipelineBuilder::new()
                .with_test_mode(true)
                .build()
                .await
                .unwrap();

            let mut results = vec![];

            // Generate many audio buffers
            for i in 0..50 {
                let text = format!("Memory pressure test {}", i);
                let audio = pipeline.synthesize(&text).await.unwrap();
                results.push(audio);
            }

            black_box(results);
        });
    });

    group.finish();
}

/// Benchmark model loading and caching
fn bench_model_caching(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_caching");
    group.measurement_time(Duration::from_secs(15));

    let runtime = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("first_load", |b| {
        b.to_async(&runtime).iter(|| async {
            let pipeline = VoirsPipelineBuilder::new()
                .with_voice("test-voice-unique")
                .with_test_mode(true)
                .build()
                .await
                .unwrap();

            let audio = pipeline.synthesize("Model loading test").await.unwrap();
            black_box(audio);
        });
    });

    // Create cached pipeline
    let cached_pipeline = runtime.block_on(async {
        VoirsPipelineBuilder::new()
            .with_voice("test-voice-cached")
            .with_test_mode(true)
            .build()
            .await
            .unwrap()
    });

    // Warmup
    runtime
        .block_on(cached_pipeline.synthesize("Warmup"))
        .unwrap();

    group.bench_function("cached_load", |b| {
        b.to_async(&runtime).iter(|| async {
            let audio = cached_pipeline
                .synthesize("Cached model test")
                .await
                .unwrap();
            black_box(audio);
        });
    });

    group.finish();
}

/// Benchmark resource cleanup
fn bench_resource_cleanup(c: &mut Criterion) {
    let mut group = c.benchmark_group("resource_cleanup");

    let runtime = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("pipeline_drop", |b| {
        b.to_async(&runtime).iter(|| async {
            let pipeline = VoirsPipelineBuilder::new()
                .with_test_mode(true)
                .build()
                .await
                .unwrap();

            let _ = pipeline.synthesize("Cleanup test").await.unwrap();

            // Pipeline dropped at end of scope
            drop(pipeline);
        });
    });

    group.finish();
}

criterion_group!(
    memory_benches,
    bench_cache_performance,
    bench_memory_allocation,
    bench_buffer_operations,
    bench_memory_pools,
    bench_cache_eviction,
    bench_memory_pressure,
    bench_model_caching,
    bench_resource_cleanup
);

criterion_main!(memory_benches);
