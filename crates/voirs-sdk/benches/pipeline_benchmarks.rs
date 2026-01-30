//! Pipeline Initialization and Management Benchmarks
//!
//! This benchmark suite measures pipeline creation, initialization,
//! and management operations including voice switching and configuration updates.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use voirs_sdk::prelude::*;

/// Benchmark pipeline builder construction
fn bench_pipeline_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_creation");

    let runtime = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("default_config", |b| {
        b.to_async(&runtime).iter(|| async {
            let pipeline = VoirsPipelineBuilder::new()
                .with_test_mode(true)
                .build()
                .await
                .unwrap();
            black_box(pipeline);
        });
    });

    group.bench_function("with_quality", |b| {
        b.to_async(&runtime).iter(|| async {
            let pipeline = VoirsPipelineBuilder::new()
                .with_quality(QualityLevel::High)
                .with_test_mode(true)
                .build()
                .await
                .unwrap();
            black_box(pipeline);
        });
    });

    group.bench_function("full_config", |b| {
        b.to_async(&runtime).iter(|| async {
            let pipeline = VoirsPipelineBuilder::new()
                .with_quality(QualityLevel::High)
                .with_voice("test-voice")
                .with_test_mode(true)
                .build()
                .await
                .unwrap();
            black_box(pipeline);
        });
    });

    group.finish();
}

/// Benchmark voice switching performance
fn bench_voice_switching(c: &mut Criterion) {
    let mut group = c.benchmark_group("voice_switching");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(async {
        VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap()
    });

    let voices = vec!["voice-1", "voice-2", "voice-3", "voice-4"];

    for voice in &voices {
        group.bench_with_input(BenchmarkId::new("switch_to", voice), voice, |b, &voice| {
            b.to_async(&runtime).iter(|| async {
                pipeline.set_voice(black_box(voice)).await.unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark voice listing and discovery
fn bench_voice_discovery(c: &mut Criterion) {
    let mut group = c.benchmark_group("voice_discovery");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(async {
        VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap()
    });

    group.bench_function("list_all_voices", |b| {
        b.to_async(&runtime).iter(|| async {
            let voices = pipeline.list_voices().await.unwrap();
            black_box(voices);
        });
    });

    group.bench_function("get_current_voice", |b| {
        b.to_async(&runtime).iter(|| async {
            let voice = pipeline.current_voice().await;
            black_box(voice);
        });
    });

    group.finish();
}

/// Benchmark configuration updates
fn bench_config_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_updates");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(async {
        VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap()
    });

    // Test voice switching as configuration update
    group.bench_function("update_voice", |b| {
        b.to_async(&runtime).iter(|| async {
            pipeline.set_voice(black_box("voice-1")).await.unwrap();
            pipeline.set_voice(black_box("voice-2")).await.unwrap();
        });
    });

    group.finish();
}

/// Benchmark pipeline state management
fn bench_state_management(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_management");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(async {
        VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap()
    });

    group.bench_function("get_state", |b| {
        b.to_async(&runtime).iter(|| async {
            let state = pipeline.get_state().await;
            black_box(state);
        });
    });

    group.bench_function("list_voices", |b| {
        b.to_async(&runtime).iter(|| async {
            let voices = pipeline.list_voices().await.unwrap();
            black_box(voices);
        });
    });

    group.finish();
}

/// Benchmark multiple pipeline instances
fn bench_multiple_pipelines(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiple_pipelines");
    group.measurement_time(Duration::from_secs(15));

    let runtime = tokio::runtime::Runtime::new().unwrap();

    let instance_counts = vec![1, 2, 4, 8];

    for count in instance_counts {
        group.bench_with_input(BenchmarkId::new("instances", count), &count, |b, &count| {
            b.to_async(&runtime).iter(|| async {
                let mut pipelines = vec![];

                for _ in 0..count {
                    let pipeline = VoirsPipelineBuilder::new()
                        .with_test_mode(true)
                        .build()
                        .await
                        .unwrap();
                    pipelines.push(pipeline);
                }

                black_box(pipelines);
            });
        });
    }

    group.finish();
}

/// Benchmark pipeline warmup
fn bench_pipeline_warmup(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_warmup");

    let runtime = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("cold_start", |b| {
        b.to_async(&runtime).iter(|| async {
            let pipeline = VoirsPipelineBuilder::new()
                .with_test_mode(true)
                .build()
                .await
                .unwrap();

            // First synthesis (cold)
            let audio = pipeline.synthesize("Warmup test").await.unwrap();
            black_box(audio);
        });
    });

    group.bench_function("warm_synthesis", |b| {
        let pipeline = runtime.block_on(async {
            let p = VoirsPipelineBuilder::new()
                .with_test_mode(true)
                .build()
                .await
                .unwrap();

            // Warmup
            let _ = p.synthesize("Warmup").await;
            p
        });

        b.to_async(&runtime).iter(|| async {
            let audio = pipeline.synthesize("Warmup test").await.unwrap();
            black_box(audio);
        });
    });

    group.finish();
}

criterion_group!(
    pipeline_benches,
    bench_pipeline_creation,
    bench_voice_switching,
    bench_voice_discovery,
    bench_config_updates,
    bench_state_management,
    bench_multiple_pipelines,
    bench_pipeline_warmup
);

criterion_main!(pipeline_benches);
