//! Synthesis Performance Benchmarks
//!
//! This benchmark suite measures the performance of core synthesis operations
//! in the VoiRS SDK, including basic synthesis, SSML processing, and quality levels.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use voirs_sdk::prelude::*;

/// Helper to create a pipeline for benchmarking
async fn create_test_pipeline() -> Result<VoirsPipeline> {
    VoirsPipelineBuilder::new()
        .with_quality(QualityLevel::Medium)
        .with_test_mode(true) // Use dummy implementations for consistent benchmarking
        .build()
        .await
}

/// Benchmark basic text synthesis with varying text lengths
fn bench_text_synthesis(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_synthesis");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_test_pipeline()).unwrap();

    // Test different text lengths
    let test_cases = vec![
        ("short", "Hello world"),
        ("medium", "The quick brown fox jumps over the lazy dog and continues running through the forest."),
        ("long", "In the heart of the ancient forest, where sunlight barely penetrates the thick canopy of leaves, a small stream meanders through moss-covered rocks. The sound of water trickling over stones creates a peaceful melody that has echoed through these woods for centuries, unchanged by the passage of time."),
    ];

    for (name, text) in test_cases {
        group.bench_with_input(BenchmarkId::from_parameter(name), &text, |b, &text| {
            b.to_async(&runtime).iter(|| async {
                let audio = pipeline.synthesize(black_box(text)).await.unwrap();
                black_box(audio);
            });
        });
    }

    group.finish();
}

/// Benchmark synthesis with different quality levels
fn bench_quality_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_levels");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let test_text = "The quality of synthesis affects both audio fidelity and processing time.";

    let quality_levels = vec![
        QualityLevel::Low,
        QualityLevel::Medium,
        QualityLevel::High,
        QualityLevel::Ultra,
    ];

    for quality in quality_levels {
        group.bench_with_input(
            BenchmarkId::new("quality", format!("{:?}", quality)),
            &quality,
            |b, &quality| {
                b.to_async(&runtime).iter(|| async {
                    let pipeline = VoirsPipelineBuilder::new()
                        .with_quality(quality)
                        .with_test_mode(true)
                        .build()
                        .await
                        .unwrap();

                    let audio = pipeline.synthesize(black_box(test_text)).await.unwrap();
                    black_box(audio);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SSML synthesis
fn bench_ssml_synthesis(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssml_synthesis");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_test_pipeline()).unwrap();

    let ssml_examples = vec![
        ("basic", r#"<speak>Hello world</speak>"#),
        (
            "with_prosody",
            r#"<speak><prosody rate="slow" pitch="high">Hello world</prosody></speak>"#,
        ),
        (
            "with_breaks",
            r#"<speak>Hello<break time="500ms"/>world<break time="1s"/>how are you?</speak>"#,
        ),
        (
            "complex",
            r#"<speak>
                <prosody rate="medium" pitch="medium">
                    Welcome to <emphasis level="strong">VoiRS</emphasis>.
                    <break time="300ms"/>
                    This is a <prosody rate="slow">slower section</prosody> of speech.
                </prosody>
            </speak>"#,
        ),
    ];

    for (name, ssml) in ssml_examples {
        group.bench_with_input(BenchmarkId::from_parameter(name), &ssml, |b, &ssml| {
            b.to_async(&runtime).iter(|| async {
                let audio = pipeline.synthesize_ssml(black_box(ssml)).await.unwrap();
                black_box(audio);
            });
        });
    }

    group.finish();
}

/// Benchmark synthesis with configuration
fn bench_synthesis_with_config(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthesis_with_config");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_test_pipeline()).unwrap();
    let test_text = "Benchmarking synthesis with custom configuration.";

    // Different speed configurations
    let speeds = vec![0.5, 0.75, 1.0, 1.25, 1.5, 2.0];

    for speed in speeds {
        let mut config = SynthesisConfig::default();
        config.speaking_rate = speed;

        group.bench_with_input(
            BenchmarkId::new("speed", format!("{:.2}x", speed)),
            &config,
            |b, config| {
                b.to_async(&runtime).iter(|| async {
                    let audio = pipeline
                        .synthesize_with_config(black_box(test_text), black_box(config))
                        .await
                        .unwrap();
                    black_box(audio);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark repeated synthesis (cache performance)
fn bench_repeated_synthesis(c: &mut Criterion) {
    let mut group = c.benchmark_group("repeated_synthesis");
    group.measurement_time(Duration::from_secs(10));

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_test_pipeline()).unwrap();
    let test_text = "This text will be synthesized multiple times to test caching.";

    group.bench_function("first_run", |b| {
        b.to_async(&runtime).iter(|| async {
            let audio = pipeline.synthesize(black_box(test_text)).await.unwrap();
            black_box(audio);
        });
    });

    // Synthesize once to populate cache
    runtime.block_on(pipeline.synthesize(test_text)).unwrap();

    group.bench_function("cached_run", |b| {
        b.to_async(&runtime).iter(|| async {
            let audio = pipeline.synthesize(black_box(test_text)).await.unwrap();
            black_box(audio);
        });
    });

    group.finish();
}

/// Benchmark audio buffer operations
fn bench_audio_buffer_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_buffer_ops");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pipeline = runtime.block_on(create_test_pipeline()).unwrap();
    let audio = runtime
        .block_on(pipeline.synthesize("Test audio for buffer operations"))
        .unwrap();

    group.bench_function("clone", |b| {
        b.iter(|| {
            let cloned = black_box(audio.clone());
            black_box(cloned);
        });
    });

    group.bench_function("resample_22050", |b| {
        b.iter(|| {
            let resampled = black_box(audio.resample(22050).unwrap());
            black_box(resampled);
        });
    });

    group.bench_function("resample_48000", |b| {
        b.iter(|| {
            let resampled = black_box(audio.resample(48000).unwrap());
            black_box(resampled);
        });
    });

    group.bench_function("normalize", |b| {
        b.iter(|| {
            let mut normalized = audio.clone();
            let _ = black_box(normalized.normalize(1.0));
        });
    });

    group.finish();
}

criterion_group!(
    synthesis_benches,
    bench_text_synthesis,
    bench_quality_levels,
    bench_ssml_synthesis,
    bench_synthesis_with_config,
    bench_repeated_synthesis,
    bench_audio_buffer_ops
);

criterion_main!(synthesis_benches);
