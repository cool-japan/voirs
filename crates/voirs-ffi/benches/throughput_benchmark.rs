//! Simplified throughput benchmarks for VoiRS FFI
//!
//! These benchmarks measure basic pipeline operations
//! to establish performance baselines.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use tokio::runtime::Runtime;

use voirs_sdk::VoirsPipeline;

fn setup_runtime() -> Runtime {
    Runtime::new().unwrap()
}

fn benchmark_pipeline_creation(c: &mut Criterion) {
    let rt = setup_runtime();

    c.bench_function("pipeline_creation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _pipeline = VoirsPipeline::builder()
                    .with_test_mode(true)
                    .build()
                    .await
                    .expect("Failed to create pipeline");
                black_box(_pipeline);
            })
        });
    });
}

fn benchmark_single_synthesis(c: &mut Criterion) {
    let rt = setup_runtime();
    let pipeline = rt.block_on(async {
        VoirsPipeline::builder()
            .with_test_mode(true)
            .build()
            .await
            .expect("Failed to create pipeline")
    });

    c.bench_function("single_synthesis", |b| {
        b.iter(|| {
            rt.block_on(async {
                let text = black_box("This is a benchmark synthesis test.");
                let _ = pipeline.synthesize(text).await;
            })
        });
    });
}

fn benchmark_synthesis_by_text_length(c: &mut Criterion) {
    let rt = setup_runtime();
    let pipeline = rt.block_on(async {
        VoirsPipeline::builder()
            .with_test_mode(true)
            .build()
            .await
            .expect("Failed to create pipeline")
    });

    let mut group = c.benchmark_group("synthesis_by_length");
    group.measurement_time(Duration::from_secs(10));

    let test_texts = vec![
        ("short", "Hello world."),
        ("medium", "This is a medium length sentence for synthesis benchmarking purposes."),
        ("long", "This is a much longer text sample that contains multiple sentences and should take more time to synthesize. It includes various words and punctuation marks to simulate realistic text that might be encountered in production environments."),
    ];

    for (name, text) in test_texts {
        group.bench_function(name, |b| {
            b.iter(|| {
                rt.block_on(async {
                    let text = black_box(text);
                    let _ = pipeline.synthesize(text).await;
                })
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_pipeline_creation,
    benchmark_single_synthesis,
    benchmark_synthesis_by_text_length
);
criterion_main!(benches);
