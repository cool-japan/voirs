//! Benchmarks for state-of-the-art research models

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use voirs_singing::prelude::*;
use voirs_singing::{
    ConsistencyModelConfig, DiffusionTransformerConfig, FlowMatchingConfig, IntegrationMethod,
    NeuralCodecConfig, ScoreBasedConfig,
};

fn benchmark_diffusion_transformer(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("diffusion_transformer");
    group.sample_size(10);

    let conditioning = vec![0.5; 1000];

    for num_steps in [10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_steps),
            num_steps,
            |b, &num_steps| {
                b.to_async(&rt).iter(|| {
                    let conditioning = conditioning.clone();
                    async move {
                        let config = DiffusionTransformerConfig {
                            timesteps: 100,
                            ..Default::default()
                        };
                        let model = DiffusionTransformer::new(config);
                        black_box(model.generate(&conditioning, num_steps).await.unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_neural_codec(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let audio = vec![0.5; 16000];

    c.bench_function("neural_codec_encode", |b| {
        b.to_async(&rt).iter(|| {
            let audio = audio.clone();
            async move {
                let config = NeuralCodecConfig::default();
                let model = NeuralCodecLanguageModel::new(config);
                black_box(model.encode(&audio).await.unwrap());
            }
        });
    });

    c.bench_function("neural_codec_decode", |b| {
        b.to_async(&rt).iter(|| async {
            let config = NeuralCodecConfig::default();
            let model = NeuralCodecLanguageModel::new(config);
            let tokens = model.encode(&vec![0.5; 16000]).await.unwrap();
            black_box(model.decode(&tokens).await.unwrap());
        });
    });
}

fn benchmark_flow_matching(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("flow_matching");
    group.sample_size(10);

    let conditioning = vec![0.5; 1000];
    let methods = [
        ("euler", IntegrationMethod::Euler),
        ("heun", IntegrationMethod::Heun),
        ("rk4", IntegrationMethod::RungeKutta4),
    ];

    for (name, method) in &methods {
        group.bench_with_input(BenchmarkId::from_parameter(name), method, |b, &method| {
            b.to_async(&rt).iter(|| {
                let conditioning = conditioning.clone();
                async move {
                    let config = FlowMatchingConfig {
                        flow_steps: 50,
                        integration_method: method,
                        ..Default::default()
                    };
                    let synthesizer = FlowMatchingSynthesizer::new(config);
                    black_box(synthesizer.generate(&conditioning, 2000).await.unwrap());
                }
            });
        });
    }

    group.finish();
}

fn benchmark_score_based_model(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("score_based_model");
    group.sample_size(10);

    let conditioning = vec![0.5; 100];

    for num_scales in [10, 25, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_scales),
            num_scales,
            |b, &num_scales| {
                b.to_async(&rt).iter(|| {
                    let conditioning = conditioning.clone();
                    async move {
                        let config = ScoreBasedConfig {
                            num_scales,
                            ..Default::default()
                        };
                        let model = ScoreBasedModel::new(config);
                        black_box(model.generate(&conditioning, 1000).await.unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_consistency_model(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let conditioning = vec![0.5; 100];

    c.bench_function("consistency_single_step", |b| {
        b.to_async(&rt).iter(|| {
            let conditioning = conditioning.clone();
            async move {
                let config = ConsistencyModelConfig::default();
                let model = ConsistencyModel::new(config);
                black_box(model.generate(&conditioning, 1000).await.unwrap());
            }
        });
    });

    c.bench_function("consistency_multi_step", |b| {
        b.to_async(&rt).iter(|| {
            let conditioning = conditioning.clone();
            async move {
                let config = ConsistencyModelConfig::default();
                let model = ConsistencyModel::new(config);
                black_box(
                    model
                        .generate_multistep(&conditioning, 1000, 5)
                        .await
                        .unwrap(),
                );
            }
        });
    });
}

criterion_group!(
    benches,
    benchmark_diffusion_transformer,
    benchmark_neural_codec,
    benchmark_flow_matching,
    benchmark_score_based_model,
    benchmark_consistency_model
);
criterion_main!(benches);
