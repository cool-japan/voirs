//! # Phase 4 Research Models Benchmarks
//!
//! Comprehensive benchmarks for Phase 4 research integration features:
//! - Autoregressive Synthesis (MusicGen-style)
//! - Stable Audio Latent Diffusion
//! - Optimal Transport Flow Matching
//! - Advanced Neural Codec
//! - Real-time Inference Engine
//!
//! Run with: `cargo bench --bench phase4_research_benchmarks`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use voirs_singing::research_integration::*;

// ==================== Autoregressive Synthesis Benchmarks ====================

fn bench_autoregressive_semantic_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("autoregressive_semantic");

    for num_tokens in [10, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*num_tokens as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_tokens),
            num_tokens,
            |b, &num_tokens| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let config = AutoregressiveConfig::fast();
                let synthesizer = AutoregressiveSynthesizer::new(config);
                let conditioning = SingingConditioning {
                    text: "Hello world".to_string(),
                    pitch_contour: vec![440.0; 10],
                    duration: 1.0,
                    style_embedding: None,
                    speaker_embedding: None,
                };

                b.to_async(&rt).iter(|| async {
                    let target_samples = num_tokens * 480; // 50 Hz frame rate at 24kHz
                    black_box(synthesizer.generate(&conditioning, target_samples).await)
                });
            },
        );
    }
    group.finish();
}

fn bench_autoregressive_cfg_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("autoregressive_cfg");

    for guidance_scale in [1.0, 3.0, 7.0].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(guidance_scale),
            guidance_scale,
            |b, &scale| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let mut config = AutoregressiveConfig::fast();
                config.cfg_scale = scale;
                let synthesizer = AutoregressiveSynthesizer::new(config);
                let conditioning = SingingConditioning::default();

                b.to_async(&rt).iter(|| async {
                    black_box(synthesizer.generate_with_cfg(&conditioning, 24000).await)
                });
            },
        );
    }
    group.finish();
}

fn bench_autoregressive_delay_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("autoregressive_patterns");

    for pattern in [
        DelayPattern::Sequential,
        DelayPattern::Parallel,
        DelayPattern::Custom,
    ]
    .iter()
    {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", pattern)),
            pattern,
            |b, &pattern| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let mut config = AutoregressiveConfig::fast();
                config.delay_pattern = pattern;
                let synthesizer = AutoregressiveSynthesizer::new(config);
                let conditioning = SingingConditioning::default();

                b.to_async(&rt)
                    .iter(|| async { black_box(synthesizer.generate(&conditioning, 12000).await) });
            },
        );
    }
    group.finish();
}

// ==================== Stable Audio Benchmarks ====================

fn bench_stable_audio_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("stable_audio_generation");

    for duration in [1.0, 2.0, 5.0].iter() {
        group.throughput(Throughput::Elements(*duration as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(duration),
            duration,
            |b, &duration| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let config = StableAudioConfig::fast();
                let model = StableAudioModel::new(config);
                let prompt = StableAudioPrompt {
                    text: "Singing voice with vibrato".to_string(),
                    duration_seconds: duration,
                    guidance_scale: 5.0,
                    seed: Some(42),
                    negative_prompt: None,
                };

                b.to_async(&rt)
                    .iter(|| async { black_box(model.generate(&prompt).await) });
            },
        );
    }
    group.finish();
}

fn bench_stable_audio_quality_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("stable_audio_quality");

    let configs = vec![
        ("fast", StableAudioConfig::fast()),
        ("high_quality", StableAudioConfig::high_quality()),
        ("ultra_hq", StableAudioConfig::ultra_high_quality()),
    ];

    for (name, config) in configs.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(name), name, |b, _| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let model = StableAudioModel::new(config.clone());
            let prompt = StableAudioPrompt {
                text: "Clear singing".to_string(),
                duration_seconds: 1.0,
                guidance_scale: 7.0,
                seed: Some(42),
                negative_prompt: None,
            };

            b.to_async(&rt)
                .iter(|| async { black_box(model.generate(&prompt).await) });
        });
    }
    group.finish();
}

fn bench_stable_audio_cfg_scales(c: &mut Criterion) {
    let mut group = c.benchmark_group("stable_audio_cfg");

    for guidance_scale in [1.0, 3.0, 5.0, 7.0, 10.0].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(guidance_scale),
            guidance_scale,
            |b, &scale| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let config = StableAudioConfig::fast();
                let model = StableAudioModel::new(config);
                let prompt = StableAudioPrompt {
                    text: "Voice".to_string(),
                    duration_seconds: 0.5,
                    guidance_scale: scale,
                    seed: Some(42),
                    negative_prompt: None,
                };

                b.to_async(&rt)
                    .iter(|| async { black_box(model.generate(&prompt).await) });
            },
        );
    }
    group.finish();
}

fn bench_stable_audio_noise_schedules(c: &mut Criterion) {
    let mut group = c.benchmark_group("stable_audio_schedules");

    for schedule in [
        NoiseScheduleType::Linear,
        NoiseScheduleType::Cosine,
        NoiseScheduleType::Sigmoid,
        NoiseScheduleType::Custom,
    ]
    .iter()
    {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", schedule)),
            schedule,
            |b, &schedule| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let mut config = StableAudioConfig::fast();
                config.noise_schedule = schedule;
                let model = StableAudioModel::new(config);
                let prompt = StableAudioPrompt::default();

                b.to_async(&rt)
                    .iter(|| async { black_box(model.generate(&prompt).await) });
            },
        );
    }
    group.finish();
}

// ==================== Optimal Transport Flow Matching Benchmarks ====================

fn bench_optimal_transport_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimal_transport");

    for num_samples in [12000, 24000, 48000].iter() {
        group.throughput(Throughput::Elements(*num_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_samples),
            num_samples,
            |b, &num_samples| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let conditioning = vec![0.1; 512];

                b.to_async(&rt).iter(|| {
                    let config = OptimalTransportConfig::default();
                    let mut flow = OptimalTransportFlow::new(config);
                    let cond_clone = conditioning.clone();
                    async move { black_box(flow.generate(&cond_clone, num_samples).await) }
                });
            },
        );
    }
    group.finish();
}

fn bench_transport_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("transport_methods");

    for method in [
        TransportMethod::Sinkhorn,
        TransportMethod::ExactLP,
        TransportMethod::SlicedWasserstein,
    ]
    .iter()
    {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", method)),
            method,
            |b, &method| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let conditioning = vec![0.1; 256];

                b.to_async(&rt).iter(|| {
                    let mut config = OptimalTransportConfig::default();
                    config.transport_method = method;
                    let mut flow = OptimalTransportFlow::new(config);
                    let cond_clone = conditioning.clone();
                    async move { black_box(flow.generate(&cond_clone, 12000).await) }
                });
            },
        );
    }
    group.finish();
}

// ==================== Real-time Inference Benchmarks ====================

fn bench_realtime_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("realtime_inference");

    for batch_size in [1, 4, 8, 16].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let mut config = InferenceConfig::low_latency();
                config.max_batch_size = batch_size;
                let engine = RealtimeInferenceEngine::new(config);
                let input = vec![0.1; 1024];

                b.to_async(&rt)
                    .iter(|| async { black_box(engine.infer(&input).await) });
            },
        );
    }
    group.finish();
}

fn bench_caching_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_strategies");

    for strategy in [
        CacheStrategy::LRU,
        CacheStrategy::LFU,
        CacheStrategy::NoCache,
    ]
    .iter()
    {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", strategy)),
            strategy,
            |b, &strategy| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let mut config = InferenceConfig::default();
                config.cache_strategy = strategy;
                let engine = RealtimeInferenceEngine::new(config);
                let input = vec![0.1; 512];

                b.to_async(&rt)
                    .iter(|| async { black_box(engine.infer(&input).await) });
            },
        );
    }
    group.finish();
}

// ==================== Advanced Codec Benchmarks ====================

fn bench_advanced_codec_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_encode");

    for num_samples in [24000, 48000, 96000].iter() {
        group.throughput(Throughput::Elements(*num_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_samples),
            num_samples,
            |b, &num_samples| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let audio = vec![0.1; num_samples];

                b.to_async(&rt).iter(|| {
                    let config = AdvancedCodecConfig::default();
                    let mut codec = AdvancedNeuralCodec::new(config);
                    let audio_clone = audio.clone();
                    async move { black_box(codec.encode(&audio_clone).await) }
                });
            },
        );
    }
    group.finish();
}

fn bench_advanced_codec_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_decode");

    for num_codes in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*num_codes as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_codes),
            num_codes,
            |b, &num_codes| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let codes = vec![vec![0; 8]; num_codes];

                b.to_async(&rt).iter(|| {
                    let config = AdvancedCodecConfig::default();
                    let mut codec = AdvancedNeuralCodec::new(config);
                    let codes_clone = codes.clone();
                    async move { black_box(codec.decode(&codes_clone).await) }
                });
            },
        );
    }
    group.finish();
}

// ==================== Velocity Field Benchmarks ====================

fn bench_velocity_field_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("velocity_field");

    for num_steps in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*num_steps as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_steps),
            num_steps,
            |b, &num_steps| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let conditioning = vec![0.1; 256];

                b.to_async(&rt).iter(|| {
                    let config = VelocityConfig::default();
                    let mut predictor = VelocityFieldPredictor::new(config);
                    let cond_clone = conditioning.clone();
                    let x = vec![0.0; 64]; // Position vector
                    async move { black_box(predictor.predict(&x, 0.5, &cond_clone).await) }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    autoregressive_benches,
    bench_autoregressive_semantic_generation,
    bench_autoregressive_cfg_generation,
    bench_autoregressive_delay_patterns,
);

criterion_group!(
    stable_audio_benches,
    bench_stable_audio_generation,
    bench_stable_audio_quality_levels,
    bench_stable_audio_cfg_scales,
    bench_stable_audio_noise_schedules,
);

criterion_group!(
    optimal_transport_benches,
    bench_optimal_transport_generation,
    bench_transport_methods,
);

criterion_group!(
    realtime_benches,
    bench_realtime_inference,
    bench_caching_strategies,
);

criterion_group!(
    codec_benches,
    bench_advanced_codec_encode,
    bench_advanced_codec_decode,
);

criterion_group!(velocity_benches, bench_velocity_field_prediction,);

criterion_main!(
    autoregressive_benches,
    stable_audio_benches,
    optimal_transport_benches,
    realtime_benches,
    codec_benches,
    velocity_benches,
);
