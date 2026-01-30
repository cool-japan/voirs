//! Benchmarks for Advanced Features
//!
//! Performance benchmarks for neural codec, latency optimizer, and VAD

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use voirs_acoustic::{NeuralCodecConfig, VadConfig, VoiceActivityDetector};

#[cfg(feature = "candle")]
use candle_core::Device;

#[cfg(feature = "candle")]
use voirs_acoustic::NeuralCodec;

#[cfg(feature = "candle")]
use voirs_acoustic::neural_codec::ResidualVectorQuantizer;

/// Benchmark Neural Codec encoding
#[cfg(feature = "candle")]
fn bench_neural_codec_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_codec_encode");

    let device = Device::Cpu;
    let configs = vec![
        ("low_latency", NeuralCodecConfig::low_latency()),
        ("default", NeuralCodecConfig::default()),
        ("high_quality", NeuralCodecConfig::high_quality()),
    ];

    for (name, config) in configs {
        let codec = NeuralCodec::new(config, &device).unwrap();

        // Create sample audio (1 second at 16kHz)
        let waveform = candle_core::Tensor::randn(0.0, 1.0, (1, 16000), &device).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(name), &codec, |b, codec| {
            b.iter(|| black_box(codec.encode(&waveform).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark Neural Codec decoding
#[cfg(feature = "candle")]
fn bench_neural_codec_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_codec_decode");

    let device = Device::Cpu;
    let config = NeuralCodecConfig::default();
    let codec = NeuralCodec::new(config, &device).unwrap();

    // Create sample codes
    let num_frames = 50;
    let codes: Vec<Vec<usize>> = (0..8)
        .map(|_| (0..num_frames).map(|_| fastrand::usize(0..1024)).collect())
        .collect();

    group.bench_function("decode_50_frames", |b| {
        b.iter(|| black_box(codec.decode(&codes).unwrap()));
    });

    group.finish();
}

/// Benchmark RVQ encoding
#[cfg(feature = "candle")]
fn bench_rvq_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("rvq_encode");

    let device = Device::Cpu;

    for num_levels in [2, 4, 8, 16] {
        let rvq = ResidualVectorQuantizer::new(num_levels, 1024, 256, &device).unwrap();

        let input = candle_core::Tensor::randn(0.0, 1.0, (1, 50, 256), &device).unwrap();

        let bench_name = format!("{}_levels", num_levels);
        group.bench_function(&bench_name, |b| {
            b.iter(|| black_box(rvq.encode(&input).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark VAD frame processing
fn bench_vad_frame_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("vad_frame_processing");

    let configs = vec![
        ("conversational", VadConfig::conversational()),
        ("studio", VadConfig::studio()),
        ("noisy", VadConfig::noisy()),
    ];

    for (name, config) in configs {
        let frame_size = config.frame_size;
        let mut vad = VoiceActivityDetector::new(config).unwrap();

        // Create sample frame
        let frame: Vec<f32> = (0..frame_size)
            .map(|i| (i as f32 * 0.01).sin() * 0.3)
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(name), &frame, |b, frame| {
            b.iter(|| black_box(vad.process_frame(frame)));
        });
    }

    group.finish();
}

/// Benchmark VAD buffer processing
fn bench_vad_buffer_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("vad_buffer_processing");

    for duration_sec in [1, 3, 5] {
        let config = VadConfig::default();
        let mut vad = VoiceActivityDetector::new(config.clone()).unwrap();

        // Create audio buffer
        let samples = duration_sec * config.sample_rate;
        let audio: Vec<f32> = (0..samples)
            .map(|i| (i as f32 * 0.01).sin() * 0.3)
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}s", duration_sec)),
            &audio,
            |b, audio| {
                b.iter(|| {
                    vad.reset();
                    black_box(vad.process_buffer(audio))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark VAD energy calculation
fn bench_vad_energy_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("vad_energy_calculation");

    for frame_size in [256, 512, 1024, 2048] {
        let frame: Vec<f32> = (0..frame_size)
            .map(|i| (i as f32 * 0.01).sin() * 0.3)
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_samples", frame_size)),
            &frame,
            |b, frame| {
                b.iter(|| black_box(VoiceActivityDetector::calculate_energy_db(frame)));
            },
        );
    }

    group.finish();
}

/// Benchmark VAD zero-crossing rate calculation
fn bench_vad_zcr_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("vad_zcr_calculation");

    for frame_size in [256, 512, 1024, 2048] {
        let frame: Vec<f32> = (0..frame_size)
            .map(|i| if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_samples", frame_size)),
            &frame,
            |b, frame| {
                b.iter(|| black_box(VoiceActivityDetector::calculate_zcr(frame)));
            },
        );
    }

    group.finish();
}

#[cfg(feature = "candle")]
criterion_group!(
    neural_codec_benches,
    bench_neural_codec_encode,
    bench_neural_codec_decode,
    bench_rvq_encode
);

criterion_group!(
    vad_benches,
    bench_vad_frame_processing,
    bench_vad_buffer_processing,
    bench_vad_energy_calculation,
    bench_vad_zcr_calculation
);

#[cfg(feature = "candle")]
criterion_main!(neural_codec_benches, vad_benches);

#[cfg(not(feature = "candle"))]
criterion_main!(vad_benches);
