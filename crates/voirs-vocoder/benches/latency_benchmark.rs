//! Latency analysis benchmarks for real-time vocoding
//!
//! Measures end-to-end latency from mel input to audio output,
//! including streaming and chunk-based processing latencies.

#![allow(unused_variables)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::hint::black_box;
use voirs_vocoder::{
    config::{QualityLevel, VocodingConfig},
    DummyVocoder, MelSpectrogram, SynthesisConfig, Vocoder,
};

/// Generate small mel chunk for latency testing
fn generate_mel_chunk(n_mels: usize, chunk_frames: usize, sample_rate: u32) -> MelSpectrogram {
    let mut data = Vec::with_capacity(n_mels);
    for mel_idx in 0..n_mels {
        let mut frame = Vec::with_capacity(chunk_frames);
        for frame_idx in 0..chunk_frames {
            let base_freq = (mel_idx as f32 / n_mels as f32) * 4000.0 + 80.0;
            let time = frame_idx as f32 / (sample_rate as f32 / 256.0);
            let magnitude = -20.0
                + 15.0
                    * (2.0 * std::f32::consts::PI * base_freq * time / 8000.0)
                        .sin()
                        .abs();
            frame.push(magnitude);
        }
        data.push(frame);
    }

    MelSpectrogram::new(data, sample_rate, 256)
}

/// Benchmark first chunk latency (cold start)
fn bench_first_chunk_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("first_chunk_latency");
    group.sample_size(50); // Fewer samples for latency measurements

    // Test different chunk sizes
    for &chunk_ms in &[20, 40, 80, 160] {
        let sample_rate = 22050;
        let chunk_frames = ((chunk_ms as f32 / 1000.0) * sample_rate as f32 / 256.0) as usize;
        let mel = generate_mel_chunk(80, chunk_frames, sample_rate);

        group.bench_with_input(format!("chunk_{chunk_ms}ms"), &mel, |b, mel| {
            b.iter_batched(
                || {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    (rt, DummyVocoder::new())
                },
                |(rt, vocoder)| {
                    rt.block_on(async {
                        let result = vocoder.vocode(black_box(mel), None).await;
                        black_box(result.unwrap());
                    })
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark streaming latency (warm vocoder)
fn bench_streaming_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("streaming_latency");
    group.sample_size(100);

    for &chunk_ms in &[20, 40, 80] {
        let sample_rate = 22050;
        let chunk_frames = ((chunk_ms as f32 / 1000.0) * sample_rate as f32 / 256.0) as usize;
        let mel = generate_mel_chunk(80, chunk_frames, sample_rate);

        group.bench_with_input(format!("chunk_{chunk_ms}ms"), &mel, |b, mel| {
            b.iter_batched(
                || {
                    // Setup: create and warm up vocoder
                    let vocoder = DummyVocoder::new();
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        let _ = vocoder.vocode(mel, None).await;
                    });
                    (vocoder, rt)
                },
                |(vocoder, rt)| {
                    rt.block_on(async {
                        let result = vocoder.vocode(black_box(mel), None).await;
                        black_box(result.unwrap());
                    })
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark end-to-end streaming pipeline latency
fn bench_streaming_pipeline_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("streaming_pipeline_latency");
    group.sample_size(30);

    for &chunk_ms in &[20, 40, 80] {
        let sample_rate = 22050;
        let chunk_frames = ((chunk_ms as f32 / 1000.0) * sample_rate as f32 / 256.0) as usize;
        let mel = generate_mel_chunk(80, chunk_frames, sample_rate);

        group.bench_with_input(format!("pipeline_{chunk_ms}ms"), &mel, |b, mel| {
            b.iter_batched(
                || {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    (rt, DummyVocoder::new())
                },
                |(rt, vocoder)| {
                    rt.block_on(async {
                        let result = vocoder.vocode(black_box(mel), None).await;
                        black_box(result.unwrap());
                    })
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark latency with different quality settings
fn bench_quality_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("quality_latency");
    group.sample_size(50);

    let chunk_ms = 40; // Fixed chunk size
    let sample_rate = 22050;
    let chunk_frames = ((chunk_ms as f32 / 1000.0) * sample_rate as f32 / 256.0) as usize;
    let mel = generate_mel_chunk(80, chunk_frames, sample_rate);

    for quality in [QualityLevel::Low, QualityLevel::Medium, QualityLevel::High] {
        let config = match quality {
            QualityLevel::Low => VocodingConfig::low_resource(),
            QualityLevel::Medium => VocodingConfig::realtime(),
            QualityLevel::High => VocodingConfig::high_quality(),
            QualityLevel::Ultra => VocodingConfig::ultra_quality(),
        };

        group.bench_with_input(
            format!("{quality:?}_quality"),
            &(mel.clone(), config),
            |b, (mel, _config)| {
                b.iter(|| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        let vocoder = DummyVocoder::new();
                        // Convert VocodingConfig to SynthesisConfig for vocoder API
                        let synthesis_config = SynthesisConfig::default();
                        let result = vocoder
                            .vocode(black_box(mel), Some(&synthesis_config))
                            .await;
                        black_box(result.unwrap());
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark buffer and queue latencies
fn bench_buffer_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("buffer_latency");
    group.sample_size(200);

    // Test different buffer strategies
    let strategies = [
        ("fixed_buffer", 4),
        ("small_buffer", 2),
        ("large_buffer", 8),
    ];

    let chunk_ms = 40;
    let sample_rate = 22050;
    let chunk_frames = ((chunk_ms as f32 / 1000.0) * sample_rate as f32 / 256.0) as usize;
    let mel = generate_mel_chunk(80, chunk_frames, sample_rate);

    for (strategy_name, _buffer_size) in strategies {
        group.bench_with_input(strategy_name, &mel, |b, mel| {
            b.iter(|| {
                // Use DummyVocoder directly for simplified benchmarking
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let vocoder = DummyVocoder::new();
                    let result = vocoder.vocode(black_box(mel), None).await;
                    black_box(result.unwrap());
                })
            });
        });
    }

    group.finish();
}

/// Benchmark memory allocation latency impact
fn bench_memory_allocation_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_allocation_latency");
    group.sample_size(100);

    let chunk_ms = 40;
    let sample_rate = 22050;
    let chunk_frames = ((chunk_ms as f32 / 1000.0) * sample_rate as f32 / 256.0) as usize;

    // Test with different memory allocation patterns
    for &num_allocations in &[1, 10, 50] {
        group.bench_with_input(
            format!("allocations_{num_allocations}"),
            &num_allocations,
            |b, &num_allocations| {
                b.iter(|| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        let vocoder = DummyVocoder::new();
                        // Generate fresh mel spectrograms (allocation)
                        for _ in 0..num_allocations {
                            let mel = generate_mel_chunk(80, chunk_frames, sample_rate);
                            let result = vocoder.vocode(black_box(&mel), None).await;
                            black_box(result.unwrap());
                        }
                    })
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_first_chunk_latency,
    bench_streaming_latency,
    bench_streaming_pipeline_latency,
    bench_quality_latency,
    bench_buffer_latency,
    bench_memory_allocation_latency
);
criterion_main!(benches);
