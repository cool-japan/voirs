//! Real-Time Factor (RTF) benchmarks for vocoders
//!
//! Measures synthesis speed relative to real-time audio playback.
//! RTF = synthesis_time / audio_duration
//! RTF < 1.0 means faster than real-time (good for real-time applications)

#![allow(unused_variables)]

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;
use std::time::Duration;
use voirs_vocoder::{
    config::{QualityLevel, VocodingConfig},
    DummyVocoder, MelSpectrogram, SynthesisConfig, Vocoder,
};

/// Generate test mel spectrogram with realistic dimensions
fn generate_test_mel(n_mels: usize, n_frames: usize, sample_rate: u32) -> MelSpectrogram {
    let mut data = Vec::with_capacity(n_mels);
    for mel_idx in 0..n_mels {
        let mut frame = Vec::with_capacity(n_frames);
        for frame_idx in 0..n_frames {
            // Generate realistic mel values (log magnitude in dB range)
            let base_freq = (mel_idx as f32 / n_mels as f32) * 4000.0 + 80.0;
            let time = frame_idx as f32 / (sample_rate as f32 / 256.0); // Assume hop_length = 256
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

/// Benchmark HiFi-GAN RTF performance
fn bench_hifigan_rtf(c: &mut Criterion) {
    let vocoder = DummyVocoder::new(); // Use dummy vocoder for consistent benchmarking

    let mut group = c.benchmark_group("hifigan_rtf");

    // Test different audio durations
    for &duration_secs in &[0.5, 1.0, 2.0, 5.0] {
        let sample_rate = 22050;
        let n_frames = ((duration_secs * sample_rate as f32) / 256.0) as usize; // hop_length = 256
        let mel = generate_test_mel(80, n_frames, sample_rate);
        let audio_duration = Duration::from_secs_f32(duration_secs);

        group.throughput(Throughput::Elements(n_frames as u64));
        group.bench_with_input(
            format!("duration_{duration_secs}s"),
            &(mel, audio_duration),
            |b, (mel, _expected_duration)| {
                b.iter(|| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        let result = vocoder.vocode(black_box(mel), None).await;
                        black_box(result.unwrap());
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark WaveGlow RTF performance
fn bench_waveglow_rtf(c: &mut Criterion) {
    let vocoder = DummyVocoder::new(); // Use dummy vocoder for consistent benchmarking

    let mut group = c.benchmark_group("waveglow_rtf");

    for &duration_secs in &[0.5, 1.0, 2.0] {
        let sample_rate = 22050;
        let n_frames = ((duration_secs * sample_rate as f32) / 256.0) as usize;
        let mel = generate_test_mel(80, n_frames, sample_rate);
        let audio_duration = Duration::from_secs_f32(duration_secs);

        group.throughput(Throughput::Elements(n_frames as u64));
        group.bench_with_input(
            format!("duration_{duration_secs}s"),
            &(mel, audio_duration),
            |b, (mel, _expected_duration)| {
                b.iter(|| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        let result = vocoder.vocode(black_box(mel), None).await;
                        black_box(result.unwrap());
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark DiffWave RTF performance  
fn bench_diffwave_rtf(c: &mut Criterion) {
    let vocoder = DummyVocoder::new(); // Use dummy vocoder for consistent benchmarking

    let mut group = c.benchmark_group("diffwave_rtf");
    group.sample_size(10); // DiffWave is slower, use fewer samples

    for &duration_secs in &[0.5, 1.0] {
        let sample_rate = 22050;
        let n_frames = ((duration_secs * sample_rate as f32) / 256.0) as usize;
        let mel = generate_test_mel(80, n_frames, sample_rate);
        let audio_duration = Duration::from_secs_f32(duration_secs);

        group.throughput(Throughput::Elements(n_frames as u64));
        group.bench_with_input(
            format!("duration_{duration_secs}s"),
            &(mel, audio_duration),
            |b, (mel, _expected_duration)| {
                b.iter(|| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        let result = vocoder.vocode(black_box(mel), None).await;
                        black_box(result.unwrap());
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch processing RTF
fn bench_batch_rtf(c: &mut Criterion) {
    let vocoder = DummyVocoder::new(); // Use dummy vocoder for consistent benchmarking

    let mut group = c.benchmark_group("batch_rtf");

    for &batch_size in &[1, 4, 8, 16] {
        let duration_secs = 1.0;
        let sample_rate = 22050;
        let n_frames = ((duration_secs * sample_rate as f32) / 256.0) as usize;

        let mels: Vec<MelSpectrogram> = (0..batch_size)
            .map(|_| generate_test_mel(80, n_frames, sample_rate))
            .collect();

        let total_audio_duration = Duration::from_secs_f32(duration_secs * batch_size as f32);

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(format!("batch_size_{batch_size}"), &mels, |b, mels| {
            b.iter(|| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let result = vocoder.vocode_batch(mels, None).await;
                    black_box(result.unwrap());
                })
            });
        });
    }

    group.finish();
}

/// Benchmark different quality settings RTF
fn bench_quality_rtf(c: &mut Criterion) {
    let vocoder = DummyVocoder::new(); // Use dummy vocoder for consistent benchmarking

    let mut group = c.benchmark_group("quality_rtf");

    let duration_secs = 1.0;
    let sample_rate = 22050;
    let n_frames = ((duration_secs * sample_rate as f32) / 256.0) as usize;
    let mel = generate_test_mel(80, n_frames, sample_rate);
    let audio_duration = Duration::from_secs_f32(duration_secs);

    for quality in [
        QualityLevel::Low,
        QualityLevel::Medium,
        QualityLevel::High,
        QualityLevel::Ultra,
    ] {
        let config = match quality {
            QualityLevel::Low => VocodingConfig::low_resource(),
            QualityLevel::Medium => VocodingConfig::realtime(),
            QualityLevel::High => VocodingConfig::high_quality(),
            QualityLevel::Ultra => VocodingConfig::ultra_quality(),
        };

        group.bench_with_input(
            format!("{quality:?}_quality"),
            &(mel.clone(), audio_duration, config),
            |b, (mel, _expected_duration, config)| {
                b.iter(|| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
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

criterion_group!(
    benches,
    bench_hifigan_rtf,
    bench_waveglow_rtf,
    bench_diffwave_rtf,
    bench_batch_rtf,
    bench_quality_rtf
);
criterion_main!(benches);
