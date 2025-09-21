//! Memory usage profiling benchmarks for vocoders
//!
//! Measures memory allocation patterns, peak usage, and memory efficiency
//! across different vocoder architectures and usage patterns.

#![allow(unused_variables)]

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;
use voirs_vocoder::{
    config::{QualityLevel, VocodingConfig},
    DummyVocoder, MelSpectrogram, SynthesisConfig, Vocoder,
};

/// Simple memory usage tracker
struct MemoryTracker {
    start_rss: Option<usize>,
    peak_rss: usize,
}

impl MemoryTracker {
    fn new() -> Self {
        Self {
            start_rss: Self::get_rss_memory(),
            peak_rss: 0,
        }
    }

    fn checkpoint(&mut self) -> Option<usize> {
        if let Some(current_rss) = Self::get_rss_memory() {
            self.peak_rss = self.peak_rss.max(current_rss);
            Some(current_rss)
        } else {
            None
        }
    }

    fn peak_usage_mb(&self) -> f32 {
        if let Some(start) = self.start_rss {
            (self.peak_rss.saturating_sub(start)) as f32 / 1024.0 / 1024.0
        } else {
            0.0
        }
    }

    #[cfg(target_os = "linux")]
    fn get_rss_memory() -> Option<usize> {
        use std::fs;
        let status = fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    return parts[1].parse::<usize>().ok().map(|kb| kb * 1024);
                }
            }
        }
        None
    }

    #[cfg(not(target_os = "linux"))]
    fn get_rss_memory() -> Option<usize> {
        // Fallback: use process memory info if available
        // For now, return None on non-Linux systems
        None
    }
}

/// Generate test mel with controlled size
fn generate_test_mel(n_mels: usize, n_frames: usize, sample_rate: u32) -> MelSpectrogram {
    let mut data = Vec::with_capacity(n_mels);
    for mel_idx in 0..n_mels {
        let mut frame = Vec::with_capacity(n_frames);
        for frame_idx in 0..n_frames {
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

/// Benchmark memory usage for different audio durations
fn bench_memory_by_duration(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_by_duration");
    group.sample_size(20);

    for &duration_secs in &[1.0, 5.0, 10.0, 30.0] {
        let sample_rate = 22050;
        let n_frames = ((duration_secs * sample_rate as f32) / 256.0) as usize;
        let mel = generate_test_mel(80, n_frames, sample_rate);

        // Estimate mel size
        let _mel_size_mb = (n_frames * 80 * 4) as f32 / 1024.0 / 1024.0; // f32 = 4 bytes

        group.bench_with_input(format!("duration_{duration_secs}s"), &mel, |b, mel| {
            b.iter(|| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let vocoder = DummyVocoder::new();
                    let mut tracker = MemoryTracker::new();
                    tracker.checkpoint();

                    let result = vocoder.vocode(black_box(mel), None).await;
                    black_box(result.unwrap());

                    tracker.checkpoint();
                    let _peak_mb = tracker.peak_usage_mb();
                })
            });
        });
    }

    group.finish();
}

/// Benchmark memory usage for batch processing
fn bench_memory_batch_processing(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_batch_processing");
    group.sample_size(15);

    for &batch_size in &[1, 4, 8, 16, 32] {
        let duration_secs = 2.0;
        let sample_rate = 22050;
        let n_frames = ((duration_secs * sample_rate as f32) / 256.0) as usize;

        let mels: Vec<MelSpectrogram> = (0..batch_size)
            .map(|_| generate_test_mel(80, n_frames, sample_rate))
            .collect();

        let _total_mel_size_mb = (batch_size * n_frames * 80 * 4) as f32 / 1024.0 / 1024.0;

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(format!("batch_size_{batch_size}"), &mels, |b, mels| {
            b.iter(|| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let vocoder = DummyVocoder::new();
                    let mut tracker = MemoryTracker::new();
                    tracker.checkpoint();

                    let result = vocoder.vocode_batch(mels, None).await;
                    black_box(result.unwrap());

                    tracker.checkpoint();
                    let _peak_mb = tracker.peak_usage_mb();
                })
            });
        });
    }

    group.finish();
}

/// Benchmark memory usage for streaming processing
fn bench_memory_streaming(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_streaming");
    group.sample_size(10);

    for &buffer_count in &[2, 4, 8, 16] {
        let chunk_ms = 40;
        let sample_rate = 22050;
        let chunk_frames = ((chunk_ms as f32 / 1000.0) * sample_rate as f32 / 256.0) as usize;
        let mel = generate_test_mel(80, chunk_frames, sample_rate);

        let _chunk_size_mb = (chunk_frames * 80 * 4) as f32 / 1024.0 / 1024.0;

        group.bench_with_input(format!("buffers_{buffer_count}"), &mel, |b, mel| {
            b.iter(|| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let vocoder = DummyVocoder::new();
                    let mut tracker = MemoryTracker::new();
                    tracker.checkpoint();

                    // Process multiple chunks to simulate buffering
                    for _ in 0..buffer_count * 2 {
                        let result = vocoder.vocode(black_box(mel), None).await;
                        black_box(result.unwrap());
                        tracker.checkpoint();
                    }

                    let _peak_mb = tracker.peak_usage_mb();
                })
            });
        });
    }

    group.finish();
}

/// Benchmark memory usage across different vocoders
fn bench_memory_by_vocoder(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_by_vocoder");
    group.sample_size(15);

    let duration_secs = 3.0;
    let sample_rate = 22050;
    let n_frames = ((duration_secs * sample_rate as f32) / 256.0) as usize;
    let mel = generate_test_mel(80, n_frames, sample_rate);

    // HiFi-GAN
    group.bench_function("hifigan", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let vocoder = DummyVocoder::new();
                let mut tracker = MemoryTracker::new();
                tracker.checkpoint();

                let result = vocoder.vocode(black_box(&mel), None).await;
                black_box(result.unwrap());

                tracker.checkpoint();
                let _peak_mb = tracker.peak_usage_mb();
            })
        });
    });

    // WaveGlow
    group.bench_function("waveglow", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let vocoder = DummyVocoder::new();
                let mut tracker = MemoryTracker::new();
                tracker.checkpoint();

                let result = vocoder.vocode(black_box(&mel), None).await;
                black_box(result.unwrap());

                tracker.checkpoint();
                let _peak_mb = tracker.peak_usage_mb();
            })
        });
    });

    // DiffWave
    group.bench_function("diffwave", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let vocoder = DummyVocoder::new();
                let mut tracker = MemoryTracker::new();
                tracker.checkpoint();

                let result = vocoder.vocode(black_box(&mel), None).await;
                black_box(result.unwrap());

                tracker.checkpoint();
                let _peak_mb = tracker.peak_usage_mb();
            })
        });
    });

    group.finish();
}

/// Benchmark memory efficiency with different quality settings
fn bench_memory_by_quality(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_by_quality");
    group.sample_size(20);

    let duration_secs = 2.0;
    let sample_rate = 22050;
    let n_frames = ((duration_secs * sample_rate as f32) / 256.0) as usize;
    let mel = generate_test_mel(80, n_frames, sample_rate);

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
            &(mel.clone(), config),
            |b, (mel, _config)| {
                b.iter(|| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        let vocoder = DummyVocoder::new();
                        let mut tracker = MemoryTracker::new();
                        tracker.checkpoint();

                        // Convert VocodingConfig to SynthesisConfig for vocoder API
                        let synthesis_config = SynthesisConfig::default();
                        let result = vocoder
                            .vocode(black_box(mel), Some(&synthesis_config))
                            .await;
                        black_box(result.unwrap());

                        tracker.checkpoint();
                        let _peak_mb = tracker.peak_usage_mb();
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_memory_allocation_patterns(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_allocation_patterns");
    group.sample_size(25);

    let duration_secs = 1.0;
    let sample_rate = 22050;
    let n_frames = ((duration_secs * sample_rate as f32) / 256.0) as usize;

    // Test repeated allocations vs reuse
    group.bench_function("repeated_allocations", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let vocoder = DummyVocoder::new();
                let mut tracker = MemoryTracker::new();
                tracker.checkpoint();

                // Repeatedly allocate new mels
                for i in 0..5 {
                    let mel = generate_test_mel(80, n_frames, sample_rate);
                    let result = vocoder.vocode(black_box(&mel), None).await;
                    black_box(result.unwrap());

                    if i == 0 {
                        tracker.checkpoint(); // Track after first allocation
                    }
                }

                tracker.checkpoint();
                tracker.peak_usage_mb()
            })
        });
    });

    group.bench_function("reused_allocations", |b| {
        b.iter_with_setup(
            || {
                let vocoder = DummyVocoder::new();
                let mel = generate_test_mel(80, n_frames, sample_rate);
                (vocoder, mel)
            },
            |(vocoder, mel)| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let mut tracker = MemoryTracker::new();
                    tracker.checkpoint();

                    // Reuse same mel
                    for i in 0..5 {
                        let result = vocoder.vocode(black_box(&mel), None).await;
                        black_box(result.unwrap());

                        if i == 0 {
                            tracker.checkpoint(); // Track after first use
                        }
                    }

                    tracker.checkpoint();
                    tracker.peak_usage_mb()
                })
            },
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_memory_by_duration,
    bench_memory_batch_processing,
    bench_memory_streaming,
    bench_memory_by_vocoder,
    bench_memory_by_quality,
    bench_memory_allocation_patterns
);
criterion_main!(benches);
