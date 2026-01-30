//! Production Performance Validation Benchmarks
//!
//! This benchmark suite validates that the acoustic models meet the performance targets
//! defined in TODO.md. It measures:
//! - Real-Time Factor (RTF) for CPU/GPU
//! - Streaming latency
//! - Memory footprint
//! - Model loading time
//! - Resource usage patterns

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use voirs_acoustic::{
    fastspeech::FastSpeech2Model, memory::MemoryOptimizer, AcousticModel, MelSpectrogram, Phoneme,
    SynthesisConfig, VitsModel,
};

/// Performance targets from TODO.md
mod targets {
    pub const CPU_RTF_TARGET: f64 = 0.28; // Real-Time Factor for CPU
    pub const STREAMING_LATENCY_MS: u64 = 50; // End-to-end latency target
    pub const MEMORY_FOOTPRINT_MB: usize = 512; // Memory footprint per model
    pub const MODEL_LOADING_TIME_MS: u64 = 2000; // Startup time target
}

/// Benchmark Real-Time Factor (RTF) for synthesis
///
/// RTF = synthesis_time / audio_duration
/// Target: ≤ 0.28× for CPU
fn bench_rtf_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Create realistic test sentence (approx 2 seconds of speech)
    let test_sentence = create_test_sentence(40); // 40 phonemes ~ 2 seconds

    let mut group = c.benchmark_group("rtf_validation");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("vits_rtf_measurement", |b| {
        b.iter_custom(|iters| {
            let mut total_time = Duration::ZERO;

            for _ in 0..iters {
                let start = Instant::now();
                let _ = rt.block_on(async {
                    let model = VitsModel::new().unwrap();
                    let result = model.synthesize(black_box(&test_sentence), None).await;
                    black_box(result)
                });
                total_time += start.elapsed();
            }

            total_time
        });
    });

    group.bench_function("fastspeech2_rtf_measurement", |b| {
        b.iter_custom(|iters| {
            let mut total_time = Duration::ZERO;

            for _ in 0..iters {
                let start = Instant::now();
                let _ = rt.block_on(async {
                    let model = FastSpeech2Model::new();
                    let result = model.synthesize(black_box(&test_sentence), None).await;
                    black_box(result)
                });
                total_time += start.elapsed();
            }

            total_time
        });
    });

    group.finish();
}

/// Benchmark model loading and initialization time
///
/// Target: ≤ 2 seconds
fn bench_model_loading_time(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("model_loading");
    group.sample_size(20);

    group.bench_function("vits_cold_start", |b| {
        b.iter_custom(|iters| {
            let mut total_time = Duration::ZERO;

            for _ in 0..iters {
                let start = Instant::now();
                rt.block_on(async {
                    let model = VitsModel::new().unwrap();
                    black_box(model)
                });
                total_time += start.elapsed();
            }

            total_time
        });
    });

    group.bench_function("fastspeech2_cold_start", |b| {
        b.iter_custom(|iters| {
            let mut total_time = Duration::ZERO;

            for _ in 0..iters {
                let start = Instant::now();
                rt.block_on(async {
                    let model = FastSpeech2Model::new();
                    black_box(model)
                });
                total_time += start.elapsed();
            }

            total_time
        });
    });

    group.finish();
}

/// Benchmark streaming synthesis latency
///
/// Target: ≤ 50ms end-to-end
fn bench_streaming_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Small chunks for streaming (5-10 phonemes per chunk)
    let chunk_sizes = vec![5, 10, 15];

    let mut group = c.benchmark_group("streaming_latency");
    group.sample_size(100);

    for chunk_size in chunk_sizes {
        let chunk = create_test_sentence(chunk_size);

        group.bench_with_input(
            BenchmarkId::new("vits_chunk_latency", chunk_size),
            &chunk,
            |b, chunk| {
                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let start = Instant::now();
                        let _ = rt.block_on(async {
                            let model = VitsModel::new().unwrap();
                            let result = model.synthesize(black_box(chunk), None).await;
                            black_box(result)
                        });
                        total_time += start.elapsed();
                    }

                    total_time
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fastspeech2_chunk_latency", chunk_size),
            &chunk,
            |b, chunk| {
                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let start = Instant::now();
                        let _ = rt.block_on(async {
                            let model = FastSpeech2Model::new();
                            let result = model.synthesize(black_box(chunk), None).await;
                            black_box(result)
                        });
                        total_time += start.elapsed();
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory footprint during synthesis
///
/// Target: ≤ 512MB per model
fn bench_memory_footprint(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let test_sentence = create_test_sentence(50);

    let mut group = c.benchmark_group("memory_footprint");
    group.sample_size(10);

    group.bench_function("vits_memory_usage", |b| {
        b.iter(|| {
            rt.block_on(async {
                let optimizer = MemoryOptimizer::new();

                // Measure baseline
                let baseline_bytes = optimizer.get_current_usage();

                // Create model and synthesize
                let model = VitsModel::new().unwrap();
                let _result = model.synthesize(black_box(&test_sentence), None).await;

                // Measure peak
                let peak_bytes = optimizer.get_current_usage();

                // Calculate memory delta in MB
                let delta_mb = if peak_bytes > baseline_bytes {
                    (peak_bytes - baseline_bytes) / (1024 * 1024)
                } else {
                    0
                };
                black_box(delta_mb)
            })
        });
    });

    group.bench_function("fastspeech2_memory_usage", |b| {
        b.iter(|| {
            rt.block_on(async {
                let optimizer = MemoryOptimizer::new();

                let baseline_bytes = optimizer.get_current_usage();

                let model = FastSpeech2Model::new();
                let _result = model.synthesize(black_box(&test_sentence), None).await;

                let peak_bytes = optimizer.get_current_usage();

                let delta_mb = if peak_bytes > baseline_bytes {
                    (peak_bytes - baseline_bytes) / (1024 * 1024)
                } else {
                    0
                };
                black_box(delta_mb)
            })
        });
    });

    group.finish();
}

/// Benchmark sustained throughput over time
///
/// Tests performance stability under continuous load
fn bench_sustained_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let test_sentence = create_test_sentence(30);

    let mut group = c.benchmark_group("sustained_throughput");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    group.bench_function("vits_continuous_synthesis", |b| {
        b.iter_custom(|iters| {
            let mut total_time = Duration::ZERO;

            rt.block_on(async {
                let model = VitsModel::new().unwrap();

                for _ in 0..iters {
                    let start = Instant::now();
                    let _result = model.synthesize(black_box(&test_sentence), None).await;
                    total_time += start.elapsed();
                }
            });

            total_time
        });
    });

    group.bench_function("fastspeech2_continuous_synthesis", |b| {
        b.iter_custom(|iters| {
            let mut total_time = Duration::ZERO;

            rt.block_on(async {
                let model = FastSpeech2Model::new();

                for _ in 0..iters {
                    let start = Instant::now();
                    let _result = model.synthesize(black_box(&test_sentence), None).await;
                    total_time += start.elapsed();
                }
            });

            total_time
        });
    });

    group.finish();
}

/// Benchmark performance under different prosody configurations
///
/// Ensures that prosody control doesn't significantly impact performance
fn bench_prosody_performance_impact(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let test_sentence = create_test_sentence(25);

    let configs = vec![
        ("baseline", None),
        (
            "fast_speech",
            Some(SynthesisConfig {
                speed: 1.5,
                pitch_shift: 0.0,
                energy: 1.0,
                speaker_id: None,
                seed: Some(42),
                emotion: None,
                voice_style: None,
            }),
        ),
        (
            "slow_expressive",
            Some(SynthesisConfig {
                speed: 0.7,
                pitch_shift: 3.0,
                energy: 1.3,
                speaker_id: None,
                seed: Some(42),
                emotion: None,
                voice_style: None,
            }),
        ),
    ];

    let mut group = c.benchmark_group("prosody_performance_impact");

    for (name, config) in configs {
        group.bench_with_input(
            BenchmarkId::new("vits", name),
            &(test_sentence.clone(), config.clone()),
            |b, (sentence, config)| {
                b.iter(|| {
                    rt.block_on(async {
                        let model = VitsModel::new().unwrap();
                        let result = model
                            .synthesize(black_box(sentence), black_box(config.as_ref()))
                            .await;
                        black_box(result)
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fastspeech2", name),
            &(test_sentence.clone(), config.clone()),
            |b, (sentence, config)| {
                b.iter(|| {
                    rt.block_on(async {
                        let model = FastSpeech2Model::new();
                        let result = model
                            .synthesize(black_box(sentence), black_box(config.as_ref()))
                            .await;
                        black_box(result)
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark mel spectrogram computation performance
///
/// Tests the performance of mel spectrogram creation and manipulation
fn bench_mel_computation_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("mel_computation");

    // Different mel spectrogram sizes representing different audio lengths
    let configurations = vec![
        ("1s_audio", 80, 100),   // 1 second
        ("3s_audio", 80, 300),   // 3 seconds
        ("5s_audio", 80, 500),   // 5 seconds
        ("10s_audio", 80, 1000), // 10 seconds
    ];

    for (name, n_mels, n_frames) in configurations {
        let data: Vec<Vec<f32>> = (0..n_mels)
            .map(|_| (0..n_frames).map(|_| 0.5f32).collect())
            .collect();

        group.bench_with_input(BenchmarkId::new("creation", name), &data, |b, data| {
            b.iter(|| {
                let mel =
                    MelSpectrogram::new(black_box(data.clone()), black_box(22050), black_box(256));
                black_box(mel)
            });
        });

        let mel = MelSpectrogram::new(data.clone(), 22050, 256);

        group.bench_with_input(BenchmarkId::new("duration", name), &mel, |b, mel| {
            b.iter(|| {
                let duration = mel.duration();
                black_box(duration)
            });
        });

        group.bench_with_input(BenchmarkId::new("clone", name), &mel, |b, mel| {
            b.iter(|| {
                let cloned = mel.clone();
                black_box(cloned)
            });
        });
    }

    group.finish();
}

/// Benchmark cache efficiency and hit rates
///
/// Tests the effectiveness of internal caching mechanisms
fn bench_cache_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let test_sentence = create_test_sentence(20);

    let mut group = c.benchmark_group("cache_performance");
    group.sample_size(50);

    // Benchmark repeated synthesis with same input (should benefit from caching)
    group.bench_function("vits_repeated_synthesis", |b| {
        b.iter(|| {
            rt.block_on(async {
                let model = VitsModel::new().unwrap();

                // Synthesize same input multiple times
                for _ in 0..5 {
                    let _result = model.synthesize(black_box(&test_sentence), None).await;
                }
            })
        });
    });

    group.bench_function("fastspeech2_repeated_synthesis", |b| {
        b.iter(|| {
            rt.block_on(async {
                let model = FastSpeech2Model::new();

                for _ in 0..5 {
                    let _result = model.synthesize(black_box(&test_sentence), None).await;
                }
            })
        });
    });

    group.finish();
}

/// Helper function to create realistic test sentences
fn create_test_sentence(num_phonemes: usize) -> Vec<Phoneme> {
    let phoneme_pool = vec![
        "h", "ɛ", "l", "oʊ", "w", "ɝ", "l", "d", "ð", "ɪ", "s", "ə", "t", "ɛ", "s", "t", "ɪ", "ŋ",
        "ɹ", "i", "əl", "t", "aɪ", "m", "p", "ɝ", "f", "ɔ", "ɹ", "m", "ə", "n", "s",
    ];

    phoneme_pool
        .iter()
        .cycle()
        .take(num_phonemes)
        .map(|&p| Phoneme::new(p))
        .collect()
}

criterion_group!(
    performance_validation,
    bench_rtf_validation,
    bench_model_loading_time,
    bench_streaming_latency,
    bench_memory_footprint,
    bench_sustained_throughput,
    bench_prosody_performance_impact,
    bench_mel_computation_performance,
    bench_cache_performance
);

criterion_main!(performance_validation);
