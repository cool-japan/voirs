//! Performance benchmarks for voirs-acoustic
//!
//! This module provides comprehensive benchmarks for acoustic model operations,
//! including synthesis performance, memory usage, and system stress tests.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tokio::runtime::Runtime;
use voirs_acoustic::{
    fastspeech::FastSpeech2Model, AcousticModel, MelSpectrogram, Phoneme, SynthesisConfig,
    VitsModel,
};

/// Benchmark VITS model synthesis performance
fn bench_vits_synthesis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Setup test data
    let phonemes = vec![
        Phoneme::new("h"),
        Phoneme::new("ɛ"),
        Phoneme::new("l"),
        Phoneme::new("oʊ"),
        Phoneme::new("w"),
        Phoneme::new("ɝ"),
        Phoneme::new("l"),
        Phoneme::new("d"),
    ];

    let mut group = c.benchmark_group("vits_synthesis");

    // Benchmark different sequence lengths
    for length in [8, 16, 32].iter() {
        let test_phonemes: Vec<Phoneme> = phonemes.iter().cycle().take(*length).cloned().collect();

        group.bench_with_input(
            BenchmarkId::new("single_sequence", length),
            &test_phonemes,
            |b, phonemes| {
                b.iter(|| {
                    rt.block_on(async {
                        let model = VitsModel::new().unwrap();
                        let result = model.synthesize(black_box(phonemes), None).await;
                        black_box(result)
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark FastSpeech2 model synthesis performance  
fn bench_fastspeech2_synthesis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let phonemes = vec![
        Phoneme::new("h"),
        Phoneme::new("ɛ"),
        Phoneme::new("l"),
        Phoneme::new("oʊ"),
        Phoneme::new("w"),
        Phoneme::new("ɝ"),
        Phoneme::new("l"),
        Phoneme::new("d"),
    ];

    let mut group = c.benchmark_group("fastspeech2_synthesis");

    for length in [8, 16, 32].iter() {
        let test_phonemes: Vec<Phoneme> = phonemes.iter().cycle().take(*length).cloned().collect();

        group.bench_with_input(
            BenchmarkId::new("single_sequence", length),
            &test_phonemes,
            |b, phonemes| {
                b.iter(|| {
                    rt.block_on(async {
                        let model = FastSpeech2Model::new();
                        let result = model.synthesize(black_box(phonemes), None).await;
                        black_box(result)
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark mel spectrogram operations
fn bench_mel_operations(c: &mut Criterion) {
    use voirs_acoustic::MelSpectrogram;

    let mut group = c.benchmark_group("mel_operations");

    // Create test mel spectrogram data
    let test_data: Vec<Vec<f32>> = (0..80)
        .map(|_| (0..100).map(|_| 0.5f32).collect())
        .collect();

    group.bench_function("mel_creation", |b| {
        let data = test_data.clone();
        b.iter(|| {
            let mel =
                MelSpectrogram::new(black_box(data.clone()), black_box(22050), black_box(256));
            black_box(mel)
        });
    });

    group.bench_function("mel_operations", |b| {
        let mel = MelSpectrogram::new(test_data.clone(), 22050, 256);
        b.iter(|| {
            let mel_copy = black_box(mel.clone());
            black_box(mel_copy)
        });
    });

    group.finish();
}

/// Benchmark batch synthesis performance
fn bench_batch_synthesis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let phonemes = vec![
        Phoneme::new("h"),
        Phoneme::new("ɛ"),
        Phoneme::new("l"),
        Phoneme::new("oʊ"),
        Phoneme::new("w"),
        Phoneme::new("ɝ"),
        Phoneme::new("l"),
        Phoneme::new("d"),
    ];

    let mut group = c.benchmark_group("batch_synthesis");

    // Test different batch sizes
    for batch_size in [1, 4, 8, 16].iter() {
        let batch: Vec<Vec<Phoneme>> = (0..*batch_size).map(|_| phonemes.clone()).collect();
        let batch_refs: Vec<&[Phoneme]> = batch.iter().map(|v| v.as_slice()).collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("vits_batch", batch_size),
            &batch_refs,
            |b, batch| {
                b.iter(|| {
                    rt.block_on(async {
                        let model = VitsModel::new().unwrap();
                        let result = model.synthesize_batch(black_box(batch), None).await;
                        black_box(result)
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fastspeech2_batch", batch_size),
            &batch_refs,
            |b, batch| {
                b.iter(|| {
                    rt.block_on(async {
                        let model = FastSpeech2Model::new();
                        let result = model.synthesize_batch(black_box(batch), None).await;
                        black_box(result)
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark synthesis with various configurations
fn bench_synthesis_configurations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let phonemes = vec![
        Phoneme::new("h"),
        Phoneme::new("ɛ"),
        Phoneme::new("l"),
        Phoneme::new("oʊ"),
        Phoneme::new("w"),
        Phoneme::new("ɝ"),
        Phoneme::new("l"),
        Phoneme::new("d"),
    ];

    let configs = vec![
        ("default", SynthesisConfig::default()),
        (
            "fast",
            SynthesisConfig {
                speed: 1.5,
                pitch_shift: 0.0,
                energy: 1.0,
                speaker_id: None,
                seed: Some(42),
                emotion: None,
                voice_style: None,
            },
        ),
        (
            "slow_high_pitch",
            SynthesisConfig {
                speed: 0.7,
                pitch_shift: 5.0,
                energy: 1.2,
                speaker_id: None,
                seed: Some(42),
                emotion: None,
                voice_style: None,
            },
        ),
        (
            "low_energy",
            SynthesisConfig {
                speed: 1.0,
                pitch_shift: -2.0,
                energy: 0.6,
                speaker_id: None,
                seed: Some(42),
                emotion: None,
                voice_style: None,
            },
        ),
    ];

    let mut group = c.benchmark_group("synthesis_configurations");

    for (config_name, config) in configs.iter() {
        group.bench_with_input(
            BenchmarkId::new("vits", config_name),
            &(phonemes.clone(), config),
            |b, (phonemes, config)| {
                b.iter(|| {
                    rt.block_on(async {
                        let model = VitsModel::new().unwrap();
                        let result = model.synthesize(black_box(phonemes), Some(config)).await;
                        black_box(result)
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fastspeech2", config_name),
            &(phonemes.clone(), config),
            |b, (phonemes, config)| {
                b.iter(|| {
                    rt.block_on(async {
                        let model = FastSpeech2Model::new();
                        let result = model.synthesize(black_box(phonemes), Some(config)).await;
                        black_box(result)
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage and allocation patterns
fn bench_memory_performance(c: &mut Criterion) {
    use voirs_acoustic::memory::{MemoryOptimizer, TensorMemoryPool};

    let mut group = c.benchmark_group("memory_performance");

    // Test memory pool performance
    group.bench_function("memory_pool_allocation", |b| {
        let pool = TensorMemoryPool::new();
        b.iter(|| {
            let buffer1 = pool.get_buffer(black_box(1024));
            let buffer2 = pool.get_buffer(black_box(2048));
            let buffer3 = pool.get_buffer(black_box(512));

            pool.return_buffer(buffer1);
            pool.return_buffer(buffer2);
            pool.return_buffer(buffer3);
        });
    });

    // Test direct allocation
    group.bench_function("direct_allocation", |b| {
        b.iter(|| {
            let _buffer1 = vec![0.0f32; black_box(1024)];
            let _buffer2 = vec![0.0f32; black_box(2048)];
            let _buffer3 = vec![0.0f32; black_box(512)];
        });
    });

    // Test memory optimizer
    group.bench_function("memory_optimizer", |b| {
        let optimizer = MemoryOptimizer::new();
        b.iter(|| {
            let usage = optimizer.get_current_usage();
            black_box(usage);
        });
    });

    group.finish();
}

/// Benchmark mel spectrogram operations with different sizes
fn bench_mel_spectrogram_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("mel_spectrogram_sizes");

    let sizes = vec![
        (40, 50),   // Small: 40 mels x 50 frames
        (80, 100),  // Medium: 80 mels x 100 frames
        (128, 200), // Large: 128 mels x 200 frames
        (256, 500), // Very large: 256 mels x 500 frames
    ];

    for (n_mels, n_frames) in sizes {
        let data: Vec<Vec<f32>> = (0..n_mels)
            .map(|_| (0..n_frames).map(|_| 0.5f32).collect())
            .collect();

        group.throughput(Throughput::Elements((n_mels * n_frames) as u64));

        group.bench_with_input(
            BenchmarkId::new("creation", format!("{n_mels}x{n_frames}")),
            &data,
            |b, data| {
                b.iter(|| {
                    let mel = MelSpectrogram::new(
                        black_box(data.clone()),
                        black_box(22050),
                        black_box(256),
                    );
                    black_box(mel)
                });
            },
        );

        let mel = MelSpectrogram::new(data.clone(), 22050, 256);

        group.bench_with_input(
            BenchmarkId::new("clone", format!("{n_mels}x{n_frames}")),
            &mel,
            |b, mel| {
                b.iter(|| {
                    let cloned = black_box(mel.clone());
                    black_box(cloned)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("duration_calc", format!("{n_mels}x{n_frames}")),
            &mel,
            |b, mel| {
                b.iter(|| {
                    let duration = mel.duration();
                    black_box(duration)
                });
            },
        );
    }

    group.finish();
}

/// Stress test with long sequences
fn bench_stress_test(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let base_phonemes = vec![
        Phoneme::new("h"),
        Phoneme::new("ɛ"),
        Phoneme::new("l"),
        Phoneme::new("oʊ"),
        Phoneme::new("w"),
        Phoneme::new("ɝ"),
        Phoneme::new("l"),
        Phoneme::new("d"),
        Phoneme::new("ð"),
        Phoneme::new("ɪ"),
        Phoneme::new("s"),
        Phoneme::new("ɪ"),
        Phoneme::new("z"),
        Phoneme::new("ə"),
        Phoneme::new("t"),
        Phoneme::new("ɛ"),
        Phoneme::new("s"),
        Phoneme::new("t"),
    ];

    let mut group = c.benchmark_group("stress_test");
    group.sample_size(10); // Reduce sample size for stress tests
    group.measurement_time(std::time::Duration::from_secs(60)); // Longer measurement time

    // Test very long sequences
    for length in [100, 200, 500].iter() {
        let long_phonemes: Vec<Phoneme> = base_phonemes
            .iter()
            .cycle()
            .take(*length)
            .cloned()
            .collect();

        group.throughput(Throughput::Elements(*length as u64));

        group.bench_with_input(
            BenchmarkId::new("vits_long_sequence", length),
            &long_phonemes,
            |b, phonemes| {
                b.iter(|| {
                    rt.block_on(async {
                        let model = VitsModel::new().unwrap();
                        let result = model.synthesize(black_box(phonemes), None).await;
                        black_box(result)
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fastspeech2_long_sequence", length),
            &long_phonemes,
            |b, phonemes| {
                b.iter(|| {
                    rt.block_on(async {
                        let model = FastSpeech2Model::new();
                        let result = model.synthesize(black_box(phonemes), None).await;
                        black_box(result)
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent synthesis operations
fn bench_concurrent_synthesis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let phonemes = vec![
        Phoneme::new("h"),
        Phoneme::new("ɛ"),
        Phoneme::new("l"),
        Phoneme::new("oʊ"),
        Phoneme::new("w"),
        Phoneme::new("ɝ"),
        Phoneme::new("l"),
        Phoneme::new("d"),
    ];

    let mut group = c.benchmark_group("concurrent_synthesis");

    for concurrency in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("vits_concurrent", concurrency),
            &phonemes,
            |b, phonemes| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut tasks = Vec::new();
                        for _ in 0..*concurrency {
                            let phonemes_clone = phonemes.clone();
                            tasks.push(tokio::spawn(async move {
                                let model = VitsModel::new().unwrap();
                                model.synthesize(&phonemes_clone, None).await
                            }));
                        }

                        let results = futures::future::join_all(tasks).await;
                        black_box(results)
                    })
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_vits_synthesis,
    bench_fastspeech2_synthesis,
    bench_mel_operations,
    bench_batch_synthesis,
    bench_synthesis_configurations,
    bench_memory_performance,
    bench_mel_spectrogram_sizes,
    bench_stress_test,
    bench_concurrent_synthesis
);

criterion_main!(benches);
