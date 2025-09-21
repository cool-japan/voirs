//! Processing performance benchmarks
//!
//! Simplified benchmarks for audio loading, basic processing operations,
//! and memory usage patterns.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use tempfile::TempDir;
use tokio::runtime::Runtime;
use voirs_dataset::audio::io::{load_audio, save_audio};
use voirs_dataset::datasets::dummy::DummyDataset;
use voirs_dataset::traits::Dataset;
use voirs_dataset::AudioData;

/// Create test audio data of various sizes
fn create_test_audio(sample_rate: u32, duration_secs: f32, channels: u32) -> AudioData {
    let sample_count = (sample_rate as f32 * duration_secs * channels as f32) as usize;
    let samples: Vec<f32> = (0..sample_count)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect();
    AudioData::new(samples, sample_rate, channels)
}

/// Benchmark audio loading throughput
fn bench_audio_loading(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();

    // Create test files of different sizes
    let test_files = vec![
        ("1sec_22khz", create_test_audio(22050, 1.0, 1)),
        ("5sec_22khz", create_test_audio(22050, 5.0, 1)),
        ("10sec_22khz", create_test_audio(22050, 10.0, 1)),
        ("1sec_48khz", create_test_audio(48000, 1.0, 1)),
        ("5sec_48khz", create_test_audio(48000, 5.0, 1)),
        ("10sec_48khz", create_test_audio(48000, 10.0, 1)),
    ];

    // Save test files
    let mut file_paths = Vec::new();
    for (name, audio) in &test_files {
        let path = temp_dir.path().join(format!("{name}.wav"));
        save_audio(audio, &path).unwrap();
        file_paths.push(((*name).to_string(), path, audio.samples().len()));
    }

    let mut group = c.benchmark_group("audio_loading");

    for (name, path, sample_count) in file_paths {
        group.throughput(Throughput::Elements(sample_count as u64));
        group.bench_function(name, |b| {
            b.iter(|| {
                let audio = load_audio(std::hint::black_box(&path)).unwrap();
                std::hint::black_box(audio);
            });
        });
    }

    group.finish();
}

/// Benchmark basic audio processing operations
fn bench_audio_processing(c: &mut Criterion) {
    let test_audio = create_test_audio(22050, 5.0, 1);
    let samples_per_sec = test_audio.sample_rate() as u64;

    let mut group = c.benchmark_group("audio_processing");
    group.throughput(Throughput::Elements(samples_per_sec));

    // Benchmark normalization
    group.bench_function("normalize", |b| {
        b.iter(|| {
            let mut audio = test_audio.clone();
            audio.normalize().unwrap();
            std::hint::black_box(audio);
        });
    });

    // Benchmark gain application (simple multiplication)
    group.bench_function("apply_gain", |b| {
        b.iter(|| {
            let mut audio = test_audio.clone();
            let gain = std::hint::black_box(1.5);
            audio.samples_mut().iter_mut().for_each(|s| *s *= gain);
            std::hint::black_box(audio);
        });
    });

    // Benchmark resampling
    group.bench_function("resample_22k_to_16k", |b| {
        b.iter(|| {
            let audio = test_audio.clone();
            let resampled = audio.resample(16000).unwrap();
            std::hint::black_box(resampled);
        });
    });

    group.bench_function("resample_22k_to_48k", |b| {
        b.iter(|| {
            let audio = test_audio.clone();
            let resampled = audio.resample(48000).unwrap();
            std::hint::black_box(resampled);
        });
    });

    group.finish();
}

/// Benchmark dataset access patterns
fn bench_dataset_access(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let datasets = vec![
        ("small", DummyDataset::small()),
        ("large", DummyDataset::large()),
    ];

    let mut group = c.benchmark_group("dataset_access");
    group.sample_size(10);

    for (name, dataset) in datasets {
        group.throughput(Throughput::Elements(dataset.len() as u64));
        group.bench_function(format!("sequential_access_{name}"), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let mut total_duration = 0.0f32;
                    for i in 0..dataset.len() {
                        let sample = dataset.get(i).await.unwrap();
                        total_duration += sample.duration();
                        std::hint::black_box(&sample);
                    }
                    std::hint::black_box(total_duration);
                });
            });
        });

        group.bench_function(format!("random_access_{name}"), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let len = dataset.len();
                    let indices: Vec<usize> = {
                        use rand::{thread_rng, Rng};
                        let mut rng = thread_rng();
                        (0..len).map(|_| rng.gen_range(0..len)).collect()
                    };
                    let mut total_duration = 0.0f32;
                    for &i in &indices {
                        let sample = dataset.get(i).await.unwrap();
                        total_duration += sample.duration();
                        std::hint::black_box(&sample);
                    }
                    std::hint::black_box(total_duration);
                });
            });
        });
    }

    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    // Test memory allocation patterns for different audio sizes
    let sizes = vec![
        ("small_1sec", 22050),
        ("medium_10sec", 220500),
        ("large_60sec", 1323000),
        ("xlarge_300sec", 6615000),
    ];

    for (name, sample_count) in sizes {
        group.throughput(Throughput::Elements(sample_count as u64));
        group.bench_function(format!("allocation_{name}"), |b| {
            b.iter(|| {
                let samples: Vec<f32> =
                    (0..sample_count).map(|i| (i as f32 * 0.01).sin()).collect();
                let audio = AudioData::new(std::hint::black_box(samples), 22050, 1);
                std::hint::black_box(audio);
            });
        });
    }

    // Test memory reuse patterns
    group.bench_function("memory_reuse", |b| {
        b.iter(|| {
            // Simulate memory reuse by creating new audio data
            let new_samples: Vec<f32> = (0..110250).map(|i| (i as f32 * 0.02).cos()).collect();
            let audio = AudioData::new(std::hint::black_box(new_samples), 22050, 1);
            std::hint::black_box(&audio);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_audio_loading,
    bench_audio_processing,
    bench_dataset_access,
    bench_memory_usage
);
criterion_main!(benches);
