//! Scalability benchmarks
//!
//! Simplified benchmarks for dataset scaling, memory usage patterns,
//! processing time scaling, and resource utilization.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures::future;
use std::time::Instant;
use tokio::runtime::Runtime;
use voirs_dataset::datasets::dummy::{AudioType, DummyConfig, DummyDataset, TextType};
use voirs_dataset::traits::Dataset;
use voirs_dataset::AudioData;

/// Create test dataset of various sizes
fn create_test_dataset(size: usize, seed: u64) -> DummyDataset {
    let config = DummyConfig {
        num_samples: size,
        seed: Some(seed),
        audio_type: AudioType::Mixed,
        text_type: TextType::Lorem,
        ..Default::default()
    };

    DummyDataset::with_config(config)
}

/// Benchmark dataset loading with different sizes
fn bench_dataset_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("dataset_scaling");
    group.sample_size(10); // Fewer samples for large datasets

    let sizes = vec![50, 100, 200, 500];

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(BenchmarkId::new("load_dataset", size), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let dataset = create_test_dataset(std::hint::black_box(size), 42);

                    // Access a few samples to ensure they're loaded
                    for i in (0..size).step_by(size / 10 + 1) {
                        let sample = dataset.get(i).await.unwrap();
                        std::hint::black_box(&sample);
                    }

                    std::hint::black_box(dataset.len());
                });
            });
        });
    }

    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling");
    group.sample_size(10);

    // Test different audio durations (memory usage scales with duration)
    let durations = vec![1.0, 5.0, 10.0, 30.0, 60.0]; // seconds
    let sample_rate = 22050;

    for duration in durations {
        let expected_samples = (sample_rate as f32 * duration) as u64;
        group.throughput(Throughput::Elements(expected_samples));

        group.bench_function(BenchmarkId::new("audio_allocation", duration as u32), |b| {
            b.iter(|| {
                let sample_count = (sample_rate as f32 * std::hint::black_box(duration)) as usize;

                // Simulate audio allocation and processing
                let samples: Vec<f32> =
                    (0..sample_count).map(|i| (i as f32 * 0.01).sin()).collect();

                let audio = AudioData::new(samples, sample_rate, 1);

                // Simulate some processing to ensure memory is actually used
                let rms = audio.samples().iter().map(|&s| s * s).sum::<f32>()
                    / audio.samples().len() as f32;

                std::hint::black_box(rms);
            });
        });
    }

    // Test concurrent memory allocation
    group.bench_function("concurrent_allocation", |b| {
        let rt = Runtime::new().unwrap();
        b.iter(|| {
            rt.block_on(async {
                let tasks: Vec<_> = (0..10)
                    .map(|i| {
                        let size = 1000 + i * 100;
                        tokio::spawn(async move {
                            let samples: Vec<f32> =
                                (0..size).map(|j| (j as f32 * 0.01).sin()).collect();
                            AudioData::new(samples, 22050, 1)
                        })
                    })
                    .collect();

                let results = future::join_all(tasks).await;
                let total_samples: usize = results
                    .into_iter()
                    .map(|r| r.unwrap().samples().len())
                    .sum();

                std::hint::black_box(total_samples);
            });
        });
    });

    group.finish();
}

/// Benchmark parallel processing scaling
fn bench_parallel_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("parallel_processing");
    group.sample_size(10);

    let thread_counts = vec![1, 2, 4, 8];
    let dataset_size = 100;

    for thread_count in thread_counts {
        group.throughput(Throughput::Elements(dataset_size as u64));
        group.bench_function(BenchmarkId::new("parallel_processing", thread_count), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let dataset = create_test_dataset(dataset_size, 42);

                    // Create processing tasks
                    let tasks: Vec<_> = (0..dataset.len())
                        .map(|i| {
                            let dataset = &dataset;
                            async move {
                                let sample = dataset.get(i).await.unwrap();

                                // Simulate processing work
                                let mut audio = sample.audio;

                                // Apply some transformations
                                let rms = audio.samples().iter().map(|&s| s * s).sum::<f32>()
                                    / audio.samples().len() as f32;

                                // Normalize based on RMS
                                let gain = if rms > 0.0 { 0.7 / rms.sqrt() } else { 1.0 };
                                audio.samples_mut().iter_mut().for_each(|s| *s *= gain);

                                audio.samples().len()
                            }
                        })
                        .collect();

                    let start = Instant::now();
                    let results = future::join_all(tasks).await;
                    let duration = start.elapsed();

                    let total_samples: usize = results.into_iter().sum();
                    std::hint::black_box((total_samples, duration));
                });
            });
        });
    }

    group.finish();
}

/// Benchmark resource utilization
fn bench_resource_utilization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("resource_utilization");
    group.sample_size(10);

    // Test CPU utilization with different workloads
    let workload_sizes = vec![50, 100, 200, 500];

    for size in workload_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(BenchmarkId::new("cpu_intensive", size), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let dataset = create_test_dataset(size, 42);

                    let start = Instant::now();
                    let mut processed_count = 0;

                    for i in 0..dataset.len() {
                        let sample = dataset.get(i).await.unwrap();

                        // Simulate CPU-intensive processing
                        let mut audio = sample.audio.clone();
                        audio.normalize().unwrap();

                        // Apply gain
                        audio.samples_mut().iter_mut().for_each(|s| *s *= 1.2);

                        processed_count += 1;
                        std::hint::black_box(&audio);
                    }

                    let duration = start.elapsed();
                    let throughput = processed_count as f64 / duration.as_secs_f64();

                    std::hint::black_box(throughput);
                });
            });
        });
    }

    // Test memory pressure scenarios
    group.bench_function("memory_pressure", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut datasets = Vec::new();

                // Create multiple datasets to simulate memory pressure
                for i in 0..5 {
                    let dataset = create_test_dataset(50, i);
                    datasets.push(dataset);
                }

                // Perform operations on all datasets simultaneously
                let tasks: Vec<_> = datasets
                    .iter()
                    .enumerate()
                    .map(|(dataset_idx, dataset)| async move {
                        let mut sample_count = 0;

                        // Process every 5th sample
                        for i in (0..dataset.len()).step_by(5) {
                            let sample = dataset.get(i).await.unwrap();

                            // Simulate memory-intensive processing
                            let audio = sample.audio;
                            let samples = audio.samples().to_vec(); // Clone samples

                            // Perform processing (memory intensive)
                            let mut processed = samples.clone();
                            for (j, sample) in processed.iter_mut().enumerate() {
                                *sample *= (j as f32 * 0.001).cos();
                            }

                            sample_count += 1;
                        }

                        (dataset_idx, sample_count)
                    })
                    .collect();

                let results = future::join_all(tasks).await;
                let total_processed: usize = results.iter().map(|(_, count)| count).sum();

                std::hint::black_box(total_processed);
            });
        });
    });

    group.finish();
}

/// Benchmark processing time scaling
fn bench_processing_time_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("processing_time_scaling");
    group.sample_size(10);

    let processing_complexities = vec![("simple", 1), ("moderate", 3), ("complex", 5)];

    let dataset_sizes = vec![25, 50, 100, 200];

    for (complexity_name, operations_count) in &processing_complexities {
        for &size in &dataset_sizes {
            group.throughput(Throughput::Elements(size as u64));
            group.bench_function(
                BenchmarkId::from_parameter(format!("{complexity_name}_{size}_samples")),
                |b| {
                    b.iter(|| {
                        rt.block_on(async {
                            let dataset = create_test_dataset(size, 42);

                            let start = Instant::now();

                            for i in 0..dataset.len() {
                                let sample = dataset.get(i).await.unwrap();
                                let mut audio = sample.audio;

                                // Apply processing based on complexity
                                for _ in 0..*operations_count {
                                    // Simple operations: normalize, gain, basic math
                                    audio.normalize().unwrap();
                                    audio.samples_mut().iter_mut().for_each(|s| *s *= 1.1);

                                    // More complex math operations
                                    audio.samples_mut().iter_mut().enumerate().for_each(
                                        |(j, s)| {
                                            *s *= (j as f32 * 0.001).sin();
                                        },
                                    );
                                }

                                std::hint::black_box(&audio);
                            }

                            let duration = start.elapsed();
                            let samples_per_second = size as f64 / duration.as_secs_f64();

                            std::hint::black_box(samples_per_second);
                        });
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dataset_scaling,
    bench_memory_scaling,
    bench_parallel_scaling,
    bench_resource_utilization,
    bench_processing_time_scaling
);
criterion_main!(benches);
