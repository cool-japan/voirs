//! Benchmarks for audio preprocessing optimizations
//!
//! This benchmark suite compares the performance of:
//! - Standard preprocessing vs optimized preprocessing
//! - SIMD operations vs scalar operations
//! - Memory pooling vs standard allocation
//! - Batch processing vs sequential processing

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use voirs_recognizer::preprocessing::*;
use voirs_sdk::AudioBuffer;

/// Benchmark standard preprocessing
fn benchmark_standard_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard_preprocessing");

    for size in [1000, 4000, 16000, 48000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let config = AudioPreprocessingConfig {
                noise_suppression: true,
                agc: true,
                echo_cancellation: false,
                bandwidth_extension: false,
                advanced_spectral: true,
                adaptive_algorithms: false,
                sample_rate: 16000,
                buffer_size: 1024,
                advanced_spectral_config: Some(AdvancedSpectralConfig::default()),
                adaptive_config: None,
            };
            let samples = vec![0.1f32; size];
            let audio = AudioBuffer::mono(samples, 16000);
            let runtime = tokio::runtime::Runtime::new().unwrap();

            b.to_async(&runtime).iter(|| {
                let mut preprocessor = AudioPreprocessor::new(config.clone()).unwrap();
                let audio = audio.clone();
                async move { black_box(preprocessor.process(&audio).await.unwrap()) }
            });
        });
    }

    group.finish();
}

/// Benchmark optimized preprocessing
fn benchmark_optimized_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimized_preprocessing");

    for size in [1000, 4000, 16000, 48000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let config = OptimizedPreprocessingConfig {
                base_config: AudioPreprocessingConfig {
                    noise_suppression: true,
                    agc: true,
                    echo_cancellation: false,
                    bandwidth_extension: false,
                    advanced_spectral: true,
                    adaptive_algorithms: false,
                    sample_rate: 16000,
                    buffer_size: 1024,
                    advanced_spectral_config: Some(AdvancedSpectralConfig::default()),
                    adaptive_config: None,
                },
                enable_memory_pooling: true,
                enable_batch_processing: true,
                batch_config: BatchProcessingConfig::default(),
                enable_simd: true,
                enable_lockfree_channels: true,
            };
            let samples = vec![0.1f32; size];
            let audio = AudioBuffer::mono(samples, 16000);
            let runtime = tokio::runtime::Runtime::new().unwrap();

            b.to_async(&runtime).iter(|| {
                let mut preprocessor = OptimizedAudioPreprocessor::new(config.clone()).unwrap();
                let audio = audio.clone();
                async move { black_box(preprocessor.process_optimized(&audio).await.unwrap()) }
            });
        });
    }

    group.finish();
}

/// Benchmark SIMD operations
fn benchmark_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");

    for size in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(size as u64));

        // Apply gain
        group.bench_with_input(BenchmarkId::new("apply_gain", size), &size, |b, &size| {
            let mut samples = vec![0.5f32; size];
            b.iter(|| SimdAudioOps::apply_gain_simd(black_box(&mut samples), black_box(2.0)));
        });

        // Normalize
        group.bench_with_input(BenchmarkId::new("normalize", size), &size, |b, &size| {
            let mut samples = vec![0.5f32; size];
            b.iter(|| SimdAudioOps::normalize_simd(black_box(&mut samples)));
        });

        // RMS computation
        group.bench_with_input(BenchmarkId::new("compute_rms", size), &size, |b, &size| {
            let samples = vec![0.5f32; size];
            b.iter(|| SimdAudioOps::compute_rms_simd(black_box(&samples)));
        });

        // DC offset removal
        group.bench_with_input(
            BenchmarkId::new("remove_dc_offset", size),
            &size,
            |b, &size| {
                let mut samples = vec![0.5f32; size];
                b.iter(|| SimdAudioOps::remove_dc_offset_simd(black_box(&mut samples)));
            },
        );
    }

    group.finish();
}

/// Benchmark memory pooling
fn benchmark_memory_pooling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pooling");

    // With pooling
    group.bench_function("with_pooling", |b| {
        let pool = AudioBufferPool::new(1024, 8);
        b.iter(|| {
            let buffer = pool.acquire();
            // Simulate some work
            let _sum: f32 = buffer.iter().sum();
            pool.release(buffer);
        });
    });

    // Without pooling
    group.bench_function("without_pooling", |b| {
        b.iter(|| {
            let buffer: Vec<f32> = Vec::with_capacity(1024);
            // Simulate some work
            let _sum: f32 = buffer.iter().sum();
            drop(buffer);
        });
    });

    group.finish();
}

/// Benchmark batch processing
fn benchmark_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");

    for batch_size in [1, 4, 8, 16] {
        group.throughput(Throughput::Elements(batch_size as u64 * 1000));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                let config = BatchProcessingConfig {
                    batch_size,
                    parallel: true,
                    num_threads: num_cpus::get(),
                };
                let processor = BatchAudioProcessor::new(config);
                let buffers: Vec<_> = (0..batch_size)
                    .map(|_| AudioBuffer::mono(vec![0.1; 1000], 16000))
                    .collect();

                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| async {
                        black_box(
                            processor
                                .process_batch(buffers.clone(), |buffer| {
                                    let mut samples = buffer.samples().to_vec();
                                    SimdAudioOps::apply_gain_simd(&mut samples, 2.0);
                                    Ok(AudioBuffer::mono(samples, buffer.sample_rate()))
                                })
                                .await
                                .unwrap(),
                        )
                    });
            },
        );
    }

    group.finish();
}

/// Benchmark cache-optimized ring buffer
fn benchmark_ring_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer");

    // Write performance
    group.bench_function("write", |b| {
        let mut buffer = CacheOptimizedRingBuffer::new(16384);
        let data = vec![0.5f32; 1024];
        b.iter(|| {
            buffer.write(black_box(&data)).unwrap();
        });
    });

    // Read performance
    group.bench_function("read", |b| {
        let mut buffer = CacheOptimizedRingBuffer::new(16384);
        let data = vec![0.5f32; 16384];
        buffer.write(&data).unwrap();
        let mut output = vec![0.0f32; 1024];
        b.iter(|| {
            buffer.read(black_box(&mut output)).unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_standard_preprocessing,
    benchmark_optimized_preprocessing,
    benchmark_simd_operations,
    benchmark_memory_pooling,
    benchmark_batch_processing,
    benchmark_ring_buffer,
);
criterion_main!(benches);
