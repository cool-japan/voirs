//! Advanced spatial audio performance benchmarks
//!
//! This benchmark suite covers critical performance paths in advanced spatial audio
//! processing including ambisonics, position calculations, and memory management.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array1;
use voirs_spatial::{
    ambisonics::{
        AmbisonicsDecoder, AmbisonicsEncoder, ChannelOrdering, NormalizationScheme,
        SpeakerConfiguration,
    },
    memory::{MemoryConfig, MemoryManager},
    Position3D,
};

/// Benchmark ambisonics encoding for different orders
fn bench_ambisonics_encoding_orders(c: &mut Criterion) {
    let mut group = c.benchmark_group("ambisonics_encoding_orders");

    for order in [1, 2, 3] {
        let channel_count = (order + 1) * (order + 1);
        group.throughput(Throughput::Elements(channel_count as u64));
        group.bench_with_input(BenchmarkId::new("order", order), &order, |b, &ord| {
            let encoder =
                AmbisonicsEncoder::new(ord, NormalizationScheme::N3D, ChannelOrdering::ACN);

            let audio = Array1::from_vec(vec![0.5f32; 1024]);
            let position = Position3D::new(1.0, 1.0, 1.0);

            b.iter(|| {
                let result = encoder.encode_mono(black_box(&audio), black_box(&position));
                let _ = black_box(result);
            });
        });
    }
    group.finish();
}

/// Benchmark ambisonics decoding for different speaker configurations
fn bench_ambisonics_decoding_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("ambisonics_decoding_configs");

    let configs = vec![
        ("stereo", SpeakerConfiguration::Stereo),
        ("quad", SpeakerConfiguration::Quadraphonic),
    ];

    for (name, config) in configs {
        group.bench_with_input(BenchmarkId::from_parameter(name), &config, |b, &cfg| {
            let decoder =
                AmbisonicsDecoder::for_speaker_config(2, cfg).expect("Failed to create decoder");

            // Create 2nd order ambisonics signal (9 channels)
            let ambi_signal = Array1::from_vec(vec![0.5f32; 1024 * 9]);
            let ambi_signal = ambi_signal.into_shape((9, 1024)).unwrap();

            b.iter(|| {
                let result = decoder.decode(black_box(&ambi_signal));
                let _ = black_box(result);
            });
        });
    }
    group.finish();
}

/// Benchmark memory buffer pool allocation patterns
fn bench_memory_buffer_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_buffer_pool");

    for buffer_size in [512, 1024, 2048, 4096] {
        group.throughput(Throughput::Bytes(buffer_size as u64 * 4)); // f32 = 4 bytes
        group.bench_with_input(
            BenchmarkId::from_parameter(buffer_size),
            &buffer_size,
            |b, &size| {
                let config = MemoryConfig {
                    max_buffer_pool_size: 16,
                    max_cache_size: 100,
                    enable_monitoring: true,
                    memory_pressure_threshold: 0.8,
                    cache_policy: voirs_spatial::memory::CachePolicy::LRU,
                    buffer_alignment: 64,
                };

                let memory_manager = MemoryManager::new(config);

                b.iter(|| {
                    // Simulate memory allocation patterns
                    // Create temporary vectors of the specified size
                    let buffer1 = vec![0.0f32; black_box(size)];
                    let buffer2 = vec![0.0f32; black_box(size)];
                    let buffer3 = vec![0.0f32; black_box(size)];
                    // Use the buffers to prevent optimization
                    let _ = (buffer1.len(), buffer2.len(), buffer3.len());
                    // Buffers are automatically released when they go out of scope
                });
            },
        );
    }
    group.finish();
}

/// Benchmark Position3D vector operations at scale
fn bench_position_vector_ops_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("position_vector_ops_batch");

    for count in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(count));

        let positions: Vec<Position3D> = (0..count)
            .map(|i| Position3D::new((i as f32).sin(), (i as f32).cos(), 1.0))
            .collect();

        group.bench_with_input(BenchmarkId::new("add_batch", count), &count, |b, _| {
            b.iter(|| {
                let mut result = Position3D::new(0.0, 0.0, 0.0);
                for pos in &positions {
                    result = result.add(black_box(pos));
                }
                black_box(result);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("normalize_batch", count),
            &count,
            |b, _| {
                b.iter(|| {
                    for pos in &positions {
                        black_box(pos.normalized());
                    }
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("distance_batch", count), &count, |b, _| {
            let listener = Position3D::new(0.0, 0.0, 0.0);
            b.iter(|| {
                for pos in &positions {
                    black_box(pos.distance_to(black_box(&listener)));
                }
            });
        });
    }
    group.finish();
}

/// Benchmark spherical coordinate conversions for ambisonics
fn bench_spherical_coordinate_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("spherical_coordinate_conversion");

    for count in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &n| {
            let positions: Vec<Position3D> = (0..n)
                .map(|i| {
                    let theta = (i as f32) * 2.0 * std::f32::consts::PI / (n as f32);
                    let phi = (i as f32) * std::f32::consts::PI / (n as f32);
                    Position3D::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos())
                })
                .collect();

            b.iter(|| {
                for pos in &positions {
                    // Simulate spherical coordinate computation
                    let r = pos.magnitude();
                    let theta = (pos.z / r).acos();
                    let phi = pos.y.atan2(pos.x);
                    black_box((r, theta, phi));
                }
            });
        });
    }
    group.finish();
}

/// Benchmark cross product operations for spatial calculations
fn bench_cross_product_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_product_operations");

    for count in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &n| {
            let positions_a: Vec<Position3D> = (0..n)
                .map(|i| Position3D::new((i as f32).sin(), (i as f32).cos(), 1.0))
                .collect();
            let positions_b: Vec<Position3D> = (0..n)
                .map(|i| Position3D::new((i as f32).cos(), (i as f32).sin(), 0.5))
                .collect();

            b.iter(|| {
                for (pos_a, pos_b) in positions_a.iter().zip(&positions_b) {
                    black_box(pos_a.cross(black_box(pos_b)));
                }
            });
        });
    }
    group.finish();
}

/// Benchmark LERP interpolation for smooth motion
fn bench_lerp_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("lerp_interpolation");

    for count in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &n| {
            let start_positions: Vec<Position3D> = (0..n)
                .map(|i| Position3D::new((i as f32) * 0.1, 0.0, 0.0))
                .collect();
            let end_positions: Vec<Position3D> = (0..n)
                .map(|i| Position3D::new(0.0, (i as f32) * 0.1, 0.0))
                .collect();

            b.iter(|| {
                for (start, end) in start_positions.iter().zip(&end_positions) {
                    // Interpolate at various t values
                    for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
                        black_box(start.lerp(black_box(end), black_box(t)));
                    }
                }
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ambisonics_encoding_orders,
    bench_ambisonics_decoding_configs,
    bench_memory_buffer_pool,
    bench_position_vector_ops_batch,
    bench_spherical_coordinate_conversion,
    bench_cross_product_operations,
    bench_lerp_interpolation,
);
criterion_main!(benches);
