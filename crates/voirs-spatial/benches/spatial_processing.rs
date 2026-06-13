//! Benchmarks for core spatial audio processing operations
//!
//! This benchmark suite measures the performance of critical spatial audio operations
//! including distance calculations, SIMD operations, and basic room simulation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array1;
use std::time::Duration;
use voirs_spatial::{Position3D, RoomSimulator, SIMDSpatialOps, SpatialConfig};

/// Benchmark distance attenuation calculations
fn bench_distance_attenuation(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_attenuation");

    // Test different distance calculation counts
    for count in [100, 1000, 10000, 100000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &n| {
            let positions: Vec<Position3D> = (0..n)
                .map(|i| {
                    let angle = (i as f32) * 0.01;
                    Position3D::new(angle.cos() * 5.0, angle.sin() * 5.0, 1.0)
                })
                .collect();
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

/// Benchmark room acoustics simulation
fn bench_room_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("room_simulation");
    group.measurement_time(Duration::from_secs(10));

    // Test room creation
    group.bench_function("room_creation", |b| {
        b.iter(|| {
            black_box(RoomSimulator::new((10.0, 8.0, 3.0), 0.5 /* reverb_time */).ok());
        });
    });

    // Test reverb processing with correct API
    for buffer_size in [512, 1024, 2048] {
        group.throughput(Throughput::Elements(buffer_size * 2)); // Left + right channels
        group.bench_with_input(
            BenchmarkId::new("reverb_processing", buffer_size),
            &buffer_size,
            |b, &size| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.to_async(&rt).iter(|| async {
                    let mut room = RoomSimulator::new((10.0, 8.0, 3.0), 0.5).unwrap();
                    let mut left_channel =
                        Array1::from_vec((0..size).map(|i| (i as f32).sin()).collect());
                    let mut right_channel =
                        Array1::from_vec((0..size).map(|i| (i as f32).cos()).collect());
                    let source_pos = Position3D::new(2.0, 2.0, 1.5);

                    black_box(
                        room.process_reverb(&mut left_channel, &mut right_channel, &source_pos)
                            .await
                            .ok(),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SIMDaccelerator spatial math operations
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");

    // Test SIMD distance calculations
    for count in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(
            BenchmarkId::new("batch_distance", count),
            &count,
            |b, &n| {
                let positions: Vec<Position3D> = (0..n)
                    .map(|i| {
                        Position3D::new(
                            (i as f32).sin(),
                            (i as f32).cos(),
                            (i as f32).tan().abs().min(10.0),
                        )
                    })
                    .collect();

                b.iter(|| {
                    black_box(SIMDSpatialOps::distances(
                        black_box(Position3D::new(0.0, 0.0, 0.0)),
                        black_box(&positions),
                    ));
                });
            },
        );
    }

    // Test SIMD normalization
    for count in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(
            BenchmarkId::new("batch_normalize", count),
            &count,
            |b, &n| {
                let positions: Vec<Position3D> = (0..n)
                    .map(|i| Position3D::new((i as f32) * 0.1, (i as f32) * 0.2, (i as f32) * 0.3))
                    .collect();

                b.iter(|| {
                    let mut positions_copy = positions.clone();
                    SIMDSpatialOps::normalize_batch(black_box(&mut positions_copy));
                    black_box(positions_copy);
                });
            },
        );
    }

    // Test SIMD dot products
    for count in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(
            BenchmarkId::new("batch_dot_products", count),
            &count,
            |b, &n| {
                let positions_a: Vec<Position3D> = (0..n)
                    .map(|i| Position3D::new((i as f32).sin(), (i as f32).cos(), 1.0))
                    .collect();
                let positions_b: Vec<Position3D> = (0..n)
                    .map(|i| Position3D::new((i as f32).cos(), (i as f32).sin(), 0.5))
                    .collect();

                b.iter(|| {
                    black_box(SIMDSpatialOps::dot_products(
                        black_box(&positions_a),
                        black_box(&positions_b),
                    ));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark position math operations
fn bench_position_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("position_math");

    let count = 10000;
    group.throughput(Throughput::Elements(count));

    let positions: Vec<Position3D> = (0..count)
        .map(|i| Position3D::new((i as f32).sin(), (i as f32).cos(), 1.0))
        .collect();

    group.bench_function("magnitude", |b| {
        b.iter(|| {
            for pos in &positions {
                black_box(pos.magnitude());
            }
        });
    });

    group.bench_function("normalize", |b| {
        b.iter(|| {
            for pos in &positions {
                black_box(pos.normalized());
            }
        });
    });

    group.bench_function("dot_product", |b| {
        let other = Position3D::new(1.0, 0.0, 0.0);
        b.iter(|| {
            for pos in &positions {
                black_box(pos.dot(black_box(&other)));
            }
        });
    });

    group.bench_function("cross_product", |b| {
        let other = Position3D::new(0.0, 1.0, 0.0);
        b.iter(|| {
            for pos in &positions {
                black_box(pos.cross(black_box(&other)));
            }
        });
    });

    group.bench_function("lerp", |b| {
        let other = Position3D::new(5.0, 5.0, 5.0);
        b.iter(|| {
            for pos in &positions {
                black_box(pos.lerp(black_box(&other), 0.5));
            }
        });
    });

    group.finish();
}

/// Benchmark memory management operations
fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");

    // Test direct buffer allocation
    group.bench_function("vector_allocation_1k", |b| {
        b.iter(|| {
            let buffer = vec![0.0f32; 1024];
            black_box(&buffer);
        });
    });

    // Test Array1 allocation
    group.bench_function("array1_allocation_1k", |b| {
        b.iter(|| {
            let array = Array1::<f32>::zeros(1024);
            black_box(&array);
        });
    });

    // Test position vector allocation
    group.bench_function("position_vec_allocation_100", |b| {
        b.iter(|| {
            let positions: Vec<Position3D> = (0..100)
                .map(|i| Position3D::new((i as f32).sin(), (i as f32).cos(), 1.0))
                .collect();
            black_box(&positions);
        });
    });

    group.finish();
}

/// Benchmark audio format conversions
fn bench_audio_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_conversion");

    for sample_count in [512, 1024, 2048, 4096] {
        group.throughput(Throughput::Elements(sample_count));
        group.bench_with_input(
            BenchmarkId::from_parameter(sample_count),
            &sample_count,
            |b, &count| {
                let audio_data: Vec<f32> = (0..count).map(|i| (i as f32).sin()).collect();

                b.iter(|| {
                    // Convert to Array1 (common operation in processing)
                    let array = Array1::from_vec(black_box(audio_data.clone()));

                    // Perform some typical processing
                    let result = &array * 0.8;
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark spatial configuration creation
fn bench_config_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_creation");

    group.bench_function("default_config", |b| {
        b.iter(|| {
            black_box(SpatialConfig::default());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_distance_attenuation,
    bench_room_simulation,
    bench_simd_operations,
    bench_position_math,
    bench_memory_operations,
    bench_audio_conversion,
    bench_config_creation,
);

criterion_main!(benches);
