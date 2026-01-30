//! Core spatial audio operation benchmarks
//!
//! This benchmark suite focuses on the most critical and frequently-used operations
//! in spatial audio processing.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array1;
use voirs_spatial::{Position3D, SIMDSpatialOps};

/// Benchmark 3D distance calculations
fn bench_distance_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_calculations");

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

/// Benchmark SIMD-accelerated distance calculations
fn bench_simd_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_distances");

    for count in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(BenchmarkId::new("batch", count), &count, |b, &n| {
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
        });
    }
    group.finish();
}

/// Benchmark vector normalization (SIMD)
fn bench_simd_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_normalization");

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
    group.finish();
}

/// Benchmark SIMD dot products
fn bench_simd_dot_products(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_dot_products");

    for count in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(BenchmarkId::new("batch", count), &count, |b, &n| {
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
        });
    }
    group.finish();
}

/// Benchmark audio buffer conversions
fn bench_audio_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_conversions");

    for sample_count in [512, 1024, 2048, 4096, 8192] {
        group.throughput(Throughput::Elements(sample_count));
        group.bench_with_input(
            BenchmarkId::new("vec_to_array", sample_count),
            &sample_count,
            |b, &count| {
                let audio_data: Vec<f32> = (0..count).map(|i| (i as f32).sin()).collect();

                b.iter(|| {
                    let array = Array1::from_vec(black_box(audio_data.clone()));
                    black_box(array);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("array_scaling", sample_count),
            &sample_count,
            |b, &count| {
                let audio_data: Vec<f32> = (0..count).map(|i| (i as f32).sin()).collect();
                let array = Array1::from_vec(audio_data);

                b.iter(|| {
                    let result = &array * 0.8;
                    black_box(result);
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

    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_memory_allocations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocations");

    // Test direct buffer allocation
    group.bench_function("vector_allocation", |b| {
        b.iter(|| {
            let buffer = vec![0.0f32; 1024];
            black_box(&buffer);
        });
    });

    // Test array allocation
    group.bench_function("array_allocation", |b| {
        b.iter(|| {
            let array = Array1::<f32>::zeros(1024);
            black_box(&array);
        });
    });

    // Test position vector allocation
    group.bench_function("position_vec_allocation", |b| {
        b.iter(|| {
            let positions: Vec<Position3D> = (0..100)
                .map(|i| Position3D::new((i as f32).sin(), (i as f32).cos(), 1.0))
                .collect();
            black_box(&positions);
        });
    });

    group.finish();
}

/// Benchmark batch vector operations
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    let positions: Vec<Position3D> = (0..1000)
        .map(|i| {
            let angle = (i as f32) * 0.01;
            Position3D::new(angle.cos() * 5.0, angle.sin() * 5.0, (angle * 2.0).sin())
        })
        .collect();

    group.bench_function("batch_normalize", |b| {
        b.iter(|| {
            let mut positions_copy = positions.clone();
            SIMDSpatialOps::normalize_batch(black_box(&mut positions_copy));
            black_box(positions_copy);
        });
    });

    let other_positions: Vec<Position3D> = (0..1000)
        .map(|i| {
            let angle = (i as f32) * 0.02;
            Position3D::new(angle.sin() * 3.0, angle.cos() * 3.0, 0.5)
        })
        .collect();

    group.bench_function("batch_dot_products", |b| {
        b.iter(|| {
            black_box(SIMDSpatialOps::dot_products(
                black_box(&positions),
                black_box(&other_positions),
            ));
        });
    });

    group.finish();
}

/// Benchmark interpolation operations
fn bench_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolation");

    let p1 = Position3D::new(0.0, 0.0, 0.0);
    let p2 = Position3D::new(5.0, 5.0, 2.0);

    group.bench_function("linear_interpolation", |b| {
        b.iter(|| {
            for i in 0..100 {
                let t = (i as f32) / 100.0;
                black_box(p1.lerp(black_box(&p2), black_box(t)));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_distance_calculations,
    bench_simd_distances,
    bench_simd_normalization,
    bench_simd_dot_products,
    bench_audio_conversions,
    bench_position_math,
    bench_memory_allocations,
    bench_batch_operations,
    bench_interpolation,
);

criterion_main!(benches);
