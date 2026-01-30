//! Minimal benchmark suite for core spatial audio operations
//!
//! This suite benchmarks the most frequently used public APIs

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array1;
use voirs_spatial::Position3D;

/// Benchmark 3D distance calculations (most common operation)
fn bench_distance_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_calculations");

    for count in [100, 1000, 10000] {
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

/// Benchmark position vector operations
fn bench_position_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("position_operations");

    let positions: Vec<Position3D> = (0..1000)
        .map(|i| Position3D::new((i as f32).sin(), (i as f32).cos(), 1.0))
        .collect();

    group.bench_function("magnitude", |b| {
        b.iter(|| {
            for pos in &positions {
                black_box(pos.magnitude());
            }
        });
    });

    group.bench_function("normalized", |b| {
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
        let target = Position3D::new(5.0, 5.0, 2.0);
        b.iter(|| {
            for pos in &positions {
                black_box(pos.lerp(black_box(&target), 0.5));
            }
        });
    });

    group.finish();
}

/// Benchmark audio buffer operations
fn bench_audio_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_operations");

    for size in [512, 1024, 2048, 4096] {
        group.throughput(Throughput::Elements(size));

        group.bench_with_input(
            BenchmarkId::new("vec_to_array", size),
            &size,
            |b, &count| {
                let audio: Vec<f32> = (0..count).map(|i| (i as f32).sin()).collect();
                b.iter(|| {
                    let array = Array1::from_vec(black_box(audio.clone()));
                    black_box(array);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("array_scaling", size),
            &size,
            |b, &count| {
                let audio: Vec<f32> = (0..count).map(|i| (i as f32).sin()).collect();
                let array = Array1::from_vec(audio);
                b.iter(|| {
                    let scaled = &array * 0.8;
                    black_box(scaled);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark config creation
fn bench_config_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_creation");

    group.bench_function("spatial_config_default", |b| {
        use voirs_spatial::SpatialConfig;
        b.iter(|| {
            black_box(SpatialConfig::default());
        });
    });

    group.bench_function("spatial_config_builder", |b| {
        use voirs_spatial::SpatialConfigBuilder;
        b.iter(|| {
            let config = SpatialConfigBuilder::new()
                .sample_rate(48000)
                .buffer_size(512)
                .build();
            let _ = black_box(config);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_distance_calculations,
    bench_position_operations,
    bench_audio_operations,
    bench_config_creation,
);

criterion_main!(benches);
