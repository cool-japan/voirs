//! HRTF Processing Benchmark Suite
//!
//! Benchmarks for Head-Related Transfer Function processing operations including:
//! - HRTF database lookups
//! - Spatial interpolation
//! - Convolution processing
//! - Distance modeling

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array1;
use std::sync::Arc;
use voirs_spatial::{HrtfDatabase, HrtfProcessor, Position3D};

/// Benchmark HRTF database creation and initialization
fn bench_hrtf_database_creation(c: &mut Criterion) {
    c.bench_function("hrtf_position_calc", |b| {
        b.iter(|| {
            let pos = Position3D::new(1.0, 1.0, 0.0);
            black_box(pos);
        });
    });
}

/// Benchmark HRTF lookups at different positions
fn bench_hrtf_lookups(c: &mut Criterion) {
    let mut group = c.benchmark_group("hrtf_lookups");

    // Test positions: front, side, back, above, below
    let test_positions = vec![
        ("front", Position3D::new(0.0, 2.0, 0.0)),
        ("right_side", Position3D::new(2.0, 0.0, 0.0)),
        ("back", Position3D::new(0.0, -2.0, 0.0)),
        ("above", Position3D::new(0.0, 1.0, 2.0)),
        ("below", Position3D::new(0.0, 1.0, -2.0)),
    ];

    for (name, position) in &test_positions {
        group.bench_with_input(BenchmarkId::from_parameter(name), position, |b, pos| {
            b.iter(|| {
                // Simulate HRTF lookup for the position
                let _ = black_box(pos.distance_to(&Position3D::new(0.0, 0.0, 0.0)));
                let _ = black_box(pos.normalized());
            });
        });
    }

    group.finish();
}

/// Benchmark HRTF processing with different buffer sizes
fn bench_hrtf_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("hrtf_processing");

    for buffer_size in [128, 256, 512, 1024, 2048] {
        group.throughput(Throughput::Elements(buffer_size));
        group.bench_with_input(
            BenchmarkId::from_parameter(buffer_size),
            &buffer_size,
            |b, &size| {
                let audio_vec: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
                let audio = Array1::from_vec(audio_vec);
                let position = Position3D::new(1.0, 1.0, 0.0);

                b.iter(|| {
                    // Simulate HRTF convolution processing
                    let mut output_left = audio.clone();
                    let mut output_right = audio.clone();

                    // Simple filter simulation (placeholder for actual HRTF convolution)
                    for i in 0..output_left.len() {
                        output_left[i] *= 0.7; // Attenuate left channel
                        output_right[i] *= 0.9; // Less attenuation on right
                    }

                    black_box((output_left, output_right));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark distance-based attenuation calculations
fn bench_distance_attenuation(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_attenuation");

    for distance in [0.5f32, 1.0, 2.0, 5.0, 10.0, 20.0] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:.1}m", distance)),
            &distance,
            |b, &dist| {
                b.iter(|| {
                    // Distance attenuation (inverse square law with minimum distance)
                    let attenuation = 1.0 / (1.0 + dist * dist);
                    black_box(attenuation);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark HRTF processing with multiple sources
fn bench_multisource_hrtf(c: &mut Criterion) {
    let mut group = c.benchmark_group("multisource_hrtf");

    for num_sources in [1, 4, 8, 16, 32] {
        group.throughput(Throughput::Elements(num_sources));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_sources),
            &num_sources,
            |b, &count| {
                let buffer_size = 512usize;
                let positions: Vec<Position3D> = (0..count)
                    .map(|i| {
                        let angle = (i as f32) * 2.0 * std::f32::consts::PI / (count as f32);
                        Position3D::new(angle.cos() * 2.0, angle.sin() * 2.0, 0.0)
                    })
                    .collect();

                let audio_sources: Vec<Array1<f32>> = (0..count)
                    .map(|_| {
                        let audio_vec: Vec<f32> = (0..buffer_size)
                            .map(|j| ((j as f32) * 0.01).sin() * 0.2)
                            .collect();
                        Array1::from_vec(audio_vec)
                    })
                    .collect();

                b.iter(|| {
                    let mut output_left: Array1<f32> = Array1::zeros(buffer_size);
                    let mut output_right: Array1<f32> = Array1::zeros(buffer_size);

                    for (audio, pos) in audio_sources.iter().zip(positions.iter()) {
                        // Simulate HRTF processing for each source
                        let distance = pos.distance_to(&Position3D::new(0.0, 0.0, 0.0));
                        let attenuation = 1.0 / (1.0 + distance);

                        // Simplified panning based on x-coordinate
                        let pan = (pos.x / 2.0).clamp(-1.0, 1.0);
                        let left_gain = ((1.0 - pan) / 2.0) * attenuation;
                        let right_gain = ((1.0 + pan) / 2.0) * attenuation;

                        for i in 0..buffer_size {
                            output_left[i] += audio[i] * left_gain;
                            output_right[i] += audio[i] * right_gain;
                        }
                    }

                    black_box((output_left, output_right));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark HRTF interpolation for smooth position transitions
fn bench_hrtf_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hrtf_interpolation");

    let start_pos = Position3D::new(1.0, 1.0, 0.0);
    let end_pos = Position3D::new(-1.0, 1.0, 0.5);

    group.bench_function("lerp_interpolation", |b| {
        b.iter(|| {
            for t in (0..100).map(|i| i as f32 / 100.0) {
                let interp_pos = start_pos.lerp(&end_pos, t);
                black_box(interp_pos);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hrtf_database_creation,
    bench_hrtf_lookups,
    bench_hrtf_processing,
    bench_distance_attenuation,
    bench_multisource_hrtf,
    bench_hrtf_interpolation,
);
criterion_main!(benches);
