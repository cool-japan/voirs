//! Room Acoustics Simulation Benchmark Suite
//!
//! Benchmarks for room acoustics processing including:
//! - Ray tracing calculations
//! - Reflection processing
//! - Reverb generation
//! - Occlusion detection

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use voirs_spatial::Position3D;

/// Benchmark room simulation creation with different configurations
fn bench_room_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("room_creation");

    let room_sizes = vec![
        ("small_room", 4.0, 3.0, 2.5),   // Small bedroom
        ("medium_room", 6.0, 5.0, 3.0),  // Living room
        ("large_room", 15.0, 10.0, 4.0), // Concert hall
        ("huge_room", 30.0, 20.0, 8.0),  // Theater
    ];

    for (name, width, length, height) in room_sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(width, length, height),
            |b, &(w, l, h)| {
                b.iter(|| {
                    // Simulate room configuration (simplified)
                    let room_volume = w * l * h;
                    let surface_area = 2.0 * (w * l + w * h + l * h);
                    let config = (room_volume, surface_area);
                    black_box(config);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark ray tracing with different ray counts
fn bench_ray_tracing(c: &mut Criterion) {
    let mut group = c.benchmark_group("ray_tracing");

    for num_rays in [100, 500, 1000, 2000, 5000] {
        group.throughput(Throughput::Elements(num_rays));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_rays),
            &num_rays,
            |b, &count| {
                let source = Position3D::new(2.0, 2.0, 1.5);
                let listener = Position3D::new(4.0, 3.0, 1.5);

                b.iter(|| {
                    // Simulate ray tracing
                    let mut total_energy = 0.0f32;
                    for i in 0..count {
                        let angle = (i as f32) * 2.0 * std::f32::consts::PI / (count as f32);
                        let elevation = (i as f32) * std::f32::consts::PI / (count as f32);

                        // Ray direction
                        let dir_x = angle.cos() * elevation.sin();
                        let dir_y = angle.sin() * elevation.sin();
                        let dir_z = elevation.cos();

                        // Simulate ray-surface intersection
                        let ray_length = (dir_x * dir_x + dir_y * dir_y + dir_z * dir_z).sqrt();
                        let energy = 1.0 / (1.0 + ray_length);
                        total_energy += energy;
                    }
                    black_box(total_energy);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark reflection calculations with varying reflection orders
fn bench_reflection_orders(c: &mut Criterion) {
    let mut group = c.benchmark_group("reflection_orders");

    for max_reflections in [1, 2, 3, 4, 5] {
        group.bench_with_input(
            BenchmarkId::from_parameter(max_reflections),
            &max_reflections,
            |b, &order| {
                let source = Position3D::new(2.0, 2.0, 1.5);
                let listener = Position3D::new(4.0, 3.0, 1.5);
                let room_dimensions = (6.0, 5.0, 3.0);

                b.iter(|| {
                    let mut total_energy = 1.0f32; // Direct path
                    let absorption = 0.3f32; // Wall absorption coefficient

                    for reflection_order in 1..=order {
                        // Simulate energy loss through multiple reflections
                        let energy = (1.0f32 - absorption).powi(reflection_order);
                        total_energy += energy * 0.5; // Attenuated reflected energy
                    }

                    black_box(total_energy);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark reverb tail generation
fn bench_reverb_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverb_generation");

    for buffer_size in [256usize, 512, 1024, 2048] {
        group.throughput(Throughput::Elements(buffer_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(buffer_size),
            &buffer_size,
            |b, &size| {
                let dry_signal_vec: Vec<f32> =
                    (0..size).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();

                b.iter(|| {
                    let mut output = dry_signal_vec.clone();

                    // Simulate simple reverb (comb filter + all-pass)
                    let delay_samples: usize = 100;
                    let feedback = 0.5;

                    for i in delay_samples..size {
                        output[i] += output[i - delay_samples] * feedback;
                    }

                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark occlusion detection calculations
fn bench_occlusion_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("occlusion_detection");

    let source = Position3D::new(2.0, 2.0, 1.5);
    let listener = Position3D::new(4.0, 3.0, 1.5);

    // Different obstacle configurations
    let obstacle_counts = vec![0, 1, 5, 10, 20];

    for &num_obstacles in &obstacle_counts {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_obstacles", num_obstacles)),
            &num_obstacles,
            |b, &count| {
                let obstacles: Vec<Position3D> = (0..count)
                    .map(|i| Position3D::new(3.0 + (i as f32) * 0.1, 2.5, 1.5))
                    .collect();

                b.iter(|| {
                    let mut total_occlusion = 0.0f32;

                    for obstacle in &obstacles {
                        // Simple line-sphere intersection test
                        let to_obstacle = Position3D::new(
                            obstacle.x - source.x,
                            obstacle.y - source.y,
                            obstacle.z - source.z,
                        );

                        let ray_dir = Position3D::new(
                            listener.x - source.x,
                            listener.y - source.y,
                            listener.z - source.z,
                        )
                        .normalized();

                        // Check if obstacle is in the path (simplified)
                        let dot = to_obstacle.dot(&ray_dir);
                        if dot > 0.0 && dot < source.distance_to(&listener) {
                            total_occlusion += 0.2; // Each obstacle adds 20% occlusion
                        }
                    }

                    total_occlusion = total_occlusion.min(1.0); // Cap at 100%
                    black_box(total_occlusion);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark multi-room acoustics simulation
fn bench_multiroom_acoustics(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiroom_acoustics");

    for num_rooms in [1, 2, 3, 5] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_rooms", num_rooms)),
            &num_rooms,
            |b, &count| {
                let source = Position3D::new(2.0, 2.0, 1.5);
                let listener = Position3D::new(10.0, 10.0, 1.5);

                b.iter(|| {
                    let mut total_energy = 1.0f32;

                    // Simulate sound propagation through multiple rooms
                    for room_idx in 0..count {
                        let door_transmission = 0.5; // 50% transmission through doorways
                        let wall_transmission = 0.1; // 10% transmission through walls

                        // Alternate between door and wall transmission
                        let transmission = if room_idx % 2 == 0 {
                            door_transmission
                        } else {
                            wall_transmission
                        };

                        total_energy *= transmission;
                    }

                    black_box(total_energy);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_room_creation,
    bench_ray_tracing,
    bench_reflection_orders,
    bench_reverb_generation,
    bench_occlusion_detection,
    bench_multiroom_acoustics,
);
criterion_main!(benches);
