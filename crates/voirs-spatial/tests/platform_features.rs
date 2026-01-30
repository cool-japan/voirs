//! Platform-specific feature integration tests
//!
//! This test suite validates platform-specific functionality including:
//! - SIMD operations (SSE2, AVX, NEON)
//! - Cross-platform compatibility
//! - Performance characteristics
//! - Memory management

use voirs_spatial::{Position3D, SIMDSpatialOps, SpatialConfig};

#[test]
fn test_simd_distance_calculations() {
    // Test SIMD-accelerated distance calculations
    let positions = vec![
        Position3D::new(1.0, 0.0, 0.0),
        Position3D::new(0.0, 1.0, 0.0),
        Position3D::new(0.0, 0.0, 1.0),
        Position3D::new(1.0, 1.0, 1.0),
    ];

    let origin = Position3D::new(0.0, 0.0, 0.0);
    let distances = SIMDSpatialOps::distances(origin, &positions);

    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 1.0).abs() < 0.001);
    assert!((distances[1] - 1.0).abs() < 0.001);
    assert!((distances[2] - 1.0).abs() < 0.001);
    assert!((distances[3] - 1.732).abs() < 0.01); // sqrt(3)
}

#[test]
fn test_simd_normalization() {
    // Test SIMD-accelerated normalization
    let mut positions = vec![
        Position3D::new(2.0, 0.0, 0.0),
        Position3D::new(0.0, 3.0, 0.0),
        Position3D::new(0.0, 0.0, 4.0),
        Position3D::new(1.0, 1.0, 1.0),
    ];

    SIMDSpatialOps::normalize_batch(&mut positions);

    // All normalized vectors should have magnitude 1.0
    for pos in &positions {
        let magnitude = (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z).sqrt();
        assert!((magnitude - 1.0).abs() < 0.001, "Magnitude: {}", magnitude);
    }
}

#[test]
fn test_simd_dot_products() {
    // Test SIMD-accelerated dot products
    let positions_a = vec![
        Position3D::new(1.0, 0.0, 0.0),
        Position3D::new(0.0, 1.0, 0.0),
        Position3D::new(1.0, 1.0, 0.0),
    ];

    let positions_b = vec![
        Position3D::new(1.0, 0.0, 0.0),
        Position3D::new(0.0, 1.0, 0.0),
        Position3D::new(0.0, 1.0, 0.0),
    ];

    let dot_products = SIMDSpatialOps::dot_products(&positions_a, &positions_b);

    assert_eq!(dot_products.len(), 3);
    assert!((dot_products[0] - 1.0).abs() < 0.001);
    assert!((dot_products[1] - 1.0).abs() < 0.001);
    assert!((dot_products[2] - 1.0).abs() < 0.001);
}

#[test]
fn test_platform_consistency() {
    // Verify that SIMD and fallback implementations produce consistent results
    let positions = vec![
        Position3D::new(1.5, 2.3, -0.7),
        Position3D::new(-2.1, 0.5, 3.2),
        Position3D::new(0.0, 0.0, 0.0),
    ];

    let origin = Position3D::new(0.0, 0.0, 0.0);

    // Test distances
    let distances_simd = SIMDSpatialOps::distances(origin, &positions);
    let distances_fallback = SIMDSpatialOps::distances_fallback(origin, &positions);

    for (simd, fallback) in distances_simd.iter().zip(distances_fallback.iter()) {
        assert!((simd - fallback).abs() < 0.001);
    }

    // Test normalization
    let mut pos_simd = positions.clone();
    let mut pos_fallback = positions.clone();

    #[cfg(all(target_feature = "sse2", target_arch = "x86_64"))]
    {
        SIMDSpatialOps::normalize_batch_simd(&mut pos_simd);
    }

    #[cfg(not(all(target_feature = "sse2", target_arch = "x86_64")))]
    {
        SIMDSpatialOps::normalize_batch_fallback(&mut pos_simd);
    }

    SIMDSpatialOps::normalize_batch_fallback(&mut pos_fallback);

    for (simd, fallback) in pos_simd.iter().zip(pos_fallback.iter()) {
        assert!((simd.x - fallback.x).abs() < 0.001);
        assert!((simd.y - fallback.y).abs() < 0.001);
        assert!((simd.z - fallback.z).abs() < 0.001);
    }
}

#[test]
fn test_large_batch_simd_operations() {
    // Test SIMD operations with large batches
    let size = 10000;
    let positions: Vec<Position3D> = (0..size)
        .map(|i| {
            let angle = (i as f32) * 0.001;
            Position3D::new(angle.cos(), angle.sin(), (angle * 0.5).sin())
        })
        .collect();

    // Test distance calculations
    let origin = Position3D::new(0.0, 0.0, 0.0);
    let distances = SIMDSpatialOps::distances(origin, &positions);
    assert_eq!(distances.len(), size as usize);

    // Verify distances are reasonable
    for dist in &distances {
        assert!(*dist >= 0.0 && *dist <= 2.0);
    }

    // Test normalization
    let mut pos_normalized = positions.clone();
    SIMDSpatialOps::normalize_batch(&mut pos_normalized);

    for pos in &pos_normalized {
        let magnitude = (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z).sqrt();
        assert!((magnitude - 1.0).abs() < 0.01);
    }
}

#[test]
fn test_position_math_operations() {
    // Test Position3D math operations
    let p1 = Position3D::new(1.0, 2.0, 3.0);
    let p2 = Position3D::new(4.0, 5.0, 6.0);

    // Test distance
    let dist = p1.distance_to(&p2);
    let expected_dist = (3.0f32.powi(2) + 3.0f32.powi(2) + 3.0f32.powi(2)).sqrt();
    assert!((dist - expected_dist).abs() < 0.001);

    // Test dot product
    let dot = p1.dot(&p2);
    assert!((dot - 32.0).abs() < 0.001); // 1*4 + 2*5 + 3*6 = 32

    // Test cross product
    let cross = p1.cross(&p2);
    assert!((cross.x - (-3.0)).abs() < 0.001);
    assert!((cross.y - 6.0).abs() < 0.001);
    assert!((cross.z - (-3.0)).abs() < 0.001);

    // Test normalization
    let normalized = p1.normalized();
    let mag = normalized.magnitude();
    assert!((mag - 1.0).abs() < 0.001);

    // Test linear interpolation
    let lerp_mid = p1.lerp(&p2, 0.5);
    assert!((lerp_mid.x - 2.5).abs() < 0.001);
    assert!((lerp_mid.y - 3.5).abs() < 0.001);
    assert!((lerp_mid.z - 4.5).abs() < 0.001);
}

#[test]
fn test_spatial_config_creation() {
    // Test SpatialConfig creation and validation
    let config = SpatialConfig::default();

    assert!(config.sample_rate > 0);
    assert!(config.buffer_size > 0);
    assert!(config.max_distance > 0.0);
    assert!(config.speed_of_sound > 0.0);
}

#[test]
fn test_config_builder_validation() {
    // Test configuration builder with valid values
    use voirs_spatial::SpatialConfigBuilder;

    let config = SpatialConfigBuilder::new()
        .sample_rate(48000)
        .buffer_size(512)
        .room_dimensions(10.0, 8.0, 3.0)
        .build()
        .unwrap();

    assert_eq!(config.sample_rate, 48000);
    assert_eq!(config.buffer_size, 512);
    assert_eq!(config.room_dimensions, (10.0, 8.0, 3.0));
}

#[test]
fn test_position_default() {
    // Test Position3D default value
    let pos = Position3D::default();
    assert_eq!(pos.x, 0.0);
    assert_eq!(pos.y, 0.0);
    assert_eq!(pos.z, 0.0);
}

#[test]
fn test_simd_fallback_compatibility() {
    // Ensure fallback implementations work on all platforms
    let positions = vec![
        Position3D::new(1.0, 2.0, 3.0),
        Position3D::new(-1.0, -2.0, -3.0),
        Position3D::new(0.5, 0.5, 0.5),
    ];

    // Test fallback distance calculation
    let origin = Position3D::new(0.0, 0.0, 0.0);
    let distances = SIMDSpatialOps::distances_fallback(origin, &positions);

    assert_eq!(distances.len(), 3);
    for dist in &distances {
        assert!(*dist >= 0.0);
    }

    // Test fallback normalization
    let mut pos_fallback = positions.clone();
    SIMDSpatialOps::normalize_batch_fallback(&mut pos_fallback);

    for pos in &pos_fallback {
        let mag = pos.magnitude();
        assert!((mag - 1.0).abs() < 0.001);
    }

    // Test fallback dot products
    let dots = SIMDSpatialOps::dot_products_fallback(&positions, &positions);
    assert_eq!(dots.len(), 3);
}

#[test]
fn test_edge_cases() {
    // Test edge cases and boundary conditions

    // Zero vector normalization
    let zero = Position3D::new(0.0, 0.0, 0.0);
    let normalized = zero.normalized();
    assert_eq!(normalized.x, 0.0);
    assert_eq!(normalized.y, 0.0);
    assert_eq!(normalized.z, 0.0);

    // Very small values
    let tiny = Position3D::new(1e-10, 1e-10, 1e-10);
    let tiny_norm = tiny.normalized();
    assert!(tiny_norm.magnitude() <= 1.0);

    // Very large values
    let large = Position3D::new(1e6, 1e6, 1e6);
    let large_norm = large.normalized();
    assert!((large_norm.magnitude() - 1.0).abs() < 0.01);

    // Mixed sign values
    let mixed = Position3D::new(-1.0, 2.0, -3.0);
    let mixed_norm = mixed.normalized();
    assert!((mixed_norm.magnitude() - 1.0).abs() < 0.001);
}

#[test]
fn test_batch_size_variations() {
    // Test different batch sizes for SIMD operations
    for size in [1, 2, 3, 4, 5, 10, 16, 100, 1000] {
        let positions: Vec<Position3D> = (0..size)
            .map(|i| Position3D::new(i as f32, (i * 2) as f32, (i * 3) as f32))
            .collect();

        let origin = Position3D::new(0.0, 0.0, 0.0);
        let distances = SIMDSpatialOps::distances(origin, &positions);
        assert_eq!(distances.len(), size);

        let mut pos_norm = positions.clone();
        SIMDSpatialOps::normalize_batch(&mut pos_norm);
        assert_eq!(pos_norm.len(), size);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_x86_64_specific() {
    // Test x86_64-specific functionality
    println!("Running on x86_64 architecture");

    // Basic SIMD test
    let positions = vec![
        Position3D::new(1.0, 0.0, 0.0),
        Position3D::new(0.0, 1.0, 0.0),
    ];

    let origin = Position3D::new(0.0, 0.0, 0.0);
    let distances = SIMDSpatialOps::distances(origin, &positions);
    assert_eq!(distances.len(), 2);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_aarch64_specific() {
    // Test ARM64-specific functionality
    println!("Running on aarch64 architecture");

    // Basic SIMD test
    let positions = vec![
        Position3D::new(1.0, 0.0, 0.0),
        Position3D::new(0.0, 1.0, 0.0),
    ];

    let origin = Position3D::new(0.0, 0.0, 0.0);
    let distances = SIMDSpatialOps::distances(origin, &positions);
    assert_eq!(distances.len(), 2);
}

#[test]
fn test_memory_efficiency() {
    // Test memory usage patterns
    use std::mem::size_of;

    // Verify Position3D is compact
    assert_eq!(size_of::<Position3D>(), 12); // 3 * f32

    // Test large allocations don't panic
    let large_vec: Vec<Position3D> = (0..100000)
        .map(|i| Position3D::new(i as f32, i as f32, i as f32))
        .collect();

    assert_eq!(large_vec.len(), 100000);
}
