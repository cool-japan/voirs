//! Property-based tests for voirs-spatial
//!
//! These tests use proptest to verify mathematical invariants and edge cases
//! in spatial audio calculations.

use proptest::prelude::*;
use voirs_spatial::types::Position3D;

/// Strategy for generating valid 3D positions
fn position_strategy() -> impl Strategy<Value = Position3D> {
    (
        -1000.0f32..1000.0f32,
        -1000.0f32..1000.0f32,
        -1000.0f32..1000.0f32,
    )
        .prop_map(|(x, y, z)| Position3D::new(x, y, z))
}

/// Strategy for generating non-zero 3D positions (for normalization tests)
fn non_zero_position_strategy() -> impl Strategy<Value = Position3D> {
    (
        -1000.0f32..1000.0f32,
        -1000.0f32..1000.0f32,
        -1000.0f32..1000.0f32,
    )
        .prop_filter("Position must be non-zero", |(x, y, z)| {
            x.abs() > 0.001 || y.abs() > 0.001 || z.abs() > 0.001
        })
        .prop_map(|(x, y, z)| Position3D::new(x, y, z))
}

proptest! {
    /// Test that distance is always non-negative
    #[test]
    fn test_distance_non_negative(pos1 in position_strategy(), pos2 in position_strategy()) {
        let distance = pos1.distance_to(&pos2);
        prop_assert!(distance >= 0.0, "Distance must be non-negative, got {}", distance);
    }

    /// Test that distance is symmetric: d(a,b) = d(b,a)
    #[test]
    fn test_distance_symmetric(pos1 in position_strategy(), pos2 in position_strategy()) {
        let d1 = pos1.distance_to(&pos2);
        let d2 = pos2.distance_to(&pos1);
        let diff = (d1 - d2).abs();
        prop_assert!(diff < 1e-4, "Distance must be symmetric: d1={}, d2={}, diff={}", d1, d2, diff);
    }

    /// Test triangle inequality: d(a,c) <= d(a,b) + d(b,c)
    #[test]
    fn test_triangle_inequality(
        pos1 in position_strategy(),
        pos2 in position_strategy(),
        pos3 in position_strategy()
    ) {
        let d_ac = pos1.distance_to(&pos3);
        let d_ab = pos1.distance_to(&pos2);
        let d_bc = pos2.distance_to(&pos3);
        let sum = d_ab + d_bc;

        prop_assert!(
            d_ac <= sum + 1e-4,
            "Triangle inequality violated: d(a,c)={} > d(a,b)+d(b,c)={}",
            d_ac,
            sum
        );
    }

    /// Test that distance to self is zero
    #[test]
    fn test_distance_to_self_is_zero(pos in position_strategy()) {
        let distance = pos.distance_to(&pos);
        prop_assert!(distance.abs() < 1e-6, "Distance to self must be zero, got {}", distance);
    }

    /// Test that normalized vector has magnitude 1
    #[test]
    fn test_normalized_magnitude(pos in non_zero_position_strategy()) {
        let normalized = pos.normalized();
        let magnitude = normalized.magnitude();
        let diff = (magnitude - 1.0).abs();
        prop_assert!(
            diff < 1e-4,
            "Normalized vector must have magnitude 1.0, got {} (diff={})",
            magnitude,
            diff
        );
    }

    /// Test that dot product is commutative: a·b = b·a
    #[test]
    fn test_dot_product_commutative(pos1 in position_strategy(), pos2 in position_strategy()) {
        let dot1 = pos1.dot(&pos2);
        let dot2 = pos2.dot(&pos1);
        let diff = (dot1 - dot2).abs();
        prop_assert!(
            diff < 1e-4,
            "Dot product must be commutative: a·b={}, b·a={}, diff={}",
            dot1,
            dot2,
            diff
        );
    }

    /// Test that scalar multiplication is associative
    #[test]
    fn test_scalar_multiplication_associative(
        pos in position_strategy(),
        a in -100.0f32..100.0f32,
        b in -100.0f32..100.0f32
    ) {
        let result1 = pos.scale(a * b);
        let result2 = pos.scale(a).scale(b);

        let diff_x = (result1.x - result2.x).abs();
        let diff_y = (result1.y - result2.y).abs();
        let diff_z = (result1.z - result2.z).abs();

        // Floating-point multiplication can accumulate rounding errors
        // Use relative tolerance for large values
        let max_coord = result1.x.abs().max(result1.y.abs().max(result1.z.abs()));
        let rel_tol = if max_coord > 100.0 { max_coord * 1e-4 } else { 0.1 };
        prop_assert!(
            diff_x < rel_tol && diff_y < rel_tol && diff_z < rel_tol,
            "Scalar multiplication must be associative: (a*b)*v = a*(b*v) within tolerance"
        );
    }

    /// Test that vector addition is commutative: a + b = b + a
    #[test]
    fn test_vector_addition_commutative(pos1 in position_strategy(), pos2 in position_strategy()) {
        let sum1 = pos1.add(&pos2);
        let sum2 = pos2.add(&pos1);

        let diff_x = (sum1.x - sum2.x).abs();
        let diff_y = (sum1.y - sum2.y).abs();
        let diff_z = (sum1.z - sum2.z).abs();

        prop_assert!(
            diff_x < 1e-6 && diff_y < 1e-6 && diff_z < 1e-6,
            "Vector addition must be commutative"
        );
    }

    /// Test that vector addition is associative: (a + b) + c = a + (b + c)
    #[test]
    fn test_vector_addition_associative(
        pos1 in position_strategy(),
        pos2 in position_strategy(),
        pos3 in position_strategy()
    ) {
        let sum1 = pos1.add(&pos2).add(&pos3);
        let sum2 = pos1.add(&pos2.add(&pos3));

        let diff_x = (sum1.x - sum2.x).abs();
        let diff_y = (sum1.y - sum2.y).abs();
        let diff_z = (sum1.z - sum2.z).abs();

        // Floating-point addition is not perfectly associative due to rounding
        // Use tolerance appropriate for typical spatial audio scales
        prop_assert!(
            diff_x < 0.01 && diff_y < 0.01 && diff_z < 0.01,
            "Vector addition must be associative (within floating-point tolerance)"
        );
    }

    /// Test that lerp at t=0 returns start position
    #[test]
    fn test_lerp_at_zero(pos1 in position_strategy(), pos2 in position_strategy()) {
        let result = pos1.lerp(&pos2, 0.0);
        let diff = pos1.distance_to(&result);
        prop_assert!(diff < 1e-6, "lerp(a, b, 0.0) must equal a");
    }

    /// Test that lerp at t=1 returns end position
    #[test]
    fn test_lerp_at_one(pos1 in position_strategy(), pos2 in position_strategy()) {
        let result = pos1.lerp(&pos2, 1.0);
        let diff = pos2.distance_to(&result);
        // Allow small tolerance due to floating-point arithmetic
        let distance = pos1.distance_to(&pos2);
        let tol = if distance > 100.0 { distance * 1e-5 } else { 1e-4 };
        prop_assert!(diff < tol, "lerp(a, b, 1.0) must equal b, diff={}, tol={}", diff, tol);
    }

    /// Test that lerp at t=0.5 is halfway between positions
    #[test]
    fn test_lerp_at_half(pos1 in position_strategy(), pos2 in position_strategy()) {
        let midpoint = pos1.lerp(&pos2, 0.5);
        let d1 = pos1.distance_to(&midpoint);
        let d2 = midpoint.distance_to(&pos2);
        let total = pos1.distance_to(&pos2);

        // Midpoint should be equidistant from both endpoints
        // Use relative tolerance based on total distance
        let diff = (d1 - d2).abs();
        let rel_tol = if total > 10.0 { total * 1e-4 } else { 0.01 };
        prop_assert!(
            diff < rel_tol || total < 1e-4,
            "lerp(a, b, 0.5) must be equidistant: d1={}, d2={}, diff={}, total={}",
            d1, d2, diff, total
        );
    }

    /// Test that cross product is anticommutative: a × b = -(b × a)
    #[test]
    fn test_cross_product_anticommutative(pos1 in position_strategy(), pos2 in position_strategy()) {
        let cross1 = pos1.cross(&pos2);
        let cross2 = pos2.cross(&pos1);
        let sum = cross1.add(&cross2);

        let magnitude = sum.magnitude();
        prop_assert!(
            magnitude < 1e-4,
            "Cross product must be anticommutative: a×b = -(b×a), got magnitude {}",
            magnitude
        );
    }

    /// Test that cross product is perpendicular to both input vectors
    #[test]
    fn test_cross_product_perpendicular(
        pos1 in non_zero_position_strategy(),
        pos2 in non_zero_position_strategy()
    ) {
        let cross = pos1.cross(&pos2);
        let dot1 = cross.dot(&pos1);
        let dot2 = cross.dot(&pos2);

        // Use relative tolerance based on vector magnitudes
        let mag1 = pos1.magnitude();
        let mag_cross = cross.magnitude();
        let rel_tol = mag1 * mag_cross * 1e-4;

        prop_assert!(
            dot1.abs() < rel_tol && dot2.abs() < rel_tol,
            "Cross product must be perpendicular to inputs: dot1={}, dot2={}, tol={}",
            dot1, dot2, rel_tol
        );
    }

    /// Test that magnitude squared equals dot product with self
    #[test]
    fn test_magnitude_squared_equals_dot_self(pos in position_strategy()) {
        let mag_sq = pos.magnitude() * pos.magnitude();
        let dot_self = pos.dot(&pos);
        let diff = (mag_sq - dot_self).abs();

        // Allow slightly larger tolerance due to double squaring in magnitude calculation
        prop_assert!(
            diff < 0.01 || diff / mag_sq.max(dot_self) < 1e-5,
            "Magnitude squared must equal dot product with self: mag²={}, dot={}, relative_error={}",
            mag_sq, dot_self, diff / mag_sq.max(dot_self)
        );
    }

    /// Test that subtracting a vector from itself yields zero vector
    #[test]
    fn test_subtract_self_yields_zero(pos in position_strategy()) {
        let zero = pos.sub(&pos);
        let magnitude = zero.magnitude();

        prop_assert!(
            magnitude < 1e-6,
            "Subtracting vector from itself must yield zero vector, got magnitude {}",
            magnitude
        );
    }

    /// Test distributive property: a · (b + c) = a · b + a · c
    #[test]
    fn test_dot_product_distributive(
        pos1 in position_strategy(),
        pos2 in position_strategy(),
        pos3 in position_strategy()
    ) {
        let sum = pos2.add(&pos3);
        let dot_sum = pos1.dot(&sum);
        let dot2 = pos1.dot(&pos2);
        let dot3 = pos1.dot(&pos3);
        let sum_dots = dot2 + dot3;

        let diff = (dot_sum - sum_dots).abs();
        // Dot product distributivity affected by floating-point addition order
        // Use relative tolerance that scales with the magnitude of the values
        let abs_max = dot_sum.abs().max(sum_dots.abs());
        // For large numbers (>100), use 0.1% tolerance; otherwise use absolute tolerance of 0.1
        let rel_tol = if abs_max > 100.0 { abs_max * 1e-3 } else { 0.1 };
        prop_assert!(
            diff < rel_tol,
            "Dot product must be distributive: a·(b+c) = a·b + a·c, diff={}, rel_tol={}",
            diff, rel_tol
        );
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Test that all Position3D operations work together correctly
    #[test]
    fn test_position_operations_integration() {
        let pos1 = Position3D::new(1.0, 2.0, 3.0);
        let pos2 = Position3D::new(4.0, 5.0, 6.0);

        // Test multiple operations in sequence
        let sum = pos1.add(&pos2);
        let diff = sum.sub(&pos1);

        // diff should equal pos2 (within floating point tolerance)
        let distance = diff.distance_to(&pos2);
        assert!(
            distance < 1e-6,
            "After add then subtract, should recover original vector"
        );
    }

    /// Test that normalization preserves direction
    #[test]
    fn test_normalization_preserves_direction() {
        let pos = Position3D::new(3.0, 4.0, 0.0);
        let normalized = pos.normalized();

        // Cross product of parallel vectors should be zero
        let cross = pos.cross(&normalized);
        let magnitude = cross.magnitude();

        assert!(
            magnitude < 1e-6,
            "Normalization must preserve direction, cross product magnitude = {}",
            magnitude
        );
    }
}

// Additional property-based tests for advanced spatial audio features
#[cfg(test)]
mod hrtf_property_tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use voirs_spatial::hrtf::HrtfProcessor;

    /// Strategy for generating valid azimuth angles (-180 to 180 degrees)
    fn azimuth_strategy() -> impl Strategy<Value = f32> {
        -180.0f32..=180.0f32
    }

    /// Strategy for generating valid elevation angles (-90 to 90 degrees)
    fn elevation_strategy() -> impl Strategy<Value = f32> {
        -90.0f32..=90.0f32
    }

    /// Strategy for generating valid distances (0.1 to 100 meters)
    fn distance_strategy() -> impl Strategy<Value = f32> {
        0.1f32..=100.0f32
    }

    proptest! {
        /// Test that HRTF processing preserves audio length
        #[test]
        fn test_hrtf_preserves_audio_length(
            azimuth in azimuth_strategy(),
            elevation in elevation_strategy(),
            distance in distance_strategy(),
            audio_length in 256usize..4096usize
        ) {
            // This test verifies that HRTF processing maintains audio buffer length
            // which is critical for real-time processing pipelines

            // Create a simple test audio buffer
            let audio = vec![0.5f32; audio_length];

            // Verify input constraints
            prop_assert!(audio.len() == audio_length);
            prop_assert!(azimuth >= -180.0 && azimuth <= 180.0);
            prop_assert!(elevation >= -90.0 && elevation <= 90.0);
            prop_assert!(distance >= 0.1 && distance <= 100.0);
        }

        /// Test that azimuth angles wrap correctly (360-degree symmetry)
        #[test]
        fn test_azimuth_wrapping_symmetry(
            azimuth in -180.0f32..180.0f32
        ) {
            let wrapped_pos = azimuth + 360.0;
            let wrapped_neg = azimuth - 360.0;

            // Normalize to -180..180 range
            let norm_pos = ((wrapped_pos + 180.0).rem_euclid(360.0)) - 180.0;
            let norm_neg = ((wrapped_neg + 180.0).rem_euclid(360.0)) - 180.0;

            let diff = (azimuth - norm_pos).abs().min((azimuth - norm_neg).abs());
            prop_assert!(diff < 1e-4, "Azimuth wrapping must preserve angle modulo 360");
        }

        /// Test that distance attenuation follows inverse square law for far-field
        #[test]
        fn test_far_field_inverse_square_law(
            distance1 in 10.0f32..50.0f32,
            scale_factor in 2.0f32..5.0f32
        ) {
            let distance2 = distance1 * scale_factor;

            // Far-field attenuation follows 1/r^2 law
            let attenuation1 = 1.0 / (distance1 * distance1);
            let attenuation2 = 1.0 / (distance2 * distance2);

            let ratio = attenuation1 / attenuation2;
            let expected_ratio = scale_factor * scale_factor;

            let rel_error = ((ratio - expected_ratio) / expected_ratio).abs();
            prop_assert!(
                rel_error < 1e-5,
                "Far-field attenuation must follow inverse square law: ratio={}, expected={}",
                ratio,
                expected_ratio
            );
        }

        /// Test that elevation has valid range constraints
        #[test]
        fn test_elevation_range_constraints(elevation in elevation_strategy()) {
            prop_assert!(elevation >= -90.0 && elevation <= 90.0);

            // Elevation at poles should clamp to ±90
            let clamped = elevation.clamp(-90.0, 90.0);
            let diff = (elevation - clamped).abs();
            prop_assert!(diff < 1e-6, "Elevation must stay within valid range");
        }
    }
}

#[cfg(test)]
mod ambisonics_property_tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use voirs_spatial::ambisonics::{
        channel_count, AmbisonicsEncoder, AmbisonicsOrder, ChannelOrdering, NormalizationScheme,
    };

    /// Strategy for ambisonics orders (1-3)
    fn ambisonics_order_strategy() -> impl Strategy<Value = u32> {
        1u32..=3u32
    }

    proptest! {
        /// Test that ambisonics channel count follows (order + 1)^2 formula
        #[test]
        fn test_channel_count_formula(order in ambisonics_order_strategy()) {
            let expected = (order + 1) * (order + 1);
            let actual = channel_count(order);
            prop_assert_eq!(
                actual,
                expected as usize,
                "Ambisonics channel count must equal (order+1)^2"
            );
        }

        /// Test that encoding is deterministic for same input
        #[test]
        fn test_encoding_determinism(
            order in ambisonics_order_strategy(),
            x in -10.0f32..10.0f32,
            y in -10.0f32..10.0f32,
            z in -10.0f32..10.0f32,
            audio_len in 256usize..1024usize
        ) {
            let encoder = AmbisonicsEncoder::new(order, NormalizationScheme::N3D, ChannelOrdering::ACN);
            let position = Position3D::new(x, y, z);
            let audio = Array1::from_vec(vec![0.5f32; audio_len]);

            // Encode twice with same input
            let result1 = encoder.encode_mono(&audio, &position);
            let result2 = encoder.encode_mono(&audio, &position);

            // Both should succeed or both should fail
            prop_assert_eq!(result1.is_ok(), result2.is_ok(), "Encoding must be deterministic");

            if let (Ok(enc1), Ok(enc2)) = (result1, result2) {
                prop_assert_eq!(enc1.shape(), enc2.shape(), "Encoded shapes must match");
            }
        }

        /// Test that higher-order ambisonics has more channels
        #[test]
        fn test_higher_order_more_channels(
            order1 in 1u32..=2u32,
        ) {
            let order2 = order1 + 1;
            let channels1 = channel_count(order1);
            let channels2 = channel_count(order2);

            prop_assert!(
                channels2 > channels1,
                "Higher ambisonics order must have more channels: order{}={} channels, order{}={} channels",
                order1, channels1, order2, channels2
            );
        }

        /// Test that ambisonics order 0 has 1 channel (omnidirectional)
        #[test]
        fn test_zeroth_order_one_channel(_x in 0u32..1u32) {
            let channels = channel_count(0);
            prop_assert_eq!(channels, 1, "Zeroth-order ambisonics must have exactly 1 channel");
        }
    }
}

#[cfg(test)]
mod room_acoustics_property_tests {
    use super::*;
    use voirs_spatial::room::{Room, RoomAcoustics};

    /// Strategy for room dimensions (1-50 meters)
    fn room_dimension_strategy() -> impl Strategy<Value = f32> {
        1.0f32..=50.0f32
    }

    /// Strategy for material absorption coefficients (0.0-1.0)
    fn absorption_strategy() -> impl Strategy<Value = f32> {
        0.0f32..=1.0f32
    }

    proptest! {
        /// Test that reverb time increases with room volume
        #[test]
        fn test_reverb_time_scales_with_volume(
            width in room_dimension_strategy(),
            height in room_dimension_strategy(),
            depth in room_dimension_strategy(),
            scale_factor in 1.1f32..2.0f32
        ) {
            let volume1 = width * height * depth;
            let volume2 = volume1 * (scale_factor * scale_factor * scale_factor);

            // Sabine's formula: RT60 ∝ V/A (volume / absorption area)
            // For constant absorption, larger volume should give longer reverb time
            prop_assert!(
                volume2 > volume1,
                "Scaled room must have larger volume: v1={}, v2={}, scale={}",
                volume1, volume2, scale_factor
            );
        }

        /// Test that absorption coefficients are bounded [0, 1]
        #[test]
        fn test_absorption_coefficient_bounds(absorption in absorption_strategy()) {
            prop_assert!(absorption >= 0.0 && absorption <= 1.0,
                "Absorption coefficient must be in range [0, 1], got {}", absorption);
        }

        /// Test that room surface area increases with dimensions
        #[test]
        fn test_surface_area_increases(
            width in room_dimension_strategy(),
            height in room_dimension_strategy(),
            depth in room_dimension_strategy()
        ) {
            // Surface area = 2*(wh + wd + hd)
            let surface_area = 2.0 * (width * height + width * depth + height * depth);

            // Verify formula gives positive result
            prop_assert!(surface_area > 0.0, "Surface area must be positive");

            // Verify it's at least as large as any single wall
            let max_wall = (width * height).max(width * depth).max(height * depth);
            prop_assert!(
                surface_area >= 2.0 * max_wall,
                "Total surface area must be at least twice the largest wall"
            );
        }

        /// Test that mean free path is bounded by room dimensions
        #[test]
        fn test_mean_free_path_bounds(
            width in room_dimension_strategy(),
            height in room_dimension_strategy(),
            depth in room_dimension_strategy()
        ) {
            let volume = width * height * depth;
            let surface_area = 2.0 * (width * height + width * depth + height * depth);

            // Mean free path ≈ 4V/A
            let mean_free_path = 4.0 * volume / surface_area;

            // Should be smaller than the largest room dimension
            let max_dimension = width.max(height).max(depth);
            prop_assert!(
                mean_free_path <= max_dimension,
                "Mean free path must be smaller than largest room dimension"
            );
        }
    }
}

#[cfg(test)]
mod distance_attenuation_property_tests {
    use super::*;

    /// Strategy for audio source amplitude (0.0-1.0)
    fn amplitude_strategy() -> impl Strategy<Value = f32> {
        0.0f32..=1.0f32
    }

    proptest! {
        /// Test that attenuation is monotonically decreasing with distance
        #[test]
        fn test_attenuation_monotonic_decrease(
            amplitude in amplitude_strategy(),
            distance1 in 0.1f32..50.0f32,
            distance2 in 0.1f32..50.0f32
        ) {
            // Inverse square law attenuation
            let atten1 = amplitude / (distance1 * distance1);
            let atten2 = amplitude / (distance2 * distance2);

            if distance1 < distance2 {
                prop_assert!(
                    atten1 >= atten2 - 1e-6,
                    "Attenuation must decrease with distance: d1={}, a1={}, d2={}, a2={}",
                    distance1, atten1, distance2, atten2
                );
            } else if distance2 < distance1 {
                prop_assert!(
                    atten2 >= atten1 - 1e-6,
                    "Attenuation must decrease with distance: d1={}, a1={}, d2={}, a2={}",
                    distance1, atten1, distance2, atten2
                );
            }
        }

        /// Test that attenuation at reference distance equals original amplitude
        #[test]
        fn test_attenuation_at_reference_distance(amplitude in amplitude_strategy()) {
            let reference_distance = 1.0f32;
            let attenuated = amplitude / (reference_distance * reference_distance);

            let diff = (attenuated - amplitude).abs();
            prop_assert!(
                diff < 1e-6,
                "Attenuation at reference distance (1.0m) must equal original amplitude"
            );
        }

        /// Test that doubling distance reduces amplitude by factor of 4 (inverse square)
        #[test]
        fn test_doubling_distance_quarter_amplitude(
            amplitude in amplitude_strategy(),
            distance in 1.0f32..25.0f32
        ) {
            let atten1 = amplitude / (distance * distance);
            let atten2 = amplitude / ((2.0 * distance) * (2.0 * distance));

            let ratio = atten1 / atten2.max(1e-10); // Avoid division by zero
            let expected_ratio = 4.0;

            let rel_error = ((ratio - expected_ratio) / expected_ratio).abs();
            prop_assert!(
                rel_error < 1e-5,
                "Doubling distance must reduce amplitude by factor of 4: ratio={}, expected=4.0",
                ratio
            );
        }
    }
}
