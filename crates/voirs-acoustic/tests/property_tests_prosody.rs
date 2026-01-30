//! Property-based tests for prosody SIMD operations
//!
//! These tests verify numerical properties and correctness of SIMD-optimized
//! prosody processing functions across a wide range of inputs.

use proptest::prelude::*;
use voirs_acoustic::prosody::simd_ops::*;

/// Strategy for generating F0 contours with voiced/unvoiced frames
fn f0_contour_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        prop_oneof![
            // Voiced frames (50-500 Hz)
            50.0f32..500.0f32,
            // Unvoiced frames (0.0)
            Just(0.0f32),
        ],
        10..200,
    )
}

/// Strategy for generating positive signal values
fn positive_signal_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(0.1f32..10.0f32, 10..200)
}

/// Strategy for generating energy envelopes
fn energy_envelope_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(0.0f32..2.0f32, 20..200)
}

proptest! {
    /// Property: F0 smoothing preserves length
    #[test]
    fn prop_f0_smooth_preserves_length(
        mut contour in f0_contour_strategy(),
        alpha in 0.0f32..1.0f32
    ) {
        let original_len = contour.len();
        smooth_f0_contour_simd(&mut contour, alpha);
        prop_assert_eq!(contour.len(), original_len);
    }

    /// Property: F0 smoothing preserves unvoiced frames (0.0)
    #[test]
    fn prop_f0_smooth_preserves_unvoiced(
        mut contour in f0_contour_strategy(),
        alpha in 0.1f32..0.9f32
    ) {
        // Record positions of unvoiced frames
        let unvoiced_positions: Vec<usize> = contour.iter()
            .enumerate()
            .filter(|(_, &f0)| f0 == 0.0)
            .map(|(i, _)| i)
            .collect();

        smooth_f0_contour_simd(&mut contour, alpha);

        // Unvoiced frames should still be 0.0
        for pos in unvoiced_positions {
            prop_assert_eq!(contour[pos], 0.0,
                "Unvoiced frame at position {} was modified", pos);
        }
    }

    /// Property: F0 smoothing reduces variation (for voiced frames)
    #[test]
    fn prop_f0_smooth_reduces_variation(
        contour in prop::collection::vec(100.0f32..200.0f32, 20..50),
        alpha in 0.5f32..0.9f32
    ) {
        let mut smoothed = contour.clone();
        smooth_f0_contour_simd(&mut smoothed, alpha);

        // Compute variance of differences
        let original_variance = compute_variance_of_diffs(&contour);
        let smoothed_variance = compute_variance_of_diffs(&smoothed);

        // Smoothing should reduce variance (with high alpha)
        if alpha > 0.6 {
            prop_assert!(smoothed_variance <= original_variance * 1.1,
                "Smoothing didn't reduce variation: {} -> {}",
                original_variance, smoothed_variance);
        }
    }

    /// Property: Pitch shifting by 0 semitones doesn't change contour
    #[test]
    fn prop_pitch_shift_identity(mut contour in f0_contour_strategy()) {
        let original = contour.clone();
        shift_pitch_simd(&mut contour, 0.0);

        for (i, (&orig, &shifted)) in original.iter().zip(contour.iter()).enumerate() {
            prop_assert!((orig - shifted).abs() < 1e-5,
                "Position {}: pitch shift by 0 changed value {} to {}", i, orig, shifted);
        }
    }

    /// Property: Pitch shifting is commutative (shift up then down = identity)
    #[test]
    fn prop_pitch_shift_commutative(
        mut contour in f0_contour_strategy(),
        semitones in -12.0f32..12.0f32
    ) {
        let original = contour.clone();

        // Shift up
        shift_pitch_simd(&mut contour, semitones);
        // Shift back down
        shift_pitch_simd(&mut contour, -semitones);

        for (i, (&orig, &final_val)) in original.iter().zip(contour.iter()).enumerate() {
            if orig > 0.0 {
                let relative_error = ((orig - final_val).abs() / orig).abs();
                prop_assert!(relative_error < 0.01,
                    "Position {}: round-trip pitch shift failed: {} -> {}",
                    i, orig, final_val);
            }
        }
    }

    /// Property: Pitch shift by +12 semitones doubles frequency
    #[test]
    fn prop_pitch_shift_octave_up(
        contour in prop::collection::vec(100.0f32..200.0f32, 10..50)
    ) {
        let mut shifted = contour.clone();
        shift_pitch_simd(&mut shifted, 12.0);

        for (&orig, &shifted_val) in contour.iter().zip(shifted.iter()) {
            let expected = orig * 2.0; // One octave up
            let relative_error = ((shifted_val - expected).abs() / expected).abs();
            prop_assert!(relative_error < 0.01,
                "Octave up failed: {} -> {} (expected {})",
                orig, shifted_val, expected);
        }
    }

    /// Property: Energy smoothing preserves length
    #[test]
    fn prop_energy_smooth_preserves_length(
        mut energy in energy_envelope_strategy(),
        window_size in prop::sample::select(vec![3, 5, 7, 9, 11])
    ) {
        let original_len = energy.len();
        smooth_energy_envelope_simd(&mut energy, window_size);
        prop_assert_eq!(energy.len(), original_len);
    }

    /// Property: Energy smoothing reduces extremes
    #[test]
    fn prop_energy_smooth_reduces_extremes(
        energy in energy_envelope_strategy(),
        window_size in prop::sample::select(vec![5, 7, 9])
    ) {
        let mut smoothed = energy.clone();
        smooth_energy_envelope_simd(&mut smoothed, window_size);

        let original_max = energy.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let smoothed_max = smoothed.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Smoothing should not increase maximum
        prop_assert!(smoothed_max <= original_max * 1.01,
            "Smoothing increased maximum: {} -> {}", original_max, smoothed_max);
    }

    /// Property: Duration scaling by 1.0 doesn't change signal
    #[test]
    fn prop_duration_scale_identity(signal in positive_signal_strategy()) {
        let scaled = scale_duration_simd(&signal, 1.0);

        prop_assert_eq!(scaled.len(), signal.len(),
            "Duration scaling by 1.0 changed length");

        for (i, (&orig, &scaled_val)) in signal.iter().zip(scaled.iter()).enumerate() {
            prop_assert!((orig - scaled_val).abs() < 1e-4,
                "Position {}: scaling by 1.0 changed value {} to {}",
                i, orig, scaled_val);
        }
    }

    /// Property: Duration scaling changes length proportionally
    #[test]
    fn prop_duration_scale_length(
        signal in positive_signal_strategy(),
        factor in 0.5f32..2.0f32
    ) {
        let scaled = scale_duration_simd(&signal, factor);
        let expected_length = (signal.len() as f32 / factor).round() as usize;

        prop_assert!(scaled.len() >= expected_length.saturating_sub(1) &&
                    scaled.len() <= expected_length + 1,
            "Scaled length {} not close to expected {}",
            scaled.len(), expected_length);
    }

    /// Property: RMS energy is always non-negative
    #[test]
    fn prop_rms_energy_non_negative(signal in positive_signal_strategy()) {
        let rms = compute_rms_energy_simd(&signal);
        prop_assert!(rms >= 0.0, "RMS energy {} is negative", rms);
    }

    /// Property: RMS of zero signal is zero
    #[test]
    fn prop_rms_energy_zero_signal(len in 10usize..100) {
        let signal = vec![0.0; len];
        let rms = compute_rms_energy_simd(&signal);
        prop_assert_eq!(rms, 0.0);
    }

    /// Property: RMS energy is monotonic with amplitude scaling
    #[test]
    fn prop_rms_energy_monotonic(
        signal in positive_signal_strategy(),
        scale in 1.0f32..2.0f32
    ) {
        let rms_original = compute_rms_energy_simd(&signal);

        let scaled: Vec<f32> = signal.iter().map(|&x| x * scale).collect();
        let rms_scaled = compute_rms_energy_simd(&scaled);

        let expected_rms = rms_original * scale;
        let relative_error = ((rms_scaled - expected_rms).abs() / expected_rms).abs();

        prop_assert!(relative_error < 0.01,
            "RMS scaling incorrect: {} * {} = {} (expected {})",
            rms_original, scale, rms_scaled, expected_rms);
    }

    /// Property: Linear interpolation endpoints match keyframes
    #[test]
    fn prop_interpolate_endpoints(
        keyframes in prop::collection::vec(50.0f32..300.0f32, 2..10)
    ) {
        let n = keyframes.len();
        let output_length = (n - 1) * 10 + 1; // 10 samples between keyframes
        let keyframe_times: Vec<usize> = (0..n).map(|i| i * 10).collect();

        let interpolated = interpolate_f0_linear_simd(&keyframes, &keyframe_times, output_length);

        // First and last values should match keyframes
        prop_assert!((interpolated[0] - keyframes[0]).abs() < 1e-4,
            "First interpolated value doesn't match keyframe");
        prop_assert!((interpolated[output_length - 1] - keyframes[n - 1]).abs() < 1e-4,
            "Last interpolated value doesn't match keyframe");
    }

    /// Property: Interpolation is monotonic between monotonic keyframes
    #[test]
    fn prop_interpolate_monotonic(
        keyframes in prop::collection::vec(50.0f32..300.0f32, 3..8)
    ) {
        // Sort keyframes to make them monotonic increasing
        let mut sorted_keyframes = keyframes.clone();
        sorted_keyframes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_keyframes.len();
        let output_length = (n - 1) * 5 + 1;
        let keyframe_times: Vec<usize> = (0..n).map(|i| i * 5).collect();

        let interpolated = interpolate_f0_linear_simd(&sorted_keyframes, &keyframe_times, output_length);

        // Check monotonicity
        for i in 1..interpolated.len() {
            prop_assert!(interpolated[i] >= interpolated[i-1] - 1e-4,
                "Interpolation not monotonic at position {}: {} > {}",
                i, interpolated[i-1], interpolated[i]);
        }
    }
}

/// Helper function to compute variance of differences
fn compute_variance_of_diffs(signal: &[f32]) -> f32 {
    if signal.len() < 2 {
        return 0.0;
    }

    let diffs: Vec<f32> = signal.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let variance = diffs.iter().map(|&d| (d - mean).powi(2)).sum::<f32>() / diffs.len() as f32;

    variance
}
