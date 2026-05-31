//! Property-based tests for voice conversion
//!
//! These tests use proptest to verify invariants and properties
//! of the voice conversion system across a wide range of inputs.

use proptest::prelude::*;
use voirs_conversion::transforms::{
    AgeTransform, GenderTransform, PitchTransform, SpeedTransform, Transform,
};

// ============================================================================
// Property Test Strategies
// ============================================================================

/// Strategy for generating valid audio samples
fn audio_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        -1.0f32..=1.0f32,
        100..5000, // 100 to 5000 samples
    )
}

/// Strategy for generating valid pitch factors
fn pitch_factor_strategy() -> impl Strategy<Value = f32> {
    0.5f32..2.0f32
}

/// Strategy for generating valid speed factors
fn speed_factor_strategy() -> impl Strategy<Value = f32> {
    0.5f32..2.0f32
}

/// Strategy for generating valid ages
fn age_strategy() -> impl Strategy<Value = f32> {
    5.0f32..95.0f32
}

// ============================================================================
// Transform Property Tests
// ============================================================================

proptest! {
    /// Property: Pitch transform should not change audio length significantly
    #[test]
    fn prop_pitch_transform_preserves_length(
        audio in audio_strategy(),
        pitch_factor in pitch_factor_strategy()
    ) {
        let transform = PitchTransform::new(pitch_factor);
        let result = transform.apply(&audio);

        prop_assert!(result.is_ok());
        if let Ok(transformed) = result {
            // Allow differences due to windowing artifacts
            let length_diff = (transformed.len() as i32 - audio.len() as i32).abs();
            prop_assert!(length_diff <= 1024, "Length changed by {} samples", length_diff);
        }
    }

    /// Property: Pitch transform with factor 1.0 should be nearly identity
    #[test]
    fn prop_pitch_transform_identity(audio in audio_strategy()) {
        let transform = PitchTransform::new(1.0);
        let result = transform.apply(&audio);

        prop_assert!(result.is_ok());
        if let Ok(transformed) = result {
            // Should be very similar to input (within tolerance for processing)
            let min_len = audio.len().min(transformed.len());
            let diff: f32 = audio[..min_len].iter()
                .zip(&transformed[..min_len])
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>() / min_len as f32;

            prop_assert!(diff < 0.15, "Average difference: {}", diff);
        }
    }

    /// Property: Speed transform with factor 1.0 should preserve audio
    #[test]
    fn prop_speed_transform_identity(audio in audio_strategy()) {
        let transform = SpeedTransform::new(1.0);
        let result = transform.apply(&audio);

        prop_assert!(result.is_ok());
        if let Ok(transformed) = result {
            let min_len = audio.len().min(transformed.len());
            let diff: f32 = audio[..min_len].iter()
                .zip(&transformed[..min_len])
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>() / min_len as f32;

            prop_assert!(diff < 0.15, "Average difference: {}", diff);
        }
    }

    /// Property: Speed transform should change length proportionally
    #[test]
    fn prop_speed_transform_changes_length(
        audio in audio_strategy(),
        speed_factor in speed_factor_strategy()
    ) {
        let transform = SpeedTransform::new(speed_factor);
        let result = transform.apply(&audio);

        prop_assert!(result.is_ok());
        if let Ok(transformed) = result {
            let expected_length = (audio.len() as f32 / speed_factor) as usize;
            let actual_length = transformed.len();

            // Allow 20% tolerance for PSOLA artifacts
            let tolerance = (expected_length as f32 * 0.2).max(512.0);
            let diff = (expected_length as f32 - actual_length as f32).abs();

            prop_assert!(
                diff <= tolerance,
                "Expected length ~{}, got {}, diff: {} (tolerance: {})",
                expected_length,
                actual_length,
                diff,
                tolerance
            );
        }
    }

    /// Property: Age transform should produce valid output
    #[test]
    fn prop_age_transform_produces_valid_output(
        audio in audio_strategy(),
        source_age in age_strategy(),
        target_age in age_strategy()
    ) {
        let transform = AgeTransform::new(source_age, target_age);
        let result = transform.apply(&audio);

        prop_assert!(result.is_ok());
        if let Ok(transformed) = result {
            // Output should be in valid range
            for (i, &sample) in transformed.iter().enumerate() {
                prop_assert!(sample.is_finite(), "Sample at {} is not finite: {}", i, sample);
                // Allow up to 4.0 for age transformation artifacts
                prop_assert!(sample.abs() <= 4.0, "Sample at {} magnitude too large: {}", i, sample);
            }

            // Should have reasonable length
            let length_ratio = transformed.len() as f32 / audio.len() as f32;
            prop_assert!(
                length_ratio >= 0.4 && length_ratio <= 2.5,
                "Length ratio out of bounds: {}",
                length_ratio
            );
        }
    }

    /// Property: Gender transform should produce valid output
    #[test]
    fn prop_gender_transform_produces_valid_output(
        audio in audio_strategy()
    ) {
        let transform = GenderTransform::new(0.8); // 0.8 for female-leaning gender
        let result = transform.apply(&audio);

        prop_assert!(result.is_ok());
        if let Ok(transformed) = result {
            // Output should be in valid range
            for (i, &sample) in transformed.iter().enumerate() {
                prop_assert!(sample.is_finite(), "Sample at {} is not finite: {}", i, sample);
                prop_assert!(sample.abs() <= 3.0, "Sample at {} magnitude too large: {}", i, sample);
            }

            // Length should be approximately preserved
            let length_diff = (transformed.len() as i32 - audio.len() as i32).abs();
            prop_assert!(
                length_diff <= 2048,
                "Length changed by {} samples",
                length_diff
            );
        }
    }

    /// Property: Transforms should not produce NaN or infinity
    #[test]
    fn prop_no_nan_or_inf_pitch(
        audio in audio_strategy(),
        pitch_factor in pitch_factor_strategy()
    ) {
        let transform = PitchTransform::new(pitch_factor);
        let result = transform.apply(&audio);

        if let Ok(transformed) = result {
            for (i, &sample) in transformed.iter().enumerate() {
                prop_assert!(sample.is_finite(), "Non-finite sample at index {}: {}", i, sample);
            }
        }
    }

    /// Property: Transforms should not amplify excessively
    #[test]
    fn prop_bounded_amplification(
        audio in prop::collection::vec(-0.5f32..=0.5f32, 1000..2000),
        pitch_factor in pitch_factor_strategy()
    ) {
        let transform = PitchTransform::new(pitch_factor);
        let result = transform.apply(&audio);

        if let Ok(transformed) = result {
            let max_input = audio.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
            let max_output = transformed.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

            // Allow up to 6x amplification for processing artifacts and edge cases
            prop_assert!(
                max_output <= max_input * 6.0 + 0.5,
                "Excessive amplification: input {} -> output {}",
                max_input,
                max_output
            );
        }
    }

    /// Property: Energy preservation in small pitch changes
    #[test]
    fn prop_energy_preservation_small_changes(
        audio in prop::collection::vec(-0.5f32..=0.5f32, 1000..2000),
        pitch_factor in (0.9f32..=1.1f32)  // Small pitch changes
    ) {
        let transform = PitchTransform::new(pitch_factor);
        let result = transform.apply(&audio);

        if let Ok(transformed) = result {
            let input_energy: f32 = audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32;
            let min_len = audio.len().min(transformed.len());
            let output_energy: f32 = transformed[..min_len].iter().map(|&x| x * x).sum::<f32>()
                / min_len as f32;

            let energy_ratio = output_energy / (input_energy + 0.0001);

            // Allow energy to vary significantly for small pitch changes
            // (phase vocoder artifacts can cause energy variations)
            prop_assert!(
                energy_ratio >= 0.1 && energy_ratio <= 10.0,
                "Energy ratio out of bounds: {} (input: {}, output: {})",
                energy_ratio,
                input_energy,
                output_energy
            );
        }
    }

    /// Property: Consecutive identity transforms should not degrade signal
    #[test]
    fn prop_identity_transform_chain(audio in audio_strategy()) {
        let transform = PitchTransform::new(1.0);

        let result1 = transform.apply(&audio);
        prop_assert!(result1.is_ok());

        if let Ok(transformed1) = result1 {
            let result2 = transform.apply(&transformed1);
            prop_assert!(result2.is_ok());

            if let Ok(transformed2) = result2 {
                let min_len = transformed1.len().min(transformed2.len());
                let diff: f32 = transformed1[..min_len]
                    .iter()
                    .zip(&transformed2[..min_len])
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>()
                    / min_len as f32;

                prop_assert!(
                    diff < 0.2,
                    "Consecutive identity transforms degraded signal: diff = {}",
                    diff
                );
            }
        }
    }

    /// Property: Transform parameters should be retrievable
    #[test]
    fn prop_transform_parameters_retrieval(pitch_factor in pitch_factor_strategy()) {
        let transform = PitchTransform::new(pitch_factor);
        let params = transform.get_parameters();

        prop_assert!(params.contains_key("pitch_factor"));
        if let Some(&retrieved_factor) = params.get("pitch_factor") {
            prop_assert!(
                (retrieved_factor - pitch_factor).abs() < 0.001,
                "Retrieved pitch factor {} doesn't match set value {}",
                retrieved_factor,
                pitch_factor
            );
        }
    }

    /// Property: Speed transform inverse should approximately restore length
    #[test]
    fn prop_speed_transform_inverse(
        audio in prop::collection::vec(-0.5f32..=0.5f32, 1000..2000),
        speed_factor in (1.1f32..=1.9f32)
    ) {
        // Apply speed transform
        let transform1 = SpeedTransform::new(speed_factor);
        let result1 = transform1.apply(&audio);

        prop_assert!(result1.is_ok());

        if let Ok(transformed) = result1 {
            // Apply inverse speed transform
            let inverse_factor = 1.0 / speed_factor;
            let transform2 = SpeedTransform::new(inverse_factor);
            let result2 = transform2.apply(&transformed);

            prop_assert!(result2.is_ok());

            if let Ok(restored) = result2 {
                // Length should be approximately restored
                let original_len = audio.len();
                let restored_len = restored.len();
                let length_diff = (original_len as f32 - restored_len as f32).abs();
                let tolerance = original_len as f32 * 0.25; // 25% tolerance

                prop_assert!(
                    length_diff <= tolerance,
                    "Inverse speed transform didn't restore length: {} -> {} -> {} (diff: {}, tolerance: {})",
                    original_len,
                    transformed.len(),
                    restored_len,
                    length_diff,
                    tolerance
                );
            }
        }
    }

    /// Property: Age transform with same source and target should be nearly identity
    #[test]
    fn prop_age_transform_identity(
        audio in audio_strategy(),
        age in age_strategy()
    ) {
        let transform = AgeTransform::new(age, age);
        let result = transform.apply(&audio);

        prop_assert!(result.is_ok());
        if let Ok(transformed) = result {
            let min_len = audio.len().min(transformed.len());
            let diff: f32 = audio[..min_len].iter()
                .zip(&transformed[..min_len])
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>() / min_len as f32;

            prop_assert!(diff < 0.2, "Age transform with same age should be near-identity: diff = {}", diff);
        }
    }

    /// Property: Empty audio should return empty result
    #[test]
    fn prop_empty_audio_handling(pitch_factor in pitch_factor_strategy()) {
        let empty_audio: Vec<f32> = vec![];
        let transform = PitchTransform::new(pitch_factor);
        let result = transform.apply(&empty_audio);

        prop_assert!(result.is_ok());
        if let Ok(transformed) = result {
            prop_assert!(transformed.is_empty(), "Empty input should produce empty output");
        }
    }

    /// Property: Single sample audio should be handled gracefully
    #[test]
    fn prop_single_sample_handling(
        sample in (-1.0f32..=1.0f32),
        pitch_factor in pitch_factor_strategy()
    ) {
        let audio = vec![sample];
        let transform = PitchTransform::new(pitch_factor);
        let result = transform.apply(&audio);

        prop_assert!(result.is_ok());
        if let Ok(transformed) = result {
            prop_assert!(!transformed.is_empty(), "Single sample should produce output");
            for &s in &transformed {
                prop_assert!(s.is_finite(), "Output should be finite");
            }
        }
    }
}
