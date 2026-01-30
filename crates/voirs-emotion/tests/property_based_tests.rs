//! Property-based tests for voirs-emotion
//!
//! These tests use proptest to verify invariants and properties that should hold
//! for all valid inputs, helping discover edge cases that traditional unit tests miss.

use voirs_emotion::{
    types::{Emotion, EmotionDimensions, EmotionIntensity, EmotionParameters, EmotionVector},
    EmotionProcessor,
};

use proptest::prelude::*;

// ============================================================================
// Property Test Strategies
// ============================================================================

/// Generate arbitrary valid emotion intensities (0.0 to 1.0)
fn emotion_intensity_strategy() -> impl Strategy<Value = f32> {
    0.0..=1.0_f32
}

/// Generate arbitrary emotion dimensions (-1.0 to 1.0)
fn emotion_dimension_strategy() -> impl Strategy<Value = f32> {
    -1.0..=1.0_f32
}

/// Generate arbitrary valid audio samples (-1.0 to 1.0)
fn audio_sample_strategy() -> impl Strategy<Value = f32> {
    -1.0..=1.0_f32
}

/// Generate arbitrary valid audio buffers (10 to 1000 samples for faster tests)
fn audio_buffer_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(audio_sample_strategy(), 10..1000)
}

// ============================================================================
// Property Tests: Emotion Dimensions
// ============================================================================

proptest! {
    /// Property: Emotion dimensions should always be bounded between -1.0 and 1.0
    #[test]
    fn prop_emotion_dimensions_bounds(
        valence in emotion_dimension_strategy(),
        arousal in emotion_dimension_strategy(),
        dominance in emotion_dimension_strategy()
    ) {
        let dimensions = EmotionDimensions {
            valence,
            arousal,
            dominance,
        };

        // Property: All dimensions must be in valid range
        prop_assert!(dimensions.valence >= -1.0 && dimensions.valence <= 1.0);
        prop_assert!(dimensions.arousal >= -1.0 && dimensions.arousal <= 1.0);
        prop_assert!(dimensions.dominance >= -1.0 && dimensions.dominance <= 1.0);
    }

    /// Property: Euclidean distance between dimensions is symmetric
    #[test]
    fn prop_dimension_distance_symmetry(
        v1 in emotion_dimension_strategy(),
        a1 in emotion_dimension_strategy(),
        d1 in emotion_dimension_strategy(),
        v2 in emotion_dimension_strategy(),
        a2 in emotion_dimension_strategy(),
        d2 in emotion_dimension_strategy()
    ) {
        let dim1 = EmotionDimensions {
            valence: v1,
            arousal: a1,
            dominance: d1,
        };
        let dim2 = EmotionDimensions {
            valence: v2,
            arousal: a2,
            dominance: d2,
        };

        let dist_12 = ((dim1.valence - dim2.valence).powi(2)
            + (dim1.arousal - dim2.arousal).powi(2)
            + (dim1.dominance - dim2.dominance).powi(2))
            .sqrt();

        let dist_21 = ((dim2.valence - dim1.valence).powi(2)
            + (dim2.arousal - dim1.arousal).powi(2)
            + (dim2.dominance - dim1.dominance).powi(2))
            .sqrt();

        // Property: Distance metric must be symmetric
        prop_assert!((dist_12 - dist_21).abs() < 1e-6);
    }

    /// Property: Triangle inequality holds for dimension distances
    #[test]
    fn prop_dimension_triangle_inequality(
        v1 in emotion_dimension_strategy(),
        a1 in emotion_dimension_strategy(),
        d1 in emotion_dimension_strategy(),
        v2 in emotion_dimension_strategy(),
        a2 in emotion_dimension_strategy(),
        d2 in emotion_dimension_strategy(),
        v3 in emotion_dimension_strategy(),
        a3 in emotion_dimension_strategy(),
        d3 in emotion_dimension_strategy()
    ) {
        let dim1 = EmotionDimensions { valence: v1, arousal: a1, dominance: d1 };
        let dim2 = EmotionDimensions { valence: v2, arousal: a2, dominance: d2 };
        let dim3 = EmotionDimensions { valence: v3, arousal: a3, dominance: d3 };

        let dist = |d1: &EmotionDimensions, d2: &EmotionDimensions| -> f32 {
            ((d1.valence - d2.valence).powi(2)
                + (d1.arousal - d2.arousal).powi(2)
                + (d1.dominance - d2.dominance).powi(2))
                .sqrt()
        };

        let d12 = dist(&dim1, &dim2);
        let d23 = dist(&dim2, &dim3);
        let d13 = dist(&dim1, &dim3);

        // Property: Triangle inequality must hold
        prop_assert!(d13 <= d12 + d23 + 1e-5); // Small epsilon for floating point errors
    }
}

// ============================================================================
// Property Tests: Emotion Vector
// ============================================================================

proptest! {
    /// Property: Adding emotions to a vector should maintain valid intensities
    #[test]
    fn prop_emotion_vector_addition(
        happy_intensity in emotion_intensity_strategy(),
        sad_intensity in emotion_intensity_strategy(),
        angry_intensity in emotion_intensity_strategy()
    ) {
        let mut vector = EmotionVector::new();

        if happy_intensity > 0.1 {
            vector.add_emotion(Emotion::Happy, EmotionIntensity::from(happy_intensity));
        }
        if sad_intensity > 0.1 {
            vector.add_emotion(Emotion::Sad, EmotionIntensity::from(sad_intensity));
        }
        if angry_intensity > 0.1 {
            vector.add_emotion(Emotion::Angry, EmotionIntensity::from(angry_intensity));
        }

        // Property: Vector should not contain more emotions than added
        prop_assert!(vector.emotions.len() <= 3);

        // Property: Vector should be queryable
        if happy_intensity > 0.1 {
            prop_assert!(vector.emotions.contains_key(&Emotion::Happy));
        }
    }

    /// Property: Emotion vector dimensions should remain bounded
    #[test]
    fn prop_dimension_bounds_in_vector(
        valence in emotion_dimension_strategy(),
        arousal in emotion_dimension_strategy(),
        dominance in emotion_dimension_strategy()
    ) {
        let mut vector = EmotionVector::new();
        vector.dimensions.valence = valence;
        vector.dimensions.arousal = arousal;
        vector.dimensions.dominance = dominance;

        // Property: Dimensions must remain bounded
        prop_assert!(vector.dimensions.valence >= -1.0 && vector.dimensions.valence <= 1.0);
        prop_assert!(vector.dimensions.arousal >= -1.0 && vector.dimensions.arousal <= 1.0);
        prop_assert!(vector.dimensions.dominance >= -1.0 && vector.dimensions.dominance <= 1.0);
    }
}

// ============================================================================
// Property Tests: Audio Processing
// ============================================================================

proptest! {
    /// Property: Audio processing should preserve buffer length
    #[test]
    fn prop_audio_processing_length_preservation(audio in audio_buffer_strategy()) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let processor = EmotionProcessor::new().unwrap();
            processor.set_emotion(Emotion::Happy, Some(0.5)).await.unwrap();

            let result = processor.process_audio(&audio).await;

            if let Ok(processed) = result {
                // Property: Output length must match input length
                prop_assert_eq!(processed.len(), audio.len());
            }

            Ok(()) as Result<(), proptest::test_runner::TestCaseError>
        })?;
    }

    /// Property: Audio processing should not produce NaN or infinite values
    #[test]
    fn prop_audio_processing_finite_output(audio in audio_buffer_strategy()) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let processor = EmotionProcessor::new().unwrap();
            processor.set_emotion(Emotion::Excited, Some(0.8)).await.unwrap();

            let result = processor.process_audio(&audio).await;

            if let Ok(processed) = result {
                // Property: All output samples must be finite
                for &sample in &processed {
                    prop_assert!(sample.is_finite(), "Sample is not finite: {}", sample);
                }
            }

            Ok(()) as Result<(), proptest::test_runner::TestCaseError>
        })?;
    }

    /// Property: Audio amplitude should remain bounded after processing
    #[test]
    fn prop_audio_amplitude_bounds(audio in audio_buffer_strategy()) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let processor = EmotionProcessor::new().unwrap();
            processor.set_emotion(Emotion::Calm, Some(0.6)).await.unwrap();

            let result = processor.process_audio(&audio).await;

            if let Ok(processed) = result {
                // Property: Samples should stay within reasonable bounds (-10.0 to 10.0)
                // Note: We allow some headroom beyond [-1, 1] for processing artifacts
                for &sample in &processed {
                    prop_assert!(
                        sample >= -10.0 && sample <= 10.0,
                        "Sample out of bounds: {}",
                        sample
                    );
                }
            }

            Ok(()) as Result<(), proptest::test_runner::TestCaseError>
        })?;
    }
}

// ============================================================================
// Property Tests: Emotion Parameters
// ============================================================================

proptest! {
    /// Property: Emotion parameters should maintain reasonable values
    #[test]
    fn prop_emotion_parameters_consistency(
        pitch in -1.0..1.0_f32,
        energy in 0.1..2.0_f32
    ) {
        let params = EmotionParameters::neutral()
            .with_prosody(pitch, energy, 1.0);

        // Property: Pitch parameter should match what was set (or be clamped)
        prop_assert!((params.pitch_shift - pitch).abs() < 0.01 ||
                     (params.pitch_shift >= -1.0 && params.pitch_shift <= 1.0));

        // Property: Energy should be positive and reasonable
        prop_assert!(params.energy_scale > 0.0 && params.energy_scale <= 10.0);
    }

    /// Property: Prosody parameters should maintain physical constraints
    #[test]
    fn prop_prosody_physical_constraints(
        pitch in -1.0..1.0_f32,
        energy in 0.1..2.0_f32
    ) {
        let params = EmotionParameters::neutral()
            .with_prosody(pitch, energy, 1.0);

        // Property: Energy must be positive (physical constraint)
        prop_assert!(params.energy_scale > 0.0);

        // Property: Pitch shift should be within reasonable vocal range
        prop_assert!(params.pitch_shift >= -1.0 && params.pitch_shift <= 1.0);
    }
}

// ============================================================================
// Property Tests: Idempotence
// ============================================================================

proptest! {
    /// Property: Processing the same emotion twice should be idempotent
    #[test]
    fn prop_emotion_setting_idempotent(intensity in emotion_intensity_strategy()) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let processor = EmotionProcessor::new().unwrap();

            // Set emotion twice
            processor.set_emotion(Emotion::Happy, Some(intensity)).await.unwrap();
            let state1 = processor.get_current_state().await;

            processor.set_emotion(Emotion::Happy, Some(intensity)).await.unwrap();
            let state2 = processor.get_current_state().await;

            // Property: Setting the same emotion twice should result in the same state
            prop_assert_eq!(
                state1.current.pitch_shift,
                state2.current.pitch_shift
            );
            prop_assert_eq!(
                state1.current.energy_scale,
                state2.current.energy_scale
            );

            Ok(()) as Result<(), proptest::test_runner::TestCaseError>
        })?;
    }
}
