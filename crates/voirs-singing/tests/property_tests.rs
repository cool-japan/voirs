//! Property-Based Testing for VoiRS Singing
//!
//! This module contains property-based tests using proptest to verify
//! invariants and edge cases across the singing synthesis system.

use proptest::prelude::*;
use voirs_singing::prelude::*;

/// Strategy for generating valid frequencies (20Hz - 20kHz)
fn frequency_strategy() -> impl Strategy<Value = f32> {
    20.0f32..20000.0
}

/// Strategy for generating valid durations (0.01s - 10s)
fn duration_strategy() -> impl Strategy<Value = f32> {
    0.01f32..10.0
}

/// Strategy for generating valid velocities (0.0 - 1.0)
fn velocity_strategy() -> impl Strategy<Value = f32> {
    0.0f32..=1.0
}

/// Strategy for generating valid sample rates
fn sample_rate_strategy() -> impl Strategy<Value = f32> {
    prop_oneof![
        Just(8000.0),
        Just(16000.0),
        Just(22050.0),
        Just(44100.0),
        Just(48000.0),
        Just(96000.0),
    ]
}

// ============================================================================
// Pitch Contour Properties
// ============================================================================

proptest! {
    /// Property: Pitch contour interpolation should always return valid frequencies
    #[test]
    fn prop_pitch_contour_interpolation_always_valid(
        f0_start in frequency_strategy(),
        f0_end in frequency_strategy(),
        interpolation_time in 0.0f32..1.0,
    ) {
        let time_points = vec![0.0, 1.0];
        let f0_values = vec![f0_start, f0_end];
        let contour = PitchContour::new(time_points, f0_values);

        let interpolated = contour.f0_at_time(interpolation_time);

        // Interpolated value should be between start and end (or equal)
        let min_f0 = f0_start.min(f0_end);
        let max_f0 = f0_start.max(f0_end);
        prop_assert!(interpolated >= min_f0 - 1.0); // Small tolerance for floating point
        prop_assert!(interpolated <= max_f0 + 1.0);

        // Should always be positive
        prop_assert!(interpolated > 0.0);

        // Should be a finite number
        prop_assert!(interpolated.is_finite());
    }

    /// Property: Pitch smoothing should produce finite values
    #[test]
    fn prop_pitch_smoothing_produces_finite_values(
        base_freq in 200.0f32..600.0,
        jump_size in 50.0f32..200.0,
    ) {
        // Create a pitch contour with a discontinuity
        let time_points = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let f0_values = vec![base_freq, base_freq, base_freq + jump_size, base_freq + jump_size, base_freq, base_freq];
        let mut contour = PitchContour::new(time_points, f0_values);

        // Apply smoothing with moderate window
        contour.smooth(0.5);

        // All values should still be finite and positive after smoothing
        for &f0 in &contour.f0_values {
            prop_assert!(f0.is_finite(), "Smoothing produced non-finite value");
            prop_assert!(f0 > 0.0, "Smoothing produced non-positive value");
        }

        // Smoothed values should be within reasonable range of original
        prop_assert!(contour.f0_values.iter().all(|&f| f >= base_freq * 0.5 && f <= (base_freq + jump_size) * 1.5));
    }
}

// ============================================================================
// Musical Note Properties
// ============================================================================

proptest! {
    /// Property: Musical note frequency calculation should be consistent
    #[test]
    fn prop_note_frequency_calculation_consistent(
        octave in 1u8..7, // Reasonable singing range (C1 to B7)
    ) {
        let note_event = NoteEvent::new("A".to_string(), octave, 1.0, 0.8);
        let note = MusicalNote::new(note_event.clone(), 0.0, 1.0);

        // Frequency should be in audible range
        prop_assert!(note.event.frequency >= 20.0 && note.event.frequency <= 20000.0);
        prop_assert!(note.event.frequency.is_finite());

        // For A notes, frequency should follow A440 standard
        // A4 = 440Hz, doubling each octave
        let expected_a4_freq = 440.0;
        let octave_diff = octave as i32 - 4;
        let expected_freq = expected_a4_freq * 2.0f32.powi(octave_diff);

        // Allow small floating point error
        prop_assert!((note.event.frequency - expected_freq).abs() < 1.0);
    }

    /// Property: Note duration should always be positive
    #[test]
    fn prop_note_duration_always_positive(
        duration in duration_strategy(),
        start_time in 0.0f32..100.0,
    ) {
        let note_event = NoteEvent::new("A".to_string(), 4, duration, 0.8);
        let note = MusicalNote::new(note_event, start_time, duration);

        prop_assert!(note.duration > 0.0);
        prop_assert!(note.duration.is_finite());
        prop_assert!(note.start_time >= 0.0);
    }
}

// ============================================================================
// Musical Score Properties
// ============================================================================

proptest! {
    /// Property: Adding notes to score should preserve note count
    #[test]
    fn prop_score_preserves_note_count(
        num_notes in 1usize..50,
    ) {
        let mut score = MusicalScore::new("Test".to_string(), "Proptest".to_string());

        for i in 0..num_notes {
            let note_event = NoteEvent::new("C".to_string(), 4, 1.0, 0.8);
            let note = MusicalNote::new(note_event, i as f32, 1.0);
            score.add_note(note);
        }

        prop_assert_eq!(score.notes.len(), num_notes);
    }

    /// Property: Score duration should equal longest note end time
    #[test]
    fn prop_score_duration_calculation(
        num_notes in 1usize..10,
    ) {
        let mut score = MusicalScore::new("Test".to_string(), "Proptest".to_string());
        let mut max_end_time = 0.0f32;

        for i in 0..num_notes {
            let start = i as f32;
            let duration = 1.0;
            max_end_time = max_end_time.max(start + duration);

            let note_event = NoteEvent::new("C".to_string(), 4, duration, 0.8);
            let note = MusicalNote::new(note_event, start, duration);
            score.add_note(note);
        }

        // Calculate score duration manually
        let calc_duration = score.notes.iter()
            .map(|n| n.start_time + n.duration)
            .fold(0.0f32, f32::max);
        prop_assert!((calc_duration - max_end_time).abs() < 0.01);
    }
}

// ============================================================================
// Audio Processing Properties
// ============================================================================

proptest! {
    /// Property: Resampling should preserve audio length relationship
    #[test]
    fn prop_resampling_preserves_length_relationship(
        input_sr in sample_rate_strategy(),
        output_sr in sample_rate_strategy(),
        input_length in 100usize..1000,
    ) {
        let mut resampler = HighQualityResampler::new(input_sr, output_sr);

        let input_audio: Vec<f32> = vec![0.0; input_length];
        let output_audio = resampler.resample(&input_audio);

        // Output length should be proportional to sample rate ratio
        let expected_length = ((input_length as f32 * output_sr / input_sr).ceil() as usize).max(1);
        let actual_length = output_audio.len();

        // Allow small tolerance due to windowing and filtering
        let tolerance = ((expected_length as f32 * 0.15) as usize).max(30);
        prop_assert!(
            actual_length >= expected_length.saturating_sub(tolerance) &&
            actual_length <= expected_length + tolerance,
            "Expected length {} ± {}, got {}",
            expected_length,
            tolerance,
            actual_length
        );
    }

    /// Property: Dynamic range processor should not corrupt audio
    #[test]
    fn prop_dynamic_range_no_corruption(
        input_amplitude in -1.0f32..=1.0,
    ) {
        let processor = DynamicRangeProcessor::new();

        let mut input_signal = vec![input_amplitude; 1000];
        let original_length = input_signal.len();

        processor.process(&mut input_signal);

        // Output should have same length
        prop_assert_eq!(input_signal.len(), original_length);

        // Output should be within valid range
        for &sample in &input_signal {
            prop_assert!(sample >= -2.0 && sample <= 2.0);
            prop_assert!(sample.is_finite());
        }
    }
}

// ============================================================================
// Synthesis Quality Properties
// ============================================================================

proptest! {
    /// Property: Note events should have valid parameters
    #[test]
    fn prop_note_event_validity(
        octave in 2u8..6,
        duration in 0.1f32..2.0,
        velocity in 0.1f32..1.0,
    ) {
        let note_event = NoteEvent::new("C".to_string(), octave, duration, velocity);

        prop_assert!(note_event.duration > 0.0);
        prop_assert!(note_event.frequency > 0.0);
        prop_assert!(note_event.frequency.is_finite());
        prop_assert!(note_event.velocity >= 0.0 && note_event.velocity <= 1.0);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_zero_duration_handling() {
        // Ensure system handles very small durations gracefully
        let note_event = NoteEvent::new("A".to_string(), 4, 0.001, 0.8);
        let note = MusicalNote::new(note_event, 0.0, 0.001);
        assert!(note.duration > 0.0);
    }

    #[test]
    fn test_extreme_frequencies() {
        // Test extremely low frequency
        let low_freq = 20.0;
        let contour = PitchContour::new(vec![0.0, 1.0], vec![low_freq, low_freq]);
        assert!(contour.f0_at_time(0.5).is_finite());

        // Test extremely high frequency
        let high_freq = 15000.0;
        let contour_high = PitchContour::new(vec![0.0, 1.0], vec![high_freq, high_freq]);
        assert!(contour_high.f0_at_time(0.5).is_finite());
    }

    #[test]
    fn test_rapid_pitch_changes() {
        // Test rapid pitch changes don't cause instability
        let time_points = vec![0.0, 0.01, 0.02, 0.03];
        let f0_values = vec![220.0, 880.0, 220.0, 880.0]; // Rapid octave jumps
        let contour = PitchContour::new(time_points, f0_values);

        for i in 0..30 {
            let t = i as f32 * 0.001;
            let f0 = contour.f0_at_time(t);
            assert!(f0.is_finite());
            assert!(f0 > 0.0);
        }
    }

    #[test]
    fn test_empty_contour_handling() {
        // System should handle empty or minimal contours
        let contour = PitchContour::new(vec![], vec![]);
        assert!(contour.time_points.is_empty());
        assert!(contour.f0_values.is_empty());
    }

    #[test]
    fn test_single_point_contour() {
        // Single point contours should work
        let contour = PitchContour::new(vec![0.0], vec![440.0]);
        let f0 = contour.f0_at_time(0.0);
        assert!((f0 - 440.0).abs() < 0.1);
    }

    #[test]
    fn test_musical_score_empty() {
        let score = MusicalScore::new("Empty".to_string(), "Test".to_string());
        assert_eq!(score.notes.len(), 0);
    }

    #[test]
    fn test_note_event_extremes() {
        // Test with extreme but valid values
        let note_event = NoteEvent::new("C".to_string(), 0, 10.0, 1.0);
        assert!(note_event.frequency > 0.0);

        let note_event2 = NoteEvent::new("B".to_string(), 8, 0.01, 0.01);
        assert!(note_event2.frequency > 0.0);
    }
}
