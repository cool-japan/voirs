//! Property-based tests for signal processing
//!
//! These tests use proptest to validate signal processing properties across
//! a wide range of inputs, ensuring robustness and correctness.

use proptest::prelude::*;
use voirs_emotion::{
    signal_processing::{ProcessingQuality, SignalProcessingConfig, SignalProcessor},
    types::Emotion,
    Result,
};

// Property: Processing should never produce NaN or Inf values
proptest! {
    #[test]
    fn test_no_nan_or_inf_in_output(
        audio_len in 100usize..10000,
        intensity in 0.0f32..1.0,
    ) {
        let config = SignalProcessingConfig::minimal();
        let mut processor = SignalProcessor::new(config, 44100.0);

        let audio: Vec<f32> = (0..audio_len)
            .map(|i| (i as f32 * 0.001).sin() * 0.5)
            .collect();

        let emotions = vec![
            Emotion::Happy, Emotion::Sad, Emotion::Angry,
            Emotion::Calm, Emotion::Excited, Emotion::Neutral,
        ];

        for emotion in emotions {
            let result = processor.process_with_emotion(&audio, &emotion, intensity);
            prop_assert!(result.is_ok());

            if let Ok(output) = result {
                prop_assert!(output.iter().all(|&x| x.is_finite()));
            }
        }
    }
}

// Property: Output length should match input length for most operations
proptest! {
    #[test]
    fn test_output_length_preservation(
        audio_len in 100usize..5000,
        intensity in 0.0f32..1.0,
    ) {
        let mut config = SignalProcessingConfig::minimal();
        config.enable_breath = false; // Disable breath to preserve length

        let mut processor = SignalProcessor::new(config, 44100.0);

        let audio: Vec<f32> = vec![0.5; audio_len];

        let result = processor.process_with_emotion(&audio, &Emotion::Happy, intensity)?;
        prop_assert_eq!(result.len(), audio_len);
    }
}

// Property: Intensity clamping should work correctly
proptest! {
    #[test]
    fn test_intensity_clamping(
        audio_len in 100usize..1000,
        raw_intensity in -10.0f32..10.0,
    ) {
        let config = SignalProcessingConfig::minimal();
        let mut processor = SignalProcessor::new(config, 44100.0);

        let audio: Vec<f32> = vec![0.5; audio_len];

        // Should not panic even with out-of-range intensity
        let result = processor.process_with_emotion(&audio, &Emotion::Happy, raw_intensity);
        prop_assert!(result.is_ok());
    }
}

// Property: Empty audio should return empty output
proptest! {
    #[test]
    fn test_empty_audio_handling(
        intensity in 0.0f32..1.0,
    ) {
        let config = SignalProcessingConfig::default();
        let mut processor = SignalProcessor::new(config, 44100.0);

        let empty_audio: Vec<f32> = vec![];

        let result = processor.process_with_emotion(&empty_audio, &Emotion::Happy, intensity)?;
        prop_assert!(result.is_empty());
    }
}

// Property: Quality presets should have consistent relationships
#[test]
fn test_quality_preset_relationships() {
    let low = SignalProcessingConfig::preset(ProcessingQuality::Low);
    let medium = SignalProcessingConfig::preset(ProcessingQuality::Medium);
    let high = SignalProcessingConfig::preset(ProcessingQuality::High);
    let ultra = SignalProcessingConfig::preset(ProcessingQuality::Ultra);

    // FFT size should increase with quality
    assert!(low.fft_size < medium.fft_size);
    assert!(medium.fft_size < high.fft_size);
    assert!(high.fft_size <= ultra.fft_size);

    // Overlap factor should increase or stay same
    assert!(low.overlap_factor <= medium.overlap_factor);
    assert!(medium.overlap_factor <= high.overlap_factor);
}

// Property: Emotion analysis should always return valid emotion and confidence
proptest! {
    #[test]
    fn test_emotion_analysis_validity(
        audio_len in 50usize..5000,
    ) {
        let config = SignalProcessingConfig::full();
        let processor = SignalProcessor::new(config, 44100.0);

        let audio: Vec<f32> = (0..audio_len)
            .map(|i| ((i as f32 * 0.01).sin() + (i as f32 * 0.02).sin()) * 0.5)
            .collect();

        let (emotion, confidence) = processor.analyze_emotion(&audio)?;

        // Confidence should be in valid range
        prop_assert!(confidence >= 0.0 && confidence <= 1.0);

        // Emotion should be one of the expected types
        prop_assert!(matches!(
            emotion,
            Emotion::Happy | Emotion::Sad | Emotion::Angry |
            Emotion::Calm | Emotion::Excited | Emotion::Neutral
        ));
    }
}

// Property: Config updates should be consistent
proptest! {
    #[test]
    fn test_config_update_consistency(
        enable_formant: bool,
        enable_spectral: bool,
        enable_breath: bool,
    ) {
        let initial_config = SignalProcessingConfig::minimal();
        let mut processor = SignalProcessor::new(initial_config, 44100.0);

        let new_config = SignalProcessingConfig {
            enable_formant,
            enable_spectral,
            enable_breath,
            quality: ProcessingQuality::Medium,
            fft_size: 2048,
            overlap_factor: 0.5,
            adaptive_intensity: true,
        };

        processor.set_config(new_config.clone());

        prop_assert_eq!(processor.config().enable_formant, new_config.enable_formant);
        prop_assert_eq!(processor.config().enable_spectral, new_config.enable_spectral);
        prop_assert_eq!(processor.config().enable_breath, new_config.enable_breath);
    }
}
