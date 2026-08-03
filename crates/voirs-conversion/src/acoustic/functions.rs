//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Error, Result};
#[cfg(feature = "acoustic-integration")]
use voirs_acoustic;

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use super::*;
    #[test]
    fn test_acoustic_adapter_creation() {
        let adapter = AcousticConversionAdapter::new();
        assert!(matches!(adapter, AcousticConversionAdapter { .. }));
    }
    #[cfg(feature = "acoustic-integration")]
    #[tokio::test]
    async fn test_acoustic_conversion_validation() {
        let adapter = AcousticConversionAdapter::new();
        let audio = vec![0.1, 0.2, 0.3, 0.4];
        let characteristics = crate::types::VoiceCharacteristics::new();
        let result = adapter
            .convert_with_acoustic_model(&[], &characteristics)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
        let result = adapter
            .convert_with_acoustic_model(&audio, &characteristics)
            .await;
        assert!(result.is_ok());
    }
    #[cfg(feature = "acoustic-integration")]
    #[tokio::test]
    async fn test_feature_interpolation() {
        let adapter = AcousticConversionAdapter::new();
        let audio = vec![0.1, 0.2, 0.3, 0.4];
        let source_features = AcousticFeatures::default();
        let target_features = AcousticFeatures::default();
        let result = adapter
            .convert_with_feature_interpolation(&audio, &source_features, &target_features, 1.5)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be between"));
        let result = adapter
            .convert_with_feature_interpolation(&audio, &source_features, &target_features, 0.5)
            .await;
        assert!(result.is_ok());
    }
    #[cfg(feature = "acoustic-integration")]
    #[test]
    fn test_acoustic_features_default() {
        let features = AcousticFeatures::default();
        assert!(!features.f0_contour.is_empty());
        assert!(!features.spectral_envelope.is_empty());
        assert_eq!(features.frame_count, 100);
        assert_eq!(features.sample_rate, 44100.0);
    }
    #[cfg(feature = "acoustic-integration")]
    #[test]
    fn test_formant_frequencies() {
        let formants = FormantFrequencies::default();
        assert!(!formants.f1.is_empty());
        assert!(!formants.f2.is_empty());
        assert!(!formants.f3.is_empty());
        assert_eq!(formants.f1.len(), 100);
    }
    #[cfg(feature = "acoustic-integration")]
    #[test]
    fn test_gender_characteristics() {
        let mut features = AcousticFeatures::default();
        let original_f0 = features.f0_contour[0];
        features.apply_male_characteristics();
        assert!(features.f0_contour[0] < original_f0);
        let mut features = AcousticFeatures::default();
        features.apply_female_characteristics();
        assert!(features.f0_contour[0] > original_f0);
    }
    #[cfg(feature = "acoustic-integration")]
    #[test]
    fn test_acoustic_conversion_context() {
        let mut context = AcousticConversionContext::new(100.0, 44100.0);
        assert!(!context.has_sufficient_context());
        let chunk1 = vec![0.1; 1000];
        context.add_audio_chunk(&chunk1);
        let chunk2 = vec![0.2; 1000];
        context.add_audio_chunk(&chunk2);
        assert!(context.has_sufficient_context());
        let context_audio = context.get_context_window();
        assert_eq!(context_audio.len(), 2000);
    }
    #[cfg(feature = "acoustic-integration")]
    #[test]
    fn test_acoustic_state_update() {
        let mut state = AcousticState::default();
        let original_f0 = state.last_f0;
        let mut features = AcousticFeatures::default();
        features.f0_contour = vec![200.0, 220.0, 240.0];
        state.update_from_features(&features);
        assert_ne!(state.last_f0, original_f0);
        assert_eq!(state.last_f0, 240.0);
    }
    #[cfg(feature = "acoustic-integration")]
    #[tokio::test]
    async fn test_quality_preservation() {
        let adapter = AcousticConversionAdapter::new();
        let audio = vec![0.1; 1000];
        let characteristics = crate::types::VoiceCharacteristics::new();
        let result = adapter
            .convert_with_quality_preservation(&audio, &characteristics, 1.5)
            .await;
        assert!(result.is_err());
        let result = adapter
            .convert_with_quality_preservation(&audio, &characteristics, 0.8)
            .await;
        assert!(result.is_ok());
        let conversion_result = result.unwrap();
        assert!(!conversion_result.audio.is_empty());
        assert!(conversion_result.quality_score >= 0.0 && conversion_result.quality_score <= 1.0);
    }
    #[cfg(feature = "acoustic-integration")]
    #[test]
    fn test_f0_extraction() {
        let adapter = AcousticConversionAdapter::new();
        let audio = vec![0.1, 0.2, -0.1, -0.2];
        let result = adapter.extract_f0_contour(&audio);
        assert!(result.is_ok());
        let f0_contour = result.unwrap();
        assert!(!f0_contour.is_empty());
    }
    #[cfg(feature = "acoustic-integration")]
    #[test]
    fn test_formant_extraction() {
        let adapter = AcousticConversionAdapter::new();
        let audio = vec![0.1; 2048];
        let result = adapter.extract_formant_frequencies(&audio);
        assert!(result.is_ok());
        let formants = result.unwrap();
        assert!(!formants.f1.is_empty());
        assert!(!formants.f2.is_empty());
        assert!(!formants.f3.is_empty());
    }
    #[cfg(feature = "acoustic-integration")]
    #[tokio::test]
    async fn test_convert_with_acoustic_model_uses_real_pitch_shift_not_amplitude_scaling() {
        let adapter = AcousticConversionAdapter::new();
        let sr = 44100.0;
        let audio: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * 150.0 * i as f32 / sr).sin() * 0.5)
            .collect();

        let mut characteristics = crate::types::VoiceCharacteristics::new();
        characteristics.pitch.mean_f0 = 300.0; // ~one octave above the tone's own F0

        let converted = adapter
            .convert_with_acoustic_model(&audio, &characteristics)
            .await
            .unwrap();

        assert_eq!(converted.len(), audio.len());
        // The old bug was `output[i] = input[i] * pitch_factor` for every
        // sample: a fixed per-sample scalar relationship. A real pitch
        // shift (moving spectral content to a different frequency) does
        // not preserve that relationship - check the per-sample ratio is
        // NOT a near-constant value (which is what plain amplitude scaling
        // would produce), rather than asserting an absolute peak-amplitude
        // bound (the real phase vocoder's exact peak level is an
        // implementation detail, not the property under test here).
        let ratios: Vec<f32> = audio
            .iter()
            .zip(converted.iter())
            .filter(|(&i, _)| i.abs() > 0.05)
            .map(|(&i, &o)| o / i)
            .collect();
        assert!(!ratios.is_empty());
        let mean_ratio = ratios.iter().sum::<f32>() / ratios.len() as f32;
        let ratio_variance =
            ratios.iter().map(|r| (r - mean_ratio).powi(2)).sum::<f32>() / ratios.len() as f32;
        assert!(
            ratio_variance > 0.01,
            "output must not be a fixed per-sample scalar multiple of the input audio (that \
             would indicate amplitude scaling, not pitch shifting); ratio variance was \
             {ratio_variance}"
        );
    }

    #[cfg(feature = "acoustic-integration")]
    #[test]
    fn test_f0_contour_detects_real_tone_frequency() {
        let adapter = AcousticConversionAdapter::new();
        let sr = 44100.0;
        let audio: Vec<f32> = (0..8192)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / sr).sin() * 0.6)
            .collect();

        let contour = adapter.extract_f0_contour(&audio).unwrap();
        let voiced: Vec<f32> = contour.into_iter().filter(|&f| f > 0.0).collect();
        assert!(
            !voiced.is_empty(),
            "a clean 200Hz tone should be detected as voiced"
        );
        let mean = voiced.iter().sum::<f32>() / voiced.len() as f32;
        assert!(
            (mean - 200.0).abs() < 20.0,
            "detected mean F0 {mean} should be close to the real 200Hz tone"
        );
    }

    #[cfg(feature = "acoustic-integration")]
    #[test]
    fn test_formant_frequencies_vary_with_spectral_content() {
        let adapter = AcousticConversionAdapter::new();
        let sr = 44100.0;
        let low: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr).sin() * 0.5)
            .collect();
        let high: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * 2500.0 * i as f32 / sr).sin() * 0.5)
            .collect();

        let low_formants = adapter.extract_formant_frequencies(&low).unwrap();
        let high_formants = adapter.extract_formant_frequencies(&high).unwrap();

        assert_ne!(
            low_formants.f1, high_formants.f1,
            "formant estimates must reflect real spectral content, not a fixed default contour"
        );
    }

    #[cfg(feature = "acoustic-integration")]
    #[tokio::test]
    async fn test_feature_interpolation_factor_actually_changes_output() {
        let adapter = AcousticConversionAdapter::new();
        let audio: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * 150.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();

        let mut source_features = AcousticFeatures::default();
        source_features.f0_contour = vec![150.0; 10];

        let mut target_features = AcousticFeatures::default();
        target_features.f0_contour = vec![300.0; 10];

        let at_zero = adapter
            .convert_with_feature_interpolation(&audio, &source_features, &target_features, 0.0)
            .await
            .unwrap();
        let at_one = adapter
            .convert_with_feature_interpolation(&audio, &source_features, &target_features, 1.0)
            .await
            .unwrap();

        assert_ne!(
            at_zero, at_one,
            "interpolation_factor must actually affect the output when source and target differ"
        );
    }

    #[cfg(feature = "acoustic-integration")]
    #[tokio::test]
    async fn test_quality_preservation_score_not_hardcoded() {
        let adapter = AcousticConversionAdapter::new();
        let audio: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * 150.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let characteristics = crate::types::VoiceCharacteristics::new();

        let result = adapter
            .convert_with_quality_preservation(&audio, &characteristics, 0.5)
            .await
            .unwrap();

        assert_ne!(
            result.quality_score, 0.85,
            "must not be the old hardcoded placeholder"
        );
        assert!((0.0..=1.0).contains(&result.quality_score));
        assert!(!result.original_features.f0_contour.is_empty());
    }

    #[cfg(feature = "acoustic-integration")]
    #[tokio::test]
    async fn test_realtime_acoustic_conversion_is_f0_aware() {
        let adapter = AcousticConversionAdapter::new();
        let mut context = AcousticConversionContext::new(50.0, 44100.0);

        let sr = 44100.0;
        let chunk: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * 150.0 * i as f32 / sr).sin() * 0.5)
            .collect();

        // Prime the context so has_sufficient_context() becomes true.
        let _ = adapter
            .convert_realtime_acoustic(&chunk, &AcousticFeatures::default(), &mut context)
            .await
            .unwrap();

        let mut target_features = AcousticFeatures::default();
        target_features.f0_contour = vec![300.0; 10]; // request ~one octave up

        let output = adapter
            .convert_realtime_acoustic(&chunk, &target_features, &mut context)
            .await
            .unwrap();

        assert_eq!(output.len(), chunk.len());
        // Same check as the non-realtime path: a fixed per-sample scalar
        // ratio would indicate amplitude scaling in disguise, not a real
        // (frequency-shifting) pitch conversion.
        let ratios: Vec<f32> = chunk
            .iter()
            .zip(output.iter())
            .filter(|(&i, _)| i.abs() > 0.05)
            .map(|(&i, &o)| o / i)
            .collect();
        assert!(!ratios.is_empty());
        let mean_ratio = ratios.iter().sum::<f32>() / ratios.len() as f32;
        let ratio_variance =
            ratios.iter().map(|r| (r - mean_ratio).powi(2)).sum::<f32>() / ratios.len() as f32;
        assert!(
            ratio_variance > 0.01,
            "real-time conversion must not be a fixed per-sample scalar multiple of the input \
             (amplitude scaling in disguise); ratio variance was {ratio_variance}"
        );
        assert!(
            context.previous_state.last_f0 > 1.0,
            "F0 state should be updated with a real estimate, not left at its default"
        );
    }

    #[cfg(not(feature = "acoustic-integration"))]
    #[tokio::test]
    async fn test_acoustic_integration_disabled() {
        let adapter = AcousticConversionAdapter::new();
        let audio = vec![0.1, 0.2, 0.3, 0.4];
        let characteristics = crate::types::VoiceCharacteristics::new();
        let result = adapter
            .convert_with_acoustic_model(&audio, &characteristics)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not enabled"));
    }
}
