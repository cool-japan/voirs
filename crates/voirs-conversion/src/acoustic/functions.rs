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
