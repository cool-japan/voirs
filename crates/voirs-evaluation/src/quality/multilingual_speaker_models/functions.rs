//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::traits::{EvaluationResult, QualityScore};
use async_trait::async_trait;
use std::collections::HashMap;
use voirs_recognizer::traits::PhonemeAlignment;
use voirs_sdk::{AudioBuffer, LanguageCode};

use super::data_types::{
    F0Statistics, FormantStatistics, MultilingualSpeakerModelConfig,
    MultilingualSpeakerModelResult, ProsodicStatistics, SpeakerModel, SpectralStatistics,
    TemporalStatistics, VoiceCharacteristics, VoiceQualityStatistics,
};
use super::types::MultilingualSpeakerModelEvaluator;

/// Multilingual speaker model evaluation trait
#[async_trait]
pub trait MultilingualSpeakerModelEvaluationTrait {
    /// Evaluate multilingual speaker model
    async fn evaluate_multilingual_speaker_model(
        &mut self,
        speaker_id: &str,
        reference_audio: &AudioBuffer,
        reference_language: LanguageCode,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        phoneme_alignments: Option<&HashMap<LanguageCode, PhonemeAlignment>>,
    ) -> EvaluationResult<MultilingualSpeakerModelResult>;
    /// Get supported languages
    fn get_supported_languages(&self) -> Vec<LanguageCode>;
    /// Get speaker model
    fn get_speaker_model(&self, speaker_id: &str) -> Option<&SpeakerModel>;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_multilingual_speaker_model_evaluator_creation() {
        let config = MultilingualSpeakerModelConfig::default();
        let evaluator = MultilingualSpeakerModelEvaluator::new(config);
        assert!(!evaluator.get_supported_languages().is_empty());
    }
    #[tokio::test]
    async fn test_voice_characteristics_extraction() {
        let config = MultilingualSpeakerModelConfig::default();
        let mut evaluator = MultilingualSpeakerModelEvaluator::new(config);
        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
        let characteristics = evaluator
            .extract_voice_characteristics(&audio, LanguageCode::EnUs)
            .await
            .unwrap();
        assert!(characteristics.f0_stats.mean_f0 >= 0.0);
        assert!(!characteristics.formant_stats.mean_formants.is_empty());
    }
    #[tokio::test]
    async fn test_multilingual_speaker_model_evaluation() {
        let config = MultilingualSpeakerModelConfig::default();
        let mut evaluator = MultilingualSpeakerModelEvaluator::new(config);
        let reference_audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
        let mut target_audios = HashMap::new();
        target_audios.insert(
            LanguageCode::EsEs,
            AudioBuffer::new(vec![0.12; 16000], 16000, 1),
        );
        target_audios.insert(
            LanguageCode::FrFr,
            AudioBuffer::new(vec![0.11; 16000], 16000, 1),
        );
        let result = evaluator
            .evaluate_multilingual_speaker_model(
                "test_speaker",
                &reference_audio,
                LanguageCode::EnUs,
                &target_audios,
                None,
            )
            .await
            .unwrap();
        assert_eq!(result.reference_language, LanguageCode::EnUs);
        assert_eq!(result.target_languages.len(), 2);
        assert!(result.overall_quality >= 0.0 && result.overall_quality <= 1.0);
        assert!(result.evaluation_confidence >= 0.0 && result.evaluation_confidence <= 1.0);
    }
    #[test]
    fn test_voice_transfer_quality_calculation() {
        let config = MultilingualSpeakerModelConfig::default();
        let evaluator = MultilingualSpeakerModelEvaluator::new(config);
        let ref_characteristics = VoiceCharacteristics {
            f0_stats: F0Statistics {
                mean_f0: 150.0,
                f0_std: 20.0,
                f0_range: (100.0, 200.0),
                f0_variability: 0.3,
            },
            formant_stats: FormantStatistics {
                mean_formants: vec![500.0, 1500.0, 2500.0],
                formant_stds: vec![50.0, 100.0, 150.0],
                formant_bandwidths: vec![60.0, 80.0, 100.0],
            },
            spectral_stats: SpectralStatistics {
                spectral_centroid: 1000.0,
                spectral_spread: 500.0,
                spectral_tilt: -6.0,
                spectral_rolloff: 4000.0,
            },
            temporal_stats: TemporalStatistics {
                speaking_rate: 4.5,
                pause_frequency: 0.5,
                pause_duration: 0.3,
                rhythm_regularity: 0.7,
            },
            voice_quality_stats: VoiceQualityStatistics {
                jitter: 0.02,
                shimmer: 0.03,
                hnr: 20.0,
                spectral_noise: 0.1,
            },
            prosodic_stats: ProsodicStatistics {
                intonation_range: 0.5,
                stress_prominence: 0.7,
                rhythm_consistency: 0.8,
                prosodic_variability: 0.3,
            },
        };
        let target_characteristics = ref_characteristics.clone();
        let transfer_quality = evaluator.calculate_voice_transfer_quality(
            &ref_characteristics,
            &target_characteristics,
            LanguageCode::EnUs,
            LanguageCode::EsEs,
        );
        assert!(transfer_quality >= 0.8);
    }
    #[test]
    fn test_speaker_identity_preservation() {
        let config = MultilingualSpeakerModelConfig::default();
        let evaluator = MultilingualSpeakerModelEvaluator::new(config);
        let ref_characteristics = VoiceCharacteristics {
            f0_stats: F0Statistics {
                mean_f0: 150.0,
                f0_std: 20.0,
                f0_range: (100.0, 200.0),
                f0_variability: 0.3,
            },
            formant_stats: FormantStatistics {
                mean_formants: vec![500.0, 1500.0, 2500.0],
                formant_stds: vec![50.0, 100.0, 150.0],
                formant_bandwidths: vec![60.0, 80.0, 100.0],
            },
            spectral_stats: SpectralStatistics {
                spectral_centroid: 1000.0,
                spectral_spread: 500.0,
                spectral_tilt: -6.0,
                spectral_rolloff: 4000.0,
            },
            temporal_stats: TemporalStatistics {
                speaking_rate: 4.5,
                pause_frequency: 0.5,
                pause_duration: 0.3,
                rhythm_regularity: 0.7,
            },
            voice_quality_stats: VoiceQualityStatistics {
                jitter: 0.02,
                shimmer: 0.03,
                hnr: 20.0,
                spectral_noise: 0.1,
            },
            prosodic_stats: ProsodicStatistics {
                intonation_range: 0.5,
                stress_prominence: 0.7,
                rhythm_consistency: 0.8,
                prosodic_variability: 0.3,
            },
        };
        let target_characteristics = ref_characteristics.clone();
        let preservation_score = evaluator.calculate_speaker_identity_preservation(
            &ref_characteristics,
            &target_characteristics,
            LanguageCode::EnUs,
            LanguageCode::EsEs,
        );
        assert!(preservation_score >= 0.8);
    }
}
