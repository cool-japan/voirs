//! # MultilingualSpeakerModelConfig - Trait Implementations
//!
//! This module contains trait implementations for `MultilingualSpeakerModelConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use voirs_sdk::LanguageCode;

use super::data_types::MultilingualSpeakerModelConfig;

impl Default for MultilingualSpeakerModelConfig {
    fn default() -> Self {
        Self {
            enable_voice_transfer_quality: true,
            enable_speaker_identity_preservation: true,
            enable_language_adaptation: true,
            enable_acoustic_consistency: true,
            enable_perceptual_similarity: true,
            voice_transfer_quality_weight: 0.25,
            speaker_identity_preservation_weight: 0.25,
            language_adaptation_weight: 0.2,
            acoustic_consistency_weight: 0.15,
            perceptual_similarity_weight: 0.15,
            min_speaker_similarity_threshold: 0.7,
            max_voice_transfer_degradation: 0.3,
            target_languages: vec![
                LanguageCode::EnUs,
                LanguageCode::EsEs,
                LanguageCode::FrFr,
                LanguageCode::DeDe,
                LanguageCode::JaJp,
                LanguageCode::ZhCn,
            ],
        }
    }
}
