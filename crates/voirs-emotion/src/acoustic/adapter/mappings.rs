//! Emotion-to-acoustic parameter mappings

use crate::types::Emotion;
use std::collections::HashMap;

use super::super::params::AcousticEmotionMapping;
use super::super::params::{
    AcousticConditioningParams, ProsodyModificationParams, SpeakerAdaptationParams,
};

/// Create default emotion-to-acoustic mappings
pub(super) fn create_default_emotion_mappings() -> HashMap<Emotion, AcousticEmotionMapping> {
    let mut mappings = HashMap::new();

    // Happy emotion mapping
    mappings.insert(
        Emotion::Happy,
        AcousticEmotionMapping {
            emotion: Emotion::Happy,
            acoustic_params: AcousticConditioningParams {
                energy_boost: 1.2,
                spectral_brightness: 0.3,
                harmonic_richness: 1.1,
                temporal_dynamics: 1.15,
            },
            speaker_params: SpeakerAdaptationParams {
                pitch_range_expansion: 1.3,
                formant_shift: 1.1,
                voice_quality_adjustment: 0.2,
            },
            prosody_params: ProsodyModificationParams {
                pitch_contour_variation: 1.4,
                rhythm_modification: 1.2,
                stress_pattern_enhancement: 1.1,
            },
        },
    );

    // Sad emotion mapping
    mappings.insert(
        Emotion::Sad,
        AcousticEmotionMapping {
            emotion: Emotion::Sad,
            acoustic_params: AcousticConditioningParams {
                energy_boost: 0.7,
                spectral_brightness: -0.3,
                harmonic_richness: 0.8,
                temporal_dynamics: 0.85,
            },
            speaker_params: SpeakerAdaptationParams {
                pitch_range_expansion: 0.7,
                formant_shift: 0.9,
                voice_quality_adjustment: -0.2,
            },
            prosody_params: ProsodyModificationParams {
                pitch_contour_variation: 0.6,
                rhythm_modification: 0.8,
                stress_pattern_enhancement: 0.9,
            },
        },
    );

    // Add other emotion mappings...
    for emotion in [
        Emotion::Angry,
        Emotion::Fear,
        Emotion::Surprise,
        Emotion::Calm,
        Emotion::Excited,
        Emotion::Tender,
        Emotion::Confident,
        Emotion::Melancholic,
    ] {
        mappings.insert(emotion.clone(), create_emotion_mapping(&emotion));
    }

    mappings
}

/// Create emotion mapping for a given emotion
pub(super) fn create_emotion_mapping(emotion: &Emotion) -> AcousticEmotionMapping {
    let (energy, brightness, pitch_exp, rhythm) = match emotion {
        Emotion::Angry => (1.4, 0.2, 1.2, 1.3),
        Emotion::Fear => (1.1, -0.1, 1.5, 1.4),
        Emotion::Surprise => (1.3, 0.4, 1.6, 1.2),
        Emotion::Calm => (0.8, 0.0, 0.8, 0.9),
        Emotion::Excited => (1.5, 0.5, 1.4, 1.4),
        Emotion::Tender => (0.9, -0.1, 0.9, 0.95),
        Emotion::Confident => (1.2, 0.1, 1.1, 1.1),
        Emotion::Melancholic => (0.6, -0.4, 0.7, 0.8),
        _ => (1.0, 0.0, 1.0, 1.0), // Neutral
    };

    AcousticEmotionMapping {
        emotion: emotion.clone(),
        acoustic_params: AcousticConditioningParams {
            energy_boost: energy,
            spectral_brightness: brightness,
            harmonic_richness: energy * 0.8,
            temporal_dynamics: rhythm,
        },
        speaker_params: SpeakerAdaptationParams {
            pitch_range_expansion: pitch_exp,
            formant_shift: 1.0 + brightness * 0.2,
            voice_quality_adjustment: brightness * 0.5,
        },
        prosody_params: ProsodyModificationParams {
            pitch_contour_variation: pitch_exp,
            rhythm_modification: rhythm,
            stress_pattern_enhancement: energy * 0.9,
        },
    }
}
