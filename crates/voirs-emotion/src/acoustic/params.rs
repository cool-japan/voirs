//! Parameter types for acoustic emotion control
//!
//! This module defines various parameter structures for controlling
//! acoustic synthesis with emotion characteristics.

use crate::types::Emotion;

/// Emotion to acoustic model parameter mapping
#[derive(Debug, Clone)]
pub struct AcousticEmotionMapping {
    /// Emotion type
    pub emotion: Emotion,
    /// Acoustic model conditioning parameters
    pub acoustic_params: AcousticConditioningParams,
    /// Speaker adaptation parameters
    pub speaker_params: SpeakerAdaptationParams,
    /// Prosody modification parameters
    pub prosody_params: ProsodyModificationParams,
}

/// Acoustic model conditioning parameters
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticConditioningParams {
    /// Energy boost factor for the emotion
    pub energy_boost: f32,
    /// Spectral brightness adjustment
    pub spectral_brightness: f32,
    /// Harmonic richness factor
    pub harmonic_richness: f32,
    /// Temporal dynamics adjustment
    pub temporal_dynamics: f32,
}

/// Speaker adaptation parameters for emotion
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerAdaptationParams {
    /// Pitch range expansion factor
    pub pitch_range_expansion: f32,
    /// Formant frequency shift
    pub formant_shift: f32,
    /// Voice quality adjustment (-1.0 to 1.0)
    pub voice_quality_adjustment: f32,
}

/// Prosody modification parameters
#[derive(Debug, Clone, PartialEq)]
pub struct ProsodyModificationParams {
    /// Pitch contour variation intensity
    pub pitch_contour_variation: f32,
    /// Rhythm modification factor
    pub rhythm_modification: f32,
    /// Stress pattern enhancement
    pub stress_pattern_enhancement: f32,
}

impl Default for AcousticConditioningParams {
    fn default() -> Self {
        Self {
            energy_boost: 1.0,
            spectral_brightness: 0.0,
            harmonic_richness: 1.0,
            temporal_dynamics: 1.0,
        }
    }
}

impl Default for SpeakerAdaptationParams {
    fn default() -> Self {
        Self {
            pitch_range_expansion: 1.0,
            formant_shift: 1.0,
            voice_quality_adjustment: 0.0,
        }
    }
}

impl Default for ProsodyModificationParams {
    fn default() -> Self {
        Self {
            pitch_contour_variation: 1.0,
            rhythm_modification: 1.0,
            stress_pattern_enhancement: 1.0,
        }
    }
}
