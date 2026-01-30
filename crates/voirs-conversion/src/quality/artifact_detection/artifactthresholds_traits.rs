//! # ArtifactThresholds - Trait Implementations
//!
//! This module contains trait implementations for `ArtifactThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ArtifactThresholds;

impl Default for ArtifactThresholds {
    fn default() -> Self {
        Self {
            click_threshold: 0.1,
            metallic_threshold: 0.15,
            buzzing_threshold: 0.12,
            pitch_variation_threshold: 0.2,
            spectral_discontinuity_threshold: 0.08,
            energy_spike_threshold: 0.25,
            hf_noise_threshold: 0.18,
            phase_artifact_threshold: 0.14,
            temporal_jitter_threshold: 0.05,
            spectral_tilt_threshold: 0.12,
            formant_tracking_threshold: 0.08,
            loudness_inconsistency_threshold: 0.15,
            channel_crosstalk_threshold: 0.1,
            interharmonic_distortion_threshold: 0.13,
            consonant_degradation_threshold: 0.11,
            vowel_coloration_threshold: 0.09,
            adaptation_rate: 0.01,
            min_confidence: 0.6,
        }
    }
}
