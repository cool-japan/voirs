//! # AcousticFeatures - Trait Implementations
//!
//! This module contains trait implementations for `AcousticFeatures`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

#[cfg(feature = "acoustic-integration")]
impl Default for AcousticFeatures {
    fn default() -> Self {
        Self {
            f0_contour: vec![150.0; 100], // Default F0 around 150 Hz
            formants: FormantFrequencies::default(),
            spectral_envelope: vec![0.0; 512],
            temporal_features: TemporalFeatures::default(),
            harmonic_features: HarmonicFeatures::default(),
            frame_count: 100,
            sample_rate: 44100.0,
        }
    }
}

#[cfg(not(feature = "acoustic-integration"))]
impl Default for AcousticFeatures {
    fn default() -> Self {
        Self { placeholder: true }
    }
}
