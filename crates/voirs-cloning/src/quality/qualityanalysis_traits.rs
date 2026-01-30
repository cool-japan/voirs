//! # QualityAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `QualityAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    ArtifactAnalysis, PerceptualAnalysis, QualityAnalysis, SNRAnalysis, SpectralAnalysis,
    TemporalAnalysis,
};

impl Default for QualityAnalysis {
    fn default() -> Self {
        Self {
            snr_analysis: SNRAnalysis::default(),
            spectral_analysis: SpectralAnalysis::default(),
            temporal_analysis: TemporalAnalysis::default(),
            perceptual_analysis: PerceptualAnalysis::default(),
            artifact_analysis: ArtifactAnalysis::default(),
        }
    }
}
