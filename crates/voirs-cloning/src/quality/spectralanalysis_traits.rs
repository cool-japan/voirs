//! # SpectralAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `SpectralAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SpectralAnalysis;

impl Default for SpectralAnalysis {
    fn default() -> Self {
        Self {
            spectral_centroid_similarity: 0.0,
            spectral_rolloff_similarity: 0.0,
            spectral_flatness_similarity: 0.0,
            harmonic_similarity: 0.0,
            formant_similarity: 0.0,
            bandwidth_similarity: 0.0,
        }
    }
}
