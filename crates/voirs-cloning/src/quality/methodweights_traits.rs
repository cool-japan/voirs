//! # MethodWeights - Trait Implementations
//!
//! This module contains trait implementations for `MethodWeights`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MethodWeights;

impl Default for MethodWeights {
    fn default() -> Self {
        Self {
            speaker_similarity_weight: 0.25,
            audio_quality_weight: 0.20,
            naturalness_weight: 0.20,
            content_preservation_weight: 0.15,
            prosodic_weight: 0.10,
            spectral_weight: 0.10,
        }
    }
}
