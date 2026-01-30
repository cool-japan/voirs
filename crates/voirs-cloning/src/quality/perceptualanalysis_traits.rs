//! # PerceptualAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `PerceptualAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PerceptualAnalysis;

impl Default for PerceptualAnalysis {
    fn default() -> Self {
        Self {
            loudness_similarity: 0.0,
            roughness_similarity: 0.0,
            sharpness_similarity: 0.0,
            pitch_similarity: 0.0,
            timber_similarity: 0.0,
        }
    }
}
