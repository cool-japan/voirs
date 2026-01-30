//! # HarmonicFeatures - Trait Implementations
//!
//! This module contains trait implementations for `HarmonicFeatures`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

#[cfg(feature = "acoustic-integration")]
impl Default for HarmonicFeatures {
    fn default() -> Self {
        Self {
            harmonic_to_noise_ratio: vec![15.0; 100],
            harmonic_strength: vec![0.8; 100],
            inharmonicity: vec![0.1; 100],
        }
    }
}
