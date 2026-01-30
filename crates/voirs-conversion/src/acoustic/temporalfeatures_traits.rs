//! # TemporalFeatures - Trait Implementations
//!
//! This module contains trait implementations for `TemporalFeatures`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

#[cfg(feature = "acoustic-integration")]
impl Default for TemporalFeatures {
    fn default() -> Self {
        Self {
            energy_contour: vec![0.5; 100],
            zero_crossing_rate: vec![0.1; 100],
            spectral_flux: vec![0.2; 100],
        }
    }
}
