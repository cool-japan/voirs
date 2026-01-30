//! # SNRAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `SNRAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SNRAnalysis;

impl Default for SNRAnalysis {
    fn default() -> Self {
        Self {
            original_snr: 0.0,
            cloned_snr: 0.0,
            snr_degradation: 0.0,
            noise_floor: 0.0,
            dynamic_range: 0.0,
        }
    }
}
