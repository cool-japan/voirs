//! # TemporalAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `TemporalAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TemporalAnalysis;

impl Default for TemporalAnalysis {
    fn default() -> Self {
        Self {
            duration_similarity: 0.0,
            rhythm_similarity: 0.0,
            energy_envelope_similarity: 0.0,
            pause_similarity: 0.0,
            speech_rate_similarity: 0.0,
        }
    }
}
