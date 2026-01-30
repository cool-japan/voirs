//! # QualityMetrics - Trait Implementations
//!
//! This module contains trait implementations for `QualityMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for QualityMetrics {
    fn default() -> Self {
        QualityMetrics {
            overall_quality: None,
            audio_quality: None,
            similarity_metrics: None,
            naturalness_score: None,
            intelligibility_score: None,
        }
    }
}
