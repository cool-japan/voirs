//! # ArtifactAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `ArtifactAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ArtifactAnalysis;

impl Default for ArtifactAnalysis {
    fn default() -> Self {
        Self {
            click_detection: 0.0,
            discontinuity_detection: 0.0,
            aliasing_detection: 0.0,
            reverb_artifacts: 0.0,
            robotic_artifacts: 0.0,
            overall_artifact_score: 0.0,
        }
    }
}
