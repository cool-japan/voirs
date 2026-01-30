//! # QualityTargetsConfig - Trait Implementations
//!
//! This module contains trait implementations for `QualityTargetsConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::QualityTargetsConfig;

impl Default for QualityTargetsConfig {
    fn default() -> Self {
        Self {
            target_similarity_threshold: 0.85,
            source_preservation_threshold: 0.90,
            mos_threshold: 4.0,
            artifact_level_threshold: 0.05,
            enable_detailed_tracking: true,
        }
    }
}
