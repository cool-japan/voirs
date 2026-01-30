//! # QualityTargetsStatistics - Trait Implementations
//!
//! This module contains trait implementations for `QualityTargetsStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::QualityTargetsStatistics;

impl Default for QualityTargetsStatistics {
    fn default() -> Self {
        Self {
            total_measurements: 0,
            target_similarity_achievement_rate: 0.0,
            source_preservation_achievement_rate: 0.0,
            mos_achievement_rate: 0.0,
            artifact_level_achievement_rate: 0.0,
            average_overall_achievement: 0.0,
        }
    }
}
