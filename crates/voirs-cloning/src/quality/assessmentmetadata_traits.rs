//! # AssessmentMetadata - Trait Implementations
//!
//! This module contains trait implementations for `AssessmentMetadata`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AssessmentMetadata;

impl Default for AssessmentMetadata {
    fn default() -> Self {
        Self {
            assessment_time: 0.0,
            assessment_duration: 0.0,
            original_duration: 0.0,
            cloned_duration: 0.0,
            sample_rate: 0,
            assessment_method: String::new(),
            quality_version: "1.0".to_string(),
        }
    }
}
