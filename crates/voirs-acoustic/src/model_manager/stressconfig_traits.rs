//! # StressConfig - Trait Implementations
//!
//! This module contains trait implementations for `StressConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::structs::StressConfig;

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            predict_stress: true,
            primary_stress_marker: "1".to_string(),
            secondary_stress_marker: "2".to_string(),
            confidence_threshold: 0.7,
        }
    }
}
