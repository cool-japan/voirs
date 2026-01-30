//! # RealtimeAssessmentConfig - Trait Implementations
//!
//! This module contains trait implementations for `RealtimeAssessmentConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::RealtimeAssessmentConfig;

impl Default for RealtimeAssessmentConfig {
    fn default() -> Self {
        Self {
            enable_realtime: false,
            assessment_interval: 1.0,
            sliding_window_size: 3.0,
            quick_assessment_mode: true,
        }
    }
}
