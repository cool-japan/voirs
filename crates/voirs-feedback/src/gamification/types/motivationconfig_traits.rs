//! # `MotivationConfig` - Trait Implementations
//!
//! This module contains trait implementations for `MotivationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MotivationConfig;

impl Default for MotivationConfig {
    fn default() -> Self {
        Self {
            enable_burnout_monitoring: true,
            enable_interventions: true,
            enable_reengagement: true,
            motivation_check_interval: 24,
        }
    }
}
