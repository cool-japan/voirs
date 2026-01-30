//! # LeakDetectionConfig - Trait Implementations
//!
//! This module contains trait implementations for `LeakDetectionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::time::Duration;

impl Default for LeakDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            track_stack_traces: false,
            max_tracked_allocations: 10000,
            detection_interval: Duration::from_secs(60),
            leak_threshold: Duration::from_secs(300),
            auto_cleanup: false,
            pressure_alert_threshold: 0.85,
            enable_profiling: true,
        }
    }
}
