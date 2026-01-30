//! # CpuUsage - Trait Implementations
//!
//! This module contains trait implementations for `CpuUsage`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for CpuUsage {
    fn default() -> Self {
        CpuUsage {
            peak_cpu_percent: 0.0,
            average_cpu_percent: 0.0,
            cpu_time_seconds: 0.0,
            cpu_cores_used: 1,
        }
    }
}
