//! # PerformanceMonitor - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceMonitor`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;

use super::types::PerformanceMonitor;

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}
