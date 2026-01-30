//! # MemoryUsage - Trait Implementations
//!
//! This module contains trait implementations for `MemoryUsage`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for MemoryUsage {
    fn default() -> Self {
        MemoryUsage {
            peak_memory_mb: 0.0,
            average_memory_mb: 0.0,
            memory_allocated_mb: 0.0,
            memory_freed_mb: 0.0,
        }
    }
}
