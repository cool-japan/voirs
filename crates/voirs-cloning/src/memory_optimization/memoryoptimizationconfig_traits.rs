//! # MemoryOptimizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `MemoryOptimizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::time::Duration;

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_optimization: true,
            max_memory_usage: 512 * 1024 * 1024,
            enable_garbage_collection: true,
            gc_interval: Duration::from_secs(30),
            memory_pressure_threshold: 0.8,
            enable_memory_pool: true,
            pool_initial_sizes: MemoryPoolSizes::default(),
            enable_compression: true,
            compression_quality: 0.7,
            enable_lazy_loading: true,
            cache_limits: CacheLimits::default(),
            enable_mobile_optimizations: true,
            enable_memory_mapping: false,
        }
    }
}
