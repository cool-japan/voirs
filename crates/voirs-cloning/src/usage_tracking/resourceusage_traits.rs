//! # ResourceUsage - Trait Implementations
//!
//! This module contains trait implementations for `ResourceUsage`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for ResourceUsage {
    fn default() -> Self {
        ResourceUsage {
            total_processing_time_ms: 0,
            cpu_usage: CpuUsage::default(),
            memory_usage: MemoryUsage::default(),
            gpu_usage: None,
            network_usage: NetworkUsage::default(),
            storage_usage: StorageUsage::default(),
            cost_estimate: None,
        }
    }
}
