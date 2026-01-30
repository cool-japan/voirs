//! # PipelineSettings - Trait Implementations
//!
//! This module contains trait implementations for `PipelineSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::structs::{G2pConfig, PipelineSettings};

impl Default for PipelineSettings {
    fn default() -> Self {
        Self {
            real_time_factor: 0.3,
            quality_level: 0.8,
            memory_usage: 0.5,
            supports_streaming: true,
            g2p_config: G2pConfig::default(),
        }
    }
}
