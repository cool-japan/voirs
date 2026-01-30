//! # StreamingConfig - Trait Implementations
//!
//! This module contains trait implementations for `StreamingConfig`.
//!
/// ## Implemented Traits
///
/// - `Default`
///
/// 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)
use super::types::StreamingConfig;
use super::types::*;

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 4096,
            buffer_count: 8,
            lookahead_time: 2.0,
            quality: StreamingQuality::Balanced,
            adaptive_buffering: true,
            background_loading: true,
            memory_limit: 64 * 1024 * 1024,
        }
    }
}
