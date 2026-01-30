//! # CompressionConfig - Trait Implementations
//!
//! This module contains trait implementations for `CompressionConfig`.
//!
/// ## Implemented Traits
///
/// - `Default`
///
/// 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)
use super::types::CompressionConfig;
use super::types::*;

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::LZ4,
            level: 6,
            preserve_quality: true,
            allow_lossy: false,
            target_ratio: 0.5,
            adaptive: true,
        }
    }
}
