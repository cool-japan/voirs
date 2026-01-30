//! # PrecomputationConfig - Trait Implementations
//!
//! This module contains trait implementations for `PrecomputationConfig`.
//!
/// ## Implemented Traits
///
/// - `Default`
///
/// 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)
use super::types::PrecomputationConfig;
use super::types::*;

impl Default for PrecomputationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategies: vec![
                PrecomputationStrategy::VoiceLoad,
                PrecomputationStrategy::IdleTime,
            ],
            background_compute: true,
            memory_limit: 256 * 1024 * 1024,
            quality_level: ComputationQuality::Balanced,
            persistent_cache: true,
            adaptive: true,
        }
    }
}
