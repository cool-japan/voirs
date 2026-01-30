//! # VoiceCacheConfig - Trait Implementations
//!
//! This module contains trait implementations for `VoiceCacheConfig`.
//!
/// ## Implemented Traits
///
/// - `Default`
///
/// 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)
use super::types::VoiceCacheConfig;
use super::types::*;

impl Default for VoiceCacheConfig {
    fn default() -> Self {
        Self {
            max_voices: 32,
            max_memory: 512 * 1024 * 1024,
            eviction_policy: EvictionPolicy::LRU,
            preload_popular: true,
            background_preload: true,
            compression: CompressionConfig::default(),
            persistent_cache: false,
            cache_directory: None,
        }
    }
}
