//! # MemoryPoolSizes - Trait Implementations
//!
//! This module contains trait implementations for `MemoryPoolSizes`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for MemoryPoolSizes {
    fn default() -> Self {
        Self {
            embeddings_pool_size: 100,
            samples_pool_size: 50,
            audio_buffers_pool_size: 20,
            temp_vectors_pool_size: 200,
        }
    }
}
