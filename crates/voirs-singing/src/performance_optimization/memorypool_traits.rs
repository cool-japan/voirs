//! # MemoryPool - Trait Implementations
//!
//! This module contains trait implementations for `MemoryPool`.
//!
/// ## Implemented Traits
///
/// - `Drop`
///
/// 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)
use super::types::MemoryPool;
use super::types::*;

/// Implement Drop to ensure proper cleanup of memory pool
impl Drop for MemoryPool {
    fn drop(&mut self) {
        self.cleanup();
        self.available_blocks.clear();
        self.allocated_blocks.clear();
    }
}
