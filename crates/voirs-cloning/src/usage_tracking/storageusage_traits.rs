//! # StorageUsage - Trait Implementations
//!
//! This module contains trait implementations for `StorageUsage`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for StorageUsage {
    fn default() -> Self {
        StorageUsage {
            temporary_storage_mb: 0.0,
            persistent_storage_mb: 0.0,
            cache_storage_mb: 0.0,
            files_created: 0,
            files_deleted: 0,
        }
    }
}
