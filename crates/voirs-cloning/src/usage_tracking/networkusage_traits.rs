//! # NetworkUsage - Trait Implementations
//!
//! This module contains trait implementations for `NetworkUsage`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for NetworkUsage {
    fn default() -> Self {
        NetworkUsage {
            bytes_uploaded: 0,
            bytes_downloaded: 0,
            requests_made: 0,
            bandwidth_peak_mbps: None,
        }
    }
}
