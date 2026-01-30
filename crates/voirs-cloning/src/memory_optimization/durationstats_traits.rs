//! # DurationStats - Trait Implementations
//!
//! This module contains trait implementations for `DurationStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::time::Duration;

impl Default for DurationStats {
    fn default() -> Self {
        Self {
            min: Duration::from_secs(0),
            max: Duration::from_secs(0),
            avg: Duration::from_secs(0),
            total: Duration::from_secs(0),
            count: 0,
        }
    }
}
