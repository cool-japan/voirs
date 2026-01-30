//! # EmotionalState - Trait Implementations
//!
//! This module contains trait implementations for `EmotionalState`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;

use super::types::EmotionalState;

impl Default for EmotionalState {
    fn default() -> Self {
        Self::Neutral
    }
}
