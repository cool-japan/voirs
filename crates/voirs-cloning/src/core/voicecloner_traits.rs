//! # VoiceCloner - Trait Implementations
//!
//! This module contains trait implementations for `VoiceCloner`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for VoiceCloner {
    fn default() -> Self {
        Self::new().expect("Failed to create default VoiceCloner")
    }
}
