//! # AcousticConversionAdapter - Trait Implementations
//!
//! This module contains trait implementations for `AcousticConversionAdapter`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

#[cfg(not(feature = "acoustic-integration"))]
impl Default for AcousticConversionAdapter {
    fn default() -> Self {
        Self::new()
    }
}
