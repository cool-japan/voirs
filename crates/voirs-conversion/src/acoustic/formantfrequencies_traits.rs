//! # FormantFrequencies - Trait Implementations
//!
//! This module contains trait implementations for `FormantFrequencies`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

#[cfg(feature = "acoustic-integration")]
impl Default for FormantFrequencies {
    fn default() -> Self {
        Self {
            f1: vec![700.0; 100],  // Default F1
            f2: vec![1220.0; 100], // Default F2
            f3: vec![2600.0; 100], // Default F3
            f4: vec![3500.0; 100], // Default F4
            bandwidths: vec![50.0; 100],
        }
    }
}

#[cfg(not(feature = "acoustic-integration"))]
impl Default for FormantFrequencies {
    fn default() -> Self {
        Self { placeholder: true }
    }
}
