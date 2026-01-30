//! # PhoneticFeatures - Trait Implementations
//!
//! This module contains trait implementations for `PhoneticFeatures`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;

use super::types::{MannerOfArticulation, PhoneticFeatures, PlaceOfArticulation};

impl Default for PhoneticFeatures {
    fn default() -> Self {
        Self::consonant(
            PlaceOfArticulation::Glottal,
            MannerOfArticulation::Approximant,
            false,
        )
    }
}
