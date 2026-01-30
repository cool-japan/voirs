//! # AcousticState - Trait Implementations
//!
//! This module contains trait implementations for `AcousticState`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

#[cfg(feature = "acoustic-integration")]
impl Default for AcousticState {
    fn default() -> Self {
        Self {
            last_f0: 150.0,
            last_formants: (700.0, 1220.0, 2600.0),
            last_energy: 0.5,
            phase_accumulator: 0.0,
        }
    }
}
