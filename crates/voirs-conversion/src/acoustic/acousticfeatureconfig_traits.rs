//! # AcousticFeatureConfig - Trait Implementations
//!
//! This module contains trait implementations for `AcousticFeatureConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

#[cfg(feature = "acoustic-integration")]
impl Default for AcousticFeatureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            frame_size: 1024,
            hop_size: 512,
            window_type: WindowType::Hann,
            high_quality: true,
        }
    }
}
