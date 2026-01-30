//! # VocoderConfig - Trait Implementations
//!
//! This module contains trait implementations for `VocoderConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::VocoderConfig;

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            hop_length: 256,
            enable_gpu: false,
            batch_size: 1,
        }
    }
}

