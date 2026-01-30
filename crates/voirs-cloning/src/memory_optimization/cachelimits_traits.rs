//! # CacheLimits - Trait Implementations
//!
//! This module contains trait implementations for `CacheLimits`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for CacheLimits {
    fn default() -> Self {
        Self {
            max_speaker_profiles: 100,
            max_embeddings: 200,
            max_voice_samples: 50,
            max_quality_assessments: 100,
        }
    }
}
