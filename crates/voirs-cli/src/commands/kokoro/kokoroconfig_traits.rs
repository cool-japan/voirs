//! # KokoroConfig - Trait Implementations
//!
//! This module contains trait implementations for `KokoroConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::KokoroConfig;

impl Default for KokoroConfig {
    fn default() -> Self {
        Self {
            default_lang: Some("en-us".to_string()),
            default_voice: None,
            default_speed: Some(1.0),
            model_dir: None,
            espeak_path: None,
        }
    }
}

