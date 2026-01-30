//! # VariantPreferences - Trait Implementations
//!
//! This module contains trait implementations for `VariantPreferences`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::structs::{FormalityLevel, VariantPreferences};
use std::collections::HashMap;

impl Default for VariantPreferences {
    fn default() -> Self {
        Self {
            accent: "general_american".to_string(),
            formality: FormalityLevel::Standard,
            speed_optimized: false,
            custom_pronunciations: HashMap::new(),
        }
    }
}
