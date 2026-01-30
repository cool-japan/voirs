//! # ModelRegistry - Trait Implementations
//!
//! This module contains trait implementations for `ModelRegistry`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::structs::{ModelRegistry, RegistryMetadata};
use std::collections::HashMap;

impl Default for ModelRegistry {
    fn default() -> Self {
        Self {
            models: HashMap::new(),
            pipelines: HashMap::new(),
            metadata: RegistryMetadata {
                version: "1.0.0".to_string(),
                last_updated: chrono::Utc::now().to_rfc3339(),
                description: "VoiRS Neural TTS Model Registry".to_string(),
            },
        }
    }
}
