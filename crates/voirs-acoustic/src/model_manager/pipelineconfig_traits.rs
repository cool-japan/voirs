//! # PipelineConfig - Trait Implementations
//!
//! This module contains trait implementations for `PipelineConfig`.
//!
//! ## Implemented Traits
//!
//! - `TryFrom`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::structs::{PipelineConfig, PipelineSettings};
use crate::{AcousticError, Result};

impl TryFrom<toml::Value> for PipelineConfig {
    type Error = AcousticError;
    fn try_from(value: toml::Value) -> Result<Self> {
        let table = value.as_table().ok_or_else(|| AcousticError::ConfigError {
            message: "Expected table for pipeline config".to_string(),
        })?;
        let name = table
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("Unnamed Pipeline")
            .to_string();
        let description = table
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("No description")
            .to_string();
        let acoustic_model = table
            .get("acoustic_model")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AcousticError::ConfigError {
                message: "Missing acoustic_model".to_string(),
            })?
            .to_string();
        let vocoder = table
            .get("vocoder")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AcousticError::ConfigError {
                message: "Missing vocoder".to_string(),
            })?
            .to_string();
        Ok(Self {
            name,
            description,
            acoustic_model,
            vocoder,
            settings: PipelineSettings::default(),
        })
    }
}
