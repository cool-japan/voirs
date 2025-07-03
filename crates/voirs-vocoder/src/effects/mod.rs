//! Audio effects and enhancement system for voirs-vocoder.
//!
//! This module provides comprehensive audio post-processing capabilities
//! including dynamic range control, frequency processing, and spatial effects.

pub mod dynamics;
pub mod frequency;
pub mod spatial;
pub mod chain;
pub mod validation;

pub use dynamics::*;
pub use frequency::*;
pub use spatial::*;
pub use chain::*;
pub use validation::*;

use crate::{AudioBuffer, Result};

/// Common interface for all audio effects
pub trait AudioEffect: Send + Sync {
    /// Process audio buffer in-place
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()>;
    
    /// Process audio buffer and return new buffer
    fn process_copy(&mut self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        let mut output = audio.clone();
        self.process(&mut output)?;
        Ok(output)
    }
    
    /// Get effect name for debugging/logging
    fn name(&self) -> &'static str;
    
    /// Reset internal state
    fn reset(&mut self);
    
    /// Check if effect is enabled
    fn is_enabled(&self) -> bool;
    
    /// Enable/disable effect
    fn set_enabled(&mut self, enabled: bool);
}

/// Effect parameter that can be automated
#[derive(Debug, Clone)]
pub struct EffectParameter {
    pub name: String,
    pub value: f32,
    pub min_value: f32,
    pub max_value: f32,
    pub default_value: f32,
}

impl EffectParameter {
    pub fn new(name: &str, default: f32, min: f32, max: f32) -> Self {
        Self {
            name: name.to_string(),
            value: default,
            min_value: min,
            max_value: max,
            default_value: default,
        }
    }
    
    pub fn set_value(&mut self, value: f32) {
        self.value = value.clamp(self.min_value, self.max_value);
    }
    
    pub fn set_normalized(&mut self, normalized: f32) {
        let normalized = normalized.clamp(0.0, 1.0);
        self.value = self.min_value + normalized * (self.max_value - self.min_value);
    }
    
    pub fn get_normalized(&self) -> f32 {
        if self.max_value == self.min_value {
            0.0
        } else {
            (self.value - self.min_value) / (self.max_value - self.min_value)
        }
    }
}