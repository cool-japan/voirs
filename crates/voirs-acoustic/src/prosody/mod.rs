//! Prosody control for natural speech synthesis.

pub mod duration;
pub mod pitch;
pub mod energy;

pub use duration::{DurationConfig, PauseDurations, RhythmPattern, DurationContext};
pub use pitch::{PitchConfig, IntonationPattern, VibratoConfig, PitchContext};
pub use energy::{EnergyConfig, EnergyContourPattern, VoiceQualityConfig, EnergyContext};

use serde::{Deserialize, Serialize};
use crate::Result;

/// Comprehensive prosody configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodyConfig {
    /// Duration control settings
    pub duration: DurationConfig,
    /// Pitch control settings
    pub pitch: PitchConfig,
    /// Energy control settings
    pub energy: EnergyConfig,
    /// Global prosody strength (0.0 - 1.0)
    pub strength: f32,
    /// Natural variation amount (0.0 - 1.0)
    pub variation: f32,
}

impl ProsodyConfig {
    /// Create new prosody config
    pub fn new() -> Self {
        Self {
            duration: DurationConfig::default(),
            pitch: PitchConfig::default(),
            energy: EnergyConfig::default(),
            strength: 1.0,
            variation: 0.1,
        }
    }
    
    /// Set duration config
    pub fn with_duration(mut self, duration: DurationConfig) -> Self {
        self.duration = duration;
        self
    }
    
    /// Set pitch config
    pub fn with_pitch(mut self, pitch: PitchConfig) -> Self {
        self.pitch = pitch;
        self
    }
    
    /// Set energy config
    pub fn with_energy(mut self, energy: EnergyConfig) -> Self {
        self.energy = energy;
        self
    }
    
    /// Set prosody strength
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }
    
    /// Set natural variation
    pub fn with_variation(mut self, variation: f32) -> Self {
        self.variation = variation.clamp(0.0, 1.0);
        self
    }
    
    /// Apply prosody adjustments to phoneme sequence
    pub fn apply_to_phonemes(&self, phonemes: &mut [crate::Phoneme]) -> Result<()> {
        // Apply duration adjustments
        self.duration.apply_to_phonemes(phonemes)?;
        
        // Note: Pitch and energy are typically applied during synthesis
        // as they affect the acoustic features rather than phoneme timing
        
        Ok(())
    }
    
    /// Validate prosody configuration
    pub fn validate(&self) -> Result<()> {
        self.duration.validate()?;
        self.pitch.validate()?;
        self.energy.validate()?;
        
        if !(0.0..=1.0).contains(&self.strength) {
            return Err(crate::AcousticError::ConfigError(
                format!("Prosody strength must be between 0.0 and 1.0, got {}", self.strength)
            ));
        }
        
        if !(0.0..=1.0).contains(&self.variation) {
            return Err(crate::AcousticError::ConfigError(
                format!("Prosody variation must be between 0.0 and 1.0, got {}", self.variation)
            ));
        }
        
        Ok(())
    }
    
    /// Create preset prosody configurations
    pub fn preset_natural() -> Self {
        Self::new()
            .with_duration(DurationConfig::natural())
            .with_pitch(PitchConfig::natural())
            .with_energy(EnergyConfig::natural())
            .with_variation(0.15)
    }
    
    pub fn preset_expressive() -> Self {
        Self::new()
            .with_duration(DurationConfig::expressive())
            .with_pitch(PitchConfig::expressive())
            .with_energy(EnergyConfig::expressive())
            .with_variation(0.25)
    }
    
    pub fn preset_monotone() -> Self {
        Self::new()
            .with_duration(DurationConfig::uniform())
            .with_pitch(PitchConfig::flat())
            .with_energy(EnergyConfig::uniform())
            .with_variation(0.05)
    }
    
    pub fn preset_fast() -> Self {
        Self::new()
            .with_duration(DurationConfig::fast())
            .with_pitch(PitchConfig::default())
            .with_energy(EnergyConfig::default())
    }
    
    pub fn preset_slow() -> Self {
        Self::new()
            .with_duration(DurationConfig::slow())
            .with_pitch(PitchConfig::default())
            .with_energy(EnergyConfig::default())
    }
}

impl Default for ProsodyConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Prosody controller for managing dynamic adjustments
#[derive(Debug, Clone)]
pub struct ProsodyController {
    /// Base prosody configuration
    base_config: ProsodyConfig,
    /// Current prosody state
    current_config: ProsodyConfig,
    /// Adjustment history
    adjustment_history: Vec<ProsodyConfig>,
    /// Maximum history length
    max_history: usize,
}

impl ProsodyController {
    /// Create new prosody controller
    pub fn new(base_config: ProsodyConfig) -> Self {
        Self {
            base_config: base_config.clone(),
            current_config: base_config,
            adjustment_history: Vec::new(),
            max_history: 20,
        }
    }
    
    /// Get current prosody configuration
    pub fn get_current_config(&self) -> &ProsodyConfig {
        &self.current_config
    }
    
    /// Set new prosody configuration
    pub fn set_config(&mut self, config: ProsodyConfig) {
        self.adjustment_history.push(self.current_config.clone());
        
        // Trim history if needed
        if self.adjustment_history.len() > self.max_history {
            self.adjustment_history.remove(0);
        }
        
        self.current_config = config;
    }
    
    /// Apply temporary adjustment
    pub fn apply_adjustment(&mut self, adjustment: ProsodyAdjustment) {
        let mut adjusted_config = self.current_config.clone();
        
        match adjustment {
            ProsodyAdjustment::Speed(factor) => {
                adjusted_config.duration.speed_factor *= factor;
            },
            ProsodyAdjustment::Pitch(shift) => {
                adjusted_config.pitch.base_frequency += shift;
            },
            ProsodyAdjustment::Energy(factor) => {
                adjusted_config.energy.base_energy *= factor;
            },
            ProsodyAdjustment::Variation(amount) => {
                adjusted_config.variation = (adjusted_config.variation + amount).clamp(0.0, 1.0);
            },
        }
        
        self.set_config(adjusted_config);
    }
    
    /// Reset to base configuration
    pub fn reset_to_base(&mut self) {
        self.set_config(self.base_config.clone());
    }
    
    /// Get adjustment history
    pub fn get_adjustment_history(&self) -> &[ProsodyConfig] {
        &self.adjustment_history
    }
}

/// Prosody adjustment types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProsodyAdjustment {
    /// Speed adjustment (factor)
    Speed(f32),
    /// Pitch adjustment (Hz shift)
    Pitch(f32),
    /// Energy adjustment (factor)
    Energy(f32),
    /// Variation adjustment (amount)
    Variation(f32),
}

impl Default for ProsodyController {
    fn default() -> Self {
        Self::new(ProsodyConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prosody_config_creation() {
        let config = ProsodyConfig::new();
        assert_eq!(config.strength, 1.0);
        assert_eq!(config.variation, 0.1);
    }
    
    #[test]
    fn test_prosody_config_builder() {
        let config = ProsodyConfig::new()
            .with_strength(0.8)
            .with_variation(0.2);
        
        assert_eq!(config.strength, 0.8);
        assert_eq!(config.variation, 0.2);
    }
    
    #[test]
    fn test_prosody_config_validation() {
        let valid_config = ProsodyConfig::new()
            .with_strength(0.5)
            .with_variation(0.3);
        assert!(valid_config.validate().is_ok());
        
        let mut invalid_config = ProsodyConfig::new();
        invalid_config.strength = 1.5; // Invalid - bypass builder validation
        assert!(invalid_config.validate().is_err());
    }
    
    #[test]
    fn test_prosody_presets() {
        let natural = ProsodyConfig::preset_natural();
        assert_eq!(natural.variation, 0.15);
        
        let expressive = ProsodyConfig::preset_expressive();
        assert_eq!(expressive.variation, 0.25);
        
        let monotone = ProsodyConfig::preset_monotone();
        assert_eq!(monotone.variation, 0.05);
    }
    
    #[test]
    fn test_prosody_controller() {
        let base_config = ProsodyConfig::preset_natural();
        let mut controller = ProsodyController::new(base_config.clone());
        
        assert_eq!(controller.get_current_config().variation, 0.15);
        
        // Apply adjustment
        controller.apply_adjustment(ProsodyAdjustment::Speed(1.2));
        assert_eq!(controller.get_adjustment_history().len(), 1);
        
        // Reset
        controller.reset_to_base();
        assert_eq!(controller.get_current_config().variation, 0.15);
    }
}