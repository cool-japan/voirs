//! Room acoustics simulator for spatial audio.

use crate::models::spatial::config::AcousticsConfig;
use anyhow::Result;

/// Room acoustics simulator
pub struct AcousticsSimulator {
    /// Configuration
    config: AcousticsConfig,
    /// Current reverb level
    reverb_level: f32,
}

impl AcousticsSimulator {
    /// Create new acoustics simulator
    pub fn new(config: &AcousticsConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            reverb_level: 0.0,
        })
    }

    /// Process audio with room acoustics
    pub fn process(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        if !self.config.enable_acoustics {
            return Ok(audio.to_vec());
        }

        let mut output = audio.to_vec();

        // Apply early reflections
        if self
            .config
            .early_reflections_config
            .enable_early_reflections
        {
            output = self.apply_early_reflections(&output)?;
        }

        // Apply reverb
        if self.config.reverb_config.enable_reverb {
            output = self.apply_reverb(&output)?;
        }

        // Apply air absorption
        if self.config.air_absorption_config.enable_air_absorption {
            output = self.apply_air_absorption(&output)?;
        }

        Ok(output)
    }

    /// Apply early reflections
    fn apply_early_reflections(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut output = audio.to_vec();
        let reflection_level = self.config.early_reflections_config.reflection_level;

        // Simple early reflections simulation
        for i in 0..audio.len() {
            if i > 100 {
                // Simple delay
                output[i] += audio[i - 100] * reflection_level * 0.5;
            }
        }

        Ok(output)
    }

    /// Apply reverb
    fn apply_reverb(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut output = audio.to_vec();
        let reverb_level = self.config.reverb_config.reverb_level;
        self.reverb_level = reverb_level;

        // Simple reverb simulation
        for i in 0..audio.len() {
            if i > 1000 {
                // Simple delay
                output[i] += audio[i - 1000] * reverb_level * 0.3;
            }
        }

        Ok(output)
    }

    /// Apply air absorption
    fn apply_air_absorption(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut output = audio.to_vec();
        let hf_rolloff = self.config.air_absorption_config.hf_rolloff;

        // Simple high-frequency attenuation
        for sample in &mut output {
            *sample *= 1.0 - hf_rolloff;
        }

        Ok(output)
    }

    /// Get current reverb level
    pub fn get_reverb_level(&self) -> f32 {
        self.reverb_level
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &AcousticsConfig) -> Result<()> {
        self.config = config.clone();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::spatial::config::AcousticsConfig;

    #[test]
    fn test_acoustics_simulator_creation() {
        let config = AcousticsConfig::default();
        let simulator = AcousticsSimulator::new(&config);
        assert!(simulator.is_ok());
    }

    #[test]
    fn test_acoustics_processing() {
        let config = AcousticsConfig::default();
        let mut simulator = AcousticsSimulator::new(&config).unwrap();

        let audio = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = simulator.process(&audio);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), audio.len());
    }
}
