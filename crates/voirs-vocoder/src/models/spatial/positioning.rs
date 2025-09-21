//! 3D positioning system for spatial audio.

use crate::models::spatial::config::PositioningConfig;
use crate::models::spatial::SpatialPosition;
use anyhow::Result;

/// 3D positioning system for spatial audio
pub struct PositioningSystem {
    /// Configuration
    config: PositioningConfig,
}

impl PositioningSystem {
    /// Create new positioning system
    pub fn new(config: &PositioningConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Position audio in 3D space
    pub fn position_audio(
        &mut self,
        audio: &[f32],
        position: &SpatialPosition,
    ) -> Result<Vec<f32>> {
        if !self.config.enable_positioning {
            return Ok(audio.to_vec());
        }

        // Apply distance attenuation
        let distance_audio = self.apply_distance_attenuation(audio, position)?;

        // Apply Doppler effect if enabled
        let doppler_audio = if self.config.doppler_config.enable_doppler {
            self.apply_doppler_effect(&distance_audio, position)?
        } else {
            distance_audio
        };

        Ok(doppler_audio)
    }

    /// Apply distance attenuation
    fn apply_distance_attenuation(
        &self,
        audio: &[f32],
        position: &SpatialPosition,
    ) -> Result<Vec<f32>> {
        let mut output = audio.to_vec();
        let distance = position
            .distance
            .max(self.config.min_distance)
            .min(self.config.max_distance);

        let attenuation = match self.config.attenuation_model {
            crate::models::spatial::config::AttenuationModel::None => 1.0,
            crate::models::spatial::config::AttenuationModel::Linear => {
                1.0 - (distance - self.config.min_distance)
                    / (self.config.max_distance - self.config.min_distance)
            }
            crate::models::spatial::config::AttenuationModel::InverseDistance => 1.0 / distance,
            crate::models::spatial::config::AttenuationModel::InverseSquare => {
                1.0 / (distance * distance)
            }
            crate::models::spatial::config::AttenuationModel::Exponential => {
                (-distance * 0.1).exp()
            }
        };

        for sample in &mut output {
            *sample *= attenuation;
        }

        Ok(output)
    }

    /// Apply Doppler effect
    fn apply_doppler_effect(&self, audio: &[f32], _position: &SpatialPosition) -> Result<Vec<f32>> {
        // Simple implementation - just return audio unchanged for now
        // In a real implementation, this would calculate velocity and apply frequency shifts
        Ok(audio.to_vec())
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &PositioningConfig) -> Result<()> {
        self.config = config.clone();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::spatial::config::PositioningConfig;

    #[test]
    fn test_positioning_system_creation() {
        let config = PositioningConfig::default();
        let system = PositioningSystem::new(&config);
        assert!(system.is_ok());
    }

    #[test]
    fn test_position_audio() {
        let config = PositioningConfig::default();
        let mut system = PositioningSystem::new(&config).unwrap();

        let audio = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let position = SpatialPosition::default();

        let result = system.position_audio(&audio, &position);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), audio.len());
    }
}
