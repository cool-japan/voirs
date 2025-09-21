//! Configuration for spatial audio processing

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for spatial audio processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialConfig {
    /// HRTF database path
    pub hrtf_database_path: Option<PathBuf>,
    /// Sample rate for processing
    pub sample_rate: u32,
    /// Buffer size for processing
    pub buffer_size: usize,
    /// Room size for acoustics simulation
    pub room_dimensions: (f32, f32, f32),
    /// Reverberation time (RT60) in seconds
    pub reverb_time: f32,
    /// Distance attenuation enabled
    pub enable_distance_attenuation: bool,
    /// Air absorption enabled
    pub enable_air_absorption: bool,
    /// Doppler effect enabled
    pub enable_doppler: bool,
    /// Maximum processing distance
    pub max_distance: f32,
    /// Speed of sound (m/s)
    pub speed_of_sound: f32,
    /// Quality level (0.0 = lowest, 1.0 = highest)
    pub quality_level: f32,
    /// Maximum concurrent audio sources
    pub max_sources: usize,
    /// Enable GPU acceleration if available
    pub use_gpu: bool,
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            hrtf_database_path: None,
            sample_rate: 44100,
            buffer_size: 1024,
            room_dimensions: (10.0, 8.0, 3.0), // meters
            reverb_time: 1.2,                  // seconds
            enable_distance_attenuation: true,
            enable_air_absorption: true,
            enable_doppler: false,
            max_distance: 100.0,   // meters
            speed_of_sound: 343.0, // m/s at 20°C
            quality_level: 0.8,    // High quality by default
            max_sources: 16,       // Reasonable default
            use_gpu: false,        // Conservative default
        }
    }
}

impl SpatialConfig {
    /// Validate configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.sample_rate == 0 {
            return Err(crate::Error::LegacyConfig(
                "Sample rate must be positive".to_string(),
            ));
        }
        if self.buffer_size == 0 {
            return Err(crate::Error::LegacyConfig(
                "Buffer size must be positive".to_string(),
            ));
        }
        if self.reverb_time < 0.0 {
            return Err(crate::Error::LegacyConfig(
                "Reverb time cannot be negative".to_string(),
            ));
        }
        if self.max_distance <= 0.0 {
            return Err(crate::Error::LegacyConfig(
                "Max distance must be positive".to_string(),
            ));
        }
        if self.speed_of_sound <= 0.0 {
            return Err(crate::Error::LegacyConfig(
                "Speed of sound must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Builder for SpatialConfig
#[derive(Debug, Default)]
pub struct SpatialConfigBuilder {
    config: SpatialConfig,
}

impl SpatialConfigBuilder {
    /// Create new config builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set HRTF database path
    pub fn hrtf_database_path(mut self, path: PathBuf) -> Self {
        self.config.hrtf_database_path = Some(path);
        self
    }

    /// Set sample rate
    pub fn sample_rate(mut self, sample_rate: u32) -> Self {
        self.config.sample_rate = sample_rate;
        self
    }

    /// Set buffer size
    pub fn buffer_size(mut self, buffer_size: usize) -> Self {
        self.config.buffer_size = buffer_size;
        self
    }

    /// Set room dimensions
    pub fn room_dimensions(mut self, width: f32, height: f32, depth: f32) -> Self {
        self.config.room_dimensions = (width, height, depth);
        self
    }

    /// Set reverb time
    pub fn reverb_time(mut self, reverb_time: f32) -> Self {
        self.config.reverb_time = reverb_time;
        self
    }

    /// Enable/disable distance attenuation
    pub fn distance_attenuation(mut self, enabled: bool) -> Self {
        self.config.enable_distance_attenuation = enabled;
        self
    }

    /// Enable/disable air absorption
    pub fn air_absorption(mut self, enabled: bool) -> Self {
        self.config.enable_air_absorption = enabled;
        self
    }

    /// Enable/disable Doppler effect
    pub fn doppler(mut self, enabled: bool) -> Self {
        self.config.enable_doppler = enabled;
        self
    }

    /// Set maximum processing distance
    pub fn max_distance(mut self, max_distance: f32) -> Self {
        self.config.max_distance = max_distance;
        self
    }

    /// Set speed of sound
    pub fn speed_of_sound(mut self, speed: f32) -> Self {
        self.config.speed_of_sound = speed;
        self
    }

    /// Build the configuration
    pub fn build(self) -> crate::Result<SpatialConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SpatialConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = SpatialConfigBuilder::new()
            .sample_rate(48000)
            .buffer_size(512)
            .reverb_time(1.5)
            .build()
            .unwrap();

        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.buffer_size, 512);
        assert_eq!(config.reverb_time, 1.5);
    }

    #[test]
    fn test_config_validation() {
        let mut config = SpatialConfig::default();
        config.sample_rate = 0;
        assert!(config.validate().is_err());

        config.sample_rate = 44100;
        config.max_distance = -1.0;
        assert!(config.validate().is_err());
    }
}
