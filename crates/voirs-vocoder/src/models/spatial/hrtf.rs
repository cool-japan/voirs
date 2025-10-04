//! HRTF (Head-Related Transfer Function) processor for spatial audio.

use crate::models::spatial::config::{DistanceModel, HrtfConfig, HrtfDatabase, HrtfInterpolation};
use crate::models::spatial::{BinauralOutput, SpatialPosition};
use anyhow::Result;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

/// HRTF processor for spatial audio localization
pub struct HrtfProcessor {
    /// Configuration
    config: HrtfConfig,
    /// HRTF database
    hrtf_database: HrtfDatabaseImpl,
    /// Sample rate
    sample_rate: u32,
    /// Filter length
    filter_length: usize,
}

/// HRTF database implementation
struct HrtfDatabaseImpl {
    /// Left ear HRTFs indexed by (azimuth, elevation)
    left_hrtfs: HashMap<(i32, i32), Array1<f32>>,
    /// Right ear HRTFs indexed by (azimuth, elevation)
    right_hrtfs: HashMap<(i32, i32), Array1<f32>>,
    /// Available azimuth angles
    azimuth_angles: Vec<i32>,
    /// Available elevation angles
    elevation_angles: Vec<i32>,
    /// Database type
    database_type: HrtfDatabase,
}

/// HRTF measurement point
#[derive(Debug, Clone)]
pub struct HrtfMeasurement {
    /// Azimuth angle (degrees)
    pub azimuth: i32,
    /// Elevation angle (degrees)
    pub elevation: i32,
    /// Left ear impulse response
    pub left_ir: Array1<f32>,
    /// Right ear impulse response
    pub right_ir: Array1<f32>,
}

impl HrtfProcessor {
    /// Create new HRTF processor
    pub fn new(config: &HrtfConfig) -> Result<Self> {
        let mut processor = Self {
            config: config.clone(),
            hrtf_database: HrtfDatabaseImpl::new(config.hrtf_database)?,
            sample_rate: 44100,
            filter_length: 128,
        };

        // Load HRTF database
        processor.load_hrtf_database()?;

        Ok(processor)
    }

    /// Process audio with HRTF
    pub fn process(&mut self, audio: &[f32], position: &SpatialPosition) -> Result<BinauralOutput> {
        if !self.config.enable_hrtf {
            // If HRTF is disabled, just duplicate mono to stereo
            return Ok(BinauralOutput {
                left: audio.to_vec(),
                right: audio.to_vec(),
            });
        }

        // Get HRTF filters for the given position
        let (left_hrtf, right_hrtf) = self.get_hrtf_filters(position)?;

        // Apply distance effects
        let distance_audio = self.apply_distance_effects(audio, position)?;

        // Convolve with HRTF filters
        let left_output = self.convolve(&distance_audio, &left_hrtf)?;
        let right_output = self.convolve(&distance_audio, &right_hrtf)?;

        Ok(BinauralOutput {
            left: left_output,
            right: right_output,
        })
    }

    /// Load HRTF database
    fn load_hrtf_database(&mut self) -> Result<()> {
        match self.config.hrtf_database {
            HrtfDatabase::MitKemar => self.load_mit_kemar_database(),
            HrtfDatabase::Cipic => self.load_cipic_database(),
            HrtfDatabase::Ari => self.load_ari_database(),
            HrtfDatabase::Generic => self.load_generic_database(),
        }
    }

    /// Load MIT KEMAR database
    fn load_mit_kemar_database(&mut self) -> Result<()> {
        // Generate synthetic HRTF data for MIT KEMAR
        // In a real implementation, this would load actual HRTF data
        for azimuth in (-180..=180).step_by(5) {
            for elevation in (-40..=90).step_by(5) {
                let left_ir = self.generate_synthetic_hrtf(azimuth, elevation, true)?;
                let right_ir = self.generate_synthetic_hrtf(azimuth, elevation, false)?;

                self.hrtf_database
                    .left_hrtfs
                    .insert((azimuth, elevation), left_ir);
                self.hrtf_database
                    .right_hrtfs
                    .insert((azimuth, elevation), right_ir);
            }
        }

        self.hrtf_database.azimuth_angles = (-180..=180).step_by(5).collect();
        self.hrtf_database.elevation_angles = (-40..=90).step_by(5).collect();

        Ok(())
    }

    /// Load CIPIC database
    fn load_cipic_database(&mut self) -> Result<()> {
        // Similar to MIT KEMAR but with different angular resolution
        for azimuth in (-80..=80).step_by(5) {
            for elevation in (-45..=230).step_by(5) {
                let left_ir = self.generate_synthetic_hrtf(azimuth, elevation, true)?;
                let right_ir = self.generate_synthetic_hrtf(azimuth, elevation, false)?;

                self.hrtf_database
                    .left_hrtfs
                    .insert((azimuth, elevation), left_ir);
                self.hrtf_database
                    .right_hrtfs
                    .insert((azimuth, elevation), right_ir);
            }
        }

        self.hrtf_database.azimuth_angles = (-80..=80).step_by(5).collect();
        self.hrtf_database.elevation_angles = (-45..=230).step_by(5).collect();

        Ok(())
    }

    /// Load ARI database
    fn load_ari_database(&mut self) -> Result<()> {
        // ARI database with high resolution
        for azimuth in (-180..=180).step_by(2) {
            for elevation in (-30..=80).step_by(2) {
                let left_ir = self.generate_synthetic_hrtf(azimuth, elevation, true)?;
                let right_ir = self.generate_synthetic_hrtf(azimuth, elevation, false)?;

                self.hrtf_database
                    .left_hrtfs
                    .insert((azimuth, elevation), left_ir);
                self.hrtf_database
                    .right_hrtfs
                    .insert((azimuth, elevation), right_ir);
            }
        }

        self.hrtf_database.azimuth_angles = (-180..=180).step_by(2).collect();
        self.hrtf_database.elevation_angles = (-30..=80).step_by(2).collect();

        Ok(())
    }

    /// Load generic database
    fn load_generic_database(&mut self) -> Result<()> {
        // Generic database with basic coverage
        for azimuth in (-180..=180).step_by(10) {
            for elevation in (-45..=90).step_by(10) {
                let left_ir = self.generate_synthetic_hrtf(azimuth, elevation, true)?;
                let right_ir = self.generate_synthetic_hrtf(azimuth, elevation, false)?;

                self.hrtf_database
                    .left_hrtfs
                    .insert((azimuth, elevation), left_ir);
                self.hrtf_database
                    .right_hrtfs
                    .insert((azimuth, elevation), right_ir);
            }
        }

        self.hrtf_database.azimuth_angles = (-180..=180).step_by(10).collect();
        self.hrtf_database.elevation_angles = (-45..=90).step_by(10).collect();

        Ok(())
    }

    /// Generate synthetic HRTF for testing
    fn generate_synthetic_hrtf(
        &self,
        azimuth: i32,
        elevation: i32,
        is_left: bool,
    ) -> Result<Array1<f32>> {
        let mut hrtf = Array1::zeros(self.filter_length);

        // Generate a synthetic HRTF based on simple acoustic principles
        let azimuth_rad = azimuth as f32 * std::f32::consts::PI / 180.0;
        let elevation_rad = elevation as f32 * std::f32::consts::PI / 180.0;

        // Simple delay and amplitude model
        let delay = if is_left {
            ((azimuth_rad + std::f32::consts::PI / 2.0).sin() * 0.5 + 0.5) * 10.0
        } else {
            ((azimuth_rad - std::f32::consts::PI / 2.0).sin() * 0.5 + 0.5) * 10.0
        };

        let amplitude = 1.0 / (1.0 + elevation_rad.abs() * 0.1);

        // Create impulse response
        let delay_samples = (delay * self.sample_rate as f32 / 1000.0) as usize;
        if delay_samples < self.filter_length {
            hrtf[delay_samples] = amplitude;

            // Add some decay
            for i in 1..10 {
                if delay_samples + i < self.filter_length {
                    hrtf[delay_samples + i] = amplitude * (-(i as f32) * 0.1).exp();
                }
            }
        }

        Ok(hrtf)
    }

    /// Get HRTF filters for a given position
    fn get_hrtf_filters(&self, position: &SpatialPosition) -> Result<(Array1<f32>, Array1<f32>)> {
        let azimuth = position.azimuth as i32;
        let elevation = position.elevation as i32;

        match self.config.interpolation_method {
            HrtfInterpolation::Nearest => self.get_nearest_hrtf(azimuth, elevation),
            HrtfInterpolation::Linear => self.get_linear_interpolated_hrtf(azimuth, elevation),
            HrtfInterpolation::Cubic => self.get_cubic_interpolated_hrtf(azimuth, elevation),
            HrtfInterpolation::Spherical => {
                self.get_spherical_interpolated_hrtf(azimuth, elevation)
            }
        }
    }

    /// Get nearest HRTF
    fn get_nearest_hrtf(&self, azimuth: i32, elevation: i32) -> Result<(Array1<f32>, Array1<f32>)> {
        // Find nearest angles in database
        let nearest_azimuth = self
            .hrtf_database
            .azimuth_angles
            .iter()
            .min_by_key(|&&a| (a - azimuth).abs())
            .copied()
            .unwrap_or(0);

        let nearest_elevation = self
            .hrtf_database
            .elevation_angles
            .iter()
            .min_by_key(|&&e| (e - elevation).abs())
            .copied()
            .unwrap_or(0);

        let left_hrtf = self
            .hrtf_database
            .left_hrtfs
            .get(&(nearest_azimuth, nearest_elevation))
            .cloned()
            .unwrap_or_else(|| Array1::zeros(self.filter_length));

        let right_hrtf = self
            .hrtf_database
            .right_hrtfs
            .get(&(nearest_azimuth, nearest_elevation))
            .cloned()
            .unwrap_or_else(|| Array1::zeros(self.filter_length));

        Ok((left_hrtf, right_hrtf))
    }

    /// Get linear interpolated HRTF
    fn get_linear_interpolated_hrtf(
        &self,
        azimuth: i32,
        elevation: i32,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        // For simplicity, fall back to nearest neighbor
        // In a real implementation, this would perform proper linear interpolation
        self.get_nearest_hrtf(azimuth, elevation)
    }

    /// Get cubic interpolated HRTF
    fn get_cubic_interpolated_hrtf(
        &self,
        azimuth: i32,
        elevation: i32,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        // For simplicity, fall back to nearest neighbor
        // In a real implementation, this would perform proper cubic interpolation
        self.get_nearest_hrtf(azimuth, elevation)
    }

    /// Get spherical interpolated HRTF
    fn get_spherical_interpolated_hrtf(
        &self,
        azimuth: i32,
        elevation: i32,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        // For simplicity, fall back to nearest neighbor
        // In a real implementation, this would perform proper spherical interpolation
        self.get_nearest_hrtf(azimuth, elevation)
    }

    /// Apply distance effects to audio
    fn apply_distance_effects(
        &self,
        audio: &[f32],
        position: &SpatialPosition,
    ) -> Result<Vec<f32>> {
        let mut output = audio.to_vec();

        match self.config.distance_model {
            DistanceModel::None => {}
            DistanceModel::Linear => {
                let attenuation = 1.0 - (position.distance - 1.0).max(0.0) * 0.1;
                for sample in &mut output {
                    *sample *= attenuation.clamp(0.0, 1.0);
                }
            }
            DistanceModel::Inverse => {
                let attenuation = 1.0 / (1.0 + position.distance);
                for sample in &mut output {
                    *sample *= attenuation;
                }
            }
            DistanceModel::InverseSquare => {
                let attenuation = 1.0 / (1.0 + position.distance * position.distance);
                for sample in &mut output {
                    *sample *= attenuation;
                }
            }
            DistanceModel::Exponential => {
                let attenuation = (-position.distance * 0.1).exp();
                for sample in &mut output {
                    *sample *= attenuation;
                }
            }
        }

        Ok(output)
    }

    /// Convolve audio with HRTF filter
    fn convolve(&self, audio: &[f32], hrtf: &Array1<f32>) -> Result<Vec<f32>> {
        let audio_len = audio.len();
        let hrtf_len = hrtf.len();
        let output_len = audio_len + hrtf_len - 1;
        let mut output = vec![0.0; output_len];

        // Simple convolution implementation
        for i in 0..audio_len {
            for j in 0..hrtf_len {
                if i + j < output_len {
                    output[i + j] += audio[i] * hrtf[j];
                }
            }
        }

        // Truncate to original length
        output.truncate(audio_len);

        Ok(output)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &HrtfConfig) -> Result<()> {
        self.config = config.clone();

        // Reload database if it changed
        if self.hrtf_database.database_type != config.hrtf_database {
            self.hrtf_database = HrtfDatabaseImpl::new(config.hrtf_database)?;
            self.load_hrtf_database()?;
        }

        Ok(())
    }

    /// Get available measurement points
    pub fn get_measurement_points(&self) -> Vec<(i32, i32)> {
        let mut points = Vec::new();
        for &azimuth in &self.hrtf_database.azimuth_angles {
            for &elevation in &self.hrtf_database.elevation_angles {
                points.push((azimuth, elevation));
            }
        }
        points
    }

    /// Get HRTF measurement for specific angles
    pub fn get_hrtf_measurement(&self, azimuth: i32, elevation: i32) -> Option<HrtfMeasurement> {
        let left_ir = self.hrtf_database.left_hrtfs.get(&(azimuth, elevation))?;
        let right_ir = self.hrtf_database.right_hrtfs.get(&(azimuth, elevation))?;

        Some(HrtfMeasurement {
            azimuth,
            elevation,
            left_ir: left_ir.clone(),
            right_ir: right_ir.clone(),
        })
    }
}

impl HrtfDatabaseImpl {
    /// Create new HRTF database
    fn new(database_type: HrtfDatabase) -> Result<Self> {
        Ok(Self {
            left_hrtfs: HashMap::new(),
            right_hrtfs: HashMap::new(),
            azimuth_angles: Vec::new(),
            elevation_angles: Vec::new(),
            database_type,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::spatial::config::HrtfConfig;

    #[test]
    fn test_hrtf_processor_creation() {
        let config = HrtfConfig::default();
        let processor = HrtfProcessor::new(&config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_hrtf_processing() {
        let config = HrtfConfig::default();
        let mut processor = HrtfProcessor::new(&config).unwrap();

        let audio = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let position = SpatialPosition::default();

        let result = processor.process(&audio, &position);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.left.len(), audio.len());
        assert_eq!(output.right.len(), audio.len());
    }

    #[test]
    fn test_hrtf_disabled() {
        let config = HrtfConfig {
            enable_hrtf: false,
            ..Default::default()
        };

        let mut processor = HrtfProcessor::new(&config).unwrap();

        let audio = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let position = SpatialPosition::default();

        let result = processor.process(&audio, &position);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.left, audio);
        assert_eq!(output.right, audio);
    }

    #[test]
    fn test_distance_effects() {
        let config = HrtfConfig::default();
        let processor = HrtfProcessor::new(&config).unwrap();

        let audio = vec![1.0, 1.0, 1.0];
        let position = SpatialPosition {
            distance: 2.0,
            ..Default::default()
        };

        let result = processor.apply_distance_effects(&audio, &position);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output[0] < 1.0); // Should be attenuated
    }

    #[test]
    fn test_convolution() {
        let config = HrtfConfig::default();
        let processor = HrtfProcessor::new(&config).unwrap();

        let audio = vec![1.0, 0.0, 0.0, 0.0];
        let hrtf = Array1::from_vec(vec![0.5, 0.3, 0.1]);

        let result = processor.convolve(&audio, &hrtf);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), audio.len());
        assert_eq!(output[0], 0.5);
    }

    #[test]
    fn test_different_databases() {
        let databases = vec![
            HrtfDatabase::MitKemar,
            HrtfDatabase::Cipic,
            HrtfDatabase::Ari,
            HrtfDatabase::Generic,
        ];

        for database in databases {
            let config = HrtfConfig {
                hrtf_database: database,
                ..Default::default()
            };

            let processor = HrtfProcessor::new(&config);
            assert!(processor.is_ok());
        }
    }

    #[test]
    fn test_measurement_points() {
        let config = HrtfConfig::default();
        let processor = HrtfProcessor::new(&config).unwrap();

        let points = processor.get_measurement_points();
        assert!(!points.is_empty());
    }

    #[test]
    fn test_hrtf_measurement() {
        let config = HrtfConfig::default();
        let processor = HrtfProcessor::new(&config).unwrap();

        let measurement = processor.get_hrtf_measurement(0, 0);
        assert!(measurement.is_some());

        let measurement = measurement.unwrap();
        assert_eq!(measurement.azimuth, 0);
        assert_eq!(measurement.elevation, 0);
        assert!(!measurement.left_ir.is_empty());
        assert!(!measurement.right_ir.is_empty());
    }

    #[test]
    fn test_config_update() {
        let config = HrtfConfig::default();
        let mut processor = HrtfProcessor::new(&config).unwrap();

        let mut new_config = config.clone();
        new_config.hrtf_database = HrtfDatabase::Generic;

        let result = processor.update_config(&new_config);
        assert!(result.is_ok());
    }
}
