//! Binaural renderer for spatial audio.

use crate::models::spatial::config::BinauralConfig;
use crate::models::spatial::{BinauralOutput, SpatialPosition};
use anyhow::Result;

/// Binaural renderer for stereo spatialization
pub struct BinauralRenderer {
    /// Configuration
    config: BinauralConfig,
    /// Sample rate
    sample_rate: u32,
}

impl BinauralRenderer {
    /// Create new binaural renderer
    pub fn new(config: &BinauralConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            sample_rate: 44100,
        })
    }

    /// Render binaural audio from processed input
    pub fn render(&mut self, audio: &[f32], position: &SpatialPosition) -> Result<BinauralOutput> {
        if !self.config.enable_binaural {
            return Ok(BinauralOutput {
                left: audio.to_vec(),
                right: audio.to_vec(),
            });
        }

        // Apply crossfeed
        let crossfeed_audio = self.apply_crossfeed(audio, position)?;

        // Apply compression if enabled
        let compressed_audio = if self.config.enable_compression {
            self.apply_compression(&crossfeed_audio)?
        } else {
            crossfeed_audio
        };

        Ok(compressed_audio)
    }

    /// Apply crossfeed to reduce fatigue
    fn apply_crossfeed(&self, audio: &[f32], position: &SpatialPosition) -> Result<BinauralOutput> {
        let mut left = audio.to_vec();
        let mut right = audio.to_vec();

        // Simple crossfeed implementation
        let crossfeed = self.config.crossfeed_amount;

        // Apply panning based on azimuth
        let pan = (position.azimuth * std::f32::consts::PI / 180.0).sin();
        let left_gain = ((1.0 - pan) / 2.0).sqrt();
        let right_gain = ((1.0 + pan) / 2.0).sqrt();

        for i in 0..audio.len() {
            let original_left = left[i];
            let original_right = right[i];

            left[i] = original_left * left_gain + original_right * crossfeed;
            right[i] = original_right * right_gain + original_left * crossfeed;
        }

        Ok(BinauralOutput { left, right })
    }

    /// Apply dynamic range compression
    fn apply_compression(&self, input: &BinauralOutput) -> Result<BinauralOutput> {
        let threshold = self.config.compression_threshold;
        let ratio = self.config.compression_ratio;
        let threshold_linear = 10.0_f32.powf(threshold / 20.0);

        let mut left = input.left.clone();
        let mut right = input.right.clone();

        // Simple compression implementation
        for sample in &mut left {
            if sample.abs() > threshold_linear {
                let excess = sample.abs() - threshold_linear;
                let compressed_excess = excess / ratio;
                *sample = (*sample).signum() * (threshold_linear + compressed_excess);
            }
        }

        for sample in &mut right {
            if sample.abs() > threshold_linear {
                let excess = sample.abs() - threshold_linear;
                let compressed_excess = excess / ratio;
                *sample = (*sample).signum() * (threshold_linear + compressed_excess);
            }
        }

        Ok(BinauralOutput { left, right })
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &BinauralConfig) -> Result<()> {
        self.config = config.clone();
        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &BinauralConfig {
        &self.config
    }

    /// Update sample rate and recalculate frequency-dependent parameters
    pub fn set_sample_rate(&mut self, sample_rate: u32) -> Result<()> {
        self.sample_rate = sample_rate;
        // Recalculate any frequency-dependent parameters here
        Ok(())
    }

    /// Get current sample rate
    pub fn get_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Apply frequency-dependent inter-aural time delay (ITD)
    /// This method uses the sample_rate for accurate delay calculations
    pub fn apply_itd(&self, audio: &[f32], position: &SpatialPosition) -> Result<BinauralOutput> {
        let mut left = audio.to_vec();
        let mut right = audio.to_vec();

        // Calculate inter-aural time delay based on azimuth
        // Maximum ITD is about 0.7ms for human head
        let max_itd_samples = (0.0007 * self.sample_rate as f32) as usize;
        let azimuth_rad = position.azimuth * std::f32::consts::PI / 180.0;

        // Calculate delay for each ear
        let itd_samples = (azimuth_rad.sin() * max_itd_samples as f32) as i32;

        if itd_samples > 0 {
            // Sound from right - delay right channel
            let delay = itd_samples as usize;
            if delay < right.len() {
                for i in (delay..right.len()).rev() {
                    right[i] = right[i - delay];
                }
                for sample in right.iter_mut().take(delay) {
                    *sample = 0.0;
                }
            }
        } else if itd_samples < 0 {
            // Sound from left - delay left channel
            let delay = (-itd_samples) as usize;
            if delay < left.len() {
                for i in (delay..left.len()).rev() {
                    left[i] = left[i - delay];
                }
                for sample in left.iter_mut().take(delay) {
                    *sample = 0.0;
                }
            }
        }

        Ok(BinauralOutput { left, right })
    }

    /// Apply high-frequency attenuation based on sample rate and position
    /// This method uses sample_rate for accurate frequency domain processing
    pub fn apply_frequency_attenuation(
        &self,
        input: &BinauralOutput,
        position: &SpatialPosition,
    ) -> Result<BinauralOutput> {
        let mut left = input.left.clone();
        let mut right = input.right.clone();

        // Calculate attenuation based on angle and distance
        let azimuth_rad = position.azimuth * std::f32::consts::PI / 180.0;
        let distance_factor = (1.0 + position.distance).recip();

        // High frequency roll-off for shadowed ear
        let hf_attenuation = if azimuth_rad > 0.0 {
            // Sound from right - attenuate left ear high frequencies
            0.7 + 0.3 * azimuth_rad.cos()
        } else {
            // Sound from left - attenuate right ear high frequencies
            0.7 + 0.3 * (-azimuth_rad).cos()
        };

        // Simple high-frequency attenuation (simulates head shadowing)
        // This is a simplified model - real HRTF would be more complex
        let cutoff_freq = 3000.0 * hf_attenuation * distance_factor;
        let cutoff_normalized = cutoff_freq / (self.sample_rate as f32 / 2.0);
        let filter_factor = (cutoff_normalized * std::f32::consts::PI).sin();

        // Apply different attenuation to each channel based on position
        if azimuth_rad > 0.0 {
            // Attenuate left channel high frequencies
            for sample in &mut left {
                *sample *= filter_factor;
            }
        } else if azimuth_rad < 0.0 {
            // Attenuate right channel high frequencies
            for sample in &mut right {
                *sample *= filter_factor;
            }
        }

        Ok(BinauralOutput { left, right })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::spatial::config::BinauralConfig;

    #[test]
    fn test_binaural_renderer_creation() {
        let config = BinauralConfig::default();
        let renderer = BinauralRenderer::new(&config);
        assert!(renderer.is_ok());
    }

    #[test]
    fn test_binaural_rendering() {
        let config = BinauralConfig::default();
        let mut renderer = BinauralRenderer::new(&config).unwrap();

        let audio = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let position = SpatialPosition::default();

        let result = renderer.render(&audio, &position);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.left.len(), audio.len());
        assert_eq!(output.right.len(), audio.len());
    }

    #[test]
    fn test_binaural_disabled() {
        let config = BinauralConfig {
            enable_binaural: false,
            ..Default::default()
        };

        let mut renderer = BinauralRenderer::new(&config).unwrap();

        let audio = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let position = SpatialPosition::default();

        let result = renderer.render(&audio, &position);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.left, audio);
        assert_eq!(output.right, audio);
    }

    #[test]
    fn test_crossfeed() {
        let config = BinauralConfig::default();
        let renderer = BinauralRenderer::new(&config).unwrap();

        let audio = vec![1.0, 1.0, 1.0];
        let position = SpatialPosition::default();

        let result = renderer.apply_crossfeed(&audio, &position);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.left.len(), audio.len());
        assert_eq!(output.right.len(), audio.len());
    }

    #[test]
    fn test_compression() {
        let config = BinauralConfig {
            enable_compression: true,
            compression_threshold: -10.0,
            compression_ratio: 2.0,
            ..Default::default()
        };

        let renderer = BinauralRenderer::new(&config).unwrap();

        let input = BinauralOutput {
            left: vec![2.0, 2.0, 2.0],
            right: vec![2.0, 2.0, 2.0],
        };

        let result = renderer.apply_compression(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.left[0] < 2.0); // Should be compressed
        assert!(output.right[0] < 2.0); // Should be compressed
    }

    #[test]
    fn test_config_update() {
        let config = BinauralConfig::default();
        let mut renderer = BinauralRenderer::new(&config).unwrap();

        let mut new_config = config.clone();
        new_config.crossfeed_amount = 0.5;

        let result = renderer.update_config(&new_config);
        assert!(result.is_ok());
        assert_eq!(renderer.config.crossfeed_amount, 0.5);
    }
}
