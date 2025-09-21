//! Format conversion module for audio format transformations

use super::{DitheringMethod, FormatConversionConfig};
use crate::{AudioBuffer, Result, VocoderError};
use fastrand;

/// Format converter for audio transformations
#[derive(Debug)]
pub struct FormatConverter {
    config: FormatConversionConfig,
    dither_state: DitheringState,
}

#[derive(Debug)]
struct DitheringState {
    // Random number generator state for dithering
    rng_state: u64,
    // Previous error for noise shaping
    prev_error: f32,
}

impl FormatConverter {
    /// Create new format converter
    pub fn new(config: &FormatConversionConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            dither_state: DitheringState {
                rng_state: fastrand::u64(..),
                prev_error: 0.0,
            },
        })
    }

    /// Process audio through format conversion
    pub fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Apply conversions in order:
        // 1. Sample rate conversion (if needed)
        if self.config.target_sample_rate > 0
            && self.config.target_sample_rate != audio.sample_rate()
        {
            self.resample_audio(audio)?;
        }

        // 2. Channel conversion (if needed)
        if self.config.target_channels as u32 != audio.channels() {
            self.convert_channels(audio)?;
        }

        // 3. Bit depth conversion with dithering (if needed)
        if self.config.target_bit_depth != 32 {
            self.convert_bit_depth(audio)?;
        }

        Ok(())
    }

    /// Resample audio to target sample rate
    fn resample_audio(&self, audio: &mut AudioBuffer) -> Result<()> {
        let current_rate = audio.sample_rate() as f32;
        let target_rate = self.config.target_sample_rate as f32;
        let ratio = target_rate / current_rate;

        if (ratio - 1.0).abs() < 1e-6 {
            return Ok(()); // No resampling needed
        }

        let current_samples = audio.samples();
        let current_channels = audio.channels();
        let new_length = (current_samples.len() as f32 * ratio) as usize;

        // Simple linear interpolation resampling
        // In production, you'd want to use a proper resampling library
        let mut new_samples = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let src_index = i as f32 / ratio;
            let src_index_floor = src_index.floor() as usize;
            let src_index_ceil = (src_index_floor + 1).min(current_samples.len() - 1);
            let frac = src_index - src_index_floor as f32;

            if src_index_floor < current_samples.len() {
                let sample1 = current_samples[src_index_floor];
                let sample2 = current_samples[src_index_ceil];
                let interpolated = sample1 + (sample2 - sample1) * frac;
                new_samples.push(interpolated);
            } else {
                new_samples.push(0.0);
            }
        }

        // Update audio buffer
        *audio = AudioBuffer::new(
            new_samples,
            self.config.target_sample_rate,
            current_channels,
        );

        Ok(())
    }

    /// Convert number of channels
    fn convert_channels(&self, audio: &mut AudioBuffer) -> Result<()> {
        let current_channels = audio.channels();
        let target_channels = self.config.target_channels as u32;

        if current_channels == target_channels {
            return Ok(());
        }

        let current_samples = audio.samples();
        let sample_rate = audio.sample_rate();
        let frames = current_samples.len() / current_channels as usize;

        let mut new_samples = Vec::with_capacity(frames * target_channels as usize);

        if current_channels == 1 && target_channels == 2 {
            // Mono to stereo - duplicate channel
            for sample in current_samples {
                new_samples.push(*sample);
                new_samples.push(*sample);
            }
        } else if current_channels == 2 && target_channels == 1 {
            // Stereo to mono - average channels
            for i in (0..current_samples.len()).step_by(2) {
                let left = current_samples[i];
                let right = if i + 1 < current_samples.len() {
                    current_samples[i + 1]
                } else {
                    left
                };
                new_samples.push((left + right) / 2.0);
            }
        } else {
            // More complex channel mapping - simplified approach
            for frame in 0..frames {
                for target_ch in 0..target_channels {
                    let source_ch = (target_ch % current_channels) as usize;
                    let source_idx = frame * current_channels as usize + source_ch;
                    if source_idx < current_samples.len() {
                        new_samples.push(current_samples[source_idx]);
                    } else {
                        new_samples.push(0.0);
                    }
                }
            }
        }

        *audio = AudioBuffer::new(new_samples, sample_rate, target_channels);

        Ok(())
    }

    /// Convert bit depth with dithering
    fn convert_bit_depth(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if self.config.target_bit_depth == 32 {
            return Ok(()); // Already 32-bit float
        }

        let samples = audio.samples_mut();
        let max_value = match self.config.target_bit_depth {
            16 => 32767.0,
            24 => 8388607.0,
            8 => 127.0,
            _ => {
                return Err(VocoderError::ConfigurationError(format!(
                    "Unsupported bit depth: {}",
                    self.config.target_bit_depth
                )))
            }
        };

        let quantization_step = 2.0 / max_value;

        for sample in samples.iter_mut() {
            // Apply dithering before quantization
            let dithered_sample = match self.config.dithering {
                DitheringMethod::None => *sample,
                DitheringMethod::RectangularPdf => {
                    *sample + self.rectangular_dither(quantization_step)
                }
                DitheringMethod::TriangularPdf => {
                    *sample + self.triangular_dither(quantization_step)
                }
                DitheringMethod::Shaped => *sample + self.shaped_dither(quantization_step),
            };

            // Quantize to target bit depth
            let quantized = (dithered_sample * max_value).round() / max_value;

            // Clamp to valid range
            *sample = quantized.clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Generate rectangular PDF dither noise
    fn rectangular_dither(&mut self, quantization_step: f32) -> f32 {
        // Simple uniform random number generation
        self.dither_state.rng_state = self
            .dither_state
            .rng_state
            .wrapping_mul(1103515245)
            .wrapping_add(12345);
        let uniform = (self.dither_state.rng_state as f32) / (u64::MAX as f32);
        (uniform - 0.5) * quantization_step
    }

    /// Generate triangular PDF dither noise
    fn triangular_dither(&mut self, quantization_step: f32) -> f32 {
        // Sum of two uniform random variables creates triangular distribution
        let dither1 = self.rectangular_dither(quantization_step);
        let dither2 = self.rectangular_dither(quantization_step);
        (dither1 + dither2) / 2.0
    }

    /// Generate shaped dither noise (simplified noise shaping)
    fn shaped_dither(&mut self, quantization_step: f32) -> f32 {
        let dither = self.triangular_dither(quantization_step);

        // Simple first-order noise shaping
        let shaped_dither = dither - self.dither_state.prev_error * 0.5;
        self.dither_state.prev_error = dither;

        shaped_dither
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &FormatConversionConfig) -> Result<()> {
        self.config = config.clone();
        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &FormatConversionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_converter_creation() {
        let config = FormatConversionConfig::default();
        let converter = FormatConverter::new(&config);
        assert!(converter.is_ok());
    }

    #[test]
    fn test_disabled_conversion() {
        let config = FormatConversionConfig {
            enabled: false,
            ..Default::default()
        };

        let mut converter = FormatConverter::new(&config).unwrap();

        let samples = vec![0.1, 0.2, 0.3];
        let original_samples = samples.clone();
        let mut audio = AudioBuffer::new(samples, 44100, 1);

        converter.process(&mut audio).unwrap();

        // Should be unchanged when disabled
        assert_eq!(audio.samples(), &original_samples);
    }

    #[test]
    fn test_mono_to_stereo_conversion() {
        let config = FormatConversionConfig {
            enabled: true,
            target_channels: 2,
            ..Default::default()
        };

        let mut converter = FormatConverter::new(&config).unwrap();

        let samples = vec![0.1, 0.2, 0.3];
        let mut audio = AudioBuffer::new(samples, 44100, 1);

        converter.process(&mut audio).unwrap();

        assert_eq!(audio.channels(), 2);
        assert_eq!(audio.samples().len(), 6); // Should be doubled

        // Check that samples are duplicated (with tolerance for floating point precision)
        let output_samples = audio.samples();
        assert!(
            (output_samples[0] - output_samples[1]).abs() < 1e-3,
            "Expected samples to be equal: {} vs {}",
            output_samples[0],
            output_samples[1]
        );
        assert!(
            (output_samples[2] - output_samples[3]).abs() < 1e-3,
            "Expected samples to be equal: {} vs {}",
            output_samples[2],
            output_samples[3]
        );
        assert!(
            (output_samples[4] - output_samples[5]).abs() < 1e-3,
            "Expected samples to be equal: {} vs {}",
            output_samples[4],
            output_samples[5]
        );
    }

    #[test]
    fn test_stereo_to_mono_conversion() {
        let config = FormatConversionConfig {
            enabled: true,
            target_channels: 1,
            ..Default::default()
        };

        let mut converter = FormatConverter::new(&config).unwrap();

        let samples = vec![0.1, 0.2, 0.3, 0.4]; // Stereo pairs
        let mut audio = AudioBuffer::new(samples, 44100, 2);

        converter.process(&mut audio).unwrap();

        assert_eq!(audio.channels(), 1);
        assert_eq!(audio.samples().len(), 2); // Should be halved

        // Check that samples are averaged (with tolerance for floating point precision)
        let output_samples = audio.samples();
        let expected0 = (0.1 + 0.2) / 2.0;
        let expected1 = (0.3 + 0.4) / 2.0;
        assert!(
            (output_samples[0] - expected0).abs() < 1e-3,
            "Expected average: {} vs actual: {}",
            expected0,
            output_samples[0]
        );
        assert!(
            (output_samples[1] - expected1).abs() < 1e-3,
            "Expected average: {} vs actual: {}",
            expected1,
            output_samples[1]
        );
    }

    #[test]
    fn test_sample_rate_conversion() {
        let config = FormatConversionConfig {
            enabled: true,
            target_sample_rate: 22050,
            ..Default::default()
        };

        let mut converter = FormatConverter::new(&config).unwrap();

        let samples = vec![0.1, 0.2, 0.3, 0.4]; // 4 samples at 44100
        let mut audio = AudioBuffer::new(samples, 44100, 1);

        converter.process(&mut audio).unwrap();

        assert_eq!(audio.sample_rate(), 22050);
        assert_eq!(audio.samples().len(), 2); // Should be roughly half
    }

    #[test]
    fn test_bit_depth_conversion() {
        let config = FormatConversionConfig {
            enabled: true,
            target_bit_depth: 16,
            dithering: DitheringMethod::None,
            ..Default::default()
        };

        let mut converter = FormatConverter::new(&config).unwrap();

        let samples = vec![0.1, 0.2, -0.3];
        let mut audio = AudioBuffer::new(samples, 44100, 1);

        converter.process(&mut audio).unwrap();

        // Samples should be quantized to 16-bit resolution
        let output_samples = audio.samples();
        for &sample in output_samples {
            // Check that quantization happened (values should be rounded to 16-bit steps)
            let quantized = (sample * 32767.0).round() / 32767.0;
            assert!((sample - quantized).abs() < 1e-6);
        }
    }

    #[test]
    fn test_triangular_dithering() {
        let config = FormatConversionConfig {
            enabled: true,
            target_bit_depth: 16,
            dithering: DitheringMethod::TriangularPdf,
            ..FormatConversionConfig::default()
        };

        let mut converter = FormatConverter::new(&config).unwrap();

        let samples = vec![0.1; 100]; // Repeated value
        let mut audio = AudioBuffer::new(samples.clone(), 44100, 1);

        converter.process(&mut audio).unwrap();

        // With dithering, values should vary slightly
        let output_samples = audio.samples();
        let mut has_variation = false;
        for i in 1..output_samples.len() {
            if (output_samples[i] - output_samples[i - 1]).abs() > 1e-6 {
                has_variation = true;
                break;
            }
        }

        // Dithering should create some variation in the output
        assert!(has_variation);
    }
}
