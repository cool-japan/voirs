//! Dynamic range compression for audio post-processing
//!
//! Implements a high-quality compressor with configurable attack/release times,
//! ratio, threshold, and soft knee characteristics.

use super::{db_to_linear, linear_to_db, soft_knee_compression, CompressionConfig};
use crate::{AudioBuffer, Result, VocoderError};

/// Dynamic range compressor with envelope following
#[derive(Debug)]
pub struct DynamicCompressor {
    config: CompressionConfig,
    sample_rate: f32,

    // Envelope follower state
    envelope: f32,
    attack_coeff: f32,
    release_coeff: f32,

    // Statistics
    current_gain_reduction: f32,
    processed_samples: u64,
}

impl DynamicCompressor {
    /// Create a new compressor with the given configuration
    pub fn new(config: &CompressionConfig, sample_rate: f32) -> Result<Self> {
        // Calculate envelope coefficients
        let attack_coeff = (-1.0 / (config.attack_ms * 0.001 * sample_rate)).exp();
        let release_coeff = (-1.0 / (config.release_ms * 0.001 * sample_rate)).exp();

        Ok(Self {
            config: config.clone(),
            sample_rate,
            envelope: 0.0,
            attack_coeff,
            release_coeff,
            current_gain_reduction: 0.0,
            processed_samples: 0,
        })
    }

    /// Process audio buffer with compression
    pub fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let samples = audio.samples_mut();
        let makeup_gain_linear = db_to_linear(self.config.makeup_gain);

        for sample in samples.iter_mut() {
            // Calculate input level
            let input_level = sample.abs();

            // Update envelope follower
            let target_envelope = input_level;
            if target_envelope > self.envelope {
                // Attack: fast response to level increases
                self.envelope =
                    target_envelope + (self.envelope - target_envelope) * self.attack_coeff;
            } else {
                // Release: slower response to level decreases
                self.envelope =
                    target_envelope + (self.envelope - target_envelope) * self.release_coeff;
            }

            // Convert to dB for compression calculation
            let envelope_db = linear_to_db(self.envelope);

            // Calculate compression amount
            let compressed_db = if self.config.soft_knee {
                soft_knee_compression(envelope_db, self.config.threshold, self.config.ratio, 4.0)
            } else {
                // Hard knee compression
                if envelope_db > self.config.threshold {
                    self.config.threshold
                        + (envelope_db - self.config.threshold) / self.config.ratio
                } else {
                    envelope_db
                }
            };

            // Calculate gain reduction
            let gain_reduction_db = compressed_db - envelope_db;
            self.current_gain_reduction = gain_reduction_db;

            // Convert back to linear and apply compression + makeup gain
            let compression_gain = db_to_linear(gain_reduction_db);
            *sample *= compression_gain * makeup_gain_linear;

            self.processed_samples += 1;
        }

        Ok(())
    }

    /// Get current gain reduction in dB
    pub fn gain_reduction(&self) -> f32 {
        self.current_gain_reduction
    }

    /// Get total processed samples
    pub fn processed_samples(&self) -> u64 {
        self.processed_samples
    }

    /// Reset internal state
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.current_gain_reduction = 0.0;
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &CompressionConfig) -> Result<()> {
        self.config = config.clone();

        // Recalculate coefficients
        self.attack_coeff = (-1.0 / (config.attack_ms * 0.001 * self.sample_rate)).exp();
        self.release_coeff = (-1.0 / (config.release_ms * 0.001 * self.sample_rate)).exp();

        Ok(())
    }
}

/// Multi-band compressor for frequency-specific compression
#[derive(Debug)]
pub struct MultibandCompressor {
    low_band: DynamicCompressor,
    mid_band: DynamicCompressor,
    high_band: DynamicCompressor,
    crossover_low: f32,
    crossover_high: f32,

    // Filter states for band splitting
    low_filter_state: BiquadFilter,
    high_filter_state: BiquadFilter,
}

impl MultibandCompressor {
    /// Create a new multiband compressor
    pub fn new(
        low_config: &CompressionConfig,
        mid_config: &CompressionConfig,
        high_config: &CompressionConfig,
        crossover_low: f32,
        crossover_high: f32,
        sample_rate: f32,
    ) -> Result<Self> {
        let low_band = DynamicCompressor::new(low_config, sample_rate)?;
        let mid_band = DynamicCompressor::new(mid_config, sample_rate)?;
        let high_band = DynamicCompressor::new(high_config, sample_rate)?;

        let low_filter_state = BiquadFilter::lowpass(crossover_low, sample_rate);
        let high_filter_state = BiquadFilter::highpass(crossover_high, sample_rate);

        Ok(Self {
            low_band,
            mid_band,
            high_band,
            crossover_low,
            crossover_high,
            low_filter_state,
            high_filter_state,
        })
    }

    /// Process audio with multiband compression
    pub fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        let sample_rate = audio.sample_rate();
        let samples = audio.samples_mut();

        for sample in samples.iter_mut() {
            let input = *sample;

            // Split into frequency bands
            let low_band_sample = self.low_filter_state.process(input);
            let high_band_sample = self.high_filter_state.process(input);
            let mid_band_sample = input - low_band_sample - high_band_sample;

            // Create temporary single-sample buffers for each band
            let mut low_buffer = AudioBuffer::new(vec![low_band_sample], sample_rate, 1);
            let mut mid_buffer = AudioBuffer::new(vec![mid_band_sample], sample_rate, 1);
            let mut high_buffer = AudioBuffer::new(vec![high_band_sample], sample_rate, 1);

            // Compress each band
            self.low_band.process(&mut low_buffer)?;
            self.mid_band.process(&mut mid_buffer)?;
            self.high_band.process(&mut high_buffer)?;

            // Sum the compressed bands
            *sample = low_buffer.samples()[0] + mid_buffer.samples()[0] + high_buffer.samples()[0];
        }

        Ok(())
    }

    /// Get the current crossover frequencies
    pub fn get_crossover_frequencies(&self) -> (f32, f32) {
        (self.crossover_low, self.crossover_high)
    }

    /// Update crossover frequencies and reconfigure filters
    pub fn set_crossover_frequencies(
        &mut self,
        crossover_low: f32,
        crossover_high: f32,
        sample_rate: f32,
    ) -> Result<()> {
        if crossover_low >= crossover_high {
            return Err(VocoderError::Other(
                "Low crossover frequency must be less than high crossover frequency".to_string(),
            ));
        }

        self.crossover_low = crossover_low;
        self.crossover_high = crossover_high;

        // Reconfigure filters with new crossover frequencies
        self.low_filter_state = BiquadFilter::lowpass(crossover_low, sample_rate);
        self.high_filter_state = BiquadFilter::highpass(crossover_high, sample_rate);

        Ok(())
    }

    /// Check if frequencies are in the low band
    pub fn is_low_band(&self, frequency: f32) -> bool {
        frequency <= self.crossover_low
    }

    /// Check if frequencies are in the mid band  
    pub fn is_mid_band(&self, frequency: f32) -> bool {
        frequency > self.crossover_low && frequency <= self.crossover_high
    }

    /// Check if frequencies are in the high band
    pub fn is_high_band(&self, frequency: f32) -> bool {
        frequency > self.crossover_high
    }
}

/// Simple biquad filter for band splitting
#[derive(Debug, Clone)]
struct BiquadFilter {
    a0: f32,
    a1: f32,
    a2: f32,
    b1: f32,
    b2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BiquadFilter {
    /// Create a lowpass filter
    fn lowpass(cutoff: f32, sample_rate: f32) -> Self {
        use std::f32::consts::PI;

        let omega = 2.0 * PI * cutoff / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let alpha = sin_omega / (2.0 * 0.707); // Q = 0.707 for Butterworth response

        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            a0: b0 / a0,
            a1: b1 / a0,
            a2: b2 / a0,
            b1: a1 / a0,
            b2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Create a highpass filter
    fn highpass(cutoff: f32, sample_rate: f32) -> Self {
        use std::f32::consts::PI;

        let omega = 2.0 * PI * cutoff / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let alpha = sin_omega / (2.0 * 0.707); // Q = 0.707 for Butterworth response

        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            a0: b0 / a0,
            a1: b1 / a0,
            a2: b2 / a0,
            b1: a1 / a0,
            b2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Process a single sample
    fn process(&mut self, input: f32) -> f32 {
        let output = self.a0 * input + self.a1 * self.x1 + self.a2 * self.x2
            - self.b1 * self.y1
            - self.b2 * self.y2;

        // Update delay line
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor_creation() {
        let config = CompressionConfig::default();
        let compressor = DynamicCompressor::new(&config, 48000.0);
        assert!(compressor.is_ok());
    }

    #[test]
    fn test_compressor_process() {
        let config = CompressionConfig {
            threshold: -20.0,
            ratio: 4.0,
            attack_ms: 5.0,
            release_ms: 50.0,
            makeup_gain: 0.0,
            soft_knee: false,
            enabled: true,
        };

        let mut compressor = DynamicCompressor::new(&config, 48000.0).unwrap();

        // Create test signal with varying amplitude
        let samples = vec![0.0, 0.5, 1.0, 0.8, 0.2];
        let mut audio = AudioBuffer::from_samples(samples, 48000.0);

        let result = compressor.process(&mut audio);
        assert!(result.is_ok());

        // Check that processing occurred
        assert!(compressor.processed_samples() > 0);
    }

    #[test]
    fn test_multiband_compressor() {
        let config = CompressionConfig::default();
        let compressor =
            MultibandCompressor::new(&config, &config, &config, 200.0, 2000.0, 48000.0);
        assert!(compressor.is_ok());
    }

    #[test]
    fn test_biquad_filter() {
        let mut filter = BiquadFilter::lowpass(1000.0, 48000.0);

        // Test with impulse
        let impulse_response = filter.process(1.0);
        assert!(impulse_response > 0.0);

        // Test stability
        for _ in 0..100 {
            let response = filter.process(0.0);
            assert!(response.is_finite());
        }
    }
}
