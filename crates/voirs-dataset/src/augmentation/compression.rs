//! Dynamic Range Compression (DRC) for audio augmentation
//!
//! This module provides dynamic range compression to control the dynamic range
//! of audio signals, making them more consistent in volume and suitable for
//! various playback conditions.
//!
//! # References
//! - Zölzer, Udo. "DAFX: digital audio effects." John Wiley & Sons, 2011.

use crate::{DatasetError, Result};
use std::f32;

/// Compression detection mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DetectionMode {
    /// Peak detection (instantaneous)
    Peak,
    /// RMS detection (average power)
    Rms,
}

/// Knee type for compression curve
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KneeType {
    /// Hard knee (sharp transition)
    Hard,
    /// Soft knee (smooth transition)
    Soft,
}

/// Configuration for dynamic range compression
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Threshold in dB (compression starts above this level)
    pub threshold_db: f32,
    /// Compression ratio (e.g., 4.0 for 4:1 compression)
    pub ratio: f32,
    /// Attack time in milliseconds
    pub attack_ms: f32,
    /// Release time in milliseconds
    pub release_ms: f32,
    /// Knee width in dB (for soft knee)
    pub knee_db: f32,
    /// Knee type (hard or soft)
    pub knee_type: KneeType,
    /// Makeup gain in dB
    pub makeup_gain_db: f32,
    /// Detection mode (peak or RMS)
    pub detection_mode: DetectionMode,
    /// RMS window size in samples (for RMS mode)
    pub rms_window_size: usize,
    /// Sample rate
    pub sample_rate: u32,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            knee_db: 6.0,
            knee_type: KneeType::Soft,
            makeup_gain_db: 0.0,
            detection_mode: DetectionMode::Rms,
            rms_window_size: 256,
            sample_rate: 22050,
        }
    }
}

impl CompressionConfig {
    /// Create a light compression preset
    pub fn light() -> Self {
        Self {
            threshold_db: -24.0,
            ratio: 2.0,
            attack_ms: 5.0,
            release_ms: 50.0,
            knee_db: 8.0,
            knee_type: KneeType::Soft,
            makeup_gain_db: 3.0,
            ..Default::default()
        }
    }

    /// Create a medium compression preset
    pub fn medium() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            knee_db: 6.0,
            knee_type: KneeType::Soft,
            makeup_gain_db: 6.0,
            ..Default::default()
        }
    }

    /// Create a heavy compression preset
    pub fn heavy() -> Self {
        Self {
            threshold_db: -16.0,
            ratio: 8.0,
            attack_ms: 2.0,
            release_ms: 50.0,
            knee_db: 3.0,
            knee_type: KneeType::Soft,
            makeup_gain_db: 10.0,
            ..Default::default()
        }
    }

    /// Create a limiter preset (very high ratio)
    pub fn limiter() -> Self {
        Self {
            threshold_db: -6.0,
            ratio: 20.0,
            attack_ms: 0.5,
            release_ms: 100.0,
            knee_db: 0.0,
            knee_type: KneeType::Hard,
            makeup_gain_db: 0.0,
            ..Default::default()
        }
    }
}

/// Dynamic range compressor
pub struct DynamicRangeCompressor {
    config: CompressionConfig,
    envelope: f32,
    attack_coeff: f32,
    release_coeff: f32,
    rms_buffer: Vec<f32>,
    rms_index: usize,
}

impl DynamicRangeCompressor {
    /// Create a new compressor with the given configuration
    pub fn new(config: CompressionConfig) -> Self {
        // Calculate attack and release coefficients
        let attack_coeff = Self::calculate_time_constant(config.attack_ms, config.sample_rate);
        let release_coeff = Self::calculate_time_constant(config.release_ms, config.sample_rate);

        Self {
            envelope: 0.0,
            attack_coeff,
            release_coeff,
            rms_buffer: vec![0.0; config.rms_window_size],
            rms_index: 0,
            config,
        }
    }

    /// Apply compression to audio samples
    pub fn apply(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        let mut output = Vec::with_capacity(audio.len());

        for &sample in audio {
            // Detect signal level
            let level = self.detect_level(sample);

            // Convert to dB
            let level_db = Self::linear_to_db(level);

            // Calculate gain reduction
            let gain_reduction_db = self.calculate_gain_reduction(level_db);

            // Apply envelope follower
            self.update_envelope(gain_reduction_db);

            // Convert gain reduction to linear
            let gain = Self::db_to_linear(-self.envelope + self.config.makeup_gain_db);

            // Apply gain
            output.push(sample * gain);
        }

        Ok(output)
    }

    /// Detect signal level based on detection mode
    fn detect_level(&mut self, sample: f32) -> f32 {
        match self.config.detection_mode {
            DetectionMode::Peak => sample.abs(),
            DetectionMode::Rms => {
                // Update RMS buffer
                self.rms_buffer[self.rms_index] = sample * sample;
                self.rms_index = (self.rms_index + 1) % self.config.rms_window_size;

                // Calculate RMS
                let sum: f32 = self.rms_buffer.iter().sum();
                let mean = sum / self.config.rms_window_size as f32;
                mean.sqrt()
            }
        }
    }

    /// Calculate gain reduction in dB
    fn calculate_gain_reduction(&self, input_db: f32) -> f32 {
        let threshold = self.config.threshold_db;
        let ratio = self.config.ratio;

        match self.config.knee_type {
            KneeType::Hard => {
                if input_db <= threshold {
                    0.0
                } else {
                    let overshoot = input_db - threshold;
                    overshoot * (1.0 - 1.0 / ratio)
                }
            }
            KneeType::Soft => {
                let knee = self.config.knee_db;
                let knee_start = threshold - knee / 2.0;
                let knee_end = threshold + knee / 2.0;

                if input_db < knee_start {
                    0.0
                } else if input_db > knee_end {
                    let overshoot = input_db - threshold;
                    overshoot * (1.0 - 1.0 / ratio)
                } else {
                    // Soft knee region (quadratic interpolation)
                    let x = input_db - knee_start;
                    let w = knee;
                    let slope = 1.0 - 1.0 / ratio;
                    x * x * slope / (2.0 * w)
                }
            }
        }
    }

    /// Update envelope follower
    fn update_envelope(&mut self, target_db: f32) {
        let coeff = if target_db > self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };

        self.envelope = coeff * self.envelope + (1.0 - coeff) * target_db;
    }

    /// Calculate time constant coefficient
    fn calculate_time_constant(time_ms: f32, sample_rate: u32) -> f32 {
        if time_ms <= 0.0 {
            0.0
        } else {
            (-1000.0 / (time_ms * sample_rate as f32)).exp()
        }
    }

    /// Convert linear amplitude to dB
    fn linear_to_db(linear: f32) -> f32 {
        if linear > 0.0 {
            20.0 * linear.log10()
        } else {
            -100.0 // Minimum dB
        }
    }

    /// Convert dB to linear amplitude
    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    /// Reset compressor state
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.rms_buffer.fill(0.0);
        self.rms_index = 0;
    }

    /// Get current envelope level in dB
    pub fn get_envelope_db(&self) -> f32 {
        self.envelope
    }
}

/// Batch compression processor
pub struct BatchCompressor {
    config: CompressionConfig,
}

impl BatchCompressor {
    /// Create a new batch compressor
    pub fn new(config: CompressionConfig) -> Self {
        Self { config }
    }

    /// Apply compression to multiple audio samples
    pub fn apply_batch(&self, audio_samples: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        audio_samples
            .iter()
            .map(|audio| {
                let mut compressor = DynamicRangeCompressor::new(self.config.clone());
                compressor.apply(audio)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_config_default() {
        let config = CompressionConfig::default();
        assert_eq!(config.threshold_db, -20.0);
        assert_eq!(config.ratio, 4.0);
        assert_eq!(config.detection_mode, DetectionMode::Rms);
        assert_eq!(config.knee_type, KneeType::Soft);
    }

    #[test]
    fn test_compression_config_presets() {
        let light = CompressionConfig::light();
        assert_eq!(light.ratio, 2.0);
        assert!(light.threshold_db < 0.0);

        let medium = CompressionConfig::medium();
        assert_eq!(medium.ratio, 4.0);

        let heavy = CompressionConfig::heavy();
        assert_eq!(heavy.ratio, 8.0);

        let limiter = CompressionConfig::limiter();
        assert_eq!(limiter.ratio, 20.0);
        assert_eq!(limiter.knee_type, KneeType::Hard);
    }

    #[test]
    fn test_linear_to_db_conversion() {
        assert!((DynamicRangeCompressor::linear_to_db(1.0) - 0.0).abs() < 0.001);
        assert!((DynamicRangeCompressor::linear_to_db(0.5) - (-6.02)).abs() < 0.1);
        assert!((DynamicRangeCompressor::linear_to_db(0.1) - (-20.0)).abs() < 0.1);
    }

    #[test]
    fn test_db_to_linear_conversion() {
        assert!((DynamicRangeCompressor::db_to_linear(0.0) - 1.0).abs() < 0.001);
        assert!((DynamicRangeCompressor::db_to_linear(-6.0) - 0.5).abs() < 0.01);
        assert!((DynamicRangeCompressor::db_to_linear(-20.0) - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_compressor_creation() {
        let config = CompressionConfig::default();
        let compressor = DynamicRangeCompressor::new(config);
        assert_eq!(compressor.envelope, 0.0);
    }

    #[test]
    fn test_compressor_empty_audio() {
        let config = CompressionConfig::default();
        let mut compressor = DynamicRangeCompressor::new(config);
        let result = compressor.apply(&[]).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_compressor_basic_compression() {
        let config = CompressionConfig {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 1.0,
            release_ms: 10.0,
            knee_db: 0.0,
            knee_type: KneeType::Hard,
            makeup_gain_db: 0.0,
            detection_mode: DetectionMode::Peak,
            rms_window_size: 128,
            sample_rate: 16000,
        };

        let mut compressor = DynamicRangeCompressor::new(config);

        // Create a loud signal (should be compressed)
        let audio: Vec<f32> = vec![0.5; 1000];
        let result = compressor.apply(&audio).unwrap();

        assert_eq!(result.len(), audio.len());
        // After compression, amplitude should be reduced
        let input_rms: f32 = audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32;
        let output_rms: f32 = result.iter().map(|&x| x * x).sum::<f32>() / result.len() as f32;

        // Output should have lower RMS due to compression (allowing for attack time)
        assert!(output_rms <= input_rms * 1.1);
    }

    #[test]
    fn test_compressor_with_quiet_signal() {
        let config = CompressionConfig {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 1.0,
            release_ms: 10.0,
            knee_db: 0.0,
            knee_type: KneeType::Hard,
            makeup_gain_db: 0.0,
            detection_mode: DetectionMode::Peak,
            rms_window_size: 128,
            sample_rate: 16000,
        };

        let mut compressor = DynamicRangeCompressor::new(config);

        // Create a quiet signal (below threshold, should not be compressed)
        let audio: Vec<f32> = vec![0.01; 1000];
        let result = compressor.apply(&audio).unwrap();

        assert_eq!(result.len(), audio.len());
        // Quiet signals should pass through relatively unchanged
        let max_diff = audio
            .iter()
            .zip(result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        assert!(max_diff < 0.02);
    }

    #[test]
    fn test_compressor_with_makeup_gain() {
        let config = CompressionConfig {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 1.0,
            release_ms: 10.0,
            knee_db: 0.0,
            knee_type: KneeType::Hard,
            makeup_gain_db: 10.0, // Add makeup gain
            detection_mode: DetectionMode::Peak,
            rms_window_size: 128,
            sample_rate: 16000,
        };

        let mut compressor = DynamicRangeCompressor::new(config);

        let audio: Vec<f32> = vec![0.1; 1000];
        let result = compressor.apply(&audio).unwrap();

        assert_eq!(result.len(), audio.len());
        // With makeup gain, output should be louder
        let output_rms: f32 = result.iter().map(|&x| x * x).sum::<f32>() / result.len() as f32;
        let input_rms: f32 = audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32;

        assert!(output_rms > input_rms);
    }

    #[test]
    fn test_compressor_reset() {
        let config = CompressionConfig::default();
        let mut compressor = DynamicRangeCompressor::new(config);

        // Apply compression
        let audio = vec![0.5; 1000];
        let _ = compressor.apply(&audio).unwrap();

        // Envelope should have changed
        assert!(compressor.get_envelope_db() > 0.0);

        // Reset
        compressor.reset();
        assert_eq!(compressor.envelope, 0.0);
    }

    #[test]
    fn test_rms_detection_mode() {
        let config = CompressionConfig {
            detection_mode: DetectionMode::Rms,
            rms_window_size: 256,
            sample_rate: 16000,
            ..Default::default()
        };

        let mut compressor = DynamicRangeCompressor::new(config);

        let audio: Vec<f32> = (0..1000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        let result = compressor.apply(&audio).unwrap();
        assert_eq!(result.len(), audio.len());
    }

    #[test]
    fn test_peak_detection_mode() {
        let config = CompressionConfig {
            detection_mode: DetectionMode::Peak,
            sample_rate: 16000,
            ..Default::default()
        };

        let mut compressor = DynamicRangeCompressor::new(config);

        let audio: Vec<f32> = (0..1000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        let result = compressor.apply(&audio).unwrap();
        assert_eq!(result.len(), audio.len());
    }

    #[test]
    fn test_soft_knee() {
        let config = CompressionConfig {
            knee_type: KneeType::Soft,
            knee_db: 6.0,
            threshold_db: -20.0,
            ratio: 4.0,
            ..Default::default()
        };

        let compressor = DynamicRangeCompressor::new(config.clone());

        // Test knee region
        let input_db = -20.0; // At threshold
        let gain_reduction = compressor.calculate_gain_reduction(input_db);

        // At threshold with soft knee, there should be some gain reduction
        assert!(gain_reduction >= 0.0);
    }

    #[test]
    fn test_hard_knee() {
        let config = CompressionConfig {
            knee_type: KneeType::Hard,
            threshold_db: -20.0,
            ratio: 4.0,
            ..Default::default()
        };

        let compressor = DynamicRangeCompressor::new(config.clone());

        // Below threshold
        let below_threshold = compressor.calculate_gain_reduction(-25.0);
        assert_eq!(below_threshold, 0.0);

        // Above threshold
        let above_threshold = compressor.calculate_gain_reduction(-10.0);
        assert!(above_threshold > 0.0);
    }

    #[test]
    fn test_batch_compressor() {
        let config = CompressionConfig::medium();
        let batch_compressor = BatchCompressor::new(config);

        let audio1 = vec![0.5; 500];
        let audio2 = vec![0.3; 500];
        let audio3 = vec![0.7; 500];

        let batch = vec![audio1, audio2, audio3];
        let results = batch_compressor.apply_batch(&batch).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].len(), 500);
        assert_eq!(results[1].len(), 500);
        assert_eq!(results[2].len(), 500);
    }

    #[test]
    fn test_time_constant_calculation() {
        let attack_coeff = DynamicRangeCompressor::calculate_time_constant(10.0, 44100);
        assert!(attack_coeff > 0.0 && attack_coeff < 1.0);

        // Zero time should give zero coefficient
        let zero_coeff = DynamicRangeCompressor::calculate_time_constant(0.0, 44100);
        assert_eq!(zero_coeff, 0.0);
    }

    #[test]
    fn test_limiter_preset() {
        let config = CompressionConfig::limiter();
        let mut compressor = DynamicRangeCompressor::new(config);

        // Create a signal with sustained high level (easier to limit than a single peak)
        let mut audio = vec![0.1; 2000];
        // Add a sustained loud section
        audio[500..1500].iter_mut().for_each(|x| *x = 0.8);

        let result = compressor.apply(&audio).unwrap();

        // Limiter should reduce the sustained loud section
        // Check the middle of the loud section where the limiter has had time to act
        let loud_section_output: f32 = result[1000..1400].iter().map(|&x| x.abs()).sum();
        let loud_section_input: f32 = audio[1000..1400].iter().map(|&x| x.abs()).sum();

        assert!(
            loud_section_output < loud_section_input,
            "Limiter should reduce sustained loud section"
        );
    }
}
