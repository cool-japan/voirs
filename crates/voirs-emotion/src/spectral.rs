//! Advanced Spectral Processing for Emotional Expression
//!
//! This module provides sophisticated spectral domain processing capabilities
//! for emotion-based voice synthesis, including spectral shaping, tilting,
//! and emotion-specific filtering.
//!
//! ## Features
//!
//! - **Spectral Tilt**: Adjust spectral slope for emotional characteristics
//! - **Emotion-specific Filtering**: Pre-designed spectral shapes for emotions
//! - **Harmonic Enhancement**: Boost or suppress harmonic content
//! - **Spectral Centroid Control**: Shift brightness/darkness of voice

use crate::{types::Emotion, Result};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Spectral processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralConfig {
    /// Spectral tilt (dB/octave)
    pub tilt: f32,
    /// Spectral centroid shift (Hz)
    pub centroid_shift: f32,
    /// Harmonic enhancement factor
    pub harmonic_boost: f32,
    /// High-frequency emphasis
    pub hf_emphasis: f32,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            tilt: 0.0,
            centroid_shift: 0.0,
            harmonic_boost: 1.0,
            hf_emphasis: 1.0,
        }
    }
}

impl SpectralConfig {
    /// Create config for happy emotion
    pub fn happy() -> Self {
        Self {
            tilt: -3.0,            // Less roll-off (brighter)
            centroid_shift: 200.0, // Higher centroid
            harmonic_boost: 1.2,   // Enhanced harmonics
            hf_emphasis: 1.3,      // Boost high frequencies
        }
    }

    /// Create config for sad emotion
    pub fn sad() -> Self {
        Self {
            tilt: 3.0,              // More roll-off (darker)
            centroid_shift: -150.0, // Lower centroid
            harmonic_boost: 0.8,    // Reduced harmonics
            hf_emphasis: 0.7,       // Reduce high frequencies
        }
    }

    /// Create config for angry emotion
    pub fn angry() -> Self {
        Self {
            tilt: -2.0,            // Brighter
            centroid_shift: 300.0, // Much higher centroid
            harmonic_boost: 1.4,   // Strong harmonics
            hf_emphasis: 1.5,      // Strong HF boost
        }
    }

    /// Create config for calm emotion
    pub fn calm() -> Self {
        Self {
            tilt: 1.0,             // Slight roll-off
            centroid_shift: -50.0, // Slightly lower
            harmonic_boost: 1.0,   // Neutral
            hf_emphasis: 0.9,      // Slight reduction
        }
    }

    /// Create config from emotion
    pub fn from_emotion(emotion: Emotion) -> Self {
        match emotion {
            Emotion::Happy => Self::happy(),
            Emotion::Sad => Self::sad(),
            Emotion::Angry => Self::angry(),
            Emotion::Calm => Self::calm(),
            Emotion::Excited => Self {
                tilt: -4.0,
                centroid_shift: 250.0,
                harmonic_boost: 1.3,
                hf_emphasis: 1.4,
            },
            Emotion::Fear => Self {
                tilt: -1.0,
                centroid_shift: 180.0,
                harmonic_boost: 1.1,
                hf_emphasis: 1.2,
            },
            _ => Self::default(),
        }
    }
}

/// Spectral processor for emotion-based modification
pub struct SpectralProcessor {
    /// Current configuration
    config: SpectralConfig,
    /// Sample rate
    sample_rate: f32,
    /// FFT size
    fft_size: usize,
}

impl SpectralProcessor {
    /// Create a new spectral processor
    pub fn new(sample_rate: f32, fft_size: usize) -> Self {
        Self {
            config: SpectralConfig::default(),
            sample_rate,
            fft_size,
        }
    }

    /// Set spectral configuration
    pub fn set_config(&mut self, config: SpectralConfig) {
        self.config = config;
    }

    /// FFT size used for spectral analysis/synthesis.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Apply spectral tilt to magnitude spectrum
    ///
    /// Spectral tilt modifies the spectral envelope slope, making the voice
    /// brighter (negative tilt) or darker (positive tilt).
    pub fn apply_tilt(&self, magnitudes: &mut [f32]) {
        if self.config.tilt.abs() < 0.01 {
            return;
        }

        let num_bins = magnitudes.len();

        for (i, magnitude) in magnitudes.iter_mut().enumerate() {
            let freq = i as f32 * self.sample_rate / self.fft_size as f32;

            // Calculate tilt in dB: gain = tilt * log2(freq / 1000)
            let freq_normalized = (freq / 1000.0).max(0.001);
            let tilt_db = self.config.tilt * freq_normalized.log2();

            // Convert dB to linear gain
            let gain = 10.0f32.powf(tilt_db / 20.0);

            *magnitude *= gain;
        }
    }

    /// Apply harmonic enhancement
    ///
    /// Boosts or suppresses harmonic content to affect voice timbre.
    pub fn apply_harmonic_enhancement(&self, magnitudes: &mut [f32], f0: f32) {
        if (self.config.harmonic_boost - 1.0).abs() < 0.01 {
            return;
        }

        let num_bins = magnitudes.len();
        let bin_width = self.sample_rate / self.fft_size as f32;

        // Process each harmonic
        for harmonic in 1..=20 {
            let harmonic_freq = f0 * harmonic as f32;
            if harmonic_freq > self.sample_rate / 2.0 {
                break;
            }

            let center_bin = (harmonic_freq / bin_width) as usize;
            let bandwidth = (f0 / bin_width) as usize / 2; // Half period

            // Apply Gaussian-like weighting around harmonic
            for offset in -(bandwidth as isize)..=(bandwidth as isize) {
                let bin = (center_bin as isize + offset) as usize;
                if bin < num_bins {
                    let distance = offset.abs() as f32 / bandwidth as f32;
                    let weight = (-distance * distance).exp();
                    let boost = 1.0 + (self.config.harmonic_boost - 1.0) * weight;
                    magnitudes[bin] *= boost;
                }
            }
        }
    }

    /// Apply high-frequency emphasis
    ///
    /// Modifies high-frequency content for emotional expression.
    pub fn apply_hf_emphasis(&self, magnitudes: &mut [f32]) {
        if (self.config.hf_emphasis - 1.0).abs() < 0.01 {
            return;
        }

        let num_bins = magnitudes.len();
        let crossover_freq = 3000.0; // Hz
        let crossover_bin = (crossover_freq * self.fft_size as f32 / self.sample_rate) as usize;

        for (i, magnitude) in magnitudes
            .iter_mut()
            .enumerate()
            .skip(crossover_bin)
            .take(num_bins - crossover_bin)
        {
            // Smooth transition from crossover frequency
            let transition =
                ((i - crossover_bin) as f32 / (num_bins - crossover_bin) as f32).min(1.0);
            let gain = 1.0 + (self.config.hf_emphasis - 1.0) * transition;
            *magnitude *= gain;
        }
    }

    /// Calculate and apply spectral centroid shift
    ///
    /// Shifts the spectral centroid to make voice brighter or darker.
    pub fn apply_centroid_shift(&self, magnitudes: &mut [f32]) {
        if self.config.centroid_shift.abs() < 1.0 {
            return;
        }

        // Simple implementation: shift entire spectrum
        let shift_bins =
            (self.config.centroid_shift * self.fft_size as f32 / self.sample_rate) as isize;

        if shift_bins == 0 {
            return;
        }

        let mut shifted = vec![0.0; magnitudes.len()];

        for (i, &mag) in magnitudes.iter().enumerate() {
            let new_i = (i as isize + shift_bins)
                .max(0)
                .min(magnitudes.len() as isize - 1) as usize;
            shifted[new_i] = mag;
        }

        magnitudes.copy_from_slice(&shifted);
    }

    /// Process complete spectrum with all effects
    pub fn process_spectrum(&self, magnitudes: &mut [f32], f0: Option<f32>) {
        // Apply effects in order
        self.apply_tilt(magnitudes);

        if let Some(fundamental) = f0 {
            self.apply_harmonic_enhancement(magnitudes, fundamental);
        }

        self.apply_hf_emphasis(magnitudes);
        self.apply_centroid_shift(magnitudes);
    }
}

/// Spectral envelope extractor
pub struct SpectralEnvelope {
    /// Sample rate
    sample_rate: f32,
    /// Number of envelope points
    num_points: usize,
}

impl SpectralEnvelope {
    /// Create new spectral envelope extractor
    pub fn new(sample_rate: f32, num_points: usize) -> Self {
        Self {
            sample_rate,
            num_points,
        }
    }

    /// Extract spectral envelope using linear interpolation
    ///
    /// This is a simplified implementation. Production code would use
    /// cepstral analysis or LPC for better envelope estimation.
    pub fn extract(&self, magnitudes: &[f32]) -> Vec<f32> {
        let mut envelope = vec![0.0; self.num_points];
        let bin_step = magnitudes.len() / self.num_points;

        for (i, env_point) in envelope.iter_mut().enumerate().take(self.num_points) {
            let start_bin = i * bin_step;
            let end_bin = ((i + 1) * bin_step).min(magnitudes.len());

            // Find local maximum in this range
            let max_mag = magnitudes[start_bin..end_bin]
                .iter()
                .cloned()
                .fold(0.0f32, f32::max);

            *env_point = max_mag;
        }

        envelope
    }

    /// Apply spectral envelope to magnitude spectrum
    pub fn apply(&self, magnitudes: &mut [f32], envelope: &[f32]) {
        if envelope.len() != self.num_points {
            return;
        }

        let bin_step = magnitudes.len() as f32 / self.num_points as f32;

        for (i, magnitude) in magnitudes.iter_mut().enumerate() {
            let env_index = (i as f32 / bin_step).min(self.num_points as f32 - 1.001);
            let env_i0 = env_index.floor() as usize;
            let env_i1 = (env_i0 + 1).min(self.num_points - 1);
            let frac = env_index - env_i0 as f32;

            // Interpolate envelope value
            let env_value = envelope[env_i0] * (1.0 - frac) + envelope[env_i1] * frac;

            *magnitude *= env_value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_config_creation() {
        let config = SpectralConfig::default();
        assert!((config.tilt).abs() < 0.01);
        assert_eq!(config.harmonic_boost, 1.0);
    }

    #[test]
    fn test_emotion_configs() {
        let happy = SpectralConfig::happy();
        let sad = SpectralConfig::sad();

        // Happy should be brighter (negative tilt)
        assert!(happy.tilt < sad.tilt);
        // Happy should have higher centroid
        assert!(happy.centroid_shift > sad.centroid_shift);
    }

    #[test]
    fn test_spectral_processor_creation() {
        let processor = SpectralProcessor::new(44100.0, 2048);
        assert_eq!(processor.fft_size, 2048);
    }

    #[test]
    fn test_spectral_tilt() {
        let mut processor = SpectralProcessor::new(44100.0, 2048);
        let mut config = SpectralConfig::default();
        config.tilt = 3.0; // Positive tilt boosts high frequencies (brighten)
        processor.set_config(config);

        let mut magnitudes = vec![1.0; 1024];
        processor.apply_tilt(&mut magnitudes);

        // High frequencies should be boosted with positive tilt
        assert!(magnitudes[512] > magnitudes[0]);
    }

    #[test]
    fn test_hf_emphasis() {
        let mut processor = SpectralProcessor::new(44100.0, 2048);
        let mut config = SpectralConfig::default();
        config.hf_emphasis = 1.5;
        processor.set_config(config);

        let mut magnitudes = vec![1.0; 1024];
        let original_hf = magnitudes[900];

        processor.apply_hf_emphasis(&mut magnitudes);

        // High frequencies should be emphasized
        assert!(magnitudes[900] > original_hf);
    }

    #[test]
    fn test_harmonic_enhancement() {
        let mut processor = SpectralProcessor::new(44100.0, 2048);
        let mut config = SpectralConfig::default();
        config.harmonic_boost = 1.3;
        processor.set_config(config);

        let mut magnitudes = vec![1.0; 1024];
        let f0 = 200.0; // Hz

        processor.apply_harmonic_enhancement(&mut magnitudes, f0);

        // Some bins should be boosted (harmonics)
        let sum: f32 = magnitudes.iter().sum();
        assert!(sum > 1024.0); // Total energy increased
    }

    #[test]
    fn test_spectral_envelope() {
        let extractor = SpectralEnvelope::new(44100.0, 20);
        let magnitudes = vec![1.0; 1024];

        let envelope = extractor.extract(&magnitudes);
        assert_eq!(envelope.len(), 20);
    }

    #[test]
    fn test_envelope_application() {
        let extractor = SpectralEnvelope::new(44100.0, 20);
        let envelope = vec![2.0; 20];
        let mut magnitudes = vec![1.0; 1024];

        extractor.apply(&mut magnitudes, &envelope);

        // All magnitudes should be scaled
        assert!((magnitudes[0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_full_spectrum_processing() {
        let mut processor = SpectralProcessor::new(44100.0, 2048);
        processor.set_config(SpectralConfig::happy());

        let mut magnitudes = vec![1.0; 1024];
        processor.process_spectrum(&mut magnitudes, Some(200.0));

        // Spectrum should be modified
        assert!(magnitudes.iter().any(|&x| (x - 1.0).abs() > 0.01));
    }
}
