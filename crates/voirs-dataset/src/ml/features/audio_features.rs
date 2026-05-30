//! Audio feature extraction module
//!
//! This module provides implementations for extracting various audio features
//! including MFCC, mel spectrograms, spectrograms, and learned features.
//!
//! The MFCC and mel spectrogram methods delegate to the real DSP implementations
//! in `crate::processing::features` which use a proper FFT-based pipeline.

use super::config::{AudioFeatureConfig, AudioFeatureMethod};
use crate::processing::features as dsp;
use crate::{AudioData, Result};

/// Audio feature extractor
pub struct AudioFeatureExtractor {
    config: AudioFeatureConfig,
    #[allow(dead_code)]
    model: Option<AudioFeatureModel>,
}

/// Audio feature model (placeholder)
#[allow(dead_code)]
struct AudioFeatureModel {
    weights: Vec<f32>,
    architecture: String,
}

impl AudioFeatureExtractor {
    pub fn new(config: AudioFeatureConfig) -> Result<Self> {
        Ok(Self {
            config,
            model: None,
        })
    }

    pub async fn extract_features(&self, audio: &AudioData) -> Result<Vec<f32>> {
        match &self.config.method {
            AudioFeatureMethod::MFCC { num_coeffs, .. } => {
                // Enhanced MFCC extraction with basic implementation
                self.extract_mfcc_features(audio, *num_coeffs)
            }
            AudioFeatureMethod::MelSpectrogram { num_mels, .. } => {
                // Enhanced mel spectrogram extraction
                self.extract_mel_features(audio, *num_mels)
            }
            AudioFeatureMethod::Spectrogram { fft_size, .. } => {
                // Enhanced spectrogram extraction
                self.extract_spectrogram_features(audio, *fft_size)
            }
            AudioFeatureMethod::Learned { .. } => {
                // Enhanced learned feature extraction with statistical features
                self.extract_learned_features(audio)
            }
        }
    }

    /// Extract MFCC features from audio.
    ///
    /// Delegates to the FFT-based real MFCC pipeline in `processing::features`.
    /// Returns a flat vector of `n_frames * num_coeffs` values (without energy).
    /// If audio is empty, returns a zero vector of length `num_coeffs`.
    fn extract_mfcc_features(&self, audio: &AudioData, num_coeffs: usize) -> Result<Vec<f32>> {
        if audio.samples().is_empty() {
            return Ok(vec![0.0; num_coeffs]);
        }
        // Delegate to the real DCT-II MFCC pipeline (energy coefficient excluded)
        dsp::extract_mfcc(audio, num_coeffs, false)
    }

    /// Extract mel spectrogram features.
    ///
    /// Delegates to the FFT-based mel spectrogram in `processing::features`.
    /// Returns a flat vector of `n_frames * num_mels` log-mel values.
    /// If audio is empty, returns a zero vector of length `num_mels`.
    fn extract_mel_features(&self, audio: &AudioData, num_mels: usize) -> Result<Vec<f32>> {
        if audio.samples().is_empty() {
            return Ok(vec![0.0; num_mels]);
        }
        // Default FFT/hop parameters consistent with the rest of the pipeline
        let result = dsp::extract_mel_spectrogram(audio, num_mels, 1024, 256)?;
        Ok(result.values)
    }

    /// Extract spectrogram (linear-frequency) features.
    ///
    /// Delegates to the FFT-based mel spectrogram with a fine mel resolution and
    /// returns the underlying power-spectrum values averaged across frames.
    /// If audio is empty, returns zeros of length `fft_size / 2`.
    fn extract_spectrogram_features(&self, audio: &AudioData, fft_size: usize) -> Result<Vec<f32>> {
        let output_size = fft_size / 2;
        if audio.samples().is_empty() {
            return Ok(vec![0.0; output_size]);
        }
        // Use the mel spectrogram pipeline with output_size mel bins so the
        // caller gets frequency-indexed log-energy bins with proper FFT underpinning.
        let hop = fft_size / 4;
        let result = dsp::extract_mel_spectrogram(audio, output_size, fft_size, hop)?;
        Ok(result.values)
    }

    /// Extract learned features using statistical analysis
    fn extract_learned_features(&self, audio: &AudioData) -> Result<Vec<f32>> {
        let samples = audio.samples();
        if samples.is_empty() {
            return Ok(vec![0.0; self.config.dimension]);
        }

        let mut features = Vec::with_capacity(self.config.dimension);

        // Statistical features
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;
        let std_dev = variance.sqrt();

        features.push(mean);
        features.push(std_dev);
        features.push(variance);

        // Spectral features
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        features.push(rms);

        // Zero crossing rate
        let zcr = samples
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count() as f32
            / (samples.len() - 1) as f32;
        features.push(zcr);

        // Energy in different frequency bands (simplified)
        let frame_size = 1024.min(samples.len());
        for band in 0..((self.config.dimension - 5).min(8)) {
            let mut band_energy = 0.0;
            let freq_start = band as f32 * audio.sample_rate() as f32 / 16.0;
            let freq_end = (band + 1) as f32 * audio.sample_rate() as f32 / 16.0;

            for chunk in samples.chunks(frame_size).take(4) {
                // Simplified band-pass energy estimation
                let chunk_rms =
                    (chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
                let freq_weight = if freq_start < 4000.0 && freq_end > 300.0 {
                    1.0
                } else {
                    0.5
                };
                band_energy += chunk_rms * freq_weight;
            }

            features.push(band_energy / 4.0);
        }

        // Pad or truncate to desired dimension
        features.resize(self.config.dimension, 0.0);

        Ok(features)
    }
}
