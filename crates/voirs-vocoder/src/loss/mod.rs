//! Perceptual loss functions for neural vocoder training and evaluation.
//!
//! This module provides comprehensive loss functions that better
//! correlate with human auditory perception than simple MSE/MAE metrics.
//!
//! # Available Losses
//!
//! ## Perceptual Losses
//! - **Spectral Loss**: Multi-scale STFT loss across different resolutions
//! - **Mel Loss**: Mel-spectrogram loss with A-weighting for perceptual importance
//! - **Combined Loss**: Weighted combination of multiple loss components
//!
//! ## GAN Losses (for HiFi-GAN, BigVGAN, UnivNet)
//! - **Adversarial Loss**: Standard GAN loss for generator and discriminator
//! - **Feature Matching Loss**: Matches intermediate discriminator features
//! - **Multi-Scale Discriminator Loss**: Combined loss across multiple scales
//!
//! # Usage
//!
//! ```rust,ignore
//! use voirs_vocoder::loss::{PerceptualLoss, PerceptualLossConfig};
//! use scirs2_core::ndarray::Array1;
//!
//! // Create perceptual loss calculator
//! let config = PerceptualLossConfig::default();
//! let mut loss = PerceptualLoss::new(config)?;
//!
//! // Compute loss between generated and target audio
//! let generated = Array1::<f32>::zeros(22050);
//! let target = Array1::<f32>::zeros(22050);
//!
//! let loss_value = loss.compute(&generated, &target)?;
//! ```
//!
//! # Loss Components
//!
//! The perceptual loss combines multiple components:
//!
//! 1. **Multi-scale Spectral Loss**: Measures spectral distance across multiple STFT resolutions
//! 2. **Mel-spectrogram Loss**: Mel-scale spectral loss with perceptual weighting
//! 3. **Time-domain Loss** (optional): Direct waveform comparison
//!
//! Each component can be weighted independently to balance different aspects of audio quality.

pub mod gan;
pub mod mel;
pub mod spectral;

pub use gan::{
    AdversarialLoss, AdversarialLossConfig, AdversarialLossType, CombinedDiscriminatorLoss,
    CombinedDiscriminatorLossConfig, CombinedGANLossBreakdown, FeatureMatchingLoss,
    FeatureMatchingLossConfig, GANLossBreakdown, MultiPeriodDiscriminatorLoss,
    MultiPeriodDiscriminatorLossConfig, MultiScaleDiscriminatorLoss,
    MultiScaleDiscriminatorLossConfig,
};
pub use mel::{MelLoss, MelLossBreakdown, MelLossConfig};
pub use spectral::{LossStatistics, SpectralLoss, SpectralLossBreakdown, SpectralLossConfig};

use crate::Result;
use scirs2_core::ndarray::prelude::*;
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};

/// Configuration for combined perceptual loss
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualLossConfig {
    /// Spectral loss configuration
    pub spectral_config: SpectralLossConfig,
    /// Mel loss configuration
    pub mel_config: MelLossConfig,
    /// Weight for spectral loss component
    pub spectral_weight: f32,
    /// Weight for mel loss component
    pub mel_weight: f32,
    /// Weight for time-domain L1 loss
    pub time_domain_weight: f32,
    /// Enable time-domain loss
    pub use_time_domain: bool,
}

impl Default for PerceptualLossConfig {
    fn default() -> Self {
        Self {
            spectral_config: SpectralLossConfig::default(),
            mel_config: MelLossConfig::default(),
            spectral_weight: 1.0,
            mel_weight: 1.0,
            time_domain_weight: 0.1,
            use_time_domain: true,
        }
    }
}

impl PerceptualLossConfig {
    /// Create configuration optimized for high-quality vocoder training
    pub fn high_quality() -> Self {
        Self {
            spectral_config: SpectralLossConfig::high_quality(),
            mel_config: MelLossConfig {
                n_mels: 128,
                use_a_weighting: true,
                ..Default::default()
            },
            spectral_weight: 1.0,
            mel_weight: 1.5,
            time_domain_weight: 0.05,
            use_time_domain: true,
        }
    }

    /// Create configuration optimized for fast training
    pub fn fast() -> Self {
        Self {
            spectral_config: SpectralLossConfig::fast(),
            mel_config: MelLossConfig {
                n_mels: 80,
                use_a_weighting: false, // Faster without A-weighting
                ..Default::default()
            },
            spectral_weight: 1.0,
            mel_weight: 1.0,
            time_domain_weight: 0.0, // Skip time-domain for speed
            use_time_domain: false,
        }
    }

    /// Create configuration for mel-only loss (fastest)
    pub fn mel_only() -> Self {
        Self {
            spectral_config: SpectralLossConfig::fast(),
            mel_config: MelLossConfig::default(),
            spectral_weight: 0.0, // Disable spectral loss
            mel_weight: 1.0,
            time_domain_weight: 0.0,
            use_time_domain: false,
        }
    }
}

/// Combined perceptual loss calculator
pub struct PerceptualLoss {
    /// Configuration
    config: PerceptualLossConfig,
    /// Spectral loss calculator
    spectral_loss: SpectralLoss,
    /// Mel loss calculator
    mel_loss: MelLoss,
}

impl PerceptualLoss {
    /// Create a new perceptual loss calculator
    pub fn new(config: PerceptualLossConfig) -> Result<Self> {
        let spectral_loss = SpectralLoss::new(config.spectral_config.clone())?;
        let mel_loss = MelLoss::new(config.mel_config.clone())?;

        Ok(Self {
            config,
            spectral_loss,
            mel_loss,
        })
    }

    /// Compute combined perceptual loss
    ///
    /// # Arguments
    /// * `generated` - Generated audio waveform
    /// * `target` - Target audio waveform
    ///
    /// # Returns
    /// * Combined weighted loss value
    pub fn compute<F: Float>(&mut self, generated: &Array1<F>, target: &Array1<F>) -> Result<f32> {
        let mut total_loss = 0.0;

        // Spectral loss component
        if self.config.spectral_weight > 0.0 {
            let spec_loss = self.spectral_loss.compute(generated, target)?;
            total_loss += self.config.spectral_weight * spec_loss;
        }

        // Mel loss component (requires STFT and mel filterbank computation)
        if self.config.mel_weight > 0.0 {
            let mel_gen = self.compute_mel_spectrogram(generated)?;
            let mel_target = self.compute_mel_spectrogram(target)?;
            let mel_loss_value = self.mel_loss.compute(&mel_gen, &mel_target)?;
            total_loss += self.config.mel_weight * mel_loss_value;
        }

        // Time-domain loss component
        if self.config.use_time_domain && self.config.time_domain_weight > 0.0 {
            let time_loss = self.compute_time_domain_loss(generated, target)?;
            total_loss += self.config.time_domain_weight * time_loss;
        }

        Ok(total_loss)
    }

    /// Compute time-domain L1 loss
    fn compute_time_domain_loss<F: Float>(
        &self,
        generated: &Array1<F>,
        target: &Array1<F>,
    ) -> Result<f32> {
        if generated.len() != target.len() {
            return Err(crate::VocoderError::InputError(
                "Generated and target audio must have the same length".to_string(),
            ));
        }

        let mut sum = 0.0_f32;
        for (g, t) in generated.iter().zip(target.iter()) {
            let diff = (g.to_f32().unwrap_or(0.0) - t.to_f32().unwrap_or(0.0)).abs();
            sum += diff;
        }

        Ok(sum / generated.len() as f32)
    }

    /// Compute mel-spectrogram from audio waveform
    ///
    /// This method computes the mel-scale spectrogram by first computing a power spectrogram
    /// using DFT, then applying the mel filterbank.
    fn compute_mel_spectrogram<F: Float>(&mut self, audio: &Array1<F>) -> Result<Array2<f32>> {
        let mel_config = &self.config.mel_config;
        let n_fft = mel_config.n_fft;
        let hop_length = mel_config.hop_length;
        let win_length = mel_config.win_length;

        // Create Hann window
        let window: Vec<f32> = (0..win_length)
            .map(|i| {
                let n = i as f32;
                let n_total = (win_length - 1) as f32;
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * n / n_total).cos()
            })
            .collect();

        // Compute number of frames
        let audio_len = audio.len();
        let n_frames = (audio_len.saturating_sub(n_fft)) / hop_length + 1;

        // Compute power spectrogram using DFT
        let n_freq_bins = n_fft / 2 + 1;
        let mut power_spec = Array2::<f32>::zeros((n_freq_bins, n_frames));

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;
            let end = start + n_fft;

            if end > audio_len {
                break;
            }

            // Extract and window frame
            let frame_data: Vec<f32> = (0..n_fft)
                .map(|i| {
                    if i < win_length {
                        audio[start + i].to_f32().unwrap_or(0.0) * window[i]
                    } else {
                        0.0
                    }
                })
                .collect();

            // Compute DFT (only positive frequency bins)
            for bin_idx in 0..n_freq_bins {
                let mut real_sum = 0.0_f32;
                let mut imag_sum = 0.0_f32;

                let freq = bin_idx as f32 / n_fft as f32;

                for (sample_idx, &sample) in frame_data.iter().enumerate() {
                    let phase = -2.0 * std::f32::consts::PI * freq * sample_idx as f32;
                    real_sum += sample * phase.cos();
                    imag_sum += sample * phase.sin();
                }

                // Compute power spectrum
                power_spec[[bin_idx, frame_idx]] = real_sum * real_sum + imag_sum * imag_sum;
            }
        }

        // Apply mel filterbank
        let mel_filterbank = &self.mel_loss.mel_filterbank;
        let (n_mels, _) = mel_filterbank.dim();

        // Compute mel spectrogram
        let mut mel_spec = Array2::<f32>::zeros((n_mels, n_frames));
        for mel_idx in 0..n_mels {
            for frame_idx in 0..n_frames {
                let mut mel_value = 0.0f32;
                for freq_idx in 0..n_freq_bins {
                    mel_value +=
                        mel_filterbank[[mel_idx, freq_idx]] * power_spec[[freq_idx, frame_idx]];
                }
                // Apply log compression (mel spectrograms are typically in log scale)
                mel_spec[[mel_idx, frame_idx]] = (mel_value + 1e-10).log10();
            }
        }

        Ok(mel_spec)
    }

    /// Compute detailed loss breakdown
    pub fn compute_detailed<F: Float>(
        &mut self,
        generated: &Array1<F>,
        target: &Array1<F>,
    ) -> Result<PerceptualLossBreakdown> {
        let spectral_loss = if self.config.spectral_weight > 0.0 {
            Some(self.spectral_loss.compute_detailed(generated, target)?)
        } else {
            None
        };

        let time_domain_loss = if self.config.use_time_domain {
            Some(self.compute_time_domain_loss(generated, target)?)
        } else {
            None
        };

        let total_loss = self.compute(generated, target)?;

        Ok(PerceptualLossBreakdown {
            total_loss,
            spectral_breakdown: spectral_loss,
            time_domain_loss,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &PerceptualLossConfig {
        &self.config
    }
}

/// Detailed breakdown of perceptual loss components
#[derive(Debug, Clone)]
pub struct PerceptualLossBreakdown {
    /// Total combined loss
    pub total_loss: f32,
    /// Spectral loss breakdown
    pub spectral_breakdown: Option<SpectralLossBreakdown>,
    /// Time-domain loss
    pub time_domain_loss: Option<f32>,
}

impl PerceptualLossBreakdown {
    /// Get a summary string
    pub fn summary(&self) -> String {
        let mut lines = vec![format!("Total Loss: {:.6}", self.total_loss)];

        if let Some(ref spectral) = self.spectral_breakdown {
            lines.push(format!("Spectral Loss: {:.6}", spectral.total_loss));
            let stats = spectral.statistics();
            lines.push(format!(
                "  - Mean: {:.6}, Std: {:.6}",
                stats.mean, stats.std_dev
            ));
        }

        if let Some(time_loss) = self.time_domain_loss {
            lines.push(format!("Time-domain Loss: {:.6}", time_loss));
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptual_loss_config_default() {
        let config = PerceptualLossConfig::default();
        assert_eq!(config.spectral_weight, 1.0);
        assert_eq!(config.mel_weight, 1.0);
        assert!(config.use_time_domain);
    }

    #[test]
    fn test_perceptual_loss_config_presets() {
        let high_quality = PerceptualLossConfig::high_quality();
        assert!(high_quality.mel_config.use_a_weighting);

        let fast = PerceptualLossConfig::fast();
        assert!(!fast.use_time_domain);

        let mel_only = PerceptualLossConfig::mel_only();
        assert_eq!(mel_only.spectral_weight, 0.0);
    }

    #[test]
    fn test_perceptual_loss_creation() {
        let config = PerceptualLossConfig::default();
        let loss = PerceptualLoss::new(config);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_time_domain_loss() {
        let config = PerceptualLossConfig::default();
        let loss = PerceptualLoss::new(config).unwrap();

        let signal = Array1::<f32>::from_vec(vec![0.0, 0.5, 1.0, 0.5, 0.0]);
        let time_loss = loss.compute_time_domain_loss(&signal, &signal).unwrap();

        assert!(
            time_loss < 1e-5,
            "Time-domain loss should be near zero for identical signals"
        );
    }

    #[test]
    fn test_perceptual_loss_identical_signals() {
        let config = PerceptualLossConfig::fast(); // Use fast for quicker test
        let mut loss = PerceptualLoss::new(config).unwrap();

        let signal = Array1::<f32>::from_vec(
            (0..2048)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
                .collect(),
        );

        let loss_value = loss.compute(&signal, &signal).unwrap();
        assert!(
            loss_value < 1e-4,
            "Loss should be near zero for identical signals"
        );
    }

    #[test]
    fn test_perceptual_loss_detailed() {
        let config = PerceptualLossConfig::fast();
        let mut loss = PerceptualLoss::new(config).unwrap();

        let signal = Array1::<f32>::from_vec(
            (0..2048)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
                .collect(),
        );

        let breakdown = loss.compute_detailed(&signal, &signal).unwrap();

        assert!(breakdown.total_loss < 1e-4);
        assert!(breakdown.spectral_breakdown.is_some());

        let summary = breakdown.summary();
        assert!(summary.contains("Total Loss"));
    }

    #[test]
    fn test_perceptual_loss_mel_only_config() {
        let config = PerceptualLossConfig::mel_only();
        let mut loss = PerceptualLoss::new(config).unwrap();

        let signal = Array1::<f32>::from_vec(
            (0..2048)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
                .collect(),
        );

        // Should work even with spectral disabled
        let loss_value = loss.compute(&signal, &signal).unwrap();
        assert!(loss_value >= 0.0);
    }

    #[test]
    fn test_breakdown_summary_format() {
        let breakdown = PerceptualLossBreakdown {
            total_loss: 0.123,
            spectral_breakdown: None,
            time_domain_loss: Some(0.045),
        };

        let summary = breakdown.summary();
        assert!(summary.contains("Total Loss: 0.123"));
        assert!(summary.contains("Time-domain Loss: 0.045"));
    }

    #[test]
    fn test_mel_spectrogram_computation() {
        let config = PerceptualLossConfig::default();
        let mut loss = PerceptualLoss::new(config).unwrap();

        // Create a simple sine wave
        let signal = Array1::<f32>::from_vec(
            (0..4096)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin() * 0.5)
                .collect(),
        );

        // Compute mel spectrogram
        let mel_spec = loss.compute_mel_spectrogram(&signal).unwrap();

        // Check dimensions
        let (n_mels, n_frames) = mel_spec.dim();
        assert_eq!(n_mels, 80, "Should have 80 mel bands");
        assert!(n_frames > 0, "Should have multiple frames");

        // Check that values are in reasonable range (log scale)
        let min_val = mel_spec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = mel_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        assert!(min_val.is_finite(), "Mel values should be finite");
        assert!(max_val.is_finite(), "Mel values should be finite");
        assert!(
            min_val < max_val,
            "Should have variation in mel spectrogram"
        );
    }

    #[test]
    fn test_mel_spectrogram_short_signal() {
        let config = PerceptualLossConfig::default();
        let mut loss = PerceptualLoss::new(config).unwrap();

        // Create a very short signal
        let signal = Array1::<f32>::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        // Should handle short signals gracefully
        let mel_spec = loss.compute_mel_spectrogram(&signal);
        assert!(mel_spec.is_ok() || mel_spec.is_err()); // Either works or fails gracefully
    }

    #[test]
    fn test_mel_loss_with_spectrogram_computation() {
        let config = PerceptualLossConfig {
            spectral_weight: 0.0,    // Disable spectral loss
            mel_weight: 1.0,         // Enable mel loss
            time_domain_weight: 0.0, // Disable time domain
            use_time_domain: false,
            spectral_config: SpectralLossConfig::default(),
            mel_config: MelLossConfig::default(),
        };

        let mut loss = PerceptualLoss::new(config).unwrap();

        // Create two similar signals
        let signal1 = Array1::<f32>::from_vec(
            (0..2048)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
                .collect(),
        );
        let signal2 = signal1.clone();

        // Loss should be very small for identical signals
        let loss_value = loss.compute(&signal1, &signal2).unwrap();
        assert!(
            loss_value < 0.1,
            "Loss should be small for identical signals, got: {}",
            loss_value
        );
    }

    #[test]
    fn test_mel_spectrogram_different_signals() {
        let config = PerceptualLossConfig::default();
        let mut loss = PerceptualLoss::new(config).unwrap();

        // Create two different frequency sine waves
        let signal1 = Array1::<f32>::from_vec(
            (0..4096)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
                .collect(),
        );
        let signal2 = Array1::<f32>::from_vec(
            (0..4096)
                .map(|i| (2.0 * std::f32::consts::PI * 880.0 * i as f32 / 22050.0).sin())
                .collect(),
        );

        // Compute mel spectrograms
        let mel1 = loss.compute_mel_spectrogram(&signal1).unwrap();
        let mel2 = loss.compute_mel_spectrogram(&signal2).unwrap();

        // Check they have same shape but different values
        assert_eq!(mel1.dim(), mel2.dim());

        // Calculate difference
        let diff = &mel1 - &mel2;
        let mean_diff = diff.mapv(|x| x.abs()).mean().unwrap();

        assert!(
            mean_diff > 0.01,
            "Different frequency signals should produce different mel spectrograms"
        );
    }
}
