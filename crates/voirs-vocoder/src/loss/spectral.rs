//! Multi-scale spectral loss functions for audio quality assessment.
//!
//! These losses measure the spectral distance between generated and target audio
//! across multiple scales, providing perceptually meaningful quality metrics.

use crate::Result;
use scirs2_core::ndarray::prelude::*;
use scirs2_core::numeric::Float;
use scirs2_fft::FftPlanner;
use serde::{Deserialize, Serialize};

/// Configuration for multi-scale spectral loss
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralLossConfig {
    /// FFT window sizes to use
    pub fft_sizes: Vec<usize>,
    /// Hop lengths for each FFT size
    pub hop_lengths: Vec<usize>,
    /// Window lengths for each FFT size
    pub win_lengths: Vec<usize>,
    /// Weight for magnitude loss component
    pub magnitude_weight: f32,
    /// Weight for log-magnitude loss component
    pub log_magnitude_weight: f32,
    /// Epsilon for numerical stability in log
    pub epsilon: f32,
}

impl Default for SpectralLossConfig {
    fn default() -> Self {
        Self {
            // Multi-scale: 64, 128, 256, 512, 1024, 2048
            fft_sizes: vec![64, 128, 256, 512, 1024, 2048],
            hop_lengths: vec![16, 32, 64, 128, 256, 512],
            win_lengths: vec![64, 128, 256, 512, 1024, 2048],
            magnitude_weight: 1.0,
            log_magnitude_weight: 1.0,
            epsilon: 1e-7,
        }
    }
}

impl SpectralLossConfig {
    /// Create configuration for high-quality vocoder training
    pub fn high_quality() -> Self {
        Self {
            fft_sizes: vec![128, 256, 512, 1024, 2048, 4096],
            hop_lengths: vec![32, 64, 128, 256, 512, 1024],
            win_lengths: vec![128, 256, 512, 1024, 2048, 4096],
            magnitude_weight: 1.0,
            log_magnitude_weight: 1.0,
            epsilon: 1e-7,
        }
    }

    /// Create configuration optimized for fast training
    pub fn fast() -> Self {
        Self {
            fft_sizes: vec![512, 1024, 2048],
            hop_lengths: vec![128, 256, 512],
            win_lengths: vec![512, 1024, 2048],
            magnitude_weight: 1.0,
            log_magnitude_weight: 1.0,
            epsilon: 1e-7,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.fft_sizes.len() != self.hop_lengths.len()
            || self.fft_sizes.len() != self.win_lengths.len()
        {
            return Err(crate::VocoderError::ConfigError(
                "FFT sizes, hop lengths, and window lengths must have the same length".to_string(),
            ));
        }

        if self.fft_sizes.is_empty() {
            return Err(crate::VocoderError::ConfigError(
                "At least one FFT size must be specified".to_string(),
            ));
        }

        for i in 0..self.fft_sizes.len() {
            if self.hop_lengths[i] > self.fft_sizes[i] {
                return Err(crate::VocoderError::ConfigError(format!(
                    "Hop length {} exceeds FFT size {} at index {}",
                    self.hop_lengths[i], self.fft_sizes[i], i
                )));
            }
        }

        Ok(())
    }
}

/// Multi-scale spectral loss calculator
pub struct SpectralLoss {
    /// Configuration
    config: SpectralLossConfig,
}

impl SpectralLoss {
    /// Create a new spectral loss calculator
    pub fn new(config: SpectralLossConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self { config })
    }

    /// Compute spectral loss between generated and target audio
    ///
    /// # Arguments
    /// * `generated` - Generated audio waveform
    /// * `target` - Target audio waveform
    ///
    /// # Returns
    /// * Total spectral loss value
    pub fn compute<F: Float>(&mut self, generated: &Array1<F>, target: &Array1<F>) -> Result<f32> {
        if generated.len() != target.len() {
            return Err(crate::VocoderError::InputError(format!(
                "Generated and target audio must have the same length: {} vs {}",
                generated.len(),
                target.len()
            )));
        }

        let mut total_loss = 0.0;
        let num_scales = self.config.fft_sizes.len();

        // Compute loss at each scale
        for i in 0..num_scales {
            let fft_size = self.config.fft_sizes[i];
            let hop_length = self.config.hop_lengths[i];
            let win_length = self.config.win_lengths[i];

            // Compute STFT for both signals
            let stft_gen = self.compute_stft(generated, fft_size, hop_length, win_length)?;
            let stft_target = self.compute_stft(target, fft_size, hop_length, win_length)?;

            // Compute magnitude loss
            let mag_loss = self.magnitude_loss(&stft_gen, &stft_target);

            // Compute log-magnitude loss
            let log_mag_loss = self.log_magnitude_loss(&stft_gen, &stft_target);

            // Weighted combination
            let scale_loss = self.config.magnitude_weight * mag_loss
                + self.config.log_magnitude_weight * log_mag_loss;

            total_loss += scale_loss;
        }

        // Average across scales
        Ok(total_loss / num_scales as f32)
    }

    /// Compute STFT magnitude spectrogram
    fn compute_stft<F: Float>(
        &mut self,
        audio: &Array1<F>,
        fft_size: usize,
        hop_length: usize,
        win_length: usize,
    ) -> Result<Array2<f32>> {
        if audio.len() < fft_size {
            return Err(crate::VocoderError::InputError(format!(
                "Audio length {} is less than FFT size {}",
                audio.len(),
                fft_size
            )));
        }

        let num_frames = (audio.len() - fft_size) / hop_length + 1;
        let num_bins = fft_size / 2 + 1;

        let mut spectrogram = Array2::<f32>::zeros((num_bins, num_frames));

        // Hann window
        let window = self.create_hann_window(win_length);

        // Process each frame
        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_length;
            let end = start + fft_size;

            if end > audio.len() {
                break;
            }

            // Extract and window frame
            let frame_data: Vec<f32> = (0..fft_size)
                .map(|i| {
                    if i < win_length {
                        audio[start + i].to_f32().unwrap_or(0.0) * window[i]
                    } else {
                        0.0
                    }
                })
                .collect();

            // Compute DFT (simplified for first implementation)
            // This computes only the positive frequency bins
            for bin_idx in 0..num_bins {
                let mut real_sum = 0.0_f32;
                let mut imag_sum = 0.0_f32;

                let freq = bin_idx as f32 / fft_size as f32;

                for (sample_idx, &sample) in frame_data.iter().enumerate() {
                    let phase = -2.0 * std::f32::consts::PI * freq * sample_idx as f32;
                    real_sum += sample * phase.cos();
                    imag_sum += sample * phase.sin();
                }

                // Compute magnitude
                spectrogram[[bin_idx, frame_idx]] =
                    (real_sum * real_sum + imag_sum * imag_sum).sqrt();
            }
        }

        Ok(spectrogram)
    }

    /// Create Hann window
    fn create_hann_window(&self, length: usize) -> Vec<f32> {
        if length == 0 {
            return vec![];
        }
        if length == 1 {
            return vec![1.0];
        }

        (0..length)
            .map(|i| {
                // Hann window: 0.5 - 0.5 * cos(2π * n / (N-1))
                let n = i as f32;
                let n_total = (length - 1) as f32;
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * n / n_total).cos()
            })
            .collect()
    }

    /// Compute L1 loss between magnitude spectrograms
    fn magnitude_loss(&self, generated: &Array2<f32>, target: &Array2<f32>) -> f32 {
        let diff = generated - target;
        diff.mapv(|x| x.abs()).mean().unwrap_or(0.0)
    }

    /// Compute L1 loss between log-magnitude spectrograms
    fn log_magnitude_loss(&self, generated: &Array2<f32>, target: &Array2<f32>) -> f32 {
        let log_gen = generated.mapv(|x| (x + self.config.epsilon).ln());
        let log_target = target.mapv(|x| (x + self.config.epsilon).ln());

        let diff = &log_gen - &log_target;
        diff.mapv(|x| x.abs()).mean().unwrap_or(0.0)
    }

    /// Compute detailed loss breakdown for analysis
    pub fn compute_detailed<F: Float>(
        &mut self,
        generated: &Array1<F>,
        target: &Array1<F>,
    ) -> Result<SpectralLossBreakdown> {
        if generated.len() != target.len() {
            return Err(crate::VocoderError::InputError(
                "Generated and target audio must have the same length".to_string(),
            ));
        }

        let mut scale_losses = Vec::new();
        let mut magnitude_losses = Vec::new();
        let mut log_magnitude_losses = Vec::new();

        for i in 0..self.config.fft_sizes.len() {
            let fft_size = self.config.fft_sizes[i];
            let hop_length = self.config.hop_lengths[i];
            let win_length = self.config.win_lengths[i];

            let stft_gen = self.compute_stft(generated, fft_size, hop_length, win_length)?;
            let stft_target = self.compute_stft(target, fft_size, hop_length, win_length)?;

            let mag_loss = self.magnitude_loss(&stft_gen, &stft_target);
            let log_mag_loss = self.log_magnitude_loss(&stft_gen, &stft_target);

            magnitude_losses.push(mag_loss);
            log_magnitude_losses.push(log_mag_loss);

            let scale_loss = self.config.magnitude_weight * mag_loss
                + self.config.log_magnitude_weight * log_mag_loss;
            scale_losses.push(scale_loss);
        }

        let total_loss = scale_losses.iter().sum::<f32>() / scale_losses.len() as f32;

        Ok(SpectralLossBreakdown {
            total_loss,
            scale_losses,
            magnitude_losses,
            log_magnitude_losses,
            fft_sizes: self.config.fft_sizes.clone(),
        })
    }
}

/// Detailed breakdown of spectral loss components
#[derive(Debug, Clone)]
pub struct SpectralLossBreakdown {
    /// Total combined loss
    pub total_loss: f32,
    /// Loss at each scale
    pub scale_losses: Vec<f32>,
    /// Magnitude loss at each scale
    pub magnitude_losses: Vec<f32>,
    /// Log-magnitude loss at each scale
    pub log_magnitude_losses: Vec<f32>,
    /// FFT sizes used
    pub fft_sizes: Vec<usize>,
}

impl SpectralLossBreakdown {
    /// Get loss statistics
    pub fn statistics(&self) -> LossStatistics {
        let min_loss = self
            .scale_losses
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let max_loss = self
            .scale_losses
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mean_loss = self.total_loss;

        // Compute standard deviation
        let variance: f32 = self
            .scale_losses
            .iter()
            .map(|&x| (x - mean_loss).powi(2))
            .sum::<f32>()
            / self.scale_losses.len() as f32;
        let std_dev = variance.sqrt();

        LossStatistics {
            min: min_loss,
            max: max_loss,
            mean: mean_loss,
            std_dev,
        }
    }
}

/// Statistical summary of losses
#[derive(Debug, Clone, Copy)]
pub struct LossStatistics {
    /// Minimum loss value
    pub min: f32,
    /// Maximum loss value
    pub max: f32,
    /// Mean loss value
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_loss_config_default() {
        let config = SpectralLossConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.fft_sizes.len(), 6);
        assert_eq!(config.magnitude_weight, 1.0);
    }

    #[test]
    fn test_spectral_loss_config_validation() {
        let mut config = SpectralLossConfig::default();
        config.hop_lengths.pop(); // Create length mismatch
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_spectral_loss_creation() {
        let config = SpectralLossConfig::fast();
        let loss = SpectralLoss::new(config);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_spectral_loss_identical_signals() {
        let config = SpectralLossConfig::fast();
        let mut loss = SpectralLoss::new(config).unwrap();

        // Create identical signals (longer than largest FFT size of 2048)
        let signal = Array1::<f32>::from_vec(
            (0..4096)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
                .collect(),
        );

        let loss_value = loss.compute(&signal, &signal).unwrap();
        assert!(
            loss_value < 1e-5,
            "Loss should be near zero for identical signals: {}",
            loss_value
        );
    }

    #[test]
    fn test_spectral_loss_different_signals() {
        let config = SpectralLossConfig::fast();
        let mut loss = SpectralLoss::new(config).unwrap();

        // Create different signals (longer than largest FFT size of 2048)
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

        let loss_value = loss.compute(&signal1, &signal2).unwrap();
        assert!(
            loss_value > 0.0,
            "Loss should be positive for different signals"
        );
    }

    #[test]
    fn test_spectral_loss_length_mismatch() {
        let config = SpectralLossConfig::fast();
        let mut loss = SpectralLoss::new(config).unwrap();

        let signal1 = Array1::<f32>::zeros(1024);
        let signal2 = Array1::<f32>::zeros(2048);

        let result = loss.compute(&signal1, &signal2);
        assert!(result.is_err());
    }

    #[test]
    fn test_spectral_loss_detailed() {
        let config = SpectralLossConfig::fast();
        let mut loss = SpectralLoss::new(config.clone()).unwrap();

        let signal = Array1::<f32>::from_vec(
            (0..2048)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
                .collect(),
        );

        let breakdown = loss.compute_detailed(&signal, &signal).unwrap();

        assert_eq!(breakdown.fft_sizes.len(), config.fft_sizes.len());
        assert_eq!(breakdown.scale_losses.len(), config.fft_sizes.len());
        assert!(breakdown.total_loss < 1e-5);

        let stats = breakdown.statistics();
        assert!(stats.mean < 1e-5);
    }

    #[test]
    fn test_hann_window() {
        let config = SpectralLossConfig::default();
        let loss = SpectralLoss::new(config).unwrap();

        let window = loss.create_hann_window(512);
        assert_eq!(window.len(), 512);

        // Window should start and end near zero
        assert!(window[0] < 0.01);
        assert!(window[511] < 0.01);

        // Window should peak near middle
        assert!(window[256] > 0.99);
    }

    #[test]
    fn test_magnitude_loss_symmetry() {
        let config = SpectralLossConfig::fast();
        let loss = SpectralLoss::new(config).unwrap();

        let spec1 = Array2::<f32>::from_shape_fn((257, 10), |(i, j)| (i + j) as f32);
        let spec2 = Array2::<f32>::from_shape_fn((257, 10), |(i, j)| (i + j + 5) as f32);

        let loss1 = loss.magnitude_loss(&spec1, &spec2);
        let loss2 = loss.magnitude_loss(&spec2, &spec1);

        assert!(
            (loss1 - loss2).abs() < 1e-5,
            "Magnitude loss should be symmetric"
        );
    }
}
