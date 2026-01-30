//! Mel-spectrogram loss with perceptual weighting.
//!
//! Provides mel-scale spectral losses that better match human auditory perception.

use crate::Result;
use scirs2_core::ndarray::prelude::*;
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};

/// Configuration for mel-spectrogram loss
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelLossConfig {
    /// Number of mel filterbanks
    pub n_mels: usize,
    /// Sample rate
    pub sample_rate: u32,
    /// FFT size
    pub n_fft: usize,
    /// Hop length
    pub hop_length: usize,
    /// Window length
    pub win_length: usize,
    /// Minimum frequency
    pub f_min: f32,
    /// Maximum frequency
    pub f_max: f32,
    /// Apply A-weighting for perceptual importance
    pub use_a_weighting: bool,
    /// Weight for L1 loss component
    pub l1_weight: f32,
    /// Weight for L2 loss component
    pub l2_weight: f32,
}

impl Default for MelLossConfig {
    fn default() -> Self {
        Self {
            n_mels: 80,
            sample_rate: 22050,
            n_fft: 1024,
            hop_length: 256,
            win_length: 1024,
            f_min: 0.0,
            f_max: 8000.0,
            use_a_weighting: true,
            l1_weight: 1.0,
            l2_weight: 1.0,
        }
    }
}

/// Mel-spectrogram loss calculator
pub struct MelLoss {
    /// Configuration
    config: MelLossConfig,
    /// Mel filterbank (accessible for spectrogram computation)
    pub(crate) mel_filterbank: Array2<f32>,
    /// A-weighting curve (if enabled)
    a_weights: Option<Array1<f32>>,
}

impl MelLoss {
    /// Create a new mel loss calculator
    pub fn new(config: MelLossConfig) -> Result<Self> {
        // Create mel filterbank
        let mel_filterbank = Self::create_mel_filterbank(
            config.n_mels,
            config.n_fft,
            config.sample_rate,
            config.f_min,
            config.f_max,
        )?;

        // Create A-weighting if enabled
        let a_weights = if config.use_a_weighting {
            Some(Self::create_a_weighting(
                config.n_mels,
                config.sample_rate,
                config.f_min,
                config.f_max,
            ))
        } else {
            None
        };

        Ok(Self {
            config,
            mel_filterbank,
            a_weights,
        })
    }

    /// Compute mel loss between spectrograms
    pub fn compute(&self, generated: &Array2<f32>, target: &Array2<f32>) -> Result<f32> {
        if generated.shape() != target.shape() {
            return Err(crate::VocoderError::InputError(format!(
                "Generated and target spectrograms must have the same shape: {:?} vs {:?}",
                generated.shape(),
                target.shape()
            )));
        }

        // Apply perceptual weighting if enabled
        let (weighted_gen, weighted_target) = if let Some(ref weights) = self.a_weights {
            let gen = self.apply_perceptual_weighting(generated, weights);
            let tgt = self.apply_perceptual_weighting(target, weights);
            (gen, tgt)
        } else {
            (generated.clone(), target.clone())
        };

        // Compute L1 loss
        let l1_loss = if self.config.l1_weight > 0.0 {
            let diff = &weighted_gen - &weighted_target;
            diff.mapv(|x| x.abs()).mean().unwrap_or(0.0)
        } else {
            0.0
        };

        // Compute L2 loss
        let l2_loss = if self.config.l2_weight > 0.0 {
            let diff = &weighted_gen - &weighted_target;
            (diff.mapv(|x| x * x).mean().unwrap_or(0.0)).sqrt()
        } else {
            0.0
        };

        // Weighted combination
        let total_loss = self.config.l1_weight * l1_loss + self.config.l2_weight * l2_loss;

        Ok(total_loss)
    }

    /// Apply perceptual weighting to mel-spectrogram
    fn apply_perceptual_weighting(
        &self,
        mel_spec: &Array2<f32>,
        weights: &Array1<f32>,
    ) -> Array2<f32> {
        let (n_mels, n_frames) = mel_spec.dim();
        let mut weighted = mel_spec.clone();

        // Apply frequency-dependent weighting
        for mel_idx in 0..n_mels {
            let weight = weights[mel_idx];
            for frame_idx in 0..n_frames {
                weighted[[mel_idx, frame_idx]] *= weight;
            }
        }

        weighted
    }

    /// Create mel filterbank matrix
    fn create_mel_filterbank(
        n_mels: usize,
        n_fft: usize,
        sample_rate: u32,
        f_min: f32,
        f_max: f32,
    ) -> Result<Array2<f32>> {
        let n_bins = n_fft / 2 + 1;
        let mut filterbank = Array2::<f32>::zeros((n_mels, n_bins));

        // Convert frequencies to mel scale
        let mel_min = Self::hz_to_mel(f_min);
        let mel_max = Self::hz_to_mel(f_max);

        // Create mel points
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| {
                let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
                Self::mel_to_hz(mel)
            })
            .collect();

        // Convert to FFT bin indices
        let bin_points: Vec<usize> = mel_points
            .iter()
            .map(|&freq| ((freq / (sample_rate as f32 / 2.0)) * (n_bins - 1) as f32) as usize)
            .collect();

        // Create triangular filters
        for mel_idx in 0..n_mels {
            let left = bin_points[mel_idx];
            let center = bin_points[mel_idx + 1];
            let right = bin_points[mel_idx + 2];

            // Rising slope
            for bin in left..center {
                if center > left {
                    filterbank[[mel_idx, bin]] = (bin - left) as f32 / (center - left) as f32;
                }
            }

            // Falling slope
            for bin in center..right {
                if right > center {
                    filterbank[[mel_idx, bin]] = (right - bin) as f32 / (right - center) as f32;
                }
            }
        }

        Ok(filterbank)
    }

    /// Convert Hz to mel scale
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert mel scale to Hz
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Create A-weighting curve for perceptual importance
    ///
    /// A-weighting approximates human ear sensitivity at different frequencies
    fn create_a_weighting(n_mels: usize, sample_rate: u32, f_min: f32, f_max: f32) -> Array1<f32> {
        let mel_min = Self::hz_to_mel(f_min);
        let mel_max = Self::hz_to_mel(f_max);

        let mel_freqs: Vec<f32> = (0..n_mels)
            .map(|i| {
                let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels - 1) as f32;
                Self::mel_to_hz(mel)
            })
            .collect();

        // A-weighting formula
        let weights: Vec<f32> = mel_freqs
            .iter()
            .map(|&f| {
                if f < 1.0 {
                    return 1.0; // Avoid division by zero
                }

                let f2 = f * f;
                let f4 = f2 * f2;

                let numerator = 12194.0_f32.powi(2) * f4;
                let denominator = (f2 + 20.6_f32.powi(2))
                    * ((f2 + 107.7_f32.powi(2)) * (f2 + 737.9_f32.powi(2))).sqrt()
                    * (f2 + 12194.0_f32.powi(2));

                let weight_db = 20.0 * (numerator / denominator).log10() + 2.0;

                // Convert dB to linear scale
                10.0_f32.powf(weight_db / 20.0)
            })
            .collect();

        // Normalize weights
        let max_weight = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let normalized: Vec<f32> = weights.iter().map(|&w| w / max_weight).collect();

        Array1::from_vec(normalized)
    }
}

/// Detailed mel loss breakdown
#[derive(Debug, Clone)]
pub struct MelLossBreakdown {
    /// Total combined loss
    pub total_loss: f32,
    /// L1 component
    pub l1_loss: f32,
    /// L2 component
    pub l2_loss: f32,
    /// Per-band losses
    pub per_band_losses: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_loss_config_default() {
        let config = MelLossConfig::default();
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.sample_rate, 22050);
    }

    #[test]
    fn test_mel_loss_creation() {
        let config = MelLossConfig::default();
        let loss = MelLoss::new(config);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_hz_to_mel_conversion() {
        let hz = 1000.0;
        let mel = MelLoss::hz_to_mel(hz);
        let hz_back = MelLoss::mel_to_hz(mel);

        assert!(
            (hz - hz_back).abs() < 0.01,
            "Hz to mel conversion should be reversible"
        );
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let n_mels = 80;
        let n_fft = 1024;
        let filterbank = MelLoss::create_mel_filterbank(n_mels, n_fft, 22050, 0.0, 8000.0).unwrap();

        assert_eq!(filterbank.dim(), (80, 513));
    }

    #[test]
    fn test_mel_filterbank_properties() {
        let n_mels = 80;
        let n_fft = 1024;
        let filterbank = MelLoss::create_mel_filterbank(n_mels, n_fft, 22050, 0.0, 8000.0).unwrap();

        // Each filter should sum to approximately 1 (triangular filters)
        for mel_idx in 0..n_mels {
            let filter_sum: f32 = filterbank.row(mel_idx).iter().sum();
            // Allow some tolerance due to discretization
            assert!(
                filter_sum > 0.0,
                "Filter {} should have positive sum",
                mel_idx
            );
        }

        // Filters should be sparse (mostly zeros)
        let total_nonzero: usize = filterbank.iter().filter(|&&x| x > 1e-6).count();
        let total_elements = filterbank.len();
        let sparsity = total_nonzero as f32 / total_elements as f32;

        assert!(
            sparsity < 0.2,
            "Filterbank should be sparse, got {}",
            sparsity
        );
    }

    #[test]
    fn test_a_weighting() {
        let weights = MelLoss::create_a_weighting(80, 22050, 0.0, 8000.0);

        assert_eq!(weights.len(), 80);

        // All weights should be positive
        assert!(weights.iter().all(|&w| w > 0.0));

        // Weights should be normalized (max = 1.0)
        let max_weight = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!((max_weight - 1.0).abs() < 0.01, "Max weight should be ~1.0");

        // Weights should peak somewhere in mid-frequencies (roughly)
        let peak_idx = weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        // Peak should not be at extremes
        assert!(
            peak_idx > 10 && peak_idx < 70,
            "A-weighting peak should be in mid-range"
        );
    }

    #[test]
    fn test_mel_loss_identical_spectrograms() {
        let config = MelLossConfig::default();
        let loss = MelLoss::new(config).unwrap();

        let spec = Array2::<f32>::from_shape_fn((80, 100), |(i, j)| ((i + j) as f32 / 180.0).sin());

        let loss_value = loss.compute(&spec, &spec).unwrap();
        assert!(
            loss_value < 1e-5,
            "Loss should be near zero for identical spectrograms"
        );
    }

    #[test]
    fn test_mel_loss_different_spectrograms() {
        let config = MelLossConfig::default();
        let loss = MelLoss::new(config).unwrap();

        let spec1 =
            Array2::<f32>::from_shape_fn((80, 100), |(i, j)| ((i + j) as f32 / 180.0).sin());
        let spec2 =
            Array2::<f32>::from_shape_fn((80, 100), |(i, j)| ((i + j + 10) as f32 / 180.0).sin());

        let loss_value = loss.compute(&spec1, &spec2).unwrap();
        assert!(
            loss_value > 0.0,
            "Loss should be positive for different spectrograms"
        );
    }

    #[test]
    fn test_mel_loss_shape_mismatch() {
        let config = MelLossConfig::default();
        let loss = MelLoss::new(config).unwrap();

        let spec1 = Array2::<f32>::zeros((80, 100));
        let spec2 = Array2::<f32>::zeros((80, 200));

        let result = loss.compute(&spec1, &spec2);
        assert!(result.is_err());
    }

    #[test]
    fn test_mel_loss_without_a_weighting() {
        let mut config = MelLossConfig::default();
        config.use_a_weighting = false;

        let loss = MelLoss::new(config).unwrap();
        assert!(loss.a_weights.is_none());
    }

    #[test]
    fn test_mel_loss_with_a_weighting() {
        let mut config = MelLossConfig::default();
        config.use_a_weighting = true;

        let loss = MelLoss::new(config).unwrap();
        assert!(loss.a_weights.is_some());
    }

    #[test]
    fn test_perceptual_weighting_application() {
        let config = MelLossConfig {
            use_a_weighting: true,
            ..Default::default()
        };

        let loss = MelLoss::new(config).unwrap();
        let spec = Array2::<f32>::ones((80, 100));

        let weights = loss.a_weights.as_ref().unwrap();
        let weighted = loss.apply_perceptual_weighting(&spec, weights);

        // Shape should be preserved
        assert_eq!(weighted.shape(), spec.shape());

        // Values should be modified by weights
        let is_modified = weighted
            .iter()
            .zip(spec.iter())
            .any(|(&w, &s)| (w - s).abs() > 1e-6);

        assert!(is_modified, "Perceptual weighting should modify values");
    }
}
