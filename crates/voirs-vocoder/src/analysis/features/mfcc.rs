//! MFCC and Mel-scale feature extraction
//!
//! This module provides MFCC and mel-scale feature extraction including:
//! - Mel-frequency cepstral coefficients (MFCC)
//! - Mel-scale spectrogram computation
//! - DCT transformation
//! - Delta and delta-delta features

use crate::{Result, VocoderError};
use scirs2_core::ndarray::{Array1, Array2, s};
use std::f32::consts::PI;

/// MFCC and mel-scale feature computation methods
pub trait MfccFeatureComputer {
    /// Compute MFCC and mel spectrogram from audio
    fn compute_mfcc_and_mel(&self, audio: &[f32]) -> Result<(Array2<f32>, Array2<f32>)>;
    
    /// Compute mel-scale spectrogram
    fn compute_mel_spectrogram(&self, power_spectrogram: &Array2<f32>) -> Result<Array2<f32>>;
    
    /// Apply DCT to log mel spectrogram to get MFCC
    fn apply_dct(&self, log_mel_spec: &Array2<f32>) -> Result<Array2<f32>>;
    
    /// Compute delta features (first derivatives)
    fn compute_delta_features(&self, features: &Array2<f32>, width: usize) -> Array2<f32>;
    
    /// Compute delta-delta features (second derivatives)
    fn compute_delta_delta_features(&self, features: &Array2<f32>) -> Array2<f32>;
    
    /// Apply pre-emphasis filter
    fn apply_preemphasis(&self, samples: &Array1<f32>) -> Array1<f32>;
}

impl MfccFeatureComputer for crate::analysis::features::FeatureExtractor {
    fn compute_mfcc_and_mel(&self, audio: &[f32]) -> Result<(Array2<f32>, Array2<f32>)> {
        let samples = Array1::from_vec(audio.to_vec());
        
        // Apply pre-emphasis
        let preemphasized = self.apply_preemphasis(&samples);
        
        // Compute power spectrogram
        let power_spectrogram = self.compute_power_spectrogram_const(&preemphasized)?;
        
        // Compute mel spectrogram
        let mel_spectrogram = self.compute_mel_spectrogram(&power_spectrogram)?;
        
        // Convert to log scale
        let log_mel_spec = mel_spectrogram.mapv(|x| {
            if x > 1e-10 {
                x.ln()
            } else {
                -23.0 // log(1e-10)
            }
        });
        
        // Apply DCT to get MFCC
        let mut mfcc = self.apply_dct(&log_mel_spec)?;
        
        // Add delta features if requested
        if self.config.delta {
            let delta_features = self.compute_delta_features(&mfcc, 2);
            mfcc = ndarray::concatenate![ndarray::Axis(1), mfcc, delta_features];
        }
        
        // Add delta-delta features if requested
        if self.config.delta_delta {
            let delta_delta_features = self.compute_delta_delta_features(&mfcc);
            mfcc = ndarray::concatenate![ndarray::Axis(1), mfcc, delta_delta_features];
        }
        
        Ok((mfcc, mel_spectrogram))
    }

    fn compute_mel_spectrogram(&self, power_spectrogram: &Array2<f32>) -> Result<Array2<f32>> {
        let n_frames = power_spectrogram.nrows();
        let mut mel_spectrogram = Array2::zeros((n_frames, self.config.n_mels));
        
        // Apply mel filterbank to each frame
        for frame_idx in 0..n_frames {
            let frame = power_spectrogram.row(frame_idx);
            let mel_frame = self.mel_filterbank.apply(frame.as_slice().unwrap());
            
            // Raise to power
            for (mel_idx, &mel_value) in mel_frame.iter().enumerate() {
                if mel_idx < self.config.n_mels {
                    mel_spectrogram[[frame_idx, mel_idx]] = mel_value.powf(self.config.power / 2.0);
                }
            }
        }
        
        Ok(mel_spectrogram)
    }

    fn apply_dct(&self, log_mel_spec: &Array2<f32>) -> Result<Array2<f32>> {
        let (n_frames, n_mels) = log_mel_spec.dim();
        let mut mfcc = Array2::zeros((n_frames, self.config.n_mfcc));
        
        for frame_idx in 0..n_frames {
            let frame = log_mel_spec.row(frame_idx);
            
            for k in 0..self.config.n_mfcc {
                let mut sum = 0.0;
                for n in 0..n_mels {
                    sum += frame[n] * (PI * k as f32 * (2.0 * n as f32 + 1.0) / (2.0 * n_mels as f32)).cos();
                }
                mfcc[[frame_idx, k]] = sum;
            }
        }
        
        // Apply normalization factor
        let normalization_factor = (2.0 / n_mels as f32).sqrt();
        mfcc.mapv_inplace(|x| x * normalization_factor);
        
        // Apply orthogonal normalization for first coefficient
        for frame_idx in 0..n_frames {
            mfcc[[frame_idx, 0]] *= 1.0 / std::f32::consts::SQRT_2;
        }
        
        Ok(mfcc)
    }

    fn compute_delta_features(&self, features: &Array2<f32>, width: usize) -> Array2<f32> {
        let (n_frames, n_coeffs) = features.dim();
        let mut delta_features = Array2::zeros((n_frames, n_coeffs));
        
        for frame_idx in 0..n_frames {
            for coeff_idx in 0..n_coeffs {
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                
                for n in 1..=width {
                    let n_f = n as f32;
                    
                    // Forward difference
                    let forward_idx = (frame_idx + n).min(n_frames - 1);
                    let forward_val = features[[forward_idx, coeff_idx]];
                    
                    // Backward difference
                    let backward_idx = frame_idx.saturating_sub(n);
                    let backward_val = features[[backward_idx, coeff_idx]];
                    
                    numerator += n_f * (forward_val - backward_val);
                    denominator += n_f * n_f;
                }
                
                if denominator > 0.0 {
                    delta_features[[frame_idx, coeff_idx]] = numerator / (2.0 * denominator);
                }
            }
        }
        
        delta_features
    }

    fn compute_delta_delta_features(&self, features: &Array2<f32>) -> Array2<f32> {
        // Delta-delta is the delta of delta features
        self.compute_delta_features(features, 1)
    }

    fn apply_preemphasis(&self, samples: &Array1<f32>) -> Array1<f32> {
        if samples.len() < 2 {
            return samples.clone();
        }
        
        let mut preemphasized = Array1::zeros(samples.len());
        preemphasized[0] = samples[0];
        
        for i in 1..samples.len() {
            preemphasized[i] = samples[i] - self.config.preemphasis * samples[i - 1];
        }
        
        preemphasized
    }
}

/// Utility functions for MFCC computation
impl crate::analysis::features::FeatureExtractor {
    /// Compute power spectrogram (const version for internal use)
    pub(crate) fn compute_power_spectrogram_const(&self, samples: &Array1<f32>) -> Result<Array2<f32>> {
        use scirs2_fft::{FftPlanner, RealFftPlanner};
    use scirs2_core::ndarray::Array1;

        let n_frames = (samples.len() + self.config.hop_length - 1) / self.config.hop_length;
        let mut power_spectrogram = Array2::zeros((n_frames, self.config.n_fft / 2 + 1));
        
        let window = self.create_window(self.config.n_fft);
        
        for frame_idx in 0..n_frames {
            let start = frame_idx * self.config.hop_length;
            let end = (start + self.config.n_fft).min(samples.len());
            
            // Extract frame and apply window
            let mut frame_data = vec![0.0; self.config.n_fft];
            for (i, &sample) in samples.slice(s![start..end]).iter().enumerate() {
                frame_data[i] = sample * window[i];
            }
            
            // Compute FFT using functional API
            let spectrum = scirs2_fft::rfft(&frame_data, None)?;
            
            // Compute power spectrum
            for (bin_idx, complex_val) in spectrum.iter().enumerate() {
                let power = complex_val.norm_sqr();
                power_spectrogram[[frame_idx, bin_idx]] = power;
            }
        }
        
        Ok(power_spectrogram)
    }
    
    /// Create window function (Hann window)
    fn create_window(&self, size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let angle = 2.0 * PI * i as f32 / (size - 1) as f32;
                0.5 * (1.0 - angle.cos())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::features::FeatureConfig;

    #[test]
    fn test_preemphasis_filter() {
        let config = FeatureConfig::default();
        let extractor = crate::analysis::features::FeatureExtractor::new(config).unwrap();
        
        let samples = Array1::from_vec(vec![1.0, 0.5, -0.2, 0.3, -0.1]);
        let preemphasized = extractor.apply_preemphasis(&samples);
        
        assert_eq!(preemphasized.len(), samples.len());
        assert_eq!(preemphasized[0], samples[0]); // First sample unchanged
        
        // Check that pre-emphasis was applied correctly
        let expected_1 = samples[1] - extractor.config.preemphasis * samples[0];
        assert!((preemphasized[1] - expected_1).abs() < 1e-6);
    }

    #[test]
    fn test_delta_features_computation() {
        let config = FeatureConfig::default();
        let extractor = crate::analysis::features::FeatureExtractor::new(config).unwrap();
        
        // Create simple test features (3 frames, 2 coefficients)
        let mut features = Array2::zeros((3, 2));
        features[[0, 0]] = 1.0;
        features[[1, 0]] = 2.0;
        features[[2, 0]] = 3.0;
        features[[0, 1]] = 0.5;
        features[[1, 1]] = 1.0;
        features[[2, 1]] = 1.5;
        
        let delta_features = extractor.compute_delta_features(&features, 1);
        
        assert_eq!(delta_features.shape(), &[3, 2]);
        
        // Delta should show the rate of change
        // For the middle frame, delta should be positive (increasing trend)
        assert!(delta_features[[1, 0]] > 0.0);
        assert!(delta_features[[1, 1]] > 0.0);
    }

    #[test]
    fn test_dct_orthogonality() {
        let config = FeatureConfig {
            n_mfcc: 4,
            n_mels: 8,
            ..FeatureConfig::default()
        };
        
        let extractor = crate::analysis::features::FeatureExtractor::new(config).unwrap();
        
        // Create test log mel spectrogram
        let mut log_mel_spec = Array2::zeros((2, 8));
        
        // Fill with some test data
        for i in 0..2 {
            for j in 0..8 {
                log_mel_spec[[i, j]] = (i + j) as f32 * 0.1;
            }
        }
        
        let mfcc = extractor.apply_dct(&log_mel_spec).unwrap();
        
        assert_eq!(mfcc.shape(), &[2, 4]);
        
        // First coefficient should be different due to normalization
        assert_ne!(mfcc[[0, 0]], mfcc[[1, 0]]);
    }
}