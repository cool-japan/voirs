//! Spectral feature extraction and computation
//!
//! This module provides spectral feature extraction including:
//! - Spectral centroid
//! - Spectral rolloff
//! - Spectral flatness
//! - Spectral bandwidth
//! - Spectral contrast
//! - Zero crossing rate estimation

use crate::{Result, VocoderError};
use scirs2_core::ndarray::Array2;

/// Spectral features computed over time
#[derive(Debug, Clone)]
pub struct SpectralFeatureMatrix {
    /// Spectral centroid over time
    pub centroid: Vec<f32>,
    
    /// Spectral rolloff over time
    pub rolloff: Vec<f32>,
    
    /// Spectral flatness over time
    pub flatness: Vec<f32>,
    
    /// Spectral bandwidth over time
    pub bandwidth: Vec<f32>,
    
    /// Spectral contrast over time
    pub contrast: Vec<f32>,
    
    /// Zero crossing rate over time
    pub zero_crossing_rate: Vec<f32>,
}

impl Default for SpectralFeatureMatrix {
    fn default() -> Self {
        Self {
            centroid: Vec::new(),
            rolloff: Vec::new(),
            flatness: Vec::new(),
            bandwidth: Vec::new(),
            contrast: Vec::new(),
            zero_crossing_rate: Vec::new(),
        }
    }
}

/// Spectral feature computation methods
pub trait SpectralFeatureComputer {
    /// Extract spectral features from power spectrogram
    fn compute_spectral_features(&self, audio: &[f32]) -> Result<SpectralFeatureMatrix>;
    
    /// Extract spectral features from power spectrogram
    fn extract_spectral_features(&self, power_spectrogram: &Array2<f32>) -> Result<SpectralFeatureMatrix>;
    
    /// Compute spectral centroid
    fn compute_spectral_centroid(&self, frequencies: &[f32], magnitudes: &[f32]) -> f32;
    
    /// Compute spectral rolloff
    fn compute_spectral_rolloff(&self, frequencies: &[f32], magnitudes: &[f32], threshold: f32) -> f32;
    
    /// Compute spectral flatness
    fn compute_spectral_flatness(&self, magnitudes: &[f32]) -> f32;
    
    /// Compute spectral bandwidth
    fn compute_spectral_bandwidth(&self, frequencies: &[f32], magnitudes: &[f32], centroid: f32) -> f32;
    
    /// Compute spectral contrast
    fn compute_spectral_contrast(&self, magnitudes: &[f32]) -> f32;
    
    /// Estimate zero crossing rate from spectrum
    fn estimate_zcr_from_spectrum(&self, magnitudes: &[f32], frequencies: &[f32]) -> f32;
}

impl SpectralFeatureComputer for crate::analysis::features::FeatureExtractor {
    fn compute_spectral_features(&self, audio: &[f32]) -> Result<SpectralFeatureMatrix> {
        // Compute power spectrogram
        let power_spectrogram = self.compute_power_spectrogram_const(&scirs2_core::ndarray::Array1::from_vec(audio.to_vec()))?;
        self.extract_spectral_features(&power_spectrogram)
    }

    fn extract_spectral_features(&self, power_spectrogram: &Array2<f32>) -> Result<SpectralFeatureMatrix> {
        let n_frames = power_spectrogram.nrows();
        let n_bins = power_spectrogram.ncols();
        
        let mut centroid = Vec::with_capacity(n_frames);
        let mut rolloff = Vec::with_capacity(n_frames);
        let mut flatness = Vec::with_capacity(n_frames);
        let mut bandwidth = Vec::with_capacity(n_frames);
        let mut contrast = Vec::with_capacity(n_frames);
        let mut zero_crossing_rate = Vec::with_capacity(n_frames);
        
        let frequencies: Vec<f32> = (0..n_bins)
            .map(|i| i as f32 * self.config.sample_rate / self.config.n_fft as f32)
            .collect();
        
        for frame_idx in 0..n_frames {
            let frame = power_spectrogram.row(frame_idx);
            let magnitudes: Vec<f32> = frame.iter().map(|&x| x.sqrt()).collect();
            
            // Spectral centroid
            let c = self.compute_spectral_centroid(&frequencies, &magnitudes);
            centroid.push(c);
            
            // Spectral rolloff
            let r = self.compute_spectral_rolloff(&frequencies, &magnitudes, 0.85);
            rolloff.push(r);
            
            // Spectral flatness
            let f = self.compute_spectral_flatness(&magnitudes);
            flatness.push(f);
            
            // Spectral bandwidth
            let b = self.compute_spectral_bandwidth(&frequencies, &magnitudes, c);
            bandwidth.push(b);
            
            // Spectral contrast
            let sc = self.compute_spectral_contrast(&magnitudes);
            contrast.push(sc);
            
            // Zero crossing rate estimated from spectral features
            let zcr = self.estimate_zcr_from_spectrum(&magnitudes, &frequencies);
            zero_crossing_rate.push(zcr);
        }
        
        Ok(SpectralFeatureMatrix {
            centroid,
            rolloff,
            flatness,
            bandwidth,
            contrast,
            zero_crossing_rate,
        })
    }

    fn compute_spectral_centroid(&self, frequencies: &[f32], magnitudes: &[f32]) -> f32 {
        let weighted_sum: f32 = frequencies.iter().zip(magnitudes.iter())
            .map(|(&freq, &mag)| freq * mag)
            .sum();
        let magnitude_sum: f32 = magnitudes.iter().sum();
        
        if magnitude_sum > 1e-10 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }
    
    fn compute_spectral_rolloff(&self, frequencies: &[f32], magnitudes: &[f32], threshold: f32) -> f32 {
        let total_energy: f32 = magnitudes.iter().map(|&m| m * m).sum();
        let target_energy = total_energy * threshold;
        
        let mut cumulative_energy = 0.0;
        for (freq, mag) in frequencies.iter().zip(magnitudes.iter()) {
            cumulative_energy += mag * mag;
            if cumulative_energy >= target_energy {
                return *freq;
            }
        }
        
        frequencies.last().copied().unwrap_or(0.0)
    }
    
    fn compute_spectral_flatness(&self, magnitudes: &[f32]) -> f32 {
        if magnitudes.len() <= 1 {
            return 0.0;
        }
        
        let relevant_mags = &magnitudes[1..];
        let log_sum: f32 = relevant_mags.iter()
            .map(|&m| if m > 1e-10 { m.ln() } else { -23.0 })
            .sum();
        let geometric_mean = (log_sum / relevant_mags.len() as f32).exp();
        let arithmetic_mean: f32 = relevant_mags.iter().sum::<f32>() / relevant_mags.len() as f32;
        
        if arithmetic_mean > 1e-10 {
            geometric_mean / arithmetic_mean
        } else {
            0.0
        }
    }
    
    fn compute_spectral_bandwidth(&self, frequencies: &[f32], magnitudes: &[f32], centroid: f32) -> f32 {
        let weighted_variance: f32 = frequencies.iter().zip(magnitudes.iter())
            .map(|(&freq, &mag)| (freq - centroid).powi(2) * mag)
            .sum();
        let magnitude_sum: f32 = magnitudes.iter().sum();
        
        if magnitude_sum > 1e-10 {
            (weighted_variance / magnitude_sum).sqrt()
        } else {
            0.0
        }
    }
    
    fn compute_spectral_contrast(&self, magnitudes: &[f32]) -> f32 {
        if magnitudes.len() < 2 {
            return 0.0;
        }
        
        let peaks: Vec<f32> = magnitudes.windows(3)
            .map(|w| if w[1] > w[0] && w[1] > w[2] { w[1] } else { 0.0 })
            .collect();
        
        let valleys: Vec<f32> = magnitudes.windows(3)
            .map(|w| if w[1] < w[0] && w[1] < w[2] { w[1] } else { 1.0 })
            .collect();
        
        let peak_mean = peaks.iter().sum::<f32>() / peaks.len() as f32;
        let valley_mean = valleys.iter().sum::<f32>() / valleys.len() as f32;
        
        if valley_mean > 1e-10 {
            peak_mean / valley_mean
        } else {
            0.0
        }
    }

    fn estimate_zcr_from_spectrum(&self, magnitudes: &[f32], frequencies: &[f32]) -> f32 {
        // Estimate ZCR from spectral centroid and spectral spread
        // Higher centroid typically indicates more zero crossings
        let centroid = self.compute_spectral_centroid(frequencies, magnitudes);
        let spread = self.compute_spectral_bandwidth(frequencies, magnitudes, centroid);
        
        // Normalize by Nyquist frequency and apply empirical scaling
        let nyquist = self.config.sample_rate / 2.0;
        let normalized_centroid = centroid / nyquist;
        let normalized_spread = spread / nyquist;
        
        // Empirical formula: ZCR correlates with spectral centroid and spread
        // Scale to reasonable ZCR range (0.0 to 0.5)
        let zcr_estimate = (normalized_centroid * 0.8 + normalized_spread * 0.2).min(0.5).max(0.0);
        
        zcr_estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_feature_matrix_default() {
        let features = SpectralFeatureMatrix::default();
        assert!(features.centroid.is_empty());
        assert!(features.rolloff.is_empty());
        assert!(features.flatness.is_empty());
        assert!(features.bandwidth.is_empty());
        assert!(features.contrast.is_empty());
        assert!(features.zero_crossing_rate.is_empty());
    }

    #[test]
    fn test_spectral_centroid_computation() {
        let frequencies = vec![100.0, 200.0, 300.0, 400.0];
        let magnitudes = vec![0.5, 1.0, 0.5, 0.2];
        
        // Create a simple mock implementation for testing
        struct MockExtractor {
            config: crate::analysis::features::FeatureConfig,
        }
        
        impl SpectralFeatureComputer for MockExtractor {
            fn compute_spectral_features(&self, _audio: &[f32]) -> Result<SpectralFeatureMatrix> {
                unimplemented!()
            }
            
            fn extract_spectral_features(&self, _power_spectrogram: &Array2<f32>) -> Result<SpectralFeatureMatrix> {
                unimplemented!()
            }
            
            fn compute_spectral_centroid(&self, frequencies: &[f32], magnitudes: &[f32]) -> f32 {
                let weighted_sum: f32 = frequencies.iter().zip(magnitudes.iter())
                    .map(|(&freq, &mag)| freq * mag)
                    .sum();
                let magnitude_sum: f32 = magnitudes.iter().sum();
                
                if magnitude_sum > 1e-10 {
                    weighted_sum / magnitude_sum
                } else {
                    0.0
                }
            }
            
            fn compute_spectral_rolloff(&self, _frequencies: &[f32], _magnitudes: &[f32], _threshold: f32) -> f32 {
                unimplemented!()
            }
            
            fn compute_spectral_flatness(&self, _magnitudes: &[f32]) -> f32 {
                unimplemented!()
            }
            
            fn compute_spectral_bandwidth(&self, _frequencies: &[f32], _magnitudes: &[f32], _centroid: f32) -> f32 {
                unimplemented!()
            }
            
            fn compute_spectral_contrast(&self, _magnitudes: &[f32]) -> f32 {
                unimplemented!()
            }
            
            fn estimate_zcr_from_spectrum(&self, _magnitudes: &[f32], _frequencies: &[f32]) -> f32 {
                unimplemented!()
            }
        }
        
        let extractor = MockExtractor {
            config: crate::analysis::features::FeatureConfig::default(),
        };
        
        let centroid = extractor.compute_spectral_centroid(&frequencies, &magnitudes);
        
        // Expected centroid: (100*0.5 + 200*1.0 + 300*0.5 + 400*0.2) / (0.5+1.0+0.5+0.2)
        // = (50 + 200 + 150 + 80) / 2.2 = 480 / 2.2 ≈ 218.18
        assert!((centroid - 218.18).abs() < 0.1);
    }
}