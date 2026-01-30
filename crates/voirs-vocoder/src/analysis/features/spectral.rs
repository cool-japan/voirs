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

        // Pre-allocate all vectors with known capacity for better performance
        let mut centroid = Vec::with_capacity(n_frames);
        let mut rolloff = Vec::with_capacity(n_frames);
        let mut flatness = Vec::with_capacity(n_frames);
        let mut bandwidth = Vec::with_capacity(n_frames);
        let mut contrast = Vec::with_capacity(n_frames);
        let mut zero_crossing_rate = Vec::with_capacity(n_frames);

        // Pre-compute frequency bins once (avoid recomputation)
        let frequencies: Vec<f32> = (0..n_bins)
            .map(|i| i as f32 * self.config.sample_rate / self.config.n_fft as f32)
            .collect();

        // Pre-allocate magnitude buffer to avoid allocations in loop
        let mut magnitudes = vec![0.0f32; n_bins];

        for frame_idx in 0..n_frames {
            let frame = power_spectrogram.row(frame_idx);

            // Compute magnitudes in-place for better cache locality
            for (i, &power) in frame.iter().enumerate() {
                magnitudes[i] = power.sqrt();
            }

            // Spectral centroid (computed first as it's needed for bandwidth)
            let c = self.compute_spectral_centroid(&frequencies, &magnitudes);
            centroid.push(c);

            // Spectral rolloff
            let r = self.compute_spectral_rolloff(&frequencies, &magnitudes, 0.85);
            rolloff.push(r);

            // Spectral flatness
            let f = self.compute_spectral_flatness(&magnitudes);
            flatness.push(f);

            // Spectral bandwidth (uses precomputed centroid)
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

    #[inline]
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

    #[inline]
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

    #[inline]
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

    #[inline]
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
    
    #[inline]
    fn compute_spectral_contrast(&self, magnitudes: &[f32]) -> f32 {
        if magnitudes.len() < 3 {
            return 0.0;
        }

        // Single-pass computation to avoid allocations
        let mut peak_sum = 0.0f32;
        let mut valley_sum = 0.0f32;
        let mut peak_count = 0;
        let mut valley_count = 0;

        for window in magnitudes.windows(3) {
            if window[1] > window[0] && window[1] > window[2] {
                // Found a peak
                peak_sum += window[1];
                peak_count += 1;
            } else if window[1] < window[0] && window[1] < window[2] {
                // Found a valley
                valley_sum += window[1];
                valley_count += 1;
            }
        }

        if peak_count == 0 || valley_count == 0 {
            return 0.0;
        }

        let peak_mean = peak_sum / peak_count as f32;
        let valley_mean = valley_sum / valley_count as f32;

        if valley_mean > 1e-10 {
            peak_mean / valley_mean
        } else {
            0.0
        }
    }

    #[inline]
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
        (normalized_centroid * 0.8 + normalized_spread * 0.2).clamp(0.0, 0.5)
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
                Ok(SpectralFeatureMatrix {
                    centroids: vec![1000.0, 1200.0, 1100.0],
                    rolloffs: vec![2000.0, 2200.0, 2100.0],
                    flatness: vec![0.5, 0.5, 0.5],
                    bandwidths: vec![500.0, 520.0, 510.0],
                    contrasts: vec![0.3, 0.3, 0.3],
                    zero_crossing_rates: vec![0.1, 0.1, 0.1],
                })
            }

            fn extract_spectral_features(&self, power_spectrogram: &Array2<f32>) -> Result<SpectralFeatureMatrix> {
                let n_frames = power_spectrogram.nrows();
                Ok(SpectralFeatureMatrix {
                    centroids: vec![1000.0; n_frames],
                    rolloffs: vec![2000.0; n_frames],
                    flatness: vec![0.5; n_frames],
                    bandwidths: vec![500.0; n_frames],
                    contrasts: vec![0.3; n_frames],
                    zero_crossing_rates: vec![0.1; n_frames],
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
                let total_energy: f32 = magnitudes.iter().sum();
                let threshold_energy = total_energy * threshold;

                let mut cumulative_energy = 0.0;
                for (freq, &mag) in frequencies.iter().zip(magnitudes.iter()) {
                    cumulative_energy += mag;
                    if cumulative_energy >= threshold_energy {
                        return *freq;
                    }
                }
                frequencies.last().copied().unwrap_or(0.0)
            }

            fn compute_spectral_flatness(&self, magnitudes: &[f32]) -> f32 {
                if magnitudes.is_empty() {
                    return 0.0;
                }

                let geometric_mean = magnitudes.iter()
                    .map(|&x| (x + 1e-10).ln())
                    .sum::<f32>() / magnitudes.len() as f32;
                let geometric_mean = geometric_mean.exp();

                let arithmetic_mean = magnitudes.iter().sum::<f32>() / magnitudes.len() as f32;

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

                let sorted_mags = {
                    let mut mags = magnitudes.to_vec();
                    mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    mags
                };

                let n = sorted_mags.len();
                let valley = sorted_mags[n / 4];
                let peak = sorted_mags[3 * n / 4];

                if valley > 1e-10 {
                    20.0 * (peak / valley).log10()
                } else {
                    40.0 // High contrast when valley is near zero
                }
            }

            fn estimate_zcr_from_spectrum(&self, magnitudes: &[f32], _frequencies: &[f32]) -> f32 {
                // Estimate zero-crossing rate from high-frequency content
                if magnitudes.len() < 4 {
                    return 0.0;
                }

                let high_freq_start = magnitudes.len() * 3 / 4;
                let high_freq_energy: f32 = magnitudes[high_freq_start..].iter().sum();
                let total_energy: f32 = magnitudes.iter().sum();

                if total_energy > 1e-10 {
                    high_freq_energy / total_energy
                } else {
                    0.0
                }
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