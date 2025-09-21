//! Timbral feature extraction
//!
//! This module provides timbral feature extraction including:
//! - Spectral centroid
//! - Spectral rolloff
//! - Spectral flux
//! - Spectral irregularity
//! - Inharmonicity
//! - Noisiness

use crate::{Result, VocoderError};
use ndarray::Array2;

/// Timbral features
#[derive(Debug, Clone)]
pub struct TimbralFeatureVector {
    /// Spectral centroid
    pub spectral_centroid: f32,
    
    /// Spectral rolloff
    pub spectral_rolloff: f32,
    
    /// Spectral flux
    pub spectral_flux: f32,
    
    /// Spectral irregularity
    pub spectral_irregularity: f32,
    
    /// Inharmonicity
    pub inharmonicity: f32,
    
    /// Noisiness
    pub noisiness: f32,
}

impl Default for TimbralFeatureVector {
    fn default() -> Self {
        Self {
            spectral_centroid: 0.0,
            spectral_rolloff: 0.0,
            spectral_flux: 0.0,
            spectral_irregularity: 0.0,
            inharmonicity: 0.0,
            noisiness: 0.0,
        }
    }
}

/// Timbral feature computation methods
pub trait TimbralFeatureComputer {
    /// Extract timbral features from audio
    fn compute_timbral_features(&self, audio: &[f32]) -> Result<TimbralFeatureVector>;
    
    /// Extract timbral features from power spectrogram
    fn extract_timbral_features(&self, power_spectrogram: &Array2<f32>) -> Result<TimbralFeatureVector>;
    
    /// Compute aggregate spectral centroid
    fn compute_aggregate_spectral_centroid(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Compute aggregate spectral rolloff
    fn compute_aggregate_spectral_rolloff(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Compute aggregate spectral flux
    fn compute_aggregate_spectral_flux(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Compute spectral irregularity
    fn compute_spectral_irregularity(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Compute inharmonicity
    fn compute_inharmonicity(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Compute noisiness
    fn compute_noisiness(&self, power_spectrogram: &Array2<f32>) -> f32;
}

impl TimbralFeatureComputer for crate::analysis::features::FeatureExtractor {
    fn compute_timbral_features(&self, audio: &[f32]) -> Result<TimbralFeatureVector> {
        let samples = ndarray::Array1::from_vec(audio.to_vec());
        let power_spectrogram = self.compute_power_spectrogram_const(&samples)?;
        self.extract_timbral_features(&power_spectrogram)
    }

    fn extract_timbral_features(&self, power_spectrogram: &Array2<f32>) -> Result<TimbralFeatureVector> {
        // Compute aggregate spectral features
        let spectral_centroid = self.compute_aggregate_spectral_centroid(power_spectrogram);
        let spectral_rolloff = self.compute_aggregate_spectral_rolloff(power_spectrogram);
        let spectral_flux = self.compute_aggregate_spectral_flux(power_spectrogram);
        let spectral_irregularity = self.compute_spectral_irregularity(power_spectrogram);
        let inharmonicity = self.compute_inharmonicity(power_spectrogram);
        let noisiness = self.compute_noisiness(power_spectrogram);
        
        Ok(TimbralFeatureVector {
            spectral_centroid,
            spectral_rolloff,
            spectral_flux,
            spectral_irregularity,
            inharmonicity,
            noisiness,
        })
    }

    fn compute_aggregate_spectral_centroid(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let n_frames = power_spectrogram.nrows();
        let n_bins = power_spectrogram.ncols();
        
        if n_frames == 0 || n_bins == 0 {
            return 0.0;
        }
        
        let frequencies: Vec<f32> = (0..n_bins)
            .map(|i| i as f32 * self.config.sample_rate / self.config.n_fft as f32)
            .collect();
        
        let mut total_centroid = 0.0;
        let mut valid_frames = 0;
        
        for row in power_spectrogram.rows() {
            let magnitudes: Vec<f32> = row.iter().map(|&x| x.sqrt()).collect();
            let magnitude_sum: f32 = magnitudes.iter().sum();
            
            if magnitude_sum > 1e-10 {
                let weighted_sum: f32 = frequencies.iter().zip(magnitudes.iter())
                    .map(|(&freq, &mag)| freq * mag)
                    .sum();
                
                total_centroid += weighted_sum / magnitude_sum;
                valid_frames += 1;
            }
        }
        
        if valid_frames > 0 {
            total_centroid / valid_frames as f32
        } else {
            0.0
        }
    }

    fn compute_aggregate_spectral_rolloff(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let n_frames = power_spectrogram.nrows();
        let n_bins = power_spectrogram.ncols();
        
        if n_frames == 0 || n_bins == 0 {
            return 0.0;
        }
        
        let frequencies: Vec<f32> = (0..n_bins)
            .map(|i| i as f32 * self.config.sample_rate / self.config.n_fft as f32)
            .collect();
        
        let mut total_rolloff = 0.0;
        let mut valid_frames = 0;
        
        for row in power_spectrogram.rows() {
            let magnitudes: Vec<f32> = row.iter().map(|&x| x.sqrt()).collect();
            let total_energy: f32 = magnitudes.iter().map(|&m| m * m).sum();
            
            if total_energy > 1e-10 {
                let target_energy = total_energy * 0.85;
                let mut cumulative_energy = 0.0;
                
                for (freq, mag) in frequencies.iter().zip(magnitudes.iter()) {
                    cumulative_energy += mag * mag;
                    if cumulative_energy >= target_energy {
                        total_rolloff += *freq;
                        valid_frames += 1;
                        break;
                    }
                }
            }
        }
        
        if valid_frames > 0 {
            total_rolloff / valid_frames as f32
        } else {
            0.0
        }
    }

    fn compute_aggregate_spectral_flux(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let n_frames = power_spectrogram.nrows();
        
        if n_frames < 2 {
            return 0.0;
        }
        
        let mut total_flux = 0.0;
        let mut valid_frames = 0;
        
        for i in 1..n_frames {
            let prev_frame = power_spectrogram.row(i - 1);
            let curr_frame = power_spectrogram.row(i);
            
            let flux: f32 = prev_frame.iter().zip(curr_frame.iter())
                .map(|(&prev, &curr)| (curr.sqrt() - prev.sqrt()).max(0.0).powi(2))
                .sum();
            
            total_flux += flux.sqrt();
            valid_frames += 1;
        }
        
        if valid_frames > 0 {
            total_flux / valid_frames as f32
        } else {
            0.0
        }
    }

    fn compute_spectral_irregularity(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let n_frames = power_spectrogram.nrows();
        let n_bins = power_spectrogram.ncols();
        
        if n_frames == 0 || n_bins < 3 {
            return 0.0;
        }
        
        let mut total_irregularity = 0.0;
        let mut valid_frames = 0;
        
        for row in power_spectrogram.rows() {
            let magnitudes: Vec<f32> = row.iter().map(|&x| x.sqrt()).collect();
            
            let mut frame_irregularity = 0.0;
            let mut valid_bins = 0;
            
            for i in 1..(magnitudes.len() - 1) {
                let left = magnitudes[i - 1];
                let center = magnitudes[i];
                let right = magnitudes[i + 1];
                
                if center > 1e-10 {
                    let local_irregularity = ((left - center).abs() + (right - center).abs()) / center;
                    frame_irregularity += local_irregularity;
                    valid_bins += 1;
                }
            }
            
            if valid_bins > 0 {
                total_irregularity += frame_irregularity / valid_bins as f32;
                valid_frames += 1;
            }
        }
        
        if valid_frames > 0 {
            total_irregularity / valid_frames as f32
        } else {
            0.0
        }
    }

    fn compute_inharmonicity(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let n_frames = power_spectrogram.nrows();
        let n_bins = power_spectrogram.ncols();
        
        if n_frames == 0 || n_bins < 4 {
            return 0.0;
        }
        
        let frequencies: Vec<f32> = (0..n_bins)
            .map(|i| i as f32 * self.config.sample_rate / self.config.n_fft as f32)
            .collect();
        
        let mut total_inharmonicity = 0.0;
        let mut valid_frames = 0;
        
        for row in power_spectrogram.rows() {
            let magnitudes: Vec<f32> = row.iter().map(|&x| x.sqrt()).collect();
            
            // Find fundamental frequency (simple peak detection)
            let mut f0_idx = 0;
            let mut max_mag = 0.0;
            
            for (i, &mag) in magnitudes.iter().enumerate() {
                if i > 2 && i < magnitudes.len() / 4 && mag > max_mag {
                    max_mag = mag;
                    f0_idx = i;
                }
            }
            
            if f0_idx > 0 && max_mag > 1e-10 {
                let f0 = frequencies[f0_idx];
                
                // Check harmonic vs inharmonic content
                let mut harmonic_energy = 0.0;
                let mut total_energy = 0.0;
                
                for (i, &mag) in magnitudes.iter().enumerate() {
                    let freq = frequencies[i];
                    total_energy += mag * mag;
                    
                    // Check if frequency is close to a harmonic
                    let harmonic_number = (freq / f0).round();
                    let expected_harmonic = f0 * harmonic_number;
                    let tolerance = f0 * 0.05; // 5% tolerance
                    
                    if (freq - expected_harmonic).abs() < tolerance && harmonic_number > 0.0 {
                        harmonic_energy += mag * mag;
                    }
                }
                
                if total_energy > 1e-10 {
                    let inharmonicity = 1.0 - (harmonic_energy / total_energy);
                    total_inharmonicity += inharmonicity;
                    valid_frames += 1;
                }
            }
        }
        
        if valid_frames > 0 {
            total_inharmonicity / valid_frames as f32
        } else {
            0.5 // Default moderate inharmonicity
        }
    }

    fn compute_noisiness(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let n_frames = power_spectrogram.nrows();
        
        if n_frames < 2 {
            return 0.0;
        }
        
        let mut total_noisiness = 0.0;
        let mut valid_frames = 0;
        
        for row in power_spectrogram.rows() {
            let magnitudes: Vec<f32> = row.iter().map(|&x| x.sqrt()).collect();
            
            if magnitudes.len() < 3 {
                continue;
            }
            
            // Compute spectral flatness as a measure of noisiness
            let log_sum: f32 = magnitudes[1..].iter()
                .map(|&m| if m > 1e-10 { m.ln() } else { -23.0 })
                .sum();
            let geometric_mean = (log_sum / (magnitudes.len() - 1) as f32).exp();
            let arithmetic_mean: f32 = magnitudes[1..].iter().sum::<f32>() / (magnitudes.len() - 1) as f32;
            
            if arithmetic_mean > 1e-10 {
                let flatness = geometric_mean / arithmetic_mean;
                total_noisiness += flatness;
                valid_frames += 1;
            }
        }
        
        if valid_frames > 0 {
            total_noisiness / valid_frames as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timbral_feature_vector_default() {
        let features = TimbralFeatureVector::default();
        assert_eq!(features.spectral_centroid, 0.0);
        assert_eq!(features.spectral_rolloff, 0.0);
        assert_eq!(features.spectral_flux, 0.0);
        assert_eq!(features.spectral_irregularity, 0.0);
        assert_eq!(features.inharmonicity, 0.0);
        assert_eq!(features.noisiness, 0.0);
    }
}