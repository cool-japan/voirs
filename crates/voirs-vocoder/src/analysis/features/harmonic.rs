//! Harmonic feature extraction
//!
//! This module provides harmonic feature extraction including:
//! - Harmonic-to-noise ratio (HNR)
//! - Fundamental frequency (F0)
//! - Pitch stability
//! - Harmonic energy distribution
//! - Spectral peaks
//! - Pitch class profile

use crate::{Result, VocoderError};
use scirs2_core::ndarray::{Array1, Array2};

/// Harmonic features
#[derive(Debug, Clone)]
pub struct HarmonicFeatureVector {
    /// Harmonic-to-noise ratio
    pub hnr: f32,
    
    /// Fundamental frequency
    pub f0: f32,
    
    /// Pitch stability
    pub pitch_stability: f32,
    
    /// Harmonic energy distribution
    pub harmonic_energy_distribution: Vec<f32>,
    
    /// Spectral peaks
    pub spectral_peaks: Vec<f32>,
    
    /// Pitch class profile
    pub pitch_class_profile: Vec<f32>,
}

impl Default for HarmonicFeatureVector {
    fn default() -> Self {
        Self {
            hnr: -20.0,
            f0: 0.0,
            pitch_stability: 0.0,
            harmonic_energy_distribution: Vec::new(),
            spectral_peaks: Vec::new(),
            pitch_class_profile: vec![0.0; 12],
        }
    }
}

/// Harmonic feature computation methods
pub trait HarmonicFeatureComputer {
    /// Extract harmonic features from audio
    fn compute_harmonic_features(&self, audio: &[f32]) -> Result<HarmonicFeatureVector>;
    
    /// Extract harmonic features from samples and power spectrogram
    fn extract_harmonic_features(&self, samples: &Array1<f32>, power_spectrogram: &Array2<f32>) -> Result<HarmonicFeatureVector>;
    
    /// Compute aggregate harmonic-to-noise ratio
    fn compute_aggregate_hnr(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Estimate fundamental frequency
    fn estimate_fundamental_frequency(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Compute pitch stability
    fn compute_pitch_stability(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Compute harmonic energy distribution
    fn compute_harmonic_energy_distribution(&self, power_spectrogram: &Array2<f32>) -> Vec<f32>;
    
    /// Extract spectral peaks
    fn extract_spectral_peaks(&self, power_spectrogram: &Array2<f32>) -> Vec<f32>;
    
    /// Compute pitch class profile
    fn compute_pitch_class_profile(&self, power_spectrogram: &Array2<f32>) -> Vec<f32>;
}

impl HarmonicFeatureComputer for crate::analysis::features::FeatureExtractor {
    fn compute_harmonic_features(&self, audio: &[f32]) -> Result<HarmonicFeatureVector> {
        let samples = Array1::from_vec(audio.to_vec());
        let power_spectrogram = self.compute_power_spectrogram_const(&samples)?;
        self.extract_harmonic_features(&samples, &power_spectrogram)
    }

    fn extract_harmonic_features(&self, samples: &Array1<f32>, power_spectrogram: &Array2<f32>) -> Result<HarmonicFeatureVector> {
        let hnr = self.compute_aggregate_hnr(power_spectrogram);
        let f0 = self.estimate_fundamental_frequency(power_spectrogram);
        let pitch_stability = self.compute_pitch_stability(power_spectrogram);
        let harmonic_energy_distribution = self.compute_harmonic_energy_distribution(power_spectrogram);
        let spectral_peaks = self.extract_spectral_peaks(power_spectrogram);
        let pitch_class_profile = self.compute_pitch_class_profile(power_spectrogram);
        
        Ok(HarmonicFeatureVector {
            hnr,
            f0,
            pitch_stability,
            harmonic_energy_distribution,
            spectral_peaks,
            pitch_class_profile,
        })
    }

    fn compute_aggregate_hnr(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let n_frames = power_spectrogram.nrows();
        let n_bins = power_spectrogram.ncols();
        
        if n_frames == 0 || n_bins < 4 {
            return -20.0;
        }
        
        let frequencies: Vec<f32> = (0..n_bins)
            .map(|i| i as f32 * self.config.sample_rate / self.config.n_fft as f32)
            .collect();
        
        let mut total_hnr = 0.0;
        let mut valid_frames = 0;
        
        for row in power_spectrogram.rows() {
            let magnitudes: Vec<f32> = row.iter().map(|&x| x.sqrt()).collect();
            
            // Find fundamental frequency
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
                
                // Compute harmonic and noise energy
                let mut harmonic_energy = 0.0;
                let mut noise_energy = 0.0;
                
                for (i, &mag) in magnitudes.iter().enumerate() {
                    let freq = frequencies[i];
                    let energy = mag * mag;
                    
                    // Check if frequency is close to a harmonic
                    let harmonic_number = (freq / f0).round();
                    let expected_harmonic = f0 * harmonic_number;
                    let tolerance = f0 * 0.05; // 5% tolerance
                    
                    if (freq - expected_harmonic).abs() < tolerance && harmonic_number > 0.0 {
                        harmonic_energy += energy;
                    } else {
                        noise_energy += energy;
                    }
                }
                
                if noise_energy > 1e-10 {
                    let hnr = 10.0 * (harmonic_energy / noise_energy).log10();
                    total_hnr += hnr.clamp(-40.0, 40.0);
                    valid_frames += 1;
                }
            }
        }
        
        if valid_frames > 0 {
            total_hnr / valid_frames as f32
        } else {
            -20.0
        }
    }

    fn estimate_fundamental_frequency(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let n_frames = power_spectrogram.nrows();
        let n_bins = power_spectrogram.ncols();
        
        if n_frames == 0 || n_bins < 4 {
            return 0.0;
        }
        
        let frequencies: Vec<f32> = (0..n_bins)
            .map(|i| i as f32 * self.config.sample_rate / self.config.n_fft as f32)
            .collect();
        
        let mut f0_estimates = Vec::new();
        
        for row in power_spectrogram.rows() {
            let magnitudes: Vec<f32> = row.iter().map(|&x| x.sqrt()).collect();
            
            // Simple peak detection for F0 estimation
            let mut peaks = Vec::new();
            
            for i in 2..(magnitudes.len() - 2) {
                if magnitudes[i] > magnitudes[i - 1] && 
                   magnitudes[i] > magnitudes[i + 1] &&
                   magnitudes[i] > magnitudes[i - 2] &&
                   magnitudes[i] > magnitudes[i + 2] {
                    peaks.push((i, magnitudes[i]));
                }
            }
            
            // Sort peaks by magnitude
            peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            // Take the first significant peak as F0 candidate
            if !peaks.is_empty() {
                let f0_candidate = frequencies[peaks[0].0];
                if f0_candidate > 80.0 && f0_candidate < 800.0 { // Reasonable F0 range
                    f0_estimates.push(f0_candidate);
                }
            }
        }
        
        if f0_estimates.is_empty() {
            return 0.0;
        }
        
        // Return median F0 estimate
        f0_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        f0_estimates[f0_estimates.len() / 2]
    }

    fn compute_pitch_stability(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let n_frames = power_spectrogram.nrows();
        
        if n_frames < 3 {
            return 0.0;
        }
        
        let mut f0_estimates = Vec::new();
        
        // Estimate F0 for each frame
        for row in power_spectrogram.rows() {
            let magnitudes: Vec<f32> = row.iter().map(|&x| x.sqrt()).collect();
            let frame_f0 = self.estimate_frame_f0(&magnitudes);
            if frame_f0 > 0.0 {
                f0_estimates.push(frame_f0);
            }
        }
        
        if f0_estimates.len() < 2 {
            return 0.0;
        }
        
        // Compute F0 variance (stability is inverse of variance)
        let mean_f0 = f0_estimates.iter().sum::<f32>() / f0_estimates.len() as f32;
        let f0_variance = f0_estimates.iter()
            .map(|&f0| (f0 - mean_f0).powi(2))
            .sum::<f32>() / f0_estimates.len() as f32;
        
        let f0_std = f0_variance.sqrt();
        
        // Normalize by mean F0 and invert for stability measure
        if mean_f0 > 0.0 {
            1.0 / (1.0 + f0_std / mean_f0)
        } else {
            0.0
        }
    }

    fn compute_harmonic_energy_distribution(&self, power_spectrogram: &Array2<f32>) -> Vec<f32> {
        let n_frames = power_spectrogram.nrows();
        let n_bins = power_spectrogram.ncols();
        
        if n_frames == 0 || n_bins < 4 {
            return vec![0.0; 8]; // Return 8 harmonic bins
        }
        
        let frequencies: Vec<f32> = (0..n_bins)
            .map(|i| i as f32 * self.config.sample_rate / self.config.n_fft as f32)
            .collect();
        
        // Estimate average F0
        let avg_f0 = self.estimate_fundamental_frequency(power_spectrogram);
        
        if avg_f0 <= 0.0 {
            return vec![0.0; 8];
        }
        
        let mut harmonic_energies = vec![0.0; 8];
        let mut valid_frames = 0;
        
        for row in power_spectrogram.rows() {
            let magnitudes: Vec<f32> = row.iter().map(|&x| x.sqrt()).collect();
            
            for harmonic in 1..=8 {
                let target_freq = avg_f0 * harmonic as f32;
                let tolerance = avg_f0 * 0.1; // 10% tolerance
                
                // Find energy near this harmonic
                let mut harmonic_energy = 0.0;
                
                for (i, &mag) in magnitudes.iter().enumerate() {
                    let freq = frequencies[i];
                    if (freq - target_freq).abs() < tolerance {
                        harmonic_energy += mag * mag;
                    }
                }
                
                harmonic_energies[harmonic - 1] += harmonic_energy;
            }
            
            valid_frames += 1;
        }
        
        // Normalize by number of frames
        if valid_frames > 0 {
            for energy in &mut harmonic_energies {
                *energy /= valid_frames as f32;
            }
        }
        
        harmonic_energies
    }

    fn extract_spectral_peaks(&self, power_spectrogram: &Array2<f32>) -> Vec<f32> {
        let n_frames = power_spectrogram.nrows();
        let n_bins = power_spectrogram.ncols();
        
        if n_frames == 0 || n_bins < 5 {
            return Vec::new();
        }
        
        let frequencies: Vec<f32> = (0..n_bins)
            .map(|i| i as f32 * self.config.sample_rate / self.config.n_fft as f32)
            .collect();
        
        // Average spectrum across all frames
        let mut avg_spectrum = vec![0.0; n_bins];
        
        for row in power_spectrogram.rows() {
            for (i, &value) in row.iter().enumerate() {
                avg_spectrum[i] += value.sqrt();
            }
        }
        
        for value in &mut avg_spectrum {
            *value /= n_frames as f32;
        }
        
        // Find peaks in average spectrum
        let mut peaks = Vec::new();
        
        for i in 2..(avg_spectrum.len() - 2) {
            if avg_spectrum[i] > avg_spectrum[i - 1] && 
               avg_spectrum[i] > avg_spectrum[i + 1] &&
               avg_spectrum[i] > avg_spectrum[i - 2] &&
               avg_spectrum[i] > avg_spectrum[i + 2] {
                
                // Only include significant peaks
                let threshold = avg_spectrum.iter().sum::<f32>() / avg_spectrum.len() as f32 * 2.0;
                if avg_spectrum[i] > threshold {
                    peaks.push(frequencies[i]);
                }
            }
        }
        
        // Sort by frequency and limit to top 10 peaks
        peaks.sort_by(|a, b| a.partial_cmp(b).unwrap());
        peaks.truncate(10);
        
        peaks
    }

    fn compute_pitch_class_profile(&self, power_spectrogram: &Array2<f32>) -> Vec<f32> {
        let n_frames = power_spectrogram.nrows();
        let n_bins = power_spectrogram.ncols();
        
        if n_frames == 0 || n_bins < 12 {
            return vec![0.0; 12];
        }
        
        let frequencies: Vec<f32> = (0..n_bins)
            .map(|i| i as f32 * self.config.sample_rate / self.config.n_fft as f32)
            .collect();
        
        let mut pitch_classes = vec![0.0; 12];
        
        for row in power_spectrogram.rows() {
            for (i, &power) in row.iter().enumerate() {
                let freq = frequencies[i];
                
                if freq > 80.0 && freq < 4000.0 { // Focus on musical range
                    // Convert frequency to MIDI note number
                    let midi_note = 69.0 + 12.0 * (freq / 440.0).log2();
                    let pitch_class = (midi_note as usize) % 12;
                    
                    pitch_classes[pitch_class] += power.sqrt();
                }
            }
        }
        
        // Normalize by total energy
        let total_energy: f32 = pitch_classes.iter().sum();
        if total_energy > 1e-10 {
            for pc in &mut pitch_classes {
                *pc /= total_energy;
            }
        }
        
        pitch_classes
    }
}

impl crate::analysis::features::FeatureExtractor {
    /// Estimate F0 for a single frame
    fn estimate_frame_f0(&self, magnitudes: &[f32]) -> f32 {
        let n_bins = magnitudes.len();
        
        if n_bins < 4 {
            return 0.0;
        }
        
        let frequencies: Vec<f32> = (0..n_bins)
            .map(|i| i as f32 * self.config.sample_rate / self.config.n_fft as f32)
            .collect();
        
        // Simple peak detection
        let mut max_mag = 0.0;
        let mut max_idx = 0;
        
        for (i, &mag) in magnitudes.iter().enumerate() {
            if i > 2 && i < magnitudes.len() / 4 && mag > max_mag {
                max_mag = mag;
                max_idx = i;
            }
        }
        
        if max_mag > 1e-10 {
            frequencies[max_idx]
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonic_feature_vector_default() {
        let features = HarmonicFeatureVector::default();
        assert_eq!(features.hnr, -20.0);
        assert_eq!(features.f0, 0.0);
        assert_eq!(features.pitch_stability, 0.0);
        assert!(features.harmonic_energy_distribution.is_empty());
        assert!(features.spectral_peaks.is_empty());
        assert_eq!(features.pitch_class_profile.len(), 12);
    }

    #[test]
    fn test_pitch_class_profile_length() {
        let features = HarmonicFeatureVector::default();
        assert_eq!(features.pitch_class_profile.len(), 12);
        
        // All should be zero by default
        for &value in &features.pitch_class_profile {
            assert_eq!(value, 0.0);
        }
    }
}