//! Perceptual feature extraction and computation
//!
//! This module provides perceptual feature extraction including:
//! - Loudness (LUFS approximation)
//! - Brightness
//! - Warmth
//! - Roughness
//! - Sharpness
//! - Fluctuation strength
//! - Tonality

use crate::{Result, VocoderError};
use ndarray::{Array1, Array2};

/// Perceptual feature vector
#[derive(Debug, Clone)]
pub struct PerceptualFeatureVector {
    /// Loudness (LUFS)
    pub loudness: f32,
    
    /// Brightness
    pub brightness: f32,
    
    /// Warmth
    pub warmth: f32,
    
    /// Roughness
    pub roughness: f32,
    
    /// Sharpness
    pub sharpness: f32,
    
    /// Fluctuation strength
    pub fluctuation_strength: f32,
    
    /// Tonality
    pub tonality: f32,
}

impl Default for PerceptualFeatureVector {
    fn default() -> Self {
        Self {
            loudness: -70.0,
            brightness: 0.0,
            warmth: 0.0,
            roughness: 0.0,
            sharpness: 0.0,
            fluctuation_strength: 0.0,
            tonality: 0.0,
        }
    }
}

/// Perceptual feature computation methods
pub trait PerceptualFeatureComputer {
    /// Extract perceptual features from audio
    fn compute_perceptual_features(&self, audio: &[f32]) -> Result<PerceptualFeatureVector>;
    
    /// Extract perceptual features from samples
    fn extract_perceptual_features(&self, samples: &Array1<f32>) -> Result<PerceptualFeatureVector>;
    
    /// Compute loudness (LUFS approximation)
    fn compute_loudness(&self, samples: &Array1<f32>) -> f32;
    
    /// Compute brightness (high-frequency content)
    fn compute_brightness(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Compute warmth (low-frequency content)
    fn compute_warmth(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Compute roughness (amplitude modulation)
    fn compute_roughness(&self, samples: &Array1<f32>) -> f32;
    
    /// Compute sharpness (spectral slope)
    fn compute_sharpness(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Compute fluctuation strength (temporal modulation)
    fn compute_fluctuation_strength(&self, samples: &Array1<f32>) -> f32;
    
    /// Compute tonality (harmonicity vs noise)
    fn compute_tonality(&self, power_spectrogram: &Array2<f32>) -> f32;
}

impl PerceptualFeatureComputer for crate::analysis::features::FeatureExtractor {
    fn compute_perceptual_features(&self, audio: &[f32]) -> Result<PerceptualFeatureVector> {
        let samples = Array1::from_vec(audio.to_vec());
        self.extract_perceptual_features(&samples)
    }

    fn extract_perceptual_features(&self, samples: &Array1<f32>) -> Result<PerceptualFeatureVector> {
        // Compute power spectrogram for perceptual analysis
        let power_spectrogram = self.compute_power_spectrogram_const(samples)?;
        
        // Loudness computation (simplified LUFS approximation)
        let loudness = self.compute_loudness(samples);
        
        // Brightness (high-frequency content relative to total energy)
        let brightness = self.compute_brightness(&power_spectrogram);
        
        // Warmth (low-frequency content)
        let warmth = self.compute_warmth(&power_spectrogram);
        
        // Roughness (amplitude modulation in critical bands)
        let roughness = self.compute_roughness(samples);
        
        // Sharpness (spectral slope and high-frequency emphasis)
        let sharpness = self.compute_sharpness(&power_spectrogram);
        
        // Fluctuation strength (temporal modulation)
        let fluctuation_strength = self.compute_fluctuation_strength(samples);
        
        // Tonality (harmonicity vs noise)
        let tonality = self.compute_tonality(&power_spectrogram);
        
        Ok(PerceptualFeatureVector {
            loudness,
            brightness,
            warmth,
            roughness,
            sharpness,
            fluctuation_strength,
            tonality,
        })
    }

    fn compute_loudness(&self, samples: &Array1<f32>) -> f32 {
        // Simplified LUFS computation
        // K-weighting approximation using high-pass and high-shelf filters
        
        if samples.is_empty() {
            return -70.0;
        }
        
        // Apply pre-filter (simplified K-weighting)
        let mut filtered = samples.clone();
        
        // High-pass filter at ~38Hz (simplified)
        let alpha = 0.99;
        let mut prev_in = 0.0;
        let mut prev_out = 0.0;
        
        for sample in filtered.iter_mut() {
            let current = *sample;
            *sample = alpha * (prev_out + current - prev_in);
            prev_in = current;
            prev_out = *sample;
        }
        
        // Compute mean square with gating
        let mean_square = filtered.iter()
            .map(|&x| x * x)
            .sum::<f32>() / filtered.len() as f32;
        
        // Convert to LUFS
        if mean_square > 1e-10 {
            -0.691 + 10.0 * mean_square.log10()
        } else {
            -70.0
        }
    }

    fn compute_brightness(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let n_bins = power_spectrogram.ncols();
        let cutoff_bin = (n_bins * 3) / 4; // Upper 25% of spectrum
        
        let mut high_freq_energy = 0.0;
        let mut total_energy = 0.0;
        
        for row in power_spectrogram.rows() {
            let frame_total: f32 = row.sum();
            let frame_high: f32 = row.slice(ndarray::s![cutoff_bin..]).sum();
            
            total_energy += frame_total;
            high_freq_energy += frame_high;
        }
        
        if total_energy > 1e-10 {
            high_freq_energy / total_energy
        } else {
            0.0
        }
    }

    fn compute_warmth(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let n_bins = power_spectrogram.ncols();
        let cutoff_bin = n_bins / 4; // Lower 25% of spectrum
        
        let mut low_freq_energy = 0.0;
        let mut total_energy = 0.0;
        
        for row in power_spectrogram.rows() {
            let frame_total: f32 = row.sum();
            let frame_low: f32 = row.slice(ndarray::s![..cutoff_bin]).sum();
            
            total_energy += frame_total;
            low_freq_energy += frame_low;
        }
        
        if total_energy > 1e-10 {
            low_freq_energy / total_energy
        } else {
            0.0
        }
    }

    fn compute_roughness(&self, samples: &Array1<f32>) -> f32 {
        // Simplified roughness computation based on amplitude modulation
        let window_size = (self.config.sample_rate * 0.02) as usize; // 20ms window
        let hop_size = window_size / 2;
        
        if samples.len() < window_size * 2 {
            return 0.0;
        }
        
        let mut envelope = Vec::new();
        
        // Compute amplitude envelope
        for start in (0..samples.len() - window_size).step_by(hop_size) {
            let end = start + window_size;
            let window = samples.slice(ndarray::s![start..end]);
            let rms = (window.iter().map(|&x| x * x).sum::<f32>() / window_size as f32).sqrt();
            envelope.push(rms);
        }
        
        if envelope.len() < 3 {
            return 0.0;
        }
        
        // Compute modulation strength (variance in envelope)
        let mean_envelope = envelope.iter().sum::<f32>() / envelope.len() as f32;
        let modulation_variance = envelope.iter()
            .map(|&x| (x - mean_envelope).powi(2))
            .sum::<f32>() / envelope.len() as f32;
        
        modulation_variance.sqrt()
    }

    fn compute_sharpness(&self, power_spectrogram: &Array2<f32>) -> f32 {
        // Compute spectral centroid and use as sharpness measure
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
            let avg_centroid = total_centroid / valid_frames as f32;
            let nyquist = self.config.sample_rate / 2.0;
            avg_centroid / nyquist // Normalize by Nyquist frequency
        } else {
            0.0
        }
    }

    fn compute_fluctuation_strength(&self, samples: &Array1<f32>) -> f32 {
        // Measure temporal modulation in the 1-20 Hz range
        let window_size = (self.config.sample_rate * 0.1) as usize; // 100ms window
        let hop_size = window_size / 4;
        
        if samples.len() < window_size * 4 {
            return 0.0;
        }
        
        let mut envelope = Vec::new();
        
        // Compute amplitude envelope
        for start in (0..samples.len() - window_size).step_by(hop_size) {
            let end = start + window_size;
            let window = samples.slice(ndarray::s![start..end]);
            let rms = (window.iter().map(|&x| x * x).sum::<f32>() / window_size as f32).sqrt();
            envelope.push(rms);
        }
        
        if envelope.len() < 8 {
            return 0.0;
        }
        
        // Compute fluctuation strength as variance in envelope changes
        let diffs: Vec<f32> = envelope.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();
        
        if diffs.is_empty() {
            0.0
        } else {
            let mean_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;
            mean_diff
        }
    }

    fn compute_tonality(&self, power_spectrogram: &Array2<f32>) -> f32 {
        // Measure harmonicity vs noise using spectral regularity
        let n_frames = power_spectrogram.nrows();
        let n_bins = power_spectrogram.ncols();
        
        if n_frames == 0 || n_bins < 4 {
            return 0.0;
        }
        
        let mut total_regularity = 0.0;
        let mut valid_frames = 0;
        
        for row in power_spectrogram.rows() {
            let magnitudes: Vec<f32> = row.iter().map(|&x| x.sqrt()).collect();
            
            // Compute spectral irregularity (opposite of regularity)
            let mut irregularity = 0.0;
            for i in 1..(magnitudes.len() - 1) {
                let left = magnitudes[i - 1];
                let center = magnitudes[i];
                let right = magnitudes[i + 1];
                
                if center > 1e-10 {
                    let local_irregularity = ((left - center).abs() + (right - center).abs()) / center;
                    irregularity += local_irregularity;
                }
            }
            
            if magnitudes.len() > 2 {
                irregularity /= (magnitudes.len() - 2) as f32;
                // Tonality is inverse of irregularity
                let regularity = 1.0 / (1.0 + irregularity);
                total_regularity += regularity;
                valid_frames += 1;
            }
        }
        
        if valid_frames > 0 {
            total_regularity / valid_frames as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptual_feature_vector_default() {
        let features = PerceptualFeatureVector::default();
        assert_eq!(features.loudness, -70.0);
        assert_eq!(features.brightness, 0.0);
        assert_eq!(features.warmth, 0.0);
        assert_eq!(features.roughness, 0.0);
        assert_eq!(features.sharpness, 0.0);
        assert_eq!(features.fluctuation_strength, 0.0);
        assert_eq!(features.tonality, 0.0);
    }

    #[test]
    fn test_brightness_warmth_complementary() {
        // Create a mock power spectrogram with energy concentrated in high frequencies
        let mut power_spec = Array2::zeros((10, 100));
        
        // Add energy to high frequencies
        for i in 0..10 {
            for j in 75..100 {
                power_spec[[i, j]] = 1.0;
            }
        }
        
        // Mock implementation for testing
        struct MockExtractor {
            config: crate::analysis::features::FeatureConfig,
        }
        
        impl PerceptualFeatureComputer for MockExtractor {
            fn compute_perceptual_features(&self, _audio: &[f32]) -> Result<PerceptualFeatureVector> {
                unimplemented!()
            }
            
            fn extract_perceptual_features(&self, _samples: &Array1<f32>) -> Result<PerceptualFeatureVector> {
                unimplemented!()
            }
            
            fn compute_loudness(&self, _samples: &Array1<f32>) -> f32 {
                unimplemented!()
            }
            
            fn compute_brightness(&self, power_spectrogram: &Array2<f32>) -> f32 {
                let n_bins = power_spectrogram.ncols();
                let cutoff_bin = (n_bins * 3) / 4; // Upper 25% of spectrum
                
                let mut high_freq_energy = 0.0;
                let mut total_energy = 0.0;
                
                for row in power_spectrogram.rows() {
                    let frame_total: f32 = row.sum();
                    let frame_high: f32 = row.slice(ndarray::s![cutoff_bin..]).sum();
                    
                    total_energy += frame_total;
                    high_freq_energy += frame_high;
                }
                
                if total_energy > 1e-10 {
                    high_freq_energy / total_energy
                } else {
                    0.0
                }
            }
            
            fn compute_warmth(&self, power_spectrogram: &Array2<f32>) -> f32 {
                let n_bins = power_spectrogram.ncols();
                let cutoff_bin = n_bins / 4; // Lower 25% of spectrum
                
                let mut low_freq_energy = 0.0;
                let mut total_energy = 0.0;
                
                for row in power_spectrogram.rows() {
                    let frame_total: f32 = row.sum();
                    let frame_low: f32 = row.slice(ndarray::s![..cutoff_bin]).sum();
                    
                    total_energy += frame_total;
                    low_freq_energy += frame_low;
                }
                
                if total_energy > 1e-10 {
                    low_freq_energy / total_energy
                } else {
                    0.0
                }
            }
            
            fn compute_roughness(&self, _samples: &Array1<f32>) -> f32 { unimplemented!() }
            fn compute_sharpness(&self, _power_spectrogram: &Array2<f32>) -> f32 { unimplemented!() }
            fn compute_fluctuation_strength(&self, _samples: &Array1<f32>) -> f32 { unimplemented!() }
            fn compute_tonality(&self, _power_spectrogram: &Array2<f32>) -> f32 { unimplemented!() }
        }
        
        let extractor = MockExtractor {
            config: crate::analysis::features::FeatureConfig::default(),
        };
        
        let brightness = extractor.compute_brightness(&power_spec);
        let warmth = extractor.compute_warmth(&power_spec);
        
        // With energy concentrated in high frequencies, brightness should be high and warmth low
        assert!(brightness > 0.8);
        assert!(warmth < 0.2);
    }
}