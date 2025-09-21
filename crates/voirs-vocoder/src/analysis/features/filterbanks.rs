//! Filterbank implementations for feature extraction
//!
//! This module provides filterbank implementations including:
//! - Mel-scale filterbank for MFCC computation
//! - Chroma filterbank for chroma feature extraction

use crate::{Result, VocoderError};
use ndarray::Array2;

/// Mel-scale filterbank for mel spectrogram computation
#[derive(Debug, Clone)]
pub struct MelFilterbank {
    filters: Array2<f32>,
    n_mels: usize,
}

impl MelFilterbank {
    /// Create a new mel filterbank
    pub fn new(
        n_fft_bins: usize,
        n_mels: usize,
        sample_rate: f32,
        fmin: f32,
        fmax: f32,
    ) -> Result<Self> {
        let mut filters = Array2::zeros((n_mels, n_fft_bins));
        
        // Convert frequency range to mel scale
        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);
        
        // Create mel-spaced frequency points
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();
        
        // Convert back to Hz
        let hz_points: Vec<f32> = mel_points.iter()
            .map(|&mel| Self::mel_to_hz(mel))
            .collect();
        
        // Convert to FFT bin indices
        let bin_points: Vec<usize> = hz_points.iter()
            .map(|&hz| ((hz * (n_fft_bins - 1) as f32 * 2.0) / sample_rate).round() as usize)
            .map(|bin| bin.min(n_fft_bins - 1))
            .collect();
        
        // Create triangular filters
        for m in 0..n_mels {
            let start = bin_points[m];
            let center = bin_points[m + 1];
            let end = bin_points[m + 2];
            
            // Rising edge
            for k in start..center {
                if center > start && k < n_fft_bins {
                    filters[[m, k]] = (k - start) as f32 / (center - start) as f32;
                }
            }
            
            // Falling edge
            for k in center..end {
                if end > center && k < n_fft_bins {
                    filters[[m, k]] = (end - k) as f32 / (end - center) as f32;
                }
            }
        }
        
        Ok(Self { filters, n_mels })
    }
    
    /// Apply mel filterbank to power spectrum
    pub fn apply(&self, power_spectrum: &[f32]) -> Vec<f32> {
        let mut mel_spectrum = vec![0.0; self.n_mels];
        
        for mel_idx in 0..self.n_mels {
            let filter = self.filters.row(mel_idx);
            let energy: f32 = power_spectrum.iter()
                .zip(filter.iter())
                .map(|(&power, &weight)| power * weight)
                .sum();
            mel_spectrum[mel_idx] = energy;
        }
        
        mel_spectrum
    }
    
    /// Convert frequency in Hz to mel scale
    pub fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }
    
    /// Convert mel scale to frequency in Hz
    pub fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }
    
    /// Get number of mel channels
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }
}

/// Chroma filterbank for chroma feature computation
#[derive(Debug, Clone)]
pub struct ChromaFilterbank {
    filters: Array2<f32>,
}

impl ChromaFilterbank {
    /// Create a new chroma filterbank
    pub fn new(
        n_fft_bins: usize,
        sample_rate: f32,
        fmin: f32,
    ) -> Result<Self> {
        let mut filters = Array2::zeros((12, n_fft_bins));
        
        for bin in 0..n_fft_bins {
            let freq = bin as f32 * sample_rate / (2.0 * (n_fft_bins - 1) as f32);
            
            if freq >= fmin && freq <= 8000.0 {
                let chroma_weights = Self::compute_chroma_weights(freq, fmin);
                
                for (chroma_idx, weight) in chroma_weights.iter().enumerate() {
                    filters[[chroma_idx, bin]] = *weight;
                }
            }
        }
        
        Ok(Self { filters })
    }
    
    /// Apply chroma filterbank to power spectrum
    pub fn apply(&self, power_spectrum: &[f32]) -> Vec<f32> {
        let mut chroma_spectrum = vec![0.0; 12];
        
        for chroma_idx in 0..12 {
            let filter = self.filters.row(chroma_idx);
            let energy: f32 = power_spectrum.iter()
                .zip(filter.iter())
                .map(|(&power, &weight)| power * weight)
                .sum();
            chroma_spectrum[chroma_idx] = energy;
        }
        
        chroma_spectrum
    }
    
    /// Convert frequency to pitch class (0-11)
    pub fn freq_to_pitch_class(freq: f32) -> usize {
        let midi_note = 69.0 + 12.0 * (freq / 440.0).log2();
        let pitch_class = (midi_note.round() as i32) % 12;
        if pitch_class < 0 {
            (pitch_class + 12) as usize
        } else {
            pitch_class as usize
        }
    }
    
    /// Compute chroma weights for a given frequency
    fn compute_chroma_weights(frequency: f32, fmin: f32) -> Vec<f32> {
        let mut weights = vec![0.0; 12];
        
        if frequency < fmin || frequency > 8000.0 {
            return weights;
        }
        
        // Convert frequency to MIDI note number
        let midi_note = 69.0 + 12.0 * (frequency / 440.0).log2();
        
        // Map to chroma class (0-11)
        let chroma_class = (midi_note as i32) % 12;
        let chroma_idx = if chroma_class < 0 {
            (chroma_class + 12) as usize
        } else {
            chroma_class as usize
        };
        
        if chroma_idx < 12 {
            // Use triangular window around the chroma class
            let fractional_part = midi_note - midi_note.floor();
            
            weights[chroma_idx] = 1.0 - fractional_part.abs();
            
            // Add contribution to neighboring chroma classes
            let prev_idx = if chroma_idx == 0 { 11 } else { chroma_idx - 1 };
            let next_idx = if chroma_idx == 11 { 0 } else { chroma_idx + 1 };
            
            if fractional_part > 0.0 {
                weights[next_idx] = fractional_part;
            } else {
                weights[prev_idx] = -fractional_part;
            }
        }
        
        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_filterbank_creation() {
        let filterbank = MelFilterbank::new(1025, 80, 22050.0, 80.0, 8000.0).unwrap();
        assert_eq!(filterbank.n_mels(), 80);
        assert_eq!(filterbank.filters.shape(), &[80, 1025]);
    }

    #[test]
    fn test_chroma_filterbank_creation() {
        let filterbank = ChromaFilterbank::new(1025, 22050.0, 80.0).unwrap();
        assert_eq!(filterbank.filters.shape(), &[12, 1025]);
    }

    #[test]
    fn test_hz_to_mel_conversion() {
        let mel_1000 = MelFilterbank::hz_to_mel(1000.0);
        assert!((mel_1000 - 1000.0).abs() < 100.0); // Approximately 1000 mel
        
        let hz_back = MelFilterbank::mel_to_hz(mel_1000);
        assert!((hz_back - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_mel_to_hz_conversion() {
        let hz_440 = MelFilterbank::mel_to_hz(MelFilterbank::hz_to_mel(440.0));
        assert!((hz_440 - 440.0).abs() < 1.0);
    }

    #[test]
    fn test_pitch_class_conversion() {
        let pitch_class_440 = ChromaFilterbank::freq_to_pitch_class(440.0); // A4
        assert_eq!(pitch_class_440, 9); // A is the 9th pitch class (0-indexed)
        
        let pitch_class_261_6 = ChromaFilterbank::freq_to_pitch_class(261.6); // C4
        assert_eq!(pitch_class_261_6, 0); // C is the 0th pitch class
    }

    #[test]
    fn test_octave_equivalence() {
        // Frequencies one octave apart should map to the same pitch class
        let pc_220 = ChromaFilterbank::freq_to_pitch_class(220.0); // A3
        let pc_440 = ChromaFilterbank::freq_to_pitch_class(440.0); // A4
        let pc_880 = ChromaFilterbank::freq_to_pitch_class(880.0); // A5
        
        assert_eq!(pc_220, pc_440);
        assert_eq!(pc_440, pc_880);
        assert_eq!(pc_220, 9); // All should be A (9th pitch class)
    }

    #[test]
    fn test_mel_filterbank_application() {
        let filterbank = MelFilterbank::new(1025, 10, 22050.0, 80.0, 8000.0).unwrap();
        
        // Create test power spectrum with peak at specific frequency
        let mut power_spectrum = vec![0.0; 1025];
        power_spectrum[100] = 1.0; // Peak at bin 100
        
        let mel_spectrum = filterbank.apply(&power_spectrum);
        
        assert_eq!(mel_spectrum.len(), 10);
        
        // Some mel channels should have non-zero energy
        let total_energy: f32 = mel_spectrum.iter().sum();
        assert!(total_energy > 0.0);
    }

    #[test]
    fn test_chroma_filterbank_application() {
        let filterbank = ChromaFilterbank::new(1025, 22050.0, 80.0).unwrap();
        
        // Create test power spectrum
        let mut power_spectrum = vec![0.0; 1025];
        power_spectrum[200] = 1.0; // Peak at bin 200
        
        let chroma_spectrum = filterbank.apply(&power_spectrum);
        
        assert_eq!(chroma_spectrum.len(), 12);
        
        // Some chroma channels should have non-zero energy
        let total_energy: f32 = chroma_spectrum.iter().sum();
        assert!(total_energy > 0.0);
    }

    #[test]
    fn test_chroma_weights_sum() {
        let weights = ChromaFilterbank::compute_chroma_weights(440.0, 80.0);
        
        assert_eq!(weights.len(), 12);
        
        // Weights should sum to approximately 1.0 for a valid frequency
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.1, "Weights sum: {}", sum);
        
        // Should have non-zero weight for A (index 9)
        assert!(weights[9] > 0.5);
    }

    #[test]
    fn test_frequency_range_limits() {
        // Test frequencies outside valid range
        let weights_low = ChromaFilterbank::compute_chroma_weights(50.0, 80.0);
        let weights_high = ChromaFilterbank::compute_chroma_weights(10000.0, 80.0);
        
        // Both should be all zeros
        assert!(weights_low.iter().all(|&x| x == 0.0));
        assert!(weights_high.iter().all(|&x| x == 0.0));
    }
}