//! Chroma feature extraction
//!
//! This module provides chroma feature extraction including:
//! - 12-dimensional chroma vectors
//! - Pitch class profile computation
//! - Harmonic analysis for music information retrieval

use crate::{Result, VocoderError};
use ndarray::{Array1, Array2};

/// Chroma feature computation methods
pub trait ChromaFeatureComputer {
    /// Compute chroma features from audio
    fn compute_chroma(&self, audio: &[f32]) -> Result<Array2<f32>>;
    
    /// Compute chroma features from power spectrogram
    fn compute_chroma_from_spectrogram(&self, power_spectrogram: &Array2<f32>) -> Result<Array2<f32>>;
    
    /// Apply chroma filterbank to spectrum
    fn apply_chroma_filterbank(&self, spectrum: &[f32]) -> Vec<f32>;
}

impl ChromaFeatureComputer for crate::analysis::features::FeatureExtractor {
    fn compute_chroma(&self, audio: &[f32]) -> Result<Array2<f32>> {
        let samples = Array1::from_vec(audio.to_vec());
        let power_spectrogram = self.compute_power_spectrogram_const(&samples)?;
        self.compute_chroma_from_spectrogram(&power_spectrogram)
    }

    fn compute_chroma_from_spectrogram(&self, power_spectrogram: &Array2<f32>) -> Result<Array2<f32>> {
        let n_frames = power_spectrogram.nrows();
        let mut chroma = Array2::zeros((n_frames, 12));
        
        for frame_idx in 0..n_frames {
            let frame = power_spectrogram.row(frame_idx);
            let chroma_frame = self.chroma_filterbank.apply(frame.as_slice().unwrap());
            
            for (chroma_idx, &chroma_value) in chroma_frame.iter().enumerate() {
                if chroma_idx < 12 {
                    chroma[[frame_idx, chroma_idx]] = chroma_value;
                }
            }
        }
        
        // Normalize each chroma vector
        for frame_idx in 0..n_frames {
            let mut row = chroma.row_mut(frame_idx);
            let norm: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            
            if norm > 1e-10 {
                row.mapv_inplace(|x| x / norm);
            }
        }
        
        Ok(chroma)
    }

    fn apply_chroma_filterbank(&self, spectrum: &[f32]) -> Vec<f32> {
        self.chroma_filterbank.apply(spectrum)
    }
}

/// Utility functions for chroma computation
impl crate::analysis::features::FeatureExtractor {
    /// Create chroma filterbank weights for a given frequency bin
    pub(crate) fn compute_chroma_weights(frequency: f32, fmin: f32) -> Vec<f32> {
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
    
    /// Apply harmonic reinforcement to chroma features
    pub(crate) fn apply_harmonic_reinforcement(&self, chroma: &mut Array2<f32>) {
        let harmonic_weights = [1.0, 0.5, 0.33, 0.25, 0.2]; // Weights for harmonics
        
        for mut frame in chroma.rows_mut() {
            let original_frame = frame.to_owned();
            
            for (chroma_idx, &original_value) in original_frame.iter().enumerate() {
                if original_value > 1e-10 {
                    // Add harmonic contributions
                    for (harmonic, &weight) in harmonic_weights.iter().enumerate().skip(1) {
                        let harmonic_chroma = (chroma_idx + harmonic * 7) % 12; // Perfect fifth intervals
                        frame[harmonic_chroma] += original_value * weight;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::features::FeatureConfig;

    #[test]
    fn test_chroma_weights_440hz() {
        // 440 Hz should map to A (chroma class 9)
        let weights = crate::analysis::features::FeatureExtractor::compute_chroma_weights(440.0, 80.0);
        
        assert_eq!(weights.len(), 12);
        
        // A should have the highest weight
        let max_idx = weights.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        
        assert_eq!(max_idx, 9); // A is chroma class 9
        assert!(weights[9] > 0.8); // Should be close to 1.0
    }

    #[test]
    fn test_chroma_normalization() {
        let config = FeatureConfig::default();
        let extractor = crate::analysis::features::FeatureExtractor::new(config).unwrap();
        
        // Create test power spectrogram
        let mut power_spec = Array2::zeros((3, 100));
        
        // Add some energy to specific bins
        power_spec[[0, 10]] = 4.0; // Some frequency
        power_spec[[1, 20]] = 9.0; // Another frequency
        power_spec[[2, 30]] = 1.0; // Third frequency
        
        let chroma = extractor.compute_chroma_from_spectrogram(&power_spec).unwrap();
        
        assert_eq!(chroma.shape(), &[3, 12]);
        
        // Check that each frame is normalized (L2 norm should be ≈ 1 for non-zero frames)
        for frame_idx in 0..3 {
            let frame = chroma.row(frame_idx);
            let norm: f32 = frame.iter().map(|&x| x * x).sum::<f32>().sqrt();
            
            // If there's any energy, norm should be close to 1
            if frame.iter().any(|&x| x > 1e-6) {
                assert!((norm - 1.0).abs() < 1e-3, "Frame {} norm: {}", frame_idx, norm);
            }
        }
    }

    #[test]
    fn test_chroma_weights_octave_equivalence() {
        // Frequencies one octave apart should map to the same chroma class
        let weights_220 = crate::analysis::features::FeatureExtractor::compute_chroma_weights(220.0, 80.0);
        let weights_440 = crate::analysis::features::FeatureExtractor::compute_chroma_weights(440.0, 80.0);
        let weights_880 = crate::analysis::features::FeatureExtractor::compute_chroma_weights(880.0, 80.0);
        
        // Find the dominant chroma class for each
        let find_max_chroma = |weights: &[f32]| {
            weights.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        
        let chroma_220 = find_max_chroma(&weights_220);
        let chroma_440 = find_max_chroma(&weights_440);
        let chroma_880 = find_max_chroma(&weights_880);
        
        // All should map to the same chroma class (A = 9)
        assert_eq!(chroma_220, chroma_440);
        assert_eq!(chroma_440, chroma_880);
        assert_eq!(chroma_220, 9); // A
    }

    #[test]
    fn test_chroma_frequency_range() {
        // Test frequencies outside the valid range
        let weights_low = crate::analysis::features::FeatureExtractor::compute_chroma_weights(50.0, 80.0);
        let weights_high = crate::analysis::features::FeatureExtractor::compute_chroma_weights(10000.0, 80.0);
        
        // Both should be all zeros
        assert!(weights_low.iter().all(|&x| x == 0.0));
        assert!(weights_high.iter().all(|&x| x == 0.0));
        
        // Valid frequency should have non-zero weights
        let weights_valid = crate::analysis::features::FeatureExtractor::compute_chroma_weights(440.0, 80.0);
        assert!(weights_valid.iter().any(|&x| x > 0.0));
    }
}