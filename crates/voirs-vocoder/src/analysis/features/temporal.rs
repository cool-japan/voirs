//! Temporal feature extraction and computation
//!
//! This module provides temporal feature extraction including:
//! - Onset density
//! - Tempo estimation
//! - Rhythmic regularity
//! - Energy envelope statistics
//! - ADSR envelope analysis

use crate::{Result, VocoderError};
use ndarray::{Array1, Array2};

/// Temporal feature vector
#[derive(Debug, Clone)]
pub struct TemporalFeatureVector {
    /// Onset density
    pub onset_density: f32,
    
    /// Tempo estimate
    pub tempo: f32,
    
    /// Rhythmic regularity
    pub rhythmic_regularity: f32,
    
    /// Energy envelope statistics
    pub energy_stats: EnergyStatistics,
    
    /// Temporal centroid
    pub temporal_centroid: f32,
    
    /// Attack time
    pub attack_time: f32,
    
    /// Decay time
    pub decay_time: f32,
    
    /// Sustain level
    pub sustain_level: f32,
    
    /// Release time
    pub release_time: f32,
}

/// Energy envelope statistics
#[derive(Debug, Clone)]
pub struct EnergyStatistics {
    /// Mean energy
    pub mean: f32,
    
    /// Energy variance
    pub variance: f32,
    
    /// Energy skewness
    pub skewness: f32,
    
    /// Energy kurtosis
    pub kurtosis: f32,
    
    /// Dynamic range
    pub dynamic_range: f32,
}

impl Default for TemporalFeatureVector {
    fn default() -> Self {
        Self {
            onset_density: 0.0,
            tempo: 0.0,
            rhythmic_regularity: 0.0,
            energy_stats: EnergyStatistics::default(),
            temporal_centroid: 0.0,
            attack_time: 0.0,
            decay_time: 0.0,
            sustain_level: 0.0,
            release_time: 0.0,
        }
    }
}

impl Default for EnergyStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            variance: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            dynamic_range: 0.0,
        }
    }
}

/// Temporal feature computation methods
pub trait TemporalFeatureComputer {
    /// Extract temporal features from audio
    fn compute_temporal_features(&self, audio: &[f32]) -> Result<TemporalFeatureVector>;
    
    /// Extract temporal features from samples and power spectrogram
    fn extract_temporal_features(&self, samples: &Array1<f32>, power_spectrogram: &Array2<f32>) -> Result<TemporalFeatureVector>;
    
    /// Compute energy envelope from power spectrogram
    fn compute_energy_envelope(&self, power_spectrogram: &Array2<f32>) -> Vec<f32>;
    
    /// Compute energy statistics
    fn compute_energy_statistics(&self, energy_envelope: &[f32]) -> EnergyStatistics;
    
    /// Compute onset density
    fn compute_onset_density(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Estimate tempo
    fn estimate_tempo(&self, power_spectrogram: &Array2<f32>) -> Result<f32>;
    
    /// Compute rhythmic regularity
    fn compute_rhythmic_regularity(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Compute temporal centroid
    fn compute_temporal_centroid(&self, energy_envelope: &[f32]) -> f32;
    
    /// Compute ADSR envelope parameters
    fn compute_adsr_envelope(&self, samples: &Array1<f32>) -> (f32, f32, f32, f32);
}

impl TemporalFeatureComputer for crate::analysis::features::FeatureExtractor {
    fn compute_temporal_features(&self, audio: &[f32]) -> Result<TemporalFeatureVector> {
        let samples = Array1::from_vec(audio.to_vec());
        let power_spectrogram = self.compute_power_spectrogram_const(&samples)?;
        self.extract_temporal_features(&samples, &power_spectrogram)
    }

    fn extract_temporal_features(&self, samples: &Array1<f32>, power_spectrogram: &Array2<f32>) -> Result<TemporalFeatureVector> {
        let energy_envelope = self.compute_energy_envelope(power_spectrogram);
        let energy_stats = self.compute_energy_statistics(&energy_envelope);
        
        let onset_density = self.compute_onset_density(power_spectrogram);
        let tempo = self.estimate_tempo(power_spectrogram)?;
        let rhythmic_regularity = self.compute_rhythmic_regularity(power_spectrogram);
        let temporal_centroid = self.compute_temporal_centroid(&energy_envelope);
        
        let (attack_time, decay_time, sustain_level, release_time) = self.compute_adsr_envelope(samples);
        
        Ok(TemporalFeatureVector {
            onset_density,
            tempo,
            rhythmic_regularity,
            energy_stats,
            temporal_centroid,
            attack_time,
            decay_time,
            sustain_level,
            release_time,
        })
    }

    fn compute_energy_envelope(&self, power_spectrogram: &Array2<f32>) -> Vec<f32> {
        power_spectrogram.rows()
            .into_iter()
            .map(|row| row.sum())
            .collect()
    }
    
    fn compute_energy_statistics(&self, energy_envelope: &[f32]) -> EnergyStatistics {
        if energy_envelope.is_empty() {
            return EnergyStatistics::default();
        }
        
        let mean = energy_envelope.iter().sum::<f32>() / energy_envelope.len() as f32;
        let variance = energy_envelope.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / energy_envelope.len() as f32;
        
        let std_dev = variance.sqrt();
        let skewness = if std_dev > 1e-10 {
            energy_envelope.iter()
                .map(|&x| ((x - mean) / std_dev).powi(3))
                .sum::<f32>() / energy_envelope.len() as f32
        } else {
            0.0
        };
        
        let kurtosis = if std_dev > 1e-10 {
            energy_envelope.iter()
                .map(|&x| ((x - mean) / std_dev).powi(4))
                .sum::<f32>() / energy_envelope.len() as f32 - 3.0
        } else {
            0.0
        };
        
        let min_energy = energy_envelope.iter().copied().fold(f32::INFINITY, f32::min);
        let max_energy = energy_envelope.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let dynamic_range = if max_energy > 1e-10 && min_energy > 1e-10 {
            20.0 * (max_energy / min_energy).log10()
        } else {
            0.0
        };
        
        EnergyStatistics {
            mean,
            variance,
            skewness,
            kurtosis,
            dynamic_range,
        }
    }

    fn compute_onset_density(&self, power_spectrogram: &Array2<f32>) -> f32 {
        if power_spectrogram.nrows() < 2 {
            return 0.0;
        }
        
        let mut onset_count = 0;
        let energy_envelope = self.compute_energy_envelope(power_spectrogram);
        
        // Simple peak detection for onset estimation
        for i in 1..(energy_envelope.len() - 1) {
            let current = energy_envelope[i];
            let prev = energy_envelope[i - 1];
            let next = energy_envelope[i + 1];
            
            // Detect local maxima that exceed threshold
            if current > prev && current > next {
                let threshold = energy_envelope.iter().sum::<f32>() / energy_envelope.len() as f32 * 1.5;
                if current > threshold {
                    onset_count += 1;
                }
            }
        }
        
        // Convert to density (onsets per second)
        let duration_seconds = power_spectrogram.nrows() as f32 * self.config.hop_length as f32 / self.config.sample_rate;
        if duration_seconds > 0.0 {
            onset_count as f32 / duration_seconds
        } else {
            0.0
        }
    }

    fn estimate_tempo(&self, power_spectrogram: &Array2<f32>) -> Result<f32> {
        if power_spectrogram.nrows() < 4 {
            return Ok(120.0); // Default tempo
        }
        
        let energy_envelope = self.compute_energy_envelope(power_spectrogram);
        
        // Simple autocorrelation-based tempo estimation
        let max_lag = (energy_envelope.len() / 4).min(200); // Limit search range
        let mut best_correlation = 0.0;
        let mut best_lag = 1;
        
        for lag in 10..max_lag {
            let mut correlation = 0.0;
            let mut count = 0;
            
            for i in lag..energy_envelope.len() {
                correlation += energy_envelope[i] * energy_envelope[i - lag];
                count += 1;
            }
            
            if count > 0 {
                correlation /= count as f32;
                if correlation > best_correlation {
                    best_correlation = correlation;
                    best_lag = lag;
                }
            }
        }
        
        // Convert lag to BPM
        let frame_rate = self.config.sample_rate / self.config.hop_length as f32;
        let period_seconds = best_lag as f32 / frame_rate;
        let tempo = if period_seconds > 0.0 {
            60.0 / period_seconds
        } else {
            120.0
        };
        
        // Clamp to reasonable range
        Ok(tempo.clamp(60.0, 200.0))
    }

    fn compute_rhythmic_regularity(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let energy_envelope = self.compute_energy_envelope(power_spectrogram);
        if energy_envelope.len() < 4 {
            return 0.0;
        }
        
        // Compute variance in energy differences (regularity measure)
        let diffs: Vec<f32> = energy_envelope.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();
        
        if diffs.is_empty() {
            return 0.0;
        }
        
        let mean_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;
        let variance = diffs.iter()
            .map(|&x| (x - mean_diff).powi(2))
            .sum::<f32>() / diffs.len() as f32;
        
        // Regularity is inverse of variance (normalized)
        if variance > 0.0 {
            1.0 / (1.0 + variance.sqrt())
        } else {
            1.0
        }
    }

    fn compute_temporal_centroid(&self, energy_envelope: &[f32]) -> f32 {
        if energy_envelope.is_empty() {
            return 0.0;
        }
        
        let total_energy: f32 = energy_envelope.iter().sum();
        if total_energy <= 1e-10 {
            return 0.5; // Center if no energy
        }
        
        let weighted_sum: f32 = energy_envelope.iter()
            .enumerate()
            .map(|(i, &energy)| i as f32 * energy)
            .sum();
        
        let centroid = weighted_sum / total_energy;
        
        // Normalize to [0, 1]
        centroid / energy_envelope.len() as f32
    }

    fn compute_adsr_envelope(&self, samples: &Array1<f32>) -> (f32, f32, f32, f32) {
        if samples.len() < 4 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        
        let sample_rate = self.config.sample_rate;
        let envelope = self.compute_amplitude_envelope(samples);
        
        // Find attack peak
        let peak_idx = envelope.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        // Attack time: time to reach peak
        let attack_time = peak_idx as f32 / sample_rate;
        
        // Find sustain region (relatively stable part)
        let sustain_start = peak_idx + (sample_rate * 0.1) as usize; // Skip initial decay
        let sustain_end = (envelope.len() * 3 / 4).min(sustain_start + (sample_rate * 0.5) as usize);
        
        let sustain_level = if sustain_start < sustain_end && sustain_end <= envelope.len() {
            envelope[sustain_start..sustain_end].iter().sum::<f32>() / (sustain_end - sustain_start) as f32
        } else {
            envelope.get(peak_idx).copied().unwrap_or(0.0) * 0.5
        };
        
        // Decay time: time from peak to sustain level
        let mut decay_idx = peak_idx;
        let sustain_threshold = sustain_level * 1.1;
        for i in peak_idx..envelope.len() {
            if envelope[i] <= sustain_threshold {
                decay_idx = i;
                break;
            }
        }
        let decay_time = (decay_idx.saturating_sub(peak_idx)) as f32 / sample_rate;
        
        // Release time: time from sustain to near zero
        let release_start = sustain_end.min(envelope.len() - 1);
        let mut release_time = 0.0;
        
        if release_start < envelope.len() - 1 {
            let target_level = sustain_level * 0.1;
            for i in release_start..envelope.len() {
                if envelope[i] <= target_level {
                    release_time = (i - release_start) as f32 / sample_rate;
                    break;
                }
            }
        }
        
        (attack_time, decay_time, sustain_level, release_time)
    }
}

impl crate::analysis::features::FeatureExtractor {
    /// Compute amplitude envelope for ADSR analysis
    fn compute_amplitude_envelope(&self, samples: &Array1<f32>) -> Vec<f32> {
        let window_size = (self.config.sample_rate * 0.01) as usize; // 10ms window
        let hop_size = window_size / 2;
        
        let mut envelope = Vec::new();
        
        for start in (0..samples.len()).step_by(hop_size) {
            let end = (start + window_size).min(samples.len());
            if start < end {
                let window = &samples.as_slice().unwrap()[start..end];
                let rms = (window.iter().map(|&x| x * x).sum::<f32>() / window.len() as f32).sqrt();
                envelope.push(rms);
            }
        }
        
        envelope
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_feature_vector_default() {
        let features = TemporalFeatureVector::default();
        assert_eq!(features.onset_density, 0.0);
        assert_eq!(features.tempo, 0.0);
        assert_eq!(features.rhythmic_regularity, 0.0);
        assert_eq!(features.temporal_centroid, 0.0);
    }

    #[test]
    fn test_energy_statistics_default() {
        let stats = EnergyStatistics::default();
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.variance, 0.0);
        assert_eq!(stats.skewness, 0.0);
        assert_eq!(stats.kurtosis, 0.0);
        assert_eq!(stats.dynamic_range, 0.0);
    }

    #[test]
    fn test_temporal_centroid_computation() {
        // Mock implementation for testing
        struct MockExtractor;
        
        impl TemporalFeatureComputer for MockExtractor {
            fn compute_temporal_features(&self, _audio: &[f32]) -> Result<TemporalFeatureVector> {
                unimplemented!()
            }
            
            fn extract_temporal_features(&self, _samples: &Array1<f32>, _power_spectrogram: &Array2<f32>) -> Result<TemporalFeatureVector> {
                unimplemented!()
            }
            
            fn compute_energy_envelope(&self, _power_spectrogram: &Array2<f32>) -> Vec<f32> {
                unimplemented!()
            }
            
            fn compute_energy_statistics(&self, _energy_envelope: &[f32]) -> EnergyStatistics {
                unimplemented!()
            }
            
            fn compute_onset_density(&self, _power_spectrogram: &Array2<f32>) -> f32 {
                unimplemented!()
            }
            
            fn estimate_tempo(&self, _power_spectrogram: &Array2<f32>) -> Result<f32> {
                unimplemented!()
            }
            
            fn compute_rhythmic_regularity(&self, _power_spectrogram: &Array2<f32>) -> f32 {
                unimplemented!()
            }
            
            fn compute_temporal_centroid(&self, energy_envelope: &[f32]) -> f32 {
                if energy_envelope.is_empty() {
                    return 0.0;
                }
                
                let total_energy: f32 = energy_envelope.iter().sum();
                if total_energy <= 1e-10 {
                    return 0.5;
                }
                
                let weighted_sum: f32 = energy_envelope.iter()
                    .enumerate()
                    .map(|(i, &energy)| i as f32 * energy)
                    .sum();
                
                let centroid = weighted_sum / total_energy;
                centroid / energy_envelope.len() as f32
            }
            
            fn compute_adsr_envelope(&self, _samples: &Array1<f32>) -> (f32, f32, f32, f32) {
                unimplemented!()
            }
        }
        
        let extractor = MockExtractor;
        let energy_envelope = vec![0.1, 0.3, 0.8, 0.6, 0.2];
        let centroid = extractor.compute_temporal_centroid(&energy_envelope);
        
        // Should be weighted toward the peak at index 2
        assert!(centroid > 0.3 && centroid < 0.7);
    }
}