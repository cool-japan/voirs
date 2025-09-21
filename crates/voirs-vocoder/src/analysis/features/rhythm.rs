//! Rhythm and beat feature extraction
//!
//! This module provides rhythm feature extraction including:
//! - Beat strength
//! - Meter clarity  
//! - Syncopation index
//! - Pulse clarity
//! - Tempo stability

use crate::{Result, VocoderError};
use ndarray::{Array1, Array2};

/// Rhythm and beat features
#[derive(Debug, Clone)]
pub struct RhythmFeatureVector {
    /// Beat strength
    pub beat_strength: f32,
    
    /// Meter clarity
    pub meter_clarity: f32,
    
    /// Syncopation index
    pub syncopation: f32,
    
    /// Pulse clarity
    pub pulse_clarity: f32,
    
    /// Tempo stability
    pub tempo_stability: f32,
}

impl Default for RhythmFeatureVector {
    fn default() -> Self {
        Self {
            beat_strength: 0.0,
            meter_clarity: 0.0,
            syncopation: 0.0,
            pulse_clarity: 0.0,
            tempo_stability: 0.0,
        }
    }
}

/// Rhythm feature computation methods
pub trait RhythmFeatureComputer {
    /// Extract rhythm features from audio
    fn compute_rhythm_features(&self, audio: &[f32]) -> Result<RhythmFeatureVector>;
    
    /// Extract rhythm features from samples
    fn extract_rhythm_features(&self, samples: &Array1<f32>) -> Result<RhythmFeatureVector>;
    
    /// Compute beat strength
    fn compute_beat_strength(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Compute meter clarity
    fn compute_meter_clarity(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Compute syncopation index
    fn compute_syncopation(&self, power_spectrogram: &Array2<f32>) -> f32;
    
    /// Compute pulse clarity
    fn compute_pulse_clarity(&self, samples: &Array1<f32>) -> f32;
    
    /// Compute tempo stability
    fn compute_tempo_stability(&self, power_spectrogram: &Array2<f32>) -> f32;
}

impl RhythmFeatureComputer for crate::analysis::features::FeatureExtractor {
    fn compute_rhythm_features(&self, audio: &[f32]) -> Result<RhythmFeatureVector> {
        let samples = Array1::from_vec(audio.to_vec());
        self.extract_rhythm_features(&samples)
    }

    fn extract_rhythm_features(&self, samples: &Array1<f32>) -> Result<RhythmFeatureVector> {
        // Compute power spectrogram for rhythm analysis
        let power_spectrogram = self.compute_power_spectrogram_const(samples)?;
        
        // Beat strength (peak detection in onset function)
        let beat_strength = self.compute_beat_strength(&power_spectrogram);
        
        // Meter clarity (periodicity strength)
        let meter_clarity = self.compute_meter_clarity(&power_spectrogram);
        
        // Syncopation index (deviation from expected beat patterns)
        let syncopation = self.compute_syncopation(&power_spectrogram);
        
        // Pulse clarity (regularity of rhythmic pulses)
        let pulse_clarity = self.compute_pulse_clarity(samples);
        
        // Tempo stability (consistency of tempo over time)
        let tempo_stability = self.compute_tempo_stability(&power_spectrogram);
        
        Ok(RhythmFeatureVector {
            beat_strength,
            meter_clarity,
            syncopation,
            pulse_clarity,
            tempo_stability,
        })
    }

    fn compute_beat_strength(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let energy_envelope: Vec<f32> = power_spectrogram.rows()
            .into_iter()
            .map(|row| row.sum())
            .collect();
        
        if energy_envelope.len() < 3 {
            return 0.0;
        }
        
        // Compute onset strength using spectral flux
        let mut onset_strength = Vec::new();
        for i in 1..energy_envelope.len() {
            let diff = (energy_envelope[i] - energy_envelope[i - 1]).max(0.0);
            onset_strength.push(diff);
        }
        
        if onset_strength.is_empty() {
            return 0.0;
        }
        
        // Beat strength is the average of significant onsets
        let threshold = onset_strength.iter().sum::<f32>() / onset_strength.len() as f32 * 1.5;
        let significant_onsets: Vec<f32> = onset_strength.iter()
            .filter(|&&x| x > threshold)
            .copied()
            .collect();
        
        if significant_onsets.is_empty() {
            0.0
        } else {
            significant_onsets.iter().sum::<f32>() / significant_onsets.len() as f32
        }
    }

    fn compute_meter_clarity(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let energy_envelope: Vec<f32> = power_spectrogram.rows()
            .into_iter()
            .map(|row| row.sum())
            .collect();
        
        if energy_envelope.len() < 8 {
            return 0.0;
        }
        
        // Test different meter periods (4/4, 3/4, etc.)
        let frame_rate = self.config.sample_rate / self.config.hop_length as f32;
        let periods_to_test = vec![
            (60.0 / 120.0 * frame_rate) as usize, // 120 BPM quarter notes
            (60.0 / 100.0 * frame_rate) as usize, // 100 BPM quarter notes
            (60.0 / 140.0 * frame_rate) as usize, // 140 BPM quarter notes
        ];
        
        let mut max_clarity = 0.0;
        
        for period in periods_to_test {
            if period > 0 && period < energy_envelope.len() / 4 {
                let mut correlation = 0.0;
                let mut count = 0;
                
                for i in period..energy_envelope.len() {
                    correlation += energy_envelope[i] * energy_envelope[i - period];
                    count += 1;
                }
                
                if count > 0 {
                    let normalized_correlation = correlation / count as f32;
                    max_clarity = max_clarity.max(normalized_correlation);
                }
            }
        }
        
        max_clarity
    }

    fn compute_syncopation(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let energy_envelope: Vec<f32> = power_spectrogram.rows()
            .into_iter()
            .map(|row| row.sum())
            .collect();
        
        if energy_envelope.len() < 4 {
            return 0.0;
        }
        
        // Simple syncopation measure: variance in energy at regular intervals
        let frame_rate = self.config.sample_rate / self.config.hop_length as f32;
        let beat_interval = (60.0 / 120.0 * frame_rate) as usize; // Assume 120 BPM
        
        if beat_interval == 0 || beat_interval >= energy_envelope.len() {
            return 0.0;
        }
        
        let mut beat_energies = Vec::new();
        let mut off_beat_energies = Vec::new();
        
        for i in (0..energy_envelope.len()).step_by(beat_interval) {
            if i < energy_envelope.len() {
                beat_energies.push(energy_envelope[i]);
            }
            
            let off_beat_idx = i + beat_interval / 2;
            if off_beat_idx < energy_envelope.len() {
                off_beat_energies.push(energy_envelope[off_beat_idx]);
            }
        }
        
        if beat_energies.is_empty() || off_beat_energies.is_empty() {
            return 0.0;
        }
        
        let beat_avg = beat_energies.iter().sum::<f32>() / beat_energies.len() as f32;
        let off_beat_avg = off_beat_energies.iter().sum::<f32>() / off_beat_energies.len() as f32;
        
        // Syncopation: off-beat energy relative to beat energy
        if beat_avg > 1e-10 {
            off_beat_avg / beat_avg
        } else {
            0.0
        }
    }

    fn compute_pulse_clarity(&self, samples: &Array1<f32>) -> f32 {
        if samples.len() < 1024 {
            return 0.0;
        }
        
        // Compute amplitude envelope for pulse detection
        let window_size = (self.config.sample_rate * 0.02) as usize; // 20ms
        let hop_size = window_size / 2;
        
        let mut envelope = Vec::new();
        for start in (0..samples.len() - window_size).step_by(hop_size) {
            let end = start + window_size;
            let window = samples.slice(ndarray::s![start..end]);
            let rms = (window.iter().map(|&x| x * x).sum::<f32>() / window_size as f32).sqrt();
            envelope.push(rms);
        }
        
        if envelope.len() < 4 {
            return 0.0;
        }
        
        // Measure regularity of pulses using autocorrelation
        let max_lag = envelope.len() / 4;
        let mut max_correlation = 0.0;
        
        for lag in 2..max_lag {
            let mut correlation = 0.0;
            let mut count = 0;
            
            for i in lag..envelope.len() {
                correlation += envelope[i] * envelope[i - lag];
                count += 1;
            }
            
            if count > 0 {
                correlation /= count as f32;
                max_correlation = max_correlation.max(correlation);
            }
        }
        
        max_correlation
    }

    fn compute_tempo_stability(&self, power_spectrogram: &Array2<f32>) -> f32 {
        let energy_envelope: Vec<f32> = power_spectrogram.rows()
            .into_iter()
            .map(|row| row.sum())
            .collect();
        
        if energy_envelope.len() < 8 {
            return 0.0;
        }
        
        // Divide into segments and compute tempo for each
        let segment_size = energy_envelope.len() / 4;
        if segment_size < 4 {
            return 0.0;
        }
        
        let mut tempos = Vec::new();
        
        for segment_start in (0..energy_envelope.len() - segment_size).step_by(segment_size) {
            let segment_end = (segment_start + segment_size).min(energy_envelope.len());
            let segment = &energy_envelope[segment_start..segment_end];
            
            // Simple tempo estimation for segment
            let max_lag = segment.len() / 4;
            let mut best_lag = 1;
            let mut best_correlation = 0.0;
            
            for lag in 2..max_lag {
                let mut correlation = 0.0;
                let mut count = 0;
                
                for i in lag..segment.len() {
                    correlation += segment[i] * segment[i - lag];
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
            
            // Convert lag to tempo
            let frame_rate = self.config.sample_rate / self.config.hop_length as f32;
            let period_seconds = best_lag as f32 / frame_rate;
            if period_seconds > 0.0 {
                let tempo = 60.0 / period_seconds;
                tempos.push(tempo.clamp(60.0, 200.0));
            }
        }
        
        if tempos.len() < 2 {
            return 0.0;
        }
        
        // Tempo stability is inverse of tempo variance
        let mean_tempo = tempos.iter().sum::<f32>() / tempos.len() as f32;
        let tempo_variance = tempos.iter()
            .map(|&t| (t - mean_tempo).powi(2))
            .sum::<f32>() / tempos.len() as f32;
        
        1.0 / (1.0 + tempo_variance.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rhythm_feature_vector_default() {
        let features = RhythmFeatureVector::default();
        assert_eq!(features.beat_strength, 0.0);
        assert_eq!(features.meter_clarity, 0.0);
        assert_eq!(features.syncopation, 0.0);
        assert_eq!(features.pulse_clarity, 0.0);
        assert_eq!(features.tempo_stability, 0.0);
    }
}