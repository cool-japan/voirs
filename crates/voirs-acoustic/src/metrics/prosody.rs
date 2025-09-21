//! Prosody-specific quality metrics
//!
//! This module provides metrics for evaluating prosodic aspects of TTS synthesis,
//! including duration accuracy, pitch correlation, stress pattern preservation,
//! and rhythm naturalness measurements.

use crate::{AcousticError, Result};
use serde::{Deserialize, Serialize};

/// Prosody quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodyMetrics {
    /// Duration prediction accuracy (0-100)
    pub duration_accuracy: f32,
    /// Pitch contour correlation (0-100)
    pub pitch_correlation: f32,
    /// Stress pattern preservation (0-100)
    pub stress_preservation: f32,
    /// Rhythm naturalness score (0-100)
    pub rhythm_naturalness: f32,
    /// Overall prosody score (0-100)
    pub overall_score: f32,
}

impl Default for ProsodyMetrics {
    fn default() -> Self {
        Self {
            duration_accuracy: 0.0,
            pitch_correlation: 0.0,
            stress_preservation: 0.0,
            rhythm_naturalness: 0.0,
            overall_score: 0.0,
        }
    }
}

/// Extracted prosody features from audio
#[derive(Debug, Clone)]
pub struct ProsodyFeatures {
    /// Phoneme-level durations in seconds
    pub durations: Vec<f32>,
    /// Frame-level pitch contour in Hz
    pub pitch_contour: Vec<f32>,
    /// Stress pattern (0=unstressed, 1=primary stress, 2=secondary stress)
    pub stress_pattern: Vec<u8>,
    /// Rhythm features (PVI, timing ratios, etc.)
    pub rhythm_features: RhythmFeatures,
}

/// Rhythm analysis features
#[derive(Debug, Clone)]
pub struct RhythmFeatures {
    /// Pairwise Variability Index for durations
    pub duration_pvi: f32,
    /// Consonant-vowel timing ratio
    pub cv_ratio: f32,
    /// Inter-stress interval variability
    pub isi_variability: f32,
    /// Speaking rate in syllables per second
    pub speaking_rate: f32,
    /// Rhythm regularity measure
    pub rhythm_regularity: f32,
}

/// Prosody quality evaluator
pub struct ProsodyEvaluator {
    /// Sample rate for audio processing
    sample_rate: u32,
    /// Frame size for pitch analysis
    #[allow(dead_code)]
    pitch_frame_size: usize,
    /// Hop length for pitch analysis
    pitch_hop_length: usize,
    /// Minimum F0 for pitch tracking
    min_f0: f32,
    /// Maximum F0 for pitch tracking
    max_f0: f32,
}

impl Default for ProsodyEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl ProsodyEvaluator {
    /// Create new prosody evaluator with default parameters
    pub fn new() -> Self {
        Self {
            sample_rate: 22050,
            pitch_frame_size: 1024,
            pitch_hop_length: 256,
            min_f0: 50.0,
            max_f0: 500.0,
        }
    }

    /// Create evaluator with custom parameters
    pub fn with_params(sample_rate: u32, min_f0: f32, max_f0: f32) -> Self {
        Self {
            sample_rate,
            pitch_frame_size: 1024,
            pitch_hop_length: 256,
            min_f0,
            max_f0,
        }
    }

    /// Extract prosody features from mel spectrogram
    pub fn extract_prosody_features(&self, mel_data: &[Vec<f32>]) -> Result<ProsodyFeatures> {
        if mel_data.is_empty() || mel_data[0].is_empty() {
            return Err(AcousticError::InputError(
                "Empty mel spectrogram data".to_string(),
            ));
        }

        // Extract pitch contour
        let pitch_contour = self.extract_pitch_from_mel(mel_data)?;

        // Extract duration information
        let durations = self.extract_durations_from_mel(mel_data)?;

        // Extract stress pattern
        let stress_pattern = self.extract_stress_pattern(mel_data, &pitch_contour)?;

        // Compute rhythm features
        let rhythm_features = self.compute_rhythm_features(&durations, &pitch_contour)?;

        Ok(ProsodyFeatures {
            durations,
            pitch_contour,
            stress_pattern,
            rhythm_features,
        })
    }

    /// Compute duration accuracy between generated and reference
    pub fn compute_duration_accuracy(
        &self,
        generated_durations: &[f32],
        reference_durations: &[f32],
    ) -> Result<f32> {
        if generated_durations.is_empty() || reference_durations.is_empty() {
            return Err(AcousticError::InputError("Empty duration data".to_string()));
        }

        let min_len = generated_durations.len().min(reference_durations.len());
        let mut total_error = 0.0f32;

        for i in 0..min_len {
            let gen_dur = generated_durations[i];
            let ref_dur = reference_durations[i];

            // Compute relative error
            let error = if ref_dur > 0.0 {
                ((gen_dur - ref_dur) / ref_dur).abs()
            } else {
                gen_dur.abs()
            };

            total_error += error;
        }

        let mean_error = total_error / min_len as f32;

        // Convert to accuracy score (0-100)
        let accuracy = ((1.0 - mean_error.min(1.0)) * 100.0).max(0.0);

        Ok(accuracy)
    }

    /// Compute pitch correlation between generated and reference
    pub fn compute_pitch_correlation(
        &self,
        generated_pitch: &[f32],
        reference_pitch: &[f32],
    ) -> Result<f32> {
        if generated_pitch.is_empty() || reference_pitch.is_empty() {
            return Err(AcousticError::InputError("Empty pitch data".to_string()));
        }

        let min_len = generated_pitch.len().min(reference_pitch.len());
        let gen_pitch = &generated_pitch[..min_len];
        let ref_pitch = &reference_pitch[..min_len];

        // Filter out unvoiced frames (0 Hz)
        let voiced_pairs: Vec<(f32, f32)> = gen_pitch
            .iter()
            .zip(ref_pitch.iter())
            .filter(|(&g, &r)| g > 0.0 && r > 0.0)
            .map(|(&g, &r)| (g, r))
            .collect();

        if voiced_pairs.len() < 2 {
            return Ok(0.0);
        }

        let correlation = self.compute_pearson_correlation(
            &voiced_pairs.iter().map(|(g, _)| *g).collect::<Vec<_>>(),
            &voiced_pairs.iter().map(|(_, r)| *r).collect::<Vec<_>>(),
        )?;

        // Convert to 0-100 scale
        Ok((correlation.abs() * 100.0).clamp(0.0, 100.0))
    }

    /// Compute stress pattern preservation
    pub fn compute_stress_preservation(
        &self,
        generated_stress: &[u8],
        reference_stress: &[u8],
    ) -> Result<f32> {
        if generated_stress.is_empty() || reference_stress.is_empty() {
            return Err(AcousticError::InputError(
                "Empty stress pattern data".to_string(),
            ));
        }

        let min_len = generated_stress.len().min(reference_stress.len());
        let mut correct_matches = 0;

        for i in 0..min_len {
            if generated_stress[i] == reference_stress[i] {
                correct_matches += 1;
            }
        }

        let accuracy = (correct_matches as f32 / min_len as f32) * 100.0;

        Ok(accuracy)
    }

    /// Compute rhythm naturalness
    pub fn compute_rhythm_naturalness(
        &self,
        generated_rhythm: &RhythmFeatures,
        reference_rhythm: &RhythmFeatures,
    ) -> Result<f32> {
        // Compare rhythm features
        let pvi_similarity = self.compute_feature_similarity(
            generated_rhythm.duration_pvi,
            reference_rhythm.duration_pvi,
        );

        let cv_similarity =
            self.compute_feature_similarity(generated_rhythm.cv_ratio, reference_rhythm.cv_ratio);

        let isi_similarity = self.compute_feature_similarity(
            generated_rhythm.isi_variability,
            reference_rhythm.isi_variability,
        );

        let rate_similarity = self.compute_feature_similarity(
            generated_rhythm.speaking_rate,
            reference_rhythm.speaking_rate,
        );

        let regularity_similarity = self.compute_feature_similarity(
            generated_rhythm.rhythm_regularity,
            reference_rhythm.rhythm_regularity,
        );

        // Weighted average of similarities
        let naturalness = (pvi_similarity * 0.25
            + cv_similarity * 0.20
            + isi_similarity * 0.20
            + rate_similarity * 0.15
            + regularity_similarity * 0.20)
            * 100.0;

        Ok(naturalness.clamp(0.0, 100.0))
    }

    /// Compute intrinsic duration quality (without reference)
    pub fn compute_intrinsic_duration_quality(&self, durations: &[f32]) -> Result<f32> {
        if durations.is_empty() {
            return Err(AcousticError::InputError("Empty duration data".to_string()));
        }

        // Check for reasonable duration values
        let mean_duration = durations.iter().sum::<f32>() / durations.len() as f32;
        let duration_variance = durations
            .iter()
            .map(|&d| (d - mean_duration).powi(2))
            .sum::<f32>()
            / durations.len() as f32;

        // Quality based on reasonable duration range and variability
        let mean_quality = if mean_duration > 0.01 && mean_duration < 1.0 {
            1.0
        } else {
            0.5
        };
        let variance_quality = if duration_variance > 0.0001 && duration_variance < 0.25 {
            1.0
        } else {
            0.7
        };

        // Check for outliers
        let outlier_penalty = self.compute_outlier_penalty(durations);

        let quality = (mean_quality * 0.4 + variance_quality * 0.4 + outlier_penalty * 0.2) * 100.0;

        Ok(quality.clamp(0.0, 100.0))
    }

    /// Compute intrinsic pitch quality (without reference)
    pub fn compute_intrinsic_pitch_quality(&self, pitch_contour: &[f32]) -> Result<f32> {
        if pitch_contour.is_empty() {
            return Err(AcousticError::InputError("Empty pitch data".to_string()));
        }

        // Filter voiced frames
        let voiced_frames: Vec<f32> = pitch_contour
            .iter()
            .filter(|&&f| f > 0.0)
            .copied()
            .collect();

        if voiced_frames.is_empty() {
            return Ok(50.0); // Neutral score for unvoiced speech
        }

        // Check pitch range and smoothness
        let range_quality = self.compute_pitch_range_quality(&voiced_frames);
        let smoothness_quality = self.compute_pitch_smoothness_quality(&voiced_frames);
        let naturalness_quality = self.compute_pitch_naturalness_quality(&voiced_frames);

        let quality =
            (range_quality * 0.3 + smoothness_quality * 0.4 + naturalness_quality * 0.3) * 100.0;

        Ok(quality.clamp(0.0, 100.0))
    }

    /// Compute intrinsic rhythm quality (without reference)
    pub fn compute_intrinsic_rhythm_quality(
        &self,
        rhythm_features: &RhythmFeatures,
    ) -> Result<f32> {
        // Evaluate rhythm features against natural speech norms

        // PVI should be in reasonable range (20-60 for English)
        let pvi_quality =
            if rhythm_features.duration_pvi > 10.0 && rhythm_features.duration_pvi < 80.0 {
                1.0
            } else {
                0.6
            };

        // CV ratio should be reasonable (0.3-0.8)
        let cv_quality = if rhythm_features.cv_ratio > 0.2 && rhythm_features.cv_ratio < 1.0 {
            1.0
        } else {
            0.7
        };

        // Speaking rate should be natural (2-8 syllables/second)
        let rate_quality =
            if rhythm_features.speaking_rate > 1.0 && rhythm_features.speaking_rate < 10.0 {
                1.0
            } else {
                0.5
            };

        // Rhythm regularity should show some variability (not too regular)
        let regularity_quality =
            if rhythm_features.rhythm_regularity > 0.2 && rhythm_features.rhythm_regularity < 0.9 {
                1.0
            } else {
                0.8
            };

        let quality: f32 = (pvi_quality * 0.25
            + cv_quality * 0.25
            + rate_quality * 0.25
            + regularity_quality * 0.25)
            * 100.0;

        Ok(quality.clamp(0.0, 100.0))
    }

    // Private helper methods

    fn extract_pitch_from_mel(&self, mel_data: &[Vec<f32>]) -> Result<Vec<f32>> {
        let n_frames = mel_data[0].len();
        let mut pitch_contour = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            // Simple pitch estimation from mel spectrogram
            let pitch = self.estimate_pitch_from_mel_frame(mel_data, frame_idx)?;
            pitch_contour.push(pitch);
        }

        // Apply median filtering for smoothness
        self.apply_median_filter(&mut pitch_contour, 5);

        Ok(pitch_contour)
    }

    fn estimate_pitch_from_mel_frame(
        &self,
        mel_data: &[Vec<f32>],
        frame_idx: usize,
    ) -> Result<f32> {
        // Find the mel bin with maximum energy in the expected pitch range
        let mut max_energy = 0.0f32;
        let mut max_bin = 0;

        for (bin_idx, mel_channel) in mel_data.iter().enumerate() {
            if frame_idx < mel_channel.len() {
                let energy = mel_channel[frame_idx].abs();
                if energy > max_energy {
                    max_energy = energy;
                    max_bin = bin_idx;
                }
            }
        }

        // Convert mel bin to approximate frequency
        let mel_freq = 2595.0
            * (1.0
                + max_bin as f32 * (self.sample_rate as f32 / 2.0)
                    / (mel_data.len() as f32 * 700.0))
                .log10();
        let hz = 700.0 * (mel_freq / 2595.0).exp() - 700.0;

        // Return 0 for unvoiced or out-of-range frequencies
        if hz < self.min_f0 || hz > self.max_f0 || max_energy < 0.01 {
            Ok(0.0)
        } else {
            Ok(hz)
        }
    }

    fn extract_durations_from_mel(&self, mel_data: &[Vec<f32>]) -> Result<Vec<f32>> {
        // Simplified duration extraction based on energy changes
        let n_frames = mel_data[0].len();
        let frame_duration = self.pitch_hop_length as f32 / self.sample_rate as f32;

        // Compute frame energies
        let mut frame_energies = Vec::with_capacity(n_frames);
        for frame_idx in 0..n_frames {
            let mut energy = 0.0f32;
            for mel_channel in mel_data {
                if frame_idx < mel_channel.len() {
                    energy += mel_channel[frame_idx].abs();
                }
            }
            frame_energies.push(energy);
        }

        // Detect phoneme boundaries based on energy changes
        let mut durations = Vec::new();
        let mut segment_start = 0;
        let energy_threshold = frame_energies.iter().sum::<f32>() / n_frames as f32 * 0.1;

        for (i, &energy) in frame_energies.iter().enumerate() {
            // Simple boundary detection (energy drops below threshold)
            if energy < energy_threshold || i == n_frames - 1 {
                let segment_duration = (i - segment_start + 1) as f32 * frame_duration;
                durations.push(segment_duration);
                segment_start = i + 1;
            }
        }

        // Ensure we have at least one duration
        if durations.is_empty() {
            durations.push(n_frames as f32 * frame_duration);
        }

        Ok(durations)
    }

    fn extract_stress_pattern(
        &self,
        mel_data: &[Vec<f32>],
        pitch_contour: &[f32],
    ) -> Result<Vec<u8>> {
        let n_frames = mel_data[0].len();
        let mut stress_pattern = Vec::with_capacity(n_frames);

        // Compute frame energies
        let mut frame_energies = Vec::with_capacity(n_frames);
        for frame_idx in 0..n_frames {
            let mut energy = 0.0f32;
            for mel_channel in mel_data {
                if frame_idx < mel_channel.len() {
                    energy += mel_channel[frame_idx].abs();
                }
            }
            frame_energies.push(energy);
        }

        // Detect stress based on energy and pitch peaks
        let energy_threshold = self.compute_percentile(&frame_energies, 0.75);
        let pitch_threshold = self.compute_percentile(pitch_contour, 0.75);

        for frame_idx in 0..n_frames {
            let energy = frame_energies[frame_idx];
            let pitch = pitch_contour[frame_idx];

            let stress_level = if energy > energy_threshold && pitch > pitch_threshold {
                1 // Primary stress
            } else if energy > energy_threshold * 0.7 || pitch > pitch_threshold * 0.8 {
                2 // Secondary stress
            } else {
                0 // Unstressed
            };

            stress_pattern.push(stress_level);
        }

        Ok(stress_pattern)
    }

    fn compute_rhythm_features(
        &self,
        durations: &[f32],
        pitch_contour: &[f32],
    ) -> Result<RhythmFeatures> {
        // Compute Pairwise Variability Index (PVI)
        let duration_pvi = self.compute_pvi(durations);

        // Estimate consonant-vowel ratio (simplified)
        let cv_ratio = self.estimate_cv_ratio(durations, pitch_contour);

        // Compute inter-stress interval variability
        let isi_variability = self.compute_isi_variability(durations);

        // Compute speaking rate
        let speaking_rate = durations.len() as f32 / durations.iter().sum::<f32>();

        // Compute rhythm regularity
        let rhythm_regularity = self.compute_rhythm_regularity(durations);

        Ok(RhythmFeatures {
            duration_pvi,
            cv_ratio,
            isi_variability,
            speaking_rate,
            rhythm_regularity,
        })
    }

    fn compute_pvi(&self, durations: &[f32]) -> f32 {
        if durations.len() < 2 {
            return 0.0;
        }

        let mut pvi_sum = 0.0f32;
        let n_pairs = durations.len() - 1;

        for i in 0..n_pairs {
            let d1 = durations[i];
            let d2 = durations[i + 1];

            if d1 + d2 > 0.0 {
                let pvi_component = 100.0 * (d1 - d2).abs() / (d1 + d2);
                pvi_sum += pvi_component;
            }
        }

        pvi_sum / n_pairs as f32
    }

    fn estimate_cv_ratio(&self, durations: &[f32], pitch_contour: &[f32]) -> f32 {
        // Simplified CV estimation based on duration and voicing
        let mut consonant_duration = 0.0f32;
        let mut vowel_duration = 0.0f32;

        for (i, &duration) in durations.iter().enumerate() {
            // Estimate if it's a vowel based on pitch (voiced segments tend to be vowels)
            let frame_range = i * 10..(i + 1) * 10; // Rough frame mapping
            let is_voiced = frame_range
                .clone()
                .filter_map(|idx| pitch_contour.get(idx))
                .any(|&pitch| pitch > 0.0);

            if is_voiced {
                vowel_duration += duration;
            } else {
                consonant_duration += duration;
            }
        }

        if vowel_duration > 0.0 {
            consonant_duration / vowel_duration
        } else {
            1.0
        }
    }

    fn compute_isi_variability(&self, durations: &[f32]) -> f32 {
        // Simple variability measure
        if durations.len() < 2 {
            return 0.0;
        }

        let mean = durations.iter().sum::<f32>() / durations.len() as f32;
        let variance =
            durations.iter().map(|&d| (d - mean).powi(2)).sum::<f32>() / durations.len() as f32;

        variance.sqrt() / mean
    }

    fn compute_rhythm_regularity(&self, durations: &[f32]) -> f32 {
        if durations.len() < 2 {
            return 0.0;
        }

        // Compute autocorrelation at lag 1
        let mut autocorr = 0.0f32;
        let mut norm = 0.0f32;

        for i in 0..(durations.len() - 1) {
            autocorr += durations[i] * durations[i + 1];
            norm += durations[i] * durations[i];
        }

        if norm > 0.0 {
            autocorr / norm
        } else {
            0.0
        }
    }

    fn compute_pearson_correlation(&self, x: &[f32], y: &[f32]) -> Result<f32> {
        if x.len() != y.len() || x.is_empty() {
            return Err(AcousticError::InputError(
                "Arrays must have same non-zero length".to_string(),
            ));
        }

        let n = x.len() as f32;
        let mean_x = x.iter().sum::<f32>() / n;
        let mean_y = y.iter().sum::<f32>() / n;

        let mut numerator = 0.0f32;
        let mut sum_sq_x = 0.0f32;
        let mut sum_sq_y = 0.0f32;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let x_centered = xi - mean_x;
            let y_centered = yi - mean_y;

            numerator += x_centered * y_centered;
            sum_sq_x += x_centered * x_centered;
            sum_sq_y += y_centered * y_centered;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator > 0.0 {
            Ok(numerator / denominator)
        } else {
            Ok(0.0)
        }
    }

    fn compute_feature_similarity(&self, generated: f32, reference: f32) -> f32 {
        let diff = (generated - reference).abs();
        let max_val = generated.max(reference);

        if max_val > 0.0 {
            1.0 - (diff / max_val).min(1.0)
        } else {
            1.0
        }
    }

    fn compute_outlier_penalty(&self, values: &[f32]) -> f32 {
        if values.len() < 3 {
            return 1.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let std_dev =
            (values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32).sqrt();

        let outlier_count = values
            .iter()
            .filter(|&&v| (v - mean).abs() > 2.0 * std_dev)
            .count();

        let outlier_ratio = outlier_count as f32 / values.len() as f32;

        1.0 - outlier_ratio.min(0.5)
    }

    fn compute_pitch_range_quality(&self, pitch_values: &[f32]) -> f32 {
        if pitch_values.is_empty() {
            return 0.0;
        }

        let min_pitch = pitch_values
            .iter()
            .fold(f32::INFINITY, |min, &val| min.min(val));
        let max_pitch = pitch_values.iter().fold(0.0f32, |max, &val| max.max(val));

        let range = max_pitch - min_pitch;

        // Good pitch range is typically 50-300 Hz
        if range > 30.0 && range < 400.0 {
            1.0
        } else if range > 10.0 && range < 600.0 {
            0.8
        } else {
            0.5
        }
    }

    fn compute_pitch_smoothness_quality(&self, pitch_values: &[f32]) -> f32 {
        if pitch_values.len() < 2 {
            return 1.0;
        }

        let mut total_variation = 0.0f32;

        for i in 1..pitch_values.len() {
            let diff = (pitch_values[i] - pitch_values[i - 1]).abs();
            total_variation += diff;
        }

        let mean_variation = total_variation / (pitch_values.len() - 1) as f32;

        // Smooth pitch should have small frame-to-frame variations
        if mean_variation < 5.0 {
            1.0
        } else if mean_variation < 15.0 {
            0.8
        } else {
            0.5
        }
    }

    fn compute_pitch_naturalness_quality(&self, pitch_values: &[f32]) -> f32 {
        let mean_pitch = pitch_values.iter().sum::<f32>() / pitch_values.len() as f32;

        // Natural pitch ranges
        match () {
            _ if mean_pitch > 80.0 && mean_pitch < 180.0 => 1.0, // Male
            _ if mean_pitch > 150.0 && mean_pitch < 300.0 => 1.0, // Female
            _ if mean_pitch > 200.0 && mean_pitch < 400.0 => 1.0, // Child
            _ => 0.7,                                            // Other ranges
        }
    }

    fn compute_percentile(&self, values: &[f32], percentile: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (percentile * (sorted_values.len() - 1) as f32) as usize;
        sorted_values[index.min(sorted_values.len() - 1)]
    }

    fn apply_median_filter(&self, data: &mut [f32], window_size: usize) {
        if data.len() < window_size || window_size < 3 {
            return;
        }

        let half_window = window_size / 2;
        let mut filtered_data = data.to_vec();

        for i in half_window..(data.len() - half_window) {
            let mut window: Vec<f32> = data[(i - half_window)..=(i + half_window)].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap());
            filtered_data[i] = window[half_window];
        }

        data.copy_from_slice(&filtered_data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_mel_data() -> Vec<Vec<f32>> {
        vec![
            vec![0.1, 0.2, 0.3, 0.2, 0.1],
            vec![0.2, 0.4, 0.6, 0.4, 0.2],
            vec![0.1, 0.3, 0.5, 0.3, 0.1],
            vec![0.05, 0.1, 0.2, 0.1, 0.05],
        ]
    }

    #[test]
    fn test_prosody_evaluator_creation() {
        let evaluator = ProsodyEvaluator::new();
        assert_eq!(evaluator.sample_rate, 22050);
        assert_eq!(evaluator.min_f0, 50.0);
        assert_eq!(evaluator.max_f0, 500.0);
    }

    #[test]
    fn test_prosody_features_extraction() {
        let evaluator = ProsodyEvaluator::new();
        let mel_data = create_test_mel_data();

        let features = evaluator.extract_prosody_features(&mel_data).unwrap();
        assert!(!features.durations.is_empty());
        assert!(!features.pitch_contour.is_empty());
        assert!(!features.stress_pattern.is_empty());
    }

    #[test]
    fn test_duration_accuracy() {
        let evaluator = ProsodyEvaluator::new();
        let durations1 = vec![0.1, 0.2, 0.15, 0.3];
        let durations2 = vec![0.1, 0.2, 0.15, 0.3];

        let accuracy = evaluator
            .compute_duration_accuracy(&durations1, &durations2)
            .unwrap();
        assert_eq!(accuracy, 100.0); // Perfect match
    }

    #[test]
    fn test_pitch_correlation() {
        let evaluator = ProsodyEvaluator::new();
        let pitch1 = vec![100.0, 150.0, 200.0, 150.0, 100.0];
        let pitch2 = vec![100.0, 150.0, 200.0, 150.0, 100.0];

        let correlation = evaluator
            .compute_pitch_correlation(&pitch1, &pitch2)
            .unwrap();
        assert!((correlation - 100.0).abs() < 1.0); // Near perfect correlation
    }

    #[test]
    fn test_stress_preservation() {
        let evaluator = ProsodyEvaluator::new();
        let stress1 = vec![0, 1, 0, 2, 0];
        let stress2 = vec![0, 1, 0, 2, 0];

        let preservation = evaluator
            .compute_stress_preservation(&stress1, &stress2)
            .unwrap();
        assert_eq!(preservation, 100.0); // Perfect match
    }

    #[test]
    fn test_pvi_computation() {
        let evaluator = ProsodyEvaluator::new();
        let durations = vec![0.1, 0.2, 0.1, 0.3, 0.15];

        let pvi = evaluator.compute_pvi(&durations);
        assert!(pvi > 0.0);
        assert!(pvi < 200.0); // Reasonable range
    }

    #[test]
    fn test_rhythm_features() {
        let evaluator = ProsodyEvaluator::new();
        let durations = vec![0.1, 0.2, 0.15, 0.3, 0.12];
        let pitch = vec![100.0, 150.0, 0.0, 200.0, 120.0];

        let rhythm = evaluator
            .compute_rhythm_features(&durations, &pitch)
            .unwrap();

        assert!(rhythm.duration_pvi >= 0.0);
        assert!(rhythm.cv_ratio >= 0.0);
        assert!(rhythm.speaking_rate > 0.0);
        assert!(rhythm.rhythm_regularity >= 0.0);
        assert!(rhythm.rhythm_regularity <= 1.0);
    }

    #[test]
    fn test_intrinsic_quality_metrics() {
        let evaluator = ProsodyEvaluator::new();
        let durations = vec![0.1, 0.2, 0.15, 0.3, 0.12];
        let pitch = vec![100.0, 150.0, 200.0, 180.0, 120.0];
        let rhythm = RhythmFeatures {
            duration_pvi: 40.0,
            cv_ratio: 0.6,
            isi_variability: 0.3,
            speaking_rate: 4.0,
            rhythm_regularity: 0.5,
        };

        let duration_quality = evaluator
            .compute_intrinsic_duration_quality(&durations)
            .unwrap();
        let pitch_quality = evaluator.compute_intrinsic_pitch_quality(&pitch).unwrap();
        let rhythm_quality = evaluator.compute_intrinsic_rhythm_quality(&rhythm).unwrap();

        assert!((0.0..=100.0).contains(&duration_quality));
        assert!((0.0..=100.0).contains(&pitch_quality));
        assert!((0.0..=100.0).contains(&rhythm_quality));
    }

    #[test]
    fn test_pearson_correlation() {
        let evaluator = ProsodyEvaluator::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let correlation = evaluator.compute_pearson_correlation(&x, &y).unwrap();
        assert!((correlation - 1.0).abs() < 0.001); // Perfect positive correlation
    }

    #[test]
    fn test_empty_input_error() {
        let evaluator = ProsodyEvaluator::new();
        let empty_data: Vec<Vec<f32>> = vec![];
        let empty_durations: Vec<f32> = vec![];

        assert!(evaluator.extract_prosody_features(&empty_data).is_err());
        assert!(evaluator
            .compute_duration_accuracy(&empty_durations, &empty_durations)
            .is_err());
    }
}
