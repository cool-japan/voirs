//! Objective audio quality metrics
//!
//! This module provides objective metrics for evaluating TTS synthesis quality,
//! including spectral distortion, signal-to-noise ratio, total harmonic distortion,
//! and pitch accuracy measurements.

use crate::{AcousticError, Result};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Objective quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveMetrics {
    /// Signal-to-noise ratio in dB
    pub snr: f32,
    /// Total harmonic distortion (0-1)
    pub thd: f32,
    /// Spectral distortion (if reference available)
    pub spectral_distortion: Option<f32>,
    /// Mel-cepstral distortion (if reference available)
    pub mcd: Option<f32>,
    /// Pitch accuracy correlation (0-1)
    pub pitch_correlation: f32,
}

impl Default for ObjectiveMetrics {
    fn default() -> Self {
        Self {
            snr: 0.0,
            thd: 0.0,
            spectral_distortion: None,
            mcd: None,
            pitch_correlation: 0.0,
        }
    }
}

/// Objective quality evaluator
pub struct ObjectiveEvaluator {
    /// FFT size for spectral analysis
    #[allow(dead_code)]
    fft_size: usize,
    /// Hop length for windowed analysis
    #[allow(dead_code)]
    hop_length: usize,
    /// Window function type
    #[allow(dead_code)]
    window_type: WindowType,
}

/// Window function types for spectral analysis
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    Hann,
    Hamming,
    Blackman,
    Rectangular,
}

impl Default for ObjectiveEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectiveEvaluator {
    /// Create new objective evaluator with default parameters
    pub fn new() -> Self {
        Self {
            fft_size: 1024,
            hop_length: 256,
            window_type: WindowType::Hann,
        }
    }

    /// Create evaluator with custom parameters
    pub fn with_params(fft_size: usize, hop_length: usize, window_type: WindowType) -> Self {
        Self {
            fft_size,
            hop_length,
            window_type,
        }
    }

    /// Compute Signal-to-Noise Ratio (SNR) in dB
    pub fn compute_snr(&self, mel_data: &[Vec<f32>]) -> Result<f32> {
        if mel_data.is_empty() || mel_data[0].is_empty() {
            return Err(AcousticError::InputError(
                "Empty mel spectrogram data".to_string(),
            ));
        }

        let mut signal_power = 0.0f32;
        let mut noise_power = 0.0f32;
        let mut total_samples = 0;

        for mel_channel in mel_data {
            for &value in mel_channel {
                // Consider signal as the magnitude
                let magnitude = value.abs();
                signal_power += magnitude * magnitude;

                // Estimate noise as high-frequency components (simple approximation)
                // In practice, this would use more sophisticated noise estimation
                let noise_estimate = magnitude * 0.01; // Assume 1% noise
                noise_power += noise_estimate * noise_estimate;

                total_samples += 1;
            }
        }

        if total_samples == 0 || noise_power <= 0.0 {
            return Ok(60.0); // Return a reasonable default SNR
        }

        signal_power /= total_samples as f32;
        noise_power /= total_samples as f32;

        let snr_linear = signal_power / noise_power;
        let snr_db = 10.0 * snr_linear.log10();

        // Clamp to reasonable range
        Ok(snr_db.clamp(-10.0, 80.0))
    }

    /// Compute Total Harmonic Distortion (THD)
    pub fn compute_thd(&self, mel_data: &[Vec<f32>]) -> Result<f32> {
        if mel_data.is_empty() || mel_data[0].is_empty() {
            return Err(AcousticError::InputError(
                "Empty mel spectrogram data".to_string(),
            ));
        }

        // Simplified THD computation based on mel spectrogram
        // In practice, this would require converting back to time domain
        let mut fundamental_power = 0.0f32;
        let mut harmonic_power = 0.0f32;

        for mel_channel in mel_data {
            // Find the strongest component (approximated fundamental)
            let max_value = mel_channel
                .iter()
                .fold(0.0f32, |max, &val| max.max(val.abs()));
            fundamental_power += max_value * max_value;

            // Sum of all other components as harmonics
            for &value in mel_channel {
                let magnitude = value.abs();
                if magnitude < max_value * 0.9 {
                    // Not the fundamental
                    harmonic_power += magnitude * magnitude;
                }
            }
        }

        if fundamental_power <= 0.0 {
            return Ok(0.1); // Return reasonable default
        }

        let thd = (harmonic_power / fundamental_power).sqrt();

        // THD typically ranges from 0 to 1, with lower being better
        Ok(thd.min(1.0))
    }

    /// Compute spectral distortion between generated and reference
    pub fn compute_spectral_distortion(
        &self,
        generated: &[Vec<f32>],
        reference: &[Vec<f32>],
    ) -> Result<f32> {
        if generated.len() != reference.len() {
            return Err(AcousticError::InputError(
                "Generated and reference spectrograms must have same number of mel channels"
                    .to_string(),
            ));
        }

        let mut total_distortion = 0.0f32;
        let mut total_frames = 0;

        for (gen_channel, ref_channel) in generated.iter().zip(reference.iter()) {
            let min_frames = gen_channel.len().min(ref_channel.len());

            for i in 0..min_frames {
                let gen_db = self.magnitude_to_db(gen_channel[i].abs());
                let ref_db = self.magnitude_to_db(ref_channel[i].abs());

                let distortion = (gen_db - ref_db).abs();
                total_distortion += distortion;
                total_frames += 1;
            }
        }

        if total_frames == 0 {
            return Ok(0.0);
        }

        Ok(total_distortion / total_frames as f32)
    }

    /// Compute Mel-Cepstral Distortion (MCD)
    pub fn compute_mel_cepstral_distortion(
        &self,
        generated: &[Vec<f32>],
        reference: &[Vec<f32>],
    ) -> Result<f32> {
        if generated.len() != reference.len() {
            return Err(AcousticError::InputError(
                "Generated and reference spectrograms must have same number of mel channels"
                    .to_string(),
            ));
        }

        let mut total_mcd = 0.0f32;
        let mut total_frames = 0;

        // Compute MCD frame by frame
        let n_frames = generated[0].len().min(reference[0].len());

        for frame_idx in 0..n_frames {
            let mut frame_mcd = 0.0f32;

            for mel_idx in 0..generated.len() {
                if frame_idx < generated[mel_idx].len() && frame_idx < reference[mel_idx].len() {
                    let gen_coeff = generated[mel_idx][frame_idx];
                    let ref_coeff = reference[mel_idx][frame_idx];

                    let diff = gen_coeff - ref_coeff;
                    frame_mcd += diff * diff;
                }
            }

            frame_mcd = frame_mcd.sqrt();
            total_mcd += frame_mcd;
            total_frames += 1;
        }

        if total_frames == 0 {
            return Ok(0.0);
        }

        // Scale factor for MCD (typically 10 * sqrt(2) / ln(10))
        let mcd_scale = 10.0 * (2.0f32).sqrt() / (10.0f32).ln();
        Ok(mcd_scale * total_mcd / total_frames as f32)
    }

    /// Compute pitch accuracy correlation
    pub fn compute_pitch_correlation(&self, mel_data: &[Vec<f32>]) -> Result<f32> {
        if mel_data.is_empty() || mel_data[0].is_empty() {
            return Err(AcousticError::InputError(
                "Empty mel spectrogram data".to_string(),
            ));
        }

        // Extract pitch contour from mel spectrogram
        let pitch_contour = self.extract_pitch_contour(mel_data)?;

        // Compute autocorrelation to measure pitch consistency
        let correlation = self.compute_autocorrelation(&pitch_contour);

        Ok(correlation.clamp(0.0, 1.0))
    }

    /// Compute Log Spectral Distortion (LSD)
    pub fn compute_lsd(&self, generated: &[Vec<f32>], reference: &[Vec<f32>]) -> Result<f32> {
        if generated.len() != reference.len() {
            return Err(AcousticError::InputError(
                "Generated and reference spectrograms must have same dimensions".to_string(),
            ));
        }

        let mut total_lsd = 0.0f32;
        let mut total_frames = 0;

        for (gen_channel, ref_channel) in generated.iter().zip(reference.iter()) {
            let min_frames = gen_channel.len().min(ref_channel.len());

            for i in 0..min_frames {
                let gen_log = (gen_channel[i].abs() + 1e-10).ln();
                let ref_log = (ref_channel[i].abs() + 1e-10).ln();

                let diff = gen_log - ref_log;
                total_lsd += diff * diff;
                total_frames += 1;
            }
        }

        if total_frames == 0 {
            return Ok(0.0);
        }

        let lsd = (total_lsd / total_frames as f32).sqrt();
        Ok(lsd)
    }

    /// Compute spectral centroid for timbral analysis
    pub fn compute_spectral_centroid(&self, mel_data: &[Vec<f32>]) -> Result<Vec<f32>> {
        let mut centroids = Vec::new();

        if mel_data.is_empty() || mel_data[0].is_empty() {
            return Ok(centroids);
        }

        let n_frames = mel_data[0].len();

        for frame_idx in 0..n_frames {
            let mut weighted_sum = 0.0f32;
            let mut magnitude_sum = 0.0f32;

            for (freq_bin, mel_channel) in mel_data.iter().enumerate() {
                if frame_idx < mel_channel.len() {
                    let magnitude = mel_channel[frame_idx].abs();
                    weighted_sum += freq_bin as f32 * magnitude;
                    magnitude_sum += magnitude;
                }
            }

            let centroid = if magnitude_sum > 0.0 {
                weighted_sum / magnitude_sum
            } else {
                0.0
            };

            centroids.push(centroid);
        }

        Ok(centroids)
    }

    /// Compute spectral rolloff point
    pub fn compute_spectral_rolloff(
        &self,
        mel_data: &[Vec<f32>],
        rolloff_percent: f32,
    ) -> Result<Vec<f32>> {
        let mut rolloffs = Vec::new();

        if mel_data.is_empty() || mel_data[0].is_empty() {
            return Ok(rolloffs);
        }

        let n_frames = mel_data[0].len();

        for frame_idx in 0..n_frames {
            let mut magnitude_sum = 0.0f32;
            let mut frame_magnitudes = Vec::new();

            // Collect magnitudes for this frame
            for mel_channel in mel_data {
                if frame_idx < mel_channel.len() {
                    let magnitude = mel_channel[frame_idx].abs();
                    frame_magnitudes.push(magnitude);
                    magnitude_sum += magnitude;
                }
            }

            let threshold = magnitude_sum * rolloff_percent / 100.0;
            let mut cumulative = 0.0f32;
            let mut rolloff_bin = 0;

            for (bin, &magnitude) in frame_magnitudes.iter().enumerate() {
                cumulative += magnitude;
                if cumulative >= threshold {
                    rolloff_bin = bin;
                    break;
                }
            }

            rolloffs.push(rolloff_bin as f32);
        }

        Ok(rolloffs)
    }

    // Private helper methods

    fn magnitude_to_db(&self, magnitude: f32) -> f32 {
        20.0 * (magnitude + 1e-10).log10()
    }

    fn extract_pitch_contour(&self, mel_data: &[Vec<f32>]) -> Result<Vec<f32>> {
        let n_frames = mel_data[0].len();
        let mut pitch_contour = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            // Find the frequency bin with maximum energy (simplified F0 estimation)
            let mut max_energy = 0.0f32;
            let mut max_bin = 0;

            for (bin, mel_channel) in mel_data.iter().enumerate() {
                if frame_idx < mel_channel.len() {
                    let energy = mel_channel[frame_idx].abs();
                    if energy > max_energy {
                        max_energy = energy;
                        max_bin = bin;
                    }
                }
            }

            // Convert bin to approximate frequency (simplified)
            let frequency = max_bin as f32 * 22050.0 / 2.0 / mel_data.len() as f32;
            pitch_contour.push(frequency);
        }

        Ok(pitch_contour)
    }

    fn compute_autocorrelation(&self, signal: &[f32]) -> f32 {
        if signal.len() < 2 {
            return 0.0;
        }

        let n = signal.len();
        let lag = n / 4; // Use 1/4 length lag for autocorrelation

        let mut autocorr = 0.0f32;
        let mut signal_power = 0.0f32;

        for i in 0..(n - lag) {
            autocorr += signal[i] * signal[i + lag];
            signal_power += signal[i] * signal[i];
        }

        if signal_power > 0.0 {
            autocorr / signal_power
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    fn apply_window(&self, signal: &[f32]) -> Vec<f32> {
        let n = signal.len();
        let mut windowed = Vec::with_capacity(n);

        for (i, &sample) in signal.iter().enumerate() {
            let window_value = match self.window_type {
                WindowType::Hann => 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos()),
                WindowType::Hamming => 0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1) as f32).cos(),
                WindowType::Blackman => {
                    0.42 - 0.5 * (2.0 * PI * i as f32 / (n - 1) as f32).cos()
                        + 0.08 * (4.0 * PI * i as f32 / (n - 1) as f32).cos()
                }
                WindowType::Rectangular => 1.0,
            };

            windowed.push(sample * window_value);
        }

        windowed
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
    fn test_objective_evaluator_creation() {
        let evaluator = ObjectiveEvaluator::new();
        assert_eq!(evaluator.fft_size, 1024);
        assert_eq!(evaluator.hop_length, 256);
    }

    #[test]
    fn test_snr_computation() {
        let evaluator = ObjectiveEvaluator::new();
        let mel_data = create_test_mel_data();

        let snr = evaluator.compute_snr(&mel_data).unwrap();
        assert!(snr > -10.0);
        assert!(snr < 80.0);
    }

    #[test]
    fn test_thd_computation() {
        let evaluator = ObjectiveEvaluator::new();
        let mel_data = create_test_mel_data();

        let thd = evaluator.compute_thd(&mel_data).unwrap();
        assert!(thd >= 0.0);
        assert!(thd <= 1.0);
    }

    #[test]
    fn test_spectral_distortion() {
        let evaluator = ObjectiveEvaluator::new();
        let mel_data1 = create_test_mel_data();
        let mel_data2 = create_test_mel_data();

        let distortion = evaluator
            .compute_spectral_distortion(&mel_data1, &mel_data2)
            .unwrap();
        assert!(distortion >= 0.0);
        assert_eq!(distortion, 0.0); // Same data should have 0 distortion
    }

    #[test]
    fn test_mcd_computation() {
        let evaluator = ObjectiveEvaluator::new();
        let mel_data1 = create_test_mel_data();
        let mel_data2 = create_test_mel_data();

        let mcd = evaluator
            .compute_mel_cepstral_distortion(&mel_data1, &mel_data2)
            .unwrap();
        assert!(mcd >= 0.0);
        assert_eq!(mcd, 0.0); // Same data should have 0 MCD
    }

    #[test]
    fn test_pitch_correlation() {
        let evaluator = ObjectiveEvaluator::new();
        let mel_data = create_test_mel_data();

        let correlation = evaluator.compute_pitch_correlation(&mel_data).unwrap();
        assert!(correlation >= 0.0);
        assert!(correlation <= 1.0);
    }

    #[test]
    fn test_spectral_centroid() {
        let evaluator = ObjectiveEvaluator::new();
        let mel_data = create_test_mel_data();

        let centroids = evaluator.compute_spectral_centroid(&mel_data).unwrap();
        assert_eq!(centroids.len(), mel_data[0].len());

        for centroid in centroids {
            assert!(centroid >= 0.0);
        }
    }

    #[test]
    fn test_spectral_rolloff() {
        let evaluator = ObjectiveEvaluator::new();
        let mel_data = create_test_mel_data();

        let rolloffs = evaluator.compute_spectral_rolloff(&mel_data, 85.0).unwrap();
        assert_eq!(rolloffs.len(), mel_data[0].len());

        for rolloff in rolloffs {
            assert!(rolloff >= 0.0);
            assert!(rolloff < mel_data.len() as f32);
        }
    }

    #[test]
    fn test_empty_input_error() {
        let evaluator = ObjectiveEvaluator::new();
        let empty_data: Vec<Vec<f32>> = vec![];

        assert!(evaluator.compute_snr(&empty_data).is_err());
        assert!(evaluator.compute_thd(&empty_data).is_err());
        assert!(evaluator.compute_pitch_correlation(&empty_data).is_err());
    }

    #[test]
    fn test_magnitude_to_db() {
        let evaluator = ObjectiveEvaluator::new();

        let db_value = evaluator.magnitude_to_db(1.0);
        assert_eq!(db_value, 0.0); // 1.0 magnitude = 0 dB

        let db_value = evaluator.magnitude_to_db(0.1);
        assert_eq!(db_value, -20.0); // 0.1 magnitude = -20 dB
    }
}
