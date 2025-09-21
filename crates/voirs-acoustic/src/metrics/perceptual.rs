//! Perceptual audio quality metrics
//!
//! This module provides perceptual metrics for evaluating TTS synthesis quality,
//! including PESQ, STOI, SI-SDR, and other metrics that correlate with human
//! perception of audio quality.

use crate::{AcousticError, Result};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Perceptual quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualMetrics {
    /// PESQ score (1.0-4.5, higher is better)
    pub pesq_score: f32,
    /// STOI score (0.0-1.0, higher is better)
    pub stoi_score: f32,
    /// SI-SDR score in dB (higher is better)
    pub si_sdr: Option<f32>,
    /// Overall perceptual score (0-100)
    pub overall_score: f32,
}

impl Default for PerceptualMetrics {
    fn default() -> Self {
        Self {
            pesq_score: 0.0,
            stoi_score: 0.0,
            si_sdr: None,
            overall_score: 0.0,
        }
    }
}

/// Perceptual quality evaluator
pub struct PerceptualEvaluator {
    /// Sample rate for audio processing
    sample_rate: u32,
    /// Frame size for STOI computation
    stoi_frame_size: usize,
    /// Overlap for STOI frames
    stoi_overlap: usize,
    /// Third-octave band filters for STOI
    stoi_bands: Vec<StoiBand>,
}

/// STOI frequency band information
#[derive(Debug, Clone)]
struct StoiBand {
    /// Center frequency of the band
    #[allow(dead_code)]
    center_freq: f32,
    /// Lower cutoff frequency
    low_freq: f32,
    /// Upper cutoff frequency
    high_freq: f32,
    /// Band weight
    #[allow(dead_code)]
    weight: f32,
}

impl Default for PerceptualEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl PerceptualEvaluator {
    /// Create new perceptual evaluator with default parameters
    pub fn new() -> Self {
        let sample_rate = 16000; // Standard for PESQ/STOI
        let stoi_frame_size = 256;
        let stoi_overlap = 128;
        let stoi_bands = Self::create_stoi_bands(sample_rate);

        Self {
            sample_rate,
            stoi_frame_size,
            stoi_overlap,
            stoi_bands,
        }
    }

    /// Create evaluator with custom sample rate
    pub fn with_sample_rate(sample_rate: u32) -> Self {
        let stoi_frame_size = 256;
        let stoi_overlap = 128;
        let stoi_bands = Self::create_stoi_bands(sample_rate);

        Self {
            sample_rate,
            stoi_frame_size,
            stoi_overlap,
            stoi_bands,
        }
    }

    /// Compute PESQ score (Perceptual Evaluation of Speech Quality)
    pub fn compute_pesq(&self, degraded: &[f32], reference: &[f32]) -> Result<f32> {
        if degraded.is_empty() || reference.is_empty() {
            return Err(AcousticError::InputError("Empty audio samples".to_string()));
        }

        // Simplified PESQ implementation
        // Real PESQ requires complex psychoacoustic modeling
        let pesq_score = self.compute_simplified_pesq(degraded, reference)?;

        // PESQ score range is 1.0 to 4.5
        Ok(pesq_score.clamp(1.0, 4.5))
    }

    /// Compute STOI score (Short-Time Objective Intelligibility)
    pub fn compute_stoi(&self, degraded: &[f32], reference: &[f32]) -> Result<f32> {
        if degraded.is_empty() || reference.is_empty() {
            return Err(AcousticError::InputError("Empty audio samples".to_string()));
        }

        // Align signals by length
        let min_len = degraded.len().min(reference.len());
        let degraded = &degraded[..min_len];
        let reference = &reference[..min_len];

        let stoi_score = self.compute_stoi_core(degraded, reference)?;

        // STOI score range is 0.0 to 1.0
        Ok(stoi_score.clamp(0.0, 1.0))
    }

    /// Compute SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
    pub fn compute_si_sdr(&self, estimated: &[f32], target: &[f32]) -> Result<f32> {
        if estimated.is_empty() || target.is_empty() {
            return Err(AcousticError::InputError("Empty audio samples".to_string()));
        }

        let min_len = estimated.len().min(target.len());
        let estimated = &estimated[..min_len];
        let target = &target[..min_len];

        // Compute the optimal scaling factor
        let alpha =
            self.compute_dot_product(estimated, target) / self.compute_dot_product(target, target);

        // Compute scaled target
        let scaled_target: Vec<f32> = target.iter().map(|&x| alpha * x).collect();

        // Compute signal and noise powers
        let signal_power = self.compute_signal_power(&scaled_target);
        let noise_power = self.compute_noise_power(estimated, &scaled_target);

        if noise_power <= 0.0 {
            return Ok(60.0); // Very high SDR for perfect match
        }

        let si_sdr = 10.0 * (signal_power / noise_power).log10();

        // Reasonable range for SI-SDR
        Ok(si_sdr.clamp(-20.0, 60.0))
    }

    /// Compute intrinsic quality score without reference
    pub fn compute_intrinsic_quality(&self, audio: &[f32]) -> Result<f32> {
        if audio.is_empty() {
            return Err(AcousticError::InputError("Empty audio samples".to_string()));
        }

        // Compute various intrinsic quality indicators
        let snr = self.compute_intrinsic_snr(audio)?;
        let spectral_quality = self.compute_spectral_quality(audio)?;
        let temporal_quality = self.compute_temporal_quality(audio)?;

        // Combine into overall quality score (PESQ-like scale)
        let quality = (snr * 0.4 + spectral_quality * 0.4 + temporal_quality * 0.2).clamp(1.0, 4.5);

        Ok(quality)
    }

    /// Compute perceptual loudness
    pub fn compute_loudness(&self, audio: &[f32]) -> Result<f32> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        // Apply A-weighting filter (simplified)
        let weighted_audio = self.apply_a_weighting(audio)?;

        // Compute RMS with perceptual weighting
        let rms = self.compute_rms(&weighted_audio);

        // Convert to loudness units (simplified)
        let loudness = 20.0 * (rms + 1e-10).log10();

        Ok(loudness)
    }

    /// Compute bark-scale spectral distortion
    pub fn compute_bark_spectral_distortion(
        &self,
        degraded: &[f32],
        reference: &[f32],
    ) -> Result<f32> {
        let min_len = degraded.len().min(reference.len());
        let degraded = &degraded[..min_len];
        let reference = &reference[..min_len];

        // Convert to bark scale representation
        let degraded_bark = self.convert_to_bark_scale(degraded)?;
        let reference_bark = self.convert_to_bark_scale(reference)?;

        // Compute distortion in bark domain
        let mut total_distortion = 0.0f32;
        let bark_bands = degraded_bark.len().min(reference_bark.len());

        for i in 0..bark_bands {
            let diff = degraded_bark[i] - reference_bark[i];
            total_distortion += diff * diff;
        }

        Ok((total_distortion / bark_bands as f32).sqrt())
    }

    // Private helper methods

    fn compute_simplified_pesq(&self, degraded: &[f32], reference: &[f32]) -> Result<f32> {
        // Simplified PESQ-like computation
        let min_len = degraded.len().min(reference.len());
        let degraded = &degraded[..min_len];
        let reference = &reference[..min_len];

        // Compute spectral similarity
        let spectral_sim = self.compute_spectral_similarity(degraded, reference)?;

        // Compute temporal similarity
        let temporal_sim = self.compute_temporal_similarity(degraded, reference)?;

        // Compute loudness similarity
        let loudness_sim = self.compute_loudness_similarity(degraded, reference)?;

        // Combine similarities into PESQ-like score
        let pesq = 1.0 + 3.5 * (spectral_sim * 0.5 + temporal_sim * 0.3 + loudness_sim * 0.2);

        Ok(pesq)
    }

    fn compute_stoi_core(&self, degraded: &[f32], reference: &[f32]) -> Result<f32> {
        let mut correlations = Vec::new();

        // Process in overlapping frames
        let mut frame_start = 0;
        while frame_start + self.stoi_frame_size <= degraded.len() {
            let deg_frame = &degraded[frame_start..frame_start + self.stoi_frame_size];
            let ref_frame = &reference[frame_start..frame_start + self.stoi_frame_size];

            // Apply filterbank to both frames
            let deg_bands = self.apply_stoi_filterbank(deg_frame)?;
            let ref_bands = self.apply_stoi_filterbank(ref_frame)?;

            // Compute correlation for each band
            for (deg_band, ref_band) in deg_bands.iter().zip(ref_bands.iter()) {
                let correlation = self.compute_correlation(deg_band, ref_band);
                correlations.push(correlation);
            }

            frame_start += self.stoi_frame_size - self.stoi_overlap;
        }

        // Average correlations
        if correlations.is_empty() {
            Ok(0.0)
        } else {
            Ok(correlations.iter().sum::<f32>() / correlations.len() as f32)
        }
    }

    fn apply_stoi_filterbank(&self, signal: &[f32]) -> Result<Vec<Vec<f32>>> {
        let mut band_outputs = Vec::new();

        for band in &self.stoi_bands {
            let filtered = self.apply_bandpass_filter(signal, band.low_freq, band.high_freq)?;
            band_outputs.push(filtered);
        }

        Ok(band_outputs)
    }

    fn apply_bandpass_filter(
        &self,
        signal: &[f32],
        low_freq: f32,
        high_freq: f32,
    ) -> Result<Vec<f32>> {
        // Simplified bandpass filter implementation
        let nyquist = self.sample_rate as f32 / 2.0;
        let low_norm = low_freq / nyquist;
        let high_norm = high_freq / nyquist;

        let mut filtered = signal.to_vec();

        // Apply simple IIR bandpass filter (simplified)
        if filtered.len() > 2 {
            let a = 0.9; // Filter coefficient
            let b = (1.0 - a) * (high_norm + low_norm) / 2.0;

            for i in 1..filtered.len() {
                filtered[i] = a * filtered[i - 1] + b * signal[i];
            }
        }

        Ok(filtered)
    }

    fn compute_correlation(&self, signal1: &[f32], signal2: &[f32]) -> f32 {
        if signal1.len() != signal2.len() || signal1.is_empty() {
            return 0.0;
        }

        let mean1 = signal1.iter().sum::<f32>() / signal1.len() as f32;
        let mean2 = signal2.iter().sum::<f32>() / signal2.len() as f32;

        let mut numerator = 0.0f32;
        let mut denom1 = 0.0f32;
        let mut denom2 = 0.0f32;

        for (&x, &y) in signal1.iter().zip(signal2.iter()) {
            let x_centered = x - mean1;
            let y_centered = y - mean2;

            numerator += x_centered * y_centered;
            denom1 += x_centered * x_centered;
            denom2 += y_centered * y_centered;
        }

        let denominator = (denom1 * denom2).sqrt();
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn compute_dot_product(&self, signal1: &[f32], signal2: &[f32]) -> f32 {
        signal1
            .iter()
            .zip(signal2.iter())
            .map(|(&x, &y)| x * y)
            .sum()
    }

    fn compute_signal_power(&self, signal: &[f32]) -> f32 {
        signal.iter().map(|&x| x * x).sum::<f32>() / signal.len() as f32
    }

    fn compute_noise_power(&self, estimated: &[f32], target: &[f32]) -> f32 {
        let noise: Vec<f32> = estimated
            .iter()
            .zip(target.iter())
            .map(|(&e, &t)| e - t)
            .collect();
        self.compute_signal_power(&noise)
    }

    fn compute_intrinsic_snr(&self, audio: &[f32]) -> Result<f32> {
        // Estimate noise level from quiet segments
        let rms = self.compute_rms(audio);
        let peak = audio.iter().fold(0.0f32, |max, &val| max.max(val.abs()));

        if rms > 0.0 {
            let snr = 20.0 * (peak / rms).log10();
            Ok(snr.clamp(0.0, 4.5))
        } else {
            Ok(2.5)
        }
    }

    fn compute_spectral_quality(&self, audio: &[f32]) -> Result<f32> {
        // Analyze spectral characteristics
        let spectral_centroid = self.compute_spectral_centroid_simple(audio);
        let spectral_spread = self.compute_spectral_spread_simple(audio);

        // Quality based on spectral characteristics
        let centroid_quality = (spectral_centroid / (self.sample_rate as f32 / 4.0)).min(1.0);
        let spread_quality = (1.0 - spectral_spread / (self.sample_rate as f32 / 2.0)).max(0.0);

        Ok((centroid_quality + spread_quality) * 2.25 + 1.0) // Scale to 1-4.5
    }

    fn compute_temporal_quality(&self, audio: &[f32]) -> Result<f32> {
        // Analyze temporal characteristics
        let zero_crossing_rate = self.compute_zero_crossing_rate(audio);
        let short_time_energy_var = self.compute_short_time_energy_variance(audio);

        // Quality based on temporal stability
        let zcr_quality = (1.0 - zero_crossing_rate / 0.5).clamp(0.0, 1.0);
        let energy_quality = (1.0 - short_time_energy_var / 10.0).clamp(0.0, 1.0);

        Ok((zcr_quality + energy_quality) * 1.75 + 1.0) // Scale to 1-4.5
    }

    fn compute_rms(&self, signal: &[f32]) -> f32 {
        if signal.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = signal.iter().map(|&x| x * x).sum();
        (sum_squares / signal.len() as f32).sqrt()
    }

    fn apply_a_weighting(&self, audio: &[f32]) -> Result<Vec<f32>> {
        // Simplified A-weighting filter
        // In practice, this would be a proper IIR filter implementation
        let mut weighted = audio.to_vec();

        if weighted.len() > 1 {
            // Simple high-pass characteristic of A-weighting
            for i in 1..weighted.len() {
                weighted[i] = 0.7 * weighted[i] + 0.3 * (audio[i] - audio[i - 1]);
            }
        }

        Ok(weighted)
    }

    fn convert_to_bark_scale(&self, signal: &[f32]) -> Result<Vec<f32>> {
        // Convert signal to bark scale representation
        // This is a simplified implementation
        let bark_bands = 24; // Standard number of bark bands
        let mut bark_spectrum = vec![0.0f32; bark_bands];

        let band_size = signal.len() / bark_bands;

        for (band_idx, bark_value) in bark_spectrum.iter_mut().enumerate() {
            let start_idx = band_idx * band_size;
            let end_idx = ((band_idx + 1) * band_size).min(signal.len());

            if start_idx < end_idx {
                let band_energy: f32 = signal[start_idx..end_idx].iter().map(|&x| x * x).sum();
                *bark_value = (band_energy / (end_idx - start_idx) as f32).sqrt();
            }
        }

        Ok(bark_spectrum)
    }

    fn compute_spectral_similarity(&self, signal1: &[f32], signal2: &[f32]) -> Result<f32> {
        let spec1 = self.compute_magnitude_spectrum(signal1)?;
        let spec2 = self.compute_magnitude_spectrum(signal2)?;

        let min_len = spec1.len().min(spec2.len());
        let similarity = self.compute_correlation(&spec1[..min_len], &spec2[..min_len]);

        Ok(similarity.abs())
    }

    fn compute_temporal_similarity(&self, signal1: &[f32], signal2: &[f32]) -> Result<f32> {
        let correlation = self.compute_correlation(signal1, signal2);
        Ok(correlation.abs())
    }

    fn compute_loudness_similarity(&self, signal1: &[f32], signal2: &[f32]) -> Result<f32> {
        let loud1 = self.compute_loudness(signal1)?;
        let loud2 = self.compute_loudness(signal2)?;

        let diff = (loud1 - loud2).abs();
        let similarity = (-diff / 10.0).exp(); // Exponential decay with difference

        Ok(similarity)
    }

    fn compute_magnitude_spectrum(&self, signal: &[f32]) -> Result<Vec<f32>> {
        // Simplified magnitude spectrum computation
        let n = signal.len();
        let mut spectrum = vec![0.0f32; n / 2];

        for (i, spec_val) in spectrum.iter_mut().enumerate() {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for (j, &sample) in signal.iter().enumerate() {
                let angle = -2.0 * PI * i as f32 * j as f32 / n as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            *spec_val = (real * real + imag * imag).sqrt();
        }

        Ok(spectrum)
    }

    fn compute_spectral_centroid_simple(&self, signal: &[f32]) -> f32 {
        let spectrum = match self.compute_magnitude_spectrum(signal) {
            Ok(spec) => spec,
            Err(_) => return 0.0,
        };

        let mut weighted_sum = 0.0f32;
        let mut magnitude_sum = 0.0f32;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            weighted_sum += i as f32 * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }

    fn compute_spectral_spread_simple(&self, signal: &[f32]) -> f32 {
        let spectrum = match self.compute_magnitude_spectrum(signal) {
            Ok(spec) => spec,
            Err(_) => return 0.0,
        };

        let centroid = self.compute_spectral_centroid_simple(signal);
        let mut weighted_sum = 0.0f32;
        let mut magnitude_sum = 0.0f32;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let diff = i as f32 - centroid;
            weighted_sum += diff * diff * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            (weighted_sum / magnitude_sum).sqrt()
        } else {
            0.0
        }
    }

    fn compute_zero_crossing_rate(&self, signal: &[f32]) -> f32 {
        if signal.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..signal.len() {
            if (signal[i] >= 0.0) != (signal[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (signal.len() - 1) as f32
    }

    fn compute_short_time_energy_variance(&self, signal: &[f32]) -> f32 {
        let frame_size = 256;
        let hop_size = 128;

        let mut energies = Vec::new();
        let mut frame_start = 0;

        while frame_start + frame_size <= signal.len() {
            let frame = &signal[frame_start..frame_start + frame_size];
            let energy = frame.iter().map(|&x| x * x).sum::<f32>() / frame_size as f32;
            energies.push(energy);
            frame_start += hop_size;
        }

        if energies.len() < 2 {
            return 0.0;
        }

        let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;
        let variance = energies
            .iter()
            .map(|&energy| (energy - mean_energy).powi(2))
            .sum::<f32>()
            / energies.len() as f32;

        variance
    }

    fn create_stoi_bands(sample_rate: u32) -> Vec<StoiBand> {
        // Create third-octave bands for STOI
        let mut bands = Vec::new();
        let nyquist = sample_rate as f32 / 2.0;

        // Standard third-octave center frequencies
        let center_freqs = vec![
            125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
            2000.0, 2500.0, 3150.0, 4000.0,
        ];

        for &center_freq in &center_freqs {
            if center_freq < nyquist {
                let bandwidth = center_freq * 0.23; // Approximate third-octave bandwidth
                let low_freq = center_freq - bandwidth / 2.0;
                let high_freq = center_freq + bandwidth / 2.0;

                bands.push(StoiBand {
                    center_freq,
                    low_freq: low_freq.max(0.0),
                    high_freq: high_freq.min(nyquist),
                    weight: 1.0,
                });
            }
        }

        bands
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_audio(length: usize, frequency: f32, sample_rate: f32) -> Vec<f32> {
        (0..length)
            .map(|i| (2.0 * PI * frequency * i as f32 / sample_rate).sin() * 0.5)
            .collect()
    }

    #[test]
    fn test_perceptual_evaluator_creation() {
        let evaluator = PerceptualEvaluator::new();
        assert_eq!(evaluator.sample_rate, 16000);
        assert_eq!(evaluator.stoi_frame_size, 256);
    }

    #[test]
    fn test_pesq_computation() {
        let evaluator = PerceptualEvaluator::new();
        let audio1 = create_test_audio(1000, 440.0, 16000.0);
        let audio2 = create_test_audio(1000, 440.0, 16000.0);

        let pesq = evaluator.compute_pesq(&audio1, &audio2).unwrap();
        assert!(pesq >= 1.0);
        assert!(pesq <= 4.5);
    }

    #[test]
    fn test_stoi_computation() {
        let evaluator = PerceptualEvaluator::new();
        let audio1 = create_test_audio(1000, 440.0, 16000.0);
        let audio2 = create_test_audio(1000, 440.0, 16000.0);

        let stoi = evaluator.compute_stoi(&audio1, &audio2).unwrap();
        assert!(stoi >= 0.0);
        assert!(stoi <= 1.0);
    }

    #[test]
    fn test_si_sdr_computation() {
        let evaluator = PerceptualEvaluator::new();
        let audio1 = create_test_audio(1000, 440.0, 16000.0);
        let audio2 = create_test_audio(1000, 440.0, 16000.0);

        let si_sdr = evaluator.compute_si_sdr(&audio1, &audio2).unwrap();
        assert!(si_sdr >= -20.0);
        assert!(si_sdr <= 60.0);
    }

    #[test]
    fn test_intrinsic_quality() {
        let evaluator = PerceptualEvaluator::new();
        let audio = create_test_audio(1000, 440.0, 16000.0);

        let quality = evaluator.compute_intrinsic_quality(&audio).unwrap();
        assert!(quality >= 1.0);
        assert!(quality <= 4.5);
    }

    #[test]
    fn test_loudness_computation() {
        let evaluator = PerceptualEvaluator::new();
        let audio = create_test_audio(1000, 440.0, 16000.0);

        let loudness = evaluator.compute_loudness(&audio).unwrap();
        assert!(loudness.is_finite());
    }

    #[test]
    fn test_bark_spectral_distortion() {
        let evaluator = PerceptualEvaluator::new();
        let audio1 = create_test_audio(1000, 440.0, 16000.0);
        let audio2 = create_test_audio(1000, 440.0, 16000.0);

        let distortion = evaluator
            .compute_bark_spectral_distortion(&audio1, &audio2)
            .unwrap();
        assert!(distortion >= 0.0);
        assert_eq!(distortion, 0.0); // Same audio should have 0 distortion
    }

    #[test]
    fn test_correlation_computation() {
        let evaluator = PerceptualEvaluator::new();
        let signal1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let signal2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let correlation = evaluator.compute_correlation(&signal1, &signal2);
        assert!((correlation - 1.0).abs() < 0.001); // Perfect correlation
    }

    #[test]
    fn test_empty_input_error() {
        let evaluator = PerceptualEvaluator::new();
        let empty_audio: Vec<f32> = vec![];
        let audio = create_test_audio(100, 440.0, 16000.0);

        assert!(evaluator.compute_pesq(&empty_audio, &audio).is_err());
        assert!(evaluator.compute_stoi(&empty_audio, &audio).is_err());
        assert!(evaluator.compute_si_sdr(&empty_audio, &audio).is_err());
        assert!(evaluator.compute_intrinsic_quality(&empty_audio).is_err());
    }
}
