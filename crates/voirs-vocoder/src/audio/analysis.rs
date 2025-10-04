//! Audio analysis module.
//!
//! This module provides audio analysis capabilities including:
//! - Peak and RMS level measurement
//! - Dynamic range analysis
//! - Spectral analysis and plotting
//! - Quality metrics computation

use crate::{AudioBuffer, Result};
use scirs2_core::Complex;
use scirs2_fft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Audio analysis metrics
#[derive(Debug, Clone)]
pub struct AudioMetrics {
    /// Peak level in dB
    pub peak_db: f32,
    /// RMS level in dB
    pub rms_db: f32,
    /// Dynamic range in dB
    pub dynamic_range_db: f32,
    /// THD+N (Total Harmonic Distortion + Noise) percentage
    pub thd_n_percent: f32,
    /// Signal-to-noise ratio in dB
    pub snr_db: f32,
    /// Crest factor (peak to RMS ratio)
    pub crest_factor: f32,
    /// Zero-crossing rate
    pub zero_crossing_rate: f32,
    /// Spectral centroid in Hz
    pub spectral_centroid: f32,
}

/// Audio analyzer
pub struct AudioAnalyzer {
    sample_rate: u32,
    fft_size: usize,
}

impl AudioAnalyzer {
    /// Create new audio analyzer
    pub fn new(sample_rate: u32, fft_size: usize) -> Result<Self> {
        Ok(Self {
            sample_rate,
            fft_size,
        })
    }

    /// Analyze audio buffer and return metrics
    pub fn analyze(&self, audio: &AudioBuffer) -> Result<AudioMetrics> {
        let samples = audio.samples();

        let peak_db = self.calculate_peak_db(samples);
        let rms_db = self.calculate_rms_db(samples);
        let dynamic_range_db = self.calculate_dynamic_range_db(samples);
        let thd_n_percent = self.calculate_thd_n(samples);
        let snr_db = self.calculate_snr_db(samples);
        let crest_factor = self.calculate_crest_factor(samples);
        let zero_crossing_rate = self.calculate_zero_crossing_rate(samples);
        let spectral_centroid = self.calculate_spectral_centroid(samples);

        Ok(AudioMetrics {
            peak_db,
            rms_db,
            dynamic_range_db,
            thd_n_percent,
            snr_db,
            crest_factor,
            zero_crossing_rate,
            spectral_centroid,
        })
    }

    /// Calculate peak level in dB
    pub fn calculate_peak_db(&self, samples: &[f32]) -> f32 {
        let peak = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        if peak > 0.0 {
            20.0 * peak.log10()
        } else {
            -f32::INFINITY
        }
    }

    /// Calculate RMS level in dB
    pub fn calculate_rms_db(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return -f32::INFINITY;
        }

        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            -f32::INFINITY
        }
    }

    /// Calculate dynamic range in dB
    pub fn calculate_dynamic_range_db(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let peak = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let noise_floor = self.estimate_noise_floor(samples);

        if peak > 0.0 && noise_floor > 0.0 {
            20.0 * (peak / noise_floor).log10()
        } else {
            0.0
        }
    }

    /// Estimate noise floor
    fn estimate_noise_floor(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // Use bottom 10% of samples as noise floor estimate
        let mut sorted_samples: Vec<f32> = samples.iter().map(|&x| x.abs()).collect();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let percentile_10 = sorted_samples.len() / 10;
        if percentile_10 > 0 {
            sorted_samples[percentile_10]
        } else {
            sorted_samples[0]
        }
    }

    /// Calculate THD+N (Total Harmonic Distortion + Noise)
    pub fn calculate_thd_n(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // Estimate fundamental frequency first
        let fundamental_freq = self.estimate_fundamental_frequency(samples);
        if fundamental_freq <= 0.0 {
            return 0.0;
        }

        // Perform FFT analysis to identify harmonics
        let fft_size = self.fft_size;
        let mut fft_input = vec![0.0; fft_size];

        // Copy samples to FFT input (with padding/truncation)
        let copy_len = fft_size.min(samples.len());
        fft_input[..copy_len].copy_from_slice(&samples[..copy_len]);

        // Apply Hann window
        for (i, sample) in fft_input.iter_mut().enumerate() {
            let window =
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos());
            *sample *= window;
        }

        // Perform FFT
        let fft_output = match scirs2_fft::rfft(&fft_input, None) {
            Ok(o) => o,
            Err(_) => return 0.0,
        };

        // Find fundamental frequency bin
        let bin_freq = self.sample_rate as f32 / fft_size as f32;
        let fundamental_bin = (fundamental_freq / bin_freq).round() as usize;
        if fundamental_bin >= fft_output.len() {
            return 0.0;
        }

        // Calculate power spectrum
        let power_spectrum: Vec<f32> = fft_output
            .iter()
            .map(|complex| complex.norm_sqr() as f32)
            .collect();

        // Find fundamental power
        let fundamental_power = power_spectrum[fundamental_bin];
        if fundamental_power <= 0.0 {
            return 0.0;
        }

        // Calculate harmonic powers (2nd, 3rd, 4th, 5th harmonics)
        let mut harmonic_power = 0.0;
        for harmonic in 2..=5 {
            let harmonic_bin = fundamental_bin * harmonic;
            if harmonic_bin < power_spectrum.len() {
                harmonic_power += power_spectrum[harmonic_bin];
            }
        }

        // Calculate total power (excluding DC component)
        let total_power: f32 = power_spectrum.iter().skip(1).sum();

        // Calculate noise power (total power minus fundamental and harmonics)
        let noise_power = total_power - fundamental_power - harmonic_power;

        // THD+N = sqrt((harmonic_power + noise_power) / fundamental_power)
        let thd_n = ((harmonic_power + noise_power) / fundamental_power).sqrt();

        // Clamp to reasonable range (0-1)
        thd_n.clamp(0.0, 1.0)
    }

    /// Estimate fundamental frequency
    fn estimate_fundamental_frequency(&self, samples: &[f32]) -> f32 {
        // Simple autocorrelation-based pitch detection
        let mut max_correlation = 0.0;
        let mut best_period = 0;

        let min_period = self.sample_rate / 800; // 800 Hz max
        let max_period = self.sample_rate / 80; // 80 Hz min

        for period in min_period..max_period.min(samples.len() as u32 / 2) {
            let mut correlation = 0.0;
            let period = period as usize;

            for i in 0..(samples.len() - period) {
                correlation += samples[i] * samples[i + period];
            }

            if correlation > max_correlation {
                max_correlation = correlation;
                best_period = period;
            }
        }

        if best_period > 0 {
            self.sample_rate as f32 / best_period as f32
        } else {
            0.0
        }
    }

    /// Calculate SNR in dB
    pub fn calculate_snr_db(&self, samples: &[f32]) -> f32 {
        let signal_power = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
        let noise_power = self.estimate_noise_power(samples);

        if signal_power > 0.0 && noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            0.0
        }
    }

    /// Estimate noise power
    fn estimate_noise_power(&self, samples: &[f32]) -> f32 {
        let noise_floor = self.estimate_noise_floor(samples);
        noise_floor * noise_floor
    }

    /// Calculate crest factor
    pub fn calculate_crest_factor(&self, samples: &[f32]) -> f32 {
        let peak = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();

        if rms > 0.0 {
            peak / rms
        } else {
            0.0
        }
    }

    /// Calculate zero-crossing rate
    pub fn calculate_zero_crossing_rate(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mut zero_crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                zero_crossings += 1;
            }
        }

        zero_crossings as f32 / samples.len() as f32
    }

    /// Calculate spectral centroid
    pub fn calculate_spectral_centroid(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let fft_size = self.fft_size;
        let mut fft_input = vec![0.0; fft_size];

        // Copy samples to FFT input (with padding/truncation)
        let copy_len = fft_size.min(samples.len());
        fft_input[..copy_len].copy_from_slice(&samples[..copy_len]);

        // Apply window function (Hann window)
        for (i, sample) in fft_input.iter_mut().enumerate() {
            let window =
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos());
            *sample *= window;
        }

        // Perform FFT
        let fft_output = match scirs2_fft::rfft(&fft_input, None) {
            Ok(o) => o,
            Err(_) => return 0.0,
        };

        // Calculate spectral centroid
        let mut weighted_sum = 0.0f32;
        let mut magnitude_sum = 0.0f32;

        for (i, complex) in fft_output.iter().enumerate() {
            let magnitude = (complex.re * complex.re + complex.im * complex.im).sqrt() as f32;
            let frequency = i as f32 * self.sample_rate as f32 / fft_size as f32;

            weighted_sum += frequency * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }
}

/// Spectral analysis result
#[derive(Debug, Clone)]
pub struct SpectralAnalysis {
    /// Frequency bins in Hz
    pub frequencies: Vec<f32>,
    /// Magnitude spectrum in dB
    pub magnitudes_db: Vec<f32>,
    /// Phase spectrum in radians
    pub phases: Vec<f32>,
}

/// Perform spectral analysis on audio buffer
pub fn analyze_spectrum(audio: &AudioBuffer, fft_size: usize) -> Result<SpectralAnalysis> {
    let samples = audio.samples();
    if samples.is_empty() {
        return Ok(SpectralAnalysis {
            frequencies: vec![],
            magnitudes_db: vec![],
            phases: vec![],
        });
    }

    let mut fft_input = vec![0.0; fft_size];

    // Copy samples to FFT input
    let copy_len = fft_size.min(samples.len());
    fft_input[..copy_len].copy_from_slice(&samples[..copy_len]);

    // Apply Hann window
    for (i, sample) in fft_input.iter_mut().enumerate() {
        let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos());
        *sample *= window;
    }

    // Perform FFT
    let fft_output = scirs2_fft::rfft(&fft_input, None)?;

    // Convert to frequency domain representation
    let mut frequencies = Vec::new();
    let mut magnitudes_db = Vec::new();
    let mut phases = Vec::new();

    for (i, complex) in fft_output.iter().enumerate() {
        let frequency = i as f32 * audio.sample_rate() as f32 / fft_size as f32;
        let magnitude = (complex.re * complex.re + complex.im * complex.im).sqrt();
        let magnitude_db = if magnitude > 0.0 {
            (20.0 * magnitude.log10()) as f32
        } else {
            -120.0 // -120 dB floor
        };
        let phase = complex.im.atan2(complex.re) as f32;

        frequencies.push(frequency);
        magnitudes_db.push(magnitude_db);
        phases.push(phase);
    }

    Ok(SpectralAnalysis {
        frequencies,
        magnitudes_db,
        phases,
    })
}

/// Quality assessment functions
pub mod quality {
    use super::*;

    /// Calculate PESQ-like quality score (simplified)
    pub fn calculate_pesq_score(reference: &AudioBuffer, degraded: &AudioBuffer) -> f32 {
        // This is a simplified PESQ calculation
        // In practice, you would use the ITU-T P.862 standard
        let _ref_metrics = calculate_simple_metrics(reference);
        let deg_metrics = calculate_simple_metrics(degraded);

        // Simple quality score based on SNR and spectral similarity
        let snr_score = (deg_metrics.snr_db / 30.0).clamp(0.0, 1.0);
        let spectral_score = calculate_spectral_similarity(reference, degraded);

        // Combine scores (PESQ-like range: 1.0 to 4.5)
        1.0 + 3.5 * (snr_score + spectral_score) / 2.0
    }

    /// Calculate simple audio metrics
    fn calculate_simple_metrics(audio: &AudioBuffer) -> AudioMetrics {
        let analyzer = AudioAnalyzer::new(audio.sample_rate(), 1024).unwrap();
        analyzer.analyze(audio).unwrap()
    }

    /// Calculate spectral similarity between two audio buffers
    fn calculate_spectral_similarity(audio1: &AudioBuffer, audio2: &AudioBuffer) -> f32 {
        let spec1 = analyze_spectrum(audio1, 1024).unwrap();
        let spec2 = analyze_spectrum(audio2, 1024).unwrap();

        if spec1.magnitudes_db.is_empty() || spec2.magnitudes_db.is_empty() {
            return 0.0;
        }

        let min_len = spec1.magnitudes_db.len().min(spec2.magnitudes_db.len());
        let mut correlation = 0.0;

        for i in 0..min_len {
            correlation += spec1.magnitudes_db[i] * spec2.magnitudes_db[i];
        }

        correlation / min_len as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_analyzer() {
        let samples = vec![0.5, -0.3, 0.8, -0.2, 0.1];
        let audio = AudioBuffer::new(samples, 22050, 1);

        let analyzer = AudioAnalyzer::new(22050, 1024).unwrap();
        let metrics = analyzer.analyze(&audio).unwrap();

        assert!(metrics.peak_db > -20.0);
        assert!(metrics.rms_db > -20.0);
        assert!(metrics.crest_factor > 0.0);
        assert!(metrics.zero_crossing_rate >= 0.0);
    }

    #[test]
    fn test_peak_calculation() {
        let samples = vec![0.5, -0.8, 0.3, -0.2];
        let audio = AudioBuffer::new(samples, 22050, 1);

        let analyzer = AudioAnalyzer::new(22050, 1024).unwrap();
        let peak_db = analyzer.calculate_peak_db(audio.samples());

        // Peak should be around 20*log10(0.8) ≈ -1.94 dB
        assert!((peak_db - (-1.94)).abs() < 0.1);
    }

    #[test]
    fn test_rms_calculation() {
        let samples = vec![0.5, -0.5, 0.5, -0.5];
        let audio = AudioBuffer::new(samples, 22050, 1);

        let analyzer = AudioAnalyzer::new(22050, 1024).unwrap();
        let rms_db = analyzer.calculate_rms_db(audio.samples());

        // RMS should be around 20*log10(0.5) ≈ -6.02 dB
        assert!((rms_db - (-6.02)).abs() < 0.1);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let samples = vec![0.1, -0.1, 0.1, -0.1, 0.1];
        let audio = AudioBuffer::new(samples, 22050, 1);

        let analyzer = AudioAnalyzer::new(22050, 1024).unwrap();
        let zcr = analyzer.calculate_zero_crossing_rate(audio.samples());

        // Should have 4 zero crossings in 5 samples
        assert!((zcr - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_spectral_analysis() {
        let samples = vec![0.5, -0.3, 0.8, -0.2, 0.1, 0.0, -0.1, 0.2];
        let audio = AudioBuffer::new(samples, 22050, 1);

        let spectrum = analyze_spectrum(&audio, 8).unwrap();

        assert!(!spectrum.frequencies.is_empty());
        assert!(!spectrum.magnitudes_db.is_empty());
        assert!(!spectrum.phases.is_empty());
        assert_eq!(spectrum.frequencies.len(), spectrum.magnitudes_db.len());
        assert_eq!(spectrum.frequencies.len(), spectrum.phases.len());
    }

    #[test]
    fn test_quality_assessment() {
        let samples = vec![0.5, -0.3, 0.8, -0.2];
        let reference = AudioBuffer::new(samples.clone(), 22050, 1);
        let degraded = AudioBuffer::new(samples, 22050, 1);

        let score = quality::calculate_pesq_score(&reference, &degraded);

        // Same audio should have high quality score
        assert!(score > 3.0);
    }
}
