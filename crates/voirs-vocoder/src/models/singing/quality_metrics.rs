//! Quality metrics for singing voice vocoder.

use crate::models::singing::config::QualityMetricsConfig;
use anyhow::Result;
use scirs2_core::Complex;
use scirs2_fft::{FftPlanner, RealFftPlanner};
use std::collections::VecDeque;

/// Quality metrics calculator for singing voice
pub struct SingingQualityMetrics {
    /// Configuration
    config: QualityMetricsConfig,
    /// Window size for analysis
    window_size: usize,
    /// Sample rate
    sample_rate: u32,
    /// Pitch history for accuracy calculation
    pitch_history: VecDeque<f32>,
    /// Spectral history for stability calculation
    spectral_history: VecDeque<Vec<f32>>,
}

/// Quality metrics result
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Pitch accuracy score (0.0-1.0)
    pub pitch_accuracy: f32,
    /// Harmonic clarity score (0.0-1.0)
    pub harmonic_clarity: f32,
    /// Spectral stability score (0.0-1.0)
    pub spectral_stability: f32,
    /// Overall singing quality score (0.0-1.0)
    pub singing_quality: f32,
    /// Additional metrics
    pub additional_metrics: AdditionalMetrics,
}

/// Additional quality metrics
#[derive(Debug, Clone)]
pub struct AdditionalMetrics {
    /// Signal-to-noise ratio
    pub snr: f32,
    /// Total harmonic distortion
    pub thd: f32,
    /// Spectral centroid
    pub spectral_centroid: f32,
    /// Spectral bandwidth
    pub spectral_bandwidth: f32,
    /// Spectral rolloff
    pub spectral_rolloff: f32,
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
}

impl SingingQualityMetrics {
    /// Create new quality metrics calculator
    pub fn new() -> Self {
        Self {
            config: QualityMetricsConfig::default(),
            window_size: 2048,
            sample_rate: 22050,
            pitch_history: VecDeque::with_capacity(100),
            spectral_history: VecDeque::with_capacity(20),
        }
    }

    /// Create new quality metrics calculator with configuration
    pub fn with_config(config: QualityMetricsConfig) -> Self {
        Self {
            config,
            window_size: 2048,
            sample_rate: 22050,
            pitch_history: VecDeque::with_capacity(100),
            spectral_history: VecDeque::with_capacity(20),
        }
    }

    /// Calculate quality metrics from audio
    pub fn calculate(&mut self, audio: &[f32]) -> Result<f32> {
        if !self.config.enable_metrics {
            return Ok(0.8); // Default quality score
        }

        // Convert audio to frames and analyze
        let frame_size = self.window_size;
        let hop_size = frame_size / 4;
        let mut all_metrics = Vec::new();

        for i in (0..audio.len()).step_by(hop_size) {
            let end = (i + frame_size).min(audio.len());
            if end - i < frame_size / 2 {
                break; // Skip incomplete frames
            }

            let frame = &audio[i..end];
            let frame_metrics = self.calculate_frame_metrics(frame)?;
            all_metrics.push(frame_metrics);
        }

        // Calculate overall quality
        let overall_quality = if all_metrics.is_empty() {
            0.8
        } else {
            all_metrics.iter().map(|m| m.singing_quality).sum::<f32>() / all_metrics.len() as f32
        };

        Ok(overall_quality)
    }

    /// Calculate metrics for a single frame
    fn calculate_frame_metrics(&mut self, frame: &[f32]) -> Result<QualityMetrics> {
        // Perform FFT analysis
        let spectrum = self.compute_spectrum(frame)?;

        // Calculate individual metrics
        let pitch_accuracy = if self.config.calculate_pitch_accuracy {
            self.calculate_pitch_accuracy(&spectrum)?
        } else {
            0.8
        };

        let harmonic_clarity = if self.config.calculate_harmonic_clarity {
            self.calculate_harmonic_clarity(&spectrum)?
        } else {
            0.8
        };

        let spectral_stability = if self.config.calculate_spectral_stability {
            self.calculate_spectral_stability(&spectrum)?
        } else {
            0.8
        };

        let singing_quality = if self.config.calculate_singing_quality {
            self.calculate_singing_quality(pitch_accuracy, harmonic_clarity, spectral_stability)?
        } else {
            (pitch_accuracy + harmonic_clarity + spectral_stability) / 3.0
        };

        // Calculate additional metrics
        let additional_metrics = self.calculate_additional_metrics(frame, &spectrum)?;

        Ok(QualityMetrics {
            pitch_accuracy,
            harmonic_clarity,
            spectral_stability,
            singing_quality,
            additional_metrics,
        })
    }

    /// Compute spectrum from audio frame
    fn compute_spectrum(&mut self, frame: &[f32]) -> Result<Vec<f32>> {
        let mut padded_frame = frame.to_vec();
        padded_frame.resize(self.window_size, 0.0);

        // Apply windowing
        for (i, sample) in padded_frame.iter_mut().enumerate() {
            let window = 0.5
                * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (self.window_size - 1) as f32)
                        .cos());
            *sample *= window;
        }

        // Perform FFT
        let fft_input: Vec<Complex<f32>> =
            padded_frame.iter().map(|&x| Complex::new(x, 0.0)).collect();

        let fft_output_f64 = scirs2_fft::fft(&fft_input, None)?;

        // Extract magnitude spectrum
        let spectrum: Vec<f32> = fft_output_f64
            .iter()
            .take(self.window_size / 2 + 1)
            .map(|c| (c.norm()) as f32)
            .collect();

        Ok(spectrum)
    }

    /// Calculate pitch accuracy
    fn calculate_pitch_accuracy(&mut self, spectrum: &[f32]) -> Result<f32> {
        // Detect fundamental frequency
        let fundamental = self.detect_fundamental_frequency(spectrum)?;

        // Add to pitch history
        self.pitch_history.push_back(fundamental);
        if self.pitch_history.len() > 100 {
            self.pitch_history.pop_front();
        }

        // Calculate pitch stability
        if self.pitch_history.len() < 3 {
            return Ok(0.8);
        }

        let pitches: Vec<f32> = self.pitch_history.iter().cloned().collect();
        let mean_pitch = pitches.iter().sum::<f32>() / pitches.len() as f32;

        if mean_pitch == 0.0 {
            return Ok(0.5);
        }

        // Calculate coefficient of variation
        let variance = pitches
            .iter()
            .map(|p| (p - mean_pitch).powi(2))
            .sum::<f32>()
            / pitches.len() as f32;

        let std_dev = variance.sqrt();
        let cv = std_dev / mean_pitch;

        // Convert to accuracy score (lower variation = higher accuracy)
        let accuracy = (1.0 - cv).clamp(0.0, 1.0);

        Ok(accuracy)
    }

    /// Detect fundamental frequency
    fn detect_fundamental_frequency(&self, spectrum: &[f32]) -> Result<f32> {
        let mut max_magnitude = 0.0;
        let mut peak_bin = 0;

        // Look for peaks in the fundamental frequency range (80-800 Hz)
        let min_bin = (80.0 * spectrum.len() as f32 * 2.0 / self.sample_rate as f32) as usize;
        let max_bin = (800.0 * spectrum.len() as f32 * 2.0 / self.sample_rate as f32) as usize;

        for (bin, &magnitude) in spectrum
            .iter()
            .enumerate()
            .take(max_bin.min(spectrum.len()))
            .skip(min_bin)
        {
            if magnitude > max_magnitude {
                max_magnitude = magnitude;
                peak_bin = bin;
            }
        }

        // Convert bin to frequency
        let fundamental =
            (peak_bin as f32 * self.sample_rate as f32) / (2.0 * spectrum.len() as f32);
        Ok(fundamental)
    }

    /// Calculate harmonic clarity
    fn calculate_harmonic_clarity(&self, spectrum: &[f32]) -> Result<f32> {
        let fundamental = self.detect_fundamental_frequency(spectrum)?;

        if fundamental == 0.0 {
            return Ok(0.5);
        }

        let mut harmonic_energy = 0.0;
        let mut total_energy = 0.0;

        // Calculate energy in harmonic and total spectrum
        for (bin, &magnitude) in spectrum.iter().enumerate() {
            let frequency = (bin as f32 * self.sample_rate as f32) / (2.0 * spectrum.len() as f32);
            let energy = magnitude * magnitude;

            total_energy += energy;

            // Check if frequency is a harmonic of fundamental
            if self.is_harmonic(frequency, fundamental) {
                harmonic_energy += energy;
            }
        }

        // Harmonic clarity is ratio of harmonic to total energy
        let clarity = if total_energy > 0.0 {
            harmonic_energy / total_energy
        } else {
            0.5
        };

        Ok(clarity.clamp(0.0, 1.0))
    }

    /// Check if frequency is a harmonic of fundamental
    fn is_harmonic(&self, frequency: f32, fundamental: f32) -> bool {
        if fundamental == 0.0 {
            return false;
        }

        let ratio = frequency / fundamental;
        let rounded_ratio = ratio.round();

        // Check if it's close to an integer multiple
        (ratio - rounded_ratio).abs() < 0.1 && (1.0..=10.0).contains(&rounded_ratio)
    }

    /// Calculate spectral stability
    fn calculate_spectral_stability(&mut self, spectrum: &[f32]) -> Result<f32> {
        // Add to spectral history
        self.spectral_history.push_back(spectrum.to_vec());
        if self.spectral_history.len() > 20 {
            self.spectral_history.pop_front();
        }

        if self.spectral_history.len() < 3 {
            return Ok(0.8);
        }

        // Calculate spectral stability as inverse of spectral flux
        let mut spectral_flux = 0.0;
        let frames = self.spectral_history.len();

        for i in 1..frames {
            let prev_frame = &self.spectral_history[i - 1];
            let curr_frame = &self.spectral_history[i];

            let mut frame_flux = 0.0;
            let min_len = prev_frame.len().min(curr_frame.len());

            for j in 0..min_len {
                let diff = curr_frame[j] - prev_frame[j];
                frame_flux += diff * diff;
            }

            spectral_flux += frame_flux.sqrt();
        }

        spectral_flux /= (frames - 1) as f32;

        // Convert to stability score (lower flux = higher stability)
        let stability = (1.0 / (1.0 + spectral_flux)).clamp(0.0, 1.0);

        Ok(stability)
    }

    /// Calculate overall singing quality
    fn calculate_singing_quality(
        &self,
        pitch_accuracy: f32,
        harmonic_clarity: f32,
        spectral_stability: f32,
    ) -> Result<f32> {
        // Weighted combination of metrics
        let quality = pitch_accuracy * 0.4 + harmonic_clarity * 0.3 + spectral_stability * 0.3;
        Ok(quality.clamp(0.0, 1.0))
    }

    /// Calculate additional metrics
    fn calculate_additional_metrics(
        &self,
        frame: &[f32],
        spectrum: &[f32],
    ) -> Result<AdditionalMetrics> {
        let snr = self.calculate_snr(spectrum)?;
        let thd = self.calculate_thd(spectrum)?;
        let spectral_centroid = self.calculate_spectral_centroid(spectrum)?;
        let spectral_bandwidth = self.calculate_spectral_bandwidth(spectrum, spectral_centroid)?;
        let spectral_rolloff = self.calculate_spectral_rolloff(spectrum)?;
        let zero_crossing_rate = self.calculate_zero_crossing_rate(frame)?;

        Ok(AdditionalMetrics {
            snr,
            thd,
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            zero_crossing_rate,
        })
    }

    /// Calculate signal-to-noise ratio
    fn calculate_snr(&self, spectrum: &[f32]) -> Result<f32> {
        // Find signal peaks (above certain threshold) and noise floor
        let sorted_spectrum: Vec<f32> = spectrum.iter().map(|x| x.abs()).collect::<Vec<_>>();
        let mut sorted = sorted_spectrum.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use bottom 10% as noise floor estimate
        let noise_samples = (spectrum.len() as f32 * 0.1).max(1.0) as usize;
        let noise_floor: f32 =
            sorted.iter().take(noise_samples).sum::<f32>() / noise_samples as f32;

        // Use top 10% as signal estimate
        let signal_samples = (spectrum.len() as f32 * 0.1).max(1.0) as usize;
        let signal_level: f32 =
            sorted.iter().rev().take(signal_samples).sum::<f32>() / signal_samples as f32;

        let snr = if noise_floor > 0.0 && signal_level > noise_floor {
            20.0 * (signal_level / noise_floor).log10()
        } else {
            60.0 // Very high SNR
        };

        Ok(snr.clamp(0.0, 60.0))
    }

    /// Calculate total harmonic distortion
    fn calculate_thd(&self, spectrum: &[f32]) -> Result<f32> {
        let fundamental = self.detect_fundamental_frequency(spectrum)?;

        if fundamental == 0.0 {
            return Ok(0.0);
        }

        let mut fundamental_energy = 0.0;
        let mut harmonic_energy = 0.0;

        for (bin, &magnitude) in spectrum.iter().enumerate() {
            let frequency = (bin as f32 * self.sample_rate as f32) / (2.0 * spectrum.len() as f32);
            let energy = magnitude * magnitude;

            if (frequency - fundamental).abs() < 10.0 {
                fundamental_energy += energy;
            } else if self.is_harmonic(frequency, fundamental) {
                harmonic_energy += energy;
            }
        }

        let thd = if fundamental_energy > 0.0 {
            (harmonic_energy / fundamental_energy).sqrt()
        } else {
            0.0
        };

        Ok(thd.clamp(0.0, 1.0))
    }

    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, spectrum: &[f32]) -> Result<f32> {
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (bin, &magnitude) in spectrum.iter().enumerate() {
            let frequency = (bin as f32 * self.sample_rate as f32) / (2.0 * spectrum.len() as f32);
            weighted_sum += frequency * magnitude;
            magnitude_sum += magnitude;
        }

        let centroid = if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        };

        Ok(centroid)
    }

    /// Calculate spectral bandwidth
    fn calculate_spectral_bandwidth(&self, spectrum: &[f32], centroid: f32) -> Result<f32> {
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (bin, &magnitude) in spectrum.iter().enumerate() {
            let frequency = (bin as f32 * self.sample_rate as f32) / (2.0 * spectrum.len() as f32);
            let deviation = frequency - centroid;
            weighted_sum += deviation * deviation * magnitude;
            magnitude_sum += magnitude;
        }

        let bandwidth = if magnitude_sum > 0.0 {
            (weighted_sum / magnitude_sum).sqrt()
        } else {
            0.0
        };

        Ok(bandwidth)
    }

    /// Calculate spectral rolloff
    fn calculate_spectral_rolloff(&self, spectrum: &[f32]) -> Result<f32> {
        let total_energy: f32 = spectrum.iter().map(|x| x * x).sum();
        let threshold = total_energy * 0.85;

        let mut cumulative_energy = 0.0;
        let mut rolloff_bin = 0;

        for (bin, &magnitude) in spectrum.iter().enumerate() {
            cumulative_energy += magnitude * magnitude;
            if cumulative_energy >= threshold {
                rolloff_bin = bin;
                break;
            }
        }

        let rolloff_freq =
            (rolloff_bin as f32 * self.sample_rate as f32) / (2.0 * spectrum.len() as f32);
        Ok(rolloff_freq)
    }

    /// Calculate zero crossing rate
    fn calculate_zero_crossing_rate(&self, frame: &[f32]) -> Result<f32> {
        let mut zero_crossings = 0;

        for i in 1..frame.len() {
            if (frame[i] >= 0.0 && frame[i - 1] < 0.0) || (frame[i] < 0.0 && frame[i - 1] >= 0.0) {
                zero_crossings += 1;
            }
        }

        let zcr = zero_crossings as f32 / frame.len() as f32;
        Ok(zcr)
    }

    /// Reset quality metrics state
    pub fn reset(&mut self) {
        self.pitch_history.clear();
        self.spectral_history.clear();
    }

    /// Update configuration
    pub fn update_config(&mut self, config: QualityMetricsConfig) {
        self.config = config;
    }
}

impl Default for SingingQualityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_metrics_creation() {
        let metrics = SingingQualityMetrics::new();
        assert!(metrics.config.enable_metrics);
    }

    #[test]
    fn test_quality_metrics_with_config() {
        let config = QualityMetricsConfig {
            enable_metrics: false,
            ..Default::default()
        };
        let metrics = SingingQualityMetrics::with_config(config);
        assert!(!metrics.config.enable_metrics);
    }

    #[test]
    fn test_audio_quality_calculation() {
        let mut metrics = SingingQualityMetrics::new();

        // Create sample audio (sine wave)
        let sample_rate = 22050;
        let duration = 1.0; // 1 second
        let frequency = 440.0; // A4
        let samples = (sample_rate as f32 * duration) as usize;

        let audio: Vec<f32> = (0..samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
            })
            .collect();

        let quality = metrics.calculate(&audio);
        assert!(quality.is_ok());
        assert!(quality.unwrap() > 0.0);
    }

    #[test]
    fn test_spectrum_computation() {
        let mut metrics = SingingQualityMetrics::new();

        // Create sample frame
        let frame: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / 1024.0).sin())
            .collect();

        let spectrum = metrics.compute_spectrum(&frame);
        assert!(spectrum.is_ok());
        assert!(!spectrum.unwrap().is_empty());
    }

    #[test]
    fn test_fundamental_frequency_detection() {
        let metrics = SingingQualityMetrics::new();

        // Create spectrum with peak at 440 Hz
        let spectrum: Vec<f32> = (0..1024)
            .map(|i| {
                let freq = i as f32 * 22050.0 / 2048.0;
                if (freq - 440.0).abs() < 10.0 {
                    1.0
                } else {
                    0.1
                }
            })
            .collect();

        let fundamental = metrics.detect_fundamental_frequency(&spectrum);
        assert!(fundamental.is_ok());
        assert!(fundamental.unwrap() > 0.0);
    }

    #[test]
    fn test_pitch_accuracy_calculation() {
        let mut metrics = SingingQualityMetrics::new();

        // Create stable spectrum
        let spectrum = vec![0.1, 0.5, 1.0, 0.3, 0.1];

        let accuracy = metrics.calculate_pitch_accuracy(&spectrum);
        assert!(accuracy.is_ok());
        let accuracy_val = accuracy.unwrap();
        assert!((0.0..=1.0).contains(&accuracy_val));
    }

    #[test]
    fn test_harmonic_clarity_calculation() {
        let metrics = SingingQualityMetrics::new();

        // Create spectrum with harmonics
        let spectrum = vec![0.1, 1.0, 0.5, 0.3, 0.2, 0.1];

        let clarity = metrics.calculate_harmonic_clarity(&spectrum);
        assert!(clarity.is_ok());
        let clarity_val = clarity.unwrap();
        assert!((0.0..=1.0).contains(&clarity_val));
    }

    #[test]
    fn test_spectral_stability_calculation() {
        let mut metrics = SingingQualityMetrics::new();

        // Add some spectral history
        let spectrum1 = vec![0.1, 0.5, 1.0, 0.3, 0.1];
        let spectrum2 = vec![0.1, 0.6, 0.9, 0.4, 0.1];

        metrics.spectral_history.push_back(spectrum1);
        let stability = metrics.calculate_spectral_stability(&spectrum2);
        assert!(stability.is_ok());
        let stability_val = stability.unwrap();
        assert!((0.0..=1.0).contains(&stability_val));
    }

    #[test]
    fn test_singing_quality_calculation() {
        let metrics = SingingQualityMetrics::new();

        let quality = metrics.calculate_singing_quality(0.8, 0.7, 0.9);
        assert!(quality.is_ok());
        let quality_val = quality.unwrap();
        assert!((0.0..=1.0).contains(&quality_val));
    }

    #[test]
    fn test_additional_metrics_calculation() {
        let metrics = SingingQualityMetrics::new();

        let frame = vec![0.1, -0.2, 0.3, -0.1, 0.5];
        let spectrum = vec![0.1, 0.5, 1.0, 0.3, 0.1];

        let additional = metrics.calculate_additional_metrics(&frame, &spectrum);
        assert!(additional.is_ok());

        let additional = additional.unwrap();
        assert!(additional.snr >= 0.0);
        assert!(additional.thd >= 0.0);
        assert!(additional.spectral_centroid >= 0.0);
        assert!(additional.spectral_bandwidth >= 0.0);
        assert!(additional.spectral_rolloff >= 0.0);
        assert!(additional.zero_crossing_rate >= 0.0);
    }

    #[test]
    fn test_snr_calculation() {
        let metrics = SingingQualityMetrics::new();

        let spectrum = vec![0.1, 0.5, 1.0, 0.3, 0.1];
        let snr = metrics.calculate_snr(&spectrum);
        assert!(snr.is_ok());
        assert!(snr.unwrap() >= 0.0);
    }

    #[test]
    fn test_thd_calculation() {
        let metrics = SingingQualityMetrics::new();

        let spectrum = vec![0.1, 0.5, 1.0, 0.3, 0.1];
        let thd = metrics.calculate_thd(&spectrum);
        assert!(thd.is_ok());
        assert!(thd.unwrap() >= 0.0);
    }

    #[test]
    fn test_spectral_centroid_calculation() {
        let metrics = SingingQualityMetrics::new();

        let spectrum = vec![0.1, 0.5, 1.0, 0.3, 0.1];
        let centroid = metrics.calculate_spectral_centroid(&spectrum);
        assert!(centroid.is_ok());
        assert!(centroid.unwrap() >= 0.0);
    }

    #[test]
    fn test_zero_crossing_rate_calculation() {
        let metrics = SingingQualityMetrics::new();

        let frame = vec![0.1, -0.2, 0.3, -0.1, 0.5];
        let zcr = metrics.calculate_zero_crossing_rate(&frame);
        assert!(zcr.is_ok());
        assert!(zcr.unwrap() >= 0.0);
    }

    #[test]
    fn test_is_harmonic() {
        let metrics = SingingQualityMetrics::new();

        assert!(metrics.is_harmonic(880.0, 440.0)); // 2nd harmonic
        assert!(metrics.is_harmonic(1320.0, 440.0)); // 3rd harmonic
        assert!(!metrics.is_harmonic(500.0, 440.0)); // Not a harmonic
    }

    #[test]
    fn test_metrics_reset() {
        let mut metrics = SingingQualityMetrics::new();

        metrics.pitch_history.push_back(440.0);
        metrics.spectral_history.push_back(vec![0.1, 0.2, 0.3]);

        metrics.reset();

        assert!(metrics.pitch_history.is_empty());
        assert!(metrics.spectral_history.is_empty());
    }

    #[test]
    fn test_config_update() {
        let mut metrics = SingingQualityMetrics::new();

        let new_config = QualityMetricsConfig {
            enable_metrics: false,
            ..Default::default()
        };

        metrics.update_config(new_config);
        assert!(!metrics.config.enable_metrics);
    }
}
