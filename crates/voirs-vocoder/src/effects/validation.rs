//! Audio validation and quality control for post-processing pipeline.
//!
//! This module implements clipping detection, DC offset removal,
//! phase coherency checking, and quality metrics computation.

use super::{AudioEffect, EffectParameter};
use crate::{AudioBuffer, Result};

/// Generate a periodic Hann window of the given length.
///
/// Tapers analysis frames before the FFT so that a single tone's energy stays
/// concentrated in a few bins instead of leaking across the spectrum.
fn hann_window(length: usize) -> Vec<f32> {
    if length <= 1 {
        return vec![1.0; length];
    }
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (length - 1) as f32).cos()))
        .collect()
}

/// Audio validation and quality control processor
pub struct AudioValidator {
    enabled: bool,

    // Configuration
    clipping_threshold: EffectParameter, // Clipping detection threshold
    dc_removal_enabled: bool,            // Enable DC offset removal
    phase_check_enabled: bool,           // Enable phase coherency checking

    // DC removal filter
    #[allow(dead_code)]
    dc_filter_alpha: f32,
    dc_filter_state: Vec<f32>,

    // Quality metrics
    last_metrics: AudioQualityMetrics,
    sample_rate: u32,
}

/// Audio quality metrics
#[derive(Debug, Clone)]
pub struct AudioQualityMetrics {
    pub peak_level_db: f32,
    pub rms_level_db: f32,
    pub crest_factor_db: f32,
    pub thd_plus_n_percent: f32,
    pub dynamic_range_db: f32,
    pub dc_offset: f32,
    pub clipping_detected: bool,
    pub phase_issues: bool,
    pub signal_to_noise_ratio_db: f32,
}

impl Default for AudioQualityMetrics {
    fn default() -> Self {
        Self {
            peak_level_db: -f32::INFINITY,
            rms_level_db: -f32::INFINITY,
            crest_factor_db: 0.0,
            thd_plus_n_percent: 0.0,
            dynamic_range_db: 0.0,
            dc_offset: 0.0,
            clipping_detected: false,
            phase_issues: false,
            signal_to_noise_ratio_db: 0.0,
        }
    }
}

impl AudioValidator {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            enabled: true,
            clipping_threshold: EffectParameter::new("clipping_threshold", 0.95, 0.8, 1.0),
            dc_removal_enabled: true,
            phase_check_enabled: true,

            dc_filter_alpha: 0.995, // High-pass filter coefficient
            dc_filter_state: Vec::new(),

            last_metrics: AudioQualityMetrics::default(),
            sample_rate,
        }
    }

    pub fn set_clipping_threshold(&mut self, threshold: f32) {
        self.clipping_threshold.set_value(threshold);
    }

    pub fn set_dc_removal_enabled(&mut self, enabled: bool) {
        self.dc_removal_enabled = enabled;
    }

    pub fn set_phase_check_enabled(&mut self, enabled: bool) {
        self.phase_check_enabled = enabled;
    }

    pub fn get_last_metrics(&self) -> &AudioQualityMetrics {
        &self.last_metrics
    }

    /// Remove DC offset using high-pass filter
    fn remove_dc_offset(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        let samples = audio.samples_mut();

        // Simple DC offset removal by subtracting the mean
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        for sample in samples.iter_mut() {
            *sample -= mean;
        }

        Ok(())
    }

    /// Detect and prevent clipping
    fn detect_and_prevent_clipping(&self, audio: &mut AudioBuffer) -> bool {
        let samples = audio.samples_mut();
        let threshold = self.clipping_threshold.value;
        let mut clipping_detected = false;

        for sample in samples.iter_mut() {
            if sample.abs() > threshold {
                clipping_detected = true;
                // Soft limiting to prevent hard clipping
                *sample = if *sample > 0.0 {
                    threshold * sample.signum() * (1.0 - (-(*sample - threshold) * 10.0).exp())
                } else {
                    -threshold * sample.signum() * (1.0 - (-(-*sample - threshold) * 10.0).exp())
                };
            }
        }

        clipping_detected
    }

    /// Check for phase coherency issues in stereo audio
    fn check_phase_coherency(&self, audio: &AudioBuffer) -> bool {
        if audio.channels() != 2 {
            return false; // No phase issues in mono
        }

        let samples = audio.samples();
        let mut correlation_sum = 0.0;
        let mut left_sum = 0.0;
        let mut right_sum = 0.0;
        let mut sample_count = 0;

        for i in (0..samples.len()).step_by(2) {
            if i + 1 < samples.len() {
                let left = samples[i];
                let right = samples[i + 1];

                correlation_sum += left * right;
                left_sum += left * left;
                right_sum += right * right;
                sample_count += 1;
            }
        }

        if sample_count == 0 || left_sum == 0.0 || right_sum == 0.0 {
            return false;
        }

        // Calculate correlation coefficient
        let correlation = correlation_sum / (left_sum.sqrt() * right_sum.sqrt());

        // If correlation is very negative, there might be phase issues
        correlation < -0.7
    }

    /// Calculate comprehensive audio quality metrics
    fn calculate_quality_metrics(&self, audio: &AudioBuffer) -> AudioQualityMetrics {
        let samples = audio.samples();

        if samples.is_empty() {
            return AudioQualityMetrics::default();
        }

        // Peak level
        let peak = samples.iter().map(|x| x.abs()).fold(0.0, f32::max);
        let peak_db = if peak > 0.0 {
            20.0 * peak.log10()
        } else {
            -f32::INFINITY
        };

        // RMS level
        let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        let rms_db = if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            -f32::INFINITY
        };

        // Crest factor
        let crest_factor_db = peak_db - rms_db;

        // DC offset
        let dc_offset = samples.iter().sum::<f32>() / samples.len() as f32;

        // Simple THD+N estimation (simplified)
        let thd_plus_n = self.estimate_thd_plus_n(samples);

        // Dynamic range estimation
        let sorted_samples: Vec<f32> = {
            let mut s = samples.iter().map(|x| x.abs()).collect::<Vec<_>>();
            s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            s
        };

        let percentile_95 = sorted_samples.get(samples.len() / 20).unwrap_or(&0.0);
        let percentile_5 = sorted_samples.get(samples.len() * 19 / 20).unwrap_or(&0.0);

        let dynamic_range_db = if *percentile_5 > 0.0 {
            20.0 * (percentile_95 / percentile_5).log10()
        } else {
            0.0
        };

        // Signal-to-noise ratio estimation
        let signal_energy = rms * rms;
        let noise_energy = self.estimate_noise_energy(samples);
        let snr_db = if noise_energy > 0.0 {
            10.0 * (signal_energy / noise_energy).log10()
        } else {
            f32::INFINITY
        };

        AudioQualityMetrics {
            peak_level_db: peak_db,
            rms_level_db: rms_db,
            crest_factor_db,
            thd_plus_n_percent: thd_plus_n * 100.0,
            dynamic_range_db,
            dc_offset: dc_offset.abs(),
            clipping_detected: peak > self.clipping_threshold.value,
            phase_issues: self.check_phase_coherency(audio),
            signal_to_noise_ratio_db: snr_db,
        }
    }

    /// THD+N estimation via real-FFT spectral analysis.
    ///
    /// Returns the total-harmonic-distortion-plus-noise ratio as a fraction (the
    /// caller scales it to a percentage). The first 4096-sample frame (or the
    /// whole signal if shorter) is Hann-windowed and transformed with a real FFT.
    /// The fundamental is the strongest peak in a plausible F0 band (50 Hz –
    /// Nyquist/4); its energy is summed over ±2 bins to recover spectral leakage.
    /// All remaining spectral energy (harmonics + noise) forms the residual:
    ///
    /// `THD+N = sqrt((total − fundamental) / fundamental)`.
    fn estimate_thd_plus_n(&self, samples: &[f32]) -> f32 {
        if samples.len() < 1024 {
            return 0.0; // Not enough samples for meaningful analysis
        }

        // Use a power-of-two analysis length capped at 4096 for an efficient FFT.
        let fft_size = 4096.min(samples.len().next_power_of_two() / 2).max(1024);
        let fft_size = fft_size.min(samples.len());

        let window = hann_window(fft_size);
        let input: Vec<f64> = samples
            .iter()
            .take(fft_size)
            .zip(window.iter())
            .map(|(&x, &w)| (x * w) as f64)
            .collect();

        let spectrum = match scirs2_fft::rfft(&input, Some(fft_size)) {
            Ok(s) => s,
            Err(_) => return 0.0,
        };

        let num_bins = fft_size / 2 + 1;
        let mag2: Vec<f64> = spectrum
            .iter()
            .take(num_bins)
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();

        let total_energy: f64 = mag2.iter().skip(1).sum();
        if total_energy <= 0.0 {
            return 0.0;
        }

        let bin_hz = self.sample_rate as f32 / fft_size as f32;
        let low_bin = ((50.0 / bin_hz).floor() as usize).max(1);
        let high_bin = (num_bins / 4).max(low_bin + 1).min(num_bins - 1);

        let mut fundamental_bin = low_bin;
        let mut peak_mag2 = 0.0_f64;
        for (bin, &m2) in mag2.iter().enumerate().take(high_bin + 1).skip(low_bin) {
            if m2 > peak_mag2 {
                peak_mag2 = m2;
                fundamental_bin = bin;
            }
        }
        if peak_mag2 <= 0.0 {
            return 0.0;
        }

        let leak = 2usize;
        let f_lo = fundamental_bin.saturating_sub(leak);
        let f_hi = (fundamental_bin + leak).min(num_bins - 1);
        let fundamental_energy: f64 = mag2[f_lo..=f_hi].iter().sum();
        if fundamental_energy <= 0.0 {
            return 0.0;
        }

        let residual = (total_energy - fundamental_energy).max(0.0);
        ((residual / fundamental_energy).sqrt() as f32).min(1.0)
    }

    /// Estimate noise floor energy
    fn estimate_noise_energy(&self, samples: &[f32]) -> f32 {
        // Find quietest 10% of samples to estimate noise floor
        let mut sorted_samples: Vec<f32> = samples.iter().map(|x| x.abs()).collect();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let noise_samples_count = samples.len() / 10;
        if noise_samples_count > 0 {
            let noise_samples = &sorted_samples[0..noise_samples_count];
            noise_samples.iter().map(|x| x * x).sum::<f32>() / noise_samples.len() as f32
        } else {
            0.0
        }
    }

    /// Generate quality report
    pub fn generate_quality_report(&self) -> String {
        let metrics = &self.last_metrics;

        format!(
            "Audio Quality Report:\n\
            =====================\n\
            Peak Level: {:.2} dB\n\
            RMS Level: {:.2} dB\n\
            Crest Factor: {:.2} dB\n\
            THD+N: {:.3}%\n\
            Dynamic Range: {:.2} dB\n\
            DC Offset: {:.6}\n\
            SNR: {:.2} dB\n\
            Clipping Detected: {}\n\
            Phase Issues: {}\n",
            metrics.peak_level_db,
            metrics.rms_level_db,
            metrics.crest_factor_db,
            metrics.thd_plus_n_percent,
            metrics.dynamic_range_db,
            metrics.dc_offset,
            metrics.signal_to_noise_ratio_db,
            if metrics.clipping_detected {
                "YES"
            } else {
                "NO"
            },
            if metrics.phase_issues { "YES" } else { "NO" }
        )
    }

    /// Check if audio quality meets standards
    pub fn quality_check_passed(&self) -> bool {
        let metrics = &self.last_metrics;

        // Basic quality thresholds
        !metrics.clipping_detected
            && !metrics.phase_issues
            && metrics.dc_offset < 0.01
            && metrics.thd_plus_n_percent < 1.0
            && metrics.signal_to_noise_ratio_db > 60.0
    }
}

impl AudioEffect for AudioValidator {
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // Calculate metrics before processing
        let mut metrics = self.calculate_quality_metrics(audio);

        // Remove DC offset if enabled
        if self.dc_removal_enabled && metrics.dc_offset > 0.001 {
            self.remove_dc_offset(audio)?;
        }

        // Detect and prevent clipping
        let clipping_detected = self.detect_and_prevent_clipping(audio);
        metrics.clipping_detected = clipping_detected;

        // Update stored metrics
        self.last_metrics = metrics;

        Ok(())
    }

    fn name(&self) -> &'static str {
        "AudioValidator"
    }

    fn reset(&mut self) {
        self.dc_filter_state.clear();
        self.last_metrics = AudioQualityMetrics::default();
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Standalone quality analyzer for post-processing validation
pub struct QualityAnalyzer {
    sample_rate: u32,
}

impl QualityAnalyzer {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Analyze audio buffer and return detailed metrics
    pub fn analyze(&self, audio: &AudioBuffer) -> AudioQualityMetrics {
        let validator = AudioValidator::new(self.sample_rate);
        validator.calculate_quality_metrics(audio)
    }

    /// Quick quality check
    pub fn quick_check(&self, audio: &AudioBuffer) -> Result<bool> {
        let metrics = self.analyze(audio);

        // Check for major issues
        let has_issues = metrics.clipping_detected
            || metrics.phase_issues
            || metrics.dc_offset > 0.05
            || metrics.thd_plus_n_percent > 5.0;

        Ok(!has_issues)
    }

    /// Generate comprehensive quality report
    pub fn generate_report(&self, audio: &AudioBuffer) -> String {
        let metrics = self.analyze(audio);

        let mut report = String::new();
        report.push_str("=== Audio Quality Analysis ===\n\n");

        // Basic measurements
        report.push_str("Level Measurements:\n");
        report.push_str(&format!("  Peak Level: {:.2} dB\n", metrics.peak_level_db));
        report.push_str(&format!("  RMS Level: {:.2} dB\n", metrics.rms_level_db));
        report.push_str(&format!(
            "  Crest Factor: {:.2} dB\n",
            metrics.crest_factor_db
        ));
        report.push_str(&format!(
            "  Dynamic Range: {:.2} dB\n\n",
            metrics.dynamic_range_db
        ));

        // Quality metrics
        report.push_str("Quality Metrics:\n");
        report.push_str(&format!("  THD+N: {:.3}%\n", metrics.thd_plus_n_percent));
        report.push_str(&format!(
            "  SNR: {:.2} dB\n",
            metrics.signal_to_noise_ratio_db
        ));
        report.push_str(&format!("  DC Offset: {:.6}\n\n", metrics.dc_offset));

        // Issues
        report.push_str("Detected Issues:\n");
        report.push_str(&format!(
            "  Clipping: {}\n",
            if metrics.clipping_detected {
                "YES"
            } else {
                "NO"
            }
        ));
        report.push_str(&format!(
            "  Phase Problems: {}\n",
            if metrics.phase_issues { "YES" } else { "NO" }
        ));

        // Overall assessment
        report.push_str("\nOverall Assessment:\n");
        let quality_score = self.calculate_quality_score(&metrics);
        report.push_str(&format!("  Quality Score: {quality_score:.1}/10.0\n"));

        let quality_grade = match quality_score {
            s if s >= 9.0 => "Excellent",
            s if s >= 8.0 => "Very Good",
            s if s >= 7.0 => "Good",
            s if s >= 6.0 => "Fair",
            s if s >= 4.0 => "Poor",
            _ => "Very Poor",
        };
        report.push_str(&format!("  Quality Grade: {quality_grade}\n"));

        report
    }

    fn calculate_quality_score(&self, metrics: &AudioQualityMetrics) -> f32 {
        let mut score = 10.0;

        // Deduct for clipping
        if metrics.clipping_detected {
            score -= 3.0;
        }

        // Deduct for phase issues
        if metrics.phase_issues {
            score -= 2.0;
        }

        // Deduct for high THD+N
        if metrics.thd_plus_n_percent > 1.0 {
            score -= (metrics.thd_plus_n_percent - 1.0).min(2.0);
        }

        // Deduct for DC offset
        if metrics.dc_offset > 0.01 {
            score -= (metrics.dc_offset * 100.0).min(1.0);
        }

        // Deduct for poor SNR
        if metrics.signal_to_noise_ratio_db < 60.0 {
            score -= ((60.0 - metrics.signal_to_noise_ratio_db) / 10.0).min(2.0);
        }

        score.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_validator_creation() {
        let validator = AudioValidator::new(44100);
        assert!(validator.is_enabled());
        assert_eq!(validator.name(), "AudioValidator");
    }

    #[test]
    fn test_quality_analyzer() {
        let analyzer = QualityAnalyzer::new(44100);

        // Create test audio with known characteristics
        let samples = vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3];
        let audio = AudioBuffer::new(samples, 44100, 1);

        let metrics = analyzer.analyze(&audio);
        assert!(metrics.peak_level_db > -f32::INFINITY);
        assert!(metrics.rms_level_db > -f32::INFINITY);

        let report = analyzer.generate_report(&audio);
        assert!(report.contains("Audio Quality Analysis"));
    }

    #[test]
    fn test_dc_offset_removal() {
        let mut validator = AudioValidator::new(44100);

        // Create audio with DC offset
        let samples = vec![0.5, 0.6, 0.4, 0.7, 0.3]; // All positive = DC offset
        let mut audio = AudioBuffer::new(samples, 44100, 1);

        validator.process(&mut audio).unwrap();

        // Check that DC was reduced
        let dc_offset = audio.samples().iter().sum::<f32>() / audio.samples().len() as f32;
        assert!(dc_offset.abs() < 0.1); // Should be much closer to zero
    }

    #[test]
    fn test_estimate_thd_plus_n_pure_sine_near_zero() {
        let validator = AudioValidator::new(44100);
        let sr = 44100.0_f32;
        let freq = 1000.0_f32;
        let signal: Vec<f32> = (0..8192)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect();

        let thd = validator.estimate_thd_plus_n(&signal);
        // Fraction (not percent): a clean tone should be tiny.
        assert!(
            thd < 0.05,
            "pure sine THD+N fraction should be near zero, got {thd}"
        );
    }

    #[test]
    fn test_estimate_thd_plus_n_with_harmonics_is_larger() {
        let validator = AudioValidator::new(44100);
        let sr = 44100.0_f32;
        let f0 = 1000.0_f32;

        let dirty: Vec<f32> = (0..8192)
            .map(|i| {
                let t = i as f32 / sr;
                (2.0 * std::f32::consts::PI * f0 * t).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * 2.0 * f0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 3.0 * f0 * t).sin()
            })
            .collect();
        let clean: Vec<f32> = (0..8192)
            .map(|i| (2.0 * std::f32::consts::PI * f0 * i as f32 / sr).sin())
            .collect();

        let thd_dirty = validator.estimate_thd_plus_n(&dirty);
        let thd_clean = validator.estimate_thd_plus_n(&clean);

        assert!(
            thd_dirty > 0.1,
            "harmonic signal should have clearly positive THD+N, got {thd_dirty}"
        );
        assert!(
            thd_dirty > thd_clean,
            "harmonic ({thd_dirty}) must exceed clean ({thd_clean})"
        );
    }
}
