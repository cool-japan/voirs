//! Quality Assessment Helper Functions
//!
//! This module provides utilities for assessing audio quality
//! including:
//! - Quick quality checks
//! - Audio artifact detection
//! - Signal quality metrics
//! - Comparative quality analysis

use crate::{AudioBuffer, Result, VocoderError};
use std::f32;

/// Audio quality assessment result
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    /// Overall quality score (0.0 - 1.0)
    pub overall_score: f32,
    /// Presence of clipping (0.0 = none, 1.0 = severe)
    pub clipping_factor: f32,
    /// Presence of DC offset (0.0 = none, 1.0 = severe)
    pub dc_offset_factor: f32,
    /// Dynamic range in dB
    pub dynamic_range_db: f32,
    /// Signal-to-noise ratio estimate (dB)
    pub snr_estimate_db: f32,
    /// Spectral flatness (0.0 = tonal, 1.0 = noise)
    pub spectral_flatness: f32,
    /// Zero-crossing rate
    pub zero_crossing_rate: f32,
    /// Quality tier classification
    pub quality_tier: QualityTier,
    /// List of detected issues
    pub issues: Vec<QualityIssue>,
}

/// Quality tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityTier {
    /// Excellent quality (score ≥ 0.9)
    Excellent,
    /// Good quality (score ≥ 0.7)
    Good,
    /// Fair quality (score ≥ 0.5)
    Fair,
    /// Poor quality (score < 0.5)
    Poor,
}

impl std::fmt::Display for QualityTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Excellent => write!(f, "⭐ Excellent"),
            Self::Good => write!(f, "✨ Good"),
            Self::Fair => write!(f, "⚡ Fair"),
            Self::Poor => write!(f, "⚠️  Poor"),
        }
    }
}

/// Quality issues that can be detected
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualityIssue {
    /// Audio clipping detected
    Clipping,
    /// Significant DC offset
    DcOffset,
    /// Low dynamic range
    LowDynamicRange,
    /// High noise floor
    HighNoise,
    /// Unexpected silence
    Silence,
    /// Spectral anomalies
    SpectralAnomalies,
}

impl std::fmt::Display for QualityIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Clipping => write!(f, "⚠️  Clipping detected"),
            Self::DcOffset => write!(f, "📊 DC offset present"),
            Self::LowDynamicRange => write!(f, "📉 Low dynamic range"),
            Self::HighNoise => write!(f, "🔊 High noise floor"),
            Self::Silence => write!(f, "🔇 Unexpected silence"),
            Self::SpectralAnomalies => write!(f, "🌊 Spectral anomalies"),
        }
    }
}

impl QualityAssessment {
    /// Create a comprehensive quality assessment
    pub fn assess(audio: &AudioBuffer) -> Result<Self> {
        let samples = audio.samples();

        if samples.is_empty() {
            return Err(VocoderError::InputError("Empty audio buffer".to_string()));
        }

        // Calculate individual quality factors
        let clipping_factor = calculate_clipping_factor(samples);
        let dc_offset_factor = calculate_dc_offset_factor(samples);
        let dynamic_range_db = calculate_dynamic_range(samples);
        let snr_estimate_db = estimate_snr(samples);
        let spectral_flatness = calculate_spectral_flatness_quick(samples);
        let zero_crossing_rate = calculate_zcr(samples);

        // Detect issues
        let mut issues = Vec::new();

        if clipping_factor > 0.1 {
            issues.push(QualityIssue::Clipping);
        }

        if dc_offset_factor > 0.05 {
            issues.push(QualityIssue::DcOffset);
        }

        if dynamic_range_db < 20.0 {
            issues.push(QualityIssue::LowDynamicRange);
        }

        if snr_estimate_db < 20.0 {
            issues.push(QualityIssue::HighNoise);
        }

        // Check for silence
        let rms = (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        if rms < 0.001 {
            issues.push(QualityIssue::Silence);
        }

        if spectral_flatness > 0.9 {
            issues.push(QualityIssue::SpectralAnomalies);
        }

        // Calculate overall score (weighted combination of factors)
        let overall_score = calculate_overall_score(
            clipping_factor,
            dc_offset_factor,
            dynamic_range_db,
            snr_estimate_db,
            spectral_flatness,
        );

        let quality_tier = if overall_score >= 0.9 {
            QualityTier::Excellent
        } else if overall_score >= 0.7 {
            QualityTier::Good
        } else if overall_score >= 0.5 {
            QualityTier::Fair
        } else {
            QualityTier::Poor
        };

        Ok(Self {
            overall_score,
            clipping_factor,
            dc_offset_factor,
            dynamic_range_db,
            snr_estimate_db,
            spectral_flatness,
            zero_crossing_rate,
            quality_tier,
            issues,
        })
    }

    /// Print a formatted quality report
    pub fn print_report(&self) {
        println!("\n🔍 Audio Quality Assessment");
        println!("═══════════════════════════════════════════════");
        println!(
            "  Overall Score: {:.2}% {}",
            self.overall_score * 100.0,
            self.quality_tier
        );
        println!();
        println!("  Metrics:");
        println!("  ├─ Clipping:       {:.1}%", self.clipping_factor * 100.0);
        println!("  ├─ DC Offset:      {:.1}%", self.dc_offset_factor * 100.0);
        println!("  ├─ Dynamic Range:  {:.1} dB", self.dynamic_range_db);
        println!("  ├─ SNR Estimate:   {:.1} dB", self.snr_estimate_db);
        println!("  ├─ Spec. Flatness: {:.3}", self.spectral_flatness);
        println!("  └─ ZCR:            {:.3}", self.zero_crossing_rate);

        if !self.issues.is_empty() {
            println!();
            println!("  Issues Detected:");
            for issue in &self.issues {
                println!("  • {}", issue);
            }
        } else {
            println!();
            println!("  ✓ No issues detected");
        }

        println!("═══════════════════════════════════════════════");
    }

    /// Check if quality meets minimum threshold
    pub fn meets_threshold(&self, min_score: f32) -> bool {
        self.overall_score >= min_score
    }

    /// Get critical issues (those requiring immediate attention)
    pub fn critical_issues(&self) -> Vec<&QualityIssue> {
        self.issues
            .iter()
            .filter(|issue| matches!(issue, QualityIssue::Clipping | QualityIssue::Silence))
            .collect()
    }
}

/// Calculate clipping factor (0.0 = no clipping, 1.0 = severe clipping)
fn calculate_clipping_factor(samples: &[f32]) -> f32 {
    let threshold = 0.99;
    let clipped_count = samples.iter().filter(|&&s| s.abs() >= threshold).count();
    (clipped_count as f32 / samples.len() as f32).min(1.0)
}

/// Calculate DC offset factor (0.0 = no offset, 1.0 = severe offset)
fn calculate_dc_offset_factor(samples: &[f32]) -> f32 {
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    mean.abs().clamp(0.0, 1.0)
}

/// Calculate dynamic range in dB
fn calculate_dynamic_range(samples: &[f32]) -> f32 {
    let max = samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
    let min = samples
        .iter()
        .map(|&s| s.abs())
        .filter(|&s| s > 1e-8)
        .fold(f32::MAX, f32::min);

    if min > 1e-8 && max > min {
        20.0 * (max / min).log10()
    } else {
        0.0
    }
}

/// Estimate signal-to-noise ratio (simplified)
fn estimate_snr(samples: &[f32]) -> f32 {
    // Estimate noise floor from minimum signal energy windows
    let window_size = 256;
    let mut window_energies = Vec::new();

    for chunk in samples.chunks(window_size) {
        let energy = chunk.iter().map(|&s| s * s).sum::<f32>() / chunk.len() as f32;
        window_energies.push(energy);
    }

    if window_energies.is_empty() {
        return 0.0;
    }

    window_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Take bottom 10% as noise estimate
    let noise_idx = (window_energies.len() as f32 * 0.1) as usize;
    let noise_energy = window_energies[noise_idx].max(1e-10);

    // Signal energy is average
    let signal_energy = window_energies.iter().sum::<f32>() / window_energies.len() as f32;

    10.0 * (signal_energy / noise_energy).log10()
}

/// Calculate the spectral flatness (Wiener entropy) of a signal.
///
/// Spectral flatness is defined as the ratio of the geometric mean to the
/// arithmetic mean of the power/magnitude spectrum:
///
/// ```text
/// flatness = geometric_mean(|X_k|) / arithmetic_mean(|X_k|)
///          = exp(mean(ln(|X_k| + ε))) / (mean(|X_k|) + ε)
/// ```
///
/// The result lies in `[0, 1]`. A value near `1.0` indicates a flat,
/// noise-like spectrum (energy spread evenly across all frequencies), while a
/// value near `0.0` indicates a tonal signal whose energy is concentrated in a
/// few spectral peaks.
///
/// This implementation computes a true magnitude spectrum via
/// [`scirs2_fft::rfft`]. The largest power-of-two window not exceeding the
/// signal length (capped to `[256, 8192]`) is extracted and Hann-windowed
/// before the transform. The DC bin (`k = 0`) is excluded because it only
/// reflects the signal mean / DC offset and would otherwise bias the flatness
/// of a centred tonal signal upward.
fn calculate_spectral_flatness_quick(samples: &[f32]) -> f32 {
    const EPSILON: f64 = 1e-10;

    // Need at least a small window to form a meaningful spectrum.
    if samples.len() < 4 {
        return 0.0;
    }

    // FFT size: largest power of two <= signal length, clamped to [256, 8192].
    let mut fft_size = 256;
    while fft_size * 2 <= samples.len() && fft_size < 8192 {
        fft_size *= 2;
    }
    // If the signal is shorter than the minimum window, shrink to fit.
    if fft_size > samples.len() {
        fft_size = samples.len();
    }

    // Apply a Hann window to reduce spectral leakage, promoting to f64 for the
    // FFT (matching the rest of the crate's spectral analysis).
    let denom = (fft_size.saturating_sub(1)).max(1) as f64;
    let windowed: Vec<f64> = (0..fft_size)
        .map(|i| {
            let hann = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / denom).cos());
            samples[i] as f64 * hann
        })
        .collect();

    // Real FFT -> half-spectrum of complex bins.
    let spectrum = match scirs2_fft::rfft(&windowed, None) {
        Ok(s) => s,
        // On failure, treat the spectrum as undefined (tonal / no flatness).
        Err(_) => return 0.0,
    };

    // Magnitudes |X_k|, skipping the DC bin (k = 0).
    let magnitudes: Vec<f64> = spectrum.iter().skip(1).map(|c| c.norm()).collect();
    if magnitudes.is_empty() {
        return 0.0;
    }

    let n = magnitudes.len() as f64;

    // Arithmetic mean of magnitudes.
    let arithmetic_mean = magnitudes.iter().sum::<f64>() / n;

    // Geometric mean computed in the log domain for numerical stability:
    //   exp( mean( ln(|X_k| + ε) ) )
    let log_mean = magnitudes.iter().map(|&m| (m + EPSILON).ln()).sum::<f64>() / n;
    let geometric_mean = log_mean.exp();

    let flatness = geometric_mean / (arithmetic_mean + EPSILON);

    (flatness as f32).clamp(0.0, 1.0)
}

/// Calculate zero-crossing rate
fn calculate_zcr(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }

    let zero_crossings = samples
        .windows(2)
        .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
        .count();

    zero_crossings as f32 / (samples.len() - 1) as f32
}

/// Calculate overall quality score from individual factors
fn calculate_overall_score(
    clipping_factor: f32,
    dc_offset_factor: f32,
    dynamic_range_db: f32,
    snr_db: f32,
    spectral_flatness: f32,
) -> f32 {
    // Penalize clipping heavily
    let clipping_score = 1.0 - clipping_factor;

    // Penalize DC offset
    let dc_score = 1.0 - dc_offset_factor;

    // Normalize dynamic range (expecting 40-80 dB)
    let dr_score = ((dynamic_range_db - 20.0) / 60.0).clamp(0.0, 1.0);

    // Normalize SNR (expecting 20-60 dB)
    let snr_score = ((snr_db - 10.0) / 50.0).clamp(0.0, 1.0);

    // Spectral flatness around 0.3-0.5 is good (not too tonal, not too noisy)
    let sf_score = 1.0 - (spectral_flatness - 0.4).abs() / 0.6;

    // Weighted average
    let weights = [0.3, 0.2, 0.2, 0.2, 0.1]; // clipping, dc, dr, snr, flatness
    let scores = [clipping_score, dc_score, dr_score, snr_score, sf_score];

    weights
        .iter()
        .zip(scores.iter())
        .map(|(&w, &s)| w * s)
        .sum::<f32>()
        .clamp(0.0, 1.0)
}

/// Quick quality check (returns true if quality is acceptable)
pub fn quick_quality_check(audio: &AudioBuffer, min_score: f32) -> Result<bool> {
    let assessment = QualityAssessment::assess(audio)?;
    Ok(assessment.overall_score >= min_score)
}

/// Compare quality between two audio buffers
pub fn compare_quality(
    reference: &AudioBuffer,
    generated: &AudioBuffer,
) -> Result<(QualityAssessment, QualityAssessment, f32)> {
    let ref_assessment = QualityAssessment::assess(reference)?;
    let gen_assessment = QualityAssessment::assess(generated)?;

    // Calculate relative quality (how much worse/better is generated vs reference)
    let relative_quality = gen_assessment.overall_score / ref_assessment.overall_score.max(0.01);

    Ok((ref_assessment, gen_assessment, relative_quality))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_assessment_clean_signal() {
        // Create a clean sine wave
        let sample_rate = 22050;
        let duration = 1.0;
        let frequency = 440.0;
        let num_samples = (sample_rate as f32 * duration) as usize;

        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect();

        let audio = AudioBuffer::new(samples, sample_rate, 1);
        let assessment = QualityAssessment::assess(&audio).unwrap();

        // Clean sine wave should have good quality
        assert!(assessment.overall_score > 0.5);
        assert_eq!(assessment.clipping_factor, 0.0);
        assert!(!assessment.issues.contains(&QualityIssue::Clipping));
    }

    #[test]
    fn test_clipping_detection() {
        // Create a signal with clipping
        let samples: Vec<f32> = vec![1.0, -1.0, 1.0, -1.0, 0.5, -0.5];
        let audio = AudioBuffer::new(samples, 22050, 1);

        let assessment = QualityAssessment::assess(&audio).unwrap();
        assert!(assessment.clipping_factor > 0.0);
        assert!(assessment.issues.contains(&QualityIssue::Clipping));
    }

    #[test]
    fn test_dc_offset_detection() {
        // Create a signal with DC offset
        let samples: Vec<f32> = vec![0.6, 0.7, 0.5, 0.6, 0.7, 0.5];
        let audio = AudioBuffer::new(samples, 22050, 1);

        let assessment = QualityAssessment::assess(&audio).unwrap();
        assert!(assessment.dc_offset_factor > 0.0);
    }

    #[test]
    fn test_silence_detection() {
        // Create a near-silent signal
        let samples: Vec<f32> = vec![0.0001, -0.0001, 0.0, 0.0001, -0.0001];
        let audio = AudioBuffer::new(samples, 22050, 1);

        let assessment = QualityAssessment::assess(&audio).unwrap();
        assert!(assessment.issues.contains(&QualityIssue::Silence));
    }

    #[test]
    fn test_quality_tier_classification() {
        let excellent = QualityAssessment {
            overall_score: 0.95,
            clipping_factor: 0.0,
            dc_offset_factor: 0.0,
            dynamic_range_db: 60.0,
            snr_estimate_db: 40.0,
            spectral_flatness: 0.4,
            zero_crossing_rate: 0.3,
            quality_tier: QualityTier::Excellent,
            issues: vec![],
        };

        assert_eq!(excellent.quality_tier, QualityTier::Excellent);
        assert!(excellent.meets_threshold(0.9));
    }

    #[test]
    fn test_compare_quality() {
        let ref_samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let gen_samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin() * 0.9).collect();

        let reference = AudioBuffer::new(ref_samples, 22050, 1);
        let generated = AudioBuffer::new(gen_samples, 22050, 1);

        let (_ref_assess, _gen_assess, relative) = compare_quality(&reference, &generated).unwrap();

        // Generated should have similar quality
        assert!(relative > 0.8);
        assert!(relative < 1.2);
    }

    /// Generate deterministic pseudo-random white noise in `[-amplitude, amplitude]`
    /// using a fixed linear congruential generator (no RNG crate dependency).
    fn deterministic_white_noise(len: usize, amplitude: f32, seed: u64) -> Vec<f32> {
        // Numerical Recipes LCG constants.
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Use the high 32 bits for better-quality output.
                let unit = (state >> 32) as f32 / u32::MAX as f32; // [0, 1]
                (unit * 2.0 - 1.0) * amplitude
            })
            .collect()
    }

    #[test]
    fn test_spectral_flatness_white_noise_near_one() {
        // White noise has a roughly flat spectrum -> flatness near 1.0.
        let samples = deterministic_white_noise(8192, 0.5, 0x1234_5678_9abc_def0);
        let flatness = calculate_spectral_flatness_quick(&samples);

        assert!(
            flatness > 0.4,
            "white-noise flatness should be high, got {flatness}"
        );
        assert!(
            (0.0..=1.0).contains(&flatness),
            "flatness must be within [0, 1], got {flatness}"
        );
    }

    #[test]
    fn test_spectral_flatness_pure_tone_near_zero() {
        // A pure sine concentrates energy in a single bin -> flatness near 0.
        let sample_rate = 22050.0_f32;
        let frequency = 440.0_f32;
        let samples: Vec<f32> = (0..8192)
            .map(|i| {
                let t = i as f32 / sample_rate;
                0.5 * (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect();

        let flatness = calculate_spectral_flatness_quick(&samples);

        assert!(
            flatness < 0.1,
            "pure-tone flatness should be near zero, got {flatness}"
        );
        assert!(
            (0.0..=1.0).contains(&flatness),
            "flatness must be within [0, 1], got {flatness}"
        );
    }

    #[test]
    fn test_spectral_flatness_noise_greater_than_tone() {
        let noise = deterministic_white_noise(8192, 0.5, 0x0fed_cba9_8765_4321);
        let sample_rate = 22050.0_f32;
        let frequency = 220.0_f32;
        let tone: Vec<f32> = (0..8192)
            .map(|i| {
                let t = i as f32 / sample_rate;
                0.5 * (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect();

        let noise_flatness = calculate_spectral_flatness_quick(&noise);
        let tone_flatness = calculate_spectral_flatness_quick(&tone);

        assert!(
            noise_flatness > tone_flatness,
            "noise flatness ({noise_flatness}) should exceed tone flatness ({tone_flatness})"
        );
    }

    #[test]
    fn test_spectral_flatness_short_signal_is_safe() {
        // Below the minimum window size, the function must not panic.
        let flatness = calculate_spectral_flatness_quick(&[0.1, -0.2, 0.05]);
        assert!((0.0..=1.0).contains(&flatness));
    }
}
