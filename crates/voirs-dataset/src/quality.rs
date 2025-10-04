//! Quality control and metrics for speech synthesis datasets
//!
//! This module provides quality assessment, filtering, and review tools
//! for maintaining high-quality speech synthesis datasets.

pub mod filters;
pub mod metrics;
pub mod review;

use crate::{DatasetSample, QualityMetrics, Result};
use serde::{Deserialize, Serialize};

/// Quality assessment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    /// Minimum SNR threshold
    pub min_snr: f32,
    /// Maximum clipping percentage
    pub max_clipping: f32,
    /// Minimum dynamic range
    pub min_dynamic_range: f32,
    /// Minimum duration in seconds
    pub min_duration: f32,
    /// Maximum duration in seconds
    pub max_duration: f32,
    /// Minimum quality score
    pub min_quality_score: f32,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            min_snr: 15.0,
            max_clipping: 0.01,
            min_dynamic_range: 20.0,
            min_duration: 0.5,
            max_duration: 15.0,
            min_quality_score: 0.5,
        }
    }
}

/// Quality assessor for dataset samples
pub struct QualityAssessor {
    config: QualityConfig,
}

impl QualityAssessor {
    /// Create new quality assessor
    pub fn new(config: QualityConfig) -> Self {
        Self { config }
    }

    /// Assess quality of a sample
    pub fn assess_sample(&self, sample: &DatasetSample) -> Result<QualityMetrics> {
        let audio = &sample.audio;
        let samples = audio.samples();

        // Calculate SNR
        let snr = self.calculate_snr(samples);

        // Calculate clipping percentage
        let clipping = self.calculate_clipping(samples);

        // Calculate dynamic range
        let dynamic_range = self.calculate_dynamic_range(samples);

        // Calculate overall quality score
        let overall_quality =
            self.calculate_overall_quality(snr, clipping, dynamic_range, sample.duration());

        // Calculate spectral quality
        let spectral_quality = self.calculate_spectral_quality(samples, audio.sample_rate());

        Ok(QualityMetrics {
            snr: Some(snr),
            clipping: Some(clipping),
            dynamic_range: Some(dynamic_range),
            spectral_quality: Some(spectral_quality),
            overall_quality: Some(overall_quality),
        })
    }

    /// Check if sample meets quality standards
    pub fn meets_quality_standards(&self, sample: &DatasetSample) -> Result<bool> {
        let metrics = self.assess_sample(sample)?;

        // Check SNR
        if let Some(snr) = metrics.snr {
            if snr < self.config.min_snr {
                return Ok(false);
            }
        }

        // Check clipping
        if let Some(clipping) = metrics.clipping {
            if clipping > self.config.max_clipping {
                return Ok(false);
            }
        }

        // Check dynamic range
        if let Some(dynamic_range) = metrics.dynamic_range {
            if dynamic_range < self.config.min_dynamic_range {
                return Ok(false);
            }
        }

        // Check duration
        let duration = sample.duration();
        if duration < self.config.min_duration || duration > self.config.max_duration {
            return Ok(false);
        }

        // Check overall quality
        if let Some(quality) = metrics.overall_quality {
            if quality < self.config.min_quality_score {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Calculate spectral quality using FFT-based analysis
    fn calculate_spectral_quality(&self, samples: &[f32], sample_rate: u32) -> f32 {
        use scirs2_core::Complex;

        if samples.is_empty() || samples.len() < 512 {
            return 0.0;
        }

        // Use a window size that's a power of 2
        let window_size = 1024.min(samples.len());

        // Take the first window_size samples, apply Hanning window, and convert to f64
        let input_f64: Vec<scirs2_core::Complex<f64>> = samples[..window_size]
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let window_val = 0.5
                    * (1.0
                        - (2.0 * std::f32::consts::PI * i as f32 / (window_size - 1) as f32).cos());
                scirs2_core::Complex::new((x * window_val) as f64, 0.0)
            })
            .collect();

        // Perform FFT
        let fft_result = scirs2_fft::fft(&input_f64, None)
            .unwrap_or_else(|_| vec![scirs2_core::Complex::new(0.0, 0.0); window_size]);
        let buffer: Vec<Complex<f32>> = fft_result
            .iter()
            .map(|c| Complex::new(c.re as f32, c.im as f32))
            .collect();

        // Calculate magnitude spectrum (only use positive frequencies)
        let nyquist = window_size / 2;
        let magnitude_spectrum: Vec<f32> = buffer[..nyquist]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        let total_power: f32 = magnitude_spectrum.iter().sum();

        if total_power == 0.0 {
            return 0.0;
        }

        // Calculate spectral quality metrics
        let freq_resolution = sample_rate as f32 / window_size as f32;

        // Spectral centroid (brightness indicator)
        let mut weighted_sum = 0.0;
        for (i, &magnitude) in magnitude_spectrum.iter().enumerate() {
            let frequency = i as f32 * freq_resolution;
            weighted_sum += frequency * magnitude;
        }
        let spectral_centroid = weighted_sum / total_power;

        // Spectral rolloff (frequency spread indicator)
        let rolloff_threshold = 0.85 * total_power;
        let mut cumulative_power = 0.0;
        let mut rolloff_frequency = 0.0;

        for (i, &magnitude) in magnitude_spectrum.iter().enumerate() {
            cumulative_power += magnitude;
            if cumulative_power >= rolloff_threshold {
                rolloff_frequency = i as f32 * freq_resolution;
                break;
            }
        }

        // Calculate spectral flatness (tonality measure)
        let geometric_mean = magnitude_spectrum
            .iter()
            .filter(|&&x| x > 1e-10)
            .map(|&x| x.ln())
            .sum::<f32>()
            / magnitude_spectrum.len() as f32;
        let arithmetic_mean = total_power / magnitude_spectrum.len() as f32;
        let spectral_flatness = if arithmetic_mean > 0.0 {
            (geometric_mean.exp() / arithmetic_mean).min(1.0)
        } else {
            0.0
        };

        // Combine metrics into quality score
        let nyquist_freq = sample_rate as f32 / 2.0;

        // Normalize spectral centroid (prefer speech-like spectral centroid around 1000-3000 Hz)
        let centroid_score = if (800.0..=4000.0).contains(&spectral_centroid) {
            1.0 - ((spectral_centroid - 2000.0).abs() / 2000.0).min(1.0)
        } else if spectral_centroid < 800.0 {
            (spectral_centroid / 800.0).min(1.0)
        } else {
            ((nyquist_freq - spectral_centroid) / (nyquist_freq - 4000.0)).clamp(0.0, 1.0)
        };

        // Normalize rolloff frequency (prefer reasonable rolloff for speech)
        let rolloff_score = if (2000.0..=8000.0).contains(&rolloff_frequency) {
            1.0
        } else if rolloff_frequency < 2000.0 {
            (rolloff_frequency / 2000.0).min(1.0)
        } else {
            ((nyquist_freq - rolloff_frequency) / (nyquist_freq - 8000.0)).clamp(0.0, 1.0)
        };

        // Spectral flatness should be moderate for speech (not too tonal, not too noisy)
        let flatness_score = if (0.1..=0.7).contains(&spectral_flatness) {
            1.0
        } else if spectral_flatness < 0.1 {
            spectral_flatness / 0.1
        } else {
            (1.0 - spectral_flatness) / 0.3
        };

        // Weighted combination of metrics
        (0.4 * centroid_score + 0.3 * rolloff_score + 0.3 * flatness_score).clamp(0.0, 1.0)
    }

    /// Calculate signal-to-noise ratio using advanced noise estimation
    fn calculate_snr(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // Enhanced SNR calculation with noise floor estimation
        // Use frame-based analysis for better noise estimation
        let frame_size = 1024.min(samples.len() / 4);
        if frame_size < 32 {
            // Fallback for very short signals
            let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
            return if rms > 0.0 {
                20.0 * rms.log10() + 60.0
            } else {
                0.0
            };
        }

        // Calculate energy for each frame
        let mut frame_energies = Vec::new();
        for chunk in samples.chunks(frame_size) {
            let energy = chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32;
            frame_energies.push(energy);
        }

        // Sort energies to find noise floor (lowest 10% of frames)
        frame_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let noise_percentile = (frame_energies.len() as f32 * 0.1).max(1.0) as usize;
        let noise_energy =
            frame_energies[..noise_percentile].iter().sum::<f32>() / noise_percentile as f32;

        // Calculate signal energy (highest 30% of frames to avoid noise influence)
        let signal_start = (frame_energies.len() as f32 * 0.7) as usize;
        let signal_energy = if signal_start < frame_energies.len() {
            frame_energies[signal_start..].iter().sum::<f32>()
                / (frame_energies.len() - signal_start) as f32
        } else {
            frame_energies.iter().sum::<f32>() / frame_energies.len() as f32
        };

        // Calculate SNR in dB
        if noise_energy > 0.0 && signal_energy > noise_energy {
            10.0 * (signal_energy / noise_energy).log10()
        } else if signal_energy > 0.0 {
            // Fallback: estimate SNR based on dynamic range
            20.0 * signal_energy.sqrt().log10() + 60.0
        } else {
            0.0
        }
    }

    /// Calculate clipping percentage
    fn calculate_clipping(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let clipping_threshold = 0.99;
        let clipped_samples = samples
            .iter()
            .filter(|&&sample| sample.abs() >= clipping_threshold)
            .count();

        clipped_samples as f32 / samples.len() as f32
    }

    /// Calculate dynamic range
    fn calculate_dynamic_range(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let peak = samples
            .iter()
            .fold(0.0f32, |max, &sample| max.max(sample.abs()));
        let noise_floor = samples
            .iter()
            .map(|&x| x.abs())
            .fold(1.0f32, |min, sample| min.min(sample.max(1e-6)));

        20.0 * (peak / noise_floor).log10()
    }

    /// Calculate overall quality score
    fn calculate_overall_quality(
        &self,
        snr: f32,
        clipping: f32,
        dynamic_range: f32,
        duration: f32,
    ) -> f32 {
        let mut score = 1.0;

        // SNR penalty
        if snr < self.config.min_snr {
            score *= snr / self.config.min_snr;
        }

        // Clipping penalty
        if clipping > self.config.max_clipping {
            score *= (self.config.max_clipping / clipping).min(1.0);
        }

        // Dynamic range penalty
        if dynamic_range < self.config.min_dynamic_range {
            score *= dynamic_range / self.config.min_dynamic_range;
        }

        // Duration penalty
        if duration < self.config.min_duration {
            score *= duration / self.config.min_duration;
        } else if duration > self.config.max_duration {
            score *= self.config.max_duration / duration;
        }

        score.clamp(0.0, 1.0)
    }
}

/// Quality report for a dataset
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Total samples assessed
    pub total_samples: usize,
    /// Samples passing quality checks
    pub passing_samples: usize,
    /// Samples failing quality checks
    pub failing_samples: usize,
    /// Average quality score
    pub average_quality: f32,
    /// Quality distribution
    pub quality_distribution: std::collections::HashMap<String, usize>,
    /// Common quality issues
    pub common_issues: Vec<String>,
}

impl QualityReport {
    /// Create new quality report
    pub fn new(total_samples: usize) -> Self {
        Self {
            total_samples,
            passing_samples: 0,
            failing_samples: 0,
            average_quality: 0.0,
            quality_distribution: std::collections::HashMap::new(),
            common_issues: Vec::new(),
        }
    }

    /// Get pass rate
    pub fn pass_rate(&self) -> f32 {
        if self.total_samples == 0 {
            return 0.0;
        }
        self.passing_samples as f32 / self.total_samples as f32
    }
}

/// Batch quality processor
pub struct BatchQualityProcessor {
    assessor: QualityAssessor,
}

impl BatchQualityProcessor {
    /// Create new batch processor
    pub fn new(config: QualityConfig) -> Self {
        Self {
            assessor: QualityAssessor::new(config),
        }
    }

    /// Process multiple samples
    pub fn process_batch(&self, samples: &[DatasetSample]) -> Result<QualityReport> {
        let mut report = QualityReport::new(samples.len());
        let mut total_quality = 0.0;

        for sample in samples {
            let metrics = self.assessor.assess_sample(sample)?;
            let passes = self.assessor.meets_quality_standards(sample)?;

            if passes {
                report.passing_samples += 1;
            } else {
                report.failing_samples += 1;
            }

            if let Some(quality) = metrics.overall_quality {
                total_quality += quality;
            }
        }

        report.average_quality = if !samples.is_empty() {
            total_quality / samples.len() as f32
        } else {
            0.0
        };

        Ok(report)
    }
}
