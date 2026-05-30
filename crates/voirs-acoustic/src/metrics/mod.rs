//! Audio quality metrics for TTS evaluation
//!
//! This module provides comprehensive audio quality metrics for evaluating
//! text-to-speech synthesis quality, including objective, perceptual, and
//! prosody-specific measurements.

use crate::{AcousticError, MelSpectrogram, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod objective;
pub mod perceptual;
pub mod prosody;

pub use objective::*;
pub use perceptual::*;
pub use prosody::*;

/// Comprehensive quality evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Objective quality metrics
    pub objective: ObjectiveMetrics,
    /// Perceptual quality metrics
    pub perceptual: PerceptualMetrics,
    /// Prosody-specific metrics
    pub prosody: ProsodyMetrics,
    /// Overall quality score (0-100)
    pub overall_score: f32,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl QualityMetrics {
    /// Calculate overall quality score from component metrics
    pub fn calculate_overall_score(&mut self) {
        // Weighted combination of different metric categories
        let objective_weight = 0.3;
        let perceptual_weight = 0.5;
        let prosody_weight = 0.2;

        let objective_score = (self.objective.snr.max(0.0) / 30.0).min(1.0) * 100.0;
        let perceptual_score = self.perceptual.overall_score;
        let prosody_score = self.prosody.overall_score;

        self.overall_score = objective_weight * objective_score
            + perceptual_weight * perceptual_score
            + prosody_weight * prosody_score;
    }
}

/// Audio quality evaluator
pub struct QualityEvaluator {
    /// Configuration for quality evaluation
    config: EvaluationConfig,
}

/// Configuration for quality evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    /// Whether to compute expensive perceptual metrics
    pub compute_perceptual: bool,
    /// Whether to compute prosody metrics
    pub compute_prosody: bool,
    /// Sample rate for audio processing
    pub sample_rate: u32,
    /// Reference audio path (if available)
    pub reference_path: Option<String>,
    /// Evaluation presets
    pub preset: EvaluationPreset,
}

/// Predefined evaluation presets
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EvaluationPreset {
    /// Fast evaluation with basic metrics
    Fast,
    /// Standard evaluation with most metrics
    Standard,
    /// Comprehensive evaluation with all metrics
    Comprehensive,
    /// Research-grade evaluation with detailed analysis
    Research,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            compute_perceptual: true,
            compute_prosody: true,
            sample_rate: 22050,
            reference_path: None,
            preset: EvaluationPreset::Standard,
        }
    }
}

impl Default for QualityEvaluator {
    fn default() -> Self {
        Self::new(EvaluationConfig::default())
    }
}

impl QualityEvaluator {
    /// Create new quality evaluator
    pub fn new(config: EvaluationConfig) -> Self {
        Self { config }
    }

    /// Create evaluator with preset configuration
    pub fn with_preset(preset: EvaluationPreset) -> Self {
        let mut config = EvaluationConfig {
            preset,
            ..Default::default()
        };

        match preset {
            EvaluationPreset::Fast => {
                config.compute_perceptual = false;
                config.compute_prosody = false;
            }
            EvaluationPreset::Standard => {
                config.compute_perceptual = true;
                config.compute_prosody = false;
            }
            EvaluationPreset::Comprehensive => {
                config.compute_perceptual = true;
                config.compute_prosody = true;
            }
            EvaluationPreset::Research => {
                config.compute_perceptual = true;
                config.compute_prosody = true;
            }
        }

        Self::new(config)
    }

    /// Evaluate audio quality from mel spectrogram
    pub fn evaluate_mel_spectrogram(
        &self,
        mel_spec: &MelSpectrogram,
        reference_mel: Option<&MelSpectrogram>,
    ) -> Result<QualityMetrics> {
        let mut metrics = QualityMetrics {
            objective: ObjectiveMetrics::default(),
            perceptual: PerceptualMetrics::default(),
            prosody: ProsodyMetrics::default(),
            overall_score: 0.0,
            metadata: HashMap::new(),
        };

        // Add metadata
        metrics
            .metadata
            .insert("sample_rate".to_string(), mel_spec.sample_rate.to_string());
        metrics
            .metadata
            .insert("n_mels".to_string(), mel_spec.n_mels.to_string());
        metrics
            .metadata
            .insert("n_frames".to_string(), mel_spec.n_frames.to_string());
        metrics
            .metadata
            .insert("duration".to_string(), mel_spec.duration().to_string());

        // Compute objective metrics
        self.compute_objective_metrics(mel_spec, reference_mel, &mut metrics.objective)?;

        // Compute perceptual metrics if enabled
        if self.config.compute_perceptual {
            self.compute_perceptual_metrics(mel_spec, reference_mel, &mut metrics.perceptual)?;
        }

        // Compute prosody metrics if enabled
        if self.config.compute_prosody {
            self.compute_prosody_metrics(mel_spec, reference_mel, &mut metrics.prosody)?;
        }

        // Calculate overall score
        metrics.calculate_overall_score();

        Ok(metrics)
    }

    /// Evaluate audio quality from raw audio samples
    pub fn evaluate_audio_samples(
        &self,
        samples: &[f32],
        reference_samples: Option<&[f32]>,
    ) -> Result<QualityMetrics> {
        // Convert audio to mel spectrogram
        let mel_spec = self.audio_to_mel_spectrogram(samples)?;
        let reference_mel = if let Some(ref_samples) = reference_samples {
            Some(self.audio_to_mel_spectrogram(ref_samples)?)
        } else {
            None
        };

        self.evaluate_mel_spectrogram(&mel_spec, reference_mel.as_ref())
    }

    /// Compare two audio samples directly
    pub fn compare_audio_samples(
        &self,
        generated: &[f32],
        reference: &[f32],
    ) -> Result<QualityMetrics> {
        self.evaluate_audio_samples(generated, Some(reference))
    }

    /// Batch evaluation of multiple audio samples
    pub fn evaluate_batch(
        &self,
        samples_batch: &[&[f32]],
        reference_batch: Option<&[&[f32]]>,
    ) -> Result<Vec<QualityMetrics>> {
        let mut results = Vec::with_capacity(samples_batch.len());

        for (i, samples) in samples_batch.iter().enumerate() {
            let reference = reference_batch.and_then(|refs| refs.get(i).copied());
            let metrics = self.evaluate_audio_samples(samples, reference)?;
            results.push(metrics);
        }

        Ok(results)
    }

    /// Compute aggregate statistics from batch evaluation
    pub fn compute_aggregate_statistics(
        &self,
        metrics_batch: &[QualityMetrics],
    ) -> QualityStatistics {
        if metrics_batch.is_empty() {
            return QualityStatistics::default();
        }

        let _n = metrics_batch.len() as f32;

        // Aggregate objective metrics
        let snr_values: Vec<f32> = metrics_batch.iter().map(|m| m.objective.snr).collect();
        let thd_values: Vec<f32> = metrics_batch.iter().map(|m| m.objective.thd).collect();

        // Aggregate perceptual metrics
        let pesq_values: Vec<f32> = metrics_batch
            .iter()
            .map(|m| m.perceptual.pesq_score)
            .collect();
        let stoi_values: Vec<f32> = metrics_batch
            .iter()
            .map(|m| m.perceptual.stoi_score)
            .collect();

        // Aggregate overall scores
        let overall_values: Vec<f32> = metrics_batch.iter().map(|m| m.overall_score).collect();

        QualityStatistics {
            count: metrics_batch.len(),
            snr: self.compute_statistics(&snr_values),
            thd: self.compute_statistics(&thd_values),
            pesq: self.compute_statistics(&pesq_values),
            stoi: self.compute_statistics(&stoi_values),
            overall_score: self.compute_statistics(&overall_values),
        }
    }

    // Private helper methods

    fn compute_objective_metrics(
        &self,
        mel_spec: &MelSpectrogram,
        reference_mel: Option<&MelSpectrogram>,
        metrics: &mut ObjectiveMetrics,
    ) -> Result<()> {
        let evaluator = ObjectiveEvaluator::new();

        // Compute SNR
        metrics.snr = evaluator.compute_snr(&mel_spec.data)?;

        // Compute THD
        metrics.thd = evaluator.compute_thd(&mel_spec.data)?;

        // Compute spectral distortion if reference is available
        if let Some(reference) = reference_mel {
            metrics.spectral_distortion =
                Some(evaluator.compute_spectral_distortion(&mel_spec.data, &reference.data)?);

            metrics.mcd =
                Some(evaluator.compute_mel_cepstral_distortion(&mel_spec.data, &reference.data)?);
        }

        // Compute pitch-related metrics
        metrics.pitch_correlation = evaluator.compute_pitch_correlation(&mel_spec.data)?;

        Ok(())
    }

    fn compute_perceptual_metrics(
        &self,
        mel_spec: &MelSpectrogram,
        reference_mel: Option<&MelSpectrogram>,
        metrics: &mut PerceptualMetrics,
    ) -> Result<()> {
        let evaluator = PerceptualEvaluator::new();

        // Convert mel to audio for perceptual evaluation
        let audio_samples = self.mel_to_audio_samples(mel_spec)?;

        if let Some(reference) = reference_mel {
            let reference_samples = self.mel_to_audio_samples(reference)?;

            // Compute PESQ
            metrics.pesq_score = evaluator.compute_pesq(&audio_samples, &reference_samples)?;

            // Compute STOI
            metrics.stoi_score = evaluator.compute_stoi(&audio_samples, &reference_samples)?;

            // Compute SI-SDR
            metrics.si_sdr = Some(evaluator.compute_si_sdr(&audio_samples, &reference_samples)?);
        } else {
            // Compute intrinsic quality metrics
            metrics.pesq_score = evaluator.compute_intrinsic_quality(&audio_samples)?;
            metrics.stoi_score = 85.0; // Default reasonable value
        }

        // Calculate overall perceptual score
        metrics.overall_score = (metrics.pesq_score * 20.0 + metrics.stoi_score) / 2.0;

        Ok(())
    }

    fn compute_prosody_metrics(
        &self,
        mel_spec: &MelSpectrogram,
        reference_mel: Option<&MelSpectrogram>,
        metrics: &mut ProsodyMetrics,
    ) -> Result<()> {
        let evaluator = ProsodyEvaluator::new();

        // Extract prosody features
        let prosody_features = evaluator.extract_prosody_features(&mel_spec.data)?;

        if let Some(reference) = reference_mel {
            let reference_features = evaluator.extract_prosody_features(&reference.data)?;

            // Compute duration accuracy
            metrics.duration_accuracy = evaluator.compute_duration_accuracy(
                &prosody_features.durations,
                &reference_features.durations,
            )?;

            // Compute pitch correlation
            metrics.pitch_correlation = evaluator.compute_pitch_correlation(
                &prosody_features.pitch_contour,
                &reference_features.pitch_contour,
            )?;

            // Compute stress pattern preservation
            metrics.stress_preservation = evaluator.compute_stress_preservation(
                &prosody_features.stress_pattern,
                &reference_features.stress_pattern,
            )?;

            // Compute rhythm naturalness
            metrics.rhythm_naturalness = evaluator.compute_rhythm_naturalness(
                &prosody_features.rhythm_features,
                &reference_features.rhythm_features,
            )?;
        } else {
            // Compute intrinsic prosody quality
            metrics.duration_accuracy =
                evaluator.compute_intrinsic_duration_quality(&prosody_features.durations)?;
            metrics.pitch_correlation =
                evaluator.compute_intrinsic_pitch_quality(&prosody_features.pitch_contour)?;
            metrics.stress_preservation = 85.0; // Default reasonable value
            metrics.rhythm_naturalness =
                evaluator.compute_intrinsic_rhythm_quality(&prosody_features.rhythm_features)?;
        }

        // Calculate overall prosody score
        metrics.overall_score = (metrics.duration_accuracy
            + metrics.pitch_correlation
            + metrics.stress_preservation
            + metrics.rhythm_naturalness)
            / 4.0;

        Ok(())
    }

    fn audio_to_mel_spectrogram(&self, samples: &[f32]) -> Result<MelSpectrogram> {
        use std::f64::consts::PI;

        if samples.is_empty() {
            return Err(AcousticError::ProcessingError {
                message: "Cannot compute mel spectrogram of empty audio".to_string(),
            });
        }

        let sample_rate = self.config.sample_rate;
        let n_fft: usize = 1024;
        let hop_length: usize = 256;
        let n_mels: usize = 80;
        let n_freqs = n_fft / 2 + 1;

        // Build Hann window
        let window: Vec<f64> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n_fft - 1) as f64).cos()))
            .collect();

        // Hz/mel conversion helpers
        let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f64| 700.0 * (10_f64.powf(mel / 2595.0) - 1.0);

        let f_min = 0.0_f64;
        let f_max = sample_rate as f64 / 2.0;
        let mel_min = hz_to_mel(f_min);
        let mel_max = hz_to_mel(f_max);

        // Build mel filterbank: shape [n_mels][n_freqs]
        let mel_points: Vec<f64> = (0..=n_mels + 1)
            .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
            .collect();
        let bin_points: Vec<usize> = mel_points
            .iter()
            .map(|&f| ((n_fft + 1) as f64 * f / sample_rate as f64).floor() as usize)
            .collect();

        let mut filterbank = vec![vec![0.0_f32; n_freqs]; n_mels];
        for (mel_idx, fb_row) in filterbank.iter_mut().enumerate() {
            let left = bin_points[mel_idx].min(n_freqs - 1);
            let center = bin_points[mel_idx + 1].min(n_freqs - 1);
            let right = bin_points[mel_idx + 2].min(n_freqs - 1);

            if center > left {
                for (offset, slot) in fb_row[left..center].iter_mut().enumerate() {
                    *slot = offset as f32 / (center - left) as f32;
                }
            }
            if right > center {
                for (offset, slot) in fb_row[center..right].iter_mut().enumerate() {
                    let bin = center + offset;
                    *slot = (right - bin) as f32 / (right - center) as f32;
                }
            }

            // Normalise each filter by its area so energy is preserved
            let area: f32 = fb_row.iter().sum();
            if area > 0.0 {
                for v in fb_row.iter_mut() {
                    *v /= area;
                }
            }
        }

        // Number of STFT frames
        let n_frames = if samples.len() >= n_fft {
            (samples.len() - n_fft) / hop_length + 1
        } else {
            1
        };

        // mel_data[mel_idx][frame_idx]
        let mut mel_data: Vec<Vec<f32>> = vec![vec![0.0_f32; n_frames]; n_mels];

        // Compute all STFT frames; scatter log-mel values into mel_data[mel_idx][frame_idx].
        // We iterate over a range because we need `frame_idx` to compute the time offset
        // AND to index the second dimension of mel_data, which is accessed via mel_row[frame_idx].
        #[allow(clippy::needless_range_loop)]
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;

            // Extract windowed frame as f64 (zero-pad at the end if necessary)
            let frame_f64: Vec<f64> = (0..n_fft)
                .map(|i| {
                    let idx = start + i;
                    if idx < samples.len() {
                        samples[idx] as f64 * window[i]
                    } else {
                        0.0
                    }
                })
                .collect();

            // Forward FFT via scirs2_fft
            let fft_result = scirs2_fft::fft(&frame_f64, Some(n_fft)).map_err(|e| {
                AcousticError::ProcessingError {
                    message: format!("FFT error: {e}"),
                }
            })?;

            // Power spectrum for the one-sided spectrum (DC through Nyquist)
            let power_spec: Vec<f32> = fft_result
                .iter()
                .take(n_freqs)
                .map(|c| (c.re * c.re + c.im * c.im) as f32)
                .collect();

            // Apply filterbank and convert to log scale; scatter into each mel row
            for (fb_row, mel_row) in filterbank.iter().zip(mel_data.iter_mut()) {
                let mel_energy: f32 = power_spec
                    .iter()
                    .zip(fb_row.iter())
                    .map(|(&p, &w)| p * w)
                    .sum();
                mel_row[frame_idx] = (mel_energy + 1e-10_f32).ln();
            }
        }

        Ok(MelSpectrogram::new(
            mel_data,
            sample_rate,
            hop_length as u32,
        ))
    }

    fn mel_to_audio_samples(&self, mel_spec: &MelSpectrogram) -> Result<Vec<f32>> {
        use std::f64::consts::PI;

        let n_fft: usize = 1024;
        let hop_length = mel_spec.hop_length as usize;
        let n_mels = mel_spec.n_mels;
        let n_frames = mel_spec.n_frames;
        let n_freqs = n_fft / 2 + 1;
        let sample_rate = mel_spec.sample_rate;

        if n_frames == 0 || n_mels == 0 {
            return Ok(vec![]);
        }

        // Reconstruct the mel filterbank matching the forward transform parameters
        let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f64| 700.0 * (10_f64.powf(mel / 2595.0) - 1.0);

        let f_min = 0.0_f64;
        let f_max = sample_rate as f64 / 2.0;
        let mel_min = hz_to_mel(f_min);
        let mel_max = hz_to_mel(f_max);

        let mel_points: Vec<f64> = (0..=n_mels + 1)
            .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
            .collect();
        let bin_points: Vec<usize> = mel_points
            .iter()
            .map(|&f| ((n_fft + 1) as f64 * f / sample_rate as f64).floor() as usize)
            .collect();

        let mut filterbank = vec![vec![0.0_f64; n_freqs]; n_mels];
        for (mel_idx, fb_row) in filterbank.iter_mut().enumerate() {
            let left = bin_points[mel_idx].min(n_freqs - 1);
            let center = bin_points[mel_idx + 1].min(n_freqs - 1);
            let right = bin_points[mel_idx + 2].min(n_freqs - 1);

            if center > left {
                for (offset, slot) in fb_row[left..center].iter_mut().enumerate() {
                    *slot = offset as f64 / (center - left) as f64;
                }
            }
            if right > center {
                for (offset, slot) in fb_row[center..right].iter_mut().enumerate() {
                    let bin = center + offset;
                    *slot = (right - bin) as f64 / (right - center) as f64;
                }
            }

            let area: f64 = fb_row.iter().sum();
            if area > 0.0 {
                for v in fb_row.iter_mut() {
                    *v /= area;
                }
            }
        }

        // Analysis (Hann) window
        let window: Vec<f64> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n_fft - 1) as f64).cos()))
            .collect();

        // Invert log scale: exp(log_mel) => linear mel energy
        // Then pseudo-invert the filterbank (transposed product) to get power spectrum per frame
        // For each frame: power_spec[k] = sum_m mel_data[m][frame] * filterbank[m][k]
        // Frames are then synthesised with zero-phase IFFT and Hann synthesis window,
        // then overlap-added into the output signal.

        let output_len = hop_length * (n_frames - 1) + n_fft;
        let mut output_signal = vec![0.0_f64; output_len];
        let mut window_sum = vec![0.0_f64; output_len];

        for frame_idx in 0..n_frames {
            // Convert log-mel back to linear mel energy for this frame
            let linear_mel: Vec<f64> = (0..n_mels)
                .map(|mel_idx| {
                    if mel_idx < mel_spec.data.len() && frame_idx < mel_spec.data[mel_idx].len() {
                        (mel_spec.data[mel_idx][frame_idx] as f64).exp()
                    } else {
                        0.0
                    }
                })
                .collect();

            // Invert filterbank via transposed product: power_spec[k] = Σ_m mel[m]*fb[m][k]
            let power_spec: Vec<f64> = (0..n_freqs)
                .map(|k| (0..n_mels).map(|m| linear_mel[m] * filterbank[m][k]).sum())
                .collect();

            // Magnitude spectrum (sqrt of power); zero-phase reconstruction
            let magnitudes: Vec<f64> = power_spec.iter().map(|&p| p.sqrt()).collect();

            // Build a full symmetric complex spectrum from the one-sided real magnitudes.
            // Zero phase means all imaginary parts are zero; the real part IS the magnitude
            // at each bin. Reconstruct via IFFT through the identity:
            //   IFFT( full_symmetric_spectrum ) = real time-domain frame.
            // We use scirs2_fft::irfft which expects one-sided complex spectrum.
            let one_sided_complex: Vec<scirs2_core::numeric::Complex64> = magnitudes
                .iter()
                .map(|&m| scirs2_core::numeric::Complex64::new(m, 0.0))
                .collect();

            let frame_td = scirs2_fft::irfft(&one_sided_complex, Some(n_fft)).map_err(|e| {
                AcousticError::ProcessingError {
                    message: format!("IRFFT error: {e}"),
                }
            })?;

            // Overlap-add with synthesis Hann window
            let start = frame_idx * hop_length;
            for i in 0..n_fft {
                let out_idx = start + i;
                if out_idx < output_len {
                    let w = window[i];
                    output_signal[out_idx] += frame_td[i] * w;
                    window_sum[out_idx] += w * w;
                }
            }
        }

        // Normalise by overlap-add window envelope to avoid amplitude distortion
        let result: Vec<f32> = output_signal
            .into_iter()
            .zip(window_sum)
            .map(|(s, w)| if w > 1e-8 { (s / w) as f32 } else { 0.0_f32 })
            .collect();

        Ok(result)
    }

    fn compute_statistics(&self, values: &[f32]) -> MetricStatistics {
        if values.is_empty() {
            return MetricStatistics::default();
        }

        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        let mut sorted_values = values.to_vec();
        // Sort with NaN handling for statistical calculations
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];
        let median = if sorted_values.len().is_multiple_of(2) {
            (sorted_values[sorted_values.len() / 2 - 1] + sorted_values[sorted_values.len() / 2])
                / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        MetricStatistics {
            mean,
            std_dev,
            min,
            max,
            median,
        }
    }
}

/// Statistical summary for metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityStatistics {
    /// Number of samples evaluated
    pub count: usize,
    /// SNR statistics
    pub snr: MetricStatistics,
    /// THD statistics
    pub thd: MetricStatistics,
    /// PESQ statistics
    pub pesq: MetricStatistics,
    /// STOI statistics
    pub stoi: MetricStatistics,
    /// Overall score statistics
    pub overall_score: MetricStatistics,
}

/// Statistical measures for a single metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStatistics {
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Median value
    pub median: f32,
}

impl Default for MetricStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_evaluator_creation() {
        let evaluator = QualityEvaluator::default();
        assert!(evaluator.config.compute_perceptual);
        assert!(evaluator.config.compute_prosody);
        assert_eq!(evaluator.config.sample_rate, 22050);
    }

    #[test]
    fn test_evaluation_presets() {
        let fast_evaluator = QualityEvaluator::with_preset(EvaluationPreset::Fast);
        assert!(!fast_evaluator.config.compute_perceptual);
        assert!(!fast_evaluator.config.compute_prosody);

        let comprehensive_evaluator =
            QualityEvaluator::with_preset(EvaluationPreset::Comprehensive);
        assert!(comprehensive_evaluator.config.compute_perceptual);
        assert!(comprehensive_evaluator.config.compute_prosody);
    }

    #[test]
    fn test_statistics_computation() {
        let evaluator = QualityEvaluator::default();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let stats = evaluator.compute_statistics(&values);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.median, 3.0);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_audio_mel_roundtrip_shape() {
        // Generate 1000 samples of a 440 Hz sine wave at 22050 Hz
        let sample_rate = 22050u32;
        let n_samples = 1000usize;
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let config = EvaluationConfig {
            sample_rate,
            ..Default::default()
        };
        let evaluator = QualityEvaluator::new(config);

        let mel = evaluator
            .audio_to_mel_spectrogram(&samples)
            .expect("mel spectrogram should succeed");
        assert!(mel.n_frames > 0, "expected at least one mel frame");
        assert_eq!(mel.n_mels, 80);

        let reconstructed = evaluator
            .mel_to_audio_samples(&mel)
            .expect("mel to audio should succeed");

        // Output length must be within 20 % of input
        let lo = (n_samples as f32 * 0.8) as usize;
        let hi = (n_samples as f32 * 1.2) as usize;
        assert!(
            reconstructed.len() >= lo && reconstructed.len() <= hi,
            "reconstructed length {} not within 20 % of input {}",
            reconstructed.len(),
            n_samples
        );
    }

    #[test]
    fn test_mel_spectrogram_nonnegative() {
        // Mel values are log-scaled, so they can be negative, but must not be NaN or -inf
        let sample_rate = 22050u32;
        let n_samples = 2048usize;
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let config = EvaluationConfig {
            sample_rate,
            ..Default::default()
        };
        let evaluator = QualityEvaluator::new(config);

        let mel = evaluator
            .audio_to_mel_spectrogram(&samples)
            .expect("mel spectrogram should succeed");

        for (mel_idx, row) in mel.data.iter().enumerate() {
            for (frame_idx, &v) in row.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "mel_data[{mel_idx}][{frame_idx}] = {v} is not finite"
                );
            }
        }
    }

    #[test]
    fn test_audio_to_mel_empty() {
        let evaluator = QualityEvaluator::default();
        let result = evaluator.audio_to_mel_spectrogram(&[]);
        assert!(
            result.is_err(),
            "empty audio should return an error, not panic"
        );
    }

    #[test]
    fn test_quality_metrics_overall_score() {
        let mut metrics = QualityMetrics {
            objective: ObjectiveMetrics {
                snr: 20.0,
                thd: 0.1,
                spectral_distortion: Some(0.5),
                mcd: Some(0.8),
                pitch_correlation: 0.9,
            },
            perceptual: PerceptualMetrics {
                pesq_score: 3.5,
                stoi_score: 0.85,
                si_sdr: Some(15.0),
                overall_score: 80.0,
            },
            prosody: ProsodyMetrics {
                duration_accuracy: 90.0,
                pitch_correlation: 85.0,
                stress_preservation: 88.0,
                rhythm_naturalness: 87.0,
                overall_score: 87.5,
            },
            overall_score: 0.0,
            metadata: HashMap::new(),
        };

        metrics.calculate_overall_score();
        assert!(metrics.overall_score > 0.0);
        assert!(metrics.overall_score <= 100.0);
    }
}
