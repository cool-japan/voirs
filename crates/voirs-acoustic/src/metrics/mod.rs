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

    fn audio_to_mel_spectrogram(&self, _samples: &[f32]) -> Result<MelSpectrogram> {
        // This would typically use a mel computation module
        // For now, return a placeholder implementation
        Err(AcousticError::Processing(
            "Audio to mel conversion not yet implemented".to_string(),
        ))
    }

    fn mel_to_audio_samples(&self, _mel_spec: &MelSpectrogram) -> Result<Vec<f32>> {
        // This would typically use a vocoder
        // For now, return a placeholder implementation
        Err(AcousticError::Processing(
            "Mel to audio conversion not yet implemented".to_string(),
        ))
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
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];
        let median = if sorted_values.len() % 2 == 0 {
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
