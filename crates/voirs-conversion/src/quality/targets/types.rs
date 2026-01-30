//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::{
    adaptive_controller::{AdaptiveQualityController, QualityTrend, QualityTrigger},
    artifact_detection::{ArtifactDetector, ArtifactType, DetectedArtifacts},
    metrics::{ObjectiveQualityMetrics, QualityMetricsSystem},
};
use crate::{Error, Result};
use std::collections::HashMap;
use tracing::{debug, info};

/// Quality targets system for measuring and tracking conversion quality goals
pub struct QualityTargetsSystem {
    /// Configuration for quality target thresholds
    config: QualityTargetsConfig,
    /// History of quality measurements for trend analysis
    measurement_history: Vec<QualityTargetMeasurement>,
    /// Maximum history size
    max_history: usize,
}
impl QualityTargetsSystem {
    /// Create a new quality targets system
    pub fn new() -> Self {
        Self::with_config(QualityTargetsConfig::default())
    }
    /// Create a new quality targets system with custom configuration
    pub fn with_config(config: QualityTargetsConfig) -> Self {
        Self {
            config,
            measurement_history: Vec::new(),
            max_history: 1000,
        }
    }
    /// Measure quality targets for a conversion result
    pub fn measure_quality_targets(
        &mut self,
        converted_audio: &[f32],
        original_audio: &[f32],
        target_reference: Option<&[f32]>,
        sample_rate: u32,
    ) -> Result<QualityTargetMeasurement> {
        let target_similarity = if let Some(target_ref) = target_reference {
            self.calculate_target_similarity(converted_audio, target_ref, sample_rate)?
        } else {
            self.calculate_speaker_characteristics_similarity(converted_audio, sample_rate)?
        };
        let source_preservation =
            self.calculate_source_preservation(converted_audio, original_audio, sample_rate)?;
        let mos_score = self.calculate_mos_score(converted_audio, sample_rate)?;
        let artifact_level = self.calculate_artifact_level(converted_audio, sample_rate)?;
        let detailed_metrics = self.calculate_detailed_metrics(
            converted_audio,
            original_audio,
            target_reference,
            sample_rate,
        )?;
        let targets_met = QualityTargetsAchievement {
            target_similarity_met: target_similarity >= self.config.target_similarity_threshold,
            source_preservation_met: source_preservation
                >= self.config.source_preservation_threshold,
            mos_threshold_met: mos_score >= self.config.mos_threshold,
            artifact_level_met: artifact_level <= self.config.artifact_level_threshold,
            overall_achievement: self.calculate_overall_achievement(
                target_similarity,
                source_preservation,
                mos_score,
                artifact_level,
            ),
        };
        let measurement = QualityTargetMeasurement {
            target_similarity,
            source_preservation,
            mos_score,
            artifact_level,
            targets_met,
            timestamp: std::time::SystemTime::now(),
            detailed_metrics,
        };
        if self.config.enable_detailed_tracking {
            self.add_to_history(measurement.clone());
        }
        Ok(measurement)
    }
    /// Calculate target similarity (how similar the converted voice is to the target voice)
    fn calculate_target_similarity(
        &self,
        converted: &[f32],
        target: &[f32],
        sample_rate: u32,
    ) -> Result<f32> {
        let spectral_similarity = self.calculate_spectral_similarity(converted, target)?;
        let prosodic_similarity =
            self.calculate_prosodic_similarity(converted, target, sample_rate)?;
        let timbral_similarity = self.calculate_timbral_similarity(converted, target)?;
        let similarity =
            spectral_similarity * 0.4 + prosodic_similarity * 0.35 + timbral_similarity * 0.25;
        Ok(similarity.clamp(0.0, 1.0))
    }
    /// Calculate source preservation (how much of the original content is preserved)
    fn calculate_source_preservation(
        &self,
        converted: &[f32],
        original: &[f32],
        sample_rate: u32,
    ) -> Result<f32> {
        let linguistic_preservation =
            self.calculate_linguistic_preservation(converted, original, sample_rate)?;
        let temporal_preservation = self.calculate_temporal_preservation(converted, original)?;
        let semantic_preservation = self.calculate_semantic_preservation(converted, original)?;
        let preservation = linguistic_preservation * 0.5
            + temporal_preservation * 0.3
            + semantic_preservation * 0.2;
        Ok(preservation.clamp(0.0, 1.0))
    }
    /// Calculate MOS (Mean Opinion Score) estimate
    fn calculate_mos_score(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        let naturalness = self.estimate_naturalness(audio, sample_rate)?;
        let clarity = self.estimate_clarity(audio)?;
        let pleasantness = self.estimate_pleasantness(audio, sample_rate)?;
        let overall_quality = self.estimate_overall_quality(audio)?;
        let quality_score =
            naturalness * 0.3 + clarity * 0.25 + pleasantness * 0.25 + overall_quality * 0.2;
        let mos = 1.0 + (quality_score * 4.0);
        Ok(mos.clamp(1.0, 5.0))
    }
    /// Calculate artifact level (percentage of noticeable artifacts)
    fn calculate_artifact_level(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        let mut detector = ArtifactDetector::new();
        let artifacts = detector.detect_artifacts(audio, sample_rate)?;
        let total_samples = audio.len() as f32;
        let mut artifact_samples = 0.0;
        for location in &artifacts.artifact_locations {
            let duration = (location.end_sample - location.start_sample) as f32;
            artifact_samples += duration * location.severity;
        }
        let artifact_level = (artifact_samples / total_samples).min(1.0);
        Ok(artifact_level)
    }
    /// Calculate detailed quality metrics
    fn calculate_detailed_metrics(
        &self,
        converted: &[f32],
        original: &[f32],
        target_reference: Option<&[f32]>,
        sample_rate: u32,
    ) -> Result<DetailedQualityMetrics> {
        Ok(DetailedQualityMetrics {
            speaker_identity_preservation: if let Some(target) = target_reference {
                self.calculate_speaker_identity_preservation(converted, target, sample_rate)?
            } else {
                0.5
            },
            prosodic_preservation: self.calculate_prosodic_preservation(
                converted,
                original,
                sample_rate,
            )?,
            linguistic_preservation: self.calculate_linguistic_preservation(
                converted,
                original,
                sample_rate,
            )?,
            spectral_fidelity: self.calculate_spectral_fidelity(converted, original)?,
            temporal_consistency: self.calculate_temporal_consistency(converted)?,
            perceptual_quality: self.estimate_perceptual_quality(converted, sample_rate)?,
        })
    }
    /// Calculate overall achievement percentage
    fn calculate_overall_achievement(
        &self,
        target_sim: f32,
        source_pres: f32,
        mos: f32,
        artifact: f32,
    ) -> f32 {
        let mut score = 0.0;
        let mut total_weight = 0.0;
        let sim_achievement = (target_sim / self.config.target_similarity_threshold).min(1.0);
        score += sim_achievement * 0.3;
        total_weight += 0.3;
        let pres_achievement = (source_pres / self.config.source_preservation_threshold).min(1.0);
        score += pres_achievement * 0.3;
        total_weight += 0.3;
        let mos_achievement = (mos / self.config.mos_threshold).min(1.0);
        score += mos_achievement * 0.25;
        total_weight += 0.25;
        let artifact_achievement =
            (1.0 - (artifact / self.config.artifact_level_threshold)).clamp(0.0, 1.0);
        score += artifact_achievement * 0.15;
        total_weight += 0.15;
        score / total_weight
    }
    /// Add measurement to history
    fn add_to_history(&mut self, measurement: QualityTargetMeasurement) {
        self.measurement_history.push(measurement);
        if self.measurement_history.len() > self.max_history {
            self.measurement_history.remove(0);
        }
    }
    /// Get quality targets achievement statistics
    pub fn get_achievement_statistics(&self) -> QualityTargetsStatistics {
        if self.measurement_history.is_empty() {
            return QualityTargetsStatistics::default();
        }
        let total_measurements = self.measurement_history.len() as f32;
        let mut target_sim_met = 0;
        let mut source_pres_met = 0;
        let mut mos_met = 0;
        let mut artifact_met = 0;
        let mut total_achievement = 0.0;
        for measurement in &self.measurement_history {
            if measurement.targets_met.target_similarity_met {
                target_sim_met += 1;
            }
            if measurement.targets_met.source_preservation_met {
                source_pres_met += 1;
            }
            if measurement.targets_met.mos_threshold_met {
                mos_met += 1;
            }
            if measurement.targets_met.artifact_level_met {
                artifact_met += 1;
            }
            total_achievement += measurement.targets_met.overall_achievement;
        }
        QualityTargetsStatistics {
            total_measurements: self.measurement_history.len(),
            target_similarity_achievement_rate: target_sim_met as f32 / total_measurements,
            source_preservation_achievement_rate: source_pres_met as f32 / total_measurements,
            mos_achievement_rate: mos_met as f32 / total_measurements,
            artifact_level_achievement_rate: artifact_met as f32 / total_measurements,
            average_overall_achievement: total_achievement / total_measurements,
        }
    }
    fn calculate_spectral_similarity(&self, audio1: &[f32], audio2: &[f32]) -> Result<f32> {
        let min_len = audio1.len().min(audio2.len());
        if min_len == 0 {
            return Ok(0.0);
        }
        let mut correlation = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;
        for i in 0..min_len {
            correlation += audio1[i] * audio2[i];
            norm1 += audio1[i] * audio1[i];
            norm2 += audio2[i] * audio2[i];
        }
        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }
        Ok((correlation / (norm1.sqrt() * norm2.sqrt())).abs())
    }
    fn calculate_prosodic_similarity(
        &self,
        audio1: &[f32],
        audio2: &[f32],
        sample_rate: u32,
    ) -> Result<f32> {
        let f0_1 = self.extract_f0_contour(audio1, sample_rate)?;
        let f0_2 = self.extract_f0_contour(audio2, sample_rate)?;
        if f0_1.is_empty() || f0_2.is_empty() {
            return Ok(0.5);
        }
        self.calculate_spectral_similarity(&f0_1, &f0_2)
    }
    fn calculate_timbral_similarity(&self, audio1: &[f32], audio2: &[f32]) -> Result<f32> {
        let centroid1 = self.calculate_spectral_centroid(audio1)?;
        let centroid2 = self.calculate_spectral_centroid(audio2)?;
        let similarity = 1.0 - ((centroid1 - centroid2).abs() / (centroid1 + centroid2).max(0.001));
        Ok(similarity.clamp(0.0, 1.0))
    }
    fn calculate_linguistic_preservation(
        &self,
        converted: &[f32],
        original: &[f32],
        _sample_rate: u32,
    ) -> Result<f32> {
        self.calculate_spectral_similarity(converted, original)
    }
    fn calculate_temporal_preservation(&self, converted: &[f32], original: &[f32]) -> Result<f32> {
        let len_ratio = converted.len() as f32 / original.len() as f32;
        let preservation = 1.0 - (1.0 - len_ratio).abs();
        Ok(preservation.clamp(0.0, 1.0))
    }
    fn calculate_semantic_preservation(&self, converted: &[f32], original: &[f32]) -> Result<f32> {
        let energy1 = converted.iter().map(|x| x * x).sum::<f32>() / converted.len() as f32;
        let energy2 = original.iter().map(|x| x * x).sum::<f32>() / original.len() as f32;
        let preservation = 1.0 - ((energy1 - energy2).abs() / (energy1 + energy2).max(0.001));
        Ok(preservation.clamp(0.0, 1.0))
    }
    fn estimate_naturalness(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        let spectral_naturalness = self.estimate_spectral_naturalness(audio)?;
        let temporal_naturalness = self.estimate_temporal_naturalness(audio, sample_rate)?;
        Ok((spectral_naturalness + temporal_naturalness) / 2.0)
    }
    fn estimate_clarity(&self, audio: &[f32]) -> Result<f32> {
        let signal_power = audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32;
        let noise_estimate = self.estimate_noise_level(audio)?;
        let snr = if noise_estimate > 0.0 {
            10.0 * (signal_power / noise_estimate).log10()
        } else {
            60.0
        };
        Ok((snr / 60.0).clamp(0.0, 1.0))
    }
    fn estimate_pleasantness(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        let harmonic_score = self.calculate_harmonic_content(audio, sample_rate)?;
        let roughness_penalty = self.calculate_roughness_penalty(audio)?;
        Ok((harmonic_score * (1.0 - roughness_penalty)).clamp(0.0, 1.0))
    }
    fn estimate_overall_quality(&self, audio: &[f32]) -> Result<f32> {
        let dynamic_range = self.calculate_dynamic_range(audio)?;
        let distortion_penalty = self.calculate_distortion_penalty(audio)?;
        Ok((dynamic_range * (1.0 - distortion_penalty)).clamp(0.0, 1.0))
    }
    fn calculate_speaker_characteristics_similarity(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<f32> {
        let pitch_stats = self.calculate_pitch_statistics(audio, sample_rate)?;
        let formant_characteristics = self.estimate_formant_characteristics(audio)?;
        Ok((pitch_stats + formant_characteristics) / 2.0)
    }
    fn calculate_speaker_identity_preservation(
        &self,
        converted: &[f32],
        target: &[f32],
        sample_rate: u32,
    ) -> Result<f32> {
        self.calculate_prosodic_similarity(converted, target, sample_rate)
    }
    fn calculate_prosodic_preservation(
        &self,
        converted: &[f32],
        original: &[f32],
        sample_rate: u32,
    ) -> Result<f32> {
        self.calculate_prosodic_similarity(converted, original, sample_rate)
    }
    fn calculate_spectral_fidelity(&self, converted: &[f32], original: &[f32]) -> Result<f32> {
        self.calculate_spectral_similarity(converted, original)
    }
    fn calculate_temporal_consistency(&self, audio: &[f32]) -> Result<f32> {
        let window_size = 1024;
        let mut consistency_score = 0.0;
        let mut window_count = 0;
        for i in (0..audio.len()).step_by(window_size / 2) {
            if i + window_size >= audio.len() {
                break;
            }
            let window = &audio[i..i + window_size];
            let energy = window.iter().map(|x| x * x).sum::<f32>() / window.len() as f32;
            if window_count > 0 {
                consistency_score += (1.0 - (energy - 0.1).abs()).max(0.0);
            }
            window_count += 1;
        }
        if window_count > 1 {
            Ok(consistency_score / (window_count - 1) as f32)
        } else {
            Ok(0.5)
        }
    }
    fn estimate_perceptual_quality(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        let naturalness = self.estimate_naturalness(audio, sample_rate)?;
        let clarity = self.estimate_clarity(audio)?;
        Ok((naturalness + clarity) / 2.0)
    }
    fn extract_f0_contour(&self, audio: &[f32], _sample_rate: u32) -> Result<Vec<f32>> {
        let window_size = 1024;
        let mut f0_contour = Vec::new();
        for i in (0..audio.len()).step_by(window_size / 2) {
            if i + window_size >= audio.len() {
                break;
            }
            let window = &audio[i..i + window_size];
            let f0 = self.estimate_f0_simple(window);
            f0_contour.push(f0);
        }
        Ok(f0_contour)
    }
    fn estimate_f0_simple(&self, window: &[f32]) -> f32 {
        let mut best_lag = 0;
        let mut max_correlation = 0.0;
        for lag in 20..400 {
            if lag >= window.len() {
                break;
            }
            let mut correlation = 0.0;
            for i in 0..(window.len() - lag) {
                correlation += window[i] * window[i + lag];
            }
            if correlation > max_correlation {
                max_correlation = correlation;
                best_lag = lag;
            }
        }
        if best_lag > 0 {
            16000.0 / best_lag as f32
        } else {
            0.0
        }
    }
    fn calculate_spectral_centroid(&self, audio: &[f32]) -> Result<f32> {
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;
        for (i, &sample) in audio.iter().enumerate() {
            let magnitude = sample.abs();
            weighted_sum += (i as f32) * magnitude;
            magnitude_sum += magnitude;
        }
        if magnitude_sum > 0.0 {
            Ok(weighted_sum / magnitude_sum)
        } else {
            Ok(0.0)
        }
    }
    fn estimate_spectral_naturalness(&self, audio: &[f32]) -> Result<f32> {
        let centroid = self.calculate_spectral_centroid(audio)?;
        let normalized_centroid = (centroid / audio.len() as f32).clamp(0.0, 1.0);
        Ok(1.0 - (normalized_centroid - 0.3).abs())
    }
    fn estimate_temporal_naturalness(&self, audio: &[f32], _sample_rate: u32) -> Result<f32> {
        let mut zcr_values = Vec::new();
        let window_size = 512;
        for i in (0..audio.len()).step_by(window_size / 2) {
            if i + window_size >= audio.len() {
                break;
            }
            let window = &audio[i..i + window_size];
            let zcr = self.calculate_zero_crossing_rate(window);
            zcr_values.push(zcr);
        }
        if zcr_values.is_empty() {
            return Ok(0.5);
        }
        let mean_zcr = zcr_values.iter().sum::<f32>() / zcr_values.len() as f32;
        let variance = zcr_values
            .iter()
            .map(|x| (x - mean_zcr).powi(2))
            .sum::<f32>()
            / zcr_values.len() as f32;
        let naturalness = 1.0 - ((variance - 0.03).abs() / 0.03).min(1.0);
        Ok(naturalness)
    }
    fn calculate_zero_crossing_rate(&self, window: &[f32]) -> f32 {
        let mut crossings = 0;
        for i in 1..window.len() {
            if (window[i] >= 0.0) != (window[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 / window.len() as f32
    }
    fn estimate_noise_level(&self, audio: &[f32]) -> Result<f32> {
        let window_size = 256;
        let mut energy_values = Vec::new();
        for i in (0..audio.len()).step_by(window_size) {
            if i + window_size >= audio.len() {
                break;
            }
            let window = &audio[i..i + window_size];
            let energy = window.iter().map(|x| x * x).sum::<f32>() / window.len() as f32;
            energy_values.push(energy);
        }
        if energy_values.is_empty() {
            return Ok(0.001);
        }
        energy_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let noise_index = (energy_values.len() as f32 * 0.1) as usize;
        Ok(energy_values.get(noise_index).copied().unwrap_or(0.001))
    }
    fn calculate_harmonic_content(&self, audio: &[f32], _sample_rate: u32) -> Result<f32> {
        let mut total_energy = 0.0;
        let mut harmonic_energy = 0.0;
        let window_size = 1024;
        for i in (0..audio.len()).step_by(window_size / 2) {
            if i + window_size >= audio.len() {
                break;
            }
            let window = &audio[i..i + window_size];
            let energy = window.iter().map(|x| x * x).sum::<f32>();
            total_energy += energy;
            let mut max_autocorr: f32 = 0.0;
            for lag in 50..400 {
                if lag >= window.len() {
                    break;
                }
                let mut autocorr = 0.0;
                for j in 0..(window.len() - lag) {
                    autocorr += window[j] * window[j + lag];
                }
                max_autocorr = max_autocorr.max(autocorr);
            }
            harmonic_energy += max_autocorr.max(0.0);
        }
        if total_energy > 0.0 {
            Ok((harmonic_energy / total_energy).clamp(0.0, 1.0))
        } else {
            Ok(0.0)
        }
    }
    fn calculate_roughness_penalty(&self, audio: &[f32]) -> Result<f32> {
        let mut roughness = 0.0;
        for i in 1..audio.len() {
            let diff = (audio[i] - audio[i - 1]).abs();
            roughness += diff;
        }
        roughness /= audio.len() as f32;
        Ok((roughness * 10.0).min(1.0))
    }
    fn calculate_dynamic_range(&self, audio: &[f32]) -> Result<f32> {
        let max_val = audio.iter().fold(0.0f32, |max, &val| max.max(val.abs()));
        let min_val = audio.iter().fold(f32::INFINITY, |min, &val| {
            if val.abs() > 0.001 {
                min.min(val.abs())
            } else {
                min
            }
        });
        if min_val != f32::INFINITY && min_val > 0.0 {
            let dynamic_range_db = 20.0 * (max_val / min_val).log10();
            Ok((dynamic_range_db / 60.0).clamp(0.0, 1.0))
        } else {
            Ok(0.0)
        }
    }
    fn calculate_distortion_penalty(&self, audio: &[f32]) -> Result<f32> {
        let mut clipped_samples = 0;
        for &sample in audio {
            if sample.abs() > 0.95 {
                clipped_samples += 1;
            }
        }
        let clipping_ratio = clipped_samples as f32 / audio.len() as f32;
        Ok(clipping_ratio.min(1.0))
    }
    fn calculate_pitch_statistics(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        let f0_contour = self.extract_f0_contour(audio, sample_rate)?;
        if f0_contour.is_empty() {
            return Ok(0.5);
        }
        let mean_f0 = f0_contour.iter().sum::<f32>() / f0_contour.len() as f32;
        let variance = f0_contour
            .iter()
            .map(|x| (x - mean_f0).powi(2))
            .sum::<f32>()
            / f0_contour.len() as f32;
        let normalized_mean = (mean_f0 / 500.0).clamp(0.0, 1.0);
        let normalized_variance = (variance / 10000.0).clamp(0.0, 1.0);
        Ok((normalized_mean + normalized_variance) / 2.0)
    }
    fn estimate_formant_characteristics(&self, audio: &[f32]) -> Result<f32> {
        let window_size = 1024;
        let mut formant_score = 0.0;
        let mut window_count = 0;
        for i in (0..audio.len()).step_by(window_size / 2) {
            if i + window_size >= audio.len() {
                break;
            }
            let window = &audio[i..i + window_size];
            let mut peaks = 0;
            for j in 1..(window.len() - 1) {
                if window[j] > window[j - 1] && window[j] > window[j + 1] && window[j] > 0.1 {
                    peaks += 1;
                }
            }
            let formant_quality = if peaks >= 3 && peaks <= 8 {
                1.0 - ((peaks as f32 - 4.0).abs() / 4.0)
            } else {
                0.5
            };
            formant_score += formant_quality;
            window_count += 1;
        }
        if window_count > 0 {
            Ok(formant_score / window_count as f32)
        } else {
            Ok(0.5)
        }
    }
}
/// Complete quality target measurement result
#[derive(Debug, Clone)]
pub struct QualityTargetMeasurement {
    /// Target similarity score (0.0-1.0, target: ≥0.85)
    pub target_similarity: f32,
    /// Source preservation score (0.0-1.0, target: ≥0.90)
    pub source_preservation: f32,
    /// Mean Opinion Score (1.0-5.0, target: ≥4.0)
    pub mos_score: f32,
    /// Artifact level (0.0-1.0, target: <0.05)
    pub artifact_level: f32,
    /// Overall quality target achievement
    pub targets_met: QualityTargetsAchievement,
    /// Timestamp of measurement
    pub timestamp: std::time::SystemTime,
    /// Additional metrics for detailed analysis
    pub detailed_metrics: DetailedQualityMetrics,
}
/// Quality targets achievement status
#[derive(Debug, Clone)]
pub struct QualityTargetsAchievement {
    /// Whether target similarity threshold is met
    pub target_similarity_met: bool,
    /// Whether source preservation threshold is met
    pub source_preservation_met: bool,
    /// Whether MOS threshold is met
    pub mos_threshold_met: bool,
    /// Whether artifact level threshold is met
    pub artifact_level_met: bool,
    /// Overall achievement percentage (0.0-1.0)
    pub overall_achievement: f32,
}
/// Statistics for quality targets achievement over time
#[derive(Debug, Clone)]
pub struct QualityTargetsStatistics {
    /// Total number of measurements
    pub total_measurements: usize,
    /// Percentage of measurements meeting target similarity threshold
    pub target_similarity_achievement_rate: f32,
    /// Percentage of measurements meeting source preservation threshold
    pub source_preservation_achievement_rate: f32,
    /// Percentage of measurements meeting MOS threshold
    pub mos_achievement_rate: f32,
    /// Percentage of measurements meeting artifact level threshold
    pub artifact_level_achievement_rate: f32,
    /// Average overall achievement across all measurements
    pub average_overall_achievement: f32,
}
/// Detailed quality metrics for in-depth analysis
#[derive(Debug, Clone)]
pub struct DetailedQualityMetrics {
    /// Speaker identity preservation score
    pub speaker_identity_preservation: f32,
    /// Prosodic characteristic preservation
    pub prosodic_preservation: f32,
    /// Linguistic content preservation
    pub linguistic_preservation: f32,
    /// Spectral fidelity score
    pub spectral_fidelity: f32,
    /// Temporal consistency score
    pub temporal_consistency: f32,
    /// Perceptual quality estimate
    pub perceptual_quality: f32,
}
/// Configuration for quality target thresholds
#[derive(Debug, Clone)]
pub struct QualityTargetsConfig {
    /// Target similarity threshold (default: 0.85 for 85%+)
    pub target_similarity_threshold: f32,
    /// Source preservation threshold (default: 0.90 for 90%+)
    pub source_preservation_threshold: f32,
    /// MOS (Mean Opinion Score) threshold (default: 4.0)
    pub mos_threshold: f32,
    /// Artifact level threshold (default: 0.05 for <5%)
    pub artifact_level_threshold: f32,
    /// Enable detailed measurement tracking
    pub enable_detailed_tracking: bool,
}
