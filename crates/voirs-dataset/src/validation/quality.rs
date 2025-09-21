//! Quality analysis and validation tools
//!
//! This module provides advanced quality analysis capabilities for speech datasets,
//! including outlier detection, statistical quality reports, and recommendation generation.

use crate::{DatasetError, DatasetSample, LanguageCode, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Advanced quality analyzer for dataset samples
#[derive(Debug, Clone)]
pub struct QualityAnalyzer {
    config: QualityAnalysisConfig,
}

/// Configuration for quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAnalysisConfig {
    /// Outlier detection sensitivity (1.0 = normal, higher = more sensitive)
    pub outlier_sensitivity: f32,
    /// Minimum sample size for statistical analysis
    pub min_sample_size: usize,
    /// Whether to analyze audio spectral properties
    pub analyze_spectral_features: bool,
    /// Whether to perform deep quality assessment
    pub deep_quality_analysis: bool,
    /// Quality score thresholds for recommendations
    pub quality_thresholds: QualityThresholds,
}

/// Quality score thresholds for different categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Excellent quality threshold (>= this score)
    pub excellent: f32,
    /// Good quality threshold (>= this score)
    pub good: f32,
    /// Acceptable quality threshold (>= this score)
    pub acceptable: f32,
    /// Poor quality threshold (< this score)
    pub poor: f32,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            excellent: 0.9,
            good: 0.7,
            acceptable: 0.5,
            poor: 0.3,
        }
    }
}

impl Default for QualityAnalysisConfig {
    fn default() -> Self {
        Self {
            outlier_sensitivity: 2.0,
            min_sample_size: 10,
            analyze_spectral_features: true,
            deep_quality_analysis: true,
            quality_thresholds: QualityThresholds::default(),
        }
    }
}

/// Comprehensive quality analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAnalysisReport {
    /// Overall dataset quality assessment
    pub overall_assessment: OverallQualityAssessment,
    /// Audio quality distribution analysis
    pub audio_quality_distribution: AudioQualityDistribution,
    /// Outlier detection results
    pub outliers: OutlierAnalysis,
    /// Statistical quality metrics
    pub statistical_analysis: StatisticalQualityAnalysis,
    /// Quality improvement recommendations
    pub recommendations: QualityRecommendations,
    /// Per-language quality breakdown
    pub language_breakdown: HashMap<LanguageCode, LanguageQualityStats>,
}

/// Overall quality assessment of the dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallQualityAssessment {
    /// Overall quality grade (A, B, C, D, F)
    pub quality_grade: QualityGrade,
    /// Overall quality score (0.0-1.0)
    pub overall_score: f32,
    /// Number of samples in each quality category
    pub quality_distribution: QualityDistributionStats,
    /// Primary quality issues identified
    pub primary_issues: Vec<String>,
    /// Dataset suitability for different use cases
    pub suitability: DatasetSuitability,
}

/// Quality grade enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityGrade {
    A, // Excellent (>= 0.9)
    B, // Good (>= 0.7)
    C, // Acceptable (>= 0.5)
    D, // Poor (>= 0.3)
    F, // Unacceptable (< 0.3)
}

/// Quality distribution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDistributionStats {
    pub excellent_count: usize,
    pub good_count: usize,
    pub acceptable_count: usize,
    pub poor_count: usize,
    pub unacceptable_count: usize,
    pub total_samples: usize,
}

/// Dataset suitability for different applications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSuitability {
    /// Suitable for production TTS systems
    pub production_tts: bool,
    /// Suitable for research and development
    pub research_development: bool,
    /// Suitable for training base models
    pub base_model_training: bool,
    /// Suitable for fine-tuning
    pub fine_tuning: bool,
    /// Suitability notes and restrictions
    pub notes: Vec<String>,
}

/// Audio quality distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityDistribution {
    /// SNR distribution statistics
    pub snr_distribution: DistributionStats,
    /// Dynamic range distribution
    pub dynamic_range_distribution: DistributionStats,
    /// Clipping analysis
    pub clipping_analysis: ClippingAnalysis,
    /// Silence analysis
    pub silence_analysis: SilenceAnalysis,
    /// Spectral quality analysis
    pub spectral_analysis: Option<SpectralQualityAnalysis>,
}

/// Statistical distribution data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionStats {
    pub mean: f32,
    pub median: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub percentile_25: f32,
    pub percentile_75: f32,
    pub outlier_count: usize,
}

/// Clipping analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClippingAnalysis {
    pub samples_with_clipping: usize,
    pub average_clipping_percent: f32,
    pub max_clipping_percent: f32,
    pub severely_clipped_samples: Vec<usize>, // Indices of samples with >10% clipping
}

/// Silence analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SilenceAnalysis {
    pub silent_samples: Vec<usize>,
    pub mostly_silent_samples: Vec<usize>, // <5% energy
    pub average_silence_ratio: f32,
    pub leading_silence_stats: DistributionStats,
    pub trailing_silence_stats: DistributionStats,
}

/// Spectral quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralQualityAnalysis {
    /// Frequency response analysis
    pub frequency_response: FrequencyResponseAnalysis,
    /// Harmonic distortion analysis
    pub harmonic_distortion: HarmonicDistortionAnalysis,
    /// Noise floor analysis
    pub noise_floor: NoiseFloorAnalysis,
}

/// Frequency response analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrequencyResponseAnalysis {
    pub spectral_centroid_stats: DistributionStats,
    pub spectral_rolloff_stats: DistributionStats,
    pub bandwidth_stats: DistributionStats,
    pub energy_distribution: Vec<f32>, // Energy in frequency bands
}

/// Harmonic distortion analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HarmonicDistortionAnalysis {
    pub thd_plus_n_stats: DistributionStats,
    pub high_distortion_samples: Vec<usize>,
    pub distortion_frequency_analysis: Vec<f32>,
}

/// Noise floor analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoiseFloorAnalysis {
    pub noise_floor_stats: DistributionStats,
    pub high_noise_samples: Vec<usize>,
    pub noise_type_classification: HashMap<String, usize>,
}

/// Outlier detection analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierAnalysis {
    /// Duration outliers
    pub duration_outliers: Vec<OutlierSample>,
    /// Text length outliers
    pub text_length_outliers: Vec<OutlierSample>,
    /// Audio quality outliers
    pub quality_outliers: Vec<OutlierSample>,
    /// Multi-dimensional outliers (combination of factors)
    pub multi_dimensional_outliers: Vec<OutlierSample>,
    /// Outlier detection summary
    pub summary: OutlierSummary,
}

/// Individual outlier sample information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierSample {
    pub index: usize,
    pub id: String,
    pub outlier_score: f32,
    pub outlier_reasons: Vec<String>,
    pub actual_value: f32,
    pub expected_range: (f32, f32),
}

/// Outlier detection summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierSummary {
    pub total_outliers: usize,
    pub outlier_percentage: f32,
    pub most_common_outlier_types: Vec<(String, usize)>,
    pub severity_distribution: HashMap<String, usize>, // mild, moderate, severe
}

/// Statistical quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalQualityAnalysis {
    /// Quality correlation analysis
    pub correlations: QualityCorrelationAnalysis,
    /// Quality trend analysis
    pub trends: QualityTrendAnalysis,
    /// Sample diversity analysis
    pub diversity: DiversityAnalysis,
}

/// Quality correlation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityCorrelationAnalysis {
    pub duration_quality_correlation: f32,
    pub text_length_quality_correlation: f32,
    pub speaker_quality_consistency: HashMap<String, f32>,
    pub language_quality_differences: HashMap<LanguageCode, f32>,
}

/// Quality trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTrendAnalysis {
    pub quality_stability: f32, // How consistent quality is across samples
    pub quality_progression: Vec<f32>, // Quality changes over dataset order
    pub batch_quality_analysis: Vec<BatchQuality>,
}

/// Batch quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchQuality {
    pub batch_id: usize,
    pub sample_range: (usize, usize),
    pub average_quality: f32,
    pub quality_variance: f32,
}

/// Sample diversity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityAnalysis {
    pub phonetic_diversity: f32,
    pub duration_diversity: f32,
    pub speaker_diversity: f32,
    pub content_diversity: f32,
    pub overall_diversity_score: f32,
}

/// Quality improvement recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRecommendations {
    /// High-priority actions
    pub high_priority: Vec<Recommendation>,
    /// Medium-priority actions
    pub medium_priority: Vec<Recommendation>,
    /// Low-priority suggestions
    pub low_priority: Vec<Recommendation>,
    /// Automatic fixes that can be applied
    pub automatic_fixes: Vec<AutomaticFix>,
}

/// Individual recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub title: String,
    pub description: String,
    pub impact: String,
    pub effort: String,
    pub affected_samples: Vec<usize>,
    pub expected_improvement: f32,
}

/// Automatic fix suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomaticFix {
    pub fix_type: String,
    pub description: String,
    pub applicable_samples: Vec<usize>,
    pub confidence: f32,
    pub reversible: bool,
}

/// Per-language quality statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageQualityStats {
    pub sample_count: usize,
    pub average_quality: f32,
    pub quality_std_dev: f32,
    pub quality_distribution: QualityDistributionStats,
    pub language_specific_issues: Vec<String>,
}

impl QualityAnalyzer {
    /// Create a new quality analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: QualityAnalysisConfig::default(),
        }
    }

    /// Create a quality analyzer with custom configuration
    pub fn with_config(config: QualityAnalysisConfig) -> Self {
        Self { config }
    }

    /// Perform comprehensive quality analysis on a dataset
    pub fn analyze_dataset<T>(&self, samples: &[T]) -> Result<QualityAnalysisReport>
    where
        T: AsRef<DatasetSample>,
    {
        if samples.len() < self.config.min_sample_size {
            return Err(DatasetError::ValidationError(format!(
                "Dataset too small for analysis ({}), minimum required: {}",
                samples.len(),
                self.config.min_sample_size
            )));
        }

        // Extract quality metrics and compute derived statistics
        let quality_metrics = self.extract_quality_metrics(samples);

        // Perform overall assessment
        let overall_assessment = self.compute_overall_assessment(&quality_metrics, samples);

        // Analyze audio quality distribution
        let audio_quality_distribution = self.analyze_audio_quality_distribution(samples);

        // Detect outliers
        let outliers = self.detect_outliers(samples);

        // Perform statistical analysis
        let statistical_analysis = self.perform_statistical_analysis(samples, &quality_metrics);

        // Generate recommendations
        let recommendations = self.generate_recommendations(samples, &quality_metrics, &outliers);

        // Analyze per-language quality
        let language_breakdown = self.analyze_language_quality(samples);

        Ok(QualityAnalysisReport {
            overall_assessment,
            audio_quality_distribution,
            outliers,
            statistical_analysis,
            recommendations,
            language_breakdown,
        })
    }

    /// Extract and compute quality metrics for all samples
    fn extract_quality_metrics<T>(&self, samples: &[T]) -> Vec<ComputedQualityMetrics>
    where
        T: AsRef<DatasetSample>,
    {
        samples
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                let sample = sample.as_ref();
                self.compute_quality_metrics(sample, index)
            })
            .collect()
    }

    /// Compute comprehensive quality metrics for a single sample
    fn compute_quality_metrics(
        &self,
        sample: &DatasetSample,
        index: usize,
    ) -> ComputedQualityMetrics {
        let audio = &sample.audio;
        let samples = audio.samples();

        // Basic audio metrics
        let duration = audio.duration();
        let rms = audio.rms().unwrap_or(0.0);
        let peak = samples.iter().fold(0.0f32, |max, &s| max.max(s.abs()));

        // Dynamic range
        let dynamic_range = if rms > 0.0 {
            20.0 * (peak / rms).log10()
        } else {
            0.0
        };

        // SNR estimation (simplified)
        let signal_power = rms * rms;
        let noise_floor = self.estimate_noise_floor(samples);
        let snr = if noise_floor > 0.0 {
            10.0 * (signal_power / noise_floor).log10()
        } else {
            60.0 // High SNR for clean signals
        };

        // Clipping detection
        let clipped_samples = samples.iter().filter(|&&s| s.abs() >= 0.999).count();
        let clipping_percent = (clipped_samples as f32 / samples.len() as f32) * 100.0;

        // Spectral features (if enabled)
        let spectral_features = if self.config.analyze_spectral_features {
            Some(self.compute_spectral_features(samples, audio.sample_rate()))
        } else {
            None
        };

        // Overall quality score computation
        let overall_quality = sample.quality.overall_quality.unwrap_or_else(|| {
            self.compute_overall_quality_score(
                snr,
                dynamic_range,
                clipping_percent,
                rms,
                duration,
                &sample.text,
            )
        });

        ComputedQualityMetrics {
            index,
            duration,
            rms,
            peak,
            dynamic_range,
            snr,
            clipping_percent,
            overall_quality,
            spectral_features,
            text_length: sample.text.len(),
            speaking_rate: sample.text.len() as f32 / duration,
        }
    }

    /// Estimate noise floor from audio samples
    fn estimate_noise_floor(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // Use the lowest 10% of RMS values as noise floor estimate
        let window_size = 1024.min(samples.len() / 10);
        let mut rms_values = Vec::new();

        for chunk in samples.chunks(window_size) {
            let rms = (chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
            rms_values.push(rms);
        }

        rms_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let percentile_10 = rms_values.len() / 10;
        rms_values.get(percentile_10).copied().unwrap_or(0.0)
            * rms_values.get(percentile_10).copied().unwrap_or(0.0)
    }

    /// Compute spectral features for audio
    fn compute_spectral_features(&self, samples: &[f32], sample_rate: u32) -> SpectralFeatures {
        if samples.is_empty() {
            return SpectralFeatures {
                spectral_centroid: 0.0,
                spectral_rolloff: 0.0,
                spectral_bandwidth: 0.0,
                zero_crossing_rate: 0.0,
            };
        }

        // Compute zero crossing rate
        let zero_crossing_rate = self.compute_zero_crossing_rate(samples);

        // Compute spectral features using frame-based analysis
        let frame_size = 1024.min(samples.len());
        let hop_size = frame_size / 2;
        let mut centroids = Vec::new();
        let mut rolloffs = Vec::new();
        let mut bandwidths = Vec::new();

        for start in (0..samples.len()).step_by(hop_size) {
            let end = (start + frame_size).min(samples.len());
            if end - start < frame_size / 2 {
                break;
            }

            let frame = &samples[start..end];
            let (centroid, rolloff, bandwidth) =
                self.compute_frame_spectral_features(frame, sample_rate);

            if centroid > 0.0 {
                centroids.push(centroid);
                rolloffs.push(rolloff);
                bandwidths.push(bandwidth);
            }
        }

        // Compute mean values
        let spectral_centroid = if centroids.is_empty() {
            0.0
        } else {
            centroids.iter().sum::<f32>() / centroids.len() as f32
        };

        let spectral_rolloff = if rolloffs.is_empty() {
            0.0
        } else {
            rolloffs.iter().sum::<f32>() / rolloffs.len() as f32
        };

        let spectral_bandwidth = if bandwidths.is_empty() {
            0.0
        } else {
            bandwidths.iter().sum::<f32>() / bandwidths.len() as f32
        };

        SpectralFeatures {
            spectral_centroid,
            spectral_rolloff,
            spectral_bandwidth,
            zero_crossing_rate,
        }
    }

    /// Compute zero crossing rate
    fn compute_zero_crossing_rate(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mut zero_crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0 && samples[i - 1] < 0.0)
                || (samples[i] < 0.0 && samples[i - 1] >= 0.0)
            {
                zero_crossings += 1;
            }
        }

        zero_crossings as f32 / (samples.len() - 1) as f32
    }

    /// Compute spectral features for a single frame
    fn compute_frame_spectral_features(&self, frame: &[f32], sample_rate: u32) -> (f32, f32, f32) {
        // Simple magnitude spectrum computation
        let mut spectrum = vec![0.0; frame.len() / 2];

        // Compute magnitude spectrum using simple DFT approximation
        for (k, spec_val) in spectrum.iter_mut().enumerate() {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for (n, &sample) in frame.iter().enumerate() {
                let phase = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / frame.len() as f32;
                real_sum += sample * phase.cos();
                imag_sum += sample * phase.sin();
            }

            *spec_val = (real_sum * real_sum + imag_sum * imag_sum).sqrt();
        }

        // Compute spectral centroid
        let total_magnitude: f32 = spectrum.iter().sum();
        let spectral_centroid = if total_magnitude > 0.0 {
            let weighted_sum: f32 = spectrum
                .iter()
                .enumerate()
                .map(|(k, &magnitude)| {
                    let freq = k as f32 * sample_rate as f32 / frame.len() as f32;
                    freq * magnitude
                })
                .sum();
            weighted_sum / total_magnitude
        } else {
            0.0
        };

        // Compute spectral rolloff (85% of spectral energy)
        let total_energy: f32 = spectrum.iter().map(|x| x * x).sum();
        let mut cumulative_energy = 0.0;
        let rolloff_threshold = 0.85 * total_energy;
        let mut spectral_rolloff = 0.0;

        for (k, &magnitude) in spectrum.iter().enumerate() {
            cumulative_energy += magnitude * magnitude;
            if cumulative_energy >= rolloff_threshold {
                spectral_rolloff = k as f32 * sample_rate as f32 / frame.len() as f32;
                break;
            }
        }

        // Compute spectral bandwidth
        let spectral_bandwidth = if total_magnitude > 0.0 {
            let variance: f32 = spectrum
                .iter()
                .enumerate()
                .map(|(k, &magnitude)| {
                    let freq = k as f32 * sample_rate as f32 / frame.len() as f32;
                    (freq - spectral_centroid).powi(2) * magnitude
                })
                .sum();
            (variance / total_magnitude).sqrt()
        } else {
            0.0
        };

        (spectral_centroid, spectral_rolloff, spectral_bandwidth)
    }

    /// Compute overall quality score based on multiple factors
    fn compute_overall_quality_score(
        &self,
        snr: f32,
        dynamic_range: f32,
        clipping_percent: f32,
        rms: f32,
        duration: f32,
        text: &str,
    ) -> f32 {
        let mut score = 1.0;

        // SNR factor (0.3 weight)
        let snr_score = (snr / 40.0).clamp(0.0, 1.0);
        score *= 0.7 + 0.3 * snr_score;

        // Dynamic range factor (0.2 weight)
        let dr_score = (dynamic_range / 60.0).clamp(0.0, 1.0);
        score *= 0.8 + 0.2 * dr_score;

        // Clipping penalty (severe)
        let clipping_penalty = (clipping_percent / 10.0).min(1.0);
        score *= 1.0 - clipping_penalty;

        // RMS level check (too quiet is bad)
        if rms < 0.01 {
            score *= 0.5; // Very quiet audio
        }

        // Duration sanity check
        if !(0.1..=30.0).contains(&duration) {
            score *= 0.7;
        }

        // Text quality factor
        if text.trim().is_empty() {
            score = 0.0;
        } else {
            let speaking_rate = text.len() as f32 / duration;
            if !(1.0..=50.0).contains(&speaking_rate) {
                score *= 0.8; // Unrealistic speaking rate
            }
        }

        score.clamp(0.0, 1.0)
    }

    /// Compute overall assessment of dataset quality
    fn compute_overall_assessment<T>(
        &self,
        quality_metrics: &[ComputedQualityMetrics],
        samples: &[T],
    ) -> OverallQualityAssessment
    where
        T: AsRef<DatasetSample>,
    {
        let mut excellent = 0;
        let mut good = 0;
        let mut acceptable = 0;
        let mut poor = 0;
        let mut unacceptable = 0;

        let mut total_score = 0.0;
        let mut primary_issues = Vec::new();

        for metric in quality_metrics {
            total_score += metric.overall_quality;

            if metric.overall_quality >= self.config.quality_thresholds.excellent {
                excellent += 1;
            } else if metric.overall_quality >= self.config.quality_thresholds.good {
                good += 1;
            } else if metric.overall_quality >= self.config.quality_thresholds.acceptable {
                acceptable += 1;
            } else if metric.overall_quality >= self.config.quality_thresholds.poor {
                poor += 1;
            } else {
                unacceptable += 1;
            }

            // Identify common issues
            if metric.clipping_percent > 5.0 {
                primary_issues.push(format!(
                    "Sample {}: High clipping ({:.1}%)",
                    metric.index, metric.clipping_percent
                ));
            }
            if metric.snr < 15.0 {
                primary_issues.push(format!(
                    "Sample {}: Low SNR ({:.1}dB)",
                    metric.index, metric.snr
                ));
            }
            if metric.speaking_rate > 40.0 || metric.speaking_rate < 2.0 {
                primary_issues.push(format!(
                    "Sample {}: Unusual speaking rate ({:.1} chars/sec)",
                    metric.index, metric.speaking_rate
                ));
            }
        }

        let overall_score = total_score / quality_metrics.len() as f32;
        let quality_grade = if overall_score >= 0.9 {
            QualityGrade::A
        } else if overall_score >= 0.7 {
            QualityGrade::B
        } else if overall_score >= 0.5 {
            QualityGrade::C
        } else if overall_score >= 0.3 {
            QualityGrade::D
        } else {
            QualityGrade::F
        };

        let suitability = DatasetSuitability {
            production_tts: overall_score >= 0.8 && unacceptable == 0,
            research_development: overall_score >= 0.6,
            base_model_training: overall_score >= 0.7 && samples.len() >= 1000,
            fine_tuning: overall_score >= 0.5,
            notes: self.generate_suitability_notes(overall_score, &quality_grade, samples.len()),
        };

        OverallQualityAssessment {
            quality_grade,
            overall_score,
            quality_distribution: QualityDistributionStats {
                excellent_count: excellent,
                good_count: good,
                acceptable_count: acceptable,
                poor_count: poor,
                unacceptable_count: unacceptable,
                total_samples: samples.len(),
            },
            primary_issues,
            suitability,
        }
    }

    /// Generate suitability notes based on quality assessment
    fn generate_suitability_notes(
        &self,
        score: f32,
        grade: &QualityGrade,
        sample_count: usize,
    ) -> Vec<String> {
        let mut notes = Vec::new();

        match grade {
            QualityGrade::A => {
                notes.push("Excellent quality dataset suitable for all applications".to_string())
            }
            QualityGrade::B => {
                notes.push("Good quality dataset suitable for most applications".to_string())
            }
            QualityGrade::C => notes
                .push("Acceptable quality, may require preprocessing for best results".to_string()),
            QualityGrade::D => {
                notes.push("Poor quality, significant preprocessing required".to_string())
            }
            QualityGrade::F => {
                notes.push("Unacceptable quality, not recommended for training".to_string())
            }
        }

        if sample_count < 100 {
            notes.push(
                "Dataset size is very small, may not be sufficient for robust training".to_string(),
            );
        } else if sample_count < 1000 {
            notes.push(
                "Dataset size is small, consider augmentation or combining with other datasets"
                    .to_string(),
            );
        }

        if score < 0.5 {
            notes.push(
                "Consider applying quality filtering to remove low-quality samples".to_string(),
            );
        }

        notes
    }

    /// Analyze audio quality distribution across the dataset
    fn analyze_audio_quality_distribution<T>(&self, samples: &[T]) -> AudioQualityDistribution
    where
        T: AsRef<DatasetSample>,
    {
        let mut snr_values = Vec::new();
        let mut dynamic_range_values = Vec::new();
        let mut clipping_samples = 0;
        let mut silent_samples = Vec::new();
        let mut clipping_percentages = Vec::new();

        for (index, sample) in samples.iter().enumerate() {
            let sample = sample.as_ref();
            let audio_samples = sample.audio.samples();

            // SNR and dynamic range analysis
            let rms = sample.audio.rms().unwrap_or(0.0);
            let peak = audio_samples
                .iter()
                .fold(0.0f32, |max, &s| max.max(s.abs()));

            if rms > 0.0 {
                let noise_floor = self.estimate_noise_floor(audio_samples);
                let snr = if noise_floor > 0.0 {
                    10.0 * ((rms * rms) / noise_floor).log10()
                } else {
                    60.0
                };
                snr_values.push(snr);

                let dynamic_range = 20.0 * (peak / rms).log10();
                dynamic_range_values.push(dynamic_range);
            }

            // Clipping analysis
            let clipped_count = audio_samples.iter().filter(|&&s| s.abs() >= 0.999).count();
            let clipping_percent = (clipped_count as f32 / audio_samples.len() as f32) * 100.0;
            clipping_percentages.push(clipping_percent);

            if clipping_percent > 0.0 {
                clipping_samples += 1;
            }

            // Silence detection
            let max_amplitude = audio_samples
                .iter()
                .fold(0.0f32, |max, &s| max.max(s.abs()));
            if max_amplitude < 0.001 {
                silent_samples.push(index);
            }
        }

        let snr_distribution = self.compute_distribution_stats(&snr_values);
        let dynamic_range_distribution = self.compute_distribution_stats(&dynamic_range_values);

        let clipping_analysis = ClippingAnalysis {
            samples_with_clipping: clipping_samples,
            average_clipping_percent: clipping_percentages.iter().sum::<f32>()
                / clipping_percentages.len() as f32,
            max_clipping_percent: clipping_percentages
                .iter()
                .fold(0.0f32, |max, &x| max.max(x)),
            severely_clipped_samples: clipping_percentages
                .iter()
                .enumerate()
                .filter(|(_, &percent)| percent > 10.0)
                .map(|(index, _)| index)
                .collect(),
        };

        // Detect mostly silent samples (samples with high silence ratio but not completely silent)
        let silence_threshold = 0.01; // Amplitude threshold for silence detection
        let mostly_silent_samples: Vec<usize> = samples
            .iter()
            .enumerate()
            .filter_map(|(index, sample)| {
                let dataset_sample = sample.as_ref();
                let silence_ratio = if dataset_sample.audio.samples.is_empty() {
                    1.0
                } else {
                    let silent_count = dataset_sample
                        .audio
                        .samples
                        .iter()
                        .filter(|&&amp| amp.abs() < silence_threshold)
                        .count();
                    silent_count as f32 / dataset_sample.audio.samples.len() as f32
                };

                // Mostly silent: 70-99% silence (not completely silent, but significant silence)
                if (0.7..0.99).contains(&silence_ratio) {
                    Some(index)
                } else {
                    None
                }
            })
            .collect();

        let silence_analysis = SilenceAnalysis {
            silent_samples: silent_samples.clone(),
            mostly_silent_samples,
            average_silence_ratio: silent_samples.len() as f32 / samples.len() as f32,
            leading_silence_stats: DistributionStats::default(),
            trailing_silence_stats: DistributionStats::default(),
        };

        let spectral_analysis = if self.config.analyze_spectral_features {
            Some(SpectralQualityAnalysis {
                frequency_response: FrequencyResponseAnalysis::default(),
                harmonic_distortion: HarmonicDistortionAnalysis::default(),
                noise_floor: NoiseFloorAnalysis::default(),
            })
        } else {
            None
        };

        AudioQualityDistribution {
            snr_distribution,
            dynamic_range_distribution,
            clipping_analysis,
            silence_analysis,
            spectral_analysis,
        }
    }

    /// Compute statistical distribution for a set of values
    fn compute_distribution_stats(&self, values: &[f32]) -> DistributionStats {
        if values.is_empty() {
            return DistributionStats::default();
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let percentile_25 = sorted[sorted.len() / 4];
        let percentile_75 = sorted[3 * sorted.len() / 4];

        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();

        // Outlier detection using IQR method
        let iqr = percentile_75 - percentile_25;
        let outlier_threshold = self.config.outlier_sensitivity * iqr;
        let outlier_count = values
            .iter()
            .filter(|&&x| {
                x < percentile_25 - outlier_threshold || x > percentile_75 + outlier_threshold
            })
            .count();

        DistributionStats {
            mean,
            median,
            std_dev,
            min,
            max,
            percentile_25,
            percentile_75,
            outlier_count,
        }
    }

    /// Detect outliers in the dataset
    fn detect_outliers<T>(&self, samples: &[T]) -> OutlierAnalysis
    where
        T: AsRef<DatasetSample>,
    {
        let mut duration_outliers = Vec::new();
        let mut text_length_outliers = Vec::new();
        let mut quality_outliers = Vec::new();

        // Collect values for outlier detection
        let durations: Vec<f32> = samples
            .iter()
            .map(|s| s.as_ref().audio.duration())
            .collect();
        let text_lengths: Vec<f32> = samples
            .iter()
            .map(|s| s.as_ref().text.len() as f32)
            .collect();
        let quality_scores: Vec<f32> = samples
            .iter()
            .map(|s| s.as_ref().quality.overall_quality.unwrap_or(0.5))
            .collect();

        // Detect outliers using statistical methods
        duration_outliers.extend(self.detect_statistical_outliers(&durations, samples, "duration"));
        text_length_outliers.extend(self.detect_statistical_outliers(
            &text_lengths,
            samples,
            "text_length",
        ));
        quality_outliers.extend(self.detect_statistical_outliers(
            &quality_scores,
            samples,
            "quality",
        ));

        // Detect multi-dimensional outliers (samples outlying in multiple dimensions)
        let multi_dimensional_outliers = self.detect_multi_dimensional_outliers(
            samples,
            &duration_outliers,
            &text_length_outliers,
            &quality_outliers,
        );

        let duration_count = duration_outliers.len();
        let text_length_count = text_length_outliers.len();
        let quality_count = quality_outliers.len();
        let multi_dim_count = multi_dimensional_outliers.len();
        let total_outliers = duration_count + text_length_count + quality_count + multi_dim_count;
        let outlier_percentage = (total_outliers as f32 / samples.len() as f32) * 100.0;

        // Calculate severity distribution
        let severity_distribution = self.calculate_severity_distribution(&[
            &duration_outliers,
            &text_length_outliers,
            &quality_outliers,
            &multi_dimensional_outliers,
        ]);

        OutlierAnalysis {
            duration_outliers,
            text_length_outliers,
            quality_outliers,
            multi_dimensional_outliers,
            summary: OutlierSummary {
                total_outliers,
                outlier_percentage,
                most_common_outlier_types: vec![
                    ("duration".to_string(), duration_count),
                    ("text_length".to_string(), text_length_count),
                    ("quality".to_string(), quality_count),
                    ("multi_dimensional".to_string(), multi_dim_count),
                ],
                severity_distribution,
            },
        }
    }

    /// Detect statistical outliers for a specific metric
    fn detect_statistical_outliers<T>(
        &self,
        values: &[f32],
        samples: &[T],
        metric_name: &str,
    ) -> Vec<OutlierSample>
    where
        T: AsRef<DatasetSample>,
    {
        let stats = self.compute_distribution_stats(values);
        let iqr = stats.percentile_75 - stats.percentile_25;
        let outlier_threshold = self.config.outlier_sensitivity * iqr;
        let lower_bound = stats.percentile_25 - outlier_threshold;
        let upper_bound = stats.percentile_75 + outlier_threshold;

        values
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| {
                if value < lower_bound || value > upper_bound {
                    let sample = samples[index].as_ref();
                    let outlier_score = if value < lower_bound {
                        (lower_bound - value) / iqr
                    } else {
                        (value - upper_bound) / iqr
                    };

                    Some(OutlierSample {
                        index,
                        id: sample.id.clone(),
                        outlier_score,
                        outlier_reasons: vec![format!("{metric_name} outlier")],
                        actual_value: value,
                        expected_range: (lower_bound, upper_bound),
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Detect multi-dimensional outliers (samples that are outliers in multiple dimensions)
    fn detect_multi_dimensional_outliers<T>(
        &self,
        samples: &[T],
        duration_outliers: &[OutlierSample],
        text_length_outliers: &[OutlierSample],
        quality_outliers: &[OutlierSample],
    ) -> Vec<OutlierSample>
    where
        T: AsRef<DatasetSample>,
    {
        let mut multi_dim_outliers = Vec::new();

        // Create sets for efficient lookup
        let duration_outlier_indices: std::collections::HashSet<usize> =
            duration_outliers.iter().map(|o| o.index).collect();
        let text_length_outlier_indices: std::collections::HashSet<usize> =
            text_length_outliers.iter().map(|o| o.index).collect();
        let quality_outlier_indices: std::collections::HashSet<usize> =
            quality_outliers.iter().map(|o| o.index).collect();

        for (index, sample) in samples.iter().enumerate() {
            let mut outlier_dimensions = Vec::new();
            let mut combined_score = 0.0;
            let mut total_weight = 0.0;

            // Check if this sample is an outlier in multiple dimensions
            if duration_outlier_indices.contains(&index) {
                outlier_dimensions.push("duration".to_string());
                if let Some(outlier) = duration_outliers.iter().find(|o| o.index == index) {
                    combined_score += outlier.outlier_score * 0.33; // Weight equally
                    total_weight += 0.33;
                }
            }

            if text_length_outlier_indices.contains(&index) {
                outlier_dimensions.push("text_length".to_string());
                if let Some(outlier) = text_length_outliers.iter().find(|o| o.index == index) {
                    combined_score += outlier.outlier_score * 0.33;
                    total_weight += 0.33;
                }
            }

            if quality_outlier_indices.contains(&index) {
                outlier_dimensions.push("quality".to_string());
                if let Some(outlier) = quality_outliers.iter().find(|o| o.index == index) {
                    combined_score += outlier.outlier_score * 0.34;
                    total_weight += 0.34;
                }
            }

            // Only consider as multi-dimensional if outlier in 2+ dimensions
            if outlier_dimensions.len() >= 2 {
                let avg_score = if total_weight > 0.0 {
                    combined_score / total_weight
                } else {
                    0.0
                };

                multi_dim_outliers.push(OutlierSample {
                    index,
                    id: sample.as_ref().id.clone(),
                    outlier_score: avg_score,
                    outlier_reasons: vec![format!(
                        "Multi-dimensional outlier: {}",
                        outlier_dimensions.join(", ")
                    )],
                    actual_value: avg_score,    // Combined severity score
                    expected_range: (0.0, 1.0), // Normalized range for multi-dimensional score
                });
            }
        }

        multi_dim_outliers
    }

    /// Calculate severity distribution of outliers
    fn calculate_severity_distribution(
        &self,
        outlier_groups: &[&Vec<OutlierSample>],
    ) -> HashMap<String, usize> {
        let mut severity_distribution = HashMap::new();
        severity_distribution.insert("mild".to_string(), 0);
        severity_distribution.insert("moderate".to_string(), 0);
        severity_distribution.insert("severe".to_string(), 0);

        for outlier_group in outlier_groups {
            for outlier in outlier_group.iter() {
                let severity = if outlier.outlier_score <= 1.5 {
                    "mild"
                } else if outlier.outlier_score <= 3.0 {
                    "moderate"
                } else {
                    "severe"
                };

                if let Some(count) = severity_distribution.get_mut(severity) {
                    *count += 1;
                } else {
                    // This should never happen since we pre-insert all keys, but handle gracefully
                    severity_distribution.insert(severity.to_string(), 1);
                }
            }
        }

        severity_distribution
    }

    /// Perform statistical analysis of dataset quality
    fn perform_statistical_analysis<T>(
        &self,
        samples: &[T],
        quality_metrics: &[ComputedQualityMetrics],
    ) -> StatisticalQualityAnalysis
    where
        T: AsRef<DatasetSample>,
    {
        // Extract relevant data for correlation analysis
        let durations: Vec<f32> = samples
            .iter()
            .map(|s| s.as_ref().audio.duration())
            .collect();

        let text_lengths: Vec<f32> = samples
            .iter()
            .map(|s| s.as_ref().text.len() as f32)
            .collect();

        let quality_scores: Vec<f32> = quality_metrics.iter().map(|m| m.overall_quality).collect();

        let _snr_values: Vec<f32> = quality_metrics.iter().map(|m| m.snr).collect();

        // Compute correlations
        let duration_quality_correlation = self.compute_correlation(&durations, &quality_scores);
        let text_length_quality_correlation =
            self.compute_correlation(&text_lengths, &quality_scores);

        // Analyze speaker quality consistency
        let speaker_quality_consistency =
            self.analyze_speaker_consistency(samples, quality_metrics);

        // Analyze language quality differences
        let language_quality_differences =
            self.analyze_language_quality_differences(samples, quality_metrics);

        // Compute quality stability (coefficient of variation)
        let quality_mean = quality_scores.iter().sum::<f32>() / quality_scores.len() as f32;
        let quality_variance = quality_scores
            .iter()
            .map(|&x| (x - quality_mean).powi(2))
            .sum::<f32>()
            / quality_scores.len() as f32;
        let quality_std = quality_variance.sqrt();
        let quality_stability = if quality_mean > 0.0 {
            1.0 - (quality_std / quality_mean) // Higher value means more stable
        } else {
            0.0
        };

        // Compute diversity metrics
        let duration_diversity = self.compute_diversity(&durations);
        let text_length_diversity = self.compute_diversity(&text_lengths);
        let speaker_diversity = self.compute_speaker_diversity(samples);
        let content_diversity = self.compute_content_diversity(samples);
        let overall_diversity_score =
            (duration_diversity + text_length_diversity + speaker_diversity + content_diversity)
                / 4.0;

        // Generate batch quality analysis
        let batch_quality_analysis = self.generate_batch_quality_analysis(samples, quality_metrics);

        StatisticalQualityAnalysis {
            correlations: QualityCorrelationAnalysis {
                duration_quality_correlation,
                text_length_quality_correlation,
                speaker_quality_consistency,
                language_quality_differences,
            },
            trends: QualityTrendAnalysis {
                quality_stability,
                quality_progression: quality_scores,
                batch_quality_analysis,
            },
            diversity: DiversityAnalysis {
                phonetic_diversity: self.compute_phonetic_diversity(samples),
                duration_diversity,
                speaker_diversity,
                content_diversity,
                overall_diversity_score,
            },
        }
    }

    /// Compute Pearson correlation coefficient
    fn compute_correlation(&self, x: &[f32], y: &[f32]) -> f32 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f32;
        let sum_x = x.iter().sum::<f32>();
        let sum_y = y.iter().sum::<f32>();
        let sum_xy = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum::<f32>();
        let sum_x2 = x.iter().map(|xi| xi * xi).sum::<f32>();
        let sum_y2 = y.iter().map(|yi| yi * yi).sum::<f32>();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Analyze speaker quality consistency
    fn analyze_speaker_consistency<T>(
        &self,
        samples: &[T],
        quality_metrics: &[ComputedQualityMetrics],
    ) -> HashMap<String, f32>
    where
        T: AsRef<DatasetSample>,
    {
        let mut speaker_qualities: HashMap<String, Vec<f32>> = HashMap::new();

        for (sample, metrics) in samples.iter().zip(quality_metrics.iter()) {
            let speaker_id = sample
                .as_ref()
                .speaker_id()
                .unwrap_or("unknown")
                .to_string();
            speaker_qualities
                .entry(speaker_id)
                .or_default()
                .push(metrics.overall_quality);
        }

        let mut consistency_scores = HashMap::new();
        for (speaker_id, qualities) in speaker_qualities {
            if qualities.len() > 1 {
                let mean = qualities.iter().sum::<f32>() / qualities.len() as f32;
                let variance = qualities.iter().map(|&q| (q - mean).powi(2)).sum::<f32>()
                    / qualities.len() as f32;
                let std_dev = variance.sqrt();

                // Consistency score (higher is more consistent)
                let consistency = if mean > 0.0 {
                    1.0 - (std_dev / mean).min(1.0)
                } else {
                    0.0
                };
                consistency_scores.insert(speaker_id, consistency);
            }
        }

        consistency_scores
    }

    /// Analyze language quality differences
    fn analyze_language_quality_differences<T>(
        &self,
        samples: &[T],
        quality_metrics: &[ComputedQualityMetrics],
    ) -> HashMap<LanguageCode, f32>
    where
        T: AsRef<DatasetSample>,
    {
        let mut language_qualities: HashMap<LanguageCode, Vec<f32>> = HashMap::new();

        for (sample, metrics) in samples.iter().zip(quality_metrics.iter()) {
            let language = sample.as_ref().language;
            language_qualities
                .entry(language)
                .or_default()
                .push(metrics.overall_quality);
        }

        let mut language_scores = HashMap::new();
        for (language, qualities) in language_qualities {
            if !qualities.is_empty() {
                let mean_quality = qualities.iter().sum::<f32>() / qualities.len() as f32;
                language_scores.insert(language, mean_quality);
            }
        }

        language_scores
    }

    /// Compute diversity metric for a set of values
    fn compute_diversity(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();

        // Normalize by mean to get coefficient of variation
        if mean > 0.0 {
            (std_dev / mean).min(1.0)
        } else {
            0.0
        }
    }

    /// Compute speaker diversity
    fn compute_speaker_diversity<T>(&self, samples: &[T]) -> f32
    where
        T: AsRef<DatasetSample>,
    {
        let unique_speakers: std::collections::HashSet<String> = samples
            .iter()
            .map(|s| s.as_ref().speaker_id().unwrap_or("unknown").to_string())
            .collect();

        // Simple diversity metric: ratio of unique speakers to total samples
        unique_speakers.len() as f32 / samples.len() as f32
    }

    /// Compute content diversity
    fn compute_content_diversity<T>(&self, samples: &[T]) -> f32
    where
        T: AsRef<DatasetSample>,
    {
        let unique_words: std::collections::HashSet<String> = samples
            .iter()
            .flat_map(|s| s.as_ref().text.split_whitespace())
            .map(|word| word.to_lowercase())
            .collect();

        let total_words: usize = samples
            .iter()
            .map(|s| s.as_ref().text.split_whitespace().count())
            .sum();

        if total_words > 0 {
            (unique_words.len() as f32 / total_words as f32).min(1.0)
        } else {
            0.0
        }
    }

    /// Compute phonetic diversity
    fn compute_phonetic_diversity<T>(&self, samples: &[T]) -> f32
    where
        T: AsRef<DatasetSample>,
    {
        // Simple phonetic diversity based on character variety
        let unique_chars: std::collections::HashSet<char> = samples
            .iter()
            .flat_map(|s| s.as_ref().text.chars())
            .filter(|c| c.is_alphabetic())
            .collect();

        // Normalize by approximate phonetic alphabet size
        (unique_chars.len() as f32 / 50.0).min(1.0)
    }

    /// Generate batch quality analysis
    fn generate_batch_quality_analysis<T>(
        &self,
        samples: &[T],
        quality_metrics: &[ComputedQualityMetrics],
    ) -> Vec<BatchQuality>
    where
        T: AsRef<DatasetSample>,
    {
        let batch_size = 100.min(samples.len());
        let mut batch_stats = Vec::new();

        for (batch_idx, chunk) in samples.chunks(batch_size).enumerate() {
            let batch_quality_metrics = &quality_metrics
                [batch_idx * batch_size..((batch_idx + 1) * batch_size).min(quality_metrics.len())];

            let batch_mean_quality = batch_quality_metrics
                .iter()
                .map(|m| m.overall_quality)
                .sum::<f32>()
                / batch_quality_metrics.len() as f32;

            let _batch_mean_snr = batch_quality_metrics.iter().map(|m| m.snr).sum::<f32>()
                / batch_quality_metrics.len() as f32;

            let _batch_mean_duration = chunk
                .iter()
                .map(|s| s.as_ref().audio.duration())
                .sum::<f32>()
                / chunk.len() as f32;

            batch_stats.push(BatchQuality {
                batch_id: batch_idx,
                sample_range: (batch_idx * batch_size, (batch_idx + 1) * batch_size),
                average_quality: batch_mean_quality,
                quality_variance: batch_quality_metrics
                    .iter()
                    .map(|m| (m.overall_quality - batch_mean_quality).powi(2))
                    .sum::<f32>()
                    / batch_quality_metrics.len() as f32,
            });
        }

        batch_stats
    }

    /// Generate quality improvement recommendations
    fn generate_recommendations<T>(
        &self,
        samples: &[T],
        quality_metrics: &[ComputedQualityMetrics],
        _outliers: &OutlierAnalysis,
    ) -> QualityRecommendations
    where
        T: AsRef<DatasetSample>,
    {
        let mut high_priority = Vec::new();
        let mut medium_priority = Vec::new();
        let low_priority = Vec::new();
        let mut automatic_fixes = Vec::new();

        // Analyze quality issues
        let low_quality_count = quality_metrics
            .iter()
            .filter(|m| m.overall_quality < self.config.quality_thresholds.acceptable)
            .count();

        if low_quality_count > samples.len() / 10 {
            high_priority.push(Recommendation {
                title: "Remove low-quality samples".to_string(),
                description: format!(
                    "Remove {low_quality_count} samples with quality scores below {}",
                    self.config.quality_thresholds.acceptable
                ),
                impact: "High".to_string(),
                effort: "Low".to_string(),
                affected_samples: quality_metrics
                    .iter()
                    .filter(|m| m.overall_quality < self.config.quality_thresholds.acceptable)
                    .map(|m| m.index)
                    .collect(),
                expected_improvement: 0.2,
            });
        }

        // Check for clipping issues
        let clipped_samples: Vec<usize> = quality_metrics
            .iter()
            .filter(|m| m.clipping_percent > 5.0)
            .map(|m| m.index)
            .collect();

        if !clipped_samples.is_empty() {
            medium_priority.push(Recommendation {
                title: "Address audio clipping".to_string(),
                description: format!(
                    "Process {} samples with significant clipping",
                    clipped_samples.len()
                ),
                impact: "Medium".to_string(),
                effort: "Medium".to_string(),
                affected_samples: clipped_samples.clone(),
                expected_improvement: 0.15,
            });

            automatic_fixes.push(AutomaticFix {
                fix_type: "Clipping reduction".to_string(),
                description: "Apply soft limiting to reduce clipping artifacts".to_string(),
                applicable_samples: clipped_samples,
                confidence: 0.8,
                reversible: true,
            });
        }

        QualityRecommendations {
            high_priority,
            medium_priority,
            low_priority,
            automatic_fixes,
        }
    }

    /// Analyze quality breakdown by language
    fn analyze_language_quality<T>(
        &self,
        samples: &[T],
    ) -> HashMap<LanguageCode, LanguageQualityStats>
    where
        T: AsRef<DatasetSample>,
    {
        let mut language_stats: HashMap<LanguageCode, Vec<f32>> = HashMap::new();

        for sample in samples {
            let sample = sample.as_ref();
            let quality_score = sample.quality.overall_quality.unwrap_or(0.5);
            language_stats
                .entry(sample.language)
                .or_default()
                .push(quality_score);
        }

        language_stats
            .into_iter()
            .map(|(lang, scores)| {
                let sample_count = scores.len();
                let average_quality = scores.iter().sum::<f32>() / scores.len() as f32;
                let variance = scores
                    .iter()
                    .map(|&x| (x - average_quality).powi(2))
                    .sum::<f32>()
                    / scores.len() as f32;
                let quality_std_dev = variance.sqrt();

                let mut excellent = 0;
                let mut good = 0;
                let mut acceptable = 0;
                let mut poor = 0;
                let mut unacceptable = 0;

                for &score in &scores {
                    if score >= 0.9 {
                        excellent += 1;
                    } else if score >= 0.7 {
                        good += 1;
                    } else if score >= 0.5 {
                        acceptable += 1;
                    } else if score >= 0.3 {
                        poor += 1;
                    } else {
                        unacceptable += 1;
                    }
                }

                (
                    lang,
                    LanguageQualityStats {
                        sample_count,
                        average_quality,
                        quality_std_dev,
                        quality_distribution: QualityDistributionStats {
                            excellent_count: excellent,
                            good_count: good,
                            acceptable_count: acceptable,
                            poor_count: poor,
                            unacceptable_count: unacceptable,
                            total_samples: sample_count,
                        },
                        language_specific_issues: Vec::new(),
                    },
                )
            })
            .collect()
    }
}

/// Internal quality metrics computation
#[derive(Debug, Clone)]
struct ComputedQualityMetrics {
    index: usize,
    duration: f32,
    rms: f32,
    peak: f32,
    dynamic_range: f32,
    snr: f32,
    clipping_percent: f32,
    overall_quality: f32,
    spectral_features: Option<SpectralFeatures>,
    text_length: usize,
    speaking_rate: f32,
}

/// Spectral features for audio analysis
#[derive(Debug, Clone)]
struct SpectralFeatures {
    spectral_centroid: f32,
    spectral_rolloff: f32,
    spectral_bandwidth: f32,
    zero_crossing_rate: f32,
}

// Default implementations for various structures
impl Default for DistributionStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            percentile_25: 0.0,
            percentile_75: 0.0,
            outlier_count: 0,
        }
    }
}

impl Default for QualityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, LanguageCode};

    fn create_test_sample(
        id: &str,
        text: &str,
        duration: f32,
        quality_score: f32,
    ) -> DatasetSample {
        let sample_rate = 22050;
        let num_samples = (duration * sample_rate as f32) as usize;
        let audio = AudioData::new(vec![0.1; num_samples], sample_rate, 1);

        let mut sample =
            DatasetSample::new(id.to_string(), text.to_string(), audio, LanguageCode::EnUs);
        sample.quality.overall_quality = Some(quality_score);
        sample.quality.snr = Some(25.0);
        sample.quality.clipping = Some(0.0);
        sample.quality.dynamic_range = Some(45.0);
        sample
    }

    #[test]
    fn test_quality_analyzer_creation() {
        let analyzer = QualityAnalyzer::new();
        assert_eq!(analyzer.config.outlier_sensitivity, 2.0);
        assert_eq!(analyzer.config.min_sample_size, 10);
    }

    #[test]
    fn test_quality_metrics_computation() {
        let analyzer = QualityAnalyzer::new();
        let sample = create_test_sample("test-001", "Hello world", 2.0, 0.8);

        let metrics = analyzer.compute_quality_metrics(&sample, 0);
        assert_eq!(metrics.index, 0);
        assert!((metrics.duration - 2.0).abs() < 0.1);
        assert!(metrics.overall_quality > 0.0);
    }

    #[test]
    fn test_dataset_analysis() {
        let analyzer = QualityAnalyzer::new();
        let samples = vec![
            create_test_sample("test-001", "Hello world", 2.0, 0.9),
            create_test_sample("test-002", "Good morning", 1.5, 0.8),
            create_test_sample("test-003", "How are you?", 1.8, 0.7),
            create_test_sample("test-004", "Fine, thanks", 1.2, 0.6),
            create_test_sample("test-005", "See you later", 2.2, 0.5),
            create_test_sample("test-006", "Goodbye", 1.0, 0.4),
            create_test_sample("test-007", "Take care", 1.3, 0.3),
            create_test_sample("test-008", "Have a nice day", 2.5, 0.2),
            create_test_sample("test-009", "Talk to you soon", 2.1, 0.1),
            create_test_sample("test-010", "Until next time", 1.9, 0.8),
        ];

        let report = analyzer.analyze_dataset(&samples).unwrap();

        assert_eq!(
            report.overall_assessment.quality_distribution.total_samples,
            10
        );
        assert!(report.overall_assessment.overall_score > 0.0);
        assert!(!report.recommendations.high_priority.is_empty());
        assert_eq!(report.language_breakdown.len(), 1);
    }

    #[test]
    fn test_outlier_detection() {
        let analyzer = QualityAnalyzer::new();
        let mut samples = vec![
            create_test_sample("normal-001", "Hello world", 2.0, 0.8),
            create_test_sample("normal-002", "Good morning", 1.8, 0.8),
            create_test_sample("normal-003", "How are you?", 2.2, 0.8),
            create_test_sample(
                "outlier-001",
                "Very long text that goes on and on and on",
                0.1,
                0.8,
            ), // Duration outlier
            create_test_sample("outlier-002", "Hi", 10.0, 0.8), // Another duration outlier
        ];

        // Add more samples to meet minimum requirement
        for i in 5..15 {
            samples.push(create_test_sample(
                &format!("normal-{i:03}"),
                "Normal text",
                2.0,
                0.8,
            ));
        }

        let outliers = analyzer.detect_outliers(&samples);
        assert!(!outliers.duration_outliers.is_empty());
        assert!(outliers.summary.total_outliers > 0);
        assert!(outliers.summary.outlier_percentage > 0.0);
    }

    #[test]
    fn test_distribution_stats() {
        let analyzer = QualityAnalyzer::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let stats = analyzer.compute_distribution_stats(&values);
        assert!((stats.mean - 5.5).abs() < 0.1);
        assert!((stats.median - 5.5).abs() < 0.1);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
    }

    #[test]
    fn test_overall_quality_score_computation() {
        let analyzer = QualityAnalyzer::new();

        // Test high quality sample
        let high_quality_score = analyzer.compute_overall_quality_score(
            30.0,          // good SNR
            50.0,          // good dynamic range
            0.0,           // no clipping
            0.1,           // good RMS
            2.0,           // reasonable duration
            "Hello world", // reasonable text
        );
        assert!(high_quality_score > 0.8);

        // Test low quality sample
        let low_quality_score = analyzer.compute_overall_quality_score(
            5.0,   // poor SNR
            20.0,  // poor dynamic range
            15.0,  // high clipping
            0.001, // very low RMS
            0.05,  // too short
            "",    // empty text
        );
        assert!(low_quality_score < 0.3);
    }
}
