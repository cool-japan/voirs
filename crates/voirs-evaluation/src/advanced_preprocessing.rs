//! Advanced Audio Preprocessing for Evaluation Enhancement
//!
//! This module provides sophisticated audio preprocessing capabilities to improve
//! evaluation reliability and handle challenging audio conditions.

use crate::{error_enhancement, precision::precise_mean, EvaluationError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use voirs_sdk::AudioBuffer;

/// Advanced preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedPreprocessingConfig {
    /// Enable automatic gain control
    pub auto_gain_control: bool,
    /// Target RMS level for normalization
    pub target_rms_level: f32,
    /// Enable noise reduction
    pub noise_reduction: bool,
    /// Noise reduction strength (0.0-1.0)
    pub noise_reduction_strength: f32,
    /// Enable click and pop removal
    pub remove_clicks_pops: bool,
    /// Enable DC offset correction
    pub dc_offset_correction: bool,
    /// Enable spectral gating for silence detection
    pub spectral_gating: bool,
    /// Minimum signal threshold for processing
    pub min_signal_threshold_db: f32,
    /// Enable adaptive filtering
    pub adaptive_filtering: bool,
    /// Sample rate for processing (None = keep original)
    pub target_sample_rate: Option<u32>,
}

impl Default for AdvancedPreprocessingConfig {
    fn default() -> Self {
        Self {
            auto_gain_control: true,
            target_rms_level: -23.0, // LUFS standard
            noise_reduction: true,
            noise_reduction_strength: 0.3,
            remove_clicks_pops: true,
            dc_offset_correction: true,
            spectral_gating: true,
            min_signal_threshold_db: -60.0,
            adaptive_filtering: false,
            target_sample_rate: None,
        }
    }
}

/// Audio quality assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityAssessment {
    /// Overall quality score (0.0-1.0)
    pub overall_quality: f32,
    /// Individual quality metrics
    pub quality_metrics: HashMap<String, f32>,
    /// Detected issues
    pub detected_issues: Vec<AudioIssue>,
    /// Preprocessing recommendations
    pub preprocessing_recommendations: Vec<PreprocessingRecommendation>,
    /// Processing statistics
    pub processing_stats: ProcessingStatistics,
}

/// Detected audio issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioIssue {
    /// Issue type
    pub issue_type: AudioIssueType,
    /// Severity (0.0-1.0, 1.0 being most severe)
    pub severity: f32,
    /// Description of the issue
    pub description: String,
    /// Start time in seconds (if applicable)
    pub start_time_sec: Option<f32>,
    /// Duration in seconds (if applicable)
    pub duration_sec: Option<f32>,
    /// Confidence in detection (0.0-1.0)
    pub confidence: f32,
}

/// Types of audio issues that can be detected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AudioIssueType {
    /// Clipping distortion
    Clipping,
    /// Excessive noise
    Noise,
    /// Click or pop artifacts
    ClickPop,
    /// DC offset
    DcOffset,
    /// Low signal level
    LowLevel,
    /// High dynamic range variation
    DynamicRange,
    /// Frequency imbalance
    FrequencyImbalance,
    /// Phase issues (stereo)
    PhaseIssue,
    /// Dropout or silence
    Dropout,
    /// Aliasing artifacts
    Aliasing,
}

/// Preprocessing recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingRecommendation {
    /// Recommendation type
    pub recommendation_type: PreprocessingType,
    /// Priority (1-10, 10 being highest)
    pub priority: u8,
    /// Description
    pub description: String,
    /// Expected improvement
    pub expected_improvement: String,
    /// Parameters for the preprocessing
    pub parameters: HashMap<String, f32>,
}

/// Types of preprocessing operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PreprocessingType {
    /// Gain adjustment
    GainAdjustment,
    /// Noise reduction
    NoiseReduction,
    /// Click removal
    ClickRemoval,
    /// DC offset correction
    DcCorrection,
    /// Equalization
    Equalization,
    /// Dynamic range compression
    Compression,
    /// High-pass filtering
    HighPassFilter,
    /// Low-pass filtering
    LowPassFilter,
    /// Resampling
    Resampling,
}

/// Processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStatistics {
    /// Original audio duration (seconds)
    pub original_duration_sec: f32,
    /// Processed audio duration (seconds)
    pub processed_duration_sec: f32,
    /// Original sample rate
    pub original_sample_rate: u32,
    /// Processed sample rate
    pub processed_sample_rate: u32,
    /// Processing time (milliseconds)
    pub processing_time_ms: u64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: Option<u64>,
    /// Operations applied
    pub operations_applied: Vec<String>,
}

/// Advanced audio preprocessor
pub struct AdvancedPreprocessor {
    config: AdvancedPreprocessingConfig,
}

impl AdvancedPreprocessor {
    /// Create a new advanced preprocessor
    pub fn new(config: AdvancedPreprocessingConfig) -> Self {
        Self { config }
    }

    /// Perform comprehensive audio analysis and preprocessing
    pub async fn process_audio(
        &self,
        audio: &AudioBuffer,
    ) -> Result<(AudioBuffer, AudioQualityAssessment), EvaluationError> {
        let start_time = std::time::Instant::now();
        let mut processed_audio = audio.clone();
        let mut operations_applied = Vec::new();
        let mut detected_issues = Vec::new();
        let mut recommendations = Vec::new();

        // Step 1: Analyze audio quality
        let initial_assessment = self.analyze_audio_quality(audio)?;
        detected_issues.extend(initial_assessment.detected_issues.clone());

        // Step 2: Apply preprocessing based on configuration and detected issues
        if self.config.dc_offset_correction {
            processed_audio = self.correct_dc_offset(&processed_audio)?;
            operations_applied.push("DC Offset Correction".to_string());
        }

        if self.config.remove_clicks_pops
            && initial_assessment
                .detected_issues
                .iter()
                .any(|issue| issue.issue_type == AudioIssueType::ClickPop)
        {
            processed_audio = self.remove_clicks_pops(&processed_audio)?;
            operations_applied.push("Click/Pop Removal".to_string());
        }

        if self.config.noise_reduction
            && initial_assessment
                .detected_issues
                .iter()
                .any(|issue| issue.issue_type == AudioIssueType::Noise)
        {
            processed_audio =
                self.apply_noise_reduction(&processed_audio, self.config.noise_reduction_strength)?;
            operations_applied.push("Noise Reduction".to_string());
        }

        if self.config.auto_gain_control {
            processed_audio =
                self.apply_auto_gain_control(&processed_audio, self.config.target_rms_level)?;
            operations_applied.push("Auto Gain Control".to_string());
        }

        if let Some(target_sr) = self.config.target_sample_rate {
            if target_sr != processed_audio.sample_rate() {
                processed_audio = self.resample_audio(&processed_audio, target_sr)?;
                operations_applied.push(format!("Resampling to {}Hz", target_sr));
            }
        }

        // Step 3: Generate recommendations for remaining issues
        recommendations =
            self.generate_preprocessing_recommendations(&initial_assessment.detected_issues);

        // Step 4: Calculate final quality metrics
        let final_quality_metrics = self.calculate_quality_metrics(&processed_audio)?;

        let processing_time_ms = start_time.elapsed().as_millis() as u64;

        let processing_stats = ProcessingStatistics {
            original_duration_sec: audio.samples().len() as f32 / audio.sample_rate() as f32,
            processed_duration_sec: processed_audio.samples().len() as f32
                / processed_audio.sample_rate() as f32,
            original_sample_rate: audio.sample_rate(),
            processed_sample_rate: processed_audio.sample_rate(),
            processing_time_ms,
            memory_usage_bytes: None, // Could be implemented with memory monitoring
            operations_applied,
        };

        let assessment = AudioQualityAssessment {
            overall_quality: final_quality_metrics
                .get("overall_quality")
                .copied()
                .unwrap_or(0.5),
            quality_metrics: final_quality_metrics,
            detected_issues: detected_issues.clone(),
            preprocessing_recommendations: recommendations,
            processing_stats,
        };

        Ok((processed_audio, assessment))
    }

    /// Analyze audio quality and detect issues
    fn analyze_audio_quality(
        &self,
        audio: &AudioBuffer,
    ) -> Result<AudioQualityAssessment, EvaluationError> {
        let mut detected_issues = Vec::new();
        let mut quality_metrics = HashMap::new();

        let samples = audio.samples();
        let sample_rate = audio.sample_rate() as f32;

        // Calculate basic statistics
        let rms_level = Self::calculate_rms(samples);
        let peak_level = samples
            .iter()
            .fold(0.0f32, |acc, &sample| acc.max(sample.abs()));
        let dc_offset = precise_mean(samples) as f32;

        quality_metrics.insert("rms_level".to_string(), rms_level);
        quality_metrics.insert("peak_level".to_string(), peak_level);
        quality_metrics.insert("dc_offset".to_string(), dc_offset.abs());

        // Detect clipping
        let clipping_count = samples.iter().filter(|&&s| s.abs() >= 0.99).count();
        let clipping_percentage = clipping_count as f32 / samples.len() as f32;
        quality_metrics.insert("clipping_percentage".to_string(), clipping_percentage);

        if clipping_percentage > 0.01 {
            // More than 1% clipped
            detected_issues.push(AudioIssue {
                issue_type: AudioIssueType::Clipping,
                severity: (clipping_percentage * 10.0).min(1.0),
                description: format!(
                    "Audio clipping detected: {:.2}% of samples",
                    clipping_percentage * 100.0
                ),
                start_time_sec: None,
                duration_sec: None,
                confidence: 0.95,
            });
        }

        // Detect DC offset
        if dc_offset.abs() > 0.01 {
            detected_issues.push(AudioIssue {
                issue_type: AudioIssueType::DcOffset,
                severity: (dc_offset.abs() * 10.0).min(1.0),
                description: format!("DC offset detected: {:.4}", dc_offset),
                start_time_sec: None,
                duration_sec: None,
                confidence: 0.9,
            });
        }

        // Detect low signal level
        let rms_db = 20.0 * rms_level.log10();
        quality_metrics.insert("rms_db".to_string(), rms_db);

        if rms_db < -40.0 {
            detected_issues.push(AudioIssue {
                issue_type: AudioIssueType::LowLevel,
                severity: ((-40.0 - rms_db) / 20.0).min(1.0),
                description: format!("Low signal level: {:.1} dB RMS", rms_db),
                start_time_sec: None,
                duration_sec: None,
                confidence: 0.85,
            });
        }

        // Detect noise (simplified - high-frequency content analysis)
        let noise_estimate = Self::estimate_noise_level(samples, sample_rate);
        quality_metrics.insert("noise_estimate".to_string(), noise_estimate);

        if noise_estimate > 0.05 {
            detected_issues.push(AudioIssue {
                issue_type: AudioIssueType::Noise,
                severity: (noise_estimate * 20.0).min(1.0),
                description: format!("High noise level detected: {:.3}", noise_estimate),
                start_time_sec: None,
                duration_sec: None,
                confidence: 0.7,
            });
        }

        // Calculate overall quality (simplified)
        let overall_quality = Self::calculate_overall_quality(&quality_metrics, &detected_issues);
        quality_metrics.insert("overall_quality".to_string(), overall_quality);

        Ok(AudioQualityAssessment {
            overall_quality,
            quality_metrics,
            detected_issues,
            preprocessing_recommendations: Vec::new(),
            processing_stats: ProcessingStatistics {
                original_duration_sec: samples.len() as f32 / sample_rate,
                processed_duration_sec: samples.len() as f32 / sample_rate,
                original_sample_rate: audio.sample_rate(),
                processed_sample_rate: audio.sample_rate(),
                processing_time_ms: 0,
                memory_usage_bytes: None,
                operations_applied: Vec::new(),
            },
        })
    }

    /// Generate preprocessing recommendations based on detected issues
    fn generate_preprocessing_recommendations(
        &self,
        issues: &[AudioIssue],
    ) -> Vec<PreprocessingRecommendation> {
        let mut recommendations = Vec::new();

        for issue in issues {
            match issue.issue_type {
                AudioIssueType::Clipping => {
                    recommendations.push(PreprocessingRecommendation {
                        recommendation_type: PreprocessingType::GainAdjustment,
                        priority: 9,
                        description: "Apply gain reduction to prevent clipping".to_string(),
                        expected_improvement: "Eliminate clipping distortion".to_string(),
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("gain_db".to_string(), -6.0);
                            params
                        },
                    });
                }
                AudioIssueType::Noise => {
                    recommendations.push(PreprocessingRecommendation {
                        recommendation_type: PreprocessingType::NoiseReduction,
                        priority: 7,
                        description: "Apply spectral noise reduction".to_string(),
                        expected_improvement: "Reduce background noise by 10-15 dB".to_string(),
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("strength".to_string(), 0.4);
                            params
                        },
                    });
                }
                AudioIssueType::DcOffset => {
                    recommendations.push(PreprocessingRecommendation {
                        recommendation_type: PreprocessingType::DcCorrection,
                        priority: 6,
                        description: "Remove DC offset bias".to_string(),
                        expected_improvement: "Eliminate DC bias for better processing".to_string(),
                        parameters: HashMap::new(),
                    });
                }
                AudioIssueType::LowLevel => {
                    recommendations.push(PreprocessingRecommendation {
                        recommendation_type: PreprocessingType::GainAdjustment,
                        priority: 5,
                        description: "Increase signal level to optimal range".to_string(),
                        expected_improvement: "Improve signal-to-noise ratio".to_string(),
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("target_rms_db".to_string(), -23.0);
                            params
                        },
                    });
                }
                AudioIssueType::ClickPop => {
                    recommendations.push(PreprocessingRecommendation {
                        recommendation_type: PreprocessingType::ClickRemoval,
                        priority: 8,
                        description: "Remove click and pop artifacts".to_string(),
                        expected_improvement: "Eliminate impulsive noise artifacts".to_string(),
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("threshold".to_string(), 0.8);
                            params
                        },
                    });
                }
                _ => {} // Other issue types can be added later
            }
        }

        // Sort by priority (highest first)
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));
        recommendations
    }

    /// Correct DC offset in audio
    fn correct_dc_offset(&self, audio: &AudioBuffer) -> Result<AudioBuffer, EvaluationError> {
        let samples = audio.samples();
        let dc_offset = precise_mean(samples) as f32;

        let corrected_samples: Vec<f32> =
            samples.iter().map(|&sample| sample - dc_offset).collect();

        Ok(AudioBuffer::new(
            corrected_samples,
            audio.sample_rate(),
            audio.channels(),
        ))
    }

    /// Remove clicks and pops from audio
    fn remove_clicks_pops(&self, audio: &AudioBuffer) -> Result<AudioBuffer, EvaluationError> {
        let samples = audio.samples();
        let mut processed_samples = samples.to_vec();

        // Simple click detection and removal based on sudden amplitude changes
        let threshold = 0.8; // Threshold for click detection

        for i in 1..processed_samples.len() - 1 {
            let current = processed_samples[i];
            let prev = processed_samples[i - 1];
            let next = processed_samples[i + 1];

            // Detect sudden amplitude spike
            if current.abs() > threshold
                && (current - prev).abs() > 0.5
                && (current - next).abs() > 0.5
            {
                // Replace with interpolated value
                processed_samples[i] = (prev + next) / 2.0;
            }
        }

        Ok(AudioBuffer::new(
            processed_samples,
            audio.sample_rate(),
            audio.channels(),
        ))
    }

    /// Apply noise reduction to audio
    fn apply_noise_reduction(
        &self,
        audio: &AudioBuffer,
        strength: f32,
    ) -> Result<AudioBuffer, EvaluationError> {
        let samples = audio.samples();

        // Simplified spectral subtraction-based noise reduction
        // In practice, this would use proper FFT and spectral processing
        let noise_floor = Self::estimate_noise_level(samples, audio.sample_rate() as f32);
        let reduction_factor = 1.0 - (strength * noise_floor).min(0.5);

        let processed_samples: Vec<f32> = samples
            .iter()
            .map(|&sample| {
                if sample.abs() < noise_floor * 2.0 {
                    sample * reduction_factor
                } else {
                    sample
                }
            })
            .collect();

        Ok(AudioBuffer::new(
            processed_samples,
            audio.sample_rate(),
            audio.channels(),
        ))
    }

    /// Apply automatic gain control
    fn apply_auto_gain_control(
        &self,
        audio: &AudioBuffer,
        target_rms_db: f32,
    ) -> Result<AudioBuffer, EvaluationError> {
        let samples = audio.samples();
        let current_rms = Self::calculate_rms(samples);
        let current_rms_db = 20.0 * current_rms.log10();

        let gain_db = target_rms_db - current_rms_db;
        let gain_linear = 10.0_f32.powf(gain_db / 20.0);

        let processed_samples: Vec<f32> =
            samples.iter().map(|&sample| sample * gain_linear).collect();

        Ok(AudioBuffer::new(
            processed_samples,
            audio.sample_rate(),
            audio.channels(),
        ))
    }

    /// Resample audio to target sample rate
    fn resample_audio(
        &self,
        audio: &AudioBuffer,
        target_sample_rate: u32,
    ) -> Result<AudioBuffer, EvaluationError> {
        if audio.sample_rate() == target_sample_rate {
            return Ok(audio.clone());
        }

        // Simple linear interpolation resampling
        let samples = audio.samples();
        let ratio = target_sample_rate as f32 / audio.sample_rate() as f32;
        let new_length = (samples.len() as f32 * ratio) as usize;

        let mut resampled = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let original_index = i as f32 / ratio;
            let index_floor = original_index.floor() as usize;
            let index_ceil = (index_floor + 1).min(samples.len() - 1);
            let fraction = original_index - index_floor as f32;

            let sample = if index_floor >= samples.len() {
                0.0
            } else if index_floor == index_ceil {
                samples[index_floor]
            } else {
                samples[index_floor] * (1.0 - fraction) + samples[index_ceil] * fraction
            };

            resampled.push(sample);
        }

        Ok(AudioBuffer::new(
            resampled,
            target_sample_rate,
            audio.channels(),
        ))
    }

    /// Calculate quality metrics for processed audio
    fn calculate_quality_metrics(
        &self,
        audio: &AudioBuffer,
    ) -> Result<HashMap<String, f32>, EvaluationError> {
        let mut metrics = HashMap::new();

        let samples = audio.samples();
        let rms_level = Self::calculate_rms(samples);
        let peak_level = samples
            .iter()
            .fold(0.0f32, |acc, &sample| acc.max(sample.abs()));
        let dynamic_range = peak_level / rms_level.max(1e-10);

        metrics.insert("rms_level".to_string(), rms_level);
        metrics.insert("peak_level".to_string(), peak_level);
        metrics.insert("dynamic_range".to_string(), dynamic_range);
        metrics.insert("rms_db".to_string(), 20.0 * rms_level.log10());

        // Calculate THD+N (simplified)
        let thd_n = Self::estimate_thd_n(samples);
        metrics.insert("thd_n".to_string(), thd_n);

        // Overall quality score based on multiple factors
        let overall_quality = Self::calculate_overall_quality(&metrics, &[]);
        metrics.insert("overall_quality".to_string(), overall_quality);

        Ok(metrics)
    }

    /// Calculate RMS level
    fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
        (sum_squares / samples.len() as f64).sqrt() as f32
    }

    /// Estimate noise level (simplified)
    fn estimate_noise_level(samples: &[f32], _sample_rate: f32) -> f32 {
        // Simple noise estimation using percentile of absolute values
        let mut abs_samples: Vec<f32> = samples.iter().map(|&s| s.abs()).collect();
        abs_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use 20th percentile as noise estimate
        let index = (abs_samples.len() as f32 * 0.2) as usize;
        abs_samples.get(index).copied().unwrap_or(0.0)
    }

    /// Estimate THD+N (simplified)
    fn estimate_thd_n(samples: &[f32]) -> f32 {
        // Simplified THD+N estimation
        // In practice, this would use proper FFT and harmonic analysis
        let rms = Self::calculate_rms(samples);
        let noise_estimate = Self::estimate_noise_level(samples, 1.0);

        (noise_estimate / rms.max(1e-10)).min(1.0)
    }

    /// Calculate overall quality score
    fn calculate_overall_quality(metrics: &HashMap<String, f32>, issues: &[AudioIssue]) -> f32 {
        let mut quality = 1.0;

        // Penalize based on detected issues
        for issue in issues {
            quality -= issue.severity * 0.3;
        }

        // Adjust based on metrics
        if let Some(rms_db) = metrics.get("rms_db") {
            if *rms_db < -40.0 {
                quality -= ((-40.0 - rms_db) / 40.0) * 0.2;
            }
        }

        if let Some(thd_n) = metrics.get("thd_n") {
            quality -= thd_n * 0.1;
        }

        quality.max(0.0).min(1.0)
    }

    /// Create a processing report
    pub fn create_processing_report(&self, assessment: &AudioQualityAssessment) -> String {
        let mut report = String::new();

        report.push_str("# Advanced Audio Preprocessing Report\n\n");

        // Quality Summary
        report.push_str("## Quality Assessment\n\n");
        report.push_str(&format!(
            "**Overall Quality:** {:.2}/1.0\n\n",
            assessment.overall_quality
        ));

        // Quality Metrics
        if !assessment.quality_metrics.is_empty() {
            report.push_str("**Quality Metrics:**\n");
            for (metric, value) in &assessment.quality_metrics {
                report.push_str(&format!("- {}: {:.4}\n", metric, value));
            }
            report.push_str("\n");
        }

        // Detected Issues
        if !assessment.detected_issues.is_empty() {
            report.push_str("## Detected Issues\n\n");
            for issue in &assessment.detected_issues {
                let severity_icon = match issue.severity {
                    s if s >= 0.8 => "🔴",
                    s if s >= 0.5 => "🟡",
                    _ => "🟢",
                };

                report.push_str(&format!(
                    "- {} **{:?}** (Severity: {:.2}): {}\n",
                    severity_icon, issue.issue_type, issue.severity, issue.description
                ));
            }
            report.push_str("\n");
        }

        // Preprocessing Recommendations
        if !assessment.preprocessing_recommendations.is_empty() {
            report.push_str("## Preprocessing Recommendations\n\n");
            for (i, rec) in assessment.preprocessing_recommendations.iter().enumerate() {
                let priority_icon = match rec.priority {
                    8..=10 => "🔴",
                    5..=7 => "🟡",
                    _ => "🟢",
                };

                report.push_str(&format!(
                    "{}. {} **{:?}** (Priority: {})\n",
                    i + 1,
                    priority_icon,
                    rec.recommendation_type,
                    rec.priority
                ));
                report.push_str(&format!("   - {}\n", rec.description));
                report.push_str(&format!("   - Expected: {}\n\n", rec.expected_improvement));
            }
        }

        // Processing Statistics
        report.push_str("## Processing Statistics\n\n");
        let stats = &assessment.processing_stats;
        report.push_str(&format!(
            "- Processing Time: {}ms\n",
            stats.processing_time_ms
        ));
        report.push_str(&format!(
            "- Operations Applied: {}\n",
            stats.operations_applied.len()
        ));
        if !stats.operations_applied.is_empty() {
            for op in &stats.operations_applied {
                report.push_str(&format!("  - {}\n", op));
            }
        }

        report
    }
}

impl Default for AdvancedPreprocessor {
    fn default() -> Self {
        Self::new(AdvancedPreprocessingConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[tokio::test]
    async fn test_advanced_preprocessor_creation() {
        let config = AdvancedPreprocessingConfig::default();
        let preprocessor = AdvancedPreprocessor::new(config);
        assert!(preprocessor.config.auto_gain_control);
    }

    #[tokio::test]
    async fn test_dc_offset_correction() {
        let config = AdvancedPreprocessingConfig {
            dc_offset_correction: true,
            auto_gain_control: false,
            noise_reduction: false,
            remove_clicks_pops: false,
            ..Default::default()
        };
        let preprocessor = AdvancedPreprocessor::new(config);

        // Create audio with DC offset
        let samples: Vec<f32> = (0..1000)
            .map(|i| {
                let t = i as f32 / 1000.0;
                0.5 * (2.0 * PI * 440.0 * t).sin() + 0.1 // Add DC offset
            })
            .collect();

        let audio = AudioBuffer::new(samples, 16000, 1);
        let (processed, _) = preprocessor.process_audio(&audio).await.unwrap();

        let dc_after = precise_mean(processed.samples()) as f32;
        assert!(dc_after.abs() < 0.01); // DC should be mostly removed
    }

    #[tokio::test]
    async fn test_quality_analysis() {
        let preprocessor = AdvancedPreprocessor::default();

        // Create test audio with known issues
        let mut samples = vec![0.1; 1000];
        // Add enough clipping samples to exceed 1% threshold
        for i in 500..520 {
            if i % 2 == 0 {
                samples[i] = 0.995; // Add a clip (above 0.99 threshold)
            } else {
                samples[i] = -0.995; // Add another clip
            }
        }

        let audio = AudioBuffer::new(samples, 16000, 1);
        let assessment = preprocessor.analyze_audio_quality(&audio).unwrap();

        assert!(!assessment.detected_issues.is_empty());
        assert!(assessment
            .detected_issues
            .iter()
            .any(|issue| issue.issue_type == AudioIssueType::Clipping));
    }

    #[tokio::test]
    async fn test_noise_reduction() {
        let config = AdvancedPreprocessingConfig {
            noise_reduction: true,
            noise_reduction_strength: 0.5,
            auto_gain_control: false,
            dc_offset_correction: false,
            remove_clicks_pops: false,
            ..Default::default()
        };
        let preprocessor = AdvancedPreprocessor::new(config);

        // Create noisy audio
        let samples: Vec<f32> = (0..1000)
            .map(|i| {
                let t = i as f32 / 1000.0;
                let signal = 0.5 * (2.0 * PI * 440.0 * t).sin();
                let noise = (rand::random::<f32>() - 0.5) * 0.1;
                signal + noise
            })
            .collect();

        let audio = AudioBuffer::new(samples, 16000, 1);
        let (processed, assessment) = preprocessor.process_audio(&audio).await.unwrap();

        assert_eq!(processed.samples().len(), audio.samples().len());
        assert!(assessment
            .processing_stats
            .operations_applied
            .contains(&"Noise Reduction".to_string()));
    }

    #[tokio::test]
    async fn test_auto_gain_control() {
        let config = AdvancedPreprocessingConfig {
            auto_gain_control: true,
            target_rms_level: -20.0,
            noise_reduction: false,
            dc_offset_correction: false,
            remove_clicks_pops: false,
            ..Default::default()
        };
        let preprocessor = AdvancedPreprocessor::new(config);

        // Create low-level audio
        let samples: Vec<f32> = (0..1000)
            .map(|i| {
                let t = i as f32 / 1000.0;
                0.01 * (2.0 * PI * 440.0 * t).sin() // Very low level
            })
            .collect();

        let audio = AudioBuffer::new(samples, 16000, 1);
        let (processed, assessment) = preprocessor.process_audio(&audio).await.unwrap();

        let processed_rms = AdvancedPreprocessor::calculate_rms(processed.samples());
        let processed_rms_db = 20.0 * processed_rms.log10();

        // Should be closer to target level
        assert!(processed_rms_db > -25.0); // Should be increased significantly
        assert!(assessment
            .processing_stats
            .operations_applied
            .contains(&"Auto Gain Control".to_string()));
    }

    #[tokio::test]
    async fn test_resampling() {
        let config = AdvancedPreprocessingConfig {
            target_sample_rate: Some(22050),
            auto_gain_control: false,
            noise_reduction: false,
            dc_offset_correction: false,
            remove_clicks_pops: false,
            ..Default::default()
        };
        let preprocessor = AdvancedPreprocessor::new(config);

        let samples = vec![0.1; 16000]; // 1 second at 16kHz
        let audio = AudioBuffer::new(samples, 16000, 1);
        let (processed, assessment) = preprocessor.process_audio(&audio).await.unwrap();

        assert_eq!(processed.sample_rate(), 22050);
        assert!(assessment
            .processing_stats
            .operations_applied
            .iter()
            .any(|op| op.contains("Resampling")));
    }

    #[test]
    fn test_quality_metrics_calculation() {
        let preprocessor = AdvancedPreprocessor::default();

        let samples = vec![0.1, 0.2, -0.1, -0.2, 0.15];
        let audio = AudioBuffer::new(samples, 16000, 1);
        let metrics = preprocessor.calculate_quality_metrics(&audio).unwrap();

        assert!(metrics.contains_key("rms_level"));
        assert!(metrics.contains_key("peak_level"));
        assert!(metrics.contains_key("overall_quality"));
    }

    #[test]
    fn test_rms_calculation() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        let rms = AdvancedPreprocessor::calculate_rms(&samples);
        assert!((rms - 1.0).abs() < 0.001);

        let zero_samples = vec![0.0; 100];
        let zero_rms = AdvancedPreprocessor::calculate_rms(&zero_samples);
        assert_eq!(zero_rms, 0.0);
    }

    #[test]
    fn test_preprocessing_recommendations() {
        let preprocessor = AdvancedPreprocessor::default();

        let issues = vec![AudioIssue {
            issue_type: AudioIssueType::Clipping,
            severity: 0.8,
            description: "Test clipping".to_string(),
            start_time_sec: None,
            duration_sec: None,
            confidence: 0.9,
        }];

        let recommendations = preprocessor.generate_preprocessing_recommendations(&issues);
        assert!(!recommendations.is_empty());
        assert_eq!(
            recommendations[0].recommendation_type,
            PreprocessingType::GainAdjustment
        );
    }

    #[test]
    fn test_processing_report() {
        let assessment = AudioQualityAssessment {
            overall_quality: 0.75,
            quality_metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("rms_level".to_string(), 0.1);
                metrics.insert("peak_level".to_string(), 0.5);
                metrics
            },
            detected_issues: vec![],
            preprocessing_recommendations: vec![],
            processing_stats: ProcessingStatistics {
                original_duration_sec: 1.0,
                processed_duration_sec: 1.0,
                original_sample_rate: 16000,
                processed_sample_rate: 16000,
                processing_time_ms: 100,
                memory_usage_bytes: None,
                operations_applied: vec!["Test Operation".to_string()],
            },
        };

        let preprocessor = AdvancedPreprocessor::default();
        let report = preprocessor.create_processing_report(&assessment);

        assert!(report.contains("Quality Assessment"));
        assert!(report.contains("0.75"));
        assert!(report.contains("Test Operation"));
    }
}
