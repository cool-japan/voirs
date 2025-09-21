//! Advanced data quality validation and dataset management utilities
//!
//! This module provides comprehensive data quality validation tools for speech synthesis
//! evaluation datasets, ensuring data integrity and reliability.

use crate::VoirsError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Data quality validation errors
#[derive(Error, Debug)]
pub enum DataQualityError {
    /// Invalid audio data detected
    #[error("Invalid audio data: {0}")]
    InvalidAudio(String),
    /// Dataset validation failed
    #[error("Dataset validation failed: {0}")]
    ValidationFailed(String),
    /// Missing required metadata
    #[error("Missing required metadata: {0}")]
    MissingMetadata(String),
    /// Data integrity check failed
    #[error("Data integrity check failed: {0}")]
    IntegrityCheckFailed(String),
}

/// Audio quality issue severity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IssueSeverity {
    /// Critical issues that prevent evaluation
    Critical,
    /// Major issues that significantly affect quality
    Major,
    /// Minor issues with minimal impact
    Minor,
    /// Warning issues for awareness
    Warning,
}

/// Audio quality issue types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AudioIssueType {
    /// Silent or near-silent audio
    SilentAudio,
    /// Clipped audio samples
    Clipping,
    /// DC offset detected
    DcOffset,
    /// High noise level
    HighNoise,
    /// Inconsistent sample rate
    SampleRateInconsistency,
    /// Channel imbalance
    ChannelImbalance,
    /// Corrupted audio data
    CorruptedData,
    /// Missing audio segments
    MissingSegments,
}

/// Individual audio quality issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityIssue {
    /// Issue type
    pub issue_type: AudioIssueType,
    /// Severity level
    pub severity: IssueSeverity,
    /// Issue description
    pub description: String,
    /// Location in audio (start time in seconds)
    pub start_time: Option<f64>,
    /// Duration of issue in seconds
    pub duration: Option<f64>,
    /// Recommended action
    pub recommendation: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
}

/// Dataset metadata validation requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataRequirements {
    /// Required metadata fields
    pub required_fields: Vec<String>,
    /// Optional metadata fields
    pub optional_fields: Vec<String>,
    /// Field value constraints
    pub field_constraints: HashMap<String, String>,
    /// Language requirements
    pub language_requirements: Option<Vec<String>>,
    /// Speaker requirements
    pub speaker_requirements: Option<HashMap<String, String>>,
}

/// Audio quality validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityConfig {
    /// Silence threshold (amplitude)
    pub silence_threshold: f64,
    /// Clipping threshold
    pub clipping_threshold: f64,
    /// DC offset threshold
    pub dc_offset_threshold: f64,
    /// Noise floor threshold (dB)
    pub noise_floor_threshold: f64,
    /// Sample rate tolerance (Hz)
    pub sample_rate_tolerance: u32,
    /// Channel balance tolerance
    pub channel_balance_tolerance: f64,
    /// Minimum audio duration (seconds)
    pub min_duration: f64,
    /// Maximum audio duration (seconds)
    pub max_duration: f64,
}

impl Default for AudioQualityConfig {
    fn default() -> Self {
        Self {
            silence_threshold: 0.001,
            clipping_threshold: 0.95,
            dc_offset_threshold: 0.1,
            noise_floor_threshold: -60.0,
            sample_rate_tolerance: 100,
            channel_balance_tolerance: 0.1,
            min_duration: 0.1,
            max_duration: 30.0,
        }
    }
}

/// Dataset validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetValidationReport {
    /// Dataset name
    pub dataset_name: String,
    /// Total number of samples
    pub total_samples: usize,
    /// Valid samples count
    pub valid_samples: usize,
    /// Invalid samples count
    pub invalid_samples: usize,
    /// Quality score (0.0 - 1.0)
    pub quality_score: f64,
    /// Audio quality issues
    pub audio_issues: Vec<AudioQualityIssue>,
    /// Metadata validation results
    pub metadata_validation: HashMap<String, bool>,
    /// Sample statistics
    pub sample_statistics: SampleStatistics,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Dataset sample statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleStatistics {
    /// Average duration (seconds)
    pub avg_duration: f64,
    /// Duration standard deviation
    pub duration_std: f64,
    /// Average sample rate
    pub avg_sample_rate: f64,
    /// Most common sample rate
    pub common_sample_rate: u32,
    /// Average dynamic range (dB)
    pub avg_dynamic_range: f64,
    /// Average RMS level
    pub avg_rms_level: f64,
    /// Language distribution
    pub language_distribution: HashMap<String, usize>,
    /// Speaker distribution
    pub speaker_distribution: HashMap<String, usize>,
}

/// Data quality validator
#[derive(Debug)]
pub struct DataQualityValidator {
    config: AudioQualityConfig,
    metadata_requirements: Option<MetadataRequirements>,
}

impl Default for DataQualityValidator {
    fn default() -> Self {
        Self::new(AudioQualityConfig::default())
    }
}

impl DataQualityValidator {
    /// Create a new data quality validator
    pub fn new(config: AudioQualityConfig) -> Self {
        Self {
            config,
            metadata_requirements: None,
        }
    }

    /// Set metadata validation requirements
    pub fn with_metadata_requirements(mut self, requirements: MetadataRequirements) -> Self {
        self.metadata_requirements = Some(requirements);
        self
    }

    /// Validate audio quality
    pub fn validate_audio_quality(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Vec<AudioQualityIssue> {
        let mut issues = Vec::new();

        // Check for silent audio
        if let Some(issue) = self.check_silence(audio) {
            issues.push(issue);
        }

        // Check for clipping
        if let Some(issue) = self.check_clipping(audio) {
            issues.push(issue);
        }

        // Check for DC offset
        if let Some(issue) = self.check_dc_offset(audio) {
            issues.push(issue);
        }

        // Check noise level
        if let Some(issue) = self.check_noise_level(audio) {
            issues.push(issue);
        }

        // Check duration
        let duration = audio.len() as f64 / sample_rate as f64;
        if let Some(issue) = self.check_duration(duration) {
            issues.push(issue);
        }

        // Check for corrupted data (NaN, Inf)
        if let Some(issue) = self.check_corrupted_data(audio) {
            issues.push(issue);
        }

        issues
    }

    /// Check for silent audio
    fn check_silence(&self, audio: &[f32]) -> Option<AudioQualityIssue> {
        let max_amplitude = audio.iter().map(|&x| x.abs()).fold(0.0, f32::max);

        if max_amplitude < self.config.silence_threshold as f32 {
            Some(AudioQualityIssue {
                issue_type: AudioIssueType::SilentAudio,
                severity: IssueSeverity::Critical,
                description: format!(
                    "Audio is silent or near-silent (max amplitude: {:.6})",
                    max_amplitude
                ),
                start_time: Some(0.0),
                duration: Some(audio.len() as f64),
                recommendation: "Check audio source and recording setup".to_string(),
                confidence: 0.95,
            })
        } else {
            None
        }
    }

    /// Check for clipping
    fn check_clipping(&self, audio: &[f32]) -> Option<AudioQualityIssue> {
        let clipped_samples = audio
            .iter()
            .filter(|&&x| x.abs() >= self.config.clipping_threshold as f32)
            .count();

        if clipped_samples > 0 {
            let clipping_percentage = (clipped_samples as f64 / audio.len() as f64) * 100.0;
            let severity = if clipping_percentage > 5.0 {
                IssueSeverity::Critical
            } else if clipping_percentage > 1.0 {
                IssueSeverity::Major
            } else {
                IssueSeverity::Minor
            };

            Some(AudioQualityIssue {
                issue_type: AudioIssueType::Clipping,
                severity,
                description: format!(
                    "Audio clipping detected ({} samples, {:.2}%)",
                    clipped_samples, clipping_percentage
                ),
                start_time: None,
                duration: None,
                recommendation: "Reduce input gain or apply limiting before recording".to_string(),
                confidence: 0.9,
            })
        } else {
            None
        }
    }

    /// Check for DC offset
    fn check_dc_offset(&self, audio: &[f32]) -> Option<AudioQualityIssue> {
        let dc_offset = audio.iter().sum::<f32>() / audio.len() as f32;

        if dc_offset.abs() > self.config.dc_offset_threshold as f32 {
            Some(AudioQualityIssue {
                issue_type: AudioIssueType::DcOffset,
                severity: IssueSeverity::Minor,
                description: format!("DC offset detected: {:.6}", dc_offset),
                start_time: Some(0.0),
                duration: Some(audio.len() as f64),
                recommendation: "Apply DC offset removal filter".to_string(),
                confidence: 0.85,
            })
        } else {
            None
        }
    }

    /// Check noise level
    fn check_noise_level(&self, audio: &[f32]) -> Option<AudioQualityIssue> {
        // Calculate RMS level
        let rms = (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        let rms_db = 20.0 * rms.log10();

        if rms_db < self.config.noise_floor_threshold as f32 {
            Some(AudioQualityIssue {
                issue_type: AudioIssueType::HighNoise,
                severity: IssueSeverity::Major,
                description: format!("High noise floor detected: {:.1} dB", rms_db),
                start_time: Some(0.0),
                duration: Some(audio.len() as f64),
                recommendation: "Apply noise reduction or improve recording environment"
                    .to_string(),
                confidence: 0.8,
            })
        } else {
            None
        }
    }

    /// Check audio duration
    fn check_duration(&self, duration: f64) -> Option<AudioQualityIssue> {
        if duration < self.config.min_duration {
            Some(AudioQualityIssue {
                issue_type: AudioIssueType::MissingSegments,
                severity: IssueSeverity::Major,
                description: format!(
                    "Audio too short: {:.2}s (minimum: {:.2}s)",
                    duration, self.config.min_duration
                ),
                start_time: Some(0.0),
                duration: Some(duration),
                recommendation: "Ensure complete audio recordings".to_string(),
                confidence: 0.95,
            })
        } else if duration > self.config.max_duration {
            Some(AudioQualityIssue {
                issue_type: AudioIssueType::MissingSegments,
                severity: IssueSeverity::Warning,
                description: format!(
                    "Audio very long: {:.2}s (maximum: {:.2}s)",
                    duration, self.config.max_duration
                ),
                start_time: Some(0.0),
                duration: Some(duration),
                recommendation: "Consider splitting long audio files".to_string(),
                confidence: 0.7,
            })
        } else {
            None
        }
    }

    /// Check for corrupted data
    fn check_corrupted_data(&self, audio: &[f32]) -> Option<AudioQualityIssue> {
        let corrupted_samples = audio.iter().filter(|&&x| !x.is_finite()).count();

        if corrupted_samples > 0 {
            Some(AudioQualityIssue {
                issue_type: AudioIssueType::CorruptedData,
                severity: IssueSeverity::Critical,
                description: format!(
                    "Corrupted audio data detected ({} samples)",
                    corrupted_samples
                ),
                start_time: None,
                duration: None,
                recommendation: "Re-acquire audio data from source".to_string(),
                confidence: 1.0,
            })
        } else {
            None
        }
    }

    /// Validate metadata against requirements
    pub fn validate_metadata(&self, metadata: &HashMap<String, String>) -> HashMap<String, bool> {
        let mut validation_results = HashMap::new();

        if let Some(requirements) = &self.metadata_requirements {
            // Check required fields
            for field in &requirements.required_fields {
                validation_results.insert(
                    field.clone(),
                    metadata.contains_key(field) && !metadata[field].is_empty(),
                );
            }

            // Check field constraints
            for (field, constraint) in &requirements.field_constraints {
                if let Some(value) = metadata.get(field) {
                    let valid = self.validate_field_constraint(value, constraint);
                    validation_results.insert(format!("{}_constraint", field), valid);
                }
            }
        }

        validation_results
    }

    /// Validate individual field constraint
    fn validate_field_constraint(&self, value: &str, constraint: &str) -> bool {
        match constraint {
            "non_empty" => !value.is_empty(),
            "numeric" => value.parse::<f64>().is_ok(),
            "language_code" => value.len() == 2 && value.chars().all(|c| c.is_ascii_lowercase()),
            _ if constraint.starts_with("min_length:") => {
                if let Ok(min_len) = constraint[11..].parse::<usize>() {
                    value.len() >= min_len
                } else {
                    false
                }
            }
            _ if constraint.starts_with("max_length:") => {
                if let Ok(max_len) = constraint[11..].parse::<usize>() {
                    value.len() <= max_len
                } else {
                    false
                }
            }
            _ => true, // Unknown constraint, assume valid
        }
    }

    /// Generate dataset validation report
    pub fn validate_dataset(
        &self,
        dataset_name: &str,
        audio_samples: &[(Vec<f32>, u32)], // (audio, sample_rate)
        metadata_samples: &[HashMap<String, String>],
    ) -> Result<DatasetValidationReport, DataQualityError> {
        if audio_samples.len() != metadata_samples.len() {
            return Err(DataQualityError::ValidationFailed(
                "Audio samples and metadata count mismatch".to_string(),
            ));
        }

        let total_samples = audio_samples.len();
        let mut all_audio_issues = Vec::new();
        let mut valid_samples = 0;
        let mut invalid_samples = 0;
        let mut all_metadata_validation = HashMap::new();

        // Statistics collection
        let mut durations = Vec::new();
        let mut sample_rates = Vec::new();
        let mut rms_levels = Vec::new();
        let mut language_counts = HashMap::new();
        let mut speaker_counts = HashMap::new();

        // Validate each sample
        for (i, ((audio, sample_rate), metadata)) in audio_samples
            .iter()
            .zip(metadata_samples.iter())
            .enumerate()
        {
            // Audio quality validation
            let audio_issues = self.validate_audio_quality(audio, *sample_rate);
            let has_critical_issues = audio_issues
                .iter()
                .any(|issue| issue.severity == IssueSeverity::Critical);

            if has_critical_issues {
                invalid_samples += 1;
            } else {
                valid_samples += 1;
            }

            // Store issues with sample index
            for mut issue in audio_issues {
                issue.description = format!("Sample {}: {}", i, issue.description);
                all_audio_issues.push(issue);
            }

            // Metadata validation
            let metadata_validation = self.validate_metadata(metadata);
            for (key, valid) in metadata_validation {
                let field_key = format!("{}_{}", key, i);
                all_metadata_validation.insert(field_key, valid);
            }

            // Collect statistics
            let duration = audio.len() as f64 / *sample_rate as f64;
            durations.push(duration);
            sample_rates.push(*sample_rate);

            let rms = (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
            rms_levels.push(rms);

            // Language and speaker statistics
            if let Some(language) = metadata.get("language") {
                *language_counts.entry(language.clone()).or_insert(0) += 1;
            }
            if let Some(speaker) = metadata.get("speaker") {
                *speaker_counts.entry(speaker.clone()).or_insert(0) += 1;
            }
        }

        // Calculate sample statistics
        let sample_statistics = SampleStatistics {
            avg_duration: durations.iter().sum::<f64>() / durations.len() as f64,
            duration_std: {
                let mean = durations.iter().sum::<f64>() / durations.len() as f64;
                let variance = durations.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / durations.len() as f64;
                variance.sqrt()
            },
            avg_sample_rate: sample_rates.iter().sum::<u32>() as f64 / sample_rates.len() as f64,
            common_sample_rate: *sample_rates
                .iter()
                .max_by_key(|&&sr| sample_rates.iter().filter(|&&x| x == sr).count())
                .unwrap_or(&0),
            avg_dynamic_range: 0.0, // Simplified for now
            avg_rms_level: (rms_levels.iter().sum::<f32>() / rms_levels.len() as f32) as f64,
            language_distribution: language_counts,
            speaker_distribution: speaker_counts,
        };

        // Calculate quality score
        let quality_score = valid_samples as f64 / total_samples as f64;

        // Generate recommendations
        let mut recommendations = Vec::new();

        if quality_score < 0.9 {
            recommendations.push("Dataset quality below recommended threshold (90%)".to_string());
        }

        let critical_issues = all_audio_issues
            .iter()
            .filter(|issue| issue.severity == IssueSeverity::Critical)
            .count();

        if critical_issues > 0 {
            recommendations.push(format!(
                "Address {} critical audio quality issues",
                critical_issues
            ));
        }

        if sample_statistics.duration_std > 5.0 {
            recommendations.push(
                "High duration variance detected, consider more consistent audio lengths"
                    .to_string(),
            );
        }

        Ok(DatasetValidationReport {
            dataset_name: dataset_name.to_string(),
            total_samples,
            valid_samples,
            invalid_samples,
            quality_score,
            audio_issues: all_audio_issues,
            metadata_validation: all_metadata_validation,
            sample_statistics,
            recommendations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_quality_validator_creation() {
        let validator = DataQualityValidator::default();
        assert_eq!(validator.config.silence_threshold, 0.001);
    }

    #[test]
    fn test_silence_detection() {
        let validator = DataQualityValidator::default();
        let silent_audio = vec![0.0; 1000];

        let issues = validator.validate_audio_quality(&silent_audio, 16000);
        assert!(!issues.is_empty());
        assert_eq!(issues[0].issue_type, AudioIssueType::SilentAudio);
    }

    #[test]
    fn test_clipping_detection() {
        let validator = DataQualityValidator::default();
        let mut audio = vec![0.5; 1000];
        audio[500] = 1.0; // Add clipped sample

        let issues = validator.validate_audio_quality(&audio, 16000);
        let clipping_issue = issues
            .iter()
            .find(|issue| issue.issue_type == AudioIssueType::Clipping);
        assert!(clipping_issue.is_some());
    }

    #[test]
    fn test_dc_offset_detection() {
        let validator = DataQualityValidator::default();
        let audio: Vec<f32> = (0..1000).map(|_| 0.2).collect(); // Constant DC offset

        let issues = validator.validate_audio_quality(&audio, 16000);
        let dc_issue = issues
            .iter()
            .find(|issue| issue.issue_type == AudioIssueType::DcOffset);
        assert!(dc_issue.is_some());
    }

    #[test]
    fn test_corrupted_data_detection() {
        let validator = DataQualityValidator::default();
        let mut audio = vec![0.1; 1000];
        audio[500] = f32::NAN; // Add corrupted sample

        let issues = validator.validate_audio_quality(&audio, 16000);
        let corruption_issue = issues
            .iter()
            .find(|issue| issue.issue_type == AudioIssueType::CorruptedData);
        assert!(corruption_issue.is_some());
    }

    #[test]
    fn test_metadata_validation() {
        let requirements = MetadataRequirements {
            required_fields: vec!["speaker".to_string(), "language".to_string()],
            optional_fields: vec!["emotion".to_string()],
            field_constraints: HashMap::from([
                ("language".to_string(), "language_code".to_string()),
                ("speaker".to_string(), "non_empty".to_string()),
            ]),
            language_requirements: None,
            speaker_requirements: None,
        };

        let validator = DataQualityValidator::default().with_metadata_requirements(requirements);

        let mut metadata = HashMap::new();
        metadata.insert("speaker".to_string(), "speaker1".to_string());
        metadata.insert("language".to_string(), "en".to_string());

        let validation = validator.validate_metadata(&metadata);
        assert_eq!(validation.get("speaker"), Some(&true));
        assert_eq!(validation.get("language"), Some(&true));
        assert_eq!(validation.get("language_constraint"), Some(&true));
    }

    #[test]
    fn test_dataset_validation_report() {
        let validator = DataQualityValidator::default();

        let audio_samples = vec![
            (vec![0.1; 16000], 16000), // Valid audio
            (vec![0.0; 16000], 16000), // Silent audio (invalid)
        ];

        let metadata_samples = vec![
            HashMap::from([("speaker".to_string(), "speaker1".to_string())]),
            HashMap::from([("speaker".to_string(), "speaker2".to_string())]),
        ];

        let report = validator
            .validate_dataset("test_dataset", &audio_samples, &metadata_samples)
            .expect("Should generate validation report");

        assert_eq!(report.total_samples, 2);
        assert_eq!(report.valid_samples, 1);
        assert_eq!(report.invalid_samples, 1);
        assert_eq!(report.quality_score, 0.5);
    }

    #[test]
    fn test_field_constraint_validation() {
        let validator = DataQualityValidator::default();

        assert!(validator.validate_field_constraint("test", "non_empty"));
        assert!(!validator.validate_field_constraint("", "non_empty"));
        assert!(validator.validate_field_constraint("123.45", "numeric"));
        assert!(!validator.validate_field_constraint("not_a_number", "numeric"));
        assert!(validator.validate_field_constraint("en", "language_code"));
        assert!(!validator.validate_field_constraint("english", "language_code"));
        assert!(validator.validate_field_constraint("hello", "min_length:3"));
        assert!(!validator.validate_field_constraint("hi", "min_length:3"));
    }
}
