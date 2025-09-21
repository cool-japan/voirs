//! Dataset validation utilities
//!
//! This module provides comprehensive validation tools for speech synthesis datasets,
//! including format checking, integrity validation, and quality assessment.

use crate::{DatasetSample, LanguageCode, Result, ValidationReport};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Comprehensive dataset validator
#[derive(Debug, Clone)]
pub struct DatasetValidator {
    config: ValidationConfig,
}

/// Configuration for dataset validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Minimum audio duration in seconds
    pub min_duration: f32,
    /// Maximum audio duration in seconds
    pub max_duration: f32,
    /// Minimum text length in characters
    pub min_text_length: usize,
    /// Maximum text length in characters
    pub max_text_length: usize,
    /// Required sample rates (if empty, any is allowed)
    pub allowed_sample_rates: Vec<u32>,
    /// Required channel counts (if empty, any is allowed)
    pub allowed_channels: Vec<u32>,
    /// Whether to enforce ID uniqueness
    pub enforce_unique_ids: bool,
    /// Whether to check audio-text alignment
    pub check_alignment: bool,
    /// Whether to validate audio quality metrics
    pub validate_quality: bool,
    /// Minimum quality score (0.0-1.0)
    pub min_quality_score: f32,
    /// Maximum clipping percentage
    pub max_clipping_percent: f32,
    /// Minimum SNR in dB
    pub min_snr_db: f32,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            min_duration: 0.1,
            max_duration: 30.0,
            min_text_length: 1,
            max_text_length: 1000,
            allowed_sample_rates: vec![],
            allowed_channels: vec![],
            enforce_unique_ids: true,
            check_alignment: true,
            validate_quality: true,
            min_quality_score: 0.3,
            max_clipping_percent: 5.0,
            min_snr_db: 10.0,
        }
    }
}

/// Detailed validation report with comprehensive analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedValidationReport {
    /// Basic validation info
    pub basic_report: ValidationReport,
    /// Format validation results
    pub format_validation: FormatValidationResult,
    /// Integrity validation results
    pub integrity_validation: IntegrityValidationResult,
    /// Quality validation results
    pub quality_validation: QualityValidationResult,
    /// Statistics about the validation
    pub validation_stats: ValidationStatistics,
}

/// Audio and text format validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatValidationResult {
    /// Audio format consistency
    pub audio_format_consistent: bool,
    /// Detected sample rates
    pub sample_rates: Vec<u32>,
    /// Detected channel counts
    pub channel_counts: Vec<u32>,
    /// Text encoding issues
    pub text_encoding_issues: Vec<String>,
    /// Character set analysis
    pub character_sets: HashMap<String, usize>,
    /// Language consistency
    pub language_consistent: bool,
    /// Detected languages
    pub detected_languages: Vec<LanguageCode>,
}

/// Data integrity validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityValidationResult {
    /// Whether all IDs are unique
    pub unique_ids: bool,
    /// Duplicate ID information
    pub duplicate_ids: Vec<String>,
    /// Empty or corrupted samples
    pub corrupted_samples: Vec<usize>,
    /// Missing required fields
    pub missing_fields: HashMap<String, Vec<usize>>,
    /// Metadata consistency
    pub metadata_consistent: bool,
    /// Type mismatches in metadata
    pub metadata_type_issues: HashMap<String, Vec<String>>,
}

/// Quality validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityValidationResult {
    /// Overall quality assessment
    pub overall_quality_ok: bool,
    /// Samples failing quality thresholds
    pub low_quality_samples: Vec<usize>,
    /// Audio corruption detection
    pub corrupted_audio_samples: Vec<usize>,
    /// Silent audio detection
    pub silent_samples: Vec<usize>,
    /// Clipped audio detection
    pub clipped_samples: Vec<usize>,
    /// Audio-text alignment issues
    pub alignment_issues: Vec<usize>,
    /// Quality score distribution
    pub quality_distribution: QualityDistribution,
}

/// Quality score distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDistribution {
    /// Excellent quality samples (>0.8)
    pub excellent_count: usize,
    /// Good quality samples (0.6-0.8)
    pub good_count: usize,
    /// Fair quality samples (0.4-0.6)
    pub fair_count: usize,
    /// Poor quality samples (<0.4)
    pub poor_count: usize,
    /// Average quality score
    pub average_score: f32,
    /// Quality score standard deviation
    pub score_std_dev: f32,
}

/// Validation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStatistics {
    /// Total samples validated
    pub total_samples: usize,
    /// Samples that passed validation
    pub passed_samples: usize,
    /// Samples with warnings
    pub warning_samples: usize,
    /// Samples with errors
    pub error_samples: usize,
    /// Validation completion time in milliseconds
    pub validation_time_ms: u64,
}

impl DatasetValidator {
    /// Create a new validator with default configuration
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
        }
    }

    /// Create a validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Validate a single dataset sample
    pub fn validate_sample(
        &self,
        sample: &DatasetSample,
        index: usize,
    ) -> Result<SampleValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate audio duration
        let duration = sample.audio.duration();
        if duration < self.config.min_duration {
            let min_duration = self.config.min_duration;
            errors.push(format!(
                "Sample {index}: Audio too short ({duration:.3}s, minimum: {min_duration:.3}s)"
            ));
        }
        if duration > self.config.max_duration {
            let max_duration = self.config.max_duration;
            warnings.push(format!(
                "Sample {index}: Audio very long ({duration:.1}s, maximum: {max_duration:.1}s)"
            ));
        }

        // Validate text length
        let text_len = sample.text.len();
        if text_len < self.config.min_text_length {
            let min_text_length = self.config.min_text_length;
            errors.push(format!(
                "Sample {index}: Text too short ({text_len} chars, minimum: {min_text_length})"
            ));
        }
        if text_len > self.config.max_text_length {
            let max_text_length = self.config.max_text_length;
            warnings.push(format!(
                "Sample {index}: Text very long ({text_len} chars, maximum: {max_text_length})"
            ));
        }

        // Validate audio format
        if !self.config.allowed_sample_rates.is_empty()
            && !self
                .config
                .allowed_sample_rates
                .contains(&sample.audio.sample_rate())
        {
            errors.push(format!(
                "Sample {index}: Invalid sample rate ({}Hz, allowed: {:?})",
                sample.audio.sample_rate(),
                self.config.allowed_sample_rates
            ));
        }

        if !self.config.allowed_channels.is_empty()
            && !self
                .config
                .allowed_channels
                .contains(&sample.audio.channels())
        {
            errors.push(format!(
                "Sample {index}: Invalid channel count ({}, allowed: {:?})",
                sample.audio.channels(),
                self.config.allowed_channels
            ));
        }

        // Validate audio integrity
        let audio_samples = sample.audio.samples();
        if audio_samples.is_empty() {
            errors.push(format!("Sample {index}: Empty audio"));
        } else {
            // Check for invalid audio values
            let invalid_count = audio_samples.iter().filter(|&&s| !s.is_finite()).count();
            if invalid_count > 0 {
                errors.push(format!(
                    "Sample {index}: Audio contains {invalid_count} invalid values (NaN/Infinity)"
                ));
            }

            // Check for silent audio
            let max_amplitude = audio_samples
                .iter()
                .fold(0.0f32, |max, &s| max.max(s.abs()));
            if max_amplitude < 0.001 {
                warnings.push(format!("Sample {index}: Audio appears to be silent"));
            }

            // Check for clipped audio
            let clipped_count = audio_samples.iter().filter(|&&s| s.abs() >= 0.999).count();
            let clipping_percent = (clipped_count as f32 / audio_samples.len() as f32) * 100.0;
            if clipping_percent > self.config.max_clipping_percent {
                warnings.push(format!(
                    "Sample {index}: High clipping detected ({clipping_percent:.1}%)"
                ));
            }
        }

        // Validate text content
        if sample.text.trim().is_empty() {
            errors.push(format!("Sample {index}: Empty or whitespace-only text"));
        }

        // Check for unusual characters
        if sample
            .text
            .chars()
            .any(|c| c.is_control() && c != '\n' && c != '\t')
        {
            warnings.push(format!("Sample {index}: Text contains control characters"));
        }

        // Validate quality metrics if available
        if self.config.validate_quality {
            if let Some(quality_score) = sample.quality.overall_quality {
                if quality_score < self.config.min_quality_score {
                    warnings.push(format!(
                        "Sample {index}: Low quality score ({quality_score:.2})"
                    ));
                }
            }

            if let Some(snr) = sample.quality.snr {
                if snr < self.config.min_snr_db {
                    warnings.push(format!("Sample {index}: Low SNR ({snr:.1}dB)"));
                }
            }

            if let Some(clipping) = sample.quality.clipping {
                if clipping > self.config.max_clipping_percent {
                    warnings.push(format!(
                        "Sample {index}: High clipping in quality metrics ({clipping:.1}%)"
                    ));
                }
            }
        }

        // Check audio-text alignment if enabled
        if self.config.check_alignment {
            let alignment_result = self.check_audio_text_alignment(sample);
            if let Some(issue) = alignment_result {
                warnings.push(format!("Sample {index}: {issue}"));
            }
        }

        Ok(SampleValidationResult {
            index,
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }

    /// Validate an entire dataset
    pub fn validate_dataset<T>(&self, samples: &[T]) -> Result<DetailedValidationReport>
    where
        T: AsRef<DatasetSample>,
    {
        let start_time = std::time::Instant::now();

        let mut all_errors = Vec::new();
        let mut all_warnings = Vec::new();
        let mut sample_results = Vec::new();

        // Validate individual samples
        for (index, sample) in samples.iter().enumerate() {
            let result = self.validate_sample(sample.as_ref(), index)?;
            all_errors.extend(result.errors.clone());
            all_warnings.extend(result.warnings.clone());
            sample_results.push(result);
        }

        // Perform dataset-level validations
        let format_result = self.validate_formats(samples);
        let integrity_result = self.validate_integrity(samples);
        let quality_result = self.validate_quality(samples);

        // Collect additional errors and warnings from dataset-level validation
        all_errors.extend(format_result.get_errors());
        all_warnings.extend(format_result.get_warnings());
        all_errors.extend(integrity_result.get_errors());
        all_warnings.extend(integrity_result.get_warnings());
        all_errors.extend(quality_result.get_errors());
        all_warnings.extend(quality_result.get_warnings());

        let validation_time = start_time.elapsed().as_millis() as u64;

        // Calculate statistics
        let error_samples = sample_results.iter().filter(|r| !r.is_valid).count();
        let warning_samples = sample_results
            .iter()
            .filter(|r| !r.warnings.is_empty())
            .count();

        let validation_stats = ValidationStatistics {
            total_samples: samples.len(),
            passed_samples: samples.len() - error_samples,
            warning_samples,
            error_samples,
            validation_time_ms: validation_time,
        };

        Ok(DetailedValidationReport {
            basic_report: ValidationReport {
                is_valid: all_errors.is_empty(),
                errors: all_errors,
                warnings: all_warnings,
                items_validated: samples.len(),
            },
            format_validation: format_result,
            integrity_validation: integrity_result,
            quality_validation: quality_result,
            validation_stats,
        })
    }

    /// Validate audio and text formats
    fn validate_formats<T>(&self, samples: &[T]) -> FormatValidationResult
    where
        T: AsRef<DatasetSample>,
    {
        let mut sample_rates = HashSet::new();
        let mut channel_counts = HashSet::new();
        let mut character_sets = HashMap::new();
        let mut languages = HashSet::new();
        let mut text_issues = Vec::new();

        for (index, sample) in samples.iter().enumerate() {
            let sample = sample.as_ref();

            // Collect audio format info
            sample_rates.insert(sample.audio.sample_rate());
            channel_counts.insert(sample.audio.channels());

            // Collect language info
            languages.insert(sample.language);

            // Analyze character sets
            let charset = self.detect_character_set(&sample.text);
            *character_sets.entry(charset).or_insert(0) += 1;

            // Check for text encoding issues
            if sample.text.chars().any(|c| c == '\u{FFFD}') {
                text_issues.push(format!(
                    "Sample {index}: Contains replacement characters (encoding issue)"
                ));
            }
        }

        FormatValidationResult {
            audio_format_consistent: sample_rates.len() <= 1 && channel_counts.len() <= 1,
            sample_rates: sample_rates.into_iter().collect(),
            channel_counts: channel_counts.into_iter().collect(),
            text_encoding_issues: text_issues,
            character_sets,
            language_consistent: languages.len() <= 1,
            detected_languages: languages.into_iter().collect(),
        }
    }

    /// Validate data integrity
    fn validate_integrity<T>(&self, samples: &[T]) -> IntegrityValidationResult
    where
        T: AsRef<DatasetSample>,
    {
        let mut id_counts = HashMap::new();
        let mut corrupted_samples = Vec::new();
        let mut missing_fields: HashMap<String, Vec<usize>> = HashMap::new();
        let mut metadata_types: HashMap<String, HashSet<String>> = HashMap::new();

        for (index, sample) in samples.iter().enumerate() {
            let sample = sample.as_ref();

            // Check ID uniqueness
            *id_counts.entry(sample.id.clone()).or_insert(0) += 1;

            // Check for corrupted samples
            if sample.audio.samples().is_empty() || sample.text.trim().is_empty() {
                corrupted_samples.push(index);
            }

            // Check metadata consistency
            for (key, value) in &sample.metadata {
                let value_type = self.get_json_value_type(value);
                metadata_types
                    .entry(key.clone())
                    .or_default()
                    .insert(value_type);
            }

            // Check for missing critical fields
            if sample.id.is_empty() {
                missing_fields
                    .entry("id".to_string())
                    .or_default()
                    .push(index);
            }
            if sample.text.is_empty() {
                missing_fields
                    .entry("text".to_string())
                    .or_default()
                    .push(index);
            }
        }

        let duplicate_ids: Vec<String> = id_counts
            .iter()
            .filter(|(_, &count)| count > 1)
            .map(|(id, _)| id.clone())
            .collect();

        let metadata_type_issues: HashMap<String, Vec<String>> = metadata_types
            .iter()
            .filter(|(_, types)| types.len() > 1)
            .map(|(key, types)| (key.clone(), types.iter().cloned().collect()))
            .collect();

        IntegrityValidationResult {
            unique_ids: duplicate_ids.is_empty(),
            duplicate_ids,
            corrupted_samples,
            missing_fields,
            metadata_consistent: metadata_type_issues.is_empty(),
            metadata_type_issues,
        }
    }

    /// Validate audio and text quality
    fn validate_quality<T>(&self, samples: &[T]) -> QualityValidationResult
    where
        T: AsRef<DatasetSample>,
    {
        let mut low_quality_samples = Vec::new();
        let mut corrupted_audio_samples = Vec::new();
        let mut silent_samples = Vec::new();
        let mut clipped_samples = Vec::new();
        let mut alignment_issues = Vec::new();
        let mut quality_scores = Vec::new();

        for (index, sample) in samples.iter().enumerate() {
            let sample = sample.as_ref();
            let audio_samples = sample.audio.samples();

            // Check audio corruption
            if audio_samples.iter().any(|&s| !s.is_finite()) {
                corrupted_audio_samples.push(index);
                continue;
            }

            // Check for silence
            let max_amplitude = audio_samples
                .iter()
                .fold(0.0f32, |max, &s| max.max(s.abs()));
            if max_amplitude < 0.001 {
                silent_samples.push(index);
            }

            // Check for clipping
            let clipped_count = audio_samples.iter().filter(|&&s| s.abs() >= 0.999).count();
            let clipping_percent = (clipped_count as f32 / audio_samples.len() as f32) * 100.0;
            if clipping_percent > self.config.max_clipping_percent {
                clipped_samples.push(index);
            }

            // Check quality metrics
            if let Some(quality_score) = sample.quality.overall_quality {
                quality_scores.push(quality_score);
                if quality_score < self.config.min_quality_score {
                    low_quality_samples.push(index);
                }
            }

            // Check alignment
            if self.config.check_alignment && self.check_audio_text_alignment(sample).is_some() {
                alignment_issues.push(index);
            }
        }

        let quality_distribution = self.calculate_quality_distribution(&quality_scores);

        QualityValidationResult {
            overall_quality_ok: low_quality_samples.is_empty()
                && corrupted_audio_samples.is_empty(),
            low_quality_samples,
            corrupted_audio_samples,
            silent_samples,
            clipped_samples,
            alignment_issues,
            quality_distribution,
        }
    }

    /// Check audio-text alignment for reasonable speaking rates
    fn check_audio_text_alignment(&self, sample: &DatasetSample) -> Option<String> {
        let duration = sample.audio.duration();
        if duration <= 0.0 {
            return Some("Zero duration audio".to_string());
        }

        let char_count = sample.text.len() as f32;
        let word_count = sample.text.split_whitespace().count() as f32;

        let chars_per_second = char_count / duration;
        let words_per_second = word_count / duration;

        // Reasonable speaking rates (empirically determined)
        if chars_per_second > 50.0 {
            return Some(format!(
                "Speaking rate too fast ({chars_per_second:.1} chars/sec)"
            ));
        }
        if chars_per_second < 1.0 && char_count > 0.0 {
            return Some(format!(
                "Speaking rate too slow ({chars_per_second:.1} chars/sec)"
            ));
        }
        if words_per_second > 15.0 {
            return Some(format!(
                "Speaking rate too fast ({words_per_second:.1} words/sec)"
            ));
        }
        if words_per_second < 0.5 && word_count > 0.0 {
            return Some(format!(
                "Speaking rate too slow ({words_per_second:.1} words/sec)"
            ));
        }

        None
    }

    /// Detect character set of text
    fn detect_character_set(&self, text: &str) -> String {
        if text.is_ascii() {
            "ASCII".to_string()
        } else if text.chars().any(|c| {
            let code = c as u32;
            (0x4E00..=0x9FFF).contains(&code) || // CJK Unified Ideographs
            (0x3040..=0x309F).contains(&code) || // Hiragana
            (0x30A0..=0x30FF).contains(&code) // Katakana
        }) {
            "CJK".to_string()
        } else if text.chars().any(|c| (c as u32) > 255) {
            "Unicode".to_string()
        } else {
            "Latin-1".to_string()
        }
    }

    /// Get JSON value type as string
    fn get_json_value_type(&self, value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::String(_) => "string".to_string(),
            serde_json::Value::Number(_) => "number".to_string(),
            serde_json::Value::Bool(_) => "boolean".to_string(),
            serde_json::Value::Array(_) => "array".to_string(),
            serde_json::Value::Object(_) => "object".to_string(),
            serde_json::Value::Null => "null".to_string(),
        }
    }

    /// Calculate quality score distribution
    fn calculate_quality_distribution(&self, scores: &[f32]) -> QualityDistribution {
        if scores.is_empty() {
            return QualityDistribution {
                excellent_count: 0,
                good_count: 0,
                fair_count: 0,
                poor_count: 0,
                average_score: 0.0,
                score_std_dev: 0.0,
            };
        }

        let mut excellent = 0;
        let mut good = 0;
        let mut fair = 0;
        let mut poor = 0;

        for &score in scores {
            if score > 0.8 {
                excellent += 1;
            } else if score > 0.6 {
                good += 1;
            } else if score > 0.4 {
                fair += 1;
            } else {
                poor += 1;
            }
        }

        let average = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance =
            scores.iter().map(|&x| (x - average).powi(2)).sum::<f32>() / scores.len() as f32;
        let std_dev = variance.sqrt();

        QualityDistribution {
            excellent_count: excellent,
            good_count: good,
            fair_count: fair,
            poor_count: poor,
            average_score: average,
            score_std_dev: std_dev,
        }
    }
}

/// Single sample validation result
#[derive(Debug, Clone)]
pub struct SampleValidationResult {
    pub index: usize,
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

// Helper trait implementations for getting errors and warnings from validation results
impl FormatValidationResult {
    fn get_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if !self.audio_format_consistent {
            errors.push("Inconsistent audio formats detected".to_string());
        }
        if !self.language_consistent {
            errors.push("Multiple languages detected in dataset".to_string());
        }
        errors.extend(self.text_encoding_issues.clone());
        errors
    }

    fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.sample_rates.len() > 1 {
            warnings.push(format!(
                "Multiple sample rates detected: {:?}",
                self.sample_rates
            ));
        }
        if self.channel_counts.len() > 1 {
            warnings.push(format!(
                "Multiple channel counts detected: {:?}",
                self.channel_counts
            ));
        }
        warnings
    }
}

impl IntegrityValidationResult {
    fn get_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if !self.unique_ids {
            let duplicate_ids = &self.duplicate_ids;
            errors.push(format!("Duplicate IDs found: {duplicate_ids:?}"));
        }
        if !self.corrupted_samples.is_empty() {
            let corrupted_samples = &self.corrupted_samples;
            errors.push(format!("Corrupted samples found: {corrupted_samples:?}"));
        }
        for (field, indices) in &self.missing_fields {
            errors.push(format!("Missing field '{field}' in samples: {indices:?}"));
        }
        errors
    }

    fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if !self.metadata_consistent {
            warnings.push("Inconsistent metadata types detected".to_string());
        }
        warnings
    }
}

impl QualityValidationResult {
    fn get_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if !self.corrupted_audio_samples.is_empty() {
            errors.push(format!(
                "Corrupted audio samples: {:?}",
                self.corrupted_audio_samples
            ));
        }
        errors
    }

    fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if !self.low_quality_samples.is_empty() {
            warnings.push(format!(
                "Low quality samples: {:?}",
                self.low_quality_samples
            ));
        }
        if !self.silent_samples.is_empty() {
            warnings.push(format!(
                "Silent samples detected: {:?}",
                self.silent_samples
            ));
        }
        if !self.clipped_samples.is_empty() {
            warnings.push(format!(
                "Clipped samples detected: {:?}",
                self.clipped_samples
            ));
        }
        if !self.alignment_issues.is_empty() {
            warnings.push(format!(
                "Audio-text alignment issues: {:?}",
                self.alignment_issues
            ));
        }
        warnings
    }
}

impl Default for DatasetValidator {
    fn default() -> Self {
        Self::new()
    }
}

// Helper trait for converting samples to references
impl AsRef<DatasetSample> for DatasetSample {
    fn as_ref(&self) -> &DatasetSample {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, LanguageCode};

    fn create_test_sample(id: &str, text: &str, duration: f32) -> DatasetSample {
        let sample_rate = 22050;
        let num_samples = (duration * sample_rate as f32) as usize;
        let audio = AudioData::new(vec![0.1; num_samples], sample_rate, 1);

        DatasetSample::new(id.to_string(), text.to_string(), audio, LanguageCode::EnUs)
    }

    #[test]
    fn test_sample_validation() {
        let validator = DatasetValidator::new();
        let sample = create_test_sample("test-001", "Hello world", 2.0);

        let result = validator.validate_sample(&sample, 0).unwrap();
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_sample_validation_errors() {
        let validator = DatasetValidator::new();
        let sample = create_test_sample("test-001", "", 0.05); // Too short, empty text

        let result = validator.validate_sample(&sample, 0).unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_dataset_validation() {
        let validator = DatasetValidator::new();
        let samples = vec![
            create_test_sample("test-001", "Hello world", 2.0),
            create_test_sample("test-002", "Good morning", 1.5),
            create_test_sample("test-003", "How are you?", 1.8),
        ];

        let result = validator.validate_dataset(&samples).unwrap();
        assert!(result.basic_report.is_valid);
        assert_eq!(result.validation_stats.total_samples, 3);
        assert_eq!(result.validation_stats.passed_samples, 3);
    }

    #[test]
    fn test_duplicate_id_detection() {
        let validator = DatasetValidator::new();
        let samples = vec![
            create_test_sample("test-001", "Hello world", 2.0),
            create_test_sample("test-001", "Duplicate ID", 1.5), // Duplicate ID
            create_test_sample("test-003", "How are you?", 1.8),
        ];

        let result = validator.validate_dataset(&samples).unwrap();
        assert!(!result.basic_report.is_valid);
        assert!(!result.integrity_validation.unique_ids);
        assert!(result
            .integrity_validation
            .duplicate_ids
            .contains(&"test-001".to_string()));
    }

    #[test]
    fn test_audio_text_alignment() {
        let validator = DatasetValidator::new();

        // Create sample with unrealistic speaking rate
        let sample = create_test_sample(
            "test-001",
            "This is way too much text for the duration",
            0.1,
        );

        let alignment_issue = validator.check_audio_text_alignment(&sample);
        assert!(alignment_issue.is_some());
        assert!(alignment_issue.unwrap().contains("Speaking rate too fast"));
    }

    #[test]
    fn test_quality_distribution() {
        let validator = DatasetValidator::new();
        let scores = vec![0.9, 0.8, 0.7, 0.5, 0.3, 0.1];

        let distribution = validator.calculate_quality_distribution(&scores);
        assert_eq!(distribution.excellent_count, 1); // 0.9
        assert_eq!(distribution.good_count, 2); // 0.8, 0.7
        assert_eq!(distribution.fair_count, 1); // 0.5
        assert_eq!(distribution.poor_count, 2); // 0.3, 0.1
    }
}
