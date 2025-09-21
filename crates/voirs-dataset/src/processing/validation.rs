//! Processing pipeline validation utilities
//!
//! This module provides comprehensive validation tools for audio datasets
//! to ensure quality and consistency across the processing pipeline.
//!
//! Features:
//! - Audio quality metrics computation
//! - Transcript-audio length alignment validation
//! - Character set and encoding validation
//! - Duplicate detection and removal
//! - Content analysis and filtering
//! - Processing pipeline quality assurance

use crate::audio::data::AudioStats;
use crate::{AudioData, DatasetError, DatasetSample, LanguageCode, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Audio quality validation thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityThresholds {
    /// Minimum acceptable SNR in dB
    pub min_snr: Option<f32>,
    /// Maximum acceptable clipping percentage
    pub max_clipping: Option<f32>,
    /// Minimum acceptable dynamic range in dB
    pub min_dynamic_range: Option<f32>,
    /// Minimum acceptable audio duration in seconds
    pub min_duration: Option<f32>,
    /// Maximum acceptable audio duration in seconds
    pub max_duration: Option<f32>,
    /// Minimum acceptable sample rate
    pub min_sample_rate: Option<u32>,
    /// Maximum acceptable sample rate
    pub max_sample_rate: Option<u32>,
    /// Maximum acceptable silence ratio (0.0-1.0)
    pub max_silence_ratio: Option<f32>,
}

impl Default for AudioQualityThresholds {
    fn default() -> Self {
        Self {
            min_snr: Some(20.0),
            max_clipping: Some(0.01), // 1%
            min_dynamic_range: Some(20.0),
            min_duration: Some(0.5),
            max_duration: Some(30.0),
            min_sample_rate: Some(16000),
            max_sample_rate: Some(48000),
            max_silence_ratio: Some(0.5), // 50%
        }
    }
}

/// Text validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextValidationConfig {
    /// Minimum acceptable text length in characters
    pub min_length: Option<usize>,
    /// Maximum acceptable text length in characters
    pub max_length: Option<usize>,
    /// Allowed character sets by language
    pub allowed_character_sets: HashMap<LanguageCode, CharacterSet>,
    /// Whether to validate encoding
    pub validate_encoding: bool,
    /// Whether to check for proper sentence structure
    pub validate_sentence_structure: bool,
    /// Maximum ratio of non-alphabetic characters
    pub max_non_alpha_ratio: Option<f32>,
}

impl Default for TextValidationConfig {
    fn default() -> Self {
        let mut allowed_character_sets = HashMap::new();

        // English character sets
        allowed_character_sets.insert(
            LanguageCode::EnUs,
            CharacterSet::new()
                .with_ascii_letters()
                .with_ascii_digits()
                .with_basic_punctuation()
                .with_whitespace(),
        );
        allowed_character_sets.insert(
            LanguageCode::EnGb,
            CharacterSet::new()
                .with_ascii_letters()
                .with_ascii_digits()
                .with_basic_punctuation()
                .with_whitespace(),
        );

        // Japanese character set
        allowed_character_sets.insert(
            LanguageCode::Ja,
            CharacterSet::new()
                .with_hiragana()
                .with_katakana()
                .with_kanji()
                .with_ascii_letters()
                .with_ascii_digits()
                .with_basic_punctuation()
                .with_whitespace(),
        );

        Self {
            min_length: Some(5),
            max_length: Some(500),
            allowed_character_sets,
            validate_encoding: true,
            validate_sentence_structure: true,
            max_non_alpha_ratio: Some(0.3), // 30%
        }
    }
}

/// Character set definition for text validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterSet {
    /// Allowed character ranges
    pub ranges: Vec<(u32, u32)>,
    /// Additional allowed characters
    pub additional_chars: HashSet<char>,
}

impl Default for CharacterSet {
    fn default() -> Self {
        Self::new()
    }
}

impl CharacterSet {
    /// Create new empty character set
    pub fn new() -> Self {
        Self {
            ranges: Vec::new(),
            additional_chars: HashSet::new(),
        }
    }

    /// Add ASCII letters (a-z, A-Z)
    pub fn with_ascii_letters(mut self) -> Self {
        self.ranges.push((0x0041, 0x005A)); // A-Z
        self.ranges.push((0x0061, 0x007A)); // a-z
        self
    }

    /// Add ASCII digits (0-9)
    pub fn with_ascii_digits(mut self) -> Self {
        self.ranges.push((0x0030, 0x0039)); // 0-9
        self
    }

    /// Add basic punctuation
    pub fn with_basic_punctuation(mut self) -> Self {
        let punctuation = ".,!?;:'\"-()[]{}";
        for ch in punctuation.chars() {
            self.additional_chars.insert(ch);
        }
        self
    }

    /// Add whitespace characters
    pub fn with_whitespace(mut self) -> Self {
        self.additional_chars.insert(' ');
        self.additional_chars.insert('\t');
        self.additional_chars.insert('\n');
        self.additional_chars.insert('\r');
        self
    }

    /// Add Hiragana characters
    pub fn with_hiragana(mut self) -> Self {
        self.ranges.push((0x3040, 0x309F)); // Hiragana
        self
    }

    /// Add Katakana characters
    pub fn with_katakana(mut self) -> Self {
        self.ranges.push((0x30A0, 0x30FF)); // Katakana
        self
    }

    /// Add Kanji characters
    pub fn with_kanji(mut self) -> Self {
        self.ranges.push((0x4E00, 0x9FAF)); // CJK Unified Ideographs
        self.ranges.push((0x3400, 0x4DBF)); // CJK Extension A
        self
    }

    /// Check if character is allowed
    pub fn contains(&self, ch: char) -> bool {
        let code_point = ch as u32;

        // Check ranges
        for &(start, end) in &self.ranges {
            if (start..=end).contains(&code_point) {
                return true;
            }
        }

        // Check additional characters
        self.additional_chars.contains(&ch)
    }
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Audio quality thresholds
    pub audio_thresholds: AudioQualityThresholds,
    /// Text validation configuration
    pub text_config: TextValidationConfig,
    /// Whether to validate audio-text alignment
    pub validate_alignment: bool,
    /// Expected words per second for alignment validation
    pub expected_words_per_second: f32,
    /// Tolerance for alignment validation (multiplier)
    pub alignment_tolerance: f32,
    /// Whether to check for duplicates
    pub check_duplicates: bool,
    /// Similarity threshold for duplicate detection (0.0-1.0)
    pub duplicate_threshold: f32,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            audio_thresholds: AudioQualityThresholds::default(),
            text_config: TextValidationConfig::default(),
            validate_alignment: true,
            expected_words_per_second: 2.5, // Average speaking rate
            alignment_tolerance: 2.0,       // 2x tolerance
            check_duplicates: true,
            duplicate_threshold: 0.95, // 95% similarity
        }
    }
}

/// Audio quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityMetrics {
    /// Signal-to-noise ratio in dB
    pub snr: Option<f32>,
    /// Clipping percentage (0.0-1.0)
    pub clipping: f32,
    /// Dynamic range in dB
    pub dynamic_range: f32,
    /// RMS amplitude
    pub rms_amplitude: f32,
    /// Peak amplitude
    pub peak_amplitude: f32,
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    /// Silence ratio (0.0-1.0)
    pub silence_ratio: f32,
    /// Spectral centroid
    pub spectral_centroid: Option<f32>,
    /// Spectral rolloff
    pub spectral_rolloff: Option<f32>,
}

/// Text validation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextValidationMetrics {
    /// Character count
    pub character_count: usize,
    /// Word count
    pub word_count: usize,
    /// Sentence count
    pub sentence_count: usize,
    /// Non-alphabetic character ratio
    pub non_alpha_ratio: f32,
    /// Invalid characters found
    pub invalid_characters: Vec<char>,
    /// Encoding issues detected
    pub encoding_issues: Vec<String>,
}

/// Alignment validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentValidation {
    /// Expected duration based on text length
    pub expected_duration: f32,
    /// Actual audio duration
    pub actual_duration: f32,
    /// Words per second rate
    pub words_per_second: f32,
    /// Whether alignment is acceptable
    pub is_aligned: bool,
    /// Alignment score (0.0-1.0)
    pub alignment_score: f32,
}

/// Sample validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleValidationResult {
    /// Sample ID
    pub sample_id: String,
    /// Whether sample passed validation
    pub is_valid: bool,
    /// Audio quality metrics
    pub audio_metrics: Option<AudioQualityMetrics>,
    /// Text validation metrics
    pub text_metrics: TextValidationMetrics,
    /// Alignment validation
    pub alignment: Option<AlignmentValidation>,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Overall quality score (0.0-1.0)
    pub quality_score: f32,
}

/// Duplicate detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateDetectionResult {
    /// Groups of duplicate samples (by sample ID)
    pub duplicate_groups: Vec<Vec<String>>,
    /// Similarity matrix
    pub similarity_scores: HashMap<(String, String), f32>,
    /// Recommendations for removal
    pub removal_recommendations: Vec<String>,
}

/// Processing validation utilities
pub struct ProcessingValidator {
    /// Validation configuration
    config: ValidationConfig,
}

impl ProcessingValidator {
    /// Create new validator with default configuration
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
        }
    }

    /// Create validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Validate a single sample
    pub async fn validate_sample(&self, sample: &DatasetSample) -> Result<SampleValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut quality_scores = Vec::new();

        // Validate text
        let text_metrics = self.validate_text(&sample.text, sample.language)?;
        if !text_metrics.invalid_characters.is_empty() {
            errors.push(format!(
                "Invalid characters found: {:?}",
                text_metrics.invalid_characters
            ));
        }
        if !text_metrics.encoding_issues.is_empty() {
            errors.push(format!(
                "Encoding issues: {:?}",
                text_metrics.encoding_issues
            ));
        }

        // Calculate text quality score
        let text_score = self.calculate_text_quality_score(&text_metrics);
        quality_scores.push(text_score);

        // Validate audio using the sample's audio data
        let audio_metrics = Some(self.compute_audio_quality_metrics(&sample.audio)?);

        if let Some(ref metrics) = audio_metrics {
            let audio_score = self.calculate_audio_quality_score(metrics);
            quality_scores.push(audio_score);

            // Check audio thresholds
            if let Some(min_snr) = self.config.audio_thresholds.min_snr {
                if let Some(snr) = metrics.snr {
                    if snr < min_snr {
                        errors.push(format!("SNR too low: {snr:.1}dB < {min_snr:.1}dB"));
                    }
                }
            }

            if let Some(max_clipping) = self.config.audio_thresholds.max_clipping {
                if metrics.clipping > max_clipping {
                    errors.push(format!(
                        "Clipping too high: {:.3} > {:.3}",
                        metrics.clipping, max_clipping
                    ));
                }
            }

            if let Some(min_dr) = self.config.audio_thresholds.min_dynamic_range {
                if metrics.dynamic_range < min_dr {
                    warnings.push(format!(
                        "Low dynamic range: {:.1}dB < {:.1}dB",
                        metrics.dynamic_range, min_dr
                    ));
                }
            }
        }

        // Validate alignment
        let alignment = if self.config.validate_alignment && audio_metrics.is_some() {
            Some(self.validate_alignment(&sample.text, sample.audio.duration()))
        } else {
            None
        };

        if let Some(ref align) = alignment {
            if !align.is_aligned {
                warnings.push(format!(
                    "Poor audio-text alignment: {:.2} words/sec",
                    align.words_per_second
                ));
            }
            quality_scores.push(align.alignment_score);
        }

        // Calculate overall quality score
        let quality_score = if quality_scores.is_empty() {
            0.0
        } else {
            quality_scores.iter().sum::<f32>() / quality_scores.len() as f32
        };

        Ok(SampleValidationResult {
            sample_id: sample.id.clone(),
            is_valid: errors.is_empty(),
            audio_metrics,
            text_metrics,
            alignment,
            errors,
            warnings,
            quality_score,
        })
    }

    /// Validate multiple samples
    pub async fn validate_samples(
        &self,
        samples: &[DatasetSample],
    ) -> Result<Vec<SampleValidationResult>> {
        let mut results = Vec::new();

        for sample in samples {
            let result = self.validate_sample(sample).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Detect duplicates in samples
    pub fn detect_duplicates(&self, samples: &[DatasetSample]) -> Result<DuplicateDetectionResult> {
        let mut similarity_scores = HashMap::new();
        let mut duplicate_groups = Vec::new();
        let mut processed = HashSet::new();

        for (i, sample1) in samples.iter().enumerate() {
            if processed.contains(&sample1.id) {
                continue;
            }

            let mut group = vec![sample1.id.clone()];

            for (_j, sample2) in samples.iter().enumerate().skip(i + 1) {
                if processed.contains(&sample2.id) {
                    continue;
                }

                let similarity = self.calculate_text_similarity(&sample1.text, &sample2.text);
                similarity_scores.insert((sample1.id.clone(), sample2.id.clone()), similarity);

                if similarity >= self.config.duplicate_threshold {
                    group.push(sample2.id.clone());
                    processed.insert(sample2.id.clone());
                }
            }

            if group.len() > 1 {
                duplicate_groups.push(group);
            }
            processed.insert(sample1.id.clone());
        }

        // Generate removal recommendations (keep first sample in each group)
        let mut removal_recommendations = Vec::new();
        for group in &duplicate_groups {
            if group.len() > 1 {
                removal_recommendations.extend(group.iter().skip(1).cloned());
            }
        }

        Ok(DuplicateDetectionResult {
            duplicate_groups,
            similarity_scores,
            removal_recommendations,
        })
    }

    /// Validate text content
    fn validate_text(&self, text: &str, language: LanguageCode) -> Result<TextValidationMetrics> {
        let character_count = text.chars().count();
        let word_count = text.split_whitespace().count();
        let sentence_count = text
            .chars()
            .filter(|&ch| ch == '.' || ch == '!' || ch == '?')
            .count()
            .max(1);

        let mut invalid_characters = Vec::new();
        let mut encoding_issues = Vec::new();

        // Check character set
        if let Some(charset) = self
            .config
            .text_config
            .allowed_character_sets
            .get(&language)
        {
            for ch in text.chars() {
                if !charset.contains(ch) {
                    invalid_characters.push(ch);
                }
            }
        }

        // Check encoding
        if self.config.text_config.validate_encoding {
            // Check for common encoding issues
            if text.contains('\u{FFFD}') {
                // Replacement character
                encoding_issues
                    .push("Replacement character found - possible encoding issue".to_string());
            }

            // Check for control characters (except whitespace)
            for ch in text.chars() {
                if ch.is_control() && !ch.is_whitespace() {
                    encoding_issues.push(format!("Control character found: U+{:04X}", ch as u32));
                }
            }
        }

        // Calculate non-alphabetic ratio
        let alpha_count = text.chars().filter(|ch| ch.is_alphabetic()).count();
        let non_alpha_ratio = if character_count > 0 {
            1.0 - (alpha_count as f32 / character_count as f32)
        } else {
            0.0
        };

        // Remove duplicates from invalid characters
        invalid_characters.sort_unstable();
        invalid_characters.dedup();

        Ok(TextValidationMetrics {
            character_count,
            word_count,
            sentence_count,
            non_alpha_ratio,
            invalid_characters,
            encoding_issues,
        })
    }

    /// Compute audio quality metrics
    fn compute_audio_quality_metrics(&self, audio: &AudioData) -> Result<AudioQualityMetrics> {
        let samples = audio.samples();

        if samples.is_empty() {
            return Err(DatasetError::ValidationError("Empty audio".to_string()));
        }

        // Calculate basic metrics
        let peak_amplitude = samples
            .iter()
            .fold(0.0f32, |max, &sample| max.max(sample.abs()));
        let rms_amplitude =
            (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        // Calculate clipping
        let clipped_count = samples.iter().filter(|&&s| s.abs() > 0.99).count();
        let clipping = clipped_count as f32 / samples.len() as f32;

        // Calculate dynamic range (simplified)
        let mut sorted_samples: Vec<f32> = samples.iter().map(|&s| s.abs()).collect();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let percentile_95 = sorted_samples[(sorted_samples.len() as f32 * 0.95) as usize];
        let percentile_5 = sorted_samples[(sorted_samples.len() as f32 * 0.05) as usize];
        let dynamic_range = 20.0 * (percentile_95 / percentile_5.max(0.001)).log10();

        // Calculate zero crossing rate
        let zero_crossings = samples
            .windows(2)
            .filter(|window| (window[0] > 0.0) != (window[1] > 0.0))
            .count();
        let zero_crossing_rate = zero_crossings as f32 / (samples.len() - 1) as f32;

        // Calculate silence ratio (simplified)
        let silence_threshold = rms_amplitude * 0.1;
        let silent_samples = samples
            .iter()
            .filter(|&&s| s.abs() < silence_threshold)
            .count();
        let silence_ratio = silent_samples as f32 / samples.len() as f32;

        // SNR calculation (simplified - assumes noise is the quietest 10% of samples)
        let noise_level = sorted_samples[(sorted_samples.len() as f32 * 0.1) as usize];
        let signal_level = rms_amplitude;
        let snr = if noise_level > 0.0 {
            Some(20.0 * (signal_level / noise_level).log10())
        } else {
            None
        };

        // Calculate spectral features using AudioStats
        let audio_stats = AudioStats::calculate(audio);
        let spectral_centroid = audio_stats.spectral_centroid;
        let spectral_rolloff = audio_stats.spectral_rolloff;

        Ok(AudioQualityMetrics {
            snr,
            clipping,
            dynamic_range,
            rms_amplitude,
            peak_amplitude,
            zero_crossing_rate,
            silence_ratio,
            spectral_centroid,
            spectral_rolloff,
        })
    }

    /// Validate audio-text alignment
    fn validate_alignment(&self, text: &str, audio_duration: f32) -> AlignmentValidation {
        let word_count = text.split_whitespace().count();
        let words_per_second = if audio_duration > 0.0 {
            word_count as f32 / audio_duration
        } else {
            0.0
        };

        let expected_duration = word_count as f32 / self.config.expected_words_per_second;

        let duration_ratio = if expected_duration > 0.0 {
            audio_duration / expected_duration
        } else {
            1.0
        };

        let tolerance = self.config.alignment_tolerance;
        let is_aligned = ((1.0 / tolerance)..=tolerance).contains(&duration_ratio);

        // Calculate alignment score based on how close the ratio is to 1.0
        let alignment_score = if duration_ratio > 0.0 {
            1.0 / (1.0 + (duration_ratio - 1.0).abs())
        } else {
            0.0
        };

        AlignmentValidation {
            expected_duration,
            actual_duration: audio_duration,
            words_per_second,
            is_aligned,
            alignment_score,
        }
    }

    /// Calculate text similarity using simple character-based metrics
    fn calculate_text_similarity(&self, text1: &str, text2: &str) -> f32 {
        if text1 == text2 {
            return 1.0;
        }

        let chars1: Vec<char> = text1.to_lowercase().chars().collect();
        let chars2: Vec<char> = text2.to_lowercase().chars().collect();

        if chars1.is_empty() && chars2.is_empty() {
            return 1.0;
        }

        if chars1.is_empty() || chars2.is_empty() {
            return 0.0;
        }

        // Simple Levenshtein-based similarity
        let max_len = chars1.len().max(chars2.len());
        let distance = levenshtein_distance(&chars1, &chars2);

        1.0 - (distance as f32 / max_len as f32)
    }

    /// Calculate text quality score
    fn calculate_text_quality_score(&self, metrics: &TextValidationMetrics) -> f32 {
        let mut score: f32 = 1.0;

        // Penalize invalid characters
        if !metrics.invalid_characters.is_empty() {
            score *= 0.5;
        }

        // Penalize encoding issues
        if !metrics.encoding_issues.is_empty() {
            score *= 0.7;
        }

        // Penalize high non-alphabetic ratio
        if let Some(max_ratio) = self.config.text_config.max_non_alpha_ratio {
            if metrics.non_alpha_ratio > max_ratio {
                score *= 0.8;
            }
        }

        // Penalize length issues
        if let Some(min_len) = self.config.text_config.min_length {
            if metrics.character_count < min_len {
                score *= 0.6;
            }
        }

        if let Some(max_len) = self.config.text_config.max_length {
            if metrics.character_count > max_len {
                score *= 0.8;
            }
        }

        score.clamp(0.0, 1.0)
    }

    /// Calculate audio quality score
    fn calculate_audio_quality_score(&self, metrics: &AudioQualityMetrics) -> f32 {
        let mut score: f32 = 1.0;

        // Penalize high clipping
        if let Some(max_clipping) = self.config.audio_thresholds.max_clipping {
            if metrics.clipping > max_clipping {
                score *= 0.5;
            }
        }

        // Penalize low dynamic range
        if let Some(min_dr) = self.config.audio_thresholds.min_dynamic_range {
            if metrics.dynamic_range < min_dr {
                score *= 0.7;
            }
        }

        // Penalize low SNR
        if let Some(min_snr) = self.config.audio_thresholds.min_snr {
            if let Some(snr) = metrics.snr {
                if snr < min_snr {
                    score *= 0.6;
                }
            }
        }

        // Penalize high silence ratio
        if let Some(max_silence) = self.config.audio_thresholds.max_silence_ratio {
            if metrics.silence_ratio > max_silence {
                score *= 0.8;
            }
        }

        score.clamp(0.0, 1.0)
    }

    /// Get validation configuration
    pub fn config(&self) -> &ValidationConfig {
        &self.config
    }

    /// Update validation configuration
    pub fn set_config(&mut self, config: ValidationConfig) {
        self.config = config;
    }
}

impl Default for ProcessingValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate Levenshtein distance between two character sequences
fn levenshtein_distance(chars1: &[char], chars2: &[char]) -> usize {
    let len1 = chars1.len();
    let len2 = chars2.len();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    // Initialize first row and column
    #[allow(clippy::needless_range_loop)]
    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    // Fill the matrix
    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };

            matrix[i][j] = (matrix[i - 1][j] + 1) // deletion
                .min(matrix[i][j - 1] + 1) // insertion
                .min(matrix[i - 1][j - 1] + cost); // substitution
        }
    }

    matrix[len1][len2]
}

/// Validate audio quality against thresholds
pub fn validate_audio_quality(
    audio: &AudioData,
    thresholds: &AudioQualityThresholds,
) -> Result<bool> {
    let duration = audio.duration();
    let sample_rate = audio.sample_rate();
    let samples = audio.samples();

    // Check duration
    if let Some(min_duration) = thresholds.min_duration {
        if duration < min_duration {
            return Ok(false);
        }
    }

    if let Some(max_duration) = thresholds.max_duration {
        if duration > max_duration {
            return Ok(false);
        }
    }

    // Check sample rate
    if let Some(min_sample_rate) = thresholds.min_sample_rate {
        if sample_rate < min_sample_rate {
            return Ok(false);
        }
    }

    if let Some(max_sample_rate) = thresholds.max_sample_rate {
        if sample_rate > max_sample_rate {
            return Ok(false);
        }
    }

    // Check for clipping
    if let Some(max_clipping) = thresholds.max_clipping {
        let clipped_samples = samples.iter().filter(|&&x| x.abs() >= 0.99).count();
        let clipping_ratio = clipped_samples as f32 / samples.len() as f32;
        if clipping_ratio > max_clipping {
            return Ok(false);
        }
    }

    // Check dynamic range
    if let Some(min_dynamic_range) = thresholds.min_dynamic_range {
        let max_val = samples.iter().fold(0.0f32, |max, &x| max.max(x.abs()));
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();

        if rms > 0.0 {
            let dynamic_range_db = 20.0 * (max_val / rms).log10();
            if dynamic_range_db < min_dynamic_range {
                return Ok(false);
            }
        }
    }

    // Check silence ratio
    if let Some(max_silence_ratio) = thresholds.max_silence_ratio {
        let silence_threshold = 0.01; // -40 dB
        let silent_samples = samples
            .iter()
            .filter(|&&x| x.abs() <= silence_threshold)
            .count();
        let silence_ratio = silent_samples as f32 / samples.len() as f32;
        if silence_ratio > max_silence_ratio {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Validate text content against configuration
pub fn validate_text(
    text: &str,
    config: &TextValidationConfig,
    language: LanguageCode,
) -> Result<bool> {
    let text_length = text.len();

    // Check length
    if let Some(min_length) = config.min_length {
        if text_length < min_length {
            return Ok(false);
        }
    }

    if let Some(max_length) = config.max_length {
        if text_length > max_length {
            return Ok(false);
        }
    }

    // Check character set
    if let Some(charset) = config.allowed_character_sets.get(&language) {
        for ch in text.chars() {
            if !charset.contains(ch) {
                return Ok(false);
            }
        }
    }

    // Check non-alphabetic ratio
    if let Some(max_non_alpha_ratio) = config.max_non_alpha_ratio {
        let non_alpha_count = text
            .chars()
            .filter(|c| !c.is_alphabetic() && !c.is_whitespace())
            .count();
        let non_alpha_ratio = non_alpha_count as f32 / text.chars().count() as f32;
        if non_alpha_ratio > max_non_alpha_ratio {
            return Ok(false);
        }
    }

    // Check encoding
    if config.validate_encoding {
        // Simple UTF-8 validation - if we can iterate chars, it's valid UTF-8
        if text.chars().any(|c| c == char::REPLACEMENT_CHARACTER) {
            return Ok(false);
        }
    }

    // Check sentence structure
    if config.validate_sentence_structure {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(false);
        }

        // Basic sentence structure check - should start with capital letter and end with punctuation
        let first_char = trimmed.chars().next().unwrap_or(' ');
        let last_char = trimmed.chars().last().unwrap_or(' ');

        if !first_char.is_uppercase() || !".,!?;:".contains(last_char) {
            return Ok(false);
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_character_set_creation() {
        let charset = CharacterSet::new()
            .with_ascii_letters()
            .with_ascii_digits()
            .with_basic_punctuation()
            .with_whitespace();

        assert!(charset.contains('A'));
        assert!(charset.contains('z'));
        assert!(charset.contains('5'));
        assert!(charset.contains('.'));
        assert!(charset.contains(' '));
        assert!(!charset.contains('α')); // Greek letter
    }

    #[test]
    fn test_levenshtein_distance() {
        let chars1: Vec<char> = "hello".chars().collect();
        let chars2: Vec<char> = "world".chars().collect();
        let distance = levenshtein_distance(&chars1, &chars2);
        assert_eq!(distance, 4);

        let chars1: Vec<char> = "test".chars().collect();
        let chars2: Vec<char> = "test".chars().collect();
        let distance = levenshtein_distance(&chars1, &chars2);
        assert_eq!(distance, 0);
    }

    #[test]
    fn test_text_similarity() {
        let validator = ProcessingValidator::new();

        let similarity = validator.calculate_text_similarity("hello world", "hello world");
        assert_eq!(similarity, 1.0);

        let similarity = validator.calculate_text_similarity("hello", "world");
        assert!(similarity < 0.5);

        let similarity = validator.calculate_text_similarity("hello world", "hello earth");
        assert!(similarity > 0.5 && similarity < 1.0);
    }

    #[test]
    fn test_alignment_validation() {
        let validator = ProcessingValidator::new();

        // Normal speaking rate (2.5 words/sec)
        let alignment = validator.validate_alignment("hello world test", 1.2); // 3 words in 1.2 seconds
        assert!(alignment.is_aligned);
        assert!(alignment.words_per_second > 2.0 && alignment.words_per_second < 3.0);

        // Too fast
        let alignment = validator.validate_alignment("hello world test sample", 0.5); // 4 words in 0.5 seconds
        assert!(!alignment.is_aligned);

        // Too slow
        let alignment = validator.validate_alignment("hello", 5.0); // 1 word in 5 seconds
        assert!(!alignment.is_aligned);
    }

    #[tokio::test]
    async fn test_text_validation() {
        let validator = ProcessingValidator::new();

        let metrics = validator
            .validate_text("Hello, world!", LanguageCode::EnUs)
            .unwrap();
        assert_eq!(metrics.character_count, 13);
        assert_eq!(metrics.word_count, 2);
        assert!(metrics.invalid_characters.is_empty());
        assert!(metrics.encoding_issues.is_empty());

        // Test with invalid characters for English
        let metrics = validator
            .validate_text("Hello α world", LanguageCode::EnUs)
            .unwrap();
        assert!(!metrics.invalid_characters.is_empty());
        assert!(metrics.invalid_characters.contains(&'α'));
    }

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert!(config.audio_thresholds.min_snr.is_some());
        assert!(config.text_config.min_length.is_some());
        assert!(config.validate_alignment);
        assert!(config.check_duplicates);
        assert_eq!(config.expected_words_per_second, 2.5);
    }
}
