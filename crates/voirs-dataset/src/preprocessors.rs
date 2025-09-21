//! Dataset preprocessing utilities.
//!
//! This module provides comprehensive preprocessing functions for dataset samples
//! including text normalization, audio processing, quality filtering, and duration filtering.

use crate::audio::processing::{
    AudioProcessingPipeline, ProcessingConfig as AudioProcessingConfig,
};
use crate::quality::metrics::{QualityConfig, QualityMetricsCalculator};
use crate::{DatasetError, DatasetSample, Result};
use serde::{Deserialize, Serialize};

/// Dataset item type alias for backward compatibility
pub type DatasetItem = DatasetSample;

/// Text preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPreprocessingConfig {
    /// Enable text normalization
    pub normalize: bool,
    /// Convert to lowercase
    pub to_lowercase: bool,
    /// Remove extra whitespace
    pub remove_extra_whitespace: bool,
    /// Normalize unicode characters
    pub normalize_unicode: bool,
    /// Expand contractions
    pub expand_contractions: bool,
    /// Expand abbreviations
    pub expand_abbreviations: bool,
    /// Remove control characters
    pub remove_control_chars: bool,
    /// Minimum text length (characters)
    pub min_length: Option<usize>,
    /// Maximum text length (characters)
    pub max_length: Option<usize>,
}

impl Default for TextPreprocessingConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            to_lowercase: false,
            remove_extra_whitespace: true,
            normalize_unicode: true,
            expand_contractions: true,
            expand_abbreviations: true,
            remove_control_chars: true,
            min_length: Some(5),
            max_length: Some(1000),
        }
    }
}

/// Audio preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioPreprocessingConfig {
    /// Target sample rate
    pub target_sample_rate: Option<u32>,
    /// Normalize audio amplitude
    pub normalize: bool,
    /// Trim silence from beginning and end
    pub trim_silence: bool,
    /// Silence threshold for trimming
    pub silence_threshold: f32,
    /// Convert to mono
    pub to_mono: bool,
    /// Apply fade in/out
    pub apply_fade: bool,
    /// Minimum audio duration (seconds)
    pub min_duration: Option<f32>,
    /// Maximum audio duration (seconds)
    pub max_duration: Option<f32>,
}

impl Default for AudioPreprocessingConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: Some(22050),
            normalize: true,
            trim_silence: true,
            silence_threshold: 0.01,
            to_mono: true,
            apply_fade: false,
            min_duration: Some(0.5),
            max_duration: Some(30.0),
        }
    }
}

/// Quality filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityFilterConfig {
    /// Enable quality filtering
    pub enabled: bool,
    /// Minimum SNR (dB)
    pub min_snr: Option<f32>,
    /// Maximum clipping percentage
    pub max_clipping: Option<f32>,
    /// Minimum dynamic range (dB)
    pub min_dynamic_range: Option<f32>,
    /// Minimum overall quality score (0.0-1.0)
    pub min_quality_score: Option<f32>,
}

impl Default for QualityFilterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_snr: Some(20.0),
            max_clipping: Some(1.0),
            min_dynamic_range: Some(30.0),
            min_quality_score: Some(0.6),
        }
    }
}

/// Comprehensive preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Text preprocessing configuration
    pub text: TextPreprocessingConfig,
    /// Audio preprocessing configuration
    pub audio: AudioPreprocessingConfig,
    /// Quality filtering configuration
    pub quality: QualityFilterConfig,
    /// Enable parallel processing
    pub parallel: bool,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            text: TextPreprocessingConfig::default(),
            audio: AudioPreprocessingConfig::default(),
            quality: QualityFilterConfig::default(),
            parallel: true,
        }
    }
}

/// Text preprocessor
pub struct TextPreprocessor {
    config: TextPreprocessingConfig,
}

impl TextPreprocessor {
    /// Create new text preprocessor with default configuration
    pub fn new() -> Self {
        Self {
            config: TextPreprocessingConfig::default(),
        }
    }

    /// Create text preprocessor with custom configuration
    pub fn with_config(config: TextPreprocessingConfig) -> Self {
        Self { config }
    }

    /// Preprocess text in dataset item
    pub fn preprocess(&self, item: &mut DatasetSample) -> Result<()> {
        if !self.config.normalize {
            return Ok(());
        }

        let mut text = item.text.clone();

        // Length filtering (before processing)
        if let Some(min_len) = self.config.min_length {
            if text.len() < min_len {
                return Err(DatasetError::ValidationError(format!(
                    "Text too short: {} characters (minimum: {})",
                    text.len(),
                    min_len
                )));
            }
        }

        if let Some(max_len) = self.config.max_length {
            if text.len() > max_len {
                return Err(DatasetError::ValidationError(format!(
                    "Text too long: {} characters (maximum: {})",
                    text.len(),
                    max_len
                )));
            }
        }

        // Basic cleaning
        text = text.trim().to_string();

        // Remove extra whitespace
        if self.config.remove_extra_whitespace {
            text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        }

        // Unicode normalization
        if self.config.normalize_unicode {
            text = text
                .replace(['"', '"', '"'], "\"")
                .replace(['\u{2018}', '\u{2019}'], "'")
                .replace(['`'], "'")
                .replace(['—', '–'], "-")
                .replace("--", "-")
                .replace(['…'], "...")
                .replace(['•'], "*");
        }

        // Remove control characters
        if self.config.remove_control_chars {
            text = text
                .chars()
                .filter(|&c| !c.is_control() || c == '\n' || c == '\t')
                .collect();
        }

        // Expand contractions
        if self.config.expand_contractions {
            text = self.expand_contractions(&text);
        }

        // Expand abbreviations
        if self.config.expand_abbreviations {
            text = self.expand_abbreviations(&text);
        }

        // Convert to lowercase
        if self.config.to_lowercase {
            text = text.to_lowercase();
        }

        // Final cleanup
        text = text.trim().to_string();

        item.text = text;
        Ok(())
    }

    /// Expand common contractions
    fn expand_contractions(&self, text: &str) -> String {
        let mut result = text.to_string();

        let contractions = [
            ("can't", "cannot"),
            ("won't", "will not"),
            ("n't", " not"),
            ("'re", " are"),
            ("'ve", " have"),
            ("'ll", " will"),
            ("'d", " would"),
            ("'m", " am"),
            ("'s", " is"),
        ];

        for (contraction, expansion) in contractions {
            result = result.replace(contraction, expansion);
        }

        result
    }

    /// Expand common abbreviations
    fn expand_abbreviations(&self, text: &str) -> String {
        let mut result = text.to_string();

        let abbreviations = [
            ("Dr.", "Doctor"),
            ("Mr.", "Mister"),
            ("Mrs.", "Missus"),
            ("Ms.", "Miss"),
            ("Prof.", "Professor"),
            ("St.", "Street"),
            ("Ave.", "Avenue"),
            ("Blvd.", "Boulevard"),
            ("etc.", "et cetera"),
            ("vs.", "versus"),
            ("e.g.", "for example"),
            ("i.e.", "that is"),
        ];

        for (abbr, expansion) in abbreviations {
            result = result.replace(abbr, expansion);
        }

        result
    }
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Audio preprocessor
pub struct AudioPreprocessor {
    config: AudioPreprocessingConfig,
    pipeline: AudioProcessingPipeline,
}

impl AudioPreprocessor {
    /// Create new audio preprocessor with default configuration
    pub fn new() -> Self {
        let config = AudioPreprocessingConfig::default();
        let pipeline = AudioProcessingPipeline::new(AudioProcessingConfig {
            target_sample_rate: config.target_sample_rate,
            normalize: config.normalize,
            trim_silence: config.trim_silence,
            silence_threshold: config.silence_threshold,
            to_mono: config.to_mono,
            apply_fade: config.apply_fade,
            fade_in_duration: 0.1,
            fade_out_duration: 0.1,
        });

        Self { config, pipeline }
    }

    /// Create audio preprocessor with custom configuration
    pub fn with_config(config: AudioPreprocessingConfig) -> Self {
        let pipeline = AudioProcessingPipeline::new(AudioProcessingConfig {
            target_sample_rate: config.target_sample_rate,
            normalize: config.normalize,
            trim_silence: config.trim_silence,
            silence_threshold: config.silence_threshold,
            to_mono: config.to_mono,
            apply_fade: config.apply_fade,
            fade_in_duration: 0.1,
            fade_out_duration: 0.1,
        });

        Self { config, pipeline }
    }

    /// Preprocess audio in dataset item
    pub fn preprocess(&self, item: &mut DatasetSample) -> Result<()> {
        // Duration filtering (before processing)
        let duration = item.audio.duration();

        if let Some(min_duration) = self.config.min_duration {
            if duration < min_duration {
                return Err(DatasetError::ValidationError(format!(
                    "Audio too short: {duration:.2}s (minimum: {min_duration:.2}s)"
                )));
            }
        }

        if let Some(max_duration) = self.config.max_duration {
            if duration > max_duration {
                return Err(DatasetError::ValidationError(format!(
                    "Audio too long: {duration:.2}s (maximum: {max_duration:.2}s)"
                )));
            }
        }

        // Process audio through pipeline
        item.audio = self.pipeline.process(&item.audio)?;

        Ok(())
    }
}

impl Default for AudioPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Quality filter for dataset samples
pub struct QualityFilter {
    config: QualityFilterConfig,
    metrics_calculator: QualityMetricsCalculator,
}

impl QualityFilter {
    /// Create new quality filter with default configuration
    pub fn new() -> Self {
        Self {
            config: QualityFilterConfig::default(),
            metrics_calculator: QualityMetricsCalculator::new(QualityConfig::default()),
        }
    }

    /// Create quality filter with custom configuration
    pub fn with_config(config: QualityFilterConfig) -> Self {
        Self {
            config,
            metrics_calculator: QualityMetricsCalculator::new(QualityConfig::default()),
        }
    }

    /// Filter sample based on quality metrics
    pub fn should_keep(&self, item: &DatasetSample) -> Result<bool> {
        if !self.config.enabled {
            return Ok(true);
        }

        let metrics = self.metrics_calculator.calculate_metrics(&item.audio)?;

        // Check SNR
        if let Some(min_snr) = self.config.min_snr {
            if metrics.snr < min_snr {
                return Ok(false);
            }
        }

        // Check clipping
        if let Some(max_clipping) = self.config.max_clipping {
            if metrics.clipping_percentage > max_clipping {
                return Ok(false);
            }
        }

        // Check dynamic range
        if let Some(min_dynamic_range) = self.config.min_dynamic_range {
            if metrics.dynamic_range < min_dynamic_range {
                return Ok(false);
            }
        }

        // Check overall quality score
        if let Some(min_quality_score) = self.config.min_quality_score {
            if metrics.overall_score < min_quality_score {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

impl Default for QualityFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive dataset preprocessor
pub struct DatasetPreprocessor {
    config: PreprocessingConfig,
    text_preprocessor: TextPreprocessor,
    audio_preprocessor: AudioPreprocessor,
    quality_filter: QualityFilter,
}

impl DatasetPreprocessor {
    /// Create new dataset preprocessor with default configuration
    pub fn new() -> Self {
        let config = PreprocessingConfig::default();
        Self {
            text_preprocessor: TextPreprocessor::with_config(config.text.clone()),
            audio_preprocessor: AudioPreprocessor::with_config(config.audio.clone()),
            quality_filter: QualityFilter::with_config(config.quality.clone()),
            config,
        }
    }

    /// Create dataset preprocessor with custom configuration
    pub fn with_config(config: PreprocessingConfig) -> Self {
        Self {
            text_preprocessor: TextPreprocessor::with_config(config.text.clone()),
            audio_preprocessor: AudioPreprocessor::with_config(config.audio.clone()),
            quality_filter: QualityFilter::with_config(config.quality.clone()),
            config,
        }
    }

    /// Preprocess a single sample
    pub fn preprocess_sample(&self, item: &mut DatasetSample) -> Result<bool> {
        // Quality filtering (before preprocessing)
        if !self.quality_filter.should_keep(item)? {
            return Ok(false);
        }

        // Text preprocessing
        self.text_preprocessor.preprocess(item)?;

        // Audio preprocessing
        self.audio_preprocessor.preprocess(item)?;

        // Quality filtering (after preprocessing)
        self.quality_filter.should_keep(item)
    }

    /// Preprocess multiple samples
    pub fn preprocess_samples(&self, items: &mut Vec<DatasetSample>) -> Result<Vec<DatasetSample>> {
        if self.config.parallel {
            use rayon::prelude::*;

            let results: Result<Vec<_>> = items
                .par_iter_mut()
                .map(|item| {
                    self.preprocess_sample(item)
                        .map(|keep| if keep { Some(item.clone()) } else { None })
                })
                .collect();

            let processed: Vec<DatasetSample> = results?.into_iter().flatten().collect();

            Ok(processed)
        } else {
            let mut processed = Vec::new();

            for item in items.iter_mut() {
                if self.preprocess_sample(item)? {
                    processed.push(item.clone());
                }
            }

            Ok(processed)
        }
    }
}

impl Default for DatasetPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}
