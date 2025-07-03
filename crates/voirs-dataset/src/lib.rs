//! # VoiRS Dataset Utilities
//! 
//! Dataset loading, preprocessing, and management utilities for training
//! and evaluation of VoiRS speech synthesis models.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Result type for dataset operations
pub type Result<T> = std::result::Result<T, DatasetError>;

/// Dataset-specific error types
#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Dataset loading failed: {0}")]
    LoadError(String),
    
    #[error("Invalid format: {0}")]
    FormatError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Audio processing error: {0}")]
    AudioError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    #[error("Preprocessing error: {0}")]
    PreprocessingError(String),
    
    #[error("Index out of bounds: {0}")]
    IndexError(usize),
    
    #[error("CSV error: {0}")]
    CsvError(#[from] csv::Error),
    
    #[error("Audio file error: {0}")]
    HoundError(#[from] hound::Error),
    
    #[error("JSON serialization error: {0}")]
    JsonError(#[from] serde_json::Error),
    
    #[error("Dataset split error: {0}")]
    SplitError(String),
    
    #[error("Processing error: {0}")]
    ProcessingError(String),
}

/// Language codes supported by VoiRS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LanguageCode {
    /// English (US)
    EnUs,
    /// English (UK)
    EnGb,
    /// Japanese
    Ja,
    /// Mandarin Chinese
    ZhCn,
    /// Korean
    Ko,
    /// German
    De,
    /// French
    Fr,
    /// Spanish
    Es,
}

impl LanguageCode {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            LanguageCode::EnUs => "en-US",
            LanguageCode::EnGb => "en-GB", 
            LanguageCode::Ja => "ja",
            LanguageCode::ZhCn => "zh-CN",
            LanguageCode::Ko => "ko",
            LanguageCode::De => "de",
            LanguageCode::Fr => "fr",
            LanguageCode::Es => "es",
        }
    }
}

/// A phoneme with its symbol and optional features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phoneme {
    /// Phoneme symbol (IPA or language-specific)
    pub symbol: String,
    /// Optional phoneme features
    pub features: Option<HashMap<String, String>>,
    /// Duration in seconds (if available)
    pub duration: Option<f32>,
}

impl Phoneme {
    /// Create new phoneme
    pub fn new<S: Into<String>>(symbol: S) -> Self {
        Self {
            symbol: symbol.into(),
            features: None,
            duration: None,
        }
    }
}

/// Audio data structure with efficient processing capabilities
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Audio samples (interleaved for multi-channel)
    samples: Vec<f32>,
    /// Sample rate in Hz
    sample_rate: u32,
    /// Number of channels
    channels: u32,
    /// Optional metadata
    metadata: HashMap<String, String>,
}

/// Audio buffer for holding PCM audio data (legacy compatibility)
pub type AudioBuffer = AudioData;

impl AudioData {
    /// Create new audio data
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u32) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
            metadata: HashMap::new(),
        }
    }
    
    /// Create silence
    pub fn silence(duration: f32, sample_rate: u32, channels: u32) -> Self {
        let num_samples = (duration * sample_rate as f32 * channels as f32) as usize;
        Self::new(vec![0.0; num_samples], sample_rate, channels)
    }
    
    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / (self.sample_rate * self.channels) as f32
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    
    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    
    /// Get number of channels
    pub fn channels(&self) -> u32 {
        self.channels
    }
    
    /// Get samples
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
    
    /// Get mutable samples
    pub fn samples_mut(&mut self) -> &mut [f32] {
        &mut self.samples
    }
    
    /// Get metadata
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
    
    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
    
    /// Resample audio to new sample rate
    pub fn resample(&self, new_sample_rate: u32) -> Result<AudioData> {
        // TODO: Implement high-quality resampling
        if new_sample_rate == self.sample_rate {
            return Ok(self.clone());
        }
        
        // Simple placeholder implementation
        let ratio = new_sample_rate as f32 / self.sample_rate as f32;
        let new_length = (self.samples.len() as f32 * ratio) as usize;
        let mut new_samples = Vec::with_capacity(new_length);
        
        for i in 0..new_length {
            let original_index = (i as f32 / ratio) as usize;
            if original_index < self.samples.len() {
                new_samples.push(self.samples[original_index]);
            } else {
                new_samples.push(0.0);
            }
        }
        
        Ok(AudioData::new(new_samples, new_sample_rate, self.channels))
    }
    
    /// Normalize audio amplitude
    pub fn normalize(&mut self) -> Result<()> {
        if self.samples.is_empty() {
            return Ok(());
        }
        
        let max_amplitude = self.samples.iter().fold(0.0f32, |max, &sample| {
            max.max(sample.abs())
        });
        
        if max_amplitude > 0.0 {
            let scale = 1.0 / max_amplitude;
            for sample in &mut self.samples {
                *sample *= scale;
            }
        }
        
        Ok(())
    }
    
    /// Calculate RMS (Root Mean Square) of the audio
    pub fn rms(&self) -> Option<f32> {
        if self.samples.is_empty() {
            return None;
        }
        
        let sum_squares: f32 = self.samples.iter().map(|&x| x * x).sum();
        let rms = (sum_squares / self.samples.len() as f32).sqrt();
        Some(rms)
    }
}

/// Audio file formats supported by VoiRS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioFormat {
    /// WAV format
    Wav,
    /// FLAC format
    Flac,
    /// MP3 format
    Mp3,
    /// OGG Vorbis format
    Ogg,
}

pub mod traits;
pub mod datasets;
pub mod audio;
pub mod processing;
pub mod augmentation;
pub mod quality;
pub mod export;
pub mod error;
pub mod utils;

// Legacy modules for backward compatibility
pub mod loaders;
pub mod preprocessors;
pub mod splits;
pub mod validation;
pub mod formats;

// Re-export split types for convenience
pub use splits::{SplitConfig, SplitStrategy, DatasetSplits, DatasetSplit, SplitStatistics};

/// Speaker information structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerInfo {
    /// Speaker identifier
    pub id: String,
    /// Speaker name (if available)
    pub name: Option<String>,
    /// Speaker gender
    pub gender: Option<String>,
    /// Speaker age
    pub age: Option<u32>,
    /// Speaker accent/region
    pub accent: Option<String>,
    /// Additional speaker metadata
    pub metadata: HashMap<String, String>,
}

/// Quality metrics for audio samples
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Signal-to-noise ratio in dB
    pub snr: Option<f32>,
    /// Clipping percentage
    pub clipping: Option<f32>,
    /// Dynamic range in dB
    pub dynamic_range: Option<f32>,
    /// Spectral quality score (0-1)
    pub spectral_quality: Option<f32>,
    /// Overall quality score (0-1)
    pub overall_quality: Option<f32>,
}

/// Dataset sample with comprehensive metadata
#[derive(Debug, Clone)]
pub struct DatasetSample {
    /// Unique identifier for this sample
    pub id: String,
    
    /// Original text
    pub text: String,
    
    /// Audio data
    pub audio: AudioData,
    
    /// Speaker information (if available)
    pub speaker: Option<SpeakerInfo>,
    
    /// Language of the text
    pub language: LanguageCode,
    
    /// Quality metrics
    pub quality: QualityMetrics,
    
    /// Phoneme sequence (if available)
    pub phonemes: Option<Vec<Phoneme>>,
    
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Dataset item containing text, phonemes, and audio (legacy compatibility)
pub type DatasetItem = DatasetSample;

impl DatasetSample {
    /// Create new dataset sample
    pub fn new(
        id: String,
        text: String,
        audio: AudioData,
        language: LanguageCode,
    ) -> Self {
        Self {
            id,
            text,
            audio,
            speaker: None,
            language,
            quality: QualityMetrics {
                snr: None,
                clipping: None,
                dynamic_range: None,
                spectral_quality: None,
                overall_quality: None,
            },
            phonemes: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Set phonemes
    pub fn with_phonemes(mut self, phonemes: Vec<Phoneme>) -> Self {
        self.phonemes = Some(phonemes);
        self
    }
    
    /// Set speaker information
    pub fn with_speaker(mut self, speaker: SpeakerInfo) -> Self {
        self.speaker = Some(speaker);
        self
    }
    
    /// Set quality metrics
    pub fn with_quality(mut self, quality: QualityMetrics) -> Self {
        self.quality = quality;
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.audio.duration()
    }
    
    /// Get speaker ID (for backward compatibility)
    pub fn speaker_id(&self) -> Option<&str> {
        self.speaker.as_ref().map(|s| s.id.as_str())
    }
}

/// Dataset trait for different dataset formats
pub trait Dataset {
    /// Get dataset name
    fn name(&self) -> &str;
    
    /// Get number of items in dataset
    fn len(&self) -> usize;
    
    /// Check if dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get item by index
    fn get_item(&self, index: usize) -> Result<DatasetItem>;
    
    /// Get all items (for small datasets)
    fn get_all_items(&self) -> Result<Vec<DatasetItem>> {
        (0..self.len())
            .map(|i| self.get_item(i))
            .collect()
    }
    
    /// Get dataset statistics
    fn statistics(&self) -> DatasetStatistics;
    
    /// Validate dataset
    fn validate(&self) -> Result<ValidationReport>;
}

/// Dataset statistics
#[derive(Debug, Clone)]
pub struct DatasetStatistics {
    /// Total number of items
    pub total_items: usize,
    
    /// Total duration in seconds
    pub total_duration: f32,
    
    /// Average duration per item
    pub average_duration: f32,
    
    /// Language distribution
    pub language_distribution: std::collections::HashMap<LanguageCode, usize>,
    
    /// Speaker distribution (if applicable)
    pub speaker_distribution: std::collections::HashMap<String, usize>,
    
    /// Text length statistics
    pub text_length_stats: LengthStatistics,
    
    /// Audio duration statistics
    pub duration_stats: DurationStatistics,
}

/// Length statistics for text
#[derive(Debug, Clone)]
pub struct LengthStatistics {
    pub min: usize,
    pub max: usize,
    pub mean: f32,
    pub median: usize,
    pub std_dev: f32,
}

/// Duration statistics for audio
#[derive(Debug, Clone)]
pub struct DurationStatistics {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub median: f32,
    pub std_dev: f32,
}

/// Dataset validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Whether the dataset is valid
    pub is_valid: bool,
    
    /// List of errors found
    pub errors: Vec<String>,
    
    /// List of warnings
    pub warnings: Vec<String>,
    
    /// Number of items validated
    pub items_validated: usize,
}

/// In-memory dataset implementation
pub struct MemoryDataset {
    name: String,
    items: Vec<DatasetItem>,
}

impl MemoryDataset {
    /// Create new in-memory dataset
    pub fn new(name: String) -> Self {
        Self {
            name,
            items: Vec::new(),
        }
    }
    
    /// Add item to dataset
    pub fn add_item(&mut self, item: DatasetItem) {
        self.items.push(item);
    }
    
    /// Add multiple items
    pub fn add_items(&mut self, items: Vec<DatasetItem>) {
        self.items.extend(items);
    }
    
    /// Clear all items
    pub fn clear(&mut self) {
        self.items.clear();
    }
}

impl Dataset for MemoryDataset {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn len(&self) -> usize {
        self.items.len()
    }
    
    fn get_item(&self, index: usize) -> Result<DatasetItem> {
        self.items
            .get(index)
            .cloned()
            .ok_or_else(|| DatasetError::ConfigError(format!("Dataset index {} out of bounds", index)))
    }
    
    fn statistics(&self) -> DatasetStatistics {
        if self.items.is_empty() {
            return DatasetStatistics {
                total_items: 0,
                total_duration: 0.0,
                average_duration: 0.0,
                language_distribution: std::collections::HashMap::new(),
                speaker_distribution: std::collections::HashMap::new(),
                text_length_stats: LengthStatistics {
                    min: 0,
                    max: 0,
                    mean: 0.0,
                    median: 0,
                    std_dev: 0.0,
                },
                duration_stats: DurationStatistics {
                    min: 0.0,
                    max: 0.0,
                    mean: 0.0,
                    median: 0.0,
                    std_dev: 0.0,
                },
            };
        }
        
        let total_items = self.items.len();
        let total_duration: f32 = self.items.iter().map(|item| item.duration()).sum();
        let average_duration = total_duration / total_items as f32;
        
        // Language distribution
        let mut language_distribution = std::collections::HashMap::new();
        for item in &self.items {
            *language_distribution.entry(item.language).or_insert(0) += 1;
        }
        
        // Speaker distribution
        let mut speaker_distribution = std::collections::HashMap::new();
        for item in &self.items {
            if let Some(speaker) = item.speaker_id() {
                *speaker_distribution.entry(speaker.to_string()).or_insert(0) += 1;
            }
        }
        
        // Text length statistics
        let text_lengths: Vec<usize> = self.items.iter().map(|item| item.text.len()).collect();
        let text_length_stats = calculate_length_stats(&text_lengths);
        
        // Duration statistics
        let durations: Vec<f32> = self.items.iter().map(|item| item.duration()).collect();
        let duration_stats = calculate_duration_stats(&durations);
        
        DatasetStatistics {
            total_items,
            total_duration,
            average_duration,
            language_distribution,
            speaker_distribution,
            text_length_stats,
            duration_stats,
        }
    }
    
    fn validate(&self) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        for (i, item) in self.items.iter().enumerate() {
            // Check for empty text
            if item.text.trim().is_empty() {
                errors.push(format!("Item {}: Empty text", i));
            }
            
            // Check for very short audio
            if item.duration() < 0.1 {
                warnings.push(format!("Item {}: Very short audio ({:.3}s)", i, item.duration()));
            }
            
            // Check for very long audio
            if item.duration() > 30.0 {
                warnings.push(format!("Item {}: Very long audio ({:.1}s)", i, item.duration()));
            }
            
            // Check for empty audio
            if item.audio.is_empty() {
                errors.push(format!("Item {}: Empty audio", i));
            }
        }
        
        Ok(ValidationReport {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            items_validated: self.items.len(),
        })
    }
}

/// Calculate length statistics
fn calculate_length_stats(values: &[usize]) -> LengthStatistics {
    if values.is_empty() {
        return LengthStatistics {
            min: 0,
            max: 0,
            mean: 0.0,
            median: 0,
            std_dev: 0.0,
        };
    }
    
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let sum: usize = values.iter().sum();
    let mean = sum as f32 / values.len() as f32;
    let median = sorted[sorted.len() / 2];
    
    let variance: f32 = values
        .iter()
        .map(|&x| (x as f32 - mean).powi(2))
        .sum::<f32>() / values.len() as f32;
    let std_dev = variance.sqrt();
    
    LengthStatistics {
        min,
        max,
        mean,
        median,
        std_dev,
    }
}

/// Calculate duration statistics
fn calculate_duration_stats(values: &[f32]) -> DurationStatistics {
    if values.is_empty() {
        return DurationStatistics {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
        };
    }
    
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let sum: f32 = values.iter().sum();
    let mean = sum / values.len() as f32;
    let median = sorted[sorted.len() / 2];
    
    let variance: f32 = values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / values.len() as f32;
    let std_dev = variance.sqrt();
    
    DurationStatistics {
        min,
        max,
        mean,
        median,
        std_dev,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LanguageCode;

    #[test]
    fn test_dataset_item_creation() {
        let audio = AudioBuffer::silence(1.0, 22050, 1);
        let item = DatasetItem::new(
            "test-001".to_string(),
            "Hello, world!".to_string(),
            audio,
            LanguageCode::EnUs,
        );
        
        assert_eq!(item.id, "test-001");
        assert_eq!(item.text, "Hello, world!");
        assert_eq!(item.language, LanguageCode::EnUs);
        assert!(item.phonemes.is_none());
        assert!(item.speaker_id().is_none());
    }

    #[test]
    fn test_memory_dataset() {
        let mut dataset = MemoryDataset::new("test-dataset".to_string());
        
        // Add test items
        for i in 0..3 {
            let audio = AudioBuffer::silence(1.0, 22050, 1);
            let item = DatasetItem::new(
                format!("item-{:03}", i),
                format!("Text number {}", i),
                audio,
                LanguageCode::EnUs,
            );
            dataset.add_item(item);
        }
        
        assert_eq!(dataset.name(), "test-dataset");
        assert_eq!(dataset.len(), 3);
        assert!(!dataset.is_empty());
        
        // Test item retrieval
        let item = dataset.get_item(1).unwrap();
        assert_eq!(item.id, "item-001");
        assert_eq!(item.text, "Text number 1");
        
        // Test statistics
        let stats = dataset.statistics();
        assert_eq!(stats.total_items, 3);
        assert!(stats.total_duration > 0.0);
        assert_eq!(stats.language_distribution[&LanguageCode::EnUs], 3);
        
        // Test validation
        let report = dataset.validate().unwrap();
        assert!(report.is_valid);
        assert_eq!(report.items_validated, 3);
    }
}