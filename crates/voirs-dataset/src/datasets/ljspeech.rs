//! LJSpeech dataset implementation
//!
//! This module provides loading and processing capabilities for the LJSpeech dataset.

use crate::audio::io::load_audio;
use crate::splits::{DatasetSplit, DatasetSplits, SplitConfig, SplitStrategy};
use crate::traits::{Dataset, DatasetMetadata};
use crate::{
    AudioData, DatasetError, DatasetSample, DatasetStatistics, LanguageCode, NormalizationConfig,
    QualityMetrics, Result, SpeakerInfo, ValidationReport,
};
use async_trait::async_trait;
use csv::ReaderBuilder;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use tokio::fs;

/// LJSpeech dataset URL
const LJSPEECH_URL: &str = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2";

/// LJSpeech dataset loading options
#[derive(Debug, Clone)]
pub struct LjSpeechOptions {
    /// Audio normalization configuration
    pub normalization: Option<NormalizationConfig>,
    /// Target sample rate for audio resampling
    pub target_sample_rate: Option<u32>,
    /// Maximum duration filter (exclude longer samples)
    pub max_duration: Option<f32>,
    /// Minimum duration filter (exclude shorter samples)
    pub min_duration: Option<f32>,
    /// Apply quality filtering
    pub apply_quality_filter: bool,
}

impl Default for LjSpeechOptions {
    fn default() -> Self {
        Self {
            normalization: Some(NormalizationConfig::peak(0.9)),
            target_sample_rate: Some(22050),
            max_duration: Some(15.0),
            min_duration: Some(0.5),
            apply_quality_filter: true,
        }
    }
}

impl LjSpeechOptions {
    /// Create new options with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set normalization configuration
    pub fn with_normalization(mut self, config: NormalizationConfig) -> Self {
        self.normalization = Some(config);
        self
    }

    /// Disable normalization
    pub fn without_normalization(mut self) -> Self {
        self.normalization = None;
        self
    }

    /// Set target sample rate
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.target_sample_rate = Some(sample_rate);
        self
    }

    /// Set duration filters
    pub fn with_duration_filter(mut self, min_duration: f32, max_duration: f32) -> Self {
        self.min_duration = Some(min_duration);
        self.max_duration = Some(max_duration);
        self
    }

    /// Enable or disable quality filtering
    pub fn with_quality_filter(mut self, enabled: bool) -> Self {
        self.apply_quality_filter = enabled;
        self
    }
}

/// LJSpeech dataset loader
pub struct LjSpeechDataset {
    /// Dataset metadata
    metadata: DatasetMetadata,
    /// Dataset samples
    samples: Vec<DatasetSample>,
    /// Base path to dataset
    base_path: PathBuf,
    /// Loading options used
    options: LjSpeechOptions,
}

impl LjSpeechDataset {
    /// Load LJSpeech dataset from directory with default options
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::load_with_options(path, LjSpeechOptions::default()).await
    }

    /// Load LJSpeech dataset from directory with custom options
    pub async fn load_with_options<P: AsRef<Path>>(
        path: P,
        options: LjSpeechOptions,
    ) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();

        // Check if dataset exists, if not try to download
        if !base_path.exists() {
            return Err(DatasetError::LoadError(format!(
                "LJSpeech dataset not found at {base_path:?}. Use download() method first."
            )));
        }

        // Load metadata file
        let metadata_path = base_path.join("metadata.csv");
        if !metadata_path.exists() {
            return Err(DatasetError::LoadError(String::from(
                "metadata.csv not found in LJSpeech dataset",
            )));
        }

        // Parse metadata
        let mut samples = Vec::new();
        let mut csv_reader = ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b'|')
            .flexible(true) // Allow variable number of fields
            .quoting(false) // Disable quote handling (treat quotes as literal)
            .from_path(&metadata_path)?;

        for result in csv_reader.records() {
            let record = result?;
            if record.len() < 3 {
                tracing::debug!("Skipping record with {} fields", record.len());
                continue;
            }

            let file_id = record[0].to_string();
            let normalized_text = record[1].to_string();
            let original_text = record[2].to_string();

            // Use normalized text as it's cleaner
            let text = normalized_text;

            // Build audio file path
            let audio_path = base_path.join("wavs").join(format!("{file_id}.wav"));

            // Check if audio file exists
            if !audio_path.exists() {
                tracing::warn!("Audio file not found: {:?}", audio_path);
                continue;
            }

            // Load audio data
            let audio_data = match load_audio(&audio_path) {
                Ok(audio) => audio,
                Err(e) => {
                    tracing::warn!("Failed to load audio file {:?}: {}", audio_path, e);
                    continue;
                }
            };

            // Create sample
            let sample = DatasetSample {
                id: file_id.clone(),
                text,
                audio: audio_data,
                speaker: Some(SpeakerInfo {
                    id: String::from("linda_johnson"),
                    name: Some(String::from("Linda Johnson")),
                    gender: Some(String::from("female")),
                    age: None,
                    accent: Some(String::from("North American")),
                    metadata: HashMap::new(),
                }),
                language: LanguageCode::EnUs,
                quality: QualityMetrics {
                    snr: None,
                    clipping: None,
                    dynamic_range: None,
                    spectral_quality: None,
                    overall_quality: None,
                },
                phonemes: None,
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert(
                        String::from("original_text"),
                        serde_json::Value::String(original_text),
                    );
                    meta.insert(String::from("file_id"), serde_json::Value::String(file_id));
                    meta.insert(
                        String::from("audio_path"),
                        serde_json::Value::String(audio_path.to_string_lossy().to_string()),
                    );
                    meta
                },
            };

            samples.push(sample);
        }

        // Calculate total duration
        let total_duration: f32 = samples.iter().map(|s| s.audio.duration()).sum();

        // Create metadata
        let metadata = DatasetMetadata {
            name: "LJSpeech".to_string(),
            version: "1.1".to_string(),
            description: Some("Linda Johnson speech dataset for speech synthesis".to_string()),
            total_samples: samples.len(),
            total_duration,
            languages: vec!["en-US".to_string()],
            speakers: vec!["linda_johnson".to_string()],
            license: Some("Public Domain".to_string()),
            metadata: HashMap::new(),
        };

        Ok(Self {
            metadata,
            samples,
            base_path,
            options,
        })
    }

    /// Download LJSpeech dataset from official source
    pub async fn download<P: AsRef<Path>>(download_path: P) -> Result<()> {
        let download_path = download_path.as_ref();

        // Create download directory if it doesn't exist
        fs::create_dir_all(download_path).await?;

        // Download archive
        let response = reqwest::get(LJSPEECH_URL)
            .await
            .map_err(|e| DatasetError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(DatasetError::NetworkError(format!(
                "Failed to download LJSpeech dataset: HTTP {status}"
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| DatasetError::NetworkError(e.to_string()))?;

        // Extract archive
        let tar_bz2 = std::io::Cursor::new(bytes);
        let tar = bzip2::read::BzDecoder::new(tar_bz2);
        let mut archive = tar::Archive::new(tar);

        archive
            .unpack(download_path)
            .map_err(DatasetError::IoError)?;

        tracing::info!(
            "Successfully downloaded and extracted LJSpeech dataset to {:?}",
            download_path
        );
        Ok(())
    }

    /// Get sample by ID
    pub fn get_by_id(&self, id: &str) -> Option<&DatasetSample> {
        self.samples.iter().find(|s| s.id == id)
    }

    /// Filter samples by duration range
    pub fn filter_by_duration(&self, min_duration: f32, max_duration: f32) -> Vec<&DatasetSample> {
        self.samples
            .iter()
            .filter(|s| {
                let duration = s.audio.duration();
                (min_duration..=max_duration).contains(&duration)
            })
            .collect()
    }

    /// Get samples in a specific text length range
    pub fn filter_by_text_length(&self, min_chars: usize, max_chars: usize) -> Vec<&DatasetSample> {
        self.samples
            .iter()
            .filter(|s| {
                let len = s.text.chars().count();
                (min_chars..=max_chars).contains(&len)
            })
            .collect()
    }

    /// Get the base path of the dataset
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Create train/validation/test splits
    pub fn create_splits(&self, split_config: SplitConfig) -> Result<DatasetSplits> {
        match split_config.strategy {
            SplitStrategy::Random => self.create_random_splits(split_config),
            SplitStrategy::Stratified => self.create_stratified_splits(split_config),
            SplitStrategy::ByDuration => self.create_duration_splits(split_config),
            SplitStrategy::ByTextLength => self.create_text_length_splits(split_config),
        }
    }

    /// Create random splits
    fn create_random_splits(&self, config: SplitConfig) -> Result<DatasetSplits> {
        use scirs2_core::random::seq::SliceRandom;
        use scirs2_core::random::SeedableRng;

        let mut rng = if let Some(seed) = config.seed {
            scirs2_core::random::Random::seed(seed)
        } else {
            scirs2_core::random::Random::seed(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            )
        };

        let mut indices: Vec<usize> = (0..self.samples.len()).collect();
        rng.shuffle(&mut indices);

        self.split_by_indices(indices, config)
    }

    /// Create stratified splits (placeholder - treats all samples equally since LJSpeech has one speaker)
    fn create_stratified_splits(&self, config: SplitConfig) -> Result<DatasetSplits> {
        // Since LJSpeech has only one speaker, stratified is same as random
        self.create_random_splits(config)
    }

    /// Create splits balanced by duration
    fn create_duration_splits(&self, config: SplitConfig) -> Result<DatasetSplits> {
        use scirs2_core::random::seq::SliceRandom;

        let mut rng = if let Some(seed) = config.seed {
            scirs2_core::random::Random::seed(seed)
        } else {
            scirs2_core::random::Random::seed(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            )
        };

        // Sort by duration to ensure balanced distribution
        let mut indexed_samples: Vec<(usize, f32)> = self
            .samples
            .iter()
            .enumerate()
            .map(|(i, sample)| (i, sample.audio.duration()))
            .collect();

        indexed_samples.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Group into buckets and shuffle within buckets for better distribution
        let bucket_size = indexed_samples.len() / 10; // 10 buckets
        for chunk in indexed_samples.chunks_mut(bucket_size) {
            chunk.shuffle(&mut rng);
        }

        let indices: Vec<usize> = indexed_samples.into_iter().map(|(i, _)| i).collect();
        self.split_by_indices(indices, config)
    }

    /// Create splits balanced by text length
    fn create_text_length_splits(&self, config: SplitConfig) -> Result<DatasetSplits> {
        use scirs2_core::random::seq::SliceRandom;

        let mut rng = if let Some(seed) = config.seed {
            scirs2_core::random::Random::seed(seed)
        } else {
            scirs2_core::random::Random::seed(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            )
        };

        // Sort by text length to ensure balanced distribution
        let mut indexed_samples: Vec<(usize, usize)> = self
            .samples
            .iter()
            .enumerate()
            .map(|(i, sample)| (i, sample.text.chars().count()))
            .collect();

        indexed_samples.sort_by(|a, b| a.1.cmp(&b.1));

        // Group into buckets and shuffle within buckets
        let bucket_size = indexed_samples.len() / 10; // 10 buckets
        for chunk in indexed_samples.chunks_mut(bucket_size) {
            chunk.shuffle(&mut rng);
        }

        let indices: Vec<usize> = indexed_samples.into_iter().map(|(i, _)| i).collect();
        self.split_by_indices(indices, config)
    }

    /// Split dataset by given indices
    fn split_by_indices(&self, indices: Vec<usize>, config: SplitConfig) -> Result<DatasetSplits> {
        let total = indices.len();

        // Calculate split sizes
        let train_size = (total as f32 * config.train_ratio) as usize;
        let val_size = (total as f32 * config.val_ratio) as usize;
        let test_size = total - train_size - val_size;

        // Ensure we don't have empty splits (unless specifically requested)
        if train_size == 0 && config.train_ratio > 0.0 {
            return Err(DatasetError::SplitError(
                "Train split would be empty".to_string(),
            ));
        }
        if val_size == 0 && config.val_ratio > 0.0 {
            return Err(DatasetError::SplitError(
                "Validation split would be empty".to_string(),
            ));
        }
        if test_size == 0 && config.test_ratio > 0.0 {
            return Err(DatasetError::SplitError(
                "Test split would be empty".to_string(),
            ));
        }

        // Create splits
        let train_indices = indices[0..train_size].to_vec();
        let val_indices = indices[train_size..train_size + val_size].to_vec();
        let test_indices = indices[train_size + val_size..].to_vec();

        // Create split samples
        let train_samples = train_indices
            .iter()
            .map(|&i| self.samples[i].clone())
            .collect();
        let val_samples = val_indices
            .iter()
            .map(|&i| self.samples[i].clone())
            .collect();
        let test_samples = test_indices
            .iter()
            .map(|&i| self.samples[i].clone())
            .collect();

        Ok(DatasetSplits {
            train: DatasetSplit {
                samples: train_samples,
                indices: train_indices,
            },
            validation: DatasetSplit {
                samples: val_samples,
                indices: val_indices,
            },
            test: DatasetSplit {
                samples: test_samples,
                indices: test_indices,
            },
            config: config.clone(),
        })
    }
}

#[async_trait]
impl Dataset for LjSpeechDataset {
    type Sample = DatasetSample;

    fn len(&self) -> usize {
        self.samples.len()
    }

    async fn get(&self, index: usize) -> Result<Self::Sample> {
        self.samples
            .get(index)
            .cloned()
            .ok_or_else(|| DatasetError::IndexError(index))
    }

    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    async fn statistics(&self) -> Result<DatasetStatistics> {
        // Calculate text length statistics
        let text_lengths: Vec<usize> = self
            .samples
            .iter()
            .map(|s| s.text.chars().count())
            .collect();

        let text_length_stats = if text_lengths.is_empty() {
            crate::LengthStatistics {
                min: 0,
                max: 0,
                mean: 0.0,
                median: 0,
                std_dev: 0.0,
            }
        } else {
            let mut sorted_lengths = text_lengths.clone();
            sorted_lengths.sort_unstable();

            let min = sorted_lengths[0];
            let max = sorted_lengths[sorted_lengths.len() - 1];
            let sum: usize = text_lengths.iter().sum();
            let mean = sum as f32 / text_lengths.len() as f32;
            let median = sorted_lengths[sorted_lengths.len() / 2];

            let variance: f32 = text_lengths
                .iter()
                .map(|&x| (x as f32 - mean).powi(2))
                .sum::<f32>()
                / text_lengths.len() as f32;
            let std_dev = variance.sqrt();

            crate::LengthStatistics {
                min,
                max,
                mean,
                median,
                std_dev,
            }
        };

        // Calculate duration statistics
        let durations: Vec<f32> = self.samples.iter().map(|s| s.audio.duration()).collect();

        let duration_stats = if durations.is_empty() {
            crate::DurationStatistics {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
            }
        } else {
            let mut sorted_durations = durations.clone();
            sorted_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let min = sorted_durations[0];
            let max = sorted_durations[sorted_durations.len() - 1];
            let sum: f32 = durations.iter().sum();
            let mean = sum / durations.len() as f32;
            let median = sorted_durations[sorted_durations.len() / 2];

            let variance: f32 =
                durations.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / durations.len() as f32;
            let std_dev = variance.sqrt();

            crate::DurationStatistics {
                min,
                max,
                mean,
                median,
                std_dev,
            }
        };

        // Language and speaker distributions
        let mut language_distribution = HashMap::new();
        let mut speaker_distribution = HashMap::new();

        for sample in &self.samples {
            *language_distribution.entry(sample.language).or_insert(0) += 1;
            if let Some(speaker) = &sample.speaker {
                *speaker_distribution.entry(speaker.id.clone()).or_insert(0) += 1;
            }
        }

        Ok(DatasetStatistics {
            total_items: self.samples.len(),
            total_duration: self.samples.iter().map(|s| s.audio.duration()).sum(),
            average_duration: if self.samples.is_empty() {
                0.0
            } else {
                self.samples.iter().map(|s| s.audio.duration()).sum::<f32>()
                    / self.samples.len() as f32
            },
            language_distribution,
            speaker_distribution,
            text_length_stats,
            duration_stats,
        })
    }

    async fn validate(&self) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        for (i, sample) in self.samples.iter().enumerate() {
            // Check for empty text
            if sample.text.trim().is_empty() {
                errors.push(format!("Sample {i}: Empty text"));
            }

            // Check for very short audio
            let duration = sample.audio.duration();
            if duration < 0.5 {
                warnings.push(format!("Sample {i}: Very short audio ({duration:.3}s)"));
            }

            // Check for very long audio
            if duration > 15.0 {
                warnings.push(format!("Sample {i}: Very long audio ({duration:.1}s)"));
            }

            // Check for empty audio
            if sample.audio.is_empty() {
                errors.push(format!("Sample {i}: Empty audio"));
            }

            // Check for invalid characters in text
            if sample
                .text
                .chars()
                .any(|c| !c.is_ascii() && !c.is_whitespace())
            {
                warnings.push(format!("Sample {i}: Contains non-ASCII characters"));
            }

            // Check for audio file path existence
            if let Some(audio_path) = sample.metadata.get("audio_path") {
                if let Some(path_str) = audio_path.as_str() {
                    let path = PathBuf::from(path_str);
                    if !path.exists() {
                        errors.push(format!("Sample {i}: Audio file not found: {path:?}"));
                    }
                }
            }
        }

        Ok(ValidationReport {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            items_validated: self.samples.len(),
        })
    }
}

/// Load audio file from path
async fn load_audio_file(path: &Path) -> Result<AudioData> {
    let mut file = std::fs::File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Use hound to read WAV file
    let cursor = std::io::Cursor::new(buffer);
    let mut reader = hound::WavReader::new(cursor)
        .map_err(|e| DatasetError::AudioError(format!("Failed to read WAV file: {e}")))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as u32;

    // Read samples
    let samples: Result<Vec<f32>> = match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => {
                let int_samples: Vec<i16> = reader
                    .samples::<i16>()
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| {
                        DatasetError::AudioError(format!("Failed to read samples: {e}"))
                    })?;

                Ok(int_samples
                    .into_iter()
                    .map(|s| s as f32 / i16::MAX as f32)
                    .collect())
            }
            24 => {
                let int_samples: Vec<i32> = reader
                    .samples::<i32>()
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| {
                        DatasetError::AudioError(format!("Failed to read samples: {e}"))
                    })?;

                Ok(int_samples
                    .into_iter()
                    .map(|s| s as f32 / ((1 << 23) as f32))
                    .collect())
            }
            32 => {
                let int_samples: Vec<i32> = reader
                    .samples::<i32>()
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| {
                        DatasetError::AudioError(format!("Failed to read samples: {e}"))
                    })?;

                Ok(int_samples
                    .into_iter()
                    .map(|s| s as f32 / i32::MAX as f32)
                    .collect())
            }
            _ => {
                let bits_per_sample = spec.bits_per_sample;
                Err(DatasetError::AudioError(format!(
                    "Unsupported bit depth: {bits_per_sample}"
                )))
            }
        },
        hound::SampleFormat::Float => {
            let float_samples: Vec<f32> = reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| DatasetError::AudioError(format!("Failed to read samples: {e}")))?;

            Ok(float_samples)
        }
    };

    let samples = samples?;
    Ok(AudioData::new(samples, sample_rate, channels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::dummy::DummyDataset;
    use tempfile::TempDir;

    /// Test dataset splitting functionality with dummy data
    #[test]
    fn test_ljspeech_splits() {
        // Create a dummy dataset to test splitting logic
        let _dummy = DummyDataset::small();

        // Simulate LJSpeech structure by creating samples
        let samples: Vec<DatasetSample> = (0..100)
            .map(|i| DatasetSample {
                id: format!("LJ{:03}-{:04}", (i / 100) + 1, i % 100),
                text: format!("Test sentence number {i}"),
                audio: crate::AudioData::new(vec![0.1; 1000], 22050, 1),
                speaker: Some(SpeakerInfo {
                    id: String::from("linda_johnson"),
                    name: Some(String::from("Linda Johnson")),
                    gender: Some(String::from("female")),
                    age: None,
                    accent: Some(String::from("North American")),
                    metadata: HashMap::new(),
                }),
                language: LanguageCode::EnUs,
                quality: QualityMetrics {
                    snr: None,
                    clipping: None,
                    dynamic_range: None,
                    spectral_quality: None,
                    overall_quality: None,
                },
                phonemes: None,
                metadata: HashMap::new(),
            })
            .collect();

        let dataset = LjSpeechDataset {
            metadata: DatasetMetadata {
                name: "LJSpeech-Test".to_string(),
                version: "1.1".to_string(),
                description: Some("Test dataset".to_string()),
                total_samples: samples.len(),
                total_duration: samples.iter().map(|s| s.audio.duration()).sum(),
                languages: vec!["en-US".to_string()],
                speakers: vec!["linda_johnson".to_string()],
                license: Some("Public Domain".to_string()),
                metadata: HashMap::new(),
            },
            samples,
            base_path: std::path::PathBuf::from("/tmp"),
            options: LjSpeechOptions::default(),
        };

        // Test default split (80/10/10)
        let config = SplitConfig::default_split().with_seed(42);
        let splits = dataset.create_splits(config).unwrap();

        assert_eq!(splits.total_samples(), 100);
        assert_eq!(splits.train.len(), 80);
        assert_eq!(splits.validation.len(), 10);
        assert_eq!(splits.test.len(), 10);

        // Verify no overlap
        assert!(splits.validate_no_overlap());

        // Test reproducibility
        let config2 = SplitConfig::default_split().with_seed(42);
        let splits2 = dataset.create_splits(config2).unwrap();

        assert_eq!(splits.train.indices, splits2.train.indices);
        assert_eq!(splits.validation.indices, splits2.validation.indices);
        assert_eq!(splits.test.indices, splits2.test.indices);
    }

    #[test]
    fn test_duration_balanced_splits() {
        // Create samples with varying durations
        let samples: Vec<DatasetSample> = (0..50)
            .map(|i| {
                let duration_samples = if i < 10 {
                    1000 // Short samples
                } else if i < 40 {
                    2000 // Medium samples
                } else {
                    4000 // Long samples
                };

                DatasetSample {
                    id: format!("test-{i:03}"),
                    text: format!("Test sentence {i}"),
                    audio: crate::AudioData::new(vec![0.1; duration_samples], 22050, 1),
                    speaker: Some(SpeakerInfo {
                        id: "speaker".to_string(),
                        name: None,
                        gender: None,
                        age: None,
                        accent: None,
                        metadata: HashMap::new(),
                    }),
                    language: LanguageCode::EnUs,
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
            })
            .collect();

        let dataset = LjSpeechDataset {
            metadata: DatasetMetadata {
                name: "Test".to_string(),
                version: "1.0".to_string(),
                description: None,
                total_samples: samples.len(),
                total_duration: samples.iter().map(|s| s.audio.duration()).sum(),
                languages: vec!["en-US".to_string()],
                speakers: vec!["speaker".to_string()],
                license: None,
                metadata: HashMap::new(),
            },
            samples,
            base_path: std::path::PathBuf::from("/tmp"),
            options: LjSpeechOptions::default(),
        };

        // Test duration-balanced split
        let config = SplitConfig::new(0.8, 0.1, 0.1, SplitStrategy::ByDuration)
            .unwrap()
            .with_seed(123);
        let splits = dataset.create_splits(config).unwrap();

        assert_eq!(splits.total_samples(), 50);
        assert!(splits.validate_no_overlap());

        // Check that duration is somewhat balanced
        let stats = splits.statistics();
        let total_duration = stats.total_duration();
        let train_ratio = stats.train_duration / total_duration;
        let val_ratio = stats.validation_duration / total_duration;
        let test_ratio = stats.test_duration / total_duration;

        // Allow for some variance due to discrete sample allocation
        assert!(
            (train_ratio - 0.8).abs() < 0.2,
            "Train duration ratio: {train_ratio}"
        );
        assert!(
            (val_ratio - 0.1).abs() < 0.1,
            "Val duration ratio: {val_ratio}"
        );
        assert!(
            (test_ratio - 0.1).abs() < 0.1,
            "Test duration ratio: {test_ratio}"
        );
    }

    #[test]
    fn test_split_config_validation() {
        // Test invalid ratios
        assert!(SplitConfig::new(1.5, 0.1, 0.1, SplitStrategy::Random).is_err());
        assert!(SplitConfig::new(0.8, 0.1, 0.2, SplitStrategy::Random).is_err()); // Sum > 1.0
        assert!(SplitConfig::new(0.7, 0.1, 0.1, SplitStrategy::Random).is_err()); // Sum < 1.0

        // Test valid ratios
        assert!(SplitConfig::new(0.8, 0.1, 0.1, SplitStrategy::Random).is_ok());
        assert!(SplitConfig::new(0.9, 0.1, 0.0, SplitStrategy::Random).is_ok());
    }

    #[test]
    fn test_split_save_load() {
        let temp_dir = TempDir::new().unwrap();

        // Create a small dataset for testing
        let samples: Vec<DatasetSample> = (0..10)
            .map(|i| DatasetSample {
                id: format!("test-{i:03}"),
                text: format!("Text {i}"),
                audio: crate::AudioData::new(vec![0.1; 100], 22050, 1),
                speaker: None,
                language: LanguageCode::EnUs,
                quality: QualityMetrics {
                    snr: None,
                    clipping: None,
                    dynamic_range: None,
                    spectral_quality: None,
                    overall_quality: None,
                },
                phonemes: None,
                metadata: HashMap::new(),
            })
            .collect();

        let dataset = LjSpeechDataset {
            metadata: DatasetMetadata {
                name: "Test".to_string(),
                version: "1.0".to_string(),
                description: None,
                total_samples: samples.len(),
                total_duration: samples.iter().map(|s| s.audio.duration()).sum(),
                languages: vec!["en-US".to_string()],
                speakers: vec![],
                license: None,
                metadata: HashMap::new(),
            },
            samples,
            base_path: std::path::PathBuf::from("/tmp"),
            options: LjSpeechOptions::default(),
        };

        let config = SplitConfig::new(0.8, 0.2, 0.0, SplitStrategy::Random)
            .unwrap()
            .with_seed(456);
        let splits = dataset.create_splits(config).unwrap();

        // Save splits
        splits.save_indices(temp_dir.path()).unwrap();

        // Load splits
        let (train_indices, val_indices, test_indices, loaded_config) =
            DatasetSplits::load_indices(temp_dir.path()).unwrap();

        assert_eq!(splits.train.indices, train_indices);
        assert_eq!(splits.validation.indices, val_indices);
        assert_eq!(splits.test.indices, test_indices);
        assert_eq!(splits.config.seed, loaded_config.seed);
    }
}
