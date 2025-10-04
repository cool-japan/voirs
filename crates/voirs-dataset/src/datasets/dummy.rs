//! Dummy dataset for testing and development
//!
//! This module provides a dummy dataset that generates synthetic audio and text samples
//! for testing purposes. It's useful for pipeline testing without requiring real data.

use crate::traits::{Dataset, DatasetMetadata};
use crate::{
    AudioData, DatasetError, DatasetSample, DatasetStatistics, LanguageCode, QualityMetrics,
    Result, SpeakerInfo, ValidationReport,
};
use async_trait::async_trait;
use scirs2_core::random::{Random, Rng, SeedableRng};
use std::collections::HashMap;

/// Configuration for dummy dataset generation
#[derive(Debug, Clone)]
pub struct DummyConfig {
    /// Number of samples to generate
    pub num_samples: usize,
    /// Sample rate for audio generation
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u32,
    /// Language for samples
    pub language: LanguageCode,
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
    /// Minimum duration per sample (seconds)
    pub min_duration: f32,
    /// Maximum duration per sample (seconds)
    pub max_duration: f32,
    /// Audio generation type
    pub audio_type: AudioType,
    /// Text generation type
    pub text_type: TextType,
}

impl Default for DummyConfig {
    fn default() -> Self {
        Self {
            num_samples: 100,
            sample_rate: 22050,
            channels: 1,
            language: LanguageCode::EnUs,
            seed: None,
            min_duration: 1.0,
            max_duration: 5.0,
            audio_type: AudioType::SineWave,
            text_type: TextType::Lorem,
        }
    }
}

/// Types of synthetic audio generation
#[derive(Debug, Clone, Copy)]
pub enum AudioType {
    /// Pure sine wave
    SineWave,
    /// White noise
    WhiteNoise,
    /// Pink noise
    PinkNoise,
    /// Silence
    Silence,
    /// Mixed (combination of above)
    Mixed,
}

/// Types of text generation
#[derive(Debug, Clone, Copy)]
pub enum TextType {
    /// Lorem ipsum style text
    Lorem,
    /// Phonetic samples
    Phonetic,
    /// Simple counting
    Numbers,
    /// Random words
    RandomWords,
}

/// Dummy dataset for testing
pub struct DummyDataset {
    /// Dataset metadata
    metadata: DatasetMetadata,
    /// Generated samples
    samples: Vec<DatasetSample>,
    /// Configuration used for generation
    config: DummyConfig,
}

impl DummyDataset {
    /// Create new dummy dataset with default configuration
    pub fn new() -> Self {
        Self::with_config(DummyConfig::default())
    }

    /// Create dummy dataset with specific configuration
    pub fn with_config(config: DummyConfig) -> Self {
        let mut rng = if let Some(seed) = config.seed {
            Random::seed(seed)
        } else {
            Random::seed(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            )
        };

        let mut samples = Vec::with_capacity(config.num_samples);
        let mut total_duration = 0.0f32;

        for i in 0..config.num_samples {
            let duration = rng.gen_range(config.min_duration..=config.max_duration);
            total_duration += duration;

            let audio = Self::generate_audio(&config, duration, &mut rng);
            let text = Self::generate_text(&config, i, &mut rng);

            let speaker_id = i % 3;
            let sample = DatasetSample {
                id: format!("dummy_{i:06}"),
                text,
                audio,
                speaker: Some(SpeakerInfo {
                    id: format!("dummy_speaker_{speaker_id}"),
                    name: Some(format!("Dummy Speaker {speaker_id}")),
                    gender: Some(if i % 2 == 0 { "female" } else { "male" }.to_string()),
                    age: Some(25 + (i % 50) as u32),
                    accent: Some("synthetic".to_string()),
                    metadata: HashMap::new(),
                }),
                language: config.language,
                quality: QualityMetrics {
                    snr: Some(40.0 + rng.gen_range(-10.0..10.0)),
                    clipping: Some(rng.gen_range(0.0..0.1)),
                    dynamic_range: Some(30.0 + rng.gen_range(-5.0..15.0)),
                    spectral_quality: Some(rng.gen_range(0.7..1.0)),
                    overall_quality: Some(rng.gen_range(0.6..1.0)),
                },
                phonemes: None,
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("synthetic".to_string(), serde_json::Value::Bool(true));
                    meta.insert(
                        "audio_type".to_string(),
                        serde_json::Value::String(format!("{0:?}", config.audio_type)),
                    );
                    meta.insert(
                        "text_type".to_string(),
                        serde_json::Value::String(format!("{0:?}", config.text_type)),
                    );
                    meta.insert(
                        "sample_index".to_string(),
                        serde_json::Value::Number(i.into()),
                    );
                    meta
                },
            };

            samples.push(sample);
        }

        let metadata = DatasetMetadata {
            name: "DummyDataset".to_string(),
            version: "1.0.0".to_string(),
            description: Some("Synthetic dataset for testing and development".to_string()),
            total_samples: samples.len(),
            total_duration,
            languages: vec![config.language.as_str().to_string()],
            speakers: vec![
                "dummy_speaker_0".to_string(),
                "dummy_speaker_1".to_string(),
                "dummy_speaker_2".to_string(),
            ],
            license: Some("Public Domain".to_string()),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("synthetic".to_string(), serde_json::Value::Bool(true));
                meta.insert(
                    "audio_type".to_string(),
                    serde_json::Value::String(format!("{0:?}", config.audio_type)),
                );
                meta.insert(
                    "text_type".to_string(),
                    serde_json::Value::String(format!("{0:?}", config.text_type)),
                );
                meta.insert(
                    "sample_rate".to_string(),
                    serde_json::Value::Number(config.sample_rate.into()),
                );
                meta.insert(
                    "channels".to_string(),
                    serde_json::Value::Number(config.channels.into()),
                );
                meta
            },
        };

        Self {
            metadata,
            samples,
            config,
        }
    }

    /// Generate synthetic audio based on configuration
    fn generate_audio<R: scirs2_core::random::Rng>(
        config: &DummyConfig,
        duration: f32,
        rng: &mut R,
    ) -> AudioData {
        let num_samples = (duration * config.sample_rate as f32 * config.channels as f32) as usize;
        let mut samples = Vec::with_capacity(num_samples);

        match config.audio_type {
            AudioType::SineWave => {
                let frequency = rng.gen_range(200.0..800.0);
                let angular_frequency =
                    2.0 * std::f32::consts::PI * frequency / config.sample_rate as f32;

                for i in 0..num_samples {
                    let t = i as f32 / config.sample_rate as f32;
                    let amplitude = 0.5 * (angular_frequency * t).sin();
                    samples.push(amplitude);
                }
            }
            AudioType::WhiteNoise => {
                for _ in 0..num_samples {
                    samples.push(rng.gen_range(-0.5..0.5));
                }
            }
            AudioType::PinkNoise => {
                // Simple pink noise approximation
                let mut b = [0.0; 7];
                for _ in 0..num_samples {
                    let white = rng.gen_range(-1.0..1.0);
                    b[0] = 0.99886 * b[0] + white * 0.0555179;
                    b[1] = 0.99332 * b[1] + white * 0.0750759;
                    b[2] = 0.96900 * b[2] + white * 0.153_852;
                    b[3] = 0.86650 * b[3] + white * 0.3104856;
                    b[4] = 0.55000 * b[4] + white * 0.5329522;
                    b[5] = -0.7616 * b[5] - white * 0.0168980;
                    let pink = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + white * 0.5362;
                    b[6] = white * 0.115926;
                    samples.push(pink * 0.11);
                }
            }
            AudioType::Silence => {
                samples.resize(num_samples, 0.0);
            }
            AudioType::Mixed => {
                // Mix different audio types
                let segment_size = num_samples / 4;
                let mut pos = 0;

                // Sine wave segment
                let frequency = rng.gen_range(200.0..800.0);
                let angular_frequency =
                    2.0 * std::f32::consts::PI * frequency / config.sample_rate as f32;
                for i in 0..segment_size {
                    let t = (pos + i) as f32 / config.sample_rate as f32;
                    let amplitude = 0.3 * (angular_frequency * t).sin();
                    samples.push(amplitude);
                }
                pos += segment_size;

                // White noise segment
                for _ in 0..segment_size {
                    samples.push(rng.gen_range(-0.3..0.3));
                }
                pos += segment_size;

                // Silence segment
                samples.resize(samples.len() + segment_size, 0.0);
                pos += segment_size;

                // Fill remaining with sine wave
                for i in 0..(num_samples - pos) {
                    let t = (pos + i) as f32 / config.sample_rate as f32;
                    let amplitude = 0.4 * (angular_frequency * t * 2.0).sin();
                    samples.push(amplitude);
                }
            }
        }

        AudioData::new(samples, config.sample_rate, config.channels)
    }

    /// Generate synthetic text based on configuration
    fn generate_text<R: scirs2_core::random::Rng>(
        config: &DummyConfig,
        index: usize,
        rng: &mut R,
    ) -> String {
        match config.text_type {
            TextType::Lorem => {
                let lorem_words = vec![
                    "lorem",
                    "ipsum",
                    "dolor",
                    "sit",
                    "amet",
                    "consectetur",
                    "adipiscing",
                    "elit",
                    "sed",
                    "do",
                    "eiusmod",
                    "tempor",
                    "incididunt",
                    "ut",
                    "labore",
                    "et",
                    "dolore",
                    "magna",
                    "aliqua",
                    "enim",
                    "ad",
                    "minim",
                    "veniam",
                    "quis",
                    "nostrud",
                    "exercitation",
                    "ullamco",
                    "laboris",
                    "nisi",
                    "aliquip",
                    "ex",
                    "ea",
                    "commodo",
                    "consequat",
                    "duis",
                    "aute",
                    "irure",
                    "in",
                    "reprehenderit",
                    "voluptate",
                    "velit",
                    "esse",
                    "cillum",
                    "fugiat",
                    "nulla",
                    "pariatur",
                    "excepteur",
                    "sint",
                    "occaecat",
                    "cupidatat",
                    "non",
                    "proident",
                    "sunt",
                    "culpa",
                    "qui",
                    "officia",
                    "deserunt",
                    "mollit",
                    "anim",
                    "id",
                    "est",
                    "laborum",
                ];

                let num_words = rng.gen_range(5..20);
                let mut text = String::new();
                for i in 0..num_words {
                    if i > 0 {
                        text.push(' ');
                    }
                    let word = lorem_words[rng.gen_range(0..lorem_words.len())];
                    text.push_str(word);
                }
                text
            }
            TextType::Phonetic => {
                let phonetic_samples = [
                    "The quick brown fox jumps over the lazy dog.",
                    "Pack my box with five dozen liquor jugs.",
                    "How vexingly quick daft zebras jump!",
                    "Bright vixens jump; dozy fowl quack.",
                    "Sphinx of black quartz, judge my vow.",
                    "Two driven jocks help fax my big quiz.",
                    "Quick zephyrs blow, vexing daft Jim.",
                    "The five boxing wizards jump quickly.",
                    "Jackdaws love my big sphinx of quartz.",
                    "Mr. Jock, TV quiz PhD., bags few lynx.",
                ];

                phonetic_samples[index % phonetic_samples.len()].to_string()
            }
            TextType::Numbers => {
                let idx_plus_one = index + 1;
                let formats = [
                    format!("This is sample number {idx_plus_one}"),
                    format!("Number {idx_plus_one} in the sequence"),
                    format!("Item {idx_plus_one} of the dummy dataset"),
                    format!("Sample {idx_plus_one} generated for testing"),
                    format!("Audio clip number {idx_plus_one}"),
                ];

                formats[index % formats.len()].clone()
            }
            TextType::RandomWords => {
                let words = vec![
                    "hello",
                    "world",
                    "test",
                    "audio",
                    "speech",
                    "synthesis",
                    "voice",
                    "dataset",
                    "sample",
                    "text",
                    "processing",
                    "natural",
                    "language",
                    "machine",
                    "learning",
                    "artificial",
                    "intelligence",
                    "deep",
                    "neural",
                    "network",
                    "model",
                    "training",
                    "evaluation",
                    "performance",
                    "quality",
                    "metrics",
                    "validation",
                    "testing",
                    "development",
                    "research",
                    "science",
                    "technology",
                    "computer",
                    "software",
                    "algorithm",
                    "data",
                    "analysis",
                    "experiment",
                    "result",
                    "conclusion",
                ];

                let num_words = rng.gen_range(3..12);
                let mut text = String::new();
                for i in 0..num_words {
                    if i > 0 {
                        text.push(' ');
                    }
                    let word = words[rng.gen_range(0..words.len())];
                    text.push_str(word);
                }
                text
            }
        }
    }

    /// Get configuration used for generation
    pub fn config(&self) -> &DummyConfig {
        &self.config
    }

    /// Create a small dummy dataset for quick testing
    pub fn small() -> Self {
        Self::with_config(DummyConfig {
            num_samples: 10,
            min_duration: 0.5,
            max_duration: 2.0,
            ..Default::default()
        })
    }

    /// Create a large dummy dataset for stress testing
    pub fn large() -> Self {
        Self::with_config(DummyConfig {
            num_samples: 10000,
            min_duration: 1.0,
            max_duration: 10.0,
            ..Default::default()
        })
    }

    /// Create a reproducible dummy dataset with fixed seed
    pub fn reproducible(seed: u64) -> Self {
        Self::with_config(DummyConfig {
            seed: Some(seed),
            ..Default::default()
        })
    }
}

impl Default for DummyDataset {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Dataset for DummyDataset {
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
            if duration < 0.1 {
                warnings.push(format!("Sample {i}: Very short audio ({duration:.3}s)"));
            }

            // Check for very long audio
            if duration > 30.0 {
                warnings.push(format!("Sample {i}: Very long audio ({duration:.1}s)"));
            }

            // Check for empty audio
            if sample.audio.is_empty() {
                errors.push(format!("Sample {i}: Empty audio"));
            }

            // Check for valid sample rate
            if sample.audio.sample_rate() == 0 {
                errors.push(format!("Sample {i}: Invalid sample rate"));
            }

            // Check for valid channels
            if sample.audio.channels() == 0 {
                errors.push(format!("Sample {i}: Invalid channel count"));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::Dataset;

    #[tokio::test]
    async fn test_dummy_dataset_creation() {
        let dataset = DummyDataset::new();
        assert_eq!(dataset.len(), 100);
        assert!(!dataset.is_empty());

        let metadata = dataset.metadata();
        assert_eq!(metadata.name, "DummyDataset");
        assert_eq!(metadata.total_samples, 100);
        assert!(metadata.total_duration > 0.0);
    }

    #[tokio::test]
    async fn test_dummy_dataset_small() {
        let dataset = DummyDataset::small();
        assert_eq!(dataset.len(), 10);

        let sample = dataset.get(0).await.unwrap();
        assert!(!sample.id.is_empty());
        assert!(!sample.text.is_empty());
        assert!(sample.audio.duration() > 0.0);
    }

    #[tokio::test]
    async fn test_dummy_dataset_reproducible() {
        let dataset1 = DummyDataset::reproducible(42);
        let dataset2 = DummyDataset::reproducible(42);

        assert_eq!(dataset1.len(), dataset2.len());

        let sample1 = dataset1.get(0).await.unwrap();
        let sample2 = dataset2.get(0).await.unwrap();

        assert_eq!(sample1.id, sample2.id);
        assert_eq!(sample1.text, sample2.text);
        assert_eq!(sample1.audio.duration(), sample2.audio.duration());
    }

    #[tokio::test]
    async fn test_dummy_dataset_statistics() {
        let dataset = DummyDataset::small();
        let stats = dataset.statistics().await.unwrap();

        assert_eq!(stats.total_items, 10);
        assert!(stats.total_duration > 0.0);
        assert!(stats.average_duration > 0.0);
        assert!(!stats.language_distribution.is_empty());
        assert!(!stats.speaker_distribution.is_empty());
    }

    #[tokio::test]
    async fn test_dummy_dataset_validation() {
        let dataset = DummyDataset::small();
        let report = dataset.validate().await.unwrap();

        assert!(report.is_valid);
        assert_eq!(report.items_validated, 10);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn test_different_audio_types() {
        let types = vec![
            AudioType::SineWave,
            AudioType::WhiteNoise,
            AudioType::PinkNoise,
            AudioType::Silence,
            AudioType::Mixed,
        ];

        for audio_type in types {
            let config = DummyConfig {
                num_samples: 5,
                audio_type,
                ..Default::default()
            };

            let dataset = DummyDataset::with_config(config);
            assert_eq!(dataset.len(), 5);
        }
    }

    #[test]
    fn test_different_text_types() {
        let types = vec![
            TextType::Lorem,
            TextType::Phonetic,
            TextType::Numbers,
            TextType::RandomWords,
        ];

        for text_type in types {
            let config = DummyConfig {
                num_samples: 5,
                text_type,
                ..Default::default()
            };

            let dataset = DummyDataset::with_config(config);
            assert_eq!(dataset.len(), 5);
        }
    }
}
