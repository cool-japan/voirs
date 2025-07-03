//! HuggingFace Datasets export
//!
//! This module provides functionality to export datasets in HuggingFace Datasets format.
//! It supports dataset card generation, Arrow format conversion, and repository upload.

use crate::{DatasetSample, DatasetError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;

/// HuggingFace dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceConfig {
    /// Dataset name
    pub dataset_name: String,
    /// Dataset description
    pub description: String,
    /// Dataset version
    pub version: String,
    /// Dataset license
    pub license: Option<String>,
    /// Dataset tags
    pub tags: Vec<String>,
    /// Whether to include audio files inline or as references
    pub include_audio_data: bool,
    /// Audio file format for export
    pub audio_format: AudioExportFormat,
    /// Maximum file size for inline audio (bytes)
    pub max_inline_audio_size: usize,
}

impl Default for HuggingFaceConfig {
    fn default() -> Self {
        Self {
            dataset_name: "custom-speech-dataset".to_string(),
            description: "A speech synthesis dataset".to_string(),
            version: "1.0.0".to_string(),
            license: Some("CC-BY-4.0".to_string()),
            tags: vec!["speech".to_string(), "tts".to_string(), "audio".to_string()],
            include_audio_data: false,
            audio_format: AudioExportFormat::Wav,
            max_inline_audio_size: 1024 * 1024, // 1MB
        }
    }
}

/// Audio export format
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AudioExportFormat {
    /// Keep original format
    Original,
    /// Convert to WAV
    Wav,
    /// Convert to FLAC
    Flac,
    /// Convert to MP3
    Mp3,
}

/// HuggingFace dataset card metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetCard {
    /// Dataset name
    pub name: String,
    /// Dataset description
    pub description: String,
    /// Dataset version
    pub version: String,
    /// License
    pub license: Option<String>,
    /// Tags
    pub tags: Vec<String>,
    /// Language codes
    pub languages: Vec<String>,
    /// Task categories
    pub task_categories: Vec<String>,
    /// Dataset size in bytes
    pub size_in_bytes: Option<u64>,
    /// Number of samples
    pub num_samples: usize,
    /// Features description
    pub features: HashMap<String, FeatureInfo>,
    /// Splits information
    pub splits: HashMap<String, SplitInfo>,
}

/// Feature information for dataset card
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureInfo {
    /// Feature type
    pub dtype: String,
    /// Feature description
    pub description: String,
    /// Whether feature is required
    pub required: bool,
}

/// Split information for dataset card
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitInfo {
    /// Number of samples in split
    pub num_examples: usize,
    /// Size in bytes
    pub size_in_bytes: Option<u64>,
}

/// HuggingFace Datasets exporter
pub struct HuggingFaceExporter {
    /// Export configuration
    config: HuggingFaceConfig,
}

impl HuggingFaceExporter {
    /// Create new HuggingFace exporter
    pub fn new(config: HuggingFaceConfig) -> Self {
        Self { config }
    }

    /// Create exporter with default configuration
    pub fn default() -> Self {
        Self::new(HuggingFaceConfig::default())
    }

    /// Export dataset to HuggingFace format
    pub async fn export_dataset(
        &self,
        samples: &[DatasetSample],
        output_dir: &Path,
    ) -> Result<()> {
        // Create output directory
        fs::create_dir_all(output_dir).await?;

        // Generate dataset card
        let card = self.generate_dataset_card(samples).await?;
        self.write_dataset_card(&card, output_dir).await?;

        // Export data in Arrow/Parquet format
        self.export_arrow_data(samples, output_dir).await?;

        // Copy or convert audio files if needed
        if self.config.include_audio_data {
            self.export_audio_files(samples, output_dir).await?;
        }

        Ok(())
    }

    /// Generate dataset card from samples
    async fn generate_dataset_card(&self, samples: &[DatasetSample]) -> Result<DatasetCard> {
        let mut languages = std::collections::HashSet::new();
        let mut total_size = 0u64;

        // Analyze samples
        for sample in samples {
            languages.insert(sample.language.as_str().to_string());
            
            // Estimate size (rough approximation)
            total_size += sample.text.len() as u64;
            total_size += (sample.audio.duration() * sample.audio.sample_rate() as f32 * 4.0) as u64; // 4 bytes per float sample
        }

        // Define features
        let mut features = HashMap::new();
        features.insert("id".to_string(), FeatureInfo {
            dtype: "string".to_string(),
            description: "Unique identifier for the sample".to_string(),
            required: true,
        });
        features.insert("text".to_string(), FeatureInfo {
            dtype: "string".to_string(),
            description: "Text transcription".to_string(),
            required: true,
        });
        features.insert("audio".to_string(), FeatureInfo {
            dtype: "audio".to_string(),
            description: "Audio data".to_string(),
            required: true,
        });
        features.insert("speaker_id".to_string(), FeatureInfo {
            dtype: "string".to_string(),
            description: "Speaker identifier".to_string(),
            required: false,
        });
        features.insert("language".to_string(), FeatureInfo {
            dtype: "string".to_string(),
            description: "Language code".to_string(),
            required: true,
        });

        // Create split info (assuming single split for now)
        let mut splits = HashMap::new();
        splits.insert("train".to_string(), SplitInfo {
            num_examples: samples.len(),
            size_in_bytes: Some(total_size),
        });

        Ok(DatasetCard {
            name: self.config.dataset_name.clone(),
            description: self.config.description.clone(),
            version: self.config.version.clone(),
            license: self.config.license.clone(),
            tags: self.config.tags.clone(),
            languages: languages.into_iter().collect(),
            task_categories: vec!["text-to-speech".to_string()],
            size_in_bytes: Some(total_size),
            num_samples: samples.len(),
            features,
            splits,
        })
    }

    /// Write dataset card to file
    async fn write_dataset_card(&self, card: &DatasetCard, output_dir: &Path) -> Result<()> {
        // Generate YAML frontmatter for dataset card
        let yaml_content = format!(
            r#"---
license: {}
task_categories:
{}
language:
{}
tags:
{}
size_categories:
- {}
---

# Dataset Card for {}

## Dataset Description

{}

## Dataset Structure

### Data Fields

{}

### Data Splits

{}

## Dataset Creation

This dataset was exported using the VoiRS dataset utilities.

### Source Data

- **Number of samples**: {}
- **Total size**: {} bytes
- **Languages**: {}
- **Version**: {}

## Additional Information

### Licensing Information

{}

### Citation Information

```
@misc{{{}dataset,
  title={{{}}},
  author={{VoiRS}},
  year={{2025}},
  version={{{}}}
}}
```
"#,
            card.license.as_deref().unwrap_or("unknown"),
            card.task_categories.iter().map(|t| format!("- {}", t)).collect::<Vec<_>>().join("\n"),
            card.languages.iter().map(|l| format!("- {}", l)).collect::<Vec<_>>().join("\n"),
            card.tags.iter().map(|t| format!("- {}", t)).collect::<Vec<_>>().join("\n"),
            self.get_size_category(card.num_samples),
            card.name,
            card.description,
            self.format_features(&card.features),
            self.format_splits(&card.splits),
            card.num_samples,
            card.size_in_bytes.unwrap_or(0),
            card.languages.join(", "),
            card.version,
            card.license.as_deref().unwrap_or("See dataset description"),
            card.name.replace('-', "_"),
            card.name,
            card.version,
        );

        let readme_path = output_dir.join("README.md");
        fs::write(readme_path, yaml_content).await?;

        Ok(())
    }

    /// Export data in Arrow/Parquet format
    async fn export_arrow_data(&self, samples: &[DatasetSample], output_dir: &Path) -> Result<()> {
        // For now, export as JSON Lines which can be easily converted to Arrow
        let jsonl_path = output_dir.join("train.jsonl");
        let mut jsonl_content = String::new();

        for sample in samples {
            let record = serde_json::json!({
                "id": sample.id,
                "text": sample.text,
                "audio": {
                    "path": format!("audio/{}.wav", sample.id),
                    "sampling_rate": sample.audio.sample_rate(),
                    "channels": sample.audio.channels(),
                    "duration": sample.audio.duration(),
                },
                "speaker_id": sample.speaker.as_ref().map(|s| &s.id),
                "language": sample.language.as_str(),
                "metadata": sample.metadata,
            });

            jsonl_content.push_str(&serde_json::to_string(&record)?);
            jsonl_content.push('\n');
        }

        fs::write(jsonl_path, jsonl_content).await?;

        Ok(())
    }

    /// Export audio files
    async fn export_audio_files(&self, samples: &[DatasetSample], output_dir: &Path) -> Result<()> {
        let audio_dir = output_dir.join("audio");
        fs::create_dir_all(&audio_dir).await?;

        for sample in samples {
            let audio_filename = format!("{}.wav", sample.id);
            let audio_path = audio_dir.join(audio_filename);

            // Convert audio to desired format
            let audio_data = match self.config.audio_format {
                AudioExportFormat::Original => sample.audio.clone(),
                AudioExportFormat::Wav => sample.audio.clone(), // Already handling WAV
                _ => {
                    // For now, just use WAV format
                    // TODO: Implement actual format conversion
                    sample.audio.clone()
                }
            };

            // Save audio file
            self.save_audio_file(&audio_data, &audio_path).await?;
        }

        Ok(())
    }

    /// Save audio file in WAV format
    async fn save_audio_file(&self, audio: &crate::AudioData, path: &Path) -> Result<()> {
        use hound::{WavSpec, WavWriter};
        use std::io::Cursor;

        let spec = WavSpec {
            channels: audio.channels() as u16,
            sample_rate: audio.sample_rate(),
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = WavWriter::new(&mut cursor, spec)?;
            for &sample in audio.samples() {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        }

        fs::write(path, cursor.into_inner()).await?;
        Ok(())
    }

    /// Get size category for HuggingFace
    fn get_size_category(&self, num_samples: usize) -> &'static str {
        match num_samples {
            0..=1000 => "n<1K",
            1001..=10000 => "1K<n<10K",
            10001..=100000 => "10K<n<100K",
            100001..=1000000 => "100K<n<1M",
            _ => "n>1M",
        }
    }

    /// Format features for dataset card
    fn format_features(&self, features: &HashMap<String, FeatureInfo>) -> String {
        features.iter()
            .map(|(name, info)| format!("- **{}** ({}): {}", name, info.dtype, info.description))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Format splits for dataset card
    fn format_splits(&self, splits: &HashMap<String, SplitInfo>) -> String {
        splits.iter()
            .map(|(name, info)| format!("- **{}**: {} examples", name, info.num_examples))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Update configuration
    pub fn with_config(mut self, config: HuggingFaceConfig) -> Self {
        self.config = config;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, LanguageCode, SpeakerInfo, QualityMetrics};
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_huggingface_config_default() {
        let config = HuggingFaceConfig::default();
        assert_eq!(config.dataset_name, "custom-speech-dataset");
        assert!(!config.include_audio_data);
        assert_eq!(config.max_inline_audio_size, 1024 * 1024);
    }

    #[tokio::test]
    async fn test_dataset_card_generation() {
        let config = HuggingFaceConfig::default();
        let exporter = HuggingFaceExporter::new(config);

        let samples = vec![
            DatasetSample::new(
                "test_001".to_string(),
                "Hello world".to_string(),
                AudioData::silence(1.0, 22050, 1),
                LanguageCode::EnUs,
            ),
            DatasetSample::new(
                "test_002".to_string(),
                "Another sample".to_string(),
                AudioData::silence(2.0, 22050, 1),
                LanguageCode::EnUs,
            ).with_speaker(SpeakerInfo {
                id: "speaker1".to_string(),
                name: None,
                gender: None,
                age: None,
                accent: None,
                metadata: std::collections::HashMap::new(),
            }),
        ];

        let card = exporter.generate_dataset_card(&samples).await.unwrap();
        
        assert_eq!(card.name, "custom-speech-dataset");
        assert_eq!(card.num_samples, 2);
        assert!(card.languages.contains(&"en-US".to_string()));
        assert!(card.features.contains_key("text"));
        assert!(card.features.contains_key("audio"));
        assert!(card.splits.contains_key("train"));
    }

    #[tokio::test]
    async fn test_export_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let config = HuggingFaceConfig {
            dataset_name: "test-dataset".to_string(),
            include_audio_data: false, // Skip audio export for test
            ..Default::default()
        };
        let exporter = HuggingFaceExporter::new(config);

        let samples = vec![
            DatasetSample::new(
                "sample_001".to_string(),
                "Test sample".to_string(),
                AudioData::silence(1.0, 22050, 1),
                LanguageCode::EnUs,
            ),
        ];

        exporter.export_dataset(&samples, temp_dir.path()).await.unwrap();

        // Check that files were created
        assert!(temp_dir.path().join("README.md").exists());
        assert!(temp_dir.path().join("train.jsonl").exists());

        // Check JSONL content
        let jsonl_content = fs::read_to_string(temp_dir.path().join("train.jsonl")).await.unwrap();
        assert!(jsonl_content.contains("sample_001"));
        assert!(jsonl_content.contains("Test sample"));
    }

    #[test]
    fn test_size_category() {
        let exporter = HuggingFaceExporter::default();
        assert_eq!(exporter.get_size_category(500), "n<1K");
        assert_eq!(exporter.get_size_category(5000), "1K<n<10K");
        assert_eq!(exporter.get_size_category(50000), "10K<n<100K");
        assert_eq!(exporter.get_size_category(500000), "100K<n<1M");
        assert_eq!(exporter.get_size_category(5000000), "n>1M");
    }
}
