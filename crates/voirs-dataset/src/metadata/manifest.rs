//! Dataset manifest generation and management
//!
//! Provides comprehensive manifest creation for datasets with support for
//! multiple output formats including JSON, CSV, and Parquet.

use crate::{DatasetError, DatasetSample, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;

/// Dataset manifest containing comprehensive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    /// Manifest metadata
    pub metadata: ManifestMetadata,
    /// Sample entries
    pub samples: Vec<SampleEntry>,
    /// Dataset statistics
    pub statistics: DatasetStatistics,
    /// Schema version
    pub schema_version: String,
}

/// Manifest metadata information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestMetadata {
    /// Dataset name
    pub name: String,
    /// Dataset version
    pub version: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Total sample count
    pub total_samples: usize,
    /// Supported languages
    pub languages: Vec<String>,
    /// Sample rate information
    pub sample_rates: Vec<u32>,
    /// Audio formats present
    pub audio_formats: Vec<String>,
    /// Total duration in seconds
    pub total_duration: f64,
    /// Creator information
    pub creator: Option<String>,
    /// License information
    pub license: Option<String>,
    /// Additional metadata
    pub extra: HashMap<String, serde_json::Value>,
}

/// Individual sample entry in manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleEntry {
    /// Unique sample ID
    pub id: String,
    /// Text content
    pub text: String,
    /// Audio file path (relative to dataset root)
    pub audio_path: PathBuf,
    /// Duration in seconds
    pub duration: f64,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u32,
    /// Speaker information
    pub speaker: Option<SpeakerEntry>,
    /// Language code
    pub language: String,
    /// Quality metrics
    pub quality: QualityEntry,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Speaker information in manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEntry {
    /// Speaker ID
    pub id: String,
    /// Speaker name
    pub name: Option<String>,
    /// Gender information
    pub gender: Option<String>,
    /// Age information
    pub age: Option<u32>,
    /// Accent/region
    pub accent: Option<String>,
}

/// Quality metrics in manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityEntry {
    /// Signal-to-noise ratio
    pub snr: Option<f32>,
    /// Clipping percentage
    pub clipping: Option<f32>,
    /// Dynamic range
    pub dynamic_range: Option<f32>,
    /// Overall quality score (0.0-1.0)
    pub overall_score: Option<f32>,
}

/// Dataset statistics in manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    /// Total sample count
    pub total_samples: usize,
    /// Total duration in seconds
    pub total_duration: f64,
    /// Average duration per sample
    pub avg_duration: f64,
    /// Min/max durations
    pub min_duration: f64,
    pub max_duration: f64,
    /// Text statistics
    pub text_stats: TextStatistics,
    /// Audio statistics
    pub audio_stats: AudioStatistics,
    /// Quality statistics
    pub quality_stats: QualityStatistics,
}

/// Text statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStatistics {
    /// Total character count
    pub total_chars: usize,
    /// Average characters per sample
    pub avg_chars: f64,
    /// Unique word count
    pub unique_words: usize,
    /// Language distribution
    pub language_distribution: HashMap<String, usize>,
}

/// Audio statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioStatistics {
    /// Sample rate distribution
    pub sample_rate_distribution: HashMap<u32, usize>,
    /// Channel distribution
    pub channel_distribution: HashMap<u32, usize>,
    /// Format distribution
    pub format_distribution: HashMap<String, usize>,
}

/// Quality statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStatistics {
    /// Average SNR
    pub avg_snr: Option<f32>,
    /// Average clipping percentage
    pub avg_clipping: Option<f32>,
    /// Average dynamic range
    pub avg_dynamic_range: Option<f32>,
    /// Overall quality distribution
    pub quality_distribution: HashMap<String, usize>, // "excellent", "good", "fair", "poor"
}

/// Manifest generation configuration
#[derive(Debug, Clone)]
pub struct ManifestConfig {
    /// Include detailed quality metrics
    pub include_quality: bool,
    /// Include speaker information
    pub include_speaker: bool,
    /// Include audio file checksums
    pub include_checksums: bool,
    /// Maximum samples to include (None = all)
    pub max_samples: Option<usize>,
    /// Base path for relative audio paths
    pub base_path: Option<PathBuf>,
    /// Output format preference
    pub format: ManifestFormat,
    /// Validate audio files during generation
    pub validate_audio: bool,
}

/// Supported manifest formats
#[derive(Debug, Clone)]
pub enum ManifestFormat {
    /// JSON format (human readable)
    Json { pretty: bool },
    /// CSV format (spreadsheet compatible)
    Csv { include_metadata: bool },
    /// Parquet format (big data tools)
    Parquet { compression: ParquetCompression },
}

/// Parquet compression options
#[derive(Debug, Clone)]
pub enum ParquetCompression {
    None,
    Snappy,
    Gzip,
    Lz4,
}

impl Default for ManifestConfig {
    fn default() -> Self {
        Self {
            include_quality: true,
            include_speaker: true,
            include_checksums: false,
            max_samples: None,
            base_path: None,
            format: ManifestFormat::Json { pretty: true },
            validate_audio: false,
        }
    }
}

/// Manifest generator for creating dataset manifests
pub struct ManifestGenerator {
    config: ManifestConfig,
}

impl ManifestGenerator {
    /// Create a new manifest generator with default configuration
    pub fn new() -> Self {
        Self {
            config: ManifestConfig::default(),
        }
    }

    /// Create a new manifest generator with custom configuration
    pub fn with_config(config: ManifestConfig) -> Self {
        Self { config }
    }

    /// Generate manifest from dataset samples
    pub async fn generate_from_samples(
        &self,
        samples: Vec<DatasetSample>,
        name: String,
    ) -> Result<DatasetManifest> {
        let mut manifest_samples = Vec::new();
        let mut text_chars = 0;
        let mut total_duration = 0.0;
        let mut min_duration = f64::MAX;
        let mut max_duration: f64 = 0.0;
        let mut words = std::collections::HashSet::new();
        let mut language_counts = HashMap::new();
        let mut sample_rate_counts = HashMap::new();
        let mut channel_counts = HashMap::new();
        let mut format_counts = HashMap::new();
        let mut quality_scores = Vec::new();

        for (idx, sample) in samples.iter().enumerate() {
            if let Some(max) = self.config.max_samples {
                if idx >= max {
                    break;
                }
            }

            let entry = self.convert_sample(sample).await?;

            // Update statistics
            text_chars += entry.text.chars().count();
            total_duration += entry.duration;
            min_duration = min_duration.min(entry.duration);
            max_duration = max_duration.max(entry.duration);

            // Count words
            for word in entry.text.split_whitespace() {
                words.insert(word.to_lowercase());
            }

            // Update distributions
            *language_counts.entry(entry.language.clone()).or_insert(0) += 1;
            *sample_rate_counts.entry(entry.sample_rate).or_insert(0) += 1;
            *channel_counts.entry(entry.channels).or_insert(0) += 1;

            if let Some(path) = entry.audio_path.extension() {
                if let Some(ext) = path.to_str() {
                    *format_counts.entry(ext.to_lowercase()).or_insert(0) += 1;
                }
            }

            if let Some(score) = entry.quality.overall_score {
                quality_scores.push(score);
            }

            manifest_samples.push(entry);
        }

        let sample_count = manifest_samples.len();
        let avg_duration = if sample_count > 0 {
            total_duration / sample_count as f64
        } else {
            0.0
        };
        let avg_chars = if sample_count > 0 {
            text_chars as f64 / sample_count as f64
        } else {
            0.0
        };

        // Calculate quality statistics
        let avg_quality = if !quality_scores.is_empty() {
            Some(quality_scores.iter().sum::<f32>() / quality_scores.len() as f32)
        } else {
            None
        };

        let quality_distribution = self.calculate_quality_distribution(&quality_scores);

        let manifest = DatasetManifest {
            metadata: ManifestMetadata {
                name: name.clone(),
                version: "1.0.0".to_string(),
                created_at: chrono::Utc::now(),
                total_samples: sample_count,
                languages: language_counts.keys().cloned().collect(),
                sample_rates: sample_rate_counts.keys().cloned().collect(),
                audio_formats: format_counts.keys().cloned().collect(),
                total_duration,
                creator: None,
                license: None,
                extra: HashMap::new(),
            },
            samples: manifest_samples,
            statistics: DatasetStatistics {
                total_samples: sample_count,
                total_duration,
                avg_duration,
                min_duration: if sample_count > 0 { min_duration } else { 0.0 },
                max_duration,
                text_stats: TextStatistics {
                    total_chars: text_chars,
                    avg_chars,
                    unique_words: words.len(),
                    language_distribution: language_counts,
                },
                audio_stats: AudioStatistics {
                    sample_rate_distribution: sample_rate_counts,
                    channel_distribution: channel_counts,
                    format_distribution: format_counts,
                },
                quality_stats: QualityStatistics {
                    avg_snr: avg_quality,
                    avg_clipping: None,
                    avg_dynamic_range: None,
                    quality_distribution,
                },
            },
            schema_version: "1.0.0".to_string(),
        };

        Ok(manifest)
    }

    /// Save manifest to file
    pub async fn save_manifest(
        &self,
        manifest: &DatasetManifest,
        output_path: &Path,
    ) -> Result<()> {
        match &self.config.format {
            ManifestFormat::Json { pretty } => {
                let json = if *pretty {
                    serde_json::to_string_pretty(manifest)
                } else {
                    serde_json::to_string(manifest)
                }
                .map_err(|e| {
                    DatasetError::FormatError(format!("JSON serialization failed: {e}"))
                })?;

                fs::write(output_path, json)
                    .await
                    .map_err(DatasetError::IoError)?;
            }
            ManifestFormat::Csv { include_metadata } => {
                self.save_as_csv(manifest, output_path, *include_metadata)
                    .await?;
            }
            ManifestFormat::Parquet { compression } => {
                self.save_as_parquet(manifest, output_path, compression)
                    .await?;
            }
        }

        Ok(())
    }

    /// Convert dataset sample to manifest entry
    async fn convert_sample(&self, sample: &crate::DatasetSample) -> Result<SampleEntry> {
        // Create audio path by using the sample ID
        let audio_path_string = format!("{}.wav", sample.id);
        let audio_path = Path::new(&audio_path_string);

        Ok(SampleEntry {
            id: sample.id.clone(),
            text: sample.text.clone(),
            audio_path: self.get_relative_path(audio_path)?,
            duration: sample.audio.duration() as f64,
            sample_rate: sample.audio.sample_rate(),
            channels: sample.audio.channels(),
            speaker: sample.speaker.as_ref().map(|s| SpeakerEntry {
                id: s.id.clone(),
                name: s.name.clone(),
                gender: s.gender.clone(),
                age: s.age,
                accent: s.accent.clone(),
            }),
            language: format!("{:?}", sample.language),
            quality: QualityEntry {
                snr: sample.quality.snr,
                clipping: sample.quality.clipping,
                dynamic_range: sample.quality.dynamic_range,
                overall_score: sample.quality.overall_quality,
            },
            metadata: sample
                .metadata
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        })
    }

    /// Get relative path for audio file
    fn get_relative_path(&self, audio_path: &Path) -> Result<PathBuf> {
        if let Some(base) = &self.config.base_path {
            audio_path
                .strip_prefix(base)
                .map(Path::to_path_buf)
                .map_err(|_| {
                    DatasetError::ConfigError("Invalid base path for audio file".to_string())
                })
        } else {
            Ok(audio_path.to_path_buf())
        }
    }

    /// Calculate quality distribution
    fn calculate_quality_distribution(&self, scores: &[f32]) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for &score in scores {
            let category = match score {
                s if s >= 0.8 => "excellent",
                s if s >= 0.6 => "good",
                s if s >= 0.4 => "fair",
                _ => "poor",
            };
            *distribution.entry(category.to_string()).or_insert(0) += 1;
        }

        distribution
    }

    /// Save manifest as CSV format
    async fn save_as_csv(
        &self,
        manifest: &DatasetManifest,
        output_path: &Path,
        _include_metadata: bool,
    ) -> Result<()> {
        let mut wtr =
            csv::Writer::from_path(output_path).map_err(|e| DatasetError::IoError(e.into()))?;

        // Write header
        let mut headers = vec![
            "id",
            "text",
            "audio_path",
            "duration",
            "sample_rate",
            "channels",
            "language",
        ];

        if self.config.include_speaker {
            headers.extend(["speaker_id", "speaker_name", "speaker_gender"]);
        }

        if self.config.include_quality {
            headers.extend(["snr", "clipping", "dynamic_range", "quality_score"]);
        }

        wtr.write_record(headers)
            .map_err(|e| DatasetError::FormatError(format!("CSV header write failed: {e}")))?;

        // Write samples
        for sample in &manifest.samples {
            let mut record = vec![
                sample.id.clone(),
                sample.text.clone(),
                sample.audio_path.to_string_lossy().to_string(),
                sample.duration.to_string(),
                sample.sample_rate.to_string(),
                sample.channels.to_string(),
                sample.language.clone(),
            ];

            if self.config.include_speaker {
                record.push(
                    sample
                        .speaker
                        .as_ref()
                        .map(|s| s.id.clone())
                        .unwrap_or_default(),
                );
                record.push(
                    sample
                        .speaker
                        .as_ref()
                        .and_then(|s| s.name.clone())
                        .unwrap_or_default(),
                );
                record.push(
                    sample
                        .speaker
                        .as_ref()
                        .and_then(|s| s.gender.clone())
                        .unwrap_or_default(),
                );
            }

            if self.config.include_quality {
                record.push(
                    sample
                        .quality
                        .snr
                        .map(|v| v.to_string())
                        .unwrap_or_default(),
                );
                record.push(
                    sample
                        .quality
                        .clipping
                        .map(|v| v.to_string())
                        .unwrap_or_default(),
                );
                record.push(
                    sample
                        .quality
                        .dynamic_range
                        .map(|v| v.to_string())
                        .unwrap_or_default(),
                );
                record.push(
                    sample
                        .quality
                        .overall_score
                        .map(|v| v.to_string())
                        .unwrap_or_default(),
                );
            }

            wtr.write_record(record)
                .map_err(|e| DatasetError::FormatError(format!("CSV record write failed: {e}")))?;
        }

        wtr.flush().map_err(DatasetError::IoError)?;

        Ok(())
    }

    /// Save manifest as native Parquet format
    async fn save_as_parquet(
        &self,
        manifest: &DatasetManifest,
        output_path: &Path,
        compression: &ParquetCompression,
    ) -> Result<()> {
        use arrow::array::*;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::arrow_writer::ArrowWriter;
        use parquet::basic::{Compression, Encoding};
        use parquet::file::properties::WriterProperties;
        use std::sync::Arc;

        // Create the base output path (without extension)
        let base_path = output_path.with_extension("");

        // Save manifest metadata as separate JSON file
        let metadata_path = base_path.with_extension("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&manifest.metadata).map_err(|e| {
            DatasetError::FormatError(format!("Metadata serialization failed: {e}"))
        })?;
        fs::write(&metadata_path, metadata_json).await?;

        // Save statistics as separate JSON file
        let stats_path = base_path.with_extension("stats.json");
        let stats_json = serde_json::to_string_pretty(&manifest.statistics).map_err(|e| {
            DatasetError::FormatError(format!("Statistics serialization failed: {e}"))
        })?;
        fs::write(&stats_path, stats_json).await?;

        // Create main data file in Parquet format
        let data_path = base_path.with_extension("parquet");

        // Define schema for the Parquet file
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("audio_path", DataType::Utf8, false),
            Field::new("duration", DataType::Float32, false),
            Field::new("sample_rate", DataType::UInt32, false),
            Field::new("channels", DataType::UInt32, false),
            Field::new("language", DataType::Utf8, false),
            Field::new("speaker_id", DataType::Utf8, true),
            Field::new("speaker_name", DataType::Utf8, true),
            Field::new("speaker_gender", DataType::Utf8, true),
            Field::new("speaker_age", DataType::UInt32, true),
            Field::new("speaker_accent", DataType::Utf8, true),
            Field::new("quality_snr", DataType::Float32, true),
            Field::new("quality_clipping", DataType::Float32, true),
            Field::new("quality_dynamic_range", DataType::Float32, true),
            Field::new("quality_overall_score", DataType::Float32, true),
            Field::new("metadata", DataType::Utf8, true),
        ]));

        // Prepare arrays for all columns
        let _num_rows = manifest.samples.len();
        let mut id_builder = StringBuilder::new();
        let mut text_builder = StringBuilder::new();
        let mut audio_path_builder = StringBuilder::new();
        let mut duration_builder = Float32Builder::new();
        let mut sample_rate_builder = UInt32Builder::new();
        let mut channels_builder = UInt32Builder::new();
        let mut language_builder = StringBuilder::new();
        let mut speaker_id_builder = StringBuilder::new();
        let mut speaker_name_builder = StringBuilder::new();
        let mut speaker_gender_builder = StringBuilder::new();
        let mut speaker_age_builder = UInt32Builder::new();
        let mut speaker_accent_builder = StringBuilder::new();
        let mut quality_snr_builder = Float32Builder::new();
        let mut quality_clipping_builder = Float32Builder::new();
        let mut quality_dynamic_range_builder = Float32Builder::new();
        let mut quality_overall_score_builder = Float32Builder::new();
        let mut metadata_builder = StringBuilder::new();

        // Populate arrays
        for sample in &manifest.samples {
            id_builder.append_value(&sample.id);
            text_builder.append_value(&sample.text);
            audio_path_builder.append_value(sample.audio_path.to_string_lossy());
            duration_builder.append_value(sample.duration as f32);
            sample_rate_builder.append_value(sample.sample_rate);
            channels_builder.append_value(sample.channels);
            language_builder.append_value(&sample.language);

            // Handle optional speaker fields
            if let Some(speaker) = &sample.speaker {
                speaker_id_builder.append_value(&speaker.id);
                speaker_name_builder.append_option(speaker.name.as_ref());
                speaker_gender_builder.append_option(speaker.gender.as_ref());
                speaker_age_builder.append_option(speaker.age);
                speaker_accent_builder.append_option(speaker.accent.as_ref());
            } else {
                speaker_id_builder.append_null();
                speaker_name_builder.append_null();
                speaker_gender_builder.append_null();
                speaker_age_builder.append_null();
                speaker_accent_builder.append_null();
            }

            // Handle optional quality fields
            quality_snr_builder.append_option(sample.quality.snr);
            quality_clipping_builder.append_option(sample.quality.clipping);
            quality_dynamic_range_builder.append_option(sample.quality.dynamic_range);
            quality_overall_score_builder.append_option(sample.quality.overall_score);

            // Serialize metadata as JSON string
            let metadata_json = serde_json::to_string(&sample.metadata).map_err(|e| {
                DatasetError::FormatError(format!("Metadata serialization failed: {e}"))
            })?;
            metadata_builder.append_value(metadata_json);
        }

        // Finish arrays
        let arrays: Vec<Arc<dyn Array>> = vec![
            Arc::new(id_builder.finish()),
            Arc::new(text_builder.finish()),
            Arc::new(audio_path_builder.finish()),
            Arc::new(duration_builder.finish()),
            Arc::new(sample_rate_builder.finish()),
            Arc::new(channels_builder.finish()),
            Arc::new(language_builder.finish()),
            Arc::new(speaker_id_builder.finish()),
            Arc::new(speaker_name_builder.finish()),
            Arc::new(speaker_gender_builder.finish()),
            Arc::new(speaker_age_builder.finish()),
            Arc::new(speaker_accent_builder.finish()),
            Arc::new(quality_snr_builder.finish()),
            Arc::new(quality_clipping_builder.finish()),
            Arc::new(quality_dynamic_range_builder.finish()),
            Arc::new(quality_overall_score_builder.finish()),
            Arc::new(metadata_builder.finish()),
        ];

        // Create record batch
        let batch = RecordBatch::try_new(schema.clone(), arrays).map_err(|e| {
            DatasetError::FormatError(format!("Failed to create record batch: {e}"))
        })?;

        // Configure compression
        let parquet_compression = match compression {
            ParquetCompression::None => Compression::UNCOMPRESSED,
            ParquetCompression::Gzip => Compression::GZIP(Default::default()),
            ParquetCompression::Snappy => Compression::SNAPPY,
            ParquetCompression::Lz4 => Compression::LZ4,
        };

        // Create writer properties
        let props = WriterProperties::builder()
            .set_compression(parquet_compression)
            .set_encoding(Encoding::PLAIN)
            .build();

        // Write to Parquet file
        let file = std::fs::File::create(&data_path).map_err(DatasetError::IoError)?;

        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).map_err(|e| {
            DatasetError::FormatError(format!("Failed to create Parquet writer: {e}"))
        })?;

        writer.write(&batch).map_err(|e| {
            DatasetError::FormatError(format!("Failed to write Parquet batch: {e}"))
        })?;

        writer.close().map_err(|e| {
            DatasetError::FormatError(format!("Failed to close Parquet writer: {e}"))
        })?;

        // Create a README file explaining the format
        let readme_path = base_path.with_extension("README.md");
        let base_name = base_path.file_name().unwrap().to_string_lossy();
        let readme_content = format!(
            r#"# Native Parquet Dataset Export

This dataset has been exported in native Apache Parquet format.

## Files

- `{base_name}.parquet` - Main dataset samples in Parquet format
- `{base_name}.metadata.json` - Dataset metadata
- `{base_name}.stats.json` - Dataset statistics
- `{base_name}.README.md` - This file

## Parquet Schema

The Parquet file contains the following columns:
- id: Sample identifier (string)
- text: Text content (string)
- audio_path: Path to audio file
- duration: Audio duration in seconds
- sample_rate: Sample rate in Hz
- channels: Number of audio channels
- language: Language code
- speaker_*: Speaker information (id, name, gender, age, accent)
- quality_*: Quality metrics (SNR, clipping, dynamic range, overall score)
- metadata: Additional metadata as JSON string

## Reading the Parquet File

You can read this Parquet file using any tool that supports Apache Parquet:

```python
# Using pandas
import pandas as pd
df = pd.read_parquet('{base_name}.parquet')
```

```python
# Using Apache Arrow
import pyarrow.parquet as pq
table = pq.read_table('{base_name}.parquet')
df = table.to_pandas()
```

```rust
// Using Arrow in Rust
use parquet::arrow::arrow_reader::ParquetFileArrowReader;
let file = std::fs::File::open("{base_name}.parquet").unwrap();
let reader = ParquetFileArrowReader::try_new(file).unwrap();
```

## Compression

Compression: {compression:?}
"#
        );
        fs::write(&readme_path, readme_content).await?;

        Ok(())
    }
}

impl Default for ManifestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::dummy::DummyDataset;
    use crate::traits::Dataset;

    #[tokio::test]
    async fn test_manifest_generation() {
        let dataset = DummyDataset::small();

        let mut samples = Vec::new();
        for i in 0..5 {
            samples.push(dataset.get(i).await.unwrap());
        }

        let generator = ManifestGenerator::new();
        let manifest = generator
            .generate_from_samples(samples, "test-dataset".to_string())
            .await
            .unwrap();

        assert_eq!(manifest.metadata.name, "test-dataset");
        assert_eq!(manifest.samples.len(), 5);
        assert!(manifest.statistics.total_duration > 0.0);
        assert!(manifest.statistics.avg_duration > 0.0);
    }

    #[tokio::test]
    async fn test_manifest_csv_export() {
        let dataset = DummyDataset::small();

        let mut samples = Vec::new();
        for i in 0..3 {
            samples.push(dataset.get(i).await.unwrap());
        }

        let manifest_config = ManifestConfig {
            format: ManifestFormat::Csv {
                include_metadata: true,
            },
            ..Default::default()
        };

        let generator = ManifestGenerator::with_config(manifest_config);
        let manifest = generator
            .generate_from_samples(samples, "test-csv".to_string())
            .await
            .unwrap();

        let temp_path = std::env::temp_dir().join("test_manifest.csv");
        generator
            .save_manifest(&manifest, &temp_path)
            .await
            .unwrap();

        assert!(temp_path.exists());

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }

    #[tokio::test]
    async fn test_manifest_parquet_export() {
        let dataset = DummyDataset::small();

        let mut samples = Vec::new();
        for i in 0..3 {
            samples.push(dataset.get(i).await.unwrap());
        }

        let manifest_config = ManifestConfig {
            format: ManifestFormat::Parquet {
                compression: ParquetCompression::None,
            },
            ..Default::default()
        };

        let generator = ManifestGenerator::with_config(manifest_config);
        let manifest = generator
            .generate_from_samples(samples, "test-parquet".to_string())
            .await
            .unwrap();

        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_manifest");
        generator
            .save_manifest(&manifest, &temp_path)
            .await
            .unwrap();

        // Check that all expected files were created
        assert!(temp_path.with_extension("parquet").exists());
        assert!(temp_path.with_extension("metadata.json").exists());
        assert!(temp_path.with_extension("stats.json").exists());
        assert!(temp_path.with_extension("README.md").exists());

        // Verify Parquet file is valid and has content
        let parquet_metadata = std::fs::metadata(temp_path.with_extension("parquet")).unwrap();
        assert!(parquet_metadata.len() > 0);

        // Try to open the Parquet file to verify it's valid
        use parquet::file::reader::{FileReader, SerializedFileReader};
        let file = std::fs::File::open(temp_path.with_extension("parquet")).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        let parquet_metadata = reader.metadata();
        assert_eq!(parquet_metadata.num_row_groups(), 1);
        assert!(parquet_metadata.file_metadata().num_rows() > 0);

        // Clean up
        let _ = std::fs::remove_file(temp_path.with_extension("parquet"));
        let _ = std::fs::remove_file(temp_path.with_extension("metadata.json"));
        let _ = std::fs::remove_file(temp_path.with_extension("stats.json"));
        let _ = std::fs::remove_file(temp_path.with_extension("README.md"));
    }

    #[tokio::test]
    async fn test_manifest_parquet_gzip_export() {
        let dataset = DummyDataset::small();

        let mut samples = Vec::new();
        for i in 0..2 {
            samples.push(dataset.get(i).await.unwrap());
        }

        let manifest_config = ManifestConfig {
            format: ManifestFormat::Parquet {
                compression: ParquetCompression::Gzip,
            },
            ..Default::default()
        };

        let generator = ManifestGenerator::with_config(manifest_config);
        let manifest = generator
            .generate_from_samples(samples, "test-parquet-gzip".to_string())
            .await
            .unwrap();

        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_manifest_gzip");
        generator
            .save_manifest(&manifest, &temp_path)
            .await
            .unwrap();

        // Check that compressed Parquet file was created
        assert!(temp_path.with_extension("parquet").exists());
        assert!(temp_path.with_extension("metadata.json").exists());

        // Verify the Parquet file is valid and has content
        let parquet_metadata = std::fs::metadata(temp_path.with_extension("parquet")).unwrap();
        assert!(parquet_metadata.len() > 0);

        // Try to open the compressed Parquet file to verify it's valid
        use parquet::file::reader::{FileReader, SerializedFileReader};
        let file = std::fs::File::open(temp_path.with_extension("parquet")).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        let parquet_metadata = reader.metadata();
        assert_eq!(parquet_metadata.num_row_groups(), 1);
        assert!(parquet_metadata.file_metadata().num_rows() > 0);

        // Clean up
        let _ = std::fs::remove_file(temp_path.with_extension("parquet"));
        let _ = std::fs::remove_file(temp_path.with_extension("metadata.json"));
        let _ = std::fs::remove_file(temp_path.with_extension("stats.json"));
        let _ = std::fs::remove_file(temp_path.with_extension("README.md"));
    }

    #[tokio::test]
    async fn test_quality_distribution() {
        let generator = ManifestGenerator::new();
        let scores = vec![0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.2];
        let distribution = generator.calculate_quality_distribution(&scores);

        assert_eq!(distribution.get("excellent"), Some(&2)); // 0.9, 0.8
        assert_eq!(distribution.get("good"), Some(&2)); // 0.7, 0.6
        assert_eq!(distribution.get("fair"), Some(&2)); // 0.5, 0.4
        assert_eq!(distribution.get("poor"), Some(&2)); // 0.3, 0.2
    }
}
