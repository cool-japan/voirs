//! Generic export formats
//!
//! This module provides a flexible generic exporter that can output datasets
//! in various common formats like CSV, JSON, and structured directories.

use crate::{DatasetSample, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;

/// Generic export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericConfig {
    /// Output format
    pub format: GenericFormat,
    /// Whether to include audio files
    pub include_audio: bool,
    /// Audio export format
    pub audio_format: AudioFormat,
    /// Whether to create a structured directory layout
    pub structured_layout: bool,
    /// Metadata fields to include
    pub metadata_fields: Vec<String>,
    /// Custom field mappings
    pub field_mappings: HashMap<String, String>,
}

impl Default for GenericConfig {
    fn default() -> Self {
        Self {
            format: GenericFormat::Csv,
            include_audio: true,
            audio_format: AudioFormat::Wav,
            structured_layout: true,
            metadata_fields: vec![
                "id".to_string(),
                "text".to_string(),
                "duration".to_string(),
                "speaker".to_string(),
            ],
            field_mappings: HashMap::new(),
        }
    }
}

/// Supported generic export formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GenericFormat {
    /// Comma-separated values
    Csv,
    /// Tab-separated values
    Tsv,
    /// JSON Lines format
    JsonLines,
    /// Single JSON file
    Json,
    /// Manifest format (simple text)
    Manifest,
}

/// Audio format for generic export
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AudioFormat {
    /// WAV format
    Wav,
    /// FLAC format (with fallback)
    Flac,
    /// OPUS format
    Opus,
    /// Original format (copy as-is)
    Original,
}

/// Generic exporter for flexible dataset export
pub struct GenericExporter {
    config: GenericConfig,
}

impl GenericExporter {
    /// Create a new generic exporter
    pub fn new(config: GenericConfig) -> Self {
        Self { config }
    }

    /// Export dataset to generic format
    pub async fn export<I>(&self, samples: I, output_dir: &Path) -> Result<()>
    where
        I: IntoIterator<Item = DatasetSample>,
    {
        // Create output directory
        fs::create_dir_all(output_dir).await?;

        let samples: Vec<DatasetSample> = samples.into_iter().collect();

        match self.config.format {
            GenericFormat::Csv => self.export_csv(&samples, output_dir).await,
            GenericFormat::Tsv => self.export_tsv(&samples, output_dir).await,
            GenericFormat::JsonLines => self.export_jsonlines(&samples, output_dir).await,
            GenericFormat::Json => self.export_json(&samples, output_dir).await,
            GenericFormat::Manifest => self.export_manifest(&samples, output_dir).await,
        }
    }

    /// Export to CSV format
    async fn export_csv(&self, samples: &[DatasetSample], output_dir: &Path) -> Result<()> {
        let csv_path = output_dir.join("dataset.csv");
        let mut writer = csv::Writer::from_path(&csv_path)?;

        // Write header
        writer.write_record(self.get_csv_headers())?;

        // Export samples and audio files
        for sample in samples {
            let audio_path = if self.config.include_audio {
                Some(self.export_audio_file(sample, output_dir).await?)
            } else {
                None
            };

            let row = self.sample_to_csv_row(sample, audio_path.as_deref());
            writer.write_record(&row)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Export to TSV format
    async fn export_tsv(&self, samples: &[DatasetSample], output_dir: &Path) -> Result<()> {
        let tsv_path = output_dir.join("dataset.tsv");
        let mut writer = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(&tsv_path)?;

        // Write header
        writer.write_record(self.get_csv_headers())?;

        // Export samples and audio files
        for sample in samples {
            let audio_path = if self.config.include_audio {
                Some(self.export_audio_file(sample, output_dir).await?)
            } else {
                None
            };

            let row = self.sample_to_csv_row(sample, audio_path.as_deref());
            writer.write_record(&row)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Export to JSON Lines format
    async fn export_jsonlines(&self, samples: &[DatasetSample], output_dir: &Path) -> Result<()> {
        let jsonl_path = output_dir.join("dataset.jsonl");
        let mut content = String::new();

        for sample in samples {
            let audio_path = if self.config.include_audio {
                Some(self.export_audio_file(sample, output_dir).await?)
            } else {
                None
            };

            let json_obj = self.sample_to_json(sample, audio_path.as_deref())?;
            content.push_str(&serde_json::to_string(&json_obj)?);
            content.push('\n');
        }

        fs::write(jsonl_path, content).await?;
        Ok(())
    }

    /// Export to single JSON format
    async fn export_json(&self, samples: &[DatasetSample], output_dir: &Path) -> Result<()> {
        let json_path = output_dir.join("dataset.json");
        let mut json_samples = Vec::new();

        for sample in samples {
            let audio_path = if self.config.include_audio {
                Some(self.export_audio_file(sample, output_dir).await?)
            } else {
                None
            };

            let json_obj = self.sample_to_json(sample, audio_path.as_deref())?;
            json_samples.push(json_obj);
        }

        let dataset = serde_json::json!({
            "version": "1.0",
            "samples": json_samples,
            "metadata": {
                "format": "generic",
                "audio_format": self.config.audio_format,
                "total_samples": json_samples.len(),
                "export_timestamp": chrono::Utc::now().to_rfc3339()
            }
        });

        let json_content = serde_json::to_string_pretty(&dataset)?;
        fs::write(json_path, json_content).await?;
        Ok(())
    }

    /// Export to manifest format
    async fn export_manifest(&self, samples: &[DatasetSample], output_dir: &Path) -> Result<()> {
        let manifest_path = output_dir.join("manifest.txt");
        let mut content = String::new();

        for sample in samples {
            let audio_path = if self.config.include_audio {
                Some(self.export_audio_file(sample, output_dir).await?)
            } else {
                None
            };

            // Simple manifest format: audio_path|text|metadata
            let line = format!(
                "{}|{}|{}\n",
                audio_path
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| "".to_string()),
                sample.text.replace('|', " "), // Escape pipe characters
                sample.id
            );
            content.push_str(&line);
        }

        fs::write(manifest_path, content).await?;
        Ok(())
    }

    /// Export audio file for a sample
    async fn export_audio_file(
        &self,
        sample: &DatasetSample,
        output_dir: &Path,
    ) -> Result<PathBuf> {
        let audio_dir = if self.config.structured_layout {
            output_dir.join("audio")
        } else {
            output_dir.to_path_buf()
        };

        fs::create_dir_all(&audio_dir).await?;

        let filename = match self.config.audio_format {
            AudioFormat::Wav => format!("{}.wav", sample.id),
            AudioFormat::Flac => format!("{}.flac", sample.id),
            AudioFormat::Opus => format!("{}.opus", sample.id),
            AudioFormat::Original => {
                // Try to preserve original extension or default to wav
                format!("{}.wav", sample.id)
            }
        };

        let audio_path = audio_dir.join(&filename);

        // Save audio file based on format
        match self.config.audio_format {
            AudioFormat::Wav => {
                self.save_wav_file(&sample.audio, &audio_path).await?;
            }
            AudioFormat::Flac => {
                self.save_flac_file(&sample.audio, &audio_path).await?;
            }
            AudioFormat::Opus => {
                // Use the implemented OPUS encoding from audio/io.rs
                match crate::audio::io::save_opus(&sample.audio, &audio_path) {
                    Ok(()) => {
                        tracing::debug!("Successfully saved OPUS file: {:?}", audio_path);
                    }
                    Err(e) => {
                        tracing::debug!("OPUS encoding failed: {}, using WAV fallback", e);
                        self.save_wav_file(&sample.audio, &audio_path).await?;
                    }
                }
            }
            AudioFormat::Original => {
                // For now, save as WAV (could be enhanced to detect/preserve original format)
                self.save_wav_file(&sample.audio, &audio_path).await?;
            }
        }

        // Return relative path for manifest entries
        if self.config.structured_layout {
            Ok(PathBuf::from("audio").join(&filename))
        } else {
            Ok(PathBuf::from(&filename))
        }
    }

    /// Save audio as WAV file
    async fn save_wav_file(&self, audio: &crate::AudioData, path: &Path) -> Result<()> {
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

    /// Save audio as FLAC file (with WAV fallback)
    async fn save_flac_file(&self, audio: &crate::AudioData, path: &Path) -> Result<()> {
        // For now, use WAV as fallback for FLAC
        // This could be enhanced with actual FLAC encoding in the future
        tracing::debug!("Using WAV fallback for FLAC export");
        self.save_wav_file(audio, path).await
    }

    /// Get CSV headers based on configuration
    fn get_csv_headers(&self) -> Vec<String> {
        let mut headers = Vec::new();

        for field in &self.config.metadata_fields {
            let mapped_field = self.config.field_mappings.get(field).unwrap_or(field);
            headers.push(mapped_field.clone());
        }

        if self.config.include_audio {
            headers.push("audio_path".to_string());
        }

        headers
    }

    /// Convert sample to CSV row
    fn sample_to_csv_row(&self, sample: &DatasetSample, audio_path: Option<&Path>) -> Vec<String> {
        let mut row = Vec::new();

        for field in &self.config.metadata_fields {
            let value = match field.as_str() {
                "id" => sample.id.to_string(),
                "text" => sample.text.to_string(),
                "duration" => {
                    let duration =
                        sample.audio.samples().len() as f64 / sample.audio.sample_rate() as f64;
                    format!("{duration:.3}")
                }
                "speaker" => sample
                    .speaker
                    .as_ref()
                    .map(|s| s.id.clone())
                    .unwrap_or_else(|| "unknown".to_string()),
                _ => "".to_string(), // Unknown fields are empty
            };
            row.push(value);
        }

        if self.config.include_audio {
            row.push(
                audio_path
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default(),
            );
        }

        row
    }

    /// Convert sample to JSON object
    fn sample_to_json(
        &self,
        sample: &DatasetSample,
        audio_path: Option<&Path>,
    ) -> Result<serde_json::Value> {
        let mut obj = serde_json::Map::new();

        for field in &self.config.metadata_fields {
            let mapped_field = self.config.field_mappings.get(field).unwrap_or(field);

            let value = match field.as_str() {
                "id" => serde_json::Value::String(sample.id.to_string()),
                "text" => serde_json::Value::String(sample.text.to_string()),
                "duration" => {
                    let duration =
                        sample.audio.samples().len() as f64 / sample.audio.sample_rate() as f64;
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(duration)
                            .unwrap_or_else(|| serde_json::Number::from(0)),
                    )
                }
                "speaker" => serde_json::Value::String(
                    sample
                        .speaker
                        .as_ref()
                        .map(|s| s.id.clone())
                        .unwrap_or_else(|| "unknown".to_string()),
                ),
                _ => serde_json::Value::Null,
            };
            obj.insert(mapped_field.clone(), value);
        }

        if self.config.include_audio {
            obj.insert(
                "audio_path".to_string(),
                serde_json::Value::String(
                    audio_path
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_default(),
                ),
            );
        }

        Ok(serde_json::Value::Object(obj))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, DatasetSample};
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_generic_exporter_creation() {
        let config = GenericConfig::default();
        let _exporter = GenericExporter::new(config);
    }

    #[tokio::test]
    async fn test_csv_export() {
        let config = GenericConfig {
            format: GenericFormat::Csv,
            include_audio: false,
            ..Default::default()
        };
        let exporter = GenericExporter::new(config);

        let audio = AudioData::new(vec![0.1, 0.2, 0.3], 44100, 1);
        let sample = DatasetSample::new(
            "test_001".to_string(),
            "Hello world".to_string(),
            audio,
            crate::LanguageCode::EnUs,
        );

        let temp_dir = TempDir::new().unwrap();
        let result = exporter.export(vec![sample], temp_dir.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_json_export() {
        let config = GenericConfig {
            format: GenericFormat::Json,
            include_audio: false,
            ..Default::default()
        };
        let exporter = GenericExporter::new(config);

        let audio = AudioData::new(vec![0.1, 0.2, 0.3], 44100, 1);
        let sample = DatasetSample::new(
            "test_001".to_string(),
            "Hello world".to_string(),
            audio,
            crate::LanguageCode::EnUs,
        );

        let temp_dir = TempDir::new().unwrap();
        let result = exporter.export(vec![sample], temp_dir.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_manifest_export() {
        let config = GenericConfig {
            format: GenericFormat::Manifest,
            include_audio: false,
            ..Default::default()
        };
        let exporter = GenericExporter::new(config);

        let audio = AudioData::new(vec![0.1, 0.2, 0.3], 44100, 1);
        let sample = DatasetSample::new(
            "test_001".to_string(),
            "Hello world".to_string(),
            audio,
            crate::LanguageCode::EnUs,
        );

        let temp_dir = TempDir::new().unwrap();
        let result = exporter.export(vec![sample], temp_dir.path()).await;
        assert!(result.is_ok());
    }
}
