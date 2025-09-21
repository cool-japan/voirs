//! Simplified dataset manifest generation compatible with current DatasetSample trait
//!
//! Provides basic manifest creation for datasets using the simplified DatasetSample API.

use crate::traits::DatasetSample;
use crate::{DatasetError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;

/// Simplified dataset manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleManifest {
    /// Dataset name
    pub name: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Sample entries
    pub samples: Vec<SimpleSampleEntry>,
    /// Basic statistics
    pub statistics: SimpleStatistics,
}

/// Simplified sample entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleSampleEntry {
    /// Sample ID
    pub id: String,
    /// Text content
    pub text: String,
    /// Duration in seconds
    pub duration: f32,
    /// Language
    pub language: String,
    /// Speaker ID (if available)
    pub speaker_id: Option<String>,
}

/// Basic dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleStatistics {
    /// Total samples
    pub total_samples: usize,
    /// Total duration
    pub total_duration: f32,
    /// Average duration
    pub avg_duration: f32,
    /// Language distribution
    pub language_distribution: HashMap<String, usize>,
    /// Speaker distribution
    pub speaker_distribution: HashMap<String, usize>,
}

/// Simple manifest generator
pub struct SimpleManifestGenerator;

impl SimpleManifestGenerator {
    /// Create a new simple manifest generator
    pub fn new() -> Self {
        Self
    }

    /// Generate manifest from dataset samples
    pub async fn generate_from_samples<T>(
        &self,
        samples: Vec<T>,
        name: String,
    ) -> Result<SimpleManifest>
    where
        T: DatasetSample,
    {
        let mut sample_entries = Vec::new();
        let mut total_duration = 0.0;
        let mut language_counts = HashMap::new();
        let mut speaker_counts = HashMap::new();

        for sample in samples.iter() {
            let entry = SimpleSampleEntry {
                id: sample.id().to_string(),
                text: sample.text().to_string(),
                duration: sample.duration(),
                language: sample.language().to_string(),
                speaker_id: sample.speaker_id().map(str::to_string),
            };

            total_duration += entry.duration;
            *language_counts.entry(entry.language.clone()).or_insert(0) += 1;

            if let Some(speaker_id) = &entry.speaker_id {
                *speaker_counts.entry(speaker_id.clone()).or_insert(0) += 1;
            }

            sample_entries.push(entry);
        }

        let sample_count = sample_entries.len();
        let avg_duration = if sample_count > 0 {
            total_duration / sample_count as f32
        } else {
            0.0
        };

        Ok(SimpleManifest {
            name,
            created_at: chrono::Utc::now(),
            samples: sample_entries,
            statistics: SimpleStatistics {
                total_samples: sample_count,
                total_duration,
                avg_duration,
                language_distribution: language_counts,
                speaker_distribution: speaker_counts,
            },
        })
    }

    /// Save manifest to JSON file
    pub async fn save_manifest(&self, manifest: &SimpleManifest, output_path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(manifest)
            .map_err(|e| DatasetError::FormatError(format!("JSON serialization failed: {e}")))?;

        fs::write(output_path, json)
            .await
            .map_err(DatasetError::IoError)?;

        Ok(())
    }

    /// Save manifest as CSV
    pub async fn save_as_csv(&self, manifest: &SimpleManifest, output_path: &Path) -> Result<()> {
        let mut wtr =
            csv::Writer::from_path(output_path).map_err(|e| DatasetError::IoError(e.into()))?;

        // Write header
        wtr.write_record(["id", "text", "duration", "language", "speaker_id"])
            .map_err(|e| DatasetError::FormatError(format!("CSV header write failed: {e}")))?;

        // Write samples
        for sample in &manifest.samples {
            wtr.write_record([
                &sample.id,
                &sample.text,
                &sample.duration.to_string(),
                &sample.language,
                sample.speaker_id.as_deref().unwrap_or(""),
            ])
            .map_err(|e| DatasetError::FormatError(format!("CSV record write failed: {e}")))?;
        }

        wtr.flush().map_err(DatasetError::IoError)?;

        Ok(())
    }
}

impl Default for SimpleManifestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::dummy::{DummyConfig, DummyDataset};
    use crate::traits::Dataset;

    #[tokio::test]
    async fn test_simple_manifest_generation() {
        let config = DummyConfig {
            num_samples: 5,
            ..Default::default()
        };
        let dataset = DummyDataset::with_config(config);

        let mut samples = Vec::new();
        for i in 0..5 {
            samples.push(dataset.get(i).await.unwrap());
        }

        let generator = SimpleManifestGenerator::new();
        let manifest = generator
            .generate_from_samples(samples, "test-dataset".to_string())
            .await
            .unwrap();

        assert_eq!(manifest.name, "test-dataset");
        assert_eq!(manifest.samples.len(), 5);
        assert!(manifest.statistics.total_duration > 0.0);
        assert!(manifest.statistics.avg_duration > 0.0);
    }

    #[tokio::test]
    async fn test_simple_manifest_csv_export() {
        let config = DummyConfig {
            num_samples: 3,
            ..Default::default()
        };
        let dataset = DummyDataset::with_config(config);

        let mut samples = Vec::new();
        for i in 0..3 {
            samples.push(dataset.get(i).await.unwrap());
        }

        let generator = SimpleManifestGenerator::new();
        let manifest = generator
            .generate_from_samples(samples, "test-csv".to_string())
            .await
            .unwrap();

        let temp_path = std::env::temp_dir().join("test_simple_manifest.csv");
        generator.save_as_csv(&manifest, &temp_path).await.unwrap();

        assert!(temp_path.exists());

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }
}
