//! Ground truth dataset management for evaluation validation
//!
//! This module provides comprehensive ground truth dataset management capabilities
//! for speech synthesis evaluation, including dataset cataloging, annotation support,
//! versioning, and validation against reference standards.

use crate::data_quality_validation::{DataQualityValidator, DatasetValidationReport};
use crate::VoirsError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Ground truth dataset management errors
#[derive(Error, Debug)]
pub enum GroundTruthError {
    /// Dataset not found
    #[error("Dataset not found: {0}")]
    DatasetNotFound(String),
    /// Invalid dataset format
    #[error("Invalid dataset format: {0}")]
    InvalidFormat(String),
    /// Annotation validation failed
    #[error("Annotation validation failed: {0}")]
    AnnotationValidationFailed(String),
    /// Version conflict detected
    #[error("Version conflict detected: {0}")]
    VersionConflict(String),
    /// Dataset corruption detected
    #[error("Dataset corruption detected: {0}")]
    DatasetCorruption(String),
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Dataset annotation quality levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnnotationQuality {
    /// Expert-level annotations with high confidence
    Expert,
    /// Professional-level annotations
    Professional,
    /// Community-contributed annotations
    Community,
    /// Automatically generated annotations
    Automatic,
    /// Research-quality annotations
    Research,
}

/// Ground truth annotation types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AnnotationType {
    /// Overall quality scores
    QualityScore,
    /// Pronunciation accuracy
    PronunciationAccuracy,
    /// Naturalness ratings
    Naturalness,
    /// Intelligibility scores
    Intelligibility,
    /// Emotional appropriateness
    EmotionalExpression,
    /// Prosodic correctness
    ProsodicAccuracy,
    /// Speaker similarity
    SpeakerSimilarity,
    /// Technical quality metrics
    TechnicalQuality,
}

/// Individual ground truth annotation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthAnnotation {
    /// Annotation ID
    pub id: String,
    /// Audio sample ID
    pub sample_id: String,
    /// Annotation type
    pub annotation_type: AnnotationType,
    /// Annotation value (0.0 - 1.0 or specific scale)
    pub value: f64,
    /// Scale used for annotation (e.g., "MOS_5", "binary", "percentage")
    pub scale: String,
    /// Annotator ID
    pub annotator_id: String,
    /// Annotation quality level
    pub quality_level: AnnotationQuality,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Optional text description
    pub description: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Audio sample in ground truth dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthSample {
    /// Sample ID
    pub id: String,
    /// Original audio file path
    pub audio_path: PathBuf,
    /// Reference audio path (if available)
    pub reference_path: Option<PathBuf>,
    /// Transcript text
    pub transcript: String,
    /// Language code
    pub language: String,
    /// Speaker ID
    pub speaker_id: String,
    /// Sample rate
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration: f64,
    /// Sample metadata
    pub metadata: HashMap<String, String>,
    /// Associated annotations
    pub annotations: Vec<GroundTruthAnnotation>,
    /// Quality validation status
    pub validation_status: ValidationStatus,
}

/// Dataset validation status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationStatus {
    /// Not yet validated
    Pending,
    /// Passed validation
    Valid,
    /// Failed validation
    Invalid,
    /// Validation in progress
    InProgress,
    /// Requires manual review
    NeedsReview,
}

/// Ground truth dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthDataset {
    /// Dataset ID
    pub id: String,
    /// Dataset name
    pub name: String,
    /// Dataset version
    pub version: String,
    /// Dataset description
    pub description: String,
    /// Creation date
    pub created_at: DateTime<Utc>,
    /// Last modification date
    pub modified_at: DateTime<Utc>,
    /// Dataset creator/organization
    pub creator: String,
    /// Dataset license
    pub license: String,
    /// Supported languages
    pub languages: Vec<String>,
    /// Number of samples
    pub sample_count: usize,
    /// Total duration in seconds
    pub total_duration: f64,
    /// Dataset purpose/domain
    pub domain: String,
    /// Annotation guidelines URL or description
    pub annotation_guidelines: Option<String>,
    /// Dataset samples
    pub samples: Vec<GroundTruthSample>,
    /// Quality metrics
    pub quality_metrics: DatasetQualityMetrics,
    /// Dataset tags
    pub tags: Vec<String>,
    /// Dataset metadata
    pub metadata: HashMap<String, String>,
}

/// Dataset quality metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetQualityMetrics {
    /// Overall quality score (0.0 - 1.0)
    pub overall_quality: f64,
    /// Annotation consistency score
    pub annotation_consistency: f64,
    /// Inter-annotator agreement (if multiple annotators)
    pub inter_annotator_agreement: Option<f64>,
    /// Audio quality score
    pub audio_quality: f64,
    /// Metadata completeness score
    pub metadata_completeness: f64,
    /// Validation completion percentage
    pub validation_completion: f64,
    /// Quality distribution by annotation type
    pub quality_by_type: HashMap<AnnotationType, f64>,
}

/// Dataset version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetVersion {
    /// Version string
    pub version: String,
    /// Creation date
    pub created_at: DateTime<Utc>,
    /// Version description/changelog
    pub description: String,
    /// Changes made in this version
    pub changes: Vec<String>,
    /// Previous version (if any)
    pub previous_version: Option<String>,
    /// Dataset snapshot path
    pub dataset_path: PathBuf,
    /// Version metadata
    pub metadata: HashMap<String, String>,
}

/// Ground truth dataset manager
#[derive(Debug)]
pub struct GroundTruthManager {
    /// Base directory for datasets
    base_path: PathBuf,
    /// Dataset catalog
    catalog: HashMap<String, GroundTruthDataset>,
    /// Version history
    version_history: HashMap<String, Vec<DatasetVersion>>,
    /// Data quality validator
    validator: DataQualityValidator,
}

impl GroundTruthManager {
    /// Create a new ground truth dataset manager
    pub fn new(base_path: PathBuf) -> Self {
        Self {
            base_path,
            catalog: HashMap::new(),
            version_history: HashMap::new(),
            validator: DataQualityValidator::default(),
        }
    }

    /// Initialize manager and load existing datasets
    pub async fn initialize(&mut self) -> Result<(), GroundTruthError> {
        self.ensure_directory_structure().await?;
        self.load_catalog().await?;
        self.load_version_history().await?;
        Ok(())
    }

    /// Create directory structure for dataset management
    async fn ensure_directory_structure(&self) -> Result<(), GroundTruthError> {
        let datasets_dir = self.base_path.join("datasets");
        let versions_dir = self.base_path.join("versions");
        let annotations_dir = self.base_path.join("annotations");
        let exports_dir = self.base_path.join("exports");

        tokio::fs::create_dir_all(&datasets_dir).await?;
        tokio::fs::create_dir_all(&versions_dir).await?;
        tokio::fs::create_dir_all(&annotations_dir).await?;
        tokio::fs::create_dir_all(&exports_dir).await?;

        Ok(())
    }

    /// Load dataset catalog from disk
    async fn load_catalog(&mut self) -> Result<(), GroundTruthError> {
        let catalog_path = self.base_path.join("catalog.json");
        if catalog_path.exists() {
            let content = tokio::fs::read_to_string(&catalog_path).await?;
            self.catalog = serde_json::from_str(&content)?;
        }
        Ok(())
    }

    /// Save dataset catalog to disk
    async fn save_catalog(&self) -> Result<(), GroundTruthError> {
        let catalog_path = self.base_path.join("catalog.json");
        let content = serde_json::to_string_pretty(&self.catalog)?;
        tokio::fs::write(&catalog_path, content).await?;
        Ok(())
    }

    /// Load version history from disk
    async fn load_version_history(&mut self) -> Result<(), GroundTruthError> {
        let versions_path = self.base_path.join("versions.json");
        if versions_path.exists() {
            let content = tokio::fs::read_to_string(&versions_path).await?;
            self.version_history = serde_json::from_str(&content)?;
        }
        Ok(())
    }

    /// Save version history to disk
    async fn save_version_history(&self) -> Result<(), GroundTruthError> {
        let versions_path = self.base_path.join("versions.json");
        let content = serde_json::to_string_pretty(&self.version_history)?;
        tokio::fs::write(&versions_path, content).await?;
        Ok(())
    }

    /// Create a new ground truth dataset
    pub async fn create_dataset(
        &mut self,
        name: String,
        description: String,
        creator: String,
        license: String,
        domain: String,
        languages: Vec<String>,
    ) -> Result<String, GroundTruthError> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now();

        let dataset = GroundTruthDataset {
            id: id.clone(),
            name,
            version: "1.0.0".to_string(),
            description,
            created_at: now,
            modified_at: now,
            creator,
            license,
            languages,
            sample_count: 0,
            total_duration: 0.0,
            domain,
            annotation_guidelines: None,
            samples: Vec::new(),
            quality_metrics: DatasetQualityMetrics {
                overall_quality: 0.0,
                annotation_consistency: 0.0,
                inter_annotator_agreement: None,
                audio_quality: 0.0,
                metadata_completeness: 0.0,
                validation_completion: 0.0,
                quality_by_type: HashMap::new(),
            },
            tags: Vec::new(),
            metadata: HashMap::new(),
        };

        // Create dataset directory
        let dataset_dir = self.base_path.join("datasets").join(&id);
        tokio::fs::create_dir_all(&dataset_dir).await?;

        // Save dataset
        self.catalog.insert(id.clone(), dataset);
        self.save_catalog().await?;

        // Create initial version
        self.create_version(
            &id,
            "1.0.0",
            "Initial dataset creation".to_string(),
            Vec::new(),
        )
        .await?;

        Ok(id)
    }

    /// Add a sample to a dataset
    pub async fn add_sample(
        &mut self,
        dataset_id: &str,
        audio_path: PathBuf,
        reference_path: Option<PathBuf>,
        transcript: String,
        language: String,
        speaker_id: String,
        metadata: HashMap<String, String>,
    ) -> Result<String, GroundTruthError> {
        let dataset = self
            .catalog
            .get_mut(dataset_id)
            .ok_or_else(|| GroundTruthError::DatasetNotFound(dataset_id.to_string()))?;

        // Validate audio file exists
        if !audio_path.exists() {
            return Err(GroundTruthError::InvalidFormat(format!(
                "Audio file not found: {:?}",
                audio_path
            )));
        }

        // Basic audio validation (could be enhanced with actual audio loading)
        let sample_id = uuid::Uuid::new_v4().to_string();
        let sample = GroundTruthSample {
            id: sample_id.clone(),
            audio_path,
            reference_path,
            transcript,
            language,
            speaker_id,
            sample_rate: 16000, // Default, should be detected from audio
            duration: 0.0,      // Should be calculated from audio
            metadata,
            annotations: Vec::new(),
            validation_status: ValidationStatus::Pending,
        };

        dataset.samples.push(sample);
        dataset.sample_count = dataset.samples.len();
        dataset.modified_at = Utc::now();

        self.save_catalog().await?;
        Ok(sample_id)
    }

    /// Add annotation to a sample
    pub async fn add_annotation(
        &mut self,
        dataset_id: &str,
        sample_id: &str,
        annotation_type: AnnotationType,
        value: f64,
        scale: String,
        annotator_id: String,
        quality_level: AnnotationQuality,
        confidence: f64,
        description: Option<String>,
        metadata: HashMap<String, String>,
    ) -> Result<String, GroundTruthError> {
        let dataset = self
            .catalog
            .get_mut(dataset_id)
            .ok_or_else(|| GroundTruthError::DatasetNotFound(dataset_id.to_string()))?;

        let sample = dataset
            .samples
            .iter_mut()
            .find(|s| s.id == sample_id)
            .ok_or_else(|| GroundTruthError::DatasetNotFound(sample_id.to_string()))?;

        let annotation_id = uuid::Uuid::new_v4().to_string();
        let annotation = GroundTruthAnnotation {
            id: annotation_id.clone(),
            sample_id: sample_id.to_string(),
            annotation_type,
            value,
            scale,
            annotator_id,
            quality_level,
            confidence,
            created_at: Utc::now(),
            description,
            metadata,
        };

        sample.annotations.push(annotation);
        dataset.modified_at = Utc::now();

        self.save_catalog().await?;
        Ok(annotation_id)
    }

    /// Validate dataset against quality standards
    pub async fn validate_dataset(
        &mut self,
        dataset_id: &str,
    ) -> Result<DatasetValidationReport, GroundTruthError> {
        let dataset = self
            .catalog
            .get_mut(dataset_id)
            .ok_or_else(|| GroundTruthError::DatasetNotFound(dataset_id.to_string()))?;

        // Prepare data for validation
        let mut audio_samples = Vec::new();
        let mut metadata_samples = Vec::new();

        for sample in &dataset.samples {
            // For now, create dummy audio data (in real implementation, load from file)
            let dummy_audio = vec![0.1_f32; 16000]; // 1 second of dummy audio
            audio_samples.push((dummy_audio, sample.sample_rate));

            // Convert sample metadata to required format
            let mut sample_metadata = sample.metadata.clone();
            sample_metadata.insert("language".to_string(), sample.language.clone());
            sample_metadata.insert("speaker".to_string(), sample.speaker_id.clone());
            sample_metadata.insert("transcript".to_string(), sample.transcript.clone());
            metadata_samples.push(sample_metadata);
        }

        // Run validation
        let validation_report = self
            .validator
            .validate_dataset(&dataset.name, &audio_samples, &metadata_samples)
            .map_err(|e| GroundTruthError::AnnotationValidationFailed(e.to_string()))?;

        // Update dataset quality metrics
        dataset.quality_metrics.overall_quality = validation_report.quality_score;
        dataset.quality_metrics.audio_quality = validation_report.quality_score;
        dataset.quality_metrics.metadata_completeness = validation_report
            .metadata_validation
            .values()
            .map(|&valid| if valid { 1.0 } else { 0.0 })
            .sum::<f64>()
            / validation_report.metadata_validation.len() as f64;

        // Update sample validation statuses
        for (i, sample) in dataset.samples.iter_mut().enumerate() {
            let sample_issues = validation_report
                .audio_issues
                .iter()
                .filter(|issue| issue.description.starts_with(&format!("Sample {}:", i)))
                .count();

            sample.validation_status = if sample_issues == 0 {
                ValidationStatus::Valid
            } else {
                ValidationStatus::Invalid
            };
        }

        dataset.modified_at = Utc::now();
        self.save_catalog().await?;

        Ok(validation_report)
    }

    /// Create a new version of a dataset
    pub async fn create_version(
        &mut self,
        dataset_id: &str,
        version: &str,
        description: String,
        changes: Vec<String>,
    ) -> Result<(), GroundTruthError> {
        let dataset = self
            .catalog
            .get(dataset_id)
            .ok_or_else(|| GroundTruthError::DatasetNotFound(dataset_id.to_string()))?;

        // Get previous version
        let previous_version = self
            .version_history
            .get(dataset_id)
            .and_then(|versions| versions.last())
            .map(|v| v.version.clone());

        // Check for version conflicts
        if let Some(versions) = self.version_history.get(dataset_id) {
            if versions.iter().any(|v| v.version == version) {
                return Err(GroundTruthError::VersionConflict(format!(
                    "Version {} already exists for dataset {}",
                    version, dataset_id
                )));
            }
        }

        // Save dataset snapshot
        let version_dir = self.base_path.join("versions").join(dataset_id);
        tokio::fs::create_dir_all(&version_dir).await?;

        let snapshot_path = version_dir.join(format!("{}.json", version));
        let dataset_content = serde_json::to_string_pretty(dataset)?;
        tokio::fs::write(&snapshot_path, dataset_content).await?;

        // Create version record
        let version_record = DatasetVersion {
            version: version.to_string(),
            created_at: Utc::now(),
            description,
            changes,
            previous_version,
            dataset_path: snapshot_path,
            metadata: HashMap::new(),
        };

        // Add to version history
        self.version_history
            .entry(dataset_id.to_string())
            .or_insert_with(Vec::new)
            .push(version_record);

        self.save_version_history().await?;
        Ok(())
    }

    /// Get dataset by ID
    pub fn get_dataset(&self, dataset_id: &str) -> Option<&GroundTruthDataset> {
        self.catalog.get(dataset_id)
    }

    /// List all datasets
    pub fn list_datasets(&self) -> Vec<&GroundTruthDataset> {
        self.catalog.values().collect()
    }

    /// Search datasets by criteria
    pub fn search_datasets(
        &self,
        language: Option<&str>,
        domain: Option<&str>,
        creator: Option<&str>,
        tags: Option<&[String]>,
    ) -> Vec<&GroundTruthDataset> {
        self.catalog
            .values()
            .filter(|dataset| {
                if let Some(lang) = language {
                    if !dataset.languages.contains(&lang.to_string()) {
                        return false;
                    }
                }
                if let Some(dom) = domain {
                    if dataset.domain != dom {
                        return false;
                    }
                }
                if let Some(cre) = creator {
                    if dataset.creator != cre {
                        return false;
                    }
                }
                if let Some(search_tags) = tags {
                    if !search_tags.iter().all(|tag| dataset.tags.contains(tag)) {
                        return false;
                    }
                }
                true
            })
            .collect()
    }

    /// Export dataset to a specified format
    pub async fn export_dataset(
        &self,
        dataset_id: &str,
        export_path: &Path,
        format: DatasetExportFormat,
    ) -> Result<(), GroundTruthError> {
        let dataset = self
            .catalog
            .get(dataset_id)
            .ok_or_else(|| GroundTruthError::DatasetNotFound(dataset_id.to_string()))?;

        match format {
            DatasetExportFormat::Json => {
                let content = serde_json::to_string_pretty(dataset)?;
                tokio::fs::write(export_path, content).await?;
            }
            DatasetExportFormat::Csv => {
                // Convert to CSV format (simplified)
                let mut csv_content = String::from(
                    "id,transcript,language,speaker_id,audio_path,annotations_count\n",
                );

                for sample in &dataset.samples {
                    csv_content.push_str(&format!(
                        "{},{},{},{},{},{}\n",
                        sample.id,
                        sample.transcript.replace(',', ";"),
                        sample.language,
                        sample.speaker_id,
                        sample.audio_path.display(),
                        sample.annotations.len()
                    ));
                }

                tokio::fs::write(export_path, csv_content).await?;
            }
        }

        Ok(())
    }

    /// Calculate inter-annotator agreement
    pub fn calculate_inter_annotator_agreement(
        &self,
        dataset_id: &str,
        annotation_type: &AnnotationType,
    ) -> Result<f64, GroundTruthError> {
        let dataset = self
            .catalog
            .get(dataset_id)
            .ok_or_else(|| GroundTruthError::DatasetNotFound(dataset_id.to_string()))?;

        let mut sample_agreements = Vec::new();

        for sample in &dataset.samples {
            let annotations: Vec<_> = sample
                .annotations
                .iter()
                .filter(|ann| ann.annotation_type == *annotation_type)
                .collect();

            if annotations.len() >= 2 {
                // Calculate pairwise agreement (simplified Pearson correlation)
                let values: Vec<f64> = annotations.iter().map(|ann| ann.value).collect();
                let mean = values.iter().sum::<f64>() / values.len() as f64;

                let variance =
                    values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

                // Use inverse of variance as agreement measure (lower variance = higher agreement)
                let agreement = if variance > 0.0 {
                    1.0 / (1.0 + variance)
                } else {
                    1.0
                };
                sample_agreements.push(agreement);
            }
        }

        if sample_agreements.is_empty() {
            Ok(0.0)
        } else {
            Ok(sample_agreements.iter().sum::<f64>() / sample_agreements.len() as f64)
        }
    }
}

/// Dataset export formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatasetExportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_ground_truth_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = GroundTruthManager::new(temp_dir.path().to_path_buf());

        manager.initialize().await.unwrap();
        assert!(manager.catalog.is_empty());
    }

    #[tokio::test]
    async fn test_dataset_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = GroundTruthManager::new(temp_dir.path().to_path_buf());
        manager.initialize().await.unwrap();

        let dataset_id = manager
            .create_dataset(
                "Test Dataset".to_string(),
                "A test dataset".to_string(),
                "Test Creator".to_string(),
                "MIT".to_string(),
                "speech".to_string(),
                vec!["en".to_string()],
            )
            .await
            .unwrap();

        assert!(!dataset_id.is_empty());
        assert!(manager.catalog.contains_key(&dataset_id));
    }

    #[tokio::test]
    async fn test_annotation_addition() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = GroundTruthManager::new(temp_dir.path().to_path_buf());
        manager.initialize().await.unwrap();

        let dataset_id = manager
            .create_dataset(
                "Test Dataset".to_string(),
                "A test dataset".to_string(),
                "Test Creator".to_string(),
                "MIT".to_string(),
                "speech".to_string(),
                vec!["en".to_string()],
            )
            .await
            .unwrap();

        // Create a temporary audio file
        let audio_file = temp_dir.path().join("test.wav");
        tokio::fs::write(&audio_file, b"dummy audio content")
            .await
            .unwrap();

        let sample_id = manager
            .add_sample(
                &dataset_id,
                audio_file,
                None,
                "Hello world".to_string(),
                "en".to_string(),
                "speaker1".to_string(),
                HashMap::new(),
            )
            .await
            .unwrap();

        let annotation_id = manager
            .add_annotation(
                &dataset_id,
                &sample_id,
                AnnotationType::QualityScore,
                0.85,
                "MOS_5".to_string(),
                "annotator1".to_string(),
                AnnotationQuality::Expert,
                0.9,
                Some("High quality sample".to_string()),
                HashMap::new(),
            )
            .await
            .unwrap();

        assert!(!annotation_id.is_empty());

        let dataset = manager.get_dataset(&dataset_id).unwrap();
        assert_eq!(dataset.samples.len(), 1);
        assert_eq!(dataset.samples[0].annotations.len(), 1);
    }

    #[tokio::test]
    async fn test_dataset_search() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = GroundTruthManager::new(temp_dir.path().to_path_buf());
        manager.initialize().await.unwrap();

        let _dataset_id1 = manager
            .create_dataset(
                "English Dataset".to_string(),
                "English speech dataset".to_string(),
                "Creator1".to_string(),
                "MIT".to_string(),
                "speech".to_string(),
                vec!["en".to_string()],
            )
            .await
            .unwrap();

        let _dataset_id2 = manager
            .create_dataset(
                "Spanish Dataset".to_string(),
                "Spanish speech dataset".to_string(),
                "Creator2".to_string(),
                "MIT".to_string(),
                "music".to_string(),
                vec!["es".to_string()],
            )
            .await
            .unwrap();

        let english_datasets = manager.search_datasets(Some("en"), None, None, None);
        assert_eq!(english_datasets.len(), 1);
        assert_eq!(english_datasets[0].name, "English Dataset");

        let speech_datasets = manager.search_datasets(None, Some("speech"), None, None);
        assert_eq!(speech_datasets.len(), 1);
        assert_eq!(speech_datasets[0].domain, "speech");
    }
}
