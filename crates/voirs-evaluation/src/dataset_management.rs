//! Evaluation Dataset Management System
//!
//! This module provides comprehensive dataset management functionality for speech synthesis
//! evaluation, including dataset registration, validation, organization, and access control.

use crate::EvaluationError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::fs;
use voirs_sdk::{AudioBuffer, LanguageCode};

/// Errors specific to dataset management
#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Dataset not found: {0}")]
    DatasetNotFound(String),
    #[error("Invalid dataset: {0}")]
    InvalidDataset(String),
    #[error("Dataset validation failed: {0}")]
    ValidationFailed(String),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    #[error("Duplicate dataset: {0}")]
    DuplicateDataset(String),
}

/// Dataset category for organization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DatasetCategory {
    /// Reference datasets for quality comparison
    Reference,
    /// Test datasets for evaluation
    Test,
    /// Training datasets (for metric training)
    Training,
    /// Validation datasets
    Validation,
    /// Benchmark datasets for standardized evaluation
    Benchmark,
    /// User-submitted datasets
    UserSubmitted,
    /// Synthetic datasets
    Synthetic,
    /// Cross-language datasets
    CrossLanguage,
}

/// Audio quality level for dataset classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AudioQuality {
    /// High quality studio recordings
    Studio,
    /// Broadcast quality
    Broadcast,
    /// Telephone quality
    Telephone,
    /// Compressed audio
    Compressed,
    /// Low quality/noisy audio
    LowQuality,
    /// Mixed quality levels
    Mixed,
}

/// Dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Unique dataset identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Dataset description
    pub description: String,
    /// Dataset category
    pub category: DatasetCategory,
    /// Primary language
    pub language: LanguageCode,
    /// Additional languages (for multilingual datasets)
    pub additional_languages: Vec<LanguageCode>,
    /// Audio quality level
    pub audio_quality: AudioQuality,
    /// Number of audio samples
    pub sample_count: usize,
    /// Total duration in seconds
    pub total_duration: f64,
    /// Sample rate (Hz)
    pub sample_rate: u32,
    /// Audio format
    pub audio_format: String,
    /// Dataset version
    pub version: String,
    /// Creator/organization
    pub creator: String,
    /// License information
    pub license: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modification timestamp
    pub modified_at: u64,
    /// Tags for searching and filtering
    pub tags: HashSet<String>,
    /// File path or URL
    pub location: PathBuf,
    /// Dataset size in bytes
    pub size_bytes: u64,
    /// Validation status
    pub is_validated: bool,
    /// Access permissions
    pub access_level: AccessLevel,
}

/// Access control level for datasets
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AccessLevel {
    /// Public access
    Public,
    /// Restricted access (requires permission)
    Restricted,
    /// Private access (owner only)
    Private,
    /// Read-only access
    ReadOnly,
}

/// Dataset validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation score (0.0 to 1.0)
    pub score: f64,
    /// Issues found during validation
    pub issues: Vec<ValidationIssue>,
    /// Validation timestamp
    pub validated_at: u64,
    /// Validation duration in milliseconds
    pub validation_duration_ms: u64,
}

/// Individual validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue category
    pub category: String,
    /// Human-readable description
    pub description: String,
    /// File or sample where issue was found
    pub location: Option<String>,
    /// Suggested fix (if available)
    pub suggestion: Option<String>,
}

/// Severity level for validation issues
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IssueSeverity {
    /// Critical issues that prevent dataset use
    Critical,
    /// Warning issues that may affect quality
    Warning,
    /// Minor issues for information
    Info,
}

/// Search criteria for finding datasets
#[derive(Debug, Clone, Default)]
pub struct DatasetSearchCriteria {
    /// Filter by category
    pub category: Option<DatasetCategory>,
    /// Filter by language
    pub language: Option<LanguageCode>,
    /// Filter by audio quality
    pub audio_quality: Option<AudioQuality>,
    /// Filter by tags (all must match)
    pub tags: HashSet<String>,
    /// Minimum sample count
    pub min_samples: Option<usize>,
    /// Maximum sample count
    pub max_samples: Option<usize>,
    /// Minimum duration (seconds)
    pub min_duration: Option<f64>,
    /// Maximum duration (seconds)
    pub max_duration: Option<f64>,
    /// Text search in name/description
    pub text_search: Option<String>,
    /// Only validated datasets
    pub validated_only: bool,
    /// Access level filter
    pub access_level: Option<AccessLevel>,
}

/// Dataset management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManagerConfig {
    /// Base directory for dataset storage
    pub base_directory: PathBuf,
    /// Maximum dataset size in bytes
    pub max_dataset_size: u64,
    /// Automatic validation on registration
    pub auto_validate: bool,
    /// Enable dataset caching
    pub enable_caching: bool,
    /// Cache size limit in bytes
    pub cache_size_limit: u64,
    /// Default access level for new datasets
    pub default_access_level: AccessLevel,
}

impl Default for DatasetManagerConfig {
    fn default() -> Self {
        Self {
            base_directory: PathBuf::from("datasets"),
            max_dataset_size: 10 * 1024 * 1024 * 1024, // 10GB
            auto_validate: true,
            enable_caching: true,
            cache_size_limit: 1024 * 1024 * 1024, // 1GB
            default_access_level: AccessLevel::Public,
        }
    }
}

/// Main dataset management system
pub struct DatasetManager {
    config: DatasetManagerConfig,
    datasets: HashMap<String, DatasetMetadata>,
    validation_cache: HashMap<String, ValidationResult>,
    access_permissions: HashMap<String, HashSet<String>>, // dataset_id -> user_ids
}

impl DatasetManager {
    /// Create a new dataset manager
    pub async fn new(config: DatasetManagerConfig) -> Result<Self, DatasetError> {
        fs::create_dir_all(&config.base_directory).await?;

        let mut manager = Self {
            config,
            datasets: HashMap::new(),
            validation_cache: HashMap::new(),
            access_permissions: HashMap::new(),
        };

        manager.load_datasets().await?;
        Ok(manager)
    }

    /// Load existing datasets from disk
    async fn load_datasets(&mut self) -> Result<(), DatasetError> {
        let metadata_dir = self.config.base_directory.join("metadata");

        if !metadata_dir.exists() {
            fs::create_dir_all(&metadata_dir).await?;
            return Ok(());
        }

        let mut entries = fs::read_dir(&metadata_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
                let contents = fs::read_to_string(entry.path()).await?;
                let metadata: DatasetMetadata = serde_json::from_str(&contents)?;
                self.datasets.insert(metadata.id.clone(), metadata);
            }
        }

        Ok(())
    }

    /// Save dataset metadata to disk
    async fn save_dataset_metadata(&self, metadata: &DatasetMetadata) -> Result<(), DatasetError> {
        let metadata_dir = self.config.base_directory.join("metadata");
        fs::create_dir_all(&metadata_dir).await?;

        let file_path = metadata_dir.join(format!("{}.json", metadata.id));
        let contents = serde_json::to_string_pretty(metadata)?;
        fs::write(file_path, contents).await?;

        Ok(())
    }

    /// Register a new dataset
    pub async fn register_dataset(
        &mut self,
        metadata: DatasetMetadata,
    ) -> Result<(), DatasetError> {
        // Check for duplicates
        if self.datasets.contains_key(&metadata.id) {
            return Err(DatasetError::DuplicateDataset(metadata.id));
        }

        // Validate dataset size
        if metadata.size_bytes > self.config.max_dataset_size {
            return Err(DatasetError::InvalidDataset(format!(
                "Dataset size ({} bytes) exceeds limit ({} bytes)",
                metadata.size_bytes, self.config.max_dataset_size
            )));
        }

        // Perform automatic validation if enabled
        if self.config.auto_validate {
            let validation_result = self.validate_dataset(&metadata).await?;
            if !validation_result.is_valid {
                return Err(DatasetError::ValidationFailed(format!(
                    "Dataset validation failed with {} issues",
                    validation_result.issues.len()
                )));
            }
            self.validation_cache
                .insert(metadata.id.clone(), validation_result);
        }

        // Save metadata
        self.save_dataset_metadata(&metadata).await?;

        // Add to registry
        self.datasets.insert(metadata.id.clone(), metadata);

        Ok(())
    }

    /// Validate a dataset
    pub async fn validate_dataset(
        &self,
        metadata: &DatasetMetadata,
    ) -> Result<ValidationResult, DatasetError> {
        let start_time = SystemTime::now();
        let mut issues = Vec::new();
        let mut score: f64 = 1.0;

        // Check if dataset location exists
        if !metadata.location.exists() {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Critical,
                category: "file_access".to_string(),
                description: "Dataset location does not exist".to_string(),
                location: Some(metadata.location.to_string_lossy().to_string()),
                suggestion: Some("Verify the dataset path is correct".to_string()),
            });
            score -= 0.5;
        }

        // Validate metadata completeness
        if metadata.name.is_empty() {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                category: "metadata".to_string(),
                description: "Dataset name is empty".to_string(),
                location: None,
                suggestion: Some("Provide a descriptive name".to_string()),
            });
            score -= 0.1;
        }

        if metadata.description.len() < 10 {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Info,
                category: "metadata".to_string(),
                description: "Dataset description is very short".to_string(),
                location: None,
                suggestion: Some("Consider adding more detailed description".to_string()),
            });
            score -= 0.05;
        }

        // Validate audio properties
        if metadata.sample_rate < 8000 || metadata.sample_rate > 48000 {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                category: "audio_quality".to_string(),
                description: format!("Unusual sample rate: {} Hz", metadata.sample_rate),
                location: None,
                suggestion: Some("Standard rates are 16kHz, 22kHz, 44.1kHz, 48kHz".to_string()),
            });
            score -= 0.1;
        }

        if metadata.sample_count == 0 {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Critical,
                category: "content".to_string(),
                description: "Dataset contains no samples".to_string(),
                location: None,
                suggestion: Some("Add audio samples to the dataset".to_string()),
            });
            score -= 0.3;
        }

        // Validate license information
        if metadata.license.is_empty() || metadata.license == "unknown" {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                category: "legal".to_string(),
                description: "No license information provided".to_string(),
                location: None,
                suggestion: Some("Specify the dataset license for legal compliance".to_string()),
            });
            score -= 0.1;
        }

        let validation_duration = start_time.elapsed().unwrap().as_millis() as u64;
        let is_valid = issues
            .iter()
            .all(|issue| issue.severity != IssueSeverity::Critical);

        Ok(ValidationResult {
            is_valid,
            score: score.max(0.0),
            issues,
            validated_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            validation_duration_ms: validation_duration,
        })
    }

    /// Search for datasets based on criteria
    pub fn search_datasets(&self, criteria: &DatasetSearchCriteria) -> Vec<&DatasetMetadata> {
        self.datasets
            .values()
            .filter(|dataset| self.matches_criteria(dataset, criteria))
            .collect()
    }

    /// Check if a dataset matches search criteria
    fn matches_criteria(
        &self,
        dataset: &DatasetMetadata,
        criteria: &DatasetSearchCriteria,
    ) -> bool {
        // Category filter
        if let Some(ref category) = criteria.category {
            if &dataset.category != category {
                return false;
            }
        }

        // Language filter
        if let Some(ref language) = criteria.language {
            if &dataset.language != language && !dataset.additional_languages.contains(language) {
                return false;
            }
        }

        // Audio quality filter
        if let Some(ref quality) = criteria.audio_quality {
            if &dataset.audio_quality != quality {
                return false;
            }
        }

        // Tags filter (all must match)
        if !criteria.tags.is_empty() && !criteria.tags.is_subset(&dataset.tags) {
            return false;
        }

        // Sample count filters
        if let Some(min_samples) = criteria.min_samples {
            if dataset.sample_count < min_samples {
                return false;
            }
        }

        if let Some(max_samples) = criteria.max_samples {
            if dataset.sample_count > max_samples {
                return false;
            }
        }

        // Duration filters
        if let Some(min_duration) = criteria.min_duration {
            if dataset.total_duration < min_duration {
                return false;
            }
        }

        if let Some(max_duration) = criteria.max_duration {
            if dataset.total_duration > max_duration {
                return false;
            }
        }

        // Text search
        if let Some(ref search_text) = criteria.text_search {
            let search_lower = search_text.to_lowercase();
            if !dataset.name.to_lowercase().contains(&search_lower)
                && !dataset.description.to_lowercase().contains(&search_lower)
            {
                return false;
            }
        }

        // Validation filter
        if criteria.validated_only && !dataset.is_validated {
            return false;
        }

        // Access level filter
        if let Some(ref access_level) = criteria.access_level {
            if &dataset.access_level != access_level {
                return false;
            }
        }

        true
    }

    /// Get dataset by ID
    pub fn get_dataset(&self, id: &str) -> Option<&DatasetMetadata> {
        self.datasets.get(id)
    }

    /// List all datasets
    pub fn list_datasets(&self) -> Vec<&DatasetMetadata> {
        self.datasets.values().collect()
    }

    /// Get dataset statistics
    pub fn get_statistics(&self) -> DatasetStatistics {
        let total_datasets = self.datasets.len();
        let total_samples: usize = self.datasets.values().map(|d| d.sample_count).sum();
        let total_duration: f64 = self.datasets.values().map(|d| d.total_duration).sum();
        let total_size: u64 = self.datasets.values().map(|d| d.size_bytes).sum();

        let mut category_counts = HashMap::new();
        let mut language_counts = HashMap::new();
        let mut quality_counts = HashMap::new();

        for dataset in self.datasets.values() {
            *category_counts.entry(dataset.category.clone()).or_insert(0) += 1;
            *language_counts.entry(dataset.language.clone()).or_insert(0) += 1;
            *quality_counts
                .entry(dataset.audio_quality.clone())
                .or_insert(0) += 1;
        }

        let validated_count = self.datasets.values().filter(|d| d.is_validated).count();

        DatasetStatistics {
            total_datasets,
            validated_datasets: validated_count,
            total_samples,
            total_duration,
            total_size,
            category_distribution: category_counts,
            language_distribution: language_counts,
            quality_distribution: quality_counts,
        }
    }

    /// Remove a dataset
    pub async fn remove_dataset(&mut self, id: &str) -> Result<(), DatasetError> {
        if !self.datasets.contains_key(id) {
            return Err(DatasetError::DatasetNotFound(id.to_string()));
        }

        // Remove metadata file
        let metadata_file = self
            .config
            .base_directory
            .join("metadata")
            .join(format!("{}.json", id));

        if metadata_file.exists() {
            fs::remove_file(metadata_file).await?;
        }

        // Remove from registry
        self.datasets.remove(id);
        self.validation_cache.remove(id);
        self.access_permissions.remove(id);

        Ok(())
    }

    /// Update dataset metadata
    pub async fn update_dataset(&mut self, metadata: DatasetMetadata) -> Result<(), DatasetError> {
        if !self.datasets.contains_key(&metadata.id) {
            return Err(DatasetError::DatasetNotFound(metadata.id.clone()));
        }

        // Re-validate if auto-validation is enabled
        if self.config.auto_validate {
            let validation_result = self.validate_dataset(&metadata).await?;
            self.validation_cache
                .insert(metadata.id.clone(), validation_result);
        }

        // Save updated metadata
        self.save_dataset_metadata(&metadata).await?;

        // Update registry
        self.datasets.insert(metadata.id.clone(), metadata);

        Ok(())
    }

    /// Generate a comprehensive report
    pub fn generate_report(&self) -> String {
        let stats = self.get_statistics();
        let mut report = String::new();

        report.push_str("# Dataset Management Report\n\n");

        // Overview
        report.push_str("## Overview\n\n");
        report.push_str(&format!("- **Total Datasets**: {}\n", stats.total_datasets));
        report.push_str(&format!(
            "- **Validated Datasets**: {} ({:.1}%)\n",
            stats.validated_datasets,
            (stats.validated_datasets as f64 / stats.total_datasets as f64) * 100.0
        ));
        report.push_str(&format!("- **Total Samples**: {}\n", stats.total_samples));
        report.push_str(&format!(
            "- **Total Duration**: {:.2} hours\n",
            stats.total_duration / 3600.0
        ));
        report.push_str(&format!(
            "- **Total Size**: {:.2} GB\n",
            stats.total_size as f64 / (1024.0 * 1024.0 * 1024.0)
        ));

        // Category distribution
        report.push_str("\n## Category Distribution\n\n");
        for (category, count) in &stats.category_distribution {
            report.push_str(&format!("- **{:?}**: {}\n", category, count));
        }

        // Language distribution
        report.push_str("\n## Language Distribution\n\n");
        for (language, count) in &stats.language_distribution {
            report.push_str(&format!("- **{:?}**: {}\n", language, count));
        }

        // Quality distribution
        report.push_str("\n## Quality Distribution\n\n");
        for (quality, count) in &stats.quality_distribution {
            report.push_str(&format!("- **{:?}**: {}\n", quality, count));
        }

        report
    }
}

/// Dataset statistics
#[derive(Debug, Clone)]
pub struct DatasetStatistics {
    pub total_datasets: usize,
    pub validated_datasets: usize,
    pub total_samples: usize,
    pub total_duration: f64,
    pub total_size: u64,
    pub category_distribution: HashMap<DatasetCategory, usize>,
    pub language_distribution: HashMap<LanguageCode, usize>,
    pub quality_distribution: HashMap<AudioQuality, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_metadata(id: &str) -> DatasetMetadata {
        let mut tags = HashSet::new();
        tags.insert("test".to_string());
        tags.insert("evaluation".to_string());

        DatasetMetadata {
            id: id.to_string(),
            name: format!("Test Dataset {}", id),
            description: "A test dataset for evaluation".to_string(),
            category: DatasetCategory::Test,
            language: LanguageCode::EnUs,
            additional_languages: vec![],
            audio_quality: AudioQuality::Studio,
            sample_count: 100,
            total_duration: 300.0,
            sample_rate: 16000,
            audio_format: "wav".to_string(),
            version: "1.0".to_string(),
            creator: "Test Creator".to_string(),
            license: "MIT".to_string(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            modified_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            tags,
            location: PathBuf::from(
                "/nonexistent/path/that/definitely/does/not/exist/test_dataset",
            ),
            size_bytes: 1024 * 1024, // 1MB
            is_validated: false,
            access_level: AccessLevel::Public,
        }
    }

    #[tokio::test]
    async fn test_dataset_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = DatasetManagerConfig {
            base_directory: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = DatasetManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_dataset_registration() {
        let temp_dir = TempDir::new().unwrap();
        let config = DatasetManagerConfig {
            base_directory: temp_dir.path().to_path_buf(),
            auto_validate: false, // Disable validation for this test
            ..Default::default()
        };

        let mut manager = DatasetManager::new(config).await.unwrap();
        let metadata = create_test_metadata("test1");

        let result = manager.register_dataset(metadata.clone()).await;
        assert!(result.is_ok());

        let retrieved = manager.get_dataset("test1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, metadata.name);
    }

    #[tokio::test]
    async fn test_dataset_search() {
        let temp_dir = TempDir::new().unwrap();
        let config = DatasetManagerConfig {
            base_directory: temp_dir.path().to_path_buf(),
            auto_validate: false,
            ..Default::default()
        };

        let mut manager = DatasetManager::new(config).await.unwrap();

        // Register multiple datasets
        for i in 1..=3 {
            let metadata = create_test_metadata(&format!("test{}", i));
            manager.register_dataset(metadata).await.unwrap();
        }

        // Search by category
        let criteria = DatasetSearchCriteria {
            category: Some(DatasetCategory::Test),
            ..Default::default()
        };

        let results = manager.search_datasets(&criteria);
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_dataset_validation() {
        let temp_dir = TempDir::new().unwrap();
        let config = DatasetManagerConfig {
            base_directory: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = DatasetManager::new(config).await.unwrap();
        let metadata = create_test_metadata("test_validation");

        let validation_result = manager.validate_dataset(&metadata).await.unwrap();
        assert!(!validation_result.is_valid); // Should fail because location doesn't exist
        assert!(!validation_result.issues.is_empty());
    }

    #[tokio::test]
    async fn test_statistics_generation() {
        let temp_dir = TempDir::new().unwrap();
        let config = DatasetManagerConfig {
            base_directory: temp_dir.path().to_path_buf(),
            auto_validate: false,
            ..Default::default()
        };

        let mut manager = DatasetManager::new(config).await.unwrap();

        // Register datasets with different categories
        let mut metadata1 = create_test_metadata("test1");
        metadata1.category = DatasetCategory::Test;
        manager.register_dataset(metadata1).await.unwrap();

        let mut metadata2 = create_test_metadata("test2");
        metadata2.category = DatasetCategory::Reference;
        manager.register_dataset(metadata2).await.unwrap();

        let stats = manager.get_statistics();
        assert_eq!(stats.total_datasets, 2);
        assert_eq!(stats.category_distribution.len(), 2);
    }
}
