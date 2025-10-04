//! Core traits for dataset handling
//!
//! This module defines the main Dataset trait and related abstractions
//! for working with speech synthesis datasets.

use crate::{DatasetError, DatasetStatistics, ValidationReport};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result type for dataset operations
pub type Result<T> = std::result::Result<T, DatasetError>;

/// Dataset metadata containing information about the dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Dataset name
    pub name: String,
    /// Dataset version
    pub version: String,
    /// Dataset description
    pub description: Option<String>,
    /// Total number of samples
    pub total_samples: usize,
    /// Total duration in seconds
    pub total_duration: f32,
    /// Languages present in the dataset
    pub languages: Vec<String>,
    /// Speakers present in the dataset
    pub speakers: Vec<String>,
    /// License information
    pub license: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

// Re-export SplitConfig from splits module to maintain compatibility
pub use crate::splits::SplitConfig;

/// Dataset split containing train, validation, and test sets
#[derive(Debug)]
pub struct DatasetSplit<T> {
    /// Training dataset
    pub train: T,
    /// Validation dataset
    pub validation: T,
    /// Test dataset
    pub test: T,
}

/// Dataset split containing indices for train, validation, and test sets
/// This is object-safe and can be used with trait objects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSplitIndices {
    /// Training set indices
    pub train: Vec<usize>,
    /// Validation set indices
    pub validation: Vec<usize>,
    /// Test set indices
    pub test: Vec<usize>,
}

impl DatasetSplitIndices {
    /// Create a new dataset split with given indices
    pub fn new(train: Vec<usize>, validation: Vec<usize>, test: Vec<usize>) -> Self {
        Self {
            train,
            validation,
            test,
        }
    }

    /// Get the total number of samples across all splits
    pub fn total_samples(&self) -> usize {
        self.train.len() + self.validation.len() + self.test.len()
    }

    /// Get the ratio of each split
    pub fn ratios(&self) -> (f32, f32, f32) {
        let total = self.total_samples() as f32;
        if total == 0.0 {
            return (0.0, 0.0, 0.0);
        }
        (
            self.train.len() as f32 / total,
            self.validation.len() as f32 / total,
            self.test.len() as f32 / total,
        )
    }

    /// Validate that all indices are unique and within bounds
    pub fn validate(&self, dataset_size: usize) -> Result<()> {
        let mut all_indices = Vec::new();
        all_indices.extend(&self.train);
        all_indices.extend(&self.validation);
        all_indices.extend(&self.test);

        // Check for out-of-bounds indices
        for &index in &all_indices {
            if index >= dataset_size {
                return Err(crate::DatasetError::IndexError(index));
            }
        }

        // Check for duplicates
        all_indices.sort_unstable();
        for window in all_indices.windows(2) {
            if window[0] == window[1] {
                return Err(crate::DatasetError::ValidationError(format!(
                    "Duplicate index {} found in splits",
                    window[0]
                )));
            }
        }

        Ok(())
    }
}

/// Main Dataset trait for different dataset implementations
#[async_trait]
pub trait Dataset: Send + Sync {
    /// Dataset sample type
    type Sample: DatasetSample;

    /// Get the number of samples in the dataset
    fn len(&self) -> usize;

    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a sample by index
    async fn get(&self, index: usize) -> Result<Self::Sample>;

    /// Get multiple samples by indices
    async fn get_batch(&self, indices: &[usize]) -> Result<Vec<Self::Sample>> {
        let mut samples = Vec::with_capacity(indices.len());
        for &index in indices {
            samples.push(self.get(index).await?);
        }
        Ok(samples)
    }

    /// Create an iterator over all samples
    fn iter(&self) -> DatasetIterator<Self::Sample> {
        DatasetIterator::new(self.len())
    }

    /// Get dataset metadata
    fn metadata(&self) -> &DatasetMetadata;

    /// Split the dataset into train/validation/test index sets (object-safe)
    /// Returns indices for each split rather than new dataset objects
    async fn split_indices(&self, config: SplitConfig) -> Result<DatasetSplitIndices> {
        use scirs2_core::random::seq::SliceRandom;
        use scirs2_core::random::SeedableRng;

        let dataset_size = self.len();
        if dataset_size == 0 {
            return Err(crate::DatasetError::SplitError(String::from(
                "Cannot split empty dataset",
            )));
        }

        // Validate split ratios
        let sum = config.train_ratio + config.val_ratio + config.test_ratio;
        if (sum - 1.0).abs() > 1e-6 {
            return Err(crate::DatasetError::SplitError(format!(
                "Split ratios must sum to 1.0, got {sum}"
            )));
        }

        let mut indices: Vec<usize> = (0..dataset_size).collect();

        // Handle different splitting strategies
        match config.strategy {
            crate::splits::SplitStrategy::Random => {
                // Random shuffle
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
                indices.shuffle(&mut rng);
            }
            crate::splits::SplitStrategy::Stratified => {
                // For stratified splitting, we need speaker information
                // This is a simplified version - ideally we'd group by speaker first
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
                indices.shuffle(&mut rng);
            }
            _ => {
                // For duration and text length based splitting, we would need to
                // collect samples and sort them, but for now use random as fallback
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
                indices.shuffle(&mut rng);
            }
        }

        // Calculate split sizes
        let train_size = (dataset_size as f32 * config.train_ratio).round() as usize;
        let val_size = (dataset_size as f32 * config.val_ratio).round() as usize;
        let _test_size = dataset_size - train_size - val_size;

        // Split indices
        let train_indices = indices[0..train_size].to_vec();
        let val_indices = indices[train_size..train_size + val_size].to_vec();
        let test_indices = indices[train_size + val_size..].to_vec();

        let split_indices = DatasetSplitIndices::new(train_indices, val_indices, test_indices);
        split_indices.validate(self.len())?;
        Ok(split_indices)
    }

    /// Get dataset statistics
    async fn statistics(&self) -> Result<DatasetStatistics>;

    /// Validate the dataset
    async fn validate(&self) -> Result<ValidationReport>;

    // Filter samples based on a predicate (not object-safe due to generics)
    // async fn filter<F>(&self, predicate: F) -> Result<Vec<usize>>
    // where
    //     F: Fn(&Self::Sample) -> bool + Send + Sync,
    // {
    //     let mut filtered_indices = Vec::new();
    //     for i in 0..self.len() {
    //         let sample = self.get(i).await?;
    //         if predicate(&sample) {
    //             filtered_indices.push(i);
    //         }
    //     }
    //     Ok(filtered_indices)
    // }

    /// Get a random sample
    async fn get_random(&self) -> Result<Self::Sample> {
        if self.is_empty() {
            return Err(DatasetError::IndexError(0));
        }
        let len = self.len();
        let index = {
            use scirs2_core::random::{thread_rng, Rng};
            let mut rng = thread_rng();
            rng.gen_range(0..len)
        };
        self.get(index).await
    }
}

/// Dataset iterator for efficient sample access
pub struct DatasetIterator<T> {
    current: usize,
    total: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> DatasetIterator<T> {
    pub fn new(total: usize) -> Self {
        Self {
            current: 0,
            total,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Iterator for DatasetIterator<T> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.total {
            let index = self.current;
            self.current += 1;
            Some(index)
        } else {
            None
        }
    }
}

/// Trait for dataset samples
pub trait DatasetSample: Clone + Send + Sync {
    /// Get the sample's unique identifier
    fn id(&self) -> &str;

    /// Get the sample's text content
    fn text(&self) -> &str;

    /// Get the sample's duration in seconds
    fn duration(&self) -> f32;

    /// Get the sample's language
    fn language(&self) -> &str;

    /// Get the sample's speaker ID (if available)
    fn speaker_id(&self) -> Option<&str>;
}

// Implement DatasetSample for our DatasetSample struct
impl DatasetSample for crate::DatasetSample {
    fn id(&self) -> &str {
        &self.id
    }

    fn text(&self) -> &str {
        &self.text
    }

    fn duration(&self) -> f32 {
        self.audio.duration()
    }

    fn language(&self) -> &str {
        self.language.as_str()
    }

    fn speaker_id(&self) -> Option<&str> {
        self.speaker.as_ref().map(|s| s.id.as_str())
    }
}
