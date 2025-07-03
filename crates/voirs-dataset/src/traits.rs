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

/// Dataset split configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitConfig {
    /// Training split ratio (0.0-1.0)
    pub train_ratio: f32,
    /// Validation split ratio (0.0-1.0)
    pub val_ratio: f32,
    /// Test split ratio (0.0-1.0)
    pub test_ratio: f32,
    /// Random seed for reproducible splits
    pub seed: Option<u64>,
    /// Split by speaker (to avoid speaker leakage)
    pub split_by_speaker: bool,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.8,
            val_ratio: 0.1,
            test_ratio: 0.1,
            seed: None,
            split_by_speaker: false,
        }
    }
}

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
    
    // TODO: Implement object-safe methods - commented out for now
    // Split the dataset into train/validation/test sets
    // async fn split(&self, config: SplitConfig) -> Result<DatasetSplit<Box<dyn Dataset<Sample = Self::Sample>>>>;
    
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
        let index = rand::random::<usize>() % self.len();
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