//! Core types and traits for feature learning
//!
//! This module provides the main trait and data structures for feature learning,
//! including the FeatureLearner trait and learned feature representations.

use crate::{AudioData, DatasetSample, QualityMetrics, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Learned features representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedFeatures {
    /// Audio features
    pub audio_features: Option<Vec<f32>>,
    /// Speaker embedding
    pub speaker_embedding: Option<Vec<f32>>,
    /// Content embedding
    pub content_embedding: Option<Vec<f32>>,
    /// Predicted quality metrics
    pub predicted_quality: Option<QualityMetrics>,
    /// Feature metadata
    pub metadata: HashMap<String, String>,
}

/// Feature learning interface
#[async_trait::async_trait]
pub trait FeatureLearner: Send + Sync {
    /// Extract audio features from a sample
    async fn extract_audio_features(&self, audio: &AudioData) -> Result<Vec<f32>>;

    /// Extract speaker embedding from audio
    async fn extract_speaker_embedding(&self, audio: &AudioData) -> Result<Vec<f32>>;

    /// Extract content embedding from text/phonemes
    async fn extract_content_embedding(&self, sample: &DatasetSample) -> Result<Vec<f32>>;

    /// Predict quality metrics
    async fn predict_quality(&self, sample: &DatasetSample) -> Result<QualityMetrics>;

    /// Extract all features from a sample
    async fn extract_all_features(&self, sample: &DatasetSample) -> Result<LearnedFeatures>;

    /// Train or fine-tune the feature extractor
    async fn train(&mut self, samples: &[DatasetSample]) -> Result<()>;

    /// Save the trained model
    async fn save_model(&self, path: &str) -> Result<()>;

    /// Load a pre-trained model
    async fn load_model(&mut self, path: &str) -> Result<()>;

    /// Get feature dimensions
    fn get_feature_dimensions(&self) -> FeatureDimensions;
}

/// Feature dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDimensions {
    /// Audio feature dimension
    pub audio: usize,
    /// Speaker embedding dimension
    pub speaker: usize,
    /// Content embedding dimension
    pub content: usize,
}
