//! Feature learning for voirs-dataset
//!
//! This module provides learned audio representations, speaker embedding extraction,
//! content embeddings, and quality prediction models using machine learning.
//!
//! The implementation has been refactored into multiple submodules for better organization:
//! - `config`: Configuration types and enums
//! - `core`: Core traits and data structures
//! - `audio_features`: Audio feature extraction
//! - `speaker_embeddings`: Speaker embedding extraction
//! - `content_embeddings`: Content embedding extraction
//! - `quality_prediction`: Quality prediction models
//! - `learner`: Main feature learner implementation

// Re-exports use the types from submodules where they are actually used

// Include the modular implementation
pub mod audio_features;
pub mod config;
pub mod content_embeddings;
pub mod core;
pub mod learner;
pub mod quality_prediction;
pub mod speaker_embeddings;

// Re-export main types for backward compatibility
pub use config::*;
pub use core::{FeatureDimensions, FeatureLearner, LearnedFeatures};
pub use learner::FeatureLearnerImpl;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, DatasetSample, LanguageCode};

    #[test]
    fn test_feature_config_default() {
        let config = FeatureConfig::default();
        assert_eq!(config.audio_features.dimension, 13);
        assert_eq!(config.speaker_embeddings.dimension, 512);
        assert_eq!(config.content_embeddings.text_dimension, 768);
    }

    #[tokio::test]
    async fn test_feature_learner_creation() {
        let config = FeatureConfig::default();
        let learner = FeatureLearnerImpl::new(config);
        assert!(learner.is_ok());
    }

    #[tokio::test]
    async fn test_audio_feature_extraction() {
        let config = FeatureConfig::default();
        let learner = FeatureLearnerImpl::new(config).unwrap();

        let audio = AudioData::silence(1.0, 22050, 1);
        let features = learner.extract_audio_features(&audio).await.unwrap();
        assert_eq!(features.len(), 13); // MFCC default
    }

    #[tokio::test]
    async fn test_speaker_embedding_extraction() {
        let config = FeatureConfig::default();
        let learner = FeatureLearnerImpl::new(config).unwrap();

        let audio = AudioData::silence(2.0, 22050, 1); // 2 seconds to meet minimum
        let embedding = learner.extract_speaker_embedding(&audio).await.unwrap();
        assert_eq!(embedding.len(), 512); // Default dimension
    }

    #[tokio::test]
    async fn test_quality_prediction() {
        let config = FeatureConfig::default();
        let learner = FeatureLearnerImpl::new(config).unwrap();

        let audio = AudioData::silence(1.0, 22050, 1);
        let sample = DatasetSample::new(
            "test".to_string(),
            "Hello world".to_string(),
            audio,
            LanguageCode::EnUs,
        );

        let quality = learner.predict_quality(&sample).await.unwrap();
        assert!(quality.snr.is_some());
        assert!(quality.overall_quality.is_some());
    }

    #[tokio::test]
    async fn test_all_features_extraction() {
        let config = FeatureConfig::default();
        let learner = FeatureLearnerImpl::new(config).unwrap();

        let audio = AudioData::silence(2.0, 22050, 1);
        let sample = DatasetSample::new(
            "test".to_string(),
            "Hello world".to_string(),
            audio,
            LanguageCode::EnUs,
        );

        let features = learner.extract_all_features(&sample).await.unwrap();
        assert!(features.audio_features.is_some());
        assert!(features.speaker_embedding.is_some());
        assert!(features.content_embedding.is_some());
        assert!(features.predicted_quality.is_some());
    }

    #[test]
    fn test_feature_dimensions() {
        let config = FeatureConfig::default();
        let learner = FeatureLearnerImpl::new(config).unwrap();

        let dims = learner.get_feature_dimensions();
        assert_eq!(dims.audio, 13);
        assert_eq!(dims.speaker, 512);
        assert_eq!(dims.content, 768);
    }
}
