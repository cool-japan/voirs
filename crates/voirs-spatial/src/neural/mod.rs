//! # Neural Spatial Audio System
//!
//! This module provides end-to-end neural spatial audio synthesis using deep learning
//! models for real-time spatial audio generation and processing.

pub mod features;
pub mod models;
pub mod processor;
pub mod quality;
pub mod training;
pub mod types;

// Re-export public types and traits
pub use models::{
    ConvolutionalModel, FeedForwardLayer, FeedforwardModel, LayerNorm, MultiHeadAttention,
    NeuralModel, TransformerDecoder, TransformerEncoder, TransformerModel,
};
pub use processor::NeuralSpatialProcessor;
pub use quality::AdaptiveQualityController;
pub use training::NeuralTrainer;
pub use types::{
    AugmentationConfig, LossFunction, NeuralInputFeatures, NeuralModelType,
    NeuralPerformanceMetrics, NeuralSpatialConfig, NeuralSpatialConfigBuilder, NeuralSpatialOutput,
    NeuralTrainingResults, OptimizerType, RealtimeConstraints, TrainingConfig,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Position3D;
    use candle_core::Device;
    use std::collections::HashMap;

    #[test]
    fn test_neural_config_creation() {
        let config = NeuralSpatialConfig::default();
        assert_eq!(config.model_type, NeuralModelType::Feedforward);
        assert_eq!(config.output_channels, 2);
        assert_eq!(config.sample_rate, 48000);
    }

    #[test]
    fn test_neural_config_builder() {
        let config = NeuralSpatialConfigBuilder::new()
            .model_type(NeuralModelType::Transformer)
            .hidden_dims(vec![256, 128])
            .input_dim(64)
            .sample_rate(44100)
            .quality(0.9)
            .build();

        assert_eq!(config.model_type, NeuralModelType::Transformer);
        assert_eq!(config.hidden_dims, vec![256, 128]);
        assert_eq!(config.input_dim, 64);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.quality, 0.9);
    }

    #[test]
    fn test_neural_processor_creation() {
        let config = NeuralSpatialConfig::default();
        let processor = NeuralSpatialProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_adaptive_quality_controller() {
        let mut controller = AdaptiveQualityController::new(20.0);
        assert_eq!(controller.get_quality(), 0.8);

        // Simulate high latency - should reduce quality
        for _ in 0..5 {
            controller.update(30.0);
        }
        let quality_after_high_latency = controller.get_quality();
        assert!(quality_after_high_latency < 0.8);

        // Simulate low latency - should increase quality
        for _ in 0..15 {
            controller.update(10.0);
        }
        let quality_after_low_latency = controller.get_quality();
        // Quality should improve from the degraded state, but may not go above 0.5 due to conservative adaptation
        assert!(quality_after_low_latency > quality_after_high_latency);
    }

    #[test]
    fn test_feedforward_model_creation() {
        let config = NeuralSpatialConfig::default();
        let device = Device::Cpu;
        let model = FeedforwardModel::new(config, device);
        assert!(model.is_ok());
    }

    #[test]
    fn test_neural_input_features() {
        let features = NeuralInputFeatures {
            position: Position3D::new(1.0, 2.0, 3.0),
            listener_orientation: [1.0, 0.0, 0.0, 0.0],
            audio_features: vec![0.5; 64],
            room_features: vec![0.3; 32],
            hrtf_features: Some(vec![0.7; 16]),
            temporal_context: vec![0.1; 8],
            user_features: Some(vec![0.9; 4]),
        };

        assert_eq!(features.position.x, 1.0);
        assert_eq!(features.audio_features.len(), 64);
        assert!(features.hrtf_features.is_some());
    }

    #[test]
    fn test_neural_spatial_output() {
        let output = NeuralSpatialOutput {
            binaural_audio: vec![vec![0.1; 1024], vec![0.2; 1024]],
            confidence: 0.9,
            latency_ms: 15.0,
            quality_score: 0.8,
            metadata: HashMap::new(),
        };

        assert_eq!(output.binaural_audio.len(), 2);
        assert_eq!(output.binaural_audio[0].len(), 1024);
        assert_eq!(output.confidence, 0.9);
        assert_eq!(output.latency_ms, 15.0);
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = NeuralPerformanceMetrics::default();
        metrics.frames_processed = 100;
        metrics.avg_processing_time_ms = 12.5;
        metrics.peak_processing_time_ms = 25.0;

        assert_eq!(metrics.frames_processed, 100);
        assert_eq!(metrics.avg_processing_time_ms, 12.5);
        assert_eq!(metrics.peak_processing_time_ms, 25.0);
    }
}
