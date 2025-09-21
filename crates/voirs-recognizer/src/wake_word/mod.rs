//! Wake word detection and keyword spotting functionality
//!
//! This module provides always-on listening capabilities, custom wake word training,
//! false positive reduction, and energy-efficient detection algorithms.

use crate::RecognitionError;
use async_trait::async_trait;
use std::time::{Duration, Instant};
use voirs_sdk::AudioBuffer;

pub mod detector;
pub mod energy_optimizer;
pub mod models;
pub mod training;

pub use detector::*;
pub use energy_optimizer::*;
pub use models::*;
pub use training::*;

/// Configuration for wake word detection
#[derive(Debug, Clone)]
pub struct WakeWordConfig {
    /// Sensitivity threshold (0.0 to 1.0)
    pub sensitivity: f32,
    /// Minimum confidence score for detection
    pub min_confidence: f32,
    /// Maximum number of false positives per hour
    pub max_false_positives_per_hour: u32,
    /// Energy saving mode enabled
    pub energy_saving: bool,
    /// Wake words to detect
    pub wake_words: Vec<String>,
    /// Custom model path (optional)
    pub custom_model_path: Option<String>,
    /// Detection window size in milliseconds
    pub detection_window_ms: u32,
    /// Overlap between detection windows
    pub overlap_ratio: f32,
}

impl Default for WakeWordConfig {
    fn default() -> Self {
        Self {
            sensitivity: 0.7,
            min_confidence: 0.8,
            max_false_positives_per_hour: 5,
            energy_saving: true,
            wake_words: vec!["hey", "wake", "listen"]
                .into_iter()
                .map(String::from)
                .collect(),
            custom_model_path: None,
            detection_window_ms: 1000,
            overlap_ratio: 0.5,
        }
    }
}

/// Result of wake word detection
#[derive(Debug, Clone)]
pub struct WakeWordDetection {
    /// Detected wake word
    pub word: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Detection timestamp
    pub timestamp: Instant,
    /// Audio segment start time (relative to stream start)
    pub start_time: Duration,
    /// Audio segment end time (relative to stream start)
    pub end_time: Duration,
    /// False positive probability (0.0 to 1.0)
    pub false_positive_prob: f32,
}

/// Wake word detection statistics
#[derive(Debug, Clone)]
pub struct WakeWordStats {
    /// Total detections
    pub total_detections: u64,
    /// False positives detected
    pub false_positives: u64,
    /// True positives detected
    pub true_positives: u64,
    /// Average confidence score
    pub avg_confidence: f32,
    /// Processing time statistics
    pub avg_processing_time_ms: f32,
    /// Energy consumption estimate (relative units)
    pub energy_consumption: f32,
    /// Detection rate (detections per hour)
    pub detection_rate: f32,
}

impl Default for WakeWordStats {
    fn default() -> Self {
        Self {
            total_detections: 0,
            false_positives: 0,
            true_positives: 0,
            avg_confidence: 0.0,
            avg_processing_time_ms: 0.0,
            energy_consumption: 0.0,
            detection_rate: 0.0,
        }
    }
}

/// Wake word detection trait
#[async_trait]
pub trait WakeWordDetector: Send + Sync {
    /// Start continuous wake word detection
    async fn start_detection(&mut self) -> Result<(), RecognitionError>;

    /// Stop wake word detection
    async fn stop_detection(&mut self) -> Result<(), RecognitionError>;

    /// Process audio chunk and detect wake words
    async fn detect_wake_words(
        &mut self,
        audio: &AudioBuffer,
    ) -> Result<Vec<WakeWordDetection>, RecognitionError>;

    /// Add a new wake word to detection
    async fn add_wake_word(&mut self, word: &str) -> Result<(), RecognitionError>;

    /// Remove a wake word from detection
    async fn remove_wake_word(&mut self, word: &str) -> Result<(), RecognitionError>;

    /// Update detection configuration
    async fn update_config(&mut self, config: WakeWordConfig) -> Result<(), RecognitionError>;

    /// Get detection statistics
    async fn get_statistics(&self) -> Result<WakeWordStats, RecognitionError>;

    /// Check if detector is currently running
    fn is_running(&self) -> bool;

    /// Get supported wake words
    fn get_supported_words(&self) -> Vec<String>;
}

/// Training data for custom wake word models
#[derive(Debug, Clone)]
pub struct WakeWordTrainingData {
    /// Positive examples (containing wake word)
    pub positive_examples: Vec<AudioBuffer>,
    /// Negative examples (not containing wake word)
    pub negative_examples: Vec<AudioBuffer>,
    /// Wake word text
    pub wake_word: String,
    /// Speaker ID (for multi-speaker models)
    pub speaker_id: Option<String>,
}

/// Custom wake word trainer
#[async_trait]
pub trait WakeWordTrainer: Send + Sync {
    /// Train a custom wake word model
    async fn train_wake_word(
        &mut self,
        training_data: WakeWordTrainingData,
    ) -> Result<String, RecognitionError>; // Returns model path

    /// Validate training data quality
    async fn validate_training_data(
        &self,
        training_data: &WakeWordTrainingData,
    ) -> Result<TrainingValidationReport, RecognitionError>;

    /// Get training progress
    async fn get_training_progress(&self) -> Result<TrainingProgress, RecognitionError>;
}

/// Training validation report
#[derive(Debug, Clone)]
pub struct TrainingValidationReport {
    /// Data quality score (0.0 to 1.0)
    pub quality_score: f32,
    /// Number of positive examples
    pub positive_count: usize,
    /// Number of negative examples  
    pub negative_count: usize,
    /// Audio quality issues found
    pub quality_issues: Vec<String>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    /// Estimated accuracy after training
    pub estimated_accuracy: f32,
}

/// Training progress information
#[derive(Debug, Clone)]
pub struct TrainingProgress {
    /// Training phase
    pub phase: TrainingPhase,
    /// Progress percentage (0.0 to 1.0)
    pub progress: f32,
    /// Current loss value
    pub current_loss: Option<f32>,
    /// Estimated time remaining
    pub eta: Option<Duration>,
    /// Current learning rate
    pub learning_rate: Option<f32>,
}

/// Training phases
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingPhase {
    /// Data preprocessing
    Preprocessing,
    /// Feature extraction
    FeatureExtraction,
    /// Model training
    Training,
    /// Validation
    Validation,
    /// Model optimization
    Optimization,
    /// Completed
    Completed,
    /// Failed
    Failed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wake_word_config_default() {
        let config = WakeWordConfig::default();
        assert_eq!(config.sensitivity, 0.7);
        assert_eq!(config.min_confidence, 0.8);
        assert_eq!(config.max_false_positives_per_hour, 5);
        assert!(config.energy_saving);
        assert_eq!(config.wake_words.len(), 3);
        assert_eq!(config.detection_window_ms, 1000);
        assert_eq!(config.overlap_ratio, 0.5);
    }

    #[test]
    fn test_wake_word_stats_default() {
        let stats = WakeWordStats::default();
        assert_eq!(stats.total_detections, 0);
        assert_eq!(stats.false_positives, 0);
        assert_eq!(stats.true_positives, 0);
        assert_eq!(stats.avg_confidence, 0.0);
    }

    #[test]
    fn test_training_phase_equality() {
        assert_eq!(TrainingPhase::Preprocessing, TrainingPhase::Preprocessing);
        assert_ne!(TrainingPhase::Training, TrainingPhase::Validation);

        let failed1 = TrainingPhase::Failed("error1".to_string());
        let failed2 = TrainingPhase::Failed("error1".to_string());
        assert_eq!(failed1, failed2);
    }
}
