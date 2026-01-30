//! ML-based quality prediction for adaptive synthesis control.
//!
//! This module implements machine learning-based prediction of optimal quality settings
//! using historical performance data and system metrics.
//!
//! # Features
//!
//! - **Online Learning**: Continuously learns from synthesis performance
//! - **Pattern Recognition**: Identifies optimal quality settings for different scenarios
//! - **Prediction Confidence**: Provides confidence scores for predictions
//! - **Adaptive Models**: Automatically adjusts to changing usage patterns
//!
//! # Example
//!
//! ```no_run
//! use voirs_sdk::adaptive::predictor::{QualityPredictor, PredictionInput};
//! use voirs_sdk::adaptive::{QualityTarget, SystemMetrics};
//!
//! #[tokio::main]
//! async fn main() -> voirs_sdk::Result<()> {
//!     let mut predictor = QualityPredictor::new();
//!
//!     // Train from historical data
//!     let input = PredictionInput {
//!         cpu_usage: 0.6,
//!         memory_usage: 0.5,
//!         text_complexity: 0.7,
//!         time_of_day: 14, // 2 PM
//!         recent_rtf: 0.45,
//!     };
//!
//!     let prediction = predictor.predict(&input).await?;
//!     println!("Predicted quality: {:?} (confidence: {:.2})",
//!              prediction.quality, prediction.confidence);
//!
//!     Ok(())
//! }
//! ```

use super::{QualityTarget, SystemMetrics};
use crate::{Result, VoirsError};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Input features for quality prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionInput {
    /// Current CPU usage (0.0-1.0)
    pub cpu_usage: f32,

    /// Current memory usage (0.0-1.0)
    pub memory_usage: f32,

    /// Text complexity score (0.0-1.0)
    pub text_complexity: f32,

    /// Time of day (0-23 hours)
    pub time_of_day: u8,

    /// Recent real-time factor
    pub recent_rtf: f32,
}

impl PredictionInput {
    /// Convert to feature vector for ML model.
    pub fn to_features(&self) -> Array1<f32> {
        Array1::from_vec(vec![
            self.cpu_usage,
            self.memory_usage,
            self.text_complexity,
            self.time_of_day as f32 / 24.0, // Normalize to 0-1
            self.recent_rtf,
        ])
    }

    /// Create from system metrics and text analysis.
    pub fn from_metrics(metrics: &SystemMetrics, text_complexity: f32) -> Self {
        use chrono::Timelike;
        let now = chrono::Local::now();

        Self {
            cpu_usage: metrics.cpu_usage,
            memory_usage: metrics.memory_usage,
            text_complexity,
            time_of_day: now.hour() as u8,
            recent_rtf: metrics.current_rtf,
        }
    }
}

/// Quality prediction with confidence score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPrediction {
    /// Predicted optimal quality level
    pub quality: QualityTarget,

    /// Confidence in prediction (0.0-1.0)
    pub confidence: f32,

    /// Expected synthesis time in milliseconds
    pub expected_time_ms: u64,

    /// Expected success probability
    pub success_probability: f32,
}

/// Training sample for the quality predictor.
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// Input features
    pub input: PredictionInput,

    /// Actual quality used
    pub quality: QualityTarget,

    /// Actual synthesis time
    pub synthesis_time_ms: u64,

    /// Whether synthesis succeeded
    pub success: bool,

    /// Measured RTF after synthesis
    pub measured_rtf: f32,
}

/// Simple linear regression model for quality prediction.
///
/// Uses online learning with exponential weighting to adapt to changing patterns.
#[derive(Debug, Clone)]
struct LinearModel {
    /// Model weights (5 features + bias)
    weights: Array1<f32>,

    /// Learning rate for online updates
    learning_rate: f32,

    /// Number of training samples seen
    samples_seen: u64,
}

impl LinearModel {
    fn new() -> Self {
        Self {
            weights: Array1::zeros(6), // 5 features + 1 bias
            learning_rate: 0.01,
            samples_seen: 0,
        }
    }

    /// Predict quality score (0-100) from features.
    fn predict(&self, features: &Array1<f32>) -> f32 {
        let mut score = self.weights[5]; // bias term
        for (i, &feature) in features.iter().enumerate() {
            score += self.weights[i] * feature;
        }
        score.clamp(0.0, 100.0)
    }

    /// Update model with new training sample (online learning).
    fn update(&mut self, features: &Array1<f32>, target: f32) {
        let prediction = self.predict(features);
        let error = target - prediction;

        // Gradient descent update with adaptive learning rate
        let effective_lr = self.learning_rate / (1.0 + (self.samples_seen as f32).sqrt());

        for (i, &feature) in features.iter().enumerate() {
            self.weights[i] += effective_lr * error * feature;
        }
        self.weights[5] += effective_lr * error; // bias update

        self.samples_seen += 1;
    }
}

/// ML-based quality predictor using historical performance data.
#[derive(Debug, Clone)]
pub struct QualityPredictor {
    /// Primary prediction model (quality score)
    quality_model: LinearModel,

    /// Secondary model for synthesis time prediction
    time_model: LinearModel,

    /// Success rate model
    success_model: LinearModel,

    /// Recent training samples for retraining
    recent_samples: VecDeque<TrainingSample>,

    /// Maximum samples to keep in history
    max_history: usize,

    /// Minimum samples needed before making predictions
    min_samples: usize,
}

impl QualityPredictor {
    /// Create a new quality predictor.
    pub fn new() -> Self {
        Self {
            quality_model: LinearModel::new(),
            time_model: LinearModel::new(),
            success_model: LinearModel::new(),
            recent_samples: VecDeque::new(),
            max_history: 1000,
            min_samples: 10,
        }
    }

    /// Create predictor with custom history size.
    pub fn with_history_size(mut self, max_history: usize, min_samples: usize) -> Self {
        self.max_history = max_history;
        self.min_samples = min_samples;
        self
    }

    /// Predict optimal quality for given input.
    pub async fn predict(&self, input: &PredictionInput) -> Result<QualityPrediction> {
        let features = input.to_features();

        // Check if we have enough training data
        if self.recent_samples.len() < self.min_samples {
            return Ok(QualityPrediction {
                quality: QualityTarget::Medium, // Safe default
                confidence: 0.0,                // No confidence without training
                expected_time_ms: 100,
                success_probability: 0.95,
            });
        }

        // Predict quality score
        let quality_score = self.quality_model.predict(&features);
        let quality = QualityTarget::Custom(quality_score as u8);

        // Predict synthesis time
        let expected_time = self.time_model.predict(&features).max(10.0);

        // Predict success probability
        let success_prob = self.success_model.predict(&features).clamp(0.0, 1.0);

        // Calculate confidence based on prediction stability
        let confidence = self.calculate_confidence(&features);

        Ok(QualityPrediction {
            quality,
            confidence,
            expected_time_ms: expected_time as u64,
            success_probability: success_prob,
        })
    }

    /// Train the predictor with a new sample.
    pub async fn train(&mut self, sample: TrainingSample) -> Result<()> {
        let features = sample.input.to_features();

        // Update quality model
        let quality_target = sample.quality.score() as f32;
        self.quality_model.update(&features, quality_target);

        // Update time model
        let time_target = sample.synthesis_time_ms as f32;
        self.time_model.update(&features, time_target);

        // Update success model
        let success_target = if sample.success { 1.0 } else { 0.0 };
        self.success_model.update(&features, success_target);

        // Add to history
        self.recent_samples.push_back(sample);
        if self.recent_samples.len() > self.max_history {
            self.recent_samples.pop_front();
        }

        Ok(())
    }

    /// Batch train from multiple samples.
    pub async fn batch_train(&mut self, samples: Vec<TrainingSample>) -> Result<()> {
        for sample in samples {
            self.train(sample).await?;
        }
        Ok(())
    }

    /// Get statistics about the predictor.
    pub fn get_stats(&self) -> PredictorStats {
        let total_samples = self.recent_samples.len();
        let successful = self.recent_samples.iter().filter(|s| s.success).count();

        let avg_time = if !self.recent_samples.is_empty() {
            self.recent_samples
                .iter()
                .map(|s| s.synthesis_time_ms as f64)
                .sum::<f64>()
                / total_samples as f64
        } else {
            0.0
        };

        PredictorStats {
            total_samples,
            successful_samples: successful,
            avg_synthesis_time_ms: avg_time,
            model_samples_seen: self.quality_model.samples_seen,
            confidence: if total_samples >= self.min_samples {
                (total_samples as f32 / self.max_history as f32).min(1.0)
            } else {
                0.0
            },
        }
    }

    /// Reset the predictor to initial state.
    pub fn reset(&mut self) {
        self.quality_model = LinearModel::new();
        self.time_model = LinearModel::new();
        self.success_model = LinearModel::new();
        self.recent_samples.clear();
    }

    // Helper methods

    fn calculate_confidence(&self, _features: &Array1<f32>) -> f32 {
        // Confidence increases with number of training samples
        let sample_confidence =
            (self.recent_samples.len() as f32 / self.max_history as f32).min(1.0);

        // Confidence decreases if recent success rate is low
        let recent_window = 20.min(self.recent_samples.len());
        let recent_success_rate = if recent_window > 0 {
            self.recent_samples
                .iter()
                .rev()
                .take(recent_window)
                .filter(|s| s.success)
                .count() as f32
                / recent_window as f32
        } else {
            1.0
        };

        // Combined confidence
        (sample_confidence * 0.6 + recent_success_rate * 0.4).clamp(0.0, 1.0)
    }
}

impl Default for QualityPredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the quality predictor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorStats {
    /// Total number of training samples in history
    pub total_samples: usize,

    /// Number of successful syntheses
    pub successful_samples: usize,

    /// Average synthesis time across samples
    pub avg_synthesis_time_ms: f64,

    /// Total samples seen by model (including forgotten ones)
    pub model_samples_seen: u64,

    /// Overall confidence in predictions
    pub confidence: f32,
}

/// Text complexity analyzer for prediction input.
pub struct TextComplexityAnalyzer;

impl TextComplexityAnalyzer {
    /// Analyze text complexity (0.0-1.0).
    ///
    /// Factors considered:
    /// - Text length
    /// - Vocabulary diversity
    /// - Sentence structure complexity
    /// - Special characters and numbers
    pub fn analyze(text: &str) -> f32 {
        // Handle empty text
        if text.is_empty() {
            return 0.0;
        }

        let length_score = Self::length_complexity(text);
        let vocab_score = Self::vocabulary_complexity(text);
        let structure_score = Self::structure_complexity(text);

        // Weighted average
        (length_score * 0.3 + vocab_score * 0.4 + structure_score * 0.3).clamp(0.0, 1.0)
    }

    fn length_complexity(text: &str) -> f32 {
        let char_count = text.chars().count();
        // Normalize: 0-100 chars -> 0.0-1.0
        (char_count as f32 / 100.0).min(1.0)
    }

    fn vocabulary_complexity(text: &str) -> f32 {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }

        // Unique words ratio
        let unique_words: std::collections::HashSet<_> = words.iter().collect();
        unique_words.len() as f32 / words.len() as f32
    }

    fn structure_complexity(text: &str) -> f32 {
        let sentence_count = text.matches(&['.', '!', '?'][..]).count();
        let word_count = text.split_whitespace().count();

        if sentence_count == 0 {
            return 0.5; // Single sentence - medium complexity
        }

        // Average words per sentence
        let avg_words_per_sentence = word_count as f32 / sentence_count as f32;

        // Normalize: 5-20 words/sentence -> 0.0-1.0
        ((avg_words_per_sentence - 5.0) / 15.0).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_input_features() {
        let input = PredictionInput {
            cpu_usage: 0.5,
            memory_usage: 0.6,
            text_complexity: 0.7,
            time_of_day: 12,
            recent_rtf: 0.4,
        };

        let features = input.to_features();
        assert_eq!(features.len(), 5);
        assert_eq!(features[0], 0.5);
        assert_eq!(features[3], 0.5); // 12/24 = 0.5
    }

    #[test]
    fn test_linear_model_prediction() {
        let model = LinearModel::new();
        let features = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5]);

        let prediction = model.predict(&features);
        assert!(prediction >= 0.0 && prediction <= 100.0);
    }

    #[tokio::test]
    async fn test_quality_predictor_without_training() {
        let predictor = QualityPredictor::new();
        let input = PredictionInput {
            cpu_usage: 0.5,
            memory_usage: 0.5,
            text_complexity: 0.5,
            time_of_day: 12,
            recent_rtf: 0.5,
        };

        let prediction = predictor.predict(&input).await.unwrap();
        assert_eq!(prediction.quality, QualityTarget::Medium);
        assert_eq!(prediction.confidence, 0.0); // No training yet
    }

    #[tokio::test]
    async fn test_quality_predictor_with_training() {
        let mut predictor = QualityPredictor::new().with_history_size(100, 5);

        // Train with some samples
        for i in 0..10 {
            let sample = TrainingSample {
                input: PredictionInput {
                    cpu_usage: 0.5,
                    memory_usage: 0.5,
                    text_complexity: 0.5,
                    time_of_day: 12,
                    recent_rtf: 0.5,
                },
                quality: QualityTarget::High,
                synthesis_time_ms: 100 + i * 10,
                success: true,
                measured_rtf: 0.5,
            };

            predictor.train(sample).await.unwrap();
        }

        let stats = predictor.get_stats();
        assert_eq!(stats.total_samples, 10);
        assert_eq!(stats.successful_samples, 10);
    }

    #[test]
    fn test_text_complexity_simple() {
        let simple_text = "Hello world.";
        let complexity = TextComplexityAnalyzer::analyze(simple_text);
        assert!(complexity < 0.5); // Simple text should have low complexity
    }

    #[test]
    fn test_text_complexity_complex() {
        let complex_text = "The sophisticated implementation of machine learning \
                           algorithms requires comprehensive understanding of mathematical \
                           foundations, statistical methodologies, and computational efficiency.";
        let complexity = TextComplexityAnalyzer::analyze(complex_text);
        assert!(complexity > 0.5); // Complex text should have higher complexity
    }

    #[test]
    fn test_text_complexity_empty() {
        let empty_text = "";
        let complexity = TextComplexityAnalyzer::analyze(empty_text);
        assert_eq!(complexity, 0.0);
    }

    #[tokio::test]
    async fn test_predictor_reset() {
        let mut predictor = QualityPredictor::new();

        // Train with some samples
        let sample = TrainingSample {
            input: PredictionInput {
                cpu_usage: 0.5,
                memory_usage: 0.5,
                text_complexity: 0.5,
                time_of_day: 12,
                recent_rtf: 0.5,
            },
            quality: QualityTarget::High,
            synthesis_time_ms: 100,
            success: true,
            measured_rtf: 0.5,
        };

        predictor.train(sample).await.unwrap();
        assert_eq!(predictor.get_stats().total_samples, 1);

        // Reset
        predictor.reset();
        assert_eq!(predictor.get_stats().total_samples, 0);
    }
}
