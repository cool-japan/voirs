//! # Adaptive Learning System
//!
//! This module implements a comprehensive adaptive learning system that enables continuous
//! improvement of singing synthesis quality through user feedback and preference learning.
//!
//! ## Features
//!
//! - **User Feedback Learning**: Collect and learn from user ratings and corrections
//! - **Preference Learning**: Adapt to individual user preferences over time
//! - **Quality Metric Fine-tuning**: Automatically adjust quality metrics based on feedback
//! - **Style Adaptation**: Learn and adapt to user-preferred singing styles
//! - **Automatic Model Improvement**: Continuously improve synthesis models
//! - **Personalized Voice Characteristics**: Develop personalized voice profiles
//!
//! ## Example
//!
//! ```rust,ignore
//! use voirs_singing::adaptive_learning::*;
//!
//! // Create adaptive learning system
//! let config = AdaptiveLearningConfig::default();
//! let mut system = AdaptiveLearningSystem::new(config);
//!
//! // Collect user feedback
//! let feedback = UserFeedback::new(
//!     "user123",
//!     sample_audio,
//!     4.5,  // rating out of 5
//!     Some("Great vibrato, but pitch could be more accurate"),
//! );
//! system.add_feedback(feedback).await?;
//!
//! // Get personalized recommendations
//! let recommendations = system.get_recommendations("user123").await?;
//! ```

use crate::synthesis::SynthesisResult;
use crate::types::{Expression, VoiceCharacteristics};
use crate::{Error, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Configuration for adaptive learning system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearningConfig {
    /// Learning rate for preference updates
    pub learning_rate: f32,
    /// Minimum samples before adaptation
    pub min_samples: usize,
    /// Weight decay for old samples
    pub decay_factor: f32,
    /// Enable automatic model fine-tuning
    pub auto_finetune: bool,
    /// Confidence threshold for adaptation
    pub confidence_threshold: f32,
}

impl Default for AdaptiveLearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            min_samples: 10,
            decay_factor: 0.95,
            auto_finetune: true,
            confidence_threshold: 0.7,
        }
    }
}

/// User feedback for synthesis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedback {
    /// Unique feedback ID
    pub id: String,
    /// User identifier
    pub user_id: String,
    /// Synthesis sample ID
    pub sample_id: String,
    /// Audio data for reference
    #[serde(skip)]
    pub audio_data: Vec<f32>,
    /// Overall rating (0.0-5.0)
    pub rating: f32,
    /// Specific quality ratings
    pub quality_ratings: QualityRatings,
    /// User comments and corrections
    pub comments: Option<String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl UserFeedback {
    /// Create new user feedback
    pub fn new(
        user_id: impl Into<String>,
        audio_data: Vec<f32>,
        rating: f32,
        comments: Option<impl Into<String>>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            user_id: user_id.into(),
            sample_id: Uuid::new_v4().to_string(),
            audio_data,
            rating: rating.clamp(0.0, 5.0),
            quality_ratings: QualityRatings::default(),
            comments: comments.map(|c| c.into()),
            timestamp: Utc::now(),
        }
    }

    /// Set quality ratings
    pub fn with_quality_ratings(mut self, ratings: QualityRatings) -> Self {
        self.quality_ratings = ratings;
        self
    }
}

/// Specific quality ratings for different aspects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRatings {
    /// Pitch accuracy (0.0-5.0)
    pub pitch_accuracy: f32,
    /// Timing precision (0.0-5.0)
    pub timing_precision: f32,
    /// Naturalness (0.0-5.0)
    pub naturalness: f32,
    /// Expression quality (0.0-5.0)
    pub expression: f32,
    /// Voice quality (0.0-5.0)
    pub voice_quality: f32,
}

impl Default for QualityRatings {
    fn default() -> Self {
        Self {
            pitch_accuracy: 3.0,
            timing_precision: 3.0,
            naturalness: 3.0,
            expression: 3.0,
            voice_quality: 3.0,
        }
    }
}

/// User preference profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// User identifier
    pub user_id: String,
    /// Preferred voice characteristics
    pub voice_preferences: VoiceCharacteristics,
    /// Preferred expression style
    pub expression_preferences: HashMap<String, f32>,
    /// Quality metric weights
    pub quality_weights: QualityWeights,
    /// Number of feedback samples
    pub sample_count: usize,
    /// Confidence in preferences (0.0-1.0)
    pub confidence: f32,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

/// Weights for different quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityWeights {
    /// Weight for pitch accuracy (0.0-1.0)
    pub pitch_weight: f32,
    /// Weight for timing precision (0.0-1.0)
    pub timing_weight: f32,
    /// Weight for naturalness quality (0.0-1.0)
    pub naturalness_weight: f32,
    /// Weight for expression quality (0.0-1.0)
    pub expression_weight: f32,
    /// Weight for voice quality (0.0-1.0)
    pub voice_quality_weight: f32,
}

impl Default for QualityWeights {
    fn default() -> Self {
        Self {
            pitch_weight: 1.0,
            timing_weight: 1.0,
            naturalness_weight: 1.0,
            expression_weight: 1.0,
            voice_quality_weight: 1.0,
        }
    }
}

/// Style adaptation parameters learned from examples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleAdaptation {
    /// Style identifier
    pub style_id: String,
    /// Vibrato parameters
    pub vibrato_params: VibratoParams,
    /// Dynamics preferences
    pub dynamics_params: DynamicsParams,
    /// Articulation preferences
    pub articulation_params: ArticulationParams,
    /// Number of examples used
    pub example_count: usize,
    /// Adaptation confidence (0.0-1.0)
    pub confidence: f32,
}

/// Vibrato parameters for style adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibratoParams {
    /// Vibrato rate in Hz
    pub rate: f32,
    /// Vibrato depth (modulation amount)
    pub depth: f32,
    /// Onset delay in seconds
    pub onset_delay: f32,
}

/// Dynamics parameters for style adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsParams {
    /// Average volume level
    pub average_level: f32,
    /// Dynamic range (loudness variation)
    pub dynamic_range: f32,
    /// Rate of crescendo/decrescendo
    pub crescendo_rate: f32,
}

/// Articulation parameters for style adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticulationParams {
    /// Amount of legato (smooth connection between notes)
    pub legato_amount: f32,
    /// Amount of staccato (sharp, detached notes)
    pub staccato_amount: f32,
    /// Strength of note accents
    pub accent_strength: f32,
}

/// Personalized recommendations for synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizedRecommendations {
    /// Recommended voice characteristics
    pub voice_characteristics: VoiceCharacteristics,
    /// Recommended parameter adjustments
    pub parameter_adjustments: HashMap<String, f32>,
    /// Recommended techniques
    pub techniques: Vec<String>,
    /// Confidence in recommendations (0.0-1.0)
    pub confidence: f32,
}

/// Model improvement metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelImprovement {
    /// Improvement iteration
    pub iteration: usize,
    /// Quality improvement over baseline
    pub quality_delta: f32,
    /// User satisfaction improvement
    pub satisfaction_delta: f32,
    /// Number of training samples used
    pub training_samples: usize,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Main adaptive learning system
pub struct AdaptiveLearningSystem {
    config: AdaptiveLearningConfig,
    user_preferences: Arc<RwLock<HashMap<String, UserPreferences>>>,
    style_adaptations: Arc<RwLock<HashMap<String, StyleAdaptation>>>,
    feedback_history: Arc<RwLock<Vec<UserFeedback>>>,
    improvement_history: Arc<RwLock<Vec<ModelImprovement>>>,
}

impl AdaptiveLearningSystem {
    /// Create new adaptive learning system
    pub fn new(config: AdaptiveLearningConfig) -> Self {
        Self {
            config,
            user_preferences: Arc::new(RwLock::new(HashMap::new())),
            style_adaptations: Arc::new(RwLock::new(HashMap::new())),
            feedback_history: Arc::new(RwLock::new(Vec::new())),
            improvement_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add user feedback and update learning models
    pub async fn add_feedback(&self, feedback: UserFeedback) -> Result<()> {
        let user_id = feedback.user_id.clone();

        // Store feedback
        self.feedback_history.write().await.push(feedback.clone());

        // Update user preferences
        self.update_user_preferences(&user_id, &feedback).await?;

        // Update style adaptations if applicable
        self.update_style_adaptations(&feedback).await?;

        // Trigger automatic fine-tuning if enabled
        if self.config.auto_finetune {
            self.check_and_finetune(&user_id).await?;
        }

        Ok(())
    }

    /// Get personalized recommendations for a user
    pub async fn get_recommendations(&self, user_id: &str) -> Result<PersonalizedRecommendations> {
        let prefs = self.user_preferences.read().await;

        let user_pref = prefs.get(user_id).ok_or_else(|| {
            Error::Processing(format!("No preferences found for user: {}", user_id))
        })?;

        // Check if we have enough confidence
        if user_pref.confidence < self.config.confidence_threshold {
            return Err(Error::Processing(
                "Insufficient data for personalized recommendations".to_string(),
            ));
        }

        // Build recommendations
        let mut parameter_adjustments = HashMap::new();

        // Add voice characteristic adjustments
        parameter_adjustments.insert(
            "vibrato_frequency".to_string(),
            user_pref.voice_preferences.vibrato_frequency,
        );
        parameter_adjustments.insert(
            "vibrato_depth".to_string(),
            user_pref.voice_preferences.vibrato_depth,
        );

        // Add expression adjustments
        for (key, value) in &user_pref.expression_preferences {
            parameter_adjustments.insert(key.clone(), *value);
        }

        // Recommend techniques based on quality weights
        let mut techniques = Vec::new();
        if user_pref.quality_weights.expression_weight > 1.2 {
            techniques.push("enhanced_expression".to_string());
        }
        if user_pref.quality_weights.naturalness_weight > 1.2 {
            techniques.push("natural_breath_patterns".to_string());
        }

        Ok(PersonalizedRecommendations {
            voice_characteristics: user_pref.voice_preferences.clone(),
            parameter_adjustments,
            techniques,
            confidence: user_pref.confidence,
        })
    }

    /// Get user preferences
    pub async fn get_user_preferences(&self, user_id: &str) -> Option<UserPreferences> {
        self.user_preferences.read().await.get(user_id).cloned()
    }

    /// Get style adaptation
    pub async fn get_style_adaptation(&self, style_id: &str) -> Option<StyleAdaptation> {
        self.style_adaptations.read().await.get(style_id).cloned()
    }

    /// Get model improvement history
    pub async fn get_improvement_history(&self) -> Vec<ModelImprovement> {
        self.improvement_history.read().await.clone()
    }

    /// Update user preferences based on feedback
    async fn update_user_preferences(&self, user_id: &str, feedback: &UserFeedback) -> Result<()> {
        let mut prefs = self.user_preferences.write().await;

        let user_pref = prefs
            .entry(user_id.to_string())
            .or_insert_with(|| UserPreferences {
                user_id: user_id.to_string(),
                voice_preferences: VoiceCharacteristics::default(),
                expression_preferences: HashMap::new(),
                quality_weights: QualityWeights::default(),
                sample_count: 0,
                confidence: 0.0,
                last_updated: Utc::now(),
            });

        // Update sample count
        user_pref.sample_count += 1;

        // Update quality weights based on ratings
        let lr = self.config.learning_rate;
        user_pref.quality_weights.pitch_weight +=
            lr * (feedback.quality_ratings.pitch_accuracy - 3.0);
        user_pref.quality_weights.timing_weight +=
            lr * (feedback.quality_ratings.timing_precision - 3.0);
        user_pref.quality_weights.naturalness_weight +=
            lr * (feedback.quality_ratings.naturalness - 3.0);
        user_pref.quality_weights.expression_weight +=
            lr * (feedback.quality_ratings.expression - 3.0);
        user_pref.quality_weights.voice_quality_weight +=
            lr * (feedback.quality_ratings.voice_quality - 3.0);

        // Normalize weights
        let weight_sum = user_pref.quality_weights.pitch_weight
            + user_pref.quality_weights.timing_weight
            + user_pref.quality_weights.naturalness_weight
            + user_pref.quality_weights.expression_weight
            + user_pref.quality_weights.voice_quality_weight;

        if weight_sum > 0.0 {
            user_pref.quality_weights.pitch_weight /= weight_sum / 5.0;
            user_pref.quality_weights.timing_weight /= weight_sum / 5.0;
            user_pref.quality_weights.naturalness_weight /= weight_sum / 5.0;
            user_pref.quality_weights.expression_weight /= weight_sum / 5.0;
            user_pref.quality_weights.voice_quality_weight /= weight_sum / 5.0;
        }

        // Update confidence based on sample count
        user_pref.confidence =
            (user_pref.sample_count as f32 / self.config.min_samples as f32).min(1.0);

        user_pref.last_updated = Utc::now();

        Ok(())
    }

    /// Update style adaptations based on feedback
    async fn update_style_adaptations(&self, feedback: &UserFeedback) -> Result<()> {
        // Extract style features from feedback
        let style_id = format!("user_{}_style", feedback.user_id);

        let mut adaptations = self.style_adaptations.write().await;

        let adaptation = adaptations
            .entry(style_id.clone())
            .or_insert_with(|| StyleAdaptation {
                style_id,
                vibrato_params: VibratoParams {
                    rate: 5.0,
                    depth: 0.5,
                    onset_delay: 0.1,
                },
                dynamics_params: DynamicsParams {
                    average_level: 0.7,
                    dynamic_range: 0.4,
                    crescendo_rate: 0.1,
                },
                articulation_params: ArticulationParams {
                    legato_amount: 0.7,
                    staccato_amount: 0.3,
                    accent_strength: 0.5,
                },
                example_count: 0,
                confidence: 0.0,
            });

        // Update adaptation parameters based on feedback rating
        let lr = self.config.learning_rate;
        if feedback.rating > 4.0 {
            // Positive feedback - strengthen current parameters
            adaptation.example_count += 1;
        } else if feedback.rating < 3.0 {
            // Negative feedback - adjust parameters
            adaptation.vibrato_params.rate *= 1.0 - lr;
            adaptation.vibrato_params.depth *= 1.0 - lr;
        }

        // Update confidence
        adaptation.confidence =
            (adaptation.example_count as f32 / self.config.min_samples as f32).min(1.0);

        Ok(())
    }

    /// Check if fine-tuning is needed and trigger it
    async fn check_and_finetune(&self, user_id: &str) -> Result<()> {
        let prefs = self.user_preferences.read().await;

        if let Some(user_pref) = prefs.get(user_id) {
            if user_pref.sample_count >= self.config.min_samples
                && user_pref.confidence >= self.config.confidence_threshold
            {
                // Simulate model fine-tuning
                drop(prefs); // Release read lock
                self.perform_finetuning(user_id).await?;
            }
        }

        Ok(())
    }

    /// Perform model fine-tuning
    async fn perform_finetuning(&self, user_id: &str) -> Result<()> {
        // Simulate fine-tuning process
        let feedback = self.feedback_history.read().await;
        let user_feedback: Vec<_> = feedback.iter().filter(|f| f.user_id == user_id).collect();

        if user_feedback.is_empty() {
            return Ok(());
        }

        // Calculate improvement metrics
        let avg_rating: f32 =
            user_feedback.iter().map(|f| f.rating).sum::<f32>() / user_feedback.len() as f32;
        let baseline_rating = 3.0;
        let quality_delta = avg_rating - baseline_rating;

        let improvement = ModelImprovement {
            iteration: self.improvement_history.read().await.len(),
            quality_delta,
            satisfaction_delta: quality_delta * 0.2, // Simplified calculation
            training_samples: user_feedback.len(),
            timestamp: Utc::now(),
        };

        self.improvement_history.write().await.push(improvement);

        Ok(())
    }

    /// Get learning statistics
    pub async fn get_statistics(&self) -> LearningStatistics {
        let prefs = self.user_preferences.read().await;
        let feedback = self.feedback_history.read().await;
        let improvements = self.improvement_history.read().await;

        let avg_rating = if !feedback.is_empty() {
            feedback.iter().map(|f| f.rating).sum::<f32>() / feedback.len() as f32
        } else {
            0.0
        };

        let total_improvement = improvements.iter().map(|i| i.quality_delta).sum::<f32>();

        LearningStatistics {
            total_users: prefs.len(),
            total_feedback: feedback.len(),
            average_rating: avg_rating,
            total_improvements: improvements.len(),
            cumulative_improvement: total_improvement,
        }
    }
}

/// Learning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStatistics {
    /// Total number of users providing feedback
    pub total_users: usize,
    /// Total number of feedback samples collected
    pub total_feedback: usize,
    /// Average user rating (1.0-5.0)
    pub average_rating: f32,
    /// Total number of model improvements performed
    pub total_improvements: usize,
    /// Cumulative quality improvement
    pub cumulative_improvement: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_feedback_collection() {
        let config = AdaptiveLearningConfig::default();
        let system = AdaptiveLearningSystem::new(config);

        let feedback = UserFeedback::new(
            "user1",
            vec![0.0; 1000],
            4.5,
            Some("Great quality".to_string()),
        );

        system.add_feedback(feedback).await.unwrap();

        let stats = system.get_statistics().await;
        assert_eq!(stats.total_feedback, 1);
        assert_eq!(stats.total_users, 1);
    }

    #[tokio::test]
    async fn test_preference_learning() {
        let config = AdaptiveLearningConfig {
            min_samples: 2,
            ..Default::default()
        };
        let system = AdaptiveLearningSystem::new(config);

        // Add multiple feedback samples
        for i in 0..3 {
            let mut feedback = UserFeedback::new(
                "user1",
                vec![0.0; 1000],
                4.0 + i as f32 * 0.2,
                None::<String>,
            );
            feedback.quality_ratings.pitch_accuracy = 4.5;
            feedback.quality_ratings.naturalness = 4.0;
            system.add_feedback(feedback).await.unwrap();
        }

        let prefs = system.get_user_preferences("user1").await;
        assert!(prefs.is_some());

        let prefs = prefs.unwrap();
        assert!(prefs.confidence > 0.5);
        assert_eq!(prefs.sample_count, 3);
    }

    #[tokio::test]
    async fn test_personalized_recommendations() {
        let config = AdaptiveLearningConfig {
            min_samples: 2,
            confidence_threshold: 0.5,
            ..Default::default()
        };
        let system = AdaptiveLearningSystem::new(config);

        // Add sufficient feedback
        for _ in 0..3 {
            let feedback = UserFeedback::new("user1", vec![0.0; 1000], 4.5, None::<String>);
            system.add_feedback(feedback).await.unwrap();
        }

        let recommendations = system.get_recommendations("user1").await;
        assert!(recommendations.is_ok());

        let recs = recommendations.unwrap();
        assert!(recs.confidence >= 0.5);
        assert!(!recs.parameter_adjustments.is_empty());
    }

    #[tokio::test]
    async fn test_style_adaptation() {
        let config = AdaptiveLearningConfig::default();
        let system = AdaptiveLearningSystem::new(config);

        let feedback = UserFeedback::new(
            "user1",
            vec![0.0; 1000],
            4.8,
            Some("Excellent style".to_string()),
        );

        system.add_feedback(feedback).await.unwrap();

        let style_id = "user_user1_style";
        let adaptation = system.get_style_adaptation(style_id).await;
        assert!(adaptation.is_some());
    }

    #[tokio::test]
    async fn test_quality_weight_updates() {
        let config = AdaptiveLearningConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let system = AdaptiveLearningSystem::new(config);

        let mut feedback = UserFeedback::new("user1", vec![0.0; 1000], 4.0, None::<String>);
        feedback.quality_ratings.pitch_accuracy = 5.0;
        feedback.quality_ratings.timing_precision = 2.0;

        system.add_feedback(feedback).await.unwrap();

        let prefs = system.get_user_preferences("user1").await.unwrap();

        // Pitch weight should increase, timing weight should decrease
        assert!(prefs.quality_weights.pitch_weight > 1.0);
        assert!(prefs.quality_weights.timing_weight < 1.0);
    }

    #[tokio::test]
    async fn test_model_improvement_tracking() {
        let config = AdaptiveLearningConfig {
            min_samples: 2,
            auto_finetune: true,
            confidence_threshold: 0.5,
            ..Default::default()
        };
        let system = AdaptiveLearningSystem::new(config);

        // Add feedback to trigger fine-tuning
        for _ in 0..3 {
            let feedback = UserFeedback::new("user1", vec![0.0; 1000], 4.5, None::<String>);
            system.add_feedback(feedback).await.unwrap();
        }

        let history = system.get_improvement_history().await;
        assert!(!history.is_empty());
    }

    #[tokio::test]
    async fn test_confidence_calculation() {
        let config = AdaptiveLearningConfig {
            min_samples: 10,
            ..Default::default()
        };
        let system = AdaptiveLearningSystem::new(config);

        // Add 5 feedback samples (50% of min_samples)
        for _ in 0..5 {
            let feedback = UserFeedback::new("user1", vec![0.0; 1000], 4.0, None::<String>);
            system.add_feedback(feedback).await.unwrap();
        }

        let prefs = system.get_user_preferences("user1").await.unwrap();
        assert!((prefs.confidence - 0.5).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_learning_statistics() {
        let config = AdaptiveLearningConfig::default();
        let system = AdaptiveLearningSystem::new(config);

        // Add feedback from multiple users
        for user_id in &["user1", "user2", "user3"] {
            for _ in 0..2 {
                let feedback = UserFeedback::new(*user_id, vec![0.0; 1000], 4.0, None::<String>);
                system.add_feedback(feedback).await.unwrap();
            }
        }

        let stats = system.get_statistics().await;
        assert_eq!(stats.total_users, 3);
        assert_eq!(stats.total_feedback, 6);
        assert!((stats.average_rating - 4.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_insufficient_data_error() {
        let config = AdaptiveLearningConfig {
            min_samples: 10,
            confidence_threshold: 0.8,
            ..Default::default()
        };
        let system = AdaptiveLearningSystem::new(config);

        // Add insufficient feedback
        let feedback = UserFeedback::new("user1", vec![0.0; 1000], 4.0, None::<String>);
        system.add_feedback(feedback).await.unwrap();

        // Should fail due to insufficient data
        let result = system.get_recommendations("user1").await;
        assert!(result.is_err());
    }
}
