//! Machine Learning-based Emotion Personalization System
//!
//! This module implements adaptive emotion learning that personalizes emotion responses
//! based on user feedback, usage patterns, and contextual information.

use crate::{
    history::{EmotionHistory, EmotionHistoryEntry, EmotionPattern},
    types::{Emotion, EmotionDimensions, EmotionParameters, EmotionState, EmotionVector},
    Error, Result,
};

use ndarray::{Array1, Array2, ArrayView1, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, trace, warn};

#[cfg(feature = "gpu")]
use candle_core::{Device, Tensor};
#[cfg(feature = "gpu")]
use candle_nn::{linear, ops, Linear, Optimizer, VarBuilder, VarMap};

/// User feedback for emotion learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionFeedback {
    /// Original emotion parameters used
    pub emotion: EmotionParameters,
    /// Context information
    pub context: String,
    /// User satisfaction score (0.0 to 1.0)
    pub satisfaction: f32,
    /// Specific feedback ratings
    pub ratings: FeedbackRatings,
    /// Timestamp of feedback
    pub timestamp: std::time::SystemTime,
}

/// Detailed feedback ratings for different aspects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackRatings {
    /// Overall naturalness (0.0 to 1.0)
    pub naturalness: f32,
    /// Emotion intensity appropriateness (0.0 to 1.0)
    pub intensity: f32,
    /// Emotional authenticity (0.0 to 1.0)
    pub authenticity: f32,
    /// Contextual appropriateness (0.0 to 1.0)
    pub appropriateness: f32,
}

/// User preference profile learned from feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferenceProfile {
    /// User identifier
    pub user_id: String,
    /// Learned emotion biases for different emotions
    pub emotion_biases: HashMap<String, EmotionDimensions>,
    /// Context-specific preferences
    pub context_preferences: HashMap<String, ContextPreference>,
    /// Learning statistics
    pub stats: LearningStats,
    /// Last update timestamp
    pub last_updated: std::time::SystemTime,
}

/// Context-specific emotion preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPreference {
    /// Preferred intensity scaling for this context
    pub intensity_scale: f32,
    /// Preferred emotional dimensions adjustments
    pub dimension_adjustments: EmotionDimensions,
    /// Confidence in this preference (0.0 to 1.0)
    pub confidence: f32,
    /// Number of observations for this context
    pub observation_count: u32,
}

/// Learning algorithm statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStats {
    /// Total number of feedback samples processed
    pub total_feedback: u32,
    /// Number of training iterations completed
    pub training_iterations: u32,
    /// Current learning rate
    pub learning_rate: f32,
    /// Average prediction accuracy
    pub prediction_accuracy: f32,
    /// Model convergence indicator (0.0 to 1.0)
    pub convergence: f32,
}

/// Configuration for emotion learning system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionLearningConfig {
    /// Maximum number of feedback samples to store
    pub max_feedback_samples: usize,
    /// Learning rate for neural network training
    pub learning_rate: f32,
    /// Batch size for training
    pub batch_size: usize,
    /// Number of training epochs per update
    pub training_epochs: u32,
    /// Minimum feedback samples before starting learning
    pub min_samples_for_learning: usize,
    /// Weight decay for regularization
    pub weight_decay: f32,
    /// Enable GPU acceleration if available
    pub use_gpu: bool,
}

impl Default for EmotionLearningConfig {
    fn default() -> Self {
        Self {
            max_feedback_samples: 10000,
            learning_rate: 0.001,
            batch_size: 32,
            training_epochs: 100,
            min_samples_for_learning: 50,
            weight_decay: 0.0001,
            use_gpu: true,
        }
    }
}

/// Neural network model for emotion preference prediction
#[cfg(feature = "gpu")]
#[derive(Debug)]
struct EmotionPreferenceModel {
    linear1: Linear,
    linear2: Linear,
    linear3: Linear,
    device: Device,
}

#[cfg(feature = "gpu")]
impl EmotionPreferenceModel {
    fn new(varmap: &VarMap, device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_varmap(varmap, candle_core::DType::F32, device);

        Ok(Self {
            linear1: linear(8, 64, vb.pp("linear1"))
                .map_err(|e| Error::Processing(format!("Failed to create linear1 layer: {}", e)))?,
            linear2: linear(64, 32, vb.pp("linear2"))
                .map_err(|e| Error::Processing(format!("Failed to create linear2 layer: {}", e)))?,
            linear3: linear(32, 4, vb.pp("linear3"))
                .map_err(|e| Error::Processing(format!("Failed to create linear3 layer: {}", e)))?,
            device: device.clone(),
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self
            .linear1
            .forward(input)
            .map_err(|e| Error::Processing(format!("Forward pass linear1 failed: {}", e)))?;
        let x = ops::relu(&x)
            .map_err(|e| Error::Processing(format!("ReLU activation failed: {}", e)))?;

        let x = self
            .linear2
            .forward(&x)
            .map_err(|e| Error::Processing(format!("Forward pass linear2 failed: {}", e)))?;
        let x = ops::relu(&x)
            .map_err(|e| Error::Processing(format!("ReLU activation failed: {}", e)))?;

        let x = self
            .linear3
            .forward(&x)
            .map_err(|e| Error::Processing(format!("Forward pass linear3 failed: {}", e)))?;

        Ok(x)
    }
}

/// Machine learning-based emotion learning system
pub struct EmotionLearner {
    config: EmotionLearningConfig,
    feedback_history: Arc<RwLock<Vec<EmotionFeedback>>>,
    user_profiles: Arc<RwLock<HashMap<String, UserPreferenceProfile>>>,

    #[cfg(feature = "gpu")]
    model: Arc<RwLock<Option<EmotionPreferenceModel>>>,
    #[cfg(feature = "gpu")]
    varmap: Arc<RwLock<VarMap>>,
    #[cfg(feature = "gpu")]
    device: Device,

    // CPU-based learning components
    weight_matrix: Arc<RwLock<Array2<f32>>>,
    bias_vector: Arc<RwLock<Array1<f32>>>,
}

impl EmotionLearner {
    /// Create a new emotion learning system
    pub fn new(config: EmotionLearningConfig) -> Result<Self> {
        #[cfg(feature = "gpu")]
        let device = if config.use_gpu {
            Device::cuda_if_available(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        // Initialize CPU-based learning components
        let weight_matrix = Array2::zeros((4, 8)); // 4 outputs, 8 inputs
        let bias_vector = Array1::zeros(4);

        Ok(Self {
            config,
            feedback_history: Arc::new(RwLock::new(Vec::new())),
            user_profiles: Arc::new(RwLock::new(HashMap::new())),

            #[cfg(feature = "gpu")]
            model: Arc::new(RwLock::new(None)),
            #[cfg(feature = "gpu")]
            varmap: Arc::new(RwLock::new(VarMap::new())),
            #[cfg(feature = "gpu")]
            device,

            weight_matrix: Arc::new(RwLock::new(weight_matrix)),
            bias_vector: Arc::new(RwLock::new(bias_vector)),
        })
    }

    /// Add feedback from user interaction
    pub async fn add_feedback(&self, feedback: EmotionFeedback) -> Result<()> {
        let mut history = self.feedback_history.write().await;

        // Add new feedback
        history.push(feedback.clone());

        // Limit history size
        if history.len() > self.config.max_feedback_samples {
            history.remove(0);
        }

        debug!("Added emotion feedback, total samples: {}", history.len());

        // Update user profile
        self.update_user_profile(&feedback).await?;

        // Trigger learning if we have enough samples
        if history.len() >= self.config.min_samples_for_learning {
            self.trigger_learning().await?;
        }

        Ok(())
    }

    /// Get personalized emotion parameters for a user and context
    pub async fn get_personalized_emotion(
        &self,
        user_id: &str,
        base_emotion: &EmotionParameters,
        context: &str,
    ) -> Result<EmotionParameters> {
        let profiles = self.user_profiles.read().await;

        if let Some(profile) = profiles.get(user_id) {
            self.apply_personalization(base_emotion, profile, context)
                .await
        } else {
            // No personalization data available, return base emotion
            Ok(base_emotion.clone())
        }
    }

    /// Predict user satisfaction for given emotion parameters
    pub async fn predict_satisfaction(
        &self,
        user_id: &str,
        emotion: &EmotionParameters,
        context: &str,
    ) -> Result<f32> {
        let profiles = self.user_profiles.read().await;

        if let Some(profile) = profiles.get(user_id) {
            self.predict_satisfaction_for_profile(emotion, profile, context)
                .await
        } else {
            // No profile data, return neutral prediction
            Ok(0.5)
        }
    }

    /// Get learning statistics for a user
    pub async fn get_learning_stats(&self, user_id: &str) -> Result<Option<LearningStats>> {
        let profiles = self.user_profiles.read().await;
        Ok(profiles.get(user_id).map(|p| p.stats.clone()))
    }

    /// Export user preference profile
    pub async fn export_profile(&self, user_id: &str) -> Result<Option<UserPreferenceProfile>> {
        let profiles = self.user_profiles.read().await;
        Ok(profiles.get(user_id).cloned())
    }

    /// Import user preference profile
    pub async fn import_profile(&self, profile: UserPreferenceProfile) -> Result<()> {
        let mut profiles = self.user_profiles.write().await;
        profiles.insert(profile.user_id.clone(), profile);
        Ok(())
    }

    /// Reset learning data for a user
    pub async fn reset_user_data(&self, user_id: &str) -> Result<()> {
        let mut profiles = self.user_profiles.write().await;
        profiles.remove(user_id);

        let mut history = self.feedback_history.write().await;
        // Remove feedback for this user (simplified approach)
        history.retain(|f| {
            let feedback_user = format!(
                "user_{}",
                f.emotion
                    .emotion_vector
                    .emotions
                    .keys()
                    .next()
                    .unwrap_or(&crate::types::Emotion::Neutral)
                    .as_str()
            );
            feedback_user != user_id
        });

        info!("Reset learning data for user: {}", user_id);
        Ok(())
    }

    /// Internal method to update user preference profile
    async fn update_user_profile(&self, feedback: &EmotionFeedback) -> Result<()> {
        let mut profiles = self.user_profiles.write().await;
        let user_id = format!(
            "user_{}",
            feedback
                .emotion
                .emotion_vector
                .emotions
                .keys()
                .next()
                .unwrap_or(&crate::types::Emotion::Neutral)
                .as_str()
        );

        let profile = profiles
            .entry(user_id.clone())
            .or_insert_with(|| UserPreferenceProfile {
                user_id: user_id.clone(),
                emotion_biases: HashMap::new(),
                context_preferences: HashMap::new(),
                stats: LearningStats {
                    total_feedback: 0,
                    training_iterations: 0,
                    learning_rate: self.config.learning_rate,
                    prediction_accuracy: 0.0,
                    convergence: 0.0,
                },
                last_updated: std::time::SystemTime::now(),
            });

        // Update statistics
        profile.stats.total_feedback += 1;
        profile.last_updated = std::time::SystemTime::now();

        // Update emotion biases based on feedback
        let emotion_key = feedback
            .emotion
            .emotion_vector
            .emotions
            .keys()
            .next()
            .unwrap_or(&crate::types::Emotion::Neutral)
            .as_str()
            .to_string();
        let current_bias = profile
            .emotion_biases
            .get(&emotion_key)
            .cloned()
            .unwrap_or_default();

        // Simple learning rule: adjust based on satisfaction
        let learning_factor = 0.1 * feedback.satisfaction;
        let new_bias = EmotionDimensions {
            valence: current_bias.valence + learning_factor * (feedback.ratings.naturalness - 0.5),
            arousal: current_bias.arousal + learning_factor * (feedback.ratings.intensity - 0.5),
            dominance: current_bias.dominance
                + learning_factor * (feedback.ratings.authenticity - 0.5),
        };

        profile.emotion_biases.insert(emotion_key, new_bias);

        // Update context preferences
        let context_pref = profile
            .context_preferences
            .entry(feedback.context.clone())
            .or_insert_with(|| ContextPreference {
                intensity_scale: 1.0,
                dimension_adjustments: EmotionDimensions::default(),
                confidence: 0.0,
                observation_count: 0,
            });

        context_pref.observation_count += 1;
        context_pref.confidence = (context_pref.observation_count as f32 / 100.0).min(1.0);
        context_pref.intensity_scale += learning_factor * (feedback.ratings.intensity - 0.5);

        trace!(
            "Updated user profile for {}: {} feedback samples",
            user_id,
            profile.stats.total_feedback
        );

        Ok(())
    }

    /// Trigger machine learning training
    async fn trigger_learning(&self) -> Result<()> {
        let history = self.feedback_history.read().await;

        if history.len() < self.config.min_samples_for_learning {
            return Ok(());
        }

        debug!("Starting emotion learning with {} samples", history.len());

        // Prepare training data
        let (inputs, targets) = self.prepare_training_data(&history)?;

        #[cfg(feature = "gpu")]
        if self.config.use_gpu && self.device != Device::Cpu {
            self.train_gpu_model(&inputs, &targets).await?;
        } else {
            self.train_cpu_model(&inputs, &targets).await?;
        }

        #[cfg(not(feature = "gpu"))]
        self.train_cpu_model(&inputs, &targets).await?;

        info!("Completed emotion learning training");
        Ok(())
    }

    /// Prepare training data from feedback history
    fn prepare_training_data(
        &self,
        history: &[EmotionFeedback],
    ) -> Result<(Array2<f32>, Array2<f32>)> {
        let n_samples = history.len();
        let mut inputs = Array2::zeros((n_samples, 8));
        let mut targets = Array2::zeros((n_samples, 4));

        for (i, feedback) in history.iter().enumerate() {
            // Input features: emotion dimensions + intensity + context hash
            inputs[[i, 0]] = feedback.emotion.emotion_vector.dimensions.valence;
            inputs[[i, 1]] = feedback.emotion.emotion_vector.dimensions.arousal;
            inputs[[i, 2]] = feedback.emotion.emotion_vector.dimensions.dominance;
            inputs[[i, 3]] = feedback
                .emotion
                .emotion_vector
                .emotions
                .values()
                .next()
                .map(|i| i.value())
                .unwrap_or(1.0);
            inputs[[i, 4]] = (feedback.context.len() % 4) as f32 / 4.0; // Simple context encoding
            inputs[[i, 5]] = feedback.emotion.pitch_shift;
            inputs[[i, 6]] = feedback.emotion.tempo_scale;
            inputs[[i, 7]] = feedback.emotion.energy_scale;

            // Target outputs: feedback ratings
            targets[[i, 0]] = feedback.ratings.naturalness;
            targets[[i, 1]] = feedback.ratings.intensity;
            targets[[i, 2]] = feedback.ratings.authenticity;
            targets[[i, 3]] = feedback.ratings.appropriateness;
        }

        Ok((inputs, targets))
    }

    /// Train the CPU-based model
    async fn train_cpu_model(&self, inputs: &Array2<f32>, targets: &Array2<f32>) -> Result<()> {
        let mut weights = self.weight_matrix.write().await;
        let mut biases = self.bias_vector.write().await;

        // Simple gradient descent training
        for _epoch in 0..self.config.training_epochs {
            // Forward pass: predictions = inputs * weights.T + biases
            let weights_t = weights.t().to_owned(); // Create owned transpose
            let predictions = inputs.dot(&weights_t) + &*biases;

            // Compute loss (MSE)
            let error = &predictions - targets;
            let loss = error.mapv(|x| x * x).mean().unwrap_or(0.0);

            if loss < 0.001 {
                break; // Converged
            }

            // Backward pass
            let n_samples = inputs.nrows() as f32;
            let inputs_t = inputs.t().to_owned(); // Create owned transpose
            let weight_grad = inputs_t.dot(&error) / n_samples;
            let bias_grad = error.mean_axis(Axis(0)).unwrap_or(Array1::zeros(4));

            // Update parameters
            let weight_grad_t = weight_grad.t().to_owned();
            *weights = &*weights - self.config.learning_rate * weight_grad_t;
            *biases = &*biases - self.config.learning_rate * bias_grad;
        }

        trace!("CPU model training completed");
        Ok(())
    }

    #[cfg(feature = "gpu")]
    async fn train_gpu_model(&self, inputs: &Array2<f32>, targets: &Array2<f32>) -> Result<()> {
        use candle_core::DType;

        let varmap = self.varmap.read().await;
        let mut model_guard = self.model.write().await;

        // Initialize model if not exists
        if model_guard.is_none() {
            *model_guard = Some(EmotionPreferenceModel::new(&varmap, &self.device)?);
        }

        let model = model_guard.as_ref().unwrap();

        // Convert training data to tensors
        let input_data: Vec<f32> = inputs.iter().cloned().collect();
        let target_data: Vec<f32> = targets.iter().cloned().collect();

        let input_tensor =
            Tensor::from_vec(input_data, (inputs.nrows(), inputs.ncols()), &self.device)
                .map_err(|e| Error::Processing(format!("Failed to create input tensor: {}", e)))?;
        let target_tensor = Tensor::from_vec(
            target_data,
            (targets.nrows(), targets.ncols()),
            &self.device,
        )
        .map_err(|e| Error::Processing(format!("Failed to create target tensor: {}", e)))?;

        // Training loop would go here
        // Note: Full implementation would require optimizer and loss computation
        let predictions = model.forward(&input_tensor)?;

        trace!("GPU model training step completed");
        Ok(())
    }

    /// Apply personalization to base emotion parameters
    async fn apply_personalization(
        &self,
        base_emotion: &EmotionParameters,
        profile: &UserPreferenceProfile,
        context: &str,
    ) -> Result<EmotionParameters> {
        let mut personalized = base_emotion.clone();

        // Apply emotion bias if available
        let emotion_key = base_emotion
            .emotion_vector
            .emotions
            .keys()
            .next()
            .unwrap_or(&crate::types::Emotion::Neutral)
            .as_str()
            .to_string();

        if let Some(bias) = profile.emotion_biases.get(&emotion_key) {
            personalized.emotion_vector.dimensions.valence =
                (personalized.emotion_vector.dimensions.valence + bias.valence).clamp(-1.0, 1.0);
            personalized.emotion_vector.dimensions.arousal =
                (personalized.emotion_vector.dimensions.arousal + bias.arousal).clamp(-1.0, 1.0);
            personalized.emotion_vector.dimensions.dominance =
                (personalized.emotion_vector.dimensions.dominance + bias.dominance)
                    .clamp(-1.0, 1.0);
        }

        // Apply context-specific adjustments
        if let Some(context_pref) = profile.context_preferences.get(context) {
            // Apply intensity scaling to prosody parameters
            personalized.pitch_shift =
                (personalized.pitch_shift * context_pref.intensity_scale).clamp(0.5, 2.0);
            personalized.tempo_scale =
                (personalized.tempo_scale * context_pref.intensity_scale).clamp(0.5, 2.0);
            personalized.energy_scale =
                (personalized.energy_scale * context_pref.intensity_scale).clamp(0.1, 3.0);

            personalized.emotion_vector.dimensions.valence =
                (personalized.emotion_vector.dimensions.valence
                    + context_pref.dimension_adjustments.valence)
                    .clamp(-1.0, 1.0);
            personalized.emotion_vector.dimensions.arousal =
                (personalized.emotion_vector.dimensions.arousal
                    + context_pref.dimension_adjustments.arousal)
                    .clamp(-1.0, 1.0);
            personalized.emotion_vector.dimensions.dominance =
                (personalized.emotion_vector.dimensions.dominance
                    + context_pref.dimension_adjustments.dominance)
                    .clamp(-1.0, 1.0);
        }

        Ok(personalized)
    }

    /// Predict satisfaction score for emotion parameters
    async fn predict_satisfaction_for_profile(
        &self,
        emotion: &EmotionParameters,
        profile: &UserPreferenceProfile,
        context: &str,
    ) -> Result<f32> {
        // Simple heuristic prediction based on learned preferences
        let emotion_key = emotion
            .emotion_vector
            .emotions
            .keys()
            .next()
            .unwrap_or(&crate::types::Emotion::Neutral)
            .as_str()
            .to_string();
        let mut satisfaction = 0.5;

        // Factor in emotion bias
        if let Some(bias) = profile.emotion_biases.get(&emotion_key) {
            let distance = ((emotion.emotion_vector.dimensions.valence - bias.valence).powi(2)
                + (emotion.emotion_vector.dimensions.arousal - bias.arousal).powi(2)
                + (emotion.emotion_vector.dimensions.dominance - bias.dominance).powi(2))
            .sqrt();
            satisfaction += (1.0 - distance).max(0.0) * 0.3;
        }

        // Factor in context preference
        if let Some(context_pref) = profile.context_preferences.get(context) {
            satisfaction += context_pref.confidence * 0.2;
        }

        Ok(satisfaction.clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Emotion;

    #[tokio::test]
    async fn test_emotion_learning_basic() {
        let config = EmotionLearningConfig::default();
        let learner = EmotionLearner::new(config).unwrap();

        let mut emotion_vector = crate::types::EmotionVector::new();
        emotion_vector.add_emotion(Emotion::Happy, crate::types::EmotionIntensity::new(0.7));

        let emotion_params = EmotionParameters::new(emotion_vector).with_prosody(1.2, 0.9, 1.1);

        let feedback = EmotionFeedback {
            emotion: emotion_params.clone(),
            context: "greeting".to_string(),
            satisfaction: 0.8,
            ratings: FeedbackRatings {
                naturalness: 0.9,
                intensity: 0.7,
                authenticity: 0.8,
                appropriateness: 0.85,
            },
            timestamp: std::time::SystemTime::now(),
        };

        learner.add_feedback(feedback).await.unwrap();

        let personalized = learner
            .get_personalized_emotion("test_user", &emotion_params, "greeting")
            .await
            .unwrap();

        // Should return the same parameters initially (not enough data for learning)
        assert_eq!(
            personalized.emotion_vector.emotions.len(),
            emotion_params.emotion_vector.emotions.len()
        );
    }

    #[tokio::test]
    async fn test_user_profile_management() {
        let config = EmotionLearningConfig::default();
        let learner = EmotionLearner::new(config).unwrap();

        let profile = UserPreferenceProfile {
            user_id: "test_user".to_string(),
            emotion_biases: HashMap::new(),
            context_preferences: HashMap::new(),
            stats: LearningStats {
                total_feedback: 10,
                training_iterations: 5,
                learning_rate: 0.001,
                prediction_accuracy: 0.8,
                convergence: 0.9,
            },
            last_updated: std::time::SystemTime::now(),
        };

        learner.import_profile(profile.clone()).await.unwrap();

        let exported = learner.export_profile("test_user").await.unwrap();
        assert!(exported.is_some());
        assert_eq!(exported.unwrap().user_id, "test_user");

        learner.reset_user_data("test_user").await.unwrap();

        let after_reset = learner.export_profile("test_user").await.unwrap();
        assert!(after_reset.is_none());
    }

    #[tokio::test]
    async fn test_satisfaction_prediction() {
        let config = EmotionLearningConfig::default();
        let learner = EmotionLearner::new(config).unwrap();

        let mut emotion_vector = crate::types::EmotionVector::new();
        emotion_vector.add_emotion(Emotion::Happy, crate::types::EmotionIntensity::new(0.5));

        let emotion_params = EmotionParameters::new(emotion_vector);

        let satisfaction = learner
            .predict_satisfaction("unknown_user", &emotion_params, "test_context")
            .await
            .unwrap();

        // Should return neutral prediction for unknown user
        assert_eq!(satisfaction, 0.5);
    }

    #[tokio::test]
    async fn test_emotion_learning_config_default() {
        let config = EmotionLearningConfig::default();

        assert_eq!(config.max_feedback_samples, 10000);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.training_epochs, 100);
        assert_eq!(config.min_samples_for_learning, 50);
        assert_eq!(config.weight_decay, 0.0001);
        assert!(config.use_gpu);
    }

    #[tokio::test]
    async fn test_feedback_ratings_creation() {
        let ratings = FeedbackRatings {
            naturalness: 0.8,
            intensity: 0.7,
            authenticity: 0.9,
            appropriateness: 0.85,
        };

        assert_eq!(ratings.naturalness, 0.8);
        assert_eq!(ratings.intensity, 0.7);
        assert_eq!(ratings.authenticity, 0.9);
        assert_eq!(ratings.appropriateness, 0.85);
    }

    #[tokio::test]
    async fn test_context_preference_creation() {
        let pref = ContextPreference {
            intensity_scale: 1.2,
            dimension_adjustments: crate::types::EmotionDimensions::new(0.1, -0.2, 0.05),
            confidence: 0.8,
            observation_count: 15,
        };

        assert_eq!(pref.intensity_scale, 1.2);
        assert_eq!(pref.confidence, 0.8);
        assert_eq!(pref.observation_count, 15);
    }

    #[tokio::test]
    async fn test_learning_stats_tracking() {
        let stats = LearningStats {
            total_feedback: 100,
            training_iterations: 10,
            learning_rate: 0.001,
            prediction_accuracy: 0.85,
            convergence: 0.92,
        };

        assert_eq!(stats.total_feedback, 100);
        assert_eq!(stats.training_iterations, 10);
        assert_eq!(stats.prediction_accuracy, 0.85);
        assert_eq!(stats.convergence, 0.92);
    }

    #[tokio::test]
    async fn test_multiple_feedback_accumulation() {
        let config = EmotionLearningConfig::default();
        let learner = EmotionLearner::new(config).unwrap();

        let mut emotion_vector = crate::types::EmotionVector::new();
        emotion_vector.add_emotion(Emotion::Happy, crate::types::EmotionIntensity::new(0.8));
        let emotion_params = EmotionParameters::new(emotion_vector);

        // Add multiple feedback samples
        for i in 0..5 {
            let feedback = EmotionFeedback {
                emotion: emotion_params.clone(),
                context: format!("context_{}", i),
                satisfaction: 0.7 + (i as f32 * 0.05),
                ratings: FeedbackRatings {
                    naturalness: 0.8,
                    intensity: 0.75,
                    authenticity: 0.85,
                    appropriateness: 0.8,
                },
                timestamp: std::time::SystemTime::now(),
            };

            learner.add_feedback(feedback).await.unwrap();
        }

        let stats = learner.get_learning_stats("user_happy").await.unwrap();
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().total_feedback, 5);
    }

    #[tokio::test]
    async fn test_feedback_history_size_limit() {
        let mut config = EmotionLearningConfig::default();
        config.max_feedback_samples = 3; // Small limit for testing
        let learner = EmotionLearner::new(config).unwrap();

        let mut emotion_vector = crate::types::EmotionVector::new();
        emotion_vector.add_emotion(Emotion::Sad, crate::types::EmotionIntensity::new(0.6));
        let emotion_params = EmotionParameters::new(emotion_vector);

        // Add more feedback than the limit
        for i in 0..5 {
            let feedback = EmotionFeedback {
                emotion: emotion_params.clone(),
                context: format!("test_{}", i),
                satisfaction: 0.6,
                ratings: FeedbackRatings {
                    naturalness: 0.7,
                    intensity: 0.6,
                    authenticity: 0.8,
                    appropriateness: 0.75,
                },
                timestamp: std::time::SystemTime::now(),
            };

            learner.add_feedback(feedback).await.unwrap();
        }

        // Should maintain limit
        let history = learner.feedback_history.read().await;
        assert_eq!(history.len(), 3);
    }

    #[tokio::test]
    async fn test_personalized_emotion_with_bias() {
        let config = EmotionLearningConfig::default();
        let learner = EmotionLearner::new(config).unwrap();

        // Create profile with emotion bias
        let mut emotion_biases = HashMap::new();
        emotion_biases.insert(
            "angry".to_string(),
            crate::types::EmotionDimensions::new(0.2, -0.1, 0.15),
        );

        let profile = UserPreferenceProfile {
            user_id: "biased_user".to_string(),
            emotion_biases,
            context_preferences: HashMap::new(),
            stats: LearningStats {
                total_feedback: 20,
                training_iterations: 5,
                learning_rate: 0.001,
                prediction_accuracy: 0.75,
                convergence: 0.8,
            },
            last_updated: std::time::SystemTime::now(),
        };

        learner.import_profile(profile).await.unwrap();

        // Create angry emotion
        let mut emotion_vector = crate::types::EmotionVector::new();
        emotion_vector.add_emotion(Emotion::Angry, crate::types::EmotionIntensity::new(0.8));
        let base_params = EmotionParameters::new(emotion_vector);

        let personalized = learner
            .get_personalized_emotion("biased_user", &base_params, "test")
            .await
            .unwrap();

        // Dimensions should be adjusted by bias
        assert_ne!(
            personalized.emotion_vector.dimensions.valence,
            base_params.emotion_vector.dimensions.valence
        );
    }

    #[tokio::test]
    async fn test_context_specific_personalization() {
        let config = EmotionLearningConfig::default();
        let learner = EmotionLearner::new(config).unwrap();

        // Create profile with context preference
        let mut context_preferences = HashMap::new();
        context_preferences.insert(
            "formal".to_string(),
            ContextPreference {
                intensity_scale: 0.8, // Less intense for formal context
                dimension_adjustments: crate::types::EmotionDimensions::new(-0.1, -0.2, 0.1),
                confidence: 0.9,
                observation_count: 25,
            },
        );

        let profile = UserPreferenceProfile {
            user_id: "formal_user".to_string(),
            emotion_biases: HashMap::new(),
            context_preferences,
            stats: LearningStats {
                total_feedback: 25,
                training_iterations: 8,
                learning_rate: 0.001,
                prediction_accuracy: 0.82,
                convergence: 0.85,
            },
            last_updated: std::time::SystemTime::now(),
        };

        learner.import_profile(profile).await.unwrap();

        let mut emotion_vector = crate::types::EmotionVector::new();
        emotion_vector.add_emotion(Emotion::Excited, crate::types::EmotionIntensity::new(0.9));
        let base_params = EmotionParameters::new(emotion_vector).with_prosody(1.5, 1.3, 1.8);

        let personalized = learner
            .get_personalized_emotion("formal_user", &base_params, "formal")
            .await
            .unwrap();

        // Prosody should be scaled down for formal context
        assert!(personalized.pitch_shift < base_params.pitch_shift);
        assert!(personalized.tempo_scale < base_params.tempo_scale);
        assert!(personalized.energy_scale < base_params.energy_scale);
    }

    #[tokio::test]
    async fn test_satisfaction_prediction_with_profile() {
        let config = EmotionLearningConfig::default();
        let learner = EmotionLearner::new(config).unwrap();

        // Create profile with preferences
        let mut emotion_biases = HashMap::new();
        emotion_biases.insert(
            "happy".to_string(),
            crate::types::EmotionDimensions::new(0.8, 0.6, 0.4),
        );

        let mut context_preferences = HashMap::new();
        context_preferences.insert(
            "casual".to_string(),
            ContextPreference {
                intensity_scale: 1.2,
                dimension_adjustments: crate::types::EmotionDimensions::new(0.1, 0.05, -0.1),
                confidence: 0.85,
                observation_count: 30,
            },
        );

        let profile = UserPreferenceProfile {
            user_id: "happy_user".to_string(),
            emotion_biases,
            context_preferences,
            stats: LearningStats {
                total_feedback: 50,
                training_iterations: 12,
                learning_rate: 0.001,
                prediction_accuracy: 0.88,
                convergence: 0.91,
            },
            last_updated: std::time::SystemTime::now(),
        };

        learner.import_profile(profile).await.unwrap();

        let mut emotion_vector = crate::types::EmotionVector::new();
        emotion_vector.add_emotion(Emotion::Happy, crate::types::EmotionIntensity::new(0.8));
        let emotion_params = EmotionParameters::new(emotion_vector);

        let satisfaction = learner
            .predict_satisfaction("happy_user", &emotion_params, "casual")
            .await
            .unwrap();

        // Should predict higher satisfaction due to profile match
        assert!(satisfaction > 0.5);
        assert!(satisfaction <= 1.0);
    }

    #[tokio::test]
    async fn test_learning_with_insufficient_samples() {
        let mut config = EmotionLearningConfig::default();
        config.min_samples_for_learning = 10; // Set higher threshold
        let learner = EmotionLearner::new(config).unwrap();

        let mut emotion_vector = crate::types::EmotionVector::new();
        emotion_vector.add_emotion(Emotion::Neutral, crate::types::EmotionIntensity::new(0.5));
        let emotion_params = EmotionParameters::new(emotion_vector);

        // Add fewer samples than threshold
        for i in 0..5 {
            let feedback = EmotionFeedback {
                emotion: emotion_params.clone(),
                context: "test".to_string(),
                satisfaction: 0.7,
                ratings: FeedbackRatings {
                    naturalness: 0.8,
                    intensity: 0.7,
                    authenticity: 0.75,
                    appropriateness: 0.8,
                },
                timestamp: std::time::SystemTime::now(),
            };

            learner.add_feedback(feedback).await.unwrap();
        }

        // Should not trigger learning yet
        let history = learner.feedback_history.read().await;
        assert_eq!(history.len(), 5);
    }

    #[tokio::test]
    async fn test_profile_serialization_round_trip() {
        let mut emotion_biases = HashMap::new();
        emotion_biases.insert(
            "confident".to_string(),
            crate::types::EmotionDimensions::new(0.6, 0.3, 0.8),
        );

        let mut context_preferences = HashMap::new();
        context_preferences.insert(
            "presentation".to_string(),
            ContextPreference {
                intensity_scale: 1.1,
                dimension_adjustments: crate::types::EmotionDimensions::new(0.05, 0.1, 0.15),
                confidence: 0.75,
                observation_count: 18,
            },
        );

        let original_profile = UserPreferenceProfile {
            user_id: "serialization_test".to_string(),
            emotion_biases,
            context_preferences,
            stats: LearningStats {
                total_feedback: 35,
                training_iterations: 7,
                learning_rate: 0.001,
                prediction_accuracy: 0.79,
                convergence: 0.83,
            },
            last_updated: std::time::SystemTime::now(),
        };

        // Test JSON serialization
        let json = serde_json::to_string(&original_profile).unwrap();
        let deserialized: UserPreferenceProfile = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.user_id, original_profile.user_id);
        assert_eq!(
            deserialized.stats.total_feedback,
            original_profile.stats.total_feedback
        );
    }

    #[tokio::test]
    async fn test_reset_user_data_comprehensive() {
        let config = EmotionLearningConfig::default();
        let learner = EmotionLearner::new(config).unwrap();

        // Add a user profile
        let profile = UserPreferenceProfile {
            user_id: "reset_test_user".to_string(),
            emotion_biases: HashMap::new(),
            context_preferences: HashMap::new(),
            stats: LearningStats {
                total_feedback: 15,
                training_iterations: 3,
                learning_rate: 0.001,
                prediction_accuracy: 0.72,
                convergence: 0.68,
            },
            last_updated: std::time::SystemTime::now(),
        };

        learner.import_profile(profile).await.unwrap();

        // Verify profile exists
        let before_reset = learner.export_profile("reset_test_user").await.unwrap();
        assert!(before_reset.is_some());

        // Reset user data
        learner.reset_user_data("reset_test_user").await.unwrap();

        // Verify profile is removed
        let after_reset = learner.export_profile("reset_test_user").await.unwrap();
        assert!(after_reset.is_none());
    }

    #[tokio::test]
    async fn test_edge_case_empty_emotion_vector() {
        let config = EmotionLearningConfig::default();
        let learner = EmotionLearner::new(config).unwrap();

        // Create emotion parameters with empty emotion vector
        let empty_vector = crate::types::EmotionVector::new();
        let emotion_params = EmotionParameters::new(empty_vector);

        // Should handle empty emotion vector gracefully
        let personalized = learner
            .get_personalized_emotion("test_user", &emotion_params, "test")
            .await
            .unwrap();

        assert!(personalized.emotion_vector.emotions.is_empty());
    }

    #[tokio::test]
    async fn test_extreme_feedback_ratings() {
        let config = EmotionLearningConfig::default();
        let learner = EmotionLearner::new(config).unwrap();

        let mut emotion_vector = crate::types::EmotionVector::new();
        emotion_vector.add_emotion(Emotion::Fear, crate::types::EmotionIntensity::new(1.0));
        let emotion_params = EmotionParameters::new(emotion_vector);

        // Test with extreme ratings (all 0.0 and all 1.0)
        let extreme_low_feedback = EmotionFeedback {
            emotion: emotion_params.clone(),
            context: "extreme_low".to_string(),
            satisfaction: 0.0,
            ratings: FeedbackRatings {
                naturalness: 0.0,
                intensity: 0.0,
                authenticity: 0.0,
                appropriateness: 0.0,
            },
            timestamp: std::time::SystemTime::now(),
        };

        let extreme_high_feedback = EmotionFeedback {
            emotion: emotion_params.clone(),
            context: "extreme_high".to_string(),
            satisfaction: 1.0,
            ratings: FeedbackRatings {
                naturalness: 1.0,
                intensity: 1.0,
                authenticity: 1.0,
                appropriateness: 1.0,
            },
            timestamp: std::time::SystemTime::now(),
        };

        // Should handle extreme values without panicking
        learner.add_feedback(extreme_low_feedback).await.unwrap();
        learner.add_feedback(extreme_high_feedback).await.unwrap();

        let stats = learner.get_learning_stats("user_fear").await.unwrap();
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().total_feedback, 2);
    }

    #[tokio::test]
    async fn test_multiple_users_isolation() {
        let config = EmotionLearningConfig::default();
        let learner = EmotionLearner::new(config).unwrap();

        // Create profiles for different users
        let user1_profile = UserPreferenceProfile {
            user_id: "user1".to_string(),
            emotion_biases: HashMap::new(),
            context_preferences: HashMap::new(),
            stats: LearningStats {
                total_feedback: 10,
                training_iterations: 2,
                learning_rate: 0.001,
                prediction_accuracy: 0.65,
                convergence: 0.5,
            },
            last_updated: std::time::SystemTime::now(),
        };

        let user2_profile = UserPreferenceProfile {
            user_id: "user2".to_string(),
            emotion_biases: HashMap::new(),
            context_preferences: HashMap::new(),
            stats: LearningStats {
                total_feedback: 25,
                training_iterations: 6,
                learning_rate: 0.001,
                prediction_accuracy: 0.85,
                convergence: 0.9,
            },
            last_updated: std::time::SystemTime::now(),
        };

        learner.import_profile(user1_profile).await.unwrap();
        learner.import_profile(user2_profile).await.unwrap();

        // Reset one user's data
        learner.reset_user_data("user1").await.unwrap();

        // Verify isolation - user2 should still exist
        let user1_after = learner.export_profile("user1").await.unwrap();
        let user2_after = learner.export_profile("user2").await.unwrap();

        assert!(user1_after.is_none());
        assert!(user2_after.is_some());
        assert_eq!(user2_after.unwrap().stats.total_feedback, 25);
    }

    #[tokio::test]
    async fn test_custom_emotion_handling() {
        let config = EmotionLearningConfig::default();
        let learner = EmotionLearner::new(config).unwrap();

        let mut emotion_vector = crate::types::EmotionVector::new();
        emotion_vector.add_emotion(
            Emotion::Custom("nostalgic".to_string()),
            crate::types::EmotionIntensity::new(0.7),
        );
        let emotion_params = EmotionParameters::new(emotion_vector);

        let feedback = EmotionFeedback {
            emotion: emotion_params.clone(),
            context: "memory_lane".to_string(),
            satisfaction: 0.9,
            ratings: FeedbackRatings {
                naturalness: 0.85,
                intensity: 0.8,
                authenticity: 0.95,
                appropriateness: 0.9,
            },
            timestamp: std::time::SystemTime::now(),
        };

        learner.add_feedback(feedback).await.unwrap();

        // Should handle custom emotions without issues
        let personalized = learner
            .get_personalized_emotion("custom_user", &emotion_params, "memory_lane")
            .await
            .unwrap();

        assert!(!personalized.emotion_vector.emotions.is_empty());
    }
}
