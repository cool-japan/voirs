//! Core adaptive feedback engine implementation
//!
//! This module contains the main AdaptiveFeedbackEngine and its core functionality.

use super::models::*;
use super::types::*;
use crate::progress::TrendDirection;
use crate::traits::{
    AdaptiveConfig, AdaptiveLearner, AdaptiveState, FeedbackContext, FeedbackProvider,
    FeedbackResponse, FeedbackResult, FeedbackStyle, FocusArea, InteractionType,
    LearningRecommendation, PerformanceData, ProgressIndicators, SkillEstimate, UserInteraction,
    UserResponse,
};
use crate::FeedbackError;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use statrs::statistics::Statistics;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Adaptive feedback engine that learns from user interactions
#[derive(Clone)]
pub struct AdaptiveFeedbackEngine {
    /// User models keyed by user ID
    user_models: Arc<RwLock<HashMap<String, UserModel>>>,
    /// Learning algorithm
    learning_algorithm: Arc<RwLock<LearningAlgorithm>>,
    /// Configuration
    config: AdaptiveConfig,
    /// System metrics
    metrics: Arc<RwLock<AdaptiveMetrics>>,
}

impl AdaptiveFeedbackEngine {
    /// Create a new adaptive feedback engine
    pub async fn new() -> Result<Self, FeedbackError> {
        Self::with_config(AdaptiveConfig::default()).await
    }

    /// Create with custom configuration
    pub async fn with_config(config: AdaptiveConfig) -> Result<Self, FeedbackError> {
        let learning_algorithm = LearningAlgorithm::new(&config)?;

        Ok(Self {
            user_models: Arc::new(RwLock::new(HashMap::new())),
            learning_algorithm: Arc::new(RwLock::new(learning_algorithm)),
            config,
            metrics: Arc::new(RwLock::new(AdaptiveMetrics::default())),
        })
    }

    /// Get or create user model
    pub async fn get_user_model(&self, user_id: &str) -> Result<UserModel, FeedbackError> {
        {
            let models = self.user_models.read().unwrap();
            if let Some(model) = models.get(user_id) {
                return Ok(model.clone());
            }
        }
        self.create_user_model(user_id).await
    }

    /// Create a new user model
    async fn create_user_model(&self, user_id: &str) -> Result<UserModel, FeedbackError> {
        let model = UserModel::new(user_id.to_string());

        {
            let mut models = self.user_models.write().unwrap();
            models.insert(user_id.to_string(), model.clone());
        }

        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_users += 1;
        }

        Ok(model)
    }

    /// Predict optimal feedback strategy for user
    pub async fn predict_feedback_strategy(
        &self,
        user_id: &str,
        context: &FeedbackContext,
    ) -> Result<FeedbackStrategy, FeedbackError> {
        let user_model = self.get_user_model(user_id).await?;

        // Simple strategy prediction based on user model
        let strategy_type = if user_model.confidence < 0.3 {
            StrategyType::Encouraging
        } else if user_model.learning_rate > 0.7 {
            StrategyType::Technical
        } else {
            StrategyType::Adaptive
        };

        let tone = if user_model.consistency_score < 0.5 {
            FeedbackTone::Positive
        } else {
            FeedbackTone::Neutral
        };

        Ok(FeedbackStrategy {
            strategy_type,
            tone,
            detail_level: user_model.skill_level,
            personalization_factors: vec!["skill_level".to_string(), "consistency".to_string()],
        })
    }

    /// Update user model with new interaction and performance data
    pub async fn update_user_model(
        &self,
        user_id: &str,
        interaction: &UserInteraction,
        performance: &PerformanceData,
    ) -> Result<(), FeedbackError> {
        let mut models = self.user_models.write().unwrap();

        if let Some(model) = models.get_mut(user_id) {
            // Update interaction history
            model.interaction_history.push_back(interaction.clone());
            if model.interaction_history.len() > 100 {
                model.interaction_history.pop_front();
            }

            // Update performance history
            model.performance_history.push_back(performance.clone());
            if model.performance_history.len() > 100 {
                model.performance_history.pop_front();
            }

            // Update skill level based on recent performance
            if let Some(latest_quality) = performance.quality_scores.last() {
                model.skill_level = (model.skill_level + latest_quality) / 2.0;
            }

            // Update consistency score
            model.consistency_score = performance.consistency;

            // Update learning rate
            model.learning_rate = performance.learning_velocity;

            // Update confidence based on amount of data
            let data_points = model.interaction_history.len() as f32;
            model.confidence = (data_points / 10.0).min(1.0);

            model.last_updated = Utc::now();
        }

        // Update system metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_adaptations += 1;
        }

        Ok(())
    }

    /// Generate personalized recommendations
    pub async fn generate_personalized_recommendations(
        &self,
        user_id: &str,
        base_feedback: &FeedbackResponse,
    ) -> Result<Vec<PersonalizedRecommendation>, FeedbackError> {
        let user_model = self.get_user_model(user_id).await?;
        let mut recommendations = Vec::new();

        // Analyze skill gaps and generate recommendations
        for (focus_area, skill_level) in &user_model.skill_breakdown {
            if *skill_level < 0.7 {
                let recommendation = PersonalizedRecommendation {
                    focus_area: focus_area.clone(),
                    recommendation_type: RecommendationType::Practice,
                    priority: 1.0 - skill_level,
                    estimated_impact: 0.8,
                    confidence: user_model.confidence,
                    explanation: format!(
                        "Focus on {} to improve from {:.1}%",
                        format!("{:?}", focus_area),
                        skill_level * 100.0
                    ),
                    exercises: vec![format!("{:?} practice exercises", focus_area)],
                };
                recommendations.push(recommendation);
            }
        }

        // Sort by priority
        recommendations.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        Ok(recommendations)
    }

    /// Get system statistics
    pub async fn get_statistics(&self) -> Result<AdaptiveSystemStats, FeedbackError> {
        let models = self.user_models.read().unwrap();
        let metrics = self.metrics.read().unwrap();

        let active_users = models
            .values()
            .filter(|model| {
                Utc::now()
                    .signed_duration_since(model.last_updated)
                    .num_hours()
                    < 24
            })
            .count();

        let average_confidence = if models.is_empty() {
            0.0
        } else {
            models.values().map(|m| m.confidence).sum::<f32>() / models.len() as f32
        };

        Ok(AdaptiveSystemStats {
            total_users: metrics.total_users,
            active_users,
            total_interactions: models.values().map(|m| m.interaction_history.len()).sum(),
            total_adaptations: metrics.total_adaptations,
            total_feedback_count: metrics.total_feedback_count,
            average_model_confidence: average_confidence,
        })
    }

    /// Extract features from user model and context
    fn extract_features(
        &self,
        user_model: &UserModel,
        context: &FeedbackContext,
    ) -> Result<FeatureVector, FeedbackError> {
        let mut features = Vec::new();

        // Add basic user model features
        features.push(user_model.skill_level);
        features.push(user_model.learning_rate);
        features.push(user_model.consistency_score);
        features.push(user_model.confidence);

        // Add contextual features
        features.push(context.history.len() as f32 / 10.0); // Normalized history length

        let feature_names = vec![
            "skill_level".to_string(),
            "learning_rate".to_string(),
            "consistency_score".to_string(),
            "confidence".to_string(),
            "history_length".to_string(),
        ];

        Ok(FeatureVector {
            features,
            feature_names,
        })
    }

    /// Analyze audio quality metrics
    fn analyze_audio_quality(&self, audio: &voirs_sdk::AudioBuffer) -> f32 {
        // Calculate signal-to-noise ratio approximation
        let samples = audio.samples();
        if samples.is_empty() {
            return 0.0;
        }

        let signal_power: f32 = samples.iter().map(|s| s * s).sum();
        let mean_power = signal_power / samples.len() as f32;

        // Estimate quality based on signal characteristics
        if mean_power > 0.1 {
            0.9 // High quality
        } else if mean_power > 0.01 {
            0.7 // Medium quality
        } else {
            0.3 // Low quality
        }
    }

    /// Analyze pronunciation quality
    fn analyze_pronunciation(&self, audio: &voirs_sdk::AudioBuffer, text: &str) -> f32 {
        let samples = audio.samples();
        if samples.is_empty() || text.is_empty() {
            return 0.0;
        }

        // Basic pronunciation analysis based on audio characteristics
        let energy_variance = self.calculate_energy_variance(samples);
        let zero_crossing_rate = self.calculate_zero_crossing_rate(samples);

        // Estimate pronunciation quality based on acoustic features
        let energy_score = (1.0 - energy_variance).max(0.0);
        let articulation_score = (zero_crossing_rate * 2.0).min(1.0);

        (energy_score + articulation_score) / 2.0
    }

    /// Analyze fluency metrics
    fn analyze_fluency(&self, audio: &voirs_sdk::AudioBuffer, text: &str) -> f32 {
        let samples = audio.samples();
        if samples.is_empty() || text.is_empty() {
            return 0.0;
        }

        // Calculate speaking rate
        let duration_seconds = audio.duration(); // duration() returns f32 representing seconds
        let word_count = text.split_whitespace().count();
        let speaking_rate = word_count as f32 / duration_seconds;

        // Optimal speaking rate is around 150-180 words per minute (2.5-3 words per second)
        let rate_score = if speaking_rate >= 2.0 && speaking_rate <= 3.5 {
            1.0 - (speaking_rate - 2.75).abs() / 0.75
        } else {
            0.3 // Outside optimal range
        };

        // Analyze pause patterns
        let pause_score = self.analyze_pause_patterns(samples);

        (rate_score + pause_score) / 2.0
    }

    /// Calculate energy variance for pronunciation analysis
    fn calculate_energy_variance(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let energy: Vec<f32> = samples
            .windows(256)
            .map(|window| window.iter().map(|s| s * s).sum::<f32>() / window.len() as f32)
            .collect();

        let mean_energy = energy.iter().sum::<f32>() / energy.len() as f32;
        let variance = energy
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f32>()
            / energy.len() as f32;

        variance.sqrt()
    }

    /// Calculate zero crossing rate for articulation analysis
    fn calculate_zero_crossing_rate(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let zero_crossings = samples
            .windows(2)
            .filter(|window| {
                (window[0] > 0.0 && window[1] < 0.0) || (window[0] < 0.0 && window[1] > 0.0)
            })
            .count();

        zero_crossings as f32 / (samples.len() - 1) as f32
    }

    /// Analyze pause patterns for fluency assessment
    fn analyze_pause_patterns(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // Simple pause detection based on energy thresholds
        let energy_threshold = 0.01;
        let window_size = 1024;

        let mut pauses = 0;
        let mut total_windows = 0;

        for window in samples.chunks(window_size) {
            let energy = window.iter().map(|s| s * s).sum::<f32>() / window.len() as f32;
            if energy < energy_threshold {
                pauses += 1;
            }
            total_windows += 1;
        }

        if total_windows == 0 {
            return 0.0;
        }

        let pause_ratio = pauses as f32 / total_windows as f32;

        // Optimal pause ratio is around 10-20%
        if pause_ratio >= 0.1 && pause_ratio <= 0.2 {
            1.0 - (pause_ratio - 0.15).abs() / 0.05
        } else {
            0.5 // Outside optimal range
        }
    }
}

#[async_trait]
impl FeedbackProvider for AdaptiveFeedbackEngine {
    async fn generate_feedback(
        &self,
        audio: &voirs_sdk::AudioBuffer,
        text: &str,
        context: &FeedbackContext,
    ) -> crate::FeedbackResult<FeedbackResponse> {
        let start_time = std::time::Instant::now();

        // Analyze audio characteristics
        let audio_duration = audio.duration();
        let audio_quality = self.analyze_audio_quality(audio);

        // Generate feedback based on user model and context
        let user_models = self.user_models.read().unwrap();
        let user_model = user_models
            .get(&context.session.user_id)
            .cloned()
            .unwrap_or_else(|| UserModel {
                user_id: context.session.user_id.clone(),
                skill_level: 0.5,
                learning_rate: 0.1,
                consistency_score: 0.0,
                skill_breakdown: HashMap::new(),
                interaction_history: VecDeque::new(),
                performance_history: VecDeque::new(),
                adaptive_state: context.session.adaptive_state.clone(),
                confidence: 0.3,
                last_updated: Utc::now(),
            });

        let mut feedback_items = Vec::new();
        let mut immediate_actions = Vec::new();
        let mut long_term_goals = Vec::new();

        // Analyze pronunciation and fluency
        let pronunciation_score = self.analyze_pronunciation(audio, text);
        let fluency_score = self.analyze_fluency(audio, text);

        // Generate feedback based on skill level
        if pronunciation_score < 0.7 {
            feedback_items.push(crate::UserFeedback {
                message: "Focus on clearer pronunciation of consonants".to_string(),
                suggestion: Some(
                    "Practice consonant sounds with focus on articulation".to_string(),
                ),
                confidence: 0.8,
                score: pronunciation_score,
                priority: 0.8,
                metadata: std::collections::HashMap::new(),
            });

            immediate_actions.push("Practice consonant sounds".to_string());
        }

        if fluency_score < 0.6 {
            feedback_items.push(crate::UserFeedback {
                message: "Work on speaking rhythm and pace".to_string(),
                suggestion: Some("Practice with metronome to improve rhythm".to_string()),
                confidence: 0.7,
                score: fluency_score,
                priority: 0.6,
                metadata: std::collections::HashMap::new(),
            });

            immediate_actions.push("Practice with metronome".to_string());
        }

        // Generate long-term goals based on user progress
        let skill_level = user_model.skill_level;

        if skill_level < 0.5 {
            long_term_goals.push("Achieve 80% pronunciation accuracy".to_string());
            long_term_goals.push("Complete basic pronunciation exercises".to_string());
            long_term_goals.push("Record 10 practice sessions".to_string());
        }

        // Calculate overall score
        let overall_score =
            (pronunciation_score * 0.6 + fluency_score * 0.4 + audio_quality * 0.2).min(1.0);

        // Determine improving/attention areas
        let mut improving_areas = Vec::new();
        let mut attention_areas = Vec::new();
        let mut stable_areas = Vec::new();

        if pronunciation_score > 0.8 {
            improving_areas.push("Pronunciation".to_string());
        } else if pronunciation_score < 0.6 {
            attention_areas.push("Pronunciation".to_string());
        } else {
            stable_areas.push("Pronunciation".to_string());
        }

        if fluency_score > 0.8 {
            improving_areas.push("Fluency".to_string());
        } else if fluency_score < 0.6 {
            attention_areas.push("Fluency".to_string());
        } else {
            stable_areas.push("Fluency".to_string());
        }

        let processing_time = start_time.elapsed();

        Ok(FeedbackResponse {
            feedback_items,
            overall_score,
            immediate_actions,
            long_term_goals,
            progress_indicators: ProgressIndicators {
                improving_areas,
                attention_areas,
                stable_areas,
                overall_trend: overall_score - user_model.skill_level,
                completion_percentage: (overall_score * 100.0).min(100.0),
            },
            timestamp: Utc::now(),
            processing_time,
            feedback_type: crate::FeedbackType::Adaptive,
        })
    }

    async fn generate_feedback_with_scores(
        &self,
        audio: &voirs_sdk::AudioBuffer,
        quality_score: &voirs_evaluation::QualityScore,
        pronunciation_score: &voirs_evaluation::PronunciationScore,
        context: &FeedbackContext,
    ) -> crate::FeedbackResult<FeedbackResponse> {
        let start_time = std::time::Instant::now();

        // Get user model for adaptive feedback
        let user_model = {
            let models = self.user_models.read().unwrap();
            models
                .get(&context.session.user_id)
                .cloned()
                .unwrap_or_default()
        };

        // Combine quality and pronunciation scores for comprehensive analysis
        let overall_score = (quality_score.overall_score + pronunciation_score.overall_score) / 2.0;

        let mut feedback_items = Vec::new();
        let mut immediate_actions = Vec::new();
        let mut long_term_goals = Vec::new();
        let mut improving_areas = Vec::new();
        let mut attention_areas = Vec::new();
        let mut stable_areas = Vec::new();

        // Quality-based feedback
        if quality_score.overall_score < 0.6 {
            feedback_items.push(crate::UserFeedback {
                message: format!(
                    "Audio quality can be improved (current: {:.1}%). Focus on recording environment and microphone placement.",
                    quality_score.overall_score * 100.0
                ),
                suggestion: Some("Try recording in a quieter environment with better microphone positioning.".to_string()),
                confidence: quality_score.confidence,
                score: quality_score.overall_score,
                priority: 0.8,
                metadata: std::collections::HashMap::new(),
            });
            attention_areas.push("Audio Quality".to_string());
            immediate_actions.push("Check recording setup and environment".to_string());
        } else if quality_score.overall_score > 0.8 {
            improving_areas.push("Audio Quality".to_string());
        } else {
            stable_areas.push("Audio Quality".to_string());
        }

        // Pronunciation-based feedback
        if pronunciation_score.overall_score < 0.7 {
            feedback_items.push(crate::UserFeedback {
                message: format!(
                    "Pronunciation accuracy: {:.1}%. Consider practicing specific phonemes.",
                    pronunciation_score.overall_score * 100.0
                ),
                suggestion: Some(
                    "Practice with phoneme-specific exercises and listen to native speakers."
                        .to_string(),
                ),
                confidence: 0.9,
                score: pronunciation_score.overall_score,
                priority: 0.9,
                metadata: std::collections::HashMap::new(),
            });
            attention_areas.push("Pronunciation Accuracy".to_string());
            immediate_actions.push("Practice challenging phonemes".to_string());
        } else if pronunciation_score.overall_score > 0.85 {
            improving_areas.push("Pronunciation Accuracy".to_string());
        } else {
            stable_areas.push("Pronunciation Accuracy".to_string());
        }

        // Fluency feedback
        if pronunciation_score.fluency_score < 0.6 {
            feedback_items.push(crate::UserFeedback {
                message: format!(
                    "Fluency score: {:.1}%. Work on smooth speech flow.",
                    pronunciation_score.fluency_score * 100.0
                ),
                suggestion: Some(
                    "Practice reading aloud regularly to improve speech flow.".to_string(),
                ),
                confidence: 0.7,
                score: pronunciation_score.fluency_score,
                priority: 0.6,
                metadata: std::collections::HashMap::new(),
            });
            attention_areas.push("Speech Fluency".to_string());
            long_term_goals.push("Achieve consistent fluent speech patterns".to_string());
        } else if pronunciation_score.fluency_score > 0.8 {
            improving_areas.push("Speech Fluency".to_string());
        } else {
            stable_areas.push("Speech Fluency".to_string());
        }

        // Add quality recommendations
        for recommendation in &quality_score.recommendations {
            feedback_items.push(crate::UserFeedback {
                message: recommendation.clone(),
                suggestion: None,
                confidence: quality_score.confidence,
                score: quality_score.overall_score,
                priority: 0.3,
                metadata: std::collections::HashMap::new(),
            });
        }

        // Adaptive elements based on user model
        if user_model.skill_level < 0.5 {
            immediate_actions.push("Focus on basic pronunciation accuracy".to_string());
            long_term_goals.push("Build fundamental speaking skills".to_string());
        } else if user_model.skill_level > 0.8 {
            immediate_actions.push("Refine advanced speaking techniques".to_string());
            long_term_goals.push("Achieve native-like fluency".to_string());
        }

        let processing_time = start_time.elapsed();

        Ok(FeedbackResponse {
            feedback_items,
            overall_score,
            immediate_actions,
            long_term_goals,
            progress_indicators: ProgressIndicators {
                improving_areas,
                attention_areas,
                stable_areas,
                overall_trend: overall_score - user_model.skill_level,
                completion_percentage: (overall_score * 100.0).min(100.0),
            },
            timestamp: chrono::Utc::now(),
            processing_time,
            feedback_type: crate::FeedbackType::Adaptive,
        })
    }

    fn supported_feedback_types(&self) -> Vec<crate::traits::FeedbackType> {
        vec![
            crate::traits::FeedbackType::Quality,
            crate::traits::FeedbackType::Pronunciation,
            crate::traits::FeedbackType::Motivational,
        ]
    }

    fn metadata(&self) -> crate::traits::FeedbackProviderMetadata {
        crate::traits::FeedbackProviderMetadata {
            name: "Adaptive Feedback Engine".to_string(),
            version: "1.0.0".to_string(),
            description: "Machine learning-driven personalized feedback system".to_string(),
            supported_types: vec![
                crate::traits::FeedbackType::Quality,
                crate::traits::FeedbackType::Pronunciation,
                crate::traits::FeedbackType::Motivational,
            ],
            response_time_ms: 100,
        }
    }
}

#[async_trait]
impl AdaptiveLearner for AdaptiveFeedbackEngine {
    async fn learn_from_interaction(
        &mut self,
        interaction: &UserInteraction,
    ) -> crate::FeedbackResult<()> {
        // Extract performance data from the interaction
        let performance = PerformanceData {
            quality_scores: vec![interaction.feedback.overall_score],
            pronunciation_scores: vec![interaction.feedback.overall_score],
            improvement_trends: HashMap::new(),
            learning_velocity: 0.1,
            consistency: 0.8,
        };

        self.update_user_model(&interaction.user_id, &performance)
            .await
            .map_err(|e| crate::VoirsError::synthesis_failed("Adaptive update failed", e))
    }

    async fn adapt_feedback(
        &self,
        base_feedback: &FeedbackResponse,
        user_state: &AdaptiveState,
    ) -> crate::FeedbackResult<FeedbackResponse> {
        // Simple adaptation - in practice this would be more sophisticated
        let mut adapted_feedback = base_feedback.clone();
        adapted_feedback.overall_score *= user_state.confidence;
        Ok(adapted_feedback)
    }

    async fn update_user_model(
        &mut self,
        user_id: &str,
        performance_data: &PerformanceData,
    ) -> crate::FeedbackResult<()> {
        // Create a simple interaction for this update
        let interaction = UserInteraction {
            user_id: user_id.to_string(),
            timestamp: Utc::now(),
            interaction_type: InteractionType::Practice,
            audio: voirs_sdk::AudioBuffer::new(vec![0.0; 1000], 16000, 1),
            text: "".to_string(),
            feedback: FeedbackResponse {
                feedback_items: Vec::new(),
                overall_score: performance_data
                    .quality_scores
                    .get(0)
                    .copied()
                    .unwrap_or(0.5),
                immediate_actions: Vec::new(),
                long_term_goals: Vec::new(),
                progress_indicators: ProgressIndicators {
                    improving_areas: Vec::new(),
                    attention_areas: Vec::new(),
                    stable_areas: Vec::new(),
                    overall_trend: 0.1,
                    completion_percentage: 50.0,
                },
                timestamp: Utc::now(),
                processing_time: Duration::from_millis(100),
                feedback_type: crate::FeedbackType::Quality,
            },
            user_response: None,
        };

        // Direct update to user model
        let mut models = self.user_models.write().unwrap();
        if let Some(model) = models.get_mut(user_id) {
            // Update skill level based on recent performance
            if let Some(latest_quality) = performance_data.quality_scores.last() {
                model.skill_level = (model.skill_level + latest_quality) / 2.0;
            }
            model.consistency_score = performance_data.consistency;
            model.learning_rate = performance_data.learning_velocity;
            model.last_updated = Utc::now();
        }
        Ok(())
    }

    async fn get_skill_estimate(
        &self,
        user_id: &str,
    ) -> crate::FeedbackResult<crate::traits::SkillEstimate> {
        let user_model = self
            .get_user_model(user_id)
            .await
            .map_err(|e| crate::VoirsError::synthesis_failed("User model retrieval failed", e))?;

        Ok(crate::traits::SkillEstimate {
            overall_skill: user_model.skill_level,
            area_skills: user_model.skill_breakdown.clone(),
            confidence: user_model.confidence,
            prediction_accuracy: user_model.confidence, // Use confidence as accuracy proxy
        })
    }

    async fn get_learning_recommendations(
        &self,
        user_id: &str,
    ) -> crate::FeedbackResult<Vec<LearningRecommendation>> {
        // Get user model to understand current state
        let user_model = {
            let models = self.user_models.read().unwrap();
            models.get(user_id).cloned().unwrap_or_default()
        };

        let mut recommendations = Vec::new();

        // Analyze user's weakest areas and generate targeted recommendations
        // Use available fields from UserModel and derive scores
        let pronunciation_score = user_model
            .skill_breakdown
            .get(&FocusArea::Pronunciation)
            .copied()
            .unwrap_or(0.5);
        let fluency_score = user_model
            .skill_breakdown
            .get(&FocusArea::Fluency)
            .copied()
            .unwrap_or(0.5);
        let quality_score = user_model.consistency_score; // Use consistency as proxy for quality
        let rhythm_score = user_model.learning_rate; // Use learning rate as proxy for rhythm

        let skill_areas = [
            (FocusArea::Pronunciation, pronunciation_score),
            (FocusArea::Fluency, fluency_score),
            (FocusArea::Quality, quality_score),
            (FocusArea::Rhythm, rhythm_score),
        ];

        // Sort by lowest scores to prioritize improvement areas
        let mut sorted_areas = skill_areas.to_vec();
        sorted_areas.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Generate recommendations for the 3 lowest scoring areas
        for (focus_area, score) in sorted_areas.iter().take(3) {
            let priority = 1.0 - score; // Higher priority for lower scores

            let (exercises, expected_improvement, time_investment) = match focus_area {
                FocusArea::Pronunciation => {
                    let exercises = if *score < 0.4 {
                        vec![
                            "Basic phoneme recognition".to_string(),
                            "Vowel sound practice".to_string(),
                            "Consonant articulation drills".to_string(),
                        ]
                    } else if *score < 0.7 {
                        vec![
                            "Advanced phoneme combinations".to_string(),
                            "Word stress patterns".to_string(),
                            "Minimal pair exercises".to_string(),
                        ]
                    } else {
                        vec![
                            "Accent reduction exercises".to_string(),
                            "Native speaker shadowing".to_string(),
                            "Complex pronunciation patterns".to_string(),
                        ]
                    };
                    (exercises, 0.15, Duration::from_secs(20 * 60)) // 20 minutes
                }
                FocusArea::Fluency => {
                    let exercises = if *score < 0.4 {
                        vec![
                            "Short phrase repetition".to_string(),
                            "Sentence building exercises".to_string(),
                            "Basic conversation starters".to_string(),
                        ]
                    } else if *score < 0.7 {
                        vec![
                            "Storytelling practice".to_string(),
                            "Impromptu speaking exercises".to_string(),
                            "Speed reading drills".to_string(),
                        ]
                    } else {
                        vec![
                            "Debate and discussion practice".to_string(),
                            "Presentation skills training".to_string(),
                            "Spontaneous conversation".to_string(),
                        ]
                    };
                    (exercises, 0.12, Duration::from_secs(25 * 60)) // 25 minutes
                }
                FocusArea::Quality => {
                    let exercises = vec![
                        "Microphone positioning practice".to_string(),
                        "Environment optimization".to_string(),
                        "Recording technique improvement".to_string(),
                        "Audio clarity exercises".to_string(),
                    ];
                    (exercises, 0.20, Duration::from_secs(15 * 60)) // 15 minutes
                }
                FocusArea::Rhythm => {
                    let exercises = vec![
                        "Metronome-based practice".to_string(),
                        "Natural pacing exercises".to_string(),
                        "Pause and emphasis training".to_string(),
                        "Rhythm pattern recognition".to_string(),
                    ];
                    (exercises, 0.18, Duration::from_secs(30 * 60)) // 30 minutes
                }
                FocusArea::Naturalness => {
                    let exercises = vec![
                        "Natural speech patterns".to_string(),
                        "Conversational flow practice".to_string(),
                        "Spontaneous speaking exercises".to_string(),
                    ];
                    (exercises, 0.15, Duration::from_secs(25 * 60)) // 25 minutes
                }
                FocusArea::Stress => {
                    let exercises = vec![
                        "Word stress identification".to_string(),
                        "Sentence stress patterns".to_string(),
                        "Stress timing practice".to_string(),
                    ];
                    (exercises, 0.16, Duration::from_secs(20 * 60)) // 20 minutes
                }
                FocusArea::Intonation => {
                    let exercises = vec![
                        "Rising and falling patterns".to_string(),
                        "Question vs statement intonation".to_string(),
                        "Emotional intonation practice".to_string(),
                    ];
                    (exercises, 0.17, Duration::from_secs(25 * 60)) // 25 minutes
                }
                FocusArea::Breathing => {
                    let exercises = vec![
                        "Diaphragmatic breathing exercises".to_string(),
                        "Breath support training".to_string(),
                        "Sustained phonation practice".to_string(),
                    ];
                    (exercises, 0.14, Duration::from_secs(15 * 60)) // 15 minutes
                }
                FocusArea::Clarity => {
                    let exercises = vec![
                        "Articulation clarity drills".to_string(),
                        "Precise consonant practice".to_string(),
                        "Clear vowel production".to_string(),
                    ];
                    (exercises, 0.15, Duration::from_secs(20 * 60)) // 20 minutes
                }
                FocusArea::Accuracy => {
                    let exercises = vec![
                        "Precision pronunciation exercises".to_string(),
                        "Accurate phoneme targeting".to_string(),
                        "Error correction practice".to_string(),
                    ];
                    (exercises, 0.16, Duration::from_secs(25 * 60)) // 25 minutes
                }
                FocusArea::Consistency => {
                    let exercises = vec![
                        "Consistent rhythm practice".to_string(),
                        "Stable pronunciation drills".to_string(),
                        "Uniform speech patterns".to_string(),
                    ];
                    (exercises, 0.14, Duration::from_secs(22 * 60)) // 22 minutes
                }
            };

            recommendations.push(LearningRecommendation {
                focus_area: focus_area.clone(),
                priority,
                exercises,
                expected_improvement,
                time_investment,
            });
        }

        // Add a bonus recommendation based on user's learning rate
        if user_model.learning_rate > 0.7 {
            // Fast learner - give advanced challenges
            recommendations.push(LearningRecommendation {
                focus_area: FocusArea::Pronunciation,
                priority: 0.6,
                exercises: vec![
                    "Advanced accent training".to_string(),
                    "Professional speech patterns".to_string(),
                    "Native-like intonation".to_string(),
                ],
                expected_improvement: 0.10,
                time_investment: Duration::from_secs(45 * 60), // 45 minutes
            });
        } else if user_model.learning_rate < 0.3 {
            // Slower learner - provide foundational support
            recommendations.push(LearningRecommendation {
                focus_area: FocusArea::Quality,
                priority: 0.8,
                exercises: vec![
                    "Basic recording setup".to_string(),
                    "Simple speech exercises".to_string(),
                    "Confidence building activities".to_string(),
                ],
                expected_improvement: 0.25,
                time_investment: Duration::from_secs(10 * 60), // 10 minutes
            });
        }

        // Sort recommendations by priority (highest first)
        recommendations.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        Ok(recommendations)
    }
}

impl LearningAlgorithm {
    /// Predict feedback strategy based on features
    pub fn predict_strategy(
        &self,
        features: &FeatureVector,
    ) -> Result<FeedbackStrategy, FeedbackError> {
        // Simple rule-based strategy prediction
        let avg_feature = features.features.iter().sum::<f32>() / features.features.len() as f32;

        let strategy_type = if avg_feature < 0.3 {
            StrategyType::Encouraging
        } else if avg_feature > 0.7 {
            StrategyType::Technical
        } else {
            StrategyType::Adaptive
        };

        Ok(FeedbackStrategy {
            strategy_type,
            tone: FeedbackTone::Neutral,
            detail_level: avg_feature,
            personalization_factors: features.feature_names.clone(),
        })
    }
}
