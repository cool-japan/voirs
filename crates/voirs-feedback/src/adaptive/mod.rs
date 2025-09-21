//! Adaptive feedback and learning system modules
//!
//! This module provides machine learning-driven personalization of feedback
//! based on user behavior, progress patterns, and learning preferences.
//!
//! The implementation has been modularized into several components:
//! - `core`: Main adaptive feedback engine
//! - `models`: Data structures and user models  
//! - `types`: Type definitions and enums

pub mod core;
pub mod models;
pub mod types;

// Re-export the main engine and key types for backwards compatibility
pub use core::AdaptiveFeedbackEngine;
pub use models::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{
        FeedbackContext, FeedbackResponse, FeedbackType, FocusArea, InteractionType,
        PerformanceData, ProgressIndicators, UserInteraction, UserResponse,
    };
    use crate::{SessionState, UserFeedback, UserPreferences};
    use chrono::Utc;
    use std::collections::HashMap;
    use std::time::Duration;
    use voirs_sdk::AudioBuffer;

    #[tokio::test]
    async fn test_adaptive_engine_creation() {
        let engine = AdaptiveFeedbackEngine::new().await.unwrap();
        let stats = engine.get_statistics().await.unwrap();
        assert_eq!(stats.total_users, 0);
        assert_eq!(stats.total_interactions, 0);
    }

    #[tokio::test]
    async fn test_user_model_creation() {
        let engine = AdaptiveFeedbackEngine::new().await.unwrap();
        let model = engine.get_user_model("test_user").await.unwrap();

        assert_eq!(model.user_id, "test_user");
        assert_eq!(model.skill_level, 0.5);
        assert!(model
            .skill_breakdown
            .contains_key(&FocusArea::Pronunciation));
    }

    #[tokio::test]
    async fn test_feedback_strategy_prediction() {
        let engine = AdaptiveFeedbackEngine::new().await.unwrap();
        let session = SessionState::new("test_user").await.unwrap();

        let context = FeedbackContext {
            session,
            exercise: None,
            history: Vec::new(),
            preferences: UserPreferences::default(),
        };

        let strategy = engine
            .predict_feedback_strategy("test_user", &context)
            .await
            .unwrap();
        assert!(matches!(
            strategy.strategy_type,
            StrategyType::Encouraging
                | StrategyType::Direct
                | StrategyType::Technical
                | StrategyType::Adaptive
        ));
        assert!(strategy.detail_level >= 0.0 && strategy.detail_level <= 1.0);
    }

    #[tokio::test]
    async fn test_model_update() {
        let engine = AdaptiveFeedbackEngine::new().await.unwrap();

        let interaction = UserInteraction {
            user_id: "test_user".to_string(),
            timestamp: Utc::now(),
            interaction_type: InteractionType::Practice,
            audio: AudioBuffer::new(vec![0.1; 1000], 16000, 1),
            text: "Hello world".to_string(),
            feedback: FeedbackResponse {
                feedback_items: Vec::new(),
                overall_score: 0.8,
                immediate_actions: Vec::new(),
                long_term_goals: Vec::new(),
                progress_indicators: ProgressIndicators {
                    improving_areas: Vec::new(),
                    attention_areas: Vec::new(),
                    stable_areas: Vec::new(),
                    overall_trend: 0.1,
                    completion_percentage: 80.0,
                },
                timestamp: Utc::now(),
                processing_time: Duration::from_millis(100),
                feedback_type: FeedbackType::Quality,
            },
            user_response: Some(UserResponse::Helpful),
        };

        let performance = PerformanceData {
            quality_scores: vec![0.8, 0.7, 0.9],
            pronunciation_scores: vec![0.7, 0.8, 0.8],
            improvement_trends: {
                let mut trends = HashMap::new();
                trends.insert(FocusArea::Pronunciation, 0.1);
                trends
            },
            learning_velocity: 0.2,
            consistency: 0.8,
        };

        let _user_model = engine.get_user_model("test_user").await.unwrap();

        engine
            .update_user_model("test_user", &interaction, &performance)
            .await
            .unwrap();

        let updated_model = engine.get_user_model("test_user").await.unwrap();
        assert_eq!(updated_model.interaction_history.len(), 1);
        assert_eq!(updated_model.performance_history.len(), 1);
        assert!(updated_model.confidence >= 0.1);
    }

    #[tokio::test]
    async fn test_personalized_recommendations() {
        let engine = AdaptiveFeedbackEngine::new().await.unwrap();

        let interaction = UserInteraction {
            user_id: "test_user".to_string(),
            timestamp: Utc::now(),
            interaction_type: InteractionType::Practice,
            audio: AudioBuffer::new(vec![0.1; 1000], 16000, 1),
            text: "Hello".to_string(),
            feedback: FeedbackResponse {
                feedback_items: Vec::new(),
                overall_score: 0.6,
                immediate_actions: Vec::new(),
                long_term_goals: Vec::new(),
                progress_indicators: ProgressIndicators {
                    improving_areas: Vec::new(),
                    attention_areas: Vec::new(),
                    stable_areas: Vec::new(),
                    overall_trend: 0.0,
                    completion_percentage: 60.0,
                },
                timestamp: Utc::now(),
                processing_time: Duration::from_millis(100),
                feedback_type: FeedbackType::Quality,
            },
            user_response: None,
        };

        let performance = PerformanceData {
            quality_scores: vec![0.6],
            pronunciation_scores: vec![0.5],
            improvement_trends: {
                let mut trends = HashMap::new();
                trends.insert(FocusArea::Pronunciation, 0.4);
                trends
            },
            learning_velocity: 0.1,
            consistency: 0.6,
        };

        let _user_model = engine.get_user_model("test_user").await.unwrap();

        engine
            .update_user_model("test_user", &interaction, &performance)
            .await
            .unwrap();

        let base_feedback = FeedbackResponse {
            feedback_items: vec![UserFeedback {
                message: "Test feedback".to_string(),
                suggestion: Some("Test suggestion".to_string()),
                confidence: 0.8,
                score: 0.6,
                priority: 0.7,
                metadata: HashMap::new(),
            }],
            overall_score: 0.6,
            immediate_actions: Vec::new(),
            long_term_goals: Vec::new(),
            progress_indicators: ProgressIndicators {
                improving_areas: Vec::new(),
                attention_areas: Vec::new(),
                stable_areas: Vec::new(),
                overall_trend: 0.0,
                completion_percentage: 60.0,
            },
            timestamp: Utc::now(),
            processing_time: Duration::from_millis(100),
            feedback_type: FeedbackType::Quality,
        };

        let recommendations = engine
            .generate_personalized_recommendations("test_user", &base_feedback)
            .await
            .unwrap();
        assert!(!recommendations.is_empty());

        for recommendation in &recommendations {
            assert!(recommendation.priority >= 0.0 && recommendation.priority <= 1.0);
            assert!(
                recommendation.estimated_impact >= 0.0 && recommendation.estimated_impact <= 1.0
            );
            assert!(!recommendation.explanation.is_empty());
        }
    }
}
