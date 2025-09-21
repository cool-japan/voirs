//! Interactive training system for speech synthesis
//!
//! This module provides a comprehensive training system with structured exercises,
//! guided learning paths, and adaptive difficulty progression. The system includes:
//!
//! - **Core functionality**: Main trainer implementation with spaced repetition
//! - **Exercise library**: 500+ exercises across all skill levels and focus areas
//! - **Session management**: Training session lifecycle and exercise management
//! - **Flow optimization**: Intelligent session management with fatigue detection
//! - **Results and feedback**: Comprehensive evaluation and improvement suggestions
//! - **Types and structures**: All data types used throughout the training system

pub mod core;
pub mod exercises;
pub mod flow;
pub mod results;
pub mod sessions;
pub mod types;

// Re-export main types and functionality
pub use core::InteractiveTrainer;
pub use types::*;

// Re-export key functionality from submodules
pub use types::{ExerciseLibrary, SessionFlowOptimizer};

// Import trait implementation
use crate::traits::{
    FeedbackResponse, FeedbackResult, ProgressIndicators, TrainingProvider, UserFeedback,
};
use crate::{FeedbackError, VoirsError};
use async_trait::async_trait;
use chrono::Utc;
use std::time::Duration;

/// Implementation of TrainingProvider trait for InteractiveTrainer
#[async_trait]
impl TrainingProvider for InteractiveTrainer {
    async fn get_exercises(
        &self,
        user_id: &str,
        skill_level: f32,
    ) -> FeedbackResult<Vec<crate::traits::TrainingExercise>> {
        let focus_areas = vec![
            crate::traits::FocusArea::Pronunciation,
            crate::traits::FocusArea::Quality,
        ];
        core::InteractiveTrainer::get_recommended_exercises(
            self,
            user_id,
            skill_level,
            &focus_areas,
        )
        .await
        .map_err(VoirsError::from)
    }

    async fn get_recommended_exercises(
        &self,
        user_id: &str,
    ) -> FeedbackResult<Vec<crate::traits::TrainingExercise>> {
        // Use default skill level and focus areas for simplified interface
        let skill_level = 0.5;
        let focus_areas = vec![
            crate::traits::FocusArea::Pronunciation,
            crate::traits::FocusArea::Quality,
        ];

        core::InteractiveTrainer::get_recommended_exercises(
            self,
            user_id,
            skill_level,
            &focus_areas,
        )
        .await
        .map_err(VoirsError::from)
    }

    async fn evaluate_exercise(
        &self,
        exercise: &crate::traits::TrainingExercise,
        audio: &voirs_sdk::AudioBuffer,
    ) -> FeedbackResult<crate::traits::TrainingResult> {
        // Create a temporary session for evaluation
        let session = self
            .start_session("eval_user", None)
            .await
            .map_err(VoirsError::from)?;

        // Start the exercise
        let _ = self
            .start_exercise(&session.session_id, &exercise.exercise_id)
            .await
            .map_err(VoirsError::from)?;

        // Submit the attempt
        let result = self
            .submit_attempt(&session.session_id, audio)
            .await
            .map_err(VoirsError::from)?;

        // Clean up the temporary session
        let _ = self.complete_session(&session.session_id).await;

        // Convert to TrainingResult format
        Ok(crate::traits::TrainingResult {
            exercise: exercise.clone(),
            success: result.success,
            attempts_made: 1,
            completion_time: result.attempt.evaluation_time,
            final_scores: crate::traits::TrainingScores {
                quality: result.attempt.quality_score,
                pronunciation: result.attempt.pronunciation_score,
                consistency: 0.8, // Default value
                improvement: 0.8, // Default value
            },
            feedback: FeedbackResponse {
                feedback_items: vec![UserFeedback {
                    message: if result.success {
                        "Exercise completed successfully".to_string()
                    } else {
                        "Exercise needs more practice".to_string()
                    },
                    suggestion: Some("Continue practicing to improve your skills".to_string()),
                    confidence: 0.9,
                    score: result.attempt.feedback.overall_score,
                    priority: 0.8,
                    metadata: std::collections::HashMap::new(),
                }],
                overall_score: result.attempt.feedback.overall_score,
                immediate_actions: result.next_steps,
                long_term_goals: result
                    .improvement_suggestions
                    .iter()
                    .map(|s| s.area.clone())
                    .collect(),
                progress_indicators: ProgressIndicators {
                    improving_areas: vec!["Pronunciation clarity".to_string()],
                    attention_areas: vec!["Rhythm consistency".to_string()],
                    stable_areas: vec!["Basic quality".to_string()],
                    overall_trend: 0.8,
                    completion_percentage: 75.0,
                },
                timestamp: Utc::now(),
                processing_time: result.attempt.evaluation_time,
                feedback_type: crate::traits::FeedbackType::Technical,
            },
            improvement_recommendations: result
                .improvement_suggestions
                .iter()
                .map(|s| s.area.clone())
                .collect(),
        })
    }

    async fn create_custom_exercise(
        &self,
        specification: &crate::traits::ExerciseSpecification,
    ) -> FeedbackResult<crate::traits::TrainingExercise> {
        use uuid::Uuid;

        Ok(crate::traits::TrainingExercise {
            exercise_id: Uuid::new_v4().to_string(),
            name: format!(
                "Custom {} Exercise",
                match specification.exercise_type {
                    crate::traits::ExerciseType::Pronunciation => "Pronunciation",
                    crate::traits::ExerciseType::Quality => "Quality",
                    crate::traits::ExerciseType::FreeForm => "Free Form",
                    crate::traits::ExerciseType::Rhythm => "Rhythm",
                    crate::traits::ExerciseType::Expression => "Expression",
                    crate::traits::ExerciseType::Fluency => "Fluency",
                    crate::traits::ExerciseType::Advanced => "Advanced",
                    crate::traits::ExerciseType::Challenge => "Challenge",
                    crate::traits::ExerciseType::Review => "Review",
                }
            ),
            description: "Custom exercise created from user specification".to_string(),
            difficulty: specification.difficulty,
            focus_areas: specification.focus_areas.clone(),
            exercise_type: specification.exercise_type.clone(),
            target_text: specification
                .custom_text
                .clone()
                .unwrap_or_else(|| "Practice text for custom exercise".to_string()),
            reference_audio: None,
            success_criteria: crate::traits::SuccessCriteria {
                min_quality_score: 0.7,
                min_pronunciation_score: 0.75,
                max_attempts: 3,
                time_limit: specification.duration_constraint,
                consistency_required: 1,
            },
            estimated_duration: specification
                .duration_constraint
                .unwrap_or_else(|| std::time::Duration::from_secs(300)),
        })
    }

    fn get_categories(&self) -> Vec<crate::traits::ExerciseCategory> {
        let library = self.exercise_library.read().unwrap();
        library.categories.clone()
    }
}
