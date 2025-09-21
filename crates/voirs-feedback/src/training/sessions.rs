//! Session management functionality
//!
//! This module contains methods for managing training sessions,
//! including starting sessions, managing exercises, and handling attempts.

use crate::training::core::InteractiveTrainer;
use crate::training::types::{
    AttemptFeedback, AttemptResult, ExerciseAttempt, ExerciseFeedback, ExerciseResult,
    ExerciseScores, ExerciseSession, ExerciseSessionStatus, LearningPath, SessionScores,
    SessionStatistics, TrainingSession, TrainingSessionConfig, TrainingSessionResult,
    TrainingSessionStatus,
};
use crate::traits::{ExerciseType, FocusArea, SuccessCriteria, TrainingExercise};
use crate::FeedbackError;
use chrono::Utc;
use std::time::{Duration, Instant};
use uuid::Uuid;
use voirs_evaluation::traits::{PronunciationEvaluator, QualityEvaluator};
use voirs_sdk::AudioBuffer;

impl InteractiveTrainer {
    /// Start a new training session
    pub async fn start_session(
        &self,
        user_id: &str,
        session_config: Option<TrainingSessionConfig>,
    ) -> Result<TrainingSession, FeedbackError> {
        let config = session_config.unwrap_or_default();
        let session_id = Uuid::new_v4().to_string();

        let session = TrainingSession {
            session_id: session_id.clone(),
            user_id: user_id.to_string(),
            config,
            current_exercise: None,
            completed_exercises: Vec::new(),
            statistics: SessionStatistics::default(),
            start_time: Utc::now(),
            status: TrainingSessionStatus::Active,
        };

        // Store session
        {
            let mut sessions = self.active_sessions.write().unwrap();
            sessions.insert(session_id.clone(), session.clone());
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_sessions += 1;
            metrics.active_sessions += 1;
        }

        Ok(session)
    }

    /// Start an exercise in a training session
    pub async fn start_exercise(
        &self,
        session_id: &str,
        exercise_id: &str,
    ) -> Result<ExerciseSession, FeedbackError> {
        let mut sessions = self.active_sessions.write().unwrap();
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| FeedbackError::TrainingError {
                message: format!("Training session not found: {session_id}"),
                source: None,
            })?;

        // Get exercise from library
        let library = self.exercise_library.read().unwrap();
        let exercise = library
            .exercises
            .iter()
            .find(|e| e.exercise_id == exercise_id)
            .ok_or_else(|| FeedbackError::TrainingError {
                message: format!("Exercise not found: {exercise_id}"),
                source: None,
            })?
            .clone();

        // Create exercise session
        let exercise_session = ExerciseSession {
            exercise: exercise.clone(),
            attempts: Vec::new(),
            start_time: Utc::now(),
            status: ExerciseSessionStatus::InProgress,
            feedback_history: Vec::new(),
            current_attempt: 0,
        };

        session.current_exercise = Some(exercise_session.clone());

        Ok(exercise_session)
    }

    /// Submit an attempt for evaluation
    pub async fn submit_attempt(
        &self,
        session_id: &str,
        audio: &AudioBuffer,
    ) -> Result<AttemptResult, FeedbackError> {
        // Extract needed data without holding the lock across await
        let (exercise, attempt_number) = {
            let sessions = self.active_sessions.read().unwrap();
            let session = sessions
                .get(session_id)
                .ok_or_else(|| FeedbackError::TrainingError {
                    message: format!("Training session not found: {session_id}"),
                    source: None,
                })?;

            let exercise_session =
                session
                    .current_exercise
                    .as_ref()
                    .ok_or_else(|| FeedbackError::TrainingError {
                        message: "No active exercise in session".to_string(),
                        source: None,
                    })?;

            let exercise = exercise_session.exercise.clone();
            let attempt_number = exercise_session.attempts.len() + 1;
            (exercise, attempt_number)
        };

        // Evaluate the attempt (no locks held here)
        let evaluation_start = Instant::now();

        // Evaluate quality with fallback for PESQ
        let quality_score = match self
            .quality_evaluator
            .evaluate_quality(audio, None, None)
            .await
        {
            Ok(score) => score,
            Err(e) => {
                if e.to_string().contains("PESQ requires reference audio") {
                    voirs_evaluation::QualityScore {
                        overall_score: 0.75,
                        component_scores: {
                            let mut scores = std::collections::HashMap::new();
                            scores.insert("spectral_quality".to_string(), 0.75);
                            scores.insert("temporal_consistency".to_string(), 0.8);
                            scores.insert("snr_estimate".to_string(), 0.7);
                            scores
                        },
                        recommendations: vec![
                            "Fallback score - PESQ unavailable without reference audio".to_string(),
                        ],
                        confidence: 0.5,
                        processing_time: None,
                    }
                } else {
                    return Err(FeedbackError::TrainingError {
                        message: format!("Quality evaluation failed: {e}"),
                        source: Some(Box::new(e)),
                    });
                }
            }
        };

        let pronunciation_score = self
            .pronunciation_evaluator
            .evaluate_pronunciation(audio, &exercise.target_text, None)
            .await;

        let pronunciation_score =
            pronunciation_score.map_err(|e| FeedbackError::TrainingError {
                message: format!("Pronunciation evaluation failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let evaluation_time = evaluation_start.elapsed();

        // Create attempt record
        let attempt = ExerciseAttempt {
            attempt_number,
            audio: audio.clone(),
            timestamp: Utc::now(),
            quality_score: quality_score.overall_score.min(1.0).max(0.0),
            pronunciation_score: pronunciation_score.overall_score.min(1.0).max(0.0),
            evaluation_time,
            feedback: AttemptFeedback {
                overall_score: (quality_score.overall_score + pronunciation_score.overall_score)
                    / 2.0,
                quality_score: quality_score.overall_score,
                pronunciation_score: pronunciation_score.overall_score,
                strengths: quality_score.recommendations.clone(),
                weaknesses: pronunciation_score
                    .feedback
                    .iter()
                    .map(|f| f.message.clone())
                    .collect(),
                suggestions: vec![
                    "Continue practicing pronunciation".to_string(),
                    "Focus on clarity and articulation".to_string(),
                ],
                encouragement: "Keep up the good work!".to_string(),
            },
        };

        // Check success criteria
        let success = self.check_success_criteria(&attempt, &exercise.success_criteria);

        let result = AttemptResult {
            attempt: attempt.clone(),
            success,
            meets_criteria: self.analyze_criteria_compliance(&attempt, &exercise.success_criteria),
            next_steps: self.generate_next_steps(&attempt, &exercise, success),
            improvement_suggestions: self.generate_improvement_suggestions(&attempt, &exercise),
        };

        // Now update the session with the new attempt
        {
            let mut sessions = self.active_sessions.write().unwrap();
            let session =
                sessions
                    .get_mut(session_id)
                    .ok_or_else(|| FeedbackError::TrainingError {
                        message: format!("Training session not found: {session_id}"),
                        source: None,
                    })?;

            let exercise_session =
                session
                    .current_exercise
                    .as_mut()
                    .ok_or_else(|| FeedbackError::TrainingError {
                        message: "No active exercise in session".to_string(),
                        source: None,
                    })?;

            // Record attempt
            exercise_session.attempts.push(attempt.clone());
            exercise_session.current_attempt = attempt_number;
            exercise_session
                .feedback_history
                .push(result.attempt.feedback.clone());

            // Update session statistics
            session.statistics.total_attempts += 1;
            if success {
                session.statistics.successful_attempts += 1;
            }
            session.statistics.total_evaluation_time += evaluation_time;

            // Check if exercise is complete
            if success || attempt_number >= exercise.success_criteria.max_attempts {
                exercise_session.status = if success {
                    ExerciseSessionStatus::Completed
                } else {
                    ExerciseSessionStatus::Failed
                };

                // Record exercise completion in history
                let exercise_result = ExerciseResult {
                    exercise: exercise.clone(),
                    attempts: exercise_session.attempts.clone(),
                    success,
                    completion_time: Utc::now()
                        .signed_duration_since(exercise_session.start_time)
                        .to_std()
                        .unwrap_or_default(),
                    final_scores: ExerciseScores {
                        quality: exercise_session
                            .attempts
                            .iter()
                            .map(|a| a.quality_score)
                            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .unwrap_or(0.0),
                        pronunciation: exercise_session
                            .attempts
                            .iter()
                            .map(|a| a.pronunciation_score)
                            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .unwrap_or(0.0),
                        consistency: self.calculate_consistency_score(&exercise_session.attempts),
                        improvement: self.calculate_improvement_score(&exercise_session.attempts),
                    },
                    feedback: ExerciseFeedback {
                        feedback_items: vec!["Exercise completed".to_string()],
                        overall_assessment: if success {
                            "Successful completion"
                        } else {
                            "Additional practice needed"
                        }
                        .to_string(),
                        improvement_recommendations: vec!["Continue practicing".to_string()],
                        encouragement: "Well done!".to_string(),
                    },
                };

                session.completed_exercises.push(exercise_result);
                session.statistics.exercises_completed += 1;
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_attempts += 1;
            if success {
                metrics.successful_attempts += 1;
            }
            metrics.total_evaluation_time += evaluation_time;
        }

        Ok(result)
    }

    /// Complete a training session
    pub async fn complete_session(
        &self,
        session_id: &str,
    ) -> Result<TrainingSessionResult, FeedbackError> {
        // Remove session and get completed exercises without holding locks across await
        let (mut session, completed_exercises) = {
            let mut sessions = self.active_sessions.write().unwrap();
            let mut session =
                sessions
                    .remove(session_id)
                    .ok_or_else(|| FeedbackError::TrainingError {
                        message: format!("Training session not found: {session_id}"),
                        source: None,
                    })?;

            session.status = TrainingSessionStatus::Completed;
            let completed_exercises = session.completed_exercises.clone();
            (session, completed_exercises)
        };

        let completion_time = Utc::now();
        let session_duration = completion_time
            .signed_duration_since(session.start_time)
            .to_std()
            .unwrap_or_default();

        // Calculate session statistics
        let total_exercises = completed_exercises.len();
        let successful_exercises = completed_exercises.iter().filter(|r| r.success).count();

        let average_scores = if completed_exercises.is_empty() {
            SessionScores {
                average_quality: 0.0,
                average_pronunciation: 0.0,
                average_fluency: 0.0,
                overall_score: 0.0,
                improvement_trend: 0.0,
            }
        } else {
            let total_quality: f32 = completed_exercises
                .iter()
                .map(|r| r.final_scores.quality)
                .sum();
            let total_pronunciation: f32 = completed_exercises
                .iter()
                .map(|r| r.final_scores.pronunciation)
                .sum();

            SessionScores {
                average_quality: total_quality / completed_exercises.len() as f32,
                average_pronunciation: total_pronunciation / completed_exercises.len() as f32,
                average_fluency: 0.8, // Default fluency score
                overall_score: (total_quality + total_pronunciation)
                    / (2.0 * completed_exercises.len() as f32),
                improvement_trend: self.calculate_session_improvement_trend(&completed_exercises),
            }
        };

        // Process achievements and recommendations (async operations)
        let achievements = self
            .check_session_achievements(&completed_exercises)
            .await?;
        let recommendations = self.generate_session_recommendations(&completed_exercises);
        let next_learning_path = self.suggest_next_learning_path(&completed_exercises);

        let result = TrainingSessionResult {
            session,
            completion_time,
            session_duration,
            total_exercises,
            successful_exercises,
            success_rate: if total_exercises > 0 {
                successful_exercises as f32 / total_exercises as f32
            } else {
                0.0
            },
            average_scores,
            achievements,
            recommendations,
            next_learning_path,
        };

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.active_sessions -= 1;
            metrics.completed_sessions += 1;
            metrics.total_session_time += session_duration;
        }

        Ok(result)
    }
}
