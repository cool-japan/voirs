//! Core InteractiveTrainer implementation
//!
//! This module contains the main InteractiveTrainer struct and its core
//! functionality including initialization, exercise recommendations,
//! and learning optimization algorithms.

use crate::training::types::{
    CollaborativeLearningSystem, ExerciseLibrary, TrainingConfig, TrainingMetrics, TrainingSession,
    TrainingSystemStats,
};
use crate::traits::{ExerciseHistory, FocusArea, TrainingExercise};
use crate::FeedbackError;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use uuid::Uuid;
use voirs_evaluation::prelude::{PronunciationEvaluatorImpl, QualityEvaluator};

/// Interactive training system
#[derive(Clone)]
pub struct InteractiveTrainer {
    /// Exercise library
    pub(crate) exercise_library: Arc<RwLock<ExerciseLibrary>>,
    /// Quality evaluator for assessment
    pub(crate) quality_evaluator: Arc<QualityEvaluator>,
    /// Pronunciation evaluator for assessment
    pub(crate) pronunciation_evaluator: Arc<PronunciationEvaluatorImpl>,
    /// Training sessions
    pub(crate) active_sessions: Arc<RwLock<HashMap<String, TrainingSession>>>,
    /// Configuration
    pub(crate) config: TrainingConfig,
    /// System metrics
    pub(crate) metrics: Arc<RwLock<TrainingMetrics>>,
    /// Collaborative learning system
    pub(crate) collaborative_learning: Arc<CollaborativeLearningSystem>,
}

impl InteractiveTrainer {
    /// Create a new interactive trainer
    pub async fn new() -> Result<Self, FeedbackError> {
        Self::with_config(TrainingConfig::default()).await
    }

    /// Create with custom configuration
    pub async fn with_config(config: TrainingConfig) -> Result<Self, FeedbackError> {
        let quality_evaluator =
            Arc::new(
                QualityEvaluator::new()
                    .await
                    .map_err(|e| FeedbackError::TrainingError {
                        message: format!("Failed to initialize quality evaluator: {e}"),
                        source: Some(Box::new(e)),
                    })?,
            );

        let pronunciation_evaluator =
            Arc::new(PronunciationEvaluatorImpl::new().await.map_err(|e| {
                FeedbackError::TrainingError {
                    message: format!("Failed to initialize pronunciation evaluator: {e}"),
                    source: Some(Box::new(e)),
                }
            })?);

        let exercise_library = Arc::new(RwLock::new(ExerciseLibrary::create_default()));
        let collaborative_learning = Arc::new(CollaborativeLearningSystem::new().await?);

        Ok(Self {
            exercise_library,
            quality_evaluator,
            pronunciation_evaluator,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
            metrics: Arc::new(RwLock::new(TrainingMetrics::default())),
            collaborative_learning,
        })
    }

    /// Get exercises for user based on skill level and preferences with spaced repetition
    pub async fn get_recommended_exercises(
        &self,
        user_id: &str,
        skill_level: f32,
        focus_areas: &[FocusArea],
    ) -> Result<Vec<TrainingExercise>, FeedbackError> {
        // Extract exercises from library before async operations
        let exercises = {
            let library = self.exercise_library.read().unwrap();
            library.exercises.clone()
        };

        let mut recommended = Vec::new();

        // Get user's exercise history for spaced repetition
        let exercise_history = self.get_user_exercise_history(user_id).await?;

        // Filter exercises by skill level and focus areas
        for exercise in &exercises {
            // Check skill level compatibility
            let skill_match = (exercise.difficulty - skill_level).abs() <= 0.3;

            // Check focus area overlap
            let focus_match = exercise
                .focus_areas
                .iter()
                .any(|area| focus_areas.contains(area));

            if skill_match && focus_match {
                recommended.push(exercise.clone());
            }
        }

        // Apply spaced repetition algorithm
        let spaced_exercises = self.apply_spaced_repetition(&recommended, &exercise_history);

        // Sort by spaced repetition priority and relevance
        let mut final_recommendations = spaced_exercises;
        final_recommendations.sort_by(|a, b| {
            let a_priority = self.calculate_spaced_repetition_priority(a, &exercise_history);
            let b_priority = self.calculate_spaced_repetition_priority(b, &exercise_history);
            b_priority
                .partial_cmp(&a_priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to reasonable number
        final_recommendations.truncate(self.config.max_recommended_exercises);

        Ok(final_recommendations)
    }

    /// Get user's exercise history for spaced repetition calculations
    async fn get_user_exercise_history(
        &self,
        user_id: &str,
    ) -> Result<HashMap<String, ExerciseHistory>, FeedbackError> {
        // In a real implementation, this would query a database
        // For now, return an empty history
        let _ = user_id; // Suppress unused parameter warning
        Ok(HashMap::new())
    }

    /// Apply spaced repetition algorithm to exercise selection
    fn apply_spaced_repetition(
        &self,
        exercises: &[TrainingExercise],
        history: &HashMap<String, ExerciseHistory>,
    ) -> Vec<TrainingExercise> {
        let mut spaced_exercises = Vec::new();
        let now = Utc::now();

        for exercise in exercises {
            if let Some(exercise_hist) = history.get(&exercise.exercise_id) {
                // Calculate if exercise is due for review based on spaced repetition
                let time_since_last = now
                    .signed_duration_since(exercise_hist.last_attempted)
                    .to_std()
                    .unwrap_or_default();

                let due_interval = self.calculate_spaced_repetition_interval(
                    exercise_hist.repetition_number,
                    exercise_hist.ease_factor,
                    exercise_hist.last_performance,
                );

                // Include exercise if it's due for review or new
                if time_since_last >= due_interval || exercise_hist.repetition_number == 0 {
                    spaced_exercises.push(exercise.clone());
                }
            } else {
                // New exercise - always include
                spaced_exercises.push(exercise.clone());
            }
        }

        spaced_exercises
    }

    /// Calculate spaced repetition interval based on SM-2 algorithm
    fn calculate_spaced_repetition_interval(
        &self,
        repetition_number: u32,
        ease_factor: f32,
        last_performance: f32,
    ) -> Duration {
        let base_interval = match repetition_number {
            0 => Duration::from_secs(86400),  // 1 day
            1 => Duration::from_secs(259200), // 3 days
            _ => {
                // Calculate interval: previous_interval * ease_factor
                let days = match repetition_number {
                    2 => 6.0,
                    3 => (6.0 * ease_factor).max(1.0),
                    _ => {
                        let mut interval = 6.0;
                        for _ in 3..repetition_number {
                            interval *= ease_factor;
                        }
                        interval.max(1.0)
                    }
                };
                Duration::from_secs((days * 86400.0) as u64)
            }
        };

        // Adjust based on last performance
        let performance_multiplier = if last_performance >= 0.8 {
            1.0 // Good performance - keep interval
        } else if last_performance >= 0.6 {
            0.8 // Moderate performance - slightly reduce interval
        } else {
            0.5 // Poor performance - significantly reduce interval
        };

        Duration::from_secs((base_interval.as_secs() as f32 * performance_multiplier) as u64)
    }

    /// Calculate spaced repetition priority for exercise ordering
    fn calculate_spaced_repetition_priority(
        &self,
        exercise: &TrainingExercise,
        history: &HashMap<String, ExerciseHistory>,
    ) -> f32 {
        if let Some(exercise_hist) = history.get(&exercise.exercise_id) {
            let now = Utc::now();
            let time_since_last = now
                .signed_duration_since(exercise_hist.last_attempted)
                .to_std()
                .unwrap_or_default();

            let due_interval = self.calculate_spaced_repetition_interval(
                exercise_hist.repetition_number,
                exercise_hist.ease_factor,
                exercise_hist.last_performance,
            );

            // Priority increases as exercise becomes more overdue
            let overdue_ratio = time_since_last.as_secs() as f32 / due_interval.as_secs() as f32;

            // Boost priority for exercises with poor past performance
            let performance_boost = if exercise_hist.last_performance < 0.7 {
                2.0
            } else if exercise_hist.last_performance < 0.8 {
                1.5
            } else {
                1.0
            };

            // Consider forgetting curve - exercises with higher repetition numbers get lower priority
            let repetition_penalty = 1.0 / (1.0 + exercise_hist.repetition_number as f32 * 0.1);

            overdue_ratio * performance_boost * repetition_penalty
        } else {
            // New exercises get high priority
            3.0
        }
    }

    /// Update exercise history after completion
    pub async fn update_exercise_history(
        &self,
        user_id: &str,
        exercise_id: &str,
        performance_score: f32,
    ) -> Result<(), FeedbackError> {
        // In a real implementation, this would update the database
        // For now, we'll just log the update
        println!(
            "Updating exercise history for user {user_id} exercise {exercise_id} with score {performance_score}"
        );
        Ok(())
    }

    /// Get optimized learning sequence using spaced repetition and interleaving
    pub async fn get_optimized_learning_sequence(
        &self,
        user_id: &str,
        skill_level: f32,
        focus_areas: &[FocusArea],
        session_duration: Duration,
    ) -> Result<Vec<TrainingExercise>, FeedbackError> {
        let available_exercises = self
            .get_recommended_exercises(user_id, skill_level, focus_areas)
            .await?;

        if available_exercises.is_empty() {
            return Ok(Vec::new());
        }

        let estimated_time_per_exercise = Duration::from_secs(300); // 5 minutes average
        let max_exercises =
            (session_duration.as_secs() / estimated_time_per_exercise.as_secs()) as usize;

        // Apply interleaving strategy - mix different focus areas and difficulty levels
        let interleaved_sequence =
            self.apply_interleaving_strategy(&available_exercises, max_exercises);

        // Apply forgetting curve optimization
        let optimized_sequence = self
            .optimize_for_forgetting_curve(&interleaved_sequence, user_id)
            .await?;

        Ok(optimized_sequence)
    }

    /// Apply interleaving strategy to mix different types of exercises
    fn apply_interleaving_strategy(
        &self,
        exercises: &[TrainingExercise],
        max_exercises: usize,
    ) -> Vec<TrainingExercise> {
        let mut interleaved = Vec::new();
        let mut by_focus_area: HashMap<FocusArea, Vec<&TrainingExercise>> = HashMap::new();

        // Group exercises by focus area
        for exercise in exercises {
            for focus_area in &exercise.focus_areas {
                by_focus_area
                    .entry(focus_area.clone())
                    .or_default()
                    .push(exercise);
            }
        }

        // Interleave exercises from different focus areas
        let focus_areas: Vec<_> = by_focus_area.keys().cloned().collect();
        let mut area_indices: HashMap<FocusArea, usize> = HashMap::new();

        for _ in 0..max_exercises {
            if interleaved.len() >= max_exercises {
                break;
            }

            for focus_area in &focus_areas {
                if let Some(exercises_in_area) = by_focus_area.get(focus_area) {
                    let current_index = area_indices.get(focus_area).unwrap_or(&0);

                    if *current_index < exercises_in_area.len() {
                        interleaved.push(exercises_in_area[*current_index].clone());
                        area_indices.insert(focus_area.clone(), current_index + 1);

                        if interleaved.len() >= max_exercises {
                            break;
                        }
                    }
                }
            }
        }

        interleaved
    }

    /// Optimize sequence based on forgetting curve theory
    async fn optimize_for_forgetting_curve(
        &self,
        exercises: &[TrainingExercise],
        user_id: &str,
    ) -> Result<Vec<TrainingExercise>, FeedbackError> {
        let history = self.get_user_exercise_history(user_id).await?;
        let mut optimized = exercises.to_vec();

        // Sort exercises to prioritize those at risk of being forgotten
        optimized.sort_by(|a, b| {
            let a_forgetting_risk = self.calculate_forgetting_risk(a, &history);
            let b_forgetting_risk = self.calculate_forgetting_risk(b, &history);
            b_forgetting_risk
                .partial_cmp(&a_forgetting_risk)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(optimized)
    }

    /// Calculate forgetting risk for an exercise based on Hermann Ebbinghaus forgetting curve
    fn calculate_forgetting_risk(
        &self,
        exercise: &TrainingExercise,
        history: &HashMap<String, ExerciseHistory>,
    ) -> f32 {
        if let Some(exercise_hist) = history.get(&exercise.exercise_id) {
            let now = Utc::now();
            let time_since_last = now
                .signed_duration_since(exercise_hist.last_attempted)
                .to_std()
                .unwrap_or_default();

            // Forgetting curve: R = e^(-t/S)
            // Where R = retention, t = time, S = stability (strength of memory)
            let stability = self.calculate_memory_stability(exercise_hist);
            let time_factor = time_since_last.as_secs() as f32 / stability;
            let retention = (-time_factor).exp();

            // Forgetting risk is inverse of retention
            1.0 - retention
        } else {
            // New exercises have medium forgetting risk
            0.5
        }
    }

    /// Calculate memory stability based on past performance and repetitions
    fn calculate_memory_stability(&self, history: &ExerciseHistory) -> f32 {
        // Base stability increases with successful repetitions
        let base_stability = 86400.0 * (1.0 + history.repetition_number as f32 * 0.5); // Days in seconds

        // Performance factor - better performance leads to stronger memory
        let performance_factor = (history.last_performance * 2.0).max(0.5);

        // Ease factor from spaced repetition algorithm
        let ease_adjustment = history.ease_factor;

        base_stability * performance_factor * ease_adjustment
    }

    /// Get collaborative learning system
    pub fn get_collaborative_learning_system(&self) -> Arc<CollaborativeLearningSystem> {
        self.collaborative_learning.clone()
    }

    /// Get training statistics
    pub async fn get_statistics(&self) -> Result<TrainingSystemStats, FeedbackError> {
        let metrics = self.metrics.read().unwrap();
        let exercise_library = self.exercise_library.read().unwrap();

        let success_rate = if metrics.total_attempts > 0 {
            metrics.successful_attempts as f32 / metrics.total_attempts as f32
        } else {
            0.0
        };

        let average_session_duration = if metrics.completed_sessions > 0 {
            Duration::from_secs(
                metrics.total_session_time.as_secs() / metrics.completed_sessions as u64,
            )
        } else {
            Duration::from_secs(0)
        };

        let average_evaluation_time = if metrics.total_attempts > 0 {
            Duration::from_secs(
                metrics.total_evaluation_time.as_secs() / metrics.total_attempts as u64,
            )
        } else {
            Duration::from_secs(0)
        };

        Ok(TrainingSystemStats {
            total_sessions: metrics.total_sessions,
            active_sessions: metrics.active_sessions,
            completed_sessions: metrics.completed_sessions,
            total_exercises: exercise_library.exercises.len(),
            total_attempts: metrics.total_attempts,
            successful_attempts: metrics.successful_attempts,
            success_rate,
            average_session_duration,
            average_evaluation_time,
        })
    }
}
