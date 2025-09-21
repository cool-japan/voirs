//! Results and feedback generation functionality
//!
//! This module contains methods for generating feedback, analyzing results,
//! calculating scores, and providing recommendations for training improvement.

use crate::training::core::InteractiveTrainer;
use crate::training::types::{
    AttemptFeedback, CriteriaCompliance, ExerciseAttempt, ExerciseResult, ImprovementSuggestion,
    LearningPath,
};
use crate::traits::{ExerciseType, FocusArea, SuccessCriteria, TrainingExercise};
use crate::FeedbackError;
use std::time::Duration;

/// Utility trait for f32 midpoint calculation
trait F32Midpoint {
    fn midpoint(a: f32, b: f32) -> f32;
}

impl F32Midpoint for f32 {
    fn midpoint(a: f32, b: f32) -> f32 {
        (a + b) / 2.0
    }
}

impl InteractiveTrainer {
    /// Generate detailed feedback for an exercise attempt
    pub async fn generate_attempt_feedback(
        &self,
        quality_score: &voirs_evaluation::QualityScore,
        pronunciation_score: &voirs_evaluation::PronunciationScore,
        exercise: &TrainingExercise,
    ) -> Result<AttemptFeedback, FeedbackError> {
        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();
        let mut suggestions = Vec::new();

        // Analyze quality
        if quality_score.overall_score > 0.8 {
            strengths.push("Excellent audio quality".to_string());
        } else if quality_score.overall_score < 0.6 {
            weaknesses.push("Audio quality needs improvement".to_string());
            suggestions.push("Check your recording setup and environment".to_string());
        }

        // Analyze pronunciation
        if pronunciation_score.overall_score > 0.8 {
            strengths.push("Good pronunciation accuracy".to_string());
        } else if pronunciation_score.overall_score < 0.6 {
            weaknesses.push("Pronunciation needs work".to_string());
            suggestions.push("Practice phoneme accuracy and word stress".to_string());
        }

        // Exercise-specific feedback
        match exercise.exercise_type {
            ExerciseType::Pronunciation => {
                if pronunciation_score.overall_score < 0.7 {
                    suggestions.push("Focus on clear articulation of each phoneme".to_string());
                }
            }
            ExerciseType::Quality => {
                if quality_score.overall_score < 0.7 {
                    suggestions
                        .push("Improve recording quality and reduce background noise".to_string());
                }
            }
            ExerciseType::Rhythm => {
                suggestions.push("Pay attention to timing and pacing".to_string());
            }
            _ => {}
        }

        let overall_score = f32::midpoint(
            quality_score.overall_score,
            pronunciation_score.overall_score,
        );

        Ok(AttemptFeedback {
            overall_score,
            quality_score: quality_score.overall_score,
            pronunciation_score: pronunciation_score.overall_score,
            strengths,
            weaknesses,
            suggestions,
            encouragement: self.generate_encouragement(overall_score),
        })
    }

    /// Generate encouraging message based on performance score
    pub fn generate_encouragement(&self, score: f32) -> String {
        match score {
            s if s > 0.9 => "Outstanding performance! You're mastering this!".to_string(),
            s if s > 0.8 => "Great work! You're doing very well.".to_string(),
            s if s > 0.7 => "Good progress! Keep it up.".to_string(),
            s if s > 0.6 => "You're improving! Stay focused.".to_string(),
            _ => "Keep practicing! Every attempt makes you better.".to_string(),
        }
    }

    /// Check if an attempt meets the success criteria
    pub fn check_success_criteria(
        &self,
        attempt: &ExerciseAttempt,
        criteria: &SuccessCriteria,
    ) -> bool {
        attempt.quality_score >= criteria.min_quality_score
            && attempt.pronunciation_score >= criteria.min_pronunciation_score
    }

    /// Analyze how well an attempt meets the success criteria
    pub fn analyze_criteria_compliance(
        &self,
        attempt: &ExerciseAttempt,
        criteria: &SuccessCriteria,
    ) -> CriteriaCompliance {
        CriteriaCompliance {
            quality_met: attempt.quality_score >= criteria.min_quality_score,
            pronunciation_met: attempt.pronunciation_score >= criteria.min_pronunciation_score,
            quality_gap: (criteria.min_quality_score - attempt.quality_score).max(0.0),
            pronunciation_gap: (criteria.min_pronunciation_score - attempt.pronunciation_score)
                .max(0.0),
        }
    }

    /// Generate next steps for the user based on attempt results
    pub fn generate_next_steps(
        &self,
        attempt: &ExerciseAttempt,
        exercise: &TrainingExercise,
        success: bool,
    ) -> Vec<String> {
        let mut steps = Vec::new();

        if success {
            steps.push("Excellent! Try a more challenging exercise.".to_string());
            steps.push("Maintain this level of quality in your next attempt.".to_string());
        } else {
            if attempt.quality_score < exercise.success_criteria.min_quality_score {
                steps.push("Focus on improving audio quality".to_string());
            }
            if attempt.pronunciation_score < exercise.success_criteria.min_pronunciation_score {
                steps.push("Work on pronunciation accuracy".to_string());
            }
            steps.push("Review the feedback and try again".to_string());
        }

        steps
    }

    /// Generate specific improvement suggestions for an attempt
    pub fn generate_improvement_suggestions(
        &self,
        attempt: &ExerciseAttempt,
        exercise: &TrainingExercise,
    ) -> Vec<ImprovementSuggestion> {
        let mut suggestions = Vec::new();

        // Quality-based suggestions
        if attempt.quality_score < 0.8 {
            suggestions.push(ImprovementSuggestion {
                area: "Audio Quality".to_string(),
                current_score: attempt.quality_score,
                target_score: exercise.success_criteria.min_quality_score,
                specific_actions: vec![
                    "Use a better microphone".to_string(),
                    "Record in a quiet environment".to_string(),
                    "Maintain consistent distance from microphone".to_string(),
                ],
                estimated_practice_time: Duration::from_secs(600), // 10 minutes
            });
        }

        // Pronunciation-based suggestions
        if attempt.pronunciation_score < 0.8 {
            suggestions.push(ImprovementSuggestion {
                area: "Pronunciation".to_string(),
                current_score: attempt.pronunciation_score,
                target_score: exercise.success_criteria.min_pronunciation_score,
                specific_actions: vec![
                    "Practice phoneme exercises".to_string(),
                    "Listen to reference pronunciations".to_string(),
                    "Work on word stress patterns".to_string(),
                ],
                estimated_practice_time: Duration::from_secs(900), // 15 minutes
            });
        }

        suggestions
    }

    /// Calculate consistency score across multiple attempts
    pub fn calculate_consistency_score(&self, attempts: &[ExerciseAttempt]) -> f32 {
        if attempts.len() < 2 {
            return 0.5;
        }

        let scores: Vec<f32> = attempts
            .iter()
            .map(|a| f32::midpoint(a.quality_score, a.pronunciation_score))
            .collect();

        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;

        // Lower variance = higher consistency
        1.0 / (1.0 + variance)
    }

    /// Calculate improvement score from first to last attempt
    pub fn calculate_improvement_score(&self, attempts: &[ExerciseAttempt]) -> f32 {
        if attempts.len() < 2 {
            return 0.0;
        }

        let first_score = f32::midpoint(attempts[0].quality_score, attempts[0].pronunciation_score);
        let last_score = f32::midpoint(
            attempts.last().unwrap().quality_score,
            attempts.last().unwrap().pronunciation_score,
        );

        (last_score - first_score).max(0.0)
    }

    /// Calculate improvement trend across a session
    pub fn calculate_session_improvement_trend(&self, exercises: &[ExerciseResult]) -> f32 {
        if exercises.len() < 2 {
            return 0.0;
        }

        let scores: Vec<f32> = exercises
            .iter()
            .map(|e| f32::midpoint(e.final_scores.quality, e.final_scores.pronunciation))
            .collect();

        // Simple linear trend calculation
        let n = scores.len() as f32;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = scores.iter().sum::<f32>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &score) in scores.iter().enumerate() {
            let x_diff = i as f32 - x_mean;
            numerator += x_diff * (score - y_mean);
            denominator += x_diff * x_diff;
        }

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Check for session achievements based on performance
    pub async fn check_session_achievements(
        &self,
        exercises: &[ExerciseResult],
    ) -> Result<Vec<String>, FeedbackError> {
        // Simplified achievement checking
        let mut achievements = Vec::new();

        if exercises.len() >= 3 {
            achievements.push("Session Champion - Completed 3+ exercises".to_string());
        }

        let success_rate =
            exercises.iter().filter(|e| e.success).count() as f32 / exercises.len() as f32;
        if success_rate >= 0.8 {
            achievements.push("High Achiever - 80%+ success rate".to_string());
        }

        Ok(achievements)
    }

    /// Generate session-level recommendations for improvement
    pub fn generate_session_recommendations(&self, exercises: &[ExerciseResult]) -> Vec<String> {
        let mut recommendations = Vec::new();

        if exercises.is_empty() {
            recommendations.push(
                "Complete at least one exercise to get personalized recommendations".to_string(),
            );
            return recommendations;
        }

        // Analyze weak areas
        let avg_quality: f32 = exercises
            .iter()
            .map(|e| e.final_scores.quality)
            .sum::<f32>()
            / exercises.len() as f32;
        let avg_pronunciation: f32 = exercises
            .iter()
            .map(|e| e.final_scores.pronunciation)
            .sum::<f32>()
            / exercises.len() as f32;

        if avg_quality < 0.7 {
            recommendations
                .push("Focus on improving audio quality in your next session".to_string());
        }

        if avg_pronunciation < 0.7 {
            recommendations
                .push("Practice pronunciation exercises to improve accuracy".to_string());
        }

        if avg_quality > 0.8 && avg_pronunciation > 0.8 {
            recommendations
                .push("Excellent work! Try more challenging exercises next time".to_string());
        }

        recommendations
    }

    /// Suggest next learning path based on session performance
    pub fn suggest_next_learning_path(&self, exercises: &[ExerciseResult]) -> LearningPath {
        if exercises.is_empty() {
            return LearningPath {
                suggested_focus_areas: vec![FocusArea::Pronunciation, FocusArea::Quality],
                difficulty_level: 0.3,
                estimated_duration: Duration::from_secs(1800), // 30 minutes
                exercise_types: vec![ExerciseType::FreeForm, ExerciseType::Pronunciation],
            };
        }

        // Analyze performance to suggest next path
        let avg_quality: f32 = exercises
            .iter()
            .map(|e| e.final_scores.quality)
            .sum::<f32>()
            / exercises.len() as f32;
        let avg_pronunciation: f32 = exercises
            .iter()
            .map(|e| e.final_scores.pronunciation)
            .sum::<f32>()
            / exercises.len() as f32;
        let overall_avg = f32::midpoint(avg_quality, avg_pronunciation);

        let mut focus_areas = Vec::new();
        if avg_pronunciation < 0.8 {
            focus_areas.push(FocusArea::Pronunciation);
        }
        if avg_quality < 0.8 {
            focus_areas.push(FocusArea::Quality);
        }
        if focus_areas.is_empty() {
            focus_areas.push(FocusArea::Naturalness);
            focus_areas.push(FocusArea::Fluency);
        }

        LearningPath {
            suggested_focus_areas: focus_areas,
            difficulty_level: (overall_avg + 0.1).min(1.0), // Slightly increase difficulty
            estimated_duration: Duration::from_secs(2400),  // 40 minutes
            exercise_types: vec![ExerciseType::Pronunciation, ExerciseType::Quality],
        }
    }
}
