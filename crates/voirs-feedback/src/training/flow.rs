//! Session flow optimization and attention span management
//!
//! This module provides intelligent session management through fatigue detection,
//! attention span modeling, and break timing algorithms.

use crate::training::types::{
    AttentionSpanConfig, BreakTimingAlgorithm, ExerciseAttempt, SessionFlowOptimizer,
    SessionFlowRecommendation, TrainingSession,
};
use crate::traits::{ExerciseType, UserBehaviorPatterns};
use chrono::Utc;
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

impl SessionFlowOptimizer {
    /// Create new session flow optimizer
    #[must_use]
    pub fn new() -> Self {
        Self {
            attention_span_config: AttentionSpanConfig::default(),
            break_timer: BreakTimingAlgorithm::default(),
            fatigue_threshold: 0.3,
            performance_decline_threshold: 0.15,
        }
    }

    /// Analyze current session flow and recommend next action
    #[must_use]
    pub fn analyze_session_flow(
        &self,
        session: &TrainingSession,
        recent_attempts: &[ExerciseAttempt],
        user_patterns: &UserBehaviorPatterns,
    ) -> SessionFlowRecommendation {
        let session_duration = Utc::now()
            .signed_duration_since(session.start_time)
            .to_std()
            .unwrap_or(Duration::from_secs(0));

        // Check for fatigue indicators
        let fatigue_level = self.calculate_fatigue_level(recent_attempts, session_duration);

        // Check attention span
        let attention_remaining =
            self.estimate_attention_remaining(session_duration, user_patterns);

        // Analyze performance trend
        let performance_trend = self.analyze_performance_trend(recent_attempts);

        // Generate recommendation
        if fatigue_level > self.fatigue_threshold {
            SessionFlowRecommendation::TakeBreak {
                reason: "Fatigue detected".to_string(),
                suggested_break_duration: self.calculate_break_duration(fatigue_level),
                resume_suggestion: "Try some light stretching or hydration".to_string(),
            }
        } else if attention_remaining < 0.2 {
            SessionFlowRecommendation::TakeBreak {
                reason: "Attention span declining".to_string(),
                suggested_break_duration: Duration::from_secs(300), // 5 minutes
                resume_suggestion: "Take a brief walk or practice deep breathing".to_string(),
            }
        } else if performance_trend < -self.performance_decline_threshold {
            SessionFlowRecommendation::AdjustDifficulty {
                reason: "Performance declining".to_string(),
                suggested_difficulty_change: -0.1,
                alternative_exercise_types: vec![ExerciseType::Review],
            }
        } else if performance_trend > 0.1 && attention_remaining > 0.5 {
            SessionFlowRecommendation::IncreaseChallenge {
                reason: "Performing well with good attention".to_string(),
                suggested_difficulty_change: 0.05,
                new_exercise_types: vec![ExerciseType::Advanced],
            }
        } else {
            SessionFlowRecommendation::Continue {
                estimated_remaining_time: Duration::from_secs(
                    (attention_remaining * 1800.0) as u64,
                ),
                motivation_message: self.generate_motivation_message(performance_trend),
            }
        }
    }

    /// Calculate current fatigue level based on performance and time
    fn calculate_fatigue_level(
        &self,
        recent_attempts: &[ExerciseAttempt],
        session_duration: Duration,
    ) -> f32 {
        if recent_attempts.is_empty() {
            return 0.0;
        }

        // Time-based fatigue (increases with session duration)
        let time_fatigue = (session_duration.as_secs() as f32 / 3600.0).min(1.0);

        // Performance-based fatigue (declining scores indicate fatigue)
        let performance_fatigue = if recent_attempts.len() >= 3 {
            let recent_scores: Vec<f32> = recent_attempts
                .iter()
                .rev()
                .take(5)
                .map(|a| f32::midpoint(a.quality_score, a.pronunciation_score))
                .collect();

            let early_avg = recent_scores.iter().rev().take(2).sum::<f32>() / 2.0;
            let late_avg = recent_scores.iter().take(2).sum::<f32>() / 2.0;

            ((early_avg - late_avg) * 2.0).max(0.0)
        } else {
            0.0
        };

        // Evaluation time fatigue (longer evaluation times indicate fatigue)
        let reaction_fatigue = if recent_attempts.len() >= 2 {
            let avg_evaluation_time = recent_attempts
                .iter()
                .rev()
                .take(3)
                .map(|a| a.evaluation_time.as_millis() as f32)
                .sum::<f32>()
                / 3.0;

            (avg_evaluation_time / 10000.0).min(0.5) // Normalize to 0-0.5 range
        } else {
            0.0
        };

        // Weighted combination
        (time_fatigue * 0.3 + performance_fatigue * 0.5 + reaction_fatigue * 0.2).min(1.0)
    }

    /// Estimate remaining attention span as a percentage
    fn estimate_attention_remaining(
        &self,
        session_duration: Duration,
        user_patterns: &UserBehaviorPatterns,
    ) -> f32 {
        let user_avg_duration = user_patterns.average_session_duration.as_secs() as f32;
        let current_duration = session_duration.as_secs() as f32;

        if user_avg_duration > 0.0 {
            ((user_avg_duration - current_duration) / user_avg_duration)
                .max(0.0)
                .min(1.0)
        } else {
            // Default attention span model: 45 minutes peak, declining after
            let peak_duration = 2700.0; // 45 minutes
            if current_duration < peak_duration {
                1.0 - (current_duration / peak_duration) * 0.3 // Slight decline to 70%
            } else {
                let overtime = current_duration - peak_duration;
                (0.7 - (overtime / 1800.0) * 0.5).max(0.0) // Decline to 20% over 30 minutes
            }
        }
    }

    /// Analyze performance trend over recent attempts
    fn analyze_performance_trend(&self, recent_attempts: &[ExerciseAttempt]) -> f32 {
        if recent_attempts.len() < 3 {
            return 0.0;
        }

        let scores: Vec<f32> = recent_attempts
            .iter()
            .rev()
            .take(5)
            .map(|a| f32::midpoint(a.quality_score, a.pronunciation_score))
            .collect();

        if scores.len() < 3 {
            return 0.0;
        }

        // Calculate linear trend
        let n = scores.len() as f32;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = scores.iter().sum::<f32>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &score) in scores.iter().enumerate() {
            let x_dev = i as f32 - x_mean;
            numerator += x_dev * (score - y_mean);
            denominator += x_dev * x_dev;
        }

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Calculate appropriate break duration based on fatigue level
    fn calculate_break_duration(&self, fatigue_level: f32) -> Duration {
        let base_break = 300; // 5 minutes
        let additional_time = (fatigue_level * 900.0) as u64; // Up to 15 additional minutes
        Duration::from_secs(base_break + additional_time)
    }

    /// Generate motivational message based on performance
    fn generate_motivation_message(&self, performance_trend: f32) -> String {
        if performance_trend > 0.05 {
            "Great progress! You're improving with each attempt.".to_string()
        } else if performance_trend < -0.05 {
            "Stay focused! Small improvements add up over time.".to_string()
        } else {
            "You're maintaining consistent performance. Keep it up!".to_string()
        }
    }
}
