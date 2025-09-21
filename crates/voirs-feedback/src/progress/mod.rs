//! Progress tracking and analytics modules
//!
//! This module provides comprehensive user progress tracking, analytics,
//! and achievement system for continuous learning improvement.
//!
//! The implementation has been modularized into several components:
//! - `types`: Core types, enums, and data structures
//! - `core`: Main ProgressAnalyzer implementation and analytics framework
//! - `skills`: Skill taxonomy and sub-skill management
//! - `analytics`: Comprehensive analytics framework and related types  
//! - `metrics`: Core metrics calculation and measurement utilities
//! - `dashboard`: Dashboard generation and real-time data

pub mod analytics;
pub mod core;
pub mod dashboard;
pub mod metrics;
pub mod skills;
pub mod types;

// Re-export the main components for backwards compatibility
pub use skills::*;

// Re-export core types and structures
pub use types::{
    AchievementAnalysis,
    AchievementCondition,
    // Achievement and progress types
    AchievementDefinition,
    AchievementProgress,

    AchievementSummary,
    // Milestone and streak types
    AdaptiveMilestone,
    AggregatedMetric,
    // Analytics types
    AnalyticsConfig,
    AnalyticsMemoryManager,
    AnalyticsMetric,
    AnalyticsSummary,
    // Memory optimization types
    CircularProgressHistory,
    CleanupAction,
    ComparativeAnalysis,
    ComparativeAnalyticsResult,
    ComprehensiveAnalyticsReport,
    ComprehensiveStreakAnalysis,
    ConsistencyPattern,
    CurrentStreak,
    DetailedProgressReport,
    DifficultyPreference,
    EssentialTrainingStats,
    FocusPattern,
    GoalAnalysis,
    HistoricalStreak,
    LearningPatternAnalysis,
    LearningPlateau,
    LearningStyleProfile,
    LongitudinalDataPoint,
    LongitudinalStudyData,
    MemoryBoundedMetrics,
    MemoryOptimized,
    MemoryOptimizedProgress,
    MemoryStats,
    MetricType,
    MilestoneCriteria,
    MilestoneReward,
    MilestoneType,
    MotivationMaintenance,
    PeakPerformanceTime,
    ProgressRecommendation,
    RecoveryMechanism,
    RecoveryStrategyType,
    StatisticalSignificanceResult,
    StreakBreakReason,
    StreakPatterns,
    StreakRecoveryPlan,

    StreakType,
    TrendAnalysis,
    TrendAnalytics,
};

// Re-export core functionality
pub use core::{ComprehensiveAnalyticsFramework, ProgressAnalyzer};

// Re-export analytics components
pub use analytics::TrendDirection;

// Re-export dashboard components
pub use dashboard::{
    DashboardConfig, DashboardConfigBuilder, DashboardMetricsGenerator, DashboardStatus,
    DashboardTheme, MetricDataPoint, RealTimeDashboardData,
};

// Re-export metrics components
pub use metrics::{ConsistencyMetrics, MetricsCalculator, ProgressSystemStats, SessionAnalytics};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{
        Achievement, AchievementTier, FocusArea, Goal, GoalMetric, ProgressConfig,
        ProgressIndicators, ProgressReport, ProgressSnapshot, ProgressTracker, ReportStatistics,
        SessionScores, SessionState, TimeRange, TrainingScores, TrainingStatistics, UserProgress,
    };
    use chrono::Utc;
    use std::collections::HashMap;
    use std::time::Duration;

    #[tokio::test]
    async fn test_progress_analyzer_creation() {
        let analyzer = ProgressAnalyzer::new().await.unwrap();
        let report = analyzer
            .generate_progress_report("test_user", None)
            .await
            .unwrap();

        assert_eq!(report.user_id, "test_user");
        assert!(report.overall_improvement >= 0.0 && report.overall_improvement <= 1.0);
    }

    #[tokio::test]
    async fn test_user_progress_creation() {
        let analyzer = ProgressAnalyzer::new().await.unwrap();
        let user_progress = analyzer.get_user_progress("test_user").await.unwrap();

        assert_eq!(user_progress.user_id, "test_user");
        assert!(
            user_progress.overall_skill_level >= 0.0 && user_progress.overall_skill_level <= 1.0
        );
    }

    #[tokio::test]
    async fn test_progress_recording() {
        let analyzer = ProgressAnalyzer::new().await.unwrap();

        let training_scores = TrainingScores {
            quality: 0.8,
            pronunciation: 0.8,
            consistency: 0.7,
            improvement: 0.1,
        };

        analyzer
            .record_training_session("test_user", &training_scores)
            .await
            .unwrap();

        let progress = analyzer.get_user_progress("test_user").await.unwrap();
        assert!(!progress.recent_sessions.is_empty());
        assert!(progress.total_practice_time > Duration::ZERO);
    }

    #[tokio::test]
    async fn test_achievement_checking() {
        let analyzer = ProgressAnalyzer::new().await.unwrap();

        let training_scores = TrainingScores {
            quality: 0.9,
            pronunciation: 0.9,
            consistency: 0.9,
            improvement: 0.2,
        };

        analyzer
            .record_training_session("test_user", &training_scores)
            .await
            .unwrap();

        let achievements = analyzer.check_achievements("test_user").await.unwrap();
        // Some achievements might be unlocked based on the high scores
        for achievement in achievements {
            assert!(!achievement.name.is_empty());
            assert!(!achievement.description.is_empty());
        }
    }

    #[tokio::test]
    async fn test_goal_setting() {
        let analyzer = ProgressAnalyzer::new().await.unwrap();

        let goal = Goal {
            goal_id: "test_goal".to_string(),
            description: "Achieve 85% pronunciation accuracy".to_string(),
            target_value: 0.85,
            progress: 0.7,
            target_metric: GoalMetric::OverallSkill,
            deadline: Some(Utc::now() + chrono::Duration::days(30)),
            active: true,
        };

        analyzer.set_goal("test_user", goal.clone()).await.unwrap();

        let goals = analyzer.get_user_goals("test_user").await.unwrap();
        assert_eq!(goals.len(), 1);
        assert_eq!(goals[0].goal_id, "test_goal");
    }

    #[tokio::test]
    async fn test_progress_report_generation() {
        let analyzer = ProgressAnalyzer::new().await.unwrap();

        // Add some training data
        let training_scores = TrainingScores {
            quality: 0.8,
            pronunciation: 0.8,
            consistency: 0.7,
            improvement: 0.1,
        };

        analyzer
            .record_training_session("test_user", &training_scores)
            .await
            .unwrap();

        let report = analyzer
            .generate_progress_report("test_user", None)
            .await
            .unwrap();

        assert_eq!(report.user_id, "test_user");
        assert!(report.overall_improvement >= 0.0 && report.overall_improvement <= 1.0);
        assert!(!report.area_improvements.is_empty());
        assert!(report.statistics.total_practice_time > Duration::ZERO);
        assert!(report.statistics.sessions_count > 0);
    }

    #[tokio::test]
    async fn test_learning_pattern_analysis() {
        let analyzer = ProgressAnalyzer::new().await.unwrap();

        // Add multiple training sessions to establish patterns
        for i in 0..5 {
            let score = 0.6 + (i as f32 * 0.05); // Gradually improving
            let training_scores = TrainingScores {
                quality: score,
                pronunciation: score,
                consistency: score,
                improvement: 0.05,
            };

            analyzer
                .record_training_session("test_user", &training_scores)
                .await
                .unwrap();
        }

        let patterns = analyzer
            .analyze_learning_patterns("test_user")
            .await
            .unwrap();

        assert!(!patterns.is_empty());
        // Should detect an improving trend
        assert!(patterns
            .iter()
            .any(|p| p.contains("improving") || p.contains("progress")));
    }
}
