//! User Experience Analytics System - Modular Implementation
//!
//! This module provides comprehensive user experience tracking and analytics to monitor
//! and improve engagement metrics, learning effectiveness, and user satisfaction.
//! Specifically designed to track the critical success factors:
//! - >90% session completion rate
//! - >4.5/5 user satisfaction score
//! - >70% daily active user retention
//! - >80% user-reported progress satisfaction

pub mod core;
pub mod engagement;
pub mod sessions;
pub mod types;

// Re-export core types and configuration
pub use types::{
    ActivityContext, ActivityType, AudioCapabilities, BottleneckSeverity, ChurnRiskLevel,
    CsfComplianceStatus, CsfMetric, DeviceInfo, DifficultyProgression, EngagementLevel,
    ExperienceLevel, FeedbackStyle, ImplementationCost, ImplementationDifficulty, InsightCategory,
    InsightPriority, IssueSeverity, LearningPace, LearningPreferences, NetworkConditions,
    RecommendationPriority, RecommendationType, ResourceRequirements, SessionCompletionStatus,
    TrendDirection, UserState, UxAnalyticsConfig,
};

// Re-export engagement analytics types
pub use engagement::{
    DropOffPoint, EngagementAnalytics, EngagementPatterns, EngagementPredictions, EngagementTrends,
    FeatureEngagementMetrics, FeatureRequest, FeedbackAnalysis, FeedbackTheme, IssueReport,
    SatisfactionAnalytics, SatisfactionDriver, SatisfactionDrivers, SatisfactionPredictions,
    SatisfactionTrends, SentimentAnalysis, SessionEngagementMetrics, UserLifecycleEngagement,
};

// Re-export session analytics types
pub use sessions::{
    BehaviorAnalytics, BehaviorInsight, BehaviorPredictions, BehaviorProfile, BehaviorSegmentation,
    CrossSegmentAnalysis, DetailedSessionRecord, FeatureUsagePattern, FlowBottleneck,
    IntensityDistribution, JourneyOptimization, JourneyStage, JourneySuccessFactor, OptimalFlow,
    QualityFactors, QualityImprovement, QualityTrends, SegmentCharacteristics, SegmentMigration,
    SegmentPerformance, SessionActivity, SessionAnalytics, SessionContext, SessionFlow,
    SessionFlowAnalysis, SessionOutcomes, SessionQualityAnalysis, UsagePattern, UsagePatterns,
    UserJourney, UserJourneyAnalysis, UserSegment, UxSessionStatistics,
};

// Re-export core implementation and additional analytics types
pub use core::{
    calculate_session_engagement_score, ChurnAnalysis, ChurnPredictionModel, ChurnReason,
    ChurnReasonsAnalysis, ChurnRiskFactor, CohortAnalysis, CohortCharacteristics, CohortInsight,
    CohortRetentionData, ImpactAssessment, LearningEffectivenessMetrics, ProgressAcceleration,
    ProgressAnalytics, ProgressTrends, RetentionAnalytics, RetentionDataPoint, RetentionDriver,
    RetentionDrivers, RetentionPredictions, RetentionStatistics, RetentionStrategy,
    RetentionTrends, SkillProgressAnalytics, UserChurnRisk, UxAnalyticsReport, UxAnalyticsTracker,
    UxInsight, UxInsightsEngine, UxPredictions, UxRecommendation,
};

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    #[tokio::test]
    async fn test_ux_analytics_tracker_creation() {
        let config = UxAnalyticsConfig::default();
        let tracker = UxAnalyticsTracker::new(config).await.unwrap();

        let report = tracker.get_ux_report().await;
        assert_eq!(report.engagement_analytics.overall_engagement_score, 0.0);
    }

    #[tokio::test]
    async fn test_session_recording() {
        let config = UxAnalyticsConfig::default();
        let tracker = UxAnalyticsTracker::new(config).await.unwrap();

        let session = DetailedSessionRecord {
            session_id: "test-session".to_string(),
            user_id: "test-user".to_string(),
            start_time: Utc::now() - Duration::minutes(30),
            end_time: Some(Utc::now()),
            duration_seconds: 1800,
            completion_status: SessionCompletionStatus::Completed,
            activities: vec![],
            context: SessionContext {
                platform: "web".to_string(),
                device_info: DeviceInfo {
                    device_type: "desktop".to_string(),
                    os: "Windows".to_string(),
                    browser: Some("Chrome".to_string()),
                    screen_resolution: Some("1920x1080".to_string()),
                    audio_capabilities: AudioCapabilities {
                        microphone_available: true,
                        speaker_available: true,
                        audio_quality_score: 0.9,
                        audio_latency_ms: 50,
                    },
                },
                network_conditions: NetworkConditions {
                    connection_type: "broadband".to_string(),
                    quality_score: 0.95,
                    latency_ms: 20,
                    bandwidth_kbps: 50000,
                    stability_score: 0.98,
                },
                user_state: UserState {
                    experience_level: ExperienceLevel::Intermediate,
                    motivation_level: 0.8,
                    fatigue_level: 0.2,
                    confidence_level: 0.7,
                    stress_level: 0.1,
                },
            },
            outcomes: SessionOutcomes {
                goals_achieved: vec!["test_goal".to_string()],
                skills_improved: vec![crate::traits::FocusArea::Pronunciation],
                learning_progress: 0.15,
                session_satisfaction: 4.2,
                achievements_unlocked: vec![],
            },
        };

        tracker.record_session(session).await.unwrap();

        let report = tracker.get_ux_report().await;
        assert_eq!(report.session_analytics.session_stats.total_sessions, 1);
        assert_eq!(report.session_analytics.session_stats.completion_rate, 1.0);
    }

    #[tokio::test]
    async fn test_engagement_score_calculation() {
        let session = DetailedSessionRecord {
            session_id: "test".to_string(),
            user_id: "user".to_string(),
            start_time: Utc::now(),
            end_time: Some(Utc::now()),
            duration_seconds: 900, // 15 minutes
            completion_status: SessionCompletionStatus::Completed,
            activities: vec![SessionActivity {
                activity_id: "1".to_string(),
                activity_type: ActivityType::VoiceSynthesis,
                start_time: Utc::now(),
                duration_seconds: 300,
                success: true,
                engagement_score: 0.8,
                context: ActivityContext {
                    feature_area: "test".to_string(),
                    difficulty_level: None,
                    assistance_required: false,
                    error_count: 0,
                },
            }],
            context: SessionContext {
                platform: "test".to_string(),
                device_info: DeviceInfo {
                    device_type: "test".to_string(),
                    os: "test".to_string(),
                    browser: None,
                    screen_resolution: None,
                    audio_capabilities: AudioCapabilities {
                        microphone_available: true,
                        speaker_available: true,
                        audio_quality_score: 1.0,
                        audio_latency_ms: 0,
                    },
                },
                network_conditions: NetworkConditions {
                    connection_type: "test".to_string(),
                    quality_score: 1.0,
                    latency_ms: 0,
                    bandwidth_kbps: 1000,
                    stability_score: 1.0,
                },
                user_state: UserState {
                    experience_level: ExperienceLevel::Beginner,
                    motivation_level: 1.0,
                    fatigue_level: 0.0,
                    confidence_level: 1.0,
                    stress_level: 0.0,
                },
            },
            outcomes: SessionOutcomes {
                goals_achieved: vec![],
                skills_improved: vec![],
                learning_progress: 0.0,
                session_satisfaction: 5.0, // Max satisfaction
                achievements_unlocked: vec![],
            },
        };

        let score = calculate_session_engagement_score(&session);
        assert!(score > 0.8); // Should be high due to completion, good engagement, sufficient duration, and high satisfaction
    }

    #[tokio::test]
    async fn test_csf_compliance_calculation() {
        let config = UxAnalyticsConfig::default();
        let tracker = UxAnalyticsTracker::new(config).await.unwrap();

        // Initialize with sample data to get some metrics
        tracker.initialize_sample_data().await.unwrap();

        let report = tracker.get_ux_report().await;

        // Check that CSF compliance is calculated
        assert!(report.csf_compliance.session_completion_rate.current >= 0.0);
        assert!(report.csf_compliance.user_satisfaction_score.current >= 0.0);
    }

    #[tokio::test]
    async fn test_config_defaults() {
        let config = UxAnalyticsConfig::default();

        assert_eq!(config.target_completion_rate, 0.90);
        assert_eq!(config.target_satisfaction_score, 4.5);
        assert_eq!(config.target_retention_rate, 0.70);
        assert_eq!(config.target_progress_satisfaction, 0.80);
        assert!(config.enable_predictive_analytics);
        assert!(config.enable_realtime_alerts);
    }
}
