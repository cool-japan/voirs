//! Core types, enums, and data structures for UX Analytics
//!
//! This module contains the fundamental types and enums used throughout
//! the UX analytics system, including configuration and shared data structures.

use crate::traits::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// User experience analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UxAnalyticsConfig {
    /// Target session completion rate (90% = 0.90)
    pub target_completion_rate: f32,
    /// Target user satisfaction score (4.5/5 = 4.5)
    pub target_satisfaction_score: f32,
    /// Target daily retention rate (70% = 0.70)
    pub target_retention_rate: f32,
    /// Target progress satisfaction (80% = 0.80)
    pub target_progress_satisfaction: f32,
    /// Analytics collection interval in seconds
    pub collection_interval_seconds: u64,
    /// Maximum number of records to keep in memory
    pub max_records: usize,
    /// Enable predictive analytics
    pub enable_predictive_analytics: bool,
    /// Enable real-time UX alerts
    pub enable_realtime_alerts: bool,
    /// Satisfaction survey trigger frequency (sessions)
    pub satisfaction_survey_frequency: u32,
}

impl Default for UxAnalyticsConfig {
    fn default() -> Self {
        Self {
            target_completion_rate: 0.90,
            target_satisfaction_score: 4.5,
            target_retention_rate: 0.70,
            target_progress_satisfaction: 0.80,
            collection_interval_seconds: 30,
            max_records: 10000,
            enable_predictive_analytics: true,
            enable_realtime_alerts: true,
            satisfaction_survey_frequency: 5,
        }
    }
}

/// Trend direction enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Strongly increasing trend
    StronglyIncreasing,
    /// Increasing trend
    Increasing,
    /// Stable trend
    Stable,
    /// Decreasing trend
    Decreasing,
    /// Strongly decreasing trend
    StronglyDecreasing,
    /// Volatile trend
    Volatile,
}

/// Session completion status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SessionCompletionStatus {
    /// Session completed successfully
    Completed,
    /// Session partially completed
    PartiallyCompleted,
    /// Session abandoned by user
    Abandoned,
    /// Session interrupted
    Interrupted,
    /// Session failed
    Failed,
}

/// Activity type enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActivityType {
    /// Voice synthesis activity
    VoiceSynthesis,
    /// Feedback review activity
    FeedbackReview,
    /// Exercise completion activity
    ExerciseCompletion,
    /// Progress review activity
    ProgressReview,
    /// Settings configuration activity
    SettingsConfiguration,
    /// Tutorial viewing activity
    TutorialViewing,
    /// Achievement viewing activity
    AchievementViewing,
    /// Social interaction activity
    SocialInteraction,
    /// Other activity type
    Other(String),
}

/// User experience level
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExperienceLevel {
    /// Beginner level
    Beginner,
    /// Intermediate level
    Intermediate,
    /// Advanced level
    Advanced,
    /// Expert level
    Expert,
}

/// Implementation difficulty levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImplementationDifficulty {
    /// Easy implementation
    Easy,
    /// Medium difficulty
    Medium,
    /// Hard implementation
    Hard,
    /// Requires research
    RequiresResearch,
}

/// Bottleneck severity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    /// Minor severity
    Minor,
    /// Moderate severity
    Moderate,
    /// Major severity
    Major,
    /// Critical severity
    Critical,
}

/// Issue severity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Churn risk levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChurnRiskLevel {
    /// Low churn risk
    Low,
    /// Medium churn risk
    Medium,
    /// High churn risk
    High,
    /// Critical churn risk
    Critical,
}

/// Implementation cost levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImplementationCost {
    /// Low cost
    Low,
    /// Medium cost
    Medium,
    /// High cost
    High,
    /// Very high cost
    VeryHigh,
}

/// Engagement level categorization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EngagementLevel {
    /// Low engagement
    Low,
    /// Medium engagement
    Medium,
    /// High engagement
    High,
    /// Very high engagement
    VeryHigh,
}

/// Learning pace preferences
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LearningPace {
    /// Slow learning pace
    Slow,
    /// Moderate learning pace
    Moderate,
    /// Fast learning pace
    Fast,
    /// Variable learning pace
    Variable,
}

/// Feedback style preferences
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeedbackStyle {
    /// Detailed feedback
    Detailed,
    /// Concise feedback
    Concise,
    /// Visual feedback
    Visual,
    /// Audio feedback
    Audio,
    /// Mixed feedback styles
    Mixed,
}

/// Difficulty progression preferences
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DifficultyProgression {
    /// Gradual progression
    Gradual,
    /// Moderate progression
    Moderate,
    /// Aggressive progression
    Aggressive,
    /// Adaptive progression
    Adaptive,
}

/// Insight priority levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InsightPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Insight categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InsightCategory {
    /// Engagement insights
    Engagement,
    /// Retention insights
    Retention,
    /// Satisfaction insights
    Satisfaction,
    /// Learning effectiveness insights
    LearningEffectiveness,
    /// User behavior insights
    UserBehavior,
    /// Performance insights
    Performance,
    /// Opportunity insights
    Opportunity,
    /// Risk insights
    Risk,
}

/// Recommendation types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Feature improvement recommendation
    FeatureImprovement,
    /// New feature recommendation
    NewFeature,
    /// User flow optimization
    UserFlowOptimization,
    /// Personalization enhancement
    PersonalizationEnhancement,
    /// Performance optimization
    PerformanceOptimization,
    /// Engagement strategy
    EngagementStrategy,
    /// Retention strategy
    RetentionStrategy,
    /// Content optimization
    ContentOptimization,
}

/// Recommendation priority levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Urgent priority
    Urgent,
}

/// Activity context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityContext {
    /// Feature or area being used
    pub feature_area: String,
    /// Difficulty level if applicable
    pub difficulty_level: Option<f32>,
    /// User assistance required
    pub assistance_required: bool,
    /// Error count during activity
    pub error_count: u32,
}

/// Device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device type
    pub device_type: String,
    /// Operating system
    pub os: String,
    /// Browser if web platform
    pub browser: Option<String>,
    /// Screen resolution
    pub screen_resolution: Option<String>,
    /// Audio capabilities
    pub audio_capabilities: AudioCapabilities,
}

/// Audio capabilities information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioCapabilities {
    /// Microphone available
    pub microphone_available: bool,
    /// Speaker/headphones available
    pub speaker_available: bool,
    /// Audio quality score
    pub audio_quality_score: f32,
    /// Audio latency in milliseconds
    pub audio_latency_ms: u32,
}

/// Network conditions information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConditions {
    /// Connection type
    pub connection_type: String,
    /// Connection quality score (0.0 to 1.0)
    pub quality_score: f32,
    /// Latency in milliseconds
    pub latency_ms: u32,
    /// Bandwidth estimate in kbps
    pub bandwidth_kbps: u32,
    /// Connection stability
    pub stability_score: f32,
}

/// User state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserState {
    /// User experience level
    pub experience_level: ExperienceLevel,
    /// User motivation level (0.0 to 1.0)
    pub motivation_level: f32,
    /// User fatigue level (0.0 to 1.0)
    pub fatigue_level: f32,
    /// User confidence level (0.0 to 1.0)
    pub confidence_level: f32,
    /// User stress level (0.0 to 1.0)
    pub stress_level: f32,
}

/// Learning preferences profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPreferences {
    /// Preferred learning pace
    pub pace: LearningPace,
    /// Preferred feedback style
    pub feedback_style: FeedbackStyle,
    /// Preferred difficulty progression
    pub difficulty_progression: DifficultyProgression,
    /// Gamification responsiveness
    pub gamification_responsiveness: f32,
}

/// Resource requirements for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Development effort estimate (person-days)
    pub development_effort_days: u32,
    /// Design effort estimate (person-days)
    pub design_effort_days: u32,
    /// Research effort estimate (person-days)
    pub research_effort_days: u32,
    /// Additional resources needed
    pub additional_resources: Vec<String>,
    /// Estimated timeline
    pub estimated_timeline_weeks: u32,
}

/// Critical Success Factor compliance status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsfComplianceStatus {
    /// Session completion rate metric
    pub session_completion_rate: CsfMetric,
    /// User satisfaction score metric
    pub user_satisfaction_score: CsfMetric,
    /// Daily retention rate metric
    pub daily_retention_rate: CsfMetric,
    /// Progress satisfaction metric
    pub progress_satisfaction: CsfMetric,
}

/// Individual CSF metric status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsfMetric {
    /// Current metric value
    pub current: f32,
    /// Target metric value
    pub target: f32,
    /// Whether metric meets target
    pub is_compliant: bool,
}
