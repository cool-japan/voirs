//! Engagement and Satisfaction Analytics
//!
//! This module contains all analytics related to user engagement and satisfaction,
//! including engagement patterns, satisfaction tracking, and feedback analysis.

use super::types::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Engagement analytics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementAnalytics {
    /// Total user engagement score (0.0 to 1.0)
    pub overall_engagement_score: f32,
    /// Session engagement breakdown
    pub session_engagement: SessionEngagementMetrics,
    /// Feature engagement tracking
    pub feature_engagement: HashMap<String, FeatureEngagementMetrics>,
    /// Time-based engagement patterns
    pub engagement_patterns: EngagementPatterns,
    /// Engagement trend analysis
    pub engagement_trends: EngagementTrends,
}

/// Session engagement metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEngagementMetrics {
    /// Average session duration in minutes
    pub avg_session_duration_minutes: f32,
    /// Session completion rate
    pub completion_rate: f32,
    /// Average interactions per session
    pub avg_interactions_per_session: f32,
    /// Drop-off points analysis
    pub drop_off_points: Vec<DropOffPoint>,
    /// Session quality score (0.0 to 1.0)
    pub session_quality_score: f32,
}

/// Feature-specific engagement metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngagementMetrics {
    /// Feature name
    pub feature_name: String,
    /// Usage frequency
    pub usage_frequency: f32,
    /// Time spent on feature (minutes)
    pub avg_time_spent_minutes: f32,
    /// User satisfaction with feature
    pub feature_satisfaction: f32,
    /// Feature adoption rate
    pub adoption_rate: f32,
    /// Feature abandonment rate
    pub abandonment_rate: f32,
}

/// Engagement patterns analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementPatterns {
    /// Peak engagement hours
    pub peak_hours: Vec<u32>,
    /// Daily engagement patterns
    pub daily_patterns: HashMap<String, f32>, // weekday -> engagement_score
    /// Seasonal engagement trends
    pub seasonal_trends: HashMap<String, f32>,
    /// User lifecycle engagement
    pub lifecycle_engagement: UserLifecycleEngagement,
}

/// User lifecycle engagement stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserLifecycleEngagement {
    /// Onboarding engagement (first 7 days)
    pub onboarding_engagement: f32,
    /// Growth engagement (days 8-30)
    pub growth_engagement: f32,
    /// Retention engagement (days 31-90)
    pub retention_engagement: f32,
    /// Mature user engagement (90+ days)
    pub mature_engagement: f32,
}

/// Engagement trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementTrends {
    /// Weekly engagement trend
    pub weekly_trend: TrendDirection,
    /// Monthly engagement trend
    pub monthly_trend: TrendDirection,
    /// Engagement velocity (rate of change)
    pub engagement_velocity: f32,
    /// Predicted future engagement
    pub predicted_engagement: f32,
    /// Trend confidence (0.0 to 1.0)
    pub trend_confidence: f32,
}

/// Drop-off point analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropOffPoint {
    /// Step or feature where users drop off
    pub step_name: String,
    /// Percentage of users who drop off at this point
    pub drop_off_percentage: f32,
    /// Average time before drop-off (minutes)
    pub avg_time_before_dropout_minutes: f32,
    /// Common reasons for drop-off
    pub common_reasons: Vec<String>,
    /// Suggested improvements
    pub improvement_suggestions: Vec<String>,
}

/// Satisfaction analytics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatisfactionAnalytics {
    /// Overall satisfaction score
    pub overall_satisfaction: f32,
    /// Satisfaction by category
    pub category_satisfaction: HashMap<String, f32>,
    /// Satisfaction trends
    pub satisfaction_trends: SatisfactionTrends,
    /// Satisfaction drivers analysis
    pub satisfaction_drivers: SatisfactionDrivers,
    /// User feedback analysis
    pub feedback_analysis: FeedbackAnalysis,
}

/// Satisfaction trends analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatisfactionTrends {
    /// Weekly satisfaction trend
    pub weekly_trend: TrendDirection,
    /// Monthly satisfaction trend
    pub monthly_trend: TrendDirection,
    /// Satisfaction by user segment
    pub segment_trends: HashMap<String, TrendDirection>,
    /// Satisfaction volatility
    pub volatility_score: f32,
}

/// Satisfaction drivers analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatisfactionDrivers {
    /// Top positive drivers
    pub positive_drivers: Vec<SatisfactionDriver>,
    /// Top negative drivers
    pub negative_drivers: Vec<SatisfactionDriver>,
    /// Driver importance scores
    pub driver_importance: HashMap<String, f32>,
}

/// Individual satisfaction driver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatisfactionDriver {
    /// Driver name
    pub name: String,
    /// Impact on satisfaction
    pub impact_score: f32,
    /// Driver frequency
    pub frequency: f32,
    /// Driver trend
    pub trend: TrendDirection,
}

/// User feedback analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackAnalysis {
    /// Common themes in feedback
    pub common_themes: Vec<FeedbackTheme>,
    /// Sentiment analysis results
    pub sentiment_analysis: SentimentAnalysis,
    /// Feature requests from feedback
    pub feature_requests: Vec<FeatureRequest>,
    /// Issue reports from feedback
    pub issue_reports: Vec<IssueReport>,
}

/// Feedback theme analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackTheme {
    /// Theme name
    pub theme: String,
    /// Theme frequency
    pub frequency: u32,
    /// Theme sentiment score
    pub sentiment_score: f32,
    /// Related keywords
    pub keywords: Vec<String>,
}

/// Sentiment analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAnalysis {
    /// Overall sentiment score (-1.0 to 1.0)
    pub overall_sentiment: f32,
    /// Positive feedback percentage
    pub positive_percentage: f32,
    /// Neutral feedback percentage
    pub neutral_percentage: f32,
    /// Negative feedback percentage
    pub negative_percentage: f32,
    /// Sentiment trends
    pub sentiment_trends: HashMap<String, f32>,
}

/// Feature request from feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureRequest {
    /// Request description
    pub description: String,
    /// Request frequency
    pub frequency: u32,
    /// Request priority score
    pub priority_score: f32,
    /// Related user segments
    pub user_segments: Vec<String>,
}

/// Issue report from feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueReport {
    /// Issue description
    pub description: String,
    /// Issue frequency
    pub frequency: u32,
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue category
    pub category: String,
}

/// Engagement predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementPredictions {
    /// Predicted engagement trend
    pub predicted_trend: TrendDirection,
    /// Predicted engagement score (next 30 days)
    pub predicted_score_30d: f32,
    /// Confidence interval
    pub confidence_interval: (f32, f32),
    /// Key factors influencing prediction
    pub influencing_factors: Vec<String>,
}

/// Satisfaction predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatisfactionPredictions {
    /// Predicted satisfaction score
    pub predicted_satisfaction: f32,
    /// Areas likely to impact satisfaction
    pub impact_areas: Vec<String>,
    /// Recommended interventions
    pub recommended_interventions: Vec<String>,
}
