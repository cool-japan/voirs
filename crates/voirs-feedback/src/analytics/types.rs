//! Core types and data structures for analytics

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Analytics error types
#[derive(Debug, thiserror::Error)]
pub enum AnalyticsError {
    #[error("Data collection failed: {message}")]
    /// Raised when analytics data collection cannot complete successfully.
    DataCollectionError {
        /// Human-readable reason for the data collection failure.
        message: String,
    },

    #[error("Report generation failed: {message}")]
    /// Raised when analytics report generation encounters an unrecoverable error.
    ReportGenerationError {
        /// Human-readable reason for the report generation failure.
        message: String,
    },

    #[error("Invalid query parameters: {message}")]
    /// Returned when a consumer provides invalid query parameters.
    InvalidQueryError {
        /// Details about why the query parameters were rejected.
        message: String,
    },

    #[error("Insufficient data for analysis: {message}")]
    /// Indicates that analytics could not proceed due to missing or insufficient data.
    InsufficientDataError {
        /// Explanation of the missing data condition.
        message: String,
    },
}

/// Result type for analytics operations
pub type AnalyticsResult<T> = Result<T, AnalyticsError>;

/// Analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Enable analytics collection
    pub enabled: bool,
    /// Maximum number of interactions to store
    pub max_interactions: usize,
    /// Maximum number of performance records to store
    pub max_performance_records: usize,
    /// Data retention period in days
    pub retention_days: i64,
    /// Enable real-time analytics
    pub enable_realtime: bool,
    /// Maximum number of active sessions
    pub max_active_sessions: Option<usize>,
    /// Analytics export formats
    pub export_formats: Vec<ExportFormat>,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_interactions: 100_000,
            max_performance_records: 10_000,
            retention_days: 90,
            enable_realtime: true,
            max_active_sessions: Some(1000),
            export_formats: vec![ExportFormat::Json, ExportFormat::Csv],
        }
    }
}

/// User interaction event for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInteractionEvent {
    /// User ID
    pub user_id: String,
    /// Timestamp of interaction
    pub timestamp: DateTime<Utc>,
    /// Type of interaction
    pub interaction_type: InteractionType,
    /// Feature that was used
    pub feature_used: String,
    /// Feedback score (if applicable)
    pub feedback_score: Option<f32>,
    /// Duration of engagement
    pub engagement_duration: std::time::Duration,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl UserInteractionEvent {
    /// Estimate memory usage of this interaction event
    pub fn estimated_memory_size(&self) -> usize {
        self.user_id.len()
            + self.feature_used.len()
            + self
                .metadata
                .iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>()
            + std::mem::size_of::<DateTime<Utc>>()
            + std::mem::size_of::<InteractionType>()
            + std::mem::size_of::<Option<f32>>()
            + std::mem::size_of::<std::time::Duration>()
    }

    /// Intern strings in this interaction to save memory
    pub fn intern_strings(&mut self, pool: &mut StringPool) {
        // Convert user_id and feature_used to use interned strings
        // This would require changing the struct to use Arc<str> in a real implementation
        let _interned_user_id = pool.intern(&self.user_id);
        let _interned_feature = pool.intern(&self.feature_used);

        // For now, just track the optimization opportunity
        log::trace!(
            "Interned strings for interaction: user={}, feature={}",
            self.user_id,
            self.feature_used
        );
    }
}

/// Types of user interactions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionType {
    /// User started a practice session
    Practice,
    /// User completed an exercise
    ExerciseCompleted,
    /// User viewed feedback
    FeedbackViewed,
    /// User changed settings
    SettingsChanged,
    /// User earned an achievement
    AchievementEarned,
    /// User shared content
    ContentShared,
}

/// Performance metrics for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Timestamp of measurement
    pub timestamp: DateTime<Utc>,
    /// Response latency in milliseconds
    pub latency_ms: f32,
    /// System throughput (requests per second)
    pub throughput: f32,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f32,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// CPU usage percentage
    pub cpu_usage: f32,
}

impl PerformanceMetrics {
    /// Estimate memory usage of this performance metrics entry
    pub fn estimated_memory_size(&self) -> usize {
        std::mem::size_of::<Self>() // All fields are fixed-size types
    }
}

/// Session data for analytics
#[derive(Debug, Clone)]
pub struct SessionData {
    /// User ID
    pub user_id: String,
    /// Number of sessions
    pub session_count: u32,
    /// Total duration across all sessions
    pub total_duration: i64,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
    /// First session timestamp
    pub first_session: DateTime<Utc>,
}

impl SessionData {
    /// Create new session data
    pub fn new(user_id: &str) -> Self {
        let now = Utc::now();
        Self {
            user_id: user_id.to_string(),
            session_count: 0,
            total_duration: 0,
            last_activity: now,
            first_session: now,
        }
    }

    /// Update session data from interaction
    pub fn update_from_interaction(&mut self, interaction: &UserInteractionEvent) {
        self.last_activity = interaction.timestamp;
        self.total_duration += interaction.engagement_duration.as_secs() as i64;

        // Count new sessions (simplified logic)
        if interaction.interaction_type == InteractionType::Practice {
            self.session_count += 1;
        }
    }

    /// Estimate memory usage of this session data
    pub fn estimated_memory_size(&self) -> usize {
        self.user_id.len() +
        std::mem::size_of::<u32>() + // session_count
        std::mem::size_of::<i64>() + // total_duration
        std::mem::size_of::<DateTime<Utc>>() * 2 // last_activity + first_session
    }

    /// Intern strings in this session data
    pub fn intern_strings(&mut self, pool: &mut StringPool) {
        let _interned_user_id = pool.intern(&self.user_id);
        log::trace!("Interned session user_id: {}", self.user_id);
    }

    /// Check if this session is important and should be retained
    pub fn is_important(&self) -> bool {
        // Keep sessions with high engagement or recent activity
        self.total_duration > 3600 || // More than 1 hour total
        self.session_count > 10 ||     // More than 10 sessions
        self.last_activity > chrono::Utc::now() - chrono::Duration::minutes(30) // Recent activity
    }

    /// Compress historical data in this session
    pub fn compress_history(&mut self) {
        // This would compress old session events into summaries
        // For now, just acknowledge the compression
        log::trace!("Compressed history for session: {}", self.user_id);
    }
}

/// Analytics query parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsQuery {
    /// Start time for the query
    pub start_time: Option<DateTime<Utc>>,
    /// End time for the query
    pub end_time: Option<DateTime<Utc>>,
    /// Filter by user ID
    pub user_id: Option<String>,
    /// Filter by interaction type
    pub interaction_type: Option<InteractionType>,
    /// Filter by feature
    pub feature: Option<String>,
    /// Include performance metrics
    pub include_performance: bool,
    /// Aggregation level
    pub aggregation: AggregationLevel,
}

/// Aggregation levels for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationLevel {
    /// Raw data
    Raw,
    /// Hourly aggregation
    Hourly,
    /// Daily aggregation
    Daily,
    /// Weekly aggregation
    Weekly,
    /// Monthly aggregation
    Monthly,
}

/// Export formats for analytics data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
}

/// Real-time dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    /// Timestamp of dashboard data
    pub timestamp: DateTime<Utc>,
    /// Number of active sessions
    pub active_sessions: usize,
    /// Recent interactions count
    pub recent_interactions: usize,
    /// Daily active users
    pub daily_active_users: usize,
    /// Average system latency
    pub average_latency_ms: f32,
    /// System health score (0.0 to 1.0)
    pub system_health: f32,
    /// Performance trends
    pub performance_trends: Vec<TrendPoint>,
    /// User satisfaction score
    pub user_satisfaction: f32,
}

/// Trend point for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Latency value
    pub latency: f32,
    /// Throughput value
    pub throughput: f32,
    /// Error rate value
    pub error_rate: f32,
}

/// User-specific analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAnalytics {
    /// User ID
    pub user_id: String,
    /// Total number of sessions
    pub total_sessions: u32,
    /// Total interactions
    pub total_interactions: usize,
    /// Average session duration
    pub average_session_duration: i64,
    /// User improvement trend
    pub improvement_trend: f32,
    /// Preferred features
    pub preferred_features: HashMap<String, u32>,
    /// Learning velocity
    pub learning_velocity: f32,
    /// User engagement score
    pub engagement_score: f32,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
}

/// System-wide usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePatterns {
    /// Timestamp of analysis
    pub timestamp: DateTime<Utc>,
    /// Peak usage hours
    pub peak_usage_hours: Vec<u8>,
    /// Feature usage distribution
    pub feature_usage_distribution: HashMap<String, f32>,
    /// User journey patterns
    pub user_journey_patterns: Vec<UserJourney>,
    /// Geographic distribution
    pub geographic_distribution: HashMap<String, u32>,
    /// Platform usage breakdown
    pub platform_usage_breakdown: HashMap<String, u32>,
    /// User retention rates
    pub retention_rates: RetentionRates,
    /// Conversion funnel analysis
    pub conversion_funnel: ConversionFunnel,
}

/// User journey analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserJourney {
    /// User ID
    pub user_id: String,
    /// Journey steps
    pub steps: Vec<JourneyStep>,
    /// Total journey duration
    pub total_duration: i64,
    /// Completion rate
    pub completion_rate: f32,
}

/// Individual step in user journey
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JourneyStep {
    /// Step timestamp
    pub timestamp: DateTime<Utc>,
    /// User action
    pub action: InteractionType,
    /// Feature used
    pub feature: String,
    /// Outcome score
    pub outcome: f32,
}

/// User retention rates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionRates {
    /// 1-day retention
    pub day_1: f32,
    /// 7-day retention
    pub day_7: f32,
    /// 30-day retention
    pub day_30: f32,
}

/// Conversion funnel analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionFunnel {
    /// Total visitors
    pub visitors: f32,
    /// Signups
    pub signups: f32,
    /// Active users
    pub active_users: f32,
    /// Power users
    pub power_users: f32,
    /// Signup conversion rate
    pub signup_rate: f32,
    /// Activation rate
    pub activation_rate: f32,
    /// Retention rate
    pub retention_rate: f32,
}

/// Complete analytics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsReport {
    /// Report ID
    pub report_id: String,
    /// Generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Report time range
    pub time_range: TimeRange,
    /// Executive summary
    pub executive_summary: ExecutiveSummary,
    /// User engagement analysis
    pub user_engagement: UserEngagementReport,
    /// Performance analysis
    pub performance_analysis: PerformanceAnalysis,
    /// Feature usage report
    pub feature_usage: FeatureUsageReport,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Time range for reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time
    pub start: DateTime<Utc>,
    /// End time
    pub end: DateTime<Utc>,
}

/// Executive summary section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutiveSummary {
    /// Total interactions
    pub total_interactions: usize,
    /// Unique users
    pub unique_users: usize,
    /// Average performance
    pub average_performance_ms: f32,
    /// User satisfaction score
    pub user_satisfaction_score: f32,
    /// Key insights
    pub key_insights: Vec<String>,
}

/// User engagement report section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserEngagementReport {
    /// Total sessions
    pub total_sessions: usize,
    /// Average session duration
    pub average_session_duration: Duration,
    /// Completion rate
    pub completion_rate: f32,
    /// Most popular features
    pub most_popular_features: Vec<String>,
    /// Engagement trends
    pub engagement_trends: Vec<TrendPoint>,
}

/// Performance analysis section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    /// Average latency
    pub average_latency_ms: f32,
    /// 95th percentile latency
    pub p95_latency_ms: f32,
    /// Error rate
    pub error_rate: f32,
    /// Throughput
    pub throughput_rps: f32,
    /// Performance trends
    pub performance_trends: Vec<TrendPoint>,
}

/// Feature usage report section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureUsageReport {
    /// Total feature interactions
    pub total_feature_interactions: u32,
    /// Usage by feature
    pub usage_by_feature: HashMap<String, f32>,
    /// Trending features
    pub trending_features: Vec<String>,
}

/// Compressed summary of interactions for memory optimization
#[derive(Debug, Clone, Default)]
pub struct InteractionSummary {
    /// Time period covered by this summary
    pub time_period: std::ops::Range<DateTime<Utc>>,
    /// Total number of interactions summarized
    pub interaction_count: u64,
    /// Unique user count
    pub user_count: u64,
    /// Interaction type distribution
    pub type_distribution: HashMap<InteractionType, u64>,
    /// Average feedback score
    pub avg_feedback_score: f32,
    /// Total engagement duration
    pub total_engagement: Duration,
    /// Most common features used
    pub top_features: Vec<(String, u64)>,
}

impl InteractionSummary {
    /// Create a new empty summary
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an interaction to this summary
    pub fn add_interaction(&mut self, interaction: &UserInteractionEvent) {
        if self.interaction_count == 0 {
            self.time_period = interaction.timestamp..interaction.timestamp;
        } else {
            self.time_period.start = self.time_period.start.min(interaction.timestamp);
            self.time_period.end = self.time_period.end.max(interaction.timestamp);
        }

        self.interaction_count += 1;
        *self
            .type_distribution
            .entry(interaction.interaction_type.clone())
            .or_insert(0) += 1;

        if let Some(score) = interaction.feedback_score {
            let total = self.avg_feedback_score * (self.interaction_count - 1) as f32 + score;
            self.avg_feedback_score = total / self.interaction_count as f32;
        }

        self.total_engagement = self.total_engagement
            + chrono::Duration::from_std(interaction.engagement_duration).unwrap_or_default();
    }

    /// Check if summary is empty
    pub fn is_empty(&self) -> bool {
        self.interaction_count == 0
    }

    /// Get memory footprint of this summary
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.type_distribution.len()
                * (std::mem::size_of::<InteractionType>() + std::mem::size_of::<u64>())
            + self
                .top_features
                .iter()
                .map(|(s, _)| s.len())
                .sum::<usize>()
    }
}

/// String interning pool for memory optimization
#[derive(Debug, Default)]
pub struct StringPool {
    /// Pool of interned strings
    strings: HashMap<String, Arc<str>>,
    /// Usage statistics
    stats: StringPoolStats,
}

impl StringPool {
    /// Create a new string pool
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a string, returning a shared reference
    pub fn intern(&mut self, string: &str) -> Arc<str> {
        self.stats.total_requests += 1;

        if let Some(interned) = self.strings.get(string) {
            self.stats.cache_hits += 1;
            interned.clone()
        } else {
            let interned: Arc<str> = string.into();
            self.strings.insert(string.to_string(), interned.clone());
            self.stats.unique_strings += 1;
            interned
        }
    }

    /// Get statistics about the string pool
    pub fn stats(&self) -> &StringPoolStats {
        &self.stats
    }

    /// Calculate memory savings from interning
    pub fn memory_savings(&self) -> usize {
        let total_requests = self.stats.total_requests as usize;
        let unique_strings = self.stats.unique_strings as usize;
        let avg_string_size =
            self.strings.keys().map(|s| s.len()).sum::<usize>() / unique_strings.max(1);

        // Memory saved = (total requests - unique strings) * average string size
        (total_requests.saturating_sub(unique_strings)) * avg_string_size
    }
}

/// Statistics for string pool usage
#[derive(Debug, Default, Clone)]
pub struct StringPoolStats {
    /// Total string interning requests
    pub total_requests: u64,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of unique strings stored
    pub unique_strings: u64,
}

impl StringPoolStats {
    /// Calculate cache hit ratio
    pub fn hit_ratio(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_requests as f64
        }
    }
}
