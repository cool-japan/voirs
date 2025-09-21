//! Core types and data structures for progress tracking
//!
//! This module contains the fundamental types, enums, and data structures
//! used throughout the progress tracking system.

use crate::adaptive::RecommendationType;
use crate::traits::{
    Achievement, AchievementTier, FocusArea, Goal, GoalMetric, ProgressSnapshot, TimeRange,
};
use crate::FeedbackError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

// ============================================================================
// Analytics Types
// ============================================================================

/// Configuration for analytics system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Whether to enable detailed analytics
    pub enable_detailed_analytics: bool,
    /// Retention period for analytics data
    pub data_retention_days: u32,
    /// Maximum number of metrics to store in memory
    pub max_metrics_capacity: usize,
    /// Memory limit in bytes for analytics storage
    pub memory_limit_bytes: usize,
    /// Cleanup interval in minutes
    pub cleanup_interval_minutes: u32,
    /// Memory usage threshold for triggering cleanup (0.0 to 1.0)
    pub memory_cleanup_threshold: f64,
    /// Enable automatic aggregation of metrics
    pub enable_auto_aggregation: bool,
    /// Maximum number of aggregated metrics to keep
    pub max_aggregated_metrics: usize,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_detailed_analytics: true,
            data_retention_days: 90,
            max_metrics_capacity: 10_000,
            memory_limit_bytes: 50 * 1024 * 1024, // 50MB
            cleanup_interval_minutes: 60,         // 1 hour
            memory_cleanup_threshold: 0.8,        // 80% memory usage
            enable_auto_aggregation: true,
            max_aggregated_metrics: 1_000,
        }
    }
}

/// Analytics metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsMetric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Metric timestamp
    pub timestamp: DateTime<Utc>,
    /// Metric type
    pub metric_type: MetricType,
}

/// Metric type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    /// Counter metric
    Counter,
    /// Gauge metric
    Gauge,
    /// Histogram metric
    Histogram,
    /// Timer metric
    Timer,
}

/// Comprehensive analytics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveAnalyticsReport {
    /// Report timestamp
    pub timestamp: DateTime<Utc>,
    /// Metrics included in the report
    pub metrics: Vec<AnalyticsMetric>,
    /// Summary statistics
    pub summary: AnalyticsSummary,
}

/// Analytics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsSummary {
    /// Total metrics count
    pub total_metrics: usize,
    /// Average metric value
    pub average_value: f64,
    /// Time range covered
    pub time_range: TimeRange,
}

/// Statistical significance result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSignificanceResult {
    /// P-value of the test
    pub p_value: f64,
    /// Whether result is statistically significant
    pub is_significant: bool,
    /// Confidence level used
    pub confidence_level: f64,
    /// Effect size
    pub effect_size: f64,
}

/// Comparative analytics result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeAnalyticsResult {
    /// Baseline metric value
    pub baseline_value: f64,
    /// Comparison metric value
    pub comparison_value: f64,
    /// Percentage change
    pub percentage_change: f64,
    /// Statistical significance
    pub statistical_significance: StatisticalSignificanceResult,
}

/// Longitudinal study data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongitudinalStudyData {
    /// Study period
    pub study_period: TimeRange,
    /// Data points collected
    pub data_points: Vec<LongitudinalDataPoint>,
    /// Trend analysis
    pub trend_analysis: TrendAnalysis,
}

/// Longitudinal data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongitudinalDataPoint {
    /// Data point timestamp
    pub timestamp: DateTime<Utc>,
    /// Metric value
    pub value: f64,
    /// Associated metadata
    pub metadata: HashMap<String, String>,
}

/// Trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Overall trend direction
    pub trend_direction: crate::progress::analytics::TrendDirection,
    /// Trend slope
    pub slope: f64,
    /// R-squared correlation
    pub r_squared: f64,
}

/// Advanced trend analytics
#[derive(Debug, Clone)]
pub struct TrendAnalytics {
    /// Rate of improvement over time
    pub improvement_velocity: f32,
    /// Stability of performance (inverse of variation)
    pub performance_stability: f32,
    /// Overall trend direction
    pub trend_direction: crate::progress::analytics::TrendDirection,
    /// Linear regression slope
    pub slope: f32,
    /// Correlation coefficient (R-squared)
    pub r_squared: f32,
}

// ============================================================================
// Achievement and Progress Analysis Types
// ============================================================================

/// Achievement definition with unlock conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchievementDefinition {
    /// Unique achievement ID
    pub id: String,
    /// Achievement name
    pub name: String,
    /// Description
    pub description: String,
    /// Condition to unlock
    pub condition: AchievementCondition,
    /// Achievement tier
    pub tier: AchievementTier,
    /// Points awarded
    pub points: u32,
}

/// Achievement unlock conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AchievementCondition {
    /// Complete a number of sessions
    SessionCount(usize),
    /// Reach a skill level
    SkillLevel(f32),
    /// Maintain a streak
    Streak(usize),
    /// Master a specific area
    AreaMastery(FocusArea, f32),
    /// Total training time
    TrainingTime(Duration),
}

/// Detailed progress report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedProgressReport {
    /// User ID
    pub user_id: String,
    /// Report period
    pub period: TimeRange,
    /// Overall improvement
    pub overall_improvement: f32,
    /// Area-specific improvements
    pub area_improvements: HashMap<FocusArea, f32>,
    /// Skill trends
    pub skill_trends: HashMap<FocusArea, crate::progress::analytics::TrendDirection>,
    /// Session analytics
    pub session_analytics: crate::progress::metrics::SessionAnalytics,
    /// Consistency metrics
    pub consistency_metrics: crate::progress::metrics::ConsistencyMetrics,
    /// Achievement progress
    pub achievement_progress: Vec<AchievementAnalysis>,
    /// Goal analysis
    pub goal_analysis: Vec<GoalAnalysis>,
    /// Recommendations
    pub recommendations: Vec<ProgressRecommendation>,
    /// Comparative analysis
    pub comparative_analysis: ComparativeAnalysis,
}

/// Achievement analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchievementAnalysis {
    /// Achievement ID
    pub achievement_id: String,
    /// Achievement name
    pub name: String,
    /// Current progress [0.0, 1.0]
    pub current_progress: f32,
    /// Whether unlocked
    pub is_unlocked: bool,
    /// Estimated time to unlock
    pub estimated_time_to_unlock: Option<Duration>,
}

/// Goal analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalAnalysis {
    /// Goal being analyzed
    pub goal: Goal,
    /// Current value
    pub current_value: f32,
    /// Progress percentage
    pub progress_percentage: f32,
    /// Whether on track
    pub on_track: bool,
    /// Estimated completion
    pub estimated_completion: Option<DateTime<Utc>>,
}

/// Progress recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressRecommendation {
    /// Type of recommendation
    pub recommendation_type: RecommendationType,
    /// Recommendation title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Priority level [0.0, 1.0]
    pub priority: f32,
    /// Estimated impact [0.0, 1.0]
    pub estimated_impact: f32,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Comparative analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeAnalysis {
    /// User's percentile ranking
    pub user_percentile: f32,
    /// Average user score
    pub average_user_score: f32,
    /// User's current score
    pub user_score: f32,
    /// Improvement rate vs average
    pub improvement_rate_vs_average: f32,
    /// User's strengths vs peers
    pub strengths_vs_peers: Vec<String>,
    /// Areas for improvement
    pub areas_for_improvement: Vec<String>,
}

/// Learning pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPatternAnalysis {
    /// Learning velocity
    pub learning_velocity: f32,
    /// Optimal session length
    pub optimal_session_length: Duration,
    /// Peak performance times
    pub peak_performance_times: Vec<PeakPerformanceTime>,
    /// Difficulty preference
    pub difficulty_preference: DifficultyPreference,
    /// Focus area patterns
    pub focus_area_patterns: HashMap<FocusArea, FocusPattern>,
    /// Consistency patterns
    pub consistency_patterns: ConsistencyPattern,
    /// Learning plateaus
    pub improvement_plateaus: Vec<LearningPlateau>,
    /// Learning style indicators
    pub learning_style_indicators: LearningStyleProfile,
}

/// Peak performance time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakPerformanceTime {
    /// Time range description
    pub time_range: String,
    /// Performance boost factor
    pub performance_boost: f32,
    /// Confidence in this finding
    pub confidence: f32,
}

/// Difficulty preference profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyPreference {
    /// Preferred difficulty level [0.0, 1.0]
    pub preferred_level: f32,
    /// Adaptability to difficulty changes
    pub adaptability: f32,
    /// Challenge seeking tendency
    pub challenge_seeking: f32,
}

/// Focus pattern for specific areas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusPattern {
    /// How often this area receives attention
    pub attention_frequency: f32,
    /// Rate of improvement in this area
    pub improvement_rate: f32,
    /// Tendency to plateau in this area
    pub plateau_tendency: f32,
}

/// Consistency pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyPattern {
    /// Overall consistency score
    pub overall_consistency: f32,
    /// Performance variance
    pub performance_variance: f32,
    /// Tendency to maintain streaks
    pub streak_tendency: f32,
}

/// Learning plateau identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPlateau {
    /// When plateau started
    pub start_date: DateTime<Utc>,
    /// When plateau ended (if applicable)
    pub end_date: DateTime<Utc>,
    /// Skill area affected
    pub skill_area: FocusArea,
    /// Plateau performance level
    pub plateau_level: f32,
    /// Suggestions to break through
    pub breakthrough_suggestions: Vec<String>,
}

/// Learning style profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStyleProfile {
    /// Visual learning preference
    pub visual_preference: f32,
    /// Auditory learning preference
    pub auditory_preference: f32,
    /// Kinesthetic learning preference
    pub kinesthetic_preference: f32,
    /// Preference for structured approach
    pub structured_preference: f32,
    /// Preference for experimental approach
    pub experimental_preference: f32,
}

/// Achievement progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchievementProgress {
    /// Achievement definition
    pub achievement_definition: AchievementDefinition,
    /// Current progress [0.0, 1.0]
    pub current_progress: f32,
    /// Whether unlocked
    pub is_unlocked: bool,
    /// Unlock date if unlocked
    pub unlock_date: Option<DateTime<Utc>>,
}

// ============================================================================
// Milestone and Streak Types
// ============================================================================

/// Adaptive milestone system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveMilestone {
    /// Unique milestone identifier
    pub milestone_id: String,
    /// Milestone title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Type of milestone
    pub milestone_type: MilestoneType,
    /// Achievement criteria
    pub criteria: MilestoneCriteria,
    /// Estimated time to completion
    pub estimated_duration: Duration,
    /// Difficulty level [0.0, 1.0]
    pub difficulty: f32,
    /// Motivational impact assessment [0.0, 1.0]
    pub motivational_impact: f32,
    /// Personalized message for user
    pub personalized_message: String,
    /// Prerequisites needed
    pub prerequisites: Vec<String>,
    /// Rewards for completion
    pub rewards: Vec<MilestoneReward>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Achievement timestamp
    pub achieved_at: Option<DateTime<Utc>>,
}

/// Types of milestones
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MilestoneType {
    /// Skill improvement milestone
    SkillImprovement,
    /// Consistency goal
    ConsistencyGoal,
    /// Achievement unlock
    AchievementGoal,
    /// Overall progress milestone
    ProgressGoal,
    /// Custom milestone
    Custom,
}

/// Milestone achievement criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MilestoneCriteria {
    /// Reach specific skill level in focus area
    SkillLevel {
        /// The focus area to improve
        focus_area: FocusArea,
        /// Target skill level to achieve
        target_level: f32,
    },
    /// Complete number of sessions
    SessionCount {
        /// Target number of sessions
        target_sessions: usize,
    },
    /// Maintain streak
    Streak {
        /// Target streak length
        target_streak: usize,
    },
    /// Reach overall skill level
    OverallSkill {
        /// Target overall skill level
        target_level: f32,
    },
    /// Complete training time
    TrainingTime {
        /// Target training duration
        target_duration: Duration,
    },
}

/// Milestone rewards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MilestoneReward {
    /// Points awarded
    Points(u32),
    /// Badge unlocked
    Badge(String),
    /// Feature unlocked
    UnlockFeature(String),
    /// Certificate earned
    Certificate(String),
    /// Custom reward
    Custom(String),
}

/// Comprehensive streak analysis system
#[derive(Debug, Clone, Default)]
pub struct ComprehensiveStreakAnalysis {
    /// Current active streaks
    pub current_streaks: HashMap<StreakType, CurrentStreak>,
    /// Historical streak data
    pub historical_streaks: Vec<HistoricalStreak>,
    /// Streak patterns and insights
    pub streak_patterns: StreakPatterns,
    /// Motivation maintenance strategies
    pub motivation_maintenance: MotivationMaintenance,
    /// Recovery mechanism recommendations
    pub recovery_mechanisms: Vec<RecoveryMechanism>,
    /// Achievement potential assessment
    pub achievement_potential: f32,
}

/// Types of streaks tracked
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreakType {
    /// Daily practice streak
    Practice,
    /// Quality performance streak
    Quality,
    /// Improvement streak
    Improvement,
    /// Consistency streak
    Consistency,
}

/// Current active streak
#[derive(Debug, Clone)]
pub struct CurrentStreak {
    /// Type of streak
    pub streak_type: StreakType,
    /// Current streak count
    pub current_count: usize,
    /// When streak started
    pub start_date: DateTime<Utc>,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
    /// Streak strength assessment [0.0, 1.0]
    pub strength: f32,
}

/// Historical streak record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalStreak {
    /// Type of streak
    pub streak_type: StreakType,
    /// Length of the streak
    pub length: usize,
    /// Start date
    pub start_date: DateTime<Utc>,
    /// End date
    pub end_date: DateTime<Utc>,
    /// Peak performance during streak
    pub peak_performance: f32,
    /// Reason for streak break
    pub break_reason: StreakBreakReason,
}

/// Streak break reasons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreakBreakReason {
    /// Time constraints
    TimeConstraints,
    /// Lack of motivation
    LackOfMotivation,
    /// Technical issues
    Technical,
    /// External circumstances
    External,
    /// Planned break
    Planned,
    /// Unknown reason
    Unknown,
    /// Streak is ongoing
    Ongoing,
}

/// Streak patterns analysis
#[derive(Debug, Clone, Default)]
pub struct StreakPatterns {
    /// Average streak length
    pub average_streak_length: f32,
    /// Longest streak ever achieved
    pub longest_streak_ever: usize,
    /// Common reasons for streak breaks
    pub common_break_reasons: Vec<StreakBreakReason>,
    /// Optimal session times for streaks
    pub optimal_session_times: Vec<String>,
    /// Seasonal variations in streak performance
    pub seasonal_variations: HashMap<String, f32>,
}

/// Motivation maintenance strategies
#[derive(Debug, Clone, Default)]
pub struct MotivationMaintenance {
    /// Current motivation level [0.0, 1.0]
    pub current_motivation_level: f32,
    /// Motivation trend (positive = improving)
    pub motivation_trend: f32,
    /// Burnout risk assessment [0.0, 1.0]
    pub burnout_risk: f32,
    /// Suggested engagement strategies
    pub engagement_strategies: Vec<String>,
}

/// Recovery mechanism for broken streaks
#[derive(Debug, Clone)]
pub struct RecoveryMechanism {
    /// Type of recovery strategy
    pub strategy_type: RecoveryStrategyType,
    /// Description of the mechanism
    pub description: String,
    /// Estimated effectiveness [0.0, 1.0]
    pub estimated_effectiveness: f32,
    /// Time commitment required
    pub time_commitment: Duration,
}

/// Recovery strategy types
#[derive(Debug, Clone)]
pub enum RecoveryStrategyType {
    /// Gradual return to practice
    GradualReturn,
    /// Motivation boost activities
    MotivationBoost,
    /// Social support engagement
    SocialSupport,
    /// Routine adjustment
    RoutineAdjustment,
    /// Goal modification
    GoalModification,
}

/// Streak recovery plan
#[derive(Debug, Clone)]
pub struct StreakRecoveryPlan {
    /// Primary recovery strategy
    pub recovery_strategy: RecoveryStrategyType,
    /// Specific actions to take
    pub suggested_actions: Vec<String>,
    /// Motivation boosting messages
    pub motivation_boosters: Vec<String>,
    /// Recommended milestone adjustments
    pub milestone_adjustments: Vec<String>,
    /// Estimated time to recover streak
    pub estimated_recovery_time: Duration,
    /// Probability of successful recovery [0.0, 1.0]
    pub success_probability: f32,
}

// ============================================================================
// Memory Optimization Types
// ============================================================================

/// Memory-bounded metrics storage with LRU eviction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBoundedMetrics {
    /// Internal storage with bounded capacity
    storage: VecDeque<(String, AnalyticsMetric)>,
    /// Maximum capacity
    capacity: usize,
    /// Quick lookup index
    index: HashMap<String, usize>,
}

impl MemoryBoundedMetrics {
    /// Create new memory-bounded metrics storage
    pub fn new(capacity: usize) -> Self {
        Self {
            storage: VecDeque::with_capacity(capacity),
            capacity,
            index: HashMap::new(),
        }
    }

    /// Insert metric with LRU eviction
    pub fn insert(&mut self, key: String, metric: AnalyticsMetric) {
        // If at capacity, remove oldest entry
        if self.storage.len() >= self.capacity {
            if let Some((old_key, _)) = self.storage.pop_front() {
                self.index.remove(&old_key);
            }
        }

        // Add new entry
        let new_index = self.storage.len();
        self.storage.push_back((key.clone(), metric));
        self.index.insert(key, new_index);

        // Rebuild index if needed (simple approach)
        if self.storage.len() != self.index.len() {
            self.rebuild_index();
        }
    }

    /// Get metric by key
    pub fn get(&self, key: &str) -> Option<&AnalyticsMetric> {
        if let Some(&index) = self.index.get(key) {
            if index < self.storage.len() {
                return Some(&self.storage[index].1);
            }
        }
        None
    }

    /// Remove metrics before given timestamp
    pub fn cleanup_before(&mut self, cutoff_time: DateTime<Utc>) {
        let mut removed_count = 0;

        // Remove old entries from front
        while let Some((key, metric)) = self.storage.front() {
            if metric.timestamp < cutoff_time {
                let (removed_key, _) = self.storage.pop_front().unwrap();
                self.index.remove(&removed_key);
                removed_count += 1;
            } else {
                break;
            }
        }

        // Rebuild index if we removed items
        if removed_count > 0 {
            self.rebuild_index();
        }
    }

    /// Rebuild index after modifications
    fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, (key, _)) in self.storage.iter().enumerate() {
            self.index.insert(key.clone(), i);
        }
    }

    /// Get number of stored metrics
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Aggregated metric for long-term storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetric {
    /// Metric name
    pub name: String,
    /// Count of data points
    pub count: u64,
    /// Sum of all values
    pub sum: f64,
    /// Sum of squares for variance calculation
    pub sum_of_squares: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
    /// Metric type
    pub metric_type: MetricType,
}

impl AggregatedMetric {
    /// Calculate mean
    pub fn mean(&self) -> f64 {
        if self.count > 0 {
            self.sum / self.count as f64
        } else {
            0.0
        }
    }

    /// Calculate variance
    pub fn variance(&self) -> f64 {
        if self.count > 1 {
            let mean = self.mean();
            (self.sum_of_squares - self.count as f64 * mean * mean) / (self.count - 1) as f64
        } else {
            0.0
        }
    }

    /// Calculate standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total number of metrics
    pub total_metrics: usize,
    /// Number of aggregated metrics
    pub aggregated_metrics: usize,
    /// Estimated memory usage in bytes
    pub estimated_memory_bytes: usize,
    /// Memory limit in bytes
    pub memory_limit_bytes: usize,
    /// Memory utilization percentage (0.0 to 1.0)
    pub memory_utilization: f64,
}

/// Memory-optimized circular buffer for historical data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularProgressHistory {
    /// Fixed-size buffer for progress snapshots
    buffer: Vec<ProgressSnapshot>,
    /// Current write position
    write_pos: usize,
    /// Number of items written (for determining if buffer is full)
    items_written: usize,
    /// Maximum capacity
    capacity: usize,
}

impl CircularProgressHistory {
    /// Create new circular buffer with specified capacity
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, || ProgressSnapshot {
            timestamp: Utc::now(),
            overall_score: 0.0,
            area_scores: HashMap::new(),
            session_count: 0,
            events: Vec::new(),
        });

        Self {
            buffer,
            write_pos: 0,
            items_written: 0,
            capacity,
        }
    }

    /// Add new progress snapshot, overwriting oldest if at capacity
    pub fn push(&mut self, snapshot: ProgressSnapshot) {
        self.buffer[self.write_pos] = snapshot;
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.items_written += 1;
    }

    /// Get most recent snapshots up to specified count
    pub fn get_recent(&self, count: usize) -> Vec<ProgressSnapshot> {
        let actual_count = count.min(self.len());
        let mut result = Vec::with_capacity(actual_count);

        for i in 0..actual_count {
            let pos = if self.write_pos >= i + 1 {
                self.write_pos - i - 1
            } else {
                self.capacity - (i + 1 - self.write_pos)
            };
            result.push(self.buffer[pos].clone());
        }

        result
    }

    /// Get number of items stored (up to capacity)
    pub fn len(&self) -> usize {
        self.items_written.min(self.capacity)
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.items_written == 0
    }

    /// Get memory usage in bytes (approximate)
    pub fn memory_usage(&self) -> usize {
        self.capacity * std::mem::size_of::<ProgressSnapshot>()
    }
}

/// Memory-optimized progress tracking
#[derive(Debug, Clone)]
pub struct MemoryOptimizedProgress {
    /// User ID
    pub user_id: String,
    /// Current overall skill level
    pub overall_skill_level: f32,
    /// Compressed skill statistics instead of full breakdown
    pub skill_stats: HashMap<FocusArea, crate::progress::skills::CompressedSkillStats>,
    /// Circular buffer for recent progress history
    pub progress_history: CircularProgressHistory,
    /// Compressed achievement data
    pub achievement_summary: AchievementSummary,
    /// Essential training stats only
    pub training_stats: EssentialTrainingStats,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Essential training statistics for memory efficiency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EssentialTrainingStats {
    /// Total sessions
    pub total_sessions: u32,
    /// Total practice time in seconds
    pub total_practice_time_secs: u64,
    /// Current streak
    pub current_streak: u16,
    /// Best streak ever
    pub best_streak: u16,
    /// Average session duration in seconds
    pub avg_session_duration_secs: u32,
}

/// Compressed achievement summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchievementSummary {
    /// Total achievements earned
    pub total_earned: u16,
    /// Achievements by tier
    pub by_tier: HashMap<AchievementTier, u16>,
    /// Recent achievements (last 5)
    pub recent: Vec<String>,
}

/// Memory management utilities for analytics
pub struct AnalyticsMemoryManager {
    /// Memory limit in bytes
    memory_limit: usize,
    /// Current memory usage estimate
    current_usage: usize,
    /// Cleanup thresholds
    cleanup_threshold: f64, // Percentage (0.0 to 1.0)
}

impl AnalyticsMemoryManager {
    /// Create new memory manager with specified limit
    pub fn new(memory_limit_mb: usize) -> Self {
        Self {
            memory_limit: memory_limit_mb * 1024 * 1024,
            current_usage: 0,
            cleanup_threshold: 0.8, // Cleanup when 80% full
        }
    }

    /// Update current memory usage estimate
    pub fn update_usage(&mut self, new_usage: usize) {
        self.current_usage = new_usage;
    }

    /// Check if cleanup is needed
    pub fn needs_cleanup(&self) -> bool {
        self.current_usage as f64 / self.memory_limit as f64 > self.cleanup_threshold
    }

    /// Get memory utilization percentage
    pub fn utilization(&self) -> f64 {
        self.current_usage as f64 / self.memory_limit as f64
    }

    /// Suggest cleanup actions
    pub fn suggest_cleanup_actions(&self) -> Vec<CleanupAction> {
        let mut actions = Vec::new();

        if self.utilization() > 0.9 {
            actions.push(CleanupAction::CompressOldData);
            actions.push(CleanupAction::RemoveOldestMetrics);
        } else if self.utilization() > 0.8 {
            actions.push(CleanupAction::CompressOldData);
        } else if self.utilization() > 0.7 {
            actions.push(CleanupAction::AggregateOldMetrics);
        }

        actions
    }
}

/// Cleanup actions for memory management
#[derive(Debug, Clone, PartialEq)]
pub enum CleanupAction {
    /// Compress old historical data
    CompressOldData,
    /// Remove oldest metrics
    RemoveOldestMetrics,
    /// Aggregate old metrics into summaries
    AggregateOldMetrics,
    /// Switch to memory-optimized representations
    OptimizeDataStructures,
}

/// Extension trait for memory optimization
pub trait MemoryOptimized {
    /// Estimate memory usage in bytes
    fn memory_usage(&self) -> usize;
    /// Compress data to reduce memory usage
    fn compress(&self) -> Self
    where
        Self: Sized;
    /// Check if compression is recommended
    fn should_compress(&self) -> bool;
}
