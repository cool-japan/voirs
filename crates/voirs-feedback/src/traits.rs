//! Core traits for the `VoiRS` feedback system
//!
//! This module defines the fundamental interfaces for feedback generation,
//! adaptive learning, progress tracking, and interactive training capabilities.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;
use voirs_evaluation::{PronunciationScore, QualityScore};
use voirs_sdk::{AudioBuffer, LanguageCode, VoirsError};

/// Result type for feedback operations
pub type FeedbackResult<T> = Result<T, VoirsError>;

// ============================================================================
// Core Data Types
// ============================================================================

/// User feedback with actionable insights
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserFeedback {
    /// Primary feedback message
    pub message: String,
    /// Suggested improvement action
    pub suggestion: Option<String>,
    /// Confidence in the feedback [0.0, 1.0]
    pub confidence: f32,
    /// Overall score for this aspect [0.0, 1.0]
    pub score: f32,
    /// Priority level [0.0, 1.0] (higher = more urgent)
    pub priority: f32,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Comprehensive feedback result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackResponse {
    /// Individual feedback items
    pub feedback_items: Vec<UserFeedback>,
    /// Overall session score [0.0, 1.0]
    pub overall_score: f32,
    /// Immediate next steps
    pub immediate_actions: Vec<String>,
    /// Long-term recommendations
    pub long_term_goals: Vec<String>,
    /// Progress indicators
    pub progress_indicators: ProgressIndicators,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Processing time
    pub processing_time: Duration,
    /// Feedback type for compatibility
    pub feedback_type: FeedbackType,
}

impl Default for FeedbackResponse {
    fn default() -> Self {
        Self {
            feedback_items: Vec::new(),
            overall_score: 0.0,
            immediate_actions: Vec::new(),
            long_term_goals: Vec::new(),
            progress_indicators: ProgressIndicators {
                improving_areas: Vec::new(),
                attention_areas: Vec::new(),
                stable_areas: Vec::new(),
                overall_trend: 0.0,
                completion_percentage: 0.0,
            },
            timestamp: chrono::Utc::now(),
            processing_time: Duration::from_millis(0),
            feedback_type: FeedbackType::Quality,
        }
    }
}

/// Progress indicators for user improvement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgressIndicators {
    /// Areas showing improvement
    pub improving_areas: Vec<String>,
    /// Areas needing attention
    pub attention_areas: Vec<String>,
    /// Stability indicators
    pub stable_areas: Vec<String>,
    /// Trend direction [-1.0, 1.0] (negative = declining, positive = improving)
    pub overall_trend: f32,
    /// Progress percentage [0.0, 100.0]
    pub completion_percentage: f32,
}

impl Default for ProgressIndicators {
    fn default() -> Self {
        Self {
            improving_areas: Vec::new(),
            attention_areas: Vec::new(),
            stable_areas: Vec::new(),
            overall_trend: 0.0,
            completion_percentage: 0.0,
        }
    }
}

/// User session state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionState {
    /// Unique session ID
    pub session_id: Uuid,
    /// User ID
    pub user_id: String,
    /// Session start time
    pub start_time: DateTime<Utc>,
    /// Last activity time
    pub last_activity: DateTime<Utc>,
    /// Current exercise or task
    pub current_task: Option<String>,
    /// Session statistics
    pub stats: SessionStats,
    /// User preferences
    pub preferences: UserPreferences,
    /// Adaptive model state
    pub adaptive_state: AdaptiveState,
    /// Current exercise (for compatibility)
    pub current_exercise: Option<ExerciseSession>,
    /// Session statistics (for compatibility)
    pub session_stats: SessionStatistics,
}

impl SessionState {
    /// Create a new session state
    pub async fn new(user_id: &str) -> FeedbackResult<Self> {
        let now = Utc::now();
        Ok(Self {
            session_id: Uuid::new_v4(),
            user_id: user_id.to_string(),
            start_time: now,
            last_activity: now,
            current_task: None,
            stats: SessionStats::default(),
            preferences: UserPreferences::default(),
            adaptive_state: AdaptiveState::default(),
            current_exercise: None,
            session_stats: SessionStatistics::default(),
        })
    }
}

impl Default for SessionState {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            session_id: Uuid::new_v4(),
            user_id: String::new(),
            start_time: now,
            last_activity: now,
            current_task: None,
            stats: SessionStats::default(),
            preferences: UserPreferences::default(),
            adaptive_state: AdaptiveState::default(),
            current_exercise: None,
            session_stats: SessionStatistics::default(),
        }
    }
}

/// Session statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionStats {
    /// Number of synthesis attempts
    pub synthesis_attempts: usize,
    /// Total feedback items received
    pub feedback_received: usize,
    /// Average quality score
    pub average_quality: f32,
    /// Average pronunciation score
    pub average_pronunciation: f32,
    /// Time spent in session
    pub session_duration: Duration,
    /// Exercises completed
    pub exercises_completed: usize,
}

impl Default for SessionStats {
    fn default() -> Self {
        Self {
            synthesis_attempts: 0,
            feedback_received: 0,
            average_quality: 0.0,
            average_pronunciation: 0.0,
            session_duration: Duration::from_secs(0),
            exercises_completed: 0,
        }
    }
}

/// User preferences for feedback
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserPreferences {
    /// User ID for compatibility
    pub user_id: String,
    /// Preferred feedback style
    pub feedback_style: FeedbackStyle,
    /// Detail level [0.0, 1.0]
    pub detail_level: f32,
    /// Enable encouragement messages
    pub enable_encouragement: bool,
    /// Enable technical details
    pub enable_technical_details: bool,
    /// Preferred language for feedback
    pub feedback_language: LanguageCode,
    /// Areas of focus
    pub focus_areas: Vec<FocusArea>,
    /// Notification preferences
    pub notifications: NotificationPreferences,
    /// Enable audio feedback modality
    pub enable_audio_feedback: bool,
    /// Enable visual feedback modality
    pub enable_visual_feedback: bool,
    /// Enable haptic feedback modality
    pub enable_haptic_feedback: bool,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            user_id: "default_user".to_string(),
            feedback_style: FeedbackStyle::Balanced,
            detail_level: 0.7,
            enable_encouragement: true,
            enable_technical_details: false,
            feedback_language: LanguageCode::EnUs,
            focus_areas: vec![FocusArea::Pronunciation, FocusArea::Naturalness],
            notifications: NotificationPreferences::default(),
            enable_audio_feedback: true,
            enable_visual_feedback: true,
            enable_haptic_feedback: false,
        }
    }
}

/// Feedback style preferences
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeedbackStyle {
    /// Gentle, encouraging feedback
    Gentle,
    /// Direct, concise feedback
    Direct,
    /// Balanced approach
    Balanced,
    /// Detailed technical feedback
    Technical,
    /// Gamified feedback with achievements
    Gamified,
}

/// Areas of focus for improvement
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FocusArea {
    /// Pronunciation accuracy
    Pronunciation,
    /// Speech naturalness
    Naturalness,
    /// Audio quality
    Quality,
    /// Speaking rhythm
    Rhythm,
    /// Stress patterns
    Stress,
    /// Intonation
    Intonation,
    /// Overall fluency
    Fluency,
    /// Breathing and pause patterns
    Breathing,
    /// Speech clarity
    Clarity,
    /// Speech accuracy
    Accuracy,
    /// Speech consistency
    Consistency,
}

/// Notification preferences
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NotificationPreferences {
    /// Enable real-time notifications
    pub enable_realtime: bool,
    /// Enable progress notifications
    pub enable_progress: bool,
    /// Enable achievement notifications
    pub enable_achievements: bool,
    /// Notification frequency
    pub frequency: NotificationFrequency,
}

impl Default for NotificationPreferences {
    fn default() -> Self {
        Self {
            enable_realtime: true,
            enable_progress: true,
            enable_achievements: true,
            frequency: NotificationFrequency::Moderate,
        }
    }
}

/// Notification frequency levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NotificationFrequency {
    /// Minimal notifications
    Minimal,
    /// Moderate notifications
    Moderate,
    /// Frequent notifications
    Frequent,
    /// All notifications
    All,
}

/// Adaptive learning state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveState {
    /// User skill level estimate [0.0, 1.0]
    pub skill_level: f32,
    /// Learning rate estimate [0.0, 1.0]
    pub learning_rate: f32,
    /// Areas of strength
    pub strengths: Vec<FocusArea>,
    /// Areas needing improvement
    pub improvement_areas: Vec<FocusArea>,
    /// Model confidence in estimates [0.0, 1.0]
    pub confidence: f32,
    /// Number of adaptations made
    pub adaptation_count: usize,
    /// Last adaptation time
    pub last_adaptation: Option<DateTime<Utc>>,
}

impl Default for AdaptiveState {
    fn default() -> Self {
        Self {
            skill_level: 0.5,
            learning_rate: 0.5,
            strengths: Vec::new(),
            improvement_areas: Vec::new(),
            confidence: 0.3,
            adaptation_count: 0,
            last_adaptation: None,
        }
    }
}

/// Training exercise definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExercise {
    /// Unique exercise ID
    pub exercise_id: String,
    /// Exercise name
    pub name: String,
    /// Description
    pub description: String,
    /// Difficulty level [0.0, 1.0]
    pub difficulty: f32,
    /// Target focus areas
    pub focus_areas: Vec<FocusArea>,
    /// Exercise type
    pub exercise_type: ExerciseType,
    /// Text to synthesize
    pub target_text: String,
    /// Reference audio (if available)
    pub reference_audio: Option<AudioBuffer>,
    /// Success criteria
    pub success_criteria: SuccessCriteria,
    /// Estimated completion time
    pub estimated_duration: Duration,
}

impl PartialEq for TrainingExercise {
    fn eq(&self, other: &Self) -> bool {
        self.exercise_id == other.exercise_id
            && self.name == other.name
            && self.description == other.description
            && self.difficulty == other.difficulty
            && self.focus_areas == other.focus_areas
            && self.exercise_type == other.exercise_type
            && self.target_text == other.target_text
            && self.success_criteria == other.success_criteria
            && self.estimated_duration == other.estimated_duration
        // Note: reference_audio is excluded from comparison as AudioBuffer doesn't implement PartialEq
    }
}

impl Default for TrainingExercise {
    fn default() -> Self {
        Self {
            exercise_id: "default".to_string(),
            name: "Default Exercise".to_string(),
            description: "A default training exercise".to_string(),
            difficulty: 0.5,
            focus_areas: vec![FocusArea::Pronunciation],
            exercise_type: ExerciseType::FreeForm,
            target_text: "Hello world".to_string(),
            reference_audio: None,
            success_criteria: SuccessCriteria {
                min_quality_score: 0.7,
                min_pronunciation_score: 0.7,
                consistency_required: 1,
                max_attempts: 3,
                time_limit: Some(Duration::from_secs(60)),
            },
            estimated_duration: Duration::from_secs(30),
        }
    }
}

/// Types of training exercises
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExerciseType {
    /// Free-form practice
    FreeForm,
    /// Guided pronunciation practice
    Pronunciation,
    /// Rhythm and timing practice
    Rhythm,
    /// Emotion and expression practice
    Expression,
    /// Technical quality focus
    Quality,
    /// Review of previous exercises
    Review,
    /// Advanced challenging exercises
    Advanced,
    /// Speed and fluency practice
    Fluency,
    /// Challenge mode
    Challenge,
}

/// Success criteria for exercises
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuccessCriteria {
    /// Minimum quality score required
    pub min_quality_score: f32,
    /// Minimum pronunciation score required
    pub min_pronunciation_score: f32,
    /// Maximum attempts allowed
    pub max_attempts: usize,
    /// Time limit (if any)
    pub time_limit: Option<Duration>,
    /// Required consistency (multiple successful attempts)
    pub consistency_required: usize,
}

/// Training exercise result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Exercise that was attempted
    pub exercise: TrainingExercise,
    /// Whether the exercise was completed successfully
    pub success: bool,
    /// Number of attempts made
    pub attempts_made: usize,
    /// Time taken to complete
    pub completion_time: Duration,
    /// Final scores achieved
    pub final_scores: TrainingScores,
    /// Detailed feedback
    pub feedback: FeedbackResponse,
    /// Areas for improvement identified
    pub improvement_recommendations: Vec<String>,
}

/// Scores achieved during training
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingScores {
    /// Quality score [0.0, 1.0]
    pub quality: f32,
    /// Pronunciation score [0.0, 1.0]
    pub pronunciation: f32,
    /// Consistency score [0.0, 1.0]
    pub consistency: f32,
    /// Improvement score [0.0, 1.0]
    pub improvement: f32,
}

impl Default for TrainingScores {
    fn default() -> Self {
        Self {
            quality: 0.0,
            pronunciation: 0.0,
            consistency: 0.0,
            improvement: 0.0,
        }
    }
}

/// User progress tracking data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserProgress {
    /// User ID
    pub user_id: String,
    /// Overall skill level [0.0, 1.0]
    pub overall_skill_level: f32,
    /// Skill breakdown by area
    pub skill_breakdown: HashMap<FocusArea, f32>,
    /// Progress history
    pub progress_history: Vec<ProgressSnapshot>,
    /// Achievements unlocked
    pub achievements: Vec<Achievement>,
    /// Training statistics
    pub training_stats: TrainingStatistics,
    /// Goals and milestones
    pub goals: Vec<Goal>,
    /// Last update time
    pub last_updated: DateTime<Utc>,
    /// Average scores (for compatibility)
    pub average_scores: SessionScores,
    /// Skill levels by area (for compatibility)
    pub skill_levels: HashMap<String, f32>,
    /// Recent sessions (for compatibility)
    pub recent_sessions: Vec<SessionSummary>,
    /// Personal bests (for compatibility)
    pub personal_bests: HashMap<String, f32>,
    /// Session count (for compatibility)
    pub session_count: usize,
    /// Total practice time (for compatibility)
    pub total_practice_time: Duration,
}

impl Default for UserProgress {
    fn default() -> Self {
        Self {
            user_id: String::new(),
            overall_skill_level: 0.0,
            skill_breakdown: HashMap::new(),
            progress_history: Vec::new(),
            achievements: Vec::new(),
            training_stats: TrainingStatistics::default(),
            goals: Vec::new(),
            last_updated: Utc::now(),
            average_scores: SessionScores::default(),
            skill_levels: HashMap::new(),
            recent_sessions: Vec::new(),
            personal_bests: HashMap::new(),
            session_count: 0,
            total_practice_time: Duration::from_secs(0),
        }
    }
}

/// Point-in-time progress snapshot
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgressSnapshot {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Overall score at this point
    pub overall_score: f32,
    /// Scores by focus area
    pub area_scores: HashMap<FocusArea, f32>,
    /// Session count at this point
    pub session_count: usize,
    /// Notable events
    pub events: Vec<String>,
}

/// Achievement unlocked by user
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Achievement {
    /// Achievement ID
    pub achievement_id: String,
    /// Achievement name
    pub name: String,
    /// Description
    pub description: String,
    /// When it was unlocked
    pub unlocked_at: DateTime<Utc>,
    /// Achievement tier
    pub tier: AchievementTier,
    /// Points awarded
    pub points: u32,
}

/// Achievement tier levels
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AchievementTier {
    /// Bronze tier
    Bronze,
    /// Silver tier
    Silver,
    /// Gold tier
    Gold,
    /// Platinum tier
    Platinum,
    /// Diamond tier
    Diamond,
    /// Rare tier
    Rare,
}

/// Training statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingStatistics {
    /// Total training sessions
    pub total_sessions: usize,
    /// Successful training sessions
    pub successful_sessions: usize,
    /// Total training time
    pub total_training_time: Duration,
    /// Exercises completed
    pub exercises_completed: usize,
    /// Success rate [0.0, 1.0]
    pub success_rate: f32,
    /// Average improvement per session
    pub average_improvement: f32,
    /// Streak information
    pub current_streak: usize,
    /// Longest streak achieved
    pub longest_streak: usize,
}

impl Default for TrainingStatistics {
    fn default() -> Self {
        Self {
            total_sessions: 0,
            successful_sessions: 0,
            total_training_time: Duration::from_secs(0),
            exercises_completed: 0,
            success_rate: 0.0,
            average_improvement: 0.0,
            current_streak: 0,
            longest_streak: 0,
        }
    }
}

/// User goal or milestone
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Goal {
    /// Goal ID
    pub goal_id: String,
    /// Goal description
    pub description: String,
    /// Target metric
    pub target_metric: GoalMetric,
    /// Target value
    pub target_value: f32,
    /// Current progress [0.0, 1.0]
    pub progress: f32,
    /// Goal deadline (if any)
    pub deadline: Option<DateTime<Utc>>,
    /// Whether the goal is active
    pub active: bool,
}

/// Goal metric types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GoalMetric {
    /// Overall skill level
    OverallSkill,
    /// Specific focus area skill
    FocusAreaSkill(FocusArea),
    /// Exercise completion count
    ExerciseCount,
    /// Training session count
    SessionCount,
    /// Training time
    TrainingTime,
    /// Consistency streak
    Streak,
}

/// Session scores for compatibility
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionScores {
    /// Average quality score
    pub average_quality: f32,
    /// Average pronunciation score
    pub average_pronunciation: f32,
    /// Average fluency score
    pub average_fluency: f32,
    /// Overall average score
    pub overall_score: f32,
    /// Improvement trend
    pub improvement_trend: f32,
}

impl Default for SessionScores {
    fn default() -> Self {
        Self {
            average_quality: 0.0,
            average_pronunciation: 0.0,
            average_fluency: 0.0,
            overall_score: 0.0,
            improvement_trend: 0.0,
        }
    }
}

/// Session summary for recent sessions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionSummary {
    /// Session ID
    pub session_id: String,
    /// Session timestamp
    pub timestamp: DateTime<Utc>,
    /// Session duration
    pub duration: Duration,
    /// Overall score achieved
    pub score: f32,
    /// Exercises completed
    pub exercises_completed: usize,
}

/// Exercise session for compatibility
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExerciseSession {
    /// Exercise ID
    pub exercise_id: String,
    /// Exercise name
    pub exercise_name: String,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// Current attempt
    pub current_attempt: usize,
    /// Progress percentage
    pub progress: f32,
}

/// Session statistics for compatibility
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionStatistics {
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: Option<DateTime<Utc>>,
    /// Total duration
    pub duration: Duration,
    /// Audio generated count
    pub audio_generated_count: usize,
    /// Average quality score
    pub average_quality_score: f32,
    /// Average pronunciation score
    pub average_pronunciation_score: f32,
    /// Exercises attempted
    pub exercises_attempted: usize,
    /// Exercises completed
    pub exercises_completed: usize,
}

impl Default for SessionStatistics {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            start_time: now,
            end_time: None,
            duration: Duration::from_secs(0),
            audio_generated_count: 0,
            average_quality_score: 0.0,
            average_pronunciation_score: 0.0,
            exercises_attempted: 0,
            exercises_completed: 0,
        }
    }
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Feedback system configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackConfig {
    /// Enable real-time feedback
    pub enable_realtime: bool,
    /// Enable adaptive feedback
    pub enable_adaptive: bool,
    /// Response timeout in milliseconds
    pub response_timeout_ms: u64,
    /// Feedback detail level [0.0, 1.0]
    pub feedback_detail_level: f32,
    /// Maximum concurrent feedback requests
    pub max_concurrent_requests: usize,
    /// Cache feedback results
    pub enable_caching: bool,
    /// Supported languages
    pub supported_languages: Vec<LanguageCode>,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            enable_realtime: true,
            enable_adaptive: true,
            response_timeout_ms: 500,
            feedback_detail_level: 0.7,
            max_concurrent_requests: 10,
            enable_caching: true,
            supported_languages: vec![
                LanguageCode::EnUs,
                LanguageCode::EnGb,
                LanguageCode::DeDe,
                LanguageCode::FrFr,
                LanguageCode::EsEs,
                LanguageCode::JaJp,
            ],
        }
    }
}

/// Adaptive learning configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Enable learning from user interactions
    pub enable_learning: bool,
    /// Learning rate [0.0, 1.0]
    pub learning_rate: f32,
    /// Adaptation threshold [0.0, 1.0]
    pub adaptation_threshold: f32,
    /// Memory decay factor [0.0, 1.0]
    pub memory_decay: f32,
    /// Minimum data points for adaptation
    pub min_data_points: usize,
    /// Maximum model complexity
    pub max_model_complexity: usize,
    /// Enable personalization
    pub enable_personalization: bool,
    /// Maximum history size for models
    pub max_history_size: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enable_learning: true,
            learning_rate: 0.01,
            adaptation_threshold: 0.1,
            memory_decay: 0.95,
            min_data_points: 10,
            max_model_complexity: 1000,
            enable_personalization: true,
            max_history_size: 1000,
        }
    }
}

/// Progress tracking configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgressConfig {
    /// Track user improvements
    pub track_improvements: bool,
    /// Snapshot frequency
    pub snapshot_frequency: Duration,
    /// History retention period
    pub history_retention: Duration,
    /// Enable achievement system
    pub enable_achievements: bool,
    /// Enable goal setting
    pub enable_goals: bool,
    /// Analytics granularity
    pub analytics_granularity: AnalyticsGranularity,
}

impl Default for ProgressConfig {
    fn default() -> Self {
        Self {
            track_improvements: true,
            snapshot_frequency: Duration::from_secs(300), // 5 minutes
            history_retention: Duration::from_secs(365 * 24 * 3600), // 1 year
            enable_achievements: true,
            enable_goals: true,
            analytics_granularity: AnalyticsGranularity::Detailed,
        }
    }
}

/// Analytics granularity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnalyticsGranularity {
    /// Basic metrics only
    Basic,
    /// Standard metrics
    Standard,
    /// Detailed metrics
    Detailed,
    /// Complete analytics
    Complete,
}

// ============================================================================
// Core Traits
// ============================================================================

/// Trait for providing feedback on speech synthesis
#[async_trait]
pub trait FeedbackProvider: Send + Sync {
    /// Generate feedback for synthesized audio
    async fn generate_feedback(
        &self,
        audio: &AudioBuffer,
        text: &str,
        context: &FeedbackContext,
    ) -> FeedbackResult<FeedbackResponse>;

    /// Generate feedback with evaluation scores
    async fn generate_feedback_with_scores(
        &self,
        audio: &AudioBuffer,
        quality_score: &QualityScore,
        pronunciation_score: &PronunciationScore,
        context: &FeedbackContext,
    ) -> FeedbackResult<FeedbackResponse>;

    /// Get supported feedback types
    fn supported_feedback_types(&self) -> Vec<FeedbackType>;

    /// Get provider metadata
    fn metadata(&self) -> FeedbackProviderMetadata;
}

/// Feedback context information
#[derive(Debug, Clone, PartialEq)]
pub struct FeedbackContext {
    /// User session state
    pub session: SessionState,
    /// Current exercise (if any)
    pub exercise: Option<TrainingExercise>,
    /// Historical performance
    pub history: Vec<ProgressSnapshot>,
    /// User preferences
    pub preferences: UserPreferences,
}

/// Types of feedback that can be generated
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeedbackType {
    /// Quality-focused feedback
    Quality,
    /// Pronunciation feedback
    Pronunciation,
    /// Naturalness feedback
    Naturalness,
    /// Technical feedback
    Technical,
    /// Motivational feedback
    Motivational,
    /// Comparative feedback
    Comparative,
    /// Success feedback
    Success,
    /// Error feedback
    Error,
    /// Warning feedback
    Warning,
    /// Informational feedback
    Info,
    /// Adaptive feedback
    Adaptive,
}

/// Feedback provider metadata
#[derive(Debug, Clone, PartialEq)]
pub struct FeedbackProviderMetadata {
    /// Provider name
    pub name: String,
    /// Version
    pub version: String,
    /// Description
    pub description: String,
    /// Supported feedback types
    pub supported_types: Vec<FeedbackType>,
    /// Response time estimate
    pub response_time_ms: u64,
}

/// Trait for managing feedback sessions
#[async_trait]
pub trait FeedbackSession: Send + Sync {
    /// Process synthesis and provide feedback
    async fn process_synthesis(
        &mut self,
        audio: &AudioBuffer,
        text: &str,
    ) -> FeedbackResult<FeedbackResponse>;

    /// Start a training exercise
    async fn start_exercise(&mut self, exercise: &TrainingExercise) -> FeedbackResult<()>;

    /// Complete current exercise
    async fn complete_exercise(&mut self) -> FeedbackResult<TrainingResult>;

    /// Update user preferences
    async fn update_preferences(&mut self, preferences: UserPreferences) -> FeedbackResult<()>;

    /// Get current session state
    fn get_state(&self) -> &SessionState;

    /// Save session progress
    async fn save_progress(&self) -> FeedbackResult<()>;
}

/// Trait for adaptive learning capabilities
#[async_trait]
pub trait AdaptiveLearner: Send + Sync {
    /// Learn from user interaction
    async fn learn_from_interaction(&mut self, interaction: &UserInteraction)
        -> FeedbackResult<()>;

    /// Adapt feedback based on user progress
    async fn adapt_feedback(
        &self,
        base_feedback: &FeedbackResponse,
        user_state: &AdaptiveState,
    ) -> FeedbackResult<FeedbackResponse>;

    /// Update user model
    async fn update_user_model(
        &mut self,
        user_id: &str,
        performance_data: &PerformanceData,
    ) -> FeedbackResult<()>;

    /// Get user skill estimate
    async fn get_skill_estimate(&self, user_id: &str) -> FeedbackResult<SkillEstimate>;

    /// Get learning recommendations
    async fn get_learning_recommendations(
        &self,
        user_id: &str,
    ) -> FeedbackResult<Vec<LearningRecommendation>>;
}

/// User interaction data for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInteraction {
    /// User ID
    pub user_id: String,
    /// Interaction timestamp
    pub timestamp: DateTime<Utc>,
    /// Type of interaction
    pub interaction_type: InteractionType,
    /// Audio produced
    pub audio: AudioBuffer,
    /// Target text
    pub text: String,
    /// Feedback provided
    pub feedback: FeedbackResponse,
    /// User response to feedback
    pub user_response: Option<UserResponse>,
}

/// Types of user interactions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InteractionType {
    /// Free practice
    Practice,
    /// Exercise completion
    Exercise,
    /// Challenge attempt
    Challenge,
    /// Feedback request
    FeedbackRequest,
}

/// User response to feedback
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UserResponse {
    /// User found feedback helpful
    Helpful,
    /// User found feedback unhelpful
    Unhelpful,
    /// User found feedback confusing
    Confusing,
    /// User implemented suggestion
    Implemented,
    /// User ignored suggestion
    Ignored,
}

/// Performance data for model updates
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceData {
    /// Recent quality scores
    pub quality_scores: Vec<f32>,
    /// Recent pronunciation scores
    pub pronunciation_scores: Vec<f32>,
    /// Improvement trends
    pub improvement_trends: HashMap<FocusArea, f32>,
    /// Learning velocity
    pub learning_velocity: f32,
    /// Consistency metrics
    pub consistency: f32,
}

/// Skill estimate for a user
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillEstimate {
    /// Overall skill level [0.0, 1.0]
    pub overall_skill: f32,
    /// Skill breakdown by area
    pub area_skills: HashMap<FocusArea, f32>,
    /// Confidence in estimate [0.0, 1.0]
    pub confidence: f32,
    /// Prediction accuracy
    pub prediction_accuracy: f32,
}

/// Learning recommendation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearningRecommendation {
    /// Focus area to work on
    pub focus_area: FocusArea,
    /// Priority level [0.0, 1.0]
    pub priority: f32,
    /// Recommended exercises
    pub exercises: Vec<String>,
    /// Expected improvement
    pub expected_improvement: f32,
    /// Time investment required
    pub time_investment: Duration,
}

/// Trait for progress tracking
#[async_trait]
pub trait ProgressTracker: Send + Sync {
    /// Record user progress
    async fn record_progress(
        &mut self,
        user_id: &str,
        session: &SessionState,
        scores: &TrainingScores,
    ) -> FeedbackResult<()>;

    /// Get user progress data
    async fn get_user_progress(&self, user_id: &str) -> FeedbackResult<UserProgress>;

    /// Generate progress report
    async fn generate_progress_report(
        &self,
        user_id: &str,
        time_range: Option<TimeRange>,
    ) -> FeedbackResult<ProgressReport>;

    /// Check for achievements
    async fn check_achievements(&self, user_id: &str) -> FeedbackResult<Vec<Achievement>>;

    /// Set user goals
    async fn set_goals(&mut self, user_id: &str, goals: Vec<Goal>) -> FeedbackResult<()>;
}

/// Time range for progress analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time
    pub start: DateTime<Utc>,
    /// End time
    pub end: DateTime<Utc>,
}

/// Progress report
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgressReport {
    /// User ID
    pub user_id: String,
    /// Report period
    pub period: TimeRange,
    /// Overall improvement
    pub overall_improvement: f32,
    /// Area-specific improvements
    pub area_improvements: HashMap<FocusArea, f32>,
    /// Key achievements
    pub achievements: Vec<Achievement>,
    /// Goal progress
    pub goal_progress: Vec<GoalProgress>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Statistics
    pub statistics: ReportStatistics,
}

/// Goal progress information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoalProgress {
    /// Goal being tracked
    pub goal: Goal,
    /// Progress made in this period
    pub period_progress: f32,
    /// On track to meet deadline
    pub on_track: bool,
    /// Projected completion date
    pub projected_completion: Option<DateTime<Utc>>,
}

/// Report statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportStatistics {
    /// Sessions in period
    pub sessions_count: usize,
    /// Total practice time
    pub total_practice_time: Duration,
    /// Average session length
    pub average_session_length: Duration,
    /// Exercises completed
    pub exercises_completed: usize,
    /// Success rate
    pub success_rate: f32,
}

/// Trait for training providers
#[async_trait]
pub trait TrainingProvider: Send + Sync {
    /// Get available exercises for user
    async fn get_exercises(
        &self,
        user_id: &str,
        skill_level: f32,
    ) -> FeedbackResult<Vec<TrainingExercise>>;

    /// Get recommended exercises
    async fn get_recommended_exercises(
        &self,
        user_id: &str,
    ) -> FeedbackResult<Vec<TrainingExercise>>;

    /// Create custom exercise
    async fn create_custom_exercise(
        &self,
        specification: &ExerciseSpecification,
    ) -> FeedbackResult<TrainingExercise>;

    /// Evaluate exercise completion
    async fn evaluate_exercise(
        &self,
        exercise: &TrainingExercise,
        result: &AudioBuffer,
    ) -> FeedbackResult<TrainingResult>;

    /// Get exercise categories
    fn get_categories(&self) -> Vec<ExerciseCategory>;
}

/// Exercise specification for custom creation
#[derive(Debug, Clone, PartialEq)]
pub struct ExerciseSpecification {
    /// Target focus areas
    pub focus_areas: Vec<FocusArea>,
    /// Difficulty level [0.0, 1.0]
    pub difficulty: f32,
    /// Exercise type
    pub exercise_type: ExerciseType,
    /// Custom text (if any)
    pub custom_text: Option<String>,
    /// Duration constraint
    pub duration_constraint: Option<Duration>,
}

/// Exercise categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExerciseCategory {
    /// Category name
    pub name: String,
    /// Category description
    pub description: String,
    /// Focus areas covered
    pub focus_areas: Vec<FocusArea>,
    /// Difficulty range
    pub difficulty_range: (f32, f32),
    /// Exercise count in category
    pub exercise_count: usize,
}

/// Real-time feedback for compatibility
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealtimeFeedback {
    /// Feedback message
    pub message: String,
    /// Confidence level
    pub confidence: f32,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Feedback type
    pub feedback_type: FeedbackType,
    /// Current score for compatibility
    pub current_score: f32,
    /// Visual indicators for compatibility
    pub visual_indicators: Vec<VisualIndicator>,
}

/// Visual indicator for real-time feedback
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisualIndicator {
    /// Indicator type
    pub indicator_type: String,
    /// Value or message
    pub value: String,
    /// Color hint
    pub color: String,
}

/// Exercise for compatibility
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Exercise {
    /// Exercise ID
    pub id: String,
    /// Exercise name
    pub name: String,
    /// Description
    pub description: String,
    /// Difficulty level
    pub difficulty: f32,
    /// Target text
    pub target_text: String,
}

/// Exercise result for compatibility
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExerciseResult {
    /// Exercise ID
    pub exercise_id: String,
    /// Success flag
    pub success: bool,
    /// Score achieved
    pub score: f32,
    /// Time taken
    pub time_taken: Duration,
    /// Attempts made
    pub attempts: usize,
}

/// Training session config for compatibility
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingSessionConfig {
    /// Session type
    pub session_type: String,
    /// Difficulty level
    pub difficulty: f32,
    /// Max duration
    pub max_duration: Duration,
    /// Target exercises
    pub target_exercises: usize,
}

impl Default for TrainingSessionConfig {
    fn default() -> Self {
        Self {
            session_type: "standard".to_string(),
            difficulty: 0.5,
            max_duration: Duration::from_secs(1800), // 30 minutes
            target_exercises: 5,
        }
    }
}

/// Training session status for compatibility
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrainingSessionStatus {
    /// Not started
    NotStarted,
    /// In progress
    InProgress,
    /// Completed
    Completed,
    /// Paused
    Paused,
    /// Cancelled
    Cancelled,
}

/// Streak milestone types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StreakMilestone {
    /// Week milestone
    Week,
    /// Month milestone
    Month,
    /// Hundred-day milestone
    Hundred,
    /// Year milestone
    Year,
}

/// Time of day preferences for user behavior analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimeOfDay {
    /// Morning practice (before 12 PM)
    Morning,
    /// Afternoon practice (12 PM - 6 PM)
    Afternoon,
    /// Evening practice (after 6 PM)
    Evening,
}

/// User behavior patterns for dynamic achievement generation
#[derive(Debug, Clone, Default)]
pub struct UserBehaviorPatterns {
    /// Average session duration
    pub average_session_duration: Duration,
    /// Preferred time of day for practice
    pub preferred_time_of_day: Option<TimeOfDay>,
    /// Consistency score (0.0 to 1.0)
    pub consistency_score: f32,
    /// Frequency of practice sessions per week
    pub weekly_frequency: f32,
    /// Preferred exercise types
    pub preferred_exercise_types: Vec<ExerciseType>,
    /// Areas of consistent improvement
    pub improvement_areas: Vec<FocusArea>,
}

/// Exercise history for spaced repetition algorithm
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExerciseHistory {
    /// Exercise ID
    pub exercise_id: String,
    /// Number of times this exercise has been repeated
    pub repetition_number: u32,
    /// Ease factor for spaced repetition (typically 2.5 initially)
    pub ease_factor: f32,
    /// Last performance score [0.0, 1.0]
    pub last_performance: f32,
    /// When exercise was last attempted
    pub last_attempted: DateTime<Utc>,
    /// All performance scores for this exercise
    pub performance_history: Vec<f32>,
    /// Average performance over all attempts
    pub average_performance: f32,
    /// Whether exercise was mastered (consistently high performance)
    pub mastered: bool,
    /// Number of times exercise was failed
    pub failure_count: usize,
    /// Total time spent on this exercise
    pub total_time_spent: Duration,
}

/// Feedback severity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeedbackSeverity {
    /// Informational feedback
    Info,
    /// Warning feedback
    Warning,
    /// Error feedback
    Error,
    /// Critical feedback
    Critical,
}

/// Feedback categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeedbackCategory {
    /// Pronunciation feedback
    Pronunciation,
    /// Fluency feedback
    Fluency,
    /// Quality feedback
    Quality,
    /// Technical feedback
    Technical,
    /// Motivational feedback
    Motivational,
}

/// Feedback priority levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeedbackPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Feedback source types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeedbackSource {
    /// Adaptive feedback system
    Adaptive,
    /// Real-time feedback system
    Realtime,
    /// Training system
    Training,
    /// Manual feedback
    Manual,
}

/// Feedback item structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackItem {
    /// Unique identifier
    pub id: String,
    /// Feedback content
    pub content: String,
    /// Severity level
    pub severity: FeedbackSeverity,
    /// Category
    pub category: FeedbackCategory,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Priority
    pub priority: FeedbackPriority,
    /// Source
    pub source: FeedbackSource,
}

/// Immediate action recommendation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImmediateAction {
    /// Action description
    pub action: String,
    /// Priority level
    pub priority: FeedbackPriority,
    /// Estimated time to complete
    pub estimated_time: Duration,
}

/// Long-term goal structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LongTermGoal {
    /// Goal description
    pub goal: String,
    /// Target completion date
    pub target_date: DateTime<Utc>,
    /// Current progress (0.0 to 1.0)
    pub progress: f32,
    /// Milestones
    pub milestones: Vec<String>,
}
