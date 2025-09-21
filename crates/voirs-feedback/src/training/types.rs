//! Core data structures and types for the training system
//!
//! This module contains all the fundamental types, enums, and data structures
//! used throughout the training system.

use crate::traits::{
    ExerciseCategory, ExerciseType, FocusArea, TrainingExercise, TrainingScores,
    UserBehaviorPatterns,
};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::time::Duration;
use voirs_sdk::AudioBuffer;

/// Exercise library containing all available training exercises
#[derive(Debug, Clone)]
pub struct ExerciseLibrary {
    /// Available exercises
    pub exercises: Vec<TrainingExercise>,
    /// Exercise categories
    pub categories: Vec<ExerciseCategory>,
}

/// Training session
#[derive(Debug, Clone)]
pub struct TrainingSession {
    /// Session ID
    pub session_id: String,
    /// User ID for this session
    pub user_id: String,
    /// Session status
    pub status: TrainingSessionStatus,
    /// Current exercise being performed
    pub current_exercise: Option<ExerciseSession>,
    /// Completed exercises in this session
    pub completed_exercises: Vec<ExerciseResult>,
    /// Session configuration
    pub config: TrainingSessionConfig,
    /// When session started
    pub start_time: DateTime<Utc>,
    /// Session statistics
    pub statistics: SessionStatistics,
}

/// Training session configuration
#[derive(Debug, Clone)]
pub struct TrainingSessionConfig {
    /// Maximum session duration
    pub max_duration: Option<Duration>,
    /// Preferred exercise types
    pub preferred_types: Vec<ExerciseType>,
    /// Target difficulty level
    pub target_difficulty: f32,
    /// Focus areas for this session
    pub focus_areas: Vec<FocusArea>,
    /// Enable adaptive difficulty
    pub adaptive_difficulty: bool,
}

impl Default for TrainingSessionConfig {
    fn default() -> Self {
        Self {
            max_duration: Some(Duration::from_secs(3600)), // 1 hour
            preferred_types: vec![ExerciseType::FreeForm, ExerciseType::Pronunciation],
            target_difficulty: 0.5,
            focus_areas: vec![FocusArea::Pronunciation, FocusArea::Quality],
            adaptive_difficulty: true,
        }
    }
}

/// Training session status
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingSessionStatus {
    /// Session is active
    Active,
    /// Session paused
    Paused,
    /// Session completed
    Completed,
    /// Session cancelled
    Cancelled,
}

/// Exercise session within a training session
#[derive(Debug, Clone)]
pub struct ExerciseSession {
    /// The exercise being performed
    pub exercise: TrainingExercise,
    /// All attempts made
    pub attempts: Vec<ExerciseAttempt>,
    /// When exercise started
    pub start_time: DateTime<Utc>,
    /// Current status
    pub status: ExerciseSessionStatus,
    /// Feedback history
    pub feedback_history: Vec<AttemptFeedback>,
    /// Current attempt number
    pub current_attempt: usize,
}

/// Exercise session status
#[derive(Debug, Clone, PartialEq)]
pub enum ExerciseSessionStatus {
    /// Exercise in progress
    InProgress,
    /// Exercise completed successfully
    Completed,
    /// Exercise failed (max attempts reached)
    Failed,
    /// Exercise paused
    Paused,
}

/// Session flow optimizer for intelligent session management
#[derive(Clone)]
pub struct SessionFlowOptimizer {
    /// Attention span modeling configuration
    pub attention_span_config: AttentionSpanConfig,
    /// Break timing algorithm
    pub break_timer: BreakTimingAlgorithm,
    /// Fatigue detection threshold
    pub fatigue_threshold: f32,
    /// Performance decline threshold for break suggestion
    pub performance_decline_threshold: f32,
}

impl Default for SessionFlowOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Attention span configuration for modeling user focus
#[derive(Debug, Clone)]
pub struct AttentionSpanConfig {
    /// Base attention span in minutes
    pub base_attention_span: u32,
    /// Decline rate after peak attention
    pub decline_rate: f32,
    /// Break effectiveness for attention restoration
    pub break_effectiveness: f32,
}

impl Default for AttentionSpanConfig {
    fn default() -> Self {
        Self {
            base_attention_span: 45,  // 45 minutes base attention span
            decline_rate: 0.1,        // 10% decline per 15 minutes after peak
            break_effectiveness: 0.8, // 80% attention restoration after break
        }
    }
}

/// Break timing algorithm for optimal break suggestions
#[derive(Debug, Clone)]
pub struct BreakTimingAlgorithm {
    /// Minimum time between break suggestions
    pub min_break_interval: Duration,
    /// Performance threshold for break suggestion
    pub performance_threshold: f32,
    /// Time-based break interval
    pub time_based_interval: Duration,
}

impl Default for BreakTimingAlgorithm {
    fn default() -> Self {
        Self {
            min_break_interval: Duration::from_secs(900), // 15 minutes
            performance_threshold: 0.75, // Suggest break if performance drops below 75%
            time_based_interval: Duration::from_secs(2700), // 45 minutes
        }
    }
}

/// Session flow recommendation
#[derive(Debug, Clone)]
pub enum SessionFlowRecommendation {
    /// Continue with current session
    Continue {
        /// Estimated remaining effective time
        estimated_remaining_time: Duration,
        /// Motivational message
        motivation_message: String,
    },
    /// Take a break
    TakeBreak {
        /// Reason for break recommendation
        reason: String,
        /// Suggested break duration
        suggested_break_duration: Duration,
        /// Suggestion for break activity
        resume_suggestion: String,
    },
    /// Adjust exercise difficulty
    AdjustDifficulty {
        /// Reason for adjustment
        reason: String,
        /// Suggested difficulty change (-1.0 to 1.0)
        suggested_difficulty_change: f32,
        /// Alternative exercise types to try
        alternative_exercise_types: Vec<ExerciseType>,
    },
    /// Increase challenge level
    IncreaseChallenge {
        /// Reason for increase
        reason: String,
        /// Suggested difficulty increase
        suggested_difficulty_change: f32,
        /// New exercise types to introduce
        new_exercise_types: Vec<ExerciseType>,
    },
    /// End session
    EndSession {
        /// Reason for ending
        reason: String,
        /// Session summary
        session_summary: String,
    },
}

/// Individual exercise attempt
#[derive(Debug, Clone)]
pub struct ExerciseAttempt {
    /// Attempt number
    pub attempt_number: usize,
    /// Audio produced
    pub audio: AudioBuffer,
    /// When attempt was made
    pub timestamp: DateTime<Utc>,
    /// Quality score achieved
    pub quality_score: f32,
    /// Pronunciation score achieved
    pub pronunciation_score: f32,
    /// Time taken for evaluation
    pub evaluation_time: Duration,
    /// Feedback for this attempt
    pub feedback: AttemptFeedback,
}

/// Feedback for an individual attempt
#[derive(Debug, Clone)]
pub struct AttemptFeedback {
    /// Overall score for the attempt
    pub overall_score: f32,
    /// Quality component score
    pub quality_score: f32,
    /// Pronunciation component score
    pub pronunciation_score: f32,
    /// Things done well
    pub strengths: Vec<String>,
    /// Areas needing improvement
    pub weaknesses: Vec<String>,
    /// Specific suggestions
    pub suggestions: Vec<String>,
    /// Encouraging message
    pub encouragement: String,
}

/// Result of an exercise attempt
#[derive(Debug, Clone)]
pub struct AttemptResult {
    /// The attempt that was evaluated
    pub attempt: ExerciseAttempt,
    /// Whether this attempt met success criteria
    pub success: bool,
    /// Detailed criteria compliance
    pub meets_criteria: CriteriaCompliance,
    /// Recommended next steps
    pub next_steps: Vec<String>,
    /// Specific improvement suggestions
    pub improvement_suggestions: Vec<ImprovementSuggestion>,
}

/// Criteria compliance analysis
#[derive(Debug, Clone)]
pub struct CriteriaCompliance {
    /// Whether quality criteria was met
    pub quality_met: bool,
    /// Whether pronunciation criteria was met
    pub pronunciation_met: bool,
    /// Gap in quality score (if not met)
    pub quality_gap: f32,
    /// Gap in pronunciation score (if not met)
    pub pronunciation_gap: f32,
}

/// Specific improvement suggestion
#[derive(Debug, Clone)]
pub struct ImprovementSuggestion {
    /// Area needing improvement
    pub area: String,
    /// Current score in this area
    pub current_score: f32,
    /// Target score to achieve
    pub target_score: f32,
    /// Specific actions to take
    pub specific_actions: Vec<String>,
    /// Estimated practice time needed
    pub estimated_practice_time: Duration,
}

/// Complete exercise result
#[derive(Debug, Clone)]
pub struct ExerciseResult {
    /// Exercise that was completed
    pub exercise: TrainingExercise,
    /// All attempts made
    pub attempts: Vec<ExerciseAttempt>,
    /// Whether exercise was completed successfully
    pub success: bool,
    /// Time taken to complete
    pub completion_time: Duration,
    /// Final scores achieved
    pub final_scores: ExerciseScores,
    /// Feedback for the overall exercise
    pub feedback: ExerciseFeedback,
}

/// Scores for an exercise
#[derive(Debug, Clone)]
pub struct ExerciseScores {
    /// Final quality score
    pub quality: f32,
    /// Final pronunciation score
    pub pronunciation: f32,
    /// Consistency across attempts
    pub consistency: f32,
    /// Improvement shown during exercise
    pub improvement: f32,
}

/// Comprehensive feedback for an exercise
#[derive(Debug, Clone)]
pub struct ExerciseFeedback {
    /// Feedback items from evaluation
    pub feedback_items: Vec<String>,
    /// Overall assessment message
    pub overall_assessment: String,
    /// Specific recommendations for improvement
    pub improvement_recommendations: Vec<String>,
    /// Encouraging remarks
    pub encouragement: String,
}

/// Session statistics
#[derive(Debug, Clone, Default)]
pub struct SessionStatistics {
    /// Total attempts across all exercises
    pub total_attempts: usize,
    /// Successful attempts
    pub successful_attempts: usize,
    /// Total time spent on evaluation
    pub total_evaluation_time: Duration,
    /// Number of exercises started
    pub exercises_started: usize,
    /// Number of exercises completed
    pub exercises_completed: usize,
}

/// Training session result
#[derive(Debug, Clone)]
pub struct TrainingSessionResult {
    /// The completed session
    pub session: TrainingSession,
    /// When session was completed
    pub completion_time: DateTime<Utc>,
    /// Total session duration
    pub session_duration: Duration,
    /// Number of exercises attempted
    pub total_exercises: usize,
    /// Number of exercises completed successfully
    pub successful_exercises: usize,
    /// Success rate
    pub success_rate: f32,
    /// Average scores across all exercises
    pub average_scores: SessionScores,
    /// Achievements earned in this session
    pub achievements: Vec<String>,
    /// Recommendations for future sessions
    pub recommendations: Vec<String>,
    /// Suggested next learning path
    pub next_learning_path: LearningPath,
}

/// Session-level scores
#[derive(Debug, Clone)]
pub struct SessionScores {
    /// Average quality score
    pub average_quality: f32,
    /// Average pronunciation score
    pub average_pronunciation: f32,
    /// Average fluency score
    pub average_fluency: f32,
    /// Overall session score
    pub overall_score: f32,
    /// Improvement trend throughout session
    pub improvement_trend: f32,
}

/// Suggested learning path
#[derive(Debug, Clone)]
pub struct LearningPath {
    /// Recommended focus areas
    pub suggested_focus_areas: Vec<FocusArea>,
    /// Recommended difficulty level
    pub difficulty_level: f32,
    /// Estimated time for next session
    pub estimated_duration: Duration,
    /// Recommended exercise types
    pub exercise_types: Vec<ExerciseType>,
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Maximum recommended exercises to return
    pub max_recommended_exercises: usize,
    /// Enable adaptive difficulty progression
    pub adaptive_difficulty: bool,
    /// Default exercise timeout
    pub default_timeout: Duration,
    /// Enable detailed feedback
    pub detailed_feedback: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_recommended_exercises: 10,
            adaptive_difficulty: true,
            default_timeout: Duration::from_secs(300),
            detailed_feedback: true,
        }
    }
}

/// Training system metrics
#[derive(Debug, Default)]
pub struct TrainingMetrics {
    /// Total sessions created
    pub total_sessions: usize,
    /// Currently active sessions
    pub active_sessions: usize,
    /// Completed sessions
    pub completed_sessions: usize,
    /// Total attempts across all sessions
    pub total_attempts: usize,
    /// Successful attempts
    pub successful_attempts: usize,
    /// Total time spent in sessions
    pub total_session_time: Duration,
    /// Total evaluation time
    pub total_evaluation_time: Duration,
}

/// Training system statistics
#[derive(Debug, Clone)]
pub struct TrainingSystemStats {
    /// Total sessions created
    pub total_sessions: usize,
    /// Currently active sessions
    pub active_sessions: usize,
    /// Successfully completed sessions
    pub completed_sessions: usize,
    /// Total exercises available
    pub total_exercises: usize,
    /// Total attempts made
    pub total_attempts: usize,
    /// Successful attempts
    pub successful_attempts: usize,
    /// Overall success rate
    pub success_rate: f32,
    /// Average session duration
    pub average_session_duration: Duration,
    /// Average evaluation time per attempt
    pub average_evaluation_time: Duration,
}

/// Collaborative learning features for peer interaction and group exercises
#[derive(Clone)]
pub struct CollaborativeLearningSystem {
    /// Peer practice sessions
    pub peer_sessions: std::sync::Arc<std::sync::RwLock<HashMap<String, PeerPracticeSession>>>,
    /// Group exercise completion tracking
    pub group_exercises: std::sync::Arc<std::sync::RwLock<HashMap<String, GroupExerciseSession>>>,
    /// Cooperative challenges
    pub cooperative_challenges:
        std::sync::Arc<std::sync::RwLock<HashMap<String, CooperativeChallenge>>>,
    /// Peer feedback system
    pub peer_feedback: std::sync::Arc<std::sync::RwLock<PeerFeedbackSystem>>,
    /// Virtual study groups
    pub virtual_study_groups: std::sync::Arc<std::sync::RwLock<HashMap<String, VirtualStudyGroup>>>,
    /// Configuration
    pub config: CollaborativeLearningConfig,
}

impl CollaborativeLearningSystem {
    /// Create new collaborative learning system
    pub async fn new() -> Result<Self, crate::FeedbackError> {
        Ok(Self {
            peer_sessions: std::sync::Arc::new(std::sync::RwLock::new(HashMap::new())),
            group_exercises: std::sync::Arc::new(std::sync::RwLock::new(HashMap::new())),
            cooperative_challenges: std::sync::Arc::new(std::sync::RwLock::new(HashMap::new())),
            peer_feedback: std::sync::Arc::new(std::sync::RwLock::new(PeerFeedbackSystem::new())),
            virtual_study_groups: std::sync::Arc::new(std::sync::RwLock::new(HashMap::new())),
            config: CollaborativeLearningConfig::default(),
        })
    }
}

/// Peer practice session
#[derive(Debug, Clone)]
pub struct PeerPracticeSession {
    /// Session identifier
    pub session_id: String,
    /// ID of the user who initiated the session
    pub initiator_id: String,
    /// ID of the peer participant
    pub peer_id: String,
    /// Exercise being practiced
    pub exercise_id: String,
    /// Current status of the session
    pub status: PeerSessionStatus,
    /// When the session was created
    pub created_at: DateTime<Utc>,
    /// When the session was started
    pub started_at: Option<DateTime<Utc>>,
    /// When the session was completed
    pub completed_at: Option<DateTime<Utc>>,
    /// Results from the initiator
    pub initiator_results: Option<TrainingScores>,
    /// Results from the peer
    pub peer_results: Option<TrainingScores>,
    /// Feedback exchanged between participants
    pub mutual_feedback: Vec<PeerFeedback>,
}

/// Group exercise session
#[derive(Debug, Clone)]
pub struct GroupExerciseSession {
    /// Session identifier
    pub session_id: String,
    /// Group identifier
    pub group_id: String,
    /// Exercise being practiced
    pub exercise_id: String,
    /// List of participant user IDs
    pub participants: Vec<String>,
    /// Current status of the session
    pub status: GroupExerciseStatus,
    /// When the session was created
    pub created_at: DateTime<Utc>,
    /// When the session was completed
    pub completed_at: Option<DateTime<Utc>>,
    /// Results from each participant
    pub participant_results: HashMap<String, TrainingScores>,
    /// Overall group score
    pub group_score: f32,
    /// Percentage of participants who completed
    pub completion_rate: f32,
}

/// Cooperative challenge
#[derive(Debug, Clone)]
pub struct CooperativeChallenge {
    /// Challenge identifier
    pub challenge_id: String,
    /// Challenge name
    pub name: String,
    /// List of participant user IDs
    pub participants: Vec<String>,
    /// Target score to achieve
    pub target_score: f32,
    /// Current cumulative score
    pub current_score: f32,
    /// Current status of the challenge
    pub status: CooperativeChallengeStatus,
    /// When the challenge was created
    pub created_at: DateTime<Utc>,
    /// Challenge deadline
    pub deadline: DateTime<Utc>,
    /// Score contributions from each participant
    pub participant_contributions: HashMap<String, f32>,
    /// Challenge milestones
    pub milestones: Vec<ChallengeMilestone>,
}

/// Peer feedback system
#[derive(Debug, Clone)]
pub struct PeerFeedbackSystem {
    /// History of feedback for each session
    pub feedback_history: HashMap<String, Vec<PeerFeedback>>,
}

impl PeerFeedbackSystem {
    /// Create a new peer feedback system
    pub fn new() -> Self {
        Self {
            feedback_history: HashMap::new(),
        }
    }

    /// Add feedback to the system
    pub fn add_feedback(&mut self, session_id: &str, feedback: PeerFeedback) {
        self.feedback_history
            .entry(session_id.to_string())
            .or_insert_with(Vec::new)
            .push(feedback);
    }
}

/// Virtual study group
#[derive(Debug, Clone)]
pub struct VirtualStudyGroup {
    /// Group identifier
    pub group_id: String,
    /// Group name
    pub name: String,
    /// ID of the user who created the group
    pub creator_id: String,
    /// List of member user IDs
    pub members: Vec<String>,
    /// Focus areas for the group
    pub focus_areas: Vec<FocusArea>,
    /// Current status of the group
    pub status: StudyGroupStatus,
    /// When the group was created
    pub created_at: DateTime<Utc>,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
    /// Group progress tracking
    pub group_progress: GroupProgress,
    /// Study sessions for the group
    pub study_sessions: Vec<GroupStudySession>,
}

/// Peer feedback
#[derive(Debug, Clone)]
pub struct PeerFeedback {
    /// Feedback identifier
    pub feedback_id: String,
    /// User ID who provided the feedback
    pub from_user: String,
    /// User ID who received the feedback
    pub to_user: String,
    /// Session identifier
    pub session_id: String,
    /// Rating score (0.0 to 5.0)
    pub rating: f32,
    /// Text comments
    pub comments: String,
    /// Identified strengths
    pub strengths: Vec<String>,
    /// Improvement suggestions
    pub suggestions: Vec<String>,
    /// When the feedback was created
    pub created_at: DateTime<Utc>,
}

/// Challenge milestone
#[derive(Debug, Clone)]
pub struct ChallengeMilestone {
    /// Milestone identifier
    pub milestone_id: String,
    /// Milestone name
    pub name: String,
    /// Target score to achieve this milestone
    pub target_score: f32,
    /// Whether the milestone has been achieved
    pub achieved: bool,
    /// When the milestone was achieved
    pub achieved_at: Option<DateTime<Utc>>,
    /// Reward for achieving the milestone
    pub reward: String,
}

/// Group progress tracking
#[derive(Debug, Clone, Default)]
pub struct GroupProgress {
    /// Total number of sessions
    pub total_sessions: usize,
    /// Number of completed exercises
    pub completed_exercises: usize,
    /// Average score across all group members
    pub average_group_score: f32,
    /// Rate of improvement over time
    pub improvement_rate: f32,
    /// Score for collaboration effectiveness
    pub collaboration_score: f32,
}

/// Group study session
#[derive(Debug, Clone)]
pub struct GroupStudySession {
    /// Session identifier
    pub session_id: String,
    /// Session topic
    pub topic: String,
    /// Scheduled start time
    pub scheduled_time: DateTime<Utc>,
    /// Session duration
    pub duration: Duration,
    /// List of attendee user IDs
    pub attendees: Vec<String>,
    /// Current status of the session
    pub status: StudySessionStatus,
}

/// Peer session status
#[derive(Debug, Clone, PartialEq)]
pub enum PeerSessionStatus {
    /// Session is pending acceptance
    Pending,
    /// Session is active and in progress
    Active,
    /// Session has been completed
    Completed,
    /// Session has been cancelled
    Cancelled,
}

/// Group exercise status
#[derive(Debug, Clone, PartialEq)]
pub enum GroupExerciseStatus {
    /// Exercise is active and in progress
    Active,
    /// Exercise has been completed
    Completed,
    /// Exercise has been cancelled
    Cancelled,
}

/// Cooperative challenge status
#[derive(Debug, Clone, PartialEq)]
pub enum CooperativeChallengeStatus {
    /// Challenge is active and in progress
    Active,
    /// Challenge has been completed successfully
    Completed,
    /// Challenge has failed to meet targets
    Failed,
    /// Challenge has been cancelled
    Cancelled,
}

/// Study group status
#[derive(Debug, Clone, PartialEq)]
pub enum StudyGroupStatus {
    /// Group is active and accepting new members
    Active,
    /// Group is inactive but not archived
    Inactive,
    /// Group has been archived
    Archived,
}

/// Study session status
#[derive(Debug, Clone, PartialEq)]
pub enum StudySessionStatus {
    /// Session is scheduled for the future
    Scheduled,
    /// Session is active and in progress
    Active,
    /// Session has been completed
    Completed,
    /// Session has been cancelled
    Cancelled,
}

/// Collaborative learning configuration
#[derive(Debug, Clone)]
pub struct CollaborativeLearningConfig {
    /// Maximum number of simultaneous peer sessions per user
    pub max_peer_sessions: usize,
    /// Maximum size for group exercises
    pub max_group_size: usize,
    /// Default duration for challenges in days
    pub challenge_duration_days: i64,
    /// Timeout for peer feedback in hours
    pub feedback_timeout_hours: i64,
    /// Maximum number of members in a study group
    pub study_group_max_members: usize,
}

impl Default for CollaborativeLearningConfig {
    fn default() -> Self {
        Self {
            max_peer_sessions: 10,
            max_group_size: 8,
            challenge_duration_days: 14,
            feedback_timeout_hours: 48,
            study_group_max_members: 20,
        }
    }
}
