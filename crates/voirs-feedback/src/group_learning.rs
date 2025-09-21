//! Group Learning Orchestration System
//!
//! This module provides comprehensive group learning capabilities including:
//! - Synchronized group exercises with real-time coordination
//! - Classroom management tools for educators
//! - Group progress tracking and analytics
//! - Collaborative challenges and competitions
//! - Virtual classroom environments

use crate::traits::{
    Achievement, FeedbackResponse, SessionScores, SessionState, TrainingExercise, UserFeedback,
    UserProgress,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

/// Result type for group learning operations
pub type GroupLearningResult<T> = Result<T, GroupLearningError>;

/// Errors that can occur during group learning operations
#[derive(Debug, thiserror::Error)]
pub enum GroupLearningError {
    #[error("Group not found: {group_id}")]
    GroupNotFound { group_id: String },
    #[error("User not found in group: {user_id}")]
    UserNotInGroup { user_id: String },
    #[error("Exercise synchronization failed: {reason}")]
    SynchronizationError { reason: String },
    #[error("Insufficient permissions for operation: {operation}")]
    InsufficientPermissions { operation: String },
    #[error("Group capacity exceeded: current {current}, max {max}")]
    GroupCapacityExceeded { current: usize, max: usize },
    #[error("Session conflict: {details}")]
    SessionConflict { details: String },
}

/// Group learning orchestrator managing all group activities
#[derive(Debug)]
pub struct GroupLearningOrchestrator {
    /// Active groups
    groups: RwLock<HashMap<String, LearningGroup>>,
    /// Active sessions
    sessions: RwLock<HashMap<String, GroupSession>>,
    /// Classroom manager
    classroom_manager: ClassroomManager,
    /// Progress tracker
    progress_tracker: GroupProgressTracker,
    /// Challenge coordinator
    challenge_coordinator: CollaborativeChallengeCoordinator,
    /// Event broadcaster for real-time updates
    event_broadcaster: broadcast::Sender<GroupEvent>,
}

/// Learning group with participants and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningGroup {
    /// Unique group identifier
    pub group_id: String,
    /// Group name
    pub name: String,
    /// Group description
    pub description: String,
    /// Group type
    pub group_type: GroupType,
    /// Participants in the group
    pub participants: HashMap<String, GroupParticipant>,
    /// Group settings
    pub settings: GroupSettings,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Group status
    pub status: GroupStatus,
    /// Current active session (if any)
    pub current_session: Option<String>,
    /// Group statistics
    pub stats: GroupStatistics,
}

/// Types of learning groups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GroupType {
    /// Teacher-led classroom
    Classroom,
    /// Peer study group
    StudyGroup,
    /// Competitive group
    Competition,
    /// Collaborative project
    Collaboration,
    /// Practice partnership
    Partnership,
}

/// Group participant with role and status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupParticipant {
    /// User ID
    pub user_id: String,
    /// Display name
    pub display_name: String,
    /// Role in group
    pub role: ParticipantRole,
    /// Join timestamp
    pub joined_at: DateTime<Utc>,
    /// Participation statistics
    pub stats: ParticipationStats,
    /// Current status
    pub status: ParticipantStatus,
}

/// Roles participants can have in groups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParticipantRole {
    /// Group leader/teacher
    Leader,
    /// Co-facilitator
    Facilitator,
    /// Regular participant
    Participant,
    /// Observer (can view but not participate)
    Observer,
}

/// Current status of a participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParticipantStatus {
    /// Online and active
    Active,
    /// Online but idle
    Idle,
    /// Offline
    Offline,
    /// Temporarily away
    Away,
}

/// Group settings and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupSettings {
    /// Maximum number of participants
    pub max_participants: usize,
    /// Whether group is public or private
    pub is_public: bool,
    /// Whether to allow late joining during sessions
    pub allow_late_join: bool,
    /// Synchronization settings
    pub sync_settings: SynchronizationSettings,
    /// Privacy settings
    pub privacy_settings: PrivacySettings,
}

/// Exercise synchronization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationSettings {
    /// Whether exercises are synchronized
    pub sync_exercises: bool,
    /// Synchronization mode
    pub sync_mode: SyncMode,
    /// Wait timeout for participant synchronization
    pub sync_timeout: Duration,
    /// Whether to allow individual pacing
    pub allow_individual_pacing: bool,
}

/// Synchronization modes for group exercises
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncMode {
    /// All participants start together
    StrictSync,
    /// Participants can start when ready, within timeout
    FlexibleSync,
    /// Leader controls pacing
    LeaderPaced,
    /// Automatic progression based on majority
    MajorityPaced,
}

/// Privacy settings for groups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacySettings {
    /// Whether scores are visible to all participants
    pub show_scores: bool,
    /// Whether to show real-time progress
    pub show_progress: bool,
    /// Whether to allow voice/video sharing
    pub allow_media_sharing: bool,
    /// Data retention settings
    pub data_retention: DataRetentionSettings,
}

/// Data retention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRetentionSettings {
    /// How long to keep session recordings
    pub session_recording_retention: Duration,
    /// How long to keep chat logs
    pub chat_retention: Duration,
    /// How long to keep performance data
    pub performance_data_retention: Duration,
}

/// Group statistics and metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupStatistics {
    /// Total sessions conducted
    pub total_sessions: usize,
    /// Total participant hours
    pub total_participant_hours: Duration,
    /// Average session duration
    pub average_session_duration: Duration,
    /// Average participation rate
    pub average_participation_rate: f32,
    /// Completion rates by exercise type
    pub completion_rates: HashMap<String, f32>,
    /// Group achievement rate
    pub achievement_rate: f32,
}

/// Participation statistics for individuals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipationStats {
    /// Sessions attended
    pub sessions_attended: usize,
    /// Total participation time
    pub total_participation_time: Duration,
    /// Average session score
    pub average_session_score: f32,
    /// Contribution score
    pub contribution_score: f32,
    /// Collaboration rating
    pub collaboration_rating: f32,
}

/// Current status of a group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GroupStatus {
    /// Group is active and accepting participants
    Active,
    /// Group is in session
    InSession,
    /// Group is paused/suspended
    Paused,
    /// Group is archived (read-only)
    Archived,
    /// Group is deleted
    Deleted,
}

/// Group learning session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupSession {
    /// Session identifier
    pub session_id: String,
    /// Associated group
    pub group_id: String,
    /// Session leader
    pub leader_id: String,
    /// Session title
    pub title: String,
    /// Session description
    pub description: Option<String>,
    /// Planned exercises
    pub exercises: Vec<SessionExercise>,
    /// Current exercise index
    pub current_exercise_index: usize,
    /// Session participants
    pub participants: HashMap<String, SessionParticipant>,
    /// Session start time
    pub started_at: DateTime<Utc>,
    /// Estimated duration
    pub estimated_duration: Duration,
    /// Session settings
    pub settings: SessionSettings,
    /// Session status
    pub status: SessionStatus,
    /// Real-time session data
    pub realtime_data: SessionRealtimeData,
}

/// Exercise within a group session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionExercise {
    /// Base exercise
    pub exercise: TrainingExercise,
    /// Exercise-specific settings for this session
    pub session_settings: ExerciseSessionSettings,
    /// Synchronization requirements
    pub sync_requirements: ExerciseSyncRequirements,
    /// Time allocation
    pub time_allocation: Duration,
    /// Exercise status
    pub status: ExerciseStatus,
}

/// Settings for exercises within sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExerciseSessionSettings {
    /// Whether exercise is mandatory
    pub is_mandatory: bool,
    /// Whether to show peer progress
    pub show_peer_progress: bool,
    /// Whether to enable peer feedback
    pub enable_peer_feedback: bool,
    /// Scoring mode
    pub scoring_mode: ScoringMode,
}

/// Scoring modes for group exercises
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoringMode {
    /// Individual scoring only
    Individual,
    /// Team-based scoring
    Team,
    /// Collaborative scoring (group average)
    Collaborative,
    /// Competitive ranking
    Competitive,
}

/// Synchronization requirements for exercises
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExerciseSyncRequirements {
    /// Whether all participants must start together
    pub synchronized_start: bool,
    /// Whether all participants must finish before proceeding
    pub synchronized_finish: bool,
    /// Maximum time to wait for stragglers
    pub max_wait_time: Duration,
    /// Minimum participants required to proceed
    pub min_participants_to_proceed: usize,
}

/// Status of an exercise within a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExerciseStatus {
    /// Exercise is planned but not yet started
    Planned,
    /// Exercise is currently active
    Active,
    /// Exercise is completed
    Completed,
    /// Exercise was skipped
    Skipped,
    /// Exercise was cancelled
    Cancelled,
}

/// Participant data within a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionParticipant {
    /// User ID
    pub user_id: String,
    /// Join time for this session
    pub joined_at: DateTime<Utc>,
    /// Current status
    pub status: SessionParticipantStatus,
    /// Exercise progress
    pub exercise_progress: HashMap<usize, ExerciseProgress>,
    /// Session scores
    pub session_scores: SessionScores,
    /// Real-time data
    pub realtime_status: ParticipantRealtimeStatus,
}

/// Status of session participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionParticipantStatus {
    /// Actively participating
    Active,
    /// Waiting for exercise to start
    Waiting,
    /// Working on current exercise
    Working,
    /// Completed current exercise
    Completed,
    /// Temporarily disconnected
    Disconnected,
    /// Left the session
    Left,
}

/// Progress on a specific exercise
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExerciseProgress {
    /// Exercise index
    pub exercise_index: usize,
    /// Start time
    pub started_at: Option<DateTime<Utc>>,
    /// Completion time
    pub completed_at: Option<DateTime<Utc>>,
    /// Current attempt number
    pub attempt_number: usize,
    /// Current score
    pub current_score: Option<f32>,
    /// Progress percentage
    pub progress_percentage: f32,
    /// Status
    pub status: ExerciseProgressStatus,
}

/// Status of exercise progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExerciseProgressStatus {
    /// Not started
    NotStarted,
    /// In progress
    InProgress,
    /// Completed successfully
    Completed,
    /// Failed (exceeded attempts)
    Failed,
    /// Skipped
    Skipped,
}

/// Real-time status of participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantRealtimeStatus {
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
    /// Current device/platform
    pub device_info: String,
    /// Network quality
    pub network_quality: NetworkQuality,
    /// Audio/video status
    pub media_status: MediaStatus,
}

/// Network quality indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkQuality {
    /// Excellent connection
    Excellent,
    /// Good connection
    Good,
    /// Fair connection
    Fair,
    /// Poor connection
    Poor,
    /// Disconnected
    Disconnected,
}

/// Media sharing status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaStatus {
    /// Audio enabled
    pub audio_enabled: bool,
    /// Video enabled
    pub video_enabled: bool,
    /// Screen sharing active
    pub screen_sharing: bool,
}

/// Session settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSettings {
    /// Whether session is recorded
    pub record_session: bool,
    /// Whether to enable chat
    pub enable_chat: bool,
    /// Whether to enable voice/video
    pub enable_media: bool,
    /// Auto-advance settings
    pub auto_advance: AutoAdvanceSettings,
}

/// Auto-advance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoAdvanceSettings {
    /// Whether to auto-advance exercises
    pub enabled: bool,
    /// Percentage of participants needed to auto-advance
    pub advance_threshold: f32,
    /// Maximum wait time before auto-advance
    pub max_wait_time: Duration,
}

/// Session status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionStatus {
    /// Session is planned but not started
    Planned,
    /// Session is starting (waiting for participants)
    Starting,
    /// Session is active
    Active,
    /// Session is paused
    Paused,
    /// Session is completed
    Completed,
    /// Session was cancelled
    Cancelled,
}

/// Real-time session data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRealtimeData {
    /// Active participant count
    pub active_participants: usize,
    /// Current exercise timing
    pub current_exercise_timing: ExerciseTiming,
    /// Overall session progress
    pub session_progress: f32,
    /// Sync status
    pub sync_status: SynchronizationStatus,
}

/// Timing data for current exercise
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExerciseTiming {
    /// Exercise start time
    pub started_at: Option<DateTime<Utc>>,
    /// Elapsed time
    pub elapsed_time: Duration,
    /// Estimated remaining time
    pub estimated_remaining: Duration,
    /// Time pressure level
    pub time_pressure: TimePressure,
}

/// Time pressure indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimePressure {
    /// Plenty of time remaining
    Low,
    /// Moderate time pressure
    Medium,
    /// High time pressure
    High,
    /// Critical time remaining
    Critical,
}

/// Synchronization status for sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationStatus {
    /// Whether group is synchronized
    pub is_synchronized: bool,
    /// Participants waiting for sync
    pub waiting_participants: Vec<String>,
    /// Participants ready to proceed
    pub ready_participants: Vec<String>,
    /// Sync timeout remaining
    pub timeout_remaining: Option<Duration>,
}

/// Events broadcast during group learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GroupEvent {
    /// Group was created
    GroupCreated { group_id: String },
    /// Participant joined group
    ParticipantJoined { group_id: String, user_id: String },
    /// Participant left group
    ParticipantLeft { group_id: String, user_id: String },
    /// Session started
    SessionStarted {
        group_id: String,
        session_id: String,
    },
    /// Exercise started
    ExerciseStarted {
        session_id: String,
        exercise_index: usize,
    },
    /// Exercise completed
    ExerciseCompleted {
        session_id: String,
        exercise_index: usize,
    },
    /// Session completed
    SessionCompleted { session_id: String },
    /// Sync status changed
    SyncStatusChanged {
        session_id: String,
        status: SynchronizationStatus,
    },
    /// Real-time progress update
    ProgressUpdate {
        session_id: String,
        user_id: String,
        progress: f32,
    },
}

impl GroupLearningOrchestrator {
    /// Create a new group learning orchestrator
    pub fn new() -> Self {
        let (event_broadcaster, _) = broadcast::channel(1000);

        Self {
            groups: RwLock::new(HashMap::new()),
            sessions: RwLock::new(HashMap::new()),
            classroom_manager: ClassroomManager::new(),
            progress_tracker: GroupProgressTracker::new(),
            challenge_coordinator: CollaborativeChallengeCoordinator::new(),
            event_broadcaster,
        }
    }

    /// Create a new learning group
    pub async fn create_group(
        &self,
        name: String,
        description: String,
        group_type: GroupType,
        creator_id: String,
        settings: GroupSettings,
    ) -> GroupLearningResult<String> {
        let group_id = Uuid::new_v4().to_string();

        let mut participants = HashMap::new();
        participants.insert(
            creator_id.clone(),
            GroupParticipant {
                user_id: creator_id,
                display_name: "Creator".to_string(),
                role: ParticipantRole::Leader,
                joined_at: Utc::now(),
                stats: ParticipationStats {
                    sessions_attended: 0,
                    total_participation_time: Duration::from_secs(0),
                    average_session_score: 0.0,
                    contribution_score: 0.0,
                    collaboration_rating: 0.0,
                },
                status: ParticipantStatus::Active,
            },
        );

        let group = LearningGroup {
            group_id: group_id.clone(),
            name,
            description,
            group_type,
            participants,
            settings,
            created_at: Utc::now(),
            status: GroupStatus::Active,
            current_session: None,
            stats: GroupStatistics {
                total_sessions: 0,
                total_participant_hours: Duration::from_secs(0),
                average_session_duration: Duration::from_secs(0),
                average_participation_rate: 0.0,
                completion_rates: HashMap::new(),
                achievement_rate: 0.0,
            },
        };

        let mut groups = self.groups.write().await;
        groups.insert(group_id.clone(), group);

        // Broadcast group creation
        let _ = self.event_broadcaster.send(GroupEvent::GroupCreated {
            group_id: group_id.clone(),
        });

        Ok(group_id)
    }

    /// Add participant to a group
    pub async fn add_participant(
        &self,
        group_id: &str,
        user_id: String,
        display_name: String,
        role: ParticipantRole,
    ) -> GroupLearningResult<()> {
        let mut groups = self.groups.write().await;
        let group = groups
            .get_mut(group_id)
            .ok_or_else(|| GroupLearningError::GroupNotFound {
                group_id: group_id.to_string(),
            })?;

        // Check capacity
        if group.participants.len() >= group.settings.max_participants {
            return Err(GroupLearningError::GroupCapacityExceeded {
                current: group.participants.len(),
                max: group.settings.max_participants,
            });
        }

        let participant = GroupParticipant {
            user_id: user_id.clone(),
            display_name,
            role,
            joined_at: Utc::now(),
            stats: ParticipationStats {
                sessions_attended: 0,
                total_participation_time: Duration::from_secs(0),
                average_session_score: 0.0,
                contribution_score: 0.0,
                collaboration_rating: 0.0,
            },
            status: ParticipantStatus::Active,
        };

        group.participants.insert(user_id.clone(), participant);

        // Broadcast participant joined
        let _ = self.event_broadcaster.send(GroupEvent::ParticipantJoined {
            group_id: group_id.to_string(),
            user_id: user_id.clone(),
        });

        Ok(())
    }

    /// Start a synchronized group session
    pub async fn start_session(
        &self,
        group_id: &str,
        leader_id: &str,
        title: String,
        description: Option<String>,
        exercises: Vec<SessionExercise>,
        settings: SessionSettings,
    ) -> GroupLearningResult<String> {
        let session_id = Uuid::new_v4().to_string();

        // Verify group exists and leader has permissions
        let groups = self.groups.read().await;
        let group = groups
            .get(group_id)
            .ok_or_else(|| GroupLearningError::GroupNotFound {
                group_id: group_id.to_string(),
            })?;

        let leader = group.participants.get(leader_id).ok_or_else(|| {
            GroupLearningError::UserNotInGroup {
                user_id: leader_id.to_string(),
            }
        })?;

        // Check leader permissions
        match leader.role {
            ParticipantRole::Leader | ParticipantRole::Facilitator => {}
            _ => {
                return Err(GroupLearningError::InsufficientPermissions {
                    operation: "start_session".to_string(),
                })
            }
        }

        drop(groups); // Release read lock

        // Create session participants from group participants
        let mut session_participants = HashMap::new();
        let groups = self.groups.read().await;
        let group = groups.get(group_id).unwrap();

        for (user_id, participant) in &group.participants {
            if matches!(
                participant.status,
                ParticipantStatus::Active | ParticipantStatus::Idle
            ) {
                session_participants.insert(
                    user_id.clone(),
                    SessionParticipant {
                        user_id: user_id.clone(),
                        joined_at: Utc::now(),
                        status: SessionParticipantStatus::Waiting,
                        exercise_progress: HashMap::new(),
                        session_scores: SessionScores::default(),
                        realtime_status: ParticipantRealtimeStatus {
                            last_activity: Utc::now(),
                            device_info: "Unknown".to_string(),
                            network_quality: NetworkQuality::Good,
                            media_status: MediaStatus {
                                audio_enabled: false,
                                video_enabled: false,
                                screen_sharing: false,
                            },
                        },
                    },
                );
            }
        }

        drop(groups);

        let estimated_duration = exercises.iter().map(|e| e.time_allocation).sum();

        let session = GroupSession {
            session_id: session_id.clone(),
            group_id: group_id.to_string(),
            leader_id: leader_id.to_string(),
            title,
            description,
            exercises,
            current_exercise_index: 0,
            participants: session_participants,
            started_at: Utc::now(),
            estimated_duration,
            settings,
            status: SessionStatus::Starting,
            realtime_data: SessionRealtimeData {
                active_participants: 0,
                current_exercise_timing: ExerciseTiming {
                    started_at: None,
                    elapsed_time: Duration::from_secs(0),
                    estimated_remaining: Duration::from_secs(0),
                    time_pressure: TimePressure::Low,
                },
                session_progress: 0.0,
                sync_status: SynchronizationStatus {
                    is_synchronized: true,
                    waiting_participants: Vec::new(),
                    ready_participants: Vec::new(),
                    timeout_remaining: None,
                },
            },
        };

        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id.clone(), session);

        // Update group current session
        let mut groups = self.groups.write().await;
        if let Some(group) = groups.get_mut(group_id) {
            group.current_session = Some(session_id.clone());
            group.status = GroupStatus::InSession;
        }

        // Broadcast session started
        let _ = self.event_broadcaster.send(GroupEvent::SessionStarted {
            group_id: group_id.to_string(),
            session_id: session_id.clone(),
        });

        Ok(session_id)
    }

    /// Subscribe to group events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<GroupEvent> {
        self.event_broadcaster.subscribe()
    }

    /// Get group information
    pub async fn get_group(&self, group_id: &str) -> GroupLearningResult<LearningGroup> {
        let groups = self.groups.read().await;
        groups
            .get(group_id)
            .cloned()
            .ok_or_else(|| GroupLearningError::GroupNotFound {
                group_id: group_id.to_string(),
            })
    }

    /// Get session information
    pub async fn get_session(&self, session_id: &str) -> GroupLearningResult<GroupSession> {
        let sessions = self.sessions.read().await;
        sessions
            .get(session_id)
            .cloned()
            .ok_or_else(|| GroupLearningError::GroupNotFound {
                group_id: session_id.to_string(),
            })
    }
}

/// Classroom management functionality
#[derive(Debug)]
pub struct ClassroomManager {
    /// Virtual classrooms
    classrooms: RwLock<HashMap<String, VirtualClassroom>>,
}

/// Virtual classroom environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualClassroom {
    /// Classroom ID
    pub classroom_id: String,
    /// Classroom name
    pub name: String,
    /// Associated group
    pub group_id: Option<String>,
    /// Classroom settings
    pub settings: ClassroomSettings,
    /// Interactive features
    pub features: ClassroomFeatures,
    /// Current status
    pub status: ClassroomStatus,
}

/// Classroom configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassroomSettings {
    /// Maximum occupancy
    pub max_occupancy: usize,
    /// Layout configuration
    pub layout: ClassroomLayout,
    /// Audio/video settings
    pub media_settings: ClassroomMediaSettings,
    /// Interactive tools enabled
    pub tools_enabled: Vec<ClassroomTool>,
}

/// Classroom layout options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClassroomLayout {
    /// Traditional rows
    Traditional,
    /// Circle/discussion format
    Circle,
    /// Small group pods
    SmallGroups,
    /// Flexible/customizable
    Flexible,
}

/// Media settings for classrooms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassroomMediaSettings {
    /// Default audio quality
    pub audio_quality: AudioQuality,
    /// Default video quality  
    pub video_quality: VideoQuality,
    /// Whether to enable spatial audio
    pub spatial_audio: bool,
    /// Background noise suppression
    pub noise_suppression: bool,
}

/// Audio quality levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioQuality {
    /// Standard quality
    Standard,
    /// High quality
    High,
    /// Studio quality
    Studio,
}

/// Video quality levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VideoQuality {
    /// Low resolution
    Low,
    /// Standard definition
    SD,
    /// High definition
    HD,
    /// Ultra high definition
    UHD,
}

/// Interactive classroom tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClassroomTool {
    /// Virtual whiteboard
    Whiteboard,
    /// Screen sharing
    ScreenShare,
    /// Breakout rooms
    BreakoutRooms,
    /// Polling/quizzes
    Polling,
    /// Hand raising
    HandRaising,
    /// Chat
    Chat,
    /// File sharing
    FileSharing,
}

/// Interactive features available in classrooms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassroomFeatures {
    /// Whiteboard availability
    pub whiteboard_enabled: bool,
    /// Breakout room support
    pub breakout_rooms: bool,
    /// Recording capability
    pub recording_enabled: bool,
    /// Real-time translation
    pub translation_enabled: bool,
    /// Accessibility features
    pub accessibility_features: Vec<AccessibilityFeature>,
}

/// Accessibility features for inclusive learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessibilityFeature {
    /// Closed captions
    ClosedCaptions,
    /// Sign language interpretation
    SignLanguage,
    /// Screen reader support
    ScreenReader,
    /// High contrast mode
    HighContrast,
    /// Font size adjustment
    FontAdjustment,
    /// Audio descriptions
    AudioDescriptions,
}

/// Current status of a classroom
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClassroomStatus {
    /// Available for use
    Available,
    /// Currently in use
    InUse,
    /// Under maintenance
    Maintenance,
    /// Temporarily unavailable
    Unavailable,
}

impl ClassroomManager {
    /// Create new classroom manager
    pub fn new() -> Self {
        Self {
            classrooms: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new virtual classroom
    pub async fn create_classroom(
        &self,
        name: String,
        settings: ClassroomSettings,
        features: ClassroomFeatures,
    ) -> String {
        let classroom_id = Uuid::new_v4().to_string();

        let classroom = VirtualClassroom {
            classroom_id: classroom_id.clone(),
            name,
            group_id: None,
            settings,
            features,
            status: ClassroomStatus::Available,
        };

        let mut classrooms = self.classrooms.write().await;
        classrooms.insert(classroom_id.clone(), classroom);

        classroom_id
    }
}

/// Group progress tracking system
#[derive(Debug)]
pub struct GroupProgressTracker {
    /// Progress data storage
    progress_data: RwLock<HashMap<String, GroupProgressData>>,
}

/// Progress data for groups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupProgressData {
    /// Group ID
    pub group_id: String,
    /// Individual progress per participant
    pub individual_progress: HashMap<String, UserProgress>,
    /// Collective progress metrics
    pub collective_metrics: CollectiveProgressMetrics,
    /// Progress history
    pub progress_history: Vec<ProgressSnapshot>,
    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Collective progress metrics for groups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveProgressMetrics {
    /// Average skill level across group
    pub average_skill_level: f32,
    /// Group cohesion score
    pub cohesion_score: f32,
    /// Collaboration effectiveness
    pub collaboration_effectiveness: f32,
    /// Collective achievements
    pub collective_achievements: Vec<Achievement>,
    /// Milestone progress
    pub milestone_progress: HashMap<String, f32>,
}

/// Progress snapshot for historical tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressSnapshot {
    /// Timestamp of snapshot
    pub timestamp: DateTime<Utc>,
    /// Snapshot data
    pub data: GroupProgressData,
    /// Notable events during this period
    pub events: Vec<String>,
}

impl GroupProgressTracker {
    /// Create new progress tracker
    pub fn new() -> Self {
        Self {
            progress_data: RwLock::new(HashMap::new()),
        }
    }

    /// Update progress for a group
    pub async fn update_group_progress(
        &self,
        group_id: &str,
        participant_progress: HashMap<String, UserProgress>,
    ) -> GroupLearningResult<()> {
        let mut progress_data = self.progress_data.write().await;

        // Calculate collective metrics
        let collective_metrics = self
            .calculate_collective_metrics(&participant_progress)
            .await;

        // Get existing progress data to preserve history
        let mut progress_history = if let Some(existing) = progress_data.get(group_id) {
            let mut history = existing.progress_history.clone();

            // Create snapshot of current state before updating
            let snapshot = ProgressSnapshot {
                timestamp: existing.last_updated,
                data: existing.clone(),
                events: self
                    .generate_progress_events(&existing.collective_metrics, &collective_metrics)
                    .await,
            };

            history.push(snapshot);

            // Keep only last 50 snapshots to prevent memory bloat
            if history.len() > 50 {
                let skip_count = history.len() - 50;
                history = history.into_iter().skip(skip_count).collect();
            }

            history
        } else {
            Vec::new()
        };

        let group_progress = GroupProgressData {
            group_id: group_id.to_string(),
            individual_progress: participant_progress,
            collective_metrics,
            progress_history,
            last_updated: Utc::now(),
        };

        progress_data.insert(group_id.to_string(), group_progress);

        Ok(())
    }

    /// Calculate collective metrics from individual progress
    async fn calculate_collective_metrics(
        &self,
        progress: &HashMap<String, UserProgress>,
    ) -> CollectiveProgressMetrics {
        let participant_count = progress.len() as f32;

        let average_skill_level = if participant_count > 0.0 {
            progress
                .values()
                .map(|p| p.overall_skill_level)
                .sum::<f32>()
                / participant_count
        } else {
            0.0
        };

        // Simplified cohesion calculation based on skill level variance
        let skill_variance = if participant_count > 1.0 {
            let avg = average_skill_level;
            let variance = progress
                .values()
                .map(|p| (p.overall_skill_level - avg).powi(2))
                .sum::<f32>()
                / participant_count;
            1.0 - (variance.sqrt() / 1.0).min(1.0) // Inverted normalized variance
        } else {
            1.0
        };

        // Calculate collaboration effectiveness based on multiple factors
        let collaboration_effectiveness = self
            .calculate_collaboration_effectiveness(progress, skill_variance)
            .await;

        CollectiveProgressMetrics {
            average_skill_level,
            cohesion_score: skill_variance,
            collaboration_effectiveness,
            collective_achievements: Vec::new(),
            milestone_progress: HashMap::new(),
        }
    }

    /// Calculate collaboration effectiveness from various metrics
    async fn calculate_collaboration_effectiveness(
        &self,
        progress: &HashMap<String, UserProgress>,
        cohesion_score: f32,
    ) -> f32 {
        if progress.is_empty() {
            return 0.0;
        }

        let participant_count = progress.len() as f32;

        // Factor 1: Group cohesion (already calculated)
        let cohesion_factor = cohesion_score;

        // Factor 2: Average consistency across participants (based on success rate)
        let consistency_factor = if participant_count > 0.0 {
            progress
                .values()
                .map(|p| p.training_stats.success_rate)
                .sum::<f32>()
                / participant_count
        } else {
            0.0
        };

        // Factor 3: Engagement level (based on recent sessions and training frequency)
        let engagement_factor = if participant_count > 0.0 {
            progress
                .values()
                .map(|p| {
                    // Combine recent session activity with overall training statistics
                    let activity_score = p.recent_sessions.len() as f32 / 10.0; // Normalize to 0-1
                    let training_momentum = if p.training_stats.total_sessions == 0 {
                        0.0
                    } else {
                        p.training_stats.average_improvement.min(1.0).max(0.0)
                    };
                    (activity_score.min(1.0) + training_momentum) / 2.0
                })
                .sum::<f32>()
                / participant_count
        } else {
            0.0
        };

        // Factor 4: Learning velocity compatibility (how well participants learn together)
        let velocity_compatibility = if participant_count > 1.0 {
            let avg_improvement = progress
                .values()
                .map(|p| p.training_stats.average_improvement)
                .sum::<f32>()
                / participant_count;

            let improvement_variance = progress
                .values()
                .map(|p| (p.training_stats.average_improvement - avg_improvement).abs())
                .sum::<f32>()
                / participant_count;

            // Lower variance means better compatibility
            1.0 - (improvement_variance / 2.0).min(1.0)
        } else {
            1.0
        };

        // Weighted combination of factors
        let weighted_effectiveness = (cohesion_factor * 0.3
            + consistency_factor * 0.25
            + engagement_factor * 0.25
            + velocity_compatibility * 0.2)
            .min(1.0)
            .max(0.0);

        weighted_effectiveness
    }

    /// Generate events that occurred between two collective metrics snapshots
    async fn generate_progress_events(
        &self,
        previous: &CollectiveProgressMetrics,
        current: &CollectiveProgressMetrics,
    ) -> Vec<String> {
        let mut events = Vec::new();

        // Check for significant skill level changes
        let skill_change = current.average_skill_level - previous.average_skill_level;
        if skill_change.abs() > 0.1 {
            if skill_change > 0.0 {
                events.push(format!(
                    "Group skill level increased by {:.2}",
                    skill_change
                ));
            } else {
                events.push(format!(
                    "Group skill level decreased by {:.2}",
                    skill_change.abs()
                ));
            }
        }

        // Check for cohesion changes
        let cohesion_change = current.cohesion_score - previous.cohesion_score;
        if cohesion_change.abs() > 0.1 {
            if cohesion_change > 0.0 {
                events.push("Group cohesion improved".to_string());
            } else {
                events.push("Group cohesion declined".to_string());
            }
        }

        // Check for collaboration effectiveness changes
        let collab_change =
            current.collaboration_effectiveness - previous.collaboration_effectiveness;
        if collab_change.abs() > 0.1 {
            if collab_change > 0.0 {
                events.push("Collaboration effectiveness improved".to_string());
            } else {
                events.push("Collaboration effectiveness declined".to_string());
            }
        }

        // Check for new achievements
        if current.collective_achievements.len() > previous.collective_achievements.len() {
            let new_achievements =
                current.collective_achievements.len() - previous.collective_achievements.len();
            events.push(format!(
                "Group earned {} new achievement(s)",
                new_achievements
            ));
        }

        events
    }

    /// Get progress history for a group
    pub async fn get_group_progress_history(
        &self,
        group_id: &str,
    ) -> Option<Vec<ProgressSnapshot>> {
        let progress_data = self.progress_data.read().await;
        progress_data
            .get(group_id)
            .map(|data| data.progress_history.clone())
    }

    /// Get group progress data including history
    pub async fn get_group_progress(&self, group_id: &str) -> Option<GroupProgressData> {
        let progress_data = self.progress_data.read().await;
        progress_data.get(group_id).cloned()
    }
}

/// Collaborative challenge coordination system
#[derive(Debug)]
pub struct CollaborativeChallengeCoordinator {
    /// Active challenges
    challenges: RwLock<HashMap<String, CollaborativeChallenge>>,
}

/// Collaborative challenge definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeChallenge {
    /// Challenge ID
    pub challenge_id: String,
    /// Challenge title
    pub title: String,
    /// Challenge description
    pub description: String,
    /// Challenge type
    pub challenge_type: ChallengeType,
    /// Participating groups
    pub participating_groups: Vec<String>,
    /// Challenge objectives
    pub objectives: Vec<ChallengeObjective>,
    /// Duration
    pub duration: Duration,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: DateTime<Utc>,
    /// Challenge status
    pub status: ChallengeStatus,
    /// Rewards
    pub rewards: ChallengeRewards,
}

/// Types of collaborative challenges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeType {
    /// Competition between groups
    InterGroupCompetition,
    /// Collaborative project
    CollaborativeProject,
    /// Community challenge
    CommunityChallenge,
    /// Skill-building marathon
    SkillMarathon,
    /// Knowledge quest
    KnowledgeQuest,
}

/// Challenge objective definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeObjective {
    /// Objective ID
    pub objective_id: String,
    /// Objective description
    pub description: String,
    /// Target metrics
    pub target_metrics: HashMap<String, f32>,
    /// Points awarded for completion
    pub points: u32,
    /// Completion status
    pub completed: bool,
}

/// Challenge status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeStatus {
    /// Challenge is planned
    Planned,
    /// Challenge is active
    Active,
    /// Challenge is paused
    Paused,
    /// Challenge is completed
    Completed,
    /// Challenge was cancelled
    Cancelled,
}

/// Rewards for challenge completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeRewards {
    /// Points awarded
    pub points: u32,
    /// Badges earned
    pub badges: Vec<String>,
    /// Special achievements
    pub achievements: Vec<Achievement>,
    /// Additional rewards
    pub additional_rewards: HashMap<String, String>,
}

impl CollaborativeChallengeCoordinator {
    /// Create new challenge coordinator
    pub fn new() -> Self {
        Self {
            challenges: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new collaborative challenge
    pub async fn create_challenge(
        &self,
        title: String,
        description: String,
        challenge_type: ChallengeType,
        objectives: Vec<ChallengeObjective>,
        duration: Duration,
        rewards: ChallengeRewards,
    ) -> String {
        let challenge_id = Uuid::new_v4().to_string();
        let start_time = Utc::now();
        let end_time = start_time + chrono::Duration::from_std(duration).unwrap();

        let challenge = CollaborativeChallenge {
            challenge_id: challenge_id.clone(),
            title,
            description,
            challenge_type,
            participating_groups: Vec::new(),
            objectives,
            duration,
            start_time,
            end_time,
            status: ChallengeStatus::Planned,
            rewards,
        };

        let mut challenges = self.challenges.write().await;
        challenges.insert(challenge_id.clone(), challenge);

        challenge_id
    }
}

// Default implementations for various types
impl Default for GroupSettings {
    fn default() -> Self {
        Self {
            max_participants: 20,
            is_public: false,
            allow_late_join: true,
            sync_settings: SynchronizationSettings::default(),
            privacy_settings: PrivacySettings::default(),
        }
    }
}

impl Default for SynchronizationSettings {
    fn default() -> Self {
        Self {
            sync_exercises: true,
            sync_mode: SyncMode::FlexibleSync,
            sync_timeout: Duration::from_secs(30),
            allow_individual_pacing: false,
        }
    }
}

impl Default for PrivacySettings {
    fn default() -> Self {
        Self {
            show_scores: true,
            show_progress: true,
            allow_media_sharing: false,
            data_retention: DataRetentionSettings::default(),
        }
    }
}

impl Default for DataRetentionSettings {
    fn default() -> Self {
        Self {
            session_recording_retention: Duration::from_secs(30 * 24 * 3600), // 30 days
            chat_retention: Duration::from_secs(7 * 24 * 3600),               // 7 days
            performance_data_retention: Duration::from_secs(90 * 24 * 3600),  // 90 days
        }
    }
}

impl Default for SessionSettings {
    fn default() -> Self {
        Self {
            record_session: false,
            enable_chat: true,
            enable_media: false,
            auto_advance: AutoAdvanceSettings::default(),
        }
    }
}

impl Default for AutoAdvanceSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            advance_threshold: 0.8, // 80% of participants
            max_wait_time: Duration::from_secs(60),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_group_creation() {
        let orchestrator = GroupLearningOrchestrator::new();

        let group_id = orchestrator
            .create_group(
                "Test Group".to_string(),
                "A test learning group".to_string(),
                GroupType::StudyGroup,
                "creator123".to_string(),
                GroupSettings::default(),
            )
            .await
            .unwrap();

        assert!(!group_id.is_empty());

        let group = orchestrator.get_group(&group_id).await.unwrap();
        assert_eq!(group.name, "Test Group");
        assert_eq!(group.participants.len(), 1);
        assert!(group.participants.contains_key("creator123"));
    }

    #[tokio::test]
    async fn test_participant_addition() {
        let orchestrator = GroupLearningOrchestrator::new();

        let group_id = orchestrator
            .create_group(
                "Test Group".to_string(),
                "A test learning group".to_string(),
                GroupType::StudyGroup,
                "creator123".to_string(),
                GroupSettings::default(),
            )
            .await
            .unwrap();

        let result = orchestrator
            .add_participant(
                &group_id,
                "participant456".to_string(),
                "Test Participant".to_string(),
                ParticipantRole::Participant,
            )
            .await;

        assert!(result.is_ok());

        let group = orchestrator.get_group(&group_id).await.unwrap();
        assert_eq!(group.participants.len(), 2);
        assert!(group.participants.contains_key("participant456"));
    }

    #[tokio::test]
    async fn test_session_creation() {
        let orchestrator = GroupLearningOrchestrator::new();

        let group_id = orchestrator
            .create_group(
                "Test Group".to_string(),
                "A test learning group".to_string(),
                GroupType::StudyGroup,
                "creator123".to_string(),
                GroupSettings::default(),
            )
            .await
            .unwrap();

        let exercises = vec![SessionExercise {
            exercise: TrainingExercise::default(),
            session_settings: ExerciseSessionSettings {
                is_mandatory: true,
                show_peer_progress: true,
                enable_peer_feedback: false,
                scoring_mode: ScoringMode::Individual,
            },
            sync_requirements: ExerciseSyncRequirements {
                synchronized_start: true,
                synchronized_finish: false,
                max_wait_time: Duration::from_secs(30),
                min_participants_to_proceed: 1,
            },
            time_allocation: Duration::from_secs(300),
            status: ExerciseStatus::Planned,
        }];

        let session_id = orchestrator
            .start_session(
                &group_id,
                "creator123",
                "Test Session".to_string(),
                Some("A test session".to_string()),
                exercises,
                SessionSettings::default(),
            )
            .await
            .unwrap();

        assert!(!session_id.is_empty());

        let session = orchestrator.get_session(&session_id).await.unwrap();
        assert_eq!(session.title, "Test Session");
        assert_eq!(session.exercises.len(), 1);
    }

    #[test]
    fn test_classroom_manager() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let manager = ClassroomManager::new();

            let classroom_id = manager
                .create_classroom(
                    "Test Classroom".to_string(),
                    ClassroomSettings {
                        max_occupancy: 30,
                        layout: ClassroomLayout::Circle,
                        media_settings: ClassroomMediaSettings {
                            audio_quality: AudioQuality::High,
                            video_quality: VideoQuality::HD,
                            spatial_audio: true,
                            noise_suppression: true,
                        },
                        tools_enabled: vec![ClassroomTool::Whiteboard, ClassroomTool::Chat],
                    },
                    ClassroomFeatures {
                        whiteboard_enabled: true,
                        breakout_rooms: false,
                        recording_enabled: false,
                        translation_enabled: false,
                        accessibility_features: vec![AccessibilityFeature::ClosedCaptions],
                    },
                )
                .await;

            assert!(!classroom_id.is_empty());
        });
    }

    #[tokio::test]
    async fn test_progress_tracker() {
        let tracker = GroupProgressTracker::new();

        let mut participant_progress = HashMap::new();
        participant_progress.insert("user1".to_string(), UserProgress::default());
        participant_progress.insert("user2".to_string(), UserProgress::default());

        let result = tracker
            .update_group_progress("group123", participant_progress)
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_progress_history_tracking() {
        let tracker = GroupProgressTracker::new();

        // First update
        let mut progress1 = HashMap::new();
        let mut user_progress1 = UserProgress::default();
        user_progress1.overall_skill_level = 0.5;
        progress1.insert("user1".to_string(), user_progress1);

        tracker
            .update_group_progress("group123", progress1)
            .await
            .unwrap();

        // Second update - should create history entry
        let mut progress2 = HashMap::new();
        let mut user_progress2 = UserProgress::default();
        user_progress2.overall_skill_level = 0.7;
        progress2.insert("user1".to_string(), user_progress2);

        tracker
            .update_group_progress("group123", progress2)
            .await
            .unwrap();

        // Check that history was created
        let history = tracker.get_group_progress_history("group123").await;
        assert!(history.is_some());
        let history = history.unwrap();
        assert_eq!(history.len(), 1); // Should have one historical snapshot

        // Check that the current progress is accessible
        let current_progress = tracker.get_group_progress("group123").await;
        assert!(current_progress.is_some());
        let current_progress = current_progress.unwrap();
        assert_eq!(current_progress.collective_metrics.average_skill_level, 0.7);
    }

    #[tokio::test]
    async fn test_collaboration_effectiveness_calculation() {
        let tracker = GroupProgressTracker::new();

        // Create progress data with varied statistics to test collaboration effectiveness
        let mut participant_progress = HashMap::new();

        let mut user1_progress = UserProgress::default();
        user1_progress.overall_skill_level = 0.8;
        user1_progress.training_stats.success_rate = 0.9;
        user1_progress.training_stats.average_improvement = 0.1;
        user1_progress.training_stats.total_sessions = 10;

        let mut user2_progress = UserProgress::default();
        user2_progress.overall_skill_level = 0.75;
        user2_progress.training_stats.success_rate = 0.85;
        user2_progress.training_stats.average_improvement = 0.12;
        user2_progress.training_stats.total_sessions = 8;

        participant_progress.insert("user1".to_string(), user1_progress);
        participant_progress.insert("user2".to_string(), user2_progress);

        tracker
            .update_group_progress("group123", participant_progress)
            .await
            .unwrap();

        let progress_data = tracker.get_group_progress("group123").await.unwrap();

        // Collaboration effectiveness should be calculated (not the hardcoded 0.75)
        let effectiveness = progress_data.collective_metrics.collaboration_effectiveness;
        assert!(effectiveness >= 0.0 && effectiveness <= 1.0);
        assert_ne!(effectiveness, 0.75); // Should not be the old hardcoded value

        // With good success rates and similar improvement rates, effectiveness should be reasonably high
        assert!(
            effectiveness > 0.5,
            "Collaboration effectiveness should be > 0.5 for good performers"
        );
    }

    #[tokio::test]
    async fn test_challenge_coordinator() {
        let coordinator = CollaborativeChallengeCoordinator::new();

        let objectives = vec![ChallengeObjective {
            objective_id: "obj1".to_string(),
            description: "Complete 10 exercises".to_string(),
            target_metrics: {
                let mut map = HashMap::new();
                map.insert("exercises_completed".to_string(), 10.0);
                map
            },
            points: 100,
            completed: false,
        }];

        let rewards = ChallengeRewards {
            points: 500,
            badges: vec!["Collaborator".to_string()],
            achievements: Vec::new(),
            additional_rewards: HashMap::new(),
        };

        let challenge_id = coordinator
            .create_challenge(
                "Team Challenge".to_string(),
                "A collaborative team challenge".to_string(),
                ChallengeType::CollaborativeProject,
                objectives,
                Duration::from_secs(7 * 24 * 3600), // 7 days
                rewards,
            )
            .await;

        assert!(!challenge_id.is_empty());
    }
}
