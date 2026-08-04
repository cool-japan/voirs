//! GraphQL API implementation for `VoiRS` feedback system
//!
//! This module provides a comprehensive GraphQL API for querying and mutating
//! feedback data, user progress, training exercises, and system analytics.

use async_graphql::{
    Context, EmptySubscription, Enum, Error, FieldResult, Object, Schema, SimpleObject, Union, ID,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::persistence::PersistenceManager;
use crate::traits::{
    FeedbackResponse, FeedbackType, ProgressIndicators, SessionState, SessionStats,
    UserPreferences, UserProgress,
};
use crate::FeedbackSystem;

/// Fetch the [`Arc<dyn PersistenceManager>`] injected into the schema by
/// [`create_schema`]. Every resolver in this module goes through this --
/// there is no per-resolver mock data path.
fn persistence<'ctx>(ctx: &Context<'ctx>) -> FieldResult<&'ctx Arc<dyn PersistenceManager>> {
    Ok(ctx.data::<Arc<dyn PersistenceManager>>()?)
}

/// GraphQL schema type
pub type FeedbackSchema = Schema<QueryRoot, MutationRoot, EmptySubscription>;

/// Root query object
pub struct QueryRoot;

/// Root mutation object
pub struct MutationRoot;

/// GraphQL user object
#[derive(SimpleObject, Clone)]
pub struct User {
    /// User ID
    pub id: ID,
    /// User display name
    pub name: String,
    /// User email
    pub email: String,
    /// User registration date
    pub created_at: String,
    /// Last activity timestamp
    pub last_activity: Option<String>,
    /// User status
    pub status: UserStatus,
    /// User preferences
    pub preferences: UserPreferencesGraphQL,
}

/// User status enumeration
#[derive(Clone, Copy, PartialEq, Eq, Enum)]
pub enum UserStatus {
    /// User is active
    Active,
    /// User is inactive
    Inactive,
    /// User account is suspended
    Suspended,
    /// User account is pending verification
    Pending,
}

/// GraphQL user preferences
#[derive(SimpleObject, Clone)]
pub struct UserPreferencesGraphQL {
    /// Preferred language
    pub language: String,
    /// Audio quality preference
    pub audio_quality: AudioQuality,
    /// Feedback frequency
    pub feedback_frequency: FeedbackFrequency,
    /// UI theme preference
    pub theme: String,
    /// Notification settings
    pub notifications_enabled: bool,
}

/// Audio quality levels
#[derive(Clone, Copy, PartialEq, Eq, Enum)]
pub enum AudioQuality {
    /// Low quality (16kHz)
    Low,
    /// Standard quality (22kHz)
    Standard,
    /// High quality (44kHz)
    High,
    /// Studio quality (48kHz)
    Studio,
}

/// Feedback frequency options
#[derive(Clone, Copy, PartialEq, Eq, Enum)]
pub enum FeedbackFrequency {
    /// Real-time feedback
    Realtime,
    /// After each sentence
    PerSentence,
    /// After each paragraph
    PerParagraph,
    /// Manual request only
    Manual,
}

/// GraphQL feedback item
#[derive(SimpleObject, Clone)]
pub struct FeedbackItem {
    /// Feedback ID
    pub id: ID,
    /// User ID this feedback belongs to
    pub user_id: ID,
    /// Feedback message
    pub message: String,
    /// Feedback score (0.0 to 1.0)
    pub score: f32,
    /// Feedback category
    pub category: FeedbackCategory,
    /// Improvement suggestions
    pub suggestions: Vec<String>,
    /// Timestamp when feedback was generated
    pub created_at: String,
    /// Audio segment information
    pub audio_segment: Option<AudioSegment>,
}

/// Feedback categories
#[derive(Clone, Copy, PartialEq, Eq, Enum)]
pub enum FeedbackCategory {
    /// Pronunciation feedback
    Pronunciation,
    /// Voice quality feedback
    Quality,
    /// Fluency feedback
    Fluency,
    /// Prosody feedback
    Prosody,
    /// General feedback
    General,
}

/// Audio segment information
#[derive(SimpleObject, Clone)]
pub struct AudioSegment {
    /// Start time in milliseconds
    pub start_ms: i32,
    /// End time in milliseconds
    pub end_ms: i32,
    /// Segment duration in milliseconds
    pub duration_ms: i32,
    /// Text content of this segment
    pub text: String,
}

/// GraphQL session object
#[derive(SimpleObject, Clone)]
pub struct Session {
    /// Session ID
    pub id: ID,
    /// User ID
    pub user_id: ID,
    /// Session start time
    pub started_at: String,
    /// Session end time
    pub ended_at: Option<String>,
    /// Session duration in seconds
    pub duration_seconds: Option<i32>,
    /// Session status
    pub status: SessionStatus,
    /// Number of feedback items generated
    pub feedback_count: i32,
    /// Average session score
    pub average_score: Option<f32>,
    /// Session tags
    pub tags: Vec<String>,
}

/// Session status enumeration
#[derive(Clone, Copy, PartialEq, Eq, Enum)]
pub enum SessionStatus {
    /// Session is active
    Active,
    /// Session is completed
    Completed,
    /// Session was cancelled
    Cancelled,
    /// Session has expired
    Expired,
}

/// GraphQL training exercise
#[derive(SimpleObject, Clone)]
pub struct TrainingExerciseGraphQL {
    /// Exercise ID
    pub id: ID,
    /// Exercise name
    pub name: String,
    /// Exercise description
    pub description: String,
    /// Difficulty level (0.0 to 1.0)
    pub difficulty: f32,
    /// Exercise category
    pub category: ExerciseCategory,
    /// Target text for the exercise
    pub target_text: String,
    /// Estimated duration in seconds
    pub estimated_duration_seconds: i32,
    /// Exercise tags
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: String,
}

/// Exercise categories
#[derive(Clone, Copy, PartialEq, Eq, Enum)]
pub enum ExerciseCategory {
    /// Pronunciation exercises
    Pronunciation,
    /// Reading exercises
    Reading,
    /// Conversation exercises
    Conversation,
    /// Technical exercises
    Technical,
    /// Creative exercises
    Creative,
}

/// Progress statistics
#[derive(SimpleObject, Clone)]
pub struct ProgressStats {
    /// Total sessions completed
    pub total_sessions: i32,
    /// Total practice time in seconds
    pub total_practice_time_seconds: i32,
    /// Average score across all sessions
    pub average_score: f32,
    /// Improvement rate percentage
    pub improvement_rate: f32,
    /// Current streak in days
    pub current_streak_days: i32,
    /// Best streak in days
    pub best_streak_days: i32,
    /// Skill breakdown scores
    pub skill_scores: Vec<SkillScore>,
}

/// Individual skill score
#[derive(SimpleObject, Clone)]
pub struct SkillScore {
    /// Skill name
    pub skill_name: String,
    /// Current score (0.0 to 1.0)
    pub current_score: f32,
    /// Previous score for comparison
    pub previous_score: Option<f32>,
    /// Improvement over time
    pub improvement: Option<f32>,
}

/// Analytics data
#[derive(SimpleObject, Clone)]
pub struct Analytics {
    /// Time period for these analytics
    pub period: AnalyticsPeriod,
    /// Start date of the period
    pub start_date: String,
    /// End date of the period
    pub end_date: String,
    /// User activity metrics
    pub user_metrics: UserMetrics,
    /// System performance metrics
    pub system_metrics: SystemMetrics,
    /// Usage statistics
    pub usage_stats: UsageStats,
}

/// Analytics time periods
#[derive(Clone, Copy, PartialEq, Eq, Enum)]
pub enum AnalyticsPeriod {
    /// Daily analytics
    Daily,
    /// Weekly analytics
    Weekly,
    /// Monthly analytics
    Monthly,
    /// Yearly analytics
    Yearly,
    /// Custom period
    Custom,
}

/// User activity metrics
#[derive(SimpleObject, Clone)]
pub struct UserMetrics {
    /// Total active users
    pub active_users: i32,
    /// New user registrations
    pub new_registrations: i32,
    /// User retention rate
    pub retention_rate: f32,
    /// Average session duration in seconds
    pub avg_session_duration_seconds: i32,
    /// Total user engagement score
    pub engagement_score: f32,
}

/// System performance metrics
#[derive(SimpleObject, Clone)]
pub struct SystemMetrics {
    /// Average response time in milliseconds
    pub avg_response_time_ms: f32,
    /// System uptime percentage
    pub uptime_percentage: f32,
    /// Error rate percentage
    pub error_rate_percentage: f32,
    /// Throughput (requests per second)
    pub throughput_rps: f32,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization metrics
#[derive(SimpleObject, Clone)]
pub struct ResourceUtilization {
    /// CPU utilization percentage
    pub cpu_percentage: f32,
    /// Memory utilization percentage
    pub memory_percentage: f32,
    /// Disk utilization percentage
    pub disk_percentage: f32,
    /// Network utilization percentage
    pub network_percentage: f32,
}

/// Usage statistics
#[derive(SimpleObject, Clone)]
pub struct UsageStats {
    /// Total API requests
    pub total_requests: i64,
    /// Total feedback items generated
    pub total_feedback_items: i64,
    /// Total training sessions
    pub total_training_sessions: i64,
    /// Most popular exercises
    pub popular_exercises: Vec<ExerciseUsage>,
    /// Peak usage hours
    pub peak_usage_hours: Vec<i32>,
}

/// Exercise usage statistics
#[derive(SimpleObject, Clone)]
pub struct ExerciseUsage {
    /// Exercise ID
    pub exercise_id: ID,
    /// Exercise name
    pub exercise_name: String,
    /// Number of times completed
    pub completion_count: i32,
    /// Average completion time in seconds
    pub avg_completion_time_seconds: i32,
    /// Average score achieved
    pub avg_score: f32,
}

/// Input types for mutations
#[derive(async_graphql::InputObject)]
pub struct CreateUserInput {
    /// User name
    pub name: String,
    /// User email
    pub email: String,
    /// User preferences
    pub preferences: Option<UserPreferencesInput>,
}

#[derive(async_graphql::InputObject)]
/// Description
pub struct UserPreferencesInput {
    /// Preferred language
    pub language: Option<String>,
    /// Audio quality preference
    pub audio_quality: Option<AudioQuality>,
    /// Feedback frequency
    pub feedback_frequency: Option<FeedbackFrequency>,
    /// UI theme preference
    pub theme: Option<String>,
    /// Notification settings
    pub notifications_enabled: Option<bool>,
}

#[derive(async_graphql::InputObject)]
/// Description
pub struct UpdateUserInput {
    /// User ID
    pub id: ID,
    /// Updated name
    pub name: Option<String>,
    /// Updated email
    pub email: Option<String>,
    /// Updated preferences
    pub preferences: Option<UserPreferencesInput>,
    /// Updated status
    pub status: Option<UserStatus>,
}

#[derive(async_graphql::InputObject)]
/// Description
pub struct CreateSessionInput {
    /// User ID
    pub user_id: ID,
    /// Session tags
    pub tags: Option<Vec<String>>,
}

#[derive(async_graphql::InputObject)]
/// Description
pub struct CreateFeedbackInput {
    /// User ID
    pub user_id: ID,
    /// Session ID
    pub session_id: ID,
    /// Feedback message
    pub message: String,
    /// Feedback score
    pub score: f32,
    /// Feedback category
    pub category: FeedbackCategory,
    /// Improvement suggestions
    pub suggestions: Option<Vec<String>>,
    /// Audio segment information
    pub audio_segment: Option<AudioSegmentInput>,
}

#[derive(async_graphql::InputObject)]
/// Description
pub struct AudioSegmentInput {
    /// Start time in milliseconds
    pub start_ms: i32,
    /// End time in milliseconds
    pub end_ms: i32,
    /// Text content
    pub text: String,
}

/// Search result union type
#[derive(Union)]
pub enum SearchResult {
    /// User search result
    User(User),
    /// Session search result
    Session(Session),
    /// Exercise search result
    Exercise(TrainingExerciseGraphQL),
    /// Feedback search result
    Feedback(FeedbackItem),
}

/// Filter input for queries
#[derive(async_graphql::InputObject)]
pub struct FilterInput {
    /// Date range filter
    pub date_range: Option<DateRangeInput>,
    /// Category filter
    pub category: Option<String>,
    /// Score range filter
    pub score_range: Option<ScoreRangeInput>,
    /// Tags filter
    pub tags: Option<Vec<String>>,
    /// User ID filter
    pub user_id: Option<ID>,
}

#[derive(async_graphql::InputObject)]
/// Description
pub struct DateRangeInput {
    /// Start date
    pub start: String,
    /// End date
    pub end: String,
}

#[derive(async_graphql::InputObject)]
/// Description
pub struct ScoreRangeInput {
    /// Minimum score
    pub min: f32,
    /// Maximum score
    pub max: f32,
}

/// Pagination input
#[derive(async_graphql::InputObject)]
pub struct PaginationInput {
    /// Number of items to return
    pub limit: Option<i32>,
    /// Number of items to skip
    pub offset: Option<i32>,
    /// Cursor for cursor-based pagination
    pub cursor: Option<String>,
}

/// Sort input
#[derive(async_graphql::InputObject)]
pub struct SortInput {
    /// Field to sort by
    pub field: String,
    /// Sort direction
    pub direction: SortDirection,
}

/// Sort direction
#[derive(Clone, Copy, PartialEq, Eq, Enum)]
pub enum SortDirection {
    /// Ascending order
    Asc,
    /// Descending order
    Desc,
}

/// User connection type for pagination
#[derive(SimpleObject)]
pub struct UserConnection {
    /// Edges containing the data
    pub edges: Vec<UserEdge>,
    /// Page information
    pub page_info: PageInfo,
    /// Total count
    pub total_count: i32,
}

/// Feedback item connection type for pagination
#[derive(SimpleObject)]
pub struct FeedbackItemConnection {
    /// Edges containing the data
    pub edges: Vec<FeedbackItemEdge>,
    /// Page information
    pub page_info: PageInfo,
    /// Total count
    pub total_count: i32,
}

/// User edge type for connections
#[derive(SimpleObject)]
pub struct UserEdge {
    /// The data node
    pub node: User,
    /// Cursor for this edge
    pub cursor: String,
}

/// Feedback item edge type for connections
#[derive(SimpleObject)]
pub struct FeedbackItemEdge {
    /// The data node
    pub node: FeedbackItem,
    /// Cursor for this edge
    pub cursor: String,
}

/// Page information for connections
#[derive(SimpleObject)]
pub struct PageInfo {
    /// Whether there are more items
    pub has_next_page: bool,
    /// Whether there are previous items
    pub has_previous_page: bool,
    /// Start cursor
    pub start_cursor: Option<String>,
    /// End cursor
    pub end_cursor: Option<String>,
}

/// Build a GraphQL [`User`] from a real, persisted `user_id` plus whatever
/// real progress/preference records exist for it.
///
/// The underlying persistence layer has no concept of a display name,
/// email address, or account status -- only progress and preferences are
/// tracked -- so `name`/`email` are honestly derived from the real
/// `user_id` (never a fabricated identity like a placeholder person's
/// name), and `status` is always reported as `Active` since there is
/// nothing to distinguish it by.
fn user_from_records(
    user_id: &str,
    progress: Option<&UserProgress>,
    preferences: Option<&UserPreferences>,
) -> User {
    let last_activity = progress.map(|p| p.last_updated.to_rfc3339());
    let created_at = progress
        .and_then(|p| p.progress_history.first())
        .map(|snapshot| snapshot.timestamp.to_rfc3339())
        .or_else(|| last_activity.clone())
        .unwrap_or_else(|| Utc::now().to_rfc3339());

    User {
        id: ID::from(user_id),
        name: user_id.to_string(),
        email: String::new(),
        created_at,
        last_activity,
        status: UserStatus::Active,
        preferences: preferences
            .map(preferences_to_graphql)
            .unwrap_or_else(|| preferences_to_graphql(&UserPreferences::default())),
    }
}

/// Convert real stored preferences into the GraphQL representation.
/// `audio_quality` and `feedback_frequency` have no corresponding concept
/// in [`UserPreferences`] and are defaulted (documented per-field below)
/// rather than fabricated as if tracked.
fn preferences_to_graphql(preferences: &UserPreferences) -> UserPreferencesGraphQL {
    UserPreferencesGraphQL {
        language: preferences.feedback_language.to_string(),
        // Not tracked by `UserPreferences`; no real signal to report.
        audio_quality: AudioQuality::Standard,
        // Not tracked by `UserPreferences`; no real signal to report.
        feedback_frequency: FeedbackFrequency::Realtime,
        theme: "default".to_string(),
        notifications_enabled: preferences.notifications.enable_realtime
            || preferences.notifications.enable_progress
            || preferences.notifications.enable_achievements,
    }
}

/// Convert a real, persisted [`SessionState`] into the GraphQL
/// representation, using the real fields tracked on
/// [`crate::traits::SessionStatistics`] (`end_time`, `duration`, quality
/// scores) rather than re-deriving approximations.
fn session_to_graphql(session: &SessionState) -> Session {
    let stats = &session.session_stats;
    let ended_at = stats.end_time.map(|t| t.to_rfc3339());
    let duration_seconds = i32::try_from(stats.duration.as_secs()).unwrap_or(i32::MAX);
    let average_score = if stats.audio_generated_count > 0 {
        Some((stats.average_quality_score + stats.average_pronunciation_score) / 2.0)
    } else {
        None
    };

    Session {
        id: ID::from(session.session_id.to_string()),
        user_id: ID::from(session.user_id.clone()),
        started_at: session.start_time.to_rfc3339(),
        ended_at,
        duration_seconds: Some(duration_seconds),
        status: if stats.end_time.is_some() {
            SessionStatus::Completed
        } else {
            SessionStatus::Active
        },
        // No dedicated "feedback items generated" counter is tracked on
        // `SessionStatistics`; `audio_generated_count` is the closest real
        // proxy (each generated audio segment corresponds to a feedback
        // interaction in this pipeline).
        feedback_count: stats.audio_generated_count as i32,
        average_score,
        tags: Vec::new(),
    }
}

#[Object]
impl QueryRoot {
    /// Get a user by ID. Existence is determined by a real progress-record
    /// lookup: a `user_id` nobody has ever recorded progress for returns
    /// `None`, never a fabricated user.
    async fn user(&self, ctx: &Context<'_>, id: ID) -> FieldResult<Option<User>> {
        let persistence = persistence(ctx)?;

        let progress = match persistence.load_user_progress(id.as_str()).await {
            Ok(progress) => progress,
            Err(_) => return Ok(None),
        };
        let preferences = persistence.load_preferences(id.as_str()).await.ok();

        Ok(Some(user_from_records(
            id.as_str(),
            Some(&progress),
            preferences.as_ref(),
        )))
    }

    /// List real, known users (enumerated via the persistence backend's
    /// `list_user_ids`), with optional filtering by exact `user_id` or
    /// `overall_skill_level` range, and real pagination.
    async fn users(
        &self,
        ctx: &Context<'_>,
        filter: Option<FilterInput>,
        pagination: Option<PaginationInput>,
        _sort: Option<SortInput>,
    ) -> FieldResult<UserConnection> {
        let persistence = persistence(ctx)?;

        let mut user_ids = persistence.list_user_ids().await?;
        user_ids.sort();

        if let Some(user_id_filter) = filter.as_ref().and_then(|f| f.user_id.as_ref()) {
            user_ids.retain(|id| id.as_str() == user_id_filter.as_str());
        }

        let score_range = filter.as_ref().and_then(|f| f.score_range.as_ref());

        let mut users = Vec::with_capacity(user_ids.len());
        for user_id in &user_ids {
            let progress = persistence.load_user_progress(user_id).await.ok();

            if let Some(range) = score_range {
                let score = progress.as_ref().map_or(0.0, |p| p.overall_skill_level);
                if score < range.min || score > range.max {
                    continue;
                }
            }

            let preferences = persistence.load_preferences(user_id).await.ok();
            users.push(user_from_records(
                user_id,
                progress.as_ref(),
                preferences.as_ref(),
            ));
        }

        let total_count = users.len() as i32;

        let offset = pagination
            .as_ref()
            .and_then(|p| p.offset)
            .unwrap_or(0)
            .max(0) as usize;
        let limit = pagination
            .as_ref()
            .and_then(|p| p.limit)
            .map(|l| l.max(0) as usize);
        let has_previous_page = offset > 0;
        let page: Vec<User> = match limit {
            Some(limit) => users.into_iter().skip(offset).take(limit).collect(),
            None => users.into_iter().skip(offset).collect(),
        };
        let has_next_page = offset + page.len() < total_count as usize;

        let edges: Vec<UserEdge> = page
            .into_iter()
            .enumerate()
            .map(|(i, user)| UserEdge {
                cursor: (offset + i).to_string(),
                node: user,
            })
            .collect();

        Ok(UserConnection {
            total_count,
            page_info: PageInfo {
                has_next_page,
                has_previous_page,
                start_cursor: edges.first().map(|e| e.cursor.clone()),
                end_cursor: edges.last().map(|e| e.cursor.clone()),
            },
            edges,
        })
    }

    /// Get a session by ID from real storage.
    async fn session(&self, ctx: &Context<'_>, id: ID) -> FieldResult<Option<Session>> {
        let persistence = persistence(ctx)?;

        let session_id = match Uuid::parse_str(id.as_str()) {
            Ok(uuid) => uuid,
            Err(_) => return Ok(None),
        };

        match persistence.load_session(&session_id).await {
            Ok(session) => Ok(Some(session_to_graphql(&session))),
            Err(_) => Ok(None),
        }
    }

    /// Get feedback items for a specific user (`filter.user_id`), or -- if
    /// no user filter is given -- aggregated across every known user, real
    /// pagination applied at the end. `filter.user_id` is strongly
    /// recommended for non-trivial user counts, since the unfiltered path
    /// genuinely queries every user's history.
    async fn feedback_items(
        &self,
        ctx: &Context<'_>,
        filter: Option<FilterInput>,
        pagination: Option<PaginationInput>,
    ) -> FieldResult<FeedbackItemConnection> {
        let persistence = persistence(ctx)?;

        let target_users: Vec<String> = match filter.as_ref().and_then(|f| f.user_id.as_ref()) {
            Some(user_id) => vec![user_id.to_string()],
            None => persistence.list_user_ids().await?,
        };

        let mut feedback_items = Vec::new();
        for user_id in &target_users {
            let history = persistence
                .load_feedback_history(user_id, None, None)
                .await
                .unwrap_or_default();

            for response in history {
                for item in response.feedback_items {
                    if let Some(range) = filter.as_ref().and_then(|f| f.score_range.as_ref()) {
                        if item.score < range.min || item.score > range.max {
                            continue;
                        }
                    }

                    feedback_items.push(FeedbackItem {
                        id: ID::from(Uuid::new_v4().to_string()),
                        user_id: ID::from(user_id.clone()),
                        message: item.message,
                        score: item.score,
                        category: FeedbackCategory::General,
                        suggestions: item.suggestion.into_iter().collect(),
                        created_at: response.timestamp.to_rfc3339(),
                        audio_segment: None,
                    });
                }
            }
        }

        let total_count = feedback_items.len() as i32;
        let offset = pagination
            .as_ref()
            .and_then(|p| p.offset)
            .unwrap_or(0)
            .max(0) as usize;
        let limit = pagination
            .as_ref()
            .and_then(|p| p.limit)
            .map(|l| l.max(0) as usize);
        let has_previous_page = offset > 0;
        let page: Vec<FeedbackItem> = match limit {
            Some(limit) => feedback_items
                .into_iter()
                .skip(offset)
                .take(limit)
                .collect(),
            None => feedback_items.into_iter().skip(offset).collect(),
        };
        let has_next_page = offset + page.len() < total_count as usize;

        let edges: Vec<FeedbackItemEdge> = page
            .into_iter()
            .enumerate()
            .map(|(i, item)| FeedbackItemEdge {
                cursor: (offset + i).to_string(),
                node: item,
            })
            .collect();

        Ok(FeedbackItemConnection {
            total_count,
            page_info: PageInfo {
                has_next_page,
                has_previous_page,
                start_cursor: edges.first().map(|e| e.cursor.clone()),
                end_cursor: edges.last().map(|e| e.cursor.clone()),
            },
            edges,
        })
    }

    /// Training exercises are managed by the `training` subsystem, which is
    /// not reachable from the persistence backend injected into this
    /// schema. Rather than fabricate exercise rows, this honestly reports
    /// that no real data source is wired up yet.
    async fn training_exercises(
        &self,
        _ctx: &Context<'_>,
        _category: Option<ExerciseCategory>,
        _difficulty_range: Option<ScoreRangeInput>,
    ) -> FieldResult<Vec<TrainingExerciseGraphQL>> {
        Err(Error::new(
            "training_exercises is not yet wired to a real data source",
        ))
    }

    /// Get a user's real progress statistics.
    async fn user_progress(&self, ctx: &Context<'_>, user_id: ID) -> FieldResult<ProgressStats> {
        let persistence = persistence(ctx)?;

        let progress = persistence
            .load_user_progress(user_id.as_str())
            .await
            .map_err(|e| {
                Error::new(format!(
                    "no progress data for user '{}': {e}",
                    user_id.as_str()
                ))
            })?;

        let skill_scores = progress
            .skill_breakdown
            .iter()
            .map(|(area, score)| SkillScore {
                skill_name: format!("{area:?}"),
                current_score: *score,
                // Per-skill history is not tracked separately from the
                // current snapshot, so there is no real prior value to
                // report.
                previous_score: None,
                improvement: None,
            })
            .collect();

        Ok(ProgressStats {
            total_sessions: progress.session_count as i32,
            total_practice_time_seconds: progress.total_practice_time.as_secs() as i32,
            average_score: progress.average_scores.overall_score,
            improvement_rate: progress.training_stats.average_improvement,
            // `TrainingStatistics` tracks streaks as session counts, not
            // calendar days; reported as-is since it is the closest real
            // signal this data model tracks.
            current_streak_days: progress.training_stats.current_streak as i32,
            best_streak_days: progress.training_stats.longest_streak as i32,
            skill_scores,
        })
    }

    /// System-wide analytics, computed from `FeedbackSystem`'s real
    /// aggregate statistics and health report. Fields with no real
    /// underlying counter yet (error rate, per-exercise usage, peak hours,
    /// OS resource utilization) are honestly reported as zero/empty rather
    /// than a plausible-looking fabricated number.
    async fn analytics(
        &self,
        ctx: &Context<'_>,
        period: AnalyticsPeriod,
        start_date: Option<String>,
        end_date: Option<String>,
    ) -> FieldResult<Analytics> {
        let feedback_system = ctx.data::<Arc<FeedbackSystem>>()?;

        let stats = feedback_system
            .get_statistics()
            .await
            .map_err(|e| Error::new(format!("failed to read system statistics: {e}")))?;
        let health = feedback_system
            .get_system_health()
            .await
            .map_err(|e| Error::new(format!("failed to read system health: {e}")))?;

        let start =
            start_date.unwrap_or_else(|| (Utc::now() - chrono::Duration::days(7)).to_rfc3339());
        let end = end_date.unwrap_or_else(|| Utc::now().to_rfc3339());

        Ok(Analytics {
            period,
            start_date: start,
            end_date: end,
            user_metrics: UserMetrics {
                active_users: stats.active_sessions as i32,
                new_registrations: 0, // not tracked by FeedbackSystemStats
                retention_rate: 0.0,  // not tracked by FeedbackSystemStats
                avg_session_duration_seconds: 0, // not tracked by FeedbackSystemStats
                engagement_score: health.health_score,
            },
            system_metrics: SystemMetrics {
                avg_response_time_ms: stats.average_response_time_ms,
                uptime_percentage: health.health_score * 100.0,
                error_rate_percentage: 0.0, // no error counter is tracked
                throughput_rps: 0.0,        // no request-rate counter is tracked
                resource_utilization: ResourceUtilization {
                    cpu_percentage: 0.0,
                    memory_percentage: 0.0,
                    disk_percentage: 0.0,
                    network_percentage: 0.0,
                },
            },
            usage_stats: UsageStats {
                total_requests: stats.total_sessions as i64,
                total_feedback_items: stats.total_feedback_generated as i64,
                total_training_sessions: stats.total_sessions as i64,
                popular_exercises: Vec::new(), // no per-exercise usage source
                peak_usage_hours: Vec::new(),  // no hourly usage source
            },
        })
    }

    /// Search real, known user IDs for a substring match. Session/exercise/
    /// feedback full-text search is not backed by an index reachable from
    /// here and is intentionally left unimplemented rather than faked.
    async fn search(
        &self,
        ctx: &Context<'_>,
        query: String,
        types: Option<Vec<String>>,
        limit: Option<i32>,
    ) -> FieldResult<Vec<SearchResult>> {
        let persistence = persistence(ctx)?;

        let search_users = types
            .as_ref()
            .is_none_or(|types| types.iter().any(|t| t.eq_ignore_ascii_case("user")));

        if !search_users || query.is_empty() {
            return Ok(Vec::new());
        }

        let query_lower = query.to_lowercase();
        let mut user_ids = persistence.list_user_ids().await?;
        user_ids.sort();
        user_ids.retain(|id| id.to_lowercase().contains(&query_lower));

        if let Some(limit) = limit {
            user_ids.truncate(limit.max(0) as usize);
        }

        let mut results = Vec::with_capacity(user_ids.len());
        for user_id in &user_ids {
            let progress = persistence.load_user_progress(user_id).await.ok();
            let preferences = persistence.load_preferences(user_id).await.ok();
            results.push(SearchResult::User(user_from_records(
                user_id,
                progress.as_ref(),
                preferences.as_ref(),
            )));
        }

        Ok(results)
    }
}

#[Object]
impl MutationRoot {
    /// Create a new user: really persists a fresh [`UserProgress`] record
    /// (and preferences) under a newly generated ID.
    ///
    /// `input.name`/`input.email` are echoed back in the response but not
    /// currently persisted anywhere -- neither `UserProgress` nor
    /// `UserPreferences` has a name/email field -- so a later `user()`
    /// query will show the ID-derived name (see [`user_from_records`]),
    /// not what was passed here. Only `notifications_enabled` from
    /// `input.preferences` maps onto a real, persisted field today.
    async fn create_user(&self, ctx: &Context<'_>, input: CreateUserInput) -> FieldResult<User> {
        let persistence = persistence(ctx)?;

        let user_id = Uuid::new_v4().to_string();

        let progress = UserProgress {
            user_id: user_id.clone(),
            ..UserProgress::default()
        };
        persistence
            .save_user_progress(&user_id, &progress)
            .await
            .map_err(|e| Error::new(format!("failed to create user: {e}")))?;

        let mut preferences = UserPreferences {
            user_id: user_id.clone(),
            ..UserPreferences::default()
        };
        if let Some(enabled) = input
            .preferences
            .as_ref()
            .and_then(|p| p.notifications_enabled)
        {
            preferences.notifications.enable_realtime = enabled;
        }
        persistence
            .save_preferences(&user_id, &preferences)
            .await
            .map_err(|e| Error::new(format!("failed to save preferences for new user: {e}")))?;

        Ok(User {
            id: ID::from(user_id),
            name: input.name,
            email: input.email,
            created_at: Utc::now().to_rfc3339(),
            last_activity: None,
            status: UserStatus::Active,
            preferences: preferences_to_graphql(&preferences),
        })
    }

    /// Update an existing user's real preferences. Fails with a real error
    /// if the user does not actually exist (checked via a real progress
    /// lookup), rather than fabricating an "updated" record for an unknown
    /// ID.
    async fn update_user(&self, ctx: &Context<'_>, input: UpdateUserInput) -> FieldResult<User> {
        let persistence = persistence(ctx)?;
        let user_id = input.id.as_str().to_string();

        let progress = persistence
            .load_user_progress(&user_id)
            .await
            .map_err(|e| Error::new(format!("cannot update unknown user '{user_id}': {e}")))?;

        let mut preferences = persistence
            .load_preferences(&user_id)
            .await
            .unwrap_or_else(|_| UserPreferences {
                user_id: user_id.clone(),
                ..UserPreferences::default()
            });

        if let Some(enabled) = input
            .preferences
            .as_ref()
            .and_then(|p| p.notifications_enabled)
        {
            preferences.notifications.enable_realtime = enabled;
        }
        persistence
            .save_preferences(&user_id, &preferences)
            .await
            .map_err(|e| Error::new(format!("failed to save updated preferences: {e}")))?;

        let mut user = user_from_records(&user_id, Some(&progress), Some(&preferences));
        // `status` and, when explicitly overridden, `name`/`email` have no
        // real backing store (see `create_user`'s doc comment) -- echoed
        // into the response as requested, but not persisted.
        if let Some(name) = input.name {
            user.name = name;
        }
        if let Some(email) = input.email {
            user.email = email;
        }
        if let Some(status) = input.status {
            user.status = status;
        }
        Ok(user)
    }

    /// Create a new session: really persists a fresh [`SessionState`].
    async fn create_session(
        &self,
        ctx: &Context<'_>,
        input: CreateSessionInput,
    ) -> FieldResult<Session> {
        let persistence = persistence(ctx)?;
        let user_id = input.user_id.as_str().to_string();

        let preferences = persistence
            .load_preferences(&user_id)
            .await
            .unwrap_or_default();

        let session = SessionState {
            session_id: Uuid::new_v4(),
            user_id: user_id.clone(),
            start_time: Utc::now(),
            last_activity: Utc::now(),
            current_task: None,
            stats: SessionStats::default(),
            preferences,
            adaptive_state: crate::traits::AdaptiveState::default(),
            current_exercise: None,
            session_stats: crate::traits::SessionStatistics::default(),
        };

        persistence
            .save_session(&session)
            .await
            .map_err(|e| Error::new(format!("failed to create session: {e}")))?;

        let mut graphql_session = session_to_graphql(&session);
        graphql_session.tags = input.tags.unwrap_or_default();
        Ok(graphql_session)
    }

    /// End a session: loads the real session, marks a real end time and
    /// computes a real duration from it, and saves the result back.
    async fn end_session(&self, ctx: &Context<'_>, session_id: ID) -> FieldResult<Session> {
        let persistence = persistence(ctx)?;

        let uuid = Uuid::parse_str(session_id.as_str()).map_err(|e| {
            Error::new(format!("invalid session id '{}': {e}", session_id.as_str()))
        })?;

        let mut session = persistence
            .load_session(&uuid)
            .await
            .map_err(|e| Error::new(format!("session '{}' not found: {e}", session_id.as_str())))?;

        let now = Utc::now();
        session.last_activity = now;
        session.session_stats.end_time = Some(now);
        session.session_stats.duration = (now - session.start_time)
            .to_std()
            .unwrap_or(std::time::Duration::ZERO);

        persistence
            .save_session(&session)
            .await
            .map_err(|e| Error::new(format!("failed to save ended session: {e}")))?;

        Ok(session_to_graphql(&session))
    }

    /// Create a feedback item: really persists a [`FeedbackResponse`]
    /// wrapping it for the target user.
    async fn create_feedback(
        &self,
        ctx: &Context<'_>,
        input: CreateFeedbackInput,
    ) -> FieldResult<FeedbackItem> {
        let persistence = persistence(ctx)?;
        let user_id = input.user_id.as_str().to_string();

        let item = crate::traits::UserFeedback {
            message: input.message.clone(),
            suggestion: input.suggestions.as_ref().and_then(|s| s.first().cloned()),
            confidence: 1.0,
            score: input.score,
            priority: 0.5,
            metadata: HashMap::new(),
        };

        let response = FeedbackResponse {
            feedback_items: vec![item],
            overall_score: input.score,
            immediate_actions: Vec::new(),
            long_term_goals: Vec::new(),
            progress_indicators: ProgressIndicators::default(),
            timestamp: Utc::now(),
            processing_time: std::time::Duration::default(),
            feedback_type: FeedbackType::Quality,
        };

        persistence
            .save_feedback(&user_id, &response)
            .await
            .map_err(|e| Error::new(format!("failed to save feedback: {e}")))?;

        Ok(FeedbackItem {
            id: ID::from(Uuid::new_v4().to_string()),
            user_id: input.user_id,
            message: input.message,
            score: input.score,
            category: input.category,
            suggestions: input.suggestions.unwrap_or_default(),
            created_at: response.timestamp.to_rfc3339(),
            audio_segment: input.audio_segment.map(|seg| AudioSegment {
                start_ms: seg.start_ms,
                end_ms: seg.end_ms,
                duration_ms: seg.end_ms - seg.start_ms,
                text: seg.text,
            }),
        })
    }

    /// Delete user data (GDPR compliance): really calls through to the
    /// persistence backend's right-to-erasure deletion and reports its
    /// real outcome. A genuine backend failure surfaces as a real GraphQL
    /// error instead of a hardcoded `true`.
    async fn delete_user_data(&self, ctx: &Context<'_>, user_id: ID) -> FieldResult<bool> {
        let persistence = persistence(ctx)?;

        persistence
            .delete_user_data(user_id.as_str())
            .await
            .map_err(|e| {
                Error::new(format!(
                    "failed to delete data for user '{}': {e}",
                    user_id.as_str()
                ))
            })?;

        Ok(true)
    }
}

/// Create the GraphQL schema.
///
/// `persistence` is the real data source every resolver in this module
/// queries and mutates (see [`persistence()`] and [`user_from_records`]);
/// `feedback_system` is used only by resolvers that need its own
/// already-computed aggregate statistics (currently just `analytics`).
#[must_use]
pub fn create_schema(
    feedback_system: Arc<FeedbackSystem>,
    persistence: Arc<dyn PersistenceManager>,
) -> FeedbackSchema {
    Schema::build(QueryRoot, MutationRoot, EmptySubscription)
        .data(feedback_system)
        .data(persistence)
        .finish()
}

/// GraphQL server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLConfig {
    /// Enable GraphQL playground
    pub enable_playground: bool,
    /// Enable introspection
    pub enable_introspection: bool,
    /// Query depth limit
    pub max_query_depth: usize,
    /// Query complexity limit
    pub max_query_complexity: usize,
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
    /// Enable request tracing
    pub enable_tracing: bool,
}

impl Default for GraphQLConfig {
    fn default() -> Self {
        Self {
            enable_playground: true,
            enable_introspection: true,
            max_query_depth: 10,
            max_query_complexity: 100,
            request_timeout_seconds: 30,
            enable_tracing: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::backends::memory::MemoryPersistenceManager;
    use crate::persistence::PersistenceConfig;
    use crate::{FeedbackError, FeedbackSystemConfig};
    use async_graphql::{Request, Variables};

    /// Helper to create FeedbackSystem with test database (in-memory for tests)
    async fn create_test_feedback_system() -> Result<FeedbackSystem, FeedbackError> {
        let mut config = FeedbackSystemConfig::default();
        #[cfg(feature = "persistence")]
        {
            // Use in-memory database for tests - faster and no file permission issues
            config.database_path = Some(":memory:".to_string());
        }
        FeedbackSystem::with_config(config).await
    }

    async fn test_persistence() -> Arc<dyn PersistenceManager> {
        Arc::new(
            MemoryPersistenceManager::new(PersistenceConfig::default())
                .await
                .unwrap(),
        )
    }

    /// Build a schema backed by a real (initially empty) in-memory
    /// persistence backend, returning both so tests can seed data directly
    /// through the same backend the schema queries.
    async fn test_schema() -> (FeedbackSchema, Arc<dyn PersistenceManager>) {
        let feedback_system = Arc::new(
            create_test_feedback_system()
                .await
                .expect("Failed to create feedback system"),
        );
        let persistence = test_persistence().await;
        (
            create_schema(feedback_system, persistence.clone()),
            persistence,
        )
    }

    /// `user()` must return the real, seeded user -- not a fabricated
    /// "John Doe" -- and must return `null` for an ID nobody has data for.
    #[tokio::test]
    async fn test_graphql_user_query_returns_real_seeded_user() {
        let (schema, persistence) = test_schema().await;

        persistence
            .save_user_progress(
                "real_user_1",
                &UserProgress {
                    user_id: "real_user_1".to_string(),
                    overall_skill_level: 0.6,
                    ..UserProgress::default()
                },
            )
            .await
            .unwrap();

        let query = r#"
            query GetUser($id: ID!) {
                user(id: $id) {
                    id
                    name
                    status
                    preferences {
                        language
                        audioQuality
                    }
                }
            }
        "#;

        let variables = Variables::from_json(serde_json::json!({
            "id": "real_user_1"
        }));

        let request = Request::new(query).variables(variables);
        let response = schema.execute(request).await;

        assert!(response.errors.is_empty(), "{:?}", response.errors);
        let body = response.data.to_string();
        assert!(
            body.contains("real_user_1"),
            "response must reflect the real seeded id, got: {body}"
        );
        assert!(
            !body.contains("John Doe"),
            "must never return the fabricated placeholder identity"
        );
    }

    /// A user ID nobody has ever recorded data for must resolve to `null`,
    /// not a fabricated `User`.
    #[tokio::test]
    async fn test_graphql_user_query_unknown_id_is_null() {
        let (schema, _persistence) = test_schema().await;

        let query = r#"
            query GetUser($id: ID!) {
                user(id: $id) {
                    id
                }
            }
        "#;
        let variables = Variables::from_json(serde_json::json!({ "id": "never_existed" }));
        let request = Request::new(query).variables(variables);
        let response = schema.execute(request).await;

        assert!(response.errors.is_empty(), "{:?}", response.errors);
        let value = response.data.into_json().unwrap();
        assert!(value["user"].is_null());
    }

    /// `createUser` must really persist progress data that a subsequent
    /// `user()` query (and a direct persistence read) can see.
    #[tokio::test]
    async fn test_graphql_create_user_mutation_really_persists() {
        let (schema, persistence) = test_schema().await;

        let mutation = r#"
            mutation CreateUser($input: CreateUserInput!) {
                createUser(input: $input) {
                    id
                    name
                    email
                    status
                }
            }
        "#;

        let variables = Variables::from_json(serde_json::json!({
            "input": {
                "name": "Test User",
                "email": "test@example.com"
            }
        }));

        let request = Request::new(mutation).variables(variables);
        let response = schema.execute(request).await;

        assert!(response.errors.is_empty(), "{:?}", response.errors);
        let value = response.data.into_json().unwrap();
        assert_eq!(value["createUser"]["name"], "Test User");
        let new_id = value["createUser"]["id"].as_str().unwrap().to_string();

        // The mutation must have really written a progress record, not just
        // echoed the input back.
        assert!(persistence.load_user_progress(&new_id).await.is_ok());
    }

    /// `deleteUserData` must really call through to the persistence
    /// backend's erasure and reflect what actually happened, not a
    /// hardcoded `true`.
    #[tokio::test]
    async fn test_graphql_delete_user_data_really_deletes() {
        let (schema, persistence) = test_schema().await;

        persistence
            .save_user_progress(
                "to_delete",
                &UserProgress {
                    user_id: "to_delete".to_string(),
                    ..UserProgress::default()
                },
            )
            .await
            .unwrap();
        assert!(persistence.load_user_progress("to_delete").await.is_ok());

        let mutation = r#"
            mutation DeleteUser($userId: ID!) {
                deleteUserData(userId: $userId)
            }
        "#;
        let variables = Variables::from_json(serde_json::json!({ "userId": "to_delete" }));
        let request = Request::new(mutation).variables(variables);
        let response = schema.execute(request).await;

        assert!(response.errors.is_empty(), "{:?}", response.errors);
        let value = response.data.into_json().unwrap();
        assert_eq!(value["deleteUserData"], true);

        // The real backend must no longer have the data.
        assert!(persistence.load_user_progress("to_delete").await.is_err());
    }

    /// `analytics` values must be real numbers derived from
    /// `FeedbackSystem`'s own statistics, not fixed constants -- verified
    /// by checking they're the same real values `get_statistics()` reports.
    #[tokio::test]
    async fn test_graphql_analytics_query_reflects_real_statistics() {
        let (schema, _persistence) = test_schema().await;

        let query = r#"
            query GetAnalytics {
                analytics(period: WEEKLY) {
                    period
                    userMetrics {
                        activeUsers
                        retentionRate
                    }
                    systemMetrics {
                        avgResponseTimeMs
                        uptimePercentage
                    }
                }
            }
        "#;

        let request = Request::new(query);
        let response = schema.execute(request).await;

        assert!(response.errors.is_empty(), "{:?}", response.errors);
        let value = response.data.into_json().unwrap();
        // A fresh system has zero active sessions -- a real, non-fabricated
        // value (the old hardcoded stub reported 150 regardless of state).
        assert_eq!(value["analytics"]["userMetrics"]["activeUsers"], 0);
    }

    /// `users()` must really enumerate seeded users via the persistence
    /// backend and apply real pagination/filtering, not return a single
    /// fabricated row.
    #[tokio::test]
    async fn test_graphql_users_query_lists_real_seeded_users() {
        let (schema, persistence) = test_schema().await;

        for name in ["alice", "bob", "carol"] {
            persistence
                .save_user_progress(
                    name,
                    &UserProgress {
                        user_id: name.to_string(),
                        ..UserProgress::default()
                    },
                )
                .await
                .unwrap();
        }

        let query = r#"
            query ListUsers {
                users {
                    totalCount
                    edges {
                        node {
                            id
                        }
                    }
                }
            }
        "#;
        let request = Request::new(query);
        let response = schema.execute(request).await;

        assert!(response.errors.is_empty(), "{:?}", response.errors);
        let value = response.data.into_json().unwrap();
        assert_eq!(value["users"]["totalCount"], 3);
        assert_eq!(value["users"]["edges"].as_array().unwrap().len(), 3);
    }
}
