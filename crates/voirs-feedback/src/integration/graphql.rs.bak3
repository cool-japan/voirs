//! GraphQL API implementation for VoiRS feedback system
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

use crate::adaptive::models::UserModel;
use crate::progress::core::ProgressAnalyzer;
use crate::traits::{
    FeedbackResponse, SessionState, TrainingExercise, UserPreferences, UserProgress,
};
use crate::FeedbackSystem;

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

#[Object]
impl QueryRoot {
    /// Get user by ID
    async fn user(&self, ctx: &Context<'_>, id: ID) -> FieldResult<Option<User>> {
        let feedback_system = ctx.data::<Arc<FeedbackSystem>>()?;

        // This would integrate with the actual user management system
        // For now, return a mock user
        Ok(Some(User {
            id: id.clone(),
            name: "John Doe".to_string(),
            email: "john@example.com".to_string(),
            created_at: Utc::now().to_rfc3339(),
            last_activity: Some(Utc::now().to_rfc3339()),
            status: UserStatus::Active,
            preferences: UserPreferencesGraphQL {
                language: "en".to_string(),
                audio_quality: AudioQuality::High,
                feedback_frequency: FeedbackFrequency::Realtime,
                theme: "dark".to_string(),
                notifications_enabled: true,
            },
        }))
    }

    /// Get all users with pagination and filtering
    async fn users(
        &self,
        ctx: &Context<'_>,
        filter: Option<FilterInput>,
        pagination: Option<PaginationInput>,
        sort: Option<SortInput>,
    ) -> FieldResult<UserConnection> {
        let _feedback_system = ctx.data::<Arc<FeedbackSystem>>()?;

        // Mock implementation - would integrate with actual user service
        let users = vec![User {
            id: "1".into(),
            name: "John Doe".to_string(),
            email: "john@example.com".to_string(),
            created_at: Utc::now().to_rfc3339(),
            last_activity: Some(Utc::now().to_rfc3339()),
            status: UserStatus::Active,
            preferences: UserPreferencesGraphQL {
                language: "en".to_string(),
                audio_quality: AudioQuality::High,
                feedback_frequency: FeedbackFrequency::Realtime,
                theme: "dark".to_string(),
                notifications_enabled: true,
            },
        }];

        let edges: Vec<UserEdge> = users
            .into_iter()
            .enumerate()
            .map(|(i, user)| UserEdge {
                cursor: i.to_string(),
                node: user,
            })
            .collect();

        Ok(UserConnection {
            total_count: edges.len() as i32,
            page_info: PageInfo {
                has_next_page: false,
                has_previous_page: false,
                start_cursor: edges.first().map(|e| e.cursor.clone()),
                end_cursor: edges.last().map(|e| e.cursor.clone()),
            },
            edges,
        })
    }

    /// Get session by ID
    async fn session(&self, ctx: &Context<'_>, id: ID) -> FieldResult<Option<Session>> {
        let _feedback_system = ctx.data::<Arc<FeedbackSystem>>()?;

        // Mock implementation
        Ok(Some(Session {
            id: id.clone(),
            user_id: "1".into(),
            started_at: Utc::now().to_rfc3339(),
            ended_at: None,
            duration_seconds: None,
            status: SessionStatus::Active,
            feedback_count: 0,
            average_score: None,
            tags: vec!["practice".to_string()],
        }))
    }

    /// Get feedback items with filtering and pagination
    async fn feedback_items(
        &self,
        ctx: &Context<'_>,
        filter: Option<FilterInput>,
        pagination: Option<PaginationInput>,
    ) -> FieldResult<FeedbackItemConnection> {
        let _feedback_system = ctx.data::<Arc<FeedbackSystem>>()?;

        // Mock implementation
        let feedback_items = vec![FeedbackItem {
            id: "1".into(),
            user_id: "1".into(),
            message: "Good pronunciation!".to_string(),
            score: 0.85,
            category: FeedbackCategory::Pronunciation,
            suggestions: vec!["Try to emphasize the 'th' sound more".to_string()],
            created_at: Utc::now().to_rfc3339(),
            audio_segment: Some(AudioSegment {
                start_ms: 1000,
                end_ms: 2000,
                duration_ms: 1000,
                text: "Hello".to_string(),
            }),
        }];

        let edges: Vec<FeedbackItemEdge> = feedback_items
            .into_iter()
            .enumerate()
            .map(|(i, item)| FeedbackItemEdge {
                cursor: i.to_string(),
                node: item,
            })
            .collect();

        Ok(FeedbackItemConnection {
            total_count: edges.len() as i32,
            page_info: PageInfo {
                has_next_page: false,
                has_previous_page: false,
                start_cursor: edges.first().map(|e| e.cursor.clone()),
                end_cursor: edges.last().map(|e| e.cursor.clone()),
            },
            edges,
        })
    }

    /// Get training exercises
    async fn training_exercises(
        &self,
        ctx: &Context<'_>,
        category: Option<ExerciseCategory>,
        difficulty_range: Option<ScoreRangeInput>,
    ) -> FieldResult<Vec<TrainingExerciseGraphQL>> {
        let _feedback_system = ctx.data::<Arc<FeedbackSystem>>()?;

        // Mock implementation
        Ok(vec![TrainingExerciseGraphQL {
            id: "1".into(),
            name: "Basic Pronunciation".to_string(),
            description: "Practice basic phoneme pronunciation".to_string(),
            difficulty: 0.3,
            category: ExerciseCategory::Pronunciation,
            target_text: "Hello world".to_string(),
            estimated_duration_seconds: 300,
            tags: vec!["beginner".to_string(), "phonemes".to_string()],
            created_at: Utc::now().to_rfc3339(),
        }])
    }

    /// Get user progress statistics
    async fn user_progress(&self, ctx: &Context<'_>, user_id: ID) -> FieldResult<ProgressStats> {
        let _feedback_system = ctx.data::<Arc<FeedbackSystem>>()?;

        // Mock implementation
        Ok(ProgressStats {
            total_sessions: 25,
            total_practice_time_seconds: 7200,
            average_score: 0.78,
            improvement_rate: 12.5,
            current_streak_days: 7,
            best_streak_days: 14,
            skill_scores: vec![
                SkillScore {
                    skill_name: "Pronunciation".to_string(),
                    current_score: 0.82,
                    previous_score: Some(0.75),
                    improvement: Some(0.07),
                },
                SkillScore {
                    skill_name: "Fluency".to_string(),
                    current_score: 0.74,
                    previous_score: Some(0.70),
                    improvement: Some(0.04),
                },
            ],
        })
    }

    /// Get analytics data
    async fn analytics(
        &self,
        ctx: &Context<'_>,
        period: AnalyticsPeriod,
        start_date: Option<String>,
        end_date: Option<String>,
    ) -> FieldResult<Analytics> {
        let _feedback_system = ctx.data::<Arc<FeedbackSystem>>()?;

        let start =
            start_date.unwrap_or_else(|| (Utc::now() - chrono::Duration::days(7)).to_rfc3339());
        let end = end_date.unwrap_or_else(|| Utc::now().to_rfc3339());

        Ok(Analytics {
            period,
            start_date: start,
            end_date: end,
            user_metrics: UserMetrics {
                active_users: 150,
                new_registrations: 12,
                retention_rate: 0.85,
                avg_session_duration_seconds: 1200,
                engagement_score: 0.78,
            },
            system_metrics: SystemMetrics {
                avg_response_time_ms: 45.2,
                uptime_percentage: 99.9,
                error_rate_percentage: 0.1,
                throughput_rps: 125.5,
                resource_utilization: ResourceUtilization {
                    cpu_percentage: 65.2,
                    memory_percentage: 72.8,
                    disk_percentage: 45.1,
                    network_percentage: 38.9,
                },
            },
            usage_stats: UsageStats {
                total_requests: 125_000,
                total_feedback_items: 8_500,
                total_training_sessions: 3_200,
                popular_exercises: vec![ExerciseUsage {
                    exercise_id: "1".into(),
                    exercise_name: "Basic Pronunciation".to_string(),
                    completion_count: 450,
                    avg_completion_time_seconds: 280,
                    avg_score: 0.76,
                }],
                peak_usage_hours: vec![14, 15, 16, 19, 20],
            },
        })
    }

    /// Search across all entities
    async fn search(
        &self,
        ctx: &Context<'_>,
        query: String,
        types: Option<Vec<String>>,
        limit: Option<i32>,
    ) -> FieldResult<Vec<SearchResult>> {
        let _feedback_system = ctx.data::<Arc<FeedbackSystem>>()?;

        // Mock search implementation
        Ok(vec![SearchResult::User(User {
            id: "1".into(),
            name: "John Doe".to_string(),
            email: "john@example.com".to_string(),
            created_at: Utc::now().to_rfc3339(),
            last_activity: Some(Utc::now().to_rfc3339()),
            status: UserStatus::Active,
            preferences: UserPreferencesGraphQL {
                language: "en".to_string(),
                audio_quality: AudioQuality::High,
                feedback_frequency: FeedbackFrequency::Realtime,
                theme: "dark".to_string(),
                notifications_enabled: true,
            },
        })])
    }
}

#[Object]
impl MutationRoot {
    /// Create a new user
    async fn create_user(&self, ctx: &Context<'_>, input: CreateUserInput) -> FieldResult<User> {
        let _feedback_system = ctx.data::<Arc<FeedbackSystem>>()?;

        Ok(User {
            id: Uuid::new_v4().to_string().into(),
            name: input.name,
            email: input.email,
            created_at: Utc::now().to_rfc3339(),
            last_activity: None,
            status: UserStatus::Active,
            preferences: UserPreferencesGraphQL {
                language: "en".to_string(),
                audio_quality: AudioQuality::Standard,
                feedback_frequency: FeedbackFrequency::Realtime,
                theme: "light".to_string(),
                notifications_enabled: true,
            },
        })
    }

    /// Update an existing user
    async fn update_user(&self, ctx: &Context<'_>, input: UpdateUserInput) -> FieldResult<User> {
        let _feedback_system = ctx.data::<Arc<FeedbackSystem>>()?;

        // Mock implementation - would update in database
        Ok(User {
            id: input.id,
            name: input.name.unwrap_or_else(|| "Updated User".to_string()),
            email: input
                .email
                .unwrap_or_else(|| "updated@example.com".to_string()),
            created_at: Utc::now().to_rfc3339(),
            last_activity: Some(Utc::now().to_rfc3339()),
            status: input.status.unwrap_or(UserStatus::Active),
            preferences: UserPreferencesGraphQL {
                language: "en".to_string(),
                audio_quality: AudioQuality::High,
                feedback_frequency: FeedbackFrequency::Realtime,
                theme: "dark".to_string(),
                notifications_enabled: true,
            },
        })
    }

    /// Create a new session
    async fn create_session(
        &self,
        ctx: &Context<'_>,
        input: CreateSessionInput,
    ) -> FieldResult<Session> {
        let _feedback_system = ctx.data::<Arc<FeedbackSystem>>()?;

        Ok(Session {
            id: Uuid::new_v4().to_string().into(),
            user_id: input.user_id,
            started_at: Utc::now().to_rfc3339(),
            ended_at: None,
            duration_seconds: None,
            status: SessionStatus::Active,
            feedback_count: 0,
            average_score: None,
            tags: input.tags.unwrap_or_default(),
        })
    }

    /// End a session
    async fn end_session(&self, ctx: &Context<'_>, session_id: ID) -> FieldResult<Session> {
        let _feedback_system = ctx.data::<Arc<FeedbackSystem>>()?;

        Ok(Session {
            id: session_id,
            user_id: "1".into(),
            started_at: (Utc::now() - chrono::Duration::minutes(30)).to_rfc3339(),
            ended_at: Some(Utc::now().to_rfc3339()),
            duration_seconds: Some(1800),
            status: SessionStatus::Completed,
            feedback_count: 15,
            average_score: Some(0.82),
            tags: vec!["practice".to_string()],
        })
    }

    /// Create feedback item
    async fn create_feedback(
        &self,
        ctx: &Context<'_>,
        input: CreateFeedbackInput,
    ) -> FieldResult<FeedbackItem> {
        let _feedback_system = ctx.data::<Arc<FeedbackSystem>>()?;

        Ok(FeedbackItem {
            id: Uuid::new_v4().to_string().into(),
            user_id: input.user_id,
            message: input.message,
            score: input.score,
            category: input.category,
            suggestions: input.suggestions.unwrap_or_default(),
            created_at: Utc::now().to_rfc3339(),
            audio_segment: input.audio_segment.map(|seg| AudioSegment {
                start_ms: seg.start_ms,
                end_ms: seg.end_ms,
                duration_ms: seg.end_ms - seg.start_ms,
                text: seg.text,
            }),
        })
    }

    /// Delete user data (GDPR compliance)
    async fn delete_user_data(&self, ctx: &Context<'_>, user_id: ID) -> FieldResult<bool> {
        let _feedback_system = ctx.data::<Arc<FeedbackSystem>>()?;

        // Would implement actual data deletion
        Ok(true)
    }
}

/// Create the GraphQL schema
pub fn create_schema(feedback_system: Arc<FeedbackSystem>) -> FeedbackSchema {
    Schema::build(QueryRoot, MutationRoot, EmptySubscription)
        .data(feedback_system)
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
    use async_graphql::{Request, Variables};

    #[tokio::test]
    async fn test_graphql_user_query() {
        let feedback_system = Arc::new(
            FeedbackSystem::new()
                .await
                .expect("Failed to create feedback system"),
        );

        let schema = create_schema(feedback_system);

        let query = r#"
            query GetUser($id: ID!) {
                user(id: $id) {
                    id
                    name
                    email
                    status
                    preferences {
                        language
                        audioQuality
                    }
                }
            }
        "#;

        let variables = Variables::from_json(serde_json::json!({
            "id": "1"
        }));

        let request = Request::new(query).variables(variables);
        let response = schema.execute(request).await;

        assert!(response.errors.is_empty());
        assert!(response.data.to_string().contains("John Doe"));
    }

    #[tokio::test]
    async fn test_graphql_create_user_mutation() {
        let feedback_system = Arc::new(
            FeedbackSystem::new()
                .await
                .expect("Failed to create feedback system"),
        );

        let schema = create_schema(feedback_system);

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

        assert!(response.errors.is_empty());
        assert!(response.data.to_string().contains("Test User"));
    }

    #[tokio::test]
    async fn test_graphql_analytics_query() {
        let feedback_system = Arc::new(
            FeedbackSystem::new()
                .await
                .expect("Failed to create feedback system"),
        );

        let schema = create_schema(feedback_system);

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

        assert!(response.errors.is_empty());
        assert!(response.data.to_string().contains("activeUsers"));
    }
}
