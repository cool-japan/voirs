//! # Comprehensive User Management System
//!
//! This module provides comprehensive user profile management, lifecycle tracking,
//! segmentation, and personalization capabilities for the `VoiRS` feedback system.
//!
//! ## Features
//!
//! - **User Profiles**: Comprehensive user profiles with preferences, history, and metadata
//! - **Lifecycle Management**: Track user journey from onboarding to retention
//! - **Segmentation**: Categorize users by behavior, progress, and engagement patterns
//! - **Engagement Scoring**: Calculate and track user engagement metrics
//! - **Recommendations**: Personalized exercise and content recommendations
//! - **Search & Filter**: Advanced user search and filtering capabilities
//! - **Analytics**: User-centric analytics and reporting
//!
//! ## Example Usage
//!
//! ```no_run
//! use voirs_feedback::user_management::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), UserManagementError> {
//!     // Create user manager
//!     let manager = UserProfileManager::new();
//!
//!     // Create a user profile
//!     let profile = UserProfileBuilder::new("user123")
//!         .name("John Doe")
//!         .email("john@example.com")
//!         .language("en-US")
//!         .timezone("America/New_York")
//!         .build();
//!
//!     manager.create_profile(profile).await?;
//!
//!     // Get user lifecycle stage
//!     let stage = manager.get_lifecycle_stage("user123").await?;
//!     println!("User lifecycle stage: {:?}", stage);
//!
//!     // Get personalized recommendations
//!     let recommendations = manager.get_recommendations("user123", 5).await?;
//!     for rec in recommendations {
//!         println!("Recommended: {}", rec.title);
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::traits::{TrainingExercise, UserProgress};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// User management error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum UserManagementError {
    /// User not found
    #[error("User not found: {user_id}")]
    UserNotFound {
        /// User ID
        user_id: String,
    },

    /// User already exists
    #[error("User already exists: {user_id}")]
    UserAlreadyExists {
        /// User ID
        user_id: String,
    },

    /// Invalid user data
    #[error("Invalid user data: {message}")]
    InvalidUserData {
        /// Error message
        message: String,
    },

    /// Profile update failed
    #[error("Profile update failed: {message}")]
    ProfileUpdateFailed {
        /// Error message
        message: String,
    },

    /// Segmentation error
    #[error("Segmentation error: {message}")]
    SegmentationError {
        /// Error message
        message: String,
    },

    /// Recommendation error
    #[error("Recommendation error: {message}")]
    RecommendationError {
        /// Error message
        message: String,
    },

    /// Database error
    #[error("Database error: {message}")]
    DatabaseError {
        /// Error message
        message: String,
    },
}

/// Result type for user management operations
pub type UserManagementResult<T> = Result<T, UserManagementError>;

/// User lifecycle stages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleStage {
    /// New user (0-7 days)
    New,
    /// Active user (engaged, making progress)
    Active,
    /// Power user (highly engaged, advanced)
    PowerUser,
    /// At-risk user (declining engagement)
    AtRisk,
    /// Dormant user (no activity in 30+ days)
    Dormant,
    /// Churned user (no activity in 90+ days)
    Churned,
    /// Reactivated user (returned after being dormant)
    Reactivated,
}

/// User learning style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningStyle {
    /// Visual learner
    Visual,
    /// Auditory learner
    Auditory,
    /// Kinesthetic learner
    Kinesthetic,
    /// Reading/writing learner
    ReadingWriting,
    /// Mixed/adaptive
    Mixed,
}

/// User skill level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum SkillLevel {
    /// Beginner
    Beginner,
    /// Intermediate
    Intermediate,
    /// Advanced
    Advanced,
    /// Expert
    Expert,
}

/// User segment
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UserSegment {
    /// High engagement, high progress
    HighPerformer,
    /// High engagement, low progress (struggling)
    Struggling,
    /// Low engagement, high progress (natural talent)
    NaturalTalent,
    /// Low engagement, low progress (at risk)
    AtRisk,
    /// Consistent moderate engagement
    Steady,
    /// Irregular engagement pattern
    Irregular,
    /// New user (not enough data)
    New,
}

/// User preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Preferred language code
    pub language: String,
    /// Timezone
    pub timezone: String,
    /// Learning style
    pub learning_style: LearningStyle,
    /// Preferred difficulty level
    pub preferred_difficulty: f32,
    /// Enable notifications
    pub notifications_enabled: bool,
    /// Email notifications
    pub email_notifications: bool,
    /// Push notifications
    pub push_notifications: bool,
    /// Preferred session duration (minutes)
    pub preferred_session_duration: u32,
    /// Daily goal (minutes)
    pub daily_goal_minutes: u32,
    /// Theme preference
    pub theme: String,
    /// Font size multiplier
    pub font_size: f32,
    /// Audio feedback enabled
    pub audio_feedback: bool,
    /// Visual feedback enabled
    pub visual_feedback: bool,
    /// Custom preferences
    pub custom: HashMap<String, String>,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            language: "en-US".to_string(),
            timezone: "UTC".to_string(),
            learning_style: LearningStyle::Mixed,
            preferred_difficulty: 0.5,
            notifications_enabled: true,
            email_notifications: true,
            push_notifications: true,
            preferred_session_duration: 30,
            daily_goal_minutes: 15,
            theme: "default".to_string(),
            font_size: 1.0,
            audio_feedback: true,
            visual_feedback: true,
            custom: HashMap::new(),
        }
    }
}

/// User activity summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserActivitySummary {
    /// Total sessions
    pub total_sessions: u32,
    /// Total practice time (minutes)
    pub total_practice_time: u64,
    /// Last active date
    pub last_active: Option<DateTime<Utc>>,
    /// Average session duration (minutes)
    pub avg_session_duration: f32,
    /// Sessions in last 7 days
    pub sessions_last_7_days: u32,
    /// Sessions in last 30 days
    pub sessions_last_30_days: u32,
    /// Current streak (days)
    pub current_streak: u32,
    /// Longest streak (days)
    pub longest_streak: u32,
    /// Completion rate (0.0-1.0)
    pub completion_rate: f32,
}

impl Default for UserActivitySummary {
    fn default() -> Self {
        Self {
            total_sessions: 0,
            total_practice_time: 0,
            last_active: None,
            avg_session_duration: 0.0,
            sessions_last_7_days: 0,
            sessions_last_30_days: 0,
            current_streak: 0,
            longest_streak: 0,
            completion_rate: 0.0,
        }
    }
}

/// User achievement summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAchievementSummary {
    /// Total achievements unlocked
    pub total_achievements: u32,
    /// Achievement points
    pub achievement_points: u64,
    /// Achievements by category
    pub achievements_by_category: HashMap<String, u32>,
    /// Recent achievements (last 10)
    pub recent_achievements: Vec<String>,
    /// Rarity score (0.0-1.0, higher is rarer)
    pub rarity_score: f32,
}

impl Default for UserAchievementSummary {
    fn default() -> Self {
        Self {
            total_achievements: 0,
            achievement_points: 0,
            achievements_by_category: HashMap::new(),
            recent_achievements: Vec::new(),
            rarity_score: 0.0,
        }
    }
}

/// Comprehensive user profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// Unique user identifier
    pub user_id: String,
    /// Display name
    pub name: String,
    /// Email address
    pub email: Option<String>,
    /// Avatar URL
    pub avatar_url: Option<String>,
    /// Account creation date
    pub created_at: DateTime<Utc>,
    /// Last updated date
    pub updated_at: DateTime<Utc>,
    /// User preferences
    pub preferences: UserPreferences,
    /// Current skill level
    pub skill_level: SkillLevel,
    /// Lifecycle stage
    pub lifecycle_stage: LifecycleStage,
    /// User segment
    pub segment: UserSegment,
    /// Engagement score (0.0-1.0)
    pub engagement_score: f32,
    /// Activity summary
    pub activity: UserActivitySummary,
    /// Achievement summary
    pub achievements: UserAchievementSummary,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Account status
    pub status: AccountStatus,
}

/// Account status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccountStatus {
    /// Active account
    Active,
    /// Suspended account
    Suspended,
    /// Deleted account
    Deleted,
    /// Pending verification
    PendingVerification,
}

/// User profile builder
pub struct UserProfileBuilder {
    user_id: String,
    name: Option<String>,
    email: Option<String>,
    avatar_url: Option<String>,
    preferences: UserPreferences,
    metadata: HashMap<String, String>,
    tags: Vec<String>,
}

impl UserProfileBuilder {
    /// Create a new user profile builder
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            name: None,
            email: None,
            avatar_url: None,
            preferences: UserPreferences::default(),
            metadata: HashMap::new(),
            tags: Vec::new(),
        }
    }

    /// Set user name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set user email
    pub fn email(mut self, email: impl Into<String>) -> Self {
        self.email = Some(email.into());
        self
    }

    /// Set avatar URL
    pub fn avatar_url(mut self, url: impl Into<String>) -> Self {
        self.avatar_url = Some(url.into());
        self
    }

    /// Set language
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.preferences.language = lang.into();
        self
    }

    /// Set timezone
    pub fn timezone(mut self, tz: impl Into<String>) -> Self {
        self.preferences.timezone = tz.into();
        self
    }

    /// Set learning style
    #[must_use]
    pub fn learning_style(mut self, style: LearningStyle) -> Self {
        self.preferences.learning_style = style;
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add tag
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Build the user profile
    #[must_use]
    pub fn build(self) -> UserProfile {
        let now = Utc::now();
        UserProfile {
            user_id: self.user_id,
            name: self.name.unwrap_or_else(|| "Unknown User".to_string()),
            email: self.email,
            avatar_url: self.avatar_url,
            created_at: now,
            updated_at: now,
            preferences: self.preferences,
            skill_level: SkillLevel::Beginner,
            lifecycle_stage: LifecycleStage::New,
            segment: UserSegment::New,
            engagement_score: 0.0,
            activity: UserActivitySummary::default(),
            achievements: UserAchievementSummary::default(),
            metadata: self.metadata,
            tags: self.tags,
            status: AccountStatus::Active,
        }
    }
}

/// User search criteria
#[derive(Debug, Clone, Default)]
pub struct UserSearchCriteria {
    /// Filter by lifecycle stage
    pub lifecycle_stage: Option<LifecycleStage>,
    /// Filter by skill level
    pub skill_level: Option<SkillLevel>,
    /// Filter by segment
    pub segment: Option<UserSegment>,
    /// Minimum engagement score
    pub min_engagement: Option<f32>,
    /// Maximum engagement score
    pub max_engagement: Option<f32>,
    /// Filter by tags
    pub tags: Vec<String>,
    /// Filter by account status
    pub status: Option<AccountStatus>,
    /// Created after date
    pub created_after: Option<DateTime<Utc>>,
    /// Created before date
    pub created_before: Option<DateTime<Utc>>,
    /// Last active after date
    pub last_active_after: Option<DateTime<Utc>>,
    /// Search query (name, email)
    pub query: Option<String>,
}

/// Content recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentRecommendation {
    /// Content ID
    pub content_id: String,
    /// Content type
    pub content_type: RecommendationType,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Relevance score (0.0-1.0)
    pub relevance_score: f32,
    /// Difficulty level
    pub difficulty: f32,
    /// Estimated duration (minutes)
    pub estimated_duration: u32,
    /// Recommendation reason
    pub reason: String,
}

/// Recommendation type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Training exercise
    Exercise,
    /// Learning module
    Module,
    /// Challenge
    Challenge,
    /// Review session
    Review,
    /// Assessment
    Assessment,
}

/// User profile manager
pub struct UserProfileManager {
    profiles: Arc<RwLock<HashMap<String, UserProfile>>>,
    activity_tracker: Arc<RwLock<HashMap<String, Vec<DateTime<Utc>>>>>,
}

impl UserProfileManager {
    /// Create a new user profile manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: Arc::new(RwLock::new(HashMap::new())),
            activity_tracker: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new user profile
    pub async fn create_profile(&self, profile: UserProfile) -> UserManagementResult<()> {
        let mut profiles = self.profiles.write().await;

        if profiles.contains_key(&profile.user_id) {
            return Err(UserManagementError::UserAlreadyExists {
                user_id: profile.user_id.clone(),
            });
        }

        profiles.insert(profile.user_id.clone(), profile);
        Ok(())
    }

    /// Get user profile by ID
    pub async fn get_profile(&self, user_id: &str) -> UserManagementResult<UserProfile> {
        let profiles = self.profiles.read().await;

        profiles
            .get(user_id)
            .cloned()
            .ok_or_else(|| UserManagementError::UserNotFound {
                user_id: user_id.to_string(),
            })
    }

    /// Update user profile
    pub async fn update_profile(
        &self,
        user_id: &str,
        update_fn: impl FnOnce(&mut UserProfile),
    ) -> UserManagementResult<()> {
        let mut profiles = self.profiles.write().await;

        let profile =
            profiles
                .get_mut(user_id)
                .ok_or_else(|| UserManagementError::UserNotFound {
                    user_id: user_id.to_string(),
                })?;

        update_fn(profile);
        profile.updated_at = Utc::now();

        Ok(())
    }

    /// Delete user profile
    pub async fn delete_profile(&self, user_id: &str) -> UserManagementResult<()> {
        let mut profiles = self.profiles.write().await;

        profiles
            .remove(user_id)
            .ok_or_else(|| UserManagementError::UserNotFound {
                user_id: user_id.to_string(),
            })?;

        Ok(())
    }

    /// Get user lifecycle stage
    pub async fn get_lifecycle_stage(&self, user_id: &str) -> UserManagementResult<LifecycleStage> {
        let profile = self.get_profile(user_id).await?;
        Ok(self.calculate_lifecycle_stage(&profile))
    }

    /// Calculate lifecycle stage based on user activity
    fn calculate_lifecycle_stage(&self, profile: &UserProfile) -> LifecycleStage {
        let now = Utc::now();

        // Check account age
        let account_age_days = (now - profile.created_at).num_days();

        if account_age_days <= 7 {
            return LifecycleStage::New;
        }

        // Check last activity
        if let Some(last_active) = profile.activity.last_active {
            let days_since_active = (now - last_active).num_days();

            if days_since_active >= 90 {
                return LifecycleStage::Churned;
            } else if days_since_active >= 30 {
                return LifecycleStage::Dormant;
            }
        }

        // Check engagement patterns
        if profile.engagement_score >= 0.8 && profile.activity.sessions_last_7_days >= 5 {
            return LifecycleStage::PowerUser;
        }

        if profile.engagement_score <= 0.3 || profile.activity.sessions_last_30_days < 2 {
            return LifecycleStage::AtRisk;
        }

        // Check for reactivation
        if profile.lifecycle_stage == LifecycleStage::Dormant
            && profile.activity.sessions_last_7_days > 0
        {
            return LifecycleStage::Reactivated;
        }

        LifecycleStage::Active
    }

    /// Update user engagement score
    pub async fn update_engagement_score(&self, user_id: &str) -> UserManagementResult<f32> {
        let mut profiles = self.profiles.write().await;

        let profile =
            profiles
                .get_mut(user_id)
                .ok_or_else(|| UserManagementError::UserNotFound {
                    user_id: user_id.to_string(),
                })?;

        let score = self.calculate_engagement_score(profile);
        profile.engagement_score = score;
        profile.updated_at = Utc::now();

        Ok(score)
    }

    /// Calculate engagement score (0.0-1.0)
    fn calculate_engagement_score(&self, profile: &UserProfile) -> f32 {
        let mut score = 0.0;

        // Activity frequency (40%)
        let frequency_score = if profile.activity.sessions_last_7_days >= 7 {
            1.0
        } else if profile.activity.sessions_last_7_days >= 3 {
            0.7
        } else if profile.activity.sessions_last_7_days >= 1 {
            0.4
        } else {
            0.0
        };
        score += frequency_score * 0.4;

        // Streak (20%)
        let streak_score = (profile.activity.current_streak as f32 / 30.0).min(1.0);
        score += streak_score * 0.2;

        // Completion rate (20%)
        score += profile.activity.completion_rate * 0.2;

        // Session duration (10%)
        let duration_score = (profile.activity.avg_session_duration / 60.0).min(1.0);
        score += duration_score * 0.1;

        // Achievements (10%)
        let achievement_score = (profile.achievements.total_achievements as f32 / 50.0).min(1.0);
        score += achievement_score * 0.1;

        score.clamp(0.0, 1.0)
    }

    /// Segment users
    pub async fn segment_user(&self, user_id: &str) -> UserManagementResult<UserSegment> {
        let profile = self.get_profile(user_id).await?;
        Ok(self.calculate_segment(&profile))
    }

    /// Calculate user segment
    fn calculate_segment(&self, profile: &UserProfile) -> UserSegment {
        if profile.lifecycle_stage == LifecycleStage::New {
            return UserSegment::New;
        }

        let high_engagement = profile.engagement_score >= 0.6;
        let high_progress = profile.skill_level >= SkillLevel::Intermediate;

        match (high_engagement, high_progress) {
            (true, true) => UserSegment::HighPerformer,
            (true, false) => UserSegment::Struggling,
            (false, true) => UserSegment::NaturalTalent,
            (false, false) => UserSegment::AtRisk,
        }
    }

    /// Search users
    pub async fn search_users(
        &self,
        criteria: &UserSearchCriteria,
    ) -> UserManagementResult<Vec<UserProfile>> {
        let profiles = self.profiles.read().await;

        let results: Vec<UserProfile> = profiles
            .values()
            .filter(|profile| self.matches_criteria(profile, criteria))
            .cloned()
            .collect();

        Ok(results)
    }

    /// Check if profile matches search criteria
    fn matches_criteria(&self, profile: &UserProfile, criteria: &UserSearchCriteria) -> bool {
        if let Some(stage) = criteria.lifecycle_stage {
            if profile.lifecycle_stage != stage {
                return false;
            }
        }

        if let Some(level) = criteria.skill_level {
            if profile.skill_level != level {
                return false;
            }
        }

        if let Some(segment) = &criteria.segment {
            if &profile.segment != segment {
                return false;
            }
        }

        if let Some(min) = criteria.min_engagement {
            if profile.engagement_score < min {
                return false;
            }
        }

        if let Some(max) = criteria.max_engagement {
            if profile.engagement_score > max {
                return false;
            }
        }

        if !criteria.tags.is_empty() && !criteria.tags.iter().any(|tag| profile.tags.contains(tag))
        {
            return false;
        }

        if let Some(status) = criteria.status {
            if profile.status != status {
                return false;
            }
        }

        if let Some(after) = criteria.created_after {
            if profile.created_at < after {
                return false;
            }
        }

        if let Some(before) = criteria.created_before {
            if profile.created_at > before {
                return false;
            }
        }

        if let Some(query) = &criteria.query {
            let query_lower = query.to_lowercase();
            if !profile.name.to_lowercase().contains(&query_lower)
                && !profile
                    .email
                    .as_ref()
                    .is_some_and(|e| e.to_lowercase().contains(&query_lower))
            {
                return false;
            }
        }

        true
    }

    /// Get personalized recommendations
    pub async fn get_recommendations(
        &self,
        user_id: &str,
        count: usize,
    ) -> UserManagementResult<Vec<ContentRecommendation>> {
        let profile = self.get_profile(user_id).await?;
        Ok(self.generate_recommendations(&profile, count))
    }

    /// Generate personalized recommendations
    fn generate_recommendations(
        &self,
        profile: &UserProfile,
        count: usize,
    ) -> Vec<ContentRecommendation> {
        let mut recommendations = Vec::new();

        // Recommend based on skill level
        let target_difficulty = match profile.skill_level {
            SkillLevel::Beginner => 0.3,
            SkillLevel::Intermediate => 0.5,
            SkillLevel::Advanced => 0.7,
            SkillLevel::Expert => 0.9,
        };

        // Sample recommendations (in production, these would come from content database)
        recommendations.push(ContentRecommendation {
            content_id: "ex_001".to_string(),
            content_type: RecommendationType::Exercise,
            title: "Pronunciation Practice".to_string(),
            description: "Practice basic pronunciation patterns".to_string(),
            relevance_score: 0.9,
            difficulty: target_difficulty,
            estimated_duration: 15,
            reason: format!(
                "Matches your {} level",
                format!("{:?}", profile.skill_level).to_lowercase()
            ),
        });

        if profile.activity.current_streak >= 7 {
            recommendations.push(ContentRecommendation {
                content_id: "challenge_001".to_string(),
                content_type: RecommendationType::Challenge,
                title: "7-Day Streak Challenge".to_string(),
                description: "Complete advanced exercises to maintain your streak".to_string(),
                relevance_score: 0.85,
                difficulty: target_difficulty + 0.1,
                estimated_duration: 20,
                reason: "Celebrate your 7-day streak!".to_string(),
            });
        }

        if profile.engagement_score < 0.4 {
            recommendations.push(ContentRecommendation {
                content_id: "module_beginner".to_string(),
                content_type: RecommendationType::Module,
                title: "Getting Started Module".to_string(),
                description: "Quick introduction to improve engagement".to_string(),
                relevance_score: 0.8,
                difficulty: 0.2,
                estimated_duration: 10,
                reason: "Short and easy to get back on track".to_string(),
            });
        }

        recommendations.truncate(count);
        recommendations
    }

    /// Record user activity
    pub async fn record_activity(&self, user_id: &str) -> UserManagementResult<()> {
        let mut tracker = self.activity_tracker.write().await;
        let activities = tracker.entry(user_id.to_string()).or_insert_with(Vec::new);
        activities.push(Utc::now());

        // Update profile
        self.update_profile(user_id, |profile| {
            profile.activity.last_active = Some(Utc::now());
            profile.activity.total_sessions += 1;
        })
        .await?;

        Ok(())
    }

    /// Get user count
    pub async fn get_user_count(&self) -> usize {
        let profiles = self.profiles.read().await;
        profiles.len()
    }

    /// Get users by lifecycle stage
    pub async fn get_users_by_stage(&self, stage: LifecycleStage) -> Vec<UserProfile> {
        let profiles = self.profiles.read().await;
        profiles
            .values()
            .filter(|p| p.lifecycle_stage == stage)
            .cloned()
            .collect()
    }

    /// Get engagement distribution
    pub async fn get_engagement_distribution(&self) -> HashMap<String, usize> {
        let profiles = self.profiles.read().await;
        let mut distribution = HashMap::new();

        for profile in profiles.values() {
            let bucket = if profile.engagement_score >= 0.8 {
                "high"
            } else if profile.engagement_score >= 0.5 {
                "medium"
            } else if profile.engagement_score >= 0.2 {
                "low"
            } else {
                "very_low"
            };

            *distribution.entry(bucket.to_string()).or_insert(0) += 1;
        }

        distribution
    }
}

impl Default for UserProfileManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_profile() {
        let manager = UserProfileManager::new();

        let profile = UserProfileBuilder::new("user123")
            .name("John Doe")
            .email("john@example.com")
            .build();

        assert!(manager.create_profile(profile).await.is_ok());
        assert_eq!(manager.get_user_count().await, 1);
    }

    #[tokio::test]
    async fn test_get_profile() {
        let manager = UserProfileManager::new();

        let profile = UserProfileBuilder::new("user123").name("John Doe").build();

        manager.create_profile(profile).await.unwrap();

        let retrieved = manager.get_profile("user123").await.unwrap();
        assert_eq!(retrieved.user_id, "user123");
        assert_eq!(retrieved.name, "John Doe");
    }

    #[tokio::test]
    async fn test_user_not_found() {
        let manager = UserProfileManager::new();

        let result = manager.get_profile("nonexistent").await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            UserManagementError::UserNotFound { .. }
        ));
    }

    #[tokio::test]
    async fn test_update_profile() {
        let manager = UserProfileManager::new();

        let profile = UserProfileBuilder::new("user123").name("John Doe").build();

        manager.create_profile(profile).await.unwrap();

        manager
            .update_profile("user123", |p| {
                p.name = "Jane Doe".to_string();
            })
            .await
            .unwrap();

        let updated = manager.get_profile("user123").await.unwrap();
        assert_eq!(updated.name, "Jane Doe");
    }

    #[tokio::test]
    async fn test_lifecycle_stage_new() {
        let manager = UserProfileManager::new();

        let profile = UserProfileBuilder::new("user123").name("New User").build();

        assert_eq!(profile.lifecycle_stage, LifecycleStage::New);
        let stage = manager.calculate_lifecycle_stage(&profile);
        assert_eq!(stage, LifecycleStage::New);
    }

    #[tokio::test]
    async fn test_engagement_score_calculation() {
        let manager = UserProfileManager::new();

        let mut profile = UserProfileBuilder::new("user123").build();
        profile.activity.sessions_last_7_days = 5;
        profile.activity.current_streak = 5;
        profile.activity.completion_rate = 0.8;
        profile.activity.avg_session_duration = 30.0;

        let score = manager.calculate_engagement_score(&profile);
        assert!(score > 0.5); // Should have decent engagement
    }

    #[tokio::test]
    async fn test_user_segmentation() {
        let manager = UserProfileManager::new();

        let mut profile = UserProfileBuilder::new("user123").build();
        profile.engagement_score = 0.8;
        profile.skill_level = SkillLevel::Advanced;
        profile.lifecycle_stage = LifecycleStage::Active; // Set to active, not new

        let segment = manager.calculate_segment(&profile);
        assert_eq!(segment, UserSegment::HighPerformer);
    }

    #[tokio::test]
    async fn test_user_search() {
        let manager = UserProfileManager::new();

        // Create multiple users
        for i in 0..5 {
            let profile = UserProfileBuilder::new(format!("user{}", i))
                .name(format!("User {}", i))
                .build();
            manager.create_profile(profile).await.unwrap();
        }

        let criteria = UserSearchCriteria {
            lifecycle_stage: Some(LifecycleStage::New),
            ..Default::default()
        };

        let results = manager.search_users(&criteria).await.unwrap();
        assert_eq!(results.len(), 5);
    }

    #[tokio::test]
    async fn test_recommendations() {
        let manager = UserProfileManager::new();

        let profile = UserProfileBuilder::new("user123").name("John Doe").build();

        manager.create_profile(profile).await.unwrap();

        let recommendations = manager.get_recommendations("user123", 3).await.unwrap();
        assert!(!recommendations.is_empty());
        assert!(recommendations.len() <= 3);
    }

    #[tokio::test]
    async fn test_record_activity() {
        let manager = UserProfileManager::new();

        let profile = UserProfileBuilder::new("user123").name("John Doe").build();

        manager.create_profile(profile).await.unwrap();

        manager.record_activity("user123").await.unwrap();

        let updated = manager.get_profile("user123").await.unwrap();
        assert_eq!(updated.activity.total_sessions, 1);
        assert!(updated.activity.last_active.is_some());
    }

    #[tokio::test]
    async fn test_engagement_distribution() {
        let manager = UserProfileManager::new();

        // Create users with different engagement levels
        for i in 0..10 {
            let mut profile = UserProfileBuilder::new(format!("user{}", i)).build();
            profile.engagement_score = i as f32 / 10.0;
            manager.create_profile(profile).await.unwrap();
        }

        let distribution = manager.get_engagement_distribution().await;
        assert!(!distribution.is_empty());
    }

    #[tokio::test]
    async fn test_delete_profile() {
        let manager = UserProfileManager::new();

        let profile = UserProfileBuilder::new("user123").name("John Doe").build();

        manager.create_profile(profile).await.unwrap();
        assert_eq!(manager.get_user_count().await, 1);

        manager.delete_profile("user123").await.unwrap();
        assert_eq!(manager.get_user_count().await, 0);
    }
}
