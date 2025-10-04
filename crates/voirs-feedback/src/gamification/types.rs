//! Common data structures and types for the gamification system

use crate::traits::{
    AchievementTier, FocusArea, SessionState, TimeOfDay, UserBehaviorPatterns, UserProgress,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

// ============================================================================
// Configuration Types
// ============================================================================

/// Gamification configuration
#[derive(Debug, Clone)]
pub struct GamificationConfig {
    /// Enable achievement system
    pub enable_achievements: bool,
    /// Enable leaderboards
    pub enable_leaderboards: bool,
    /// Points awarded for level up
    pub level_up_bonus: u32,
    /// Points awarded for streak milestones
    pub streak_bonus_points: u32,
    /// Maximum leaderboard size
    pub max_leaderboard_size: usize,
}

impl Default for GamificationConfig {
    fn default() -> Self {
        Self {
            enable_achievements: true,
            enable_leaderboards: true,
            level_up_bonus: 50,
            streak_bonus_points: 25,
            max_leaderboard_size: 100,
        }
    }
}

/// Social system configuration
#[derive(Debug, Clone)]
pub struct SocialConfig {
    /// Enable peer comparisons
    pub enable_peer_comparisons: bool,
    /// Enable collaborative challenges
    pub enable_collaborative_challenges: bool,
    /// Enable mentorship matching
    pub enable_mentorship: bool,
    /// Enable community forums
    pub enable_forums: bool,
    /// Maximum peer group size
    pub max_peer_group_size: usize,
}

impl Default for SocialConfig {
    fn default() -> Self {
        Self {
            enable_peer_comparisons: true,
            enable_collaborative_challenges: true,
            enable_mentorship: true,
            enable_forums: true,
            max_peer_group_size: 10,
        }
    }
}

/// Point system configuration
#[derive(Debug, Clone)]
pub struct PointSystemConfig {
    /// Base points per session
    pub base_points_per_session: u32,
    /// Bonus multiplier for streaks
    pub streak_bonus_multiplier: f32,
    /// Enable point marketplace
    pub enable_marketplace: bool,
    /// Enable point transfers
    pub enable_transfers: bool,
    /// Maximum daily points
    pub max_daily_points: u32,
}

impl Default for PointSystemConfig {
    fn default() -> Self {
        Self {
            base_points_per_session: 10,
            streak_bonus_multiplier: 1.5,
            enable_marketplace: true,
            enable_transfers: true,
            max_daily_points: 1000,
        }
    }
}

/// Challenge system configuration
#[derive(Debug, Clone)]
pub struct ChallengeConfig {
    /// Maximum active challenges per user
    pub max_active_challenges: usize,
    /// Challenge refresh interval (days)
    pub challenge_refresh_days: u32,
    /// Enable time-limited events
    pub enable_time_limited_events: bool,
    /// Enable community challenges
    pub enable_community_challenges: bool,
}

impl Default for ChallengeConfig {
    fn default() -> Self {
        Self {
            max_active_challenges: 5,
            challenge_refresh_days: 7,
            enable_time_limited_events: true,
            enable_community_challenges: true,
        }
    }
}

/// Motivation system configuration
#[derive(Debug, Clone)]
pub struct MotivationConfig {
    /// Enable burnout monitoring
    pub enable_burnout_monitoring: bool,
    /// Enable intervention system
    pub enable_interventions: bool,
    /// Enable re-engagement campaigns
    pub enable_reengagement: bool,
    /// Motivation check interval (hours)
    pub motivation_check_interval: u32,
}

impl Default for MotivationConfig {
    fn default() -> Self {
        Self {
            enable_burnout_monitoring: true,
            enable_interventions: true,
            enable_reengagement: true,
            motivation_check_interval: 24,
        }
    }
}

// ============================================================================
// Achievement System Types
// ============================================================================

/// Achievement definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchievementDefinition {
    /// Unique achievement ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: String,
    /// Unlock condition
    pub unlock_condition: AchievementUnlockCondition,
    /// Points awarded
    pub points: u32,
    /// Achievement tier
    pub tier: AchievementTier,
    /// Category
    pub category: AchievementCategory,
    /// Icon or emoji
    pub icon: String,
    /// Whether hidden until unlocked
    pub hidden: bool,
}

/// Achievement unlock conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AchievementUnlockCondition {
    /// Complete a number of sessions
    SessionCount(usize),
    /// Earn a number of points
    PointsEarned(u32),
    /// Reach skill level in specific area
    SkillLevel {
        /// Focus area to improve
        area: FocusArea,
        /// Target skill level
        level: f32,
    },
    /// Maintain consecutive streak
    ConsecutiveStreak(usize),
    /// Total time spent practicing
    TimeSpent(Duration),
    /// Perfect session performance
    PerfectSession,
    /// High improvement rate
    ImprovementRate(f32),
    /// Unlock multiple achievements
    MultipleAchievements(usize),
    /// Custom metric condition
    CustomMetric {
        /// Metric name
        metric: String,
        /// Target value
        value: f32,
    },
}

/// Achievement categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AchievementCategory {
    /// Beginner achievements
    Beginner,
    /// Progress milestones
    Progress,
    /// Skill mastery
    Mastery,
    /// Consistency achievements
    Consistency,
    /// Performance achievements
    Performance,
    /// Dedication achievements
    Dedication,
    /// Special/hidden achievements
    Special,
    /// Practice-related achievements
    Practice,
    /// Habit formation achievements
    Habit,
    /// Social and collaboration achievements
    Social,
    /// Special legendary achievements
    Legendary,
}

/// User's achievement progress
#[derive(Debug, Clone)]
pub struct UserAchievementProgress {
    /// User ID
    pub user_id: String,
    /// Unlocked achievements
    pub unlocked_achievements: HashMap<String, UnlockedAchievement>,
    /// Total achievement points
    pub achievement_points: u32,
    /// Current level
    pub current_level: u32,
    /// Experience points toward next level
    pub experience_points: u32,
    /// Streak tracking
    pub streak_data: StreakData,
    /// Progress toward milestone achievements
    pub milestone_progress: HashMap<String, f32>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

/// Unlocked achievement record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnlockedAchievement {
    /// Achievement that was unlocked
    pub achievement: AchievementDefinition,
    /// When it was unlocked
    pub unlock_timestamp: DateTime<Utc>,
    /// Session ID where it was unlocked
    pub unlock_session: Uuid,
    /// Bonus points awarded
    pub bonus_points: u32,
}

/// Streak tracking data
#[derive(Debug, Clone, Default)]
pub struct StreakData {
    /// Current active streak
    pub current_streak: usize,
    /// Longest streak ever achieved
    pub longest_streak: usize,
    /// Last activity date
    pub last_activity: DateTime<Utc>,
    /// Streak freeze count (for skipped days)
    pub freeze_count: usize,
}

/// User achievement summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAchievementSummary {
    /// User ID
    pub user_id: String,
    /// Total points earned
    pub total_points: u32,
    /// Current level
    pub current_level: u32,
    /// Experience points
    pub experience_points: u32,
    /// XP needed for next level
    pub next_level_requirement: u32,
    /// Number of unlocked achievements
    pub unlocked_achievements: usize,
    /// Total available achievements
    pub total_achievements: usize,
    /// Completion percentage
    pub completion_percentage: f32,
    /// Current practice streak
    pub current_streak: usize,
    /// Longest practice streak
    pub longest_streak: usize,
    /// Upcoming achievements
    pub upcoming_achievements: Vec<UpcomingAchievement>,
    /// Recently unlocked achievements
    pub recent_unlocks: Vec<UnlockedAchievement>,
    /// User badges
    pub badges: Vec<Badge>,
}

/// Upcoming achievement (not yet unlocked)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpcomingAchievement {
    /// Achievement definition
    pub achievement: AchievementDefinition,
    /// Progress toward unlocking (0-100%)
    pub progress_percentage: f32,
    /// Estimated time to unlock
    pub estimated_time: Option<Duration>,
}

/// Badge representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Badge {
    /// Badge ID
    pub id: String,
    /// Badge name
    pub name: String,
    /// Badge description
    pub description: String,
    /// Badge rarity
    pub rarity: BadgeRarity,
    /// Badge icon
    pub icon: String,
    /// When earned
    pub earned_at: DateTime<Utc>,
}

/// Badge rarity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BadgeRarity {
    /// Common badge
    Common,
    /// Rare badge
    Rare,
    /// Epic badge
    Epic,
    /// Legendary badge
    Legendary,
}

/// Achievement system statistics
#[derive(Debug, Clone, Default)]
pub struct AchievementStats {
    /// Total achievements created
    pub total_achievements: usize,
    /// Total unlocks across all users
    pub total_unlocks: usize,
    /// Most popular achievement
    pub most_popular_achievement: Option<String>,
    /// Average completion rate
    pub average_completion_rate: f32,
}

// ============================================================================
// Social System Types
// ============================================================================

/// Peer comparison data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerComparison {
    /// Description
    pub user_id: String,
    /// Description
    pub peer_id: String,
    /// Description
    pub comparison_type: String,
    /// Description
    pub score: f64,
    /// Description
    pub timestamp: DateTime<Utc>,
}

/// Mentorship system for connecting users
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MentorshipSystem {
    /// Active mentorship pairs
    pub active_pairs: HashMap<String, MentorshipPair>,
    /// Mentorship applications
    pub applications: Vec<MentorshipApplication>,
    /// Mentorship settings
    pub settings: MentorshipSettings,
}

/// Mentorship pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MentorshipPair {
    /// Mentor user ID
    pub mentor_id: String,
    /// Mentee user ID
    pub mentee_id: String,
    /// Relationship start date
    pub start_date: DateTime<Utc>,
    /// Relationship status
    pub status: MentorshipStatus,
}

/// Mentorship application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MentorshipApplication {
    /// Applicant user ID
    pub applicant_id: String,
    /// Application type
    pub application_type: MentorshipApplicationType,
    /// Application message
    pub message: String,
    /// Application timestamp
    pub timestamp: DateTime<Utc>,
}

/// Mentorship application type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MentorshipApplicationType {
    /// Wants to be a mentor
    Mentor,
    /// Wants to find a mentor
    Mentee,
}

/// Mentorship status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MentorshipStatus {
    /// Active mentorship
    Active,
    /// Paused mentorship
    Paused,
    /// Completed mentorship
    Completed,
    /// Cancelled mentorship
    Cancelled,
}

/// Mentorship settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MentorshipSettings {
    /// Maximum mentees per mentor
    pub max_mentees_per_mentor: u32,
    /// Minimum mentor experience level
    pub min_mentor_level: u32,
    /// Auto-matching enabled
    pub auto_matching: bool,
}

/// Community forums structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityForums {
    /// Forum categories
    pub categories: Vec<ForumCategory>,
    /// Recent posts
    pub recent_posts: Vec<ForumPost>,
    /// Forum statistics
    pub stats: ForumStats,
}

/// Forum category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForumCategory {
    /// Category ID
    pub id: String,
    /// Category name
    pub name: String,
    /// Category description
    pub description: String,
    /// Number of posts
    pub post_count: u32,
    /// Last post timestamp
    pub last_post: Option<DateTime<Utc>>,
}

/// Forum post
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForumPost {
    /// Post ID
    pub id: String,
    /// Author user ID
    pub author_id: String,
    /// Post title
    pub title: String,
    /// Post content
    pub content: String,
    /// Category ID
    pub category_id: String,
    /// Post timestamp
    pub timestamp: DateTime<Utc>,
    /// Number of replies
    pub reply_count: u32,
    /// Number of likes
    pub like_count: u32,
}

/// Forum statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForumStats {
    /// Total posts
    pub total_posts: u32,
    /// Total users
    pub total_users: u32,
    /// Posts today
    pub posts_today: u32,
    /// Most active category
    pub most_active_category: Option<String>,
}

/// Social learning network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialLearningNetwork {
    /// User connections
    pub connections: HashMap<String, Vec<String>>,
    /// Learning groups
    pub learning_groups: Vec<LearningGroup>,
    /// Network statistics
    pub stats: NetworkStats,
}

/// Learning group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningGroup {
    /// Group ID
    pub id: String,
    /// Group name
    pub name: String,
    /// Group description
    pub description: String,
    /// Member user IDs
    pub members: Vec<String>,
    /// Group owner ID
    pub owner_id: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Group type
    pub group_type: LearningGroupType,
}

/// Learning group type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningGroupType {
    /// Study group
    Study,
    /// Practice group
    Practice,
    /// Challenge group
    Challenge,
    /// Social group
    Social,
}

/// Network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    /// Total connections
    pub total_connections: u32,
    /// Active groups
    pub active_groups: u32,
    /// Average group size
    pub avg_group_size: f32,
    /// Network density
    pub network_density: f32,
}

// ============================================================================
// Challenge System Types
// ============================================================================

/// Individual challenge for personal achievement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Challenge {
    /// Unique challenge identifier
    pub id: String,
    /// Challenge name
    pub name: String,
    /// Challenge description
    pub description: String,
    /// Challenge type
    pub challenge_type: ChallengeType,
    /// Target to achieve
    pub target: ChallengeTarget,
    /// Current progress
    pub progress: f64,
    /// Deadline for completion
    pub deadline: Option<DateTime<Utc>>,
    /// Reward for completion
    pub reward: ChallengeReward,
    /// Difficulty level (1-5)
    pub difficulty: u8,
    /// Whether challenge is active
    pub is_active: bool,
}

/// Challenge types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeType {
    /// Practice-based challenge
    Practice,
    /// Skill improvement challenge
    Skill,
    /// Consistency challenge
    Consistency,
    /// Time-based challenge
    Time,
    /// Social challenge
    Social,
    /// Custom challenge
    Custom,
}

/// Challenge target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeTarget {
    /// Target number of sessions
    Sessions(u32),
    /// Target score
    Score(f64),
    /// Target improvement percentage
    Improvement(f64),
    /// Target streak length
    Streak(u32),
    /// Target practice time
    PracticeTime(Duration),
    /// Custom target
    Custom(String),
}

/// Challenge reward
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeReward {
    /// Points awarded
    pub points: u32,
    /// Experience points
    pub experience: u32,
    /// Badge awarded
    pub badge: Option<Badge>,
    /// Achievement unlocked
    pub achievement: Option<String>,
    /// Special reward
    pub special_reward: Option<String>,
}

/// Challenge status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeStatus {
    /// Challenge is active
    Active,
    /// Challenge is completed
    Completed,
    /// Challenge is failed
    Failed,
    /// Challenge is paused
    Paused,
}

/// Collaborative challenge structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeChallenge {
    /// Description
    pub id: String,
    /// Description
    pub name: String,
    /// Description
    pub participants: Vec<String>,
    /// Description
    pub target_score: f64,
    /// Description
    pub current_score: f64,
    /// Description
    pub deadline: DateTime<Utc>,
}

/// Time-limited event for special occasions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeLimitedEvent {
    /// Unique event identifier
    pub id: String,
    /// Event name
    pub name: String,
    /// Event description
    pub description: String,
    /// Event start time
    pub start_time: DateTime<Utc>,
    /// Event end time
    pub end_time: DateTime<Utc>,
    /// Event type
    pub event_type: EventType,
    /// Participation requirements
    pub requirements: Vec<String>,
    /// Event rewards
    pub rewards: Vec<EventReward>,
    /// Maximum participants (None for unlimited)
    pub max_participants: Option<u32>,
    /// Current participant count
    pub current_participants: u32,
    /// Event status
    pub status: EventStatus,
}

/// Event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    /// Weekly challenge
    Weekly,
    /// Monthly challenge
    Monthly,
    /// Seasonal event
    Seasonal,
    /// Special holiday event
    Holiday,
    /// Community milestone
    Milestone,
}

/// Event reward
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventReward {
    /// Reward name
    pub name: String,
    /// Reward description
    pub description: String,
    /// Points awarded
    pub points: u32,
    /// Badge awarded
    pub badge: Option<Badge>,
    /// Special item
    pub special_item: Option<String>,
}

/// Event status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventStatus {
    /// Event is scheduled
    Scheduled,
    /// Event is active
    Active,
    /// Event is completed
    Completed,
    /// Event is cancelled
    Cancelled,
}

/// Community-wide challenge for group participation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityChallenge {
    /// Unique challenge identifier
    pub id: String,
    /// Challenge name
    pub name: String,
    /// Challenge description
    pub description: String,
    /// Community target
    pub community_target: f64,
    /// Current community progress
    pub community_progress: f64,
    /// Individual contributions
    pub contributions: HashMap<String, f64>,
    /// Challenge deadline
    pub deadline: DateTime<Utc>,
    /// Participation threshold for rewards
    pub participation_threshold: f64,
    /// Community rewards
    pub community_rewards: Vec<CommunityReward>,
    /// Challenge status
    pub status: ChallengeStatus,
}

/// Community reward
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityReward {
    /// Reward name
    pub name: String,
    /// Reward description
    pub description: String,
    /// Points for each participant
    pub points_per_participant: u32,
    /// Special community badge
    pub community_badge: Option<Badge>,
    /// Unlock requirement
    pub unlock_requirement: f64,
}

/// Challenge generator for creating personalized challenges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeGenerator {
    /// User-specific challenge history
    pub user_challenges: HashMap<String, Vec<Challenge>>,
    /// Challenge templates
    pub templates: Vec<ChallengeTemplate>,
    /// Generation settings
    pub settings: ChallengeGeneratorSettings,
}

/// Challenge template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeTemplate {
    /// Template ID
    pub id: String,
    /// Template name
    pub name: String,
    /// Template description
    pub description: String,
    /// Challenge type
    pub challenge_type: ChallengeType,
    /// Difficulty range
    pub difficulty_range: (u8, u8),
    /// Target template
    pub target_template: ChallengeTargetTemplate,
    /// Reward template
    pub reward_template: ChallengeRewardTemplate,
}

/// Challenge target template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeTargetTemplate {
    /// Variable session count
    Sessions { min: u32, max: u32 },
    /// Variable score target
    Score { min: f64, max: f64 },
    /// Variable improvement target
    Improvement { min: f64, max: f64 },
    /// Variable streak target
    Streak { min: u32, max: u32 },
    /// Variable time target
    Time { min: Duration, max: Duration },
}

/// Challenge reward template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeRewardTemplate {
    /// Base points
    pub base_points: u32,
    /// Points multiplier based on difficulty
    pub difficulty_multiplier: f64,
    /// Possible badges
    pub possible_badges: Vec<String>,
    /// Possible achievements
    pub possible_achievements: Vec<String>,
}

/// Challenge generator settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeGeneratorSettings {
    /// Maximum challenges per user
    pub max_challenges_per_user: u32,
    /// Challenge refresh interval
    pub refresh_interval: Duration,
    /// Difficulty adjustment factor
    pub difficulty_adjustment: f64,
    /// Personalization weight
    pub personalization_weight: f64,
}

// ============================================================================
// Motivation System Types
// ============================================================================

/// Personality traits for personalized messaging and motivation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityTraits {
    /// Extroversion level (0.0-1.0)
    pub extroversion: f64,
    /// Conscientiousness level (0.0-1.0)
    pub conscientiousness: f64,
    /// Openness to experience (0.0-1.0)
    pub openness: f64,
    /// Neuroticism level (0.0-1.0)
    pub neuroticism: f64,
    /// Agreeableness level (0.0-1.0)
    pub agreeableness: f64,
    /// Preferred communication style
    pub communication_style: CommunicationStyle,
    /// Response to feedback
    pub feedback_preference: FeedbackPreference,
}

/// Communication styles for personality-based messaging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationStyle {
    /// Direct and straightforward
    Direct,
    /// Encouraging and supportive
    Supportive,
    /// Analytical and detailed
    Analytical,
    /// Casual and friendly
    Casual,
    /// Motivational and energetic
    Motivational,
}

/// Feedback preference styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackPreference {
    /// Immediate and frequent feedback
    Immediate,
    /// Detailed analysis and suggestions
    Detailed,
    /// Simple and concise feedback
    Concise,
    /// Positive reinforcement focused
    Positive,
    /// Challenge and improvement focused
    Challenge,
}

/// Motivation factors that drive user engagement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotivationFactors {
    /// Primary motivation drivers
    pub primary_drivers: Vec<MotivationDriver>,
    /// Current motivation level (0.0-1.0)
    pub current_level: f64,
    /// Response to challenges
    pub challenge_response: ChallengeResponse,
    /// Social motivation preferences
    pub social_preferences: SocialPreferences,
    /// Achievement orientation
    pub achievement_orientation: AchievementOrientation,
}

/// Primary motivation drivers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MotivationDriver {
    /// Driven by achievement and success
    Achievement,
    /// Driven by learning and growth
    Learning,
    /// Driven by social connection
    Social,
    /// Driven by competition
    Competition,
    /// Driven by recognition
    Recognition,
    /// Driven by personal improvement
    SelfImprovement,
    /// Driven by fun and enjoyment
    Fun,
}

/// Response to challenge types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeResponse {
    /// Thrives on difficult challenges
    Ambitious,
    /// Prefers gradual progression
    Steady,
    /// Avoids high-pressure situations
    Cautious,
    /// Motivated by competition
    Competitive,
}

/// Social motivation preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialPreferences {
    /// Prefers collaborative activities
    pub prefers_collaboration: bool,
    /// Enjoys competition with others
    pub enjoys_competition: bool,
    /// Wants public recognition
    pub wants_recognition: bool,
    /// Prefers private achievements
    pub prefers_privacy: bool,
}

/// Achievement orientation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AchievementOrientation {
    /// Focuses on mastery and learning
    Mastery,
    /// Focuses on performance and outcomes
    Performance,
    /// Focuses on progress and improvement
    Progress,
    /// Focuses on social comparison
    Social,
}

/// User motivation profile for personalized engagement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotivationProfile {
    /// User identifier
    pub user_id: String,
    /// Personality traits for messaging
    pub personality_traits: PersonalityTraits,
    /// Motivation factors
    pub motivation_factors: MotivationFactors,
    /// Current motivation level (0.0-1.0)
    pub current_motivation_level: f64,
    /// Intervention history
    pub intervention_history: Vec<InterventionRecord>,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Record of intervention applied to user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionRecord {
    /// Intervention type
    pub intervention_type: String,
    /// Timestamp of intervention
    pub timestamp: DateTime<Utc>,
    /// Effectiveness score (0.0-1.0)
    pub effectiveness: Option<f64>,
    /// User response
    pub user_response: Option<String>,
}

/// Context for personalized message generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageContext {
    /// User has low motivation
    LowMotivation,
    /// User achieved something significant
    Achievement,
    /// User is struggling with exercises
    Struggle,
    /// User is making good progress
    Progress,
    /// User has hit a plateau
    Plateau,
    /// Session completed successfully
    SessionComplete,
    /// User returned after absence
    Return,
}

/// Type of motivational message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    /// Encouraging message
    Encouragement,
    /// Celebration message
    Celebration,
    /// Support message
    Support,
    /// Progress acknowledgment
    Acknowledgment,
    /// Motivation boost
    Motivation,
    /// Challenge invitation
    Challenge,
    /// Analytical feedback
    Analysis,
    /// Reminder message
    Reminder,
}

/// Personalized message generated for user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizedMessage {
    /// Target user ID
    pub user_id: String,
    /// Message content
    pub content: String,
    /// Type of message
    pub message_type: MessageType,
    /// Description of personality adaptation
    pub personality_adaptation: String,
    /// When message was generated
    pub timestamp: DateTime<Utc>,
    /// Predicted effectiveness (0.0-1.0)
    pub effectiveness_prediction: f64,
}

/// Re-engagement campaign for inactive users
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReengagementCampaign {
    /// Campaign identifier
    pub id: String,
    /// Target user ID
    pub user_id: String,
    /// Campaign type
    pub campaign_type: ReengagementCampaignType,
    /// Start date
    pub start_date: DateTime<Utc>,
    /// End date
    pub end_date: DateTime<Utc>,
    /// Messages sent
    pub messages_sent: u32,
    /// Engagement rate
    pub engagement_rate: f64,
    /// Whether campaign is active
    pub is_active: bool,
}

/// Re-engagement campaign types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReengagementCampaignType {
    /// Gentle reminder campaign
    Gentle,
    /// Motivational campaign
    Motivational,
    /// Achievement-focused campaign
    Achievement,
    /// Social campaign
    Social,
}

/// User segment for targeting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserSegment {
    /// Inactive users
    Inactive,
    /// Struggling users
    Struggling,
    /// Advanced users
    Advanced,
    /// New users
    New,
    /// Returning users
    Returning,
}

/// Motivation monitoring system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotivationMonitor {
    /// User motivation levels
    pub user_motivation: HashMap<String, f64>,
    /// Monitoring settings
    pub settings: MotivationMonitorSettings,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

/// Motivation monitoring settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotivationMonitorSettings {
    /// Check interval in hours
    pub check_interval: u32,
    /// Minimum sessions for trend analysis
    pub min_sessions_for_trend: u32,
    /// Enable predictive analysis
    pub enable_predictive_analysis: bool,
}

/// Alert thresholds for motivation monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Low motivation threshold
    pub low_motivation: f64,
    /// Critical motivation threshold
    pub critical_motivation: f64,
    /// Burnout risk threshold
    pub burnout_risk: f64,
}

/// Intervention system for user support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionSystem {
    /// Available interventions
    pub interventions: Vec<InterventionType>,
    /// Intervention history
    pub intervention_history: HashMap<String, Vec<InterventionRecord>>,
    /// System settings
    pub settings: InterventionSettings,
}

/// Intervention type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterventionType {
    /// Motivational message
    MotivationalMessage,
    /// Difficulty adjustment
    DifficultyAdjustment,
    /// Break suggestion
    BreakSuggestion,
    /// Goal adjustment
    GoalAdjustment,
    /// Social connection
    SocialConnection,
    /// Achievement highlight
    AchievementHighlight,
}

/// Intervention settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionSettings {
    /// Maximum interventions per day
    pub max_interventions_per_day: u32,
    /// Minimum time between interventions
    pub min_time_between_interventions: Duration,
    /// Enable automated interventions
    pub enable_automated: bool,
}

/// Burnout prevention system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnoutPrevention {
    /// User burnout risk scores
    pub burnout_risk: HashMap<String, f64>,
    /// Prevention strategies
    pub prevention_strategies: Vec<BurnoutPreventionStrategy>,
    /// System settings
    pub settings: BurnoutPreventionSettings,
}

/// Burnout prevention strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BurnoutPreventionStrategy {
    /// Suggest taking breaks
    BreakSuggestion,
    /// Reduce session intensity
    IntensityReduction,
    /// Gamify activities
    Gamification,
    /// Encourage social interaction
    SocialEngagement,
    /// Vary practice routines
    RoutineVariation,
}

/// Burnout prevention settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnoutPreventionSettings {
    /// Risk assessment interval
    pub risk_assessment_interval: Duration,
    /// High risk threshold
    pub high_risk_threshold: f64,
    /// Enable proactive prevention
    pub enable_proactive_prevention: bool,
}

// ============================================================================
// Point System Types
// ============================================================================

/// User point balance for multi-currency system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPointBalance {
    /// User identifier
    pub user_id: String,
    /// Currency balances
    pub currencies: HashMap<PointCurrency, u32>,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Point currency types for multi-currency economy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PointCurrency {
    /// Basic experience points
    Experience,
    /// Achievement points
    Achievement,
    /// Social points for community interaction
    Social,
    /// Premium currency for special purchases
    Premium,
    /// Event-specific currency
    Event,
    /// Skill-specific currency
    Skill,
}

/// Point transfer record for transaction history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointTransfer {
    /// Transfer identifier
    pub transfer_id: String,
    /// Sender user ID
    pub from_user: String,
    /// Recipient user ID
    pub to_user: String,
    /// Currency transferred
    pub currency: PointCurrency,
    /// Amount transferred
    pub amount: u32,
    /// Transfer fee charged
    pub fee: u32,
    /// Transfer timestamp
    pub timestamp: DateTime<Utc>,
    /// Transfer status
    pub status: TransferStatus,
}

/// Transfer status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferStatus {
    /// Transfer is pending
    Pending,
    /// Transfer completed successfully
    Completed,
    /// Transfer failed
    Failed,
    /// Transfer was cancelled
    Cancelled,
}

/// Marketplace purchase record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplacePurchase {
    /// Purchase identifier
    pub purchase_id: String,
    /// Buyer user ID
    pub user_id: String,
    /// Item purchased
    pub item_id: String,
    /// Item name
    pub item_name: String,
    /// Currency used for purchase
    pub currency: PointCurrency,
    /// Amount paid
    pub amount: u32,
    /// Purchase timestamp
    pub timestamp: DateTime<Utc>,
    /// Purchase status
    pub status: PurchaseStatus,
}

/// Purchase status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PurchaseStatus {
    /// Purchase completed
    Completed,
    /// Purchase failed
    Failed,
    /// Purchase was refunded
    Refunded,
}

/// Marketplace item definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceItem {
    /// Item identifier
    pub id: String,
    /// Item name
    pub name: String,
    /// Item description
    pub description: String,
    /// Currency required
    pub currency: PointCurrency,
    /// Price in points
    pub price: u32,
    /// Item category
    pub category: ItemCategory,
    /// Whether item is available
    pub available: bool,
    /// Stock quantity (None for unlimited)
    pub stock: Option<u32>,
    /// Item rarity
    pub rarity: ItemRarity,
}

/// Item category enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ItemCategory {
    /// Cosmetic items
    Cosmetic,
    /// Power-ups and boosts
    PowerUp,
    /// Feature unlocks
    Feature,
    /// Consumable items
    Consumable,
    /// Equipment items
    Equipment,
}

/// Item rarity enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ItemRarity {
    /// Common items
    Common,
    /// Rare items
    Rare,
    /// Epic items
    Epic,
    /// Legendary items
    Legendary,
}

/// Special offer structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialOffer {
    /// Offer identifier
    pub id: String,
    /// Offer title
    pub title: String,
    /// Offer description
    pub description: String,
    /// Discount percentage
    pub discount_percentage: f32,
    /// Affected items
    pub affected_items: Vec<String>,
    /// Offer start time
    pub start_time: DateTime<Utc>,
    /// Offer end time
    pub end_time: DateTime<Utc>,
    /// Maximum uses
    pub max_uses: Option<u32>,
    /// Current uses
    pub current_uses: u32,
}

/// Point marketplace structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointMarketplace {
    /// Available items
    pub items: HashMap<String, MarketplaceItem>,
    /// User inventories
    pub inventories: HashMap<String, Vec<String>>,
    /// Transaction history
    pub transactions: Vec<MarketplacePurchase>,
    /// Special offers
    pub special_offers: Vec<SpecialOffer>,
}

/// Point transfer system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointTransferSystem {
    /// Transfer records
    pub transfers: Vec<PointTransfer>,
    /// Transfer settings
    pub settings: TransferSettings,
    /// Daily transfer limits
    pub daily_limits: HashMap<String, u32>,
}

/// Transfer settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferSettings {
    /// Transfer fee percentage
    pub fee_percentage: f32,
    /// Minimum transfer amount
    pub min_transfer_amount: u32,
    /// Maximum transfer amount
    pub max_transfer_amount: u32,
    /// Daily transfer limit
    pub daily_limit: u32,
}

/// Bonus point event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BonusPointEvent {
    /// Event identifier
    pub id: String,
    /// Event name
    pub name: String,
    /// Event description
    pub description: String,
    /// Point multiplier
    pub multiplier: f32,
    /// Event start time
    pub start_time: DateTime<Utc>,
    /// Event end time
    pub end_time: DateTime<Utc>,
    /// Affected currencies
    pub affected_currencies: Vec<PointCurrency>,
    /// Event status
    pub status: EventStatus,
}

/// Reward type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RewardType {
    /// Point reward
    Points,
    /// Badge reward
    Badge,
    /// Achievement reward
    Achievement,
    /// Item reward
    Item,
    /// Experience reward
    Experience,
}

// ============================================================================
// Leaderboard System Types
// ============================================================================

/// Leaderboard structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Leaderboard {
    /// Type of leaderboard
    pub leaderboard_type: LeaderboardType,
    /// Leaderboard entries
    pub entries: Vec<LeaderboardEntry>,
    /// Last update time
    pub last_updated: DateTime<Utc>,
    /// Total participants
    pub total_participants: usize,
    /// Leaderboard configuration
    pub config: LeaderboardConfig,
    /// Season information (for seasonal leaderboards)
    pub season_info: Option<SeasonInfo>,
    /// Team information (for team-based leaderboards)
    pub teams: HashMap<String, TeamInfo>,
    /// Anonymous participation settings
    pub allow_anonymous: bool,
    /// Fair competition grouping enabled
    pub fair_competition: bool,
}

/// Leaderboard type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeaderboardType {
    /// Global leaderboard
    Global,
    /// Weekly leaderboard
    Weekly,
    /// Monthly leaderboard
    Monthly,
    /// Seasonal leaderboard
    Seasonal,
    /// Points-based leaderboard
    Points,
    /// Skills-based leaderboard
    Skills,
    /// Team-based leaderboard
    Team,
}

/// Leaderboard entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    /// User identifier
    pub user_id: String,
    /// User display name
    pub display_name: String,
    /// User score
    pub score: f32,
    /// User rank
    pub rank: usize,
    /// Points earned
    pub points: u32,
    /// User tier
    pub tier: UserTier,
    /// Achievement count
    pub achievement_count: u32,
    /// Streak count
    pub streak_count: u32,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
    /// Whether user is anonymous
    pub is_anonymous: bool,
}

/// User tier enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserTier {
    /// Beginner tier
    Beginner,
    /// Intermediate tier
    Intermediate,
    /// Advanced tier
    Advanced,
    /// Expert tier
    Expert,
    /// Master tier
    Master,
}

/// Leaderboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardConfig {
    /// Maximum entries to display
    pub max_entries: usize,
    /// Update frequency in seconds
    pub update_frequency_sec: u64,
    /// Enable real-time updates
    pub enable_realtime: bool,
    /// Anonymous display probability (0.0-1.0)
    pub anonymous_probability: f32,
    /// Fair competition grouping enabled
    pub enable_fair_grouping: bool,
    /// Minimum participants for tier-based grouping
    pub min_tier_participants: usize,
}

impl Default for LeaderboardConfig {
    fn default() -> Self {
        Self {
            max_entries: 100,
            update_frequency_sec: 300,
            enable_realtime: true,
            anonymous_probability: 0.1,
            enable_fair_grouping: true,
            min_tier_participants: 10,
        }
    }
}

/// Season information for seasonal leaderboards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonInfo {
    /// Season name
    pub name: String,
    /// Season year
    pub year: u16,
    /// Season start date
    pub start_date: DateTime<Utc>,
    /// Season end date
    pub end_date: DateTime<Utc>,
    /// Season rewards
    pub rewards: Vec<SeasonReward>,
    /// Season theme
    pub theme: String,
}

/// Season rewards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonReward {
    /// Rank range (e.g., 1-3 for top 3)
    pub rank_range: (usize, usize),
    /// Reward description
    pub description: String,
    /// Points awarded
    pub points: u32,
    /// Badge awarded
    pub badge: Option<Badge>,
    /// Special achievement
    pub special_achievement: Option<String>,
}

/// Team information for team-based leaderboards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamInfo {
    /// Team ID
    pub team_id: String,
    /// Team name
    pub team_name: String,
    /// Team members
    pub members: Vec<String>,
    /// Team total score
    pub total_score: f32,
    /// Team creation date
    pub created_at: DateTime<Utc>,
    /// Team captain
    pub captain: String,
    /// Team motto/description
    pub description: String,
}

// ============================================================================
// Strategy Types
// ============================================================================

/// Feedback strategy type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StrategyType {
    /// Encouraging strategy
    Encouraging,
    /// Direct strategy
    Direct,
    /// Technical strategy
    Technical,
    /// Adaptive strategy
    Adaptive,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_gamification_config_default() {
        let config = GamificationConfig::default();

        assert!(config.enable_achievements);
        assert!(config.enable_leaderboards);
        assert_eq!(config.level_up_bonus, 50);
        assert_eq!(config.streak_bonus_points, 25);
        assert_eq!(config.max_leaderboard_size, 100);
    }

    #[test]
    fn test_gamification_config_custom() {
        let config = GamificationConfig {
            enable_achievements: false,
            enable_leaderboards: true,
            level_up_bonus: 100,
            streak_bonus_points: 50,
            max_leaderboard_size: 200,
        };

        assert!(!config.enable_achievements);
        assert!(config.enable_leaderboards);
        assert_eq!(config.level_up_bonus, 100);
        assert_eq!(config.streak_bonus_points, 50);
        assert_eq!(config.max_leaderboard_size, 200);
    }

    #[test]
    fn test_social_config_default() {
        let config = SocialConfig::default();

        assert!(config.enable_peer_comparisons);
        assert!(config.enable_collaborative_challenges);
        assert!(config.enable_mentorship);
        assert!(config.enable_forums);
        assert_eq!(config.max_peer_group_size, 10);
    }

    #[test]
    fn test_social_config_disabled_features() {
        let config = SocialConfig {
            enable_peer_comparisons: false,
            enable_collaborative_challenges: false,
            enable_mentorship: false,
            enable_forums: false,
            max_peer_group_size: 5,
        };

        assert!(!config.enable_peer_comparisons);
        assert!(!config.enable_collaborative_challenges);
        assert!(!config.enable_mentorship);
        assert!(!config.enable_forums);
        assert_eq!(config.max_peer_group_size, 5);
    }

    #[test]
    fn test_point_system_config_default() {
        let config = PointSystemConfig::default();

        assert_eq!(config.base_points_per_session, 10);
        assert_eq!(config.streak_bonus_multiplier, 1.5);
        assert!(config.enable_marketplace);
        assert!(config.enable_transfers);
        assert_eq!(config.max_daily_points, 1000);
    }

    #[test]
    fn test_point_system_config_custom() {
        let config = PointSystemConfig {
            base_points_per_session: 20,
            streak_bonus_multiplier: 2.0,
            enable_marketplace: false,
            enable_transfers: false,
            max_daily_points: 2000,
        };

        assert_eq!(config.base_points_per_session, 20);
        assert_eq!(config.streak_bonus_multiplier, 2.0);
        assert!(!config.enable_marketplace);
        assert!(!config.enable_transfers);
        assert_eq!(config.max_daily_points, 2000);
    }

    #[test]
    fn test_challenge_config_default() {
        let config = ChallengeConfig::default();

        assert_eq!(config.max_active_challenges, 5);
        assert_eq!(config.challenge_refresh_days, 7);
        assert!(config.enable_time_limited_events);
        assert!(config.enable_community_challenges);
    }

    #[test]
    fn test_challenge_config_custom() {
        let config = ChallengeConfig {
            max_active_challenges: 10,
            challenge_refresh_days: 14,
            enable_time_limited_events: false,
            enable_community_challenges: false,
        };

        assert_eq!(config.max_active_challenges, 10);
        assert_eq!(config.challenge_refresh_days, 14);
        assert!(!config.enable_time_limited_events);
        assert!(!config.enable_community_challenges);
    }

    #[test]
    fn test_motivation_config_default() {
        let config = MotivationConfig::default();

        assert!(config.enable_burnout_monitoring);
        assert!(config.enable_interventions);
        assert!(config.enable_reengagement);
        assert_eq!(config.motivation_check_interval, 24);
    }

    #[test]
    fn test_motivation_config_custom() {
        let config = MotivationConfig {
            enable_burnout_monitoring: false,
            enable_interventions: false,
            enable_reengagement: false,
            motivation_check_interval: 48,
        };

        assert!(!config.enable_burnout_monitoring);
        assert!(!config.enable_interventions);
        assert!(!config.enable_reengagement);
        assert_eq!(config.motivation_check_interval, 48);
    }

    #[test]
    fn test_strategy_type_variants() {
        // Test all strategy type variants
        let encouraging = StrategyType::Encouraging;
        let direct = StrategyType::Direct;
        let technical = StrategyType::Technical;
        let adaptive = StrategyType::Adaptive;

        // Test that they can be cloned and compared
        assert_eq!(encouraging.clone(), StrategyType::Encouraging);
        assert_eq!(direct.clone(), StrategyType::Direct);
        assert_eq!(technical.clone(), StrategyType::Technical);
        assert_eq!(adaptive.clone(), StrategyType::Adaptive);
    }

    #[test]
    fn test_team_info_creation() {
        let team = TeamInfo {
            team_id: "team_123".to_string(),
            team_name: "Test Team".to_string(),
            members: vec!["user1".to_string(), "user2".to_string()],
            total_score: 1500.0,
            created_at: Utc::now(),
            captain: "user1".to_string(),
            description: "A test team for unit testing".to_string(),
        };

        assert_eq!(team.team_id, "team_123");
        assert_eq!(team.team_name, "Test Team");
        assert_eq!(team.members.len(), 2);
        assert_eq!(team.total_score, 1500.0);
        assert_eq!(team.captain, "user1");
        assert_eq!(team.description, "A test team for unit testing");
        assert!(team.members.contains(&"user1".to_string()));
        assert!(team.members.contains(&"user2".to_string()));
    }

    #[test]
    fn test_season_reward_creation() {
        let reward = SeasonReward {
            rank_range: (1, 3),
            description: "Top 3 finisher".to_string(),
            points: 500,
            badge: None,
            special_achievement: Some("Season Champion".to_string()),
        };

        assert_eq!(reward.rank_range, (1, 3));
        assert_eq!(reward.description, "Top 3 finisher");
        assert_eq!(reward.points, 500);
        assert!(reward.badge.is_none());
        assert_eq!(
            reward.special_achievement,
            Some("Season Champion".to_string())
        );
    }

    #[test]
    fn test_config_cloning() {
        let original_config = GamificationConfig::default();
        let cloned_config = original_config.clone();

        assert_eq!(
            original_config.enable_achievements,
            cloned_config.enable_achievements
        );
        assert_eq!(original_config.level_up_bonus, cloned_config.level_up_bonus);
        assert_eq!(
            original_config.max_leaderboard_size,
            cloned_config.max_leaderboard_size
        );
    }

    #[test]
    fn test_config_debugging() {
        let config = PointSystemConfig::default();
        let debug_string = format!("{:?}", config);

        // Ensure debug formatting works
        assert!(debug_string.contains("PointSystemConfig"));
        assert!(debug_string.contains("base_points_per_session"));
        assert!(debug_string.contains("10"));
    }

    #[test]
    fn test_extreme_config_values() {
        // Test with extreme values to ensure robustness
        let config = GamificationConfig {
            enable_achievements: true,
            enable_leaderboards: true,
            level_up_bonus: 0,
            streak_bonus_points: u32::MAX,
            max_leaderboard_size: 0,
        };

        assert_eq!(config.level_up_bonus, 0);
        assert_eq!(config.streak_bonus_points, u32::MAX);
        assert_eq!(config.max_leaderboard_size, 0);
    }

    #[test]
    fn test_point_system_multiplier_bounds() {
        let config = PointSystemConfig {
            base_points_per_session: 10,
            streak_bonus_multiplier: 0.0,
            enable_marketplace: true,
            enable_transfers: true,
            max_daily_points: 1000,
        };

        assert_eq!(config.streak_bonus_multiplier, 0.0);

        let config_high = PointSystemConfig {
            base_points_per_session: 10,
            streak_bonus_multiplier: 100.0,
            enable_marketplace: true,
            enable_transfers: true,
            max_daily_points: 1000,
        };

        assert_eq!(config_high.streak_bonus_multiplier, 100.0);
    }

    #[test]
    fn test_serialization_compatibility() {
        // Test that our strategy types can be serialized/deserialized
        let strategy = StrategyType::Adaptive;
        let serialized = serde_json::to_string(&strategy).expect("Failed to serialize");
        let deserialized: StrategyType =
            serde_json::from_str(&serialized).expect("Failed to deserialize");

        assert_eq!(strategy, deserialized);
    }
}
