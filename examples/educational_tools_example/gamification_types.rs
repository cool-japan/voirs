//! Data structures supporting the gamification subsystem (challenges,
//! achievements, social activities, and motivation strategies).

use crate::types::SkillLevel;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

// Additional supporting types for gamification
#[derive(Debug, Clone)]
pub struct LearningChallenge {
    pub challenge_id: Uuid,
    pub title: String,
    pub description: String,
    pub challenge_type: ChallengeType,
    pub target_metric: String,
    pub target_value: f32,
    pub reward_points: u32,
    pub time_limit: Option<Duration>,
    pub difficulty: ChallengeDifficulty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeType {
    Pronunciation,
    Vocabulary,
    Grammar,
    Conversation,
    Listening,
    Culture,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeDifficulty {
    Easy,
    Medium,
    Hard,
    Expert,
}

#[derive(Debug, Clone)]
pub struct Achievement {
    pub achievement_id: Uuid,
    pub name: String,
    pub description: String,
    pub category: AchievementCategory,
    pub target_progress: u32,
    pub current_progress: u32,
    pub reward_points: u32,
    pub badge_icon: String,
    pub unlocked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AchievementCategory {
    Progress,
    Skill,
    Knowledge,
    Consistency,
    Social,
    Challenge,
}

#[derive(Debug, Clone)]
pub struct SocialActivity {
    pub activity_id: Uuid,
    pub activity_type: SocialActivityType,
    pub title: String,
    pub description: String,
    pub max_participants: u8,
    pub skill_level_requirement: SkillLevel,
    pub language: String,
    pub scheduled_time: Option<SystemTime>,
    pub duration: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SocialActivityType {
    GroupConversation,
    PeerReview,
    Competition,
    Collaboration,
    Mentoring,
}

#[derive(Debug, Clone)]
pub struct MotivationStrategy {
    pub strategy_id: Uuid,
    pub strategy_name: String,
    pub description: String,
    pub target_motivation_type: MotivationType,
    pub effectiveness: f32,
    pub implementation_notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MotivationType {
    Achievement,
    Social,
    Mastery,
    Purpose,
    Autonomy,
}

// Extension traits for enum names
impl SocialActivityType {
    pub(crate) fn name(&self) -> &str {
        match self {
            SocialActivityType::GroupConversation => "Group Conversation",
            SocialActivityType::PeerReview => "Peer Review",
            SocialActivityType::Competition => "Competition",
            SocialActivityType::Collaboration => "Collaboration",
            SocialActivityType::Mentoring => "Mentoring",
        }
    }
}
