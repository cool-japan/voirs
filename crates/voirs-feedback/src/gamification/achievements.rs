//! Achievement system for user engagement and progression tracking
//!
//! This module provides a comprehensive achievement framework including:
//! - Badge management and progression tracking
//! - Achievement unlock conditions and validation
//! - Tiered achievement system with rewards
//! - Progress tracking and completion notifications

use crate::traits::{AchievementTier, FocusArea, SessionScores, TrainingStatistics, UserProgress};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

/// Achievement system manager
#[derive(Debug, Clone)]
pub struct AchievementSystem {
    /// Registered achievements
    achievements: HashMap<String, Achievement>,
    /// User achievements
    user_achievements: HashMap<Uuid, Vec<UserAchievement>>,
    /// Achievement categories
    categories: Vec<AchievementCategory>,
}

impl AchievementSystem {
    /// Create a new achievement system
    pub fn new() -> Self {
        let mut system = Self {
            achievements: HashMap::new(),
            user_achievements: HashMap::new(),
            categories: Vec::new(),
        };

        system.initialize_default_achievements();
        system
    }

    /// Initialize default achievements
    fn initialize_default_achievements(&mut self) {
        // Beginner achievements
        self.register_achievement(Achievement {
            id: "first_session".to_string(),
            name: "First Steps".to_string(),
            description: "Complete your first practice session".to_string(),
            category: AchievementCategory::Progress,
            tier: AchievementTier::Bronze,
            unlock_condition: UnlockCondition::SessionCount(1),
            rewards: vec![
                Reward::Points(10),
                Reward::Badge("first_session".to_string()),
            ],
            is_hidden: false,
            icon_url: None,
        });

        self.register_achievement(Achievement {
            id: "week_streak".to_string(),
            name: "Consistent Learner".to_string(),
            description: "Maintain a 7-day practice streak".to_string(),
            category: AchievementCategory::Consistency,
            tier: AchievementTier::Silver,
            unlock_condition: UnlockCondition::StreakDays(7),
            rewards: vec![Reward::Points(50), Reward::Badge("week_streak".to_string())],
            is_hidden: false,
            icon_url: None,
        });

        self.register_achievement(Achievement {
            id: "pronunciation_master".to_string(),
            name: "Pronunciation Master".to_string(),
            description: "Achieve 95% pronunciation accuracy in 10 sessions".to_string(),
            category: AchievementCategory::Skill,
            tier: AchievementTier::Gold,
            unlock_condition: UnlockCondition::PronunciationAccuracy {
                threshold: 0.95,
                sessions: 10,
            },
            rewards: vec![
                Reward::Points(100),
                Reward::Badge("pronunciation_master".to_string()),
                Reward::Title("Pronunciation Master".to_string()),
            ],
            is_hidden: false,
            icon_url: None,
        });

        self.register_achievement(Achievement {
            id: "speed_demon".to_string(),
            name: "Speed Demon".to_string(),
            description: "Complete 5 sessions in under 2 minutes each".to_string(),
            category: AchievementCategory::Performance,
            tier: AchievementTier::Silver,
            unlock_condition: UnlockCondition::SpeedChallenge {
                max_duration_seconds: 120,
                count: 5,
            },
            rewards: vec![Reward::Points(75), Reward::Badge("speed_demon".to_string())],
            is_hidden: false,
            icon_url: None,
        });

        self.register_achievement(Achievement {
            id: "social_butterfly".to_string(),
            name: "Social Butterfly".to_string(),
            description: "Complete 3 collaborative challenges with peers".to_string(),
            category: AchievementCategory::Social,
            tier: AchievementTier::Bronze,
            unlock_condition: UnlockCondition::CollaborativeChallenges(3),
            rewards: vec![
                Reward::Points(30),
                Reward::Badge("social_butterfly".to_string()),
            ],
            is_hidden: false,
            icon_url: None,
        });

        // Advanced achievements
        self.register_achievement(Achievement {
            id: "perfectionist".to_string(),
            name: "Perfectionist".to_string(),
            description: "Achieve 100% accuracy in any session".to_string(),
            category: AchievementCategory::Skill,
            tier: AchievementTier::Platinum,
            unlock_condition: UnlockCondition::PerfectSession,
            rewards: vec![
                Reward::Points(200),
                Reward::Badge("perfectionist".to_string()),
                Reward::Title("Perfectionist".to_string()),
            ],
            is_hidden: true,
            icon_url: None,
        });

        self.register_achievement(Achievement {
            id: "month_streak".to_string(),
            name: "Dedicated Learner".to_string(),
            description: "Maintain a 30-day practice streak".to_string(),
            category: AchievementCategory::Consistency,
            tier: AchievementTier::Platinum,
            unlock_condition: UnlockCondition::StreakDays(30),
            rewards: vec![
                Reward::Points(300),
                Reward::Badge("month_streak".to_string()),
                Reward::Title("Dedicated Learner".to_string()),
            ],
            is_hidden: false,
            icon_url: None,
        });

        // Initialize categories
        self.categories = vec![
            AchievementCategory::Progress,
            AchievementCategory::Skill,
            AchievementCategory::Consistency,
            AchievementCategory::Social,
            AchievementCategory::Performance,
            AchievementCategory::Special,
        ];
    }

    /// Register a new achievement
    pub fn register_achievement(&mut self, achievement: Achievement) {
        self.achievements
            .insert(achievement.id.clone(), achievement);
    }

    /// Check and unlock achievements for a user
    pub fn check_achievements(
        &mut self,
        user_id: Uuid,
        progress: &UserProgress,
    ) -> Vec<UnlockedAchievement> {
        let mut unlocked = Vec::new();

        for achievement in self.achievements.values() {
            if self.is_achievement_unlocked(user_id, &achievement.id) {
                continue;
            }

            if self.evaluate_unlock_condition(&achievement.unlock_condition, progress) {
                let user_achievement = UserAchievement {
                    achievement_id: achievement.id.clone(),
                    unlocked_at: Utc::now(),
                    progress: 1.0,
                    is_notified: false,
                };

                self.user_achievements
                    .entry(user_id)
                    .or_insert_with(Vec::new)
                    .push(user_achievement);

                unlocked.push(UnlockedAchievement {
                    achievement: achievement.clone(),
                    unlocked_at: Utc::now(),
                    rewards: achievement.rewards.clone(),
                });
            }
        }

        unlocked
    }

    /// Check if achievement is already unlocked
    pub fn is_achievement_unlocked(&self, user_id: Uuid, achievement_id: &str) -> bool {
        self.user_achievements
            .get(&user_id)
            .map(|achievements| {
                achievements
                    .iter()
                    .any(|a| a.achievement_id == achievement_id)
            })
            .unwrap_or(false)
    }

    /// Get user's achievement progress
    pub fn get_achievement_progress(
        &self,
        user_id: Uuid,
        achievement_id: &str,
        progress: &UserProgress,
    ) -> f32 {
        if self.is_achievement_unlocked(user_id, achievement_id) {
            return 1.0;
        }

        if let Some(achievement) = self.achievements.get(achievement_id) {
            self.calculate_progress(&achievement.unlock_condition, progress)
        } else {
            0.0
        }
    }

    /// Get all achievements for a category
    pub fn get_achievements_by_category(&self, category: AchievementCategory) -> Vec<&Achievement> {
        self.achievements
            .values()
            .filter(|a| a.category == category)
            .collect()
    }

    /// Get user's unlocked achievements
    pub fn get_user_achievements(&self, user_id: Uuid) -> Vec<&Achievement> {
        self.user_achievements
            .get(&user_id)
            .map(|user_achievements| {
                user_achievements
                    .iter()
                    .filter_map(|ua| self.achievements.get(&ua.achievement_id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get achievement completion statistics
    pub fn get_completion_stats(&self, user_id: Uuid) -> AchievementStats {
        let total_achievements = self.achievements.len();
        let unlocked_achievements = self
            .user_achievements
            .get(&user_id)
            .map(|a| a.len())
            .unwrap_or(0);

        let completion_rate = if total_achievements > 0 {
            unlocked_achievements as f32 / total_achievements as f32
        } else {
            0.0
        };

        let mut tier_stats = HashMap::new();
        for achievement in self.achievements.values() {
            let is_unlocked = self.is_achievement_unlocked(user_id, &achievement.id);
            let entry = tier_stats.entry(achievement.tier.clone()).or_insert((0, 0));
            entry.1 += 1; // total count
            if is_unlocked {
                entry.0 += 1; // unlocked count
            }
        }

        AchievementStats {
            total_achievements,
            unlocked_achievements,
            completion_rate,
            tier_stats,
        }
    }

    /// Evaluate unlock condition
    fn evaluate_unlock_condition(
        &self,
        condition: &UnlockCondition,
        progress: &UserProgress,
    ) -> bool {
        match condition {
            UnlockCondition::SessionCount(required) => {
                progress.training_stats.total_sessions >= (*required as usize)
            }
            UnlockCondition::StreakDays(required) => {
                progress.training_stats.current_streak >= (*required as usize)
            }
            UnlockCondition::PronunciationAccuracy {
                threshold,
                sessions,
            } => {
                progress.training_stats.total_sessions >= (*sessions as usize)
                    && progress.average_scores.average_pronunciation >= *threshold
            }
            UnlockCondition::SpeedChallenge {
                max_duration_seconds,
                count,
            } => progress.training_stats.total_sessions >= (*count as usize),
            UnlockCondition::CollaborativeChallenges(required) => {
                progress.training_stats.total_sessions >= (*required as usize)
            }
            UnlockCondition::PerfectSession => progress.training_stats.success_rate >= 1.0,
            UnlockCondition::TotalPoints(required) => {
                progress.overall_skill_level >= (*required as f32 / 100.0)
            }
            UnlockCondition::FocusAreaMastery(focus_area) => progress
                .skill_breakdown
                .get(focus_area)
                .map(|&score| score >= 0.9)
                .unwrap_or(false),
        }
    }

    /// Calculate progress towards achievement
    fn calculate_progress(&self, condition: &UnlockCondition, progress: &UserProgress) -> f32 {
        match condition {
            UnlockCondition::SessionCount(required) => {
                (progress.training_stats.total_sessions as f32 / *required as f32).min(1.0)
            }
            UnlockCondition::StreakDays(required) => {
                (progress.training_stats.current_streak as f32 / *required as f32).min(1.0)
            }
            UnlockCondition::PronunciationAccuracy {
                threshold,
                sessions,
            } => {
                let session_progress =
                    (progress.training_stats.total_sessions as f32 / *sessions as f32).min(1.0);
                let accuracy_progress =
                    (progress.average_scores.average_pronunciation / threshold).min(1.0);
                (session_progress + accuracy_progress) / 2.0
            }
            UnlockCondition::SpeedChallenge { count, .. } => {
                (progress.training_stats.total_sessions as f32 / *count as f32).min(1.0)
            }
            UnlockCondition::CollaborativeChallenges(required) => {
                (progress.training_stats.total_sessions as f32 / *required as f32).min(1.0)
            }
            UnlockCondition::PerfectSession => {
                if progress.training_stats.success_rate >= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
            UnlockCondition::TotalPoints(required) => {
                (progress.overall_skill_level * 100.0 / *required as f32).min(1.0)
            }
            UnlockCondition::FocusAreaMastery(focus_area) => progress
                .skill_breakdown
                .get(focus_area)
                .copied()
                .unwrap_or(0.0),
        }
    }
}

impl Default for AchievementSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Achievement definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Achievement {
    /// Unique achievement identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: String,
    /// Achievement category
    pub category: AchievementCategory,
    /// Achievement tier
    pub tier: AchievementTier,
    /// Condition to unlock
    pub unlock_condition: UnlockCondition,
    /// Rewards for unlocking
    pub rewards: Vec<Reward>,
    /// Whether achievement is hidden until unlocked
    pub is_hidden: bool,
    /// Optional icon URL
    pub icon_url: Option<String>,
}

/// Achievement categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AchievementCategory {
    /// Description
    Progress,
    /// Description
    Skill,
    /// Description
    Consistency,
    /// Description
    Social,
    /// Description
    Performance,
    /// Description
    Special,
}

/// User's achievement record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAchievement {
    /// Achievement ID
    pub achievement_id: String,
    /// When it was unlocked
    pub unlocked_at: DateTime<Utc>,
    /// Progress towards achievement (0.0 to 1.0)
    pub progress: f32,
    /// Whether user has been notified
    pub is_notified: bool,
}

/// Unlock conditions for achievements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnlockCondition {
    /// Complete a certain number of sessions
    SessionCount(u32),
    /// Maintain streak for days
    StreakDays(u32),
    /// Achieve pronunciation accuracy threshold
    PronunciationAccuracy { threshold: f32, sessions: u32 },
    /// Complete speed challenge
    SpeedChallenge {
        /// Description
        max_duration_seconds: u32,
        /// Description
        count: u32,
    },
    /// Complete collaborative challenges
    CollaborativeChallenges(u32),
    /// Achieve perfect session score
    PerfectSession,
    /// Accumulate total points
    TotalPoints(u32),
    /// Master a focus area
    FocusAreaMastery(FocusArea),
}

/// Achievement rewards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Reward {
    /// Point reward
    Points(u32),
    /// Badge reward
    Badge(String),
    /// Title reward
    Title(String),
    /// Custom reward
    Custom { name: String, description: String },
}

/// Recently unlocked achievement
#[derive(Debug, Clone)]
pub struct UnlockedAchievement {
    /// The achievement
    pub achievement: Achievement,
    /// When it was unlocked
    pub unlocked_at: DateTime<Utc>,
    /// Rewards earned
    pub rewards: Vec<Reward>,
}

/// Achievement completion statistics
#[derive(Debug, Clone)]
pub struct AchievementStats {
    /// Total number of achievements
    pub total_achievements: usize,
    /// Number of unlocked achievements
    pub unlocked_achievements: usize,
    /// Completion rate (0.0 to 1.0)
    pub completion_rate: f32,
    /// Statistics by tier (unlocked, total)
    pub tier_stats: HashMap<AchievementTier, (usize, usize)>,
}

/// Badge manager for visual achievement representation
#[derive(Debug, Clone)]
pub struct BadgeManager {
    /// Available badges
    badges: HashMap<String, Badge>,
    /// User badge collections
    user_badges: HashMap<Uuid, Vec<UserBadge>>,
}

impl BadgeManager {
    /// Create a new badge manager
    pub fn new() -> Self {
        Self {
            badges: HashMap::new(),
            user_badges: HashMap::new(),
        }
    }

    /// Register a new badge
    pub fn register_badge(&mut self, badge: Badge) {
        self.badges.insert(badge.id.clone(), badge);
    }

    /// Award badge to user
    pub fn award_badge(&mut self, user_id: Uuid, badge_id: &str) -> Result<(), String> {
        if !self.badges.contains_key(badge_id) {
            return Err(format!("Badge '{}' not found", badge_id));
        }

        let user_badge = UserBadge {
            badge_id: badge_id.to_string(),
            earned_at: Utc::now(),
            is_displayed: false,
        };

        self.user_badges
            .entry(user_id)
            .or_insert_with(Vec::new)
            .push(user_badge);

        Ok(())
    }

    /// Get user's badges
    pub fn get_user_badges(&self, user_id: Uuid) -> Vec<(&Badge, DateTime<Utc>)> {
        self.user_badges
            .get(&user_id)
            .map(|user_badges| {
                user_badges
                    .iter()
                    .filter_map(|ub| {
                        self.badges
                            .get(&ub.badge_id)
                            .map(|badge| (badge, ub.earned_at))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Set badge display status
    pub fn set_badge_display(&mut self, user_id: Uuid, badge_id: &str, displayed: bool) {
        if let Some(user_badges) = self.user_badges.get_mut(&user_id) {
            if let Some(user_badge) = user_badges.iter_mut().find(|ub| ub.badge_id == badge_id) {
                user_badge.is_displayed = displayed;
            }
        }
    }
}

impl Default for BadgeManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Badge definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Badge {
    /// Unique badge identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: String,
    /// Icon URL
    pub icon_url: String,
    /// Badge rarity
    pub rarity: BadgeRarity,
    /// Associated achievement
    pub achievement_id: Option<String>,
}

/// Badge rarity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BadgeRarity {
    /// Description
    Common,
    /// Description
    Uncommon,
    /// Description
    Rare,
    /// Description
    Epic,
    /// Description
    Legendary,
}

/// User's badge record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserBadge {
    /// Badge ID
    pub badge_id: String,
    /// When badge was earned
    pub earned_at: DateTime<Utc>,
    /// Whether badge is displayed on profile
    pub is_displayed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_achievement_system_creation() {
        let system = AchievementSystem::new();
        assert!(!system.achievements.is_empty());
        assert!(!system.categories.is_empty());
    }

    #[test]
    fn test_achievement_unlock() {
        let mut system = AchievementSystem::new();
        let user_id = Uuid::new_v4();

        let mut training_stats = TrainingStatistics::default();
        training_stats.total_sessions = 1;

        let progress = UserProgress {
            user_id: user_id.to_string(),
            overall_skill_level: 0.8,
            skill_breakdown: HashMap::new(),
            progress_history: Vec::new(),
            achievements: Vec::new(),
            training_stats,
            goals: Vec::new(),
            last_updated: chrono::Utc::now(),
            average_scores: SessionScores::default(),
            skill_levels: HashMap::new(),
            recent_sessions: Vec::new(),
            personal_bests: HashMap::new(),
            session_count: 1,
            total_practice_time: Duration::from_secs(300),
        };

        let unlocked = system.check_achievements(user_id, &progress);
        assert!(!unlocked.is_empty());

        // Should unlock "first_session" achievement
        assert!(unlocked.iter().any(|a| a.achievement.id == "first_session"));
    }

    #[test]
    fn test_achievement_progress_calculation() {
        let system = AchievementSystem::new();
        let user_id = Uuid::new_v4();

        let mut training_stats = TrainingStatistics::default();
        training_stats.current_streak = 3; // 3 out of 7 days = ~0.43

        let progress = UserProgress {
            user_id: user_id.to_string(),
            overall_skill_level: 0.8,
            skill_breakdown: HashMap::new(),
            progress_history: Vec::new(),
            achievements: Vec::new(),
            training_stats,
            goals: Vec::new(),
            last_updated: chrono::Utc::now(),
            average_scores: SessionScores::default(),
            skill_levels: HashMap::new(),
            recent_sessions: Vec::new(),
            personal_bests: HashMap::new(),
            session_count: 5,
            total_practice_time: Duration::from_secs(1500),
        };

        let week_streak_progress =
            system.get_achievement_progress(user_id, "week_streak", &progress);
        assert!((week_streak_progress - 0.43).abs() < 0.1); // 3/7 ≈ 0.43
    }

    #[test]
    fn test_badge_manager() {
        let mut badge_manager = BadgeManager::new();
        let user_id = Uuid::new_v4();

        let badge = Badge {
            id: "test_badge".to_string(),
            name: "Test Badge".to_string(),
            description: "A test badge".to_string(),
            icon_url: "http://example.com/badge.png".to_string(),
            rarity: BadgeRarity::Common,
            achievement_id: Some("test_achievement".to_string()),
        };

        badge_manager.register_badge(badge);

        let result = badge_manager.award_badge(user_id, "test_badge");
        assert!(result.is_ok());

        let user_badges = badge_manager.get_user_badges(user_id);
        assert_eq!(user_badges.len(), 1);
        assert_eq!(user_badges[0].0.id, "test_badge");
    }

    #[test]
    fn test_achievement_stats() {
        let mut system = AchievementSystem::new();
        let user_id = Uuid::new_v4();

        // Unlock one achievement
        let progress = UserProgress {
            user_id: user_id.to_string(),
            overall_skill_level: 0.8,
            skill_breakdown: HashMap::new(),
            progress_history: Vec::new(),
            achievements: Vec::new(),
            training_stats: crate::traits::TrainingStatistics {
                total_sessions: 1,
                successful_sessions: 1,
                total_training_time: std::time::Duration::from_secs(300),
                exercises_completed: 5,
                success_rate: 0.8,
                average_improvement: 0.1,
                current_streak: 0,
                longest_streak: 0,
            },
            goals: Vec::new(),
            last_updated: chrono::Utc::now(),
            average_scores: crate::traits::SessionScores::default(),
            skill_levels: HashMap::new(),
            recent_sessions: Vec::new(),
            personal_bests: HashMap::new(),
            session_count: 1,
            total_practice_time: std::time::Duration::from_secs(300),
        };

        system.check_achievements(user_id, &progress);

        let stats = system.get_completion_stats(user_id);
        assert!(stats.unlocked_achievements > 0);
        assert!(stats.completion_rate > 0.0 && stats.completion_rate <= 1.0);
    }
}
