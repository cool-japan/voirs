//! Challenge framework with personalized generation and time-limited events
//!
//! This module provides comprehensive challenge features including:
//! - Personalized challenge generation based on user progress
//! - Progressive difficulty scaling and adaptive challenges
//! - Time-limited events and community challenges
//! - Achievement-unlocked challenges and skill improvement targets

use crate::traits::{FocusArea, UserProgress};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

/// Challenge system manager
#[derive(Debug, Clone)]
pub struct ChallengeSystem {
    /// Active challenges
    active_challenges: HashMap<Uuid, Challenge>,
    /// User challenge progress
    user_progress: HashMap<Uuid, Vec<UserChallengeProgress>>,
    /// Challenge templates
    templates: Vec<ChallengeTemplate>,
    /// Event challenges
    event_challenges: Vec<EventChallenge>,
}

impl ChallengeSystem {
    /// Create new challenge system
    pub fn new() -> Self {
        let mut system = Self {
            active_challenges: HashMap::new(),
            user_progress: HashMap::new(),
            templates: Vec::new(),
            event_challenges: Vec::new(),
        };
        system.initialize_templates();
        system
    }

    /// Initialize challenge templates
    fn initialize_templates(&mut self) {
        self.templates = vec![
            ChallengeTemplate {
                id: "pronunciation_accuracy".to_string(),
                name: "Pronunciation Master".to_string(),
                description: "Achieve {target_accuracy}% pronunciation accuracy".to_string(),
                challenge_type: ChallengeType::SkillImprovement,
                focus_areas: vec![FocusArea::Pronunciation],
                difficulty_scaling: DifficultyScaling::Progressive {
                    base_target: 0.7,
                    increment: 0.05,
                },
                duration: chrono::Duration::days(7),
                rewards: vec![
                    ChallengeReward::Points {
                        currency: "skill".to_string(),
                        amount: 100,
                    },
                    ChallengeReward::Badge("pronunciation_master".to_string()),
                ],
                requirements: ChallengeRequirements {
                    min_level: 1,
                    required_achievements: Vec::new(),
                    cooldown: Some(chrono::Duration::days(3)),
                },
            },
            ChallengeTemplate {
                id: "consistency_streak".to_string(),
                name: "Consistency Champion".to_string(),
                description: "Practice for {target_days} consecutive days".to_string(),
                challenge_type: ChallengeType::Consistency,
                focus_areas: Vec::new(),
                difficulty_scaling: DifficultyScaling::Linear {
                    min_target: 3,
                    max_target: 30,
                },
                duration: chrono::Duration::days(30),
                rewards: vec![
                    ChallengeReward::Points {
                        currency: "achievement".to_string(),
                        amount: 200,
                    },
                    ChallengeReward::Title("Consistency Champion".to_string()),
                ],
                requirements: ChallengeRequirements {
                    min_level: 1,
                    required_achievements: Vec::new(),
                    cooldown: Some(chrono::Duration::days(7)),
                },
            },
            ChallengeTemplate {
                id: "speed_demon".to_string(),
                name: "Speed Demon".to_string(),
                description:
                    "Complete {target_sessions} sessions in under {time_limit} seconds each"
                        .to_string(),
                challenge_type: ChallengeType::Performance,
                focus_areas: vec![FocusArea::Fluency],
                difficulty_scaling: DifficultyScaling::Adaptive,
                duration: chrono::Duration::days(14),
                rewards: vec![
                    ChallengeReward::Points {
                        currency: "experience".to_string(),
                        amount: 150,
                    },
                    ChallengeReward::Badge("speed_demon".to_string()),
                ],
                requirements: ChallengeRequirements {
                    min_level: 5,
                    required_achievements: vec!["first_session".to_string()],
                    cooldown: Some(chrono::Duration::days(5)),
                },
            },
        ];
    }

    /// Generate personalized challenge for user
    pub fn generate_personalized_challenge(
        &mut self,
        user_id: Uuid,
        user_progress: &UserProgress,
    ) -> Option<Challenge> {
        let user_level = self.calculate_user_level(user_progress);
        let weak_areas = self.identify_weak_areas(user_progress);

        // Find suitable template
        let template = self
            .templates
            .iter()
            .filter(|t| t.requirements.min_level <= user_level)
            .filter(|t| self.check_cooldown(user_id, &t.id))
            .filter(|t| self.check_requirements(user_progress, &t.requirements))
            .min_by_key(|t| {
                // Prioritize challenges for weak areas
                if t.focus_areas.iter().any(|area| weak_areas.contains(area)) {
                    0
                } else {
                    1
                }
            })?;

        let challenge = self.create_challenge_from_template(user_id, template, user_progress);
        self.active_challenges
            .insert(challenge.id, challenge.clone());

        Some(challenge)
    }

    /// Create challenge from template
    fn create_challenge_from_template(
        &self,
        user_id: Uuid,
        template: &ChallengeTemplate,
        user_progress: &UserProgress,
    ) -> Challenge {
        let target_value = self.calculate_target_value(template, user_progress);
        let difficulty = self.calculate_difficulty(template, user_progress);

        Challenge {
            id: Uuid::new_v4(),
            template_id: template.id.clone(),
            name: self.personalize_name(&template.name, target_value),
            description: self.personalize_description(&template.description, target_value),
            challenge_type: template.challenge_type.clone(),
            focus_areas: template.focus_areas.clone(),
            target_value,
            current_progress: 0.0,
            difficulty,
            status: ChallengeStatus::Active,
            created_at: Utc::now(),
            expires_at: Utc::now() + template.duration,
            rewards: template.rewards.clone(),
            participants: vec![user_id],
        }
    }

    /// Update challenge progress
    pub fn update_progress(
        &mut self,
        user_id: Uuid,
        session_data: &SessionData,
    ) -> Vec<ChallengeUpdate> {
        let mut updates = Vec::new();

        let user_challenges: Vec<Uuid> = self
            .active_challenges
            .values()
            .filter(|c| c.participants.contains(&user_id) && c.status == ChallengeStatus::Active)
            .map(|c| c.id)
            .collect();

        for challenge_id in user_challenges {
            let new_progress = if let Some(challenge) = self.active_challenges.get(&challenge_id) {
                self.calculate_new_progress(challenge, session_data)
            } else {
                continue;
            };

            if let Some(challenge) = self.active_challenges.get_mut(&challenge_id) {
                let old_progress = challenge.current_progress;
                challenge.current_progress = new_progress;

                if challenge.current_progress >= challenge.target_value {
                    challenge.status = ChallengeStatus::Completed;
                    updates.push(ChallengeUpdate {
                        challenge_id,
                        update_type: UpdateType::Completed,
                        old_progress,
                        new_progress: challenge.current_progress,
                        rewards_earned: challenge.rewards.clone(),
                    });
                } else {
                    updates.push(ChallengeUpdate {
                        challenge_id,
                        update_type: UpdateType::Progress,
                        old_progress,
                        new_progress: challenge.current_progress,
                        rewards_earned: Vec::new(),
                    });
                }
            }
        }

        updates
    }

    /// Get user's active challenges
    pub fn get_user_challenges(&self, user_id: Uuid) -> Vec<&Challenge> {
        self.active_challenges
            .values()
            .filter(|c| c.participants.contains(&user_id))
            .collect()
    }

    /// Create event challenge
    pub fn create_event_challenge(&mut self, config: EventChallengeConfig) -> Uuid {
        let event_challenge = EventChallenge {
            id: Uuid::new_v4(),
            name: config.name,
            description: config.description,
            event_type: config.event_type,
            start_time: config.start_time,
            end_time: config.end_time,
            target_metrics: config.target_metrics,
            rewards: config.rewards,
            participants: Vec::new(),
            leaderboard: Vec::new(),
            status: EventStatus::Pending,
        };

        let id = event_challenge.id;
        self.event_challenges.push(event_challenge);
        id
    }

    /// Helper methods
    fn calculate_user_level(&self, user_progress: &UserProgress) -> u32 {
        // Simplified level calculation
        (user_progress.training_stats.total_sessions / 10 + 1).min(50) as u32
    }

    fn identify_weak_areas(&self, user_progress: &UserProgress) -> Vec<FocusArea> {
        user_progress
            .skill_breakdown
            .iter()
            .filter(|(_, &score)| score < 0.7)
            .map(|(area, _)| area.clone())
            .collect()
    }

    fn check_cooldown(&self, user_id: Uuid, template_id: &str) -> bool {
        // Check if user has completed this challenge recently
        if let Some(progress_list) = self.user_progress.get(&user_id) {
            progress_list
                .iter()
                .filter(|p| p.template_id == template_id && p.status == ChallengeStatus::Completed)
                .all(|p| Utc::now() - p.completed_at.unwrap() > chrono::Duration::days(3))
        } else {
            true
        }
    }

    fn check_requirements(
        &self,
        user_progress: &UserProgress,
        requirements: &ChallengeRequirements,
    ) -> bool {
        let user_level = self.calculate_user_level(user_progress);
        user_level >= requirements.min_level
        // In a real implementation, you'd also check required achievements
    }

    fn calculate_target_value(
        &self,
        template: &ChallengeTemplate,
        user_progress: &UserProgress,
    ) -> f32 {
        match &template.difficulty_scaling {
            DifficultyScaling::Progressive {
                base_target,
                increment,
            } => {
                let user_level = self.calculate_user_level(user_progress) as f32;
                base_target + (user_level - 1.0) * increment
            }
            DifficultyScaling::Linear {
                min_target,
                max_target,
            } => {
                let user_level = self.calculate_user_level(user_progress) as f32;
                let ratio = (user_level - 1.0) / 49.0; // Assuming max level 50
                *min_target as f32 + ratio * (*max_target - *min_target) as f32
            }
            DifficultyScaling::Adaptive => {
                // Base on user's historical performance
                match template.challenge_type {
                    ChallengeType::SkillImprovement => {
                        user_progress.average_scores.average_pronunciation + 0.1
                    }
                    ChallengeType::Performance => 5.0, // 5 sessions
                    ChallengeType::Consistency => 7.0, // 7 days
                    ChallengeType::Social => 3.0,      // 3 collaborative sessions
                }
            }
        }
    }

    fn calculate_difficulty(
        &self,
        template: &ChallengeTemplate,
        user_progress: &UserProgress,
    ) -> ChallengeDifficulty {
        let user_level = self.calculate_user_level(user_progress);

        match user_level {
            1..=10 => ChallengeDifficulty::Beginner,
            11..=25 => ChallengeDifficulty::Intermediate,
            26..=40 => ChallengeDifficulty::Advanced,
            _ => ChallengeDifficulty::Expert,
        }
    }

    fn personalize_name(&self, name: &str, target_value: f32) -> String {
        name.replace("{target_accuracy}", &format!("{:.0}", target_value * 100.0))
            .replace("{target_days}", &format!("{:.0}", target_value))
            .replace("{target_sessions}", &format!("{:.0}", target_value))
    }

    fn personalize_description(&self, description: &str, target_value: f32) -> String {
        description
            .replace("{target_accuracy}", &format!("{:.0}", target_value * 100.0))
            .replace("{target_days}", &format!("{:.0}", target_value))
            .replace("{target_sessions}", &format!("{:.0}", target_value))
            .replace("{time_limit}", "120")
    }

    fn calculate_new_progress(&self, challenge: &Challenge, session_data: &SessionData) -> f32 {
        match challenge.challenge_type {
            ChallengeType::SkillImprovement => {
                if challenge.focus_areas.contains(&FocusArea::Pronunciation) {
                    session_data.pronunciation_accuracy
                } else {
                    session_data.overall_score
                }
            }
            ChallengeType::Performance => {
                if session_data.duration <= 120.0 {
                    challenge.current_progress + 1.0
                } else {
                    challenge.current_progress
                }
            }
            ChallengeType::Consistency => {
                // This would track consecutive days in a real implementation
                challenge.current_progress + 1.0
            }
            ChallengeType::Social => {
                if session_data.was_collaborative {
                    challenge.current_progress + 1.0
                } else {
                    challenge.current_progress
                }
            }
        }
    }
}

impl Default for ChallengeSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Challenge definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Challenge {
    /// Unique challenge identifier
    pub id: Uuid,
    /// Template this challenge was created from
    pub template_id: String,
    /// Challenge display name
    pub name: String,
    /// Challenge description
    pub description: String,
    /// Type of challenge
    pub challenge_type: ChallengeType,
    /// Areas this challenge focuses on
    pub focus_areas: Vec<FocusArea>,
    /// Target value to achieve
    pub target_value: f32,
    /// Current progress towards target
    pub current_progress: f32,
    /// Difficulty level of challenge
    pub difficulty: ChallengeDifficulty,
    /// Current status
    pub status: ChallengeStatus,
    /// When challenge was created
    pub created_at: DateTime<Utc>,
    /// When challenge expires
    pub expires_at: DateTime<Utc>,
    /// Rewards for completing challenge
    pub rewards: Vec<ChallengeReward>,
    /// Users participating in challenge
    pub participants: Vec<Uuid>,
}

/// Challenge template
#[derive(Debug, Clone)]
pub struct ChallengeTemplate {
    /// Template identifier
    pub id: String,
    /// Template display name
    pub name: String,
    /// Template description
    pub description: String,
    /// Type of challenge
    pub challenge_type: ChallengeType,
    /// Focus areas for this challenge
    pub focus_areas: Vec<FocusArea>,
    /// Difficulty scaling strategy
    pub difficulty_scaling: DifficultyScaling,
    /// Challenge duration
    pub duration: chrono::Duration,
    /// Rewards for completion
    pub rewards: Vec<ChallengeReward>,
    /// Requirements to unlock challenge
    pub requirements: ChallengeRequirements,
}

/// Challenge types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChallengeType {
    /// Focus on improving specific skills
    SkillImprovement,
    /// Focus on performance metrics
    Performance,
    /// Focus on consistent practice
    Consistency,
    /// Focus on social interaction
    Social,
}

/// Difficulty levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChallengeDifficulty {
    /// Beginner level challenge
    Beginner,
    /// Intermediate level challenge
    Intermediate,
    /// Advanced level challenge
    Advanced,
    /// Expert level challenge
    Expert,
}

/// Challenge status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChallengeStatus {
    /// Challenge is currently active
    Active,
    /// Challenge has been completed
    Completed,
    /// Challenge has failed
    Failed,
    /// Challenge has expired
    Expired,
}

/// Difficulty scaling strategies
#[derive(Debug, Clone)]
pub enum DifficultyScaling {
    /// Progressive scaling with base target and increment
    Progressive {
        /// Base target value
        base_target: f32,
        /// Increment per level
        increment: f32
    },
    /// Linear scaling between min and max
    Linear {
        /// Minimum target
        min_target: u32,
        /// Maximum target
        max_target: u32
    },
    /// Adaptive scaling based on user performance
    Adaptive,
}

/// Challenge requirements
#[derive(Debug, Clone)]
pub struct ChallengeRequirements {
    /// Minimum user level required
    pub min_level: u32,
    /// Required achievement IDs
    pub required_achievements: Vec<String>,
    /// Cooldown period before challenge can be taken again
    pub cooldown: Option<chrono::Duration>,
}

/// Challenge rewards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeReward {
    /// Points reward
    Points {
        /// Currency type
        currency: String,
        /// Amount of currency
        amount: u32
    },
    /// Badge reward
    Badge(String),
    /// Title reward
    Title(String),
    /// Item reward
    Item(String),
}

/// User challenge progress
#[derive(Debug, Clone)]
pub struct UserChallengeProgress {
    /// Challenge identifier
    pub challenge_id: Uuid,
    /// Template this challenge is based on
    pub template_id: String,
    /// Current progress value
    pub current_progress: f32,
    /// Target value to achieve
    pub target_value: f32,
    /// Current challenge status
    pub status: ChallengeStatus,
    /// When challenge was started
    pub started_at: DateTime<Utc>,
    /// When challenge was completed
    pub completed_at: Option<DateTime<Utc>>,
}

/// Session data for progress calculation
#[derive(Debug, Clone)]
pub struct SessionData {
    /// Pronunciation accuracy score
    pub pronunciation_accuracy: f32,
    /// Overall session score
    pub overall_score: f32,
    /// Session duration in seconds
    pub duration: f32,
    /// Whether session was collaborative
    pub was_collaborative: bool,
    /// Focus areas covered in session
    pub focus_areas: Vec<FocusArea>,
}

/// Challenge update result
#[derive(Debug, Clone)]
pub struct ChallengeUpdate {
    /// Challenge that was updated
    pub challenge_id: Uuid,
    /// Type of update
    pub update_type: UpdateType,
    /// Previous progress value
    pub old_progress: f32,
    /// New progress value
    pub new_progress: f32,
    /// Rewards earned from this update
    pub rewards_earned: Vec<ChallengeReward>,
}

/// Update types
#[derive(Debug, Clone, PartialEq)]
pub enum UpdateType {
    /// Progress update
    Progress,
    /// Challenge completed
    Completed,
    /// Challenge failed
    Failed,
}

/// Event challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventChallenge {
    /// Unique event identifier
    pub id: Uuid,
    /// Event name
    pub name: String,
    /// Event description
    pub description: String,
    /// Type of event
    pub event_type: EventType,
    /// Event start time
    pub start_time: DateTime<Utc>,
    /// Event end time
    pub end_time: DateTime<Utc>,
    /// Target metrics for event
    pub target_metrics: HashMap<String, f32>,
    /// Rewards for event completion
    pub rewards: Vec<ChallengeReward>,
    /// Participating users
    pub participants: Vec<Uuid>,
    /// Event leaderboard
    pub leaderboard: Vec<LeaderboardEntry>,
    /// Current event status
    pub status: EventStatus,
}

/// Event challenge configuration
#[derive(Debug, Clone)]
pub struct EventChallengeConfig {
    /// Event name
    pub name: String,
    /// Event description
    pub description: String,
    /// Type of event
    pub event_type: EventType,
    /// Event start time
    pub start_time: DateTime<Utc>,
    /// Event end time
    pub end_time: DateTime<Utc>,
    /// Target metrics for event
    pub target_metrics: HashMap<String, f32>,
    /// Event rewards
    pub rewards: Vec<ChallengeReward>,
}

/// Event types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventType {
    /// Weekend event
    Weekend,
    /// Seasonal event
    Seasonal,
    /// Community event
    Community,
    /// Special event
    Special,
}

/// Event status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventStatus {
    /// Event is pending
    Pending,
    /// Event is currently active
    Active,
    /// Event has been completed
    Completed,
    /// Event has been cancelled
    Cancelled,
}

/// Leaderboard entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    /// User identifier
    pub user_id: Uuid,
    /// User display name
    pub user_name: String,
    /// User score
    pub score: f32,
    /// User rank
    pub rank: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_challenge_system_creation() {
        let system = ChallengeSystem::new();
        assert!(!system.templates.is_empty());
    }

    #[test]
    fn test_personalized_challenge_generation() {
        let mut system = ChallengeSystem::new();
        let user_id = Uuid::new_v4();

        let user_progress = UserProgress {
            user_id: user_id.to_string(),
            overall_skill_level: 0.8,
            skill_breakdown: HashMap::new(),
            progress_history: Vec::new(),
            achievements: Vec::new(),
            training_stats: crate::traits::TrainingStatistics {
                total_sessions: 15,
                successful_sessions: 12,
                total_training_time: std::time::Duration::from_secs(4500),
                exercises_completed: 75,
                success_rate: 0.8,
                average_improvement: 0.05,
                current_streak: 3,
                longest_streak: 5,
            },
            goals: Vec::new(),
            last_updated: chrono::Utc::now(),
            average_scores: crate::traits::SessionScores::default(),
            skill_levels: HashMap::new(),
            recent_sessions: Vec::new(),
            personal_bests: HashMap::new(),
            session_count: 15,
            total_practice_time: std::time::Duration::from_secs(4500),
        };

        let challenge = system.generate_personalized_challenge(user_id, &user_progress);
        assert!(challenge.is_some());

        let challenge = challenge.unwrap();
        assert!(challenge.participants.contains(&user_id));
        assert_eq!(challenge.status, ChallengeStatus::Active);
    }

    #[test]
    fn test_challenge_progress_update() {
        let mut system = ChallengeSystem::new();
        let user_id = Uuid::new_v4();

        // Create a challenge manually for testing
        let challenge = Challenge {
            id: Uuid::new_v4(),
            template_id: "pronunciation_accuracy".to_string(),
            name: "Test Challenge".to_string(),
            description: "Test challenge".to_string(),
            challenge_type: ChallengeType::SkillImprovement,
            focus_areas: vec![FocusArea::Pronunciation],
            target_value: 0.9,
            current_progress: 0.0,
            difficulty: ChallengeDifficulty::Beginner,
            status: ChallengeStatus::Active,
            created_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::days(7),
            rewards: Vec::new(),
            participants: vec![user_id],
        };

        let challenge_id = challenge.id;
        system.active_challenges.insert(challenge_id, challenge);

        let session_data = SessionData {
            pronunciation_accuracy: 0.95,
            overall_score: 0.9,
            duration: 60.0,
            was_collaborative: false,
            focus_areas: vec![FocusArea::Pronunciation],
        };

        let updates = system.update_progress(user_id, &session_data);
        assert!(!updates.is_empty());

        let update = &updates[0];
        assert_eq!(update.challenge_id, challenge_id);
        assert_eq!(update.update_type, UpdateType::Completed);
    }
}
