//! Leaderboard system with rankings and competitive elements
//!
//! This module provides comprehensive leaderboard features including:
//! - Multiple leaderboard types (global, regional, skill-based)
//! - Ranking algorithms and tier systems
//! - Seasonal competitions and time-based rankings
//! - Privacy controls and anonymous participation options

use crate::traits::{FocusArea, UserProgress};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

/// Leaderboard system manager
#[derive(Debug, Clone)]
pub struct LeaderboardSystem {
    /// Active leaderboards
    leaderboards: HashMap<Uuid, Leaderboard>,
    /// User rankings cache
    user_rankings: HashMap<Uuid, Vec<UserRanking>>,
    /// Seasonal competitions
    seasonal_competitions: Vec<SeasonalCompetition>,
    /// Tier definitions
    tier_system: TierSystem,
}

impl LeaderboardSystem {
    /// Create new leaderboard system
    pub fn new() -> Self {
        let mut system = Self {
            leaderboards: HashMap::new(),
            user_rankings: HashMap::new(),
            seasonal_competitions: Vec::new(),
            tier_system: TierSystem::default(),
        };
        system.initialize_default_leaderboards();
        system
    }

    /// Initialize default leaderboards
    fn initialize_default_leaderboards(&mut self) {
        // Global leaderboards
        self.create_leaderboard(LeaderboardConfig {
            name: "Global Experience Leaders".to_string(),
            description: "Top users by total experience points".to_string(),
            leaderboard_type: LeaderboardType::Global,
            metric: LeaderboardMetric::TotalPoints,
            time_period: TimePeriod::AllTime,
            max_entries: 100,
            update_frequency: UpdateFrequency::RealTime,
            privacy_level: PrivacyLevel::Public,
            focus_area: None,
            region: None,
        });

        self.create_leaderboard(LeaderboardConfig {
            name: "Weekly Accuracy Champions".to_string(),
            description: "Best pronunciation accuracy this week".to_string(),
            leaderboard_type: LeaderboardType::Weekly,
            metric: LeaderboardMetric::AverageAccuracy,
            time_period: TimePeriod::Week,
            max_entries: 50,
            update_frequency: UpdateFrequency::Daily,
            privacy_level: PrivacyLevel::Public,
            focus_area: Some(FocusArea::Pronunciation),
            region: None,
        });

        self.create_leaderboard(LeaderboardConfig {
            name: "Consistency Streaks".to_string(),
            description: "Longest practice streaks".to_string(),
            leaderboard_type: LeaderboardType::Global,
            metric: LeaderboardMetric::StreakDays,
            time_period: TimePeriod::AllTime,
            max_entries: 25,
            update_frequency: UpdateFrequency::Daily,
            privacy_level: PrivacyLevel::Public,
            focus_area: None,
            region: None,
        });

        self.create_leaderboard(LeaderboardConfig {
            name: "Monthly Session Leaders".to_string(),
            description: "Most practice sessions this month".to_string(),
            leaderboard_type: LeaderboardType::Monthly,
            metric: LeaderboardMetric::SessionCount,
            time_period: TimePeriod::Month,
            max_entries: 30,
            update_frequency: UpdateFrequency::Hourly,
            privacy_level: PrivacyLevel::Public,
            focus_area: None,
            region: None,
        });

        self.create_leaderboard(LeaderboardConfig {
            name: "Rising Stars".to_string(),
            description: "Users with biggest improvements this week".to_string(),
            leaderboard_type: LeaderboardType::Weekly,
            metric: LeaderboardMetric::ImprovementRate,
            time_period: TimePeriod::Week,
            max_entries: 20,
            update_frequency: UpdateFrequency::Daily,
            privacy_level: PrivacyLevel::Public,
            focus_area: None,
            region: None,
        });
    }

    /// Create new leaderboard
    pub fn create_leaderboard(&mut self, config: LeaderboardConfig) -> Uuid {
        let leaderboard = Leaderboard {
            id: Uuid::new_v4(),
            name: config.name,
            description: config.description,
            leaderboard_type: config.leaderboard_type,
            metric: config.metric,
            time_period: config.time_period,
            entries: Vec::new(),
            max_entries: config.max_entries,
            created_at: Utc::now(),
            last_updated: Utc::now(),
            update_frequency: config.update_frequency,
            privacy_level: config.privacy_level,
            focus_area: config.focus_area,
            region: config.region,
            is_active: true,
        };

        let id = leaderboard.id;
        self.leaderboards.insert(id, leaderboard);
        id
    }

    /// Update user ranking in leaderboards
    pub fn update_user_ranking(
        &mut self,
        user_id: Uuid,
        user_progress: &UserProgress,
    ) -> Vec<RankingUpdate> {
        let mut updates = Vec::new();

        for leaderboard in self.leaderboards.values_mut() {
            if !leaderboard.is_active {
                continue;
            }

            let current_score =
                Self::calculate_leaderboard_score_static(leaderboard, user_progress);
            let old_rank = leaderboard
                .entries
                .iter()
                .position(|e| e.user_id == user_id);

            // Update or add entry
            if let Some(pos) = leaderboard
                .entries
                .iter_mut()
                .position(|e| e.user_id == user_id)
            {
                let old_score = leaderboard.entries[pos].score;
                leaderboard.entries[pos].score = current_score;
                leaderboard.entries[pos].last_updated = Utc::now();

                if (current_score - old_score).abs() > 0.001 {
                    Self::resort_leaderboard_static(leaderboard);
                    let new_rank = leaderboard
                        .entries
                        .iter()
                        .position(|e| e.user_id == user_id);

                    updates.push(RankingUpdate {
                        leaderboard_id: leaderboard.id,
                        leaderboard_name: leaderboard.name.clone(),
                        old_rank: old_rank.map(|r| r + 1),
                        new_rank: new_rank.map(|r| r + 1),
                        score_change: current_score - old_score,
                        tier_change: Self::calculate_tier_change_static(old_rank, new_rank),
                    });
                }
            } else {
                // New entry
                let entry = LeaderboardEntry {
                    user_id,
                    user_name: format!("User_{}", user_id.simple()), // In real app, get actual name
                    score: current_score,
                    rank: 0, // Will be set during resort
                    tier: LeaderboardTier::Bronze,
                    achievement_count: user_progress.training_stats.total_sessions as u32, // Simplified
                    last_updated: Utc::now(),
                    is_anonymous: false,
                };

                leaderboard.entries.push(entry);
                Self::resort_leaderboard_static(leaderboard);

                let new_rank = leaderboard
                    .entries
                    .iter()
                    .position(|e| e.user_id == user_id);
                updates.push(RankingUpdate {
                    leaderboard_id: leaderboard.id,
                    leaderboard_name: leaderboard.name.clone(),
                    old_rank: None,
                    new_rank: new_rank.map(|r| r + 1),
                    score_change: current_score,
                    tier_change: None,
                });
            }

            leaderboard.last_updated = Utc::now();
        }

        // Update user rankings cache
        let user_ranks: Vec<UserRanking> = self
            .leaderboards
            .values()
            .filter_map(|lb| {
                lb.entries
                    .iter()
                    .position(|e| e.user_id == user_id)
                    .map(|pos| UserRanking {
                        leaderboard_id: lb.id,
                        leaderboard_name: lb.name.clone(),
                        rank: pos + 1,
                        score: lb.entries[pos].score,
                        tier: lb.entries[pos].tier,
                        total_participants: lb.entries.len(),
                        percentile: self.calculate_percentile(pos + 1, lb.entries.len()),
                    })
            })
            .collect();

        self.user_rankings.insert(user_id, user_ranks);

        updates
    }

    /// Get leaderboard
    pub fn get_leaderboard(
        &self,
        leaderboard_id: Uuid,
        start_rank: Option<usize>,
        limit: Option<usize>,
    ) -> Option<LeaderboardView> {
        let leaderboard = self.leaderboards.get(&leaderboard_id)?;

        let start = start_rank.unwrap_or(1).saturating_sub(1);
        let end = if let Some(limit) = limit {
            (start + limit).min(leaderboard.entries.len())
        } else {
            leaderboard.entries.len()
        };

        let entries = leaderboard.entries[start..end].to_vec();

        Some(LeaderboardView {
            id: leaderboard.id,
            name: leaderboard.name.clone(),
            description: leaderboard.description.clone(),
            metric: leaderboard.metric,
            time_period: leaderboard.time_period,
            entries,
            total_entries: leaderboard.entries.len(),
            last_updated: leaderboard.last_updated,
            focus_area: leaderboard.focus_area.clone(),
        })
    }

    /// Get user's rankings across all leaderboards
    pub fn get_user_rankings(&self, user_id: Uuid) -> Vec<UserRanking> {
        self.user_rankings
            .get(&user_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get user's position in specific leaderboard
    pub fn get_user_position(&self, user_id: Uuid, leaderboard_id: Uuid) -> Option<UserPosition> {
        let leaderboard = self.leaderboards.get(&leaderboard_id)?;
        let position = leaderboard
            .entries
            .iter()
            .position(|e| e.user_id == user_id)?;
        let entry = &leaderboard.entries[position];

        let users_above = position;
        let users_below = leaderboard.entries.len() - position - 1;

        Some(UserPosition {
            rank: position + 1,
            score: entry.score,
            tier: entry.tier,
            users_above,
            users_below,
            percentile: self.calculate_percentile(position + 1, leaderboard.entries.len()),
            points_to_next_rank: if position > 0 {
                Some(leaderboard.entries[position - 1].score - entry.score)
            } else {
                None
            },
            points_from_previous_rank: if position < leaderboard.entries.len() - 1 {
                Some(entry.score - leaderboard.entries[position + 1].score)
            } else {
                None
            },
        })
    }

    /// Start seasonal competition
    pub fn start_seasonal_competition(&mut self, config: SeasonalCompetitionConfig) -> Uuid {
        let competition = SeasonalCompetition {
            id: Uuid::new_v4(),
            name: config.name,
            description: config.description,
            season_type: config.season_type,
            start_date: config.start_date,
            end_date: config.end_date,
            leaderboard_ids: Vec::new(),
            rewards: config.rewards,
            participants: Vec::new(),
            status: CompetitionStatus::Active,
        };

        let id = competition.id;
        self.seasonal_competitions.push(competition);
        id
    }

    /// Get available leaderboards
    pub fn get_available_leaderboards(&self) -> Vec<LeaderboardSummary> {
        self.leaderboards
            .values()
            .filter(|lb| lb.is_active)
            .map(|lb| LeaderboardSummary {
                id: lb.id,
                name: lb.name.clone(),
                description: lb.description.clone(),
                leaderboard_type: lb.leaderboard_type,
                metric: lb.metric,
                time_period: lb.time_period,
                participant_count: lb.entries.len(),
                last_updated: lb.last_updated,
                focus_area: lb.focus_area.clone(),
            })
            .collect()
    }

    /// Calculate leaderboard score based on metric
    fn calculate_leaderboard_score(
        &self,
        leaderboard: &Leaderboard,
        user_progress: &UserProgress,
    ) -> f32 {
        match leaderboard.metric {
            LeaderboardMetric::TotalPoints => user_progress.overall_skill_level * 100.0,
            LeaderboardMetric::AverageAccuracy => {
                user_progress.average_scores.average_pronunciation
            }
            LeaderboardMetric::StreakDays => user_progress.training_stats.current_streak as f32,
            LeaderboardMetric::SessionCount => user_progress.training_stats.total_sessions as f32,
            LeaderboardMetric::ImprovementRate => {
                // Calculate improvement rate based on trend data
                if user_progress.training_stats.total_sessions > 5 {
                    // Use the improvement trend from average scores
                    let base_improvement = user_progress.average_scores.improvement_trend * 100.0;

                    // Factor in overall training statistics improvement
                    let training_improvement =
                        user_progress.training_stats.average_improvement * 100.0;

                    // Weighted combination: 70% from score trends, 30% from training stats
                    (base_improvement * 0.7 + training_improvement * 0.3).max(0.0)
                } else {
                    // For users with few sessions, use basic improvement metric
                    user_progress.training_stats.average_improvement * 100.0
                }
            }
            LeaderboardMetric::CompletionRate => {
                // Simplified completion rate
                if user_progress.training_stats.total_sessions > 0 {
                    user_progress.training_stats.success_rate
                } else {
                    0.0
                }
            }
        }
    }

    /// Resort leaderboard entries
    fn resort_leaderboard(&mut self, leaderboard: &mut Leaderboard) {
        leaderboard.entries.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update ranks and tiers
        let total_entries = leaderboard.entries.len();
        for (index, entry) in leaderboard.entries.iter_mut().enumerate() {
            entry.rank = index + 1;
            entry.tier = self.calculate_tier(index + 1, total_entries);
        }

        // Trim to max entries
        if leaderboard.entries.len() > leaderboard.max_entries {
            leaderboard.entries.truncate(leaderboard.max_entries);
        }
    }

    /// Calculate tier based on rank
    fn calculate_tier(&self, rank: usize, total_participants: usize) -> LeaderboardTier {
        let percentile = self.calculate_percentile(rank, total_participants);

        if percentile >= 95.0 {
            LeaderboardTier::Diamond
        } else if percentile >= 85.0 {
            LeaderboardTier::Platinum
        } else if percentile >= 70.0 {
            LeaderboardTier::Gold
        } else if percentile >= 50.0 {
            LeaderboardTier::Silver
        } else {
            LeaderboardTier::Bronze
        }
    }

    /// Calculate percentile
    fn calculate_percentile(&self, rank: usize, total_participants: usize) -> f32 {
        if total_participants <= 1 {
            return 100.0;
        }

        let percentage =
            ((total_participants - rank) as f32 / (total_participants - 1) as f32) * 100.0;
        percentage.max(0.0).min(100.0)
    }

    /// Calculate tier change
    fn calculate_tier_change(
        &self,
        old_rank: Option<usize>,
        new_rank: Option<usize>,
    ) -> Option<TierChange> {
        match (old_rank, new_rank) {
            (Some(old), Some(new)) if old != new => {
                if new < old {
                    Some(TierChange::Promoted)
                } else {
                    Some(TierChange::Demoted)
                }
            }
            (None, Some(_)) => Some(TierChange::Entered),
            (Some(_), None) => Some(TierChange::Dropped),
            _ => None,
        }
    }

    // Static versions of methods to avoid borrowing conflicts
    fn calculate_leaderboard_score_static(
        leaderboard: &Leaderboard,
        user_progress: &UserProgress,
    ) -> f32 {
        match leaderboard.metric {
            LeaderboardMetric::TotalPoints => user_progress.overall_skill_level * 100.0,
            LeaderboardMetric::AverageAccuracy => {
                user_progress.average_scores.average_pronunciation
            }
            LeaderboardMetric::StreakDays => user_progress.training_stats.current_streak as f32,
            LeaderboardMetric::SessionCount => user_progress.training_stats.total_sessions as f32,
            LeaderboardMetric::ImprovementRate => {
                // Calculate improvement rate based on trend data
                if user_progress.training_stats.total_sessions > 5 {
                    // Use the improvement trend from average scores
                    let base_improvement = user_progress.average_scores.improvement_trend * 100.0;

                    // Factor in overall training statistics improvement
                    let training_improvement =
                        user_progress.training_stats.average_improvement * 100.0;

                    // Weighted combination: 70% from score trends, 30% from training stats
                    (base_improvement * 0.7 + training_improvement * 0.3).max(0.0)
                } else {
                    // For users with few sessions, use basic improvement metric
                    user_progress.training_stats.average_improvement * 100.0
                }
            }
            LeaderboardMetric::CompletionRate => {
                // Simplified completion rate
                if user_progress.training_stats.total_sessions > 0 {
                    user_progress.training_stats.success_rate
                } else {
                    0.0
                }
            }
        }
    }

    fn resort_leaderboard_static(leaderboard: &mut Leaderboard) {
        leaderboard.entries.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update ranks and tiers
        let total_entries = leaderboard.entries.len();
        for (index, entry) in leaderboard.entries.iter_mut().enumerate() {
            entry.rank = index + 1;
            entry.tier = Self::calculate_tier_static(index + 1, total_entries);
        }

        // Trim to max entries
        if leaderboard.entries.len() > leaderboard.max_entries {
            leaderboard.entries.truncate(leaderboard.max_entries);
        }
    }

    fn calculate_tier_static(rank: usize, total_entries: usize) -> LeaderboardTier {
        let percentage = rank as f32 / total_entries as f32;
        if percentage <= 0.1 {
            LeaderboardTier::Diamond
        } else if percentage <= 0.25 {
            LeaderboardTier::Gold
        } else if percentage <= 0.5 {
            LeaderboardTier::Silver
        } else {
            LeaderboardTier::Bronze
        }
    }

    fn calculate_tier_change_static(
        old_rank: Option<usize>,
        new_rank: Option<usize>,
    ) -> Option<TierChange> {
        match (old_rank, new_rank) {
            (Some(old), Some(new)) if old != new => {
                if new < old {
                    Some(TierChange::Promoted)
                } else {
                    Some(TierChange::Demoted)
                }
            }
            (None, Some(_)) => Some(TierChange::Entered),
            (Some(_), None) => Some(TierChange::Dropped),
            _ => None,
        }
    }
}

impl Default for LeaderboardSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Leaderboard definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Leaderboard {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub leaderboard_type: LeaderboardType,
    pub metric: LeaderboardMetric,
    pub time_period: TimePeriod,
    pub entries: Vec<LeaderboardEntry>,
    pub max_entries: usize,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub update_frequency: UpdateFrequency,
    pub privacy_level: PrivacyLevel,
    pub focus_area: Option<FocusArea>,
    pub region: Option<String>,
    pub is_active: bool,
}

/// Leaderboard configuration
#[derive(Debug, Clone)]
pub struct LeaderboardConfig {
    pub name: String,
    pub description: String,
    pub leaderboard_type: LeaderboardType,
    pub metric: LeaderboardMetric,
    pub time_period: TimePeriod,
    pub max_entries: usize,
    pub update_frequency: UpdateFrequency,
    pub privacy_level: PrivacyLevel,
    pub focus_area: Option<FocusArea>,
    pub region: Option<String>,
}

/// Leaderboard types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LeaderboardType {
    Global,
    Regional,
    Weekly,
    Monthly,
    Seasonal,
    FocusArea,
}

/// Leaderboard metrics
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LeaderboardMetric {
    TotalPoints,
    AverageAccuracy,
    StreakDays,
    SessionCount,
    ImprovementRate,
    CompletionRate,
}

/// Time periods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TimePeriod {
    AllTime,
    Year,
    Month,
    Week,
    Day,
}

/// Update frequencies
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum UpdateFrequency {
    RealTime,
    Hourly,
    Daily,
    Weekly,
}

/// Privacy levels
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PrivacyLevel {
    Public,
    FriendsOnly,
    Anonymous,
}

/// Leaderboard entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub user_id: Uuid,
    pub user_name: String,
    pub score: f32,
    pub rank: usize,
    pub tier: LeaderboardTier,
    pub achievement_count: u32,
    pub last_updated: DateTime<Utc>,
    pub is_anonymous: bool,
}

/// Leaderboard tiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LeaderboardTier {
    Bronze,
    Silver,
    Gold,
    Platinum,
    Diamond,
}

/// Leaderboard view for clients
#[derive(Debug, Clone)]
pub struct LeaderboardView {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub metric: LeaderboardMetric,
    pub time_period: TimePeriod,
    pub entries: Vec<LeaderboardEntry>,
    pub total_entries: usize,
    pub last_updated: DateTime<Utc>,
    pub focus_area: Option<FocusArea>,
}

/// User ranking information
#[derive(Debug, Clone)]
pub struct UserRanking {
    pub leaderboard_id: Uuid,
    pub leaderboard_name: String,
    pub rank: usize,
    pub score: f32,
    pub tier: LeaderboardTier,
    pub total_participants: usize,
    pub percentile: f32,
}

/// User position details
#[derive(Debug, Clone)]
pub struct UserPosition {
    pub rank: usize,
    pub score: f32,
    pub tier: LeaderboardTier,
    pub users_above: usize,
    pub users_below: usize,
    pub percentile: f32,
    pub points_to_next_rank: Option<f32>,
    pub points_from_previous_rank: Option<f32>,
}

/// Ranking update notification
#[derive(Debug, Clone)]
pub struct RankingUpdate {
    pub leaderboard_id: Uuid,
    pub leaderboard_name: String,
    pub old_rank: Option<usize>,
    pub new_rank: Option<usize>,
    pub score_change: f32,
    pub tier_change: Option<TierChange>,
}

/// Tier change types
#[derive(Debug, Clone, PartialEq)]
pub enum TierChange {
    Promoted,
    Demoted,
    Entered,
    Dropped,
}

/// Leaderboard summary
#[derive(Debug, Clone)]
pub struct LeaderboardSummary {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub leaderboard_type: LeaderboardType,
    pub metric: LeaderboardMetric,
    pub time_period: TimePeriod,
    pub participant_count: usize,
    pub last_updated: DateTime<Utc>,
    pub focus_area: Option<FocusArea>,
}

/// Seasonal competition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalCompetition {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub season_type: SeasonType,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub leaderboard_ids: Vec<Uuid>,
    pub rewards: Vec<SeasonalReward>,
    pub participants: Vec<Uuid>,
    pub status: CompetitionStatus,
}

/// Seasonal competition configuration
#[derive(Debug, Clone)]
pub struct SeasonalCompetitionConfig {
    pub name: String,
    pub description: String,
    pub season_type: SeasonType,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub rewards: Vec<SeasonalReward>,
}

/// Season types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SeasonType {
    Spring,
    Summer,
    Fall,
    Winter,
    Custom(String),
}

/// Seasonal rewards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalReward {
    pub rank_range: (usize, usize),
    pub reward_type: RewardType,
    pub value: u32,
    pub description: String,
}

/// Reward types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RewardType {
    Points,
    Badge,
    Title,
    Trophy,
}

/// Competition status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompetitionStatus {
    Upcoming,
    Active,
    Ended,
    Cancelled,
}

/// Tier system configuration
#[derive(Debug, Clone)]
pub struct TierSystem {
    pub tier_thresholds: HashMap<LeaderboardTier, f32>,
    pub tier_benefits: HashMap<LeaderboardTier, Vec<String>>,
}

impl Default for TierSystem {
    fn default() -> Self {
        let mut tier_thresholds = HashMap::new();
        tier_thresholds.insert(LeaderboardTier::Bronze, 0.0);
        tier_thresholds.insert(LeaderboardTier::Silver, 50.0);
        tier_thresholds.insert(LeaderboardTier::Gold, 70.0);
        tier_thresholds.insert(LeaderboardTier::Platinum, 85.0);
        tier_thresholds.insert(LeaderboardTier::Diamond, 95.0);

        let mut tier_benefits = HashMap::new();
        tier_benefits.insert(
            LeaderboardTier::Bronze,
            vec!["Basic recognition".to_string()],
        );
        tier_benefits.insert(
            LeaderboardTier::Silver,
            vec!["Silver badge".to_string(), "Bonus points".to_string()],
        );
        tier_benefits.insert(
            LeaderboardTier::Gold,
            vec!["Gold badge".to_string(), "Premium features".to_string()],
        );
        tier_benefits.insert(
            LeaderboardTier::Platinum,
            vec![
                "Platinum badge".to_string(),
                "Exclusive content".to_string(),
            ],
        );
        tier_benefits.insert(
            LeaderboardTier::Diamond,
            vec!["Diamond badge".to_string(), "Elite status".to_string()],
        );

        Self {
            tier_thresholds,
            tier_benefits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_leaderboard_system_creation() {
        let system = LeaderboardSystem::new();
        assert!(!system.leaderboards.is_empty());
    }

    #[test]
    fn test_leaderboard_creation() {
        let mut system = LeaderboardSystem::new();

        let config = LeaderboardConfig {
            name: "Test Leaderboard".to_string(),
            description: "A test leaderboard".to_string(),
            leaderboard_type: LeaderboardType::Global,
            metric: LeaderboardMetric::TotalPoints,
            time_period: TimePeriod::AllTime,
            max_entries: 10,
            update_frequency: UpdateFrequency::RealTime,
            privacy_level: PrivacyLevel::Public,
            focus_area: None,
            region: None,
        };

        let id = system.create_leaderboard(config);
        assert!(system.leaderboards.contains_key(&id));
    }

    #[test]
    fn test_user_ranking_update() {
        let mut system = LeaderboardSystem::new();
        let user_id = Uuid::new_v4();

        let user_progress = UserProgress {
            user_id: user_id.to_string(),
            overall_skill_level: 0.85,
            skill_breakdown: HashMap::new(),
            progress_history: Vec::new(),
            achievements: Vec::new(),
            training_stats: crate::traits::TrainingStatistics {
                total_sessions: 10,
                successful_sessions: 8,
                total_training_time: std::time::Duration::from_secs(3000),
                exercises_completed: 50,
                success_rate: 0.85,
                average_improvement: 0.08,
                current_streak: 5,
                longest_streak: 7,
            },
            goals: Vec::new(),
            last_updated: chrono::Utc::now(),
            average_scores: crate::traits::SessionScores::default(),
            skill_levels: HashMap::new(),
            recent_sessions: Vec::new(),
            personal_bests: HashMap::new(),
            session_count: 10,
            total_practice_time: std::time::Duration::from_secs(3000),
        };

        let updates = system.update_user_ranking(user_id, &user_progress);
        assert!(!updates.is_empty());

        // User should now appear in leaderboards
        let rankings = system.get_user_rankings(user_id);
        assert!(!rankings.is_empty());
    }

    #[test]
    fn test_percentile_calculation() {
        let system = LeaderboardSystem::new();

        assert_eq!(system.calculate_percentile(1, 10), 100.0);
        assert!((system.calculate_percentile(5, 10) - 55.55556).abs() < 0.00001);
        assert_eq!(system.calculate_percentile(10, 10), 0.0);
    }

    #[test]
    fn test_tier_calculation() {
        let system = LeaderboardSystem::new();

        assert_eq!(system.calculate_tier(1, 100), LeaderboardTier::Diamond);
        assert_eq!(system.calculate_tier(15, 100), LeaderboardTier::Platinum);
        assert_eq!(system.calculate_tier(30, 100), LeaderboardTier::Gold);
        assert_eq!(system.calculate_tier(50, 100), LeaderboardTier::Silver);
        assert_eq!(system.calculate_tier(80, 100), LeaderboardTier::Bronze);
    }

    #[test]
    fn test_leaderboard_view() {
        let mut system = LeaderboardSystem::new();
        let user_id = Uuid::new_v4();

        // Add user to leaderboards
        let user_progress = UserProgress {
            user_id: user_id.to_string(),
            overall_skill_level: 0.85,
            skill_breakdown: HashMap::new(),
            progress_history: Vec::new(),
            achievements: Vec::new(),
            training_stats: crate::traits::TrainingStatistics {
                total_sessions: 10,
                successful_sessions: 8,
                total_training_time: std::time::Duration::from_secs(3000),
                exercises_completed: 50,
                success_rate: 0.85,
                average_improvement: 0.08,
                current_streak: 5,
                longest_streak: 7,
            },
            goals: Vec::new(),
            last_updated: chrono::Utc::now(),
            average_scores: crate::traits::SessionScores::default(),
            skill_levels: HashMap::new(),
            recent_sessions: Vec::new(),
            personal_bests: HashMap::new(),
            session_count: 10,
            total_practice_time: std::time::Duration::from_secs(3000),
        };

        system.update_user_ranking(user_id, &user_progress);

        // Get first leaderboard
        let leaderboard_id = system.leaderboards.keys().next().copied().unwrap();
        let view = system.get_leaderboard(leaderboard_id, None, None);

        assert!(view.is_some());
        let view = view.unwrap();
        assert!(!view.entries.is_empty());
    }

    #[test]
    fn test_seasonal_competition() {
        let mut system = LeaderboardSystem::new();

        let config = SeasonalCompetitionConfig {
            name: "Spring Championship".to_string(),
            description: "Spring season competition".to_string(),
            season_type: SeasonType::Spring,
            start_date: Utc::now(),
            end_date: Utc::now() + chrono::Duration::days(30),
            rewards: vec![SeasonalReward {
                rank_range: (1, 3),
                reward_type: RewardType::Trophy,
                value: 1,
                description: "Spring Trophy".to_string(),
            }],
        };

        let competition_id = system.start_seasonal_competition(config);

        assert!(system
            .seasonal_competitions
            .iter()
            .any(|c| c.id == competition_id));
    }
}
