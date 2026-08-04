//! Social features for peer interaction and community engagement
//!
//! This module provides comprehensive social features including:
//! - Peer comparison and ranking systems
//! - Collaborative challenges and group activities
//! - Mentorship matching and guidance
//! - Community forums and discussion boards
//! - Social learning networks and study groups

use crate::traits::{FocusArea, SessionState, UserProgress};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Social system manager
#[derive(Debug, Clone)]
pub struct SocialSystem {
    /// Peer groups
    peer_groups: HashMap<Uuid, PeerGroup>,
    /// Mentorship relationships
    mentorships: HashMap<Uuid, Vec<MentorshipPair>>,
    /// Community forums
    forums: HashMap<Uuid, Forum>,
    /// Study groups
    study_groups: HashMap<Uuid, StudyGroup>,
    /// Social network connections
    connections: HashMap<Uuid, Vec<SocialConnection>>,
    /// Collaborative challenges
    collaborative_challenges: HashMap<Uuid, CollaborativeChallenge>,
}

impl SocialSystem {
    /// Create a new social system
    #[must_use]
    pub fn new() -> Self {
        Self {
            peer_groups: HashMap::new(),
            mentorships: HashMap::new(),
            forums: HashMap::new(),
            study_groups: HashMap::new(),
            connections: HashMap::new(),
            collaborative_challenges: HashMap::new(),
        }
    }

    /// Create a peer group
    pub fn create_peer_group(&mut self, creator_id: Uuid, config: PeerGroupConfig) -> Uuid {
        let group_id = Uuid::new_v4();
        let peer_group = PeerGroup {
            id: group_id,
            name: config.name,
            description: config.description,
            creator_id,
            members: vec![creator_id],
            max_members: config.max_members,
            focus_areas: config.focus_areas,
            privacy_level: config.privacy_level,
            created_at: Utc::now(),
            is_active: true,
        };

        self.peer_groups.insert(group_id, peer_group);
        group_id
    }

    /// Join a peer group
    pub fn join_peer_group(&mut self, user_id: Uuid, group_id: Uuid) -> Result<(), String> {
        if let Some(group) = self.peer_groups.get_mut(&group_id) {
            if group.members.len() >= group.max_members {
                return Err("Group is full".to_string());
            }

            if group.members.contains(&user_id) {
                return Err("User already in group".to_string());
            }

            group.members.push(user_id);
            Ok(())
        } else {
            Err("Group not found".to_string())
        }
    }

    /// Get peer comparison for user.
    ///
    /// `peer_progress` supplies each peer's *real*, caller-fetched
    /// [`UserProgress`] (e.g. loaded from the persistence layer), keyed by
    /// user id. A peer present in a shared group but absent from
    /// `peer_progress` is honestly omitted from the result rather than
    /// being replaced by a fabricated stand-in -- if none of the user's
    /// group peers have real data available, this returns an empty
    /// `Vec` rather than synthesizing comparisons.
    #[must_use]
    pub fn get_peer_comparison(
        &self,
        user_id: Uuid,
        user_progress: &UserProgress,
        peer_progress: &HashMap<Uuid, UserProgress>,
    ) -> Vec<PeerComparison> {
        let mut comparisons = Vec::new();

        // Find all groups user belongs to
        for group in self.peer_groups.values() {
            if !group.members.contains(&user_id) {
                continue;
            }
            for &peer_id in &group.members {
                if peer_id == user_id {
                    continue;
                }
                // Only compare against peers we have real, caller-supplied
                // progress data for -- never a fabricated stand-in.
                let Some(peer) = peer_progress.get(&peer_id) else {
                    continue;
                };

                let peer_comparison = PeerComparison {
                    peer_id,
                    peer_name: Self::display_name(&peer.user_id, peer_id, "User"),
                    user_rank: Self::rank_in_group(
                        group,
                        user_id,
                        user_id,
                        user_progress,
                        peer_progress,
                    ),
                    peer_rank: Self::rank_in_group(
                        group,
                        peer_id,
                        user_id,
                        user_progress,
                        peer_progress,
                    ),
                    metrics: self.compare_metrics(user_progress, peer),
                    improvement_suggestions: Self::generate_improvement_suggestions(user_progress),
                };
                comparisons.push(peer_comparison);
            }
        }

        comparisons
    }

    /// Create a collaborative challenge.
    ///
    /// The created [`CollaborativeChallenge`] is stored (not just its id
    /// returned) so it can actually be retrieved afterward via
    /// [`Self::get_collaborative_challenge`] -- creating a challenge is a
    /// real, persisted state change, not a discarded value behind a
    /// freshly-minted id.
    pub fn create_collaborative_challenge(
        &mut self,
        creator_id: Uuid,
        config: CollaborativeChallengeConfig,
    ) -> Uuid {
        let challenge_id = Uuid::new_v4();
        let challenge = CollaborativeChallenge {
            id: challenge_id,
            title: config.title,
            description: config.description,
            creator_id,
            participants: vec![creator_id],
            target_metrics: config.target_metrics,
            duration: config.duration,
            rewards: config.rewards,
            created_at: Utc::now(),
            starts_at: config.starts_at,
            ends_at: config.starts_at + config.duration,
            status: ChallengeStatus::Pending,
            progress: HashMap::new(),
        };

        // Add to study group if specified
        if let Some(group_id) = config.study_group_id {
            if let Some(study_group) = self.study_groups.get_mut(&group_id) {
                study_group.active_challenges.push(challenge_id);
            }
        }

        self.collaborative_challenges
            .insert(challenge_id, challenge);

        challenge_id
    }

    /// Look up a previously created collaborative challenge by id.
    #[must_use]
    pub fn get_collaborative_challenge(
        &self,
        challenge_id: Uuid,
    ) -> Option<&CollaborativeChallenge> {
        self.collaborative_challenges.get(&challenge_id)
    }

    /// Find mentorship matches.
    ///
    /// `mentor_progress` supplies each candidate mentor's *real*,
    /// caller-fetched [`UserProgress`] (specifically its `skill_breakdown`),
    /// keyed by user id. A mentor connection with no entry in
    /// `mentor_progress` is honestly skipped rather than assigned a
    /// fabricated compatibility score or expertise list.
    #[must_use]
    pub fn find_mentorship_matches(
        &self,
        mentee_id: Uuid,
        preferences: &MentorshipPreferences,
        mentor_progress: &HashMap<Uuid, UserProgress>,
    ) -> Vec<MentorshipMatch> {
        let mut matches = Vec::new();

        for connection in self.connections.get(&mentee_id).unwrap_or(&Vec::new()) {
            if let ConnectionType::Mentor = connection.connection_type {
                let Some(mentor) = mentor_progress.get(&connection.user_id) else {
                    continue;
                };

                let compatibility_score = Self::calculate_mentor_compatibility(preferences, mentor);
                if compatibility_score > 0.7 {
                    matches.push(MentorshipMatch {
                        mentor_id: connection.user_id,
                        mentor_name: Self::display_name(
                            &mentor.user_id,
                            connection.user_id,
                            "Mentor",
                        ),
                        compatibility_score,
                        shared_focus_areas: Self::shared_focus_areas(preferences, mentor),
                        mentor_expertise: Self::calculate_mentor_expertise(mentor),
                        availability: TimeSlot {
                            start_time: Utc::now(),
                            end_time: Utc::now() + chrono::Duration::hours(1),
                            recurrence: Recurrence::Weekly,
                        },
                    });
                }
            }
        }

        matches.sort_by(|a, b| b.compatibility_score.total_cmp(&a.compatibility_score));
        matches
    }

    /// Create mentorship relationship
    pub fn create_mentorship(
        &mut self,
        mentor_id: Uuid,
        mentee_id: Uuid,
        config: MentorshipConfig,
    ) -> Uuid {
        let mentorship_id = Uuid::new_v4();
        let mentorship = MentorshipPair {
            id: mentorship_id,
            mentor_id,
            mentee_id,
            focus_areas: config.focus_areas,
            meeting_schedule: config.meeting_schedule,
            goals: config.goals,
            created_at: Utc::now(),
            status: MentorshipStatus::Active,
            progress: MentorshipProgress::default(),
        };

        self.mentorships
            .entry(mentor_id)
            .or_default()
            .push(mentorship);

        mentorship_id
    }

    /// Create study group
    pub fn create_study_group(&mut self, creator_id: Uuid, config: StudyGroupConfig) -> Uuid {
        let group_id = Uuid::new_v4();
        let study_group = StudyGroup {
            id: group_id,
            name: config.name,
            description: config.description,
            creator_id,
            members: vec![creator_id],
            focus_areas: config.focus_areas,
            meeting_schedule: config.meeting_schedule,
            goals: config.goals,
            active_challenges: Vec::new(),
            created_at: Utc::now(),
            is_active: true,
            progress: StudyGroupProgress::default(),
        };

        self.study_groups.insert(group_id, study_group);
        group_id
    }

    /// Get social learning recommendations
    #[must_use]
    pub fn get_social_learning_recommendations(
        &self,
        user_id: Uuid,
        user_progress: &UserProgress,
    ) -> SocialLearningRecommendations {
        let mut recommendations = SocialLearningRecommendations {
            suggested_peer_groups: Vec::new(),
            mentor_recommendations: Vec::new(),
            study_group_suggestions: Vec::new(),
            collaborative_opportunities: Vec::new(),
        };

        // Suggest peer groups based on skill level and focus areas
        for group in self.peer_groups.values() {
            if !group.members.contains(&user_id) && group.members.len() < group.max_members {
                let compatibility = Self::calculate_group_compatibility(user_progress, group);
                if compatibility > 0.6 {
                    // Real intersection between the group's focus areas and
                    // the areas the user actually has tracked skill data
                    // for -- not a blind clone of the group's areas.
                    let shared_focus_areas: Vec<FocusArea> = group
                        .focus_areas
                        .iter()
                        .filter(|area| user_progress.skill_breakdown.contains_key(area))
                        .cloned()
                        .collect();
                    recommendations
                        .suggested_peer_groups
                        .push(PeerGroupSuggestion {
                            group_id: group.id,
                            group_name: group.name.clone(),
                            compatibility_score: compatibility,
                            shared_focus_areas,
                            member_count: group.members.len(),
                        });
                }
            }
        }

        // Sort by compatibility. `total_cmp` (rather than
        // `partial_cmp().expect(...)`) avoids a NaN panic now that
        // `compatibility_score` is a real computed value instead of a
        // hardcoded constant.
        recommendations
            .suggested_peer_groups
            .sort_by(|a, b| b.compatibility_score.total_cmp(&a.compatibility_score));

        recommendations
    }

    /// Real per-metric rank of `target_id` within `group`, computed from
    /// actual tracked progress data (average pronunciation score). Only
    /// members with real progress data available -- the querying user via
    /// `user_progress`, or other members present in `peer_progress` -- are
    /// included in the ranking; members with no known progress are neither
    /// ranked nor allowed to distort others' ranks. If `target_id` itself
    /// has no real data among the ranked members (should not normally
    /// happen), it is placed last rather than assigned a fabricated rank.
    fn rank_in_group(
        group: &PeerGroup,
        target_id: Uuid,
        user_id: Uuid,
        user_progress: &UserProgress,
        peer_progress: &HashMap<Uuid, UserProgress>,
    ) -> usize {
        let mut ranked: Vec<(Uuid, f32)> = group
            .members
            .iter()
            .filter_map(|&member_id| {
                let score = if member_id == user_id {
                    Some(user_progress.average_scores.average_pronunciation)
                } else {
                    peer_progress
                        .get(&member_id)
                        .map(|p| p.average_scores.average_pronunciation)
                };
                score.map(|s| (member_id, s))
            })
            .collect();

        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

        ranked
            .iter()
            .position(|(id, _)| *id == target_id)
            .map_or(ranked.len() + 1, |pos| pos + 1)
    }

    fn compare_metrics(
        &self,
        user_progress: &UserProgress,
        peer_progress: &UserProgress,
    ) -> Vec<MetricComparison> {
        vec![
            MetricComparison {
                metric_name: "Total Sessions".to_string(),
                user_value: user_progress.training_stats.total_sessions as f32,
                peer_value: peer_progress.training_stats.total_sessions as f32,
                user_percentile: self
                    .calculate_percentile_for_sessions(user_progress.training_stats.total_sessions),
            },
            MetricComparison {
                metric_name: "Average Accuracy".to_string(),
                user_value: user_progress.average_scores.average_pronunciation,
                peer_value: peer_progress.average_scores.average_pronunciation,
                user_percentile: self.calculate_percentile_for_accuracy(
                    user_progress.average_scores.average_pronunciation,
                ),
            },
            MetricComparison {
                metric_name: "Fluency Score".to_string(),
                user_value: user_progress.average_scores.average_fluency,
                peer_value: peer_progress.average_scores.average_fluency,
                user_percentile: self.calculate_percentile_for_accuracy(
                    user_progress.average_scores.average_fluency,
                ),
            },
            MetricComparison {
                metric_name: "Quality Score".to_string(),
                user_value: user_progress.average_scores.average_quality,
                peer_value: peer_progress.average_scores.average_quality,
                user_percentile: self.calculate_percentile_for_accuracy(
                    user_progress.average_scores.average_quality,
                ),
            },
        ]
    }

    /// Percentile of a real session count under an assumed reference
    /// population distribution (normal, mean=20, std=10 -- there is no
    /// real cross-user population data source in this crate to derive
    /// these parameters from empirically). The input is always the
    /// caller's real, measured session count; only the reference
    /// distribution against which it is scored is an assumption.
    fn calculate_percentile_for_sessions(&self, sessions: usize) -> f32 {
        let mean = 20.0;
        let std_dev = 10.0;
        let z_score = (sessions as f32 - mean) / std_dev;

        // tanh-based approximation of the normal CDF.
        let percentile = 50.0 + 30.0 * z_score.tanh();
        percentile.clamp(1.0, 99.0)
    }

    /// Percentile of a real accuracy score under an assumed reference
    /// population distribution (normal, mean=0.7, std=0.15 -- see
    /// [`Self::calculate_percentile_for_sessions`] for why these are
    /// assumed rather than empirical). The input is always the caller's
    /// real, measured score.
    fn calculate_percentile_for_accuracy(&self, score: f32) -> f32 {
        let mean = 0.7;
        let std_dev = 0.15;
        let z_score = (score - mean) / std_dev;

        // tanh-based approximation of the normal CDF.
        let percentile = 50.0 + 30.0 * z_score.tanh();
        percentile.clamp(1.0, 99.0)
    }

    /// Real improvement suggestions derived from which of the user's
    /// tracked average scores are actually below a "solid" threshold,
    /// rather than a fixed list independent of `user_progress`.
    fn generate_improvement_suggestions(user_progress: &UserProgress) -> Vec<String> {
        const SOLID_THRESHOLD: f32 = 0.7;
        const MIN_SESSIONS_FOR_CONSISTENCY: usize = 10;

        let mut suggestions = Vec::new();
        let scores = &user_progress.average_scores;

        if scores.average_pronunciation < SOLID_THRESHOLD {
            suggestions.push("Focus on pronunciation accuracy".to_string());
        }
        if scores.average_fluency < SOLID_THRESHOLD {
            suggestions.push("Practice speaking fluently with fewer pauses".to_string());
        }
        if scores.average_quality < SOLID_THRESHOLD {
            suggestions.push("Work on overall speech clarity and quality".to_string());
        }
        if user_progress.training_stats.total_sessions < MIN_SESSIONS_FOR_CONSISTENCY {
            suggestions.push("Increase practice frequency".to_string());
        }
        if suggestions.is_empty() {
            suggestions
                .push("Strong performance across the board -- try a collaborative challenge to push further".to_string());
        }

        suggestions
    }

    /// Real mentor expertise areas: the focus areas from the mentor's own
    /// tracked `skill_breakdown` where their skill level meets an
    /// "expert" threshold, ordered strongest-first (ties broken by a fixed
    /// area ordering for determinism).
    fn calculate_mentor_expertise(mentor_profile: &UserProgress) -> Vec<FocusArea> {
        const EXPERTISE_THRESHOLD: f32 = 0.75;

        let mut expertise: Vec<(FocusArea, f32)> = mentor_profile
            .skill_breakdown
            .iter()
            .filter(|&(_, &level)| level >= EXPERTISE_THRESHOLD)
            .map(|(area, &level)| (area.clone(), level))
            .collect();

        expertise.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| format!("{:?}", a.0).cmp(&format!("{:?}", b.0)))
        });

        expertise.into_iter().map(|(area, _)| area).collect()
    }

    /// The subset of the mentee's preferred focus areas that the mentor
    /// actually has tracked skill data for -- a real intersection, not a
    /// blind clone of the mentee's preferences.
    fn shared_focus_areas(
        preferences: &MentorshipPreferences,
        mentor_profile: &UserProgress,
    ) -> Vec<FocusArea> {
        preferences
            .focus_areas
            .iter()
            .filter(|area| mentor_profile.skill_breakdown.contains_key(area))
            .cloned()
            .collect()
    }

    /// Real weighted-similarity compatibility score: the mentor's average
    /// real skill level across the mentee's preferred focus areas (areas
    /// the mentor has no tracked data for count as 0.0 skill in that area).
    /// Falls back to the mentor's overall skill level when the mentee
    /// stated no focus-area preference at all.
    fn calculate_mentor_compatibility(
        preferences: &MentorshipPreferences,
        mentor_profile: &UserProgress,
    ) -> f32 {
        if preferences.focus_areas.is_empty() {
            return mentor_profile.overall_skill_level.clamp(0.0, 1.0);
        }

        let scores: Vec<f32> = preferences
            .focus_areas
            .iter()
            .map(|area| {
                mentor_profile
                    .skill_breakdown
                    .get(area)
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect();

        (scores.iter().sum::<f32>() / scores.len() as f32).clamp(0.0, 1.0)
    }

    /// Real display name for a peer/mentor: the caller-supplied real
    /// `user_id` (from their fetched [`UserProgress`]) when non-empty,
    /// falling back to a `<prefix>_<uuid>` placeholder only when no real
    /// identifier is available (e.g. a default/uninitialized profile).
    fn display_name(real_user_id: &str, id: Uuid, prefix: &str) -> String {
        if real_user_id.is_empty() {
            format!("{prefix}_{}", id.simple())
        } else {
            real_user_id.to_string()
        }
    }

    /// Real group compatibility: the fraction of the group's stated focus
    /// areas that the user actually has tracked skill data for -- a real
    /// overlap measure between the user's real progress and the group's
    /// real focus areas.
    fn calculate_group_compatibility(user_progress: &UserProgress, group: &PeerGroup) -> f32 {
        if group.focus_areas.is_empty() {
            return 0.0;
        }

        let matched = group
            .focus_areas
            .iter()
            .filter(|area| user_progress.skill_breakdown.contains_key(area))
            .count();

        matched as f32 / group.focus_areas.len() as f32
    }
}

impl Default for SocialSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Peer group for collaborative learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerGroup {
    /// Group ID
    pub id: Uuid,
    /// Group name
    pub name: String,
    /// Group description
    pub description: String,
    /// Creator ID
    pub creator_id: Uuid,
    /// Member IDs
    pub members: Vec<Uuid>,
    /// Maximum members
    pub max_members: usize,
    /// Focus areas
    pub focus_areas: Vec<FocusArea>,
    /// Privacy level
    pub privacy_level: PrivacyLevel,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Whether group is active
    pub is_active: bool,
}

/// Peer group configuration
#[derive(Debug, Clone)]
pub struct PeerGroupConfig {
    /// Group name
    pub name: String,
    /// Group description
    pub description: String,
    /// Maximum members
    pub max_members: usize,
    /// Focus areas
    pub focus_areas: Vec<FocusArea>,
    /// Privacy level
    pub privacy_level: PrivacyLevel,
}

/// Privacy levels for groups
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PrivacyLevel {
    /// Description
    Public,
    /// Description
    Private,
    /// Description
    InviteOnly,
}

/// Peer comparison result
#[derive(Debug, Clone)]
pub struct PeerComparison {
    /// Peer user ID
    pub peer_id: Uuid,
    /// Peer display name
    pub peer_name: String,
    /// User's rank in group
    pub user_rank: usize,
    /// Peer's rank in group
    pub peer_rank: usize,
    /// Metric comparisons
    pub metrics: Vec<MetricComparison>,
    /// Improvement suggestions
    pub improvement_suggestions: Vec<String>,
}

/// Metric comparison between users
#[derive(Debug, Clone)]
pub struct MetricComparison {
    /// Metric name
    pub metric_name: String,
    /// User's value
    pub user_value: f32,
    /// Peer's value
    pub peer_value: f32,
    /// User's percentile ranking
    pub user_percentile: f32,
}

/// Collaborative challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeChallenge {
    /// Challenge ID
    pub id: Uuid,
    /// Challenge title
    pub title: String,
    /// Challenge description
    pub description: String,
    /// Creator ID
    pub creator_id: Uuid,
    /// Participant IDs
    pub participants: Vec<Uuid>,
    /// Target metrics
    pub target_metrics: HashMap<String, f32>,
    /// Challenge duration
    pub duration: chrono::Duration,
    /// Rewards
    pub rewards: Vec<ChallengeReward>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Start timestamp
    pub starts_at: DateTime<Utc>,
    /// End timestamp
    pub ends_at: DateTime<Utc>,
    /// Challenge status
    pub status: ChallengeStatus,
    /// Participant progress
    pub progress: HashMap<Uuid, f32>,
}

/// Collaborative challenge configuration
#[derive(Debug, Clone)]
pub struct CollaborativeChallengeConfig {
    /// Challenge title
    pub title: String,
    /// Challenge description
    pub description: String,
    /// Target metrics
    pub target_metrics: HashMap<String, f32>,
    /// Challenge duration
    pub duration: chrono::Duration,
    /// Rewards
    pub rewards: Vec<ChallengeReward>,
    /// Start time
    pub starts_at: DateTime<Utc>,
    /// Optional study group
    pub study_group_id: Option<Uuid>,
}

/// Challenge status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChallengeStatus {
    /// Description
    Pending,
    /// Description
    Active,
    /// Description
    Completed,
    /// Description
    Cancelled,
}

/// Challenge reward
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeReward {
    /// Reward type
    pub reward_type: RewardType,
    /// Reward value
    pub value: u32,
    /// Reward description
    pub description: String,
}

/// Reward types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RewardType {
    /// Description
    Points,
    /// Description
    Badge,
    /// Description
    Title,
    /// Description
    Certification,
}

/// Mentorship relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MentorshipPair {
    /// Mentorship ID
    pub id: Uuid,
    /// Mentor ID
    pub mentor_id: Uuid,
    /// Mentee ID
    pub mentee_id: Uuid,
    /// Focus areas
    pub focus_areas: Vec<FocusArea>,
    /// Meeting schedule
    pub meeting_schedule: TimeSlot,
    /// Goals
    pub goals: Vec<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Status
    pub status: MentorshipStatus,
    /// Progress tracking
    pub progress: MentorshipProgress,
}

/// Mentorship preferences
#[derive(Debug, Clone)]
pub struct MentorshipPreferences {
    /// Preferred focus areas
    pub focus_areas: Vec<FocusArea>,
    /// Preferred meeting times
    pub preferred_times: Vec<TimeSlot>,
    /// Experience level seeking
    pub experience_level: ExperienceLevel,
    /// Communication style preference
    pub communication_style: CommunicationStyle,
}

/// Mentorship match suggestion
#[derive(Debug, Clone)]
pub struct MentorshipMatch {
    /// Mentor ID
    pub mentor_id: Uuid,
    /// Mentor name
    pub mentor_name: String,
    /// Compatibility score (0.0 to 1.0)
    pub compatibility_score: f32,
    /// Shared focus areas
    pub shared_focus_areas: Vec<FocusArea>,
    /// Mentor expertise
    pub mentor_expertise: Vec<FocusArea>,
    /// Availability
    pub availability: TimeSlot,
}

/// Mentorship configuration
#[derive(Debug, Clone)]
pub struct MentorshipConfig {
    /// Focus areas
    pub focus_areas: Vec<FocusArea>,
    /// Meeting schedule
    pub meeting_schedule: TimeSlot,
    /// Goals
    pub goals: Vec<String>,
}

/// Mentorship status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MentorshipStatus {
    /// Description
    Active,
    /// Description
    Paused,
    /// Description
    Completed,
    /// Description
    Cancelled,
}

/// Mentorship progress tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MentorshipProgress {
    /// Sessions completed
    pub sessions_completed: u32,
    /// Goals achieved
    pub goals_achieved: u32,
    /// Satisfaction rating
    pub satisfaction_rating: Option<f32>,
    /// Notes
    pub notes: Vec<String>,
}

/// Study group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudyGroup {
    /// Group ID
    pub id: Uuid,
    /// Group name
    pub name: String,
    /// Group description
    pub description: String,
    /// Creator ID
    pub creator_id: Uuid,
    /// Member IDs
    pub members: Vec<Uuid>,
    /// Focus areas
    pub focus_areas: Vec<FocusArea>,
    /// Meeting schedule
    pub meeting_schedule: TimeSlot,
    /// Group goals
    pub goals: Vec<String>,
    /// Active challenges
    pub active_challenges: Vec<Uuid>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Whether group is active
    pub is_active: bool,
    /// Progress tracking
    pub progress: StudyGroupProgress,
}

/// Study group configuration
#[derive(Debug, Clone)]
pub struct StudyGroupConfig {
    /// Group name
    pub name: String,
    /// Group description
    pub description: String,
    /// Focus areas
    pub focus_areas: Vec<FocusArea>,
    /// Meeting schedule
    pub meeting_schedule: TimeSlot,
    /// Goals
    pub goals: Vec<String>,
}

/// Study group progress
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StudyGroupProgress {
    /// Total sessions
    pub total_sessions: u32,
    /// Average attendance
    pub average_attendance: f32,
    /// Goals completed
    pub goals_completed: u32,
    /// Group satisfaction
    pub group_satisfaction: Option<f32>,
}

/// Time slot for scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSlot {
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: DateTime<Utc>,
    /// Recurrence pattern
    pub recurrence: Recurrence,
}

/// Recurrence patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Recurrence {
    /// Description
    None,
    /// Description
    Daily,
    /// Description
    Weekly,
    /// Description
    Monthly,
}

/// Experience levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExperienceLevel {
    /// Description
    Beginner,
    /// Description
    Intermediate,
    /// Description
    Advanced,
    /// Description
    Expert,
}

/// Communication styles
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CommunicationStyle {
    /// Description
    Formal,
    /// Description
    Casual,
    /// Description
    Direct,
    /// Description
    Supportive,
    /// Description
    Analytical,
}

/// Social connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialConnection {
    /// Connected user ID
    pub user_id: Uuid,
    /// Connection type
    pub connection_type: ConnectionType,
    /// Connection strength (0.0 to 1.0)
    pub strength: f32,
    /// When connection was established
    pub connected_at: DateTime<Utc>,
}

/// Connection types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Description
    Friend,
    /// Description
    StudyPartner,
    /// Description
    Mentor,
    /// Description
    Mentee,
    /// Description
    PeerGroup,
}

/// Forum for community discussions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Forum {
    /// Forum ID
    pub id: Uuid,
    /// Forum name
    pub name: String,
    /// Forum description
    pub description: String,
    /// Category
    pub category: ForumCategory,
    /// Moderators
    pub moderators: Vec<Uuid>,
    /// Thread IDs
    pub threads: Vec<Uuid>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Whether forum is active
    pub is_active: bool,
}

/// Forum categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ForumCategory {
    /// Description
    General,
    /// Description
    QuestionsAndAnswers,
    /// Description
    TipsAndTricks,
    /// Description
    Challenges,
    /// Description
    Announcements,
    /// Description
    Feedback,
}

/// Social learning recommendations
#[derive(Debug, Clone)]
pub struct SocialLearningRecommendations {
    /// Suggested peer groups
    pub suggested_peer_groups: Vec<PeerGroupSuggestion>,
    /// Mentor recommendations
    pub mentor_recommendations: Vec<MentorshipMatch>,
    /// Study group suggestions
    pub study_group_suggestions: Vec<StudyGroupSuggestion>,
    /// Collaborative opportunities
    pub collaborative_opportunities: Vec<CollaborativeOpportunity>,
}

/// Peer group suggestion
#[derive(Debug, Clone)]
pub struct PeerGroupSuggestion {
    /// Group ID
    pub group_id: Uuid,
    /// Group name
    pub group_name: String,
    /// Compatibility score
    pub compatibility_score: f32,
    /// Shared focus areas
    pub shared_focus_areas: Vec<FocusArea>,
    /// Current member count
    pub member_count: usize,
}

/// Study group suggestion
#[derive(Debug, Clone)]
pub struct StudyGroupSuggestion {
    /// Group ID
    pub group_id: Uuid,
    /// Group name
    pub group_name: String,
    /// Compatibility score
    pub compatibility_score: f32,
    /// Focus areas
    pub focus_areas: Vec<FocusArea>,
    /// Meeting schedule
    pub meeting_schedule: TimeSlot,
}

/// Collaborative opportunity
#[derive(Debug, Clone)]
pub struct CollaborativeOpportunity {
    /// Opportunity type
    pub opportunity_type: String,
    /// Description
    pub description: String,
    /// Potential partners
    pub potential_partners: Vec<Uuid>,
    /// Expected benefits
    pub expected_benefits: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_social_system_creation() {
        let system = SocialSystem::new();
        assert!(system.peer_groups.is_empty());
        assert!(system.mentorships.is_empty());
    }

    #[test]
    fn test_peer_group_creation() {
        let mut system = SocialSystem::new();
        let creator_id = Uuid::new_v4();

        let config = PeerGroupConfig {
            name: "Test Group".to_string(),
            description: "A test peer group".to_string(),
            max_members: 5,
            focus_areas: vec![FocusArea::Pronunciation],
            privacy_level: PrivacyLevel::Public,
        };

        let group_id = system.create_peer_group(creator_id, config);
        assert!(system.peer_groups.contains_key(&group_id));

        let group = &system.peer_groups[&group_id];
        assert_eq!(group.creator_id, creator_id);
        assert_eq!(group.members.len(), 1);
        assert!(group.members.contains(&creator_id));
    }

    #[test]
    fn test_peer_group_joining() {
        let mut system = SocialSystem::new();
        let creator_id = Uuid::new_v4();
        let joiner_id = Uuid::new_v4();

        let config = PeerGroupConfig {
            name: "Test Group".to_string(),
            description: "A test peer group".to_string(),
            max_members: 5,
            focus_areas: vec![FocusArea::Pronunciation],
            privacy_level: PrivacyLevel::Public,
        };

        let group_id = system.create_peer_group(creator_id, config);
        let result = system.join_peer_group(joiner_id, group_id);

        assert!(result.is_ok());

        let group = &system.peer_groups[&group_id];
        assert_eq!(group.members.len(), 2);
        assert!(group.members.contains(&joiner_id));
    }

    #[test]
    fn test_collaborative_challenge_creation() {
        let mut system = SocialSystem::new();
        let creator_id = Uuid::new_v4();

        let config = CollaborativeChallengeConfig {
            title: "Pronunciation Challenge".to_string(),
            description: "Improve pronunciation accuracy together".to_string(),
            target_metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("accuracy".to_string(), 0.9);
                metrics
            },
            duration: chrono::Duration::days(7),
            rewards: vec![ChallengeReward {
                reward_type: RewardType::Points,
                value: 100,
                description: "100 bonus points".to_string(),
            }],
            starts_at: Utc::now() + chrono::Duration::hours(1),
            study_group_id: None,
        };

        let challenge_id = system.create_collaborative_challenge(creator_id, config);

        // The created challenge must be real, persisted state -- retrievable
        // afterward with the same data that was passed in, not a discarded
        // value behind a freshly-minted id.
        let stored = system
            .get_collaborative_challenge(challenge_id)
            .expect("challenge should be retrievable after creation");
        assert_eq!(stored.id, challenge_id);
        assert_eq!(stored.title, "Pronunciation Challenge");
        assert_eq!(stored.creator_id, creator_id);
        assert_eq!(stored.participants, vec![creator_id]);
        assert_eq!(stored.status, ChallengeStatus::Pending);
    }

    /// A challenge id that was never created must honestly return `None`,
    /// not a fabricated challenge.
    #[test]
    fn test_get_collaborative_challenge_unknown_id_is_none() {
        let system = SocialSystem::new();
        assert!(system.get_collaborative_challenge(Uuid::new_v4()).is_none());
    }

    #[test]
    fn test_study_group_creation() {
        let mut system = SocialSystem::new();
        let creator_id = Uuid::new_v4();

        let config = StudyGroupConfig {
            name: "Advanced Pronunciation Study Group".to_string(),
            description: "Focus on advanced pronunciation techniques".to_string(),
            focus_areas: vec![FocusArea::Pronunciation, FocusArea::Intonation],
            meeting_schedule: TimeSlot {
                start_time: Utc::now(),
                end_time: Utc::now() + chrono::Duration::hours(1),
                recurrence: Recurrence::Weekly,
            },
            goals: vec!["Improve accuracy by 10%".to_string()],
        };

        let group_id = system.create_study_group(creator_id, config);
        assert!(system.study_groups.contains_key(&group_id));

        let group = &system.study_groups[&group_id];
        assert_eq!(group.creator_id, creator_id);
        assert_eq!(group.members.len(), 1);
    }

    #[test]
    fn test_mentorship_preferences() {
        let preferences = MentorshipPreferences {
            focus_areas: vec![FocusArea::Pronunciation],
            preferred_times: vec![TimeSlot {
                start_time: Utc::now(),
                end_time: Utc::now() + chrono::Duration::hours(1),
                recurrence: Recurrence::Weekly,
            }],
            experience_level: ExperienceLevel::Intermediate,
            communication_style: CommunicationStyle::Supportive,
        };

        assert_eq!(preferences.focus_areas.len(), 1);
        assert_eq!(preferences.experience_level, ExperienceLevel::Intermediate);
    }

    fn progress_with_pronunciation(score: f32) -> UserProgress {
        UserProgress {
            average_scores: crate::traits::SessionScores {
                average_pronunciation: score,
                ..crate::traits::SessionScores::default()
            },
            ..UserProgress::default()
        }
    }

    /// A peer present in a shared group but absent from the caller-supplied
    /// `peer_progress` map must be honestly omitted -- never replaced by a
    /// fabricated stand-in.
    #[test]
    fn test_get_peer_comparison_omits_peers_without_real_data() {
        let mut system = SocialSystem::new();
        let user_id = Uuid::new_v4();
        let known_peer = Uuid::new_v4();
        let unknown_peer = Uuid::new_v4();

        let group_id = system.create_peer_group(
            user_id,
            PeerGroupConfig {
                name: "Group".to_string(),
                description: String::new(),
                max_members: 10,
                focus_areas: vec![FocusArea::Pronunciation],
                privacy_level: PrivacyLevel::Public,
            },
        );
        system.join_peer_group(known_peer, group_id).unwrap();
        system.join_peer_group(unknown_peer, group_id).unwrap();

        let user_progress = progress_with_pronunciation(0.5);
        let mut peer_progress = HashMap::new();
        peer_progress.insert(known_peer, progress_with_pronunciation(0.6));
        // `unknown_peer` deliberately has no entry.

        let comparisons = system.get_peer_comparison(user_id, &user_progress, &peer_progress);

        assert_eq!(comparisons.len(), 1);
        assert_eq!(comparisons[0].peer_id, known_peer);
    }

    /// With zero real peer data available, the result must be an honest
    /// empty list, not a synthesized comparison.
    #[test]
    fn test_get_peer_comparison_empty_without_any_peer_data() {
        let mut system = SocialSystem::new();
        let user_id = Uuid::new_v4();
        let peer_id = Uuid::new_v4();

        let group_id = system.create_peer_group(
            user_id,
            PeerGroupConfig {
                name: "Group".to_string(),
                description: String::new(),
                max_members: 10,
                focus_areas: vec![],
                privacy_level: PrivacyLevel::Public,
            },
        );
        system.join_peer_group(peer_id, group_id).unwrap();

        let user_progress = progress_with_pronunciation(0.5);
        let comparisons = system.get_peer_comparison(user_id, &user_progress, &HashMap::new());

        assert!(comparisons.is_empty());
    }

    /// Ranks must reflect real, differing average-pronunciation scores --
    /// the highest real scorer ranks first.
    #[test]
    fn test_rank_in_group_reflects_real_scores() {
        let mut system = SocialSystem::new();
        let user_id = Uuid::new_v4();
        let strong_peer = Uuid::new_v4();
        let weak_peer = Uuid::new_v4();

        let group_id = system.create_peer_group(
            user_id,
            PeerGroupConfig {
                name: "Group".to_string(),
                description: String::new(),
                max_members: 10,
                focus_areas: vec![],
                privacy_level: PrivacyLevel::Public,
            },
        );
        system.join_peer_group(strong_peer, group_id).unwrap();
        system.join_peer_group(weak_peer, group_id).unwrap();

        let user_progress = progress_with_pronunciation(0.5); // middle
        let mut peer_progress = HashMap::new();
        peer_progress.insert(strong_peer, progress_with_pronunciation(0.9)); // top
        peer_progress.insert(weak_peer, progress_with_pronunciation(0.1)); // bottom

        let comparisons = system.get_peer_comparison(user_id, &user_progress, &peer_progress);
        assert_eq!(comparisons.len(), 2);

        // The user (0.5) ranks between the strong peer (0.9, rank 1) and
        // the weak peer (0.1, rank 3).
        for comparison in &comparisons {
            assert_eq!(comparison.user_rank, 2);
            if comparison.peer_id == strong_peer {
                assert_eq!(comparison.peer_rank, 1);
            } else {
                assert_eq!(comparison.peer_rank, 3);
            }
        }
    }

    /// A peer's real, tracked `user_id` must be used as their display name
    /// instead of a fabricated `User_<uuid>` placeholder.
    #[test]
    fn test_get_peer_comparison_uses_real_user_id_as_name() {
        let mut system = SocialSystem::new();
        let user_id = Uuid::new_v4();
        let named_peer = Uuid::new_v4();
        let unnamed_peer = Uuid::new_v4();

        let group_id = system.create_peer_group(
            user_id,
            PeerGroupConfig {
                name: "Group".to_string(),
                description: String::new(),
                max_members: 10,
                focus_areas: vec![],
                privacy_level: PrivacyLevel::Public,
            },
        );
        system.join_peer_group(named_peer, group_id).unwrap();
        system.join_peer_group(unnamed_peer, group_id).unwrap();

        let user_progress = progress_with_pronunciation(0.5);
        let mut peer_progress = HashMap::new();
        peer_progress.insert(
            named_peer,
            UserProgress {
                user_id: "real_alice".to_string(),
                ..progress_with_pronunciation(0.6)
            },
        );
        // `unnamed_peer` has a default (empty) `user_id`, so the honest
        // fallback placeholder must still be used for them.
        peer_progress.insert(unnamed_peer, progress_with_pronunciation(0.4));

        let comparisons = system.get_peer_comparison(user_id, &user_progress, &peer_progress);
        assert_eq!(comparisons.len(), 2);

        for comparison in &comparisons {
            if comparison.peer_id == named_peer {
                assert_eq!(comparison.peer_name, "real_alice");
            } else {
                assert_eq!(
                    comparison.peer_name,
                    format!("User_{}", unnamed_peer.simple())
                );
            }
        }
    }

    /// A mentor with real, high skill in the mentee's preferred focus areas
    /// must score more compatible than one with no tracked skill there.
    #[test]
    fn test_calculate_mentor_compatibility_reflects_real_skill() {
        let preferences = MentorshipPreferences {
            focus_areas: vec![FocusArea::Pronunciation, FocusArea::Fluency],
            preferred_times: vec![],
            experience_level: ExperienceLevel::Intermediate,
            communication_style: CommunicationStyle::Supportive,
        };

        let mut skilled_mentor = UserProgress::default();
        skilled_mentor
            .skill_breakdown
            .insert(FocusArea::Pronunciation, 0.95);
        skilled_mentor
            .skill_breakdown
            .insert(FocusArea::Fluency, 0.9);

        let unskilled_mentor = UserProgress::default();

        let skilled_score =
            SocialSystem::calculate_mentor_compatibility(&preferences, &skilled_mentor);
        let unskilled_score =
            SocialSystem::calculate_mentor_compatibility(&preferences, &unskilled_mentor);

        assert!((skilled_score - 0.925).abs() < 1e-6);
        assert_eq!(unskilled_score, 0.0);
        assert!(skilled_score > unskilled_score);
    }

    /// Expertise must be drawn from the mentor's real `skill_breakdown`,
    /// only areas at/above the expertise threshold, strongest first.
    #[test]
    fn test_calculate_mentor_expertise_reflects_real_skill_breakdown() {
        let mut mentor = UserProgress::default();
        mentor.skill_breakdown.insert(FocusArea::Pronunciation, 0.9);
        mentor.skill_breakdown.insert(FocusArea::Fluency, 0.5); // below threshold
        mentor.skill_breakdown.insert(FocusArea::Rhythm, 0.8);

        let expertise = SocialSystem::calculate_mentor_expertise(&mentor);

        assert_eq!(expertise, vec![FocusArea::Pronunciation, FocusArea::Rhythm]);
    }

    /// A mentor connection with no real profile data available must be
    /// honestly skipped, never assigned a fabricated compatibility score.
    #[test]
    fn test_find_mentorship_matches_omits_mentors_without_real_data() {
        let mut system = SocialSystem::new();
        let mentee_id = Uuid::new_v4();
        let known_mentor = Uuid::new_v4();
        let unknown_mentor = Uuid::new_v4();

        system.connections.insert(
            mentee_id,
            vec![
                SocialConnection {
                    user_id: known_mentor,
                    connection_type: ConnectionType::Mentor,
                    strength: 1.0,
                    connected_at: Utc::now(),
                },
                SocialConnection {
                    user_id: unknown_mentor,
                    connection_type: ConnectionType::Mentor,
                    strength: 1.0,
                    connected_at: Utc::now(),
                },
            ],
        );

        let preferences = MentorshipPreferences {
            focus_areas: vec![FocusArea::Pronunciation],
            preferred_times: vec![],
            experience_level: ExperienceLevel::Intermediate,
            communication_style: CommunicationStyle::Supportive,
        };

        let mut known_mentor_profile = UserProgress::default();
        known_mentor_profile
            .skill_breakdown
            .insert(FocusArea::Pronunciation, 0.95);

        let mut mentor_progress = HashMap::new();
        mentor_progress.insert(known_mentor, known_mentor_profile);
        // `unknown_mentor` deliberately has no entry.

        let matches = system.find_mentorship_matches(mentee_id, &preferences, &mentor_progress);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].mentor_id, known_mentor);
    }

    /// Group compatibility must reflect the real overlap between the
    /// user's tracked skill areas and the group's focus areas.
    #[test]
    fn test_calculate_group_compatibility_reflects_overlap() {
        let group = PeerGroup {
            id: Uuid::new_v4(),
            name: "Group".to_string(),
            description: String::new(),
            creator_id: Uuid::new_v4(),
            members: vec![],
            max_members: 10,
            focus_areas: vec![FocusArea::Pronunciation, FocusArea::Fluency],
            privacy_level: PrivacyLevel::Public,
            created_at: Utc::now(),
            is_active: true,
        };

        let mut full_overlap = UserProgress::default();
        full_overlap
            .skill_breakdown
            .insert(FocusArea::Pronunciation, 0.5);
        full_overlap.skill_breakdown.insert(FocusArea::Fluency, 0.5);

        let no_overlap = UserProgress::default();

        assert_eq!(
            SocialSystem::calculate_group_compatibility(&full_overlap, &group),
            1.0
        );
        assert_eq!(
            SocialSystem::calculate_group_compatibility(&no_overlap, &group),
            0.0
        );
    }

    /// Improvement suggestions must vary with the user's real average
    /// scores, not be a fixed list.
    #[test]
    fn test_generate_improvement_suggestions_reflects_scores() {
        let strong = UserProgress {
            average_scores: crate::traits::SessionScores {
                average_pronunciation: 0.95,
                average_fluency: 0.95,
                average_quality: 0.95,
                ..crate::traits::SessionScores::default()
            },
            training_stats: crate::traits::TrainingStatistics {
                total_sessions: 50,
                ..crate::traits::TrainingStatistics::default()
            },
            ..UserProgress::default()
        };
        let weak = UserProgress {
            average_scores: crate::traits::SessionScores {
                average_pronunciation: 0.2,
                average_fluency: 0.2,
                average_quality: 0.2,
                ..crate::traits::SessionScores::default()
            },
            training_stats: crate::traits::TrainingStatistics {
                total_sessions: 1,
                ..crate::traits::TrainingStatistics::default()
            },
            ..UserProgress::default()
        };

        let strong_suggestions = SocialSystem::generate_improvement_suggestions(&strong);
        let weak_suggestions = SocialSystem::generate_improvement_suggestions(&weak);

        assert_ne!(strong_suggestions, weak_suggestions);
        assert!(weak_suggestions.len() > strong_suggestions.len());
    }
}
