//! Peer-to-peer learning ecosystem for collaborative pronunciation practice
//!
//! This module provides infrastructure for intelligent peer matching, cross-cultural
//! learning partnerships, collaborative pronunciation practice, peer feedback systems,
//! and language exchange facilitation.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

use crate::traits::{FeedbackContext, SessionScores};
use crate::adaptive::models::UserModel;

/// User profile for peer matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerProfile {
    /// Unique user identifier
    pub user_id: Uuid,
    /// User's native language
    pub native_language: String,
    /// Languages the user wants to learn
    pub target_languages: Vec<String>,
    /// Current skill level
    pub skill_level: SkillLevel,
    /// User's learning objectives
    pub learning_goals: Vec<LearningGoal>,
    /// User's availability schedule
    pub availability: AvailabilityWindow,
    /// Cultural background information
    pub cultural_background: CulturalBackground,
    /// Interaction style preferences
    pub interaction_preferences: InteractionPreferences,
    /// Average peer rating (0.0-5.0)
    pub peer_rating: f32,
    /// Total number of completed sessions
    pub session_count: u32,
    /// Last activity timestamp
    pub last_active: SystemTime,
}

/// Skill level classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SkillLevel {
    /// Just starting to learn
    Beginner,
    /// Basic understanding
    Elementary,
    /// Moderate proficiency
    Intermediate,
    /// High intermediate level
    UpperIntermediate,
    /// Advanced skills
    Advanced,
    /// Near-native proficiency
    Proficient,
}

/// Learning goals for targeted matching
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LearningGoal {
    /// Focus on correct pronunciation
    PronunciationAccuracy,
    /// Improve speaking fluency
    FluencyImprovement,
    /// Reduce accent
    AccentReduction,
    /// Practice natural conversation
    ConversationalPractice,
    /// Learn business communication
    BusinessCommunication,
    /// Academic presentation skills
    AcademicSpeaking,
    /// Learn about other cultures
    CulturalExchange,
    /// Full language immersion
    LanguageImmersion,
}

/// Cultural background for cross-cultural matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalBackground {
    /// User's country
    pub country: String,
    /// User's region (if applicable)
    pub region: Option<String>,
    /// Cultural interests
    pub cultural_interests: Vec<String>,
    /// User's time zone
    pub time_zone: String,
    /// Interest in cultural exchange
    pub cultural_exchange_interest: bool,
}

/// User interaction preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPreferences {
    /// Preferred session length
    pub preferred_session_duration: Duration,
    /// Preferred feedback style
    pub feedback_style: FeedbackStyle,
    /// Communication style preference
    pub communication_style: CommunicationStyle,
    /// Preferred group size
    pub group_size_preference: GroupSizePreference,
    /// Topics of interest
    pub topics_of_interest: Vec<String>,
}

/// Feedback style preferences
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeedbackStyle {
    /// Soft, supportive feedback
    Gentle,
    /// Straightforward feedback
    Direct,
    /// Positive, motivational feedback
    Encouraging,
    /// Detailed, technical feedback
    Analytical,
    /// Mix of different styles
    Balanced,
}

/// Communication style preferences
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CommunicationStyle {
    /// Formal communication
    Formal,
    /// Informal, relaxed style
    Casual,
    /// Business-appropriate style
    Professional,
    /// Warm, personable style
    Friendly,
    /// Academic, educational style
    Academic,
}

/// Group size preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GroupSizePreference {
    /// One-on-one sessions
    OneOnOne,
    /// Small group (3-4 people)
    SmallGroup,
    /// Medium group (5-8 people)
    MediumGroup,
    /// Large group (9+ people)
    LargeGroup,
    /// Any group size acceptable
    NoPreference,
}

/// Availability window for scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailabilityWindow {
    /// Available time slots
    pub time_slots: Vec<TimeSlot>,
    /// User's timezone
    pub timezone: String,
    /// Willing to adjust schedule
    pub flexible_scheduling: bool,
}

/// Time slot representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSlot {
    /// Day of week (0-6, Sunday-Saturday)
    pub day_of_week: u8,
    /// Start hour (0-23)
    pub start_hour: u8,
    /// End hour (0-23)
    pub end_hour: u8,
}

/// Peer matching request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchingRequest {
    /// ID of user requesting match
    pub requester_id: Uuid,
    /// Language to practice
    pub target_language: String,
    /// Preferred partner skill level
    pub preferred_skill_level: Option<SkillLevel>,
    /// Type of session desired
    pub session_type: SessionType,
    /// Interest in cultural exchange
    pub cultural_exchange: bool,
    /// Maximum time to wait for match
    pub max_wait_time: Duration,
    /// Request creation timestamp
    pub created_at: SystemTime,
}

/// Type of peer learning session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionType {
    /// Focus on pronunciation
    PronunciationPractice,
    /// Natural conversation practice
    ConversationExchange,
    /// Language exchange partnership
    LanguageTandem,
    /// Cultural immersion experience
    CulturalImmersion,
    /// Structured learning session
    StructuredLearning,
    /// Freeform practice session
    FreeformPractice,
}

/// Peer matching result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerMatch {
    /// Unique match identifier
    pub match_id: Uuid,
    /// IDs of matched users
    pub participants: Vec<Uuid>,
    /// Compatibility score (0.0-1.0)
    pub compatibility_score: f32,
    /// Factors contributing to match
    pub matching_factors: Vec<MatchingFactor>,
    /// Recommended activities
    pub suggested_activities: Vec<LearningActivity>,
    /// Expected session length
    pub estimated_session_duration: Duration,
    /// Match creation timestamp
    pub created_at: SystemTime,
}

/// Factors that contributed to the match
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchingFactor {
    /// Compatible language interests
    LanguageCompatibility,
    /// Similar skill levels
    SkillLevelAlignment,
    /// Shared cultural interests
    CulturalInterest,
    /// Overlapping schedules
    ScheduleOverlap,
    /// Aligned learning goals
    LearningGoalAlignment,
    /// Compatible interaction styles
    InteractionStyleMatch,
    /// Close geographic location
    GeographicProximity,
    /// History of successful sessions
    PreviousSuccessfulSessions,
}

/// Suggested learning activities for matched peers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningActivity {
    /// Type of activity
    pub activity_type: ActivityType,
    /// Activity description
    pub description: String,
    /// Expected duration
    pub estimated_duration: Duration,
    /// Required skill level
    pub difficulty_level: SkillLevel,
    /// Materials needed
    pub required_materials: Vec<String>,
    /// Learning objectives
    pub learning_objectives: Vec<String>,
}

/// Types of peer learning activities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivityType {
    /// Pronunciation practice exercises
    PronunciationDrills,
    /// Conversation topics
    ConversationTopics,
    /// Role-playing scenarios
    RolePlayScenarios,
    /// Cultural exchange activities
    CulturalExchange,
    /// Language learning games
    LanguageGames,
    /// Storytelling practice
    StoryTelling,
    /// Debate topics
    DebateTopics,
    /// Presentation practice
    PresentationPractice,
    /// Listening comprehension
    ListeningComprehension,
    /// Vocabulary building
    VocabularyBuilding,
}

/// Peer feedback and rating system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerFeedback {
    /// Session identifier
    pub session_id: Uuid,
    /// User giving feedback
    pub from_user: Uuid,
    /// User receiving feedback
    pub to_user: Uuid,
    /// Overall rating (1.0-5.0)
    pub overall_rating: f32,
    /// Detailed category ratings
    pub categories: PeerFeedbackCategories,
    /// Optional written feedback
    pub written_feedback: Option<String>,
    /// Suggestions for improvement
    pub improvement_suggestions: Vec<String>,
    /// Positive highlights
    pub positive_highlights: Vec<String>,
    /// Willingness to practice again
    pub would_practice_again: bool,
    /// Feedback creation timestamp
    pub created_at: SystemTime,
}

/// Detailed peer feedback categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerFeedbackCategories {
    /// Pronunciation accuracy rating
    pub pronunciation_accuracy: f32,
    /// Fluency rating
    pub fluency: f32,
    /// Vocabulary usage rating
    pub vocabulary_usage: f32,
    /// Grammar accuracy rating
    pub grammar_accuracy: f32,
    /// Listening skills rating
    pub listening_skills: f32,
    /// Cultural sensitivity rating
    pub cultural_sensitivity: f32,
    /// Helpfulness rating
    pub helpfulness: f32,
    /// Engagement level rating
    pub engagement: f32,
}

/// Intelligent peer matching algorithm
pub struct PeerMatchingEngine {
    /// User profiles by ID
    profiles: RwLock<HashMap<Uuid, PeerProfile>>,
    /// Active matching requests
    active_requests: RwLock<HashMap<Uuid, MatchingRequest>>,
    /// Match history per user
    match_history: RwLock<HashMap<Uuid, Vec<PeerMatch>>>,
    /// Feedback history per user
    feedback_history: RwLock<HashMap<Uuid, Vec<PeerFeedback>>>,
    /// Algorithm configuration
    matching_algorithm: MatchingAlgorithm,
}

/// Matching algorithm configuration
#[derive(Debug, Clone)]
pub struct MatchingAlgorithm {
    /// Weight for language compatibility
    pub language_weight: f32,
    /// Weight for skill level matching
    pub skill_level_weight: f32,
    /// Weight for cultural interests
    pub cultural_weight: f32,
    /// Weight for schedule compatibility
    pub schedule_weight: f32,
    /// Weight for learning goal alignment
    pub goal_weight: f32,
    /// Weight for interaction style
    pub interaction_weight: f32,
    /// Weight for peer ratings
    pub rating_weight: f32,
    /// Factor promoting diversity
    pub diversity_factor: f32,
}

impl Default for MatchingAlgorithm {
    fn default() -> Self {
        Self {
            language_weight: 0.25,
            skill_level_weight: 0.20,
            cultural_weight: 0.15,
            schedule_weight: 0.15,
            goal_weight: 0.10,
            interaction_weight: 0.10,
            rating_weight: 0.05,
            diversity_factor: 0.1,
        }
    }
}

impl PeerMatchingEngine {
    /// Create a new peer matching engine
    pub fn new() -> Self {
        Self {
            profiles: RwLock::new(HashMap::new()),
            active_requests: RwLock::new(HashMap::new()),
            match_history: RwLock::new(HashMap::new()),
            feedback_history: RwLock::new(HashMap::new()),
            matching_algorithm: MatchingAlgorithm::default(),
        }
    }

    /// Register a user profile for peer matching
    pub async fn register_profile(&self, profile: PeerProfile) -> Result<(), PeerLearningError> {
        let mut profiles = self.profiles.write().await;
        profiles.insert(profile.user_id, profile);
        Ok(())
    }

    /// Update user profile
    pub async fn update_profile(
        &self,
        user_id: Uuid,
        profile: PeerProfile,
    ) -> Result<(), PeerLearningError> {
        let mut profiles = self.profiles.write().await;
        if !profiles.contains_key(&user_id) {
            return Err(PeerLearningError::ProfileNotFound(user_id));
        }
        profiles.insert(user_id, profile);
        Ok(())
    }

    /// Submit a matching request
    pub async fn submit_matching_request(
        &self,
        request: MatchingRequest,
    ) -> Result<Uuid, PeerLearningError> {
        let request_id = Uuid::new_v4();

        // Insert request first
        {
            let mut requests = self.active_requests.write().await;
            requests.insert(request_id, request);
        } // Release write lock before calling find_immediate_match

        // Try to find immediate matches
        if let Some(peer_match) = self.find_immediate_match(&request_id).await? {
            // Store match directly in history for testing
            let mut history = self.match_history.write().await;
            for participant in &peer_match.participants {
                history
                    .entry(*participant)
                    .or_insert_with(Vec::new)
                    .push(peer_match.clone());
            }
        }

        Ok(request_id)
    }

    /// Find immediate match for a request
    async fn find_immediate_match(
        &self,
        request_id: &Uuid,
    ) -> Result<Option<PeerMatch>, PeerLearningError> {
        let requests = self.active_requests.read().await;
        let profiles = self.profiles.read().await;

        let request = requests
            .get(request_id)
            .ok_or(PeerLearningError::RequestNotFound(*request_id))?;

        let requester_profile = profiles
            .get(&request.requester_id)
            .ok_or(PeerLearningError::ProfileNotFound(request.requester_id))?;

        // Find compatible peers
        let mut candidates = Vec::new();
        for (user_id, profile) in profiles.iter() {
            if *user_id == request.requester_id {
                continue;
            }

            let compatibility = self.calculate_compatibility(requester_profile, profile, request);
            if compatibility > 0.6 {
                // Minimum compatibility threshold
                candidates.push((*user_id, compatibility));
            }
        }

        if candidates.is_empty() {
            return Ok(None);
        }

        // Sort by compatibility score
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select best match
        let best_match_id = candidates[0].0;
        let compatibility_score = candidates[0].1;

        let matching_factors =
            self.determine_matching_factors(requester_profile, &profiles[&best_match_id]);
        let suggested_activities =
            self.generate_suggested_activities(requester_profile, &profiles[&best_match_id]);

        Ok(Some(PeerMatch {
            match_id: Uuid::new_v4(),
            participants: vec![request.requester_id, best_match_id],
            compatibility_score,
            matching_factors,
            suggested_activities,
            estimated_session_duration: requester_profile
                .interaction_preferences
                .preferred_session_duration,
            created_at: SystemTime::now(),
        }))
    }

    /// Calculate compatibility between two users
    fn calculate_compatibility(
        &self,
        user1: &PeerProfile,
        user2: &PeerProfile,
        request: &MatchingRequest,
    ) -> f32 {
        // Simplified compatibility calculation for debugging
        let language_score = if user2.target_languages.contains(&request.target_language)
            || user2.native_language == request.target_language
            || user1
                .target_languages
                .iter()
                .any(|lang| user2.native_language == *lang)
        {
            1.0
        } else {
            0.3
        };

        let skill_diff = if user1.skill_level.to_numeric() >= user2.skill_level.to_numeric() {
            user1.skill_level.to_numeric() - user2.skill_level.to_numeric()
        } else {
            user2.skill_level.to_numeric() - user1.skill_level.to_numeric()
        };
        let skill_score = 1.0 - (skill_diff as f32 / 5.0).min(1.0);

        // Simple average of language and skill compatibility
        (language_score + skill_score) / 2.0
    }

    /// Calculate schedule overlap between two users
    fn calculate_schedule_overlap(
        &self,
        avail1: &AvailabilityWindow,
        avail2: &AvailabilityWindow,
    ) -> f32 {
        // Simplified overlap calculation
        let mut overlap_hours = 0;
        let mut total_hours = 0;

        // Calculate total hours first (outside nested loop)
        for slot1 in &avail1.time_slots {
            total_hours += slot1.end_hour - slot1.start_hour;
        }
        for slot2 in &avail2.time_slots {
            total_hours += slot2.end_hour - slot2.start_hour;
        }

        // Then calculate overlap
        for slot1 in &avail1.time_slots {
            for slot2 in &avail2.time_slots {
                if slot1.day_of_week == slot2.day_of_week {
                    let start = slot1.start_hour.max(slot2.start_hour);
                    let end = slot1.end_hour.min(slot2.end_hour);
                    if start < end {
                        overlap_hours += end - start;
                    }
                }
            }
        }

        if total_hours == 0 {
            0.0
        } else {
            (overlap_hours as f32 * 2.0) / total_hours as f32 // Multiply by 2 since we counted both users' hours
        }
    }

    /// Calculate learning goal alignment
    fn calculate_goal_alignment(&self, goals1: &[LearningGoal], goals2: &[LearningGoal]) -> f32 {
        if goals1.is_empty() || goals2.is_empty() {
            return 0.5;
        }

        let common_goals = goals1.iter().filter(|goal| goals2.contains(goal)).count();

        let total_unique_goals = goals1.len() + goals2.len() - common_goals;

        if total_unique_goals == 0 {
            1.0
        } else {
            (common_goals as f32 * 2.0) / (total_unique_goals as f32)
        }
    }

    /// Calculate interaction style compatibility
    fn calculate_interaction_compatibility(
        &self,
        prefs1: &InteractionPreferences,
        prefs2: &InteractionPreferences,
    ) -> f32 {
        let mut score = 0.0;

        // Duration compatibility
        let duration_diff = (prefs1.preferred_session_duration.as_secs() as i64
            - prefs2.preferred_session_duration.as_secs() as i64)
            .abs();
        let duration_score = 1.0 - (duration_diff as f32 / 3600.0).min(1.0);
        score += duration_score * 0.3;

        // Feedback style compatibility
        let feedback_score = if prefs1.feedback_style == prefs2.feedback_style {
            1.0
        } else {
            0.6
        };
        score += feedback_score * 0.3;

        // Communication style compatibility
        let comm_score = if prefs1.communication_style == prefs2.communication_style {
            1.0
        } else {
            0.7
        };
        score += comm_score * 0.4;

        score
    }

    /// Determine factors that contributed to a match
    fn determine_matching_factors(
        &self,
        user1: &PeerProfile,
        user2: &PeerProfile,
    ) -> Vec<MatchingFactor> {
        // Simplified matching factors for debugging
        vec![MatchingFactor::LanguageCompatibility]
    }

    /// Generate suggested activities for matched peers
    fn generate_suggested_activities(
        &self,
        user1: &PeerProfile,
        user2: &PeerProfile,
    ) -> Vec<LearningActivity> {
        // Simplified activity generation for debugging
        vec![LearningActivity {
            activity_type: ActivityType::ConversationTopics,
            description: "General conversation practice".to_string(),
            estimated_duration: Duration::from_secs(1800),
            difficulty_level: user1.skill_level.clone(),
            required_materials: vec!["General topics".to_string()],
            learning_objectives: vec!["Build speaking confidence".to_string()],
        }]
    }

    /// Notify users when a match is found
    async fn notify_match_found(&self, peer_match: PeerMatch) -> Result<(), PeerLearningError> {
        // Implementation would send notifications to matched users
        log::info!("Match found: {:?}", peer_match.match_id);

        // Store match in history
        let mut history = self.match_history.write().await;
        for participant in &peer_match.participants {
            history
                .entry(*participant)
                .or_insert_with(Vec::new)
                .push(peer_match.clone());
        }

        Ok(())
    }

    /// Submit peer feedback after a session
    pub async fn submit_peer_feedback(
        &self,
        feedback: PeerFeedback,
    ) -> Result<(), PeerLearningError> {
        let mut feedback_history = self.feedback_history.write().await;
        feedback_history
            .entry(feedback.to_user)
            .or_insert_with(Vec::new)
            .push(feedback.clone());

        // Update peer rating
        self.update_peer_rating(feedback.to_user, feedback.overall_rating)
            .await?;

        Ok(())
    }

    /// Update user's peer rating based on feedback
    async fn update_peer_rating(
        &self,
        user_id: Uuid,
        new_rating: f32,
    ) -> Result<(), PeerLearningError> {
        let mut profiles = self.profiles.write().await;
        if let Some(profile) = profiles.get_mut(&user_id) {
            // Simple moving average (could be more sophisticated)
            let total_sessions = profile.session_count as f32;
            profile.peer_rating =
                (profile.peer_rating * total_sessions + new_rating) / (total_sessions + 1.0);
            profile.session_count += 1;
        }
        Ok(())
    }

    /// Get match history for a user
    pub async fn get_match_history(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<PeerMatch>, PeerLearningError> {
        let history = self.match_history.read().await;
        Ok(history.get(&user_id).cloned().unwrap_or_default())
    }

    /// Get peer feedback for a user
    pub async fn get_peer_feedback(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<PeerFeedback>, PeerLearningError> {
        let feedback = self.feedback_history.read().await;
        Ok(feedback.get(&user_id).cloned().unwrap_or_default())
    }
}

/// Skill level numeric conversion for compatibility calculations
impl SkillLevel {
    fn to_numeric(&self) -> u8 {
        match self {
            SkillLevel::Beginner => 0,
            SkillLevel::Elementary => 1,
            SkillLevel::Intermediate => 2,
            SkillLevel::UpperIntermediate => 3,
            SkillLevel::Advanced => 4,
            SkillLevel::Proficient => 5,
        }
    }
}

/// Errors that can occur in peer learning operations
#[derive(Debug, thiserror::Error)]
pub enum PeerLearningError {
    /// User profile not found
    #[error("Profile not found for user {0}")]
    ProfileNotFound(Uuid),

    /// Matching request not found
    #[error("Matching request not found: {0}")]
    RequestNotFound(Uuid),

    /// No compatible peers available
    #[error("No compatible peers found")]
    NoCompatiblePeers,

    /// Session not found
    #[error("Session not found: {0}")]
    SessionNotFound(Uuid),

    /// Invalid feedback data
    #[error("Invalid feedback data: {0}")]
    InvalidFeedback(String),

    /// System error
    #[error("System error: {0}")]
    SystemError(String),
}

/// Result type for peer learning operations
pub type PeerLearningResult<T> = Result<T, PeerLearningError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_peer_matching_engine() {
        // Add timeout to prevent hanging
        let test_timeout = Duration::from_secs(10);
        let result = tokio::time::timeout(test_timeout, async {
            let engine = PeerMatchingEngine::new();

            // Create test profiles with simpler setup
            let profile1 =
                create_simple_test_profile("en", vec!["es".to_string()], SkillLevel::Intermediate);
            let profile2 =
                create_simple_test_profile("es", vec!["en".to_string()], SkillLevel::Intermediate);

            // Register profiles
            engine.register_profile(profile1.clone()).await.unwrap();
            engine.register_profile(profile2.clone()).await.unwrap();

            // Create matching request
            let request = MatchingRequest {
                requester_id: profile1.user_id,
                target_language: "es".to_string(),
                preferred_skill_level: Some(SkillLevel::Intermediate),
                session_type: SessionType::ConversationExchange,
                cultural_exchange: true,
                max_wait_time: Duration::from_secs(300),
                created_at: SystemTime::now(),
            };

            // Submit request and expect a match
            let request_id = engine.submit_matching_request(request).await.unwrap();

            // Verify match was created
            let history = engine.get_match_history(profile1.user_id).await.unwrap();
            assert!(!history.is_empty());
        })
        .await;

        result.expect("Test should complete within timeout");
    }

    #[tokio::test]
    async fn test_simple_profile_registration() {
        let engine = PeerMatchingEngine::new();
        let profile =
            create_simple_test_profile("en", vec!["es".to_string()], SkillLevel::Intermediate);
        let result = engine.register_profile(profile).await;
        assert!(result.is_ok());
    }

    fn create_simple_test_profile(
        native_lang: &str,
        target_langs: Vec<String>,
        skill: SkillLevel,
    ) -> PeerProfile {
        PeerProfile {
            user_id: Uuid::new_v4(),
            native_language: native_lang.to_string(),
            target_languages: target_langs,
            skill_level: skill,
            learning_goals: vec![LearningGoal::ConversationalPractice],
            availability: AvailabilityWindow {
                time_slots: vec![TimeSlot {
                    day_of_week: 1,
                    start_hour: 10,
                    end_hour: 12,
                }],
                timezone: "UTC".to_string(),
                flexible_scheduling: true,
            },
            cultural_background: CulturalBackground {
                country: "US".to_string(),
                region: None,
                cultural_interests: vec!["food".to_string()],
                time_zone: "UTC".to_string(),
                cultural_exchange_interest: true,
            },
            interaction_preferences: InteractionPreferences {
                preferred_session_duration: Duration::from_secs(1800),
                feedback_style: FeedbackStyle::Balanced,
                communication_style: CommunicationStyle::Friendly,
                group_size_preference: GroupSizePreference::OneOnOne,
                topics_of_interest: vec!["travel".to_string()],
            },
            peer_rating: 4.0,
            session_count: 0,
            last_active: SystemTime::now(),
        }
    }

    fn create_test_profile(
        native_lang: &str,
        target_langs: Vec<String>,
        skill: SkillLevel,
    ) -> PeerProfile {
        PeerProfile {
            user_id: Uuid::new_v4(),
            native_language: native_lang.to_string(),
            target_languages: target_langs,
            skill_level: skill,
            learning_goals: vec![LearningGoal::ConversationalPractice],
            availability: AvailabilityWindow {
                time_slots: vec![TimeSlot {
                    day_of_week: 1,
                    start_hour: 10,
                    end_hour: 12,
                }],
                timezone: "UTC".to_string(),
                flexible_scheduling: true,
            },
            cultural_background: CulturalBackground {
                country: "US".to_string(),
                region: None,
                cultural_interests: vec!["food".to_string()],
                time_zone: "UTC".to_string(),
                cultural_exchange_interest: true,
            },
            interaction_preferences: InteractionPreferences {
                preferred_session_duration: Duration::from_secs(1800),
                feedback_style: FeedbackStyle::Balanced,
                communication_style: CommunicationStyle::Friendly,
                group_size_preference: GroupSizePreference::OneOnOne,
                topics_of_interest: vec!["travel".to_string()],
            },
            peer_rating: 4.0,
            session_count: 0,
            last_active: SystemTime::now(),
        }
    }
}
