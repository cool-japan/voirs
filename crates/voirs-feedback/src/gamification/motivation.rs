//! Motivation monitoring and intervention system
//!
//! This module provides adaptive motivation features including:
//! - Personality-based messaging and communication adaptation
//! - Motivation monitoring and burnout prevention
//! - Intervention optimization and re-engagement campaigns
//! - Personalized encouragement and progress celebration

use crate::traits::UserProgress;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Motivation system manager
#[derive(Debug, Clone)]
pub struct MotivationSystem {
    /// User personality profiles
    personality_profiles: HashMap<Uuid, PersonalityProfile>,
    /// Motivation tracking
    motivation_tracking: HashMap<Uuid, MotivationTracker>,
    /// Intervention history
    intervention_history: HashMap<Uuid, Vec<Intervention>>,
    /// Message templates
    message_templates: HashMap<CommunicationStyle, Vec<MessageTemplate>>,
}

impl MotivationSystem {
    /// Create new motivation system
    pub fn new() -> Self {
        let mut system = Self {
            personality_profiles: HashMap::new(),
            motivation_tracking: HashMap::new(),
            intervention_history: HashMap::new(),
            message_templates: HashMap::new(),
        };
        system.initialize_message_templates();
        system
    }

    /// Initialize message templates for different communication styles
    fn initialize_message_templates(&mut self) {
        // Direct style messages
        self.message_templates.insert(CommunicationStyle::Direct, vec![
            MessageTemplate {
                message_type: MessageType::Encouragement,
                content: "Your pronunciation accuracy improved by {improvement}%. Keep pushing forward!".to_string(),
                effectiveness_factors: vec![PersonalityFactor::Conscientiousness],
            },
            MessageTemplate {
                message_type: MessageType::Challenge,
                content: "Ready for the next level? Your skills are sharp enough to tackle harder challenges.".to_string(),
                effectiveness_factors: vec![PersonalityFactor::Openness],
            },
        ]);

        // Supportive style messages
        self.message_templates.insert(CommunicationStyle::Supportive, vec![
            MessageTemplate {
                message_type: MessageType::Encouragement,
                content: "You're doing amazing! Every practice session brings you closer to your goals. 🌟".to_string(),
                effectiveness_factors: vec![PersonalityFactor::Agreeableness],
            },
            MessageTemplate {
                message_type: MessageType::Comfort,
                content: "It's okay to have difficult days. What matters is that you're here, trying your best.".to_string(),
                effectiveness_factors: vec![PersonalityFactor::Neuroticism],
            },
        ]);

        // Analytical style messages
        self.message_templates.insert(CommunicationStyle::Analytical, vec![
            MessageTemplate {
                message_type: MessageType::Progress,
                content: "Data shows you've improved {metric} by {percentage}% over the last {timeframe} sessions.".to_string(),
                effectiveness_factors: vec![PersonalityFactor::Conscientiousness, PersonalityFactor::Openness],
            },
            MessageTemplate {
                message_type: MessageType::Insight,
                content: "Your performance pattern suggests focusing on {area} could yield a {prediction}% improvement.".to_string(),
                effectiveness_factors: vec![PersonalityFactor::Openness],
            },
        ]);

        // Casual style messages
        self.message_templates.insert(
            CommunicationStyle::Casual,
            vec![
                MessageTemplate {
                    message_type: MessageType::Encouragement,
                    content: "Nice work today! You're getting the hang of this! 🎉".to_string(),
                    effectiveness_factors: vec![PersonalityFactor::Extroversion],
                },
                MessageTemplate {
                    message_type: MessageType::Celebration,
                    content: "Woohoo! You just crushed that session! Time to celebrate! 🎊"
                        .to_string(),
                    effectiveness_factors: vec![PersonalityFactor::Extroversion],
                },
            ],
        );

        // Motivational style messages
        self.message_templates.insert(CommunicationStyle::Motivational, vec![
            MessageTemplate {
                message_type: MessageType::Inspiration,
                content: "Every expert was once a beginner. You're building skills that will last a lifetime!".to_string(),
                effectiveness_factors: vec![PersonalityFactor::Conscientiousness],
            },
            MessageTemplate {
                message_type: MessageType::Challenge,
                content: "Champions are made through consistent effort. You have what it takes!".to_string(),
                effectiveness_factors: vec![PersonalityFactor::Conscientiousness, PersonalityFactor::Extroversion],
            },
        ]);
    }

    /// Set user personality profile
    pub fn set_personality_profile(&mut self, user_id: Uuid, profile: PersonalityProfile) {
        self.personality_profiles.insert(user_id, profile);
        self.motivation_tracking
            .insert(user_id, MotivationTracker::new());
    }

    /// Update motivation tracking
    pub fn update_motivation(
        &mut self,
        user_id: Uuid,
        session_data: &SessionMotivationData,
    ) -> MotivationAssessment {
        let tracker = self
            .motivation_tracking
            .entry(user_id)
            .or_insert_with(MotivationTracker::new);

        // Update metrics
        tracker.engagement_score = session_data.engagement_score;
        tracker.frustration_level = session_data.frustration_level;
        tracker.satisfaction_rating = session_data.satisfaction_rating;
        tracker.session_completion_rate = session_data.completion_rate;
        tracker.last_updated = Utc::now();

        // Add to history
        tracker
            .engagement_history
            .push(session_data.engagement_score);
        tracker
            .frustration_history
            .push(session_data.frustration_level);

        // Keep only recent history
        if tracker.engagement_history.len() > 20 {
            tracker.engagement_history.remove(0);
            tracker.frustration_history.remove(0);
        }

        // Clone necessary data to avoid borrowing conflicts
        let engagement_history = tracker.engagement_history.clone();
        let frustration_history = tracker.frustration_history.clone();
        let tracker_clone = tracker.clone();

        // Calculate trends
        let engagement_trend = self.calculate_trend(&engagement_history);
        let frustration_trend = self.calculate_trend(&frustration_history);

        // Assess motivation state
        let motivation_state = self.assess_motivation_state(&tracker_clone);
        let risk_level = self.calculate_risk_level(&tracker_clone);
        let recommended_actions = self.generate_recommendations(user_id, &tracker_clone);

        MotivationAssessment {
            current_state: motivation_state,
            engagement_trend,
            frustration_trend,
            risk_level,
            recommended_actions,
        }
    }

    /// Generate personalized message
    pub fn generate_personalized_message(
        &self,
        user_id: Uuid,
        context: MessageContext,
    ) -> Option<PersonalizedMessage> {
        let profile = self.personality_profiles.get(&user_id)?;
        let communication_style = self.determine_communication_style(profile, &context);

        let templates = self.message_templates.get(&communication_style)?;
        let relevant_templates: Vec<&MessageTemplate> = templates
            .iter()
            .filter(|t| t.message_type == context.message_type)
            .collect();

        if relevant_templates.is_empty() {
            return None;
        }

        // Select best template based on personality factors
        let best_template = relevant_templates.iter().max_by(|a, b| {
            self.calculate_template_effectiveness(a, profile)
                .partial_cmp(&self.calculate_template_effectiveness(b, profile))
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

        let personalized_content = self.personalize_content(&best_template.content, &context);

        Some(PersonalizedMessage {
            content: personalized_content,
            communication_style,
            message_type: context.message_type,
            effectiveness_score: self.calculate_template_effectiveness(best_template, profile),
            timestamp: Utc::now(),
        })
    }

    /// Check for intervention needs
    pub fn check_intervention_needs(
        &mut self,
        user_id: Uuid,
    ) -> Option<InterventionRecommendation> {
        let tracker = self.motivation_tracking.get(&user_id)?;
        let profile = self.personality_profiles.get(&user_id)?;

        // Check various risk factors
        if self.detect_burnout_risk(tracker) {
            return Some(self.create_burnout_intervention(user_id, profile));
        }

        if self.detect_disengagement(tracker) {
            return Some(self.create_re_engagement_intervention(user_id, profile));
        }

        if self.detect_frustration_spike(tracker) {
            return Some(self.create_frustration_intervention(user_id, profile));
        }

        None
    }

    /// Implement intervention
    pub fn implement_intervention(
        &mut self,
        user_id: Uuid,
        intervention: Intervention,
    ) -> InterventionResult {
        // Record intervention
        self.intervention_history
            .entry(user_id)
            .or_insert_with(Vec::new)
            .push(intervention.clone());

        // In a real implementation, this would trigger actual intervention actions
        // such as adjusting difficulty, sending messages, or scheduling breaks

        InterventionResult {
            intervention_id: intervention.id,
            implemented_at: Utc::now(),
            expected_outcome: intervention.expected_outcome.clone(),
            follow_up_required: intervention.follow_up_duration.is_some(),
        }
    }

    /// Helper methods
    fn calculate_trend(&self, history: &[f32]) -> Trend {
        if history.len() < 3 {
            return Trend::Stable;
        }

        let recent_avg = history.iter().rev().take(3).sum::<f32>() / 3.0;
        let older_avg = history.iter().rev().skip(3).take(3).sum::<f32>() / 3.0;

        let difference = recent_avg - older_avg;

        if difference > 0.1 {
            Trend::Improving
        } else if difference < -0.1 {
            Trend::Declining
        } else {
            Trend::Stable
        }
    }

    fn assess_motivation_state(&self, tracker: &MotivationTracker) -> MotivationState {
        let avg_engagement = tracker.engagement_history.iter().sum::<f32>()
            / tracker.engagement_history.len() as f32;
        let avg_frustration = tracker.frustration_history.iter().sum::<f32>()
            / tracker.frustration_history.len() as f32;

        match (avg_engagement, avg_frustration) {
            (e, f) if e > 0.7 && f < 0.3 => MotivationState::Highly,
            (e, f) if e > 0.5 && f < 0.5 => MotivationState::Moderately,
            (e, f) if e > 0.3 && f < 0.7 => MotivationState::Somewhat,
            _ => MotivationState::Low,
        }
    }

    fn calculate_risk_level(&self, tracker: &MotivationTracker) -> RiskLevel {
        let recent_engagement = tracker.engagement_history.iter().rev().take(5).sum::<f32>() / 5.0;
        let recent_frustration = tracker
            .frustration_history
            .iter()
            .rev()
            .take(5)
            .sum::<f32>()
            / 5.0;

        if recent_engagement < 0.3 || recent_frustration > 0.7 {
            RiskLevel::High
        } else if recent_engagement < 0.5 || recent_frustration > 0.5 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    fn generate_recommendations(&self, user_id: Uuid, tracker: &MotivationTracker) -> Vec<String> {
        let mut recommendations = Vec::new();

        if tracker.engagement_score < 0.5 {
            recommendations.push("Consider adjusting difficulty level".to_string());
            recommendations.push("Try shorter practice sessions".to_string());
        }

        if tracker.frustration_level > 0.6 {
            recommendations.push("Take a break and return refreshed".to_string());
            recommendations.push("Focus on easier exercises to build confidence".to_string());
        }

        if tracker.session_completion_rate < 0.7 {
            recommendations.push("Set smaller, achievable goals".to_string());
        }

        recommendations
    }

    fn determine_communication_style(
        &self,
        profile: &PersonalityProfile,
        context: &MessageContext,
    ) -> CommunicationStyle {
        match context.urgency {
            MessageUrgency::High => {
                if profile.personality_traits.conscientiousness > 0.7 {
                    CommunicationStyle::Direct
                } else {
                    CommunicationStyle::Supportive
                }
            }
            MessageUrgency::Medium => {
                if profile.personality_traits.openness > 0.7 {
                    CommunicationStyle::Analytical
                } else if profile.personality_traits.extroversion > 0.6 {
                    CommunicationStyle::Casual
                } else {
                    CommunicationStyle::Supportive
                }
            }
            MessageUrgency::Low => {
                if profile.personality_traits.extroversion > 0.7 {
                    CommunicationStyle::Motivational
                } else {
                    CommunicationStyle::Supportive
                }
            }
        }
    }

    fn calculate_template_effectiveness(
        &self,
        template: &MessageTemplate,
        profile: &PersonalityProfile,
    ) -> f32 {
        template
            .effectiveness_factors
            .iter()
            .map(|factor| match factor {
                PersonalityFactor::Extroversion => profile.personality_traits.extroversion,
                PersonalityFactor::Agreeableness => profile.personality_traits.agreeableness,
                PersonalityFactor::Conscientiousness => {
                    profile.personality_traits.conscientiousness
                }
                PersonalityFactor::Neuroticism => 1.0 - profile.personality_traits.neuroticism, // Inverted
                PersonalityFactor::Openness => profile.personality_traits.openness,
            })
            .sum::<f32>()
            / template.effectiveness_factors.len() as f32
    }

    fn personalize_content(&self, content: &str, context: &MessageContext) -> String {
        content
            .replace(
                "{improvement}",
                &format!("{:.1}", context.improvement_percentage.unwrap_or(0.0)),
            )
            .replace(
                "{metric}",
                &context
                    .metric_name
                    .clone()
                    .unwrap_or_else(|| "performance".to_string()),
            )
            .replace(
                "{percentage}",
                &format!("{:.1}", context.percentage_value.unwrap_or(0.0)),
            )
            .replace(
                "{timeframe}",
                &context
                    .timeframe
                    .clone()
                    .unwrap_or_else(|| "recent".to_string()),
            )
            .replace(
                "{area}",
                &context
                    .focus_area
                    .clone()
                    .unwrap_or_else(|| "practice".to_string()),
            )
            .replace(
                "{prediction}",
                &format!("{:.0}", context.prediction_value.unwrap_or(5.0)),
            )
    }

    fn detect_burnout_risk(&self, tracker: &MotivationTracker) -> bool {
        tracker.engagement_score < 0.3 && tracker.frustration_level > 0.7
    }

    fn detect_disengagement(&self, tracker: &MotivationTracker) -> bool {
        tracker.session_completion_rate < 0.5 && tracker.engagement_score < 0.4
    }

    fn detect_frustration_spike(&self, tracker: &MotivationTracker) -> bool {
        tracker.frustration_level > 0.8
            || (tracker.frustration_history.len() >= 3
                && tracker
                    .frustration_history
                    .iter()
                    .rev()
                    .take(3)
                    .all(|&f| f > 0.6))
    }

    fn create_burnout_intervention(
        &self,
        user_id: Uuid,
        profile: &PersonalityProfile,
    ) -> InterventionRecommendation {
        InterventionRecommendation {
            intervention_type: InterventionType::BurnoutPrevention,
            priority: InterventionPriority::High,
            suggested_actions: vec![
                "Reduce session frequency".to_string(),
                "Lower difficulty temporarily".to_string(),
                "Suggest taking a break".to_string(),
            ],
            message_content: Some(
                "You've been working hard! Consider taking a short break to recharge.".to_string(),
            ),
            duration: Some(chrono::Duration::days(3)),
        }
    }

    fn create_re_engagement_intervention(
        &self,
        user_id: Uuid,
        profile: &PersonalityProfile,
    ) -> InterventionRecommendation {
        InterventionRecommendation {
            intervention_type: InterventionType::ReEngagement,
            priority: InterventionPriority::Medium,
            suggested_actions: vec![
                "Introduce variety in exercises".to_string(),
                "Offer new challenges".to_string(),
                "Connect with peer groups".to_string(),
            ],
            message_content: Some(
                "Ready to try something new? We have some exciting challenges waiting for you!"
                    .to_string(),
            ),
            duration: Some(chrono::Duration::days(7)),
        }
    }

    fn create_frustration_intervention(
        &self,
        user_id: Uuid,
        profile: &PersonalityProfile,
    ) -> InterventionRecommendation {
        InterventionRecommendation {
            intervention_type: InterventionType::FrustrationReduction,
            priority: InterventionPriority::High,
            suggested_actions: vec![
                "Provide easier exercises".to_string(),
                "Offer additional guidance".to_string(),
                "Celebrate small wins".to_string(),
            ],
            message_content: Some("Let's take it step by step. You're making progress, even if it doesn't always feel like it!".to_string()),
            duration: Some(chrono::Duration::days(2)),
        }
    }
}

impl Default for MotivationSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Personality profile based on Big Five model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityProfile {
    /// Big Five personality traits
    pub personality_traits: PersonalityTraits,
    /// Communication preferences
    pub communication_preference: CommunicationPreference,
    /// Motivation factors
    pub motivation_factors: MotivationFactors,
}

/// Big Five personality traits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityTraits {
    /// Extroversion (0.0 to 1.0)
    pub extroversion: f32,
    /// Agreeableness (0.0 to 1.0)
    pub agreeableness: f32,
    /// Conscientiousness (0.0 to 1.0)
    pub conscientiousness: f32,
    /// Neuroticism (0.0 to 1.0)
    pub neuroticism: f32,
    /// Openness to experience (0.0 to 1.0)
    pub openness: f32,
}

/// Communication preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPreference {
    /// Preferred communication style
    pub preferred_style: CommunicationStyle,
    /// Desired feedback frequency
    pub feedback_frequency: FeedbackFrequency,
    /// Preferred message tone
    pub message_tone: MessageTone,
}

/// Communication styles
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommunicationStyle {
    /// Direct, concise communication
    Direct,
    /// Supportive, empathetic communication
    Supportive,
    /// Analytical, data-driven communication
    Analytical,
    /// Casual, informal communication
    Casual,
    /// Motivational, inspiring communication
    Motivational,
}

/// Feedback frequencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackFrequency {
    /// Immediate real-time feedback
    Immediate,
    /// Feedback after each session
    AfterSession,
    /// Daily summary feedback
    Daily,
    /// Weekly summary feedback
    Weekly,
}

/// Message tones
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageTone {
    /// Professional tone
    Professional,
    /// Friendly tone
    Friendly,
    /// Encouraging tone
    Encouraging,
    /// Humorous tone
    Humorous,
}

/// Motivation factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotivationFactors {
    /// Response to challenges
    pub challenge_response: ChallengeResponse,
    /// Social learning preference
    pub social_preference: SocialPreference,
    /// Achievement motivation type
    pub achievement_orientation: AchievementOrientation,
}

/// Challenge response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeResponse {
    /// Thrives on difficult challenges
    ThriveOnChallenge,
    /// Prefers gradual progression
    PreferGradualProgress,
    /// Avoids difficulty
    AvoidDifficulty,
}

/// Social preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SocialPreference {
    /// Prefers collaborative learning
    Collaborative,
    /// Prefers competitive environments
    Competitive,
    /// Prefers independent learning
    Independent,
}

/// Achievement orientations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AchievementOrientation {
    /// Motivated by skill mastery
    MasteryOriented,
    /// Motivated by performance outcomes
    PerformanceOriented,
    /// Motivated by social recognition
    SocialRecognition,
}

/// Motivation tracking data
#[derive(Debug, Clone)]
pub struct MotivationTracker {
    /// Current engagement score (0.0 - 1.0)
    pub engagement_score: f32,
    /// Current frustration level (0.0 - 1.0)
    pub frustration_level: f32,
    /// User satisfaction rating (0.0 - 1.0)
    pub satisfaction_rating: f32,
    /// Session completion rate (0.0 - 1.0)
    pub session_completion_rate: f32,
    /// Historical engagement scores
    pub engagement_history: Vec<f32>,
    /// Historical frustration levels
    pub frustration_history: Vec<f32>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

impl MotivationTracker {
    fn new() -> Self {
        Self {
            engagement_score: 0.5,
            frustration_level: 0.3,
            satisfaction_rating: 0.7,
            session_completion_rate: 0.8,
            engagement_history: Vec::new(),
            frustration_history: Vec::new(),
            last_updated: Utc::now(),
        }
    }
}

/// Session motivation data
#[derive(Debug, Clone)]
pub struct SessionMotivationData {
    /// Session engagement score
    pub engagement_score: f32,
    /// Session frustration level
    pub frustration_level: f32,
    /// Session satisfaction rating
    pub satisfaction_rating: f32,
    /// Session completion rate
    pub completion_rate: f32,
}

/// Motivation assessment result
#[derive(Debug, Clone)]
pub struct MotivationAssessment {
    /// Current motivation state
    pub current_state: MotivationState,
    /// Engagement trend direction
    pub engagement_trend: Trend,
    /// Frustration trend direction
    pub frustration_trend: Trend,
    /// Burnout risk level
    pub risk_level: RiskLevel,
    /// Recommended interventions
    pub recommended_actions: Vec<String>,
}

/// Motivation states
#[derive(Debug, Clone, PartialEq)]
pub enum MotivationState {
    /// Highly motivated
    Highly,
    /// Moderately motivated
    Moderately,
    /// Somewhat motivated
    Somewhat,
    /// Low motivation
    Low,
}

/// Trend directions
#[derive(Debug, Clone, PartialEq)]
pub enum Trend {
    /// Improving trend
    Improving,
    /// Stable trend
    Stable,
    /// Declining trend
    Declining,
}

/// Risk levels
#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
}

/// Message template
#[derive(Debug, Clone)]
pub struct MessageTemplate {
    /// Type of message
    pub message_type: MessageType,
    /// Message content template
    pub content: String,
    /// Personality factors that affect effectiveness
    pub effectiveness_factors: Vec<PersonalityFactor>,
}

/// Message types
#[derive(Debug, Clone, PartialEq)]
pub enum MessageType {
    /// Encouragement message
    Encouragement,
    /// Challenge message
    Challenge,
    /// Comfort message
    Comfort,
    /// Progress update message
    Progress,
    /// Insight message
    Insight,
    /// Celebration message
    Celebration,
    /// Inspiration message
    Inspiration,
}

/// Personality factors for message effectiveness
#[derive(Debug, Clone)]
pub enum PersonalityFactor {
    /// Extroversion factor
    Extroversion,
    /// Agreeableness factor
    Agreeableness,
    /// Conscientiousness factor
    Conscientiousness,
    /// Neuroticism factor
    Neuroticism,
    /// Openness factor
    Openness,
}

/// Message context for personalization
#[derive(Debug, Clone)]
pub struct MessageContext {
    /// Type of message
    pub message_type: MessageType,
    /// Message urgency
    pub urgency: MessageUrgency,
    /// Improvement percentage for template
    pub improvement_percentage: Option<f32>,
    /// Metric name for template
    pub metric_name: Option<String>,
    /// Percentage value for template
    pub percentage_value: Option<f32>,
    /// Timeframe for template
    pub timeframe: Option<String>,
    /// Focus area for template
    pub focus_area: Option<String>,
    /// Prediction value for template
    pub prediction_value: Option<f32>,
}

/// Message urgency levels
#[derive(Debug, Clone)]
pub enum MessageUrgency {
    /// Low urgency
    Low,
    /// Medium urgency
    Medium,
    /// High urgency
    High,
}

/// Personalized message
#[derive(Debug, Clone)]
pub struct PersonalizedMessage {
    /// Message content
    pub content: String,
    /// Communication style used
    pub communication_style: CommunicationStyle,
    /// Message type
    pub message_type: MessageType,
    /// Predicted effectiveness score
    pub effectiveness_score: f32,
    /// Message timestamp
    pub timestamp: DateTime<Utc>,
}

/// Intervention types
#[derive(Debug, Clone, PartialEq)]
pub enum InterventionType {
    /// Burnout prevention intervention
    BurnoutPrevention,
    /// Re-engagement intervention
    ReEngagement,
    /// Frustration reduction intervention
    FrustrationReduction,
    /// Motivation boost intervention
    MotivationBoost,
}

/// Intervention priorities
#[derive(Debug, Clone, PartialEq)]
pub enum InterventionPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Intervention recommendation
#[derive(Debug, Clone)]
pub struct InterventionRecommendation {
    /// Type of intervention
    pub intervention_type: InterventionType,
    /// Intervention priority
    pub priority: InterventionPriority,
    /// Suggested actions to take
    pub suggested_actions: Vec<String>,
    /// Optional message to user
    pub message_content: Option<String>,
    /// Recommended intervention duration
    pub duration: Option<chrono::Duration>,
}

/// Intervention record
#[derive(Debug, Clone)]
pub struct Intervention {
    /// Intervention identifier
    pub id: Uuid,
    /// Type of intervention
    pub intervention_type: InterventionType,
    /// Implementation timestamp
    pub implemented_at: DateTime<Utc>,
    /// Actions that were taken
    pub actions_taken: Vec<String>,
    /// Expected outcome description
    pub expected_outcome: String,
    /// Optional follow-up duration
    pub follow_up_duration: Option<chrono::Duration>,
}

/// Intervention result
#[derive(Debug, Clone)]
pub struct InterventionResult {
    /// Intervention identifier
    pub intervention_id: Uuid,
    /// Implementation timestamp
    pub implemented_at: DateTime<Utc>,
    /// Expected outcome description
    pub expected_outcome: String,
    /// Whether follow-up is required
    pub follow_up_required: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motivation_system_creation() {
        let system = MotivationSystem::new();
        assert!(!system.message_templates.is_empty());
    }

    #[test]
    fn test_personality_profile_setting() {
        let mut system = MotivationSystem::new();
        let user_id = Uuid::new_v4();

        let profile = PersonalityProfile {
            personality_traits: PersonalityTraits {
                extroversion: 0.7,
                agreeableness: 0.8,
                conscientiousness: 0.6,
                neuroticism: 0.3,
                openness: 0.7,
            },
            communication_preference: CommunicationPreference {
                preferred_style: CommunicationStyle::Supportive,
                feedback_frequency: FeedbackFrequency::Immediate,
                message_tone: MessageTone::Encouraging,
            },
            motivation_factors: MotivationFactors {
                challenge_response: ChallengeResponse::ThriveOnChallenge,
                social_preference: SocialPreference::Collaborative,
                achievement_orientation: AchievementOrientation::MasteryOriented,
            },
        };

        system.set_personality_profile(user_id, profile);
        assert!(system.personality_profiles.contains_key(&user_id));
        assert!(system.motivation_tracking.contains_key(&user_id));
    }

    #[test]
    fn test_motivation_update() {
        let mut system = MotivationSystem::new();
        let user_id = Uuid::new_v4();

        // Initialize tracker
        system
            .motivation_tracking
            .insert(user_id, MotivationTracker::new());

        let session_data = SessionMotivationData {
            engagement_score: 0.8,
            frustration_level: 0.2,
            satisfaction_rating: 0.9,
            completion_rate: 0.95,
        };

        let assessment = system.update_motivation(user_id, &session_data);
        assert_eq!(assessment.current_state, MotivationState::Highly);
        assert_eq!(assessment.risk_level, RiskLevel::Low);
    }

    #[test]
    fn test_personalized_message_generation() {
        let mut system = MotivationSystem::new();
        let user_id = Uuid::new_v4();

        let profile = PersonalityProfile {
            personality_traits: PersonalityTraits {
                extroversion: 0.5, // ≤ 0.6 to get Supportive style
                agreeableness: 0.8,
                conscientiousness: 0.6,
                neuroticism: 0.3,
                openness: 0.7, // ≤ 0.7 to avoid Analytical style
            },
            communication_preference: CommunicationPreference {
                preferred_style: CommunicationStyle::Supportive,
                feedback_frequency: FeedbackFrequency::Immediate,
                message_tone: MessageTone::Encouraging,
            },
            motivation_factors: MotivationFactors {
                challenge_response: ChallengeResponse::ThriveOnChallenge,
                social_preference: SocialPreference::Collaborative,
                achievement_orientation: AchievementOrientation::MasteryOriented,
            },
        };

        system.set_personality_profile(user_id, profile);

        let context = MessageContext {
            message_type: MessageType::Encouragement,
            urgency: MessageUrgency::Medium,
            improvement_percentage: Some(15.0),
            metric_name: Some("pronunciation accuracy".to_string()),
            percentage_value: None,
            timeframe: None,
            focus_area: None,
            prediction_value: None,
        };

        let message = system.generate_personalized_message(user_id, context);
        assert!(message.is_some());

        let message = message.unwrap();
        assert_eq!(message.communication_style, CommunicationStyle::Supportive);
        assert!(message.effectiveness_score > 0.0);
    }

    #[test]
    fn test_intervention_detection() {
        let mut system = MotivationSystem::new();
        let user_id = Uuid::new_v4();

        // Create tracker with high frustration
        let mut tracker = MotivationTracker::new();
        tracker.engagement_score = 0.2;
        tracker.frustration_level = 0.9;
        tracker.session_completion_rate = 0.4;

        system.motivation_tracking.insert(user_id, tracker);
        system.personality_profiles.insert(
            user_id,
            PersonalityProfile {
                personality_traits: PersonalityTraits {
                    extroversion: 0.5,
                    agreeableness: 0.6,
                    conscientiousness: 0.7,
                    neuroticism: 0.4,
                    openness: 0.6,
                },
                communication_preference: CommunicationPreference {
                    preferred_style: CommunicationStyle::Supportive,
                    feedback_frequency: FeedbackFrequency::Immediate,
                    message_tone: MessageTone::Encouraging,
                },
                motivation_factors: MotivationFactors {
                    challenge_response: ChallengeResponse::PreferGradualProgress,
                    social_preference: SocialPreference::Independent,
                    achievement_orientation: AchievementOrientation::MasteryOriented,
                },
            },
        );

        let intervention = system.check_intervention_needs(user_id);
        assert!(intervention.is_some());

        let intervention = intervention.unwrap();
        assert_eq!(intervention.priority, InterventionPriority::High);
    }
}
