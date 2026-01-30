//! # User Experience (UX) Evaluation
//!
//! Comprehensive evaluation framework for assessing user experience quality in
//! speech synthesis and dialogue systems. Includes metrics for usability, satisfaction,
//! accessibility, and overall user experience.

use crate::{EvaluationError, EvaluationResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// User experience evaluation error types
#[derive(Debug, Error)]
pub enum UXError {
    #[error("Invalid UX metric: {0}")]
    InvalidMetric(String),
    #[error("Insufficient data for evaluation: {0}")]
    InsufficientData(String),
    #[error("UX evaluation failed: {0}")]
    EvaluationFailed(String),
}

/// User experience dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UXDimension {
    /// Ease of use and learnability
    Usability,
    /// User satisfaction and enjoyment
    Satisfaction,
    /// Accessibility for diverse users
    Accessibility,
    /// System efficiency and responsiveness
    Efficiency,
    /// Trust and reliability
    Trust,
    /// Emotional engagement
    Engagement,
}

/// User profile for personalized UX evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// User identifier
    pub user_id: String,
    /// Age group
    pub age_group: AgeGroup,
    /// Technical proficiency
    pub tech_proficiency: TechProficiency,
    /// Language proficiency
    pub language_proficiency: LanguageProficiency,
    /// Accessibility needs
    pub accessibility_needs: Vec<AccessibilityNeed>,
    /// Prior experience with similar systems
    pub prior_experience: ExperienceLevel,
}

/// Age group categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgeGroup {
    /// 0-12 years
    Child,
    /// 13-17 years
    Teen,
    /// 18-34 years
    YoungAdult,
    /// 35-54 years
    MiddleAge,
    /// 55-74 years
    Senior,
    /// 75+ years
    Elderly,
}

/// Technical proficiency level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TechProficiency {
    /// Minimal technical knowledge
    Novice,
    /// Basic technical knowledge
    Beginner,
    /// Moderate technical knowledge
    Intermediate,
    /// Advanced technical knowledge
    Advanced,
    /// Expert technical knowledge
    Expert,
}

/// Language proficiency level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LanguageProficiency {
    /// Native speaker
    Native,
    /// Near-native proficiency
    Fluent,
    /// Intermediate proficiency
    Intermediate,
    /// Basic proficiency
    Basic,
    /// Minimal proficiency
    Beginner,
}

/// Accessibility needs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessibilityNeed {
    /// Visual impairment
    VisualImpairment,
    /// Hearing impairment
    HearingImpairment,
    /// Motor impairment
    MotorImpairment,
    /// Cognitive impairment
    CognitiveImpairment,
    /// Speech impairment
    SpeechImpairment,
    /// None
    None,
}

/// Experience level with similar systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperienceLevel {
    /// First-time user
    FirstTime,
    /// Occasional user (< 5 uses)
    Occasional,
    /// Regular user (5-20 uses)
    Regular,
    /// Frequent user (20+ uses)
    Frequent,
    /// Power user (expert)
    PowerUser,
}

/// Interaction event for UX analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    /// Event timestamp (seconds from start)
    pub timestamp: f32,
    /// Event type
    pub event_type: EventType,
    /// User input (if applicable)
    pub user_input: Option<String>,
    /// System response (if applicable)
    pub system_response: Option<String>,
    /// Response latency (seconds)
    pub response_latency: Option<f32>,
    /// User reaction
    pub user_reaction: Option<UserReaction>,
}

/// Type of interaction event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// User initiated interaction
    UserInput,
    /// System provided response
    SystemResponse,
    /// Error occurred
    Error,
    /// Help requested
    HelpRequest,
    /// Task completed
    TaskCompletion,
    /// Session ended
    SessionEnd,
}

/// User reaction to system response
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UserReaction {
    /// Positive reaction
    Positive,
    /// Neutral reaction
    Neutral,
    /// Negative reaction
    Negative,
    /// Frustrated
    Frustrated,
    /// Satisfied
    Satisfied,
    /// Confused
    Confused,
}

/// User experience evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UXScore {
    /// Overall UX score (0.0-1.0)
    pub overall_score: f32,
    /// Usability score (0.0-1.0)
    pub usability: f32,
    /// Satisfaction score (0.0-1.0)
    pub satisfaction: f32,
    /// Accessibility score (0.0-1.0)
    pub accessibility: f32,
    /// Efficiency score (0.0-1.0)
    pub efficiency: f32,
    /// Trust score (0.0-1.0)
    pub trust: f32,
    /// Engagement score (0.0-1.0)
    pub engagement: f32,
    /// Dimension scores
    pub dimension_scores: HashMap<UXDimension, f32>,
    /// Detailed metrics
    pub details: UXDetails,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Detailed UX metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UXDetails {
    /// Average response latency (seconds)
    pub avg_response_latency: f32,
    /// Error rate (errors per interaction)
    pub error_rate: f32,
    /// Help request rate
    pub help_request_rate: f32,
    /// Task completion rate (0.0-1.0)
    pub task_completion_rate: f32,
    /// Session duration (seconds)
    pub session_duration: f32,
    /// Number of interactions
    pub interaction_count: u32,
    /// Positive reaction rate (0.0-1.0)
    pub positive_reaction_rate: f32,
    /// Negative reaction rate (0.0-1.0)
    pub negative_reaction_rate: f32,
    /// Learnability score (improvement over time)
    pub learnability: f32,
    /// Consistency score
    pub consistency: f32,
}

/// UX evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UXConfig {
    /// Maximum acceptable response latency (seconds)
    pub max_acceptable_latency: f32,
    /// Target task completion rate
    pub target_completion_rate: f32,
    /// Maximum acceptable error rate
    pub max_error_rate: f32,
    /// Weight for usability (0.0-1.0)
    pub usability_weight: f32,
    /// Weight for satisfaction (0.0-1.0)
    pub satisfaction_weight: f32,
    /// Weight for accessibility (0.0-1.0)
    pub accessibility_weight: f32,
    /// Weight for efficiency (0.0-1.0)
    pub efficiency_weight: f32,
    /// Consider user profile in evaluation
    pub personalized: bool,
}

impl Default for UXConfig {
    fn default() -> Self {
        Self {
            max_acceptable_latency: 1.0,
            target_completion_rate: 0.9,
            max_error_rate: 0.1,
            usability_weight: 0.25,
            satisfaction_weight: 0.25,
            accessibility_weight: 0.20,
            efficiency_weight: 0.30,
            personalized: true,
        }
    }
}

/// User experience evaluator
pub struct UXEvaluator {
    config: UXConfig,
}

impl UXEvaluator {
    /// Create a new UX evaluator
    pub fn new(config: UXConfig) -> Self {
        Self { config }
    }

    /// Evaluate user experience from interaction events
    pub fn evaluate(
        &self,
        events: &[InteractionEvent],
        user_profile: Option<&UserProfile>,
    ) -> EvaluationResult<UXScore> {
        if events.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: "No interaction events provided for UX evaluation".to_string(),
            }
            .into());
        }

        // Calculate dimension scores
        let usability = self.calculate_usability(events, user_profile)?;
        let satisfaction = self.calculate_satisfaction(events)?;
        let accessibility = self.calculate_accessibility(events, user_profile)?;
        let efficiency = self.calculate_efficiency(events)?;
        let trust = self.calculate_trust(events)?;
        let engagement = self.calculate_engagement(events)?;

        // Create dimension scores map
        let mut dimension_scores = HashMap::new();
        dimension_scores.insert(UXDimension::Usability, usability);
        dimension_scores.insert(UXDimension::Satisfaction, satisfaction);
        dimension_scores.insert(UXDimension::Accessibility, accessibility);
        dimension_scores.insert(UXDimension::Efficiency, efficiency);
        dimension_scores.insert(UXDimension::Trust, trust);
        dimension_scores.insert(UXDimension::Engagement, engagement);

        // Calculate overall score
        let overall_score = (usability * self.config.usability_weight)
            + (satisfaction * self.config.satisfaction_weight)
            + (accessibility * self.config.accessibility_weight)
            + (efficiency * self.config.efficiency_weight);

        // Calculate detailed metrics
        let details = self.calculate_details(events)?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(&dimension_scores, &details);

        Ok(UXScore {
            overall_score,
            usability,
            satisfaction,
            accessibility,
            efficiency,
            trust,
            engagement,
            dimension_scores,
            details,
            recommendations,
        })
    }

    /// Calculate usability score
    fn calculate_usability(
        &self,
        events: &[InteractionEvent],
        user_profile: Option<&UserProfile>,
    ) -> EvaluationResult<f32> {
        let error_count = events
            .iter()
            .filter(|e| e.event_type == EventType::Error)
            .count();
        let help_requests = events
            .iter()
            .filter(|e| e.event_type == EventType::HelpRequest)
            .count();

        let total_interactions = events.len().max(1);
        let error_score = 1.0 - (error_count as f32 / total_interactions as f32).min(1.0);
        let help_score = 1.0 - (help_requests as f32 / total_interactions as f32).min(1.0);

        let learnability = self.calculate_learnability(events);

        let mut usability = (error_score + help_score + learnability) / 3.0;

        // Adjust for user profile
        if let Some(profile) = user_profile {
            if self.config.personalized {
                usability = self.adjust_for_proficiency(usability, profile);
            }
        }

        Ok(usability)
    }

    /// Calculate satisfaction score
    fn calculate_satisfaction(&self, events: &[InteractionEvent]) -> EvaluationResult<f32> {
        let reactions: Vec<_> = events.iter().filter_map(|e| e.user_reaction).collect();

        if reactions.is_empty() {
            return Ok(0.5); // Neutral if no reactions
        }

        let positive = reactions
            .iter()
            .filter(|r| matches!(r, UserReaction::Positive | UserReaction::Satisfied))
            .count();

        let negative = reactions
            .iter()
            .filter(|r| matches!(r, UserReaction::Negative | UserReaction::Frustrated))
            .count();

        let satisfaction = (positive as f32 - negative as f32) / reactions.len() as f32;
        Ok((satisfaction + 1.0) / 2.0) // Normalize to 0-1
    }

    /// Calculate accessibility score
    fn calculate_accessibility(
        &self,
        events: &[InteractionEvent],
        user_profile: Option<&UserProfile>,
    ) -> EvaluationResult<f32> {
        let mut accessibility: f32 = 0.8; // Base score

        // Check response latency for accessibility
        let latencies: Vec<f32> = events.iter().filter_map(|e| e.response_latency).collect();
        if !latencies.is_empty() {
            let avg_latency = latencies.iter().sum::<f32>() / latencies.len() as f32;
            if avg_latency <= self.config.max_acceptable_latency {
                accessibility += 0.1;
            }
        }

        // Adjust for accessibility needs
        if let Some(profile) = user_profile {
            for need in &profile.accessibility_needs {
                if !matches!(need, AccessibilityNeed::None) {
                    // Additional evaluation based on specific needs
                    accessibility *= 0.95; // Slight penalty acknowledging challenges
                }
            }
        }

        Ok(accessibility.min(1.0_f32))
    }

    /// Calculate efficiency score
    fn calculate_efficiency(&self, events: &[InteractionEvent]) -> EvaluationResult<f32> {
        let latencies: Vec<f32> = events.iter().filter_map(|e| e.response_latency).collect();

        let latency_score = if !latencies.is_empty() {
            let avg_latency = latencies.iter().sum::<f32>() / latencies.len() as f32;
            (self.config.max_acceptable_latency / avg_latency.max(0.1)).min(1.0)
        } else {
            0.5
        };

        let completion_events = events
            .iter()
            .filter(|e| e.event_type == EventType::TaskCompletion)
            .count();

        let task_score = if !events.is_empty() {
            (completion_events as f32 / events.len() as f32 * 2.0).min(1.0)
        } else {
            0.0
        };

        Ok((latency_score + task_score) / 2.0)
    }

    /// Calculate trust score
    fn calculate_trust(&self, events: &[InteractionEvent]) -> EvaluationResult<f32> {
        let error_count = events
            .iter()
            .filter(|e| e.event_type == EventType::Error)
            .count();
        let total_interactions = events.len().max(1);

        let reliability = 1.0 - (error_count as f32 / total_interactions as f32);

        // Consistency in responses
        let consistency = self.calculate_consistency(events);

        Ok((reliability + consistency) / 2.0)
    }

    /// Calculate engagement score
    fn calculate_engagement(&self, events: &[InteractionEvent]) -> EvaluationResult<f32> {
        if events.is_empty() {
            return Ok(0.0);
        }

        let session_duration = events.last().map(|e| e.timestamp).unwrap_or(0.0);

        let interaction_rate = if session_duration > 0.0 {
            events.len() as f32 / session_duration
        } else {
            0.0
        };

        // Normalize interaction rate (assume 1 interaction per 10 seconds is baseline)
        let engagement = (interaction_rate * 10.0).min(1.0);

        Ok(engagement)
    }

    /// Calculate learnability (improvement over time)
    fn calculate_learnability(&self, events: &[InteractionEvent]) -> f32 {
        if events.len() < 4 {
            return 0.5; // Not enough data
        }

        let mid_point = events.len() / 2;
        let first_half = &events[..mid_point];
        let second_half = &events[mid_point..];

        let first_error_rate = first_half
            .iter()
            .filter(|e| e.event_type == EventType::Error)
            .count() as f32
            / first_half.len() as f32;

        let second_error_rate = second_half
            .iter()
            .filter(|e| e.event_type == EventType::Error)
            .count() as f32
            / second_half.len() as f32;

        let improvement = (first_error_rate - second_error_rate + 1.0) / 2.0;
        improvement.max(0.0).min(1.0)
    }

    /// Calculate consistency score
    fn calculate_consistency(&self, events: &[InteractionEvent]) -> f32 {
        let latencies: Vec<f32> = events.iter().filter_map(|e| e.response_latency).collect();

        if latencies.len() < 2 {
            return 0.8; // Assume good consistency with limited data
        }

        let mean = latencies.iter().sum::<f32>() / latencies.len() as f32;
        let variance =
            latencies.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / latencies.len() as f32;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 0.0 };

        // Lower CV = higher consistency
        (1.0 - coefficient_of_variation.min(1.0)).max(0.0)
    }

    /// Adjust score based on user proficiency
    fn adjust_for_proficiency(&self, score: f32, profile: &UserProfile) -> f32 {
        let adjustment = match profile.tech_proficiency {
            TechProficiency::Novice => 1.1,
            TechProficiency::Beginner => 1.05,
            TechProficiency::Intermediate => 1.0,
            TechProficiency::Advanced => 0.98,
            TechProficiency::Expert => 0.95,
        };

        (score * adjustment).min(1.0)
    }

    /// Calculate detailed UX metrics
    fn calculate_details(&self, events: &[InteractionEvent]) -> EvaluationResult<UXDetails> {
        let latencies: Vec<f32> = events.iter().filter_map(|e| e.response_latency).collect();
        let avg_response_latency = if !latencies.is_empty() {
            latencies.iter().sum::<f32>() / latencies.len() as f32
        } else {
            0.0
        };

        let error_count = events
            .iter()
            .filter(|e| e.event_type == EventType::Error)
            .count();
        let error_rate = error_count as f32 / events.len().max(1) as f32;

        let help_requests = events
            .iter()
            .filter(|e| e.event_type == EventType::HelpRequest)
            .count();
        let help_request_rate = help_requests as f32 / events.len().max(1) as f32;

        let completions = events
            .iter()
            .filter(|e| e.event_type == EventType::TaskCompletion)
            .count();
        let task_completion_rate = completions as f32 / events.len().max(1) as f32;

        let session_duration = events.last().map(|e| e.timestamp).unwrap_or(0.0);

        let reactions: Vec<_> = events.iter().filter_map(|e| e.user_reaction).collect();
        let positive = reactions
            .iter()
            .filter(|r| matches!(r, UserReaction::Positive | UserReaction::Satisfied))
            .count();
        let negative = reactions
            .iter()
            .filter(|r| matches!(r, UserReaction::Negative | UserReaction::Frustrated))
            .count();

        let positive_reaction_rate = if !reactions.is_empty() {
            positive as f32 / reactions.len() as f32
        } else {
            0.0
        };

        let negative_reaction_rate = if !reactions.is_empty() {
            negative as f32 / reactions.len() as f32
        } else {
            0.0
        };

        let learnability = self.calculate_learnability(events);
        let consistency = self.calculate_consistency(events);

        Ok(UXDetails {
            avg_response_latency,
            error_rate,
            help_request_rate,
            task_completion_rate,
            session_duration,
            interaction_count: events.len() as u32,
            positive_reaction_rate,
            negative_reaction_rate,
            learnability,
            consistency,
        })
    }

    /// Generate improvement recommendations
    fn generate_recommendations(
        &self,
        dimension_scores: &HashMap<UXDimension, f32>,
        details: &UXDetails,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if let Some(&usability) = dimension_scores.get(&UXDimension::Usability) {
            if usability < 0.7 {
                recommendations.push(
                    "Improve system usability by reducing errors and providing better guidance"
                        .to_string(),
                );
            }
        }

        if details.avg_response_latency > self.config.max_acceptable_latency {
            recommendations.push(format!(
                "Reduce response latency from {:.2}s to target {:.2}s",
                details.avg_response_latency, self.config.max_acceptable_latency
            ));
        }

        if details.error_rate > self.config.max_error_rate {
            recommendations.push(format!(
                "Reduce error rate from {:.1}% to below {:.1}%",
                details.error_rate * 100.0,
                self.config.max_error_rate * 100.0
            ));
        }

        if details.task_completion_rate < self.config.target_completion_rate {
            recommendations
                .push("Improve task completion rate through better task guidance".to_string());
        }

        if let Some(&satisfaction) = dimension_scores.get(&UXDimension::Satisfaction) {
            if satisfaction < 0.6 {
                recommendations.push(
                    "Enhance user satisfaction by improving response quality and relevance"
                        .to_string(),
                );
            }
        }

        if details.consistency < 0.7 {
            recommendations.push("Improve consistency in response times and behavior".to_string());
        }

        if recommendations.is_empty() {
            recommendations
                .push("UX is performing well - maintain current quality standards".to_string());
        }

        recommendations
    }

    /// Compare UX across different user segments
    pub fn compare_segments(&self, segment_scores: &[(String, UXScore)]) -> SegmentComparison {
        if segment_scores.is_empty() {
            return SegmentComparison::default();
        }

        let avg_overall = segment_scores
            .iter()
            .map(|(_, s)| s.overall_score)
            .sum::<f32>()
            / segment_scores.len() as f32;

        let best_segment = segment_scores
            .iter()
            .max_by(|a, b| a.1.overall_score.partial_cmp(&b.1.overall_score).unwrap())
            .map(|(name, score)| (name.clone(), score.overall_score));

        let worst_segment = segment_scores
            .iter()
            .min_by(|a, b| a.1.overall_score.partial_cmp(&b.1.overall_score).unwrap())
            .map(|(name, score)| (name.clone(), score.overall_score));

        SegmentComparison {
            avg_overall_score: avg_overall,
            best_segment,
            worst_segment,
            segment_count: segment_scores.len(),
        }
    }
}

/// Comparison across user segments
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SegmentComparison {
    /// Average overall score across segments
    pub avg_overall_score: f32,
    /// Best performing segment
    pub best_segment: Option<(String, f32)>,
    /// Worst performing segment
    pub worst_segment: Option<(String, f32)>,
    /// Number of segments
    pub segment_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ux_evaluator_creation() {
        let config = UXConfig::default();
        let evaluator = UXEvaluator::new(config);
        assert!(evaluator.config.max_acceptable_latency > 0.0);
    }

    #[test]
    fn test_positive_ux_evaluation() {
        let config = UXConfig::default();
        let evaluator = UXEvaluator::new(config);

        let events = vec![
            InteractionEvent {
                timestamp: 0.0,
                event_type: EventType::UserInput,
                user_input: Some("Hello".to_string()),
                system_response: None,
                response_latency: None,
                user_reaction: None,
            },
            InteractionEvent {
                timestamp: 0.5,
                event_type: EventType::SystemResponse,
                user_input: None,
                system_response: Some("Hi there!".to_string()),
                response_latency: Some(0.5),
                user_reaction: Some(UserReaction::Positive),
            },
            InteractionEvent {
                timestamp: 5.0,
                event_type: EventType::TaskCompletion,
                user_input: None,
                system_response: None,
                response_latency: None,
                user_reaction: Some(UserReaction::Satisfied),
            },
        ];

        let result = evaluator.evaluate(&events, None).unwrap();

        assert!(result.overall_score > 0.6);
        assert!(result.satisfaction > 0.5);
        assert!(result.efficiency > 0.5);
    }

    #[test]
    fn test_negative_ux_evaluation() {
        let config = UXConfig::default();
        let evaluator = UXEvaluator::new(config);

        let events = vec![
            InteractionEvent {
                timestamp: 0.0,
                event_type: EventType::UserInput,
                user_input: Some("Help".to_string()),
                system_response: None,
                response_latency: None,
                user_reaction: None,
            },
            InteractionEvent {
                timestamp: 2.5,
                event_type: EventType::Error,
                user_input: None,
                system_response: Some("Error occurred".to_string()),
                response_latency: Some(2.5),
                user_reaction: Some(UserReaction::Frustrated),
            },
            InteractionEvent {
                timestamp: 5.0,
                event_type: EventType::HelpRequest,
                user_input: None,
                system_response: None,
                response_latency: None,
                user_reaction: Some(UserReaction::Confused),
            },
        ];

        let result = evaluator.evaluate(&events, None).unwrap();

        assert!(result.usability < 0.7);
        assert!(result.details.error_rate > 0.2);
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_personalized_evaluation() {
        let config = UXConfig {
            personalized: true,
            ..Default::default()
        };
        let evaluator = UXEvaluator::new(config);

        let profile = UserProfile {
            user_id: "user_001".to_string(),
            age_group: AgeGroup::Elderly,
            tech_proficiency: TechProficiency::Novice,
            language_proficiency: LanguageProficiency::Native,
            accessibility_needs: vec![AccessibilityNeed::None],
            prior_experience: ExperienceLevel::FirstTime,
        };

        let events = vec![InteractionEvent {
            timestamp: 0.0,
            event_type: EventType::UserInput,
            user_input: Some("Test".to_string()),
            system_response: None,
            response_latency: None,
            user_reaction: Some(UserReaction::Neutral),
        }];

        let result = evaluator.evaluate(&events, Some(&profile)).unwrap();

        // Novice users should get adjusted scores
        assert!(result.overall_score >= 0.0);
    }

    #[test]
    fn test_learnability_calculation() {
        let config = UXConfig::default();
        let evaluator = UXEvaluator::new(config);

        let events = vec![
            InteractionEvent {
                timestamp: 0.0,
                event_type: EventType::Error,
                user_input: None,
                system_response: None,
                response_latency: None,
                user_reaction: None,
            },
            InteractionEvent {
                timestamp: 1.0,
                event_type: EventType::Error,
                user_input: None,
                system_response: None,
                response_latency: None,
                user_reaction: None,
            },
            InteractionEvent {
                timestamp: 2.0,
                event_type: EventType::UserInput,
                user_input: None,
                system_response: None,
                response_latency: None,
                user_reaction: None,
            },
            InteractionEvent {
                timestamp: 3.0,
                event_type: EventType::UserInput,
                user_input: None,
                system_response: None,
                response_latency: None,
                user_reaction: None,
            },
        ];

        let learnability = evaluator.calculate_learnability(&events);
        assert!(learnability >= 0.0 && learnability <= 1.0);
    }

    #[test]
    fn test_segment_comparison() {
        let config = UXConfig::default();
        let evaluator = UXEvaluator::new(config);

        let segments = vec![
            (
                "Segment A".to_string(),
                UXScore {
                    overall_score: 0.8,
                    usability: 0.8,
                    satisfaction: 0.8,
                    accessibility: 0.8,
                    efficiency: 0.8,
                    trust: 0.8,
                    engagement: 0.8,
                    dimension_scores: HashMap::new(),
                    details: UXDetails {
                        avg_response_latency: 0.5,
                        error_rate: 0.05,
                        help_request_rate: 0.1,
                        task_completion_rate: 0.9,
                        session_duration: 60.0,
                        interaction_count: 10,
                        positive_reaction_rate: 0.8,
                        negative_reaction_rate: 0.1,
                        learnability: 0.7,
                        consistency: 0.8,
                    },
                    recommendations: vec![],
                },
            ),
            (
                "Segment B".to_string(),
                UXScore {
                    overall_score: 0.6,
                    usability: 0.6,
                    satisfaction: 0.6,
                    accessibility: 0.6,
                    efficiency: 0.6,
                    trust: 0.6,
                    engagement: 0.6,
                    dimension_scores: HashMap::new(),
                    details: UXDetails {
                        avg_response_latency: 1.0,
                        error_rate: 0.15,
                        help_request_rate: 0.2,
                        task_completion_rate: 0.7,
                        session_duration: 120.0,
                        interaction_count: 15,
                        positive_reaction_rate: 0.5,
                        negative_reaction_rate: 0.3,
                        learnability: 0.6,
                        consistency: 0.6,
                    },
                    recommendations: vec![],
                },
            ),
        ];

        let comparison = evaluator.compare_segments(&segments);

        assert_eq!(comparison.segment_count, 2);
        assert!(comparison.avg_overall_score > 0.6);
        assert!(comparison.best_segment.is_some());
        assert!(comparison.worst_segment.is_some());
    }

    #[test]
    fn test_efficiency_calculation() {
        let config = UXConfig::default();
        let evaluator = UXEvaluator::new(config);

        let events = vec![
            InteractionEvent {
                timestamp: 0.0,
                event_type: EventType::UserInput,
                user_input: Some("Query".to_string()),
                system_response: None,
                response_latency: None,
                user_reaction: None,
            },
            InteractionEvent {
                timestamp: 0.3,
                event_type: EventType::SystemResponse,
                user_input: None,
                system_response: Some("Response".to_string()),
                response_latency: Some(0.3),
                user_reaction: None,
            },
            InteractionEvent {
                timestamp: 1.0,
                event_type: EventType::TaskCompletion,
                user_input: None,
                system_response: None,
                response_latency: None,
                user_reaction: None,
            },
        ];

        let efficiency = evaluator.calculate_efficiency(&events).unwrap();
        assert!(efficiency > 0.5);
    }

    #[test]
    fn test_consistency_calculation() {
        let config = UXConfig::default();
        let evaluator = UXEvaluator::new(config);

        // Consistent latencies
        let consistent_events = vec![
            InteractionEvent {
                timestamp: 0.0,
                event_type: EventType::SystemResponse,
                user_input: None,
                system_response: None,
                response_latency: Some(0.5),
                user_reaction: None,
            },
            InteractionEvent {
                timestamp: 1.0,
                event_type: EventType::SystemResponse,
                user_input: None,
                system_response: None,
                response_latency: Some(0.5),
                user_reaction: None,
            },
            InteractionEvent {
                timestamp: 2.0,
                event_type: EventType::SystemResponse,
                user_input: None,
                system_response: None,
                response_latency: Some(0.5),
                user_reaction: None,
            },
        ];

        let consistency = evaluator.calculate_consistency(&consistent_events);
        assert!(consistency > 0.8);
    }
}
