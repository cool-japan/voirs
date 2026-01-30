//! Cultural Adaptation System
//!
//! This module provides cultural adaptation features for the `VoiRS` feedback system,
//! ensuring that feedback messages, interactions, and UI elements are culturally
//! appropriate and sensitive to different cultural contexts and norms.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Cultural adaptation errors
#[derive(Error, Debug, Clone)]
pub enum CulturalError {
    /// Culture not supported
    #[error("Culture '{culture}' is not supported")]
    CultureNotSupported {
        /// Culture code that is not supported
        culture: String,
    },

    /// Cultural rule not found
    #[error("Cultural rule '{rule}' not found for culture '{culture}'")]
    RuleNotFound {
        /// Rule identifier
        rule: String,
        /// Culture code
        culture: String,
    },

    /// Inappropriate content detected
    #[error("Content contains culturally inappropriate elements for '{culture}': {reason}")]
    InappropriateContent {
        /// Culture code
        culture: String,
        /// Reason for inappropriateness
        reason: String,
    },
}

/// Result type for cultural operations
pub type CulturalResult<T> = Result<T, CulturalError>;

/// Cultural context for different regions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CulturalContext {
    /// Western culture (US, UK, Western Europe)
    Western,
    /// East Asian culture (China, Japan, Korea)
    EastAsian,
    /// South Asian culture (India, Pakistan, Bangladesh)
    SouthAsian,
    /// Middle Eastern culture (Arab countries, Iran)
    MiddleEastern,
    /// African culture
    African,
    /// Latin American culture
    LatinAmerican,
    /// Southeast Asian culture (Thailand, Vietnam, Indonesia)
    SoutheastAsian,
    /// Custom culture
    Custom(String),
}

impl CulturalContext {
    /// Get culture code
    #[must_use]
    pub fn code(&self) -> &str {
        match self {
            CulturalContext::Western => "western",
            CulturalContext::EastAsian => "east-asian",
            CulturalContext::SouthAsian => "south-asian",
            CulturalContext::MiddleEastern => "middle-eastern",
            CulturalContext::African => "african",
            CulturalContext::LatinAmerican => "latin-american",
            CulturalContext::SoutheastAsian => "southeast-asian",
            CulturalContext::Custom(code) => code,
        }
    }

    /// Get culture name
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            CulturalContext::Western => "Western",
            CulturalContext::EastAsian => "East Asian",
            CulturalContext::SouthAsian => "South Asian",
            CulturalContext::MiddleEastern => "Middle Eastern",
            CulturalContext::African => "African",
            CulturalContext::LatinAmerican => "Latin American",
            CulturalContext::SoutheastAsian => "Southeast Asian",
            CulturalContext::Custom(name) => name,
        }
    }
}

/// Communication style preferences
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommunicationStyle {
    /// Direct and explicit communication
    Direct,
    /// Indirect and implicit communication
    Indirect,
    /// Formal communication
    Formal,
    /// Casual communication
    Casual,
    /// Hierarchical (respect for authority)
    Hierarchical,
    /// Egalitarian (equal status)
    Egalitarian,
}

/// Feedback tone preferences
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackTone {
    /// Encouraging and positive
    Encouraging,
    /// Neutral and objective
    Neutral,
    /// Critical and analytical
    Critical,
    /// Gentle and supportive
    Gentle,
    /// Enthusiastic and energetic
    Enthusiastic,
}

/// Time perception style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimePerception {
    /// Monochronic (time is linear, punctuality important)
    Monochronic,
    /// Polychronic (time is flexible, relationships > schedules)
    Polychronic,
}

/// Personal space preferences
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PersonalSpace {
    /// Large personal space (1.2m+)
    Large,
    /// Medium personal space (0.6-1.2m)
    Medium,
    /// Small personal space (<0.6m)
    Small,
}

/// Gesture appropriateness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureRule {
    /// Gesture description
    pub name: String,
    /// Whether gesture is appropriate
    pub is_appropriate: bool,
    /// Alternative if inappropriate
    pub alternative: Option<String>,
    /// Explanation
    pub explanation: String,
}

/// Cultural colors and their meanings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorMeaning {
    /// Color code (hex)
    pub color: String,
    /// Positive associations
    pub positive_meanings: Vec<String>,
    /// Negative associations
    pub negative_meanings: Vec<String>,
    /// Usage recommendations
    pub usage: String,
}

/// Cultural taboos and sensitivities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalTaboo {
    /// Topic or action
    pub topic: String,
    /// Severity (1-10, 10 = highly offensive)
    pub severity: u8,
    /// Explanation
    pub explanation: String,
    /// Alternative approach
    pub alternative: Option<String>,
}

/// Greeting style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GreetingStyle {
    /// Formal greeting
    pub formal: String,
    /// Informal greeting
    pub informal: String,
    /// Time-of-day specific greetings
    pub time_based: HashMap<String, String>,
    /// Physical contact appropriateness
    pub contact_appropriate: bool,
    /// Bow/handshake preference
    pub gesture: String,
}

/// Cultural profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalProfile {
    /// Cultural context
    pub context: CulturalContext,
    /// Communication style
    pub communication_style: CommunicationStyle,
    /// Preferred feedback tone
    pub feedback_tone: FeedbackTone,
    /// Time perception
    pub time_perception: TimePerception,
    /// Personal space preference
    pub personal_space: PersonalSpace,
    /// Gestures
    pub gestures: Vec<GestureRule>,
    /// Color meanings
    pub colors: Vec<ColorMeaning>,
    /// Cultural taboos
    pub taboos: Vec<CulturalTaboo>,
    /// Greeting styles
    pub greetings: GreetingStyle,
    /// Honor/respect indicators
    pub uses_honorifics: bool,
    /// Collectivist vs individualist
    pub is_collectivist: bool,
    /// High context vs low context communication
    pub is_high_context: bool,
}

impl CulturalProfile {
    /// Create a Western cultural profile
    #[must_use]
    pub fn western() -> Self {
        Self {
            context: CulturalContext::Western,
            communication_style: CommunicationStyle::Direct,
            feedback_tone: FeedbackTone::Encouraging,
            time_perception: TimePerception::Monochronic,
            personal_space: PersonalSpace::Large,
            gestures: vec![
                GestureRule {
                    name: "Thumbs up".to_string(),
                    is_appropriate: true,
                    alternative: None,
                    explanation: "Positive gesture indicating approval".to_string(),
                },
                GestureRule {
                    name: "OK sign".to_string(),
                    is_appropriate: true,
                    alternative: None,
                    explanation: "Indicates agreement or approval".to_string(),
                },
            ],
            colors: vec![
                ColorMeaning {
                    color: "#FF0000".to_string(),
                    positive_meanings: vec!["Passion".to_string(), "Energy".to_string()],
                    negative_meanings: vec!["Danger".to_string(), "Error".to_string()],
                    usage: "Use for alerts and important notifications".to_string(),
                },
                ColorMeaning {
                    color: "#00FF00".to_string(),
                    positive_meanings: vec!["Success".to_string(), "Growth".to_string()],
                    negative_meanings: vec![],
                    usage: "Use for success messages and positive feedback".to_string(),
                },
            ],
            taboos: vec![CulturalTaboo {
                topic: "Direct criticism in public".to_string(),
                severity: 3,
                explanation:
                    "Can be embarrassing but generally acceptable in professional contexts"
                        .to_string(),
                alternative: Some("Private feedback with constructive framing".to_string()),
            }],
            greetings: GreetingStyle {
                formal: "Good morning/afternoon/evening".to_string(),
                informal: "Hi/Hello".to_string(),
                time_based: [
                    ("morning".to_string(), "Good morning".to_string()),
                    ("afternoon".to_string(), "Good afternoon".to_string()),
                    ("evening".to_string(), "Good evening".to_string()),
                ]
                .iter()
                .cloned()
                .collect(),
                contact_appropriate: true,
                gesture: "Handshake".to_string(),
            },
            uses_honorifics: false,
            is_collectivist: false,
            is_high_context: false,
        }
    }

    /// Create an East Asian cultural profile
    #[must_use]
    pub fn east_asian() -> Self {
        Self {
            context: CulturalContext::EastAsian,
            communication_style: CommunicationStyle::Indirect,
            feedback_tone: FeedbackTone::Gentle,
            time_perception: TimePerception::Monochronic,
            personal_space: PersonalSpace::Medium,
            gestures: vec![
                GestureRule {
                    name: "Thumbs up".to_string(),
                    is_appropriate: false,
                    alternative: Some("Nodding".to_string()),
                    explanation: "Can be considered rude in some contexts".to_string(),
                },
                GestureRule {
                    name: "Bowing".to_string(),
                    is_appropriate: true,
                    alternative: None,
                    explanation: "Traditional sign of respect".to_string(),
                },
            ],
            colors: vec![
                ColorMeaning {
                    color: "#FF0000".to_string(),
                    positive_meanings: vec!["Luck".to_string(), "Celebration".to_string()],
                    negative_meanings: vec![],
                    usage: "Positive color, good for celebrations".to_string(),
                },
                ColorMeaning {
                    color: "#FFFFFF".to_string(),
                    positive_meanings: vec!["Purity".to_string()],
                    negative_meanings: vec!["Death".to_string(), "Mourning".to_string()],
                    usage: "Use with caution, associated with funerals".to_string(),
                },
            ],
            taboos: vec![
                CulturalTaboo {
                    topic: "Public criticism or confrontation".to_string(),
                    severity: 9,
                    explanation: "Causes loss of face and social embarrassment".to_string(),
                    alternative: Some(
                        "Private, indirect feedback with emphasis on improvement".to_string(),
                    ),
                },
                CulturalTaboo {
                    topic: "Number 4".to_string(),
                    severity: 6,
                    explanation: "Sounds like 'death' in Chinese/Japanese".to_string(),
                    alternative: Some("Use alternative numbers or avoid emphasis".to_string()),
                },
            ],
            greetings: GreetingStyle {
                formal: "おはようございます / 你好".to_string(),
                informal: "こんにちは / 嗨".to_string(),
                time_based: [
                    ("morning".to_string(), "おはようございます".to_string()),
                    ("afternoon".to_string(), "こんにちは".to_string()),
                    ("evening".to_string(), "こんばんは".to_string()),
                ]
                .iter()
                .cloned()
                .collect(),
                contact_appropriate: false,
                gesture: "Bow".to_string(),
            },
            uses_honorifics: true,
            is_collectivist: true,
            is_high_context: true,
        }
    }

    /// Create a Middle Eastern cultural profile
    #[must_use]
    pub fn middle_eastern() -> Self {
        Self {
            context: CulturalContext::MiddleEastern,
            communication_style: CommunicationStyle::Formal,
            feedback_tone: FeedbackTone::Encouraging,
            time_perception: TimePerception::Polychronic,
            personal_space: PersonalSpace::Small,
            gestures: vec![
                GestureRule {
                    name: "Thumbs up".to_string(),
                    is_appropriate: false,
                    alternative: Some("Head nod or verbal affirmation".to_string()),
                    explanation: "Can be offensive in some Middle Eastern countries".to_string(),
                },
                GestureRule {
                    name: "Left hand use".to_string(),
                    is_appropriate: false,
                    alternative: Some("Always use right hand".to_string()),
                    explanation: "Left hand considered unclean".to_string(),
                },
            ],
            colors: vec![
                ColorMeaning {
                    color: "#00FF00".to_string(),
                    positive_meanings: vec![
                        "Islam".to_string(),
                        "Nature".to_string(),
                        "Fertility".to_string(),
                    ],
                    negative_meanings: vec![],
                    usage: "Highly positive color in Islamic culture".to_string(),
                },
                ColorMeaning {
                    color: "#FFD700".to_string(),
                    positive_meanings: vec!["Wealth".to_string(), "Prosperity".to_string()],
                    negative_meanings: vec![],
                    usage: "Gold represents success and achievement".to_string(),
                },
            ],
            taboos: vec![
                CulturalTaboo {
                    topic: "Showing soles of feet".to_string(),
                    severity: 8,
                    explanation: "Highly disrespectful gesture".to_string(),
                    alternative: Some("Keep feet on ground, don't cross legs".to_string()),
                },
                CulturalTaboo {
                    topic: "Direct eye contact with opposite gender".to_string(),
                    severity: 7,
                    explanation: "Can be considered inappropriate or disrespectful".to_string(),
                    alternative: Some("Respectful averted gaze".to_string()),
                },
            ],
            greetings: GreetingStyle {
                formal: "السلام عليكم (As-salamu alaykum)".to_string(),
                informal: "مرحبا (Marhaba)".to_string(),
                time_based: [
                    (
                        "morning".to_string(),
                        "صباح الخير (Sabah al-khayr)".to_string(),
                    ),
                    (
                        "evening".to_string(),
                        "مساء الخير (Masa al-khayr)".to_string(),
                    ),
                ]
                .iter()
                .cloned()
                .collect(),
                contact_appropriate: true,
                gesture: "Handshake or kiss on cheek (same gender)".to_string(),
            },
            uses_honorifics: true,
            is_collectivist: true,
            is_high_context: true,
        }
    }
}

/// Cultural adaptation manager
pub struct CulturalAdaptationManager {
    profiles: HashMap<CulturalContext, CulturalProfile>,
    active_context: CulturalContext,
}

impl CulturalAdaptationManager {
    /// Create a new cultural adaptation manager
    #[must_use]
    pub fn new() -> Self {
        let mut profiles = HashMap::new();

        // Initialize default profiles
        profiles.insert(CulturalContext::Western, CulturalProfile::western());
        profiles.insert(CulturalContext::EastAsian, CulturalProfile::east_asian());
        profiles.insert(
            CulturalContext::MiddleEastern,
            CulturalProfile::middle_eastern(),
        );

        Self {
            profiles,
            active_context: CulturalContext::Western,
        }
    }

    /// Set active cultural context
    pub fn set_context(&mut self, context: CulturalContext) -> CulturalResult<()> {
        if !self.profiles.contains_key(&context) {
            return Err(CulturalError::CultureNotSupported {
                culture: context.code().to_string(),
            });
        }
        self.active_context = context;
        Ok(())
    }

    /// Get active cultural context
    #[must_use]
    pub fn get_context(&self) -> &CulturalContext {
        &self.active_context
    }

    /// Get active cultural profile
    #[must_use]
    pub fn get_profile(&self) -> Option<&CulturalProfile> {
        self.profiles.get(&self.active_context)
    }

    /// Register a custom cultural profile
    pub fn register_profile(&mut self, profile: CulturalProfile) {
        self.profiles.insert(profile.context.clone(), profile);
    }

    /// Get appropriate greeting for current context
    #[must_use]
    pub fn get_greeting(&self, is_formal: bool, time_of_day: Option<&str>) -> String {
        if let Some(profile) = self.get_profile() {
            if let Some(time) = time_of_day {
                if let Some(greeting) = profile.greetings.time_based.get(time) {
                    return greeting.clone();
                }
            }

            if is_formal {
                profile.greetings.formal.clone()
            } else {
                profile.greetings.informal.clone()
            }
        } else {
            "Hello".to_string()
        }
    }

    /// Adapt feedback message for cultural context
    #[must_use]
    pub fn adapt_feedback(&self, message: &str, score: f32) -> String {
        if let Some(profile) = self.get_profile() {
            match profile.feedback_tone {
                FeedbackTone::Encouraging => {
                    if score < 0.5 {
                        format!("{message} Keep practicing, you're making progress!")
                    } else if score < 0.8 {
                        format!("{message} You're doing well!")
                    } else {
                        format!("{message} Excellent work!")
                    }
                }
                FeedbackTone::Gentle => {
                    if score < 0.5 {
                        format!("{message} Consider trying this approach...")
                    } else if score < 0.8 {
                        format!("{message} This is good progress.")
                    } else {
                        format!("{message} Very nice improvement.")
                    }
                }
                FeedbackTone::Critical => {
                    if score < 0.5 {
                        format!("{message} Significant improvement needed.")
                    } else if score < 0.8 {
                        format!("{message} Acceptable, but can be better.")
                    } else {
                        format!("{message} Meets expectations.")
                    }
                }
                FeedbackTone::Neutral => message.to_string(),
                FeedbackTone::Enthusiastic => {
                    if score < 0.5 {
                        format!("{message} Let's keep working on this together!")
                    } else if score < 0.8 {
                        format!("{message} Great job!")
                    } else {
                        format!("{message} Amazing work! 🎉")
                    }
                }
            }
        } else {
            message.to_string()
        }
    }

    /// Check if a gesture is appropriate
    pub fn is_gesture_appropriate(&self, gesture_name: &str) -> CulturalResult<bool> {
        if let Some(profile) = self.get_profile() {
            if let Some(gesture) = profile.gestures.iter().find(|g| g.name == gesture_name) {
                Ok(gesture.is_appropriate)
            } else {
                Err(CulturalError::RuleNotFound {
                    rule: gesture_name.to_string(),
                    culture: self.active_context.code().to_string(),
                })
            }
        } else {
            Ok(true) // Default to appropriate if no profile
        }
    }

    /// Get alternative for inappropriate gesture
    #[must_use]
    pub fn get_gesture_alternative(&self, gesture_name: &str) -> Option<String> {
        if let Some(profile) = self.get_profile() {
            if let Some(gesture) = profile.gestures.iter().find(|g| g.name == gesture_name) {
                gesture.alternative.clone()
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Check if a topic is taboo
    #[must_use]
    pub fn is_taboo(&self, topic: &str) -> bool {
        if let Some(profile) = self.get_profile() {
            profile
                .taboos
                .iter()
                .any(|t| t.topic.to_lowercase().contains(&topic.to_lowercase()))
        } else {
            false
        }
    }

    /// Get color meaning for current context
    #[must_use]
    pub fn get_color_meaning(&self, color: &str) -> Option<&ColorMeaning> {
        if let Some(profile) = self.get_profile() {
            profile
                .colors
                .iter()
                .find(|c| c.color.eq_ignore_ascii_case(color))
        } else {
            None
        }
    }

    /// Adapt communication style
    #[must_use]
    pub fn adapt_communication(&self, direct_message: &str) -> String {
        if let Some(profile) = self.get_profile() {
            match profile.communication_style {
                CommunicationStyle::Direct => direct_message.to_string(),
                CommunicationStyle::Indirect => {
                    // Soften direct statements
                    if direct_message.starts_with("You should") {
                        direct_message.replace("You should", "It might be helpful to")
                    } else if direct_message.starts_with("You need to") {
                        direct_message.replace("You need to", "Perhaps you could consider")
                    } else {
                        format!("It seems that {}", direct_message.to_lowercase())
                    }
                }
                CommunicationStyle::Formal => {
                    format!("Respectfully, {direct_message}")
                }
                CommunicationStyle::Casual => {
                    direct_message.replace("please", "").replace("kindly", "")
                }
                CommunicationStyle::Hierarchical => {
                    if profile.uses_honorifics {
                        format!("Dear user, {direct_message}")
                    } else {
                        direct_message.to_string()
                    }
                }
                CommunicationStyle::Egalitarian => direct_message
                    .replace("Dear", "Hi")
                    .replace("Respectfully", ""),
            }
        } else {
            direct_message.to_string()
        }
    }

    /// Get all taboos for current context
    #[must_use]
    pub fn get_taboos(&self) -> Vec<&CulturalTaboo> {
        if let Some(profile) = self.get_profile() {
            profile.taboos.iter().collect()
        } else {
            vec![]
        }
    }

    /// Get cultural sensitivity score for content (0.0-1.0)
    #[must_use]
    pub fn score_cultural_sensitivity(&self, content: &str) -> f32 {
        if let Some(profile) = self.get_profile() {
            let mut score = 1.0;

            // Check for taboo topics
            for taboo in &profile.taboos {
                if content.to_lowercase().contains(&taboo.topic.to_lowercase()) {
                    score -= f32::from(taboo.severity) / 100.0;
                }
            }

            // Check for inappropriate gestures mentioned
            for gesture in &profile.gestures {
                if !gesture.is_appropriate
                    && content
                        .to_lowercase()
                        .contains(&gesture.name.to_lowercase())
                {
                    score -= 0.1;
                }
            }

            score.max(0.0)
        } else {
            1.0 // No cultural context, assume appropriate
        }
    }
}

impl Default for CulturalAdaptationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cultural_context_codes() {
        assert_eq!(CulturalContext::Western.code(), "western");
        assert_eq!(CulturalContext::EastAsian.code(), "east-asian");
        assert_eq!(CulturalContext::MiddleEastern.code(), "middle-eastern");
    }

    #[test]
    fn test_western_profile() {
        let profile = CulturalProfile::western();
        assert_eq!(profile.communication_style, CommunicationStyle::Direct);
        assert_eq!(profile.time_perception, TimePerception::Monochronic);
        assert!(!profile.uses_honorifics);
        assert!(!profile.is_collectivist);
    }

    #[test]
    fn test_east_asian_profile() {
        let profile = CulturalProfile::east_asian();
        assert_eq!(profile.communication_style, CommunicationStyle::Indirect);
        assert!(profile.uses_honorifics);
        assert!(profile.is_collectivist);
        assert!(profile.is_high_context);
    }

    #[test]
    fn test_middle_eastern_profile() {
        let profile = CulturalProfile::middle_eastern();
        assert_eq!(profile.communication_style, CommunicationStyle::Formal);
        assert_eq!(profile.time_perception, TimePerception::Polychronic);
        assert!(profile.uses_honorifics);
    }

    #[test]
    fn test_manager_creation() {
        let manager = CulturalAdaptationManager::new();
        assert_eq!(manager.get_context(), &CulturalContext::Western);
    }

    #[test]
    fn test_set_context() {
        let mut manager = CulturalAdaptationManager::new();

        manager.set_context(CulturalContext::EastAsian).unwrap();
        assert_eq!(manager.get_context(), &CulturalContext::EastAsian);

        // Test unsupported culture
        let result = manager.set_context(CulturalContext::African);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_greeting() {
        let mut manager = CulturalAdaptationManager::new();

        // Western greeting
        let greeting = manager.get_greeting(true, Some("morning"));
        assert_eq!(greeting, "Good morning");

        // East Asian greeting
        manager.set_context(CulturalContext::EastAsian).unwrap();
        let greeting = manager.get_greeting(false, None);
        assert!(greeting.contains("こんにちは") || greeting.contains("嗨"));
    }

    #[test]
    fn test_adapt_feedback() {
        let mut manager = CulturalAdaptationManager::new();

        // Western encouraging tone
        let adapted = manager.adapt_feedback("Good pronunciation", 0.4);
        assert!(adapted.contains("Keep practicing"));

        let adapted = manager.adapt_feedback("Good pronunciation", 0.9);
        assert!(adapted.contains("Excellent"));

        // East Asian gentle tone
        manager.set_context(CulturalContext::EastAsian).unwrap();
        let adapted = manager.adapt_feedback("Good pronunciation", 0.4);
        assert!(adapted.contains("Consider"));
    }

    #[test]
    fn test_gesture_appropriateness() {
        let mut manager = CulturalAdaptationManager::new();

        // Thumbs up is appropriate in Western culture
        assert!(manager.is_gesture_appropriate("Thumbs up").unwrap());

        // Thumbs up is not appropriate in East Asian culture
        manager.set_context(CulturalContext::EastAsian).unwrap();
        assert!(!manager.is_gesture_appropriate("Thumbs up").unwrap());

        // Check alternative
        let alt = manager.get_gesture_alternative("Thumbs up");
        assert!(alt.is_some());
        assert_eq!(alt.unwrap(), "Nodding");
    }

    #[test]
    fn test_taboo_detection() {
        let mut manager = CulturalAdaptationManager::new();

        manager.set_context(CulturalContext::EastAsian).unwrap();

        assert!(manager.is_taboo("public criticism"));
        assert!(manager.is_taboo("number 4"));
        assert!(!manager.is_taboo("general feedback"));
    }

    #[test]
    fn test_color_meanings() {
        let mut manager = CulturalAdaptationManager::new();

        manager.set_context(CulturalContext::EastAsian).unwrap();

        let red_meaning = manager.get_color_meaning("#FF0000");
        assert!(red_meaning.is_some());

        let meaning = red_meaning.unwrap();
        assert!(meaning.positive_meanings.contains(&"Luck".to_string()));
    }

    #[test]
    fn test_adapt_communication() {
        let mut manager = CulturalAdaptationManager::new();

        // Direct communication (Western)
        let adapted = manager.adapt_communication("You should practice more");
        assert_eq!(adapted, "You should practice more");

        // Indirect communication (East Asian)
        manager.set_context(CulturalContext::EastAsian).unwrap();
        let adapted = manager.adapt_communication("You should practice more");
        assert!(adapted.contains("helpful"));
        assert!(!adapted.contains("You should"));
    }

    #[test]
    fn test_cultural_sensitivity_score() {
        let mut manager = CulturalAdaptationManager::new();

        manager.set_context(CulturalContext::EastAsian).unwrap();

        // Content without taboos
        let score = manager.score_cultural_sensitivity("Please practice your pronunciation");
        assert_eq!(score, 1.0);

        // Content with taboo topic - must contain exact taboo phrase
        let score =
            manager.score_cultural_sensitivity("I will use public criticism or confrontation");
        assert!(score < 1.0);
        assert!((score - 0.91).abs() < 0.01); // Should be 1.0 - 0.09 = 0.91 (severity 9)
    }

    #[test]
    fn test_get_taboos() {
        let mut manager = CulturalAdaptationManager::new();

        manager.set_context(CulturalContext::MiddleEastern).unwrap();
        let taboos = manager.get_taboos();

        assert!(!taboos.is_empty());
        assert!(taboos.iter().any(|t| t.topic.contains("feet")));
    }
}
