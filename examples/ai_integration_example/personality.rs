//! Personality profiling and emotion analysis.
//!
//! This module hosts the [`PersonalityEngine`], which analyzes user and
//! message personality/emotion signals, along with the supporting data
//! structures shared by the rest of the AI integration example (personality
//! traits, emotional state tracking, and emotion/sentiment analysis
//! results).

use crate::error::AIVoiceError;
use crate::voice_synthesis::VoiceCharacteristics;
use std::collections::HashMap;
use std::time::Duration;

/// Personality and emotion analysis engine
#[allow(dead_code)]
pub struct PersonalityEngine {
    /// Personality trait analyzer
    pub(crate) trait_analyzer: PersonalityTraitAnalyzer,
    /// Emotion state tracker
    pub(crate) emotion_tracker: EmotionStateTracker,
    /// Personality-voice mapping
    pub(crate) personality_voice_mapper: PersonalityVoiceMapper,
}

#[derive(Debug, Clone)]
pub struct PersonalityTraitAnalyzer {
    pub trait_models: HashMap<String, TraitModel>,
    pub analysis_confidence: f32,
}

#[derive(Debug, Clone)]
pub struct EmotionStateTracker {
    pub current_state: EmotionalState,
    pub state_history: Vec<EmotionalState>,
    pub transition_patterns: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct PersonalityVoiceMapper {
    pub personality_mappings: HashMap<String, VoiceCharacteristics>,
    pub dynamic_adjustment: bool,
}

#[derive(Debug, Clone)]
pub struct TraitModel {
    pub trait_name: String,
    pub detection_patterns: Vec<String>,
    pub confidence_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct EmotionalState {
    pub current_emotions: HashMap<String, f32>,
    pub mood: String,
    pub energy_level: f32,
    pub stability: f32,
}

#[derive(Debug, Clone)]
pub struct PersonalityProfile {
    pub traits: HashMap<String, f32>, // Big Five + additional traits
    pub voice_style: String,
    pub interaction_style: String,
    pub emotional_expressiveness: f32,
}

#[derive(Debug, Clone)]
pub struct EmotionAnalysis {
    pub detected_emotions: Vec<EmotionScore>,
    pub sentiment: SentimentScore,
    pub emotion_progression: Vec<EmotionPoint>,
}

// Additional supporting types
#[derive(Debug, Clone)]
pub struct EmotionScore {
    pub emotion: String,
    pub score: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct SentimentScore {
    pub polarity: f32,  // -1.0 to 1.0
    pub magnitude: f32, // 0.0 to 1.0
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct EmotionPoint {
    pub timestamp: Duration,
    pub emotion: String,
    pub intensity: f32,
}

impl PersonalityEngine {
    pub(crate) fn new() -> Self {
        Self {
            trait_analyzer: PersonalityTraitAnalyzer {
                trait_models: HashMap::new(),
                analysis_confidence: 0.75,
            },
            emotion_tracker: EmotionStateTracker {
                current_state: EmotionalState {
                    current_emotions: HashMap::new(),
                    mood: "neutral".to_string(),
                    energy_level: 0.5,
                    stability: 0.8,
                },
                state_history: Vec::new(),
                transition_patterns: HashMap::new(),
            },
            personality_voice_mapper: PersonalityVoiceMapper {
                personality_mappings: HashMap::new(),
                dynamic_adjustment: true,
            },
        }
    }

    pub(crate) async fn analyze_user_personality(
        &self,
        _user_id: &str,
    ) -> Result<PersonalityProfile, AIVoiceError> {
        // Mock personality analysis
        Ok(PersonalityProfile {
            traits: HashMap::from([
                ("openness".to_string(), 0.7),
                ("conscientiousness".to_string(), 0.6),
                ("extraversion".to_string(), 0.5),
                ("agreeableness".to_string(), 0.8),
                ("neuroticism".to_string(), 0.3),
            ]),
            voice_style: "balanced".to_string(),
            interaction_style: "collaborative".to_string(),
            emotional_expressiveness: 0.6,
        })
    }

    pub(crate) async fn analyze_message_emotion(
        &self,
        message: &str,
    ) -> Result<EmotionAnalysis, AIVoiceError> {
        // Simple emotion analysis based on keywords and sentiment
        let message_lower = message.to_lowercase();
        let mut detected_emotions = Vec::new();

        // Check for specific emotions
        if message_lower.contains("happy")
            || message_lower.contains("great")
            || message_lower.contains("wonderful")
        {
            detected_emotions.push(EmotionScore {
                emotion: "joy".to_string(),
                score: 0.8,
                confidence: 0.85,
            });
        }

        if message_lower.contains("sad")
            || message_lower.contains("upset")
            || message_lower.contains("trouble")
        {
            detected_emotions.push(EmotionScore {
                emotion: "sadness".to_string(),
                score: 0.7,
                confidence: 0.75,
            });
        }

        if message_lower.contains("excited")
            || message_lower.contains("amazing")
            || message_lower.contains("fantastic")
        {
            detected_emotions.push(EmotionScore {
                emotion: "excitement".to_string(),
                score: 0.9,
                confidence: 0.8,
            });
        }

        if message_lower.contains("calm")
            || message_lower.contains("peaceful")
            || message_lower.contains("relax")
        {
            detected_emotions.push(EmotionScore {
                emotion: "calm".to_string(),
                score: 0.7,
                confidence: 0.7,
            });
        }

        // Default to neutral if no emotions detected
        if detected_emotions.is_empty() {
            detected_emotions.push(EmotionScore {
                emotion: "neutral".to_string(),
                score: 0.6,
                confidence: 0.6,
            });
        }

        // Calculate sentiment
        let positive_words = [
            "good",
            "great",
            "wonderful",
            "amazing",
            "fantastic",
            "happy",
            "love",
            "excellent",
        ];
        let negative_words = [
            "bad", "terrible", "awful", "hate", "sad", "upset", "trouble", "problem",
        ];

        let positive_count = positive_words
            .iter()
            .map(|word| message_lower.matches(word).count())
            .sum::<usize>();
        let negative_count = negative_words
            .iter()
            .map(|word| message_lower.matches(word).count())
            .sum::<usize>();

        let polarity = if positive_count > negative_count {
            0.5 + (positive_count as f32 - negative_count as f32) / 10.0
        } else if negative_count > positive_count {
            -0.5 - (negative_count as f32 - positive_count as f32) / 10.0
        } else {
            0.0
        };

        let magnitude =
            (positive_count + negative_count) as f32 / message.split_whitespace().count() as f32;

        let sentiment = SentimentScore {
            polarity: polarity.clamp(-1.0, 1.0),
            magnitude: magnitude.min(1.0),
            confidence: 0.75,
        };

        // Create emotion progression (simplified)
        let emotion_progression = detected_emotions
            .iter()
            .enumerate()
            .map(|(i, emotion)| EmotionPoint {
                timestamp: Duration::from_secs(i as u64),
                emotion: emotion.emotion.clone(),
                intensity: emotion.score,
            })
            .collect();

        Ok(EmotionAnalysis {
            detected_emotions,
            sentiment,
            emotion_progression,
        })
    }
}
