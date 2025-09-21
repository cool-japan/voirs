use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use voirs_sdk::types::SynthesisConfig;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EmotionType {
    Joy,
    Sadness,
    Anger,
    Fear,
    Surprise,
    Disgust,
    Neutral,
    Excitement,
    Calm,
    Confident,
    Uncertain,
    Authoritative,
    Friendly,
    Professional,
    Dramatic,
    Whispering,
    Shouting,
    Conversational,
    Narrating,
    Questioning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionConfig {
    pub emotion_type: EmotionType,
    pub intensity: f32, // 0.0 to 1.0
    pub duration_ms: Option<u32>,
    pub fade_in_ms: Option<u32>,
    pub fade_out_ms: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionTransition {
    pub from_emotion: EmotionType,
    pub to_emotion: EmotionType,
    pub transition_duration_ms: u32,
    pub transition_curve: TransitionCurve,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionCurve {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Exponential,
    Sigmoid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAwareEmotionConfig {
    pub context_keywords: HashMap<String, EmotionType>,
    pub punctuation_emotions: HashMap<char, EmotionType>,
    pub default_emotion: EmotionType,
    pub emotion_continuity: EmotionContinuity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionContinuity {
    pub carry_over_percentage: f32, // 0.0 to 1.0
    pub minimum_duration_ms: u32,
    pub maximum_duration_ms: u32,
    pub decay_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionSynthesisConfig {
    pub base_config: SynthesisConfig,
    pub emotion_config: EmotionConfig,
    pub context_awareness: Option<ContextAwareEmotionConfig>,
    pub emotion_transitions: Vec<EmotionTransition>,
    pub prosody_adjustments: ProsodyAdjustments,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodyAdjustments {
    pub pitch_adjustment: f32,  // Relative adjustment (-1.0 to 1.0)
    pub speed_adjustment: f32,  // Relative adjustment (0.5 to 2.0)
    pub volume_adjustment: f32, // Relative adjustment (-1.0 to 1.0)
    pub pause_adjustment: f32,  // Relative adjustment (0.0 to 2.0)
    pub emphasis_strength: f32, // 0.0 to 1.0
}

pub struct EmotionSynthesizer {
    current_emotion: EmotionType,
    emotion_history: Vec<(EmotionType, f32)>, // (emotion, intensity)
    context_config: Option<ContextAwareEmotionConfig>,
    transition_state: Option<EmotionTransition>,
}

impl EmotionSynthesizer {
    pub fn new() -> Self {
        Self {
            current_emotion: EmotionType::Neutral,
            emotion_history: Vec::new(),
            context_config: None,
            transition_state: None,
        }
    }

    pub fn with_context_config(mut self, config: ContextAwareEmotionConfig) -> Self {
        self.context_config = Some(config);
        self
    }

    pub fn analyze_text_emotion(&self, text: &str) -> EmotionType {
        if let Some(config) = &self.context_config {
            // Check for context keywords
            for (keyword, emotion) in &config.context_keywords {
                if text.to_lowercase().contains(&keyword.to_lowercase()) {
                    return emotion.clone();
                }
            }

            // Check punctuation-based emotions
            for (punct, emotion) in &config.punctuation_emotions {
                if text.contains(*punct) {
                    return emotion.clone();
                }
            }

            return config.default_emotion.clone();
        }

        EmotionType::Neutral
    }

    pub fn apply_emotion_continuity(
        &mut self,
        new_emotion: EmotionType,
        intensity: f32,
    ) -> EmotionType {
        if let Some(config) = &self.context_config {
            let continuity = &config.emotion_continuity;

            // Apply carry-over from previous emotion
            if let Some((prev_emotion, prev_intensity)) = self.emotion_history.last() {
                let carry_over = prev_intensity * continuity.carry_over_percentage;
                if carry_over > 0.1 {
                    // Blend emotions if carry-over is significant
                    return self.blend_emotions(prev_emotion.clone(), new_emotion, carry_over);
                }
            }
        }

        self.emotion_history.push((new_emotion.clone(), intensity));

        // Keep history size manageable
        if self.emotion_history.len() > 10 {
            self.emotion_history.remove(0);
        }

        new_emotion
    }

    fn blend_emotions(
        &self,
        emotion1: EmotionType,
        emotion2: EmotionType,
        blend_factor: f32,
    ) -> EmotionType {
        // For simplicity, return the stronger emotion
        // In a real implementation, this would create a blended emotion state
        if blend_factor > 0.5 {
            emotion1
        } else {
            emotion2
        }
    }

    pub fn create_emotion_synthesis_config(
        &mut self,
        text: &str,
        base_config: SynthesisConfig,
        emotion_config: EmotionConfig,
    ) -> EmotionSynthesisConfig {
        let detected_emotion = self.analyze_text_emotion(text);
        let final_emotion =
            self.apply_emotion_continuity(detected_emotion, emotion_config.intensity);

        let prosody_adjustments =
            self.calculate_prosody_adjustments(&final_emotion, &emotion_config);

        EmotionSynthesisConfig {
            base_config,
            emotion_config: EmotionConfig {
                emotion_type: final_emotion,
                ..emotion_config
            },
            context_awareness: self.context_config.clone(),
            emotion_transitions: Vec::new(),
            prosody_adjustments,
        }
    }

    fn calculate_prosody_adjustments(
        &self,
        emotion: &EmotionType,
        config: &EmotionConfig,
    ) -> ProsodyAdjustments {
        let intensity = config.intensity;

        match emotion {
            EmotionType::Joy => ProsodyAdjustments {
                pitch_adjustment: 0.2 * intensity,
                speed_adjustment: 1.0 + (0.2 * intensity),
                volume_adjustment: 0.1 * intensity,
                pause_adjustment: 0.8,
                emphasis_strength: 0.7 * intensity,
            },
            EmotionType::Sadness => ProsodyAdjustments {
                pitch_adjustment: -0.3 * intensity,
                speed_adjustment: 1.0 - (0.3 * intensity),
                volume_adjustment: -0.2 * intensity,
                pause_adjustment: 1.5,
                emphasis_strength: 0.3 * intensity,
            },
            EmotionType::Anger => ProsodyAdjustments {
                pitch_adjustment: 0.4 * intensity,
                speed_adjustment: 1.0 + (0.4 * intensity),
                volume_adjustment: 0.3 * intensity,
                pause_adjustment: 0.6,
                emphasis_strength: 0.9 * intensity,
            },
            EmotionType::Fear => ProsodyAdjustments {
                pitch_adjustment: 0.5 * intensity,
                speed_adjustment: 1.0 + (0.5 * intensity),
                volume_adjustment: -0.1 * intensity,
                pause_adjustment: 0.7,
                emphasis_strength: 0.8 * intensity,
            },
            EmotionType::Excitement => ProsodyAdjustments {
                pitch_adjustment: 0.3 * intensity,
                speed_adjustment: 1.0 + (0.3 * intensity),
                volume_adjustment: 0.2 * intensity,
                pause_adjustment: 0.7,
                emphasis_strength: 0.8 * intensity,
            },
            EmotionType::Calm => ProsodyAdjustments {
                pitch_adjustment: -0.1 * intensity,
                speed_adjustment: 1.0 - (0.1 * intensity),
                volume_adjustment: -0.05 * intensity,
                pause_adjustment: 1.2,
                emphasis_strength: 0.4 * intensity,
            },
            EmotionType::Confident => ProsodyAdjustments {
                pitch_adjustment: 0.1 * intensity,
                speed_adjustment: 1.0,
                volume_adjustment: 0.15 * intensity,
                pause_adjustment: 0.9,
                emphasis_strength: 0.7 * intensity,
            },
            EmotionType::Whispering => ProsodyAdjustments {
                pitch_adjustment: -0.2 * intensity,
                speed_adjustment: 1.0 - (0.2 * intensity),
                volume_adjustment: -0.5 * intensity,
                pause_adjustment: 1.3,
                emphasis_strength: 0.2 * intensity,
            },
            EmotionType::Shouting => ProsodyAdjustments {
                pitch_adjustment: 0.6 * intensity,
                speed_adjustment: 1.0 + (0.1 * intensity),
                volume_adjustment: 0.4 * intensity,
                pause_adjustment: 0.8,
                emphasis_strength: 0.9 * intensity,
            },
            _ => ProsodyAdjustments {
                pitch_adjustment: 0.0,
                speed_adjustment: 1.0,
                volume_adjustment: 0.0,
                pause_adjustment: 1.0,
                emphasis_strength: 0.5,
            },
        }
    }

    pub fn set_current_emotion(&mut self, emotion: EmotionType) {
        self.current_emotion = emotion;
    }

    pub fn get_current_emotion(&self) -> &EmotionType {
        &self.current_emotion
    }

    pub fn get_emotion_history(&self) -> &Vec<(EmotionType, f32)> {
        &self.emotion_history
    }
}

impl Default for EmotionSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EmotionConfig {
    fn default() -> Self {
        Self {
            emotion_type: EmotionType::Neutral,
            intensity: 0.5,
            duration_ms: None,
            fade_in_ms: None,
            fade_out_ms: None,
        }
    }
}

impl Default for ContextAwareEmotionConfig {
    fn default() -> Self {
        let mut context_keywords = HashMap::new();
        context_keywords.insert("happy".to_string(), EmotionType::Joy);
        context_keywords.insert("sad".to_string(), EmotionType::Sadness);
        context_keywords.insert("angry".to_string(), EmotionType::Anger);
        context_keywords.insert("excited".to_string(), EmotionType::Excitement);
        context_keywords.insert("calm".to_string(), EmotionType::Calm);
        context_keywords.insert("confident".to_string(), EmotionType::Confident);

        let mut punctuation_emotions = HashMap::new();
        punctuation_emotions.insert('!', EmotionType::Excitement);
        punctuation_emotions.insert('?', EmotionType::Questioning);
        punctuation_emotions.insert('.', EmotionType::Neutral);

        Self {
            context_keywords,
            punctuation_emotions,
            default_emotion: EmotionType::Neutral,
            emotion_continuity: EmotionContinuity {
                carry_over_percentage: 0.3,
                minimum_duration_ms: 500,
                maximum_duration_ms: 5000,
                decay_rate: 0.1,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_synthesizer_creation() {
        let synthesizer = EmotionSynthesizer::new();
        assert_eq!(synthesizer.current_emotion, EmotionType::Neutral);
        assert!(synthesizer.emotion_history.is_empty());
    }

    #[test]
    fn test_emotion_analysis() {
        let synthesizer =
            EmotionSynthesizer::new().with_context_config(ContextAwareEmotionConfig::default());

        let joy_text = "I'm so happy about this!";
        let detected_emotion = synthesizer.analyze_text_emotion(joy_text);
        assert_eq!(detected_emotion, EmotionType::Joy);

        let question_text = "What is happening?";
        let detected_emotion = synthesizer.analyze_text_emotion(question_text);
        assert_eq!(detected_emotion, EmotionType::Questioning);
    }

    #[test]
    fn test_prosody_adjustments() {
        let synthesizer = EmotionSynthesizer::new();
        let config = EmotionConfig {
            emotion_type: EmotionType::Joy,
            intensity: 0.8,
            ..Default::default()
        };

        let adjustments = synthesizer.calculate_prosody_adjustments(&EmotionType::Joy, &config);
        assert!(adjustments.pitch_adjustment > 0.0);
        assert!(adjustments.speed_adjustment > 1.0);
        assert!(adjustments.volume_adjustment > 0.0);
    }

    #[test]
    fn test_emotion_config_serialization() {
        let config = EmotionConfig {
            emotion_type: EmotionType::Joy,
            intensity: 0.7,
            duration_ms: Some(1000),
            fade_in_ms: Some(100),
            fade_out_ms: Some(200),
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: EmotionConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.emotion_type, EmotionType::Joy);
        assert_eq!(deserialized.intensity, 0.7);
        assert_eq!(deserialized.duration_ms, Some(1000));
    }

    #[test]
    fn test_emotion_continuity() {
        let mut synthesizer =
            EmotionSynthesizer::new().with_context_config(ContextAwareEmotionConfig::default());

        // First emotion
        let emotion1 = synthesizer.apply_emotion_continuity(EmotionType::Joy, 0.8);
        assert_eq!(emotion1, EmotionType::Joy);

        // Second emotion should be influenced by continuity
        let emotion2 = synthesizer.apply_emotion_continuity(EmotionType::Sadness, 0.6);
        assert!(synthesizer.emotion_history.len() >= 1);
    }
}
