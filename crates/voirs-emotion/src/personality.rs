//! Long-term personality trait modeling for emotion control
//!
//! This module provides personality-based emotion modeling that influences
//! how emotions are expressed over time, creating consistent personality
//! profiles for speakers.

use crate::{
    core::EmotionProcessor,
    types::{Emotion, EmotionIntensity, EmotionParameters, EmotionVector},
    Error, Result,
};

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Personality trait model for long-term emotion characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityModel {
    /// Big Five personality traits
    pub big_five: BigFiveTraits,
    /// Emotional tendencies
    pub emotional_tendencies: EmotionalTendencies,
    /// Cultural background influence
    pub cultural_background: String,
    /// Adaptation rate (how quickly personality can change)
    pub adaptation_rate: f32,
    /// Stability over time
    pub stability: f32,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last update timestamp
    pub updated_at: SystemTime,
}

/// Big Five personality traits (OCEAN model)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BigFiveTraits {
    /// Openness to experience (0.0 to 1.0)
    pub openness: f32,
    /// Conscientiousness (0.0 to 1.0)
    pub conscientiousness: f32,
    /// Extraversion (0.0 to 1.0)
    pub extraversion: f32,
    /// Agreeableness (0.0 to 1.0)
    pub agreeableness: f32,
    /// Neuroticism (0.0 to 1.0)
    pub neuroticism: f32,
}

/// Emotional tendencies based on personality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalTendencies {
    /// Default emotional state
    pub baseline_emotion: Emotion,
    /// Typical emotional intensity
    pub baseline_intensity: f32,
    /// Emotional volatility (tendency to change emotions)
    pub volatility: f32,
    /// Emotional recovery rate (how quickly emotions return to baseline)
    pub recovery_rate: f32,
    /// Preferred emotions for different contexts
    pub contextual_preferences: HashMap<String, Vec<(Emotion, f32)>>,
    /// Emotion suppression tendencies
    pub suppression_tendencies: HashMap<Emotion, f32>,
}

/// Personality-based emotion modifier
#[derive(Debug)]
pub struct PersonalityEmotionModifier {
    /// Current personality model
    personality: RwLock<PersonalityModel>,
    /// Historical emotion patterns
    emotion_history: RwLock<Vec<(SystemTime, Emotion, f32)>>,
    /// Adaptation learning rate
    learning_rate: f32,
    /// Maximum history size
    max_history_size: usize,
}

impl PersonalityEmotionModifier {
    /// Create new personality modifier
    pub fn new(personality: PersonalityModel) -> Self {
        Self {
            personality: RwLock::new(personality),
            emotion_history: RwLock::new(Vec::new()),
            learning_rate: 0.01,
            max_history_size: 1000,
        }
    }

    /// Create with default personality
    pub fn default_personality() -> Self {
        let personality = PersonalityModel::default();
        Self::new(personality)
    }

    /// Modify emotion parameters based on personality
    pub async fn modify_emotion_for_personality(
        &self,
        mut params: EmotionParameters,
        context: Option<&str>,
    ) -> Result<EmotionParameters> {
        let personality = self.personality.read().await;

        debug!("Modifying emotion parameters for personality");

        // Apply Big Five traits to emotion parameters
        params = self.apply_big_five_traits(&params, &personality.big_five)?;

        // Apply emotional tendencies
        params =
            self.apply_emotional_tendencies(&params, &personality.emotional_tendencies, context)?;

        // Record emotion for learning
        if let Some((emotion, intensity)) = params.emotion_vector.dominant_emotion() {
            self.record_emotion_pattern((emotion.as_str().to_string(), intensity.value()))
                .await;
        }

        debug!("Emotion parameters modified for personality");
        Ok(params)
    }

    /// Apply Big Five personality traits to emotion parameters
    fn apply_big_five_traits(
        &self,
        params: &EmotionParameters,
        traits: &BigFiveTraits,
    ) -> Result<EmotionParameters> {
        let mut modified = params.clone();

        // Extraversion affects energy and social emotions
        if traits.extraversion > 0.6 {
            modified.energy_scale *= 1.0 + (traits.extraversion - 0.5) * 0.4;
            modified.tempo_scale *= 1.0 + (traits.extraversion - 0.5) * 0.2;
        }

        // Neuroticism affects emotional intensity and stability
        if traits.neuroticism > 0.5 {
            let neuroticism_factor = (traits.neuroticism - 0.5) * 2.0;
            // Increase emotional intensity for negative emotions
            if let Some((emotion, _)) = modified.emotion_vector.dominant_emotion() {
                if matches!(
                    emotion,
                    Emotion::Angry | Emotion::Sad | Emotion::Fear | Emotion::Melancholic
                ) {
                    modified.energy_scale *= 1.0 + neuroticism_factor * 0.3;
                    modified.pitch_shift *= 1.0 + neuroticism_factor * 0.2;
                }
            }
        }

        // Openness affects expressiveness and variation
        if traits.openness > 0.7 {
            modified.breathiness += (traits.openness - 0.5) * 0.3;
            modified.roughness -= (traits.openness - 0.5) * 0.2; // Less roughness for open personality
        }

        // Conscientiousness affects consistency and control
        if traits.conscientiousness > 0.6 {
            // Reduce extreme variations
            let control_factor = traits.conscientiousness - 0.5;
            modified.pitch_shift =
                modified.pitch_shift * (1.0 - control_factor * 0.2) + control_factor * 0.2;
            modified.tempo_scale =
                modified.tempo_scale * (1.0 - control_factor * 0.1) + control_factor * 0.1;
        }

        // Agreeableness affects prosody and voice quality
        if traits.agreeableness > 0.6 {
            modified.breathiness -= (traits.agreeableness - 0.5) * 0.3; // Less harsh
            modified.roughness -= (traits.agreeableness - 0.5) * 0.4; // Smoother voice
        }

        Ok(modified)
    }

    /// Apply emotional tendencies to parameters
    fn apply_emotional_tendencies(
        &self,
        params: &EmotionParameters,
        tendencies: &EmotionalTendencies,
        context: Option<&str>,
    ) -> Result<EmotionParameters> {
        let mut modified = params.clone();

        // Apply volatility (how much emotions can change)
        if tendencies.volatility < 0.3 {
            // Low volatility - dampen emotional changes
            let damping_factor = 1.0 - tendencies.volatility;
            modified.energy_scale =
                modified.energy_scale * (1.0 - damping_factor * 0.3) + damping_factor * 0.3;
            modified.pitch_shift =
                modified.pitch_shift * (1.0 - damping_factor * 0.2) + damping_factor * 0.2;
        }

        // Apply contextual preferences
        if let Some(context_key) = context {
            if let Some(preferences) = tendencies.contextual_preferences.get(context_key) {
                // Find the closest preferred emotion
                if let Some((dominant_emotion, intensity)) =
                    modified.emotion_vector.dominant_emotion()
                {
                    for (preferred_emotion, preference_strength) in preferences {
                        if *preferred_emotion == dominant_emotion {
                            // Boost this emotion since it's preferred in this context
                            let boost = preference_strength * 0.2;
                            modified.energy_scale *= 1.0 + boost;
                            modified.tempo_scale *= 1.0 + boost * 0.5;
                            break;
                        }
                    }
                }
            }
        }

        // Apply suppression tendencies
        if let Some((dominant_emotion, _)) = modified.emotion_vector.dominant_emotion() {
            if let Some(&suppression_level) =
                tendencies.suppression_tendencies.get(&dominant_emotion)
            {
                if suppression_level > 0.0 {
                    // Suppress this emotion
                    let suppression_factor = 1.0 - suppression_level;
                    modified.energy_scale *= suppression_factor;
                    modified.pitch_shift =
                        modified.pitch_shift * suppression_factor + suppression_level;
                    modified.tempo_scale =
                        modified.tempo_scale * suppression_factor + suppression_level;
                }
            }
        }

        Ok(modified)
    }

    /// Record emotion pattern for learning
    async fn record_emotion_pattern(&self, emotion_info: (String, f32)) {
        let mut history = self.emotion_history.write().await;

        // Parse emotion string to enum
        if let Ok(emotion) = emotion_info.0.parse::<Emotion>() {
            history.push((SystemTime::now(), emotion, emotion_info.1));

            // Limit history size
            if history.len() > self.max_history_size {
                history.remove(0);
            }
        }
    }

    /// Adapt personality based on observed patterns
    pub async fn adapt_personality(&self) -> Result<()> {
        let history = self.emotion_history.read().await;

        if history.len() < 10 {
            return Ok(()); // Not enough data for adaptation
        }

        debug!(
            "Adapting personality based on {} observations",
            history.len()
        );

        // Analyze recent emotional patterns
        let recent_emotions: Vec<_> = history.iter().rev().take(50).collect();

        // Calculate average intensity and volatility
        let avg_intensity: f32 = recent_emotions
            .iter()
            .map(|(_, _, intensity)| *intensity)
            .sum::<f32>()
            / recent_emotions.len() as f32;

        // Calculate emotional volatility (how much emotions change)
        let mut volatility_sum = 0.0;
        for window in recent_emotions.windows(2) {
            let (_, emotion1, intensity1) = window[0];
            let (_, emotion2, intensity2) = window[1];

            let emotion_change = if emotion1 != emotion2 { 1.0 } else { 0.0 };
            let intensity_change = (intensity1 - intensity2).abs();
            volatility_sum += emotion_change + intensity_change;
        }
        let observed_volatility = volatility_sum / (recent_emotions.len() - 1).max(1) as f32;

        // Update personality model
        let mut personality = self.personality.write().await;

        // Adapt emotional tendencies
        let learning_rate = self.learning_rate * personality.adaptation_rate;
        personality.emotional_tendencies.baseline_intensity =
            personality.emotional_tendencies.baseline_intensity * (1.0 - learning_rate)
                + avg_intensity * learning_rate;

        personality.emotional_tendencies.volatility = personality.emotional_tendencies.volatility
            * (1.0 - learning_rate)
            + observed_volatility * learning_rate;

        // Update timestamp
        personality.updated_at = SystemTime::now();

        info!(
            "Personality adapted: avg_intensity={:.2}, volatility={:.2}",
            avg_intensity, observed_volatility
        );

        Ok(())
    }

    /// Get current personality model
    pub async fn get_personality(&self) -> PersonalityModel {
        self.personality.read().await.clone()
    }

    /// Update personality model
    pub async fn set_personality(&self, personality: PersonalityModel) {
        let mut current = self.personality.write().await;
        *current = personality;
        debug!("Personality model updated");
    }

    /// Export personality model to JSON
    pub async fn export_personality(&self) -> Result<String> {
        let personality = self.personality.read().await;
        serde_json::to_string_pretty(&*personality).map_err(|e| Error::Serialization(e))
    }

    /// Import personality model from JSON
    pub async fn import_personality(&self, json: &str) -> Result<()> {
        let personality: PersonalityModel = serde_json::from_str(json)?;
        self.set_personality(personality).await;
        Ok(())
    }

    /// Get personality statistics
    pub async fn get_personality_stats(&self) -> PersonalityStats {
        let personality = self.personality.read().await;
        let history = self.emotion_history.read().await;

        // Calculate emotion distribution
        let mut emotion_counts = HashMap::new();
        for (_, emotion, _) in history.iter() {
            *emotion_counts.entry(emotion.clone()).or_insert(0) += 1;
        }

        let emotion_distribution: HashMap<Emotion, f32> = emotion_counts
            .into_iter()
            .map(|(emotion, count)| (emotion, count as f32 / history.len().max(1) as f32))
            .collect();

        PersonalityStats {
            big_five_summary: format!(
                "O:{:.2} C:{:.2} E:{:.2} A:{:.2} N:{:.2}",
                personality.big_five.openness,
                personality.big_five.conscientiousness,
                personality.big_five.extraversion,
                personality.big_five.agreeableness,
                personality.big_five.neuroticism
            ),
            baseline_emotion: personality.emotional_tendencies.baseline_emotion.clone(),
            baseline_intensity: personality.emotional_tendencies.baseline_intensity,
            volatility: personality.emotional_tendencies.volatility,
            adaptation_rate: personality.adaptation_rate,
            stability: personality.stability,
            observation_count: history.len(),
            emotion_distribution,
            created_at: personality.created_at,
            updated_at: personality.updated_at,
        }
    }
}

/// Statistics about personality model
#[derive(Debug, Clone)]
pub struct PersonalityStats {
    /// Big Five traits summary
    pub big_five_summary: String,
    /// Baseline emotion
    pub baseline_emotion: Emotion,
    /// Baseline intensity
    pub baseline_intensity: f32,
    /// Emotional volatility
    pub volatility: f32,
    /// Adaptation rate
    pub adaptation_rate: f32,
    /// Personality stability
    pub stability: f32,
    /// Number of observations
    pub observation_count: usize,
    /// Distribution of observed emotions
    pub emotion_distribution: HashMap<Emotion, f32>,
    /// Model creation time
    pub created_at: SystemTime,
    /// Last update time
    pub updated_at: SystemTime,
}

impl Default for PersonalityModel {
    fn default() -> Self {
        let now = SystemTime::now();
        Self {
            big_five: BigFiveTraits::default(),
            emotional_tendencies: EmotionalTendencies::default(),
            cultural_background: "neutral".to_string(),
            adaptation_rate: 0.1,
            stability: 0.8,
            created_at: now,
            updated_at: now,
        }
    }
}

impl Default for BigFiveTraits {
    fn default() -> Self {
        Self {
            openness: 0.5,
            conscientiousness: 0.5,
            extraversion: 0.5,
            agreeableness: 0.5,
            neuroticism: 0.3, // Lower default for stability
        }
    }
}

impl Default for EmotionalTendencies {
    fn default() -> Self {
        Self {
            baseline_emotion: Emotion::Neutral,
            baseline_intensity: 0.5,
            volatility: 0.3,
            recovery_rate: 0.7,
            contextual_preferences: HashMap::new(),
            suppression_tendencies: HashMap::new(),
        }
    }
}

// Implement string parsing for Emotion enum
impl std::str::FromStr for Emotion {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "neutral" => Ok(Emotion::Neutral),
            "happy" => Ok(Emotion::Happy),
            "sad" => Ok(Emotion::Sad),
            "angry" => Ok(Emotion::Angry),
            "fear" => Ok(Emotion::Fear),
            "surprise" => Ok(Emotion::Surprise),
            "disgust" => Ok(Emotion::Disgust),
            "excited" => Ok(Emotion::Excited),
            "calm" => Ok(Emotion::Calm),
            "tender" => Ok(Emotion::Tender),
            "confident" => Ok(Emotion::Confident),
            "melancholic" => Ok(Emotion::Melancholic),
            _ => Ok(Emotion::Custom(s.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_personality_model_default() {
        let model = PersonalityModel::default();
        assert_eq!(model.big_five.extraversion, 0.5);
        assert_eq!(model.big_five.neuroticism, 0.3);
        assert_eq!(
            model.emotional_tendencies.baseline_emotion,
            Emotion::Neutral
        );
    }

    #[tokio::test]
    async fn test_personality_modifier_creation() {
        let modifier = PersonalityEmotionModifier::default_personality();
        let personality = modifier.get_personality().await;
        assert_eq!(
            personality.emotional_tendencies.baseline_emotion,
            Emotion::Neutral
        );
    }

    #[tokio::test]
    async fn test_big_five_traits_application() {
        let modifier = PersonalityEmotionModifier::default_personality();

        // Create test emotion parameters
        let mut test_params = EmotionParameters::neutral();
        test_params.energy_scale = 1.0;
        test_params.tempo_scale = 1.0;

        // Apply personality modification
        let modified = modifier
            .modify_emotion_for_personality(test_params, Some("general"))
            .await
            .unwrap();

        // Parameters should be modified (exact values depend on default personality)
        assert!(modified.energy_scale > 0.0);
        assert!(modified.tempo_scale > 0.0);
    }

    #[tokio::test]
    async fn test_extraverted_personality() {
        let mut personality = PersonalityModel::default();
        personality.big_five.extraversion = 0.8; // High extraversion

        let modifier = PersonalityEmotionModifier::new(personality);

        let mut test_params = EmotionParameters::neutral();
        test_params.energy_scale = 1.0;
        test_params.tempo_scale = 1.0;

        let modified = modifier
            .modify_emotion_for_personality(test_params.clone(), None)
            .await
            .unwrap();

        // Extraverted personality should increase energy and tempo
        assert!(modified.energy_scale > test_params.energy_scale);
        assert!(modified.tempo_scale >= test_params.tempo_scale);
    }

    #[tokio::test]
    async fn test_neurotic_personality() {
        let mut personality = PersonalityModel::default();
        personality.big_five.neuroticism = 0.8; // High neuroticism

        let modifier = PersonalityEmotionModifier::new(personality);

        // Test with a negative emotion (angry)
        let mut test_params = EmotionParameters::neutral();
        test_params
            .emotion_vector
            .add_emotion(Emotion::Angry, EmotionIntensity::new(0.7));
        test_params.energy_scale = 1.0;

        let modified = modifier
            .modify_emotion_for_personality(test_params.clone(), None)
            .await
            .unwrap();

        // Neurotic personality should intensify negative emotions
        assert!(modified.energy_scale > test_params.energy_scale);
    }

    #[tokio::test]
    async fn test_personality_serialization() {
        let modifier = PersonalityEmotionModifier::default_personality();

        // Export personality
        let json = modifier.export_personality().await.unwrap();
        assert!(!json.is_empty());

        // Create new modifier and import
        let new_modifier = PersonalityEmotionModifier::default_personality();
        new_modifier.import_personality(&json).await.unwrap();

        let original = modifier.get_personality().await;
        let imported = new_modifier.get_personality().await;

        assert_eq!(
            original.big_five.extraversion,
            imported.big_five.extraversion
        );
        assert_eq!(
            original.emotional_tendencies.baseline_emotion,
            imported.emotional_tendencies.baseline_emotion
        );
    }

    #[tokio::test]
    async fn test_personality_adaptation() {
        let modifier = PersonalityEmotionModifier::default_personality();

        // Simulate emotion history
        for _ in 0..20 {
            let mut params = EmotionParameters::neutral();
            params
                .emotion_vector
                .add_emotion(Emotion::Happy, EmotionIntensity::new(0.8));
            modifier
                .modify_emotion_for_personality(params, Some("test"))
                .await
                .unwrap();
        }

        let initial_intensity = modifier
            .get_personality()
            .await
            .emotional_tendencies
            .baseline_intensity;

        // Adapt personality
        modifier.adapt_personality().await.unwrap();

        let adapted_intensity = modifier
            .get_personality()
            .await
            .emotional_tendencies
            .baseline_intensity;

        // Personality should have adapted to the observed high intensity
        assert_ne!(initial_intensity, adapted_intensity);
    }

    #[tokio::test]
    async fn test_personality_stats() {
        let modifier = PersonalityEmotionModifier::default_personality();

        // Add some emotion history
        for _ in 0..10 {
            let mut params = EmotionParameters::neutral();
            params
                .emotion_vector
                .add_emotion(Emotion::Happy, EmotionIntensity::new(0.7));
            modifier
                .modify_emotion_for_personality(params, None)
                .await
                .unwrap();
        }

        let stats = modifier.get_personality_stats().await;
        assert_eq!(stats.baseline_emotion, Emotion::Neutral);
        assert!(stats.observation_count > 0);
        assert!(!stats.big_five_summary.is_empty());
    }

    #[test]
    fn test_emotion_parsing() {
        assert_eq!("happy".parse::<Emotion>().unwrap(), Emotion::Happy);
        assert_eq!("ANGRY".parse::<Emotion>().unwrap(), Emotion::Angry);
        assert_eq!("Neutral".parse::<Emotion>().unwrap(), Emotion::Neutral);

        // Custom emotions are created for unknown strings, not errors
        let custom_result = "invalid_emotion".parse::<Emotion>().unwrap();
        assert_eq!(
            custom_result,
            Emotion::Custom("invalid_emotion".to_string())
        );
    }
}
