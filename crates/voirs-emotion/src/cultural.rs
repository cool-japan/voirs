//! Cross-cultural emotion mapping and adaptation
//!
//! This module provides culture-specific emotion mappings and allows for
//! emotional expressions to be adapted based on cultural context and norms.

use crate::types::{Emotion, EmotionDimensions, EmotionIntensity, EmotionVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cultural context and emotion mapping system
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CulturalContext {
    /// Cultural identifier (e.g., "japanese", "western", "arabic", etc.)
    pub culture_id: String,
    /// Human-readable culture name
    pub culture_name: String,
    /// Culture-specific emotion mappings
    pub emotion_mappings: HashMap<Emotion, CulturalEmotionMapping>,
    /// Expression modifiers based on cultural norms
    pub expression_modifiers: CulturalExpressionModifiers,
    /// Social hierarchy considerations
    pub hierarchy_considerations: HierarchyConsiderations,
}

/// Culture-specific mapping for individual emotions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CulturalEmotionMapping {
    /// Adjusted dimensional values for this culture
    pub adjusted_dimensions: EmotionDimensions,
    /// Intensity modifiers (e.g., some cultures express emotions more/less intensely)
    pub intensity_modifier: f32,
    /// Social appropriateness in different contexts
    pub appropriateness: HashMap<SocialContext, AppropratenessLevel>,
    /// Alternative expressions or substitutions
    pub alternative_expressions: Vec<String>,
    /// Cultural-specific prosody adjustments
    pub prosody_adjustments: CulturalProsodyAdjustment,
}

/// Expression modifiers based on cultural communication patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CulturalExpressionModifiers {
    /// Overall emotional expressiveness (0.0 = very reserved, 2.0 = very expressive)
    pub expressiveness_scale: f32,
    /// Preference for indirect vs direct emotional expression
    pub directness_preference: f32, // 0.0 = very indirect, 1.0 = very direct
    /// Formality level adjustments
    pub formality_adjustment: f32,
    /// Politeness considerations
    pub politeness_factor: f32,
}

/// Social hierarchy considerations for emotion expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchyConsiderations {
    /// Whether hierarchy affects emotion expression
    pub hierarchy_sensitive: bool,
    /// Modifiers for speaking to superiors
    pub superior_modifiers: EmotionExpressionModifier,
    /// Modifiers for speaking to peers
    pub peer_modifiers: EmotionExpressionModifier,
    /// Modifiers for speaking to subordinates
    pub subordinate_modifiers: EmotionExpressionModifier,
}

/// Specific modifiers for emotion expression based on social relationship
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionExpressionModifier {
    /// Intensity scaling factor
    pub intensity_scale: f32,
    /// Valence adjustment (shift toward positive/negative)
    pub valence_adjustment: f32,
    /// Arousal adjustment (energy level adjustment)
    pub arousal_adjustment: f32,
    /// Dominance adjustment (confidence/assertiveness adjustment)
    pub dominance_adjustment: f32,
}

/// Cultural-specific prosody adjustments
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CulturalProsodyAdjustment {
    /// Pitch range modifications
    pub pitch_range_modifier: f32,
    /// Speaking rate adjustments
    pub rate_modifier: f32,
    /// Pause and timing adjustments
    pub pause_modifier: f32,
    /// Volume adjustments
    pub volume_modifier: f32,
}

/// Social context for appropriateness evaluation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SocialContext {
    /// Formal settings requiring proper etiquette and reserved emotional expression
    Formal,
    /// Casual, relaxed settings with fewer social constraints
    Informal,
    /// Work or business-related contexts with professional conduct expectations
    Professional,
    /// Personal, intimate settings with close relationships
    Personal,
    /// Open, public spaces with many observers
    Public,
    /// Private settings with limited audience
    Private,
    /// Family gatherings or interactions with relatives
    Family,
    /// Interactions with unfamiliar people or strangers
    Strangers,
}

/// Level of social appropriateness
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AppropratenessLevel {
    /// Highly appropriate and encouraged
    Encouraged,
    /// Socially appropriate
    Appropriate,
    /// Neutral - neither encouraged nor discouraged
    Neutral,
    /// Mildly inappropriate but tolerated
    Discouraged,
    /// Socially inappropriate
    Inappropriate,
    /// Completely unacceptable
    Unacceptable,
}

/// Cultural emotion adaptation system
#[derive(Debug)]
pub struct CulturalEmotionAdapter {
    /// Available cultural contexts
    pub cultural_contexts: HashMap<String, CulturalContext>,
    /// Currently active cultural context
    pub active_context: Option<String>,
}

impl CulturalEmotionAdapter {
    /// Create new cultural adapter
    pub fn new() -> Self {
        let mut adapter = Self {
            cultural_contexts: HashMap::new(),
            active_context: None,
        };

        // Add default cultural contexts
        adapter.add_default_cultures();
        adapter
    }

    /// Set the active cultural context
    pub fn set_active_culture(&mut self, culture_id: &str) -> Result<(), String> {
        if self.cultural_contexts.contains_key(culture_id) {
            self.active_context = Some(culture_id.to_string());
            Ok(())
        } else {
            Err(format!("Cultural context '{}' not found", culture_id))
        }
    }

    /// Get the currently active cultural context
    pub fn get_active_context(&self) -> Option<&CulturalContext> {
        self.active_context
            .as_ref()
            .and_then(|id| self.cultural_contexts.get(id))
    }

    /// Register a new cultural context
    pub fn register_culture(&mut self, context: CulturalContext) {
        self.cultural_contexts
            .insert(context.culture_id.clone(), context);
    }

    /// Adapt emotion vector based on active cultural context
    pub fn adapt_emotion(
        &self,
        emotion: &Emotion,
        intensity: EmotionIntensity,
        social_context: SocialContext,
        hierarchy: Option<SocialHierarchy>,
    ) -> EmotionVector {
        let base_vector = self.create_base_emotion_vector(emotion, intensity);

        if let Some(context) = self.get_active_context() {
            self.apply_cultural_adaptation(&base_vector, context, social_context, hierarchy)
        } else {
            base_vector
        }
    }

    /// Create base emotion vector without cultural adaptation
    fn create_base_emotion_vector(
        &self,
        emotion: &Emotion,
        intensity: EmotionIntensity,
    ) -> EmotionVector {
        let mut vector = EmotionVector::new();
        vector.add_emotion(emotion.clone(), intensity);
        vector
    }

    /// Apply cultural adaptation to emotion vector
    fn apply_cultural_adaptation(
        &self,
        base_vector: &EmotionVector,
        context: &CulturalContext,
        social_context: SocialContext,
        hierarchy: Option<SocialHierarchy>,
    ) -> EmotionVector {
        let mut adapted_vector = base_vector.clone();

        // Apply cultural mappings for each emotion
        for (emotion, intensity) in &base_vector.emotions {
            if let Some(mapping) = context.emotion_mappings.get(emotion) {
                // Apply intensity modifier
                let modified_intensity = EmotionIntensity::new(
                    intensity.value()
                        * mapping.intensity_modifier
                        * context.expression_modifiers.expressiveness_scale,
                );

                // Check social appropriateness
                let appropriateness = mapping
                    .appropriateness
                    .get(&social_context)
                    .unwrap_or(&AppropratenessLevel::Neutral);

                let final_intensity = match appropriateness {
                    AppropratenessLevel::Unacceptable => EmotionIntensity::new(0.0),
                    AppropratenessLevel::Inappropriate => {
                        EmotionIntensity::new(modified_intensity.value() * 0.2)
                    }
                    AppropratenessLevel::Discouraged => {
                        EmotionIntensity::new(modified_intensity.value() * 0.5)
                    }
                    AppropratenessLevel::Neutral => modified_intensity,
                    AppropratenessLevel::Appropriate => modified_intensity,
                    AppropratenessLevel::Encouraged => {
                        EmotionIntensity::new(modified_intensity.value() * 1.2)
                    }
                };

                adapted_vector
                    .emotions
                    .insert(emotion.clone(), final_intensity);

                // Apply dimensional adjustments based on cultural mapping
                let base_dims = &base_vector.dimensions;
                let cultural_dims = &mapping.adjusted_dimensions;

                adapted_vector.dimensions = EmotionDimensions::new(
                    base_dims.valence + (cultural_dims.valence - base_dims.valence) * 0.5,
                    base_dims.arousal + (cultural_dims.arousal - base_dims.arousal) * 0.5,
                    base_dims.dominance + (cultural_dims.dominance - base_dims.dominance) * 0.5,
                );
            } else {
                // If no cultural mapping exists, use the original emotion with slight cultural moderation
                let cultural_intensity = EmotionIntensity::new(
                    intensity.value() * context.expression_modifiers.expressiveness_scale,
                );
                adapted_vector
                    .emotions
                    .insert(emotion.clone(), cultural_intensity);
            }
        }

        // Apply hierarchy considerations
        if let (Some(hierarchy_level), true) = (
            hierarchy,
            context.hierarchy_considerations.hierarchy_sensitive,
        ) {
            adapted_vector =
                self.apply_hierarchy_adjustments(adapted_vector, context, hierarchy_level);
        }

        adapted_vector
    }

    /// Apply social hierarchy adjustments
    fn apply_hierarchy_adjustments(
        &self,
        mut vector: EmotionVector,
        context: &CulturalContext,
        hierarchy: SocialHierarchy,
    ) -> EmotionVector {
        let modifier = match hierarchy {
            SocialHierarchy::Superior => &context.hierarchy_considerations.superior_modifiers,
            SocialHierarchy::Peer => &context.hierarchy_considerations.peer_modifiers,
            SocialHierarchy::Subordinate => &context.hierarchy_considerations.subordinate_modifiers,
        };

        // Adjust emotion intensities
        for intensity in vector.emotions.values_mut() {
            *intensity = EmotionIntensity::new(intensity.value() * modifier.intensity_scale);
        }

        // Adjust dimensions
        vector.dimensions = EmotionDimensions::new(
            (vector.dimensions.valence + modifier.valence_adjustment).clamp(-1.0, 1.0),
            (vector.dimensions.arousal + modifier.arousal_adjustment).clamp(-1.0, 1.0),
            (vector.dimensions.dominance + modifier.dominance_adjustment).clamp(-1.0, 1.0),
        );

        vector
    }

    /// Add default cultural contexts
    fn add_default_cultures(&mut self) {
        // Japanese culture - emphasis on harmony and indirect expression
        let japanese = self.create_japanese_culture();
        self.register_culture(japanese);

        // Western culture - more direct emotional expression
        let western = self.create_western_culture();
        self.register_culture(western);

        // Arabic culture - expressive with formal considerations
        let arabic = self.create_arabic_culture();
        self.register_culture(arabic);

        // Scandinavian culture - reserved but genuine expression
        let scandinavian = self.create_scandinavian_culture();
        self.register_culture(scandinavian);
    }

    /// Create Japanese cultural context
    fn create_japanese_culture(&self) -> CulturalContext {
        let mut emotion_mappings = HashMap::new();

        // Happy - more subdued in Japanese culture
        emotion_mappings.insert(
            Emotion::Happy,
            CulturalEmotionMapping {
                adjusted_dimensions: EmotionDimensions::new(0.6, 0.3, 0.2),
                intensity_modifier: 0.7,
                appropriateness: {
                    let mut app = HashMap::new();
                    app.insert(SocialContext::Formal, AppropratenessLevel::Discouraged);
                    app.insert(SocialContext::Professional, AppropratenessLevel::Neutral);
                    app.insert(SocialContext::Personal, AppropratenessLevel::Appropriate);
                    app
                },
                alternative_expressions: vec!["content".to_string(), "satisfied".to_string()],
                prosody_adjustments: CulturalProsodyAdjustment {
                    pitch_range_modifier: 0.8,
                    rate_modifier: 0.9,
                    pause_modifier: 1.2,
                    volume_modifier: 0.8,
                },
            },
        );

        // Angry - strongly discouraged in Japanese culture
        emotion_mappings.insert(
            Emotion::Angry,
            CulturalEmotionMapping {
                adjusted_dimensions: EmotionDimensions::new(-0.3, 0.4, -0.2),
                intensity_modifier: 0.3,
                appropriateness: {
                    let mut app = HashMap::new();
                    app.insert(SocialContext::Formal, AppropratenessLevel::Unacceptable);
                    app.insert(
                        SocialContext::Professional,
                        AppropratenessLevel::Inappropriate,
                    );
                    app.insert(SocialContext::Personal, AppropratenessLevel::Discouraged);
                    app
                },
                alternative_expressions: vec!["troubled".to_string(), "concerned".to_string()],
                prosody_adjustments: CulturalProsodyAdjustment {
                    pitch_range_modifier: 0.6,
                    rate_modifier: 0.8,
                    pause_modifier: 1.5,
                    volume_modifier: 0.6,
                },
            },
        );

        CulturalContext {
            culture_id: "japanese".to_string(),
            culture_name: "Japanese".to_string(),
            emotion_mappings,
            expression_modifiers: CulturalExpressionModifiers {
                expressiveness_scale: 0.6,
                directness_preference: 0.2,
                formality_adjustment: 1.4,
                politeness_factor: 1.8,
            },
            hierarchy_considerations: HierarchyConsiderations {
                hierarchy_sensitive: true,
                superior_modifiers: EmotionExpressionModifier {
                    intensity_scale: 0.5,
                    valence_adjustment: 0.1,
                    arousal_adjustment: -0.2,
                    dominance_adjustment: -0.4,
                },
                peer_modifiers: EmotionExpressionModifier {
                    intensity_scale: 0.8,
                    valence_adjustment: 0.0,
                    arousal_adjustment: -0.1,
                    dominance_adjustment: -0.1,
                },
                subordinate_modifiers: EmotionExpressionModifier {
                    intensity_scale: 1.0,
                    valence_adjustment: 0.0,
                    arousal_adjustment: 0.0,
                    dominance_adjustment: 0.1,
                },
            },
        }
    }

    /// Create Western cultural context
    fn create_western_culture(&self) -> CulturalContext {
        let mut emotion_mappings = HashMap::new();

        // Happy - generally appropriate in Western culture
        emotion_mappings.insert(
            Emotion::Happy,
            CulturalEmotionMapping {
                adjusted_dimensions: EmotionDimensions::new(0.8, 0.5, 0.3),
                intensity_modifier: 1.1,
                appropriateness: {
                    let mut app = HashMap::new();
                    app.insert(SocialContext::Formal, AppropratenessLevel::Appropriate);
                    app.insert(
                        SocialContext::Professional,
                        AppropratenessLevel::Appropriate,
                    );
                    app.insert(SocialContext::Personal, AppropratenessLevel::Encouraged);
                    app
                },
                alternative_expressions: vec!["pleased".to_string(), "joyful".to_string()],
                prosody_adjustments: CulturalProsodyAdjustment {
                    pitch_range_modifier: 1.2,
                    rate_modifier: 1.1,
                    pause_modifier: 0.9,
                    volume_modifier: 1.1,
                },
            },
        );

        // Angry - more direct expression acceptable
        emotion_mappings.insert(
            Emotion::Angry,
            CulturalEmotionMapping {
                adjusted_dimensions: EmotionDimensions::new(-0.5, 0.8, 0.6),
                intensity_modifier: 0.9,
                appropriateness: {
                    let mut app = HashMap::new();
                    app.insert(SocialContext::Formal, AppropratenessLevel::Discouraged);
                    app.insert(SocialContext::Professional, AppropratenessLevel::Neutral);
                    app.insert(SocialContext::Personal, AppropratenessLevel::Appropriate);
                    app
                },
                alternative_expressions: vec!["frustrated".to_string(), "upset".to_string()],
                prosody_adjustments: CulturalProsodyAdjustment {
                    pitch_range_modifier: 1.3,
                    rate_modifier: 1.2,
                    pause_modifier: 0.7,
                    volume_modifier: 1.3,
                },
            },
        );

        CulturalContext {
            culture_id: "western".to_string(),
            culture_name: "Western".to_string(),
            emotion_mappings,
            expression_modifiers: CulturalExpressionModifiers {
                expressiveness_scale: 1.2,
                directness_preference: 0.8,
                formality_adjustment: 1.0,
                politeness_factor: 1.0,
            },
            hierarchy_considerations: HierarchyConsiderations {
                hierarchy_sensitive: false,
                superior_modifiers: EmotionExpressionModifier {
                    intensity_scale: 0.9,
                    valence_adjustment: 0.0,
                    arousal_adjustment: 0.0,
                    dominance_adjustment: -0.1,
                },
                peer_modifiers: EmotionExpressionModifier {
                    intensity_scale: 1.0,
                    valence_adjustment: 0.0,
                    arousal_adjustment: 0.0,
                    dominance_adjustment: 0.0,
                },
                subordinate_modifiers: EmotionExpressionModifier {
                    intensity_scale: 1.0,
                    valence_adjustment: 0.0,
                    arousal_adjustment: 0.0,
                    dominance_adjustment: 0.0,
                },
            },
        }
    }

    /// Create Arabic cultural context
    fn create_arabic_culture(&self) -> CulturalContext {
        let mut emotion_mappings = HashMap::new();

        // Happy - very expressive in Arabic culture
        emotion_mappings.insert(
            Emotion::Happy,
            CulturalEmotionMapping {
                adjusted_dimensions: EmotionDimensions::new(0.9, 0.7, 0.4),
                intensity_modifier: 1.3,
                appropriateness: {
                    let mut app = HashMap::new();
                    app.insert(SocialContext::Formal, AppropratenessLevel::Appropriate);
                    app.insert(
                        SocialContext::Professional,
                        AppropratenessLevel::Appropriate,
                    );
                    app.insert(SocialContext::Personal, AppropratenessLevel::Encouraged);
                    app
                },
                alternative_expressions: vec!["delighted".to_string(), "blessed".to_string()],
                prosody_adjustments: CulturalProsodyAdjustment {
                    pitch_range_modifier: 1.4,
                    rate_modifier: 1.2,
                    pause_modifier: 0.8,
                    volume_modifier: 1.4,
                },
            },
        );

        CulturalContext {
            culture_id: "arabic".to_string(),
            culture_name: "Arabic".to_string(),
            emotion_mappings,
            expression_modifiers: CulturalExpressionModifiers {
                expressiveness_scale: 1.4,
                directness_preference: 0.6,
                formality_adjustment: 1.2,
                politeness_factor: 1.3,
            },
            hierarchy_considerations: HierarchyConsiderations {
                hierarchy_sensitive: true,
                superior_modifiers: EmotionExpressionModifier {
                    intensity_scale: 0.7,
                    valence_adjustment: 0.2,
                    arousal_adjustment: -0.1,
                    dominance_adjustment: -0.3,
                },
                peer_modifiers: EmotionExpressionModifier {
                    intensity_scale: 1.0,
                    valence_adjustment: 0.0,
                    arousal_adjustment: 0.0,
                    dominance_adjustment: 0.0,
                },
                subordinate_modifiers: EmotionExpressionModifier {
                    intensity_scale: 1.1,
                    valence_adjustment: 0.0,
                    arousal_adjustment: 0.1,
                    dominance_adjustment: 0.2,
                },
            },
        }
    }

    /// Create Scandinavian cultural context
    fn create_scandinavian_culture(&self) -> CulturalContext {
        let mut emotion_mappings = HashMap::new();

        // Happy - reserved but genuine in Scandinavian culture
        emotion_mappings.insert(
            Emotion::Happy,
            CulturalEmotionMapping {
                adjusted_dimensions: EmotionDimensions::new(0.7, 0.4, 0.2),
                intensity_modifier: 0.8,
                appropriateness: {
                    let mut app = HashMap::new();
                    app.insert(SocialContext::Formal, AppropratenessLevel::Neutral);
                    app.insert(
                        SocialContext::Professional,
                        AppropratenessLevel::Appropriate,
                    );
                    app.insert(SocialContext::Personal, AppropratenessLevel::Appropriate);
                    app
                },
                alternative_expressions: vec!["content".to_string(), "pleased".to_string()],
                prosody_adjustments: CulturalProsodyAdjustment {
                    pitch_range_modifier: 0.9,
                    rate_modifier: 0.95,
                    pause_modifier: 1.1,
                    volume_modifier: 0.9,
                },
            },
        );

        CulturalContext {
            culture_id: "scandinavian".to_string(),
            culture_name: "Scandinavian".to_string(),
            emotion_mappings,
            expression_modifiers: CulturalExpressionModifiers {
                expressiveness_scale: 0.8,
                directness_preference: 0.6,
                formality_adjustment: 0.9,
                politeness_factor: 1.1,
            },
            hierarchy_considerations: HierarchyConsiderations {
                hierarchy_sensitive: false,
                superior_modifiers: EmotionExpressionModifier {
                    intensity_scale: 1.0,
                    valence_adjustment: 0.0,
                    arousal_adjustment: 0.0,
                    dominance_adjustment: 0.0,
                },
                peer_modifiers: EmotionExpressionModifier {
                    intensity_scale: 1.0,
                    valence_adjustment: 0.0,
                    arousal_adjustment: 0.0,
                    dominance_adjustment: 0.0,
                },
                subordinate_modifiers: EmotionExpressionModifier {
                    intensity_scale: 1.0,
                    valence_adjustment: 0.0,
                    arousal_adjustment: 0.0,
                    dominance_adjustment: 0.0,
                },
            },
        }
    }
}

/// Social hierarchy relationships
#[derive(Debug, Clone, PartialEq)]
pub enum SocialHierarchy {
    /// Speaking to someone of higher status
    Superior,
    /// Speaking to someone of equal status
    Peer,
    /// Speaking to someone of lower status
    Subordinate,
}

impl Default for CulturalEmotionAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CulturalExpressionModifiers {
    fn default() -> Self {
        Self {
            expressiveness_scale: 1.0,
            directness_preference: 0.5,
            formality_adjustment: 1.0,
            politeness_factor: 1.0,
        }
    }
}

impl Default for EmotionExpressionModifier {
    fn default() -> Self {
        Self {
            intensity_scale: 1.0,
            valence_adjustment: 0.0,
            arousal_adjustment: 0.0,
            dominance_adjustment: 0.0,
        }
    }
}

impl Default for CulturalProsodyAdjustment {
    fn default() -> Self {
        Self {
            pitch_range_modifier: 1.0,
            rate_modifier: 1.0,
            pause_modifier: 1.0,
            volume_modifier: 1.0,
        }
    }
}

impl Default for HierarchyConsiderations {
    fn default() -> Self {
        Self {
            hierarchy_sensitive: false,
            superior_modifiers: EmotionExpressionModifier::default(),
            peer_modifiers: EmotionExpressionModifier::default(),
            subordinate_modifiers: EmotionExpressionModifier::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cultural_adapter_creation() {
        let adapter = CulturalEmotionAdapter::new();

        // Should have default cultures
        assert!(adapter.cultural_contexts.contains_key("japanese"));
        assert!(adapter.cultural_contexts.contains_key("western"));
        assert!(adapter.cultural_contexts.contains_key("arabic"));
        assert!(adapter.cultural_contexts.contains_key("scandinavian"));
    }

    #[test]
    fn test_set_active_culture() {
        let mut adapter = CulturalEmotionAdapter::new();

        // Should successfully set existing culture
        assert!(adapter.set_active_culture("japanese").is_ok());
        assert_eq!(adapter.active_context, Some("japanese".to_string()));

        // Should fail for non-existent culture
        assert!(adapter.set_active_culture("nonexistent").is_err());
    }

    #[test]
    fn test_cultural_emotion_adaptation() {
        let mut adapter = CulturalEmotionAdapter::new();
        adapter.set_active_culture("japanese").unwrap();

        // Test happy emotion adaptation in Japanese culture
        let adapted = adapter.adapt_emotion(
            &Emotion::Happy,
            EmotionIntensity::HIGH,
            SocialContext::Formal,
            Some(SocialHierarchy::Superior),
        );

        // Japanese culture should reduce emotional intensity, especially in formal contexts with superiors
        let happy_intensity = adapted
            .emotions
            .get(&Emotion::Happy)
            .map(|i| i.value())
            .unwrap_or(0.0);

        assert!(happy_intensity < EmotionIntensity::HIGH.value());

        // Dimensions should be adjusted for Japanese cultural norms
        assert!(adapted.dimensions.valence > 0.0); // Still positive but reduced
        assert!(adapted.dimensions.arousal < 0.5); // Lower arousal in formal settings
        assert!(adapted.dimensions.dominance < 0.0); // Lower dominance when speaking to superior
    }

    #[test]
    fn test_western_vs_japanese_adaptation() {
        let mut adapter = CulturalEmotionAdapter::new();

        // Test same emotion in different cultures
        adapter.set_active_culture("western").unwrap();
        let western_adapted = adapter.adapt_emotion(
            &Emotion::Happy,
            EmotionIntensity::HIGH,
            SocialContext::Personal,
            None,
        );

        adapter.set_active_culture("japanese").unwrap();
        let japanese_adapted = adapter.adapt_emotion(
            &Emotion::Happy,
            EmotionIntensity::HIGH,
            SocialContext::Personal,
            None,
        );

        // Western culture should be more expressive
        let western_intensity = western_adapted
            .emotions
            .get(&Emotion::Happy)
            .map(|i| i.value())
            .unwrap_or(0.0);
        let japanese_intensity = japanese_adapted
            .emotions
            .get(&Emotion::Happy)
            .map(|i| i.value())
            .unwrap_or(0.0);

        assert!(western_intensity > japanese_intensity);
    }

    #[test]
    fn test_social_context_appropriateness() {
        let mut adapter = CulturalEmotionAdapter::new();
        adapter.set_active_culture("japanese").unwrap();

        // Test angry emotion in formal context (should be heavily reduced)
        let formal_anger = adapter.adapt_emotion(
            &Emotion::Angry,
            EmotionIntensity::HIGH,
            SocialContext::Formal,
            None,
        );

        // Test angry emotion in personal context (should be less reduced)
        let personal_anger = adapter.adapt_emotion(
            &Emotion::Angry,
            EmotionIntensity::HIGH,
            SocialContext::Personal,
            None,
        );

        let formal_intensity = formal_anger
            .emotions
            .get(&Emotion::Angry)
            .map(|i| i.value())
            .unwrap_or(0.0);
        let personal_intensity = personal_anger
            .emotions
            .get(&Emotion::Angry)
            .map(|i| i.value())
            .unwrap_or(0.0);

        // Anger should be more suppressed in formal contexts
        assert!(formal_intensity < personal_intensity);
    }

    #[test]
    fn test_hierarchy_sensitive_adaptation() {
        let mut adapter = CulturalEmotionAdapter::new();
        adapter.set_active_culture("japanese").unwrap(); // Hierarchy-sensitive culture

        let superior_adapted = adapter.adapt_emotion(
            &Emotion::Confident,
            EmotionIntensity::HIGH,
            SocialContext::Professional,
            Some(SocialHierarchy::Superior),
        );

        let subordinate_adapted = adapter.adapt_emotion(
            &Emotion::Confident,
            EmotionIntensity::HIGH,
            SocialContext::Professional,
            Some(SocialHierarchy::Subordinate),
        );

        // Confidence should be reduced when speaking to superiors
        assert!(superior_adapted.dimensions.dominance < subordinate_adapted.dimensions.dominance);
    }
}
