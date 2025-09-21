//! Custom emotion vector definitions and registry
//!
//! This module provides support for user-defined emotion characteristics,
//! allowing users to create custom emotions with specific dimensional properties
//! and behavioral traits.

use crate::types::{Emotion, EmotionDimensions, EmotionIntensity, EmotionParameters};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Custom emotion definition with dimensional characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomEmotionDefinition {
    /// Name of the custom emotion
    pub name: String,
    /// Description of the emotion
    pub description: Option<String>,
    /// Dimensional characteristics (Valence, Arousal, Dominance)
    pub dimensions: EmotionDimensions,
    /// Default prosody parameters
    pub default_prosody: CustomProsodyTemplate,
    /// Voice quality characteristics  
    pub voice_quality: VoiceQualityTemplate,
    /// Cultural context (optional)
    pub cultural_context: Option<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Prosody template for custom emotions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomProsodyTemplate {
    /// Default pitch shift multiplier
    pub pitch_shift: f32,
    /// Default tempo scale multiplier
    pub tempo_scale: f32,
    /// Default energy scale multiplier
    pub energy_scale: f32,
    /// Pitch variation range
    pub pitch_variation: f32,
    /// Tempo variation range
    pub tempo_variation: f32,
}

impl Default for CustomProsodyTemplate {
    fn default() -> Self {
        Self {
            pitch_shift: 1.0,
            tempo_scale: 1.0,
            energy_scale: 1.0,
            pitch_variation: 0.1,
            tempo_variation: 0.1,
        }
    }
}

/// Voice quality template for custom emotions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceQualityTemplate {
    /// Default breathiness level
    pub breathiness: f32,
    /// Default roughness level
    pub roughness: f32,
    /// Brightness/spectral tilt adjustment
    pub brightness: f32,
    /// Resonance adjustment
    pub resonance: f32,
}

impl Default for VoiceQualityTemplate {
    fn default() -> Self {
        Self {
            breathiness: 0.0,
            roughness: 0.0,
            brightness: 0.0,
            resonance: 0.0,
        }
    }
}

/// Builder for creating custom emotion definitions
#[derive(Debug)]
pub struct CustomEmotionBuilder {
    name: String,
    description: Option<String>,
    dimensions: Option<EmotionDimensions>,
    prosody: CustomProsodyTemplate,
    voice_quality: VoiceQualityTemplate,
    cultural_context: Option<String>,
    tags: Vec<String>,
}

impl CustomEmotionBuilder {
    /// Create a new builder for the given emotion name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            dimensions: None,
            prosody: CustomProsodyTemplate::default(),
            voice_quality: VoiceQualityTemplate::default(),
            cultural_context: None,
            tags: Vec::new(),
        }
    }

    /// Set emotion description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set emotion dimensions (valence, arousal, dominance)
    /// Note: Values will be validated during build() and must be in range [-1.0, 1.0]
    pub fn dimensions(mut self, valence: f32, arousal: f32, dominance: f32) -> Self {
        self.dimensions = Some(EmotionDimensions {
            valence,
            arousal,
            dominance,
        });
        self
    }

    /// Set from existing emotion dimensions
    pub fn dimensions_from(mut self, dims: EmotionDimensions) -> Self {
        self.dimensions = Some(dims);
        self
    }

    /// Set prosody characteristics
    pub fn prosody(mut self, pitch_shift: f32, tempo_scale: f32, energy_scale: f32) -> Self {
        self.prosody.pitch_shift = pitch_shift;
        self.prosody.tempo_scale = tempo_scale;
        self.prosody.energy_scale = energy_scale;
        self
    }

    /// Set prosody variation ranges
    pub fn prosody_variation(mut self, pitch_var: f32, tempo_var: f32) -> Self {
        self.prosody.pitch_variation = pitch_var;
        self.prosody.tempo_variation = tempo_var;
        self
    }

    /// Set voice quality characteristics
    pub fn voice_quality(
        mut self,
        breathiness: f32,
        roughness: f32,
        brightness: f32,
        resonance: f32,
    ) -> Self {
        self.voice_quality.breathiness = breathiness;
        self.voice_quality.roughness = roughness;
        self.voice_quality.brightness = brightness;
        self.voice_quality.resonance = resonance;
        self
    }

    /// Set cultural context
    pub fn cultural_context(mut self, context: impl Into<String>) -> Self {
        self.cultural_context = Some(context.into());
        self
    }

    /// Add tags for categorization
    pub fn tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(|t| t.into()));
        self
    }

    /// Add a single tag
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Build the custom emotion definition
    pub fn build(self) -> Result<CustomEmotionDefinition, String> {
        // Validate emotion name
        if self.name.trim().is_empty() {
            return Err("Emotion name cannot be empty".to_string());
        }

        // Validate dimensions are in valid range before clamping
        if let Some(dims) = &self.dimensions {
            if dims.valence.abs() > 1.0 || dims.arousal.abs() > 1.0 || dims.dominance.abs() > 1.0 {
                return Err("Emotion dimensions must be in range [-1.0, 1.0]".to_string());
            }
        }

        // Use clamped dimensions for the final result
        let dimensions = self
            .dimensions
            .map(|d| EmotionDimensions::new(d.valence, d.arousal, d.dominance))
            .unwrap_or_else(EmotionDimensions::neutral);

        Ok(CustomEmotionDefinition {
            name: self.name,
            description: self.description,
            dimensions,
            default_prosody: self.prosody,
            voice_quality: self.voice_quality,
            cultural_context: self.cultural_context,
            tags: self.tags,
        })
    }
}

/// Registry for managing custom emotion definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomEmotionRegistry {
    /// Custom emotion definitions by name
    emotions: HashMap<String, CustomEmotionDefinition>,
    /// Version for compatibility tracking
    version: String,
}

impl CustomEmotionRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            emotions: HashMap::new(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Register a new custom emotion
    pub fn register(&mut self, definition: CustomEmotionDefinition) -> Result<(), String> {
        // Check for duplicate names
        if self.emotions.contains_key(&definition.name) {
            return Err(format!(
                "Emotion '{}' is already registered",
                definition.name
            ));
        }

        self.emotions.insert(definition.name.clone(), definition);
        Ok(())
    }

    /// Remove a custom emotion
    pub fn unregister(&mut self, name: &str) -> Option<CustomEmotionDefinition> {
        self.emotions.remove(name)
    }

    /// Get a custom emotion definition
    pub fn get(&self, name: &str) -> Option<&CustomEmotionDefinition> {
        self.emotions.get(name)
    }

    /// Get mutable reference to a custom emotion definition
    pub fn get_mut(&mut self, name: &str) -> Option<&mut CustomEmotionDefinition> {
        self.emotions.get_mut(name)
    }

    /// List all registered custom emotions
    pub fn list_emotions(&self) -> Vec<&str> {
        self.emotions.keys().map(|s| s.as_str()).collect()
    }

    /// Search emotions by tag
    pub fn search_by_tag(&self, tag: &str) -> Vec<&CustomEmotionDefinition> {
        self.emotions
            .values()
            .filter(|def| def.tags.contains(&tag.to_string()))
            .collect()
    }

    /// Search emotions by cultural context
    pub fn search_by_culture(&self, context: &str) -> Vec<&CustomEmotionDefinition> {
        self.emotions
            .values()
            .filter(|def| {
                def.cultural_context
                    .as_ref()
                    .map(|c| c.contains(context))
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Get dimensions for a custom emotion
    pub fn get_dimensions(&self, emotion_name: &str) -> Option<EmotionDimensions> {
        self.emotions.get(emotion_name).map(|def| def.dimensions)
    }

    /// Create emotion parameters from custom definition
    pub fn create_emotion_parameters(
        &self,
        emotion_name: &str,
        intensity: EmotionIntensity,
    ) -> Option<EmotionParameters> {
        self.emotions.get(emotion_name).map(|def| {
            let mut params = EmotionParameters::neutral();

            // Set up emotion vector
            params
                .emotion_vector
                .add_emotion(Emotion::Custom(emotion_name.to_string()), intensity);

            // Apply prosody template
            params.pitch_shift = def.default_prosody.pitch_shift;
            params.tempo_scale = def.default_prosody.tempo_scale;
            params.energy_scale = def.default_prosody.energy_scale;

            // Apply voice quality template
            params.breathiness = def.voice_quality.breathiness;
            params.roughness = def.voice_quality.roughness;

            // Add custom voice quality parameters
            params
                .custom_params
                .insert("brightness".to_string(), def.voice_quality.brightness);
            params
                .custom_params
                .insert("resonance".to_string(), def.voice_quality.resonance);

            params
        })
    }

    /// Get the number of registered emotions
    pub fn count(&self) -> usize {
        self.emotions.len()
    }

    /// Clear all registered emotions
    pub fn clear(&mut self) {
        self.emotions.clear();
    }

    /// Save registry to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Load registry from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Save registry to file
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load registry from file
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let registry = Self::from_json(&json)?;
        Ok(registry)
    }
}

impl Default for CustomEmotionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for EmotionVector to work with custom emotion registry
pub trait EmotionVectorExt {
    /// Update dimensions using custom emotion registry
    fn update_dimensions_with_registry(&mut self, registry: &CustomEmotionRegistry);
}

impl EmotionVectorExt for crate::types::EmotionVector {
    fn update_dimensions_with_registry(&mut self, registry: &CustomEmotionRegistry) {
        let mut valence = 0.0;
        let mut arousal = 0.0;
        let mut dominance = 0.0;
        let mut total_weight = 0.0;

        for (emotion, intensity) in &self.emotions {
            let weight = intensity.value();
            let (v, a, d) = match emotion {
                Emotion::Happy => (0.8, 0.5, 0.3),
                Emotion::Sad => (-0.7, -0.3, -0.5),
                Emotion::Angry => (-0.5, 0.8, 0.7),
                Emotion::Fear => (-0.6, 0.7, -0.8),
                Emotion::Surprise => (0.2, 0.8, 0.0),
                Emotion::Disgust => (-0.7, 0.3, 0.2),
                Emotion::Calm => (0.3, -0.7, 0.2),
                Emotion::Excited => (0.7, 0.9, 0.5),
                Emotion::Tender => (0.6, -0.2, -0.1),
                Emotion::Confident => (0.5, 0.3, 0.8),
                Emotion::Melancholic => (-0.4, -0.5, -0.3),
                Emotion::Neutral => (0.0, 0.0, 0.0),
                Emotion::Custom(name) => {
                    // Use custom emotion registry for dimensions
                    if let Some(dims) = registry.get_dimensions(name) {
                        (dims.valence, dims.arousal, dims.dominance)
                    } else {
                        (0.0, 0.0, 0.0) // Fallback to neutral
                    }
                }
            };

            valence += v * weight;
            arousal += a * weight;
            dominance += d * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            self.dimensions = EmotionDimensions::new(
                valence / total_weight,
                arousal / total_weight,
                dominance / total_weight,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_emotion_builder() {
        let emotion = CustomEmotionBuilder::new("nostalgic")
            .description("A bittersweet longing for the past")
            .dimensions(-0.2, -0.3, -0.1)
            .prosody(0.9, 0.8, 0.7)
            .voice_quality(0.3, 0.1, -0.2, 0.2)
            .tag("bittersweet")
            .tag("memory")
            .cultural_context("Western")
            .build()
            .unwrap();

        assert_eq!(emotion.name, "nostalgic");
        assert!(emotion.description.is_some());
        assert_eq!(emotion.dimensions.valence, -0.2);
        assert_eq!(emotion.tags.len(), 2);
    }

    #[test]
    fn test_custom_emotion_registry() {
        let mut registry = CustomEmotionRegistry::new();

        let nostalgic = CustomEmotionBuilder::new("nostalgic")
            .dimensions(-0.2, -0.3, -0.1)
            .tag("memory")
            .build()
            .unwrap();

        let euphoric = CustomEmotionBuilder::new("euphoric")
            .dimensions(0.9, 0.8, 0.7)
            .tag("intense")
            .build()
            .unwrap();

        registry.register(nostalgic).unwrap();
        registry.register(euphoric).unwrap();

        assert_eq!(registry.count(), 2);
        assert!(registry.get("nostalgic").is_some());
        assert!(registry.get("euphoric").is_some());

        let memory_emotions = registry.search_by_tag("memory");
        assert_eq!(memory_emotions.len(), 1);
        assert_eq!(memory_emotions[0].name, "nostalgic");
    }

    #[test]
    fn test_emotion_parameters_creation() {
        let mut registry = CustomEmotionRegistry::new();

        let serene = CustomEmotionBuilder::new("serene")
            .description("Calm and peaceful state")
            .dimensions(0.4, -0.6, 0.2)
            .prosody(0.9, 0.8, 0.7)
            .voice_quality(0.2, 0.0, 0.1, 0.3)
            .build()
            .unwrap();

        registry.register(serene).unwrap();

        let params = registry
            .create_emotion_parameters("serene", EmotionIntensity::HIGH)
            .unwrap();
        assert_eq!(params.pitch_shift, 0.9);
        assert_eq!(params.tempo_scale, 0.8);
        assert_eq!(params.energy_scale, 0.7);
        assert_eq!(params.breathiness, 0.2);
    }

    #[test]
    fn test_registry_serialization() {
        let mut registry = CustomEmotionRegistry::new();

        let wistful = CustomEmotionBuilder::new("wistful")
            .description("A gentle sadness")
            .dimensions(-0.3, -0.2, -0.1)
            .build()
            .unwrap();

        registry.register(wistful).unwrap();

        let json = registry.to_json().unwrap();
        let loaded_registry = CustomEmotionRegistry::from_json(&json).unwrap();

        assert_eq!(loaded_registry.count(), 1);
        assert!(loaded_registry.get("wistful").is_some());
    }

    #[test]
    fn test_emotion_vector_custom_dimensions() {
        let mut registry = CustomEmotionRegistry::new();

        let blissful = CustomEmotionBuilder::new("blissful")
            .dimensions(0.9, 0.3, 0.5)
            .build()
            .unwrap();

        registry.register(blissful).unwrap();

        let mut emotion_vector = crate::types::EmotionVector::new();
        emotion_vector.add_emotion(
            Emotion::Custom("blissful".to_string()),
            EmotionIntensity::HIGH,
        );

        // Use the registry extension to update dimensions
        emotion_vector.update_dimensions_with_registry(&registry);

        // Dimensions should now reflect the custom emotion
        assert!((emotion_vector.dimensions.valence - 0.9).abs() < 0.01);
        assert!((emotion_vector.dimensions.arousal - 0.3).abs() < 0.01);
        assert!((emotion_vector.dimensions.dominance - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_builder_validation() {
        // Test empty name validation
        let result = CustomEmotionBuilder::new("").build();
        assert!(result.is_err());

        // Test dimension range validation
        let result = CustomEmotionBuilder::new("test")
            .dimensions(2.0, 0.0, 0.0) // Invalid valence > 1.0
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_registration() {
        let mut registry = CustomEmotionRegistry::new();

        let emotion1 = CustomEmotionBuilder::new("test").build().unwrap();
        let emotion2 = CustomEmotionBuilder::new("test").build().unwrap();

        registry.register(emotion1).unwrap();
        let result = registry.register(emotion2);
        assert!(result.is_err());
    }
}
