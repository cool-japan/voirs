//! Emotion modeling for expressive speech synthesis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{Result, AcousticError};

/// Basic emotion types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmotionType {
    /// Neutral emotion (default)
    Neutral,
    /// Happy/joyful emotion
    Happy,
    /// Sad emotion
    Sad,
    /// Angry emotion
    Angry,
    /// Fearful emotion
    Fear,
    /// Surprised emotion
    Surprise,
    /// Disgusted emotion
    Disgust,
    /// Excited emotion
    Excited,
    /// Calm/relaxed emotion
    Calm,
    /// Loving/affectionate emotion
    Love,
    /// Custom emotion (user-defined)
    Custom(String),
}

impl EmotionType {
    /// Get string representation
    pub fn as_str(&self) -> &str {
        match self {
            EmotionType::Neutral => "neutral",
            EmotionType::Happy => "happy",
            EmotionType::Sad => "sad",
            EmotionType::Angry => "angry",
            EmotionType::Fear => "fear",
            EmotionType::Surprise => "surprise",
            EmotionType::Disgust => "disgust",
            EmotionType::Excited => "excited",
            EmotionType::Calm => "calm",
            EmotionType::Love => "love",
            EmotionType::Custom(name) => name,
        }
    }
    
    /// Parse from string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "neutral" => EmotionType::Neutral,
            "happy" => EmotionType::Happy,
            "sad" => EmotionType::Sad,
            "angry" => EmotionType::Angry,
            "fear" => EmotionType::Fear,
            "surprise" => EmotionType::Surprise,
            "disgust" => EmotionType::Disgust,
            "excited" => EmotionType::Excited,
            "calm" => EmotionType::Calm,
            "love" => EmotionType::Love,
            custom => EmotionType::Custom(custom.to_string()),
        }
    }
    
    /// Get all basic emotion types
    pub fn all_basic() -> Vec<EmotionType> {
        vec![
            EmotionType::Neutral,
            EmotionType::Happy,
            EmotionType::Sad,
            EmotionType::Angry,
            EmotionType::Fear,
            EmotionType::Surprise,
            EmotionType::Disgust,
            EmotionType::Excited,
            EmotionType::Calm,
            EmotionType::Love,
        ]
    }
}

impl Default for EmotionType {
    fn default() -> Self {
        EmotionType::Neutral
    }
}

/// Emotion intensity level
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EmotionIntensity {
    /// Very low intensity (0.0 - 0.2)
    VeryLow,
    /// Low intensity (0.2 - 0.4)
    Low,
    /// Medium intensity (0.4 - 0.6)
    Medium,
    /// High intensity (0.6 - 0.8)
    High,
    /// Very high intensity (0.8 - 1.0)
    VeryHigh,
    /// Custom intensity level (0.0 - 1.0)
    Custom(f32),
}

impl EmotionIntensity {
    /// Get intensity as float value (0.0 - 1.0)
    pub fn as_f32(&self) -> f32 {
        match self {
            EmotionIntensity::VeryLow => 0.1,
            EmotionIntensity::Low => 0.3,
            EmotionIntensity::Medium => 0.5,
            EmotionIntensity::High => 0.7,
            EmotionIntensity::VeryHigh => 0.9,
            EmotionIntensity::Custom(value) => value.clamp(0.0, 1.0),
        }
    }
    
    /// Create from float value
    pub fn from_f32(value: f32) -> Self {
        let clamped = value.clamp(0.0, 1.0);
        match clamped {
            x if x <= 0.2 => EmotionIntensity::VeryLow,
            x if x <= 0.4 => EmotionIntensity::Low,
            x if x <= 0.6 => EmotionIntensity::Medium,
            x if x <= 0.8 => EmotionIntensity::High,
            _ => EmotionIntensity::VeryHigh,
        }
    }
}

impl Default for EmotionIntensity {
    fn default() -> Self {
        EmotionIntensity::Medium
    }
}

/// Emotion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionConfig {
    /// Primary emotion type
    pub emotion_type: EmotionType,
    /// Emotion intensity (0.0 - 1.0)
    pub intensity: EmotionIntensity,
    /// Secondary emotions for blending
    pub secondary_emotions: Vec<(EmotionType, f32)>,
    /// Custom emotion parameters
    pub custom_params: HashMap<String, f32>,
}

impl EmotionConfig {
    /// Create new emotion config
    pub fn new(emotion_type: EmotionType) -> Self {
        Self {
            emotion_type,
            intensity: EmotionIntensity::default(),
            secondary_emotions: Vec::new(),
            custom_params: HashMap::new(),
        }
    }
    
    /// Set emotion intensity
    pub fn with_intensity(mut self, intensity: EmotionIntensity) -> Self {
        self.intensity = intensity;
        self
    }
    
    /// Add secondary emotion
    pub fn with_secondary(mut self, emotion: EmotionType, weight: f32) -> Self {
        self.secondary_emotions.push((emotion, weight.clamp(0.0, 1.0)));
        self
    }
    
    /// Add custom parameter
    pub fn with_custom_param(mut self, name: String, value: f32) -> Self {
        self.custom_params.insert(name, value);
        self
    }
    
    /// Get total emotion intensity
    pub fn total_intensity(&self) -> f32 {
        let primary_intensity = self.intensity.as_f32();
        let secondary_intensity: f32 = self.secondary_emotions
            .iter()
            .map(|(_, weight)| weight)
            .sum();
        
        (primary_intensity + secondary_intensity).min(1.0)
    }
    
    /// Check if emotion is neutral
    pub fn is_neutral(&self) -> bool {
        matches!(self.emotion_type, EmotionType::Neutral) && 
        self.secondary_emotions.is_empty() &&
        self.intensity.as_f32() <= 0.1
    }
    
    /// Blend with another emotion config
    pub fn blend(&self, other: &EmotionConfig, alpha: f32) -> EmotionConfig {
        let alpha = alpha.clamp(0.0, 1.0);
        
        // If alpha is 0, return self; if 1, return other
        if alpha == 0.0 {
            return self.clone();
        }
        if alpha == 1.0 {
            return other.clone();
        }
        
        // Blend intensities
        let blended_intensity = EmotionIntensity::from_f32(
            self.intensity.as_f32() * (1.0 - alpha) + other.intensity.as_f32() * alpha
        );
        
        // For simplicity, use primary emotion of the dominant config
        let primary_emotion = if alpha > 0.5 {
            other.emotion_type.clone()
        } else {
            self.emotion_type.clone()
        };
        
        // Blend custom parameters
        let mut blended_params = self.custom_params.clone();
        for (key, value) in &other.custom_params {
            let self_value = self.custom_params.get(key).unwrap_or(&0.0);
            blended_params.insert(key.clone(), self_value * (1.0 - alpha) + value * alpha);
        }
        
        EmotionConfig {
            emotion_type: primary_emotion,
            intensity: blended_intensity,
            secondary_emotions: Vec::new(), // Simplified for now
            custom_params: blended_params,
        }
    }
}

impl Default for EmotionConfig {
    fn default() -> Self {
        Self::new(EmotionType::Neutral)
            .with_intensity(EmotionIntensity::VeryLow) // Use very low intensity for neutral
    }
}

/// Emotion vector representation for neural models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionVector {
    /// Emotion embedding vector
    pub vector: Vec<f32>,
    /// Vector dimension
    pub dimension: usize,
    /// Associated emotion config
    pub config: EmotionConfig,
}

impl EmotionVector {
    /// Create new emotion vector
    pub fn new(config: EmotionConfig, dimension: usize) -> Self {
        let vector = Self::config_to_vector(&config, dimension);
        Self {
            vector,
            dimension,
            config,
        }
    }
    
    /// Convert emotion config to vector representation
    fn config_to_vector(config: &EmotionConfig, dimension: usize) -> Vec<f32> {
        let mut vector = vec![0.0; dimension];
        
        // Basic approach: use first few dimensions for basic emotions
        let basic_emotions = EmotionType::all_basic();
        let basic_count = basic_emotions.len().min(dimension);
        
        // Set primary emotion
        if let Some(index) = basic_emotions.iter().position(|e| e == &config.emotion_type) {
            if index < basic_count {
                vector[index] = config.intensity.as_f32();
            }
        }
        
        // Add secondary emotions
        for (emotion, weight) in &config.secondary_emotions {
            if let Some(index) = basic_emotions.iter().position(|e| e == emotion) {
                if index < basic_count {
                    vector[index] += weight * 0.5; // Reduced weight for secondary emotions
                }
            }
        }
        
        // Use remaining dimensions for custom parameters
        let mut custom_index = basic_count;
        for (_, value) in &config.custom_params {
            if custom_index < dimension {
                vector[custom_index] = *value;
                custom_index += 1;
            }
        }
        
        // Normalize to prevent overflow
        let max_val = vector.iter().cloned().fold(0.0f32, f32::max);
        if max_val > 1.0 {
            for val in &mut vector {
                *val /= max_val;
            }
        }
        
        vector
    }
    
    /// Get vector as slice
    pub fn as_slice(&self) -> &[f32] {
        &self.vector
    }
    
    /// Interpolate with another emotion vector
    pub fn interpolate(&self, other: &EmotionVector, alpha: f32) -> Result<EmotionVector> {
        if self.dimension != other.dimension {
            return Err(AcousticError::InputError(
                "Emotion vectors must have same dimension".to_string()
            ));
        }
        
        let alpha = alpha.clamp(0.0, 1.0);
        let mut interpolated = Vec::with_capacity(self.dimension);
        
        for (a, b) in self.vector.iter().zip(other.vector.iter()) {
            interpolated.push(a * (1.0 - alpha) + b * alpha);
        }
        
        // Blend configs
        let blended_config = self.config.blend(&other.config, alpha);
        
        Ok(EmotionVector {
            vector: interpolated,
            dimension: self.dimension,
            config: blended_config,
        })
    }
}

/// Emotion model for managing emotion states
#[derive(Debug, Clone)]
pub struct EmotionModel {
    /// Current emotion state
    current_emotion: EmotionConfig,
    /// Emotion history
    emotion_history: Vec<EmotionConfig>,
    /// Maximum history length
    max_history: usize,
    /// Vector dimension for neural models
    vector_dimension: usize,
}

impl EmotionModel {
    /// Create new emotion model
    pub fn new(vector_dimension: usize) -> Self {
        Self {
            current_emotion: EmotionConfig::default(),
            emotion_history: Vec::new(),
            max_history: 10,
            vector_dimension,
        }
    }
    
    /// Set current emotion
    pub fn set_emotion(&mut self, emotion: EmotionConfig) {
        // Add current emotion to history
        self.emotion_history.push(self.current_emotion.clone());
        
        // Trim history if needed
        if self.emotion_history.len() > self.max_history {
            self.emotion_history.remove(0);
        }
        
        self.current_emotion = emotion;
    }
    
    /// Get current emotion
    pub fn get_current_emotion(&self) -> &EmotionConfig {
        &self.current_emotion
    }
    
    /// Get emotion vector for current state
    pub fn get_emotion_vector(&self) -> EmotionVector {
        EmotionVector::new(self.current_emotion.clone(), self.vector_dimension)
    }
    
    /// Transition to new emotion with smooth blending
    pub fn transition_to(&mut self, target_emotion: EmotionConfig, blend_factor: f32) {
        let blended = self.current_emotion.blend(&target_emotion, blend_factor);
        self.set_emotion(blended);
    }
    
    /// Get emotion history
    pub fn get_emotion_history(&self) -> &[EmotionConfig] {
        &self.emotion_history
    }
    
    /// Reset to neutral emotion
    pub fn reset_to_neutral(&mut self) {
        self.set_emotion(EmotionConfig::default());
    }
    
    /// Create quick emotion configs
    pub fn quick_emotion(emotion_type: EmotionType, intensity: f32) -> EmotionConfig {
        EmotionConfig::new(emotion_type)
            .with_intensity(EmotionIntensity::from_f32(intensity))
    }
}

impl Default for EmotionModel {
    fn default() -> Self {
        Self::new(256) // Default vector dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_emotion_type_string_conversion() {
        assert_eq!(EmotionType::Happy.as_str(), "happy");
        assert_eq!(EmotionType::from_str("happy"), EmotionType::Happy);
        assert_eq!(EmotionType::from_str("HAPPY"), EmotionType::Happy);
        assert_eq!(EmotionType::from_str("custom"), EmotionType::Custom("custom".to_string()));
    }
    
    #[test]
    fn test_emotion_intensity() {
        assert_eq!(EmotionIntensity::Medium.as_f32(), 0.5);
        assert_eq!(EmotionIntensity::from_f32(0.3), EmotionIntensity::Low);
        assert_eq!(EmotionIntensity::from_f32(1.5), EmotionIntensity::VeryHigh); // Clamped
    }
    
    #[test]
    fn test_emotion_config() {
        let config = EmotionConfig::new(EmotionType::Happy)
            .with_intensity(EmotionIntensity::High)
            .with_secondary(EmotionType::Excited, 0.3)
            .with_custom_param("energy".to_string(), 0.8);
        
        assert_eq!(config.emotion_type, EmotionType::Happy);
        assert_eq!(config.intensity.as_f32(), 0.7);
        assert_eq!(config.secondary_emotions.len(), 1);
        assert_eq!(config.custom_params.get("energy"), Some(&0.8));
    }
    
    #[test]
    fn test_emotion_config_blend() {
        let config1 = EmotionConfig::new(EmotionType::Happy)
            .with_intensity(EmotionIntensity::from_f32(0.8)); // High (0.7)
        let config2 = EmotionConfig::new(EmotionType::Sad)
            .with_intensity(EmotionIntensity::from_f32(0.6)); // Medium (0.5)
        
        let blended = config1.blend(&config2, 0.5);
        // Blend of High (0.7) and Medium (0.5) with alpha 0.5 = 0.6, which maps to Medium (0.5)
        assert_eq!(blended.intensity.as_f32(), 0.5);
    }
    
    #[test]
    fn test_emotion_vector_creation() {
        let config = EmotionConfig::new(EmotionType::Happy)
            .with_intensity(EmotionIntensity::High);
        
        let vector = EmotionVector::new(config, 64);
        assert_eq!(vector.dimension, 64);
        assert_eq!(vector.vector.len(), 64);
        
        // Happy should be at index 1 in all_basic()
        assert!(vector.vector[1] > 0.0);
    }
    
    #[test]
    fn test_emotion_vector_interpolation() {
        let config1 = EmotionConfig::new(EmotionType::Happy)
            .with_intensity(EmotionIntensity::from_f32(1.0));
        let config2 = EmotionConfig::new(EmotionType::Sad)
            .with_intensity(EmotionIntensity::from_f32(1.0));
        
        let vector1 = EmotionVector::new(config1, 32);
        let vector2 = EmotionVector::new(config2, 32);
        
        let interpolated = vector1.interpolate(&vector2, 0.5).unwrap();
        assert_eq!(interpolated.dimension, 32);
        
        // Should be blend of both emotions
        assert!(interpolated.vector[1] > 0.0); // Happy component
        assert!(interpolated.vector[2] > 0.0); // Sad component
    }
    
    #[test]
    fn test_emotion_model() {
        let mut model = EmotionModel::new(128);
        
        // Initially neutral
        assert!(model.get_current_emotion().is_neutral());
        
        // Set emotion
        let happy_config = EmotionConfig::new(EmotionType::Happy)
            .with_intensity(EmotionIntensity::High);
        model.set_emotion(happy_config);
        
        assert_eq!(model.get_current_emotion().emotion_type, EmotionType::Happy);
        assert_eq!(model.get_emotion_history().len(), 1);
        
        // Transition
        let sad_config = EmotionConfig::new(EmotionType::Sad)
            .with_intensity(EmotionIntensity::Medium);
        model.transition_to(sad_config, 0.3);
        
        // Should be blended
        assert_eq!(model.get_emotion_history().len(), 2);
    }
    
    #[test]
    fn test_emotion_model_reset() {
        let mut model = EmotionModel::new(64);
        
        model.set_emotion(EmotionConfig::new(EmotionType::Angry));
        assert_eq!(model.get_current_emotion().emotion_type, EmotionType::Angry);
        
        model.reset_to_neutral();
        assert!(model.get_current_emotion().is_neutral());
    }
    
    #[test]
    fn test_quick_emotion() {
        let config = EmotionModel::quick_emotion(EmotionType::Excited, 0.9);
        assert_eq!(config.emotion_type, EmotionType::Excited);
        assert_eq!(config.intensity.as_f32(), 0.9);
    }
}