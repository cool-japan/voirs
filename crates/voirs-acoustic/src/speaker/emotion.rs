//! Emotion modeling for expressive speech synthesis.

use crate::{AcousticError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;

/// Basic emotion types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EmotionType {
    /// Neutral emotion (default)
    #[default]
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
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        s.parse()
            .unwrap_or_else(|_| EmotionType::Custom(s.to_string()))
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

impl FromStr for EmotionType {
    type Err = ();

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
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
        })
    }
}

/// Emotion intensity level
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum EmotionIntensity {
    /// Very low intensity (0.0 - 0.2)
    VeryLow,
    /// Low intensity (0.2 - 0.4)
    Low,
    /// Medium intensity (0.4 - 0.6)
    #[default]
    Medium,
    /// High intensity (0.6 - 0.8)
    High,
    /// Very high intensity (0.8 - 1.0)
    VeryHigh,
    /// Custom intensity level (0.0 - 1.0)
    Custom(f32),
}

impl std::fmt::Display for EmotionIntensity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmotionIntensity::VeryLow => write!(f, "very_low"),
            EmotionIntensity::Low => write!(f, "low"),
            EmotionIntensity::Medium => write!(f, "medium"),
            EmotionIntensity::High => write!(f, "high"),
            EmotionIntensity::VeryHigh => write!(f, "very_high"),
            EmotionIntensity::Custom(value) => write!(f, "custom({value})"),
        }
    }
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
        self.secondary_emotions
            .push((emotion, weight.clamp(0.0, 1.0)));
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
        let secondary_intensity: f32 = self
            .secondary_emotions
            .iter()
            .map(|(_, weight)| weight)
            .sum();

        (primary_intensity + secondary_intensity).min(1.0)
    }

    /// Check if emotion is neutral
    pub fn is_neutral(&self) -> bool {
        matches!(self.emotion_type, EmotionType::Neutral)
            && self.secondary_emotions.is_empty()
            && self.intensity.as_f32() <= 0.1
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
            self.intensity.as_f32() * (1.0 - alpha) + other.intensity.as_f32() * alpha,
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
        Self::new(EmotionType::Neutral).with_intensity(EmotionIntensity::VeryLow)
        // Use very low intensity for neutral
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
        if let Some(index) = basic_emotions
            .iter()
            .position(|e| e == &config.emotion_type)
        {
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
        for value in config.custom_params.values() {
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
                "Emotion vectors must have same dimension".to_string(),
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
        EmotionConfig::new(emotion_type).with_intensity(EmotionIntensity::from_f32(intensity))
    }
}

impl Default for EmotionModel {
    fn default() -> Self {
        Self::new(256) // Default vector dimension
    }
}

/// Emotion parameter validation and preprocessing
pub struct EmotionValidator {
    /// Supported emotion types
    supported_emotions: Vec<EmotionType>,
    /// Minimum intensity threshold
    min_intensity: f32,
    /// Maximum intensity threshold
    max_intensity: f32,
    /// Maximum number of secondary emotions
    max_secondary_emotions: usize,
}

impl EmotionValidator {
    /// Create new emotion validator
    pub fn new() -> Self {
        Self {
            supported_emotions: EmotionType::all_basic(),
            min_intensity: 0.0,
            max_intensity: 1.0,
            max_secondary_emotions: 5,
        }
    }

    /// Create validator with custom settings
    pub fn with_settings(
        supported_emotions: Vec<EmotionType>,
        min_intensity: f32,
        max_intensity: f32,
        max_secondary_emotions: usize,
    ) -> Self {
        Self {
            supported_emotions,
            min_intensity,
            max_intensity,
            max_secondary_emotions,
        }
    }

    /// Validate emotion configuration
    pub fn validate(&self, config: &EmotionConfig) -> Result<()> {
        // Validate primary emotion type
        if !self.is_emotion_supported(&config.emotion_type) {
            return Err(AcousticError::InputError(format!(
                "Emotion type {:?} is not supported",
                config.emotion_type
            )));
        }

        // Validate intensity
        let intensity_value = config.intensity.as_f32();
        if intensity_value < self.min_intensity || intensity_value > self.max_intensity {
            return Err(AcousticError::InputError(format!(
                "Emotion intensity {} is out of range [{}, {}]",
                intensity_value, self.min_intensity, self.max_intensity
            )));
        }

        // Validate secondary emotions
        if config.secondary_emotions.len() > self.max_secondary_emotions {
            return Err(AcousticError::InputError(format!(
                "Too many secondary emotions: {} (max: {})",
                config.secondary_emotions.len(),
                self.max_secondary_emotions
            )));
        }

        for (emotion, weight) in &config.secondary_emotions {
            if !self.is_emotion_supported(emotion) {
                return Err(AcousticError::InputError(format!(
                    "Secondary emotion type {emotion:?} is not supported"
                )));
            }

            if *weight < 0.0 || *weight > 1.0 {
                return Err(AcousticError::InputError(format!(
                    "Secondary emotion weight {weight} is out of range [0.0, 1.0]"
                )));
            }
        }

        // Validate custom parameters
        for (param_name, param_value) in &config.custom_params {
            if param_name.is_empty() {
                return Err(AcousticError::InputError(
                    "Custom parameter name cannot be empty".to_string(),
                ));
            }

            if !param_value.is_finite() {
                return Err(AcousticError::InputError(format!(
                    "Custom parameter '{param_name}' has invalid value: {param_value}"
                )));
            }
        }

        Ok(())
    }

    /// Check if emotion type is supported
    fn is_emotion_supported(&self, emotion: &EmotionType) -> bool {
        match emotion {
            EmotionType::Custom(_) => true, // Custom emotions are always supported
            _ => self.supported_emotions.contains(emotion),
        }
    }

    /// Preprocess emotion configuration
    pub fn preprocess(&self, config: &mut EmotionConfig) -> Result<()> {
        // Validate first
        self.validate(config)?;

        // Normalize intensity
        let intensity_value = config.intensity.as_f32();
        let normalized_intensity = intensity_value.clamp(self.min_intensity, self.max_intensity);
        config.intensity = EmotionIntensity::from_f32(normalized_intensity);

        // Normalize secondary emotion weights
        for (_, weight) in &mut config.secondary_emotions {
            *weight = weight.clamp(0.0, 1.0);
        }

        // Remove duplicate secondary emotions (keep the last one)
        let mut unique_emotions = HashMap::new();
        for (emotion, weight) in &config.secondary_emotions {
            unique_emotions.insert(emotion.clone(), *weight);
        }
        config.secondary_emotions = unique_emotions.into_iter().collect();

        // Sort secondary emotions by weight (descending)
        config
            .secondary_emotions
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Limit secondary emotions to max count
        config
            .secondary_emotions
            .truncate(self.max_secondary_emotions);

        // Clamp custom parameters to reasonable ranges
        for (param_name, param_value) in &mut config.custom_params {
            match param_name.as_str() {
                "energy" | "arousal" | "valence" => {
                    *param_value = param_value.clamp(0.0, 1.0);
                }
                "pitch_shift" => {
                    *param_value = param_value.clamp(-2.0, 2.0); // Semitones
                }
                "speed_factor" => {
                    *param_value = param_value.clamp(0.5, 2.0); // 0.5x to 2x speed
                }
                _ => {
                    // Generic parameter, clamp to [-1.0, 1.0]
                    *param_value = param_value.clamp(-1.0, 1.0);
                }
            }
        }

        Ok(())
    }

    /// Create safe emotion configuration from raw parameters
    pub fn create_safe_config(
        &self,
        emotion_type: EmotionType,
        intensity: f32,
        secondary_emotions: Vec<(EmotionType, f32)>,
        custom_params: HashMap<String, f32>,
    ) -> Result<EmotionConfig> {
        let mut config = EmotionConfig {
            emotion_type,
            intensity: EmotionIntensity::from_f32(intensity),
            secondary_emotions,
            custom_params,
        };

        self.preprocess(&mut config)?;
        Ok(config)
    }
}

impl Default for EmotionValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Emotion parameter preprocessing utilities
pub struct EmotionPreprocessor {
    validator: EmotionValidator,
}

impl EmotionPreprocessor {
    /// Create new emotion preprocessor
    pub fn new() -> Self {
        Self {
            validator: EmotionValidator::new(),
        }
    }

    /// Create preprocessor with custom validator
    pub fn with_validator(validator: EmotionValidator) -> Self {
        Self { validator }
    }

    /// Preprocess emotion configuration for neural models
    pub fn preprocess_for_neural(&self, config: &EmotionConfig) -> Result<EmotionConfig> {
        let mut processed = config.clone();

        // Validate and preprocess
        self.validator.preprocess(&mut processed)?;

        // Add neural-specific preprocessing
        self.add_neural_preprocessing(&mut processed)?;

        Ok(processed)
    }

    /// Add neural-specific preprocessing
    fn add_neural_preprocessing(&self, config: &mut EmotionConfig) -> Result<()> {
        // Add energy parameter based on emotion type and intensity
        let energy_value =
            self.calculate_energy_for_emotion(&config.emotion_type, config.intensity.as_f32());
        config
            .custom_params
            .insert("energy".to_string(), energy_value);

        // Add arousal and valence parameters
        let (arousal, valence) = self.calculate_arousal_valence(&config.emotion_type);
        config.custom_params.insert("arousal".to_string(), arousal);
        config.custom_params.insert("valence".to_string(), valence);

        // Add pitch shift based on emotion
        let pitch_shift =
            self.calculate_pitch_shift(&config.emotion_type, config.intensity.as_f32());
        config
            .custom_params
            .insert("pitch_shift".to_string(), pitch_shift);

        // Add speed factor
        let speed_factor =
            self.calculate_speed_factor(&config.emotion_type, config.intensity.as_f32());
        config
            .custom_params
            .insert("speed_factor".to_string(), speed_factor);

        Ok(())
    }

    /// Calculate energy parameter for emotion
    fn calculate_energy_for_emotion(&self, emotion_type: &EmotionType, intensity: f32) -> f32 {
        let base_energy = match emotion_type {
            EmotionType::Neutral => 0.5,
            EmotionType::Happy => 0.8,
            EmotionType::Sad => 0.3,
            EmotionType::Angry => 0.9,
            EmotionType::Fear => 0.7,
            EmotionType::Surprise => 0.8,
            EmotionType::Disgust => 0.6,
            EmotionType::Excited => 0.9,
            EmotionType::Calm => 0.4,
            EmotionType::Love => 0.6,
            EmotionType::Custom(_) => 0.5,
        };

        (base_energy * intensity).clamp(0.0, 1.0)
    }

    /// Calculate arousal and valence for emotion
    fn calculate_arousal_valence(&self, emotion_type: &EmotionType) -> (f32, f32) {
        match emotion_type {
            EmotionType::Neutral => (0.5, 0.5),
            EmotionType::Happy => (0.8, 0.8),
            EmotionType::Sad => (0.3, 0.2),
            EmotionType::Angry => (0.9, 0.2),
            EmotionType::Fear => (0.8, 0.3),
            EmotionType::Surprise => (0.7, 0.6),
            EmotionType::Disgust => (0.6, 0.3),
            EmotionType::Excited => (0.9, 0.7),
            EmotionType::Calm => (0.2, 0.6),
            EmotionType::Love => (0.6, 0.8),
            EmotionType::Custom(_) => (0.5, 0.5),
        }
    }

    /// Calculate pitch shift for emotion
    fn calculate_pitch_shift(&self, emotion_type: &EmotionType, intensity: f32) -> f32 {
        let base_shift = match emotion_type {
            EmotionType::Neutral => 0.0,
            EmotionType::Happy => 0.3,
            EmotionType::Sad => -0.2,
            EmotionType::Angry => 0.2,
            EmotionType::Fear => 0.4,
            EmotionType::Surprise => 0.5,
            EmotionType::Disgust => -0.1,
            EmotionType::Excited => 0.4,
            EmotionType::Calm => -0.1,
            EmotionType::Love => 0.1,
            EmotionType::Custom(_) => 0.0,
        };

        (base_shift * intensity).clamp(-2.0, 2.0)
    }

    /// Calculate speed factor for emotion
    fn calculate_speed_factor(&self, emotion_type: &EmotionType, intensity: f32) -> f32 {
        let base_factor = match emotion_type {
            EmotionType::Neutral => 1.0,
            EmotionType::Happy => 1.1,
            EmotionType::Sad => 0.9,
            EmotionType::Angry => 1.2,
            EmotionType::Fear => 1.3,
            EmotionType::Surprise => 1.2,
            EmotionType::Disgust => 0.95,
            EmotionType::Excited => 1.25,
            EmotionType::Calm => 0.85,
            EmotionType::Love => 0.95,
            EmotionType::Custom(_) => 1.0,
        };

        let factor = 1.0 + (base_factor - 1.0) * intensity;
        factor.clamp(0.5, 2.0)
    }
}

impl Default for EmotionPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced emotion interpolation for smooth transitions
pub struct EmotionInterpolator {
    /// Interpolation algorithm to use
    algorithm: InterpolationAlgorithm,
    /// Transition duration in seconds
    transition_duration: f32,
    /// Number of intermediate steps for smooth transitions
    interpolation_steps: usize,
    /// Whether to apply easing functions
    use_easing: bool,
}

/// Interpolation algorithms for emotion transitions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationAlgorithm {
    /// Linear interpolation
    Linear,
    /// Cubic interpolation for smoother transitions
    Cubic,
    /// Smoothstep interpolation (sigmoid-like)
    Smoothstep,
    /// Smootherstep interpolation (even smoother)
    Smootherstep,
    /// Exponential interpolation for dramatic transitions
    Exponential,
    /// Sine-based interpolation for natural transitions
    Sine,
}

impl EmotionInterpolator {
    /// Create new emotion interpolator
    pub fn new() -> Self {
        Self {
            algorithm: InterpolationAlgorithm::Smoothstep,
            transition_duration: 0.5,
            interpolation_steps: 10,
            use_easing: true,
        }
    }

    /// Set interpolation algorithm
    pub fn with_algorithm(mut self, algorithm: InterpolationAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set transition duration
    pub fn with_duration(mut self, duration: f32) -> Self {
        self.transition_duration = duration.max(0.0);
        self
    }

    /// Set interpolation steps
    pub fn with_steps(mut self, steps: usize) -> Self {
        self.interpolation_steps = steps.max(1);
        self
    }

    /// Enable or disable easing functions
    pub fn with_easing(mut self, use_easing: bool) -> Self {
        self.use_easing = use_easing;
        self
    }

    /// Generate smooth transition between two emotions
    pub fn interpolate_emotions(
        &self,
        from: &EmotionConfig,
        to: &EmotionConfig,
    ) -> Result<Vec<EmotionConfig>> {
        let mut result = Vec::with_capacity(self.interpolation_steps);

        for i in 0..self.interpolation_steps {
            let t = i as f32 / (self.interpolation_steps - 1) as f32;
            let alpha = self.apply_interpolation_algorithm(t);

            let interpolated = self.interpolate_single_step(from, to, alpha)?;
            result.push(interpolated);
        }

        Ok(result)
    }

    /// Interpolate between emotions with custom alpha value
    pub fn interpolate_with_alpha(
        &self,
        from: &EmotionConfig,
        to: &EmotionConfig,
        alpha: f32,
    ) -> Result<EmotionConfig> {
        let processed_alpha = self.apply_interpolation_algorithm(alpha.clamp(0.0, 1.0));
        self.interpolate_single_step(from, to, processed_alpha)
    }

    /// Apply interpolation algorithm to alpha value
    fn apply_interpolation_algorithm(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);

        match self.algorithm {
            InterpolationAlgorithm::Linear => t,
            InterpolationAlgorithm::Cubic => {
                // Cubic ease-in-out: 3t² - 2t³
                let t2 = t * t;
                let t3 = t2 * t;
                3.0 * t2 - 2.0 * t3
            }
            InterpolationAlgorithm::Smoothstep => {
                // Smoothstep: t² * (3 - 2t)
                let t2 = t * t;
                t2 * (3.0 - 2.0 * t)
            }
            InterpolationAlgorithm::Smootherstep => {
                // Smootherstep: t³ * (t * (6t - 15) + 10)
                let t3 = t * t * t;
                t3 * (t * (6.0 * t - 15.0) + 10.0)
            }
            InterpolationAlgorithm::Exponential => {
                if t == 0.0 {
                    0.0
                } else if t == 1.0 {
                    1.0
                } else {
                    (2.0_f32.powf(10.0 * (t - 1.0))).clamp(0.0, 1.0)
                }
            }
            InterpolationAlgorithm::Sine => {
                // Sine ease-in-out: (1 - cos(πt)) / 2
                let pi = std::f32::consts::PI;
                (1.0 - (pi * t).cos()) / 2.0
            }
        }
    }

    /// Perform single interpolation step
    fn interpolate_single_step(
        &self,
        from: &EmotionConfig,
        to: &EmotionConfig,
        alpha: f32,
    ) -> Result<EmotionConfig> {
        // Interpolate primary emotion type
        let primary_emotion = if alpha < 0.5 {
            from.emotion_type.clone()
        } else {
            to.emotion_type.clone()
        };

        // Interpolate intensity
        let from_intensity = from.intensity.as_f32();
        let to_intensity = to.intensity.as_f32();
        let interpolated_intensity = from_intensity * (1.0 - alpha) + to_intensity * alpha;

        // Interpolate secondary emotions
        let mut interpolated_secondary = Vec::new();

        // Create a combined set of all secondary emotions
        let mut emotion_weights = HashMap::new();

        // Add from emotions with weight (1 - alpha)
        for (emotion, weight) in &from.secondary_emotions {
            emotion_weights.insert(emotion.clone(), weight * (1.0 - alpha));
        }

        // Add to emotions with weight alpha
        for (emotion, weight) in &to.secondary_emotions {
            let current_weight = emotion_weights.get(emotion).unwrap_or(&0.0);
            emotion_weights.insert(emotion.clone(), current_weight + weight * alpha);
        }

        // Convert back to vector, filtering out very small weights
        for (emotion, weight) in emotion_weights {
            if weight > 0.01 {
                // Filter out very small weights
                interpolated_secondary.push((emotion, weight));
            }
        }

        // Sort by weight (descending) and limit to reasonable number
        interpolated_secondary
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        interpolated_secondary.truncate(5);

        // Interpolate custom parameters
        let mut interpolated_params = HashMap::new();

        // Get all parameter names from both configs
        let mut all_param_names = std::collections::HashSet::new();
        for name in from.custom_params.keys() {
            all_param_names.insert(name.clone());
        }
        for name in to.custom_params.keys() {
            all_param_names.insert(name.clone());
        }

        // Interpolate each parameter
        for param_name in all_param_names {
            let from_value = from.custom_params.get(&param_name).unwrap_or(&0.0);
            let to_value = to.custom_params.get(&param_name).unwrap_or(&0.0);
            let interpolated_value = from_value * (1.0 - alpha) + to_value * alpha;
            interpolated_params.insert(param_name, interpolated_value);
        }

        Ok(EmotionConfig {
            emotion_type: primary_emotion,
            intensity: EmotionIntensity::from_f32(interpolated_intensity),
            secondary_emotions: interpolated_secondary,
            custom_params: interpolated_params,
        })
    }

    /// Create temporal emotion transition sequence
    pub fn create_temporal_transition(
        &self,
        from: &EmotionConfig,
        to: &EmotionConfig,
        sample_rate: f32,
    ) -> Result<Vec<(f32, EmotionConfig)>> {
        let total_samples = (self.transition_duration * sample_rate) as usize;
        let mut result = Vec::with_capacity(total_samples);

        for i in 0..total_samples {
            let t = i as f32 / (total_samples - 1) as f32;
            let alpha = self.apply_interpolation_algorithm(t);
            let timestamp = i as f32 / sample_rate;

            let interpolated = self.interpolate_single_step(from, to, alpha)?;
            result.push((timestamp, interpolated));
        }

        Ok(result)
    }

    /// Create emotion transition with custom timing curve
    pub fn create_custom_transition(
        &self,
        from: &EmotionConfig,
        to: &EmotionConfig,
        timing_curve: &[f32],
    ) -> Result<Vec<EmotionConfig>> {
        let mut result = Vec::with_capacity(timing_curve.len());

        for &t in timing_curve {
            let alpha = self.apply_interpolation_algorithm(t.clamp(0.0, 1.0));
            let interpolated = self.interpolate_single_step(from, to, alpha)?;
            result.push(interpolated);
        }

        Ok(result)
    }

    /// Interpolate emotion vectors for neural models
    pub fn interpolate_emotion_vectors(
        &self,
        from: &EmotionVector,
        to: &EmotionVector,
    ) -> Result<Vec<EmotionVector>> {
        if from.dimension != to.dimension {
            return Err(AcousticError::InputError(
                "Emotion vectors must have the same dimension".to_string(),
            ));
        }

        let mut result = Vec::with_capacity(self.interpolation_steps);

        for i in 0..self.interpolation_steps {
            let t = i as f32 / (self.interpolation_steps - 1) as f32;
            let alpha = self.apply_interpolation_algorithm(t);

            let interpolated_vector = from.interpolate(to, alpha)?;
            result.push(interpolated_vector);
        }

        Ok(result)
    }

    /// Create emotion crossfade for overlapping segments
    pub fn create_emotion_crossfade(
        &self,
        emotions: &[EmotionConfig],
        overlap_ratio: f32,
    ) -> Result<Vec<EmotionConfig>> {
        if emotions.len() < 2 {
            return Ok(emotions.to_vec());
        }

        let overlap_ratio = overlap_ratio.clamp(0.0, 0.5);
        let mut result = Vec::new();

        for i in 0..emotions.len() - 1 {
            let current = &emotions[i];
            let next = &emotions[i + 1];

            // Add current emotion
            result.push(current.clone());

            // Add crossfade transition
            let crossfade_steps = (self.interpolation_steps as f32 * overlap_ratio) as usize;
            for j in 1..=crossfade_steps {
                let t = j as f32 / crossfade_steps as f32;
                let alpha = self.apply_interpolation_algorithm(t);
                let interpolated = self.interpolate_single_step(current, next, alpha)?;
                result.push(interpolated);
            }
        }

        // Add final emotion
        result.push(emotions[emotions.len() - 1].clone());

        Ok(result)
    }

    /// Get interpolation algorithm name
    pub fn get_algorithm_name(&self) -> &'static str {
        match self.algorithm {
            InterpolationAlgorithm::Linear => "linear",
            InterpolationAlgorithm::Cubic => "cubic",
            InterpolationAlgorithm::Smoothstep => "smoothstep",
            InterpolationAlgorithm::Smootherstep => "smootherstep",
            InterpolationAlgorithm::Exponential => "exponential",
            InterpolationAlgorithm::Sine => "sine",
        }
    }

    /// Get transition duration
    pub fn get_transition_duration(&self) -> f32 {
        self.transition_duration
    }

    /// Get interpolation steps
    pub fn get_interpolation_steps(&self) -> usize {
        self.interpolation_steps
    }
}

impl Default for EmotionInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

/// Emotion transition manager for complex emotion sequences
pub struct EmotionTransitionManager {
    /// Interpolator for transitions
    interpolator: EmotionInterpolator,
    /// Current emotion sequence
    emotion_sequence: Vec<EmotionConfig>,
    /// Transition timings
    transition_timings: Vec<f32>,
    /// Current position in sequence
    current_position: usize,
}

impl EmotionTransitionManager {
    /// Create new emotion transition manager
    pub fn new(interpolator: EmotionInterpolator) -> Self {
        Self {
            interpolator,
            emotion_sequence: Vec::new(),
            transition_timings: Vec::new(),
            current_position: 0,
        }
    }

    /// Add emotion to sequence
    pub fn add_emotion(&mut self, emotion: EmotionConfig, timing: f32) {
        self.emotion_sequence.push(emotion);
        self.transition_timings.push(timing);
    }

    /// Get emotion at specific time
    pub fn get_emotion_at_time(&self, time: f32) -> Result<EmotionConfig> {
        if self.emotion_sequence.is_empty() {
            return Err(AcousticError::InputError(
                "No emotions in sequence".to_string(),
            ));
        }

        if self.emotion_sequence.len() == 1 {
            return Ok(self.emotion_sequence[0].clone());
        }

        // Find the appropriate segment
        let mut current_time = 0.0;
        for i in 0..self.transition_timings.len() - 1 {
            let segment_duration = self.transition_timings[i + 1] - self.transition_timings[i];

            if time >= current_time && time < current_time + segment_duration {
                let t = (time - current_time) / segment_duration;
                return self.interpolator.interpolate_with_alpha(
                    &self.emotion_sequence[i],
                    &self.emotion_sequence[i + 1],
                    t,
                );
            }

            current_time += segment_duration;
        }

        // Return last emotion if time is beyond sequence
        Ok(self.emotion_sequence.last().unwrap().clone())
    }

    /// Generate complete emotion sequence
    pub fn generate_sequence(&self) -> Result<Vec<(f32, EmotionConfig)>> {
        let mut result = Vec::new();

        for i in 0..self.emotion_sequence.len() - 1 {
            let from = &self.emotion_sequence[i];
            let to = &self.emotion_sequence[i + 1];
            let duration = self.transition_timings[i + 1] - self.transition_timings[i];

            let transition =
                self.interpolator
                    .create_temporal_transition(from, to, 1.0 / duration)?;

            for (timestamp, emotion) in transition {
                result.push((self.transition_timings[i] + timestamp, emotion));
            }
        }

        Ok(result)
    }

    /// Clear emotion sequence
    pub fn clear(&mut self) {
        self.emotion_sequence.clear();
        self.transition_timings.clear();
        self.current_position = 0;
    }

    /// Get current sequence length
    pub fn len(&self) -> usize {
        self.emotion_sequence.len()
    }

    /// Check if sequence is empty
    pub fn is_empty(&self) -> bool {
        self.emotion_sequence.is_empty()
    }
}

impl Default for EmotionTransitionManager {
    fn default() -> Self {
        Self::new(EmotionInterpolator::default())
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
        assert_eq!(
            EmotionType::from_str("custom"),
            EmotionType::Custom("custom".to_string())
        );
    }

    #[test]
    fn test_emotion_intensity() {
        assert_eq!(EmotionIntensity::Medium.as_f32(), 0.5);
        assert_eq!(EmotionIntensity::from_f32(0.3), EmotionIntensity::Low);
        assert_eq!(EmotionIntensity::from_f32(1.5), EmotionIntensity::VeryHigh);
        // Clamped
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
        let config1 =
            EmotionConfig::new(EmotionType::Happy).with_intensity(EmotionIntensity::from_f32(0.8)); // High (0.7)
        let config2 =
            EmotionConfig::new(EmotionType::Sad).with_intensity(EmotionIntensity::from_f32(0.6)); // Medium (0.5)

        let blended = config1.blend(&config2, 0.5);
        // Blend of High (0.7) and Medium (0.5) with alpha 0.5 = 0.6, which maps to Medium (0.5)
        assert_eq!(blended.intensity.as_f32(), 0.5);
    }

    #[test]
    fn test_emotion_vector_creation() {
        let config = EmotionConfig::new(EmotionType::Happy).with_intensity(EmotionIntensity::High);

        let vector = EmotionVector::new(config, 64);
        assert_eq!(vector.dimension, 64);
        assert_eq!(vector.vector.len(), 64);

        // Happy should be at index 1 in all_basic()
        assert!(vector.vector[1] > 0.0);
    }

    #[test]
    fn test_emotion_vector_interpolation() {
        let config1 =
            EmotionConfig::new(EmotionType::Happy).with_intensity(EmotionIntensity::from_f32(1.0));
        let config2 =
            EmotionConfig::new(EmotionType::Sad).with_intensity(EmotionIntensity::from_f32(1.0));

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
        let happy_config =
            EmotionConfig::new(EmotionType::Happy).with_intensity(EmotionIntensity::High);
        model.set_emotion(happy_config);

        assert_eq!(model.get_current_emotion().emotion_type, EmotionType::Happy);
        assert_eq!(model.get_emotion_history().len(), 1);

        // Transition
        let sad_config =
            EmotionConfig::new(EmotionType::Sad).with_intensity(EmotionIntensity::Medium);
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
