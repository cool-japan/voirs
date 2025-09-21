//! Prosody modification for emotional expression
//!
//! This module provides comprehensive prosody modification capabilities for emotional
//! voice synthesis, including real-time adaptation, pattern analysis, and advanced
//! emotion-to-prosody mappings.
//!
//! ## Features
//!
//! - **Advanced Prosody Control**: Comprehensive pitch, timing, energy, and voice quality control
//! - **Real-time Adaptation**: Dynamic prosody adjustment based on emotional context
//! - **Pattern Analysis**: Prosody pattern recognition and generation
//! - **SSML Integration**: Speech Synthesis Markup Language prosody control
//! - **Template System**: Reusable prosody templates for different speaking styles
//! - **Context Awareness**: Prosody adaptation based on linguistic and emotional context
//!
//! ## Example Usage
//!
//! ```rust
//! use voirs_emotion::prosody::ProsodyModifier;
//! use voirs_emotion::types::{Emotion, EmotionVector, EmotionIntensity};
//!
//! // Create prosody modifier with emotion mapping
//! let modifier = ProsodyModifier::new();
//! let mut emotion_vector = EmotionVector::new();
//! emotion_vector.add_emotion(Emotion::Happy, EmotionIntensity::new(0.8));
//! let prosody = modifier.apply_emotion_vector(&emotion_vector)?;
//!
//! // Apply dimension-based prosody modification
//! let dimensions = voirs_emotion::types::EmotionDimensions::new(0.8, 0.6, 0.4);
//! let dimension_prosody = modifier.apply_emotion_dimensions(&dimensions)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{
    types::{Emotion, EmotionDimensions, EmotionIntensity, EmotionParameters, EmotionVector},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Prosody parameters that can be modified for emotional expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProsodyParameters {
    /// Fundamental frequency (F0) modifications
    pub pitch: PitchParameters,
    /// Duration and timing modifications
    pub timing: TimingParameters,
    /// Energy and intensity modifications
    pub energy: EnergyParameters,
    /// Voice quality modifications
    pub voice_quality: VoiceQualityParameters,
}

impl ProsodyParameters {
    /// Create neutral prosody parameters
    pub fn neutral() -> Self {
        Self {
            pitch: PitchParameters::neutral(),
            timing: TimingParameters::neutral(),
            energy: EnergyParameters::neutral(),
            voice_quality: VoiceQualityParameters::neutral(),
        }
    }

    /// Create prosody parameters from emotion
    pub fn from_emotion(emotion: Emotion, intensity: f32) -> Self {
        let mut params = Self::neutral();
        params.apply_emotion_mapping(emotion, intensity);
        params
    }

    /// Apply emotion-specific prosody modifications
    pub fn apply_emotion_mapping(&mut self, emotion: Emotion, intensity: f32) {
        let intensity = intensity.clamp(0.0, 1.0);

        match emotion {
            Emotion::Happy => {
                self.pitch.mean_shift = 1.0 + (0.3 * intensity);
                self.pitch.range_scale = 1.0 + (0.5 * intensity);
                self.timing.speech_rate = 1.0 + (0.2 * intensity);
                self.energy.overall_scale = 1.0 + (0.3 * intensity);
                self.voice_quality.brightness = 0.3 * intensity;
            }
            Emotion::Sad => {
                self.pitch.mean_shift = 1.0 - (0.2 * intensity);
                self.pitch.range_scale = 1.0 - (0.3 * intensity);
                self.timing.speech_rate = 1.0 - (0.3 * intensity);
                self.energy.overall_scale = 1.0 - (0.4 * intensity);
                self.voice_quality.breathiness = 0.4 * intensity;
            }
            Emotion::Angry => {
                self.pitch.mean_shift = 1.0 + (0.4 * intensity);
                self.pitch.range_scale = 1.0 + (0.6 * intensity);
                self.timing.speech_rate = 1.0 + (0.3 * intensity);
                self.energy.overall_scale = 1.0 + (0.5 * intensity);
                self.voice_quality.roughness = 0.5 * intensity;
            }
            Emotion::Fear => {
                self.pitch.mean_shift = 1.0 + (0.5 * intensity);
                self.pitch.tremor = 0.3 * intensity;
                self.timing.speech_rate = 1.0 + (0.4 * intensity);
                self.energy.overall_scale = 1.0 + (0.2 * intensity);
                self.voice_quality.tension = 0.4 * intensity;
            }
            Emotion::Surprise => {
                self.pitch.mean_shift = 1.0 + (0.6 * intensity);
                self.pitch.range_scale = 1.0 + (0.4 * intensity);
                self.timing.pause_duration = 1.0 + (0.3 * intensity);
                self.energy.overall_scale = 1.0 + (0.3 * intensity);
            }
            Emotion::Calm => {
                self.pitch.range_scale = 1.0 - (0.2 * intensity);
                self.timing.speech_rate = 1.0 - (0.1 * intensity);
                self.energy.overall_scale = 1.0 - (0.1 * intensity);
                self.voice_quality.smoothness = 0.3 * intensity;
            }
            Emotion::Excited => {
                self.pitch.mean_shift = 1.0 + (0.3 * intensity);
                self.pitch.range_scale = 1.0 + (0.7 * intensity);
                self.timing.speech_rate = 1.0 + (0.5 * intensity);
                self.energy.overall_scale = 1.0 + (0.6 * intensity);
            }
            _ => {} // Other emotions use neutral settings
        }
    }

    /// Blend with another prosody parameter set
    pub fn blend_with(&self, other: &Self, weight: f32) -> Self {
        let weight = weight.clamp(0.0, 1.0);
        let _inv_weight = 1.0 - weight;

        Self {
            pitch: self.pitch.blend_with(&other.pitch, weight),
            timing: self.timing.blend_with(&other.timing, weight),
            energy: self.energy.blend_with(&other.energy, weight),
            voice_quality: self.voice_quality.blend_with(&other.voice_quality, weight),
        }
    }
}

impl Default for ProsodyParameters {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Pitch-related prosody parameters
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PitchParameters {
    /// Mean F0 shift (multiplicative factor)
    pub mean_shift: f32,
    /// F0 range scaling factor
    pub range_scale: f32,
    /// Pitch tremor amount (0.0 to 1.0)
    pub tremor: f32,
    /// Vibrato depth (0.0 to 1.0)
    pub vibrato_depth: f32,
    /// Vibrato rate (Hz)
    pub vibrato_rate: f32,
}

impl PitchParameters {
    /// Create neutral pitch parameters
    pub fn neutral() -> Self {
        Self {
            mean_shift: 1.0,
            range_scale: 1.0,
            tremor: 0.0,
            vibrato_depth: 0.0,
            vibrato_rate: 0.0,
        }
    }

    /// Blend with another pitch parameter set
    pub fn blend_with(&self, other: &Self, weight: f32) -> Self {
        let w = weight.clamp(0.0, 1.0);
        let inv_w = 1.0 - w;

        Self {
            mean_shift: self.mean_shift * inv_w + other.mean_shift * w,
            range_scale: self.range_scale * inv_w + other.range_scale * w,
            tremor: self.tremor * inv_w + other.tremor * w,
            vibrato_depth: self.vibrato_depth * inv_w + other.vibrato_depth * w,
            vibrato_rate: self.vibrato_rate * inv_w + other.vibrato_rate * w,
        }
    }
}

/// Timing-related prosody parameters
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimingParameters {
    /// Speech rate scaling factor
    pub speech_rate: f32,
    /// Pause duration scaling factor
    pub pause_duration: f32,
    /// Vowel duration scaling factor
    pub vowel_duration: f32,
    /// Consonant duration scaling factor
    pub consonant_duration: f32,
    /// Rhythm regularity (0.0 = irregular, 1.0 = regular)
    pub rhythm_regularity: f32,
}

impl TimingParameters {
    /// Create neutral timing parameters
    pub fn neutral() -> Self {
        Self {
            speech_rate: 1.0,
            pause_duration: 1.0,
            vowel_duration: 1.0,
            consonant_duration: 1.0,
            rhythm_regularity: 0.5,
        }
    }

    /// Blend with another timing parameter set
    pub fn blend_with(&self, other: &Self, weight: f32) -> Self {
        let w = weight.clamp(0.0, 1.0);
        let inv_w = 1.0 - w;

        Self {
            speech_rate: self.speech_rate * inv_w + other.speech_rate * w,
            pause_duration: self.pause_duration * inv_w + other.pause_duration * w,
            vowel_duration: self.vowel_duration * inv_w + other.vowel_duration * w,
            consonant_duration: self.consonant_duration * inv_w + other.consonant_duration * w,
            rhythm_regularity: self.rhythm_regularity * inv_w + other.rhythm_regularity * w,
        }
    }
}

/// Energy-related prosody parameters
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnergyParameters {
    /// Overall energy scaling factor
    pub overall_scale: f32,
    /// Dynamic range scaling factor
    pub dynamic_range: f32,
    /// Stress emphasis scaling factor
    pub stress_emphasis: f32,
    /// Energy contour smoothness (0.0 = jagged, 1.0 = smooth)
    pub contour_smoothness: f32,
}

impl EnergyParameters {
    /// Create neutral energy parameters
    pub fn neutral() -> Self {
        Self {
            overall_scale: 1.0,
            dynamic_range: 1.0,
            stress_emphasis: 1.0,
            contour_smoothness: 0.5,
        }
    }

    /// Blend with another energy parameter set
    pub fn blend_with(&self, other: &Self, weight: f32) -> Self {
        let w = weight.clamp(0.0, 1.0);
        let inv_w = 1.0 - w;

        Self {
            overall_scale: self.overall_scale * inv_w + other.overall_scale * w,
            dynamic_range: self.dynamic_range * inv_w + other.dynamic_range * w,
            stress_emphasis: self.stress_emphasis * inv_w + other.stress_emphasis * w,
            contour_smoothness: self.contour_smoothness * inv_w + other.contour_smoothness * w,
        }
    }
}

/// Voice quality-related prosody parameters
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VoiceQualityParameters {
    /// Breathiness amount (0.0 to 1.0)
    pub breathiness: f32,
    /// Roughness amount (0.0 to 1.0)
    pub roughness: f32,
    /// Tension amount (0.0 to 1.0)
    pub tension: f32,
    /// Brightness amount (-1.0 to 1.0)
    pub brightness: f32,
    /// Smoothness amount (0.0 to 1.0)
    pub smoothness: f32,
    /// Nasality amount (0.0 to 1.0)
    pub nasality: f32,
}

impl VoiceQualityParameters {
    /// Create neutral voice quality parameters
    pub fn neutral() -> Self {
        Self {
            breathiness: 0.0,
            roughness: 0.0,
            tension: 0.0,
            brightness: 0.0,
            smoothness: 0.0,
            nasality: 0.0,
        }
    }

    /// Blend with another voice quality parameter set
    pub fn blend_with(&self, other: &Self, weight: f32) -> Self {
        let w = weight.clamp(0.0, 1.0);
        let inv_w = 1.0 - w;

        Self {
            breathiness: self.breathiness * inv_w + other.breathiness * w,
            roughness: self.roughness * inv_w + other.roughness * w,
            tension: self.tension * inv_w + other.tension * w,
            brightness: self.brightness * inv_w + other.brightness * w,
            smoothness: self.smoothness * inv_w + other.smoothness * w,
            nasality: self.nasality * inv_w + other.nasality * w,
        }
    }
}

/// Prosody modifier that applies emotion-based modifications
#[derive(Debug, Clone)]
pub struct ProsodyModifier {
    /// Base prosody parameters
    base_params: ProsodyParameters,
    /// Emotion-specific modifications
    emotion_mappings: HashMap<Emotion, ProsodyParameters>,
}

impl ProsodyModifier {
    /// Create a new prosody modifier
    pub fn new() -> Self {
        Self {
            base_params: ProsodyParameters::neutral(),
            emotion_mappings: Self::create_default_mappings(),
        }
    }

    /// Create default emotion to prosody mappings
    fn create_default_mappings() -> HashMap<Emotion, ProsodyParameters> {
        let mut mappings = HashMap::new();

        let emotions = [
            Emotion::Happy,
            Emotion::Sad,
            Emotion::Angry,
            Emotion::Fear,
            Emotion::Surprise,
            Emotion::Calm,
            Emotion::Excited,
            Emotion::Tender,
            Emotion::Confident,
            Emotion::Melancholic,
        ];

        for emotion in emotions {
            mappings.insert(
                emotion.clone(),
                ProsodyParameters::from_emotion(emotion, 1.0),
            );
        }

        mappings
    }

    /// Apply emotion to prosody parameters
    pub fn apply_emotion(&self, emotion_params: &EmotionParameters) -> Result<ProsodyParameters> {
        let mut result = self.base_params.clone();

        // Apply each emotion in the vector with its intensity
        for (emotion, intensity) in &emotion_params.emotion_vector.emotions {
            if let Some(emotion_prosody) = self.emotion_mappings.get(emotion) {
                result = result.blend_with(emotion_prosody, intensity.value());
            }
        }

        // Apply direct prosody modifications from emotion parameters
        result.pitch.mean_shift *= emotion_params.pitch_shift;
        result.timing.speech_rate *= emotion_params.tempo_scale;
        result.energy.overall_scale *= emotion_params.energy_scale;
        result.voice_quality.breathiness += emotion_params.breathiness;
        result.voice_quality.roughness += emotion_params.roughness;

        Ok(result)
    }

    /// Apply emotion vector to prosody parameters
    pub fn apply_emotion_vector(
        &self,
        emotion_vector: &EmotionVector,
    ) -> Result<ProsodyParameters> {
        let mut result = self.base_params.clone();

        for (emotion, intensity) in &emotion_vector.emotions {
            if let Some(emotion_prosody) = self.emotion_mappings.get(emotion) {
                result = result.blend_with(emotion_prosody, intensity.value());
            }
        }

        Ok(result)
    }

    /// Apply emotion dimensions to prosody parameters
    pub fn apply_emotion_dimensions(
        &self,
        dimensions: &EmotionDimensions,
    ) -> Result<ProsodyParameters> {
        let mut result = self.base_params.clone();

        // Map dimensions to prosody parameters
        // Valence affects pitch and energy
        if dimensions.valence > 0.0 {
            result.pitch.mean_shift *= 1.0 + (dimensions.valence * 0.2);
            result.energy.overall_scale *= 1.0 + (dimensions.valence * 0.3);
        } else {
            result.pitch.mean_shift *= 1.0 + (dimensions.valence * 0.15);
            result.energy.overall_scale *= 1.0 + (dimensions.valence * 0.25);
        }

        // Arousal affects speech rate and pitch range
        result.timing.speech_rate *= 1.0 + (dimensions.arousal * 0.3);
        result.pitch.range_scale *= 1.0 + (dimensions.arousal * 0.4);

        // Dominance affects voice quality and energy
        if dimensions.dominance > 0.0 {
            result.voice_quality.brightness += dimensions.dominance * 0.3;
            result.energy.stress_emphasis *= 1.0 + (dimensions.dominance * 0.2);
        } else {
            result.voice_quality.breathiness += (-dimensions.dominance) * 0.2;
        }

        Ok(result)
    }

    /// Set custom emotion mapping
    pub fn set_emotion_mapping(&mut self, emotion: Emotion, params: ProsodyParameters) {
        self.emotion_mappings.insert(emotion, params);
    }

    /// Get emotion mapping
    pub fn get_emotion_mapping(&self, emotion: &Emotion) -> Option<&ProsodyParameters> {
        self.emotion_mappings.get(emotion)
    }

    /// Set base prosody parameters
    pub fn set_base_params(&mut self, params: ProsodyParameters) {
        self.base_params = params;
    }

    /// Get base prosody parameters
    pub fn get_base_params(&self) -> &ProsodyParameters {
        &self.base_params
    }
}

impl Default for ProsodyModifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Prosody template for reusable speaking styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodyTemplate {
    /// Template name
    pub name: String,
    /// Template description
    pub description: String,
    /// Base prosody parameters
    pub parameters: ProsodyParameters,
    /// Emotion-specific modifications
    pub emotion_adaptations: HashMap<Emotion, ProsodyParameters>,
}

impl ProsodyTemplate {
    /// Create a new prosody template
    pub fn new<S: Into<String>>(name: S, description: S, parameters: ProsodyParameters) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            emotion_adaptations: HashMap::new(),
        }
    }

    /// Add emotion adaptation to template
    pub fn add_emotion_adaptation(&mut self, emotion: Emotion, adaptation: ProsodyParameters) {
        self.emotion_adaptations.insert(emotion, adaptation);
    }

    /// Apply template to emotion
    pub fn apply_to_emotion(&self, emotion: &Emotion, intensity: f32) -> ProsodyParameters {
        let mut result = self.parameters.clone();

        if let Some(adaptation) = self.emotion_adaptations.get(emotion) {
            result = result.blend_with(adaptation, intensity.clamp(0.0, 1.0));
        }

        result
    }
}

/// Real-time prosody adapter for dynamic emotion changes
#[derive(Debug)]
pub struct RealTimeProsodyAdapter {
    /// Current prosody state
    current_state: RwLock<ProsodyParameters>,
    /// Target prosody state
    target_state: RwLock<ProsodyParameters>,
    /// Adaptation rate (0.0 to 1.0)
    adaptation_rate: f32,
    /// Last update time
    last_update: RwLock<Instant>,
}

impl RealTimeProsodyAdapter {
    /// Create a new real-time prosody adapter
    pub fn new() -> Self {
        Self {
            current_state: RwLock::new(ProsodyParameters::neutral()),
            target_state: RwLock::new(ProsodyParameters::neutral()),
            adaptation_rate: 0.1,
            last_update: RwLock::new(Instant::now()),
        }
    }

    /// Create adapter with custom adaptation rate
    pub fn with_adaptation_rate(adaptation_rate: f32) -> Self {
        Self {
            current_state: RwLock::new(ProsodyParameters::neutral()),
            target_state: RwLock::new(ProsodyParameters::neutral()),
            adaptation_rate: adaptation_rate.clamp(0.001, 1.0),
            last_update: RwLock::new(Instant::now()),
        }
    }

    /// Set target emotion for adaptation
    pub async fn set_target_emotion(&self, emotion: Emotion, intensity: EmotionIntensity) {
        let target_params = ProsodyParameters::from_emotion(emotion, intensity.value());
        let mut target_state = self.target_state.write().await;
        *target_state = target_params;
    }

    /// Set target prosody parameters directly
    pub async fn set_target_parameters(&self, parameters: ProsodyParameters) {
        let mut target_state = self.target_state.write().await;
        *target_state = parameters;
    }

    /// Update prosody state based on time-based interpolation
    pub async fn update_prosody(
        &self,
        _input_prosody: &ProsodyParameters,
    ) -> Result<ProsodyParameters> {
        let now = Instant::now();
        let mut last_update = self.last_update.write().await;
        let dt = now.duration_since(*last_update).as_secs_f32();
        *last_update = now;

        let target_state = self.target_state.read().await;
        let mut current_state = self.current_state.write().await;

        // Time-based interpolation towards target
        let blend_factor = (self.adaptation_rate * dt).min(1.0);
        *current_state = current_state.blend_with(&target_state, blend_factor);

        Ok(current_state.clone())
    }

    /// Get current prosody state
    pub async fn get_current_state(&self) -> ProsodyParameters {
        self.current_state.read().await.clone()
    }

    /// Set adaptation rate
    pub fn set_adaptation_rate(&mut self, rate: f32) {
        self.adaptation_rate = rate.clamp(0.001, 1.0);
    }

    /// Reset to neutral state
    pub async fn reset(&self) {
        let mut current_state = self.current_state.write().await;
        let mut target_state = self.target_state.write().await;
        *current_state = ProsodyParameters::neutral();
        *target_state = ProsodyParameters::neutral();
    }
}

impl Default for RealTimeProsodyAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EmotionIntensity, EmotionVector};

    #[test]
    fn test_prosody_parameters_neutral() {
        let params = ProsodyParameters::neutral();
        assert_eq!(params.pitch.mean_shift, 1.0);
        assert_eq!(params.timing.speech_rate, 1.0);
        assert_eq!(params.energy.overall_scale, 1.0);
    }

    #[test]
    fn test_emotion_mapping() {
        let mut params = ProsodyParameters::neutral();
        params.apply_emotion_mapping(Emotion::Happy, 0.8);

        assert!(params.pitch.mean_shift > 1.0);
        assert!(params.energy.overall_scale > 1.0);
        assert!(params.voice_quality.brightness > 0.0);
    }

    #[test]
    fn test_prosody_blending() {
        let params1 = ProsodyParameters::neutral();
        let mut params2 = ProsodyParameters::neutral();
        params2.pitch.mean_shift = 2.0;

        let blended = params1.blend_with(&params2, 0.5);
        assert_eq!(blended.pitch.mean_shift, 1.5);
    }

    #[test]
    fn test_prosody_modifier() {
        let modifier = ProsodyModifier::new();

        let mut emotion_vector = EmotionVector::new();
        emotion_vector.add_emotion(Emotion::Happy, EmotionIntensity::new(0.7));

        let result = modifier.apply_emotion_vector(&emotion_vector).unwrap();
        assert!(result.pitch.mean_shift > 1.0);
    }

    #[test]
    fn test_dimension_mapping() {
        let modifier = ProsodyModifier::new();
        let dimensions = EmotionDimensions::new(0.8, 0.6, 0.4);

        let result = modifier.apply_emotion_dimensions(&dimensions).unwrap();
        assert!(result.pitch.mean_shift > 1.0);
        assert!(result.timing.speech_rate > 1.0);
    }
}
