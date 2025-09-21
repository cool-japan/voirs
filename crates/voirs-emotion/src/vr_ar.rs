//! # VR/AR Emotion Integration
//!
//! This module provides comprehensive VR/AR integration for immersive emotional audio experiences.
//! It includes spatial emotion processing, environment-aware adaptation, and gesture-based emotion control.
//!
//! ## Features
//!
//! - **Spatial Emotion Processing**: 3D spatial audio with emotion mapping
//! - **Environment Awareness**: Emotion adaptation based on VR/AR scene context
//! - **Gesture Integration**: Hand and body gesture recognition for emotion control
//! - **Haptic Feedback**: Tactile emotion feedback for immersive experiences
//! - **Avatar Emotion Sync**: Synchronize voice emotions with avatar expressions
//!
//! ## Example Usage
//!
//! ```rust
//! use voirs_emotion::vr_ar::{VREmotionProcessor, VREnvironmentType, SpatialEmotionConfig, HapticPattern};
//! use voirs_emotion::types::{Emotion, EmotionIntensity};
//!
//! // Create VR configuration
//! let config = SpatialEmotionConfig::new()
//!     .with_3d_positioning(true)
//!     .with_environment_adaptation(true)
//!     .with_haptic_feedback(true);
//!
//! // Create processor and set environment
//! let mut processor = VREmotionProcessor::new(config)?;
//! processor.set_environment(VREnvironmentType::SocialSpace);
//! processor.set_user_position([0.0, 1.7, 0.0]); // Head height
//!
//! // Test environment intensity modifier
//! let social_space = VREnvironmentType::SocialSpace;
//! assert_eq!(social_space.intensity_modifier(), 1.2);
//!
//! // Test haptic pattern creation
//! let haptic = HapticPattern::for_emotion(Emotion::Excited, EmotionIntensity::HIGH);
//! assert_eq!(haptic.pulse_count, 4);
//! assert_eq!(haptic.frequency, 300.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::types::{
    Emotion, EmotionDimensions, EmotionIntensity, EmotionParameters, EmotionVector,
};
use crate::{Error, Result};

/// 3D position in VR/AR space
pub type Position3D = [f32; 3];

/// 3D direction vector (normalized)
pub type Direction3D = [f32; 3];

/// VR/AR environment types that affect emotional processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VREnvironmentType {
    /// Home/personal space with relaxed emotional expression
    PersonalSpace,
    /// Social gathering space with heightened social awareness
    SocialSpace,
    /// Game environment with dynamic emotional responses
    GameEnvironment,
    /// Educational/training environment with focused emotions
    EducationalSpace,
    /// Business/professional meeting space
    ProfessionalSpace,
    /// Entertainment venue with amplified emotions
    EntertainmentVenue,
    /// Therapeutic/meditation space with calming emotions
    TherapeuticSpace,
    /// Outdoor virtual environment
    OutdoorSpace,
}

impl VREnvironmentType {
    /// Get emotional intensity modifier for this environment
    pub fn intensity_modifier(&self) -> f32 {
        match self {
            VREnvironmentType::PersonalSpace => 0.8,
            VREnvironmentType::SocialSpace => 1.2,
            VREnvironmentType::GameEnvironment => 1.5,
            VREnvironmentType::EducationalSpace => 0.9,
            VREnvironmentType::ProfessionalSpace => 0.7,
            VREnvironmentType::EntertainmentVenue => 1.4,
            VREnvironmentType::TherapeuticSpace => 0.6,
            VREnvironmentType::OutdoorSpace => 1.1,
        }
    }

    /// Get emotional adaptation weights [valence, arousal, dominance]
    pub fn adaptation_weights(&self) -> [f32; 3] {
        match self {
            VREnvironmentType::PersonalSpace => [1.0, 0.8, 0.9],
            VREnvironmentType::SocialSpace => [1.1, 1.2, 1.0],
            VREnvironmentType::GameEnvironment => [1.2, 1.5, 1.3],
            VREnvironmentType::EducationalSpace => [0.9, 0.8, 1.1],
            VREnvironmentType::ProfessionalSpace => [0.8, 0.7, 1.2],
            VREnvironmentType::EntertainmentVenue => [1.3, 1.4, 1.1],
            VREnvironmentType::TherapeuticSpace => [1.1, 0.5, 0.7],
            VREnvironmentType::OutdoorSpace => [1.0, 1.1, 1.0],
        }
    }
}

/// Hand gesture types for emotion control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HandGesture {
    /// Open palm (relaxed/calm)
    OpenPalm,
    /// Closed fist (anger/intensity)
    ClosedFist,
    /// Thumbs up (positive emotions)
    ThumbsUp,
    /// Thumbs down (negative emotions)
    ThumbsDown,
    /// Peace sign (peaceful/happy)
    PeaceSign,
    /// Waving (greeting/excitement)
    Wave,
    /// Pointing (focus/emphasis)
    Point,
    /// Heart shape (love/affection)
    HeartShape,
}

impl HandGesture {
    /// Get emotion mapping for this gesture
    pub fn to_emotion_influence(&self) -> (Emotion, f32) {
        match self {
            HandGesture::OpenPalm => (Emotion::Calm, 0.7),
            HandGesture::ClosedFist => (Emotion::Angry, 0.8),
            HandGesture::ThumbsUp => (Emotion::Happy, 0.9),
            HandGesture::ThumbsDown => (Emotion::Sad, 0.8),
            HandGesture::PeaceSign => (Emotion::Happy, 0.6),
            HandGesture::Wave => (Emotion::Excited, 0.7),
            HandGesture::Point => (Emotion::Confident, 0.8),
            HandGesture::HeartShape => (Emotion::Happy, 1.0),
        }
    }
}

/// Haptic feedback patterns for different emotions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HapticPattern {
    /// Vibration intensity (0.0 to 1.0)
    pub intensity: f32,
    /// Vibration frequency in Hz
    pub frequency: f32,
    /// Pattern duration in milliseconds
    pub duration_ms: u32,
    /// Number of pulses
    pub pulse_count: u32,
    /// Delay between pulses in milliseconds
    pub pulse_delay_ms: u32,
}

impl HapticPattern {
    /// Create haptic pattern for specific emotion
    pub fn for_emotion(emotion: Emotion, intensity: EmotionIntensity) -> Self {
        let base_intensity = intensity.value();
        match emotion {
            Emotion::Happy => HapticPattern {
                intensity: base_intensity * 0.6,
                frequency: 200.0,
                duration_ms: 150,
                pulse_count: 3,
                pulse_delay_ms: 50,
            },
            Emotion::Sad => HapticPattern {
                intensity: base_intensity * 0.4,
                frequency: 80.0,
                duration_ms: 300,
                pulse_count: 1,
                pulse_delay_ms: 0,
            },
            Emotion::Angry => HapticPattern {
                intensity: base_intensity * 0.9,
                frequency: 400.0,
                duration_ms: 100,
                pulse_count: 5,
                pulse_delay_ms: 20,
            },
            Emotion::Excited => HapticPattern {
                intensity: base_intensity * 0.8,
                frequency: 300.0,
                duration_ms: 200,
                pulse_count: 4,
                pulse_delay_ms: 30,
            },
            Emotion::Calm => HapticPattern {
                intensity: base_intensity * 0.3,
                frequency: 60.0,
                duration_ms: 500,
                pulse_count: 1,
                pulse_delay_ms: 0,
            },
            _ => HapticPattern {
                intensity: base_intensity * 0.5,
                frequency: 150.0,
                duration_ms: 200,
                pulse_count: 2,
                pulse_delay_ms: 100,
            },
        }
    }
}

/// Configuration for spatial emotion processing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialEmotionConfig {
    /// Enable 3D positioning of emotion sources
    pub enable_3d_positioning: bool,
    /// Enable environment-based emotion adaptation
    pub enable_environment_adaptation: bool,
    /// Enable haptic feedback for emotions
    pub enable_haptic_feedback: bool,
    /// Enable gesture-based emotion control
    pub enable_gesture_control: bool,
    /// Maximum distance for emotion source (meters)
    pub max_source_distance: f32,
    /// Distance falloff factor for emotion intensity
    pub distance_falloff: f32,
    /// Update frequency for spatial processing (Hz)
    pub update_frequency: f32,
    /// Avatar emotion synchronization enabled
    pub enable_avatar_sync: bool,
}

impl Default for SpatialEmotionConfig {
    fn default() -> Self {
        SpatialEmotionConfig {
            enable_3d_positioning: true,
            enable_environment_adaptation: true,
            enable_haptic_feedback: false,
            enable_gesture_control: false,
            max_source_distance: 10.0,
            distance_falloff: 0.8,
            update_frequency: 60.0,
            enable_avatar_sync: false,
        }
    }
}

impl SpatialEmotionConfig {
    /// Create new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable/disable 3D positioning
    pub fn with_3d_positioning(mut self, enabled: bool) -> Self {
        self.enable_3d_positioning = enabled;
        self
    }

    /// Enable/disable environment adaptation
    pub fn with_environment_adaptation(mut self, enabled: bool) -> Self {
        self.enable_environment_adaptation = enabled;
        self
    }

    /// Enable/disable haptic feedback
    pub fn with_haptic_feedback(mut self, enabled: bool) -> Self {
        self.enable_haptic_feedback = enabled;
        self
    }

    /// Enable/disable gesture control
    pub fn with_gesture_control(mut self, enabled: bool) -> Self {
        self.enable_gesture_control = enabled;
        self
    }

    /// Set maximum source distance
    pub fn with_max_distance(mut self, distance: f32) -> Self {
        self.max_source_distance = distance.max(0.1);
        self
    }

    /// Set distance falloff factor
    pub fn with_distance_falloff(mut self, falloff: f32) -> Self {
        self.distance_falloff = falloff.clamp(0.1, 2.0);
        self
    }
}

/// Spatial emotion source in VR/AR space
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialEmotionSource {
    /// Unique identifier for this source
    pub id: String,
    /// Current position in 3D space
    pub position: Position3D,
    /// Direction the source is facing (optional)
    pub direction: Option<Direction3D>,
    /// Current emotion parameters
    pub emotion_params: EmotionParameters,
    /// Last update timestamp
    #[serde(skip)]
    pub last_update: Option<Instant>,
    /// Source priority (higher = more influence)
    pub priority: f32,
}

impl SpatialEmotionSource {
    /// Create new spatial emotion source
    pub fn new(id: String, position: Position3D, emotion_params: EmotionParameters) -> Self {
        SpatialEmotionSource {
            id,
            position,
            direction: None,
            emotion_params,
            last_update: Some(Instant::now()),
            priority: 1.0,
        }
    }

    /// Update source position
    pub fn set_position(&mut self, position: Position3D) {
        self.position = position;
        self.last_update = Some(Instant::now());
    }

    /// Update source direction
    pub fn set_direction(&mut self, direction: Direction3D) {
        self.direction = Some(normalize_vector(direction));
        self.last_update = Some(Instant::now());
    }

    /// Update emotion parameters
    pub fn update_emotion(&mut self, emotion_params: EmotionParameters) {
        self.emotion_params = emotion_params;
        self.last_update = Some(Instant::now());
    }

    /// Calculate distance to user position
    pub fn distance_to(&self, user_position: Position3D) -> f32 {
        distance_3d(self.position, user_position)
    }

    /// Calculate influence based on distance and priority
    pub fn calculate_influence(
        &self,
        user_position: Position3D,
        config: &SpatialEmotionConfig,
    ) -> f32 {
        let distance = self.distance_to(user_position);

        if distance > config.max_source_distance {
            return 0.0;
        }

        let distance_factor =
            1.0 - (distance / config.max_source_distance).powf(config.distance_falloff);
        distance_factor * self.priority
    }
}

/// Avatar emotion synchronization data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AvatarEmotionSync {
    /// Avatar facial expression mapping
    pub facial_expression: HashMap<String, f32>,
    /// Body posture parameters
    pub posture_params: HashMap<String, f32>,
    /// Animation blend weights
    pub animation_weights: HashMap<String, f32>,
    /// Synchronization confidence (0.0 to 1.0)
    pub sync_confidence: f32,
}

impl AvatarEmotionSync {
    /// Create emotion sync from emotion parameters
    pub fn from_emotion_params(params: &EmotionParameters) -> Self {
        let mut facial_expression = HashMap::new();
        let mut posture_params = HashMap::new();
        let mut animation_weights = HashMap::new();

        // Map emotion dimensions to facial expressions
        facial_expression.insert(
            "eyebrow_height".to_string(),
            params.emotion_vector.dimensions.arousal * 0.5,
        );
        facial_expression.insert(
            "mouth_curve".to_string(),
            params.emotion_vector.dimensions.valence * 0.8,
        );
        facial_expression.insert(
            "eye_openness".to_string(),
            0.5 + params.emotion_vector.dimensions.arousal * 0.3,
        );
        facial_expression.insert(
            "jaw_tension".to_string(),
            params.emotion_vector.dimensions.dominance.abs() * 0.4,
        );

        // Map to body posture
        posture_params.insert(
            "spine_straightness".to_string(),
            0.5 + params.emotion_vector.dimensions.dominance * 0.3,
        );
        posture_params.insert(
            "shoulder_tension".to_string(),
            params.emotion_vector.dimensions.arousal * 0.4,
        );
        posture_params.insert(
            "head_tilt".to_string(),
            params.emotion_vector.dimensions.valence * 0.2,
        );

        // Animation weights
        if params.emotion_vector.dimensions.arousal > 0.5 {
            animation_weights.insert(
                "energetic_idle".to_string(),
                params.emotion_vector.dimensions.arousal,
            );
        } else {
            animation_weights.insert(
                "calm_idle".to_string(),
                1.0 - params.emotion_vector.dimensions.arousal,
            );
        }

        AvatarEmotionSync {
            facial_expression,
            posture_params,
            animation_weights,
            sync_confidence: 0.85,
        }
    }
}

/// Main VR/AR emotion processor
pub struct VREmotionProcessor {
    /// Configuration
    config: SpatialEmotionConfig,
    /// Current environment type
    current_environment: VREnvironmentType,
    /// User's position in VR/AR space
    user_position: Position3D,
    /// User's head orientation
    user_orientation: Direction3D,
    /// Active spatial emotion sources
    emotion_sources: RwLock<HashMap<String, SpatialEmotionSource>>,
    /// Current combined emotion state
    combined_emotion: RwLock<EmotionParameters>,
    /// Recent gesture history
    gesture_history: RwLock<Vec<(HandGesture, Instant)>>,
    /// Avatar synchronization data
    avatar_sync: RwLock<Option<AvatarEmotionSync>>,
    /// Last processing update
    last_update: RwLock<Instant>,
}

impl VREmotionProcessor {
    /// Create new VR emotion processor
    pub fn new(config: SpatialEmotionConfig) -> Result<Self> {
        Ok(VREmotionProcessor {
            config,
            current_environment: VREnvironmentType::PersonalSpace,
            user_position: [0.0, 1.7, 0.0],     // Default head height
            user_orientation: [0.0, 0.0, -1.0], // Looking forward
            emotion_sources: RwLock::new(HashMap::new()),
            combined_emotion: RwLock::new(EmotionParameters::neutral()),
            gesture_history: RwLock::new(Vec::new()),
            avatar_sync: RwLock::new(None),
            last_update: RwLock::new(Instant::now()),
        })
    }

    /// Set the current VR/AR environment
    pub fn set_environment(&mut self, environment: VREnvironmentType) {
        self.current_environment = environment;
    }

    /// Set user position in VR/AR space
    pub fn set_user_position(&mut self, position: Position3D) {
        self.user_position = position;
    }

    /// Set user head orientation
    pub fn set_user_orientation(&mut self, orientation: Direction3D) {
        self.user_orientation = normalize_vector(orientation);
    }

    /// Add or update a spatial emotion source
    pub async fn add_emotion_source(&self, source: SpatialEmotionSource) -> Result<()> {
        let mut sources = self.emotion_sources.write().await;
        sources.insert(source.id.clone(), source);
        drop(sources);

        self.update_combined_emotion().await?;
        Ok(())
    }

    /// Remove an emotion source
    pub async fn remove_emotion_source(&self, id: &str) -> Result<()> {
        let mut sources = self.emotion_sources.write().await;
        sources.remove(id);
        drop(sources);

        self.update_combined_emotion().await?;
        Ok(())
    }

    /// Process spatial emotion with position and direction
    pub async fn process_spatial_emotion(
        &self,
        emotion: Emotion,
        intensity: EmotionIntensity,
        source_position: Position3D,
        source_direction: Option<Direction3D>,
    ) -> Result<EmotionParameters> {
        let source_id = format!("temp_source_{}", Instant::now().elapsed().as_millis());

        let mut emotion_vector = EmotionVector::new();
        emotion_vector.add_emotion(emotion, intensity);
        let mut emotion_params = EmotionParameters::new(emotion_vector);

        // Apply environment adaptation
        if self.config.enable_environment_adaptation {
            emotion_params = self.apply_environment_adaptation(emotion_params);
        }

        // Apply spatial processing
        if self.config.enable_3d_positioning {
            emotion_params =
                self.apply_spatial_processing(emotion_params, source_position, source_direction);
        }

        // Create temporary source
        let mut source =
            SpatialEmotionSource::new(source_id.clone(), source_position, emotion_params.clone());
        if let Some(dir) = source_direction {
            source.set_direction(dir);
        }

        // Add to sources temporarily
        self.add_emotion_source(source).await?;

        // Generate haptic feedback if enabled
        if self.config.enable_haptic_feedback {
            self.generate_haptic_feedback(&emotion_params).await?;
        }

        // Update avatar sync if enabled
        if self.config.enable_avatar_sync {
            self.update_avatar_sync(&emotion_params).await?;
        }

        // Clean up temporary source after processing
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            // Note: In real implementation, you'd want to remove the temp source
        });

        Ok(emotion_params)
    }

    /// Register hand gesture for emotion influence
    pub async fn register_gesture(&self, gesture: HandGesture) -> Result<()> {
        if !self.config.enable_gesture_control {
            return Ok(());
        }

        let mut history = self.gesture_history.write().await;
        history.push((gesture, Instant::now()));

        // Keep only recent gestures (last 5 seconds)
        let cutoff = Instant::now() - Duration::from_secs(5);
        history.retain(|(_, timestamp)| *timestamp > cutoff);

        drop(history);

        // Apply gesture influence to current emotion
        self.apply_gesture_influence(gesture).await?;
        Ok(())
    }

    /// Get current combined emotion state
    pub async fn get_combined_emotion(&self) -> EmotionParameters {
        self.combined_emotion.read().await.clone()
    }

    /// Get current avatar synchronization data
    pub async fn get_avatar_sync(&self) -> Option<AvatarEmotionSync> {
        self.avatar_sync.read().await.clone()
    }

    /// Get haptic pattern for current emotion state
    pub async fn get_haptic_pattern(&self) -> Result<HapticPattern> {
        let emotion_params = self.get_combined_emotion().await;
        let (primary_emotion, intensity) = emotion_params
            .emotion_vector
            .dominant_emotion()
            .unwrap_or((Emotion::Neutral, EmotionIntensity::VERY_LOW));

        Ok(HapticPattern::for_emotion(primary_emotion, intensity))
    }

    /// Apply environment adaptation to emotion parameters
    fn apply_environment_adaptation(&self, mut params: EmotionParameters) -> EmotionParameters {
        let intensity_mod = self.current_environment.intensity_modifier();
        let [val_weight, aro_weight, dom_weight] = self.current_environment.adaptation_weights();

        // Apply intensity modification
        params.emotion_vector.emotions = params
            .emotion_vector
            .emotions
            .into_iter()
            .map(|(emotion, intensity)| {
                (
                    emotion,
                    EmotionIntensity::new(intensity.value() * intensity_mod),
                )
            })
            .collect();

        // Apply dimensional weights
        params.emotion_vector.dimensions.valence *= val_weight;
        params.emotion_vector.dimensions.arousal *= aro_weight;
        params.emotion_vector.dimensions.dominance *= dom_weight;

        // Clamp values to valid range
        params.emotion_vector.dimensions.valence =
            params.emotion_vector.dimensions.valence.clamp(-1.0, 1.0);
        params.emotion_vector.dimensions.arousal =
            params.emotion_vector.dimensions.arousal.clamp(-1.0, 1.0);
        params.emotion_vector.dimensions.dominance =
            params.emotion_vector.dimensions.dominance.clamp(-1.0, 1.0);

        params
    }

    /// Apply spatial processing effects
    fn apply_spatial_processing(
        &self,
        mut params: EmotionParameters,
        source_position: Position3D,
        _source_direction: Option<Direction3D>,
    ) -> EmotionParameters {
        let distance = distance_3d(source_position, self.user_position);

        if distance > self.config.max_source_distance {
            // Source too far away, minimal influence
            params.emotion_vector.emotions = params
                .emotion_vector
                .emotions
                .into_iter()
                .map(|(emotion, intensity)| {
                    (emotion, EmotionIntensity::new(intensity.value() * 0.1))
                })
                .collect();
            return params;
        }

        // Calculate distance attenuation
        let distance_factor =
            1.0 - (distance / self.config.max_source_distance).powf(self.config.distance_falloff);

        // Apply distance-based intensity modification
        params.emotion_vector.emotions = params
            .emotion_vector
            .emotions
            .into_iter()
            .map(|(emotion, intensity)| {
                (
                    emotion,
                    EmotionIntensity::new(intensity.value() * distance_factor),
                )
            })
            .collect();

        // Apply spatial audio effects (simplified)
        if distance > 2.0 {
            // Add some "distance" characteristics to prosody
            params.pitch_shift *= 0.95; // Slightly lower pitch for distant sources
            params.energy_scale *= distance_factor; // Reduce energy based on distance
        }

        params
    }

    /// Update combined emotion from all sources
    async fn update_combined_emotion(&self) -> Result<()> {
        let sources = self.emotion_sources.read().await;

        if sources.is_empty() {
            let mut combined = self.combined_emotion.write().await;
            *combined = EmotionParameters::neutral();
            return Ok(());
        }

        // Calculate weighted combination of all sources
        let mut total_weight = 0.0;
        let mut combined_dimensions = EmotionDimensions::default();
        let mut combined_emotions: HashMap<Emotion, f32> = HashMap::new();

        for source in sources.values() {
            let influence = source.calculate_influence(self.user_position, &self.config);
            if influence <= 0.0 {
                continue;
            }

            total_weight += influence;

            // Combine dimensions
            combined_dimensions.valence +=
                source.emotion_params.emotion_vector.dimensions.valence * influence;
            combined_dimensions.arousal +=
                source.emotion_params.emotion_vector.dimensions.arousal * influence;
            combined_dimensions.dominance +=
                source.emotion_params.emotion_vector.dimensions.dominance * influence;

            // Combine emotions
            for (emotion, intensity) in &source.emotion_params.emotion_vector.emotions {
                let weighted_intensity = intensity.value() * influence;
                *combined_emotions.entry(emotion.clone()).or_insert(0.0) += weighted_intensity;
            }
        }

        if total_weight > 0.0 {
            // Normalize by total weight
            combined_dimensions.valence /= total_weight;
            combined_dimensions.arousal /= total_weight;
            combined_dimensions.dominance /= total_weight;

            for intensity in combined_emotions.values_mut() {
                *intensity /= total_weight;
            }

            // Create combined emotion parameters
            let mut emotion_vector = EmotionVector::new();
            emotion_vector.dimensions = combined_dimensions;

            for (emotion, intensity) in combined_emotions {
                emotion_vector.add_emotion(emotion, EmotionIntensity::new(intensity));
            }

            let combined_params = EmotionParameters::new(emotion_vector);
            let mut combined = self.combined_emotion.write().await;
            *combined = combined_params;
        }

        Ok(())
    }

    /// Generate haptic feedback for emotion
    async fn generate_haptic_feedback(&self, params: &EmotionParameters) -> Result<()> {
        let (primary_emotion, intensity) = params
            .emotion_vector
            .dominant_emotion()
            .unwrap_or((Emotion::Neutral, EmotionIntensity::VERY_LOW));
        let _pattern = HapticPattern::for_emotion(primary_emotion, intensity);

        // In a real implementation, this would send haptic commands to VR controllers
        // For now, we just calculate the pattern

        Ok(())
    }

    /// Apply gesture influence to emotion
    async fn apply_gesture_influence(&self, gesture: HandGesture) -> Result<()> {
        let (gesture_emotion, influence_strength) = gesture.to_emotion_influence();

        let mut combined = self.combined_emotion.write().await;

        // Apply gesture influence to current emotion
        let gesture_intensity = EmotionIntensity::new(influence_strength * 0.3); // 30% influence
        combined
            .emotion_vector
            .add_emotion(gesture_emotion, gesture_intensity);

        // Re-normalize to prevent overflow
        combined.emotion_vector.normalize();

        Ok(())
    }

    /// Update avatar synchronization
    async fn update_avatar_sync(&self, params: &EmotionParameters) -> Result<()> {
        let sync_data = AvatarEmotionSync::from_emotion_params(params);
        let mut avatar_sync = self.avatar_sync.write().await;
        *avatar_sync = Some(sync_data);
        Ok(())
    }
}

// Utility functions

/// Calculate 3D distance between two points
fn distance_3d(pos1: Position3D, pos2: Position3D) -> f32 {
    let dx = pos1[0] - pos2[0];
    let dy = pos1[1] - pos2[1];
    let dz = pos1[2] - pos2[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Normalize a 3D vector
fn normalize_vector(mut vec: Direction3D) -> Direction3D {
    let length = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]).sqrt();
    if length > 0.0 {
        vec[0] /= length;
        vec[1] /= length;
        vec[2] /= length;
    }
    vec
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Emotion;

    #[test]
    fn test_vr_environment_modifiers() {
        let game_env = VREnvironmentType::GameEnvironment;
        assert_eq!(game_env.intensity_modifier(), 1.5);

        let therapeutic = VREnvironmentType::TherapeuticSpace;
        assert_eq!(therapeutic.intensity_modifier(), 0.6);
    }

    #[test]
    fn test_hand_gesture_emotion_mapping() {
        let thumbs_up = HandGesture::ThumbsUp;
        let (emotion, strength) = thumbs_up.to_emotion_influence();
        assert_eq!(emotion, Emotion::Happy);
        assert_eq!(strength, 0.9);

        let fist = HandGesture::ClosedFist;
        let (emotion, strength) = fist.to_emotion_influence();
        assert_eq!(emotion, Emotion::Angry);
        assert_eq!(strength, 0.8);
    }

    #[test]
    fn test_haptic_pattern_creation() {
        let pattern = HapticPattern::for_emotion(Emotion::Happy, EmotionIntensity::HIGH);
        assert_eq!(pattern.pulse_count, 3);
        assert_eq!(pattern.frequency, 200.0);
        // Happy has intensity multiplier of 0.6, so HIGH (0.7) * 0.6 = 0.42
        assert!((pattern.intensity - 0.42).abs() < 0.01);

        let sad_pattern = HapticPattern::for_emotion(Emotion::Sad, EmotionIntensity::MEDIUM);
        assert_eq!(sad_pattern.pulse_count, 1);
        assert_eq!(sad_pattern.frequency, 80.0);
        // Sad has intensity multiplier of 0.4, so MEDIUM (0.5) * 0.4 = 0.2
        assert!((sad_pattern.intensity - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_spatial_config_builder() {
        let config = SpatialEmotionConfig::new()
            .with_3d_positioning(true)
            .with_haptic_feedback(true)
            .with_max_distance(15.0)
            .with_distance_falloff(0.5);

        assert!(config.enable_3d_positioning);
        assert!(config.enable_haptic_feedback);
        assert_eq!(config.max_source_distance, 15.0);
        assert_eq!(config.distance_falloff, 0.5);
    }

    #[tokio::test]
    async fn test_vr_processor_creation() {
        let config = SpatialEmotionConfig::default();
        let processor = VREmotionProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[tokio::test]
    async fn test_spatial_emotion_source() {
        let emotion_params = EmotionParameters::neutral();
        let mut source =
            SpatialEmotionSource::new("test_source".to_string(), [1.0, 2.0, 3.0], emotion_params);

        assert_eq!(source.position, [1.0, 2.0, 3.0]);

        source.set_position([4.0, 5.0, 6.0]);
        assert_eq!(source.position, [4.0, 5.0, 6.0]);

        let distance = source.distance_to([0.0, 0.0, 0.0]);
        assert!((distance - 8.77).abs() < 0.1); // sqrt(4²+5²+6²) ≈ 8.77
    }

    #[test]
    fn test_distance_calculation() {
        let dist = distance_3d([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]);
        assert_eq!(dist, 5.0); // 3-4-5 triangle
    }

    #[test]
    fn test_vector_normalization() {
        let normalized = normalize_vector([3.0, 4.0, 0.0]);
        assert!((normalized[0] - 0.6).abs() < 0.001);
        assert!((normalized[1] - 0.8).abs() < 0.001);
        assert!((normalized[2] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_avatar_emotion_sync() {
        let mut emotion_vector = EmotionVector::new();
        emotion_vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);
        let params = EmotionParameters::new(emotion_vector);

        let sync = AvatarEmotionSync::from_emotion_params(&params);
        assert!(sync.facial_expression.contains_key("mouth_curve"));
        assert!(sync.posture_params.contains_key("spine_straightness"));
        assert!(sync.sync_confidence > 0.8);
    }

    #[tokio::test]
    async fn test_environment_adaptation() {
        let config = SpatialEmotionConfig::new().with_environment_adaptation(true);
        let mut processor = VREmotionProcessor::new(config).unwrap();
        processor.set_environment(VREnvironmentType::GameEnvironment);

        let mut emotion_vector = EmotionVector::new();
        emotion_vector.add_emotion(Emotion::Excited, EmotionIntensity::MEDIUM);
        let original_params = EmotionParameters::new(emotion_vector);

        let adapted = processor.apply_environment_adaptation(original_params);

        // Game environment should amplify emotions
        let (_, primary_intensity) = adapted
            .emotion_vector
            .dominant_emotion()
            .unwrap_or((Emotion::Neutral, EmotionIntensity::VERY_LOW));
        assert!(primary_intensity.value() > 0.5); // Should be amplified
    }

    #[tokio::test]
    async fn test_gesture_registration() {
        let config = SpatialEmotionConfig::new().with_gesture_control(true);
        let processor = VREmotionProcessor::new(config).unwrap();

        let result = processor.register_gesture(HandGesture::ThumbsUp).await;
        assert!(result.is_ok());

        // Check that gesture was recorded
        let history = processor.gesture_history.read().await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].0, HandGesture::ThumbsUp);
    }

    #[tokio::test]
    async fn test_spatial_emotion_processing() {
        let config = SpatialEmotionConfig::new()
            .with_3d_positioning(true)
            .with_environment_adaptation(true);
        let processor = VREmotionProcessor::new(config).unwrap();

        let result = processor
            .process_spatial_emotion(
                Emotion::Happy,
                EmotionIntensity::HIGH,
                [2.0, 1.0, -1.0],
                Some([0.0, 0.0, -1.0]),
            )
            .await;

        assert!(result.is_ok());
        let params = result.unwrap();
        let (primary_emotion, _) = params
            .emotion_vector
            .dominant_emotion()
            .unwrap_or((Emotion::Neutral, EmotionIntensity::VERY_LOW));
        assert_eq!(primary_emotion, Emotion::Happy);
    }
}
