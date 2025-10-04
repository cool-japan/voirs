//! Feature-specific builder extensions for advanced voice capabilities.
//!
//! This module provides consistent builder patterns for integrating advanced features
//! like emotion control, voice cloning, voice conversion, singing synthesis, and
//! 3D spatial audio into the main pipeline builder.

use super::builder_impl::VoirsPipelineBuilder;

/// Trait for feature-specific builders to ensure consistency
#[async_trait::async_trait]
pub trait FeatureBuilder<T> {
    /// Enable the feature with default configuration
    fn with_enabled(enabled: bool) -> Self;
    /// Apply custom configuration
    fn with_config(config: T) -> Self;
    /// Build the feature controller
    async fn build(self) -> crate::Result<Box<dyn std::any::Any + Send + Sync>>;
}

impl VoirsPipelineBuilder {
    // =============================================================================
    // Emotion Control Feature
    // =============================================================================

    /// Enable emotion control with default settings
    #[cfg(feature = "emotion")]
    pub fn with_emotion_control_enabled(mut self, enabled: bool) -> Self {
        if enabled {
            self.config.default_synthesis.enable_emotion = true;
        } else {
            self.config.default_synthesis.enable_emotion = false;
        }
        self
    }

    /// Configure emotion control with custom builder
    #[cfg(feature = "emotion")]
    pub fn with_emotion_control(
        mut self,
        builder: crate::emotion::EmotionControllerBuilder,
    ) -> Self {
        self.config.default_synthesis.enable_emotion = true;
        // Store the builder for later use during pipeline initialization
        // For now, we'll use the configuration system to pass parameters
        self
    }

    /// Set default emotion for synthesis
    #[cfg(feature = "emotion")]
    pub fn with_default_emotion(mut self, emotion_type: &str, intensity: f32) -> Self {
        self.config.default_synthesis.emotion_type = Some(emotion_type.to_string());
        self.config.default_synthesis.emotion_intensity = intensity.clamp(0.0, 1.0);
        self.config.default_synthesis.enable_emotion = true;
        self
    }

    /// Apply emotion preset configuration
    #[cfg(feature = "emotion")]
    pub fn with_emotion_preset(mut self, preset: EmotionPreset) -> Self {
        let (emotion_type, intensity) = match preset {
            EmotionPreset::Happy => ("happy", 0.8),
            EmotionPreset::Sad => ("sad", 0.7),
            EmotionPreset::Excited => ("excited", 0.9),
            EmotionPreset::Calm => ("calm", 0.6),
            EmotionPreset::Angry => ("angry", 0.8),
            EmotionPreset::Neutral => ("neutral", 0.0),
        };
        self.with_default_emotion(emotion_type, intensity)
    }

    /// Enable automatic emotion detection from text
    #[cfg(feature = "emotion")]
    pub fn with_auto_emotion_detection(mut self, enabled: bool) -> Self {
        self.config.default_synthesis.auto_emotion_detection = enabled;
        self
    }

    // =============================================================================
    // Voice Cloning Feature
    // =============================================================================

    /// Enable voice cloning with default settings
    #[cfg(feature = "cloning")]
    pub fn with_voice_cloning_enabled(mut self, enabled: bool) -> Self {
        if enabled {
            self.config.default_synthesis.enable_cloning = true;
        } else {
            self.config.default_synthesis.enable_cloning = false;
        }
        self
    }

    /// Configure voice cloning with custom builder
    #[cfg(feature = "cloning")]
    pub fn with_voice_cloning(mut self, builder: crate::cloning::VoiceClonerBuilder) -> Self {
        self.config.default_synthesis.enable_cloning = true;
        // Store the builder configuration for later use
        self
    }

    /// Set default cloning method
    #[cfg(feature = "cloning")]
    pub fn with_cloning_method(mut self, method: CloningMethod) -> Self {
        self.config.default_synthesis.cloning_method = Some(method);
        self.config.default_synthesis.enable_cloning = true;
        self
    }

    /// Apply cloning preset configuration
    #[cfg(feature = "cloning")]
    pub fn with_cloning_preset(mut self, preset: CloningPreset) -> Self {
        let (method, quality) = match preset {
            CloningPreset::HighQuality => (CloningMethod::DeepClone, 0.95),
            CloningPreset::Fast => (CloningMethod::QuickClone, 0.7),
            CloningPreset::Balanced => (CloningMethod::AdaptiveClone, 0.85),
        };
        self.config.default_synthesis.cloning_method = Some(method);
        self.config.default_synthesis.cloning_quality = quality;
        self.config.default_synthesis.enable_cloning = true;
        self
    }

    // =============================================================================
    // Voice Conversion Feature
    // =============================================================================

    /// Enable voice conversion with default settings
    #[cfg(feature = "conversion")]
    pub fn with_voice_conversion_enabled(mut self, enabled: bool) -> Self {
        if enabled {
            self.config.default_synthesis.enable_conversion = true;
        } else {
            self.config.default_synthesis.enable_conversion = false;
        }
        self
    }

    /// Configure voice conversion with custom builder
    #[cfg(feature = "conversion")]
    pub fn with_voice_conversion(
        mut self,
        builder: crate::conversion::VoiceConverterBuilder,
    ) -> Self {
        self.config.default_synthesis.enable_conversion = true;
        // Store the builder configuration for later use
        self
    }

    /// Set default conversion target
    #[cfg(feature = "conversion")]
    pub fn with_conversion_target(mut self, target: ConversionTarget) -> Self {
        self.config.default_synthesis.conversion_target = Some(target);
        self.config.default_synthesis.enable_conversion = true;
        self
    }

    /// Apply conversion preset configuration
    #[cfg(feature = "conversion")]
    pub fn with_conversion_preset(mut self, preset: ConversionPreset) -> Self {
        let target = match preset {
            ConversionPreset::MaleToFemale => ConversionTarget::Gender(Gender::Female),
            ConversionPreset::FemaleToMale => ConversionTarget::Gender(Gender::Male),
            ConversionPreset::YoungToOld => ConversionTarget::Age(AgeGroup::Senior),
            ConversionPreset::OldToYoung => ConversionTarget::Age(AgeGroup::Young),
        };
        self.with_conversion_target(target)
    }

    /// Enable real-time conversion
    #[cfg(feature = "conversion")]
    pub fn with_realtime_conversion(mut self, enabled: bool) -> Self {
        self.config.default_synthesis.realtime_conversion = enabled;
        self
    }

    // =============================================================================
    // Singing Synthesis Feature
    // =============================================================================

    /// Enable singing synthesis with default settings
    #[cfg(feature = "singing")]
    pub fn with_singing_synthesis_enabled(mut self, enabled: bool) -> Self {
        if enabled {
            self.config.default_synthesis.enable_singing = true;
        } else {
            self.config.default_synthesis.enable_singing = false;
        }
        self
    }

    /// Configure singing synthesis with custom builder
    #[cfg(feature = "singing")]
    pub fn with_singing_synthesis(
        mut self,
        builder: crate::singing::SingingControllerBuilder,
    ) -> Self {
        self.config.default_synthesis.enable_singing = true;
        // Store the builder configuration for later use
        self
    }

    /// Set default singing voice type
    #[cfg(feature = "singing")]
    pub fn with_singing_voice_type(mut self, voice_type: SingingVoiceType) -> Self {
        self.config.default_synthesis.singing_voice_type = Some(voice_type);
        self.config.default_synthesis.enable_singing = true;
        self
    }

    /// Apply singing preset configuration
    #[cfg(feature = "singing")]
    pub fn with_singing_preset(mut self, preset: SingingPreset) -> Self {
        let (voice_type, technique) = match preset {
            SingingPreset::PopSinger => (SingingVoiceType::PopVocalist, SingingTechnique::modern()),
            SingingPreset::OperaSinger => {
                (SingingVoiceType::OperaSinger, SingingTechnique::classical())
            }
            SingingPreset::JazzSinger => (SingingVoiceType::JazzVocalist, SingingTechnique::jazz()),
            SingingPreset::RockSinger => (SingingVoiceType::RockVocalist, SingingTechnique::rock()),
        };
        self.config.default_synthesis.singing_voice_type = Some(voice_type);
        self.config.default_synthesis.singing_technique = Some(technique);
        self.config.default_synthesis.enable_singing = true;
        self
    }

    /// Set default musical key
    #[cfg(feature = "singing")]
    pub fn with_musical_key(mut self, key: MusicalKey) -> Self {
        self.config.default_synthesis.musical_key = Some(key);
        self
    }

    /// Set default tempo (BPM)
    #[cfg(feature = "singing")]
    pub fn with_tempo(mut self, bpm: f32) -> Self {
        self.config.default_synthesis.tempo = Some(bpm);
        self
    }

    // =============================================================================
    // 3D Spatial Audio Feature
    // =============================================================================

    /// Enable 3D spatial audio with default settings
    #[cfg(feature = "spatial")]
    pub fn with_spatial_audio_enabled(mut self, enabled: bool) -> Self {
        if enabled {
            self.config.default_synthesis.enable_spatial = true;
        } else {
            self.config.default_synthesis.enable_spatial = false;
        }
        self
    }

    /// Configure 3D spatial audio with custom builder
    #[cfg(feature = "spatial")]
    pub fn with_spatial_audio(
        mut self,
        builder: crate::spatial::SpatialAudioControllerBuilder,
    ) -> Self {
        self.config.default_synthesis.enable_spatial = true;
        // Store the builder configuration for later use
        self
    }

    /// Set default listener position
    #[cfg(feature = "spatial")]
    pub fn with_listener_position(mut self, position: Position3D) -> Self {
        self.config.default_synthesis.listener_position = Some(position);
        self.config.default_synthesis.enable_spatial = true;
        self
    }

    /// Apply spatial audio preset configuration
    #[cfg(feature = "spatial")]
    pub fn with_spatial_preset(mut self, preset: SpatialPreset) -> Self {
        let (hrtf_enabled, room_size, reverb_level) = match preset {
            SpatialPreset::Headphones => (true, RoomSize::Small, 0.2),
            SpatialPreset::Speakers => (false, RoomSize::Medium, 0.4),
            SpatialPreset::VirtualReality => (true, RoomSize::Large, 0.6),
            SpatialPreset::AugmentedReality => (true, RoomSize::Medium, 0.3),
        };
        self.config.default_synthesis.hrtf_enabled = hrtf_enabled;
        self.config.default_synthesis.room_size = Some(room_size);
        self.config.default_synthesis.reverb_level = reverb_level;
        self.config.default_synthesis.enable_spatial = true;
        self
    }

    /// Enable HRTF processing
    #[cfg(feature = "spatial")]
    pub fn with_hrtf_processing(mut self, enabled: bool) -> Self {
        self.config.default_synthesis.hrtf_enabled = enabled;
        self
    }

    /// Set room acoustics parameters
    #[cfg(feature = "spatial")]
    pub fn with_room_acoustics(mut self, room_size: RoomSize, reverb_level: f32) -> Self {
        self.config.default_synthesis.room_size = Some(room_size);
        self.config.default_synthesis.reverb_level = reverb_level.clamp(0.0, 1.0);
        self
    }
}

// =============================================================================
// Feature Preset Enums
// =============================================================================

/// Emotion control presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmotionPreset {
    Happy,
    Sad,
    Excited,
    Calm,
    Angry,
    Neutral,
}

/// Voice cloning presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloningPreset {
    HighQuality,
    Fast,
    Balanced,
}

/// Voice conversion presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionPreset {
    MaleToFemale,
    FemaleToMale,
    YoungToOld,
    OldToYoung,
}

/// Singing synthesis presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingingPreset {
    PopSinger,
    OperaSinger,
    JazzSinger,
    RockSinger,
}

/// 3D spatial audio presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialPreset {
    Headphones,
    Speakers,
    VirtualReality,
    AugmentedReality,
}

// =============================================================================
// Feature Configuration Types
// =============================================================================

/// Voice cloning method
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CloningMethod {
    QuickClone,
    DeepClone,
    AdaptiveClone,
}

/// Voice conversion target
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ConversionTarget {
    Gender(Gender),
    Age(AgeGroup),
    Voice(String),
}

/// Singing voice types
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SingingVoiceType {
    PopVocalist,
    OperaSinger,
    JazzVocalist,
    RockVocalist,
    Choir,
}

/// Musical key
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MusicalKey {
    C,
    CSharp,
    D,
    DSharp,
    E,
    F,
    FSharp,
    G,
    GSharp,
    A,
    ASharp,
    B,
}

/// Room size for spatial audio
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RoomSize {
    Small,
    Medium,
    Large,
    Huge,
}

/// Gender for voice conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Gender {
    Male,
    Female,
}

/// Age group for voice conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AgeGroup {
    Child,
    Young,
    Adult,
    Senior,
}

/// 3D position
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Position3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Singing technique configuration
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SingingTechnique {
    pub breath_control: f32,
    pub vibrato_depth: f32,
    pub vocal_fry: f32,
    pub head_voice_ratio: f32,
}

impl SingingTechnique {
    pub fn modern() -> Self {
        Self {
            breath_control: 0.7,
            vibrato_depth: 0.3,
            vocal_fry: 0.2,
            head_voice_ratio: 0.6,
        }
    }

    pub fn classical() -> Self {
        Self {
            breath_control: 0.9,
            vibrato_depth: 0.5,
            vocal_fry: 0.0,
            head_voice_ratio: 0.8,
        }
    }

    pub fn jazz() -> Self {
        Self {
            breath_control: 0.6,
            vibrato_depth: 0.4,
            vocal_fry: 0.3,
            head_voice_ratio: 0.5,
        }
    }

    pub fn rock() -> Self {
        Self {
            breath_control: 0.8,
            vibrato_depth: 0.2,
            vocal_fry: 0.4,
            head_voice_ratio: 0.3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_presets() {
        #[cfg(feature = "emotion")]
        {
            let happy_builder =
                VoirsPipelineBuilder::new().with_emotion_preset(EmotionPreset::Happy);
            assert!(happy_builder.config.default_synthesis.enable_emotion);
            assert_eq!(
                happy_builder.config.default_synthesis.emotion_type,
                Some("happy".to_string())
            );
            assert_eq!(
                happy_builder.config.default_synthesis.emotion_intensity,
                0.8
            );

            let calm_builder = VoirsPipelineBuilder::new().with_emotion_preset(EmotionPreset::Calm);
            assert!(calm_builder.config.default_synthesis.enable_emotion);
            assert_eq!(
                calm_builder.config.default_synthesis.emotion_type,
                Some("calm".to_string())
            );
            assert_eq!(calm_builder.config.default_synthesis.emotion_intensity, 0.6);
        }
    }

    #[test]
    fn test_feature_combinations() {
        #[cfg(all(feature = "emotion", feature = "spatial"))]
        {
            let combined_builder = VoirsPipelineBuilder::new()
                .with_emotion_preset(EmotionPreset::Happy)
                .with_spatial_preset(SpatialPreset::Headphones);

            assert!(combined_builder.config.default_synthesis.enable_emotion);
            assert!(combined_builder.config.default_synthesis.enable_spatial);
            assert!(combined_builder.config.default_synthesis.hrtf_enabled);
        }
    }

    #[test]
    fn test_singing_technique_presets() {
        let modern = SingingTechnique::modern();
        assert_eq!(modern.breath_control, 0.7);
        assert_eq!(modern.vibrato_depth, 0.3);

        let classical = SingingTechnique::classical();
        assert_eq!(classical.breath_control, 0.9);
        assert_eq!(classical.vibrato_depth, 0.5);
        assert_eq!(classical.vocal_fry, 0.0);
    }
}
