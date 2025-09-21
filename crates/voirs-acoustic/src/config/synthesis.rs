//! Synthesis configuration for acoustic models
//!
//! This module defines configuration structures for controlling synthesis
//! parameters including speaker settings, prosody control, and quality options.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{AcousticError, Result};

/// Synthesis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisConfig {
    /// Speaker configuration
    pub speaker: SpeakerConfig,
    /// Prosody control parameters
    pub prosody: ProsodyConfig,
    /// Quality vs speed trade-offs
    pub quality: QualityConfig,
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
    /// Batch size for batch processing
    pub batch_size: Option<u32>,
    /// Device selection
    pub device: Option<String>,
}

impl SynthesisConfig {
    /// Create new synthesis configuration
    pub fn new() -> Self {
        Self {
            speaker: SpeakerConfig::default(),
            prosody: ProsodyConfig::default(),
            quality: QualityConfig::default(),
            seed: None,
            batch_size: None,
            device: None,
        }
    }

    /// Create configuration for specific speaker
    pub fn with_speaker(speaker_id: u32) -> Self {
        let mut config = Self::new();
        config.speaker.speaker_id = Some(speaker_id);
        config
    }

    /// Create configuration with prosody control
    pub fn with_prosody(speed: f32, pitch_shift: f32, energy: f32) -> Self {
        let mut config = Self::new();
        config.prosody.speed = speed;
        config.prosody.pitch_shift = pitch_shift;
        config.prosody.energy = energy;
        config
    }

    /// Validate synthesis configuration
    pub fn validate(&self) -> Result<()> {
        self.speaker.validate()?;
        self.prosody.validate()?;
        self.quality.validate()?;

        if let Some(batch_size) = self.batch_size {
            if batch_size == 0 {
                return Err(AcousticError::ConfigError(
                    "Batch size must be > 0".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Merge with another synthesis configuration
    pub fn merge(&mut self, other: &SynthesisConfig) {
        self.speaker.merge(&other.speaker);
        self.prosody.merge(&other.prosody);
        self.quality.merge(&other.quality);

        if other.seed.is_some() {
            self.seed = other.seed;
        }
        if other.batch_size.is_some() {
            self.batch_size = other.batch_size;
        }
        if other.device.is_some() {
            self.device = other.device.clone();
        }
    }
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Speaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerConfig {
    /// Speaker ID for multi-speaker models
    pub speaker_id: Option<u32>,
    /// Speaker embedding vector (alternative to ID)
    pub speaker_embedding: Option<Vec<f32>>,
    /// Speaker interpolation weights (for voice morphing)
    pub speaker_mix: Option<HashMap<u32, f32>>,
    /// Emotion control
    pub emotion: Option<EmotionConfig>,
    /// Voice characteristics
    pub voice_characteristics: VoiceCharacteristics,
}

impl SpeakerConfig {
    /// Create new speaker configuration
    pub fn new() -> Self {
        Self {
            speaker_id: None,
            speaker_embedding: None,
            speaker_mix: None,
            emotion: None,
            voice_characteristics: VoiceCharacteristics::default(),
        }
    }

    /// Create configuration for specific speaker
    pub fn with_speaker_id(speaker_id: u32) -> Self {
        let mut config = Self::new();
        config.speaker_id = Some(speaker_id);
        config
    }

    /// Create configuration with speaker embedding
    pub fn with_embedding(embedding: Vec<f32>) -> Self {
        let mut config = Self::new();
        config.speaker_embedding = Some(embedding);
        config
    }

    /// Validate speaker configuration
    pub fn validate(&self) -> Result<()> {
        if let Some(speaker_mix) = &self.speaker_mix {
            let total_weight: f32 = speaker_mix.values().sum();
            if (total_weight - 1.0).abs() > 1e-6 {
                return Err(AcousticError::ConfigError(
                    "Speaker mix weights must sum to 1.0".to_string(),
                ));
            }
        }

        if let Some(emotion) = &self.emotion {
            emotion.validate()?;
        }

        self.voice_characteristics.validate()?;
        Ok(())
    }

    /// Merge with another speaker configuration
    pub fn merge(&mut self, other: &SpeakerConfig) {
        if other.speaker_id.is_some() {
            self.speaker_id = other.speaker_id;
        }
        if other.speaker_embedding.is_some() {
            self.speaker_embedding = other.speaker_embedding.clone();
        }
        if other.speaker_mix.is_some() {
            self.speaker_mix = other.speaker_mix.clone();
        }
        if other.emotion.is_some() {
            self.emotion = other.emotion.clone();
        }
        self.voice_characteristics
            .merge(&other.voice_characteristics);
    }
}

impl Default for SpeakerConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Emotion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionConfig {
    /// Emotion type
    pub emotion_type: EmotionType,
    /// Emotion intensity (0.0 to 1.0)
    pub intensity: f32,
    /// Emotion vector (alternative to type + intensity)
    pub emotion_vector: Option<Vec<f32>>,
}

impl EmotionConfig {
    /// Create new emotion configuration
    pub fn new(emotion_type: EmotionType, intensity: f32) -> Self {
        Self {
            emotion_type,
            intensity,
            emotion_vector: None,
        }
    }

    /// Create configuration with emotion vector
    pub fn with_vector(emotion_vector: Vec<f32>) -> Self {
        Self {
            emotion_type: EmotionType::Neutral,
            intensity: 1.0,
            emotion_vector: Some(emotion_vector),
        }
    }

    /// Validate emotion configuration
    pub fn validate(&self) -> Result<()> {
        if self.intensity < 0.0 || self.intensity > 1.0 {
            return Err(AcousticError::ConfigError(
                "Emotion intensity must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Emotion types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmotionType {
    Neutral,
    Happy,
    Sad,
    Angry,
    Fearful,
    Disgusted,
    Surprised,
    Excited,
    Calm,
    Confident,
    Uncertain,
}

impl EmotionType {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            EmotionType::Neutral => "neutral",
            EmotionType::Happy => "happy",
            EmotionType::Sad => "sad",
            EmotionType::Angry => "angry",
            EmotionType::Fearful => "fearful",
            EmotionType::Disgusted => "disgusted",
            EmotionType::Surprised => "surprised",
            EmotionType::Excited => "excited",
            EmotionType::Calm => "calm",
            EmotionType::Confident => "confident",
            EmotionType::Uncertain => "uncertain",
        }
    }
}

/// Voice characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCharacteristics {
    /// Age category
    pub age: Option<AgeCategory>,
    /// Gender
    pub gender: Option<Gender>,
    /// Accent/dialect
    pub accent: Option<String>,
    /// Voice quality parameters
    pub quality: VoiceQuality,
}

impl VoiceCharacteristics {
    /// Create new voice characteristics
    pub fn new() -> Self {
        Self {
            age: None,
            gender: None,
            accent: None,
            quality: VoiceQuality::default(),
        }
    }

    /// Validate voice characteristics
    pub fn validate(&self) -> Result<()> {
        self.quality.validate()?;
        Ok(())
    }

    /// Merge with another voice characteristics
    pub fn merge(&mut self, other: &VoiceCharacteristics) {
        if other.age.is_some() {
            self.age = other.age;
        }
        if other.gender.is_some() {
            self.gender = other.gender;
        }
        if other.accent.is_some() {
            self.accent = other.accent.clone();
        }
        self.quality.merge(&other.quality);
    }
}

impl Default for VoiceCharacteristics {
    fn default() -> Self {
        Self::new()
    }
}

/// Age categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgeCategory {
    Child,
    Young,
    Adult,
    Middle,
    Senior,
}

impl AgeCategory {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            AgeCategory::Child => "child",
            AgeCategory::Young => "young",
            AgeCategory::Adult => "adult",
            AgeCategory::Middle => "middle",
            AgeCategory::Senior => "senior",
        }
    }
}

/// Gender categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Gender {
    Male,
    Female,
    NonBinary,
}

impl Gender {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Gender::Male => "male",
            Gender::Female => "female",
            Gender::NonBinary => "non-binary",
        }
    }
}

/// Voice quality parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQuality {
    /// Breathiness (0.0 to 1.0)
    pub breathiness: f32,
    /// Roughness (0.0 to 1.0)
    pub roughness: f32,
    /// Tenseness (0.0 to 1.0)
    pub tenseness: f32,
    /// Creakiness (0.0 to 1.0)
    pub creakiness: f32,
}

impl VoiceQuality {
    /// Create new voice quality
    pub fn new() -> Self {
        Self {
            breathiness: 0.0,
            roughness: 0.0,
            tenseness: 0.0,
            creakiness: 0.0,
        }
    }

    /// Validate voice quality
    pub fn validate(&self) -> Result<()> {
        let qualities = [
            self.breathiness,
            self.roughness,
            self.tenseness,
            self.creakiness,
        ];
        for (i, &quality) in qualities.iter().enumerate() {
            if !(0.0..=1.0).contains(&quality) {
                let name = match i {
                    0 => "breathiness",
                    1 => "roughness",
                    2 => "tenseness",
                    3 => "creakiness",
                    _ => "unknown_quality",
                };
                return Err(AcousticError::ConfigError(format!(
                    "{name} must be between 0.0 and 1.0"
                )));
            }
        }
        Ok(())
    }

    /// Merge with another voice quality
    pub fn merge(&mut self, other: &VoiceQuality) {
        self.breathiness = other.breathiness;
        self.roughness = other.roughness;
        self.tenseness = other.tenseness;
        self.creakiness = other.creakiness;
    }
}

impl Default for VoiceQuality {
    fn default() -> Self {
        Self::new()
    }
}

/// Prosody configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodyConfig {
    /// Speaking rate multiplier (1.0 = normal)
    pub speed: f32,
    /// Pitch shift in semitones
    pub pitch_shift: f32,
    /// Energy/volume multiplier
    pub energy: f32,
    /// Duration control
    pub duration: DurationControl,
    /// Pitch control
    pub pitch: PitchControl,
    /// Rhythm control
    pub rhythm: RhythmControl,
}

impl ProsodyConfig {
    /// Create new prosody configuration
    pub fn new() -> Self {
        Self {
            speed: 1.0,
            pitch_shift: 0.0,
            energy: 1.0,
            duration: DurationControl::default(),
            pitch: PitchControl::default(),
            rhythm: RhythmControl::default(),
        }
    }

    /// Create configuration with basic parameters
    pub fn with_basic(speed: f32, pitch_shift: f32, energy: f32) -> Self {
        let mut config = Self::new();
        config.speed = speed;
        config.pitch_shift = pitch_shift;
        config.energy = energy;
        config
    }

    /// Validate prosody configuration
    pub fn validate(&self) -> Result<()> {
        if self.speed <= 0.0 {
            return Err(AcousticError::ConfigError(
                "Speed must be > 0.0".to_string(),
            ));
        }
        if self.energy <= 0.0 {
            return Err(AcousticError::ConfigError(
                "Energy must be > 0.0".to_string(),
            ));
        }

        self.duration.validate()?;
        self.pitch.validate()?;
        self.rhythm.validate()?;

        Ok(())
    }

    /// Merge with another prosody configuration
    pub fn merge(&mut self, other: &ProsodyConfig) {
        self.speed = other.speed;
        self.pitch_shift = other.pitch_shift;
        self.energy = other.energy;
        self.duration.merge(&other.duration);
        self.pitch.merge(&other.pitch);
        self.rhythm.merge(&other.rhythm);
    }
}

impl Default for ProsodyConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Duration control parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationControl {
    /// Phoneme-level duration scaling
    pub phoneme_duration_scale: f32,
    /// Pause duration scaling
    pub pause_duration_scale: f32,
    /// Minimum phoneme duration in ms
    pub min_phoneme_duration: f32,
    /// Maximum phoneme duration in ms
    pub max_phoneme_duration: f32,
}

impl DurationControl {
    /// Create new duration control
    pub fn new() -> Self {
        Self {
            phoneme_duration_scale: 1.0,
            pause_duration_scale: 1.0,
            min_phoneme_duration: 20.0,
            max_phoneme_duration: 500.0,
        }
    }

    /// Validate duration control
    pub fn validate(&self) -> Result<()> {
        if self.phoneme_duration_scale <= 0.0 {
            return Err(AcousticError::ConfigError(
                "Phoneme duration scale must be > 0.0".to_string(),
            ));
        }
        if self.pause_duration_scale <= 0.0 {
            return Err(AcousticError::ConfigError(
                "Pause duration scale must be > 0.0".to_string(),
            ));
        }
        if self.min_phoneme_duration <= 0.0 {
            return Err(AcousticError::ConfigError(
                "Minimum phoneme duration must be > 0.0".to_string(),
            ));
        }
        if self.max_phoneme_duration <= self.min_phoneme_duration {
            return Err(AcousticError::ConfigError(
                "Maximum phoneme duration must be > minimum".to_string(),
            ));
        }
        Ok(())
    }

    /// Merge with another duration control
    pub fn merge(&mut self, other: &DurationControl) {
        self.phoneme_duration_scale = other.phoneme_duration_scale;
        self.pause_duration_scale = other.pause_duration_scale;
        self.min_phoneme_duration = other.min_phoneme_duration;
        self.max_phoneme_duration = other.max_phoneme_duration;
    }
}

impl Default for DurationControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Pitch control parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchControl {
    /// Base pitch shift in Hz
    pub base_pitch_hz: Option<f32>,
    /// Pitch range multiplier
    pub pitch_range_scale: f32,
    /// Intonation strength
    pub intonation_strength: f32,
    /// Stress emphasis
    pub stress_emphasis: f32,
}

impl PitchControl {
    /// Create new pitch control
    pub fn new() -> Self {
        Self {
            base_pitch_hz: None,
            pitch_range_scale: 1.0,
            intonation_strength: 1.0,
            stress_emphasis: 1.0,
        }
    }

    /// Validate pitch control
    pub fn validate(&self) -> Result<()> {
        if let Some(base_pitch) = self.base_pitch_hz {
            if base_pitch <= 0.0 {
                return Err(AcousticError::ConfigError(
                    "Base pitch must be > 0.0".to_string(),
                ));
            }
        }
        if self.pitch_range_scale <= 0.0 {
            return Err(AcousticError::ConfigError(
                "Pitch range scale must be > 0.0".to_string(),
            ));
        }
        if self.intonation_strength < 0.0 {
            return Err(AcousticError::ConfigError(
                "Intonation strength must be >= 0.0".to_string(),
            ));
        }
        if self.stress_emphasis < 0.0 {
            return Err(AcousticError::ConfigError(
                "Stress emphasis must be >= 0.0".to_string(),
            ));
        }
        Ok(())
    }

    /// Merge with another pitch control
    pub fn merge(&mut self, other: &PitchControl) {
        if other.base_pitch_hz.is_some() {
            self.base_pitch_hz = other.base_pitch_hz;
        }
        self.pitch_range_scale = other.pitch_range_scale;
        self.intonation_strength = other.intonation_strength;
        self.stress_emphasis = other.stress_emphasis;
    }
}

impl Default for PitchControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Rhythm control parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmControl {
    /// Rhythm pattern strength
    pub rhythm_strength: f32,
    /// Syllable timing variation
    pub syllable_timing_variation: f32,
    /// Word boundary emphasis
    pub word_boundary_emphasis: f32,
    /// Sentence boundary emphasis
    pub sentence_boundary_emphasis: f32,
}

impl RhythmControl {
    /// Create new rhythm control
    pub fn new() -> Self {
        Self {
            rhythm_strength: 1.0,
            syllable_timing_variation: 0.1,
            word_boundary_emphasis: 1.0,
            sentence_boundary_emphasis: 1.0,
        }
    }

    /// Validate rhythm control
    pub fn validate(&self) -> Result<()> {
        if self.rhythm_strength < 0.0 {
            return Err(AcousticError::ConfigError(
                "Rhythm strength must be >= 0.0".to_string(),
            ));
        }
        if self.syllable_timing_variation < 0.0 {
            return Err(AcousticError::ConfigError(
                "Syllable timing variation must be >= 0.0".to_string(),
            ));
        }
        if self.word_boundary_emphasis < 0.0 {
            return Err(AcousticError::ConfigError(
                "Word boundary emphasis must be >= 0.0".to_string(),
            ));
        }
        if self.sentence_boundary_emphasis < 0.0 {
            return Err(AcousticError::ConfigError(
                "Sentence boundary emphasis must be >= 0.0".to_string(),
            ));
        }
        Ok(())
    }

    /// Merge with another rhythm control
    pub fn merge(&mut self, other: &RhythmControl) {
        self.rhythm_strength = other.rhythm_strength;
        self.syllable_timing_variation = other.syllable_timing_variation;
        self.word_boundary_emphasis = other.word_boundary_emphasis;
        self.sentence_boundary_emphasis = other.sentence_boundary_emphasis;
    }
}

impl Default for RhythmControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Quality vs speed trade-off configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    /// Quality level
    pub quality_level: QualityLevel,
    /// Use fast inference optimizations
    pub fast_inference: bool,
    /// Precision mode
    pub precision: PrecisionMode,
    /// Maximum inference time in ms
    pub max_inference_time: Option<u32>,
}

impl QualityConfig {
    /// Create new quality configuration
    pub fn new() -> Self {
        Self {
            quality_level: QualityLevel::High,
            fast_inference: false,
            precision: PrecisionMode::FP32,
            max_inference_time: None,
        }
    }

    /// Create configuration for fast inference
    pub fn fast() -> Self {
        Self {
            quality_level: QualityLevel::Medium,
            fast_inference: true,
            precision: PrecisionMode::FP16,
            max_inference_time: Some(1000),
        }
    }

    /// Create configuration for high quality
    pub fn high_quality() -> Self {
        Self {
            quality_level: QualityLevel::VeryHigh,
            fast_inference: false,
            precision: PrecisionMode::FP32,
            max_inference_time: None,
        }
    }

    /// Validate quality configuration
    pub fn validate(&self) -> Result<()> {
        if let Some(max_time) = self.max_inference_time {
            if max_time == 0 {
                return Err(AcousticError::ConfigError(
                    "Max inference time must be > 0".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Merge with another quality configuration
    pub fn merge(&mut self, other: &QualityConfig) {
        self.quality_level = other.quality_level;
        self.fast_inference = other.fast_inference;
        self.precision = other.precision;
        if other.max_inference_time.is_some() {
            self.max_inference_time = other.max_inference_time;
        }
    }
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QualityLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

impl QualityLevel {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            QualityLevel::Low => "low",
            QualityLevel::Medium => "medium",
            QualityLevel::High => "high",
            QualityLevel::VeryHigh => "very_high",
        }
    }
}

/// Precision modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionMode {
    FP16,
    FP32,
    INT8,
}

impl PrecisionMode {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            PrecisionMode::FP16 => "fp16",
            PrecisionMode::FP32 => "fp32",
            PrecisionMode::INT8 => "int8",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_config_validation() {
        let config = SynthesisConfig::new();
        assert!(config.validate().is_ok());

        let mut config = SynthesisConfig::new();
        config.batch_size = Some(0);
        assert!(config.validate().is_err());

        config.batch_size = Some(32);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_speaker_config_validation() {
        let config = SpeakerConfig::new();
        assert!(config.validate().is_ok());

        let mut config = SpeakerConfig::new();
        let mut speaker_mix = HashMap::new();
        speaker_mix.insert(0, 0.5);
        speaker_mix.insert(1, 0.3); // Sum = 0.8, should fail
        config.speaker_mix = Some(speaker_mix);
        assert!(config.validate().is_err());

        let mut speaker_mix = HashMap::new();
        speaker_mix.insert(0, 0.6);
        speaker_mix.insert(1, 0.4); // Sum = 1.0, should pass
        config.speaker_mix = Some(speaker_mix);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_prosody_config_validation() {
        let config = ProsodyConfig::new();
        assert!(config.validate().is_ok());

        let mut config = ProsodyConfig::new();
        config.speed = 0.0;
        assert!(config.validate().is_err());

        config.speed = 1.0;
        config.energy = -1.0;
        assert!(config.validate().is_err());

        config.energy = 1.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_emotion_config_validation() {
        let config = EmotionConfig::new(EmotionType::Happy, 0.5);
        assert!(config.validate().is_ok());

        let mut config = EmotionConfig::new(EmotionType::Happy, 1.5);
        assert!(config.validate().is_err());

        config.intensity = 0.8;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_voice_quality_validation() {
        let config = VoiceQuality::new();
        assert!(config.validate().is_ok());

        let mut config = VoiceQuality::new();
        config.breathiness = 1.5;
        assert!(config.validate().is_err());

        config.breathiness = 0.5;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_duration_control_validation() {
        let config = DurationControl::new();
        assert!(config.validate().is_ok());

        let mut config = DurationControl::new();
        config.max_phoneme_duration = 10.0;
        config.min_phoneme_duration = 20.0;
        assert!(config.validate().is_err());

        config.max_phoneme_duration = 500.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_quality_config() {
        let fast_config = QualityConfig::fast();
        assert_eq!(fast_config.quality_level, QualityLevel::Medium);
        assert!(fast_config.fast_inference);
        assert_eq!(fast_config.precision, PrecisionMode::FP16);

        let high_config = QualityConfig::high_quality();
        assert_eq!(high_config.quality_level, QualityLevel::VeryHigh);
        assert!(!high_config.fast_inference);
        assert_eq!(high_config.precision, PrecisionMode::FP32);
    }

    #[test]
    fn test_enum_string_representations() {
        assert_eq!(EmotionType::Happy.as_str(), "happy");
        assert_eq!(AgeCategory::Adult.as_str(), "adult");
        assert_eq!(Gender::Female.as_str(), "female");
        assert_eq!(QualityLevel::High.as_str(), "high");
        assert_eq!(PrecisionMode::FP16.as_str(), "fp16");
    }
}
