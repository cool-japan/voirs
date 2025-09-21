//! SSML element definitions and types.

use crate::LanguageCode;
use std::collections::HashMap;
use std::fmt;

/// SSML element types with enhanced functionality
#[derive(Debug, Clone, PartialEq)]
pub enum SsmlElement {
    /// Root speak element
    Speak {
        language: Option<LanguageCode>,
        version: Option<String>,
        content: Vec<SsmlElement>,
    },
    /// Text content
    Text(String),
    /// Phoneme override with enhanced attributes
    Phoneme {
        alphabet: String,
        ph: String,
        text: String,
        /// Additional phonetic metadata
        metadata: Option<PhonemeMetadata>,
    },
    /// Language switching with enhanced context
    Lang {
        lang: LanguageCode,
        content: Vec<SsmlElement>,
        /// Regional variant (e.g., en-US, en-GB)
        variant: Option<String>,
        /// Accent information
        accent: Option<String>,
    },
    /// Emphasis with enhanced control
    Emphasis {
        level: EmphasisLevel,
        content: Vec<SsmlElement>,
        /// Custom emphasis parameters
        custom_params: Option<EmphasisParams>,
    },
    /// Break/pause with enhanced timing
    Break {
        time: Option<String>,
        strength: Option<BreakStrength>,
        /// Custom timing parameters
        custom_timing: Option<BreakTiming>,
    },
    /// Say-as for specific pronunciation
    SayAs {
        interpret_as: InterpretAs,
        format: Option<String>,
        content: String,
        /// Language-specific format details
        detail: Option<String>,
    },
    /// Prosody control with enhanced parameters
    Prosody {
        rate: Option<String>,
        pitch: Option<String>,
        volume: Option<String>,
        content: Vec<SsmlElement>,
        /// Enhanced prosody parameters
        enhanced: Option<EnhancedProsody>,
    },
    /// Voice selection and control
    Voice {
        name: Option<String>,
        gender: Option<VoiceGender>,
        age: Option<String>,
        content: Vec<SsmlElement>,
        /// Voice characteristics
        characteristics: Option<VoiceCharacteristics>,
    },
    /// Custom pronunciation dictionary reference
    Dictionary {
        ref_name: String,
        scope: DictionaryScope,
    },
    /// Mark element for synchronization
    Mark { name: String },
    /// Paragraph element
    Paragraph {
        content: Vec<SsmlElement>,
        /// Paragraph-level prosody
        prosody: Option<ParagraphProsody>,
    },
    /// Sentence element
    Sentence {
        content: Vec<SsmlElement>,
        /// Sentence-level prosody
        prosody: Option<SentenceProsody>,
    },
}

/// Enhanced phoneme metadata
#[derive(Debug, Clone, PartialEq)]
pub struct PhonemeMetadata {
    /// Confidence level of the phoneme override
    pub confidence: f32,
    /// Duration hint in milliseconds
    pub duration_ms: Option<f32>,
    /// Stress level override
    pub stress: Option<u8>,
    /// Custom phonetic features
    pub features: Option<HashMap<String, String>>,
}

/// Enhanced emphasis parameters
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EmphasisParams {
    /// Custom emphasis strength (0.0-2.0)
    pub strength: Option<f32>,
    /// Duration multiplier
    pub duration_factor: Option<f32>,
    /// Pitch modification
    pub pitch_factor: Option<f32>,
}

/// Enhanced break timing
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BreakTiming {
    /// Exact duration in milliseconds
    pub duration_ms: Option<u32>,
    /// Silence type
    pub silence_type: Option<SilenceType>,
    /// Fade in/out parameters
    pub fade: Option<FadeParams>,
}

/// Enhanced prosody parameters
#[derive(Debug, Clone, PartialEq)]
pub struct EnhancedProsody {
    /// Detailed rate control
    pub rate_params: Option<RateParams>,
    /// Detailed pitch control
    pub pitch_params: Option<PitchParams>,
    /// Detailed volume control
    pub volume_params: Option<VolumeParams>,
    /// Contour parameters
    pub contour: Option<ProsodyContour>,
}

/// Voice characteristics
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceCharacteristics {
    /// Voice quality descriptors
    pub quality: Option<VoiceQuality>,
    /// Accent information
    pub accent: Option<RegionalAccent>,
    /// Speaker characteristics
    pub speaker: Option<SpeakerCharacteristics>,
}

/// Paragraph-level prosody
#[derive(Debug, Clone, PartialEq)]
pub struct ParagraphProsody {
    /// Overall speaking rate
    pub rate: Option<String>,
    /// Paragraph-level pitch pattern
    pub pitch_pattern: Option<PitchPattern>,
    /// Pause between sentences
    pub sentence_break: Option<String>,
}

/// Sentence-level prosody
#[derive(Debug, Clone, PartialEq)]
pub struct SentenceProsody {
    /// Sentence-level intonation
    pub intonation: Option<IntonationPattern>,
    /// Final lengthening
    pub final_lengthening: Option<f32>,
    /// Boundary tone
    pub boundary_tone: Option<BoundaryTone>,
}

/// Emphasis levels with enhanced granularity
#[derive(Debug, Clone, PartialEq)]
pub enum EmphasisLevel {
    None,
    Reduced,
    Moderate,
    Strong,
    /// Custom emphasis level (0.0-2.0)
    Custom(f32),
}

impl fmt::Display for EmphasisLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmphasisLevel::None => write!(f, "none"),
            EmphasisLevel::Reduced => write!(f, "reduced"),
            EmphasisLevel::Moderate => write!(f, "moderate"),
            EmphasisLevel::Strong => write!(f, "strong"),
            EmphasisLevel::Custom(level) => write!(f, "custom:{level}"),
        }
    }
}

/// Break strength levels with enhanced control
#[derive(Debug, Clone, PartialEq)]
pub enum BreakStrength {
    None,
    XWeak,
    Weak,
    Medium,
    Strong,
    XStrong,
    /// Custom break strength (0.0-2.0)
    Custom(f32),
}

impl fmt::Display for BreakStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BreakStrength::None => write!(f, "none"),
            BreakStrength::XWeak => write!(f, "x-weak"),
            BreakStrength::Weak => write!(f, "weak"),
            BreakStrength::Medium => write!(f, "medium"),
            BreakStrength::Strong => write!(f, "strong"),
            BreakStrength::XStrong => write!(f, "x-strong"),
            BreakStrength::Custom(strength) => write!(f, "custom:{strength}"),
        }
    }
}

/// Enhanced interpretation types
#[derive(Debug, Clone, PartialEq)]
pub enum InterpretAs {
    Characters,
    SpellOut,
    Cardinal,
    Ordinal,
    Digits,
    Fraction,
    Unit,
    Date,
    Time,
    Telephone,
    Address,
    Currency,
    Measure,
    /// Custom interpretation with format
    Custom(String),
}

impl fmt::Display for InterpretAs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterpretAs::Characters => write!(f, "characters"),
            InterpretAs::SpellOut => write!(f, "spell-out"),
            InterpretAs::Cardinal => write!(f, "cardinal"),
            InterpretAs::Ordinal => write!(f, "ordinal"),
            InterpretAs::Digits => write!(f, "digits"),
            InterpretAs::Fraction => write!(f, "fraction"),
            InterpretAs::Unit => write!(f, "unit"),
            InterpretAs::Date => write!(f, "date"),
            InterpretAs::Time => write!(f, "time"),
            InterpretAs::Telephone => write!(f, "telephone"),
            InterpretAs::Address => write!(f, "address"),
            InterpretAs::Currency => write!(f, "currency"),
            InterpretAs::Measure => write!(f, "measure"),
            InterpretAs::Custom(format) => write!(f, "custom:{format}"),
        }
    }
}

/// Voice gender specification
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceGender {
    Male,
    Female,
    Neutral,
    Child,
}

impl fmt::Display for VoiceGender {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VoiceGender::Male => write!(f, "male"),
            VoiceGender::Female => write!(f, "female"),
            VoiceGender::Neutral => write!(f, "neutral"),
            VoiceGender::Child => write!(f, "child"),
        }
    }
}

/// Dictionary scope
#[derive(Debug, Clone, PartialEq)]
pub enum DictionaryScope {
    Global,
    Local,
    Document,
    Element,
}

/// Voice quality descriptors
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceQuality {
    Clear,
    Warm,
    Bright,
    Deep,
    Soft,
    Rough,
    Breathy,
    Nasal,
    Resonant,
}

/// Regional accent information
#[derive(Debug, Clone, PartialEq)]
pub struct RegionalAccent {
    pub region: String,
    pub variant: Option<String>,
    pub strength: Option<f32>, // 0.0-1.0
}

/// Speaker characteristics
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerCharacteristics {
    pub age_group: Option<AgeGroup>,
    pub personality: Option<PersonalityTraits>,
    pub speaking_style: Option<SpeakingStyle>,
}

/// Age groups
#[derive(Debug, Clone, PartialEq)]
pub enum AgeGroup {
    Child,
    Teenager,
    YoungAdult,
    MiddleAged,
    Senior,
}

/// Personality traits
#[derive(Debug, Clone, PartialEq)]
pub struct PersonalityTraits {
    pub energy_level: Option<f32>, // 0.0-1.0
    pub formality: Option<f32>,    // 0.0-1.0
    pub warmth: Option<f32>,       // 0.0-1.0
}

/// Speaking style
#[derive(Debug, Clone, PartialEq)]
pub enum SpeakingStyle {
    Conversational,
    Formal,
    Dramatic,
    News,
    Storytelling,
    Educational,
}

/// Silence type for breaks
#[derive(Debug, Clone, PartialEq)]
pub enum SilenceType {
    Complete,
    Breath,
    Pause,
    Gap,
}

/// Fade parameters for breaks
#[derive(Debug, Clone, PartialEq)]
pub struct FadeParams {
    pub fade_in_ms: Option<u32>,
    pub fade_out_ms: Option<u32>,
}

/// Detailed rate control parameters
#[derive(Debug, Clone, PartialEq)]
pub struct RateParams {
    pub speed_factor: f32,         // 0.1-3.0
    pub acceleration: Option<f32>, // Rate change over time
    pub syllable_timing: Option<SyllableTiming>,
}

/// Detailed pitch control parameters
#[derive(Debug, Clone, PartialEq)]
pub struct PitchParams {
    pub base_frequency: Option<f32>,  // Hz
    pub range_semitones: Option<f32>, // Pitch range
    pub contour_points: Option<Vec<PitchPoint>>,
}

/// Detailed volume control parameters
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeParams {
    pub level_db: Option<f32>,
    pub dynamic_range: Option<f32>,
    pub compression: Option<CompressionParams>,
}

/// Prosody contour definition
#[derive(Debug, Clone, PartialEq)]
pub struct ProsodyContour {
    pub pitch_contour: Option<Vec<PitchPoint>>,
    pub volume_contour: Option<Vec<VolumePoint>>,
    pub rate_contour: Option<Vec<RatePoint>>,
}

/// Syllable timing control
#[derive(Debug, Clone, PartialEq)]
pub struct SyllableTiming {
    pub stressed_duration_factor: f32,
    pub unstressed_duration_factor: f32,
    pub final_lengthening: Option<f32>,
}

/// Pitch point for contour definition
#[derive(Debug, Clone, PartialEq)]
pub struct PitchPoint {
    pub position: f32, // 0.0-1.0 (relative position)
    pub frequency_hz: f32,
    pub transition: PitchTransition,
}

/// Volume point for contour definition
#[derive(Debug, Clone, PartialEq)]
pub struct VolumePoint {
    pub position: f32, // 0.0-1.0
    pub level_db: f32,
    pub transition: VolumeTransition,
}

/// Rate point for contour definition
#[derive(Debug, Clone, PartialEq)]
pub struct RatePoint {
    pub position: f32, // 0.0-1.0
    pub speed_factor: f32,
    pub transition: RateTransition,
}

/// Pitch transition types
#[derive(Debug, Clone, PartialEq)]
pub enum PitchTransition {
    Linear,
    Smooth,
    Sharp,
    Glide,
}

/// Volume transition types
#[derive(Debug, Clone, PartialEq)]
pub enum VolumeTransition {
    Linear,
    Smooth,
    Sharp,
    Fade,
}

/// Rate transition types
#[derive(Debug, Clone, PartialEq)]
pub enum RateTransition {
    Linear,
    Smooth,
    Abrupt,
    Gradual,
}

/// Compression parameters for volume control
#[derive(Debug, Clone, PartialEq)]
pub struct CompressionParams {
    pub threshold_db: f32,
    pub ratio: f32, // 1.0-20.0
    pub attack_ms: f32,
    pub release_ms: f32,
}

/// Pitch pattern for paragraphs
#[derive(Debug, Clone, PartialEq)]
pub enum PitchPattern {
    Declarative,
    Interrogative,
    Exclamatory,
    Listing,
    Parenthetical,
}

/// Intonation pattern for sentences
#[derive(Debug, Clone, PartialEq)]
pub enum IntonationPattern {
    Rising,
    Falling,
    RisingFalling,
    FallingRising,
    Flat,
    Question,
    Statement,
    Exclamation,
}

/// Boundary tone for sentence endings
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryTone {
    High,
    Mid,
    Low,
    Rising,
    Falling,
}

impl Default for PhonemeMetadata {
    fn default() -> Self {
        Self {
            confidence: 1.0,
            duration_ms: None,
            stress: None,
            features: None,
        }
    }
}

impl Default for PersonalityTraits {
    fn default() -> Self {
        Self {
            energy_level: Some(0.5),
            formality: Some(0.5),
            warmth: Some(0.5),
        }
    }
}
