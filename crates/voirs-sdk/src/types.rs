//! Core types for VoiRS speech synthesis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;

/// Result type alias for VoiRS operations
pub type VoirsResult<T> = std::result::Result<T, crate::VoirsError>;

/// Language code identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LanguageCode {
    /// English (US)
    EnUs,
    /// English (UK)
    EnGb,
    /// Japanese
    JaJp,
    /// Spanish (Spain)
    EsEs,
    /// Spanish (Mexico)
    EsMx,
    /// French (France)
    FrFr,
    /// German (Germany)
    DeDe,
    /// Chinese (Simplified)
    ZhCn,
    /// Portuguese (Brazil)
    PtBr,
    /// Russian
    RuRu,
    /// Italian
    ItIt,
    /// Korean
    KoKr,
    /// Dutch
    NlNl,
    /// Swedish
    SvSe,
    /// Norwegian
    NoNo,
    /// Danish
    DaDk,

    // Additional short language codes for compatibility
    /// German (short code)
    De,
    /// French (short code)
    Fr,
    /// Spanish (short code)
    Es,
    /// Italian (short code)
    It,
    /// Portuguese (short code)
    Pt,
    /// Japanese (short code)
    Ja,
    /// Korean (short code)
    Ko,
    /// Russian (short code)
    Ru,
    /// Arabic
    Ar,
    /// Hindi
    Hi,
    /// Thai
    Th,
    /// Vietnamese
    Vi,
    /// Dutch (short code)
    Nl,
    /// Swedish (short code)
    Sv,
    /// Norwegian (short code)
    No,
    /// Danish (short code)
    Da,
    /// Polish
    Pl,
    /// Czech
    Cs,
    /// Slovak
    Sk,
    /// Hungarian
    Hu,
    /// Romanian
    Ro,
    /// Bulgarian
    Bg,
    /// Croatian
    Hr,
    /// Serbian
    Sr,
    /// Slovenian
    Sl,
    /// Estonian
    Et,
    /// Latvian
    Lv,
    /// Lithuanian
    Lt,
    /// Finnish
    Fi,
    /// Greek
    El,
    /// Turkish
    Tr,
    /// Hebrew
    He,
    /// Persian/Farsi
    Fa,
    /// Urdu
    Ur,
    /// Bengali
    Bn,
    /// Tamil
    Ta,
    /// Telugu
    Te,
    /// Malayalam
    Ml,
    /// Kannada
    Kn,
    /// Gujarati
    Gu,
    /// Marathi
    Mr,
    /// Punjabi
    Pa,
    /// Odia
    Or,
    /// Assamese
    As,
}

impl LanguageCode {
    /// Get the language code as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EnUs => "en-US",
            Self::EnGb => "en-GB",
            Self::JaJp => "ja-JP",
            Self::EsEs => "es-ES",
            Self::EsMx => "es-MX",
            Self::FrFr => "fr-FR",
            Self::DeDe => "de-DE",
            Self::ZhCn => "zh-CN",
            Self::PtBr => "pt-BR",
            Self::RuRu => "ru-RU",
            Self::ItIt => "it-IT",
            Self::KoKr => "ko-KR",
            Self::NlNl => "nl-NL",
            Self::SvSe => "sv-SE",
            Self::NoNo => "no-NO",
            Self::DaDk => "da-DK",

            // Short language codes
            Self::De => "de",
            Self::Fr => "fr",
            Self::Es => "es",
            Self::It => "it",
            Self::Pt => "pt",
            Self::Ja => "ja",
            Self::Ko => "ko",
            Self::Ru => "ru",
            Self::Ar => "ar",
            Self::Hi => "hi",
            Self::Th => "th",
            Self::Vi => "vi",
            Self::Nl => "nl",
            Self::Sv => "sv",
            Self::No => "no",
            Self::Da => "da",
            Self::Pl => "pl",
            Self::Cs => "cs",
            Self::Sk => "sk",
            Self::Hu => "hu",
            Self::Ro => "ro",
            Self::Bg => "bg",
            Self::Hr => "hr",
            Self::Sr => "sr",
            Self::Sl => "sl",
            Self::Et => "et",
            Self::Lv => "lv",
            Self::Lt => "lt",
            Self::Fi => "fi",
            Self::El => "el",
            Self::Tr => "tr",
            Self::He => "he",
            Self::Fa => "fa",
            Self::Ur => "ur",
            Self::Bn => "bn",
            Self::Ta => "ta",
            Self::Te => "te",
            Self::Ml => "ml",
            Self::Kn => "kn",
            Self::Gu => "gu",
            Self::Mr => "mr",
            Self::Pa => "pa",
            Self::Or => "or",
            Self::As => "as",
        }
    }

    /// Parse language code from string
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "en-US" => Some(Self::EnUs),
            "en-GB" => Some(Self::EnGb),
            "ja-JP" => Some(Self::JaJp),
            "es-ES" => Some(Self::EsEs),
            "es-MX" => Some(Self::EsMx),
            "fr-FR" => Some(Self::FrFr),
            "de-DE" => Some(Self::DeDe),
            "zh-CN" => Some(Self::ZhCn),
            "pt-BR" => Some(Self::PtBr),
            "ru-RU" => Some(Self::RuRu),
            "it-IT" => Some(Self::ItIt),
            "ko-KR" => Some(Self::KoKr),
            "nl-NL" => Some(Self::NlNl),
            "sv-SE" => Some(Self::SvSe),
            "no-NO" => Some(Self::NoNo),
            "da-DK" => Some(Self::DaDk),

            // Short language codes
            "de" => Some(Self::De),
            "fr" => Some(Self::Fr),
            "es" => Some(Self::Es),
            "it" => Some(Self::It),
            "pt" => Some(Self::Pt),
            "ja" => Some(Self::Ja),
            "ko" => Some(Self::Ko),
            "ru" => Some(Self::Ru),
            "ar" => Some(Self::Ar),
            "hi" => Some(Self::Hi),
            "th" => Some(Self::Th),
            "vi" => Some(Self::Vi),
            "nl" => Some(Self::Nl),
            "sv" => Some(Self::Sv),
            "no" => Some(Self::No),
            "da" => Some(Self::Da),
            "pl" => Some(Self::Pl),
            "cs" => Some(Self::Cs),
            "sk" => Some(Self::Sk),
            "hu" => Some(Self::Hu),
            "ro" => Some(Self::Ro),
            "bg" => Some(Self::Bg),
            "hr" => Some(Self::Hr),
            "sr" => Some(Self::Sr),
            "sl" => Some(Self::Sl),
            "et" => Some(Self::Et),
            "lv" => Some(Self::Lv),
            "lt" => Some(Self::Lt),
            "fi" => Some(Self::Fi),
            "el" => Some(Self::El),
            "tr" => Some(Self::Tr),
            "he" => Some(Self::He),
            "fa" => Some(Self::Fa),
            "ur" => Some(Self::Ur),
            "bn" => Some(Self::Bn),
            "ta" => Some(Self::Ta),
            "te" => Some(Self::Te),
            "ml" => Some(Self::Ml),
            "kn" => Some(Self::Kn),
            "gu" => Some(Self::Gu),
            "mr" => Some(Self::Mr),
            "pa" => Some(Self::Pa),
            "or" => Some(Self::Or),
            "as" => Some(Self::As),
            _ => None,
        }
    }
}

impl std::fmt::Display for LanguageCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Phoneme representation with IPA symbol and metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phoneme {
    /// Primary symbol representation
    pub symbol: String,

    /// IPA symbol (e.g., "æ", "t̪", "d͡ʒ")
    pub ipa_symbol: String,

    /// Stress level (0=none, 1=primary, 2=secondary)
    pub stress: u8,

    /// Position within syllable
    pub syllable_position: SyllablePosition,

    /// Predicted duration in milliseconds
    pub duration_ms: Option<f32>,

    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

impl Phoneme {
    /// Create a new phoneme with symbol
    pub fn new(symbol: impl Into<String>) -> Self {
        let symbol_str = symbol.into();
        Self {
            symbol: symbol_str.clone(),
            ipa_symbol: symbol_str, // Default to same as symbol
            stress: 0,
            syllable_position: SyllablePosition::Unknown,
            duration_ms: None,
            confidence: 1.0,
        }
    }

    /// Create phoneme with stress
    pub fn with_stress(mut self, stress: u8) -> Self {
        self.stress = stress;
        self
    }

    /// Create phoneme with duration
    pub fn with_duration(mut self, duration_ms: f32) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }
}

/// Position of phoneme within syllable
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyllablePosition {
    /// Position unknown
    Unknown,
    /// Syllable onset
    Onset,
    /// Syllable nucleus (vowel)
    Nucleus,
    /// Syllable coda
    Coda,
}

/// Mel spectrogram representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelSpectrogram {
    /// Mel filterbank features [n_mels, n_frames]
    pub data: Vec<Vec<f32>>,

    /// Sample rate in Hz
    pub sample_rate: u32,

    /// Hop length in samples
    pub hop_length: u32,

    /// Number of mel channels
    pub n_mels: u32,

    /// Number of time frames
    pub n_frames: u32,
}

impl MelSpectrogram {
    /// Create new mel spectrogram
    pub fn new(data: Vec<Vec<f32>>, sample_rate: u32, hop_length: u32) -> Self {
        let n_mels = data.len() as u32;
        let n_frames = data.first().map(|row| row.len()).unwrap_or(0) as u32;

        Self {
            data,
            sample_rate,
            hop_length,
            n_mels,
            n_frames,
        }
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        (self.n_frames * self.hop_length) as f32 / self.sample_rate as f32
    }

    /// Get mel values at specific frame
    pub fn frame(&self, frame_idx: usize) -> Option<Vec<f32>> {
        if frame_idx >= self.n_frames as usize {
            return None;
        }

        Some(self.data.iter().map(|row| row[frame_idx]).collect())
    }
}

/// Audio sample representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioSample {
    /// Sample value (typically -1.0 to 1.0)
    pub value: f32,
    /// Sample index in audio stream
    pub index: usize,
}

/// Voice configuration and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// Voice identifier
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Language code
    pub language: LanguageCode,

    /// Voice characteristics
    pub characteristics: VoiceCharacteristics,

    /// Model paths and configuration
    pub model_config: ModelConfig,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Voice characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCharacteristics {
    /// Gender (if applicable)
    pub gender: Option<Gender>,

    /// Age range
    pub age: Option<AgeRange>,

    /// Speaking style
    pub style: SpeakingStyle,

    /// Emotion capability
    pub emotion_support: bool,

    /// Quality level
    pub quality: QualityLevel,
}

impl Default for VoiceCharacteristics {
    fn default() -> Self {
        Self {
            gender: None,
            age: None,
            style: SpeakingStyle::Neutral,
            emotion_support: false,
            quality: QualityLevel::Medium,
        }
    }
}

/// Gender classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Gender {
    Male,
    Female,
    NonBinary,
}

impl std::fmt::Display for Gender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Gender::Male => write!(f, "Male"),
            Gender::Female => write!(f, "Female"),
            Gender::NonBinary => write!(f, "NonBinary"),
        }
    }
}

/// Age range classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgeRange {
    Child,      // 5-12
    Teen,       // 13-19
    YoungAdult, // 20-35
    Adult,      // 36-60
    Senior,     // 60+
}

/// Speaking style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeakingStyle {
    Neutral,
    Conversational,
    News,
    Formal,
    Casual,
    Energetic,
    Calm,
    Dramatic,
    Whisper,
}

impl std::fmt::Display for SpeakingStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpeakingStyle::Neutral => write!(f, "Neutral"),
            SpeakingStyle::Conversational => write!(f, "Conversational"),
            SpeakingStyle::News => write!(f, "News"),
            SpeakingStyle::Formal => write!(f, "Formal"),
            SpeakingStyle::Casual => write!(f, "Casual"),
            SpeakingStyle::Energetic => write!(f, "Energetic"),
            SpeakingStyle::Calm => write!(f, "Calm"),
            SpeakingStyle::Dramatic => write!(f, "Dramatic"),
            SpeakingStyle::Whisper => write!(f, "Whisper"),
        }
    }
}

/// Quality level for synthesis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityLevel {
    Low,
    Medium,
    High,
    Ultra,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// G2P model path
    pub g2p_model: Option<String>,

    /// Acoustic model path
    pub acoustic_model: String,

    /// Vocoder model path
    pub vocoder_model: String,

    /// Model format (candle, onnx, etc.)
    pub format: ModelFormat,

    /// Device requirements
    pub device_requirements: DeviceRequirements,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            g2p_model: None,
            acoustic_model: "default-acoustic.safetensors".to_string(),
            vocoder_model: "default-vocoder.safetensors".to_string(),
            format: ModelFormat::Candle,
            device_requirements: DeviceRequirements::default(),
        }
    }
}

/// Model format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    Candle,
    Onnx,
    PyTorch,
    TensorFlow,
}

/// Device requirements for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceRequirements {
    /// Minimum memory in MB
    pub min_memory_mb: u32,

    /// GPU support
    pub gpu_support: bool,

    /// Supported compute capabilities
    pub compute_capabilities: Vec<String>,
}

impl Default for DeviceRequirements {
    fn default() -> Self {
        Self {
            min_memory_mb: 512,
            gpu_support: false,
            compute_capabilities: vec![],
        }
    }
}

/// Audio format specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    Wav,
    Flac,
    Mp3,
    Opus,
    Ogg,
}

impl AudioFormat {
    /// Get file extension for format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
            Self::Ogg => "ogg",
        }
    }
}

impl std::fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.extension())
    }
}

/// Audio effect types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AudioEffect {
    /// Reverb effect
    Reverb {
        room_size: f32,
        damping: f32,
        wet_level: f32,
    },
    /// Delay effect
    Delay {
        delay_time: f32,
        feedback: f32,
        wet_level: f32,
    },
    /// Equalizer effect
    Equalizer {
        low_gain: f32,
        mid_gain: f32,
        high_gain: f32,
    },
    /// Compressor effect
    Compressor {
        threshold: f32,
        ratio: f32,
        attack: f32,
        release: f32,
    },
}

/// Synthesis configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SynthesisConfig {
    /// Speaking rate multiplier (0.5 - 2.0)
    pub speaking_rate: f32,

    /// Pitch shift in semitones (-12.0 - 12.0)
    pub pitch_shift: f32,

    /// Volume gain in dB (-20.0 - 20.0)
    pub volume_gain: f32,

    /// Enable audio enhancement
    pub enable_enhancement: bool,

    /// Output audio format
    pub output_format: AudioFormat,

    /// Sample rate
    pub sample_rate: u32,

    /// Quality level
    pub quality: QualityLevel,

    /// Language for synthesis
    pub language: LanguageCode,

    /// Audio effects to apply
    pub effects: Vec<AudioEffect>,

    /// Streaming chunk size in words
    pub streaming_chunk_size: Option<usize>,

    /// Random seed for reproducible generation
    pub seed: Option<u64>,

    /// Enable emotion processing
    pub enable_emotion: bool,

    /// Emotion type to apply
    pub emotion_type: Option<String>,

    /// Emotion intensity (0.0 - 1.0)
    pub emotion_intensity: f32,

    /// Emotion preset name
    pub emotion_preset: Option<String>,

    /// Enable automatic emotion detection from text
    pub auto_emotion_detection: bool,

    // Voice cloning configuration
    /// Enable voice cloning
    pub enable_cloning: bool,
    /// Cloning method
    pub cloning_method: Option<crate::builder::features::CloningMethod>,
    /// Cloning quality level (0.0 - 1.0)
    pub cloning_quality: f32,

    // Voice conversion configuration
    /// Enable voice conversion
    pub enable_conversion: bool,
    /// Conversion target
    pub conversion_target: Option<crate::builder::features::ConversionTarget>,
    /// Enable real-time conversion
    pub realtime_conversion: bool,

    // Singing synthesis configuration
    /// Enable singing synthesis
    pub enable_singing: bool,
    /// Singing voice type
    pub singing_voice_type: Option<crate::builder::features::SingingVoiceType>,
    /// Singing technique configuration
    pub singing_technique: Option<crate::builder::features::SingingTechnique>,
    /// Musical key
    pub musical_key: Option<crate::builder::features::MusicalKey>,
    /// Tempo in BPM
    pub tempo: Option<f32>,

    // 3D spatial audio configuration
    /// Enable 3D spatial audio
    pub enable_spatial: bool,
    /// Listener position
    pub listener_position: Option<crate::builder::features::Position3D>,
    /// Enable HRTF processing
    pub hrtf_enabled: bool,
    /// Room size
    pub room_size: Option<crate::builder::features::RoomSize>,
    /// Reverb level (0.0 - 1.0)
    pub reverb_level: f32,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            speaking_rate: 1.0,
            pitch_shift: 0.0,
            volume_gain: 0.0,
            enable_enhancement: true,
            output_format: AudioFormat::Wav,
            sample_rate: 22050,
            quality: QualityLevel::High,
            language: LanguageCode::EnUs,
            effects: Vec::new(),
            streaming_chunk_size: None,
            seed: None,
            enable_emotion: false,
            emotion_type: None,
            emotion_intensity: 0.7,
            emotion_preset: None,
            auto_emotion_detection: false,

            // Voice cloning defaults
            enable_cloning: false,
            cloning_method: None,
            cloning_quality: 0.85,

            // Voice conversion defaults
            enable_conversion: false,
            conversion_target: None,
            realtime_conversion: false,

            // Singing synthesis defaults
            enable_singing: false,
            singing_voice_type: None,
            singing_technique: None,
            musical_key: None,
            tempo: None,

            // 3D spatial audio defaults
            enable_spatial: false,
            listener_position: None,
            hrtf_enabled: false,
            room_size: None,
            reverb_level: 0.3,
        }
    }
}

impl crate::config::hierarchy::ConfigHierarchy for SynthesisConfig {
    fn merge_with(&mut self, other: &Self) {
        if (other.speaking_rate - 1.0).abs() > f32::EPSILON {
            self.speaking_rate = other.speaking_rate;
        }
        if other.pitch_shift.abs() > f32::EPSILON {
            self.pitch_shift = other.pitch_shift;
        }
        if other.volume_gain.abs() > f32::EPSILON {
            self.volume_gain = other.volume_gain;
        }
        if !other.enable_enhancement {
            self.enable_enhancement = other.enable_enhancement;
        }
        if other.output_format != AudioFormat::Wav {
            self.output_format = other.output_format;
        }
        if other.sample_rate != 22050 {
            self.sample_rate = other.sample_rate;
        }
        if other.quality != QualityLevel::High {
            self.quality = other.quality;
        }
        if other.language != LanguageCode::EnUs {
            self.language = other.language;
        }
        if other.streaming_chunk_size.is_some() {
            self.streaming_chunk_size = other.streaming_chunk_size;
        }

        // Merge emotion settings
        if other.enable_emotion {
            self.enable_emotion = other.enable_emotion;
        }
        if other.emotion_type.is_some() {
            self.emotion_type = other.emotion_type.clone();
        }
        if (other.emotion_intensity - 0.7).abs() > f32::EPSILON {
            self.emotion_intensity = other.emotion_intensity;
        }
        if other.emotion_preset.is_some() {
            self.emotion_preset = other.emotion_preset.clone();
        }
        if other.auto_emotion_detection {
            self.auto_emotion_detection = other.auto_emotion_detection;
        }

        // Merge effects (append to existing)
        self.effects.extend(other.effects.clone());
    }

    fn validate(&self) -> Result<(), crate::config::hierarchy::ConfigValidationError> {
        if self.speaking_rate < 0.5 || self.speaking_rate > 2.0 {
            return Err(crate::config::hierarchy::ConfigValidationError {
                field: "speaking_rate".to_string(),
                message: "Speaking rate must be between 0.5 and 2.0".to_string(),
            });
        }

        if self.pitch_shift < -12.0 || self.pitch_shift > 12.0 {
            return Err(crate::config::hierarchy::ConfigValidationError {
                field: "pitch_shift".to_string(),
                message: "Pitch shift must be between -12.0 and 12.0 semitones".to_string(),
            });
        }

        if self.volume_gain < -20.0 || self.volume_gain > 20.0 {
            return Err(crate::config::hierarchy::ConfigValidationError {
                field: "volume_gain".to_string(),
                message: "Volume gain must be between -20.0 and 20.0 dB".to_string(),
            });
        }

        if self.sample_rate < 8000 || self.sample_rate > 96000 {
            return Err(crate::config::hierarchy::ConfigValidationError {
                field: "sample_rate".to_string(),
                message: "Sample rate must be between 8000 and 96000 Hz".to_string(),
            });
        }

        if self.emotion_intensity < 0.0 || self.emotion_intensity > 1.0 {
            return Err(crate::config::hierarchy::ConfigValidationError {
                field: "emotion_intensity".to_string(),
                message: "Emotion intensity must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(())
    }
}

/// Default implementation for AudioFormat
impl Default for AudioFormat {
    fn default() -> Self {
        AudioFormat::Wav
    }
}

/// FromStr implementation for AudioFormat
impl FromStr for AudioFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "wav" => Ok(AudioFormat::Wav),
            "flac" => Ok(AudioFormat::Flac),
            "mp3" => Ok(AudioFormat::Mp3),
            "opus" => Ok(AudioFormat::Opus),
            "ogg" => Ok(AudioFormat::Ogg),
            _ => Err(format!("Unknown audio format: {s}")),
        }
    }
}

/// Model features supported by VoiRS components
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFeature {
    /// Multi-speaker support
    MultiSpeaker,
    /// Emotion control support
    EmotionControl,
    /// Style control support
    StyleControl,
    /// Prosody control support
    ProsodyControl,
    /// Voice cloning capability
    VoiceCloning,
    /// Streaming support
    StreamingSupport,
    /// Batch processing support
    BatchProcessing,
    /// GPU acceleration support
    GPUAcceleration,
}

/// System capability detection and negotiation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemCapabilities {
    /// Available features in the current runtime
    pub available_features: Vec<AdvancedFeature>,
    /// Hardware capabilities
    pub hardware: HardwareCapabilities,
    /// Resource constraints
    pub resource_limits: ResourceLimits,
    /// Model capabilities by voice
    pub model_capabilities: HashMap<String, ModelCapabilities>,
}

/// Advanced voice features
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdvancedFeature {
    /// Emotion expression control
    EmotionControl,
    /// Voice cloning capability
    VoiceCloning,
    /// Real-time voice conversion
    VoiceConversion,
    /// Singing voice synthesis
    SingingSynthesis,
    /// 3D spatial audio processing
    SpatialAudio,
    /// Streaming synthesis
    StreamingSynthesis,
    /// GPU acceleration
    GpuAcceleration,
    /// WebAssembly compatibility
    WasmSupport,
    /// Cloud processing
    CloudProcessing,
    /// High-quality vocoding
    HighQualityVocoding,
    /// Real-time processing
    RealtimeProcessing,
}

/// Hardware capability detection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    /// Available GPU compute capability
    pub gpu_available: bool,
    /// GPU memory in MB
    pub gpu_memory_mb: Option<u64>,
    /// CPU core count
    pub cpu_cores: u32,
    /// System RAM in MB
    pub system_memory_mb: u64,
    /// Storage type (SSD/HDD)
    pub fast_storage: bool,
}

/// Resource constraints for capability negotiation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in MB
    pub max_memory_mb: u64,
    /// Maximum CPU usage percentage (0-100)
    pub max_cpu_percent: u8,
    /// Maximum latency tolerance in milliseconds
    pub max_latency_ms: u32,
    /// Battery optimization (for mobile)
    pub battery_optimization: bool,
}

/// Model-specific capabilities
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Supported advanced features
    pub supported_features: Vec<AdvancedFeature>,
    /// Required hardware features
    pub hardware_requirements: HardwareRequirements,
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
}

/// Hardware requirements for a model
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareRequirements {
    /// Minimum memory in MB
    pub min_memory_mb: u64,
    /// Minimum GPU memory in MB (if GPU required)
    pub min_gpu_memory_mb: Option<u64>,
    /// Requires GPU acceleration
    pub requires_gpu: bool,
    /// Minimum CPU cores
    pub min_cpu_cores: u32,
}

/// Performance characteristics of a model
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformanceProfile {
    /// Initialization latency in milliseconds
    pub init_latency_ms: u32,
    /// Synthesis latency per second of audio
    pub synthesis_latency_ms_per_sec: u32,
    /// Memory usage during synthesis in MB
    pub synthesis_memory_mb: u64,
    /// Quality score (0.0-1.0)
    pub quality_score: u8, // Stored as u8 (0-100) for Eq trait
}

/// Capability negotiation request
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityRequest {
    /// Desired features
    pub desired_features: Vec<AdvancedFeature>,
    /// Priority of features (matched by index with desired_features)
    pub feature_priorities: Vec<FeaturePriority>,
    /// Resource constraints
    pub constraints: ResourceLimits,
    /// Fallback strategy
    pub fallback_strategy: FallbackStrategy,
}

/// Priority level for features
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FeaturePriority {
    /// Feature is optional
    Optional,
    /// Feature is preferred but not required
    Preferred,
    /// Feature is required
    Required,
    /// Feature is critical - fail if not available
    Critical,
}

/// Fallback strategy when features are unavailable
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FallbackStrategy {
    /// Fail immediately if any required feature is unavailable
    FailFast,
    /// Degrade gracefully by disabling unavailable features
    GracefulDegradation,
    /// Use alternative implementations
    UseAlternatives,
    /// Fall back to basic functionality only
    BasicFunctionality,
}

/// Capability negotiation result
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityNegotiation {
    /// Features that will be enabled
    pub enabled_features: Vec<AdvancedFeature>,
    /// Features that were requested but unavailable
    pub unavailable_features: Vec<AdvancedFeature>,
    /// Warnings about resource constraints
    pub warnings: Vec<String>,
    /// Selected models and configurations
    pub selected_models: HashMap<String, String>,
    /// Estimated resource usage
    pub estimated_usage: ResourceUsage,
}

/// Estimated resource usage for a configuration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Memory usage in MB
    pub memory_mb: u64,
    /// Initialization time in milliseconds
    pub init_time_ms: u32,
    /// Processing latency in milliseconds
    pub processing_latency_ms: u32,
    /// CPU usage percentage (0-100)
    pub cpu_usage_percent: u8,
}

impl Default for SystemCapabilities {
    fn default() -> Self {
        Self {
            available_features: vec![
                AdvancedFeature::EmotionControl,
                AdvancedFeature::StreamingSynthesis,
                AdvancedFeature::RealtimeProcessing,
            ],
            hardware: HardwareCapabilities::default(),
            resource_limits: ResourceLimits::default(),
            model_capabilities: HashMap::new(),
        }
    }
}

impl Default for HardwareCapabilities {
    fn default() -> Self {
        Self {
            gpu_available: false,
            gpu_memory_mb: None,
            cpu_cores: num_cpus::get() as u32,
            system_memory_mb: 4096, // Conservative default
            fast_storage: true,
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 2048,
            max_cpu_percent: 80,
            max_latency_ms: 500,
            battery_optimization: false,
        }
    }
}

impl Default for CapabilityRequest {
    fn default() -> Self {
        Self {
            desired_features: vec![AdvancedFeature::StreamingSynthesis],
            feature_priorities: vec![FeaturePriority::Preferred],
            constraints: ResourceLimits::default(),
            fallback_strategy: FallbackStrategy::GracefulDegradation,
        }
    }
}

/// Default implementation for QualityLevel
impl Default for QualityLevel {
    fn default() -> Self {
        QualityLevel::High
    }
}

/// FromStr implementation for QualityLevel
impl FromStr for QualityLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" => Ok(QualityLevel::Low),
            "medium" => Ok(QualityLevel::Medium),
            "high" => Ok(QualityLevel::High),
            "ultra" => Ok(QualityLevel::Ultra),
            _ => Err(format!("Unknown quality level: {s}")),
        }
    }
}
