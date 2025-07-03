//! Core types for VoiRS speech synthesis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;

/// Language code identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
        }
    }

    /// Parse language code from string
    pub fn from_str(s: &str) -> Option<Self> {
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
    /// IPA symbol (e.g., "æ", "t̪", "d͡ʒ")
    pub symbol: String,
    
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
        Self {
            symbol: symbol.into(),
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
            _ => Err(format!("Unknown audio format: {}", s)),
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
            _ => Err(format!("Unknown quality level: {}", s)),
        }
    }
}