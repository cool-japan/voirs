//! Core traits for the `VoiRS` recognition system
//!
//! This module defines the fundamental interfaces for automatic speech recognition (ASR),
//! phoneme recognition, and audio analysis capabilities.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use std::time::Duration;
use tokio_stream::Stream;
use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme, VoirsError};

/// Result type for recognition operations
pub type RecognitionResult<T> = Result<T, VoirsError>;

/// Stream type for real-time audio processing
pub type AudioStream = Pin<Box<dyn Stream<Item = AudioBuffer> + Send>>;

/// Stream type for real-time transcript processing
pub type TranscriptStream = Pin<Box<dyn Stream<Item = RecognitionResult<TranscriptChunk>> + Send>>;

// ============================================================================
// Core Data Types
// ============================================================================

/// Represents a transcribed text with metadata
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    /// The transcribed text
    pub text: String,
    /// Detected or specified language
    pub language: LanguageCode,
    /// Overall confidence score [0.0, 1.0]
    pub confidence: f32,
    /// Word-level timestamps
    pub word_timestamps: Vec<WordTimestamp>,
    /// Sentence boundaries
    pub sentence_boundaries: Vec<SentenceBoundary>,
    /// Processing duration
    pub processing_duration: Option<Duration>,
}

/// Word-level timestamp information
#[derive(Debug, Clone, PartialEq)]
pub struct WordTimestamp {
    /// The word text
    pub word: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Confidence score for this word [0.0, 1.0]
    pub confidence: f32,
}

/// Sentence boundary information
#[derive(Debug, Clone, PartialEq)]
pub struct SentenceBoundary {
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Sentence text
    pub text: String,
    /// Confidence score [0.0, 1.0]
    pub confidence: f32,
}

/// Chunk of transcript from streaming recognition
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptChunk {
    /// Partial or complete text
    pub text: String,
    /// Whether this chunk is final
    pub is_final: bool,
    /// Start time of this chunk
    pub start_time: f32,
    /// End time of this chunk (may be provisional)
    pub end_time: f32,
    /// Confidence score [0.0, 1.0]
    pub confidence: f32,
}

/// Phoneme alignment result
#[derive(Debug, Clone, PartialEq)]
pub struct PhonemeAlignment {
    /// Aligned phonemes with timing
    pub phonemes: Vec<AlignedPhoneme>,
    /// Total duration of the audio
    pub total_duration: f32,
    /// Overall alignment confidence [0.0, 1.0]
    pub alignment_confidence: f32,
    /// Word-level alignment information
    pub word_alignments: Vec<WordAlignment>,
}

/// Individual phoneme with timing information
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedPhoneme {
    /// The phoneme
    pub phoneme: Phoneme,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Alignment confidence [0.0, 1.0]
    pub confidence: f32,
}

/// Word-level alignment information
#[derive(Debug, Clone, PartialEq)]
pub struct WordAlignment {
    /// The word text
    pub word: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Phonemes that make up this word
    pub phonemes: Vec<AlignedPhoneme>,
    /// Alignment confidence [0.0, 1.0]
    pub confidence: f32,
}

/// Comprehensive audio analysis result
#[derive(Debug, Clone, PartialEq)]
pub struct AudioAnalysis {
    /// Quality metrics (SNR, THD, etc.)
    pub quality_metrics: HashMap<String, f32>,
    /// Prosody analysis
    pub prosody: ProsodyAnalysis,
    /// Speaker characteristics
    pub speaker_characteristics: SpeakerCharacteristics,
    /// Emotional analysis
    pub emotional_analysis: EmotionalAnalysis,
    /// Processing duration
    pub processing_duration: Option<Duration>,
}

/// Prosody analysis results
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProsodyAnalysis {
    /// Pitch information
    pub pitch: PitchAnalysis,
    /// Rhythm information
    pub rhythm: RhythmAnalysis,
    /// Stress patterns
    pub stress: StressAnalysis,
    /// Intonation patterns
    pub intonation: IntonationAnalysis,
    /// Energy information
    pub energy: EnergyAnalysis,
}

/// Pitch analysis
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PitchAnalysis {
    /// Mean fundamental frequency (Hz)
    pub mean_f0: f32,
    /// Standard deviation of F0
    pub f0_std: f32,
    /// F0 range (max - min)
    pub f0_range: f32,
    /// Pitch contour over time
    pub pitch_contour: Vec<f32>,
}

/// Rhythm analysis
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RhythmAnalysis {
    /// Speaking rate (syllables per second)
    pub speaking_rate: f32,
    /// Pause duration statistics
    pub pause_statistics: PauseStatistics,
    /// Rhythm regularity score
    pub regularity_score: f32,
}

/// Pause statistics
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PauseStatistics {
    /// Total pause duration
    pub total_pause_duration: f32,
    /// Average pause duration
    pub average_pause_duration: f32,
    /// Number of pauses
    pub pause_count: usize,
    /// Pause positions
    pub pause_positions: Vec<f32>,
}

/// Stress analysis
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StressAnalysis {
    /// Stress pattern over time
    pub stress_pattern: Vec<f32>,
    /// Primary stress locations
    pub primary_stress: Vec<f32>,
    /// Secondary stress locations
    pub secondary_stress: Vec<f32>,
}

/// Intonation analysis
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IntonationAnalysis {
    /// Intonation pattern type
    pub pattern_type: IntonationPattern,
    /// Boundary tones
    pub boundary_tones: Vec<BoundaryTone>,
    /// Pitch accents
    pub pitch_accents: Vec<PitchAccent>,
}

/// Energy analysis
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EnergyAnalysis {
    /// Mean energy level
    pub mean_energy: f32,
    /// Energy standard deviation
    pub energy_std: f32,
    /// Energy range (max - min)
    pub energy_range: f32,
    /// Energy contour over time
    pub energy_contour: Vec<f32>,
}

/// Intonation pattern types
#[derive(Debug, Clone, PartialEq, Default)]
pub enum IntonationPattern {
    /// Declarative statement pattern (falling intonation)
    #[default]
    Declarative,
    /// Interrogative/question pattern (rising intonation)
    Interrogative,
    /// Exclamative pattern (dramatic intonation)
    Exclamative,
    /// Imperative/command pattern (firm intonation)
    Imperative,
    /// Mixed or unclear pattern
    Mixed,
}

/// Boundary tone information
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryTone {
    /// Time position
    pub time: f32,
    /// Tone type (rising, falling, level)
    pub tone_type: ToneType,
    /// Confidence score
    pub confidence: f32,
}

/// Pitch accent information
#[derive(Debug, Clone, PartialEq)]
pub struct PitchAccent {
    /// Time position
    pub time: f32,
    /// Accent type
    pub accent_type: AccentType,
    /// Confidence score
    pub confidence: f32,
}

/// Tone types
#[derive(Debug, Clone, PartialEq)]
pub enum ToneType {
    /// Rising tone (pitch increases)
    Rising,
    /// Falling tone (pitch decreases)
    Falling,
    /// Level tone (pitch remains stable)
    Level,
    /// Rising-falling tone (pitch rises then falls)
    RisingFalling,
    /// Falling-rising tone (pitch falls then rises)
    FallingRising,
}

/// Accent types
#[derive(Debug, Clone, PartialEq)]
pub enum AccentType {
    /// Primary accent (strongest stress)
    Primary,
    /// Secondary accent (moderate stress)
    Secondary,
    /// Tertiary accent (weakest stress)
    Tertiary,
}

/// Speaker characteristics
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpeakerCharacteristics {
    /// Estimated speaker gender
    pub gender: Option<Gender>,
    /// Estimated age range
    pub age_range: Option<AgeRange>,
    /// Voice characteristics
    pub voice_characteristics: VoiceCharacteristics,
    /// Accent information
    pub accent: Option<AccentInfo>,
}

/// Gender classification
#[derive(Debug, Clone, PartialEq)]
pub enum Gender {
    /// Male voice classification
    Male,
    /// Female voice classification
    Female,
    /// Other/non-binary voice classification
    Other,
}

/// Age range classification
#[derive(Debug, Clone, PartialEq)]
pub enum AgeRange {
    /// Child voice (0-12 years)
    Child,
    /// Teen voice (13-19 years)
    Teen,
    /// Adult voice (20-59 years)
    Adult,
    /// Senior voice (60+ years)
    Senior,
}

/// Voice characteristics
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VoiceCharacteristics {
    /// Fundamental frequency range
    pub f0_range: (f32, f32),
    /// Formant frequencies
    pub formants: Vec<f32>,
    /// Voice quality measures
    pub voice_quality: VoiceQuality,
}

/// Voice quality measures
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VoiceQuality {
    /// Jitter (pitch perturbation)
    pub jitter: f32,
    /// Shimmer (amplitude perturbation)
    pub shimmer: f32,
    /// Harmonic-to-noise ratio
    pub hnr: f32,
}

/// Accent information
#[derive(Debug, Clone, PartialEq)]
pub struct AccentInfo {
    /// Detected accent type
    pub accent_type: String,
    /// Confidence score
    pub confidence: f32,
    /// Regional indicators
    pub regional_indicators: Vec<String>,
}

/// Emotional analysis results
#[derive(Debug, Clone, PartialEq)]
pub struct EmotionalAnalysis {
    /// Primary emotion
    pub primary_emotion: Emotion,
    /// Secondary emotions with scores
    pub emotion_scores: HashMap<Emotion, f32>,
    /// Emotional intensity [0.0, 1.0]
    pub intensity: f32,
    /// Emotional valence [-1.0, 1.0] (negative to positive)
    pub valence: f32,
    /// Emotional arousal [0.0, 1.0] (calm to excited)
    pub arousal: f32,
}

/// Emotion types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Emotion {
    /// Joy/happiness emotion
    Joy,
    /// Sadness/melancholy emotion
    Sadness,
    /// Anger/irritation emotion
    Anger,
    /// Fear/anxiety emotion
    Fear,
    /// Surprise/astonishment emotion
    Surprise,
    /// Disgust/revulsion emotion
    Disgust,
    /// Neutral/calm emotion (default)
    Neutral,
    /// Contempt/disdain emotion
    Contempt,
    /// Pride/satisfaction emotion
    Pride,
    /// Shame/embarrassment emotion
    Shame,
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for ASR models
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ASRConfig {
    /// Target language (None for auto-detection)
    pub language: Option<LanguageCode>,
    /// Enable word-level timestamps
    pub word_timestamps: bool,
    /// Enable sentence segmentation
    pub sentence_segmentation: bool,
    /// Minimum confidence threshold
    pub confidence_threshold: f32,
    /// Maximum audio duration (seconds)
    pub max_duration: Option<f32>,
    /// Enable language detection
    pub language_detection: bool,
    /// Custom vocabulary
    pub custom_vocabulary: Option<Vec<String>>,
    /// Model variant (if supported)
    pub model_variant: Option<String>,
    /// Whisper model size (tiny, base, small, medium, large)
    pub whisper_model_size: Option<String>,
    /// Preferred ASR models in order of preference
    pub preferred_models: Vec<String>,
    /// Enable voice activity detection
    pub enable_voice_activity_detection: bool,
    /// Chunk duration in milliseconds for streaming
    pub chunk_duration_ms: u32,
}

impl Default for ASRConfig {
    fn default() -> Self {
        Self {
            language: None,
            word_timestamps: true,
            sentence_segmentation: true,
            confidence_threshold: 0.5,
            max_duration: Some(60.0),
            language_detection: true,
            custom_vocabulary: None,
            model_variant: None,
            whisper_model_size: Some("base".to_string()),
            preferred_models: vec!["whisper".to_string()],
            enable_voice_activity_detection: true,
            chunk_duration_ms: 30000,
        }
    }
}

/// Configuration for phoneme recognition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonemeRecognitionConfig {
    /// Target language
    pub language: LanguageCode,
    /// Enable word-level alignment
    pub word_alignment: bool,
    /// Alignment method
    pub alignment_method: AlignmentMethod,
    /// Minimum confidence threshold
    pub confidence_threshold: f32,
    /// Custom pronunciation dictionary
    pub pronunciation_dict: Option<HashMap<String, Vec<String>>>,
}

impl Default for PhonemeRecognitionConfig {
    fn default() -> Self {
        Self {
            language: LanguageCode::EnUs,
            word_alignment: true,
            alignment_method: AlignmentMethod::Forced,
            confidence_threshold: 0.3,
            pronunciation_dict: None,
        }
    }
}

/// Alignment methods for phoneme recognition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlignmentMethod {
    /// Forced alignment using acoustic models
    Forced,
    /// Automatic alignment using neural networks
    Automatic,
    /// Hybrid approach combining forced and automatic methods
    Hybrid,
}

/// Configuration for audio analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioAnalysisConfig {
    /// Enable quality metrics
    pub quality_metrics: bool,
    /// Enable prosody analysis
    pub prosody_analysis: bool,
    /// Enable speaker analysis
    pub speaker_analysis: bool,
    /// Enable emotional analysis
    pub emotional_analysis: bool,
    /// Quality metrics to compute
    pub quality_metrics_list: Vec<AudioMetric>,
    /// Frame size for analysis
    pub frame_size: usize,
    /// Hop size for analysis
    pub hop_size: usize,
}

impl Default for AudioAnalysisConfig {
    fn default() -> Self {
        Self {
            quality_metrics: true,
            prosody_analysis: true,
            speaker_analysis: true,
            emotional_analysis: true,
            quality_metrics_list: vec![
                AudioMetric::SNR,
                AudioMetric::THD,
                AudioMetric::SpectralCentroid,
                AudioMetric::SpectralRolloff,
            ],
            frame_size: 1024,
            hop_size: 512,
        }
    }
}

/// Audio quality metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AudioMetric {
    /// Signal-to-Noise Ratio
    SNR,
    /// Total Harmonic Distortion
    THD,
    /// Spectral Centroid
    SpectralCentroid,
    /// Spectral Rolloff
    SpectralRolloff,
    /// Zero Crossing Rate
    ZeroCrossingRate,
    /// Mel Frequency Cepstral Coefficients
    MelFrequencyCepstralCoefficients,
    /// Chroma Features
    ChromaFeatures,
    /// Spectral Contrast
    SpectralContrast,
    /// Tonnetz Features
    TonnetzFeatures,
    /// Root Mean Square Energy
    RootMeanSquare,
}

// ============================================================================
// Metadata Types
// ============================================================================

/// Metadata for ASR models
#[derive(Debug, Clone, PartialEq)]
pub struct ASRMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model description
    pub description: String,
    /// Supported languages
    pub supported_languages: Vec<LanguageCode>,
    /// Model architecture
    pub architecture: String,
    /// Model size in MB
    pub model_size_mb: f32,
    /// Expected inference time (relative to audio duration)
    pub inference_speed: f32,
    /// Word Error Rate benchmarks
    pub wer_benchmarks: HashMap<LanguageCode, f32>,
    /// Supported features
    pub supported_features: Vec<ASRFeature>,
}

/// ASR model features
#[derive(Debug, Clone, PartialEq)]
pub enum ASRFeature {
    /// Word-level timestamps
    WordTimestamps,
    /// Sentence boundary detection
    SentenceSegmentation,
    /// Language detection
    LanguageDetection,
    /// Noise robustness
    NoiseRobustness,
    /// Streaming inference
    StreamingInference,
    /// Custom vocabulary support
    CustomVocabulary,
    /// Speaker diarization
    SpeakerDiarization,
    /// Emotion recognition
    EmotionRecognition,
}

/// Metadata for phoneme recognizers
#[derive(Debug, Clone, PartialEq)]
pub struct PhonemeRecognizerMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model description
    pub description: String,
    /// Supported languages
    pub supported_languages: Vec<LanguageCode>,
    /// Alignment methods supported
    pub alignment_methods: Vec<AlignmentMethod>,
    /// Average alignment accuracy
    pub alignment_accuracy: f32,
    /// Supported features
    pub supported_features: Vec<PhonemeRecognitionFeature>,
}

/// Phoneme recognition features
#[derive(Debug, Clone, PartialEq)]
pub enum PhonemeRecognitionFeature {
    /// Word alignment
    WordAlignment,
    /// Custom pronunciation
    CustomPronunciation,
    /// Multi-language support
    MultiLanguage,
    /// Real-time alignment
    RealTimeAlignment,
    /// Confidence scoring
    ConfidenceScoring,
    /// Pronunciation assessment
    PronunciationAssessment,
}

/// Metadata for audio analyzers
#[derive(Debug, Clone, PartialEq)]
pub struct AudioAnalyzerMetadata {
    /// Analyzer name
    pub name: String,
    /// Analyzer version
    pub version: String,
    /// Analyzer description
    pub description: String,
    /// Supported metrics
    pub supported_metrics: Vec<AudioMetric>,
    /// Processing capabilities
    pub capabilities: Vec<AnalysisCapability>,
    /// Expected processing time
    pub processing_speed: f32,
}

/// Analysis capabilities
#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisCapability {
    /// Quality metrics
    QualityMetrics,
    /// Prosody analysis
    ProsodyAnalysis,
    /// Speaker characteristics
    SpeakerCharacteristics,
    /// Emotional analysis
    EmotionalAnalysis,
    /// Real-time analysis
    RealtimeAnalysis,
    /// Batch processing
    BatchProcessing,
    /// Streaming analysis
    StreamingAnalysis,
}

// ============================================================================
// Core Traits
// ============================================================================

/// Trait for Automatic Speech Recognition (ASR) models
#[async_trait]
pub trait ASRModel: Send + Sync {
    /// Transcribe audio to text
    async fn transcribe(
        &self,
        audio: &AudioBuffer,
        config: Option<&ASRConfig>,
    ) -> RecognitionResult<Transcript>;

    /// Stream-based transcription for real-time processing
    async fn transcribe_streaming(
        &self,
        audio_stream: AudioStream,
        config: Option<&ASRConfig>,
    ) -> RecognitionResult<TranscriptStream>;

    /// Get supported languages
    fn supported_languages(&self) -> Vec<LanguageCode>;

    /// Get model metadata
    fn metadata(&self) -> ASRMetadata;

    /// Check if a feature is supported
    fn supports_feature(&self, feature: ASRFeature) -> bool;

    /// Detect language from audio
    async fn detect_language(&self, _audio: &AudioBuffer) -> RecognitionResult<LanguageCode> {
        // Default implementation - models can override
        Err(VoirsError::ModelError {
            model_type: voirs_sdk::error::ModelType::ASR,
            message: "Language detection not implemented for this model".to_string(),
            source: None,
        })
    }
}

/// Trait for phoneme recognition and alignment
#[async_trait]
pub trait PhonemeRecognizer: Send + Sync {
    /// Recognize phonemes from audio
    async fn recognize_phonemes(
        &self,
        audio: &AudioBuffer,
        config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<Vec<Phoneme>>;

    /// Align phonemes with expected sequence
    async fn align_phonemes(
        &self,
        audio: &AudioBuffer,
        expected: &[Phoneme],
        config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<PhonemeAlignment>;

    /// Align text with audio (forced alignment)
    async fn align_text(
        &self,
        audio: &AudioBuffer,
        text: &str,
        config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<PhonemeAlignment>;

    /// Get model metadata
    fn metadata(&self) -> PhonemeRecognizerMetadata;

    /// Check if a feature is supported
    fn supports_feature(&self, feature: PhonemeRecognitionFeature) -> bool;
}

/// Trait for audio analysis
#[async_trait]
pub trait AudioAnalyzer: Send + Sync {
    /// Analyze audio for various characteristics
    async fn analyze(
        &self,
        audio: &AudioBuffer,
        config: Option<&AudioAnalysisConfig>,
    ) -> RecognitionResult<AudioAnalysis>;

    /// Analyze audio in streaming mode
    async fn analyze_streaming(
        &self,
        audio_stream: AudioStream,
        config: Option<&AudioAnalysisConfig>,
    ) -> RecognitionResult<Pin<Box<dyn Stream<Item = RecognitionResult<AudioAnalysis>> + Send>>>;

    /// Get supported metrics
    fn supported_metrics(&self) -> Vec<AudioMetric>;

    /// Get analyzer metadata
    fn metadata(&self) -> AudioAnalyzerMetadata;

    /// Check if a capability is supported
    fn supports_capability(&self, capability: AnalysisCapability) -> bool;
}

// ============================================================================
// Utility Traits
// ============================================================================

/// Trait for models that can be configured
pub trait Configurable {
    /// Configuration type for this model
    type Config;

    /// Apply configuration to the model
    fn configure(&mut self, config: &Self::Config) -> RecognitionResult<()>;

    /// Get current configuration
    fn get_config(&self) -> &Self::Config;
}

/// Trait for models that support batching
#[async_trait]
pub trait BatchProcessing<Input, Output> {
    /// Process multiple inputs in a batch
    async fn process_batch(&self, inputs: &[Input]) -> RecognitionResult<Vec<Output>>;

    /// Get optimal batch size
    fn optimal_batch_size(&self) -> usize;
}

/// Trait for resource management
pub trait ResourceManager {
    /// Load model resources
    fn load_resources(&mut self) -> RecognitionResult<()>;

    /// Unload model resources
    fn unload_resources(&mut self) -> RecognitionResult<()>;

    /// Check if resources are loaded
    fn is_loaded(&self) -> bool;

    /// Get resource usage statistics
    fn resource_usage(&self) -> ResourceUsage;
}

/// Resource usage statistics
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Memory usage in MB
    pub memory_mb: f32,
    /// CPU usage percentage
    pub cpu_percent: f32,
    /// GPU usage percentage (if applicable)
    pub gpu_percent: Option<f32>,
    /// GPU memory usage in MB (if applicable)
    pub gpu_memory_mb: Option<f32>,
}
