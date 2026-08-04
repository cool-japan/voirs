//! Plain data types, configuration structs, and the [`FeedbackModel`]/
//! [`FeatureExtractor`] traits for [`super::DeepLearningFeedbackSystem`].
//!
//! Split out of `mod.rs` to keep individual files under the workspace's
//! ~2000-line policy; the transformer-specific types
//! ([`super::TransformerFeedbackModel`] and friends) stay in `mod.rs`
//! alongside their `impl` blocks. This module has no behavior of its own
//! beyond `Default`/constructor impls for the plain config/cache types.

use crate::traits::{FeedbackResponse, FocusArea, SessionState, UserFeedback, UserProgress};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use voirs_sdk::AudioBuffer;

use super::DeepLearningResult;

/// Configuration for deep learning models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepLearningConfig {
    /// Model directory path
    pub model_path: String,
    /// Maximum sequence length for transformers
    pub max_sequence_length: usize,
    /// Batch size for inference
    pub batch_size: usize,
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
    /// Model precision (fp16, fp32)
    pub precision: ModelPrecision,
    /// Cache configuration
    pub cache_config: CacheConfig,
    /// Model-specific configurations
    pub model_configs: HashMap<String, ModelConfig>,
    /// Feature extraction settings
    pub feature_config: FeatureExtractionConfig,
}

/// Model precision options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelPrecision {
    /// Half precision (faster, less memory)
    FP16,
    /// Full precision (slower, more accurate)
    FP32,
    /// Mixed precision
    Mixed,
}

/// Cache configuration for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable model caching
    pub enabled: bool,
    /// Maximum cache size in MB
    pub max_size_mb: usize,
    /// Cache eviction policy
    pub eviction_policy: CacheEvictionPolicy,
    /// Preload models on startup
    pub preload_models: Vec<String>,
}

/// Cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheEvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
    /// Weighted by model size and usage
    Weighted,
}

/// Configuration for specific models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model type
    pub model_type: ModelType,
    /// Model file path
    pub model_file: String,
    /// Tokenizer configuration
    pub tokenizer_config: Option<TokenizerConfig>,
    /// Model-specific parameters
    pub parameters: HashMap<String, f32>,
    /// Input/output dimensions
    pub dimensions: ModelDimensions,
    /// Quantization settings
    pub quantization: Option<QuantizationConfig>,
}

/// Types of deep learning models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    /// Transformer encoder model
    TransformerEncoder,
    /// Transformer decoder model
    TransformerDecoder,
    /// Encoder-decoder transformer
    EncoderDecoder,
    /// Convolutional neural network
    CNN,
    /// Recurrent neural network
    RNN,
    /// Generative adversarial network
    GAN,
    /// Variational autoencoder
    VAE,
    /// Custom model architecture
    Custom {
        /// Name or description of the custom architecture.
        architecture: String,
    },
}

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Vocabulary file path
    pub vocab_file: String,
    /// Special tokens
    pub special_tokens: HashMap<String, String>,
    /// Maximum token length
    pub max_token_length: usize,
    /// Tokenization strategy
    pub strategy: TokenizationStrategy,
}

/// Tokenization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenizationStrategy {
    /// Byte Pair Encoding
    BPE,
    /// `WordPiece`
    WordPiece,
    /// `SentencePiece`
    SentencePiece,
    /// Character-level
    Character,
    /// Phoneme-based
    Phoneme,
}

/// Model dimensions configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDimensions {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Number of attention heads
    pub num_heads: Option<usize>,
    /// Number of layers
    pub num_layers: Option<usize>,
}

/// Quantization configuration for model compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization method
    pub method: QuantizationMethod,
    /// Number of bits for quantization
    pub bits: usize,
    /// Quantization scope
    pub scope: QuantizationScope,
}

/// Quantization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationMethod {
    /// Post-training quantization
    PostTraining,
    /// Quantization-aware training
    QAT,
    /// Dynamic quantization
    Dynamic,
}

/// Quantization scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationScope {
    /// Quantize weights only
    WeightsOnly,
    /// Quantize activations only
    ActivationsOnly,
    /// Quantize both weights and activations
    Full,
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionConfig {
    /// Audio preprocessing settings
    pub audio_preprocessing: AudioPreprocessingConfig,
    /// Text preprocessing settings
    pub text_preprocessing: TextPreprocessingConfig,
    /// Feature types to extract
    pub feature_types: Vec<FeatureType>,
    /// Feature normalization
    pub normalization: FeatureNormalization,
}

/// Audio preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioPreprocessingConfig {
    /// Sample rate for processing
    pub target_sample_rate: usize,
    /// Window size for analysis
    pub window_size: usize,
    /// Hop length for overlapping windows
    pub hop_length: usize,
    /// Number of mel filters
    pub n_mels: usize,
    /// Frequency range
    pub freq_range: (f32, f32),
    /// Apply noise reduction
    pub noise_reduction: bool,
}

/// Text preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPreprocessingConfig {
    /// Convert to lowercase
    pub lowercase: bool,
    /// Remove punctuation
    pub remove_punctuation: bool,
    /// Normalize unicode
    pub normalize_unicode: bool,
    /// Handle contractions
    pub expand_contractions: bool,
    /// Language-specific preprocessing
    pub language_specific: HashMap<String, String>,
}

/// Types of features to extract
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureType {
    /// Mel-frequency cepstral coefficients
    MFCC,
    /// Mel-scale spectrograms
    MelSpectrogram,
    /// Raw audio waveform
    RawAudio,
    /// Fundamental frequency
    F0,
    /// Spectral centroid
    SpectralCentroid,
    /// Zero crossing rate
    ZeroCrossingRate,
    /// Chroma features
    Chroma,
    /// Prosodic features
    Prosodic,
    /// Linguistic features
    Linguistic,
    /// Contextual embeddings
    ContextualEmbeddings,
}

/// Feature normalization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureNormalization {
    /// No normalization
    None,
    /// Z-score normalization
    ZScore,
    /// Min-max normalization
    MinMax,
    /// Robust scaling
    RobustScaling,
    /// Unit vector scaling
    UnitVector,
}

/// Trait for feedback generation models
#[async_trait]
pub trait FeedbackModel {
    /// Generate feedback from input features
    async fn generate_feedback(
        &self,
        features: &FeatureBundle,
        context: &FeedbackContext,
    ) -> DeepLearningResult<FeedbackResponse>;

    /// Get model information
    fn model_info(&self) -> ModelInfo;

    /// Check if model is loaded
    fn is_loaded(&self) -> bool;

    /// Load model from file
    async fn load(&mut self, model_path: &Path) -> DeepLearningResult<()>;

    /// Unload model to free memory
    async fn unload(&mut self) -> DeepLearningResult<()>;
}

/// Trait for feature extraction
#[async_trait]
pub trait FeatureExtractor {
    /// Extract features from audio
    async fn extract_audio_features(
        &self,
        audio: &AudioBuffer,
        config: &AudioPreprocessingConfig,
    ) -> DeepLearningResult<AudioFeatures>;

    /// Extract features from text
    async fn extract_text_features(
        &self,
        text: &str,
        config: &TextPreprocessingConfig,
    ) -> DeepLearningResult<TextFeatures>;

    /// Get supported feature types
    fn supported_features(&self) -> Vec<FeatureType>;
}

/// Bundle of extracted features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureBundle {
    /// Audio features
    pub audio_features: AudioFeatures,
    /// Text features
    pub text_features: TextFeatures,
    /// Contextual features
    pub contextual_features: ContextualFeatures,
    /// Temporal features
    pub temporal_features: TemporalFeatures,
}

/// Audio-specific features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeatures {
    /// MFCC coefficients
    pub mfcc: Option<Vec<Vec<f32>>>,
    /// Mel-scale spectrogram
    pub mel_spectrogram: Option<Vec<Vec<f32>>>,
    /// Raw audio samples
    pub raw_audio: Option<Vec<f32>>,
    /// Fundamental frequency
    pub f0: Option<Vec<f32>>,
    /// Spectral features
    pub spectral_features: SpectralFeatures,
    /// Prosodic features
    pub prosodic_features: ProsodicFeatures,
}

/// Spectral feature components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralFeatures {
    /// Spectral centroid
    pub centroid: Option<Vec<f32>>,
    /// Spectral rolloff
    pub rolloff: Option<Vec<f32>>,
    /// Spectral flux
    pub flux: Option<Vec<f32>>,
    /// Zero crossing rate
    pub zcr: Option<Vec<f32>>,
    /// Chroma features
    pub chroma: Option<Vec<Vec<f32>>>,
}

/// Prosodic feature components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicFeatures {
    /// Pitch contour
    pub pitch: Option<Vec<f32>>,
    /// Energy contour
    pub energy: Option<Vec<f32>>,
    /// Duration features
    pub duration: Option<Vec<f32>>,
    /// Rhythm features
    pub rhythm: Option<RhythmFeatures>,
}

/// Rhythm-specific features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmFeatures {
    /// Beat tracking
    pub beats: Option<Vec<f32>>,
    /// Tempo estimation
    pub tempo: Option<f32>,
    /// Rhythmic patterns
    pub patterns: Option<Vec<f32>>,
}

/// Text-specific features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFeatures {
    /// Token embeddings
    pub token_embeddings: Option<Vec<Vec<f32>>>,
    /// Sentence embeddings
    pub sentence_embeddings: Option<Vec<f32>>,
    /// Linguistic features
    pub linguistic_features: LinguisticFeatures,
    /// Semantic features
    pub semantic_features: SemanticFeatures,
}

/// Linguistic feature components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticFeatures {
    /// Part-of-speech tags
    pub pos_tags: Option<Vec<String>>,
    /// Named entity recognition
    pub ner_tags: Option<Vec<String>>,
    /// Phoneme sequences
    pub phonemes: Option<Vec<String>>,
    /// Syllable structure
    pub syllables: Option<Vec<String>>,
    /// Stress patterns
    pub stress_patterns: Option<Vec<usize>>,
}

/// Semantic feature components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFeatures {
    /// Word sense disambiguation
    pub word_senses: Option<HashMap<String, String>>,
    /// Sentiment scores
    pub sentiment: Option<SentimentScores>,
    /// Topic modeling
    pub topics: Option<Vec<(String, f32)>>,
    /// Contextual relationships
    pub relationships: Option<Vec<(String, String, f32)>>,
}

/// Sentiment analysis scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentScores {
    /// Overall sentiment
    pub overall: f32,
    /// Positive sentiment
    pub positive: f32,
    /// Negative sentiment
    pub negative: f32,
    /// Neutral sentiment
    pub neutral: f32,
    /// Emotional valence
    pub emotional_valence: HashMap<String, f32>,
}

/// Contextual features from user/session data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualFeatures {
    /// User skill level
    pub skill_level: f32,
    /// Session progress
    pub session_progress: f32,
    /// Recent performance
    pub recent_performance: Vec<f32>,
    /// Focus areas
    pub focus_areas: Vec<FocusArea>,
    /// Difficulty level
    pub difficulty_level: f32,
    /// User preferences
    pub preferences: HashMap<String, String>,
}

/// Temporal features for sequence modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalFeatures {
    /// Time since session start
    pub session_time: f32,
    /// Time since last feedback
    pub last_feedback_time: f32,
    /// Historical patterns
    pub historical_patterns: Vec<f32>,
    /// Trend indicators
    pub trend_indicators: HashMap<String, f32>,
}

/// Context for feedback generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackContext {
    /// Current user progress
    pub user_progress: UserProgress,
    /// Session state
    pub session_state: SessionState,
    /// Target text
    pub target_text: String,
    /// Previous feedback
    pub previous_feedback: Vec<UserFeedback>,
    /// Feedback preferences
    pub preferences: FeedbackPreferences,
}

/// User preferences for feedback generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackPreferences {
    /// Preferred feedback style
    pub style: FeedbackStyle,
    /// Verbosity level
    pub verbosity: VerbosityLevel,
    /// Focus areas of interest
    pub focus_areas: Vec<FocusArea>,
    /// Language preferences
    pub language: String,
    /// Personalization level
    pub personalization: PersonalizationLevel,
}

/// Feedback generation styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackStyle {
    /// Encouraging and supportive
    Encouraging,
    /// Direct and technical
    Technical,
    /// Balanced approach
    Balanced,
    /// Gamified and fun
    Gamified,
    /// Professional coaching style
    Professional,
}

/// Verbosity levels for feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerbosityLevel {
    /// Minimal feedback
    Minimal,
    /// Concise feedback
    Concise,
    /// Detailed feedback
    Detailed,
    /// Comprehensive feedback
    Comprehensive,
}

/// Personalization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersonalizationLevel {
    /// Generic feedback
    Generic,
    /// Basic personalization
    Basic,
    /// Advanced personalization
    Advanced,
    /// Highly personalized
    HighlyPersonalized,
}

/// Model information and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model architecture
    pub architecture: String,
    /// Training dataset info
    pub training_data: String,
    /// Model size in MB
    pub size_mb: usize,
    /// Supported languages
    pub supported_languages: Vec<String>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f32>,
}

/// Model cache for performance optimization
pub struct ModelCache {
    /// Cached models
    pub(crate) cached_models: HashMap<String, CachedModel>,
    /// Current cache size in MB
    pub(crate) current_size_mb: usize,
    /// Maximum cache size in MB
    pub(crate) max_size_mb: usize,
    /// Access history for LRU eviction
    #[allow(dead_code)]
    pub(crate) access_history: Vec<String>,
}

/// Cached model entry
pub struct CachedModel {
    /// Model instance
    model: Box<dyn FeedbackModel + Send + Sync>,
    /// Model size in MB
    size_mb: usize,
    /// Last access time
    last_accessed: chrono::DateTime<chrono::Utc>,
    /// Access count
    access_count: usize,
}

/// Inference statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStatistics {
    /// Total inferences performed
    pub total_inferences: usize,
    /// Average inference time (ms)
    pub avg_inference_time_ms: f32,
    /// Model usage counts
    pub model_usage: HashMap<String, usize>,
    /// Error counts by type
    pub error_counts: HashMap<String, usize>,
    /// Performance trends
    pub performance_trends: Vec<PerformanceSnapshot>,
}

/// Performance snapshot for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Inference time
    pub inference_time_ms: f32,
    /// Memory usage
    pub memory_usage_mb: f32,
    /// Model name
    pub model_name: String,
    /// Success rate
    pub success_rate: f32,
}
impl ModelCache {
    /// Description
    #[must_use]
    pub fn new(max_size_mb: usize) -> Self {
        Self {
            cached_models: HashMap::new(),
            current_size_mb: 0,
            max_size_mb,
            access_history: Vec::new(),
        }
    }
}

impl Default for InferenceStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceStatistics {
    /// Description
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_inferences: 0,
            avg_inference_time_ms: 0.0,
            model_usage: HashMap::new(),
            error_counts: HashMap::new(),
            performance_trends: Vec::new(),
        }
    }
}
// Default implementations
impl Default for DeepLearningConfig {
    fn default() -> Self {
        Self {
            model_path: "./models".to_string(),
            max_sequence_length: 512,
            batch_size: 1,
            use_gpu: false,
            precision: ModelPrecision::FP32,
            cache_config: CacheConfig::default(),
            model_configs: HashMap::new(),
            feature_config: FeatureExtractionConfig::default(),
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size_mb: 1024,
            eviction_policy: CacheEvictionPolicy::LRU,
            preload_models: Vec::new(),
        }
    }
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            audio_preprocessing: AudioPreprocessingConfig::default(),
            text_preprocessing: TextPreprocessingConfig::default(),
            feature_types: vec![
                FeatureType::MFCC,
                FeatureType::MelSpectrogram,
                FeatureType::F0,
                FeatureType::SpectralCentroid,
            ],
            normalization: FeatureNormalization::ZScore,
        }
    }
}

impl Default for AudioPreprocessingConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 16000,
            window_size: 1024,
            hop_length: 512,
            n_mels: 80,
            freq_range: (0.0, 8000.0),
            noise_reduction: true,
        }
    }
}

impl Default for TextPreprocessingConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: false,
            normalize_unicode: true,
            expand_contractions: true,
            language_specific: HashMap::new(),
        }
    }
}

impl Default for FeedbackPreferences {
    fn default() -> Self {
        Self {
            style: FeedbackStyle::Balanced,
            verbosity: VerbosityLevel::Detailed,
            focus_areas: vec![FocusArea::Pronunciation],
            language: "en".to_string(),
            personalization: PersonalizationLevel::Advanced,
        }
    }
}
