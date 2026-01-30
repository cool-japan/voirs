//! Type definitions for speaker embedding system

use candle_core::Device;
use candle_nn::{Conv2d, Linear};
use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Speaker embedding vector with metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerEmbedding {
    /// Embedding vector
    pub vector: Vec<f32>,
    /// Dimension of the embedding
    pub dimension: usize,
    /// Confidence score of the embedding (0.0-1.0)
    pub confidence: f32,
    /// Speaker characteristics metadata
    pub metadata: EmbeddingMetadata,
}

/// Metadata associated with speaker embedding
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    /// Gender classification (if available)
    pub gender: Option<String>,
    /// Age estimation (if available)
    pub age_estimate: Option<f32>,
    /// Language/accent information
    pub language: Option<String>,
    /// Emotional characteristics
    pub emotion: Option<String>,
    /// Voice quality indicators
    pub voice_quality: VoiceQuality,
    /// Extraction timestamp
    pub extraction_time: Option<f64>,
}

/// Voice quality characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceQuality {
    /// Fundamental frequency statistics
    pub f0_mean: f32,
    pub f0_std: f32,
    /// Spectral characteristics
    pub spectral_centroid: f32,
    pub spectral_bandwidth: f32,
    /// Voice quality metrics
    pub jitter: f32,
    pub shimmer: f32,
    /// Energy characteristics
    pub energy_mean: f32,
    pub energy_std: f32,
}

/// Advanced speaker embedding extractor with neural networks (thread-safe)
pub struct SpeakerEmbeddingExtractor {
    /// Model configuration (immutable, can be shared)
    pub(super) config: Arc<EmbeddingConfig>,
    /// Neural network device
    pub(super) device: Device,
    /// Embedding network (thread-safe)
    pub(super) embedding_network: Arc<RwLock<Option<EmbeddingNetwork>>>,
    /// Feature extractor (thread-safe)
    pub(super) feature_extractor: Arc<RwLock<FeatureExtractor>>,
    /// Normalization statistics (thread-safe)
    pub(super) normalization_stats: Arc<RwLock<Option<NormalizationStats>>>,
}

impl std::fmt::Debug for SpeakerEmbeddingExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeakerEmbeddingExtractor")
            .field("config", &self.config)
            .field("device", &"<Device>")
            .field(
                "embedding_network",
                &"<Arc<RwLock<Option<EmbeddingNetwork>>>>",
            )
            .field("feature_extractor", &"<Arc<RwLock<FeatureExtractor>>>")
            .field(
                "normalization_stats",
                &"<Arc<RwLock<Option<NormalizationStats>>>>",
            )
            .finish()
    }
}

/// Configuration for embedding extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Window size for processing (samples)
    pub window_size: usize,
    /// Hop size for overlapping windows
    pub hop_size: usize,
    /// FFT size for spectral analysis
    pub fft_size: usize,
    /// Number of mel filters
    pub num_mel_filters: usize,
    /// Neural network architecture
    pub network_architecture: NetworkArchitecture,
    /// Feature extraction method
    pub feature_method: FeatureExtractionMethod,
    /// Preprocessing options
    pub preprocessing: PreprocessingConfig,
    /// Batch processing size
    pub batch_size: usize,
}

/// Neural network architecture options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkArchitecture {
    /// ResNet-based architecture
    ResNet { layers: Vec<usize> },
    /// Transformer-based architecture
    Transformer {
        num_layers: usize,
        num_heads: usize,
        hidden_dim: usize,
    },
    /// CNN-based architecture
    CNN {
        conv_layers: Vec<(usize, usize)>, // (channels, kernel_size)
        fc_layers: Vec<usize>,
    },
    /// TDNN (Time Delay Neural Network)
    TDNN { layers: Vec<TDNNLayer> },
}

/// TDNN layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDNNLayer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub context: Vec<i32>, // Time context offsets
}

/// Feature extraction methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureExtractionMethod {
    /// Mel-frequency cepstral coefficients
    MFCC,
    /// Mel-scale spectrogram
    MelSpectrogram,
    /// Raw spectrogram
    Spectrogram,
    /// Log mel-scale spectrogram
    LogMel,
    /// Perceptual Linear Prediction coefficients
    PLP,
    /// Filter bank features
    FilterBank,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Apply voice activity detection
    pub vad_enabled: bool,
    /// Noise reduction enabled
    pub noise_reduction: bool,
    /// Normalization method
    pub normalization: NormalizationMethod,
    /// Augmentation options
    pub augmentation: Option<AugmentationConfig>,
}

/// Normalization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Z-score normalization
    ZScore,
    /// Min-max normalization
    MinMax,
    /// Cepstral mean normalization
    CMN,
    /// Cepstral mean and variance normalization
    CMVN,
}

/// Data augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    /// Add noise (SNR range)
    pub noise_snr_range: Option<(f32, f32)>,
    /// Speed perturbation range
    pub speed_range: Option<(f32, f32)>,
    /// Volume perturbation range
    pub volume_range: Option<(f32, f32)>,
    /// Spectral augmentation
    pub spec_augment: bool,
}

/// Normalization statistics
#[derive(Debug, Clone)]
pub struct NormalizationStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub min: Vec<f32>,
    pub max: Vec<f32>,
}

/// Online learning configuration for adaptive embedding updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineLearningConfig {
    /// Initial learning rate for adaptation
    pub initial_learning_rate: f32,
    /// Learning rate decay factor per step
    pub decay_factor: f32,
    /// Minimum learning rate
    pub min_learning_rate: f32,
    /// Convergence threshold for similarity
    pub convergence_threshold: f32,
    /// Maximum number of adaptation steps
    pub max_steps: usize,
}

/// Configuration for streaming embedding extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Duration of each processing window (seconds)
    pub window_duration: f32,
    /// Hop duration between windows (seconds)
    pub hop_duration: f32,
    /// Enable temporal smoothing across windows
    pub temporal_smoothing: bool,
    /// Smoothing factor for temporal averaging
    pub smoothing_factor: f32,
    /// Method for aggregating window embeddings
    pub aggregation_method: AggregationMethod,
    /// Minimum confidence threshold for window inclusion
    pub min_confidence_threshold: f32,
}

/// Configuration for adaptive refinement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementConfig {
    /// Base refinement weight
    pub base_refinement_weight: f32,
    /// Quality amplification factor
    pub quality_amplification: f32,
    /// Maximum refinement weight
    pub max_refinement_weight: f32,
    /// Convergence threshold for refinement
    pub convergence_threshold: f32,
    /// Window size for convergence checking
    pub convergence_window: usize,
}

/// Aggregation methods for streaming embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Simple averaging
    Average,
    /// Confidence-weighted averaging
    Weighted,
    /// Quality-weighted averaging
    QualityWeighted,
    /// Temporal-weighted averaging (recent samples have higher weight)
    TemporalWeighted,
}

/// Metrics for adaptation steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationMetrics {
    pub step: usize,
    pub learning_rate: f32,
    pub similarity_to_base: f32,
    pub similarity_to_previous: f32,
    pub confidence_change: f32,
    pub adaptation_time: std::time::Duration,
    pub quality_score: f32,
}

/// Result of streaming embedding extraction
#[derive(Debug, Clone)]
pub struct StreamingEmbeddingResult {
    /// Aggregated embedding from all windows
    pub aggregated_embedding: SpeakerEmbedding,
    /// Individual window embeddings
    pub window_embeddings: Vec<SpeakerEmbedding>,
    /// Streaming statistics
    pub streaming_stats: StreamingStats,
}

/// Statistics for streaming processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingStats {
    pub num_windows: usize,
    pub avg_confidence: f32,
    pub confidence_std: f32,
    pub total_duration: f32,
    pub processing_time: std::time::Duration,
}

/// Result of adaptive refinement
#[derive(Debug, Clone)]
pub struct RefinementResult {
    /// Final refined embedding
    pub refined_embedding: SpeakerEmbedding,
    /// History of refinement iterations
    pub refinement_history: Vec<RefinementIteration>,
    /// Total similarity improvement
    pub total_similarity_improvement: f32,
    /// Total confidence change
    pub total_confidence_change: f32,
    /// Whether convergence was achieved
    pub convergence_achieved: bool,
}

/// Metrics for individual refinement iterations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementIteration {
    pub iteration: usize,
    pub sample_index: usize,
    pub quality_feedback: f32,
    pub adaptive_weight: f32,
    pub similarity_before: f32,
    pub similarity_after: f32,
    pub confidence_change: f32,
    pub iteration_time: std::time::Duration,
}

/// Embedding stability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingStability {
    /// Consistency of embeddings over time
    pub temporal_consistency: f32,
    /// Stability of confidence scores
    pub confidence_stability: f32,
    /// Rate of embedding drift
    pub drift_rate: f32,
    /// Overall stability score
    pub stability_score: f32,
}

/// Speaker identification match result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerMatch {
    /// Speaker identifier
    pub speaker_id: String,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f32,
    /// Confidence of the match
    pub confidence: f32,
    /// Distance metric (lower is better)
    pub distance: f32,
}

/// Feature extractor for audio preprocessing
pub struct FeatureExtractor {
    pub(super) config: EmbeddingConfig,
    pub(super) mel_filterbank: Option<Array2<f32>>,
    pub(super) dct_matrix: Option<Array2<f32>>,
}

/// Neural network for embedding extraction
#[derive(Debug)]
pub(super) struct EmbeddingNetwork {
    pub layers: Vec<Linear>,
    pub conv_layers: Vec<Conv2d>,
    pub device: Device,
    pub architecture: NetworkArchitecture,
}
