//! Configuration types for feature learning
//!
//! This module provides configuration structures and enums for various
//! feature learning components including audio features, speaker embeddings,
//! content embeddings, and quality prediction.

use serde::{Deserialize, Serialize};

/// Feature learning configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FeatureConfig {
    /// Audio feature extraction settings
    pub audio_features: AudioFeatureConfig,
    /// Speaker embedding settings
    pub speaker_embeddings: SpeakerEmbeddingConfig,
    /// Content embedding settings
    pub content_embeddings: ContentEmbeddingConfig,
    /// Quality prediction settings
    pub quality_prediction: QualityPredictionConfig,
}

/// Audio feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeatureConfig {
    /// Feature extraction method
    pub method: AudioFeatureMethod,
    /// Feature dimension
    pub dimension: usize,
    /// Window size for frame-based features
    pub window_size: usize,
    /// Hop size for frame-based features
    pub hop_size: usize,
    /// Enable normalization
    pub normalize: bool,
}

/// Audio feature extraction methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioFeatureMethod {
    /// Mel-frequency cepstral coefficients
    MFCC {
        /// Number of MFCC coefficients
        num_coeffs: usize,
        /// Number of mel filters
        num_filters: usize,
    },
    /// Mel spectrogram
    MelSpectrogram {
        /// Number of mel bins
        num_mels: usize,
        /// Frequency range
        freq_range: (f32, f32),
    },
    /// Raw spectrogram
    Spectrogram {
        /// FFT size
        fft_size: usize,
        /// Window function
        window: WindowFunction,
    },
    /// Learned representations (neural network)
    Learned {
        /// Model architecture
        architecture: String,
        /// Model path
        model_path: String,
        /// Input preprocessing
        preprocessing: Vec<String>,
    },
}

/// Window functions for spectral analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowFunction {
    Hann,
    Hamming,
    Blackman,
    Bartlett,
}

/// Speaker embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEmbeddingConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Embedding method
    pub method: SpeakerEmbeddingMethod,
    /// Minimum segment length for embedding
    pub min_segment_length: f32,
    /// Use speaker verification loss
    pub use_verification_loss: bool,
}

/// Speaker embedding methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpeakerEmbeddingMethod {
    /// X-vector embeddings
    XVector {
        /// Model path
        model_path: String,
        /// Use PLDA backend
        use_plda: bool,
    },
    /// Deep neural network embeddings
    DNN {
        /// Network architecture
        architecture: String,
        /// Model path
        model_path: String,
        /// Pooling method
        pooling: PoolingMethod,
    },
    /// Traditional i-vector
    IVector {
        /// UBM model path
        ubm_path: String,
        /// Total variability matrix path
        tv_matrix_path: String,
        /// Dimension
        dimension: usize,
    },
}

/// Pooling methods for embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolingMethod {
    Mean,
    Max,
    Attention,
    StatisticalPooling,
}

/// Content embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentEmbeddingConfig {
    /// Text embedding dimension
    pub text_dimension: usize,
    /// Phoneme embedding dimension
    pub phoneme_dimension: usize,
    /// Content embedding method
    pub method: ContentEmbeddingMethod,
    /// Use contextual embeddings
    pub use_contextual: bool,
}

/// Content embedding methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentEmbeddingMethod {
    /// Word2Vec embeddings
    Word2Vec {
        /// Model path
        model_path: String,
        /// Vector size
        vector_size: usize,
    },
    /// BERT-based embeddings
    BERT {
        /// Model name
        model_name: String,
        /// Use fine-tuned model
        fine_tuned: bool,
        /// Layer to extract from
        layer: i32,
    },
    /// Phoneme embeddings
    Phoneme {
        /// Embedding dimension
        dimension: usize,
        /// Use pre-trained embeddings
        pretrained: bool,
        /// Phoneme set
        phoneme_set: String,
    },
}

/// Quality prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPredictionConfig {
    /// Prediction model type
    pub model_type: QualityModelType,
    /// Features to use for prediction
    pub input_features: Vec<QualityFeature>,
    /// Target quality metrics
    pub target_metrics: Vec<QualityTarget>,
    /// Model path
    pub model_path: String,
}

/// Quality prediction model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityModelType {
    /// Random Forest
    RandomForest {
        /// Number of trees
        n_trees: usize,
        /// Maximum depth
        max_depth: Option<usize>,
    },
    /// Support Vector Machine
    SVM {
        /// Kernel type
        kernel: String,
        /// Regularization parameter
        c: f32,
    },
    /// Neural Network
    NeuralNetwork {
        /// Hidden layer sizes
        hidden_sizes: Vec<usize>,
        /// Activation function
        activation: String,
        /// Dropout rate
        dropout: f32,
    },
}

/// Quality features for prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityFeature {
    /// Spectral features
    Spectral,
    /// Temporal features
    Temporal,
    /// Perceptual features
    Perceptual,
    /// Speaker characteristics
    Speaker,
    /// Content complexity
    Content,
}

/// Quality prediction targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityTarget {
    SNR,
    Clipping,
    DynamicRange,
    SpectralQuality,
    OverallQuality,
}

impl Default for AudioFeatureConfig {
    fn default() -> Self {
        Self {
            method: AudioFeatureMethod::MFCC {
                num_coeffs: 13,
                num_filters: 26,
            },
            dimension: 13,
            window_size: 1024,
            hop_size: 512,
            normalize: true,
        }
    }
}

impl Default for SpeakerEmbeddingConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            method: SpeakerEmbeddingMethod::XVector {
                model_path: "models/xvector.bin".to_string(),
                use_plda: true,
            },
            min_segment_length: 1.0,
            use_verification_loss: true,
        }
    }
}

impl Default for ContentEmbeddingConfig {
    fn default() -> Self {
        Self {
            text_dimension: 768,
            phoneme_dimension: 128,
            method: ContentEmbeddingMethod::BERT {
                model_name: "bert-base-uncased".to_string(),
                fine_tuned: false,
                layer: -1,
            },
            use_contextual: true,
        }
    }
}

impl Default for QualityPredictionConfig {
    fn default() -> Self {
        Self {
            model_type: QualityModelType::RandomForest {
                n_trees: 100,
                max_depth: Some(10),
            },
            input_features: vec![
                QualityFeature::Spectral,
                QualityFeature::Temporal,
                QualityFeature::Perceptual,
            ],
            target_metrics: vec![QualityTarget::SNR, QualityTarget::OverallQuality],
            model_path: "models/quality_predictor.bin".to_string(),
        }
    }
}
