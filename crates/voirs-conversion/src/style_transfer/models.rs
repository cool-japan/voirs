//! Style model structures and metadata

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use super::characteristics::StyleCharacteristics;

/// Style model for style transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleModel {
    /// Model identifier
    pub id: String,

    /// Model name
    pub name: String,

    /// Style characteristics
    pub style_characteristics: StyleCharacteristics,

    /// Model parameters
    pub parameters: StyleModelParameters,

    /// Training information
    pub training_info: StyleTrainingInfo,

    /// Quality metrics
    pub quality_metrics: StyleModelQualityMetrics,

    /// Creation timestamp
    #[serde(skip)]
    pub created: Option<Instant>,

    /// Last updated timestamp
    #[serde(skip)]
    pub last_updated: Option<Instant>,
}

/// Style model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleModelParameters {
    /// Encoder parameters
    pub encoder_params: EncoderParameters,

    /// Decoder parameters
    pub decoder_params: DecoderParameters,

    /// Discriminator parameters
    pub discriminator_params: Option<DiscriminatorParameters>,

    /// Model architecture
    pub architecture: ModelArchitecture,
}

/// Encoder parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderParameters {
    /// Input dimension
    pub input_dim: usize,

    /// Hidden dimensions
    pub hidden_dims: Vec<usize>,

    /// Output dimension
    pub output_dim: usize,

    /// Layer types
    pub layer_types: Vec<LayerType>,

    /// Activation functions
    pub activations: Vec<ActivationType>,
}

/// Decoder parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderParameters {
    /// Input dimension
    pub input_dim: usize,

    /// Hidden dimensions
    pub hidden_dims: Vec<usize>,

    /// Output dimension
    pub output_dim: usize,

    /// Layer types
    pub layer_types: Vec<LayerType>,

    /// Activation functions
    pub activations: Vec<ActivationType>,
}

/// Discriminator parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscriminatorParameters {
    /// Input dimension
    pub input_dim: usize,

    /// Hidden dimensions
    pub hidden_dims: Vec<usize>,

    /// Number of classes
    pub num_classes: usize,

    /// Layer types
    pub layer_types: Vec<LayerType>,
}

/// Layer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// Linear layer
    Linear,

    /// Convolutional layer
    Convolutional,

    /// LSTM layer
    LSTM,

    /// GRU layer
    GRU,

    /// Transformer layer
    Transformer,

    /// Attention layer
    Attention,
}

/// Activation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    /// ReLU activation
    ReLU,

    /// Leaky ReLU activation
    LeakyReLU,

    /// Tanh activation
    Tanh,

    /// Sigmoid activation
    Sigmoid,

    /// GELU activation
    GELU,

    /// Swish activation
    Swish,
}

/// Model architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    /// Architecture name
    pub name: String,

    /// Architecture type
    pub architecture_type: ArchitectureType,

    /// Model components
    pub components: Vec<ModelComponent>,

    /// Connection patterns
    pub connections: Vec<ConnectionPattern>,
}

/// Architecture type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchitectureType {
    /// Autoencoder architecture
    Autoencoder,

    /// GAN architecture
    GAN,

    /// VAE architecture
    VAE,

    /// Transformer architecture
    Transformer,

    /// Diffusion model architecture
    Diffusion,
}

/// Model component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComponent {
    /// Component name
    pub name: String,

    /// Component type
    pub component_type: ComponentType,

    /// Input shapes
    pub input_shapes: Vec<Vec<usize>>,

    /// Output shapes
    pub output_shapes: Vec<Vec<usize>>,
}

/// Component type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentType {
    /// Encoder component
    Encoder,

    /// Decoder component
    Decoder,

    /// Discriminator component
    Discriminator,

    /// Generator component
    Generator,

    /// Attention component
    Attention,
}

/// Connection pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPattern {
    /// Source component
    pub source: String,

    /// Target component
    pub target: String,

    /// Connection type
    pub connection_type: ConnectionType,

    /// Connection weight
    pub weight: f32,
}

/// Connection type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Direct connection
    Direct,

    /// Residual connection
    Residual,

    /// Skip connection
    Skip,

    /// Attention connection
    Attention,
}

/// Style training information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleTrainingInfo {
    /// Training dataset information
    pub dataset_info: DatasetInfo,

    /// Training hyperparameters
    pub hyperparameters: TrainingHyperparameters,

    /// Training metrics
    pub training_metrics: TrainingMetrics,

    /// Validation metrics
    pub validation_metrics: ValidationMetrics,
}

/// Dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Dataset name
    pub name: String,

    /// Dataset size
    pub size: usize,

    /// Number of speakers
    pub num_speakers: usize,

    /// Total duration (hours)
    pub total_duration: f32,

    /// Languages
    pub languages: Vec<String>,

    /// Speaking styles
    pub speaking_styles: Vec<String>,
}

/// Training hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHyperparameters {
    /// Learning rate
    pub learning_rate: f32,

    /// Batch size
    pub batch_size: usize,

    /// Number of epochs
    pub num_epochs: usize,

    /// Optimizer type
    pub optimizer: OptimizerType,

    /// Loss function weights
    pub loss_weights: HashMap<String, f32>,

    /// Regularization parameters
    pub regularization: RegularizationParameters,
}

/// Optimizer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Adam optimizer
    Adam,

    /// AdamW optimizer
    AdamW,

    /// SGD optimizer
    SGD,

    /// RMSprop optimizer
    RMSprop,

    /// AdaGrad optimizer
    AdaGrad,
}

/// Regularization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationParameters {
    /// L1 regularization weight
    pub l1_weight: f32,

    /// L2 regularization weight
    pub l2_weight: f32,

    /// Dropout rate
    pub dropout_rate: f32,

    /// Batch normalization
    pub batch_norm: bool,

    /// Layer normalization
    pub layer_norm: bool,
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss history
    pub loss_history: Vec<f32>,

    /// Training accuracy history
    pub accuracy_history: Vec<f32>,

    /// Training time per epoch
    pub time_per_epoch: Vec<f32>,

    /// Convergence information
    pub convergence_info: ConvergenceInfo,
}

/// Validation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    /// Validation loss history
    pub loss_history: Vec<f32>,

    /// Validation accuracy history
    pub accuracy_history: Vec<f32>,

    /// Best validation score
    pub best_score: f32,

    /// Early stopping information
    pub early_stopping: EarlyStoppingInfo,
}

/// Convergence information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceInfo {
    /// Converged flag
    pub converged: bool,

    /// Convergence epoch
    pub convergence_epoch: Option<usize>,

    /// Convergence criteria
    pub criteria: ConvergenceCriteria,
}

/// Convergence criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceCriteria {
    /// Loss tolerance
    pub loss_tolerance: f32,

    /// Patience epochs
    pub patience: usize,

    /// Minimum improvement
    pub min_improvement: f32,
}

/// Early stopping information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingInfo {
    /// Early stopped flag
    pub early_stopped: bool,

    /// Stopping epoch
    pub stopping_epoch: Option<usize>,

    /// Stopping reason
    pub stopping_reason: Option<String>,
}

/// Style model quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleModelQualityMetrics {
    /// Overall quality score
    pub overall_quality: f32,

    /// Style transfer accuracy
    pub transfer_accuracy: f32,

    /// Content preservation score
    pub content_preservation: f32,

    /// Style consistency score
    pub style_consistency: f32,

    /// Perceptual quality scores
    pub perceptual_scores: PerceptualQualityScores,

    /// Objective quality metrics
    pub objective_metrics: ObjectiveQualityMetrics,
}

/// Perceptual quality scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualQualityScores {
    /// Naturalness score
    pub naturalness: f32,

    /// Similarity to target style
    pub style_similarity: f32,

    /// Intelligibility score
    pub intelligibility: f32,

    /// Overall preference score
    pub preference: f32,

    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f32, f32)>,
}

/// Objective quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveQualityMetrics {
    /// Mel-cepstral distortion
    pub mcd: f32,

    /// Fundamental frequency RMSE
    pub f0_rmse: f32,

    /// Voicing decision error
    pub voicing_error: f32,

    /// Spectral distortion
    pub spectral_distortion: f32,

    /// Prosodic feature correlation
    pub prosodic_correlation: f32,
}

/// Style model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleModelMetadata {
    /// Model creation date
    #[serde(skip)]
    pub created: Option<Instant>,

    /// Model version
    pub version: String,

    /// Model author
    pub author: String,

    /// Model description
    pub description: String,

    /// Model tags
    pub tags: Vec<String>,

    /// Model license
    pub license: String,

    /// Model file size
    pub file_size: u64,

    /// Model checksum
    pub checksum: String,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceMetrics {
    /// Inference time (ms)
    pub inference_time: f32,

    /// Memory usage (MB)
    pub memory_usage: f32,

    /// GPU utilization (%)
    pub gpu_utilization: f32,

    /// Throughput (samples/second)
    pub throughput: f32,

    /// Real-time factor
    pub real_time_factor: f32,
}

/// Model usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUsageStatistics {
    /// Number of times used
    pub usage_count: u64,

    /// Average quality rating
    pub avg_quality_rating: f32,

    /// Success rate
    pub success_rate: f32,

    /// Last used timestamp
    #[serde(skip)]
    pub last_used: Option<Instant>,

    /// Usage contexts
    pub usage_contexts: HashMap<String, u32>,
}

/// Repository configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryConfig {
    /// Maximum number of models
    pub max_models: usize,

    /// Cache size limit (MB)
    pub cache_size_limit: u64,

    /// Auto-cleanup enabled
    pub auto_cleanup: bool,

    /// Cleanup threshold
    pub cleanup_threshold: f32,

    /// Model versioning enabled
    pub versioning_enabled: bool,
}
