//! Core types and configurations for neural spatial audio processing

use crate::types::Position3D;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for neural spatial audio processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralSpatialConfig {
    /// Model architecture type
    pub model_type: NeuralModelType,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Input feature dimensions
    pub input_dim: usize,
    /// Output audio channels (typically 2 for binaural)
    pub output_channels: usize,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Buffer size for processing
    pub buffer_size: usize,
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
    /// Model quality setting (0.0-1.0)
    pub quality: f32,
    /// Real-time processing constraints
    pub realtime_constraints: RealtimeConstraints,
    /// Training parameters
    pub training_config: Option<TrainingConfig>,
}

/// Types of neural models available
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NeuralModelType {
    /// Feedforward neural network for basic spatial synthesis
    Feedforward,
    /// Convolutional neural network for temporal-spatial processing
    Convolutional,
    /// Recurrent neural network for temporal modeling
    Recurrent,
    /// Transformer model for attention-based spatial processing
    Transformer,
    /// Generative Adversarial Network for high-quality synthesis
    GAN,
    /// Variational Autoencoder for latent space spatial modeling
    VAE,
    /// Diffusion model for high-fidelity spatial audio generation
    Diffusion,
    /// Hybrid model combining multiple architectures
    Hybrid,
}

/// Real-time processing constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeConstraints {
    /// Maximum latency in milliseconds
    pub max_latency_ms: f32,
    /// Maximum CPU usage percentage
    pub max_cpu_usage: f32,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Target frame rate for processing
    pub target_fps: u32,
    /// Enable adaptive quality adjustment
    pub adaptive_quality: bool,
}

/// Training configuration for neural models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Validation split ratio
    pub validation_split: f32,
    /// Loss function type
    pub loss_function: LossFunction,
    /// Optimizer type
    pub optimizer: OptimizerType,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Data augmentation settings
    pub augmentation: AugmentationConfig,
}

/// Neural network loss functions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LossFunction {
    /// Mean Squared Error for regression
    MSE,
    /// Mean Absolute Error
    MAE,
    /// Spectral loss for audio quality
    SpectralLoss,
    /// Perceptual loss based on human auditory system
    PerceptualLoss,
    /// Multi-scale spectral loss
    MultiScaleSpectralLoss,
    /// Combined loss function
    Combined,
}

/// Optimizer types for training
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Adam optimizer
    Adam,
    /// Stochastic Gradient Descent
    SGD,
    /// AdamW with weight decay
    AdamW,
    /// RMSprop optimizer
    RMSprop,
}

/// Data augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    /// Enable noise injection
    pub noise_injection: bool,
    /// Enable time stretching
    pub time_stretching: bool,
    /// Enable pitch shifting
    pub pitch_shifting: bool,
    /// Enable reverb augmentation
    pub reverb_augmentation: bool,
    /// Random gain variation range
    pub gain_variation: f32,
}

/// Input features for neural spatial processing
#[derive(Debug, Clone)]
pub struct NeuralInputFeatures {
    /// 3D position of the sound source
    pub position: Position3D,
    /// Listener orientation (quaternion: w, x, y, z)
    pub listener_orientation: [f32; 4],
    /// Audio content features (e.g., spectral features)
    pub audio_features: Vec<f32>,
    /// Room acoustics parameters
    pub room_features: Vec<f32>,
    /// HRTF parameters if available
    pub hrtf_features: Option<Vec<f32>>,
    /// Temporal context from previous frames
    pub temporal_context: Vec<f32>,
    /// User-specific features (age, head size, etc.)
    pub user_features: Option<Vec<f32>>,
}

/// Output from neural spatial processing
#[derive(Debug, Clone)]
pub struct NeuralSpatialOutput {
    /// Synthesized binaural audio (left, right channels)
    pub binaural_audio: Vec<Vec<f32>>,
    /// Confidence score for the synthesis
    pub confidence: f32,
    /// Processing latency in milliseconds
    pub latency_ms: f32,
    /// Quality score (0.0-1.0)
    pub quality_score: f32,
    /// Additional metadata
    pub metadata: HashMap<String, f32>,
}

/// Performance metrics for neural processing
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct NeuralPerformanceMetrics {
    /// Total number of processed frames
    pub frames_processed: u64,
    /// Average processing time per frame (ms)
    pub avg_processing_time_ms: f32,
    /// Peak processing time (ms)
    pub peak_processing_time_ms: f32,
    /// Memory usage in MB
    pub memory_usage_mb: f32,
    /// GPU utilization percentage
    pub gpu_utilization: f32,
    /// Model inference time (ms)
    pub inference_time_ms: f32,
    /// Quality degradation events
    pub quality_degradations: u32,
    /// Real-time violations
    pub realtime_violations: u32,
    /// Last updated timestamp (seconds since UNIX epoch)
    pub last_updated: u64,
}

/// Training results from neural model training
#[derive(Debug, Clone)]
pub struct NeuralTrainingResults {
    /// Training loss per epoch
    pub training_loss: Vec<f32>,
    /// Validation loss per epoch
    pub validation_loss: Vec<f32>,
    /// Final training accuracy
    pub final_accuracy: f32,
    /// Training duration in seconds
    pub training_duration_secs: f32,
    /// Number of epochs completed
    pub epochs_completed: usize,
    /// Whether early stopping was triggered
    pub early_stopped: bool,
}

/// Builder for neural spatial processor configuration
pub struct NeuralSpatialConfigBuilder {
    config: NeuralSpatialConfig,
}

impl Default for NeuralSpatialConfig {
    fn default() -> Self {
        Self {
            model_type: NeuralModelType::Feedforward,
            hidden_dims: vec![512, 256, 128],
            input_dim: 128,
            output_channels: 2,
            sample_rate: 48000,
            buffer_size: 1024,
            use_gpu: true,
            quality: 0.8,
            realtime_constraints: RealtimeConstraints {
                max_latency_ms: 20.0,
                max_cpu_usage: 25.0,
                max_memory_mb: 512,
                target_fps: 60,
                adaptive_quality: true,
            },
            training_config: None,
        }
    }
}

impl Default for RealtimeConstraints {
    fn default() -> Self {
        Self {
            max_latency_ms: 20.0,
            max_cpu_usage: 25.0,
            max_memory_mb: 512,
            target_fps: 60,
            adaptive_quality: true,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            validation_split: 0.2,
            loss_function: LossFunction::MultiScaleSpectralLoss,
            optimizer: OptimizerType::Adam,
            early_stopping_patience: 10,
            augmentation: AugmentationConfig::default(),
        }
    }
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            noise_injection: true,
            time_stretching: true,
            pitch_shifting: true,
            reverb_augmentation: true,
            gain_variation: 0.1,
        }
    }
}

impl NeuralSpatialConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: NeuralSpatialConfig::default(),
        }
    }

    /// Set the neural model type
    pub fn model_type(mut self, model_type: NeuralModelType) -> Self {
        self.config.model_type = model_type;
        self
    }

    /// Set the hidden layer dimensions
    pub fn hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.config.hidden_dims = dims;
        self
    }

    /// Set the input dimension
    pub fn input_dim(mut self, dim: usize) -> Self {
        self.config.input_dim = dim;
        self
    }

    /// Set the number of output channels
    pub fn output_channels(mut self, channels: usize) -> Self {
        self.config.output_channels = channels;
        self
    }

    /// Set the audio sample rate
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.config.sample_rate = rate;
        self
    }

    /// Set the audio buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }

    /// Enable or disable GPU usage
    pub fn use_gpu(mut self, use_gpu: bool) -> Self {
        self.config.use_gpu = use_gpu;
        self
    }

    /// Set the quality level (0.0-1.0)
    pub fn quality(mut self, quality: f32) -> Self {
        self.config.quality = quality.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum latency in milliseconds
    pub fn max_latency_ms(mut self, latency: f32) -> Self {
        self.config.realtime_constraints.max_latency_ms = latency;
        self
    }

    /// Set the training configuration
    pub fn training_config(mut self, training_config: TrainingConfig) -> Self {
        self.config.training_config = Some(training_config);
        self
    }

    /// Build the neural spatial configuration
    pub fn build(self) -> NeuralSpatialConfig {
        self.config
    }
}

impl Default for NeuralSpatialConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
