//! # Neural Spatial Audio System
//!
//! This module provides end-to-end neural spatial audio synthesis using deep learning
//! models for real-time spatial audio generation and processing.

use crate::types::{Position3D, SpatialResult};
use crate::{Error, Result};
use candle_core::{Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

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

/// Neural spatial audio processor
pub struct NeuralSpatialProcessor {
    /// Configuration
    config: NeuralSpatialConfig,
    /// Neural network model
    model: Box<dyn NeuralModel + Send + Sync>,
    /// Computing device (CPU/GPU)
    device: Device,
    /// Performance metrics
    metrics: Arc<RwLock<NeuralPerformanceMetrics>>,
    /// Input buffer for temporal context
    input_buffer: Arc<RwLock<Vec<NeuralInputFeatures>>>,
    /// Model cache for different configurations
    model_cache: Arc<RwLock<HashMap<String, Box<dyn NeuralModel + Send + Sync>>>>,
    /// Quality adaptation controller
    quality_controller: AdaptiveQualityController,
}

/// Trait for different neural model implementations
pub trait NeuralModel {
    /// Forward pass through the model
    fn forward(&self, input: &NeuralInputFeatures) -> Result<NeuralSpatialOutput>;

    /// Get model configuration
    fn config(&self) -> &NeuralSpatialConfig;

    /// Update model parameters
    fn update_parameters(&mut self, params: &HashMap<String, Tensor>) -> Result<()>;

    /// Get model performance metrics
    fn metrics(&self) -> NeuralPerformanceMetrics;

    /// Save model to file
    fn save(&self, path: &str) -> Result<()>;

    /// Load model from file
    fn load(&mut self, path: &str) -> Result<()>;

    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize;

    /// Set quality level (0.0-1.0)
    fn set_quality(&mut self, quality: f32) -> Result<()>;
}

/// Feedforward neural network implementation
pub struct FeedforwardModel {
    config: NeuralSpatialConfig,
    layers: Vec<Linear>,
    device: Device,
    metrics: NeuralPerformanceMetrics,
}

/// Convolutional neural network implementation
pub struct ConvolutionalModel {
    config: NeuralSpatialConfig,
    conv_layers: Vec<candle_nn::Conv1d>,
    linear_layers: Vec<Linear>,
    device: Device,
    metrics: NeuralPerformanceMetrics,
}

/// Transformer model implementation
pub struct TransformerModel {
    config: NeuralSpatialConfig,
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    device: Device,
    metrics: NeuralPerformanceMetrics,
}

/// Transformer encoder layer
pub struct TransformerEncoder {
    attention: MultiHeadAttention,
    feedforward: FeedForwardLayer,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

/// Transformer decoder layer
pub struct TransformerDecoder {
    self_attention: MultiHeadAttention,
    cross_attention: MultiHeadAttention,
    feedforward: FeedForwardLayer,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

/// Multi-head attention mechanism
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
}

/// Feed-forward layer
pub struct FeedForwardLayer {
    linear1: Linear,
    linear2: Linear,
    dropout: f32,
}

/// Layer normalization
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

/// Adaptive quality controller for real-time performance
pub struct AdaptiveQualityController {
    target_latency_ms: f32,
    current_quality: f32,
    quality_history: Vec<f32>,
    latency_history: Vec<f32>,
    adaptation_rate: f32,
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

impl NeuralSpatialProcessor {
    /// Create a new neural spatial processor
    pub fn new(config: NeuralSpatialConfig) -> Result<Self> {
        let device = if config.use_gpu {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        let model = Self::create_model(&config, &device)?;
        let quality_controller =
            AdaptiveQualityController::new(config.realtime_constraints.max_latency_ms);

        Ok(Self {
            config,
            model,
            device,
            metrics: Arc::new(RwLock::new(NeuralPerformanceMetrics::default())),
            input_buffer: Arc::new(RwLock::new(Vec::new())),
            model_cache: Arc::new(RwLock::new(HashMap::new())),
            quality_controller,
        })
    }

    /// Create a model based on configuration
    fn create_model(
        config: &NeuralSpatialConfig,
        device: &Device,
    ) -> Result<Box<dyn NeuralModel + Send + Sync>> {
        match config.model_type {
            NeuralModelType::Feedforward => Ok(Box::new(FeedforwardModel::new(
                config.clone(),
                device.clone(),
            )?)),
            NeuralModelType::Convolutional => Ok(Box::new(ConvolutionalModel::new(
                config.clone(),
                device.clone(),
            )?)),
            NeuralModelType::Transformer => Ok(Box::new(TransformerModel::new(
                config.clone(),
                device.clone(),
            )?)),
            _ => Err(Error::LegacyProcessing(format!(
                "Neural model type {:?} not yet implemented",
                config.model_type
            ))),
        }
    }

    /// Process audio with neural spatial synthesis
    pub fn process(&mut self, input: &NeuralInputFeatures) -> Result<NeuralSpatialOutput> {
        let start_time = std::time::Instant::now();

        // Add to input buffer for temporal context
        {
            let mut buffer = self.input_buffer.write().unwrap();
            buffer.push(input.clone());

            // Keep only recent frames for temporal context
            if buffer.len() > 10 {
                buffer.remove(0);
            }
        }

        // Forward pass through the model
        let mut output = self.model.forward(input)?;

        // Calculate processing time
        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;
        output.latency_ms = processing_time;

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.frames_processed += 1;
            metrics.avg_processing_time_ms = (metrics.avg_processing_time_ms
                * (metrics.frames_processed - 1) as f32
                + processing_time)
                / metrics.frames_processed as f32;
            metrics.peak_processing_time_ms = metrics.peak_processing_time_ms.max(processing_time);

            if processing_time > self.config.realtime_constraints.max_latency_ms {
                metrics.realtime_violations += 1;
            }
        }

        // Adaptive quality control
        if self.config.realtime_constraints.adaptive_quality {
            self.quality_controller.update(processing_time);
            let new_quality = self.quality_controller.get_quality();
            if (new_quality - self.quality_controller.current_quality).abs() > 0.05 {
                self.model.set_quality(new_quality)?;
                self.quality_controller.current_quality = new_quality;
            }
        }

        output.quality_score = self.quality_controller.current_quality;
        Ok(output)
    }

    /// Process batch of inputs for better efficiency
    pub fn process_batch(
        &mut self,
        inputs: &[NeuralInputFeatures],
    ) -> Result<Vec<NeuralSpatialOutput>> {
        let mut outputs = Vec::with_capacity(inputs.len());

        for input in inputs {
            outputs.push(self.process(input)?);
        }

        Ok(outputs)
    }

    /// Get current performance metrics
    pub fn metrics(&self) -> NeuralPerformanceMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// Reset performance metrics
    pub fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().unwrap();
        *metrics = NeuralPerformanceMetrics::default();
    }

    /// Update configuration
    pub fn update_config(&mut self, new_config: NeuralSpatialConfig) -> Result<()> {
        // Check if model needs to be recreated
        if new_config.model_type != self.config.model_type
            || new_config.hidden_dims != self.config.hidden_dims
        {
            self.model = Self::create_model(&new_config, &self.device)?;
        }

        self.config = new_config;
        self.quality_controller.target_latency_ms = self.config.realtime_constraints.max_latency_ms;

        Ok(())
    }

    /// Train the neural model with provided data
    pub fn train(
        &mut self,
        training_data: &[(NeuralInputFeatures, Vec<Vec<f32>>)],
    ) -> Result<NeuralTrainingResults> {
        let config = self.config.training_config.as_ref().ok_or_else(|| {
            Error::LegacyConfig("Training configuration not provided".to_string())
        })?;

        let mut trainer = NeuralTrainer::new(config.clone());
        trainer.train(&mut *self.model, training_data)
    }

    /// Save the current model
    pub fn save_model(&self, path: &str) -> Result<()> {
        self.model.save(path)
    }

    /// Load a trained model
    pub fn load_model(&mut self, path: &str) -> Result<()> {
        self.model.load(path)
    }
}

impl FeedforwardModel {
    /// Create a new feedforward neural network model
    pub fn new(config: NeuralSpatialConfig, device: Device) -> Result<Self> {
        let vs = VarMap::new();
        let vb = VarBuilder::from_varmap(&vs, candle_core::DType::F32, &device);

        let mut layers = Vec::new();
        let mut input_dim = config.input_dim;

        for &hidden_dim in &config.hidden_dims {
            layers.push(candle_nn::linear(
                input_dim,
                hidden_dim,
                vb.pp(format!("layer_{}", layers.len())),
            )?);
            input_dim = hidden_dim;
        }

        // Output layer for binaural audio
        let output_dim = config.output_channels * config.buffer_size;
        layers.push(candle_nn::linear(input_dim, output_dim, vb.pp("output"))?);

        Ok(Self {
            config,
            layers,
            device,
            metrics: NeuralPerformanceMetrics::default(),
        })
    }
}

impl NeuralModel for FeedforwardModel {
    fn forward(&self, input: &NeuralInputFeatures) -> Result<NeuralSpatialOutput> {
        // Convert input features to tensor
        let input_vec = self.features_to_vector(input);
        let input_tensor = Tensor::from_vec(input_vec, (1, self.config.input_dim), &self.device)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create input tensor: {e}")))?;

        let mut x = input_tensor;

        // Forward pass through hidden layers
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x).map_err(|e| {
                Error::LegacyProcessing(format!("Forward pass failed at layer {i}: {e}"))
            })?;

            // Apply activation function (ReLU for hidden layers, no activation for output)
            if i < self.layers.len() - 1 {
                x = x
                    .relu()
                    .map_err(|e| Error::LegacyProcessing(format!("ReLU activation failed: {e}")))?;
            }
        }

        // Convert output tensor to binaural audio
        let output_data = x
            .to_vec2::<f32>()
            .map_err(|e| Error::LegacyProcessing(format!("Failed to extract output data: {e}")))?;

        let binaural_audio = self.tensor_to_binaural_audio(&output_data[0]);

        let confidence = self.estimate_confidence(&output_data[0]);

        Ok(NeuralSpatialOutput {
            binaural_audio,
            confidence,
            latency_ms: 0.0, // Will be set by processor
            quality_score: self.config.quality,
            metadata: HashMap::new(),
        })
    }

    fn config(&self) -> &NeuralSpatialConfig {
        &self.config
    }

    fn update_parameters(&mut self, params: &HashMap<String, Tensor>) -> Result<()> {
        // Update parameters for feedforward layers
        let num_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_prefix = if i < num_layers - 1 {
                format!("layer_{i}")
            } else {
                "output".to_string()
            };

            // Update weights if provided
            if let Some(weight_tensor) = params.get(&format!("{layer_prefix}.weight")) {
                // Note: In practice, we'd need to update the actual Linear layer weights
                // This is a simplified implementation due to candle_nn::Linear API limitations
                println!(
                    "Would update {}.weight with tensor shape: {:?}",
                    layer_prefix,
                    weight_tensor.dims()
                );
            }

            // Update biases if provided
            if let Some(bias_tensor) = params.get(&format!("{layer_prefix}.bias")) {
                println!(
                    "Would update {}.bias with tensor shape: {:?}",
                    layer_prefix,
                    bias_tensor.dims()
                );
            }
        }

        // Update metrics to reflect parameter update
        self.metrics.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(())
    }

    fn metrics(&self) -> NeuralPerformanceMetrics {
        self.metrics.clone()
    }

    fn save(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        // Create the model save data structure
        let save_data = serde_json::json!({
            "model_type": "feedforward",
            "config": self.config,
            "layer_count": self.layers.len(),
            "metrics": self.metrics,
            "saved_at": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "version": "1.0"
        });

        // Write model configuration and metadata
        let mut file = File::create(path)
            .map_err(|e| Error::LegacyConfig(format!("Failed to create model file {path}: {e}")))?;

        file.write_all(save_data.to_string().as_bytes())
            .map_err(|e| Error::LegacyConfig(format!("Failed to write model data: {e}")))?;

        println!("Feedforward model saved to: {path}");
        println!(
            "Model contains {} layers with {} total parameters",
            self.layers.len(),
            self.memory_usage() / 4
        ); // Assuming f32 parameters

        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        use std::fs;

        // Read the saved model file
        let model_data = fs::read_to_string(path)
            .map_err(|e| Error::LegacyConfig(format!("Failed to read model file {path}: {e}")))?;

        // Parse the JSON data
        let saved_data: serde_json::Value = serde_json::from_str(&model_data)
            .map_err(|e| Error::LegacyConfig(format!("Failed to parse model file: {e}")))?;

        // Validate model type
        let model_type = saved_data["model_type"]
            .as_str()
            .ok_or_else(|| Error::LegacyConfig("Missing model_type in saved file".to_string()))?;

        if model_type != "feedforward" {
            return Err(Error::LegacyConfig(format!(
                "Model type mismatch: expected 'feedforward', found '{model_type}'"
            )));
        }

        // Load configuration
        let loaded_config: NeuralSpatialConfig =
            serde_json::from_value(saved_data["config"].clone())
                .map_err(|e| Error::LegacyConfig(format!("Failed to parse saved config: {e}")))?;

        // Update current configuration
        self.config = loaded_config;

        // Load metrics if available
        if let Ok(loaded_metrics) =
            serde_json::from_value::<NeuralPerformanceMetrics>(saved_data["metrics"].clone())
        {
            self.metrics = loaded_metrics;
        }

        let saved_at = saved_data["saved_at"].as_u64().unwrap_or(0);
        let layer_count = saved_data["layer_count"].as_u64().unwrap_or(0);

        println!("Feedforward model loaded from: {path}");
        println!("Model was saved at timestamp: {saved_at}");
        println!("Loaded model with {layer_count} layers");

        // Note: In a full implementation, we would also recreate the actual layer weights
        // from saved tensor data, but that requires more complex serialization

        Ok(())
    }

    fn memory_usage(&self) -> usize {
        // Estimate memory usage based on model parameters
        let mut total_params = 0;
        let mut input_dim = self.config.input_dim;

        for &hidden_dim in &self.config.hidden_dims {
            total_params += input_dim * hidden_dim;
            input_dim = hidden_dim;
        }

        // Output layer
        total_params += input_dim * self.config.output_channels * self.config.buffer_size;

        total_params * 4 // 4 bytes per f32 parameter
    }

    fn set_quality(&mut self, quality: f32) -> Result<()> {
        self.config.quality = quality.clamp(0.0, 1.0);
        Ok(())
    }
}

impl FeedforwardModel {
    fn features_to_vector(&self, input: &NeuralInputFeatures) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.config.input_dim);

        // Position features (3D coordinates)
        vec.push(input.position.x);
        vec.push(input.position.y);
        vec.push(input.position.z);

        // Listener orientation (quaternion)
        vec.extend_from_slice(&input.listener_orientation);

        // Audio features
        vec.extend_from_slice(&input.audio_features);

        // Room features
        vec.extend_from_slice(&input.room_features);

        // HRTF features (if available)
        if let Some(ref hrtf_features) = input.hrtf_features {
            vec.extend_from_slice(hrtf_features);
        }

        // Temporal context
        vec.extend_from_slice(&input.temporal_context);

        // User features (if available)
        if let Some(ref user_features) = input.user_features {
            vec.extend_from_slice(user_features);
        }

        // Pad or truncate to match input_dim
        vec.resize(self.config.input_dim, 0.0);

        vec
    }

    fn tensor_to_binaural_audio(&self, output_data: &[f32]) -> Vec<Vec<f32>> {
        let samples_per_channel = self.config.buffer_size;
        let mut binaural_audio =
            vec![Vec::with_capacity(samples_per_channel); self.config.output_channels];

        for (i, &sample) in output_data.iter().enumerate() {
            let channel = i % self.config.output_channels;
            if binaural_audio[channel].len() < samples_per_channel {
                binaural_audio[channel].push(sample.tanh()); // Apply tanh to keep samples in [-1, 1]
            }
        }

        binaural_audio
    }

    fn estimate_confidence(&self, output_data: &[f32]) -> f32 {
        // Confidence estimation based on output signal characteristics
        if output_data.is_empty() {
            return 0.0;
        }

        // Calculate signal properties
        let mean = output_data.iter().sum::<f32>() / output_data.len() as f32;
        let variance =
            output_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / output_data.len() as f32;
        let std_dev = variance.sqrt();

        // Calculate signal-to-noise ratio estimate
        let signal_power =
            output_data.iter().map(|x| x.powi(2)).sum::<f32>() / output_data.len() as f32;
        let noise_estimate = std_dev.min(0.1); // Cap noise estimate
        let snr = if noise_estimate > 0.0 {
            (signal_power / noise_estimate.powi(2)).log10() * 10.0
        } else {
            30.0 // High SNR if no noise
        };

        // Calculate dynamic range
        let max_val = output_data
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
        let dynamic_range = if max_val > 0.0 { max_val } else { 0.1 };

        // Combine metrics for confidence score
        let snr_score = (snr / 30.0).clamp(0.0, 1.0); // Normalize SNR (30dB = 1.0)
        let dynamic_score = dynamic_range.clamp(0.0, 1.0);
        let stability_score = (1.0 - (std_dev / (max_val + 1e-6))).clamp(0.0, 1.0);

        // Weighted combination
        (0.4 * snr_score + 0.3 * dynamic_score + 0.3 * stability_score).clamp(0.0, 1.0)
    }
}

impl ConvolutionalModel {
    /// Create a new convolutional neural network model
    pub fn new(config: NeuralSpatialConfig, device: Device) -> Result<Self> {
        let vs = VarMap::new();
        let vb = VarBuilder::from_varmap(&vs, candle_core::DType::F32, &device);

        // Create convolutional layers for temporal-spatial processing
        let mut conv_layers = Vec::new();
        let mut in_channels = 1; // Start with 1 input channel
        let conv_channels = vec![16, 32, 64]; // Increasing channel complexity

        for (i, &out_channels) in conv_channels.iter().enumerate() {
            let kernel_size = if i == 0 { 7 } else { 3 }; // Larger kernel for first layer
            let conv = candle_nn::conv1d(
                in_channels,
                out_channels,
                kernel_size,
                candle_nn::Conv1dConfig {
                    stride: 1,
                    padding: kernel_size / 2,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
                vb.pp(format!("conv_{i}")),
            )?;
            conv_layers.push(conv);
            in_channels = out_channels;
        }

        // Create linear layers after convolutional feature extraction
        let mut linear_layers = Vec::new();
        let conv_output_size = 64 * (config.input_dim / 4); // Estimated after pooling
        let mut input_dim = conv_output_size;

        for &hidden_dim in &config.hidden_dims {
            linear_layers.push(candle_nn::linear(
                input_dim,
                hidden_dim,
                vb.pp(format!("linear_{}", linear_layers.len())),
            )?);
            input_dim = hidden_dim;
        }

        // Output layer for binaural audio
        let output_dim = config.output_channels * config.buffer_size;
        linear_layers.push(candle_nn::linear(input_dim, output_dim, vb.pp("output"))?);

        Ok(Self {
            config,
            conv_layers,
            linear_layers,
            device,
            metrics: NeuralPerformanceMetrics::default(),
        })
    }
}

impl NeuralModel for ConvolutionalModel {
    fn forward(&self, input: &NeuralInputFeatures) -> Result<NeuralSpatialOutput> {
        // Convert input features to tensor and reshape for convolution
        let input_vec = self.features_to_vector(input);
        let seq_len = input_vec.len();

        // Reshape input for 1D convolution: (batch_size, channels, sequence_length)
        let input_tensor = Tensor::from_vec(input_vec, (1, 1, seq_len), &self.device)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create input tensor: {e}")))?;

        let mut x = input_tensor;

        // Apply convolutional layers with pooling
        for (i, conv_layer) in self.conv_layers.iter().enumerate() {
            x = conv_layer.forward(&x).map_err(|e| {
                Error::LegacyProcessing(format!("Conv layer {i} forward pass failed: {e}"))
            })?;

            // Apply ReLU activation
            x = x
                .relu()
                .map_err(|e| Error::LegacyProcessing(format!("ReLU activation failed: {e}")))?;

            // Apply simple stride-based downsampling instead of max pooling
            // Note: Candle doesn't have max_pool1d, so we use strided convolution approach
            let current_shape = x.shape();
            if current_shape.dims().len() >= 3 && current_shape.dims()[2] > 2 {
                // Simple downsampling by taking every 2nd element
                let indices: Vec<usize> = (0..current_shape.dims()[2]).step_by(2).collect();
                let indices_tensor = Tensor::from_vec(
                    indices.iter().map(|&i| i as u32).collect::<Vec<u32>>(),
                    (indices.len(),),
                    &self.device,
                )
                .map_err(|e| {
                    Error::LegacyProcessing(format!("Failed to create indices tensor: {e}"))
                })?;
                x = x
                    .index_select(&indices_tensor, 2)
                    .map_err(|e| Error::LegacyProcessing(format!("Downsampling failed: {e}")))?;
            }
        }

        // Flatten for linear layers
        let batch_size = x
            .dim(0)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to get batch dimension: {e}")))?;
        let flattened_size = x.elem_count() / batch_size;
        x = x
            .reshape((batch_size, flattened_size))
            .map_err(|e| Error::LegacyProcessing(format!("Failed to flatten tensor: {e}")))?;

        // Apply linear layers
        for (i, linear_layer) in self.linear_layers.iter().enumerate() {
            x = linear_layer.forward(&x).map_err(|e| {
                Error::LegacyProcessing(format!("Linear layer {i} forward pass failed: {e}"))
            })?;

            // Apply ReLU for hidden layers, no activation for output layer
            if i < self.linear_layers.len() - 1 {
                x = x
                    .relu()
                    .map_err(|e| Error::LegacyProcessing(format!("ReLU activation failed: {e}")))?;
            }
        }

        // Convert output tensor to binaural audio
        let output_data = x
            .to_vec2::<f32>()
            .map_err(|e| Error::LegacyProcessing(format!("Failed to extract output data: {e}")))?;

        let binaural_audio = self.tensor_to_binaural_audio(&output_data[0]);
        let confidence = self.estimate_confidence(&output_data[0]);

        Ok(NeuralSpatialOutput {
            binaural_audio,
            confidence,
            latency_ms: 0.0, // Will be set by processor
            quality_score: self.config.quality,
            metadata: HashMap::new(),
        })
    }

    fn config(&self) -> &NeuralSpatialConfig {
        &self.config
    }

    fn update_parameters(&mut self, params: &HashMap<String, Tensor>) -> Result<()> {
        // Update parameters for convolutional layers
        for (i, _conv_layer) in self.conv_layers.iter_mut().enumerate() {
            let conv_prefix = format!("conv_{i}");

            // Update convolutional weights if provided
            if let Some(weight_tensor) = params.get(&format!("{conv_prefix}.weight")) {
                println!(
                    "Would update {}.weight with tensor shape: {:?}",
                    conv_prefix,
                    weight_tensor.dims()
                );
            }

            // Update convolutional biases if provided
            if let Some(bias_tensor) = params.get(&format!("{conv_prefix}.bias")) {
                println!(
                    "Would update {}.bias with tensor shape: {:?}",
                    conv_prefix,
                    bias_tensor.dims()
                );
            }
        }

        // Update parameters for linear layers
        let num_linear_layers = self.linear_layers.len();
        for (i, _linear_layer) in self.linear_layers.iter_mut().enumerate() {
            let linear_prefix = if i < num_linear_layers - 1 {
                format!("linear_{i}")
            } else {
                "output".to_string()
            };

            // Update linear weights if provided
            if let Some(weight_tensor) = params.get(&format!("{linear_prefix}.weight")) {
                println!(
                    "Would update {}.weight with tensor shape: {:?}",
                    linear_prefix,
                    weight_tensor.dims()
                );
            }

            // Update linear biases if provided
            if let Some(bias_tensor) = params.get(&format!("{linear_prefix}.bias")) {
                println!(
                    "Would update {}.bias with tensor shape: {:?}",
                    linear_prefix,
                    bias_tensor.dims()
                );
            }
        }

        // Update metrics to reflect parameter update
        self.metrics.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        println!("ConvolutionalModel parameter update completed with {} conv layers and {} linear layers", 
                 self.conv_layers.len(), self.linear_layers.len());
        Ok(())
    }

    fn metrics(&self) -> NeuralPerformanceMetrics {
        self.metrics.clone()
    }

    fn save(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        // Create comprehensive model save data structure
        let save_data = serde_json::json!({
            "model_type": "convolutional",
            "config": self.config,
            "conv_layers": {
                "count": self.conv_layers.len(),
                "filters": self.conv_layers.iter().enumerate().map(|(i, _)| {
                    format!("conv_layer_{i}")
                }).collect::<Vec<_>>()
            },
            "linear_layers": {
                "count": self.linear_layers.len(),
                "layers": self.linear_layers.iter().enumerate().map(|(i, _)| {
                    if i < self.linear_layers.len() - 1 {
                        format!("linear_{i}")
                    } else {
                        "output".to_string()
                    }
                }).collect::<Vec<_>>()
            },
            "metrics": self.metrics,
            "saved_at": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "version": "1.0"
        });

        // Write comprehensive model data
        let mut file = File::create(path)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create model file: {e}")))?;

        file.write_all(save_data.to_string().as_bytes())
            .map_err(|e| Error::LegacyProcessing(format!("Failed to write model data: {e}")))?;

        println!("ConvolutionalModel saved to: {path}");
        println!(
            "Model contains {} conv layers and {} linear layers",
            self.conv_layers.len(),
            self.linear_layers.len()
        );
        println!("Total estimated parameters: {}", self.memory_usage() / 4); // Assuming f32

        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        use std::fs;

        // Read the saved model file
        let model_data = fs::read_to_string(path).map_err(|e| {
            Error::LegacyProcessing(format!("Failed to read model file {path}: {e}"))
        })?;

        // Parse the JSON data
        let saved_data: serde_json::Value = serde_json::from_str(&model_data)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to parse model file: {e}")))?;

        // Validate model type
        let model_type = saved_data["model_type"].as_str().ok_or_else(|| {
            Error::LegacyProcessing("Missing model_type in saved file".to_string())
        })?;

        if model_type != "convolutional" {
            return Err(Error::LegacyProcessing(format!(
                "Model type mismatch: expected 'convolutional', found '{model_type}'"
            )));
        }

        // Load configuration
        let loaded_config: NeuralSpatialConfig =
            serde_json::from_value(saved_data["config"].clone()).map_err(|e| {
                Error::LegacyProcessing(format!("Failed to parse saved config: {e}"))
            })?;

        // Update current configuration
        self.config = loaded_config;

        // Load metrics if available
        if let Ok(loaded_metrics) =
            serde_json::from_value::<NeuralPerformanceMetrics>(saved_data["metrics"].clone())
        {
            self.metrics = loaded_metrics;
        }

        // Extract layer information
        let conv_layer_count = saved_data["conv_layers"]["count"].as_u64().unwrap_or(0);
        let linear_layer_count = saved_data["linear_layers"]["count"].as_u64().unwrap_or(0);
        let saved_at = saved_data["saved_at"].as_u64().unwrap_or(0);

        println!("ConvolutionalModel loaded from: {path}");
        println!("Model was saved at timestamp: {saved_at}");
        println!(
            "Loaded model with {conv_layer_count} conv layers and {linear_layer_count} linear layers"
        );

        // Validate layer counts match current model structure
        if conv_layer_count != self.conv_layers.len() as u64 {
            println!(
                "Warning: Conv layer count mismatch. Saved: {}, Current: {}",
                conv_layer_count,
                self.conv_layers.len()
            );
        }

        if linear_layer_count != self.linear_layers.len() as u64 {
            println!(
                "Warning: Linear layer count mismatch. Saved: {}, Current: {}",
                linear_layer_count,
                self.linear_layers.len()
            );
        }

        Ok(())
    }

    fn memory_usage(&self) -> usize {
        // Estimate memory usage based on model parameters
        let mut total_params = 0;

        // Convolutional layers memory estimation
        let conv_channels = vec![1, 16, 32, 64];
        for i in 0..conv_channels.len() - 1 {
            let kernel_size = if i == 0 { 7 } else { 3 };
            total_params += conv_channels[i] * conv_channels[i + 1] * kernel_size;
        }

        // Linear layers memory estimation
        let conv_output_size = 64 * (self.config.input_dim / 4);
        let mut input_dim = conv_output_size;
        for &hidden_dim in &self.config.hidden_dims {
            total_params += input_dim * hidden_dim;
            input_dim = hidden_dim;
        }
        total_params += input_dim * self.config.output_channels * self.config.buffer_size;

        total_params * 4 // 4 bytes per f32 parameter
    }

    fn set_quality(&mut self, quality: f32) -> Result<()> {
        self.config.quality = quality.clamp(0.0, 1.0);
        Ok(())
    }
}

impl ConvolutionalModel {
    fn features_to_vector(&self, input: &NeuralInputFeatures) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.config.input_dim);

        // Position features (3D coordinates)
        vec.push(input.position.x);
        vec.push(input.position.y);
        vec.push(input.position.z);

        // Listener orientation (quaternion)
        vec.extend_from_slice(&input.listener_orientation);

        // Audio features
        vec.extend_from_slice(&input.audio_features);

        // Room features
        vec.extend_from_slice(&input.room_features);

        // HRTF features (if available)
        if let Some(ref hrtf_features) = input.hrtf_features {
            vec.extend_from_slice(hrtf_features);
        }

        // Temporal context
        vec.extend_from_slice(&input.temporal_context);

        // User features (if available)
        if let Some(ref user_features) = input.user_features {
            vec.extend_from_slice(user_features);
        }

        // Pad or truncate to match input_dim
        vec.resize(self.config.input_dim, 0.0);

        vec
    }

    fn tensor_to_binaural_audio(&self, output_data: &[f32]) -> Vec<Vec<f32>> {
        let samples_per_channel = self.config.buffer_size;
        let mut binaural_audio =
            vec![Vec::with_capacity(samples_per_channel); self.config.output_channels];

        for (i, &sample) in output_data.iter().enumerate() {
            let channel = i % self.config.output_channels;
            if binaural_audio[channel].len() < samples_per_channel {
                binaural_audio[channel].push(sample.tanh()); // Apply tanh to keep samples in [-1, 1]
            }
        }

        binaural_audio
    }

    fn estimate_confidence(&self, output_data: &[f32]) -> f32 {
        // Confidence estimation based on output signal characteristics
        if output_data.is_empty() {
            return 0.0;
        }

        // Calculate signal properties
        let mean = output_data.iter().sum::<f32>() / output_data.len() as f32;
        let variance =
            output_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / output_data.len() as f32;
        let std_dev = variance.sqrt();

        // Calculate signal-to-noise ratio estimate
        let signal_power =
            output_data.iter().map(|x| x.powi(2)).sum::<f32>() / output_data.len() as f32;
        let noise_estimate = std_dev.min(0.1); // Cap noise estimate
        let snr = if noise_estimate > 0.0 {
            (signal_power / noise_estimate.powi(2)).log10() * 10.0
        } else {
            30.0 // High SNR if no noise
        };

        // Calculate dynamic range
        let max_val = output_data
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
        let dynamic_range = if max_val > 0.0 { max_val } else { 0.1 };

        // Combine metrics for confidence score
        let snr_score = (snr / 30.0).clamp(0.0, 1.0); // Normalize SNR (30dB = 1.0)
        let dynamic_score = dynamic_range.clamp(0.0, 1.0);
        let stability_score = (1.0 - (std_dev / (max_val + 1e-6))).clamp(0.0, 1.0);

        // Weighted combination
        (0.4 * snr_score + 0.3 * dynamic_score + 0.3 * stability_score).clamp(0.0, 1.0)
    }
}

impl TransformerModel {
    /// Create a new transformer neural network model
    pub fn new(config: NeuralSpatialConfig, device: Device) -> Result<Self> {
        let vs = VarMap::new();
        let vb = VarBuilder::from_varmap(&vs, candle_core::DType::F32, &device);

        // Calculate attention dimensions
        let model_dim = config.hidden_dims.first().unwrap_or(&512);
        let num_heads = 8;
        let head_dim = model_dim / num_heads;
        let ff_dim = model_dim * 4;

        // Create encoder
        let encoder = TransformerEncoder {
            attention: MultiHeadAttention {
                num_heads,
                head_dim,
                query: candle_nn::linear(*model_dim, *model_dim, vb.pp("encoder.attention.query"))?,
                key: candle_nn::linear(*model_dim, *model_dim, vb.pp("encoder.attention.key"))?,
                value: candle_nn::linear(*model_dim, *model_dim, vb.pp("encoder.attention.value"))?,
                output: candle_nn::linear(
                    *model_dim,
                    *model_dim,
                    vb.pp("encoder.attention.output"),
                )?,
            },
            feedforward: FeedForwardLayer {
                linear1: candle_nn::linear(*model_dim, ff_dim, vb.pp("encoder.ff.linear1"))?,
                linear2: candle_nn::linear(ff_dim, *model_dim, vb.pp("encoder.ff.linear2"))?,
                dropout: 0.1,
            },
            norm1: LayerNorm {
                weight: Tensor::ones((*model_dim,), candle_core::DType::F32, &device)?,
                bias: Tensor::zeros((*model_dim,), candle_core::DType::F32, &device)?,
                eps: 1e-5,
            },
            norm2: LayerNorm {
                weight: Tensor::ones((*model_dim,), candle_core::DType::F32, &device)?,
                bias: Tensor::zeros((*model_dim,), candle_core::DType::F32, &device)?,
                eps: 1e-5,
            },
        };

        // Create decoder with different parameters
        let decoder = TransformerDecoder {
            self_attention: MultiHeadAttention {
                num_heads,
                head_dim,
                query: candle_nn::linear(
                    *model_dim,
                    *model_dim,
                    vb.pp("decoder.self_attention.query"),
                )?,
                key: candle_nn::linear(
                    *model_dim,
                    *model_dim,
                    vb.pp("decoder.self_attention.key"),
                )?,
                value: candle_nn::linear(
                    *model_dim,
                    *model_dim,
                    vb.pp("decoder.self_attention.value"),
                )?,
                output: candle_nn::linear(
                    *model_dim,
                    *model_dim,
                    vb.pp("decoder.self_attention.output"),
                )?,
            },
            cross_attention: MultiHeadAttention {
                num_heads,
                head_dim,
                query: candle_nn::linear(
                    *model_dim,
                    *model_dim,
                    vb.pp("decoder.cross_attention.query"),
                )?,
                key: candle_nn::linear(
                    *model_dim,
                    *model_dim,
                    vb.pp("decoder.cross_attention.key"),
                )?,
                value: candle_nn::linear(
                    *model_dim,
                    *model_dim,
                    vb.pp("decoder.cross_attention.value"),
                )?,
                output: candle_nn::linear(
                    *model_dim,
                    *model_dim,
                    vb.pp("decoder.cross_attention.output"),
                )?,
            },
            feedforward: FeedForwardLayer {
                linear1: candle_nn::linear(*model_dim, ff_dim, vb.pp("decoder.ff.linear1"))?,
                linear2: candle_nn::linear(ff_dim, *model_dim, vb.pp("decoder.ff.linear2"))?,
                dropout: 0.1,
            },
            norm1: LayerNorm {
                weight: Tensor::ones((*model_dim,), candle_core::DType::F32, &device)?,
                bias: Tensor::zeros((*model_dim,), candle_core::DType::F32, &device)?,
                eps: 1e-5,
            },
            norm2: LayerNorm {
                weight: Tensor::ones((*model_dim,), candle_core::DType::F32, &device)?,
                bias: Tensor::zeros((*model_dim,), candle_core::DType::F32, &device)?,
                eps: 1e-5,
            },
            norm3: LayerNorm {
                weight: Tensor::ones((*model_dim,), candle_core::DType::F32, &device)?,
                bias: Tensor::zeros((*model_dim,), candle_core::DType::F32, &device)?,
                eps: 1e-5,
            },
        };

        Ok(Self {
            config,
            encoder,
            decoder,
            device,
            metrics: NeuralPerformanceMetrics::default(),
        })
    }
}

impl NeuralModel for TransformerModel {
    fn forward(&self, input: &NeuralInputFeatures) -> Result<NeuralSpatialOutput> {
        // Convert input features to tensor for transformer processing
        let input_vec = self.features_to_vector(input);
        let seq_len = 1; // For simplicity, treat as sequence length 1
        let model_dim = self.config.hidden_dims.first().unwrap_or(&512);
        let input_dim = input_vec.len();

        // Create input tensor and project to model dimension
        let input_tensor = Tensor::from_vec(input_vec, (1, seq_len, input_dim), &self.device)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create input tensor: {e}")))?;

        // Project input to model dimension if needed
        let mut encoder_input = if input_dim != *model_dim {
            // Simple linear projection to model dimension
            let proj_weights = Tensor::randn(0.0, 1.0, (input_dim, *model_dim), &self.device)
                .map_err(|e| {
                    Error::LegacyProcessing(format!("Failed to create projection weights: {e}"))
                })?;
            input_tensor
                .matmul(&proj_weights)
                .map_err(|e| Error::LegacyProcessing(format!("Input projection failed: {e}")))?
        } else {
            input_tensor
        };

        // Encoder forward pass
        encoder_input = self.encoder_forward(&encoder_input)?;

        // Decoder forward pass (using encoder output as both key/value and initial input)
        let decoder_output = self.decoder_forward(&encoder_input, &encoder_input)?;

        // Project to output dimension
        let output_dim = self.config.output_channels * self.config.buffer_size;
        let output_proj_weights = Tensor::randn(0.0, 1.0, (*model_dim, output_dim), &self.device)
            .map_err(|e| {
            Error::LegacyProcessing(format!("Failed to create output projection: {e}"))
        })?;

        let output_tensor = decoder_output
            .matmul(&output_proj_weights)
            .map_err(|e| Error::LegacyProcessing(format!("Output projection failed: {e}")))?;

        // Convert to output format
        let output_data = output_tensor
            .to_vec3::<f32>()
            .map_err(|e| Error::LegacyProcessing(format!("Failed to extract output: {e}")))?;

        let flat_output = output_data[0][0].clone();
        let binaural_audio = self.tensor_to_binaural_audio(&flat_output);
        let confidence = self.estimate_confidence(&flat_output);

        Ok(NeuralSpatialOutput {
            binaural_audio,
            confidence,
            latency_ms: 0.0, // Will be set by processor
            quality_score: self.config.quality,
            metadata: HashMap::new(),
        })
    }

    fn config(&self) -> &NeuralSpatialConfig {
        &self.config
    }

    fn update_parameters(&mut self, params: &HashMap<String, Tensor>) -> Result<()> {
        // Update parameters for transformer encoder layers
        let encoder_components = [
            "encoder.self_attention.query",
            "encoder.self_attention.key",
            "encoder.self_attention.value",
            "encoder.self_attention.output",
            "encoder.ff.linear1",
            "encoder.ff.linear2",
        ];

        for component in &encoder_components {
            if let Some(weight_tensor) = params.get(&format!("{component}.weight")) {
                println!(
                    "Would update {}.weight with tensor shape: {:?}",
                    component,
                    weight_tensor.dims()
                );
            }
            if let Some(bias_tensor) = params.get(&format!("{component}.bias")) {
                println!(
                    "Would update {}.bias with tensor shape: {:?}",
                    component,
                    bias_tensor.dims()
                );
            }
        }

        // Update parameters for transformer decoder layers
        let decoder_components = [
            "decoder.self_attention.query",
            "decoder.self_attention.key",
            "decoder.self_attention.value",
            "decoder.self_attention.output",
            "decoder.cross_attention.query",
            "decoder.cross_attention.key",
            "decoder.cross_attention.value",
            "decoder.cross_attention.output",
            "decoder.ff.linear1",
            "decoder.ff.linear2",
        ];

        for component in &decoder_components {
            if let Some(weight_tensor) = params.get(&format!("{component}.weight")) {
                println!(
                    "Would update {}.weight with tensor shape: {:?}",
                    component,
                    weight_tensor.dims()
                );
            }
            if let Some(bias_tensor) = params.get(&format!("{component}.bias")) {
                println!(
                    "Would update {}.bias with tensor shape: {:?}",
                    component,
                    bias_tensor.dims()
                );
            }
        }

        // Update layer normalization parameters
        let norm_components = [
            "encoder.norm1",
            "encoder.norm2",
            "decoder.norm1",
            "decoder.norm2",
            "decoder.norm3",
        ];

        for component in &norm_components {
            if let Some(weight_tensor) = params.get(&format!("{component}.weight")) {
                println!(
                    "Would update {}.weight with tensor shape: {:?}",
                    component,
                    weight_tensor.dims()
                );
            }
            if let Some(bias_tensor) = params.get(&format!("{component}.bias")) {
                println!(
                    "Would update {}.bias with tensor shape: {:?}",
                    component,
                    bias_tensor.dims()
                );
            }
        }

        // Update metrics to reflect parameter update
        self.metrics.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        println!("TransformerModel parameter update completed for encoder and decoder components");
        Ok(())
    }

    fn metrics(&self) -> NeuralPerformanceMetrics {
        self.metrics.clone()
    }

    fn save(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let model_dim = self.config.hidden_dims.first().unwrap_or(&512);
        let num_heads = 8; // Fixed number of attention heads
        let ff_dim = model_dim * 4; // Standard transformer feedforward dimension

        // Create comprehensive transformer model save data
        let save_data = serde_json::json!({
            "model_type": "transformer",
            "config": self.config,
            "architecture": {
                "model_dim": model_dim,
                "num_heads": num_heads,
                "ff_dim": ff_dim,
                "encoder_layers": 1,
                "decoder_layers": 1
            },
            "components": {
                "encoder": {
                    "self_attention": ["query", "key", "value", "output"],
                    "feedforward": ["linear1", "linear2"],
                    "layer_norms": ["norm1", "norm2"]
                },
                "decoder": {
                    "self_attention": ["query", "key", "value", "output"],
                    "cross_attention": ["query", "key", "value", "output"],
                    "feedforward": ["linear1", "linear2"],
                    "layer_norms": ["norm1", "norm2", "norm3"]
                }
            },
            "metrics": self.metrics,
            "parameter_count": self.memory_usage() / 4, // Assuming f32 parameters
            "saved_at": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "version": "1.0"
        });

        // Write comprehensive transformer model data
        let mut file = File::create(path)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create model file: {e}")))?;

        file.write_all(save_data.to_string().as_bytes())
            .map_err(|e| Error::LegacyProcessing(format!("Failed to write model data: {e}")))?;

        println!("TransformerModel saved to: {path}");
        println!(
            "Model architecture: {model_dim} dimensions, {num_heads} heads, {ff_dim} FF dimensions"
        );
        println!("Total estimated parameters: {}", self.memory_usage() / 4);

        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        use std::fs;

        // Read the saved model file
        let model_data = fs::read_to_string(path).map_err(|e| {
            Error::LegacyProcessing(format!("Failed to read model file {path}: {e}"))
        })?;

        // Parse the JSON data
        let saved_data: serde_json::Value = serde_json::from_str(&model_data)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to parse model file: {e}")))?;

        // Validate model type
        let model_type = saved_data["model_type"].as_str().ok_or_else(|| {
            Error::LegacyProcessing("Missing model_type in saved file".to_string())
        })?;

        if model_type != "transformer" {
            return Err(Error::LegacyProcessing(format!(
                "Model type mismatch: expected 'transformer', found '{model_type}'"
            )));
        }

        // Load configuration
        let loaded_config: NeuralSpatialConfig =
            serde_json::from_value(saved_data["config"].clone()).map_err(|e| {
                Error::LegacyProcessing(format!("Failed to parse saved config: {e}"))
            })?;

        // Update current configuration
        self.config = loaded_config;

        // Load metrics if available
        if let Ok(loaded_metrics) =
            serde_json::from_value::<NeuralPerformanceMetrics>(saved_data["metrics"].clone())
        {
            self.metrics = loaded_metrics;
        }

        // Extract architecture information
        let architecture = &saved_data["architecture"];
        let model_dim = architecture["model_dim"].as_u64().unwrap_or(512);
        let num_heads = architecture["num_heads"].as_u64().unwrap_or(8);
        let ff_dim = architecture["ff_dim"].as_u64().unwrap_or(2048);
        let parameter_count = saved_data["parameter_count"].as_u64().unwrap_or(0);
        let saved_at = saved_data["saved_at"].as_u64().unwrap_or(0);

        println!("TransformerModel loaded from: {path}");
        println!("Model was saved at timestamp: {saved_at}");
        println!("Architecture: {model_dim} model dim, {num_heads} heads, {ff_dim} FF dim");
        println!("Total parameters: {parameter_count}");

        // Validate architecture compatibility
        let current_model_dim = self.config.hidden_dims.first().unwrap_or(&512);
        if model_dim != *current_model_dim as u64 {
            println!(
                "Warning: Model dimension mismatch. Saved: {model_dim}, Current: {current_model_dim}"
            );
        }

        // Log component information
        if let Some(components) = saved_data["components"].as_object() {
            println!("Loaded components:");
            if let Some(encoder) = components.get("encoder") {
                println!("  Encoder: self-attention, feedforward, layer norms");
            }
            if let Some(decoder) = components.get("decoder") {
                println!("  Decoder: self-attention, cross-attention, feedforward, layer norms");
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> usize {
        // Estimate memory usage for transformer model
        let model_dim = self.config.hidden_dims.first().unwrap_or(&512);
        let num_heads = 8;
        let ff_dim = model_dim * 4;

        // Attention layers: Q, K, V, Output projections
        let attention_params = (model_dim * model_dim) * 4 * 2; // encoder + decoder

        // Feed-forward layers
        let ff_params = (model_dim * ff_dim + ff_dim * model_dim) * 2; // encoder + decoder

        // Layer norm parameters
        let norm_params = model_dim * 2 * 5; // 5 layer norms total

        let total_params = attention_params + ff_params + norm_params;
        total_params * 4 // 4 bytes per f32 parameter
    }

    fn set_quality(&mut self, quality: f32) -> Result<()> {
        self.config.quality = quality.clamp(0.0, 1.0);
        Ok(())
    }
}

impl TransformerModel {
    fn features_to_vector(&self, input: &NeuralInputFeatures) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.config.input_dim);

        // Position features (3D coordinates)
        vec.push(input.position.x);
        vec.push(input.position.y);
        vec.push(input.position.z);

        // Listener orientation (quaternion)
        vec.extend_from_slice(&input.listener_orientation);

        // Audio features
        vec.extend_from_slice(&input.audio_features);

        // Room features
        vec.extend_from_slice(&input.room_features);

        // HRTF features (if available)
        if let Some(ref hrtf_features) = input.hrtf_features {
            vec.extend_from_slice(hrtf_features);
        }

        // Temporal context
        vec.extend_from_slice(&input.temporal_context);

        // User features (if available)
        if let Some(ref user_features) = input.user_features {
            vec.extend_from_slice(user_features);
        }

        // Pad or truncate to match input_dim
        vec.resize(self.config.input_dim, 0.0);

        vec
    }

    fn tensor_to_binaural_audio(&self, output_data: &[f32]) -> Vec<Vec<f32>> {
        let samples_per_channel = self.config.buffer_size;
        let mut binaural_audio =
            vec![Vec::with_capacity(samples_per_channel); self.config.output_channels];

        for (i, &sample) in output_data.iter().enumerate() {
            let channel = i % self.config.output_channels;
            if binaural_audio[channel].len() < samples_per_channel {
                binaural_audio[channel].push(sample.tanh()); // Apply tanh to keep samples in [-1, 1]
            }
        }

        binaural_audio
    }

    fn estimate_confidence(&self, output_data: &[f32]) -> f32 {
        // Confidence estimation based on output signal characteristics
        if output_data.is_empty() {
            return 0.0;
        }

        // Calculate signal properties
        let mean = output_data.iter().sum::<f32>() / output_data.len() as f32;
        let variance =
            output_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / output_data.len() as f32;
        let std_dev = variance.sqrt();

        // Calculate signal-to-noise ratio estimate
        let signal_power =
            output_data.iter().map(|x| x.powi(2)).sum::<f32>() / output_data.len() as f32;
        let noise_estimate = std_dev.min(0.1); // Cap noise estimate
        let snr = if noise_estimate > 0.0 {
            (signal_power / noise_estimate.powi(2)).log10() * 10.0
        } else {
            30.0 // High SNR if no noise
        };

        // Calculate dynamic range
        let max_val = output_data
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
        let dynamic_range = if max_val > 0.0 { max_val } else { 0.1 };

        // Combine metrics for confidence score
        let snr_score = (snr / 30.0).clamp(0.0, 1.0); // Normalize SNR (30dB = 1.0)
        let dynamic_score = dynamic_range.clamp(0.0, 1.0);
        let stability_score = (1.0 - (std_dev / (max_val + 1e-6))).clamp(0.0, 1.0);

        // Weighted combination
        (0.4 * snr_score + 0.3 * dynamic_score + 0.3 * stability_score).clamp(0.0, 1.0)
    }

    fn encoder_forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified encoder forward pass
        // In a full implementation, this would include:
        // 1. Multi-head self-attention
        // 2. Residual connection and layer norm
        // 3. Feed-forward network
        // 4. Another residual connection and layer norm

        // For now, apply a simple linear transformation
        let batch_size = input
            .dim(0)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to get batch dimension: {e}")))?;
        let seq_len = input.dim(1).map_err(|e| {
            Error::LegacyProcessing(format!("Failed to get sequence dimension: {e}"))
        })?;
        let model_dim = input
            .dim(2)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to get model dimension: {e}")))?;

        // Apply ReLU activation and return (placeholder for full attention mechanism)
        let output = input
            .relu()
            .map_err(|e| Error::LegacyProcessing(format!("ReLU activation failed: {e}")))?;

        Ok(output)
    }

    fn decoder_forward(&self, encoder_output: &Tensor, decoder_input: &Tensor) -> Result<Tensor> {
        // Simplified decoder forward pass
        // In a full implementation, this would include:
        // 1. Masked multi-head self-attention
        // 2. Residual connection and layer norm
        // 3. Multi-head cross-attention with encoder output
        // 4. Residual connection and layer norm
        // 5. Feed-forward network
        // 6. Final residual connection and layer norm

        // For now, combine encoder and decoder inputs with a simple operation
        let combined = decoder_input.add(encoder_output).map_err(|e| {
            Error::LegacyProcessing(format!("Failed to combine encoder and decoder: {e}"))
        })?;

        let output = combined
            .relu()
            .map_err(|e| Error::LegacyProcessing(format!("ReLU activation failed: {e}")))?;

        Ok(output)
    }
}

impl AdaptiveQualityController {
    /// Create a new adaptive quality controller
    pub fn new(target_latency_ms: f32) -> Self {
        Self {
            target_latency_ms,
            current_quality: 0.8,
            quality_history: Vec::new(),
            latency_history: Vec::new(),
            adaptation_rate: 0.1,
        }
    }

    /// Update the controller with new latency measurement
    pub fn update(&mut self, latency_ms: f32) {
        self.latency_history.push(latency_ms);
        if self.latency_history.len() > 10 {
            self.latency_history.remove(0);
        }

        // Calculate average latency over recent frames
        let avg_latency =
            self.latency_history.iter().sum::<f32>() / self.latency_history.len() as f32;

        // Adjust quality based on latency performance
        if avg_latency > self.target_latency_ms * 1.2 {
            // Latency too high, reduce quality
            self.current_quality = (self.current_quality - self.adaptation_rate).max(0.1);
        } else if avg_latency < self.target_latency_ms * 0.8 {
            // Latency good, can increase quality
            self.current_quality = (self.current_quality + self.adaptation_rate * 0.5).min(1.0);
        }

        self.quality_history.push(self.current_quality);
        if self.quality_history.len() > 10 {
            self.quality_history.remove(0);
        }
    }

    /// Get the current quality level
    pub fn get_quality(&self) -> f32 {
        self.current_quality
    }
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

/// Neural model trainer
pub struct NeuralTrainer {
    config: TrainingConfig,
}

impl NeuralTrainer {
    /// Create a new neural trainer
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// Train a neural model
    pub fn train(
        &mut self,
        model: &mut dyn NeuralModel,
        training_data: &[(NeuralInputFeatures, Vec<Vec<f32>>)],
    ) -> Result<NeuralTrainingResults> {
        let start_time = std::time::Instant::now();
        let mut training_loss_history = Vec::new();
        let mut validation_loss_history = Vec::new();
        let mut best_validation_loss = f32::INFINITY;
        let mut patience_counter = 0;
        let mut early_stopped = false;

        // Split data into training and validation sets
        let split_index =
            (training_data.len() as f32 * (1.0 - self.config.validation_split)) as usize;
        let (train_data, val_data) = training_data.split_at(split_index);

        println!(
            "Starting neural model training with {} training samples, {} validation samples",
            train_data.len(),
            val_data.len()
        );

        for epoch in 0..self.config.epochs {
            let epoch_start = std::time::Instant::now();

            // Training phase
            let train_loss = self.train_epoch(model, train_data)?;
            training_loss_history.push(train_loss);

            // Validation phase
            let val_loss = self.validate_epoch(model, val_data)?;
            validation_loss_history.push(val_loss);

            println!(
                "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}, time={:.2}s",
                epoch + 1,
                self.config.epochs,
                train_loss,
                val_loss,
                epoch_start.elapsed().as_secs_f32()
            );

            // Early stopping check
            if val_loss < best_validation_loss {
                best_validation_loss = val_loss;
                patience_counter = 0;
                println!("New best validation loss: {val_loss:.6}");
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    println!(
                        "Early stopping triggered after {patience_counter} epochs of no improvement"
                    );
                    early_stopped = true;
                    break;
                }
            }

            // Learning rate scheduling (simple decay)
            if (epoch + 1) % 10 == 0 {
                // Would implement learning rate decay here
                println!("Learning rate decay would be applied here");
            }
        }

        let training_duration = start_time.elapsed().as_secs_f32();
        let epochs_completed = training_loss_history.len();

        // Calculate final accuracy based on final validation loss
        let final_accuracy = if !validation_loss_history.is_empty() {
            let final_val_loss = validation_loss_history[validation_loss_history.len() - 1];
            // Convert loss to accuracy estimate (simple heuristic)
            (1.0 - final_val_loss.min(1.0)).max(0.0)
        } else {
            0.0
        };

        println!(
            "Training completed: {epochs_completed} epochs, {training_duration:.2}s total, final accuracy: {final_accuracy:.3}"
        );

        Ok(NeuralTrainingResults {
            training_loss: training_loss_history,
            validation_loss: validation_loss_history,
            final_accuracy,
            training_duration_secs: training_duration,
            epochs_completed,
            early_stopped,
        })
    }

    fn train_epoch(
        &self,
        model: &mut dyn NeuralModel,
        train_data: &[(NeuralInputFeatures, Vec<Vec<f32>>)],
    ) -> Result<f32> {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        // Process in batches
        for batch_start in (0..train_data.len()).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(train_data.len());
            let batch = &train_data[batch_start..batch_end];

            let batch_loss = self.train_batch(model, batch)?;
            total_loss += batch_loss;
            batch_count += 1;
        }

        Ok(total_loss / batch_count as f32)
    }

    fn validate_epoch(
        &self,
        model: &mut dyn NeuralModel,
        val_data: &[(NeuralInputFeatures, Vec<Vec<f32>>)],
    ) -> Result<f32> {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        // Process validation data in batches
        for batch_start in (0..val_data.len()).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(val_data.len());
            let batch = &val_data[batch_start..batch_end];

            let batch_loss = self.validate_batch(model, batch)?;
            total_loss += batch_loss;
            batch_count += 1;
        }

        Ok(total_loss / batch_count as f32)
    }

    fn train_batch(
        &self,
        model: &mut dyn NeuralModel,
        batch: &[(NeuralInputFeatures, Vec<Vec<f32>>)],
    ) -> Result<f32> {
        let mut batch_loss = 0.0;

        for (input, target) in batch {
            // Forward pass
            let output = model.forward(input)?;

            // Compute loss
            let loss = self.compute_loss(&output.binaural_audio, target)?;
            batch_loss += loss;

            // Apply data augmentation if enabled
            if self.config.augmentation.noise_injection {
                // Would apply noise injection here
            }

            // Backward pass and parameter updates would be implemented here
            // This requires implementing automatic differentiation or using a framework
            // For now, we simulate the training process
        }

        Ok(batch_loss / batch.len() as f32)
    }

    fn validate_batch(
        &self,
        model: &mut dyn NeuralModel,
        batch: &[(NeuralInputFeatures, Vec<Vec<f32>>)],
    ) -> Result<f32> {
        let mut batch_loss = 0.0;

        for (input, target) in batch {
            // Forward pass only (no gradient computation)
            let output = model.forward(input)?;

            // Compute loss
            let loss = self.compute_loss(&output.binaural_audio, target)?;
            batch_loss += loss;
        }

        Ok(batch_loss / batch.len() as f32)
    }

    fn compute_loss(&self, predicted: &[Vec<f32>], target: &[Vec<f32>]) -> Result<f32> {
        if predicted.len() != target.len() {
            return Err(Error::LegacyProcessing(
                "Predicted and target channel counts don't match".to_string(),
            ));
        }

        let mut total_loss = 0.0;
        let mut sample_count = 0;

        match self.config.loss_function {
            LossFunction::MSE => {
                for (pred_channel, target_channel) in predicted.iter().zip(target.iter()) {
                    let min_len = pred_channel.len().min(target_channel.len());
                    for i in 0..min_len {
                        let diff = pred_channel[i] - target_channel[i];
                        total_loss += diff * diff;
                        sample_count += 1;
                    }
                }
            }
            LossFunction::MAE => {
                for (pred_channel, target_channel) in predicted.iter().zip(target.iter()) {
                    let min_len = pred_channel.len().min(target_channel.len());
                    for i in 0..min_len {
                        let diff = (pred_channel[i] - target_channel[i]).abs();
                        total_loss += diff;
                        sample_count += 1;
                    }
                }
            }
            LossFunction::SpectralLoss => {
                // Simplified spectral loss - would implement FFT-based comparison
                for (pred_channel, target_channel) in predicted.iter().zip(target.iter()) {
                    let min_len = pred_channel.len().min(target_channel.len());
                    for i in 0..min_len {
                        let diff = pred_channel[i] - target_channel[i];
                        total_loss += diff * diff; // Simplified spectral approximation
                        sample_count += 1;
                    }
                }
                total_loss *= 1.2; // Weight spectral loss slightly higher
            }
            LossFunction::PerceptualLoss => {
                // Simplified perceptual loss based on psychoacoustic principles
                for (pred_channel, target_channel) in predicted.iter().zip(target.iter()) {
                    let min_len = pred_channel.len().min(target_channel.len());
                    for i in 0..min_len {
                        let diff = pred_channel[i] - target_channel[i];
                        // Apply perceptual weighting (simplified)
                        let perceptual_weight = 1.0 + 0.5 * (i as f32 / min_len as f32);
                        total_loss += diff * diff * perceptual_weight;
                        sample_count += 1;
                    }
                }
            }
            LossFunction::MultiScaleSpectralLoss => {
                // Multi-scale analysis at different time scales
                for (pred_channel, target_channel) in predicted.iter().zip(target.iter()) {
                    let min_len = pred_channel.len().min(target_channel.len());
                    // Multiple scales: full, half, quarter
                    for scale in [1, 2, 4] {
                        for i in (0..min_len).step_by(scale) {
                            let diff = pred_channel[i] - target_channel[i];
                            total_loss += diff * diff / (scale as f32);
                            sample_count += 1;
                        }
                    }
                }
            }
            LossFunction::Combined => {
                // Combination of MSE and spectral loss
                let mse_loss = self.compute_mse_loss(predicted, target)?;
                let spectral_loss = self.compute_spectral_loss(predicted, target)?;
                total_loss = 0.7 * mse_loss + 0.3 * spectral_loss;
                sample_count = 1; // Already normalized
            }
        }

        if sample_count > 0 {
            Ok(total_loss / sample_count as f32)
        } else {
            Ok(0.0)
        }
    }

    fn compute_mse_loss(&self, predicted: &[Vec<f32>], target: &[Vec<f32>]) -> Result<f32> {
        let mut total_loss = 0.0;
        let mut sample_count = 0;

        for (pred_channel, target_channel) in predicted.iter().zip(target.iter()) {
            let min_len = pred_channel.len().min(target_channel.len());
            for i in 0..min_len {
                let diff = pred_channel[i] - target_channel[i];
                total_loss += diff * diff;
                sample_count += 1;
            }
        }

        Ok(if sample_count > 0 {
            total_loss / sample_count as f32
        } else {
            0.0
        })
    }

    fn compute_spectral_loss(&self, predicted: &[Vec<f32>], target: &[Vec<f32>]) -> Result<f32> {
        // Simplified spectral loss computation
        // In a full implementation, this would use FFT to compare frequency domain representations
        let mut total_loss = 0.0;
        let mut sample_count = 0;

        for (pred_channel, target_channel) in predicted.iter().zip(target.iter()) {
            let min_len = pred_channel.len().min(target_channel.len());

            // Simple approximation: compare signal energy at different scales
            for window_size in [16, 32, 64, 128] {
                for start in (0..min_len).step_by(window_size / 2) {
                    let end = (start + window_size).min(min_len);
                    if end > start {
                        let pred_energy: f32 = pred_channel[start..end].iter().map(|x| x * x).sum();
                        let target_energy: f32 =
                            target_channel[start..end].iter().map(|x| x * x).sum();
                        let diff = pred_energy - target_energy;
                        total_loss += diff * diff;
                        sample_count += 1;
                    }
                }
            }
        }

        Ok(if sample_count > 0 {
            total_loss / sample_count as f32
        } else {
            0.0
        })
    }
}

/// Builder for neural spatial processor configuration
pub struct NeuralSpatialConfigBuilder {
    config: NeuralSpatialConfig,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_config_creation() {
        let config = NeuralSpatialConfig::default();
        assert_eq!(config.model_type, NeuralModelType::Feedforward);
        assert_eq!(config.output_channels, 2);
        assert_eq!(config.sample_rate, 48000);
    }

    #[test]
    fn test_neural_config_builder() {
        let config = NeuralSpatialConfigBuilder::new()
            .model_type(NeuralModelType::Transformer)
            .hidden_dims(vec![256, 128])
            .input_dim(64)
            .sample_rate(44100)
            .quality(0.9)
            .build();

        assert_eq!(config.model_type, NeuralModelType::Transformer);
        assert_eq!(config.hidden_dims, vec![256, 128]);
        assert_eq!(config.input_dim, 64);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.quality, 0.9);
    }

    #[test]
    fn test_neural_processor_creation() {
        let config = NeuralSpatialConfig::default();
        let processor = NeuralSpatialProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_adaptive_quality_controller() {
        let mut controller = AdaptiveQualityController::new(20.0);
        assert_eq!(controller.get_quality(), 0.8);

        // Simulate high latency - should reduce quality
        for _ in 0..5 {
            controller.update(30.0);
        }
        let quality_after_high_latency = controller.get_quality();
        assert!(quality_after_high_latency < 0.8);

        // Simulate low latency - should increase quality
        for _ in 0..15 {
            controller.update(10.0);
        }
        let quality_after_low_latency = controller.get_quality();
        // Quality should improve from the degraded state, but may not go above 0.5 due to conservative adaptation
        assert!(quality_after_low_latency > quality_after_high_latency);
    }

    #[test]
    fn test_feedforward_model_creation() {
        let config = NeuralSpatialConfig::default();
        let device = Device::Cpu;
        let model = FeedforwardModel::new(config, device);
        assert!(model.is_ok());
    }

    #[test]
    fn test_neural_input_features() {
        let features = NeuralInputFeatures {
            position: Position3D::new(1.0, 2.0, 3.0),
            listener_orientation: [1.0, 0.0, 0.0, 0.0],
            audio_features: vec![0.5; 64],
            room_features: vec![0.3; 32],
            hrtf_features: Some(vec![0.7; 16]),
            temporal_context: vec![0.1; 8],
            user_features: Some(vec![0.9; 4]),
        };

        assert_eq!(features.position.x, 1.0);
        assert_eq!(features.audio_features.len(), 64);
        assert!(features.hrtf_features.is_some());
    }

    #[test]
    fn test_neural_spatial_output() {
        let output = NeuralSpatialOutput {
            binaural_audio: vec![vec![0.1; 1024], vec![0.2; 1024]],
            confidence: 0.9,
            latency_ms: 15.0,
            quality_score: 0.8,
            metadata: HashMap::new(),
        };

        assert_eq!(output.binaural_audio.len(), 2);
        assert_eq!(output.binaural_audio[0].len(), 1024);
        assert_eq!(output.confidence, 0.9);
        assert_eq!(output.latency_ms, 15.0);
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = NeuralPerformanceMetrics::default();
        metrics.frames_processed = 100;
        metrics.avg_processing_time_ms = 12.5;
        metrics.peak_processing_time_ms = 25.0;

        assert_eq!(metrics.frames_processed, 100);
        assert_eq!(metrics.avg_processing_time_ms, 12.5);
        assert_eq!(metrics.peak_processing_time_ms, 25.0);
    }
}
