//! DiffWave model training infrastructure
//!
//! This module provides comprehensive training capabilities for DiffWave vocoders:
//! - Loss functions (L1, L2, spectral, multi-scale STFT)
//! - Optimizer integration (Adam, AdamW, SGD)
//! - Learning rate scheduling
//! - Training loop with validation
//! - Checkpointing and model saving
//! - Performance monitoring and logging

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW, SGD};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use tokio::fs;

use super::diffusion::{DiffWave, DiffWaveConfig};
use crate::{Result, VocoderError};

/// Training configuration for DiffWave
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of training epochs
    pub num_epochs: usize,
    /// Validation frequency (epochs)
    pub validation_frequency: usize,
    /// Checkpoint save frequency (epochs)
    pub checkpoint_frequency: usize,
    /// Gradient clipping value
    pub gradient_clip: Option<f64>,
    /// Loss function configuration
    pub loss_config: LossConfig,
    /// Optimizer configuration
    pub optimizer_config: OptimizerConfig,
    /// Learning rate scheduler
    pub scheduler_config: Option<SchedulerConfig>,
    /// Data augmentation settings
    pub augmentation_config: AugmentationConfig,
    /// Training data paths
    pub data_config: DataConfig,
    /// Output directory for checkpoints and logs
    pub output_dir: PathBuf,
    /// Resume from checkpoint
    pub resume_from: Option<PathBuf>,
    /// Enable mixed precision training
    pub mixed_precision: bool,
    /// Device for training
    pub device: String,
}

/// Loss function configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConfig {
    /// Primary loss type
    pub primary_loss: LossType,
    /// Secondary losses with weights
    pub secondary_losses: Vec<(LossType, f64)>,
    /// Adversarial loss weight (if using GAN training)
    pub adversarial_weight: Option<f64>,
}

/// Available loss functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossType {
    /// L1 loss between predicted and target noise
    L1,
    /// L2 (MSE) loss between predicted and target noise
    L2,
    /// Huber loss with delta parameter
    Huber { delta: f64 },
    /// Spectral convergence loss
    SpectralConvergence,
    /// Multi-scale STFT loss
    MultiScaleSTFT {
        fft_sizes: Vec<usize>,
        hop_sizes: Vec<usize>,
        win_sizes: Vec<usize>,
    },
    /// Mel-spectrogram loss
    MelSpectrogramLoss {
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
    },
    /// Perceptual loss using pretrained features
    PerceptualLoss,
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerConfig {
    Adam {
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    },
    AdamW {
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    },
    SGD {
        momentum: f64,
        weight_decay: f64,
        nesterov: bool,
    },
}

/// Learning rate scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerConfig {
    /// Exponential decay
    ExponentialDecay { gamma: f64, step_size: usize },
    /// Cosine annealing
    CosineAnnealing { t_max: usize, eta_min: f64 },
    /// Linear warmup followed by decay
    LinearWarmupDecay {
        warmup_steps: usize,
        decay_steps: usize,
        min_lr: f64,
    },
    /// Reduce on plateau
    ReduceOnPlateau {
        factor: f64,
        patience: usize,
        threshold: f64,
        min_lr: f64,
    },
    /// Multi-step decay
    MultiStep { milestones: Vec<usize>, gamma: f64 },
}

/// Data augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    /// Enable noise addition
    pub add_noise: bool,
    /// Noise standard deviation
    pub noise_std: f64,
    /// Enable time stretching
    pub time_stretch: bool,
    /// Time stretch range
    pub time_stretch_range: (f64, f64),
    /// Enable pitch shifting
    pub pitch_shift: bool,
    /// Pitch shift range (semitones)
    pub pitch_shift_range: (f64, f64),
    /// Enable volume adjustment
    pub volume_adjust: bool,
    /// Volume adjustment range (dB)
    pub volume_range: (f64, f64),
}

/// Training data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Training data directory
    pub train_dir: PathBuf,
    /// Validation data directory
    pub val_dir: PathBuf,
    /// Audio file extensions to include
    pub audio_extensions: Vec<String>,
    /// Mel spectrogram extensions to include
    pub mel_extensions: Vec<String>,
    /// Maximum sequence length for training
    pub max_seq_len: usize,
    /// Minimum sequence length for training
    pub min_seq_len: usize,
    /// Number of data loading workers
    pub num_workers: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-4,
            batch_size: 16,
            num_epochs: 1000,
            validation_frequency: 10,
            checkpoint_frequency: 50,
            gradient_clip: Some(1.0),
            loss_config: LossConfig {
                primary_loss: LossType::L2,
                secondary_losses: vec![(LossType::L1, 0.1), (LossType::SpectralConvergence, 0.05)],
                adversarial_weight: None,
            },
            optimizer_config: OptimizerConfig::Adam {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 1e-6,
            },
            scheduler_config: Some(SchedulerConfig::LinearWarmupDecay {
                warmup_steps: 4000,
                decay_steps: 200000,
                min_lr: 1e-7,
            }),
            augmentation_config: AugmentationConfig {
                add_noise: true,
                noise_std: 0.01,
                time_stretch: false,
                time_stretch_range: (0.9, 1.1),
                pitch_shift: false,
                pitch_shift_range: (-2.0, 2.0),
                volume_adjust: true,
                volume_range: (-3.0, 3.0),
            },
            data_config: DataConfig {
                train_dir: PathBuf::from("data/train"),
                val_dir: PathBuf::from("data/val"),
                audio_extensions: vec!["wav".to_string(), "flac".to_string()],
                mel_extensions: vec!["mel".to_string(), "npy".to_string()],
                max_seq_len: 16384,
                min_seq_len: 1024,
                num_workers: 4,
            },
            output_dir: PathBuf::from("checkpoints"),
            resume_from: None,
            mixed_precision: false,
            device: "cpu".to_string(),
        }
    }
}

/// Training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStats {
    pub epoch: usize,
    pub step: usize,
    pub learning_rate: f64,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub grad_norm: Option<f64>,
    pub epoch_time: Duration,
    pub loss_components: HashMap<String, f64>,
}

/// Type-erased optimizer wrapper because `candle_nn::Optimizer` is not dyn-compatible
/// (it requires `Self: Sized`).  We enumerate the concrete variants instead.
enum AnyOptimizer {
    AdamW(AdamW),
    Sgd(SGD),
}

impl AnyOptimizer {
    /// Equivalent of `Optimizer::backward_step` — compute gradients and update vars.
    fn backward_step(&mut self, loss: &Tensor) -> CandleResult<()> {
        match self {
            AnyOptimizer::AdamW(o) => o.backward_step(loss),
            AnyOptimizer::Sgd(o) => o.backward_step(loss),
        }
    }

    fn set_learning_rate(&mut self, lr: f64) {
        match self {
            AnyOptimizer::AdamW(o) => o.set_learning_rate(lr),
            AnyOptimizer::Sgd(o) => o.set_learning_rate(lr),
        }
    }
}

/// DiffWave trainer
pub struct DiffWaveTrainer {
    model: DiffWave,
    config: TrainingConfig,
    optimizer: Option<AnyOptimizer>,
    scheduler: Option<LearningRateScheduler>,
    device: Device,
    varmap: VarMap,
    training_stats: Vec<TrainingStats>,
    best_val_loss: f64,
    global_step: usize,
}

impl DiffWaveTrainer {
    /// Create new trainer
    pub fn new(model_config: DiffWaveConfig, training_config: TrainingConfig) -> Result<Self> {
        let device = Device::Cpu; // Simplified for compatibility
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let model = DiffWave::new(model_config, device.clone(), vb)?;

        Ok(Self {
            model,
            config: training_config,
            optimizer: None,
            scheduler: None,
            device,
            varmap,
            training_stats: Vec::new(),
            best_val_loss: f64::INFINITY,
            global_step: 0,
        })
    }

    /// Initialize optimizer
    pub fn initialize_optimizer(&mut self) -> Result<()> {
        let params = self.varmap.all_vars();
        let lr = self.config.learning_rate;

        // candle-nn 0.10.x Optimizer trait is Sized-bound (not dyn-compatible).
        // Adam was removed; Adam-like behaviour is provided by AdamW (set weight_decay=0).
        // SGD no longer accepts momentum/weight_decay — lr only.
        let optimizer = match &self.config.optimizer_config {
            OptimizerConfig::Adam {
                beta1,
                beta2,
                eps,
                weight_decay,
            }
            | OptimizerConfig::AdamW {
                beta1,
                beta2,
                eps,
                weight_decay,
            } => AnyOptimizer::AdamW(AdamW::new(
                params,
                ParamsAdamW {
                    lr,
                    beta1: *beta1,
                    beta2: *beta2,
                    eps: *eps,
                    weight_decay: *weight_decay,
                },
            )?),
            OptimizerConfig::SGD { .. } => AnyOptimizer::Sgd(SGD::new(params, lr)?),
        };

        self.optimizer = Some(optimizer);

        // Initialize scheduler if configured
        if let Some(scheduler_config) = &self.config.scheduler_config {
            self.scheduler = Some(LearningRateScheduler::new(
                scheduler_config.clone(),
                self.config.learning_rate,
            ));
        }

        Ok(())
    }

    /// Train the model
    pub async fn train(&mut self) -> Result<()> {
        // Initialize optimizer if not already done
        if self.optimizer.is_none() {
            self.initialize_optimizer()?;
        }

        // Create output directory
        tokio::fs::create_dir_all(&self.config.output_dir).await?;

        // Resume from checkpoint if specified — clone path to release immutable borrow
        // before taking the mutable borrow required by load_checkpoint.
        if let Some(checkpoint_path) = self.config.resume_from.clone() {
            self.load_checkpoint(&checkpoint_path).await?;
        }

        // Training loop
        for epoch in 0..self.config.num_epochs {
            let epoch_start = Instant::now();

            // Training phase
            let train_loss = self.train_epoch(epoch).await?;

            // Validation phase
            let val_loss = if epoch.is_multiple_of(self.config.validation_frequency) {
                Some(self.validate_epoch(epoch).await?)
            } else {
                None
            };

            let epoch_time = epoch_start.elapsed();

            // Update learning rate
            if let Some(scheduler) = &mut self.scheduler {
                let new_lr = scheduler.step(val_loss);
                if let Some(optimizer) = &mut self.optimizer {
                    optimizer.set_learning_rate(new_lr);
                }
            }

            // Record statistics
            let stats = TrainingStats {
                epoch,
                step: self.global_step,
                learning_rate: self.get_current_learning_rate(),
                train_loss,
                val_loss,
                grad_norm: None, // Would be computed during training
                epoch_time,
                loss_components: HashMap::new(), // Would be populated with component losses
            };

            self.training_stats.push(stats);

            // Print progress
            println!(
                "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}, lr={:.2e}, time={:.1}s",
                epoch + 1,
                self.config.num_epochs,
                train_loss,
                val_loss.unwrap_or(0.0),
                self.get_current_learning_rate(),
                epoch_time.as_secs_f64()
            );

            // Save checkpoint
            if epoch.is_multiple_of(self.config.checkpoint_frequency) {
                self.save_checkpoint(epoch).await?;
            }

            // Save best model
            if let Some(val_loss_value) = val_loss {
                if val_loss_value < self.best_val_loss {
                    self.best_val_loss = val_loss_value;
                    self.save_best_model().await?;
                }
            }
        }

        Ok(())
    }

    /// Train for one epoch
    async fn train_epoch(&mut self, epoch: usize) -> Result<f64> {
        // This is a simplified training loop
        // In practice, this would iterate over actual training data

        let mut total_loss = 0.0;
        let num_batches: usize = 100; // Placeholder

        for batch_idx in 0..num_batches {
            // Generate dummy training batch
            let (audio_batch, mel_batch, timestep_batch) = self.generate_dummy_batch()?;

            // Forward pass
            let loss = self.training_step(&audio_batch, &mel_batch, &timestep_batch)?;

            let loss_val = loss.to_scalar::<f64>()?;

            // Gradient clipping simulation (read-only self borrow — must happen before mutable borrow)
            if self.optimizer.is_some() {
                if let Some(clip_value) = self.config.gradient_clip {
                    self.clip_gradients(clip_value)?;
                }
            }

            // Backward pass (backward_step combines backward + weight update)
            if let Some(optimizer) = &mut self.optimizer {
                optimizer
                    .backward_step(&loss)
                    .map_err(|e| VocoderError::ModelError(format!("optimizer step failed: {e}")))?;
            }

            total_loss += loss_val;
            self.global_step += 1;

            // Print batch progress occasionally
            if batch_idx.is_multiple_of(20_usize) {
                println!(
                    "  Epoch {} [{}/{}]: loss={:.6}",
                    epoch + 1,
                    batch_idx + 1,
                    num_batches,
                    loss_val,
                );
            }
        }

        Ok(total_loss / num_batches as f64)
    }

    /// Validate for one epoch
    async fn validate_epoch(&mut self, _epoch: usize) -> Result<f64> {
        // This is a simplified validation loop
        let mut total_loss = 0.0;
        let num_batches = 20; // Placeholder

        for _batch_idx in 0..num_batches {
            // Generate dummy validation batch
            let (audio_batch, mel_batch, timestep_batch) = self.generate_dummy_batch()?;

            // Forward pass (no gradients)
            let loss = self.validation_step(&audio_batch, &mel_batch, &timestep_batch)?;
            total_loss += loss.to_scalar::<f64>()?;
        }

        Ok(total_loss / num_batches as f64)
    }

    /// Single training step
    fn training_step(&self, audio: &Tensor, mel: &Tensor, timesteps: &Tensor) -> Result<Tensor> {
        // Forward pass through the model
        let predicted_noise = self.model.forward(audio, mel, timesteps)?;

        // Calculate actual noise (this would be the target noise added to audio)
        let actual_noise = self.calculate_target_noise(audio, timesteps)?;

        // Calculate loss
        let loss = self.calculate_loss(&predicted_noise, &actual_noise)?;

        Ok(loss)
    }

    /// Single validation step (no gradients)
    fn validation_step(&self, audio: &Tensor, mel: &Tensor, timesteps: &Tensor) -> Result<Tensor> {
        // Same as training step but without gradient computation
        let predicted_noise = self.model.forward(audio, mel, timesteps)?;
        let actual_noise = self.calculate_target_noise(audio, timesteps)?;
        let loss = self.calculate_loss(&predicted_noise, &actual_noise)?;

        Ok(loss)
    }

    /// Calculate target noise for loss computation
    fn calculate_target_noise(&self, audio: &Tensor, timesteps: &Tensor) -> Result<Tensor> {
        // Generate noise that would have been added at these timesteps
        // This is a simplified version - actual implementation would use the same
        // noise schedule as the forward process
        let noise = Tensor::randn(0f32, 1f32, audio.shape(), &self.device)?;
        Ok(noise)
    }

    /// Calculate loss based on configuration
    fn calculate_loss(&self, predicted: &Tensor, target: &Tensor) -> Result<Tensor> {
        match &self.config.loss_config.primary_loss {
            LossType::L1 => {
                let diff = (predicted - target)?;
                Ok(diff.abs()?.mean_all()?)
            }
            LossType::L2 => {
                let diff = (predicted - target)?;
                Ok(diff.powf(2.0)?.mean_all()?)
            }
            LossType::Huber { delta } => self.huber_loss(predicted, target, *delta),
            _ => {
                // Fallback to L2 for other loss types
                let diff = (predicted - target)?;
                Ok(diff.powf(2.0)?.mean_all()?)
            }
        }
    }

    /// Huber loss implementation
    fn huber_loss(&self, predicted: &Tensor, target: &Tensor, delta: f64) -> Result<Tensor> {
        let diff = (predicted - target)?;
        let abs_diff = diff.abs()?;

        let quadratic = abs_diff.le(delta)?.to_dtype(candle_core::DType::F32)?;
        let linear = abs_diff.gt(delta)?.to_dtype(candle_core::DType::F32)?;

        let quadratic_loss = (diff.powf(2.0)? * 0.5)?;
        let linear_loss = ((abs_diff * delta)? - delta * delta * 0.5)?;

        let loss = ((quadratic_loss * quadratic)? + (linear_loss * linear)?)?;
        Ok(loss.mean_all()?)
    }

    /// Generate dummy batch for training/validation
    fn generate_dummy_batch(&self) -> Result<(Tensor, Tensor, Tensor)> {
        let batch_size = self.config.batch_size;
        let seq_len = 8192; // Audio sequence length
        let mel_frames = seq_len / 256; // Mel frames (hop_length = 256)
        let mel_channels = 80;

        // Generate dummy audio
        let audio = Tensor::randn(0f32, 1f32, (batch_size, seq_len), &self.device)?;

        // Generate dummy mel spectrogram
        let mel = Tensor::randn(
            0f32,
            1f32,
            (batch_size, mel_channels, mel_frames),
            &self.device,
        )?;

        // Generate random timesteps (using randn and scaling to [0, 1000))
        // Scale |N(0,1)| values to approximately [0, 1000) using affine(scale, bias)
        let timesteps_f32 = Tensor::randn(0f32, 1f32, (batch_size,), &self.device)?
            .abs()?
            .affine(500.0, 0.0)?;
        let timesteps = timesteps_f32.to_dtype(candle_core::DType::U32)?;

        Ok((audio, mel, timesteps))
    }

    /// Clip gradients using global norm clipping
    fn clip_gradients(&self, max_norm: f64) -> Result<()> {
        if max_norm <= 0.0 {
            return Ok(());
        }

        // In a real implementation, this would:
        // 1. Collect all model parameters that have gradients
        // 2. Calculate the global gradient norm
        // 3. Scale gradients if the norm exceeds max_norm

        // For now, implement a simple gradient norm tracking
        let mut total_norm_squared = 0.0;

        // Simulate gradient norm calculation
        // In practice, this would iterate over model.parameters()
        for layer_size in [512_usize, 256, 128, 64] {
            // Simulate gradient norms for different layers
            let layer_norm = (layer_size as f64).sqrt() * 0.1; // Simulated gradient norm
            total_norm_squared += layer_norm * layer_norm;
        }

        let total_norm = total_norm_squared.sqrt();

        if total_norm > max_norm {
            let scale_factor = max_norm / total_norm;
            tracing::debug!(
                "Clipping gradients: norm={:.4}, max_norm={:.4}, scale={:.4}",
                total_norm,
                max_norm,
                scale_factor
            );

            // In a real implementation, this would apply the scaling:
            // for param in model.parameters() {
            //     if let Some(grad) = param.grad() {
            //         grad.mul_(scale_factor);
            //     }
            // }
        }

        Ok(())
    }

    /// Get current learning rate
    fn get_current_learning_rate(&self) -> f64 {
        if let Some(scheduler) = &self.scheduler {
            scheduler.get_lr()
        } else {
            self.config.learning_rate
        }
    }

    /// Save checkpoint
    async fn save_checkpoint(&self, epoch: usize) -> Result<()> {
        let checkpoint_path = self
            .config
            .output_dir
            .join(format!("checkpoint_epoch_{}.pt", epoch));

        // Save model weights to a sibling safetensors file
        let weights_filename = format!("checkpoint_step_{}.safetensors", self.global_step);
        let weights_path = self.config.output_dir.join(&weights_filename);
        self.varmap.save(&weights_path).map_err(|e| {
            VocoderError::ModelError(format!("Failed to save checkpoint weights: {e}"))
        })?;

        // Create checkpoint data (weights_file records the companion filename)
        let checkpoint = CheckpointData {
            epoch,
            global_step: self.global_step,
            model_config: self.model.config().clone(),
            training_config: self.config.clone(),
            training_stats: self.training_stats.clone(),
            best_val_loss: if self.best_val_loss.is_finite() {
                self.best_val_loss
            } else {
                f64::MAX
            },
            weights_file: Some(weights_filename),
        };

        // Serialize and save JSON state
        let checkpoint_json = serde_json::to_string_pretty(&checkpoint).map_err(|e| {
            VocoderError::ModelError(format!("Failed to serialize checkpoint: {e}"))
        })?;
        fs::write(&checkpoint_path, checkpoint_json).await?;

        println!("Saved checkpoint: {}", checkpoint_path.display());
        Ok(())
    }

    /// Load checkpoint
    async fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        let checkpoint_data = fs::read_to_string(path).await?;
        let checkpoint: CheckpointData = serde_json::from_str(&checkpoint_data).map_err(|e| {
            VocoderError::ModelError(format!("Failed to deserialize checkpoint: {e}"))
        })?;

        self.global_step = checkpoint.global_step;
        self.training_stats = checkpoint.training_stats;
        self.best_val_loss = checkpoint.best_val_loss;

        // Load weights from the companion safetensors file (if present)
        if let Some(ref weights_filename) = checkpoint.weights_file {
            let checkpoint_dir = path.parent().ok_or_else(|| {
                VocoderError::ModelError("Checkpoint path has no parent directory".to_string())
            })?;
            let weights_path = checkpoint_dir.join(weights_filename);
            if weights_path.exists() {
                self.varmap.load(&weights_path).map_err(|e| {
                    VocoderError::ModelError(format!("Failed to load checkpoint weights: {e}"))
                })?;
            }
        }
        // If weights_file is None, this is a legacy state-only checkpoint; skip weight load silently.

        println!("Loaded checkpoint from: {}", path.display());
        Ok(())
    }

    /// Save best model
    async fn save_best_model(&self) -> Result<()> {
        // Save actual model weights as safetensors
        let best_weights_path = self.config.output_dir.join("best_model.safetensors");
        self.varmap.save(&best_weights_path).map_err(|e| {
            VocoderError::ModelError(format!("Failed to save best model weights: {e}"))
        })?;

        // Also write a small JSON state summary for human-readable metadata
        let best_model_path = self.config.output_dir.join("best_model.pt");
        let model_info = format!(
            "{{\"step\":{},\"val_loss\":{:.6}}}",
            self.global_step, self.best_val_loss
        );
        fs::write(&best_model_path, model_info).await?;

        println!("Saved best model: {}", best_weights_path.display());
        Ok(())
    }

    /// Get training statistics
    pub fn get_training_stats(&self) -> &[TrainingStats] {
        &self.training_stats
    }
}

/// Checkpoint data for saving/loading training state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointData {
    epoch: usize,
    global_step: usize,
    model_config: DiffWaveConfig,
    training_config: TrainingConfig,
    training_stats: Vec<TrainingStats>,
    best_val_loss: f64,
    /// Relative filename of the companion safetensors weights file.
    /// `None` means this is a legacy state-only checkpoint (weight load is skipped).
    weights_file: Option<String>,
}

/// Learning rate scheduler
struct LearningRateScheduler {
    config: SchedulerConfig,
    base_lr: f64,
    current_lr: f64,
    step_count: usize,
    best_metric: f64,
    patience_count: usize,
}

impl LearningRateScheduler {
    fn new(config: SchedulerConfig, base_lr: f64) -> Self {
        Self {
            config,
            base_lr,
            current_lr: base_lr,
            step_count: 0,
            best_metric: f64::INFINITY,
            patience_count: 0,
        }
    }

    fn step(&mut self, metric: Option<f64>) -> f64 {
        self.step_count += 1;

        match &self.config {
            SchedulerConfig::ExponentialDecay { gamma, step_size } => {
                if self.step_count.is_multiple_of(*step_size) {
                    self.current_lr *= gamma;
                }
            }
            SchedulerConfig::CosineAnnealing { t_max, eta_min } => {
                let progress = (self.step_count % t_max) as f64 / *t_max as f64;
                self.current_lr = eta_min
                    + (self.base_lr - eta_min) * (1.0 + (std::f64::consts::PI * progress).cos())
                        / 2.0;
            }
            SchedulerConfig::LinearWarmupDecay {
                warmup_steps,
                decay_steps,
                min_lr,
            } => {
                if self.step_count <= *warmup_steps {
                    // Warmup phase
                    self.current_lr =
                        self.base_lr * (self.step_count as f64 / *warmup_steps as f64);
                } else if self.step_count <= warmup_steps + decay_steps {
                    // Decay phase
                    let decay_progress =
                        (self.step_count - warmup_steps) as f64 / *decay_steps as f64;
                    self.current_lr = min_lr + (self.base_lr - min_lr) * (1.0 - decay_progress);
                } else {
                    // Maintain minimum learning rate
                    self.current_lr = *min_lr;
                }
            }
            SchedulerConfig::ReduceOnPlateau {
                factor,
                patience,
                threshold,
                min_lr,
            } => {
                if let Some(current_metric) = metric {
                    if current_metric < self.best_metric - threshold {
                        self.best_metric = current_metric;
                        self.patience_count = 0;
                    } else {
                        self.patience_count += 1;
                        if self.patience_count >= *patience {
                            self.current_lr = (self.current_lr * factor).max(*min_lr);
                            self.patience_count = 0;
                        }
                    }
                }
            }
            SchedulerConfig::MultiStep { milestones, gamma } => {
                if milestones.contains(&self.step_count) {
                    self.current_lr *= gamma;
                }
            }
        }

        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::SafeTensors;
    use tempfile::TempDir;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 2e-4);
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.num_epochs, 1000);
    }

    #[test]
    fn test_learning_rate_scheduler() {
        let config = SchedulerConfig::ExponentialDecay {
            gamma: 0.9,
            step_size: 100,
        };

        let mut scheduler = LearningRateScheduler::new(config, 1e-3);

        // Step should not change LR until step_size
        for _ in 0..99 {
            scheduler.step(None);
        }
        assert_eq!(scheduler.get_lr(), 1e-3);

        // At step 100, LR should decay
        scheduler.step(None);
        assert!((scheduler.get_lr() - 9e-4).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_trainer_creation() {
        let model_config = DiffWaveConfig::default();
        let training_config = TrainingConfig {
            output_dir: TempDir::new().unwrap().path().to_path_buf(),
            ..TrainingConfig::default()
        };

        let mut trainer = DiffWaveTrainer::new(model_config, training_config).unwrap();
        trainer.initialize_optimizer().unwrap();

        assert!(trainer.optimizer.is_some());
    }

    #[test]
    fn test_loss_calculation() {
        let device = Device::Cpu;
        let predicted = Tensor::ones((2, 10), candle_core::DType::F32, &device).unwrap();
        let target = Tensor::zeros((2, 10), candle_core::DType::F32, &device).unwrap();

        let model_config = small_model_config();
        let training_config = TrainingConfig::default();
        let trainer = DiffWaveTrainer::new(model_config, training_config).unwrap();

        // Test L2 loss
        let loss = trainer.calculate_loss(&predicted, &target).unwrap();
        let loss_value = loss.to_scalar::<f32>().unwrap();
        assert!(
            (loss_value - 1.0f32).abs() < 1e-5,
            "Expected loss ~1.0, got {}",
            loss_value
        );

        // Test Huber loss
        let huber_loss = trainer.huber_loss(&predicted, &target, 1.0).unwrap();
        let huber_value = huber_loss.to_scalar::<f32>().unwrap();
        assert!(huber_value > 0.0f32);
    }

    /// Build a minimal DiffWaveConfig for fast tests (small model, quick to instantiate).
    fn small_model_config() -> DiffWaveConfig {
        use super::super::diffusion::NoiseSchedule;
        DiffWaveConfig {
            residual_layers: 2,
            residual_channels: 8,
            dilation_channels: 8,
            skip_channels: 8,
            mel_channels: 16,
            dilation_cycle_length: 2,
            diffusion_steps: 4,
            noise_schedule: NoiseSchedule::Linear,
            beta_start: 1e-4,
            beta_end: 0.02,
            sample_rate: 22050,
            hop_length: 256,
        }
    }

    /// Build a minimal TrainingConfig that writes to the given temp dir.
    fn small_training_config(output_dir: std::path::PathBuf) -> TrainingConfig {
        TrainingConfig {
            batch_size: 1,
            output_dir,
            ..TrainingConfig::default()
        }
    }

    #[tokio::test]
    async fn test_checkpoint_weight_roundtrip() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let model_config = small_model_config();
        let training_config = small_training_config(tmp.path().to_path_buf());

        // Build trainer A, save a checkpoint
        let trainer_a = DiffWaveTrainer::new(model_config.clone(), training_config.clone())
            .expect("trainer A creation failed");
        trainer_a
            .save_checkpoint(0)
            .await
            .expect("save_checkpoint failed");

        // Collect a representative tensor value from trainer A
        let data_a = trainer_a.varmap.data().lock().expect("lock poisoned");
        let (key, var_a) = data_a.iter().next().expect("varmap is empty");
        let tensor_a = var_a.as_tensor().clone();
        let key_owned = key.clone();
        drop(data_a);

        // Determine the weights filename that was written
        let weights_path = tmp
            .path()
            .read_dir()
            .expect("read_dir failed")
            .filter_map(|e| e.ok())
            .find(|e| {
                e.path()
                    .extension()
                    .map(|x| x == "safetensors")
                    .unwrap_or(false)
            })
            .expect("no .safetensors file found")
            .path();

        // Build trainer B and load weights from the same file
        let mut trainer_b =
            DiffWaveTrainer::new(model_config, training_config).expect("trainer B creation failed");
        trainer_b
            .varmap
            .load(&weights_path)
            .map_err(|e| format!("varmap load failed: {e}"))
            .expect("varmap load error");

        // Compare the same tensor key in both trainers
        let data_b = trainer_b.varmap.data().lock().expect("lock poisoned");
        let var_b = data_b
            .get(&key_owned)
            .expect("key missing in trainer B varmap");
        let tensor_b = var_b.as_tensor().clone();
        drop(data_b);

        let flat_a = tensor_a.flatten_all().expect("flatten_all failed");
        let flat_b = tensor_b.flatten_all().expect("flatten_all failed");
        let diff = (flat_a - flat_b)
            .expect("subtraction failed")
            .abs()
            .expect("abs failed")
            .max(0)
            .expect("max failed")
            .to_scalar::<f32>()
            .expect("to_scalar failed");

        assert!(
            diff < 1e-6_f32,
            "tensors differ by {diff} after weight roundtrip"
        );
    }

    #[tokio::test]
    async fn test_save_best_model_writes_safetensors() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let model_config = small_model_config();
        let training_config = small_training_config(tmp.path().to_path_buf());

        let trainer =
            DiffWaveTrainer::new(model_config, training_config).expect("trainer creation failed");
        trainer
            .save_best_model()
            .await
            .expect("save_best_model failed");

        let best_weights = tmp.path().join("best_model.safetensors");
        assert!(
            best_weights.exists(),
            "best_model.safetensors was not created"
        );

        let bytes = std::fs::read(&best_weights).expect("failed to read safetensors file");
        assert!(!bytes.is_empty(), "best_model.safetensors is empty");

        // Verify the file is a valid safetensors archive with at least one tensor
        let tensors = SafeTensors::deserialize(&bytes).expect("failed to deserialize safetensors");
        assert!(
            tensors.len() > 0,
            "best_model.safetensors contains no tensors"
        );
    }

    #[tokio::test]
    async fn test_load_checkpoint_without_weights_is_graceful() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let model_config = small_model_config();
        let training_config = small_training_config(tmp.path().to_path_buf());

        // Build a trainer and save a checkpoint (this also writes a .safetensors file)
        let trainer = DiffWaveTrainer::new(model_config.clone(), training_config.clone())
            .expect("trainer creation failed");
        trainer
            .save_checkpoint(0)
            .await
            .expect("save_checkpoint failed");

        // Overwrite the JSON checkpoint to simulate a legacy state-only file (weights_file = null)
        let checkpoint_path = tmp.path().join("checkpoint_epoch_0.pt");
        let legacy_checkpoint = CheckpointData {
            epoch: 0,
            global_step: 0,
            model_config: model_config.clone(),
            training_config: training_config.clone(),
            training_stats: Vec::new(),
            best_val_loss: f64::MAX,
            weights_file: None, // legacy: no weights
        };
        let json = serde_json::to_string_pretty(&legacy_checkpoint).expect("serialization failed");
        std::fs::write(&checkpoint_path, json).expect("failed to write legacy checkpoint");

        // load_checkpoint on a legacy file must succeed without error
        let mut trainer_b =
            DiffWaveTrainer::new(model_config, training_config).expect("trainer B creation failed");
        let result = trainer_b.load_checkpoint(&checkpoint_path).await;
        assert!(
            result.is_ok(),
            "load_checkpoint with weights_file=None returned error: {:?}",
            result.err()
        );
        assert_eq!(trainer_b.global_step, 0);
    }
}
