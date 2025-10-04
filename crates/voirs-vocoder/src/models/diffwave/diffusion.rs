//! DiffWave diffusion model implementation
//!
//! This module implements the DiffWave vocoder, a neural vocoder based on diffusion probabilistic models.
//! DiffWave generates high-quality audio waveforms from mel spectrograms using a denoising diffusion process.
//!
//! Key features:
//! - U-Net architecture with residual blocks
//! - Support for DDPM (Denoising Diffusion Probabilistic Models) sampling
//! - Support for DDIM (Denoising Diffusion Implicit Models) sampling
//! - Configurable noise schedules and sampling steps
//! - Conditional generation from mel spectrograms

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{conv1d, Conv1d, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{Result, VocoderError};

/// DiffWave model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffWaveConfig {
    /// Number of residual layers in each block
    pub residual_layers: usize,
    /// Number of residual channels
    pub residual_channels: usize,
    /// Number of dilation channels
    pub dilation_channels: usize,
    /// Number of skip channels
    pub skip_channels: usize,
    /// Number of mel spectrogram channels
    pub mel_channels: usize,
    /// Dilation cycle length
    pub dilation_cycle_length: usize,
    /// Number of diffusion steps during training
    pub diffusion_steps: usize,
    /// Noise schedule type
    pub noise_schedule: NoiseSchedule,
    /// Beta start value for noise schedule
    pub beta_start: f64,
    /// Beta end value for noise schedule
    pub beta_end: f64,
    /// Audio sampling rate
    pub sample_rate: u32,
    /// Hop length for mel spectrogram alignment
    pub hop_length: usize,
}

impl Default for DiffWaveConfig {
    fn default() -> Self {
        Self {
            residual_layers: 30,
            residual_channels: 64,
            dilation_channels: 256,
            skip_channels: 256,
            mel_channels: 80,
            dilation_cycle_length: 10,
            diffusion_steps: 1000,
            noise_schedule: NoiseSchedule::Linear,
            beta_start: 1e-4,
            beta_end: 0.02,
            sample_rate: 22050,
            hop_length: 256,
        }
    }
}

/// Noise schedule types for diffusion process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoiseSchedule {
    /// Linear schedule
    Linear,
    /// Cosine schedule
    Cosine,
    /// Exponential schedule
    Exponential,
    /// Custom schedule with specific beta values
    Custom(Vec<f64>),
}

/// Sampling method for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingMethod {
    /// DDPM (Denoising Diffusion Probabilistic Models)
    DDPM {
        /// Number of inference steps
        steps: usize,
    },
    /// DDIM (Denoising Diffusion Implicit Models)
    DDIM {
        /// Number of inference steps
        steps: usize,
        /// Eta parameter for stochasticity control
        eta: f64,
    },
    /// Fast sampling with fewer steps
    FastSampling {
        /// Number of inference steps
        steps: usize,
        /// Acceleration method
        method: FastSamplingMethod,
    },
}

/// Fast sampling acceleration methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FastSamplingMethod {
    /// PNDM (Pseudo Numerical Methods for Diffusion Models)
    PNDM,
    /// DPMSolver
    DPMSolver,
    /// PLMS (Pseudo Linear Multistep)
    PLMS,
}

impl Default for SamplingMethod {
    fn default() -> Self {
        Self::DDIM {
            steps: 50,
            eta: 0.0,
        }
    }
}

/// DiffWave model structure
pub struct DiffWave {
    config: DiffWaveConfig,
    device: Device,

    // Model components
    input_projection: Linear,
    time_embedding: TimeEmbedding,
    residual_layers: Vec<ResidualBlock>,
    skip_projection: Conv1d,
    output_projection: Conv1d,

    // Diffusion parameters
    betas: Tensor,
    alphas: Tensor,
    alpha_bars: Tensor,
    sqrt_alpha_bars: Tensor,
    sqrt_one_minus_alpha_bars: Tensor,
}

impl DiffWave {
    /// Create new DiffWave model
    pub fn new(config: DiffWaveConfig, device: Device, vb: VarBuilder) -> Result<Self> {
        let input_projection =
            candle_nn::linear(1, config.residual_channels, vb.pp("input_projection"))?;

        let time_embedding = TimeEmbedding::new(config.residual_channels, vb.pp("time_embedding"))?;

        let mut residual_layers = Vec::new();
        for i in 0..config.residual_layers {
            let dilation = 2_usize.pow((i % config.dilation_cycle_length) as u32);
            let block = ResidualBlock::new(
                config.residual_channels,
                config.dilation_channels,
                config.skip_channels,
                config.mel_channels,
                dilation,
                vb.pp(format!("residual_layer_{}", i)),
            )?;
            residual_layers.push(block);
        }

        let skip_projection = conv1d(
            config.skip_channels,
            config.skip_channels,
            1,
            Default::default(),
            vb.pp("skip_projection"),
        )?;
        let output_projection = conv1d(
            config.skip_channels,
            1,
            1,
            Default::default(),
            vb.pp("output_projection"),
        )?;

        // Initialize diffusion parameters
        let (betas, alphas, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars) =
            Self::initialize_diffusion_params(&config, &device)?;

        Ok(Self {
            config,
            device,
            input_projection,
            time_embedding,
            residual_layers,
            skip_projection,
            output_projection,
            betas,
            alphas,
            alpha_bars,
            sqrt_alpha_bars,
            sqrt_one_minus_alpha_bars,
        })
    }

    /// Initialize diffusion parameters (betas, alphas, etc.)
    fn initialize_diffusion_params(
        config: &DiffWaveConfig,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let betas = match &config.noise_schedule {
            NoiseSchedule::Linear => {
                let step_size =
                    (config.beta_end - config.beta_start) / (config.diffusion_steps as f64 - 1.0);
                let values: Vec<f32> = (0..config.diffusion_steps)
                    .map(|i| (config.beta_start + step_size * i as f64) as f32)
                    .collect();
                Tensor::from_vec(values, config.diffusion_steps, device)?
            }
            NoiseSchedule::Cosine => {
                let values: Vec<f32> = (0..config.diffusion_steps)
                    .map(|i| {
                        let t = i as f64 / config.diffusion_steps as f64;
                        let alpha_bar = (0.5 * std::f64::consts::PI * (t + 0.008) / 1.008)
                            .cos()
                            .powi(2);
                        let alpha_bar_prev = if i == 0 {
                            1.0
                        } else {
                            let t_prev = (i - 1) as f64 / config.diffusion_steps as f64;
                            (0.5 * std::f64::consts::PI * (t_prev + 0.008) / 1.008)
                                .cos()
                                .powi(2)
                        };
                        ((1.0 - alpha_bar / alpha_bar_prev).min(0.999)) as f32
                    })
                    .collect();
                Tensor::from_vec(values, config.diffusion_steps, device)?
            }
            NoiseSchedule::Exponential => {
                let log_beta_start = config.beta_start.ln();
                let log_beta_end = config.beta_end.ln();
                let values: Vec<f32> = (0..config.diffusion_steps)
                    .map(|i| {
                        let t = i as f64 / (config.diffusion_steps as f64 - 1.0);
                        ((log_beta_start * (1.0 - t) + log_beta_end * t).exp()) as f32
                    })
                    .collect();
                Tensor::from_vec(values, config.diffusion_steps, device)?
            }
            NoiseSchedule::Custom(values) => {
                if values.len() != config.diffusion_steps {
                    return Err(VocoderError::ConfigurationError(
                        "Custom beta schedule length must match diffusion_steps".to_string(),
                    ));
                }
                let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
                Tensor::from_vec(values_f32, config.diffusion_steps, device)?
            }
        };

        let alphas = ((&betas * -1.0)? + 1.0)?;
        // Cumulative product (cumprod not available in Candle, compute manually)
        let alpha_bars = {
            let alpha_vec = alphas.to_vec1::<f32>()?;
            let mut cumul = Vec::with_capacity(alpha_vec.len());
            let mut prod = 1.0f32;
            for &alpha in &alpha_vec {
                prod *= alpha;
                cumul.push(prod);
            }
            Tensor::from_vec(cumul, alpha_vec.len(), device)?
        };
        let sqrt_alpha_bars = alpha_bars.sqrt()?;
        let sqrt_one_minus_alpha_bars = ((&alpha_bars * -1.0)? + 1.0)?.sqrt()?;

        Ok((
            betas,
            alphas,
            alpha_bars,
            sqrt_alpha_bars,
            sqrt_one_minus_alpha_bars,
        ))
    }

    /// Forward pass during training
    pub fn forward(&self, audio: &Tensor, mel: &Tensor, timesteps: &Tensor) -> Result<Tensor> {
        // Add noise to audio based on timesteps
        let (noisy_audio, noise) = self.add_noise(audio, timesteps)?;

        // Predict noise
        let predicted_noise = self.forward_model(&noisy_audio, mel, timesteps)?;

        Ok(predicted_noise)
    }

    /// Forward pass during training - returns both predicted and actual noise
    pub fn forward_with_target(
        &self,
        audio: &Tensor,
        mel: &Tensor,
        timesteps: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Add noise to audio based on timesteps
        let (noisy_audio, actual_noise) = self.add_noise(audio, timesteps)?;

        // Predict noise
        let predicted_noise = self.forward_model(&noisy_audio, mel, timesteps)?;

        Ok((predicted_noise, actual_noise))
    }

    /// Forward pass through the model
    fn forward_model(
        &self,
        noisy_audio: &Tensor,
        mel: &Tensor,
        timesteps: &Tensor,
    ) -> Result<Tensor> {
        // Project input audio: [batch, length] -> [batch, length, 1] -> [batch, length, residual_channels]
        let x = noisy_audio.unsqueeze(2)?; // [batch, length, 1]
        let x = self.input_projection.forward(&x)?; // [batch, length, residual_channels]
        let x = x.transpose(1, 2)?; // [batch, residual_channels, length]

        // Get time embeddings
        let time_emb = self.time_embedding.forward(timesteps)?;

        // Upsample mel to match audio length
        let audio_length = x.dim(2)?;
        let mel_length = mel.dim(2)?;
        let mel_upsampled = if mel_length != audio_length {
            // Simple repeat upsampling
            let upsample_factor = (audio_length + mel_length - 1) / mel_length;
            let mel_repeated = mel.repeat(&[1, 1, upsample_factor])?;
            // Crop to exact length
            mel_repeated.narrow(2, 0, audio_length)?
        } else {
            mel.clone()
        };

        // Process through residual layers
        let mut residual = x;
        let mut skip_connections = Vec::new();

        for layer in &self.residual_layers {
            let (new_residual, skip) = layer.forward(&residual, &mel_upsampled, &time_emb)?;
            residual = new_residual;
            skip_connections.push(skip);
        }

        // Sum skip connections
        let mut skip_sum = skip_connections[0].clone();
        for skip in &skip_connections[1..] {
            skip_sum = (&skip_sum + skip)?;
        }

        // Apply final projections
        let output = self.skip_projection.forward(&skip_sum)?;
        let output = output.relu()?;
        let output = self.output_projection.forward(&output)?;

        Ok(output.squeeze(1)?)
    }

    /// Add noise to audio for training
    fn add_noise(&self, audio: &Tensor, timesteps: &Tensor) -> Result<(Tensor, Tensor)> {
        let batch_size = audio.dims()[0];
        let audio_length = audio.dims()[1];

        // Generate random noise
        let noise = Tensor::randn(0f32, 1f32, audio.shape(), &self.device)?;

        // Get alpha_bar values for the timesteps
        let alpha_bars_t = self.alpha_bars.gather(timesteps, 0)?;
        let sqrt_alpha_bars_t = alpha_bars_t.sqrt()?.reshape((batch_size, 1))?;
        let sqrt_one_minus_alpha_bars_t = ((alpha_bars_t * -1.0)? + 1.0)?
            .sqrt()?
            .reshape((batch_size, 1))?;

        // Broadcast to match audio shape [batch, length]
        let sqrt_alpha_bars_t = sqrt_alpha_bars_t.broadcast_as(audio.shape())?;
        let sqrt_one_minus_alpha_bars_t =
            sqrt_one_minus_alpha_bars_t.broadcast_as(audio.shape())?;

        // Add noise: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        let noisy_audio =
            ((audio * &sqrt_alpha_bars_t)? + (&noise * &sqrt_one_minus_alpha_bars_t)?)?;

        Ok((noisy_audio, noise))
    }

    /// Generate audio from mel spectrogram using DDPM sampling
    pub fn sample_ddpm(&self, mel: &Tensor, steps: usize) -> Result<Tensor> {
        let batch_size = mel.dims()[0];
        let audio_length = mel.dims()[2] * self.config.hop_length;

        // Start with random noise
        let mut x = Tensor::randn(0f32, 1f32, (batch_size, audio_length), &self.device)?;

        // Reverse diffusion process
        for i in (0..steps).rev() {
            let t = Tensor::full(i as f64, (batch_size,), &self.device)?;
            x = self.ddpm_step(&x, mel, &t, i)?;
        }

        Ok(x)
    }

    /// Single DDPM denoising step
    fn ddpm_step(
        &self,
        x: &Tensor,
        mel: &Tensor,
        timesteps: &Tensor,
        step: usize,
    ) -> Result<Tensor> {
        // Predict noise
        let predicted_noise = self.forward_model(x, mel, timesteps)?;

        // Get diffusion parameters
        let alpha_t = self.alphas.get(step)?;
        let alpha_bar_t = self.alpha_bars.get(step)?;
        let beta_t = self.betas.get(step)?;

        // Calculate mean
        let alpha_t_sqrt = alpha_t.sqrt()?;
        let one_minus_alpha_bar_t_sqrt = ((alpha_bar_t * -1.0)? + 1.0)?.sqrt()?;

        let mean =
            ((x - (predicted_noise * &beta_t)?)? / &one_minus_alpha_bar_t_sqrt)? / &alpha_t_sqrt;

        // Add noise for non-final steps
        if step > 0 {
            let noise = Tensor::randn(0f32, 1f32, x.shape(), &self.device)?;
            let sigma = beta_t.sqrt()?;
            let result = (mean + (noise * &sigma)?)?;
            Ok(result)
        } else {
            Ok(mean?)
        }
    }

    /// Generate audio from mel spectrogram using DDIM sampling
    pub fn sample_ddim(&self, mel: &Tensor, steps: usize, eta: f64) -> Result<Tensor> {
        let batch_size = mel.dims()[0];
        let audio_length = mel.dims()[2] * self.config.hop_length;

        // Start with random noise
        let mut x = Tensor::randn(0f32, 1f32, (batch_size, audio_length), &self.device)?;

        // Create timestep schedule for DDIM
        let timestep_schedule: Vec<usize> = (0..self.config.diffusion_steps)
            .step_by(self.config.diffusion_steps / steps)
            .collect();

        // Reverse diffusion process
        for i in (0..timestep_schedule.len()).rev() {
            let t_curr = timestep_schedule[i];
            let t_prev = if i == 0 { 0 } else { timestep_schedule[i - 1] };

            let t = Tensor::full(t_curr as f64, (batch_size,), &self.device)?;
            x = self.ddim_step(&x, mel, &t, t_curr, t_prev, eta)?;
        }

        Ok(x)
    }

    /// Single DDIM denoising step
    fn ddim_step(
        &self,
        x: &Tensor,
        mel: &Tensor,
        timesteps: &Tensor,
        t_curr: usize,
        t_prev: usize,
        eta: f64,
    ) -> Result<Tensor> {
        // Predict noise
        let predicted_noise = self.forward_model(x, mel, timesteps)?;

        // Get alpha bars
        let alpha_bar_curr = self.alpha_bars.get(t_curr)?;
        let alpha_bar_prev = if t_prev == 0 {
            Tensor::ones(&[], DType::F32, &self.device)?
        } else {
            self.alpha_bars.get(t_prev)?
        };

        // Calculate predicted x0
        let sqrt_alpha_bar_curr = alpha_bar_curr.sqrt()?;
        let sqrt_one_minus_alpha_bar_curr = ((alpha_bar_curr * -1.0)? + 1.0)?.sqrt()?;

        let pred_x0 =
            ((x - (&predicted_noise * &sqrt_one_minus_alpha_bar_curr)?)? / &sqrt_alpha_bar_curr)?;

        // Calculate direction
        let sqrt_alpha_bar_prev = alpha_bar_prev.sqrt()?;
        let sqrt_one_minus_alpha_bar_prev = ((alpha_bar_prev * -1.0)? + 1.0)?.sqrt()?;

        let direction = ((&predicted_noise * &sqrt_one_minus_alpha_bar_prev)? * eta)?;

        // Calculate result
        let result = ((pred_x0 * &sqrt_alpha_bar_prev)? + direction)?;

        Ok(result)
    }

    /// Get model configuration
    pub fn config(&self) -> &DiffWaveConfig {
        &self.config
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        // This would calculate the actual number of parameters
        // For now, return an estimate
        let residual_params = self.config.residual_layers
            * (self.config.residual_channels * self.config.dilation_channels * 2
                + self.config.residual_channels * self.config.skip_channels);
        let projection_params = self.config.residual_channels + self.config.skip_channels * 2;

        residual_params + projection_params
    }

    /// Load model from SafeTensors checkpoint
    pub fn load_from_safetensors(
        checkpoint_path: &std::path::Path,
        device: Device,
    ) -> Result<Self> {
        use safetensors::SafeTensors;

        // Read checkpoint files
        let safetensors_path = checkpoint_path;
        let json_path = checkpoint_path.with_extension("json");

        // Load metadata from JSON
        let metadata_str = std::fs::read_to_string(&json_path).map_err(|e| {
            VocoderError::ConfigError(format!("Failed to read checkpoint metadata: {}", e))
        })?;

        let metadata: serde_json::Value = serde_json::from_str(&metadata_str).map_err(|e| {
            VocoderError::ConfigError(format!("Failed to parse checkpoint metadata: {}", e))
        })?;

        // Extract model type and validate
        let model_type = metadata
            .get("model_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                VocoderError::ConfigError("Missing model_type in checkpoint".to_string())
            })?;

        if model_type != "DiffWave" {
            return Err(VocoderError::ConfigError(format!(
                "Expected DiffWave model, got {}",
                model_type
            )));
        }

        // Load SafeTensors file
        let safetensors_data = std::fs::read(safetensors_path).map_err(|e| {
            VocoderError::ConfigError(format!("Failed to read SafeTensors file: {}", e))
        })?;

        let tensors = SafeTensors::deserialize(&safetensors_data).map_err(|e| {
            VocoderError::ConfigError(format!("Failed to deserialize SafeTensors: {}", e))
        })?;

        // Create default config (in production, this should come from checkpoint metadata)
        let config = DiffWaveConfig::default();

        // Create VarBuilder from SafeTensors
        let vb = Self::create_var_builder_from_safetensors(&tensors, &device)?;

        // Create model using the loaded tensors
        Self::new(config, device, vb)
    }

    /// Create VarBuilder from SafeTensors data
    fn create_var_builder_from_safetensors<'a>(
        tensors: &'a safetensors::SafeTensors<'a>,
        device: &Device,
    ) -> Result<VarBuilder<'a>> {
        // Extract tensor data and convert to Candle tensors
        let mut tensor_map = std::collections::HashMap::new();

        for tensor_name in tensors.names() {
            let tensor_view = tensors.tensor(tensor_name).map_err(|e| {
                VocoderError::ConfigError(format!("Failed to get tensor {}: {}", tensor_name, e))
            })?;

            // Convert SafeTensors tensor to Candle tensor
            let shape: Vec<usize> = tensor_view.shape().to_vec();
            let data = tensor_view.data();

            // Convert bytes to f32 values
            let f32_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let tensor = Tensor::from_vec(f32_data, shape.as_slice(), device)?;
            tensor_map.insert(tensor_name.to_string(), tensor);
        }

        // Create VarBuilder from the tensor map
        let vb = candle_nn::VarBuilder::from_tensors(tensor_map, DType::F32, device);

        Ok(vb)
    }

    /// Perform inference: convert mel spectrogram to audio waveform
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram tensor with shape [batch, mel_channels, time]
    /// * `method` - Sampling method to use (DDPM or DDIM)
    ///
    /// # Returns
    /// Audio waveform tensor with shape [batch, samples]
    pub fn inference(&self, mel: &Tensor, method: SamplingMethod) -> Result<Tensor> {
        match method {
            SamplingMethod::DDPM { steps } => self.sample_ddpm(mel, steps),
            SamplingMethod::DDIM { steps, eta } => self.sample_ddim(mel, steps, eta),
            SamplingMethod::FastSampling { steps, .. } => {
                // For fast sampling, use DDIM with eta=0
                self.sample_ddim(mel, steps, 0.0)
            }
        }
    }
}

/// Time embedding layer for diffusion timesteps
struct TimeEmbedding {
    linear1: Linear,
    linear2: Linear,
    embedding_dim: usize,
}

impl TimeEmbedding {
    fn new(embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear(embedding_dim, embedding_dim * 4, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(embedding_dim * 4, embedding_dim, vb.pp("linear2"))?;

        Ok(Self {
            linear1,
            linear2,
            embedding_dim,
        })
    }

    fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        // Convert timesteps to f32 for embedding calculation
        let timesteps_f32 = timesteps.to_dtype(candle_core::DType::F32)?;

        // Create sinusoidal position embeddings
        let half_dim = self.embedding_dim / 2;
        let emb = (10000.0_f32).ln() / (half_dim - 1) as f32;

        let mut embeddings = Vec::new();
        for i in 0..half_dim {
            let freq = (-emb * i as f32).exp();
            embeddings.push(freq);
        }

        let emb_tensor = Tensor::from_vec(embeddings, half_dim, timesteps.device())?;
        let emb_tensor = emb_tensor.unsqueeze(0)?; // [1, half_dim]
        let expanded_timesteps = timesteps_f32.unsqueeze(1)?; // [batch, 1]

        // Get batch size for broadcasting
        let batch_size = expanded_timesteps.dims()[0];

        // Broadcast emb_tensor to [batch, half_dim]
        let emb_broadcast = emb_tensor.broadcast_as((batch_size, half_dim))?;

        // Multiply and apply sin/cos
        let sin_emb =
            (&expanded_timesteps.broadcast_as((batch_size, half_dim))? * &emb_broadcast)?.sin()?;
        let cos_emb =
            (&expanded_timesteps.broadcast_as((batch_size, half_dim))? * &emb_broadcast)?.cos()?;

        let full_emb = Tensor::cat(&[sin_emb, cos_emb], 1)?;

        let h = self.linear1.forward(&full_emb)?.silu()?;
        let h = self.linear2.forward(&h)?;

        Ok(h)
    }
}

/// Residual block with dilation convolution
struct ResidualBlock {
    conv_filter: Conv1d,
    conv_gate: Conv1d,
    conv_residual: Conv1d,
    conv_skip: Conv1d,
    conv_mel: Conv1d,
    conv_time: Linear,
}

impl ResidualBlock {
    fn new(
        residual_channels: usize,
        dilation_channels: usize,
        skip_channels: usize,
        mel_channels: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_config = candle_nn::Conv1dConfig {
            padding: dilation,
            dilation,
            ..Default::default()
        };

        let conv_filter = conv1d(
            residual_channels,
            dilation_channels,
            3,
            conv_config,
            vb.pp("conv_filter"),
        )?;
        let conv_gate = conv1d(
            residual_channels,
            dilation_channels,
            3,
            conv_config,
            vb.pp("conv_gate"),
        )?;
        let conv_residual = conv1d(
            dilation_channels,
            residual_channels,
            1,
            Default::default(),
            vb.pp("conv_residual"),
        )?;
        let conv_skip = conv1d(
            dilation_channels,
            skip_channels,
            1,
            Default::default(),
            vb.pp("conv_skip"),
        )?;
        let conv_mel = conv1d(
            mel_channels,
            dilation_channels * 2,
            1,
            Default::default(),
            vb.pp("conv_mel"),
        )?;
        let conv_time =
            candle_nn::linear(residual_channels, dilation_channels * 2, vb.pp("conv_time"))?;

        Ok(Self {
            conv_filter,
            conv_gate,
            conv_residual,
            conv_skip,
            conv_mel,
            conv_time,
        })
    }

    fn forward(&self, x: &Tensor, mel: &Tensor, time_emb: &Tensor) -> Result<(Tensor, Tensor)> {
        let residual = x.clone();

        // Apply convolutions
        let filter_output = self.conv_filter.forward(x)?;
        let gate_output = self.conv_gate.forward(x)?;

        // Add mel conditioning
        let mel_output = self.conv_mel.forward(mel)?;
        let filter_with_mel =
            (&filter_output + &mel_output.narrow(1, 0, filter_output.dim(1)?)?)?;
        let gate_with_mel =
            (&gate_output + &mel_output.narrow(1, filter_output.dim(1)?, gate_output.dim(1)?)?)?;

        // Add time conditioning
        let time_output = self.conv_time.forward(time_emb)?.unsqueeze(2)?; // [batch, channels, 1]
                                                                           // Broadcast to match audio length
        let audio_length = filter_with_mel.dim(2)?;
        let time_output =
            time_output.broadcast_as((time_output.dim(0)?, time_output.dim(1)?, audio_length))?;

        let filter_with_time =
            (&filter_with_mel + &time_output.narrow(1, 0, filter_with_mel.dim(1)?)?)?;
        let gate_with_time = (&gate_with_mel
            + &time_output.narrow(1, filter_with_mel.dim(1)?, gate_with_mel.dim(1)?)?)?;

        // Apply gating (sigmoid(x) = 1 / (1 + exp(-x)))
        let sigmoid_gate = ((gate_with_time.neg()?.exp()? + 1.0)?.recip())?;
        let gated = (filter_with_time.tanh()? * sigmoid_gate)?;

        // Generate residual and skip outputs
        let residual_output = self.conv_residual.forward(&gated)?;
        let skip_output = self.conv_skip.forward(&gated)?;

        let new_residual = (&residual + &residual_output)?;

        Ok((new_residual, skip_output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_diffwave_config_default() {
        let config = DiffWaveConfig::default();
        assert_eq!(config.residual_layers, 30);
        assert_eq!(config.residual_channels, 64);
        assert_eq!(config.mel_channels, 80);
        assert_eq!(config.diffusion_steps, 1000);
    }

    #[test]
    fn test_noise_schedule_linear() {
        let config = DiffWaveConfig {
            diffusion_steps: 10,
            beta_start: 0.0001,
            beta_end: 0.02,
            noise_schedule: NoiseSchedule::Linear,
            ..Default::default()
        };

        let device = Device::Cpu;
        let (betas, _, _, _, _) = DiffWave::initialize_diffusion_params(&config, &device).unwrap();

        assert_eq!(betas.dims(), &[10]);
        // First beta should be close to beta_start
        let first_beta = betas.get(0).unwrap().to_scalar::<f64>().unwrap();
        assert!((first_beta - config.beta_start).abs() < 1e-6);
    }

    #[test]
    fn test_sampling_method_default() {
        let method = SamplingMethod::default();
        match method {
            SamplingMethod::DDIM { steps, eta } => {
                assert_eq!(steps, 50);
                assert_eq!(eta, 0.0);
            }
            _ => panic!("Default sampling method should be DDIM"),
        }
    }
}
