//! VITS Posterior Encoder implementation
//!
//! The posterior encoder processes mel spectrograms to compute the posterior distribution
//! for the latent variables in the VAE framework.

use candle_core::{Tensor, Device, DType, Result as CandleResult};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder, Module, linear, layer_norm};
use serde::{Deserialize, Serialize};

use crate::{Result, AcousticError};

/// Configuration for VITS posterior encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PosteriorConfig {
    /// Number of mel channels
    pub n_mel_channels: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of CNN layers
    pub n_layers: usize,
    /// Kernel size for convolutions
    pub kernel_size: usize,
    /// Stride for convolutions
    pub stride: usize,
    /// Latent dimension
    pub latent_dim: usize,
    /// Dropout probability
    pub dropout: f64,
}

impl Default for PosteriorConfig {
    fn default() -> Self {
        Self {
            n_mel_channels: 80,
            hidden_dim: 192,
            n_layers: 16,
            kernel_size: 5,
            stride: 1,
            latent_dim: 192,
            dropout: 0.0,
        }
    }
}

/// Residual convolutional block for posterior encoder
pub struct ResidualBlock {
    conv1: Conv1d,
    conv2: Conv1d,
    norm1: candle_nn::LayerNorm,
    norm2: candle_nn::LayerNorm,
    dropout: f64,
}

impl ResidualBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dropout: f64,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let conv_config = Conv1dConfig {
            padding: kernel_size / 2,
            stride: 1,
            ..Default::default()
        };
        
        let conv1 = candle_nn::conv1d(
            in_channels,
            out_channels,
            kernel_size,
            conv_config,
            vb.pp("conv1"),
        )?;
        
        let conv2 = candle_nn::conv1d(
            out_channels,
            out_channels,
            kernel_size,
            conv_config,
            vb.pp("conv2"),
        )?;
        
        let norm1 = layer_norm(out_channels, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(out_channels, 1e-5, vb.pp("norm2"))?;
        
        Ok(Self {
            conv1,
            conv2,
            norm1,
            norm2,
            dropout,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x.clone();
        
        // First convolution + norm + activation
        let mut out = self.conv1.forward(x)?;
        out = self.norm1.forward(&out)?;
        out = out.relu()?;
        
        // Apply dropout if training
        if self.dropout > 0.0 {
            out = candle_nn::ops::dropout(&out, self.dropout as f32)?;
        }
        
        // Second convolution + norm
        out = self.conv2.forward(&out)?;
        out = self.norm2.forward(&out)?;
        
        // Residual connection
        let out = if out.dims() == residual.dims() {
            (&out + &residual)?
        } else {
            out
        };
        
        out.relu()
    }
}

/// VITS Posterior Encoder
pub struct PosteriorEncoder {
    config: PosteriorConfig,
    device: Device,
    
    // Network layers
    pre_conv: Conv1d,
    blocks: Vec<ResidualBlock>,
    post_conv: Conv1d,
    
    // Output projections for mean and log variance
    mean_proj: candle_nn::Linear,
    logvar_proj: candle_nn::Linear,
}

impl PosteriorEncoder {
    pub fn new(config: PosteriorConfig, device: Device) -> Result<Self> {
        let vs = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vs, DType::F32, &device);
        
        Self::load_with_varbuilder(config, device, vb)
    }
    
    pub fn load_with_varbuilder(
        config: PosteriorConfig,
        device: Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Pre-convolution to project mel channels to hidden dimension
        let pre_conv_config = Conv1dConfig {
            padding: config.kernel_size / 2,
            stride: 1,
            ..Default::default()
        };
        
        let pre_conv = candle_nn::conv1d(
            config.n_mel_channels,
            config.hidden_dim,
            config.kernel_size,
            pre_conv_config,
            vb.pp("pre_conv"),
        ).map_err(|e| AcousticError::ModelError(format!("Failed to create pre_conv: {}", e)))?;
        
        // Residual blocks
        let mut blocks = Vec::new();
        for i in 0..config.n_layers {
            let block = ResidualBlock::new(
                config.hidden_dim,
                config.hidden_dim,
                config.kernel_size,
                config.dropout,
                vb.pp(&format!("block_{}", i)),
            ).map_err(|e| AcousticError::ModelError(format!("Failed to create block {}: {}", i, e)))?;
            blocks.push(block);
        }
        
        // Post-convolution to reduce dimension
        let post_conv = candle_nn::conv1d(
            config.hidden_dim,
            config.latent_dim,
            config.kernel_size,
            pre_conv_config,
            vb.pp("post_conv"),
        ).map_err(|e| AcousticError::ModelError(format!("Failed to create post_conv: {}", e)))?;
        
        // Output projections
        let mean_proj = linear(
            config.latent_dim,
            config.latent_dim,
            vb.pp("mean_proj"),
        ).map_err(|e| AcousticError::ModelError(format!("Failed to create mean_proj: {}", e)))?;
        
        let logvar_proj = linear(
            config.latent_dim,
            config.latent_dim,
            vb.pp("logvar_proj"),
        ).map_err(|e| AcousticError::ModelError(format!("Failed to create logvar_proj: {}", e)))?;
        
        Ok(Self {
            config,
            device,
            pre_conv,
            blocks,
            post_conv,
            mean_proj,
            logvar_proj,
        })
    }
    
    /// Forward pass through the posterior encoder
    /// 
    /// # Arguments
    /// * `mel` - Mel spectrogram tensor with shape [batch_size, n_mel_channels, n_frames]
    /// 
    /// # Returns
    /// * `(mean, logvar)` - Mean and log variance tensors for VAE posterior
    pub fn forward(&self, mel: &Tensor) -> Result<(Tensor, Tensor)> {
        // Validate input shape
        let input_shape = mel.dims();
        if input_shape.len() != 3 {
            return Err(AcousticError::InputError(
                format!("Expected 3D tensor [batch, mel_channels, frames], got {:?}", input_shape)
            ));
        }
        
        let (batch_size, n_mel_channels, n_frames) = mel.dims3()
            .map_err(|e| AcousticError::ModelError(format!("Failed to get tensor dimensions: {}", e)))?;
        
        if n_mel_channels != self.config.n_mel_channels {
            return Err(AcousticError::InputError(
                format!("Expected {} mel channels, got {}", self.config.n_mel_channels, n_mel_channels)
            ));
        }
        
        tracing::debug!(
            "PosteriorEncoder forward: input shape [{}, {}, {}]",
            batch_size, n_mel_channels, n_frames
        );
        
        // Pre-convolution
        let mut x = self.pre_conv.forward(mel)
            .map_err(|e| AcousticError::ModelError(format!("Pre-convolution failed: {}", e)))?;
        
        tracing::debug!("After pre_conv: {:?}", x.dims());
        
        // Apply residual blocks
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x)
                .map_err(|e| AcousticError::ModelError(format!("Block {} failed: {}", i, e)))?;
        }
        
        tracing::debug!("After residual blocks: {:?}", x.dims());
        
        // Post-convolution
        x = self.post_conv.forward(&x)
            .map_err(|e| AcousticError::ModelError(format!("Post-convolution failed: {}", e)))?;
        
        tracing::debug!("After post_conv: {:?}", x.dims());
        
        // Apply activation
        x = x.relu()
            .map_err(|e| AcousticError::ModelError(format!("Activation failed: {}", e)))?;
        
        // Global average pooling over time dimension
        let x = x.mean(2)
            .map_err(|e| AcousticError::ModelError(format!("Global average pooling failed: {}", e)))?;
        
        tracing::debug!("After global pooling: {:?}", x.dims());
        
        // Project to mean and log variance
        let mean = self.mean_proj.forward(&x)
            .map_err(|e| AcousticError::ModelError(format!("Mean projection failed: {}", e)))?;
        
        let logvar = self.logvar_proj.forward(&x)
            .map_err(|e| AcousticError::ModelError(format!("Log variance projection failed: {}", e)))?;
        
        tracing::debug!("Output shapes - mean: {:?}, logvar: {:?}", mean.dims(), logvar.dims());
        
        Ok((mean, logvar))
    }
    
    /// Sample from the posterior distribution using the reparameterization trick
    /// 
    /// # Arguments
    /// * `mean` - Mean tensor from posterior
    /// * `logvar` - Log variance tensor from posterior
    /// * `device` - Device to create tensors on
    /// 
    /// # Returns
    /// * Sampled latent tensor
    pub fn sample(&self, mean: &Tensor, logvar: &Tensor, device: &Device) -> Result<Tensor> {
        let std = (logvar * 0.5)
            .map_err(|e| AcousticError::ModelError(format!("Log variance processing failed: {}", e)))?
            .exp()
            .map_err(|e| AcousticError::ModelError(format!("Exponential failed: {}", e)))?;
        
        // Sample from standard normal distribution
        let shape = mean.dims();
        let eps = Tensor::randn(0f32, 1f32, shape, device)
            .map_err(|e| AcousticError::ModelError(format!("Random sampling failed: {}", e)))?;
        
        // Reparameterization trick: z = mean + std * eps
        let mult_result = (&std * &eps)
            .map_err(|e| AcousticError::ModelError(format!("Multiplication failed: {}", e)))?;
        let z = (mean + mult_result)
            .map_err(|e| AcousticError::ModelError(format!("Reparameterization failed: {}", e)))?;
        
        Ok(z)
    }
    
    /// Compute KL divergence between posterior and prior (standard normal)
    /// 
    /// # Arguments
    /// * `mean` - Mean tensor from posterior
    /// * `logvar` - Log variance tensor from posterior
    /// 
    /// # Returns
    /// * KL divergence scalar
    pub fn kl_divergence(&self, mean: &Tensor, logvar: &Tensor) -> Result<Tensor> {
        // KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        let mean_sq = mean.sqr()
            .map_err(|e| AcousticError::ModelError(format!("Mean squared failed: {}", e)))?;
        
        let exp_logvar = logvar.exp()
            .map_err(|e| AcousticError::ModelError(format!("Exponential failed: {}", e)))?;
        
        let ones = Tensor::ones(logvar.dims(), DType::F32, &self.device)
            .map_err(|e| AcousticError::ModelError(format!("Ones tensor creation failed: {}", e)))?;
        
        let kl = (&ones + logvar - &mean_sq - &exp_logvar)
            .map_err(|e| AcousticError::ModelError(format!("KL computation failed: {}", e)))?;
        let kl = (kl.sum_all()
            .map_err(|e| AcousticError::ModelError(format!("Sum failed: {}", e)))? * (-0.5))
            .map_err(|e| AcousticError::ModelError(format!("Multiplication failed: {}", e)))?;
        
        Ok(kl)
    }
}