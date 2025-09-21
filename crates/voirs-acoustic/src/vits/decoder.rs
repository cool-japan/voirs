//! VITS Decoder/Generator implementation
//!
//! The decoder generates mel spectrograms from latent representations using
//! transposed convolutions and multi-receptive field (MRF) blocks.

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::{AcousticError, Result};

/// Configuration for VITS decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderConfig {
    /// Input latent dimension
    pub latent_dim: usize,
    /// Number of mel channels
    pub n_mel_channels: usize,
    /// Number of upsampling layers
    pub n_layers: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Kernel size for convolutions
    pub kernel_size: usize,
    /// Dilation rates for residual blocks
    pub dilations: Vec<Vec<usize>>,
    /// Upsampling factors
    pub upsample_rates: Vec<usize>,
    /// Initial upsampling kernel size
    pub upsample_kernel_sizes: Vec<usize>,
    /// Dropout probability
    pub dropout: f64,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            latent_dim: 80, // Match normalizing flows output channels (mel_channels)
            n_mel_channels: 80,
            n_layers: 6,
            hidden_dim: 256, // Reduce hidden dimension
            kernel_size: 3,  // Smaller kernel size
            dilations: vec![
                vec![1, 3], // Simpler dilation pattern
                vec![1, 3],
            ],
            upsample_rates: vec![2, 2, 2], // Simpler upsampling - total factor 8x
            upsample_kernel_sizes: vec![4, 4, 4], // Smaller kernels
            dropout: 0.0,
        }
    }
}

/// Multi-Receptive Field (MRF) block
pub struct MRFBlock {
    convs: Vec<Conv1d>,
    dropout: f64,
}

impl MRFBlock {
    pub fn new(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        dropout: f64,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let mut convs = Vec::new();

        for (i, &dilation) in dilations.iter().enumerate() {
            let padding = (kernel_size - 1) * dilation / 2;
            let conv_config = Conv1dConfig {
                padding,
                stride: 1,
                dilation,
                ..Default::default()
            };

            let conv = candle_nn::conv1d(
                channels,
                channels,
                kernel_size,
                conv_config,
                vb.pp(format!("conv_{i}")),
            )?;
            convs.push(conv);
        }

        Ok(Self { convs, dropout })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut outputs = Vec::new();

        for conv in &self.convs {
            let mut h = conv.forward(x)?;
            h = h.relu()?;

            // Apply dropout
            if self.dropout > 0.0 {
                h = candle_nn::ops::dropout(&h, self.dropout as f32)?;
            }

            outputs.push(h);
        }

        // Sum all outputs
        let mut result = outputs[0].clone();
        for output in outputs.iter().skip(1) {
            result = (&result + output)?;
        }

        Ok(result)
    }
}

/// Residual MRF block with skip connections
pub struct ResidualMRFBlock {
    mrf: MRFBlock,
    skip_conv: Conv1d,
}

impl ResidualMRFBlock {
    pub fn new(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        dropout: f64,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let mrf = MRFBlock::new(channels, kernel_size, dilations, dropout, vb.pp("mrf"))?;

        let skip_conv =
            candle_nn::conv1d(channels, channels, 1, Default::default(), vb.pp("skip"))?;

        Ok(Self { mrf, skip_conv })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x.clone();
        let h = self.mrf.forward(x)?;
        let skip = self.skip_conv.forward(&h)?;

        // Residual connection
        let output = (&residual + &skip)?;
        Ok(output)
    }
}

/// Transposed convolution for upsampling
pub struct UpsampleLayer {
    conv_transpose: Conv1d,
    mrf_blocks: Vec<ResidualMRFBlock>,
}

impl UpsampleLayer {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilations: &[Vec<usize>],
        dropout: f64,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        // For transposed convolutions, padding calculation is different
        // Use safe calculation to avoid underflow
        let padding = if kernel_size >= stride {
            (kernel_size - stride) / 2
        } else {
            0
        };
        let conv_config = Conv1dConfig {
            padding,
            stride,
            ..Default::default()
        };

        // Create transposed convolution for upsampling
        let conv_transpose = candle_nn::conv1d(
            in_channels,
            out_channels,
            kernel_size,
            conv_config,
            vb.pp("upsample"),
        )?;

        // Create MRF blocks
        let mut mrf_blocks = Vec::new();
        for (i, dilation_set) in dilations.iter().enumerate() {
            let block = ResidualMRFBlock::new(
                out_channels,
                7, // kernel size for MRF
                dilation_set,
                dropout,
                vb.pp(format!("mrf_{i}")),
            )?;
            mrf_blocks.push(block);
        }

        Ok(Self {
            conv_transpose,
            mrf_blocks,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Upsample using transposed convolution
        let mut h = self.conv_transpose.forward(x)?;
        h = h.relu()?;

        // Apply MRF blocks
        for mrf_block in &self.mrf_blocks {
            h = mrf_block.forward(&h)?;
        }

        Ok(h)
    }
}

/// VITS Decoder/Generator
pub struct Decoder {
    config: DecoderConfig,
    #[allow(dead_code)]
    device: Device,

    // Network layers
    pre_conv: Conv1d,
    upsample_layers: Vec<UpsampleLayer>,
    post_conv: Conv1d,
}

impl Decoder {
    pub fn new(config: DecoderConfig, device: Device) -> Result<Self> {
        let vs = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vs, DType::F32, &device);

        Self::load_with_varbuilder(config, device, vb)
    }

    pub fn load_with_varbuilder(
        config: DecoderConfig,
        device: Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Pre-convolution to project latent to hidden dimension
        let pre_conv = candle_nn::conv1d(
            config.latent_dim,
            config.hidden_dim,
            config.kernel_size,
            Conv1dConfig {
                padding: config.kernel_size / 2,
                stride: 1,
                ..Default::default()
            },
            vb.pp("pre_conv"),
        )
        .map_err(|e| AcousticError::ModelError(format!("Failed to create pre_conv: {e}")))?;

        // Create upsampling layers
        let mut upsample_layers = Vec::new();
        let mut current_channels = config.hidden_dim;

        for (i, (&upsample_rate, &kernel_size)) in config
            .upsample_rates
            .iter()
            .zip(config.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let out_channels = current_channels / 2;

            let layer = UpsampleLayer::new(
                current_channels,
                out_channels,
                kernel_size,
                upsample_rate,
                &config.dilations,
                config.dropout,
                vb.pp(format!("upsample_{i}")),
            )
            .map_err(|e| {
                AcousticError::ModelError(format!("Failed to create upsample layer {i}: {e}"))
            })?;

            upsample_layers.push(layer);
            current_channels = out_channels;
        }

        // Post-convolution to generate mel spectrogram
        let post_conv = candle_nn::conv1d(
            current_channels,
            config.n_mel_channels,
            config.kernel_size,
            Conv1dConfig {
                padding: config.kernel_size / 2,
                stride: 1,
                ..Default::default()
            },
            vb.pp("post_conv"),
        )
        .map_err(|e| AcousticError::ModelError(format!("Failed to create post_conv: {e}")))?;

        Ok(Self {
            config,
            device,
            pre_conv,
            upsample_layers,
            post_conv,
        })
    }

    /// Generate mel spectrogram from latent representation
    ///
    /// # Arguments
    /// * `z` - Latent tensor with shape [batch_size, latent_dim, n_frames]
    ///
    /// # Returns
    /// * Mel spectrogram tensor with shape [batch_size, n_mel_channels, upsampled_frames]
    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        // Validate input shape
        let input_shape = z.dims();
        if input_shape.len() != 3 {
            return Err(AcousticError::InputError(format!(
                "Expected 3D tensor [batch, latent_dim, frames], got {input_shape:?}"
            )));
        }

        let (batch_size, latent_dim, n_frames) = z.dims3().map_err(|e| {
            AcousticError::ModelError(format!("Failed to get tensor dimensions: {e}"))
        })?;

        if latent_dim != self.config.latent_dim {
            return Err(AcousticError::InputError(format!(
                "Expected {} latent dimensions, got {latent_dim}",
                self.config.latent_dim
            )));
        }

        tracing::debug!("Decoder forward: input shape [{batch_size}, {latent_dim}, {n_frames}]");

        // Pre-convolution
        let mut h = self
            .pre_conv
            .forward(z)
            .map_err(|e| AcousticError::ModelError(format!("Pre-convolution failed: {e}")))?;

        tracing::debug!("After pre_conv: {:?}", h.dims());

        // Apply upsampling layers
        for (i, layer) in self.upsample_layers.iter().enumerate() {
            h = layer.forward(&h).map_err(|e| {
                AcousticError::ModelError(format!("Upsample layer {i} failed: {e}"))
            })?;

            tracing::debug!("After upsample layer {i}: {:?}", h.dims());
        }

        // Post-convolution to generate mel spectrogram
        let mel = self
            .post_conv
            .forward(&h)
            .map_err(|e| AcousticError::ModelError(format!("Post-convolution failed: {e}")))?;

        // Apply tanh activation to keep values in reasonable range for mel spectrograms
        let mel = mel
            .tanh()
            .map_err(|e| AcousticError::ModelError(format!("Tanh activation failed: {e}")))?;

        tracing::debug!("Output mel spectrogram shape: {:?}", mel.dims());

        Ok(mel)
    }

    /// Get the total upsampling factor
    pub fn upsample_factor(&self) -> usize {
        self.config.upsample_rates.iter().product()
    }

    /// Predict the output size given input size
    pub fn predict_output_size(&self, input_frames: usize) -> usize {
        input_frames * self.upsample_factor()
    }
}
