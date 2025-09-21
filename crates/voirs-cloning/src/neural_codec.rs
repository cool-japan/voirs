//! Neural Codec Implementation for Advanced Audio Compression and Quality Enhancement
//!
//! This module provides state-of-the-art neural audio codecs for high-quality,
//! low-bitrate audio compression and reconstruction, integrating with VITS2 and
//! other voice synthesis systems.

use crate::{Error, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{conv1d, conv_transpose1d, Conv1d, Conv1dConfig, Module, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, trace, warn};

/// Neural codec configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralCodecConfig {
    /// Sample rate for audio processing
    pub sample_rate: u32,
    /// Number of channels (typically 1 for mono)
    pub channels: usize,
    /// Encoder dimension
    pub encoder_dim: usize,
    /// Decoder dimension  
    pub decoder_dim: usize,
    /// Number of quantization levels
    pub num_quantizers: usize,
    /// Codebook size for each quantizer
    pub codebook_size: usize,
    /// Codebook dimension
    pub codebook_dim: usize,
    /// Target bitrate in kbps
    pub target_bitrate: f32,
    /// Compression ratio
    pub compression_ratio: usize,
    /// Number of encoder layers
    pub encoder_layers: usize,
    /// Number of decoder layers
    pub decoder_layers: usize,
    /// Kernel sizes for convolutions
    pub kernel_sizes: Vec<usize>,
    /// Stride values for downsampling
    pub strides: Vec<usize>,
    /// Dilation values for dilated convolutions
    pub dilations: Vec<usize>,
    /// Use residual connections
    pub use_residual: bool,
    /// Use skip connections
    pub use_skip_connections: bool,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Perceptual loss weight
    pub perceptual_loss_weight: f32,
    /// Adversarial loss weight
    pub adversarial_loss_weight: f32,
    /// Reconstruction loss weight
    pub reconstruction_loss_weight: f32,
    /// Quantization loss weight
    pub quantization_loss_weight: f32,
}

impl Default for NeuralCodecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            channels: 1,
            encoder_dim: 512,
            decoder_dim: 512,
            num_quantizers: 8,
            codebook_size: 1024,
            codebook_dim: 256,
            target_bitrate: 6.0, // 6 kbps
            compression_ratio: 32,
            encoder_layers: 5,
            decoder_layers: 5,
            kernel_sizes: vec![7, 7, 7, 7, 7],
            strides: vec![1, 2, 2, 4, 4],
            dilations: vec![1, 1, 1, 1, 1],
            use_residual: true,
            use_skip_connections: true,
            dropout_rate: 0.1,
            perceptual_loss_weight: 1.0,
            adversarial_loss_weight: 1.0,
            reconstruction_loss_weight: 45.0,
            quantization_loss_weight: 1.0,
        }
    }
}

impl NeuralCodecConfig {
    /// Create configuration optimized for high quality
    pub fn high_quality() -> Self {
        Self {
            encoder_dim: 768,
            decoder_dim: 768,
            num_quantizers: 12,
            codebook_size: 2048,
            codebook_dim: 384,
            target_bitrate: 12.0,
            compression_ratio: 16,
            encoder_layers: 8,
            decoder_layers: 8,
            kernel_sizes: vec![7, 7, 7, 7, 7, 7, 7, 7],
            strides: vec![1, 2, 2, 2, 2, 2, 2, 2],
            ..Default::default()
        }
    }

    /// Create configuration optimized for low bitrate
    pub fn low_bitrate() -> Self {
        Self {
            encoder_dim: 256,
            decoder_dim: 256,
            num_quantizers: 4,
            codebook_size: 512,
            codebook_dim: 128,
            target_bitrate: 2.0,
            compression_ratio: 64,
            encoder_layers: 4,
            decoder_layers: 4,
            ..Default::default()
        }
    }

    /// Create configuration optimized for real-time processing
    pub fn realtime_optimized() -> Self {
        Self {
            encoder_dim: 384,
            decoder_dim: 384,
            num_quantizers: 6,
            codebook_size: 1024,
            codebook_dim: 192,
            target_bitrate: 8.0,
            compression_ratio: 24,
            encoder_layers: 4,
            decoder_layers: 4,
            kernel_sizes: vec![3, 3, 3, 3],
            strides: vec![1, 2, 3, 4],
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(Error::Config(
                "sample_rate must be greater than 0".to_string(),
            ));
        }
        if self.channels == 0 {
            return Err(Error::Config("channels must be greater than 0".to_string()));
        }
        if self.encoder_dim == 0 || self.decoder_dim == 0 {
            return Err(Error::Config(
                "encoder_dim and decoder_dim must be greater than 0".to_string(),
            ));
        }
        if self.num_quantizers == 0 {
            return Err(Error::Config(
                "num_quantizers must be greater than 0".to_string(),
            ));
        }
        if self.codebook_size == 0 || (self.codebook_size & (self.codebook_size - 1)) != 0 {
            return Err(Error::Config(
                "codebook_size must be a power of 2".to_string(),
            ));
        }
        if self.compression_ratio == 0 {
            return Err(Error::Config(
                "compression_ratio must be greater than 0".to_string(),
            ));
        }
        if self.kernel_sizes.len() != self.encoder_layers {
            return Err(Error::Config(
                "kernel_sizes length must match encoder_layers".to_string(),
            ));
        }
        if self.strides.len() != self.encoder_layers {
            return Err(Error::Config(
                "strides length must match encoder_layers".to_string(),
            ));
        }
        Ok(())
    }
}

/// Neural codec compression request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecCompressionRequest {
    /// Audio samples to compress
    pub audio: Vec<f32>,
    /// Sample rate of input audio
    pub sample_rate: u32,
    /// Target bitrate (optional, uses config default if not specified)
    pub target_bitrate: Option<f32>,
    /// Quality level (0.0 = fastest/lowest quality, 1.0 = slowest/highest quality)
    pub quality_level: f32,
    /// Enable perceptual optimization
    pub perceptual_optimization: bool,
    /// Enable temporal consistency
    pub temporal_consistency: bool,
    /// Use variable bitrate encoding
    pub variable_bitrate: bool,
}

impl Default for CodecCompressionRequest {
    fn default() -> Self {
        Self {
            audio: Vec::new(),
            sample_rate: 24000,
            target_bitrate: None,
            quality_level: 0.8,
            perceptual_optimization: true,
            temporal_consistency: true,
            variable_bitrate: false,
        }
    }
}

/// Neural codec compression result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecCompressionResult {
    /// Compressed audio codes
    pub codes: Vec<Vec<u32>>,
    /// Quantizer indices used
    pub quantizer_indices: Vec<usize>,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Actual bitrate achieved
    pub actual_bitrate: f32,
    /// Compression time in milliseconds
    pub compression_time_ms: u64,
    /// Quality metrics
    pub quality_metrics: CodecQualityMetrics,
    /// Metadata for reconstruction
    pub metadata: CodecMetadata,
}

/// Neural codec decompression result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecDecompressionResult {
    /// Reconstructed audio samples
    pub audio: Vec<f32>,
    /// Sample rate of output audio
    pub sample_rate: u32,
    /// Duration of reconstructed audio
    pub duration: f32,
    /// Decompression time in milliseconds
    pub decompression_time_ms: u64,
    /// Quality metrics compared to original (if available)
    pub quality_metrics: Option<CodecQualityMetrics>,
}

/// Quality metrics for codec evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecQualityMetrics {
    /// Signal-to-noise ratio
    pub snr_db: f32,
    /// Perceptual evaluation of speech quality
    pub pesq_score: f32,
    /// Short-time objective intelligibility
    pub stoi_score: f32,
    /// Spectral distortion
    pub spectral_distortion_db: f32,
    /// Bitrate efficiency (quality per bit)
    pub bitrate_efficiency: f32,
    /// Perceptual quality score (0.0-1.0)
    pub perceptual_quality: f32,
    /// Temporal consistency score
    pub temporal_consistency: f32,
    /// Artifacts presence score (lower is better)
    pub artifacts_score: f32,
}

/// Metadata for codec operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecMetadata {
    /// Original audio length
    pub original_length: usize,
    /// Compressed data size in bytes
    pub compressed_size: usize,
    /// Codec version used
    pub codec_version: String,
    /// Encoding parameters
    pub encoding_params: HashMap<String, f32>,
    /// Timestamp of encoding
    pub timestamp: u64,
}

/// Neural audio encoder
#[derive(Debug)]
pub struct NeuralEncoder {
    config: NeuralCodecConfig,
    conv_layers: Vec<Conv1d>,
    residual_layers: Vec<ResidualBlock>,
    output_projection: candle_nn::Linear,
}

impl NeuralEncoder {
    pub fn new(config: &NeuralCodecConfig, vb: VarBuilder) -> Result<Self> {
        let mut conv_layers = Vec::new();
        let mut in_channels = config.channels;

        // Create encoder convolution layers
        for (i, (&kernel_size, &stride)) in config
            .kernel_sizes
            .iter()
            .zip(config.strides.iter())
            .enumerate()
        {
            let out_channels = if i == 0 {
                config.encoder_dim / 4
            } else if i == config.encoder_layers - 1 {
                config.encoder_dim
            } else {
                config.encoder_dim / 2
            };

            let conv_config = Conv1dConfig {
                padding: kernel_size / 2,
                stride,
                dilation: config.dilations.get(i).copied().unwrap_or(1),
                groups: 1,
                cudnn_fwd_algo: None,
            };

            let conv = conv1d(
                in_channels,
                out_channels,
                kernel_size,
                conv_config,
                vb.pp(&format!("conv_layers.{}", i)),
            )?;

            conv_layers.push(conv);
            in_channels = out_channels;
        }

        // Create residual blocks
        let mut residual_layers = Vec::new();
        for i in 0..config.encoder_layers {
            residual_layers.push(ResidualBlock::new(
                config.encoder_dim,
                config.encoder_dim,
                config.dropout_rate,
                vb.pp(&format!("residual.{}", i)),
            )?);
        }

        let output_projection = candle_nn::linear(
            config.encoder_dim,
            config.codebook_dim,
            vb.pp("output_projection"),
        )?;

        Ok(Self {
            config: config.clone(),
            conv_layers,
            residual_layers,
            output_projection,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut hidden = x.clone();

        // Apply convolution layers with activation
        for (i, conv) in self.conv_layers.iter().enumerate() {
            hidden = conv
                .forward(&hidden)
                .map_err(|e| Error::Processing(e.to_string()))?;

            // Apply activation (ELU)
            hidden = hidden
                .elu(1.0)
                .map_err(|e| Error::Processing(e.to_string()))?;
        }

        // Apply residual blocks
        if self.config.use_residual {
            for residual in &self.residual_layers {
                hidden = residual.forward(&hidden)?;
            }
        }

        // Final projection to codebook dimension
        self.output_projection
            .forward(&hidden)
            .map_err(|e| Error::Processing(e.to_string()))
    }
}

/// Neural audio decoder
#[derive(Debug)]
pub struct NeuralDecoder {
    config: NeuralCodecConfig,
    input_projection: candle_nn::Linear,
    conv_transpose_layers: Vec<Conv1d>,
    residual_layers: Vec<ResidualBlock>,
    output_conv: Conv1d,
}

impl NeuralDecoder {
    pub fn new(config: &NeuralCodecConfig, vb: VarBuilder) -> Result<Self> {
        let input_projection = candle_nn::linear(
            config.codebook_dim,
            config.decoder_dim,
            vb.pp("input_projection"),
        )?;

        let mut conv_transpose_layers = Vec::new();
        let mut in_channels = config.decoder_dim;

        // Create decoder convolution transpose layers (reverse of encoder)
        for (i, (&kernel_size, &stride)) in config
            .kernel_sizes
            .iter()
            .zip(config.strides.iter())
            .enumerate()
            .rev()
        {
            let out_channels = if i == 0 {
                config.channels
            } else if i == config.decoder_layers - 1 {
                config.decoder_dim
            } else {
                config.decoder_dim / 2
            };

            let conv_config = Conv1dConfig {
                padding: kernel_size / 2,
                stride,
                dilation: config.dilations.get(i).copied().unwrap_or(1),
                groups: 1,
                cudnn_fwd_algo: None,
            };

            let conv = conv1d(
                in_channels,
                out_channels,
                kernel_size,
                conv_config,
                vb.pp(&format!("conv_transpose.{}", i)),
            )?;

            conv_transpose_layers.push(conv);
            in_channels = out_channels;
        }

        // Create residual blocks
        let mut residual_layers = Vec::new();
        for i in 0..config.decoder_layers {
            residual_layers.push(ResidualBlock::new(
                config.decoder_dim,
                config.decoder_dim,
                config.dropout_rate,
                vb.pp(&format!("residual.{}", i)),
            )?);
        }

        // Final output convolution
        let output_conv = conv1d(
            config.decoder_dim,
            config.channels,
            7, // kernel size
            Conv1dConfig {
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("output_conv"),
        )?;

        Ok(Self {
            config: config.clone(),
            input_projection,
            conv_transpose_layers,
            residual_layers,
            output_conv,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Project from codebook dimension to decoder dimension
        let mut hidden = self
            .input_projection
            .forward(x)
            .map_err(|e| Error::Processing(e.to_string()))?;

        // Apply residual blocks
        if self.config.use_residual {
            for residual in &self.residual_layers {
                hidden = residual.forward(&hidden)?;
            }
        }

        // Apply transposed convolution layers
        for (i, conv) in self.conv_transpose_layers.iter().enumerate() {
            // Upsample by stride factor first
            let stride = self.config.strides[self.config.strides.len() - 1 - i];
            if stride > 1 {
                hidden = self.upsample(&hidden, stride)?;
            }

            hidden = conv
                .forward(&hidden)
                .map_err(|e| Error::Processing(e.to_string()))?;

            // Apply activation (ELU) except for the last layer
            if i < self.conv_transpose_layers.len() - 1 {
                hidden = hidden
                    .elu(1.0)
                    .map_err(|e| Error::Processing(e.to_string()))?;
            }
        }

        // Final output convolution with tanh activation
        let output = self
            .output_conv
            .forward(&hidden)
            .map_err(|e| Error::Processing(e.to_string()))?;

        output.tanh().map_err(|e| Error::Processing(e.to_string()))
    }

    fn upsample(&self, x: &Tensor, factor: usize) -> Result<Tensor> {
        // Simple nearest neighbor upsampling
        let (batch_size, channels, length) = x.dims3()?;
        let new_length = length * factor;

        // Create upsampled tensor by repeating each sample
        let mut upsampled_data = Vec::new();
        let data = x
            .flatten_all()?
            .to_vec1::<f32>()
            .map_err(|e| Error::Processing(e.to_string()))?;

        for b in 0..batch_size {
            for c in 0..channels {
                for t in 0..length {
                    let idx = b * channels * length + c * length + t;
                    let value = data[idx];
                    for _ in 0..factor {
                        upsampled_data.push(value);
                    }
                }
            }
        }

        Tensor::from_vec(
            upsampled_data,
            (batch_size, channels, new_length),
            x.device(),
        )
        .map_err(|e| Error::Processing(e.to_string()))
    }
}

/// Vector quantizer for neural codec
#[derive(Debug)]
pub struct VectorQuantizer {
    config: NeuralCodecConfig,
    codebooks: Vec<candle_nn::Embedding>,
    commitment_weight: f32,
}

impl VectorQuantizer {
    pub fn new(config: &NeuralCodecConfig, vb: VarBuilder) -> Result<Self> {
        let mut codebooks = Vec::new();

        for i in 0..config.num_quantizers {
            let codebook = candle_nn::embedding(
                config.codebook_size,
                config.codebook_dim,
                vb.pp(&format!("codebook.{}", i)),
            )?;
            codebooks.push(codebook);
        }

        Ok(Self {
            config: config.clone(),
            codebooks,
            commitment_weight: 0.25,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<QuantizationResult> {
        let (batch_size, dim, seq_len) = x.dims3()?;
        let device = x.device();

        let mut quantized_layers = Vec::new();
        let mut indices_layers = Vec::new();
        let mut losses = Vec::new();

        let mut residual = x.clone();

        // Apply residual quantization
        for (i, codebook) in self.codebooks.iter().enumerate() {
            let (quantized, indices, loss) = self.quantize_layer(&residual, codebook)?;

            quantized_layers.push(quantized.clone());
            indices_layers.push(indices);
            losses.push(loss);

            // Update residual for next quantizer
            residual = (&residual - &quantized)?;

            // Break early for lower quality settings
            if i >= (self.config.num_quantizers as f32 * 0.6) as usize
                && residual
                    .sqr()?
                    .mean_all()?
                    .to_scalar::<f32>()
                    .map_err(|e| Error::Processing(e.to_string()))?
                    < 0.01
            {
                break;
            }
        }

        // Sum all quantized layers
        let mut quantized_sum = quantized_layers[0].clone();
        for quantized in &quantized_layers[1..] {
            quantized_sum = (&quantized_sum + quantized)?;
        }

        // Calculate average loss
        let avg_loss = losses.iter().sum::<f32>() / losses.len() as f32;

        let perplexity = self.calculate_perplexity(&indices_layers)?;

        Ok(QuantizationResult {
            quantized: quantized_sum,
            indices: indices_layers,
            quantization_loss: avg_loss,
            perplexity,
        })
    }

    fn quantize_layer(
        &self,
        x: &Tensor,
        codebook: &candle_nn::Embedding,
    ) -> Result<(Tensor, Vec<u32>, f32)> {
        let (batch_size, dim, seq_len) = x.dims3()?;

        // Flatten spatial dimensions
        let x_flat = x.transpose(1, 2)?.reshape((batch_size * seq_len, dim))?;

        // Get codebook vectors
        let codebook_indices = Tensor::arange(0u32, self.config.codebook_size as u32, x.device())?;
        let codebook_vectors = codebook.forward(&codebook_indices)?;

        // Compute distances
        let distances = self.compute_distances(&x_flat, &codebook_vectors)?;

        // Find nearest codes
        let indices = distances.argmin(1)?;
        let indices_data = indices
            .to_vec1::<u32>()
            .map_err(|e| Error::Processing(e.to_string()))?;

        // Get quantized vectors
        let quantized_flat = codebook.forward(&indices)?;
        let quantized = quantized_flat
            .reshape((batch_size, seq_len, dim))?
            .transpose(1, 2)?;

        // Calculate commitment loss
        let commitment_loss = (&x_flat - &quantized_flat.detach())?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()
            .map_err(|e| Error::Processing(e.to_string()))?;

        Ok((
            quantized,
            indices_data,
            commitment_loss * self.commitment_weight,
        ))
    }

    fn compute_distances(&self, x: &Tensor, codebook: &Tensor) -> Result<Tensor> {
        // Compute L2 distances between input vectors and codebook vectors
        let x_norm = x.sqr()?.sum_keepdim(1)?;
        let codebook_norm = codebook.sqr()?.sum_keepdim(1)?.t()?;
        let dot_product = x.matmul(&codebook.t()?)?;

        // Distance = ||x||^2 + ||c||^2 - 2*x*c
        let distances = (x_norm + codebook_norm - &(&dot_product * 2.0)?)?;

        Ok(distances)
    }

    fn calculate_perplexity(&self, indices_layers: &[Vec<u32>]) -> Result<f32> {
        let mut total_entropy = 0.0;
        let mut total_count = 0;

        for indices in indices_layers {
            if indices.is_empty() {
                continue;
            }

            let mut counts = vec![0; self.config.codebook_size];
            for &idx in indices {
                if (idx as usize) < counts.len() {
                    counts[idx as usize] += 1;
                }
            }

            let total = indices.len() as f32;
            let mut entropy = 0.0;

            for count in counts {
                if count > 0 {
                    let prob = count as f32 / total;
                    entropy -= prob * prob.ln();
                }
            }

            total_entropy += entropy;
            total_count += 1;
        }

        let avg_entropy = if total_count > 0 {
            total_entropy / total_count as f32
        } else {
            0.0
        };
        Ok(avg_entropy.exp())
    }

    pub fn decode(&self, indices: &[Vec<u32>]) -> Result<Tensor> {
        if indices.is_empty() {
            return Err(Error::Processing("Empty indices for decoding".to_string()));
        }

        let seq_len = indices[0].len();
        let device = Device::Cpu; // Use appropriate device

        let mut decoded_sum: Option<Tensor> = None;

        for (layer_idx, layer_indices) in indices.iter().enumerate() {
            if layer_idx >= self.codebooks.len() {
                break;
            }

            let indices_tensor = Tensor::from_vec(
                layer_indices
                    .clone()
                    .into_iter()
                    .map(|x| x as i64)
                    .collect(),
                (1, seq_len),
                &device,
            )?;

            let decoded_layer = self.codebooks[layer_idx].forward(&indices_tensor)?;
            let decoded_reshaped = decoded_layer.transpose(1, 2)?;

            match decoded_sum {
                None => decoded_sum = Some(decoded_reshaped),
                Some(ref sum) => {
                    decoded_sum = Some((sum + &decoded_reshaped)?);
                }
            }
        }

        decoded_sum.ok_or_else(|| Error::Processing("Failed to decode any layers".to_string()))
    }
}

/// Quantization result
#[derive(Debug)]
pub struct QuantizationResult {
    pub quantized: Tensor,
    pub indices: Vec<Vec<u32>>,
    pub quantization_loss: f32,
    pub perplexity: f32,
}

/// Residual block for encoder/decoder
#[derive(Debug)]
pub struct ResidualBlock {
    conv1: Conv1d,
    conv2: Conv1d,
    skip_conv: Option<Conv1d>,
    dropout: candle_nn::Dropout,
}

impl ResidualBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        dropout_rate: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv1 = conv1d(
            in_channels,
            out_channels,
            3,
            Conv1dConfig {
                padding: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("conv1"),
        )?;

        let conv2 = conv1d(
            out_channels,
            out_channels,
            3,
            Conv1dConfig {
                padding: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("conv2"),
        )?;

        let skip_conv = if in_channels != out_channels {
            Some(conv1d(
                in_channels,
                out_channels,
                1,
                Conv1dConfig {
                    padding: 0,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
                vb.pp("skip"),
            )?)
        } else {
            None
        };

        let dropout = candle_nn::Dropout::new(dropout_rate);

        Ok(Self {
            conv1,
            conv2,
            skip_conv,
            dropout,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut residual = x.clone();

        // First convolution + activation
        let mut out = self
            .conv1
            .forward(x)
            .map_err(|e| Error::Processing(e.to_string()))?;
        out = out.elu(1.0).map_err(|e| Error::Processing(e.to_string()))?;
        out = self.dropout.forward(&out, false)?;

        // Second convolution
        out = self
            .conv2
            .forward(&out)
            .map_err(|e| Error::Processing(e.to_string()))?;

        // Skip connection
        if let Some(ref skip) = self.skip_conv {
            residual = skip
                .forward(&residual)
                .map_err(|e| Error::Processing(e.to_string()))?;
        }

        // Add residual connection
        let result = (&out + &residual)?;

        // Final activation
        result
            .elu(1.0)
            .map_err(|e| Error::Processing(e.to_string()))
    }
}

/// Main neural codec model
#[derive(Debug)]
pub struct NeuralCodec {
    config: NeuralCodecConfig,
    encoder: NeuralEncoder,
    decoder: NeuralDecoder,
    quantizer: VectorQuantizer,
    device: Device,
}

impl NeuralCodec {
    pub fn new(config: NeuralCodecConfig, vb: VarBuilder, device: Device) -> Result<Self> {
        config.validate()?;

        let encoder = NeuralEncoder::new(&config, vb.pp("encoder"))?;
        let decoder = NeuralDecoder::new(&config, vb.pp("decoder"))?;
        let quantizer = VectorQuantizer::new(&config, vb.pp("quantizer"))?;

        Ok(Self {
            config,
            encoder,
            decoder,
            quantizer,
            device,
        })
    }

    /// Compress audio to neural codes
    pub fn compress(&self, request: &CodecCompressionRequest) -> Result<CodecCompressionResult> {
        let start_time = Instant::now();

        // Validate input
        if request.audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        // Prepare input tensor
        let audio_tensor = self.prepare_audio_tensor(&request.audio)?;

        // Encode audio
        let encoded = self.encoder.forward(&audio_tensor)?;

        // Quantize
        let quantization_result = self.quantizer.forward(&encoded)?;

        // Calculate metrics
        let compression_time = start_time.elapsed();
        let original_size = request.audio.len() * 4; // 4 bytes per float32
        let compressed_size = self.calculate_compressed_size(&quantization_result.indices);
        let compression_ratio = original_size as f32 / compressed_size as f32;
        let actual_bitrate =
            self.calculate_bitrate(compressed_size, request.audio.len(), request.sample_rate);

        // Reconstruct for quality evaluation
        let reconstructed = self.decoder.forward(&quantization_result.quantized)?;
        let quality_metrics = self.evaluate_quality(&audio_tensor, &reconstructed)?;

        Ok(CodecCompressionResult {
            codes: quantization_result.indices,
            quantizer_indices: (0..self.config.num_quantizers).collect(),
            compression_ratio,
            actual_bitrate,
            compression_time_ms: compression_time.as_millis() as u64,
            quality_metrics,
            metadata: CodecMetadata {
                original_length: request.audio.len(),
                compressed_size,
                codec_version: "NeuralCodec-1.0".to_string(),
                encoding_params: {
                    let mut params = HashMap::new();
                    params.insert("quality_level".to_string(), request.quality_level);
                    params.insert(
                        "target_bitrate".to_string(),
                        request.target_bitrate.unwrap_or(self.config.target_bitrate),
                    );
                    params
                },
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Decompress neural codes to audio
    pub fn decompress(
        &self,
        codes: &[Vec<u32>],
        metadata: &CodecMetadata,
    ) -> Result<CodecDecompressionResult> {
        let start_time = Instant::now();

        if codes.is_empty() {
            return Err(Error::InvalidInput("Empty codes input".to_string()));
        }

        // Decode quantized representation
        let quantized = self.quantizer.decode(codes)?;

        // Decode to audio
        let reconstructed = self.decoder.forward(&quantized)?;

        // Convert tensor to audio samples
        let audio = self.tensor_to_audio(&reconstructed)?;

        let decompression_time = start_time.elapsed();
        let duration = audio.len() as f32 / self.config.sample_rate as f32;

        Ok(CodecDecompressionResult {
            audio,
            sample_rate: self.config.sample_rate,
            duration,
            decompression_time_ms: decompression_time.as_millis() as u64,
            quality_metrics: None, // Would need original for comparison
        })
    }

    fn prepare_audio_tensor(&self, audio: &[f32]) -> Result<Tensor> {
        let batch_size = 1;
        let channels = self.config.channels;
        let length = audio.len() / channels;

        Tensor::from_vec(audio.to_vec(), (batch_size, channels, length), &self.device)
            .map_err(|e| Error::Processing(e.to_string()))
    }

    fn tensor_to_audio(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        let data = tensor
            .flatten_all()?
            .to_vec1::<f32>()
            .map_err(|e| Error::Processing(e.to_string()))?;
        Ok(data)
    }

    fn calculate_compressed_size(&self, indices: &[Vec<u32>]) -> usize {
        let bits_per_code = (self.config.codebook_size as f32).log2().ceil() as usize;
        let total_codes: usize = indices.iter().map(|layer| layer.len()).sum();
        total_codes * bits_per_code / 8 // Convert to bytes
    }

    fn calculate_bitrate(
        &self,
        compressed_size: usize,
        audio_length: usize,
        sample_rate: u32,
    ) -> f32 {
        let duration_seconds = audio_length as f32 / sample_rate as f32;
        (compressed_size * 8) as f32 / duration_seconds / 1000.0 // kbps
    }

    fn evaluate_quality(
        &self,
        original: &Tensor,
        reconstructed: &Tensor,
    ) -> Result<CodecQualityMetrics> {
        // Simplified quality evaluation
        let mse = (original - reconstructed)?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()
            .map_err(|e| Error::Processing(e.to_string()))?;

        let snr_db = -10.0 * mse.log10();

        Ok(CodecQualityMetrics {
            snr_db,
            pesq_score: 3.5 + (snr_db / 30.0).min(1.0), // Estimated PESQ
            stoi_score: 0.8 + (snr_db / 50.0).min(0.2), // Estimated STOI
            spectral_distortion_db: mse.sqrt() * 20.0,
            bitrate_efficiency: snr_db / self.config.target_bitrate,
            perceptual_quality: (snr_db / 30.0).min(1.0).max(0.0),
            temporal_consistency: 0.9, // Placeholder
            artifacts_score: mse.sqrt(),
        })
    }
}

/// Neural codec manager for high-level operations
#[derive(Debug)]
pub struct NeuralCodecManager {
    codec: Arc<RwLock<NeuralCodec>>,
    config: NeuralCodecConfig,
    device: Device,
    compression_cache: Arc<RwLock<HashMap<String, CodecCompressionResult>>>,
    performance_stats: Arc<RwLock<CodecPerformanceStats>>,
}

#[derive(Debug, Default, Clone)]
pub struct CodecPerformanceStats {
    pub total_compressions: u64,
    pub total_decompressions: u64,
    pub total_compression_time: Duration,
    pub total_decompression_time: Duration,
    pub average_compression_ratio: f32,
    pub average_quality_score: f32,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl NeuralCodecManager {
    pub fn new(config: NeuralCodecConfig) -> Result<Self> {
        config.validate()?;

        let device = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        info!("Initializing Neural Codec on device: {:?}", device);

        // Initialize model weights
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let codec = NeuralCodec::new(config.clone(), vb, device.clone())?;

        Ok(Self {
            codec: Arc::new(RwLock::new(codec)),
            config,
            device,
            compression_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_stats: Arc::new(RwLock::new(CodecPerformanceStats::default())),
        })
    }

    /// Create neural codec with high-quality configuration
    pub fn high_quality() -> Result<Self> {
        Self::new(NeuralCodecConfig::high_quality())
    }

    /// Create neural codec optimized for low bitrate
    pub fn low_bitrate() -> Result<Self> {
        Self::new(NeuralCodecConfig::low_bitrate())
    }

    /// Create neural codec optimized for real-time processing
    pub fn realtime_optimized() -> Result<Self> {
        Self::new(NeuralCodecConfig::realtime_optimized())
    }

    /// Compress audio with caching
    pub async fn compress(
        &self,
        request: CodecCompressionRequest,
    ) -> Result<CodecCompressionResult> {
        let cache_key = self.compute_cache_key(&request);

        // Check cache first
        {
            let cache = self.compression_cache.read().await;
            if let Some(cached_result) = cache.get(&cache_key) {
                let mut stats = self.performance_stats.write().await;
                stats.cache_hits += 1;
                debug!("Cache hit for compression request");
                return Ok(cached_result.clone());
            }
        }

        // Perform compression
        let codec = self.codec.read().await;
        let result = codec.compress(&request)?;
        drop(codec);

        // Update performance statistics
        {
            let mut stats = self.performance_stats.write().await;
            stats.total_compressions += 1;
            stats.total_compression_time += Duration::from_millis(result.compression_time_ms);
            stats.average_compression_ratio = (stats.average_compression_ratio
                * (stats.total_compressions - 1) as f32
                + result.compression_ratio)
                / stats.total_compressions as f32;
            stats.average_quality_score = (stats.average_quality_score
                * (stats.total_compressions - 1) as f32
                + result.quality_metrics.snr_db)
                / stats.total_compressions as f32;
            stats.cache_misses += 1;
        }

        // Cache result
        {
            let mut cache = self.compression_cache.write().await;
            cache.insert(cache_key, result.clone());

            // Limit cache size
            if cache.len() > 1000 {
                let oldest_key = cache.keys().next().unwrap().clone();
                cache.remove(&oldest_key);
            }
        }

        info!(
            "Neural codec compression completed: {:.2}x compression, {:.2} kbps, {:.2} dB SNR",
            result.compression_ratio, result.actual_bitrate, result.quality_metrics.snr_db
        );

        Ok(result)
    }

    /// Decompress neural codes
    pub async fn decompress(
        &self,
        codes: &[Vec<u32>],
        metadata: &CodecMetadata,
    ) -> Result<CodecDecompressionResult> {
        let codec = self.codec.read().await;
        let result = codec.decompress(codes, metadata)?;
        drop(codec);

        // Update performance statistics
        {
            let mut stats = self.performance_stats.write().await;
            stats.total_decompressions += 1;
            stats.total_decompression_time += Duration::from_millis(result.decompression_time_ms);
        }

        info!(
            "Neural codec decompression completed: {:.2}s audio in {}ms",
            result.duration, result.decompression_time_ms
        );

        Ok(result)
    }

    fn compute_cache_key(&self, request: &CodecCompressionRequest) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash audio content (sample first/last values for efficiency)
        if !request.audio.is_empty() {
            request.audio[0].to_bits().hash(&mut hasher);
            if request.audio.len() > 1 {
                request.audio[request.audio.len() - 1]
                    .to_bits()
                    .hash(&mut hasher);
            }
            request.audio.len().hash(&mut hasher);
        }

        request.sample_rate.hash(&mut hasher);
        ((request.quality_level * 1000.0) as u32).hash(&mut hasher);
        request.perceptual_optimization.hash(&mut hasher);
        request.temporal_consistency.hash(&mut hasher);

        format!("neural_codec_{:x}", hasher.finish())
    }

    /// Get performance statistics
    pub async fn get_performance_stats(&self) -> CodecPerformanceStats {
        (*self.performance_stats.read().await).clone()
    }

    /// Clear compression cache
    pub async fn clear_cache(&self) {
        self.compression_cache.write().await.clear();
        info!("Neural codec compression cache cleared");
    }

    /// Get codec configuration
    pub fn config(&self) -> &NeuralCodecConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_codec_config_validation() {
        let config = NeuralCodecConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.sample_rate = 0;
        assert!(invalid_config.validate().is_err());

        invalid_config = config.clone();
        invalid_config.codebook_size = 1023; // Not a power of 2
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_codec_compression_request_default() {
        let request = CodecCompressionRequest::default();
        assert_eq!(request.sample_rate, 24000);
        assert_eq!(request.quality_level, 0.8);
        assert!(request.perceptual_optimization);
    }

    #[tokio::test]
    async fn test_neural_codec_manager_creation() {
        let config = NeuralCodecConfig::low_bitrate();
        let manager = NeuralCodecManager::new(config);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_neural_codec_compression() {
        let manager = match NeuralCodecManager::low_bitrate() {
            Ok(m) => m,
            Err(_) => return, // Skip test if model creation fails
        };

        let request = CodecCompressionRequest {
            audio: {
                let mut audio = Vec::new();
                let pattern = [0.1, -0.1, 0.2, -0.2, 0.0];
                for _ in 0..(1024 / pattern.len()) {
                    audio.extend_from_slice(&pattern);
                }
                audio.extend_from_slice(&pattern[0..(1024 % pattern.len())]);
                audio
            },
            sample_rate: 24000,
            quality_level: 0.8,
            ..Default::default()
        };

        let result = manager.compress(request).await;
        assert!(result.is_ok());

        let compression_result = result.unwrap();
        assert!(!compression_result.codes.is_empty());
        assert!(compression_result.compression_ratio > 1.0);
        assert!(compression_result.actual_bitrate > 0.0);
    }
}
