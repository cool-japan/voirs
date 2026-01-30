//! VITS model loader from SafeTensors format
//!
//! Loads pre-trained VITS models from SafeTensors files.

use candle_core::{Device, Module};
use candle_nn::VarBuilder;
use std::path::Path;

use crate::{AcousticError, Result};

use super::{TextEncoderConfig, VitsConfig};

/// Load VITS model from SafeTensors file
///
/// This uses candle's VarBuilder to directly load weights from SafeTensors
pub fn load_vits_from_safetensors<P: AsRef<Path>>(
    model_path: P,
    device: Device,
) -> Result<VitsInference> {
    let path = model_path.as_ref();

    // Create VarBuilder from SafeTensors file
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[path], candle_core::DType::F32, &device).map_err(
            |e| AcousticError::ModelError {
                message: format!("Failed to load SafeTensors: {}", e),
            },
        )?
    };

    // Infer configuration from model structure
    let config = infer_config_from_varbuilder(&vb)?;

    // Create inference model
    VitsInference::new(config, vb, device)
}

/// Infer model configuration from VarBuilder
fn infer_config_from_varbuilder(vb: &VarBuilder) -> Result<VitsConfig> {
    // Try to get embedding weight to infer vocab size and hidden dim
    let emb_vb = vb.pp("model.generator.text_encoder.emb");

    // Get weight tensor to infer dimensions
    let weight = emb_vb
        .get((78, 192), "weight")
        .map_err(|e| AcousticError::ModelError {
            message: format!("Failed to get embedding weight: {}", e),
        })?;

    let (vocab_size, hidden_dim) = weight.dims2().map_err(|e| AcousticError::ModelError {
        message: format!("Invalid embedding shape: {}", e),
    })?;

    tracing::info!(
        "Inferred config: vocab_size={}, hidden_dim={}",
        vocab_size,
        hidden_dim
    );

    // Create configuration
    let mut config = VitsConfig::default();

    // Text encoder config
    config.text_encoder = TextEncoderConfig {
        n_layers: 6,
        d_model: hidden_dim,
        n_heads: 2,
        d_ff: hidden_dim * 4,
        dropout: 0.0, // No dropout for inference
        max_seq_len: 1000,
        vocab_size,
        kernel_size: 3,
        n_conv_layers: 3,
        use_relative_pos: true,
    };

    Ok(config)
}

/// Simplified VITS inference model (no training components)
pub struct VitsInference {
    config: VitsConfig,
    text_encoder: TextEncoderInference,
    decoder: DecoderInference,
    device: Device,
}

impl VitsInference {
    pub fn new(config: VitsConfig, vb: VarBuilder, device: Device) -> Result<Self> {
        // Create text encoder with proper path
        let te_vb = vb.pp("model.generator.text_encoder");
        let text_encoder = TextEncoderInference::new(&config.text_encoder, te_vb).map_err(|e| {
            AcousticError::ModelError {
                message: format!("Text encoder init failed: {}", e),
            }
        })?;

        // Create decoder with proper path
        let dec_vb = vb.pp("model.generator.decoder");
        let decoder = DecoderInference::new(dec_vb).map_err(|e| AcousticError::ModelError {
            message: format!("Decoder init failed: {}", e),
        })?;

        Ok(Self {
            config,
            text_encoder,
            decoder,
            device,
        })
    }

    /// Synthesize audio from text tokens
    pub fn synthesize(&self, token_ids: &[i64]) -> Result<Vec<f32>> {
        use candle_core::Tensor;

        // Convert token IDs to tensor
        let tokens =
            Tensor::from_slice(token_ids, (1, token_ids.len()), &self.device).map_err(|e| {
                AcousticError::ModelError {
                    message: format!("Failed to create token tensor: {}", e),
                }
            })?;

        // Text encoding
        let hidden = self
            .text_encoder
            .forward(&tokens)
            .map_err(|e| AcousticError::ModelError {
                message: format!("Text encoding failed: {}", e),
            })?;

        // Decode to audio
        let audio = self
            .decoder
            .forward(&hidden)
            .map_err(|e| AcousticError::ModelError {
                message: format!("Decoding failed: {}", e),
            })?;

        // Convert to Vec<f32> (remove batch dimension)
        let audio_1d = audio.get(0).map_err(|e| AcousticError::ModelError {
            message: format!("Failed to get first batch: {}", e),
        })?;

        audio_1d.to_vec1().map_err(|e| AcousticError::ModelError {
            message: format!("Failed to convert audio tensor: {}", e),
        })
    }
}

/// Simplified text encoder for inference
struct TextEncoderInference {
    embedding: candle_nn::Embedding,
}

impl TextEncoderInference {
    fn new(config: &TextEncoderConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let embedding = candle_nn::embedding(config.vocab_size, config.d_model, vb.pp("emb"))?;

        Ok(Self { embedding })
    }

    fn forward(&self, tokens: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        self.embedding.forward(tokens)
    }
}

/// Enhanced decoder for inference with HiFi-GAN architecture
struct DecoderInference {
    input_conv: candle_nn::Conv1d,
    upsample_layers: Vec<UpsampleBlock>,
    mrf_blocks: Vec<MultiReceptiveFieldBlock>,
    output_conv: candle_nn::Conv1d,
}

/// Upsampling block with transposed convolution
struct UpsampleBlock {
    conv_transpose: candle_nn::ConvTranspose1d,
    upsample_rate: usize,
}

impl UpsampleBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        use candle_nn::ConvTranspose1dConfig;

        let conv_transpose = candle_nn::conv_transpose1d(
            in_channels,
            out_channels,
            kernel_size,
            ConvTranspose1dConfig {
                stride,
                padding: (kernel_size - stride) / 2,
                ..Default::default()
            },
            vb,
        )?;

        Ok(Self {
            conv_transpose,
            upsample_rate: stride,
        })
    }

    fn forward(&self, x: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        let h = self.conv_transpose.forward(x)?;
        h.relu()
    }
}

/// Multi-Receptive Field block for diverse temporal patterns
struct MultiReceptiveFieldBlock {
    residual_convs: Vec<candle_nn::Conv1d>,
}

impl MultiReceptiveFieldBlock {
    fn new(channels: usize, kernel_sizes: &[usize], vb: VarBuilder) -> candle_core::Result<Self> {
        use candle_nn::Conv1dConfig;

        let mut residual_convs = Vec::new();

        for (i, &kernel_size) in kernel_sizes.iter().enumerate() {
            let conv = candle_nn::conv1d(
                channels,
                channels,
                kernel_size,
                Conv1dConfig {
                    padding: kernel_size / 2,
                    ..Default::default()
                },
                vb.pp(format!("resblock_{}", i)),
            )?;
            residual_convs.push(conv);
        }

        Ok(Self { residual_convs })
    }

    fn forward(&self, x: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        let mut outputs = Vec::new();

        // Process input through each receptive field path
        for conv in &self.residual_convs {
            let h = conv.forward(x)?.relu()?;
            outputs.push(h);
        }

        // Sum all paths (multi-receptive field fusion)
        let mut result = outputs[0].clone();
        for output in outputs.iter().skip(1) {
            result = (result + output)?;
        }

        // Add residual connection
        (result + x)?.relu()
    }
}

impl DecoderInference {
    fn new(vb: VarBuilder) -> candle_core::Result<Self> {
        use candle_nn::Conv1dConfig;

        // Load input convolution (80 -> 512 channels)
        let input_conv = candle_nn::conv1d(
            80,  // From mel/flow output
            512, // HiFi-GAN hidden dim
            7,   // Kernel size
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("input_conv"),
        )?;

        // HiFi-GAN upsampling configuration: 8x, 8x, 2x, 2x = 256x total
        let upsample_configs = vec![
            (512, 256, 16, 8), // 8x upsampling
            (256, 128, 16, 8), // 8x upsampling
            (128, 64, 4, 2),   // 2x upsampling
            (64, 32, 4, 2),    // 2x upsampling
        ];

        let mut upsample_layers = Vec::new();
        for (i, (in_ch, out_ch, kernel, stride)) in upsample_configs.iter().enumerate() {
            let layer = UpsampleBlock::new(
                *in_ch,
                *out_ch,
                *kernel,
                *stride,
                vb.pp(format!("upsample_{}", i)),
            )?;
            upsample_layers.push(layer);
        }

        // Multi-receptive field blocks after each upsampling
        let mrf_configs = vec![
            (256, vec![3, 7, 11]), // After 1st upsample
            (128, vec![3, 7, 11]), // After 2nd upsample
            (64, vec![3, 7, 11]),  // After 3rd upsample
            (32, vec![3, 7, 11]),  // After 4th upsample
        ];

        let mut mrf_blocks = Vec::new();
        for (i, (channels, kernel_sizes)) in mrf_configs.iter().enumerate() {
            let block = MultiReceptiveFieldBlock::new(
                *channels,
                kernel_sizes,
                vb.pp(format!("mrf_{}", i)),
            )?;
            mrf_blocks.push(block);
        }

        // Output projection to waveform (32 -> 1 channel)
        let output_conv = candle_nn::conv1d(
            32,
            1,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("output_conv"),
        )?;

        Ok(Self {
            input_conv,
            upsample_layers,
            mrf_blocks,
            output_conv,
        })
    }

    fn forward(&self, hidden: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        // Input: [batch, hidden_dim, seq_len]
        // Output: [batch, audio_len]

        // Apply input convolution (80 -> 512 channels)
        let mut h = self.input_conv.forward(hidden)?;
        h = h.relu()?;

        // Progressive upsampling with MRF blocks (HiFi-GAN architecture)
        // Total upsampling: 8x * 8x * 2x * 2x = 256x (for hop_length=256)
        for (upsample_layer, mrf_block) in self.upsample_layers.iter().zip(self.mrf_blocks.iter()) {
            // Upsample with transposed convolution
            h = upsample_layer.forward(&h)?;

            // Apply multi-receptive field block for rich temporal patterns
            h = mrf_block.forward(&h)?;
        }

        // Final projection to waveform (32 -> 1 channel)
        let audio = self.output_conv.forward(&h)?;

        // Apply tanh activation for [-1, 1] range
        let audio = audio.tanh()?;

        // Remove channel dimension: [batch, 1, audio_len] -> [batch, audio_len]
        audio.squeeze(1)
    }
}
