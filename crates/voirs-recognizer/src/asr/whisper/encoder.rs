//! Whisper encoder implementation with transformer blocks and MLP layers
//!
//! This module provides the audio encoder for Whisper models with optimized
//! transformer blocks and multi-layer perceptron implementations.

use super::attention::MultiHeadAttention;
use crate::RecognitionError;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};

/// GELU activation function implementation
fn gelu_activation(x: &Tensor) -> Result<Tensor, candle_core::Error> {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    // Approximation: GELU(x) ≈ x * sigmoid(1.702 * x)
    let sigmoid_input = (x * 1.702)?;
    let sigmoid = candle_nn::ops::sigmoid(&sigmoid_input)?;
    x * sigmoid
}

/// Whisper encoder configuration
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// Number of mel filter banks in input spectrogram
    pub n_mels: usize,
    /// Context length for audio processing
    pub n_audio_ctx: usize,
    /// Hidden state dimension for audio encoder
    pub n_audio_state: usize,
    /// Number of attention heads in audio encoder
    pub n_audio_head: usize,
    /// Number of transformer layers in audio encoder
    pub n_audio_layer: usize,
    /// Size of the vocabulary
    pub n_vocab: usize,
    /// Context length for text processing
    pub n_text_ctx: usize,
    /// Hidden state dimension for text decoder
    pub n_text_state: usize,
    /// Number of attention heads in text decoder
    pub n_text_head: usize,
    /// Number of transformer layers in text decoder
    pub n_text_layer: usize,
    /// Model size identifier (e.g., "tiny", "base", "small", "medium", "large")
    pub model_size: String,
    /// Whether the model supports multiple languages
    pub multilingual: bool,
    /// Length of audio chunks for processing (in frames)
    pub chunk_length: usize,
    /// Hop length for STFT window (in samples)
    pub hop_length: usize,
    /// Audio sample rate in Hz
    pub sample_rate: u32,
    /// Quantization mode for the model
    pub quantization: QuantizationMode,
}

/// Quantization mode for model optimization
#[derive(Debug, Clone, Default)]
pub enum QuantizationMode {
    /// No quantization (full precision)
    #[default]
    None,
    /// 16-bit floating point quantization
    F16,
    /// 8-bit integer quantization with scaling parameters
    INT8 {
        /// Scaling factor for quantization
        scale: f32,
        /// Zero point offset for quantization
        zero_point: i8,
    },
    /// Dynamic quantization (determined at runtime)
    Dynamic,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 512,
            n_audio_head: 8,
            n_audio_layer: 6,
            n_vocab: 51865,
            n_text_ctx: 448,
            n_text_state: 512,
            n_text_head: 8,
            n_text_layer: 6,
            model_size: "base".to_string(),
            multilingual: true,
            chunk_length: 30,
            hop_length: 160,
            sample_rate: 16000,
            quantization: QuantizationMode::default(),
        }
    }
}

impl WhisperConfig {
    /// Create configuration for Whisper Tiny model (~39 MB)
    pub fn tiny() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 384,
            n_audio_head: 6,
            n_audio_layer: 4,
            n_vocab: 51865,
            n_text_ctx: 448,
            n_text_state: 384,
            n_text_head: 6,
            n_text_layer: 4,
            model_size: "tiny".to_string(),
            multilingual: true,
            chunk_length: 30,
            hop_length: 160,
            sample_rate: 16000,
            quantization: QuantizationMode::default(),
        }
    }

    /// Create configuration for Whisper Base model (~74 MB)
    pub fn base() -> Self {
        Self::default()
    }

    /// Create configuration for Whisper Small model (~244 MB)
    pub fn small() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 768,
            n_audio_head: 12,
            n_audio_layer: 12,
            n_vocab: 51865,
            n_text_ctx: 448,
            n_text_state: 768,
            n_text_head: 12,
            n_text_layer: 12,
            model_size: "small".to_string(),
            multilingual: true,
            chunk_length: 30,
            hop_length: 160,
            sample_rate: 16000,
            quantization: QuantizationMode::default(),
        }
    }

    /// Create configuration for Whisper Medium model (~769 MB)
    pub fn medium() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 1024,
            n_audio_head: 16,
            n_audio_layer: 24,
            n_vocab: 51865,
            n_text_ctx: 448,
            n_text_state: 1024,
            n_text_head: 16,
            n_text_layer: 24,
            model_size: "medium".to_string(),
            multilingual: true,
            chunk_length: 30,
            hop_length: 160,
            sample_rate: 16000,
            quantization: QuantizationMode::default(),
        }
    }

    /// Create configuration for Whisper Large model (~1550 MB)
    pub fn large() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 1280,
            n_audio_head: 20,
            n_audio_layer: 32,
            n_vocab: 51865,
            n_text_ctx: 448,
            n_text_state: 1280,
            n_text_head: 20,
            n_text_layer: 32,
            model_size: "large".to_string(),
            multilingual: true,
            chunk_length: 30,
            hop_length: 160,
            sample_rate: 16000,
            quantization: QuantizationMode::default(),
        }
    }

    /// Create configuration for Whisper Large-v2 model (~1550 MB)
    pub fn large_v2() -> Self {
        let mut config = Self::large();
        config.model_size = "large-v2".to_string();
        config
    }

    /// Create configuration for Whisper Large-v3 model (~1550 MB)
    pub fn large_v3() -> Self {
        let mut config = Self::large();
        config.model_size = "large-v3".to_string();
        config
    }

    /// Create configuration from WhisperModelSize enum
    pub fn from_model_size(model_size: crate::asr::WhisperModelSize) -> Self {
        match model_size {
            crate::asr::WhisperModelSize::Tiny => Self::tiny(),
            crate::asr::WhisperModelSize::Base => Self::base(),
            crate::asr::WhisperModelSize::Small => Self::small(),
            crate::asr::WhisperModelSize::Medium => Self::medium(),
            crate::asr::WhisperModelSize::Large => Self::large(),
            crate::asr::WhisperModelSize::LargeV2 => Self::large_v2(),
            crate::asr::WhisperModelSize::LargeV3 => Self::large_v3(),
        }
    }

    /// Get model size as enum
    pub fn get_model_size(&self) -> Option<crate::asr::WhisperModelSize> {
        match self.model_size.as_str() {
            "tiny" => Some(crate::asr::WhisperModelSize::Tiny),
            "base" => Some(crate::asr::WhisperModelSize::Base),
            "small" => Some(crate::asr::WhisperModelSize::Small),
            "medium" => Some(crate::asr::WhisperModelSize::Medium),
            "large" => Some(crate::asr::WhisperModelSize::Large),
            "large-v2" => Some(crate::asr::WhisperModelSize::LargeV2),
            "large-v3" => Some(crate::asr::WhisperModelSize::LargeV3),
            _ => None,
        }
    }

    /// Get estimated model size in MB
    pub fn estimated_size_mb(&self) -> f32 {
        match self.model_size.as_str() {
            "tiny" => 39.0,
            "base" => 74.0,
            "small" => 244.0,
            "medium" => 769.0,
            "large" | "large-v2" | "large-v3" => 1550.0,
            _ => 100.0, // Default estimate
        }
    }

    /// Get model parameters count
    pub fn parameter_count(&self) -> usize {
        match self.model_size.as_str() {
            "tiny" => 39_000_000,
            "base" => 74_000_000,
            "small" => 244_000_000,
            "medium" => 769_000_000,
            "large" | "large-v2" | "large-v3" => 1_550_000_000,
            _ => 100_000_000, // Default estimate
        }
    }

    /// Check if model supports multilingual
    pub fn is_multilingual(&self) -> bool {
        self.multilingual
    }

    /// Get recommended batch size for model
    pub fn recommended_batch_size(&self) -> usize {
        match self.model_size.as_str() {
            "tiny" => 8,
            "base" => 6,
            "small" => 4,
            "medium" => 2,
            "large" | "large-v2" | "large-v3" => 1,
            _ => 4, // Default
        }
    }

    /// Get recommended quantization mode for model
    pub fn recommended_quantization(&self) -> QuantizationMode {
        match self.model_size.as_str() {
            "tiny" | "base" => QuantizationMode::None,
            "small" => QuantizationMode::F16,
            "medium" => QuantizationMode::F16,
            "large" | "large-v2" | "large-v3" => QuantizationMode::Dynamic,
            _ => QuantizationMode::None,
        }
    }

    /// Enable quantization for the model
    pub fn with_quantization(mut self, quantization: QuantizationMode) -> Self {
        self.quantization = quantization;
        self
    }

    /// Set multilingual support
    pub fn with_multilingual(mut self, multilingual: bool) -> Self {
        self.multilingual = multilingual;
        self
    }

    /// Set custom sample rate
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set custom chunk length
    pub fn with_chunk_length(mut self, chunk_length: usize) -> Self {
        self.chunk_length = chunk_length;
        self
    }
}

/// Whisper encoder implementation
pub struct WhisperEncoder {
    /// Positional embedding
    positional_embedding: Tensor,
    /// Convolutional layers
    conv1: candle_nn::Conv1d,
    conv2: candle_nn::Conv1d,
    /// Transformer blocks
    blocks: Vec<TransformerBlock>,
    /// Layer normalization
    ln_post: candle_nn::LayerNorm,
}

/// Transformer block for encoder
pub struct TransformerBlock {
    /// Self-attention
    attn: MultiHeadAttention,
    /// Layer normalization 1
    attn_ln: candle_nn::LayerNorm,
    /// MLP
    mlp: MLP,
    /// Layer normalization 2
    mlp_ln: candle_nn::LayerNorm,
}

/// MLP implementation with GELU activation
pub struct MLP {
    /// First linear layer
    c_fc: candle_nn::Linear,
    /// Second linear layer  
    c_proj: candle_nn::Linear,
}

impl WhisperEncoder {
    /// Creates a new Whisper encoder with the specified configuration
    ///
    /// # Arguments
    /// * `config` - Whisper model configuration containing encoder parameters
    /// * `device` - Device to create tensors on (CPU, CUDA, etc.)
    ///
    /// # Returns
    /// A new encoder instance or an error if initialization fails
    pub async fn new(config: &WhisperConfig, device: &Device) -> Result<Self, RecognitionError> {
        // Create positional embedding
        let positional_embedding = Tensor::zeros(
            (config.n_audio_ctx, config.n_audio_state),
            DType::F32,
            device,
        )
        .map_err(|e| RecognitionError::ModelLoadError {
            message: format!("Failed to create positional embedding: {e}"),
            source: Some(Box::new(e)),
        })?;

        // Create variable builder for weight initialization
        let vs = VarBuilder::zeros(DType::F32, device);

        // Create convolutional layers
        let conv1 = candle_nn::conv1d(
            config.n_mels,
            config.n_audio_state,
            3,
            Default::default(),
            vs.pp("conv1"),
        )
        .map_err(|e| RecognitionError::ModelLoadError {
            message: format!("Failed to create conv1: {e}"),
            source: Some(Box::new(e)),
        })?;

        let conv2 = candle_nn::conv1d(
            config.n_audio_state,
            config.n_audio_state,
            3,
            Default::default(),
            vs.pp("conv2"),
        )
        .map_err(|e| RecognitionError::ModelLoadError {
            message: format!("Failed to create conv2: {e}"),
            source: Some(Box::new(e)),
        })?;

        // Create transformer blocks
        let mut blocks = Vec::new();
        for i in 0..config.n_audio_layer {
            let block =
                TransformerBlock::new(config, device, &vs.pp(format!("blocks.{i}"))).await?;
            blocks.push(block);
        }

        // Create final layer normalization
        let ln_post =
            candle_nn::layer_norm(config.n_audio_state, 1e-5, vs.pp("ln_post")).map_err(|e| {
                RecognitionError::ModelLoadError {
                    message: format!("Failed to create layer norm: {e}"),
                    source: Some(Box::new(e)),
                }
            })?;

        Ok(Self {
            positional_embedding,
            conv1,
            conv2,
            blocks,
            ln_post,
        })
    }

    /// Forward pass through the encoder
    ///
    /// # Arguments
    /// * `x` - Input mel spectrogram tensor
    ///
    /// # Returns
    /// Encoded audio features tensor
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, RecognitionError> {
        let mut x = x.clone();

        // Apply first convolution with GELU
        x = self
            .conv1
            .forward(&x)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Conv1 forward failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        x = gelu_activation(&x).map_err(|e| RecognitionError::ModelError {
            message: format!("GELU activation failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        // Apply second convolution with GELU
        x = self
            .conv2
            .forward(&x)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Conv2 forward failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        x = gelu_activation(&x).map_err(|e| RecognitionError::ModelError {
            message: format!("GELU activation failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        // Add positional embedding
        x = x.broadcast_add(&self.positional_embedding).map_err(|e| {
            RecognitionError::ModelError {
                message: format!("Positional embedding addition failed: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        // Apply transformer blocks
        for (i, block) in self.blocks.iter().enumerate() {
            x = block
                .forward(&x)
                .map_err(|e| RecognitionError::ModelError {
                    message: format!("Transformer block {i} failed: {e}"),
                    source: Some(Box::new(e)),
                })?;
        }

        // Apply final layer normalization
        x = self
            .ln_post
            .forward(&x)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Final layer norm failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        Ok(x)
    }

    /// Forward pass with streaming support for real-time processing
    pub async fn forward_streaming(
        &self,
        x: &Tensor,
        chunk_size: usize,
    ) -> Result<Tensor, RecognitionError> {
        let seq_len = x.dim(1).map_err(|e| RecognitionError::ModelError {
            message: format!("Failed to get sequence length: {e}"),
            source: Some(Box::new(e)),
        })?;

        if seq_len <= chunk_size {
            return self.forward(x);
        }

        let mut outputs = Vec::new();
        let num_chunks = (seq_len + chunk_size - 1) / chunk_size;

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(seq_len);

            let chunk =
                x.narrow(1, start, end - start)
                    .map_err(|e| RecognitionError::ModelError {
                        message: format!("Failed to extract chunk {i}: {e}"),
                        source: Some(Box::new(e)),
                    })?;

            let output = self.forward(&chunk)?;
            outputs.push(output);
        }

        // Concatenate outputs
        if outputs.len() == 1 {
            Ok(outputs.into_iter().next().unwrap())
        } else {
            let output_refs: Vec<&Tensor> = outputs.iter().collect();
            Tensor::cat(&output_refs, 1).map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to concatenate streaming outputs: {e}"),
                source: Some(Box::new(e)),
            })
        }
    }
}

impl TransformerBlock {
    /// Creates a new transformer block for the encoder
    ///
    /// # Arguments
    /// * `config` - Whisper model configuration
    /// * `_device` - Device for tensor operations (currently unused)
    /// * `vs` - Variable builder for loading model weights
    ///
    /// # Returns
    /// A new transformer block instance
    pub async fn new(
        config: &WhisperConfig,
        _device: &Device,
        vs: &VarBuilder<'_>,
    ) -> Result<Self, RecognitionError> {
        let attn =
            MultiHeadAttention::new(config.n_audio_state, config.n_audio_head, &vs.pp("attn"))?;

        let attn_ln =
            candle_nn::layer_norm(config.n_audio_state, 1e-5, vs.pp("attn_ln")).map_err(|e| {
                RecognitionError::ModelLoadError {
                    message: format!("Failed to create attention layer norm: {e}"),
                    source: Some(Box::new(e)),
                }
            })?;

        let mlp = MLP::new(config.n_audio_state, &vs.pp("mlp"))?;

        let mlp_ln =
            candle_nn::layer_norm(config.n_audio_state, 1e-5, vs.pp("mlp_ln")).map_err(|e| {
                RecognitionError::ModelLoadError {
                    message: format!("Failed to create MLP layer norm: {e}"),
                    source: Some(Box::new(e)),
                }
            })?;

        Ok(Self {
            attn,
            attn_ln,
            mlp,
            mlp_ln,
        })
    }

    /// Forward pass through the transformer block
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// Output tensor after self-attention and MLP processing
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, RecognitionError> {
        // Pre-norm architecture: LayerNorm -> Attention -> Residual
        let attn_input = self
            .attn_ln
            .forward(x)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Attention layer norm failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let attn_output = self.attn.forward(&attn_input, &attn_input, &attn_input)?;
        let x = (x + attn_output).map_err(|e| RecognitionError::ModelError {
            message: format!("Attention residual connection failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        // Pre-norm architecture: LayerNorm -> MLP -> Residual
        let mlp_input = self
            .mlp_ln
            .forward(&x)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("MLP layer norm failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let mlp_output = self.mlp.forward(&mlp_input)?;
        let output = (x + mlp_output).map_err(|e| RecognitionError::ModelError {
            message: format!("MLP residual connection failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        Ok(output)
    }
}

impl MLP {
    /// Create a new MLP block for Whisper transformer
    ///
    /// # Arguments
    /// * `n_state` - The model dimension
    /// * `vs` - Variable builder for loading weights
    pub fn new(n_state: usize, vs: &VarBuilder) -> Result<Self, RecognitionError> {
        // Whisper uses 4x expansion in MLP
        let n_inner = n_state * 4;

        let c_fc = candle_nn::linear(n_state, n_inner, vs.pp("c_fc")).map_err(|e| {
            RecognitionError::ModelLoadError {
                message: format!("Failed to create MLP first layer: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        let c_proj = candle_nn::linear(n_inner, n_state, vs.pp("c_proj")).map_err(|e| {
            RecognitionError::ModelLoadError {
                message: format!("Failed to create MLP projection layer: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        Ok(Self { c_fc, c_proj })
    }

    /// Forward pass through the MLP layer
    ///
    /// Applies two linear transformations with GELU activation in between:
    /// output = Linear2(GELU(Linear1(x)))
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, seq_len, hidden_size]
    ///
    /// # Returns
    /// Transformed tensor with the same shape as input
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, RecognitionError> {
        // First linear layer
        let x = self
            .c_fc
            .forward(x)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("MLP first layer failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        // GELU activation
        let x = gelu_activation(&x).map_err(|e| RecognitionError::ModelError {
            message: format!("MLP GELU activation failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        // Projection layer
        let x = self
            .c_proj
            .forward(&x)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("MLP projection layer failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        Ok(x)
    }
}
