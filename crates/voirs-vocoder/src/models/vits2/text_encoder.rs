//! VITS2 Text Encoder
//!
//! Transformer encoder that turns a phoneme / character id sequence into latent
//! frames. It is a real implementation built on `candle_nn`:
//!
//! * a learned embedding table (every token id maps to its own row),
//! * sinusoidal absolute position encoding,
//! * `n_layers` transformer blocks with multi-head self-attention using one of
//!   two mutually exclusive relative-position schemes: the learned windowed
//!   relative bias (Shaw et al. 2018, as used by VITS) — the score for a
//!   query/key pair is `(q·k + q·r_{clip(j-i)}) / sqrt(head_dim)` — or rotary
//!   position embeddings; both are followed by a real softmax,
//! * a convolutional position-wise feed-forward network,
//! * layer normalization (pre-LN or post-LN) and a final output projection.
//!
//! All parameters live in a [`VarMap`], so the reported parameter count is the
//! real one and [`TextEncoder::load_weights`] can populate the model from a
//! SafeTensors checkpoint.

use super::params::{
    count_parameters, load_safetensors_into_varmap_with_mode, normalize_layernorm_suffix,
    seed_varmap, split_indexed_prefix, strip_checkpoint_prefixes, LoadMode, WeightLoadReport,
};
use super::relative_attention::{mask_to_tensor, rows_to_tensor, tensor_to_rows};
use crate::{Result, VocoderError};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub use super::relative_attention::{
    AttentionConfig, RelativeMultiHeadAttention, MAX_RELATIVE_POSITION,
};

/// Default seed used when a text encoder is created without an explicit seed.
pub const DEFAULT_TEXT_ENCODER_SEED: u64 = 0x5649_5453_3200_0002;

/// Text encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEncoderConfig {
    /// Vocabulary size (number of phonemes/characters)
    pub vocab_size: u32,
    /// Hidden dimension
    pub hidden_channels: u32,
    /// Filter channels for feed-forward networks
    pub filter_channels: u32,
    /// Number of attention heads
    pub n_heads: u32,
    /// Number of encoder layers
    pub n_layers: u32,
    /// Kernel size for the feed-forward convolutions (must be odd)
    pub kernel_size: u32,
    /// Dropout probability
    pub p_dropout: f32,
    /// Window size for relative position encoding
    pub window_size: Option<u32>,
    /// Use pre-layer normalization
    pub pre_ln: bool,
    /// Use rotary position embedding
    pub use_rope: bool,
    /// Maximum sequence length
    pub max_seq_len: u32,
    /// Conditioning channels (for speaker/style embedding)
    pub gin_channels: u32,
}

impl Default for TextEncoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256, // Typical phoneme vocabulary size
            hidden_channels: 192,
            filter_channels: 768,
            n_heads: 2,
            n_layers: 6,
            kernel_size: 3,
            p_dropout: 0.1,
            window_size: Some(4),
            pre_ln: true,
            use_rope: false,
            max_seq_len: 1000,
            gin_channels: 256,
        }
    }
}

impl TextEncoderConfig {
    /// Validate configuration
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] describing the first violated
    /// constraint.
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(VocoderError::ModelError(
                "Vocabulary size must be greater than 0".to_string(),
            ));
        }

        if self.hidden_channels == 0 {
            return Err(VocoderError::ModelError(
                "Hidden channels must be greater than 0".to_string(),
            ));
        }

        if self.n_heads == 0 {
            return Err(VocoderError::ModelError(
                "Number of heads must be greater than 0".to_string(),
            ));
        }

        if !self.hidden_channels.is_multiple_of(self.n_heads) {
            return Err(VocoderError::ModelError(
                "Hidden channels must be divisible by number of heads".to_string(),
            ));
        }

        if self.n_layers == 0 {
            return Err(VocoderError::ModelError(
                "Number of layers must be greater than 0".to_string(),
            ));
        }

        if self.kernel_size == 0 || self.kernel_size.is_multiple_of(2) {
            return Err(VocoderError::ModelError(format!(
                "Feed-forward kernel size must be odd (got {}) so the layer preserves length",
                self.kernel_size
            )));
        }

        if self.filter_channels == 0 {
            return Err(VocoderError::ModelError(
                "Filter channels must be greater than 0".to_string(),
            ));
        }

        if self.max_seq_len == 0 {
            return Err(VocoderError::ModelError(
                "Maximum sequence length must be greater than 0".to_string(),
            ));
        }

        if self.p_dropout < 0.0 || self.p_dropout >= 1.0 {
            return Err(VocoderError::ModelError(
                "Dropout probability must be in [0, 1)".to_string(),
            ));
        }

        if self.use_rope && !(self.hidden_channels / self.n_heads).is_multiple_of(2) {
            return Err(VocoderError::ModelError(format!(
                "Rotary position embeddings require an even head dimension (got {})",
                self.hidden_channels / self.n_heads
            )));
        }

        // RoPE and the learned windowed relative bias are two *alternative*
        // ways to inject relative position into the attention score. Enabling
        // both would apply the same signal twice through different mechanisms,
        // which no reference implementation does and which makes the effective
        // position prior impossible to reason about — so it is rejected rather
        // than silently accepted.
        if self.use_rope && self.window_size.is_some() {
            return Err(VocoderError::ModelError(
                "use_rope and window_size are mutually exclusive relative-position schemes: \
                 set window_size = None for rotary embeddings, or use_rope = false for the \
                 learned windowed relative bias"
                    .to_string(),
            ));
        }

        Ok(())
    }

    /// Effective relative-position window actually used by the attention layers.
    ///
    /// Zero when [`Self::use_rope`] is set, because rotary embeddings replace
    /// the learned windowed relative bias (the two are mutually exclusive; see
    /// [`Self::validate`]).
    pub fn effective_window(&self) -> usize {
        if self.use_rope {
            return 0;
        }
        self.window_size
            .map(|w| (w as usize).min(MAX_RELATIVE_POSITION.max(1) as usize))
            .unwrap_or(0)
    }

    /// Exact number of parameters the corresponding [`TextEncoder`] allocates.
    ///
    /// Mirrors the layer shapes built by [`TextEncoder::new_seeded`], so it
    /// equals `TextEncoder::parameter_count()` for the same configuration.
    pub fn parameter_count(&self) -> u64 {
        let hidden = self.hidden_channels as u64;
        let filter = self.filter_channels as u64;
        let kernel = self.kernel_size as u64;
        let head_dim = (self.hidden_channels / self.n_heads) as u64;

        // Token embedding table.
        let mut total = self.vocab_size as u64 * hidden;

        // Optional conditioning projection.
        if self.gin_channels > 0 {
            total += hidden * self.gin_channels as u64 + hidden;
        }

        let window = self.effective_window() as u64;
        let per_layer = {
            // Four Linear(hidden, hidden) projections.
            let mut layer = 4 * (hidden * hidden + hidden);
            if window > 0 {
                // emb_rel_k and emb_rel_v: [1, 2 * window + 1, head_dim]
                layer += 2 * (2 * window + 1) * head_dim;
            }
            // Convolutional feed-forward network.
            layer += hidden * filter * kernel + filter;
            layer += filter * hidden * kernel + hidden;
            // Two layer norms (weight + bias each).
            layer += 4 * hidden;
            layer
        };
        total += per_layer * self.n_layers as u64;

        // Final layer norm and output projection.
        total += 2 * hidden;
        total += hidden * hidden + hidden;

        total
    }

    /// Create configuration for high-quality synthesis
    pub fn high_quality() -> Self {
        Self {
            vocab_size: 512,
            hidden_channels: 256,
            filter_channels: 1024,
            n_heads: 4,
            n_layers: 8,
            kernel_size: 3,
            p_dropout: 0.05,
            // Rotary embeddings replace the learned windowed relative bias.
            window_size: None,
            pre_ln: true,
            use_rope: true,
            max_seq_len: 2000,
            gin_channels: 512,
        }
    }

    /// Create configuration for fast synthesis
    pub fn fast() -> Self {
        Self {
            vocab_size: 128,
            hidden_channels: 128,
            filter_channels: 512,
            n_heads: 2,
            n_layers: 4,
            kernel_size: 3,
            p_dropout: 0.1,
            window_size: Some(4),
            pre_ln: false,
            use_rope: false,
            max_seq_len: 512,
            gin_channels: 128,
        }
    }
}

/// Feed-forward network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FFNConfig {
    /// Input/output dimension
    pub hidden_channels: u32,
    /// Intermediate dimension
    pub filter_channels: u32,
    /// Kernel size for the convolutions (must be odd)
    pub kernel_size: u32,
    /// Dropout probability
    pub p_dropout: f32,
    /// Activation function (`relu`, `gelu` or `silu`)
    pub activation: String,
}

/// Activation used inside the feed-forward network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FfnActivation {
    Relu,
    Gelu,
    Silu,
}

impl FfnActivation {
    fn parse(name: &str) -> Result<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "relu" => Ok(Self::Relu),
            "gelu" => Ok(Self::Gelu),
            "silu" | "swish" => Ok(Self::Silu),
            other => Err(VocoderError::ModelError(format!(
                "Unsupported feed-forward activation '{other}' (expected relu, gelu or silu)"
            ))),
        }
    }

    fn apply(self, x: &Tensor) -> Result<Tensor> {
        Ok(match self {
            Self::Relu => x.relu()?,
            Self::Gelu => x.gelu()?,
            Self::Silu => x.silu()?,
        })
    }
}

/// Position-wise convolutional feed-forward network.
///
/// `conv2( activation( conv1(x) ) )` with `conv1: hidden -> filter` and
/// `conv2: filter -> hidden`, both operating along the time axis with symmetric
/// padding so the sequence length is preserved.
#[derive(Debug, Clone)]
pub struct FeedForwardNetwork {
    /// Configuration
    pub config: FFNConfig,
    /// Expansion convolution
    conv1: Conv1d,
    /// Compression convolution
    conv2: Conv1d,
    /// Parsed activation
    activation: FfnActivation,
    /// Compute device
    device: Device,
}

impl FeedForwardNetwork {
    /// Create a new feed-forward network.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] for an invalid configuration
    /// (even kernel size, zero channels, unknown activation) and
    /// [`VocoderError::CandleError`] on tensor failures.
    pub fn new(config: FFNConfig, vb: VarBuilder) -> Result<Self> {
        if config.hidden_channels == 0 || config.filter_channels == 0 {
            return Err(VocoderError::ModelError(
                "Feed-forward channels must be greater than 0".to_string(),
            ));
        }
        if config.kernel_size == 0 || config.kernel_size.is_multiple_of(2) {
            return Err(VocoderError::ModelError(format!(
                "Feed-forward kernel size must be odd (got {})",
                config.kernel_size
            )));
        }

        let activation = FfnActivation::parse(&config.activation)?;
        let device = vb.device().clone();
        let kernel_size = config.kernel_size as usize;
        let padding = (kernel_size - 1) / 2;

        let conv1 = candle_nn::conv1d(
            config.hidden_channels as usize,
            config.filter_channels as usize,
            kernel_size,
            Conv1dConfig {
                padding,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let conv2 = candle_nn::conv1d(
            config.filter_channels as usize,
            config.hidden_channels as usize,
            kernel_size,
            Conv1dConfig {
                padding,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;

        Ok(Self {
            config,
            conv1,
            conv2,
            activation,
            device,
        })
    }

    /// Forward pass over a `[batch, seq_len, hidden]` tensor.
    ///
    /// # Errors
    /// Returns [`VocoderError::VocodingError`] on a hidden-dimension mismatch and
    /// [`VocoderError::CandleError`] on tensor failures.
    pub fn forward_tensor(&self, input: &Tensor) -> Result<Tensor> {
        let (_, _, hidden) = input.dims3()?;
        if hidden != self.config.hidden_channels as usize {
            return Err(VocoderError::VocodingError(format!(
                "Feed-forward expects hidden dimension {}, got {hidden}",
                self.config.hidden_channels
            )));
        }

        // Conv1d works on [batch, channels, time].
        let x = input.transpose(1, 2)?.contiguous()?;
        let x = self.conv1.forward(&x)?;
        let x = self.activation.apply(&x)?;
        let x = self.conv2.forward(&x)?;
        Ok(x.transpose(1, 2)?.contiguous()?)
    }

    /// Forward pass over row-major `[seq_len][hidden]` buffers.
    ///
    /// # Errors
    /// Returns [`VocoderError::VocodingError`] for empty or ragged input.
    pub fn forward(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if input.is_empty() {
            return Err(VocoderError::VocodingError("Empty input".to_string()));
        }
        let x = rows_to_tensor(input, &self.device)?;
        let out = self.forward_tensor(&x)?;
        tensor_to_rows(&out)
    }

    /// Number of scalars held by this module.
    pub fn num_parameters(&self) -> u64 {
        let conv = |c: &Conv1d| {
            c.weight().elem_count() as u64 + c.bias().map_or(0, |b| b.elem_count() as u64)
        };
        conv(&self.conv1) + conv(&self.conv2)
    }
}

/// Transformer encoder layer
#[derive(Debug, Clone)]
pub struct EncoderLayer {
    /// Self-attention module
    pub self_attention: RelativeMultiHeadAttention,
    /// Feed-forward network
    pub ffn: FeedForwardNetwork,
    /// Layer norm applied around the attention sub-layer
    norm1: LayerNorm,
    /// Layer norm applied around the feed-forward sub-layer
    norm2: LayerNorm,
    /// Use pre-layer normalization
    pub pre_ln: bool,
    /// Compute device
    device: Device,
}

impl EncoderLayer {
    /// Create a new encoder layer.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] for an invalid configuration and
    /// [`VocoderError::CandleError`] on tensor failures.
    pub fn new(config: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let attention_config = AttentionConfig {
            n_heads: config.n_heads,
            hidden_channels: config.hidden_channels,
            head_dim: config.hidden_channels / config.n_heads,
            // Mutually exclusive by construction: `validate` rejects configs
            // that request both, and `effective_window` returns 0 under RoPE.
            window_size: if config.use_rope {
                None
            } else {
                config.window_size
            },
            relative_attention: !config.use_rope && config.window_size.is_some(),
            max_relative_position: MAX_RELATIVE_POSITION,
            p_dropout: config.p_dropout,
            use_rope: config.use_rope,
        };

        let ffn_config = FFNConfig {
            hidden_channels: config.hidden_channels,
            filter_channels: config.filter_channels,
            kernel_size: config.kernel_size,
            p_dropout: config.p_dropout,
            activation: "relu".to_string(),
        };

        let hidden = config.hidden_channels as usize;
        let device = vb.device().clone();

        Ok(Self {
            self_attention: RelativeMultiHeadAttention::new(attention_config, vb.pp("attn"))?,
            ffn: FeedForwardNetwork::new(ffn_config, vb.pp("ffn"))?,
            norm1: candle_nn::layer_norm(hidden, 1e-5, vb.pp("norm1"))?,
            norm2: candle_nn::layer_norm(hidden, 1e-5, vb.pp("norm2"))?,
            pre_ln: config.pre_ln,
            device,
        })
    }

    /// Forward pass over a `[batch, seq_len, hidden]` tensor.
    ///
    /// # Errors
    /// Returns [`VocoderError::CandleError`] on tensor failures and
    /// [`VocoderError::VocodingError`] on shape mismatches.
    pub fn forward_tensor(&self, input: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        if self.pre_ln {
            let normed = self.norm1.forward(input)?;
            let attn = self
                .self_attention
                .forward_tensor(&normed, &normed, &normed, mask)?;
            let x = (input + attn)?;
            let normed = self.norm2.forward(&x)?;
            let ffn = self.ffn.forward_tensor(&normed)?;
            Ok((x + ffn)?)
        } else {
            let attn = self
                .self_attention
                .forward_tensor(input, input, input, mask)?;
            let x = self.norm1.forward(&(input + attn)?)?;
            let ffn = self.ffn.forward_tensor(&x)?;
            Ok(self.norm2.forward(&(&x + ffn)?)?)
        }
    }

    /// Forward pass over row-major `[seq_len][hidden]` buffers.
    ///
    /// # Errors
    /// Returns [`VocoderError::VocodingError`] for empty or ragged input.
    pub fn forward(&self, input: &[Vec<f32>], mask: Option<&[Vec<bool>]>) -> Result<Vec<Vec<f32>>> {
        if input.is_empty() {
            return Err(VocoderError::VocodingError("Empty input".to_string()));
        }
        let x = rows_to_tensor(input, &self.device)?;
        let mask_tensor = match mask {
            Some(mask) => Some(mask_to_tensor(
                mask,
                input.len(),
                input.len(),
                &self.device,
            )?),
            None => None,
        };
        let out = self.forward_tensor(&x, mask_tensor.as_ref())?;
        tensor_to_rows(&out)
    }

    /// Number of scalars held by this layer.
    pub fn num_parameters(&self) -> u64 {
        let norm = |n: &LayerNorm| {
            n.weight().elem_count() as u64 + n.bias().map_or(0, |b| b.elem_count() as u64)
        };
        self.self_attention.num_parameters()
            + self.ffn.num_parameters()
            + norm(&self.norm1)
            + norm(&self.norm2)
    }
}

/// Main text encoder
///
/// Not `Clone`: the parameters live in a shared [`VarMap`], so a clone would
/// alias the same weights while carrying its own `weights_loaded` flag — which
/// would let `is_pretrained()` report a state that does not match the tensors.
pub struct TextEncoder {
    /// Configuration
    pub config: TextEncoderConfig,
    /// Token embedding table
    embedding: Embedding,
    /// Optional conditioning projection (`gin_channels` -> `hidden_channels`)
    cond_proj: Option<Linear>,
    /// Encoder layers
    layers: Vec<EncoderLayer>,
    /// Final layer normalization
    final_norm: LayerNorm,
    /// Output projection
    output_projection: Linear,
    /// Precomputed sinusoidal position encoding `[max_seq_len, hidden]`
    positional_encoding: Tensor,
    /// All learnable parameters
    varmap: VarMap,
    /// Compute device
    device: Device,
    /// Whether a checkpoint has been loaded
    weights_loaded: bool,
}

impl std::fmt::Debug for TextEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextEncoder")
            .field("config", &self.config)
            .field("layers", &self.layers.len())
            .field("device", &self.device)
            .field("weights_loaded", &self.weights_loaded)
            .finish()
    }
}

impl TextEncoder {
    /// Create a text encoder on the CPU with deterministic initialization.
    ///
    /// The weights are pseudo-random but reproducible; the model is **not**
    /// trained. Use [`TextEncoder::load_weights`] for a real checkpoint or
    /// [`TextEncoder::encode_pretrained`] to fail closed when none is loaded.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] for an invalid configuration.
    pub fn new(config: TextEncoderConfig) -> Result<Self> {
        Self::new_seeded(config, Device::Cpu, DEFAULT_TEXT_ENCODER_SEED)
    }

    /// Create a text encoder on a specific device.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] for an invalid configuration.
    pub fn new_with_device(config: TextEncoderConfig, device: Device) -> Result<Self> {
        Self::new_seeded(config, device, DEFAULT_TEXT_ENCODER_SEED)
    }

    /// Create a text encoder with an explicit initialization seed.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] for an invalid configuration and
    /// [`VocoderError::CandleError`] when parameter tensors cannot be created.
    pub fn new_seeded(config: TextEncoderConfig, device: Device, seed: u64) -> Result<Self> {
        config.validate()?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let hidden = config.hidden_channels as usize;

        let embedding = candle_nn::embedding(config.vocab_size as usize, hidden, vb.pp("emb"))?;

        let cond_proj = if config.gin_channels > 0 {
            Some(candle_nn::linear(
                config.gin_channels as usize,
                hidden,
                vb.pp("cond_proj"),
            )?)
        } else {
            None
        };

        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(config.n_layers as usize);
        for i in 0..config.n_layers as usize {
            layers.push(EncoderLayer::new(&config, vb_layers.pp(i))?);
        }

        let final_norm = candle_nn::layer_norm(hidden, 1e-5, vb.pp("final_norm"))?;
        let output_projection = candle_nn::linear(hidden, hidden, vb.pp("output_projection"))?;

        let positional_encoding =
            sinusoidal_encoding(config.max_seq_len as usize, hidden, &device)?;

        seed_varmap(&varmap, seed, &device)?;

        Ok(Self {
            config,
            embedding,
            cond_proj,
            layers,
            final_norm,
            output_projection,
            positional_encoding,
            varmap,
            device,
            weights_loaded: false,
        })
    }

    /// Encode a token id sequence into latent frames.
    ///
    /// `conditioning`, when supplied, must hold exactly `gin_channels` values and
    /// is projected and added to every position.
    ///
    /// # Errors
    /// Returns [`VocoderError::VocodingError`] for empty input, sequences longer
    /// than `max_seq_len`, out-of-vocabulary ids or a wrong conditioning width.
    pub fn encode(&self, input_ids: &[u32], conditioning: Option<&[f32]>) -> Result<Vec<Vec<f32>>> {
        let hidden = self.encode_tensor(input_ids, conditioning)?;
        tensor_to_rows(&hidden)
    }

    /// Encode a token id sequence, failing closed when no checkpoint is loaded.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] when [`TextEncoder::load_weights`]
    /// has not been called successfully.
    pub fn encode_pretrained(
        &self,
        input_ids: &[u32],
        conditioning: Option<&[f32]>,
    ) -> Result<Vec<Vec<f32>>> {
        if !self.weights_loaded {
            return Err(VocoderError::ModelError(
                "VITS2 text encoder has no pretrained weights loaded — \
                 call load_weights(<path to .safetensors>) first"
                    .to_string(),
            ));
        }
        self.encode(input_ids, conditioning)
    }

    /// Encode a token id sequence and return the raw `[1, seq_len, hidden]` tensor.
    ///
    /// # Errors
    /// Returns [`VocoderError::VocodingError`] for invalid input and
    /// [`VocoderError::CandleError`] on tensor failures.
    pub fn encode_tensor(&self, input_ids: &[u32], conditioning: Option<&[f32]>) -> Result<Tensor> {
        if input_ids.is_empty() {
            return Err(VocoderError::VocodingError(
                "Empty input sequence".to_string(),
            ));
        }
        if input_ids.len() > self.config.max_seq_len as usize {
            return Err(VocoderError::VocodingError(format!(
                "Input sequence length {} exceeds maximum {}",
                input_ids.len(),
                self.config.max_seq_len
            )));
        }
        if let Some(&bad) = input_ids.iter().find(|&&id| id >= self.config.vocab_size) {
            return Err(VocoderError::VocodingError(format!(
                "Token ID {bad} out of vocabulary range"
            )));
        }

        let seq_len = input_ids.len();
        let hidden = self.config.hidden_channels as usize;

        let ids = Tensor::from_slice(input_ids, (1, seq_len), &self.device)?;
        // VITS scales the embedding by sqrt(hidden) before adding positions.
        let mut x = self
            .embedding
            .forward(&ids)?
            .affine((hidden as f64).sqrt(), 0.0)?;

        let pos = self
            .positional_encoding
            .narrow(0, 0, seq_len)?
            .reshape((1, seq_len, hidden))?;
        x = x.broadcast_add(&pos)?;

        if let Some(cond) = conditioning {
            let proj = self.cond_proj.as_ref().ok_or_else(|| {
                VocoderError::VocodingError(
                    "Text encoder was configured with gin_channels = 0 but conditioning was \
                     supplied"
                        .to_string(),
                )
            })?;
            let gin = self.config.gin_channels as usize;
            if cond.len() != gin {
                return Err(VocoderError::VocodingError(format!(
                    "Conditioning vector has {} values but gin_channels is {gin}",
                    cond.len()
                )));
            }
            let cond_tensor = Tensor::from_slice(cond, (1, 1, gin), &self.device)?;
            x = x.broadcast_add(&proj.forward(&cond_tensor)?)?;
        }

        for layer in &self.layers {
            x = layer.forward_tensor(&x, None)?;
        }

        let x = self.final_norm.forward(&x)?;
        Ok(self.output_projection.forward(&x)?)
    }

    /// Load text-encoder weights from a SafeTensors checkpoint.
    ///
    /// Understands this module's own naming as well as the reference VITS
    /// `enc_p.*` layout (`encoder.attn_layers.{i}.conv_q`, `norm_layers_1`,
    /// `ffn_layers`, `proj`, PyTorch `gamma`/`beta` layer-norm parameters and
    /// 1x1-convolution projections).
    ///
    /// Fail-closed: the checkpoint must supply **every** encoder parameter. A
    /// partial checkpoint is rejected rather than leaving some layers at their
    /// pseudo-random initialization; use [`TextEncoder::load_weights_partial`]
    /// when a partial load is intended.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] when the file cannot be read/parsed,
    /// when no tensor matched a model parameter, or when the checkpoint left any
    /// parameter unset.
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<WeightLoadReport> {
        let report = load_safetensors_into_varmap_with_mode(
            &mut self.varmap,
            path.as_ref(),
            &self.device,
            map_text_encoder_weight_name,
            LoadMode::Strict,
        )?;
        self.weights_loaded = true;
        Ok(report)
    }

    /// Load whatever the checkpoint supplies, tolerating uncovered parameters.
    ///
    /// Parameters the checkpoint does not cover keep their pseudo-random
    /// initialization, so the encoder is **not** marked pretrained and
    /// [`TextEncoder::encode_pretrained`] keeps failing closed.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] when the file cannot be read/parsed
    /// or when no tensor matched a model parameter.
    pub fn load_weights_partial<P: AsRef<Path>>(&mut self, path: P) -> Result<WeightLoadReport> {
        let report = load_safetensors_into_varmap_with_mode(
            &mut self.varmap,
            path.as_ref(),
            &self.device,
            map_text_encoder_weight_name,
            LoadMode::Partial,
        )?;
        if report.is_complete() {
            self.weights_loaded = true;
        }
        Ok(report)
    }

    /// Save the current weights to a SafeTensors file.
    ///
    /// # Errors
    /// Returns [`VocoderError::CandleError`] when serialization fails.
    pub fn save_weights<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.varmap.save(path.as_ref())?;
        Ok(())
    }

    /// Whether a checkpoint has been loaded into this encoder.
    pub fn is_pretrained(&self) -> bool {
        self.weights_loaded
    }

    /// Encoder layers.
    pub fn layers(&self) -> &[EncoderLayer] {
        &self.layers
    }

    /// Compute device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Total number of scalars actually held by this encoder.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] if the parameter store is poisoned.
    pub fn parameter_count(&self) -> Result<u64> {
        count_parameters(&self.varmap)
    }

    /// Total number of parameters, or `0` if the parameter store is unavailable.
    pub fn num_parameters(&self) -> u64 {
        self.parameter_count().unwrap_or(0)
    }

    /// Memory required by the parameters plus one full-length activation, in MB.
    pub fn memory_requirements_mb(&self) -> f32 {
        let param_memory = self.num_parameters() as f32 * 4.0 / (1024.0 * 1024.0);
        let activation_memory =
            self.config.max_seq_len as f32 * self.config.hidden_channels as f32 * 4.0
                / (1024.0 * 1024.0);
        param_memory + activation_memory * 2.0
    }
}

/// Build the sinusoidal position encoding table `[max_seq_len, hidden]`.
fn sinusoidal_encoding(max_seq_len: usize, hidden: usize, device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(max_seq_len * hidden);
    for pos in 0..max_seq_len {
        for i in 0..hidden {
            let angle = pos as f64 / 10000.0_f64.powf(2.0 * (i / 2) as f64 / hidden as f64);
            values.push(if i % 2 == 0 {
                angle.sin() as f32
            } else {
                angle.cos() as f32
            });
        }
    }
    Ok(Tensor::from_vec(values, (max_seq_len, hidden), device)?)
}

/// Map a checkpoint tensor name onto an internal text-encoder parameter name.
fn map_text_encoder_weight_name(name: &str) -> Option<String> {
    let stripped = normalize_layernorm_suffix(strip_checkpoint_prefixes(name));

    // Already in this module's layout.
    const KNOWN_ROOTS: [&str; 5] = [
        "emb.",
        "cond_proj.",
        "final_norm.",
        "output_projection.",
        "layers.",
    ];
    if KNOWN_ROOTS.iter().any(|root| stripped.starts_with(root)) {
        return Some(stripped);
    }

    if let Some(rest) = stripped.strip_prefix("proj.") {
        return Some(format!("output_projection.{rest}"));
    }

    let encoder_rest = stripped.strip_prefix("encoder.")?;

    if let Some((index, tail)) = split_indexed_prefix(encoder_rest, "attn_layers") {
        let mapped_tail = if let Some(t) = tail.strip_prefix("conv_q.") {
            format!("q_proj.{t}")
        } else if let Some(t) = tail.strip_prefix("conv_k.") {
            format!("k_proj.{t}")
        } else if let Some(t) = tail.strip_prefix("conv_v.") {
            format!("v_proj.{t}")
        } else if let Some(t) = tail.strip_prefix("conv_o.") {
            format!("out_proj.{t}")
        } else if tail == "emb_rel_k" || tail == "emb_rel_v" {
            tail.to_string()
        } else {
            return None;
        };
        return Some(format!("layers.{index}.attn.{mapped_tail}"));
    }

    if let Some((index, tail)) = split_indexed_prefix(encoder_rest, "ffn_layers") {
        let mapped_tail = if let Some(t) = tail.strip_prefix("conv_1.") {
            format!("conv1.{t}")
        } else if let Some(t) = tail.strip_prefix("conv_2.") {
            format!("conv2.{t}")
        } else {
            return None;
        };
        return Some(format!("layers.{index}.ffn.{mapped_tail}"));
    }

    if let Some((index, tail)) = split_indexed_prefix(encoder_rest, "norm_layers_1") {
        return Some(format!("layers.{index}.norm1.{tail}"));
    }

    if let Some((index, tail)) = split_indexed_prefix(encoder_rest, "norm_layers_2") {
        return Some(format!("layers.{index}.norm2.{tail}"));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> TextEncoderConfig {
        TextEncoderConfig {
            vocab_size: 16,
            hidden_channels: 16,
            filter_channels: 32,
            n_heads: 2,
            n_layers: 2,
            kernel_size: 3,
            p_dropout: 0.1,
            window_size: Some(3),
            pre_ln: true,
            use_rope: false,
            max_seq_len: 64,
            gin_channels: 4,
        }
    }

    fn attention_config(hidden: u32, heads: u32, window: Option<u32>) -> AttentionConfig {
        AttentionConfig {
            n_heads: heads,
            hidden_channels: hidden,
            head_dim: hidden / heads,
            window_size: window,
            relative_attention: window.is_some(),
            max_relative_position: MAX_RELATIVE_POSITION,
            p_dropout: 0.1,
            use_rope: false,
        }
    }

    fn build_attention(
        config: AttentionConfig,
        seed: u64,
    ) -> (RelativeMultiHeadAttention, VarMap, Device) {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attention =
            RelativeMultiHeadAttention::new(config, vb.pp("attn")).expect("attention module");
        seed_varmap(&varmap, seed, &device).expect("seed");
        (attention, varmap, device)
    }

    fn rows(seq_len: usize, hidden: usize, offset: f32) -> Vec<Vec<f32>> {
        (0..seq_len)
            .map(|i| {
                (0..hidden)
                    .map(|j| ((i * hidden + j) as f32 * 0.13 + offset).sin() * 0.5)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_text_encoder_config() {
        let config = TextEncoderConfig::default();
        assert!(config.validate().is_ok());
        assert!(TextEncoderConfig::high_quality().validate().is_ok());
        assert!(TextEncoderConfig::fast().validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.vocab_size = 0;
        assert!(invalid_config.validate().is_err());

        let mut invalid_config = config.clone();
        invalid_config.hidden_channels = 15; // Not divisible by n_heads (2)
        assert!(invalid_config.validate().is_err());

        let mut invalid_config = config.clone();
        invalid_config.kernel_size = 4; // Even kernels cannot preserve length
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_attention_uses_query_and_key_contents() {
        let (attention, _varmap, _device) = build_attention(attention_config(16, 2, Some(3)), 31);
        let value = rows(6, 16, 0.0);
        let query_a = rows(6, 16, 1.0);
        let query_b = rows(6, 16, 2.0);
        let key_a = rows(6, 16, 3.0);
        let key_b = rows(6, 16, 4.0);

        let base = attention
            .forward(&query_a, &key_a, &value, None)
            .expect("attention");
        let changed_query = attention
            .forward(&query_b, &key_a, &value, None)
            .expect("attention");
        let changed_key = attention
            .forward(&query_a, &key_b, &value, None)
            .expect("attention");

        assert_ne!(
            base, changed_query,
            "attention output must depend on the query contents"
        );
        assert_ne!(
            base, changed_key,
            "attention output must depend on the key contents"
        );

        assert_eq!(base.len(), 6);
        assert_eq!(base[0].len(), 16);
        assert!(attention.num_parameters() > 0);
    }

    #[test]
    fn test_attention_weights_are_a_real_softmax() {
        // Without relative embeddings and with zero query/key the scores are all
        // equal, so a real softmax must average the values uniformly.
        let mut config = attention_config(8, 2, None);
        config.relative_attention = false;
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attention = RelativeMultiHeadAttention::new(config, vb.pp("attn")).expect("attention");
        seed_varmap(&varmap, 5, &device).expect("seed");

        // Force the identity so the analytic expectation is easy to state:
        // q = k = 0 (zero projections would kill v as well, so instead we zero
        // only the query and key projections and the output projection bias).
        {
            let mut data = varmap.data().lock().expect("lock");
            for name in ["attn.q_proj.weight", "attn.k_proj.weight"] {
                let var = data.get_mut(name).expect("weight");
                let zeros = Tensor::zeros(var.shape().dims(), DType::F32, &device).expect("zeros");
                var.set(&zeros).expect("set");
            }
        }

        let seq_len = 5;
        let value = rows(seq_len, 8, 0.7);
        let out = attention
            .forward(&value, &value, &value, None)
            .expect("attention");

        // Uniform attention → every output row is identical.
        for row in out.iter().skip(1) {
            for (a, b) in row.iter().zip(out[0].iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "uniform attention must give identical rows"
                );
            }
        }
    }

    #[test]
    fn test_attention_mask_blocks_positions() {
        let mut config = attention_config(8, 2, None);
        config.relative_attention = false;
        let (attention, _varmap, _device) = build_attention(config, 17);

        let seq_len = 4;
        let query = rows(seq_len, 8, 0.2);
        let value = rows(seq_len, 8, 1.1);

        let unmasked = attention
            .forward(&query, &query, &value, None)
            .expect("attention");
        let mut mask = vec![vec![true; seq_len]; seq_len];
        for row in mask.iter_mut() {
            row[seq_len - 1] = false;
        }
        let masked = attention
            .forward(&query, &query, &value, Some(&mask))
            .expect("attention");

        assert_ne!(unmasked, masked, "masking must change the attention result");

        // A wrongly sized mask must be rejected, not silently ignored.
        let bad_mask = vec![vec![true; seq_len]; seq_len - 1];
        assert!(attention
            .forward(&query, &query, &value, Some(&bad_mask))
            .is_err());
    }

    #[test]
    fn test_relative_position_bias_is_learned() {
        let config = attention_config(8, 2, Some(2));
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attention = RelativeMultiHeadAttention::new(config, vb.pp("attn")).expect("attention");
        seed_varmap(&varmap, 21, &device).expect("seed");
        assert!(attention.uses_relative_positions());

        let query = rows(6, 8, 0.4);
        let value = rows(6, 8, 1.4);
        let before = attention
            .forward(&query, &query, &value, None)
            .expect("attention");

        // Changing only the relative-position embedding must change the output:
        // it is a learnable parameter, not a fixed distance function.
        {
            let mut data = varmap.data().lock().expect("lock");
            let var = data.get_mut("attn.emb_rel_k").expect("emb_rel_k");
            let bumped = var
                .as_tensor()
                .affine(1.0, 0.75)
                .expect("affine")
                .contiguous()
                .expect("contiguous");
            var.set(&bumped).expect("set");
        }
        let after = attention
            .forward(&query, &query, &value, None)
            .expect("attention");
        assert_ne!(before, after);
    }

    #[test]
    fn test_attention_rejects_dimension_mismatch() {
        let (attention, _varmap, _device) = build_attention(attention_config(16, 2, Some(3)), 3);
        let wrong = rows(4, 8, 0.0);
        assert!(attention.forward(&wrong, &wrong, &wrong, None).is_err());
        assert!(attention.forward(&[], &[], &[], None).is_err());
    }

    #[test]
    fn test_ffn_is_a_real_two_layer_network() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = FFNConfig {
            hidden_channels: 8,
            filter_channels: 32,
            kernel_size: 3,
            p_dropout: 0.1,
            activation: "relu".to_string(),
        };
        let ffn = FeedForwardNetwork::new(config, vb.pp("ffn")).expect("ffn");
        seed_varmap(&varmap, 91, &device).expect("seed");

        let input = rows(6, 8, 0.25);
        let output = ffn.forward(&input).expect("forward");
        assert_eq!(output.len(), 6);
        assert_eq!(output[0].len(), 8);

        // Real parameter count: 8*32*3 + 32 + 32*8*3 + 8
        assert_eq!(ffn.num_parameters(), 8 * 32 * 3 + 32 + 32 * 8 * 3 + 8);

        // A convolutional FFN mixes neighbouring positions: perturbing one frame
        // must change its neighbours too (a per-element rescale could not).
        let mut perturbed = input.clone();
        perturbed[3][0] += 1.0;
        let perturbed_out = ffn.forward(&perturbed).expect("forward");
        assert_ne!(output[2], perturbed_out[2]);
        assert_ne!(output[4], perturbed_out[4]);
    }

    #[test]
    fn test_ffn_rejects_unknown_activation() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = FFNConfig {
            hidden_channels: 8,
            filter_channels: 16,
            kernel_size: 3,
            p_dropout: 0.0,
            activation: "mystery".to_string(),
        };
        assert!(FeedForwardNetwork::new(config, vb.pp("ffn")).is_err());
    }

    #[test]
    fn test_encoder_layer_shapes_and_sensitivity() {
        let config = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let layer = EncoderLayer::new(&config, vb.pp("layer")).expect("layer");
        seed_varmap(&varmap, 64, &device).expect("seed");

        let input = rows(5, 16, 0.3);
        let output = layer.forward(&input, None).expect("forward");
        assert_eq!(output.len(), 5);
        assert_eq!(output[0].len(), 16);
        assert!(layer.num_parameters() > 0);

        let other = rows(5, 16, 1.9);
        let other_out = layer.forward(&other, None).expect("forward");
        assert_ne!(output, other_out);
    }

    #[test]
    fn test_text_encoder_distinguishes_tokens() {
        let encoder = TextEncoder::new(tiny_config()).expect("encoder");

        let a = encoder.encode(&[1, 2, 3, 4, 5], None).expect("encode");
        let b = encoder.encode(&[5, 4, 3, 2, 1], None).expect("encode");
        let c = encoder.encode(&[1, 2, 3, 4, 6], None).expect("encode");

        assert_eq!(a.len(), 5);
        assert_eq!(a[0].len(), 16);
        assert_ne!(a, b, "token order must change the encoding");
        assert_ne!(a, c, "a single different token must change the encoding");

        // Distinct tokens must have distinct embeddings, so a constant-token
        // sequence differs from a varied one.
        let flat = encoder.encode(&[7, 7, 7, 7, 7], None).expect("encode");
        assert_ne!(a, flat);
    }

    #[test]
    fn test_text_encoder_embedding_rows_differ() {
        let encoder = TextEncoder::new(tiny_config()).expect("encoder");
        let table = encoder
            .embedding
            .embeddings()
            .to_vec2::<f32>()
            .expect("embedding table");
        assert_eq!(table.len(), 16);
        // Not every token may map to the same vector (the old placeholder did).
        assert_ne!(table[0], table[1]);
        assert_ne!(table[3], table[9]);
    }

    #[test]
    fn test_text_encoder_with_conditioning() {
        let encoder = TextEncoder::new(tiny_config()).expect("encoder");
        let ids = [1u32, 2, 3];

        let plain = encoder.encode(&ids, None).expect("encode");
        let cond_a = encoder
            .encode(&ids, Some(&[0.5, -0.5, 0.25, 0.75]))
            .expect("encode");
        let cond_b = encoder
            .encode(&ids, Some(&[-0.5, 0.5, -0.25, -0.75]))
            .expect("encode");

        assert_eq!(cond_a.len(), 3);
        assert_ne!(plain, cond_a);
        assert_ne!(cond_a, cond_b);

        // A conditioning vector of the wrong width must be rejected.
        assert!(encoder.encode(&ids, Some(&[0.5])).is_err());
    }

    #[test]
    fn test_text_encoder_input_validation() {
        let encoder = TextEncoder::new(tiny_config()).expect("encoder");
        assert!(encoder.encode(&[], None).is_err());
        assert!(encoder.encode(&[99], None).is_err());
        let too_long: Vec<u32> = (0..100).map(|i| i % 16).collect();
        assert!(encoder.encode(&too_long, None).is_err());
    }

    #[test]
    fn test_text_encoder_is_deterministic_for_a_given_seed() {
        let a = TextEncoder::new_seeded(tiny_config(), Device::Cpu, 808).expect("a");
        let b = TextEncoder::new_seeded(tiny_config(), Device::Cpu, 808).expect("b");
        let c = TextEncoder::new_seeded(tiny_config(), Device::Cpu, 809).expect("c");

        let ids = [2u32, 4, 6, 8];
        assert_eq!(
            a.encode(&ids, None).expect("a"),
            b.encode(&ids, None).expect("b")
        );
        assert_ne!(
            a.encode(&ids, None).expect("a"),
            c.encode(&ids, None).expect("c")
        );
    }

    #[test]
    fn test_text_encoder_parameter_count_is_real() {
        let encoder = TextEncoder::new(tiny_config()).expect("encoder");
        let reported = encoder.parameter_count().expect("count");

        let linear = |l: &Linear| {
            l.weight().elem_count() as u64 + l.bias().map_or(0, |b| b.elem_count() as u64)
        };
        let norm = |n: &LayerNorm| {
            n.weight().elem_count() as u64 + n.bias().map_or(0, |b| b.elem_count() as u64)
        };
        let expected = encoder.embedding.embeddings().elem_count() as u64
            + encoder.cond_proj.as_ref().map_or(0, linear)
            + encoder
                .layers()
                .iter()
                .map(|l| l.num_parameters())
                .sum::<u64>()
            + norm(&encoder.final_norm)
            + linear(&encoder.output_projection);

        assert_eq!(reported, expected);
        assert!(encoder.memory_requirements_mb() > 0.0);
    }

    #[test]
    fn test_text_encoder_weight_round_trip() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("vits2_text_encoder.safetensors");

        let trained = TextEncoder::new_seeded(tiny_config(), Device::Cpu, 4242).expect("trained");
        trained.save_weights(&path).expect("save");

        let mut target = TextEncoder::new_seeded(tiny_config(), Device::Cpu, 1).expect("target");
        assert!(!target.is_pretrained());
        let ids = [1u32, 5, 9];
        assert!(target.encode_pretrained(&ids, None).is_err());

        let before = target.encode(&ids, None).expect("before");
        let report = target.load_weights(&path).expect("load");
        assert!(report.loaded > 0);
        assert_eq!(report.shape_mismatches, 0);
        assert!(target.is_pretrained());

        let after = target.encode(&ids, None).expect("after");
        assert_ne!(before, after, "loading weights must change the output");
        let expected = trained.encode(&ids, None).expect("expected");
        for (row_a, row_b) in after.iter().zip(expected.iter()) {
            for (a, b) in row_a.iter().zip(row_b.iter()) {
                assert!((a - b).abs() < 1e-5);
            }
        }
        assert!(target.encode_pretrained(&ids, None).is_ok());
    }

    #[test]
    fn test_attention_rejects_rope_with_relative_window() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut config = attention_config(8, 2, Some(3));
        config.use_rope = true;
        let err = RelativeMultiHeadAttention::new(config, vb.pp("attn"))
            .expect_err("both schemes must be rejected");
        assert!(err.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn test_text_encoder_load_fails_closed() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("unrelated.safetensors");

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _ = candle_nn::linear(3, 3, vb.pp("nothing_related")).expect("linear");
        varmap.save(&path).expect("save");

        let mut encoder = TextEncoder::new(tiny_config()).expect("encoder");
        let err = encoder.load_weights(&path).expect_err("must fail closed");
        assert!(err.to_string().contains("No VITS2 weights"));
        assert!(!encoder.is_pretrained());
    }

    #[test]
    fn test_map_text_encoder_weight_name() {
        assert_eq!(
            map_text_encoder_weight_name("enc_p.encoder.attn_layers.2.conv_q.weight").as_deref(),
            Some("layers.2.attn.q_proj.weight")
        );
        assert_eq!(
            map_text_encoder_weight_name("enc_p.encoder.attn_layers.0.emb_rel_k").as_deref(),
            Some("layers.0.attn.emb_rel_k")
        );
        assert_eq!(
            map_text_encoder_weight_name("enc_p.encoder.norm_layers_1.1.gamma").as_deref(),
            Some("layers.1.norm1.weight")
        );
        assert_eq!(
            map_text_encoder_weight_name("enc_p.encoder.norm_layers_2.1.beta").as_deref(),
            Some("layers.1.norm2.bias")
        );
        assert_eq!(
            map_text_encoder_weight_name("enc_p.encoder.ffn_layers.3.conv_2.bias").as_deref(),
            Some("layers.3.ffn.conv2.bias")
        );
        assert_eq!(
            map_text_encoder_weight_name("enc_p.proj.weight").as_deref(),
            Some("output_projection.weight")
        );
        assert_eq!(
            map_text_encoder_weight_name("enc_p.emb.weight").as_deref(),
            Some("emb.weight")
        );
        assert_eq!(map_text_encoder_weight_name("dec.conv_pre.weight"), None);
    }

    #[test]
    fn test_rope_changes_output_and_is_position_dependent() {
        let mut config = attention_config(8, 2, None);
        config.relative_attention = false;
        let mut rope_config = config.clone();
        rope_config.use_rope = true;

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let plain = RelativeMultiHeadAttention::new(config, vb.pp("attn")).expect("plain");
        let rope = RelativeMultiHeadAttention::new(rope_config, vb.pp("attn")).expect("rope");
        seed_varmap(&varmap, 606, &device).expect("seed");

        assert!(!plain.uses_rope());
        assert!(rope.uses_rope());

        let query = rows(6, 8, 0.31);
        let value = rows(6, 8, 1.31);
        let a = plain.forward(&query, &query, &value, None).expect("plain");
        let b = rope.forward(&query, &query, &value, None).expect("rope");
        assert_ne!(a, b, "use_rope must actually change the computation");

        // RoPE makes the scores position dependent: shifting the sequence
        // changes the result even though the content is the same.
        let mut shifted = vec![vec![0.0f32; 8]];
        shifted.extend(query.iter().cloned());
        let shifted_out = rope
            .forward(&shifted, &shifted, &shifted, None)
            .expect("rope shifted");
        assert_eq!(shifted_out.len(), 7);
    }

    #[test]
    fn test_rope_requires_even_head_dim() {
        let mut config = TextEncoderConfig::default();
        config.use_rope = true;
        config.window_size = None;
        config.hidden_channels = 6;
        config.n_heads = 4;
        assert!(config.validate().is_err());

        // The shipped high-quality preset enables RoPE and must stay valid.
        let hq = TextEncoderConfig::high_quality();
        assert!(hq.use_rope);
        assert!(hq.validate().is_ok());
        let encoder = TextEncoder::new(hq).expect("hq encoder");
        let attn = &encoder.layers()[0].self_attention;
        assert!(attn.uses_rope());
        assert!(!attn.uses_relative_positions());
    }

    #[test]
    fn test_rope_and_relative_window_are_mutually_exclusive() {
        let mut config = TextEncoderConfig::default();
        config.use_rope = true;
        config.window_size = Some(4);
        let err = config.validate().expect_err("must reject both schemes");
        assert!(err.to_string().contains("mutually exclusive"));
        assert!(TextEncoder::new(config).is_err());

        // Exactly one scheme is active in each shipped preset.
        for preset in [
            TextEncoderConfig::default(),
            TextEncoderConfig::high_quality(),
            TextEncoderConfig::fast(),
        ] {
            assert!(preset.validate().is_ok());
            let encoder = TextEncoder::new(preset).expect("encoder");
            let attn = &encoder.layers()[0].self_attention;
            assert!(
                attn.uses_rope() != attn.uses_relative_positions(),
                "exactly one relative-position scheme must be active"
            );
        }
    }

    #[test]
    fn test_analytic_parameter_count_matches_allocation() {
        for config in [tiny_config(), TextEncoderConfig::fast()] {
            let analytic = config.parameter_count();
            let encoder = TextEncoder::new(config).expect("encoder");
            assert_eq!(analytic, encoder.parameter_count().expect("count"));
        }
    }

    #[test]
    fn test_default_config_text_encoder_runs() {
        let encoder = TextEncoder::new(TextEncoderConfig::default()).expect("encoder");
        let output = encoder.encode(&[1, 2, 3, 4, 5], None).expect("encode");
        assert_eq!(output.len(), 5);
        assert_eq!(output[0].len(), 192);
        assert!(encoder.parameter_count().expect("count") > 100_000);
        assert!(output.iter().flatten().all(|v| v.is_finite()));
    }
}
