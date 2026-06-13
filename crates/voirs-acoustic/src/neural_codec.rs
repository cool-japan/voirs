//! Neural Audio Codec Integration
//!
//! This module provides integration with neural audio codecs for high-quality
//! audio representation and compression. Supports modern codecs like EnCodec
//! and SoundStream for efficient audio encoding and decoding.
//!
//! # Features
//! - Multi-scale quantization for varying bitrates
//! - Residual vector quantization (RVQ) for high fidelity
//! - Streaming-friendly encoding with low latency
//! - Perceptual quality optimization
//! - Adaptive bitrate based on content complexity

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};
use scirs2_fft::rfft;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::{AcousticError, Result};

/// Neural codec types supported by VoiRS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodecType {
    /// EnCodec - Meta's neural audio codec
    EnCodec,
    /// SoundStream - Google's neural audio codec
    SoundStream,
    /// Custom neural codec architecture
    Custom,
}

/// Configuration for neural audio codec
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralCodecConfig {
    /// Type of neural codec to use
    pub codec_type: CodecType,
    /// Target bitrate in kbps
    pub target_bitrate: f32,
    /// Number of codebook levels for RVQ
    pub num_codebooks: usize,
    /// Codebook size (vocabulary)
    pub codebook_size: usize,
    /// Frame rate (frames per second)
    pub frame_rate: f32,
    /// Encoder hidden dimension
    pub encoder_dim: usize,
    /// Decoder hidden dimension
    pub decoder_dim: usize,
    /// Number of encoder layers
    pub num_encoder_layers: usize,
    /// Number of decoder layers
    pub num_decoder_layers: usize,
    /// Use perceptual loss weighting
    pub perceptual_weighting: bool,
    /// Compression level (1-10, 10 = maximum compression)
    pub compression_level: u8,
    /// Enable streaming mode for low latency
    pub streaming_mode: bool,
    /// Hop length for STFT analysis
    pub hop_length: usize,
    /// Enable adaptive bitrate based on content
    pub adaptive_bitrate: bool,
}

impl Default for NeuralCodecConfig {
    fn default() -> Self {
        Self {
            codec_type: CodecType::EnCodec,
            target_bitrate: 6.0, // 6 kbps baseline
            num_codebooks: 8,    // Multi-scale RVQ
            codebook_size: 1024,
            frame_rate: 75.0, // 75 Hz frame rate
            encoder_dim: 512,
            decoder_dim: 512,
            num_encoder_layers: 4,
            num_decoder_layers: 4,
            perceptual_weighting: true,
            compression_level: 5,
            streaming_mode: true,
            hop_length: 320, // 20ms at 16kHz
            adaptive_bitrate: true,
        }
    }
}

impl NeuralCodecConfig {
    /// Create configuration optimized for high quality
    pub fn high_quality() -> Self {
        Self {
            target_bitrate: 24.0,
            num_codebooks: 16,
            codebook_size: 2048,
            compression_level: 3,
            ..Default::default()
        }
    }

    /// Create configuration optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            target_bitrate: 3.0,
            num_codebooks: 4,
            frame_rate: 100.0, // Higher frame rate for lower latency
            num_encoder_layers: 3,
            num_decoder_layers: 3,
            streaming_mode: true,
            compression_level: 7,
            ..Default::default()
        }
    }

    /// Create configuration optimized for low bandwidth
    pub fn low_bandwidth() -> Self {
        Self {
            target_bitrate: 1.5,
            num_codebooks: 2,
            codebook_size: 512,
            compression_level: 10,
            adaptive_bitrate: true,
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.target_bitrate <= 0.0 || self.target_bitrate > 320.0 {
            return Err(AcousticError::ConfigError {
                message: format!(
                    "Invalid target bitrate: {} (must be 0 < bitrate <= 320 kbps)",
                    self.target_bitrate
                ),
            });
        }

        if self.num_codebooks == 0 || self.num_codebooks > 32 {
            return Err(AcousticError::ConfigError {
                message: format!(
                    "Invalid num_codebooks: {} (must be 1-32)",
                    self.num_codebooks
                ),
            });
        }

        if self.codebook_size < 256 || self.codebook_size > 8192 {
            return Err(AcousticError::ConfigError {
                message: format!(
                    "Invalid codebook_size: {} (must be 256-8192)",
                    self.codebook_size
                ),
            });
        }

        if !(1..=10).contains(&self.compression_level) {
            return Err(AcousticError::ConfigError {
                message: format!(
                    "Invalid compression_level: {} (must be 1-10)",
                    self.compression_level
                ),
            });
        }

        Ok(())
    }

    /// Calculate theoretical bits per frame
    pub fn bits_per_frame(&self) -> f32 {
        let bits_per_codebook = (self.codebook_size as f32).log2();
        bits_per_codebook * self.num_codebooks as f32
    }

    /// Calculate expected latency in milliseconds
    pub fn expected_latency_ms(&self) -> f32 {
        if self.streaming_mode {
            1000.0 / self.frame_rate // One frame latency
        } else {
            2000.0 / self.frame_rate // Two frames for lookahead
        }
    }
}

/// Residual Vector Quantizer for neural codec
#[derive(Debug)]
pub struct ResidualVectorQuantizer {
    /// Number of quantization levels
    num_levels: usize,
    /// Codebook size per level
    codebook_size: usize,
    /// Embedding dimension
    embedding_dim: usize,
    /// Codebooks for each quantization level
    codebooks: Vec<Tensor>,
    /// Device for computation
    device: Device,
}

impl ResidualVectorQuantizer {
    /// Create a new RVQ with specified configuration
    pub fn new(
        num_levels: usize,
        codebook_size: usize,
        embedding_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let mut codebooks = Vec::with_capacity(num_levels);

        for _ in 0..num_levels {
            // Initialize codebook with random values
            let codebook = Tensor::randn(
                0.0,
                1.0 / (embedding_dim as f64).sqrt(),
                (codebook_size, embedding_dim),
                device,
            )
            .map_err(|e| AcousticError::ModelError {
                message: format!("Failed to initialize codebook: {}", e),
            })?;
            codebooks.push(codebook);
        }

        Ok(Self {
            num_levels,
            codebook_size,
            embedding_dim,
            codebooks,
            device: device.clone(),
        })
    }

    /// Encode input tensor to discrete codes
    pub fn encode(&self, input: &Tensor) -> Result<Vec<Vec<usize>>> {
        let mut residual = input.clone();
        let mut all_indices = Vec::with_capacity(self.num_levels);

        for level in 0..self.num_levels {
            // Find nearest codebook entries
            let indices = self.find_nearest_codes(&residual, level)?;
            all_indices.push(indices.clone());

            // Quantize using the codebook
            let quantized = self.quantize_indices(&indices, level)?;

            // Compute residual for next level
            residual = residual
                .sub(&quantized)
                .map_err(|e| AcousticError::ProcessingError {
                    message: format!("Failed to compute residual: {}", e),
                })?;
        }

        Ok(all_indices)
    }

    /// Decode discrete codes back to continuous representation
    pub fn decode(&self, codes: &[Vec<usize>]) -> Result<Tensor> {
        if codes.len() != self.num_levels {
            return Err(AcousticError::InputError {
                message: format!(
                    "Expected {} codebook levels, got {}",
                    self.num_levels,
                    codes.len()
                ),
            });
        }

        let mut output: Option<Tensor> = None;

        for (level, indices) in codes.iter().enumerate() {
            let quantized = self.quantize_indices(indices, level)?;

            output = Some(if let Some(prev) = output {
                prev.add(&quantized)
                    .map_err(|e| AcousticError::ProcessingError {
                        message: format!("Failed to accumulate quantized values: {}", e),
                    })?
            } else {
                quantized
            });
        }

        output.ok_or_else(|| AcousticError::ProcessingError {
            message: "Failed to decode: no output generated".to_string(),
        })
    }

    /// Find nearest codebook entries for input using L2 distance.
    ///
    /// Uses the L2 identity to avoid materialising a full `(N, C, D)` intermediate tensor.
    ///
    /// - `input`: any shape whose total element count is divisible by `embedding_dim` (e.g., `[B, T, D]` or `[N, D]`)
    /// - `level`: which codebook level to search
    /// - returns: `N = total_elements / embedding_dim` argmin indices
    fn find_nearest_codes(&self, input: &Tensor, level: usize) -> Result<Vec<usize>> {
        let codebook = &self.codebooks[level]; // [C, D]

        // Flatten all leading dimensions → [N, D]
        let n = input.elem_count() / self.embedding_dim;
        let x =
            input
                .reshape((n, self.embedding_dim))
                .map_err(|e| AcousticError::ProcessingError {
                    message: format!("Failed to reshape input for nearest-code search: {}", e),
                })?; // [N, D]

        // ||x||^2  → [N, 1]
        let x_sq = x
            .sqr()
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to compute x^2: {}", e),
            })?
            .sum_keepdim(1)
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to sum x^2: {}", e),
            })?; // [N, 1]

        // ||c||^2  → [1, C]
        let c_sq = codebook
            .sqr()
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to compute codebook^2: {}", e),
            })?
            .sum_keepdim(1)
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to sum codebook^2: {}", e),
            })?
            .t()
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to transpose codebook norms: {}", e),
            })?; // [1, C]

        // x * c^T  → [N, C]
        let x_ct = x
            .matmul(&codebook.t().map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to transpose codebook: {}", e),
            })?)
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to compute x * codebook^T: {}", e),
            })?; // [N, C]

        // dist² = ||x||² - 2·x·c^T + ||c||²   → [N, C]
        // Note: subtracting 2·x·c^T  is equivalent to affine(-2.0, 0.0) then add norms.
        let dist = x_sq
            .broadcast_add(&c_sq)
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to broadcast-add norms: {}", e),
            })?
            .sub(
                &x_ct
                    .affine(2.0, 0.0)
                    .map_err(|e| AcousticError::ProcessingError {
                        message: format!("Failed to scale cross term: {}", e),
                    })?,
            )
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to subtract cross term from dist: {}", e),
            })?; // [N, C]

        // Argmin along codebook axis → [N]
        let indices_tensor = dist.argmin(1).map_err(|e| AcousticError::ProcessingError {
            message: format!("Failed to compute argmin: {}", e),
        })?;

        let raw: Vec<u32> =
            indices_tensor
                .to_vec1()
                .map_err(|e| AcousticError::ProcessingError {
                    message: format!("Failed to extract argmin indices: {}", e),
                })?;

        Ok(raw.into_iter().map(|i| i as usize).collect())
    }

    /// Gather codebook rows corresponding to `indices`.
    ///
    /// Returns a tensor of shape `[num_indices, embedding_dim]`.
    fn quantize_indices(&self, indices: &[usize], level: usize) -> Result<Tensor> {
        let codebook = &self.codebooks[level]; // [C, D]

        // Build a U32 index tensor from the flat slice.
        let idx_tensor = Tensor::from_iter(indices.iter().map(|&i| i as u32), &self.device)
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to create index tensor: {}", e),
            })?; // [N]

        // index_select gathers rows: codebook[[i0, i1, ...]] → [N, D]
        codebook
            .index_select(&idx_tensor, 0)
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to gather codebook entries: {}", e),
            })
    }

    /// Compute commitment loss for training
    pub fn commitment_loss(&self, input: &Tensor, quantized: &Tensor, beta: f32) -> Result<f32> {
        // Commitment loss: beta * ||input - sg(quantized)||^2
        // where sg is stop gradient

        let diff = input
            .sub(quantized)
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to compute difference: {}", e),
            })?;

        let squared = diff.sqr().map_err(|e| AcousticError::ProcessingError {
            message: format!("Failed to compute squared difference: {}", e),
        })?;

        let loss = squared
            .mean_all()
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to compute mean: {}", e),
            })?
            .to_scalar::<f32>()
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to convert to scalar: {}", e),
            })?;

        Ok(beta * loss)
    }
}

// ---------------------------------------------------------------------------
// Internal DSP helpers
// ---------------------------------------------------------------------------

/// Build a DCT-II derived linear projection of shape `[out_dim, in_dim]`.
///
/// Row `k` is the k-th DCT-II basis vector evaluated at `in_dim` sample points,
/// normalised to unit L2 norm.  The first `out_dim` rows form an orthonormal set
/// (for `out_dim <= in_dim`), making this an excellent deterministic encoder
/// initialisation that captures the dominant frequency content of each frame.
fn build_dct_projection(out_dim: usize, in_dim: usize, device: &Device) -> Result<Linear> {
    use std::f32::consts::PI;

    let mut weights = vec![0.0_f32; out_dim * in_dim];

    for k in 0..out_dim {
        let scale = if k == 0 {
            (1.0_f32 / in_dim as f32).sqrt()
        } else {
            (2.0_f32 / in_dim as f32).sqrt()
        };
        for n in 0..in_dim {
            let val =
                scale * (PI * k as f32 * (2.0 * n as f32 + 1.0) / (2.0 * in_dim as f32)).cos();
            weights[k * in_dim + n] = val;
        }
    }

    let weight_tensor = Tensor::from_vec(weights, (out_dim, in_dim), device).map_err(|e| {
        AcousticError::ProcessingError {
            message: format!("Failed to build DCT projection tensor: {}", e),
        }
    })?;

    Ok(Linear::new(weight_tensor, None))
}

/// Build the transposed DCT back-projection of shape `[in_dim, out_dim]`.
///
/// This is the pseudo-inverse of `build_dct_projection(out_dim, in_dim)`:
/// i.e., the DCT-III synthesis matrix.  When the latent code lives exactly
/// in the DCT-II subspace (which it does after encode + RVQ decode), applying
/// this projection reconstructs a scaled version of the original frame.
fn build_dct_back_projection(enc_dim: usize, hop_length: usize, device: &Device) -> Result<Linear> {
    use std::f32::consts::PI;

    // Decoder weight shape: [hop_length, enc_dim]  (maps enc_dim → hop_length)
    let mut weights = vec![0.0_f32; hop_length * enc_dim];

    for k in 0..enc_dim {
        let scale = if k == 0 {
            (1.0_f32 / hop_length as f32).sqrt()
        } else {
            (2.0_f32 / hop_length as f32).sqrt()
        };
        for n in 0..hop_length {
            // Transposed: weight[n, k] = dct_basis[k, n]
            let val =
                scale * (PI * k as f32 * (2.0 * n as f32 + 1.0) / (2.0 * hop_length as f32)).cos();
            weights[n * enc_dim + k] = val;
        }
    }

    let weight_tensor = Tensor::from_vec(weights, (hop_length, enc_dim), device).map_err(|e| {
        AcousticError::ProcessingError {
            message: format!("Failed to build DCT back-projection tensor: {}", e),
        }
    })?;

    Ok(Linear::new(weight_tensor, None))
}

/// Per-frame z-score normalisation.
///
/// For a `[N, D]` tensor: each row is shifted by its mean and divided by its
/// standard deviation (with a small `eps` floor to avoid division by zero).
/// This is the standard pre-processing applied before a linear audio codec
/// projection layer.
fn per_frame_normalise(frames: &Tensor) -> Result<Tensor> {
    // Mean per frame → [N, 1]
    let mean = frames
        .mean_keepdim(1)
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("per_frame_normalise mean failed: {}", e),
        })?;

    // Subtract mean → [N, D]
    let centred = frames
        .broadcast_sub(&mean)
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("per_frame_normalise sub failed: {}", e),
        })?;

    // Variance per frame → [N, 1]
    let var = centred
        .sqr()
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("per_frame_normalise sqr failed: {}", e),
        })?
        .mean_keepdim(1)
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("per_frame_normalise var failed: {}", e),
        })?;

    // Std = sqrt(var + eps)
    let std = (var + 1e-8_f64)
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("per_frame_normalise var+eps failed: {}", e),
        })?
        .sqrt()
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("per_frame_normalise sqrt failed: {}", e),
        })?;

    centred
        .broadcast_div(&std)
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("per_frame_normalise div failed: {}", e),
        })
}

// ---------------------------------------------------------------------------
// End of DSP helpers
// ---------------------------------------------------------------------------

/// Neural audio codec encoder
pub struct NeuralEncoder {
    /// Configuration
    config: NeuralCodecConfig,
    /// Projection layer: maps frame (hop_length) → encoder_dim via DCT-derived weights
    proj: Linear,
    /// RVQ for quantization
    rvq: ResidualVectorQuantizer,
    /// Device
    device: Device,
}

impl std::fmt::Debug for NeuralEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralEncoder")
            .field("config", &self.config)
            .field("proj", &"<Linear projection>")
            .field("device", &self.device)
            .finish()
    }
}

impl NeuralEncoder {
    /// Create new neural encoder.
    ///
    /// The encoder projection is initialised from a truncated DCT-II matrix so
    /// that the first `encoder_dim` basis vectors form an orthonormal frame for
    /// the `hop_length`-dimensional input.  This gives a meaningful, invertible
    /// starting point that works without pretrained weights.
    pub fn new(config: NeuralCodecConfig, device: &Device) -> Result<Self> {
        config.validate()?;

        let rvq = ResidualVectorQuantizer::new(
            config.num_codebooks,
            config.codebook_size,
            config.encoder_dim,
            device,
        )?;

        let proj = build_dct_projection(config.encoder_dim, config.hop_length, device)?;

        Ok(Self {
            config,
            proj,
            rvq,
            device: device.clone(),
        })
    }

    /// Encode audio waveform to discrete codes.
    pub fn encode(&self, waveform: &Tensor) -> Result<Vec<Vec<usize>>> {
        // 1. Encode waveform to continuous representation
        let encoded = self.encode_continuous(waveform)?;

        // 2. Quantize to discrete codes using RVQ
        self.rvq.encode(&encoded)
    }

    /// Encode waveform to continuous latent representation.
    ///
    /// Processing pipeline:
    ///   1. Segment the waveform into non-overlapping frames of `hop_length`.
    ///   2. Normalise each frame to zero mean and unit variance.
    ///   3. Apply the DCT-derived linear projection (`encoder_dim × hop_length`)
    ///      to obtain a compact latent vector per frame.
    ///
    /// Input shape:  `[batch, waveform_len]`
    /// Output shape: `[batch, seq_len, encoder_dim]`
    ///   where `seq_len = waveform_len / hop_length`.
    pub(crate) fn encode_continuous(&self, waveform: &Tensor) -> Result<Tensor> {
        let dims = waveform.dims();
        if dims.len() < 2 {
            return Err(AcousticError::InputError {
                message: format!(
                    "encode_continuous expects [batch, waveform_len] tensor, got {} dims",
                    dims.len()
                ),
            });
        }
        let batch_size = dims[0];
        let waveform_len = dims[1];
        let hop = self.config.hop_length;
        let seq_len = waveform_len / hop;

        if seq_len == 0 {
            return Err(AcousticError::InputError {
                message: format!(
                    "Waveform length {} is shorter than hop_length {}",
                    waveform_len, hop
                ),
            });
        }

        // Slice to exact multiple of hop_length → [batch, seq_len * hop]
        let waveform_trimmed =
            waveform
                .narrow(1, 0, seq_len * hop)
                .map_err(|e| AcousticError::ProcessingError {
                    message: format!("Failed to trim waveform: {}", e),
                })?;

        // Reshape to frames → [batch * seq_len, hop]
        let frames = waveform_trimmed
            .reshape((batch_size * seq_len, hop))
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to reshape waveform to frames: {}", e),
            })?;

        // Per-frame z-score normalization: (x - mean) / (std + eps)
        let frames_normalised = per_frame_normalise(&frames)?;

        // Linear projection → [batch * seq_len, encoder_dim]
        let projected =
            self.proj
                .forward(&frames_normalised)
                .map_err(|e| AcousticError::ProcessingError {
                    message: format!("Encoder projection failed: {}", e),
                })?;

        // Reshape to [batch, seq_len, encoder_dim]
        projected
            .reshape((batch_size, seq_len, self.config.encoder_dim))
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to reshape encoder output: {}", e),
            })
    }

    /// Get encoder configuration.
    pub fn config(&self) -> &NeuralCodecConfig {
        &self.config
    }
}

/// Neural audio codec decoder
pub struct NeuralDecoder {
    /// Configuration
    config: NeuralCodecConfig,
    /// Back-projection layer: maps encoder_dim → hop_length via transposed DCT weights
    back_proj: Linear,
    /// RVQ for dequantization
    rvq: Arc<ResidualVectorQuantizer>,
    /// Device
    device: Device,
}

impl std::fmt::Debug for NeuralDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralDecoder")
            .field("config", &self.config)
            .field("back_proj", &"<Linear back-projection>")
            .field("device", &self.device)
            .finish()
    }
}

impl NeuralDecoder {
    /// Create new neural decoder.
    ///
    /// The back-projection is the transpose of the encoder's DCT matrix, giving
    /// an exact pseudo-inverse when the input lives in the DCT subspace.
    pub fn new(
        config: NeuralCodecConfig,
        rvq: Arc<ResidualVectorQuantizer>,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;

        // Decoder uses the transposed DCT projection: (hop_length × encoder_dim)^T → (encoder_dim, hop_length)
        let back_proj = build_dct_back_projection(config.encoder_dim, config.hop_length, device)?;

        Ok(Self {
            config,
            back_proj,
            rvq,
            device: device.clone(),
        })
    }

    /// Decode discrete codes back to audio waveform.
    pub fn decode(&self, codes: &[Vec<usize>]) -> Result<Tensor> {
        // 1. Dequantize codes to continuous representation
        let continuous = self.rvq.decode(codes)?;

        // 2. Decode continuous representation to waveform
        self.decode_continuous(&continuous)
    }

    /// Decode continuous latent representation back to waveform.
    ///
    /// Processing pipeline:
    ///   1. Reshape to `[batch * seq_len, encoder_dim]`.
    ///   2. Apply the transposed-DCT back-projection to get `[batch * seq_len, hop_length]`.
    ///   3. Reshape to `[batch, seq_len * hop_length]`.
    ///
    /// Input shape:  `[batch, seq_len, encoder_dim]`  (or `[seq_len, encoder_dim]` for single-batch RVQ output)
    /// Output shape: `[batch, waveform_len]`
    pub(crate) fn decode_continuous(&self, encoded: &Tensor) -> Result<Tensor> {
        let dims = encoded.dims();

        // Accept both [seq_len, encoder_dim] and [batch, seq_len, encoder_dim]
        let (batch_size, seq_len) = match dims.len() {
            2 => (1usize, dims[0]),
            3 => (dims[0], dims[1]),
            _ => {
                return Err(AcousticError::InputError {
                    message: format!(
                        "decode_continuous expects 2-D or 3-D tensor, got {} dims",
                        dims.len()
                    ),
                })
            }
        };

        let enc_dim = *dims
            .last()
            .expect("dims verified to be 2-D or 3-D so last() is always Some");
        if enc_dim != self.config.encoder_dim {
            return Err(AcousticError::InputError {
                message: format!(
                    "Latent dim {} does not match encoder_dim {}",
                    enc_dim, self.config.encoder_dim
                ),
            });
        }

        // Flatten to [batch * seq_len, encoder_dim]
        let flat = encoded
            .reshape((batch_size * seq_len, self.config.encoder_dim))
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to flatten encoded tensor: {}", e),
            })?;

        // Back-project → [batch * seq_len, hop_length]
        let frames = self
            .back_proj
            .forward(&flat)
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Decoder back-projection failed: {}", e),
            })?;

        // Reshape to [batch, waveform_len]
        let waveform_len = seq_len * self.config.hop_length;
        frames
            .reshape((batch_size, waveform_len))
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to reshape decoded waveform: {}", e),
            })
    }

    /// Get decoder configuration.
    pub fn config(&self) -> &NeuralCodecConfig {
        &self.config
    }
}

/// Complete neural audio codec (encoder + decoder)
#[derive(Debug)]
pub struct NeuralCodec {
    /// Encoder component
    encoder: NeuralEncoder,
    /// Decoder component
    decoder: NeuralDecoder,
    /// Shared RVQ
    rvq: Arc<ResidualVectorQuantizer>,
    /// Configuration
    config: NeuralCodecConfig,
}

impl NeuralCodec {
    /// Create new neural codec
    pub fn new(config: NeuralCodecConfig, device: &Device) -> Result<Self> {
        config.validate()?;

        let rvq = Arc::new(ResidualVectorQuantizer::new(
            config.num_codebooks,
            config.codebook_size,
            config.encoder_dim,
            device,
        )?);

        let encoder = NeuralEncoder::new(config.clone(), device)?;
        let decoder = NeuralDecoder::new(config.clone(), rvq.clone(), device)?;

        Ok(Self {
            encoder,
            decoder,
            rvq,
            config,
        })
    }

    /// Encode audio waveform to discrete codes
    pub fn encode(&self, waveform: &Tensor) -> Result<Vec<Vec<usize>>> {
        self.encoder.encode(waveform)
    }

    /// Decode discrete codes to audio waveform
    pub fn decode(&self, codes: &[Vec<usize>]) -> Result<Tensor> {
        self.decoder.decode(codes)
    }

    /// Get codec configuration
    pub fn config(&self) -> &NeuralCodecConfig {
        &self.config
    }

    /// Calculate compression ratio
    pub fn compression_ratio(&self, input_samples: usize) -> f32 {
        let input_bits = input_samples * 16; // 16-bit audio
        let encoded_frames = input_samples / self.config.hop_length;
        let encoded_bits = encoded_frames as f32 * self.config.bits_per_frame();
        input_bits as f32 / encoded_bits
    }

    /// Estimate bitrate for given sample rate
    pub fn estimate_bitrate(&self, sample_rate: usize) -> f32 {
        let bits_per_second = self.config.frame_rate * self.config.bits_per_frame();
        bits_per_second / 1000.0 // Convert to kbps
    }
}

/// Perceptual quality metrics for codec evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecQualityMetrics {
    /// Signal-to-noise ratio (dB)
    pub snr_db: f32,
    /// Perceptual Evaluation of Speech Quality
    pub pesq_score: f32,
    /// Short-Time Objective Intelligibility
    pub stoi_score: f32,
    /// Mel-cepstral distortion
    pub mcd: f32,
    /// Bitrate (kbps)
    pub bitrate_kbps: f32,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Encoding/decoding latency (ms)
    pub latency_ms: f32,
}

impl CodecQualityMetrics {
    /// Create placeholder metrics (to be filled by actual evaluation).
    ///
    /// Use [`CodecQualityMetrics::compute_from_audio`] when the actual audio
    /// samples are available.
    pub fn placeholder() -> Self {
        Self {
            snr_db: 0.0,
            pesq_score: 0.0,
            stoi_score: 0.0,
            mcd: 0.0,
            bitrate_kbps: 0.0,
            compression_ratio: 0.0,
            latency_ms: 0.0,
        }
    }

    /// Compute quality metrics directly from audio samples.
    ///
    /// # Arguments
    /// * `samples`           – PCM audio samples in `[-1.0, 1.0]`.
    /// * `sample_rate`       – Sample rate in Hz (e.g. 16 000).
    /// * `bitrate_kbps`      – Codec bitrate in kbps.
    /// * `compression_ratio` – Codec compression ratio (raw PCM bits / encoded bits).
    /// * `latency_ms`        – Measured encoding + decoding latency in milliseconds.
    ///
    /// # Signal-derived metrics
    /// * **snr_db** – 20·log10(RMS_signal / noise_floor) where the noise floor
    ///   is estimated as the 5th-percentile of per-frame RMS values.
    /// * **bandwidth_hz** (stored in `stoi_score` as a normalised 0–1 fraction of
    ///   Nyquist) – the highest FFT bin whose power exceeds −60 dBFS vs. the peak.
    /// * **mcd** – Spectral flatness (Wiener entropy): geometric mean / arithmetic
    ///   mean of the power spectrum.  Low = tonal; high (→ 1) = noise-like.
    ///   Stored as `mcd` for structural compatibility.
    /// * **pesq_score** – Estimated from SNR via an ITU-T P.862 approximation
    ///   curve: 0 dB → 1.0, 40 dB → 4.5.
    pub fn compute_from_audio(
        samples: &[f32],
        sample_rate: u32,
        bitrate_kbps: f32,
        compression_ratio: f32,
        latency_ms: f32,
    ) -> Self {
        if samples.is_empty() {
            return Self {
                bitrate_kbps,
                compression_ratio,
                latency_ms,
                ..Self::placeholder()
            };
        }

        // --- SNR via frame-RMS analysis ---
        let frame_len: usize = (sample_rate as usize / 100).max(64); // 10 ms or 64 samples
        let n_frames = samples.len() / frame_len;

        let mut frame_rms: Vec<f32> = (0..n_frames)
            .map(|i| {
                let start = i * frame_len;
                let frame = &samples[start..start + frame_len];
                let mean_sq: f32 = frame.iter().map(|&s| s * s).sum::<f32>() / frame.len() as f32;
                mean_sq.sqrt()
            })
            .collect();

        // Overall RMS (signal level)
        let signal_rms = {
            let mean_sq: f32 = samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32;
            mean_sq.sqrt().max(1e-12)
        };

        // Noise floor = 5th-percentile frame RMS
        frame_rms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p5_idx = (frame_rms.len() as f32 * 0.05).floor() as usize;
        let noise_floor = frame_rms
            .get(p5_idx)
            .copied()
            .unwrap_or(1e-12_f32)
            .max(1e-12);

        let snr_db = 20.0 * (signal_rms / noise_floor).log10();

        // --- Bandwidth via power-spectral roll-off ---
        // One-sided power spectrum over the longest power-of-two window that
        // fits in `samples`, computed with the SciRS2 real-FFT (`rfft`,
        // O(N log N)) rather than a direct O(N²) DFT. `rfft` is the
        // unnormalised forward transform, so each bin is the exact `|X[k]|²`
        // the previous direct DFT produced — identical length, scaling and
        // (rectangular) windowing, leaving every downstream metric unchanged.
        let fft_len: usize = {
            let mut n = 1usize;
            while n * 2 <= samples.len().min(8192) {
                n *= 2;
            }
            n
        };
        let window = &samples[..fft_len];

        // Rectangular-windowed |X[k]|² for k = 0..=fft_len/2 (length fft_len/2 + 1).
        let power = Self::one_sided_power_spectrum(window);
        let n_bins = power.len();

        let peak_power = power.iter().cloned().fold(0.0_f32, f32::max).max(1e-30);
        let threshold = peak_power * 10.0_f32.powf(-60.0 / 10.0); // −60 dBFS

        let bandwidth_bin = power
            .iter()
            .enumerate()
            .rev()
            .find(|(_, &p)| p > threshold)
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Normalise to [0, 1] fraction of Nyquist
        let bandwidth_fraction = bandwidth_bin as f32 / (n_bins - 1).max(1) as f32;

        // --- Spectral flatness (Wiener entropy) → stored as `mcd` ---
        let eps = 1e-30_f32;
        let log_sum: f32 = power.iter().map(|&p| (p + eps).ln()).sum::<f32>();
        let geom_mean = (log_sum / n_bins as f32).exp();
        let arith_mean: f32 = power.iter().sum::<f32>() / n_bins as f32;
        let spectral_flatness = (geom_mean / arith_mean.max(eps)).clamp(0.0, 1.0);

        // --- PESQ approximation from SNR (P.862 curve fit) ---
        // Piecewise linear approximation: SNR ∈ [0, 40] → PESQ ∈ [1.0, 4.5]
        let pesq_score = (1.0_f32 + (snr_db.clamp(0.0, 40.0) / 40.0) * 3.5).clamp(1.0, 4.5);

        Self {
            snr_db,
            pesq_score,
            stoi_score: bandwidth_fraction,
            mcd: spectral_flatness,
            bitrate_kbps,
            compression_ratio,
            latency_ms,
        }
    }

    /// One-sided power spectrum `|X[k]|²` for `k = 0..=window.len()/2`.
    ///
    /// Computed with [`scirs2_fft::rfft`] — the unnormalised forward real-FFT
    /// (`X[k] = Σₙ x[n]·e^{-2πi k n / N}`) — applied to the rectangularly
    /// windowed (i.e. simply truncated) signal. The result therefore matches
    /// a direct O(N²) DFT exactly in length (`N/2 + 1`), scaling (none) and
    /// windowing, but runs in O(N log N). On the (practically unreachable)
    /// FFT error path it returns an all-zero spectrum of the correct length so
    /// callers stay infallible.
    fn one_sided_power_spectrum(window: &[f32]) -> Vec<f32> {
        let fft_len = window.len();
        let n_bins = fft_len / 2 + 1;

        // Promote to f64 for the transform; `rfft` is unnormalised, so the
        // magnitude-squared of each bin equals the previous direct-DFT value
        // up to floating-point rounding.
        let buf_f64: Vec<f64> = window.iter().map(|&s| f64::from(s)).collect();

        match rfft(&buf_f64, Some(fft_len)) {
            Ok(spectrum) => spectrum.iter().map(|c| c.norm_sqr() as f32).collect(),
            Err(_) => vec![0.0_f32; n_bins],
        }
    }

    /// Check if quality meets minimum thresholds
    pub fn meets_quality_threshold(&self) -> bool {
        self.pesq_score >= 3.5 && self.stoi_score >= 0.85
    }

    /// Generate quality report
    pub fn report(&self) -> String {
        format!(
            "Codec Quality Metrics:\n\
             - SNR: {:.2} dB\n\
             - PESQ: {:.3}\n\
             - STOI: {:.3}\n\
             - MCD: {:.3}\n\
             - Bitrate: {:.2} kbps\n\
             - Compression: {:.1}x\n\
             - Latency: {:.2} ms",
            self.snr_db,
            self.pesq_score,
            self.stoi_score,
            self.mcd,
            self.bitrate_kbps,
            self.compression_ratio,
            self.latency_ms
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_config_validation() {
        let config = NeuralCodecConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = NeuralCodecConfig {
            target_bitrate: 0.0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_codec_config_presets() {
        let hq = NeuralCodecConfig::high_quality();
        assert_eq!(hq.target_bitrate, 24.0);
        assert_eq!(hq.num_codebooks, 16);

        let ll = NeuralCodecConfig::low_latency();
        assert!(ll.streaming_mode);
        assert_eq!(ll.frame_rate, 100.0);

        let lb = NeuralCodecConfig::low_bandwidth();
        assert_eq!(lb.target_bitrate, 1.5);
        assert_eq!(lb.compression_level, 10);
    }

    #[test]
    fn test_bits_per_frame_calculation() {
        let config = NeuralCodecConfig {
            num_codebooks: 8,
            codebook_size: 1024,
            ..Default::default()
        };

        let bits = config.bits_per_frame();
        assert_eq!(bits, 80.0); // 8 codebooks * 10 bits/codebook (log2(1024) = 10)
    }

    #[test]
    fn test_latency_estimation() {
        let streaming_config = NeuralCodecConfig {
            streaming_mode: true,
            frame_rate: 75.0,
            ..Default::default()
        };
        assert!((streaming_config.expected_latency_ms() - 13.33).abs() < 0.1);

        let non_streaming = NeuralCodecConfig {
            streaming_mode: false,
            frame_rate: 75.0,
            ..Default::default()
        };
        assert!((non_streaming.expected_latency_ms() - 26.67).abs() < 0.1);
    }

    #[test]
    fn test_rvq_creation() {
        let device = Device::Cpu;
        let rvq = ResidualVectorQuantizer::new(4, 512, 256, &device);
        assert!(rvq.is_ok());

        let rvq = rvq.unwrap();
        assert_eq!(rvq.num_levels, 4);
        assert_eq!(rvq.codebook_size, 512);
        assert_eq!(rvq.embedding_dim, 256);
    }

    #[test]
    fn test_codec_compression_ratio() {
        let device = Device::Cpu;
        let config = NeuralCodecConfig::default();
        let codec = NeuralCodec::new(config, &device).unwrap();

        let input_samples = 16000; // 1 second at 16kHz
        let ratio = codec.compression_ratio(input_samples);
        assert!(ratio > 1.0); // Should compress
    }

    #[test]
    fn test_codec_bitrate_estimation() {
        let device = Device::Cpu;
        let config = NeuralCodecConfig {
            frame_rate: 75.0,
            num_codebooks: 8,
            codebook_size: 1024,
            ..Default::default()
        };
        let codec = NeuralCodec::new(config, &device).unwrap();

        let bitrate = codec.estimate_bitrate(16000);
        // 75 frames/sec * 80 bits/frame = 6000 bits/sec = 6 kbps
        assert!((bitrate - 6.0).abs() < 0.1);
    }

    #[test]
    fn test_quality_metrics_threshold() {
        let good_metrics = CodecQualityMetrics {
            pesq_score: 4.0,
            stoi_score: 0.90,
            ..CodecQualityMetrics::placeholder()
        };
        assert!(good_metrics.meets_quality_threshold());

        let poor_metrics = CodecQualityMetrics {
            pesq_score: 2.0,
            stoi_score: 0.70,
            ..CodecQualityMetrics::placeholder()
        };
        assert!(!poor_metrics.meets_quality_threshold());
    }

    #[test]
    fn test_quality_metrics_report() {
        let metrics = CodecQualityMetrics {
            snr_db: 25.5,
            pesq_score: 4.2,
            stoi_score: 0.92,
            mcd: 4.8,
            bitrate_kbps: 6.0,
            compression_ratio: 21.3,
            latency_ms: 13.3,
        };

        let report = metrics.report();
        assert!(report.contains("SNR: 25.50 dB"));
        assert!(report.contains("PESQ: 4.200"));
        assert!(report.contains("Bitrate: 6.00 kbps"));
    }

    // ------------------------------------------------------------------
    // Real nearest-code / quantize-indices tests
    // ------------------------------------------------------------------

    /// Helper: build an RVQ where the level-0 codebook is replaced by a
    /// known tensor so we can write deterministic assertions.
    fn make_rvq_with_known_codebook(
        codebook_size: usize,
        embedding_dim: usize,
        rows: Vec<f32>,
        device: &Device,
    ) -> crate::Result<ResidualVectorQuantizer> {
        let mut rvq = ResidualVectorQuantizer::new(1, codebook_size, embedding_dim, device)?;
        let cb = Tensor::from_vec(rows, (codebook_size, embedding_dim), device).map_err(|e| {
            AcousticError::ProcessingError {
                message: format!("Failed to build test codebook: {}", e),
            }
        })?;
        rvq.codebooks[0] = cb;
        Ok(rvq)
    }

    #[test]
    fn test_rvq_nearest_code_exact_match() {
        // codebook_size = 4, embedding_dim = 8
        // Build four orthogonal-ish rows (scaled basis vectors).
        let embedding_dim = 8usize;
        let codebook_size = 4usize;
        let mut rows = vec![0.0f32; codebook_size * embedding_dim];
        for i in 0..codebook_size {
            rows[i * embedding_dim + i * 2] = 1.0; // distinct non-overlapping non-zero entries
        }

        let device = Device::Cpu;
        let rvq = make_rvq_with_known_codebook(codebook_size, embedding_dim, rows.clone(), &device)
            .expect("Failed to build RVQ for test");

        // Input = exact copy of codebook row 2, shaped [1, 1, embedding_dim]
        let row2: Vec<f32> = rows[2 * embedding_dim..(2 + 1) * embedding_dim].to_vec();
        let input = Tensor::from_vec(row2, (1usize, 1usize, embedding_dim), &device)
            .expect("Failed to build input tensor");

        let indices = rvq
            .find_nearest_codes(&input, 0)
            .expect("find_nearest_codes failed");

        assert_eq!(
            indices.len(),
            1,
            "Should return one index for a [1,1,D] input"
        );
        assert_eq!(
            indices[0], 2,
            "Exact match for row 2 must map to index 2, got {}",
            indices[0]
        );
    }

    #[test]
    fn test_rvq_quantize_roundtrip() {
        let embedding_dim = 8usize;
        let codebook_size = 4usize;
        let mut rows = vec![0.0f32; codebook_size * embedding_dim];
        for i in 0..codebook_size {
            rows[i * embedding_dim + i * 2] = 1.0;
        }

        let device = Device::Cpu;
        let rvq = make_rvq_with_known_codebook(codebook_size, embedding_dim, rows.clone(), &device)
            .expect("Failed to build RVQ for test");

        // Gather index 2 and verify we get back row 2.
        let gathered = rvq
            .quantize_indices(&[2], 0)
            .expect("quantize_indices failed");

        assert_eq!(
            gathered.dims(),
            &[1, embedding_dim],
            "Gathered tensor should be [1, embedding_dim]"
        );

        let gathered_data: Vec<f32> = gathered
            .to_vec2::<f32>()
            .expect("to_vec2 failed")
            .into_iter()
            .flatten()
            .collect();

        let expected: Vec<f32> = rows[2 * embedding_dim..(2 + 1) * embedding_dim].to_vec();
        for (got, exp) in gathered_data.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-6,
                "Mismatch: got {got} expected {exp}"
            );
        }
    }

    #[test]
    fn test_rvq_commitment_loss_zero_for_codebook_member() {
        // When input == quantized the commitment loss should be exactly 0.
        let embedding_dim = 8usize;
        let codebook_size = 4usize;
        let mut rows = vec![0.0f32; codebook_size * embedding_dim];
        for i in 0..codebook_size {
            rows[i * embedding_dim + i * 2] = 1.0;
        }

        let device = Device::Cpu;
        let rvq = make_rvq_with_known_codebook(codebook_size, embedding_dim, rows.clone(), &device)
            .expect("Failed to build RVQ for test");

        // Build input = codebook row 2, shaped [1, embedding_dim]
        let row2: Vec<f32> = rows[2 * embedding_dim..(2 + 1) * embedding_dim].to_vec();
        let input = Tensor::from_vec(row2.clone(), (1usize, embedding_dim), &device)
            .expect("Failed to build input");
        let quantized = rvq
            .quantize_indices(&[2], 0)
            .expect("quantize_indices failed");

        let loss = rvq
            .commitment_loss(&input, &quantized, 1.0)
            .expect("commitment_loss failed");

        assert!(
            loss.abs() < 1e-6,
            "Commitment loss should be ~0 when input == quantized entry, got {loss}"
        );
    }

    // ------------------------------------------------------------------
    // encode_continuous / decode_continuous shape tests
    // ------------------------------------------------------------------

    /// `encode_continuous` must produce [batch, seq_len, encoder_dim] where
    /// seq_len = floor(waveform_len / hop_length).
    #[test]
    fn test_encode_continuous_output_shape() {
        let device = Device::Cpu;
        let config = NeuralCodecConfig {
            hop_length: 16,
            encoder_dim: 32,
            // keep codebook small so the test is fast
            num_codebooks: 2,
            codebook_size: 256,
            ..Default::default()
        };
        let encoder =
            NeuralEncoder::new(config.clone(), &device).expect("Failed to create NeuralEncoder");

        let batch = 2usize;
        let waveform_samples = 256usize; // 16 frames of 16 samples each
        let waveform = Tensor::randn(0.0_f32, 0.1_f32, (batch, waveform_samples), &device)
            .expect("Failed to create waveform tensor");

        let encoded = encoder
            .encode_continuous(&waveform)
            .expect("encode_continuous failed");

        let expected_seq_len = waveform_samples / config.hop_length;
        assert_eq!(
            encoded.dims(),
            &[batch, expected_seq_len, config.encoder_dim],
            "encode_continuous output shape mismatch: got {:?}",
            encoded.dims()
        );
    }

    /// `decode_continuous` on the output of `encode_continuous` must produce
    /// `[batch, waveform_len]` with `waveform_len = seq_len * hop_length`.
    #[test]
    fn test_decode_continuous_output_shape() {
        let device = Device::Cpu;
        let config = NeuralCodecConfig {
            hop_length: 16,
            encoder_dim: 32,
            num_codebooks: 2,
            codebook_size: 256,
            ..Default::default()
        };

        let rvq = Arc::new(
            ResidualVectorQuantizer::new(
                config.num_codebooks,
                config.codebook_size,
                config.encoder_dim,
                &device,
            )
            .expect("Failed to create RVQ"),
        );

        let encoder =
            NeuralEncoder::new(config.clone(), &device).expect("Failed to create NeuralEncoder");
        let decoder = NeuralDecoder::new(config.clone(), rvq.clone(), &device)
            .expect("Failed to create NeuralDecoder");

        let batch = 2usize;
        let waveform_samples = 256usize;
        let waveform = Tensor::randn(0.0_f32, 0.1_f32, (batch, waveform_samples), &device)
            .expect("Failed to create waveform tensor");

        let encoded = encoder
            .encode_continuous(&waveform)
            .expect("encode_continuous failed");

        let decoded = decoder
            .decode_continuous(&encoded)
            .expect("decode_continuous failed");

        let expected_waveform_len = (waveform_samples / config.hop_length) * config.hop_length;
        assert_eq!(
            decoded.dims(),
            &[batch, expected_waveform_len],
            "decode_continuous output shape mismatch: got {:?}",
            decoded.dims()
        );
    }

    /// `decode_continuous` output must not be all-zeros (non-trivial transformation).
    #[test]
    fn test_decode_continuous_non_zero_output() {
        let device = Device::Cpu;
        let config = NeuralCodecConfig {
            hop_length: 16,
            encoder_dim: 32,
            num_codebooks: 2,
            codebook_size: 256,
            ..Default::default()
        };

        let rvq = Arc::new(
            ResidualVectorQuantizer::new(
                config.num_codebooks,
                config.codebook_size,
                config.encoder_dim,
                &device,
            )
            .expect("Failed to create RVQ"),
        );

        let encoder =
            NeuralEncoder::new(config.clone(), &device).expect("Failed to create NeuralEncoder");
        let decoder = NeuralDecoder::new(config.clone(), rvq, &device)
            .expect("Failed to create NeuralDecoder");

        // Use a non-trivial input signal (a sine-like sawtooth) so the latent is not zero.
        let n_samples = 256usize;
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| ((i % 32) as f32 / 32.0 - 0.5) * 0.8)
            .collect();
        let waveform = Tensor::from_vec(samples, (1usize, n_samples), &device)
            .expect("Failed to create waveform");

        let encoded = encoder
            .encode_continuous(&waveform)
            .expect("encode_continuous failed");

        let decoded = decoder
            .decode_continuous(&encoded)
            .expect("decode_continuous failed");

        // Max absolute value should be > 0
        let vals: Vec<f32> = decoded
            .flatten_all()
            .expect("flatten failed")
            .to_vec1()
            .expect("to_vec1 failed");

        let max_abs = vals.iter().cloned().map(f32::abs).fold(0.0_f32, f32::max);
        assert!(
            max_abs > 1e-6,
            "decode_continuous output is effectively zero (max_abs = {max_abs}); expected a real transformation"
        );
    }

    // ------------------------------------------------------------------
    // CodecQualityMetrics::compute_from_audio tests
    // ------------------------------------------------------------------

    /// Metrics computed from a non-trivial signal must be non-zero and finite.
    #[test]
    fn test_quality_metrics_from_audio_non_zero_non_nan() {
        // Generate a simple 100 Hz sine wave at 16 kHz, 0.5 s duration.
        let sample_rate = 16_000u32;
        let n_samples = 8_000usize;
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 100.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let metrics = CodecQualityMetrics::compute_from_audio(
            &samples,
            sample_rate,
            6.0,  // bitrate_kbps
            21.3, // compression_ratio
            13.3, // latency_ms
        );

        // All fields must be finite
        assert!(
            metrics.snr_db.is_finite(),
            "snr_db must be finite, got {}",
            metrics.snr_db
        );
        assert!(
            metrics.pesq_score.is_finite(),
            "pesq_score must be finite, got {}",
            metrics.pesq_score
        );
        assert!(
            metrics.stoi_score.is_finite(),
            "stoi_score must be finite, got {}",
            metrics.stoi_score
        );
        assert!(
            metrics.mcd.is_finite(),
            "mcd (spectral_flatness) must be finite, got {}",
            metrics.mcd
        );

        // SNR of a sine wave should be noticeably positive
        assert!(
            metrics.snr_db > 0.0,
            "SNR of a sine wave should be > 0, got {}",
            metrics.snr_db
        );

        // PESQ score must be in valid range
        assert!(
            (1.0..=4.5).contains(&metrics.pesq_score),
            "PESQ out of range: {}",
            metrics.pesq_score
        );

        // Bandwidth fraction (stoi_score proxy) must be in [0, 1]
        assert!(
            (0.0..=1.0).contains(&metrics.stoi_score),
            "Bandwidth fraction out of range: {}",
            metrics.stoi_score
        );

        // Passed-through fields must match
        assert!(
            (metrics.bitrate_kbps - 6.0).abs() < 1e-6,
            "bitrate_kbps mismatch"
        );
        assert!(
            (metrics.compression_ratio - 21.3).abs() < 1e-5,
            "compression_ratio mismatch"
        );
        assert!(
            (metrics.latency_ms - 13.3).abs() < 1e-5,
            "latency_ms mismatch"
        );
    }

    /// Silence input must produce a stable (non-NaN) result even if SNR is degenerate.
    #[test]
    fn test_quality_metrics_from_silence_stable() {
        let samples = vec![0.0_f32; 8000];
        let metrics = CodecQualityMetrics::compute_from_audio(&samples, 16_000, 6.0, 21.0, 13.0);

        // Must not produce NaN
        assert!(!metrics.snr_db.is_nan(), "snr_db is NaN for silence input");
        assert!(
            !metrics.pesq_score.is_nan(),
            "pesq_score is NaN for silence input"
        );
        assert!(
            !metrics.stoi_score.is_nan(),
            "stoi_score is NaN for silence input"
        );
        assert!(!metrics.mcd.is_nan(), "mcd is NaN for silence input");
    }

    /// White noise should yield a higher spectral flatness (mcd ≈ 1) than a pure sine (mcd ≈ 0).
    #[test]
    fn test_quality_metrics_spectral_flatness_ordering() {
        let sample_rate = 16_000u32;
        let n_samples = 8_000usize;

        // Pure 100 Hz sine — tonal, low flatness expected
        let sine: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 100.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        // Deterministic pseudo-noise via simple LCG to avoid rand dependency
        let mut state = 0x12345678u32;
        let noise: Vec<f32> = (0..n_samples)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        let sine_metrics =
            CodecQualityMetrics::compute_from_audio(&sine, sample_rate, 6.0, 21.0, 13.0);
        let noise_metrics =
            CodecQualityMetrics::compute_from_audio(&noise, sample_rate, 6.0, 21.0, 13.0);

        assert!(
            noise_metrics.mcd > sine_metrics.mcd,
            "Noise should have higher spectral flatness than sine: noise_mcd={} sine_mcd={}",
            noise_metrics.mcd,
            sine_metrics.mcd
        );
    }

    /// The SciRS2 `rfft`-based one-sided power spectrum must match a reference
    /// direct O(N²) DFT bin-for-bin, proving the FFT upgrade is behaviour-
    /// preserving: identical length (`N/2 + 1`), scaling (unnormalised) and
    /// rectangular windowing.
    #[test]
    fn test_one_sided_power_spectrum_matches_direct_dft() {
        // Deterministic, non-trivial signal: DC offset + two tones so that
        // both low and high bins carry energy.
        let fft_len = 64usize;
        let window: Vec<f32> = (0..fft_len)
            .map(|n| {
                let t = n as f32;
                0.3 + (2.0 * std::f32::consts::PI * 3.0 * t / fft_len as f32).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * 11.0 * t / fft_len as f32).cos()
            })
            .collect();

        // Reference: naive one-sided |X[k]|² via the textbook DFT formula,
        // accumulated in f64 to mirror the production rfft path's precision.
        let n_bins = fft_len / 2 + 1;
        let mut reference: Vec<f32> = Vec::with_capacity(n_bins);
        for k in 0..n_bins {
            let theta = -2.0 * std::f64::consts::PI * k as f64 / fft_len as f64;
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for (n, &s) in window.iter().enumerate() {
                let angle = theta * n as f64;
                re += f64::from(s) * angle.cos();
                im += f64::from(s) * angle.sin();
            }
            reference.push((re * re + im * im) as f32);
        }

        let power = CodecQualityMetrics::one_sided_power_spectrum(&window);

        assert_eq!(
            power.len(),
            reference.len(),
            "power spectrum length must equal fft_len/2 + 1"
        );

        // Both paths sum in f64 and cast to f32, so only float rounding can
        // separate them. Bound the per-bin error relative to the spectral peak
        // (with a tiny absolute floor for near-zero bins).
        let peak = reference.iter().copied().fold(0.0_f32, f32::max).max(1e-12);
        let tol = 1e-5_f32 * peak + 1e-6_f32;
        for (k, (&p, &r)) in power.iter().zip(reference.iter()).enumerate() {
            assert!(
                (p - r).abs() <= tol,
                "bin {k}: rfft power {p} vs reference DFT {r} exceeds tolerance {tol}"
            );
        }
    }
}
