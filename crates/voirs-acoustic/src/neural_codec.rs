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

use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Linear, Module};
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

/// Neural audio codec encoder
pub struct NeuralEncoder {
    /// Configuration
    config: NeuralCodecConfig,
    /// Convolutional layers for encoding
    #[allow(dead_code)]
    layers: Vec<Arc<dyn Module + Send + Sync>>,
    /// RVQ for quantization
    rvq: ResidualVectorQuantizer,
    /// Device
    device: Device,
}

impl std::fmt::Debug for NeuralEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralEncoder")
            .field("config", &self.config)
            .field("layers", &format!("<{} layers>", self.layers.len()))
            .field("device", &self.device)
            .finish()
    }
}

impl NeuralEncoder {
    /// Create new neural encoder
    pub fn new(config: NeuralCodecConfig, device: &Device) -> Result<Self> {
        config.validate()?;

        let rvq = ResidualVectorQuantizer::new(
            config.num_codebooks,
            config.codebook_size,
            config.encoder_dim,
            device,
        )?;

        Ok(Self {
            config,
            layers: Vec::new(), // Placeholder
            rvq,
            device: device.clone(),
        })
    }

    /// Encode audio waveform to discrete codes
    pub fn encode(&self, waveform: &Tensor) -> Result<Vec<Vec<usize>>> {
        // 1. Encode waveform to continuous representation
        let encoded = self.encode_continuous(waveform)?;

        // 2. Quantize to discrete codes using RVQ
        self.rvq.encode(&encoded)
    }

    /// Encode waveform to continuous representation (before quantization)
    fn encode_continuous(&self, waveform: &Tensor) -> Result<Tensor> {
        // Placeholder: return input reshaped
        // In production, this would apply convolutional encoding layers

        let batch_size = waveform.dims()[0];
        let waveform_len = if waveform.dims().len() > 1 {
            waveform.dims()[1]
        } else {
            1
        };

        // Compute output sequence length based on hop length
        let seq_len = waveform_len / self.config.hop_length;

        Tensor::zeros(
            (batch_size, seq_len, self.config.encoder_dim),
            candle_core::DType::F32,
            &self.device,
        )
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("Failed to create encoded tensor: {}", e),
        })
    }

    /// Get encoder configuration
    pub fn config(&self) -> &NeuralCodecConfig {
        &self.config
    }
}

/// Neural audio codec decoder
pub struct NeuralDecoder {
    /// Configuration
    config: NeuralCodecConfig,
    /// Transposed convolutional layers for decoding
    #[allow(dead_code)]
    layers: Vec<Arc<dyn Module + Send + Sync>>,
    /// RVQ for dequantization
    rvq: Arc<ResidualVectorQuantizer>,
    /// Device
    device: Device,
}

impl std::fmt::Debug for NeuralDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralDecoder")
            .field("config", &self.config)
            .field("layers", &format!("<{} layers>", self.layers.len()))
            .field("device", &self.device)
            .finish()
    }
}

impl NeuralDecoder {
    /// Create new neural decoder
    pub fn new(
        config: NeuralCodecConfig,
        rvq: Arc<ResidualVectorQuantizer>,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            layers: Vec::new(), // Placeholder
            rvq,
            device: device.clone(),
        })
    }

    /// Decode discrete codes back to audio waveform
    pub fn decode(&self, codes: &[Vec<usize>]) -> Result<Tensor> {
        // 1. Dequantize codes to continuous representation
        let continuous = self.rvq.decode(codes)?;

        // 2. Decode continuous representation to waveform
        self.decode_continuous(&continuous)
    }

    /// Decode continuous representation to waveform
    fn decode_continuous(&self, encoded: &Tensor) -> Result<Tensor> {
        // Placeholder: return dummy waveform
        // In production, this would apply transposed convolutional layers

        let batch_size = encoded.dims()[0];
        let seq_len = encoded.dims()[1];

        // Compute output waveform length
        let waveform_len = seq_len * self.config.hop_length;

        Tensor::zeros(
            (batch_size, waveform_len),
            candle_core::DType::F32,
            &self.device,
        )
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("Failed to create decoded waveform: {}", e),
        })
    }

    /// Get decoder configuration
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
    /// Create placeholder metrics (to be filled by actual evaluation)
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
}
