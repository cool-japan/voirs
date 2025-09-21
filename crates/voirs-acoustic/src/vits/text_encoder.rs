//! VITS Text Encoder implementation
//!
//! This module implements the transformer-based text encoder for VITS,
//! which converts phoneme sequences to latent representations.

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Activation, Embedding, LayerNorm, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{AcousticError, Phoneme, Result};

/// Configuration for VITS text encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEncoderConfig {
    /// Number of transformer layers
    pub n_layers: usize,
    /// Hidden dimension size
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Feed-forward dimension
    pub d_ff: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Vocabulary size (number of unique phonemes)
    pub vocab_size: usize,
    /// Kernel size for convolution layers
    pub kernel_size: usize,
    /// Number of convolution layers
    pub n_conv_layers: usize,
    /// Whether to use relative positional encoding
    pub use_relative_pos: bool,
}

impl Default for TextEncoderConfig {
    fn default() -> Self {
        Self {
            n_layers: 6,
            d_model: 192,
            n_heads: 2,
            d_ff: 768,
            dropout: 0.1,
            max_seq_len: 1000,
            vocab_size: 256,
            kernel_size: 3,
            n_conv_layers: 3,
            use_relative_pos: true,
        }
    }
}

/// Multi-head self-attention layer
pub struct MultiHeadAttention {
    config: TextEncoderConfig,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    scale: f64,
}

impl MultiHeadAttention {
    pub fn new(config: &TextEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
        let d_model = config.d_model;
        let n_heads = config.n_heads;

        if d_model % n_heads != 0 {
            return Err(candle_core::Error::Msg(
                "d_model must be divisible by n_heads".to_string(),
            ));
        }

        let scale = 1.0 / ((d_model / n_heads) as f64).sqrt();

        Ok(Self {
            config: config.clone(),
            q_proj: candle_nn::linear(d_model, d_model, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear(d_model, d_model, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear(d_model, d_model, vb.pp("v_proj"))?,
            out_proj: candle_nn::linear(d_model, d_model, vb.pp("out_proj"))?,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let (batch_size, seq_len, d_model) = x.dims3()?;
        let n_heads = self.config.n_heads;
        let d_head = d_model / n_heads;

        // Linear projections
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = q
            .reshape((batch_size, seq_len, n_heads, d_head))?
            .transpose(1, 2)?; // (batch_size, n_heads, seq_len, d_head)
        let k = k
            .reshape((batch_size, seq_len, n_heads, d_head))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, n_heads, d_head))?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores * self.scale)?;

        // Apply mask if provided
        let attn_weights = if let Some(mask) = mask {
            let mask = mask.unsqueeze(1)?.unsqueeze(1)?; // (batch_size, 1, 1, seq_len)
            let large_neg = Tensor::full(-1e9f32, scores.shape(), scores.device())?;
            let masked_scores = mask.where_cond(&scores, &large_neg)?;
            candle_nn::ops::softmax_last_dim(&masked_scores)?
        } else {
            candle_nn::ops::softmax_last_dim(&scores)?
        };

        // Apply attention to values
        let out = attn_weights.matmul(&v)?;

        // Reshape and project output
        let out = out
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, d_model))?;

        self.out_proj.forward(&out)
    }
}

/// Position-wise feed-forward network
pub struct PositionwiseFeedForward {
    linear1: Linear,
    linear2: Linear,
    activation: Activation,
}

impl PositionwiseFeedForward {
    pub fn new(config: &TextEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
        let d_model = config.d_model;
        let d_ff = config.d_ff;

        Ok(Self {
            linear1: candle_nn::linear(d_model, d_ff, vb.pp("linear1"))?,
            linear2: candle_nn::linear(d_ff, d_model, vb.pp("linear2"))?,
            activation: Activation::Relu,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = self.activation.forward(&x)?;
        self.linear2.forward(&x)
    }
}

/// Transformer encoder layer
pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    feed_forward: PositionwiseFeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    #[allow(dead_code)]
    dropout: f64,
}

impl TransformerEncoderLayer {
    pub fn new(config: &TextEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
        let d_model = config.d_model;

        Ok(Self {
            self_attn: MultiHeadAttention::new(config, vb.pp("self_attn"))?,
            feed_forward: PositionwiseFeedForward::new(config, vb.pp("feed_forward"))?,
            norm1: candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm1"))?,
            norm2: candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm2"))?,
            dropout: config.dropout,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> CandleResult<Tensor> {
        // Self-attention with residual connection and layer norm
        let attn_out = self.self_attn.forward(x, mask)?;
        let x = self.norm1.forward(&(x + attn_out)?)?;

        // Feed-forward with residual connection and layer norm
        let ff_out = self.feed_forward.forward(&x)?;
        let x = self.norm2.forward(&(x + ff_out)?)?;

        Ok(x)
    }
}

/// Positional encoding for transformer
pub struct PositionalEncoding {
    encoding: Tensor,
    max_seq_len: usize,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_seq_len: usize, device: &Device) -> CandleResult<Self> {
        let mut encoding = Vec::new();

        for pos in 0..max_seq_len {
            let mut pos_encoding = Vec::new();
            for i in 0..d_model {
                let angle = pos as f64 / 10000.0_f64.powf(2.0 * (i / 2) as f64 / d_model as f64);
                if i % 2 == 0 {
                    pos_encoding.push(angle.sin() as f32);
                } else {
                    pos_encoding.push(angle.cos() as f32);
                }
            }
            encoding.push(pos_encoding);
        }

        let encoding = Tensor::from_vec(
            encoding.into_iter().flatten().collect::<Vec<f32>>(),
            (max_seq_len, d_model),
            device,
        )?;

        Ok(Self {
            encoding,
            max_seq_len,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (batch_size, seq_len, d_model) = x.dims3()?;

        if seq_len > self.max_seq_len {
            return Err(candle_core::Error::Msg(format!(
                "Sequence length {seq_len} exceeds maximum {}",
                self.max_seq_len
            )));
        }

        let pos_enc = self.encoding.narrow(0, 0, seq_len)?;
        let pos_enc = pos_enc
            .unsqueeze(0)?
            .expand((batch_size, seq_len, d_model))?;

        x + pos_enc
    }
}

/// Phoneme embedding layer
pub struct PhonemeEmbedding {
    embedding: Embedding,
    phoneme_to_id: HashMap<String, u32>,
    #[allow(dead_code)]
    id_to_phoneme: HashMap<u32, String>,
}

impl PhonemeEmbedding {
    pub fn new(config: &TextEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
        let embedding =
            candle_nn::embedding(config.vocab_size, config.d_model, vb.pp("embedding"))?;

        // Initialize phoneme mapping (simplified - in practice this would be loaded from a file)
        let mut phoneme_to_id = HashMap::new();
        let mut id_to_phoneme = HashMap::new();

        // Add common phonemes (this is a simplified set)
        let common_phonemes = vec![
            "<pad>", "<unk>", "<bos>", "<eos>", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D",
            "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG",
            "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
        ];

        for (i, phoneme) in common_phonemes.iter().enumerate() {
            phoneme_to_id.insert(phoneme.to_string(), i as u32);
            id_to_phoneme.insert(i as u32, phoneme.to_string());
        }

        Ok(Self {
            embedding,
            phoneme_to_id,
            id_to_phoneme,
        })
    }

    pub fn encode_phonemes(&self, phonemes: &[Phoneme]) -> Vec<u32> {
        phonemes
            .iter()
            .map(|p| {
                self.phoneme_to_id.get(&p.symbol).copied().unwrap_or(1) // <unk> token
            })
            .collect()
    }

    pub fn forward(&self, phoneme_ids: &[u32], device: &Device) -> CandleResult<Tensor> {
        let ids_tensor = Tensor::from_vec(
            phoneme_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
            (phoneme_ids.len(),),
            device,
        )?;

        self.embedding.forward(&ids_tensor)
    }
}

/// VITS Text Encoder
pub struct TextEncoder {
    config: TextEncoderConfig,
    phoneme_embedding: PhonemeEmbedding,
    pos_encoding: PositionalEncoding,
    layers: Vec<TransformerEncoderLayer>,
    final_norm: LayerNorm,
    device: Device,
}

impl TextEncoder {
    pub fn new(config: TextEncoderConfig, device: Device) -> Result<Self> {
        // Create a VarBuilder with random initialization for testing
        let mut tensors = HashMap::new();

        // Add all the required tensors with proper initialization (embedding is [vocab_size, d_model])
        tensors.insert(
            "phoneme_embedding.embedding.weight".to_string(),
            Tensor::randn(0f32, 1f32, (config.vocab_size, config.d_model), &device)
                .map_err(|e| AcousticError::ModelError(e.to_string()))?,
        );
        tensors.insert(
            "final_norm.weight".to_string(),
            Tensor::ones((config.d_model,), DType::F32, &device)
                .map_err(|e| AcousticError::ModelError(e.to_string()))?,
        );
        tensors.insert(
            "final_norm.bias".to_string(),
            Tensor::zeros((config.d_model,), DType::F32, &device)
                .map_err(|e| AcousticError::ModelError(e.to_string()))?,
        );

        // Add tensors for each transformer layer
        for i in 0..config.n_layers {
            let layer_prefix = format!("layer_{i}");

            // Self-attention tensors (note: weight shape is [output_features, input_features])
            tensors.insert(
                format!("{layer_prefix}.self_attn.q_proj.weight"),
                Tensor::randn(0f32, 1f32, (config.d_model, config.d_model), &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );
            tensors.insert(
                format!("{layer_prefix}.self_attn.q_proj.bias"),
                Tensor::zeros((config.d_model,), DType::F32, &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );
            tensors.insert(
                format!("{layer_prefix}.self_attn.k_proj.weight"),
                Tensor::randn(0f32, 1f32, (config.d_model, config.d_model), &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );
            tensors.insert(
                format!("{layer_prefix}.self_attn.k_proj.bias"),
                Tensor::zeros((config.d_model,), DType::F32, &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );
            tensors.insert(
                format!("{layer_prefix}.self_attn.v_proj.weight"),
                Tensor::randn(0f32, 1f32, (config.d_model, config.d_model), &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );
            tensors.insert(
                format!("{layer_prefix}.self_attn.v_proj.bias"),
                Tensor::zeros((config.d_model,), DType::F32, &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );
            tensors.insert(
                format!("{layer_prefix}.self_attn.out_proj.weight"),
                Tensor::randn(0f32, 1f32, (config.d_model, config.d_model), &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );
            tensors.insert(
                format!("{layer_prefix}.self_attn.out_proj.bias"),
                Tensor::zeros((config.d_model,), DType::F32, &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );

            // Feed-forward tensors (note: weight shape is [output_features, input_features])
            tensors.insert(
                format!("{layer_prefix}.feed_forward.linear1.weight"),
                Tensor::randn(0f32, 1f32, (config.d_ff, config.d_model), &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );
            tensors.insert(
                format!("{layer_prefix}.feed_forward.linear1.bias"),
                Tensor::zeros((config.d_ff,), DType::F32, &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );
            tensors.insert(
                format!("{layer_prefix}.feed_forward.linear2.weight"),
                Tensor::randn(0f32, 1f32, (config.d_model, config.d_ff), &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );
            tensors.insert(
                format!("{layer_prefix}.feed_forward.linear2.bias"),
                Tensor::zeros((config.d_model,), DType::F32, &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );

            // Layer norm tensors
            tensors.insert(
                format!("{layer_prefix}.norm1.weight"),
                Tensor::ones((config.d_model,), DType::F32, &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );
            tensors.insert(
                format!("{layer_prefix}.norm1.bias"),
                Tensor::zeros((config.d_model,), DType::F32, &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );
            tensors.insert(
                format!("{layer_prefix}.norm2.weight"),
                Tensor::ones((config.d_model,), DType::F32, &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );
            tensors.insert(
                format!("{layer_prefix}.norm2.bias"),
                Tensor::zeros((config.d_model,), DType::F32, &device)
                    .map_err(|e| AcousticError::ModelError(e.to_string()))?,
            );
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let phoneme_embedding = PhonemeEmbedding::new(&config, vb.pp("phoneme_embedding"))
            .map_err(|e| {
                AcousticError::ModelError(format!("Failed to create phoneme embedding: {e}"))
            })?;

        let pos_encoding = PositionalEncoding::new(config.d_model, config.max_seq_len, &device)
            .map_err(|e| {
                AcousticError::ModelError(format!("Failed to create positional encoding: {e}"))
            })?;

        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            let layer = TransformerEncoderLayer::new(&config, vb.pp(format!("layer_{i}")))
                .map_err(|e| {
                    AcousticError::ModelError(format!(
                        "Failed to create transformer layer {i}: {e}"
                    ))
                })?;
            layers.push(layer);
        }

        let final_norm = candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("final_norm"))
            .map_err(|e| AcousticError::ModelError(format!("Failed to create final norm: {e}")))?;

        Ok(Self {
            config,
            phoneme_embedding,
            pos_encoding,
            layers,
            final_norm,
            device,
        })
    }

    /// Forward pass through the text encoder
    pub fn forward(&self, phonemes: &[Phoneme], lengths: Option<&[usize]>) -> Result<Tensor> {
        if phonemes.is_empty() {
            return Err(AcousticError::InputError(
                "Empty phoneme sequence".to_string(),
            ));
        }

        // Encode phonemes to IDs
        let phoneme_ids = self.phoneme_embedding.encode_phonemes(phonemes);

        // Get embeddings
        let mut x = self
            .phoneme_embedding
            .forward(&phoneme_ids, &self.device)
            .map_err(|e| AcousticError::InferenceError(format!("Failed to embed phonemes: {e}")))?;

        // Add batch dimension
        x = x.unsqueeze(0).map_err(|e| {
            AcousticError::InferenceError(format!("Failed to add batch dimension: {e}"))
        })?;

        // Add positional encoding
        x = self.pos_encoding.forward(&x).map_err(|e| {
            AcousticError::InferenceError(format!("Failed to add positional encoding: {e}"))
        })?;

        // Create attention mask if lengths are provided
        let mask = if let Some(lengths) = lengths {
            Some(
                self.create_attention_mask(lengths, phonemes.len())
                    .map_err(|e| {
                        AcousticError::InferenceError(format!(
                            "Failed to create attention mask: {e}"
                        ))
                    })?,
            )
        } else {
            None
        };

        // Pass through transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, mask.as_ref()).map_err(|e| {
                AcousticError::InferenceError(format!("Failed in transformer layer {i}: {e}"))
            })?;
        }

        // Final layer normalization
        x = self.final_norm.forward(&x).map_err(|e| {
            AcousticError::InferenceError(format!("Failed in final normalization: {e}"))
        })?;

        Ok(x)
    }

    /// Create attention mask for variable length sequences
    fn create_attention_mask(&self, lengths: &[usize], max_len: usize) -> CandleResult<Tensor> {
        let batch_size = lengths.len();
        let mut mask_data = vec![false; batch_size * max_len];

        for (batch_idx, &length) in lengths.iter().enumerate() {
            for pos in 0..length.min(max_len) {
                mask_data[batch_idx * max_len + pos] = true;
            }
        }

        Tensor::from_vec(
            mask_data
                .into_iter()
                .map(|b| if b { 1.0f32 } else { 0.0f32 })
                .collect(),
            (batch_size, max_len),
            &self.device,
        )
    }

    /// Get configuration
    pub fn config(&self) -> &TextEncoderConfig {
        &self.config
    }

    /// Get phoneme vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.phoneme_embedding.phoneme_to_id.len()
    }

    /// Get supported phonemes
    pub fn supported_phonemes(&self) -> Vec<String> {
        self.phoneme_embedding
            .phoneme_to_id
            .keys()
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn create_test_config() -> TextEncoderConfig {
        TextEncoderConfig {
            n_layers: 2,
            d_model: 64,
            n_heads: 2,
            d_ff: 128,
            dropout: 0.1,
            max_seq_len: 100,
            vocab_size: 50,
            kernel_size: 3,
            n_conv_layers: 2,
            use_relative_pos: false,
        }
    }

    fn create_test_phonemes() -> Vec<Phoneme> {
        vec![
            Phoneme::new("HH"),
            Phoneme::new("EH"),
            Phoneme::new("L"),
            Phoneme::new("OW"),
        ]
    }

    #[test]
    fn test_text_encoder_config() {
        let config = TextEncoderConfig::default();
        assert_eq!(config.n_layers, 6);
        assert_eq!(config.d_model, 192);
        assert_eq!(config.n_heads, 2);
        assert!(config.d_model % config.n_heads == 0);
    }

    #[test]
    fn test_phoneme_embedding() {
        let config = create_test_config();
        let device = Device::Cpu;

        // Create VarBuilder with embedding tensor
        let mut tensors = HashMap::new();
        tensors.insert(
            "embedding.embedding.weight".to_string(),
            Tensor::randn(0f32, 1f32, (config.vocab_size, config.d_model), &device).unwrap(),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let phoneme_embedding = PhonemeEmbedding::new(&config, vb.pp("embedding")).unwrap();
        let phonemes = create_test_phonemes();
        let ids = phoneme_embedding.encode_phonemes(&phonemes);

        assert_eq!(ids.len(), phonemes.len());
        assert!(ids
            .iter()
            .all(|&id| (id as usize) < phoneme_embedding.phoneme_to_id.len()));
    }

    #[test]
    fn test_positional_encoding() {
        let device = Device::Cpu;
        let d_model = 64;
        let max_seq_len = 100;

        let pos_enc = PositionalEncoding::new(d_model, max_seq_len, &device).unwrap();

        // Test with a sequence
        let seq_len = 10;
        let batch_size = 2;
        let x = Tensor::zeros((batch_size, seq_len, d_model), DType::F32, &device).unwrap();

        let result = pos_enc.forward(&x).unwrap();
        assert_eq!(result.dims(), [batch_size, seq_len, d_model]);
    }

    #[test]
    fn test_text_encoder_creation() {
        let config = create_test_config();
        let device = Device::Cpu;

        let encoder = TextEncoder::new(config.clone(), device);
        assert!(encoder.is_ok());

        let encoder = encoder.unwrap();
        assert_eq!(encoder.config().n_layers, config.n_layers);
        assert_eq!(encoder.config().d_model, config.d_model);
    }

    #[test]
    fn test_text_encoder_forward() {
        let config = create_test_config();
        let device = Device::Cpu;

        let encoder = TextEncoder::new(config, device).unwrap();
        let phonemes = create_test_phonemes();

        let result = encoder.forward(&phonemes, None);
        match &result {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Text encoder forward failed: {e:?}");
                panic!("Text encoder forward failed: {e:?}");
            }
        }

        let output = result.unwrap();
        assert_eq!(output.dims(), [1, phonemes.len(), encoder.config().d_model]);
    }

    #[test]
    fn test_text_encoder_empty_input() {
        let config = create_test_config();
        let device = Device::Cpu;

        let encoder = TextEncoder::new(config, device).unwrap();
        let result = encoder.forward(&[], None);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AcousticError::InputError(_)));
    }

    #[test]
    fn test_attention_mask_creation() {
        let config = create_test_config();
        let device = Device::Cpu;

        let encoder = TextEncoder::new(config, device).unwrap();
        let lengths = vec![3, 5, 2];
        let max_len = 6;

        let mask = encoder.create_attention_mask(&lengths, max_len).unwrap();
        assert_eq!(mask.dims(), [3, 6]);
    }

    #[test]
    fn test_supported_phonemes() {
        let config = create_test_config();
        let device = Device::Cpu;

        let encoder = TextEncoder::new(config, device).unwrap();
        let phonemes = encoder.supported_phonemes();

        assert!(!phonemes.is_empty());
        assert!(phonemes.contains(&"<pad>".to_string()));
        assert!(phonemes.contains(&"<unk>".to_string()));
        assert!(phonemes.contains(&"AA".to_string()));
    }
}
