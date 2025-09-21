//! Multi-head attention implementation for Whisper
//!
//! This module provides memory-efficient attention computation with support for
//! causal masking, KV-caching, and Flash Attention optimizations.

use crate::RecognitionError;
use candle_core::{Device, Tensor};
use candle_nn::{ops::softmax_last_dim, Module, VarBuilder};

/// Multi-head attention layer with memory-efficient computation
pub struct MultiHeadAttention {
    /// Query projection
    query: candle_nn::Linear,
    /// Key projection
    key: candle_nn::Linear,
    /// Value projection
    value: candle_nn::Linear,
    /// Output projection
    out: candle_nn::Linear,
    /// Number of heads
    n_head: usize,
    /// Head dimension
    head_dim: usize,
}

/// KV-cache for efficient autoregressive generation
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Cached key states for each layer
    key_cache: Vec<Option<Tensor>>,
    /// Cached value states for each layer
    value_cache: Vec<Option<Tensor>>,
    /// Current sequence length
    seq_len: usize,
    /// Maximum sequence length
    #[allow(dead_code)]
    max_seq_len: usize,
    /// Number of layers
    n_layers: usize,
}

impl KVCache {
    /// Create a new KV cache
    #[must_use]
    pub fn new(n_layers: usize, max_seq_len: usize) -> Self {
        Self {
            key_cache: vec![None; n_layers],
            value_cache: vec![None; n_layers],
            seq_len: 0,
            max_seq_len,
            n_layers,
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.key_cache.fill(None);
        self.value_cache.fill(None);
        self.seq_len = 0;
    }

    /// Get cached key-value pairs for a layer
    #[must_use]
    pub fn get_kv(&self, layer_idx: usize) -> Option<(&Tensor, &Tensor)> {
        if layer_idx >= self.n_layers {
            return None;
        }

        match (&self.key_cache[layer_idx], &self.value_cache[layer_idx]) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    /// Update cached key-value pairs for a layer
    ///
    /// # Errors
    ///
    /// Returns `RecognitionError::ModelError` if the layer index is out of bounds
    pub fn update_kv(
        &mut self,
        layer_idx: usize,
        key: Tensor,
        value: Tensor,
    ) -> Result<(), RecognitionError> {
        if layer_idx >= self.n_layers {
            return Err(RecognitionError::ModelError {
                message: format!("Layer index {layer_idx} out of bounds"),
                source: None,
            });
        }

        let new_key = if let Some(cached_key) = &self.key_cache[layer_idx] {
            Tensor::cat(&[cached_key, &key], 2).map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to concatenate key cache: {e}"),
                source: Some(Box::new(e)),
            })?
        } else {
            key
        };

        let new_value = if let Some(cached_value) = &self.value_cache[layer_idx] {
            Tensor::cat(&[cached_value, &value], 2).map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to concatenate value cache: {e}"),
                source: Some(Box::new(e)),
            })?
        } else {
            value
        };

        self.key_cache[layer_idx] = Some(new_key);
        self.value_cache[layer_idx] = Some(new_value);

        Ok(())
    }

    /// Get current sequence length
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Set sequence length
    pub fn set_seq_len(&mut self, len: usize) {
        self.seq_len = len;
    }
}

impl MultiHeadAttention {
    /// Creates a new multi-head attention layer
    ///
    /// # Errors
    ///
    /// Returns `RecognitionError::ModelLoadError` if the projection layers fail to initialize
    pub fn new(n_state: usize, n_head: usize, vs: &VarBuilder) -> Result<Self, RecognitionError> {
        let head_dim = n_state / n_head;

        let query = candle_nn::linear(n_state, n_state, vs.pp("q_proj")).map_err(|e| {
            RecognitionError::ModelLoadError {
                message: format!("Failed to create query projection: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        let key = candle_nn::linear(n_state, n_state, vs.pp("k_proj")).map_err(|e| {
            RecognitionError::ModelLoadError {
                message: format!("Failed to create key projection: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        let value = candle_nn::linear(n_state, n_state, vs.pp("v_proj")).map_err(|e| {
            RecognitionError::ModelLoadError {
                message: format!("Failed to create value projection: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        let out = candle_nn::linear(n_state, n_state, vs.pp("out_proj")).map_err(|e| {
            RecognitionError::ModelLoadError {
                message: format!("Failed to create output projection: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
            head_dim,
        })
    }

    /// Performs forward pass through the attention mechanism
    ///
    /// # Errors
    ///
    /// Returns `RecognitionError` if tensor operations fail during attention computation
    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, RecognitionError> {
        self.forward_with_flash_attention(q, k, v, false)
    }

    /// Forward pass with causal masking (for decoder self-attention)
    ///
    /// # Errors
    ///
    /// Returns `RecognitionError` if tensor operations fail during causal attention computation
    pub fn forward_causal(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Tensor, RecognitionError> {
        self.forward_with_flash_attention(q, k, v, true)
    }

    /// Memory-efficient attention computation with Flash Attention-like optimization
    fn forward_with_flash_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
    ) -> Result<Tensor, RecognitionError> {
        let (_batch_size, seq_len, _) = q.dims3().map_err(|e| RecognitionError::ModelError {
            message: format!("Invalid query tensor shape: {e}"),
            source: Some(Box::new(e)),
        })?;

        // Project to Q, K, V
        let q = self
            .query
            .forward(q)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Query projection failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let k = self
            .key
            .forward(k)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Key projection failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let v = self
            .value
            .forward(v)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Value projection failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        // For long sequences, use tiled computation to reduce memory usage
        if seq_len > 1024 {
            self.forward_tiled(&q, &k, &v, causal)
        } else {
            self.forward_standard(&q, &k, &v, causal)
        }
    }

    /// Standard attention computation for shorter sequences
    #[allow(clippy::too_many_lines)]
    fn forward_standard(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
    ) -> Result<Tensor, RecognitionError> {
        let (batch_size, seq_len, _) = q.dims3().map_err(|e| RecognitionError::ModelError {
            message: format!("Invalid query tensor shape: {e}"),
            source: Some(Box::new(e)),
        })?;

        let k_seq_len = k.dim(1).map_err(|e| RecognitionError::ModelError {
            message: format!("Key sequence length extraction failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        // Reshape for multi-head attention: [batch, seq, n_head, head_dim]
        let q = q
            .reshape((batch_size, seq_len, self.n_head, self.head_dim))
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Query reshape failed: {e}"),
                source: Some(Box::new(e)),
            })?
            .transpose(1, 2)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Query transpose failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let k = k
            .reshape((batch_size, k_seq_len, self.n_head, self.head_dim))
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Key reshape failed: {e}"),
                source: Some(Box::new(e)),
            })?
            .transpose(1, 2)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Key transpose failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let v = v
            .reshape((batch_size, k_seq_len, self.n_head, self.head_dim))
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Value reshape failed: {e}"),
                source: Some(Box::new(e)),
            })?
            .transpose(1, 2)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Value transpose failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Compute attention scores: Q @ K^T / sqrt(head_dim)
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let k_t = k
            .transpose(2, 3)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Key transpose for attention failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let scores = q.matmul(&k_t).map_err(|e| RecognitionError::ModelError {
            message: format!("Attention score computation failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        let scores = (scores * scale).map_err(|e| RecognitionError::ModelError {
            message: format!("Attention score scaling failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        // Apply causal mask if needed
        let scores = if causal {
            let mask = self.create_causal_mask(seq_len, q.device()).map_err(|e| {
                RecognitionError::ModelError {
                    message: format!("Causal mask creation failed: {e}"),
                    source: Some(Box::new(e)),
                }
            })?;
            scores
                .broadcast_add(&mask)
                .map_err(|e| RecognitionError::ModelError {
                    message: format!("Causal mask application failed: {e}"),
                    source: Some(Box::new(e)),
                })?
        } else {
            scores
        };

        // Apply softmax
        let weights = softmax_last_dim(&scores).map_err(|e| RecognitionError::ModelError {
            message: format!("Attention softmax failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        // Apply attention to values
        let output = weights
            .matmul(&v)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Attention application failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Reshape back: [batch, n_head, seq, head_dim] -> [batch, seq, n_state]
        let output = output
            .transpose(1, 2)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Output transpose failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let output = output
            .reshape((batch_size, seq_len, self.n_head * self.head_dim))
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Output reshape failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Final projection
        let output = self
            .out
            .forward(&output)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Output projection failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        Ok(output)
    }

    /// Tiled attention computation for memory efficiency with long sequences
    #[allow(clippy::too_many_lines)]
    fn forward_tiled(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
    ) -> Result<Tensor, RecognitionError> {
        let (batch_size, seq_len, _) = q.dims3().map_err(|e| RecognitionError::ModelError {
            message: format!("Invalid query tensor shape: {e}"),
            source: Some(Box::new(e)),
        })?;

        let k_seq_len = k.dim(1).map_err(|e| RecognitionError::ModelError {
            message: format!("Key sequence length extraction failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        // Tile size for memory efficiency
        let tile_size = 512;
        let num_tiles = (seq_len + tile_size - 1) / tile_size;

        // Reshape for multi-head attention
        let q = q
            .reshape((batch_size, seq_len, self.n_head, self.head_dim))
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Query reshape failed: {e}"),
                source: Some(Box::new(e)),
            })?
            .transpose(1, 2)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Query transpose failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let k = k
            .reshape((batch_size, k_seq_len, self.n_head, self.head_dim))
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Key reshape failed: {e}"),
                source: Some(Box::new(e)),
            })?
            .transpose(1, 2)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Key transpose failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let v = v
            .reshape((batch_size, k_seq_len, self.n_head, self.head_dim))
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Value reshape failed: {e}"),
                source: Some(Box::new(e)),
            })?
            .transpose(1, 2)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Value transpose failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let k_t = k
            .transpose(2, 3)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Key transpose for attention failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let mut output_tiles = Vec::new();

        // Process in tiles to reduce memory usage
        for tile_idx in 0..num_tiles {
            let start_idx = tile_idx * tile_size;
            let end_idx = (start_idx + tile_size).min(seq_len);

            // Extract query tile
            let q_tile = q.narrow(2, start_idx, end_idx - start_idx).map_err(|e| {
                RecognitionError::ModelError {
                    message: format!("Query tile extraction failed: {e}"),
                    source: Some(Box::new(e)),
                }
            })?;

            // Compute attention for this tile
            let scores = q_tile
                .matmul(&k_t)
                .map_err(|e| RecognitionError::ModelError {
                    message: format!("Attention score computation failed: {e}"),
                    source: Some(Box::new(e)),
                })?;

            let scores = (scores * scale).map_err(|e| RecognitionError::ModelError {
                message: format!("Attention score scaling failed: {e}"),
                source: Some(Box::new(e)),
            })?;

            // Apply causal mask if needed
            let scores = if causal {
                let mask = self
                    .create_causal_mask_tile(start_idx, end_idx, k_seq_len, q.device())
                    .map_err(|e| RecognitionError::ModelError {
                        message: format!("Causal mask creation failed: {e}"),
                        source: Some(Box::new(e)),
                    })?;
                scores
                    .broadcast_add(&mask)
                    .map_err(|e| RecognitionError::ModelError {
                        message: format!("Causal mask application failed: {e}"),
                        source: Some(Box::new(e)),
                    })?
            } else {
                scores
            };

            // Apply softmax
            let weights = softmax_last_dim(&scores).map_err(|e| RecognitionError::ModelError {
                message: format!("Attention softmax failed: {e}"),
                source: Some(Box::new(e)),
            })?;

            // Apply attention to values
            let output_tile = weights
                .matmul(&v)
                .map_err(|e| RecognitionError::ModelError {
                    message: format!("Attention application failed: {e}"),
                    source: Some(Box::new(e)),
                })?;

            output_tiles.push(output_tile);
        }

        // Concatenate tiles
        let output = if output_tiles.len() == 1 {
            output_tiles.into_iter().next().unwrap()
        } else {
            let tile_refs: Vec<&Tensor> = output_tiles.iter().collect();
            Tensor::cat(&tile_refs, 2).map_err(|e| RecognitionError::ModelError {
                message: format!("Output tile concatenation failed: {e}"),
                source: Some(Box::new(e)),
            })?
        };

        // Reshape back
        let output = output
            .transpose(1, 2)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Output transpose failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let output = output
            .reshape((batch_size, seq_len, self.n_head * self.head_dim))
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Output reshape failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Final projection
        let output = self
            .out
            .forward(&output)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Output projection failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        Ok(output)
    }

    /// Create causal mask for autoregressive generation
    #[allow(clippy::unused_self)]
    fn create_causal_mask(
        &self,
        seq_len: usize,
        device: &Device,
    ) -> Result<Tensor, candle_core::Error> {
        let mut mask_data = vec![-f32::INFINITY; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 0.0;
            }
        }

        Tensor::from_slice(&mask_data, (seq_len, seq_len), device)
    }

    /// Create causal mask for a specific tile
    #[allow(clippy::unused_self)]
    fn create_causal_mask_tile(
        &self,
        start_idx: usize,
        end_idx: usize,
        k_seq_len: usize,
        device: &Device,
    ) -> Result<Tensor, candle_core::Error> {
        let tile_len = end_idx - start_idx;
        let mut mask_data = vec![-f32::INFINITY; tile_len * k_seq_len];

        for i in 0..tile_len {
            for j in 0..=(start_idx + i).min(k_seq_len - 1) {
                mask_data[i * k_seq_len + j] = 0.0;
            }
        }

        Tensor::from_slice(&mask_data, (tile_len, k_seq_len), device)
    }

    /// Forward pass with causal masking and KV-cache support for efficient generation
    ///
    /// # Errors
    ///
    /// Returns `RecognitionError` if tensor operations fail during cached attention computation
    pub async fn forward_causal_with_cache(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cache_layer_idx: usize,
        cache: &mut KVCache,
        use_cache: bool,
    ) -> Result<Tensor, RecognitionError> {
        if use_cache {
            // Update cache with new key-value pairs
            let k_proj = self
                .key
                .forward(k)
                .map_err(|e| RecognitionError::ModelError {
                    message: format!("Key projection failed: {e}"),
                    source: Some(Box::new(e)),
                })?;

            let v_proj = self
                .value
                .forward(v)
                .map_err(|e| RecognitionError::ModelError {
                    message: format!("Value projection failed: {e}"),
                    source: Some(Box::new(e)),
                })?;

            cache.update_kv(cache_layer_idx, k_proj, v_proj)?;

            // Get cached key-value pairs
            if let Some((cached_k, cached_v)) = cache.get_kv(cache_layer_idx) {
                return self.forward_causal(q, cached_k, cached_v);
            }
        }

        // Fallback to standard causal forward
        self.forward_causal(q, k, v)
    }
}
