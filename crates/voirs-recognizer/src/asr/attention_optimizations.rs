//! Advanced Attention Mechanism Optimizations
//!
//! This module implements state-of-the-art attention optimizations including
//! sparse attention, multi-scale attention, local attention, and memory-efficient
//! attention variants for improved performance and scalability.

use std::collections::HashMap;

/// Sparse attention pattern types
#[derive(Debug, Clone, PartialEq)]
/// Sparse Attention Pattern
pub enum SparseAttentionPattern {
    /// Local sliding window attention
    Local {
        /// Window size for attention
        window_size: usize,
    },
    /// Strided attention with dilated patterns
    Strided {
        /// Stride value
        stride: usize,
        /// Dilation factor
        dilation: usize,
    },
    /// Random sparse attention
    Random {
        /// Sparsity factor
        sparsity: f32,
    },
    /// Block-wise sparse attention
    BlockSparse {
        /// Block size
        block_size: usize,
    },
    /// Longformer-style global + local attention
    GlobalLocal {
        /// Window size
        window_size: usize,
        /// Global tokens
        global_tokens: usize,
    },
}

/// Optimized multi-head attention with advanced patterns
#[derive(Debug, Clone)]
/// Optimized Multi Head Attention
pub struct OptimizedMultiHeadAttention {
    /// Number of attention heads
    pub num_heads: usize,
    /// Model dimension
    pub model_dim: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Query weights
    pub w_q: Vec<Vec<f32>>,
    /// Key weights
    pub w_k: Vec<Vec<f32>>,
    /// Value weights
    pub w_v: Vec<Vec<f32>>,
    /// Output projection weights
    pub w_o: Vec<Vec<f32>>,
    /// Dropout rate
    pub dropout: f32,
    /// Sparse attention pattern
    pub sparse_pattern: Option<SparseAttentionPattern>,
    /// Key-value cache for inference
    pub kv_cache: Option<KVCache>,
    /// Gradient checkpointing enabled
    pub gradient_checkpointing: bool,
}

/// Key-Value cache for efficient inference
#[derive(Debug, Clone)]
/// K V Cache
pub struct KVCache {
    /// Cached keys per head
    pub keys: Vec<Vec<Vec<f32>>>,
    /// Cached values per head
    pub values: Vec<Vec<Vec<f32>>>,
    /// Cache capacity
    pub max_length: usize,
    /// Current cache length
    pub current_length: usize,
}

impl KVCache {
    /// Create new KV cache
    pub fn new(num_heads: usize, head_dim: usize, max_length: usize) -> Self {
        Self {
            keys: vec![Vec::new(); num_heads],
            values: vec![Vec::new(); num_heads],
            max_length,
            current_length: 0,
        }
    }

    /// Update cache with new key-value pairs
    pub fn update(&mut self, new_keys: &[Vec<Vec<f32>>], new_values: &[Vec<Vec<f32>>]) {
        for head in 0..self.keys.len() {
            if head < new_keys.len() && head < new_values.len() {
                // Append new keys and values
                self.keys[head].extend_from_slice(&new_keys[head]);
                self.values[head].extend_from_slice(&new_values[head]);

                // Evict old entries if cache is full
                if self.keys[head].len() > self.max_length {
                    let overflow = self.keys[head].len() - self.max_length;
                    self.keys[head].drain(0..overflow);
                    self.values[head].drain(0..overflow);
                }
            }
        }

        self.current_length = self.keys[0].len();
    }

    /// Get cached keys and values
    pub fn get(&self) -> (&[Vec<Vec<f32>>], &[Vec<Vec<f32>>]) {
        (&self.keys, &self.values)
    }
}

impl OptimizedMultiHeadAttention {
    /// Create optimized multi-head attention
    pub fn new(
        num_heads: usize,
        model_dim: usize,
        dropout: f32,
        sparse_pattern: Option<SparseAttentionPattern>,
        enable_kv_cache: bool,
        gradient_checkpointing: bool,
    ) -> Self {
        assert_eq!(
            model_dim % num_heads,
            0,
            "Model dimension must be divisible by number of heads"
        );

        let head_dim = model_dim / num_heads;
        let kv_cache = if enable_kv_cache {
            Some(KVCache::new(num_heads, head_dim, 2048))
        } else {
            None
        };

        Self {
            num_heads,
            model_dim,
            head_dim,
            w_q: Self::init_weights(model_dim, model_dim),
            w_k: Self::init_weights(model_dim, model_dim),
            w_v: Self::init_weights(model_dim, model_dim),
            w_o: Self::init_weights(model_dim, model_dim),
            dropout,
            sparse_pattern,
            kv_cache,
            gradient_checkpointing,
        }
    }

    /// Initialize weights with improved initialization
    fn init_weights(rows: usize, cols: usize) -> Vec<Vec<f32>> {
        let limit = (2.0 / (rows + cols) as f32).sqrt(); // He initialization
        let mut weights = vec![vec![0.0; cols]; rows];

        for row in &mut weights {
            for weight in row {
                *weight = (scirs2_core::random::random::<f32>() - 0.5) * 2.0 * limit;
            }
        }

        weights
    }

    /// Flash attention implementation for memory efficiency
    pub fn flash_attention(
        &self,
        query: &[Vec<f32>],
        key: &[Vec<f32>],
        value: &[Vec<f32>],
        mask: Option<&[Vec<bool>]>,
    ) -> Vec<Vec<f32>> {
        let seq_len = query.len();
        let block_size = 64.min(seq_len); // Tile size for flash attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        let mut output = vec![vec![0.0; self.head_dim]; seq_len];
        let mut row_maxes = vec![f32::NEG_INFINITY; seq_len];
        let mut row_sums = vec![0.0; seq_len];

        // Process in blocks for memory efficiency
        for i in (0..seq_len).step_by(block_size) {
            let i_end = (i + block_size).min(seq_len);

            for j in (0..seq_len).step_by(block_size) {
                let j_end = (j + block_size).min(seq_len);

                // Compute block attention scores
                let mut block_scores = vec![vec![f32::NEG_INFINITY; j_end - j]; i_end - i];

                for (q_idx, q_row) in query[i..i_end].iter().enumerate() {
                    for (k_idx, k_row) in key[j..j_end].iter().enumerate() {
                        if let Some(mask) = mask {
                            if mask[i + q_idx][j + k_idx] {
                                continue;
                            }
                        }

                        let mut score = 0.0;
                        for (&q_val, &k_val) in q_row.iter().zip(k_row.iter()) {
                            score += q_val * k_val;
                        }
                        block_scores[q_idx][k_idx] = score * scale;
                    }
                }

                // Update global statistics and compute block output
                for (q_idx, score_row) in block_scores.iter().enumerate() {
                    let global_q_idx = i + q_idx;

                    // Find max for numerical stability
                    let block_max = score_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let new_max = row_maxes[global_q_idx].max(block_max);

                    // Compute exponentials and sum
                    let mut block_sum = 0.0;
                    let mut block_weighted_values = vec![0.0; self.head_dim];

                    for (k_idx, &score) in score_row.iter().enumerate() {
                        if score > f32::NEG_INFINITY {
                            let prob = (score - new_max).exp();
                            block_sum += prob;

                            let global_k_idx = j + k_idx;
                            for (dim, &val) in value[global_k_idx].iter().enumerate() {
                                block_weighted_values[dim] += prob * val;
                            }
                        }
                    }

                    // Update global output with stability correction
                    let correction = (row_maxes[global_q_idx] - new_max).exp();
                    let new_sum = row_sums[global_q_idx] * correction + block_sum;

                    for dim in 0..self.head_dim {
                        output[global_q_idx][dim] =
                            (output[global_q_idx][dim] * row_sums[global_q_idx] * correction
                                + block_weighted_values[dim])
                                / new_sum;
                    }

                    row_maxes[global_q_idx] = new_max;
                    row_sums[global_q_idx] = new_sum;
                }
            }
        }

        output
    }

    /// Sparse attention computation
    pub fn sparse_attention(
        &self,
        query: &[Vec<f32>],
        key: &[Vec<f32>],
        value: &[Vec<f32>],
        pattern: &SparseAttentionPattern,
        mask: Option<&[Vec<bool>]>,
    ) -> Vec<Vec<f32>> {
        let seq_len = query.len();
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        match pattern {
            SparseAttentionPattern::Local { window_size } => {
                self.local_attention(query, key, value, *window_size, scale, mask)
            }
            SparseAttentionPattern::Strided { stride, dilation } => {
                self.strided_attention(query, key, value, *stride, *dilation, scale, mask)
            }
            SparseAttentionPattern::Random { sparsity } => {
                self.random_sparse_attention(query, key, value, *sparsity, scale, mask)
            }
            SparseAttentionPattern::BlockSparse { block_size } => {
                self.block_sparse_attention(query, key, value, *block_size, scale, mask)
            }
            SparseAttentionPattern::GlobalLocal {
                window_size,
                global_tokens,
            } => self.global_local_attention(
                query,
                key,
                value,
                *window_size,
                *global_tokens,
                scale,
                mask,
            ),
        }
    }

    /// Local sliding window attention
    fn local_attention(
        &self,
        query: &[Vec<f32>],
        key: &[Vec<f32>],
        value: &[Vec<f32>],
        window_size: usize,
        scale: f32,
        mask: Option<&[Vec<bool>]>,
    ) -> Vec<Vec<f32>> {
        let seq_len = query.len();
        let mut output = vec![vec![0.0; self.head_dim]; seq_len];

        for i in 0..seq_len {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(seq_len);

            let mut scores = Vec::new();
            let mut indices = Vec::new();

            for j in start..end {
                if let Some(mask) = mask {
                    if mask[i][j] {
                        continue;
                    }
                }

                let mut score = 0.0;
                for (&q_val, &k_val) in query[i].iter().zip(key[j].iter()) {
                    score += q_val * k_val;
                }
                scores.push(score * scale);
                indices.push(j);
            }

            // Apply softmax
            if !scores.is_empty() {
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();

                for (idx, &exp_score) in exp_scores.iter().enumerate() {
                    let prob = exp_score / sum_exp;
                    let j = indices[idx];

                    for (dim, &val) in value[j].iter().enumerate() {
                        output[i][dim] += prob * val;
                    }
                }
            }
        }

        output
    }

    /// Strided attention pattern
    fn strided_attention(
        &self,
        query: &[Vec<f32>],
        key: &[Vec<f32>],
        value: &[Vec<f32>],
        stride: usize,
        dilation: usize,
        scale: f32,
        mask: Option<&[Vec<bool>]>,
    ) -> Vec<Vec<f32>> {
        let seq_len = query.len();
        let mut output = vec![vec![0.0; self.head_dim]; seq_len];

        for i in 0..seq_len {
            let mut scores = Vec::new();
            let mut indices = Vec::new();

            // Attend to strided positions
            let mut j = i % stride;
            while j < seq_len {
                if let Some(mask) = mask {
                    if mask[i][j] {
                        j += stride * dilation;
                        continue;
                    }
                }

                let mut score = 0.0;
                for (&q_val, &k_val) in query[i].iter().zip(key[j].iter()) {
                    score += q_val * k_val;
                }
                scores.push(score * scale);
                indices.push(j);

                j += stride * dilation;
            }

            // Apply softmax
            if !scores.is_empty() {
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();

                for (idx, &exp_score) in exp_scores.iter().enumerate() {
                    let prob = exp_score / sum_exp;
                    let j = indices[idx];

                    for (dim, &val) in value[j].iter().enumerate() {
                        output[i][dim] += prob * val;
                    }
                }
            }
        }

        output
    }

    /// Random sparse attention
    fn random_sparse_attention(
        &self,
        query: &[Vec<f32>],
        key: &[Vec<f32>],
        value: &[Vec<f32>],
        sparsity: f32,
        scale: f32,
        mask: Option<&[Vec<bool>]>,
    ) -> Vec<Vec<f32>> {
        let seq_len = query.len();
        let mut output = vec![vec![0.0; self.head_dim]; seq_len];
        let keep_prob = 1.0 - sparsity;

        for i in 0..seq_len {
            let mut scores = Vec::new();
            let mut indices = Vec::new();

            for j in 0..seq_len {
                // Random sampling
                if scirs2_core::random::random::<f32>() > keep_prob {
                    continue;
                }

                if let Some(mask) = mask {
                    if mask[i][j] {
                        continue;
                    }
                }

                let mut score = 0.0;
                for (&q_val, &k_val) in query[i].iter().zip(key[j].iter()) {
                    score += q_val * k_val;
                }
                scores.push(score * scale);
                indices.push(j);
            }

            // Apply softmax
            if !scores.is_empty() {
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();

                for (idx, &exp_score) in exp_scores.iter().enumerate() {
                    let prob = exp_score / sum_exp / keep_prob; // Correct for sampling
                    let j = indices[idx];

                    for (dim, &val) in value[j].iter().enumerate() {
                        output[i][dim] += prob * val;
                    }
                }
            }
        }

        output
    }

    /// Block sparse attention
    fn block_sparse_attention(
        &self,
        query: &[Vec<f32>],
        key: &[Vec<f32>],
        value: &[Vec<f32>],
        block_size: usize,
        scale: f32,
        mask: Option<&[Vec<bool>]>,
    ) -> Vec<Vec<f32>> {
        let seq_len = query.len();
        let mut output = vec![vec![0.0; self.head_dim]; seq_len];

        for block_i in (0..seq_len).step_by(block_size) {
            let block_i_end = (block_i + block_size).min(seq_len);

            for block_j in (0..seq_len).step_by(block_size) {
                let block_j_end = (block_j + block_size).min(seq_len);

                // Compute block attention
                for i in block_i..block_i_end {
                    let mut scores = Vec::new();
                    let mut indices = Vec::new();

                    for j in block_j..block_j_end {
                        if let Some(mask) = mask {
                            if mask[i][j] {
                                continue;
                            }
                        }

                        let mut score = 0.0;
                        for (&q_val, &k_val) in query[i].iter().zip(key[j].iter()) {
                            score += q_val * k_val;
                        }
                        scores.push(score * scale);
                        indices.push(j);
                    }

                    // Apply softmax within block
                    if !scores.is_empty() {
                        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let exp_scores: Vec<f32> =
                            scores.iter().map(|&s| (s - max_score).exp()).collect();
                        let sum_exp: f32 = exp_scores.iter().sum();

                        for (idx, &exp_score) in exp_scores.iter().enumerate() {
                            let prob = exp_score / sum_exp;
                            let j = indices[idx];

                            for (dim, &val) in value[j].iter().enumerate() {
                                output[i][dim] += prob * val;
                            }
                        }
                    }
                }
            }
        }

        output
    }

    /// Global + local attention (Longformer-style)
    fn global_local_attention(
        &self,
        query: &[Vec<f32>],
        key: &[Vec<f32>],
        value: &[Vec<f32>],
        window_size: usize,
        global_tokens: usize,
        scale: f32,
        mask: Option<&[Vec<bool>]>,
    ) -> Vec<Vec<f32>> {
        let seq_len = query.len();
        let mut output = vec![vec![0.0; self.head_dim]; seq_len];

        for i in 0..seq_len {
            let mut scores = Vec::new();
            let mut indices = Vec::new();

            // Global tokens (always attend to first few tokens)
            for j in 0..global_tokens.min(seq_len) {
                if let Some(mask) = mask {
                    if mask[i][j] {
                        continue;
                    }
                }

                let mut score = 0.0;
                for (&q_val, &k_val) in query[i].iter().zip(key[j].iter()) {
                    score += q_val * k_val;
                }
                scores.push(score * scale);
                indices.push(j);
            }

            // Local window
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(seq_len);

            for j in start..end {
                if j < global_tokens {
                    continue; // Already added as global token
                }

                if let Some(mask) = mask {
                    if mask[i][j] {
                        continue;
                    }
                }

                let mut score = 0.0;
                for (&q_val, &k_val) in query[i].iter().zip(key[j].iter()) {
                    score += q_val * k_val;
                }
                scores.push(score * scale);
                indices.push(j);
            }

            // Apply softmax
            if !scores.is_empty() {
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();

                for (idx, &exp_score) in exp_scores.iter().enumerate() {
                    let prob = exp_score / sum_exp;
                    let j = indices[idx];

                    for (dim, &val) in value[j].iter().enumerate() {
                        output[i][dim] += prob * val;
                    }
                }
            }
        }

        output
    }

    /// Main forward pass with optimizations
    pub fn forward(
        &mut self,
        query: &[Vec<f32>],
        key: &[Vec<f32>],
        value: &[Vec<f32>],
        mask: Option<&[Vec<bool>]>,
        use_cache: bool,
    ) -> Vec<Vec<f32>> {
        // Linear projections
        let q = self.linear_transform(query, &self.w_q.clone());
        let k = self.linear_transform(key, &self.w_k.clone());
        let v = self.linear_transform(value, &self.w_v.clone());

        // Update cache if enabled
        if use_cache && self.kv_cache.is_some() {
            let k_heads = self.reshape_for_heads(&k);
            let v_heads = self.reshape_for_heads(&v);

            if let Some(cache) = &mut self.kv_cache {
                cache.update(&k_heads, &v_heads);
            }
        }

        // Reshape for heads
        let q_heads = self.reshape_for_heads(&q);
        let k_heads = self.reshape_for_heads(&k);
        let v_heads = self.reshape_for_heads(&v);

        // Get keys and values (from cache or current)
        let (final_k_heads, final_v_heads) = if use_cache && self.kv_cache.is_some() {
            let (cached_k, cached_v) = self.kv_cache.as_ref().unwrap().get();
            (cached_k.to_vec(), cached_v.to_vec())
        } else {
            (k_heads, v_heads)
        };

        // Apply attention (sparse or dense)
        let mut head_outputs = Vec::new();
        for head in 0..self.num_heads {
            let attention_output = if let Some(pattern) = &self.sparse_pattern {
                // Use sparse attention
                self.sparse_attention(
                    &q_heads[head],
                    &final_k_heads[head],
                    &final_v_heads[head],
                    pattern,
                    mask,
                )
            } else {
                // Use flash attention for efficiency
                self.flash_attention(
                    &q_heads[head],
                    &final_k_heads[head],
                    &final_v_heads[head],
                    mask,
                )
            };
            head_outputs.push(attention_output);
        }

        // Concatenate heads
        let concatenated = self.concatenate_heads(&head_outputs);

        // Final linear projection
        self.linear_transform(&concatenated, &self.w_o.clone())
    }

    /// Linear transformation helper
    fn linear_transform(&self, input: &[Vec<f32>], weights: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut output = vec![vec![0.0; weights[0].len()]; input.len()];

        for (i, input_row) in input.iter().enumerate() {
            for (j, weight_col) in weights[0].iter().enumerate() {
                for (k, &input_val) in input_row.iter().enumerate() {
                    if k < weights.len() {
                        output[i][j] += input_val * weights[k][j];
                    }
                }
            }
        }

        output
    }

    /// Reshape for multi-head processing
    fn reshape_for_heads(&self, input: &[Vec<f32>]) -> Vec<Vec<Vec<f32>>> {
        let seq_len = input.len();
        let mut heads = vec![vec![vec![0.0; self.head_dim]; seq_len]; self.num_heads];

        for (i, row) in input.iter().enumerate() {
            for head in 0..self.num_heads {
                let start_idx = head * self.head_dim;
                let end_idx = start_idx + self.head_dim;
                if end_idx <= row.len() {
                    heads[head][i] = row[start_idx..end_idx].to_vec();
                }
            }
        }

        heads
    }

    /// Concatenate multi-head outputs
    fn concatenate_heads(&self, heads: &[Vec<Vec<f32>>]) -> Vec<Vec<f32>> {
        let seq_len = heads[0].len();
        let mut output = vec![vec![0.0; self.model_dim]; seq_len];

        for i in 0..seq_len {
            for (head_idx, head) in heads.iter().enumerate() {
                let start_idx = head_idx * self.head_dim;
                for j in 0..self.head_dim {
                    if start_idx + j < self.model_dim {
                        output[i][start_idx + j] = head[i][j];
                    }
                }
            }
        }

        output
    }
}

/// Multi-scale attention for hierarchical processing
#[derive(Debug, Clone)]
/// Multi Scale Attention
pub struct MultiScaleAttention {
    /// Attention layers at different scales
    pub scales: Vec<OptimizedMultiHeadAttention>,
    /// Scale weights for combining outputs
    pub scale_weights: Vec<f32>,
    /// Pooling sizes for each scale
    pub pooling_sizes: Vec<usize>,
}

impl MultiScaleAttention {
    /// Create multi-scale attention
    pub fn new(num_heads: usize, model_dim: usize, dropout: f32, scales: Vec<usize>) -> Self {
        let mut attention_scales = Vec::new();
        let mut pooling_sizes = Vec::new();

        for &scale in &scales {
            let pattern = Some(SparseAttentionPattern::Local { window_size: scale });
            let attention = OptimizedMultiHeadAttention::new(
                num_heads, model_dim, dropout, pattern, false, false,
            );
            attention_scales.push(attention);
            pooling_sizes.push(scale);
        }

        let scale_weights = vec![1.0 / scales.len() as f32; scales.len()];

        Self {
            scales: attention_scales,
            scale_weights,
            pooling_sizes,
        }
    }

    /// Forward pass with multi-scale processing
    pub fn forward(
        &mut self,
        query: &[Vec<f32>],
        key: &[Vec<f32>],
        value: &[Vec<f32>],
        mask: Option<&[Vec<bool>]>,
    ) -> Vec<Vec<f32>> {
        let mut outputs = Vec::new();

        // Process at each scale
        for (scale_idx, attention) in self.scales.iter_mut().enumerate() {
            let scale_output = attention.forward(query, key, value, mask, false);
            outputs.push(scale_output);
        }

        // Combine outputs with learned weights
        let seq_len = query.len();
        let model_dim = query[0].len();
        let mut combined = vec![vec![0.0; model_dim]; seq_len];

        for (scale_idx, output) in outputs.iter().enumerate() {
            let weight = self.scale_weights[scale_idx];
            for i in 0..seq_len {
                for j in 0..model_dim {
                    combined[i][j] += weight * output[i][j];
                }
            }
        }

        combined
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimized_attention_creation() {
        let attention = OptimizedMultiHeadAttention::new(
            8,
            512,
            0.1,
            Some(SparseAttentionPattern::Local { window_size: 64 }),
            true,
            false,
        );
        assert_eq!(attention.num_heads, 8);
        assert_eq!(attention.model_dim, 512);
        assert!(attention.kv_cache.is_some());
    }

    #[test]
    fn test_kv_cache() {
        let mut cache = KVCache::new(8, 64, 100);

        let keys = vec![vec![vec![1.0; 64]; 10]; 8];
        let values = vec![vec![vec![2.0; 64]; 10]; 8];

        cache.update(&keys, &values);
        assert_eq!(cache.current_length, 10);

        let (cached_k, cached_v) = cache.get();
        assert_eq!(cached_k.len(), 8);
        assert_eq!(cached_v.len(), 8);
    }

    #[test]
    fn test_sparse_patterns() {
        let patterns = vec![
            SparseAttentionPattern::Local { window_size: 32 },
            SparseAttentionPattern::Strided {
                stride: 4,
                dilation: 2,
            },
            SparseAttentionPattern::Random { sparsity: 0.9 },
            SparseAttentionPattern::BlockSparse { block_size: 16 },
            SparseAttentionPattern::GlobalLocal {
                window_size: 32,
                global_tokens: 4,
            },
        ];

        for pattern in patterns {
            let attention =
                OptimizedMultiHeadAttention::new(4, 256, 0.1, Some(pattern), false, false);
            assert!(attention.sparse_pattern.is_some());
        }
    }

    #[test]
    fn test_multi_scale_attention() {
        let scales = vec![16, 32, 64];
        let mut multi_scale = MultiScaleAttention::new(8, 512, 0.1, scales);

        let seq_len = 128;
        let input = vec![vec![1.0; 512]; seq_len];

        let output = multi_scale.forward(&input, &input, &input, None);
        assert_eq!(output.len(), seq_len);
        assert_eq!(output[0].len(), 512);
    }

    #[test]
    fn test_flash_attention() {
        let attention = OptimizedMultiHeadAttention::new(8, 512, 0.1, None, false, false);

        let seq_len = 64;
        let query = vec![vec![1.0; 64]; seq_len];
        let key = vec![vec![2.0; 64]; seq_len];
        let value = vec![vec![3.0; 64]; seq_len];

        let output = attention.flash_attention(&query, &key, &value, None);
        assert_eq!(output.len(), seq_len);
        assert_eq!(output[0].len(), 64);
    }
}
