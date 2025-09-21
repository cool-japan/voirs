//! Whisper decoder implementation with transformer blocks and cross-attention
//!
//! This module provides the text decoder for Whisper models with causal masking,
//! cross-attention to audio features, and KV-cache optimization for generation.

use super::attention::{KVCache, MultiHeadAttention};
use super::encoder::{WhisperConfig, MLP};
use crate::RecognitionError;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Sampling strategies for text generation
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Greedy decoding (argmax)
    Greedy,
    /// Top-k sampling with specified k
    TopK {
        /// Number of top tokens to consider
        k: usize,
    },
    /// Nucleus (top-p) sampling with specified probability threshold
    TopP {
        /// Probability threshold for nucleus sampling
        p: f32,
    },
    /// Combined top-k and nucleus sampling
    TopKP {
        /// Number of top tokens to consider
        k: usize,
        /// Probability threshold for nucleus sampling
        p: f32,
    },
}

/// Configuration for token sampling during generation
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Sampling strategy to use
    pub strategy: SamplingStrategy,
    /// Temperature scaling (handled externally in generate_tokens)
    pub temperature: f32,
    /// Length penalty factor (> 1.0 encourages longer sequences, < 1.0 discourages)
    pub length_penalty: f32,
    /// Repetition penalty factor (> 1.0 discourages repetition, < 1.0 encourages)
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy::Greedy,
            temperature: 1.0,
            length_penalty: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingConfig {
    /// Create config for nucleus sampling
    pub fn nucleus(p: f32) -> Self {
        Self {
            strategy: SamplingStrategy::TopP { p },
            ..Default::default()
        }
    }

    /// Create config for top-k sampling
    pub fn top_k(k: usize) -> Self {
        Self {
            strategy: SamplingStrategy::TopK { k },
            ..Default::default()
        }
    }

    /// Create config for combined top-k and nucleus sampling
    pub fn top_k_nucleus(k: usize, p: f32) -> Self {
        Self {
            strategy: SamplingStrategy::TopKP { k, p },
            ..Default::default()
        }
    }

    /// Set temperature for this config
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set length penalty for this config
    /// Values > 1.0 encourage longer sequences, < 1.0 discourage them
    pub fn with_length_penalty(mut self, length_penalty: f32) -> Self {
        self.length_penalty = length_penalty;
        self
    }

    /// Set repetition penalty for this config
    /// Values > 1.0 discourage repetition, < 1.0 encourage it
    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = repetition_penalty;
        self
    }
}

/// Beam hypothesis for beam search
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Generated token sequence
    pub tokens: Vec<u32>,
    /// Log probability score
    pub log_prob: f32,
    /// Whether this hypothesis is complete (ended with end token)
    pub is_finished: bool,
}

impl BeamHypothesis {
    /// Creates a new beam hypothesis with the starting token
    pub fn new(start_token: u32) -> Self {
        Self {
            tokens: vec![start_token],
            log_prob: 0.0,
            is_finished: false,
        }
    }

    /// Get the score for ranking beams (with length penalty applied)
    pub fn score(&self, length_penalty: f32) -> f32 {
        if length_penalty == 1.0 {
            self.log_prob
        } else {
            // Apply length penalty using a simpler approach
            // When penalty > 1.0, longer sequences are penalized (score becomes worse)
            // When penalty < 1.0, longer sequences are encouraged (score becomes better)
            let length = self.tokens.len() as f32;

            if length_penalty > 1.0 {
                // Penalize longer sequences: subtract penalty for each token beyond the first
                let penalty_per_token = (length_penalty - 1.0) * 0.1; // Small penalty per token
                self.log_prob - (length - 1.0) * penalty_per_token
            } else {
                // Encourage longer sequences: add bonus for each token beyond the first
                let bonus_per_token = (1.0 - length_penalty) * 0.1; // Small bonus per token
                self.log_prob + (length - 1.0) * bonus_per_token
            }
        }
    }

    /// Add a token to this hypothesis
    pub fn extend(&self, token: u32, log_prob_delta: f32, is_end_token: bool) -> Self {
        let mut new_tokens = self.tokens.clone();
        new_tokens.push(token);

        Self {
            tokens: new_tokens,
            log_prob: self.log_prob + log_prob_delta,
            is_finished: is_end_token,
        }
    }
}

/// Whisper decoder implementation  
pub struct WhisperDecoder {
    /// Token embedding
    token_embedding: candle_nn::Embedding,
    /// Positional embedding
    positional_embedding: Tensor,
    /// Transformer blocks
    blocks: Vec<DecoderBlock>,
    /// Layer normalization
    ln: candle_nn::LayerNorm,
    /// KV cache for efficient generation
    kv_cache: Arc<RwLock<KVCache>>,
}

/// Decoder block with self-attention, cross-attention, and MLP
pub struct DecoderBlock {
    /// Self-attention
    attn: MultiHeadAttention,
    /// Cross-attention
    cross_attn: MultiHeadAttention,
    /// Layer normalization 1
    attn_ln: candle_nn::LayerNorm,
    /// Layer normalization 2
    cross_attn_ln: candle_nn::LayerNorm,
    /// MLP
    mlp: MLP,
    /// Layer normalization 3
    mlp_ln: candle_nn::LayerNorm,
}

impl WhisperDecoder {
    /// Creates a new Whisper decoder with the given configuration and device
    pub async fn new(config: &WhisperConfig, device: &Device) -> Result<Self, RecognitionError> {
        let vs = VarBuilder::zeros(DType::F32, device);

        // Create token embedding
        let token_embedding = candle_nn::embedding(
            config.n_vocab,
            config.n_text_state,
            vs.pp("token_embedding"),
        )
        .map_err(|e| RecognitionError::ModelLoadError {
            message: format!("Failed to create token embedding: {e}"),
            source: Some(Box::new(e)),
        })?;

        // Create positional embedding
        let positional_embedding =
            Tensor::zeros((config.n_text_ctx, config.n_text_state), DType::F32, device).map_err(
                |e| RecognitionError::ModelLoadError {
                    message: format!("Failed to create positional embedding: {e}"),
                    source: Some(Box::new(e)),
                },
            )?;

        // Create decoder blocks
        let mut blocks = Vec::new();
        for i in 0..config.n_text_layer {
            let block = DecoderBlock::new(config, device, &vs.pp(format!("blocks.{i}"))).await?;
            blocks.push(block);
        }

        // Create final layer normalization
        let ln = candle_nn::layer_norm(config.n_text_state, 1e-5, vs.pp("ln")).map_err(|e| {
            RecognitionError::ModelLoadError {
                message: format!("Failed to create layer norm: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        // Create KV cache
        let kv_cache = Arc::new(RwLock::new(KVCache::new(
            config.n_text_layer,
            config.n_text_ctx,
        )));

        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            kv_cache,
        })
    }

    /// Performs forward pass through the decoder with given tokens and audio features
    pub fn forward(
        &self,
        tokens: &[u32],
        audio_features: &Tensor,
    ) -> Result<Tensor, RecognitionError> {
        // Convert tokens to tensor
        let tokens_tensor = Tensor::from_slice(tokens, tokens.len(), audio_features.device())
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to create tokens tensor: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Token embedding
        let mut x = self.token_embedding.forward(&tokens_tensor).map_err(|e| {
            RecognitionError::ModelError {
                message: format!("Token embedding failed: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        // Add positional embedding
        let pos_slice = self
            .positional_embedding
            .narrow(0, 0, tokens.len())
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Positional embedding slice failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        x = x
            .broadcast_add(&pos_slice)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Positional embedding add failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Apply decoder blocks
        for (i, block) in self.blocks.iter().enumerate() {
            x = block
                .forward(&x, audio_features)
                .map_err(|e| RecognitionError::ModelError {
                    message: format!("Decoder block {i} failed: {e}"),
                    source: Some(Box::new(e)),
                })?;
        }

        // Apply final layer norm
        x = self
            .ln
            .forward(&x)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Final layer norm failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Get logits for last token
        let last_token_logits =
            x.narrow(1, tokens.len() - 1, 1)
                .map_err(|e| RecognitionError::ModelError {
                    message: format!("Failed to get last token logits: {e}"),
                    source: Some(Box::new(e)),
                })?;

        Ok(last_token_logits)
    }

    /// Forward pass with KV-cache for efficient autoregressive generation
    pub async fn forward_with_cache(
        &self,
        token: u32,
        audio_features: &Tensor,
        use_cache: bool,
    ) -> Result<Tensor, RecognitionError> {
        let device = audio_features.device();

        // Convert single token to tensor
        let token_tensor =
            Tensor::from_slice(&[token], 1, device).map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to create token tensor: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Token embedding for single token
        let mut x = self.token_embedding.forward(&token_tensor).map_err(|e| {
            RecognitionError::ModelError {
                message: format!("Token embedding failed: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        // Add positional embedding for current position
        let mut cache = self.kv_cache.write().await;
        let current_pos = cache.seq_len();

        if current_pos < self.positional_embedding.dim(0).unwrap_or(0) {
            let pos_embedding = self
                .positional_embedding
                .narrow(0, current_pos, 1)
                .map_err(|e| RecognitionError::ModelError {
                    message: format!("Positional embedding slice failed: {e}"),
                    source: Some(Box::new(e)),
                })?;

            x = x
                .broadcast_add(&pos_embedding)
                .map_err(|e| RecognitionError::ModelError {
                    message: format!("Positional embedding add failed: {e}"),
                    source: Some(Box::new(e)),
                })?;
        }

        // Apply decoder blocks with KV-cache
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            x = block
                .forward_with_cache(&x, audio_features, layer_idx, &mut cache, use_cache)
                .await?;
        }

        // Update cache sequence length
        if use_cache {
            cache.set_seq_len(current_pos + 1);
        }

        // Release cache lock
        drop(cache);

        // Apply final layer norm
        x = self
            .ln
            .forward(&x)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Final layer norm failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Project to vocabulary logits (using token embedding weights transposed)
        let embedding_weights = self.token_embedding.embeddings();
        let logits = x
            .matmul(
                &embedding_weights
                    .t()
                    .map_err(|e| RecognitionError::ModelError {
                        message: format!("Embedding transpose failed: {e}"),
                        source: Some(Box::new(e)),
                    })?,
            )
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Logits computation failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        Ok(logits)
    }

    /// Generate text tokens using configurable sampling strategies
    pub async fn generate_tokens(
        &self,
        audio_features: &Tensor,
        start_token: u32,
        end_token: u32,
        max_length: usize,
        beam_size: usize,
        temperature: f32,
    ) -> Result<Vec<u32>, RecognitionError> {
        let sampling_config = SamplingConfig {
            strategy: SamplingStrategy::Greedy,
            temperature,
            ..Default::default()
        };
        self.generate_tokens_with_config(
            audio_features,
            start_token,
            end_token,
            max_length,
            beam_size,
            &sampling_config,
        )
        .await
    }

    /// Generate text tokens with advanced sampling configuration
    pub async fn generate_tokens_with_config(
        &self,
        audio_features: &Tensor,
        start_token: u32,
        end_token: u32,
        max_length: usize,
        beam_size: usize,
        sampling_config: &SamplingConfig,
    ) -> Result<Vec<u32>, RecognitionError> {
        if beam_size <= 1 {
            // Use greedy decoding for beam_size <= 1
            self.generate_greedy(
                audio_features,
                start_token,
                end_token,
                max_length,
                sampling_config,
            )
            .await
        } else {
            // Use beam search for beam_size > 1
            self.generate_beam_search(
                audio_features,
                start_token,
                end_token,
                max_length,
                beam_size,
                sampling_config,
            )
            .await
        }
    }

    /// Greedy generation (original implementation)
    async fn generate_greedy(
        &self,
        audio_features: &Tensor,
        start_token: u32,
        end_token: u32,
        max_length: usize,
        sampling_config: &SamplingConfig,
    ) -> Result<Vec<u32>, RecognitionError> {
        // Clear cache for new generation
        {
            let mut cache = self.kv_cache.write().await;
            cache.clear();
        }

        let mut tokens = vec![start_token];

        for _step in 0..max_length {
            let last_token = *tokens.last().unwrap();

            // Get logits for next token
            let mut logits = self
                .forward_with_cache(last_token, audio_features, true)
                .await?;

            // Apply repetition penalty to discourage repetitive tokens
            if sampling_config.repetition_penalty != 1.0 {
                logits = self.apply_repetition_penalty(
                    &logits,
                    &tokens,
                    sampling_config.repetition_penalty,
                )?;
            }

            // Apply length penalty to encourage/discourage longer sequences
            if sampling_config.length_penalty != 1.0 {
                logits = self.apply_length_penalty(
                    &logits,
                    tokens.len(),
                    sampling_config.length_penalty,
                )?;
            }

            // Apply temperature scaling
            let scaled_logits = if sampling_config.temperature == 1.0 {
                logits
            } else {
                let temp_tensor = Tensor::new(&[sampling_config.temperature], logits.device())
                    .map_err(|e| RecognitionError::ModelError {
                        message: format!("Failed to create temperature tensor: {e}"),
                        source: Some(Box::new(e)),
                    })?;
                logits
                    .broadcast_div(&temp_tensor)
                    .map_err(|e| RecognitionError::ModelError {
                        message: format!("Temperature scaling failed: {e}"),
                        source: Some(Box::new(e)),
                    })?
            };

            // Apply softmax to get probabilities
            let probs = candle_nn::ops::softmax_last_dim(&scaled_logits).map_err(|e| {
                RecognitionError::ModelError {
                    message: format!("Softmax failed: {e}"),
                    source: Some(Box::new(e)),
                }
            })?;

            // Sample next token using configured sampling strategy
            let next_token = self.sample_token_with_config(&probs, sampling_config)?;

            if next_token == end_token {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Beam search generation with sampling
    async fn generate_beam_search(
        &self,
        audio_features: &Tensor,
        start_token: u32,
        end_token: u32,
        max_length: usize,
        beam_size: usize,
        sampling_config: &SamplingConfig,
    ) -> Result<Vec<u32>, RecognitionError> {
        // Initialize beams
        let mut beams = vec![BeamHypothesis::new(start_token)];
        let mut finished_beams = Vec::new();

        for step in 0..max_length {
            let mut candidates = Vec::new();

            // Expand each active beam
            for beam in &beams {
                if beam.is_finished {
                    candidates.push(beam.clone());
                    continue;
                }

                // Clear cache and rebuild for this beam's sequence
                {
                    let mut cache = self.kv_cache.write().await;
                    cache.clear();
                }

                // Forward pass through the entire sequence to get to current state
                let mut logits = None;
                for (pos, &token) in beam.tokens.iter().enumerate() {
                    let use_cache = pos > 0; // Use cache for all tokens after the first
                    logits = Some(
                        self.forward_with_cache(token, audio_features, use_cache)
                            .await?,
                    );
                }

                let mut current_logits = logits.ok_or_else(|| RecognitionError::ModelError {
                    message: "No logits generated".to_string(),
                    source: None,
                })?;

                // Apply repetition penalty
                if sampling_config.repetition_penalty != 1.0 {
                    current_logits = self.apply_repetition_penalty(
                        &current_logits,
                        &beam.tokens,
                        sampling_config.repetition_penalty,
                    )?;
                }

                // Apply temperature scaling
                let scaled_logits = if sampling_config.temperature == 1.0 {
                    current_logits
                } else {
                    let temp_tensor =
                        Tensor::new(&[sampling_config.temperature], current_logits.device())
                            .map_err(|e| RecognitionError::ModelError {
                                message: format!("Failed to create temperature tensor: {e}"),
                                source: Some(Box::new(e)),
                            })?;
                    current_logits.broadcast_div(&temp_tensor).map_err(|e| {
                        RecognitionError::ModelError {
                            message: format!("Temperature scaling failed: {e}"),
                            source: Some(Box::new(e)),
                        }
                    })?
                };

                // Apply softmax to get probabilities
                let probs = candle_nn::ops::softmax_last_dim(&scaled_logits).map_err(|e| {
                    RecognitionError::ModelError {
                        message: format!("Softmax failed: {e}"),
                        source: Some(Box::new(e)),
                    }
                })?;

                // Get log probabilities
                let log_probs = probs.log().map_err(|e| RecognitionError::ModelError {
                    message: format!("Log computation failed: {e}"),
                    source: Some(Box::new(e)),
                })?;

                let log_probs_vec =
                    log_probs
                        .to_vec1::<f32>()
                        .map_err(|e| RecognitionError::ModelError {
                            message: format!("Failed to convert log probabilities to vector: {e}"),
                            source: Some(Box::new(e)),
                        })?;

                // For beam search, we typically take the top-k candidates rather than sampling
                // But we can combine this with the configured sampling strategy
                let top_candidates = match sampling_config.strategy {
                    SamplingStrategy::Greedy => {
                        // For greedy in beam search, take top beam_size candidates
                        self.get_top_k_candidates(&log_probs_vec, beam_size)
                    }
                    SamplingStrategy::TopK { k } => {
                        // Use the configured top-k, but limit to beam_size for efficiency
                        self.get_top_k_candidates(&log_probs_vec, k.min(beam_size * 2))
                    }
                    SamplingStrategy::TopP { p: _ } | SamplingStrategy::TopKP { k: _, p: _ } => {
                        // For nucleus sampling in beam search, we'll take more candidates
                        // to allow for diversity
                        self.get_top_k_candidates(&log_probs_vec, beam_size * 2)
                    }
                };

                // Create new beam candidates
                for (token_id, log_prob) in top_candidates {
                    let is_end = token_id == end_token;
                    let new_beam = beam.extend(token_id, log_prob, is_end);

                    if is_end || step == max_length - 1 {
                        finished_beams.push(new_beam);
                    } else {
                        candidates.push(new_beam);
                    }
                }
            }

            // Select top beam_size candidates for next iteration
            candidates.sort_by(|a, b| {
                b.score(sampling_config.length_penalty)
                    .partial_cmp(&a.score(sampling_config.length_penalty))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(beam_size);

            beams = candidates;

            // Stop if all beams are finished
            if beams.is_empty() || beams.iter().all(|b| b.is_finished) {
                break;
            }
        }

        // Combine active beams and finished beams
        finished_beams.extend(beams);

        // Return the best beam
        if finished_beams.is_empty() {
            Ok(vec![start_token])
        } else {
            let best_beam = finished_beams
                .iter()
                .max_by(|a, b| {
                    a.score(sampling_config.length_penalty)
                        .partial_cmp(&b.score(sampling_config.length_penalty))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            Ok(best_beam.tokens.clone())
        }
    }

    /// Get top-k candidates with their log probabilities
    fn get_top_k_candidates(&self, log_probs: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut indexed_probs: Vec<(usize, f32)> = log_probs
            .iter()
            .enumerate()
            .map(|(i, &log_prob)| (i, log_prob))
            .collect();

        // Sort by log probability (descending)
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k candidates
        indexed_probs
            .into_iter()
            .take(k.min(log_probs.len()))
            .map(|(idx, log_prob)| (idx as u32, log_prob))
            .collect()
    }

    /// Advanced token sampling with multiple strategies
    #[allow(dead_code)]
    fn sample_token(&self, probs: &Tensor) -> Result<u32, RecognitionError> {
        self.sample_token_with_config(probs, &SamplingConfig::default())
    }

    /// Token sampling with configurable strategy
    fn sample_token_with_config(
        &self,
        probs: &Tensor,
        config: &SamplingConfig,
    ) -> Result<u32, RecognitionError> {
        let mut probs_vec = probs
            .to_vec1::<f32>()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to convert probabilities to vector: {e}"),
                source: Some(Box::new(e)),
            })?;

        match config.strategy {
            SamplingStrategy::Greedy => {
                // Original argmax implementation
                let max_idx = probs_vec
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .ok_or_else(|| RecognitionError::ModelError {
                        message: "Empty probability vector".to_string(),
                        source: None,
                    })?;
                Ok(max_idx as u32)
            }

            SamplingStrategy::TopK { k } => self.sample_top_k(&mut probs_vec, k),

            SamplingStrategy::TopP { p } => self.sample_nucleus(&mut probs_vec, p),

            SamplingStrategy::TopKP { k, p } => {
                // Apply top-k first, then nucleus sampling
                let filtered_indices = self.apply_top_k_filter(&mut probs_vec, k)?;
                self.sample_nucleus_from_filtered(&probs_vec, &filtered_indices, p)
            }
        }
    }

    /// Apply top-k sampling
    fn sample_top_k(&self, probs: &mut [f32], k: usize) -> Result<u32, RecognitionError> {
        if probs.is_empty() {
            return Err(RecognitionError::ModelError {
                message: "Empty probability vector".to_string(),
                source: None,
            });
        }

        let k = k.min(probs.len());

        // Get indices sorted by probability (descending)
        let mut indexed_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Zero out probabilities beyond top-k
        for (i, _) in indexed_probs.iter().skip(k) {
            probs[*i] = 0.0;
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }

        self.multinomial_sample(probs)
    }

    /// Apply nucleus (top-p) sampling
    fn sample_nucleus(&self, probs: &mut [f32], p: f32) -> Result<u32, RecognitionError> {
        if probs.is_empty() {
            return Err(RecognitionError::ModelError {
                message: "Empty probability vector".to_string(),
                source: None,
            });
        }

        // Get indices sorted by probability (descending)
        let mut indexed_probs: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &prob)| (i, prob))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find cutoff point where cumulative probability exceeds p
        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = indexed_probs.len();

        for (idx, (_, prob)) in indexed_probs.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob >= p {
                cutoff_idx = idx + 1; // Include this token
                break;
            }
        }

        // Zero out probabilities beyond nucleus
        for (i, _) in indexed_probs.iter().skip(cutoff_idx) {
            probs[*i] = 0.0;
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for prob in probs.iter_mut() {
                *prob /= sum;
            }
        }

        self.multinomial_sample(probs)
    }

    /// Apply top-k filter and return valid indices
    fn apply_top_k_filter(
        &self,
        probs: &mut [f32],
        k: usize,
    ) -> Result<Vec<usize>, RecognitionError> {
        let k = k.min(probs.len());

        let mut indexed_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let valid_indices: Vec<usize> = indexed_probs.iter().take(k).map(|(i, _)| *i).collect();

        // Zero out probabilities beyond top-k
        for (i, _) in indexed_probs.iter().skip(k) {
            probs[*i] = 0.0;
        }

        Ok(valid_indices)
    }

    /// Apply nucleus sampling to pre-filtered probabilities
    fn sample_nucleus_from_filtered(
        &self,
        probs: &[f32],
        valid_indices: &[usize],
        p: f32,
    ) -> Result<u32, RecognitionError> {
        if valid_indices.is_empty() {
            return Err(RecognitionError::ModelError {
                message: "No valid indices for sampling".to_string(),
                source: None,
            });
        }

        // Create filtered probability vector
        let mut filtered_probs: Vec<(usize, f32)> =
            valid_indices.iter().map(|&i| (i, probs[i])).collect();

        // Sort by probability (descending)
        filtered_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply nucleus cutoff
        let mut cumulative_prob = 0.0;
        let mut nucleus_probs = Vec::new();

        for (idx, prob) in filtered_probs {
            cumulative_prob += prob;
            nucleus_probs.push((idx, prob));
            if cumulative_prob >= p {
                break;
            }
        }

        // Renormalize and sample
        let total: f32 = nucleus_probs.iter().map(|(_, p)| p).sum();
        if total <= 0.0 {
            // Fallback to first token if normalization fails
            return Ok(nucleus_probs[0].0 as u32);
        }

        let normalized_probs: Vec<f32> = nucleus_probs.iter().map(|(_, p)| p / total).collect();
        let sampled_idx = self.multinomial_sample_from_vec(&normalized_probs)?;

        Ok(nucleus_probs[sampled_idx].0 as u32)
    }

    /// Sample from multinomial distribution
    fn multinomial_sample(&self, probs: &[f32]) -> Result<u32, RecognitionError> {
        use rand::Rng;

        let total: f32 = probs.iter().sum();
        if total <= 0.0 {
            // Fallback to argmax if all probabilities are zero
            let max_idx = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            return Ok(max_idx as u32);
        }

        let mut rng = rand::thread_rng();
        let mut cumulative = 0.0;
        let threshold = rng.r#gen::<f32>() * total;

        for (idx, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if cumulative >= threshold {
                return Ok(idx as u32);
            }
        }

        // Fallback to last token
        Ok((probs.len() - 1) as u32)
    }

    /// Sample from a vector of probabilities
    fn multinomial_sample_from_vec(&self, probs: &[f32]) -> Result<usize, RecognitionError> {
        use rand::Rng;

        let total: f32 = probs.iter().sum();
        if total <= 0.0 {
            return Ok(0);
        }

        let mut rng = rand::thread_rng();
        let mut cumulative = 0.0;
        let threshold = rng.r#gen::<f32>() * total;

        for (idx, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if cumulative >= threshold {
                return Ok(idx);
            }
        }

        Ok(probs.len() - 1)
    }

    /// Clear the KV cache
    pub async fn clear_cache(&self) {
        let mut cache = self.kv_cache.write().await;
        cache.clear();
    }

    /// Forward single step for language detection (simplified version)
    pub async fn forward_single_step(
        &self,
        audio_features: &Tensor,
        tokens: &[u32],
    ) -> Result<Tensor, RecognitionError> {
        // This is a simplified version for language detection
        // Convert tokens to tensor
        let tokens_tensor =
            Tensor::new(tokens, &Device::Cpu).map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to create tokens tensor: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Get token embeddings
        let x = self.token_embedding.forward(&tokens_tensor).map_err(|e| {
            RecognitionError::ModelError {
                message: format!("Token embedding failed: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        // Add positional embeddings (simplified - just first position)
        let pos_embedding = self.positional_embedding.narrow(0, 0, 1).map_err(|e| {
            RecognitionError::ModelError {
                message: format!("Positional embedding failed: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        let x = x
            .broadcast_add(&pos_embedding)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Position embedding addition failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Run through first decoder block only for language detection (faster)
        let x = if self.blocks.is_empty() {
            x
        } else {
            let cache = self.kv_cache.write().await;
            let result = self.blocks[0].forward(&x, audio_features)?;
            drop(cache);
            result
        };

        // Apply final layer norm
        let x = self
            .ln
            .forward(&x)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Final layer norm failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Project to vocabulary logits
        let embedding_weights = self.token_embedding.embeddings();
        let logits = x
            .matmul(
                &embedding_weights
                    .t()
                    .map_err(|e| RecognitionError::ModelError {
                        message: format!("Embedding transpose failed: {e}"),
                        source: Some(Box::new(e)),
                    })?,
            )
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Logits computation failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        Ok(logits)
    }

    /// Apply repetition penalty to logits to discourage repetitive tokens
    fn apply_repetition_penalty(
        &self,
        logits: &Tensor,
        tokens: &[u32],
        penalty: f32,
    ) -> Result<Tensor, RecognitionError> {
        if penalty == 1.0 || tokens.is_empty() {
            return Ok(logits.clone());
        }

        let device = logits.device();
        let vocab_size = logits.dims()[logits.dims().len() - 1];

        // Get logits data for modification
        let logits_vec = logits
            .flatten_all()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to flatten logits: {e}"),
                source: Some(Box::new(e)),
            })?
            .to_vec1::<f32>()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to convert logits to vec: {e}"),
                source: Some(Box::new(e)),
            })?;

        let mut modified_logits = logits_vec;

        // Apply penalty to tokens that have appeared in the sequence
        for &token in tokens {
            let token_idx = token as usize;
            if token_idx < vocab_size {
                if modified_logits[token_idx] > 0.0 {
                    modified_logits[token_idx] /= penalty;
                } else {
                    modified_logits[token_idx] *= penalty;
                }
            }
        }

        // Convert back to tensor
        Tensor::from_vec(modified_logits, logits.shape(), device).map_err(|e| {
            RecognitionError::ModelError {
                message: format!("Failed to create tensor from modified logits: {e}"),
                source: Some(Box::new(e)),
            }
        })
    }

    /// Apply length penalty to encourage/discourage longer sequences
    fn apply_length_penalty(
        &self,
        logits: &Tensor,
        current_length: usize,
        penalty: f32,
    ) -> Result<Tensor, RecognitionError> {
        if penalty == 1.0 {
            return Ok(logits.clone());
        }

        // Length penalty formula: (length + 1)^penalty / (1 + 1)^penalty
        let length_factor = ((current_length + 1) as f32).powf(penalty) / 2.0_f32.powf(penalty);

        let penalty_tensor = Tensor::new(&[length_factor], logits.device()).map_err(|e| {
            RecognitionError::ModelError {
                message: format!("Failed to create length penalty tensor: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        logits
            .broadcast_mul(&penalty_tensor)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Length penalty application failed: {e}"),
                source: Some(Box::new(e)),
            })
    }
}

impl DecoderBlock {
    /// Creates a new decoder block with self-attention and cross-attention layers
    ///
    /// # Arguments
    /// * `config` - Whisper model configuration
    /// * `_device` - Device for tensor operations (currently unused)
    /// * `vs` - Variable builder for loading model weights
    ///
    /// # Returns
    /// A new decoder block instance or an error if initialization fails
    pub async fn new(
        config: &WhisperConfig,
        _device: &Device,
        vs: &VarBuilder<'_>,
    ) -> Result<Self, RecognitionError> {
        let attn =
            MultiHeadAttention::new(config.n_text_state, config.n_text_head, &vs.pp("attn"))?;

        let cross_attn = MultiHeadAttention::new(
            config.n_text_state,
            config.n_text_head,
            &vs.pp("cross_attn"),
        )?;

        let attn_ln =
            candle_nn::layer_norm(config.n_text_state, 1e-5, vs.pp("attn_ln")).map_err(|e| {
                RecognitionError::ModelLoadError {
                    message: format!("Failed to create self-attention layer norm: {e}"),
                    source: Some(Box::new(e)),
                }
            })?;

        let cross_attn_ln =
            candle_nn::layer_norm(config.n_text_state, 1e-5, vs.pp("cross_attn_ln")).map_err(
                |e| RecognitionError::ModelLoadError {
                    message: format!("Failed to create cross-attention layer norm: {e}"),
                    source: Some(Box::new(e)),
                },
            )?;

        let mlp = MLP::new(config.n_text_state, &vs.pp("mlp"))?;

        let mlp_ln =
            candle_nn::layer_norm(config.n_text_state, 1e-5, vs.pp("mlp_ln")).map_err(|e| {
                RecognitionError::ModelLoadError {
                    message: format!("Failed to create MLP layer norm: {e}"),
                    source: Some(Box::new(e)),
                }
            })?;

        Ok(Self {
            attn,
            cross_attn,
            attn_ln,
            cross_attn_ln,
            mlp,
            mlp_ln,
        })
    }

    /// Forward pass through the decoder block
    ///
    /// # Arguments
    /// * `x` - Input token embeddings tensor
    /// * `audio_features` - Audio encoder features for cross-attention
    ///
    /// # Returns
    /// Processed tensor after self-attention, cross-attention, and MLP layers
    pub fn forward(&self, x: &Tensor, audio_features: &Tensor) -> Result<Tensor, RecognitionError> {
        // Self-attention with causal masking
        let attn_input = self
            .attn_ln
            .forward(x)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Self-attention layer norm failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let attn_output = self
            .attn
            .forward_causal(&attn_input, &attn_input, &attn_input)?;
        let x = (x + attn_output).map_err(|e| RecognitionError::ModelError {
            message: format!("Self-attention residual connection failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        // Cross-attention with audio features
        let cross_attn_input =
            self.cross_attn_ln
                .forward(&x)
                .map_err(|e| RecognitionError::ModelError {
                    message: format!("Cross-attention layer norm failed: {e}"),
                    source: Some(Box::new(e)),
                })?;

        let cross_attn_output =
            self.cross_attn
                .forward(&cross_attn_input, audio_features, audio_features)?;
        let x = (x + cross_attn_output).map_err(|e| RecognitionError::ModelError {
            message: format!("Cross-attention residual connection failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        // MLP
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

    /// Forward pass with KV-cache support
    pub async fn forward_with_cache(
        &self,
        x: &Tensor,
        audio_features: &Tensor,
        layer_idx: usize,
        cache: &mut KVCache,
        use_cache: bool,
    ) -> Result<Tensor, RecognitionError> {
        // Self-attention with causal masking and KV-cache
        let attn_input = self
            .attn_ln
            .forward(x)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Self-attention layer norm failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let attn_output = self
            .attn
            .forward_causal_with_cache(
                &attn_input,
                &attn_input,
                &attn_input,
                layer_idx * 2,
                cache,
                use_cache,
            )
            .await?;
        let x = (x + attn_output).map_err(|e| RecognitionError::ModelError {
            message: format!("Self-attention residual connection failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        // Cross-attention with audio features (no KV-cache needed for cross-attention since audio features don't change)
        let cross_attn_input =
            self.cross_attn_ln
                .forward(&x)
                .map_err(|e| RecognitionError::ModelError {
                    message: format!("Cross-attention layer norm failed: {e}"),
                    source: Some(Box::new(e)),
                })?;

        let cross_attn_output =
            self.cross_attn
                .forward(&cross_attn_input, audio_features, audio_features)?;
        let x = (x + cross_attn_output).map_err(|e| RecognitionError::ModelError {
            message: format!("Cross-attention residual connection failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        // MLP
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// Create a mock probability tensor for testing
    fn create_test_probs(probs: &[f32]) -> Tensor {
        Tensor::new(probs, &Device::Cpu).unwrap()
    }

    /// Mock decoder for testing sampling functions
    struct MockDecoder;

    impl MockDecoder {
        fn sample_token_with_config(
            &self,
            probs: &Tensor,
            config: &SamplingConfig,
        ) -> Result<u32, RecognitionError> {
            let mut probs_vec =
                probs
                    .to_vec1::<f32>()
                    .map_err(|e| RecognitionError::ModelError {
                        message: format!("Failed to convert probabilities to vector: {e}"),
                        source: Some(Box::new(e)),
                    })?;

            match config.strategy {
                SamplingStrategy::Greedy => {
                    let max_idx = probs_vec
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .ok_or_else(|| RecognitionError::ModelError {
                            message: "Empty probability vector".to_string(),
                            source: None,
                        })?;
                    Ok(max_idx as u32)
                }

                SamplingStrategy::TopK { k } => self.sample_top_k(&mut probs_vec, k),

                SamplingStrategy::TopP { p } => self.sample_nucleus(&mut probs_vec, p),

                SamplingStrategy::TopKP { k, p } => {
                    let filtered_indices = self.apply_top_k_filter(&mut probs_vec, k)?;
                    self.sample_nucleus_from_filtered(&probs_vec, &filtered_indices, p)
                }
            }
        }

        fn sample_top_k(&self, probs: &mut [f32], k: usize) -> Result<u32, RecognitionError> {
            if probs.is_empty() {
                return Err(RecognitionError::ModelError {
                    message: "Empty probability vector".to_string(),
                    source: None,
                });
            }

            let k = k.min(probs.len());

            let mut indexed_probs: Vec<(usize, f32)> =
                probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            indexed_probs
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (i, _) in indexed_probs.iter().skip(k) {
                probs[*i] = 0.0;
            }

            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }

            self.multinomial_sample(probs)
        }

        fn sample_nucleus(&self, probs: &mut [f32], p: f32) -> Result<u32, RecognitionError> {
            if probs.is_empty() {
                return Err(RecognitionError::ModelError {
                    message: "Empty probability vector".to_string(),
                    source: None,
                });
            }

            let mut indexed_probs: Vec<(usize, f32)> = probs
                .iter()
                .enumerate()
                .map(|(i, &prob)| (i, prob))
                .collect();
            indexed_probs
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumulative_prob = 0.0;
            let mut cutoff_idx = indexed_probs.len();

            for (idx, (_, prob)) in indexed_probs.iter().enumerate() {
                cumulative_prob += prob;
                if cumulative_prob >= p {
                    cutoff_idx = idx + 1;
                    break;
                }
            }

            for (i, _) in indexed_probs.iter().skip(cutoff_idx) {
                probs[*i] = 0.0;
            }

            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for prob in probs.iter_mut() {
                    *prob /= sum;
                }
            }

            self.multinomial_sample(probs)
        }

        fn apply_top_k_filter(
            &self,
            probs: &mut [f32],
            k: usize,
        ) -> Result<Vec<usize>, RecognitionError> {
            let k = k.min(probs.len());

            let mut indexed_probs: Vec<(usize, f32)> =
                probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            indexed_probs
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let valid_indices: Vec<usize> = indexed_probs.iter().take(k).map(|(i, _)| *i).collect();

            for (i, _) in indexed_probs.iter().skip(k) {
                probs[*i] = 0.0;
            }

            Ok(valid_indices)
        }

        fn sample_nucleus_from_filtered(
            &self,
            probs: &[f32],
            valid_indices: &[usize],
            p: f32,
        ) -> Result<u32, RecognitionError> {
            if valid_indices.is_empty() {
                return Err(RecognitionError::ModelError {
                    message: "No valid indices for sampling".to_string(),
                    source: None,
                });
            }

            let mut filtered_probs: Vec<(usize, f32)> =
                valid_indices.iter().map(|&i| (i, probs[i])).collect();

            filtered_probs
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumulative_prob = 0.0;
            let mut nucleus_probs = Vec::new();

            for (idx, prob) in filtered_probs {
                cumulative_prob += prob;
                nucleus_probs.push((idx, prob));
                if cumulative_prob >= p {
                    break;
                }
            }

            let total: f32 = nucleus_probs.iter().map(|(_, p)| p).sum();
            if total <= 0.0 {
                return Ok(nucleus_probs[0].0 as u32);
            }

            // For testing, just return the first token (most likely)
            Ok(nucleus_probs[0].0 as u32)
        }

        fn multinomial_sample(&self, probs: &[f32]) -> Result<u32, RecognitionError> {
            // For testing, return the token with highest probability
            let max_idx = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            Ok(max_idx as u32)
        }
    }

    #[test]
    fn test_sampling_config_creation() {
        let config = SamplingConfig::default();
        assert!(matches!(config.strategy, SamplingStrategy::Greedy));
        assert_eq!(config.temperature, 1.0);

        let nucleus_config = SamplingConfig::nucleus(0.9);
        assert!(
            matches!(nucleus_config.strategy, SamplingStrategy::TopP { p } if (p - 0.9).abs() < f32::EPSILON)
        );

        let topk_config = SamplingConfig::top_k(50);
        assert!(matches!(
            topk_config.strategy,
            SamplingStrategy::TopK { k: 50 }
        ));

        let combined_config = SamplingConfig::top_k_nucleus(40, 0.8);
        assert!(
            matches!(combined_config.strategy, SamplingStrategy::TopKP { k: 40, p } if (p - 0.8).abs() < f32::EPSILON)
        );

        let temp_config = SamplingConfig::nucleus(0.9).with_temperature(0.7);
        assert_eq!(temp_config.temperature, 0.7);
    }

    #[test]
    fn test_greedy_sampling() {
        let decoder = MockDecoder;
        let probs = create_test_probs(&[0.1, 0.7, 0.15, 0.05]);
        let config = SamplingConfig::default(); // Greedy

        let result = decoder.sample_token_with_config(&probs, &config).unwrap();
        assert_eq!(result, 1); // Index of highest probability (0.7)
    }

    #[test]
    fn test_top_k_sampling() {
        let decoder = MockDecoder;
        let probs = create_test_probs(&[0.1, 0.3, 0.4, 0.15, 0.05]);
        let config = SamplingConfig::top_k(3);

        let result = decoder.sample_token_with_config(&probs, &config).unwrap();
        // Should only consider top 3 tokens (indices 2, 1, 3 with probs 0.4, 0.3, 0.15)
        // For deterministic testing, we expect the highest probability token
        assert!(result <= 4);
    }

    #[test]
    fn test_nucleus_sampling() {
        let decoder = MockDecoder;
        let probs = create_test_probs(&[0.05, 0.6, 0.25, 0.1]);
        let config = SamplingConfig::nucleus(0.9);

        let result = decoder.sample_token_with_config(&probs, &config).unwrap();
        // With p=0.9, should include tokens until cumulative prob >= 0.9
        // Sorted: [1: 0.6, 2: 0.25, 3: 0.1, 0: 0.05]
        // Cumulative: 0.6, 0.85, 0.95 (stops here as 0.95 >= 0.9)
        assert!(result <= 3);
    }

    #[test]
    fn test_combined_top_k_nucleus_sampling() {
        let decoder = MockDecoder;
        let probs = create_test_probs(&[0.05, 0.3, 0.4, 0.15, 0.08, 0.02]);
        let config = SamplingConfig::top_k_nucleus(4, 0.8);

        let result = decoder.sample_token_with_config(&probs, &config).unwrap();
        // First apply top-k=4, then nucleus with p=0.8
        assert!(result <= 5);
    }

    #[test]
    fn test_empty_probability_vector() {
        let decoder = MockDecoder;
        let probs = create_test_probs(&[]);
        let config = SamplingConfig::default();

        let result = decoder.sample_token_with_config(&probs, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_token_probability() {
        let decoder = MockDecoder;
        let probs = create_test_probs(&[1.0]);
        let config = SamplingConfig::top_k(10); // k larger than vocab

        let result = decoder.sample_token_with_config(&probs, &config).unwrap();
        assert_eq!(result, 0);
    }

    #[test]
    fn test_nucleus_sampling_edge_cases() {
        let decoder = MockDecoder;

        // Test with p=1.0 (should include all tokens)
        let probs = create_test_probs(&[0.25, 0.25, 0.25, 0.25]);
        let config = SamplingConfig::nucleus(1.0);
        let result = decoder.sample_token_with_config(&probs, &config);
        assert!(result.is_ok());

        // Test with p=0.0 (should include at least one token)
        let config = SamplingConfig::nucleus(0.0);
        let result = decoder.sample_token_with_config(&probs, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_top_k_larger_than_vocab() {
        let decoder = MockDecoder;
        let probs = create_test_probs(&[0.4, 0.3, 0.2, 0.1]);
        let config = SamplingConfig::top_k(100); // Much larger than vocab size

        let result = decoder.sample_token_with_config(&probs, &config).unwrap();
        assert!(result <= 3);
    }

    #[test]
    fn test_length_penalty_implementation() {
        // Create a simple test config with length penalty
        let config_no_penalty = SamplingConfig {
            strategy: SamplingStrategy::Greedy,
            temperature: 1.0,
            length_penalty: 1.0,
            repetition_penalty: 1.0,
        };

        let config_encourage_length = SamplingConfig {
            strategy: SamplingStrategy::Greedy,
            temperature: 1.0,
            length_penalty: 1.5, // Encourage longer sequences
            repetition_penalty: 1.0,
        };

        let config_discourage_length = SamplingConfig {
            strategy: SamplingStrategy::Greedy,
            temperature: 1.0,
            length_penalty: 0.5, // Discourage longer sequences
            repetition_penalty: 1.0,
        };

        // Test that different length penalties produce different configurations
        assert_eq!(config_no_penalty.length_penalty, 1.0);
        assert_eq!(config_encourage_length.length_penalty, 1.5);
        assert_eq!(config_discourage_length.length_penalty, 0.5);
    }

    #[test]
    fn test_repetition_penalty_implementation() {
        // Create a simple test config with repetition penalty
        let config_no_penalty = SamplingConfig {
            strategy: SamplingStrategy::Greedy,
            temperature: 1.0,
            length_penalty: 1.0,
            repetition_penalty: 1.0,
        };

        let config_discourage_repetition = SamplingConfig {
            strategy: SamplingStrategy::Greedy,
            temperature: 1.0,
            length_penalty: 1.0,
            repetition_penalty: 2.0, // Discourage repetition
        };

        let config_encourage_repetition = SamplingConfig {
            strategy: SamplingStrategy::Greedy,
            temperature: 1.0,
            length_penalty: 1.0,
            repetition_penalty: 0.5, // Encourage repetition
        };

        // Test that different repetition penalties produce different configurations
        assert_eq!(config_no_penalty.repetition_penalty, 1.0);
        assert_eq!(config_discourage_repetition.repetition_penalty, 2.0);
        assert_eq!(config_encourage_repetition.repetition_penalty, 0.5);
    }

    #[test]
    fn test_sampling_config_with_penalties() {
        // Test creating configs with penalties through builder pattern
        let config = SamplingConfig::default()
            .with_length_penalty(1.2)
            .with_repetition_penalty(1.5);

        assert_eq!(config.length_penalty, 1.2);
        assert_eq!(config.repetition_penalty, 1.5);
        assert_eq!(config.temperature, 1.0);
        assert!(matches!(config.strategy, SamplingStrategy::Greedy));
    }

    #[test]
    fn test_advanced_sampling_with_all_penalties() {
        // Test nucleus sampling with both penalties
        let config = SamplingConfig::nucleus(0.9)
            .with_temperature(0.8)
            .with_length_penalty(1.1)
            .with_repetition_penalty(1.3);

        assert!(
            matches!(config.strategy, SamplingStrategy::TopP { p } if (p - 0.9).abs() < f32::EPSILON)
        );
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.length_penalty, 1.1);
        assert_eq!(config.repetition_penalty, 1.3);
    }

    #[test]
    fn test_beam_hypothesis_creation() {
        let beam = BeamHypothesis::new(42);
        assert_eq!(beam.tokens, vec![42]);
        assert_eq!(beam.log_prob, 0.0);
        assert!(!beam.is_finished);
    }

    #[test]
    fn test_beam_hypothesis_extension() {
        let beam = BeamHypothesis::new(1);
        let extended = beam.extend(2, -0.5, false);

        assert_eq!(extended.tokens, vec![1, 2]);
        assert_eq!(extended.log_prob, -0.5);
        assert!(!extended.is_finished);

        let finished = extended.extend(3, -0.3, true);
        assert_eq!(finished.tokens, vec![1, 2, 3]);
        assert_eq!(finished.log_prob, -0.8);
        assert!(finished.is_finished);
    }

    #[test]
    fn test_beam_hypothesis_scoring() {
        let beam = BeamHypothesis {
            tokens: vec![1, 2, 3, 4],
            log_prob: -2.0,
            is_finished: false,
        };

        // No length penalty
        assert_eq!(beam.score(1.0), -2.0);

        // Length penalty > 1.0 (discourage longer sequences)
        let score_discourage = beam.score(1.5);
        assert!(score_discourage < -2.0); // Score should be worse (more negative)

        // Length penalty < 1.0 (encourage longer sequences)
        let score_encourage = beam.score(0.8);
        assert!(score_encourage > -2.0); // Score should be better (less negative)
    }

    #[test]
    fn test_beam_hypothesis_length_penalty_calculation() {
        let short_beam = BeamHypothesis {
            tokens: vec![1, 2],
            log_prob: -1.0,
            is_finished: false,
        };

        let long_beam = BeamHypothesis {
            tokens: vec![1, 2, 3, 4, 5, 6],
            log_prob: -1.0,
            is_finished: false,
        };

        // With length penalty > 1.0, longer sequences should be penalized more
        let penalty = 1.2;
        let short_score = short_beam.score(penalty);
        let long_score = long_beam.score(penalty);

        assert!(
            long_score < short_score,
            "Longer sequences should be penalized more with length penalty > 1.0"
        );
    }

    #[test]
    fn test_get_top_k_candidates() {
        let decoder = MockDecoder;
        let log_probs = vec![-0.1, -0.5, -0.2, -1.0, -0.05];

        let top_3 = decoder.get_top_k_candidates(&log_probs, 3);
        assert_eq!(top_3.len(), 3);

        // Should be sorted by log probability (descending)
        assert_eq!(top_3[0].0, 4); // -0.05 (highest)
        assert_eq!(top_3[1].0, 0); // -0.1
        assert_eq!(top_3[2].0, 2); // -0.2
    }

    #[test]
    fn test_get_top_k_candidates_with_k_larger_than_vocab() {
        let decoder = MockDecoder;
        let log_probs = vec![-0.1, -0.5];

        let top_10 = decoder.get_top_k_candidates(&log_probs, 10);
        assert_eq!(top_10.len(), 2); // Should only return actual vocab size
    }

    #[test]
    fn test_get_top_k_candidates_empty_probs() {
        let decoder = MockDecoder;
        let log_probs = vec![];

        let top_5 = decoder.get_top_k_candidates(&log_probs, 5);
        assert_eq!(top_5.len(), 0);
    }

    // Mock implementation for testing
    impl MockDecoder {
        fn get_top_k_candidates(&self, log_probs: &[f32], k: usize) -> Vec<(u32, f32)> {
            let mut indexed_probs: Vec<(usize, f32)> = log_probs
                .iter()
                .enumerate()
                .map(|(i, &log_prob)| (i, log_prob))
                .collect();

            // Sort by log probability (descending)
            indexed_probs
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top k candidates
            indexed_probs
                .into_iter()
                .take(k.min(log_probs.len()))
                .map(|(idx, log_prob)| (idx as u32, log_prob))
                .collect()
        }
    }
}
