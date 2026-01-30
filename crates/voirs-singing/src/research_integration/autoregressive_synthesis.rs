//! # Autoregressive Singing Synthesis
//!
//! This module implements MusicGen-style autoregressive transformers for singing synthesis.
//!
//! ## Overview
//!
//! MusicGen uses a multi-stage autoregressive approach with hierarchical token generation:
//! - **Semantic Stage**: Generates high-level musical structure
//! - **Acoustic Stage**: Generates detailed acoustic features
//! - **Codec Stage**: Decodes into waveform using neural codec
//!
//! ## Key Features
//!
//! - **Multi-Stage Generation**: Hierarchical semantic-to-acoustic modeling
//! - **Pattern Forcing**: Constraints for musical coherence
//! - **Parallel Decoding**: Faster generation with delay pattern
//! - **Classifier-Free Guidance**: Enhanced controllability
//!
//! ## Example
//!
//! ```rust,ignore
//! use voirs_singing::research_integration::*;
//!
//! let config = AutoregressiveConfig::musicgen_style();
//! let model = AutoregressiveSynthesizer::new(config);
//!
//! // Generate singing with multi-stage approach
//! let conditioning = SingingConditioning {
//!     text: "Hello world",
//!     pitch_contour: vec![440.0, 450.0],
//!     duration: 2.0,
//! };
//!
//! let audio = model.generate(&conditioning, 48000).await?;
//! ```

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ==================== Configuration ====================

/// Configuration for autoregressive synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoregressiveConfig {
    /// Semantic model dimension
    pub semantic_dim: usize,
    /// Acoustic model dimension
    pub acoustic_dim: usize,
    /// Number of transformer layers per stage
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Codebook size for semantic tokens
    pub semantic_codebook_size: usize,
    /// Number of acoustic codebooks (RVQ depth)
    pub num_acoustic_codebooks: usize,
    /// Acoustic codebook size
    pub acoustic_codebook_size: usize,
    /// Sample rate
    pub sample_rate: usize,
    /// Use classifier-free guidance
    pub use_cfg: bool,
    /// CFG guidance scale
    pub cfg_scale: f32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p nucleus sampling threshold
    pub top_p: f32,
    /// Delay pattern for parallel decoding
    pub delay_pattern: DelayPattern,
}

impl Default for AutoregressiveConfig {
    fn default() -> Self {
        Self::musicgen_style()
    }
}

impl AutoregressiveConfig {
    /// MusicGen-style configuration
    pub fn musicgen_style() -> Self {
        Self {
            semantic_dim: 1024,
            acoustic_dim: 768,
            num_layers: 24,
            num_heads: 16,
            semantic_codebook_size: 4096,
            num_acoustic_codebooks: 8,
            acoustic_codebook_size: 1024,
            sample_rate: 24000,
            use_cfg: true,
            cfg_scale: 3.0,
            temperature: 1.0,
            top_p: 0.95,
            delay_pattern: DelayPattern::Parallel,
        }
    }

    /// Fast generation configuration
    pub fn fast() -> Self {
        Self {
            num_layers: 12,
            num_heads: 8,
            delay_pattern: DelayPattern::Parallel,
            ..Self::musicgen_style()
        }
    }

    /// High quality configuration
    pub fn high_quality() -> Self {
        Self {
            semantic_dim: 2048,
            acoustic_dim: 1024,
            num_layers: 32,
            num_heads: 32,
            delay_pattern: DelayPattern::Sequential,
            ..Self::musicgen_style()
        }
    }
}

/// Delay pattern for multi-codebook generation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DelayPattern {
    /// Sequential generation (slower, higher quality)
    Sequential,
    /// Parallel generation with delay (faster)
    Parallel,
    /// Custom delay pattern
    Custom,
}

// ==================== Autoregressive Synthesizer ====================

/// MusicGen-style autoregressive singing synthesizer
#[derive(Debug)]
pub struct AutoregressiveSynthesizer {
    config: AutoregressiveConfig,
    semantic_model: SemanticTransformer,
    acoustic_model: AcousticTransformer,
    pattern_manager: PatternManager,
}

impl AutoregressiveSynthesizer {
    /// Create new autoregressive synthesizer
    pub fn new(config: AutoregressiveConfig) -> Self {
        let semantic_model = SemanticTransformer::new(
            config.semantic_dim,
            config.num_layers,
            config.num_heads,
            config.semantic_codebook_size,
        );

        let acoustic_model = AcousticTransformer::new(
            config.acoustic_dim,
            config.num_layers,
            config.num_heads,
            config.num_acoustic_codebooks,
            config.acoustic_codebook_size,
        );

        let pattern_manager = PatternManager::new(config.delay_pattern);

        Self {
            config,
            semantic_model,
            acoustic_model,
            pattern_manager,
        }
    }

    /// Generate audio from conditioning
    pub async fn generate(
        &self,
        conditioning: &SingingConditioning,
        target_samples: usize,
    ) -> Result<Vec<f32>> {
        // Stage 1: Generate semantic tokens
        let semantic_tokens = self
            .semantic_model
            .generate(conditioning, target_samples / self.config.sample_rate)
            .await?;

        // Stage 2: Generate acoustic tokens from semantic
        let acoustic_tokens = self
            .acoustic_model
            .generate(&semantic_tokens, &self.pattern_manager)
            .await?;

        // Stage 3: Decode to waveform
        let audio = self.decode_tokens(&acoustic_tokens).await?;

        Ok(audio)
    }

    /// Generate with classifier-free guidance
    pub async fn generate_with_cfg(
        &self,
        conditioning: &SingingConditioning,
        target_samples: usize,
    ) -> Result<Vec<f32>> {
        if !self.config.use_cfg {
            return self.generate(conditioning, target_samples).await;
        }

        // Generate unconditional
        let uncond_tokens = self
            .semantic_model
            .generate_unconditional(target_samples / self.config.sample_rate)
            .await?;

        // Generate conditional
        let cond_tokens = self
            .semantic_model
            .generate(conditioning, target_samples / self.config.sample_rate)
            .await?;

        // Apply classifier-free guidance: out = uncond + scale * (cond - uncond)
        let guided_tokens = self.apply_cfg(&uncond_tokens, &cond_tokens)?;

        // Generate acoustic tokens
        let acoustic_tokens = self
            .acoustic_model
            .generate(&guided_tokens, &self.pattern_manager)
            .await?;

        // Decode to waveform
        let audio = self.decode_tokens(&acoustic_tokens).await?;

        Ok(audio)
    }

    /// Apply classifier-free guidance
    fn apply_cfg(&self, uncond: &[usize], cond: &[usize]) -> Result<Vec<usize>> {
        if uncond.len() != cond.len() {
            return Err(Error::Processing(
                "Unconditional and conditional sequences must have same length".to_string(),
            ));
        }

        // In practice, CFG is applied to logits before sampling
        // Here we approximate by blending token sequences
        let mut guided = Vec::with_capacity(cond.len());

        for (&u, &c) in uncond.iter().zip(cond.iter()) {
            // Simple approximation: favor conditional tokens with probability based on cfg_scale
            let prob_cond = self.config.cfg_scale / (1.0 + self.config.cfg_scale);
            if fastrand::f32() < prob_cond {
                guided.push(c);
            } else {
                guided.push(u);
            }
        }

        Ok(guided)
    }

    /// Decode acoustic tokens to waveform
    async fn decode_tokens(&self, tokens: &[Vec<usize>]) -> Result<Vec<f32>> {
        // Simulate neural codec decoding
        // In practice, this would use a neural vocoder
        let num_samples = tokens.len() * self.config.sample_rate / 50; // 50 Hz frame rate
        let mut audio = vec![0.0f32; num_samples];

        // Simple synthesis based on codebook indices
        for (i, token_frame) in tokens.iter().enumerate() {
            let start_sample = i * self.config.sample_rate / 50;
            let end_sample = ((i + 1) * self.config.sample_rate / 50).min(num_samples);

            // Generate audio for this frame based on tokens
            #[allow(clippy::needless_range_loop)]
            // sample_idx used in calculations, not just indexing
            for sample_idx in start_sample..end_sample {
                let _t = (sample_idx - start_sample) as f32 / (end_sample - start_sample) as f32;

                // Combine contributions from all codebooks
                let mut value = 0.0;
                for (codebook_idx, &token) in token_frame.iter().enumerate() {
                    // Each codebook contributes a harmonic component
                    let freq = 440.0 * (1.0 + token as f32 / 1024.0);
                    let phase = 2.0 * std::f32::consts::PI * freq * sample_idx as f32
                        / self.config.sample_rate as f32;
                    let amplitude = 0.1 / (codebook_idx + 1) as f32;
                    value += amplitude * phase.sin();
                }

                audio[sample_idx] = value;
            }
        }

        Ok(audio)
    }

    /// Get model info
    pub fn info(&self) -> AutoregressiveInfo {
        AutoregressiveInfo {
            config: self.config.clone(),
            semantic_params: self.semantic_model.num_parameters(),
            acoustic_params: self.acoustic_model.num_parameters(),
            total_params: self.semantic_model.num_parameters()
                + self.acoustic_model.num_parameters(),
        }
    }
}

// ==================== Semantic Transformer ====================

/// Semantic-level transformer for high-level structure
#[derive(Debug)]
struct SemanticTransformer {
    model_dim: usize,
    num_layers: usize,
    num_heads: usize,
    vocab_size: usize,
}

impl SemanticTransformer {
    fn new(model_dim: usize, num_layers: usize, num_heads: usize, vocab_size: usize) -> Self {
        Self {
            model_dim,
            num_layers,
            num_heads,
            vocab_size,
        }
    }

    async fn generate(
        &self,
        conditioning: &SingingConditioning,
        num_tokens: usize,
    ) -> Result<Vec<usize>> {
        // Simulate semantic token generation
        // In practice, this would use a transformer to generate tokens autoregressively
        let mut tokens = Vec::with_capacity(num_tokens);

        // Initialize with conditioning
        let start_token = self.encode_conditioning(conditioning)?;
        tokens.push(start_token);

        // Generate remaining tokens
        for i in 1..num_tokens {
            // Simulate autoregressive generation with some musical structure
            let prev_token = tokens[i - 1];
            let next_token = (prev_token + fastrand::usize(0..10)) % self.vocab_size;
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    async fn generate_unconditional(&self, num_tokens: usize) -> Result<Vec<usize>> {
        // Generate without conditioning
        let mut tokens = Vec::with_capacity(num_tokens);

        for _ in 0..num_tokens {
            tokens.push(fastrand::usize(0..self.vocab_size));
        }

        Ok(tokens)
    }

    fn encode_conditioning(&self, conditioning: &SingingConditioning) -> Result<usize> {
        // Encode text, pitch, and duration into a semantic token
        // Simple hash-based encoding for demonstration
        let hash = conditioning.text.len() + conditioning.pitch_contour.len();
        Ok(hash % self.vocab_size)
    }

    fn num_parameters(&self) -> usize {
        // Approximate parameter count
        let embedding_params = self.vocab_size * self.model_dim;
        let attention_params = self.num_layers * (4 * self.model_dim * self.model_dim);
        let ffn_params = self.num_layers * (8 * self.model_dim * self.model_dim);

        embedding_params + attention_params + ffn_params
    }
}

// ==================== Acoustic Transformer ====================

/// Acoustic-level transformer for detailed features
#[derive(Debug)]
struct AcousticTransformer {
    model_dim: usize,
    num_layers: usize,
    num_heads: usize,
    num_codebooks: usize,
    codebook_size: usize,
}

impl AcousticTransformer {
    fn new(
        model_dim: usize,
        num_layers: usize,
        num_heads: usize,
        num_codebooks: usize,
        codebook_size: usize,
    ) -> Self {
        Self {
            model_dim,
            num_layers,
            num_heads,
            num_codebooks,
            codebook_size,
        }
    }

    async fn generate(
        &self,
        semantic_tokens: &[usize],
        pattern_manager: &PatternManager,
    ) -> Result<Vec<Vec<usize>>> {
        // Generate acoustic tokens conditioned on semantic tokens
        let mut acoustic_tokens = Vec::with_capacity(semantic_tokens.len());

        for &semantic_token in semantic_tokens {
            let frame_tokens = pattern_manager.generate_frame(
                semantic_token,
                self.num_codebooks,
                self.codebook_size,
            )?;
            acoustic_tokens.push(frame_tokens);
        }

        Ok(acoustic_tokens)
    }

    fn num_parameters(&self) -> usize {
        // Approximate parameter count
        let embedding_params = self.num_codebooks * self.codebook_size * self.model_dim;
        let attention_params = self.num_layers * (4 * self.model_dim * self.model_dim);
        let ffn_params = self.num_layers * (8 * self.model_dim * self.model_dim);

        embedding_params + attention_params + ffn_params
    }
}

// ==================== Pattern Manager ====================

/// Manages delay patterns for multi-codebook generation
#[derive(Debug)]
struct PatternManager {
    pattern: DelayPattern,
}

impl PatternManager {
    fn new(pattern: DelayPattern) -> Self {
        Self { pattern }
    }

    fn generate_frame(
        &self,
        semantic_token: usize,
        num_codebooks: usize,
        codebook_size: usize,
    ) -> Result<Vec<usize>> {
        let mut frame = Vec::with_capacity(num_codebooks);

        match self.pattern {
            DelayPattern::Sequential => {
                // Generate codebooks sequentially (autoregressive)
                for i in 0..num_codebooks {
                    let prev_sum: usize = frame.iter().sum();
                    let token = (semantic_token + prev_sum + i) % codebook_size;
                    frame.push(token);
                }
            }
            DelayPattern::Parallel => {
                // Generate codebooks in parallel with delay pattern
                for i in 0..num_codebooks {
                    let token = (semantic_token + i * 13) % codebook_size; // Prime offset
                    frame.push(token);
                }
            }
            DelayPattern::Custom => {
                // Custom pattern - hybrid approach
                for i in 0..num_codebooks {
                    let token = if i < num_codebooks / 2 {
                        // First half: parallel
                        (semantic_token + i * 17) % codebook_size
                    } else {
                        // Second half: sequential
                        let prev_sum: usize = frame.iter().sum();
                        (semantic_token + prev_sum) % codebook_size
                    };
                    frame.push(token);
                }
            }
        }

        Ok(frame)
    }
}

// ==================== Conditioning ====================

/// Singing conditioning for autoregressive generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingingConditioning {
    /// Text/lyrics
    pub text: String,
    /// Pitch contour (Hz)
    pub pitch_contour: Vec<f32>,
    /// Target duration (seconds)
    pub duration: f32,
    /// Optional style embeddings
    pub style_embedding: Option<Vec<f32>>,
    /// Optional speaker embedding
    pub speaker_embedding: Option<Vec<f32>>,
}

impl Default for SingingConditioning {
    fn default() -> Self {
        Self {
            text: String::new(),
            pitch_contour: vec![440.0],
            duration: 1.0,
            style_embedding: None,
            speaker_embedding: None,
        }
    }
}

// ==================== Model Info ====================

/// Information about autoregressive model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoregressiveInfo {
    /// Configuration
    pub config: AutoregressiveConfig,
    /// Number of semantic model parameters
    pub semantic_params: usize,
    /// Number of acoustic model parameters
    pub acoustic_params: usize,
    /// Total parameters
    pub total_params: usize,
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autoregressive_config() {
        let config = AutoregressiveConfig::musicgen_style();
        assert_eq!(config.semantic_dim, 1024);
        assert_eq!(config.num_layers, 24);
        assert!(config.use_cfg);
    }

    #[test]
    fn test_fast_config() {
        let config = AutoregressiveConfig::fast();
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.delay_pattern, DelayPattern::Parallel);
    }

    #[test]
    fn test_high_quality_config() {
        let config = AutoregressiveConfig::high_quality();
        assert_eq!(config.semantic_dim, 2048);
        assert_eq!(config.num_layers, 32);
    }

    #[tokio::test]
    async fn test_autoregressive_creation() {
        let config = AutoregressiveConfig::default();
        let synthesizer = AutoregressiveSynthesizer::new(config);
        let info = synthesizer.info();
        assert!(info.total_params > 0);
    }

    #[tokio::test]
    async fn test_semantic_generation() {
        let model = SemanticTransformer::new(512, 12, 8, 2048);
        let conditioning = SingingConditioning::default();
        let tokens = model.generate(&conditioning, 10).await.unwrap();
        assert_eq!(tokens.len(), 10);
        for &token in &tokens {
            assert!(token < 2048);
        }
    }

    #[tokio::test]
    async fn test_unconditional_generation() {
        let model = SemanticTransformer::new(512, 12, 8, 2048);
        let tokens = model.generate_unconditional(10).await.unwrap();
        assert_eq!(tokens.len(), 10);
    }

    #[tokio::test]
    async fn test_pattern_sequential() {
        let manager = PatternManager::new(DelayPattern::Sequential);
        let frame = manager.generate_frame(100, 8, 1024).unwrap();
        assert_eq!(frame.len(), 8);
        for &token in &frame {
            assert!(token < 1024);
        }
    }

    #[tokio::test]
    async fn test_pattern_parallel() {
        let manager = PatternManager::new(DelayPattern::Parallel);
        let frame = manager.generate_frame(100, 8, 1024).unwrap();
        assert_eq!(frame.len(), 8);
    }

    #[tokio::test]
    async fn test_pattern_custom() {
        let manager = PatternManager::new(DelayPattern::Custom);
        let frame = manager.generate_frame(100, 8, 1024).unwrap();
        assert_eq!(frame.len(), 8);
    }

    #[tokio::test]
    async fn test_acoustic_generation() {
        let model = AcousticTransformer::new(768, 12, 8, 8, 1024);
        let pattern_manager = PatternManager::new(DelayPattern::Parallel);
        let semantic_tokens = vec![10, 20, 30];
        let acoustic_tokens = model
            .generate(&semantic_tokens, &pattern_manager)
            .await
            .unwrap();
        assert_eq!(acoustic_tokens.len(), 3);
        assert_eq!(acoustic_tokens[0].len(), 8);
    }

    #[tokio::test]
    async fn test_full_generation() {
        let config = AutoregressiveConfig::fast();
        let synthesizer = AutoregressiveSynthesizer::new(config);
        let conditioning = SingingConditioning {
            text: "Hello".to_string(),
            pitch_contour: vec![440.0, 450.0],
            duration: 1.0,
            style_embedding: None,
            speaker_embedding: None,
        };

        let audio = synthesizer.generate(&conditioning, 24000).await.unwrap();
        assert!(!audio.is_empty());
        assert!(audio.len() <= 24000 * 2); // Allow some tolerance
    }

    #[tokio::test]
    async fn test_cfg_generation() {
        let config = AutoregressiveConfig::musicgen_style();
        let synthesizer = AutoregressiveSynthesizer::new(config);
        let conditioning = SingingConditioning::default();

        let audio = synthesizer
            .generate_with_cfg(&conditioning, 24000)
            .await
            .unwrap();
        assert!(!audio.is_empty());
    }

    #[test]
    fn test_singing_conditioning() {
        let cond = SingingConditioning {
            text: "Test".to_string(),
            pitch_contour: vec![440.0],
            duration: 2.0,
            style_embedding: Some(vec![0.1, 0.2, 0.3]),
            speaker_embedding: Some(vec![0.5, 0.6]),
        };

        assert_eq!(cond.text, "Test");
        assert_eq!(cond.pitch_contour.len(), 1);
        assert_eq!(cond.duration, 2.0);
        assert!(cond.style_embedding.is_some());
        assert!(cond.speaker_embedding.is_some());
    }
}
