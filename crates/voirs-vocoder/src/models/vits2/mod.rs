//! VITS2: Variational Inference with adversarial learning for end-to-end Text-to-Speech Version 2
//!
//! This module implements the VITS2 architecture, which provides significant improvements over VITS:
//! - Enhanced neural vocoder with better audio quality
//! - Improved alignment learning through monotonic alignment search
//! - Advanced flow-based models for latent representation
//! - Better speaker conditioning and multi-speaker support
//! - Reduced synthesis artifacts and improved naturalness

use crate::{Result, VocoderError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod attention;
pub mod duration_predictor;
pub mod encoder;
pub mod flow;
pub mod generator;
pub mod mas;
pub mod modules;
pub mod posterior_encoder;
pub mod stochastic_duration_predictor;
pub mod text_encoder;

/// VITS2 model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vits2Config {
    /// Model name
    pub model_name: String,

    /// Audio configuration
    pub sample_rate: u32,
    pub hop_size: u32,
    pub win_size: u32,
    pub n_fft: u32,
    pub n_mel: u32,
    pub mel_fmin: f32,
    pub mel_fmax: f32,

    /// Text encoder configuration
    pub text_encoder_hidden_channels: u32,
    pub text_encoder_filter_channels: u32,
    pub text_encoder_n_heads: u32,
    pub text_encoder_n_layers: u32,
    pub text_encoder_kernel_size: u32,
    pub text_encoder_p_dropout: f32,

    /// Posterior encoder configuration
    pub posterior_encoder_hidden_channels: u32,
    pub posterior_encoder_kernel_size: u32,
    pub posterior_encoder_dilation_rate: u32,
    pub posterior_encoder_n_layers: u32,

    /// Flow configuration
    pub flow_hidden_channels: u32,
    pub flow_kernel_size: u32,
    pub flow_dilation_rate: u32,
    pub flow_n_blocks: u32,
    pub flow_n_layers: u32,
    pub flow_n_split: u32,
    pub flow_n_sqz: u32,

    /// Generator configuration
    pub generator_hidden_channels: u32,
    pub generator_kernel_size: u32,
    pub generator_dilation_rate: u32,
    pub generator_n_layers: u32,
    pub generator_upsample_rates: Vec<u32>,
    pub generator_upsample_kernel_sizes: Vec<u32>,
    pub generator_upsample_initial_channel: u32,
    pub generator_resblock_kernel_sizes: Vec<u32>,
    pub generator_resblock_dilation_sizes: Vec<Vec<u32>>,

    /// Duration predictor configuration
    pub duration_predictor_hidden_channels: u32,
    pub duration_predictor_kernel_size: u32,
    pub duration_predictor_p_dropout: f32,

    /// Stochastic duration predictor configuration
    pub stochastic_duration_predictor_hidden_channels: u32,
    pub stochastic_duration_predictor_kernel_size: u32,
    pub stochastic_duration_predictor_p_dropout: f32,
    pub stochastic_duration_predictor_n_flows: u32,
    pub stochastic_duration_predictor_gin_channels: u32,

    /// Speaker embedding configuration
    pub n_speakers: u32,
    pub gin_channels: u32,

    /// Training configuration
    pub segment_size: u32,
    pub n_layers_q: u32,
    pub use_spectral_norm: bool,
    pub use_transformer_flows: bool,
    pub transformer_flow_type: String,

    /// VITS2 specific improvements
    pub use_duration_discriminator: bool,
    pub use_monotonic_alignment_search: bool,
    pub mas_noise_scale: f32,
    pub mas_noise_scale_decay: f32,
    pub use_phoneme_level_energy: bool,
    pub use_phoneme_level_pitch: bool,

    /// Advanced features
    pub use_speaker_classifier: bool,
    pub speaker_classifier_layers: u32,
    pub use_emotion_conditioning: bool,
    pub emotion_embedding_dim: u32,
    pub use_style_encoder: bool,
    pub style_encoder_layers: u32,
}

impl Default for Vits2Config {
    fn default() -> Self {
        Self {
            model_name: "VITS2".to_string(),

            // Audio config
            sample_rate: 22050,
            hop_size: 256,
            win_size: 1024,
            n_fft: 1024,
            n_mel: 80,
            mel_fmin: 0.0,
            mel_fmax: 11025.0,

            // Text encoder
            text_encoder_hidden_channels: 192,
            text_encoder_filter_channels: 768,
            text_encoder_n_heads: 2,
            text_encoder_n_layers: 6,
            text_encoder_kernel_size: 3,
            text_encoder_p_dropout: 0.1,

            // Posterior encoder
            posterior_encoder_hidden_channels: 512,
            posterior_encoder_kernel_size: 5,
            posterior_encoder_dilation_rate: 1,
            posterior_encoder_n_layers: 16,

            // Flow
            flow_hidden_channels: 192,
            flow_kernel_size: 5,
            flow_dilation_rate: 1,
            flow_n_blocks: 4,
            flow_n_layers: 4,
            flow_n_split: 4,
            flow_n_sqz: 2,

            // Generator
            generator_hidden_channels: 512,
            generator_kernel_size: 7,
            generator_dilation_rate: 1,
            generator_n_layers: 3,
            generator_upsample_rates: vec![8, 8, 2, 2],
            generator_upsample_kernel_sizes: vec![16, 16, 4, 4],
            generator_upsample_initial_channel: 512,
            generator_resblock_kernel_sizes: vec![3, 7, 11],
            generator_resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],

            // Duration predictor
            duration_predictor_hidden_channels: 256,
            duration_predictor_kernel_size: 3,
            duration_predictor_p_dropout: 0.5,

            // Stochastic duration predictor
            stochastic_duration_predictor_hidden_channels: 192,
            stochastic_duration_predictor_kernel_size: 3,
            stochastic_duration_predictor_p_dropout: 0.5,
            stochastic_duration_predictor_n_flows: 4,
            stochastic_duration_predictor_gin_channels: 256,

            // Speaker configuration
            n_speakers: 1,
            gin_channels: 256,

            // Training
            segment_size: 8192,
            n_layers_q: 3,
            use_spectral_norm: false,
            use_transformer_flows: true,
            transformer_flow_type: "monotonic_attention".to_string(),

            // VITS2 improvements
            use_duration_discriminator: true,
            use_monotonic_alignment_search: true,
            mas_noise_scale: 1.0,
            mas_noise_scale_decay: 0.999,
            use_phoneme_level_energy: true,
            use_phoneme_level_pitch: true,

            // Advanced features
            use_speaker_classifier: false,
            speaker_classifier_layers: 2,
            use_emotion_conditioning: false,
            emotion_embedding_dim: 64,
            use_style_encoder: false,
            style_encoder_layers: 2,
        }
    }
}

impl Vits2Config {
    /// Create a high-quality configuration for production use
    pub fn high_quality() -> Self {
        let mut config = Self::default();
        config.model_name = "VITS2-HQ".to_string();
        config.sample_rate = 44100;
        config.mel_fmax = 22050.0;
        config.text_encoder_hidden_channels = 256;
        config.text_encoder_filter_channels = 1024;
        config.text_encoder_n_layers = 8;
        config.posterior_encoder_hidden_channels = 768;
        config.flow_hidden_channels = 256;
        config.generator_hidden_channels = 768;
        config.generator_upsample_initial_channel = 768;
        config
    }

    /// Create a fast configuration for real-time use
    pub fn fast() -> Self {
        let mut config = Self::default();
        config.model_name = "VITS2-Fast".to_string();
        config.text_encoder_hidden_channels = 128;
        config.text_encoder_filter_channels = 512;
        config.text_encoder_n_layers = 4;
        config.posterior_encoder_hidden_channels = 256;
        config.flow_hidden_channels = 128;
        config.flow_n_blocks = 2;
        config.generator_hidden_channels = 256;
        config.generator_upsample_initial_channel = 256;
        config
    }

    /// Create a multi-speaker configuration
    pub fn multi_speaker(n_speakers: u32) -> Self {
        let mut config = Self::default();
        config.model_name = format!("VITS2-MultiSpeaker-{}", n_speakers);
        config.n_speakers = n_speakers;
        config.gin_channels = if n_speakers > 100 { 512 } else { 256 };
        config.use_speaker_classifier = true;
        config
    }

    /// Enable emotion conditioning
    pub fn with_emotion_conditioning(mut self) -> Self {
        self.use_emotion_conditioning = true;
        self.emotion_embedding_dim = 128;
        self
    }

    /// Enable style encoding
    pub fn with_style_encoding(mut self) -> Self {
        self.use_style_encoder = true;
        self.style_encoder_layers = 3;
        self
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(VocoderError::ModelError(
                "Sample rate must be greater than 0".to_string(),
            ));
        }

        if self.n_mel == 0 {
            return Err(VocoderError::ModelError(
                "Number of mel channels must be greater than 0".to_string(),
            ));
        }

        if self.text_encoder_hidden_channels == 0 {
            return Err(VocoderError::ModelError(
                "Text encoder hidden channels must be greater than 0".to_string(),
            ));
        }

        if self.generator_upsample_rates.is_empty() {
            return Err(VocoderError::ModelError(
                "Generator upsample rates cannot be empty".to_string(),
            ));
        }

        if self.generator_upsample_rates.len() != self.generator_upsample_kernel_sizes.len() {
            return Err(VocoderError::ModelError(
                "Generator upsample rates and kernel sizes must have the same length".to_string(),
            ));
        }

        let total_upsample = self.generator_upsample_rates.iter().product::<u32>();
        if total_upsample != self.hop_size {
            return Err(VocoderError::ModelError(format!(
                "Total upsample factor ({}) must equal hop size ({})",
                total_upsample, self.hop_size
            )));
        }

        Ok(())
    }

    /// Get total model parameters estimate
    pub fn estimated_parameters(&self) -> u64 {
        let text_encoder_params = self.text_encoder_hidden_channels as u64
            * self.text_encoder_filter_channels as u64
            * self.text_encoder_n_layers as u64;
        let posterior_encoder_params =
            self.posterior_encoder_hidden_channels as u64 * self.posterior_encoder_n_layers as u64;
        let flow_params = self.flow_hidden_channels as u64
            * self.flow_n_blocks as u64
            * self.flow_n_layers as u64;
        let generator_params =
            self.generator_hidden_channels as u64 * self.generator_n_layers as u64;

        text_encoder_params + posterior_encoder_params + flow_params + generator_params
    }

    /// Get memory requirements estimate in MB
    pub fn estimated_memory_mb(&self) -> f32 {
        let params = self.estimated_parameters();
        let param_memory = params as f32 * 4.0 / (1024.0 * 1024.0); // 4 bytes per float32
        let activation_memory =
            self.segment_size as f32 * self.n_mel as f32 * 4.0 / (1024.0 * 1024.0);
        param_memory + activation_memory * 2.0 // Factor for intermediate activations
    }
}

/// VITS2 synthesis parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vits2SynthesisParams {
    /// Noise scale for decoder
    pub noise_scale: f32,
    /// Noise scale for duration predictor
    pub noise_scale_w: f32,
    /// Length scale for duration
    pub length_scale: f32,
    /// Speaker ID (for multi-speaker models)
    pub speaker_id: Option<u32>,
    /// Emotion embedding (if available)
    pub emotion: Option<Vec<f32>>,
    /// Style embedding (if available)
    pub style: Option<Vec<f32>>,
    /// Temperature for sampling
    pub temperature: f32,
    /// Use deterministic inference
    pub deterministic: bool,
}

impl Default for Vits2SynthesisParams {
    fn default() -> Self {
        Self {
            noise_scale: 0.667,
            noise_scale_w: 0.8,
            length_scale: 1.0,
            speaker_id: None,
            emotion: None,
            style: None,
            temperature: 1.0,
            deterministic: false,
        }
    }
}

impl Vits2SynthesisParams {
    /// Create parameters for high-quality synthesis
    pub fn high_quality() -> Self {
        Self {
            noise_scale: 0.333,
            noise_scale_w: 0.4,
            length_scale: 1.0,
            temperature: 0.7,
            deterministic: true,
            ..Default::default()
        }
    }

    /// Create parameters for fast synthesis
    pub fn fast() -> Self {
        Self {
            noise_scale: 1.0,
            noise_scale_w: 1.0,
            length_scale: 0.8,
            temperature: 1.2,
            deterministic: false,
            ..Default::default()
        }
    }

    /// Set speaker for multi-speaker synthesis
    pub fn with_speaker(mut self, speaker_id: u32) -> Self {
        self.speaker_id = Some(speaker_id);
        self
    }

    /// Set emotion for emotional synthesis
    pub fn with_emotion(mut self, emotion: Vec<f32>) -> Self {
        self.emotion = Some(emotion);
        self
    }

    /// Set style for stylistic synthesis
    pub fn with_style(mut self, style: Vec<f32>) -> Self {
        self.style = Some(style);
        self
    }
}

/// VITS2 model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vits2Metadata {
    /// Model version
    pub version: String,
    /// Training dataset information
    pub dataset: String,
    /// Training steps
    pub training_steps: u64,
    /// Model quality metrics
    pub quality_metrics: HashMap<String, f32>,
    /// Supported languages
    pub languages: Vec<String>,
    /// Supported speakers (for multi-speaker models)
    pub speakers: Vec<String>,
    /// Model size in MB
    pub model_size_mb: f32,
    /// Creation timestamp
    pub created_at: String,
}

impl Default for Vits2Metadata {
    fn default() -> Self {
        Self {
            version: "2.0.0".to_string(),
            dataset: "unknown".to_string(),
            training_steps: 0,
            quality_metrics: HashMap::new(),
            languages: vec!["en".to_string()],
            speakers: vec![],
            model_size_mb: 0.0,
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vits2_config_default() {
        let config = Vits2Config::default();
        assert_eq!(config.model_name, "VITS2");
        assert_eq!(config.sample_rate, 22050);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_vits2_config_high_quality() {
        let config = Vits2Config::high_quality();
        assert_eq!(config.model_name, "VITS2-HQ");
        assert_eq!(config.sample_rate, 44100);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_vits2_config_fast() {
        let config = Vits2Config::fast();
        assert_eq!(config.model_name, "VITS2-Fast");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_vits2_config_multi_speaker() {
        let config = Vits2Config::multi_speaker(100);
        assert_eq!(config.n_speakers, 100);
        assert!(config.use_speaker_classifier);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_vits2_config_validation() {
        let mut config = Vits2Config::default();

        // Valid config should pass
        assert!(config.validate().is_ok());

        // Invalid sample rate should fail
        config.sample_rate = 0;
        assert!(config.validate().is_err());

        // Reset and test invalid mel channels
        config = Vits2Config::default();
        config.n_mel = 0;
        assert!(config.validate().is_err());

        // Reset and test mismatched upsample arrays
        config = Vits2Config::default();
        config.generator_upsample_kernel_sizes.pop();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_vits2_synthesis_params() {
        let params = Vits2SynthesisParams::default();
        assert_eq!(params.noise_scale, 0.667);
        assert_eq!(params.length_scale, 1.0);

        let hq_params = Vits2SynthesisParams::high_quality();
        assert_eq!(hq_params.noise_scale, 0.333);
        assert!(hq_params.deterministic);

        let fast_params = Vits2SynthesisParams::fast();
        assert_eq!(fast_params.length_scale, 0.8);
        assert!(!fast_params.deterministic);
    }

    #[test]
    fn test_vits2_memory_estimation() {
        let config = Vits2Config::default();
        let memory_mb = config.estimated_memory_mb();
        assert!(memory_mb > 0.0);
        assert!(memory_mb < 10000.0); // Reasonable upper bound

        let params = config.estimated_parameters();
        assert!(params > 0);
    }

    #[test]
    fn test_vits2_metadata() {
        let metadata = Vits2Metadata::default();
        assert_eq!(metadata.version, "2.0.0");
        assert!(!metadata.languages.is_empty());
        assert!(!metadata.created_at.is_empty());
    }
}
