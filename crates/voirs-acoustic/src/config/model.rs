//! Model-specific configuration structures
//!
//! This module defines configuration structures for different acoustic model
//! architectures including VITS, FastSpeech2, and other models.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{AcousticError, LanguageCode, Result};

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture type
    pub architecture: ModelArchitecture,
    /// Model file path or HuggingFace Hub ID
    pub model_path: String,
    /// Model version
    pub version: String,
    /// Architecture-specific parameters
    pub architecture_params: ArchitectureParams,
    /// Supported languages
    pub supported_languages: Vec<LanguageCode>,
    /// Model metadata
    pub metadata: ModelMetadata,
}

impl ModelConfig {
    /// Create new model configuration
    pub fn new(architecture: ModelArchitecture, model_path: String) -> Self {
        let architecture_params =
            ArchitectureParams::default_for_architecture(architecture.clone());
        Self {
            architecture,
            model_path,
            version: "1.0.0".to_string(),
            architecture_params,
            supported_languages: vec![LanguageCode::EnUs],
            metadata: ModelMetadata::default(),
        }
    }

    /// Validate model configuration
    pub fn validate(&self) -> Result<()> {
        if self.model_path.is_empty() {
            return Err(AcousticError::ConfigError(
                "Model path cannot be empty".to_string(),
            ));
        }

        if self.supported_languages.is_empty() {
            return Err(AcousticError::ConfigError(
                "At least one language must be supported".to_string(),
            ));
        }

        self.architecture_params.validate()?;
        Ok(())
    }

    /// Merge with another model configuration
    pub fn merge(&mut self, other: &ModelConfig) {
        self.architecture = other.architecture.clone();
        self.model_path = other.model_path.clone();
        self.version = other.version.clone();
        self.architecture_params.merge(&other.architecture_params);
        self.supported_languages = other.supported_languages.clone();
        self.metadata.merge(&other.metadata);
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::new(ModelArchitecture::Vits, "dummy".to_string())
    }
}

/// Supported model architectures
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
    Vits,
    /// FastSpeech2 (Fast and High-Quality End-to-End Text-to-Speech)
    FastSpeech2,
    /// Tacotron2 (Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions)
    Tacotron2,
    /// Custom architecture
    Custom(String),
}

impl ModelArchitecture {
    /// Get string representation
    pub fn as_str(&self) -> &str {
        match self {
            ModelArchitecture::Vits => "vits",
            ModelArchitecture::FastSpeech2 => "fastspeech2",
            ModelArchitecture::Tacotron2 => "tacotron2",
            ModelArchitecture::Custom(name) => name,
        }
    }
}

/// Architecture-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureParams {
    /// VITS-specific parameters
    pub vits: Option<VitsParams>,
    /// FastSpeech2-specific parameters
    pub fastspeech2: Option<FastSpeech2Params>,
    /// Tacotron2-specific parameters
    pub tacotron2: Option<Tacotron2Params>,
    /// Custom parameters
    pub custom: Option<HashMap<String, serde_json::Value>>,
}

impl ArchitectureParams {
    /// Create default parameters for an architecture
    pub fn default_for_architecture(arch: ModelArchitecture) -> Self {
        match arch {
            ModelArchitecture::Vits => Self {
                vits: Some(VitsParams::default()),
                fastspeech2: None,
                tacotron2: None,
                custom: None,
            },
            ModelArchitecture::FastSpeech2 => Self {
                vits: None,
                fastspeech2: Some(FastSpeech2Params::default()),
                tacotron2: None,
                custom: None,
            },
            ModelArchitecture::Tacotron2 => Self {
                vits: None,
                fastspeech2: None,
                tacotron2: Some(Tacotron2Params::default()),
                custom: None,
            },
            ModelArchitecture::Custom(_) => Self {
                vits: None,
                fastspeech2: None,
                tacotron2: None,
                custom: Some(HashMap::new()),
            },
        }
    }

    /// Validate architecture parameters
    pub fn validate(&self) -> Result<()> {
        if let Some(vits) = &self.vits {
            vits.validate()?;
        }
        if let Some(fastspeech2) = &self.fastspeech2 {
            fastspeech2.validate()?;
        }
        if let Some(tacotron2) = &self.tacotron2 {
            tacotron2.validate()?;
        }
        Ok(())
    }

    /// Merge with another architecture parameters
    pub fn merge(&mut self, other: &ArchitectureParams) {
        if let Some(other_vits) = &other.vits {
            if let Some(vits) = &mut self.vits {
                vits.merge(other_vits);
            } else {
                self.vits = Some(other_vits.clone());
            }
        }

        if let Some(other_fs2) = &other.fastspeech2 {
            if let Some(fs2) = &mut self.fastspeech2 {
                fs2.merge(other_fs2);
            } else {
                self.fastspeech2 = Some(other_fs2.clone());
            }
        }

        if let Some(other_taco2) = &other.tacotron2 {
            if let Some(taco2) = &mut self.tacotron2 {
                taco2.merge(other_taco2);
            } else {
                self.tacotron2 = Some(other_taco2.clone());
            }
        }

        if let Some(other_custom) = &other.custom {
            if let Some(custom) = &mut self.custom {
                custom.extend(other_custom.clone());
            } else {
                self.custom = Some(other_custom.clone());
            }
        }
    }
}

impl Default for ArchitectureParams {
    fn default() -> Self {
        Self::default_for_architecture(ModelArchitecture::Vits)
    }
}

/// VITS model parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VitsParams {
    /// Text encoder parameters
    pub text_encoder: TextEncoderParams,
    /// Posterior encoder parameters
    pub posterior_encoder: PosteriorEncoderParams,
    /// Flow parameters
    pub flow: FlowParams,
    /// Decoder parameters
    pub decoder: DecoderParams,
    /// Number of speakers (for multi-speaker models)
    pub n_speakers: Option<u32>,
    /// Speaker embedding dimension
    pub speaker_embed_dim: Option<u32>,
}

impl VitsParams {
    /// Validate VITS parameters
    pub fn validate(&self) -> Result<()> {
        self.text_encoder.validate()?;
        self.posterior_encoder.validate()?;
        self.flow.validate()?;
        self.decoder.validate()?;

        if let Some(n_speakers) = self.n_speakers {
            if n_speakers == 0 {
                return Err(AcousticError::ConfigError(
                    "Number of speakers must be > 0".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Merge with another VITS parameters
    pub fn merge(&mut self, other: &VitsParams) {
        self.text_encoder.merge(&other.text_encoder);
        self.posterior_encoder.merge(&other.posterior_encoder);
        self.flow.merge(&other.flow);
        self.decoder.merge(&other.decoder);

        if other.n_speakers.is_some() {
            self.n_speakers = other.n_speakers;
        }
        if other.speaker_embed_dim.is_some() {
            self.speaker_embed_dim = other.speaker_embed_dim;
        }
    }
}

/// Text encoder parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEncoderParams {
    /// Number of layers
    pub n_layers: u32,
    /// Hidden dimension
    pub hidden_dim: u32,
    /// Number of attention heads
    pub n_heads: u32,
    /// Filter dimension
    pub filter_dim: u32,
    /// Dropout rate
    pub dropout: f32,
}

impl TextEncoderParams {
    /// Validate text encoder parameters
    pub fn validate(&self) -> Result<()> {
        if self.n_layers == 0 {
            return Err(AcousticError::ConfigError(
                "Number of layers must be > 0".to_string(),
            ));
        }
        if self.hidden_dim == 0 {
            return Err(AcousticError::ConfigError(
                "Hidden dimension must be > 0".to_string(),
            ));
        }
        if self.n_heads == 0 {
            return Err(AcousticError::ConfigError(
                "Number of heads must be > 0".to_string(),
            ));
        }
        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err(AcousticError::ConfigError(
                "Dropout must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(())
    }

    /// Merge with another text encoder parameters
    pub fn merge(&mut self, other: &TextEncoderParams) {
        self.n_layers = other.n_layers;
        self.hidden_dim = other.hidden_dim;
        self.n_heads = other.n_heads;
        self.filter_dim = other.filter_dim;
        self.dropout = other.dropout;
    }
}

impl Default for TextEncoderParams {
    fn default() -> Self {
        Self {
            n_layers: 6,
            hidden_dim: 192,
            n_heads: 2,
            filter_dim: 768,
            dropout: 0.1,
        }
    }
}

/// Posterior encoder parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PosteriorEncoderParams {
    /// Number of layers
    pub n_layers: u32,
    /// Hidden channels
    pub hidden_channels: u32,
    /// Kernel size
    pub kernel_size: u32,
    /// Dilation rate
    pub dilation_rate: u32,
}

impl PosteriorEncoderParams {
    /// Validate posterior encoder parameters
    pub fn validate(&self) -> Result<()> {
        if self.n_layers == 0 {
            return Err(AcousticError::ConfigError(
                "Number of layers must be > 0".to_string(),
            ));
        }
        if self.hidden_channels == 0 {
            return Err(AcousticError::ConfigError(
                "Hidden channels must be > 0".to_string(),
            ));
        }
        if self.kernel_size == 0 {
            return Err(AcousticError::ConfigError(
                "Kernel size must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Merge with another posterior encoder parameters
    pub fn merge(&mut self, other: &PosteriorEncoderParams) {
        self.n_layers = other.n_layers;
        self.hidden_channels = other.hidden_channels;
        self.kernel_size = other.kernel_size;
        self.dilation_rate = other.dilation_rate;
    }
}

impl Default for PosteriorEncoderParams {
    fn default() -> Self {
        Self {
            n_layers: 16,
            hidden_channels: 192,
            kernel_size: 5,
            dilation_rate: 1,
        }
    }
}

/// Flow parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowParams {
    /// Number of flows
    pub n_flows: u32,
    /// Number of layers per flow
    pub n_layers: u32,
    /// Hidden channels
    pub hidden_channels: u32,
    /// Kernel size
    pub kernel_size: u32,
}

impl FlowParams {
    /// Validate flow parameters
    pub fn validate(&self) -> Result<()> {
        if self.n_flows == 0 {
            return Err(AcousticError::ConfigError(
                "Number of flows must be > 0".to_string(),
            ));
        }
        if self.n_layers == 0 {
            return Err(AcousticError::ConfigError(
                "Number of layers must be > 0".to_string(),
            ));
        }
        if self.hidden_channels == 0 {
            return Err(AcousticError::ConfigError(
                "Hidden channels must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Merge with another flow parameters
    pub fn merge(&mut self, other: &FlowParams) {
        self.n_flows = other.n_flows;
        self.n_layers = other.n_layers;
        self.hidden_channels = other.hidden_channels;
        self.kernel_size = other.kernel_size;
    }
}

impl Default for FlowParams {
    fn default() -> Self {
        Self {
            n_flows: 4,
            n_layers: 4,
            hidden_channels: 192,
            kernel_size: 5,
        }
    }
}

/// Decoder parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderParams {
    /// Initial channels
    pub initial_channels: u32,
    /// Resblock kernel sizes
    pub resblock_kernel_sizes: Vec<u32>,
    /// Resblock dilation sizes
    pub resblock_dilation_sizes: Vec<Vec<u32>>,
    /// Upsample rates
    pub upsample_rates: Vec<u32>,
    /// Upsample kernel sizes
    pub upsample_kernel_sizes: Vec<u32>,
    /// Upsample initial channels
    pub upsample_initial_channels: u32,
}

impl DecoderParams {
    /// Validate decoder parameters
    pub fn validate(&self) -> Result<()> {
        if self.initial_channels == 0 {
            return Err(AcousticError::ConfigError(
                "Initial channels must be > 0".to_string(),
            ));
        }
        if self.resblock_kernel_sizes.is_empty() {
            return Err(AcousticError::ConfigError(
                "Resblock kernel sizes cannot be empty".to_string(),
            ));
        }
        if self.upsample_rates.is_empty() {
            return Err(AcousticError::ConfigError(
                "Upsample rates cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    /// Merge with another decoder parameters
    pub fn merge(&mut self, other: &DecoderParams) {
        self.initial_channels = other.initial_channels;
        self.resblock_kernel_sizes = other.resblock_kernel_sizes.clone();
        self.resblock_dilation_sizes = other.resblock_dilation_sizes.clone();
        self.upsample_rates = other.upsample_rates.clone();
        self.upsample_kernel_sizes = other.upsample_kernel_sizes.clone();
        self.upsample_initial_channels = other.upsample_initial_channels;
    }
}

impl Default for DecoderParams {
    fn default() -> Self {
        Self {
            initial_channels: 512,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            upsample_rates: vec![8, 8, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            upsample_initial_channels: 128,
        }
    }
}

/// FastSpeech2 model parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FastSpeech2Params {
    /// Text encoder parameters
    pub text_encoder: TextEncoderParams,
    /// Variance adaptor parameters
    pub variance_adaptor: VarianceAdaptorParams,
    /// Mel decoder parameters
    pub mel_decoder: MelDecoderParams,
    /// Number of speakers
    pub n_speakers: Option<u32>,
}

impl FastSpeech2Params {
    /// Validate FastSpeech2 parameters
    pub fn validate(&self) -> Result<()> {
        self.text_encoder.validate()?;
        self.variance_adaptor.validate()?;
        self.mel_decoder.validate()?;
        Ok(())
    }

    /// Merge with another FastSpeech2 parameters
    pub fn merge(&mut self, other: &FastSpeech2Params) {
        self.text_encoder.merge(&other.text_encoder);
        self.variance_adaptor.merge(&other.variance_adaptor);
        self.mel_decoder.merge(&other.mel_decoder);

        if other.n_speakers.is_some() {
            self.n_speakers = other.n_speakers;
        }
    }
}

/// Variance adaptor parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceAdaptorParams {
    /// Filter size
    pub filter_size: u32,
    /// Kernel size
    pub kernel_size: u32,
    /// Dropout rate
    pub dropout: f32,
}

impl VarianceAdaptorParams {
    /// Validate variance adaptor parameters
    pub fn validate(&self) -> Result<()> {
        if self.filter_size == 0 {
            return Err(AcousticError::ConfigError(
                "Filter size must be > 0".to_string(),
            ));
        }
        if self.kernel_size == 0 {
            return Err(AcousticError::ConfigError(
                "Kernel size must be > 0".to_string(),
            ));
        }
        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err(AcousticError::ConfigError(
                "Dropout must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(())
    }

    /// Merge with another variance adaptor parameters
    pub fn merge(&mut self, other: &VarianceAdaptorParams) {
        self.filter_size = other.filter_size;
        self.kernel_size = other.kernel_size;
        self.dropout = other.dropout;
    }
}

impl Default for VarianceAdaptorParams {
    fn default() -> Self {
        Self {
            filter_size: 256,
            kernel_size: 3,
            dropout: 0.5,
        }
    }
}

/// Mel decoder parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelDecoderParams {
    /// Number of layers
    pub n_layers: u32,
    /// Hidden dimension
    pub hidden_dim: u32,
    /// Number of attention heads
    pub n_heads: u32,
    /// Filter dimension
    pub filter_dim: u32,
    /// Dropout rate
    pub dropout: f32,
}

impl MelDecoderParams {
    /// Validate mel decoder parameters
    pub fn validate(&self) -> Result<()> {
        if self.n_layers == 0 {
            return Err(AcousticError::ConfigError(
                "Number of layers must be > 0".to_string(),
            ));
        }
        if self.hidden_dim == 0 {
            return Err(AcousticError::ConfigError(
                "Hidden dimension must be > 0".to_string(),
            ));
        }
        if self.n_heads == 0 {
            return Err(AcousticError::ConfigError(
                "Number of heads must be > 0".to_string(),
            ));
        }
        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err(AcousticError::ConfigError(
                "Dropout must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(())
    }

    /// Merge with another mel decoder parameters
    pub fn merge(&mut self, other: &MelDecoderParams) {
        self.n_layers = other.n_layers;
        self.hidden_dim = other.hidden_dim;
        self.n_heads = other.n_heads;
        self.filter_dim = other.filter_dim;
        self.dropout = other.dropout;
    }
}

impl Default for MelDecoderParams {
    fn default() -> Self {
        Self {
            n_layers: 6,
            hidden_dim: 256,
            n_heads: 2,
            filter_dim: 1024,
            dropout: 0.1,
        }
    }
}

/// Tacotron2 model parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Tacotron2Params {
    /// Encoder parameters
    pub encoder: EncoderParams,
    /// Decoder parameters
    pub decoder: AttentionDecoderParams,
    /// Postnet parameters
    pub postnet: PostnetParams,
}

impl Tacotron2Params {
    /// Validate Tacotron2 parameters
    pub fn validate(&self) -> Result<()> {
        self.encoder.validate()?;
        self.decoder.validate()?;
        self.postnet.validate()?;
        Ok(())
    }

    /// Merge with another Tacotron2 parameters
    pub fn merge(&mut self, other: &Tacotron2Params) {
        self.encoder.merge(&other.encoder);
        self.decoder.merge(&other.decoder);
        self.postnet.merge(&other.postnet);
    }
}

/// Encoder parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderParams {
    /// Embedding dimension
    pub embedding_dim: u32,
    /// Number of convolution layers
    pub n_convolutions: u32,
    /// Kernel size
    pub kernel_size: u32,
}

impl EncoderParams {
    /// Validate encoder parameters
    pub fn validate(&self) -> Result<()> {
        if self.embedding_dim == 0 {
            return Err(AcousticError::ConfigError(
                "Embedding dimension must be > 0".to_string(),
            ));
        }
        if self.n_convolutions == 0 {
            return Err(AcousticError::ConfigError(
                "Number of convolutions must be > 0".to_string(),
            ));
        }
        if self.kernel_size == 0 {
            return Err(AcousticError::ConfigError(
                "Kernel size must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Merge with another encoder parameters
    pub fn merge(&mut self, other: &EncoderParams) {
        self.embedding_dim = other.embedding_dim;
        self.n_convolutions = other.n_convolutions;
        self.kernel_size = other.kernel_size;
    }
}

impl Default for EncoderParams {
    fn default() -> Self {
        Self {
            embedding_dim: 512,
            n_convolutions: 3,
            kernel_size: 5,
        }
    }
}

/// Attention decoder parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionDecoderParams {
    /// Number of mel channels
    pub n_mel_channels: u32,
    /// Number of frames per step
    pub n_frames_per_step: u32,
    /// Encoder embedding dimension
    pub encoder_embedding_dim: u32,
    /// Attention RNN dimension
    pub attention_rnn_dim: u32,
    /// Decoder RNN dimension
    pub decoder_rnn_dim: u32,
    /// Prenet dimension
    pub prenet_dim: u32,
    /// Max decoder steps
    pub max_decoder_steps: u32,
    /// Gate threshold
    pub gate_threshold: f32,
    /// Probability threshold
    pub p_attention_dropout: f32,
    /// Decoder dropout
    pub p_decoder_dropout: f32,
}

impl AttentionDecoderParams {
    /// Validate attention decoder parameters
    pub fn validate(&self) -> Result<()> {
        if self.n_mel_channels == 0 {
            return Err(AcousticError::ConfigError(
                "Number of mel channels must be > 0".to_string(),
            ));
        }
        if self.n_frames_per_step == 0 {
            return Err(AcousticError::ConfigError(
                "Frames per step must be > 0".to_string(),
            ));
        }
        if self.max_decoder_steps == 0 {
            return Err(AcousticError::ConfigError(
                "Max decoder steps must be > 0".to_string(),
            ));
        }
        if self.gate_threshold < 0.0 || self.gate_threshold > 1.0 {
            return Err(AcousticError::ConfigError(
                "Gate threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(())
    }

    /// Merge with another attention decoder parameters
    pub fn merge(&mut self, other: &AttentionDecoderParams) {
        self.n_mel_channels = other.n_mel_channels;
        self.n_frames_per_step = other.n_frames_per_step;
        self.encoder_embedding_dim = other.encoder_embedding_dim;
        self.attention_rnn_dim = other.attention_rnn_dim;
        self.decoder_rnn_dim = other.decoder_rnn_dim;
        self.prenet_dim = other.prenet_dim;
        self.max_decoder_steps = other.max_decoder_steps;
        self.gate_threshold = other.gate_threshold;
        self.p_attention_dropout = other.p_attention_dropout;
        self.p_decoder_dropout = other.p_decoder_dropout;
    }
}

impl Default for AttentionDecoderParams {
    fn default() -> Self {
        Self {
            n_mel_channels: 80,
            n_frames_per_step: 1,
            encoder_embedding_dim: 512,
            attention_rnn_dim: 1024,
            decoder_rnn_dim: 1024,
            prenet_dim: 256,
            max_decoder_steps: 1000,
            gate_threshold: 0.5,
            p_attention_dropout: 0.1,
            p_decoder_dropout: 0.1,
        }
    }
}

/// Postnet parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostnetParams {
    /// Number of mel channels
    pub n_mel_channels: u32,
    /// Postnet embedding dimension
    pub postnet_embedding_dim: u32,
    /// Postnet kernel size
    pub postnet_kernel_size: u32,
    /// Number of postnet convolutions
    pub postnet_n_convolutions: u32,
    /// Postnet dropout
    pub postnet_dropout: f32,
}

impl PostnetParams {
    /// Validate postnet parameters
    pub fn validate(&self) -> Result<()> {
        if self.n_mel_channels == 0 {
            return Err(AcousticError::ConfigError(
                "Number of mel channels must be > 0".to_string(),
            ));
        }
        if self.postnet_embedding_dim == 0 {
            return Err(AcousticError::ConfigError(
                "Postnet embedding dimension must be > 0".to_string(),
            ));
        }
        if self.postnet_kernel_size == 0 {
            return Err(AcousticError::ConfigError(
                "Postnet kernel size must be > 0".to_string(),
            ));
        }
        if self.postnet_n_convolutions == 0 {
            return Err(AcousticError::ConfigError(
                "Number of postnet convolutions must be > 0".to_string(),
            ));
        }
        if self.postnet_dropout < 0.0 || self.postnet_dropout > 1.0 {
            return Err(AcousticError::ConfigError(
                "Postnet dropout must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(())
    }

    /// Merge with another postnet parameters
    pub fn merge(&mut self, other: &PostnetParams) {
        self.n_mel_channels = other.n_mel_channels;
        self.postnet_embedding_dim = other.postnet_embedding_dim;
        self.postnet_kernel_size = other.postnet_kernel_size;
        self.postnet_n_convolutions = other.postnet_n_convolutions;
        self.postnet_dropout = other.postnet_dropout;
    }
}

impl Default for PostnetParams {
    fn default() -> Self {
        Self {
            n_mel_channels: 80,
            postnet_embedding_dim: 512,
            postnet_kernel_size: 5,
            postnet_n_convolutions: 5,
            postnet_dropout: 0.5,
        }
    }
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model description
    pub description: String,
    /// Model author
    pub author: String,
    /// Model license
    pub license: String,
    /// Model URL
    pub url: Option<String>,
    /// Model tags
    pub tags: Vec<String>,
    /// Creation date
    pub created: Option<String>,
    /// Last modified date
    pub modified: Option<String>,
}

impl ModelMetadata {
    /// Merge with another model metadata
    pub fn merge(&mut self, other: &ModelMetadata) {
        if !other.name.is_empty() {
            self.name = other.name.clone();
        }
        if !other.description.is_empty() {
            self.description = other.description.clone();
        }
        if !other.author.is_empty() {
            self.author = other.author.clone();
        }
        if !other.license.is_empty() {
            self.license = other.license.clone();
        }
        if other.url.is_some() {
            self.url = other.url.clone();
        }
        if !other.tags.is_empty() {
            self.tags = other.tags.clone();
        }
        if other.created.is_some() {
            self.created = other.created.clone();
        }
        if other.modified.is_some() {
            self.modified = other.modified.clone();
        }
    }
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: "Unnamed Model".to_string(),
            description: "No description provided".to_string(),
            author: "Unknown".to_string(),
            license: "Unknown".to_string(),
            url: None,
            tags: Vec::new(),
            created: None,
            modified: None,
        }
    }
}

// Conversion implementations for TOML parsing
impl TryFrom<toml::Value> for ModelConfig {
    type Error = AcousticError;

    fn try_from(value: toml::Value) -> Result<Self> {
        let table = value.as_table().ok_or_else(|| {
            AcousticError::ConfigError("Expected table for model config".to_string())
        })?;

        // Parse architecture
        let architecture_str = table
            .get("architecture")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AcousticError::ConfigError("Missing architecture".to_string()))?;

        let architecture = match architecture_str {
            "vits" => ModelArchitecture::Vits,
            "fastspeech2" => ModelArchitecture::FastSpeech2,
            "tacotron2" => ModelArchitecture::Tacotron2,
            custom => ModelArchitecture::Custom(custom.to_string()),
        };

        // Parse basic fields
        let model_path = table
            .get("model_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AcousticError::ConfigError("Missing model_path".to_string()))?
            .to_string();

        let version = table
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("1.0.0")
            .to_string();

        // Parse supported languages
        let supported_languages = if let Some(langs) = table.get("supported_languages") {
            if let Some(lang_array) = langs.as_array() {
                lang_array
                    .iter()
                    .filter_map(|v| v.as_str())
                    .filter_map(|s| match s {
                        "en-US" | "en_us" | "en" => Some(LanguageCode::EnUs),
                        "en-GB" | "en_gb" => Some(LanguageCode::EnGb),
                        "ja" => Some(LanguageCode::JaJp),
                        "zh-CN" | "zh_cn" | "zh" => Some(LanguageCode::ZhCn),
                        "ko" => Some(LanguageCode::KoKr),
                        "de" => Some(LanguageCode::DeDe),
                        "fr" => Some(LanguageCode::FrFr),
                        "es" => Some(LanguageCode::EsEs),
                        _ => None,
                    })
                    .collect()
            } else {
                vec![LanguageCode::EnUs] // Default
            }
        } else {
            vec![LanguageCode::EnUs] // Default
        };

        // Parse metadata
        let mut metadata = ModelMetadata::default();
        if let Some(name) = table.get("name").and_then(|v| v.as_str()) {
            metadata.name = name.to_string();
        }
        if let Some(description) = table.get("description").and_then(|v| v.as_str()) {
            metadata.description = description.to_string();
        }
        if let Some(author) = table.get("author").and_then(|v| v.as_str()) {
            metadata.author = author.to_string();
        }
        if let Some(license) = table.get("license").and_then(|v| v.as_str()) {
            metadata.license = license.to_string();
        }
        if let Some(tags) = table.get("tags").and_then(|v| v.as_array()) {
            metadata.tags = tags
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect();
        }

        // Parse architecture parameters (simplified)
        let architecture_params =
            ArchitectureParams::default_for_architecture(architecture.clone());

        Ok(Self {
            architecture,
            model_path,
            version,
            architecture_params,
            supported_languages,
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_validation() {
        let mut config = ModelConfig::default();
        assert!(config.validate().is_ok());

        config.model_path = "".to_string();
        assert!(config.validate().is_err());

        config.model_path = "test_model".to_string();
        config.supported_languages = vec![];
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_architecture_params() {
        let vits_params = ArchitectureParams::default_for_architecture(ModelArchitecture::Vits);
        assert!(vits_params.vits.is_some());
        assert!(vits_params.fastspeech2.is_none());

        let fs2_params =
            ArchitectureParams::default_for_architecture(ModelArchitecture::FastSpeech2);
        assert!(fs2_params.fastspeech2.is_some());
        assert!(fs2_params.vits.is_none());
    }

    #[test]
    fn test_vits_params_validation() {
        let mut params = VitsParams::default();
        assert!(params.validate().is_ok());

        params.n_speakers = Some(0);
        assert!(params.validate().is_err());

        params.n_speakers = Some(256);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_text_encoder_params_validation() {
        let mut params = TextEncoderParams::default();
        assert!(params.validate().is_ok());

        params.n_layers = 0;
        assert!(params.validate().is_err());

        params.n_layers = 6;
        params.dropout = 1.5;
        assert!(params.validate().is_err());

        params.dropout = 0.1;
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_model_architecture_string() {
        assert_eq!(ModelArchitecture::Vits.as_str(), "vits");
        assert_eq!(ModelArchitecture::FastSpeech2.as_str(), "fastspeech2");
        assert_eq!(ModelArchitecture::Tacotron2.as_str(), "tacotron2");

        let custom = ModelArchitecture::Custom("custom_arch".to_string());
        assert_eq!(custom.as_str(), "custom_arch");
    }
}
