//! VITS (Variational Inference Text-to-Speech) model implementation
//!
//! This module contains the complete VITS architecture including:
//! - Text encoder (transformer-based)
//! - Posterior encoder (CNN-based)
//! - Normalizing flows
//! - Decoder/generator
//! - Duration predictor
//! - Style transfer capabilities
//! - Voice cloning support
//! - Emotion control

use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::{MemoryOptimizer, PerformanceMonitor, Phoneme, ProsodyController, TensorMemoryPool};

// Sub-module declarations for VITS components
// (candle-core/candle-nn are hard dependencies; see voirs-acoustic/Cargo.toml)
pub mod decoder;
pub mod duration;
pub mod flows;
pub mod loader;
pub mod posterior;
pub mod text_encoder;
pub mod trainer;

#[cfg(feature = "onnx")]
pub mod onnx_loader;

#[cfg(feature = "onnx")]
pub mod onnx_chinese;

#[cfg(feature = "onnx")]
pub mod onnx_kokoro;

// Internal module declarations for refactored components
mod acoustic_impl;
mod model_core;
mod model_synthesis;
mod style_transfer;
mod utils;
mod voice_cloning;

// Re-export main components
pub use loader::{load_vits_from_safetensors, VitsInference};
pub use text_encoder::{PhonemeEmbedding, TextEncoder, TextEncoderConfig};

// Re-export implemented components
pub use decoder::{Decoder, DecoderConfig};
pub use duration::{DurationConfig, DurationPredictor};
pub use flows::{FlowConfig, NormalizingFlows};
pub use posterior::{PosteriorConfig, PosteriorEncoder};
pub use trainer::{
    MultiPeriodDiscriminator, MultiScaleDiscriminator, TrainingMetrics, ValidationMetrics,
    VitsTrainer, VitsTrainingConfig,
};

// Re-export refactored components
pub use style_transfer::{StyleTransfer, StyleTransferConfig};
pub use voice_cloning::{SpeakerEmbedding, VoiceCloning, VoiceCloningConfig, VoiceQualityMetrics};

/// VITS model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitsConfig {
    /// Text encoder configuration
    pub text_encoder: TextEncoderConfig,
    /// Posterior encoder configuration
    pub posterior_encoder: PosteriorConfig,
    /// Duration predictor configuration
    pub duration_predictor: DurationConfig,
    /// Normalizing flows configuration
    pub flows: FlowConfig,
    /// Decoder configuration
    pub decoder: DecoderConfig,

    /// Sample rate for audio generation
    pub sample_rate: u32,
    /// Number of mel spectrogram channels
    pub mel_channels: usize,
    /// Whether the model supports multiple speakers
    pub multi_speaker: bool,
    /// Number of speakers (if multi-speaker)
    pub speaker_count: Option<usize>,
    /// Whether the model supports emotion control
    pub emotion_enabled: bool,
    /// Emotion embedding dimensions
    pub emotion_embedding_dim: Option<usize>,
    /// Number of emotion types supported
    pub emotion_count: Option<usize>,
}

impl Default for VitsConfig {
    fn default() -> Self {
        Self {
            text_encoder: TextEncoderConfig::default(),
            posterior_encoder: PosteriorConfig::default(),
            duration_predictor: DurationConfig::default(),
            flows: FlowConfig::default(),
            decoder: DecoderConfig::default(),
            sample_rate: 22050,
            mel_channels: 80,
            multi_speaker: false,
            speaker_count: None,
            emotion_enabled: true,            // Enable emotion by default
            emotion_embedding_dim: Some(256), // Default emotion embedding dimension
            emotion_count: Some(10),          // Default number of emotion types
        }
    }
}

/// Streaming state for VITS model
#[derive(Debug, Clone)]
pub struct VitsStreamingState {
    /// Pending phonemes waiting to be processed
    pub pending_phonemes: Vec<Phoneme>,
    /// Current chunk size for processing
    pub chunk_size: usize,
    /// Minimum chunk size
    pub min_chunk_size: usize,
    /// Maximum chunk size
    pub max_chunk_size: usize,
    /// Buffer size for phoneme accumulation
    pub buffer_size: usize,
    /// Total number of phonemes processed
    pub processed_count: usize,
}

/// VITS model implementation
pub struct VitsModel {
    pub(crate) config: VitsConfig,
    pub(crate) text_encoder: Arc<TextEncoder>,
    pub(crate) posterior_encoder: Arc<PosteriorEncoder>,
    pub(crate) duration_predictor: Arc<DurationPredictor>,
    pub(crate) flows: Arc<std::sync::Mutex<NormalizingFlows>>,
    pub(crate) decoder: Arc<Decoder>,
    pub(crate) device: Device,
    pub(crate) prosody_controller: Option<Arc<ProsodyController>>,
    /// Memory pool for tensor operations
    pub(crate) memory_pool: Arc<TensorMemoryPool>,
    /// Performance monitoring
    pub(crate) performance_monitor: Arc<PerformanceMonitor>,
    /// Enable performance optimizations
    pub(crate) optimization_enabled: bool,
    /// Style transfer module for voice style adaptation
    pub(crate) style_transfer: Option<Arc<StyleTransfer>>,
    /// Voice cloning module for speaker adaptation
    pub(crate) voice_cloning: Option<Arc<VoiceCloning>>,
    /// Current emotion state for synthesis
    pub(crate) current_emotion: Option<crate::config::synthesis::EmotionConfig>,
    /// Emotion embedding lookup table
    pub(crate) emotion_embeddings: Option<HashMap<String, Tensor>>,
}
