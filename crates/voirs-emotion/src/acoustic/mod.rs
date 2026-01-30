//! Acoustic model integration for emotion control
//!
//! This module provides enhanced integration with the voirs-acoustic crate,
//! enabling emotion-aware acoustic model conditioning and synthesis.

pub mod adapter;
pub mod config;
pub mod features;
pub mod integration;
pub mod params;
pub mod synthesis;

// Re-export main types for convenient access
pub use adapter::AcousticEmotionAdapter;
pub use config::{AcousticIntegrationConfig, AcousticQualityPreset};
pub use features::{ProsodyPatterns, SpeakerEmotionFeatures, VoiceQualityProfile};
pub use integration::{
    EmotionSpeakerMapping, VocoderEmotionConfig, VoiceQualityMapping, VoirsAcousticEmotionConfig,
};
pub use params::{
    AcousticConditioningParams, AcousticEmotionMapping, ProsodyModificationParams,
    SpeakerAdaptationParams,
};
