//! Prelude module with commonly used imports.

// Re-export most commonly used types and traits
pub use crate::{
    audio::{AudioBuffer, AudioMetadata},
    config::{PipelineConfig},
    error::{Result, VoirsError},
    pipeline::{VoirsPipeline, VoirsPipelineBuilder},
    traits::{AcousticModel, G2p, Vocoder},
    types::{
        AudioFormat, LanguageCode, MelSpectrogram, Phoneme, QualityLevel, 
        SpeakingStyle, SynthesisConfig, VoiceConfig, VoiceCharacteristics
    },
};

// Re-export async trait for users implementing traits
pub use async_trait::async_trait;

// Re-export commonly used std types
pub use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

// Re-export tokio types for async operations
pub use tokio::sync::RwLock;