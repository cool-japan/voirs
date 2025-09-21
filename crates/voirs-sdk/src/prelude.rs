//! Prelude module with commonly used imports.

// Re-export most commonly used types and traits
pub use crate::{
    audio::{AudioBuffer, AudioMetadata},
    capabilities::{CapabilityManager, FeatureDetector},
    config::PipelineConfig,
    error::{Result, VoirsError},
    performance::{PerformanceMetrics, PerformanceMonitor},
    pipeline::{VoirsPipeline, VoirsPipelineBuilder},
    traits::{AcousticModel, G2p, Vocoder},
    types::{
        AdvancedFeature, AudioFormat, CapabilityNegotiation, CapabilityRequest,
        HardwareCapabilities, LanguageCode, MelSpectrogram, Phoneme, QualityLevel, ResourceLimits,
        SpeakingStyle, SynthesisConfig, SystemCapabilities, VoiceCharacteristics, VoiceConfig,
    },
};

// Advanced voice features (when enabled)
#[cfg(feature = "emotion")]
pub use crate::emotion::{EmotionController, EmotionControllerBuilder, EmotionStatistics};
#[cfg(feature = "emotion")]
pub use voirs_emotion::{
    Emotion, EmotionConfig, EmotionIntensity, EmotionParameters, EmotionPresetLibrary,
};

#[cfg(feature = "cloning")]
pub use crate::cloning::{CloningStatistics, ValidationResult, VoiceCloner, VoiceClonerBuilder};
#[cfg(feature = "cloning")]
pub use voirs_cloning::{
    CloningConfig, CloningMethod, SpeakerProfile, VoiceCloneResult, VoiceSample,
};

#[cfg(feature = "conversion")]
pub use crate::conversion::{
    AudioValidationResult, ConversionStatistics, VoiceConverter, VoiceConverterBuilder,
};
#[cfg(feature = "conversion")]
pub use voirs_conversion::types::{AgeGroup, Gender};
#[cfg(feature = "conversion")]
pub use voirs_conversion::{
    ConversionConfig, ConversionResult, ConversionTarget, ConversionType,
    VoiceCharacteristics as ConversionVoiceCharacteristics,
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
