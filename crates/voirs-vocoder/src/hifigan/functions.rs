//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "candle")]
use crate::models::hifigan::{generator::HiFiGanGenerator, inference::HiFiGanInference};
pub use crate::models::hifigan::{HiFiGanConfig, HiFiGanVariant, HiFiGanVariants};
use crate::{
    conditioning::{VocoderConditioner, VocoderConditioningConfig},
    conversion::{VoiceConversionConfig, VoiceConverter},
    effects::{EffectChain, EffectPresets},
    AudioBuffer, MelSpectrogram, Result, SynthesisConfig, Vocoder, VocoderError, VocoderFeature,
    VocoderMetadata,
};
use async_trait::async_trait;
use futures::Stream;
