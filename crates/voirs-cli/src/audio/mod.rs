//! Audio processing and playback modules.
//!
//! This module provides functionality for audio playback, real-time streaming,
//! and audio effects processing.

pub mod playback;
pub mod realtime;
pub mod effects;

pub use playback::{AudioPlayer, PlaybackConfig, AudioDevice, PlaybackQueue, AudioData};
pub use realtime::{RealTimeAudioStream, RealTimeStreamConfig, BufferConfig};
pub use effects::{AudioEffect, EffectChain, EffectConfig};