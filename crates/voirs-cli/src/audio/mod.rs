//! Audio processing and playback modules.
//!
//! This module provides functionality for audio playback, real-time streaming,
//! audio effects processing, and metadata handling.

pub mod effects;
pub mod metadata;
pub mod playback;
pub mod realtime;

pub use effects::{AudioEffect, EffectChain, EffectConfig};
pub use metadata::{AlbumArt, AudioMetadata, MetadataReader, MetadataWriter, PictureType};
pub use playback::{AudioData, AudioDevice, AudioPlayer, PlaybackConfig, PlaybackQueue};
pub use realtime::{BufferConfig, RealTimeAudioStream, RealTimeStreamConfig};
