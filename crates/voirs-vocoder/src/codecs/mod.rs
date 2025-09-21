//! Audio codec implementations.
//!
//! This module provides encoding support for various audio formats:
//! - MP3 encoding with LAME
//! - FLAC compression
//! - Opus encoding for streaming
//! - AAC encoding (optional)

use crate::{AudioBuffer, Result};
use std::path::Path;

pub mod aac;
pub mod flac;
pub mod mp3;
pub mod opus;

/// Audio codec type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodec {
    /// MP3 codec
    Mp3,
    /// FLAC codec  
    Flac,
    /// Opus codec
    Opus,
    /// AAC codec (optional)
    Aac,
}

/// Audio codec configuration
#[derive(Debug, Clone)]
pub struct CodecConfig {
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Bit rate (for lossy codecs)
    pub bit_rate: Option<u32>,
    /// Quality setting (codec-specific)
    pub quality: Option<f32>,
    /// Compression level (for lossless codecs)
    pub compression_level: Option<u32>,
}

impl Default for CodecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            channels: 1,
            bit_rate: Some(128000),     // 128 kbps default
            quality: Some(0.5),         // Medium quality
            compression_level: Some(5), // Medium compression
        }
    }
}

/// Audio codec encoder
pub struct AudioCodecEncoder {
    codec: AudioCodec,
    config: CodecConfig,
}

impl AudioCodecEncoder {
    /// Create new codec encoder
    pub fn new(codec: AudioCodec, config: CodecConfig) -> Self {
        Self { codec, config }
    }

    /// Encode audio buffer to file
    pub fn encode_to_file<P: AsRef<Path>>(&self, audio: &AudioBuffer, path: P) -> Result<()> {
        match self.codec {
            AudioCodec::Mp3 => mp3::encode_mp3(audio, path, &self.config),
            AudioCodec::Flac => flac::encode_flac(audio, path, &self.config),
            AudioCodec::Opus => opus::encode_opus(audio, path, &self.config),
            AudioCodec::Aac => aac::encode_aac(audio, path, &self.config),
        }
    }

    /// Encode audio buffer to bytes
    pub fn encode_to_bytes(&self, audio: &AudioBuffer) -> Result<Vec<u8>> {
        match self.codec {
            AudioCodec::Mp3 => mp3::encode_mp3_bytes(audio, &self.config),
            AudioCodec::Flac => flac::encode_flac_bytes(audio, &self.config),
            AudioCodec::Opus => opus::encode_opus_bytes(audio, &self.config),
            AudioCodec::Aac => aac::encode_aac_bytes(audio, &self.config),
        }
    }

    /// Get recommended file extension for codec
    pub fn file_extension(&self) -> &'static str {
        match self.codec {
            AudioCodec::Mp3 => "mp3",
            AudioCodec::Flac => "flac",
            AudioCodec::Opus => "opus",
            AudioCodec::Aac => "aac",
        }
    }

    /// Get codec properties
    pub fn properties(&self) -> CodecProperties {
        match self.codec {
            AudioCodec::Mp3 => CodecProperties {
                is_lossless: false,
                supports_variable_bitrate: true,
                max_channels: 2,
                max_sample_rate: 48000,
                typical_compression_ratio: 10.0,
            },
            AudioCodec::Flac => CodecProperties {
                is_lossless: true,
                supports_variable_bitrate: false,
                max_channels: 8,
                max_sample_rate: 192000,
                typical_compression_ratio: 2.0,
            },
            AudioCodec::Opus => CodecProperties {
                is_lossless: false,
                supports_variable_bitrate: true,
                max_channels: 2,
                max_sample_rate: 48000,
                typical_compression_ratio: 15.0,
            },
            AudioCodec::Aac => CodecProperties {
                is_lossless: false,
                supports_variable_bitrate: true,
                max_channels: 8,
                max_sample_rate: 96000,
                typical_compression_ratio: 12.0,
            },
        }
    }
}

/// Codec properties
#[derive(Debug, Clone)]
pub struct CodecProperties {
    /// Whether the codec is lossless
    pub is_lossless: bool,
    /// Whether variable bitrate is supported
    pub supports_variable_bitrate: bool,
    /// Maximum number of channels supported
    pub max_channels: u16,
    /// Maximum sample rate supported
    pub max_sample_rate: u32,
    /// Typical compression ratio (uncompressed:compressed)
    pub typical_compression_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_config_default() {
        let config = CodecConfig::default();
        assert_eq!(config.sample_rate, 22050);
        assert_eq!(config.channels, 1);
        assert_eq!(config.bit_rate, Some(128000));
        assert_eq!(config.quality, Some(0.5));
        assert_eq!(config.compression_level, Some(5));
    }

    #[test]
    fn test_encoder_creation() {
        let config = CodecConfig::default();
        let encoder = AudioCodecEncoder::new(AudioCodec::Mp3, config);
        assert_eq!(encoder.file_extension(), "mp3");
    }

    #[test]
    fn test_codec_properties() {
        let config = CodecConfig::default();
        let mp3_encoder = AudioCodecEncoder::new(AudioCodec::Mp3, config.clone());
        let flac_encoder = AudioCodecEncoder::new(AudioCodec::Flac, config);

        let mp3_props = mp3_encoder.properties();
        let flac_props = flac_encoder.properties();

        assert!(!mp3_props.is_lossless);
        assert!(flac_props.is_lossless);
        assert!(mp3_props.supports_variable_bitrate);
        assert!(!flac_props.supports_variable_bitrate);
    }
}
