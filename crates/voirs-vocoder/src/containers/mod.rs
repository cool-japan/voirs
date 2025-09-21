//! Audio container format implementations.
//!
//! This module provides container format support for various audio formats:
//! - WAV containers (already implemented in audio::io)
//! - OGG containers for Vorbis/Opus
//! - MP4 containers for AAC/MP3
//! - FLAC containers

use crate::{AudioBuffer, Result};
use std::path::Path;

pub mod mp4;
pub mod ogg;

/// Audio container format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerFormat {
    /// WAV container
    Wav,
    /// OGG container  
    Ogg,
    /// MP4 container
    Mp4,
    /// FLAC container (native format)
    Flac,
}

/// Container configuration
#[derive(Debug, Clone)]
pub struct ContainerConfig {
    /// Container format
    pub format: ContainerFormat,
    /// Codec configuration (codec-specific)
    pub codec_config: Option<crate::codecs::CodecConfig>,
    /// Metadata (title, artist, etc.)
    pub metadata: Option<AudioMetadata>,
}

/// Audio metadata for containers
#[derive(Debug, Clone)]
pub struct AudioMetadata {
    /// Title
    pub title: Option<String>,
    /// Artist
    pub artist: Option<String>,
    /// Album
    pub album: Option<String>,
    /// Year
    pub year: Option<u32>,
    /// Genre
    pub genre: Option<String>,
    /// Track number
    pub track: Option<u32>,
    /// Duration in seconds
    pub duration: Option<f64>,
}

impl Default for ContainerConfig {
    fn default() -> Self {
        Self {
            format: ContainerFormat::Wav,
            codec_config: None,
            metadata: None,
        }
    }
}

/// Audio container writer
pub struct ContainerWriter {
    config: ContainerConfig,
}

impl ContainerWriter {
    /// Create new container writer
    pub fn new(config: ContainerConfig) -> Self {
        Self { config }
    }

    /// Write audio buffer to container file
    pub fn write_to_file<P: AsRef<Path>>(&self, audio: &AudioBuffer, path: P) -> Result<()> {
        match self.config.format {
            ContainerFormat::Wav => {
                // Use existing WAV implementation
                crate::audio::io::convenience::write_wav(audio, path)
            }
            ContainerFormat::Ogg => ogg::write_ogg_container(audio, path, &self.config),
            ContainerFormat::Mp4 => mp4::write_mp4_container(audio, path, &self.config),
            ContainerFormat::Flac => {
                // FLAC is both codec and container
                if let Some(codec_config) = &self.config.codec_config {
                    crate::codecs::flac::encode_flac(audio, path, codec_config)
                } else {
                    let default_config = crate::codecs::CodecConfig::default();
                    crate::codecs::flac::encode_flac(audio, path, &default_config)
                }
            }
        }
    }

    /// Get recommended file extension for container
    pub fn file_extension(&self) -> &'static str {
        match self.config.format {
            ContainerFormat::Wav => "wav",
            ContainerFormat::Ogg => "ogg",
            ContainerFormat::Mp4 => "mp4",
            ContainerFormat::Flac => "flac",
        }
    }

    /// Get container properties
    pub fn properties(&self) -> ContainerProperties {
        match self.config.format {
            ContainerFormat::Wav => ContainerProperties {
                supports_metadata: false,
                supports_multiple_streams: false,
                supports_chapters: false,
                max_file_size_gb: 4, // 4GB WAV limit
            },
            ContainerFormat::Ogg => ContainerProperties {
                supports_metadata: true,
                supports_multiple_streams: true,
                supports_chapters: false,
                max_file_size_gb: 0, // No practical limit
            },
            ContainerFormat::Mp4 => ContainerProperties {
                supports_metadata: true,
                supports_multiple_streams: true,
                supports_chapters: true,
                max_file_size_gb: 0, // No practical limit
            },
            ContainerFormat::Flac => ContainerProperties {
                supports_metadata: true,
                supports_multiple_streams: false,
                supports_chapters: false,
                max_file_size_gb: 0, // No practical limit
            },
        }
    }
}

/// Container properties
#[derive(Debug, Clone)]
pub struct ContainerProperties {
    /// Whether metadata is supported
    pub supports_metadata: bool,
    /// Whether multiple audio streams are supported
    pub supports_multiple_streams: bool,
    /// Whether chapter markers are supported
    pub supports_chapters: bool,
    /// Maximum file size in GB (0 = no limit)
    pub max_file_size_gb: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_config_default() {
        let config = ContainerConfig::default();
        assert_eq!(config.format, ContainerFormat::Wav);
        assert!(config.codec_config.is_none());
        assert!(config.metadata.is_none());
    }

    #[test]
    fn test_container_writer_creation() {
        let config = ContainerConfig::default();
        let writer = ContainerWriter::new(config);
        assert_eq!(writer.file_extension(), "wav");
    }

    #[test]
    fn test_container_properties() {
        let wav_config = ContainerConfig {
            format: ContainerFormat::Wav,
            ..Default::default()
        };
        let ogg_config = ContainerConfig {
            format: ContainerFormat::Ogg,
            ..Default::default()
        };

        let wav_writer = ContainerWriter::new(wav_config);
        let ogg_writer = ContainerWriter::new(ogg_config);

        let wav_props = wav_writer.properties();
        let ogg_props = ogg_writer.properties();

        assert!(!wav_props.supports_metadata);
        assert!(ogg_props.supports_metadata);
        assert!(!wav_props.supports_multiple_streams);
        assert!(ogg_props.supports_multiple_streams);
    }

    #[test]
    fn test_file_extensions() {
        let formats = vec![
            (ContainerFormat::Wav, "wav"),
            (ContainerFormat::Ogg, "ogg"),
            (ContainerFormat::Mp4, "mp4"),
            (ContainerFormat::Flac, "flac"),
        ];

        for (format, expected_ext) in formats {
            let config = ContainerConfig {
                format,
                ..Default::default()
            };
            let writer = ContainerWriter::new(config);
            assert_eq!(writer.file_extension(), expected_ext);
        }
    }
}
