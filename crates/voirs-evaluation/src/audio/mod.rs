//! Comprehensive audio I/O and format support for VoiRS evaluation.
//!
//! This module provides comprehensive audio format support for loading, processing,
//! and converting various audio formats commonly used in speech evaluation.
//!
//! ## Supported Formats
//!
//! - **WAV**: Uncompressed PCM audio (via `hound`)
//! - **FLAC**: Lossless audio compression (via `claxon`)
//! - **MP3**: MPEG Layer-3 audio (via `minimp3`)
//! - **OGG**: Ogg Vorbis audio (via `lewton`)
//! - **M4A**: AAC audio in MP4 container (via `mp4parse`)
//! - **AIFF**: Audio Interchange File Format
//!
//! ## Features
//!
//! - Automatic format detection
//! - Sample rate conversion with high-quality resampling
//! - Multi-channel to mono/stereo conversion
//! - Audio normalization and gain control
//! - Streaming audio support for real-time evaluation
//! - Memory-efficient chunked processing
//!
//! ## Examples
//!
//! ```rust
//! use voirs_evaluation::audio::LoadOptions;
//! use voirs_sdk::AudioBuffer;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create test audio buffer
//! let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
//!
//! // Demonstrate load options creation
//! let options = LoadOptions::new()
//!     .target_sample_rate(16000)
//!     .target_channels(1)
//!     .normalize(true);
//!     
//! println!("Audio format options configured for {:?} Hz", options.target_sample_rate);
//! # Ok(())
//! # }
//! ```

use crate::EvaluationError;
use std::path::Path;
use voirs_sdk::AudioBuffer;

pub mod auto_conversion;
pub mod conversion;
pub mod formats;
pub mod loader;
pub mod streaming;
pub mod validation;

// Re-export key types
pub use auto_conversion::*;
pub use conversion::*;
pub use formats::*;
pub use loader::*;
pub use streaming::*;
pub use validation::*;

/// Audio format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    /// WAV format (uncompressed PCM)
    Wav,
    /// FLAC format (lossless compression)
    Flac,
    /// MP3 format (lossy compression)
    Mp3,
    /// OGG Vorbis format (lossy compression)
    Ogg,
    /// M4A format (AAC in MP4 container)
    M4a,
    /// AIFF format (uncompressed PCM)
    Aiff,
    /// Unknown or unsupported format
    Unknown,
}

impl AudioFormat {
    /// Detect audio format from file extension
    pub fn from_extension(path: &Path) -> Self {
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            match ext.to_lowercase().as_str() {
                "wav" | "wave" => Self::Wav,
                "flac" => Self::Flac,
                "mp3" => Self::Mp3,
                "ogg" => Self::Ogg,
                "m4a" | "aac" => Self::M4a,
                "aiff" | "aif" => Self::Aiff,
                _ => Self::Unknown,
            }
        } else {
            Self::Unknown
        }
    }

    /// Get common file extensions for this format
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Self::Wav => &["wav", "wave"],
            Self::Flac => &["flac"],
            Self::Mp3 => &["mp3"],
            Self::Ogg => &["ogg"],
            Self::M4a => &["m4a", "aac"],
            Self::Aiff => &["aiff", "aif"],
            Self::Unknown => &[],
        }
    }

    /// Check if format supports metadata
    pub fn supports_metadata(&self) -> bool {
        matches!(self, Self::Flac | Self::Mp3 | Self::Ogg | Self::M4a)
    }

    /// Check if format is lossless
    pub fn is_lossless(&self) -> bool {
        matches!(self, Self::Wav | Self::Flac | Self::Aiff)
    }
}

/// Audio loading options
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Target sample rate (None = keep original)
    pub target_sample_rate: Option<u32>,
    /// Target number of channels (None = keep original)
    pub target_channels: Option<u32>,
    /// Normalize audio to [-1.0, 1.0] range
    pub normalize: bool,
    /// Apply DC offset removal
    pub remove_dc_offset: bool,
    /// Start offset in seconds
    pub start_offset: Option<f64>,
    /// Duration to load in seconds (None = load all)
    pub duration: Option<f64>,
    /// Quality level for sample rate conversion (0-10)
    pub resample_quality: u8,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            target_sample_rate: None,
            target_channels: None,
            normalize: false,
            remove_dc_offset: true,
            start_offset: None,
            duration: None,
            resample_quality: 7, // High quality resampling
        }
    }
}

impl LoadOptions {
    /// Create new loading options with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set target sample rate
    pub fn target_sample_rate(mut self, sample_rate: u32) -> Self {
        self.target_sample_rate = Some(sample_rate);
        self
    }

    /// Set target number of channels
    pub fn target_channels(mut self, channels: u32) -> Self {
        self.target_channels = Some(channels);
        self
    }

    /// Enable or disable normalization
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Enable or disable DC offset removal
    pub fn remove_dc_offset(mut self, remove_dc: bool) -> Self {
        self.remove_dc_offset = remove_dc;
        self
    }

    /// Set start offset in seconds
    pub fn start_offset(mut self, offset: f64) -> Self {
        self.start_offset = Some(offset);
        self
    }

    /// Set duration to load in seconds
    pub fn duration(mut self, duration: f64) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set resampling quality (0-10, higher is better)
    pub fn resample_quality(mut self, quality: u8) -> Self {
        self.resample_quality = quality.min(10);
        self
    }
}

/// Audio metadata extracted from files
#[derive(Debug, Clone, Default)]
pub struct AudioMetadata {
    /// Title of the audio
    pub title: Option<String>,
    /// Artist name
    pub artist: Option<String>,
    /// Album name
    pub album: Option<String>,
    /// Track number
    pub track: Option<u32>,
    /// Year of release
    pub year: Option<u32>,
    /// Genre
    pub genre: Option<String>,
    /// Duration in seconds
    pub duration: Option<f64>,
    /// Bit rate (for compressed formats)
    pub bitrate: Option<u32>,
}

/// Error types specific to audio I/O operations
#[derive(Debug, thiserror::Error)]
pub enum AudioIoError {
    /// Unsupported audio format
    #[error("Unsupported audio format: {format:?}")]
    UnsupportedFormat {
        /// Audio format
        format: AudioFormat,
    },

    /// File I/O error
    #[error("File I/O error: {message}")]
    IoError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Audio decoding error
    #[error("Audio decoding error: {message}")]
    DecodingError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Invalid audio parameters
    #[error("Invalid audio parameters: {message}")]
    InvalidParameters {
        /// Error message
        message: String,
    },

    /// Conversion error
    #[error("Audio conversion error: {message}")]
    ConversionError {
        /// Error message
        message: String,
    },
}

impl From<AudioIoError> for EvaluationError {
    fn from(err: AudioIoError) -> Self {
        EvaluationError::AudioProcessingError {
            message: err.to_string(),
            source: Some(Box::new(err)),
        }
    }
}

/// Result type for audio I/O operations
pub type AudioIoResult<T> = Result<T, AudioIoError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_format_detection() {
        assert_eq!(
            AudioFormat::from_extension(Path::new("test.wav")),
            AudioFormat::Wav
        );
        assert_eq!(
            AudioFormat::from_extension(Path::new("test.flac")),
            AudioFormat::Flac
        );
        assert_eq!(
            AudioFormat::from_extension(Path::new("test.mp3")),
            AudioFormat::Mp3
        );
        assert_eq!(
            AudioFormat::from_extension(Path::new("test.ogg")),
            AudioFormat::Ogg
        );
        assert_eq!(
            AudioFormat::from_extension(Path::new("test.m4a")),
            AudioFormat::M4a
        );
        assert_eq!(
            AudioFormat::from_extension(Path::new("test.aiff")),
            AudioFormat::Aiff
        );
        assert_eq!(
            AudioFormat::from_extension(Path::new("test.xyz")),
            AudioFormat::Unknown
        );
    }

    #[test]
    fn test_format_properties() {
        assert!(AudioFormat::Wav.is_lossless());
        assert!(AudioFormat::Flac.is_lossless());
        assert!(!AudioFormat::Mp3.is_lossless());
        assert!(!AudioFormat::Ogg.is_lossless());

        assert!(!AudioFormat::Wav.supports_metadata());
        assert!(AudioFormat::Flac.supports_metadata());
        assert!(AudioFormat::Mp3.supports_metadata());
    }

    #[test]
    fn test_load_options() {
        let options = LoadOptions::new()
            .target_sample_rate(16000)
            .target_channels(1)
            .normalize(true);

        assert_eq!(options.target_sample_rate, Some(16000));
        assert_eq!(options.target_channels, Some(1));
        assert!(options.normalize);
    }

    #[test]
    fn test_load_options_defaults() {
        let options = LoadOptions::default();
        assert_eq!(options.target_sample_rate, None);
        assert_eq!(options.target_channels, None);
        assert!(!options.normalize);
        assert!(options.remove_dc_offset);
        assert_eq!(options.resample_quality, 7);
    }
}
