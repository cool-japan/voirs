//! # Audio Format Support
//!
//! Comprehensive audio format handling and conversion utilities for the VoiRS recognition system.
//! This module provides unified loading capabilities for various audio formats with automatic
//! format detection, high-quality resampling, and intelligent preprocessing.
//!
//! ## Supported Formats
//!
//! - **WAV**: Uncompressed audio with support for 8, 16, 24, 32-bit integer and 32-bit float
//! - **FLAC**: Lossless compression with full bit-depth and sample rate support
//! - **MP3**: MPEG Layer 3 compressed audio with CBR/VBR support
//! - **OGG**: Ogg Vorbis compressed audio
//! - **M4A/AAC**: MPEG-4 audio container with AAC compression using Symphonia decoder
//!
//! ## Features
//!
//! - **Automatic Format Detection**: Smart detection from file extensions, MIME types, and binary headers
//! - **Universal Audio Loader**: Single API for loading any supported format
//! - **High-Quality Resampling**: Multiple algorithms (linear, cubic, sinc) with configurable quality
//! - **Audio Preprocessing**: Sample rate conversion, mono mixing, normalization, DC offset removal
//! - **Memory Efficient**: Streaming support and configurable duration limits
//!
//! ## Quick Example
//!
//! ```rust,no_run
//! use voirs_recognizer::audio_formats::{load_audio, AudioLoadConfig, UniversalAudioLoader};
//!
//! // Quick loading with ASR-optimized defaults (16kHz mono, normalized)
//! let audio = load_audio("speech.m4a")?;
//! println!("Loaded {} samples at {}Hz", audio.samples().len(), audio.sample_rate());
//!
//! // Custom configuration for high-quality analysis
//! let config = AudioLoadConfig {
//!     target_sample_rate: Some(44100),
//!     force_mono: false,
//!     normalize: false,
//!     remove_dc: true,
//!     max_duration_seconds: Some(30.0),
//! };
//! let loader = UniversalAudioLoader::with_config(config);
//! let stereo_audio = loader.load_from_path("music.flac")?;
//! # Ok::<(), voirs_recognizer::RecognitionError>(())
//! ```

use crate::RecognitionError;
use std::path::Path;
use voirs_sdk::AudioBuffer;

pub mod detection;
pub mod loaders;
pub mod resampling;

pub use detection::*;
pub use loaders::*;
pub use resampling::*;

/// Supported audio formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    /// WAV format (uncompressed)
    Wav,
    /// FLAC format (lossless compression)
    Flac,
    /// MP3 format (lossy compression)
    Mp3,
    /// OGG Vorbis format
    Ogg,
    /// M4A format (AAC)
    M4a,
    /// Unknown or unsupported format
    Unknown,
}

impl AudioFormat {
    /// Get file extensions for this format
    #[must_use]
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            AudioFormat::Wav => &["wav", "wave"],
            AudioFormat::Flac => &["flac"],
            AudioFormat::Mp3 => &["mp3"],
            AudioFormat::Ogg => &["ogg", "oga"],
            AudioFormat::M4a => &["m4a", "aac", "mp4"],
            AudioFormat::Unknown => &[],
        }
    }

    /// Get MIME types for this format
    #[must_use]
    pub fn mime_types(&self) -> &'static [&'static str] {
        match self {
            AudioFormat::Wav => &["audio/wav", "audio/wave", "audio/x-wav"],
            AudioFormat::Flac => &["audio/flac", "audio/x-flac"],
            AudioFormat::Mp3 => &["audio/mpeg", "audio/mp3"],
            AudioFormat::Ogg => &["audio/ogg", "application/ogg"],
            AudioFormat::M4a => &["audio/mp4", "audio/aac"],
            AudioFormat::Unknown => &[],
        }
    }
}

/// Configuration for audio loading and preprocessing.
///
/// `AudioLoadConfig` controls how audio files are processed during loading,
/// including sample rate conversion, channel mixing, normalization, and
/// duration limits. The default configuration is optimized for automatic
/// speech recognition (ASR) applications.
///
/// ## Default Configuration
///
/// The default settings are optimized for ASR compatibility:
/// - **Target sample rate**: 16kHz (standard for speech recognition)
/// - **Force mono**: Enabled (reduces computational complexity)
/// - **Normalize**: Enabled (ensures consistent amplitude levels)
/// - **Remove DC**: Enabled (removes DC offset artifacts)
/// - **Max duration**: None (load entire file)
///
/// ## Example Configurations
///
/// ```rust
/// use voirs_recognizer::audio_formats::AudioLoadConfig;
///
/// // ASR-optimized (default)
/// let asr_config = AudioLoadConfig::default();
///
/// // High-quality music analysis
/// let music_config = AudioLoadConfig {
///     target_sample_rate: Some(44100),
///     force_mono: false,
///     normalize: false,
///     remove_dc: true,
///     max_duration_seconds: None,
/// };
///
/// // Low-latency streaming (30-second chunks)
/// let streaming_config = AudioLoadConfig {
///     target_sample_rate: Some(16000),
///     force_mono: true,
///     normalize: true,
///     remove_dc: true,
///     max_duration_seconds: Some(30.0),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct AudioLoadConfig {
    /// Target sample rate (if None, use original)
    pub target_sample_rate: Option<u32>,
    /// Convert to mono (if false, keep original channels)
    pub force_mono: bool,
    /// Normalize audio amplitude to [-1.0, 1.0]
    pub normalize: bool,
    /// Apply DC offset removal
    pub remove_dc: bool,
    /// Maximum duration to load (None for full file)
    pub max_duration_seconds: Option<f32>,
}

impl Default for AudioLoadConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: Some(16000), // Standard for ASR
            force_mono: true,
            normalize: true,
            remove_dc: true,
            max_duration_seconds: None,
        }
    }
}

/// Universal audio loader with automatic format detection and preprocessing.
///
/// The `UniversalAudioLoader` provides a unified interface for loading audio files
/// of any supported format. It automatically detects the audio format from file
/// extensions, MIME types, or binary content headers, then delegates to the
/// appropriate specialized loader.
///
/// ## Features
///
/// - Automatic format detection for WAV, FLAC, MP3, OGG, and M4A/AAC files
/// - Configurable audio preprocessing (resampling, mono conversion, normalization)
/// - Memory-efficient loading with optional duration limits
/// - Consistent error handling across all formats
/// - Support for both file paths and byte buffers
///
/// ## Configuration
///
/// The loader accepts an [`AudioLoadConfig`] that controls:
/// - Target sample rate (default: 16kHz for ASR compatibility)
/// - Mono conversion (default: enabled)
/// - Audio normalization (default: enabled)
/// - DC offset removal (default: enabled)
/// - Maximum duration limits
///
/// ## Example
///
/// ```rust,no_run
/// use voirs_recognizer::audio_formats::{UniversalAudioLoader, AudioLoadConfig};
///
/// // Create loader with default ASR-optimized settings
/// let loader = UniversalAudioLoader::new();
/// let audio = loader.load_from_path("speech.m4a")?;
///
/// // Custom configuration for music analysis
/// let config = AudioLoadConfig {
///     target_sample_rate: Some(44100),
///     force_mono: false,
///     normalize: false,
///     ..Default::default()
/// };
/// let music_loader = UniversalAudioLoader::with_config(config);
/// let stereo_audio = music_loader.load_from_path("music.flac")?;
/// # Ok::<(), voirs_recognizer::RecognitionError>(())
/// ```
pub struct UniversalAudioLoader {
    config: AudioLoadConfig,
}

impl UniversalAudioLoader {
    /// Create a new universal audio loader with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: AudioLoadConfig::default(),
        }
    }

    /// Create a new universal audio loader with custom configuration
    #[must_use]
    pub fn with_config(config: AudioLoadConfig) -> Self {
        Self { config }
    }

    /// Load audio from a file path with automatic format detection.
    ///
    /// This method automatically detects the audio format from the file extension
    /// and delegates to the appropriate specialized loader. Supported formats include
    /// WAV, FLAC, MP3, OGG, and M4A/AAC.
    ///
    /// The audio is processed according to the loader's configuration, which may
    /// include resampling, mono conversion, normalization, and DC offset removal.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the audio file (supports any type that implements `AsRef<Path>`)
    ///
    /// # Returns
    ///
    /// Returns an [`AudioBuffer`] containing the loaded and processed audio data.
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - The file cannot be read or doesn't exist
    /// - The audio format is not supported or cannot be detected
    /// - The file is corrupted or contains invalid audio data
    /// - Audio processing (resampling, conversion) fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use voirs_recognizer::audio_formats::UniversalAudioLoader;
    ///
    /// let loader = UniversalAudioLoader::new();
    ///
    /// // Load various formats - format detection is automatic
    /// let wav_audio = loader.load_from_path("speech.wav")?;
    /// let mp3_audio = loader.load_from_path("music.mp3")?;
    /// let m4a_audio = loader.load_from_path("podcast.m4a")?;
    ///
    /// println!("Loaded {} samples at {}Hz",
    ///          wav_audio.samples().len(), wav_audio.sample_rate());
    /// # Ok::<(), voirs_recognizer::RecognitionError>(())
    /// ```
    pub fn load_from_path<P: AsRef<Path>>(&self, path: P) -> Result<AudioBuffer, RecognitionError> {
        let path = path.as_ref();
        let format = detect_format_from_path(path)?;

        match format {
            AudioFormat::Wav => WavLoader::new(self.config.clone()).load_from_path(path),
            AudioFormat::Flac => FlacLoader::new(self.config.clone()).load_from_path(path),
            AudioFormat::Mp3 => Mp3Loader::new(self.config.clone()).load_from_path(path),
            AudioFormat::Ogg => OggLoader::new(self.config.clone()).load_from_path(path),
            AudioFormat::M4a => M4aLoader::new(self.config.clone()).load_from_path(path),
            AudioFormat::Unknown => Err(RecognitionError::UnsupportedFormat(format!(
                "Unknown audio format for file: {path:?}"
            ))),
        }
    }

    /// Load audio from a byte buffer with format detection.
    ///
    /// This method loads audio from raw byte data, either using a provided format
    /// hint or by analyzing the binary content to detect the format automatically.
    /// This is useful for loading audio from network streams, embedded resources,
    /// or when the file extension is unavailable.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw audio file data as a byte slice
    /// * `hint` - Optional format hint to skip detection (use [`None`] for auto-detection)
    ///
    /// # Returns
    ///
    /// Returns an [`AudioBuffer`] containing the loaded and processed audio data.
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - The data cannot be parsed as a valid audio file
    /// - The audio format is not supported or cannot be detected
    /// - The data is corrupted or incomplete
    /// - Audio processing (resampling, conversion) fails
    ///
    /// # Performance Note
    ///
    /// Providing a format hint via the `hint` parameter can improve performance
    /// by skipping the format detection step.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use voirs_recognizer::audio_formats::{UniversalAudioLoader, AudioFormat};
    /// use std::fs;
    ///
    /// let loader = UniversalAudioLoader::new();
    ///
    /// // Load with automatic format detection
    /// let audio_data = fs::read("audio.mp3")?;
    /// let audio = loader.load_from_bytes(&audio_data, None)?;
    ///
    /// // Load with format hint for better performance
    /// let wav_data = fs::read("speech.wav")?;
    /// let wav_audio = loader.load_from_bytes(&wav_data, Some(AudioFormat::Wav))?;
    ///
    /// println!("Loaded audio: {} channels, {} samples",
    ///          audio.channels(), audio.samples().len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn load_from_bytes(
        &self,
        data: &[u8],
        hint: Option<AudioFormat>,
    ) -> Result<AudioBuffer, RecognitionError> {
        let format = if let Some(format) = hint {
            format
        } else {
            detect_format_from_bytes(data)?
        };

        match format {
            AudioFormat::Wav => WavLoader::new(self.config.clone()).load_from_bytes(data),
            AudioFormat::Flac => FlacLoader::new(self.config.clone()).load_from_bytes(data),
            AudioFormat::Mp3 => Mp3Loader::new(self.config.clone()).load_from_bytes(data),
            AudioFormat::Ogg => OggLoader::new(self.config.clone()).load_from_bytes(data),
            AudioFormat::M4a => M4aLoader::new(self.config.clone()).load_from_bytes(data),
            AudioFormat::Unknown => Err(RecognitionError::UnsupportedFormat(
                "Could not detect audio format from bytes".to_string(),
            )),
        }
    }
}

impl Default for UniversalAudioLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Load audio with default ASR-optimized settings.
///
/// This is a convenience function that creates a [`UniversalAudioLoader`] with
/// default configuration and loads the specified audio file. The default settings
/// are optimized for automatic speech recognition:
///
/// - 16kHz sample rate
/// - Mono (single channel)
/// - Normalized amplitude
/// - DC offset removal
///
/// # Arguments
///
/// * `path` - Path to the audio file
///
/// # Returns
///
/// Returns an [`AudioBuffer`] with processed audio ready for ASR.
///
/// # Example
///
/// ```rust,no_run
/// use voirs_recognizer::audio_formats::load_audio;
///
/// // Quick loading for ASR applications
/// let audio = load_audio("speech.wav")?;
/// assert_eq!(audio.sample_rate(), 16000);
/// assert_eq!(audio.channels(), 1);
/// # Ok::<(), voirs_recognizer::RecognitionError>(())
/// ```
pub fn load_audio<P: AsRef<Path>>(path: P) -> Result<AudioBuffer, RecognitionError> {
    UniversalAudioLoader::new().load_from_path(path)
}

/// Load audio with a custom sample rate.
///
/// This convenience function loads audio with a specified sample rate while
/// keeping other settings at their ASR-optimized defaults (mono, normalized,
/// DC offset removal).
///
/// # Arguments
///
/// * `path` - Path to the audio file
/// * `sample_rate` - Target sample rate in Hz
///
/// # Returns
///
/// Returns an [`AudioBuffer`] resampled to the specified sample rate.
///
/// # Example
///
/// ```rust,no_run
/// use voirs_recognizer::audio_formats::load_audio_with_sample_rate;
///
/// // Load audio at 22kHz instead of default 16kHz
/// let audio = load_audio_with_sample_rate("music.flac", 22050)?;
/// assert_eq!(audio.sample_rate(), 22050);
/// # Ok::<(), voirs_recognizer::RecognitionError>(())
/// ```
pub fn load_audio_with_sample_rate<P: AsRef<Path>>(
    path: P,
    sample_rate: u32,
) -> Result<AudioBuffer, RecognitionError> {
    let config = AudioLoadConfig {
        target_sample_rate: Some(sample_rate),
        ..Default::default()
    };
    UniversalAudioLoader::with_config(config).load_from_path(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_format_extensions() {
        assert!(AudioFormat::Wav.extensions().contains(&"wav"));
        assert!(AudioFormat::Flac.extensions().contains(&"flac"));
        assert!(AudioFormat::Mp3.extensions().contains(&"mp3"));
        assert!(AudioFormat::Ogg.extensions().contains(&"ogg"));
    }

    #[test]
    fn test_default_config() {
        let config = AudioLoadConfig::default();
        assert_eq!(config.target_sample_rate, Some(16000));
        assert!(config.force_mono);
        assert!(config.normalize);
        assert!(config.remove_dc);
    }

    #[test]
    fn test_universal_loader_creation() {
        let loader = UniversalAudioLoader::new();
        assert_eq!(loader.config.target_sample_rate, Some(16000));

        let custom_config = AudioLoadConfig {
            target_sample_rate: Some(22050),
            force_mono: false,
            ..Default::default()
        };
        let custom_loader = UniversalAudioLoader::with_config(custom_config.clone());
        assert_eq!(custom_loader.config.target_sample_rate, Some(22050));
        assert!(!custom_loader.config.force_mono);
    }
}
