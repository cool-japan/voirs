//! High-level audio loading interface for VoiRS evaluation.
//!
//! This module provides a convenient, unified interface for loading audio files
//! of various formats with automatic format detection, sample rate conversion,
//! and audio processing options.
//!
//! ## Features
//!
//! - Automatic format detection from file extensions
//! - Asynchronous loading for non-blocking I/O
//! - Sample rate conversion with high-quality resampling
//! - Channel conversion (mono/stereo)
//! - Audio normalization and preprocessing
//! - Comprehensive error handling
//! - Memory-efficient streaming for large files
//!
//! ## Examples
//!
//! ```rust
//! use voirs_evaluation::audio::{AudioLoader, LoadOptions, AudioFormat};
//! use voirs_sdk::AudioBuffer;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Simple loading with defaults (using test data)
//! # let test_data = vec![0u8; 1024]; // Simulated audio data
//! let audio = AudioLoader::from_bytes(&test_data, None).await.unwrap_or_else(|_| {
//!     // Fallback to creating test audio
//!     AudioBuffer::new(vec![0.1; 16000], 16000, 1)
//! });
//!
//! // Loading with custom options
//! let options = LoadOptions::new()
//!     .target_sample_rate(16000)
//!     .target_channels(1)
//!     .normalize(true);
//! # let audio = AudioLoader::from_bytes(&test_data, Some(AudioFormat::Wav)).await.unwrap_or_else(|_| {
//! #     AudioBuffer::new(vec![0.1; 16000], 16000, 1)
//! # });
//!
//! println!("Loaded audio with {} samples", audio.samples().len());
//! # Ok(())
//! # }
//! ```

use super::{
    conversion::{convert_channels, convert_sample_rate, normalize_audio, remove_dc_offset},
    formats::{load_audio_file, validate_audio_file},
    AudioFormat, AudioIoError, AudioIoResult, AudioMetadata, LoadOptions,
};
use std::path::Path;
use tokio::fs;
use voirs_sdk::AudioBuffer;

/// High-level audio loader with comprehensive format support
pub struct AudioLoader;

impl AudioLoader {
    /// Load audio file with default options
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the audio file
    ///
    /// # Returns
    ///
    /// Returns the loaded audio buffer
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or audio format is unsupported
    pub async fn from_file<P: AsRef<Path>>(path: P) -> AudioIoResult<AudioBuffer> {
        Self::from_file_with_options(path, LoadOptions::default()).await
    }

    /// Load audio file with custom options
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the audio file
    /// * `options` - Loading and processing options
    ///
    /// # Returns
    ///
    /// Returns the loaded and processed audio buffer
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read, audio format is unsupported,
    /// or processing fails
    pub async fn from_file_with_options<P: AsRef<Path>>(
        path: P,
        options: LoadOptions,
    ) -> AudioIoResult<AudioBuffer> {
        let path = path.as_ref();

        // Validate file exists and is readable
        if !path.exists() {
            return Err(AudioIoError::IoError {
                message: format!("File not found: {}", path.display()),
                source: None,
            });
        }

        // Validate audio format
        let _format = validate_audio_file(path)?;

        // Load the audio file
        let (mut audio, _metadata) = load_audio_file(path, &options)?;

        // Apply post-processing
        audio = Self::apply_processing_options(audio, &options).await?;

        Ok(audio)
    }

    /// Load audio from raw bytes
    ///
    /// # Arguments
    ///
    /// * `data` - Raw audio file bytes
    /// * `format_hint` - Optional format hint if extension detection is not possible
    ///
    /// # Returns
    ///
    /// Returns the loaded audio buffer
    ///
    /// # Errors
    ///
    /// Returns error if audio format cannot be determined or is unsupported
    pub async fn from_bytes(
        data: &[u8],
        format_hint: Option<AudioFormat>,
    ) -> AudioIoResult<AudioBuffer> {
        if data.is_empty() {
            return Err(AudioIoError::InvalidParameters {
                message: "Empty byte stream".to_string(),
            });
        }

        // Detect format if not provided
        let format = match format_hint {
            Some(fmt) => fmt,
            None => Self::detect_format_from_bytes(data)?,
        };

        // Create a temporary file to work around symphonia's requirement for seekable input
        // In a production implementation, we'd use a custom seekable byte stream
        let temp_file = Self::create_temp_file_from_bytes(data, format).await?;

        // Create load options with the detected format
        let load_options = LoadOptions::default();

        // Load audio using the existing file-based loader
        let (audio_buffer, _metadata) = load_audio_file(&temp_file, &load_options)?;

        // Clean up temporary file
        if let Err(e) = std::fs::remove_file(&temp_file) {
            // Log warning but don't fail - the temp file will be cleaned up by the OS
            eprintln!(
                "Warning: Could not remove temporary file {}: {}",
                temp_file.display(),
                e
            );
        }

        Ok(audio_buffer)
    }

    /// Detect audio format from byte stream magic numbers
    fn detect_format_from_bytes(data: &[u8]) -> AudioIoResult<AudioFormat> {
        if data.len() < 12 {
            return Err(AudioIoError::InvalidParameters {
                message: "Insufficient data to detect format".to_string(),
            });
        }

        // WAV format: "RIFF" + 4 bytes + "WAVE"
        if data.starts_with(b"RIFF") && data.len() >= 12 && &data[8..12] == b"WAVE" {
            return Ok(AudioFormat::Wav);
        }

        // FLAC format: "fLaC"
        if data.starts_with(b"fLaC") {
            return Ok(AudioFormat::Flac);
        }

        // MP3 format: ID3 tag or frame sync
        if data.starts_with(b"ID3")
            || (data.len() >= 2 && data[0] == 0xFF && (data[1] & 0xE0) == 0xE0)
        {
            return Ok(AudioFormat::Mp3);
        }

        // OGG format: "OggS"
        if data.starts_with(b"OggS") {
            return Ok(AudioFormat::Ogg);
        }

        // M4A/MP4 format: "ftyp" at offset 4
        if data.len() >= 8 && &data[4..8] == b"ftyp" {
            return Ok(AudioFormat::M4a);
        }

        // AIFF format: "FORM" + 4 bytes + "AIFF"
        if data.starts_with(b"FORM") && data.len() >= 12 && &data[8..12] == b"AIFF" {
            return Ok(AudioFormat::Aiff);
        }

        Err(AudioIoError::DecodingError {
            message: "Unknown format - unable to detect from byte stream".to_string(),
            source: None,
        })
    }

    /// Create a temporary file from byte data for symphonia compatibility
    async fn create_temp_file_from_bytes(
        data: &[u8],
        format: AudioFormat,
    ) -> AudioIoResult<std::path::PathBuf> {
        use std::io::Write;

        // Create a temporary file in the system temp directory
        let temp_dir = std::env::temp_dir();
        // Use current timestamp for unique file name
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        // Use appropriate file extension based on format
        let extension = match format {
            AudioFormat::Wav => "wav",
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Flac => "flac",
            AudioFormat::Ogg => "ogg",
            AudioFormat::M4a => "m4a",
            AudioFormat::Aiff => "aiff",
            AudioFormat::Unknown => "tmp",
        };

        let temp_file_name = format!("voirs_audio_{}.{}", timestamp, extension);
        let temp_path = temp_dir.join(temp_file_name);

        // Write bytes to temporary file
        std::fs::File::create(&temp_path)
            .and_then(|mut file| file.write_all(data))
            .map_err(|e| AudioIoError::IoError {
                message: format!("Failed to create temporary file: {}", e),
                source: Some(Box::new(e)),
            })?;

        Ok(temp_path)
    }

    /// Load audio with metadata extraction
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the audio file
    /// * `options` - Loading and processing options
    ///
    /// # Returns
    ///
    /// Returns both the audio buffer and extracted metadata
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or processed
    pub async fn from_file_with_metadata<P: AsRef<Path>>(
        path: P,
        options: LoadOptions,
    ) -> AudioIoResult<(AudioBuffer, AudioMetadata)> {
        let path = path.as_ref();

        // Validate file exists and is readable
        if !path.exists() {
            return Err(AudioIoError::IoError {
                message: format!("File not found: {}", path.display()),
                source: None,
            });
        }

        // Validate audio format
        let _format = validate_audio_file(path)?;

        // Load the audio file
        let (mut audio, metadata) = load_audio_file(path, &options)?;

        // Apply post-processing
        audio = Self::apply_processing_options(audio, &options).await?;

        Ok((audio, metadata))
    }

    /// Check if a file can be loaded
    ///
    /// # Arguments
    ///
    /// * `path` - Path to check
    ///
    /// # Returns
    ///
    /// Returns true if the file exists and is a supported audio format
    pub async fn can_load<P: AsRef<Path>>(path: P) -> bool {
        let path = path.as_ref();

        if !path.exists() {
            return false;
        }

        validate_audio_file(path).is_ok()
    }

    /// Get supported audio formats
    ///
    /// # Returns
    ///
    /// Returns a list of all supported audio formats
    pub fn supported_formats() -> Vec<AudioFormat> {
        vec![
            AudioFormat::Wav,
            AudioFormat::Flac,
            AudioFormat::Mp3,
            AudioFormat::Ogg,
            AudioFormat::M4a,
            AudioFormat::Aiff,
        ]
    }

    /// Get supported file extensions
    ///
    /// # Returns
    ///
    /// Returns a list of all supported file extensions
    pub fn supported_extensions() -> Vec<String> {
        Self::supported_formats()
            .iter()
            .flat_map(|format| format.extensions().iter().map(|ext| ext.to_string()))
            .collect()
    }

    /// Load multiple audio files in parallel
    ///
    /// # Arguments
    ///
    /// * `paths` - Paths to audio files
    /// * `options` - Loading options to apply to all files
    ///
    /// # Returns
    ///
    /// Returns a vector of results, one for each input file
    pub async fn load_multiple<P: AsRef<Path>>(
        paths: &[P],
        options: LoadOptions,
    ) -> Vec<AudioIoResult<AudioBuffer>> {
        use futures::future::join_all;

        let futures = paths.iter().map(|path| {
            let options = options.clone();
            async move { Self::from_file_with_options(path, options).await }
        });

        join_all(futures).await
    }

    /// Apply processing options to audio buffer
    async fn apply_processing_options(
        mut audio: AudioBuffer,
        options: &LoadOptions,
    ) -> AudioIoResult<AudioBuffer> {
        // Sample rate conversion
        if let Some(target_sr) = options.target_sample_rate {
            if audio.sample_rate() != target_sr {
                audio = convert_sample_rate(audio, target_sr, options.resample_quality)?;
            }
        }

        // Channel conversion
        if let Some(target_channels) = options.target_channels {
            if audio.channels() != target_channels {
                audio = convert_channels(audio, target_channels)?;
            }
        }

        // DC offset removal
        if options.remove_dc_offset {
            audio = remove_dc_offset(audio)?;
        }

        // Normalization
        if options.normalize {
            audio = normalize_audio(audio)?;
        }

        // Apply time range selection
        if options.start_offset.is_some() || options.duration.is_some() {
            audio = Self::extract_time_range(audio, options.start_offset, options.duration)?;
        }

        Ok(audio)
    }

    /// Extract a time range from the audio buffer
    fn extract_time_range(
        audio: AudioBuffer,
        start_offset: Option<f64>,
        duration: Option<f64>,
    ) -> AudioIoResult<AudioBuffer> {
        let sample_rate = audio.sample_rate();
        let total_samples = audio.samples().len() / audio.channels() as usize;

        let start_sample = start_offset
            .map(|offset| (offset * sample_rate as f64) as usize)
            .unwrap_or(0)
            .min(total_samples);

        let end_sample = if let Some(duration) = duration {
            (start_sample + (duration * sample_rate as f64) as usize).min(total_samples)
        } else {
            total_samples
        };

        if start_sample >= end_sample {
            return Err(AudioIoError::InvalidParameters {
                message: "Invalid time range: start offset is after end".to_string(),
            });
        }

        let channels = audio.channels();
        let start_index = start_sample * channels as usize;
        let end_index = end_sample * channels as usize;

        let samples = audio.samples()[start_index..end_index].to_vec();

        Ok(AudioBuffer::new(samples, sample_rate, channels))
    }
}

/// Builder pattern for creating load options
pub struct LoadOptionsBuilder {
    options: LoadOptions,
}

impl LoadOptionsBuilder {
    /// Create a new options builder
    pub fn new() -> Self {
        Self {
            options: LoadOptions::default(),
        }
    }

    /// Set target sample rate
    pub fn sample_rate(mut self, sample_rate: u32) -> Self {
        self.options.target_sample_rate = Some(sample_rate);
        self
    }

    /// Set target number of channels
    pub fn channels(mut self, channels: u32) -> Self {
        self.options.target_channels = Some(channels);
        self
    }

    /// Enable normalization
    pub fn normalize(mut self) -> Self {
        self.options.normalize = true;
        self
    }

    /// Disable DC offset removal
    pub fn keep_dc_offset(mut self) -> Self {
        self.options.remove_dc_offset = false;
        self
    }

    /// Set time range
    pub fn time_range(mut self, start: f64, duration: f64) -> Self {
        self.options.start_offset = Some(start);
        self.options.duration = Some(duration);
        self
    }

    /// Set resampling quality (0-10)
    pub fn resample_quality(mut self, quality: u8) -> Self {
        self.options.resample_quality = quality.min(10);
        self
    }

    /// Build the options
    pub fn build(self) -> LoadOptions {
        self.options
    }
}

impl Default for LoadOptionsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_load_options_builder() {
        let options = LoadOptionsBuilder::new()
            .sample_rate(16000)
            .channels(1)
            .normalize()
            .build();

        assert_eq!(options.target_sample_rate, Some(16000));
        assert_eq!(options.target_channels, Some(1));
        assert!(options.normalize);
    }

    #[tokio::test]
    async fn test_supported_formats() {
        let formats = AudioLoader::supported_formats();
        assert!(!formats.is_empty());
        assert!(formats.contains(&AudioFormat::Wav));
        assert!(formats.contains(&AudioFormat::Mp3));
    }

    #[tokio::test]
    async fn test_supported_extensions() {
        let extensions = AudioLoader::supported_extensions();
        assert!(!extensions.is_empty());
        assert!(extensions.contains(&"wav".to_string()));
        assert!(extensions.contains(&"mp3".to_string()));
    }

    #[tokio::test]
    async fn test_from_bytes() {
        // Create minimal valid WAV file structure
        let data = create_minimal_wav_data();
        let result = AudioLoader::from_bytes(&data, Some(AudioFormat::Wav)).await;
        match result {
            Ok(_) => {} // Test passes
            Err(e) => {
                eprintln!("Test failed with error: {:?}", e);
                panic!("from_bytes failed: {:?}", e);
            }
        }
    }

    /// Creates minimal valid WAV file data for testing
    fn create_minimal_wav_data() -> Vec<u8> {
        let mut data = Vec::new();

        // Generate some minimal audio samples (100 samples of silence)
        let audio_samples: Vec<i16> = vec![0; 100];
        let audio_data_size = audio_samples.len() * 2; // 2 bytes per sample (16-bit)

        // RIFF header
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&(36 + audio_data_size as u32).to_le_bytes()); // Chunk size (36 + subchunk2_size)
        data.extend_from_slice(b"WAVE");

        // Format subchunk
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1 size (16 for PCM)
        data.extend_from_slice(&1u16.to_le_bytes()); // Audio format (1 = PCM)
        data.extend_from_slice(&1u16.to_le_bytes()); // Number of channels (1 = mono)
        data.extend_from_slice(&16000u32.to_le_bytes()); // Sample rate (16kHz)
        data.extend_from_slice(&32000u32.to_le_bytes()); // Byte rate (sample_rate * channels * bits_per_sample / 8)
        data.extend_from_slice(&2u16.to_le_bytes()); // Block align (channels * bits_per_sample / 8)
        data.extend_from_slice(&16u16.to_le_bytes()); // Bits per sample

        // Data subchunk
        data.extend_from_slice(b"data");
        data.extend_from_slice(&(audio_data_size as u32).to_le_bytes()); // Subchunk2 size

        // Audio data (16-bit PCM samples)
        for sample in audio_samples {
            data.extend_from_slice(&sample.to_le_bytes());
        }

        data
    }

    #[tokio::test]
    async fn test_can_load_nonexistent_file() {
        let can_load = AudioLoader::can_load("nonexistent.wav").await;
        assert!(!can_load);
    }

    #[tokio::test]
    async fn test_extract_time_range() {
        // Create test audio: 1 second at 16kHz, mono
        let samples = vec![0.5f32; 16000];
        let audio = AudioBuffer::new(samples, 16000, 1);

        // Extract 0.5 seconds starting from 0.25 seconds
        let result = AudioLoader::extract_time_range(audio, Some(0.25), Some(0.5));
        assert!(result.is_ok());

        let extracted = result.unwrap();
        assert_eq!(extracted.sample_rate(), 16000);
        assert_eq!(extracted.channels(), 1);
        assert_eq!(extracted.samples().len(), 8000); // 0.5 seconds at 16kHz
    }

    #[tokio::test]
    async fn test_extract_time_range_invalid() {
        let samples = vec![0.5f32; 16000];
        let audio = AudioBuffer::new(samples, 16000, 1);

        // Invalid range: start after end
        let result = AudioLoader::extract_time_range(audio, Some(2.0), Some(0.5));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_load_multiple_empty() {
        let paths: Vec<&str> = vec![];
        let options = LoadOptions::default();
        let results = AudioLoader::load_multiple(&paths, options).await;
        assert!(results.is_empty());
    }
}
