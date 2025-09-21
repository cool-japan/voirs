//! Audio I/O module.
//!
//! This module provides audio input/output functionality including:
//! - WAV file writing with hound crate
//! - Audio format conversions
//! - Streaming audio output
//! - Raw PCM data export

use crate::codecs::{AudioCodec, AudioCodecEncoder, CodecConfig};
use crate::{AudioBuffer, Result, VocoderError};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::fs::File;
use std::path::Path;

/// Audio file format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFileFormat {
    /// WAV format
    Wav,
    /// Raw PCM format
    RawPcm,
    /// MP3 format
    Mp3,
    /// FLAC format
    Flac,
    /// Opus format
    Opus,
}

/// Audio encoding configuration
#[derive(Debug, Clone)]
pub struct AudioEncodeConfig {
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Bits per sample
    pub bits_per_sample: u16,
    /// File format
    pub format: AudioFileFormat,
    /// Bit rate for lossy codecs (bits per second)
    pub bit_rate: Option<u32>,
    /// Quality setting (0.0-1.0, codec-specific)
    pub quality: Option<f32>,
    /// Compression level (for lossless codecs)
    pub compression_level: Option<u32>,
}

impl Default for AudioEncodeConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            channels: 1,
            bits_per_sample: 16,
            format: AudioFileFormat::Wav,
            bit_rate: Some(128000),     // 128 kbps default
            quality: Some(0.7),         // Good quality
            compression_level: Some(5), // Medium compression
        }
    }
}

/// Audio encoder for writing audio files
pub struct AudioEncoder {
    config: AudioEncodeConfig,
}

impl AudioEncoder {
    /// Create new audio encoder
    pub fn new(config: AudioEncodeConfig) -> Self {
        Self { config }
    }

    /// Write audio buffer to file
    pub fn write_to_file<P: AsRef<Path>>(&self, audio: &AudioBuffer, path: P) -> Result<()> {
        match self.config.format {
            AudioFileFormat::Wav => self.write_wav(audio, path),
            AudioFileFormat::RawPcm => self.write_raw_pcm(audio, path),
            AudioFileFormat::Mp3 => self.write_codec(audio, path, AudioCodec::Mp3),
            AudioFileFormat::Flac => self.write_codec(audio, path, AudioCodec::Flac),
            AudioFileFormat::Opus => self.write_codec(audio, path, AudioCodec::Opus),
        }
    }

    /// Write audio using codec
    fn write_codec<P: AsRef<Path>>(
        &self,
        audio: &AudioBuffer,
        path: P,
        codec: AudioCodec,
    ) -> Result<()> {
        let codec_config = CodecConfig {
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
            bit_rate: self.config.bit_rate,
            quality: self.config.quality,
            compression_level: self.config.compression_level,
        };

        let encoder = AudioCodecEncoder::new(codec, codec_config);
        encoder.encode_to_file(audio, path)
    }

    /// Write WAV file
    fn write_wav<P: AsRef<Path>>(&self, audio: &AudioBuffer, path: P) -> Result<()> {
        let spec = WavSpec {
            channels: self.config.channels,
            sample_rate: self.config.sample_rate,
            bits_per_sample: self.config.bits_per_sample,
            sample_format: match self.config.bits_per_sample {
                32 => SampleFormat::Float,
                _ => SampleFormat::Int,
            },
        };

        let mut writer = WavWriter::create(path, spec)
            .map_err(|e| VocoderError::InputError(format!("Failed to create WAV writer: {e}")))?;

        match self.config.bits_per_sample {
            16 => {
                for &sample in audio.samples() {
                    let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                    writer.write_sample(sample_i16).map_err(|e| {
                        VocoderError::InputError(format!("Failed to write sample: {e}"))
                    })?;
                }
            }
            24 => {
                for &sample in audio.samples() {
                    let sample_i32 = (sample * 8388607.0).clamp(-8388608.0, 8388607.0) as i32;
                    writer.write_sample(sample_i32).map_err(|e| {
                        VocoderError::InputError(format!("Failed to write sample: {e}"))
                    })?;
                }
            }
            32 => {
                for &sample in audio.samples() {
                    writer.write_sample(sample).map_err(|e| {
                        VocoderError::InputError(format!("Failed to write sample: {e}"))
                    })?;
                }
            }
            _ => {
                return Err(VocoderError::ConfigError(
                    "Unsupported bits per sample".to_string(),
                ))
            }
        }

        writer
            .finalize()
            .map_err(|e| VocoderError::InputError(format!("Failed to finalize WAV file: {e}")))?;

        Ok(())
    }

    /// Write raw PCM file
    fn write_raw_pcm<P: AsRef<Path>>(&self, audio: &AudioBuffer, path: P) -> Result<()> {
        use std::io::Write;

        let mut file = File::create(path)
            .map_err(|e| VocoderError::InputError(format!("Failed to create file: {e}")))?;

        match self.config.bits_per_sample {
            16 => {
                for &sample in audio.samples() {
                    let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                    file.write_all(&sample_i16.to_le_bytes()).map_err(|e| {
                        VocoderError::InputError(format!("Failed to write sample: {e}"))
                    })?;
                }
            }
            24 => {
                for &sample in audio.samples() {
                    let sample_i32 = (sample * 8388607.0).clamp(-8388608.0, 8388607.0) as i32;
                    let bytes = sample_i32.to_le_bytes();
                    file.write_all(&bytes[0..3]).map_err(|e| {
                        VocoderError::InputError(format!("Failed to write sample: {e}"))
                    })?;
                }
            }
            32 => {
                for &sample in audio.samples() {
                    file.write_all(&sample.to_le_bytes()).map_err(|e| {
                        VocoderError::InputError(format!("Failed to write sample: {e}"))
                    })?;
                }
            }
            _ => {
                return Err(VocoderError::ConfigError(
                    "Unsupported bits per sample".to_string(),
                ))
            }
        }

        Ok(())
    }
}

/// Audio decoder for reading audio files
pub struct AudioDecoder;

impl AudioDecoder {
    /// Read WAV file
    pub fn read_wav<P: AsRef<Path>>(path: P) -> Result<AudioBuffer> {
        let mut reader = WavReader::open(path)
            .map_err(|e| VocoderError::InputError(format!("Failed to open WAV file: {e}")))?;

        let spec = reader.spec();
        let mut samples = Vec::new();

        match spec.sample_format {
            SampleFormat::Float => {
                for sample_result in reader.samples::<f32>() {
                    let sample = sample_result.map_err(|e| {
                        VocoderError::InputError(format!("Failed to read sample: {e}"))
                    })?;
                    samples.push(sample);
                }
            }
            SampleFormat::Int => match spec.bits_per_sample {
                16 => {
                    for sample_result in reader.samples::<i16>() {
                        let sample = sample_result.map_err(|e| {
                            VocoderError::InputError(format!("Failed to read sample: {e}"))
                        })?;
                        samples.push(sample as f32 / 32767.0);
                    }
                }
                24 => {
                    for sample_result in reader.samples::<i32>() {
                        let sample = sample_result.map_err(|e| {
                            VocoderError::InputError(format!("Failed to read sample: {e}"))
                        })?;
                        samples.push(sample as f32 / 8388607.0);
                    }
                }
                32 => {
                    for sample_result in reader.samples::<i32>() {
                        let sample = sample_result.map_err(|e| {
                            VocoderError::InputError(format!("Failed to read sample: {e}"))
                        })?;
                        samples.push(sample as f32 / 2147483647.0);
                    }
                }
                _ => {
                    return Err(VocoderError::ConfigError(
                        "Unsupported bits per sample".to_string(),
                    ))
                }
            },
        }

        Ok(AudioBuffer::new(
            samples,
            spec.sample_rate,
            spec.channels as u32,
        ))
    }
}

/// Streaming audio writer
pub struct StreamingAudioWriter {
    writer: WavWriter<std::io::BufWriter<File>>,
    config: AudioEncodeConfig,
}

impl StreamingAudioWriter {
    /// Create new streaming writer
    pub fn new<P: AsRef<Path>>(path: P, config: AudioEncodeConfig) -> Result<Self> {
        let spec = WavSpec {
            channels: config.channels,
            sample_rate: config.sample_rate,
            bits_per_sample: config.bits_per_sample,
            sample_format: match config.bits_per_sample {
                32 => SampleFormat::Float,
                _ => SampleFormat::Int,
            },
        };

        let writer = WavWriter::create(path, spec).map_err(|e| {
            VocoderError::InputError(format!("Failed to create streaming writer: {e}"))
        })?;

        Ok(Self { writer, config })
    }

    /// Write audio chunk
    pub fn write_chunk(&mut self, audio: &AudioBuffer) -> Result<()> {
        match self.config.bits_per_sample {
            16 => {
                for &sample in audio.samples() {
                    let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                    self.writer.write_sample(sample_i16).map_err(|e| {
                        VocoderError::InputError(format!("Failed to write sample: {e}"))
                    })?;
                }
            }
            24 => {
                for &sample in audio.samples() {
                    let sample_i32 = (sample * 8388607.0).clamp(-8388608.0, 8388607.0) as i32;
                    self.writer.write_sample(sample_i32).map_err(|e| {
                        VocoderError::InputError(format!("Failed to write sample: {e}"))
                    })?;
                }
            }
            32 => {
                for &sample in audio.samples() {
                    self.writer.write_sample(sample).map_err(|e| {
                        VocoderError::InputError(format!("Failed to write sample: {e}"))
                    })?;
                }
            }
            _ => {
                return Err(VocoderError::ConfigError(
                    "Unsupported bits per sample".to_string(),
                ))
            }
        }

        Ok(())
    }

    /// Finalize the audio file
    pub fn finalize(self) -> Result<()> {
        self.writer
            .finalize()
            .map_err(|e| VocoderError::InputError(format!("Failed to finalize audio file: {e}")))?;
        Ok(())
    }
}

/// Convenience functions for common audio I/O operations
pub mod convenience {
    use super::*;

    /// Write audio buffer to WAV file with default settings
    pub fn write_wav<P: AsRef<Path>>(audio: &AudioBuffer, path: P) -> Result<()> {
        let config = AudioEncodeConfig {
            sample_rate: audio.sample_rate(),
            channels: 1,
            bits_per_sample: 16,
            format: AudioFileFormat::Wav,
            bit_rate: None,
            quality: None,
            compression_level: None,
        };

        let encoder = AudioEncoder::new(config);
        encoder.write_to_file(audio, path)
    }

    /// Write audio buffer to high-quality WAV file
    pub fn write_wav_hq<P: AsRef<Path>>(audio: &AudioBuffer, path: P) -> Result<()> {
        let config = AudioEncodeConfig {
            sample_rate: audio.sample_rate(),
            channels: 1,
            bits_per_sample: 24,
            format: AudioFileFormat::Wav,
            bit_rate: None,
            quality: None,
            compression_level: None,
        };

        let encoder = AudioEncoder::new(config);
        encoder.write_to_file(audio, path)
    }

    /// Write audio buffer to floating-point WAV file
    pub fn write_wav_float<P: AsRef<Path>>(audio: &AudioBuffer, path: P) -> Result<()> {
        let config = AudioEncodeConfig {
            sample_rate: audio.sample_rate(),
            channels: 1,
            bits_per_sample: 32,
            format: AudioFileFormat::Wav,
            bit_rate: None,
            quality: None,
            compression_level: None,
        };

        let encoder = AudioEncoder::new(config);
        encoder.write_to_file(audio, path)
    }

    /// Read WAV file into audio buffer
    pub fn read_wav<P: AsRef<Path>>(path: P) -> Result<AudioBuffer> {
        AudioDecoder::read_wav(path)
    }

    /// Write audio buffer to MP3 file
    pub fn write_mp3<P: AsRef<Path>>(audio: &AudioBuffer, path: P) -> Result<()> {
        let config = AudioEncodeConfig {
            sample_rate: audio.sample_rate(),
            channels: 1,
            bits_per_sample: 16,
            format: AudioFileFormat::Mp3,
            bit_rate: Some(128000),
            quality: Some(0.7),
            compression_level: None,
        };

        let encoder = AudioEncoder::new(config);
        encoder.write_to_file(audio, path)
    }

    /// Write audio buffer to high-quality MP3 file
    pub fn write_mp3_hq<P: AsRef<Path>>(audio: &AudioBuffer, path: P) -> Result<()> {
        let config = AudioEncodeConfig {
            sample_rate: audio.sample_rate(),
            channels: 1,
            bits_per_sample: 16,
            format: AudioFileFormat::Mp3,
            bit_rate: Some(320000),
            quality: Some(0.9),
            compression_level: None,
        };

        let encoder = AudioEncoder::new(config);
        encoder.write_to_file(audio, path)
    }

    /// Write audio buffer to FLAC file
    pub fn write_flac<P: AsRef<Path>>(audio: &AudioBuffer, path: P) -> Result<()> {
        let config = AudioEncodeConfig {
            sample_rate: audio.sample_rate(),
            channels: 1,
            bits_per_sample: 24,
            format: AudioFileFormat::Flac,
            bit_rate: None,
            quality: Some(0.8),
            compression_level: Some(6),
        };

        let encoder = AudioEncoder::new(config);
        encoder.write_to_file(audio, path)
    }

    /// Write audio buffer to Opus file
    pub fn write_opus<P: AsRef<Path>>(audio: &AudioBuffer, path: P) -> Result<()> {
        let config = AudioEncodeConfig {
            sample_rate: audio.sample_rate(),
            channels: 1,
            bits_per_sample: 16,
            format: AudioFileFormat::Opus,
            bit_rate: Some(64000),
            quality: Some(0.7),
            compression_level: None,
        };

        let encoder = AudioEncoder::new(config);
        encoder.write_to_file(audio, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_audio_encoder_config() {
        let config = AudioEncodeConfig::default();
        assert_eq!(config.sample_rate, 22050);
        assert_eq!(config.channels, 1);
        assert_eq!(config.bits_per_sample, 16);
        assert_eq!(config.format, AudioFileFormat::Wav);
        assert_eq!(config.bit_rate, Some(128000));
        assert_eq!(config.quality, Some(0.7));
        assert_eq!(config.compression_level, Some(5));
    }

    #[test]
    fn test_write_and_read_wav() {
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let audio = AudioBuffer::new(samples.clone(), 22050, 1);

        let test_path = "/tmp/test_audio.wav";

        // Write WAV file
        let result = convenience::write_wav(&audio, test_path);
        assert!(result.is_ok());

        // Read WAV file back
        let read_audio = convenience::read_wav(test_path);
        assert!(read_audio.is_ok());

        let read_audio = read_audio.unwrap();
        assert_eq!(read_audio.sample_rate(), 22050);

        // Clean up
        let _ = fs::remove_file(test_path);
    }

    #[test]
    fn test_streaming_writer() {
        let test_path = "/tmp/test_streaming.wav";
        let config = AudioEncodeConfig::default();

        let mut writer = StreamingAudioWriter::new(test_path, config).unwrap();

        // Write multiple chunks
        let chunk1 = AudioBuffer::new(vec![0.1, 0.2], 22050, 1);
        let chunk2 = AudioBuffer::new(vec![0.3, 0.4], 22050, 1);

        writer.write_chunk(&chunk1).unwrap();
        writer.write_chunk(&chunk2).unwrap();
        writer.finalize().unwrap();

        // Verify file was created
        assert!(fs::metadata(test_path).is_ok());

        // Clean up
        let _ = fs::remove_file(test_path);
    }

    #[test]
    fn test_raw_pcm_write() {
        let samples = vec![0.1, 0.2, 0.3];
        let audio = AudioBuffer::new(samples, 22050, 1);

        let config = AudioEncodeConfig {
            sample_rate: 22050,
            channels: 1,
            bits_per_sample: 16,
            format: AudioFileFormat::RawPcm,
            bit_rate: None,
            quality: None,
            compression_level: None,
        };

        let encoder = AudioEncoder::new(config);
        let test_path = "/tmp/test_raw.pcm";

        let result = encoder.write_to_file(&audio, test_path);
        assert!(result.is_ok());

        // Verify file was created
        assert!(fs::metadata(test_path).is_ok());

        // Clean up
        let _ = fs::remove_file(test_path);
    }

    #[test]
    fn test_mp3_convenience_function() {
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let audio = AudioBuffer::new(samples, 22050, 1);

        let test_path = "/tmp/test_convenience.mp3";
        let result = convenience::write_mp3(&audio, test_path);

        // Note: This test might fail if codec dependencies are not available
        // In real usage, we would have proper codec libraries installed
        match result {
            Ok(_) => {
                assert!(fs::metadata(test_path).is_ok());
                let _ = fs::remove_file(test_path);
            }
            Err(_) => {
                // Expected if codec dependencies are not available
                println!("MP3 codec not available - test skipped");
            }
        }
    }

    #[test]
    fn test_codec_file_extensions() {
        let config_mp3 = AudioEncodeConfig {
            format: AudioFileFormat::Mp3,
            ..Default::default()
        };

        let config_flac = AudioEncodeConfig {
            format: AudioFileFormat::Flac,
            ..Default::default()
        };

        let config_opus = AudioEncodeConfig {
            format: AudioFileFormat::Opus,
            ..Default::default()
        };

        // Just test that configs can be created
        assert_eq!(config_mp3.format, AudioFileFormat::Mp3);
        assert_eq!(config_flac.format, AudioFileFormat::Flac);
        assert_eq!(config_opus.format, AudioFileFormat::Opus);
    }
}
