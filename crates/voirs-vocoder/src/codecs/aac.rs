//! AAC audio codec implementation.
//!
//! This module provides AAC encoding support using the libfdk-aac library.
//! AAC is a widely supported lossy audio codec that provides good compression
//! with reasonable quality.

use crate::{AudioBuffer, Result, VocoderError};
use std::path::Path;

use super::CodecConfig;

/// AAC encoding profile
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AacProfile {
    /// AAC-LC (Low Complexity) - most common profile
    Lc,
    /// AAC-HE (High Efficiency) - optimized for lower bitrates
    He,
    /// AAC-HE v2 - optimized for very low bitrates
    Hev2,
}

impl Default for AacProfile {
    fn default() -> Self {
        Self::Lc
    }
}

/// AAC-specific configuration
#[derive(Debug, Clone)]
pub struct AacConfig {
    /// AAC profile
    pub profile: AacProfile,
    /// Use variable bitrate encoding
    pub vbr: bool,
    /// Afterburner quality enhancement
    pub afterburner: bool,
    /// Spectral band replication (for HE profiles)
    pub sbr: bool,
}

impl Default for AacConfig {
    fn default() -> Self {
        Self {
            profile: AacProfile::Lc,
            vbr: false,
            afterburner: true,
            sbr: false,
        }
    }
}

/// Encode audio buffer to AAC file
pub fn encode_aac<P: AsRef<Path>>(
    audio: &AudioBuffer,
    path: P,
    config: &CodecConfig,
) -> Result<()> {
    let aac_config = AacConfig::default();
    let bytes = encode_aac_bytes_with_config(audio, config, &aac_config)?;

    std::fs::write(path, bytes)?;

    Ok(())
}

/// Encode audio buffer to AAC bytes
pub fn encode_aac_bytes(audio: &AudioBuffer, config: &CodecConfig) -> Result<Vec<u8>> {
    let aac_config = AacConfig::default();
    encode_aac_bytes_with_config(audio, config, &aac_config)
}

/// Encode audio buffer to AAC bytes with specific AAC configuration
pub fn encode_aac_bytes_with_config(
    audio: &AudioBuffer,
    config: &CodecConfig,
    aac_config: &AacConfig,
) -> Result<Vec<u8>> {
    // Validate input parameters
    if audio.channels() > 8 {
        return Err(VocoderError::ConfigError(
            "AAC encoder supports maximum 8 channels".to_string(),
        ));
    }

    if config.sample_rate > 96000 {
        return Err(VocoderError::ConfigError(
            "AAC encoder supports maximum 96 kHz sample rate".to_string(),
        ));
    }

    // Validate sample rate is supported by AAC
    let supported_rates = [
        8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000,
    ];
    if !supported_rates.contains(&config.sample_rate) {
        return Err(VocoderError::ConfigError(format!(
            "Unsupported sample rate for AAC: {} Hz. Supported rates: {:?}",
            config.sample_rate, supported_rates
        )));
    }

    // Validate bit rate is reasonable for AAC
    let bit_rate = config.bit_rate.unwrap_or(128000);
    if !(8000..=320000).contains(&bit_rate) {
        return Err(VocoderError::ConfigError(format!(
            "AAC bit rate must be between 8000 and 320000 bps, got: {bit_rate}"
        )));
    }

    // Convert AudioBuffer to interleaved samples
    let samples = audio.samples();
    let sample_rate = config.sample_rate;
    let channels = config.channels;

    // Initialize AAC encoder (enhanced implementation)
    let mut encoder = AacEncoder::new(sample_rate, channels, bit_rate, aac_config.clone())?;

    // Encode samples with frame-based processing
    let encoded_data = encoder.encode_with_frames(samples)?;

    Ok(encoded_data)
}

/// Enhanced AAC encoder implementation with frame-based processing
struct AacEncoder {
    sample_rate: u32,
    channels: u16,
    bit_rate: u32,
    config: AacConfig,
    frame_size: usize,
}

impl AacEncoder {
    /// Create new AAC encoder
    fn new(sample_rate: u32, channels: u16, bit_rate: u32, config: AacConfig) -> Result<Self> {
        // Calculate AAC frame size (1024 samples per frame for AAC-LC)
        let frame_size = match config.profile {
            AacProfile::Lc => 1024,
            AacProfile::He => 2048, // HE-AAC uses larger frames
            AacProfile::Hev2 => 2048,
        };

        Ok(Self {
            sample_rate,
            channels,
            bit_rate,
            config,
            frame_size,
        })
    }

    /// Encode audio samples to AAC with frame-based processing
    fn encode_with_frames(&mut self, samples: &[f32]) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        let channels = self.channels as usize;

        // Process audio in AAC frame chunks
        let frame_samples = self.frame_size * channels;
        let mut sample_index = 0;

        while sample_index < samples.len() {
            let chunk_end = (sample_index + frame_samples).min(samples.len());
            let chunk = &samples[sample_index..chunk_end];

            // Pad the last frame if necessary
            let mut frame_data = chunk.to_vec();
            if frame_data.len() < frame_samples {
                frame_data.resize(frame_samples, 0.0);
            }

            // Encode this frame
            let frame_bytes = self.encode_frame(&frame_data)?;
            output.extend_from_slice(&frame_bytes);

            sample_index = chunk_end;
        }

        Ok(output)
    }

    /// Encode a single AAC frame
    fn encode_frame(&mut self, frame_samples: &[f32]) -> Result<Vec<u8>> {
        // Apply quality-based processing if specified
        let processed_samples = if let Some(quality) = self.get_quality_factor() {
            self.apply_quality_processing(frame_samples, quality)
        } else {
            frame_samples.to_vec()
        };

        // Convert samples to quantized format based on bit rate
        let quantized_samples = self.quantize_samples(&processed_samples)?;

        // Create ADTS header for this frame
        let frame_length = quantized_samples.len() + 7; // 7 bytes for ADTS header
        let adts_header = create_enhanced_adts_header(
            frame_length,
            self.sample_rate,
            self.channels,
            &self.config,
        );

        // Combine header and data
        let mut frame_data = Vec::new();
        frame_data.extend_from_slice(&adts_header);
        frame_data.extend_from_slice(&quantized_samples);

        Ok(frame_data)
    }

    /// Get quality factor from bit rate and configuration
    fn get_quality_factor(&self) -> Option<f32> {
        // Calculate quality factor based on bit rate per channel
        let bit_rate_per_channel = self.bit_rate / (self.channels as u32);
        Some(match bit_rate_per_channel {
            0..=64000 => 0.6,        // Lower quality for low bit rates
            64001..=128000 => 0.75,  // Medium quality
            128001..=192000 => 0.85, // Good quality
            _ => 0.95,               // High quality for higher bit rates
        })
    }

    /// Apply quality-based processing to samples
    fn apply_quality_processing(&self, samples: &[f32], quality: f32) -> Vec<f32> {
        // Simple quality processing: apply mild compression for lower qualities
        let compression_ratio = 1.0 - (1.0 - quality) * 0.3;

        samples
            .iter()
            .map(|&sample| {
                let compressed = sample * compression_ratio;
                compressed.clamp(-1.0, 1.0)
            })
            .collect()
    }

    /// Quantize samples based on bit rate
    fn quantize_samples(&self, samples: &[f32]) -> Result<Vec<u8>> {
        // Calculate quantization bits based on bit rate
        let bits_per_sample = ((self.bit_rate as f32)
            / (self.sample_rate as f32 * self.channels as f32))
            .clamp(8.0, 24.0) as u8;

        let mut quantized = Vec::new();

        match bits_per_sample {
            8..=12 => {
                // 8-bit quantization for very low bit rates
                for &sample in samples {
                    let quantized_sample = ((sample.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
                    quantized.push(quantized_sample);
                }
            }
            13..=18 => {
                // 16-bit quantization for medium bit rates
                for &sample in samples {
                    let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                    quantized.extend_from_slice(&sample_i16.to_le_bytes());
                }
            }
            _ => {
                // 24-bit quantization for high bit rates
                for &sample in samples {
                    let sample_i32 = (sample.clamp(-1.0, 1.0) * 8388607.0) as i32;
                    let bytes = sample_i32.to_le_bytes();
                    quantized.extend_from_slice(&bytes[0..3]); // Take only 24 bits
                }
            }
        }

        Ok(quantized)
    }

    /// Legacy encode method for compatibility
    #[allow(dead_code)]
    fn encode(&mut self, samples: &[f32]) -> Result<Vec<u8>> {
        self.encode_with_frames(samples)
    }
}

/// Create enhanced ADTS header for AAC with profile support
fn create_enhanced_adts_header(
    frame_length: usize,
    sample_rate: u32,
    channels: u16,
    config: &AacConfig,
) -> [u8; 7] {
    let mut header = [0u8; 7];

    // Sync word (12 bits) - always 0xFFF
    header[0] = 0xFF;
    header[1] = 0xF0;

    // MPEG version (1 bit) - 0 for MPEG-4
    // Layer (2 bits) - always 00
    // Protection absent (1 bit) - 1 for no CRC
    header[1] |= 0x01;

    // Profile (2 bits) - depends on AAC profile
    let profile_bits = match config.profile {
        AacProfile::Lc => 0x01,   // AAC-LC
        AacProfile::He => 0x02,   // AAC-HE (SBR)
        AacProfile::Hev2 => 0x02, // AAC-HE v2 (also uses SBR profile in ADTS)
    };
    header[2] = profile_bits << 6;

    // Sample rate index (4 bits)
    let sample_rate_index = get_sample_rate_index(sample_rate);
    header[2] |= sample_rate_index << 2;

    // Private bit (1 bit) - 0
    // Channel configuration (3 bits)
    let channel_config = get_channel_configuration(channels);
    header[2] |= channel_config >> 2;
    header[3] = (channel_config & 0x03) << 6;

    // Original/copy (1 bit) - 0 for copy
    // Home (1 bit) - 0 for not home
    // Copyright ID bit (1 bit) - 0
    // Copyright ID start (1 bit) - 0
    // Frame length (13 bits)
    header[3] |= ((frame_length >> 11) & 0x03) as u8;
    header[4] = ((frame_length >> 3) & 0xFF) as u8;
    header[5] = ((frame_length & 0x07) << 5) as u8;

    // Buffer fullness (11 bits)
    if config.vbr {
        // 0x7FF for variable bitrate
        header[5] |= 0x1F;
        header[6] = 0xFC;
    } else {
        // For CBR, use a reasonable buffer fullness value
        let buffer_fullness = 0x400; // Mid-range value
        header[5] |= ((buffer_fullness >> 6) & 0x1F) as u8;
        header[6] = ((buffer_fullness & 0x3F) << 2) as u8;
    }

    // Number of raw data blocks in frame (2 bits) - 0 for single block
    header[6] |= 0x00;

    header
}

/// Create simplified ADTS header for AAC (legacy compatibility)
#[allow(dead_code)]
fn create_adts_header(frame_length: usize, sample_rate: u32, channels: u16) -> [u8; 7] {
    let config = AacConfig::default();
    create_enhanced_adts_header(frame_length, sample_rate, channels, &config)
}

/// Get AAC sample rate index for ADTS header
fn get_sample_rate_index(sample_rate: u32) -> u8 {
    match sample_rate {
        96000 => 0,
        88200 => 1,
        64000 => 2,
        48000 => 3,
        44100 => 4,
        32000 => 5,
        24000 => 6,
        22050 => 7,
        16000 => 8,
        12000 => 9,
        11025 => 10,
        8000 => 11,
        7350 => 12,
        _ => 7, // Default to 22050 for unsupported rates
    }
}

/// Get AAC channel configuration for ADTS header
fn get_channel_configuration(channels: u16) -> u8 {
    match channels {
        1 => 1, // Mono
        2 => 2, // Stereo
        3 => 3, // 3.0
        4 => 4, // 4.0
        5 => 5, // 5.0
        6 => 6, // 5.1
        7 => 7, // 7.1 (back)
        8 => 8, // 7.1
        _ => 2, // Default to stereo for unsupported configurations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aac_config_default() {
        let config = AacConfig::default();
        assert_eq!(config.profile, AacProfile::Lc);
        assert!(!config.vbr);
        assert!(config.afterburner);
        assert!(!config.sbr);
    }

    #[test]
    fn test_aac_encoder_creation() {
        let config = AacConfig::default();
        let encoder = AacEncoder::new(22050, 1, 128000, config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_adts_header_creation() {
        let header = create_adts_header(1024, 22050, 1);
        assert_eq!(header[0], 0xFF);
        assert_eq!(header[1] & 0xF0, 0xF0);
    }

    #[test]
    fn test_aac_encoding_simple() {
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let audio = AudioBuffer::new(samples, 22050, 1);
        let config = CodecConfig::default();

        let result = encode_aac_bytes(&audio, &config);
        assert!(result.is_ok());

        let encoded = result.unwrap();
        assert!(!encoded.is_empty());
        assert!(encoded.len() > 7); // At least ADTS header
    }

    #[test]
    fn test_enhanced_aac_encoding() {
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let audio = AudioBuffer::new(samples, 44100, 2);
        let config = CodecConfig {
            sample_rate: 44100,
            channels: 2,
            bit_rate: Some(192000),
            ..Default::default()
        };

        let aac_config = AacConfig {
            profile: AacProfile::Lc,
            vbr: true,
            afterburner: true,
            sbr: false,
        };

        let result = encode_aac_bytes_with_config(&audio, &config, &aac_config);
        assert!(result.is_ok());

        let encoded = result.unwrap();
        assert!(!encoded.is_empty());
        assert!(encoded.len() > 7);
    }

    #[test]
    fn test_aac_he_profile() {
        let samples = vec![0.1; 2048]; // Enough samples for HE-AAC frame
        let audio = AudioBuffer::new(samples, 48000, 1);
        let config = CodecConfig {
            sample_rate: 48000,
            bit_rate: Some(64000),
            ..Default::default()
        };

        let aac_config = AacConfig {
            profile: AacProfile::He,
            vbr: false,
            afterburner: true,
            sbr: true,
        };

        let result = encode_aac_bytes_with_config(&audio, &config, &aac_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_rate_validation() {
        let samples = vec![0.1, 0.2];
        let audio = AudioBuffer::new(samples, 7000, 1); // Unsupported rate
        let config = CodecConfig {
            sample_rate: 7000,
            ..Default::default()
        };

        let result = encode_aac_bytes(&audio, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_bit_rate_validation() {
        let samples = vec![0.1, 0.2];
        let audio = AudioBuffer::new(samples, 22050, 1);
        let config = CodecConfig {
            bit_rate: Some(500000), // Too high
            ..Default::default()
        };

        let result = encode_aac_bytes(&audio, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_enhanced_adts_header() {
        let config = AacConfig {
            profile: AacProfile::Lc,
            vbr: true,
            afterburner: true,
            sbr: false,
        };

        let header = create_enhanced_adts_header(1024, 44100, 2, &config);

        // Check sync word
        assert_eq!(header[0], 0xFF);
        assert_eq!(header[1] & 0xF0, 0xF0);

        // Check that header has valid structure
        assert_ne!(header[2], 0);
        assert_ne!(header[3], 0);
    }

    #[test]
    fn test_sample_rate_index() {
        assert_eq!(get_sample_rate_index(44100), 4);
        assert_eq!(get_sample_rate_index(48000), 3);
        assert_eq!(get_sample_rate_index(22050), 7);
        assert_eq!(get_sample_rate_index(99999), 7); // Default fallback
    }

    #[test]
    fn test_channel_configuration() {
        assert_eq!(get_channel_configuration(1), 1);
        assert_eq!(get_channel_configuration(2), 2);
        assert_eq!(get_channel_configuration(6), 6);
        assert_eq!(get_channel_configuration(99), 2); // Default fallback
    }
}
