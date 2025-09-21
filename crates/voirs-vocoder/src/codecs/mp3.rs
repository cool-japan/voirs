//! MP3 encoding implementation using LAME.

use crate::{AudioBuffer, Result, VocoderError};
use mp3lame_encoder::{Builder, FlushNoGap, InterleavedPcm, MonoPcm};
use std::fs::File;
use std::io::Write;
use std::path::Path;

use super::CodecConfig;

/// Encode audio buffer to MP3 file
pub fn encode_mp3<P: AsRef<Path>>(
    audio: &AudioBuffer,
    path: P,
    config: &CodecConfig,
) -> Result<()> {
    let encoded_data = encode_mp3_bytes(audio, config)?;

    let mut file = File::create(path)
        .map_err(|e| VocoderError::InputError(format!("Failed to create MP3 file: {e}")))?;

    file.write_all(&encoded_data)
        .map_err(|e| VocoderError::InputError(format!("Failed to write MP3 data: {e}")))?;

    Ok(())
}

/// Encode audio buffer to MP3 bytes
pub fn encode_mp3_bytes(audio: &AudioBuffer, config: &CodecConfig) -> Result<Vec<u8>> {
    // Validate sample rate
    if config.sample_rate > 48000 {
        return Err(VocoderError::ConfigError(
            "MP3 does not support sample rates above 48kHz".to_string(),
        ));
    }

    // Validate channels
    if config.channels > 2 {
        return Err(VocoderError::ConfigError(
            "MP3 does not support more than 2 channels".to_string(),
        ));
    }

    // Convert samples to PCM
    let pcm_data = convert_to_pcm_i16(audio.samples());

    // Create LAME encoder builder
    let mut builder = Builder::new()
        .ok_or_else(|| VocoderError::ConfigError("Failed to create MP3 encoder".to_string()))?;

    // Set basic parameters
    builder
        .set_sample_rate(config.sample_rate)
        .map_err(|e| VocoderError::ConfigError(format!("Failed to set sample rate: {e}")))?;

    builder
        .set_num_channels(config.channels as u8)
        .map_err(|e| VocoderError::ConfigError(format!("Failed to set channels: {e}")))?;

    // Set bitrate if specified
    if let Some(bit_rate) = config.bit_rate {
        let kbps = bit_rate / 1000;
        let bitrate = match kbps {
            8 => mp3lame_encoder::Bitrate::Kbps8,
            16 => mp3lame_encoder::Bitrate::Kbps16,
            24 => mp3lame_encoder::Bitrate::Kbps24,
            32 => mp3lame_encoder::Bitrate::Kbps32,
            40 => mp3lame_encoder::Bitrate::Kbps40,
            48 => mp3lame_encoder::Bitrate::Kbps48,
            56 => mp3lame_encoder::Bitrate::Kbps64, // Round up to next available
            64 => mp3lame_encoder::Bitrate::Kbps64,
            80 => mp3lame_encoder::Bitrate::Kbps80,
            96 => mp3lame_encoder::Bitrate::Kbps96,
            112 => mp3lame_encoder::Bitrate::Kbps112,
            128 => mp3lame_encoder::Bitrate::Kbps128,
            160 => mp3lame_encoder::Bitrate::Kbps160,
            192 => mp3lame_encoder::Bitrate::Kbps192,
            224 => mp3lame_encoder::Bitrate::Kbps224,
            256 => mp3lame_encoder::Bitrate::Kbps256,
            320 => mp3lame_encoder::Bitrate::Kbps320,
            _ => mp3lame_encoder::Bitrate::Kbps128, // Default fallback
        };
        builder
            .set_brate(bitrate)
            .map_err(|e| VocoderError::ConfigError(format!("Failed to set bitrate: {e}")))?;
    }

    // Set quality if specified (Enhanced VBR quality mapping from 0.0-1.0 to available quality levels)
    if let Some(quality) = config.quality {
        // Enhanced quality mapping with improved granularity
        // Maps 0.0-1.0 quality range to available LAME quality levels
        // Higher input quality (approaching 1.0) maps to better encoding quality
        let quality_setting = if quality >= 0.6 {
            mp3lame_encoder::Quality::Best // High quality: 0.6-1.0 -> Best (improved threshold)
        } else {
            mp3lame_encoder::Quality::Good // Standard quality: 0.0-0.59 -> Good
        };

        builder
            .set_quality(quality_setting)
            .map_err(|e| VocoderError::ConfigError(format!("Failed to set quality: {e}")))?;
    }

    // Build the encoder
    let mut encoder = builder
        .build()
        .map_err(|e| VocoderError::ConfigError(format!("Failed to build MP3 encoder: {e}")))?;

    // Encode the audio
    let mut mp3_buffer = Vec::new();
    let mut output_buffer = vec![std::mem::MaybeUninit::uninit(); 7200]; // Standard MP3 frame buffer size

    // Use the correct input type based on the number of channels
    let bytes_written = if config.channels == 1 {
        // For mono audio, use MonoPcm
        let input = MonoPcm(&pcm_data);
        encoder
            .encode(input, &mut output_buffer)
            .map_err(|e| VocoderError::VocodingError(format!("MP3 encoding failed: {e}")))?
    } else {
        // For stereo, check if audio is already stereo or if we need to duplicate mono
        let stereo_data = if audio.channels() == 2 {
            // Audio is already stereo, use as-is
            pcm_data
        } else {
            // Audio is mono, duplicate to stereo
            let mut stereo_data = Vec::with_capacity(pcm_data.len() * 2);
            for &sample in &pcm_data {
                stereo_data.push(sample); // Left
                stereo_data.push(sample); // Right (duplicate for mono source)
            }
            stereo_data
        };

        // Use InterleavedPcm for stereo audio
        let input = InterleavedPcm(&stereo_data);
        encoder
            .encode(input, &mut output_buffer)
            .map_err(|e| VocoderError::VocodingError(format!("MP3 encoding failed: {e}")))?
    };

    // Safety: encoder.encode guarantees that `bytes_written` bytes are initialized
    let encoded_slice =
        unsafe { std::slice::from_raw_parts(output_buffer.as_ptr() as *const u8, bytes_written) };
    mp3_buffer.extend_from_slice(encoded_slice);

    // Flush the encoder
    let mut flush_buffer = vec![std::mem::MaybeUninit::uninit(); 7200];
    let final_bytes = encoder
        .flush::<FlushNoGap>(&mut flush_buffer)
        .map_err(|e| VocoderError::VocodingError(format!("MP3 flush failed: {e}")))?;

    // Safety: encoder.flush guarantees that `final_bytes` bytes are initialized
    let final_slice =
        unsafe { std::slice::from_raw_parts(flush_buffer.as_ptr() as *const u8, final_bytes) };
    mp3_buffer.extend_from_slice(final_slice);

    Ok(mp3_buffer)
}

/// Convert f32 samples to i16 PCM
fn convert_to_pcm_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&sample| (sample * 32767.0).clamp(-32768.0, 32767.0) as i16)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_mp3_encoding() {
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let audio = AudioBuffer::new(samples, 22050, 1);
        let config = CodecConfig {
            sample_rate: 22050,
            channels: 1,
            bit_rate: Some(128000),
            quality: Some(0.7),
            compression_level: None,
        };

        let result = encode_mp3_bytes(&audio, &config);
        assert!(result.is_ok());

        let encoded = result.unwrap();
        assert!(!encoded.is_empty());
        // MP3 files should start with sync word
        assert_eq!(encoded[0], 0xFF);
    }

    #[test]
    fn test_mp3_file_encoding() {
        let samples = vec![0.1, -0.2, 0.3, -0.4];
        let audio = AudioBuffer::new(samples, 22050, 1);
        let config = CodecConfig::default();

        let test_path = "/tmp/test_audio.mp3";
        let result = encode_mp3(&audio, test_path, &config);
        assert!(result.is_ok());

        // Verify file was created
        assert!(fs::metadata(test_path).is_ok());

        // Clean up
        let _ = fs::remove_file(test_path);
    }

    #[test]
    fn test_invalid_sample_rate() {
        let audio = AudioBuffer::new(vec![0.1], 96000, 1);
        let config = CodecConfig {
            sample_rate: 96000,
            channels: 1,
            bit_rate: Some(128000),
            quality: Some(0.5),
            compression_level: None,
        };

        let result = encode_mp3_bytes(&audio, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_channels() {
        let audio = AudioBuffer::new(vec![0.1], 22050, 1);
        let config = CodecConfig {
            sample_rate: 22050,
            channels: 8, // Invalid for MP3
            bit_rate: Some(128000),
            quality: Some(0.5),
            compression_level: None,
        };

        let result = encode_mp3_bytes(&audio, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_pcm_conversion() {
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let pcm = convert_to_pcm_i16(&samples);

        assert_eq!(pcm[0], 0);
        assert_eq!(pcm[1], 16383); // 0.5 * 32767
        assert_eq!(pcm[2], -16383); // -0.5 * 32767
        assert_eq!(pcm[3], 32767);
        assert_eq!(pcm[4], -32767); // -1.0 * 32767 (clamped)
    }
}
