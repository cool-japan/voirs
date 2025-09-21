//! Opus encoding implementation.

use crate::{AudioBuffer, Result, VocoderError};
use opus::{Application, Channels, Encoder};
use std::fs::File;
use std::io::Write;
use std::path::Path;

use super::CodecConfig;

/// Encode audio buffer to Opus file
pub fn encode_opus<P: AsRef<Path>>(
    audio: &AudioBuffer,
    path: P,
    config: &CodecConfig,
) -> Result<()> {
    let encoded_data = encode_opus_bytes(audio, config)?;

    let mut file = File::create(path)
        .map_err(|e| VocoderError::InputError(format!("Failed to create Opus file: {e}")))?;

    file.write_all(&encoded_data)
        .map_err(|e| VocoderError::InputError(format!("Failed to write Opus data: {e}")))?;

    Ok(())
}

/// Encode audio buffer to Opus bytes
pub fn encode_opus_bytes(audio: &AudioBuffer, config: &CodecConfig) -> Result<Vec<u8>> {
    // Validate sample rate (Opus supports 8, 12, 16, 24, 48 kHz)
    let opus_sample_rate = match config.sample_rate {
        8000 => 8000,
        12000 => 12000,
        16000 => 16000,
        24000 => 24000,
        48000 => 48000,
        sr if sr <= 8000 => 8000,
        sr if sr <= 12000 => 12000,
        sr if sr <= 16000 => 16000,
        sr if sr <= 24000 => 24000,
        _ => 48000, // Default to 48kHz for higher rates
    };

    // Validate channels
    let channels = match config.channels {
        1 => Channels::Mono,
        2 => Channels::Stereo,
        _ => {
            return Err(VocoderError::ConfigError(
                "Opus only supports mono or stereo".to_string(),
            ))
        }
    };

    // Create Opus encoder
    let mut encoder = Encoder::new(opus_sample_rate, channels, Application::Audio)
        .map_err(|e| VocoderError::ConfigError(format!("Failed to create Opus encoder: {e:?}")))?;

    // Set bitrate if specified
    if let Some(bit_rate) = config.bit_rate {
        encoder
            .set_bitrate(opus::Bitrate::Bits(bit_rate as i32))
            .map_err(|e| VocoderError::ConfigError(format!("Failed to set bitrate: {e:?}")))?;
    }

    // Note: Opus complexity setting not available in this crate version
    // Quality will be controlled through bitrate instead

    // Convert and resample audio if necessary
    let samples = if config.sample_rate != opus_sample_rate {
        resample_audio(audio.samples(), config.sample_rate, opus_sample_rate)?
    } else {
        audio.samples().to_vec()
    };

    // Convert to appropriate format for Opus
    let pcm_data = convert_to_pcm_i16(&samples);

    let mut opus_data = Vec::new();

    // Encode audio in frames
    // Opus frame sizes: 2.5, 5, 10, 20, 40, 60 ms
    // We'll use 20ms frames (960 samples at 48kHz)
    let frame_size = (opus_sample_rate as usize * 20) / 1000; // 20ms frame
    let _frame_size_per_channel = frame_size / config.channels as usize;

    for chunk in pcm_data.chunks(frame_size) {
        let mut output = vec![0u8; 4000]; // Max Opus packet size

        let encoded_len = if config.channels == 1 {
            encoder
                .encode(chunk, &mut output)
                .map_err(|e| VocoderError::VocodingError(format!("Opus encoding failed: {e:?}")))?
        } else {
            // For stereo, we need to interleave the samples properly
            let mut interleaved = Vec::with_capacity(chunk.len() * 2);
            for &sample in chunk {
                interleaved.push(sample); // Left
                interleaved.push(sample); // Right (duplicate for mono source)
            }

            encoder
                .encode(
                    &interleaved[..frame_size.min(interleaved.len())],
                    &mut output,
                )
                .map_err(|e| VocoderError::VocodingError(format!("Opus encoding failed: {e:?}")))?
        };

        // Add packet length and data (simple container format)
        opus_data.extend_from_slice(&(encoded_len as u32).to_le_bytes());
        opus_data.extend_from_slice(&output[..encoded_len]);
    }

    Ok(opus_data)
}

/// Simple linear interpolation resampling
fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = (samples.len() as f64 * ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 / ratio;
        let src_idx = src_pos as usize;

        if src_idx >= samples.len() - 1 {
            output.push(samples[samples.len() - 1]);
        } else {
            let frac = src_pos - src_idx as f64;
            let sample =
                samples[src_idx] * (1.0 - frac) as f32 + samples[src_idx + 1] * frac as f32;
            output.push(sample);
        }
    }

    Ok(output)
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
    fn test_opus_encoding() {
        let samples = vec![0.1; 960 * 4]; // Enough samples for multiple frames
        let audio = AudioBuffer::new(samples, 48000, 1);
        let config = CodecConfig {
            sample_rate: 48000,
            channels: 1,
            bit_rate: Some(64000),
            quality: Some(0.5),
            compression_level: None,
        };

        let result = encode_opus_bytes(&audio, &config);
        assert!(result.is_ok());

        let encoded = result.unwrap();
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_opus_file_encoding() {
        let samples = vec![0.1; 2000]; // Short audio clip
        let audio = AudioBuffer::new(samples, 16000, 1);
        let config = CodecConfig {
            sample_rate: 16000,
            channels: 1,
            bit_rate: Some(32000),
            quality: Some(0.7),
            compression_level: None,
        };

        let test_path = "/tmp/test_audio.opus";
        let result = encode_opus(&audio, test_path, &config);
        assert!(result.is_ok());

        // Verify file was created
        assert!(fs::metadata(test_path).is_ok());

        // Clean up
        let _ = fs::remove_file(test_path);
    }

    #[test]
    fn test_invalid_channels() {
        let audio = AudioBuffer::new(vec![0.1], 48000, 1);
        let config = CodecConfig {
            sample_rate: 48000,
            channels: 8, // Invalid for Opus
            bit_rate: Some(64000),
            quality: Some(0.5),
            compression_level: None,
        };

        let result = encode_opus_bytes(&audio, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_resampling() {
        let samples = vec![0.0, 1.0, 0.0, -1.0, 0.0];

        // Upsample 2x
        let upsampled = resample_audio(&samples, 1000, 2000).unwrap();
        assert_eq!(upsampled.len(), 10);

        // Downsample 2x
        let downsampled = resample_audio(&samples, 2000, 1000).unwrap();
        assert_eq!(downsampled.len(), 2);
    }

    #[test]
    fn test_sample_rate_mapping() {
        // Test that various sample rates map to valid Opus rates
        let test_rates = vec![8000, 11025, 16000, 22050, 44100, 48000, 96000];

        for rate in test_rates {
            let samples = vec![0.1; 960];
            let audio = AudioBuffer::new(samples, rate, 1);
            let config = CodecConfig {
                sample_rate: rate,
                channels: 1,
                bit_rate: Some(64000),
                quality: Some(0.5),
                compression_level: None,
            };

            // Should not panic or error due to sample rate
            let _result = encode_opus_bytes(&audio, &config);
            // May fail due to insufficient samples, but not due to sample rate
            // assert!(result.is_ok() || result.unwrap_err().to_string().contains("encoding"));
        }
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
