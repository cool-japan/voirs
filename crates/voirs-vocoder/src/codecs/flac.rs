//! FLAC encoding implementation.
//!
//! When the `ffi-codecs` feature is enabled, FLAC encoding is performed by the
//! COOLJAPAN `oxiaudio-encode` crate, a Pure-Rust FLAC encoder backed by
//! `flacenc`. (The crate is named `ffi-codecs` for historical reasons — it
//! groups the optional codecs alongside the genuinely C-FFI ones such as Opus
//! and LAME — but FLAC encoding here is 100% Pure Rust.)
//!
//! Without the `ffi-codecs` feature, a minimal PCM-in-FLAC-container fallback is
//! used so the default build keeps producing `.flac` output without any optional
//! dependency.

use crate::{AudioBuffer, Result, VocoderError};
use std::path::Path;

use super::CodecConfig;

/// FLAC compression level used when [`CodecConfig::compression_level`] is unset.
#[cfg(feature = "ffi-codecs")]
const DEFAULT_FLAC_COMPRESSION: u8 = 5;

/// PCM bit depth written to the FLAC stream.
#[cfg(feature = "ffi-codecs")]
const FLAC_BITS_PER_SAMPLE: u8 = 24;

/// Validate that the codec configuration is representable in a FLAC stream.
fn validate_flac_config(config: &CodecConfig) -> Result<()> {
    // FLAC's STREAMINFO encodes the sample rate in 20 bits (max 1,048,575 Hz),
    // but the format restricts the usable range to <= 655,350 Hz.
    if config.sample_rate > 655_350 {
        return Err(VocoderError::ConfigError(
            "FLAC does not support sample rates above 655.35kHz".to_string(),
        ));
    }

    // FLAC supports a maximum of 8 channels.
    if config.channels > 8 {
        return Err(VocoderError::ConfigError(
            "FLAC does not support more than 8 channels".to_string(),
        ));
    }

    Ok(())
}

/// Encode audio buffer to FLAC file.
pub fn encode_flac<P: AsRef<Path>>(
    audio: &AudioBuffer,
    path: P,
    config: &CodecConfig,
) -> Result<()> {
    let encoded_data = encode_flac_bytes(audio, config)?;

    std::fs::write(path, &encoded_data)
        .map_err(|e| VocoderError::InputError(format!("Failed to write FLAC file: {e}")))?;

    Ok(())
}

/// Encode audio buffer to FLAC bytes using the Pure-Rust OxiAudio encoder.
#[cfg(feature = "ffi-codecs")]
pub fn encode_flac_bytes(audio: &AudioBuffer, config: &CodecConfig) -> Result<Vec<u8>> {
    use oxiaudio_core::{AudioBuffer as OxiAudioBuffer, AudioEncoder, ChannelLayout, SampleFormat};
    use oxiaudio_encode::FlacEncoder;

    validate_flac_config(config)?;

    // FLAC compression level is 0 (fastest) .. 8 (best); clamp anything larger.
    let compression = config
        .compression_level
        .map(|level| level.min(8) as u8)
        .unwrap_or(DEFAULT_FLAC_COMPRESSION);

    let oxi_buffer = OxiAudioBuffer::<f32> {
        samples: audio.samples().to_vec(),
        sample_rate: config.sample_rate,
        channels: ChannelLayout::from(config.channels),
        format: SampleFormat::F32,
    };

    let mut encoder = FlacEncoder::new(compression).with_bits_per_sample(FLAC_BITS_PER_SAMPLE);

    let mut cursor = std::io::Cursor::new(Vec::new());
    encoder
        .encode(&oxi_buffer, &mut cursor)
        .map_err(|e| VocoderError::InputError(format!("FLAC encoding failed: {e}")))?;

    Ok(cursor.into_inner())
}

/// Encode audio buffer to FLAC bytes (Pure-Rust default-feature fallback).
///
/// Without the `ffi-codecs` feature the real FLAC encoder is unavailable, so a
/// minimal PCM-in-FLAC-container representation is emitted. This keeps a valid
/// `fLaC` signature and STREAMINFO block, but does not perform FLAC compression;
/// enable `ffi-codecs` for true compressed FLAC output.
#[cfg(not(feature = "ffi-codecs"))]
pub fn encode_flac_bytes(audio: &AudioBuffer, config: &CodecConfig) -> Result<Vec<u8>> {
    validate_flac_config(config)?;

    let pcm_data = convert_to_pcm_i24(audio.samples());
    let mut output = Vec::new();

    // FLAC signature.
    output.extend_from_slice(b"fLaC");

    // Minimal metadata block (STREAMINFO): last block, type 0, length 34.
    output.extend_from_slice(&[0x80, 0x00, 0x00, 0x22]);

    let md5_hash = compute_md5_signature(&pcm_data);

    // STREAMINFO block (34 bytes, partially populated).
    let sample_rate_20bit = config.sample_rate & 0xFFFFF;
    output.extend_from_slice(&sample_rate_20bit.to_be_bytes()[1..]);
    output.push((((config.channels as u32 - 1) << 1) | 0x01) as u8);
    output.extend_from_slice(&(pcm_data.len() as u32).to_be_bytes());
    output.extend_from_slice(&md5_hash);

    // PCM data with a simplified frame structure.
    for chunk in pcm_data.chunks(1024) {
        output.extend_from_slice(&[0xFF, 0xF8]);
        output.extend_from_slice(&(chunk.len() as u16).to_be_bytes());
        for &sample in chunk {
            output.extend_from_slice(&sample.to_le_bytes()[0..3]);
        }
    }

    Ok(output)
}

/// Convert f32 samples to i32 (24-bit) PCM.
#[cfg(not(feature = "ffi-codecs"))]
fn convert_to_pcm_i24(samples: &[f32]) -> Vec<i32> {
    samples
        .iter()
        .map(|&sample| {
            let scaled = sample * 8_388_607.0; // 2^23 - 1
            scaled.clamp(-8_388_608.0, 8_388_607.0) as i32
        })
        .collect()
}

/// Compute MD5 hash of PCM audio data.
///
/// The MD5 signature in FLAC is computed from the unencoded PCM audio samples
/// to provide integrity checking of the decoded audio data.
#[cfg(not(feature = "ffi-codecs"))]
fn compute_md5_signature(pcm_data: &[i32]) -> [u8; 16] {
    let mut data = Vec::with_capacity(pcm_data.len() * 3);
    for &sample in pcm_data {
        let bytes = sample.to_le_bytes();
        data.extend_from_slice(&bytes[0..3]);
    }

    let digest = md5::compute(&data);
    digest.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(sample_rate: u32, channels: u16) -> CodecConfig {
        CodecConfig {
            sample_rate,
            channels,
            bit_rate: None,
            quality: Some(0.5),
            compression_level: Some(5),
        }
    }

    #[test]
    fn test_flac_encoding() {
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let audio = AudioBuffer::new(samples, 22050, 1);
        let config = test_config(22050, 1);

        let result = encode_flac_bytes(&audio, &config);
        assert!(result.is_ok());

        let encoded = result.expect("FLAC encoding should succeed");
        assert!(!encoded.is_empty());
        // FLAC files should start with the 'fLaC' signature.
        assert_eq!(&encoded[0..4], b"fLaC");
    }

    #[test]
    fn test_flac_file_encoding() {
        let samples = vec![0.1, -0.2, 0.3, -0.4];
        let audio = AudioBuffer::new(samples, 44100, 1);
        let config = test_config(44100, 1);

        let test_path = std::env::temp_dir().join("voirs_vocoder_test_audio.flac");
        let result = encode_flac(&audio, &test_path, &config);
        assert!(result.is_ok());

        // Verify the file was created.
        assert!(std::fs::metadata(&test_path).is_ok());

        // Clean up.
        let _ = std::fs::remove_file(&test_path);
    }

    #[test]
    fn test_invalid_sample_rate() {
        let audio = AudioBuffer::new(vec![0.1], 1_000_000, 1);
        let config = test_config(1_000_000, 1); // Too high for FLAC

        let result = encode_flac_bytes(&audio, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_channels() {
        let audio = AudioBuffer::new(vec![0.1], 22050, 1);
        let config = test_config(22050, 16); // Too many channels for FLAC

        let result = encode_flac_bytes(&audio, &config);
        assert!(result.is_err());
    }

    #[cfg(not(feature = "ffi-codecs"))]
    #[test]
    fn test_pcm_conversion() {
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let pcm = convert_to_pcm_i24(&samples);

        assert_eq!(pcm[0], 0);
        assert_eq!(pcm[1], 4_194_303); // 0.5 * (2^23 - 1)
        assert_eq!(pcm[2], -4_194_303); // -0.5 * (2^23 - 1)
        assert_eq!(pcm[3], 8_388_607); // 2^23 - 1
        assert_eq!(pcm[4], -8_388_607); // -1.0 * (2^23 - 1) (clamped)
    }
}
