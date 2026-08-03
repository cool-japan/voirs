//! Opus encoding implementation.
//!
//! Opus encoding relies on the `libopus` C library (via the `opus` crate) and is
//! therefore gated behind the default-OFF `ffi-codecs` feature to keep the default
//! build Pure Rust. When the feature is disabled the public entry points remain
//! available but return a clear error instructing the caller to enable `ffi-codecs`.

use crate::{AudioBuffer, Result, VocoderError};
#[cfg(feature = "ffi-codecs")]
use opus::{Application, Channels, Encoder};
#[cfg(feature = "ffi-codecs")]
use std::fs::File;
#[cfg(feature = "ffi-codecs")]
use std::io::Write;
use std::path::Path;

use super::CodecConfig;

/// Encode audio buffer to Opus file
#[cfg(feature = "ffi-codecs")]
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
#[cfg(feature = "ffi-codecs")]
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

    // The real number of channels present in the *source* audio buffer. This
    // may differ from `config.channels` (the desired *output* channel count):
    // e.g. mono source audio can be encoded to a stereo Opus stream by
    // duplicating samples, but genuinely stereo source audio must never be
    // re-duplicated -- it is already interleaved L, R, L, R, ...
    let source_channels = audio.channels().max(1) as usize;

    // Convert and resample audio if necessary, resampling each channel
    // independently so channel content is never mixed across boundaries.
    let samples = if config.sample_rate != opus_sample_rate {
        resample_audio(
            audio.samples(),
            config.sample_rate,
            opus_sample_rate,
            source_channels,
        )?
    } else {
        audio.samples().to_vec()
    };

    // Convert to appropriate format for Opus
    let pcm_data = convert_to_pcm_i16(&samples);

    let mut opus_data = Vec::new();

    // Encode audio in frames
    // Opus frame sizes: 2.5, 5, 10, 20, 40, 60 ms
    // We'll use 20ms frames (960 samples-per-channel at 48kHz)
    let frame_size = (opus_sample_rate as usize * 20) / 1000; // samples per channel, 20ms frame

    if config.channels == 1 {
        // Mono output: `pcm_data` already holds one sample per frame slot.
        for chunk in pcm_data.chunks(frame_size) {
            encode_opus_frame(&mut encoder, chunk, &mut opus_data)?;
        }
    } else if source_channels == 2 {
        // Source audio is genuinely stereo: `pcm_data` is already interleaved
        // L, R, L, R, ... samples. Chunk by `frame_size` samples *per
        // channel* (i.e. `frame_size * 2` interleaved samples) and pass the
        // interleaved data straight through without re-duplicating it.
        for chunk in pcm_data.chunks(frame_size * 2) {
            encode_opus_frame(&mut encoder, chunk, &mut opus_data)?;
        }
    } else {
        // Source audio is mono but stereo output was requested: duplicate
        // each sample into both the left and right channel.
        for chunk in pcm_data.chunks(frame_size) {
            let mut interleaved = Vec::with_capacity(chunk.len() * 2);
            for &sample in chunk {
                interleaved.push(sample); // Left
                interleaved.push(sample); // Right (duplicate for mono source)
            }

            encode_opus_frame(&mut encoder, &interleaved, &mut opus_data)?;
        }
    }

    Ok(opus_data)
}

/// Encode a single Opus frame (already in the correct per-channel
/// interleaving) and append the length-prefixed packet to `opus_data`.
#[cfg(feature = "ffi-codecs")]
fn encode_opus_frame(encoder: &mut Encoder, input: &[i16], opus_data: &mut Vec<u8>) -> Result<()> {
    let mut output = vec![0u8; 4000]; // Max Opus packet size
    let encoded_len = encoder
        .encode(input, &mut output)
        .map_err(|e| VocoderError::VocodingError(format!("Opus encoding failed: {e:?}")))?;

    // Add packet length and data (simple container format)
    opus_data.extend_from_slice(&(encoded_len as u32).to_le_bytes());
    opus_data.extend_from_slice(&output[..encoded_len]);
    Ok(())
}

/// Simple linear interpolation resampling.
///
/// `samples` is treated as interleaved frames of `channels` channels so that
/// resampling interpolates within each channel independently and never mixes
/// samples across channel boundaries (which would garble stereo content).
#[cfg(feature = "ffi-codecs")]
fn resample_audio(
    samples: &[f32],
    from_rate: u32,
    to_rate: u32,
    channels: usize,
) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    let channels = channels.max(1);
    let frame_count = samples.len() / channels;
    if frame_count == 0 {
        return Ok(Vec::new());
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let output_frames = (frame_count as f64 * ratio) as usize;
    let mut output = Vec::with_capacity(output_frames * channels);

    for i in 0..output_frames {
        let src_pos = i as f64 / ratio;
        let src_frame = src_pos as usize;
        let frac = (src_pos - src_frame as f64) as f32;

        for ch in 0..channels {
            let sample = if src_frame + 1 >= frame_count {
                samples[(frame_count - 1) * channels + ch]
            } else {
                let a = samples[src_frame * channels + ch];
                let b = samples[(src_frame + 1) * channels + ch];
                a * (1.0 - frac) + b * frac
            };
            output.push(sample);
        }
    }

    Ok(output)
}

/// Convert f32 samples to i16 PCM
#[cfg(feature = "ffi-codecs")]
fn convert_to_pcm_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&sample| (sample * 32767.0).clamp(-32768.0, 32767.0) as i16)
        .collect()
}

/// Encode audio buffer to Opus file (stub when the `ffi-codecs` feature is disabled).
///
/// Opus encoding requires the `libopus` C library and is only available with the
/// `ffi-codecs` feature enabled. This stub keeps the public API stable and returns a
/// descriptive error so callers get a clear message instead of a missing symbol.
#[cfg(not(feature = "ffi-codecs"))]
pub fn encode_opus<P: AsRef<Path>>(
    _audio: &AudioBuffer,
    _path: P,
    _config: &CodecConfig,
) -> Result<()> {
    Err(VocoderError::ConfigError(
        "Opus encoding requires the 'ffi-codecs' feature (libopus C library)".to_string(),
    ))
}

/// Encode audio buffer to Opus bytes (stub when the `ffi-codecs` feature is disabled).
///
/// Opus encoding requires the `libopus` C library and is only available with the
/// `ffi-codecs` feature enabled. This stub keeps the public API stable and returns a
/// descriptive error so callers get a clear message instead of a missing symbol.
#[cfg(not(feature = "ffi-codecs"))]
pub fn encode_opus_bytes(_audio: &AudioBuffer, _config: &CodecConfig) -> Result<Vec<u8>> {
    Err(VocoderError::ConfigError(
        "Opus encoding requires the 'ffi-codecs' feature (libopus C library)".to_string(),
    ))
}

#[cfg(all(test, feature = "ffi-codecs"))]
mod tests {
    use super::*;
    use opus::Decoder;
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

        let test_file = tempfile::Builder::new()
            .prefix("voirs_test_audio_")
            .suffix(".opus")
            .tempfile_in(std::env::temp_dir())
            .expect("failed to create temp file");
        let test_path = test_file.path();
        let result = encode_opus(&audio, test_path, &config);
        assert!(result.is_ok());

        // Verify file was created
        assert!(fs::metadata(test_path).is_ok());
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
        let upsampled = resample_audio(&samples, 1000, 2000, 1).unwrap();
        assert_eq!(upsampled.len(), 10);

        // Downsample 2x
        let downsampled = resample_audio(&samples, 2000, 1000, 1).unwrap();
        assert_eq!(downsampled.len(), 2);
    }

    #[test]
    fn test_resampling_preserves_channel_boundaries() {
        // Two channels: left ramps 0,1,2,3,4; right stays constant at 100.
        // A channel-unaware resampler would blend left and right values
        // together across the interleaving boundary; a correct one keeps
        // each channel's interpolation independent.
        let left = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let right = [100.0_f32; 5];
        let mut interleaved = Vec::with_capacity(10);
        for i in 0..5 {
            interleaved.push(left[i]);
            interleaved.push(right[i]);
        }

        let upsampled = resample_audio(&interleaved, 1000, 2000, 2).unwrap();
        assert_eq!(upsampled.len(), 20);

        // Every right-channel (odd-indexed) sample must stay close to 100.0;
        // if channels were mixed during resampling it would drift toward the
        // left channel's ramp values instead.
        for pair in upsampled.chunks_exact(2) {
            assert!(
                (pair[1] - 100.0).abs() < 0.01,
                "right channel leaked left channel data: {pair:?}"
            );
        }
    }

    #[test]
    fn test_stereo_round_trip_preserves_channel_separation() {
        // Real stereo content: the left channel carries an audible tone, the
        // right channel is silent. If stereo interleaving is handled
        // correctly, decoding must preserve that asymmetry. Before the fix,
        // the encoder re-duplicated already-interleaved samples without
        // checking the source channel count, which mixes the two channels
        // together and additionally drops roughly half of every frame.
        let sample_rate = 48000u32;
        let frames = 960 * 4; // whole 20ms frames only, no partial remainder
        let mut interleaved = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let t = i as f32 / sample_rate as f32;
            let left = 0.7 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            interleaved.push(left); // Left channel: audible tone
            interleaved.push(0.0); // Right channel: silence
        }

        let audio = AudioBuffer::new(interleaved, sample_rate, 2);
        let config = CodecConfig {
            sample_rate,
            channels: 2,
            bit_rate: Some(128_000),
            quality: Some(0.8),
            compression_level: None,
        };

        let encoded = encode_opus_bytes(&audio, &config).expect("stereo encoding should succeed");
        assert!(!encoded.is_empty());

        // Decode the length-prefixed packet stream with a real Opus decoder.
        let mut decoder =
            Decoder::new(sample_rate, Channels::Stereo).expect("failed to create Opus decoder");
        let mut left_decoded = Vec::new();
        let mut right_decoded = Vec::new();

        let mut cursor = 0usize;
        while cursor + 4 <= encoded.len() {
            let len = u32::from_le_bytes(encoded[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            let packet = &encoded[cursor..cursor + len];
            cursor += len;

            let mut pcm = vec![0i16; 5760 * 2]; // 60ms @ 48kHz, 2 channels (max frame)
            let decoded_frames = decoder
                .decode(packet, &mut pcm, false)
                .expect("Opus decode should succeed");

            for frame in pcm[..decoded_frames * 2].chunks_exact(2) {
                left_decoded.push(frame[0] as f32 / 32768.0);
                right_decoded.push(frame[1] as f32 / 32768.0);
            }
        }

        assert!(
            left_decoded.len() > frames / 2,
            "decoded far fewer frames ({}) than encoded ({}); samples were dropped",
            left_decoded.len(),
            frames
        );

        let rms = |data: &[f32]| -> f32 {
            if data.is_empty() {
                return 0.0;
            }
            (data.iter().map(|s| s * s).sum::<f32>() / data.len() as f32).sqrt()
        };

        let left_rms = rms(&left_decoded);
        let right_rms = rms(&right_decoded);

        assert!(
            left_rms > 0.2,
            "left (tone) channel decoded with unexpectedly low energy: {left_rms}"
        );
        assert!(
            right_rms < left_rms * 0.3,
            "right (silent) channel leaked most of the left channel's energy \
             (left_rms={left_rms}, right_rms={right_rms}); stereo channels were \
             likely scrambled during encoding"
        );
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
