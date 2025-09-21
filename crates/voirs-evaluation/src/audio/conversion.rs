//! Audio conversion utilities for sample rate, channel, and format conversion.
//!
//! This module provides high-quality audio conversion functions for processing
//! audio data in the VoiRS evaluation pipeline.

use super::{AudioIoError, AudioIoResult};
use voirs_sdk::AudioBuffer;

/// Convert audio sample rate using high-quality resampling
///
/// # Arguments
///
/// * `audio` - Input audio buffer
/// * `target_sample_rate` - Target sample rate in Hz
/// * `quality` - Resampling quality (0-10, higher is better)
///
/// # Returns
///
/// Returns the resampled audio buffer
///
/// # Errors
///
/// Returns error if resampling fails or invalid parameters are provided
pub fn convert_sample_rate(
    audio: AudioBuffer,
    target_sample_rate: u32,
    quality: u8,
) -> AudioIoResult<AudioBuffer> {
    if audio.sample_rate() == target_sample_rate {
        return Ok(audio);
    }

    let ratio = audio.sample_rate() as f64 / target_sample_rate as f64;
    let channels = audio.channels();
    let input_samples = audio.samples();
    let frames_in = input_samples.len() / channels as usize;
    let frames_out = (frames_in as f64 / ratio).ceil() as usize;

    let mut output_samples = Vec::with_capacity(frames_out * channels as usize);

    // Use higher quality interpolation based on quality setting
    if quality >= 7 {
        // High-quality cubic interpolation
        for frame_out in 0..frames_out {
            let pos = frame_out as f64 * ratio;
            let input_frame = pos.floor() as usize;
            let frac = pos - input_frame as f64;

            for ch in 0..channels as usize {
                let sample = cubic_interpolate(
                    input_samples,
                    input_frame,
                    ch,
                    channels as usize,
                    frac,
                    frames_in,
                );
                output_samples.push(sample);
            }
        }
    } else {
        // Linear interpolation for lower quality settings
        for frame_out in 0..frames_out {
            let pos = frame_out as f64 * ratio;
            let input_frame = pos.floor() as usize;
            let frac = pos - input_frame as f64;

            for ch in 0..channels as usize {
                let sample = linear_interpolate(
                    input_samples,
                    input_frame,
                    ch,
                    channels as usize,
                    frac,
                    frames_in,
                );
                output_samples.push(sample);
            }
        }
    }

    Ok(AudioBuffer::new(
        output_samples,
        target_sample_rate,
        channels,
    ))
}

/// Convert audio between different channel configurations
///
/// # Arguments
///
/// * `audio` - Input audio buffer
/// * `target_channels` - Target number of channels
///
/// # Returns
///
/// Returns the channel-converted audio buffer
///
/// # Errors
///
/// Returns error if conversion fails or invalid parameters are provided
pub fn convert_channels(audio: AudioBuffer, target_channels: u32) -> AudioIoResult<AudioBuffer> {
    if audio.channels() == target_channels {
        return Ok(audio);
    }

    let input_samples = audio.samples();
    let input_channels = audio.channels();
    let frames = input_samples.len() / input_channels as usize;
    let mut output_samples = Vec::with_capacity(frames * target_channels as usize);

    for frame in 0..frames {
        match (input_channels, target_channels) {
            (1, 2) => {
                // Mono to stereo - duplicate channel
                let sample = input_samples[frame];
                output_samples.push(sample);
                output_samples.push(sample);
            }
            (2, 1) => {
                // Stereo to mono - average channels
                let left = input_samples[frame * 2];
                let right = input_samples[frame * 2 + 1];
                output_samples.push((left + right) * 0.5);
            }
            (1, n) if n > 2 => {
                // Mono to multi-channel - duplicate to all channels
                let sample = input_samples[frame];
                for _ in 0..n {
                    output_samples.push(sample);
                }
            }
            (n, 1) if n > 2 => {
                // Multi-channel to mono - average all channels
                let mut sum = 0.0;
                for ch in 0..n as usize {
                    sum += input_samples[frame * n as usize + ch];
                }
                output_samples.push(sum / n as f32);
            }
            (n, 2) if n > 2 => {
                // Multi-channel to stereo - mix down
                let mut left = 0.0;
                let mut right = 0.0;
                for ch in 0..n as usize {
                    let sample = input_samples[frame * n as usize + ch];
                    if ch % 2 == 0 {
                        left += sample;
                    } else {
                        right += sample;
                    }
                }
                let left_count = (n + 1) / 2;
                let right_count = n / 2;
                output_samples.push(left / left_count as f32);
                output_samples.push(if right_count > 0 {
                    right / right_count as f32
                } else {
                    left / left_count as f32
                });
            }
            (2, n) if n > 2 => {
                // Stereo to multi-channel - distribute left/right
                let left = input_samples[frame * 2];
                let right = input_samples[frame * 2 + 1];
                for ch in 0..n as usize {
                    if ch % 2 == 0 {
                        output_samples.push(left);
                    } else {
                        output_samples.push(right);
                    }
                }
            }
            (from, to) => {
                // General case - simple mapping
                for ch in 0..to as usize {
                    if ch < from as usize {
                        output_samples.push(input_samples[frame * from as usize + ch]);
                    } else {
                        // Pad with zeros
                        output_samples.push(0.0);
                    }
                }
            }
        }
    }

    Ok(AudioBuffer::new(
        output_samples,
        audio.sample_rate(),
        target_channels,
    ))
}

/// Normalize audio to [-1.0, 1.0] range
///
/// # Arguments
///
/// * `audio` - Input audio buffer
///
/// # Returns
///
/// Returns the normalized audio buffer
///
/// # Errors
///
/// Returns error if normalization fails
pub fn normalize_audio(audio: AudioBuffer) -> AudioIoResult<AudioBuffer> {
    let samples = audio.samples();

    // Find peak amplitude
    let peak = samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);

    if peak == 0.0 || peak == 1.0 {
        // Already normalized or silent
        return Ok(audio);
    }

    // Apply normalization
    let scale = 1.0 / peak;
    let normalized_samples: Vec<f32> = samples.iter().map(|&s| s * scale).collect();

    Ok(AudioBuffer::new(
        normalized_samples,
        audio.sample_rate(),
        audio.channels(),
    ))
}

/// Remove DC offset from audio
///
/// # Arguments
///
/// * `audio` - Input audio buffer
///
/// # Returns
///
/// Returns the DC offset corrected audio buffer
///
/// # Errors
///
/// Returns error if DC removal fails
pub fn remove_dc_offset(audio: AudioBuffer) -> AudioIoResult<AudioBuffer> {
    let samples = audio.samples();
    let channels = audio.channels() as usize;
    let frames = samples.len() / channels;

    // Calculate DC offset for each channel
    let mut dc_offsets = vec![0.0f32; channels];
    for ch in 0..channels {
        let mut sum = 0.0;
        for frame in 0..frames {
            sum += samples[frame * channels + ch];
        }
        dc_offsets[ch] = sum / frames as f32;
    }

    // Remove DC offset
    let corrected_samples: Vec<f32> = samples
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let ch = i % channels;
            s - dc_offsets[ch]
        })
        .collect();

    Ok(AudioBuffer::new(
        corrected_samples,
        audio.sample_rate(),
        audio.channels(),
    ))
}

/// Apply gain to audio buffer
///
/// # Arguments
///
/// * `audio` - Input audio buffer
/// * `gain_db` - Gain in decibels
///
/// # Returns
///
/// Returns the gain-adjusted audio buffer
///
/// # Errors
///
/// Returns error if gain application fails
pub fn apply_gain(audio: AudioBuffer, gain_db: f32) -> AudioIoResult<AudioBuffer> {
    let gain_linear = 10.0f32.powf(gain_db / 20.0);
    let samples = audio.samples();

    let amplified_samples: Vec<f32> = samples
        .iter()
        .map(|&s| (s * gain_linear).clamp(-1.0, 1.0))
        .collect();

    Ok(AudioBuffer::new(
        amplified_samples,
        audio.sample_rate(),
        audio.channels(),
    ))
}

/// High-quality cubic interpolation
fn cubic_interpolate(
    samples: &[f32],
    frame: usize,
    channel: usize,
    channels: usize,
    frac: f64,
    total_frames: usize,
) -> f32 {
    let get_sample = |f: isize| -> f32 {
        if f < 0 || f as usize >= total_frames {
            0.0
        } else {
            samples[f as usize * channels + channel]
        }
    };

    let y0 = get_sample(frame as isize - 1);
    let y1 = get_sample(frame as isize);
    let y2 = get_sample(frame as isize + 1);
    let y3 = get_sample(frame as isize + 2);

    let frac = frac as f32;
    let a0 = y3 - y2 - y0 + y1;
    let a1 = y0 - y1 - a0;
    let a2 = y2 - y0;
    let a3 = y1;

    a0 * frac * frac * frac + a1 * frac * frac + a2 * frac + a3
}

/// Linear interpolation
fn linear_interpolate(
    samples: &[f32],
    frame: usize,
    channel: usize,
    channels: usize,
    frac: f64,
    total_frames: usize,
) -> f32 {
    let s0 = if frame < total_frames {
        samples[frame * channels + channel]
    } else {
        0.0
    };

    let s1 = if frame + 1 < total_frames {
        samples[(frame + 1) * channels + channel]
    } else {
        s0
    };

    s0 + frac as f32 * (s1 - s0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_rate_conversion() {
        let samples = vec![0.5f32; 8000]; // 0.5 seconds at 16kHz
        let audio = AudioBuffer::new(samples, 16000, 1);

        let result = convert_sample_rate(audio, 32000, 7);
        assert!(result.is_ok());

        let converted = result.unwrap();
        assert_eq!(converted.sample_rate(), 32000);
        assert_eq!(converted.channels(), 1);
    }

    #[test]
    fn test_sample_rate_no_conversion() {
        let samples = vec![0.5f32; 8000];
        let audio = AudioBuffer::new(samples, 16000, 1);

        let result = convert_sample_rate(audio, 16000, 7);
        assert!(result.is_ok());

        let converted = result.unwrap();
        assert_eq!(converted.sample_rate(), 16000);
    }

    #[test]
    fn test_mono_to_stereo() {
        let samples = vec![0.5f32; 1000];
        let audio = AudioBuffer::new(samples, 16000, 1);

        let result = convert_channels(audio, 2);
        assert!(result.is_ok());

        let converted = result.unwrap();
        assert_eq!(converted.channels(), 2);
        assert_eq!(converted.samples().len(), 2000);
    }

    #[test]
    fn test_stereo_to_mono() {
        let samples = vec![0.5f32; 2000];
        let audio = AudioBuffer::new(samples, 16000, 2);

        let result = convert_channels(audio, 1);
        assert!(result.is_ok());

        let converted = result.unwrap();
        assert_eq!(converted.channels(), 1);
        assert_eq!(converted.samples().len(), 1000);
    }

    #[test]
    fn test_normalization() {
        let samples = vec![0.1f32, 0.2f32, -0.3f32, 0.4f32];
        let audio = AudioBuffer::new(samples, 16000, 1);

        let result = normalize_audio(audio);
        assert!(result.is_ok());

        let normalized = result.unwrap();
        let peak = normalized
            .samples()
            .iter()
            .map(|&s| s.abs())
            .fold(0.0f32, f32::max);
        assert!((peak - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_dc_offset_removal() {
        let samples = vec![0.6f32, 0.7f32, 0.5f32, 0.8f32]; // DC offset of ~0.65
        let audio = AudioBuffer::new(samples, 16000, 1);

        let result = remove_dc_offset(audio);
        assert!(result.is_ok());

        let corrected = result.unwrap();
        let mean = corrected.samples().iter().sum::<f32>() / corrected.samples().len() as f32;
        assert!(mean.abs() < 0.001); // Mean should be close to zero
    }

    #[test]
    fn test_gain_application() {
        let samples = vec![0.1f32; 1000];
        let audio = AudioBuffer::new(samples, 16000, 1);

        let result = apply_gain(audio, 6.0); // +6dB gain
        assert!(result.is_ok());

        let amplified = result.unwrap();
        let expected_amplitude = 0.1 * 10.0f32.powf(6.0 / 20.0);
        assert!((amplified.samples()[0] - expected_amplitude).abs() < 0.001);
    }

    #[test]
    fn test_cubic_interpolation() {
        let samples = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let channels = 1;
        let total_frames = 5;

        let result = cubic_interpolate(&samples, 1, 0, channels, 0.5, total_frames);
        // Should interpolate between 1.0 and 0.0
        assert!(result > 0.0 && result < 1.0);
    }

    #[test]
    fn test_linear_interpolation() {
        let samples = vec![0.0, 1.0, 0.0];
        let channels = 1;
        let total_frames = 3;

        let result = linear_interpolate(&samples, 0, 0, channels, 0.5, total_frames);
        assert_eq!(result, 0.5); // Midpoint between 0.0 and 1.0
    }
}
