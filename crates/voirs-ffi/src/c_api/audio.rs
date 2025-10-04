//! Extended C API for audio operations.
//!
//! This module provides enhanced audio processing functions for advanced
//! audio manipulation and analysis through the C API.

use crate::{VoirsAudioBuffer, VoirsErrorCode};
use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_int, c_uint};

/// Audio processing mode for enhanced operations
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoirsAudioMode {
    /// Real-time processing mode
    RealTime = 0,
    /// High-quality processing mode
    HighQuality = 1,
    /// Batch processing mode
    Batch = 2,
}

/// Audio effect configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VoirsAudioEffectConfig {
    /// Effect type (0=none, 1=reverb, 2=compression, 3=eq)
    pub effect_type: c_uint,
    /// Effect strength (0.0-1.0)
    pub strength: c_float,
    /// Additional parameter 1
    pub param1: c_float,
    /// Additional parameter 2
    pub param2: c_float,
    /// Enable/disable effect
    pub enabled: c_int,
}

impl Default for VoirsAudioEffectConfig {
    fn default() -> Self {
        Self {
            effect_type: 0,
            strength: 0.5,
            param1: 0.0,
            param2: 0.0,
            enabled: 0,
        }
    }
}

/// Apply audio effects to a buffer
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_apply_effects(
    buffer: *mut VoirsAudioBuffer,
    config: *const VoirsAudioEffectConfig,
) -> VoirsErrorCode {
    if buffer.is_null() || config.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let audio_buffer = &mut *buffer;
    let effect_config = &*config;

    if effect_config.enabled == 0 {
        return VoirsErrorCode::Success;
    }

    if audio_buffer.samples.is_null() || audio_buffer.length == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let samples =
        std::slice::from_raw_parts_mut(audio_buffer.samples, audio_buffer.length as usize);

    match effect_config.effect_type {
        1 => apply_reverb(samples, effect_config.strength, effect_config.param1),
        2 => apply_compression(samples, effect_config.strength, effect_config.param1),
        3 => apply_eq(
            samples,
            effect_config.strength,
            effect_config.param1,
            effect_config.param2,
        ),
        _ => {} // No effect
    }

    VoirsErrorCode::Success
}

/// Get audio buffer statistics
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_get_statistics(
    buffer: *const VoirsAudioBuffer,
    peak_level: *mut c_float,
    rms_level: *mut c_float,
    dynamic_range: *mut c_float,
) -> VoirsErrorCode {
    if buffer.is_null() || peak_level.is_null() || rms_level.is_null() || dynamic_range.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let audio_buffer = &*buffer;

    if audio_buffer.samples.is_null() || audio_buffer.length == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let samples = std::slice::from_raw_parts(audio_buffer.samples, audio_buffer.length as usize);

    let peak = samples.iter().map(|&x| x.abs()).fold(0.0, f32::max);
    let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
    let min_sample = samples.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_sample = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_sample - min_sample;

    *peak_level = peak;
    *rms_level = rms;
    *dynamic_range = range;

    VoirsErrorCode::Success
}

/// Create a copy of an audio buffer
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_duplicate(
    source: *const VoirsAudioBuffer,
    destination: *mut VoirsAudioBuffer,
) -> VoirsErrorCode {
    if source.is_null() || destination.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let src_buffer = &*source;
    let dst_buffer = &mut *destination;

    if src_buffer.samples.is_null() || src_buffer.length == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    // Allocate new memory for the destination
    let layout = std::alloc::Layout::from_size_align(
        src_buffer.length as usize * std::mem::size_of::<f32>(),
        std::mem::align_of::<f32>(),
    );

    if layout.is_err() {
        return VoirsErrorCode::InternalError;
    }

    let new_samples = std::alloc::alloc(layout.unwrap()) as *mut c_float;
    if new_samples.is_null() {
        return VoirsErrorCode::OutOfMemory;
    }

    // Copy the data
    std::ptr::copy_nonoverlapping(src_buffer.samples, new_samples, src_buffer.length as usize);

    // Set up destination buffer
    dst_buffer.samples = new_samples;
    dst_buffer.length = src_buffer.length;
    dst_buffer.sample_rate = src_buffer.sample_rate;
    dst_buffer.channels = src_buffer.channels;

    VoirsErrorCode::Success
}

/// Mix two audio buffers together
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_mix(
    buffer1: *mut VoirsAudioBuffer,
    buffer2: *const VoirsAudioBuffer,
    mix_ratio: c_float, // 0.0 = only buffer1, 1.0 = only buffer2, 0.5 = equal mix
) -> VoirsErrorCode {
    if buffer1.is_null() || buffer2.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let buf1 = &mut *buffer1;
    let buf2 = &*buffer2;

    if buf1.samples.is_null() || buf2.samples.is_null() || buf1.length == 0 || buf2.length == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    if buf1.length != buf2.length
        || buf1.channels != buf2.channels
        || buf1.sample_rate != buf2.sample_rate
    {
        return VoirsErrorCode::InvalidParameter;
    }

    let samples1 = std::slice::from_raw_parts_mut(buf1.samples, buf1.length as usize);
    let samples2 = std::slice::from_raw_parts(buf2.samples, buf2.length as usize);

    let ratio = mix_ratio.clamp(0.0, 1.0);
    let inv_ratio = 1.0 - ratio;

    for (s1, &s2) in samples1.iter_mut().zip(samples2.iter()) {
        *s1 = *s1 * inv_ratio + s2 * ratio;
    }

    VoirsErrorCode::Success
}

/// Crossfade between two audio buffers
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_crossfade(
    buffer1: *mut VoirsAudioBuffer,
    buffer2: *const VoirsAudioBuffer,
    fade_duration_ms: c_uint,
) -> VoirsErrorCode {
    if buffer1.is_null() || buffer2.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let buf1 = &mut *buffer1;
    let buf2 = &*buffer2;

    if buf1.samples.is_null() || buf2.samples.is_null() || buf1.length == 0 || buf2.length == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    if buf1.length != buf2.length
        || buf1.channels != buf2.channels
        || buf1.sample_rate != buf2.sample_rate
    {
        return VoirsErrorCode::InvalidParameter;
    }

    let samples1 = std::slice::from_raw_parts_mut(buf1.samples, buf1.length as usize);
    let samples2 = std::slice::from_raw_parts(buf2.samples, buf2.length as usize);

    let fade_samples = (fade_duration_ms as f32 * buf1.sample_rate as f32 / 1000.0) as usize;
    let fade_samples = fade_samples.min(samples1.len());

    for i in 0..fade_samples {
        let fade_ratio = i as f32 / fade_samples as f32;
        let inv_ratio = 1.0 - fade_ratio;
        samples1[i] = samples1[i] * inv_ratio + samples2[i] * fade_ratio;
    }

    VoirsErrorCode::Success
}

/// Save audio buffer as FLAC file
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_save_flac(
    buffer: *const VoirsAudioBuffer,
    filename: *const c_char,
    compression_level: c_uint, // 0-8, where 8 is highest compression
) -> VoirsErrorCode {
    if buffer.is_null() || filename.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let audio_buffer = &*buffer;

    if audio_buffer.samples.is_null() || audio_buffer.length == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    // Convert C string to Rust string
    let filename_str = match CStr::from_ptr(filename).to_str() {
        Ok(s) => s,
        Err(_) => return VoirsErrorCode::InvalidParameter,
    };

    // Get audio samples
    let samples = std::slice::from_raw_parts(audio_buffer.samples, audio_buffer.length as usize);

    // Validate compression level
    let compression = compression_level.min(8);

    // Save FLAC file using hound for now (integration with vocoder FLAC encoder would be ideal)
    match save_audio_as_flac(
        samples,
        audio_buffer.sample_rate,
        audio_buffer.channels.try_into().unwrap_or(2),
        filename_str,
        compression,
    ) {
        Ok(_) => VoirsErrorCode::Success,
        Err(_) => VoirsErrorCode::InternalError,
    }
}

/// Save audio buffer as MP3 file  
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_save_mp3(
    buffer: *const VoirsAudioBuffer,
    filename: *const c_char,
    bitrate: c_uint, // Bitrate in kbps (e.g., 128, 192, 320)
    quality: c_uint, // Quality level 0-9, where 0 is highest quality
) -> VoirsErrorCode {
    if buffer.is_null() || filename.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let audio_buffer = &*buffer;

    if audio_buffer.samples.is_null() || audio_buffer.length == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    // Convert C string to Rust string
    let filename_str = match CStr::from_ptr(filename).to_str() {
        Ok(s) => s,
        Err(_) => return VoirsErrorCode::InvalidParameter,
    };

    // Get audio samples
    let samples = std::slice::from_raw_parts(audio_buffer.samples, audio_buffer.length as usize);

    // Validate parameters
    let bitrate = if bitrate == 0 {
        128
    } else {
        bitrate.clamp(32, 320)
    };
    let quality = quality.min(9);

    // Save MP3 file using vocoder MP3 encoder or fallback
    match save_audio_as_mp3(
        samples,
        audio_buffer.sample_rate,
        audio_buffer.channels.try_into().unwrap_or(2),
        filename_str,
        bitrate,
        quality,
    ) {
        Ok(_) => VoirsErrorCode::Success,
        Err(_) => VoirsErrorCode::InternalError,
    }
}

/// Get supported audio formats
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_get_supported_formats(
    formats: *mut *const c_char,
    count: *mut c_uint,
) -> VoirsErrorCode {
    if formats.is_null() || count.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    // Static list of supported formats as null-terminated byte strings
    const FORMAT_STRINGS: [&[u8]; 5] = [b"wav\0", b"flac\0", b"mp3\0", b"ogg\0", b"opus\0"];

    // Thread-local storage for format pointers (safe for FFI)
    thread_local! {
        static FORMAT_BUFFER: [*const c_char; 5] = [
            FORMAT_STRINGS[0].as_ptr() as *const c_char,
            FORMAT_STRINGS[1].as_ptr() as *const c_char,
            FORMAT_STRINGS[2].as_ptr() as *const c_char,
            FORMAT_STRINGS[3].as_ptr() as *const c_char,
            FORMAT_STRINGS[4].as_ptr() as *const c_char,
        ];
    }

    FORMAT_BUFFER.with(|buffer| {
        *formats = buffer.as_ptr() as *const c_char;
    });
    *count = FORMAT_STRINGS.len() as c_uint;

    VoirsErrorCode::Success
}

// Helper functions for audio file I/O
fn save_audio_as_flac(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    filename: &str,
    compression_level: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    use voirs_vocoder::{
        codecs::{AudioCodec, AudioCodecEncoder, CodecConfig},
        AudioBuffer,
    };

    // Create AudioBuffer from samples
    let audio_buffer = AudioBuffer::new(samples.to_vec(), sample_rate, channels as u32);

    // Create codec configuration
    let config = CodecConfig {
        sample_rate,
        channels,
        bit_rate: None, // Not used for FLAC
        quality: None,  // Not used for FLAC
        compression_level: Some(compression_level),
    };

    // Create FLAC encoder
    let encoder = AudioCodecEncoder::new(AudioCodec::Flac, config);

    // Encode audio to FLAC file
    encoder
        .encode_to_file(&audio_buffer, filename)
        .map_err(|e| format!("FLAC encoding failed: {}", e))?;

    println!(
        "FLAC file saved with compression level {}",
        compression_level
    );

    Ok(())
}

fn save_audio_as_mp3(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    filename: &str,
    bitrate: u32,
    quality: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    use voirs_vocoder::{
        codecs::{AudioCodec, AudioCodecEncoder, CodecConfig},
        AudioBuffer,
    };

    // For MP3, ensure we have at least 2 channels (stereo) to avoid encoder issues
    let (audio_samples, final_channels) = if channels == 1 {
        // Convert mono to stereo by duplicating samples
        let mut stereo_samples = Vec::with_capacity(samples.len() * 2);
        for &sample in samples {
            stereo_samples.push(sample); // Left channel
            stereo_samples.push(sample); // Right channel (duplicate)
        }
        (stereo_samples, 2u16)
    } else {
        (samples.to_vec(), channels)
    };

    // Create AudioBuffer from samples
    let audio_buffer = AudioBuffer::new(audio_samples, sample_rate, final_channels as u32);

    // Create codec configuration
    let config = CodecConfig {
        sample_rate,
        channels: final_channels,
        bit_rate: Some(bitrate * 1000), // Convert from kbps to bps
        quality: Some(quality as f32 / 9.0), // Convert from 0-9 to 0.0-1.0
        compression_level: None,        // Not used for MP3
    };

    // Create MP3 encoder
    let encoder = AudioCodecEncoder::new(AudioCodec::Mp3, config);

    // Encode audio to MP3 file
    encoder
        .encode_to_file(&audio_buffer, filename)
        .map_err(|e| format!("MP3 encoding failed: {}", e))?;

    println!(
        "MP3 file saved with settings: {}kbps, quality {}",
        bitrate, quality
    );

    Ok(())
}

// Helper functions for audio effects
fn apply_reverb(samples: &mut [f32], strength: f32, decay: f32) {
    // Simple reverb implementation using delayed feedback
    let delay_samples = (samples.len() / 8).min(1024);
    let feedback = (strength * 0.6).clamp(0.0, 0.8);
    let decay_factor = decay.clamp(0.1, 0.9);

    for i in delay_samples..samples.len() {
        let delayed = samples[i - delay_samples] * feedback * decay_factor;
        samples[i] = samples[i] * (1.0 - strength * 0.3) + delayed * strength;
    }
}

fn apply_compression(samples: &mut [f32], ratio: f32, threshold: f32) {
    // Simple dynamic range compression
    let comp_ratio = (ratio * 4.0 + 1.0).clamp(1.0, 10.0);
    let thresh = threshold.clamp(0.1, 0.9);

    for sample in samples.iter_mut() {
        let abs_sample = sample.abs();
        if abs_sample > thresh {
            let excess = abs_sample - thresh;
            let compressed_excess = excess / comp_ratio;
            let new_magnitude = thresh + compressed_excess;
            *sample *= new_magnitude / abs_sample;
        }
    }
}

fn apply_eq(samples: &mut [f32], gain: f32, frequency: f32, q_factor: f32) {
    // Simple peak EQ filter
    let gain_linear = gain * 2.0 - 1.0; // Convert 0-1 to -1 to 1
    let _freq_norm = frequency.clamp(0.0, 1.0);
    let _q = q_factor.clamp(0.1, 10.0);

    // Simple gain adjustment based on frequency content
    // This is a simplified EQ - a real implementation would use proper filter coefficients
    for sample in samples.iter_mut() {
        *sample *= 1.0 + gain_linear * 0.3; // Simple gain adjustment
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_audio_effects_config_default() {
        let config = VoirsAudioEffectConfig::default();
        assert_eq!(config.effect_type, 0);
        assert_eq!(config.strength, 0.5);
        assert_eq!(config.enabled, 0);
    }

    #[test]
    fn test_audio_statistics() {
        // Use Box to ensure data is heap-allocated and stable
        let samples = Box::new([0.5f32, -0.5, 1.0, -1.0, 0.0]);
        let buffer = VoirsAudioBuffer {
            samples: samples.as_ptr() as *mut f32,
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let mut peak = 0.0;
        let mut rms = 0.0;
        let mut range = 0.0;

        unsafe {
            let result = voirs_audio_get_statistics(&buffer, &mut peak, &mut rms, &mut range);
            assert_eq!(result, VoirsErrorCode::Success);

            // The peak should be 1.0 (max absolute value)
            assert_eq!(peak, 1.0);
            assert!(rms > 0.0);
            assert_eq!(range, 2.0); // 1.0 - (-1.0) = 2.0
        }
    }

    #[test]
    fn test_audio_mix() {
        // Use Box to ensure data is heap-allocated and stable
        let mut samples1 = Box::new([0.5f32, 0.5, 0.5, 0.5]);
        let samples2 = Box::new([1.0f32, 1.0, 1.0, 1.0]);

        let mut buffer1 = VoirsAudioBuffer {
            samples: samples1.as_mut_ptr(),
            length: samples1.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples1.len() as f32 / 44100.0,
        };

        let buffer2 = VoirsAudioBuffer {
            samples: samples2.as_ptr() as *mut f32,
            length: samples2.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples2.len() as f32 / 44100.0,
        };

        unsafe {
            let result = voirs_audio_mix(&mut buffer1, &buffer2, 0.5);
            assert_eq!(result, VoirsErrorCode::Success);

            // Check the modified samples1 data directly
            assert_eq!(samples1[0], 0.75); // 0.5 * 0.5 + 1.0 * 0.5 = 0.25 + 0.5 = 0.75
        }
    }

    #[test]
    fn test_invalid_parameters() {
        unsafe {
            // Test null pointers
            let result = voirs_audio_apply_effects(ptr::null_mut(), ptr::null());
            assert_eq!(result, VoirsErrorCode::InvalidParameter);

            let result = voirs_audio_get_statistics(
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            );
            assert_eq!(result, VoirsErrorCode::InvalidParameter);
        }
    }

    #[test]
    fn test_flac_save_function() {
        use std::ffi::CString;

        // Create test audio data
        let samples = Box::new([0.1f32, 0.2, -0.1, -0.2, 0.0]);
        let buffer = VoirsAudioBuffer {
            samples: samples.as_ptr() as *mut f32,
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let filename = CString::new("/tmp/test_audio.flac").unwrap();

        unsafe {
            let result = voirs_audio_save_flac(&buffer, filename.as_ptr(), 5);
            // Should succeed in creating the file (even if using WAV fallback for now)
            assert_eq!(result, VoirsErrorCode::Success);
        }

        // Test invalid parameters
        unsafe {
            let result = voirs_audio_save_flac(ptr::null(), filename.as_ptr(), 5);
            assert_eq!(result, VoirsErrorCode::InvalidParameter);

            let result = voirs_audio_save_flac(&buffer, ptr::null(), 5);
            assert_eq!(result, VoirsErrorCode::InvalidParameter);
        }
    }

    #[test]
    fn test_mp3_save_function() {
        use std::ffi::CString;

        // Create test audio data
        let samples = Box::new([0.1f32, 0.2, -0.1, -0.2, 0.0]);
        let buffer = VoirsAudioBuffer {
            samples: samples.as_ptr() as *mut f32,
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let filename = CString::new("/tmp/test_audio.mp3").unwrap();

        unsafe {
            let result = voirs_audio_save_mp3(&buffer, filename.as_ptr(), 192, 2);
            // Should succeed in creating the file (even if using WAV fallback for now)
            assert_eq!(result, VoirsErrorCode::Success);
        }

        // Test invalid parameters
        unsafe {
            let result = voirs_audio_save_mp3(ptr::null(), filename.as_ptr(), 192, 2);
            assert_eq!(result, VoirsErrorCode::InvalidParameter);

            let result = voirs_audio_save_mp3(&buffer, ptr::null(), 192, 2);
            assert_eq!(result, VoirsErrorCode::InvalidParameter);
        }
    }

    #[test]
    fn test_supported_formats() {
        unsafe {
            let mut formats: *const c_char = ptr::null();
            let mut count: c_uint = 0;

            let result = voirs_audio_get_supported_formats(&mut formats, &mut count);
            assert_eq!(result, VoirsErrorCode::Success);
            assert!(count > 0);
            assert!(!formats.is_null());

            // Verify we have at least wav, flac, mp3
            assert!(count >= 3);
        }

        // Test invalid parameters
        unsafe {
            let result = voirs_audio_get_supported_formats(ptr::null_mut(), ptr::null_mut());
            assert_eq!(result, VoirsErrorCode::InvalidParameter);
        }
    }
}
