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
///
/// # Safety
/// The `buffer` pointer must be valid and point to a properly initialized VoirsAudioBuffer.
/// The `config` pointer must be valid and point to a properly initialized VoirsAudioEffectConfig.
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

    let sample_rate = audio_buffer.sample_rate as f32;
    let samples =
        std::slice::from_raw_parts_mut(audio_buffer.samples, audio_buffer.length as usize);

    match effect_config.effect_type {
        1 => apply_reverb(samples, effect_config.strength, effect_config.param1),
        2 => apply_compression(samples, effect_config.strength, effect_config.param1),
        3 => apply_eq(
            samples,
            sample_rate,
            effect_config.strength,
            effect_config.param1,
            effect_config.param2,
        ),
        _ => {} // No effect
    }

    VoirsErrorCode::Success
}

/// Get audio buffer statistics
///
/// # Safety
/// The `buffer` pointer must be valid and point to a properly initialized VoirsAudioBuffer.
/// The `peak_level`, `rms_level`, and `dynamic_range` pointers must be valid and point to properly allocated memory for c_float values.
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
///
/// # Safety
/// The `source` pointer must be valid and point to a properly initialized VoirsAudioBuffer.
/// The `destination` pointer must be valid and point to properly allocated memory for a VoirsAudioBuffer.
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

    let new_samples = std::alloc::alloc(layout.expect("checked is_err above")) as *mut c_float;
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
///
/// # Safety
/// The `buffer1` pointer must be valid and point to a properly initialized VoirsAudioBuffer.
/// The `buffer2` pointer must be valid and point to a properly initialized VoirsAudioBuffer.
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
///
/// # Safety
/// The `buffer1` pointer must be valid and point to a properly initialized VoirsAudioBuffer.
/// The `buffer2` pointer must be valid and point to a properly initialized VoirsAudioBuffer.
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
///
/// # Safety
/// The `buffer` pointer must be valid and point to a properly initialized VoirsAudioBuffer.
/// The `filename` pointer must be valid and point to a null-terminated C string.
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
///
/// # Safety
/// The `buffer` pointer must be valid and point to a properly initialized VoirsAudioBuffer.
/// The `filename` pointer must be valid and point to a null-terminated C string.
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
///
/// # Safety
/// The `formats` pointer must be valid and point to properly allocated memory for a pointer to c_char array.
/// The `count` pointer must be valid and point to properly allocated memory for a c_uint value.
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
        static FORMAT_BUFFER: [*const c_char; 5] = const {
            [
                FORMAT_STRINGS[0].as_ptr() as *const c_char,
                FORMAT_STRINGS[1].as_ptr() as *const c_char,
                FORMAT_STRINGS[2].as_ptr() as *const c_char,
                FORMAT_STRINGS[3].as_ptr() as *const c_char,
                FORMAT_STRINGS[4].as_ptr() as *const c_char,
            ]
        };
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

/// Apply a parametric peaking equalizer to a buffer in place.
///
/// The C-ABI [`VoirsAudioEffectConfig`] only exposes three scalar parameters, so
/// they are mapped onto real EQ units as follows:
/// * `gain` is the `0.0..=1.0` effect strength; `0.5` is neutral and the usable
///   range is mapped linearly onto `±15 dB` of peaking gain. A strength of `0.5`
///   therefore produces a bit-exact identity.
/// * `frequency` is the band centre frequency in Hz (`param1`).
/// * `q_factor` is the band quality factor (`param2`); a non-positive value
///   selects a sensible default of `1.0`.
///
/// Multiple bands can be applied by calling [`apply_peaking_eq`] repeatedly on
/// the same buffer (each call filters in series).
fn apply_eq(samples: &mut [f32], sample_rate: f32, gain: f32, frequency: f32, q_factor: f32) {
    const MAX_EQ_GAIN_DB: f32 = 15.0;
    let gain_db = (gain.clamp(0.0, 1.0) - 0.5) * 2.0 * MAX_EQ_GAIN_DB;
    let q = if q_factor > 0.0 { q_factor } else { 1.0 };
    apply_peaking_eq(samples, sample_rate, frequency, q, gain_db);
}

/// Apply a single RBJ Audio-EQ-Cookbook peaking-EQ biquad band, in place.
///
/// Coefficients follow Robert Bristow-Johnson's cookbook (peaking EQ):
///
/// ```text
/// A   = 10^(gain_db / 40)
/// w0  = 2*pi * f0 / fs
/// a   = sin(w0) / (2*Q)          (alpha)
///
/// b0 = 1 + a*A          a0 = 1 + a/A
/// b1 = -2*cos(w0)       a1 = -2*cos(w0)
/// b2 = 1 - a*A          a2 = 1 - a/A
/// ```
///
/// The coefficients are normalized by `a0` and the difference equation is
/// evaluated with a direct-form II transposed structure:
///
/// ```text
/// y[n] = b0*x[n] + s1
/// s1   = b1*x[n] - a1*y[n] + s2
/// s2   = b2*x[n] - a2*y[n]
/// ```
///
/// A gain of exactly `0 dB` is a bit-exact identity (the function returns early),
/// as are degenerate parameters (non-finite/out-of-range `f0`, `Q <= 0`, or a
/// non-positive sample rate), which would otherwise yield NaN/inf coefficients.
fn apply_peaking_eq(samples: &mut [f32], sample_rate: f32, f0: f32, q: f32, gain_db: f32) {
    // 0 dB peaking gain leaves the signal untouched; short-circuit so the buffer
    // is returned bit-for-bit identical rather than relying on float round-trips.
    if gain_db == 0.0 || samples.is_empty() {
        return;
    }

    let nyquist = sample_rate * 0.5;
    if !sample_rate.is_finite()
        || sample_rate <= 0.0
        || !f0.is_finite()
        || f0 <= 0.0
        || f0 >= nyquist
        || !q.is_finite()
        || q <= 0.0
    {
        return;
    }

    let a = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f32::consts::PI * f0 / sample_rate;
    let (sin_w0, cos_w0) = w0.sin_cos();
    let alpha = sin_w0 / (2.0 * q);

    // Un-normalized peaking-EQ coefficients (RBJ cookbook).
    let a0 = 1.0 + alpha / a;
    let b0 = (1.0 + alpha * a) / a0;
    let b1 = (-2.0 * cos_w0) / a0;
    let b2 = (1.0 - alpha * a) / a0;
    let a1 = (-2.0 * cos_w0) / a0;
    let a2 = (1.0 - alpha / a) / a0;

    // Direct-form II transposed evaluation of the biquad.
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    for sample in samples.iter_mut() {
        let x = *sample;
        let y = b0 * x + s1;
        s1 = b1 * x - a1 * y + s2;
        s2 = b2 * x - a2 * y;
        *sample = y;
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

    /// Sum of squares ("energy") of a signal.
    fn energy(samples: &[f32]) -> f32 {
        samples.iter().map(|&x| x * x).sum()
    }

    /// Generate a unit-amplitude sine tone.
    fn sine_tone(freq: f32, sample_rate: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|n| (2.0 * std::f32::consts::PI * freq * n as f32 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_eq_0db_is_identity() {
        // A 0 dB peaking band must leave the signal bit-for-bit unchanged.
        let sample_rate = 48_000.0;
        let original = sine_tone(440.0, sample_rate, 2048);
        let mut processed = original.clone();
        apply_peaking_eq(&mut processed, sample_rate, 1_000.0, 1.0, 0.0);
        assert_eq!(original, processed);
    }

    #[test]
    fn test_eq_neutral_strength_is_identity() {
        // Strength 0.5 maps to 0 dB through `apply_eq`, i.e. an identity filter.
        let sample_rate = 44_100.0;
        let original = sine_tone(440.0, sample_rate, 1024);
        let mut processed = original.clone();
        apply_eq(&mut processed, sample_rate, 0.5, 1_000.0, 1.0);
        for (a, b) in original.iter().zip(processed.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_eq_boost_raises_energy_at_center_only() {
        let sample_rate = 48_000.0;
        let len = 9_600; // 0.2 s
        let f0 = 1_000.0;
        let gain_db = 12.0;
        let q = 1.0;

        // Tone at the EQ centre frequency.
        let center_tone = sine_tone(f0, sample_rate, len);
        let mut center_boosted = center_tone.clone();
        apply_peaking_eq(&mut center_boosted, sample_rate, f0, q, gain_db);

        // Tone three octaves above the centre (well outside the band).
        let distant_freq = 8_000.0;
        let distant_tone = sine_tone(distant_freq, sample_rate, len);
        let mut distant_processed = distant_tone.clone();
        apply_peaking_eq(&mut distant_processed, sample_rate, f0, q, gain_db);

        // Measure steady-state energy over the second half to skip filter warm-up.
        let half = len / 2;
        let center_before = energy(&center_tone[half..]);
        let center_after = energy(&center_boosted[half..]);
        let distant_before = energy(&distant_tone[half..]);
        let distant_after = energy(&distant_processed[half..]);

        // A +12 dB boost at f0 multiplies on-band energy by ~10^(12/10) ≈ 15.8×.
        assert!(
            center_after > center_before * 4.0,
            "expected strong boost at f0: {center_before} -> {center_after}"
        );
        // The distant tone should be essentially untouched (within 15%).
        let ratio = distant_after / distant_before;
        assert!(
            (0.85..=1.15).contains(&ratio),
            "distant tone energy changed too much: ratio = {ratio}"
        );
    }

    #[test]
    fn test_eq_invalid_parameters_are_identity() {
        let sample_rate = 48_000.0;
        let original = sine_tone(440.0, sample_rate, 512);

        // f0 above Nyquist -> identity.
        let mut above_nyquist = original.clone();
        apply_peaking_eq(&mut above_nyquist, sample_rate, 30_000.0, 1.0, 6.0);
        assert_eq!(original, above_nyquist);

        // Non-positive Q -> identity.
        let mut bad_q = original.clone();
        apply_peaking_eq(&mut bad_q, sample_rate, 1_000.0, 0.0, 6.0);
        assert_eq!(original, bad_q);

        // f0 == 0 (DC) -> identity.
        let mut dc = original.clone();
        apply_peaking_eq(&mut dc, sample_rate, 0.0, 1.0, 6.0);
        assert_eq!(original, dc);
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

        let samples = Box::new([0.1f32, 0.2, -0.1, -0.2, 0.0]);
        let buffer = VoirsAudioBuffer {
            samples: samples.as_ptr() as *mut f32,
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let tmp_path =
            std::env::temp_dir().join(format!("voirs_test_audio_{}.mp3", std::process::id()));
        let filename = CString::new(tmp_path.to_str().unwrap()).unwrap();

        unsafe {
            let result = voirs_audio_save_mp3(&buffer, filename.as_ptr(), 192, 2);
            // MP3 encoding is backed by voirs_vocoder's real codecs::AudioCodecEncoder
            // (AudioCodec::Mp3), so this now succeeds and writes a real file.
            assert_eq!(result, VoirsErrorCode::Success);
        }
        assert!(
            tmp_path.exists(),
            "expected voirs_audio_save_mp3 to create {}",
            tmp_path.display()
        );
        assert!(
            std::fs::metadata(&tmp_path)
                .map(|meta| meta.len() > 0)
                .unwrap_or(false),
            "expected the saved MP3 file to be non-empty"
        );

        // Test invalid parameters — always works regardless of feature
        unsafe {
            let result = voirs_audio_save_mp3(ptr::null(), filename.as_ptr(), 192, 2);
            assert_eq!(result, VoirsErrorCode::InvalidParameter);

            let result = voirs_audio_save_mp3(&buffer, ptr::null(), 192, 2);
            assert_eq!(result, VoirsErrorCode::InvalidParameter);
        }

        let _ = std::fs::remove_file(&tmp_path);
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
