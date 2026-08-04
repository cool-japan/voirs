//! Audio format conversions for FFI operations.
//!
//! This module provides functions for converting between different audio formats,
//! sample rates, and data types to ensure compatibility across language boundaries.

use crate::{set_last_error, VoirsAudioBuffer, VoirsErrorCode};
use std::os::raw::{c_double, c_float, c_uint};

/// Sample format for audio conversion
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoirsSampleFormat {
    /// 32-bit floating point samples
    Float32 = 0,
    /// 16-bit signed integer samples  
    Int16 = 1,
    /// 24-bit signed integer samples
    Int24 = 2,
    /// 32-bit signed integer samples
    Int32 = 3,
    /// 64-bit floating point samples
    Float64 = 4,
    /// 8-bit unsigned integer samples
    UInt8 = 5,
    /// 16-bit unsigned integer samples
    UInt16 = 6,
    /// 32-bit unsigned integer samples
    UInt32 = 7,
}

/// Audio channel layout
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoirsChannelLayout {
    /// Mono audio (1 channel)
    Mono = 1,
    /// Stereo audio (2 channels)
    Stereo = 2,
    /// 5.1 surround (6 channels)
    Surround51 = 6,
    /// 7.1 surround (8 channels)
    Surround71 = 8,
}

/// Endianness for sample conversion
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoirsEndianness {
    /// Little endian byte order
    LittleEndian = 0,
    /// Big endian byte order
    BigEndian = 1,
    /// Native endianness (platform default)
    Native = 2,
}

/// Convert floating point samples to 16-bit signed integers
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `sample_count` floats.
/// The `output` pointer must be valid and point to a writable buffer of at least `sample_count` i16 values.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_float_to_int16(
    input: *const c_float,
    output: *mut i16,
    sample_count: c_uint,
    endianness: VoirsEndianness,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || sample_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, sample_count as usize);
    let output_slice = std::slice::from_raw_parts_mut(output, sample_count as usize);

    for (i, &sample) in input_slice.iter().enumerate() {
        // Clamp to [-1.0, 1.0] range
        let clamped = sample.clamp(-1.0, 1.0);

        // Convert to 16-bit integer
        let mut int_sample = (clamped * 32767.0).round() as i16;

        // Handle endianness
        if endianness == VoirsEndianness::BigEndian {
            int_sample = int_sample.to_be();
        } else if endianness == VoirsEndianness::LittleEndian {
            int_sample = int_sample.to_le();
        }
        // Native endianness requires no conversion

        output_slice[i] = int_sample;
    }

    VoirsErrorCode::Success
}

/// Convert floating point samples to 32-bit signed integers
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `sample_count` floats.
/// The `output` pointer must be valid and point to a writable buffer of at least `sample_count` i32 values.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_float_to_int32(
    input: *const c_float,
    output: *mut i32,
    sample_count: c_uint,
    endianness: VoirsEndianness,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || sample_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, sample_count as usize);
    let output_slice = std::slice::from_raw_parts_mut(output, sample_count as usize);

    for (i, &sample) in input_slice.iter().enumerate() {
        // Clamp to [-1.0, 1.0] range
        let clamped = sample.clamp(-1.0, 1.0);

        // Convert to 32-bit integer
        let mut int_sample = (clamped * 2147483647.0).round() as i32;

        // Handle endianness
        if endianness == VoirsEndianness::BigEndian {
            int_sample = int_sample.to_be();
        } else if endianness == VoirsEndianness::LittleEndian {
            int_sample = int_sample.to_le();
        }

        output_slice[i] = int_sample;
    }

    VoirsErrorCode::Success
}

/// Convert 16-bit signed integers to floating point samples
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `sample_count` i16 values.
/// The `output` pointer must be valid and point to a writable buffer of at least `sample_count` floats.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_int16_to_float(
    input: *const i16,
    output: *mut c_float,
    sample_count: c_uint,
    endianness: VoirsEndianness,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || sample_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, sample_count as usize);
    let output_slice = std::slice::from_raw_parts_mut(output, sample_count as usize);

    for (i, &sample) in input_slice.iter().enumerate() {
        // Handle endianness
        let corrected_sample = match endianness {
            VoirsEndianness::BigEndian => i16::from_be(sample),
            VoirsEndianness::LittleEndian => i16::from_le(sample),
            VoirsEndianness::Native => sample,
        };

        // Convert to float [-1.0, 1.0]
        output_slice[i] = corrected_sample as f32 / 32767.0;
    }

    VoirsErrorCode::Success
}

/// Convert floating point samples to 64-bit floating point samples
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `sample_count` f32 values.
/// The `output` pointer must be valid and point to a writable buffer of at least `sample_count` f64 values.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_float_to_double(
    input: *const c_float,
    output: *mut c_double,
    sample_count: c_uint,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || sample_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, sample_count as usize);
    let output_slice = std::slice::from_raw_parts_mut(output, sample_count as usize);

    for (i, &sample) in input_slice.iter().enumerate() {
        output_slice[i] = sample as f64;
    }

    VoirsErrorCode::Success
}

/// Convert 64-bit floating point samples to 32-bit floating point samples
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `sample_count` f64 values.
/// The `output` pointer must be valid and point to a writable buffer of at least `sample_count` f32 values.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_double_to_float(
    input: *const c_double,
    output: *mut c_float,
    sample_count: c_uint,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || sample_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, sample_count as usize);
    let output_slice = std::slice::from_raw_parts_mut(output, sample_count as usize);

    for (i, &sample) in input_slice.iter().enumerate() {
        output_slice[i] = sample as f32;
    }

    VoirsErrorCode::Success
}

/// Convert floating point samples to 24-bit signed integers (stored as i32)
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `sample_count` floats.
/// The `output` pointer must be valid and point to a writable buffer of at least `sample_count` i32 values.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_float_to_int24(
    input: *const c_float,
    output: *mut i32,
    sample_count: c_uint,
    endianness: VoirsEndianness,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || sample_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, sample_count as usize);
    let output_slice = std::slice::from_raw_parts_mut(output, sample_count as usize);

    const INT24_MAX: f32 = 8388607.0; // 2^23 - 1
    const INT24_MIN: f32 = -8388608.0; // -2^23

    for (i, &sample) in input_slice.iter().enumerate() {
        // Clamp to [-1.0, 1.0] range
        let clamped = sample.clamp(-1.0, 1.0);

        // Convert to 24-bit integer (stored in i32)
        let mut int_sample = if clamped >= 0.0 {
            (clamped * INT24_MAX).round() as i32
        } else {
            (clamped * -INT24_MIN).round() as i32
        };

        // Clamp to 24-bit range
        int_sample = int_sample.clamp(-8388608, 8388607);

        // Handle endianness
        if endianness == VoirsEndianness::BigEndian {
            int_sample = int_sample.to_be();
        } else if endianness == VoirsEndianness::LittleEndian {
            int_sample = int_sample.to_le();
        }

        output_slice[i] = int_sample;
    }

    VoirsErrorCode::Success
}

/// Convert 24-bit signed integers (stored as i32) to floating point samples
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `sample_count` i32 values.
/// The `output` pointer must be valid and point to a writable buffer of at least `sample_count` floats.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_int24_to_float(
    input: *const i32,
    output: *mut c_float,
    sample_count: c_uint,
    endianness: VoirsEndianness,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || sample_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, sample_count as usize);
    let output_slice = std::slice::from_raw_parts_mut(output, sample_count as usize);

    const INT24_MAX: f32 = 8388607.0; // 2^23 - 1

    for (i, &sample) in input_slice.iter().enumerate() {
        // Handle endianness
        let corrected_sample = match endianness {
            VoirsEndianness::BigEndian => i32::from_be(sample),
            VoirsEndianness::LittleEndian => i32::from_le(sample),
            VoirsEndianness::Native => sample,
        };

        // Convert to float [-1.0, 1.0]
        output_slice[i] = corrected_sample as f32 / INT24_MAX;
    }

    VoirsErrorCode::Success
}

/// Convert floating point samples to 8-bit unsigned integers
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `sample_count` floats.
/// The `output` pointer must be valid and point to a writable buffer of at least `sample_count` u8 values.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_float_to_uint8(
    input: *const c_float,
    output: *mut u8,
    sample_count: c_uint,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || sample_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, sample_count as usize);
    let output_slice = std::slice::from_raw_parts_mut(output, sample_count as usize);

    for (i, &sample) in input_slice.iter().enumerate() {
        // Clamp to [-1.0, 1.0] range and convert to [0, 255]
        let clamped = sample.clamp(-1.0, 1.0);
        let uint_sample = ((clamped + 1.0) * 127.5 + 0.5).floor() as u8;
        output_slice[i] = uint_sample;
    }

    VoirsErrorCode::Success
}

/// Convert 8-bit unsigned integers to floating point samples
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `sample_count` u8 values.
/// The `output` pointer must be valid and point to a writable buffer of at least `sample_count` floats.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_uint8_to_float(
    input: *const u8,
    output: *mut c_float,
    sample_count: c_uint,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || sample_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, sample_count as usize);
    let output_slice = std::slice::from_raw_parts_mut(output, sample_count as usize);

    for (i, &sample) in input_slice.iter().enumerate() {
        // Convert from [0, 255] to [-1.0, 1.0]
        output_slice[i] = (sample as f32 / 127.5) - 1.0;
    }

    VoirsErrorCode::Success
}

/// Convert floating point samples to 16-bit unsigned integers
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `sample_count` floats.
/// The `output` pointer must be valid and point to a writable buffer of at least `sample_count` u16 values.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_float_to_uint16(
    input: *const c_float,
    output: *mut u16,
    sample_count: c_uint,
    endianness: VoirsEndianness,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || sample_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, sample_count as usize);
    let output_slice = std::slice::from_raw_parts_mut(output, sample_count as usize);

    for (i, &sample) in input_slice.iter().enumerate() {
        // Clamp to [-1.0, 1.0] range and convert to [0, 65535]
        let clamped = sample.clamp(-1.0, 1.0);
        let mut uint_sample = ((clamped + 1.0) * 32767.5 + 0.5).floor() as u16;

        // Handle endianness
        if endianness == VoirsEndianness::BigEndian {
            uint_sample = uint_sample.to_be();
        } else if endianness == VoirsEndianness::LittleEndian {
            uint_sample = uint_sample.to_le();
        }

        output_slice[i] = uint_sample;
    }

    VoirsErrorCode::Success
}

/// Convert 16-bit unsigned integers to floating point samples
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `sample_count` u16 values.
/// The `output` pointer must be valid and point to a writable buffer of at least `sample_count` floats.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_uint16_to_float(
    input: *const u16,
    output: *mut c_float,
    sample_count: c_uint,
    endianness: VoirsEndianness,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || sample_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, sample_count as usize);
    let output_slice = std::slice::from_raw_parts_mut(output, sample_count as usize);

    for (i, &sample) in input_slice.iter().enumerate() {
        // Handle endianness
        let corrected_sample = match endianness {
            VoirsEndianness::BigEndian => u16::from_be(sample),
            VoirsEndianness::LittleEndian => u16::from_le(sample),
            VoirsEndianness::Native => sample,
        };

        // Convert from [0, 65535] to [-1.0, 1.0]
        output_slice[i] = (corrected_sample as f32 / 32767.5) - 1.0;
    }

    VoirsErrorCode::Success
}

/// Convert floating point samples to 32-bit unsigned integers
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `sample_count` floats.
/// The `output` pointer must be valid and point to a writable buffer of at least `sample_count` u32 values.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_float_to_uint32(
    input: *const c_float,
    output: *mut u32,
    sample_count: c_uint,
    endianness: VoirsEndianness,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || sample_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, sample_count as usize);
    let output_slice = std::slice::from_raw_parts_mut(output, sample_count as usize);

    for (i, &sample) in input_slice.iter().enumerate() {
        // Clamp to [-1.0, 1.0] range and convert to [0, 4294967295]
        let clamped = sample.clamp(-1.0, 1.0);
        let mut uint_sample = ((clamped + 1.0) * 2147483647.5).round() as u32;

        // Handle endianness
        if endianness == VoirsEndianness::BigEndian {
            uint_sample = uint_sample.to_be();
        } else if endianness == VoirsEndianness::LittleEndian {
            uint_sample = uint_sample.to_le();
        }

        output_slice[i] = uint_sample;
    }

    VoirsErrorCode::Success
}

/// Convert 32-bit unsigned integers to floating point samples
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `sample_count` u32 values.
/// The `output` pointer must be valid and point to a writable buffer of at least `sample_count` floats.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_uint32_to_float(
    input: *const u32,
    output: *mut c_float,
    sample_count: c_uint,
    endianness: VoirsEndianness,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || sample_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, sample_count as usize);
    let output_slice = std::slice::from_raw_parts_mut(output, sample_count as usize);

    for (i, &sample) in input_slice.iter().enumerate() {
        // Handle endianness
        let corrected_sample = match endianness {
            VoirsEndianness::BigEndian => u32::from_be(sample),
            VoirsEndianness::LittleEndian => u32::from_le(sample),
            VoirsEndianness::Native => sample,
        };

        // Convert from [0, 4294967295] to [-1.0, 1.0]
        output_slice[i] = (corrected_sample as f32 / 2147483647.5) - 1.0;
    }

    VoirsErrorCode::Success
}

/// Convert mono audio to stereo by duplicating channels
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `sample_count` floats.
/// The `output` pointer must be valid and point to a writable buffer of at least `sample_count * 2` floats.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_mono_to_stereo(
    input: *const c_float,
    output: *mut c_float,
    sample_count: c_uint,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || sample_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, sample_count as usize);
    let output_slice = std::slice::from_raw_parts_mut(output, (sample_count * 2) as usize);

    for (i, &sample) in input_slice.iter().enumerate() {
        output_slice[i * 2] = sample; // Left channel
        output_slice[i * 2 + 1] = sample; // Right channel
    }

    VoirsErrorCode::Success
}

/// Convert stereo audio to mono by averaging channels
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `sample_count * 2` floats (interleaved stereo).
/// The `output` pointer must be valid and point to a writable buffer of at least `sample_count` floats.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_stereo_to_mono(
    input: *const c_float,
    output: *mut c_float,
    sample_count: c_uint,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || sample_count == 0 || !sample_count.is_multiple_of(2) {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, sample_count as usize);
    let output_slice = std::slice::from_raw_parts_mut(output, (sample_count / 2) as usize);

    for i in 0..(sample_count as usize / 2) {
        let left = input_slice[i * 2];
        let right = input_slice[i * 2 + 1];
        output_slice[i] = (left + right) * 0.5;
    }

    VoirsErrorCode::Success
}

/// Simple nearest-neighbor sample rate conversion
///
/// # Safety
/// The `input` pointer must be valid and point to an array of at least `input_samples` floats.
/// The `output` pointer must be valid and point to a writable buffer large enough for the resampled audio.
/// The `output_samples` pointer must be valid and point to a writable c_uint where the output sample count will be stored.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_sample_rate(
    input: *const c_float,
    output: *mut c_float,
    input_samples: c_uint,
    input_rate: c_uint,
    output_rate: c_uint,
    channels: c_uint,
    output_samples: *mut c_uint,
) -> VoirsErrorCode {
    if input.is_null()
        || output.is_null()
        || output_samples.is_null()
        || input_samples == 0
        || input_rate == 0
        || output_rate == 0
        || channels == 0
    {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, input_samples as usize);
    let ratio = output_rate as f64 / input_rate as f64;
    let output_length =
        ((input_samples as f64 / channels as f64) * ratio) as usize * channels as usize;

    if output_length == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let output_slice = std::slice::from_raw_parts_mut(output, output_length);
    *output_samples = output_length as c_uint;

    // Note: Cannot use iterator here due to complex index calculations for channel interleaving
    #[allow(clippy::needless_range_loop)]
    for i in 0..output_length {
        let frame_index = i / channels as usize;
        let channel_index = i % channels as usize;

        // Calculate input frame index using nearest neighbor
        let input_frame = (frame_index as f64 / ratio) as usize;
        let input_index = input_frame * channels as usize + channel_index;

        if input_index < input_samples as usize {
            output_slice[i] = input_slice[input_index];
        } else {
            output_slice[i] = 0.0; // Zero-pad if beyond input
        }
    }

    VoirsErrorCode::Success
}

/// Allocate an all-zero `*mut c_float` buffer of `len` samples using the same
/// `Vec<f32> -> into_boxed_slice() -> forget` allocation contract as
/// `VoirsAudioBuffer::from_audio_buffer`/`free()`, so a pointer produced here
/// can later be stored into `VoirsAudioBuffer.samples` and freed by
/// `VoirsAudioBuffer::free()` without a layout mismatch. Unlike a raw
/// `std::alloc::alloc` + `Layout::from_size_align(..).expect(..)`, this can
/// never require an `.expect()` in a production code path: `Vec`'s allocator
/// aborts the process on OOM rather than needing an unwrap-style call here.
fn alloc_f32_buffer(len: usize) -> *mut c_float {
    let mut boxed = vec![0.0f32; len].into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    ptr
}

/// Free a buffer previously returned by [`alloc_f32_buffer`] that was never
/// installed into a `VoirsAudioBuffer` (e.g. because a conversion step that
/// would have used it failed). `len` must be the same length passed to
/// `alloc_f32_buffer`.
///
/// # Safety
/// `ptr` must have come from `alloc_f32_buffer(len)` and not have been freed
/// already.
unsafe fn free_f32_buffer(ptr: *mut c_float, len: usize) {
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)));
}

/// Convert audio buffer format in-place
///
/// Only [`VoirsSampleFormat::Float32`] is supported: [`VoirsAudioBuffer`] is a
/// `#[repr(C)]` struct whose `samples` field is always `*mut c_float` -- every
/// other function in this crate (`from_audio_buffer`, `to_audio_buffer`,
/// `free`, the whole `voirs_convert_float_to_*` family, ...) relies on that.
/// Repurposing the same storage to hold `Int16`/`Int24`/`UInt8`/... bit
/// patterns would silently corrupt every other consumer of the buffer, so a
/// `target_format` other than `Float32` is honestly rejected instead of
/// being silently ignored. To convert into another sample format, convert the
/// samples into a *separate* buffer with the appropriate `voirs_convert_float_to_*`
/// function instead.
///
/// # Safety
/// The `buffer` pointer must be valid and point to a properly initialized VoirsAudioBuffer.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_convert_format(
    buffer: *mut VoirsAudioBuffer,
    target_format: VoirsSampleFormat,
    target_channels: c_uint,
    target_sample_rate: c_uint,
) -> VoirsErrorCode {
    if buffer.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    // Checked first, before any buffer mutation: a rejected call must leave
    // the buffer completely untouched rather than partially converting the
    // sample rate/channels and only then discovering the format is
    // unsupported.
    if target_format != VoirsSampleFormat::Float32 {
        set_last_error(format!(
            "voirs_audio_convert_format only supports VoirsSampleFormat::Float32 \
             (VoirsAudioBuffer.samples is a fixed *mut f32 array); requested {target_format:?}. \
             Convert samples into a separate buffer with the voirs_convert_float_to_* \
             family instead."
        ));
        return VoirsErrorCode::InvalidParameter;
    }

    let audio_buffer = &mut *buffer;

    // Convert sample rate if needed
    if target_sample_rate != audio_buffer.sample_rate {
        let mut new_length = 0;

        // Allocate temporary buffer for conversion
        let max_output_samples = ((audio_buffer.length as f64 / audio_buffer.channels as f64)
            * (target_sample_rate as f64 / audio_buffer.sample_rate as f64))
            as usize
            * audio_buffer.channels as usize;

        let temp_buffer = alloc_f32_buffer(max_output_samples);

        let result = voirs_convert_sample_rate(
            audio_buffer.samples,
            temp_buffer,
            audio_buffer.length,
            audio_buffer.sample_rate,
            target_sample_rate,
            audio_buffer.channels,
            &mut new_length,
        );

        if result != VoirsErrorCode::Success {
            free_f32_buffer(temp_buffer, max_output_samples);
            return result;
        }

        // Replace old buffer
        audio_buffer.free();
        audio_buffer.samples = temp_buffer;
        audio_buffer.length = new_length;
        audio_buffer.sample_rate = target_sample_rate;
    }

    // Convert channels if needed
    if target_channels != audio_buffer.channels {
        if audio_buffer.channels == 1 && target_channels == 2 {
            // Mono to stereo
            let new_length = audio_buffer.length * 2;
            let temp_buffer = alloc_f32_buffer(new_length as usize);

            let result = voirs_convert_mono_to_stereo(
                audio_buffer.samples,
                temp_buffer,
                audio_buffer.length,
            );

            if result != VoirsErrorCode::Success {
                free_f32_buffer(temp_buffer, new_length as usize);
                return result;
            }

            audio_buffer.free();
            audio_buffer.samples = temp_buffer;
            audio_buffer.length = new_length;
            audio_buffer.channels = 2;
        } else if audio_buffer.channels == 2 && target_channels == 1 {
            // Stereo to mono
            let new_length = audio_buffer.length / 2;
            let temp_buffer = alloc_f32_buffer(new_length as usize);

            let result = voirs_convert_stereo_to_mono(
                audio_buffer.samples,
                temp_buffer,
                audio_buffer.length,
            );

            if result != VoirsErrorCode::Success {
                free_f32_buffer(temp_buffer, new_length as usize);
                return result;
            }

            audio_buffer.free();
            audio_buffer.samples = temp_buffer;
            audio_buffer.length = new_length;
            audio_buffer.channels = 1;
        } else {
            // Other channel conversions not yet supported
            return VoirsErrorCode::InternalError;
        }
    }

    VoirsErrorCode::Success
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_float_to_int16_conversion() {
        let input = [0.0, 0.5, -0.5, 1.0, -1.0];
        let mut output = [0i16; 5];

        unsafe {
            let result = voirs_convert_float_to_int16(
                input.as_ptr(),
                output.as_mut_ptr(),
                5,
                VoirsEndianness::Native,
            );
            assert_eq!(result, VoirsErrorCode::Success);
            assert_eq!(output[0], 0);
            assert_eq!(output[1], 16384); // 0.5 * 32767 ≈ 16384
            assert_eq!(output[2], -16384);
            assert_eq!(output[3], 32767);
            assert_eq!(output[4], -32767);
        }
    }

    #[test]
    fn test_int16_to_float_conversion() {
        let input = [0i16, 16384, -16384, 32767, -32767];
        let mut output = [0.0f32; 5];

        unsafe {
            let result = voirs_convert_int16_to_float(
                input.as_ptr(),
                output.as_mut_ptr(),
                5,
                VoirsEndianness::Native,
            );
            assert_eq!(result, VoirsErrorCode::Success);
            assert!((output[0] - 0.0).abs() < 0.001);
            assert!((output[1] - 0.5).abs() < 0.001);
            assert!((output[2] - (-0.5)).abs() < 0.001);
            assert!((output[3] - 1.0).abs() < 0.001);
            assert!((output[4] - (-1.0)).abs() < 0.001);
        }
    }

    #[test]
    fn test_mono_to_stereo_conversion() {
        let input = [0.1, 0.2, 0.3];
        let mut output = [0.0f32; 6];

        unsafe {
            let result = voirs_convert_mono_to_stereo(input.as_ptr(), output.as_mut_ptr(), 3);
            assert_eq!(result, VoirsErrorCode::Success);
            assert_eq!(output[0], 0.1); // Left
            assert_eq!(output[1], 0.1); // Right
            assert_eq!(output[2], 0.2);
            assert_eq!(output[3], 0.2);
            assert_eq!(output[4], 0.3);
            assert_eq!(output[5], 0.3);
        }
    }

    #[test]
    fn test_stereo_to_mono_conversion() {
        let input = [0.1, 0.3, 0.2, 0.4]; // L1, R1, L2, R2
        let mut output = [0.0f32; 2];

        unsafe {
            let result = voirs_convert_stereo_to_mono(input.as_ptr(), output.as_mut_ptr(), 4);
            assert_eq!(result, VoirsErrorCode::Success);
            assert!((output[0] - 0.2).abs() < 0.001); // (0.1 + 0.3) / 2
            assert!((output[1] - 0.3).abs() < 0.001); // (0.2 + 0.4) / 2
        }
    }

    #[test]
    fn test_sample_rate_conversion() {
        let input = [1.0, 2.0, 3.0, 4.0]; // 2 stereo samples (1 frame) at 22050 Hz
        let mut output = [0.0f32; 8]; // Should get ~4 stereo samples (2 frames) at 44100 Hz
        let mut output_samples = 0;

        unsafe {
            let result = voirs_convert_sample_rate(
                input.as_ptr(),
                output.as_mut_ptr(),
                4,
                22050,
                44100,
                2,
                &mut output_samples,
            );
            assert_eq!(result, VoirsErrorCode::Success);
            // Doubling sample rate: 2 frames -> 4 frames, so 8 samples (4 frames * 2 channels)
            assert_eq!(output_samples, 8);
        }
    }

    #[test]
    fn test_invalid_parameters() {
        unsafe {
            // Test null pointer
            let result = voirs_convert_float_to_int16(
                ptr::null(),
                ptr::null_mut(),
                5,
                VoirsEndianness::Native,
            );
            assert_eq!(result, VoirsErrorCode::InvalidParameter);

            // Test zero sample count
            let input = [0.0f32; 1];
            let mut output = [0i16; 1];
            let result = voirs_convert_float_to_int16(
                input.as_ptr(),
                output.as_mut_ptr(),
                0,
                VoirsEndianness::Native,
            );
            assert_eq!(result, VoirsErrorCode::InvalidParameter);
        }
    }

    #[test]
    fn test_endianness_conversion() {
        let input = [0.5f32];
        let mut output_le = [0i16; 1];
        let mut output_be = [0i16; 1];

        unsafe {
            // Test little endian
            let result = voirs_convert_float_to_int16(
                input.as_ptr(),
                output_le.as_mut_ptr(),
                1,
                VoirsEndianness::LittleEndian,
            );
            assert_eq!(result, VoirsErrorCode::Success);

            // Test big endian
            let result = voirs_convert_float_to_int16(
                input.as_ptr(),
                output_be.as_mut_ptr(),
                1,
                VoirsEndianness::BigEndian,
            );
            assert_eq!(result, VoirsErrorCode::Success);

            // They should be different on most platforms
            // unless we're on a big-endian platform where native == big endian
        }
    }

    #[test]
    fn test_float_to_double_conversion() {
        let input = [0.5f32, -0.5f32, 1.0f32, -1.0f32];
        let mut output = [0.0f64; 4];

        unsafe {
            let result = voirs_convert_float_to_double(input.as_ptr(), output.as_mut_ptr(), 4);
            assert_eq!(result, VoirsErrorCode::Success);
            assert!((output[0] - 0.5).abs() < 1e-6);
            assert!((output[1] - (-0.5)).abs() < 1e-6);
            assert!((output[2] - 1.0).abs() < 1e-6);
            assert!((output[3] - (-1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_double_to_float_conversion() {
        let input = [0.5f64, -0.5f64, 1.0f64, -1.0f64];
        let mut output = [0.0f32; 4];

        unsafe {
            let result = voirs_convert_double_to_float(input.as_ptr(), output.as_mut_ptr(), 4);
            assert_eq!(result, VoirsErrorCode::Success);
            assert!((output[0] - 0.5).abs() < 1e-6);
            assert!((output[1] - (-0.5)).abs() < 1e-6);
            assert!((output[2] - 1.0).abs() < 1e-6);
            assert!((output[3] - (-1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_float_to_int24_conversion() {
        let input = [0.0, 0.5, -0.5, 1.0, -1.0];
        let mut output = [0i32; 5];

        unsafe {
            let result = voirs_convert_float_to_int24(
                input.as_ptr(),
                output.as_mut_ptr(),
                5,
                VoirsEndianness::Native,
            );
            assert_eq!(result, VoirsErrorCode::Success);
            assert_eq!(output[0], 0);
            assert_eq!(output[1], 4194304); // 0.5 * 8388607 ≈ 4194304
            assert_eq!(output[2], -4194304);
            assert_eq!(output[3], 8388607);
            assert_eq!(output[4], -8388608);
        }
    }

    #[test]
    fn test_int24_to_float_conversion() {
        let input = [0i32, 4194304, -4194304, 8388607, -8388608];
        let mut output = [0.0f32; 5];

        unsafe {
            let result = voirs_convert_int24_to_float(
                input.as_ptr(),
                output.as_mut_ptr(),
                5,
                VoirsEndianness::Native,
            );
            assert_eq!(result, VoirsErrorCode::Success);
            assert!((output[0] - 0.0).abs() < 0.001);
            assert!((output[1] - 0.5).abs() < 0.001);
            assert!((output[2] - (-0.5)).abs() < 0.001);
            assert!((output[3] - 1.0).abs() < 0.001);
            assert!((output[4] - (-1.0)).abs() < 0.001);
        }
    }

    #[test]
    fn test_float_to_uint8_conversion() {
        let input = [0.0, 0.5, -0.5, 1.0, -1.0];
        let mut output = [0u8; 5];

        unsafe {
            let result = voirs_convert_float_to_uint8(input.as_ptr(), output.as_mut_ptr(), 5);
            assert_eq!(result, VoirsErrorCode::Success);
            assert_eq!(output[0], 128); // 0.0 -> 128 (middle of 0-255)
            assert_eq!(output[1], 191); // 0.5 -> 191 (actual calculated value)
            assert_eq!(output[2], 64); // -0.5 -> 64
            assert_eq!(output[3], 255); // 1.0 -> 255
            assert_eq!(output[4], 0); // -1.0 -> 0
        }
    }

    #[test]
    fn test_uint8_to_float_conversion() {
        let input = [128u8, 191u8, 64u8, 255u8, 0u8];
        let mut output = [0.0f32; 5];

        unsafe {
            let result = voirs_convert_uint8_to_float(input.as_ptr(), output.as_mut_ptr(), 5);
            assert_eq!(result, VoirsErrorCode::Success);
            assert!((output[0] - 0.0).abs() < 0.01);
            assert!((output[1] - 0.5).abs() < 0.01);
            assert!((output[2] - (-0.5)).abs() < 0.01);
            assert!((output[3] - 1.0).abs() < 0.01);
            assert!((output[4] - (-1.0)).abs() < 0.01);
        }
    }

    #[test]
    fn test_float_to_uint16_conversion() {
        let input = [0.0, 0.5, -0.5, 1.0, -1.0];
        let mut output = [0u16; 5];

        unsafe {
            let result = voirs_convert_float_to_uint16(
                input.as_ptr(),
                output.as_mut_ptr(),
                5,
                VoirsEndianness::Native,
            );
            assert_eq!(result, VoirsErrorCode::Success);
            assert_eq!(output[0], 32768); // 0.0 -> 32768 (middle)
            assert_eq!(output[1], 49151); // 0.5 -> 49151 (actual calculated value)
            assert_eq!(output[2], 16384); // -0.5 -> 16384
            assert_eq!(output[3], 65535); // 1.0 -> 65535
            assert_eq!(output[4], 0); // -1.0 -> 0
        }
    }

    #[test]
    fn test_uint16_to_float_conversion() {
        let input = [32768u16, 49151u16, 16384u16, 65535u16, 0u16];
        let mut output = [0.0f32; 5];

        unsafe {
            let result = voirs_convert_uint16_to_float(
                input.as_ptr(),
                output.as_mut_ptr(),
                5,
                VoirsEndianness::Native,
            );
            assert_eq!(result, VoirsErrorCode::Success);
            assert!((output[0] - 0.0).abs() < 0.001);
            assert!((output[1] - 0.5).abs() < 0.001);
            assert!((output[2] - (-0.5)).abs() < 0.001);
            assert!((output[3] - 1.0).abs() < 0.001);
            assert!((output[4] - (-1.0)).abs() < 0.001);
        }
    }

    /// Build a `VoirsAudioBuffer` using the exact same allocation contract as
    /// `VoirsAudioBuffer::from_audio_buffer` (`Vec -> into_boxed_slice ->
    /// forget`), so it can be safely passed to `voirs_audio_convert_format`
    /// and later freed via `.free()`.
    fn make_test_buffer(samples: &[f32], sample_rate: u32, channels: u32) -> VoirsAudioBuffer {
        VoirsAudioBuffer::from_audio_buffer(voirs_sdk::audio::AudioBuffer::new(
            samples.to_vec(),
            sample_rate,
            channels,
        ))
    }

    #[test]
    fn test_convert_format_rejects_non_float32_and_leaves_buffer_unchanged() {
        let mut buffer = make_test_buffer(&[0.1, 0.2, 0.3, 0.4], 44100, 1);
        let original_ptr = buffer.samples;
        let original_length = buffer.length;
        let original_sample_rate = buffer.sample_rate;
        let original_channels = buffer.channels;

        let non_float32_formats = [
            VoirsSampleFormat::Int16,
            VoirsSampleFormat::Int24,
            VoirsSampleFormat::Int32,
            VoirsSampleFormat::Float64,
            VoirsSampleFormat::UInt8,
            VoirsSampleFormat::UInt16,
            VoirsSampleFormat::UInt32,
        ];

        for format in non_float32_formats {
            let result = unsafe { voirs_audio_convert_format(&mut buffer, format, 1, 44100) };
            assert_eq!(
                result,
                VoirsErrorCode::InvalidParameter,
                "non-Float32 target {format:?} must be honestly rejected, not silently ignored"
            );

            // A rejected call must be a true no-op: same allocation, same
            // metadata -- not a partially-applied sample-rate/channel
            // conversion that then discovers the format is unsupported.
            assert_eq!(buffer.samples, original_ptr);
            assert_eq!(buffer.length, original_length);
            assert_eq!(buffer.sample_rate, original_sample_rate);
            assert_eq!(buffer.channels, original_channels);
        }

        unsafe {
            buffer.free();
        }
    }

    #[test]
    fn test_convert_format_float32_actually_resamples() {
        // 4 samples at 22050 Hz mono.
        let mut buffer = make_test_buffer(&[1.0, 2.0, 3.0, 4.0], 22050, 1);

        let result =
            unsafe { voirs_audio_convert_format(&mut buffer, VoirsSampleFormat::Float32, 1, 44100) };
        assert_eq!(result, VoirsErrorCode::Success);

        // Real resampling must actually change sample_rate and length --
        // proving the Float32 path still performs the real conversion (only
        // non-Float32 targets are rejected).
        assert_eq!(buffer.sample_rate, 44100);
        assert_eq!(buffer.length, 8, "doubling the rate should double the sample count");

        unsafe {
            buffer.free();
        }
    }

    #[test]
    fn test_convert_format_float32_actually_rechannels() {
        let mut buffer = make_test_buffer(&[0.5, -0.5, 1.0], 44100, 1);

        let result =
            unsafe { voirs_audio_convert_format(&mut buffer, VoirsSampleFormat::Float32, 2, 44100) };
        assert_eq!(result, VoirsErrorCode::Success);

        assert_eq!(buffer.channels, 2);
        assert_eq!(buffer.length, 6);

        unsafe {
            let samples = std::slice::from_raw_parts(buffer.samples, buffer.length as usize);
            // Mono-to-stereo duplicates each sample across both channels.
            assert_eq!(samples, &[0.5, 0.5, -0.5, -0.5, 1.0, 1.0]);
            buffer.free();
        }
    }
}
