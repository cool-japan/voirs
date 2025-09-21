//! Recognition functions for the C API.

use super::core::VoirsRecognizerInternal;
use super::memory::c_string_to_string;
use super::types::{
    VoirsAudioFormat, VoirsAudioFormatType, VoirsError, VoirsRecognitionResult, VoirsRecognizer,
    VoirsSegment,
};
use crate::{AudioBuffer, LanguageCode};
use std::ffi::{c_char, c_void};
use std::slice;
use std::time::Instant;

/// Recognize speech from audio data
///
/// # Arguments
/// * `recognizer` - Pointer to the recognizer instance
/// * `audio_data` - Pointer to audio data buffer
/// * `audio_size` - Size of audio data in bytes
/// * `audio_format` - Audio format information. If null, assumes 16-bit PCM at 16kHz mono
/// * `result` - Output pointer to store the recognition result
///
/// # Returns
/// VoirsError::Success on success, or an error code on failure.
#[no_mangle]
pub extern "C" fn voirs_recognize(
    recognizer: *mut VoirsRecognizer,
    audio_data: *const u8,
    audio_size: usize,
    audio_format: *const VoirsAudioFormat,
    result: *mut *const VoirsRecognitionResult,
) -> VoirsError {
    if recognizer.is_null() || audio_data.is_null() || result.is_null() {
        return VoirsError::NullPointer;
    }

    if audio_size == 0 {
        return VoirsError::InvalidArgument;
    }

    let catch_result = std::panic::catch_unwind(|| {
        let internal = unsafe { &mut *(recognizer as *mut VoirsRecognizerInternal) };

        // Get audio format (use default if not provided)
        let format = if audio_format.is_null() {
            VoirsAudioFormat::default()
        } else {
            unsafe { *audio_format }
        };

        // Convert audio data to slice
        let audio_bytes = unsafe { slice::from_raw_parts(audio_data, audio_size) };

        // Perform recognition
        let start_time = Instant::now();

        let recognition_result = internal.runtime.block_on(async {
            // Validate audio format
            if !is_supported_format(&format) {
                return Err("Unsupported audio format".to_string());
            }

            // Process audio based on format
            let processed_audio = match format.format {
                VoirsAudioFormatType::PCM16 => {
                    // Audio is already in the right format
                    audio_bytes.to_vec()
                }
                VoirsAudioFormatType::PCM32 => {
                    // Convert 32-bit to 16-bit
                    convert_pcm32_to_pcm16(audio_bytes)
                }
                VoirsAudioFormatType::Float32 => {
                    // Convert float32 to 16-bit PCM
                    convert_float32_to_pcm16(audio_bytes)
                }
                _ => {
                    // For compressed formats, we would need proper decoding
                    // For now, assume they're already decoded
                    audio_bytes.to_vec()
                }
            };

            // Recognize the audio
            internal
                .pipeline
                .recognize_bytes(&processed_audio)
                .await
                .map_err(|e| e.to_string())
        });

        let processing_time = start_time.elapsed();

        match recognition_result {
            Ok(rec_result) => {
                // Convert segments
                let segments: Vec<VoirsSegment> = rec_result
                    .transcription
                    .as_ref()
                    .map(|t| {
                        t.word_timestamps
                            .iter()
                            .map(|word| VoirsSegment {
                                start_time: word.start_time as f64,
                                end_time: word.end_time as f64,
                                text: internal.memory_manager.store_string(&word.word),
                                confidence: word.confidence,
                                no_speech_prob: 1.0 - word.confidence,
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                // Create result structure
                let c_result = VoirsRecognitionResult {
                    text: internal.memory_manager.store_string(
                        &rec_result
                            .transcription
                            .as_ref()
                            .map(|t| &t.text)
                            .unwrap_or(&String::new()),
                    ),
                    confidence: rec_result
                        .transcription
                        .as_ref()
                        .map(|t| t.confidence)
                        .unwrap_or(0.0),
                    language: rec_result
                        .transcription
                        .as_ref()
                        .map(|t| {
                            internal
                                .memory_manager
                                .store_string(&t.language.to_string())
                        })
                        .unwrap_or(std::ptr::null()),
                    processing_time_ms: processing_time.as_secs_f64() * 1000.0,
                    audio_duration_s: rec_result
                        .transcription
                        .as_ref()
                        .and_then(|t| t.processing_duration)
                        .unwrap_or_default()
                        .as_secs_f64(),
                    segment_count: segments.len(),
                    segments: if segments.is_empty() {
                        std::ptr::null()
                    } else {
                        internal.memory_manager.store_segments(&segments)
                    },
                };

                unsafe {
                    *result = internal.memory_manager.store_result(c_result);
                }

                VoirsError::Success
            }
            Err(_) => VoirsError::RecognitionFailed,
        }
    });

    match catch_result {
        Ok(error) => error,
        Err(_) => VoirsError::InternalError,
    }
}

/// Recognize speech from a file
///
/// # Arguments
/// * `recognizer` - Pointer to the recognizer instance
/// * `file_path` - Path to the audio file (null-terminated C string)
/// * `result` - Output pointer to store the recognition result
///
/// # Returns
/// VoirsError::Success on success, or an error code on failure.
#[no_mangle]
pub extern "C" fn voirs_recognize_file(
    recognizer: *mut VoirsRecognizer,
    file_path: *const c_char,
    result: *mut *const VoirsRecognitionResult,
) -> VoirsError {
    if recognizer.is_null() || file_path.is_null() || result.is_null() {
        return VoirsError::NullPointer;
    }

    let catch_result = std::panic::catch_unwind(|| {
        let internal = unsafe { &mut *(recognizer as *mut VoirsRecognizerInternal) };

        let path_str = match c_string_to_string(file_path) {
            Ok(s) => s,
            Err(_) => return VoirsError::InvalidArgument,
        };

        // Read audio file
        let audio_data = match std::fs::read(&path_str) {
            Ok(data) => data,
            Err(_) => return VoirsError::InvalidArgument,
        };

        // Determine format from file extension
        let format = match get_format_from_extension(&path_str) {
            Some(f) => f,
            None => return VoirsError::UnsupportedFormat,
        };

        // Call the main recognition function
        voirs_recognize(
            recognizer,
            audio_data.as_ptr(),
            audio_data.len(),
            &format,
            result,
        )
    });

    match catch_result {
        Ok(error) => error,
        Err(_) => VoirsError::InternalError,
    }
}

/// Free memory allocated for recognition results
///
/// # Arguments
/// * `recognizer` - Pointer to the recognizer instance
/// * `result` - Pointer to the result to free
///
/// # Returns
/// VoirsError::Success on success, or an error code on failure.
#[no_mangle]
pub extern "C" fn voirs_free_result(
    recognizer: *mut VoirsRecognizer,
    result: *const VoirsRecognitionResult,
) -> VoirsError {
    if recognizer.is_null() || result.is_null() {
        return VoirsError::NullPointer;
    }

    let catch_result = std::panic::catch_unwind(|| {
        let internal = unsafe { &mut *(recognizer as *mut VoirsRecognizerInternal) };

        // The memory manager will handle cleanup automatically
        // when the recognizer is destroyed, but we can mark it for cleanup
        internal
            .memory_manager
            .mark_for_cleanup(result as *const _ as *const c_void);

        VoirsError::Success
    });

    match catch_result {
        Ok(error) => error,
        Err(_) => VoirsError::InternalError,
    }
}

// Helper functions

fn is_supported_format(format: &VoirsAudioFormat) -> bool {
    matches!(
        format.format,
        VoirsAudioFormatType::PCM16
            | VoirsAudioFormatType::PCM32
            | VoirsAudioFormatType::Float32
            | VoirsAudioFormatType::WAV
            | VoirsAudioFormatType::MP3
            | VoirsAudioFormatType::FLAC
            | VoirsAudioFormatType::OGG
            | VoirsAudioFormatType::M4A
    )
}

fn convert_pcm32_to_pcm16(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len() / 2);

    for chunk in data.chunks_exact(4) {
        let sample = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let sample_16 = (sample >> 16) as i16;
        result.extend_from_slice(&sample_16.to_le_bytes());
    }

    result
}

fn convert_float32_to_pcm16(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len() / 2);

    for chunk in data.chunks_exact(4) {
        let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let sample_16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        result.extend_from_slice(&sample_16.to_le_bytes());
    }

    result
}

fn get_format_from_extension(path: &str) -> Option<VoirsAudioFormat> {
    let extension = std::path::Path::new(path)
        .extension()?
        .to_str()?
        .to_lowercase();

    let format_type = match extension.as_str() {
        "wav" => VoirsAudioFormatType::WAV,
        "mp3" => VoirsAudioFormatType::MP3,
        "flac" => VoirsAudioFormatType::FLAC,
        "ogg" => VoirsAudioFormatType::OGG,
        "m4a" | "aac" => VoirsAudioFormatType::M4A,
        _ => return None,
    };

    Some(VoirsAudioFormat {
        format: format_type,
        ..VoirsAudioFormat::default()
    })
}
