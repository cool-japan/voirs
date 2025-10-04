//! Streaming recognition functions for the C API.

use super::core::VoirsRecognizerInternal;
use super::memory::c_string_to_string;
use super::types::*;
use std::collections::VecDeque;
use std::ffi::{c_char, c_void};
use std::slice;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Streaming context structure
pub struct StreamingContext {
    /// is active
    pub is_active: bool,
    /// config
    pub config: VoirsStreamingConfig,
    /// audio buffer
    pub audio_buffer: VecDeque<u8>,
    /// callback
    pub callback: Option<VoirsStreamingCallback>,
    /// user data
    pub user_data: *mut c_void,
    /// chunk count
    pub chunk_count: usize,
    /// total audio duration
    pub total_audio_duration: f64,
    /// average latency
    pub average_latency: f64,
    /// latency measurements
    pub latency_measurements: VecDeque<f64>,
}

impl StreamingContext {
    fn new() -> Self {
        Self {
            is_active: false,
            config: VoirsStreamingConfig::default(),
            audio_buffer: VecDeque::new(),
            callback: None,
            user_data: std::ptr::null_mut(),
            chunk_count: 0,
            total_audio_duration: 0.0,
            average_latency: 0.0,
            latency_measurements: VecDeque::new(),
        }
    }
}

/// Start streaming recognition mode
///
/// # Arguments
/// * `recognizer` - Pointer to the recognizer instance
/// * `config` - Streaming configuration. If null, default configuration will be used.
/// * `callback` - Callback function to receive streaming results
/// * `user_data` - User data pointer passed to the callback
///
/// # Returns
/// VoirsError::Success on success, or an error code on failure.
#[no_mangle]
pub extern "C" fn voirs_start_streaming(
    recognizer: *mut VoirsRecognizer,
    config: *const VoirsStreamingConfig,
    callback: VoirsStreamingCallback,
    user_data: *mut c_void,
) -> VoirsError {
    if recognizer.is_null() {
        return VoirsError::NullPointer;
    }

    let catch_result = std::panic::catch_unwind(|| {
        let internal = unsafe { &mut *(recognizer as *mut VoirsRecognizerInternal) };

        let streaming_config = if config.is_null() {
            VoirsStreamingConfig::default()
        } else {
            unsafe { (*config).clone() }
        };

        // Create streaming context and store it in the internal recognizer
        let mut context = StreamingContext::new();
        context.config = streaming_config;
        context.callback = Some(callback);
        context.user_data = user_data;
        context.is_active = true;

        internal.streaming_context = Some(Arc::new(Mutex::new(context)));

        VoirsError::Success
    });

    match catch_result {
        Ok(error) => error,
        Err(_) => VoirsError::InternalError,
    }
}

/// Stop streaming recognition mode
///
/// # Arguments
/// * `recognizer` - Pointer to the recognizer instance
///
/// # Returns
/// VoirsError::Success on success, or an error code on failure.
#[no_mangle]
/// Item
pub extern "C" fn voirs_stop_streaming(recognizer: *mut VoirsRecognizer) -> VoirsError {
    if recognizer.is_null() {
        return VoirsError::NullPointer;
    }

    let catch_result = std::panic::catch_unwind(|| {
        let internal = unsafe { &mut *(recognizer as *mut VoirsRecognizerInternal) };

        // Clear streaming context and stop processing
        if let Some(context_arc) = &internal.streaming_context {
            if let Ok(mut context) = context_arc.lock() {
                context.is_active = false;
                context.audio_buffer.clear();
                context.callback = None;
            }
        }
        internal.streaming_context = None;

        VoirsError::Success
    });

    match catch_result {
        Ok(error) => error,
        Err(_) => VoirsError::InternalError,
    }
}

/// Process an audio chunk in streaming mode
///
/// # Arguments
/// * `recognizer` - Pointer to the recognizer instance
/// * `audio_data` - Pointer to audio data buffer
/// * `audio_size` - Size of audio data in bytes
///
/// # Returns
/// VoirsError::Success on success, or an error code on failure.
/// Results are delivered via the callback function registered with voirs_start_streaming.
#[no_mangle]
pub extern "C" fn voirs_stream_audio(
    recognizer: *mut VoirsRecognizer,
    audio_data: *const u8,
    audio_size: usize,
) -> VoirsError {
    if recognizer.is_null() || audio_data.is_null() {
        return VoirsError::NullPointer;
    }

    if audio_size == 0 {
        return VoirsError::InvalidArgument;
    }

    let catch_result = std::panic::catch_unwind(|| {
        let internal = unsafe { &mut *(recognizer as *mut VoirsRecognizerInternal) };

        // Check if streaming is active
        let streaming_context = match &internal.streaming_context {
            Some(context) => context.clone(),
            None => return VoirsError::StreamingNotStarted,
        };

        // Convert audio data to slice
        let audio_bytes = unsafe { slice::from_raw_parts(audio_data, audio_size) };
        let start_time = Instant::now();

        // Buffer the audio data
        {
            let mut context = streaming_context.lock().unwrap();
            if !context.is_active {
                return VoirsError::StreamingNotStarted;
            }

            context.audio_buffer.extend(audio_bytes.iter());
        }

        // Process audio chunks when buffer is large enough
        let chunk_size = {
            let context = streaming_context.lock().unwrap();
            (context.config.chunk_duration * internal.config.sample_rate as f32 * 2.0) as usize
            // 2 bytes per sample for 16-bit
        };

        let callback_data = {
            let mut context = streaming_context.lock().unwrap();
            if context.audio_buffer.len() >= chunk_size {
                // Extract chunk for processing
                let chunk: Vec<u8> = context.audio_buffer.drain(..chunk_size).collect();
                context.chunk_count += 1;

                Some((chunk, context.callback, context.user_data))
            } else {
                None
            }
        };

        if let Some((chunk, callback_opt, user_data)) = callback_data {
            // Perform actual streaming recognition
            let recognition_result = internal
                .runtime
                .block_on(async { internal.pipeline.recognize_bytes(&chunk).await });

            let processing_time = start_time.elapsed().as_millis() as f64;

            // Update metrics
            internal.metrics.processed_chunks += 1;
            internal.metrics.total_processing_time_ms += processing_time;
            internal.metrics.total_audio_duration_s +=
                chunk.len() as f64 / (internal.config.sample_rate as f64 * 2.0);
            if processing_time > internal.metrics.peak_processing_time_ms {
                internal.metrics.peak_processing_time_ms = processing_time;
            }

            match recognition_result {
                Ok(result) => {
                    // Convert to C API result format
                    let segments: Vec<VoirsSegment> = result
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

                    let c_result = VoirsRecognitionResult {
                        text: internal.memory_manager.store_string(
                            &result
                                .transcription
                                .as_ref()
                                .map(|t| &t.text)
                                .unwrap_or(&String::new()),
                        ),
                        confidence: result
                            .transcription
                            .as_ref()
                            .map(|t| t.confidence)
                            .unwrap_or(0.0),
                        language: result
                            .transcription
                            .as_ref()
                            .map(|t| {
                                internal
                                    .memory_manager
                                    .store_string(&t.language.to_string())
                            })
                            .unwrap_or(std::ptr::null()),
                        processing_time_ms: processing_time,
                        audio_duration_s: result
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

                    // Call the registered callback with the result
                    if let Some(callback) = callback_opt {
                        callback(&c_result, user_data);
                    }

                    // Update latency measurements
                    {
                        let mut context = streaming_context.lock().unwrap();
                        context.latency_measurements.push_back(processing_time);
                        if context.latency_measurements.len() > 100 {
                            context.latency_measurements.pop_front();
                        }
                        context.average_latency = context.latency_measurements.iter().sum::<f64>()
                            / context.latency_measurements.len() as f64;
                        context.total_audio_duration += result
                            .transcription
                            .as_ref()
                            .and_then(|t| t.processing_duration)
                            .unwrap_or_default()
                            .as_secs_f64();
                    }

                    VoirsError::Success
                }
                Err(_) => {
                    internal.metrics.failed_recognitions += 1;
                    VoirsError::RecognitionFailed
                }
            }
        } else {
            // Buffer is not full yet, just accumulate data
            VoirsError::Success
        }
    });

    match catch_result {
        Ok(error) => error,
        Err(_) => VoirsError::InternalError,
    }
}

/// Check if streaming mode is active
///
/// # Arguments
/// * `recognizer` - Pointer to the recognizer instance
///
/// # Returns
/// true if streaming is active, false otherwise
#[no_mangle]
/// Item
pub extern "C" fn voirs_is_streaming_active(recognizer: *mut VoirsRecognizer) -> bool {
    if recognizer.is_null() {
        return false;
    }

    let catch_result = std::panic::catch_unwind(|| {
        let internal = unsafe { &*(recognizer as *const VoirsRecognizerInternal) };

        // Return actual streaming status
        if let Some(context_arc) = &internal.streaming_context {
            if let Ok(context) = context_arc.lock() {
                return context.is_active;
            }
        }
        false
    });

    match catch_result {
        Ok(is_active) => is_active,
        Err(_) => false,
    }
}

/// Get streaming buffer information
///
/// # Arguments
/// * `recognizer` - Pointer to the recognizer instance
/// * `buffer_size` - Output pointer for current buffer size in bytes
/// * `buffer_duration` - Output pointer for buffer duration in seconds
///
/// # Returns
/// VoirsError::Success on success, or an error code on failure.
#[no_mangle]
pub extern "C" fn voirs_get_streaming_buffer_info(
    recognizer: *mut VoirsRecognizer,
    buffer_size: *mut usize,
    buffer_duration: *mut f64,
) -> VoirsError {
    if recognizer.is_null() || buffer_size.is_null() || buffer_duration.is_null() {
        return VoirsError::NullPointer;
    }

    let catch_result = std::panic::catch_unwind(|| {
        let internal = unsafe { &*(recognizer as *const VoirsRecognizerInternal) };

        // Get actual buffer information
        if let Some(context_arc) = &internal.streaming_context {
            if let Ok(context) = context_arc.lock() {
                let size = context.audio_buffer.len();
                let duration = size as f64 / (internal.config.sample_rate as f64 * 2.0); // 2 bytes per sample

                unsafe {
                    *buffer_size = size;
                    *buffer_duration = duration;
                }

                return VoirsError::Success;
            }
        }

        unsafe {
            *buffer_size = 0;
            *buffer_duration = 0.0;
        }

        VoirsError::StreamingNotStarted
    });

    match catch_result {
        Ok(error) => error,
        Err(_) => VoirsError::InternalError,
    }
}

/// Configure streaming parameters during active streaming
///
/// # Arguments
/// * `recognizer` - Pointer to the recognizer instance
/// * `config` - New streaming configuration
///
/// # Returns
/// VoirsError::Success on success, or an error code on failure.
#[no_mangle]
pub extern "C" fn voirs_configure_streaming(
    recognizer: *mut VoirsRecognizer,
    config: *const VoirsStreamingConfig,
) -> VoirsError {
    if recognizer.is_null() || config.is_null() {
        return VoirsError::NullPointer;
    }

    let catch_result = std::panic::catch_unwind(|| {
        let internal = unsafe { &mut *(recognizer as *mut VoirsRecognizerInternal) };

        let new_config = unsafe { (*config).clone() };

        // Validate configuration
        if new_config.chunk_duration <= 0.0 || new_config.chunk_duration > 10.0 {
            return VoirsError::InvalidConfiguration;
        }

        if new_config.overlap_duration < 0.0
            || new_config.overlap_duration >= new_config.chunk_duration
        {
            return VoirsError::InvalidConfiguration;
        }

        if new_config.vad_threshold < 0.0 || new_config.vad_threshold > 1.0 {
            return VoirsError::InvalidConfiguration;
        }

        // Update streaming configuration
        if let Some(context_arc) = &internal.streaming_context {
            if let Ok(mut context) = context_arc.lock() {
                context.config = new_config;
                return VoirsError::Success;
            }
        }

        VoirsError::StreamingNotStarted
    });

    match catch_result {
        Ok(error) => error,
        Err(_) => VoirsError::InternalError,
    }
}

/// Flush any remaining audio in the streaming buffer
///
/// # Arguments
/// * `recognizer` - Pointer to the recognizer instance
///
/// # Returns
/// VoirsError::Success on success, or an error code on failure.
/// Final results are delivered via the callback function.
#[no_mangle]
/// Item
pub extern "C" fn voirs_flush_streaming_buffer(recognizer: *mut VoirsRecognizer) -> VoirsError {
    if recognizer.is_null() {
        return VoirsError::NullPointer;
    }

    let catch_result = std::panic::catch_unwind(|| {
        let internal = unsafe { &mut *(recognizer as *mut VoirsRecognizerInternal) };

        // Process any remaining audio in the buffer
        if let Some(context_arc) = &internal.streaming_context {
            let mut context = context_arc.lock().unwrap();
            if !context.is_active {
                return VoirsError::StreamingNotStarted;
            }

            if !context.audio_buffer.is_empty() {
                // Process remaining audio data
                let remaining_data: Vec<u8> = context.audio_buffer.drain(..).collect();
                let callback_data = (context.callback, context.user_data);

                // Drop the lock before processing
                drop(context);

                let start_time = Instant::now();
                let recognition_result = internal
                    .runtime
                    .block_on(async { internal.pipeline.recognize_bytes(&remaining_data).await });

                let processing_time = start_time.elapsed().as_millis() as f64;

                // Update metrics
                internal.metrics.processed_chunks += 1;
                internal.metrics.total_processing_time_ms += processing_time;
                internal.metrics.total_audio_duration_s +=
                    remaining_data.len() as f64 / (internal.config.sample_rate as f64 * 2.0);

                match recognition_result {
                    Ok(result) => {
                        // Convert to C API result format
                        let segments: Vec<VoirsSegment> = result
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

                        let c_result = VoirsRecognitionResult {
                            text: internal.memory_manager.store_string(
                                &result
                                    .transcription
                                    .as_ref()
                                    .map(|t| &t.text)
                                    .unwrap_or(&String::new()),
                            ),
                            confidence: result
                                .transcription
                                .as_ref()
                                .map(|t| t.confidence)
                                .unwrap_or(0.0),
                            language: result
                                .transcription
                                .as_ref()
                                .map(|t| {
                                    internal
                                        .memory_manager
                                        .store_string(&t.language.to_string())
                                })
                                .unwrap_or(std::ptr::null()),
                            processing_time_ms: processing_time,
                            audio_duration_s: result
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

                        // Call the callback if available
                        if let (Some(callback), user_data) = callback_data {
                            callback(&c_result, user_data);
                        }

                        VoirsError::Success
                    }
                    Err(_) => {
                        internal.metrics.failed_recognitions += 1;
                        VoirsError::RecognitionFailed
                    }
                }
            } else {
                VoirsError::Success
            }
        } else {
            VoirsError::StreamingNotStarted
        }
    });

    match catch_result {
        Ok(error) => error,
        Err(_) => VoirsError::InternalError,
    }
}

/// Get streaming statistics
///
/// # Arguments
/// * `recognizer` - Pointer to the recognizer instance
/// * `chunks_processed` - Output pointer for number of chunks processed
/// * `total_audio_duration` - Output pointer for total audio duration processed
/// * `average_latency` - Output pointer for average processing latency
///
/// # Returns
/// VoirsError::Success on success, or an error code on failure.
#[no_mangle]
pub extern "C" fn voirs_get_streaming_stats(
    recognizer: *mut VoirsRecognizer,
    chunks_processed: *mut usize,
    total_audio_duration: *mut f64,
    average_latency: *mut f64,
) -> VoirsError {
    if recognizer.is_null() {
        return VoirsError::NullPointer;
    }

    let catch_result = std::panic::catch_unwind(|| {
        let internal = unsafe { &*(recognizer as *const VoirsRecognizerInternal) };

        // Get actual streaming statistics
        if let Some(context_arc) = &internal.streaming_context {
            if let Ok(context) = context_arc.lock() {
                if !chunks_processed.is_null() {
                    unsafe {
                        *chunks_processed = context.chunk_count;
                    }
                }

                if !total_audio_duration.is_null() {
                    unsafe {
                        *total_audio_duration = context.total_audio_duration;
                    }
                }

                if !average_latency.is_null() {
                    unsafe {
                        *average_latency = context.average_latency;
                    }
                }

                return VoirsError::Success;
            }
        }

        // Fallback values if streaming not active
        if !chunks_processed.is_null() {
            unsafe {
                *chunks_processed = 0;
            }
        }

        if !total_audio_duration.is_null() {
            unsafe {
                *total_audio_duration = 0.0;
            }
        }

        if !average_latency.is_null() {
            unsafe {
                *average_latency = 0.0;
            }
        }

        VoirsError::StreamingNotStarted
    });

    match catch_result {
        Ok(error) => error,
        Err(_) => VoirsError::InternalError,
    }
}
