#!/usr/bin/env python3
"""
VoiRS Recognizer C Header Generator

This script generates a C header file from the Rust C API definitions.
"""

import os
import re
from pathlib import Path

def generate_header():
    """Generate the C header file for VoiRS Recognizer."""
    
    header_content = '''
/* VoiRS Speech Recognition C API
 * 
 * This header file provides C/C++ bindings for the VoiRS speech recognition library.
 * 
 * Usage:
 *   1. Include this header in your C/C++ project
 *   2. Link against libvoirs_recognizer (compiled with --features c-api)
 *   3. Call voirs_init() before using any other functions
 *   4. Create a recognizer with voirs_recognizer_create()
 *   5. Use voirs_recognize() or streaming functions for recognition
 *   6. Clean up with voirs_recognizer_destroy()
 * 
 * Example:
 *   VoirsRecognizer* recognizer = NULL;
 *   VoirsError error = voirs_recognizer_create(NULL, &recognizer);
 *   if (error == VOIRS_SUCCESS) {
 *       // Use recognizer...
 *       voirs_recognizer_destroy(recognizer);
 *   }
 */

#ifndef VOIRS_RECOGNIZER_H
#define VOIRS_RECOGNIZER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Version Information */
#define VOIRS_VERSION_MAJOR 0
#define VOIRS_VERSION_MINOR 1
#define VOIRS_VERSION_PATCH 0
#define VOIRS_VERSION_STRING "0.1.0"

/* Forward Declarations */
typedef struct VoirsRecognizer VoirsRecognizer;

/* Error Codes */
typedef enum {
    VOIRS_SUCCESS = 0,
    VOIRS_INVALID_ARGUMENT = 1,
    VOIRS_NULL_POINTER = 2,
    VOIRS_INITIALIZATION_FAILED = 3,
    VOIRS_MODEL_LOAD_FAILED = 4,
    VOIRS_RECOGNITION_FAILED = 5,
    VOIRS_UNSUPPORTED_FORMAT = 6,
    VOIRS_OUT_OF_MEMORY = 7,
    VOIRS_INTERNAL_ERROR = 8,
    VOIRS_STREAMING_NOT_STARTED = 9,
    VOIRS_INVALID_CONFIGURATION = 10
} VoirsError;

/* Audio Format Types */
typedef enum {
    VOIRS_AUDIO_PCM16 = 0,
    VOIRS_AUDIO_PCM32 = 1,
    VOIRS_AUDIO_FLOAT32 = 2,
    VOIRS_AUDIO_WAV = 3,
    VOIRS_AUDIO_MP3 = 4,
    VOIRS_AUDIO_FLAC = 5,
    VOIRS_AUDIO_OGG = 6,
    VOIRS_AUDIO_M4A = 7
} VoirsAudioFormatType;

/* Audio Format Structure */
typedef struct {
    uint32_t sample_rate;
    uint16_t channels;
    uint16_t bits_per_sample;
    VoirsAudioFormatType format;
} VoirsAudioFormat;

/* Recognition Configuration */
typedef struct {
    const char* model_name;
    const char* language;
    uint32_t sample_rate;
    bool enable_vad;
    float confidence_threshold;
    size_t beam_size;
    float temperature;
    bool suppress_blank;
} VoirsRecognitionConfig;

/* Streaming Configuration */
typedef struct {
    float chunk_duration;
    float overlap_duration;
    float vad_threshold;
    float silence_duration;
    size_t max_chunk_size;
} VoirsStreamingConfig;

/* Speech Segment */
typedef struct {
    double start_time;
    double end_time;
    const char* text;
    float confidence;
    float no_speech_prob;
} VoirsSegment;

/* Recognition Result */
typedef struct {
    const char* text;
    float confidence;
    const char* language;
    double processing_time_ms;
    double audio_duration_s;
    size_t segment_count;
    const VoirsSegment* segments;
} VoirsRecognitionResult;

/* Version Information */
typedef struct {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
    const char* version_string;
    const char* build_timestamp;
} VoirsVersion;

/* Capabilities */
typedef struct {
    bool streaming;
    bool multilingual;
    bool vad;
    bool confidence_scoring;
    bool segment_timestamps;
    bool language_detection;
    size_t supported_models_count;
    const char* const* supported_models;
    size_t supported_languages_count;
    const char* const* supported_languages;
} VoirsCapabilities;

/* Performance Metrics */
typedef struct {
    float real_time_factor;
    float avg_processing_time_ms;
    float peak_processing_time_ms;
    size_t memory_usage_bytes;
    size_t processed_chunks;
    size_t failed_recognitions;
} VoirsPerformanceMetrics;

/* Callback Types */
typedef void (*VoirsStreamingCallback)(
    const VoirsRecognitionResult* result,
    void* user_data
);

typedef void (*VoirsProgressCallback)(
    float progress,
    const char* message,
    void* user_data
);

/* Core Functions */

/**
 * Initialize the VoiRS recognizer library.
 * This function must be called before any other VoiRS functions.
 * 
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_init(void);

/**
 * Get version information of the VoiRS library.
 * 
 * @return Pointer to version information structure
 */
const VoirsVersion* voirs_get_version(void);

/**
 * Create a new VoiRS recognizer instance.
 * 
 * @param config Configuration for the recognizer (can be NULL for defaults)
 * @param recognizer Output pointer to store the created recognizer instance
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_recognizer_create(
    const VoirsRecognitionConfig* config,
    VoirsRecognizer** recognizer
);

/**
 * Destroy a VoiRS recognizer instance.
 * 
 * @param recognizer Pointer to the recognizer instance to destroy
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_recognizer_destroy(VoirsRecognizer* recognizer);

/**
 * Get the capabilities of the recognizer.
 * 
 * @param recognizer Pointer to the recognizer instance
 * @param capabilities Output pointer to store the capabilities
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_recognizer_get_capabilities(
    VoirsRecognizer* recognizer,
    const VoirsCapabilities** capabilities
);

/**
 * Get performance metrics from the recognizer.
 * 
 * @param recognizer Pointer to the recognizer instance
 * @param metrics Output pointer to store the metrics
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_recognizer_get_metrics(
    VoirsRecognizer* recognizer,
    const VoirsPerformanceMetrics** metrics
);

/**
 * Switch to a different model.
 * 
 * @param recognizer Pointer to the recognizer instance
 * @param model_name Name of the model to switch to
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_recognizer_switch_model(
    VoirsRecognizer* recognizer,
    const char* model_name
);

/* Recognition Functions */

/**
 * Recognize speech from audio data.
 * 
 * @param recognizer Pointer to the recognizer instance
 * @param audio_data Pointer to audio data buffer
 * @param audio_size Size of audio data in bytes
 * @param audio_format Audio format information (can be NULL for defaults)
 * @param result Output pointer to store the recognition result
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_recognize(
    VoirsRecognizer* recognizer,
    const uint8_t* audio_data,
    size_t audio_size,
    const VoirsAudioFormat* audio_format,
    const VoirsRecognitionResult** result
);

/**
 * Recognize speech from a file.
 * 
 * @param recognizer Pointer to the recognizer instance
 * @param file_path Path to the audio file
 * @param result Output pointer to store the recognition result
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_recognize_file(
    VoirsRecognizer* recognizer,
    const char* file_path,
    const VoirsRecognitionResult** result
);

/**
 * Free memory allocated for recognition results.
 * 
 * @param recognizer Pointer to the recognizer instance
 * @param result Pointer to the result to free
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_free_result(
    VoirsRecognizer* recognizer,
    const VoirsRecognitionResult* result
);

/* Streaming Functions */

/**
 * Start streaming recognition mode.
 * 
 * @param recognizer Pointer to the recognizer instance
 * @param config Streaming configuration (can be NULL for defaults)
 * @param callback Callback function to receive streaming results
 * @param user_data User data pointer passed to the callback
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_start_streaming(
    VoirsRecognizer* recognizer,
    const VoirsStreamingConfig* config,
    VoirsStreamingCallback callback,
    void* user_data
);

/**
 * Stop streaming recognition mode.
 * 
 * @param recognizer Pointer to the recognizer instance
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_stop_streaming(VoirsRecognizer* recognizer);

/**
 * Process an audio chunk in streaming mode.
 * 
 * @param recognizer Pointer to the recognizer instance
 * @param audio_data Pointer to audio data buffer
 * @param audio_size Size of audio data in bytes
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_stream_audio(
    VoirsRecognizer* recognizer,
    const uint8_t* audio_data,
    size_t audio_size
);

/**
 * Check if streaming mode is active.
 * 
 * @param recognizer Pointer to the recognizer instance
 * @return true if streaming is active, false otherwise
 */
bool voirs_is_streaming_active(VoirsRecognizer* recognizer);

/**
 * Get streaming buffer information.
 * 
 * @param recognizer Pointer to the recognizer instance
 * @param buffer_size Output pointer for current buffer size in bytes
 * @param buffer_duration Output pointer for buffer duration in seconds
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_get_streaming_buffer_info(
    VoirsRecognizer* recognizer,
    size_t* buffer_size,
    double* buffer_duration
);

/**
 * Configure streaming parameters during active streaming.
 * 
 * @param recognizer Pointer to the recognizer instance
 * @param config New streaming configuration
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_configure_streaming(
    VoirsRecognizer* recognizer,
    const VoirsStreamingConfig* config
);

/**
 * Flush any remaining audio in the streaming buffer.
 * 
 * @param recognizer Pointer to the recognizer instance
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_flush_streaming_buffer(VoirsRecognizer* recognizer);

/**
 * Get streaming statistics.
 * 
 * @param recognizer Pointer to the recognizer instance
 * @param chunks_processed Output pointer for number of chunks processed
 * @param total_audio_duration Output pointer for total audio duration processed
 * @param average_latency Output pointer for average processing latency
 * @return VOIRS_SUCCESS on success, or an error code on failure
 */
VoirsError voirs_get_streaming_stats(
    VoirsRecognizer* recognizer,
    size_t* chunks_processed,
    double* total_audio_duration,
    double* average_latency
);

/* Error Handling Functions */

/**
 * Get the last error message.
 * 
 * @return Pointer to error message string, or NULL if no error
 */
const char* voirs_get_last_error(void);

/**
 * Clear the last error message.
 */
void voirs_clear_error(void);

/**
 * Convert a VoirsError to a human-readable string.
 * 
 * @param error The error code to convert
 * @return Pointer to error description string
 */
const char* voirs_error_to_string(VoirsError error);

/**
 * Check if an error code represents success.
 * 
 * @param error The error code to check
 * @return true if success, false otherwise
 */
bool voirs_is_success(VoirsError error);

/**
 * Check if an error code represents a failure.
 * 
 * @param error The error code to check
 * @return true if failure, false otherwise
 */
bool voirs_is_error(VoirsError error);

/* Utility Macros */

#define VOIRS_CHECK(call) \
    do { \
        VoirsError __err = (call); \
        if (__err != VOIRS_SUCCESS) { \
            fprintf(stderr, "VoiRS Error: %s\n", voirs_error_to_string(__err)); \
            return __err; \
        } \
    } while(0)

#define VOIRS_DEFAULT_CONFIG() \
    (VoirsRecognitionConfig) { \
        .model_name = NULL, \
        .language = NULL, \
        .sample_rate = 16000, \
        .enable_vad = true, \
        .confidence_threshold = 0.5f, \
        .beam_size = 5, \
        .temperature = 0.0f, \
        .suppress_blank = true \
    }

#define VOIRS_DEFAULT_STREAMING_CONFIG() \
    (VoirsStreamingConfig) { \
        .chunk_duration = 1.0f, \
        .overlap_duration = 0.1f, \
        .vad_threshold = 0.5f, \
        .silence_duration = 2.0f, \
        .max_chunk_size = 16000 \
    }

#define VOIRS_DEFAULT_AUDIO_FORMAT() \
    (VoirsAudioFormat) { \
        .sample_rate = 16000, \
        .channels = 1, \
        .bits_per_sample = 16, \
        .format = VOIRS_AUDIO_PCM16 \
    }

#ifdef __cplusplus
}
#endif

#endif /* VOIRS_RECOGNIZER_H */
'''
    
    # Write the header file
    header_path = Path(__file__).parent / "include" / "voirs_recognizer.h"
    header_path.parent.mkdir(exist_ok=True)
    
    with open(header_path, 'w') as f:
        f.write(header_content)
    
    print(f"Generated C header: {header_path}")
    return header_path

if __name__ == "__main__":
    generate_header()
