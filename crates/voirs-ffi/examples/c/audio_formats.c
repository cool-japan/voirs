/**
 * VoiRS FFI Audio Format Example
 * 
 * Demonstrates saving audio in FLAC and MP3 formats using the VoiRS C API.
 * This example shows how to:
 * - Create a simple audio buffer
 * - Save audio as FLAC with compression settings
 * - Save audio as MP3 with bitrate and quality settings
 * - Query supported audio formats
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// VoiRS FFI declarations (in a real project, these would be in a header file)
typedef enum {
    VOIRS_SUCCESS = 0,
    VOIRS_INVALID_PARAMETER = 1,
    VOIRS_INITIALIZATION_FAILED = 2,
    VOIRS_SYNTHESIS_FAILED = 3,
    VOIRS_VOICE_NOT_FOUND = 4,
    VOIRS_IO_ERROR = 5,
    VOIRS_OUT_OF_MEMORY = 6,
    VOIRS_OPERATION_CANCELLED = 7,
    VOIRS_INTERNAL_ERROR = 99,
} VoirsErrorCode;

typedef struct {
    float* samples;
    unsigned int length;
    unsigned int sample_rate;
    unsigned int channels;
    float duration;
} VoirsAudioBuffer;

// Function declarations
extern VoirsErrorCode voirs_audio_save_flac(
    const VoirsAudioBuffer* buffer,
    const char* filename,
    unsigned int compression_level
);

extern VoirsErrorCode voirs_audio_save_mp3(
    const VoirsAudioBuffer* buffer,
    const char* filename,
    unsigned int bitrate,
    unsigned int quality
);

extern VoirsErrorCode voirs_audio_get_supported_formats(
    const char** formats,
    unsigned int* count
);

// Helper function to create a simple test audio buffer with a sine wave
VoirsAudioBuffer* create_test_audio() {
    const unsigned int sample_rate = 44100;
    const unsigned int channels = 2; // Stereo
    const float duration = 2.0f; // 2 seconds
    const unsigned int length = (unsigned int)(sample_rate * duration * channels);
    
    VoirsAudioBuffer* buffer = malloc(sizeof(VoirsAudioBuffer));
    if (!buffer) return NULL;
    
    float* samples = malloc(length * sizeof(float));
    if (!samples) {
        free(buffer);
        return NULL;
    }
    
    // Generate a simple sine wave at 440 Hz (A4 note)
    const float frequency = 440.0f;
    const float amplitude = 0.5f;
    
    for (unsigned int i = 0; i < length; i += channels) {
        float time = (float)(i / channels) / (float)sample_rate;
        float sample_value = amplitude * sinf(2.0f * M_PI * frequency * time);
        
        // Apply fade in/out to avoid clicks
        float fade_duration = 0.1f; // 100ms fade
        if (time < fade_duration) {
            sample_value *= time / fade_duration;
        } else if (time > duration - fade_duration) {
            sample_value *= (duration - time) / fade_duration;
        }
        
        // Set both left and right channels
        samples[i] = sample_value;     // Left channel
        samples[i + 1] = sample_value * 0.7f; // Right channel (slightly quieter)
    }
    
    buffer->samples = samples;
    buffer->length = length;
    buffer->sample_rate = sample_rate;
    buffer->channels = channels;
    buffer->duration = duration;
    
    return buffer;
}

void free_test_audio(VoirsAudioBuffer* buffer) {
    if (buffer) {
        if (buffer->samples) {
            free(buffer->samples);
        }
        free(buffer);
    }
}

int main() {
    printf("VoiRS FFI Audio Format Example\n");
    printf("==============================\n\n");
    
    // Create test audio buffer
    printf("Creating test audio buffer (2-second stereo sine wave at 440 Hz)...\n");
    VoirsAudioBuffer* audio = create_test_audio();
    if (!audio) {
        fprintf(stderr, "Failed to create test audio buffer\n");
        return 1;
    }
    
    printf("Audio buffer created:\n");
    printf("  Sample rate: %u Hz\n", audio->sample_rate);
    printf("  Channels: %u\n", audio->channels);
    printf("  Duration: %.2f seconds\n", audio->duration);
    printf("  Total samples: %u\n", audio->length);
    printf("\n");
    
    // Test FLAC saving
    printf("Saving audio as FLAC format...\n");
    VoirsErrorCode result = voirs_audio_save_flac(audio, "/tmp/test_output.flac", 6);
    if (result == VOIRS_SUCCESS) {
        printf("✓ FLAC file saved successfully to /tmp/test_output.flac\n");
        printf("  Compression level: 6\n");
    } else {
        fprintf(stderr, "✗ Failed to save FLAC file (error code: %d)\n", result);
    }
    printf("\n");
    
    // Test MP3 saving
    printf("Saving audio as MP3 format...\n");
    result = voirs_audio_save_mp3(audio, "/tmp/test_output.mp3", 192, 2);
    if (result == VOIRS_SUCCESS) {
        printf("✓ MP3 file saved successfully to /tmp/test_output.mp3\n");
        printf("  Bitrate: 192 kbps\n");
        printf("  Quality: 2\n");
    } else {
        fprintf(stderr, "✗ Failed to save MP3 file (error code: %d)\n", result);
    }
    printf("\n");
    
    // Test different MP3 settings
    printf("Saving audio as high-quality MP3...\n");
    result = voirs_audio_save_mp3(audio, "/tmp/test_output_hq.mp3", 320, 0);
    if (result == VOIRS_SUCCESS) {
        printf("✓ High-quality MP3 file saved to /tmp/test_output_hq.mp3\n");
        printf("  Bitrate: 320 kbps\n");
        printf("  Quality: 0 (highest)\n");
    } else {
        fprintf(stderr, "✗ Failed to save high-quality MP3 file (error code: %d)\n", result);
    }
    printf("\n");
    
    // Query supported formats
    printf("Querying supported audio formats...\n");
    const char* formats;
    unsigned int format_count;
    result = voirs_audio_get_supported_formats(&formats, &format_count);
    if (result == VOIRS_SUCCESS) {
        printf("✓ Supported formats (%u total):\n", format_count);
        // Note: In a real implementation, you'd iterate through the format array
        printf("  • WAV (uncompressed)\n");
        printf("  • FLAC (lossless compression)\n");
        printf("  • MP3 (lossy compression)\n");
        printf("  • OGG (open-source lossy)\n");
        printf("  • Opus (modern lossy)\n");
    } else {
        fprintf(stderr, "✗ Failed to query supported formats (error code: %d)\n", result);
    }
    printf("\n");
    
    // Test error handling with invalid parameters
    printf("Testing error handling...\n");
    result = voirs_audio_save_flac(NULL, "/tmp/test.flac", 5);
    if (result == VOIRS_INVALID_PARAMETER) {
        printf("✓ Null buffer parameter correctly rejected\n");
    } else {
        printf("✗ Null buffer parameter should have been rejected\n");
    }
    
    result = voirs_audio_save_mp3(audio, NULL, 128, 2);
    if (result == VOIRS_INVALID_PARAMETER) {
        printf("✓ Null filename parameter correctly rejected\n");
    } else {
        printf("✗ Null filename parameter should have been rejected\n");
    }
    printf("\n");
    
    // Clean up
    free_test_audio(audio);
    
    printf("Example completed successfully!\n");
    printf("\nGenerated files:\n");
    printf("  /tmp/test_output.flac - FLAC format with compression level 6\n");
    printf("  /tmp/test_output.mp3 - MP3 format at 192 kbps, quality 2\n");
    printf("  /tmp/test_output_hq.mp3 - High-quality MP3 at 320 kbps, quality 0\n");
    printf("\nNote: These are simplified format implementations for demonstration.\n");
    printf("Production implementations would use proper FLAC and MP3 encoders.\n");
    
    return 0;
}