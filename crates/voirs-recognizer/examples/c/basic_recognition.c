/*
 * VoiRS Speech Recognition - Basic C Example
 * 
 * This example demonstrates basic usage of the VoiRS speech recognition
 * library from C code.
 * 
 * Compile with:
 *   gcc -o basic_recognition basic_recognition.c -lvoirs_recognizer
 * 
 * Make sure to build the Rust library with:
 *   cargo build --release --features c-api
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../include/voirs_recognizer.h"

// Function to read a WAV file (simplified - assumes 16-bit PCM)
static uint8_t* read_wav_file(const char* filename, size_t* size, VoirsAudioFormat* format) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }
    
    // Simple WAV header parsing (44 bytes)
    uint8_t header[44];
    if (fread(header, 1, 44, file) != 44) {
        fprintf(stderr, "Error: Invalid WAV file\n");
        fclose(file);
        return NULL;
    }
    
    // Extract format information from WAV header
    uint16_t audio_format = *(uint16_t*)(header + 20);
    uint16_t channels = *(uint16_t*)(header + 22);
    uint32_t sample_rate = *(uint32_t*)(header + 24);
    uint16_t bits_per_sample = *(uint16_t*)(header + 34);
    uint32_t data_size = *(uint32_t*)(header + 40);
    
    // Set format structure
    format->sample_rate = sample_rate;
    format->channels = channels;
    format->bits_per_sample = bits_per_sample;
    format->format = VOIRS_AUDIO_WAV;
    
    printf("WAV file info: %d Hz, %d channels, %d bits\n", 
           sample_rate, channels, bits_per_sample);
    
    // Read audio data
    uint8_t* audio_data = malloc(data_size);
    if (!audio_data) {
        fprintf(stderr, "Error: Could not allocate memory\n");
        fclose(file);
        return NULL;
    }
    
    if (fread(audio_data, 1, data_size, file) != data_size) {
        fprintf(stderr, "Error: Could not read audio data\n");
        free(audio_data);
        fclose(file);
        return NULL;
    }
    
    fclose(file);
    *size = data_size;
    return audio_data;
}

// Function to generate sample sine wave audio
static uint8_t* generate_sample_audio(size_t* size, VoirsAudioFormat* format) {
    const uint32_t sample_rate = 16000;
    const float duration = 2.0f; // 2 seconds
    const float frequency = 440.0f; // A4 note
    
    size_t sample_count = (size_t)(sample_rate * duration);
    *size = sample_count * 2; // 16-bit samples
    
    uint8_t* audio_data = malloc(*size);
    if (!audio_data) {
        return NULL;
    }
    
    int16_t* samples = (int16_t*)audio_data;
    
    for (size_t i = 0; i < sample_count; i++) {
        float t = (float)i / sample_rate;
        float sample = 0.1f * sinf(2.0f * 3.14159f * frequency * t);
        samples[i] = (int16_t)(sample * 32767.0f);
    }
    
    format->sample_rate = sample_rate;
    format->channels = 1;
    format->bits_per_sample = 16;
    format->format = VOIRS_AUDIO_PCM16;
    
    printf("Generated %zu samples of sine wave audio\n", sample_count);
    return audio_data;
}

// Function to print recognition result
static void print_result(const VoirsRecognitionResult* result) {
    printf("\n=== Recognition Result ===\n");
    printf("Text: \"%s\"\n", result->text ? result->text : "(no text)");
    printf("Confidence: %.1f%%\n", result->confidence * 100.0f);
    printf("Language: %s\n", result->language ? result->language : "unknown");
    printf("Processing time: %.1f ms\n", result->processing_time_ms);
    printf("Audio duration: %.2f seconds\n", result->audio_duration_s);
    printf("Segment count: %zu\n", result->segment_count);
    
    if (result->segments && result->segment_count > 0) {
        printf("\n=== Segments ===\n");
        for (size_t i = 0; i < result->segment_count; i++) {
            const VoirsSegment* seg = &result->segments[i];
            printf("%zu. [%.2fs - %.2fs] \"%s\" (%.1f%%)\n",
                   i + 1, seg->start_time, seg->end_time, 
                   seg->text ? seg->text : "", seg->confidence * 100.0f);
        }
    }
}

// Function to print capabilities
static void print_capabilities(const VoirsCapabilities* caps) {
    printf("\n=== Recognizer Capabilities ===\n");
    printf("Streaming: %s\n", caps->streaming ? "Yes" : "No");
    printf("Multilingual: %s\n", caps->multilingual ? "Yes" : "No");
    printf("VAD: %s\n", caps->vad ? "Yes" : "No");
    printf("Confidence scoring: %s\n", caps->confidence_scoring ? "Yes" : "No");
    printf("Segment timestamps: %s\n", caps->segment_timestamps ? "Yes" : "No");
    printf("Language detection: %s\n", caps->language_detection ? "Yes" : "No");
    
    printf("\nSupported models (%zu):\n", caps->supported_models_count);
    for (size_t i = 0; i < caps->supported_models_count; i++) {
        printf("  - %s\n", caps->supported_models[i]);
    }
    
    printf("\nSupported languages (%zu):\n", caps->supported_languages_count);
    for (size_t i = 0; i < caps->supported_languages_count; i++) {
        printf("  - %s\n", caps->supported_languages[i]);
    }
}

int main(int argc, char* argv[]) {
    printf("VoiRS Speech Recognition - Basic C Example\n");
    printf("==========================================\n\n");
    
    // Initialize the library
    printf("Initializing VoiRS library...\n");
    VoirsError error = voirs_init();
    if (error != VOIRS_SUCCESS) {
        fprintf(stderr, "Failed to initialize VoiRS: %s\n", voirs_error_to_string(error));
        return 1;
    }
    
    // Get version information
    const VoirsVersion* version = voirs_get_version();
    printf("VoiRS Version: %s\n", version->version_string);
    printf("Build timestamp: %s\n\n", version->build_timestamp);
    
    // Create recognizer configuration
    VoirsRecognitionConfig config = VOIRS_DEFAULT_CONFIG();
    config.model_name = "whisper-base";
    config.language = "en";
    
    // Create recognizer
    printf("Creating recognizer...\n");
    VoirsRecognizer* recognizer = NULL;
    error = voirs_recognizer_create(&config, &recognizer);
    if (error != VOIRS_SUCCESS) {
        fprintf(stderr, "Failed to create recognizer: %s\n", voirs_error_to_string(error));
        return 1;
    }
    
    printf("Recognizer created successfully!\n");
    
    // Get capabilities
    const VoirsCapabilities* capabilities = NULL;
    error = voirs_recognizer_get_capabilities(recognizer, &capabilities);
    if (error == VOIRS_SUCCESS && capabilities) {
        print_capabilities(capabilities);
    }
    
    // Prepare audio data
    uint8_t* audio_data = NULL;
    size_t audio_size = 0;
    VoirsAudioFormat audio_format;
    
    if (argc > 1) {
        // Use provided audio file
        printf("\nReading audio file: %s\n", argv[1]);
        audio_data = read_wav_file(argv[1], &audio_size, &audio_format);
    } else {
        // Generate sample audio
        printf("\nNo audio file provided, generating sample audio...\n");
        audio_data = generate_sample_audio(&audio_size, &audio_format);
    }
    
    if (!audio_data) {
        fprintf(stderr, "Failed to prepare audio data\n");
        voirs_recognizer_destroy(recognizer);
        return 1;
    }
    
    // Perform recognition
    printf("\nPerforming speech recognition...\n");
    const VoirsRecognitionResult* result = NULL;
    error = voirs_recognize(recognizer, audio_data, audio_size, &audio_format, &result);
    
    if (error == VOIRS_SUCCESS && result) {
        print_result(result);
        
        // Free the result
        voirs_free_result(recognizer, result);
    } else {
        fprintf(stderr, "Recognition failed: %s\n", voirs_error_to_string(error));
        const char* last_error = voirs_get_last_error();
        if (last_error) {
            fprintf(stderr, "Last error: %s\n", last_error);
        }
    }
    
    // Get performance metrics
    const VoirsPerformanceMetrics* metrics = NULL;
    error = voirs_recognizer_get_metrics(recognizer, &metrics);
    if (error == VOIRS_SUCCESS && metrics) {
        printf("\n=== Performance Metrics ===\n");
        printf("Real-time factor: %.2f\n", metrics->real_time_factor);
        printf("Average processing time: %.1f ms\n", metrics->avg_processing_time_ms);
        printf("Peak processing time: %.1f ms\n", metrics->peak_processing_time_ms);
        printf("Memory usage: %.1f MB\n", metrics->memory_usage_bytes / (1024.0 * 1024.0));
        printf("Processed chunks: %zu\n", metrics->processed_chunks);
        printf("Failed recognitions: %zu\n", metrics->failed_recognitions);
    }
    
    // Cleanup
    free(audio_data);
    voirs_recognizer_destroy(recognizer);
    
    printf("\nExample completed successfully!\n");
    return 0;
}
