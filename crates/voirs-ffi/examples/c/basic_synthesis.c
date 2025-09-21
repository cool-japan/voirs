/**
 * Basic Speech Synthesis Example
 * 
 * This example demonstrates the basic usage of the VoiRS C API
 * for text-to-speech synthesis. It shows how to:
 * 
 * 1. Initialize a VoiRS pipeline
 * 2. Synthesize text to audio
 * 3. Handle errors
 * 4. Clean up resources
 * 
 * Compile with:
 * gcc -o basic_synthesis basic_synthesis.c -lvoirs_ffi -lm
 * 
 * Run with:
 * ./basic_synthesis "Hello, World!"
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// VoiRS C API types and functions
typedef unsigned int VoirsPipelineId;
typedef int VoirsErrorCode;

typedef struct {
    float* samples;
    unsigned int sample_count;
    unsigned int sample_rate;
    unsigned int channels;
} VoirsAudioBuffer;

typedef struct {
    VoirsAudioBuffer* audio;
    float quality_score;
    unsigned int synthesis_time_ms;
    unsigned int text_length;
    char* metadata;
} VoirsSynthesisResult;

typedef struct {
    float speed;
    float pitch;
    float volume;
    int voice_id;
    int quality_level;  // 0=Low, 1=Medium, 2=High, 3=Ultra
} VoirsSynthesisConfig;

// Core functions
extern VoirsPipelineId voirs_create_pipeline(void);
extern VoirsErrorCode voirs_destroy_pipeline(VoirsPipelineId pipeline_id);
extern const char* voirs_get_last_error(void);

// Synthesis functions
extern VoirsSynthesisResult* voirs_synthesize_advanced(
    const char* text,
    const VoirsSynthesisConfig* config
);
extern void voirs_free_synthesis_result(VoirsSynthesisResult* result);

// Utility functions
extern const char* voirs_get_version_string(void);
extern VoirsErrorCode voirs_initialize_logging(const char* level);

// Error codes
#define VOIRS_SUCCESS 0
#define VOIRS_ERROR_INVALID_PIPELINE 1
#define VOIRS_ERROR_SYNTHESIS_FAILED 2

void print_usage(const char* program_name) {
    printf("Usage: %s <text>\n", program_name);
    printf("Example: %s \"Hello, World!\"\n", program_name);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* text = argv[1];
    
    printf("VoiRS Basic Synthesis Example\n");
    printf("Version: %s\n", voirs_get_version_string());
    printf("Text to synthesize: \"%s\"\n\n", text);

    // Initialize logging (optional)
    voirs_initialize_logging("info");

    // Create pipeline
    printf("Creating VoiRS pipeline...\n");
    VoirsPipelineId pipeline = voirs_create_pipeline();
    if (pipeline == 0) {
        fprintf(stderr, "Failed to create pipeline: %s\n", voirs_get_last_error());
        return 1;
    }
    printf("Pipeline created successfully (ID: %u)\n", pipeline);

    // Configure synthesis parameters
    VoirsSynthesisConfig config = {
        .speed = 1.0f,        // Normal speed
        .pitch = 1.0f,        // Normal pitch
        .volume = 0.8f,       // 80% volume
        .voice_id = 0,        // Default voice
        .quality_level = 2    // High quality
    };

    // Perform synthesis
    printf("Synthesizing text...\n");
    VoirsSynthesisResult* result = voirs_synthesize_advanced(text, &config);
    
    if (result == NULL) {
        fprintf(stderr, "Synthesis failed: %s\n", voirs_get_last_error());
        voirs_destroy_pipeline(pipeline);
        return 1;
    }

    // Display synthesis results
    printf("Synthesis completed successfully!\n");
    printf("Audio properties:\n");
    printf("  - Sample count: %u\n", result->audio->sample_count);
    printf("  - Sample rate: %u Hz\n", result->audio->sample_rate);
    printf("  - Channels: %u\n", result->audio->channels);
    printf("  - Duration: %.2f seconds\n", 
           (float)result->audio->sample_count / result->audio->sample_rate);
    
    printf("Synthesis metrics:\n");
    printf("  - Quality score: %.2f\n", result->quality_score);
    printf("  - Synthesis time: %u ms\n", result->synthesis_time_ms);
    printf("  - Text length: %u characters\n", result->text_length);
    
    if (result->metadata) {
        printf("  - Metadata: %s\n", result->metadata);
    }

    // Calculate some audio statistics
    if (result->audio->samples && result->audio->sample_count > 0) {
        float max_amplitude = 0.0f;
        float sum_squares = 0.0f;
        
        for (unsigned int i = 0; i < result->audio->sample_count; i++) {
            float sample = result->audio->samples[i];
            if (sample < 0) sample = -sample;  // Absolute value
            if (sample > max_amplitude) {
                max_amplitude = sample;
            }
            sum_squares += result->audio->samples[i] * result->audio->samples[i];
        }
        
        float rms = sqrt(sum_squares / result->audio->sample_count);
        
        printf("Audio analysis:\n");
        printf("  - Peak amplitude: %.4f\n", max_amplitude);
        printf("  - RMS level: %.4f\n", rms);
        printf("  - Dynamic range: %.2f dB\n", 20.0f * log10f(max_amplitude / (rms + 1e-10f)));
    }

    // Save audio to file (basic WAV format)
    printf("\nSaving audio to 'output.raw' (32-bit float samples)...\n");
    FILE* output_file = fopen("output.raw", "wb");
    if (output_file) {
        size_t written = fwrite(result->audio->samples, 
                               sizeof(float), 
                               result->audio->sample_count, 
                               output_file);
        fclose(output_file);
        printf("Wrote %zu samples to 'output.raw'\n", written);
        printf("To convert to WAV: ffmpeg -f f32le -ar %u -ac %u -i output.raw output.wav\n",
               result->audio->sample_rate, result->audio->channels);
    } else {
        fprintf(stderr, "Warning: Could not save audio file\n");
    }

    // Clean up
    printf("\nCleaning up...\n");
    voirs_free_synthesis_result(result);
    
    VoirsErrorCode destroy_result = voirs_destroy_pipeline(pipeline);
    if (destroy_result != VOIRS_SUCCESS) {
        fprintf(stderr, "Warning: Failed to destroy pipeline: %s\n", voirs_get_last_error());
    } else {
        printf("Pipeline destroyed successfully\n");
    }

    printf("Synthesis example completed!\n");
    return 0;
}