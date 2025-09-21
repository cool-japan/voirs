/**
 * Streaming Speech Synthesis Example
 * 
 * This example demonstrates real-time streaming synthesis using the VoiRS C API.
 * It shows how to:
 * 
 * 1. Set up streaming synthesis with callbacks
 * 2. Process audio chunks as they become available
 * 3. Handle progress updates
 * 4. Manage concurrent operations
 * 
 * Compile with:
 * gcc -o streaming_synthesis streaming_synthesis.c -lvoirs_ffi -lm -lpthread
 * 
 * Run with:
 * ./streaming_synthesis "This is a longer text that will be synthesized in real-time chunks"
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>

// VoiRS C API types
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
    int quality_level;
} VoirsSynthesisConfig;

// Callback types
typedef void (*VoirsSynthesisProgressCallback)(VoirsPipelineId pipeline_id, float progress, void* user_data);
typedef void (*VoirsSynthesisCompleteCallback)(VoirsPipelineId pipeline_id, VoirsSynthesisResult* result, void* user_data);

// Core functions
extern VoirsPipelineId voirs_create_pipeline(void);
extern VoirsErrorCode voirs_destroy_pipeline(VoirsPipelineId pipeline_id);
extern const char* voirs_get_last_error(void);

// Streaming functions
extern VoirsSynthesisResult* voirs_synthesize_streaming_advanced(
    const char* text,
    const VoirsSynthesisConfig* config
);
extern void voirs_free_synthesis_result(VoirsSynthesisResult* result);

// Threading and callback functions
extern VoirsErrorCode voirs_register_callbacks(
    VoirsPipelineId pipeline_id,
    VoirsSynthesisProgressCallback progress_callback,
    VoirsSynthesisCompleteCallback complete_callback,
    void* user_data
);
extern VoirsErrorCode voirs_unregister_callbacks(VoirsPipelineId pipeline_id);
extern VoirsErrorCode voirs_set_global_thread_count(unsigned int thread_count);
extern unsigned int voirs_get_global_thread_count(void);

// Utility functions
extern const char* voirs_get_version_string(void);
extern VoirsErrorCode voirs_initialize_logging(const char* level);

// Error codes
#define VOIRS_SUCCESS 0

// Global state for the example
typedef struct {
    VoirsPipelineId pipeline_id;
    int synthesis_complete;
    int total_chunks_received;
    FILE* output_file;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} StreamingContext;

static StreamingContext g_context = {0};
static volatile int g_should_exit = 0;

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    printf("\nReceived signal %d, shutting down gracefully...\n", sig);
    g_should_exit = 1;
}

// Progress callback - called when synthesis progress updates
void progress_callback(VoirsPipelineId pipeline_id, float progress, void* user_data) {
    StreamingContext* ctx = (StreamingContext*)user_data;
    
    pthread_mutex_lock(&ctx->mutex);
    
    printf("\rSynthesis progress: %.1f%%", progress * 100.0f);
    fflush(stdout);
    
    // Check if we should cancel
    if (g_should_exit) {
        printf("\nCancellation requested...\n");
    }
    
    pthread_mutex_unlock(&ctx->mutex);
}

// Completion callback - called when synthesis completes or produces a chunk
void complete_callback(VoirsPipelineId pipeline_id, VoirsSynthesisResult* result, void* user_data) {
    StreamingContext* ctx = (StreamingContext*)user_data;
    
    pthread_mutex_lock(&ctx->mutex);
    
    if (result && result->audio) {
        ctx->total_chunks_received++;
        
        printf("\n[Chunk %d] Received audio chunk:\n", ctx->total_chunks_received);
        printf("  - Samples: %u\n", result->audio->sample_count);
        printf("  - Duration: %.3f seconds\n", 
               (float)result->audio->sample_count / result->audio->sample_rate);
        printf("  - Quality: %.2f\n", result->quality_score);
        
        // Write audio chunk to file
        if (ctx->output_file && result->audio->samples) {
            size_t written = fwrite(result->audio->samples, 
                                   sizeof(float), 
                                   result->audio->sample_count, 
                                   ctx->output_file);
            printf("  - Written to file: %zu samples\n", written);
        }
        
        // Calculate audio statistics for this chunk
        if (result->audio->samples && result->audio->sample_count > 0) {
            float max_amp = 0.0f;
            float sum_squares = 0.0f;
            
            for (unsigned int i = 0; i < result->audio->sample_count; i++) {
                float sample = result->audio->samples[i];
                if (sample < 0) sample = -sample;
                if (sample > max_amp) max_amp = sample;
                sum_squares += result->audio->samples[i] * result->audio->samples[i];
            }
            
            float rms = sqrt(sum_squares / result->audio->sample_count);
            printf("  - Peak amplitude: %.4f\n", max_amp);
            printf("  - RMS level: %.4f\n", rms);
        }
    } else {
        printf("\nSynthesis completed!\n");
        ctx->synthesis_complete = 1;
    }
    
    pthread_cond_signal(&ctx->cond);
    pthread_mutex_unlock(&ctx->mutex);
}

void print_usage(const char* program_name) {
    printf("Usage: %s <text>\n", program_name);
    printf("Example: %s \"This is a longer text for streaming synthesis\"\n", program_name);
    printf("\nPress Ctrl+C to cancel synthesis\n");
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* text = argv[1];
    
    printf("VoiRS Streaming Synthesis Example\n");
    printf("Version: %s\n", voirs_get_version_string());
    printf("Text to synthesize: \"%s\"\n", text);
    printf("Text length: %zu characters\n\n", strlen(text));

    // Set up signal handling for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize context
    pthread_mutex_init(&g_context.mutex, NULL);
    pthread_cond_init(&g_context.cond, NULL);
    g_context.synthesis_complete = 0;
    g_context.total_chunks_received = 0;

    // Initialize logging
    voirs_initialize_logging("info");

    // Configure threading
    unsigned int num_threads = 4;  // Use 4 threads for better streaming performance
    voirs_set_global_thread_count(num_threads);
    printf("Configured %u threads for synthesis\n", voirs_get_global_thread_count());

    // Create pipeline
    printf("Creating streaming pipeline...\n");
    g_context.pipeline_id = voirs_create_pipeline();
    if (g_context.pipeline_id == 0) {
        fprintf(stderr, "Failed to create pipeline: %s\n", voirs_get_last_error());
        return 1;
    }
    printf("Pipeline created (ID: %u)\n", g_context.pipeline_id);

    // Open output file for streaming audio
    g_context.output_file = fopen("streaming_output.raw", "wb");
    if (!g_context.output_file) {
        fprintf(stderr, "Warning: Could not open output file\n");
    }

    // Register callbacks for streaming
    printf("Registering streaming callbacks...\n");
    VoirsErrorCode callback_result = voirs_register_callbacks(
        g_context.pipeline_id,
        progress_callback,
        complete_callback,
        &g_context
    );
    
    if (callback_result != VOIRS_SUCCESS) {
        fprintf(stderr, "Failed to register callbacks: %s\n", voirs_get_last_error());
        if (g_context.output_file) fclose(g_context.output_file);
        voirs_destroy_pipeline(g_context.pipeline_id);
        return 1;
    }

    // Configure synthesis for streaming
    VoirsSynthesisConfig config = {
        .speed = 1.0f,
        .pitch = 1.0f,
        .volume = 0.8f,
        .voice_id = 0,
        .quality_level = 1  // Medium quality for better streaming performance
    };

    // Start streaming synthesis
    printf("Starting streaming synthesis...\n");
    printf("(Press Ctrl+C to cancel)\n\n");
    
    VoirsSynthesisResult* result = voirs_synthesize_streaming_advanced(text, &config);
    
    if (result == NULL) {
        fprintf(stderr, "Failed to start streaming synthesis: %s\n", voirs_get_last_error());
        voirs_unregister_callbacks(g_context.pipeline_id);
        if (g_context.output_file) fclose(g_context.output_file);
        voirs_destroy_pipeline(g_context.pipeline_id);
        return 1;
    }

    // Wait for synthesis to complete or be cancelled
    pthread_mutex_lock(&g_context.mutex);
    while (!g_context.synthesis_complete && !g_should_exit) {
        pthread_cond_wait(&g_context.cond, &g_context.mutex);
    }
    pthread_mutex_unlock(&g_context.mutex);

    printf("\n\nStreaming synthesis summary:\n");
    printf("  - Total chunks received: %d\n", g_context.total_chunks_received);
    printf("  - Status: %s\n", g_should_exit ? "Cancelled by user" : "Completed");
    
    if (result) {
        printf("  - Final quality score: %.2f\n", result->quality_score);
        printf("  - Total synthesis time: %u ms\n", result->synthesis_time_ms);
    }

    // Clean up
    printf("\nCleaning up...\n");
    
    voirs_unregister_callbacks(g_context.pipeline_id);
    
    if (result) {
        voirs_free_synthesis_result(result);
    }
    
    if (g_context.output_file) {
        fclose(g_context.output_file);
        printf("Audio saved to 'streaming_output.raw'\n");
        printf("Convert with: ffmpeg -f f32le -ar 22050 -ac 1 -i streaming_output.raw streaming_output.wav\n");
    }
    
    VoirsErrorCode destroy_result = voirs_destroy_pipeline(g_context.pipeline_id);
    if (destroy_result != VOIRS_SUCCESS) {
        fprintf(stderr, "Warning: Failed to destroy pipeline: %s\n", voirs_get_last_error());
    }

    pthread_mutex_destroy(&g_context.mutex);
    pthread_cond_destroy(&g_context.cond);

    printf("Streaming synthesis example completed!\n");
    return g_should_exit ? 130 : 0;  // Return 130 for SIGINT
}