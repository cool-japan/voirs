# macOS Integration Guide

VoiRS provides comprehensive macOS integration through Core Audio framework, AVFoundation support, Objective-C runtime bindings, and macOS-specific performance optimizations.

## Table of Contents

- [Core Audio Integration](#core-audio-integration)
- [AVFoundation Support](#avfoundation-support)
- [Objective-C Runtime](#objective-c-runtime)
- [Performance Optimization](#performance-optimization)
- [Build Configuration](#build-configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Core Audio Integration

VoiRS leverages Core Audio for high-performance, low-latency audio processing on macOS.

### Basic Core Audio Setup

```c
#include "voirs/platform/macos.h"

// Initialize Core Audio system
VoirsMacOSCoreAudio* core_audio = voirs_macos_init_core_audio();
if (!core_audio) {
    fprintf(stderr, "Failed to initialize Core Audio\n");
    return -1;
}

// Core Audio automatically cleaned up when destroyed
voirs_macos_destroy_core_audio(core_audio);
```

### Audio Device Management

```c
VoirsMacOSCoreAudio* core_audio = voirs_macos_init_core_audio();

// Get available audio devices
VoirsAudioDevice* devices;
int device_count = voirs_macos_get_audio_devices(core_audio, &devices);

for (int i = 0; i < device_count; i++) {
    printf("Device %d: %s\n", devices[i].id, devices[i].name);
    printf("  Sample Rate: %.0f Hz\n", devices[i].sample_rate);
    printf("  Channels: %u\n", devices[i].channels);
    printf("  %s\n", devices[i].is_input ? "Input" : "Output");
    printf("  %s\n", devices[i].is_default ? "Default" : "Non-default");
}

voirs_macos_free_device_list(devices, device_count);
```

### Audio Unit Configuration

```c
// Create and configure Audio Unit for real-time processing
VoirsCoreAudioUnit* audio_unit = voirs_macos_create_audio_unit();

// Set audio format
VoirsCoreAudioFormat format = {
    .sample_rate = 44100.0,
    .channels = 2,
    .bits_per_sample = 32,
    .format_flags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked
};

voirs_macos_set_audio_unit_format(audio_unit, &format);

// Set buffer size for low latency
voirs_macos_set_audio_unit_buffer_size(audio_unit, 256);

// Register audio processing callback
voirs_macos_set_audio_unit_callback(audio_unit, audio_render_callback, user_data);

// Start audio processing
voirs_macos_start_audio_unit(audio_unit);
```

### Audio Processing Callback

```c
OSStatus audio_render_callback(
    void* user_data,
    AudioUnitRenderActionFlags* flags,
    const AudioTimeStamp* timestamp,
    UInt32 bus_number,
    UInt32 frame_count,
    AudioBufferList* data
) {
    VoirsPipeline* pipeline = (VoirsPipeline*)user_data;
    
    // Generate audio samples
    float* output = (float*)data->mBuffers[0].mData;
    voirs_process_audio_realtime(pipeline, output, frame_count);
    
    return noErr;
}
```

### Sample Rate and Format Control

```c
// Set device sample rate
AudioDeviceID device_id = 1; // Default output device
voirs_macos_set_device_sample_rate(core_audio, device_id, 48000.0);

// Get supported sample rates
double* supported_rates;
int rate_count = voirs_macos_get_supported_sample_rates(core_audio, device_id, &supported_rates);

for (int i = 0; i < rate_count; i++) {
    printf("Supported rate: %.0f Hz\n", supported_rates[i]);
}

voirs_macos_free_sample_rates(supported_rates);
```

## AVFoundation Support

AVFoundation provides audio session management, permissions, and high-level audio features.

### Audio Session Configuration

```c
// Initialize AVFoundation
VoirsMacOSAVFoundation* av_foundation = voirs_macos_init_avfoundation();
if (!av_foundation) {
    fprintf(stderr, "Failed to initialize AVFoundation\n");
    return -1;
}

// Configure audio session for speech synthesis
voirs_macos_configure_synthesis_session(av_foundation);

// Set audio session category
voirs_macos_set_audio_session_category(av_foundation, VOIRS_AUDIO_CATEGORY_PLAYBACK);

// Enable background audio
voirs_macos_set_audio_session_mode(av_foundation, VOIRS_AUDIO_MODE_SPOKEN_AUDIO);
```

### Microphone Permissions

```c
// Check microphone permission status
VoirsPermissionStatus status = voirs_macos_get_microphone_permission_status(av_foundation);

switch (status) {
    case VOIRS_PERMISSION_GRANTED:
        printf("Microphone permission granted\n");
        break;
    case VOIRS_PERMISSION_DENIED:
        printf("Microphone permission denied\n");
        break;
    case VOIRS_PERMISSION_NOT_DETERMINED:
        printf("Microphone permission not determined, requesting...\n");
        // Request permission
        voirs_macos_request_microphone_permission(av_foundation, permission_callback, NULL);
        break;
}

void permission_callback(VoirsPermissionStatus status, void* user_data) {
    if (status == VOIRS_PERMISSION_GRANTED) {
        printf("Microphone permission granted by user\n");
        // Proceed with audio input functionality
    } else {
        printf("Microphone permission denied by user\n");
        // Disable audio input features
    }
}
```

### Audio Interruption Handling

```c
// Register for audio interruption notifications
void audio_interruption_callback(VoirsAudioInterruptionType type, void* user_data) {
    switch (type) {
        case VOIRS_AUDIO_INTERRUPTION_BEGAN:
            printf("Audio interrupted (phone call, etc.)\n");
            // Pause synthesis
            voirs_pipeline_pause(pipeline);
            break;
        case VOIRS_AUDIO_INTERRUPTION_ENDED:
            printf("Audio interruption ended\n");
            // Resume synthesis
            voirs_pipeline_resume(pipeline);
            break;
    }
}

voirs_macos_register_interruption_callback(av_foundation, audio_interruption_callback, NULL);
```

## Objective-C Runtime

Access native macOS APIs and system features through Objective-C runtime integration.

### System Information

```c
// Get system language preference
char* language = voirs_macos_get_system_language();
printf("System language: %s\n", language);
voirs_free_string(language);

// Get system appearance (light/dark mode)
char* appearance = voirs_macos_get_system_appearance();
printf("System appearance: %s\n", appearance);
voirs_free_string(appearance);

// Get system locale information
VoirsMacOSLocaleInfo locale;
voirs_macos_get_locale_info(&locale);
printf("Locale: %s\n", locale.identifier);
printf("Language Code: %s\n", locale.language_code);
printf("Country Code: %s\n", locale.country_code);
```

### Native Notifications

```c
// Show native macOS notification
voirs_macos_show_notification(
    "VoiRS", 
    "Speech synthesis completed", 
    "Your audio file has been generated successfully."
);

// Show notification with custom options
VoirsMacOSNotificationOptions options = {
    .title = "VoiRS Synthesis",
    .subtitle = "Processing Complete",
    .body = "Your speech synthesis is ready.",
    .sound_name = "default",
    .has_action_button = true,
    .action_button_title = "Open File"
};

voirs_macos_show_notification_with_options(&options, notification_callback, NULL);

void notification_callback(VoirsMacOSNotificationResponse response, void* user_data) {
    if (response == VOIRS_NOTIFICATION_ACTION_CLICKED) {
        printf("User clicked notification action\n");
        // Open the generated file
    }
}
```

### File System Integration

```c
// Get system directories
char* documents_dir = voirs_macos_get_documents_directory();
char* downloads_dir = voirs_macos_get_downloads_directory();
char* temp_dir = voirs_macos_get_temporary_directory();

printf("Documents: %s\n", documents_dir);
printf("Downloads: %s\n", downloads_dir);
printf("Temporary: %s\n", temp_dir);

voirs_free_string(documents_dir);
voirs_free_string(downloads_dir);
voirs_free_string(temp_dir);

// File access permissions
bool has_documents_access = voirs_macos_check_documents_access();
bool has_downloads_access = voirs_macos_check_downloads_access();

if (!has_documents_access) {
    printf("No documents access, requesting permission...\n");
    voirs_macos_request_documents_access();
}
```

## Performance Optimization

macOS-specific performance optimizations for real-time audio processing.

### Real-time Thread Configuration

```c
// Enable low-latency audio mode
voirs_macos_enable_low_latency_mode();

// Set real-time thread priorities
voirs_macos_set_audio_thread_priority(VOIRS_MACOS_THREAD_PRIORITY_REALTIME);

// Configure thread time constraints
VoirsMacOSThreadConstraints constraints = {
    .period = 2902,        // ~128 samples at 44.1kHz
    .computation = 2177,   // 75% of period
    .constraint = 2902,    // Same as period
    .preemptible = true
};

voirs_macos_set_thread_constraints(&constraints);
```

### Memory Optimization

```c
// Configure memory for optimal performance
voirs_macos_configure_vm_pressure();

// Enable large page support (if available)
voirs_macos_enable_large_pages();

// Set memory pressure handling
void memory_pressure_callback(VoirsMacOSMemoryPressure pressure, void* user_data) {
    switch (pressure) {
        case VOIRS_MEMORY_PRESSURE_NORMAL:
            // Normal operation
            break;
        case VOIRS_MEMORY_PRESSURE_WARNING:
            printf("Memory pressure warning\n");
            // Reduce memory usage
            voirs_pipeline_reduce_memory_usage(pipeline);
            break;
        case VOIRS_MEMORY_PRESSURE_CRITICAL:
            printf("Critical memory pressure\n");
            // Aggressively reduce memory usage
            voirs_pipeline_emergency_memory_reduction(pipeline);
            break;
    }
}

voirs_macos_register_memory_pressure_callback(memory_pressure_callback, NULL);
```

### Core Animation Integration

```c
// Enable Core Animation for UI updates
void synthesis_progress_callback(float progress, void* user_data) {
    // Update UI on main thread
    dispatch_async(dispatch_get_main_queue(), ^{
        // Update progress indicator
        voirs_macos_update_progress_indicator(progress);
    });
}

voirs_set_synthesis_progress_callback(pipeline, synthesis_progress_callback, NULL);
```

### Metal Performance Shaders

```c
// Enable Metal acceleration (if available)
if (voirs_macos_supports_metal()) {
    printf("Metal Performance Shaders available\n");
    voirs_macos_enable_metal_acceleration();
    
    // Configure Metal device
    VoirsMacOSMetalConfig metal_config = {
        .device_type = VOIRS_METAL_DEVICE_TYPE_DISCRETE,  // Prefer discrete GPU
        .memory_mode = VOIRS_METAL_MEMORY_MODE_SHARED,
        .enable_profiling = false
    };
    
    voirs_macos_configure_metal(&metal_config);
}
```

## Build Configuration

### Xcode Project Configuration

```xml
<!-- Info.plist entries -->
<key>NSMicrophoneUsageDescription</key>
<string>VoiRS needs microphone access for voice input features</string>

<key>NSDocumentsFolderUsageDescription</key>
<string>VoiRS saves audio files to your Documents folder</string>

<key>NSDownloadsFolderUsageDescription</key>
<string>VoiRS saves downloaded voice models to your Downloads folder</string>

<!-- Hardened Runtime entitlements -->
<key>com.apple.security.device.audio-input</key>
<true/>
<key>com.apple.security.files.user-selected.read-write</key>
<true/>
<key>com.apple.security.files.downloads.read-write</key>
<true/>
```

### CMake Configuration

```cmake
# macOS-specific configuration
if(APPLE)
    find_library(CORE_AUDIO_FRAMEWORK CoreAudio)
    find_library(AUDIO_UNIT_FRAMEWORK AudioUnit)
    find_library(AVFOUNDATION_FRAMEWORK AVFoundation)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(METAL_FRAMEWORK Metal)
    
    target_link_libraries(your_app 
        voirs_ffi
        ${CORE_AUDIO_FRAMEWORK}
        ${AUDIO_UNIT_FRAMEWORK}
        ${AVFOUNDATION_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${METAL_FRAMEWORK}
    )
    
    # macOS version targeting
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
    
    # Enable features
    target_compile_definitions(your_app PRIVATE 
        VOIRS_ENABLE_CORE_AUDIO=1
        VOIRS_ENABLE_AVFOUNDATION=1
        VOIRS_ENABLE_METAL=1
    )
endif()
```

### Compiler Flags

```bash
# Compile with Core Audio support
gcc -o example example.c \
    -lvoirs_ffi \
    -framework CoreAudio \
    -framework AudioUnit \
    -framework AVFoundation \
    -framework Foundation \
    -framework Metal \
    -mmacosx-version-min=10.15
```

## Examples

### Complete macOS Integration Example

```c
#include "voirs/voirs_ffi.h"
#include "voirs/platform/macos.h"

int main() {
    // Initialize macOS-specific features
    if (!voirs_macos_initialize()) {
        fprintf(stderr, "Failed to initialize macOS features\n");
        return 1;
    }
    
    // Initialize Core Audio
    VoirsMacOSCoreAudio* core_audio = voirs_macos_init_core_audio();
    if (!core_audio) {
        fprintf(stderr, "Failed to initialize Core Audio\n");
        return 1;
    }
    
    // Initialize AVFoundation
    VoirsMacOSAVFoundation* av_foundation = voirs_macos_init_avfoundation();
    if (!av_foundation) {
        fprintf(stderr, "Failed to initialize AVFoundation\n");
        voirs_macos_destroy_core_audio(core_audio);
        return 1;
    }
    
    // Configure for optimal macOS performance
    VoirsAudioConfig config = {
        .backend = "coreaudio",
        .sample_rate = 44100,
        .buffer_size = 256,
        .channels = 2,
        .use_exclusive_mode = false  // Core Audio manages this
    };
    
    // Create pipeline with macOS optimizations
    VoirsPipeline* pipeline = voirs_create_pipeline_with_config(&config);
    
    // Enable macOS-specific optimizations
    voirs_macos_enable_low_latency_mode();
    voirs_macos_configure_vm_pressure();
    
    // Configure audio session
    voirs_macos_configure_synthesis_session(av_foundation);
    
    // Synthesize speech
    const char* text = "Hello from VoiRS on macOS!";
    VoirsAudioBuffer* audio = voirs_synthesize(pipeline, text);
    
    if (audio) {
        // Save using Core Audio services
        voirs_macos_save_audio_core_audio(audio, "output.m4a", VOIRS_CORE_AUDIO_FORMAT_AAC);
        
        // Show completion notification
        voirs_macos_show_notification("VoiRS", "Synthesis Complete", "Audio saved successfully");
        
        voirs_destroy_audio_buffer(audio);
    }
    
    // Clean up
    voirs_destroy_pipeline(pipeline);
    voirs_macos_destroy_avfoundation(av_foundation);
    voirs_macos_destroy_core_audio(core_audio);
    voirs_macos_cleanup();
    
    return 0;
}
```

### Real-time Audio Processing with Core Audio

```c
// Set up real-time audio processing
void setup_realtime_processing() {
    // Create Audio Unit
    VoirsCoreAudioUnit* audio_unit = voirs_macos_create_audio_unit();
    
    // Configure for low latency
    voirs_macos_set_audio_unit_buffer_size(audio_unit, 128);
    voirs_macos_enable_low_latency_mode();
    
    // Set up real-time thread
    pthread_t audio_thread;
    pthread_create(&audio_thread, NULL, audio_thread_func, audio_unit);
    
    // Set thread priority
    struct sched_param param;
    param.sched_priority = 63;  // High priority
    pthread_setschedparam(audio_thread, SCHED_FIFO, &param);
}

void* audio_thread_func(void* arg) {
    VoirsCoreAudioUnit* audio_unit = (VoirsCoreAudioUnit*)arg;
    
    // Set thread time constraints for real-time processing
    VoirsMacOSThreadConstraints constraints = {
        .period = 2902,        // ~128 samples at 44.1kHz
        .computation = 2177,   // 75% of period
        .constraint = 2902,
        .preemptible = true
    };
    
    voirs_macos_set_thread_constraints(&constraints);
    
    // Start audio processing
    voirs_macos_start_audio_unit(audio_unit);
    
    return NULL;
}
```

## Troubleshooting

### Common Issues

#### Core Audio Initialization Failed
```c
// Check Core Audio availability
if (!voirs_macos_check_core_audio_availability()) {
    printf("Core Audio not available\n");
    // Fall back to alternative audio backend
}

// Verify audio hardware
VoirsCoreAudioDeviceInfo info;
if (voirs_macos_get_default_device_info(&info)) {
    printf("Default device: %s\n", info.name);
    printf("Sample rate: %.0f Hz\n", info.sample_rate);
} else {
    printf("No audio devices available\n");
}
```

#### Permission Issues
```c
// Check and request permissions
if (!voirs_macos_has_microphone_permission()) {
    printf("Requesting microphone permission...\n");
    voirs_macos_request_microphone_permission(av_foundation, NULL, NULL);
}

if (!voirs_macos_has_documents_access()) {
    printf("Requesting documents folder access...\n");
    voirs_macos_request_documents_access();
}
```

#### Audio Unit Problems
```c
// Debug Audio Unit issues
OSStatus status = voirs_macos_get_audio_unit_status(audio_unit);
if (status != noErr) {
    printf("Audio Unit error: %d\n", (int)status);
    
    // Get detailed error information
    char error_string[256];
    voirs_macos_get_audio_unit_error_string(status, error_string, sizeof(error_string));
    printf("Error details: %s\n", error_string);
}
```

### Debug Tools

```c
// Enable macOS-specific debugging
voirs_macos_enable_debug_mode();
voirs_macos_set_debug_level(VOIRS_DEBUG_VERBOSE);

// Core Audio debugging
voirs_macos_enable_core_audio_debugging();
voirs_macos_log_audio_devices();
voirs_macos_log_audio_unit_graph();

// Performance monitoring
VoirsMacOSPerformanceMonitor* monitor = voirs_macos_create_performance_monitor();
voirs_macos_start_monitoring(monitor);

// ... perform operations ...

VoirsMacOSPerformanceReport report;
voirs_macos_get_performance_report(monitor, &report);
printf("Average latency: %.2f ms\n", report.average_latency_ms);
printf("CPU usage: %.2f%%\n", report.cpu_usage);
printf("Memory pressure events: %u\n", report.memory_pressure_events);

voirs_macos_destroy_performance_monitor(monitor);
```

### Performance Tuning

```c
// Optimize for specific scenarios
typedef enum {
    VOIRS_MACOS_PROFILE_REALTIME,     // Lowest latency
    VOIRS_MACOS_PROFILE_BALANCED,     // Good latency/quality balance
    VOIRS_MACOS_PROFILE_QUALITY       // Highest quality
} VoirsMacOSPerformanceProfile;

voirs_macos_set_performance_profile(VOIRS_MACOS_PROFILE_REALTIME);

// Fine-tune based on system capabilities
VoirsMacOSSystemCapabilities caps;
voirs_macos_get_system_capabilities(&caps);

if (caps.has_discrete_gpu) {
    voirs_macos_enable_metal_acceleration();
}

if (caps.memory_gb >= 16) {
    voirs_macos_enable_large_buffers();
}

if (caps.cpu_cores >= 8) {
    voirs_macos_enable_parallel_processing();
}
```

## Security Considerations

- Microphone access requires user permission and usage description
- File system access requires appropriate sandbox entitlements
- Code signing required for distribution outside Mac App Store
- Hardened Runtime may affect some advanced features

## Version Compatibility

VoiRS macOS integration supports:
- macOS 10.15 (Catalina) and later (recommended)
- macOS 11.0 (Big Sur) and later for full Metal support
- Limited support for macOS 10.14 (without some AVFoundation features)

## Related Documentation

- [C API Reference](api_reference.md)
- [Memory Management](memory_management.md)
- [Threading Guide](threading.md)
- [Performance Optimization](performance.md)