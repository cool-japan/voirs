# Windows Integration Guide

VoiRS provides comprehensive Windows integration through COM interfaces, WASAPI support, Registry configuration, and Windows-specific performance optimizations.

## Table of Contents

- [COM Integration](#com-integration)
- [WASAPI Audio Support](#wasapi-audio-support)
- [Registry Configuration](#registry-configuration)
- [Performance Monitoring](#performance-monitoring)
- [Build Configuration](#build-configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## COM Integration

VoiRS integrates with Windows Component Object Model (COM) for advanced audio device management.

### Basic COM Setup

```c
#include "voirs/platform/windows.h"

// Initialize COM for Windows integration
VoirsComManager* com_manager = voirs_windows_init_com();
if (!com_manager) {
    fprintf(stderr, "Failed to initialize COM\n");
    return -1;
}

// COM is automatically cleaned up when manager is destroyed
voirs_windows_destroy_com(com_manager);
```

### Audio Device Enumeration

```c
// Create Windows Audio Session manager
VoirsWindowsAudioSession* session = voirs_windows_init_audio_session();
if (!session) {
    fprintf(stderr, "Failed to initialize Windows Audio Session\n");
    return -1;
}

// Get available audio devices
char** devices;
int device_count = voirs_windows_get_audio_devices(session, &devices);

for (int i = 0; i < device_count; i++) {
    printf("Device %d: %s\n", i, devices[i]);
}

// Clean up
voirs_windows_free_device_list(devices, device_count);
voirs_windows_destroy_audio_session(session);
```

## WASAPI Audio Support

Windows Audio Session API (WASAPI) provides low-latency audio I/O capabilities.

### Basic WASAPI Usage

```c
// Configure for low-latency audio
VoirsAudioConfig config = voirs_get_optimal_audio_config();
config.use_exclusive_mode = true;  // Enable exclusive mode for lower latency
config.buffer_size = 128;          // Smaller buffer for lower latency

VoirsPipeline* pipeline = voirs_create_pipeline_with_config(&config);

// Set WASAPI-specific options
voirs_windows_set_audio_session_guid(pipeline, "{12345678-1234-1234-1234-123456789ABC}");
voirs_windows_enable_audio_ducking(pipeline, true);  // Duck other audio when synthesizing
```

### Volume Control

```c
VoirsWindowsAudioSession* session = voirs_windows_init_audio_session();

// Get current system volume (0.0 to 1.0)
float volume = voirs_windows_get_system_volume(session);
printf("Current volume: %.0f%%\n", volume * 100);

// Set system volume
voirs_windows_set_system_volume(session, 0.75f);  // 75% volume

voirs_windows_destroy_audio_session(session);
```

### Audio Session Management

```c
// Register for audio session notifications
void audio_session_callback(VoirsAudioSessionEvent event, void* user_data) {
    switch (event) {
        case VOIRS_AUDIO_SESSION_DISCONNECTED:
            printf("Audio device disconnected\n");
            break;
        case VOIRS_AUDIO_SESSION_FORMAT_CHANGED:
            printf("Audio format changed\n");
            break;
        case VOIRS_AUDIO_SESSION_VOLUME_CHANGED:
            printf("Volume changed\n");
            break;
    }
}

voirs_windows_register_session_callback(session, audio_session_callback, NULL);
```

## Registry Configuration

VoiRS can store and retrieve configuration from the Windows Registry.

### Reading Configuration

```c
// Read VoiRS configuration from Registry
char* voice_model = voirs_windows_read_registry_config("DefaultVoiceModel");
if (voice_model) {
    printf("Default voice model: %s\n", voice_model);
    voirs_free_string(voice_model);
}

// Read with default value
char* quality_setting = voirs_windows_read_registry_config_with_default(
    "QualityLevel", "high"
);
printf("Quality level: %s\n", quality_setting);
voirs_free_string(quality_setting);
```

### Writing Configuration

```c
// Write configuration to Registry
bool success = voirs_windows_write_registry_config("DefaultVoiceModel", "neural_voice_v2");
if (!success) {
    fprintf(stderr, "Failed to write to registry\n");
}

// Write numeric configuration
voirs_windows_write_registry_config_int("BufferSize", 512);
voirs_windows_write_registry_config_float("DefaultSpeed", 1.0f);
```

### Registry Paths

VoiRS uses the following registry locations:
- User settings: `HKEY_CURRENT_USER\SOFTWARE\VoiRS\Config`
- System settings: `HKEY_LOCAL_MACHINE\SOFTWARE\VoiRS\Config`
- Voice models: `HKEY_CURRENT_USER\SOFTWARE\VoiRS\VoiceModels`

## Performance Monitoring

Windows-specific performance monitoring using Performance Counters and system APIs.

### System Performance Metrics

```c
VoirsWindowsMetrics metrics;
VoirsErrorCode result = voirs_windows_get_performance_metrics(&metrics);

if (result == VOIRS_SUCCESS) {
    printf("CPU Usage: %.2f%%\n", metrics.cpu_usage);
    printf("Memory Usage: %.2f%%\n", metrics.memory_usage);
    printf("Audio Latency: %.2f ms\n", metrics.audio_latency_ms);
    printf("Audio Dropouts: %u\n", metrics.audio_dropouts);
    printf("COM Objects Active: %u\n", metrics.com_objects_active);
}
```

### Performance Optimization

```c
// Enable Windows-specific optimizations
voirs_windows_enable_mmcss();          // Multimedia Class Scheduler Service
voirs_windows_set_timer_resolution(1); // 1ms timer resolution
voirs_windows_enable_large_pages();    // Large page support (requires privilege)

// Set process priority for audio processing
voirs_windows_set_process_priority(VOIRS_PRIORITY_HIGH);

// Configure thread priorities
voirs_windows_set_audio_thread_priority(VOIRS_THREAD_PRIORITY_TIME_CRITICAL);
```

### Memory Management

```c
// Windows-specific memory optimizations
voirs_windows_enable_heap_optimization();
voirs_windows_set_working_set_size(64 * 1024 * 1024, 128 * 1024 * 1024); // 64-128 MB

// Use Windows memory pools
VoirsWindowsMemoryPool* pool = voirs_windows_create_memory_pool(
    65536,      // chunk size
    10,         // initial chunks
    100,        // max chunks
    true        // use large pages
);
```

## Build Configuration

### CMake Configuration

```cmake
# Windows-specific configuration
if(WIN32)
    target_link_libraries(your_app voirs_ffi ole32 oleaut32 winmm)
    
    # For COM support
    target_compile_definitions(your_app PRIVATE VOIRS_ENABLE_COM=1)
    
    # For WASAPI support
    target_compile_definitions(your_app PRIVATE VOIRS_ENABLE_WASAPI=1)
    
    # Windows version targeting
    target_compile_definitions(your_app PRIVATE 
        WINVER=0x0A00 
        _WIN32_WINNT=0x0A00
    )
endif()
```

### MSVC Project Configuration

```xml
<!-- .vcxproj settings -->
<PropertyGroup>
    <TargetPlatformVersion>10.0</TargetPlatformVersion>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
</PropertyGroup>

<ItemDefinitionGroup>
    <ClCompile>
        <PreprocessorDefinitions>
            VOIRS_ENABLE_COM=1;
            VOIRS_ENABLE_WASAPI=1;
            %(PreprocessorDefinitions)
        </PreprocessorDefinitions>
    </ClCompile>
    <Link>
        <AdditionalDependencies>
            voirs_ffi.lib;
            ole32.lib;
            oleaut32.lib;
            winmm.lib;
            %(AdditionalDependencies)
        </AdditionalDependencies>
    </Link>
</ItemDefinitionGroup>
```

## Examples

### Complete Windows Integration Example

```c
#include "voirs/voirs_ffi.h"
#include "voirs/platform/windows.h"

int main() {
    // Initialize Windows-specific features
    if (!voirs_windows_initialize()) {
        fprintf(stderr, "Failed to initialize Windows features\n");
        return 1;
    }
    
    // Create COM manager
    VoirsComManager* com = voirs_windows_init_com();
    if (!com) {
        fprintf(stderr, "Failed to initialize COM\n");
        return 1;
    }
    
    // Create audio session
    VoirsWindowsAudioSession* session = voirs_windows_init_audio_session();
    if (!session) {
        fprintf(stderr, "Failed to initialize audio session\n");
        voirs_windows_destroy_com(com);
        return 1;
    }
    
    // Configure for optimal Windows performance
    VoirsAudioConfig config = {
        .backend = "wasapi",
        .sample_rate = 48000,
        .buffer_size = 256,
        .channels = 2,
        .use_exclusive_mode = true
    };
    
    // Create pipeline with Windows optimizations
    VoirsPipeline* pipeline = voirs_create_pipeline_with_config(&config);
    
    // Enable Windows-specific optimizations
    voirs_windows_enable_mmcss();
    voirs_windows_set_timer_resolution(1);
    
    // Synthesize speech
    const char* text = "Hello from VoiRS on Windows!";
    VoirsAudioBuffer* audio = voirs_synthesize(pipeline, text);
    
    if (audio) {
        // Save using Windows Media Foundation (if available)
        voirs_windows_save_audio_wmf(audio, "output.wav", VOIRS_WMF_FORMAT_WAV);
        voirs_destroy_audio_buffer(audio);
    }
    
    // Clean up
    voirs_destroy_pipeline(pipeline);
    voirs_windows_destroy_audio_session(session);
    voirs_windows_destroy_com(com);
    voirs_windows_cleanup();
    
    return 0;
}
```

### Real-time Audio Processing

```c
// Windows-specific real-time audio processing
void setup_realtime_audio() {
    // Set multimedia thread characteristics
    DWORD task_index;
    HANDLE mmcss_handle = AvSetMmThreadCharacteristics(L"Audio", &task_index);
    
    // Set thread priority
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    
    // Create WASAPI client
    VoirsWasapiClient* client = voirs_windows_create_wasapi_client();
    voirs_windows_set_exclusive_mode(client, true);
    voirs_windows_set_buffer_size(client, 128);
    
    // Start real-time processing
    voirs_windows_start_audio_processing(client, audio_callback, NULL);
}

void audio_callback(float* output, int frames, void* user_data) {
    // Process audio in real-time
    VoirsPipeline* pipeline = (VoirsPipeline*)user_data;
    voirs_process_audio_realtime(pipeline, output, frames);
}
```

## Troubleshooting

### Common Issues

#### COM Initialization Failed
```c
// Check if COM is already initialized
HRESULT hr = CoInitialize(NULL);
if (hr == RPC_E_CHANGED_MODE) {
    // COM already initialized in different mode
    printf("COM already initialized\n");
} else if (FAILED(hr)) {
    printf("COM initialization failed: 0x%x\n", hr);
}
```

#### Audio Device Access Issues
```c
// Check audio device permissions
if (!voirs_windows_check_audio_permissions()) {
    printf("Audio permissions not granted\n");
    // Guide user to grant permissions
}

// Verify exclusive mode support
if (!voirs_windows_supports_exclusive_mode()) {
    printf("Exclusive mode not supported, using shared mode\n");
    config.use_exclusive_mode = false;
}
```

#### Registry Access Problems
```c
// Check registry permissions
if (!voirs_windows_check_registry_access()) {
    printf("Registry access denied, using default configuration\n");
    // Fall back to file-based configuration
}
```

### Debug Tools

```c
// Enable Windows-specific debugging
voirs_windows_enable_debug_mode();
voirs_windows_set_debug_output_console();

// Monitor COM object lifetime
voirs_windows_enable_com_debugging();

// Audio session debugging
voirs_windows_enable_audio_debugging();
voirs_windows_log_audio_devices();
```

### Performance Tuning

```c
// Measure and optimize performance
VoirsWindowsPerformanceProfiler* profiler = voirs_windows_create_profiler();

voirs_windows_start_profiling(profiler);
// ... perform audio operations ...
voirs_windows_stop_profiling(profiler);

VoirsWindowsPerformanceReport report;
voirs_windows_get_performance_report(profiler, &report);

printf("Average latency: %.2f ms\n", report.average_latency_ms);
printf("Max latency: %.2f ms\n", report.max_latency_ms);
printf("Buffer underruns: %u\n", report.buffer_underruns);

voirs_windows_destroy_profiler(profiler);
```

## Security Considerations

- COM initialization requires appropriate security contexts
- Registry access may require elevated privileges for system-wide settings
- Exclusive mode audio requires appropriate process privileges
- Large page support requires `SeLockMemoryPrivilege`

## Version Compatibility

VoiRS Windows integration supports:
- Windows 10 version 1903 and later (recommended)
- Windows Server 2019 and later
- Limited support for Windows 8.1 (without exclusive mode features)

## Related Documentation

- [C API Reference](api_reference.md)
- [Memory Management](memory_management.md)
- [Threading Guide](threading.md)
- [Performance Optimization](performance.md)