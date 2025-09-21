# VoiRS C API Documentation

This directory contains comprehensive documentation for the VoiRS C API, including platform-specific integrations for Windows, macOS, and Linux.

## Documentation Structure

### Core References
- [**API Reference**](api_reference.md) - Complete C API function documentation
- [**Data Types**](data_types.md) - Structures, enums, and type definitions
- [**Quick Start Guide**](quick_start.md) - Getting started with the C API
- [**Error Handling**](error_handling.md) - Error codes and debugging

### Platform-Specific Documentation
- [**Windows Integration**](windows.md) - COM, WASAPI, and Windows-specific features
- [**macOS Integration**](macos.md) - Core Audio, AVFoundation, and Objective-C bindings
- [**Linux Integration**](linux.md) - PulseAudio, ALSA, D-Bus, and SystemD integration

### Advanced Topics
- [**Memory Management**](memory_management.md) - Memory pools, allocation strategies, and optimization
- [**Threading**](threading.md) - Multi-threading, real-time audio processing
- [**Performance Optimization**](performance.md) - Platform-specific optimizations and benchmarking

### Examples and Tutorials
- [**Basic Examples**](examples/) - Simple usage examples
- [**Platform Examples**](platform_examples/) - Platform-specific code samples
- [**Integration Examples**](integration_examples/) - Real-world integration scenarios

## Getting Started

### Basic Usage

```c
#include "voirs/voirs_ffi.h"

int main() {
    // Initialize VoiRS
    VoirsPipeline* pipeline = voirs_create_pipeline(NULL);
    if (!pipeline) {
        fprintf(stderr, "Failed to create pipeline\n");
        return 1;
    }
    
    // Synthesize speech
    const char* text = "Hello, world!";
    VoirsAudioBuffer* audio = voirs_synthesize(pipeline, text);
    
    if (audio) {
        // Save to file
        voirs_save_audio(audio, "output.wav", VOIRS_FORMAT_WAV);
        
        // Clean up
        voirs_destroy_audio_buffer(audio);
    }
    
    voirs_destroy_pipeline(pipeline);
    return 0;
}
```

### Compilation

#### Linux/macOS
```bash
gcc -o example example.c -lvoirs_ffi -lpthread
```

#### Windows (MSVC)
```cmd
cl example.c voirs_ffi.lib
```

#### Windows (MinGW)
```bash
gcc -o example.exe example.c -lvoirs_ffi -lws2_32 -lole32
```

## Platform-Specific Features

### Windows
- **COM Integration**: Access Windows audio system through COM interfaces
- **WASAPI Support**: Low-latency audio input/output
- **Registry Configuration**: Store and retrieve settings from Windows Registry
- **Performance Monitoring**: Windows Performance Counters integration

### macOS
- **Core Audio Integration**: Native macOS audio framework support
- **AVFoundation Support**: Audio session management and permissions
- **Objective-C Runtime**: Access to native macOS APIs
- **Metal Performance Shaders**: Hardware-accelerated audio processing

### Linux
- **PulseAudio Integration**: Modern Linux audio server support
- **ALSA Support**: Low-level ALSA (Advanced Linux Sound Architecture) access
- **D-Bus Integration**: System service communication
- **SystemD Integration**: Service management and system integration
- **Real-time Scheduling**: RT priority for audio threads

## Advanced Memory Management

VoiRS provides sophisticated memory management features:

### Memory Pools
```c
// Create a memory pool for audio buffers
VoirsMemoryPool* pool = voirs_create_memory_pool(65536, 10, 50);

// Allocate from pool
void* buffer = voirs_pool_allocate(pool, 4096);

// Return to pool
voirs_pool_deallocate(pool, buffer, 4096);

// Destroy pool
voirs_destroy_memory_pool(pool);
```

### Lock-Free Audio Ring
```c
// Create lock-free ring buffer for real-time audio
VoirsLockFreeRing* ring = voirs_create_lockfree_audio_ring(8, 4096);

// Producer thread
VoirsAudioBuffer* write_buffer = voirs_ring_get_write_buffer(ring);
if (write_buffer) {
    // Fill buffer with audio data
    voirs_ring_commit_write_buffer(ring);
}

// Consumer thread  
VoirsAudioBuffer* read_buffer = voirs_ring_get_read_buffer(ring);
if (read_buffer) {
    // Process audio data
    voirs_ring_commit_read_buffer(ring);
}

voirs_destroy_lockfree_audio_ring(ring);
```

### Adaptive Memory Allocation
```c
// Create adaptive allocator that optimizes strategy based on usage
VoirsAdaptiveAllocator* allocator = voirs_create_adaptive_allocator();

// Allocator automatically chooses best strategy
void* ptr = voirs_adaptive_allocate(allocator, 1024);

voirs_adaptive_deallocate(allocator, ptr, 1024);
voirs_destroy_adaptive_allocator(allocator);
```

## Real-Time Audio Processing

### Lock-Free Threading
```c
// Set up real-time audio processing
voirs_enable_realtime_mode();

// Create lock-free structures for audio pipeline
VoirsLockFreeQueue* queue = voirs_create_lockfree_queue(1024);

// Audio callback (runs in real-time thread)
void audio_callback(float* output, int frames) {
    VoirsAudioChunk chunk;
    if (voirs_queue_try_pop(queue, &chunk)) {
        memcpy(output, chunk.data, frames * sizeof(float));
    }
}
```

### Platform-Specific Optimizations
```c
// Enable platform-specific optimizations
#ifdef _WIN32
    voirs_windows_enable_audio_session();
#elif __APPLE__
    voirs_macos_enable_low_latency_mode();
#elif __linux__
    voirs_linux_enable_rt_scheduling();
#endif
```

## Error Handling

VoiRS provides comprehensive error reporting:

```c
VoirsErrorCode result = voirs_synthesize_to_file(pipeline, text, "output.wav");

switch (result) {
    case VOIRS_SUCCESS:
        printf("Synthesis completed successfully\n");
        break;
    case VOIRS_ERROR_INVALID_PARAMETER:
        fprintf(stderr, "Invalid parameter provided\n");
        break;
    case VOIRS_ERROR_OUT_OF_MEMORY:
        fprintf(stderr, "Out of memory\n");
        break;
    default:
        fprintf(stderr, "Unknown error: %d\n", result);
}

// Get detailed error message
const char* error_msg = voirs_get_last_error_message();
if (error_msg) {
    fprintf(stderr, "Error details: %s\n", error_msg);
}
```

## Performance Monitoring

```c
// Get performance metrics
VoirsPerformanceMetrics metrics;
voirs_get_performance_metrics(&metrics);

printf("CPU Usage: %.2f%%\n", metrics.cpu_usage);
printf("Memory Usage: %zu MB\n", metrics.memory_usage_mb);
printf("Audio Latency: %.2f ms\n", metrics.audio_latency_ms);

// Platform-specific metrics
#ifdef _WIN32
VoirsWindowsMetrics win_metrics;
voirs_windows_get_metrics(&win_metrics);
printf("COM Objects Active: %u\n", win_metrics.com_objects_active);
#endif
```

## Thread Safety

All VoiRS C API functions are thread-safe unless otherwise noted. For optimal performance in multi-threaded applications:

```c
// Create pipeline per thread for best performance
VoirsPipeline* pipeline = voirs_create_pipeline(NULL);

// Or use thread-safe synthesis with shared pipeline
voirs_synthesize_threadsafe(shared_pipeline, text, callback, user_data);
```

## Best Practices

1. **Memory Management**: Always match create/destroy calls
2. **Error Checking**: Check return values and error codes
3. **Resource Cleanup**: Use proper cleanup in error paths
4. **Thread Safety**: Use appropriate synchronization for shared resources
5. **Platform Optimization**: Enable platform-specific features when available

## Troubleshooting

### Common Issues

1. **Linking Errors**: Ensure all required libraries are linked
2. **Audio Device Issues**: Check audio device permissions and availability
3. **Memory Leaks**: Use memory debugging tools and proper cleanup
4. **Performance Issues**: Enable platform optimizations and use appropriate buffer sizes

### Debug Mode

```c
// Enable debug mode for detailed logging
voirs_set_debug_mode(true);
voirs_set_log_level(VOIRS_LOG_DEBUG);

// Memory debugging
voirs_enable_memory_debugging();
VoirsMemoryStats stats;
voirs_get_memory_stats(&stats);
```

## Examples

See the [examples/](examples/) directory for complete working examples:

- `basic_synthesis.c` - Simple text-to-speech
- `streaming_synthesis.c` - Real-time streaming synthesis  
- `platform_integration.c` - Platform-specific features
- `memory_optimization.c` - Advanced memory management
- `threading_example.c` - Multi-threaded synthesis

## Support

For issues and questions:
- GitHub Issues: [VoiRS Issues](https://github.com/voirs/voirs/issues)
- Documentation: [VoiRS Docs](https://docs.voirs.com)
- Community: [VoiRS Discord](https://discord.gg/voirs)