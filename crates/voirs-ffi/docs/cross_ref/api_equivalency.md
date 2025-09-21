# API Equivalency Tables

This document provides cross-reference tables showing how to perform the same operations across different language bindings for VoiRS FFI.

## Basic Operations

### Pipeline Creation

| Language | Code |
|----------|------|
| **C** | `voirs_pipeline_t* pipeline = voirs_pipeline_create(&config);` |
| **Python** | `pipeline = voirs_ffi.Pipeline(config)` |
| **Node.js** | `const pipeline = new VoirsFFI.Pipeline(config);` |

### Audio Synthesis

| Language | Code |
|----------|------|
| **C** | `voirs_synthesis_result_t* result = voirs_synthesize(pipeline, text, config);` |
| **Python** | `result = pipeline.synthesize(text, config)` |
| **Node.js** | `const result = await pipeline.synthesize(text, config);` |

### Memory Management

| Language | Code |
|----------|------|
| **C** | `voirs_pipeline_destroy(pipeline);` |
| **Python** | `del pipeline  # Automatic via __del__` |
| **Node.js** | `pipeline.dispose();  // Manual cleanup` |

## Configuration

### Basic Configuration

| Language | Code |
|----------|------|
| **C** | ```c
voirs_synthesis_config_t config = {0};
config.quality = VOIRS_QUALITY_HIGH;
config.speed = 1.0f;
config.output_format = VOIRS_FORMAT_WAV;
``` |
| **Python** | ```python
config = voirs_ffi.SynthesisConfig(
    quality=voirs_ffi.Quality.HIGH,
    speed=1.0,
    output_format=voirs_ffi.Format.WAV
)
``` |
| **Node.js** | ```javascript
const config = {
    quality: VoirsFFI.Quality.HIGH,
    speed: 1.0,
    outputFormat: VoirsFFI.Format.WAV
};
``` |

### Performance Configuration

| Language | Code |
|----------|------|
| **C** | ```c
voirs_performance_config_t perf = {0};
perf.thread_count = 4;
perf.use_simd = 1;
perf.cache_size = 1024 * 1024;
``` |
| **Python** | ```python
perf = voirs_ffi.PerformanceConfig(
    thread_count=4,
    use_simd=True,
    cache_size=1024*1024
)
``` |
| **Node.js** | ```javascript
const perf = {
    threadCount: 4,
    useSIMD: true,
    cacheSize: 1024 * 1024
};
``` |

## Audio Processing

### Format Conversion

| Language | Code |
|----------|------|
| **C** | ```c
voirs_audio_result_t* converted = voirs_convert_format(
    audio_data, input_format, output_format
);
``` |
| **Python** | ```python
converted = voirs_ffi.convert_format(
    audio_data, input_format, output_format
)
``` |
| **Node.js** | ```javascript
const converted = VoirsFFI.convertFormat(
    audioData, inputFormat, outputFormat
);
``` |

### Audio Effects

| Language | Code |
|----------|------|
| **C** | ```c
voirs_audio_effects_config_t effects = {0};
effects.reverb = 0.3f;
effects.chorus = 0.2f;
voirs_apply_effects(audio, &effects);
``` |
| **Python** | ```python
effects = voirs_ffi.AudioEffectsConfig(
    reverb=0.3,
    chorus=0.2
)
audio.apply_effects(effects)
``` |
| **Node.js** | ```javascript
const effects = {
    reverb: 0.3,
    chorus: 0.2
};
audio.applyEffects(effects);
``` |

## Error Handling

### Error Checking

| Language | Code |
|----------|------|
| **C** | ```c
if (result == NULL) {
    voirs_error_code_t error = voirs_get_last_error();
    const char* msg = voirs_get_error_message(error);
    fprintf(stderr, "Error: %s\n", msg);
}
``` |
| **Python** | ```python
try:
    result = pipeline.synthesize(text)
except voirs_ffi.VoirsError as e:
    print(f"Error: {e}")
``` |
| **Node.js** | ```javascript
try {
    const result = await pipeline.synthesize(text);
} catch (error) {
    console.error(`Error: ${error.message}`);
}
``` |

## Advanced Features

### Streaming Synthesis

| Language | Code |
|----------|------|
| **C** | ```c
voirs_streaming_synthesis_t* stream = 
    voirs_streaming_synthesis_create(pipeline, config);
voirs_streaming_synthesis_write(stream, chunk);
voirs_audio_result_t* result = 
    voirs_streaming_synthesis_read(stream);
``` |
| **Python** | ```python
with pipeline.stream(config) as stream:
    stream.write(chunk)
    result = stream.read()
``` |
| **Node.js** | ```javascript
const stream = pipeline.createStream(config);
stream.write(chunk);
const result = await stream.read();
``` |

### Callback Registration

| Language | Code |
|----------|------|
| **C** | ```c
void progress_callback(float progress, void* user_data) {
    printf("Progress: %.2f%%\n", progress * 100);
}
voirs_register_progress_callback(pipeline, progress_callback, NULL);
``` |
| **Python** | ```python
def progress_callback(progress):
    print(f"Progress: {progress*100:.2f}%")

pipeline.register_progress_callback(progress_callback)
``` |
| **Node.js** | ```javascript
pipeline.onProgress((progress) => {
    console.log(`Progress: ${(progress*100).toFixed(2)}%`);
});
``` |

## Memory Management Patterns

### Zero-Copy Operations

| Language | Code |
|----------|------|
| **C** | ```c
voirs_zero_copy_buffer_t* buffer = 
    voirs_zero_copy_buffer_create(data, size);
voirs_zero_copy_view_t* view = 
    voirs_zero_copy_view_create(buffer, offset, length);
``` |
| **Python** | ```python
buffer = voirs_ffi.ZeroCopyBuffer(data)
view = buffer.create_view(offset, length)
``` |
| **Node.js** | ```javascript
const buffer = new VoirsFFI.ZeroCopyBuffer(data);
const view = buffer.createView(offset, length);
``` |

### Memory Pool Management

| Language | Code |
|----------|------|
| **C** | ```c
voirs_allocator_config_t pool_config = {0};
pool_config.type = VOIRS_ALLOCATOR_POOL;
pool_config.pool_size = 1024 * 1024;
voirs_set_allocator(&pool_config);
``` |
| **Python** | ```python
voirs_ffi.set_allocator(
    voirs_ffi.AllocatorType.POOL,
    pool_size=1024*1024
)
``` |
| **Node.js** | ```javascript
VoirsFFI.setAllocator({
    type: VoirsFFI.AllocatorType.POOL,
    poolSize: 1024 * 1024
});
``` |

## Type Mappings

### Basic Types

| VoiRS Type | C Type | Python Type | Node.js Type |
|------------|--------|-------------|--------------|
| `f32` | `float` | `float` | `number` |
| `f64` | `double` | `float` | `number` |
| `i32` | `int32_t` | `int` | `number` |
| `u32` | `uint32_t` | `int` | `number` |
| `bool` | `int` (0/1) | `bool` | `boolean` |
| `String` | `const char*` | `str` | `string` |

### Complex Types

| VoiRS Type | C Type | Python Type | Node.js Type |
|------------|--------|-------------|--------------|
| `AudioFormat` | `voirs_audio_format_t` | `voirs_ffi.AudioFormat` | `VoirsFFI.AudioFormat` |
| `QualityLevel` | `voirs_quality_level_t` | `voirs_ffi.Quality` | `VoirsFFI.Quality` |
| `ErrorCode` | `voirs_error_code_t` | `voirs_ffi.ErrorCode` | `VoirsFFI.ErrorCode` |

## Constants and Enums

### Audio Formats

| Constant | C | Python | Node.js |
|----------|---|--------|---------|
| WAV | `VOIRS_FORMAT_WAV` | `voirs_ffi.Format.WAV` | `VoirsFFI.Format.WAV` |
| MP3 | `VOIRS_FORMAT_MP3` | `voirs_ffi.Format.MP3` | `VoirsFFI.Format.MP3` |
| FLAC | `VOIRS_FORMAT_FLAC` | `voirs_ffi.Format.FLAC` | `VoirsFFI.Format.FLAC` |

### Quality Levels

| Constant | C | Python | Node.js |
|----------|---|--------|---------|
| Low | `VOIRS_QUALITY_LOW` | `voirs_ffi.Quality.LOW` | `VoirsFFI.Quality.LOW` |
| Medium | `VOIRS_QUALITY_MEDIUM` | `voirs_ffi.Quality.MEDIUM` | `VoirsFFI.Quality.MEDIUM` |
| High | `VOIRS_QUALITY_HIGH` | `voirs_ffi.Quality.HIGH` | `VoirsFFI.Quality.HIGH` |
| Ultra | `VOIRS_QUALITY_ULTRA` | `voirs_ffi.Quality.ULTRA` | `VoirsFFI.Quality.ULTRA` |

## Platform-Specific Considerations

### Thread Management

| Platform | C | Python | Node.js |
|----------|---|--------|---------|
| **Windows** | Uses Windows threads | GIL management | V8 isolates |
| **macOS** | pthread with Grand Central Dispatch | GIL + multiprocessing | libuv thread pool |
| **Linux** | pthread with CPU affinity | GIL + threading | Worker threads |

### Memory Allocation

| Platform | C | Python | Node.js |
|----------|---|--------|---------|
| **Windows** | HeapAlloc/VirtualAlloc | Python heap | V8 garbage collector |
| **macOS** | malloc/mmap | Python heap | V8 garbage collector |
| **Linux** | malloc/mmap with NUMA | Python heap | V8 garbage collector |

## Best Practices by Language

### C
- Always check return values for NULL
- Use proper error handling with `voirs_get_last_error()`
- Manually manage memory with create/destroy pairs
- Use thread-safe functions for concurrent access

### Python
- Use context managers for automatic cleanup
- Handle exceptions with try/catch blocks
- Leverage Python's garbage collection
- Use asyncio for concurrent operations

### Node.js
- Use promises/async-await for asynchronous operations
- Handle errors with try/catch or .catch()
- Call dispose() methods for manual cleanup
- Use Worker threads for CPU-intensive tasks