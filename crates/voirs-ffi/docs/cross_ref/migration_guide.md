# Migration Guide

This guide helps you migrate between different VoiRS FFI language bindings and from other voice synthesis libraries.

## Table of Contents

1. [Migrating Between Language Bindings](#migrating-between-language-bindings)
2. [Migrating from Other Libraries](#migrating-from-other-libraries)
3. [Version Migration](#version-migration)
4. [Performance Migration](#performance-migration)

## Migrating Between Language Bindings

### From C to Python

#### Basic Setup Migration

**Before (C):**
```c
#include "voirs_ffi.h"

int main() {
    voirs_synthesis_config_t config = {0};
    config.quality = VOIRS_QUALITY_HIGH;
    config.speed = 1.0f;
    
    voirs_pipeline_t* pipeline = voirs_pipeline_create(&config);
    if (!pipeline) {
        fprintf(stderr, "Failed to create pipeline\n");
        return 1;
    }
    
    voirs_synthesis_result_t* result = voirs_synthesize(pipeline, "Hello World", NULL);
    if (!result) {
        fprintf(stderr, "Synthesis failed\n");
        voirs_pipeline_destroy(pipeline);
        return 1;
    }
    
    // Use result...
    
    voirs_synthesis_result_destroy(result);
    voirs_pipeline_destroy(pipeline);
    return 0;
}
```

**After (Python):**
```python
import voirs_ffi

def main():
    config = voirs_ffi.SynthesisConfig(
        quality=voirs_ffi.Quality.HIGH,
        speed=1.0
    )
    
    try:
        pipeline = voirs_ffi.Pipeline(config)
        result = pipeline.synthesize("Hello World")
        
        # Use result...
        
    except voirs_ffi.VoirsError as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
```

#### Key Changes:
- **Memory Management**: Python handles cleanup automatically
- **Error Handling**: Use try/catch instead of return value checking
- **Configuration**: Use named parameters instead of struct initialization
- **Types**: Use Python enums instead of C constants

### From Python to Node.js

#### Async/Await Migration

**Before (Python):**
```python
import voirs_ffi

def synthesize_text(text):
    config = voirs_ffi.SynthesisConfig(quality=voirs_ffi.Quality.HIGH)
    pipeline = voirs_ffi.Pipeline(config)
    result = pipeline.synthesize(text)
    return result.audio_data

# Usage
audio = synthesize_text("Hello World")
```

**After (Node.js):**
```javascript
const VoirsFFI = require('voirs-ffi');

async function synthesizeText(text) {
    const config = {
        quality: VoirsFFI.Quality.HIGH
    };
    
    const pipeline = new VoirsFFI.Pipeline(config);
    try {
        const result = await pipeline.synthesize(text);
        return result.audioData;
    } finally {
        pipeline.dispose();
    }
}

// Usage
(async () => {
    const audio = await synthesizeText("Hello World");
})();
```

#### Key Changes:
- **Async Operations**: Use async/await for synthesis operations
- **Manual Cleanup**: Call dispose() explicitly
- **Naming Convention**: camelCase instead of snake_case
- **Module System**: Use require/import statements

### From Node.js to C

#### Callback Migration

**Before (Node.js):**
```javascript
const pipeline = new VoirsFFI.Pipeline(config);

pipeline.onProgress((progress) => {
    console.log(`Progress: ${(progress * 100).toFixed(2)}%`);
});

pipeline.onComplete((result) => {
    console.log("Synthesis completed");
    processAudio(result.audioData);
});

await pipeline.synthesize(text);
```

**After (C):**
```c
void progress_callback(float progress, void* user_data) {
    printf("Progress: %.2f%%\n", progress * 100);
}

void complete_callback(voirs_synthesis_result_t* result, void* user_data) {
    printf("Synthesis completed\n");
    process_audio(result);
}

int main() {
    voirs_pipeline_t* pipeline = voirs_pipeline_create(&config);
    
    voirs_register_progress_callback(pipeline, progress_callback, NULL);
    voirs_register_complete_callback(pipeline, complete_callback, NULL);
    
    voirs_synthesis_result_t* result = voirs_synthesize(pipeline, text, NULL);
    
    // Cleanup...
    return 0;
}
```

#### Key Changes:
- **Callbacks**: Use function pointers instead of lambda functions
- **Memory Management**: Manual create/destroy lifecycle
- **Error Handling**: Check return values explicitly
- **Types**: Use C types instead of JavaScript objects

## Migrating from Other Libraries

### From eSpeak-NG

#### Basic API Migration

**Before (eSpeak-NG):**
```c
#include <espeak-ng/speak_lib.h>

int main() {
    espeak_Initialize(AUDIO_OUTPUT_PLAYBACK, 0, NULL, 0);
    espeak_SetVoiceByName("en");
    espeak_SetParameter(espeakRATE, 175, 0);
    espeak_SetParameter(espeakVOLUME, 100, 0);
    
    espeak_Synth("Hello World", strlen("Hello World"), 0, POS_CHARACTER, 0, 
                espeakCHARS_AUTO, NULL, NULL);
    
    espeak_Terminate();
    return 0;
}
```

**After (VoiRS FFI):**
```c
#include "voirs_ffi.h"

int main() {
    voirs_synthesis_config_t config = {0};
    config.quality = VOIRS_QUALITY_HIGH;
    config.speed = 1.0f;  // Normal speed
    config.volume = 1.0f; // Full volume
    
    voirs_pipeline_t* pipeline = voirs_pipeline_create(&config);
    if (!pipeline) return 1;
    
    voirs_synthesis_result_t* result = voirs_synthesize(pipeline, "Hello World", NULL);
    if (result) {
        // Process result->audio_data
        voirs_synthesis_result_destroy(result);
    }
    
    voirs_pipeline_destroy(pipeline);
    return 0;
}
```

#### Migration Map:

| eSpeak-NG | VoiRS FFI | Notes |
|-----------|-----------|-------|
| `espeak_Initialize()` | `voirs_pipeline_create()` | VoiRS uses pipeline-based approach |
| `espeak_SetVoiceByName()` | `config.voice_id` | Voice selection via configuration |
| `espeak_SetParameter(espeakRATE)` | `config.speed` | Normalized 0.1-3.0 range |
| `espeak_Synth()` | `voirs_synthesize()` | Returns audio data directly |
| `espeak_Terminate()` | `voirs_pipeline_destroy()` | Per-pipeline cleanup |

### From Festival

#### Python API Migration

**Before (Festival):**
```python
import festival

def synthesize_text(text):
    festival.initialize()
    festival.say_text(text)
    festival.shutdown()

synthesize_text("Hello World")
```

**After (VoiRS FFI):**
```python
import voirs_ffi

def synthesize_text(text):
    config = voirs_ffi.SynthesisConfig(
        quality=voirs_ffi.Quality.HIGH,
        output_format=voirs_ffi.Format.WAV
    )
    
    pipeline = voirs_ffi.Pipeline(config)
    result = pipeline.synthesize(text)
    
    # Save or play audio data
    with open("output.wav", "wb") as f:
        f.write(result.audio_data)

synthesize_text("Hello World")
```

#### Key Differences:
- **Audio Output**: VoiRS returns audio data instead of direct playback
- **Configuration**: More granular control over synthesis parameters
- **Error Handling**: Structured exception handling
- **Performance**: Better memory management and threading

### From Amazon Polly SDK

#### AWS SDK Migration

**Before (Polly):**
```python
import boto3
from botocore.exceptions import BotoCoreError, ClientError

def synthesize_speech(text):
    polly = boto3.client('polly')
    
    try:
        response = polly.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId='Joanna'
        )
        
        with open('output.mp3', 'wb') as file:
            file.write(response['AudioStream'].read())
            
    except (BotoCoreError, ClientError) as error:
        print(f"Error: {error}")

synthesize_speech("Hello World")
```

**After (VoiRS FFI):**
```python
import voirs_ffi

def synthesize_speech(text):
    config = voirs_ffi.SynthesisConfig(
        output_format=voirs_ffi.Format.MP3,
        voice_id="neural_female_01",  # Equivalent voice
        quality=voirs_ffi.Quality.HIGH
    )
    
    try:
        pipeline = voirs_ffi.Pipeline(config)
        result = pipeline.synthesize(text)
        
        with open('output.mp3', 'wb') as file:
            file.write(result.audio_data)
            
    except voirs_ffi.VoirsError as error:
        print(f"Error: {error}")

synthesize_speech("Hello World")
```

#### Migration Benefits:
- **Offline Processing**: No network dependency
- **Lower Latency**: Local processing
- **Cost Efficiency**: No per-request charges
- **Privacy**: Audio never leaves your system

## Version Migration

### VoiRS FFI v0.1.x to v0.2.x (Future)

#### Breaking Changes (Anticipated):

1. **Config Structure Changes:**
```c
// v0.1.x
voirs_synthesis_config_t config = {0};
config.quality = VOIRS_QUALITY_HIGH;

// v0.2.x (anticipated)
voirs_synthesis_config_v2_t config = voirs_synthesis_config_default();
config.quality_profile = VOIRS_PROFILE_NATURAL_HIGH;
```

2. **Error Handling Improvements:**
```c
// v0.1.x
voirs_error_code_t error = voirs_get_last_error();

// v0.2.x (anticipated)
voirs_error_info_t error_info = voirs_get_detailed_error();
```

### Migration Strategy:

1. **Use Compatibility Headers:**
```c
#define VOIRS_FFI_COMPATIBILITY_V1
#include "voirs_ffi.h"
```

2. **Gradual Migration:**
   - Update one component at a time
   - Use feature flags to toggle between versions
   - Maintain comprehensive tests

## Performance Migration

### Optimizing During Migration

#### Memory Management Improvements

**Before (Basic):**
```c
voirs_synthesis_result_t* result = voirs_synthesize(pipeline, text, NULL);
// Process immediately
voirs_synthesis_result_destroy(result);
```

**After (Optimized):**
```c
// Enable pool allocator
voirs_allocator_config_t allocator = {0};
allocator.type = VOIRS_ALLOCATOR_POOL;
allocator.pool_size = 10 * 1024 * 1024; // 10MB pool
voirs_set_allocator(&allocator);

// Use batch processing
voirs_batch_synthesis_t* batch = voirs_batch_synthesis_create(pipeline);
for (int i = 0; i < text_count; i++) {
    voirs_batch_synthesis_add(batch, texts[i]);
}
voirs_synthesis_result_t** results = voirs_batch_synthesis_execute(batch);
```

#### Threading Optimization

**Before (Sequential):**
```python
results = []
for text in texts:
    result = pipeline.synthesize(text)
    results.append(result)
```

**After (Parallel):**
```python
import concurrent.futures
import voirs_ffi

def synthesize_text(text):
    # Each thread gets its own pipeline
    config = voirs_ffi.SynthesisConfig(quality=voirs_ffi.Quality.HIGH)
    pipeline = voirs_ffi.Pipeline(config)
    return pipeline.synthesize(text)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(synthesize_text, texts))
```

## Common Migration Issues

### Memory Leaks

**Problem:** Forgetting to destroy C resources
```c
voirs_pipeline_t* pipeline = voirs_pipeline_create(&config);
voirs_synthesis_result_t* result = voirs_synthesize(pipeline, text, NULL);
// Missing cleanup!
```

**Solution:** Always pair create/destroy calls
```c
voirs_pipeline_t* pipeline = voirs_pipeline_create(&config);
if (pipeline) {
    voirs_synthesis_result_t* result = voirs_synthesize(pipeline, text, NULL);
    if (result) {
        // Use result...
        voirs_synthesis_result_destroy(result);
    }
    voirs_pipeline_destroy(pipeline);
}
```

### Thread Safety

**Problem:** Sharing pipeline between threads
```c
// Thread 1
voirs_synthesize(shared_pipeline, text1, NULL);

// Thread 2 (concurrent)
voirs_synthesize(shared_pipeline, text2, NULL); // Race condition!
```

**Solution:** Use thread-local pipelines or synchronization
```c
// Option 1: Thread-local pipelines
thread_local voirs_pipeline_t* thread_pipeline = NULL;

// Option 2: Mutex protection
pthread_mutex_t pipeline_mutex = PTHREAD_MUTEX_INITIALIZER;
```

### Performance Degradation

**Problem:** Not using optimal configuration
```python
# Suboptimal
config = voirs_ffi.SynthesisConfig()  # Uses defaults
```

**Solution:** Configure for your use case
```python
# Optimized for real-time
config = voirs_ffi.SynthesisConfig(
    quality=voirs_ffi.Quality.MEDIUM,  # Lower quality for speed
    thread_count=2,  # Limit threads for latency
    use_simd=True,   # Enable SIMD acceleration
    cache_size=512*1024  # Smaller cache for real-time
)
```

## Migration Testing

### Validation Checklist

1. **Functional Testing:**
   - [ ] All synthesis operations work correctly
   - [ ] Audio output quality is maintained
   - [ ] Error handling functions properly

2. **Performance Testing:**
   - [ ] Latency is within acceptable bounds
   - [ ] Memory usage is optimized
   - [ ] CPU utilization is reasonable

3. **Integration Testing:**
   - [ ] Works with existing application code
   - [ ] Compatible with deployment environment
   - [ ] Handles edge cases properly

### Automated Migration Tools

#### Configuration Converter
```python
# Convert eSpeak parameters to VoiRS config
def convert_espeak_config(rate, volume, voice):
    return voirs_ffi.SynthesisConfig(
        speed=rate / 175.0,  # Normalize eSpeak rate
        volume=volume / 100.0,  # Normalize volume
        voice_id=map_espeak_voice(voice)
    )
```

#### Batch Migration Script
```bash
#!/bin/bash
# Migrate C code from eSpeak to VoiRS
sed -i 's/espeak_Initialize/voirs_pipeline_create/g' *.c
sed -i 's/espeak_Synth/voirs_synthesize/g' *.c
sed -i 's/espeak_Terminate/voirs_pipeline_destroy/g' *.c
```

This migration guide provides comprehensive support for transitioning between different VoiRS FFI language bindings and migrating from other voice synthesis libraries. Follow the specific sections relevant to your migration path and use the provided examples as templates for your own code.