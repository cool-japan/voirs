# voirs-ffi

[![Crates.io](https://img.shields.io/crates/v/voirs-ffi.svg)](https://crates.io/crates/voirs-ffi)
[![Documentation](https://docs.rs/voirs-ffi/badge.svg)](https://docs.rs/voirs-ffi)

**Foreign Function Interface (FFI) bindings for VoiRS speech synthesis.**

This crate provides C-compatible bindings and Python integration for the VoiRS speech synthesis framework, enabling seamless integration with applications written in other programming languages.

## Features

- **C API**: Zero-cost C-compatible bindings with comprehensive error handling
- **Python Bindings**: PyO3-based Python package with native performance
- **Memory Safety**: Rust's memory safety guarantees extended to FFI boundaries
- **Thread Safety**: Safe concurrent access from multiple threads
- **Error Handling**: Comprehensive error propagation across language boundaries
- **Streaming Support**: Real-time synthesis through callback interfaces

## Supported Languages

| Language | Binding Type | Status | Features |
|----------|-------------|--------|----------|
| **C/C++** | Native FFI | ✅ Stable | Full API, callbacks, threading |
| **Python** | PyO3 | ✅ Stable | Async/await, NumPy, type hints |
| **Node.js** | NAPI | 🚧 Beta | Async, TypeScript definitions |
| **Java** | JNI | 📋 Planned | JVM integration |
| **C#/.NET** | P/Invoke | 📋 Planned | .NET Core support |
| **Go** | CGO | 📋 Planned | Go module integration |

## Quick Start

### C/C++ Usage

```c
#include "voirs.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Initialize VoiRS
    VoirsHandle* handle = voirs_create();
    if (!handle) {
        fprintf(stderr, "Failed to initialize VoiRS\n");
        return 1;
    }
    
    // Set voice
    if (voirs_set_voice(handle, "en-US-female-calm") != VOIRS_SUCCESS) {
        fprintf(stderr, "Failed to set voice\n");
        voirs_destroy(handle);
        return 1;
    }
    
    // Synthesize text
    VoirsAudioBuffer* audio = NULL;
    VoirsResult result = voirs_synthesize_text(
        handle,
        "Hello, world! This is VoiRS speaking from C.",
        &audio
    );
    
    if (result == VOIRS_SUCCESS) {
        // Save audio to file
        voirs_audio_save_wav(audio, "output.wav");
        printf("Synthesis complete! Audio saved to output.wav\n");
        
        // Cleanup
        voirs_audio_free(audio);
    } else {
        fprintf(stderr, "Synthesis failed: %s\n", voirs_get_last_error(handle));
    }
    
    voirs_destroy(handle);
    return 0;
}
```

### Python Usage

```python
import voirs
import asyncio

async def main():
    # Initialize VoiRS pipeline
    pipeline = await voirs.VoirsPipeline.create(
        voice="en-US-female-calm",
        quality=voirs.Quality.HIGH
    )
    
    # Synthesize text
    audio = await pipeline.synthesize("Hello from Python!")
    
    # Save to file
    audio.save_wav("output.wav")
    
    # Access raw audio data as NumPy array
    samples = audio.numpy()
    print(f"Generated {len(samples)} audio samples")
    print(f"Sample rate: {audio.sample_rate} Hz")
    print(f"Duration: {audio.duration:.2f} seconds")

# Run async function
asyncio.run(main())
```

## C API Reference

### Core Types

```c
// Opaque handle to VoiRS instance
typedef struct VoirsHandle VoirsHandle;

// Audio buffer containing synthesized speech
typedef struct VoirsAudioBuffer VoirsAudioBuffer;

// Result codes
typedef enum {
    VOIRS_SUCCESS = 0,
    VOIRS_ERROR_INVALID_HANDLE = -1,
    VOIRS_ERROR_INVALID_PARAMETER = -2,
    VOIRS_ERROR_VOICE_NOT_FOUND = -3,
    VOIRS_ERROR_SYNTHESIS_FAILED = -4,
    VOIRS_ERROR_IO_ERROR = -5,
    VOIRS_ERROR_OUT_OF_MEMORY = -6,
    VOIRS_ERROR_THREAD_ERROR = -7,
} VoirsResult;

// Audio format configuration
typedef struct {
    uint32_t sample_rate;    // Sample rate in Hz
    uint16_t channels;       // Number of channels (1=mono, 2=stereo)
    uint16_t bit_depth;      // Bit depth (16, 24, 32)
} VoirsAudioFormat;

// Synthesis configuration
typedef struct {
    float speaking_rate;     // Speaking rate multiplier (0.5 - 2.0)
    float pitch_shift;       // Pitch shift in semitones (-12.0 - 12.0)
    float volume_gain;       // Volume gain in dB (-20.0 - 20.0)
    bool enable_enhancement; // Enable audio enhancement
    VoirsAudioFormat format; // Output audio format
} VoirsSynthesisConfig;
```

### Core Functions

```c
// Instance management
VoirsHandle* voirs_create(void);
VoirsHandle* voirs_create_with_config(const char* config_path);
void voirs_destroy(VoirsHandle* handle);

// Voice management
VoirsResult voirs_set_voice(VoirsHandle* handle, const char* voice_id);
VoirsResult voirs_get_voice(VoirsHandle* handle, char* buffer, size_t buffer_size);
VoirsResult voirs_list_voices(VoirsHandle* handle, char*** voices, size_t* count);
void voirs_free_voice_list(char** voices, size_t count);

// Synthesis
VoirsResult voirs_synthesize_text(
    VoirsHandle* handle,
    const char* text,
    VoirsAudioBuffer** audio
);

VoirsResult voirs_synthesize_text_with_config(
    VoirsHandle* handle,
    const char* text,
    const VoirsSynthesisConfig* config,
    VoirsAudioBuffer** audio
);

VoirsResult voirs_synthesize_ssml(
    VoirsHandle* handle,
    const char* ssml,
    VoirsAudioBuffer** audio
);

// Streaming synthesis
typedef void (*VoirsAudioCallback)(
    const float* samples,
    size_t sample_count,
    void* user_data
);

VoirsResult voirs_synthesize_streaming(
    VoirsHandle* handle,
    const char* text,
    VoirsAudioCallback callback,
    void* user_data
);

// Audio buffer operations
size_t voirs_audio_get_sample_count(const VoirsAudioBuffer* audio);
uint32_t voirs_audio_get_sample_rate(const VoirsAudioBuffer* audio);
uint16_t voirs_audio_get_channels(const VoirsAudioBuffer* audio);
float voirs_audio_get_duration(const VoirsAudioBuffer* audio);

const float* voirs_audio_get_samples(const VoirsAudioBuffer* audio);
VoirsResult voirs_audio_copy_samples(
    const VoirsAudioBuffer* audio,
    float* buffer,
    size_t buffer_size
);

VoirsResult voirs_audio_save_wav(const VoirsAudioBuffer* audio, const char* filename);
VoirsResult voirs_audio_save_format(
    const VoirsAudioBuffer* audio,
    const char* filename,
    const char* format  // "wav", "flac", "mp3", "opus"
);

void voirs_audio_free(VoirsAudioBuffer* audio);

// Error handling
const char* voirs_get_last_error(VoirsHandle* handle);
VoirsResult voirs_clear_error(VoirsHandle* handle);
const char* voirs_result_to_string(VoirsResult result);

// Threading
VoirsResult voirs_set_thread_count(VoirsHandle* handle, uint32_t thread_count);
uint32_t voirs_get_thread_count(VoirsHandle* handle);

// Configuration
VoirsResult voirs_set_config_value(
    VoirsHandle* handle,
    const char* key,
    const char* value
);
VoirsResult voirs_get_config_value(
    VoirsHandle* handle,
    const char* key,
    char* buffer,
    size_t buffer_size
);
```

## Python API Reference

### Installation

```bash
# Install from PyPI (when released)
pip install voirs

# Install from source
pip install maturin
maturin develop --release
```

### Core Classes

```python
class VoirsPipeline:
    """Main VoiRS synthesis pipeline."""
    
    @classmethod
    async def create(
        cls,
        voice: str = "en-US-female-calm",
        quality: Quality = Quality.HIGH,
        device: str = "auto",
        **kwargs
    ) -> "VoirsPipeline":
        """Create a new VoiRS pipeline."""
        ...
    
    async def synthesize(
        self,
        text: str,
        *,
        speed: float = 1.0,
        pitch: float = 0.0,
        volume: float = 0.0,
        enhance: bool = True
    ) -> "AudioBuffer":
        """Synthesize text to audio."""
        ...
    
    async def synthesize_ssml(self, ssml: str) -> "AudioBuffer":
        """Synthesize SSML markup to audio."""
        ...
    
    async def synthesize_stream(
        self,
        text: str,
        chunk_size: int = 256
    ) -> AsyncIterator["AudioBuffer"]:
        """Stream synthesis for long texts."""
        ...
    
    def set_voice(self, voice: str) -> None:
        """Change the active voice."""
        ...
    
    def get_voices(self) -> List[str]:
        """Get list of available voices."""
        ...

class AudioBuffer:
    """Audio buffer containing synthesized speech."""
    
    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        ...
    
    @property
    def channels(self) -> int:
        """Number of audio channels."""
        ...
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        ...
    
    def numpy(self) -> np.ndarray:
        """Get audio samples as NumPy array."""
        ...
    
    def save_wav(self, filename: str) -> None:
        """Save audio as WAV file."""
        ...
    
    def save(self, filename: str, format: str = "wav") -> None:
        """Save audio in specified format."""
        ...
    
    def play(self) -> None:
        """Play audio through system speakers."""
        ...

class Quality(Enum):
    """Synthesis quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class VoirsError(Exception):
    """Base exception for VoiRS errors."""
    pass
```

### Usage Examples

#### Basic Synthesis

```python
import voirs
import asyncio

async def basic_example():
    pipeline = await voirs.VoirsPipeline.create()
    audio = await pipeline.synthesize("Hello, world!")
    audio.save_wav("hello.wav")

asyncio.run(basic_example())
```

#### Advanced Synthesis with Controls

```python
import voirs
import asyncio

async def advanced_example():
    pipeline = await voirs.VoirsPipeline.create(
        voice="en-US-male-news",
        quality=voirs.Quality.ULTRA
    )
    
    audio = await pipeline.synthesize(
        "This is an important announcement!",
        speed=0.9,      # Slightly slower
        pitch=0.5,      # Slightly higher pitch
        volume=3.0,     # 3dB louder
        enhance=True    # Enable enhancement
    )
    
    audio.save("announcement.flac", format="flac")
    print(f"Generated {audio.duration:.2f}s of audio")

asyncio.run(advanced_example())
```

#### Streaming Synthesis

```python
import voirs
import asyncio

async def streaming_example():
    pipeline = await voirs.VoirsPipeline.create()
    
    long_text = "This is a very long text that will be synthesized in chunks..."
    
    async for audio_chunk in pipeline.synthesize_stream(long_text):
        # Process each chunk as it's generated
        samples = audio_chunk.numpy()
        print(f"Received chunk: {len(samples)} samples")
        
        # Could play or save each chunk immediately
        # audio_chunk.play()

asyncio.run(streaming_example())
```

#### SSML Synthesis

```python
import voirs
import asyncio

async def ssml_example():
    pipeline = await voirs.VoirsPipeline.create()
    
    ssml_text = """
    <speak>
        <p>Welcome to <emphasis level="strong">VoiRS</emphasis>!</p>
        <break time="1s"/>
        <p><prosody rate="slow" pitch="low">This is slow and low.</prosody></p>
        <p><prosody rate="fast" pitch="high">This is fast and high!</prosody></p>
    </speak>
    """
    
    audio = await pipeline.synthesize_ssml(ssml_text)
    audio.save_wav("ssml_demo.wav")

asyncio.run(ssml_example())
```

#### NumPy Integration

```python
import voirs
import numpy as np
import matplotlib.pyplot as plt
import asyncio

async def numpy_example():
    pipeline = await voirs.VoirsPipeline.create()
    audio = await pipeline.synthesize("Hello, NumPy!")
    
    # Get audio data as NumPy array
    samples = audio.numpy()
    
    # Analyze audio
    print(f"Shape: {samples.shape}")
    print(f"Max amplitude: {np.max(np.abs(samples)):.3f}")
    print(f"RMS level: {np.sqrt(np.mean(samples**2)):.3f}")
    
    # Plot waveform
    time = np.linspace(0, audio.duration, len(samples))
    plt.figure(figsize=(10, 4))
    plt.plot(time, samples)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("VoiRS Synthesis Waveform")
    plt.grid(True)
    plt.show()

asyncio.run(numpy_example())
```

## Building and Installation

### C/C++ Integration

#### CMake Integration

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(my_project)

# Find VoiRS
find_package(PkgConfig REQUIRED)
pkg_check_modules(VOIRS REQUIRED voirs-ffi)

# Add executable
add_executable(my_app main.c)

# Link VoiRS
target_link_libraries(my_app ${VOIRS_LIBRARIES})
target_include_directories(my_app PRIVATE ${VOIRS_INCLUDE_DIRS})
target_compile_options(my_app PRIVATE ${VOIRS_CFLAGS_OTHER})
```

#### Manual Compilation

```bash
# Linux/macOS
gcc -o my_app main.c -lvoirs_ffi -lpthread -ldl -lm

# Windows (MSVC)
cl main.c voirs_ffi.lib
```

### Python Development

#### Building from Source

```bash
# Clone repository
git clone https://github.com/cool-japan/voirs.git
cd voirs/crates/voirs-ffi

# Install Python dependencies
pip install maturin numpy

# Build and install
maturin develop --release

# Run tests
python -m pytest tests/
```

#### Creating Wheel Packages

```bash
# Build wheel for current platform
maturin build --release

# Build wheels for multiple platforms
maturin build --release --target x86_64-unknown-linux-gnu
maturin build --release --target x86_64-pc-windows-msvc
maturin build --release --target x86_64-apple-darwin
```

## Error Handling

### C Error Handling

```c
#include "voirs.h"

VoirsResult handle_synthesis_error(VoirsHandle* handle, VoirsResult result) {
    if (result != VOIRS_SUCCESS) {
        const char* error_msg = voirs_get_last_error(handle);
        const char* result_str = voirs_result_to_string(result);
        
        fprintf(stderr, "VoiRS Error [%s]: %s\n", result_str, error_msg);
        
        // Handle specific error types
        switch (result) {
            case VOIRS_ERROR_VOICE_NOT_FOUND:
                fprintf(stderr, "Available voices:\n");
                
                char** voices;
                size_t count;
                if (voirs_list_voices(handle, &voices, &count) == VOIRS_SUCCESS) {
                    for (size_t i = 0; i < count; i++) {
                        fprintf(stderr, "  - %s\n", voices[i]);
                    }
                    voirs_free_voice_list(voices, count);
                }
                break;
                
            case VOIRS_ERROR_OUT_OF_MEMORY:
                fprintf(stderr, "Insufficient memory. Try reducing quality or text length.\n");
                break;
                
            default:
                break;
        }
    }
    
    return result;
}
```

### Python Error Handling

```python
import voirs

async def safe_synthesis():
    try:
        pipeline = await voirs.VoirsPipeline.create(voice="nonexistent-voice")
        audio = await pipeline.synthesize("Hello, world!")
        
    except voirs.VoiceNotFoundError as e:
        print(f"Voice not found: {e}")
        
        # Get available voices
        voices = voirs.get_available_voices()
        print("Available voices:")
        for voice in voices:
            print(f"  - {voice}")
            
    except voirs.SynthesisError as e:
        print(f"Synthesis failed: {e}")
        
    except voirs.VoirsError as e:
        print(f"VoiRS error: {e}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
```

## Performance Considerations

### C Performance Tips

```c
// Reuse VoiRS handle for multiple syntheses
VoirsHandle* handle = voirs_create();

// Configure synthesis parameters once
VoirsSynthesisConfig config = {
    .speaking_rate = 1.0f,
    .pitch_shift = 0.0f,
    .volume_gain = 0.0f,
    .enable_enhancement = true,
    .format = {
        .sample_rate = 22050,
        .channels = 1,
        .bit_depth = 16
    }
};

// Synthesize multiple texts efficiently
for (int i = 0; i < num_texts; i++) {
    VoirsAudioBuffer* audio;
    if (voirs_synthesize_text_with_config(handle, texts[i], &config, &audio) == VOIRS_SUCCESS) {
        // Process audio...
        voirs_audio_free(audio);
    }
}

voirs_destroy(handle);
```

### Python Performance Tips

```python
import voirs
import asyncio

async def efficient_batch_synthesis():
    # Create pipeline once
    pipeline = await voirs.VoirsPipeline.create(
        quality=voirs.Quality.HIGH,
        device="cuda:0"  # Use GPU if available
    )
    
    texts = ["Text 1", "Text 2", "Text 3", ...]
    
    # Process in parallel
    tasks = [pipeline.synthesize(text) for text in texts]
    audio_buffers = await asyncio.gather(*tasks)
    
    # Save results
    for i, audio in enumerate(audio_buffers):
        audio.save_wav(f"output_{i:03d}.wav")
```

## Thread Safety

### C Thread Safety

```c
#include <pthread.h>
#include "voirs.h"

// VoiRS handles are thread-safe for synthesis operations
void* synthesis_thread(void* arg) {
    VoirsHandle* handle = (VoirsHandle*)arg;
    
    VoirsAudioBuffer* audio;
    VoirsResult result = voirs_synthesize_text(
        handle,
        "Thread-safe synthesis",
        &audio
    );
    
    if (result == VOIRS_SUCCESS) {
        // Process audio...
        voirs_audio_free(audio);
    }
    
    return NULL;
}

int main() {
    VoirsHandle* handle = voirs_create();
    
    pthread_t threads[4];
    for (int i = 0; i < 4; i++) {
        pthread_create(&threads[i], NULL, synthesis_thread, handle);
    }
    
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    
    voirs_destroy(handle);
    return 0;
}
```

### Python Thread Safety

```python
import voirs
import asyncio
import concurrent.futures

async def thread_safe_example():
    pipeline = await voirs.VoirsPipeline.create()
    
    # VoiRS Python bindings are thread-safe
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        loop = asyncio.get_event_loop()
        
        futures = [
            loop.run_in_executor(executor, sync_synthesize, pipeline, f"Text {i}")
            for i in range(10)
        ]
        
        results = await asyncio.gather(*futures)
        return results

def sync_synthesize(pipeline, text):
    # Note: This would need a sync version of the API
    # or proper async handling within threads
    return f"Synthesized: {text}"
```

## Troubleshooting

### Common Issues

**Library not found:**
```bash
# Linux: Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# macOS: Add to DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH

# Windows: Add to PATH or place DLL in executable directory
```

**Python import errors:**
```bash
# Ensure maturin is installed
pip install maturin

# Rebuild bindings
maturin develop --release

# Check Python path
python -c "import sys; print(sys.path)"
```

**Memory issues:**
```c
// Always free audio buffers
VoirsAudioBuffer* audio;
if (voirs_synthesize_text(handle, text, &audio) == VOIRS_SUCCESS) {
    // Use audio...
    voirs_audio_free(audio);  // Important!
}

// Destroy handle when done
voirs_destroy(handle);
```

## Contributing

We welcome contributions! Please see the [main repository](https://github.com/cool-japan/voirs) for contribution guidelines.

### Development Setup

```bash
git clone https://github.com/cool-japan/voirs.git
cd voirs/crates/voirs-ffi

# Install development dependencies
pip install maturin pytest numpy

# Build and test
maturin develop
python -m pytest tests/

# Run C tests
make test-c

# Check bindings
cargo test
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.