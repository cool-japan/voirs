# Quick Start Guide

Get up and running with VoiRS Python bindings in just a few minutes.

## Installation

### Basic Installation

```bash
pip install voirs-ffi
```

### With Optional Dependencies

```bash
# For NumPy support (recommended)
pip install voirs-ffi[numpy]

# For GPU acceleration (requires CUDA)
pip install voirs-ffi[gpu]

# Full installation
pip install voirs-ffi[all]
```

## Basic Usage

### 1. Simple Text-to-Speech

```python
from voirs_ffi import VoirsPipeline

# Create a pipeline
pipeline = VoirsPipeline()

# Synthesize text
audio = pipeline.synthesize("Hello, world!")

# Save to file
audio.save("hello.wav")
```

### 2. Using Convenience Functions

```python
from voirs_ffi import synthesize_text

# Quick synthesis
audio = synthesize_text("Hello, world!")
audio.save("hello.wav")
```

### 3. Play Audio Directly

```python
from voirs_ffi import VoirsPipeline

pipeline = VoirsPipeline()
audio = pipeline.synthesize("Hello, world!")

# Play through speakers
audio.play()
```

## Configuration

### Basic Configuration

```python
from voirs_ffi import VoirsPipeline, SynthesisConfig

# Create custom configuration
config = SynthesisConfig(
    sample_rate=44100,
    use_gpu=True,
    num_threads=8,
    language="en-US",
    quality="high"
)

# Create pipeline with config
pipeline = VoirsPipeline(config)
```

### Using Keyword Arguments

```python
from voirs_ffi import VoirsPipeline

# More convenient configuration
pipeline = VoirsPipeline.with_config(
    sample_rate=44100,
    use_gpu=True,
    num_threads=8,
    language="en-US",
    quality="high"
)
```

## Working with Voices

### List Available Voices

```python
from voirs_ffi import VoirsPipeline

pipeline = VoirsPipeline()

# Get all voices
voices = pipeline.get_voices()

# Print voice information
for voice in voices:
    print(f"Voice: {voice.name}")
    print(f"  ID: {voice.id}")
    print(f"  Language: {voice.language}")
    print(f"  Gender: {voice.gender}")
    print(f"  Age: {voice.age}")
    print()
```

### Set Default Voice

```python
from voirs_ffi import VoirsPipeline

pipeline = VoirsPipeline()

# Set voice for all subsequent synthesis
pipeline.set_voice("female-1")

# All synthesis will use this voice
audio = pipeline.synthesize("Hello, world!")
```

### Use Voice for Single Synthesis

```python
from voirs_ffi import VoirsPipeline

pipeline = VoirsPipeline()

# Use specific voice for this synthesis only
audio = pipeline.synthesize("Hello, world!", voice="female-1")
```

## Audio Processing

### Basic Audio Operations

```python
from voirs_ffi import VoirsPipeline

pipeline = VoirsPipeline()
audio = pipeline.synthesize("Hello, world!")

# Get audio properties
print(f"Sample rate: {audio.sample_rate}")
print(f"Channels: {audio.channels}")
print(f"Duration: {audio.duration:.2f} seconds")

# Apply gain
audio.apply_gain(0.5)  # Reduce volume by half

# Normalize audio
audio.normalize()
```

### Working with NumPy (Optional)

```python
import numpy as np
from voirs_ffi import VoirsPipeline

pipeline = VoirsPipeline()
audio = pipeline.synthesize("Hello, world!")

# Convert to NumPy array
audio_array = audio.to_numpy()

# Process with NumPy
processed = np.fft.fft(audio_array)

# Convert back to audio buffer
# (Note: This requires additional processing)
```

## SSML Support

### Basic SSML

```python
from voirs_ffi import VoirsPipeline

pipeline = VoirsPipeline()

# Simple SSML
ssml = '''
<speak>
    Hello, <break time="1s"/> world!
    <emphasis level="strong">This is emphasized text.</emphasis>
</speak>
'''

audio = pipeline.synthesize_ssml(ssml)
audio.save("ssml_output.wav")
```

### Advanced SSML

```python
from voirs_ffi import VoirsPipeline

pipeline = VoirsPipeline()

# Advanced SSML with prosody
ssml = '''
<speak>
    <prosody rate="slow" pitch="low">
        This is spoken slowly and at a low pitch.
    </prosody>
    <break time="2s"/>
    <prosody rate="fast" pitch="high">
        This is spoken quickly and at a high pitch.
    </prosody>
</speak>
'''

audio = pipeline.synthesize_ssml(ssml)
audio.save("advanced_ssml.wav")
```

## Error Handling

### Basic Error Handling

```python
from voirs_ffi import VoirsPipeline, VoirsError

try:
    pipeline = VoirsPipeline()
    audio = pipeline.synthesize("Hello, world!")
    audio.save("output.wav")
except VoirsError as e:
    print(f"VoiRS error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Specific Error Types

```python
from voirs_ffi import (
    VoirsPipeline, 
    VoirsError, 
    SynthesisError,
    VoiceNotFoundError
)

try:
    pipeline = VoirsPipeline()
    pipeline.set_voice("nonexistent-voice")
    audio = pipeline.synthesize("Hello, world!")
except VoiceNotFoundError as e:
    print(f"Voice not found: {e}")
except SynthesisError as e:
    print(f"Synthesis failed: {e}")
except VoirsError as e:
    print(f"General VoiRS error: {e}")
```

## Performance Tips

### GPU Acceleration

```python
from voirs_ffi import VoirsPipeline, check_compatibility

# Check if GPU is available
info = check_compatibility()
if info['gpu']:
    print("GPU acceleration available")
    pipeline = VoirsPipeline.with_config(use_gpu=True)
else:
    print("GPU not available, using CPU")
    pipeline = VoirsPipeline()
```

### Threading Configuration

```python
from voirs_ffi import VoirsPipeline
import multiprocessing

# Use all available CPU cores
num_cores = multiprocessing.cpu_count()
pipeline = VoirsPipeline.with_config(num_threads=num_cores)
```

### Quality vs Speed

```python
from voirs_ffi import VoirsPipeline

# Fast synthesis (lower quality)
fast_pipeline = VoirsPipeline.with_config(quality="low")

# High quality synthesis (slower)
quality_pipeline = VoirsPipeline.with_config(quality="high")
```

## File Format Support

### Supported Output Formats

```python
from voirs_ffi import VoirsPipeline

pipeline = VoirsPipeline()
audio = pipeline.synthesize("Hello, world!")

# Save in different formats
audio.save("output.wav", format="wav")
audio.save("output.mp3", format="mp3")
audio.save("output.flac", format="flac")
audio.save("output.ogg", format="ogg")
```

### Format-Specific Options

```python
from voirs_ffi import VoirsPipeline

pipeline = VoirsPipeline()
audio = pipeline.synthesize("Hello, world!")

# High-quality FLAC
audio.save("output.flac", format="flac", compression_level=8)

# MP3 with specific bitrate
audio.save("output.mp3", format="mp3", bitrate=320)
```

## System Requirements

### Minimum Requirements

- Python 3.8+
- 2GB RAM
- 1GB disk space

### Recommended Requirements

- Python 3.9+
- 4GB RAM
- 2GB disk space
- GPU with CUDA support (for GPU acceleration)

### Check System Compatibility

```python
from voirs_ffi import check_compatibility
import json

info = check_compatibility()
print(json.dumps(info, indent=2))
```

## Next Steps

- Read the [Class Reference](class_reference.md) for detailed API documentation
- Check out [Tutorial Notebooks](tutorials/) for interactive examples
- Review the [Performance Guide](performance_guide.md) for optimization tips
- Explore [Integration Examples](integration_examples.md) for real-world use cases

## Common Issues

### Import Errors

If you get import errors, ensure the package is installed correctly:

```bash
pip install --upgrade voirs-ffi
```

### Audio Playback Issues

If audio playback doesn't work, install additional audio dependencies:

```bash
pip install sounddevice  # For cross-platform audio
```

### GPU Not Detected

Ensure CUDA is installed and accessible:

```bash
nvidia-smi  # Check CUDA installation
```

For more troubleshooting, see the [Error Handling Guide](error_handling.md).