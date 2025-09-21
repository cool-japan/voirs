# Configuration Guide

Comprehensive guide to configuring VoiRS for optimal performance and quality.

## Configuration Overview

VoiRS uses the `SynthesisConfig` class to manage all synthesis parameters. Configuration can be set at pipeline creation, modified during runtime, or loaded from external files.

## Basic Configuration

### Creating Configuration Objects

```python
from voirs_ffi import SynthesisConfig

# Default configuration
config = SynthesisConfig()

# Custom configuration
config = SynthesisConfig(
    sample_rate=44100,
    use_gpu=True,
    num_threads=8,
    language="en-US",
    quality="high"
)
```

### Using Keyword Arguments

```python
from voirs_ffi import VoirsPipeline

# Direct configuration during pipeline creation
pipeline = VoirsPipeline.with_config(
    sample_rate=44100,
    use_gpu=True,
    num_threads=8,
    language="en-US",
    quality="high"
)
```

## Configuration Parameters

### Audio Settings

#### sample_rate

Audio sample rate in Hz.

- **Type**: `int`
- **Default**: `22050`
- **Valid values**: `8000`, `16000`, `22050`, `44100`, `48000`
- **Impact**: Higher rates provide better quality but larger file sizes

```python
config = SynthesisConfig(sample_rate=44100)  # CD quality
config = SynthesisConfig(sample_rate=22050)  # Standard quality
config = SynthesisConfig(sample_rate=16000)  # Phone quality
```

#### quality

Synthesis quality level.

- **Type**: `str`
- **Default**: `"medium"`
- **Valid values**: `"low"`, `"medium"`, `"high"`, `"ultra"`
- **Impact**: Higher quality increases processing time and memory usage

```python
config = SynthesisConfig(quality="low")    # Fast, lower quality
config = SynthesisConfig(quality="medium") # Balanced
config = SynthesisConfig(quality="high")   # Slow, higher quality
config = SynthesisConfig(quality="ultra")  # Slowest, best quality
```

### Performance Settings

#### use_gpu

Enable GPU acceleration.

- **Type**: `bool`
- **Default**: `False`
- **Requirements**: CUDA-compatible GPU
- **Impact**: 2-10x speed improvement when available

```python
from voirs_ffi import check_compatibility

# Check GPU availability first
compatibility = check_compatibility()
if compatibility['gpu']:
    config = SynthesisConfig(use_gpu=True)
else:
    config = SynthesisConfig(use_gpu=False)
```

#### num_threads

Number of CPU threads for processing.

- **Type**: `int`
- **Default**: `4`
- **Valid range**: `1` to `32`
- **Recommendation**: Number of CPU cores

```python
import multiprocessing

# Use all available CPU cores
num_cores = multiprocessing.cpu_count()
config = SynthesisConfig(num_threads=num_cores)

# Conservative threading
config = SynthesisConfig(num_threads=4)
```

#### memory_limit_mb

Maximum memory usage in megabytes.

- **Type**: `int`
- **Default**: `512`
- **Valid range**: `128` to `8192`
- **Impact**: Affects caching and model loading

```python
config = SynthesisConfig(memory_limit_mb=1024)  # 1GB limit
config = SynthesisConfig(memory_limit_mb=256)   # Low memory
```

### Voice and Language Settings

#### language

Target language for synthesis.

- **Type**: `str`
- **Default**: `"en-US"`
- **Format**: ISO language-country code
- **Impact**: Affects pronunciation and voice availability

```python
config = SynthesisConfig(language="en-US")  # American English
config = SynthesisConfig(language="en-GB")  # British English
config = SynthesisConfig(language="es-ES")  # Spanish
config = SynthesisConfig(language="fr-FR")  # French
config = SynthesisConfig(language="de-DE")  # German
config = SynthesisConfig(language="ja-JP")  # Japanese
```

#### voice_id

Default voice identifier.

- **Type**: `Optional[str]`
- **Default**: `None` (auto-select)
- **Impact**: Determines voice characteristics

```python
from voirs_ffi import list_voices

# List available voices
voices = list_voices()
for voice in voices:
    print(f"{voice.id}: {voice.name} ({voice.language})")

# Set specific voice
config = SynthesisConfig(voice_id="female-adult-1")
```

### Prosody Settings

#### speaking_rate

Speech rate multiplier.

- **Type**: `float`
- **Default**: `1.0`
- **Valid range**: `0.25` to `4.0`
- **Impact**: Controls speech speed

```python
config = SynthesisConfig(speaking_rate=0.8)  # 20% slower
config = SynthesisConfig(speaking_rate=1.0)  # Normal speed
config = SynthesisConfig(speaking_rate=1.5)  # 50% faster
```

#### pitch_shift

Pitch adjustment in semitones.

- **Type**: `float`
- **Default**: `0.0`
- **Valid range**: `-12.0` to `12.0`
- **Impact**: Changes voice pitch

```python
config = SynthesisConfig(pitch_shift=-2.0)  # Lower pitch
config = SynthesisConfig(pitch_shift=0.0)   # No change
config = SynthesisConfig(pitch_shift=3.0)   # Higher pitch
```

#### volume_gain

Volume gain multiplier.

- **Type**: `float`
- **Default**: `1.0`
- **Valid range**: `0.1` to `3.0`
- **Impact**: Controls output volume

```python
config = SynthesisConfig(volume_gain=0.5)  # Quieter
config = SynthesisConfig(volume_gain=1.0)  # Normal volume
config = SynthesisConfig(volume_gain=1.5)  # Louder
```

### Processing Settings

#### enable_preprocessing

Enable text preprocessing.

- **Type**: `bool`
- **Default**: `True`
- **Impact**: Improves text normalization and pronunciation

```python
config = SynthesisConfig(enable_preprocessing=True)   # Recommended
config = SynthesisConfig(enable_preprocessing=False)  # Raw text
```

#### enable_postprocessing

Enable audio postprocessing.

- **Type**: `bool`
- **Default**: `True`
- **Impact**: Applies audio enhancement and normalization

```python
config = SynthesisConfig(enable_postprocessing=True)   # Better quality
config = SynthesisConfig(enable_postprocessing=False)  # Faster
```

#### cache_size

Model cache size (number of models).

- **Type**: `int`
- **Default**: `100`
- **Valid range**: `10` to `1000`
- **Impact**: Affects memory usage and loading speed

```python
config = SynthesisConfig(cache_size=50)   # Low memory
config = SynthesisConfig(cache_size=200)  # More caching
```

## Configuration Profiles

### Performance Profiles

#### High Performance (GPU)

Optimized for speed with GPU acceleration.

```python
high_performance_config = SynthesisConfig(
    sample_rate=22050,
    quality="medium",
    use_gpu=True,
    num_threads=8,
    memory_limit_mb=1024,
    enable_preprocessing=True,
    enable_postprocessing=False,  # Skip for speed
    cache_size=200
)
```

#### High Quality

Optimized for audio quality.

```python
high_quality_config = SynthesisConfig(
    sample_rate=44100,
    quality="ultra",
    use_gpu=False,  # CPU often better for quality
    num_threads=4,
    memory_limit_mb=2048,
    enable_preprocessing=True,
    enable_postprocessing=True,
    cache_size=100
)
```

#### Low Memory

Optimized for systems with limited memory.

```python
low_memory_config = SynthesisConfig(
    sample_rate=16000,
    quality="low",
    use_gpu=False,
    num_threads=2,
    memory_limit_mb=128,
    enable_preprocessing=False,
    enable_postprocessing=False,
    cache_size=10
)
```

#### Balanced

Good balance of quality and performance.

```python
balanced_config = SynthesisConfig(
    sample_rate=22050,
    quality="medium",
    use_gpu=True,
    num_threads=4,
    memory_limit_mb=512,
    enable_preprocessing=True,
    enable_postprocessing=True,
    cache_size=50
)
```

### Use Case Profiles

#### Real-time Applications

For interactive applications requiring low latency.

```python
realtime_config = SynthesisConfig(
    sample_rate=16000,
    quality="low",
    use_gpu=True,
    num_threads=2,
    enable_preprocessing=False,
    enable_postprocessing=False,
    cache_size=50
)
```

#### Batch Processing

For processing large amounts of text offline.

```python
batch_config = SynthesisConfig(
    sample_rate=22050,
    quality="high",
    use_gpu=True,
    num_threads=multiprocessing.cpu_count(),
    memory_limit_mb=2048,
    cache_size=200
)
```

#### Podcast/Audiobook Production

For high-quality narrative content.

```python
production_config = SynthesisConfig(
    sample_rate=44100,
    quality="ultra",
    use_gpu=False,
    num_threads=4,
    enable_preprocessing=True,
    enable_postprocessing=True,
    speaking_rate=0.95,  # Slightly slower for clarity
    volume_gain=0.9      # Conservative volume
)
```

## Configuration Management

### Saving and Loading Configuration

#### Save to File

```python
import json
from voirs_ffi import SynthesisConfig

config = SynthesisConfig(sample_rate=44100, quality="high")

# Save as JSON
with open("config.json", "w") as f:
    json.dump(config.to_dict(), f, indent=2)

# Using built-in method
from voirs_ffi import save_config
save_config(config, "config.json")
```

#### Load from File

```python
import json
from voirs_ffi import SynthesisConfig, load_config

# Load as JSON
with open("config.json", "r") as f:
    config_dict = json.load(f)
config = SynthesisConfig.from_dict(config_dict)

# Using built-in method
config = load_config("config.json")
```

#### YAML Configuration

```python
import yaml
from voirs_ffi import SynthesisConfig

# Save as YAML
config = SynthesisConfig(sample_rate=44100, quality="high")
with open("config.yaml", "w") as f:
    yaml.dump(config.to_dict(), f, default_flow_style=False)

# Load from YAML
with open("config.yaml", "r") as f:
    config_dict = yaml.safe_load(f)
config = SynthesisConfig.from_dict(config_dict)
```

### Runtime Configuration Updates

#### Updating Existing Configuration

```python
from voirs_ffi import VoirsPipeline

pipeline = VoirsPipeline()

# Get current configuration
config = pipeline.get_config()

# Update specific parameters
config.quality = "high"
config.sample_rate = 44100

# Apply updates
pipeline.update_config(config)

# Or update directly
pipeline.update_config(quality="high", sample_rate=44100)
```

#### Configuration Validation

```python
from voirs_ffi import SynthesisConfig, ConfigurationError

try:
    config = SynthesisConfig(
        sample_rate=99999,  # Invalid sample rate
        quality="invalid"   # Invalid quality
    )
    config.validate()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## Environment Variables

VoiRS respects certain environment variables for configuration:

```bash
# Performance settings
export VOIRS_NUM_THREADS=8
export VOIRS_USE_GPU=true
export VOIRS_MEMORY_LIMIT_MB=1024

# Quality settings
export VOIRS_DEFAULT_QUALITY=high
export VOIRS_SAMPLE_RATE=44100

# Behavior settings
export VOIRS_ENABLE_PREPROCESSING=true
export VOIRS_ENABLE_POSTPROCESSING=true
export VOIRS_CACHE_SIZE=100

# Debugging
export VOIRS_LOG_LEVEL=info
export VOIRS_ENABLE_PROFILING=false
```

Using environment variables:

```python
import os
from voirs_ffi import SynthesisConfig

# Environment variables override defaults
config = SynthesisConfig()
print(f"Threads: {config.num_threads}")  # Uses VOIRS_NUM_THREADS if set
```

## Advanced Configuration

### Custom Voice Configuration

```python
from voirs_ffi import VoirsPipeline, SynthesisConfig

# Load custom voice models
config = SynthesisConfig(
    language="custom",
    voice_id="my-custom-voice"
)

pipeline = VoirsPipeline(config)

# Verify voice loaded correctly
voices = pipeline.get_voices()
custom_voices = [v for v in voices if v.id.startswith("my-custom")]
print(f"Loaded {len(custom_voices)} custom voices")
```

### Multi-language Configuration

```python
from voirs_ffi import VoirsPipeline

# Create language-specific pipelines
configs = {
    "english": SynthesisConfig(language="en-US", voice_id="female-us-1"),
    "spanish": SynthesisConfig(language="es-ES", voice_id="female-es-1"),
    "french": SynthesisConfig(language="fr-FR", voice_id="female-fr-1")
}

pipelines = {
    lang: VoirsPipeline(config) 
    for lang, config in configs.items()
}

# Use appropriate pipeline for each language
text_samples = {
    "english": "Hello, how are you today?",
    "spanish": "Hola, ¿cómo estás hoy?",
    "french": "Bonjour, comment allez-vous aujourd'hui?"
}

for lang, text in text_samples.items():
    audio = pipelines[lang].synthesize(text)
    audio.save(f"greeting_{lang}.wav")
```

### Dynamic Configuration

```python
from voirs_ffi import VoirsPipeline
import time

pipeline = VoirsPipeline()

def adaptive_quality(text_length):
    """Adjust quality based on text length"""
    if text_length < 50:
        return "low"      # Short text, fast processing
    elif text_length < 200:
        return "medium"   # Medium text, balanced
    else:
        return "high"     # Long text, quality matters

def synthesize_adaptive(text):
    quality = adaptive_quality(len(text))
    pipeline.update_config(quality=quality)
    
    start_time = time.time()
    audio = pipeline.synthesize(text)
    duration = time.time() - start_time
    
    print(f"Synthesized {len(text)} chars with {quality} quality in {duration:.2f}s")
    return audio

# Test adaptive synthesis
texts = [
    "Hi!",
    "This is a medium length sentence for testing.",
    "This is a much longer text that contains multiple sentences and should be processed with higher quality settings to ensure the best possible audio output for the end user."
]

for text in texts:
    audio = synthesize_adaptive(text)
    # Process audio as needed
```

## Troubleshooting Configuration

### Common Configuration Issues

#### GPU Not Available

```python
from voirs_ffi import check_compatibility, SynthesisConfig

compatibility = check_compatibility()
if not compatibility['gpu']:
    print("GPU not available, using CPU")
    config = SynthesisConfig(use_gpu=False)
else:
    print(f"GPU available: CUDA {compatibility['cuda_version']}")
    config = SynthesisConfig(use_gpu=True)
```

#### Memory Limitations

```python
import psutil
from voirs_ffi import SynthesisConfig

# Check available memory
available_mb = psutil.virtual_memory().available // (1024 * 1024)
print(f"Available memory: {available_mb} MB")

# Configure based on available memory
if available_mb < 1024:  # Less than 1GB
    config = SynthesisConfig(
        memory_limit_mb=min(256, available_mb // 4),
        cache_size=10,
        quality="low"
    )
elif available_mb < 4096:  # Less than 4GB
    config = SynthesisConfig(
        memory_limit_mb=512,
        cache_size=50,
        quality="medium"
    )
else:  # 4GB or more
    config = SynthesisConfig(
        memory_limit_mb=1024,
        cache_size=100,
        quality="high"
    )
```

#### Voice Not Found

```python
from voirs_ffi import VoirsPipeline, VoiceNotFoundError, list_voices

try:
    config = SynthesisConfig(voice_id="nonexistent-voice")
    pipeline = VoirsPipeline(config)
except VoiceNotFoundError as e:
    print(f"Voice '{e.voice_id}' not found")
    
    # List available voices
    voices = list_voices()
    print("Available voices:")
    for voice in voices:
        print(f"  {voice.id}: {voice.name}")
    
    # Use default voice
    config = SynthesisConfig()  # No voice_id specified
    pipeline = VoirsPipeline(config)
```

### Performance Debugging

```python
from voirs_ffi import VoirsPipeline, ProfiledPipeline
import time

# Create profiled pipeline for performance monitoring
base_pipeline = VoirsPipeline()
pipeline = ProfiledPipeline(base_pipeline)

# Test synthesis
text = "This is a test sentence for performance monitoring."
audio, stats = pipeline.synthesize(text)

print("Performance stats:")
print(f"  Synthesis time: {stats['synthesis_time']:.3f}s")
print(f"  Memory usage: {stats['memory_mb']:.1f} MB")
print(f"  GPU utilization: {stats['gpu_utilization']:.1f}%")
```

## Best Practices

### Configuration Guidelines

1. **Start with defaults**: Use default configuration and adjust as needed
2. **Profile your use case**: Measure performance with your specific workload
3. **Consider your constraints**: Balance quality, speed, and memory usage
4. **Test thoroughly**: Validate configuration changes with real content
5. **Document settings**: Save and version your configuration files

### Performance Optimization

1. **Use GPU when available**: Significant speed improvements for synthesis
2. **Match threads to cores**: Don't exceed your CPU core count
3. **Cache frequently used voices**: Increase cache size for repeated synthesis
4. **Batch processing**: Process multiple texts together when possible
5. **Appropriate quality levels**: Don't use highest quality unless necessary

### Memory Management

1. **Monitor memory usage**: Use system monitoring tools
2. **Adjust cache size**: Balance memory usage vs. loading speed
3. **Clear cache periodically**: Use `pipeline.reset()` for long-running applications
4. **Process in chunks**: Break large texts into smaller pieces
5. **Limit concurrent operations**: Avoid too many simultaneous synthesis operations