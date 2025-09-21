# VoiRS Python API Reference

Complete reference for all VoiRS Python binding functions, classes, and constants.

## Module: voirs_ffi

### Main Classes

#### VoirsPipeline

Main class for text-to-speech synthesis operations.

```python
class VoirsPipeline:
    def __init__(self, config: Optional[SynthesisConfig] = None) -> None
    
    @classmethod
    def with_config(cls, **kwargs) -> 'VoirsPipeline'
    
    def synthesize(self, text: str, voice: Optional[str] = None) -> PyAudioBuffer
    def synthesize_ssml(self, ssml: str) -> PyAudioBuffer
    def get_voices(self) -> List[VoiceInfo]
    def set_voice(self, voice_id: str) -> None
    def get_config(self) -> SynthesisConfig
    def update_config(self, **kwargs) -> None
    def reset(self) -> None
```

#### PyAudioBuffer

Container for audio data with processing capabilities.

```python
class PyAudioBuffer:
    @property
    def samples(self) -> List[float]
    
    @property
    def sample_rate(self) -> int
    
    @property
    def channels(self) -> int
    
    @property
    def duration(self) -> float
    
    @property
    def length(self) -> int
    
    def save(self, path: str, format: str = "wav", **kwargs) -> None
    def play(self) -> None
    def to_numpy(self) -> numpy.ndarray
    def to_bytes(self) -> bytes
    def apply_gain(self, gain: float) -> None
    def normalize(self) -> None
    def fade_in(self, duration: float) -> None
    def fade_out(self, duration: float) -> None
    def trim_silence(self, threshold: float = 0.01) -> None
    def resample(self, target_rate: int) -> 'PyAudioBuffer'
    def to_mono(self) -> 'PyAudioBuffer'
    def to_stereo(self) -> 'PyAudioBuffer'
```

#### SynthesisConfig

Configuration object for synthesis pipeline.

```python
class SynthesisConfig:
    def __init__(
        self,
        sample_rate: int = 22050,
        use_gpu: bool = False,
        num_threads: int = 4,
        language: str = "en-US",
        quality: str = "medium",
        voice_id: Optional[str] = None,
        speaking_rate: float = 1.0,
        pitch_shift: float = 0.0,
        volume_gain: float = 1.0,
        enable_preprocessing: bool = True,
        enable_postprocessing: bool = True,
        cache_size: int = 100,
        memory_limit_mb: int = 512
    ) -> None
    
    def copy(self) -> 'SynthesisConfig'
    def update(self, **kwargs) -> None
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, data: dict) -> 'SynthesisConfig'
    def validate(self) -> None
```

#### VoiceInfo

Information about an available voice.

```python
class VoiceInfo:
    @property
    def id(self) -> str
    
    @property
    def name(self) -> str
    
    @property
    def language(self) -> str
    
    @property
    def gender(self) -> str
    
    @property
    def age(self) -> str
    
    @property
    def style(self) -> str
    
    @property
    def sample_rate(self) -> int
    
    @property
    def quality(self) -> str
    
    def __str__(self) -> str
    def __repr__(self) -> str
    def to_dict(self) -> dict
```

### Convenience Functions

#### create_pipeline()

```python
def create_pipeline(**kwargs) -> VoirsPipeline
```

Creates a VoiRS pipeline with optional configuration.

**Parameters:**
- `**kwargs`: Configuration options as keyword arguments

**Returns:**
- `VoirsPipeline`: Configured pipeline instance

#### synthesize_text()

```python
def synthesize_text(
    text: str, 
    voice: Optional[str] = None,
    **config
) -> PyAudioBuffer
```

Quick text-to-speech synthesis with automatic pipeline management.

**Parameters:**
- `text` (str): Text to synthesize
- `voice` (str, optional): Voice ID to use
- `**config`: Configuration options

**Returns:**
- `PyAudioBuffer`: Synthesized audio

#### synthesize_ssml()

```python
def synthesize_ssml(
    ssml: str,
    **config
) -> PyAudioBuffer
```

Quick SSML synthesis with automatic pipeline management.

**Parameters:**
- `ssml` (str): SSML markup to synthesize
- `**config`: Configuration options

**Returns:**
- `PyAudioBuffer`: Synthesized audio

#### list_voices()

```python
def list_voices() -> List[VoiceInfo]
```

Returns list of all available voices.

**Returns:**
- `List[VoiceInfo]`: List of voice information objects

#### get_voice()

```python
def get_voice(voice_id: str) -> Optional[VoiceInfo]
```

Get information about a specific voice.

**Parameters:**
- `voice_id` (str): Voice identifier

**Returns:**
- `Optional[VoiceInfo]`: Voice information or None if not found

#### check_compatibility()

```python
def check_compatibility() -> dict
```

Checks system compatibility and available features.

**Returns:**
- `dict`: Dictionary with compatibility information
  - `gpu` (bool): GPU acceleration available
  - `numpy` (bool): NumPy support available
  - `cuda_version` (str): CUDA version if available
  - `memory_gb` (float): Available system memory in GB
  - `cpu_cores` (int): Number of CPU cores
  - `supported_formats` (List[str]): Supported audio formats

### Audio Processing Functions

#### load_audio()

```python
def load_audio(path: str) -> PyAudioBuffer
```

Load audio from file.

**Parameters:**
- `path` (str): Path to audio file

**Returns:**
- `PyAudioBuffer`: Loaded audio data

#### save_audio()

```python
def save_audio(
    audio: PyAudioBuffer, 
    path: str, 
    format: str = "wav",
    **kwargs
) -> None
```

Save audio to file.

**Parameters:**
- `audio` (PyAudioBuffer): Audio data to save
- `path` (str): Output file path
- `format` (str): Audio format
- `**kwargs`: Format-specific options

#### convert_audio()

```python
def convert_audio(
    audio: PyAudioBuffer,
    target_rate: Optional[int] = None,
    target_channels: Optional[int] = None,
    target_format: Optional[str] = None
) -> PyAudioBuffer
```

Convert audio format.

**Parameters:**
- `audio` (PyAudioBuffer): Source audio
- `target_rate` (int, optional): Target sample rate
- `target_channels` (int, optional): Target channel count
- `target_format` (str, optional): Target format

**Returns:**
- `PyAudioBuffer`: Converted audio

### Configuration Management

#### get_default_config()

```python
def get_default_config() -> SynthesisConfig
```

Get default synthesis configuration.

**Returns:**
- `SynthesisConfig`: Default configuration

#### load_config()

```python
def load_config(path: str) -> SynthesisConfig
```

Load configuration from file.

**Parameters:**
- `path` (str): Path to configuration file (JSON, YAML, or TOML)

**Returns:**
- `SynthesisConfig`: Loaded configuration

#### save_config()

```python
def save_config(config: SynthesisConfig, path: str) -> None
```

Save configuration to file.

**Parameters:**
- `config` (SynthesisConfig): Configuration to save
- `path` (str): Output file path

### Exception Classes

#### VoirsError

Base exception for all VoiRS errors.

```python
class VoirsError(Exception):
    def __init__(self, message: str, error_code: Optional[int] = None) -> None
    
    @property
    def error_code(self) -> Optional[int]
```

#### SynthesisError

Errors during text-to-speech synthesis.

```python
class SynthesisError(VoirsError):
    pass
```

#### ConfigurationError

Invalid configuration parameters.

```python
class ConfigurationError(VoirsError):
    pass
```

#### VoiceNotFoundError

Requested voice is not available.

```python
class VoiceNotFoundError(VoirsError):
    def __init__(self, voice_id: str) -> None
    
    @property
    def voice_id(self) -> str
```

#### AudioProcessingError

Errors during audio processing.

```python
class AudioProcessingError(VoirsError):
    pass
```

#### SystemCompatibilityError

System compatibility issues.

```python
class SystemCompatibilityError(VoirsError):
    pass
```

### Constants

#### Version Information

```python
__version__: str  # Package version string
__build__: str    # Build information
__git_hash__: str # Git commit hash
```

#### Feature Flags

```python
HAS_NUMPY: bool      # True if NumPy support is available
HAS_GPU: bool        # True if GPU acceleration is available
HAS_CUDA: bool       # True if CUDA is available
HAS_SOUNDDEVICE: bool # True if sounddevice is available
```

#### Quality Levels

```python
QUALITY_LOW: str = "low"
QUALITY_MEDIUM: str = "medium"
QUALITY_HIGH: str = "high"
QUALITY_ULTRA: str = "ultra"
```

#### Audio Formats

```python
FORMAT_WAV: str = "wav"
FORMAT_MP3: str = "mp3"
FORMAT_FLAC: str = "flac"
FORMAT_OGG: str = "ogg"
FORMAT_AAC: str = "aac"

SUPPORTED_FORMATS: List[str] = [
    FORMAT_WAV, FORMAT_MP3, FORMAT_FLAC, 
    FORMAT_OGG, FORMAT_AAC
]
```

#### Sample Rates

```python
SAMPLE_RATE_8K: int = 8000
SAMPLE_RATE_16K: int = 16000
SAMPLE_RATE_22K: int = 22050
SAMPLE_RATE_44K: int = 44100
SAMPLE_RATE_48K: int = 48000

SUPPORTED_SAMPLE_RATES: List[int] = [
    SAMPLE_RATE_8K, SAMPLE_RATE_16K, SAMPLE_RATE_22K,
    SAMPLE_RATE_44K, SAMPLE_RATE_48K
]
```

### Performance Utilities

#### ProfiledPipeline

Pipeline wrapper with performance monitoring.

```python
class ProfiledPipeline:
    def __init__(self, pipeline: VoirsPipeline) -> None
    
    def synthesize(self, text: str, **kwargs) -> Tuple[PyAudioBuffer, dict]
    def get_stats(self) -> dict
    def reset_stats(self) -> None
```

#### benchmark_synthesis()

```python
def benchmark_synthesis(
    texts: List[str],
    config: Optional[SynthesisConfig] = None,
    iterations: int = 10
) -> dict
```

Benchmark synthesis performance.

**Parameters:**
- `texts` (List[str]): Test texts
- `config` (SynthesisConfig, optional): Configuration to test
- `iterations` (int): Number of iterations per test

**Returns:**
- `dict`: Benchmark results

### Type Hints

```python
from typing import List, Optional, Union, Dict, Any, Tuple, Callable

AudioData = Union[List[float], numpy.ndarray, bytes]
ConfigDict = Dict[str, Any]
SynthesisCallback = Callable[[str, float], None]
ErrorCallback = Callable[[Exception], None]
```

## Usage Examples

### Basic Synthesis

```python
import voirs_ffi as vf

# Simple synthesis
audio = vf.synthesize_text("Hello, world!")
audio.save("output.wav")

# With configuration
audio = vf.synthesize_text(
    "Hello, world!", 
    sample_rate=44100,
    quality="high",
    use_gpu=True
)
```

### Pipeline Management

```python
import voirs_ffi as vf

# Create and configure pipeline
pipeline = vf.VoirsPipeline.with_config(
    sample_rate=22050,
    quality="medium",
    num_threads=4
)

# List available voices
voices = pipeline.get_voices()
print(f"Available voices: {len(voices)}")

# Set default voice
pipeline.set_voice("female-1")

# Synthesize multiple texts
texts = ["Hello", "How are you?", "Goodbye"]
for text in texts:
    audio = pipeline.synthesize(text)
    audio.save(f"{text.replace(' ', '_')}.wav")
```

### Error Handling

```python
import voirs_ffi as vf

try:
    pipeline = vf.VoirsPipeline()
    audio = pipeline.synthesize("Hello, world!")
    audio.save("output.wav")
except vf.VoiceNotFoundError as e:
    print(f"Voice not found: {e.voice_id}")
except vf.SynthesisError as e:
    print(f"Synthesis failed: {e}")
except vf.VoirsError as e:
    print(f"VoiRS error: {e}")
```

### Audio Processing

```python
import voirs_ffi as vf

# Load existing audio
audio = vf.load_audio("input.wav")

# Process audio
audio.normalize()
audio.apply_gain(0.8)
audio.fade_in(0.5)
audio.fade_out(1.0)

# Convert format
stereo_audio = audio.to_stereo()
high_rate_audio = audio.resample(44100)

# Save processed audio
audio.save("processed.wav")
```

## Performance Considerations

### Memory Usage

- Audio buffers use approximately 4 bytes per sample per channel
- Large audio files should be processed in chunks
- Use `PyAudioBuffer.to_bytes()` for memory-efficient storage

### GPU Acceleration

- Requires CUDA-compatible GPU
- Provides 2-10x speed improvement for synthesis
- Memory usage increases with GPU processing

### Threading

- Default configuration uses 4 threads
- Optimal thread count is typically equal to CPU core count
- Too many threads can reduce performance due to context switching

### Caching

- Voice models are cached automatically
- Configuration objects can be reused for better performance
- Clear cache with `pipeline.reset()` if memory is constrained