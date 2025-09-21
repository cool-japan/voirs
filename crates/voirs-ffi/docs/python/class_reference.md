# Python Class Reference

This document provides detailed documentation for all classes in the VoiRS Python bindings.

## Core Classes

### VoirsPipeline

The main class for speech synthesis operations.

#### Constructor

```python
VoirsPipeline(config: Optional[SynthesisConfig] = None)
```

Creates a new synthesis pipeline with optional configuration.

**Parameters:**
- `config` (SynthesisConfig, optional): Configuration object for the pipeline

**Example:**
```python
from voirs_ffi import VoirsPipeline, SynthesisConfig

# Default configuration
pipeline = VoirsPipeline()

# Custom configuration
config = SynthesisConfig(sample_rate=22050, use_gpu=True)
pipeline = VoirsPipeline(config)
```

#### Alternative Constructors

```python
@classmethod
VoirsPipeline.with_config(**kwargs) -> VoirsPipeline
```

Creates a pipeline with keyword arguments for configuration.

**Parameters:**
- `**kwargs`: Configuration options as keyword arguments

**Example:**
```python
pipeline = VoirsPipeline.with_config(
    sample_rate=22050,
    use_gpu=True,
    num_threads=4,
    language="en-US"
)
```

#### Methods

##### synthesize()

```python
def synthesize(self, text: str, voice: Optional[str] = None) -> PyAudioBuffer
```

Synthesizes text to speech.

**Parameters:**
- `text` (str): Text to synthesize
- `voice` (str, optional): Voice ID to use for synthesis

**Returns:**
- `PyAudioBuffer`: Generated audio buffer

**Example:**
```python
audio = pipeline.synthesize("Hello, world!")
audio_with_voice = pipeline.synthesize("Hello!", voice="female-1")
```

##### synthesize_ssml()

```python
def synthesize_ssml(self, ssml: str) -> PyAudioBuffer
```

Synthesizes SSML markup to speech.

**Parameters:**
- `ssml` (str): SSML markup to synthesize

**Returns:**
- `PyAudioBuffer`: Generated audio buffer

**Example:**
```python
ssml = '<speak>Hello <break time="1s"/> world!</speak>'
audio = pipeline.synthesize_ssml(ssml)
```

##### get_voices()

```python
def get_voices(self) -> List[VoiceInfo]
```

Returns list of available voices.

**Returns:**
- `List[VoiceInfo]`: List of voice information objects

**Example:**
```python
voices = pipeline.get_voices()
for voice in voices:
    print(f"Voice: {voice.name} (ID: {voice.id})")
```

##### set_voice()

```python
def set_voice(self, voice_id: str) -> None
```

Sets the current voice for synthesis.

**Parameters:**
- `voice_id` (str): Voice ID to set as current

**Example:**
```python
pipeline.set_voice("female-1")
```

---

### PyAudioBuffer

Container for audio data with processing capabilities.

#### Properties

```python
@property
def samples(self) -> List[float]
```

Raw audio samples as a list of floats.

```python
@property
def sample_rate(self) -> int
```

Sample rate of the audio in Hz.

```python
@property
def channels(self) -> int
```

Number of audio channels.

```python
@property
def duration(self) -> float
```

Duration of the audio in seconds.

#### Methods

##### save()

```python
def save(self, path: str, format: str = "wav") -> None
```

Saves audio to a file.

**Parameters:**
- `path` (str): Output file path
- `format` (str, optional): Audio format ("wav", "mp3", "flac")

**Example:**
```python
audio.save("output.wav")
audio.save("output.mp3", format="mp3")
```

##### play()

```python
def play(self) -> None
```

Plays audio through the default audio device.

**Example:**
```python
audio.play()
```

##### to_numpy()

```python
def to_numpy(self) -> numpy.ndarray
```

Converts audio to NumPy array (requires NumPy).

**Returns:**
- `numpy.ndarray`: Audio data as NumPy array

**Example:**
```python
import numpy as np
audio_array = audio.to_numpy()
```

##### apply_gain()

```python
def apply_gain(self, gain: float) -> None
```

Applies gain to the audio buffer.

**Parameters:**
- `gain` (float): Gain factor (1.0 = no change)

**Example:**
```python
audio.apply_gain(0.5)  # Reduce volume by half
```

##### normalize()

```python
def normalize(self) -> None
```

Normalizes audio to prevent clipping.

**Example:**
```python
audio.normalize()
```

---

### SynthesisConfig

Configuration object for synthesis pipeline.

#### Constructor

```python
SynthesisConfig(
    sample_rate: int = 22050,
    use_gpu: bool = False,
    num_threads: int = 4,
    language: str = "en-US",
    quality: str = "medium"
)
```

**Parameters:**
- `sample_rate` (int): Audio sample rate in Hz
- `use_gpu` (bool): Whether to use GPU acceleration
- `num_threads` (int): Number of threads for processing
- `language` (str): Language code for synthesis
- `quality` (str): Quality level ("low", "medium", "high")

**Example:**
```python
config = SynthesisConfig(
    sample_rate=44100,
    use_gpu=True,
    num_threads=8,
    language="en-US",
    quality="high"
)
```

#### Properties

All constructor parameters are available as properties:

```python
config.sample_rate = 44100
config.use_gpu = True
config.num_threads = 8
config.language = "en-US"
config.quality = "high"
```

---

### VoiceInfo

Information about an available voice.

#### Properties

```python
@property
def id(self) -> str
```

Unique voice identifier.

```python
@property
def name(self) -> str
```

Human-readable voice name.

```python
@property
def language(self) -> str
```

Language code for the voice.

```python
@property
def gender(self) -> str
```

Voice gender ("male", "female", "neutral").

```python
@property
def age(self) -> str
```

Voice age category ("child", "adult", "elderly").

#### Methods

##### __str__()

```python
def __str__(self) -> str
```

String representation of the voice.

**Example:**
```python
voice = voices[0]
print(voice)  # "Voice: Alice (ID: alice-en-us)"
```

---

## Convenience Functions

### create_pipeline()

```python
def create_pipeline(**kwargs) -> VoirsPipeline
```

Creates a VoiRS pipeline with optional configuration.

**Parameters:**
- `**kwargs`: Configuration options

**Returns:**
- `VoirsPipeline`: Configured pipeline instance

**Example:**
```python
pipeline = create_pipeline(use_gpu=True, num_threads=4)
```

### synthesize_text()

```python
def synthesize_text(text: str, **config) -> PyAudioBuffer
```

Quick text-to-speech synthesis with automatic pipeline management.

**Parameters:**
- `text` (str): Text to synthesize
- `**config`: Optional configuration for the pipeline

**Returns:**
- `PyAudioBuffer`: Synthesized audio

**Example:**
```python
audio = synthesize_text("Hello, world!", use_gpu=True)
```

### check_compatibility()

```python
def check_compatibility() -> dict
```

Checks system compatibility and available features.

**Returns:**
- `dict`: Dictionary with compatibility information

**Example:**
```python
info = check_compatibility()
print(f"GPU available: {info['gpu']}")
print(f"NumPy support: {info['numpy']}")
```

---

## Constants

### Feature Flags

```python
HAS_NUMPY: bool  # True if NumPy support is available
HAS_GPU: bool    # True if GPU acceleration is available
```

### Version Information

```python
__version__: str  # Package version string
```

## Error Handling

All VoiRS operations can raise the following exceptions:

- `VoirsError`: Base exception for all VoiRS errors
- `SynthesisError`: Errors during text-to-speech synthesis
- `ConfigurationError`: Invalid configuration parameters
- `VoiceNotFoundError`: Requested voice is not available
- `AudioProcessingError`: Errors during audio processing

**Example:**
```python
try:
    audio = pipeline.synthesize("Hello, world!")
except VoirsError as e:
    print(f"Synthesis failed: {e}")
```