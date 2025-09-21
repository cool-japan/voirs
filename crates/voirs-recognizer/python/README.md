# VoiRS Recognizer Python Bindings

Python bindings for the VoiRS voice recognition and analysis library.

## Installation

### Prerequisites

- Python 3.8 or higher
- Rust toolchain (for building from source)
- maturin (for building Python extensions)

### Install from source

```bash
# Install maturin
pip install maturin

# Clone the repository
git clone https://github.com/cool-japan/voirs
cd voirs/crates/voirs-recognizer

# Build and install the Python package
maturin develop --features python
```

### Install from PyPI (when published)

```bash
pip install voirs-recognizer
```

## Quick Start

```python
import voirs_recognizer

# Load audio file
audio = voirs_recognizer.load_audio_file("speech.wav")

# Create ASR system
recognizer = voirs_recognizer.VoiRSRecognizer()

# Perform speech recognition
result = recognizer.recognize(audio)
print(f"Transcript: {result.text()}")
print(f"Confidence: {result.confidence():.2f}")

# Analyze audio quality
analyzer = voirs_recognizer.AudioAnalyzer()
analysis = analyzer.analyze(audio)
print(f"SNR: {analysis.get_quality_metric('snr'):.2f} dB")
```

## Features

- **Automatic Speech Recognition (ASR)**: Convert speech to text with confidence scores
- **Audio Analysis**: Extract quality metrics, prosodic features, and speaker characteristics
- **Real-time Processing**: Efficient processing suitable for real-time applications
- **Multi-language Support**: Support for multiple languages and accents
- **Performance Validation**: Built-in performance monitoring and optimization tools

## API Reference

### Core Classes

#### `VoiRSRecognizer`
Main class for speech recognition.

```python
recognizer = voirs_recognizer.VoiRSRecognizer(config=None)
result = recognizer.recognize(audio)
```

#### `AudioAnalyzer`
Class for audio analysis and feature extraction.

```python
analyzer = voirs_recognizer.AudioAnalyzer(config=None)
analysis = analyzer.analyze(audio)
```

#### `AudioBuffer`
Container for audio data.

```python
# Create from samples
audio = voirs_recognizer.AudioBuffer(samples, sample_rate)

# Load from file
audio = voirs_recognizer.load_audio_file("speech.wav")
```

### Configuration Classes

#### `ASRConfig`
Configuration for speech recognition.

```python
config = voirs_recognizer.ASRConfig(
    language="en-US",
    word_timestamps=True,
    confidence_threshold=0.5
)
```

#### `AudioAnalysisConfig`
Configuration for audio analysis.

```python
config = voirs_recognizer.AudioAnalysisConfig(
    quality_metrics=True,
    prosody_analysis=True,
    speaker_analysis=True
)
```

### Result Classes

#### `RecognitionResult`
Contains speech recognition results.

```python
print(result.text())              # Transcribed text
print(result.confidence())        # Confidence score
print(result.language())          # Detected language
print(result.word_timestamps())   # Word-level timestamps
```

#### `AudioAnalysisResult`
Contains audio analysis results.

```python
quality = analysis.quality_metrics()    # Quality metrics
prosody = analysis.prosody_features()   # Prosodic features
speaker = analysis.speaker_features()   # Speaker characteristics
```

### Utility Functions

#### `load_audio_file(path, sample_rate=None)`
Load audio from file.

#### `confidence_to_label(confidence)`
Convert confidence score to human-readable label.

#### `create_recognizer(language, word_timestamps, confidence_threshold)`
Create recognizer with common configuration.

#### `create_analyzer(quality_metrics, prosody_analysis, speaker_analysis)`
Create analyzer with common configuration.

## Examples

### Basic Recognition

```python
import voirs_recognizer

# Load audio
audio = voirs_recognizer.load_audio_file("speech.wav")

# Create recognizer
recognizer = voirs_recognizer.create_recognizer(
    language="en-US",
    word_timestamps=True,
    confidence_threshold=0.5
)

# Recognize speech
result = recognizer.recognize(audio)
print(f"Text: {result.text()}")
print(f"Confidence: {result.confidence():.2f}")

# Show word timestamps
for word in result.word_timestamps():
    print(f"'{word.word()}' [{word.start():.2f}s - {word.end():.2f}s]")
```

### Audio Analysis

```python
import voirs_recognizer

# Load audio
audio = voirs_recognizer.load_audio_file("speech.wav")

# Create analyzer
analyzer = voirs_recognizer.create_analyzer(
    quality_metrics=True,
    prosody_analysis=True,
    speaker_analysis=True
)

# Analyze audio
analysis = analyzer.analyze(audio)

# Show quality metrics
quality = analysis.quality_metrics()
for metric, value in quality.items():
    print(f"{metric}: {value:.3f}")

# Show prosodic features
prosody = analysis.prosody_features()
print(f"Pitch mean: {prosody.get('pitch_mean', 0):.1f} Hz")
print(f"Speaking rate: {prosody.get('speaking_rate', 0):.1f} words/s")
```

### Performance Validation

```python
import voirs_recognizer
import time

# Load audio
audio = voirs_recognizer.load_audio_file("speech.wav")

# Create recognizer
recognizer = voirs_recognizer.VoiRSRecognizer()

# Measure processing time
start_time = time.time()
result = recognizer.recognize(audio)
processing_time = time.time() - start_time

# Validate performance
validator = voirs_recognizer.PerformanceValidator()
rtf, rtf_passed = validator.validate_rtf(audio, processing_time)

print(f"RTF: {rtf:.3f} ({'PASS' if rtf_passed else 'FAIL'})")
print(f"Processing time: {processing_time:.3f}s")
print(f"Audio duration: {audio.duration():.3f}s")
```

## Error Handling

```python
import voirs_recognizer

try:
    # Load audio
    audio = voirs_recognizer.load_audio_file("speech.wav")
    
    # Create recognizer
    recognizer = voirs_recognizer.VoiRSRecognizer()
    
    # Recognize speech
    result = recognizer.recognize(audio)
    
    print(f"Success: {result.text()}")
    
except Exception as e:
    print(f"Error: {e}")
```

## Development

### Building from Source

```bash
# Install development dependencies
pip install maturin pytest

# Build in development mode
maturin develop --features python

# Run tests
python -m pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under MIT OR Apache-2.0.

## Support

- GitHub Issues: https://github.com/cool-japan/voirs/issues
- Documentation: https://github.com/cool-japan/voirs
- Repository: https://github.com/cool-japan/voirs