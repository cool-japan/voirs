# Python Module Structure

## Overview

The Python bindings for VoiRS have been refactored into a modular structure to improve maintainability and comply with the 2000-line per file limit.

**Refactoring Date**: 2025-11-29
**Original Size**: 2,254 lines (python.rs)
**New Size**: 159 lines (python.rs) + modular sub-modules
**Test Status**: ✅ All 268 tests passing

## Module Organization

The Python bindings are now organized into logical sub-modules under `src/python/`:

### Core Modules

#### `common.rs` (23 lines)
- **Purpose**: Shared imports and utilities
- **Contents**: Common PyO3 imports, VoiRS SDK types, NumPy integration imports
- **Used by**: All other Python modules

#### `error.rs` (61 lines)
- **Purpose**: Error types and exception handling
- **Classes**:
  - `VoirsErrorInfo`: Structured error information with code, message, details, and suggestions
  - `VoirsException`: Enhanced Python exception with structured error info
- **Features**: Python-friendly error reporting with `__str__` and `__repr__`

#### `metrics.rs` (64 lines)
- **Purpose**: Performance metrics and synthesis results
- **Classes**:
  - `SynthesisMetrics`: Tracks processing time, real-time factor, memory usage, cache hit rate
  - `SynthesisResult`: Combines audio buffer with synthesis metrics
- **Use Case**: Performance monitoring and optimization

### Audio Processing

#### `audio_buffer.rs` (727 lines)
- **Purpose**: Audio buffer management and NumPy integration
- **Class**: `PyAudioBuffer`
- **Features**:
  - Audio sample access as bytes, lists, or NumPy arrays
  - Multi-channel audio support (mono/stereo/multi-channel)
  - Zero-copy NumPy integration (when `numpy` feature enabled)
  - Audio format conversion (WAV, MP3, FLAC)
  - Advanced audio analysis (RMS, peak detection, spectral features)
  - Audio effects (normalization, fade in/out, filtering)
- **NumPy Integration**: Automatic conversion between interleaved and planar formats

#### `analyzer.rs` (114 lines)
- **Purpose**: Advanced audio analysis tools
- **Class**: `PyAudioAnalyzer`
- **Features**:
  - Spectral analysis (FFT, spectral centroid, flux, rolloff)
  - Time-domain analysis (zero-crossing rate, envelope)
  - Energy and loudness measurements
  - NumPy-based batch processing
- **Requirements**: `numpy` feature must be enabled

### Voice and Pipeline

#### `pipeline.rs` (547 lines)
- **Purpose**: Main TTS pipeline wrapper
- **Class**: `VoirsPipeline`
- **Features**:
  - Text-to-speech synthesis (sync and async)
  - Voice discovery and management
  - Model loading and configuration
  - Streaming synthesis support
  - Batch processing
  - Advanced synthesis parameters (prosody, emotion, speaker)
  - GPU acceleration support (when `gpu` feature enabled)
  - Progress callbacks and monitoring

#### `voice.rs` (33 lines)
- **Purpose**: Voice information and metadata
- **Class**: `PyVoiceInfo`
- **Features**:
  - Voice properties (ID, name, language, gender, style)
  - Sample rate and quality information
  - Voice metadata serialization

### Advanced Features

#### `streaming.rs` (96 lines)
- **Purpose**: Real-time streaming audio processing
- **Class**: `PyStreamingProcessor`
- **Features**:
  - Chunk-based audio processing
  - Real-time synthesis streaming
  - Low-latency audio generation
  - NumPy-based chunk processing
  - Streaming configuration (chunk size, overlap)
- **Requirements**: `numpy` feature must be enabled

#### `config.rs` (555 lines)
- **Purpose**: Synthesis configuration and parameters
- **Class**: `PySynthesisConfig`
- **Features**:
  - Comprehensive synthesis parameters
  - Quality and performance trade-offs
  - Voice selection and customization
  - Prosody control (pitch, rate, volume)
  - Emotion and style parameters
  - Output format configuration
  - GPU and acceleration settings
  - Validation and defaults

#### `recognition.rs` (512 lines)
- **Purpose**: Speech recognition integration
- **Classes**:
  - `PyASRModel`: Automatic Speech Recognition model wrapper
  - `PyPhonemeRecognizer`: Phoneme-level recognition
  - `PyRecognitionResult`: Recognition results with confidence scores
  - `PyTranscript`: Transcript with word-level timing
  - `PyAudioAnalysis`: Audio analysis for recognition
  - `PyPerformanceMetrics`: Recognition performance metrics
- **Requirements**: `recognition` feature must be enabled
- **Features**: Integration with voirs-recognizer for ASR and forced alignment

## Main Entry Point

### `python.rs` (159 lines)
- **Purpose**: Feature-gated module wrapper and Python module registration
- **Structure**:
  ```rust
  #[cfg(feature = "python")]
  pub use python_impl::*;

  #[cfg(feature = "python")]
  mod python_impl {
      // Re-export all modules
      pub mod error;
      pub mod metrics;
      pub mod pipeline;
      // ... etc

      #[pymodule]
      fn voirs_ffi(m: &Bound<'_, PyModule>) -> PyResult<()> {
          // Register all Python classes
      }
  }
  ```
- **Responsibilities**:
  - Feature gate management (`python`, `numpy`, `gpu`, `recognition`)
  - PyO3 module registration
  - Public API re-exports
  - Module documentation

## File Size Distribution

All files comply with the 2000-line limit:

| Module | Lines | Purpose |
|--------|-------|---------|
| `python.rs` | 159 | Main entry point and re-exports |
| `audio_buffer.rs` | 727 | Audio processing and NumPy integration |
| `config.rs` | 555 | Synthesis configuration |
| `pipeline.rs` | 547 | TTS pipeline wrapper |
| `recognition.rs` | 512 | Speech recognition (feature-gated) |
| `analyzer.rs` | 114 | Audio analysis tools |
| `streaming.rs` | 96 | Real-time streaming |
| `metrics.rs` | 64 | Performance metrics |
| `error.rs` | 61 | Error handling |
| `voice.rs` | 33 | Voice information |
| `common.rs` | 23 | Shared imports |
| **Total** | **2,891** | Across 11 well-organized files |

## Feature Flags

The Python bindings support conditional compilation based on features:

- **`python`**: Enable Python bindings (default in Cargo.toml features)
- **`numpy`**: Enable NumPy integration (`PyStreamingProcessor`, `PyAudioAnalyzer`)
- **`gpu`**: Enable GPU acceleration support in pipeline
- **`recognition`**: Enable speech recognition classes from voirs-recognizer

## Testing

All 268 existing tests continue to pass after refactoring:

```bash
$ cargo test -p voirs-ffi --lib
test result: ok. 268 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Migration Notes

For developers working on the Python bindings:

1. **Imports**: All modules import from `super::common::*` for shared dependencies
2. **Cross-module references**: Use `super::module_name::Type` (e.g., `super::audio_buffer::PyAudioBuffer`)
3. **Feature gates**: Maintain `#[cfg(feature = "...")]` attributes where appropriate
4. **PyO3 attributes**: Keep `#[pyclass]`, `#[pymethods]`, `#[new]`, etc. on relevant types and methods
5. **Module registration**: Add new classes to the `voirs_ffi` function in `python.rs`

## Benefits of Modular Structure

1. **Maintainability**: Logical separation of concerns makes code easier to navigate and modify
2. **Compliance**: All files now under 2000-line limit per project guidelines
3. **Compilation speed**: Smaller files can be compiled in parallel more efficiently
4. **Testing**: Module-level organization makes it easier to add module-specific tests
5. **Feature isolation**: Feature-gated code (numpy, recognition) is clearly separated
6. **Documentation**: Each module has focused documentation on its specific functionality

## Future Enhancements

Potential areas for further improvement:

1. Split large modules if they grow beyond 1500 lines:
   - `audio_buffer.rs` (727 lines) could be split into core and effects
   - `config.rs` (555 lines) could separate validation from configuration
   - `pipeline.rs` (547 lines) could separate sync and async interfaces

2. Add module-level integration tests in `tests/python/`

3. Consider extracting common patterns into shared utilities

4. Add performance benchmarks for cross-module call overhead (should be zero due to inlining)
