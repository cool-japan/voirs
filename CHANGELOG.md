# Changelog

All notable changes to VoiRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-07

First public release of VoiRS, a pure-Rust neural speech-synthesis (TTS) framework with a modular, pipeline-based architecture.

### Added

#### Architecture
- Modular pipeline architecture: Text → G2P → Acoustic Model → Vocoder → Audio Output
- Cargo workspace organizing 15 specialized crates with strict workspace dependency management
- Unified high-level API through the `voirs-sdk` crate orchestrating the full pipeline

#### Workspace Crates
- **voirs-g2p**: Grapheme-to-phoneme conversion with multiple backends (Phonetisaurus, OpenJTalk, Neural)
- **voirs-acoustic**: Neural acoustic models (VITS, FastSpeech2) converting phonemes to mel spectrograms
- **voirs-vocoder**: Neural vocoders (HiFi-GAN, DiffWave) converting mel spectrograms to waveforms, including a training pipeline
- **voirs-dataset**: Dataset loading, preprocessing, and training-data utilities
- **voirs-sdk**: Unified high-level API exposing all features through a consistent interface
- **voirs-cli**: Command-line tool (`voirs` binary) for synthesis, voice management, training, and utilities
- **voirs-ffi**: Foreign Function Interface with C, Python, and Node.js bindings
- **voirs-recognizer**: Speech recognition (Whisper, DeepSpeech, Wav2Vec2) with forced alignment
- **voirs-evaluation**: Quality metrics, MOS prediction, and A/B testing framework
- **voirs-feedback**: Real-time feedback systems with adaptive learning and progress tracking
- **voirs-emotion**: Multi-dimensional emotion control and prosody manipulation
- **voirs-cloning**: Voice cloning with few-shot learning, cross-lingual support, and ethical safeguards
- **voirs-conversion**: Real-time voice conversion with zero-shot capabilities
- **voirs-singing**: Singing synthesis with MusicXML/MIDI support and breath modeling
- **voirs-spatial**: 3D spatial audio with HRTF, binaural rendering, and VR/AR integration

#### Core Features
- Real-time and streaming text-to-speech synthesis with low latency
- SSML markup support for advanced prosody control
- VITS acoustic modeling paired with HiFi-GAN and DiffWave vocoders
- DiffWave vocoder training pipeline with gradient-based updates and SafeTensors checkpointing
- Voice cloning and speaker adaptation
- Multi-dimensional emotion and prosody control
- Singing voice synthesis
- 3D spatial audio positioning
- Real-time voice conversion between speakers
- Speech recognition with forced alignment
- Quality evaluation metrics and benchmarking
- Batch processing utilities and a comprehensive example collection

#### Bindings
- C, Python, and Node.js bindings via `voirs-ffi`

#### Platform Support
- Linux, macOS, and Windows on x86_64 and aarch64 architectures
- Optional GPU acceleration via CUDA (Linux/Windows) and Metal (macOS)
- WebAssembly (wasm32) target support (CPU-only)

### Notes
- 100% Pure Rust default feature set — no C/C++/Fortran dependencies in default builds
- Full COOLJAPAN Pure-Rust policy compliance: compression via oxiarc-bzip2, FFT via oxifft, and a rustls-based TLS stack
- SciRS2-Core integration providing SIMD, parallel, random, and ndarray abstractions across the workspace
- Comprehensive test suite passing across all crates with zero clippy warnings
- Licensed under Apache-2.0

### Security
- Memory-safe Rust implementation throughout
- Secure consent management for voice cloning
- No embedded secrets or sensitive data; licensed under Apache-2.0

[0.1.0]: https://github.com/cool-japan/voirs/releases/tag/v0.1.0
