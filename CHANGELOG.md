# Changelog

All notable changes to VoiRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.1] - 2025-09-21

### Initial Alpha Release

This is the first public alpha release of VoiRS (Voice Synthesis in Rust), a cutting-edge Text-to-Speech / Speech Recognition framework built entirely in Rust.

#### 🎯 Added

**Core Architecture:**
- Complete modular pipeline architecture (Text → G2P → Acoustic → Vocoder → Audio)
- Workspace-based crate organization with 14+ specialized components
- Unified public API through `voirs-sdk` crate

**Components:**
- **voirs-g2p**: Grapheme-to-phoneme conversion with rule-based backend
- **voirs-acoustic**: Neural acoustic models with VITS implementation
- **voirs-vocoder**: Neural vocoders with HiFi-GAN support
- **voirs-dataset**: Audio dataset loading and preprocessing utilities
- **voirs-sdk**: High-level unified API for easy integration
- **voirs-cli**: Command-line interface for synthesis operations
- **voirs-ffi**: Foreign function interface bindings
- **voirs-recognizer**: Speech recognition capabilities
- **voirs-evaluation**: Quality assessment and benchmarking tools
- **voirs-feedback**: User feedback and quality monitoring
- **voirs-emotion**: Emotional speech synthesis control
- **voirs-cloning**: Voice cloning and adaptation features
- **voirs-conversion**: Voice conversion between speakers
- **voirs-singing**: Singing voice synthesis capabilities
- **voirs-spatial**: 3D spatial audio positioning

**Features:**
- Real-time text-to-speech synthesis
- Streaming audio generation with low latency
- Comprehensive example collection (50+ examples)
- Multi-platform support (CPU/GPU backends)
- SSML markup support for advanced prosody control
- Voice cloning and adaptation capabilities
- Emotional speech synthesis
- Spatial audio positioning
- Batch processing utilities
- Production-ready error handling

**Development & Testing:**
- Comprehensive test suite with integration tests
- Benchmarking and performance evaluation framework
- CI/CD pipeline setup
- Code quality tools (clippy, formatting)
- Documentation with examples and tutorials

#### 🔧 Technical Details

**Performance:**
- Optimized for real-time synthesis (≤ 0.3× RTF on consumer CPUs)
- Memory-efficient implementation with streaming support
- GPU acceleration support (CUDA/Metal backends)

**Quality:**
- Neural models achieving high naturalness scores
- Support for multiple voice types and styles
- Advanced prosody and emotion control

**Security:**
- Memory-safe Rust implementation
- Secure consent management for voice cloning
- Privacy protection features

#### 🚨 Known Issues

**Dependencies:**
- 3 non-critical security advisories in transitive dependencies:
  - RSA timing sidechannel (medium) - from optional database features
  - slice-ring-buffer double-free - from optional audio codecs
  - time crate segfault (medium) - from optional audio interface
- 5 warnings for unmaintained dependencies (non-core features)

**Limitations:**
- Alpha quality - APIs may change in future releases
- Limited model zoo (production models coming in beta)
- Some advanced features are experimental
- Documentation improvements needed

#### 📦 Breaking Changes

N/A - Initial release

#### 🛡️ Security

- All crates use workspace version management
- Dual-licensed under MIT/Apache-2.0
- No embedded secrets or sensitive data
- Secure by default configuration

---

## [Unreleased]

### Planned for Beta (0.1.0-beta.1)
- Production-quality pre-trained models
- Enhanced GPU acceleration
- WebAssembly optimization
- Performance improvements
- API stabilization
- Comprehensive documentation