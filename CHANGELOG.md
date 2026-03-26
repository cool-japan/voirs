# Changelog

All notable changes to VoiRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-rc.1] - 2026-03-21

### Added
- Comprehensive test suite reaching 9487/9487 tests passing with zero failures
- Full COOLJAPAN Pure Rust Policy compliance across all crates (oxiarc-bzip2, oxifft, oxiblas)
- SciRS2-Core v0.3.4 integration with SIMD, parallel, and random abstractions
- Python bindings version alignment to PEP 440 format (0.1.0rc1) in pyproject.toml files
- Complete workspace dependency management — all internal crates pinned at 0.1.0-rc.1

### Changed
- Upgraded all workspace dependencies to latest stable versions available on crates.io
- Replaced all `bzip2` usage with `oxiarc-bzip2` per COOLJAPAN policy
- Stabilized public APIs across voirs-g2p, voirs-acoustic, voirs-vocoder, voirs-sdk
- Improved CUDA/GPU device handling on macOS to avoid panics when CUDA is unavailable
- WASM feature gates hardened to use `#[cfg(all(feature = "wasm", target_arch = "wasm32"))]`

### Fixed
- Integer overflow in conformer CTC decoding (`token_id as u8` wrapping for token_id > 255)
- Flaky timing tests across evaluation, conversion, CLI, feedback, and recognizer crates (relaxed budgets 4-10x)
- FFI test interference between C API error handling and pyo3 context
- numrs2 dependency updated to crates.io release (0.3.1), patch entry removed
- Zero clippy warnings across all crates and targets with `--all-features`
- Zero rustdoc warnings with `-D warnings` flag

### Technical Improvements
- All files comply with 2000-line refactoring policy (splitrs applied where needed)
- No unwrap() usage in production code paths
- All crates use `.workspace = true` for shared metadata fields
- 100% Pure Rust default feature set — no C/Fortran dependencies in default builds

---

## [0.1.0-beta.1] - 2026-02-26

### Added
- Production-quality pre-trained model support infrastructure
- Comprehensive workspace metadata for crates.io publishing readiness
- `readme` field added to all subcrate Cargo.toml files

### Changed
- API stabilization for beta milestone
- Updated SciRS2-Core integration to version 0.2.0 with improved SIMD and parallel operations
- Updated workspace dependencies to latest versions (reqwest 0.13, bytes security fix)
- Fixed invalid crates.io category in voirs-feedback (`accessibility` -> `development-tools`)
- Added missing `rust-version` workspace inheritance to voirs-feedback and voirs-singing
- Updated examples dependency versions to match workspace version
- Enhanced GPU acceleration support
- WebAssembly optimization improvements
- Performance improvements across the pipeline

### Fixed
- Resolved workspace metadata inconsistencies for crates.io compatibility
- Fixed stale version references in examples/Cargo.toml
- Improved documentation version references

---

## [0.1.0-alpha.3] - 2026-01-29

### Added
- Enhanced CUDA GPU acceleration support across acoustic and vocoder pipelines
- Improved performance optimizations for real-time synthesis
- Additional SIMD optimizations for mel spectrogram computation
- Enhanced error handling and diagnostics across all crates

### Changed
- Refactored multiple modules to comply with 2000-line file policy using splitrs
- Updated SciRS2-Core integration to version 0.1.3 with improved SIMD and parallel operations
- Improved code quality and eliminated additional unwrap() calls following no-unwrap policy
- Enhanced workspace dependency management and version consistency

### Fixed
- Resolved various clippy warnings and lints across the workspace
- Fixed potential edge cases in audio processing pipelines
- Improved test coverage and stability

### Technical Improvements
- Comprehensive codebase refactoring for better maintainability
- Enhanced documentation and inline comments
- Improved trait implementations and abstractions
- Better separation of concerns across modules

---

## [0.1.0-alpha.2] - 2025-10-04

### Added
- DiffWave vocoder training pipeline with gradient-based updates, SafeTensors checkpointing, and improved CLI ergonomics for long-running training runs.
- Automatic IPA Kokoro demo leveraging eSpeak NG plus expanded training documentation covering streaming, batch, and evaluation workflows.

### Changed
- Replaced direct `rand`, `rustfft`, `realfft`, `num_complex`, `rayon`, and `ndarray` usage with `scirs2_*` abstractions (Beta 3) across the workspace to unify the DSP stack.
- Updated installation instructions, quick-start material, and roadmap to reflect the alpha.2 capabilities and training-first workflow.

### Known Issues
- Real FFT planner trait objects and `plan_fft_forward` support remain blocked pending SCIRS2 Beta 4; temporary functional FFT fallbacks ship in this release.
- A handful of `scirs2_core::random` and `parallel_ops` helpers still require upstream improvements—see `MIGRATION_STATUS.md` for active workstreams.

---

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
- Licensed under Apache-2.0
- No embedded secrets or sensitive data
- Secure by default configuration

---

## [Unreleased]

### Planned for 0.1.0-beta.2
- Expanded production-quality pre-trained model zoo
- Further GPU acceleration improvements
- WebAssembly optimization enhancements
- Continued performance improvements
- Comprehensive documentation expansion
- Additional language support