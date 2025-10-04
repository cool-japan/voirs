# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VoiRS is a pure-Rust neural speech synthesis (TTS) framework with modular architecture. Current version: **0.1.0-alpha.2**.

### Core Architecture

VoiRS follows a pipeline architecture:
```
Text Input → G2P → Acoustic Model → Vocoder → Audio Output
```

Each stage is implemented as a separate crate in the workspace, allowing independent development and optional feature inclusion.

## Workspace Structure

VoiRS uses Cargo workspaces with strict workspace dependency management. All version numbers and common dependencies are defined in the root `Cargo.toml` under `[workspace.dependencies]` and `[workspace.package]`.

### Core Pipeline Crates
- **voirs-g2p**: Grapheme-to-Phoneme conversion with multiple backends (Phonetisaurus, OpenJTalk, Neural)
- **voirs-acoustic**: Neural acoustic models (VITS, FastSpeech2) converting phonemes to mel spectrograms
- **voirs-vocoder**: Neural vocoders (HiFi-GAN, DiffWave) converting mel spectrograms to waveforms
- **voirs-dataset**: Dataset loading, preprocessing, and training data utilities

### Advanced Feature Crates
- **voirs-emotion**: Multi-dimensional emotion control and prosody manipulation
- **voirs-cloning**: Voice cloning with few-shot learning, cross-lingual support, and ethical safeguards
- **voirs-singing**: Singing synthesis with MusicXML/MIDI support and breath modeling
- **voirs-spatial**: 3D spatial audio with HRTF, binaural rendering, and VR/AR integration
- **voirs-conversion**: Real-time voice conversion with zero-shot capabilities

### Integration Crates
- **voirs-recognizer**: Speech recognition (Whisper, DeepSpeech, Wav2Vec2) with forced alignment
- **voirs-evaluation**: Quality metrics, MOS prediction, A/B testing framework
- **voirs-feedback**: Real-time feedback systems with adaptive learning and progress tracking

### Public API Crates
- **voirs-sdk**: Unified high-level API exposing all features through a consistent interface
- **voirs-cli**: Command-line tool (`voirs` binary) for synthesis, voice management, and utilities
- **voirs-ffi**: Foreign Function Interface with C/Python/Node.js bindings

## Build Commands

### Basic Build and Test
```bash
# Build entire workspace (CPU-only)
cargo build --release

# Build with GPU acceleration
cargo build --release --features gpu

# Build with all features
cargo build --release --all-features

# Run tests with nextest (preferred)
cargo nextest run --no-fail-fast

# Run all tests including doc tests
cargo test --all-features

# Run benchmarks
cargo bench
```

### Feature-Specific Testing
```bash
# Test specific crate
cargo nextest run -p voirs-acoustic

# Test with specific features
cargo test --features "emotion,cloning,conversion"

# Test GPU features (requires CUDA)
cargo test --features gpu

# Test WASM build
cargo build --target wasm32-unknown-unknown --release
```

### Code Quality
```bash
# Format code
cargo fmt --all

# Check formatting
cargo fmt --all -- --check

# Lint with clippy (strict, no warnings allowed)
cargo clippy --all-targets --all-features -- -D warnings

# Security audit
cargo audit

# Check all crates compile
cargo check --all-features
```

### Running Examples
```bash
# Run specific example
cargo run --example simple_synthesis --features emotion

# Run CLI tool
cargo run -p voirs-cli -- synth "Hello world" output.wav

# List available voices
cargo run -p voirs-cli -- voices list
```

## Development Guidelines

### Workspace Policy

**CRITICAL RULES:**

1. **Version Control**:
   - All crate versions MUST use `version.workspace = true` in individual `Cargo.toml` files
   - NEVER specify version numbers in individual crate `Cargo.toml` files
   - All version numbers are managed centrally in root `Cargo.toml` under `[workspace.package]`

2. **Common Package Metadata**:
   - Use `.workspace = true` for: `edition`, `authors`, `license`, `repository`, `homepage`, `rust-version`
   - These are shared across all subcrates via workspace inheritance

3. **Keywords and Categories (EXCEPTION TO WORKSPACE RULE)**:
   - **❌ WRONG**: `keywords.workspace = true` (DO NOT USE)
   - **✅ CORRECT**: Each subcrate MUST define its own specific keywords
   - **❌ WRONG**: `categories.workspace = true` (DO NOT USE for most subcrates)
   - **✅ CORRECT**: Each subcrate should define its own specific categories
   - **Rationale**: Each subcrate serves different purposes and needs targeted keywords/categories for discoverability

4. **Dependency Management**:
   - Use `.workspace = true` for all shared dependencies (tokio, serde, candle-core, etc.)
   - NEVER specify versions directly in individual crate dependencies (except for crate-specific dependencies not in workspace)
   - All dependency versions are managed centrally in root `Cargo.toml` under `[workspace.dependencies]`

**Example Correct Subcrate Cargo.toml:**
```toml
[package]
name = "voirs-g2p"
version.workspace = true           # ✅ Use workspace
edition.workspace = true           # ✅ Use workspace
authors.workspace = true           # ✅ Use workspace
license.workspace = true           # ✅ Use workspace
repository.workspace = true        # ✅ Use workspace
homepage.workspace = true          # ✅ Use workspace
rust-version.workspace = true      # ✅ Use workspace

# ✅ CORRECT - Define unique keywords per subcrate
keywords = ["voirs", "g2p", "phoneme", "text-processing", "tts"]

# ✅ CORRECT - Define appropriate categories per subcrate
categories = ["text-processing", "algorithms", "science"]

description = "Grapheme-to-Phoneme conversion for VoiRS speech synthesis"
documentation = "https://docs.rs/voirs-g2p"

[dependencies]
tokio.workspace = true             # ✅ Use workspace
serde.workspace = true             # ✅ Use workspace
# ...
```

### Code Quality Standards
- **Zero Warnings Policy**: All code must compile without warnings. Use `#[allow(clippy::...)]` sparingly with justification
- **Line Length**: Single files should not exceed 2000 lines. Refactor into modules if approaching this limit
- **Naming Conventions**: Use `snake_case` for variables/functions, `PascalCase` for types/traits, `SCREAMING_SNAKE_CASE` for constants
- **Test Coverage**: Target 90%+ code coverage for all crates
- **Documentation**: All public APIs must have doc comments with examples

### Testing Requirements
- **Property-Based Testing**: Use `proptest` for edge case validation and robustness testing
- **Integration Tests**: Place in `tests/` directory, use `tempfile` with `std::env::temp_dir()` for file operations
- **Benchmarks**: Use `criterion` for performance benchmarks in `benches/` directory
- **No Hardcoded Paths**: Always use temporary directories or project-relative paths in tests

### Error Handling
- Each crate defines its own error type (e.g., `G2pError`, `AcousticError`)
- Use `thiserror` for error types with descriptive messages
- Provide diagnostic context where possible (see `G2pDiagnosticContext` pattern)
- Return `Result<T>` types, avoid unwrapping except in tests

### Async/Concurrency
- Use `tokio` runtime (version 1.47.1+) with `features = ["full"]`
- Async trait methods use `async-trait` crate
- Prefer `tokio::spawn` for concurrent tasks
- Use `Arc<Mutex<T>>` or `Arc<RwLock<T>>` for shared state

### Dependency Policy
- **Latest Versions**: Always use latest stable versions available on crates.io (see CLAUDE.md in ~/.claude/ for "Latest crates policy")
- **Candle ML Framework**: Use `candle-core`/`candle-nn` for neural network operations

### SciRS2 Integration Policy

**VoiRS follows the SciRS2 ecosystem's layered abstraction architecture.** See `~/work/scirs/SCIRS2_POLICY.md` (v3.0.0) for complete policy details.

**Core Principle:** Only `scirs2-core` may use external dependencies directly. All VoiRS crates MUST use SciRS2-Core abstractions.

**CRITICAL RULES:**

1. **NEVER use these crates directly** in VoiRS code (use SciRS2-Core abstractions instead):
   - ❌ `rand`, `rand_distr` → ✅ Use `scirs2_core::random::*`
   - ❌ `ndarray`, `ndarray-*` → ✅ Use `scirs2_core::ndarray::*`
   - ❌ `num_complex`, `num-traits` → ✅ Use `scirs2_core::numeric::*`
   - ❌ `rayon` → ✅ Use `scirs2_core::parallel_ops::*`
   - ❌ `nalgebra` (for linear algebra) → ✅ Use `scirs2_core::linalg::*`

2. **Required Imports Pattern:**
   ```rust
   // ✅ CORRECT - Always use SciRS2-Core abstractions
   use scirs2_core::random::*;           // Complete rand + rand_distr functionality
   use scirs2_core::ndarray::*;          // Complete ndarray ecosystem + macros (array!, s!, azip!)
   use scirs2_core::numeric::*;          // Complex, Float, Zero, One, Num, etc.
   use scirs2_core::parallel_ops::*;     // Parallel processing (Rayon abstractions)
   use scirs2_core::simd_ops::SimdUnifiedOps;  // SIMD operations

   // ❌ FORBIDDEN - Never import external crates directly
   use rand::Rng;
   use ndarray::{Array1, Array2};
   use num_complex::Complex;
   use rayon::prelude::*;
   ```

3. **Workspace Dependencies:**
   ```toml
   [workspace.dependencies]
   # ✅ REQUIRED - SciRS2 ecosystem crates (RC.1)
   scirs2-core = { version = "0.1.0-rc.1", features = ["array", "random", "simd", "parallel"] }
   scirs2-fft = "0.1.0-rc.1"

   # ❌ REMOVED - These dependencies are NO LONGER in workspace
   # rand, ndarray, num-complex, rayon, nalgebra
   # Use scirs2-core abstractions instead
   ```

4. **For Simple Random Needs:**
   - If you only need basic random number generation (not distributions), use `fastrand` (workspace dependency)
   - For any statistical distributions or complex random operations, use `scirs2_core::random::*`

5. **Performance Benefits:**
   - Automatic SIMD optimizations for audio signal processing (AVX2, AVX512, NEON)
   - Unified platform detection and optimization selection
   - GPU kernel access for deep learning inference
   - Consistent APIs across all cool-japan projects
   - Type safety: Prevents mixing incompatible dependency versions

6. **This Policy Applies To:**
   - All source code in `src/` directories
   - All tests in `tests/` directories
   - All examples in `examples/` directories
   - All benchmarks in `benches/` directories
   - **NO EXCEPTIONS** - All code must use SciRS2-Core abstractions

7. **Compliance Status:**
   - ✅ Workspace Cargo.toml: Prohibited dependencies removed
   - ✅ Policy Documentation: Updated to v2.0.0 (RC.1)
   - ⏳ Subcrate Migration: In progress (see SCIRS2_INTEGRATION_POLICY.md)

**See Also:**
- Complete policy: `~/work/scirs/SCIRS2_POLICY.md` (v3.0.0)
- VoiRS-specific integration guide: `SCIRS2_INTEGRATION_POLICY.md` (v2.0.0)
- Migration checklist: See SCIRS2_INTEGRATION_POLICY.md

## Critical Architecture Details

### Feature Flag Architecture
VoiRS uses extensive feature flags for optional functionality:
- **GPU Support**: `gpu` feature enables CUDA/Metal acceleration across acoustic and vocoder crates
- **ONNX Runtime**: `onnx` feature enables ONNX model inference
- **Advanced Features**: `emotion`, `cloning`, `conversion`, `singing`, `spatial` enable respective crates
- **Complete Feature Set**: `full` or `all-features` enables everything including GPU and all advanced features

Feature propagation follows workspace hierarchy: enabling `gpu` in root enables it in all sub-crates that support it.

### Pipeline Integration Pattern
Crates integrate through shared trait definitions:
- `G2p` trait in voirs-g2p defines phoneme conversion interface
- `AcousticModel` trait in voirs-acoustic defines mel spectrogram generation
- `Vocoder` trait in voirs-vocoder defines waveform synthesis
- voirs-sdk orchestrates these through `VoirsPipeline` and `VoirsPipelineBuilder`

### Memory Management
- Use `Arc<T>` for shared ownership across async tasks
- Streaming synthesis uses chunk-based processing to minimize memory footprint
- Large models use memory-mapped files via `memmap2` where possible
- Target: <2GB memory for typical synthesis workloads

### Performance Expectations
- **Real-Time Factor (RTF)**: Target <0.1× (current: ~0.25× on CPU)
- **Latency**: Target <100ms for streaming synthesis (current: ~200ms)
- **Quality**: MOS score 4.4+ for production voices
- Use `criterion` benchmarks to track performance regressions

## Common Development Tasks

### Adding a New Feature to Existing Crate
1. Implement feature logic in `src/` directory
2. Add feature flag to crate's `Cargo.toml` if optional
3. Add comprehensive tests in `tests/` directory
4. Add benchmarks if performance-critical
5. Update crate-level documentation
6. Propagate feature flag to voirs-sdk if public-facing
7. Run `cargo clippy` and `cargo test --all-features`

### Creating a New Crate in Workspace
1. Create crate: `cargo new --lib crates/voirs-newfeature`
2. Update root `Cargo.toml` workspace members list
3. Use workspace dependencies: `dependency.workspace = true`
4. Use workspace package metadata: `version.workspace = true`
5. Define unique keywords/categories for the crate
6. Add to voirs-sdk dependencies if part of public API
7. Update this CLAUDE.md with architectural notes

### Running Comprehensive Quality Checks
```bash
# Full quality check sequence
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo nextest run --all-features --no-fail-fast
cargo bench --no-run  # Verify benchmarks compile
cargo doc --all-features --no-deps
```

### Debugging Test Failures
```bash
# Run specific test with output
cargo nextest run test_name -- --nocapture

# Run with backtrace
RUST_BACKTRACE=1 cargo nextest run test_name

# Run with detailed logging
RUST_LOG=debug cargo nextest run test_name
```

## Integration with cool-japan Ecosystem

VoiRS integrates with other cool-japan projects:
- **SciRS2** (`~/work/scirs/`): Advanced DSP operations and signal processing
- **NumRS2** (`~/work/numrs/`): High-performance linear algebra operations
- **TrustformeRS** (`~/work/trustformers/`): Transformer models and LLM integration
- **ToRSh** (`~/work/torsh/`): PyTorch-like tensor operations

When implementing features requiring these capabilities, check reference implementations in these projects.

## Platform Support

VoiRS targets multiple platforms with varying feature support:
- **Linux**: Full support including GPU (CUDA)
- **macOS**: Full support including GPU (Metal)
- **Windows**: Full support including GPU (CUDA)
- **WebAssembly**: Limited support (CPU-only, no GPU)
- **Mobile** (iOS/Android): Planned via FFI bindings

Cross-platform code should use conditional compilation for platform-specific optimizations.

## Version and Release Notes

Current version is 0.1.0-alpha.2. This is an alpha release with:
- ✅ Core TTS pipeline working and tested
- ✅ Advanced features (emotion, cloning, spatial) implemented
- ✅ CLI tool and examples functional
- 🚧 APIs may change in future alpha/beta versions
- 🚧 Production models and GPU acceleration in progress

When making breaking changes, update version appropriately and document in CHANGELOG.md.