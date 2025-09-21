# Installation

This guide will help you install VoiRS Text-to-Speech synthesis framework on your system.

## System Requirements

### Minimum Requirements
- **Operating System**: Linux, macOS, or Windows
- **RAM**: 4GB (8GB+ recommended for real-time synthesis)
- **CPU**: x86_64 or ARM64 processor
- **Storage**: 3GB free space for models and dependencies
- **Rust**: 1.70+ (if building from source)

### Recommended Requirements
- **RAM**: 16GB+ for optimal performance
- **CPU**: Multi-core processor with SIMD support
- **GPU**: CUDA-compatible GPU or Apple Silicon for acceleration (optional)
- **Audio**: Audio output device for playback testing

## Installation Methods

### Method 1: Using Cargo (Recommended)

Add VoiRS to your `Cargo.toml`:

```toml
[dependencies]
voirs = "0.1.0-alpha.1"

# Or with specific features
voirs = { version = "0.1.0-alpha.1", features = [
    "gpu",           # GPU acceleration support
    "onnx",          # ONNX runtime support
    "recognition",   # Include speech recognition features
    "evaluation",    # Quality evaluation tools
    "feedback",      # User feedback systems
] }
```

### Method 2: From Source

Clone and build from source:

```bash
git clone https://github.com/cool-japan/voirs
cd voirs
cargo build --release --all-features
```

### Method 3: Using the CLI Tool

Install the VoiRS CLI for command-line synthesis:

```bash
cargo install voirs-cli

# Or from source
cd voirs
cargo install --path crates/voirs-cli
```

## Feature Flags

VoiRS provides several optional features:

| Feature | Description | Default |
|---------|-------------|---------|
| `gpu` | CUDA/Metal GPU acceleration | ❌ |
| `onnx` | ONNX runtime support | ❌ |
| `recognition` | Speech recognition capabilities | ❌ |
| `evaluation` | Quality assessment tools | ❌ |
| `feedback` | User feedback systems | ❌ |
| `whisper` | Whisper ASR integration | ❌ |
| `all-asr-models` | All recognition models | ❌ |
| `full` | All available features | ❌ |

### Feature Selection Guide

For **basic synthesis**:
```toml
voirs = "0.1.0-alpha.1"  # Core TTS functionality included
```

For **high-performance applications**:
```toml
voirs = { version = "0.1.0-alpha.1", features = [
    "gpu",           # GPU acceleration
    "onnx",          # Optimized inference
] }
```

For **comprehensive speech applications**:
```toml
voirs = { version = "0.1.0-alpha.1", features = [
    "full",          # All features enabled
] }
```

For **evaluation and testing**:
```toml
voirs = { version = "0.1.0-alpha.1", features = [
    "evaluation",    # Quality metrics
    "feedback",      # User feedback systems
    "recognition",   # Bidirectional speech processing
] }
```

## System Dependencies

### Linux (Ubuntu/Debian)

Install required system libraries:

```bash
sudo apt update
sudo apt install -y libasound2-dev pkg-config
```

For GPU acceleration:
```bash
# CUDA support (NVIDIA)
sudo apt install nvidia-cuda-toolkit

# Or install via NVIDIA's official repositories
```

### macOS

Install dependencies via Homebrew:

```bash
brew install pkg-config
```

GPU acceleration is automatically available on Apple Silicon Macs.

### Windows

Install the following:

1. **Visual Studio Build Tools** with C++ support
2. **pkg-config** via vcpkg or chocolatey:

```powershell
# Using chocolatey
choco install pkgconfiglite

# Or using vcpkg
vcpkg install pkgconf
```

## Verification

Verify your installation:

```bash
# Test the library build
cargo test

# Run a basic synthesis example
cargo run --example simple_synthesis
```

Expected output:
```
🎤 VoiRS Simple Synthesis Example
=================================

✅ Pipeline ready!
📝 Text to synthesize: "Hello, world! This is VoiRS speaking in pure Rust."
✅ Synthesis completed in 1.23 seconds
💾 Saving audio to: output.wav
✅ Audio saved successfully!
...
🎉 Simple synthesis complete!
```

## Troubleshooting

### Common Issues

**"could not find system library 'alsa'"** (Linux):
```bash
sudo apt install libasound2-dev
```

**"linking with 'cc' failed"** (macOS):
```bash
xcode-select --install
```

**"failed to run custom build command for 'some-sys'"**:
- Ensure you have a C compiler installed
- Update your Rust toolchain: `rustup update`

### Performance Issues

If you experience slow performance:

1. **Enable optimizations**: Use `--release` flag when building
2. **Enable SIMD**: Ensure your CPU supports modern instruction sets
3. **Check memory**: Monitor RAM usage during processing
4. **GPU acceleration**: Enable if you have compatible hardware

## Next Steps

Now that you have VoiRS installed, check out:

- [Quick Start Guide](./quick-start.md) - Your first synthesis tasks
- [Examples](../../../examples/) - Comprehensive synthesis examples
- [Performance Tuning](./performance.md) - Optimize for your use case
- [SSML Guide](./ssml.md) - Advanced speech markup
- [Voice Cloning](./voice-cloning.md) - Custom voice creation

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](./troubleshooting.md)
2. Search [GitHub Issues](https://github.com/cool-japan/voirs/issues)
3. Ask in [GitHub Discussions](https://github.com/cool-japan/voirs/discussions)