# VoiRS Evaluation - Build System Documentation

This document describes the comprehensive cross-platform build system for the VoiRS Evaluation crate.

## Overview

The VoiRS Evaluation project includes multiple build automation tools to support development across different platforms and preferences:

- **`build.sh`** - Unix shell script (Linux, macOS, FreeBSD)
- **`build.ps1`** - PowerShell script (Windows)
- **`build.py`** - Cross-platform Python script
- **`Makefile`** - Traditional Make-based automation
- **`justfile`** - Modern command runner using `just`

## Quick Start

Choose the build tool that works best for your environment:

### Unix/Linux/macOS
```bash
# Using shell script
./build.sh ci

# Using Make
make ci

# Using just (if installed)
just ci

# Using Python
python3 build.py ci
```

### Windows
```powershell
# Using PowerShell
.\build.ps1 ci

# Using Python
python build.py ci
```

## Build Tools Comparison

| Tool | Platform Support | Dependencies | Best For |
|------|------------------|--------------|----------|
| `build.sh` | Unix-like systems | bash, standard Unix tools | Linux/macOS development |
| `build.ps1` | Windows | PowerShell 5.1+ | Windows development |
| `build.py` | Cross-platform | Python 3.6+ | CI/CD, cross-platform |
| `Makefile` | Cross-platform | make | Traditional workflows |
| `justfile` | Cross-platform | just command runner | Modern development |

## Common Commands

All build tools support these common commands:

### Build Commands
- `build` - Build the project in debug mode
- `build --release` - Build in release mode
- `build --all-features` - Build with all features enabled
- `build --features <features>` - Build with specific features

### Testing Commands
- `test` - Run all tests
- `test --release` - Run tests in release mode
- `bench` - Run benchmarks

### Quality Commands
- `check` - Run code quality checks (format + clippy)
- `docs` - Generate documentation
- `clean` - Clean build artifacts

### Utility Commands
- `install` - Install the package
- `python` - Build Python bindings
- `ci` - Run full CI pipeline

## Detailed Usage

### Shell Script (`build.sh`)

The shell script provides comprehensive Unix-like system support:

```bash
# Show help
./build.sh help

# Build with specific features
./build.sh build --features python

# Run full CI pipeline
./build.sh ci

# Build and open documentation
./build.sh docs --open
```

**Features:**
- Platform detection (macOS, Linux, FreeBSD)
- Architecture detection (x86_64, aarch64, armv7)
- Colored output
- Dependency checking
- Environment setup

### PowerShell Script (`build.ps1`)

The PowerShell script provides Windows-specific optimizations:

```powershell
# Show help
.\build.ps1 help

# Build in release mode with all features
.\build.ps1 build -Release -AllFeatures

# Run tests with specific features
.\build.ps1 test -Features python

# Generate and open documentation
.\build.ps1 docs -Open
```

**Features:**
- Windows-specific dependency detection
- Visual Studio Build Tools integration
- PowerShell parameter validation
- Colored output support
- Error handling

### Python Script (`build.py`)

The Python script offers maximum cross-platform compatibility:

```bash
# Show help
python3 build.py --help

# Build with all features
python3 build.py build --all-features --release

# Run CI pipeline
python3 build.py ci

# Build Python bindings
python3 build.py python
```

**Features:**
- True cross-platform support
- Detailed logging and error reporting
- Platform-specific optimizations
- Progress tracking
- Extensible architecture

### Makefile

Traditional Make-based automation with extensive targets:

```bash
# Show available targets
make help

# Build and test
make build test

# Run CI pipeline
make ci

# Development cycle
make dev

# Build with specific features
make build FEATURES=python
```

**Features:**
- Compatible with GNU Make and BSD Make
- Parallel execution support
- Variable customization
- Platform detection
- Extensive target library

### Justfile

Modern command runner with `just`:

```bash
# List available recipes
just --list

# Quick development cycle
just dev

# Build with profiling
just profile-build profile

# Watch files and run tests
just watch test

# Setup development environment
just dev-setup
```

**Features:**
- Modern syntax and features
- Built-in parallelism
- Shell script integration
- Platform-specific recipes
- Advanced automation features

## Environment Variables

All build tools respect these environment variables:

- `CARGO_TARGET_DIR` - Override target directory (default: `target`)
- `RUST_BACKTRACE` - Enable Rust backtrace (set to `1` by scripts)
- `RUSTFLAGS` - Additional Rust compiler flags

Platform-specific variables:
- macOS: `SDKROOT` - SDK path (auto-detected)
- Linux: `LD_LIBRARY_PATH` - Library search path
- Windows: `VCINSTALLDIR` - Visual C++ installation

## Feature Flags

The VoiRS Evaluation crate supports these feature flags:

- `quality` - Quality evaluation metrics (default)
- `pronunciation` - Pronunciation assessment (default)
- `comparison` - Comparative analysis (default)
- `perceptual` - Perceptual evaluation metrics
- `python` - Python bindings (requires PyO3)
- `all-metrics` - Enable all metric types

Example usage:
```bash
# Build with Python bindings
./build.sh build --features python

# Build with all features
make build-all

# Test with specific features
just test --features "python,perceptual"
```

## Platform-Specific Notes

### macOS
- Requires Xcode Command Line Tools
- Uses SDK path detection for proper compilation
- Supports both Intel and Apple Silicon

### Linux
- Requires standard build tools (gcc, pkg-config)
- Supports various distributions
- ARM64 and x86_64 architectures

### Windows
- Requires Visual Studio Build Tools or Visual Studio
- PowerShell 5.1+ recommended
- Supports both x64 and ARM64

### FreeBSD
- Uses BSD Make compatibility
- Standard Unix toolchain required

## CI/CD Integration

### GitHub Actions
```yaml
- name: Build and test
  run: python3 build.py ci
```

### GitLab CI
```yaml
script:
  - ./build.sh ci
```

### Jenkins
```groovy
sh 'python3 build.py ci'
```

## Development Workflows

### Quick Development
```bash
# Format, lint, and test
just dev

# Watch files and test on changes
just watch test

# Format and check before commit
make check
```

### Release Preparation
```bash
# Run full CI pipeline
./build.sh ci

# Build optimized release
just profile-build perf

# Generate documentation
make docs
```

### Python Bindings
```bash
# Setup Python environment
pip install maturin

# Build Python wheels
python3 build.py python

# Test Python bindings
python3 -c "import voirs_evaluation"
```

## Troubleshooting

### Common Issues

1. **Rust not found**
   - Install Rust from https://rustup.rs/
   - Ensure `cargo` and `rustc` are in PATH

2. **Build failures on Windows**
   - Install Visual Studio Build Tools
   - Use x64 Native Tools Command Prompt

3. **Python bindings fail**
   - Install Python development headers
   - Install maturin: `pip install maturin`

4. **Permission denied on Unix**
   - Make scripts executable: `chmod +x build.sh`

5. **Make not found**
   - Install build tools for your platform
   - Use alternative build scripts

### Debug Mode

Enable verbose output for debugging:

```bash
# Shell script
RUST_BACKTRACE=full ./build.sh build

# Python script
python3 build.py build --verbose

# Make with debug
make build CARGO_FLAGS="--verbose"
```

## Contributing

When adding new build functionality:

1. Update all build scripts consistently
2. Test on multiple platforms
3. Update this documentation
4. Add appropriate error handling
5. Follow existing conventions

## Performance Optimization

### Build Performance
- Use release mode for final builds: `--release`
- Enable LTO for size: `RUSTFLAGS="-C lto=fat"`
- Use target-cpu native: `RUSTFLAGS="-C target-cpu=native"`

### Development Performance
- Use incremental compilation (default in debug)
- Leverage build caching with consistent target directory
- Use parallel builds: `cargo build -j $(nproc)`

## Security Considerations

- Always verify script integrity before execution
- Use official Rust toolchain installations
- Keep dependencies updated with `cargo update`
- Run security audits with `cargo audit`

## License

This build system is part of the VoiRS project and follows the same licensing terms.

---

For more information, see:
- [Rust Documentation](https://doc.rust-lang.org/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [VoiRS Documentation](../README.md)