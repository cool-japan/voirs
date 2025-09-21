# VoiRS Build Tools

This directory contains the build and CI/CD infrastructure for the VoiRS project.

## Files Overview

### Build System
- **`enhanced_build_system.py`** - Advanced Python build system with parallel execution, testing, and validation
- **`build_examples.sh`** - Bash script for comprehensive build and test operations
- **`build_config.toml`** - Configuration file for build system settings and test parameters

### Docker Infrastructure
- **`../Dockerfile.ci`** - Multi-stage Docker image for CI/CD pipelines with development, testing, benchmarking, and production environments

### GitHub Actions
- **`../.github/workflows/examples.yml`** - Comprehensive CI/CD workflow for multi-platform builds, testing, and deployment

### Makefile Integration
- **`../examples/Makefile`** - Enhanced Makefile with convenient targets for building, testing, and validation

## Quick Start

### Local Development

From the `examples/` directory:

```bash
# Build all examples
make build

# Run tests
make test

# Quick development cycle
make quick

# Full comprehensive testing
make full

# Clean artifacts
make clean
```

### Using Python Build System Directly

```bash
# Basic build
python3 ../tools/enhanced_build_system.py --build-only

# Run tests with custom settings
python3 ../tools/enhanced_build_system.py \
    --parallel 8 \
    --timeout 120 \
    --examples "*benchmark*" \
    --report build_report.json

# CI mode
python3 ../tools/enhanced_build_system.py --ci
```

### Using Bash Script

```bash
# Quick mode
../tools/build_examples.sh --quick

# Full validation
../tools/build_examples.sh --full

# Specific category
../tools/build_examples.sh --category performance
```

## Configuration

### Build Profiles

Edit `build_config.toml` to customize:

- **development** - Fast builds for development
- **ci** - Optimized for CI/CD pipelines  
- **production** - Full validation for production releases
- **quick** - Quick testing for development
- **full** - Comprehensive testing

### Example Categories

- **beginner** - Basic examples (hello_world, simple_synthesis)
- **intermediate** - Medium complexity (streaming, emotion control)
- **advanced** - Complex features (voice cloning, spatial audio)
- **production** - Production-ready examples
- **testing** - Test and validation examples

## Docker Usage

### Build CI Image

```bash
docker build -f Dockerfile.ci -t voirs-ci .
```

### Run Full CI Pipeline

```bash
docker run -v $(pwd):/ci-workspace voirs-ci full
```

### Run Specific Commands

```bash
# Build only
docker run -v $(pwd):/ci-workspace voirs-ci build

# Test only  
docker run -v $(pwd):/ci-workspace voirs-ci test

# Quality checks
docker run -v $(pwd):/ci-workspace voirs-ci quality

# Benchmarks
docker run -v $(pwd):/ci-workspace voirs-ci benchmark
```

## GitHub Actions Integration

The CI/CD pipeline automatically:

1. **Quality Checks** - Code formatting, linting, security audit
2. **Multi-Platform Builds** - Linux, Windows, macOS
3. **Comprehensive Testing** - Parallel test execution by category
4. **Performance Benchmarking** - Optional performance regression detection
5. **Artifact Management** - Build artifacts and test reports
6. **Deployment** - Documentation deployment on successful builds

### Workflow Triggers

- Push to `main`/`develop` branches
- Pull requests
- Scheduled nightly builds (2 AM UTC)
- Manual workflow dispatch

### Manual Triggers

```bash
# Trigger with specific settings
gh workflow run examples.yml \
    -f test_category=performance \
    -f build_profile=production \
    -f run_benchmarks=true
```

## Monitoring and Reports

### Report Generation

All build and test operations generate detailed JSON reports:

- **Build Report** - Compilation status, timing, artifacts
- **Test Report** - Test results, performance metrics, failures
- **Benchmark Report** - Performance measurements, regression detection

### Report Locations

- `examples/build_reports/` - Local build reports
- GitHub Actions Artifacts - CI/CD reports with 30-day retention

### Performance Metrics

The system tracks:
- **RTF (Real-Time Factor)** - Synthesis speed relative to audio duration
- **Memory Usage** - Peak memory consumption during processing
- **Build Times** - Compilation duration for performance tracking
- **Test Success Rates** - Overall test health metrics

## Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Check detailed logs
   make debug-build
   
   # Single example with verbose output
   make debug-test EXAMPLES_PATTERN="hello_world"
   ```

2. **Permission Issues**
   ```bash
   # Fix script permissions
   chmod +x ../tools/*.sh ../tools/*.py
   ```

3. **Missing Dependencies**
   ```bash
   # Install system dependencies (Ubuntu/Debian)
   sudo apt-get install libasound2-dev pkg-config
   
   # Install Rust components
   rustup component add rustfmt clippy
   ```

### Performance Issues

- Use `--parallel N` to control parallel jobs based on system capacity
- Adjust timeout values in `build_config.toml` for slower systems
- Use `--quick` mode during development for faster iteration

### Getting Help

```bash
# Show available make targets
make help

# Show build system options
python3 ../tools/enhanced_build_system.py --help

# Show bash script options  
../tools/build_examples.sh --help
```

## Development Guidelines

### Adding New Examples

1. Add example to `examples/Cargo.toml`
2. Update category mappings in `build_config.toml`
3. Set appropriate timeout and resource limits
4. Test with `make examples-<name>` before committing

### Modifying Build System

1. Test changes locally with various profiles
2. Validate Docker build still works
3. Check GitHub Actions workflow compatibility
4. Update documentation for any new features

### Performance Optimization

- Profile builds with `--verbose` flag to identify bottlenecks
- Use `cargo audit` for security validation
- Monitor memory usage during large batch operations
- Consider CPU affinity for performance-critical tests

## Integration Examples

### VS Code Integration

Add to `.vscode/tasks.json`:
```json
{
    "label": "VoiRS Build",
    "type": "shell",
    "command": "make",
    "args": ["build"],
    "group": "build",
    "options": {
        "cwd": "${workspaceFolder}/examples"
    }
}
```

### IDE Integration

Most IDEs can integrate with the Makefile targets for convenient development workflow.

---

For more information, see the main project documentation and the individual tool help output.