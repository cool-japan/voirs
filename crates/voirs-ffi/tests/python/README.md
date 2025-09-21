# VoiRS Python Bindings Test Suite

This comprehensive test suite validates the Python bindings for VoiRS speech synthesis.

## Test Structure

```
tests/python/
├── conftest.py              # Pytest configuration and fixtures
├── run_tests.py             # Test runner script
├── unit/                    # Unit tests
│   └── test_pipeline.py     # VoirsPipeline unit tests
├── integration/             # Integration tests
│   └── test_end_to_end.py   # End-to-end workflow tests
└── performance/             # Performance and stress tests
    ├── test_benchmark.py    # Performance benchmarks
    └── test_stress.py       # Stress tests
```

## Prerequisites

- Python 3.7+
- pytest
- VoiRS Python bindings (`voirs_ffi`)
- Optional: numpy (for NumPy integration tests)
- Optional: psutil (for resource monitoring)
- Optional: coverage (for coverage reporting)

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py --verbose

# Run specific test types
python run_tests.py --test-type unit
python run_tests.py --test-type integration
python run_tests.py --test-type performance
```

### Test Types

#### Unit Tests
Test individual components and functions:
```bash
python run_tests.py --test-type unit
```

#### Integration Tests
Test complete workflows and real-world scenarios:
```bash
python run_tests.py --test-type integration
```

#### Performance Tests
Test performance characteristics and benchmarks:
```bash
python run_tests.py --test-type performance
```

#### Stress Tests
Test system behavior under extreme conditions:
```bash
python run_tests.py --test-type stress
```

#### Memory Tests
Test memory usage and leak detection:
```bash
python run_tests.py --test-type memory
```

#### Coverage Tests
Run tests with coverage reporting:
```bash
python run_tests.py --test-type coverage
```

### Using pytest directly

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m performance

# Run with coverage
pytest --cov=voirs_ffi --cov-report=html

# Run specific test file
pytest unit/test_pipeline.py
```

## Test Categories

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.stress` - Stress tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.memory` - Memory-related tests
- `@pytest.mark.audio` - Audio processing tests

## Fixtures

The test suite provides comprehensive fixtures in `conftest.py`:

- `voirs_pipeline` - Pre-configured VoiRS pipeline
- `sample_texts` - Various text samples for testing
- `synthesis_configs` - Different synthesis configurations
- `temp_audio_dir` - Temporary directory for audio files
- `audio_validator` - Audio validation utilities
- `performance_metrics` - Performance measurement tools
- `memory_tracker` - Memory usage tracking
- `numpy_available` - NumPy availability check

## Test Coverage

The test suite covers:

### Core Functionality
- Pipeline creation and configuration
- Text synthesis with various inputs
- Voice management and switching
- Audio format handling
- Error handling and recovery

### Real-world Scenarios
- Podcast generation
- System notifications
- Accessibility features
- Batch processing
- Multi-language support

### Performance Testing
- Synthesis latency and throughput
- Memory usage patterns
- Concurrent operations
- Resource utilization

### Stress Testing
- High-frequency requests
- Memory pressure conditions
- Resource exhaustion
- Long-running operations
- Recovery after failures

## Configuration

Tests can be configured through environment variables:

```bash
# Skip tests requiring specific features
export SKIP_GPU_TESTS=1
export SKIP_SLOW_TESTS=1

# Adjust test timeouts
export TEST_TIMEOUT=60

# Set test verbosity
export PYTEST_VERBOSITY=2
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'voirs_ffi'**
   - Ensure the VoiRS Python bindings are built and available
   - Check that the module is in your Python path

2. **Tests are skipped**
   - Check that all prerequisites are installed
   - Verify VoiRS bindings are properly configured

3. **Memory tests fail**
   - Install `psutil` for memory monitoring
   - Ensure sufficient system memory is available

4. **Performance tests are slow**
   - Performance tests are designed to be comprehensive
   - Use `--test-type unit` for faster feedback during development

### Debug Mode

For debugging test failures:

```bash
# Run with maximum verbosity
python run_tests.py --verbose

# Run single test with debugging
pytest -xvs unit/test_pipeline.py::TestVoirsPipeline::test_pipeline_creation

# Drop into debugger on failure
pytest --pdb
```

## Contributing

When adding new tests:

1. Use appropriate pytest markers
2. Include docstrings describing test purpose
3. Handle expected failures gracefully
4. Add new fixtures to `conftest.py` if needed
5. Update this README with new test categories

## CI/CD Integration

The test suite is designed for CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run Python tests
  run: |
    cd crates/voirs-ffi/tests/python
    python run_tests.py --test-type all
```

For production CI, consider:
- Running unit and integration tests on every commit
- Running performance tests on releases
- Excluding stress tests from regular CI (too resource-intensive)
- Generating coverage reports for code quality metrics