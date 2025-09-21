# VoiRS Cross-Language Testing Framework

This comprehensive testing framework validates consistency and compatibility across all VoiRS language bindings, ensuring that C API, Python, Node.js, and WebAssembly bindings produce consistent results and maintain API compatibility.

## Overview

The cross-language testing framework provides:

- **Consistency Testing** - Verifies that all bindings produce identical results for the same inputs
- **Performance Comparison** - Compares performance characteristics across bindings
- **Memory Usage Analysis** - Monitors memory usage patterns and detects potential leaks
- **API Compatibility** - Ensures all bindings expose equivalent functionality
- **Automated Testing** - Comprehensive test automation with detailed reporting

## Test Structure

```
tests/cross_lang/
├── README.md                     # This documentation
├── test_consistency.py           # Python-based consistency tester
├── run_cross_lang_tests.sh      # Main test orchestration script
└── cross_lang_test_report.md    # Generated test report
```

## Supported Bindings

| Binding | Language | Technology | Status |
|---------|----------|------------|---------|
| C API | C/C++ | Direct FFI | ✅ Always Available |
| Python | Python 3.7+ | PyO3 | ✅ Available when built |
| Node.js | JavaScript/TypeScript | NAPI-RS | 🚧 Available when built |
| WebAssembly | JavaScript (Browser) | wasm-bindgen | 🚧 Available when built |

## Prerequisites

### Required
- Rust toolchain (for building FFI library)
- VoiRS FFI library compiled (`cargo build`)

### Optional (for specific bindings)
- **Python**: Python 3.7+, maturin (for building Python bindings)
- **Node.js**: Node.js 14+, npm (for building Node.js bindings)
- **WebAssembly**: wasm-pack (for building WASM bindings)

## Running Tests

### Quick Start

```bash
# Run all available cross-language tests
./run_cross_lang_tests.sh

# Check which bindings are available
./run_cross_lang_tests.sh --check

# Build missing components
./run_cross_lang_tests.sh --build
```

### Manual Testing

#### Check Binding Availability
```bash
./run_cross_lang_tests.sh --check
```

#### Run Rust-based Tests
```bash
cd .. && cargo test cross_lang
```

#### Run Python Consistency Tests
```bash
python3 test_consistency.py
```

## Test Categories

### 1. Consistency Tests

Verify that all bindings produce identical results:

- **Audio Output Consistency** - Same text produces same audio samples
- **Configuration Handling** - Same config parameters produce same results
- **Error Handling** - Same errors reported across bindings
- **Voice Management** - Same voice operations work consistently

**Test Cases:**
- Multiple text inputs (short, medium, long)
- Various synthesis configurations (speed, pitch, quality)
- Different voice selections
- Error conditions and edge cases

### 2. Performance Comparison

Compare performance characteristics:

- **Synthesis Latency** - Time to generate audio
- **Throughput** - Operations per second
- **Memory Usage** - RAM consumption patterns
- **FFI Overhead** - Cost of language binding layer

**Metrics Tracked:**
- Synthesis time per text length
- Memory usage before/after operations
- Peak memory consumption
- Garbage collection impact (where applicable)

### 3. Memory Analysis

Monitor memory usage patterns:

- **Leak Detection** - Long-running operations
- **Peak Usage** - Maximum memory consumption
- **Cleanup Verification** - Proper resource cleanup
- **Reference Counting** - Shared resource management

### 4. API Compatibility

Ensure equivalent functionality:

- **Function Availability** - All core functions present
- **Parameter Compatibility** - Same parameters accepted
- **Return Value Consistency** - Same return types and values
- **Error Code Mapping** - Consistent error reporting

## Test Configuration

### Default Test Parameters

```python
TEST_TEXTS = [
    "Hello world",
    "This is a test sentence for cross-language consistency.",
    "Testing numbers: 123, 456.78, and special characters: @#$%",
]

TEST_CONFIGS = [
    {"speaking_rate": 1.0, "pitch_shift": 0.0},
    {"speaking_rate": 1.5, "pitch_shift": 2.0},
    {"speaking_rate": 0.8, "pitch_shift": -1.0},
]
```

### Similarity Thresholds

- **Audio Similarity**: 95% correlation required
- **Duration Tolerance**: ±5% variation allowed
- **Sample Rate Match**: Exact match required
- **Channel Count Match**: Exact match required

## Building Bindings

### Python Bindings
```bash
# Install maturin if not available
pip install maturin

# Build Python bindings
cd /path/to/voirs-ffi
maturin develop --features python
```

### Node.js Bindings
```bash
cd /path/to/voirs-ffi
npm install
npm run build
```

### WebAssembly Bindings
```bash
# Install wasm-pack if not available
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build WASM bindings
cd /path/to/voirs-ffi
wasm-pack build --target web --features wasm
```

## Test Results

### Success Criteria

Tests pass when:
- All available bindings produce consistent results (>95% similarity)
- No memory leaks detected
- Performance within acceptable thresholds
- All error conditions handled consistently

### Common Issues

#### Binding Not Available
```
WARN: Python: Not available - voirs_ffi module not found
```
**Solution**: Build the Python bindings with `maturin develop --features python`

#### Compilation Errors
```
ERROR: Failed to build Rust FFI library
```
**Solution**: Run `cargo build` in the voirs-ffi directory

#### Inconsistent Results
```
ERROR: Consistency test failed - audio similarity: 87%
```
**Solution**: Check for implementation differences between bindings

### Troubleshooting

#### Debug Binding Issues
```bash
# Test C API directly
cargo test c_api

# Test Python bindings
python3 -c "import voirs_ffi; print('OK')"

# Test Node.js bindings
node -e "require('./index.js')"

# Check WASM bindings
ls pkg/voirs_ffi.js
```

#### Memory Issues
```bash
# Run with memory monitoring
python3 -c "
import psutil
import voirs_ffi
print(f'Memory: {psutil.Process().memory_info().rss/1024/1024:.1f} MB')
"
```

## Advanced Usage

### Custom Test Configuration

Create a custom test configuration file:

```python
# custom_config.py
CUSTOM_TEST_CONFIG = {
    "test_texts": [
        "Custom test text 1",
        "Custom test text 2",
    ],
    "synthesis_configs": [
        {"speaking_rate": 1.2, "volume_gain": 1.1},
    ],
    "tolerance": 0.03,  # 3% tolerance
}
```

### Integration with CI/CD

```yaml
# .github/workflows/cross-lang-tests.yml
name: Cross-Language Tests
on: [push, pull_request]

jobs:
  cross-lang-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Setup Node.js
      uses: actions/setup-node@v2
      with:
        node-version: '16'
    - name: Install dependencies
      run: |
        pip install maturin
        cargo install wasm-pack
    - name: Run cross-language tests
      run: |
        cd crates/voirs-ffi/tests/cross_lang
        ./run_cross_lang_tests.sh
```

### Performance Benchmarking

Run extended performance tests:

```bash
# Run with performance focus
PERFORMANCE_MODE=1 ./run_cross_lang_tests.sh

# Generate performance report
python3 test_consistency.py --performance --iterations 100
```

## Contributing

When adding new language bindings:

1. Implement the binding following existing patterns
2. Add binding detection to `run_cross_lang_tests.sh`
3. Create a tester class in `test_consistency.py`
4. Add binding-specific tests in `cross_lang.rs`
5. Update documentation

### Test Coverage Requirements

New bindings must provide:
- Basic synthesis functionality
- Configuration parameter support
- Error handling
- Memory management
- Performance benchmarks

## FAQ

**Q: Why do some tests get skipped?**
A: Tests are skipped when required bindings are not available. Build the missing bindings to enable all tests.

**Q: What if consistency tests fail?**
A: Check for implementation differences between bindings. Common issues include floating-point precision, configuration parameter handling, and audio format differences.

**Q: How accurate are the similarity measurements?**
A: Audio similarity uses normalized cross-correlation, which is robust for detecting functional differences while allowing for minor implementation variations.

**Q: Can I run tests for specific bindings only?**
A: Yes, modify the `run_cross_lang_tests.sh` script or run individual test components directly.

**Q: How long do the tests take?**
A: Complete cross-language tests typically take 2-5 minutes depending on available bindings and system performance.

## Support

For issues with cross-language testing:

1. Check binding availability with `./run_cross_lang_tests.sh --check`
2. Review the generated test report
3. Check individual binding implementations
4. Report issues with detailed logs and system information