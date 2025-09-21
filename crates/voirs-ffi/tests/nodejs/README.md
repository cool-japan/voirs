# VoiRS Node.js Testing Infrastructure

Comprehensive test suite for VoiRS FFI Node.js bindings, providing thorough validation of functionality, performance, and reliability.

## Overview

This testing infrastructure provides comprehensive coverage of the VoiRS Node.js bindings with multiple test categories:

- **Unit Tests** - Core functionality and API surface testing
- **Integration Tests** - End-to-end workflows and real-world usage scenarios  
- **Error Tests** - Error handling, edge cases, and recovery testing
- **Performance Tests** - Benchmarks, throughput, and latency testing
- **Memory Tests** - Memory usage, leak detection, and resource management
- **Concurrency Tests** - Thread safety and multi-threaded operations

## Quick Start

### Running All Tests

```bash
# Run the comprehensive test suite
node test-runner.js

# Or use npm scripts
npm test
```

### Running Specific Test Categories

```bash
# Unit tests only
npm run test:unit

# Integration tests
npm run test:integration

# Error handling tests
npm run test:errors

# Performance tests
npm run test:performance

# Memory tests  
npm run test:memory

# Concurrency tests
npm run test:concurrent

# All tests sequentially
npm run test:all
```

## Test Configuration

### Environment Variables

- `VOIRS_SKIP_SLOW_TESTS=true` - Skip performance and memory tests (fast mode)
- `CI=true` - Automatically enables fast mode for CI environments
- `VERBOSE=true` - Enable verbose output with detailed test information
- `NODE_ENV=test` - Set automatically by test runner

### Fast Mode

When `VOIRS_SKIP_SLOW_TESTS=true` or `CI=true`:
- Performance tests are skipped
- Memory tests are skipped  
- Long-running operations are bypassed
- Tests complete in under 10 seconds

### Full Mode (Default)

- All test categories run
- Complete performance benchmarks
- Comprehensive memory analysis
- Full test suite takes 2-5 minutes

## Test Architecture

### Mock Implementations

When VoiRS bindings are not available (e.g., not compiled), the tests use sophisticated mock implementations that:

- Simulate realistic processing times
- Generate proper audio buffer structures
- Implement error conditions and edge cases
- Support concurrent operations
- Track memory usage patterns

### Test Framework Features

- **Parallel Execution** - Tests run concurrently when safe
- **Timeout Protection** - Prevents hanging tests
- **Memory Tracking** - Monitors resource usage
- **Error Recovery** - Tests continue after failures
- **Detailed Reporting** - Comprehensive metrics and diagnostics

## Test Categories

### Unit Tests (`unit-tests.js`)

Tests core API functionality:
- Pipeline construction and configuration
- Voice management (list, set, get)
- Basic synthesis operations
- SSML processing
- Callback mechanisms
- Input validation

**Coverage**: Basic API surface, parameter validation, return types

### Integration Tests (`integration-tests.js`)

Tests complete workflows:
- End-to-end text-to-speech pipelines
- File output workflows
- Multi-voice synthesis
- Batch processing
- Complex SSML workflows
- Streaming synthesis
- Configuration management

**Coverage**: Real-world usage patterns, workflow completion

### Error Tests (`error-tests.js`)

Tests error conditions and recovery:
- Invalid input handling
- Resource exhaustion
- Concurrent error scenarios
- Recovery after errors
- Edge case inputs
- Timeout handling

**Coverage**: Robustness, error messaging, graceful degradation

### Performance Tests (`performance-tests.js`)

Benchmarks and performance analysis:
- Single synthesis latency
- Batch throughput measurement
- Concurrent operation performance
- Streaming synthesis benchmarks
- Voice switching performance
- Quality level comparisons
- Memory efficiency
- CPU utilization

**Coverage**: Performance characteristics, scalability, optimization

### Memory Tests (`memory-tests.js`)

Memory usage and leak detection:
- Single operation memory usage
- Memory leak detection across iterations
- Large buffer handling
- Concurrent memory usage
- Streaming memory efficiency
- Pipeline cleanup verification
- Buffer reference management

**Coverage**: Memory safety, resource cleanup, leak prevention

### Concurrency Tests (`concurrent-tests.js`)

Multi-threading and thread safety:
- Parallel synthesis operations
- Multiple pipeline concurrency
- Concurrent voice switching
- Streaming concurrency
- Mixed operation scenarios
- Race condition handling
- High load testing
- Thread safety validation

**Coverage**: Thread safety, resource sharing, concurrent access

## Test Results and Metrics

### Success Criteria

- **Unit Tests**: All core APIs function correctly
- **Integration Tests**: Complete workflows execute successfully
- **Error Tests**: Proper error handling and recovery
- **Performance Tests**: Meet latency and throughput benchmarks
- **Memory Tests**: No memory leaks, bounded resource usage
- **Concurrency Tests**: Thread-safe operation under load

### Performance Benchmarks

- **Latency**: Single synthesis < 1 second
- **Throughput**: Batch processing > 0.1x real-time
- **Concurrency**: Handle 8+ parallel operations
- **Memory**: < 200MB growth across test suite
- **Streaming**: First chunk < 500ms

### Metrics Collected

- Operation timing and latency
- Memory usage (heap, RSS, external)
- Throughput and operations per second
- Error rates and recovery success
- Resource utilization
- Thread safety validation

## Troubleshooting

### Common Issues

**Tests Timeout**
- Check if VoiRS bindings are properly compiled
- Verify system resources are available
- Enable fast mode with `VOIRS_SKIP_SLOW_TESTS=true`

**Memory Tests Fail**
- Ensure Node.js has sufficient memory
- Check for other applications consuming memory
- Run with `--expose-gc` flag for better garbage collection

**Concurrency Tests Fail**
- Verify multi-threading support
- Check for system resource limits
- Reduce concurrency in resource-constrained environments

### Debugging

```bash
# Run with verbose output
VERBOSE=true node test-runner.js

# Run single test category for debugging
node unit-tests.js

# Enable garbage collection for memory tests
node --expose-gc memory-tests.js
```

## Development

### Adding New Tests

1. Add test to appropriate category file
2. Follow existing test patterns
3. Include proper assertions and error handling
4. Add performance metrics where applicable
5. Test both with real bindings and mocks

### Mock Implementation Updates

When adding new APIs:
1. Update mock classes in test files
2. Implement realistic behavior
3. Add appropriate error conditions
4. Maintain consistency across test files

### CI Integration

The test suite is designed for CI environments:
- Automatic fast mode detection
- Proper exit codes for pass/fail
- Comprehensive reporting
- Timeout protection
- Resource cleanup

## Dependencies

- Node.js >= 16
- VoiRS FFI bindings (optional, uses mocks if unavailable)
- Standard Node.js testing capabilities

No external testing frameworks required - uses custom lightweight test runner optimized for VoiRS testing needs.