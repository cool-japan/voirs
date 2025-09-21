# Performance Regression Testing Infrastructure

This directory contains the performance regression testing infrastructure for VoiRS Recognizer, designed to automatically detect performance regressions and track performance metrics over time.

## Overview

The performance regression testing system provides:

- **Automated Regression Detection**: Compares current performance against established baselines
- **Historical Tracking**: Maintains performance history for trend analysis
- **CI Integration**: Structured output suitable for continuous integration pipelines
- **Multi-Configuration Testing**: Tests various model sizes, audio durations, and configurations
- **Severity Classification**: Categorizes regressions as Minor, Major, or Critical

## Files

- `baseline.json` - Baseline performance metrics for regression comparison
- `history.json` - Historical performance data (last 100 benchmark runs)
- `README.md` - This documentation file

## Usage

### Running Performance Tests

```bash
# Run the full automated regression suite
cargo test test_automated_performance_regression_suite -- --nocapture

# Run quick CI check
CI_QUICK_CHECK=1 cargo test test_ci_quick_regression_check -- --nocapture

# Run model scaling tests
cargo test test_model_scaling_performance -- --nocapture

# Run audio duration scaling tests
cargo test test_audio_duration_scaling -- --nocapture

# Run memory pressure tests
cargo test test_memory_pressure_scenarios -- --nocapture
```

### Updating Baseline

To update the performance baseline (when intentional performance changes are made):

```bash
UPDATE_PERFORMANCE_BASELINE=1 cargo test test_automated_performance_regression_suite -- --nocapture
```

### CI Integration

The tests are designed to work in CI environments:

```bash
# In CI pipeline
CI=1 cargo test automated_regression_suite -- --nocapture
```

## Test Configurations

The regression tests cover multiple scenarios:

### Model Sizes
- **Small**: Fastest processing, lowest memory usage
- **Base**: Balanced performance and accuracy
- **Large**: Higher accuracy, increased resource usage

### Audio Configurations
- **Sample Rates**: 16kHz (primary), 48kHz (high quality)
- **Channels**: Mono and stereo audio
- **Durations**: 0.5s to 10s audio clips

### Performance Metrics

The system tracks these key metrics:

1. **Real-Time Factor (RTF)**: Processing time / audio duration
2. **Memory Usage**: Peak memory consumption in bytes
3. **Startup Time**: Model initialization time in milliseconds
4. **Streaming Latency**: End-to-end processing latency
5. **Throughput**: Samples processed per second
6. **CPU Utilization**: Estimated CPU usage percentage

## Regression Thresholds

Default regression detection thresholds:

- **RTF**: 15% increase triggers alert
- **Memory**: 20% increase triggers alert
- **Startup Time**: 25% increase triggers alert
- **Latency**: 10% increase triggers alert
- **Throughput**: 10% decrease triggers alert

## Severity Levels

Regressions are classified by severity:

- **Minor** (5-15% degradation): Monitoring recommended
- **Major** (15-30% degradation): Investigation required
- **Critical** (>30% degradation): Immediate attention required

## Output Formats

### Console Output
Human-readable progress and results with emoji indicators:
- ✅ Success/Pass
- ⚠️ Warning/Minor regression
- 🚨 Major regression
- 🔥 Critical regression
- 🚀 Performance improvement

### CI Report Format
Structured output suitable for CI systems:
```
❌ PERFORMANCE REGRESSION DETECTED

🚨 MAJOR RTF: 18.50% regression (current: 0.237, baseline: 0.200)
⚠️ MINOR Memory Usage: 12.30% regression (current: 561.2MB, baseline: 500.0MB)

🚀 Performance Improvements:
  • Throughput: 8.50% improvement

Overall Performance Delta: +12.30%
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Performance Regression Tests

on:
  pull_request:
    branches: [ main ]
  push:
    branches: [ main ]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        
    - name: Run quick performance check
      run: |
        CI_QUICK_CHECK=1 cargo test test_ci_quick_regression_check -- --nocapture
        
    - name: Run full regression suite
      if: github.event_name == 'push' && github.ref == 'refs/heads/main'
      run: |
        CI=1 cargo test test_automated_performance_regression_suite -- --nocapture
```

### Setting Performance Baselines

When making intentional performance changes (optimizations, model updates):

1. Run the regression suite with baseline update:
   ```bash
   UPDATE_PERFORMANCE_BASELINE=1 cargo test test_automated_performance_regression_suite
   ```

2. Commit the updated `baseline.json` file:
   ```bash
   git add tests/benchmarks/baseline.json
   git commit -m "Update performance baseline after optimization"
   ```

## Troubleshooting

### Common Issues

1. **No baseline found**: Run with `UPDATE_PERFORMANCE_BASELINE=1` to create initial baseline
2. **Platform differences**: Baselines are platform-specific; CI should use consistent environments
3. **Memory estimation**: On unsupported platforms, intelligent estimation is used

### Debug Mode

For detailed debugging information:

```bash
RUST_LOG=debug cargo test automated_regression_suite -- --nocapture
```

## Best Practices

1. **Consistent Environment**: Use same hardware/OS for baseline and regression testing
2. **Regular Updates**: Update baselines when making intentional performance changes
3. **Monitor Trends**: Review historical data for gradual performance degradation
4. **Quick Feedback**: Use CI_QUICK_CHECK for fast PR validation
5. **Full Validation**: Run complete suite for releases and major changes

## Future Enhancements

Planned improvements:

- [ ] Platform-specific baselines
- [ ] Performance trend analysis and predictions
- [ ] Integration with external monitoring systems
- [ ] Automated performance optimization suggestions
- [ ] Visual performance dashboards
- [ ] Statistical significance testing for regressions