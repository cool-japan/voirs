# VoiRS SDK Benchmarks

This directory contains comprehensive benchmarks for the VoiRS SDK, measuring performance across all key operations.

## Running Benchmarks

### Run All Benchmarks

```bash
cargo bench --benches -p voirs-sdk
```

### Run Specific Benchmark Suite

```bash
# Synthesis benchmarks
cargo bench --bench synthesis_benchmarks -p voirs-sdk

# Streaming benchmarks
cargo bench --bench streaming_benchmarks -p voirs-sdk

# Pipeline benchmarks
cargo bench --bench pipeline_benchmarks -p voirs-sdk

# Concurrent operations benchmarks
cargo bench --bench concurrent_benchmarks -p voirs-sdk

# Memory and cache benchmarks
cargo bench --bench memory_benchmarks -p voirs-sdk
```

### Run Specific Benchmark

```bash
# Run only text synthesis benchmarks
cargo bench --bench synthesis_benchmarks -p voirs-sdk -- text_synthesis

# Run only cache performance benchmarks
cargo bench --bench memory_benchmarks -p voirs-sdk -- cache_performance
```

## Benchmark Suites

### 1. Synthesis Benchmarks (`synthesis_benchmarks.rs`)

Measures core synthesis performance:

- **Text Synthesis**: Performance with varying text lengths (short, medium, long)
- **Quality Levels**: Synthesis time vs quality (Low, Medium, High, Ultra)
- **SSML Processing**: SSML parsing and synthesis with various complexity levels
- **Synthesis Configuration**: Impact of different speaking rates (0.5x - 2.0x)
- **Cache Performance**: First run vs cached synthesis performance
- **Audio Buffer Operations**: Clone, resample, normalize performance

**Key Metrics**:
- Synthesis latency (ms)
- Throughput (characters/second)
- Cache hit rate
- Memory allocation patterns

### 2. Streaming Benchmarks (`streaming_benchmarks.rs`)

Measures streaming synthesis performance:

- **Streaming Latency**: Time to first audio chunk
- **Streaming Throughput**: Full stream processing time
- **Chunk Sizes**: Performance with different chunk sizes (512-8192 samples)
- **Concurrent Streaming**: Multiple simultaneous streams (1-16 concurrent)
- **Memory Efficiency**: Streaming vs batch memory usage
- **Backpressure Handling**: Performance with slow consumers

**Key Metrics**:
- Time to first chunk (ms)
- Total streaming time (s)
- Memory footprint (MB)
- Concurrent stream capacity

### 3. Pipeline Benchmarks (`pipeline_benchmarks.rs`)

Measures pipeline lifecycle performance:

- **Pipeline Creation**: Initialization time with different configurations
- **Voice Switching**: Voice change latency
- **Voice Discovery**: Voice listing and query performance
- **Configuration Updates**: Runtime configuration change overhead
- **State Management**: State query and statistics retrieval
- **Multiple Pipelines**: Resource usage with multiple instances
- **Pipeline Warmup**: Cold start vs warm synthesis comparison

**Key Metrics**:
- Initialization time (ms)
- Voice switch latency (ms)
- Configuration update overhead (ms)
- Memory per pipeline instance (MB)

### 4. Concurrent Benchmarks (`concurrent_benchmarks.rs`)

Measures multi-threaded performance and scalability:

- **Concurrent Synthesis**: Multiple simultaneous synthesis operations (1-16 tasks)
- **Concurrent Voice Switching**: Parallel voice changes
- **Mixed Operations**: Synthesis + queries + state access
- **Queue Saturation**: Performance under high load (10-200 queued tasks)
- **Reader-Writer Patterns**: Heavy read vs balanced workloads
- **Contention**: Performance with high resource contention

**Key Metrics**:
- Throughput (ops/second)
- Latency under load (ms)
- Scalability factor
- Lock contention overhead

### 5. Memory Benchmarks (`memory_benchmarks.rs`)

Measures memory management and caching:

- **Cache Performance**: Hit vs miss performance comparison
- **Memory Allocation**: Allocation patterns for different text sizes
- **Buffer Operations**: Clone, resample, format conversion costs
- **Memory Pools**: Sequential vs concurrent allocation efficiency
- **Cache Eviction**: Performance during cache overflow
- **Memory Pressure**: Behavior under memory constraints
- **Model Caching**: Model load time first vs cached
- **Resource Cleanup**: Cleanup and deallocation overhead

**Key Metrics**:
- Cache hit/miss ratio
- Allocation overhead (µs)
- Peak memory usage (MB)
- Cleanup time (ms)

## Benchmark Configuration

All benchmarks use:
- **Test Mode**: Dummy implementations for consistent, reproducible results
- **Measurement Time**: 10-20 seconds per benchmark (configurable)
- **Sample Size**: Automatic based on benchmark variance
- **Criterion**: HTML reports with statistical analysis

## Interpreting Results

### Performance Targets

Based on the VoiRS SDK goals:

| Metric | Target | Current (Typical) |
|--------|--------|-------------------|
| Synthesis Latency | <100ms | ~250ms (CPU) |
| Streaming First Chunk | <50ms | ~100ms |
| Cache Hit Speedup | >10x | ~20x |
| Concurrent Throughput | 100+ ops/s | ~80 ops/s |
| Memory Overhead | <50MB | ~35MB |

### Reading Benchmark Reports

Criterion generates HTML reports in `target/criterion/`:

```bash
# Open reports in browser
open target/criterion/report/index.html
```

Key statistics:
- **Mean**: Average execution time
- **Std Dev**: Performance stability (lower is better)
- **Median**: Typical performance (less affected by outliers)
- **MAD**: Median Absolute Deviation (robustness metric)

### Performance Regression

Criterion automatically detects performance regressions:
- ✅ **Green**: Performance improved
- ⚠️  **Yellow**: No significant change
- ❌ **Red**: Performance regressed (>5% slower)

## Continuous Benchmarking

### CI Integration

Add to `.github/workflows/benchmarks.yml`:

```yaml
name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run benchmarks
        run: cargo bench --benches -p voirs-sdk
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: target/criterion
```

### Local Development

Run benchmarks regularly during development:

```bash
# Quick check (shorter measurement time)
cargo bench --benches -p voirs-sdk -- --quick

# Full benchmark suite
cargo bench --benches -p voirs-sdk

# Compare with baseline
cargo bench --benches -p voirs-sdk -- --save-baseline main
# ... make changes ...
cargo bench --benches -p voirs-sdk -- --baseline main
```

## Adding New Benchmarks

1. Choose the appropriate benchmark file based on what you're measuring
2. Follow the existing pattern:
   ```rust
   fn bench_new_feature(c: &mut Criterion) {
       let mut group = c.benchmark_group("feature_name");

       // Setup
       let runtime = tokio::runtime::Runtime::new().unwrap();

       group.bench_function("test_case", |b| {
           b.to_async(&runtime).iter(|| async {
               // Benchmark code here
               black_box(result);
           });
       });

       group.finish();
   }
   ```
3. Add to `criterion_group!` at bottom of file
4. Test with `cargo bench --bench <file_name> -p voirs-sdk`

## Troubleshooting

### Benchmarks Take Too Long

Reduce measurement time:
```rust
group.measurement_time(Duration::from_secs(5));
```

### High Variance in Results

- Close other applications
- Disable CPU frequency scaling
- Run multiple times and average
- Increase sample size

### Memory Benchmarks Inconsistent

- Run with `--release` profile only
- Clear system caches between runs
- Monitor system memory pressure

## Best Practices

1. **Always use test mode** for reproducible results
2. **Run benchmarks in release mode** (`cargo bench`, not `cargo test`)
3. **Minimize system load** during benchmarking
4. **Compare against baselines** to track changes
5. **Document performance characteristics** in code comments
6. **Set realistic targets** based on use cases
7. **Profile before optimizing** using these benchmarks to identify bottlenecks

## Further Reading

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [VoiRS Performance Targets](../TODO.md#-performance-targets)
