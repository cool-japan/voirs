# VoiRS Singing Benchmarks

This directory contains comprehensive performance benchmarks for the Version 3.0.0 research-grade features of the VoiRS Singing synthesis system.

## Available Benchmark Suites

### 1. Adaptive Learning Benchmarks
**File**: `adaptive_learning_benchmarks.rs`

Measures performance of the adaptive learning system components.

**Run**:
```bash
cargo bench --bench adaptive_learning_benchmarks
```

**Benchmarks Included**:
- **feedback_collection_single**: Single feedback item processing
- **preference_learning**: Batch learning with 10/50/100/200 samples
- **get_recommendations**: Recommendation generation speed
- **quality_weight_updates**: Weight update performance

**Expected Results**:
- Feedback processing: ~1-10 μs per item
- Preference learning: Linear scaling with sample count
- Recommendations: ~100-500 μs
- Weight updates: ~5-20 μs

---

### 2. Neural Architecture Search Benchmarks
**File**: `neural_architecture_search_benchmarks.rs`

Measures performance of NAS and optimization components.

**Run**:
```bash
cargo bench --bench neural_architecture_search_benchmarks
```

**Benchmarks Included**:
- **population_initialization**: Creating populations of 5/10/20/50 architectures
- **architecture_search**: Full search with 10/25/50 iterations
- **model_compression**: Compression of models from 100K to 5M parameters
- **hardware_optimization_cpu**: CPU-specific optimization
- **hardware_optimization_gpu**: GPU-specific optimization
- **energy_efficiency_optimization**: Energy-aware optimization

**Expected Results**:
- Population init: ~10-100 ms depending on size
- Architecture search: 1-10 seconds per iteration
- Model compression: ~1-50 ms depending on size
- Hardware optimization: ~100-500 μs
- Energy optimization: ~100-300 μs

---

### 3. Research Models Benchmarks
**File**: `research_models_benchmarks.rs`

Measures performance of state-of-the-art research models.

**Run**:
```bash
cargo bench --bench research_models_benchmarks
```

**Benchmarks Included**:
- **diffusion_transformer**: Generation with 10/20/50 denoising steps
- **neural_codec_encode**: Audio encoding to discrete tokens
- **neural_codec_decode**: Token decoding to audio
- **flow_matching**: Flow synthesis with Euler/Heun/RK4 methods
- **score_based_model**: Score-based generation with 10/25/50 scales
- **consistency_single_step**: Single-step generation
- **consistency_multi_step**: Multi-step refinement

**Expected Results**:
- Diffusion: 10-100 ms per step
- Neural codec encode: 1-10 ms
- Neural codec decode: 1-10 ms
- Flow matching: 50-200 ms (method-dependent)
- Score-based: 100-500 ms (scale-dependent)
- Consistency single: 5-20 ms
- Consistency multi: 20-100 ms

---

## Running All Benchmarks

To run all benchmark suites:

```bash
cargo bench
```

To run a specific benchmark:

```bash
cargo bench --bench adaptive_learning_benchmarks
```

To run benchmarks with filtering:

```bash
cargo bench feedback_collection
```

## Benchmark Configuration

Benchmarks use [Criterion.rs](https://github.com/bheisler/criterion.rs) for statistical analysis:

- **Sample Size**: 10-100 depending on benchmark duration
- **Measurement Time**: 5 seconds per benchmark
- **Warm-up Time**: 3 seconds
- **Statistical Analysis**: Automatic outlier detection and regression analysis

## Interpreting Results

Criterion produces detailed HTML reports in `target/criterion/`:

1. **Time per iteration**: Average execution time
2. **Throughput**: Operations per second (where applicable)
3. **Regression analysis**: Detecting performance changes
4. **Comparison**: Against previous runs

## Performance Targets

### Adaptive Learning
- ✅ Feedback collection: < 10 μs per item
- ✅ Preference learning: Linear O(n) scaling
- ✅ Recommendations: < 1 ms

### Neural Architecture Search
- ✅ Population init: < 100 ms for 50 architectures
- ✅ Architecture search: < 10s per iteration
- ✅ Model compression: < 50 ms for 5M parameters

### Research Models
- ✅ Diffusion: Real-time factor < 0.1×
- ✅ Neural codec: < 10 ms encode/decode
- ✅ Flow matching: < 200 ms generation
- ✅ Consistency: < 20 ms single-step

## Continuous Integration

These benchmarks can be integrated into CI/CD pipelines:

```bash
# Run benchmarks without HTML report generation
cargo bench --no-fail-fast -- --save-baseline main

# Compare against baseline
cargo bench -- --baseline main
```

## Hardware Requirements

- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 4GB+ available
- **Disk**: 100MB for Criterion reports

## Further Documentation

- **Examples**: See `../examples/README.md`
- **API Documentation**: `cargo doc --no-deps --open`
- **Criterion Docs**: https://bheisler.github.io/criterion.rs/book/
