# Benchmarks

## Status

The benchmark files in this directory (*.rs.disabled) have been temporarily disabled because they use an outdated API that no longer matches the current implementation.

## Files

- `quality_benchmarks.rs.disabled` - Quality assessment and optimization benchmarks
- `core_conversion_benchmarks.rs.disabled` - Core conversion functionality benchmarks
- `transform_benchmarks.rs.disabled` - Audio transformation benchmarks

## TODO

These benchmarks need to be rewritten to match the current API. When updating:

1. Review the current API in `src/lib.rs` and module documentation
2. Check working examples in `examples/` directory for correct usage patterns
3. Reference test files for accurate API calls
4. Use the following as a starting template:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use voirs_conversion::{/* import current types */};

fn bench_basic_conversion(c: &mut Criterion) {
    // TODO: Implement using current API
}

criterion_group!(benches, bench_basic_conversion);
criterion_main!(benches);
```

## Running Benchmarks

Once benchmarks are updated, run them with:

```bash
cargo bench
```

Or run a specific benchmark:

```bash
cargo bench --bench <benchmark_name>
```
