# VoiRS SciRS2 Integration Policy

## Overview

This document establishes the integration policy for VoiRS (Voice Recognition System) with the SciRS2 ecosystem. VoiRS is a voice processing library that leverages SciRS2-Core for scientific computing operations.

**Based on:** `~/work/scirs/SCIRS2_POLICY.md` (v3.0.0)

**NOTE:** The workspace policy and high-level SciRS2 integration rules are documented in `CLAUDE.md`. This document provides additional VoiRS-specific implementation details and migration guidance.

## Core Principle: Layered Abstraction Architecture

VoiRS follows the SciRS2 ecosystem's layered abstraction architecture where **only scirs2-core may use external dependencies directly**, while all other crates (including VoiRS) **MUST use SciRS2-Core abstractions**.

### Why This Matters

- **Consistency**: All modules use the same optimized implementations
- **Maintainability**: Updates and improvements are made in one place
- **Performance**: Optimizations are available to all modules
- **Portability**: Platform-specific code is isolated in core
- **Version Control**: Only core manages external dependency versions
- **Type Safety**: Prevents mixing external types with SciRS2 types

## Mandatory Dependency Rules

### Prohibited Direct Dependencies in Cargo.toml

```toml
# ❌ FORBIDDEN in VoiRS crates (workspace and subcrates)
[workspace.dependencies]
rand = "..."                  # ❌ Use scirs2-core::random
ndarray = "..."               # ❌ Use scirs2-core::ndarray
num-complex = "..."           # ❌ Use scirs2-core::numeric
rayon = "..."                 # ❌ Use scirs2-core::parallel_ops
nalgebra = "..."              # ❌ Use scirs2-core::linalg
num-traits = "..."            # ❌ Use scirs2-core::numeric
```

### Required SciRS2 Dependencies

```toml
# ✅ REQUIRED in workspace dependencies
[workspace.dependencies]
scirs2-core = { version = "0.1.0-rc.1", features = ["array", "random", "simd", "parallel"] }
scirs2-fft = "0.1.0-rc.1"
scirs2-signal = "0.1.0-rc.1"  # Optional - for advanced signal processing
scirs2-linalg = "0.1.0-rc.1"  # Optional - for linear algebra
```

### Subcrate Cargo.toml Pattern

```toml
[dependencies]
scirs2-core.workspace = true
# Add other scirs2-* crates as needed
scirs2-fft.workspace = true    # For FFT operations
```

## Prohibited Code Patterns

```rust
// ❌ FORBIDDEN - Direct external imports
use rand::Rng;
use rand::thread_rng;
use rand_distr::{Beta, Normal, StudentT};
use ndarray::{Array, Array1, Array2};
use ndarray::{array, s!};  // Macros
use num_complex::Complex;
use num_traits::{Float, Zero};
use rayon::prelude::*;
use nalgebra::*;
```

## Required Code Patterns

```rust
// ✅ REQUIRED - SciRS2-Core abstractions

// Random Number Generation
use scirs2_core::random::*;
// Provides: thread_rng, Rng, SliceRandom
// All distributions: Beta, Cauchy, ChiSquared, Normal, StudentT, Weibull, etc.

// Array Operations
use scirs2_core::ndarray::*;
// Provides: Array, Array1, Array2, ArrayView, array!, s!, azip! macros
// Full ndarray ecosystem functionality

// Complex Numbers
use scirs2_core::numeric::*;
// Provides: Complex, Float, Zero, One, Num, etc.

// Parallel Processing
use scirs2_core::parallel_ops::*;
// Provides: into_par_iter, par_iter, par_chunks, etc.

// SIMD Operations
use scirs2_core::simd_ops::SimdUnifiedOps;
// Provides: simd_add, simd_mul, simd_dot, etc.

// Linear Algebra (when needed)
use scirs2_core::linalg::*;
// Provides: nalgebra abstractions
```

## Complete Dependency Mapping for VoiRS

| External Crate | SciRS2-Core Module | VoiRS Usage |
|----------------|-------------------|-------------|
| `rand` | `scirs2_core::random` | Audio noise generation, sampling |
| `rand_distr` | `scirs2_core::random` | Statistical distributions for augmentation |
| `ndarray` | `scirs2_core::ndarray` | Audio buffers, spectrograms, feature matrices |
| `num-complex` | `scirs2_core::numeric` | FFT, complex spectral analysis |
| `num-traits` | `scirs2_core::numeric` | Generic numerical operations |
| `rayon` | `scirs2_core::parallel_ops` | Parallel batch processing |
| `nalgebra` | `scirs2_core::linalg` | Advanced linear algebra (if needed) |

## VoiRS-Specific Integration Patterns

### Audio Signal Processing with SIMD

```rust
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::simd_ops::SimdUnifiedOps;

pub fn process_audio_frame(samples: &ArrayView1<f32>) -> Array1<f32> {
    // Automatic SIMD acceleration for audio buffers
    let mut output = Array1::zeros(samples.len());
    f32::simd_scalar_mul(samples, 0.5, &mut output.view_mut());
    output
}

pub fn mix_channels(left: &ArrayView1<f32>, right: &ArrayView1<f32>) -> Array1<f32> {
    // SIMD-accelerated channel mixing
    let mut mixed = Array1::zeros(left.len());
    f32::simd_add(left, right, &mut mixed.view_mut());
    f32::simd_scalar_mul(&mixed.view(), 0.5, &mut mixed.view_mut());
    mixed
}
```

### FFT and Spectral Analysis

```rust
use scirs2_core::numeric::Complex;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_fft::{RealFft, FftDirection};

pub fn compute_spectrum(signal: &ArrayView1<f32>) -> Result<Array1<Complex<f32>>> {
    let fft_size = signal.len().next_power_of_two();
    let mut fft_input = vec![0.0; fft_size];
    fft_input[..signal.len()].copy_from_slice(signal.as_slice().unwrap());

    let fft = RealFft::new(fft_size, FftDirection::Forward)?;
    let spectrum = fft.transform(&fft_input)?;

    Ok(Array1::from_vec(spectrum))
}
```

### Feature Extraction with Random Sampling

```rust
use scirs2_core::random::*;
use scirs2_core::ndarray::{Array2, ArrayView1};

pub fn extract_features_with_augmentation(audio: &ArrayView1<f32>) -> Array2<f32> {
    let mut rng = thread_rng();
    let noise_dist = Normal::new(0.0, 0.01).unwrap();

    // Add noise augmentation for robustness
    let augmented: Vec<f32> = audio
        .iter()
        .map(|&sample| sample + noise_dist.sample(&mut rng) as f32)
        .collect();

    // Extract MFCC features or other audio features
    extract_mfcc(&augmented)
}
```

### Parallel Batch Processing

```rust
use scirs2_core::parallel_ops::*;
use scirs2_core::ndarray::{Array2, ArrayView2};

pub fn process_audio_batch(batch: &ArrayView2<f32>) -> Vec<Vec<f32>> {
    // Automatic parallel processing for large batches
    (0..batch.nrows())
        .into_par_iter()
        .map(|i| {
            let sample = batch.row(i);
            process_single_sample(&sample)
        })
        .collect()
}

pub fn adaptive_batch_processing(samples: &[Vec<f32>]) -> Vec<Vec<f32>> {
    // Use parallel only for large batches
    if is_parallel_enabled() && samples.len() > 100 {
        samples
            .into_par_iter()
            .map(|sample| process_audio(sample))
            .collect()
    } else {
        samples
            .iter()
            .map(|sample| process_audio(sample))
            .collect()
    }
}
```

### Performance Optimization

```rust
use scirs2_core::simd_ops::{AutoOptimizer, PlatformCapabilities};

pub fn process_with_optimization(samples: &[f32]) -> Vec<f32> {
    let optimizer = AutoOptimizer::new();
    let caps = PlatformCapabilities::detect();

    if caps.simd_available && optimizer.should_use_simd(samples.len()) {
        // Use SIMD path for large audio buffers
        process_simd(samples)
    } else {
        // Use scalar path for small buffers
        process_scalar(samples)
    }
}
```

## Test and Example Code

**ALL tests and examples MUST use SciRS2-Core abstractions:**

```rust
#[cfg(test)]
mod tests {
    use scirs2_core::random::*;
    use scirs2_core::ndarray::*;

    #[test]
    fn test_audio_processing() {
        // Use SciRS2-Core random
        let mut rng = thread_rng();
        let signal = Array1::from_vec(
            (0..1000).map(|_| rng.gen::<f32>()).collect()
        );

        // Use SciRS2-Core array operations
        let processed = process_audio(&signal.view());

        assert_eq!(processed.len(), signal.len());
    }

    #[test]
    fn test_feature_extraction() {
        // Use array! macro from scirs2_core::ndarray
        let audio = array![0.1, 0.2, 0.3, 0.4, 0.5];
        let features = extract_features(&audio.view());

        assert!(features.nrows() > 0);
    }
}
```

## Error Handling

```rust
use scirs2_core::error::CoreError;

#[derive(Debug, thiserror::Error)]
pub enum VoirsError {
    #[error(transparent)]
    Core(#[from] CoreError),

    #[error("Invalid audio format: {0}")]
    InvalidFormat(String),

    #[error("Processing failed: {0}")]
    ProcessingFailed(String),

    #[error("Model error: {0}")]
    ModelError(String),
}

pub type Result<T> = std::result::Result<T, VoirsError>;
```

## Migration Checklist

### Code-Level Changes (ALL VoiRS Crates)

- [x] Replace `use rand::*` with `use scirs2_core::random::*`
- [x] Replace `use rand_distr::*` with `use scirs2_core::random::*`
- [x] Replace `use ndarray::*` with `use scirs2_core::ndarray::*`
- [x] Replace `use num_complex::*` with `use scirs2_core::numeric::*`
- [x] Replace `use rayon::prelude::*` with `use scirs2_core::parallel_ops::*`
- [x] Replace `use nalgebra::*` with `use scirs2_core::linalg::*` (when needed)
- [ ] Update all tests to use SciRS2-Core abstractions
- [ ] Update all examples to use SciRS2-Core abstractions
- [ ] Update all benchmarks to use SciRS2-Core abstractions

### Workspace Cargo.toml Changes

- [x] Update `scirs2-core` to `0.1.0-rc.1`
- [x] Update `scirs2-fft` to `0.1.0-rc.1`
- [x] Add `array` feature to scirs2-core
- [x] Remove direct `rand` dependency from workspace
- [x] Remove direct `ndarray` dependency from workspace
- [x] Remove direct `num-complex` dependency from workspace
- [x] Remove direct `rayon` dependency from workspace
- [x] Remove direct `nalgebra` dependency from workspace

### Subcrate Cargo.toml Changes

For each VoiRS subcrate:

- [ ] Remove direct `rand.workspace = true` from `[dependencies]`
- [ ] Remove direct `ndarray.workspace = true` from `[dependencies]`
- [ ] Remove direct `num-complex.workspace = true` from `[dependencies]`
- [ ] Remove direct `rayon.workspace = true` from `[dependencies]`
- [ ] Remove direct `nalgebra.workspace = true` from `[dependencies]`
- [ ] Ensure `scirs2-core.workspace = true` is present
- [ ] Remove `keywords.workspace = true` and define specific keywords
- [ ] Optionally customize `categories` per subcrate

## Benefits for VoiRS

1. **Optimized Audio Processing**: Automatic SIMD for signal processing (AVX2, AVX512, NEON)
2. **GPU Acceleration**: Access to GPU kernels for deep learning inference
3. **Consistent APIs**: Same interfaces across all SciRS2 projects
4. **Parallel Batch Processing**: Efficient multi-threaded audio batch processing
5. **Version Management**: Single point of dependency version control
6. **Cross-Platform**: Consistent behavior across Linux, macOS, Windows
7. **Future-Proof**: Automatic access to new SciRS2-Core optimizations
8. **Type Safety**: Prevents mixing incompatible dependency versions

## Compliance Verification

### Quick Check Commands

```bash
# Check for prohibited imports in source code
grep -r "use rand::" crates/*/src/
grep -r "use ndarray::" crates/*/src/
grep -r "use num_complex::" crates/*/src/
grep -r "use rayon::" crates/*/src/

# Check for prohibited dependencies in Cargo.toml
grep -r "rand.workspace" crates/*/Cargo.toml
grep -r "ndarray.workspace" crates/*/Cargo.toml
grep -r "num-complex.workspace" crates/*/Cargo.toml
grep -r "rayon.workspace" crates/*/Cargo.toml
```

### Expected Results (After Full Compliance)

All commands should return **zero results** ✅

## Related Documentation

- **SciRS2 Core Policy**: `~/work/scirs/SCIRS2_POLICY.md` (v3.0.0)
- **VoiRS Development Guidelines**: `CLAUDE.md`
- **SciRS2-Core Documentation**: https://docs.rs/scirs2-core
- **NumRS2 Integration**: For numpy .npz file support (separate from SciRS2)

## Policy Version

- **Policy Version**: 2.0.0 (Updated for SciRS2 RC.1)
- **Effective Date**: 2025-10-04
- **VoiRS Version**: 0.1.0-alpha.2
- **SciRS2-Core Version**: 0.1.0-rc.1
- **Status**: Active - Migration in Progress

---

*This policy ensures VoiRS maintains full compatibility with the SciRS2 ecosystem and benefits from centralized optimizations, type safety, and consistent abstractions across all scientific computing operations.*
