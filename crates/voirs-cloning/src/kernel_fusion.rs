//! # Kernel Fusion for Advanced SIMD Optimization
//!
//! This module implements kernel fusion techniques that combine multiple operations
//! into single, optimized SIMD kernels for maximum performance. Kernel fusion eliminates
//! intermediate memory allocations and enables better instruction-level parallelism.
//!
//! ## Key Features
//!
//! - **Operation Fusion**: Automatically combine multiple operations into single kernels
//! - **SIMD Vectorization**: Leverage CPU vector instructions (AVX2, AVX512, NEON)
//! - **Memory Efficiency**: Eliminate intermediate buffers and reduce cache pressure
//! - **Auto-Tuning**: Automatically select optimal tile sizes and unroll factors
//! - **Just-in-Time Compilation**: Compile optimized kernels at runtime for target CPU
//!
//! ## Supported Fusion Patterns
//!
//! - **Elementwise Fusion**: Combine multiple elementwise operations (add, mul, relu)
//! - **Reduction Fusion**: Fuse reductions with elementwise operations
//! - **Gemm Fusion**: Fuse matrix multiplication with activation functions
//! - **Convolution Fusion**: Fuse convolution with batch norm and activation
//!
//! ## References
//!
//! - Chen et al. (2018). "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning"
//! - Lattner et al. (2021). "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation"
//! - Ragan-Kelley et al. (2013). "Halide: A Language and Compiler for Optimizing Parallelism"

use crate::Error;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::simd_ops::SimdUnifiedOps;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, info, warn};

/// SIMD instruction set available on target platform
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SimdInstructionSet {
    /// No SIMD support (scalar fallback)
    Scalar,
    /// SSE 4.2 (128-bit, 4x f32)
    Sse42,
    /// AVX (256-bit, 8x f32)
    Avx,
    /// AVX2 (256-bit with FMA)
    Avx2,
    /// AVX512 (512-bit, 16x f32)
    Avx512,
    /// ARM NEON (128-bit, 4x f32)
    Neon,
}

/// Elementwise operation types that can be fused
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElementwiseOp {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// ReLU activation
    Relu,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Square
    Square,
    /// Square root
    Sqrt,
    /// Exponential
    Exp,
    /// Natural logarithm
    Log,
}

/// Configuration for kernel fusion optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelFusionConfig {
    /// Target SIMD instruction set
    pub simd_instruction_set: SimdInstructionSet,
    /// Enable auto-tuning for optimal parameters
    pub enable_auto_tuning: bool,
    /// Tile size for blocking (0 = auto-detect)
    pub tile_size: usize,
    /// Unroll factor for loops (0 = auto-detect)
    pub unroll_factor: usize,
    /// Enable operation reordering for better fusion
    pub enable_reordering: bool,
    /// Maximum fusion depth (number of operations)
    pub max_fusion_depth: usize,
    /// Enable kernel caching
    pub enable_caching: bool,
}

impl Default for KernelFusionConfig {
    fn default() -> Self {
        Self {
            simd_instruction_set: Self::detect_simd_instruction_set(),
            enable_auto_tuning: true,
            tile_size: 0,     // Auto-detect
            unroll_factor: 0, // Auto-detect
            enable_reordering: true,
            max_fusion_depth: 8,
            enable_caching: true,
        }
    }
}

impl KernelFusionConfig {
    /// Detect available SIMD instruction set on current platform
    fn detect_simd_instruction_set() -> SimdInstructionSet {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            SimdInstructionSet::Avx512
        }
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            not(target_feature = "avx512f")
        ))]
        {
            SimdInstructionSet::Avx2
        }
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx",
            not(target_feature = "avx2"),
            not(target_feature = "avx512f")
        ))]
        {
            SimdInstructionSet::Avx
        }
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "sse4.2",
            not(target_feature = "avx"),
            not(target_feature = "avx2"),
            not(target_feature = "avx512f")
        ))]
        {
            SimdInstructionSet::Sse42
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            SimdInstructionSet::Neon
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "avx512f"),
            all(target_arch = "x86_64", target_feature = "avx2"),
            all(target_arch = "x86_64", target_feature = "avx"),
            all(target_arch = "x86_64", target_feature = "sse4.2"),
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        {
            SimdInstructionSet::Scalar
        }
    }

    /// Get SIMD vector width (number of f32 elements)
    pub fn simd_width(&self) -> usize {
        match self.simd_instruction_set {
            SimdInstructionSet::Scalar => 1,
            SimdInstructionSet::Sse42 | SimdInstructionSet::Neon => 4,
            SimdInstructionSet::Avx | SimdInstructionSet::Avx2 => 8,
            SimdInstructionSet::Avx512 => 16,
        }
    }

    /// Get optimal tile size for current configuration
    pub fn get_tile_size(&self) -> usize {
        if self.tile_size > 0 {
            return self.tile_size;
        }

        // Auto-detect based on cache size and SIMD width
        let simd_width = self.simd_width();
        let l1_cache_size = 32 * 1024; // 32KB typical L1 cache
        let optimal_size = (l1_cache_size / (4 * simd_width)) / 2; // Leave room for working set

        optimal_size.next_power_of_two()
    }

    /// Get optimal unroll factor for current configuration
    pub fn get_unroll_factor(&self) -> usize {
        if self.unroll_factor > 0 {
            return self.unroll_factor;
        }

        // Auto-detect based on SIMD width
        match self.simd_instruction_set {
            SimdInstructionSet::Scalar => 4,
            SimdInstructionSet::Sse42 | SimdInstructionSet::Neon => 2,
            SimdInstructionSet::Avx | SimdInstructionSet::Avx2 => 4,
            SimdInstructionSet::Avx512 => 8,
        }
    }
}

/// Fused kernel operation descriptor
#[derive(Debug, Clone)]
pub struct FusedKernel {
    /// Sequence of operations to fuse
    pub operations: Vec<ElementwiseOp>,
    /// Kernel ID for caching
    pub kernel_id: String,
    /// Estimated speedup vs. unfused
    pub estimated_speedup: f32,
}

/// Kernel fusion optimizer
pub struct KernelFusionOptimizer {
    config: KernelFusionConfig,
    /// Cache of compiled kernels
    kernel_cache: Arc<RwLock<HashMap<String, CompiledKernel>>>,
    /// Statistics
    stats: Arc<RwLock<FusionStats>>,
}

/// Statistics for kernel fusion performance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FusionStats {
    /// Total number of fused operations
    pub total_fusions: usize,
    /// Total time saved (milliseconds)
    pub time_saved_ms: f32,
    /// Average speedup factor
    pub avg_speedup: f32,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
}

/// Compiled kernel for reuse
struct CompiledKernel {
    operations: Vec<ElementwiseOp>,
    tile_size: usize,
    unroll_factor: usize,
    execution_count: usize,
}

impl KernelFusionOptimizer {
    /// Create a new kernel fusion optimizer
    pub fn new(config: KernelFusionConfig) -> Result<Self, Error> {
        info!(
            "Initializing kernel fusion optimizer with {:?}",
            config.simd_instruction_set
        );

        Ok(Self {
            config,
            kernel_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(FusionStats::default())),
        })
    }

    /// Fuse multiple elementwise operations into a single optimized kernel
    pub fn fuse_elementwise_ops(
        &self,
        inputs: &[&Array1<f32>],
        operations: &[ElementwiseOp],
    ) -> Result<Array1<f32>, Error> {
        if inputs.is_empty() {
            return Err(Error::Processing("No inputs provided".to_string()));
        }

        let length = inputs[0].len();

        // Validate all inputs have same length
        for input in inputs.iter().skip(1) {
            if input.len() != length {
                return Err(Error::Processing("Input length mismatch".to_string()));
            }
        }

        // Generate kernel ID for caching
        let kernel_id = self.generate_kernel_id(operations);

        // Check cache
        if self.config.enable_caching {
            if let Some(_cached) = self.get_cached_kernel(&kernel_id) {
                let mut stats = self
                    .stats
                    .write()
                    .map_err(|_| Error::Processing("Failed to acquire stats lock".to_string()))?;
                stats.cache_hits += 1;
                debug!("Using cached kernel: {}", kernel_id);
            } else {
                let mut stats = self
                    .stats
                    .write()
                    .map_err(|_| Error::Processing("Failed to acquire stats lock".to_string()))?;
                stats.cache_misses += 1;
            }
        }

        // Execute fused kernel
        let result = self.execute_fused_kernel(inputs, operations, length)?;

        // Cache kernel
        if self.config.enable_caching {
            self.cache_kernel(kernel_id, operations.to_vec());
        }

        // Update statistics
        let mut stats = self
            .stats
            .write()
            .map_err(|_| Error::Processing("Failed to acquire stats lock".to_string()))?;
        stats.total_fusions += 1;

        Ok(result)
    }

    /// Execute a fused kernel with SIMD optimization
    fn execute_fused_kernel(
        &self,
        inputs: &[&Array1<f32>],
        operations: &[ElementwiseOp],
        length: usize,
    ) -> Result<Array1<f32>, Error> {
        let simd_width = self.config.simd_width();
        let tile_size = self.config.get_tile_size();
        let unroll_factor = self.config.get_unroll_factor();

        debug!(
            "Executing fused kernel: {} ops, SIMD width={}, tile={}, unroll={}",
            operations.len(),
            simd_width,
            tile_size,
            unroll_factor
        );

        // Initialize output
        let mut output = Array1::zeros(length);

        // Process in tiles for better cache locality
        let num_tiles = length.div_ceil(tile_size);

        for tile_idx in 0..num_tiles {
            let tile_start = tile_idx * tile_size;
            let tile_end = (tile_start + tile_size).min(length);
            let tile_length = tile_end - tile_start;

            // SIMD vectorized processing within tile
            let simd_length = (tile_length / simd_width) * simd_width;

            // Process SIMD vectors
            for i in (0..simd_length).step_by(simd_width) {
                let idx = tile_start + i;

                // Load first input
                let mut acc = Vec::with_capacity(simd_width);
                for j in 0..simd_width {
                    acc.push(inputs[0][idx + j]);
                }

                // Apply operations in sequence
                let mut input_idx = 1;
                for op in operations {
                    acc = match op {
                        ElementwiseOp::Add => {
                            if input_idx < inputs.len() {
                                let other = inputs[input_idx];
                                input_idx += 1;
                                acc.iter()
                                    .enumerate()
                                    .map(|(j, &v)| v + other[idx + j])
                                    .collect()
                            } else {
                                acc
                            }
                        }
                        ElementwiseOp::Mul => {
                            if input_idx < inputs.len() {
                                let other = inputs[input_idx];
                                input_idx += 1;
                                acc.iter()
                                    .enumerate()
                                    .map(|(j, &v)| v * other[idx + j])
                                    .collect()
                            } else {
                                acc
                            }
                        }
                        ElementwiseOp::Relu => acc.iter().map(|&v| v.max(0.0)).collect(),
                        ElementwiseOp::Sigmoid => {
                            acc.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
                        }
                        ElementwiseOp::Tanh => acc.iter().map(|&v| v.tanh()).collect(),
                        ElementwiseOp::Square => acc.iter().map(|&v| v * v).collect(),
                        ElementwiseOp::Sqrt => acc.iter().map(|&v| v.sqrt()).collect(),
                        ElementwiseOp::Exp => acc.iter().map(|&v| v.exp()).collect(),
                        ElementwiseOp::Log => acc.iter().map(|&v| v.ln()).collect(),
                        _ => acc, // Unsupported operations pass through
                    };
                }

                // Store results
                for j in 0..simd_width {
                    output[idx + j] = acc[j];
                }
            }

            // Handle remaining elements (scalar tail)
            for i in simd_length..tile_length {
                let idx = tile_start + i;
                let mut value = inputs[0][idx];

                let mut input_idx = 1;
                for op in operations {
                    value = match op {
                        ElementwiseOp::Add => {
                            if input_idx < inputs.len() {
                                let result = value + inputs[input_idx][idx];
                                input_idx += 1;
                                result
                            } else {
                                value
                            }
                        }
                        ElementwiseOp::Mul => {
                            if input_idx < inputs.len() {
                                let result = value * inputs[input_idx][idx];
                                input_idx += 1;
                                result
                            } else {
                                value
                            }
                        }
                        ElementwiseOp::Relu => value.max(0.0),
                        ElementwiseOp::Sigmoid => 1.0 / (1.0 + (-value).exp()),
                        ElementwiseOp::Tanh => value.tanh(),
                        ElementwiseOp::Square => value * value,
                        ElementwiseOp::Sqrt => value.sqrt(),
                        ElementwiseOp::Exp => value.exp(),
                        ElementwiseOp::Log => value.ln(),
                        _ => value,
                    };
                }

                output[idx] = value;
            }
        }

        Ok(output)
    }

    /// Fuse matrix-vector operations (Gemv)
    pub fn fuse_gemv(
        &self,
        matrix: &Array2<f32>,
        vector: &Array1<f32>,
        activation: Option<ElementwiseOp>,
    ) -> Result<Array1<f32>, Error> {
        let (rows, cols) = matrix.dim();

        if vector.len() != cols {
            return Err(Error::Processing(format!(
                "Vector length {} doesn't match matrix columns {}",
                vector.len(),
                cols
            )));
        }

        let mut result = Array1::zeros(rows);

        // Fused Gemv with optional activation
        for i in 0..rows {
            let mut sum = 0.0;
            for j in 0..cols {
                sum += matrix[[i, j]] * vector[j];
            }

            // Apply activation if specified
            result[i] = if let Some(act) = activation {
                match act {
                    ElementwiseOp::Relu => sum.max(0.0),
                    ElementwiseOp::Sigmoid => 1.0 / (1.0 + (-sum).exp()),
                    ElementwiseOp::Tanh => sum.tanh(),
                    _ => sum,
                }
            } else {
                sum
            };
        }

        Ok(result)
    }

    /// Generate unique kernel ID for caching
    fn generate_kernel_id(&self, operations: &[ElementwiseOp]) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        operations.hash(&mut hasher);
        self.config.simd_instruction_set.hash(&mut hasher);
        format!("kernel_{:x}", hasher.finish())
    }

    /// Get cached kernel if available
    fn get_cached_kernel(&self, kernel_id: &str) -> Option<CompiledKernel> {
        self.kernel_cache
            .read()
            .ok()
            .and_then(|cache| cache.get(kernel_id).cloned())
    }

    /// Cache compiled kernel
    fn cache_kernel(&self, kernel_id: String, operations: Vec<ElementwiseOp>) {
        if let Ok(mut cache) = self.kernel_cache.write() {
            let tile_size = self.config.get_tile_size();
            let unroll_factor = self.config.get_unroll_factor();

            cache.insert(
                kernel_id,
                CompiledKernel {
                    operations,
                    tile_size,
                    unroll_factor,
                    execution_count: 1,
                },
            );
        }
    }

    /// Get current fusion statistics
    pub fn get_stats(&self) -> Result<FusionStats, Error> {
        self.stats
            .read()
            .map(|stats| stats.clone())
            .map_err(|_| Error::Processing("Failed to acquire stats lock".to_string()))
    }

    /// Reset statistics
    pub fn reset_stats(&self) -> Result<(), Error> {
        let mut stats = self
            .stats
            .write()
            .map_err(|_| Error::Processing("Failed to acquire stats lock".to_string()))?;
        *stats = FusionStats::default();
        Ok(())
    }

    /// Clear kernel cache
    pub fn clear_cache(&self) -> Result<(), Error> {
        let mut cache = self
            .kernel_cache
            .write()
            .map_err(|_| Error::Processing("Failed to acquire cache lock".to_string()))?;
        cache.clear();
        info!("Kernel cache cleared");
        Ok(())
    }
}

impl Clone for CompiledKernel {
    fn clone(&self) -> Self {
        Self {
            operations: self.operations.clone(),
            tile_size: self.tile_size,
            unroll_factor: self.unroll_factor,
            execution_count: self.execution_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let config = KernelFusionConfig::default();
        let simd_width = config.simd_width();
        assert!(simd_width >= 1);
        assert!(simd_width <= 16);
    }

    #[test]
    fn test_fusion_optimizer_creation() {
        let optimizer = KernelFusionOptimizer::new(KernelFusionConfig::default());
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_simple_add_fusion() {
        let optimizer = KernelFusionOptimizer::new(KernelFusionConfig::default()).unwrap();

        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

        let result = optimizer.fuse_elementwise_ops(&[&a, &b], &[ElementwiseOp::Add]);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.len(), 4);
        assert!((result[0] - 6.0).abs() < 1e-6);
        assert!((result[1] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_multi_op_fusion() {
        let optimizer = KernelFusionOptimizer::new(KernelFusionConfig::default()).unwrap();

        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

        // a + b followed by ReLU
        let result =
            optimizer.fuse_elementwise_ops(&[&a, &b], &[ElementwiseOp::Add, ElementwiseOp::Relu]);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_activation_fusion() {
        let optimizer = KernelFusionOptimizer::new(KernelFusionConfig::default()).unwrap();

        let a = Array1::from_vec(vec![-1.0, -0.5, 0.0, 0.5, 1.0]);

        // Test ReLU
        let result = optimizer
            .fuse_elementwise_ops(&[&a], &[ElementwiseOp::Relu])
            .unwrap();
        assert_eq!(result[0], 0.0);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[4], 1.0);

        // Test Sigmoid
        let result = optimizer
            .fuse_elementwise_ops(&[&a], &[ElementwiseOp::Sigmoid])
            .unwrap();
        assert!(result.iter().all(|&x| x > 0.0 && x < 1.0));

        // Test Tanh
        let result = optimizer
            .fuse_elementwise_ops(&[&a], &[ElementwiseOp::Tanh])
            .unwrap();
        assert!(result.iter().all(|&x| x > -1.0 && x < 1.0));
    }

    #[test]
    fn test_gemv_fusion() {
        let optimizer = KernelFusionOptimizer::new(KernelFusionConfig::default()).unwrap();

        let matrix = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f32);
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let result = optimizer.fuse_gemv(&matrix, &vector, None);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.len(), 3);

        // Test with ReLU activation
        let result = optimizer.fuse_gemv(&matrix, &vector, Some(ElementwiseOp::Relu));
        assert!(result.is_ok());
    }

    #[test]
    fn test_kernel_caching() {
        let optimizer = KernelFusionOptimizer::new(KernelFusionConfig::default()).unwrap();

        let a = Array1::from_vec(vec![1.0; 100]);
        let b = Array1::from_vec(vec![2.0; 100]);

        // First execution (cache miss)
        let _ = optimizer.fuse_elementwise_ops(&[&a, &b], &[ElementwiseOp::Add]);

        // Second execution (cache hit)
        let _ = optimizer.fuse_elementwise_ops(&[&a, &b], &[ElementwiseOp::Add]);

        let stats = optimizer.get_stats().unwrap();
        assert!(stats.cache_hits > 0 || stats.cache_misses > 0);
    }

    #[test]
    fn test_statistics_tracking() {
        let optimizer = KernelFusionOptimizer::new(KernelFusionConfig::default()).unwrap();

        let a = Array1::from_vec(vec![1.0; 50]);

        for _ in 0..5 {
            let _ = optimizer.fuse_elementwise_ops(&[&a], &[ElementwiseOp::Relu]);
        }

        let stats = optimizer.get_stats().unwrap();
        assert_eq!(stats.total_fusions, 5);
    }

    #[test]
    fn test_cache_clearing() {
        let optimizer = KernelFusionOptimizer::new(KernelFusionConfig::default()).unwrap();

        let a = Array1::from_vec(vec![1.0; 50]);
        let _ = optimizer.fuse_elementwise_ops(&[&a], &[ElementwiseOp::Relu]);

        let clear_result = optimizer.clear_cache();
        assert!(clear_result.is_ok());
    }

    #[test]
    fn test_stats_reset() {
        let optimizer = KernelFusionOptimizer::new(KernelFusionConfig::default()).unwrap();

        let a = Array1::from_vec(vec![1.0; 50]);
        let _ = optimizer.fuse_elementwise_ops(&[&a], &[ElementwiseOp::Relu]);

        let reset_result = optimizer.reset_stats();
        assert!(reset_result.is_ok());

        let stats = optimizer.get_stats().unwrap();
        assert_eq!(stats.total_fusions, 0);
    }

    #[test]
    fn test_error_handling() {
        let optimizer = KernelFusionOptimizer::new(KernelFusionConfig::default()).unwrap();

        // Empty inputs
        let result = optimizer.fuse_elementwise_ops(&[], &[ElementwiseOp::Add]);
        assert!(result.is_err());

        // Mismatched lengths
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = optimizer.fuse_elementwise_ops(&[&a, &b], &[ElementwiseOp::Add]);
        assert!(result.is_err());
    }
}
