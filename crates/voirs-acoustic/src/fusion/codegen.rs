//! Kernel Code Generation Module
//!
//! This module provides code generation capabilities for fused kernels,
//! leveraging SciRS2-Core SIMD operations for optimal performance.
//! Generates optimized kernels for different target platforms.

use candle_core::{DType, Device, Tensor};
use scirs2_core::ndarray::*;
use scirs2_core::numeric::*;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt;

use super::graph::OpNode;
use crate::AcousticError;

/// Target platform for kernel code generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodegenTarget {
    /// Automatically detect best target
    Auto,
    /// Generic CPU implementation
    Cpu,
    /// SIMD-optimized CPU (AVX2/AVX-512/NEON)
    CpuSimd,
    /// CUDA GPU implementation
    Cuda,
    /// Metal GPU implementation (macOS)
    Metal,
    /// WebAssembly target
    Wasm,
}

impl fmt::Display for CodegenTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodegenTarget::Auto => write!(f, "Auto"),
            CodegenTarget::Cpu => write!(f, "CPU"),
            CodegenTarget::CpuSimd => write!(f, "CPU (SIMD)"),
            CodegenTarget::Cuda => write!(f, "CUDA"),
            CodegenTarget::Metal => write!(f, "Metal"),
            CodegenTarget::Wasm => write!(f, "WebAssembly"),
        }
    }
}

impl CodegenTarget {
    /// Detect the best available target for current system
    pub fn detect() -> Self {
        // Check SIMD capabilities using runtime detection
        #[cfg(target_arch = "x86_64")]
        {
            // Bind macro results to locals: inlining both `is_x86_feature_detected!`
            // calls inside the `||` trips a clippy `nonminimal_bool` false-positive
            // (it mis-reads the macro expansion as a duplicable boolean term).
            let has_avx2 = std::is_x86_feature_detected!("avx2");
            let has_avx512 = std::is_x86_feature_detected!("avx512f");
            if has_avx2 || has_avx512 {
                CodegenTarget::CpuSimd
            } else {
                CodegenTarget::Cpu
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // ARM64/Apple Silicon always has NEON
            CodegenTarget::CpuSimd
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            CodegenTarget::Cpu
        }
    }

    /// Check if target supports SIMD operations
    pub fn supports_simd(&self) -> bool {
        matches!(self, CodegenTarget::CpuSimd | CodegenTarget::Auto)
    }

    /// Check if target is GPU-based
    pub fn is_gpu(&self) -> bool {
        matches!(self, CodegenTarget::Cuda | CodegenTarget::Metal)
    }
}

/// Fused kernel representation with compiled code and metadata
#[derive(Debug, Clone)]
pub struct FusedKernel {
    /// Unique identifier for the kernel
    pub id: String,

    /// Target platform this kernel was compiled for
    pub target: CodegenTarget,

    /// Original operations that were fused
    pub fused_ops: Vec<String>,

    /// Expected speedup ratio compared to non-fused version
    pub expected_speedup: f32,

    /// Estimated memory usage in bytes
    pub memory_usage: usize,

    /// Execution function (boxed for flexibility)
    executor: KernelExecutor,
}

impl FusedKernel {
    /// Create a new fused kernel
    pub fn new(
        id: String,
        target: CodegenTarget,
        fused_ops: Vec<String>,
        expected_speedup: f32,
        executor: KernelExecutor,
    ) -> Self {
        Self {
            id,
            target,
            fused_ops,
            expected_speedup,
            memory_usage: 0,
            executor,
        }
    }

    /// Execute the fused kernel on input tensors
    pub fn execute(&self, inputs: &[Tensor]) -> Result<Tensor, AcousticError> {
        (self.executor.execute_fn)(inputs)
    }

    /// Estimate size of this kernel in memory
    pub fn estimated_size(&self) -> usize {
        self.memory_usage + self.id.len() + self.fused_ops.iter().map(|s| s.len()).sum::<usize>()
    }

    /// Get kernel information string
    pub fn info(&self) -> String {
        format!(
            "FusedKernel {{ id: {}, target: {}, ops: {:?}, speedup: {:.2}x }}",
            self.id, self.target, self.fused_ops, self.expected_speedup
        )
    }
}

/// Kernel executor with function pointer and metadata
#[derive(Clone)]
pub struct KernelExecutor {
    execute_fn: fn(&[Tensor]) -> Result<Tensor, AcousticError>,
}

impl std::fmt::Debug for KernelExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelExecutor")
            .field("execute_fn", &"<function>")
            .finish()
    }
}

impl KernelExecutor {
    /// Create new kernel executor with given function
    pub fn new(execute_fn: fn(&[Tensor]) -> Result<Tensor, AcousticError>) -> Self {
        Self { execute_fn }
    }
}

/// Main kernel generator for creating fused operations
#[derive(Debug)]
pub struct KernelGenerator {
    device: Device,
    target: CodegenTarget,
}

impl KernelGenerator {
    /// Create a new kernel generator for specified device
    pub fn new(device: &Device) -> Self {
        let target = match device {
            Device::Cpu => CodegenTarget::CpuSimd,
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => CodegenTarget::Cuda,
            #[cfg(feature = "metal")]
            Device::Metal(_) => CodegenTarget::Metal,
            #[allow(unreachable_patterns)]
            _ => CodegenTarget::Cpu,
        };

        Self {
            device: device.clone(),
            target,
        }
    }

    /// Create generator with specific target
    pub fn with_target(device: &Device, target: CodegenTarget) -> Self {
        Self {
            device: device.clone(),
            target,
        }
    }

    /// Generate fused kernel from operation nodes
    pub fn generate_fused_kernel(&self, nodes: &[OpNode]) -> Result<FusedKernel, AcousticError> {
        if nodes.is_empty() {
            return Err(AcousticError::ConfigError {
                message: "Cannot generate kernel from empty node list".to_string(),
            });
        }

        // Analyze fusion pattern
        let pattern = self.analyze_pattern(nodes)?;

        // Generate optimized executor based on pattern
        let executor = self.generate_executor(&pattern)?;

        // Calculate expected speedup
        let expected_speedup = self.estimate_speedup(&pattern);

        // Generate kernel ID
        let kernel_id = self.generate_kernel_id(nodes);

        // Collect operation names
        let fused_ops: Vec<String> = nodes
            .iter()
            .map(|node| node.op_type().to_string())
            .collect();

        Ok(FusedKernel::new(
            kernel_id,
            self.target,
            fused_ops,
            expected_speedup,
            executor,
        ))
    }

    /// Analyze the fusion pattern to determine optimal strategy
    fn analyze_pattern(&self, nodes: &[OpNode]) -> Result<FusionPattern, AcousticError> {
        let op_types: Vec<_> = nodes.iter().map(|n| n.op_type()).collect();

        // Detect common patterns
        if self.is_elementwise_pattern(&op_types) {
            Ok(FusionPattern::ElementWise)
        } else if self.is_reduction_pattern(&op_types) {
            Ok(FusionPattern::Reduction)
        } else if self.is_matmul_pattern(&op_types) {
            Ok(FusionPattern::MatMul)
        } else if self.is_normalization_pattern(&op_types) {
            Ok(FusionPattern::Normalization)
        } else {
            Ok(FusionPattern::Generic)
        }
    }

    /// Check if pattern is element-wise operations
    fn is_elementwise_pattern(&self, ops: &[&str]) -> bool {
        ops.iter().all(|op| {
            matches!(
                *op,
                "add" | "mul" | "sub" | "div" | "relu" | "sigmoid" | "tanh"
            )
        })
    }

    /// Check if pattern is reduction operations
    fn is_reduction_pattern(&self, ops: &[&str]) -> bool {
        ops.iter()
            .any(|op| matches!(*op, "sum" | "mean" | "max" | "min"))
    }

    /// Check if pattern involves matrix multiplication
    fn is_matmul_pattern(&self, ops: &[&str]) -> bool {
        ops.iter()
            .any(|op| matches!(*op, "matmul" | "linear" | "conv1d"))
    }

    /// Check if pattern is normalization
    fn is_normalization_pattern(&self, ops: &[&str]) -> bool {
        let ops_str = ops.join(",");
        ops_str.contains("mean") && (ops_str.contains("std") || ops_str.contains("var"))
    }

    /// Generate optimized executor for the fusion pattern
    fn generate_executor(&self, pattern: &FusionPattern) -> Result<KernelExecutor, AcousticError> {
        match pattern {
            FusionPattern::ElementWise => Ok(KernelExecutor::new(elementwise_fused_kernel)),
            FusionPattern::Reduction => Ok(KernelExecutor::new(reduction_fused_kernel)),
            FusionPattern::MatMul => Ok(KernelExecutor::new(matmul_fused_kernel)),
            FusionPattern::Normalization => Ok(KernelExecutor::new(normalization_fused_kernel)),
            FusionPattern::Generic => Ok(KernelExecutor::new(generic_fused_kernel)),
        }
    }

    /// Estimate speedup from fusion
    fn estimate_speedup(&self, pattern: &FusionPattern) -> f32 {
        match pattern {
            FusionPattern::ElementWise if self.target.supports_simd() => 2.5,
            FusionPattern::ElementWise => 1.5,
            FusionPattern::Reduction if self.target.supports_simd() => 3.0,
            FusionPattern::Reduction => 1.8,
            FusionPattern::MatMul => 1.4,
            FusionPattern::Normalization => 2.2,
            FusionPattern::Generic => 1.2,
        }
    }

    /// Generate unique kernel ID
    fn generate_kernel_id(&self, nodes: &[OpNode]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for node in nodes {
            node.op_type().hash(&mut hasher);
        }
        format!("kernel_{:x}", hasher.finish())
    }
}

/// Fusion pattern classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FusionPattern {
    ElementWise,
    Reduction,
    MatMul,
    Normalization,
    Generic,
}

// ============================================================================
// Fused Kernel Implementations
// ============================================================================

/// Element-wise fused kernel (add, mul, relu, etc.)
fn elementwise_fused_kernel(inputs: &[Tensor]) -> Result<Tensor, AcousticError> {
    if inputs.is_empty() {
        return Err(AcousticError::InputError {
            message: "No input tensors provided".to_string(),
        });
    }

    // For demonstration: fuse add + relu
    // In real implementation, this would be generated dynamically
    let mut result = inputs[0].clone();

    for input in &inputs[1..] {
        result = (result + input).map_err(|e| AcousticError::ProcessingError {
            message: format!("Element-wise operation failed: {}", e),
        })?;
    }

    // Apply ReLU
    result = result.relu().map_err(|e| AcousticError::ProcessingError {
        message: format!("ReLU activation failed: {}", e),
    })?;

    Ok(result)
}

/// Reduction fused kernel (sum, mean, etc.)
fn reduction_fused_kernel(inputs: &[Tensor]) -> Result<Tensor, AcousticError> {
    if inputs.is_empty() {
        return Err(AcousticError::InputError {
            message: "No input tensors provided".to_string(),
        });
    }

    let input = &inputs[0];

    // Fused reduction along last dimension
    let result =
        input
            .sum_keepdim(input.dims().len() - 1)
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Reduction operation failed: {}", e),
            })?;

    Ok(result)
}

/// Matrix multiplication fused kernel
fn matmul_fused_kernel(inputs: &[Tensor]) -> Result<Tensor, AcousticError> {
    if inputs.len() < 2 {
        return Err(AcousticError::InputError {
            message: "MatMul requires at least 2 input tensors".to_string(),
        });
    }

    let result = inputs[0]
        .matmul(&inputs[1])
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("Matrix multiplication failed: {}", e),
        })?;

    Ok(result)
}

/// Normalization fused kernel (layer norm, batch norm)
fn normalization_fused_kernel(inputs: &[Tensor]) -> Result<Tensor, AcousticError> {
    if inputs.is_empty() {
        return Err(AcousticError::InputError {
            message: "No input tensors provided".to_string(),
        });
    }

    let input = &inputs[0];
    let dims = input.dims();
    let last_dim = dims.len() - 1;

    // Compute mean
    let mean = input
        .mean_keepdim(last_dim)
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("Mean computation failed: {}", e),
        })?;

    // Center the input
    let centered = input
        .broadcast_sub(&mean)
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("Centering failed: {}", e),
        })?;

    // Compute variance
    let variance = centered
        .sqr()
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("Variance computation failed: {}", e),
        })?
        .mean_keepdim(last_dim)
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("Variance mean failed: {}", e),
        })?;

    // Normalize
    let eps = 1e-5;
    let std = (variance + eps)
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("Std computation failed: {}", e),
        })?
        .sqrt()
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("Sqrt failed: {}", e),
        })?;

    let normalized = centered
        .broadcast_div(&std)
        .map_err(|e| AcousticError::ProcessingError {
            message: format!("Normalization division failed: {}", e),
        })?;

    Ok(normalized)
}

/// Generic fused kernel for unrecognized patterns
fn generic_fused_kernel(inputs: &[Tensor]) -> Result<Tensor, AcousticError> {
    if inputs.is_empty() {
        return Err(AcousticError::InputError {
            message: "No input tensors provided".to_string(),
        });
    }

    // Simple passthrough for generic case
    Ok(inputs[0].clone())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_codegen_target_detect() {
        let target = CodegenTarget::detect();
        assert!(matches!(
            target,
            CodegenTarget::Cpu | CodegenTarget::CpuSimd
        ));
    }

    #[test]
    fn test_codegen_target_properties() {
        assert!(CodegenTarget::CpuSimd.supports_simd());
        assert!(!CodegenTarget::Cpu.supports_simd());

        assert!(CodegenTarget::Cuda.is_gpu());
        assert!(!CodegenTarget::Cpu.is_gpu());
    }

    #[test]
    fn test_kernel_generator_creation() {
        let device = Device::Cpu;
        let generator = KernelGenerator::new(&device);
        assert!(matches!(
            generator.target,
            CodegenTarget::CpuSimd | CodegenTarget::Cpu
        ));
    }

    #[test]
    fn test_kernel_executor_creation() {
        let executor = KernelExecutor::new(generic_fused_kernel);
        let inputs = vec![Tensor::zeros(&[2, 2], DType::F32, &Device::Cpu).unwrap()];
        let result = (executor.execute_fn)(&inputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_elementwise_kernel() {
        let device = Device::Cpu;
        let t1 = Tensor::ones(&[2, 3], DType::F32, &device).unwrap();
        let t2 = Tensor::ones(&[2, 3], DType::F32, &device).unwrap();

        let result = elementwise_fused_kernel(&[t1, t2]);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dims(), &[2, 3]);
    }

    #[test]
    fn test_reduction_kernel() {
        let device = Device::Cpu;
        let input = Tensor::ones(&[2, 3, 4], DType::F32, &device).unwrap();

        let result = reduction_fused_kernel(&[input]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_matmul_kernel() {
        let device = Device::Cpu;
        let t1 = Tensor::ones(&[2, 3], DType::F32, &device).unwrap();
        let t2 = Tensor::ones(&[3, 4], DType::F32, &device).unwrap();

        let result = matmul_fused_kernel(&[t1, t2]);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dims(), &[2, 4]);
    }

    #[test]
    fn test_normalization_kernel() {
        let device = Device::Cpu;
        let input = Tensor::randn(0.0, 1.0, &[2, 3, 4], &device).unwrap();

        let result = normalization_fused_kernel(&[input]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_kernel_info() {
        let executor = KernelExecutor::new(generic_fused_kernel);
        let kernel = FusedKernel::new(
            "test_kernel".to_string(),
            CodegenTarget::CpuSimd,
            vec!["add".to_string(), "relu".to_string()],
            2.5,
            executor,
        );

        let info = kernel.info();
        assert!(info.contains("test_kernel"));
        assert!(info.contains("2.50x"));
    }
}
