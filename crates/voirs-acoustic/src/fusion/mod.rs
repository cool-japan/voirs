//! Kernel Fusion Optimization Module
//! 
//! This module provides kernel fusion capabilities for optimizing neural network
//! operations by combining multiple operations into single, more efficient kernels.
//! This reduces memory bandwidth requirements and improves computational efficiency.

use std::collections::HashMap;
use candle_core::{Device, Tensor, DType};
use crate::error::AcousticError;

pub mod graph;
pub mod patterns;
pub mod codegen;

pub use graph::{OpGraph, OpNode, FusionGraph};
pub use patterns::{FusionPattern, PatternMatcher, FusionRule};
pub use codegen::{KernelGenerator, FusedKernel, CodegenTarget};

/// Main kernel fusion optimizer
#[derive(Debug)]
pub struct KernelFusion {
    device: Device,
    patterns: Vec<FusionPattern>,
    cache: HashMap<String, FusedKernel>,
    enabled: bool,
}

impl KernelFusion {
    /// Create a new kernel fusion optimizer
    pub fn new(device: Device) -> Self {
        Self {
            device,
            patterns: Self::default_patterns(),
            cache: HashMap::new(),
            enabled: true,
        }
    }

    /// Create with custom fusion patterns
    pub fn with_patterns(device: Device, patterns: Vec<FusionPattern>) -> Self {
        Self {
            device,
            patterns,
            cache: HashMap::new(),
            enabled: true,
        }
    }

    /// Enable or disable kernel fusion
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if fusion is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Optimize a computation graph by applying fusion patterns
    pub fn optimize_graph(&mut self, graph: OpGraph) -> Result<FusionGraph, AcousticError> {
        if !self.enabled {
            return Ok(FusionGraph::from_op_graph(graph));
        }

        let mut fusion_graph = FusionGraph::from_op_graph(graph);
        let matcher = PatternMatcher::new(&self.patterns);
        
        // Apply fusion patterns iteratively until no more matches found
        let mut changed = true;
        while changed {
            changed = false;
            
            if let Some(matches) = matcher.find_matches(&fusion_graph)? {
                for fusion_match in matches {
                    fusion_graph.apply_fusion(fusion_match)?;
                    changed = true;
                }
            }
        }

        Ok(fusion_graph)
    }

    /// Generate fused kernel from a fusion group
    pub fn generate_kernel(&mut self, nodes: &[OpNode]) -> Result<FusedKernel, AcousticError> {
        let cache_key = self.compute_cache_key(nodes)?;
        
        if let Some(cached_kernel) = self.cache.get(&cache_key) {
            return Ok(cached_kernel.clone());
        }

        let generator = KernelGenerator::new(&self.device);
        let kernel = generator.generate_fused_kernel(nodes)?;
        
        self.cache.insert(cache_key, kernel.clone());
        Ok(kernel)
    }

    /// Clear the kernel cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            size: self.cache.len(),
            memory_usage: self.estimate_cache_memory(),
        }
    }

    /// Get default fusion patterns for common operations
    fn default_patterns() -> Vec<FusionPattern> {
        vec![
            // Element-wise operations fusion
            FusionPattern::new("add_mul", vec!["add", "mul"])
                .with_rule(FusionRule::ElementWise)
                .with_priority(10),
            
            // Activation function fusion
            FusionPattern::new("linear_relu", vec!["linear", "relu"])
                .with_rule(FusionRule::Pointwise)
                .with_priority(15),
            
            // Convolution + bias + activation
            FusionPattern::new("conv_bias_activation", vec!["conv1d", "add", "relu"])
                .with_rule(FusionRule::ConvolutionBased)
                .with_priority(20),
            
            // Matrix multiplication chains
            FusionPattern::new("matmul_chain", vec!["matmul", "matmul"])
                .with_rule(FusionRule::LinearAlgebra)
                .with_priority(12),
            
            // Normalization patterns
            FusionPattern::new("layer_norm", vec!["mean", "sub", "pow", "mean", "add", "sqrt", "div"])
                .with_rule(FusionRule::Normalization)
                .with_priority(25),
        ]
    }

    /// Compute cache key for a set of operations
    fn compute_cache_key(&self, nodes: &[OpNode]) -> Result<String, AcousticError> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        
        for node in nodes {
            node.op_type().hash(&mut hasher);
            node.input_shapes().hash(&mut hasher);
            node.output_shape().hash(&mut hasher);
        }
        
        Ok(format!("{:x}", hasher.finish()))
    }

    /// Estimate memory usage of cached kernels
    fn estimate_cache_memory(&self) -> usize {
        self.cache.iter()
            .map(|(key, kernel)| key.len() + kernel.estimated_size())
            .sum()
    }
}

/// Cache statistics for kernel fusion
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub memory_usage: usize,
}

/// Configuration for kernel fusion optimization
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Maximum number of operations to fuse together
    pub max_fusion_size: usize,
    
    /// Minimum expected speedup to apply fusion
    pub min_speedup_ratio: f32,
    
    /// Maximum memory overhead allowed for fusion
    pub max_memory_overhead: f32,
    
    /// Enable aggressive fusion optimizations
    pub aggressive_fusion: bool,
    
    /// Target platform for code generation
    pub target: CodegenTarget,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            max_fusion_size: 8,
            min_speedup_ratio: 1.2,
            max_memory_overhead: 0.1,
            aggressive_fusion: false,
            target: CodegenTarget::Auto,
        }
    }
}

impl FusionConfig {
    /// Create conservative fusion configuration
    pub fn conservative() -> Self {
        Self {
            max_fusion_size: 4,
            min_speedup_ratio: 1.5,
            max_memory_overhead: 0.05,
            aggressive_fusion: false,
            target: CodegenTarget::Auto,
        }
    }

    /// Create aggressive fusion configuration
    pub fn aggressive() -> Self {
        Self {
            max_fusion_size: 16,
            min_speedup_ratio: 1.1,
            max_memory_overhead: 0.2,
            aggressive_fusion: true,
            target: CodegenTarget::Auto,
        }
    }

    /// Set target platform
    pub fn with_target(mut self, target: CodegenTarget) -> Self {
        self.target = target;
        self
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), AcousticError> {
        if self.max_fusion_size == 0 {
            return Err(AcousticError::ConfigurationError {
                field: "max_fusion_size".to_string(),
                message: "Must be greater than 0".to_string(),
            });
        }

        if self.min_speedup_ratio < 1.0 {
            return Err(AcousticError::ConfigurationError {
                field: "min_speedup_ratio".to_string(),
                message: "Must be at least 1.0".to_string(),
            });
        }

        if self.max_memory_overhead < 0.0 {
            return Err(AcousticError::ConfigurationError {
                field: "max_memory_overhead".to_string(),
                message: "Must be non-negative".to_string(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_kernel_fusion_creation() {
        let device = Device::Cpu;
        let fusion = KernelFusion::new(device);
        
        assert!(fusion.is_enabled());
        assert_eq!(fusion.cache_stats().size, 0);
    }

    #[test]
    fn test_fusion_config_validation() {
        let config = FusionConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = FusionConfig {
            max_fusion_size: 0,
            ..FusionConfig::default()
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_fusion_config_presets() {
        let conservative = FusionConfig::conservative();
        assert_eq!(conservative.max_fusion_size, 4);
        assert_eq!(conservative.min_speedup_ratio, 1.5);

        let aggressive = FusionConfig::aggressive();
        assert_eq!(aggressive.max_fusion_size, 16);
        assert_eq!(aggressive.min_speedup_ratio, 1.1);
        assert!(aggressive.aggressive_fusion);
    }

    #[test]
    fn test_kernel_fusion_enable_disable() {
        let device = Device::Cpu;
        let mut fusion = KernelFusion::new(device);
        
        assert!(fusion.is_enabled());
        
        fusion.set_enabled(false);
        assert!(!fusion.is_enabled());
        
        fusion.set_enabled(true);
        assert!(fusion.is_enabled());
    }

    #[test]
    fn test_cache_operations() {
        let device = Device::Cpu;
        let mut fusion = KernelFusion::new(device);
        
        let initial_stats = fusion.cache_stats();
        assert_eq!(initial_stats.size, 0);
        
        fusion.clear_cache();
        let cleared_stats = fusion.cache_stats();
        assert_eq!(cleared_stats.size, 0);
    }
}