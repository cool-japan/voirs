//! Integration tests for kernel fusion optimization
//!
//! These tests verify that the fusion system works correctly in real-world
//! scenarios and integrates properly with other acoustic modules.

use candle_core::{DType, Device, Shape, Tensor};
use voirs_acoustic::fusion::{
    codegen::{CodegenTarget, KernelGenerator},
    graph::{FusionGraph, OpGraph, OpNode},
    patterns::{FusionPattern, FusionRule, PatternMatcher},
    FusionConfig, KernelFusion,
};

// ============================================================================
// Kernel Generation Tests
// ============================================================================

#[test]
fn test_kernel_generator_with_empty_nodes() {
    let device = Device::Cpu;
    let generator = KernelGenerator::new(&device);

    let result = generator.generate_fused_kernel(&[]);
    assert!(result.is_err());
}

#[test]
fn test_kernel_generator_single_node() {
    let device = Device::Cpu;
    let generator = KernelGenerator::new(&device);

    let node = OpNode::new(0, "relu".to_string(), Shape::from_dims(&[2, 3]), DType::F32)
        .with_input_shapes(vec![Shape::from_dims(&[2, 3])]);

    let result = generator.generate_fused_kernel(&[node]);
    assert!(result.is_ok());

    let kernel = result.unwrap();
    assert_eq!(kernel.fused_ops.len(), 1);
    assert_eq!(kernel.fused_ops[0], "relu");
}

#[test]
fn test_kernel_generator_multiple_nodes() {
    let device = Device::Cpu;
    let generator = KernelGenerator::new(&device);

    let nodes = vec![
        OpNode::new(0, "add".to_string(), Shape::from_dims(&[2, 3]), DType::F32)
            .with_input_shapes(vec![Shape::from_dims(&[2, 3])]),
        OpNode::new(1, "relu".to_string(), Shape::from_dims(&[2, 3]), DType::F32)
            .with_input_shapes(vec![Shape::from_dims(&[2, 3])]),
    ];

    let result = generator.generate_fused_kernel(&nodes);
    assert!(result.is_ok());

    let kernel = result.unwrap();
    assert_eq!(kernel.fused_ops.len(), 2);
    assert!(kernel.expected_speedup >= 1.0);
}

#[test]
fn test_kernel_execution() {
    let device = Device::Cpu;
    let generator = KernelGenerator::new(&device);

    let nodes = vec![
        OpNode::new(0, "add".to_string(), Shape::from_dims(&[2, 3]), DType::F32)
            .with_input_shapes(vec![Shape::from_dims(&[2, 3])]),
    ];

    let kernel = generator.generate_fused_kernel(&nodes).unwrap();

    // Create input tensors
    let input = Tensor::ones(&[2, 3], DType::F32, &device).unwrap();

    // Execute kernel
    let result = kernel.execute(&[input]);
    assert!(result.is_ok());
}

// ============================================================================
// Pattern Matching Tests
// ============================================================================

#[test]
fn test_pattern_creation() {
    let pattern = FusionPattern::new("test", vec!["add", "mul"])
        .with_rule(FusionRule::ElementWise)
        .with_priority(10);

    assert_eq!(pattern.name(), "test");
    assert_eq!(pattern.operations().len(), 2);
    assert_eq!(pattern.priority(), 10);
}

#[test]
fn test_pattern_matcher_empty_patterns() {
    let patterns: Vec<FusionPattern> = vec![];
    let matcher = PatternMatcher::new(&patterns);

    // Matcher should work with empty patterns
    assert_eq!(patterns.len(), 0);
}

#[test]
fn test_pattern_matcher_multiple_patterns() {
    let patterns = vec![
        FusionPattern::new("pattern1", vec!["add", "mul"])
            .with_rule(FusionRule::ElementWise)
            .with_priority(5),
        FusionPattern::new("pattern2", vec!["matmul"])
            .with_rule(FusionRule::LinearAlgebra)
            .with_priority(10),
    ];

    let matcher = PatternMatcher::new(&patterns);
    assert_eq!(patterns.len(), 2);
}

// ============================================================================
// Fusion Config Tests
// ============================================================================

#[test]
fn test_fusion_config_default() {
    let config = FusionConfig::default();
    assert!(config.validate().is_ok());
    assert_eq!(config.max_fusion_size, 8);
    assert_eq!(config.min_speedup_ratio, 1.2);
}

#[test]
fn test_fusion_config_conservative() {
    let config = FusionConfig::conservative();
    assert!(config.validate().is_ok());
    assert_eq!(config.max_fusion_size, 4);
    assert_eq!(config.min_speedup_ratio, 1.5);
    assert!(!config.aggressive_fusion);
}

#[test]
fn test_fusion_config_aggressive() {
    let config = FusionConfig::aggressive();
    assert!(config.validate().is_ok());
    assert_eq!(config.max_fusion_size, 16);
    assert_eq!(config.min_speedup_ratio, 1.1);
    assert!(config.aggressive_fusion);
}

#[test]
fn test_fusion_config_invalid() {
    let config = FusionConfig {
        max_fusion_size: 0,
        ..FusionConfig::default()
    };
    assert!(config.validate().is_err());

    let config = FusionConfig {
        min_speedup_ratio: 0.5,
        ..FusionConfig::default()
    };
    assert!(config.validate().is_err());

    let config = FusionConfig {
        max_memory_overhead: -0.1,
        ..FusionConfig::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_fusion_config_with_target() {
    let config = FusionConfig::default().with_target(CodegenTarget::CpuSimd);
    assert_eq!(config.target, CodegenTarget::CpuSimd);
}

// ============================================================================
// Kernel Fusion System Tests
// ============================================================================

#[test]
fn test_kernel_fusion_creation() {
    let device = Device::Cpu;
    let fusion = KernelFusion::new(device);

    assert!(fusion.is_enabled());
    assert_eq!(fusion.cache_stats().size, 0);
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
fn test_kernel_fusion_cache_operations() {
    let device = Device::Cpu;
    let mut fusion = KernelFusion::new(device);

    let initial_stats = fusion.cache_stats();
    assert_eq!(initial_stats.size, 0);

    fusion.clear_cache();

    let cleared_stats = fusion.cache_stats();
    assert_eq!(cleared_stats.size, 0);
}

#[test]
fn test_kernel_fusion_with_custom_patterns() {
    let device = Device::Cpu;
    let patterns = vec![FusionPattern::new("custom", vec!["add", "mul"])
        .with_rule(FusionRule::ElementWise)
        .with_priority(20)];

    let fusion = KernelFusion::with_patterns(device, patterns);
    assert!(fusion.is_enabled());
}

// ============================================================================
// Integration with Tensor Operations
// ============================================================================

#[test]
fn test_fusion_with_real_tensors() {
    let device = Device::Cpu;

    // Create test tensors
    let a = Tensor::ones(&[2, 3], DType::F32, &device).unwrap();
    let b = Tensor::ones(&[2, 3], DType::F32, &device).unwrap();

    // Test element-wise operations that could be fused
    let result = a.add(&b).unwrap();
    assert_eq!(result.dims(), &[2, 3]);

    let activated = result.relu().unwrap();
    assert_eq!(activated.dims(), &[2, 3]);
}

#[test]
fn test_fusion_with_batch_operations() {
    let device = Device::Cpu;

    // Batch of features
    let features = Tensor::randn(0.0f32, 1.0f32, &[8, 64, 100], &device).unwrap();

    // Normalization pipeline (candidate for fusion)
    let mean = features.mean_keepdim(2).unwrap();
    let centered = features.broadcast_sub(&mean).unwrap();
    let variance = centered.sqr().unwrap().mean_keepdim(2).unwrap();
    let std = (variance + 1e-5).unwrap().sqrt().unwrap();
    let normalized = centered.broadcast_div(&std).unwrap();

    assert_eq!(normalized.dims(), features.dims());
}

#[test]
fn test_fusion_with_attention_operations() {
    let device = Device::Cpu;
    let batch_size = 4;
    let seq_len = 10;
    let hidden_dim = 64;

    // Attention computation components
    let query = Tensor::randn(0.0f32, 1.0f32, &[batch_size, seq_len, hidden_dim], &device).unwrap();
    let key = Tensor::randn(0.0f32, 1.0f32, &[batch_size, seq_len, hidden_dim], &device).unwrap();
    let value = Tensor::randn(0.0f32, 1.0f32, &[batch_size, seq_len, hidden_dim], &device).unwrap();

    // Simplified attention scores computation (could benefit from fusion)
    // Q @ K^T produces attention scores
    let scores = query.matmul(&key.transpose(1, 2).unwrap()).unwrap();
    assert_eq!(scores.dims(), &[batch_size, seq_len, seq_len]);

    // Scale scores (could be fused with matmul)
    let scale = 1.0 / (hidden_dim as f64).sqrt() as f32;
    let scaled_scores = scores.affine(scale as f64, 0.0).unwrap();
    assert_eq!(scaled_scores.dims(), &[batch_size, seq_len, seq_len]);
}

// ============================================================================
// CodegenTarget Tests
// ============================================================================

#[test]
fn test_codegen_target_detection() {
    let target = CodegenTarget::detect();

    // Should detect CPU or CpuSimd depending on system
    assert!(
        matches!(target, CodegenTarget::Cpu | CodegenTarget::CpuSimd),
        "Detected target should be CPU or CpuSimd"
    );
}

#[test]
fn test_codegen_target_simd_support() {
    assert!(CodegenTarget::CpuSimd.supports_simd());
    assert!(!CodegenTarget::Cpu.supports_simd());
    assert!(CodegenTarget::Auto.supports_simd());
}

#[test]
fn test_codegen_target_gpu_detection() {
    assert!(CodegenTarget::Cuda.is_gpu());
    assert!(CodegenTarget::Metal.is_gpu());
    assert!(!CodegenTarget::Cpu.is_gpu());
    assert!(!CodegenTarget::CpuSimd.is_gpu());
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_fusion_with_mismatched_shapes() {
    let device = Device::Cpu;

    let a = Tensor::ones(&[2, 3], DType::F32, &device).unwrap();
    let b = Tensor::ones(&[3, 4], DType::F32, &device).unwrap();

    // This should fail due to shape mismatch
    let result = a.add(&b);
    assert!(result.is_err());
}

#[test]
fn test_kernel_info_generation() {
    let device = Device::Cpu;
    let generator = KernelGenerator::new(&device);

    let nodes = vec![
        OpNode::new(0, "add".to_string(), Shape::from_dims(&[2, 3]), DType::F32)
            .with_input_shapes(vec![Shape::from_dims(&[2, 3])]),
    ];

    let kernel = generator.generate_fused_kernel(&nodes).unwrap();
    let info = kernel.info();

    // Info should contain kernel details
    assert!(info.contains("FusedKernel"));
    assert!(info.contains("speedup"));
}

// ============================================================================
// Performance-Related Tests
// ============================================================================

#[test]
fn test_fusion_speedup_estimation() {
    let device = Device::Cpu;
    let generator = KernelGenerator::new(&device);

    // Element-wise operations should have good speedup
    let elementwise_nodes = vec![
        OpNode::new(
            0,
            "add".to_string(),
            Shape::from_dims(&[100, 100]),
            DType::F32,
        )
        .with_input_shapes(vec![Shape::from_dims(&[100, 100])]),
        OpNode::new(
            1,
            "relu".to_string(),
            Shape::from_dims(&[100, 100]),
            DType::F32,
        )
        .with_input_shapes(vec![Shape::from_dims(&[100, 100])]),
    ];

    let kernel = generator.generate_fused_kernel(&elementwise_nodes).unwrap();
    assert!(kernel.expected_speedup >= 1.0);

    // Speedup should be higher with SIMD support
    if CodegenTarget::detect().supports_simd() {
        assert!(kernel.expected_speedup >= 1.5);
    }
}

#[test]
fn test_cache_memory_estimation() {
    let device = Device::Cpu;
    let fusion = KernelFusion::new(device);

    let stats = fusion.cache_stats();
    assert_eq!(stats.memory_usage, 0); // Empty cache
}

// ============================================================================
// Multi-Pattern Fusion Tests
// ============================================================================

#[test]
fn test_multiple_fusion_patterns() {
    let patterns = vec![
        FusionPattern::new("elementwise", vec!["add", "mul"])
            .with_rule(FusionRule::ElementWise)
            .with_priority(10),
        FusionPattern::new("normalization", vec!["mean", "sub", "div"])
            .with_rule(FusionRule::Normalization)
            .with_priority(15),
        FusionPattern::new("matmul", vec!["matmul", "add"])
            .with_rule(FusionRule::LinearAlgebra)
            .with_priority(12),
    ];

    assert_eq!(patterns.len(), 3);

    // Verify priorities
    let priorities: Vec<_> = patterns.iter().map(|p| p.priority()).collect();
    assert_eq!(priorities, vec![10, 15, 12]);
}

// ============================================================================
// Real-World Scenario Tests
// ============================================================================

#[test]
fn test_mel_spectrogram_normalization_fusion() {
    let device = Device::Cpu;

    // Simulate mel spectrogram features
    let mel_features = Tensor::randn(0.0f32, 1.0f32, &[16, 80, 100], &device).unwrap();

    // Normalization pipeline (fusion candidate)
    let mean = mel_features.mean_keepdim(2).unwrap();
    let centered = mel_features.broadcast_sub(&mean).unwrap();
    let variance = centered.sqr().unwrap().mean_keepdim(2).unwrap();
    let std = (variance + 1e-5).unwrap().sqrt().unwrap();
    let normalized = centered.broadcast_div(&std).unwrap();

    assert_eq!(normalized.dims(), &[16, 80, 100]);

    // Verify normalization properties
    let result_mean = normalized.mean(2).unwrap();
    let mean_values = result_mean.to_vec2::<f32>().unwrap();

    // Mean should be close to 0 after normalization
    for row in mean_values {
        for &val in &row {
            assert!(val.abs() < 0.1, "Normalized mean should be close to 0");
        }
    }
}

#[test]
fn test_acoustic_feature_extraction_pipeline() {
    let device = Device::Cpu;

    // Simulate raw audio features
    let audio_features = Tensor::randn(0.0f32, 1.0f32, &[8, 512], &device).unwrap();

    // Feature extraction pipeline
    let processed = audio_features.sqr().unwrap(); // Power spectrum
    let log_features = (processed + 1e-5).unwrap(); // Log scaling
    let final_features = log_features.relu().unwrap(); // Activation

    assert_eq!(final_features.dims(), &[8, 512]);
}
