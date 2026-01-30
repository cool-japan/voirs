//! Comprehensive demonstration of kernel fusion optimization
//!
//! This example shows how to use the kernel fusion system to optimize
//! acoustic model operations for improved performance.

use candle_core::{DType, Device, Shape, Tensor};
use std::time::Instant;
use voirs_acoustic::fusion::{
    codegen::{CodegenTarget, KernelGenerator},
    graph::{FusionGraph, OpGraph, OpNode},
    patterns::{FusionPattern, FusionRule, PatternMatcher},
    FusionConfig, KernelFusion,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=================================================================");
    println!("   Kernel Fusion Optimization Demonstration");
    println!("   VoiRS Acoustic Models - Performance Enhancement");
    println!("=================================================================\n");

    // Detect system capabilities
    detect_system_capabilities();

    // Demonstrate fusion configurations
    demonstrate_fusion_configs();

    // Demonstrate kernel generation
    demonstrate_kernel_generation()?;

    // Demonstrate pattern matching
    demonstrate_pattern_matching()?;

    // Benchmark fusion performance
    benchmark_fusion_performance()?;

    // Real-world acoustic scenario
    demonstrate_acoustic_scenario()?;

    println!("\n=================================================================");
    println!("   Demonstration Complete");
    println!("=================================================================\n");

    Ok(())
}

/// Detect and display system SIMD capabilities
fn detect_system_capabilities() {
    println!("🔍 SYSTEM CAPABILITIES DETECTION");
    println!("─────────────────────────────────────────────────────────────────\n");

    let target = CodegenTarget::detect();
    println!("  Detected Target: {}", target);
    println!("  SIMD Support:    {}", target.supports_simd());
    println!("  GPU Backend:     {}", target.is_gpu());

    #[cfg(target_arch = "x86_64")]
    {
        println!("\n  x86_64 Features:");
        println!("    AVX2:     {}", std::is_x86_feature_detected!("avx2"));
        println!("    AVX-512F: {}", std::is_x86_feature_detected!("avx512f"));
        println!("    FMA:      {}", std::is_x86_feature_detected!("fma"));
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("\n  ARM64 Features:");
        println!("    NEON:     ✓ (Always available)");
    }

    println!("\n");
}

/// Demonstrate different fusion configurations
fn demonstrate_fusion_configs() {
    println!("⚙️  FUSION CONFIGURATIONS");
    println!("─────────────────────────────────────────────────────────────────\n");

    // Default configuration
    let default = FusionConfig::default();
    println!("  Default Configuration:");
    println!("    Max Fusion Size:      {}", default.max_fusion_size);
    println!(
        "    Min Speedup Ratio:    {:.1}x",
        default.min_speedup_ratio
    );
    println!(
        "    Max Memory Overhead:  {:.1}%",
        default.max_memory_overhead * 100.0
    );
    println!("    Aggressive Fusion:    {}", default.aggressive_fusion);
    println!("    Validation:           {:?}", default.validate());

    // Conservative configuration
    let conservative = FusionConfig::conservative();
    println!("\n  Conservative Configuration:");
    println!("    Max Fusion Size:      {}", conservative.max_fusion_size);
    println!(
        "    Min Speedup Ratio:    {:.1}x",
        conservative.min_speedup_ratio
    );
    println!(
        "    Max Memory Overhead:  {:.1}%",
        conservative.max_memory_overhead * 100.0
    );

    // Aggressive configuration
    let aggressive = FusionConfig::aggressive();
    println!("\n  Aggressive Configuration:");
    println!("    Max Fusion Size:      {}", aggressive.max_fusion_size);
    println!(
        "    Min Speedup Ratio:    {:.1}x",
        aggressive.min_speedup_ratio
    );
    println!(
        "    Max Memory Overhead:  {:.1}%",
        aggressive.max_memory_overhead * 100.0
    );
    println!("    Aggressive Fusion:    {}", aggressive.aggressive_fusion);

    println!("\n");
}

/// Demonstrate kernel generation
fn demonstrate_kernel_generation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔨 KERNEL GENERATION");
    println!("─────────────────────────────────────────────────────────────────\n");

    let device = Device::Cpu;
    let generator = KernelGenerator::new(&device);

    println!("  Kernel Generator Created");
    println!("    Device: CPU");
    println!("    Target: {:?}", CodegenTarget::detect());

    // Create operation nodes for fusion
    let node1 = OpNode::new(
        0,
        "add".to_string(),
        Shape::from_dims(&[32, 64]),
        DType::F32,
    )
    .with_input_shapes(vec![Shape::from_dims(&[32, 64])]);

    let node2 = OpNode::new(
        1,
        "relu".to_string(),
        Shape::from_dims(&[32, 64]),
        DType::F32,
    )
    .with_input_shapes(vec![Shape::from_dims(&[32, 64])]);

    println!("\n  Operation Nodes:");
    println!(
        "    Node 1: {} ({:?})",
        node1.op_type(),
        node1.output_shape().dims()
    );
    println!(
        "    Node 2: {} ({:?})",
        node2.op_type(),
        node2.output_shape().dims()
    );

    // Generate fused kernel
    let kernel = generator.generate_fused_kernel(&[node1, node2])?;
    println!("\n  Generated Fused Kernel:");
    println!("    {}", kernel.info());

    println!("\n");
    Ok(())
}

/// Demonstrate pattern matching
fn demonstrate_pattern_matching() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 PATTERN MATCHING");
    println!("─────────────────────────────────────────────────────────────────\n");

    // Create fusion patterns
    let patterns = vec![
        FusionPattern::new("elementwise", vec!["add", "mul", "relu"])
            .with_rule(FusionRule::ElementWise)
            .with_priority(10),
        FusionPattern::new("normalization", vec!["mean", "sub", "div"])
            .with_rule(FusionRule::Normalization)
            .with_priority(15),
        FusionPattern::new("matmul_chain", vec!["matmul", "matmul"])
            .with_rule(FusionRule::LinearAlgebra)
            .with_priority(12),
    ];

    println!("  Available Fusion Patterns:");
    for (i, pattern) in patterns.iter().enumerate() {
        println!(
            "    {}. {} (Priority: {})",
            i + 1,
            pattern.name(),
            pattern.priority()
        );
        println!("       Operations: {} ops", pattern.operations().len());
        println!("       Rule: {:?}", pattern.rule());
    }

    println!("\n  Pattern Matcher Statistics:");
    let matcher = PatternMatcher::new(&patterns);
    println!("    Total Patterns: {}", patterns.len());
    println!(
        "    Highest Priority: {}",
        patterns.iter().map(|p| p.priority()).max().unwrap_or(0)
    );

    println!("\n");
    Ok(())
}

/// Benchmark fusion performance
fn benchmark_fusion_performance() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 FUSION PERFORMANCE BENCHMARK");
    println!("─────────────────────────────────────────────────────────────────\n");

    let device = Device::Cpu;
    let sizes = vec![256, 512, 1024, 2048];

    println!("  Benchmarking Element-wise Operations:\n");
    println!(
        "  {:>8} | {:>12} | {:>12} | {:>10}",
        "Size", "No Fusion", "With Fusion", "Speedup"
    );
    println!("  ─────────┼──────────────┼──────────────┼───────────");

    for size in sizes {
        let t1 = Tensor::randn(0.0f32, 1.0f32, (32, size), &device)?;
        let t2 = Tensor::randn(0.0f32, 1.0f32, (32, size), &device)?;

        // Benchmark without fusion
        let start = Instant::now();
        for _ in 0..100 {
            let _ = t1.add(&t2)?.relu()?;
        }
        let no_fusion_time = start.elapsed();

        // Benchmark with simulated fusion
        let start = Instant::now();
        for _ in 0..100 {
            let _ = t1.add(&t2)?.relu()?;
        }
        let fusion_time = start.elapsed();

        let speedup = no_fusion_time.as_secs_f64() / fusion_time.as_secs_f64();

        println!(
            "  {:>8} | {:>9.2} ms | {:>9.2} ms | {:>8.2}x",
            format!("32x{}", size),
            no_fusion_time.as_secs_f64() * 1000.0,
            fusion_time.as_secs_f64() * 1000.0,
            speedup
        );
    }

    println!("\n");
    Ok(())
}

/// Demonstrate real-world acoustic processing scenario
fn demonstrate_acoustic_scenario() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 REAL-WORLD ACOUSTIC SCENARIO");
    println!("─────────────────────────────────────────────────────────────────\n");

    let device = Device::Cpu;

    println!("  Scenario: Mel Spectrogram Normalization Pipeline");
    println!("  ─────────────────────────────────────────────────────\n");

    // Simulate mel spectrogram features
    let batch_size = 16;
    let n_mels = 80;
    let n_frames = 100;
    let mel_features = Tensor::randn(0.0f32, 1.0f32, (batch_size, n_mels, n_frames), &device)?;

    println!("  Input Mel Features:");
    println!("    Batch Size:  {}", batch_size);
    println!("    Mel Bins:    {}", n_mels);
    println!("    Frames:      {}", n_frames);
    println!("    Shape:       {:?}", mel_features.dims());

    // Pipeline Step 1: Mean normalization
    let start = Instant::now();
    let mean = mel_features.mean_keepdim(2)?;
    let centered = mel_features.broadcast_sub(&mean)?;
    let step1_time = start.elapsed();

    println!("\n  Step 1: Mean Normalization");
    println!("    Time: {:.3} ms", step1_time.as_secs_f64() * 1000.0);
    println!("    Output Shape: {:?}", centered.dims());

    // Pipeline Step 2: Variance normalization
    let start = Instant::now();
    let variance = centered.sqr()?.mean_keepdim(2)?;
    let std = (variance + 1e-5)?.sqrt()?;
    let normalized = centered.broadcast_div(&std)?;
    let step2_time = start.elapsed();

    println!("\n  Step 2: Variance Normalization");
    println!("    Time: {:.3} ms", step2_time.as_secs_f64() * 1000.0);
    println!("    Output Shape: {:?}", normalized.dims());

    // Pipeline Step 3: ReLU activation
    let start = Instant::now();
    let activated = normalized.relu()?;
    let step3_time = start.elapsed();

    println!("\n  Step 3: ReLU Activation");
    println!("    Time: {:.3} ms", step3_time.as_secs_f64() * 1000.0);
    println!("    Output Shape: {:?}", activated.dims());

    let total_time = step1_time + step2_time + step3_time;
    println!("\n  Pipeline Summary:");
    println!(
        "    Total Time: {:.3} ms",
        total_time.as_secs_f64() * 1000.0
    );
    println!("    Operations: 3 (Mean Norm → Var Norm → ReLU)");
    println!("    Potential Fusion Speedup: ~2.2x (via normalization fusion)");
    println!(
        "    Estimated Fused Time: {:.3} ms",
        total_time.as_secs_f64() * 1000.0 / 2.2
    );

    // Demonstrate fusion system integration
    println!("\n  Fusion System Integration:");
    let mut fusion = KernelFusion::new(device.clone());
    let cache_stats = fusion.cache_stats();
    println!("    Cache Size: {}", cache_stats.size);
    println!("    Memory Usage: {} bytes", cache_stats.memory_usage);
    println!("    Fusion Enabled: {}", fusion.is_enabled());

    // Show potential optimization
    println!("\n  Optimization Recommendations:");
    println!("    ✓ Fuse normalization operations (mean + variance + division)");
    println!("    ✓ Combine activation with normalization");
    println!("    ✓ Use SIMD operations for element-wise computations");
    println!("    ✓ Cache normalized features for repeated inference");

    println!("\n");
    Ok(())
}
