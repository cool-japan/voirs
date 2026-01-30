//! # Neural Architecture Search Demo
//!
//! This example demonstrates automatic neural architecture optimization using
//! evolutionary algorithms to find optimal model configurations.

use voirs_singing::prelude::*;
use voirs_singing::{EnergyEfficiencyConfig, EnergyEfficiencyOptimizer};
use voirs_singing::{HardwareOptimizer, HardwareOptimizerConfig, HardwarePlatform};

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== VoiRS Neural Architecture Search Demo ===\n");

    // 1. Configure NAS
    println!("1. Configuring Neural Architecture Search...");
    let nas_config = NasConfig {
        search_iterations: 50,
        population_size: 10,
        mutation_rate: 0.2,
        crossover_rate: 0.7,
        performance_weight: 0.6,
        efficiency_weight: 0.4,
    };
    let mut searcher = NeuralArchitectureSearcher::new(nas_config);
    println!(
        "   ✓ NAS configured with {} iterations and population size {}\n",
        50, 10
    );

    // 2. Run architecture search
    println!("2. Running architecture search...");
    println!("   (This may take a moment...)\n");
    let optimal_arch = searcher.search().await?;

    println!("   ✓ Search complete!\n");

    // 3. Display optimal architecture
    println!("3. Optimal Architecture Found:");
    println!("   ID: {}", optimal_arch.id);
    println!("   Number of Layers: {}", optimal_arch.num_layers);
    println!(
        "   Hidden Dimensions: {:?}",
        &optimal_arch.hidden_dims[..3.min(optimal_arch.hidden_dims.len())]
    );
    println!(
        "   Attention Heads: {:?}",
        &optimal_arch.attention_heads[..3.min(optimal_arch.attention_heads.len())]
    );
    println!(
        "   Feed-Forward Expansion: {:.2}",
        optimal_arch.ff_expansion
    );
    println!(
        "   Performance Score: {:.3}",
        optimal_arch.performance_score
    );
    println!("   Efficiency Score: {:.3}", optimal_arch.efficiency_score);
    println!("   Fitness: {:.3}", optimal_arch.fitness);
    println!(
        "   Estimated Size: {} parameters\n",
        optimal_arch.estimate_size()
    );

    // 4. Get top architectures
    println!("4. Top 5 Architectures:");
    let top_archs = searcher.get_top_architectures(5);
    for (i, arch) in top_archs.iter().enumerate() {
        println!(
            "   #{} - Layers: {}, Fitness: {:.3}, Size: {}M params",
            i + 1,
            arch.num_layers,
            arch.fitness,
            arch.estimate_size() / 1_000_000
        );
    }
    println!();

    // 5. Model Compression Demo
    println!("5. Model Compression Demo:");
    let compression_config = ModelCompressionConfig {
        compression_ratio: 0.7,
        pruning_threshold: 0.005,
        quantization_bits: 8,
        quality_threshold: 0.90,
    };
    let compressor = ModelCompressor::new(compression_config);

    let original_size = optimal_arch.estimate_size();
    let compressed = compressor.compress(original_size).await?;

    println!(
        "   Original Size: {}M parameters",
        original_size / 1_000_000
    );
    println!(
        "   Compressed Size: {}M parameters",
        compressed.compressed_size / 1_000_000
    );
    println!(
        "   Compression Ratio: {:.1}%",
        compressed.compression_ratio * 100.0
    );
    println!(
        "   Quality Degradation: {:.1}%",
        compressed.quality_degradation * 100.0
    );
    println!("   Speedup Factor: {:.2}x", compressed.speedup_factor);
    println!("   Techniques Applied:");
    for technique in &compressed.techniques {
        println!("     - {}", technique);
    }
    println!();

    // 6. Hardware Optimization Demo
    println!("6. Hardware-Specific Optimization:");
    let platforms = [
        ("CPU x86", HardwarePlatform::CpuX86),
        ("CPU ARM", HardwarePlatform::CpuArm),
        ("GPU NVIDIA", HardwarePlatform::GpuNvidia),
        ("Mobile", HardwarePlatform::Mobile),
    ];

    for (name, platform) in &platforms {
        let hw_config = HardwareOptimizerConfig {
            platform: *platform,
            enable_simd: true,
            enable_fusion: true,
            memory_opt_level: 2,
        };
        let optimizer = HardwareOptimizer::new(hw_config);
        let result = optimizer.optimize().await?;

        println!("   {} Optimization:", name);
        println!("     Speedup: {:.1}x", result.speedup_factor);
        println!(
            "     Memory Reduction: {:.0}%",
            result.memory_reduction * 100.0
        );
        println!("     Optimizations: {}", result.optimizations.join(", "));
    }
    println!();

    // 7. Energy Efficiency Optimization
    println!("7. Energy-Efficient Inference:");
    let energy_config = EnergyEfficiencyConfig {
        power_budget: 50.0,
        enable_dvfs: true,
        optimize_batch_size: true,
        reduce_precision: true,
    };
    let energy_optimizer = EnergyEfficiencyOptimizer::new(energy_config);
    let energy_result = energy_optimizer.optimize().await?;

    println!(
        "   Power Consumption: {:.1}W",
        energy_result.power_consumption
    );
    println!(
        "   Energy Reduction: {:.0}%",
        energy_result.energy_reduction * 100.0
    );
    println!(
        "   Performance Impact: {:.1}%",
        energy_result.performance_impact * 100.0
    );
    println!("   Techniques:");
    for technique in &energy_result.techniques {
        println!("     - {}", technique);
    }

    println!("\n=== Demo Complete ===");
    println!("Neural Architecture Search automatically finds optimal model configurations,");
    println!("while compression and hardware-specific optimizations maximize efficiency.");

    Ok(())
}
