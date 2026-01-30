//! Advanced Memory Optimization Example
//!
//! Demonstrates memory-efficient synthesis techniques including:
//! - Memory pooling and reuse
//! - Streaming synthesis for long sequences
//! - Memory pressure handling
//! - Batch size optimization
//! - Memory profiling and monitoring

use std::time::Instant;
use voirs_acoustic::{Phoneme, SynthesisConfig};

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Initial pool size in megabytes
    pub initial_pool_size_mb: usize,
    /// Maximum pool size in megabytes
    pub max_pool_size_mb: usize,
    /// Enable dynamic growth
    pub enable_dynamic_growth: bool,
    /// Growth factor when expanding pool
    pub growth_factor: f32,
    /// Shrink threshold in megabytes
    pub shrink_threshold_mb: usize,
    /// Buffer alignment in bytes
    pub buffer_alignment_bytes: usize,
}

/// Memory optimization strategies
#[derive(Debug, Clone, Copy)]
enum OptimizationStrategy {
    /// Standard synthesis without optimization
    Standard,
    /// Use memory pooling
    Pooled,
    /// Use streaming for long sequences
    Streaming,
    /// Use adaptive batch sizing
    AdaptiveBatch,
}

impl std::fmt::Display for OptimizationStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizationStrategy::Standard => write!(f, "Standard"),
            OptimizationStrategy::Pooled => write!(f, "Memory Pooled"),
            OptimizationStrategy::Streaming => write!(f, "Streaming"),
            OptimizationStrategy::AdaptiveBatch => write!(f, "Adaptive Batch"),
        }
    }
}

/// Memory usage snapshot
#[derive(Debug, Clone)]
struct MemorySnapshot {
    strategy: OptimizationStrategy,
    peak_memory_mb: f64,
    average_memory_mb: f64,
    total_allocations: usize,
    allocation_overhead_percent: f64,
    processing_time_ms: u128,
}

impl MemorySnapshot {
    fn print(&self) {
        println!("  Strategy:              {}", self.strategy);
        println!("  Peak Memory:           {:.2} MB", self.peak_memory_mb);
        println!("  Average Memory:        {:.2} MB", self.average_memory_mb);
        println!("  Total Allocations:     {}", self.total_allocations);
        println!(
            "  Allocation Overhead:   {:.1}%",
            self.allocation_overhead_percent
        );
        println!("  Processing Time:       {} ms", self.processing_time_ms);
    }

    fn efficiency_score(&self) -> f64 {
        // Lower is better - combines memory and time efficiency
        (self.peak_memory_mb / 100.0) + (self.processing_time_ms as f64 / 1000.0)
    }
}

/// Simulate synthesis with different memory strategies
fn synthesize_with_strategy(
    phonemes: &[Phoneme],
    _config: &SynthesisConfig,
    strategy: OptimizationStrategy,
) -> MemorySnapshot {
    let start = Instant::now();
    let peak_memory;
    let total_memory;
    let samples;
    let allocations;

    match strategy {
        OptimizationStrategy::Standard => {
            // Standard allocation - one large buffer
            allocations = 1;
            let buffer_size_mb = (phonemes.len() * 256 * 4) as f64 / 1024.0 / 1024.0;
            peak_memory = buffer_size_mb;
            total_memory = buffer_size_mb;
            samples = 1;

            // Simulate processing
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        OptimizationStrategy::Pooled => {
            // Memory pooling - reuse buffers
            allocations = 3; // Pool, buffer, result
            let pool_size_mb = 10.0;
            let buffer_size_mb = (phonemes.len() * 256 * 4) as f64 / 1024.0 / 1024.0;
            peak_memory = pool_size_mb + buffer_size_mb;
            total_memory = pool_size_mb; // Amortized over many calls
            samples = 1;

            // Simulate processing with pooling overhead
            std::thread::sleep(std::time::Duration::from_millis(95));
        }
        OptimizationStrategy::Streaming => {
            // Streaming synthesis - process in chunks
            let chunk_size = 50; // phonemes per chunk
            let chunks = (phonemes.len() + chunk_size - 1) / chunk_size;
            allocations = chunks * 2; // Buffer + result per chunk

            let chunk_buffer_mb = (chunk_size * 256 * 4) as f64 / 1024.0 / 1024.0;
            peak_memory = chunk_buffer_mb * 2.0; // Double buffering
            total_memory = chunk_buffer_mb * chunks as f64;
            samples = chunks;

            // Simulate streaming processing
            std::thread::sleep(std::time::Duration::from_millis(110)); // Slight overhead
        }
        OptimizationStrategy::AdaptiveBatch => {
            // Adaptive batching based on available memory
            let available_memory_mb = 256.0; // Simulated available memory
            let max_batch_size = ((available_memory_mb * 0.8) / 4.0) as usize; // 80% utilization
            let batch_size = max_batch_size.min(phonemes.len());

            let batches = (phonemes.len() + batch_size - 1) / batch_size;
            allocations = batches;

            let batch_buffer_mb = (batch_size * 256 * 4) as f64 / 1024.0 / 1024.0;
            peak_memory = batch_buffer_mb;
            total_memory = batch_buffer_mb;
            samples = batches;

            // Simulate adaptive processing
            std::thread::sleep(std::time::Duration::from_millis(105));
        }
    }

    let duration = start.elapsed();

    MemorySnapshot {
        strategy,
        peak_memory_mb: peak_memory,
        average_memory_mb: total_memory / samples as f64,
        total_allocations: allocations,
        allocation_overhead_percent: (allocations as f64 / samples as f64 - 1.0) * 100.0,
        processing_time_ms: duration.as_millis(),
    }
}

fn main() {
    println!("VoiRS Acoustic - Advanced Memory Optimization Example\n");

    // Test different sequence lengths
    let test_cases = vec![
        ("Short sequence", 50),
        ("Medium sequence", 200),
        ("Long sequence", 1000),
        ("Very long sequence", 5000),
    ];

    for (description, phoneme_count) in test_cases {
        println!("=== {} ({} phonemes) ===\n", description, phoneme_count);

        // Generate phoneme sequence
        let phonemes: Vec<Phoneme> = (0..phoneme_count)
            .map(|i| {
                let symbols = ["HH", "EH", "L", "OW", "W", "ER", "D"];
                Phoneme::new(symbols[i % symbols.len()])
            })
            .collect();

        let config = SynthesisConfig::default();

        // Test all optimization strategies
        let strategies = vec![
            OptimizationStrategy::Standard,
            OptimizationStrategy::Pooled,
            OptimizationStrategy::Streaming,
            OptimizationStrategy::AdaptiveBatch,
        ];

        let mut results = Vec::new();

        for strategy in strategies {
            println!("Testing: {}", strategy);
            let snapshot = synthesize_with_strategy(&phonemes, &config, strategy);
            snapshot.print();
            println!(
                "  Efficiency Score:      {:.2}\n",
                snapshot.efficiency_score()
            );
            results.push(snapshot);
        }

        // Find best strategy
        let best = results
            .iter()
            .min_by(|a, b| {
                a.efficiency_score()
                    .partial_cmp(&b.efficiency_score())
                    .unwrap()
            })
            .unwrap();

        println!("✓ Best strategy for {}: {}", description, best.strategy);
        println!(
            "  Peak Memory: {:.2} MB, Time: {} ms\n",
            best.peak_memory_mb, best.processing_time_ms
        );
    }

    // Demonstrate memory pool configuration
    println!("=== Memory Pool Configuration Examples ===\n");

    println!("1. Conservative Configuration (Low Memory Systems):");
    let conservative = MemoryPoolConfig {
        initial_pool_size_mb: 64,
        max_pool_size_mb: 256,
        enable_dynamic_growth: true,
        growth_factor: 1.2,
        shrink_threshold_mb: 128,
        buffer_alignment_bytes: 64,
    };
    println!("{:#?}\n", conservative);

    println!("2. Balanced Configuration (General Purpose):");
    let balanced = MemoryPoolConfig {
        initial_pool_size_mb: 128,
        max_pool_size_mb: 512,
        enable_dynamic_growth: true,
        growth_factor: 1.5,
        shrink_threshold_mb: 256,
        buffer_alignment_bytes: 64,
    };
    println!("{:#?}\n", balanced);

    println!("3. Aggressive Configuration (High Performance):");
    let aggressive = MemoryPoolConfig {
        initial_pool_size_mb: 256,
        max_pool_size_mb: 1024,
        enable_dynamic_growth: true,
        growth_factor: 2.0,
        shrink_threshold_mb: 512,
        buffer_alignment_bytes: 64,
    };
    println!("{:#?}\n", aggressive);

    // Memory pressure handling recommendations
    println!("=== Memory Pressure Handling Recommendations ===\n");

    println!("For systems with < 2GB RAM:");
    println!("  - Use streaming synthesis for sequences > 100 phonemes");
    println!("  - Enable conservative memory pooling");
    println!("  - Set max_pool_size_mb to 256 MB or less");
    println!("  - Use batch_size <= 32 phonemes\n");

    println!("For systems with 2-8GB RAM:");
    println!("  - Use balanced memory configuration");
    println!("  - Enable adaptive batch sizing");
    println!("  - Set max_pool_size_mb to 512 MB");
    println!("  - Use batch_size <= 128 phonemes\n");

    println!("For systems with > 8GB RAM:");
    println!("  - Use aggressive memory configuration");
    println!("  - Disable streaming for sequences < 5000 phonemes");
    println!("  - Set max_pool_size_mb to 1024 MB or more");
    println!("  - Use batch_size <= 512 phonemes\n");

    println!("Memory optimization example completed!");
}
