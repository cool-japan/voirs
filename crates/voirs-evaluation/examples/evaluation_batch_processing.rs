//! Batch Processing and Performance Optimization Example
//!
//! This example demonstrates how to efficiently process large batches
//! of audio samples using the VoiRS evaluation framework with performance
//! optimizations including parallel processing and GPU acceleration.

use std::time::Instant;
use tokio::time::Duration;
use voirs_evaluation::performance::{GpuAccelerator, LRUCache, PerformanceMonitor};
use voirs_evaluation::prelude::*;
use voirs_sdk::AudioBuffer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ VoiRS Batch Processing & Performance Example");
    println!("===============================================");

    // Create evaluators
    println!("\n🏗️  Setting up evaluators and performance tools...");
    let quality_evaluator = QualityEvaluator::new().await?;
    let comparative_evaluator = ComparativeEvaluatorImpl::new().await?;

    // Setup performance monitoring
    let performance_monitor = PerformanceMonitor::new();
    let cache = LRUCache::new(100);

    // Setup GPU acceleration if available
    let gpu_accelerator = match GpuAccelerator::new() {
        Ok(gpu) => {
            println!("  🚀 GPU acceleration enabled: {:?}", gpu.device());
            Some(gpu)
        }
        Err(_) => {
            println!("  💻 Using CPU processing (GPU not available)");
            None
        }
    };

    // Example 1: Basic batch processing
    println!("\n📦 Example 1: Basic batch processing");

    let sample_rate = 16000;
    let duration_samples = sample_rate; // 1 second samples

    // Generate a batch of test samples
    let batch_size = 50;
    let batch_samples: Vec<(AudioBuffer, Option<AudioBuffer>)> = (0..batch_size)
        .map(|i| {
            let frequency = 200.0 + (i as f32 * 10.0); // Varying frequencies
            let samples: Vec<f32> = (0..duration_samples)
                .map(|j| {
                    let t = j as f32 / sample_rate as f32;
                    0.3 * (2.0 * std::f32::consts::PI * frequency * t).sin()
                })
                .collect();

            let audio = AudioBuffer::new(samples, sample_rate, 1);

            // Generate reference for some samples
            let reference = if i % 3 == 0 {
                let ref_samples: Vec<f32> = (0..duration_samples)
                    .map(|j| {
                        let t = j as f32 / sample_rate as f32;
                        0.32 * (2.0 * std::f32::consts::PI * (frequency + 5.0) * t).sin()
                    })
                    .collect();
                Some(AudioBuffer::new(ref_samples, sample_rate, 1))
            } else {
                None
            };

            (audio, reference)
        })
        .collect();

    println!("  Generated {batch_size} audio samples for batch processing");

    // Process batch with timing
    let start_time = Instant::now();
    let batch_results = performance_monitor.time_operation("batch_quality_evaluation", || {
        tokio::runtime::Handle::current().block_on(async {
            quality_evaluator
                .evaluate_quality_batch(&batch_samples, None)
                .await
        })
    })?;
    let processing_time = start_time.elapsed();

    println!("  Batch processing completed:");
    println!("    Total samples: {}", batch_results.len());
    println!("    Processing time: {:.2}s", processing_time.as_secs_f32());
    println!(
        "    Throughput: {:.1} samples/second",
        batch_size as f32 / processing_time.as_secs_f32()
    );

    // Calculate batch statistics
    let avg_score =
        batch_results.iter().map(|r| r.overall_score).sum::<f32>() / batch_results.len() as f32;
    let min_score = batch_results
        .iter()
        .map(|r| r.overall_score)
        .fold(f32::INFINITY, f32::min);
    let max_score = batch_results
        .iter()
        .map(|r| r.overall_score)
        .fold(f32::NEG_INFINITY, f32::max);

    println!("  Batch Quality Statistics:");
    println!("    Average Score: {avg_score:.3}");
    println!("    Score Range: {min_score:.3} - {max_score:.3}");

    // Example 2: Parallel batch processing with chunking
    println!("\n🔄 Example 2: Chunked parallel processing");

    let large_batch_size = 200;
    let chunk_size = 25;

    // Generate larger batch
    let large_batch: Vec<(AudioBuffer, Option<AudioBuffer>)> = (0..large_batch_size)
        .map(|i| {
            let frequency = 150.0 + (i as f32 * 5.0);
            let samples: Vec<f32> = (0..duration_samples)
                .map(|j| {
                    let t = j as f32 / sample_rate as f32;
                    0.25 * (2.0 * std::f32::consts::PI * frequency * t).sin()
                        + 0.1 * (2.0 * std::f32::consts::PI * (frequency * 2.0) * t).sin()
                })
                .collect();
            (AudioBuffer::new(samples, sample_rate, 1), None)
        })
        .collect();

    println!("  Processing {large_batch_size} samples in chunks of {chunk_size}");

    let chunked_start = Instant::now();
    let mut chunked_results = Vec::new();

    // Process in chunks to optimize memory usage
    for (chunk_idx, chunk) in large_batch.chunks(chunk_size).enumerate() {
        let chunk_start = Instant::now();
        let chunk_results = quality_evaluator
            .evaluate_quality_batch(chunk, None)
            .await?;
        let chunk_time = chunk_start.elapsed();

        println!(
            "    Chunk {} ({} samples): {:.2}s",
            chunk_idx + 1,
            chunk.len(),
            chunk_time.as_secs_f32()
        );

        chunked_results.extend(chunk_results);

        // Optional: Add small delay to prevent resource exhaustion
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let total_chunked_time = chunked_start.elapsed();
    println!("  Chunked processing completed:");
    println!("    Total time: {:.2}s", total_chunked_time.as_secs_f32());
    println!(
        "    Throughput: {:.1} samples/second",
        large_batch_size as f32 / total_chunked_time.as_secs_f32()
    );

    // Example 3: GPU-accelerated processing (if available)
    println!("\n🚀 Example 3: GPU-accelerated processing");

    if let Some(ref gpu) = gpu_accelerator {
        println!("  Testing GPU-accelerated operations...");

        // Test GPU correlation
        let test_signal1: Vec<f32> = (0..1024).map(|i| (i as f32 / 1024.0).sin()).collect();
        let test_signal2: Vec<f32> = (0..1024)
            .map(|i| ((i as f32 / 1024.0) + 0.1).sin())
            .collect();

        let gpu_start = Instant::now();
        let gpu_correlation = gpu.gpu_correlation(&test_signal1, &test_signal2)?;
        let gpu_time = gpu_start.elapsed();

        // Compare with CPU implementation
        let cpu_start = Instant::now();
        let cpu_correlation = voirs_evaluation::calculate_correlation(&test_signal1, &test_signal2);
        let cpu_time = cpu_start.elapsed();

        println!(
            "    GPU Correlation: {:.6} ({:.2}μs)",
            gpu_correlation,
            gpu_time.as_micros()
        );
        println!(
            "    CPU Correlation: {:.6} ({:.2}μs)",
            cpu_correlation,
            cpu_time.as_micros()
        );
        println!(
            "    Speedup: {:.2}x",
            cpu_time.as_nanos() as f32 / gpu_time.as_nanos() as f32
        );

        // Test GPU spectral analysis
        let spectral_start = Instant::now();
        let spectral_features = gpu.gpu_spectral_analysis(&test_signal1, sample_rate as f32)?;
        let spectral_time = spectral_start.elapsed();

        println!(
            "    GPU Spectral Analysis ({:.2}μs):",
            spectral_time.as_micros()
        );
        println!("      Centroid: {:.1} Hz", spectral_features.centroid);
        println!("      Spread: {:.1} Hz", spectral_features.spread);
        println!("      Rolloff: {:.1} Hz", spectral_features.rolloff);

        // Clear GPU memory
        gpu.clear_memory_pool();
    }

    // Example 4: Caching for repeated evaluations
    println!("\n💾 Example 4: Caching optimization");

    // Generate some test audio
    let cached_audio = AudioBuffer::new(vec![0.2; duration_samples as usize], sample_rate, 1);
    let cache_key = format!(
        "audio_{}_{}",
        cached_audio.samples().len(),
        cached_audio.samples().iter().sum::<f32>()
    );

    // First evaluation (cache miss)
    let cache_start = Instant::now();
    let first_result = quality_evaluator
        .evaluate_quality(&cached_audio, None, None)
        .await?;
    let first_time = cache_start.elapsed();

    // Store in cache
    cache.insert(cache_key.clone(), first_result.overall_score);

    // Second evaluation (check cache first)
    let cached_start = Instant::now();
    let cached_score = cache.get(&cache_key);
    let cached_time = cached_start.elapsed();

    if let Some(score) = cached_score {
        println!("  Cache performance:");
        println!(
            "    First evaluation: {:.2}ms (score: {:.3})",
            first_time.as_millis(),
            first_result.overall_score
        );
        println!(
            "    Cached retrieval: {:.2}μs (score: {:.3})",
            cached_time.as_micros(),
            score
        );
        println!(
            "    Cache speedup: {:.0}x",
            first_time.as_nanos() as f32 / cached_time.as_nanos() as f32
        );
    }

    // Example 5: Performance monitoring and profiling
    println!("\n📊 Example 5: Performance monitoring");

    // Run multiple operations for profiling
    for i in 0..5 {
        let test_audio = AudioBuffer::new(
            vec![0.1 + (i as f32 * 0.05); duration_samples as usize],
            sample_rate,
            1,
        );

        performance_monitor.time_operation("single_evaluation", || {
            tokio::runtime::Handle::current().block_on(async {
                quality_evaluator
                    .evaluate_quality(&test_audio, None, None)
                    .await
            })
        })?;
    }

    // Display performance statistics
    if let Some(avg_time) = performance_monitor.get_average_time("single_evaluation") {
        println!("  Performance Statistics:");
        println!(
            "    Average single evaluation time: {:.2}ms",
            avg_time.as_millis()
        );
    }

    if let Some(batch_avg) = performance_monitor.get_average_time("batch_quality_evaluation") {
        println!(
            "    Average batch evaluation time: {:.2}s",
            batch_avg.as_secs_f32()
        );
    }

    // Example 6: Memory usage optimization
    println!("\n🧠 Example 6: Memory optimization strategies");

    // Process with explicit memory management
    let memory_start = Instant::now();
    let mut streaming_results = Vec::new();

    // Process samples one by one to minimize memory footprint
    for i in 0..20 {
        let frequency = 300.0 + (i as f32 * 8.0);
        let samples: Vec<f32> = (0..duration_samples)
            .map(|j| {
                let t = j as f32 / sample_rate as f32;
                0.2 * (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect();

        let audio = AudioBuffer::new(samples, sample_rate, 1);
        let result = quality_evaluator
            .evaluate_quality(&audio, None, None)
            .await?;

        // Store only essential information to save memory
        streaming_results.push((i, result.overall_score));

        // Explicit cleanup (Rust handles this automatically, but shown for illustration)
        drop(audio);
    }

    let memory_time = memory_start.elapsed();
    println!("  Streaming processing:");
    println!(
        "    Processed 20 samples individually in {:.2}s",
        memory_time.as_secs_f32()
    );
    println!("    Memory-efficient approach suitable for large datasets");

    // Example 7: Comprehensive benchmark
    println!("\n🏁 Example 7: Comprehensive benchmark");

    let benchmark_sizes = vec![10, 25, 50, 100];

    for &size in &benchmark_sizes {
        let benchmark_samples: Vec<(AudioBuffer, Option<AudioBuffer>)> = (0..size)
            .map(|i| {
                let samples: Vec<f32> = (0..duration_samples)
                    .map(|j| {
                        let t = j as f32 / sample_rate as f32;
                        0.25 * (2.0 * std::f32::consts::PI * (250.0 + i as f32 * 3.0) * t).sin()
                    })
                    .collect();
                (AudioBuffer::new(samples, sample_rate, 1), None)
            })
            .collect();

        let benchmark_start = Instant::now();
        let benchmark_results = quality_evaluator
            .evaluate_quality_batch(&benchmark_samples, None)
            .await?;
        let benchmark_time = benchmark_start.elapsed();

        let throughput = size as f32 / benchmark_time.as_secs_f32();
        let avg_per_sample = benchmark_time.as_millis() as f32 / size as f32;

        println!(
            "  Batch size {}: {:.2}s ({:.1} samples/s, {:.1}ms/sample)",
            size,
            benchmark_time.as_secs_f32(),
            throughput,
            avg_per_sample
        );

        // Verify all results are valid
        assert_eq!(benchmark_results.len(), size);
        assert!(benchmark_results.iter().all(|r| r.overall_score > 0.0));
    }

    // Example 8: Resource cleanup and optimization tips
    println!("\n🧹 Example 8: Resource management best practices");

    // Clear caches
    cache.clear();

    // Clear GPU memory if available
    if let Some(ref gpu) = gpu_accelerator {
        gpu.clear_memory_pool();
        println!("  ✓ GPU memory pool cleared");
    }

    // Clear performance monitor
    performance_monitor.clear();
    println!("  ✓ Performance monitor cleared");

    println!("  💡 Optimization Tips:");
    println!("    • Use batch processing for multiple samples");
    println!("    • Implement caching for repeated evaluations");
    println!("    • Process in chunks for large datasets");
    println!("    • Use GPU acceleration when available");
    println!("    • Monitor memory usage for streaming applications");
    println!("    • Profile your specific use case for optimal parameters");

    println!("\n✅ Batch processing and performance examples completed successfully!");
    Ok(())
}
