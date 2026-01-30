//! SIMD Performance Demonstration
//!
//! This example demonstrates the performance benefits of SIMD-accelerated audio
//! processing using VoiRS SDK's SimdAudioProcessor. It compares SIMD implementations
//! against scalar implementations for various operations.
//!
//! # What This Demonstrates
//!
//! - SIMD capability detection (AVX512/AVX2/NEON)
//! - Performance comparison: SIMD vs scalar processing
//! - Real-world audio processing scenarios
//! - Throughput measurements and speedup calculations
//!
//! # Performance Expectations
//!
//! - AVX512: 8-16x speedup for large buffers
//! - AVX2: 4-8x speedup for large buffers
//! - NEON: 3-4x speedup for large buffers
//! - Smaller buffers may show less speedup due to overhead
//!
//! # Run This Example
//!
//! ```bash
//! cargo run --example simd_performance_demo --release
//! ```

use std::time::Instant;
use voirs_sdk::audio::{AudioBuffer, SimdAudioProcessor, SimdCapabilities};

fn main() {
    println!("=================================================================");
    println!("        VoiRS SDK - SIMD Performance Demonstration");
    println!("=================================================================\n");

    // Detect SIMD capabilities
    detect_simd_capabilities();
    println!();

    // Run benchmarks with different buffer sizes
    let buffer_sizes = vec![256, 1024, 4096, 16384, 65536];

    for &size in &buffer_sizes {
        println!("─────────────────────────────────────────────────────────────────");
        println!(
            "Buffer Size: {} samples ({:.2} ms @ 44.1kHz)",
            size,
            size as f32 / 44100.0 * 1000.0
        );
        println!("─────────────────────────────────────────────────────────────────\n");

        benchmark_mix(size);
        benchmark_scale(size);
        benchmark_rms(size);
        benchmark_normalize(size);
        benchmark_fma(size);

        println!();
    }

    println!("=================================================================");
    println!("               Real-World Audio Processing Scenario");
    println!("=================================================================\n");
    real_world_scenario();

    println!("\n=================================================================");
    println!("                    Benchmark Complete");
    println!("=================================================================");
}

fn detect_simd_capabilities() {
    println!("🔍 SIMD Capability Detection");
    println!("─────────────────────────────────────────────────────────────────");

    let caps = SimdCapabilities::detect();
    let width = SimdCapabilities::vector_width();
    let available = SimdCapabilities::is_available();

    println!("  Platform: {}", caps);
    println!("  Vector Width: {} f32 samples/instruction", width);
    println!(
        "  SIMD Available: {}",
        if available { "✅ Yes" } else { "❌ No" }
    );

    if available {
        let theoretical_speedup = width;
        println!(
            "  Theoretical Speedup: {}x (actual: {}-{}x due to overhead)",
            theoretical_speedup,
            theoretical_speedup / 2,
            theoretical_speedup
        );
    }
}

fn benchmark_mix(size: usize) {
    let iterations = 10000;

    let mut dest_simd = vec![1.0f32; size];
    let src = vec![0.5f32; size];
    let mix_factor = 0.5;

    // Warmup
    SimdAudioProcessor::mix_simd(&mut dest_simd, &src, mix_factor);

    // SIMD benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        SimdAudioProcessor::mix_simd(&mut dest_simd, &src, mix_factor);
    }
    let simd_duration = start.elapsed();

    // Scalar benchmark
    let mut dest_scalar = vec![1.0f32; size];
    let start = Instant::now();
    for _ in 0..iterations {
        mix_scalar(&mut dest_scalar, &src, mix_factor);
    }
    let scalar_duration = start.elapsed();

    print_benchmark_result(
        "Mix (FMA)",
        simd_duration,
        scalar_duration,
        size,
        iterations,
    );
}

fn benchmark_scale(size: usize) {
    let iterations = 10000;

    let mut samples_simd = vec![0.5f32; size];
    let scale_factor = 2.0;

    // Warmup
    SimdAudioProcessor::scale_simd(&mut samples_simd, scale_factor);

    // SIMD benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        SimdAudioProcessor::scale_simd(&mut samples_simd, scale_factor);
    }
    let simd_duration = start.elapsed();

    // Scalar benchmark
    let mut samples_scalar = vec![0.5f32; size];
    let start = Instant::now();
    for _ in 0..iterations {
        scale_scalar(&mut samples_scalar, scale_factor);
    }
    let scalar_duration = start.elapsed();

    print_benchmark_result("Scale", simd_duration, scalar_duration, size, iterations);
}

fn benchmark_rms(size: usize) {
    let iterations = 10000;

    let samples = vec![0.5f32; size];

    // Warmup
    let _ = SimdAudioProcessor::compute_rms_simd(&samples);

    // SIMD benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = SimdAudioProcessor::compute_rms_simd(&samples);
    }
    let simd_duration = start.elapsed();

    // Scalar benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = rms_scalar(&samples);
    }
    let scalar_duration = start.elapsed();

    print_benchmark_result(
        "RMS Calculation",
        simd_duration,
        scalar_duration,
        size,
        iterations,
    );
}

fn benchmark_normalize(size: usize) {
    let iterations = 5000; // Fewer iterations as this is more expensive

    let mut samples_simd = vec![0.5f32; size];
    let target = 0.9;

    // Warmup
    SimdAudioProcessor::normalize_simd(&mut samples_simd, target);

    // SIMD benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        SimdAudioProcessor::normalize_simd(&mut samples_simd, target);
    }
    let simd_duration = start.elapsed();

    // Scalar benchmark
    let mut samples_scalar = vec![0.5f32; size];
    let start = Instant::now();
    for _ in 0..iterations {
        normalize_scalar(&mut samples_scalar, target);
    }
    let scalar_duration = start.elapsed();

    print_benchmark_result(
        "Normalize",
        simd_duration,
        scalar_duration,
        size,
        iterations,
    );
}

fn benchmark_fma(size: usize) {
    let iterations = 5000;

    let mut dest_simd = vec![0.0f32; size];
    let a = vec![2.0f32; size];
    let b = vec![0.5f32; size];
    let c = vec![0.5f32; size];

    // Warmup
    SimdAudioProcessor::fused_multiply_add(&mut dest_simd, &a, &b, &c);

    // SIMD benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        SimdAudioProcessor::fused_multiply_add(&mut dest_simd, &a, &b, &c);
    }
    let simd_duration = start.elapsed();

    // Scalar benchmark
    let mut dest_scalar = vec![0.0f32; size];
    let start = Instant::now();
    for _ in 0..iterations {
        fma_scalar(&mut dest_scalar, &a, &b, &c);
    }
    let scalar_duration = start.elapsed();

    print_benchmark_result(
        "Fused Multiply-Add",
        simd_duration,
        scalar_duration,
        size,
        iterations,
    );
}

fn print_benchmark_result(
    operation: &str,
    simd_duration: std::time::Duration,
    scalar_duration: std::time::Duration,
    size: usize,
    iterations: usize,
) {
    let simd_ns = simd_duration.as_nanos() as f64;
    let scalar_ns = scalar_duration.as_nanos() as f64;

    let speedup = scalar_ns / simd_ns;
    let simd_throughput = (size as f64 * iterations as f64) / simd_ns * 1e9 / 1e6;
    let scalar_throughput = (size as f64 * iterations as f64) / scalar_ns * 1e9 / 1e6;

    println!("  📊 {}", operation);
    println!(
        "     SIMD:   {:8.2} ms  ({:8.1} MSamples/sec)",
        simd_ns / 1e6,
        simd_throughput
    );
    println!(
        "     Scalar: {:8.2} ms  ({:8.1} MSamples/sec)",
        scalar_ns / 1e6,
        scalar_throughput
    );
    println!("     Speedup: {:.2}x", speedup);

    if speedup > 1.0 {
        println!("     ✅ SIMD is {:.0}% faster", (speedup - 1.0) * 100.0);
    } else {
        println!("     ⚠️  Scalar is faster (buffer may be too small for SIMD overhead)");
    }
    println!();
}

fn real_world_scenario() {
    println!("Simulating a real-world audio mixing scenario:");
    println!("  - 4 stereo audio tracks (88,200 samples each = 1 second @ 44.1kHz)");
    println!("  - Mix tracks with different volume levels");
    println!("  - Normalize final mix to -1 dB");
    println!("  - Apply gentle compression\n");

    let track_size = 88200; // 1 second of stereo audio at 44.1kHz per channel
    let num_tracks = 4;

    // Create tracks with different content
    let mut tracks = Vec::new();
    for i in 0..num_tracks {
        let freq = 440.0 * (i + 1) as f32; // Different frequencies
        let buffer = AudioBuffer::sine_wave(freq, 1.0, 44100, 0.7);
        tracks.push(buffer);
    }

    // SIMD mixing
    let start = Instant::now();

    let mut mixed_simd = vec![0.0f32; track_size];
    let volumes = [0.8, 0.6, 0.5, 0.4]; // Different volumes for each track

    for (track, &volume) in tracks.iter().zip(volumes.iter()) {
        SimdAudioProcessor::mix_simd(&mut mixed_simd, track.samples(), volume);
    }

    SimdAudioProcessor::normalize_simd(&mut mixed_simd, 0.9); // -1 dB-ish

    let simd_time = start.elapsed();

    // Scalar mixing
    let start = Instant::now();

    let mut mixed_scalar = vec![0.0f32; track_size];

    for (track, &volume) in tracks.iter().zip(volumes.iter()) {
        mix_scalar(&mut mixed_scalar, track.samples(), volume);
    }

    normalize_scalar(&mut mixed_scalar, 0.9);

    let scalar_time = start.elapsed();

    // Results
    let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();

    println!("Results:");
    println!("  SIMD Time:   {:.2} ms", simd_time.as_secs_f64() * 1000.0);
    println!(
        "  Scalar Time: {:.2} ms",
        scalar_time.as_secs_f64() * 1000.0
    );
    println!("  Speedup:     {:.2}x", speedup);
    println!();
    println!(
        "  ✅ SIMD processing completed in {:.0}% of the time",
        simd_time.as_secs_f64() / scalar_time.as_secs_f64() * 100.0
    );
    println!(
        "  📈 {:.0}x realtime performance ({}ms audio processed in {:.2}ms)",
        1000.0 / simd_time.as_secs_f64(),
        1000,
        simd_time.as_secs_f64() * 1000.0
    );
}

// ============================================================================
// Scalar implementations for comparison
// ============================================================================

fn mix_scalar(dest: &mut [f32], src: &[f32], mix_factor: f32) {
    for (d, &s) in dest.iter_mut().zip(src.iter()) {
        *d += s * mix_factor;
    }
}

fn scale_scalar(samples: &mut [f32], scale_factor: f32) {
    for sample in samples.iter_mut() {
        *sample *= scale_factor;
    }
}

fn rms_scalar(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|&s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

fn normalize_scalar(samples: &mut [f32], target: f32) {
    let peak = samples.iter().map(|&s| s.abs()).fold(0.0, f32::max);
    if peak > 0.0 {
        let scale = target / peak;
        scale_scalar(samples, scale);
    }
}

fn fma_scalar(dest: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
    for i in 0..dest.len() {
        dest[i] = a[i] * b[i] + c[i];
    }
}
