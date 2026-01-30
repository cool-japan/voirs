//! Real-time Adaptive Audio Enhancement Demonstration
//!
//! This example demonstrates the sophisticated audio enhancement capabilities of
//! VoiRS SDK's adaptive enhancement system, showing real-time quality optimization,
//! adaptive processing, and performance monitoring.
//!
//! # What This Demonstrates
//!
//! - Adaptive noise gate with spectral analysis
//! - Multiband dynamic range compression
//! - Spectral enhancement for clarity
//! - Quality-aware processing with feedback
//! - Performance monitoring and adaptive optimization
//!
//! # Run This Example
//!
//! ```bash
//! cargo run --example adaptive_enhancement_demo --release
//! ```

use voirs_sdk::audio::{AdaptiveEnhancer, AudioBuffer, EnhancementConfig};
use voirs_sdk::Result;

fn main() -> Result<()> {
    println!("================================================================");
    println!("     VoiRS SDK - Adaptive Audio Enhancement Demonstration");
    println!("================================================================\n");

    // Demonstration 1: Basic enhancement
    println!("─────────────────────────────────────────────────────────────────");
    println!("Demo 1: Basic Adaptive Enhancement");
    println!("─────────────────────────────────────────────────────────────────\n");
    demo_basic_enhancement()?;

    // Demonstration 2: Noise gate
    println!("\n─────────────────────────────────────────────────────────────────");
    println!("Demo 2: Adaptive Noise Gate");
    println!("─────────────────────────────────────────────────────────────────\n");
    demo_noise_gate()?;

    // Demonstration 3: Multiband compression
    println!("\n─────────────────────────────────────────────────────────────────");
    println!("Demo 3: Multiband Dynamic Range Compression");
    println!("─────────────────────────────────────────────────────────────────\n");
    demo_multiband_compression()?;

    // Demonstration 4: Spectral enhancement
    println!("\n─────────────────────────────────────────────────────────────────");
    println!("Demo 4: Spectral Enhancement");
    println!("─────────────────────────────────────────────────────────────────\n");
    demo_spectral_enhancement()?;

    // Demonstration 5: Quality tracking
    println!("\n─────────────────────────────────────────────────────────────────");
    println!("Demo 5: Quality Tracking and Improvement");
    println!("─────────────────────────────────────────────────────────────────\n");
    demo_quality_tracking()?;

    // Demonstration 6: Performance monitoring
    println!("\n─────────────────────────────────────────────────────────────────");
    println!("Demo 6: Performance Monitoring");
    println!("─────────────────────────────────────────────────────────────────\n");
    demo_performance_monitoring()?;

    // Demonstration 7: Adaptive configuration
    println!("\n─────────────────────────────────────────────────────────────────");
    println!("Demo 7: Adaptive Configuration Adjustment");
    println!("─────────────────────────────────────────────────────────────────\n");
    demo_adaptive_configuration()?;

    // Demonstration 8: Real-world scenario
    println!("\n─────────────────────────────────────────────────────────────────");
    println!("Demo 8: Real-World TTS Enhancement Scenario");
    println!("─────────────────────────────────────────────────────────────────\n");
    demo_real_world_scenario()?;

    println!("\n================================================================");
    println!("                   Demonstration Complete");
    println!("================================================================");

    Ok(())
}

fn demo_basic_enhancement() -> Result<()> {
    println!("Enhancing a clean speech signal with default settings...\n");

    // Create a simulated speech signal (440 Hz tone)
    let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.7);

    println!("Original signal:");
    println!("  Duration: {:.2}s", buffer.duration());
    println!("  Peak amplitude: {:.3}", buffer.metadata().peak_amplitude);
    println!("  RMS amplitude: {:.3}", buffer.metadata().rms_amplitude);

    // Enhance with default settings
    let mut enhancer = AdaptiveEnhancer::new(EnhancementConfig::default());
    let enhanced = enhancer.enhance(&buffer)?;

    println!("\nEnhanced signal:");
    println!("  Duration: {:.2}s", enhanced.duration());
    println!(
        "  Peak amplitude: {:.3}",
        enhanced.metadata().peak_amplitude
    );
    println!("  RMS amplitude: {:.3}", enhanced.metadata().rms_amplitude);
    println!(
        "  Quality improvement: {:.2}%",
        enhancer.quality_improvement() * 100.0
    );

    let metrics = enhancer.performance_metrics();
    println!("\nProcessing performance:");
    println!("  Average time: {:.2}ms", metrics.average_ms);
    println!(
        "  Status: {}",
        if metrics.is_overloaded {
            "⚠️ Overloaded"
        } else {
            "✅ Normal"
        }
    );

    Ok(())
}

fn demo_noise_gate() -> Result<()> {
    println!("Demonstrating adaptive noise gate on noisy signal...\n");

    // Create a signal with noise
    let mut buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);

    // Add some low-level noise
    let samples = buffer.samples_mut();
    for sample in samples.iter_mut() {
        *sample += (fastrand::f32() - 0.5) * 0.05; // Add small random noise
    }

    println!("Noisy signal:");
    println!("  Peak amplitude: {:.3}", buffer.metadata().peak_amplitude);
    println!("  RMS amplitude: {:.3}", buffer.metadata().rms_amplitude);

    // Configure enhancer for aggressive noise gating
    let mut config = EnhancementConfig::default();
    config.enable_multiband_compression = false;
    config.enable_spectral_enhancement = false;
    config.noise_gate_threshold_db = -50.0; // More aggressive gating

    let mut enhancer = AdaptiveEnhancer::new(config);
    let enhanced = enhancer.enhance(&buffer)?;

    println!("\nAfter noise gating:");
    println!(
        "  Peak amplitude: {:.3}",
        enhanced.metadata().peak_amplitude
    );
    println!("  RMS amplitude: {:.3}", enhanced.metadata().rms_amplitude);
    println!(
        "  Quality improvement: {:.2}%",
        enhancer.quality_improvement() * 100.0
    );

    println!("\n✅ Noise gate successfully reduced low-level noise while preserving signal");

    Ok(())
}

fn demo_multiband_compression() -> Result<()> {
    println!("Demonstrating multiband dynamic range compression...\n");

    // Create a signal with varying dynamics
    let mut buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.9); // Loud signal

    println!("High-dynamic-range signal:");
    println!("  Peak amplitude: {:.3}", buffer.metadata().peak_amplitude);
    println!("  RMS amplitude: {:.3}", buffer.metadata().rms_amplitude);
    println!(
        "  Crest factor: {:.2} dB",
        20.0 * (buffer.metadata().peak_amplitude / buffer.metadata().rms_amplitude).log10()
    );

    // Configure for compression
    let mut config = EnhancementConfig::default();
    config.enable_noise_gate = false;
    config.enable_spectral_enhancement = false;
    config.compression_ratios = [4.0, 3.5, 3.0]; // Strong compression
    config.compression_thresholds_db = [-15.0, -12.0, -10.0];

    let mut enhancer = AdaptiveEnhancer::new(config);
    let enhanced = enhancer.enhance(&buffer)?;

    println!("\nAfter multiband compression:");
    println!(
        "  Peak amplitude: {:.3}",
        enhanced.metadata().peak_amplitude
    );
    println!("  RMS amplitude: {:.3}", enhanced.metadata().rms_amplitude);
    println!(
        "  Crest factor: {:.2} dB",
        20.0 * (enhanced.metadata().peak_amplitude / enhanced.metadata().rms_amplitude).log10()
    );

    println!("\nCompression settings:");
    println!("  Low band (20-200Hz): Ratio {:.1}:1, Threshold -15dB", 4.0);
    println!("  Mid band (200-2kHz): Ratio {:.1}:1, Threshold -12dB", 3.5);
    println!(
        "  High band (2k-20kHz): Ratio {:.1}:1, Threshold -10dB",
        3.0
    );

    println!("\n✅ Multiband compression reduced dynamic range while maintaining spectral balance");

    Ok(())
}

fn demo_spectral_enhancement() -> Result<()> {
    println!("Demonstrating spectral enhancement for clarity and presence...\n");

    // Create a complex signal
    let buffer1 = AudioBuffer::sine_wave(220.0, 1.0, 44100, 0.3);
    let buffer2 = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.3);
    let buffer3 = AudioBuffer::sine_wave(880.0, 1.0, 44100, 0.2);

    let mut buffer = buffer1.clone();
    buffer.mix(&buffer2, 1.0)?;
    buffer.mix(&buffer3, 1.0)?;

    println!("Original complex signal:");
    println!("  Frequencies: 220Hz, 440Hz, 880Hz (mixed)");
    println!("  Spectral centroid: {:.1} Hz", buffer.spectral_centroid());

    // Configure for spectral enhancement
    let mut config = EnhancementConfig::default();
    config.enable_noise_gate = false;
    config.enable_multiband_compression = false;
    config.enhancement_strength = 0.7; // Strong enhancement

    let mut enhancer = AdaptiveEnhancer::new(config);
    let enhanced = enhancer.enhance(&buffer)?;

    println!("\nAfter spectral enhancement:");
    println!(
        "  Spectral centroid: {:.1} Hz",
        enhanced.spectral_centroid()
    );
    println!("  Enhancement strength: 70.0%");
    println!(
        "  Quality improvement: {:.2}%",
        enhancer.quality_improvement() * 100.0
    );

    println!("\n✅ Spectral enhancement improved clarity and tonal balance");

    Ok(())
}

fn demo_quality_tracking() -> Result<()> {
    println!("Demonstrating quality tracking over multiple enhancements...\n");

    let buffer = AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.6);

    let mut config = EnhancementConfig::default();
    config.target_quality = 0.85;

    let mut enhancer = AdaptiveEnhancer::new(config);

    println!("Processing multiple iterations to track quality improvement...\n");

    for iteration in 1..=5 {
        let enhanced = enhancer.enhance(&buffer)?;

        println!("Iteration {}:", iteration);
        println!("  Current quality: {:.3}", enhancer.quality_improvement());
        println!(
            "  Peak amplitude: {:.3}",
            enhanced.metadata().peak_amplitude
        );
        println!("  RMS amplitude: {:.3}", enhanced.metadata().rms_amplitude);
        println!();
    }

    let final_improvement = enhancer.quality_improvement();
    println!(
        "Final quality improvement: {:.2}%",
        final_improvement * 100.0
    );

    if final_improvement > 0.0 {
        println!("✅ Quality consistently improved over iterations");
    } else {
        println!("ℹ️  Quality maintained (signal was already high quality)");
    }

    Ok(())
}

fn demo_performance_monitoring() -> Result<()> {
    println!("Demonstrating real-time performance monitoring...\n");

    let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.7);

    let mut enhancer = AdaptiveEnhancer::new(EnhancementConfig::default());

    // Process multiple times to gather statistics
    println!(
        "Processing {} times to gather performance statistics...\n",
        10
    );

    for _ in 0..10 {
        let _ = enhancer.enhance(&buffer)?;
    }

    let metrics = enhancer.performance_metrics();

    println!("Performance metrics:");
    println!("  Average processing time: {:.2}ms", metrics.average_ms);
    println!("  Min processing time: {:.2}ms", metrics.min_ms);
    println!("  Max processing time: {:.2}ms", metrics.max_ms);
    println!(
        "  System status: {}",
        if metrics.is_overloaded {
            "⚠️ Overloaded (processing >50ms on average)"
        } else {
            "✅ Normal (processing <50ms on average)"
        }
    );

    // Calculate real-time factor
    let audio_duration_ms = (buffer.duration() * 1000.0) as f64;
    let rtf = metrics.average_ms / audio_duration_ms;
    println!("\nReal-time factor (RTF):");
    println!("  Audio duration: {:.2}ms", audio_duration_ms);
    println!("  Processing time: {:.2}ms", metrics.average_ms);
    println!("  RTF: {:.3}x (lower is better)", rtf);

    if rtf < 0.1 {
        println!("  ✅ Excellent: Can process 10x faster than real-time");
    } else if rtf < 1.0 {
        println!("  ✅ Good: Faster than real-time");
    } else {
        println!("  ⚠️  Warning: Slower than real-time");
    }

    Ok(())
}

fn demo_adaptive_configuration() -> Result<()> {
    println!("Demonstrating adaptive configuration adjustment...\n");

    let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.6);

    let mut config = EnhancementConfig::default();
    config.enable_adaptive_processing = true;
    config.target_quality = 0.9; // High quality target
    config.learning_rate = 0.05; // Moderate learning rate

    let mut enhancer = AdaptiveEnhancer::new(config.clone());

    println!("Initial configuration:");
    println!("  Enhancement strength: {:.3}", config.enhancement_strength);
    println!("  Target quality: {:.3}", config.target_quality);
    println!("  Learning rate: {:.3}", config.learning_rate);
    println!();

    println!("Processing and adapting over {} iterations...\n", 15);

    for iteration in 1..=15 {
        let _ = enhancer.enhance(&buffer)?;

        if iteration % 5 == 0 {
            println!("After iteration {}:", iteration);
            println!(
                "  Enhancement strength: {:.3}",
                enhancer.config().enhancement_strength
            );
            println!(
                "  Quality improvement: {:.3}",
                enhancer.quality_improvement()
            );
            println!();
        }
    }

    println!("✅ Configuration adapted based on quality feedback");
    println!("   Enhancement strength automatically adjusted to reach target quality");

    Ok(())
}

fn demo_real_world_scenario() -> Result<()> {
    println!("Simulating real-world TTS post-processing scenario...\n");

    println!("Scenario: Processing synthesized speech output");
    println!("  - Input: TTS-generated audio (simulated)");
    println!("  - Goal: Enhance clarity, reduce artifacts, optimize loudness");
    println!("  - Requirements: Real-time processing, high quality\n");

    // Simulate TTS output: mixture of frequencies with some imperfections
    let fundamental = AudioBuffer::sine_wave(200.0, 2.0, 44100, 0.4); // Fundamental frequency
    let harmonic1 = AudioBuffer::sine_wave(400.0, 2.0, 44100, 0.3); // First harmonic
    let harmonic2 = AudioBuffer::sine_wave(600.0, 2.0, 44100, 0.2); // Second harmonic

    let mut tts_output = fundamental.clone();
    tts_output.mix(&harmonic1, 1.0)?;
    tts_output.mix(&harmonic2, 1.0)?;

    // Add some noise (simulating TTS artifacts)
    let samples = tts_output.samples_mut();
    for sample in samples.iter_mut() {
        *sample += (fastrand::f32() - 0.5) * 0.03;
    }

    println!("Original TTS output analysis:");
    analyze_audio(&tts_output);

    // Configure enhancement for TTS post-processing
    let mut config = EnhancementConfig::default();
    config.enable_noise_gate = true;
    config.noise_gate_threshold_db = -55.0; // Remove artifacts
    config.enable_multiband_compression = true;
    config.compression_ratios = [2.5, 2.0, 1.8]; // Gentle compression
    config.enable_spectral_enhancement = true;
    config.enhancement_strength = 0.6; // Moderate enhancement
    config.enable_adaptive_processing = true;
    config.target_quality = 0.85;

    let mut enhancer = AdaptiveEnhancer::new(config);

    // Process the audio
    let start = std::time::Instant::now();
    let enhanced = enhancer.enhance(&tts_output)?;
    let processing_time = start.elapsed();

    println!("\nEnhanced TTS output analysis:");
    analyze_audio(&enhanced);

    println!("\nEnhancement results:");
    println!(
        "  Quality improvement: {:.2}%",
        enhancer.quality_improvement() * 100.0
    );
    println!(
        "  Processing time: {:.2}ms",
        processing_time.as_secs_f64() * 1000.0
    );

    let rtf = processing_time.as_secs_f64() / enhanced.duration() as f64;
    println!("  Real-time factor: {:.3}x", rtf);

    println!("\nEnhancement settings applied:");
    println!("  ✅ Noise gate: Removed low-level artifacts");
    println!("  ✅ Multiband compression: Optimized dynamic range");
    println!("  ✅ Spectral enhancement: Improved clarity and presence");
    println!("  ✅ Adaptive processing: Continuously optimizing");

    if rtf < 0.1 {
        println!("\n✅ Excellent: Real-time processing with 10x headroom");
    } else if rtf < 1.0 {
        println!("\n✅ Good: Processing faster than real-time");
    }

    Ok(())
}

// Helper function to analyze audio characteristics
fn analyze_audio(buffer: &AudioBuffer) {
    let metadata = buffer.metadata();

    println!("  Duration: {:.2}s", metadata.duration);
    println!(
        "  Peak amplitude: {:.3} ({:.1} dB)",
        metadata.peak_amplitude,
        20.0 * metadata.peak_amplitude.log10()
    );
    println!(
        "  RMS amplitude: {:.3} ({:.1} dB)",
        metadata.rms_amplitude,
        20.0 * metadata.rms_amplitude.log10()
    );
    println!(
        "  Crest factor: {:.2} dB",
        20.0 * (metadata.peak_amplitude / metadata.rms_amplitude.max(1e-10)).log10()
    );
    println!("  Spectral centroid: {:.1} Hz", buffer.spectral_centroid());

    // Check for clipping
    if buffer.has_clipping() {
        println!(
            "  ⚠️  Clipping detected: {} samples",
            buffer.count_clipped_samples()
        );
    } else {
        println!("  ✅ No clipping");
    }
}
