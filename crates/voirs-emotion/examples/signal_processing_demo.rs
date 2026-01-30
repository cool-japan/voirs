//! Comprehensive demonstration of the integrated signal processing module
//!
//! This example showcases the unified signal processing pipeline that combines
//! formant manipulation, spectral processing, and breath control for emotion-aware
//! audio synthesis.
//!
//! Run with: `cargo run --example signal_processing_demo`

use voirs_emotion::{
    signal_processing::{ProcessingQuality, SignalProcessingConfig, SignalProcessor},
    types::{Emotion, EmotionIntensity, EmotionVector},
    Result,
};

fn main() -> Result<()> {
    println!("🎵 VoiRS Emotion Signal Processing Demo\n");
    println!("========================================\n");

    // Demo 1: Basic emotion processing with different quality settings
    demo_quality_presets()?;

    // Demo 2: Emotion analysis from audio
    demo_emotion_analysis()?;

    // Demo 3: Text-aware processing with natural pauses
    demo_text_processing()?;

    // Demo 4: Adaptive intensity processing
    demo_adaptive_intensity()?;

    // Demo 5: Custom configuration
    demo_custom_config()?;

    // Demo 6: Performance comparison
    demo_performance_comparison()?;

    println!("\n✅ All demos completed successfully!");

    Ok(())
}

/// Demonstrate different quality presets
fn demo_quality_presets() -> Result<()> {
    println!("📊 Demo 1: Quality Presets Comparison");
    println!("─────────────────────────────────────\n");

    let audio = generate_test_audio(44100, 440.0); // 1 second of 440Hz tone

    let qualities = vec![
        (ProcessingQuality::Low, "Low (Speed-optimized)"),
        (ProcessingQuality::Medium, "Medium (Balanced)"),
        (ProcessingQuality::High, "High (Quality-optimized)"),
        (ProcessingQuality::Ultra, "Ultra (Maximum quality)"),
    ];

    for (quality, description) in qualities {
        let config = SignalProcessingConfig::preset(quality);
        let mut processor = SignalProcessor::new(config.clone(), 44100.0);

        println!("  {} Quality:", description);
        println!("    FFT Size: {}", config.fft_size);
        println!("    Overlap:  {:.1}%", config.overlap_factor * 100.0);

        let start = std::time::Instant::now();
        let result = processor.process_with_emotion(&audio, &Emotion::Happy, 0.8)?;
        let elapsed = start.elapsed();

        println!("    Output:   {} samples", result.len());
        println!("    Time:     {:.2}ms", elapsed.as_micros() as f64 / 1000.0);
        println!();
    }

    Ok(())
}

/// Demonstrate emotion analysis from audio
fn demo_emotion_analysis() -> Result<()> {
    println!("🎭 Demo 2: Emotion Analysis from Audio");
    println!("───────────────────────────────────────\n");

    let config = SignalProcessingConfig::full();
    let processor = SignalProcessor::new(config, 44100.0);

    // Create different audio signals representing different emotions
    let test_cases = vec![
        ("High energy", generate_energetic_audio(44100)),
        ("Low energy", generate_calm_audio(44100)),
        ("Mixed frequency", generate_complex_audio(44100)),
    ];

    for (description, audio) in test_cases {
        let (detected_emotion, confidence) = processor.analyze_emotion(&audio)?;

        println!("  {}:", description);
        println!("    Detected:   {:?}", detected_emotion);
        println!("    Confidence: {:.1}%", confidence * 100.0);
        println!();
    }

    Ok(())
}

/// Demonstrate text-aware processing with natural pauses
fn demo_text_processing() -> Result<()> {
    println!("📝 Demo 3: Text-Aware Processing");
    println!("─────────────────────────────────\n");

    let config = SignalProcessingConfig::default();
    let mut processor = SignalProcessor::new(config, 44100.0);

    let text = "Hello, world. How are you today? I hope you're doing well!";
    let audio = generate_test_audio(88200, 440.0); // 2 seconds

    println!("  Input text:");
    println!("    \"{}\"", text);
    println!("\n  Processing with natural pauses...");

    let start = std::time::Instant::now();
    let result = processor.process_text_with_emotion(text, &audio, &Emotion::Calm, 0.7)?;
    let elapsed = start.elapsed();

    println!("\n  Results:");
    println!(
        "    Original length:  {} samples ({:.2}s)",
        audio.len(),
        audio.len() as f32 / 44100.0
    );
    println!(
        "    Processed length: {} samples ({:.2}s)",
        result.len(),
        result.len() as f32 / 44100.0
    );
    println!(
        "    Added pauses:     {} samples ({:.2}s)",
        result.len().saturating_sub(audio.len()),
        (result.len().saturating_sub(audio.len())) as f32 / 44100.0
    );
    println!(
        "    Processing time:  {:.2}ms",
        elapsed.as_micros() as f64 / 1000.0
    );
    println!();

    Ok(())
}

/// Demonstrate adaptive intensity processing
fn demo_adaptive_intensity() -> Result<()> {
    println!("⚡ Demo 4: Adaptive Intensity Processing");
    println!("────────────────────────────────────────\n");

    let mut config = SignalProcessingConfig::default();
    config.adaptive_intensity = true;

    let mut processor = SignalProcessor::new(config, 44100.0);

    let audio = generate_test_audio(44100, 440.0);

    let emotions = vec![
        Emotion::Excited,
        Emotion::Happy,
        Emotion::Neutral,
        Emotion::Sad,
        Emotion::Calm,
    ];

    let base_intensity = 0.8;

    println!("  Processing with adaptive intensity enabled");
    println!("  Base intensity: {:.1}\n", base_intensity);

    for emotion in emotions {
        let result = processor.process_with_emotion(&audio, &emotion, base_intensity)?;

        println!("  {:?}:", emotion);
        println!("    Output samples: {}", result.len());
        println!("    Adaptive mode:  ✓ enabled");
        println!();
    }

    Ok(())
}

/// Demonstrate custom configuration
fn demo_custom_config() -> Result<()> {
    println!("🔧 Demo 5: Custom Configuration");
    println!("────────────────────────────────\n");

    // Create a custom configuration for real-time processing
    let custom_config = SignalProcessingConfig {
        enable_formant: false, // Disable formant for speed
        enable_spectral: true, // Keep spectral processing
        enable_breath: false,  // Disable breath for speed
        quality: ProcessingQuality::Low,
        fft_size: 1024,
        overlap_factor: 0.25,
        adaptive_intensity: true,
    };

    let mut processor = SignalProcessor::new(custom_config.clone(), 44100.0);

    println!("  Configuration:");
    println!(
        "    Formant:     {}",
        if custom_config.enable_formant {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "    Spectral:    {}",
        if custom_config.enable_spectral {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "    Breath:      {}",
        if custom_config.enable_breath {
            "✓"
        } else {
            "✗"
        }
    );
    println!("    FFT Size:    {}", custom_config.fft_size);
    println!(
        "    Adaptive:    {}",
        if custom_config.adaptive_intensity {
            "✓"
        } else {
            "✗"
        }
    );
    println!();

    let audio = generate_test_audio(44100, 440.0);

    let start = std::time::Instant::now();
    let result = processor.process_with_emotion(&audio, &Emotion::Happy, 0.8)?;
    let elapsed = start.elapsed();

    println!("  Performance:");
    println!(
        "    Processing time: {:.2}ms",
        elapsed.as_micros() as f64 / 1000.0
    );
    println!("    Output samples:  {}", result.len());
    println!(
        "    Real-time factor: {:.3}×",
        (audio.len() as f64 / 44100.0) / (elapsed.as_secs_f64())
    );
    println!();

    Ok(())
}

/// Demonstrate performance comparison across emotions
fn demo_performance_comparison() -> Result<()> {
    println!("⏱️  Demo 6: Performance Comparison");
    println!("──────────────────────────────────\n");

    let config = SignalProcessingConfig::preset(ProcessingQuality::Medium);
    let mut processor = SignalProcessor::new(config, 44100.0);

    let audio = generate_test_audio(88200, 440.0); // 2 seconds

    let emotions = vec![
        Emotion::Happy,
        Emotion::Sad,
        Emotion::Angry,
        Emotion::Calm,
        Emotion::Excited,
    ];

    println!(
        "  Processing {} samples with different emotions:\n",
        audio.len()
    );

    let mut total_time = std::time::Duration::ZERO;

    for emotion in emotions {
        let start = std::time::Instant::now();
        let _result = processor.process_with_emotion(&audio, &emotion, 0.8)?;
        let elapsed = start.elapsed();

        total_time += elapsed;

        let rtf = (audio.len() as f64 / 44100.0) / elapsed.as_secs_f64();

        println!("  {:?}:", emotion);
        println!(
            "    Time: {:.2}ms  |  RTF: {:.2}×",
            elapsed.as_micros() as f64 / 1000.0,
            rtf
        );
    }

    let avg_time = total_time / 5;
    println!(
        "\n  Average processing time: {:.2}ms",
        avg_time.as_micros() as f64 / 1000.0
    );
    println!();

    Ok(())
}

// Helper functions for generating test audio

fn generate_test_audio(samples: usize, frequency: f32) -> Vec<f32> {
    let sample_rate = 44100.0;
    (0..samples)
        .map(|i| (i as f32 * frequency * 2.0 * std::f32::consts::PI / sample_rate).sin() * 0.5)
        .collect()
}

fn generate_energetic_audio(samples: usize) -> Vec<f32> {
    let sample_rate = 44100.0;
    (0..samples)
        .map(|i| {
            // High frequency, high amplitude
            ((i as f32 * 1000.0 * 2.0 * std::f32::consts::PI / sample_rate).sin() * 0.8)
                + ((i as f32 * 1500.0 * 2.0 * std::f32::consts::PI / sample_rate).sin() * 0.6)
        })
        .map(|x| x * 0.5)
        .collect()
}

fn generate_calm_audio(samples: usize) -> Vec<f32> {
    let sample_rate = 44100.0;
    (0..samples)
        .map(|i| {
            // Low frequency, low amplitude
            ((i as f32 * 200.0 * 2.0 * std::f32::consts::PI / sample_rate).sin() * 0.3)
                + ((i as f32 * 300.0 * 2.0 * std::f32::consts::PI / sample_rate).sin() * 0.2)
        })
        .collect()
}

fn generate_complex_audio(samples: usize) -> Vec<f32> {
    let sample_rate = 44100.0;
    (0..samples)
        .map(|i| {
            // Mixed frequency content
            ((i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate).sin() * 0.4)
                + ((i as f32 * 880.0 * 2.0 * std::f32::consts::PI / sample_rate).sin() * 0.3)
                + ((i as f32 * 1320.0 * 2.0 * std::f32::consts::PI / sample_rate).sin() * 0.2)
        })
        .collect()
}
