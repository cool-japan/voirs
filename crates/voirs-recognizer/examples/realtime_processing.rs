//! Real-time Audio Processing Example
//!
//! This example demonstrates real-time audio processing capabilities including
//! streaming recognition, voice activity detection, and low-latency processing.
//!
//! Usage:
//! ```bash
//! cargo run --example realtime_processing --features="whisper-pure"
//! ```

use std::time::Duration;
use tokio::time::sleep;
use voirs_recognizer::prelude::*;
use voirs_recognizer::{AudioPreprocessingConfig, AudioPreprocessor, RecognitionError};

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    println!("🚀 VoiRS Real-time Audio Processing Example");
    println!("===========================================\n");

    // Step 1: Set up real-time preprocessing
    println!("🔧 Setting up real-time audio preprocessing...");

    let preprocessing_config = AudioPreprocessingConfig {
        noise_suppression: true,
        agc: true,
        echo_cancellation: true,
        bandwidth_extension: true,
        ..Default::default()
    };

    let preprocessor = AudioPreprocessor::new(preprocessing_config.clone())?;
    println!("✅ Audio preprocessor initialized");
    println!("   • Noise suppression: enabled");
    println!("   • Automatic gain control: enabled");
    println!("   • Echo cancellation: enabled");
    println!("   • Bandwidth extension: enabled");
    println!("   • Real-time feature extraction: enabled");

    // Step 2: Create streaming audio chunks to simulate real-time input
    println!("\n📊 Simulating real-time audio stream...");
    let sample_rate = 16000;
    let chunk_duration_ms = 100; // 100ms chunks for low latency
    let chunk_size = (sample_rate as f32 * chunk_duration_ms as f32 / 1000.0) as usize;

    println!(
        "   • Chunk size: {} samples ({} ms)",
        chunk_size, chunk_duration_ms
    );
    println!("   • Sample rate: {} Hz", sample_rate);

    // Create a series of audio chunks with different characteristics
    let chunks = create_test_audio_chunks(sample_rate, chunk_size, 10);
    println!(
        "   • Generated {} audio chunks for processing",
        chunks.len()
    );

    // Step 3: Set up Voice Activity Detection
    println!("\n🗣️ Setting up Voice Activity Detection...");
    let analyzer_config = AudioAnalysisConfig {
        quality_metrics: true,
        prosody_analysis: false, // Disable for real-time to reduce latency
        speaker_analysis: false, // Disable for real-time to reduce latency
        ..Default::default()
    };

    let analyzer = AudioAnalyzerImpl::new(analyzer_config.clone()).await?;
    println!("✅ VAD analyzer initialized for real-time processing");

    // Step 4: Process chunks in real-time simulation
    println!("\n⚡ Starting real-time processing simulation...");

    let mut total_speech_duration = 0.0;
    let mut total_chunks_processed = 0;
    let mut chunks_with_speech = 0;

    for (i, chunk) in chunks.iter().enumerate() {
        println!("\n📋 Processing chunk {} of {}...", i + 1, chunks.len());

        // Simulate real-time timing
        let start_time = std::time::Instant::now();

        // Perform VAD on this chunk
        let analysis = analyzer.analyze(chunk, Some(&analyzer_config)).await?;

        let processing_time = start_time.elapsed();
        let real_time_factor = processing_time.as_secs_f64() / (chunk_duration_ms as f64 / 1000.0);

        // Check for voice activity
        // Check for voice activity using quality metrics
        if let Some(energy) = analysis.quality_metrics.get("energy") {
            if *energy > 0.1 {
                // Basic energy threshold
                chunks_with_speech += 1;
                let speech_duration = 0.1; // chunk duration
                total_speech_duration += speech_duration;

                println!(
                    "   🗣️ Speech detected! Energy: {:.3}, Duration: {:.3}s",
                    energy, speech_duration
                );
            } else {
                println!("   🔇 No speech detected (energy: {:.3})", energy);
            }
        }

        // Display quality metrics for this chunk
        if let Some(snr) = analysis.quality_metrics.get("snr") {
            println!("   📈 Chunk SNR: {:.2} dB", snr);
        }

        // Performance metrics
        println!(
            "   ⏱️ Processing time: {:.2}ms (RTF: {:.3})",
            processing_time.as_secs_f64() * 1000.0,
            real_time_factor
        );

        if real_time_factor > 1.0 {
            println!("   ⚠️ Processing slower than real-time!");
        } else {
            println!("   ✅ Real-time performance maintained");
        }

        total_chunks_processed += 1;

        // Simulate real-time delay between chunks
        sleep(Duration::from_millis(chunk_duration_ms as u64)).await;
    }

    // Step 5: Performance validation
    println!("\n📊 Real-time Processing Summary:");
    println!("   • Total chunks processed: {}", total_chunks_processed);
    println!("   • Chunks with speech: {}", chunks_with_speech);
    println!("   • Total speech duration: {:.2}s", total_speech_duration);
    println!(
        "   • Speech detection rate: {:.1}%",
        (chunks_with_speech as f32 / total_chunks_processed as f32) * 100.0
    );

    // Step 6: Demonstrate latency optimization
    println!("\n⚡ Latency Optimization Techniques:");
    println!(
        "   • Chunk-based processing: {} ms chunks",
        chunk_duration_ms
    );
    println!("   • Disabled heavy analysis for real-time");
    println!("   • Optimized VAD for low latency");
    println!("   • Streaming-compatible preprocessing");

    // Step 7: Performance validation with requirements
    println!("\n🎯 Performance Validation:");
    let requirements = PerformanceRequirements {
        max_rtf: 0.3,                         // Stricter for real-time
        max_memory_usage: 1024 * 1024 * 1024, // 1GB for real-time systems
        max_startup_time_ms: 2000,            // 2 seconds
        max_streaming_latency_ms: 200,        // 200ms
    };

    let validator = PerformanceValidator::with_requirements(requirements);
    println!(
        "   • RTF requirement: < {:.2}",
        validator.requirements().max_rtf
    );
    println!(
        "   • Memory requirement: < {:.1} GB",
        validator.requirements().max_memory_usage as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "   • Latency requirement: < {} ms",
        validator.requirements().max_streaming_latency_ms
    );

    println!("\n✅ Real-time processing example completed!");
    println!("💡 Key takeaways:");
    println!("   • Use chunk-based processing for low latency");
    println!("   • Enable only necessary analysis features");
    println!("   • Monitor real-time factor (RTF) continuously");
    println!("   • Configure preprocessing for streaming scenarios");
    println!("   • Balance accuracy vs latency requirements");

    Ok(())
}

/// Create test audio chunks with varying characteristics for realistic simulation
fn create_test_audio_chunks(
    sample_rate: u32,
    chunk_size: usize,
    num_chunks: usize,
) -> Vec<AudioBuffer> {
    let mut chunks = Vec::new();

    for i in 0..num_chunks {
        let mut samples = Vec::with_capacity(chunk_size);

        // Create different types of audio for variety
        match i % 4 {
            0 => {
                // Silence
                for _ in 0..chunk_size {
                    samples.push(0.0);
                }
            }
            1 => {
                // Low-level noise
                for j in 0..chunk_size {
                    let noise = (j as f32 * 0.001).sin() * 0.01;
                    samples.push(noise);
                }
            }
            2 => {
                // Speech-like signal (multiple harmonics)
                for j in 0..chunk_size {
                    let t = j as f32 / sample_rate as f32;
                    let f0 = 150.0; // Typical male voice F0
                    let mut signal = 0.0;
                    // Add harmonics to simulate speech
                    for harmonic in 1..=5 {
                        let freq = f0 * harmonic as f32;
                        let amplitude = 1.0 / harmonic as f32; // Decreasing amplitude
                        signal += amplitude * (2.0 * std::f32::consts::PI * freq * t).sin();
                    }
                    samples.push(signal * 0.1);
                }
            }
            _ => {
                // Complex signal with modulation
                for j in 0..chunk_size {
                    let t = j as f32 / sample_rate as f32;
                    let carrier = 300.0 + 100.0 * (10.0 * t).sin(); // Frequency modulation
                    let envelope = 0.5 + 0.5 * (5.0 * t).sin(); // Amplitude modulation
                    let signal = envelope * (2.0 * std::f32::consts::PI * carrier * t).sin();
                    samples.push(signal * 0.08);
                }
            }
        }

        chunks.push(AudioBuffer::mono(samples, sample_rate));
    }

    chunks
}
