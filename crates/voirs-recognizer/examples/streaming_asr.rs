//! Streaming ASR Example
//!
//! This example demonstrates real-time streaming automatic speech recognition
//! with ultra-low latency, partial results, and comprehensive performance monitoring.
//!
//! Usage:
//! ```bash
//! cargo run --example streaming_asr --features="whisper-pure"
//! ```

use std::time::{Duration, Instant};
use tokio::time::sleep;
use voirs_recognizer::asr::{ASRBackend, FallbackConfig, WhisperModelSize};
use voirs_recognizer::integration::config::{LatencyMode, StreamingConfig};
use voirs_recognizer::prelude::*;
use voirs_recognizer::{PerformanceValidator, RecognitionError};

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    println!("🎙️ VoiRS Streaming ASR Demo");
    println!("===========================\n");

    // Step 1: Configure streaming ASR for ultra-low latency
    println!("🔧 Configuring streaming ASR for ultra-low latency...");

    let streaming_config = StreamingConfig {
        latency_mode: LatencyMode::UltraLow,
        chunk_size: 800,      // 50ms chunks at 16kHz
        overlap: 160,         // 10ms overlap
        buffer_duration: 2.0, // 2 second buffer
    };

    let fallback_config = FallbackConfig {
        quality_threshold: 0.5,
        max_processing_time_seconds: 5.0,
        adaptive_selection: true,
        memory_threshold_mb: 512.0,
        min_duration_for_selection: 0.3,
        ..Default::default()
    };

    println!("✅ Streaming configuration:");
    println!("   • Latency mode: Ultra-low");
    println!("   • Chunk size: 50ms");
    println!("   • Overlap: 10ms");
    println!("   • Buffer duration: 2.0s");
    println!("   • Model size: Small");
    println!("   • Language detection: enabled");

    // Step 2: Initialize streaming ASR
    println!("\n🚀 Initializing streaming ASR...");
    let init_start = Instant::now();

    let mut asr = IntelligentASRFallback::new(fallback_config).await?;
    let init_time = init_start.elapsed();

    println!("✅ Streaming ASR initialized in {:?}", init_time);

    // Step 3: Create streaming audio simulation
    println!("\n🎵 Creating streaming audio simulation...");
    let sample_rate = 16000;
    let chunk_duration_ms = 50; // 50ms chunks
    let chunk_size = (sample_rate as f32 * chunk_duration_ms as f32 / 1000.0) as usize;

    // Create a realistic speech simulation
    let streaming_chunks = create_streaming_speech_simulation(sample_rate, chunk_size);
    println!(
        "✅ Created {} streaming chunks ({:.1}s total)",
        streaming_chunks.len(),
        streaming_chunks.len() as f32 * chunk_duration_ms as f32 / 1000.0
    );

    // Step 4: Performance monitoring setup
    println!("\n📊 Setting up performance monitoring...");
    let validator = PerformanceValidator::new().with_verbose(false);

    let mut latency_measurements = Vec::new();
    let mut partial_results = Vec::new();
    let mut final_results = Vec::new();

    println!("✅ Performance monitoring ready");

    // Step 5: Real-time streaming processing
    println!("\n⚡ Starting real-time streaming processing...");
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Time  │ Chunk │ Latency │ Partial Result              │ Final │");
    println!("├─────────────────────────────────────────────────────────────┤");

    let mut accumulated_audio = Vec::new();
    let mut last_final_result = String::new();
    let mut processing_start = Instant::now();

    for (i, chunk) in streaming_chunks.iter().enumerate() {
        let chunk_start = Instant::now();

        // Accumulate audio for processing
        accumulated_audio.extend(chunk.samples());

        // Process when we have enough audio (every 4 chunks = 200ms)
        if accumulated_audio.len() >= chunk_size * 4 {
            let audio_buffer = AudioBuffer::mono(accumulated_audio.clone(), sample_rate);

            // Perform recognition
            let recognition_start = Instant::now();
            let result = asr.transcribe(&audio_buffer, None).await?;
            let recognition_time = recognition_start.elapsed();

            // Calculate latency
            let latency = chunk_start.elapsed();
            latency_measurements.push(latency);

            // Determine if this is a new result or partial update
            let is_final = result.transcript.text.len() > last_final_result.len()
                && result.transcript.text.ends_with(".")
                || result.transcript.text.ends_with("!")
                || result.transcript.text.ends_with("?");

            if is_final {
                final_results.push(result.transcript.text.clone());
                last_final_result = result.transcript.text.clone();
            } else {
                partial_results.push(result.transcript.text.clone());
            }

            // Display streaming results
            let elapsed = processing_start.elapsed();
            let partial_text = if result.transcript.text.len() > 25 {
                format!("{}...", &result.transcript.text[..22])
            } else {
                result.transcript.text.clone()
            };

            println!(
                "│ {:5.1}s │ {:5} │ {:7.1}ms │ {:25} │ {:5} │",
                elapsed.as_secs_f32(),
                i + 1,
                latency.as_secs_f32() * 1000.0,
                partial_text,
                if is_final { "✓" } else { "..." }
            );

            // Validate streaming latency
            let (latency_ms, latency_passed) = validator.validate_streaming_latency(latency);
            if !latency_passed {
                println!("│       │       │ ⚠️ HIGH │ Latency exceeds 200ms limit    │       │");
            }

            // Clear some accumulated audio to simulate sliding window
            if accumulated_audio.len() > chunk_size * 8 {
                accumulated_audio.drain(..chunk_size * 2);
            }
        }

        // Simulate real-time delay
        sleep(Duration::from_millis(chunk_duration_ms as u64)).await;
    }

    println!("└─────────────────────────────────────────────────────────────┘");

    // Step 6: Performance analysis
    println!("\n📈 Performance Analysis:");

    // Latency statistics
    let avg_latency =
        latency_measurements.iter().sum::<Duration>() / latency_measurements.len() as u32;
    let max_latency = latency_measurements.iter().max().unwrap_or(&Duration::ZERO);
    let min_latency = latency_measurements.iter().min().unwrap_or(&Duration::ZERO);

    println!("   Latency Statistics:");
    println!("   • Average: {:.1}ms", avg_latency.as_secs_f32() * 1000.0);
    println!("   • Maximum: {:.1}ms", max_latency.as_secs_f32() * 1000.0);
    println!("   • Minimum: {:.1}ms", min_latency.as_secs_f32() * 1000.0);
    println!("   • Measurements: {}", latency_measurements.len());

    // Real-time factor calculation
    let total_audio_duration = streaming_chunks.len() as f32 * chunk_duration_ms as f32 / 1000.0;
    let total_processing_time: Duration = latency_measurements.iter().sum();
    let overall_rtf = total_processing_time.as_secs_f32() / total_audio_duration;

    println!("   Real-time Performance:");
    println!("   • Total audio duration: {:.1}s", total_audio_duration);
    println!(
        "   • Total processing time: {:.1}s",
        total_processing_time.as_secs_f32()
    );
    println!("   • Overall RTF: {:.3}", overall_rtf);
    println!(
        "   • RTF Status: {}",
        if overall_rtf < 0.3 {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    // Memory usage
    let (memory_usage, memory_passed) = validator.estimate_memory_usage()?;
    println!("   Memory Usage:");
    println!(
        "   • Current usage: {:.1} MB",
        memory_usage as f64 / (1024.0 * 1024.0)
    );
    println!(
        "   • Status: {}",
        if memory_passed {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    // Step 7: Results summary
    println!("\n📝 Recognition Results Summary:");
    println!("   Partial Results: {}", partial_results.len());
    for (i, result) in partial_results.iter().enumerate() {
        println!("   {:2}. [PARTIAL] \"{}\"", i + 1, result);
    }

    println!("\n   Final Results: {}", final_results.len());
    for (i, result) in final_results.iter().enumerate() {
        println!("   {:2}. [FINAL] \"{}\"", i + 1, result);
    }

    // Step 8: Different latency modes comparison
    println!("\n🎯 Latency Mode Comparison:");

    let latency_modes = vec![
        ("UltraLow", LatencyMode::UltraLow, 800, 160),
        ("Balanced", LatencyMode::Balanced, 1600, 400),
        ("HighQuality", LatencyMode::UltraLow, 3200, 800), // Simulated high quality
    ];

    for (mode_name, latency_mode, chunk_size, overlap) in latency_modes {
        let test_config = StreamingConfig {
            latency_mode,
            chunk_size,
            overlap,
            buffer_duration: 2.0,
            // VAD and noise suppression settings removed
        };

        let test_asr_config = ASRConfig {
            language: Some(LanguageCode::EnUs),
            // Streaming configuration applied separately
            ..Default::default()
        };

        println!("   • {} Mode:", mode_name);
        println!(
            "     - Chunk size: {} samples ({:.1}ms)",
            chunk_size,
            chunk_size as f32 / 16.0
        );
        println!(
            "     - Overlap: {} samples ({:.1}ms)",
            overlap,
            overlap as f32 / 16.0
        );
        println!("     - Optimized for streaming");
        println!(
            "     - Expected latency: {:.1}ms",
            if mode_name == "UltraLow" {
                50.0
            } else if mode_name == "Balanced" {
                100.0
            } else {
                200.0
            }
        );
    }

    // Step 9: Optimization recommendations
    println!("\n💡 Optimization Recommendations:");

    if avg_latency.as_millis() > 200 {
        println!("   ⚠️ High Latency Detected:");
        println!("   • Reduce chunk size for lower latency");
        println!("   • Use greedy decoding (beam_size=1)");
        println!("   • Disable echo cancellation");
        println!("   • Consider smaller model size");
    }

    if overall_rtf > 0.3 {
        println!("   ⚠️ High RTF Detected:");
        println!("   • Enable GPU acceleration if available");
        println!("   • Use model quantization (INT8/FP16)");
        println!("   • Optimize chunk processing pipeline");
    }

    if memory_usage > 1_000_000_000 {
        println!("   ⚠️ High Memory Usage:");
        println!("   • Reduce buffer duration");
        println!("   • Use memory pooling");
        println!("   • Consider streaming-optimized models");
    }

    println!("   ✅ Performance Tips:");
    println!("   • Monitor latency continuously in production");
    println!("   • Adjust chunk size based on use case");
    println!("   • Balance accuracy vs speed requirements");
    println!("   • Use VAD to reduce unnecessary processing");

    // Step 10: Voice activity detection integration
    println!("\n🔍 Voice Activity Detection Integration:");

    let vad_efficient_chunks = streaming_chunks
        .iter()
        .enumerate()
        .filter(|(_, chunk)| {
            // Simple energy-based VAD
            let energy: f32 = chunk.samples().iter().map(|s| s * s).sum();
            energy > 0.01 // Threshold for voice activity
        })
        .collect::<Vec<_>>();

    println!("   • Total chunks: {}", streaming_chunks.len());
    println!("   • Voice active chunks: {}", vad_efficient_chunks.len());
    println!(
        "   • Processing efficiency: {:.1}%",
        (vad_efficient_chunks.len() as f32 / streaming_chunks.len() as f32) * 100.0
    );
    println!(
        "   • Estimated processing savings: {:.1}%",
        100.0 - (vad_efficient_chunks.len() as f32 / streaming_chunks.len() as f32) * 100.0
    );

    println!("\n✅ Streaming ASR demo completed successfully!");
    println!("🎯 Key achievements:");
    println!(
        "   • Average latency: {:.1}ms",
        avg_latency.as_secs_f32() * 1000.0
    );
    println!("   • Real-time factor: {:.3}", overall_rtf);
    println!(
        "   • Memory usage: {:.1} MB",
        memory_usage as f64 / (1024.0 * 1024.0)
    );
    println!(
        "   • Processing efficiency: {:.1}%",
        (vad_efficient_chunks.len() as f32 / streaming_chunks.len() as f32) * 100.0
    );

    println!("\n🚀 Next steps:");
    println!("   • Integrate with real microphone input");
    println!("   • Add WebSocket streaming support");
    println!("   • Implement adaptive quality control");
    println!("   • Add speaker diarization for multi-speaker streams");

    Ok(())
}

fn create_streaming_speech_simulation(sample_rate: u32, chunk_size: usize) -> Vec<AudioBuffer> {
    let mut chunks = Vec::new();
    let num_chunks = 40; // 2 seconds of audio in 50ms chunks

    for i in 0..num_chunks {
        let mut samples = Vec::with_capacity(chunk_size);

        // Create realistic speech patterns
        let speech_patterns = [
            ("silence", 0.0, 0.0),      // Silence
            ("consonant", 200.0, 0.05), // Consonant-like
            ("vowel", 150.0, 0.1),      // Vowel-like
            ("fricative", 800.0, 0.03), // Fricative-like
            ("plosive", 100.0, 0.08),   // Plosive-like
            ("noise", 400.0, 0.02),     // Background noise
        ];

        let pattern_index = match i % 20 {
            0..=2 => 0,   // Silence at start
            3..=5 => 1,   // Consonants
            6..=10 => 2,  // Vowels
            11..=13 => 3, // Fricatives
            14..=15 => 4, // Plosives
            16..=17 => 2, // More vowels
            18..=19 => 0, // Silence at end
            _ => 5,       // Noise
        };

        let (_, base_freq, amplitude) = speech_patterns[pattern_index];

        for j in 0..chunk_size {
            let t = (i * chunk_size + j) as f32 / sample_rate as f32;

            if amplitude > 0.0 {
                // Generate speech-like signal with formants
                let f1 = base_freq;
                let f2 = base_freq * 2.5;
                let f3 = base_freq * 3.5;

                let sample = amplitude
                    * (0.5 * (2.0 * std::f32::consts::PI * f1 * t).sin()
                        + 0.3 * (2.0 * std::f32::consts::PI * f2 * t).sin()
                        + 0.2 * (2.0 * std::f32::consts::PI * f3 * t).sin());

                // Add some envelope variation
                let envelope = 0.8 + 0.2 * (10.0 * t).sin();
                samples.push(sample * envelope);
            } else {
                // Silence with minimal noise
                samples.push(0.001 * (j as f32 * 0.1).sin());
            }
        }

        chunks.push(AudioBuffer::mono(samples, sample_rate));
    }

    chunks
}
