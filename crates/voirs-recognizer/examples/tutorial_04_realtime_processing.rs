//! Tutorial 04: Real-time Processing
//!
//! This tutorial teaches you how to implement real-time speech recognition
//! with streaming audio, partial results, and latency optimization.
//!
//! Learning Objectives:
//! - Understand real-time vs batch processing
//! - Configure streaming recognition
//! - Handle partial results and incremental updates
//! - Optimize for low latency
//! - Implement voice activity detection
//! - Handle streaming errors and reconnection
//!
//! Prerequisites: Complete Tutorials 01-03
//!
//! Usage:
//! ```bash
//! cargo run --example tutorial_04_realtime_processing --features="whisper-pure"
//! ```

use std::error::Error;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use voirs_recognizer::asr::{ASRBackend, FallbackConfig, WhisperModelSize};
use voirs_recognizer::integration::config::{LatencyMode, StreamingConfig};
use voirs_recognizer::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🎓 Tutorial 04: Real-time Processing");
    println!("====================================\n");

    // Step 1: Introduction to real-time processing
    println!("📚 Learning Goal: Master real-time speech recognition");
    println!("   • Configure streaming recognition");
    println!("   • Handle partial results");
    println!("   • Optimize for low latency");
    println!("   • Implement voice activity detection");
    println!("   • Handle streaming errors\n");

    // Step 2: Understanding real-time vs batch
    println!("🔍 Step 1: Real-time vs Batch Processing");
    explain_processing_modes();

    // Step 3: Configure streaming
    println!(
        "
🔧 Step 2: Configuring Streaming Recognition"
    );
    let streaming_configs = demonstrate_streaming_configs().await?;

    // Step 4: Demonstrate different latency modes
    println!(
        "
⚡ Step 3: Latency Optimization"
    );
    for (mode_name, config) in streaming_configs {
        println!(
            "   
   {} Mode:",
            mode_name
        );
        demonstrate_streaming_mode(&config).await?;
    }

    // Step 5: Voice Activity Detection
    println!(
        "
🎤 Step 4: Voice Activity Detection"
    );
    demonstrate_voice_activity_detection().await?;

    // Step 6: Error handling and recovery
    println!(
        "
🛠️ Step 5: Error Handling and Recovery"
    );
    demonstrate_error_handling().await?;

    // Step 7: Performance monitoring
    println!(
        "
📊 Step 6: Performance Monitoring"
    );
    demonstrate_performance_monitoring().await?;

    // Step 8: Conclusion
    println!(
        "
🎉 Congratulations! You've completed Tutorial 04!"
    );
    println!(
        "
📖 What you learned:"
    );
    println!("   • Difference between real-time and batch processing");
    println!("   • How to configure streaming recognition");
    println!("   • How to handle partial results");
    println!("   • How to optimize for low latency");
    println!("   • How to implement voice activity detection");
    println!("   • How to handle streaming errors");

    println!(
        "
🚀 Next Steps:"
    );
    println!("   • Tutorial 05: Multi-language support");
    println!("   • Tutorial 06: Performance optimization");
    println!("   • Tutorial 07: Integration examples");

    Ok(())
}

fn explain_processing_modes() {
    println!(
        "   
   📊 Processing Mode Comparison:"
    );
    println!(
        "   
   Batch Processing:"
    );
    println!("   • Process complete audio files");
    println!("   • Higher accuracy (full context)");
    println!("   • Higher latency (wait for completion)");
    println!("   • Better for transcription, analysis");

    println!(
        "   
   Real-time Processing:"
    );
    println!("   • Process audio as it arrives");
    println!("   • Lower latency (immediate feedback)");
    println!("   • Partial results available");
    println!("   • Better for live applications");

    println!(
        "   
   📋 Key Differences:"
    );
    println!("   • Latency: Batch (seconds) vs Real-time (milliseconds)");
    println!("   • Accuracy: Batch (higher) vs Real-time (good)");
    println!("   • Memory: Batch (variable) vs Real-time (constant)");
    println!("   • Use case: Batch (analysis) vs Real-time (interaction)");
}

async fn demonstrate_streaming_configs(
) -> Result<Vec<(&'static str, StreamingConfig)>, Box<dyn Error>> {
    println!("   Real-time processing requires different configurations:");

    let configs = vec![
        (
            "🚀 Ultra-Low Latency",
            StreamingConfig {
                latency_mode: LatencyMode::UltraLow,
                chunk_size: 400,      // 25ms chunks at 16kHz
                overlap: 80,          // 5ms overlap
                buffer_duration: 1.0, // 1 second buffer
            },
        ),
        (
            "⚖️ Balanced",
            StreamingConfig {
                latency_mode: LatencyMode::Balanced,
                chunk_size: 1600,     // 100ms chunks at 16kHz
                overlap: 320,         // 20ms overlap
                buffer_duration: 3.0, // 3 second buffer
            },
        ),
        (
            "🎯 High Accuracy",
            StreamingConfig {
                latency_mode: LatencyMode::Accurate,
                chunk_size: 4800,     // 300ms chunks at 16kHz
                overlap: 800,         // 50ms overlap
                buffer_duration: 5.0, // 5 second buffer
            },
        ),
    ];

    for (name, config) in &configs {
        println!(
            "   
   {} Configuration:",
            name
        );
        println!("   • Chunk size: {}ms", config.chunk_size as f32 / 16.0);
        println!("   • Overlap: {}ms", config.overlap as f32 / 16.0);
        println!("   • Buffer duration: {:.1}s", config.buffer_duration);
        println!("   • Expected latency: ~{}ms", estimate_latency(&config));
    }

    Ok(configs)
}

fn estimate_latency(config: &StreamingConfig) -> u32 {
    let chunk_latency = config.chunk_size as f32 / 16.0; // Convert to ms
    let processing_overhead = match config.latency_mode {
        LatencyMode::UltraLow => 10.0,
        LatencyMode::Low => 25.0,
        LatencyMode::Balanced => 50.0,
        LatencyMode::HighAccuracy => 100.0,
        LatencyMode::Accurate => 100.0,
    };

    (chunk_latency + processing_overhead) as u32
}

async fn demonstrate_streaming_mode(config: &StreamingConfig) -> Result<(), Box<dyn Error>> {
    println!(
        "   🔄 Simulating streaming with {} mode...",
        match config.latency_mode {
            LatencyMode::UltraLow => "Ultra-Low Latency",
            LatencyMode::Low => "Low Latency",
            LatencyMode::Balanced => "Balanced",
            LatencyMode::HighAccuracy => "High Accuracy",
            LatencyMode::Accurate => "High Accuracy",
        }
    );

    // Simulate streaming audio chunks
    let chunk_duration_ms = config.chunk_size as f32 / 16.0;
    let total_chunks = 10;

    println!(
        "   📊 Processing {} chunks of {:.1}ms each...",
        total_chunks, chunk_duration_ms
    );

    let mut partial_transcript = String::new();
    let mut total_latency = 0.0;

    for i in 1..=total_chunks {
        let start_time = Instant::now();

        // Simulate chunk processing
        let processing_time = simulate_chunk_processing(config).await;
        let latency = start_time.elapsed();
        total_latency += latency.as_millis() as f32;

        // Simulate partial result
        let word = match i {
            1..=3 => "Hello",
            4..=6 => "Hello world",
            7..=8 => "Hello world this",
            9..=10 => "Hello world this is",
            _ => "Hello world this is a test",
        };

        partial_transcript = word.to_string();

        println!(
            "   Chunk {}/{}:  \"{}\" ({}ms latency)",
            i,
            total_chunks,
            partial_transcript,
            latency.as_millis()
        );

        // Simulate real-time arrival of next chunk
        sleep(Duration::from_millis(chunk_duration_ms as u64)).await;
    }

    println!("   ✅ Streaming complete!");
    println!("   • Final transcript: \"{}\"", partial_transcript);
    println!(
        "   • Average latency: {:.1}ms",
        total_latency / total_chunks as f32
    );
    println!("   • Total processing time: {:.1}s", total_latency / 1000.0);

    Ok(())
}

async fn simulate_chunk_processing(config: &StreamingConfig) -> Duration {
    let base_processing_time = match config.latency_mode {
        LatencyMode::UltraLow => 5,
        LatencyMode::Low => 15,
        LatencyMode::Balanced => 25,
        LatencyMode::HighAccuracy => 75,
        LatencyMode::Accurate => 75,
    };

    // Add some randomness to simulate real processing
    let jitter = (scirs2_core::random::random::<f32>() * 10.0) as u64;
    let processing_time = base_processing_time + jitter;

    sleep(Duration::from_millis(processing_time)).await;
    Duration::from_millis(processing_time)
}

async fn demonstrate_voice_activity_detection() -> Result<(), Box<dyn Error>> {
    println!("   Voice Activity Detection (VAD) is crucial for real-time processing:");

    println!(
        "   
   🎯 VAD Benefits:"
    );
    println!("   • Reduces unnecessary processing");
    println!("   • Saves battery on mobile devices");
    println!("   • Improves recognition accuracy");
    println!("   • Enables push-to-talk alternatives");

    println!(
        "   
   🔍 VAD Simulation:"
    );

    // Simulate audio stream with voice activity
    let audio_segments = vec![
        ("🔇 Silence", 0.0, false),
        ("🎤 Speech", 0.85, true),
        ("🔇 Silence", 0.1, false),
        ("🎤 Speech", 0.92, true),
        ("🔇 Silence", 0.05, false),
        ("🎤 Speech", 0.78, true),
        ("🔇 Silence", 0.0, false),
    ];

    let mut total_processed = 0;
    let mut total_skipped = 0;

    for (segment_type, energy, should_process) in audio_segments {
        println!("   {} (energy: {:.2})", segment_type, energy);

        if should_process {
            println!("   → Processing speech segment...");
            total_processed += 1;
            sleep(Duration::from_millis(50)).await; // Simulate processing
        } else {
            println!("   → Skipping silence (VAD saves processing)");
            total_skipped += 1;
        }
    }

    println!(
        "   
   📊 VAD Results:"
    );
    println!("   • Segments processed: {}", total_processed);
    println!("   • Segments skipped: {}", total_skipped);
    println!(
        "   • Processing saved: {:.1}%",
        (total_skipped as f32 / (total_processed + total_skipped) as f32) * 100.0
    );

    Ok(())
}

async fn demonstrate_error_handling() -> Result<(), Box<dyn Error>> {
    println!("   Real-time processing requires robust error handling:");

    println!(
        "   
   🛠️ Common Streaming Errors:"
    );
    println!("   • Audio buffer overflow");
    println!("   • Network connectivity issues");
    println!("   • Processing timeout");
    println!("   • Model loading failures");
    println!("   • Resource exhaustion");

    println!(
        "   
   🔄 Error Recovery Strategies:"
    );

    // Simulate different error scenarios
    let error_scenarios = vec![
        (
            "Buffer Overflow",
            "Increase buffer size, implement backpressure",
        ),
        (
            "Processing Timeout",
            "Switch to faster model, reduce chunk size",
        ),
        (
            "Network Issues",
            "Implement local fallback, queue for retry",
        ),
        ("Memory Exhaustion", "Clear old buffers, reduce model size"),
        (
            "Model Loading",
            "Use cached model, implement model fallback",
        ),
    ];

    for (error_type, solution) in error_scenarios {
        println!(
            "   
   ⚠️ {} Error:",
            error_type
        );
        println!("   → Solution: {}", solution);

        // Simulate error recovery
        sleep(Duration::from_millis(100)).await;
        println!("   ✅ Recovered successfully");
    }

    println!(
        "   
   🛡️ Best Practices:"
    );
    println!("   • Implement exponential backoff for retries");
    println!("   • Use circuit breakers for failing services");
    println!("   • Monitor system resources continuously");
    println!("   • Provide graceful degradation options");
    println!("   • Log errors for debugging and monitoring");

    Ok(())
}

async fn demonstrate_performance_monitoring() -> Result<(), Box<dyn Error>> {
    println!("   Performance monitoring is essential for real-time systems:");

    println!(
        "   
   📊 Key Metrics to Monitor:"
    );
    println!("   • Latency: Time from audio input to result");
    println!("   • Throughput: Audio processed per second");
    println!("   • CPU usage: Processing overhead");
    println!("   • Memory usage: Buffer and model memory");
    println!("   • Queue depth: Backlog of pending audio");

    println!(
        "   
   🔄 Simulating Performance Monitoring:"
    );

    // Simulate performance monitoring
    let monitoring_duration = 5; // seconds
    for second in 1..=monitoring_duration {
        let latency = 45.0 + (scirs2_core::random::random::<f32>() * 20.0); // 45-65ms
        let cpu_usage = 25.0 + (scirs2_core::random::random::<f32>() * 15.0); // 25-40%
        let memory_usage = 150.0 + (scirs2_core::random::random::<f32>() * 50.0); // 150-200MB
        let queue_depth = (scirs2_core::random::random::<f32>() * 5.0) as u32; // 0-5 items

        println!(
            "   Second {}: latency={:.1}ms, CPU={:.1}%, memory={:.0}MB, queue={}",
            second, latency, cpu_usage, memory_usage, queue_depth
        );

        // Check for performance issues
        if latency > 60.0 {
            println!("   ⚠️ High latency detected!");
        }
        if cpu_usage > 35.0 {
            println!("   ⚠️ High CPU usage detected!");
        }
        if queue_depth > 3 {
            println!("   ⚠️ Queue backlog detected!");
        }

        sleep(Duration::from_millis(200)).await;
    }

    println!(
        "   
   📈 Performance Recommendations:"
    );
    println!("   • Target latency: < 100ms for interactive applications");
    println!("   • Target CPU usage: < 30% for sustainable performance");
    println!("   • Target memory usage: < 500MB for mobile devices");
    println!("   • Target queue depth: < 2 items for responsive experience");

    Ok(())
}

// Simple random number generator for demo
mod rand {
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn random<T>() -> T
    where
        T: From<f32>,
    {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let seed = now.as_nanos() as u64;
        let value = (seed % 1000) as f32 / 1000.0;
        T::from(value)
    }
}
