//! Streaming Synthesis Example - Real-time VoiRS Audio Generation
//!
//! This example demonstrates VoiRS's streaming synthesis capabilities for real-time audio generation.
//! It's ideal for applications requiring low latency, such as live conversations or interactive applications.
//!
//! ## What this example does:
//! 1. Sets up a streaming TTS pipeline
//! 2. Processes text in chunks for minimal latency
//! 3. Streams audio chunks in real-time
//! 4. Combines chunks into a final audio file
//! 5. Provides detailed streaming metrics and performance analysis
//!
//! ## Key Features Demonstrated:
//! - Real-time audio streaming
//! - Chunk-based processing for low latency
//! - Progress monitoring and metrics collection
//! - Error handling for stream processing
//! - Performance analysis (chunk timing, throughput)
//!
//! ## Prerequisites:
//! - Rust 1.70+ with async/await support
//! - VoiRS dependencies configured
//! - Futures crate for stream handling
//!
//! ## Running this example:
//! ```bash
//! cargo run --example streaming_synthesis
//! ```
//!
//! ## Expected output:
//! - Real-time chunk processing progress
//! - Streaming performance metrics
//! - `streaming_output.wav` file with combined audio
//! - Comprehensive performance analysis

use anyhow::{Context, Result};
use futures::StreamExt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use voirs::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize comprehensive logging for streaming analysis
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🎵 VoiRS Streaming Synthesis Example");
    println!("=====================================");
    println!();

    // Create TTS components with error handling
    info!("Setting up streaming TTS components...");
    let g2p = create_g2p(G2pBackend::RuleBased);
    let acoustic = create_acoustic(AcousticBackend::Vits);
    let vocoder = create_vocoder(VocoderBackend::HifiGan);

    println!("🔧 Building streaming pipeline...");
    let pipeline = Arc::new(
        VoirsPipelineBuilder::new()
            .with_g2p(g2p)
            .with_acoustic_model(acoustic)
            .with_vocoder(vocoder)
            .build()
            .await
            .context("Failed to build streaming synthesis pipeline")?,
    );
    println!("✅ Streaming pipeline ready!");

    let text = "This is a comprehensive demonstration of streaming speech synthesis technology. \
                Each audio chunk will be generated and delivered as quickly as possible, \
                enabling real-time playback with minimal latency and optimal user experience. \
                The advanced streaming system processes text incrementally, allowing for \
                immediate audio feedback and responsive interactive applications.";

    println!("\n📝 Input Text Analysis:");
    println!("   Text: \"{}\"", text);
    println!("   Length: {} characters", text.len());
    println!("   Words: ~{} words", text.split_whitespace().count());

    // Start streaming synthesis with timing
    let stream_start = Instant::now();
    info!("Initiating streaming synthesis...");

    let mut stream = pipeline
        .synthesize_stream(text)
        .await
        .context("Failed to start streaming synthesis")?;

    println!("\n🎵 Processing Audio Stream:");
    println!("   Processing chunks in real-time...");

    // Stream processing with comprehensive metrics
    let mut total_chunks = 0;
    let mut total_duration = 0.0f32;
    let mut combined_samples = Vec::new();
    let mut sample_rate = 22050; // Default sample rate
    let mut chunk_timings = Vec::new();
    let mut largest_chunk = 0.0f32;
    let mut smallest_chunk = f32::MAX;

    while let Some(chunk_result) = stream.next().await {
        let chunk_start = Instant::now();

        let chunk = chunk_result.context("Failed to process audio chunk from stream")?;

        let chunk_processing_time = chunk_start.elapsed();
        chunk_timings.push(chunk_processing_time);

        total_chunks += 1;
        let chunk_duration = chunk.duration();
        total_duration += chunk_duration;

        // Track chunk size statistics
        if chunk_duration > largest_chunk {
            largest_chunk = chunk_duration;
        }
        if chunk_duration < smallest_chunk {
            smallest_chunk = chunk_duration;
        }

        // Combine samples for final output
        if combined_samples.is_empty() {
            sample_rate = chunk.sample_rate();
            info!("Stream sample rate: {} Hz", sample_rate);
        }
        combined_samples.extend_from_slice(chunk.samples());

        // Real-time progress feedback
        let real_time_factor = chunk_processing_time.as_secs_f64() / chunk_duration as f64;
        println!(
            "   Chunk {:2}: {:.2}s audio | Processing: {:.1}ms | RTF: {:.2}x | Total: {:.2}s",
            total_chunks,
            chunk_duration,
            chunk_processing_time.as_millis(),
            real_time_factor,
            total_duration
        );

        // Warning for slow chunks
        if real_time_factor > 1.0 {
            warn!(
                "Chunk {} processing slower than real-time (RTF: {:.2}x)",
                total_chunks, real_time_factor
            );
        }

        debug!(
            "Chunk {} samples: {}, duration: {:.3}s",
            total_chunks,
            chunk.samples().len(),
            chunk_duration
        );
    }

    let total_streaming_time = stream_start.elapsed();

    // Comprehensive streaming analysis
    println!("\n✅ Streaming synthesis complete!");
    println!("\n📊 Streaming Performance Analysis:");
    println!("   Total Chunks: {}", total_chunks);
    println!("   Total Audio Duration: {:.2} seconds", total_duration);
    println!(
        "   Total Processing Time: {:.2} seconds",
        total_streaming_time.as_secs_f32()
    );
    println!(
        "   Overall Real-time Factor: {:.2}x",
        total_streaming_time.as_secs_f64() / total_duration as f64
    );

    if total_chunks > 0 {
        println!(
            "   Average Chunk Size: {:.2} seconds",
            total_duration as f64 / total_chunks as f64
        );
        println!("   Largest Chunk: {:.2} seconds", largest_chunk);
        println!(
            "   Smallest Chunk: {:.2} seconds",
            if smallest_chunk == f32::MAX {
                0.0
            } else {
                smallest_chunk
            }
        );

        let avg_processing_time =
            chunk_timings.iter().map(|d| d.as_secs_f64()).sum::<f64>() / chunk_timings.len() as f64;
        println!(
            "   Average Processing Time per Chunk: {:.1} ms",
            avg_processing_time * 1000.0
        );
    }

    // Save combined audio with error handling
    println!("\n💾 Saving combined audio...");
    let output_file = "streaming_output.wav";

    let final_audio = AudioBuffer::new(combined_samples, sample_rate, 1);
    final_audio
        .save_wav(output_file)
        .context("Failed to save streaming audio output")?;

    println!("✅ Audio saved to: {}", output_file);

    // Final audio information
    println!("\n📊 Final Audio Information:");
    println!("   File: {}", output_file);
    println!("   Sample Rate: {} Hz", sample_rate);
    println!("   Duration: {:.2} seconds", final_audio.duration());
    println!("   Channels: {}", final_audio.channels());
    println!("   Total Samples: {}", final_audio.samples().len());

    // Performance assessment
    let overall_rtf = total_streaming_time.as_secs_f64() / total_duration as f64;
    println!("\n🚀 Performance Assessment:");
    if overall_rtf < 0.5 {
        println!(
            "   ✅ Excellent real-time performance (RTF: {:.2}x)",
            overall_rtf
        );
    } else if overall_rtf < 1.0 {
        println!(
            "   ✅ Good real-time performance (RTF: {:.2}x)",
            overall_rtf
        );
    } else {
        println!("   ⚠️  Slower than real-time (RTF: {:.2}x)", overall_rtf);
    }

    println!("\n🎉 Streaming synthesis demonstration complete!");
    println!("💡 Next steps:");
    println!("   - Play '{}' to hear the streaming result", output_file);
    println!("   - Try with longer or shorter text for different chunk patterns");
    println!("   - Explore real-time applications with this streaming capability");

    Ok(())
}
