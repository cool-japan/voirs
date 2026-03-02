//! # Streaming Synthesis Example
//!
//! This example demonstrates real-time streaming synthesis simulation,
//! which is essential for applications requiring low latency
//! and immediate audio feedback.

use std::sync::Arc;
use std::time::Instant;
use tokio::time::{sleep, Duration};
use voirs_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Streaming Synthesis Demo ===\n");

    // Create a pipeline optimized for streaming
    let pipeline = Arc::new(
        VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await?,
    );

    // Basic streaming example
    basic_streaming_example(&pipeline).await?;

    // Real-time streaming example
    real_time_streaming_example(&pipeline).await?;

    // Interactive streaming example
    interactive_streaming_example(&pipeline).await?;

    // Performance monitoring example
    performance_monitoring_example(&pipeline).await?;

    Ok(())
}

async fn basic_streaming_example(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("=== Basic Streaming ===");

    let text = "This is a streaming synthesis example. The audio will be generated \
               in real-time as chunks, allowing for immediate playback without \
               waiting for the entire synthesis to complete.";

    println!("Text: {}", text);
    println!("Starting streaming synthesis...\n");

    let start_time = Instant::now();
    let audio = pipeline.synthesize(text).await?;

    // Simulate chunked delivery with a chunk size of 1024 samples
    let chunk_size = 1024_usize;
    let samples = audio.samples();
    let mut chunk_count = 0;
    let mut total_samples = 0;

    for chunk in samples.chunks(chunk_size) {
        chunk_count += 1;
        total_samples += chunk.len();

        let elapsed = start_time.elapsed();
        println!(
            "Chunk {}: {} samples at {:.2}s",
            chunk_count,
            chunk.len(),
            elapsed.as_secs_f64()
        );

        // Simulate real-time processing
        process_audio_chunk(chunk, chunk_count).await?;
    }

    let total_time = start_time.elapsed();
    let audio_duration = audio.duration() as f64;
    let real_time_factor = if audio_duration > 0.0 {
        total_time.as_secs_f64() / audio_duration
    } else {
        0.0
    };

    println!("\nStreaming complete!");
    println!("Total chunks: {}", chunk_count);
    println!("Total samples: {}", total_samples);
    println!("Audio duration: {:.2}s", audio_duration);
    println!("Processing time: {:.2}s", total_time.as_secs_f64());
    println!("Real-time factor: {:.2}", real_time_factor);
    println!();

    Ok(())
}

async fn real_time_streaming_example(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("=== Real-time Streaming ===");

    let sentences = vec![
        "Welcome to the real-time streaming demo.",
        "Each sentence will be synthesized separately.",
        "This allows for immediate audio feedback.",
        "Perfect for live applications and assistants.",
    ];

    for (i, sentence) in sentences.iter().enumerate() {
        println!("Streaming sentence {}: {}", i + 1, sentence);

        let start = Instant::now();
        let audio = pipeline.synthesize(sentence).await?;

        // Simulate chunked playback
        let chunk_size = 1024_usize;
        let samples = audio.samples();
        for chunk in samples.chunks(chunk_size) {
            // Simulate real-time playback
            let playback_duration =
                Duration::from_millis((chunk.len() as f64 / 22050.0 * 1000.0) as u64);
            sleep(playback_duration).await;
        }

        let synthesis_time = start.elapsed();
        let audio_duration = audio.duration() as f64;
        let rtf = if audio_duration > 0.0 {
            synthesis_time.as_secs_f64() / audio_duration
        } else {
            0.0
        };

        println!(
            "  Synthesis: {:.2}s, Audio: {:.2}s, RTF: {:.2}",
            synthesis_time.as_secs_f64(),
            audio_duration,
            rtf
        );
        println!();
    }

    Ok(())
}

async fn interactive_streaming_example(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("=== Interactive Streaming ===");
    println!("Simulating interactive conversation...\n");

    let conversation = vec![
        ("User", "Hello, how are you today?"),
        (
            "Assistant",
            "I'm doing great! Thanks for asking. How can I help you?",
        ),
        ("User", "Can you explain streaming synthesis?"),
        (
            "Assistant",
            "Streaming synthesis generates audio in real-time chunks, \
                      enabling immediate playback without waiting for completion.",
        ),
        ("User", "That sounds very useful!"),
        (
            "Assistant",
            "Absolutely! It's perfect for conversational AI and live applications.",
        ),
    ];

    for (speaker, text) in conversation {
        println!("{}: {}", speaker, text);

        if speaker == "Assistant" {
            // Synthesize the assistant's response
            let audio = pipeline.synthesize(text).await?;
            print!("  Audio: ");

            let chunk_size = 1024_usize;
            let samples = audio.samples();
            for chunk in samples.chunks(chunk_size) {
                print!("*"); // Visual indicator of audio chunk
                std::io::Write::flush(&mut std::io::stdout()).ok();

                // Simulate audio playback timing
                let chunk_duration = chunk.len() as f64 / 22050.0;
                sleep(Duration::from_millis((chunk_duration * 100.0) as u64)).await;
            }
            println!(" (synthesis complete)");
        }

        println!();
        sleep(Duration::from_millis(100)).await; // Pause between exchanges
    }

    Ok(())
}

async fn performance_monitoring_example(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("=== Performance Monitoring ===");

    let test_texts = vec![
        "Short text.",
        "This is a medium length text that should take a bit longer to synthesize.",
        "This is a very long text that will be used to test the streaming performance \
         under different conditions. It contains multiple sentences and should provide \
         a good benchmark for measuring latency, throughput, and real-time factor \
         across various text lengths and complexities.",
    ];

    for (i, text) in test_texts.iter().enumerate() {
        println!("Test {}: {} characters", i + 1, text.len());

        let metrics = measure_streaming_performance(pipeline, text).await?;

        println!("  Results:");
        println!(
            "    Total processing time: {:.2}s",
            metrics.total_processing_time_s
        );
        println!("    Audio duration: {:.2}s", metrics.audio_duration_s);
        println!("    Real-time factor: {:.2}", metrics.real_time_factor);
        println!(
            "    Throughput: {:.1} chars/sec",
            metrics.throughput_chars_per_sec
        );
        println!("    Chunks generated: {}", metrics.chunk_count);
        println!();
    }

    Ok(())
}

async fn process_audio_chunk(chunk: &[f32], chunk_id: usize) -> Result<()> {
    // Simulate audio processing/playback
    // In a real application, this would:
    // 1. Send audio to output device
    // 2. Apply real-time effects
    // 3. Stream to network clients
    // 4. Save to buffer for later use

    // Simulate processing time (10x faster than real-time)
    let processing_time = Duration::from_micros(chunk.len() as u64 * 45); // ~22050Hz/10
    sleep(processing_time).await;

    // Optional: log first few chunks
    if chunk_id <= 3 {
        tracing::debug!("Processed chunk {}: {} samples", chunk_id, chunk.len());
    }

    Ok(())
}

#[derive(Debug)]
struct StreamingMetrics {
    total_processing_time_s: f64,
    audio_duration_s: f64,
    real_time_factor: f64,
    throughput_chars_per_sec: f64,
    chunk_count: usize,
}

async fn measure_streaming_performance(
    pipeline: &Arc<VoirsPipeline>,
    text: &str,
) -> Result<StreamingMetrics> {
    let start_time = Instant::now();
    let audio = pipeline.synthesize(text).await?;

    let chunk_size = 1024_usize;
    let samples = audio.samples();
    let chunk_count = samples.chunks(chunk_size).count();
    let total_samples = samples.len();

    let total_time = start_time.elapsed();
    let audio_duration = audio.duration() as f64;
    let real_time_factor = if audio_duration > 0.0 {
        total_time.as_secs_f64() / audio_duration
    } else {
        0.0
    };

    let _ = total_samples;

    Ok(StreamingMetrics {
        total_processing_time_s: total_time.as_secs_f64(),
        audio_duration_s: audio_duration,
        real_time_factor,
        throughput_chars_per_sec: text.len() as f64 / total_time.as_secs_f64(),
        chunk_count,
    })
}
