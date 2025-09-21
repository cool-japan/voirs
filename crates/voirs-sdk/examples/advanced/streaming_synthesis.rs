//! # Streaming Synthesis Example
//!
//! This example demonstrates real-time streaming synthesis,
//! which is essential for applications requiring low latency
//! and immediate audio feedback.

use voirs_sdk::prelude::*;
use futures::StreamExt;
use std::time::Instant;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    voirs_sdk::logging::init_logging("info")?;
    
    println!("=== Streaming Synthesis Demo ===\n");
    
    // Create a pipeline optimized for streaming
    let pipeline = VoirsPipelineBuilder::new()
        .with_streaming_mode(true)
        .with_latency_target(100) // 100ms target latency
        .with_chunk_size(1024)    // 1024 samples per chunk
        .build()
        .await?;
    
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

async fn basic_streaming_example(pipeline: &VoirsPipeline) -> Result<(), VoirsError> {
    println!("=== Basic Streaming ===");
    
    let text = "This is a streaming synthesis example. The audio will be generated \
               in real-time as chunks, allowing for immediate playback without \
               waiting for the entire synthesis to complete.";
    
    println!("Text: {}", text);
    println!("Starting streaming synthesis...\n");
    
    let start_time = Instant::now();
    let mut stream = pipeline.synthesize_streaming(text)?;
    let mut chunk_count = 0;
    let mut total_samples = 0;
    
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        chunk_count += 1;
        total_samples += chunk.len();
        
        let elapsed = start_time.elapsed();
        println!("Chunk {}: {} samples at {:.2}s", 
                 chunk_count, chunk.len(), elapsed.as_secs_f64());
        
        // Simulate real-time processing
        // In a real application, you would send this to an audio output device
        process_audio_chunk(&chunk, chunk_count).await?;
    }
    
    let total_time = start_time.elapsed();
    let audio_duration = total_samples as f64 / 22050.0; // Assuming 22050 Hz
    let real_time_factor = total_time.as_secs_f64() / audio_duration;
    
    println!("\nStreaming complete!");
    println!("Total chunks: {}", chunk_count);
    println!("Total samples: {}", total_samples);
    println!("Audio duration: {:.2}s", audio_duration);
    println!("Processing time: {:.2}s", total_time.as_secs_f64());
    println!("Real-time factor: {:.2}", real_time_factor);
    println!();
    
    Ok(())
}

async fn real_time_streaming_example(pipeline: &VoirsPipeline) -> Result<(), VoirsError> {
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
        let mut stream = pipeline.synthesize_streaming(sentence)?;
        let mut audio_buffer = AudioBuffer::new(22050.0, 1);
        
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            audio_buffer.append(&chunk)?;
            
            // Simulate real-time playback
            let playback_duration = Duration::from_millis(
                (chunk.len() as f64 / 22050.0 * 1000.0) as u64
            );
            sleep(playback_duration).await;
        }
        
        let synthesis_time = start.elapsed();
        let audio_duration = audio_buffer.duration();
        
        println!("  Synthesis: {:.2}s, Audio: {:.2}s, RTF: {:.2}", 
                 synthesis_time.as_secs_f64(), 
                 audio_duration,
                 synthesis_time.as_secs_f64() / audio_duration);
        
        // Save each sentence
        let filename = format!("streaming_sentence_{}.wav", i + 1);
        audio_buffer.save_wav(&filename)?;
        
        println!("  Saved: {}\n", filename);
    }
    
    Ok(())
}

async fn interactive_streaming_example(pipeline: &VoirsPipeline) -> Result<(), VoirsError> {
    println!("=== Interactive Streaming ===");
    println!("Simulating interactive conversation...\n");
    
    let conversation = vec![
        ("User", "Hello, how are you today?"),
        ("Assistant", "I'm doing great! Thanks for asking. How can I help you?"),
        ("User", "Can you explain streaming synthesis?"),
        ("Assistant", "Streaming synthesis generates audio in real-time chunks, \
                      enabling immediate playback without waiting for completion."),
        ("User", "That sounds very useful!"),
        ("Assistant", "Absolutely! It's perfect for conversational AI and live applications."),
    ];
    
    for (speaker, text) in conversation {
        println!("{}: {}", speaker, text);
        
        if speaker == "Assistant" {
            // Stream the assistant's response
            let mut stream = pipeline.synthesize_streaming(text)?;
            print!("🔊 ");
            
            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result?;
                print!("♪"); // Visual indicator of audio chunk
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                
                // Simulate audio playback timing
                let chunk_duration = chunk.len() as f64 / 22050.0;
                sleep(Duration::from_millis((chunk_duration * 1000.0) as u64)).await;
            }
            println!(" (synthesis complete)");
        }
        
        println!();
        sleep(Duration::from_millis(500)).await; // Pause between exchanges
    }
    
    Ok(())
}

async fn performance_monitoring_example(pipeline: &VoirsPipeline) -> Result<(), VoirsError> {
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
        println!("    First chunk latency: {:.2}ms", metrics.first_chunk_latency_ms);
        println!("    Average chunk latency: {:.2}ms", metrics.average_chunk_latency_ms);
        println!("    Total processing time: {:.2}s", metrics.total_processing_time_s);
        println!("    Audio duration: {:.2}s", metrics.audio_duration_s);
        println!("    Real-time factor: {:.2}", metrics.real_time_factor);
        println!("    Throughput: {:.1} chars/sec", metrics.throughput_chars_per_sec);
        println!("    Chunks generated: {}", metrics.chunk_count);
        println!();
    }
    
    Ok(())
}

async fn process_audio_chunk(chunk: &AudioBuffer, chunk_id: usize) -> Result<(), VoirsError> {
    // Simulate audio processing/playback
    // In a real application, this would:
    // 1. Send audio to output device
    // 2. Apply real-time effects
    // 3. Stream to network clients
    // 4. Save to buffer for later use
    
    // Simulate processing time
    let processing_time = Duration::from_millis(chunk.len() as u64 / 100);
    sleep(processing_time).await;
    
    // Optional: Save chunk for debugging
    if chunk_id <= 3 { // Save first few chunks
        let filename = format!("chunk_{:03}.wav", chunk_id);
        chunk.save_wav(&filename)?;
    }
    
    Ok(())
}

#[derive(Debug)]
struct StreamingMetrics {
    first_chunk_latency_ms: f64,
    average_chunk_latency_ms: f64,
    total_processing_time_s: f64,
    audio_duration_s: f64,
    real_time_factor: f64,
    throughput_chars_per_sec: f64,
    chunk_count: usize,
}

async fn measure_streaming_performance(
    pipeline: &VoirsPipeline, 
    text: &str
) -> Result<StreamingMetrics, VoirsError> {
    let start_time = Instant::now();
    let mut stream = pipeline.synthesize_streaming(text)?;
    
    let mut first_chunk_time = None;
    let mut chunk_times = Vec::new();
    let mut total_samples = 0;
    let mut chunk_count = 0;
    
    while let Some(chunk_result) = stream.next().await {
        let chunk_time = start_time.elapsed();
        
        if first_chunk_time.is_none() {
            first_chunk_time = Some(chunk_time);
        }
        
        chunk_times.push(chunk_time.as_millis() as f64);
        
        let chunk = chunk_result?;
        total_samples += chunk.len();
        chunk_count += 1;
    }
    
    let total_time = start_time.elapsed();
    let audio_duration = total_samples as f64 / 22050.0; // Assuming 22050 Hz
    
    let first_chunk_latency_ms = first_chunk_time
        .map(|t| t.as_millis() as f64)
        .unwrap_or(0.0);
    
    let average_chunk_latency_ms = if !chunk_times.is_empty() {
        chunk_times.iter().sum::<f64>() / chunk_times.len() as f64
    } else {
        0.0
    };
    
    Ok(StreamingMetrics {
        first_chunk_latency_ms,
        average_chunk_latency_ms,
        total_processing_time_s: total_time.as_secs_f64(),
        audio_duration_s: audio_duration,
        real_time_factor: total_time.as_secs_f64() / audio_duration,
        throughput_chars_per_sec: text.len() as f64 / total_time.as_secs_f64(),
        chunk_count,
    })
}

/// Example of concurrent streaming
#[tokio::main]
async fn concurrent_streaming_example() -> Result<(), VoirsError> {
    let pipeline = VoirsPipelineBuilder::new()
        .with_concurrent_limit(4)
        .build()
        .await?;
    
    let texts = vec![
        "First concurrent stream.",
        "Second concurrent stream.",
        "Third concurrent stream.",
        "Fourth concurrent stream.",
    ];
    
    let futures: Vec<_> = texts.into_iter().enumerate().map(|(i, text)| {
        let pipeline = &pipeline;
        async move {
            let mut stream = pipeline.synthesize_streaming(text)?;
            let mut total_samples = 0;
            
            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result?;
                total_samples += chunk.len();
            }
            
            println!("Stream {} completed: {} samples", i + 1, total_samples);
            Ok::<_, VoirsError>(total_samples)
        }
    }).collect();
    
    let results = futures::future::try_join_all(futures).await?;
    println!("All concurrent streams completed: {:?} samples", results);
    
    Ok(())
}