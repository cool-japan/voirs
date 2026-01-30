//! # Real-time Streaming Voice Conversion Example
//!
//! This example demonstrates real-time streaming voice conversion with
//! stream-based processing, adaptive quality control, and low-latency optimization.
//!
//! ## Features Demonstrated
//! - Real-time streaming conversion
//! - Stream-based audio processing
//! - Adaptive quality control
//! - Backpressure handling
//! - Latency monitoring
//! - Stream state management

use futures::stream;
use std::time::Duration;
use voirs_conversion::{
    streaming::{StreamConfig, StreamState, StreamingConverter},
    types::{
        AgeGroup, ConversionTarget, ConversionType, Gender, PitchCharacteristics,
        QualityCharacteristics, SpectralCharacteristics, TimingCharacteristics,
        VoiceCharacteristics,
    },
    Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Real-time Streaming Voice Conversion Example ===\n");

    // Step 1: Configure streaming converter
    println!("1. Configuring real-time streaming converter...");
    let stream_config = StreamConfig {
        chunk_size: 1024,            // Samples per chunk (64ms at 16kHz)
        sample_rate: 16000,          // 16kHz sample rate
        target_latency_ms: 100.0,    // Target 100ms latency
        buffer_capacity: 8192,       // Buffer capacity for accumulation
        channel_buffer_size: 32,     // Async channel buffer size
        max_concurrent_streams: 8,   // Support up to 8 concurrent streams
        adaptive_buffering: true,    // Enable adaptive buffering
        quality_vs_latency: 0.7,     // Favor quality slightly over latency
        enable_error_recovery: true, // Enable error recovery
        stream_timeout_secs: 30,     // 30 second timeout
    };

    let mut converter = StreamingConverter::new(stream_config.clone())?;
    println!("   Streaming converter initialized");
    println!(
        "   - Chunk size: {} samples ({:.1}ms)",
        stream_config.chunk_size,
        stream_config.chunk_size as f32 / stream_config.sample_rate as f32 * 1000.0
    );
    println!(
        "   - Target latency: {:.1}ms",
        stream_config.target_latency_ms
    );
    println!(
        "   - Max concurrent streams: {}",
        stream_config.max_concurrent_streams
    );

    // Step 2: Define conversion target
    println!("\n2. Defining conversion target...");
    let characteristics = VoiceCharacteristics {
        pitch: PitchCharacteristics {
            mean_f0: 220.0,
            range: 50.0,
            jitter: 0.02,
            stability: 0.9,
        },
        timing: TimingCharacteristics {
            speaking_rate: 1.0,
            pause_duration: 0.3,
            rhythm_regularity: 0.8,
        },
        spectral: SpectralCharacteristics {
            formant_shift: 1.1,
            brightness: 0.2,
            spectral_tilt: -6.0,
            harmonicity: 0.8,
        },
        quality: QualityCharacteristics {
            breathiness: 0.2,
            roughness: 0.1,
            stability: 0.9,
            resonance: 0.85,
        },
        age_group: Some(AgeGroup::YoungAdult),
        gender: Some(Gender::Female),
        accent: None,
        custom_params: Default::default(),
    };

    let target = ConversionTarget::new(characteristics);
    converter.set_conversion_target(target);

    println!("   Target: Female voice, 220Hz mean pitch");

    // Step 3: Start streaming session
    println!("\n3. Starting streaming session...");
    converter.start().await?;
    let state = converter.state();
    println!("   Stream state: {:?}", state);

    // Step 4: Create audio stream for processing
    println!("\n4. Processing audio stream in real-time...");

    // Create a stream of audio chunks (simulating microphone input)
    let chunk_count = 20; // Process 20 chunks (~1.28 seconds at 16kHz)
    let chunk_size = stream_config.chunk_size;
    let audio_stream = stream::iter(
        (0..chunk_count).map(move |i| generate_audio_chunk(180.0 + (i as f32 * 2.0), chunk_size)),
    );

    // Process stream with backpressure handling
    let start_time = std::time::Instant::now();
    let mut converted_stream = converter
        .process_stream_with_backpressure(Box::pin(audio_stream))
        .await?;

    // Consume the converted stream
    let mut chunks_received = 0;
    use futures::StreamExt;
    while let Some(converted_chunk) = converted_stream.next().await {
        chunks_received += 1;
        if chunks_received % 5 == 0 {
            println!("   Processed chunk {}/{}", chunks_received, chunk_count);
        }
    }

    let total_time = start_time.elapsed();
    println!("   Total processing time: {:?}", total_time);
    println!(
        "   Average latency per chunk: {:.2}ms",
        total_time.as_millis() as f32 / chunk_count as f32
    );

    // Step 5: Get streaming statistics
    println!("\n5. Streaming Statistics:");
    let stats = converter.get_stats().await;
    println!(
        "   - Total chunks processed: {}",
        stats.total_chunks_processed
    );
    println!("   - Total samples: {}", stats.total_samples);
    println!(
        "   - Total processing time: {:.2}ms",
        stats.total_processing_time_ms
    );
    println!(
        "   - Average chunk latency: {:.2}ms",
        stats.average_chunk_latency_ms()
    );
    println!(
        "   - Max chunk latency: {:.2}ms",
        stats.max_chunk_latency_ms
    );
    println!(
        "   - Min chunk latency: {:.2}ms",
        stats.min_chunk_latency_ms
    );
    println!("   - Total errors: {}", stats.total_errors);
    println!("   - Error rate: {:.2}%", stats.error_rate() * 100.0);
    println!(
        "   - Throughput: {:.0} samples/sec",
        stats.throughput_samples_per_sec()
    );

    // Step 6: Test pause and resume
    println!("\n6. Testing pause and resume...");
    converter.pause().await?;
    println!("   Stream paused");
    println!("   State: {:?}", converter.state());

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Resume by starting again
    converter.start().await?;
    println!("   Stream resumed");
    println!("   State: {:?}", converter.state());

    // Process another batch
    let chunk_size = stream_config.chunk_size;
    let second_batch = stream::iter(
        (0..5).map(move |i| generate_audio_chunk(200.0 + (i as f32 * 3.0), chunk_size)),
    );

    let mut resumed_stream = converter.process_stream(Box::pin(second_batch)).await?;

    let mut resumed_chunks = 0;
    while (resumed_stream.next().await).is_some() {
        resumed_chunks += 1;
    }
    println!("   Processed {} additional chunks", resumed_chunks);

    // Step 7: Check system health
    println!("\n7. System Health Check:");
    let is_healthy = converter.is_healthy().await;
    println!("   System healthy: {}", is_healthy);

    // Step 8: Final statistics after all processing
    println!("\n8. Final Statistics:");
    let final_stats = converter.get_stats().await;
    println!(
        "   - Total chunks processed: {}",
        final_stats.total_chunks_processed
    );
    println!("   - Total samples: {}", final_stats.total_samples);
    println!(
        "   - Average latency: {:.2}ms",
        final_stats.average_chunk_latency_ms()
    );
    println!(
        "   - Processing efficiency: {:.2}ms/chunk",
        final_stats.total_processing_time_ms / final_stats.total_chunks_processed as f64
    );

    // Step 9: Graceful shutdown
    println!("\n9. Graceful shutdown...");
    converter.stop().await?;
    println!("   Stream stopped");
    println!("   Final state: {:?}", converter.state());

    println!("\n=== Real-time Streaming Example Complete ===");
    Ok(())
}

/// Generate sample audio chunk with specified fundamental frequency
fn generate_audio_chunk(f0: f32, size: usize) -> Vec<f32> {
    use std::f32::consts::PI;

    (0..size)
        .map(|i| {
            let t = i as f32 / 16000.0; // 16kHz sample rate
            let fundamental = (2.0 * PI * f0 * t).sin() * 0.5;
            let harmonic2 = (2.0 * PI * f0 * 2.0 * t).sin() * 0.25;
            let harmonic3 = (2.0 * PI * f0 * 3.0 * t).sin() * 0.125;
            fundamental + harmonic2 + harmonic3
        })
        .collect()
}
