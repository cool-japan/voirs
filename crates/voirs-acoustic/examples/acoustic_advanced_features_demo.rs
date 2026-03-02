//! Advanced Features Demonstration
//!
//! This example demonstrates the new advanced features in voirs-acoustic:
//! - Neural Audio Codec for compression
//! - Advanced Latency Optimizer for real-time performance
//! - Voice Activity Detection for speech segmentation

use voirs_acoustic::{
    AdvancedLatencyOptimizer, CodecType, LatencyBudget, NeuralCodec, NeuralCodecConfig,
    ProcessingPriority, VadConfig, VoiceActivityDetector,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎙️  VoiRS Advanced Features Demo\n");

    // 1. Neural Audio Codec Demo
    demo_neural_codec()?;

    // 2. Latency Optimizer Demo
    demo_latency_optimizer().await?;

    // 3. Voice Activity Detection Demo
    demo_vad()?;

    // 4. Integration Demo - Using all features together
    demo_integration().await?;

    println!("\n✅ All demos completed successfully!");
    Ok(())
}

/// Demonstrate Neural Audio Codec capabilities
fn demo_neural_codec() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════");
    println!("1️⃣  Neural Audio Codec Demo");
    println!("═══════════════════════════════════════════════\n");

    // Create codec with different quality presets
    let configs = vec![
        ("High Quality", NeuralCodecConfig::high_quality()),
        ("Low Latency", NeuralCodecConfig::low_latency()),
        ("Low Bandwidth", NeuralCodecConfig::low_bandwidth()),
    ];

    for (name, config) in configs {
        println!("📊 {} Codec:", name);
        println!("   Codec Type: {:?}", config.codec_type);
        println!("   Target Bitrate: {:.1} kbps", config.target_bitrate);
        println!("   Codebooks: {}", config.num_codebooks);
        println!("   Codebook Size: {}", config.codebook_size);
        println!("   Bits per Frame: {:.1}", config.bits_per_frame());
        println!(
            "   Expected Latency: {:.1} ms",
            config.expected_latency_ms()
        );

        // Estimate compression ratio
        let sample_rate = 22050;
        let duration_sec = 1.0;
        let samples = (sample_rate as f32 * duration_sec) as usize;

        #[cfg(feature = "candle")]
        {
            use candle_core::Device;
            let device = Device::Cpu;
            let codec = NeuralCodec::new(config.clone(), &device)?;

            let compression = codec.compression_ratio(samples);
            let bitrate = codec.estimate_bitrate(sample_rate);

            println!("   Compression Ratio: {:.1}x", compression);
            println!("   Estimated Bitrate: {:.2} kbps", bitrate);
        }

        println!();
    }

    println!("✅ Codec demo complete\n");
    Ok(())
}

/// Demonstrate Latency Optimizer capabilities
async fn demo_latency_optimizer() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════");
    println!("2️⃣  Advanced Latency Optimizer Demo");
    println!("═══════════════════════════════════════════════\n");

    // Create optimizers for different scenarios
    let scenarios = vec![
        ("Conversational", AdvancedLatencyOptimizer::conversational()),
        (
            "Interactive Gaming",
            AdvancedLatencyOptimizer::interactive(),
        ),
        ("Broadcast Quality", AdvancedLatencyOptimizer::broadcast()),
    ];

    for (name, optimizer) in scenarios {
        println!("⚡ {} Scenario:", name);

        // Simulate some processing operations
        for i in 0..5 {
            let priority = if i == 0 {
                ProcessingPriority::High
            } else {
                ProcessingPriority::Normal
            };

            let mut measurement = optimizer.start_measurement(priority);

            // Simulate processing (in real use, this would be actual synthesis)
            tokio::time::sleep(tokio::time::Duration::from_millis(10 + i * 5)).await;

            measurement.finish(100, true);
            optimizer.record_measurement(measurement).await;
        }

        // Get statistics
        let stats = optimizer.get_statistics().await;
        println!("   Total Measurements: {}", stats.total_measurements);
        println!("   Average Latency: {:.2} ms", stats.avg_latency_ms);
        println!("   P95 Latency: {:.2} ms", stats.p95_latency_ms);
        println!("   Budget Met Rate: {:.1}%", stats.budget_met_rate * 100.0);
        println!("   Current Chunk Size: {}", stats.current_chunk_size);

        let quality = optimizer.recommended_quality().await;
        println!("   Recommended Quality: {:.1}%", quality * 100.0);

        println!();
    }

    println!("✅ Latency optimizer demo complete\n");
    Ok(())
}

/// Demonstrate Voice Activity Detection capabilities
fn demo_vad() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════");
    println!("3️⃣  Voice Activity Detection Demo");
    println!("═══════════════════════════════════════════════\n");

    // Create VAD with different presets
    let configs = vec![
        ("Conversational", VadConfig::conversational()),
        ("Studio Quality", VadConfig::studio()),
        ("Noisy Environment", VadConfig::noisy()),
    ];

    for (name, config) in configs {
        println!("🎤 {} VAD:", name);
        println!("   Energy Threshold: {:.1} dB", config.energy_threshold_db);
        println!(
            "   Min Speech Duration: {:.0} ms",
            config.min_speech_duration_ms
        );
        println!(
            "   Min Silence Duration: {:.0} ms",
            config.min_silence_duration_ms
        );
        println!("   Adaptive Threshold: {}", config.adaptive_threshold);
        println!("   Frame Rate: {:.1} Hz", config.frames_per_second());

        let mut vad = VoiceActivityDetector::new(config.clone())?;

        // Simulate audio with speech and silence
        let mut audio = Vec::new();

        // Speech segment (0.5 seconds)
        for i in 0..11025 {
            audio.push((i as f32 * 0.02).sin() * 0.3);
        }

        // Silence segment (0.3 seconds)
        audio.resize(audio.len() + 6615, 0.001);

        // Speech segment (0.5 seconds)
        for i in 0..11025 {
            audio.push((i as f32 * 0.02).sin() * 0.3);
        }

        let segments = vad.process_buffer(&audio);

        println!("   Detected {} segments:", segments.len());
        for (i, segment) in segments.iter().enumerate() {
            println!(
                "     Segment {}: {:?} ({:.2}s - {:.2}s, {:.0}ms)",
                i + 1,
                segment.activity,
                segment.start_time,
                segment.end_time,
                segment.duration_ms()
            );
        }

        println!();
    }

    println!("✅ VAD demo complete\n");
    Ok(())
}

/// Demonstrate integration of all advanced features
async fn demo_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════");
    println!("4️⃣  Integrated Features Demo");
    println!("═══════════════════════════════════════════════\n");

    println!("🔗 Real-Time TTS Pipeline with Advanced Features:\n");

    // 1. Setup latency optimizer
    let latency_optimizer = AdvancedLatencyOptimizer::interactive();
    println!("✓ Latency optimizer configured for interactive use");

    // 2. Setup VAD for silence detection
    let vad_config = VadConfig::conversational();
    let _vad = VoiceActivityDetector::new(vad_config)?;
    println!("✓ VAD configured for conversational speech");

    // 3. Setup neural codec for bandwidth optimization
    #[cfg(feature = "candle")]
    {
        use candle_core::Device;
        let codec_config = NeuralCodecConfig::low_latency();
        let device = Device::Cpu;
        let _codec = NeuralCodec::new(codec_config, &device)?;
        println!("✓ Neural codec configured for low latency");
    }

    println!("\n📈 Simulating Real-Time Processing:\n");

    // Simulate processing pipeline
    for chunk_num in 0..3 {
        let chunk_size = latency_optimizer.recommended_chunk_size().await;
        let quality = latency_optimizer.recommended_quality().await;

        println!(
            "   Chunk {}: size={}, quality={:.1}%",
            chunk_num + 1,
            chunk_size,
            quality * 100.0
        );

        // Start latency measurement
        let mut measurement = latency_optimizer.start_measurement(ProcessingPriority::High);

        // Simulate processing
        tokio::time::sleep(tokio::time::Duration::from_millis(15)).await;

        // Finish measurement
        measurement.finish(chunk_size, true);
        latency_optimizer.record_measurement(measurement).await;
    }

    // Get final statistics
    let stats = latency_optimizer.get_statistics().await;
    println!("\n📊 Pipeline Statistics:");
    println!("{}", stats.report());

    println!("\n✅ Integration demo complete\n");
    Ok(())
}
