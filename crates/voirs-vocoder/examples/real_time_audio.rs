//! Real-time audio output example with ML enhancement
//!
//! This example demonstrates how to use the real-time audio drivers
//! with machine learning enhancement to output high-quality vocoded audio
//! directly to the system audio device.

use std::time::Duration;
use tokio::time::sleep;
use voirs_vocoder::{
    drivers::{AudioDriverFactory, AudioStreamConfig, RealTimeAudioOutput},
    ml::{MLEnhancementConfig, MLEnhancerFactory, ProcessingMode, QualityLevel},
    DummyVocoder, MelSpectrogram, Vocoder,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize simple logging (tracing_subscriber not included in dependencies)
    // tracing_subscriber::init();

    println!("VoiRS Real-time Audio Output Example with ML Enhancement");
    println!("=========================================================");

    // Initialize ML enhancer
    println!("Initializing ML audio enhancer...");
    let mut ml_enhancer = match MLEnhancerFactory::create_neural_enhancer() {
        Ok(enhancer) => Some(enhancer),
        Err(e) => {
            println!("Warning: Failed to create ML enhancer: {e}");
            println!("Continuing without ML enhancement...");
            None
        }
    };

    // Configure ML enhancement for real-time processing
    let ml_config = MLEnhancementConfig {
        strength: 0.6,
        target_quality: QualityLevel::High,
        preserve_dynamics: true,
        mode: ProcessingMode::Speed, // Prioritize speed for real-time
        model_params: std::collections::HashMap::new(),
    };

    // Check available audio drivers
    let available_drivers = AudioDriverFactory::available_drivers();
    println!("Available audio drivers: {available_drivers:?}");

    if available_drivers.is_empty() {
        println!("No audio drivers available on this platform");
        return Ok(());
    }

    // Create audio stream configuration
    let audio_config = AudioStreamConfig {
        sample_rate: 22050,
        channels: 1,
        buffer_size: 256,
        target_latency_ms: 10.0,
    };

    println!("Audio configuration: {audio_config:?}");

    // Initialize ML enhancer with audio sample rate
    if let Some(ref mut _enhancer) = ml_enhancer {
        // This is a bit tricky since we need to downcast to the specific type
        // For now, we'll try to initialize with a reasonable default
        println!(
            "Initializing neural enhancer with sample rate: {} Hz",
            audio_config.sample_rate
        );
        // Note: In a real implementation, we'd need a better way to initialize
        // the enhancer. For now, we'll assume it's initialized internally.
    }

    // Create real-time audio output
    let mut audio_output = match RealTimeAudioOutput::new(audio_config) {
        Ok(output) => output,
        Err(e) => {
            println!("Failed to create audio output: {e}");
            return Ok(());
        }
    };

    // Start the audio stream
    match audio_output.start().await {
        Ok(()) => println!("Audio stream started successfully"),
        Err(e) => {
            println!("Failed to start audio stream: {e}");
            return Ok(());
        }
    }

    // Create a dummy vocoder for generating test audio
    let vocoder = DummyVocoder::new();

    // Create test mel spectrograms and convert to audio
    println!("\nGenerating and playing audio...");

    for i in 0..5 {
        // Create a mel spectrogram (dummy data)
        let mel_data = vec![vec![0.5 + (i as f32 * 0.1); 50]; 80]; // 80 mel channels, 50 frames
        let mel = MelSpectrogram::new(mel_data, 22050, 256);

        // Convert to audio using vocoder
        match vocoder.vocode(&mel, None).await {
            Ok(mut audio_buffer) => {
                println!(
                    "Generated audio chunk {}: {:.2}s duration",
                    i + 1,
                    audio_buffer.duration()
                );

                // Apply ML enhancement if available
                if let Some(ref enhancer) = ml_enhancer {
                    match enhancer.enhance(&audio_buffer, &ml_config).await {
                        Ok(enhanced_audio) => {
                            audio_buffer = enhanced_audio;
                            println!("  ✨ Applied ML enhancement");
                        }
                        Err(e) => {
                            println!("  ⚠️  ML enhancement failed: {e}");
                            // Continue with original audio
                        }
                    }
                }

                // Queue the audio for playback
                if let Err(e) = audio_output.queue_audio(audio_buffer) {
                    println!("Failed to queue audio: {e}");
                }
            }
            Err(e) => {
                println!("Failed to generate audio: {e}");
            }
        }

        // Wait a bit between chunks
        sleep(Duration::from_millis(200)).await;
    }

    // Let the audio play for a bit
    println!("\nLetting audio play...");
    sleep(Duration::from_secs(2)).await;

    // Show audio metrics
    let metrics = audio_output.metrics();
    println!("\nAudio metrics:");
    println!("  Frames processed: {}", metrics.frames_processed);
    println!("  Buffer underruns: {}", metrics.underruns);
    println!("  Buffer overruns: {}", metrics.overruns);
    println!("  Current latency: {:.2}ms", metrics.current_latency_ms);
    println!("  CPU load: {:.1}%", metrics.cpu_load_percent);
    println!("  Is active: {}", metrics.is_active);

    // Stop the audio stream
    match audio_output.stop().await {
        Ok(()) => println!("\nAudio stream stopped successfully"),
        Err(e) => println!("Failed to stop audio stream: {e}"),
    }

    Ok(())
}
