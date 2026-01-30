//! Real-time streaming TTS synthesis command implementation.
//!
//! Provides low-latency, chunk-based synthesis for real-time applications.

use crate::audio::realtime::{BufferConfig, RealTimeAudioStream, RealTimeStreamConfig};
use crate::GlobalOptions;
use futures_util::StreamExt;
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use voirs_sdk::config::AppConfig;
use voirs_sdk::pipeline::VoirsPipelineBuilder;
use voirs_sdk::streaming::StreamingConfig;
use voirs_sdk::Result;

/// Run streaming synthesis with configurable latency and chunking.
pub async fn run_streaming_synthesis(
    initial_text: Option<&str>,
    target_latency_ms: u64,
    chunk_size: usize,
    buffer_chunks: usize,
    enable_playback: bool,
    _config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("🎙️  VoiRS Streaming Synthesis Mode");
        println!("Target latency: {}ms", target_latency_ms);
        println!("Chunk size: {} frames", chunk_size);
        println!("Buffer: {} chunks", buffer_chunks);
        println!(
            "Audio playback: {}",
            if enable_playback {
                "enabled"
            } else {
                "disabled"
            }
        );
        println!("\nType text to synthesize (Ctrl+C to exit):");
        println!();
    }

    // Build VoiRS pipeline for streaming synthesis
    let pipeline = Arc::new(
        VoirsPipelineBuilder::new()
            .with_voice("default")
            .build()
            .await?,
    );

    // Configure streaming settings
    let _streaming_config = StreamingConfig::low_latency();
    // Note: streaming_config parameters would be used if the API supported custom configuration
    // Current implementation uses default streaming settings from the pipeline
    let _target_latency_ms = target_latency_ms;
    let _chunk_size = chunk_size;

    // Create streaming synthesis pipeline
    let (tx, mut rx) = mpsc::channel::<String>(buffer_chunks);

    // Audio output channel
    let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<f32>>(buffer_chunks);

    // Clone pipeline for synthesis task
    let pipeline_for_task = pipeline.clone();

    // Synthesis task using real VoiRS streaming
    let synthesis_handle = tokio::spawn(async move {
        while let Some(text) = rx.recv().await {
            let start = Instant::now();

            // Clone Arc for each synthesis call (synthesize_stream consumes Arc)
            let pipeline_clone = pipeline_for_task.clone();

            // Use real streaming synthesis from VoiRS SDK
            match pipeline_clone.synthesize_stream(&text).await {
                Ok(mut stream) => {
                    while let Some(chunk_result) = stream.next().await {
                        match chunk_result {
                            Ok(audio_chunk) => {
                                // Extract audio samples from AudioBuffer
                                let samples = audio_chunk.samples().to_vec();

                                if let Err(e) = audio_tx.send(samples).await {
                                    tracing::error!("Failed to send audio chunk: {}", e);
                                    break;
                                }
                            }
                            Err(e) => {
                                tracing::error!("Synthesis chunk error: {}", e);
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to start streaming synthesis: {}", e);
                }
            }

            let elapsed = start.elapsed();
            tracing::debug!("Synthesized '{}' in {:?}", text, elapsed);
        }
    });

    // Audio playback task (if enabled)
    let playback_handle = if enable_playback {
        // Create audio stream configuration
        let stream_config = RealTimeStreamConfig {
            sample_rate: 22050, // Default sample rate for TTS
            channels: 1,        // Mono audio
            buffer_size: chunk_size as u32,
            target_latency_ms: target_latency_ms as u32,
            device_name: None, // Use default device
        };

        let buffer_config = BufferConfig {
            buffer_count: buffer_chunks,
            buffer_size: chunk_size,
            underrun_threshold: 2,
        };

        if !global.quiet {
            println!("🔊 Initializing audio playback...");
        }

        // Spawn audio playback in a dedicated thread since cpal::Stream is not Send
        Some(std::thread::spawn(move || {
            // Create runtime for async operations within the audio thread (for channel recv)
            let rt = tokio::runtime::Runtime::new().expect("Failed to create audio runtime");

            rt.block_on(async {
                // Create and start audio stream
                match RealTimeAudioStream::new(stream_config, buffer_config) {
                    Ok(mut audio_stream) => {
                        if let Err(e) = audio_stream.start().await {
                            tracing::error!("Failed to start audio stream: {}", e);
                            eprintln!("❌ Failed to start audio playback: {}", e);
                            return;
                        }

                        tracing::info!("Audio playback stream started successfully");

                        // Process audio chunks
                        while let Some(audio_chunk) = audio_rx.recv().await {
                            tracing::debug!("Playing audio chunk of {} samples", audio_chunk.len());

                            // Write audio samples to the stream
                            if let Err(e) = audio_stream.write_samples(&audio_chunk) {
                                tracing::error!("Failed to write audio samples: {}", e);
                                break;
                            }
                        }

                        // Stop playback
                        if let Err(e) = audio_stream.stop() {
                            tracing::warn!("Error stopping audio stream: {}", e);
                        }

                        tracing::info!("Audio playback completed");
                    }
                    Err(e) => {
                        tracing::error!("Failed to create audio stream: {}", e);
                        eprintln!("❌ Audio playback initialization failed: {}", e);
                        eprintln!("💡 Tip: Ensure your audio device is connected and accessible");
                    }
                }
            });
        }))
    } else {
        None
    };

    // Process initial text if provided
    if let Some(text) = initial_text {
        if !global.quiet {
            println!("Synthesizing: {}", text);
        }
        tx.send(text.to_string()).await.map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!("Failed to send text: {}", e))
        })?;
    }

    // Interactive text input loop
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    let mut buffer = String::new();

    while handle.read_line(&mut buffer).is_ok() {
        let text = buffer.trim();

        if text.is_empty() {
            buffer.clear();
            continue;
        }

        if !global.quiet {
            println!("Synthesizing: {}", text);
        }

        // Send text for synthesis
        if tx.send(text.to_string()).await.is_err() {
            break;
        }

        buffer.clear();
    }

    // Cleanup
    drop(tx); // Close sender to stop synthesis task

    if let Err(e) = synthesis_handle.await {
        tracing::error!("Synthesis task error: {}", e);
        if !global.quiet {
            eprintln!("❌ Synthesis task encountered an error: {}", e);
            eprintln!("💡 Tip: Check your voice models and network connectivity");
        }
    }

    if let Some(handle) = playback_handle {
        if let Err(e) = handle.join() {
            tracing::error!("Playback thread panicked: {:?}", e);
            if !global.quiet {
                eprintln!("❌ Audio playback thread encountered an error");
                eprintln!("💡 Tip: Check your audio device configuration and permissions");
            }
        }
    }

    if !global.quiet {
        println!("\n✅ Streaming synthesis session completed");
    }

    Ok(())
}
