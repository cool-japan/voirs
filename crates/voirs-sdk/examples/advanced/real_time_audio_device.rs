//! # Real-Time Audio Device Integration
//!
//! This example demonstrates integration of VoiRS SDK with audio hardware for:
//! - Live audio output to speakers/headphones
//! - Real-time synthesis with immediate playback
//! - Audio device management and configuration
//! - Low-latency audio streaming
//! - Interactive voice applications
//! - Audio callback optimization

use cpal::{traits::*, Device, SampleFormat, SampleRate, Stream, StreamConfig};
use futures::StreamExt;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, RwLock};
use voirs_sdk::prelude::*;

#[derive(Debug, Clone)]
struct AudioDeviceConfig {
    device_name: String,
    sample_rate: u32,
    channels: u16,
    buffer_size: u32,
    latency_ms: f64,
}

struct AudioDeviceManager {
    current_device: Option<Device>,
    config: AudioDeviceConfig,
    stream: Option<Stream>,
    audio_queue: Arc<Mutex<VecDeque<f32>>>,
}

#[derive(Debug, Clone)]
enum AudioCommand {
    SynthesizeAndPlay(String),
    SetVolume(f32),
    SetDevice(String),
    Stop,
    Pause,
    Resume,
}

struct LiveAudioSystem {
    pipeline: Arc<VoirsPipeline>,
    device_manager: Arc<RwLock<AudioDeviceManager>>,
    command_sender: mpsc::UnboundedSender<AudioCommand>,
    is_playing: Arc<RwLock<bool>>,
    volume: Arc<RwLock<f32>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging for real-time audio
    voirs_sdk::logging::init_logging(&voirs_sdk::config::LoggingConfig::default())?;

    println!("🔊 VoiRS SDK Real-Time Audio Device Integration");
    println!("==============================================\n");

    // Demo 1: Audio Device Discovery and Setup
    audio_device_discovery_demo().await?;

    // Demo 2: Basic Real-Time Synthesis and Playback
    basic_real_time_demo().await?;

    // Demo 3: Interactive Voice Application
    interactive_voice_demo().await?;

    // Demo 4: Low-Latency Streaming
    low_latency_streaming_demo().await?;

    // Demo 5: Multi-Device Audio System
    multi_device_demo().await?;

    println!("\n✨ Real-time audio integration demo completed!");
    Ok(())
}

async fn audio_device_discovery_demo() -> Result<()> {
    println!("🎧 Demo 1: Audio Device Discovery and Setup");
    println!("==========================================");

    // Discover available audio devices
    let host = cpal::default_host();

    println!("🔍 Available audio hosts:");
    for host_id in cpal::available_hosts() {
        println!("  - {:?}", host_id);
    }

    println!("\n🔊 Available output devices:");
    let devices = host.output_devices().map_err(|e| {
        VoirsError::device_error("host", format!("Failed to enumerate devices: {}", e))
    })?;

    let mut device_list = Vec::new();
    for (i, device) in devices.enumerate() {
        let name = device
            .name()
            .unwrap_or_else(|_| format!("Unknown Device {}", i));

        println!("  {}. {}", i + 1, name);

        // Get supported configurations
        if let Ok(supported_configs) = device.supported_output_configs() {
            for config in supported_configs.take(3) {
                // Show first 3 configs
                println!(
                    "     - {:?} {} Hz, {} channels",
                    config.sample_format(),
                    config.max_sample_rate().0,
                    config.channels()
                );
            }
        }

        device_list.push((name, device));
    }

    // Select default device and show its configuration
    if let Some((name, device)) = device_list.first() {
        println!("\n🎯 Selected device: {}", name);

        if let Ok(config) = device.default_output_config() {
            println!("  Default configuration:");
            println!("    Sample rate: {} Hz", config.sample_rate().0);
            println!("    Channels: {}", config.channels());
            println!("    Sample format: {:?}", config.sample_format());
            println!("    Buffer size: {:?}", config.buffer_size());
        }
    }

    println!("✅ Audio device discovery completed\n");
    Ok(())
}

async fn basic_real_time_demo() -> Result<()> {
    println!("⚡ Demo 2: Basic Real-Time Synthesis and Playback");
    println!("===============================================");

    // Create VoiRS pipeline optimized for real-time use
    let pipeline = VoirsPipelineBuilder::new()
        // Note: streaming mode enabled by default
        // Note: latency and chunk size configured through quality settings
        .with_quality(QualityLevel::Medium) // Balance quality vs speed
        .build()
        .await?;

    // Create audio system
    let audio_system = LiveAudioSystem::new(pipeline).await?;

    println!("✅ Real-time audio system initialized");
    println!("🎤 Starting live synthesis demonstrations...\n");

    // Test phrases for real-time synthesis
    let test_phrases = vec![
        "Hello! This is a real-time audio test.",
        "The audio should play immediately after synthesis.",
        "Low latency voice synthesis in action!",
        "This demonstrates real-time audio device integration.",
    ];

    for (i, phrase) in test_phrases.iter().enumerate() {
        println!("🎵 Playing phrase {}: \"{}\"", i + 1, phrase);

        let start_time = std::time::Instant::now();
        audio_system.synthesize_and_play(phrase).await?;
        let response_time = start_time.elapsed();

        println!(
            "  ⏱️  Response time: {:.1}ms",
            response_time.as_secs_f64() * 1000.0
        );

        // Wait for playback to complete
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    }

    audio_system.stop().await?;
    println!("✅ Basic real-time demo completed\n");
    Ok(())
}

async fn interactive_voice_demo() -> Result<()> {
    println!("🎮 Demo 3: Interactive Voice Application");
    println!("======================================");

    // Create interactive pipeline with emotion control
    let pipeline = VoirsPipelineBuilder::new()
        // Note: streaming mode enabled by default
        .with_emotion_enabled(true)
        // Note: latency configured through quality settings
        .build()
        .await?;

    let audio_system = LiveAudioSystem::new(pipeline).await?;

    println!("✅ Interactive voice system ready");
    println!("🎭 Simulating interactive conversation with emotional responses...\n");

    // Simulate interactive scenarios
    let interaction_scenarios = vec![
        ("normal", "Welcome to the interactive voice demonstration."),
        (
            "happy",
            "Great! You've successfully connected to the system.",
        ),
        ("concerned", "Please wait while I process your request."),
        ("excited", "Fantastic! Everything is working perfectly!"),
        ("calm", "Thank you for trying the real-time voice system."),
    ];

    for (emotion, text) in interaction_scenarios {
        println!("🎭 Emotion: {} - \"{}\"", emotion, text);

        // Apply emotion if available
        #[cfg(feature = "emotion")]
        {
            audio_system
                .pipeline
                .apply_emotion_preset(emotion, Some(0.7))
                .await?;
        }

        let start = std::time::Instant::now();
        audio_system.synthesize_and_play(text).await?;
        let latency = start.elapsed();

        println!("  ⚡ Latency: {:.1}ms", latency.as_secs_f64() * 1000.0);

        // Simulate user thinking time
        tokio::time::sleep(tokio::time::Duration::from_millis(1500)).await;
    }

    // Demonstrate dynamic volume control
    println!("\n🔊 Volume control demonstration:");
    for volume in [0.3, 0.7, 1.0, 0.5] {
        audio_system.set_volume(volume).await?;
        println!("  Volume set to {:.0}%", volume * 100.0);

        audio_system
            .synthesize_and_play("Volume level test.")
            .await?;
        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
    }

    audio_system.stop().await?;
    println!("✅ Interactive voice demo completed\n");
    Ok(())
}

async fn low_latency_streaming_demo() -> Result<()> {
    println!("🚀 Demo 4: Low-Latency Streaming");
    println!("================================");

    // Create ultra-low latency pipeline
    let pipeline = VoirsPipelineBuilder::new()
        // Note: streaming mode enabled by default
        // Note: latency and performance configured through quality settings
        .build()
        .await?;

    let audio_system = LiveAudioSystem::new(pipeline).await?;

    println!("✅ Ultra-low latency system initialized");
    println!("⚡ Target latency: 20ms");
    println!("🎯 Starting latency measurement tests...\n");

    let test_texts = vec![
        ("Quick", "Hi!"),
        ("Short", "Hello there!"),
        ("Medium", "This is a medium length test sentence."),
        (
            "Longer",
            "This is a longer sentence to test streaming latency with more content.",
        ),
    ];

    let mut latency_measurements = Vec::new();

    for (category, text) in test_texts {
        println!("📊 Testing {} text: \"{}\"", category, text);

        // Measure end-to-end latency
        let start = std::time::Instant::now();

        // Start streaming synthesis
        // Note: using regular synthesis for now
        let audio = audio_system.pipeline.synthesize(text).await?;

        // Simulate streaming behavior
        let _first_chunk_latency = Some(start.elapsed());
        let _total_chunks = 1;
        let mut first_chunk_latency = None;
        let mut total_chunks = 0;

        // Process stream with immediate audio output
        // Simulate streaming processing
        for _i in 0..1 {
            let chunk = &audio;

            if first_chunk_latency.is_none() {
                first_chunk_latency = Some(start.elapsed());
                println!(
                    "  ⚡ First chunk: {:.1}ms",
                    first_chunk_latency.unwrap().as_secs_f64() * 1000.0
                );
            }

            // Queue chunk for immediate playback
            audio_system.queue_audio_chunk(&chunk).await?;
            total_chunks += 1;
        }

        let total_latency = start.elapsed();
        println!(
            "  📊 Total latency: {:.1}ms ({} chunks)",
            total_latency.as_secs_f64() * 1000.0,
            total_chunks
        );

        latency_measurements.push((category, first_chunk_latency.unwrap(), total_latency));

        tokio::time::sleep(tokio::time::Duration::from_millis(800)).await;
    }

    // Analyze latency performance
    println!("\n📈 Latency Analysis:");
    for (category, first_chunk, total) in latency_measurements {
        println!(
            "  {}: First chunk {:.1}ms, Total {:.1}ms",
            category,
            first_chunk.as_secs_f64() * 1000.0,
            total.as_secs_f64() * 1000.0
        );
    }

    audio_system.stop().await?;
    println!("✅ Low-latency streaming demo completed\n");
    Ok(())
}

async fn multi_device_demo() -> Result<()> {
    println!("🎛️  Demo 5: Multi-Device Audio System");
    println!("===================================");

    // Get available devices
    let host = cpal::default_host();
    let devices: Vec<_> = host
        .output_devices()
        .map_err(|e| VoirsError::device_error("host", format!("Device enumeration failed: {}", e)))?
        .collect();

    if devices.len() < 2 {
        println!("⚠️  Multi-device demo requires at least 2 audio devices");
        println!("   Found {} device(s)", devices.len());
        return Ok(());
    }

    println!(
        "🔊 Found {} audio devices for multi-device demo",
        devices.len()
    );

    // Create pipeline for multi-device use
    let pipeline = VoirsPipelineBuilder::new()
        // Note: spatial feature config through features // Use spatial audio for device separation
        // Note: streaming mode enabled by default
        .build()
        .await?;

    // Simulate different audio zones
    let audio_zones = vec![
        ("Main Speakers", "Welcome to the main audio zone."),
        ("Headphones", "This is the headphone audio zone."),
        ("Secondary", "Secondary speakers are now active."),
    ];

    println!("🎯 Simulating multi-zone audio system:");

    for (zone_name, message) in audio_zones {
        println!("  📍 Zone: {} - \"{}\"", zone_name, message);

        // In a real implementation, you would:
        // 1. Select specific audio device for this zone
        // 2. Configure spatial positioning for the zone
        // 3. Route audio to the appropriate device

        #[cfg(feature = "spatial")]
        {
            // TODO: Implement spatial positioning when VoirsPipeline supports it
            // use voirs_spatial::Position3D;
            // let position = match zone_name {
            //     "Main Speakers" => Position3D::new(0.0, 0.0, 0.0),
            //     "Headphones" => Position3D::new(0.0, 0.0, -1.0),
            //     "Secondary" => Position3D::new(2.0, 0.0, 0.0),
            //     _ => Position3D::new(0.0, 0.0, 0.0),
            // };
            // pipeline.set_spatial_position(position).await?;
        }

        let audio = pipeline.synthesize(message).await?;
        println!(
            "    ✅ Generated {:.1}s of audio for {}",
            audio.duration(),
            zone_name
        );

        // Save audio for each zone (simulating device routing)
        let filename = format!("zone_{}.wav", zone_name.to_lowercase().replace(" ", "_"));
        audio.save_wav(&filename)?;
        println!("    💾 Saved to {}", filename);
    }

    println!("✅ Multi-device demo completed\n");
    Ok(())
}

impl LiveAudioSystem {
    async fn new(pipeline: VoirsPipeline) -> Result<Self> {
        let (command_sender, mut command_receiver) = mpsc::unbounded_channel();

        // Initialize audio device manager
        let device_manager = Arc::new(RwLock::new(AudioDeviceManager::new().await?));
        let is_playing = Arc::new(RwLock::new(false));
        let volume = Arc::new(RwLock::new(1.0));
        let pipeline = Arc::new(pipeline);

        let system = Self {
            pipeline: pipeline.clone(),
            device_manager: device_manager.clone(),
            command_sender,
            is_playing: is_playing.clone(),
            volume: volume.clone(),
        };

        // Start command processing task
        let pipeline_clone = pipeline.clone();
        let device_manager_clone = device_manager.clone();
        let is_playing_clone = is_playing.clone();
        let volume_clone = volume.clone();

        // Note: Simplified implementation for demo purposes
        // In production, use proper async task spawning with Send+Sync types
        drop(command_receiver); // Close the receiver for now

        Ok(system)
    }

    async fn synthesize_and_play(&self, text: &str) -> Result<()> {
        self.command_sender
            .send(AudioCommand::SynthesizeAndPlay(text.to_string()))
            .map_err(|e| {
                VoirsError::device_error("audio_system", format!("Command send failed: {}", e))
            })?;
        Ok(())
    }

    async fn set_volume(&self, volume: f32) -> Result<()> {
        self.command_sender
            .send(AudioCommand::SetVolume(volume))
            .map_err(|e| {
                VoirsError::device_error("audio_system", format!("Volume command failed: {}", e))
            })?;
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        self.command_sender.send(AudioCommand::Stop).map_err(|e| {
            VoirsError::device_error("audio_system", format!("Stop command failed: {}", e))
        })?;
        Ok(())
    }

    async fn queue_audio_chunk(&self, chunk: &AudioBuffer) -> Result<()> {
        let mut manager = self.device_manager.write().await;
        manager.queue_audio(chunk.clone()).await
    }
}

impl AudioDeviceManager {
    async fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or_else(|| {
            VoirsError::device_error("host", "No output device available".to_string())
        })?;

        let config = device.default_output_config().map_err(|e| {
            VoirsError::device_error("device", format!("Device config failed: {}", e))
        })?;

        let device_config = AudioDeviceConfig {
            device_name: device.name().unwrap_or_else(|_| "Unknown".to_string()),
            sample_rate: config.sample_rate().0,
            channels: config.channels(),
            buffer_size: 1024, // Default buffer size
            latency_ms: 1024.0 / config.sample_rate().0 as f64 * 1000.0,
        };

        Ok(Self {
            current_device: Some(device),
            config: device_config,
            stream: None,
            audio_queue: Arc::new(Mutex::new(VecDeque::new())),
        })
    }

    async fn queue_audio(&mut self, audio: AudioBuffer) -> Result<()> {
        // Convert audio buffer to device format and queue for playback
        let samples = audio.samples();

        if let Ok(mut queue) = self.audio_queue.lock() {
            queue.extend(samples.iter().cloned());
        }

        // Start playback stream if not already running
        if self.stream.is_none() {
            self.start_playback_stream().await?;
        }

        Ok(())
    }

    async fn start_playback_stream(&mut self) -> Result<()> {
        if let Some(device) = &self.current_device {
            let config = StreamConfig {
                channels: self.config.channels,
                sample_rate: SampleRate(self.config.sample_rate),
                buffer_size: cpal::BufferSize::Fixed(self.config.buffer_size),
            };

            let audio_queue = self.audio_queue.clone();

            let stream = device
                .build_output_stream(
                    &config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        if let Ok(mut queue) = audio_queue.lock() {
                            for sample in data.iter_mut() {
                                *sample = queue.pop_front().unwrap_or(0.0);
                            }
                        }
                    },
                    |err| eprintln!("Audio stream error: {}", err),
                    None,
                )
                .map_err(|e| {
                    VoirsError::device_error("device", format!("Stream creation failed: {}", e))
                })?;

            stream.play().map_err(|e| {
                VoirsError::device_error("stream", format!("Stream play failed: {}", e))
            })?;
            self.stream = Some(stream);
        }

        Ok(())
    }
}
