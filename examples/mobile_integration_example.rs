//! Mobile Integration Example - VoiRS for iOS and Android Applications
//!
//! This example demonstrates how to integrate VoiRS with mobile applications for iOS and Android.
//! It showcases mobile-optimized configurations, memory management, battery efficiency,
//! and platform-specific integration patterns.
//!
//! ## What this example demonstrates:
//! 1. Mobile-optimized VoiRS configuration for resource constraints
//! 2. Battery-efficient synthesis with configurable quality levels
//! 3. Background processing patterns for mobile apps
//! 4. Memory management for limited mobile resources
//! 5. Offline synthesis capabilities for mobile use cases
//! 6. Platform-specific audio output handling
//!
//! ## Key Mobile Optimization Features:
//! - Low-memory synthesis configurations
//! - Battery usage optimization
//! - Background/foreground processing adaptation
//! - Chunk-based synthesis for responsive UI
//! - Quality vs. performance trade-offs
//! - Offline voice model deployment
//!
//! ## Platform Integration Patterns:
//! - iOS: Swift integration with C FFI
//! - Android: JNI integration with Kotlin/Java
//! - React Native: Native module bindings
//! - Flutter: Platform channel integration
//!
//! ## Prerequisites:
//! - Rust with mobile targets (iOS: aarch64-apple-ios, Android: aarch64-linux-android)
//! - Platform-specific development tools (Xcode, Android Studio)
//! - Cross-compilation toolchains
//!
//! ## Building for mobile platforms:
//! ```bash
//! # iOS
//! cargo build --target aarch64-apple-ios --release
//!
//! # Android
//! cargo build --target aarch64-linux-android --release
//! ```
//!
//! ## Expected output:
//! - Mobile-optimized audio synthesis
//! - Performance metrics for mobile deployment
//! - Memory usage analysis for resource-constrained devices
//! - Battery usage optimization recommendations

use anyhow::{Context, Result};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info};
use voirs_sdk::prelude::*;

/// Mobile-specific synthesis configuration
#[derive(Debug, Clone)]
pub struct MobileConfig {
    /// Quality level optimized for mobile (lower = more battery efficient)
    pub quality_level: MobileQualityLevel,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Enable background processing
    pub background_processing: bool,
    /// Battery optimization mode
    pub battery_optimization: BatteryMode,
    /// Chunk size for responsive synthesis
    pub chunk_duration_ms: u64,
}

#[derive(Debug, Clone)]
pub enum MobileQualityLevel {
    /// Ultra-low quality for maximum battery life
    UltraLow,
    /// Low quality for good battery life
    Low,
    /// Balanced quality and performance
    Balanced,
    /// High quality (higher battery usage)
    High,
}

#[derive(Debug, Clone)]
pub enum BatteryMode {
    /// Maximum battery saving
    MaxSaver,
    /// Balanced battery usage
    Balanced,
    /// Performance priority
    Performance,
}

impl Default for MobileConfig {
    fn default() -> Self {
        MobileConfig {
            quality_level: MobileQualityLevel::Balanced,
            max_memory_mb: 128, // 128MB limit for mobile
            background_processing: true,
            battery_optimization: BatteryMode::Balanced,
            chunk_duration_ms: 200, // 200ms chunks for responsiveness
        }
    }
}

/// Maps a `MobileQualityLevel` to the SDK `QualityLevel`.
fn mobile_quality_to_sdk(level: &MobileQualityLevel) -> QualityLevel {
    match level {
        MobileQualityLevel::UltraLow => QualityLevel::Low,
        MobileQualityLevel::Low => QualityLevel::Low,
        MobileQualityLevel::Balanced => QualityLevel::Medium,
        MobileQualityLevel::High => QualityLevel::High,
    }
}

/// Mobile-optimized VoiRS synthesizer
pub struct MobileSynthesizer {
    pipeline: Arc<VoirsPipeline>,
    config: MobileConfig,
    stats: Arc<Mutex<MobileStats>>,
}

#[derive(Debug, Default)]
pub struct MobileStats {
    synthesis_count: usize,
    total_processing_time: Duration,
    total_audio_duration: f64,
    battery_efficient_ops: usize,
}

impl Clone for MobileStats {
    fn clone(&self) -> Self {
        MobileStats {
            synthesis_count: self.synthesis_count,
            total_processing_time: self.total_processing_time,
            total_audio_duration: self.total_audio_duration,
            battery_efficient_ops: self.battery_efficient_ops,
        }
    }
}

impl MobileSynthesizer {
    /// Create a new mobile-optimized synthesizer
    pub async fn new(config: MobileConfig) -> Result<Self> {
        info!("Creating mobile-optimized VoiRS synthesizer");
        info!("Configuration: {:?}", config);

        let sdk_quality = mobile_quality_to_sdk(&config.quality_level);
        info!(
            "Memory budget: {}MB, battery mode: {:?}",
            config.max_memory_mb, config.battery_optimization
        );

        let pipeline = VoirsPipelineBuilder::new()
            .with_quality(sdk_quality)
            .with_gpu_acceleration(false) // Mobile defaults to CPU-only
            .build()
            .await
            .context("Failed to build mobile-optimized VoiRS pipeline")?;

        Ok(MobileSynthesizer {
            pipeline: Arc::new(pipeline),
            config,
            stats: Arc::new(Mutex::new(MobileStats::default())),
        })
    }

    /// Mobile-optimized synthesis with chunking for UI responsiveness
    pub async fn synthesize_mobile(&self, text: &str) -> Result<AudioBuffer> {
        let start_time = Instant::now();
        info!("Starting mobile synthesis: '{}'", text);

        // Use chunked synthesis for longer texts or when background processing is enabled
        let should_chunk = text.len() > 100 || self.config.background_processing;

        let audio = if should_chunk {
            self.synthesize_chunked(text).await?
        } else {
            self.pipeline
                .synthesize(text)
                .await
                .context("Mobile synthesis failed")?
        };

        let processing_time = start_time.elapsed();

        // Update mobile statistics
        self.update_stats(processing_time, f64::from(audio.duration()))
            .await;

        // Log mobile-specific metrics
        let rtf = processing_time.as_secs_f64() / f64::from(audio.duration());
        let battery_efficient = rtf < 0.5; // Consider < 0.5 RTF as battery efficient

        info!("Mobile synthesis complete:");
        info!("   Processing time: {:.2}s", processing_time.as_secs_f32());
        info!("   Real-time factor: {:.2}x", rtf);
        info!("   Battery efficient: {}", battery_efficient);
        info!(
            "   Memory usage: ~{:.1}MB",
            self.estimate_memory_usage(&audio)
        );

        Ok(audio)
    }

    /// Chunked synthesis for mobile responsiveness
    async fn synthesize_chunked(&self, text: &str) -> Result<AudioBuffer> {
        debug!("Using chunked synthesis for mobile");

        // Split text into mobile-friendly chunks
        let chunks = self.split_text_for_mobile(text);
        let mut combined_samples: Vec<f32> = Vec::new();
        let mut sample_rate = 22050u32;

        for (i, chunk) in chunks.iter().enumerate() {
            debug!(
                "Processing mobile chunk {}/{}: '{}'",
                i + 1,
                chunks.len(),
                chunk
            );

            let chunk_audio = self
                .pipeline
                .synthesize(chunk)
                .await
                .with_context(|| format!("Failed to synthesize mobile chunk {}", i + 1))?;

            if combined_samples.is_empty() {
                sample_rate = chunk_audio.sample_rate();
            }

            combined_samples.extend_from_slice(chunk_audio.samples());

            // Yield control for UI responsiveness (in real mobile app, use proper async yielding)
            tokio::task::yield_now().await;
        }

        Ok(AudioBuffer::new(combined_samples, sample_rate, 1))
    }

    /// Split text into mobile-friendly chunks
    fn split_text_for_mobile(&self, text: &str) -> Vec<String> {
        let target_chunk_size = 50; // characters per chunk for mobile
        let sentences: Vec<&str> = text.split(&['.', '!', '?'][..]).collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        for sentence in sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }

            if current_chunk.len() + sentence.len() > target_chunk_size && !current_chunk.is_empty()
            {
                chunks.push(current_chunk.clone());
                current_chunk = sentence.to_string();
            } else {
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                current_chunk.push_str(sentence);
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        debug!("Split text into {} mobile chunks", chunks.len());
        chunks
    }

    /// Estimate memory usage for mobile monitoring
    fn estimate_memory_usage(&self, audio: &AudioBuffer) -> f64 {
        // Rough estimation: samples * bytes_per_sample + overhead
        let audio_memory = audio.samples().len() * 4; // f32 = 4 bytes
        let overhead = 10 * 1024 * 1024; // 10MB estimated overhead
        (audio_memory + overhead) as f64 / (1024.0 * 1024.0)
    }

    /// Update mobile performance statistics
    async fn update_stats(&self, processing_time: Duration, audio_duration: f64) {
        let mut stats = self.stats.lock().expect("MobileStats mutex poisoned");
        stats.synthesis_count += 1;
        stats.total_processing_time += processing_time;
        stats.total_audio_duration += audio_duration;

        if processing_time.as_secs_f64() / audio_duration < 0.5 {
            stats.battery_efficient_ops += 1;
        }
    }

    /// Get mobile performance statistics
    pub async fn get_mobile_stats(&self) -> MobileStats {
        self.stats
            .lock()
            .expect("MobileStats mutex poisoned")
            .clone()
    }
}

/// Mobile platform integration helpers
pub mod platform_integration {
    /// iOS FFI integration patterns
    #[cfg(target_os = "ios")]
    pub mod ios {
        use super::super::MobileSynthesizer;

        // Example C FFI functions for iOS integration
        #[no_mangle]
        pub extern "C" fn voirs_mobile_create() -> *mut MobileSynthesizer {
            // In real implementation, would handle this properly with Box::into_raw
            std::ptr::null_mut()
        }

        #[no_mangle]
        pub extern "C" fn voirs_mobile_synthesize(
            _synthesizer: *mut MobileSynthesizer,
            _text: *const std::os::raw::c_char,
        ) -> i32 {
            // C FFI implementation for iOS
            0
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize mobile-optimized logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("VoiRS Mobile Integration Example");
    println!("===================================");
    println!();

    // Test different mobile configurations
    let mobile_configs = [
        (
            "Ultra-Low Power",
            MobileConfig {
                quality_level: MobileQualityLevel::UltraLow,
                battery_optimization: BatteryMode::MaxSaver,
                max_memory_mb: 64,
                ..Default::default()
            },
        ),
        ("Balanced", MobileConfig::default()),
        (
            "High Quality",
            MobileConfig {
                quality_level: MobileQualityLevel::High,
                battery_optimization: BatteryMode::Performance,
                max_memory_mb: 256,
                ..Default::default()
            },
        ),
    ];

    for (config_name, config) in mobile_configs.iter() {
        println!("Testing {} Mobile Configuration", config_name);
        println!("{}", "-".repeat(35 + config_name.len()));

        let mobile_start = Instant::now();
        let synthesizer = MobileSynthesizer::new(config.clone()).await?;
        let setup_time = mobile_start.elapsed();

        println!(
            "Mobile synthesizer ready in {:.2}s",
            setup_time.as_secs_f32()
        );

        // Test mobile synthesis patterns
        let mobile_texts = [
            "Mobile synthesis test for quick responses.",
            "This is a longer mobile text that will be processed in chunks to maintain UI responsiveness and optimize battery usage.",
        ];

        let tmp_dir = std::env::temp_dir();
        for (i, text) in mobile_texts.iter().enumerate() {
            println!("   Mobile synthesis {}...", i + 1);

            let audio = synthesizer.synthesize_mobile(text).await?;
            let filename = format!(
                "mobile_{}_{:02}.wav",
                config_name.to_lowercase().replace(' ', "_"),
                i + 1
            );
            let output_path = tmp_dir.join(&filename);
            audio
                .save_wav(&output_path)
                .context("Failed to save mobile audio")?;

            println!(
                "   Generated: {} ({:.2}s audio)",
                output_path.display(),
                audio.duration()
            );
        }

        // Display mobile statistics
        let stats = synthesizer.get_mobile_stats().await;
        println!("Mobile Performance Stats:");
        println!("   Syntheses: {}", stats.synthesis_count);
        println!(
            "   Battery efficient ops: {}/{}",
            stats.battery_efficient_ops, stats.synthesis_count
        );
        if stats.total_audio_duration > 0.0 {
            println!(
                "   Average RTF: {:.2}x",
                stats.total_processing_time.as_secs_f64() / stats.total_audio_duration
            );
        }
        println!();
    }

    // Mobile integration guidance
    println!("Mobile Platform Integration Guide:");
    println!("====================================");
    println!();

    println!("iOS Integration (Swift):");
    println!("  1. Build Rust library: cargo build --target aarch64-apple-ios --release");
    println!("  2. Create C headers for Swift bridging");
    println!("  3. Import in Swift project and use C FFI");
    println!();

    println!("Android Integration (Kotlin/Java):");
    println!("  1. Build Rust library: cargo build --target aarch64-linux-android --release");
    println!("  2. Create JNI wrapper functions");
    println!("  3. Load native library in Android app");
    println!();

    println!("React Native Integration:");
    println!("  1. Create native module wrapper");
    println!("  2. Expose async JavaScript interface");
    println!("  3. Handle background processing properly");
    println!();

    println!("Mobile Optimization Tips:");
    println!("  - Use chunked synthesis for responsiveness");
    println!("  - Monitor memory usage and clean up resources");
    println!("  - Implement background/foreground state handling");
    println!("  - Cache frequently used voice models");
    println!("  - Consider offline model deployment for better UX");

    println!("\nMobile Integration Example Complete!");

    Ok(())
}
