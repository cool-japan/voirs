//! # IoT/Edge Device Voice Conversion Example
//!
//! This example demonstrates voice conversion optimized for resource-constrained
//! IoT and edge devices (Raspberry Pi, embedded Linux, etc.).
//!
//! ## Features Demonstrated
//! - Memory-constrained operation
//! - Power-aware processing
//! - Thermal throttling
//! - Offline operation
//! - Cloud synchronization
//! - SIMD optimization for ARM NEON

#[cfg(feature = "iot")]
use voirs_conversion::{
    iot::{
        IoTConversionConfig, IoTDeviceStatus, IoTPlatform, IoTPowerMode, IoTProcessingMode,
        IoTVoiceConverter, ResourceConstraints,
    },
    ConversionTarget, ConversionType, VoiceCharacteristics,
};

use std::time::Duration;

#[cfg(feature = "iot")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== IoT/Edge Device Voice Conversion Example ===\n");

    // Step 1: Detect platform and capabilities
    println!("1. Detecting platform capabilities...");
    let platform = detect_platform();
    let constraints = platform.typical_constraints();

    println!("   Platform: {:?}", platform);
    println!("   Max Memory: {} MB", constraints.max_memory_mb);
    println!("   Network: {}", constraints.has_network);
    println!("   Threading: {}", constraints.supports_threading);

    // Step 2: Configure for resource constraints
    println!("\n2. Configuring for resource constraints...");
    let config = IoTConversionConfig {
        platform: platform.clone(),
        power_mode: IoTPowerMode::Balanced,
        processing_mode: IoTProcessingMode::Hybrid,
        max_memory_mb: constraints.max_memory_mb.min(128),
        enable_cloud_fallback: false, // Offline mode
        enable_cloud_sync: false,
        cloud_endpoint: None,
        local_quality_level: 70,
        cloud_timeout_seconds: 30,
        enable_compression: true,
        compression_level: 6,
        enable_caching: true,
        cache_size_mb: 16,
        sample_rate: 16000,
        channels: 1,
        buffer_size: 512,
        cloud_sync_interval_seconds: 300,
    };

    println!("   Power Mode: {:?}", config.power_mode);
    println!("   Max Memory: {} MB", config.max_memory_mb);
    println!("   Processing Mode: {:?}", config.processing_mode);

    // Step 3: Create IoT converter
    println!("\n3. Creating IoT voice converter...");
    let mut converter = IoTVoiceConverter::with_config(config).await?;
    println!("   Converter initialized successfully");

    // Step 4: Define conversion target
    println!("\n4. Defining conversion target...");

    let characteristics = VoiceCharacteristics {
        pitch: voirs_conversion::types::PitchCharacteristics {
            mean_f0: 200.0,
            range: 50.0,
            jitter: 0.02,
            stability: 0.9,
        },
        timing: voirs_conversion::types::TimingCharacteristics {
            speaking_rate: 1.0,
            pause_duration: 0.3,
            rhythm_regularity: 0.8,
        },
        spectral: voirs_conversion::types::SpectralCharacteristics {
            formant_shift: 1.1,
            brightness: 0.2,
            spectral_tilt: -6.0,
            harmonicity: 0.8,
        },
        quality: voirs_conversion::types::QualityCharacteristics {
            breathiness: 0.2,
            roughness: 0.1,
            stability: 0.9,
            resonance: 0.85,
        },
        age_group: Some(voirs_conversion::types::AgeGroup::YoungAdult),
        gender: Some(voirs_conversion::types::Gender::Female),
        accent: None,
        custom_params: Default::default(),
    };

    let target = ConversionTarget::new(characteristics);

    println!("   Target: Female voice, 200Hz pitch");

    // Step 5: Simulate audio stream processing
    println!("\n5. Processing audio stream (simulated)...");

    let chunk_size = 512; // Small chunks for low latency on IoT
    let num_chunks = 20;

    for i in 0..num_chunks {
        // Generate simulated audio chunk
        let audio_chunk = generate_audio_chunk(180.0, chunk_size);

        // Create conversion request
        use voirs_conversion::ConversionRequest;
        let request = ConversionRequest::new(
            format!("chunk_{}", i),
            audio_chunk,
            16000,
            ConversionType::SpeakerConversion,
            target.clone(),
        );

        // Process chunk
        let start = std::time::Instant::now();
        let _converted = converter.convert_with_fallback(&request).await?;
        let latency = start.elapsed();

        if i % 5 == 0 {
            println!("   Chunk {}: latency {:?}", i + 1, latency);
        }

        // Check device status
        let status = converter.get_device_status().await;
        monitor_device_status(&status);

        // Simulate thermal event
        if i == 10 {
            println!("   [Simulated] Temperature spike detected");
            converter
                .set_power_mode(IoTPowerMode::BatteryOptimized)
                .await?;
            println!("   Switched to BatteryOptimized mode");
        }

        // Small delay to simulate real-time constraints
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Step 6: Performance statistics
    println!("\n6. Performance Statistics:");
    let stats = converter.get_statistics();
    println!("   Total conversions: {}", stats.total_conversions);
    println!("   Local conversions: {}", stats.local_conversions);
    println!("   Cloud conversions: {}", stats.cloud_conversions);
    println!("   Cache hits: {}", stats.cache_hits);
    println!(
        "   Average processing time: {:.2} ms",
        stats.average_processing_time_ms
    );

    // Get current device status for additional metrics
    let status = converter.get_device_status().await;
    println!(
        "   Current memory usage: {:.2} MB",
        status.resource_usage.memory_mb
    );
    println!(
        "   Current temperature: {:.1}°C",
        status.temperature_celsius
    );

    // Step 7: Different power modes
    println!("\n7. Testing different power modes...");

    let power_modes = vec![
        IoTPowerMode::HighPerformance,
        IoTPowerMode::Balanced,
        IoTPowerMode::BatteryOptimized,
        IoTPowerMode::UltraLowPower,
    ];

    for mode in power_modes {
        converter.set_power_mode(mode).await?;
        println!("   Testing {:?} mode...", mode);

        let test_audio = generate_audio_chunk(180.0, 1024);

        // Create pitch shift target
        use voirs_conversion::ConversionRequest;
        let pitch_target = ConversionTarget::new(VoiceCharacteristics::default());
        let pitch_request = ConversionRequest::new(
            "power_mode_test".to_string(),
            test_audio,
            16000,
            ConversionType::PitchShift,
            pitch_target,
        );

        let start = std::time::Instant::now();
        let _result = converter.convert_with_fallback(&pitch_request).await?;
        let latency = start.elapsed();

        println!("     Latency: {:?}", latency);
        let status = converter.get_device_status().await;
        println!(
            "     CPU: {:.1}%, Temp: {:.1}°C",
            status.resource_usage.cpu_percent, status.temperature_celsius
        );
    }

    // Step 8: Cloud synchronization (when available)
    println!("\n8. Cloud Synchronization:");
    if constraints.has_network {
        println!("   Network available - cloud sync capability ready");
        // Note: Cloud sync configuration is set during converter initialization
        // In real implementation, this would upload stats to cloud endpoint
        println!("   Cloud sync would be performed automatically based on config");
    } else {
        println!("   No network - operating in offline mode");
    }

    // Step 9: Resource monitoring
    println!("\n9. Resource Monitoring:");
    let status = converter.get_device_status().await;
    let resources = &status.resource_usage;
    println!("   CPU Usage: {:.1}%", resources.cpu_percent);
    println!("   Memory: {:.2} MB", resources.memory_mb);
    println!("   Storage: {:.2} MB", resources.storage_mb);

    if resources.memory_mb > constraints.max_memory_mb as f64 * 0.8 {
        println!("   ⚠️  High memory usage detected");
        println!("   Note: Cache management is handled automatically by the converter");
    }

    // Step 10: Graceful shutdown
    println!("\n10. Graceful shutdown...");
    // Converter will be automatically cleaned up when it goes out of scope
    println!("   Converter will be shut down gracefully");

    println!("\n=== IoT Example Complete ===");
    Ok(())
}

#[cfg(feature = "iot")]
/// Detect IoT platform
fn detect_platform() -> IoTPlatform {
    #[cfg(target_arch = "aarch64")]
    {
        IoTPlatform::RaspberryPi
    }

    #[cfg(target_arch = "arm")]
    {
        IoTPlatform::GenericEmbedded
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        IoTPlatform::EdgeComputing
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "arm",
        all(target_os = "linux", target_arch = "x86_64")
    )))]
    {
        IoTPlatform::GenericEmbedded
    }
}

#[cfg(feature = "iot")]
/// Get device capabilities
fn get_device_capabilities(platform: &IoTPlatform) -> ResourceConstraints {
    match platform {
        IoTPlatform::RaspberryPi => ResourceConstraints {
            max_memory_mb: 1024,
            max_cpu_percent: 80.0,
            max_storage_mb: 8192,
            has_network: true,
            has_gpu: false,
            supports_threading: true,
            supports_floating_point: true,
        },
        IoTPlatform::ESP32 => ResourceConstraints {
            max_memory_mb: 4,
            max_cpu_percent: 85.0,
            max_storage_mb: 16,
            has_network: true,
            has_gpu: false,
            supports_threading: true,
            supports_floating_point: true,
        },
        IoTPlatform::EdgeComputing => ResourceConstraints {
            max_memory_mb: 2048,
            max_cpu_percent: 70.0,
            max_storage_mb: 32768,
            has_network: true,
            has_gpu: true,
            supports_threading: true,
            supports_floating_point: true,
        },
        IoTPlatform::GenericEmbedded => ResourceConstraints {
            max_memory_mb: 64,
            max_cpu_percent: 80.0,
            max_storage_mb: 256,
            has_network: true,
            has_gpu: false,
            supports_threading: true,
            supports_floating_point: true,
        },
        _ => platform.typical_constraints(),
    }
}

#[cfg(feature = "iot")]
/// Monitor device status and warn if thresholds exceeded
fn monitor_device_status(status: &IoTDeviceStatus) {
    if status.temperature_celsius > 75.0 {
        eprintln!(
            "   ⚠️  High temperature: {:.1}°C",
            status.temperature_celsius
        );
    }

    let memory_usage_mb = status.resource_usage.memory_mb;
    if memory_usage_mb > 100.0 {
        eprintln!("   ⚠️  High memory usage: {:.1} MB", memory_usage_mb);
    }

    // Power consumption monitoring would require additional hardware monitoring
    // For now, we check CPU usage as a proxy
    if status.resource_usage.cpu_percent > 90.0 {
        eprintln!(
            "   ⚠️  High CPU usage: {:.1}%",
            status.resource_usage.cpu_percent
        );
    }
}

#[cfg(feature = "iot")]
/// Generate sample audio chunk
fn generate_audio_chunk(f0: f32, size: usize) -> Vec<f32> {
    use std::f32::consts::PI;

    (0..size)
        .map(|i| {
            let t = i as f32 / 16000.0;
            let fundamental = (2.0 * PI * f0 * t).sin() * 0.5;
            let harmonic2 = (2.0 * PI * f0 * 2.0 * t).sin() * 0.25;
            fundamental + harmonic2
        })
        .collect()
}

#[cfg(not(feature = "iot"))]
fn main() {
    println!("This example requires the 'iot' feature.");
    println!("Run with: cargo run --example iot_edge_device --features iot");
}
