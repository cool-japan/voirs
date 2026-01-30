//! Mobile Platform Optimizations Example
//!
//! This example demonstrates how to use mobile-specific optimizations
//! for efficient speech recognition on iOS and Android devices.
//!
//! Run with:
//! ```
//! cargo run --example mobile_optimizations --all-features
//! ```

use voirs_recognizer::mobile::{
    LifecycleEvent, MobileConfig, MobileOptimizer, NetworkType, PowerMode,
};
use voirs_recognizer::RecognitionError;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("VoiRS Mobile Platform Optimizations Example\n");
    println!("==========================================\n");

    // Create mobile-optimized configuration
    let config = MobileConfig {
        power_mode: PowerMode::Balanced,
        max_memory_mb: 256,
        enable_background_processing: true,
        thermal_throttle: true,
        network_aware: true,
        min_battery_percent: 20.0,
        max_cpu_temp: 80.0,
        prefer_ondevice: true,
    };

    println!("Configuration:");
    println!("  Power Mode: {:?}", config.power_mode);
    println!("  Max Memory: {}MB", config.max_memory_mb);
    println!(
        "  Background Processing: {}",
        config.enable_background_processing
    );
    println!("  Thermal Throttling: {}", config.thermal_throttle);
    println!("  Network Aware: {}", config.network_aware);
    println!();

    // Initialize mobile optimizer
    let optimizer = MobileOptimizer::new(config).await?;
    println!("✓ Mobile optimizer initialized\n");

    // Get platform capabilities
    println!("Platform Capabilities:");
    let capabilities = optimizer.get_platform_capabilities().await?;
    println!("  CPU Cores: {}", capabilities.cpu_cores);
    println!("  Total RAM: {}MB", capabilities.total_ram_mb);
    println!("  Available RAM: {}MB", capabilities.available_ram_mb);
    println!("  Has GPU: {}", capabilities.has_gpu);
    println!("  Has Neural Engine: {}", capabilities.has_neural_engine);
    println!("  Has NNAPI: {}", capabilities.has_nnapi);
    println!(
        "  Battery Level: {:.1}%",
        capabilities.battery_level * 100.0
    );
    println!("  Is Charging: {}", capabilities.is_charging);
    println!("  Network Type: {:?}", capabilities.network_type);
    println!();

    // Battery optimization
    println!("Battery Optimization:");
    match optimizer.optimize_for_battery().await {
        Ok(()) => println!("  ✓ Battery optimization successful"),
        Err(RecognitionError::ResourceError { message, .. }) => {
            println!("  ⚠ Battery warning: {}", message);
        }
        Err(e) => println!("  ✗ Battery optimization error: {}", e),
    }
    println!();

    // Thermal check
    println!("Thermal Management:");
    let should_throttle = optimizer.should_throttle().await?;
    if should_throttle {
        println!("  ⚠ Thermal throttling recommended");
    } else {
        println!("  ✓ Temperature normal, no throttling needed");
    }
    println!();

    // Network check for model downloads
    println!("Network Status:");
    let can_download = optimizer.can_download_models().await?;
    match capabilities.network_type {
        NetworkType::Wifi | NetworkType::Cellular5G => {
            println!("  ✓ Network suitable for downloads: {}", can_download);
        }
        NetworkType::Cellular4G | NetworkType::Cellular3G => {
            println!("  ⚠ Limited network, consider WiFi for large downloads");
        }
        NetworkType::None => {
            println!("  ✗ No network connectivity");
        }
    }
    println!();

    // Simulate app lifecycle events
    println!("App Lifecycle Events:");

    // App going to background
    println!("  Simulating app entering background...");
    optimizer
        .handle_lifecycle_event(LifecycleEvent::WillResignActive)
        .await?;
    println!("  ✓ App inactive");

    optimizer
        .handle_lifecycle_event(LifecycleEvent::DidEnterBackground)
        .await?;
    println!("  ✓ App in background");

    // App returning to foreground
    println!("  Simulating app returning to foreground...");
    optimizer
        .handle_lifecycle_event(LifecycleEvent::WillEnterForeground)
        .await?;
    println!("  ✓ App will enter foreground");

    optimizer
        .handle_lifecycle_event(LifecycleEvent::DidBecomeActive)
        .await?;
    println!("  ✓ App active");
    println!();

    // Memory optimization
    println!("Memory Optimization:");
    optimizer.optimize_memory().await?;
    println!("  ✓ Memory optimized");
    println!();

    // Power mode recommendations
    println!("Power Mode Recommendations:");
    for mode in [
        PowerMode::Performance,
        PowerMode::Balanced,
        PowerMode::LowPower,
        PowerMode::UltraLowPower,
    ] {
        println!("  {:?}: For {:?} usage", mode, get_usage_scenario(mode));
    }
    println!();

    println!("Example completed successfully!");
    Ok(())
}

fn get_usage_scenario(mode: PowerMode) -> &'static str {
    match mode {
        PowerMode::Performance => "maximum accuracy and speed (while charging)",
        PowerMode::Balanced => "typical usage with good battery life",
        PowerMode::LowPower => "extended battery life with reduced accuracy",
        PowerMode::UltraLowPower => "critical battery levels (< 20%)",
    }
}
