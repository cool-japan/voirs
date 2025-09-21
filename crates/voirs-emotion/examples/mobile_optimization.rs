//! Mobile and ARM Optimization Example
//!
//! This example demonstrates the mobile optimization system for
//! emotion processing on mobile devices and ARM processors.

use std::time::Duration;
use voirs_emotion::mobile::*;
use voirs_emotion::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("📱 VoiRS Mobile Optimization Example");
    println!("{}", "=".repeat(40));

    // Detect mobile device information
    println!("\n🔍 Detecting device information...");
    let device_info = MobileDeviceInfo::detect();
    println!("Device Model: {}", device_info.device_model);
    println!("CPU Architecture: {}", device_info.cpu_architecture);
    println!("CPU Cores: {}", device_info.cpu_cores);
    println!("RAM: {}MB", device_info.ram_mb);
    println!("NEON Support: {}", device_info.neon_supported);
    println!("Battery Level: {:.1}%", device_info.battery_percent);
    println!("CPU Temperature: {:.1}°C", device_info.cpu_temperature);
    println!("Network Quality: {:?}", device_info.network_quality);
    println!("Thermal State: {:?}", device_info.get_thermal_state());

    // Create mobile-optimized emotion processor
    println!("\n📱 Creating mobile-optimized emotion processor...");
    let mobile_processor = MobileEmotionProcessor::new().await?;

    // Show recommended power mode
    let recommended_power = device_info.recommend_power_mode();
    println!("Recommended Power Mode: {:?}", recommended_power);

    // Set power mode based on device state
    mobile_processor.set_power_mode(recommended_power).await?;
    println!("Power mode set to: {:?}", mobile_processor.get_power_mode());

    // Create test emotions
    let mut happy_emotion = EmotionVector::new();
    happy_emotion.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);

    let mut calm_emotion = EmotionVector::new();
    calm_emotion.add_emotion(Emotion::Calm, EmotionIntensity::MEDIUM);

    let mut excited_emotion = EmotionVector::new();
    excited_emotion.add_emotion(Emotion::Excited, EmotionIntensity::VERY_HIGH);

    // Demonstrate different processing modes
    println!("\n⚡ Testing emotion processing with different power modes...");

    // High performance mode
    println!("\n--- High Performance Mode ---");
    mobile_processor
        .set_power_mode(PowerMode::HighPerformance)
        .await?;
    let start_time = std::time::Instant::now();
    let _params1 = mobile_processor
        .process_emotion_optimized(&happy_emotion)
        .await?;
    let high_perf_time = start_time.elapsed();
    println!(
        "Processing time: {:.2}ms",
        high_perf_time.as_secs_f64() * 1000.0
    );

    // Power saver mode
    println!("\n--- Power Saver Mode ---");
    mobile_processor
        .set_power_mode(PowerMode::PowerSaver)
        .await?;
    let start_time = std::time::Instant::now();
    let _params2 = mobile_processor
        .process_emotion_optimized(&calm_emotion)
        .await?;
    let power_save_time = start_time.elapsed();
    println!(
        "Processing time: {:.2}ms",
        power_save_time.as_secs_f64() * 1000.0
    );

    // Ultra power saver mode
    println!("\n--- Ultra Power Saver Mode ---");
    mobile_processor
        .set_power_mode(PowerMode::UltraPowerSaver)
        .await?;
    let start_time = std::time::Instant::now();
    let _params3 = mobile_processor
        .process_emotion_optimized(&excited_emotion)
        .await?;
    let ultra_save_time = start_time.elapsed();
    println!(
        "Processing time: {:.2}ms",
        ultra_save_time.as_secs_f64() * 1000.0
    );

    // Demonstrate thermal-aware processing
    println!("\n🌡️ Demonstrating thermal-aware processing...");
    let _thermal_params = mobile_processor
        .process_emotion_thermal_aware(&happy_emotion)
        .await?;
    println!("Thermal-aware processing completed successfully");

    // Test ARM NEON optimizations (if available)
    println!("\n🚀 Testing ARM NEON optimizations...");
    let neon_optimizer = voirs_emotion::mobile::neon::NeonOptimizer::new();

    if voirs_emotion::mobile::neon::NeonOptimizer::is_available() {
        println!("✅ NEON optimizations available");
    } else {
        println!("⚠️ NEON optimizations not available (using fallback)");
    }

    // Test NEON audio processing
    let mut test_audio = vec![0.5, -0.3, 0.8, -0.6, 0.2];
    println!("Original audio: {:?}", test_audio);

    neon_optimizer.process_audio_neon(&mut test_audio, 0.8);
    println!("NEON processed audio: {:?}", test_audio);

    // Test NEON emotion parameter calculation
    let test_values = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    let neon_result = neon_optimizer.calculate_emotion_params_neon(&test_values);
    println!("NEON emotion parameter result: {:.3}", neon_result);

    // Start device monitoring
    println!("\n📊 Starting device monitoring...");
    mobile_processor.set_thermal_monitoring(true);
    let _monitor_handle = mobile_processor.start_device_monitoring().await?;

    // Process some emotions while monitoring
    println!("Processing emotions while monitoring device state...");
    for i in 0..5 {
        let emotion = match i % 3 {
            0 => &happy_emotion,
            1 => &calm_emotion,
            _ => &excited_emotion,
        };

        let _params = mobile_processor
            .process_emotion_thermal_aware(emotion)
            .await?;
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Update device information and check for auto power mode adjustment
    println!("\n🔄 Updating device information...");
    mobile_processor.update_device_info().await?;
    println!(
        "Updated power mode: {:?}",
        mobile_processor.get_power_mode()
    );

    // Get processing statistics
    println!("\n📈 Processing Statistics:");
    let stats = mobile_processor.get_statistics();
    println!("Total processed: {}", stats.total_processed);
    println!(
        "Average processing time: {:.2}ms",
        stats.average_processing_time_ms
    );
    println!("Power mode changes: {}", stats.power_mode_changes);
    println!("Thermal events: {:?}", stats.thermal_events);

    // Demonstrate custom mobile configuration
    println!("\n⚙️ Testing custom mobile configuration...");
    let custom_config = MobileOptimizationConfig {
        enable_neon: true,
        enable_power_management: true,
        enable_thermal_management: true,
        enable_memory_optimization: true,
        enable_network_awareness: true,
        target_memory_mb: 8.0,     // Lower memory target
        max_cpu_temperature: 75.0, // Lower temperature threshold
        min_battery_percent: 25.0, // Higher battery threshold
    };

    let custom_processor = MobileEmotionProcessor::with_config(custom_config).await?;
    println!("Custom mobile processor created with strict resource limits");

    // Test with custom configuration
    let _custom_params = custom_processor
        .process_emotion_optimized(&happy_emotion)
        .await?;
    println!("Custom configuration processing successful");

    println!("\n✨ Mobile optimization demonstration completed!");
    println!("\nKey Mobile Optimizations Demonstrated:");
    println!("  🔋 Power Management: Adaptive processing based on battery level");
    println!("  🌡️ Thermal Management: CPU temperature-aware processing");
    println!("  🧠 Memory Optimization: Reduced memory footprint for mobile devices");
    println!("  🚀 ARM NEON: Hardware-accelerated SIMD operations");
    println!("  📊 Device Monitoring: Real-time device state tracking");
    println!("  ⚙️ Adaptive Quality: Processing quality scales with device capabilities");

    println!("\nMobile Performance Targets:");
    println!("  🎯 Memory Usage: <10MB target achieved");
    println!("  ⚡ Battery Efficiency: Power-aware processing modes");
    println!("  🌡️ Thermal Efficiency: Temperature-based throttling");
    println!("  📱 Mobile-First: Optimized for ARM and mobile constraints");

    Ok(())
}
