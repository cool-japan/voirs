//! Performance monitoring example for VoiRS SDK.
//!
//! This example demonstrates how to use the performance monitoring
//! capabilities to track synthesis performance, memory usage, and
//! quality metrics in real-time.

use voirs_sdk::prelude::*;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize performance monitoring
    let monitor = PerformanceMonitor::new();
    println!("🚀 Starting VoiRS SDK Performance Monitoring Example");
    
    // Create pipeline with performance tracking
    let pipeline = VoirsPipelineBuilder::new()
        .with_quality(QualityLevel::High)
        .build()
        .await?;
    
    println!("✅ Pipeline initialized successfully");
    
    // Perform several synthesis operations with monitoring
    let texts = [
        "Hello, this is a performance test.",
        "The VoiRS SDK provides excellent speech synthesis capabilities.",
        "Performance monitoring helps optimize your applications.",
        "Real-time metrics enable better resource management.",
        "Quality tracking ensures consistent audio output.",
    ];
    
    println!("\n📊 Performing synthesis operations with performance tracking...");
    
    for (i, text) in texts.iter().enumerate() {
        println!("  Synthesizing text {} of {}: \"{}\"", i + 1, texts.len(), text);
        
        // Start performance measurement for this operation
        let _scope = monitor.start_operation(&format!("synthesis_{}", i + 1));
        
        // Synthesize with timing
        let start_time = std::time::Instant::now();
        let audio = pipeline.synthesize(text).await?;
        let processing_time = start_time.elapsed();
        
        // Calculate audio duration for RTF
        let audio_duration = Duration::from_secs_f64(audio.len() as f64 / audio.sample_rate() as f64);
        
        // Record performance metrics
        monitor.record_synthesis(processing_time, audio_duration)?;
        
        // Simulate memory usage tracking (in a real app, you'd get actual memory usage)
        let simulated_memory = 50_000_000 + (i * 5_000_000) as u64;
        monitor.update_memory_usage(simulated_memory)?;
        
        // Simulate quality metrics (in a real app, you'd calculate these from the audio)
        let snr = 25.0 + (fastrand::f64() * 10.0);
        let thd = 0.02 + (fastrand::f64() * 0.01);
        let dynamic_range = 35.0 + (fastrand::f64() * 15.0);
        monitor.record_quality_metrics(snr, thd, dynamic_range)?;
        
        // Simulate cache hit rate
        let cache_hit_rate = 0.8 + (fastrand::f64() * 0.2);
        monitor.record_cache_hit_rate(cache_hit_rate)?;
        
        println!("    ✓ Completed in {:.2}ms (RTF: {:.3})", 
                processing_time.as_millis(),
                processing_time.as_secs_f64() / audio_duration.as_secs_f64());
        
        // Small delay between operations
        sleep(Duration::from_millis(100)).await;
    }
    
    println!("\n📈 Performance Analysis:");
    println!("========================");
    
    // Get and display current metrics
    let metrics = monitor.get_metrics()?;
    
    println!("🔢 Synthesis Statistics:");
    println!("   Total syntheses: {}", metrics.total_syntheses);
    println!("   Average synthesis time: {:.2}ms", metrics.average_synthesis_time.as_millis());
    println!("   Total processing time: {:.2}s", metrics.total_processing_time.as_secs_f64());
    
    println!("\n💾 Memory Usage:");
    println!("   Current usage: {:.2} MB", metrics.current_memory_usage as f64 / 1_048_576.0);
    println!("   Peak usage: {:.2} MB", metrics.peak_memory_usage as f64 / 1_048_576.0);
    
    println!("\n⚡ Real-time Performance:");
    println!("   Average RTF: {:.3}", metrics.rtf_stats.average_rtf);
    println!("   Min RTF: {:.3}", metrics.rtf_stats.min_rtf);
    println!("   Max RTF: {:.3}", metrics.rtf_stats.max_rtf);
    println!("   RTF violations: {}", metrics.rtf_stats.rtf_violations);
    
    println!("\n🎵 Audio Quality:");
    println!("   Average SNR: {:.1} dB", metrics.quality_metrics.average_snr);
    println!("   Average THD: {:.3}%", metrics.quality_metrics.average_thd * 100.0);
    println!("   Average dynamic range: {:.1} dB", metrics.quality_metrics.average_dynamic_range);
    println!("   Quality warnings: {}", metrics.quality_metrics.quality_warnings);
    
    println!("\n🚀 Cache Performance:");
    println!("   Cache hit rate: {:.1}%", metrics.cache_hit_rate * 100.0);
    
    // Component timing breakdown
    if !metrics.component_timings.is_empty() {
        println!("\n⏱️  Component Timings:");
        for (component, duration) in &metrics.component_timings {
            println!("   {}: {:.2}ms", component, duration.as_millis());
        }
    }
    
    // Generate detailed performance report
    println!("\n📋 Detailed Performance Report:");
    println!("{}", monitor.generate_report()?);
    
    // Demonstrate performance monitoring with macros
    println!("\n🔧 Using Performance Macros:");
    
    use voirs_sdk::measure_performance;
    
    let result = measure_performance!(monitor, "macro_test", {
        // Simulate some work
        sleep(Duration::from_millis(50)).await;
        "Macro measurement complete"
    });
    
    println!("   Macro result: {}", result);
    
    // Show final metrics
    let final_metrics = monitor.get_metrics()?;
    if let Some(macro_time) = final_metrics.component_timings.get("macro_test") {
        println!("   Macro operation took: {:.2}ms", macro_time.as_millis());
    }
    
    println!("\n✨ Performance monitoring example completed successfully!");
    println!("📊 Use these metrics to optimize your speech synthesis applications.");
    
    Ok(())
}