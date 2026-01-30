//! # Style Transfer Example
//!
//! This example demonstrates basic voice style transfer.
//!
//! Note: This is a simplified demonstration. In a real application, you would:
//! - Load pre-trained style models
//! - Use actual audio files
//! - Configure the system based on your specific requirements

use voirs_conversion::{
    style_transfer::{StyleTransferConfig, StyleTransferMethod, StyleTransferSystem},
    Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Voice Style Transfer Example ===\n");

    // Create style transfer system with default configuration
    println!("1. Initializing style transfer system...");
    let mut config = StyleTransferConfig::default();
    config.transfer_method = StyleTransferMethod::NeuralStyleTransfer;
    config.style_transfer_strength = 0.8;
    config.content_preservation_weight = 0.7;

    let _system = StyleTransferSystem::new(config);
    println!("   System initialized successfully");
    println!(
        "   Transfer method: {:?}",
        StyleTransferMethod::NeuralStyleTransfer
    );
    println!("   Style strength: 0.8");
    println!("   Content preservation: 0.7");

    // Generate sample audio for demonstration
    println!("\n2. Generating sample audio...");
    let source_audio = generate_sample_audio(200.0, 16000);
    let sample_rate = 16000;
    println!(
        "   Generated {} samples at {} Hz",
        source_audio.len(),
        sample_rate
    );
    println!(
        "   Duration: {:.2}s",
        source_audio.len() as f32 / sample_rate as f32
    );

    // Demonstrate different transfer methods
    println!("\n3. Available transfer methods:");
    let methods = vec![
        StyleTransferMethod::ContentStyleDecomposition,
        StyleTransferMethod::AdversarialTransfer,
        StyleTransferMethod::CycleConsistentTransfer,
        StyleTransferMethod::NeuralStyleTransfer,
        StyleTransferMethod::SemanticStyleTransfer,
        StyleTransferMethod::HierarchicalTransfer,
    ];

    for method in methods {
        let mut method_config = StyleTransferConfig::default();
        method_config.transfer_method = method;
        let _method_system = StyleTransferSystem::new(method_config);
        println!("   - {:?}", method);
    }

    // Configuration examples
    println!("\n4. Configuration examples:");

    println!("   High Quality (prioritize quality over speed):");
    let mut hq_config = StyleTransferConfig::default();
    hq_config.quality_threshold = 0.9;
    hq_config
        .synthesis_settings
        .quality_enhancement
        .target_quality = 0.95;
    println!("      Quality threshold: {}", hq_config.quality_threshold);
    println!(
        "      Target quality: {}",
        hq_config
            .synthesis_settings
            .quality_enhancement
            .target_quality
    );

    println!("\n   Real-time Processing:");
    let mut rt_config = StyleTransferConfig::default();
    rt_config.realtime_settings.enabled = true;
    rt_config.realtime_settings.max_latency = 50.0; // 50ms
    rt_config.realtime_settings.chunk_size = 512;
    println!("      Enabled: {}", rt_config.realtime_settings.enabled);
    println!(
        "      Max latency: {}ms",
        rt_config.realtime_settings.max_latency
    );
    println!(
        "      Chunk size: {} samples",
        rt_config.realtime_settings.chunk_size
    );

    println!("\n   Balanced (quality vs. speed):");
    let mut balanced_config = StyleTransferConfig::default();
    balanced_config.quality_threshold = 0.75;
    balanced_config.style_transfer_strength = 0.7;
    balanced_config.content_preservation_weight = 0.8;
    println!("      Quality: {}", balanced_config.quality_threshold);
    println!(
        "      Style strength: {}",
        balanced_config.style_transfer_strength
    );
    println!(
        "      Content preservation: {}",
        balanced_config.content_preservation_weight
    );

    // Feature extraction settings
    println!("\n5. Feature extraction capabilities:");
    println!("   - Prosodic features (F0, rhythm, intonation)");
    println!("   - Spectral features (formants, harmonics)");
    println!("   - Temporal features (timing, duration)");
    println!("   - Semantic features (content, meaning)");

    // Synthesis settings
    println!("\n6. Synthesis capabilities:");
    println!("   - Neural vocoder synthesis");
    println!("   - Parametric synthesis");
    println!("   - Hybrid synthesis");
    println!("   - Direct waveform synthesis");

    // Post-processing features
    println!("\n7. Post-processing features:");
    println!("   - Noise reduction");
    println!("   - Dynamic range compression");
    println!("   - Spectral enhancement");
    println!("   - Artifacts removal");
    println!("   - Super-resolution");
    println!("   - Bandwidth extension");

    println!("\n=== Example completed successfully ===");
    println!("\nNote: This is a demonstration of the API structure.");
    println!("For full functionality, you would need to:");
    println!("  1. Train or load pre-trained style models");
    println!("  2. Register styles using add_style_model()");
    println!("  3. Perform actual style transfer using transfer_style()");
    println!("\nSee the test files for complete working examples.");

    Ok(())
}

/// Generate sample audio signal for demonstration
fn generate_sample_audio(f0: f32, sample_count: usize) -> Vec<f32> {
    use std::f32::consts::PI;

    (0..sample_count)
        .map(|i| {
            let t = i as f32 / 16000.0;
            // Generate simple harmonic series
            let fundamental = (2.0 * PI * f0 * t).sin() * 0.5;
            let harmonic2 = (2.0 * PI * f0 * 2.0 * t).sin() * 0.25;
            let harmonic3 = (2.0 * PI * f0 * 3.0 * t).sin() * 0.15;
            fundamental + harmonic2 + harmonic3
        })
        .collect()
}
