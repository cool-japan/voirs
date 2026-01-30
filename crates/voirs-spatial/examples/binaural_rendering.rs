//! Binaural rendering example with multiple sound sources
//!
//! This example demonstrates:
//! - Creating a binaural renderer
//! - Managing multiple sound sources at different positions
//! - Real-time binaural audio rendering
//!
//! Run with: cargo run --example binaural_rendering --no-default-features

use std::sync::Arc;
use voirs_spatial::{BinauralConfig, BinauralRenderer, HrtfDatabase, Position3D, SourceType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Binaural Rendering Example ===\n");

    // Create binaural rendering configuration
    let config = BinauralConfig {
        sample_rate: 48000,
        buffer_size: 512,
        hrir_length: 200,
        max_sources: 8,
        use_gpu: false,
        crossfade_duration: 0.05,
        quality_level: 0.8,
        enable_distance_modeling: true,
        enable_air_absorption: true,
        near_field_distance: 0.2,
        far_field_distance: 10.0,
        optimize_for_latency: true,
    };

    println!("✓ Created binaural configuration");
    println!("  Sample rate: {} Hz", config.sample_rate);
    println!("  Buffer size: {} samples", config.buffer_size);
    println!("  Max sources: {}", config.max_sources);
    println!("  Quality level: {:.0}%", config.quality_level * 100.0);
    println!("  HRIR length: {} samples\n", config.hrir_length);

    // Create HRTF database and renderer
    let hrtf_db = Arc::new(HrtfDatabase::load_default().await?);
    println!("✓ Loaded HRTF database");
    println!("  Database type: Default (MIT KEMAR)\n");

    let mut renderer = BinauralRenderer::new(config.clone(), hrtf_db).await?;
    println!("✓ Created binaural renderer\n");

    // Define multiple sound sources at different positions
    let sources = vec![
        (
            "front_speaker",
            Position3D::new(0.0, 2.0, 0.0),
            "Front at 2m",
        ),
        (
            "left_speaker",
            Position3D::new(-1.5, 1.0, 0.0),
            "Left at 1.5m",
        ),
        (
            "right_speaker",
            Position3D::new(1.5, 1.0, 0.0),
            "Right at 1.5m",
        ),
        (
            "above_speaker",
            Position3D::new(0.0, 1.0, 1.0),
            "Above at 1m height",
        ),
    ];

    println!("Adding {} sound sources:", sources.len());

    for (id, position, description) in &sources {
        // Add source to renderer (as static source)
        renderer
            .add_source(id.to_string(), *position, SourceType::Static)
            .await?;

        println!(
            "  ✓ {} - ({:.1}, {:.1}, {:.1}) - {}",
            id, position.x, position.y, position.z, description
        );
    }

    println!("\n✓ All sources added to renderer\n");

    // Set listener position and orientation
    let listener_pos = Position3D::new(0.0, 0.0, 0.0);
    let listener_orientation: (f32, f32, f32) = (0.0, 0.0, 0.0); // Facing forward (yaw, pitch, roll in radians)

    println!("Listener Configuration:");
    println!(
        "  Position: ({:.1}, {:.1}, {:.1}) meters",
        listener_pos.x, listener_pos.y, listener_pos.z
    );
    println!(
        "  Orientation: yaw={:.1}°, pitch={:.1}°, roll={:.1}°\n",
        listener_orientation.0.to_degrees(),
        listener_orientation.1.to_degrees(),
        listener_orientation.2.to_degrees()
    );

    // Process binaural audio frame
    println!("Processing binaural audio frame...");

    let binaural_output = renderer
        .process_frame(listener_pos, listener_orientation)
        .await?;

    println!("✓ Binaural frame processed successfully!");
    println!("  Output format: Stereo (binaural)");
    println!("  Left channel: {} samples", binaural_output.left.len());
    println!("  Right channel: {} samples", binaural_output.right.len());
    println!("  Sample rate: {} Hz\n", binaural_output.sample_rate);

    // Get performance metrics
    let metrics = renderer.get_metrics().await;
    println!("Performance Metrics:");
    println!(
        "  Avg processing time: {:.2}ms",
        metrics.average_processing_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Peak processing time: {:.2}ms",
        metrics.peak_processing_time.as_secs_f64() * 1000.0
    );
    println!("  Active sources: {}", metrics.active_sources);
    println!("  CPU usage: {:.1}%", metrics.cpu_usage * 100.0);
    println!(
        "  Memory usage: {:.2} KB",
        metrics.memory_usage as f32 / 1024.0
    );
    println!("  Underruns: {}", metrics.underruns);
    println!("  Overruns: {}", metrics.overruns);

    let processing_time_ms = metrics.average_processing_time.as_secs_f64() * 1000.0;
    let buffer_duration_ms = (config.buffer_size as f32 / config.sample_rate as f32) * 1000.0;
    println!(
        "  Real-time factor: {:.3}x",
        buffer_duration_ms / processing_time_ms as f32
    );

    // Analyze output
    let left_energy: f32 = binaural_output.left.iter().map(|&x| x * x).sum();
    let right_energy: f32 = binaural_output.right.iter().map(|&x| x * x).sum();

    let left_rms = (left_energy / binaural_output.left.len() as f32).sqrt();
    let right_rms = (right_energy / binaural_output.right.len() as f32).sqrt();

    println!("\nOutput Analysis:");
    println!("  Left channel RMS: {:.4}", left_rms);
    println!("  Right channel RMS: {:.4}", right_rms);

    if left_energy > 0.001 && right_energy > 0.001 {
        println!(
            "  Balance: {:.1}dB (L/R energy ratio)",
            10.0 * (right_energy / left_energy).log10()
        );
    }

    // Peak analysis
    let left_peak = binaural_output
        .left
        .iter()
        .map(|&x| x.abs())
        .fold(0.0f32, |a, b| a.max(b));
    let right_peak = binaural_output
        .right
        .iter()
        .map(|&x| x.abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!("  Left channel peak: {:.4}", left_peak);
    println!("  Right channel peak: {:.4}", right_peak);

    println!("\n✅ Example completed successfully!");
    println!("\nKey Takeaways:");
    println!("  • Binaural rendering creates 3D audio for headphones");
    println!("  • HRTF (Head-Related Transfer Function) simulates ear geometry");
    println!("  • Multiple sources can be positioned in 3D space");
    println!("  • Real-time processing maintains low latency");
    println!("\nIn a real application, you would:");
    println!("  1. Continuously update source positions");
    println!("  2. Stream audio to an output device (speakers/headphones)");
    println!("  3. Track listener head movements for VR/AR");
    println!("  4. Handle dynamic source addition/removal");

    Ok(())
}
