//! Basic spatial audio processing example
//!
//! This example demonstrates how to:
//! - Create a spatial audio processor
//! - Process audio with 3D positioning
//! - Apply HRTF and distance attenuation
//!
//! Run with: cargo run --example basic_spatial_audio --no-default-features

use voirs_spatial::{Position3D, SpatialConfig, SpatialEffect, SpatialProcessor, SpatialRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Spatial Audio Example ===\n");

    // Create spatial audio configuration
    let config = SpatialConfig::default();
    println!("✓ Created spatial audio configuration");
    println!("  Sample rate: {} Hz", config.sample_rate);
    println!("  Buffer size: {} samples\n", config.buffer_size);

    // Create spatial processor
    let mut processor = SpatialProcessor::new(config).await?;
    println!("✓ Created spatial audio processor\n");

    // Generate a simple test audio signal (1 second of a sine wave)
    let sample_rate = 48000;
    let duration = 1.0; // seconds
    let frequency = 440.0; // A4 note
    let num_samples = (sample_rate as f32 * duration) as usize;

    let audio_data: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.3
        })
        .collect();

    println!("✓ Generated test audio signal");
    println!("  Duration: {:.2}s", duration);
    println!("  Frequency: {}Hz", frequency);
    println!("  Samples: {}\n", num_samples);

    // Position sound source to the right of the listener
    let source_position = Position3D::new(2.0, 0.0, 0.0); // 2 meters to the right
    let listener_position = Position3D::new(0.0, 0.0, 0.0); // Listener at origin
    let listener_orientation: (f32, f32, f32) = (0.0, 0.0, 0.0); // Facing forward (yaw, pitch, roll)

    println!("✓ Positioned sound source");
    println!(
        "  Source: ({:.1}, {:.1}, {:.1}) meters",
        source_position.x, source_position.y, source_position.z
    );
    println!(
        "  Listener: ({:.1}, {:.1}, {:.1}) meters",
        listener_position.x, listener_position.y, listener_position.z
    );
    println!(
        "  Orientation: yaw={:.1}°, pitch={:.1}°, roll={:.1}°\n",
        listener_orientation.0.to_degrees(),
        listener_orientation.1.to_degrees(),
        listener_orientation.2.to_degrees()
    );

    // Create spatial audio request
    let request = SpatialRequest {
        id: "example_source".to_string(),
        audio: audio_data,
        sample_rate,
        source_position,
        listener_position,
        listener_orientation,
        effects: vec![
            SpatialEffect::Hrtf,                // Apply HRTF for 3D positioning
            SpatialEffect::DistanceAttenuation, // Attenuate based on distance
            SpatialEffect::Reverb,              // Add room reverb
        ],
        parameters: std::collections::HashMap::new(),
    };

    println!("✓ Created spatial audio request");
    println!("  Effects: HRTF, Distance Attenuation, Reverb\n");

    // Process the spatial audio
    println!("Processing spatial audio...");
    let result = processor.process_request(request).await?;

    println!("\n✓ Spatial audio processed successfully!");
    println!("  Left channel: {} samples", result.audio.left.len());
    println!("  Right channel: {} samples", result.audio.right.len());
    println!(
        "  Processing time: {:.2}ms",
        result.processing_time.as_secs_f64() * 1000.0
    );
    println!("  Request ID: {}", result.request_id);

    // Calculate some basic statistics
    let left_max = result
        .audio
        .left
        .iter()
        .map(|&x| x.abs())
        .fold(0.0f32, |a, b| a.max(b));

    let right_max = result
        .audio
        .right
        .iter()
        .map(|&x| x.abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!("\nOutput Statistics:");
    println!("  Left channel peak: {:.3}", left_max);
    println!("  Right channel peak: {:.3}", right_max);

    if left_max > 0.001 {
        println!(
            "  Stereo difference: {:.1}dB",
            20.0 * (right_max / left_max).log10()
        );
    }

    println!("\n✅ Example completed successfully!");
    println!("\nNote: This example generates binaural audio but doesn't save it to a file.");
    println!(
        "To save the output, you would need to use a library like 'hound' to write WAV files."
    );

    Ok(())
}
