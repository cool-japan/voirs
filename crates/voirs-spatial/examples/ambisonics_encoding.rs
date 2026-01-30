//! Ambisonics encoding and decoding example
//!
//! This example demonstrates:
//! - Encoding audio to ambisonics format
//! - Decoding ambisonics to different speaker configurations
//! - Working with spherical coordinates
//!
//! Run with: cargo run --example ambisonics_encoding --no-default-features

use scirs2_core::ndarray::Array1;
use voirs_spatial::{
    AmbisonicsDecoder, AmbisonicsEncoder, ChannelOrdering, NormalizationScheme, Position3D,
    SpeakerConfiguration, SphericalCoordinate,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Ambisonics Encoding/Decoding Example ===\n");

    // Create second-order ambisonics encoder (order = 2)
    let encoder = AmbisonicsEncoder::new(
        2, // Second order = 9 channels
        NormalizationScheme::N3D,
        ChannelOrdering::ACN,
    );

    println!("✓ Created second-order ambisonics encoder");
    println!("  Order: 2 (9 channels)");
    println!("  Normalization: N3D");
    println!("  Ordering: ACN\n");

    // Generate test audio (mono)
    let audio_vec: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.01).sin() * 0.5).collect();
    let audio_mono = Array1::from_vec(audio_vec);

    println!("✓ Generated test audio: {} samples\n", audio_mono.len());

    // Define source position in spherical coordinates
    let source_spherical = SphericalCoordinate {
        azimuth: 45.0f32.to_radians(),   // 45 degrees right
        elevation: 30.0f32.to_radians(), // 30 degrees up
        distance: 2.0,                   // 2 meters away
    };

    println!("Source Position (spherical):");
    println!("  Azimuth: {:.1}°", source_spherical.azimuth.to_degrees());
    println!(
        "  Elevation: {:.1}°",
        source_spherical.elevation.to_degrees()
    );
    println!("  Distance: {:.1}m\n", source_spherical.distance);

    // Convert to Cartesian for API
    let source_position = source_spherical.to_cartesian();
    println!("Source Position (cartesian):");
    println!(
        "  X: {:.2}m, Y: {:.2}m, Z: {:.2}m\n",
        source_position.x, source_position.y, source_position.z
    );

    // Encode audio to ambisonics
    println!("Encoding audio to ambisonics format...");
    let ambisonics_encoded = encoder.encode_mono(&audio_mono, &source_position)?;

    println!(
        "✓ Audio encoded to {} ambisonics channels",
        ambisonics_encoded.shape()[0]
    );
    for i in 0..ambisonics_encoded.shape()[0] {
        println!("  Channel {}: {} samples", i, ambisonics_encoded.shape()[1]);
    }
    println!();

    // Decode to different speaker configurations
    // Use helper method for_speaker_config which handles speaker positions automatically
    println!("Decoding to different speaker configurations:\n");

    let configs = vec![
        ("Stereo", SpeakerConfiguration::Stereo),
        ("Quadraphonic", SpeakerConfiguration::Quadraphonic),
        ("5.1 Surround", SpeakerConfiguration::FiveDotOne),
        ("7.1 Surround", SpeakerConfiguration::SevenDotOne),
    ];

    for (name, config) in configs {
        let decoder = AmbisonicsDecoder::for_speaker_config(2, config)?;
        let decoded = decoder.decode(&ambisonics_encoded)?;

        println!("✓ {} Configuration:", name);
        println!("  Output channels: {}", decoded.shape()[0]);

        // Calculate energy per channel
        for ch in 0..decoded.shape()[0] {
            let channel_data = decoded.row(ch);
            let energy: f32 = channel_data.iter().map(|&x| x * x).sum();
            let rms = (energy / channel_data.len() as f32).sqrt();
            println!("  Channel {} RMS: {:.3}", ch, rms);
        }
        println!();
    }

    // Demonstrate rotating the source
    println!("Rotating sound source around listener:\n");

    // Create stereo decoder once for all rotations
    let stereo_decoder = AmbisonicsDecoder::for_speaker_config(2, SpeakerConfiguration::Stereo)?;

    for angle_deg in (0..360).step_by(45) {
        let angle_rad = (angle_deg as f32).to_radians();
        let rotated_spherical = SphericalCoordinate {
            azimuth: angle_rad,
            elevation: 0.0,
            distance: 2.0,
        };
        let rotated_pos = rotated_spherical.to_cartesian();

        let encoded_rotated = encoder.encode_mono(&audio_mono, &rotated_pos)?;

        // Decode to stereo
        let stereo = stereo_decoder.decode(&encoded_rotated)?;

        // Calculate left/right balance
        let left_channel = stereo.row(0);
        let right_channel = stereo.row(1);
        let left_energy: f32 = left_channel.iter().map(|&x| x * x).sum();
        let right_energy: f32 = right_channel.iter().map(|&x| x * x).sum();

        println!(
            "  {:3}° - L/R balance: {:+.1}dB",
            angle_deg,
            10.0 * (right_energy / left_energy.max(0.001)).log10()
        );
    }

    println!("\n✅ Example completed successfully!");
    println!("\nKey takeaways:");
    println!("  - Ambisonics is format-independent");
    println!("  - Can decode to any speaker configuration");
    println!("  - Higher orders = better spatial resolution");
    println!("  - Rotation in ambisonics domain is simple");

    Ok(())
}
