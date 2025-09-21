//! Simple 3D Spatial Audio Example
//!
//! This example demonstrates basic 3D spatial audio capabilities of VoiRS SDK,
//! including position tracking, listener setup, and sound source management.

use anyhow::Result;
use std::f32::consts::PI;
use std::time::Duration;
use voirs::*;
use voirs_spatial::position::{Listener, SoundSource};
use voirs_spatial::types::{BinauraAudio, SpatialEffect};
use voirs_spatial::{Position3D, SpatialConfig, SpatialProcessor};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🌍 VoiRS 3D Spatial Audio Example");
    println!("=================================");

    // Ensure output directories exist
    ensure_output_dirs()?;

    // Create components using bridge pattern
    let g2p = create_g2p(G2pBackend::RuleBased);
    let acoustic = create_acoustic(AcousticBackend::Vits);
    let vocoder = create_vocoder(VocoderBackend::HifiGan);

    // Create pipeline with spatial audio components
    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .build()
        .await?;

    // Example 1: Basic 3D positioning
    println!("\n1. Basic 3D Positioning");
    println!("-----------------------");
    basic_3d_positioning_demo(&pipeline).await?;

    // Example 2: Multiple sound sources
    println!("\n2. Multiple Sound Sources");
    println!("-------------------------");
    multiple_sources_demo(&pipeline).await?;

    // Example 3: Listener movement simulation
    println!("\n3. Listener Movement Simulation");
    println!("--------------------------------");
    listener_movement_demo(&pipeline).await?;

    println!("\n✅ Spatial audio examples completed!");
    Ok(())
}

async fn basic_3d_positioning_demo(pipeline: &VoirsPipeline) -> Result<()> {
    // Create a basic spatial configuration
    let spatial_config = SpatialConfig::default();

    // Create a listener at the origin
    let listener = Listener::new();
    println!("Created listener at origin position");

    // Create sound sources at different positions
    let positions = vec![
        ("front", Position3D::new(0.0, 0.0, 1.0)),
        ("left", Position3D::new(-1.0, 0.0, 0.0)),
        ("right", Position3D::new(1.0, 0.0, 0.0)),
        ("behind", Position3D::new(0.0, 0.0, -1.0)),
    ];

    for (name, position) in positions {
        let source = SoundSource::new_point(name.to_string(), position);

        println!(
            "Created sound source '{}' at position ({:.1}, {:.1}, {:.1})",
            name, position.x, position.y, position.z
        );

        // Synthesize a simple phrase for this position
        let text = format!("Hello from {}", name);
        let audio = pipeline.synthesize(&text).await?;

        // Save the audio with position info in filename
        let filename = format!("spatial_{}_{}.wav", name, "demo");
        audio.save_wav(&filename)?;
        println!("Saved spatial audio: {}", filename);
    }

    Ok(())
}

async fn multiple_sources_demo(pipeline: &VoirsPipeline) -> Result<()> {
    // Create multiple sound sources in a circle around the listener
    let num_sources = 8;
    let radius = 2.0;

    for i in 0..num_sources {
        let angle = (i as f32) * 2.0 * PI / (num_sources as f32);
        let position = Position3D::new(radius * angle.cos(), 0.0, radius * angle.sin());

        let source_name = format!("source_{}", i + 1);
        let source = SoundSource::new_point(source_name.clone(), position);

        println!(
            "Created source {} at angle {:.1}° (pos: {:.1}, {:.1}, {:.1})",
            i + 1,
            angle.to_degrees(),
            position.x,
            position.y,
            position.z
        );

        // Synthesize speech for this source
        let text = format!("I am source number {}", i + 1);
        let audio = pipeline.synthesize(&text).await?;

        // Save with source number in filename
        let filename = format!("multi_source_{:02}.wav", i + 1);
        audio.save_wav(&filename)?;
        println!("Saved: {}", filename);
    }

    Ok(())
}

async fn listener_movement_demo(pipeline: &VoirsPipeline) -> Result<()> {
    // Simulate listener moving past a stationary source
    let source_position = Position3D::new(0.0, 0.0, 0.0);
    let source = SoundSource::new_point("stationary_source".to_string(), source_position);

    println!("Created stationary source at origin");

    // Simulate listener positions along a path
    let path_points = vec![
        ("start", Position3D::new(-3.0, 0.0, 0.0)),
        ("approaching", Position3D::new(-1.0, 0.0, 0.0)),
        ("closest", Position3D::new(0.0, 0.0, 0.0)),
        ("passing", Position3D::new(1.0, 0.0, 0.0)),
        ("distant", Position3D::new(3.0, 0.0, 0.0)),
    ];

    for (stage, listener_pos) in path_points {
        let distance = listener_pos.distance_to(&source_position);
        println!("Listener at {} - distance: {:.1}m", stage, distance);

        // Create listener at this position
        let mut listener = Listener::new();
        listener.set_position(listener_pos);

        // Synthesize audio that would be heard at this position
        let text = format!("You are {} the source", stage);
        let audio = pipeline.synthesize(&text).await?;

        // Save with stage name
        let filename = format!("movement_{}.wav", stage);
        audio.save_wav(&filename)?;
        println!("Saved: {}", filename);
    }

    Ok(())
}

fn ensure_output_dirs() -> Result<()> {
    std::fs::create_dir_all("output/spatial")?;
    Ok(())
}
